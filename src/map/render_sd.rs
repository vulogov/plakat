//! MAP-6 — the **tiled-SD painted render**. The one GPU step on the map track:
//! feed the styled base map (the MAP-2 geometry, no labels) through an **SDXL
//! img2img + Canny ControlNet** pass so the map looks hand-painted, then
//! re-composite the 1.5.0 labels + furniture over the result so it stays legible.
//!
//! Memory-wall discipline (the 1.1.0 lesson): the *conditioning* (the base image
//! together with its Canny edges) is a deterministic artifact — byte-stable,
//! corpus-proven, no GPU. Only the SD denoise is non-deterministic / memory-bound,
//! and it's decoupled here behind `--map-render-sd` so the geometry pipeline never
//! needs a GPU. This phase ships the **1×1 on-box** path; tiled multi-tile follows.

use anyhow::{Context, Result};
use candle_core::Device;
use image::RgbImage;
use std::path::{Path, PathBuf};

use super::render::{self, Geometry, Style};
use super::spec::MapSpec;
use crate::pipelines::controlnet::{ControlKind, ControlSpec};
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, LoadRequest};
use crate::pipelines::scheduler::SchedulerKind;
use std::str::FromStr;

/// The default cartography LoRA — SDXL fantasy-map style (HF `Muapi/fantasy-map`).
/// Painted-map look on top of the base SDXL checkpoint. SDXL-only, so it auto-
/// applies only to SDXL-family models (any other model renders LoRA-free).
pub const DEFAULT_MAP_LORA: &str = "Muapi/fantasy-map";

/// Is this an SDXL-family model? (The fantasy-map LoRA is SDXL — applying it to a
/// SD1.5 / Flux / SD3.5 / PixArt / Cascade backbone would mismatch tensor shapes.)
pub fn is_sdxl_family(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("sdxl") || m.contains("xl") || m.contains("pony") || m.contains("illustrious")
}

/// The auto LoRA set for a model when the user didn't specify any: the fantasy-map
/// style for SDXL-family backbones, nothing otherwise (so every model still works).
pub fn default_loras_for_model(model: &str) -> Vec<String> {
    if is_sdxl_family(model) {
        vec![DEFAULT_MAP_LORA.into()]
    } else {
        Vec::new()
    }
}

/// Knobs for the SD pass (sensible cartographic defaults).
#[derive(Debug, Clone)]
pub struct SdOptions {
    pub model: String,
    /// LoRA specs (CLI grammar — HF `org/name[:scale]`, civitai:, or a local
    /// path). Empty → none. Defaults to the fantasy-map style LoRA.
    pub loras: Vec<String>,
    pub lora_scale: f32,
    /// img2img strength — how far the paint moves from the base. Modest so the
    /// geometry (coast, rivers, roads) survives.
    pub strength: f32,
    pub steps: usize,
    pub guidance: f64,
    /// Canny ControlNet strength (structure lock).
    pub control_strength: f32,
    /// Skip the label/furniture re-composite (raw painted output).
    pub raw: bool,
    /// Tile size in px for the multi-tile path. A canvas wider/taller than this
    /// paints in overlapping tiles (each a full img2img+Canny pass that fits
    /// memory), feather-blended back — the memory-safe path for large maps.
    pub tile_size: u32,
    /// Tile origin stride in px (smaller = more overlap = smoother seams).
    pub tile_stride: u32,
}

impl Default for SdOptions {
    fn default() -> Self {
        SdOptions {
            model: "sdxl".into(),
            loras: default_loras_for_model("sdxl"),
            lora_scale: 0.9,
            strength: 0.55,
            steps: 28,
            guidance: 6.5,
            control_strength: 0.9,
            raw: false,
            tile_size: 1024,
            tile_stride: 768,
        }
    }
}

impl SdOptions {
    /// Parse the LoRA CLI strings into specs (HF/civitai/local).
    fn lora_specs(&self) -> Result<Vec<LoraSpec>> {
        self.loras
            .iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| LoraSpec::from_str(s).with_context(|| format!("parsing --map-sd-lora {s:?}")))
            .collect()
    }
}

/// Round down to a multiple of 8 (SD latent constraint), min 8.
fn round8(v: u32) -> u32 {
    (v / 8 * 8).max(8)
}

/// A deterministic cartography prompt + negative, derived from the spec's words
/// (climate / era / dominant elevation / biomes present). No randomness.
pub fn cartography_prompt(spec: &MapSpec) -> (String, String) {
    // Leads with "fantasy map" — the trigger phrase for the Muapi/fantasy-map
    // SDXL style LoRA (a no-op for other models, just descriptive there).
    let mut pos = String::from(
        "fantasy map, a hand-painted antique world map, top-down aerial cartography, \
         aged parchment, ink coastline and linework, muted watercolor wash, \
         hill-shaded mountains, forests, winding rivers, fine detail",
    );
    if let Some(c) = spec.climate.as_deref().filter(|s| !s.is_empty()) {
        pos.push_str(&format!(", {c} climate"));
    }
    if let Some(e) = spec.era.as_deref().filter(|s| !s.is_empty()) {
        pos.push_str(&format!(", {e} setting"));
    }
    let elev = spec.terrain.dominant_elevation.trim();
    if !elev.is_empty() {
        pos.push_str(&format!(", {elev} terrain"));
    }
    // A few biome words from the spec's regions, in order, deduped.
    let mut biomes: Vec<&str> = Vec::new();
    for r in &spec.regions {
        let b = r.biome.trim();
        if !b.is_empty() && !biomes.contains(&b) {
            biomes.push(b);
        }
    }
    if !biomes.is_empty() {
        pos.push_str(", ");
        pos.push_str(&biomes.join(", ").replace('_', " "));
    }
    let neg = String::from(
        "photo, photograph, realistic photo, 3d render, satellite imagery, \
         modern, text, words, labels, watermark, signature, blurry, lowres, \
         people, characters, frame, border, ui",
    );
    (pos, neg)
}

/// The SD conditioning base — the styled base map (terrain/coast/rivers/roads),
/// **no labels**. Deterministic; the corpus byte-checks this.
pub fn build_conditioning(spec: &MapSpec, seed: u64, style: Style) -> Result<RgbImage> {
    let geo = Geometry::compute(spec, seed)?;
    Ok(render::paint_base_map(&geo, style))
}

/// Write the conditioning base PNG (deterministic — corpus artifact).
pub fn save_conditioning(spec: &MapSpec, seed: u64, style: Style, path: &Path) -> Result<()> {
    let img = build_conditioning(spec, seed, style)?;
    write_png(&img, path)
}

/// Render the painted map: base → SDXL img2img + Canny ControlNet → restore
/// linework + labels → `out`. A canvas larger than `tile_size` paints in
/// overlapping tiles (memory-safe). Requires a GPU build; model downloads on use.
pub async fn render_sd(
    spec: &MapSpec,
    seed: u64,
    style: Style,
    opts: &SdOptions,
    device: Device,
    out: &Path,
) -> Result<()> {
    let geo = Geometry::compute(spec, seed)?;
    let base = render::paint_base_map(&geo, style); // the conditioning (no labels)
    let (w, h) = base.dimensions();
    let scratch = scratch_dir(seed)?;

    // Load the SD backbone once (model + LoRA) — reused across every tile.
    let pipeline = portrait::Pipeline::load(LoadRequest {
        model: opts.model.clone(),
        device: device.clone(),
        loras: opts.lora_specs()?,
        lora_scale: opts.lora_scale,
        identity: None,
        shared_clip_h: None,
    })
    .await
    .context("loading SD pipeline for map render")?;

    let tile = round8(opts.tile_size);
    let mut painted = if w > tile || h > tile {
        let cols = tile_starts(w, tile.min(w), round8(opts.tile_stride)).len();
        let rows = tile_starts(h, tile.min(h), round8(opts.tile_stride)).len();
        tracing::info!(target: "plakat", "map: tiled paint {cols}x{rows} tiles ({tile}px) over {w}x{h}");
        paint_tiled(&pipeline, spec, opts, seed, &base, &device, &scratch).await?
    } else {
        paint_one(&pipeline, spec, opts, seed, &base, &scratch).await?
    };

    // Restore the crisp linework (coast/rivers/roads the paint washes out), then
    // re-composite labels + furniture, unless --map-sd-raw wants the bare painting.
    if !opts.raw {
        if painted.dimensions() == (geo.hf.width, geo.hf.height) {
            render::apply_linework(&mut painted, &geo, style);
            render::apply_labels_and_furniture(&mut painted, spec, &geo, style);
        } else {
            tracing::warn!(
                target: "plakat",
                "map: SD output {:?} != geometry {:?}; skipping linework+label overlay",
                painted.dimensions(), (geo.hf.width, geo.hf.height)
            );
        }
    }
    write_png(&painted, out)?;
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}

/// One img2img + Canny pass over `src` using the already-loaded `pipeline`;
/// returns the painted image (same size as `src`, dims rounded to /8).
async fn paint_one(
    pipeline: &portrait::Pipeline,
    spec: &MapSpec,
    opts: &SdOptions,
    seed: u64,
    src: &RgbImage,
    scratch: &Path,
) -> Result<RgbImage> {
    let (w, h) = (round8(src.width()), round8(src.height()));
    let in_path = scratch.join(format!("tile-in-{seed}.png"));
    write_png(src, &in_path)?;
    let out_dir = scratch.join(format!("tile-out-{seed}"));
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let (prompt, negative) = cartography_prompt(spec);
    let req = crate::pipelines::img2img::Request {
        prompt,
        negative,
        model: opts.model.clone(),
        device: pipeline.core().device.clone(),
        loras: Vec::new(), // already merged into the loaded pipeline
        lora_scale: opts.lora_scale,
        input: in_path.clone(),
        mask: None,
        mask_feather: 0,
        mask_invert: false,
        width: w,
        height: h,
        count: 1,
        steps: opts.steps,
        guidance: opts.guidance,
        scheduler: SchedulerKind::Default,
        strength: opts.strength,
        seed: Some(seed),
        out_dir: out_dir.clone(),
        controls: vec![ControlSpec {
            kind: ControlKind::Canny,
            image: None,
            from: Some(in_path.clone()),
            video: None,
            strength: opts.control_strength,
            start: 0.0,
            end: 1.0,
        }],
    };
    crate::pipelines::img2img::run_with_pipeline(pipeline, &req)
        .await
        .context("SDXL img2img+Canny map tile")?;
    let painted = newest_png(&out_dir, &in_path)?.context("img2img produced no output image")?;
    Ok(image::open(&painted).with_context(|| format!("reading SD tile {}", painted.display()))?.to_rgb8())
}

/// Paint a large canvas in overlapping tiles, each a full img2img+Canny pass
/// (memory-safe), feather-blended back. The conditioning base supplies the global
/// structure, so independent per-tile denoise stays coherent.
async fn paint_tiled(
    pipeline: &portrait::Pipeline,
    spec: &MapSpec,
    opts: &SdOptions,
    seed: u64,
    base: &RgbImage,
    _device: &Device,
    scratch: &Path,
) -> Result<RgbImage> {
    let (w, h) = (base.width(), base.height());
    let tile = round8(opts.tile_size);
    let stride = round8(opts.tile_stride).max(8);
    let (tw, th) = (tile.min(w), tile.min(h));
    let xs = tile_starts(w, tw, stride);
    let ys = tile_starts(h, th, stride);

    // f32 accumulators (RGB) + per-pixel weight, for Hann-feathered blending.
    let n = (w * h) as usize;
    let mut acc = vec![0f32; n * 3];
    let mut wsum = vec![0f32; n];
    let win = hann2d(tw, th);

    for &ty in &ys {
        for &tx in &xs {
            let crop = image::imageops::crop_imm(base, tx, ty, tw, th).to_image();
            let painted = paint_one(pipeline, spec, opts, seed, &crop, scratch).await?;
            // paint_one rounds dims to /8; tw/th are already /8 so they match.
            for j in 0..th {
                for i in 0..tw {
                    let wv = win[(j * tw + i) as usize];
                    let p = painted.get_pixel(i, j).0;
                    let gi = ((ty + j) * w + (tx + i)) as usize;
                    acc[gi * 3] += p[0] as f32 * wv;
                    acc[gi * 3 + 1] += p[1] as f32 * wv;
                    acc[gi * 3 + 2] += p[2] as f32 * wv;
                    wsum[gi] += wv;
                }
            }
        }
    }

    let mut out = RgbImage::new(w, h);
    for gi in 0..n {
        let wv = wsum[gi].max(1e-6);
        let px = [
            (acc[gi * 3] / wv).round().clamp(0.0, 255.0) as u8,
            (acc[gi * 3 + 1] / wv).round().clamp(0.0, 255.0) as u8,
            (acc[gi * 3 + 2] / wv).round().clamp(0.0, 255.0) as u8,
        ];
        out.put_pixel((gi as u32) % w, (gi as u32) / w, image::Rgb(px));
    }
    Ok(out)
}

/// Tile origin starts along one axis: step by `stride`, snap the final tile to the
/// edge so the whole length is covered (mirrors `pipelines::tiled`).
fn tile_starts(total: u32, tile: u32, stride: u32) -> Vec<u32> {
    if total <= tile {
        return vec![0];
    }
    let mut v = Vec::new();
    let mut p = 0u32;
    loop {
        v.push(p);
        if p + tile >= total {
            break;
        }
        p += stride;
        if p + tile > total {
            v.push(total - tile);
            break;
        }
    }
    v
}

/// A 2D raised-cosine (Hann) window `tw×th`, peaking at the centre, ~0 at edges —
/// the per-tile blend weight that feathers seams. Floored so edges still register.
fn hann2d(tw: u32, th: u32) -> Vec<f32> {
    let hann = |k: u32, n: u32| -> f32 {
        if n <= 1 {
            return 1.0;
        }
        let v = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * k as f32 / (n as f32 - 1.0)).cos());
        v.max(1e-3)
    };
    let (cx, cy): (Vec<f32>, Vec<f32>) =
        ((0..tw).map(|i| hann(i, tw)).collect(), (0..th).map(|j| hann(j, th)).collect());
    let mut win = vec![0f32; (tw * th) as usize];
    for j in 0..th {
        for i in 0..tw {
            win[(j * tw + i) as usize] = cx[i as usize] * cy[j as usize];
        }
    }
    win
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn write_png(img: &RgbImage, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(path).map_err(|e| anyhow::anyhow!("writing {}: {e}", path.display()))
}

/// A per-run scratch dir under the system temp (seed keeps concurrent runs apart).
fn scratch_dir(seed: u64) -> Result<PathBuf> {
    let d = std::env::temp_dir().join(format!("plakat-map-sd-{seed}"));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// The most-recently-modified `.png` in `dir` that isn't `exclude`.
fn newest_png(dir: &Path, exclude: &Path) -> Result<Option<PathBuf>> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().map(|x| x.eq_ignore_ascii_case("png")).unwrap_or(false) && p != exclude {
            let m = p.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            if best.as_ref().is_none_or(|(bm, _)| m >= *bm) {
                best = Some((m, p));
            }
        }
    }
    Ok(best.map(|(_, p)| p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn island() -> MapSpec {
        serde_json::from_str(include_str!("../../corpus/map/island.spec.json")).unwrap()
    }

    #[test]
    fn conditioning_is_deterministic_and_label_free() {
        let a = build_conditioning(&island(), 42, Style::default()).unwrap();
        let b = build_conditioning(&island(), 42, Style::default()).unwrap();
        assert!(a.as_raw() == b.as_raw(), "conditioning must be byte-stable");
        // The conditioning is the same size as the full render but carries no ink
        // frame at the very corner (labels/furniture are absent) — sanity that it's
        // the base, not the labelled render.
        let full = render::render(&island(), 42, Style::default()).unwrap();
        assert_eq!(a.dimensions(), full.dimensions());
        assert!(a.as_raw() != full.as_raw(), "conditioning differs from the labelled render");
    }

    #[test]
    fn prompt_includes_spec_words() {
        let (pos, neg) = cartography_prompt(&island());
        assert!(pos.contains("temperate maritime"), "climate in prompt: {pos}");
        assert!(pos.contains("late medieval"), "era in prompt");
        assert!(pos.contains("mountainous"), "dominant elevation in prompt");
        assert!(pos.contains("volcanic"), "a region biome in prompt");
        assert!(neg.contains("text"), "negative suppresses baked-in text");
        // Deterministic.
        assert_eq!(cartography_prompt(&island()).0, pos);
    }

    #[test]
    fn round8_floors_to_multiple_of_eight() {
        assert_eq!(round8(512), 512);
        assert_eq!(round8(515), 512);
        assert_eq!(round8(7), 8);
    }

    #[test]
    fn tile_starts_cover_the_axis_with_edge_snap() {
        // Canvas fits in one tile → single tile at 0.
        assert_eq!(tile_starts(512, 1024, 768), vec![0]);
        // 512 canvas, 384 tile, 256 stride → [0, 128] (last snapped to 512-384).
        assert_eq!(tile_starts(512, 384, 256), vec![0, 128]);
        // The union of [start, start+tile) must cover [0, total).
        let (total, tile, stride) = (2048u32, 1024u32, 768u32);
        let starts = tile_starts(total, tile, stride);
        assert_eq!(*starts.first().unwrap(), 0);
        assert_eq!(starts.last().unwrap() + tile, total, "last tile reaches the edge");
        // No gap between consecutive tiles.
        for pair in starts.windows(2) {
            assert!(pair[1] <= pair[0] + tile, "gap between tiles {pair:?}");
        }
    }

    #[test]
    fn hann2d_peaks_at_centre_and_floors_at_edges() {
        let (tw, th) = (16u32, 16u32);
        let win = hann2d(tw, th);
        let centre = win[((th / 2) * tw + tw / 2) as usize];
        let corner = win[0];
        assert!(centre > corner, "window peaks in the middle");
        assert!(corner > 0.0, "edges floored (product of per-axis floors) so they still register");
        assert!(centre > 0.5, "centre weight is substantial");
        assert_eq!(win.len(), (tw * th) as usize);
    }
}
