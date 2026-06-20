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

/// Render the painted map: base → SDXL img2img + Canny ControlNet → (re-composite
/// labels) → `out`. Requires a GPU-capable build; the model downloads on first use.
pub async fn render_sd(
    spec: &MapSpec,
    seed: u64,
    style: Style,
    opts: &SdOptions,
    device: Device,
    out: &Path,
) -> Result<()> {
    let geo = Geometry::compute(spec, seed)?;
    let (w, h) = (round8(geo.hf.width), round8(geo.hf.height));

    // 1) Conditioning base (img2img init + Canny source) → a scratch PNG.
    let base = render::paint_base_map(&geo, style);
    let scratch = scratch_dir(seed)?;
    let cond_path = scratch.join("conditioning.png");
    write_png(&base, &cond_path)?;

    // 2) SDXL img2img + Canny ControlNet over the base. Reuses the img2img
    //    pipeline wholesale; the Canny annotator edges the conditioning image.
    let (prompt, negative) = cartography_prompt(spec);
    let req = crate::pipelines::img2img::Request {
        prompt,
        negative,
        model: opts.model.clone(),
        device,
        loras: opts.lora_specs()?,
        lora_scale: opts.lora_scale,
        input: cond_path.clone(),
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
        out_dir: scratch.clone(),
        controls: vec![ControlSpec {
            kind: ControlKind::Canny,
            image: None,
            from: Some(cond_path.clone()),
            video: None,
            strength: opts.control_strength,
            start: 0.0,
            end: 1.0,
        }],
    };
    crate::pipelines::img2img::run(req).await.context("SDXL img2img+ControlNet map render")?;

    // 3) Collect the single painted PNG the pipeline wrote.
    let painted_path = newest_png(&scratch, &cond_path)?
        .context("img2img produced no output image")?;
    let mut painted = image::open(&painted_path)
        .with_context(|| format!("reading SD output {}", painted_path.display()))?
        .to_rgb8();

    // 4) Restore the crisp linework (coast/rivers/roads — washed out by the paint)
    //    then re-composite labels + furniture, so the painted map stays a usable
    //    map (unless --map-sd-raw asks for the bare painting).
    if !opts.raw {
        // Geometry is canvas-sized; the SD output is round8(canvas) — equal here,
        // but guard so the overlay only runs when the dimensions match.
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
}
