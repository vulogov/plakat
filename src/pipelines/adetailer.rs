//! v0.16 phase 6: ADetailer-style face refinement.
//!
//! After-Detailer (a.k.a. "ADetailer" from the Auto1111 extension)
//! is a post-processing pass that fixes the lo-fi faces SD/SDXL
//! often produce at non-face working resolutions. The recipe:
//!
//! 1. **Detect** each face in the generated image with SCRFD.
//! 2. **Crop** an expanded bounding box (default +25% on each side,
//!    clamped to the image) — gives the inpaint pass some context
//!    around the face.
//! 3. **img2img** the crop at higher detail using the same SD model
//!    + LoRA stack the main generation used. Strength = 0.4 by
//!    default (gentle — preserves identity / colour, only crisps
//!    detail).
//! 4. **Resize** the result back to the expanded-bbox dimensions.
//! 5. **Feather-composite** onto the original with a tapered alpha
//!    that fades from `1.0` at the bbox centre to `0.0` at its
//!    edge — hides the seam where the refined crop meets the
//!    rest of the image.
//!
//! SD-family only (portrait::Pipeline rejects Flux at load time).
//! No-op when SCRFD weights aren't configured — the caller has
//! already bailed at the CLI level before reaching this module.

use anyhow::{Context, Result};
use candle_core::Device;
use image::{ImageBuffer, Rgb};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::pipelines::img2img;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, LoadRequest};
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::scrfd::{Face, SCRFDConfig, SCRFDDetector};
use crate::pipelines::sd_core::SdCore;

/// Configuration for one batch of ADetailer refinement passes.
pub struct Config {
    /// SD model alias / repo. Must match the one the input files were
    /// generated with — ADetailer reuses the same SdCore when one is
    /// supplied via `shared_core`. Diverging models silently produce
    /// stylistically inconsistent refinements.
    pub model: String,
    /// LoRA stack to apply during the face img2img pass. Typically a
    /// face-specific LoRA (e.g. "perfect-eyes-xl") or empty.
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    /// Prompt used for the face img2img pass. Defaults to a generic
    /// "detailed face, sharp focus, high quality" but the caller can
    /// override (e.g. via `--adetailer-prompt`).
    pub prompt: String,
    /// Negative prompt for the face pass. Defaults to a low-quality
    /// blocker if empty.
    pub negative: String,
    /// img2img strength on each face crop. `0.4` is a good default —
    /// preserves identity / colour, only crisps detail. Higher
    /// strengths can change the face significantly.
    pub strength: f32,
    /// Working resolution of the face img2img pass (square). `512` is
    /// SD 1.5 native; `1024` matches SDXL native. Snapped to multiples
    /// of 8 by the img2img pipeline.
    pub working_size: u32,
    /// Step count of the face pass.
    pub steps: usize,
    /// CFG guidance for the face pass.
    pub guidance: f64,
    /// Scheduler for the face pass.
    pub scheduler: SchedulerKind,
    /// SCRFD score threshold — faces below this score are skipped.
    /// Default `0.5` matches InsightFace.
    pub confidence: f32,
    /// Bbox expansion factor — `0.25` adds 25% on each side (50% total
    /// dim growth). Trades context (more = better blending, less
    /// resolution per face) against detail (less = sharper, harder seam).
    pub padding: f32,
    /// Feather fraction — `0.25` means the outer 25% of the bbox fades
    /// from 1.0 → 0.0. Larger feather = softer seam.
    pub feather: f32,
    pub device: Device,
}

impl Config {
    /// Defaults that match Auto1111's ADetailer extension at the
    /// "Face — yolov8n" model with strength 0.4. Caller fills `model`,
    /// `device`, etc.
    pub fn defaults() -> Self {
        Self {
            model: String::new(),
            loras: Vec::new(),
            lora_scale: 1.0,
            prompt: "detailed face, sharp focus, high quality".to_string(),
            negative: "lowres, bad anatomy, blurry, deformed".to_string(),
            strength: 0.4,
            working_size: 512,
            steps: 28,
            guidance: 7.5,
            scheduler: SchedulerKind::Default,
            confidence: 0.5,
            padding: 0.25,
            feather: 0.25,
            device: Device::Cpu,
        }
    }
}

/// v0.16 phase 6: run ADetailer over `files` in place. Each file's
/// detected faces (above `cfg.confidence`) get a fresh img2img pass
/// on the expanded crop, then feather-composited back. SCRFD weights
/// are auto-resolved via the existing env-var path
/// (`PLAKAT_SCRFD_WEIGHTS` or `PLAKAT_SCRFD_HF`).
///
/// `shared_core` lets the caller reuse the SD backbone t2i just
/// loaded — same pattern as artefact-blend. Pass `None` to make
/// ADetailer load its own.
///
/// Returns the total number of faces refined (0 = no detections,
/// not an error).
pub async fn refine_files(
    cfg: &Config,
    files: &[PathBuf],
    shared_core: Option<Arc<SdCore>>,
) -> Result<usize> {
    if files.is_empty() {
        return Ok(0);
    }

    // SCRFD weights: bail loud (not warn) — the user explicitly
    // asked for --adetailer; an unset SCRFD config is a misconfig,
    // not an opt-out.
    let scrfd_path =
        crate::pipelines::scrfd::resolve_scrfd_weights()
            .await
            .context("resolving SCRFD weights for ADetailer")?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "ADetailer requested but neither PLAKAT_SCRFD_WEIGHTS nor \
                     PLAKAT_SCRFD_HF is set. ADetailer needs a face detector; \
                     point one of those env vars at an SCRFD safetensors \
                     (typically the same one --portrait uses)."
                )
            })?;

    let detector = SCRFDDetector::load(
        &scrfd_path,
        SCRFDConfig::default(),
        &cfg.device,
        candle_core::DType::F32,
    )
    .context("loading SCRFD detector for ADetailer")?;

    let pipeline = match shared_core {
        Some(core) => portrait::Pipeline::from_core(core),
        None => portrait::Pipeline::load(LoadRequest {
            model: cfg.model.clone(),
            device: cfg.device.clone(),
            loras: cfg.loras.clone(),
            lora_scale: cfg.lora_scale,
            identity: None,
            shared_clip_h: None,
        })
        .await
        .context("loading SD pipeline for ADetailer")?,
    };

    let mut total_refined = 0usize;
    for path in files {
        let n = refine_one(cfg, path, &detector, &pipeline).await
            .with_context(|| format!("ADetailer pass on {}", path.display()))?;
        total_refined += n;
    }
    Ok(total_refined)
}

async fn refine_one(
    cfg: &Config,
    path: &Path,
    detector: &SCRFDDetector,
    pipeline: &portrait::Pipeline,
) -> Result<usize> {
    // 1. Detect.
    let faces = detector.detect(path).context("SCRFD detect")?;
    let keep: Vec<&Face> = faces.iter().filter(|f| f.score >= cfg.confidence).collect();
    if keep.is_empty() {
        tracing::debug!(
            target: "plakat",
            "ADetailer: no faces ≥ confidence {:.2} in {}",
            cfg.confidence,
            path.display()
        );
        return Ok(0);
    }

    // 2. Load the original into an editable RGB buffer.
    let original = image::open(path)
        .with_context(|| format!("opening {} for ADetailer", path.display()))?
        .to_rgb8();
    let (img_w, img_h) = original.dimensions();
    let mut composite = original.clone();

    // 3. Crop → img2img → composite, per face.
    let tmpdir = tempfile::Builder::new()
        .prefix("plakat-adetailer-")
        .tempdir()
        .context("creating ADetailer tempdir")?;

    for (i, face) in keep.iter().enumerate() {
        let bbox = expand_bbox(&face.bbox, cfg.padding, img_w, img_h);
        // Skip degenerate crops (face entirely off-screen or too thin
        // to make 8×8 latents).
        if bbox.w < 8 || bbox.h < 8 {
            tracing::warn!(
                target: "plakat",
                "ADetailer: face {} in {} has degenerate expanded bbox \
                 ({}x{}) — skipping.",
                i, path.display(), bbox.w, bbox.h
            );
            continue;
        }

        let crop_buf = crop_rgb(&original, bbox.x, bbox.y, bbox.w, bbox.h)?;
        let crop_path = tmpdir.path().join(format!("face_{i}.png"));
        crop_buf
            .save(&crop_path)
            .with_context(|| format!("writing face crop {}", crop_path.display()))?;

        // 4. img2img the crop at the working size.
        let working = round_down_8(cfg.working_size);
        let req = img2img::Request {
            prompt: cfg.prompt.clone(),
            negative: cfg.negative.clone(),
            model: cfg.model.clone(),
            device: cfg.device.clone(),
            loras: cfg.loras.clone(),
            lora_scale: cfg.lora_scale,
            input: crop_path.clone(),
            mask: None,
            mask_feather: 0,
            mask_invert: false,
            width: working,
            height: working,
            count: 1,
            steps: cfg.steps,
            guidance: cfg.guidance,
            scheduler: cfg.scheduler,
            strength: cfg.strength,
            seed: None,
            out_dir: tmpdir.path().to_path_buf(),
            controls: Vec::new(),
        };
        img2img::run_with_pipeline(pipeline, &req)
            .await
            .with_context(|| format!("img2img on face crop {i}"))?;

        // img2img writes `plakat-img2img-<seed>.png` into the out_dir.
        // Pick whatever it produced (one file per call).
        let refined_path = find_output(tmpdir.path(), "plakat-img2img-")?;
        let refined_img = image::open(&refined_path)
            .with_context(|| format!("opening refined crop {}", refined_path.display()))?
            .to_rgb8();
        // Resize back to the expanded-bbox dims.
        let refined_resized = image::imageops::resize(
            &refined_img,
            bbox.w,
            bbox.h,
            image::imageops::FilterType::Lanczos3,
        );
        // Clean up so the next face's `find_output` is unambiguous.
        let _ = std::fs::remove_file(&refined_path);

        // 5. Feather-composite onto the running canvas.
        composite_feathered(&mut composite, &refined_resized, &bbox, cfg.feather);
    }

    // 6. Save back to the original path.
    composite
        .save(path)
        .with_context(|| format!("writing ADetailer composite to {}", path.display()))?;
    Ok(keep.len())
}

/// Expanded crop bounding box in original-image pixel space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandedBBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// v0.16 phase 6: expand a face bbox `[x1, y1, x2, y2]` by `padding`
/// on each side (relative to the bbox dimensions), then clamp to the
/// image. Returns `(x, y, w, h)` in pixels.
///
/// Example: a `(100, 100, 200, 200)` face with padding=0.25 expands
/// by 25 px on each side → `(75, 75, 250, 250)` then clamped.
pub fn expand_bbox(bbox: &[f32; 4], padding: f32, img_w: u32, img_h: u32) -> ExpandedBBox {
    let x1 = bbox[0];
    let y1 = bbox[1];
    let x2 = bbox[2];
    let y2 = bbox[3];
    let fw = (x2 - x1).max(1.0);
    let fh = (y2 - y1).max(1.0);
    let pad_x = fw * padding;
    let pad_y = fh * padding;
    let nx1 = (x1 - pad_x).floor().clamp(0.0, (img_w - 1) as f32) as u32;
    let ny1 = (y1 - pad_y).floor().clamp(0.0, (img_h - 1) as f32) as u32;
    let nx2 = (x2 + pad_x).ceil().clamp(0.0, img_w as f32) as u32;
    let ny2 = (y2 + pad_y).ceil().clamp(0.0, img_h as f32) as u32;
    let w = nx2.saturating_sub(nx1);
    let h = ny2.saturating_sub(ny1);
    ExpandedBBox { x: nx1, y: ny1, w, h }
}

/// Round down to the nearest multiple of 8 (VAE downsample factor).
/// `0..8` snaps to `8` (min working resolution).
fn round_down_8(n: u32) -> u32 {
    (n.max(8) / 8) * 8
}

/// Crop an RGB buffer to `(x, y, w, h)`. Returns an owned
/// `ImageBuffer`. Caller must guarantee the rect is in-bounds —
/// `expand_bbox` enforces this.
fn crop_rgb(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    if x.saturating_add(w) > img.width() || y.saturating_add(h) > img.height() {
        anyhow::bail!(
            "ADetailer crop ({x}, {y}, {w}, {h}) out of bounds for image \
             {}x{}",
            img.width(),
            img.height()
        );
    }
    Ok(image::imageops::crop_imm(img, x, y, w, h).to_image())
}

/// Find the single image file in `dir` whose name starts with `prefix`.
/// Returns an error if there's not exactly one match — guards against
/// the per-face loop accumulating stray files.
fn find_output(dir: &Path, prefix: &str) -> Result<PathBuf> {
    let mut matches = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with(prefix) {
                matches.push(path);
            }
        }
    }
    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => anyhow::bail!("no img2img output found in {} (prefix={})", dir.display(), prefix),
        n => anyhow::bail!(
            "{n} img2img outputs found in {} (prefix={}) — expected exactly 1",
            dir.display(),
            prefix
        ),
    }
}

/// v0.16 phase 6: feather-composite `refined` (sized exactly to bbox)
/// onto `canvas` at bbox origin, with a tapered alpha that fades from
/// `1.0` at the bbox centre to `0.0` at its edge. `feather` is the
/// fraction of the bbox half-dimension over which the fade happens —
/// `0.25` means the outer 25% fades, the inner 75% stays at full
/// opacity.
///
/// Math: per pixel, compute a normalised distance-to-edge in
/// `[0, 1]`:
///
/// ```text
///     d = min(x/w, (w-1-x)/w, y/h, (h-1-y)/h)
/// ```
///
/// Then `alpha = clamp(d / feather, 0, 1)` (so `d ≥ feather` gives
/// full opacity; `d == 0` gives zero opacity; linear in between).
pub fn composite_feathered(
    canvas: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    refined: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    bbox: &ExpandedBBox,
    feather: f32,
) {
    let w = bbox.w;
    let h = bbox.h;
    let fw = w as f32;
    let fh = h as f32;
    let feather = feather.clamp(0.0, 0.5);
    for ry in 0..h {
        for rx in 0..w {
            let cx = bbox.x + rx;
            let cy = bbox.y + ry;
            if cx >= canvas.width() || cy >= canvas.height() {
                continue;
            }
            // Distance to nearest edge, normalised to bbox-half.
            let d_left = (rx as f32) / (fw - 1.0).max(1.0);
            let d_right = (w - 1 - rx) as f32 / (fw - 1.0).max(1.0);
            let d_top = (ry as f32) / (fh - 1.0).max(1.0);
            let d_bot = (h - 1 - ry) as f32 / (fh - 1.0).max(1.0);
            let d = d_left.min(d_right).min(d_top).min(d_bot);
            let alpha = if feather > 0.0 {
                (d / feather).clamp(0.0, 1.0)
            } else {
                1.0
            };

            let dst = canvas.get_pixel_mut(cx, cy);
            let src = refined.get_pixel(rx, ry);
            for c in 0..3 {
                let a = alpha;
                let blended = (1.0 - a) * (dst.0[c] as f32) + a * (src.0[c] as f32);
                dst.0[c] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_bbox_grows_by_padding_fraction() {
        // A (100, 100, 200, 200) face — 100×100 bbox. Padding 0.25
        // → +25 px each side → (75, 75, 225, 225) → 150×150.
        let b = expand_bbox(&[100.0, 100.0, 200.0, 200.0], 0.25, 500, 500);
        assert_eq!(b, ExpandedBBox { x: 75, y: 75, w: 150, h: 150 });
    }

    #[test]
    fn expand_bbox_clamps_to_image_bounds() {
        // Face near the top-left corner — expanded bbox would go
        // negative, gets clamped to (0, 0).
        let b = expand_bbox(&[10.0, 10.0, 50.0, 50.0], 0.5, 200, 200);
        assert_eq!(b.x, 0);
        assert_eq!(b.y, 0);
        // x1 was clamped from -10 → 0; x2 grows by 20 to 70 — so
        // width = 70.
        assert_eq!(b.w, 70);
        assert_eq!(b.h, 70);
    }

    #[test]
    fn expand_bbox_clamps_to_image_max() {
        // Face near the bottom-right — bbox extends past the image.
        let b = expand_bbox(&[180.0, 180.0, 220.0, 220.0], 0.5, 200, 200);
        // x2 clamped from 240 → 200, x1 expanded from 180→160.
        assert_eq!(b.x, 160);
        assert_eq!(b.y, 160);
        assert_eq!(b.w, 40);
        assert_eq!(b.h, 40);
    }

    #[test]
    fn round_down_8_snaps_correctly() {
        assert_eq!(round_down_8(0), 8);
        assert_eq!(round_down_8(7), 8);
        assert_eq!(round_down_8(8), 8);
        assert_eq!(round_down_8(9), 8);
        assert_eq!(round_down_8(15), 8);
        assert_eq!(round_down_8(16), 16);
        assert_eq!(round_down_8(512), 512);
        assert_eq!(round_down_8(513), 512);
    }

    fn solid_rgb(w: u32, h: u32, color: [u8; 3]) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        ImageBuffer::from_fn(w, h, |_, _| Rgb(color))
    }

    #[test]
    fn composite_feathered_no_feather_is_hard_paste() {
        // feather=0 → alpha=1 everywhere → refined pixels overwrite
        // the canvas inside bbox, nothing changes outside.
        let mut canvas = solid_rgb(20, 20, [10, 20, 30]);
        let refined = solid_rgb(10, 10, [200, 100, 50]);
        let bbox = ExpandedBBox { x: 5, y: 5, w: 10, h: 10 };
        composite_feathered(&mut canvas, &refined, &bbox, 0.0);
        // Inside bbox center: refined colour.
        assert_eq!(canvas.get_pixel(10, 10).0, [200, 100, 50]);
        // Outside bbox: original.
        assert_eq!(canvas.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(canvas.get_pixel(19, 19).0, [10, 20, 30]);
    }

    #[test]
    fn composite_feathered_edge_alpha_is_zero() {
        // With feather > 0, the bbox's outermost pixel has d = 0
        // → alpha = 0 → original colour unchanged.
        let mut canvas = solid_rgb(20, 20, [10, 20, 30]);
        let refined = solid_rgb(10, 10, [200, 100, 50]);
        let bbox = ExpandedBBox { x: 5, y: 5, w: 10, h: 10 };
        composite_feathered(&mut canvas, &refined, &bbox, 0.25);
        // Corner pixel of bbox (canvas coord 5,5) — d_left=0 → alpha=0.
        assert_eq!(canvas.get_pixel(5, 5).0, [10, 20, 30]);
    }

    #[test]
    fn composite_feathered_center_alpha_is_full() {
        // Bbox centre is well inside the feather threshold → full
        // alpha → refined colour wins.
        let mut canvas = solid_rgb(20, 20, [0, 0, 0]);
        let refined = solid_rgb(20, 20, [255, 255, 255]);
        let bbox = ExpandedBBox { x: 0, y: 0, w: 20, h: 20 };
        composite_feathered(&mut canvas, &refined, &bbox, 0.25);
        // Centre (10,10): d_left = 10/19 ≈ 0.526; feather=0.25 →
        // alpha = clamp(0.526/0.25, 0, 1) = 1. Refined wins.
        assert_eq!(canvas.get_pixel(10, 10).0, [255, 255, 255]);
    }

    #[test]
    fn composite_feathered_bbox_at_image_edge() {
        // Bbox flush against the canvas edge — pixels at the canvas
        // boundary still receive feathered blending; no out-of-bounds.
        let mut canvas = solid_rgb(10, 10, [10, 10, 10]);
        let refined = solid_rgb(5, 5, [100, 100, 100]);
        // bbox covers the bottom-right quadrant.
        let bbox = ExpandedBBox { x: 5, y: 5, w: 5, h: 5 };
        composite_feathered(&mut canvas, &refined, &bbox, 0.25);
        // Centre of the bbox (canvas 7,7) gets some blend.
        let c = canvas.get_pixel(7, 7).0;
        // Should be between 10 and 100 (lerp), not panic.
        assert!(c[0] >= 10 && c[0] <= 100);
    }
}
