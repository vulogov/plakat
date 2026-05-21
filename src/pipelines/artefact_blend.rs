//! Masked low-strength img2img blending pass for artefact compositing
//! (v2). Runs *after* v1's alpha composite to integrate the pasted-on
//! cutouts with the surrounding generated content — fixes hard edges
//! and modest lighting mismatches at the cost of one short denoise
//! pass per image.
//!
//! Pipeline:
//!
//! 1. Load the SD pipeline (text-only — identity adapters skipped).
//! 2. VAE-encode the already-composited PNG → `base_latents`.
//! 3. Build a latent-space mask: `1.0` inside every artefact's zone
//!    (feathered ~16 px at image resolution, downsampled by VAE
//!    factor 8), `0.0` elsewhere.
//! 4. Run [`portrait::Pipeline::blend_latents_one`] — RePaint-style
//!    masked partial-strength denoise.
//! 5. VAE-decode and overwrite the composited PNG.
//!
//! Flux is not supported (portrait pipeline rejects it). SD 1.5 +
//! SDXL routes through this module.
//!
//! Design notes:
//!
//! * The mask uses the artefact's **zone rect** (not the cropped
//!   target rect inside the zone). The zone is broader, which both
//!   simplifies the math (we already have it on `ResolvedArtefact`)
//!   and creates a natural feathering margin — the denoiser blends
//!   the artefact edge with the surrounding pixels rather than just
//!   re-touching the artefact silhouette.
//! * Feathering is a separable box blur, not a Gaussian — cheap and
//!   visually indistinguishable at the strengths we use.
//! * `strength` follows standard img2img semantics: 0.0 = no-op,
//!   1.0 = full re-noise + denoise inside the mask. Sweet spot for
//!   blending is `0.25 – 0.4`. Above ~0.6 the model starts redrawing
//!   the artefact's silhouette and may "fix" it into something
//!   unrecognisable.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

use crate::artefacts::{
    resolve_specs, ArtefactLibrary, ArtefactSpec, Rect, ResolvedArtefact, ZoneOverrides,
};
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, GenRequest, LoadRequest};
use crate::pipelines::scheduler::SchedulerKind;

/// Feathering radius (pixels) at image resolution. Applied to the
/// union-of-zones mask before downsampling to latent space. ~2 % of a
/// 1024 px canvas — enough to soften the transition without bleeding
/// into half the image.
const DEFAULT_FEATHER_PX: u32 = 16;

/// SD's VAE downsample factor. Pipelines elsewhere also assume this.
const VAE_FACTOR: u32 = 8;

/// Configuration for one blend pass.
pub struct BlendConfig {
    pub model: String,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    pub prompt: String,
    pub negative: String,
    pub image_w: u32,
    pub image_h: u32,
    pub steps: usize,
    pub guidance: f64,
    pub scheduler: SchedulerKind,
    /// 0..1 img2img strength. ~0.3 is the recommended default.
    pub strength: f32,
    /// Feather radius in pixels. `None` → [`DEFAULT_FEATHER_PX`].
    pub feather_px: Option<u32>,
}

/// Run the blend pass on every file in `files` in-place. Empty
/// `specs` or empty `files` is a no-op (no model load).
///
/// When `smart` is supplied, the artefact zone rects (and therefore
/// the blend mask) are recomputed per file from each image's own
/// depth + luminance — same behaviour as the v3 compositor.
///
/// Phase 7d: `shared_core`, when `Some`, lets the blend pass reuse a
/// previously-loaded SD backbone (e.g. the one `t2i::run` just used
/// to generate the base images) instead of paying for a second load.
/// Pass `None` to keep the standalone behaviour — the function loads
/// its own portrait pipeline from `cfg.model` / `cfg.loras`. Callers
/// that supply `shared_core` are responsible for ensuring it was
/// loaded with the same model / device / LoRA set the blend pass
/// expects; this function does not re-validate.
pub async fn blend_files(
    cfg: BlendConfig,
    specs: &[ArtefactSpec],
    library_dir: &Path,
    files: &[PathBuf],
    zone_overrides: &ZoneOverrides,
    base_seed: Option<u64>,
    smart: Option<&crate::pipelines::depth::DepthPipeline>,
    shared_core: Option<std::sync::Arc<crate::pipelines::sd_core::SdCore>>,
) -> Result<()> {
    if specs.is_empty() || files.is_empty() {
        return Ok(());
    }
    let lib = ArtefactLibrary::load(library_dir)
        .with_context(|| format!("loading artefact library {}", library_dir.display()))?;

    let smart_tag = if smart.is_some() { " (smart zones)" } else { "" };
    let reuse_tag = if shared_core.is_some() {
        " (shared SD backbone)"
    } else {
        ""
    };
    crate::ui::progress::println(&format!(
        "  {} blending {} artefact(s) into {} image(s) (strength={:.2}){smart_tag}{reuse_tag}",
        console::style("◆").cyan().bold(),
        specs.len(),
        files.len(),
        cfg.strength,
    ));

    let pipeline = match shared_core {
        Some(core) => portrait::Pipeline::from_core(core),
        None => portrait::Pipeline::load(LoadRequest {
            model: cfg.model.clone(),
            device: cfg.device.clone(),
            loras: cfg.loras.clone(),
            lora_scale: cfg.lora_scale,
            identity: None,
        })
        .await
        .context("loading SD pipeline for artefact blend")?,
    };

    let feather_px = cfg.feather_px.unwrap_or(DEFAULT_FEATHER_PX);

    let start = base_seed.unwrap_or(0);
    for (i, path) in files.iter().enumerate() {
        let seed = start.wrapping_add(i as u64);

        // Per-file zone resolution. With smart=None this just clones
        // the base overrides, so resolve_specs / build_artefact_mask
        // run with consistent inputs across files — identical to
        // pre-v3 behaviour byte-for-byte.
        let effective = crate::artefacts::resolve_overrides_for(
            path,
            cfg.image_w,
            cfg.image_h,
            zone_overrides,
            smart,
        );
        let resolved = resolve_specs(specs, &lib, cfg.image_w, cfg.image_h, &effective)?;
        let mask = build_artefact_mask(
            &resolved,
            cfg.image_w,
            cfg.image_h,
            feather_px,
            pipeline.device(),
            pipeline.latent_dtype(),
        )?;

        let base_latents = pipeline
            .vae_encode_image_file(path, cfg.image_w, cfg.image_h)
            .with_context(|| format!("VAE-encoding {}", path.display()))?;
        let req = GenRequest {
            prompt: cfg.prompt.clone(),
            negative: cfg.negative.clone(),
            photos: Vec::new(),
            width: cfg.image_w,
            height: cfg.image_h,
            count: 1,
            steps: cfg.steps,
            guidance: cfg.guidance,
            seed: Some(seed),
            out_dir: path.parent().map(Path::to_path_buf).unwrap_or_default(),
            scheduler: cfg.scheduler,
            refine: None,
            refine_strength: 0.0,
            face_strength: 0.0,
            face_bbox: None,
            face_landmarks: None,
        };
        let new_latents = pipeline
            // Artefact-blend doesn't expose --control; the conditioner
            // here is the artefact mask itself, not a ControlNet guide.
            .blend_latents_one(&base_latents, &mask, &req, cfg.strength, seed, None)
            .with_context(|| format!("blend denoise on {}", path.display()))?;
        pipeline
            .save_image(&new_latents, path)
            .with_context(|| format!("writing blended {}", path.display()))?;
    }
    Ok(())
}

/// Build a latent-space artefact mask:
///
/// 1. Start with zeros at image resolution.
/// 2. Set every zone rect to `1.0`.
/// 3. Apply a separable box blur of radius `feather_px` for soft
///    edges.
/// 4. Average-pool by `VAE_FACTOR` to land in latent space.
/// 5. Return `(1, 1, latent_h, latent_w)` at `dtype` on `device`.
pub fn build_artefact_mask(
    resolved: &[ResolvedArtefact],
    image_w: u32,
    image_h: u32,
    feather_px: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let iw = image_w as usize;
    let ih = image_h as usize;
    let mut mask = vec![0f32; iw * ih];

    // Step 1: union of zone rects = 1.0.
    for r in resolved {
        rect_fill(&mut mask, iw, ih, &r.zone, 1.0);
    }

    // Step 2: separable box blur (horizontal then vertical).
    if feather_px > 0 {
        box_blur_inplace(&mut mask, iw, ih, feather_px as usize);
    }

    // Step 3: average-pool 8×8 into latent space.
    let latent_w = iw / VAE_FACTOR as usize;
    let latent_h = ih / VAE_FACTOR as usize;
    if latent_w == 0 || latent_h == 0 {
        anyhow::bail!(
            "artefact mask: image {iw}x{ih} too small to downsample by {}",
            VAE_FACTOR
        );
    }
    let latent = avg_pool_8x(&mask, iw, ih, latent_w, latent_h);

    let t = Tensor::from_vec(latent, (1, 1, latent_h, latent_w), device)?;
    Ok(t.to_dtype(dtype)?)
}

fn rect_fill(buf: &mut [f32], w: usize, h: usize, rect: &Rect, value: f32) {
    let x0 = (rect.x0 as usize).min(w);
    let x1 = (rect.x1 as usize).min(w);
    let y0 = (rect.y0 as usize).min(h);
    let y1 = (rect.y1 as usize).min(h);
    for y in y0..y1 {
        let row = &mut buf[y * w + x0..y * w + x1];
        for v in row.iter_mut() {
            *v = v.max(value); // union, not overwrite
        }
    }
}

/// Separable box blur (horizontal pass then vertical pass). Reflect
/// boundary handling is implicit — pixels past the edge act as 0,
/// which fades the mask toward the edges (intended for feathering).
fn box_blur_inplace(buf: &mut [f32], w: usize, h: usize, radius: usize) {
    if radius == 0 {
        return;
    }
    let k = (2 * radius + 1) as f32;
    let mut scratch = vec![0f32; buf.len()];

    // Horizontal.
    for y in 0..h {
        let row_start = y * w;
        // Rolling sum.
        let mut sum = 0f32;
        for x in 0..radius.min(w) {
            sum += buf[row_start + x];
        }
        for x in 0..w {
            let add = x + radius;
            if add < w {
                sum += buf[row_start + add];
            }
            if x > radius {
                sum -= buf[row_start + x - radius - 1];
            }
            scratch[row_start + x] = sum / k;
        }
    }

    // Vertical (read from scratch, write to buf).
    for x in 0..w {
        let mut sum = 0f32;
        for y in 0..radius.min(h) {
            sum += scratch[y * w + x];
        }
        for y in 0..h {
            let add = y + radius;
            if add < h {
                sum += scratch[add * w + x];
            }
            if y > radius {
                sum -= scratch[(y - radius - 1) * w + x];
            }
            buf[y * w + x] = sum / k;
        }
    }
}

/// Average-pool by `VAE_FACTOR`. Source dimensions need not be exact
/// multiples — partial cells at the right/bottom edge are dropped
/// (matches what the VAE would see after image-side resizing in
/// `sd_image_tensor`, which rescales to exact multiples up front).
fn avg_pool_8x(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let factor = VAE_FACTOR as usize;
    let norm = 1.0 / (factor * factor) as f32;
    let mut dst = vec![0f32; dst_w * dst_h];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let mut sum = 0f32;
            for ky in 0..factor {
                let sy = dy * factor + ky;
                if sy >= src_h {
                    continue;
                }
                for kx in 0..factor {
                    let sx = dx * factor + kx;
                    if sx >= src_w {
                        continue;
                    }
                    sum += src[sy * src_w + sx];
                }
            }
            dst[dy * dst_w + dx] = sum * norm;
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_fill_marks_correct_region() {
        let mut buf = vec![0f32; 16 * 16];
        rect_fill(
            &mut buf,
            16,
            16,
            &Rect {
                x0: 4,
                y0: 4,
                x1: 12,
                y1: 12,
            },
            1.0,
        );
        // Inside.
        assert_eq!(buf[5 * 16 + 5], 1.0);
        // Outside.
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[15 * 16 + 15], 0.0);
        // Boundary: x1/y1 are exclusive.
        assert_eq!(buf[12 * 16 + 12], 0.0);
        assert_eq!(buf[11 * 16 + 11], 1.0);
    }

    #[test]
    fn rect_fill_unions_overlapping_rects() {
        let mut buf = vec![0f32; 16 * 16];
        rect_fill(&mut buf, 16, 16, &Rect { x0: 0, y0: 0, x1: 8, y1: 8 }, 1.0);
        rect_fill(&mut buf, 16, 16, &Rect { x0: 4, y0: 4, x1: 12, y1: 12 }, 1.0);
        // Overlap region stays 1.0 (not 2.0).
        assert_eq!(buf[5 * 16 + 5], 1.0);
        // First-rect-only.
        assert_eq!(buf[1 * 16 + 1], 1.0);
        // Second-rect-only.
        assert_eq!(buf[10 * 16 + 10], 1.0);
        // Outside both.
        assert_eq!(buf[13 * 16 + 13], 0.0);
    }

    #[test]
    fn box_blur_softens_edges() {
        let mut buf = vec![0f32; 32 * 32];
        // Solid 16×16 block in the centre.
        for y in 8..24 {
            for x in 8..24 {
                buf[y * 32 + x] = 1.0;
            }
        }
        box_blur_inplace(&mut buf, 32, 32, 4);
        // Centre remains saturated.
        assert!(buf[16 * 32 + 16] > 0.95);
        // Boundary (just inside) drops below saturation.
        assert!(buf[8 * 32 + 8] < 0.5 && buf[8 * 32 + 8] > 0.0);
        // Outside far edge approaches 0.
        assert!(buf[0] < 0.05);
    }

    #[test]
    fn avg_pool_8x_downsamples_correctly() {
        // 16×16 source → 2×2 dst. Fill top-left 8×8 with 1.0, rest 0.
        let mut src = vec![0f32; 16 * 16];
        for y in 0..8 {
            for x in 0..8 {
                src[y * 16 + x] = 1.0;
            }
        }
        let dst = avg_pool_8x(&src, 16, 16, 2, 2);
        assert!((dst[0] - 1.0).abs() < 1e-5, "top-left = 1.0");
        assert!(dst[1].abs() < 1e-5, "top-right = 0.0");
        assert!(dst[2].abs() < 1e-5, "bottom-left = 0.0");
        assert!(dst[3].abs() < 1e-5, "bottom-right = 0.0");
    }

    #[test]
    fn build_artefact_mask_for_empty_resolved_is_all_zero() {
        let dev = Device::Cpu;
        let m = build_artefact_mask(&[], 64, 64, 4, &dev, DType::F32).unwrap();
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.abs() < 1e-5));
        // Shape (1, 1, 8, 8) for a 64×64 image with VAE factor 8.
        let dims = m.dims();
        assert_eq!(dims, &[1, 1, 8, 8]);
    }
}
