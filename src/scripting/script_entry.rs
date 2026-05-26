//! Thin façade over plakat's pipelines for the `plakat.*` host
//! words. Cache-aware as of v0.22 phase 1.
//!
//! In v0.21 every image-producing word called `t2i::run` /
//! `img2img::run` / `portrait::run`, which each reloaded the
//! model. v0.22 phase 1 caches the loaded pipeline in
//! [`ScriptCtx::loaded`] and reuses it across calls — a single
//! SDXL load amortises across a whole script.
//!
//! Architectural choice (RFC §7): cache a `portrait::Pipeline`
//! because it generalises across the three image-producing
//! words. Phases 2-3 will add `flux::Pipeline` and
//! `sd3::Pipeline` variants and lift the SD-family gate.
//!
//! All three `*_one` functions are sync as of phase 1. The
//! pipeline calls themselves are sync; the model-load happens
//! inside [`ScriptCtx::get_or_load_sd_family`] which uses
//! `tokio::task::block_in_place` internally. The img2img path
//! is the one remaining async caller (`img2img::run_with_pipeline`
//! is async); it bridges via `block_in_place` here.

use anyhow::{Context, Result, anyhow, bail};
use image::DynamicImage;
use std::path::{Path, PathBuf};

use crate::pipelines::{ip_adapter::WeightedPhoto, portrait, t2i};
use crate::scripting::ctx::ScriptCtx;

/// SD-family gate. Phases 2-3 will lift this in favour of a
/// family-dispatching variant of [`ScriptCtx::get_or_load_sd_family`].
pub fn validate_supported_for_phase_2(model: &str) -> Result<()> {
    let variant = t2i::Variant::detect(model);
    if variant.is_flux() {
        bail!(
            "plakat.load: Flux models aren't wired in v0.22 phase 1 \
             (got {model:?}). Phase 2 lands Flux support. For now \
             use SD-family aliases: sd15, sd21, sdxl, sdxl-turbo."
        );
    }
    if variant.is_sd3() {
        bail!(
            "plakat.load: SD3 / SD3.5 models aren't wired in v0.22 \
             phase 1 (got {model:?}). Phase 3 lands SD3 support. \
             For now use SD-family aliases: sd15, sd21, sdxl, \
             sdxl-turbo."
        );
    }
    Ok(())
}

/// Pick the per-family default size used when the script hasn't
/// set width / height explicitly. SD 1.5 / 2.1 → 512²;
/// SDXL / SDXL-Turbo → 1024². Reads the alias on `ctx.loaded`.
fn default_size_for_loaded(ctx: &ScriptCtx) -> (u32, u32) {
    let alias = ctx
        .loaded_model()
        .expect("default_size called without a loaded pipeline");
    let resolved = if alias.contains('/') {
        alias.to_string()
    } else {
        crate::hf::resolve_alias(alias).to_string()
    };
    let variant = t2i::Variant::detect(&resolved);
    if variant.is_xl() { (1024, 1024) } else { (512, 512) }
}

/// Build a `portrait::GenRequest` from the script's accumulated
/// `GenerationConfig`. Shared across all three image-producing
/// host words; only `prompt` + `photos` + `out_dir` differ
/// per-call.
fn build_gen_request(
    ctx: &ScriptCtx,
    prompt: &str,
    photos: Vec<WeightedPhoto>,
    out_dir: PathBuf,
) -> portrait::GenRequest {
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        default_size_for_loaded(ctx)
    };
    portrait::GenRequest {
        prompt: prompt.to_string(),
        negative: ctx.config.negative.clone(),
        photos,
        width,
        height,
        count: 1,
        steps: ctx.config.steps,
        guidance: ctx.config.guidance,
        seed: ctx.config.seed,
        out_dir,
        scheduler: ctx.config.scheduler,
        refine: None,
        refine_strength: 0.3,
        face_strength: ctx.config.face_strength,
        face_bbox: None,
        face_landmarks: None,
    }
}

/// Locate the single PNG `pipeline.generate` writes into `dir`
/// and load it as a [`DynamicImage`]. Pipelines name their
/// outputs `plakat-<seed>.png`, `plakat-portrait-<seed>.png`,
/// or `plakat-img2img-<seed>.png`; we don't try to predict the
/// filename — we just grab the single PNG file.
fn read_rendered_png(dir: &Path) -> Result<DynamicImage> {
    let entry = std::fs::read_dir(dir)
        .with_context(|| format!("reading tempdir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .find(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            anyhow!(
                "pipeline produced no PNG in {} — pipeline may have \
                 silently failed",
                dir.display()
            )
        })?;
    image::open(entry.path())
        .with_context(|| format!("reading rendered PNG {}", entry.path().display()))
}

/// v0.22 phase 1: render one image with the cached SD-family
/// pipeline. Bails if no model has been loaded.
pub fn generate_one(ctx: &mut ScriptCtx, prompt: &str) -> Result<DynamicImage> {
    let alias = ctx
        .loaded_model()
        .ok_or_else(|| {
            anyhow!(
                "plakat.generate: no model loaded. Call \"sd15\" plakat.load \
                 (or your model of choice) before plakat.generate."
            )
        })?
        .to_string();

    // Build the GenRequest first (immutable read of ctx.config +
    // alias-derived defaults). Then borrow the pipeline mutably.
    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-gen-")
        .tempdir()
        .context("creating tempdir for plakat.generate output")?;
    let req = build_gen_request(ctx, prompt, Vec::new(), tmp.path().to_path_buf());

    let pipeline = ctx.get_or_load_sd_family(&alias)?;
    pipeline.generate(&req, &[])
        .context("portrait::Pipeline::generate (plakat.generate path)")?;
    read_rendered_png(tmp.path())
}

/// v0.22 phase 1: render one img2img image. `input_path` may be
/// any filesystem path the script provides (or a tempfile from a
/// handle materialisation in the host word).
pub fn img2img_one(
    ctx: &mut ScriptCtx,
    prompt: &str,
    input_path: &Path,
) -> Result<DynamicImage> {
    let alias = ctx
        .loaded_model()
        .ok_or_else(|| {
            anyhow!(
                "plakat.img2img: no model loaded. Call \"sd15\" plakat.load \
                 (or your model of choice) before plakat.img2img."
            )
        })?
        .to_string();

    // Working size: explicit config wins; else input image dims
    // snapped to /8 (downward). Read config + dims first, then
    // borrow the pipeline mutably.
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        let dims = image::image_dimensions(input_path).with_context(|| {
            format!(
                "reading dimensions of {} for plakat.img2img working size",
                input_path.display()
            )
        })?;
        let (w, h) = dims;
        ((w / 8) * 8, (h / 8) * 8)
    };
    if width == 0 || height == 0 {
        bail!(
            "plakat.img2img: working size {width}x{height} collapsed to 0 \
             after /8 snap. Input image is too small (< 8 pixels on a side)."
        );
    }

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-i2i-")
        .tempdir()
        .context("creating tempdir for plakat.img2img output")?;
    let req = crate::pipelines::img2img::Request {
        prompt: prompt.to_string(),
        negative: ctx.config.negative.clone(),
        model: alias.clone(),
        device: ctx.device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        input: input_path.to_path_buf(),
        mask: None,
        mask_feather: 0,
        mask_invert: false,
        width,
        height,
        count: 1,
        steps: ctx.config.steps,
        guidance: ctx.config.guidance,
        scheduler: ctx.config.scheduler,
        strength: ctx.config.strength,
        seed: ctx.config.seed,
        out_dir: tmp.path().to_path_buf(),
        controls: Vec::new(),
    };

    let pipeline = ctx.get_or_load_sd_family(&alias)?;

    // run_with_pipeline is async; bridge here via block_in_place
    // (same pattern as get_or_load_sd_family's internal load).
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow!(
            "plakat.img2img: no tokio runtime in scope (eval must run on \
             a multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    tokio::task::block_in_place(|| {
        handle.block_on(crate::pipelines::img2img::run_with_pipeline(pipeline, &req))
    })
    .context("img2img::run_with_pipeline (plakat.img2img path)")?;
    read_rendered_png(tmp.path())
}

/// v0.22 phase 1: render one portrait. Uses the cached
/// pipeline's identity encoder; if the pipeline was loaded
/// without one (sd21), `pipeline.generate` bails with the
/// v0.21 "no identity encoder" message.
pub fn portrait_one(
    ctx: &mut ScriptCtx,
    prompt: &str,
    photo_path: &Path,
) -> Result<DynamicImage> {
    let alias = ctx
        .loaded_model()
        .ok_or_else(|| {
            anyhow!(
                "plakat.portrait: no model loaded. Call \"sd15\" plakat.load \
                 (or \"sdxl\" plakat.load) before plakat.portrait."
            )
        })?
        .to_string();

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-portrait-")
        .tempdir()
        .context("creating tempdir for plakat.portrait output")?;
    let photos = vec![WeightedPhoto::single(photo_path.to_path_buf())];
    let mut req = build_gen_request(ctx, prompt, photos, tmp.path().to_path_buf());
    // Override per-family default size for portrait: 3:4
    // is the CLI default. Honour size_explicit override.
    if !ctx.config.size_explicit {
        let (w, h) = default_size_for_loaded(ctx);
        // CLI portrait default is 3:4; for SDXL → 768×1024,
        // for SD 1.5 → 512×768. Map from the square default.
        req.width = w * 3 / 4;
        req.height = h;
        // VAE-snap to /8.
        req.width = (req.width / 8) * 8;
        req.height = (req.height / 8) * 8;
    }
    // Normalize photo weights (the pipeline's invariant).
    crate::pipelines::ip_adapter::normalize_photo_weights(&mut req.photos)?;

    let pipeline = ctx.get_or_load_sd_family(&alias)?;
    pipeline.generate(&req, &[])
        .context("portrait::Pipeline::generate (plakat.portrait path)")?;
    read_rendered_png(tmp.path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_2_gate_accepts_sd_family_aliases() {
        for alias in &["sd15", "sd21", "sdxl", "sdxl-turbo"] {
            validate_supported_for_phase_2(alias).unwrap_or_else(|e| {
                panic!("alias {alias:?} should be accepted in phase 1: {e}")
            });
        }
    }

    #[test]
    fn phase_2_gate_rejects_flux_aliases() {
        for alias in &["flux-dev", "flux-schnell", "flux-kontext-dev"] {
            let err = validate_supported_for_phase_2(alias).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("Flux"), "alias {alias:?}: {msg}");
            assert!(msg.contains("Phase 2"), "alias {alias:?}: {msg}");
        }
    }

    #[test]
    fn phase_2_gate_rejects_sd3_aliases() {
        for alias in &["sd35-medium", "sd35-large", "sd3-medium"] {
            let err = validate_supported_for_phase_2(alias).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("SD3"), "alias {alias:?}: {msg}");
            assert!(msg.contains("Phase 3"), "alias {alias:?}: {msg}");
        }
    }

    #[test]
    fn phase_2_gate_passes_canonical_hf_repos_for_sd_family() {
        validate_supported_for_phase_2(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        )
        .unwrap();
    }
}
