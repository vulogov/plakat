//! v0.21 phase 2: thin façade over plakat's pipelines for the
//! `plakat.*` host words.
//!
//! `cli::generate::run` is ~600 lines that include wildcard
//! expansion, fast presets, recipe loading, style detection, ADetailer,
//! Hires fix, etc — all CLI-flow concerns the script doesn't need.
//! [`generate_one`] strips that down to "build a minimal `t2i::Request`,
//! call `t2i::run`, read the rendered PNG back into memory."
//!
//! The output PNG is written to a tempdir per call so we don't pollute
//! the user's `--out` directory until `plakat.save` is invoked
//! explicitly. The image is held in `ScriptCtx.images` keyed by the
//! integer handle the host word pushes onto the stack.
//!
//! Limitations (intentional for phase 2):
//!
//! * **SD-family only** (sd15 / sd21 / sdxl / sdxl-turbo).
//!   Flux / SD3 bail loud in [`validate_supported_for_phase_2`].
//!   Full family coverage lands in phase 2b / a follow-up.
//! * **No pipeline reuse.** Every `plakat.generate` call re-runs
//!   [`t2i::run`] from scratch, which reloads the model. Acceptable
//!   for the smoke + phase 2 walkthrough; phase 4 (`plakat.img2img`)
//!   will likely revisit the cache story to avoid paying the load
//!   cost three times in a row.
//! * **Hardcoded defaults** for steps / guidance / size / seed.
//!   Phase 3 (`plakat.config.set`) makes these scriptable.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::Device;
use image::DynamicImage;
use std::path::PathBuf;

use crate::pipelines::t2i;
use crate::scripting::config::GenerationConfig;

/// v0.21 phase 2 gate: only SD-family models go through the
/// script entry. Flux + SD3 require additional plumbing
/// (different Request fields, different default sizes, T5
/// considerations) and land in phase 2b.
pub fn validate_supported_for_phase_2(model: &str) -> Result<()> {
    let variant = t2i::Variant::detect(model);
    if variant.is_flux() {
        bail!(
            "plakat.load: Flux models aren't wired in v0.21 phase 2 \
             (got {model:?}). Phase 2b lands Flux + SD3 support. For \
             now use SD-family aliases: sd15, sd21, sdxl, sdxl-turbo."
        );
    }
    if variant.is_sd3() {
        bail!(
            "plakat.load: SD3 / SD3.5 models aren't wired in v0.21 \
             phase 2 (got {model:?}). Phase 2b lands Flux + SD3 \
             support. For now use SD-family aliases: sd15, sd21, \
             sdxl, sdxl-turbo."
        );
    }
    Ok(())
}

/// v0.21 phase 2 + 3: render one image using the script's
/// accumulated [`GenerationConfig`].
///
/// Returns the rendered image as an in-memory [`DynamicImage`]
/// (read back from the tempdir `t2i::run` writes into). The
/// caller stores it in `ScriptCtx.images` and pushes a handle.
///
/// `config.size_explicit == false` means the script never called
/// `plakat.config.set width|height`; in that case we pick the
/// SD-family default for the loaded model (SDXL → 1024², everything
/// else → 512²) so a minimal `"sd15" plakat.load "fox" plakat.generate`
/// still works without a manual size call.
pub async fn generate_one(
    model: &str,
    prompt: &str,
    device: Device,
    config: &GenerationConfig,
) -> Result<DynamicImage> {
    validate_supported_for_phase_2(model)?;

    let variant = t2i::Variant::detect(model);
    let (width, height) = if config.size_explicit {
        if config.width == 0 || config.height == 0 {
            bail!(
                "plakat.generate: size_explicit set but width/height is 0 — \
                 only one of plakat.config.set width / height was called?"
            );
        }
        (config.width, config.height)
    } else if variant.is_xl() {
        (1024u32, 1024u32)
    } else {
        (512u32, 512u32)
    };

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-")
        .tempdir()
        .context("creating tempdir for plakat.generate output")?;
    let tmp_path: PathBuf = tmp.path().to_path_buf();

    let req = t2i::Request {
        prompt: prompt.to_string(),
        negative: config.negative.clone(),
        model: model.to_string(),
        width,
        height,
        count: 1,
        steps: config.steps,
        guidance: config.guidance,
        seed: config.seed,
        out_dir: tmp_path.clone(),
        device,
        loras: Vec::new(),
        lora_scale: 1.0,
        scheduler: config.scheduler,
        refine: None,
        refine_strength: 0.3,
        use_refiner: false,
        refiner_frac: 0.8,
        controls: Vec::new(),
        tiled: None,
        kontext_bucket: false,
        quantize_t5: false,
        flux_quant_level: None,
        t5_quant_level: None,
        redux_images: Vec::new(),
        flux_concept_image: None,
        clip_skip: 1,
        embeddings: Vec::new(),
        // We're rendering into a tempdir that gets dropped right
        // after this fn returns; the metadata sidecar would die
        // with it. Disable so we don't waste IO on files no one
        // reads. (When `plakat.save` lands, it'll re-embed metadata
        // at write time if asked.)
        write_metadata: false,
        preview_every: None,
        preview_size: Some(384),
        output_format: crate::imaging::io::OutputFormat::Png,
    };

    let _ = t2i::run(req).await.context("t2i::run in plakat.generate")?;

    // t2i writes `plakat-<seed>.png` (one entry since count=1).
    // Find the PNG. We don't know the seed up-front (it was
    // generated randomly inside the pipeline when we passed
    // `seed: None`), so glob the tempdir for the single PNG file.
    let rendered = std::fs::read_dir(&tmp_path)
        .with_context(|| format!("reading tempdir {}", tmp_path.display()))?
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
                "plakat.generate: t2i::run produced no PNG in {} \
                 — pipeline may have silently failed",
                tmp_path.display()
            )
        })?;

    let img = image::open(rendered.path())
        .with_context(|| format!("reading rendered PNG {}", rendered.path().display()))?;
    Ok(img)
}

/// v0.21 phase 4: render one img2img image using the script's
/// accumulated [`GenerationConfig`]. `input_path` is the source
/// image — the caller is responsible for materialising
/// in-memory handles to disk before invoking this (see
/// `words::img2img`).
///
/// Working size resolution:
/// * `config.size_explicit == true` → use `config.width × config.height`
/// * else → read the input image's dimensions, snap to /8
///
/// The /8 snap is what `cli::img2img` does too; downsizing happens
/// inside the pipeline.
pub async fn img2img_one(
    model: &str,
    prompt: &str,
    input_path: &std::path::Path,
    device: Device,
    config: &GenerationConfig,
) -> Result<DynamicImage> {
    validate_supported_for_phase_2(model)?;

    let (width, height) = if config.size_explicit {
        if config.width == 0 || config.height == 0 {
            bail!(
                "plakat.img2img: size_explicit set but width/height is 0 — \
                 only one of plakat.config.set width / height was called?"
            );
        }
        (config.width, config.height)
    } else {
        let dims = image::image_dimensions(input_path).with_context(|| {
            format!(
                "reading dimensions of {} for plakat.img2img working size",
                input_path.display()
            )
        })?;
        // Snap to /8 (VAE constraint). Round down so we never
        // upscale the input silently.
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
    let tmp_path: std::path::PathBuf = tmp.path().to_path_buf();

    let req = crate::pipelines::img2img::Request {
        prompt: prompt.to_string(),
        negative: config.negative.clone(),
        model: model.to_string(),
        device,
        loras: Vec::new(),
        lora_scale: 1.0,
        input: input_path.to_path_buf(),
        mask: None,
        mask_feather: 0,
        mask_invert: false,
        width,
        height,
        count: 1,
        steps: config.steps,
        guidance: config.guidance,
        scheduler: config.scheduler,
        strength: config.strength,
        seed: config.seed,
        out_dir: tmp_path.clone(),
        controls: Vec::new(),
    };

    let _ = crate::pipelines::img2img::run(req)
        .await
        .context("img2img::run in plakat.img2img")?;

    // img2img writes `plakat-img2img-<seed>.png` (single output
    // since count=1). Glob for the single PNG.
    let rendered = std::fs::read_dir(&tmp_path)
        .with_context(|| format!("reading tempdir {}", tmp_path.display()))?
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
                "plakat.img2img: img2img::run produced no PNG in {} \
                 — pipeline may have silently failed",
                tmp_path.display()
            )
        })?;

    let img = image::open(rendered.path())
        .with_context(|| format!("reading rendered PNG {}", rendered.path().display()))?;
    Ok(img)
}

/// v0.21 phase 5: pick the IP-Adapter identity strategy for a
/// model alias.
///
/// SD 1.5 → `PlusFace`; SDXL → `PlusFaceSdxl`. SD 2.1 has no
/// shipped IP-Adapter-Plus-Face checkpoint, so it bails — the
/// caller should switch to `sd15` or `sdxl`. FaceID variants
/// require user-supplied ArcFace env vars + better suit a
/// multi-photo workflow; deferred to v0.22 per RFC §5.1.
pub fn pick_portrait_identity(
    model: &str,
) -> Result<crate::pipelines::ip_adapter::IdentityKind> {
    use crate::pipelines::ip_adapter::IdentityKind;
    // Resolve aliases (sd21 → stabilityai/stable-diffusion-2-1)
    // before variant detection — Variant::detect matches against
    // substrings of the resolved repo id, not against the short
    // alias.
    let resolved = if model.contains('/') {
        model.to_string()
    } else {
        crate::hf::resolve_alias(model).to_string()
    };
    let variant = t2i::Variant::detect(&resolved);
    if variant.is_xl() {
        return Ok(IdentityKind::PlusFaceSdxl);
    }
    // SD 1.5 vs SD 2.1 — Variant::detect doesn't expose an
    // is_sd15-only helper, so cover the negative space directly.
    if matches!(variant, t2i::Variant::Sd21) {
        bail!(
            "plakat.portrait: SD 2.1 has no IP-Adapter-Plus-Face checkpoint \
             (got {model:?}). Use --model sd15 or sdxl for v0.21 phase 5."
        );
    }
    // Anything else SD-family (sd15, default) → PlusFace.
    Ok(IdentityKind::PlusFace)
}

/// v0.21 phase 5: render one IP-Adapter portrait. Single reference
/// photo only — FaceID + multi-photo blends deferred to v0.22
/// per RFC §5.1. Same SD-family gate as `generate_one` /
/// `img2img_one`.
pub async fn portrait_one(
    model: &str,
    prompt: &str,
    photo_path: &std::path::Path,
    device: Device,
    config: &GenerationConfig,
) -> Result<DynamicImage> {
    validate_supported_for_phase_2(model)?;
    let identity = pick_portrait_identity(model)?;

    // Same size resolution as generate_one — SD 1.5 → 512², SDXL →
    // 1024², unless the script set width/height explicitly.
    let variant = t2i::Variant::detect(model);
    let (width, height) = if config.size_explicit {
        if config.width == 0 || config.height == 0 {
            bail!(
                "plakat.portrait: size_explicit set but width/height is 0 — \
                 only one of plakat.config.set width / height was called?"
            );
        }
        (config.width, config.height)
    } else if variant.is_xl() {
        (1024u32, 1024u32)
    } else {
        (512u32, 512u32)
    };

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-portrait-")
        .tempdir()
        .context("creating tempdir for plakat.portrait output")?;
    let tmp_path: std::path::PathBuf = tmp.path().to_path_buf();

    let req = crate::pipelines::portrait::Request {
        prompt: prompt.to_string(),
        negative: config.negative.clone(),
        photos: vec![crate::pipelines::ip_adapter::WeightedPhoto::single(
            photo_path.to_path_buf(),
        )],
        model: model.to_string(),
        width,
        height,
        count: 1,
        steps: config.steps,
        guidance: config.guidance,
        seed: config.seed,
        out_dir: tmp_path.clone(),
        device,
        loras: Vec::new(),
        lora_scale: 1.0,
        scheduler: config.scheduler,
        refine: None,
        refine_strength: 0.3,
        face_strength: config.face_strength,
        // v0.21 phase 5 MVP: no manual landmarks / bbox. SCRFD
        // auto-detect kicks in if the user has PLAKAT_SCRFD_*
        // configured; otherwise the pipeline falls back to a
        // centre crop, same as cli::portrait does without the
        // flags.
        face_bbox: None,
        face_landmarks: None,
        identity: Some(identity),
        // Phase 7f shared CLIP-H: irrelevant for a script that
        // isn't running a style pass at the same time. Set to
        // None; the encoder will mmap its own copy.
        shared_clip_h: None,
        controls: Vec::new(),
    };

    let _ = crate::pipelines::portrait::run(req)
        .await
        .context("portrait::run in plakat.portrait")?;

    // portrait writes `plakat-portrait-<seed>.png` (single output
    // since count=1). Glob for the single PNG.
    let rendered = std::fs::read_dir(&tmp_path)
        .with_context(|| format!("reading tempdir {}", tmp_path.display()))?
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
                "plakat.portrait: portrait::run produced no PNG in {} \
                 — pipeline may have silently failed",
                tmp_path.display()
            )
        })?;

    let img = image::open(rendered.path())
        .with_context(|| format!("reading rendered PNG {}", rendered.path().display()))?;
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_2_gate_accepts_sd_family_aliases() {
        for alias in &["sd15", "sd21", "sdxl", "sdxl-turbo"] {
            validate_supported_for_phase_2(alias).unwrap_or_else(|e| {
                panic!("alias {alias:?} should be accepted in phase 2: {e}")
            });
        }
    }

    #[test]
    fn phase_2_gate_rejects_flux_aliases_with_helpful_message() {
        for alias in &["flux-dev", "flux-schnell", "flux-kontext-dev"] {
            let err = validate_supported_for_phase_2(alias).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("Flux"), "alias {alias:?}: {msg}");
            assert!(msg.contains("Phase 2b"), "alias {alias:?}: {msg}");
            assert!(msg.contains("sd15"), "alias {alias:?}: {msg}");
        }
    }

    #[test]
    fn phase_2_gate_rejects_sd3_aliases_with_helpful_message() {
        for alias in &["sd35-medium", "sd35-large", "sd3-medium"] {
            let err = validate_supported_for_phase_2(alias).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("SD3"), "alias {alias:?}: {msg}");
            assert!(msg.contains("Phase 2b"), "alias {alias:?}: {msg}");
        }
    }

    // v0.21 phase 4: img2img_one shares the validate_supported_for_phase_2
    // gate with generate_one, so the same Flux/SD3 bail fires. We
    // don't need a separate test for that — the existing
    // phase_2_gate_rejects_flux_aliases_with_helpful_message
    // covers it through validate_supported_for_phase_2 directly.

    // v0.21 phase 5: portrait identity auto-pick.

    #[test]
    fn pick_portrait_identity_sd15_returns_plus_face() {
        use crate::pipelines::ip_adapter::IdentityKind;
        let id = pick_portrait_identity("sd15").unwrap();
        assert!(matches!(id, IdentityKind::PlusFace));
    }

    #[test]
    fn pick_portrait_identity_sdxl_returns_plus_face_sdxl() {
        use crate::pipelines::ip_adapter::IdentityKind;
        let id = pick_portrait_identity("sdxl").unwrap();
        assert!(matches!(id, IdentityKind::PlusFaceSdxl));
        let id = pick_portrait_identity("sdxl-turbo").unwrap();
        assert!(matches!(id, IdentityKind::PlusFaceSdxl));
    }

    #[test]
    fn pick_portrait_identity_sd21_bails_with_helpful_message() {
        let err = pick_portrait_identity("sd21").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("SD 2.1"), "got {msg}");
        assert!(msg.contains("sd15"), "got {msg}");
        assert!(msg.contains("sdxl"), "got {msg}");
    }

    #[test]
    fn phase_2_gate_passes_canonical_hf_repos_when_they_resolve_to_sd_family() {
        // Variant::detect runs on the raw string, so HF repo paths
        // that resolve to SD-family should also pass.
        validate_supported_for_phase_2(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        )
        .unwrap();
    }
}
