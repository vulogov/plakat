//! Image-to-image and inpaint pipeline.
//!
//! Thin wrapper over [`portrait::Pipeline`] — the actual denoise
//! primitives (`vae_encode_image_file`, `blend_latents_one`) were
//! built for v2 artefact-blend and already do everything we need.
//! This module just wires up a CLI-friendly entry point.
//!
//! Two modes, picked by the caller:
//!
//! * **img2img** (no mask) — every pixel re-denoised at `strength`.
//!   Equivalent to passing an all-ones mask.
//! * **inpaint** (mask supplied) — only mask=1 pixels denoised at
//!   `strength`, mask=0 pixels preserved. Edge feathering on the
//!   mask softens the boundary.
//!
//! Flux is not supported — [`portrait::Pipeline`] rejects Flux at
//! load time, and this module inherits that constraint.

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::{Path, PathBuf};

use crate::imaging::mask::Mask;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, GenRequest, LoadRequest};
use crate::pipelines::scheduler::SchedulerKind;

/// One img2img / inpaint request.
pub struct Request {
    pub prompt: String,
    pub negative: String,
    pub model: String,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,

    /// Source image to transform.
    pub input: PathBuf,
    /// `None` → img2img (whole image denoised at `strength`).
    /// `Some(path)` → inpaint (only mask=1 pixels denoised).
    pub mask: Option<PathBuf>,
    pub mask_feather: u32,
    pub mask_invert: bool,

    /// Working resolution. Resized from the input.
    pub width: u32,
    pub height: u32,

    pub count: u32,
    pub steps: usize,
    pub guidance: f64,
    pub scheduler: SchedulerKind,
    /// img2img strength in `[0, 1]`. `1.0` = full re-noise + denoise.
    pub strength: f32,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,

    // ---------- v0.9 ControlNet (v0.11: multi) ----------
    /// Stack of ControlNet conditioners. See `t2i::Request::controls`.
    /// For `plakat img2img`, when a spec has neither `image=` nor
    /// `from=`, the input image is auto-annotated (img2img-specific
    /// default — t2i errors out instead because it has no canonical
    /// source image).
    pub controls: Vec<crate::pipelines::controlnet::ControlSpec>,
}

/// Run the pipeline. Loads the SD model once and iterates over
/// `count` seeds, writing `plakat-img2img-<seed>.png` (or
/// `plakat-inpaint-<seed>.png` when a mask is supplied) for each.
///
/// Returns the loaded SD backbone (`Arc<SdCore>`) so a follow-on
/// `--artefact-blend` pass can reuse the same weights — same pattern
/// the v0.10 generate / portrait paths use.
pub async fn run(
    req: Request,
) -> Result<std::sync::Arc<crate::pipelines::sd_core::SdCore>> {
    let pipeline = portrait::Pipeline::load(LoadRequest {
        model: req.model.clone(),
        device: req.device.clone(),
        loras: req.loras.clone(),
        lora_scale: req.lora_scale,
        identity: None,
        // img2img doesn't run identity encoding, so no CLIP-H needed.
        shared_clip_h: None,
    })
    .await
    .context("loading SD pipeline for img2img")?;
    run_with_pipeline(&pipeline, &req).await?;
    Ok(pipeline.core())
}

/// v0.14 phase 7: per-call img2img / inpaint body, factored out so
/// callers that already have a loaded `portrait::Pipeline` (e.g.
/// scenarios reusing the t2i SdCore via `portrait::Pipeline::from_core`)
/// don't pay the load cost a second time.
///
/// `pipeline` must be loaded with the model / device / LoRA set the
/// request expects — this function does not re-validate. The same
/// constraint applies to `Pipeline::from_core` callers in the v0.10
/// artefact-blend path.
pub async fn run_with_pipeline(
    pipeline: &portrait::Pipeline,
    req: &Request,
) -> Result<()> {
    // Pre-load the ControlNet stack + conditioning(s) once per call.
    // The owned data lives on this frame; ControlRequest borrows from
    // it below. img2img's distinguishing feature is the "auto-annotate
    // the input image" fallback when a spec has neither image= nor
    // from= — surface that as `fallback_input = &req.input`.
    let cn_dtype = if matches!(req.device, candle_core::Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::F16
    };
    let control_owned = crate::pipelines::controlnet::load_control_stack(
        &req.controls,
        &req.model,
        req.width,
        req.height,
        &req.device,
        cn_dtype,
        Some(&req.input),
    )
    .await?;

    // Build the mask once. img2img mode → solid 1.0; inpaint mode →
    // load from disk, optionally invert + feather.
    let mut mask = match req.mask.as_deref() {
        None => Mask::solid_one(req.width, req.height),
        Some(path) => Mask::load(path, req.width, req.height)
            .with_context(|| format!("loading mask {}", path.display()))?,
    };
    if req.mask_invert {
        mask.invert();
    }
    // Feather only meaningful when a real mask is present — solid_one
    // is already uniform, blurring it is a no-op but cheap. We still
    // skip to avoid wasted work.
    if req.mask.is_some() {
        mask.feather(req.mask_feather);
    }
    let mask_tensor = mask
        .to_latent_tensor(pipeline.device(), pipeline.latent_dtype())
        .context("encoding mask into latent space")?;

    // v0.12 Inpaint UNet (9-channel): prepare the VAE-encoded
    // `input × (1 - mask)` pixel image once. The inpaint UNet concats
    // it (plus the mask) onto the latent input at every denoise step,
    // so we only need to produce it once per img2img run rather than
    // per step. Covers both SD 1.5 inpaint and SDXL-Inpaint.
    let inpaint_masked_latents = if pipeline.core().is_inpaint && req.mask.is_some() {
        let masked_pixels = build_masked_pixels_tensor(
            &req.input,
            &mask,
            req.width,
            req.height,
            pipeline.device(),
            pipeline.latent_dtype(),
        )
        .context("preparing Inpaint UNet masked-image pixels")?;
        Some(
            pipeline
                .vae_encode_pixels(&masked_pixels)
                .context("VAE-encoding Inpaint UNet masked image")?,
        )
    } else {
        None
    };

    std::fs::create_dir_all(&req.out_dir)?;

    let start = req.seed.unwrap_or_else(rand::random);
    let mode_tag = if req.mask.is_some() { "inpaint" } else { "img2img" };
    crate::ui::progress::println(&format!(
        "  {} {} {} from {} (strength={:.2}, seed={start})",
        console::style("◆").cyan().bold(),
        req.count,
        mode_tag,
        req.input.display(),
        req.strength,
    ));

    for i in 0..req.count {
        let seed = start.wrapping_add(i as u64);

        let base_latents = pipeline
            .vae_encode_image_file(&req.input, req.width, req.height)
            .with_context(|| format!("VAE-encoding {}", req.input.display()))?;

        let gen_req = GenRequest {
            prompt: req.prompt.clone(),
            negative: req.negative.clone(),
            photos: Vec::new(),
            width: req.width,
            height: req.height,
            count: 1,
            steps: req.steps,
            guidance: req.guidance,
            seed: Some(seed),
            out_dir: req.out_dir.clone(),
            scheduler: req.scheduler,
            refine: None,
            refine_strength: 0.0,
            face_strength: 0.0,
            face_bbox: None,
            face_landmarks: None,
        };

        let control_reqs: Vec<crate::pipelines::controlnet::ControlRequest> = control_owned
            .iter()
            .map(|owned| crate::pipelines::controlnet::ControlRequest {
                net: &owned.net,
                conditioning: owned.conditioning.clone(),
                strength: owned.strength,
                start: owned.start,
                end: owned.end,
            })
            .collect();

        let new_latents = pipeline
            .blend_latents_one(
                &base_latents,
                &mask_tensor,
                &gen_req,
                req.strength,
                seed,
                &control_reqs,
                inpaint_masked_latents.as_ref(),
            )
            .with_context(|| format!("denoise (seed {seed})"))?;

        let out_path = output_path(&req.out_dir, mode_tag, seed);
        pipeline.save_image(&new_latents, &out_path)?;
    }

    Ok(())
}

/// Build the `(1, 3, H, W)` masked-image tensor a 9-channel inpaint
/// UNet expects (SD 1.5 inpaint and SDXL-Inpaint use the same shape).
/// Loads `input`, resizes to `(w, h)`, multiplies each RGB channel by
/// `(1 - mask)` so masked pixels become 0 (the diffusers convention
/// for `masked_image`), then packs into f32 normalised to `[-1, 1]`.
fn build_masked_pixels_tensor(
    input: &Path,
    mask: &Mask,
    w: u32,
    h: u32,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<candle_core::Tensor> {
    let img = image::open(input)
        .with_context(|| format!("opening {} for Inpaint UNet masked image", input.display()))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &img,
        w,
        h,
        image::imageops::FilterType::CatmullRom,
    );
    if mask.pixels.len() != (w as usize) * (h as usize) {
        anyhow::bail!(
            "mask resolution {}×{} ({} pixels) doesn't match working size {w}×{h} \
             ({} pixels). The mask must be resized to the working size before \
             building masked-image latents.",
            mask.width,
            mask.height,
            mask.pixels.len(),
            (w as usize) * (h as usize)
        );
    }
    let total = (w as usize) * (h as usize);
    let mut data: Vec<f32> = vec![0.0; 3 * total];
    let (r_dst, rest) = data.split_at_mut(total);
    let (g_dst, b_dst) = rest.split_at_mut(total);
    let scale = 1.0 / 127.5;
    let raw = resized.as_raw();
    // Diffusers masked-image convention: mask=1 → zero out, mask=0 →
    // keep. After multiplying raw pixels by (1 - mask) in [0, 255]
    // space, we apply the standard SD VAE normalisation (`x/127.5 - 1`)
    // to land in [-1, 1]. Masked (zeroed) pixels land at -1 (black).
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        let inv_m = 1.0 - mask.pixels[i].clamp(0.0, 1.0);
        r_dst[i] = (chunk[0] as f32 * inv_m) * scale - 1.0;
        g_dst[i] = (chunk[1] as f32 * inv_m) * scale - 1.0;
        b_dst[i] = (chunk[2] as f32 * inv_m) * scale - 1.0;
    }
    let t = candle_core::Tensor::from_vec(data, (1, 3, h as usize, w as usize), device)?
        .to_dtype(dtype)?;
    Ok(t)
}

fn output_path(out_dir: &Path, mode_tag: &str, seed: u64) -> PathBuf {
    out_dir.join(format!("plakat-{mode_tag}-{seed}.png"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_uses_mode_tag() {
        let p = output_path(Path::new("/tmp/out"), "img2img", 42);
        assert_eq!(p, PathBuf::from("/tmp/out/plakat-img2img-42.png"));
        let p = output_path(Path::new("/tmp/out"), "inpaint", 7);
        assert_eq!(p, PathBuf::from("/tmp/out/plakat-inpaint-7.png"));
    }
}
