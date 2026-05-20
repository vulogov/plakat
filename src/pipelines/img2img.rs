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
}

/// Run the pipeline. Loads the SD model once and iterates over
/// `count` seeds, writing `plakat-img2img-<seed>.png` (or
/// `plakat-inpaint-<seed>.png` when a mask is supplied) for each.
pub async fn run(req: Request) -> Result<()> {
    let pipeline = portrait::Pipeline::load(LoadRequest {
        model: req.model.clone(),
        device: req.device.clone(),
        loras: req.loras.clone(),
        lora_scale: req.lora_scale,
        identity: None,
    })
    .await
    .context("loading SD pipeline for img2img")?;

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

        let new_latents = pipeline
            .blend_latents_one(&base_latents, &mask_tensor, &gen_req, req.strength, seed)
            .with_context(|| format!("denoise (seed {seed})"))?;

        let out_path = output_path(&req.out_dir, mode_tag, seed);
        pipeline.save_image(&new_latents, &out_path)?;
    }

    Ok(())
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
