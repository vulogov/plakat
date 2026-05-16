//! Style-reference pipeline: IN + REF → OUT.
//!
//! Architecture:
//!   1. VAE-encode IN → latents.
//!   2. CLIP-H image-encode REF → image_embeds (1024-d). Project via
//!      IP-Adapter `image_proj` → 4 image tokens (768-d each).
//!   3. Concat empty-text tokens (77) with image tokens (4) → (1, 81, 768)
//!      encoder_hidden_states.
//!   4. Img2img denoise: add noise to IN-latents at strength·T, run the
//!      denoising loop from that timestep with the conditioning above.
//!   5. VAE-decode → OUT.
//!
//! Currently SD 1.5 only. SDXL IP-Adapter (different image encoder dims,
//! different projection target) is a follow-up.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::ip_adapter::{ImageEncoder, ImageProj};
use crate::ui::progress;

pub struct Request {
    pub input: PathBuf,
    pub reference: PathBuf,
    pub out: PathBuf,
    pub strength: f32,
    pub model: String,
    pub steps: usize,
    pub seed: Option<u64>,
    pub device: Device,
}

const IPA_REPO: &str = "h94/IP-Adapter";
const SD15_CROSS_ATTN_DIM: usize = 768;
const IPA_TOKENS: usize = 4;
const CLIP_H_PROJ_DIM: usize = 1024;
const CLIP_H_INPUT: u32 = 224;

pub async fn run(req: Request) -> Result<()> {
    let base_repo = if req.model.contains('/') {
        req.model.clone()
    } else {
        crate::hf::resolve_alias(&req.model).to_string()
    };
    if base_repo.to_lowercase().contains("xl") || base_repo.to_lowercase().contains("flux") {
        bail!(
            "stylize currently supports SD 1.5 only. Use --model sd15 (or any HF SD-1.5 repo)."
        );
    }

    // Resolve output dims from IN. SD 1.5 expects multiples of 8.
    let (in_w, in_h) = read_image_size(&req.input)?;
    let (w, h) = sd_dims(in_w, in_h);
    let cfg = StableDiffusionConfig::v1_5(None, Some(h as usize), Some(w as usize));
    let dtype = if matches!(req.device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    };
    let strength = req.strength.clamp(0.0, 1.0);

    let mp = progress::multi();

    // -------- download weights --------
    let dl = progress::spinner(&mp, &format!("Downloading SD 1.5 + IP-Adapter weights"));

    let tokenizer_path = crate::hf::download::get_first_of(&[
        (&base_repo, "tokenizer/tokenizer.json"),
        ("openai/clip-vit-large-patch14", "tokenizer.json"),
    ])
    .await
    .with_context(|| format!("tokenizer for {base_repo}"))?;
    let text_enc_path = crate::hf::download::get_first_of(&[
        (&base_repo, "text_encoder/model.fp16.safetensors"),
        (&base_repo, "text_encoder/model.safetensors"),
    ])
    .await?;
    let unet_path = crate::hf::download::get_first_of(&[
        (&base_repo, "unet/diffusion_pytorch_model.fp16.safetensors"),
        (&base_repo, "unet/diffusion_pytorch_model.safetensors"),
    ])
    .await?;
    let vae_path = crate::hf::download::get_first_of(&[
        (&base_repo, "vae/diffusion_pytorch_model.fp16.safetensors"),
        (&base_repo, "vae/diffusion_pytorch_model.safetensors"),
    ])
    .await?;

    let ipa_weights =
        crate::hf::download::get_file(IPA_REPO, "models/ip-adapter_sd15.safetensors").await?;
    let img_enc_weights =
        crate::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors").await?;
    dl.finish_with_message("✓ weights ready");

    // -------- load models --------
    let build = progress::spinner(&mp, "Loading models");
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let text_encoder =
        stable_diffusion::build_clip_transformer(&cfg.clip, &text_enc_path, &req.device, dtype)?;
    let vae = cfg.build_vae(&vae_path, &req.device, dtype)?;
    let unet = cfg.build_unet(&unet_path, &req.device, 4, false, dtype)?;
    let image_encoder = ImageEncoder::load(&img_enc_weights, &req.device, dtype)?;
    let image_proj = ImageProj::load(
        &ipa_weights,
        CLIP_H_PROJ_DIM,
        SD15_CROSS_ATTN_DIM,
        IPA_TOKENS,
        &req.device,
        dtype,
    )?;
    build.finish_with_message("✓ models loaded");

    // -------- encode REF → image tokens --------
    let style = progress::spinner(&mp, "Encoding reference image");
    let ref_pixels =
        crate::imaging::preprocess::clip_image_tensor(&req.reference, CLIP_H_INPUT, &req.device, dtype)?;
    let img_embeds = image_encoder.encode(&ref_pixels)?;
    let image_tokens = image_proj.forward(&img_embeds)?; // (1, 4, 768)
    style.finish_with_message("✓ reference encoded");

    // -------- text embeddings (empty) --------
    let pad_id = tokenizer
        .token_to_id("<|endoftext|>")
        .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?;
    let mut text_ids = tokenizer
        .encode("", true)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    text_ids.resize(cfg.clip.max_position_embeddings, pad_id);
    let text_ids = Tensor::new(text_ids.as_slice(), &req.device)?.unsqueeze(0)?;
    let text_embeds = text_encoder.forward(&text_ids)?.to_dtype(dtype)?;

    // (1, 77, 768) ⊕ (1, 4, 768) → (1, 81, 768)
    let encoder_hidden_states = Tensor::cat(&[&text_embeds, &image_tokens], 1)?;

    // -------- encode IN → latents --------
    let enc_in = progress::spinner(&mp, "Encoding input image");
    let in_pixels = crate::imaging::preprocess::sd_image_tensor(&req.input, w, h, &req.device, dtype)?;
    // VAE encode: stable_diffusion::vae returns a `DiagonalGaussianDistribution`;
    // we sample the mean (deterministic) and scale by 0.18215.
    let init_dist = vae.encode(&in_pixels)?;
    let init_latents = (init_dist.sample()? * 0.18215)?;
    enc_in.finish_with_message("✓ input encoded");

    // -------- img2img denoise --------
    let seed = req.seed.unwrap_or_else(rand::random) & (u32::MAX as u64);
    if let Err(e) = req.device.set_seed(seed) {
        tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
    }

    let mut scheduler = cfg.build_scheduler(req.steps)?;
    let timesteps = scheduler.timesteps().to_vec();

    // img2img: skip the first (1 - strength) fraction of the schedule.
    let init_skip =
        ((req.steps as f32) * (1.0 - strength)).round().max(0.0) as usize;
    let init_skip = init_skip.min(req.steps.saturating_sub(1));
    let active = &timesteps[init_skip..];
    let start_t = *active.first().ok_or_else(|| anyhow!("empty timestep list"))?;

    // Add noise at the starting timestep.
    let shape = init_latents.shape();
    let noise = Tensor::randn(0f32, 1f32, shape, &req.device)?.to_dtype(dtype)?;
    let mut latents = scheduler.add_noise(&init_latents, noise, start_t)?;

    let bar = progress::step_bar(&mp, active.len() as u64, "stylize");
    for &timestep in active {
        let latent_in = scheduler.scale_model_input(latents.clone(), timestep)?;
        let noise_pred = unet.forward(&latent_in, timestep as f64, &encoder_hidden_states)?;
        latents = scheduler.step(&noise_pred, timestep, &latents)?;
        bar.inc(1);
        bar.set_message(format!("t={timestep} strength={strength:.2}"));
    }
    bar.finish_and_clear();

    // -------- decode + save --------
    let image = vae.decode(&(&latents / 0.18215)?)?;
    let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
    let image = (image * 255.0)?.to_dtype(DType::U8)?.i(0)?.permute((1, 2, 0))?;
    let (oh, ow, _) = image.dims3()?;
    let buf = image.flatten_all()?.to_vec1::<u8>()?;
    if let Some(parent) = req.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &req.out)?;
    tracing::info!(target: "plakat", "→ {}", req.out.display());
    Ok(())
}

fn read_image_size(path: &std::path::Path) -> Result<(u32, u32)> {
    let img = image::open(path)?;
    Ok(image::GenericImageView::dimensions(&img))
}

/// Round IN dims to multiples of 8, capped at a sensible SD 1.5 max.
fn sd_dims(in_w: u32, in_h: u32) -> (u32, u32) {
    let cap = 768u32;
    let scale = (cap as f32 / in_w.max(in_h) as f32).min(1.0);
    let w = ((in_w as f32) * scale).round() as u32;
    let h = ((in_h as f32) * scale).round() as u32;
    ((w / 8).max(1) * 8, (h / 8).max(1) * 8)
}
