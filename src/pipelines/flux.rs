//! FLUX.1-{schnell,dev} text-to-image pipeline.
//!
//! Architecture (per BFL): T5-XXL + CLIP-L text encoders → rectified-flow
//! transformer (DiT) → autoencoder. Uses candle's `flux::*` modules.
//!
//! Weight layout used here (BFL-native + diffusers text encoders):
//!   * `flux1-{schnell,dev}.safetensors`              transformer single file
//!   * `ae.safetensors`                               BFL-native VAE
//!   * `text_encoder/model.safetensors`               CLIP-L
//!   * `text_encoder_2/model-{1,2}-of-2.safetensors`  T5-XXL (sharded)
//!   * `tokenizer/`, `tokenizer_2/`                   tokenizers
//!
//! Resource notes:
//!   * Total weights ≈ 33 GB fp16. Fits comfortably on 24+ GB GPUs / Apple
//!     unified memory; will swap on 16 GB.
//!   * Schnell: 4 steps, no guidance (guidance_embed=false). Dev: 20–50 steps,
//!     guidance ≈ 3.5, gated repo (HF_TOKEN required).

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_core::Module;
use candle_nn::VarBuilder;
use candle_transformers::models::{
    flux::{autoencoder as fae, model as fmodel, sampling},
    stable_diffusion::clip as sdclip,
    t5,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::ui::progress;

const CLIP_EOT: u32 = 49407;

#[derive(Clone, Copy, Debug)]
pub enum Variant {
    Schnell,
    Dev,
}

impl Variant {
    fn is_dev(self) -> bool {
        matches!(self, Self::Dev)
    }
    fn main_filename(self) -> &'static str {
        match self {
            Self::Schnell => "flux1-schnell.safetensors",
            Self::Dev => "flux1-dev.safetensors",
        }
    }
    fn t5_seq_len(self) -> usize {
        match self {
            Self::Schnell => 256,
            Self::Dev => 512,
        }
    }
    fn flux_config(self) -> fmodel::Config {
        match self {
            Self::Schnell => fmodel::Config::schnell(),
            Self::Dev => fmodel::Config::dev(),
        }
    }
    fn ae_config(self) -> fae::Config {
        match self {
            Self::Schnell => fae::Config::schnell(),
            Self::Dev => fae::Config::dev(),
        }
    }
    fn default_guidance(self) -> f64 {
        match self {
            Self::Schnell => 1.0,
            Self::Dev => 3.5,
        }
    }
    fn default_steps(self) -> usize {
        match self {
            Self::Schnell => 4,
            Self::Dev => 28,
        }
    }
}

pub struct Request {
    pub prompt: String,
    pub variant: Variant,
    pub repo: String,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    pub device: Device,
}

pub async fn run(req: Request) -> Result<()> {
    let dtype = if matches!(req.device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    };
    let steps = req.steps.unwrap_or_else(|| req.variant.default_steps());
    let guidance = req.guidance.unwrap_or_else(|| req.variant.default_guidance());
    let w = (req.width as usize / 16) * 16;
    let h = (req.height as usize / 16) * 16;
    if w == 0 || h == 0 {
        bail!("Flux requires width and height divisible by 16, both ≥ 16");
    }


    // ---------- download weights ----------
    let dl = progress::spinner(&format!("Downloading weights for {}", req.repo));
    let main_path = crate::hf::download::get_file(&req.repo, req.variant.main_filename())
        .await
        .with_context(|| format!("{}", req.variant.main_filename()))?;
    let ae_path = crate::hf::download::get_file(&req.repo, "ae.safetensors").await?;

    let clip_weights = crate::hf::download::get_first_of(&[
        (&req.repo, "text_encoder/model.fp16.safetensors"),
        (&req.repo, "text_encoder/model.safetensors"),
    ])
    .await?;
    let clip_tokenizer = crate::hf::download::get_first_of(&[
        (&req.repo, "tokenizer/tokenizer.json"),
        ("openai/clip-vit-large-patch14", "tokenizer.json"),
    ])
    .await?;

    let t5_shard1 = crate::hf::download::get_file(
        &req.repo,
        "text_encoder_2/model-00001-of-00002.safetensors",
    )
    .await?;
    let t5_shard2 = crate::hf::download::get_file(
        &req.repo,
        "text_encoder_2/model-00002-of-00002.safetensors",
    )
    .await?;
    let t5_config_path =
        crate::hf::download::get_file(&req.repo, "text_encoder_2/config.json").await?;
    let t5_tokenizer = crate::hf::download::get_file(&req.repo, "tokenizer_2/tokenizer.json")
        .await?;
    dl.finish_with_message("✓ weights ready");

    // ---------- load text encoders ----------
    let build = progress::spinner("Loading text encoders");

    let clip_cfg = sdclip::Config::v1_5(); // CLIP-L
    let clip_text =
        candle_transformers::models::stable_diffusion::build_clip_transformer(
            &clip_cfg,
            &clip_weights,
            &req.device,
            dtype,
        )?;
    let clip_tok = Tokenizer::from_file(&clip_tokenizer)
        .map_err(|e| anyhow!("CLIP tokenizer: {e}"))?;

    let t5_cfg_str = std::fs::read_to_string(&t5_config_path)?;
    let t5_cfg: t5::Config = serde_json::from_str(&t5_cfg_str)
        .with_context(|| format!("parse T5 config from {}", t5_config_path.display()))?;
    let t5_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&t5_shard1, &t5_shard2], dtype, &req.device)?
    };
    let mut t5_enc = t5::T5EncoderModel::load(t5_vb, &t5_cfg)?;
    let t5_tok = Tokenizer::from_file(&t5_tokenizer)
        .map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
    build.finish_with_message("✓ text encoders ready");

    // ---------- encode prompt ----------
    let enc = progress::spinner("Encoding prompt");

    // CLIP-L: tokenize to 77, run, pool at EOT (highest token id position).
    let mut clip_ids = clip_tok
        .encode(req.prompt.as_str(), true)
        .map_err(|e| anyhow!("CLIP encode: {e}"))?
        .get_ids()
        .to_vec();
    clip_ids.resize(clip_cfg.max_position_embeddings, CLIP_EOT);
    let clip_eot_pos = clip_ids.iter().position(|&t| t == CLIP_EOT).unwrap_or(0);
    let clip_ids_t = Tensor::new(clip_ids.as_slice(), &req.device)?.unsqueeze(0)?;
    let clip_seq = clip_text.forward(&clip_ids_t)?;
    let clip_pooled = clip_seq.i((.., clip_eot_pos, ..))?.to_dtype(dtype)?; // (1, 768)

    // T5: tokenize to variant.t5_seq_len(), pad with id 0, run encoder.
    let t5_seq_len = req.variant.t5_seq_len();
    let mut t5_ids = t5_tok
        .encode(req.prompt.as_str(), true)
        .map_err(|e| anyhow!("T5 encode: {e}"))?
        .get_ids()
        .to_vec();
    t5_ids.truncate(t5_seq_len);
    t5_ids.resize(t5_seq_len, 0);
    let t5_ids_t = Tensor::new(t5_ids.as_slice(), &req.device)?.unsqueeze(0)?;
    let t5_emb = t5_enc.forward(&t5_ids_t)?.to_dtype(dtype)?; // (1, seq, 4096)
    enc.finish_with_message("✓ prompt encoded");

    // ---------- load flux + ae ----------
    let load = progress::spinner("Loading transformer + autoencoder");
    let flux_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&main_path], dtype, &req.device)? };
    let flux_model = fmodel::Flux::new(&req.variant.flux_config(), flux_vb)?;
    let ae_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&ae_path], dtype, &req.device)? };
    let ae_model = fae::AutoEncoder::new(&req.variant.ae_config(), ae_vb)?;
    load.finish_with_message("✓ models loaded");

    // ---------- sample ----------
    std::fs::create_dir_all(&req.out_dir)?;
    let ae_cfg = req.variant.ae_config();
    let lat_h = (h + 15) / 16;
    let lat_w = (w + 15) / 16;
    let image_seq_len = lat_h * lat_w;

    for idx in 0..req.count {
        let seed = req
            .seed
            .map(|s| s + idx as u64)
            .unwrap_or_else(rand::random)
            & (u32::MAX as u64);
        if let Err(e) = req.device.set_seed(seed) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }

        let img = sampling::get_noise(1, h, w, &req.device)?.to_dtype(dtype)?;
        let state = sampling::State::new(&t5_emb, &clip_pooled, &img)?;

        let shift = if req.variant.is_dev() {
            Some((image_seq_len, 0.5_f64, 1.15_f64))
        } else {
            None
        };
        let timesteps = sampling::get_schedule(steps, shift);

        let bar = progress::step_bar(
            (timesteps.len().saturating_sub(1)) as u64,
            &format!("img {}/{}", idx + 1, req.count),
        );
        // candle's `denoise` runs the whole loop without per-step callbacks,
        // so the bar reflects "started/finished" rather than per-step ticks.
        bar.set_message(format!("flow-match denoise, {} steps, seed={seed}", steps));
        let denoised = sampling::denoise(
            &flux_model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            guidance,
        )?;
        bar.set_position(timesteps.len().saturating_sub(1) as u64);
        bar.finish_with_message("✓ denoised");

        // Un-pack the packed (b, h*w, c*4) latents into (b, c, h, w).
        let unpacked = sampling::unpack(&denoised, h, w)?;
        // BFL AE expects: x = decode((z / scale) + shift)
        let pre_decode = ((&unpacked / ae_cfg.scale_factor)? + ae_cfg.shift_factor)?;
        let decoded = ae_model.decode(&pre_decode)?;
        let img_norm = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 0.5)?;
        let img_u8 = (img_norm * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = img_u8.dims3()?;
        let buf = img_u8.flatten_all()?.to_vec1::<u8>()?;

        let out_path = req.out_dir.join(format!("plakat-flux-{seed}.png"));
        crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
        tracing::info!(target: "plakat", "→ {}", out_path.display());
    }
    Ok(())
}
