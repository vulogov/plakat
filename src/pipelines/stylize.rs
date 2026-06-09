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
//!
//! Two ways to use this module:
//!   * `stylize::run(Request)` — single-shot. `plakat stylize` uses this.
//!   * `Pipeline::load(...)` + repeated `Pipeline::stylize_one(...)` —
//!     share loaded weights (notably the 2.5 GB CLIP-H image encoder)
//!     across many calls. `plakat scenario` uses this when tasks declare
//!     a `style` reference.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::{
    StableDiffusionConfig, unet_2d::UNet2DConditionModel, vae::AutoEncoderKL,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::ip_adapter::{ImageEncoder, ImageProj};
use crate::ui::progress;

// =====================================================================
// Single-shot request type — back-compat with the CLI subcommand.
// =====================================================================

pub struct Request {
    pub input: PathBuf,
    pub reference: PathBuf,
    pub out: PathBuf,
    pub strength: f32,
    pub model: String,
    pub steps: usize,
    pub seed: Option<u64>,
    pub ref_blur: f32,
    pub ref_weight: f32,
    pub device: Device,
}

const IPA_REPO: &str = "h94/IP-Adapter";
const SD15_CROSS_ATTN_DIM: usize = 768;
const IPA_TOKENS: usize = 4;
const CLIP_H_PROJ_DIM: usize = 1024;
const CLIP_H_INPUT: u32 = 224;

// =====================================================================
// Pipeline: load once, stylize many.
// =====================================================================

pub struct LoadRequest {
    pub model: String,
    pub device: Device,
    /// Phase 7f. Optional pre-loaded CLIP-H image encoder to share
    /// with `portrait::Pipeline`'s identity encoder. `None` causes
    /// stylize to download + load CLIP-H itself (pre-7f behaviour).
    pub shared_clip_h: Option<std::sync::Arc<ImageEncoder>>,
}

pub struct GenRequest {
    pub input: PathBuf,
    pub reference: PathBuf,
    pub out: PathBuf,
    pub strength: f32,
    pub steps: usize,
    pub seed: Option<u64>,
    /// Gaussian-blur the reference before CLIP-encoding it (sigma; 0 = off).
    /// Blurring wipes the ref's fine content (the subject/face) while keeping
    /// its broad style — palette, texture, composition — so the transfer is
    /// *style*, not subject. The cheap "style not content" knob.
    pub ref_blur: f32,
    /// Scale the reference's image-token contribution (1.0 = full). Lower lets
    /// the prompt own the subject while the ref owns the look.
    pub ref_weight: f32,
}

/// Blur the reference to a temp PNG when `sigma > 0` (the style-not-content
/// heuristic), else return the original path. Normalises the short side to
/// 512 first so `sigma` means the same thing at any reference resolution.
fn maybe_blur_ref(path: &std::path::Path, sigma: f32) -> Result<std::path::PathBuf> {
    if sigma <= 0.0 {
        return Ok(path.to_path_buf());
    }
    let img = image::open(path)
        .with_context(|| format!("opening reference {} for blur", path.display()))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let (rw, rh) = if w < h {
        (512, ((h as f32) * 512.0 / (w as f32)).round() as u32)
    } else {
        (((w as f32) * 512.0 / (h as f32)).round() as u32, 512)
    };
    let resized = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Triangle);
    let blurred = image::imageops::blur(&resized, sigma);
    let tmp = std::env::temp_dir().join(format!("plakat-stylize-ref-{}.png", std::process::id()));
    blurred
        .save(&tmp)
        .with_context(|| format!("writing blurred reference {}", tmp.display()))?;
    Ok(tmp)
}

pub struct Pipeline {
    cfg: StableDiffusionConfig,
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    text_encoder: crate::pipelines::vendored_clip::ClipTextTransformer,
    vae: AutoEncoderKL,
    unet: UNet2DConditionModel,
    /// Phase 7f: `Arc` so the same CLIP-H weights can back both this
    /// pipeline and portrait's identity encoder when both run in one
    /// process.
    image_encoder: std::sync::Arc<ImageEncoder>,
    image_proj: ImageProj,
    /// Pre-computed empty-prompt text embeddings (1, 77, 768) at this
    /// pipeline's dtype. Same across every stylize call — cached so we
    /// don't re-run the text encoder for an empty prompt every time.
    empty_text_embeds: Tensor,
    device: Device,
    dtype: DType,
}

impl Pipeline {
    /// Download + load SD 1.5 base + IP-Adapter (image encoder + projection).
    /// First run downloads ~2.5 GB of CLIP-H weights plus SD 1.5 base.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        if base_repo.to_lowercase().contains("xl") || base_repo.to_lowercase().contains("flux") {
            bail!(
                "stylize currently supports SD 1.5 only. Use --model sd15 \
                 (or any HF SD-1.5 repo)."
            );
        }

        // Placeholder dims — not baked into model weights, only stored in cfg.
        let cfg = StableDiffusionConfig::v1_5(None, Some(512), Some(512));
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        // -------- download weights --------
        let dl = progress::spinner("Downloading SD 1.5 + IP-Adapter weights");
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
        // Phase 7f: skip the CLIP-H download entirely when the caller
        // supplied a pre-loaded encoder.
        let img_enc_weights = if req.shared_clip_h.is_none() {
            Some(
                crate::hf::download::get_file(
                    IPA_REPO,
                    "models/image_encoder/model.safetensors",
                )
                .await?,
            )
        } else {
            None
        };
        dl.finish_with_message("✓ weights ready");

        // -------- build models --------
        let build = progress::spinner("Loading stylize models");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
        // v0.32 phase 1: vendored CLIP-L. Same numerics as
        // `cfg.clip` for SD 1.5; built via plakat's vendored module
        // to match SdCore / AnimateDiff / SD3 / Flux.
        let clip_l_cfg = crate::pipelines::vendored_clip::Config::v1_5();
        let text_encoder = crate::pipelines::vendored_clip::build_clip_transformer(
            &clip_l_cfg,
            &text_enc_path,
            &req.device,
            dtype,
        )?;
        let vae = cfg.build_vae(&vae_path, &req.device, dtype)?;
        let unet = cfg.build_unet(&unet_path, &req.device, 4, false, dtype)?;
        let image_encoder = match req.shared_clip_h {
            Some(shared) => shared,
            None => std::sync::Arc::new(ImageEncoder::load(
                img_enc_weights
                    .as_ref()
                    .expect("img_enc_weights set when shared_clip_h is None"),
                &req.device,
                dtype,
            )?),
        };
        let image_proj = ImageProj::load(
            &ipa_weights,
            CLIP_H_PROJ_DIM,
            SD15_CROSS_ATTN_DIM,
            IPA_TOKENS,
            &req.device,
            dtype,
        )?;
        // Pre-compute empty-text embeddings — constant per pipeline.
        let pad_id = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?;
        let mut ids = tokenizer
            .encode("", true)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        ids.resize(cfg.clip.max_position_embeddings, pad_id);
        let ids_t = Tensor::new(ids.as_slice(), &req.device)?.unsqueeze(0)?;
        let empty_text_embeds = text_encoder.forward(&ids_t)?.to_dtype(dtype)?;
        build.finish_with_message("✓ stylize models loaded");

        Ok(Self {
            cfg,
            tokenizer,
            text_encoder,
            vae,
            unet,
            image_encoder,
            image_proj,
            empty_text_embeds,
            device: req.device,
            dtype,
        })
    }

    /// Apply one IN + REF → OUT stylization using the loaded models.
    pub fn stylize_one(&self, req: &GenRequest) -> Result<()> {
        // Resolve output dims from IN. SD 1.5 expects multiples of 8.
        let (in_w, in_h) = read_image_size(&req.input)?;
        let (w, h) = sd_dims(in_w, in_h);
        let strength = req.strength.clamp(0.0, 1.0);

        // -------- encode REF → image tokens --------
        let s = progress::spinner("Encoding reference image");
        // Cheap "style not content" heuristic: blur the reference first so
        // CLIP sees its broad style (palette/texture/composition), not the
        // fine content that otherwise hijacks the subject.
        let ref_for_clip = maybe_blur_ref(&req.reference, req.ref_blur)?;
        let ref_pixels = crate::imaging::preprocess::clip_image_tensor(
            &ref_for_clip,
            CLIP_H_INPUT,
            &self.device,
            self.dtype,
        )?;
        let img_embeds = self.image_encoder.encode(&ref_pixels)?;
        let mut image_tokens = self.image_proj.forward(&img_embeds)?; // (1, 4, 768)
        if (req.ref_weight - 1.0).abs() > f32::EPSILON {
            image_tokens = (image_tokens * req.ref_weight as f64)?;
        }
        s.finish_with_message("✓ reference encoded");

        // (1, 77, 768) ⊕ (1, 4, 768) → (1, 81, 768)
        let encoder_hidden_states = Tensor::cat(&[&self.empty_text_embeds, &image_tokens], 1)?;

        // v0.34 phase 1 fix: seed the device RNG BEFORE VAE encode.
        // `init_dist.sample()` below is RNG-touching — pre-v0.34
        // it used leftover state from prior ops, ignoring --seed.
        // Also: device-aware seed prep replaces the old `& u32::MAX`
        // mask. CPU/CUDA now get full u64 entropy; Metal high seeds
        // hash through SplitMix64 instead of colliding to low bits.
        let seed = req.seed.unwrap_or_else(rand::random);
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
        if let Err(e) = self.device.set_seed(prepared) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }

        // -------- encode IN → latents --------
        let s = progress::spinner("Encoding input image");
        let in_pixels = crate::imaging::preprocess::sd_image_tensor(
            &req.input,
            w,
            h,
            &self.device,
            self.dtype,
        )?;
        let init_dist = self.vae.encode(&in_pixels)?;
        let init_latents = (init_dist.sample()? * 0.18215)?;
        s.finish_with_message("✓ input encoded");

        // -------- img2img denoise --------

        let mut scheduler = self.cfg.build_scheduler(req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();
        let init_skip = ((req.steps as f32) * (1.0 - strength)).round().max(0.0) as usize;
        let init_skip = init_skip.min(req.steps.saturating_sub(1));
        let active = &timesteps[init_skip..];
        let start_t = *active.first().ok_or_else(|| anyhow!("empty timestep list"))?;

        let noise = Tensor::randn(0f32, 1f32, init_latents.shape(), &self.device)?
            .to_dtype(self.dtype)?;
        let mut latents = scheduler.add_noise(&init_latents, noise, start_t)?;

        let bar = progress::step_bar(active.len() as u64, "stylize");
        for &timestep in active {
            let latent_in = scheduler.scale_model_input(latents.clone(), timestep)?;
            let noise_pred =
                self.unet.forward(&latent_in, timestep as f64, &encoder_hidden_states)?;
            latents = scheduler.step(&noise_pred, timestep, &latents)?;
            bar.inc(1);
            bar.set_message(format!("t={timestep} strength={strength:.2}"));
        }
        bar.finish_and_clear();

        // -------- decode + save --------
        let image = self.vae.decode(&(&latents / 0.18215)?)?;
        let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;
        if let Some(parent) = req.out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &req.out)?;
        crate::ui::progress::println(&format!("→ {}", req.out.display()));
        Ok(())
    }
}

// =====================================================================
// Single-shot entry — preserves the existing `plakat stylize` API.
// =====================================================================

pub async fn run(req: Request) -> Result<()> {
    let p = Pipeline::load(LoadRequest {
        model: req.model,
        device: req.device,
        shared_clip_h: None,
    })
    .await?;
    p.stylize_one(&GenRequest {
        input: req.input,
        reference: req.reference,
        out: req.out,
        strength: req.strength,
        steps: req.steps,
        seed: req.seed,
        ref_blur: req.ref_blur,
        ref_weight: req.ref_weight,
    })
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
