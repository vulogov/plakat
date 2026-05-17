//! Portrait generation pipeline.
//!
//! Phase 1: IP-Adapter-Plus-Face on Stable Diffusion 1.5.
//!   * Optional reference photo → CLIP-H penultimate hidden state →
//!     Perceiver resampler (16 image tokens, 768-d) → concat onto text
//!     tokens → standard SD denoise from pure noise → VAE decode.
//!   * Without a photo, behaves like a text-only portrait-tuned generate
//!     (still benefits from portrait-specific defaults: 3:4 aspect, baked-in
//!     anatomy negatives, face-friendly scheduler).
//!
//! Phase 2 (planned): plug in `FaceIdEncoder` (InsightFace ArcFace
//! embedding) and `InstantIdEncoder` (ID + landmarks) via the
//! `IdentityEncoder` trait without touching this module's pipeline loop.
//!
//! Limitations carried over from our `stylize` IP-Adapter integration:
//!   * candle 0.8 has no UNet attention hooks, so the *decoupled* cross-
//!     attention path (separate to_k_ip / to_v_ip in every block) is not
//!     wired up. Identity tokens travel via the same cross-attention as
//!     text. Quality is recognisable but not pixel-perfect — typically
//!     ~50–70% of diffusers' reference. Phase 2 (FaceID/InstantID) is
//!     where we'd expect a meaningful jump.
//!   * SD 1.5 only for now. SDXL Plus-Face uses different image-encoder
//!     dims (CLIP-G) and a separate safetensors file; that's a Phase 1.5
//!     follow-up.
//!   * No automatic face crop. Pass a reasonably tight head-and-shoulders
//!     photo for best results.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::{
    self, StableDiffusionConfig, clip as sdclip, unet_2d::UNet2DConditionModel,
    vae::AutoEncoderKL,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::ip_adapter::IdentityEncoder;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;
use crate::ui::progress;

// Re-export so callers (CLI, future scenario integration) can keep using
// `portrait::IdentityKind` even though the enum lives next to its loaders
// in `ip_adapter`. Phase-2 strategies are added there, not here.
pub use crate::pipelines::ip_adapter::IdentityKind;

// =====================================================================
// Request types.
// =====================================================================

/// Single-shot back-compat request (mirrors the t2i::Request shape).
pub struct Request {
    pub prompt: String,
    pub negative: String,
    pub photo: Option<PathBuf>,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    pub scheduler: SchedulerKind,
    pub refine: Option<usize>,
    pub refine_strength: f32,
    pub face_strength: f32,
    /// Which identity strategy to wire up. `None` collapses portrait into a
    /// portrait-tuned text-only generate.
    pub identity: Option<IdentityKind>,
}

pub struct LoadRequest {
    pub model: String,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    /// `Some(kind)` to pre-load the identity encoder. If `None`, the loaded
    /// pipeline can only do text-only portrait generation even if the
    /// caller later passes a `photo`.
    pub identity: Option<IdentityKind>,
}

pub struct GenRequest {
    pub prompt: String,
    pub negative: String,
    pub photo: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    pub scheduler: SchedulerKind,
    pub refine: Option<usize>,
    pub refine_strength: f32,
    /// 0..1 multiplier applied to image-token contribution. Diffusers'
    /// IP-Adapter `set_scale` equivalent — at 1.0 image tokens carry full
    /// weight, at 0.0 they vanish (= text-only).
    pub face_strength: f32,
}

// =====================================================================
// Pipeline.
// =====================================================================

pub struct Pipeline {
    cfg: StableDiffusionConfig,
    tokenizer: Tokenizer,
    text_encoder: sdclip::ClipTextTransformer,
    vae: AutoEncoderKL,
    unet: UNet2DConditionModel,
    identity_encoder: Option<Box<dyn IdentityEncoder>>,
    /// Number of image tokens emitted by `identity_encoder`, when present.
    /// Cached so a zero-tokens tensor for the CFG uncond branch is the
    /// right shape without re-querying the trait.
    identity_num_tokens: usize,
    device: Device,
    dtype: DType,
    /// Kept alive so merged-LoRA safetensors mmaps stay valid for the
    /// pipeline's lifetime.
    _lora_tmp: Vec<tempfile::NamedTempFile>,
}

impl Pipeline {
    /// Phase 1: SD 1.5 only. Errors out early on SDXL / Flux models so the
    /// user sees a clear message rather than a mid-load shape mismatch.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        let lc = base_repo.to_lowercase();
        if lc.contains("xl") || lc.contains("flux") {
            bail!(
                "portrait Phase 1 supports SD 1.5 only. Use --model sd15 \
                 (or any HF SD-1.5 repo). SDXL Plus-Face is on the Phase 1.5 \
                 roadmap; FaceID/InstantID are Phase 2."
            );
        }

        let cfg = StableDiffusionConfig::v1_5(None, Some(512), Some(512));
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        // -------- download base SD 1.5 weights --------
        let dl = progress::spinner("Resolving SD 1.5 weights");
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
        dl.finish_with_message("✓ SD 1.5 weights ready");

        // -------- resolve LoRA files (once) --------
        let mut lora_tmps: Vec<tempfile::NamedTempFile> = Vec::new();
        let resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> = if req.loras.is_empty() {
            Vec::new()
        } else {
            let s = progress::spinner("Resolving LoRA file(s)");
            let mut v = Vec::with_capacity(req.loras.len());
            for spec in &req.loras {
                v.push(spec.resolve().await?);
            }
            s.finish_with_message(format!("✓ resolved {} LoRA file(s)", v.len()));
            v
        };

        // -------- build models --------
        let build = progress::spinner("Loading portrait models");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let vae = cfg.build_vae(&vae_path, &req.device, dtype)?;

        // UNet (with optional LoRA merge).
        let effective_unet_path = if resolved_loras.is_empty() {
            unet_path.clone()
        } else {
            let spin = progress::spinner("Merging LoRA into UNet");
            let tmp = tempfile::Builder::new()
                .prefix("plakat-portrait-unet-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                &unet_path,
                tmp.path(),
                &resolved_loras,
                req.lora_scale,
                &req.device,
                crate::pipelines::lora::MergeTarget::UNET,
            )?;
            spin.finish_with_message(format!(
                "✓ merged {modified}/{targets} UNet LoRA target(s)"
            ));
            let p = tmp.path().to_path_buf();
            lora_tmps.push(tmp);
            p
        };
        let unet = cfg.build_unet(&effective_unet_path, &req.device, 4, false, dtype)?;

        // Text encoder (with optional LoRA merge).
        let effective_te_path = if resolved_loras.is_empty() {
            text_enc_path.clone()
        } else {
            let target = crate::pipelines::lora::MergeTarget::TE_SD15;
            let spin = progress::spinner(&format!("Merging LoRA into {}", target.name));
            let tmp = tempfile::Builder::new()
                .prefix("plakat-portrait-te-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                &text_enc_path,
                tmp.path(),
                &resolved_loras,
                req.lora_scale,
                &req.device,
                target,
            )?;
            spin.finish_with_message(format!(
                "✓ merged {modified}/{targets} {} LoRA target(s)",
                target.name
            ));
            let p = tmp.path().to_path_buf();
            lora_tmps.push(tmp);
            p
        };
        let text_encoder = stable_diffusion::build_clip_transformer(
            &cfg.clip,
            &effective_te_path,
            &req.device,
            dtype,
        )?;

        build.finish_with_message("✓ portrait models loaded");

        // Identity encoder, if requested. The download + module construction
        // is fully contained in `IdentityKind::load_encoder`, so adding a new
        // strategy (FaceID, InstantID, …) is a Phase-2 edit in `ip_adapter`
        // that this function never has to learn about.
        let (identity_encoder, identity_num_tokens) = if let Some(kind) = req.identity {
            // Sanity: SD 1.5 UNet has cross_attn_dim 768. Refuse a strategy
            // whose tokens won't fit, even though Phase 1 only ships one.
            if kind.cross_attn_dim() != 768 {
                bail!(
                    "identity {:?} targets cross_attn_dim {} but SD 1.5 UNet expects 768",
                    kind,
                    kind.cross_attn_dim()
                );
            }
            let enc = kind.load_encoder(&req.device, dtype).await?;
            let n = enc.num_tokens();
            (Some(enc), n)
        } else {
            (None, 0)
        };

        Ok(Self {
            cfg,
            tokenizer,
            text_encoder,
            vae,
            unet,
            identity_encoder,
            identity_num_tokens,
            device: req.device,
            dtype,
            _lora_tmp: lora_tmps,
        })
    }

    /// Encode text → `(1, 77, 768)` at the pipeline's dtype.
    fn encode_text(&self, text: &str) -> Result<Tensor> {
        let pad_id = self
            .tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?;
        let mut ids = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        ids.resize(self.cfg.clip.max_position_embeddings, pad_id);
        let ids_t = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        Ok(self.text_encoder.forward(&ids_t)?.to_dtype(self.dtype)?)
    }

    /// Run `req.count` portraits. Reuses loaded weights across calls.
    pub fn generate(&self, req: &GenRequest) -> Result<()> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.device)?;
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        let (w, h) = (req.width as usize, req.height as usize);
        let do_cfg = req.guidance > 1.0;

        // Encode text (and uncond text for CFG). (1, 77, 768) each.
        let text_cond = self.encode_text(&req.prompt)?;
        let text_uncond = if do_cfg {
            Some(self.encode_text(&req.negative)?)
        } else {
            None
        };

        // Optionally encode identity tokens.
        let face_strength = req.face_strength.clamp(0.0, 2.0);
        let identity_tokens = match (&self.identity_encoder, req.photo.as_ref()) {
            (Some(enc), Some(p)) => {
                let s = progress::spinner("Encoding reference photo");
                let tok = enc.encode(p)?;
                let tok = (tok * (face_strength as f64))?.to_dtype(self.dtype)?;
                s.finish_with_message("✓ identity encoded");
                Some(tok)
            }
            (None, Some(_)) => {
                bail!(
                    "this Pipeline was loaded without an identity encoder \
                     but a photo was provided. Reload with `identity: Some(IdentityKind::PlusFace)`."
                );
            }
            (Some(_), None) => None, // identity loaded but caller chose text-only
            (None, None) => None,
        };

        // Build the final encoder_hidden_states. With CFG that's
        // (2, 77+K, 768): row 0 = uncond text + zero image tokens, row 1 =
        // cond text + scaled image tokens. Without CFG just the cond row.
        let cond_full = match &identity_tokens {
            Some(img) => Tensor::cat(&[&text_cond, img], 1)?,
            None => text_cond.clone(),
        };
        let encoder_hidden_states = if do_cfg {
            let uncond_text = text_uncond.as_ref().unwrap();
            let uncond_full = match &identity_tokens {
                Some(img) => {
                    // Zero image tokens for the uncond branch — standard
                    // IP-Adapter CFG setup. Allocate at the right dtype.
                    let zero = img.zeros_like()?;
                    Tensor::cat(&[uncond_text, &zero], 1)?
                }
                None => uncond_text.clone(),
            };
            Tensor::cat(&[&uncond_full, &cond_full], 0)?
        } else {
            cond_full
        };

        let bsz: usize = 1;
        let latent_h = h / 8;
        let latent_w = w / 8;
        let vae_scale: f64 = 0.18215;

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random)
                & (u32::MAX as u64);
            if let Err(e) = self.device.set_seed(seed) {
                tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
            }

            let mut scheduler =
                crate::pipelines::scheduler::build(req.scheduler, &self.cfg, req.steps)?;
            let timesteps = scheduler.timesteps().to_vec();

            let mut latents =
                Tensor::randn(0f32, 1f32, (bsz, 4, latent_h, latent_w), &self.device)?
                    .to_dtype(self.dtype)?;
            latents = (latents * scheduler.init_noise_sigma())?;

            let face_tag = if identity_tokens.is_some() { "+face" } else { "txt" };
            let bar = progress::step_bar(
                timesteps.len() as u64,
                &format!("portrait {}/{} {}", idx + 1, req.count, face_tag),
            );
            for &timestep in &timesteps {
                latents = self.denoise_step(
                    &latents,
                    timestep,
                    &encoder_hidden_states,
                    &mut scheduler,
                    req.guidance,
                    do_cfg,
                )?;
                bar.inc(1);
                bar.set_message(format!("t={timestep} seed={seed}"));
            }
            bar.finish_and_clear();

            // Optional same-model polish pass.
            if let Some(rsteps) = req.refine {
                if rsteps > 0 {
                    let strength = req.refine_strength.clamp(0.0, 1.0);
                    let mut polish =
                        crate::pipelines::scheduler::build(req.scheduler, &self.cfg, rsteps)?;
                    let pts = polish.timesteps().to_vec();
                    let init_skip = ((rsteps as f32) * (1.0 - strength)).round() as usize;
                    let init_skip = init_skip.min(rsteps.saturating_sub(1));
                    let active = &pts[init_skip..];
                    if let Some(&start_t) = active.first() {
                        let noise = Tensor::randn(0f32, 1f32, latents.shape(), &self.device)?
                            .to_dtype(self.dtype)?;
                        latents = polish.add_noise(&latents, noise, start_t)?;
                        let rbar = progress::step_bar(
                            active.len() as u64,
                            &format!("polish {}/{}", idx + 1, req.count),
                        );
                        for &timestep in active {
                            latents = self.denoise_step(
                                &latents,
                                timestep,
                                &encoder_hidden_states,
                                &mut polish,
                                req.guidance,
                                do_cfg,
                            )?;
                            rbar.inc(1);
                            rbar.set_message(format!("polish t={timestep}"));
                        }
                        rbar.finish_and_clear();
                    }
                }
            }

            // Decode + save.
            let image = self.vae.decode(&(&latents / vae_scale)?)?;
            let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
            let image = (image * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = image.dims3()?;
            let buf = image.flatten_all()?.to_vec1::<u8>()?;
            let out_path = req.out_dir.join(format!("plakat-portrait-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
        }
        Ok(())
    }

    fn denoise_step(
        &self,
        latents: &Tensor,
        timestep: usize,
        encoder_hidden_states: &Tensor,
        scheduler: &mut Box<dyn stable_diffusion::schedulers::Scheduler>,
        guidance: f64,
        do_cfg: bool,
    ) -> Result<Tensor> {
        let latent_in = if do_cfg {
            Tensor::cat(&[latents, latents], 0)?
        } else {
            latents.clone()
        };
        let latent_in = scheduler.scale_model_input(latent_in, timestep)?;
        let noise_pred = self
            .unet
            .forward(&latent_in, timestep as f64, encoder_hidden_states)?;
        let noise_pred = if do_cfg {
            let chunks = noise_pred.chunk(2, 0)?;
            let uncond = &chunks[0];
            let text = &chunks[1];
            (uncond + ((text - uncond)? * guidance)?)?
        } else {
            noise_pred
        };
        Ok(scheduler.step(&noise_pred, timestep, latents)?)
    }
}

/// Suppress dead-code warning on the cached token count until something
/// else queries it (debug logging, etc.). Kept on the struct so future
/// callers don't have to thread a separate value.
impl Pipeline {
    #[allow(dead_code)]
    pub fn identity_num_tokens(&self) -> usize {
        self.identity_num_tokens
    }
}

// =====================================================================
// Single-shot entry — what `plakat portrait` calls.
// =====================================================================

pub async fn run(req: Request) -> Result<()> {
    let pipeline = Pipeline::load(LoadRequest {
        model: req.model,
        device: req.device,
        loras: req.loras,
        lora_scale: req.lora_scale,
        identity: req.identity,
    })
    .await?;

    pipeline.generate(&GenRequest {
        prompt: req.prompt,
        negative: req.negative,
        photo: req.photo,
        width: req.width,
        height: req.height,
        count: req.count,
        steps: req.steps,
        guidance: req.guidance,
        seed: req.seed,
        out_dir: req.out_dir,
        scheduler: req.scheduler,
        refine: req.refine,
        refine_strength: req.refine_strength,
        face_strength: req.face_strength,
    })
}
