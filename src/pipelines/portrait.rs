//! Portrait generation pipeline.
//!
//! Supports two SD variants:
//!   * **SD 1.5** with `IdentityKind::PlusFace` — CLIP-H penultimate hidden
//!     state → Perceiver resampler (16 tokens × 768-d) → concat onto
//!     `(1, 77, 768)` text tokens → denoise from noise.
//!   * **SDXL** with `IdentityKind::PlusFaceSdxl` — same CLIP-H encoder
//!     (the `vit-h` SDXL Plus-Face variant), but the Resampler emits at
//!     SDXL's 2048-d cross-attn dim; concat onto SDXL's dual-encoder
//!     `(1, 77, 2048)` text tokens.
//!
//! Without a photo, behaves like a text-only portrait-tuned generate
//! (3:4 aspect default, face/anatomy negatives baked in at the CLI layer).
//!
//! Limitations carried over from `stylize`'s IP-Adapter integration:
//!   * candle 0.8 has no UNet attention hooks, so the *decoupled* cross-
//!     attention path (separate `to_k_ip` / `to_v_ip` per block) is not
//!     wired up. Identity tokens travel via the same cross-attention as
//!     text. Quality is recognisable but not pixel-perfect — typically
//!     ~50–70% of diffusers' reference. FaceID / InstantID (Phase-3+) are
//!     the path to better identity preservation.
//!   * SDXL micro-conditioning (`text_time` add-embedding from pooled
//!     CLIP-G + size/crop time-ids) is not wired up — candle's UNet has
//!     no `add_embedding` projection. Same gap as our base SDXL t2i path.
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

// Re-export so callers (CLI, scenario, future tools) can keep using
// `portrait::IdentityKind` even though the enum lives next to its loaders
// in `ip_adapter`. New strategies are added there, not here.
pub use crate::pipelines::ip_adapter::IdentityKind;

/// SD variant the portrait pipeline routes through. Detected from the
/// `model` alias / repo at load time. SD 1.5 is the default (alias `sd15`);
/// SDXL is selected by any alias / repo containing `xl` (case-insensitive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Sd15,
    Sdxl,
}

impl Variant {
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
        if m.contains("flux") {
            // Caller validates this earlier; treat as SD 1.5 to keep the
            // type total. Flux portraits are not a thing.
            return Self::Sd15;
        }
        if m.contains("xl") {
            Self::Sdxl
        } else {
            Self::Sd15
        }
    }

    pub fn cross_attn_dim(self) -> usize {
        match self {
            Self::Sd15 => 768,
            Self::Sdxl => 2048,
        }
    }

    pub fn vae_scale(self) -> f64 {
        match self {
            Self::Sd15 => 0.18215,
            Self::Sdxl => 0.13025,
        }
    }

    pub fn config(self, w: usize, h: usize) -> StableDiffusionConfig {
        match self {
            Self::Sd15 => StableDiffusionConfig::v1_5(None, Some(h), Some(w)),
            Self::Sdxl => StableDiffusionConfig::sdxl(None, Some(h), Some(w)),
        }
    }
}

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
    variant: Variant,
    cfg: StableDiffusionConfig,
    tokenizer_l: Tokenizer,
    /// SDXL only — the CLIP-G tokenizer + encoder. `None` for SD 1.5.
    tokenizer_g: Option<Tokenizer>,
    text_encoder_l: sdclip::ClipTextTransformer,
    text_encoder_g: Option<sdclip::ClipTextTransformer>,
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
    /// Load weights for SD 1.5 or SDXL based on the model alias / repo.
    /// Flux models are rejected (portrait is a SD-architecture feature).
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        let lc = base_repo.to_lowercase();
        if lc.contains("flux") {
            bail!(
                "portrait does not support Flux. Use --model sd15 (default) \
                 or --model sdxl. Flux portraits would need a separate \
                 identity-adapter family — not yet ported."
            );
        }
        let variant = Variant::detect(&base_repo);

        // Sanity-check the identity strategy against the model variant.
        // Catches `--model sdxl --identity plus-face` (or vice versa)
        // before the model load eats seconds of download time.
        if let Some(kind) = req.identity {
            if kind.cross_attn_dim() != variant.cross_attn_dim() {
                bail!(
                    "identity strategy {:?} targets cross_attn_dim {} but \
                     model {:?} ({:?}) expects {}. Pick an identity that \
                     matches the model: SD 1.5 → `plus-face`, SDXL → \
                     `plus-face-sdxl`.",
                    kind,
                    kind.cross_attn_dim(),
                    base_repo,
                    variant,
                    variant.cross_attn_dim(),
                );
            }
        }

        let cfg = variant.config(512, 512);
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        // -------- download base weights (variant-aware) --------
        let dl = progress::spinner(&format!(
            "Resolving {} weights",
            match variant { Variant::Sd15 => "SD 1.5", Variant::Sdxl => "SDXL" }
        ));
        let tokenizer_l_path = crate::hf::download::get_first_of(&[
            (&base_repo, "tokenizer/tokenizer.json"),
            ("openai/clip-vit-large-patch14", "tokenizer.json"),
        ])
        .await
        .with_context(|| format!("tokenizer (CLIP-L) for {base_repo}"))?;
        let text_enc_l_path = crate::hf::download::get_first_of(&[
            (&base_repo, "text_encoder/model.fp16.safetensors"),
            (&base_repo, "text_encoder/model.safetensors"),
        ])
        .await?;
        let (tokenizer_g_path, text_enc_g_path) = match variant {
            Variant::Sd15 => (None, None),
            Variant::Sdxl => {
                let t = crate::hf::download::get_first_of(&[
                    (&base_repo, "tokenizer_2/tokenizer.json"),
                    ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json"),
                    ("openai/clip-vit-large-patch14", "tokenizer.json"),
                ])
                .await
                .with_context(|| format!("tokenizer (CLIP-G) for {base_repo}"))?;
                let e = crate::hf::download::get_first_of(&[
                    (&base_repo, "text_encoder_2/model.fp16.safetensors"),
                    (&base_repo, "text_encoder_2/model.safetensors"),
                ])
                .await
                .with_context(|| format!("text_encoder_2 in {base_repo}"))?;
                (Some(t), Some(e))
            }
        };
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
        dl.finish_with_message("✓ base weights ready");

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
        let tokenizer_l = Tokenizer::from_file(&tokenizer_l_path)
            .map_err(|e| anyhow!("tokenizer (CLIP-L): {e}"))?;
        let tokenizer_g = match tokenizer_g_path.as_ref() {
            Some(p) => Some(
                Tokenizer::from_file(p).map_err(|e| anyhow!("tokenizer (CLIP-G): {e}"))?,
            ),
            None => None,
        };
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

        // CLIP-L text encoder (with optional LoRA merge).
        let te_l_target = match variant {
            Variant::Sd15 => crate::pipelines::lora::MergeTarget::TE_SD15,
            Variant::Sdxl => crate::pipelines::lora::MergeTarget::TE1_SDXL,
        };
        let effective_te_l_path = if resolved_loras.is_empty() {
            text_enc_l_path.clone()
        } else {
            let spin = progress::spinner(&format!("Merging LoRA into {}", te_l_target.name));
            let tmp = tempfile::Builder::new()
                .prefix("plakat-portrait-te-l-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                &text_enc_l_path,
                tmp.path(),
                &resolved_loras,
                req.lora_scale,
                &req.device,
                te_l_target,
            )?;
            spin.finish_with_message(format!(
                "✓ merged {modified}/{targets} {} LoRA target(s)",
                te_l_target.name
            ));
            let p = tmp.path().to_path_buf();
            lora_tmps.push(tmp);
            p
        };
        let text_encoder_l = stable_diffusion::build_clip_transformer(
            &cfg.clip,
            &effective_te_l_path,
            &req.device,
            dtype,
        )?;

        // SDXL only: CLIP-G text encoder (with optional LoRA merge).
        let text_encoder_g = match variant {
            Variant::Sd15 => None,
            Variant::Sdxl => {
                let cfg_g = cfg
                    .clip2
                    .as_ref()
                    .ok_or_else(|| anyhow!("SDXL config is missing clip2"))?;
                let p = text_enc_g_path
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing text_encoder_2 path"))?;
                let effective_te_g_path = if resolved_loras.is_empty() {
                    p.clone()
                } else {
                    let target = crate::pipelines::lora::MergeTarget::TE2_SDXL;
                    let spin = progress::spinner(&format!("Merging LoRA into {}", target.name));
                    let tmp = tempfile::Builder::new()
                        .prefix("plakat-portrait-te-g-")
                        .suffix(".safetensors")
                        .tempfile()?;
                    let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                        p,
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
                    let path = tmp.path().to_path_buf();
                    lora_tmps.push(tmp);
                    path
                };
                Some(stable_diffusion::build_clip_transformer(
                    cfg_g,
                    &effective_te_g_path,
                    &req.device,
                    dtype,
                )?)
            }
        };

        build.finish_with_message("✓ portrait models loaded");

        // Identity encoder, if requested. The download + module construction
        // is fully contained in `IdentityKind::load_encoder`, so adding a new
        // strategy is an `ip_adapter` edit that this function never has to
        // learn about.
        let (identity_encoder, identity_num_tokens) = if let Some(kind) = req.identity {
            let enc = kind.load_encoder(&req.device, dtype).await?;
            let n = enc.num_tokens();
            (Some(enc), n)
        } else {
            (None, 0)
        };

        Ok(Self {
            variant,
            cfg,
            tokenizer_l,
            tokenizer_g,
            text_encoder_l,
            text_encoder_g,
            vae,
            unet,
            identity_encoder,
            identity_num_tokens,
            device: req.device,
            dtype,
            _lora_tmp: lora_tmps,
        })
    }

    /// Encode text into the form the UNet expects:
    ///   * SD 1.5 — `(1, 77, 768)` from CLIP-L's final hidden state.
    ///   * SDXL   — `(1, 77, 2048)` from `concat(CLIP-L penultimate,
    ///              CLIP-G penultimate)` along the channel dim.
    fn encode_text(&self, text: &str) -> Result<Tensor> {
        match self.variant {
            Variant::Sd15 => self.encode_text_sd15(text),
            Variant::Sdxl => self.encode_text_sdxl(text),
        }
    }

    fn encode_text_sd15(&self, text: &str) -> Result<Tensor> {
        let ids = tokenize_padded(&self.tokenizer_l, &self.cfg.clip, text, &self.device)?;
        Ok(self.text_encoder_l.forward(&ids)?.to_dtype(self.dtype)?)
    }

    fn encode_text_sdxl(&self, text: &str) -> Result<Tensor> {
        let cfg_g = self
            .cfg
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL Pipeline missing clip2 config"))?;
        let tok_g = self
            .tokenizer_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL Pipeline missing tokenizer_g"))?;
        let enc_g = self
            .text_encoder_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL Pipeline missing text_encoder_g"))?;
        let ids_l = tokenize_padded(&self.tokenizer_l, &self.cfg.clip, text, &self.device)?;
        let ids_g = tokenize_padded(tok_g, cfg_g, text, &self.device)?;
        let (_final_l, hidden_l) = self
            .text_encoder_l
            .forward_until_encoder_layer(&ids_l, usize::MAX, -2)?;
        let (_final_g, hidden_g) =
            enc_g.forward_until_encoder_layer(&ids_g, usize::MAX, -2)?;
        Ok(Tensor::cat(&[&hidden_l, &hidden_g], 2)?.to_dtype(self.dtype)?)
    }

    /// Run `req.count` portraits. Reuses loaded weights across calls.
    pub fn generate(&self, req: &GenRequest) -> Result<()> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.device)?;
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        let (w, h) = (req.width as usize, req.height as usize);
        let do_cfg = req.guidance > 1.0;

        let (encoder_hidden_states, has_face) = self.build_encoder_hidden_states(
            &req.prompt,
            &req.negative,
            req.photo.as_deref(),
            req.face_strength,
            do_cfg,
        )?;

        let bsz: usize = 1;
        let latent_h = h / 8;
        let latent_w = w / 8;
        let vae_scale: f64 = self.variant.vae_scale();

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

            let face_tag = if has_face { "+face" } else { "txt" };
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

    // =================================================================
    // Phase-2 multi-persona compositing primitives.
    //
    // The scenario runner orchestrates by calling these in sequence:
    //   let mut latents = pipeline.generate_latents_one(&base_req, seed)?;
    //   for (persona_req, mask) in passes {
    //       latents = pipeline.inpaint_latents_one(&latents, &mask, &persona_req, seed)?;
    //   }
    //   pipeline.save_image(&latents, &out_path)?;
    // =================================================================

    /// Build the encoder-hidden-states tensor for one call. With CFG this
    /// is `(2, 77 + K, 768)` where K = 0 (no face), 4 (plain), or 16
    /// (Plus). Returns the tensor plus a flag indicating whether image
    /// tokens were included (used for progress-bar labels).
    fn build_encoder_hidden_states(
        &self,
        prompt: &str,
        negative: &str,
        photo: Option<&std::path::Path>,
        face_strength: f32,
        do_cfg: bool,
    ) -> Result<(Tensor, bool)> {
        let text_cond = self.encode_text(prompt)?;
        let text_uncond = if do_cfg {
            Some(self.encode_text(negative)?)
        } else {
            None
        };

        let face_strength = face_strength.clamp(0.0, 2.0);
        let identity_tokens = match (&self.identity_encoder, photo) {
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
            (Some(_), None) => None,
            (None, None) => None,
        };

        let cond_full = match &identity_tokens {
            Some(img) => Tensor::cat(&[&text_cond, img], 1)?,
            None => text_cond.clone(),
        };
        let ehs = if do_cfg {
            let uncond_text = text_uncond.as_ref().unwrap();
            let uncond_full = match &identity_tokens {
                Some(img) => {
                    let zero = img.zeros_like()?;
                    Tensor::cat(&[uncond_text, &zero], 1)?
                }
                None => uncond_text.clone(),
            };
            Tensor::cat(&[&uncond_full, &cond_full], 0)?
        } else {
            cond_full
        };
        Ok((ehs, identity_tokens.is_some()))
    }

    /// Generate one sample of latents from text alone (no inpainting).
    /// Used as the base for multi-persona compositing. Skips the polish
    /// pass — orchestrator may run polish on the final composite.
    pub fn generate_latents_one(&self, req: &GenRequest, seed: u64) -> Result<Tensor> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.device)?;
        let (w, h) = (req.width as usize, req.height as usize);
        let do_cfg = req.guidance > 1.0;
        let (ehs, has_face) = self.build_encoder_hidden_states(
            &req.prompt,
            &req.negative,
            req.photo.as_deref(),
            req.face_strength,
            do_cfg,
        )?;

        if let Err(e) = self.device.set_seed(seed) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }
        let mut scheduler =
            crate::pipelines::scheduler::build(req.scheduler, &self.cfg, req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();
        let latent_h = h / 8;
        let latent_w = w / 8;
        let mut latents = Tensor::randn(0f32, 1f32, (1, 4, latent_h, latent_w), &self.device)?
            .to_dtype(self.dtype)?;
        latents = (latents * scheduler.init_noise_sigma())?;

        let face_tag = if has_face { "+face" } else { "txt" };
        let bar = progress::step_bar(
            timesteps.len() as u64,
            &format!("composite-base {face_tag}"),
        );
        for &t in &timesteps {
            latents = self.denoise_step(&latents, t, &ehs, &mut scheduler, req.guidance, do_cfg)?;
            bar.inc(1);
            bar.set_message(format!("t={t} seed={seed}"));
        }
        bar.finish_and_clear();
        Ok(latents)
    }

    /// Inpaint one persona into `base_latents` inside `mask`. Uses
    /// RePaint-style latent blending: at each timestep, the unmasked
    /// region is replaced with a re-noised copy of `base_latents`, so
    /// the denoiser only meaningfully drives the masked region.
    ///
    /// `mask` is `(1, 1, latent_h, latent_w)` at the pipeline's dtype,
    /// values in `[0, 1]` (1 = inpaint here, 0 = preserve base).
    pub fn inpaint_latents_one(
        &self,
        base_latents: &Tensor,
        mask: &Tensor,
        req: &GenRequest,
        seed: u64,
    ) -> Result<Tensor> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.device)?;
        let do_cfg = req.guidance > 1.0;
        let (ehs, has_face) = self.build_encoder_hidden_states(
            &req.prompt,
            &req.negative,
            req.photo.as_deref(),
            req.face_strength,
            do_cfg,
        )?;

        if let Err(e) = self.device.set_seed(seed) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }
        let mut scheduler =
            crate::pipelines::scheduler::build(req.scheduler, &self.cfg, req.steps)?;
        let timesteps = scheduler.timesteps().to_vec();
        let first_t = *timesteps
            .first()
            .ok_or_else(|| anyhow!("inpaint scheduler produced 0 timesteps"))?;

        // Start: re-noise the base at the first timestep. The masked region
        // gets driven by the denoiser; the unmasked region gets re-noised
        // again at each step so the masked region sees a coherent neighbour.
        let initial_noise = Tensor::randn(0f32, 1f32, base_latents.shape(), &self.device)?
            .to_dtype(self.dtype)?;
        let mut latents = scheduler.add_noise(base_latents, initial_noise, first_t)?;

        let inv_mask = (mask.ones_like()? - mask)?;
        let face_tag = if has_face { "+face" } else { "txt" };
        let bar = progress::step_bar(
            timesteps.len() as u64,
            &format!("inpaint {face_tag}"),
        );
        for &t in &timesteps {
            // RePaint: re-noise the BASE (not the running latents) outside
            // the mask. This pins the unmasked region to the base image while
            // letting the denoiser walk freely inside the mask.
            let fresh_noise = Tensor::randn(0f32, 1f32, base_latents.shape(), &self.device)?
                .to_dtype(self.dtype)?;
            let base_noised = scheduler.add_noise(base_latents, fresh_noise, t)?;
            latents = (latents.broadcast_mul(mask)?
                + base_noised.broadcast_mul(&inv_mask)?)?;

            latents = self.denoise_step(&latents, t, &ehs, &mut scheduler, req.guidance, do_cfg)?;
            bar.inc(1);
            bar.set_message(format!("t={t} seed={seed}"));
        }
        bar.finish_and_clear();

        // Final blend: pin unmasked region to the *clean* base latents (no
        // residual noise). The masked region keeps the denoiser's output.
        let composited = (latents.broadcast_mul(mask)?
            + base_latents.broadcast_mul(&inv_mask)?)?;
        Ok(composited)
    }

    /// VAE-decode `latents` and save as PNG at `out_path`.
    pub fn save_image(
        &self,
        latents: &Tensor,
        out_path: &std::path::Path,
    ) -> Result<()> {
        let vae_scale: f64 = self.variant.vae_scale();
        let image = self.vae.decode(&(latents / vae_scale)?)?;
        let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, out_path)?;
        crate::ui::progress::println(&format!("→ {}", out_path.display()));
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

    /// The dtype the pipeline's tensors live at (F16 on accelerators,
    /// F32 on CPU). Callers building masks for `inpaint_latents_one`
    /// need this so the mask matches.
    pub fn latent_dtype(&self) -> DType {
        self.dtype
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

// =====================================================================
// Tokenisation helper — shared by SD 1.5 + SDXL encode paths.
// Mirrors `t2i::tokenize_padded`. Lives here to avoid making t2i's helper
// pub; both modules now have their own copy. Trivial duplication is
// preferable to a third "shared" module just for this.
// =====================================================================
fn tokenize_padded(
    tokenizer: &Tokenizer,
    cfg: &sdclip::Config,
    text: &str,
    device: &Device,
) -> Result<Tensor> {
    let pad_id: u32 = match &cfg.pad_with {
        Some(s) => tokenizer
            .token_to_id(s)
            .ok_or_else(|| anyhow!("tokenizer missing pad token {s:?}"))?,
        None => tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?,
    };
    let mut ids = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    ids.resize(cfg.max_position_embeddings, pad_id);
    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
}
