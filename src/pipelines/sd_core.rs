//! Shared SD-family backbone — the UNet, VAE, text encoder(s), and
//! associated config that every SD-based plakat pipeline (`t2i`,
//! `portrait`, `stylize`, `img2img`, `artefact_blend`) needs.
//!
//! Status: phase 7a of the v0.10 shared-pipeline refactor. Defines
//! the [`SdCore`] type and an async [`load`](SdCore::load)
//! constructor that mirrors the SD-loading half of
//! `portrait::Pipeline::load`. **portrait.rs and t2i.rs do not use
//! this module yet** — that's phase 7b/7c.
//!
//! The duplication between this file and `portrait::Pipeline::load`
//! is intentional and temporary. Phase 7b will rewrite
//! `portrait::Pipeline::load` to delegate to `SdCore::load`, then
//! the duplication collapses.
//!
//! # What lives in `SdCore`
//!
//! Everything that's identical across SD-based pipelines:
//!
//! * `variant` — SD 1.5 vs SDXL (detected from the model id).
//! * `cfg` — candle's `StableDiffusionConfig` for the variant.
//! * `tokenizer_l` + `text_encoder_l` — CLIP-L (used by both SD 1.5 and SDXL).
//! * `tokenizer_g` + `text_encoder_g` — CLIP-G (SDXL only; `None` for SD 1.5).
//! * `vae` — AutoEncoder-KL for VAE encode/decode.
//! * `unet` — the noise-prediction UNet (with any user LoRAs merged in).
//! * `device`, `dtype` — F16 on GPU, F32 on CPU.
//! * `_lora_tmp` — temp-file handles keeping merged LoRA mmaps alive.
//!
//! # What does **not** live here
//!
//! Task-specific add-ons stay with the pipeline that owns them:
//!
//! * `portrait::Pipeline.identity_encoder` (CLIP-H / FaceID for IP-Adapter)
//! * `t2i::Pipeline.refiner_unet` (SDXL refiner)
//! * `stylize::Pipeline.image_encoder` (CLIP-H for style transfer)
//!
//! Those modules will hold an `Arc<SdCore>` plus their own
//! task-specific fields after phases 7b/7c.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device};
use candle_transformers::models::stable_diffusion::{
    self, StableDiffusionConfig, clip as sdclip, unet_2d::UNet2DConditionModel,
    vae::AutoEncoderKL,
};
use tokenizers::Tokenizer;

use crate::pipelines::lora::ResolvedLora;
use crate::ui::progress;

/// SD variant the backbone routes through. Detected from the model
/// alias / repo at load time. Mirrors `portrait::Variant` (and
/// `t2i::Variant` modulo Flux) — kept separate so this module
/// stays free of portrait-specific concerns.
///
/// Flux is **not** supported by `SdCore`. Flux's pipeline has a
/// different architecture (transformer + T5, not UNet + CLIP) and
/// stays in `pipelines::flux`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdVariant {
    Sd15,
    Sdxl,
}

impl SdVariant {
    /// Same heuristic as `portrait::Variant::detect`: any "xl" in
    /// the model name means SDXL, otherwise SD 1.5.
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
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

/// Load-time inputs for the SD core. Identity / refiner / image
/// encoder / etc. live on the task-specific wrappers, not here.
///
/// `loras` is already resolved (paths + scales + display names);
/// the caller decides where they came from. portrait::Pipeline
/// uses this to inject FaceID's auto-LoRA before resolution so
/// the SD core merges it transparently.
pub struct SdLoadRequest {
    pub model: String,
    pub device: Device,
    pub loras: Vec<ResolvedLora>,
    pub lora_scale: f32,
}

/// The shared SD backbone. Held behind `Arc` by every task-specific
/// pipeline that consumes it — letting `plakat generate
/// --artefact-blend` load weights once and reuse them across the
/// base generation pass + the blend pass.
///
/// Fields are public-in-crate so the wrapping pipelines (in
/// phases 7b/7c) can call methods directly on the underlying
/// candle objects rather than going through accessor methods.
pub struct SdCore {
    pub variant: SdVariant,
    pub cfg: StableDiffusionConfig,
    pub tokenizer_l: Tokenizer,
    /// SDXL only — the CLIP-G tokenizer. `None` for SD 1.5.
    pub tokenizer_g: Option<Tokenizer>,
    pub text_encoder_l: sdclip::ClipTextTransformer,
    pub text_encoder_g: Option<sdclip::ClipTextTransformer>,
    pub vae: AutoEncoderKL,
    pub unet: UNet2DConditionModel,
    pub device: Device,
    pub dtype: DType,
    /// Kept alive so merged-LoRA safetensors mmaps stay valid for
    /// the core's lifetime. Don't drop unless you also drop every
    /// pipeline holding an `Arc<SdCore>`.
    pub _lora_tmp: Vec<tempfile::NamedTempFile>,
}

impl SdCore {
    /// Resolve the model id, download weights, build the UNet / VAE
    /// / text encoder(s), and merge any user-supplied LoRAs.
    ///
    /// Flux models are rejected here (SdCore is SD-architecture only).
    /// Task-specific load (identity encoder, refiner, etc.) happens
    /// on the wrapping pipeline that owns the `Arc<SdCore>`.
    pub async fn load(req: SdLoadRequest) -> Result<Self> {
        let base_repo = if req.model.contains('/') {
            req.model.clone()
        } else {
            crate::hf::resolve_alias(&req.model).to_string()
        };
        let lc = base_repo.to_lowercase();
        if lc.contains("flux") {
            bail!(
                "SdCore does not support Flux (different architecture). \
                 Use --model sd15 (default), sd21, sdxl, sdxl-turbo, or any \
                 SD-family HF repo. Flux routes through pipelines::flux."
            );
        }
        let variant = SdVariant::detect(&base_repo);
        let cfg = variant.config(512, 512);
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        // -------- download base weights (variant-aware) --------
        let dl = progress::spinner(&format!(
            "Resolving {} weights",
            match variant {
                SdVariant::Sd15 => "SD 1.5",
                SdVariant::Sdxl => "SDXL",
            }
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
            SdVariant::Sd15 => (None, None),
            SdVariant::Sdxl => {
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

        // LoRAs arrive pre-resolved. Temp-file handles for merged
        // weights are accumulated below.
        let mut lora_tmps: Vec<tempfile::NamedTempFile> = Vec::new();
        let resolved_loras = &req.loras;

        // -------- build models --------
        let build = progress::spinner("Loading SD core");
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
                .prefix("plakat-sd-unet-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                &unet_path,
                tmp.path(),
                resolved_loras,
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
            SdVariant::Sd15 => crate::pipelines::lora::MergeTarget::TE_SD15,
            SdVariant::Sdxl => crate::pipelines::lora::MergeTarget::TE1_SDXL,
        };
        let effective_te_l_path = if resolved_loras.is_empty() {
            text_enc_l_path.clone()
        } else {
            let spin = progress::spinner(&format!("Merging LoRA into {}", te_l_target.name));
            let tmp = tempfile::Builder::new()
                .prefix("plakat-sd-te-l-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                &text_enc_l_path,
                tmp.path(),
                resolved_loras,
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
            SdVariant::Sd15 => None,
            SdVariant::Sdxl => {
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
                        .prefix("plakat-sd-te-g-")
                        .suffix(".safetensors")
                        .tempfile()?;
                    let (modified, targets) = crate::pipelines::lora::merge_loras_into_weights(
                        p,
                        tmp.path(),
                        resolved_loras,
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

        build.finish_with_message("✓ SD core loaded");

        Ok(Self {
            variant,
            cfg,
            tokenizer_l,
            tokenizer_g,
            text_encoder_l,
            text_encoder_g,
            vae,
            unet,
            device: req.device,
            dtype,
            _lora_tmp: lora_tmps,
        })
    }

    /// Device the SD core lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// dtype the SD core's tensors live at (F16 on accelerator,
    /// F32 on CPU).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn variant(&self) -> SdVariant {
        self.variant
    }

    pub fn cfg(&self) -> &StableDiffusionConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd_variant_detect() {
        assert_eq!(SdVariant::detect("sd15"), SdVariant::Sd15);
        assert_eq!(
            SdVariant::detect("stable-diffusion-v1-5/stable-diffusion-v1-5"),
            SdVariant::Sd15,
        );
        assert_eq!(SdVariant::detect("sdxl"), SdVariant::Sdxl);
        assert_eq!(SdVariant::detect("SDXL-turbo"), SdVariant::Sdxl);
    }

    #[test]
    fn sd_variant_cross_attn_dim() {
        assert_eq!(SdVariant::Sd15.cross_attn_dim(), 768);
        assert_eq!(SdVariant::Sdxl.cross_attn_dim(), 2048);
    }

    #[test]
    fn sd_variant_vae_scale_matches_sd_constants() {
        // diffusers' SD 1.5 and SDXL VAE scaling factors (verified
        // against the upstream config.json's `scaling_factor`).
        assert!((SdVariant::Sd15.vae_scale() - 0.18215).abs() < 1e-6);
        assert!((SdVariant::Sdxl.vae_scale() - 0.13025).abs() < 1e-6);
    }
}
