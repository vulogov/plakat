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
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{
    self, StableDiffusionConfig, clip as sdclip,
    vae::AutoEncoderKL,
};
use tokenizers::Tokenizer;

use crate::pipelines::lora::ResolvedLora;
use crate::pipelines::sdxl_clip::SdxlClipGTextTransformer;
use crate::pipelines::sdxl_unet::{SdUNet, SdxlAddEmbedConfig, SdxlUNet2DConditionModel};
use crate::ui::progress;

/// SD variant the backbone routes through. Detected from the model
/// alias / repo at load time. Covers every SD-family architecture
/// plakat supports (SD 1.5, SD 2.1, SDXL — SDXL-Turbo is
/// architecturally identical to SDXL and uses the `Sdxl` variant;
/// only the caller's scheduler defaults differ).
///
/// Flux is **not** supported by `SdCore`. Flux's pipeline has a
/// different architecture (transformer + T5, not UNet + CLIP) and
/// stays in `pipelines::flux`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdVariant {
    /// SD 1.5. CLIP-L only, `cross_attention_dim = 768`.
    Sd15,
    /// SD 2.1. OpenCLIP-H, `cross_attention_dim = 1024`,
    /// `use_linear_projection = true`. Architecturally distinct
    /// from SD 1.5.
    Sd21,
    /// SDXL (and SDXL-Turbo). Dual CLIP-L + CLIP-G,
    /// `cross_attention_dim = 2048`, `use_linear_projection = true`.
    Sdxl,
}

impl SdVariant {
    /// Detect the variant from a model name / HF repo id.
    /// Priority: Flux markers raise an error to the caller (we
    /// can't return None from a `-> Self` function — t2i checks
    /// for Flux separately before calling SdCore::load).
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
        // SDXL Turbo / SDXL / SDXL-Inpaint → Sdxl (same architecture
        // apart from the inpaint UNet's conv_in channel count, which
        // SdCore handles via the `is_inpaint` flag).
        if m.contains("xl") {
            return Self::Sdxl;
        }
        // SD 2.1: explicit "2-1" / "2.1" / "v2" markers.
        if m.contains("2-1") || m.contains("2.1") || m.contains("v2") {
            return Self::Sd21;
        }
        Self::Sd15
    }

    pub fn cross_attn_dim(self) -> usize {
        match self {
            Self::Sd15 => 768,
            Self::Sd21 => 1024,
            Self::Sdxl => 2048,
        }
    }

    pub fn vae_scale(self) -> f64 {
        match self {
            // SD 1.5 and SD 2.1 share the same VAE scaling factor.
            Self::Sd15 | Self::Sd21 => 0.18215,
            Self::Sdxl => 0.13025,
        }
    }

    pub fn config(self, w: usize, h: usize) -> StableDiffusionConfig {
        match self {
            Self::Sd15 => StableDiffusionConfig::v1_5(None, Some(h), Some(w)),
            Self::Sd21 => StableDiffusionConfig::v2_1(None, Some(h), Some(w)),
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
    /// SDXL only — CLIP-G (text_encoder_2) wrapped with the v0.11
    /// `text_projection` pooling head needed by the UNet's
    /// `add_embedding`. `None` for SD 1.5 / SD 2.1.
    pub text_encoder_g: Option<SdxlClipGTextTransformer>,
    pub vae: AutoEncoderKL,
    /// Backbone UNet. `SdUNet::Sd` for SD 1.5 / SD 2.1 (candle's
    /// upstream type); `SdUNet::Sdxl` for SDXL (v0.11 phase 8 — adds
    /// `text_time` micro-conditioning that diffusers' SDXL relies on
    /// for full-quality outputs).
    pub unet: SdUNet,
    /// v0.12: 9-channel inpainting UNet (
    /// `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` for SDXL,
    /// `stable-diffusion-v1-5/stable-diffusion-inpainting` for SD 1.5,
    /// or any SD 2.x inpainting mirror that follows the same naming).
    /// When set, the loaded UNet expects a 9-channel input
    /// `[noisy_latents(4), mask(1), masked_image_latents(4)]`. The
    /// img2img/portrait masked paths skip RePaint-style mask blending
    /// and instead concat the mask + masked-image latents along the
    /// channel dim at every denoise step. `false` for everything else
    /// (regular 4-channel UNets — RePaint blending stays in play).
    pub is_inpaint: bool,
    pub device: Device,
    pub dtype: DType,
    /// Kept alive so merged-LoRA safetensors mmaps stay valid for
    /// the core's lifetime. Don't drop unless you also drop every
    /// pipeline holding an `Arc<SdCore>`.
    pub _lora_tmp: Vec<tempfile::NamedTempFile>,
}

/// v0.12: does this resolved repo id name a 9-channel inpainting
/// UNet? Determines `SdCore.is_inpaint`. The check is intentionally
/// loose — any SD-architecture repo whose id contains "inpaint" /
/// "inpainting" gets the 9-channel UNet build path. Covers stock
/// SDXL-Inpaint, SD 1.5 inpaint mirrors, and community SD 2.x
/// inpainting checkpoints that follow the same naming.
pub fn detect_inpaint(base_repo: &str) -> bool {
    let m = base_repo.to_lowercase();
    m.contains("inpaint") || m.contains("inpainting")
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
        let is_inpaint = detect_inpaint(&base_repo);
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
                SdVariant::Sd21 => "SD 2.1",
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
            // SD 1.5 + SD 2.1 each have a single text encoder
            // (no CLIP-G dual encoder).
            SdVariant::Sd15 | SdVariant::Sd21 => (None, None),
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
        // v0.12: inpainting checkpoints (SD 1.5 / SD 2.1 / SDXL) carry
        // 9 input channels instead of 4 — same UNet architecture
        // otherwise, only `conv_in` changes shape.
        let unet_in_channels = if is_inpaint { 9 } else { 4 };
        let unet = match variant {
            // SD 1.5 / SD 2.1 — candle's stock UNet (no add_embedding).
            SdVariant::Sd15 | SdVariant::Sd21 => SdUNet::Sd(
                cfg.build_unet(
                    &effective_unet_path,
                    &req.device,
                    unet_in_channels,
                    false,
                    dtype,
                )?,
            ),
            // SDXL — vendored UNet with `text_time` add_embedding.
            // Reuses controlnet::sdxl_unet_config() so the SDXL UNet
            // shape stays defined in one place. AddEmbedConfig::base()
            // = 6 time_ids; the refiner gets its own variant in 8e.
            SdVariant::Sdxl => {
                let vs_unet = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[effective_unet_path.as_path()],
                        dtype,
                        &req.device,
                    )?
                };
                let sdxl_unet = SdxlUNet2DConditionModel::new(
                    vs_unet,
                    unet_in_channels,
                    4,
                    false,
                    crate::pipelines::controlnet::sdxl_unet_config(),
                    SdxlAddEmbedConfig::base(),
                )?;
                SdUNet::Sdxl(sdxl_unet)
            }
        };

        // CLIP-L text encoder (with optional LoRA merge).
        // SD 2.1 uses the same key naming as SD 1.5 for LoRA merge
        // targets (both have a single `text_encoder` module on disk).
        let te_l_target = match variant {
            SdVariant::Sd15 | SdVariant::Sd21 => crate::pipelines::lora::MergeTarget::TE_SD15,
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
            SdVariant::Sd15 | SdVariant::Sd21 => None,
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
                // v0.11 phase 8b: load via the SdxlClipGTextTransformer
                // wrapper so the `text_projection` Linear is also
                // pulled out of the safetensors. embed_dim = 1280 is
                // the stock SDXL CLIP-G width (candle's Config::embed_dim
                // is private so we pass it explicitly).
                let vs_g = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[effective_te_g_path.as_path()],
                        dtype,
                        &req.device,
                    )?
                };
                Some(SdxlClipGTextTransformer::new(vs_g, cfg_g, 1280)?)
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
            is_inpaint,
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
