//! Text-to-image inference pipeline.
//!
//! Supported in candle 0.8:
//!   * SD 1.5 / 2.1            — single CLIP-L text encoder, VAE scale 0.18215
//!   * SDXL / SDXL-Turbo       — dual encoder (CLIP-L + CLIP-G), penultimate
//!                               hidden states concatenated to 2048 channels;
//!                               VAE scale 0.13025
//!
//! Flux is detected but routes out to `pipelines::flux`.
//!
//! Two ways to use this module:
//!   * `t2i::run(Request)` — single-shot. Loads everything then generates.
//!     This is what `plakat generate` uses.
//!   * `Pipeline::load(...)` then `pipeline.generate(...)` per task. Reuses
//!     loaded weights across calls so multi-task scenarios don't pay the
//!     ~10s model-load overhead N times. This is what `plakat scenario` uses.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{
    self, clip as sdclip,
    unet_2d::{BlockConfig, UNet2DConditionModelConfig},
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;
use crate::ui::progress;

// =====================================================================
// Public types: single-shot Request/run (back-compat) + Pipeline API.
// =====================================================================

pub struct Request {
    pub prompt: String,
    pub negative: String,
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
    /// Enable the real SDXL refiner. Loads `stable-diffusion-xl-refiner-1.0`
    /// UNet alongside the base. Ignored unless `model` resolves to SDXL/Turbo.
    pub use_refiner: bool,
    /// Fraction of the schedule where the refiner takes over (default 0.8).
    pub refiner_frac: f32,

    // ---------- v0.9 ControlNet (v0.11: multi) ----------
    /// Stack of ControlNet conditioners. Empty disables ControlNet
    /// entirely (preserves byte-identical pre-v0.9 behaviour). One
    /// entry mirrors the v0.9–v0.10 single-conditioner path. Two+
    /// entries run diffusers-style multi-ControlNet — residuals from
    /// each active conditioner are summed before being fed to the UNet.
    /// All conditioners share the SD/SDXL variant (determined by `model`).
    pub controls: Vec<crate::pipelines::controlnet::ControlSpec>,

    // ---------- v0.12 / v0.13 / v0.14 tiled hi-res ----------
    /// When `Some`, run MultiDiffusion-style tiled denoise on a
    /// canvas of `(width, height)`. The UNet only ever sees
    /// `tile_size × tile_size` crops; overlapping tiles are blended
    /// per step via a 2D Hann window. Lets the backbone produce 4K+
    /// outputs without exceeding its trained working resolution.
    ///
    /// Supported on SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo (v0.12 +
    /// v0.14 phase 4) and Flux (v0.13 phase 4). SD3 (MMDiT) is the
    /// remaining gap. ControlNet + the SDXL refiner are still
    /// rejected — the per-tile residual concat and the scheduler-
    /// switch mid-stream don't compose with MultiDiffusion.
    pub tiled: Option<crate::pipelines::tiled::TiledConfig>,

    /// v0.13 phase 1b: quantize T5-XXL when running Flux GGUF.
    /// Only meaningful with `--model flux-*-gguf`; bails loud on
    /// BF16 Flux. Ignored entirely on SD-family models.
    pub quantize_t5: bool,
    /// v0.13 phase 5: GGUF quant level for the Flux transformer. `None`
    /// → `"Q4_K_S"`. Validated against city96's published levels.
    /// Ignored on BF16 Flux and SD-family models.
    pub flux_quant_level: Option<String>,
    /// v0.13 phase 5: GGUF quant level for the T5-XXL encoder. `None`
    /// → `"Q4_K_M"`. Ignored unless `quantize_t5` is `true`.
    pub t5_quant_level: Option<String>,
    /// v0.14 phase 3 / 3c: zero or more Flux Redux reference images.
    /// Each entry is encoded via SigLIP-so400m + the BFL Redux
    /// adapter and contributes 729 tokens (scaled by its weight) to
    /// the T5 text embedding. Empty disables Redux. Cap of 4 enforced
    /// at `Pipeline::generate`. Ignored on SD-family models.
    pub redux_images: Vec<crate::pipelines::flux_redux::ReduxSpec>,
    /// v0.15 phase 4: conditioning map for Flux.1-Canny-dev /
    /// Flux.1-Depth-dev. Required for those variants; ignored otherwise.
    pub flux_concept_image: Option<PathBuf>,
}

/// Stuff that's fixed for the lifetime of a Pipeline.
pub struct LoadRequest {
    pub model: String,
    pub device: Device,
    pub loras: Vec<LoraSpec>,
    pub lora_scale: f32,
    /// If `true` AND the variant is SDXL/SDXL-Turbo, also load the official
    /// `stabilityai/stable-diffusion-xl-refiner-1.0` UNet for a two-pass
    /// schedule. Adds a ~6 GB download on first run.
    pub use_refiner: bool,
}

/// Stuff that can vary per `Pipeline::generate` call.
pub struct GenRequest {
    pub prompt: String,
    pub negative: String,
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
    /// Fraction of the schedule (0..1) at which to switch from base UNet to
    /// the refiner UNet. Only takes effect when the Pipeline was built with
    /// `use_refiner: true`. Default 0.8 — last ~20% of steps use refiner.
    /// `None` = no refiner pass even if the refiner is loaded.
    pub refiner_frac: Option<f32>,
}

// =====================================================================
// Variant detection.
// =====================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Sd15,
    Sd21,
    Sdxl,
    SdxlTurbo,
    FluxSchnell,
    FluxDev,
    /// v0.13 phase 2: Flux.1-Fill-dev. 384-channel `img_in`, inpaint-only.
    FluxFillDev,
    /// v0.15 phase 4: Flux.1-Canny-dev. BFL "concept" checkpoint with
    /// canny-edge conditioning baked into a 128-channel `img_in`.
    FluxCannyDev,
    /// v0.15 phase 4: Flux.1-Depth-dev. Same shape as Canny-dev but
    /// trained on depth-map conditioning.
    FluxDepthDev,
    /// v0.14 phase 1a: Stable Diffusion 3.5 Medium (MMDiT).
    Sd35Medium,
    /// v0.14 phase 8a: SD3.5 Large (8B-param flagship MMDiT).
    Sd35Large,
    /// v0.14 phase 8a: SD3.5 Large Turbo (4-step distillation).
    Sd35LargeTurbo,
    /// v0.14 phase 8a: original Stable Diffusion 3 Medium (June 2024).
    Sd3Medium,
}

impl Variant {
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
        // SD3 detection precedes SDXL/SD because "sd3" / "sd3.5" /
        // "stable-diffusion-3.5" contain "sd" but should route to the
        // MMDiT pipeline. Sub-variant differentiation:
        //   * "large-turbo" / "3.5-large-turbo" → Sd35LargeTurbo
        //   * "large"       / "3.5-large"       → Sd35Large
        //   * "3.5"         / "3-5"             → Sd35Medium
        //   * "sd3" + "medium" (no "3.5")       → Sd3Medium
        // Order matters: check "turbo" before "large" since the turbo
        // string contains "large".
        let is_sd3_family = m.contains("sd3")
            || m.contains("sd-3")
            || m.contains("stable-diffusion-3");
        if is_sd3_family {
            if m.contains("turbo") {
                return Self::Sd35LargeTurbo;
            }
            if m.contains("large") {
                return Self::Sd35Large;
            }
            // Pick the SD3.x sub-version: 3.5 ships everything except
            // the original 3-medium. Match "3.5" or "3-5" (HF repos
            // use the dash form: `stable-diffusion-3.5-medium`).
            if m.contains("3.5") || m.contains("3-5") {
                return Self::Sd35Medium;
            }
            // Original SD3-medium (no `.5` in the name).
            if m.contains("medium") {
                return Self::Sd3Medium;
            }
            // Default for `sd3` / `sd-3` strings without a sub-flag —
            // pick the recommended Medium variant from the modern lineup.
            return Self::Sd35Medium;
        }
        if m.contains("flux") {
            // v0.15 phase 4: Canny / Depth concept variants precede
            // the generic "dev" check — "flux-canny-dev" contains
            // "dev" but routes to the 128-channel `img_in` config.
            if m.contains("canny") {
                Self::FluxCannyDev
            } else if m.contains("depth") {
                Self::FluxDepthDev
            } else if m.contains("fill") {
                Self::FluxFillDev
            } else if m.contains("dev") {
                Self::FluxDev
            } else {
                Self::FluxSchnell
            }
        } else if m.contains("turbo") {
            Self::SdxlTurbo
        } else if m.contains("xl") {
            Self::Sdxl
        } else if m.contains("2-1") || m.contains("2.1") || m.contains("v2") {
            Self::Sd21
        } else {
            Self::Sd15
        }
    }

    pub fn is_xl(self) -> bool {
        matches!(self, Self::Sdxl | Self::SdxlTurbo)
    }
    pub fn is_flux(self) -> bool {
        matches!(
            self,
            Self::FluxSchnell
                | Self::FluxDev
                | Self::FluxFillDev
                | Self::FluxCannyDev
                | Self::FluxDepthDev
        )
    }
    /// v0.15 phase 4: BFL "concept" Flux variants (Canny-dev /
    /// Depth-dev). Conditioning is baked into a 128-channel `img_in`
    /// rather than via a separate ControlNet.
    pub fn is_flux_concept(self) -> bool {
        matches!(self, Self::FluxCannyDev | Self::FluxDepthDev)
    }
    /// v0.14 phase 1a / 8a: any SD3 / SD3.5 variant. Routes to the
    /// MMDiT pipeline in `pipelines::sd3`.
    pub fn is_sd3(self) -> bool {
        matches!(
            self,
            Self::Sd3Medium
                | Self::Sd35Medium
                | Self::Sd35Large
                | Self::Sd35LargeTurbo
        )
    }
}

fn resolve_repo(model: &str) -> String {
    if model.contains('/') {
        model.to_string()
    } else {
        crate::hf::resolve_alias(model).to_string()
    }
}

/// v0.12 phase 2b: pick a community Flux ControlNet for the user's
/// requested conditioning kind. Routes to Shakker-Labs/Union-Pro-v2
/// for the multi-mode conditioners (canny, softedge, openpose, depth,
/// lineart) and falls back to specialised CNs where Union doesn't
/// cover a kind.
///
/// Shakker-Labs Union Pro v2 mode mapping (per the model card):
///   0: canny       (also covers `lineart` in practice)
///   1: softedge    (HED-style)
///   2: openpose
///   3: depth
///   4: gray
pub(crate) fn flux_controlnet_load_for(
    kind: crate::pipelines::controlnet::ControlKind,
    fvar: crate::pipelines::flux::Variant,
    strength: f32,
) -> Result<crate::pipelines::flux::FluxControlNetLoad> {
    use crate::pipelines::controlnet::ControlKind;
    use crate::pipelines::flux;
    use crate::pipelines::flux_controlnet;
    // v0.14 phase 5: allow ControlNet on both Flux.1-dev and
    // Flux.1-Fill-dev. The Union Pro v2 weights were trained against
    // Flux.1-dev, but Fill shares everything except `img_in`'s width;
    // CN residuals are added at the hidden state level (3072d, post
    // `img_in`), so the same CN model composes with Fill's noise
    // tokens cleanly. Schnell stays gated — fewer real-world reports
    // of Schnell-CN compatibility and the rectified-flow schedule
    // diverges enough that we want explicit validation first.
    if !matches!(fvar, flux::Variant::Dev | flux::Variant::FillDev) {
        anyhow::bail!(
            "Flux ControlNet is wired for flux-dev and flux-fill-dev. \
             FLUX.1-schnell ControlNets exist but aren't yet validated."
        );
    }
    // Default route: Shakker-Labs Union Pro v2. Covers canny / softedge
    // / openpose / depth / lineart with one checkpoint via mode index.
    let union_pro_v2_repo = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0";
    let union_pro_v2_file = "diffusion_pytorch_model.safetensors";
    let union_mode = match kind {
        ControlKind::Canny => Some(0u32),
        ControlKind::SoftEdge => Some(1u32),
        ControlKind::OpenPose => Some(2u32),
        ControlKind::Depth => Some(3u32),
        // Lineart isn't a separate Union Pro v2 mode but produces
        // close-enough results via the canny channel (canny is
        // strict edge detection; lineart is softer pen-style).
        ControlKind::Lineart => Some(0u32),
    };
    Ok(flux::FluxControlNetLoad {
        repo: union_pro_v2_repo.to_string(),
        file: union_pro_v2_file.to_string(),
        cfg: flux_controlnet::Config::shakker_union_pro_v2(),
        scale: strength,
        mode: union_mode,
        // Caller fills in the conditioning path + start/end after this
        // returns — we don't have access to the per-spec data here.
        conditioning: None,
        // Sane defaults: active the entire schedule. The caller
        // overwrites from `ControlSpec.start` / `.end`.
        start: 0.0,
        end: 1.0,
    })
}

/// v0.13 phase 8: take the `(1, 3, H, W)` `[0, 1]` tensor a
/// ControlNet auto-annotator produces and write it as an 8-bit RGB
/// PNG. The Flux ControlNet path consumes its conditioning via a path
/// (`encode_conditioning` reads + VAE-encodes), so this is the bridge
/// from "annotator output in tensor land" to "path the Flux pipeline
/// loads".
pub(crate) fn write_annotator_tensor_as_png(anno: &Tensor, out_path: &std::path::Path) -> Result<()> {
    // Annotator emits (1, 3, H, W) in [0, 1]. Convert to (H, W, 3) u8.
    let (b, c, h, w) = anno.dims4()?;
    if b != 1 || c != 3 {
        anyhow::bail!(
            "annotator output expected shape (1, 3, H, W), got ({b}, {c}, {h}, {w})"
        );
    }
    let scaled = (anno * 255.0)?
        .clamp(0f32, 255f32)?
        .to_dtype(DType::U8)?
        .i(0)?
        .permute((1, 2, 0))?;
    let buf = scaled.flatten_all()?.to_vec1::<u8>()?;
    crate::imaging::io::save_rgb_u8(&buf, w as u32, h as u32, out_path)?;
    Ok(())
}

async fn fetch_first(repo: &str, candidates: &[&str]) -> Result<PathBuf> {
    let mut last_err = None;
    for f in candidates {
        match crate::hf::download::get_file(repo, f).await {
            Ok(p) => return Ok(p),
            Err(e) => {
                tracing::debug!(target: "plakat", "miss {repo}/{f}: {e}");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("no candidates given")))
}

// =====================================================================
// Pipeline: load once, generate many.
// =====================================================================

/// t2i wrapping pipeline. Phase 7c: holds an `Arc<SdCore>` for the
/// shared SD backbone, plus the t2i-specific `variant` (which knows
/// about SDXL-Turbo / SD 2.1 distinctions that the architectural
/// SdCore collapses) and the optional SDXL refiner UNet.
///
/// Sharing the SdCore across pipelines (e.g. with portrait::Pipeline
/// when --artefact-blend is set, v0.10 phase 7d) eliminates
/// redundant model loads.
pub struct Pipeline {
    pub variant: Variant,
    /// Resolved HF repo id this pipeline was loaded from (after alias resolution).
    #[allow(dead_code)]
    pub repo: String,
    core: std::sync::Arc<crate::pipelines::sd_core::SdCore>,
    /// Optional second UNet from `stabilityai/stable-diffusion-xl-refiner-1.0`.
    /// Present only when `LoadRequest::use_refiner == true` and the variant
    /// is SDXL/SDXL-Turbo. Not shared via SdCore — the refiner is
    /// t2i-specific.
    ///
    /// Wrapped in [`SdUNet`] for uniform denoise dispatch. Phase 8d
    /// holds it as `SdUNet::Sd` (no add_embedding — quality gap
    /// preserved from pre-v0.11 behaviour). Phase 8e switches this to
    /// `SdUNet::Sdxl` with the refiner's 5-time-id add_embedding so the
    /// refiner pass also gets `text_time` micro-conditioning.
    refiner_unet: Option<crate::pipelines::sdxl_unet::SdUNet>,
}

const SDXL_REFINER_REPO: &str = "stabilityai/stable-diffusion-xl-refiner-1.0";

/// SDXL refiner UNet configuration, hand-derived from the HF
/// `stable-diffusion-xl-refiner-1.0/unet/config.json`. Architectural notes:
///   * 4 blocks (vs base's 3): two DownBlock2D + two CrossAttnDownBlock2D
///   * Cross-attention dim 1280 (CLIP-G only) — vs base's 2048 concat
///   * Per-block transformer layer counts [1, 4, 4, 1] (only middle two
///     have cross-attention)
///   * Per-block attention head dims [4, 8, 16, 16]
///
/// Known limitation: the refiner is trained with `addition_embed_type:
/// text_time` (pooled CLIP-G + time_ids micro-conditioning). candle 0.8's
/// UNet has no `add_embedding` projection so we silently skip that. The
/// model loads and runs, but output quality is lower than the diffusers
/// reference. Same gap our base-SDXL path takes.
fn sdxl_refiner_unet_config() -> UNet2DConditionModelConfig {
    UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig {
                out_channels: 384,
                use_cross_attn: None,
                attention_head_dim: 4,
            },
            BlockConfig {
                out_channels: 768,
                use_cross_attn: Some(4),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 1536,
                use_cross_attn: Some(4),
                attention_head_dim: 16,
            },
            BlockConfig {
                out_channels: 1536,
                use_cross_attn: None,
                attention_head_dim: 16,
            },
        ],
        center_input_sample: false,
        cross_attention_dim: 1280,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

async fn fetch_refiner_unet() -> Result<PathBuf> {
    fetch_first(
        SDXL_REFINER_REPO,
        &[
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
        ],
    )
    .await
    .with_context(|| format!("refiner unet weights from {SDXL_REFINER_REPO}"))
}

impl Pipeline {
    /// Download + load + merge LoRAs once. SD/SDXL only — Flux
    /// routes to `pipelines::flux::run`.
    ///
    /// Phase 7c: the SD backbone load delegates to
    /// [`SdCore::load`](crate::pipelines::sd_core::SdCore::load).
    /// t2i-specific concerns (Flux rejection, optional SDXL refiner
    /// UNet) stay here.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let variant = Variant::detect(&req.model);
        if variant.is_flux() {
            anyhow::bail!(
                "Pipeline::load is SD-only; Flux models use pipelines::flux::run"
            );
        }
        let repo = resolve_repo(&req.model);

        // Resolve user LoRAs (t2i has no auto-LoRAs; that's a
        // portrait-pipeline concern for FaceID).
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

        // Delegate the SD backbone load. SdVariant::detect collapses
        // t2i's SdxlTurbo→Sdxl mapping (same architecture; only
        // scheduler defaults differ, which we don't carry through
        // SdCore).
        let core = crate::pipelines::sd_core::SdCore::load(
            crate::pipelines::sd_core::SdLoadRequest {
                model: req.model.clone(),
                device: req.device.clone(),
                loras: resolved_loras,
                lora_scale: req.lora_scale,
            },
        )
        .await
        .context("loading SD backbone for t2i pipeline")?;
        let dtype = core.dtype;

        // Optional second UNet: SDXL refiner. t2i-specific.
        let refiner_unet = if req.use_refiner {
            if !variant.is_xl() {
                anyhow::bail!(
                    "SDXL refiner is only valid with --model sdxl or sdxl-turbo; \
                     `{}` is {:?}",
                    req.model,
                    variant
                );
            }
            let refiner_spinner = progress::spinner(&format!(
                "Downloading SDXL refiner UNet from {SDXL_REFINER_REPO}"
            ));
            let weights = fetch_refiner_unet().await?;
            refiner_spinner.finish_with_message("✓ refiner weights ready");

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &req.device)?
            };
            // v0.11 phase 8e: refiner now loads via SdxlUNet too. Its
            // add_embedding takes a 5-id time vector (aesthetic_score
            // replaces target_size compared to the base UNet's 6-id
            // vector); SdxlAddEmbedConfig::refiner() captures that.
            // Refiner cross_attn_dim stays 1280 (CLIP-G only) per
            // sdxl_refiner_unet_config.
            let r_unet =
                crate::pipelines::sdxl_unet::SdxlUNet2DConditionModel::new(
                    vb,
                    4,
                    4,
                    false,
                    sdxl_refiner_unet_config(),
                    crate::pipelines::sdxl_unet::SdxlAddEmbedConfig::refiner(),
                )?;
            Some(crate::pipelines::sdxl_unet::SdUNet::Sdxl(r_unet))
        } else {
            None
        };

        Ok(Self {
            variant,
            repo,
            core: std::sync::Arc::new(core),
            refiner_unet,
        })
    }

    /// Hand out a cheap `Arc` clone of the loaded SD backbone so a
    /// follow-on step (e.g. `--artefact-blend`) can build its own
    /// pipeline (`portrait::Pipeline::from_core`) without paying for
    /// a second model load. Phase 7d.
    pub fn core(&self) -> std::sync::Arc<crate::pipelines::sd_core::SdCore> {
        std::sync::Arc::clone(&self.core)
    }

    /// Encode `prompt` (and optionally `negative` for CFG) into the
    /// `encoder_hidden_states` tensor the UNet expects.
    /// Returns `(hidden_states, pooled_text_for_sdxl)`:
    ///   * For SD 1.5 / SD 2.1: pooled is `None` (no add_embedding).
    ///   * For SDXL: pooled is `Some((B, 1280))` feeding the UNet's
    ///     `add_embedding`. Caller pairs it with an `add_time_ids`
    ///     tensor built in `generate`.
    fn encode_prompt(
        &self,
        prompt: &str,
        negative: &str,
        do_cfg: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if self.variant.is_xl() {
            let (hidden, pooled) = self.encode_xl(prompt, negative, do_cfg)?;
            Ok((hidden, Some(pooled)))
        } else {
            let hidden = self.encode_single(prompt, negative, do_cfg)?;
            Ok((hidden, None))
        }
    }

    fn encode_single(&self, prompt: &str, negative: &str, do_cfg: bool) -> Result<Tensor> {
        let cond_ids = tokenize_padded(&self.core.tokenizer_l, &self.core.cfg.clip, prompt, &self.core.device)?;
        let cond = self.core.text_encoder_l.forward(&cond_ids)?;
        if !do_cfg {
            return Ok(cond.to_dtype(self.core.dtype)?);
        }
        let uncond_ids =
            tokenize_padded(&self.core.tokenizer_l, &self.core.cfg.clip, negative, &self.core.device)?;
        let uncond = self.core.text_encoder_l.forward(&uncond_ids)?;
        Ok(Tensor::cat(&[&uncond, &cond], 0)?.to_dtype(self.core.dtype)?)
    }

    /// CLIP-G-only encoding for the SDXL refiner UNet. Produces
    /// `(B, 77, 1280)` — base SDXL uses `(B, 77, 2048)` instead.
    fn encode_g_only(&self, prompt: &str, negative: &str, do_cfg: bool) -> Result<Tensor> {
        let cfg_g = self
            .core
            .cfg
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow!("refiner encoding needs clip2 (SDXL config)"))?;
        let tok_g = self
            .core
            .tokenizer_g
            .as_ref()
            .ok_or_else(|| anyhow!("refiner encoding needs tokenizer_g"))?;
        let enc_g = self
            .core
            .text_encoder_g
            .as_ref()
            .ok_or_else(|| anyhow!("refiner encoding needs text_encoder_g"))?;

        let cond = embed_g_only(prompt, tok_g, cfg_g, enc_g, &self.core.device)?;
        if !do_cfg {
            return Ok(cond.to_dtype(self.core.dtype)?);
        }
        let uncond = embed_g_only(negative, tok_g, cfg_g, enc_g, &self.core.device)?;
        Ok(Tensor::cat(&[&uncond, &cond], 0)?.to_dtype(self.core.dtype)?)
    }

    /// SDXL base encoding. Returns:
    ///   * `hidden_states` — `(B, 77, 2048)`. CFG batches uncond then
    ///     cond along the first dim (matches the existing convention).
    ///   * `pooled_text`  — `(B, 1280)` from the projected EOT row of
    ///     CLIP-G's final layer norm output. Feeds the UNet's
    ///     `add_embedding`. Same uncond-then-cond batching.
    ///
    /// `pooled_text` is the v0.11 phase-8d addition. Pre-8d the base
    /// SDXL path threw this away (and the UNet ran with `add_embedding`
    /// silently inactive). Caller now passes it through `denoise_step`.
    fn encode_xl(
        &self,
        prompt: &str,
        negative: &str,
        do_cfg: bool,
    ) -> Result<(Tensor, Tensor)> {
        let cfg_g = self
            .core
            .cfg
            .clip2
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL config is missing clip2"))?;
        let tok_g = self
            .core
            .tokenizer_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL missing tokenizer_g"))?;
        let enc_g = self
            .core
            .text_encoder_g
            .as_ref()
            .ok_or_else(|| anyhow!("SDXL missing text_encoder_g"))?;

        let (cond_hidden, cond_pooled) = embed_xl(
            prompt,
            &self.core.tokenizer_l,
            tok_g,
            &self.core.cfg.clip,
            cfg_g,
            &self.core.text_encoder_l,
            enc_g,
            &self.core.device,
        )?;
        if !do_cfg {
            return Ok((
                cond_hidden.to_dtype(self.core.dtype)?,
                cond_pooled.to_dtype(self.core.dtype)?,
            ));
        }
        let (uncond_hidden, uncond_pooled) = embed_xl(
            negative,
            &self.core.tokenizer_l,
            tok_g,
            &self.core.cfg.clip,
            cfg_g,
            &self.core.text_encoder_l,
            enc_g,
            &self.core.device,
        )?;
        let hidden = Tensor::cat(&[&uncond_hidden, &cond_hidden], 0)?.to_dtype(self.core.dtype)?;
        let pooled = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?.to_dtype(self.core.dtype)?;
        Ok((hidden, pooled))
    }

    /// Generate `req.count` images for one prompt. Reuses the loaded
    /// UNet/VAE/text encoder.
    ///
    /// `controls` is the v0.9 ControlNet hook, extended to a stack in
    /// v0.11. Every denoise step picks the subset whose timestep
    /// window is active at that step's progress (see
    /// `ControlRequest::active_at`), sums each one's residuals, and
    /// hands the combined residuals to the UNet via
    /// `forward_with_additional_residuals`. Empty slice = byte-identical
    /// pre-v0.9 behaviour.
    pub fn generate(
        &self,
        req: &GenRequest,
        controls: &[crate::pipelines::controlnet::ControlRequest],
    ) -> Result<()> {
        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.core.device)?;
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        let (w, h) = (req.width as usize, req.height as usize);
        let do_cfg = req.guidance > 1.0;
        let (text_embeddings, pooled_text_sdxl) =
            self.encode_prompt(&req.prompt, &req.negative, do_cfg)?;

        // If the refiner is loaded AND the caller asked for it, prepare the
        // CLIP-G-only embeddings the refiner needs (different cross_attn_dim
        // means we can't reuse `text_embeddings`).
        let refiner_embeddings = match (&self.refiner_unet, req.refiner_frac) {
            (Some(_), Some(_)) => Some(self.encode_g_only(&req.prompt, &req.negative, do_cfg)?),
            _ => None,
        };

        // v0.11 phase 8d: build the SDXL base add_time_ids tensor once
        // per generate() call (it's a function of target size only —
        // identical across denoise steps and across cond/uncond CFG
        // branches). For non-SDXL variants this stays None and the
        // UNet enum ignores the parameter.
        let add_time_ids_base = if self.variant.is_xl() && pooled_text_sdxl.is_some() {
            let cond_ids = crate::pipelines::sdxl_unet::build_add_time_ids_base(
                req.height,
                req.width,
                &self.core.device,
                self.core.dtype,
            )?;
            // For CFG the same vector is replicated across uncond + cond
            // — diffusers does the same.
            let stacked = if do_cfg {
                Tensor::cat(&[&cond_ids, &cond_ids], 0)?
            } else {
                cond_ids
            };
            Some(stacked)
        } else {
            None
        };

        // v0.11 phase 8e: refiner add_time_ids (5 floats; last slot is
        // aesthetic_score). cond uses POS=6.0; uncond uses NEG=2.5 so
        // CFG pulls toward higher aesthetics — matching diffusers'
        // default refiner inference setup. Only built when the refiner
        // is actually going to run.
        let add_time_ids_refiner = match (&self.refiner_unet, req.refiner_frac, &pooled_text_sdxl) {
            (Some(_), Some(_), Some(_)) => {
                let cond_ids = crate::pipelines::sdxl_unet::build_add_time_ids_refiner(
                    req.height,
                    req.width,
                    crate::pipelines::sdxl_unet::REFINER_AESTHETIC_SCORE_POS,
                    &self.core.device,
                    self.core.dtype,
                )?;
                let stacked = if do_cfg {
                    let uncond_ids = crate::pipelines::sdxl_unet::build_add_time_ids_refiner(
                        req.height,
                        req.width,
                        crate::pipelines::sdxl_unet::REFINER_AESTHETIC_SCORE_NEG,
                        &self.core.device,
                        self.core.dtype,
                    )?;
                    Tensor::cat(&[&uncond_ids, &cond_ids], 0)?
                } else {
                    cond_ids
                };
                Some(stacked)
            }
            _ => None,
        };

        let bsz: usize = 1;
        let latent_h = h / 8;
        let latent_w = w / 8;
        let vae_scale: f64 = self.core.variant.vae_scale();

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random)
                & (u32::MAX as u64);
            if let Err(e) = self.core.device.set_seed(seed) {
                tracing::debug!(
                    target: "plakat",
                    "set_seed not supported ({e}); using global RNG"
                );
            }

            let mut scheduler =
                crate::pipelines::scheduler::build(req.scheduler, &self.core.cfg, req.steps)?;
            let timesteps = scheduler.timesteps().to_vec();

            let mut latents =
                Tensor::randn(0f32, 1f32, (bsz, 4, latent_h, latent_w), &self.core.device)?
                    .to_dtype(self.core.dtype)?;
            latents = (latents * scheduler.init_noise_sigma())?;

            // Compute the index where to switch from base UNet to the
            // refiner UNet (only when both are present). `switch == len`
            // means all base, no refiner step. `switch == 0` means all
            // refiner — generally not what the user wants.
            let switch = match (&self.refiner_unet, req.refiner_frac, refiner_embeddings.as_ref())
            {
                (Some(_), Some(frac), Some(_)) => {
                    let f = frac.clamp(0.0, 1.0);
                    ((timesteps.len() as f32) * f).round() as usize
                }
                _ => timesteps.len(),
            };
            let switch = switch.min(timesteps.len());

            let bar = progress::step_bar(
                timesteps.len() as u64,
                &format!("img {}/{}", idx + 1, req.count),
            );
            let total_steps = timesteps.len();
            for (step_i, &timestep) in timesteps.iter().enumerate() {
                let in_base = step_i < switch;
                let (unet_ref, embeds, tag) = if in_base {
                    (&self.core.unet, &text_embeddings, "base")
                } else {
                    (
                        self.refiner_unet.as_ref().unwrap(),
                        refiner_embeddings.as_ref().unwrap(),
                        "refiner",
                    )
                };
                // 8e: pooled_text is shared between base and refiner
                // (same CLIP-G projection). Only the time_ids differ —
                // base uses 6 floats with target_size; refiner uses 5
                // floats with aesthetic_score. Both passes hit the
                // SdxlUNet::Sdxl path so both require pooled + time_ids.
                let (sdxl_pooled, sdxl_time_ids) = if in_base {
                    (pooled_text_sdxl.as_ref(), add_time_ids_base.as_ref())
                } else {
                    (pooled_text_sdxl.as_ref(), add_time_ids_refiner.as_ref())
                };
                let progress = step_i as f32 / total_steps as f32;
                let active_controls: Vec<&crate::pipelines::controlnet::ControlRequest> =
                    controls.iter().filter(|c| c.active_at(progress)).collect();
                latents = self.denoise_step(
                    unet_ref,
                    &latents,
                    timestep,
                    embeds,
                    sdxl_pooled,
                    sdxl_time_ids,
                    &mut scheduler,
                    req.guidance,
                    do_cfg,
                    &active_controls,
                )?;
                bar.inc(1);
                bar.set_message(format!("{tag} t={timestep} seed={seed}"));
            }
            bar.finish_and_clear();

            // Optional polish pass: img2img with same model at low strength.
            if let Some(rsteps) = req.refine {
                if rsteps > 0 {
                    let strength = req.refine_strength.clamp(0.0, 1.0);
                    let mut polish = crate::pipelines::scheduler::build(
                        req.scheduler,
                        &self.core.cfg,
                        rsteps,
                    )?;
                    let pts = polish.timesteps().to_vec();
                    let init_skip = ((rsteps as f32) * (1.0 - strength)).round() as usize;
                    let init_skip = init_skip.min(rsteps.saturating_sub(1));
                    let active = &pts[init_skip..];
                    if let Some(&start_t) = active.first() {
                        let noise = Tensor::randn(0f32, 1f32, latents.shape(), &self.core.device)?
                            .to_dtype(self.core.dtype)?;
                        latents = polish.add_noise(&latents, noise, start_t)?;
                        let rbar = progress::step_bar(
                            active.len() as u64,
                            &format!("polish {}/{}", idx + 1, req.count),
                        );
                        let total_polish = active.len();
                        for (step_idx, &timestep) in active.iter().enumerate() {
                            let progress = step_idx as f32 / total_polish as f32;
                            let active_controls: Vec<
                                &crate::pipelines::controlnet::ControlRequest,
                            > = controls
                                .iter()
                                .filter(|c| c.active_at(progress))
                                .collect();
                            latents = self.denoise_step(
                                &self.core.unet,
                                &latents,
                                timestep,
                                &text_embeddings,
                                pooled_text_sdxl.as_ref(),
                                add_time_ids_base.as_ref(),
                                &mut polish,
                                req.guidance,
                                do_cfg,
                                &active_controls,
                            )?;
                            rbar.inc(1);
                            rbar.set_message(format!("polish t={timestep}"));
                        }
                        rbar.finish_and_clear();
                    }
                }
            }

            // VAE decode + save
            let image = self.core.vae.decode(&(&latents / vae_scale)?)?;
            let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
            let image = (image * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = image.dims3()?;
            let buf = image.flatten_all()?.to_vec1::<u8>()?;
            let out_path = req.out_dir.join(format!("plakat-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
        }
        Ok(())
    }

    /// v0.12 / v0.14 tiled hi-res — MultiDiffusion-style generation at
    /// arbitrary target sizes. The UNet only ever sees
    /// `cfg.tile_size × cfg.tile_size` crops; per-step noise
    /// predictions from overlapping tiles blend via a 2D Hann window.
    /// See `pipelines::tiled` for the windowing math and tile-position
    /// generator.
    ///
    /// v0.14 phase 4: SD 1.5 / 2.1 tiled mode added. The SDXL-only
    /// micro-conditioning (`add_text_embeds` + `add_time_ids`) is
    /// skipped for non-XL variants; everything else (tile loop, Hann
    /// blending, CFG inside the tile, scheduler step on the
    /// accumulated full-canvas noise prediction) works the same.
    ///
    /// Restrictions enforced upstream in `run()`:
    ///   * No ControlNet, no SDXL refiner.
    ///
    /// `controls` parameter from `generate` is omitted entirely here —
    /// the validation in `run()` guarantees an empty stack when tiled
    /// is engaged.
    pub fn generate_tiled(
        &self,
        req: &GenRequest,
        cfg: crate::pipelines::tiled::TiledConfig,
    ) -> Result<()> {
        use crate::pipelines::tiled::{hann_window_2d, tile_positions, TilePos};

        crate::pipelines::scheduler::check_device_support(req.scheduler, &self.core.device)?;
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        // Dim sanity. The pipeline guarantees `req.width / req.height`
        // are positive multiples of 8 (CLI enforces this); we add a
        // tile-vs-canvas sanity check here.
        if req.width < cfg.tile_size || req.height < cfg.tile_size {
            anyhow::bail!(
                "--tiled: canvas {}x{} smaller than --tile-size {} — use regular \
                 generate instead of --tiled",
                req.width,
                req.height,
                cfg.tile_size,
            );
        }
        if cfg.tile_size % 8 != 0 || cfg.stride % 8 != 0 {
            anyhow::bail!(
                "--tile-size ({}) and --stride ({}) must be multiples of 8",
                cfg.tile_size,
                cfg.stride
            );
        }

        let (w, h) = (req.width as usize, req.height as usize);
        let latent_h = h / 8;
        let latent_w = w / 8;
        let tile_latent = (cfg.tile_size as usize) / 8;
        let stride_latent = (cfg.stride as usize) / 8;

        let do_cfg = req.guidance > 1.0;
        let (text_embeddings, pooled_text_sdxl) =
            self.encode_prompt(&req.prompt, &req.negative, do_cfg)?;
        // SDXL needs the pooled text embed for `add_text_embeds`; SD
        // 1.5 / 2.1 don't have one. The UNet forward accepts both as
        // `Option<&Tensor>`, so this just gates which arm runs.
        let is_xl = self.core.variant.is_xl();
        if is_xl && pooled_text_sdxl.is_none() {
            anyhow::bail!(
                "tiled SDXL: pooled text embed missing — encode_prompt should always \
                 produce one when variant.is_xl()"
            );
        }

        // 2D Hann window for blending per-tile noise predictions.
        // Same dims as a per-tile noise tensor's spatial axes; one
        // window reused for every tile, every step.
        let window = hann_window_2d(tile_latent, &self.core.device, self.core.dtype)?;
        // Spatial broadcast helper for the weight buffer — needs a
        // single-channel (1, 1, tile, tile) shape, which `window`
        // already has.
        let positions = tile_positions(latent_h, latent_w, tile_latent, stride_latent);
        let backbone_tag = if is_xl { "SDXL" } else { "SD" };
        crate::ui::progress::println(&format!(
            "  {} tiled {}: {} × {} ({}×{} latent), {} tile(s) at {}px stride {}px",
            console::style("◆").cyan().bold(),
            backbone_tag,
            w,
            h,
            latent_w,
            latent_h,
            positions.len(),
            cfg.tile_size,
            cfg.stride,
        ));

        let vae_scale: f64 = self.core.variant.vae_scale();

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random)
                & (u32::MAX as u64);
            if let Err(e) = self.core.device.set_seed(seed) {
                tracing::debug!(
                    target: "plakat",
                    "set_seed not supported ({e}); using global RNG"
                );
            }

            let mut scheduler =
                crate::pipelines::scheduler::build(req.scheduler, &self.core.cfg, req.steps)?;
            let timesteps = scheduler.timesteps().to_vec();

            // Full-size latent, scaled by init_noise_sigma to match
            // the scheduler's first step expectation.
            let mut latents = Tensor::randn(
                0f32,
                1f32,
                (1usize, 4usize, latent_h, latent_w),
                &self.core.device,
            )?
            .to_dtype(self.core.dtype)?;
            latents = (latents * scheduler.init_noise_sigma())?;

            let bar = crate::ui::progress::step_bar(
                timesteps.len() as u64,
                &format!("tiled img {}/{}", idx + 1, req.count),
            );

            for &timestep in &timesteps {
                // Accumulator + weight buffers, full-latent-sized.
                // `acc` holds Σ window·noise_pred, `weights` holds Σ window.
                let mut acc = Tensor::zeros(
                    (1, 4, latent_h, latent_w),
                    self.core.dtype,
                    &self.core.device,
                )?;
                let mut weights = Tensor::zeros(
                    (1, 1, latent_h, latent_w),
                    self.core.dtype,
                    &self.core.device,
                )?;

                for TilePos { y, x, size } in positions.iter().copied() {
                    let tile_latents = latents.narrow(2, y, size)?.narrow(3, x, size)?;

                    // Per-tile micro-conditioning (SDXL only — SD 1.5
                    // / 2.1 UNets have no `add_embedding`). For XL the
                    // `original_size` / `target_size` stay at the full
                    // canvas (the model thinks it's painting at that
                    // resolution), with `crops_coords_top_left` telling
                    // SDXL where this tile sits in the target.
                    // Diffusers' DemoFusion / Tiled-SDXL takes this
                    // approach.
                    let tile_add_time_ids = if is_xl {
                        let tile_y_px = (y * 8) as u32;
                        let tile_x_px = (x * 8) as u32;
                        let t = crate::pipelines::sdxl_unet::build_tile_add_time_ids(
                            req.height,
                            req.width,
                            tile_y_px,
                            tile_x_px,
                            &self.core.device,
                            self.core.dtype,
                        )?;
                        if do_cfg {
                            Some(Tensor::cat(&[&t, &t], 0)?)
                        } else {
                            Some(t)
                        }
                    } else {
                        None
                    };

                    let latent_in = if do_cfg {
                        Tensor::cat(&[&tile_latents, &tile_latents], 0)?
                    } else {
                        tile_latents.clone()
                    };
                    let latent_in =
                        scheduler.scale_model_input(latent_in, timestep)?;

                    let tile_noise_pred = self.core.unet.forward(
                        &latent_in,
                        timestep as f64,
                        &text_embeddings,
                        pooled_text_sdxl.as_ref(),
                        tile_add_time_ids.as_ref(),
                    )?;
                    // Merge CFG inside the tile so the accumulator
                    // sees one batch row, not two.
                    let tile_noise_pred = if do_cfg {
                        let chunks = tile_noise_pred.chunk(2, 0)?;
                        let uncond = &chunks[0];
                        let text = &chunks[1];
                        (uncond + ((text - uncond)? * req.guidance)?)?
                    } else {
                        tile_noise_pred
                    };

                    // Weight by the Hann window (broadcasts (1, 1, t, t)
                    // across the (1, 4, t, t) noise pred).
                    let weighted = tile_noise_pred.broadcast_mul(&window)?;

                    // Accumulate into the full-size buffers via
                    // slice_assign. We narrow the current acc region,
                    // add the weighted contribution, then write back.
                    let acc_region = acc.narrow(2, y, size)?.narrow(3, x, size)?;
                    let acc_updated = (acc_region + &weighted)?;
                    acc = acc.slice_assign(&[0..1, 0..4, y..y + size, x..x + size], &acc_updated)?;

                    let w_region = weights.narrow(2, y, size)?.narrow(3, x, size)?;
                    let w_updated = w_region.broadcast_add(&window)?;
                    weights = weights.slice_assign(
                        &[0..1, 0..1, y..y + size, x..x + size],
                        &w_updated,
                    )?;
                }

                // Average: noise_pred_full = acc / weights (broadcast
                // across the 4 channels).
                let noise_pred = acc.broadcast_div(&weights)?;
                latents = scheduler.step(&noise_pred, timestep, &latents)?;

                bar.inc(1);
                bar.set_message(format!("t={timestep} seed={seed}"));
            }
            bar.finish_and_clear();

            // VAE decode + save. At 4K the whole-canvas decode is the
            // memory-tightest step in the pipeline — use the tiled
            // VAE decoder (same MultiDiffusion Hann-blend math, just
            // applied at pixel resolution).
            let pre_decode = (&latents / vae_scale)?;
            // Use the same tile_latent size as the denoise tiles so
            // each VAE-decode tile matches the receptive field the
            // UNet operated on. Stride too matches.
            let vae_spin = crate::ui::progress::spinner(&format!(
                "Tiled VAE decode ({}×{} px)",
                w, h
            ));
            let image = {
                let vae = &self.core.vae;
                crate::pipelines::tiled::tile_decode_2d(
                    &pre_decode,
                    tile_latent,
                    stride_latent,
                    8, // SDXL VAE downsample factor
                    |tile| Ok(vae.decode(tile)?),
                )?
            };
            vae_spin.finish_with_message("✓ VAE decoded");
            let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
            let image = (image * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = image.dims3()?;
            let buf = image.flatten_all()?.to_vec1::<u8>()?;
            let out_path = req.out_dir.join(format!("plakat-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_step(
        &self,
        unet: &crate::pipelines::sdxl_unet::SdUNet,
        latents: &Tensor,
        timestep: usize,
        text_embeddings: &Tensor,
        // v0.11 phase 8d: pooled CLIP-G + time_ids for SDXL base's
        // add_embedding. None on SD 1.5 / SD 2.1 and on the refiner
        // pass (8d). SdUNet::Sdxl errors if these are None; SdUNet::Sd
        // ignores them — making this signature uniform across variants.
        add_text_embeds: Option<&Tensor>,
        add_time_ids: Option<&Tensor>,
        scheduler: &mut Box<dyn stable_diffusion::schedulers::Scheduler>,
        guidance: f64,
        do_cfg: bool,
        // v0.11: multi-ControlNet. Caller pre-filters to controls
        // active at this step. Empty slice = plain forward (no
        // residuals); single entry mirrors the v0.9–v0.10 behaviour;
        // 2+ entries sum residuals diffusers-style.
        active_controls: &[&crate::pipelines::controlnet::ControlRequest],
    ) -> Result<Tensor> {
        let latent_in = if do_cfg {
            Tensor::cat(&[latents, latents], 0)?
        } else {
            latents.clone()
        };
        let latent_in = scheduler.scale_model_input(latent_in, timestep)?;
        let noise_pred = if active_controls.is_empty() {
            unet.forward(
                &latent_in,
                timestep as f64,
                text_embeddings,
                add_text_embeds,
                add_time_ids,
            )?
        } else {
            let (down, mid) = crate::pipelines::controlnet::sum_controlnet_residuals(
                active_controls,
                &latent_in,
                timestep,
                text_embeddings,
                do_cfg,
                // v0.12: SDXL ControlNet now consumes the same
                // text_time micro-conditioning as the UNet.
                add_text_embeds,
                add_time_ids,
            )?;
            unet.forward_with_additional_residuals(
                &latent_in,
                timestep as f64,
                text_embeddings,
                add_text_embeds,
                add_time_ids,
                Some(&down),
                Some(&mid),
            )?
        };
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

// =====================================================================
// Tokenization + SDXL embedding helpers (used by Pipeline methods).
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

/// Penultimate CLIP-G hidden states only — for the SDXL refiner UNet whose
/// `cross_attention_dim` is 1280, not the 2048 the base UNet expects.
fn embed_g_only(
    text: &str,
    tok_g: &Tokenizer,
    cfg_g: &sdclip::Config,
    enc_g: &crate::pipelines::sdxl_clip::SdxlClipGTextTransformer,
    device: &Device,
) -> Result<Tensor> {
    let ids_g = tokenize_padded(tok_g, cfg_g, text, device)?;
    let (_final_g, hidden_g) = enc_g.forward_until_encoder_layer(&ids_g, usize::MAX, -2)?;
    Ok(hidden_g)
}

/// Encode one branch (cond or uncond) for SDXL base: concat'd CLIP-L
/// penultimate + CLIP-G penultimate for cross-attention, and the
/// pooled CLIP-G output for the UNet's `add_embedding`.
#[allow(clippy::too_many_arguments)]
fn embed_xl(
    text: &str,
    tok_l: &Tokenizer,
    tok_g: &Tokenizer,
    cfg_l: &sdclip::Config,
    cfg_g: &sdclip::Config,
    enc_l: &sdclip::ClipTextTransformer,
    enc_g: &crate::pipelines::sdxl_clip::SdxlClipGTextTransformer,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let ids_l = tokenize_padded(tok_l, cfg_l, text, device)?;
    let ids_g = tokenize_padded(tok_g, cfg_g, text, device)?;
    let (_final_l, hidden_l) = enc_l.forward_until_encoder_layer(&ids_l, usize::MAX, -2)?;
    let (hidden_g, pooled_g) = enc_g.forward_for_sdxl(&ids_g)?;
    let cat = Tensor::cat(&[&hidden_l, &hidden_g], 2)?;
    Ok((cat, pooled_g))
}

// =====================================================================
// Public entry point — single-shot wrapper for back-compat with
// `plakat generate` and any direct caller.
// =====================================================================

/// Run a t2i task. Returns the loaded `SdCore` so a follow-on step
/// (e.g. `--artefact-blend`) can reuse the same weights via
/// [`portrait::Pipeline::from_core`] instead of paying for a second
/// load. Returns `Ok(None)` when the request routed through the Flux
/// pipeline (which has no shared SdCore — Flux uses its own
/// transformer-based backbone).
pub async fn run(req: Request) -> Result<Option<std::sync::Arc<crate::pipelines::sd_core::SdCore>>> {
    let variant = Variant::detect(&req.model);

    // v0.14 phase 1a: SD3 / SD3.5 routes to the MMDiT pipeline. Phase
    // 1a is t2i only; LoRA / ControlNet / img2img are bail-loud for
    // now since those code paths don't yet know about MMDiT.
    if variant.is_sd3() {
        use crate::pipelines::sd3;
        if !req.loras.is_empty() {
            anyhow::bail!(
                "SD3 LoRAs aren't wired yet (v0.14 phase 1a t2i only). \
                 Drop --loras or switch to an SDXL / Flux model."
            );
        }
        if !req.controls.is_empty() {
            anyhow::bail!(
                "SD3 ControlNet isn't wired yet (v0.14 phase 1a t2i only). \
                 Drop --control-spec or switch to an SDXL / Flux model."
            );
        }
        if req.tiled.is_some() {
            anyhow::bail!(
                "SD3 --tiled isn't wired yet (v0.14 phase 1a t2i only)."
            );
        }
        if req.quantize_t5 {
            anyhow::bail!(
                "SD3 --quantize-t5 isn't wired yet (Flux-only in v0.13)."
            );
        }
        let sd3_variant = match variant {
            Variant::Sd3Medium => sd3::Variant::Sd3Medium,
            Variant::Sd35Medium => sd3::Variant::Sd35Medium,
            Variant::Sd35Large => sd3::Variant::Sd35Large,
            Variant::Sd35LargeTurbo => sd3::Variant::Sd35LargeTurbo,
            _ => unreachable!("is_sd3() implies one of the SD3 variants"),
        };
        sd3::run(sd3::Request {
            prompt: req.prompt,
            negative: req.negative,
            variant: sd3_variant,
            repo: resolve_repo(&req.model),
            width: req.width,
            height: req.height,
            count: req.count,
            steps: if req.steps == 28 { None } else { Some(req.steps) },
            guidance: if (req.guidance - 7.5).abs() < f64::EPSILON {
                None
            } else {
                Some(req.guidance)
            },
            seed: req.seed,
            out_dir: req.out_dir,
            device: req.device,
            // v0.15 phase 2: t2i path doesn't carry img2img args; the
            // img2img CLI dispatch (cli/img2img.rs::run_sd3_img2img)
            // builds an sd3::Request directly with these populated.
            init_image: None,
            mask: None,
            mask_feather: 0,
            mask_invert: false,
            strength: None,
            // v0.15 phase 3: SD3 LoRA support. Resolved at load time
            // via sd3_lora::merge_sd3_loras_into_weights — a tempfile
            // replaces the base MMDiT mmap for the VarBuilder.
            loras: req.loras,
            lora_scale: req.lora_scale,
        })
        .await?;
        return Ok(None);
    }

    // Flux routes to its own pipeline. v0.12: LoRAs ARE supported via
    // the new flux_lora merge path (diffusers PEFT format).
    if variant.is_flux() {
        use crate::pipelines::flux;
        let fvar = match variant {
            Variant::FluxDev => flux::Variant::Dev,
            Variant::FluxFillDev => flux::Variant::FillDev,
            Variant::FluxCannyDev => flux::Variant::CannyDev,
            Variant::FluxDepthDev => flux::Variant::DepthDev,
            _ => flux::Variant::Schnell,
        };
        // Resolve LoraSpec → ResolvedLora for Flux's API. Errors out
        // early if any LoRA file can't be fetched / opened.
        let resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> =
            if req.loras.is_empty() {
                Vec::new()
            } else {
                let s = progress::spinner("Resolving Flux LoRA file(s)");
                let mut v = Vec::with_capacity(req.loras.len());
                for spec in &req.loras {
                    v.push(spec.resolve().await?);
                }
                s.finish_with_message(format!("✓ resolved {} Flux LoRA file(s)", v.len()));
                v
            };
        // v0.12 phase 2b: Flux ControlNet routing. At most one
        // --control-spec for Flux; multi-CN is a follow-up. The
        // spec's `image=PATH` is the conditioning image (pre-rendered
        // canny / depth / etc.); `from=PATH` is rejected for Flux for
        // now because we don't ship Flux auto-annotators yet —
        // pre-rendered is the supported workflow.
        // v0.12 multi: every `--control-spec` becomes one entry in
        // the Flux ControlNet stack. Each loads its own weights and
        // contributes residuals that sum at denoise time.
        //
        // v0.13 phase 8: `--control-spec depth:from=photo.jpg` runs
        // the matching auto-annotator (canny / depth / hed / lineart /
        // openpose) and writes the result to a temp PNG that the Flux
        // pipeline then VAE-encodes as standard conditioning. The
        // tempdir holds those PNGs alive across the awaited
        // `flux::run` below — dropping it after `.await?` lets the OS
        // clean up.
        let flux_anno_dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::BF16
        };
        let flux_anno_tmp = tempfile::Builder::new()
            .prefix("plakat-flux-anno-")
            .tempdir()
            .context("creating tempdir for Flux ControlNet auto-annotator output")?;
        let mut flux_controlnets: Vec<flux::FluxControlNetLoad> = Vec::new();
        for (cn_idx, spec) in req.controls.iter().enumerate() {
            let cond_path: PathBuf = match (spec.image.as_ref(), spec.from.as_ref()) {
                (Some(p), None) => p.clone(),
                (None, Some(from_path)) => {
                    // Auto-annotate at the requested generation size so
                    // the conditioning matches the canvas dims exactly
                    // — VAE encode then needs no resize.
                    let spin = progress::spinner(&format!(
                        "Auto-annotating Flux ControlNet #{} ({})",
                        cn_idx + 1,
                        spec.kind.slug()
                    ));
                    let anno = crate::pipelines::controlnet_annotator::annotate(
                        spec.kind,
                        from_path,
                        req.width,
                        req.height,
                        &req.device,
                        flux_anno_dtype,
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "auto-annotating {} for Flux ControlNet (--control-spec {}:from={})",
                            spec.kind.slug(),
                            spec.kind.slug(),
                            from_path.display()
                        )
                    })?;
                    let out_path = flux_anno_tmp.path().join(format!(
                        "cn{}-{}.png",
                        cn_idx,
                        spec.kind.slug()
                    ));
                    write_annotator_tensor_as_png(&anno, &out_path).with_context(|| {
                        format!(
                            "writing auto-annotated {} → {}",
                            spec.kind.slug(),
                            out_path.display()
                        )
                    })?;
                    spin.finish_with_message(format!(
                        "✓ auto-annotated {} → {}",
                        spec.kind.slug(),
                        out_path.display()
                    ));
                    out_path
                }
                (Some(_), Some(_)) => anyhow::bail!(
                    "--control-spec for kind={:?}: image= and from= are mutually exclusive",
                    spec.kind
                ),
                (None, None) => anyhow::bail!(
                    "--control-spec for kind={:?}: requires image=PATH or from=PATH on Flux",
                    spec.kind
                ),
            };
            let mut cn_load = flux_controlnet_load_for(spec.kind, fvar, spec.strength)?;
            cn_load.conditioning = Some(cond_path);
            // v0.13 phase 6: thread the `--control-spec start=…:end=…`
            // gating window (or the legacy `--control-start/-end`
            // flags) into the Flux CN. Gates the CN to only contribute
            // residuals during a slice of the schedule — useful for
            // structure-only-early or structure-only-late workflows.
            cn_load.start = spec.start;
            cn_load.end = spec.end;
            flux_controlnets.push(cn_load);
        }

        flux::run(flux::Request {
            prompt: req.prompt,
            variant: fvar,
            repo: resolve_repo(&req.model),
            width: req.width,
            height: req.height,
            count: req.count,
            steps: if req.steps == 28 { None } else { Some(req.steps) },
            guidance: if (req.guidance - 7.5).abs() < f64::EPSILON {
                None
            } else {
                Some(req.guidance)
            },
            seed: req.seed,
            out_dir: req.out_dir,
            device: req.device,
            loras: resolved_loras,
            lora_scale: req.lora_scale,
            controlnets: flux_controlnets,
            // Per-CN conditioning lives on each FluxControlNetLoad now.
            conditioning: None,
            quantize_t5: req.quantize_t5,
            flux_quant_level: req.flux_quant_level.clone(),
            t5_quant_level: req.t5_quant_level.clone(),
            // `plakat generate` doesn't take inpaint / img2img inputs
            // — those flow in via `plakat img2img --model flux-…`.
            // If the user picks Fill via `generate` we'll bail loud in
            // `Pipeline::generate`.
            init_image: None,
            mask: None,
            strength: None,
            // v0.13 phase 4: route `plakat generate --tiled` to the
            // Flux pipeline. The pipeline itself bails loud if --tiled
            // is combined with ControlNet or Fill in this phase.
            tiled: req.tiled,
            // v0.14 phase 3 / 3c: thread the Redux image stack and
            // the implied `redux: true` load-time flag. When the user
            // doesn't ask for Redux, neither the SigLIP nor the
            // adapter is loaded.
            redux: !req.redux_images.is_empty(),
            redux_images: req.redux_images.clone(),
            // v0.15 phase 4: concept conditioning (Canny-dev / Depth-dev).
            // The t2i path doesn't carry this yet — pass None and let
            // the CLI dispatch the concept variants directly via a
            // dedicated flag in cli/generate.rs. The Flux pipeline
            // bails loud if a concept variant is loaded without
            // concept_conditioning set.
            concept_conditioning: req.flux_concept_image.clone(),
        })
        .await?;
        // Tempdir survives until here so the auto-annotated PNGs are
        // alive across `flux::run` (the Flux pipeline reads them inside
        // `Pipeline::generate`). Now safe to drop — the conditioning
        // tensors are already VAE-encoded and resident in the loaded
        // pipeline.
        drop(flux_anno_tmp);
        return Ok(None);
    }

    // v0.12 tiled hi-res: validate combinations early. MultiDiffusion
    // currently doesn't compose with ControlNet (we'd need to crop
    // the conditioning per tile and re-run each ControlNet inside the
    // tile loop) or the SDXL refiner (the per-step refiner switch
    // crosses the tile-blend boundary). Both are reachable follow-ups.
    //
    // v0.13 phase 4: Flux variants get tiled support via their own
    // pipeline (handled before this guard via the early `is_flux()`
    // dispatch). SD 1.5 / SD 2.1 tiled is still the lone gap.
    if req.tiled.is_some() {
        // v0.14 phase 4: SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo / Flux all
        // support tiled. SD3 is the lone remaining gap — its MMDiT
        // forward signature is different enough that the generate_tiled
        // loop would need its own dispatch.
        if variant.is_sd3() {
            anyhow::bail!(
                "--tiled isn't wired for SD3 yet (the MMDiT forward signature differs \
                 from SD UNets). Use Flux or SDXL for tiled hi-res, or skip --tiled."
            );
        }
        if !req.controls.is_empty() {
            anyhow::bail!(
                "--tiled does not yet compose with --control-spec. The \
                 conditioning would need to be cropped per tile and each \
                 ControlNet re-run inside the tile loop; tracked as a \
                 follow-up."
            );
        }
        if req.use_refiner {
            anyhow::bail!(
                "--tiled does not compose with --refiner. The refiner UNet \
                 switch mid-schedule crosses the tile-blend boundary."
            );
        }
    }

    // -- ControlNet preload (v0.9 / v0.11 multi). Owned data lives on
    //    this stack frame; `ControlRequest`s are built from references
    //    to it just before `generate` is called.
    let dtype = if matches!(req.device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    };
    let control_owned = crate::pipelines::controlnet::load_control_stack(
        &req.controls,
        &req.model,
        req.width,
        req.height,
        &req.device,
        dtype,
        None, // t2i has no fallback input image to auto-annotate.
    )
    .await?;

    let pipeline = Pipeline::load(LoadRequest {
        model: req.model,
        device: req.device,
        loras: req.loras,
        lora_scale: req.lora_scale,
        use_refiner: req.use_refiner,
    })
    .await?;

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

    let gen_req = GenRequest {
        prompt: req.prompt,
        negative: req.negative,
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
        refiner_frac: if req.use_refiner {
            Some(req.refiner_frac)
        } else {
            None
        },
    };
    match req.tiled {
        None => pipeline.generate(&gen_req, &control_reqs)?,
        Some(cfg) => pipeline.generate_tiled(&gen_req, cfg)?,
    }
    Ok(Some(pipeline.core()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // v0.13 phase 8 — annotator → PNG bridge.

    #[test]
    fn annotator_png_rejects_wrong_shape() {
        // (1, 1, H, W) is the depth pipeline's raw output before the
        // grayscale-to-RGB replicate — the bridge expects post-replicate.
        let bad = Tensor::zeros((1, 1, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        let err = write_annotator_tensor_as_png(&bad, tmp.path()).unwrap_err();
        assert!(format!("{err}").contains("(1, 3, H, W)"));
    }

    #[test]
    fn annotator_png_writes_rgb_pixels() {
        // Solid red (1.0, 0.0, 0.0) → 255-0-0 PNG. Tests that the
        // 0..1 → 0..255 scaling + channel ordering are right.
        let r = Tensor::ones((1, 1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let g = Tensor::zeros((1, 1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::zeros((1, 1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let rgb = Tensor::cat(&[&r, &g, &b], 1).unwrap();
        // `.png` suffix so `image::save` picks the right encoder.
        let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        write_annotator_tensor_as_png(&rgb, tmp.path()).unwrap();
        let read = image::open(tmp.path()).unwrap().to_rgb8();
        assert_eq!(read.dimensions(), (4, 4));
        assert_eq!(read.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(read.get_pixel(3, 3).0, [255, 0, 0]);
    }

    // v0.15 phase 4 — variant detection for the BFL concept models.

    #[test]
    fn detects_flux_canny_dev() {
        assert_eq!(Variant::detect("flux-canny-dev"), Variant::FluxCannyDev);
        assert_eq!(Variant::detect("flux1-canny-dev"), Variant::FluxCannyDev);
        assert_eq!(
            Variant::detect("black-forest-labs/FLUX.1-Canny-dev"),
            Variant::FluxCannyDev
        );
    }

    #[test]
    fn detects_flux_depth_dev() {
        assert_eq!(Variant::detect("flux-depth-dev"), Variant::FluxDepthDev);
        assert_eq!(Variant::detect("flux1-depth-dev"), Variant::FluxDepthDev);
        assert_eq!(
            Variant::detect("black-forest-labs/FLUX.1-Depth-dev"),
            Variant::FluxDepthDev
        );
    }

    #[test]
    fn concept_variants_preempt_dev() {
        // "flux-canny-dev" contains "dev" — the canny/depth checks
        // must precede the generic dev check, otherwise we'd
        // route to plain FluxDev and the 128ch `img_in` would be
        // mis-sized at load time.
        assert_eq!(Variant::detect("flux-canny-dev"), Variant::FluxCannyDev);
        assert_eq!(Variant::detect("flux-depth-dev"), Variant::FluxDepthDev);
        // Sanity: plain dev still routes to FluxDev.
        assert_eq!(Variant::detect("flux-dev"), Variant::FluxDev);
        assert_eq!(Variant::detect("flux-fill-dev"), Variant::FluxFillDev);
    }

    #[test]
    fn is_flux_includes_concept_variants() {
        assert!(Variant::FluxCannyDev.is_flux());
        assert!(Variant::FluxDepthDev.is_flux());
    }

    #[test]
    fn is_flux_concept_predicate() {
        assert!(Variant::FluxCannyDev.is_flux_concept());
        assert!(Variant::FluxDepthDev.is_flux_concept());
        assert!(!Variant::FluxDev.is_flux_concept());
        assert!(!Variant::FluxFillDev.is_flux_concept());
        assert!(!Variant::Sd15.is_flux_concept());
    }
}
