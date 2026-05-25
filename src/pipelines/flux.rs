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
//!
//! Two ways to use this module:
//!   * `flux::run(Request)` — single-shot. `plakat generate --model flux-*`
//!     goes through this path.
//!   * `Pipeline::load(...)` + repeated `Pipeline::generate(...)` — share
//!     loaded weights across many tasks. `plakat scenario` uses this for
//!     Flux models so each task doesn't re-download/re-build ~33 GB.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::Module;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    flux::{autoencoder as fae, sampling},
    quantized_t5 as qt5,
    stable_diffusion::clip as sdclip,
    t5,
};
// v0.12 phase 2a: use plakat's vendored Flux model (with the
// residual-aware forward hook) instead of candle's upstream
// flux::model. The vendored type is byte-identical to upstream when
// no residuals are passed, so the existing Flux generation path
// behaves the same.
use crate::pipelines::flux_inner as fmodel;
// v0.13 phase 1c: plakat's vendored quantized Flux (mirror of the
// BF16 vendor, but every Linear is `quantized_nn::Linear` so the GGUF
// tensors stay 4-bit until the forward dequantises them). This vendor
// re-uses the same `Config` / `EmbedNd` / helpers as `flux_inner`, and
// adds the matching `forward_with_residuals` hook so a single
// FluxControlNet can drive either backbone.
use crate::pipelines::flux_quantized_inner as qmodel;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::ui::progress;

const CLIP_EOT: u32 = 49407;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Schnell,
    Dev,
    /// v0.13 phase 2: Flux.1-Fill-dev. Same architecture as Dev except
    /// `img_in` has 384 input channels (noise + masked-latent + mask).
    /// Always runs in inpainting mode — caller must supply an init
    /// image + mask.
    FillDev,
    /// v0.15 phase 4: Flux.1-Canny-dev. Full BFL "concept" checkpoint
    /// with canny-edge conditioning baked into `img_in` (128 channels =
    /// 64 noise + 64 conditioning latent). Caller supplies a canny
    /// edge map (or a photo that gets auto-annotated).
    CannyDev,
    /// v0.15 phase 4: Flux.1-Depth-dev. Same shape as Canny-dev but
    /// trained on depth maps instead of canny edges.
    DepthDev,
    /// v0.18: Flux.1-Kontext-dev. Image-editing variant. Architecture
    /// matches Dev (img_in stays at 64 channels); the difference is
    /// at the pipeline level — a reference image is VAE-encoded and
    /// its tokens are sequence-concatenated onto the noise tokens
    /// with `image_ids[..., 0] = 1` as the positional marker.
    KontextDev,
}

impl Variant {
    pub fn is_dev(self) -> bool {
        matches!(
            self,
            Self::Dev | Self::FillDev | Self::CannyDev | Self::DepthDev | Self::KontextDev
        )
    }
    /// v0.13 phase 2: does this variant expect inpainting inputs?
    pub fn is_fill(self) -> bool {
        matches!(self, Self::FillDev)
    }
    /// v0.15 phase 4: BFL "concept" variants with conditioning baked
    /// into a 128-channel `img_in`. Canny-dev or Depth-dev.
    pub fn is_concept(self) -> bool {
        matches!(self, Self::CannyDev | Self::DepthDev)
    }
    /// v0.18: Flux.1-Kontext-dev expects a reference image fed via
    /// sequence-concat rather than channel-concat. Distinct from the
    /// `is_concept` variants (Canny/Depth) which widen `img_in`.
    pub fn is_kontext(self) -> bool {
        matches!(self, Self::KontextDev)
    }
    /// v0.15 phase 4: which concept conditioner does the variant expect?
    /// `None` for non-concept variants.
    pub fn concept_kind(self) -> Option<&'static str> {
        match self {
            Self::CannyDev => Some("canny"),
            Self::DepthDev => Some("depth"),
            _ => None,
        }
    }
    fn main_filename(self) -> &'static str {
        match self {
            Self::Schnell => "flux1-schnell.safetensors",
            Self::Dev => "flux1-dev.safetensors",
            Self::FillDev => "flux1-fill-dev.safetensors",
            Self::CannyDev => "flux1-canny-dev.safetensors",
            Self::DepthDev => "flux1-depth-dev.safetensors",
            Self::KontextDev => "flux1-kontext-dev.safetensors",
        }
    }
    fn t5_seq_len(self) -> usize {
        match self {
            Self::Schnell => 256,
            // Fill / Canny / Depth / Kontext all use the same
            // 512-token T5 budget as Dev.
            Self::Dev
            | Self::FillDev
            | Self::CannyDev
            | Self::DepthDev
            | Self::KontextDev => 512,
        }
    }
    fn flux_config(self) -> fmodel::Config {
        match self {
            Self::Schnell => fmodel::Config::schnell(),
            Self::Dev => fmodel::Config::dev(),
            Self::FillDev => fmodel::Config::fill_dev(),
            // Canny + Depth share the 128-channel `img_in` config.
            Self::CannyDev | Self::DepthDev => fmodel::Config::canny_or_depth_dev(),
            // Kontext keeps `img_in` at 64 — the difference is at
            // the seq-concat level, handled by the pipeline.
            Self::KontextDev => fmodel::Config::kontext_dev(),
        }
    }
    fn ae_config(self) -> fae::Config {
        match self {
            // Fill / Canny / Depth / Kontext all share Dev's autoencoder.
            Self::Schnell => fae::Config::schnell(),
            Self::Dev
            | Self::FillDev
            | Self::CannyDev
            | Self::DepthDev
            | Self::KontextDev => fae::Config::dev(),
        }
    }
    pub fn default_guidance(self) -> f64 {
        match self {
            Self::Schnell => 1.0,
            Self::Dev => 3.5,
            // BFL's Fill model card recommends guidance ~30 (much
            // higher than standard Flux.1-dev's 3.5) — the mask signal
            // needs a stronger guidance to actually respect the
            // conditioning. Callers can override via `--guidance`.
            Self::FillDev => 30.0,
            // BFL's Canny + Depth model cards recommend guidance ~30
            // for the same reason — the conditioning latent needs
            // strong guidance to actually steer the output.
            Self::CannyDev | Self::DepthDev => 30.0,
            // Kontext's model card recommends guidance 3.5 (same as
            // Dev) — the reference signal flows through cross-attention
            // on the sequence-concatenated tokens rather than via a
            // channel-concat conditioning latent, so standard Dev
            // guidance is sufficient.
            Self::KontextDev => 3.5,
        }
    }
    pub fn default_steps(self) -> usize {
        match self {
            Self::Schnell => 4,
            Self::Dev
            | Self::FillDev
            | Self::CannyDev
            | Self::DepthDev
            | Self::KontextDev => 28,
        }
    }
}

// =====================================================================
// Single-shot request type — back-compat with the existing entry point.
// =====================================================================

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
    /// v0.12: Flux LoRAs (already resolved to local safetensors paths
    /// by the caller). Empty disables LoRA merging.
    pub loras: Vec<crate::pipelines::lora::ResolvedLora>,
    pub lora_scale: f32,
    /// v0.12: Flux ControlNet stack. Each entry is loaded
    /// independently; residuals sum at denoise time.
    pub controlnets: Vec<FluxControlNetLoad>,
    /// Back-compat for single-CN callers — applies to the first
    /// loaded CN. Per-CN paths in `controlnets[i].conditioning`
    /// override this for `i >= 1`.
    pub conditioning: Option<PathBuf>,
    /// v0.13 phase 1b: quantize T5-XXL via city96's GGUF mirror.
    pub quantize_t5: bool,
    /// v0.13 phase 5: GGUF quant level for the Flux transformer
    /// (e.g. `"Q4_K_S"`, `"Q5_K_M"`, `"Q8_0"`, `"F16"`). `None` →
    /// `"Q4_K_S"`. Ignored on BF16 (`--model flux-dev|flux-schnell`).
    pub flux_quant_level: Option<String>,
    /// v0.13 phase 5: GGUF quant level for the T5-XXL encoder. `None`
    /// → `"Q4_K_M"`. Ignored unless `quantize_t5` is `true`.
    pub t5_quant_level: Option<String>,
    /// v0.14 phase 3 / 3c: zero or more reference images for Flux
    /// Redux conditioning. Each entry encodes through SigLIP + the
    /// Redux adapter, scales by `weight`, then seq-concats 729 tokens
    /// onto T5's hidden state. Empty disables Redux. Up to 4 images
    /// supported (cap is a soft attention-cost guardrail — see
    /// `Pipeline::generate`'s validation).
    pub redux_images: Vec<crate::pipelines::flux_redux::ReduxSpec>,
    /// v0.14 phase 3: see `LoadRequest::redux`.
    pub redux: bool,
    /// v0.13 phase 2: Flux.1-Fill-dev inputs.
    pub init_image: Option<PathBuf>,
    pub mask: Option<PathBuf>,
    /// v0.13 phase 3: Flux img2img strength in `[0, 1]`. Only used when
    /// `init_image` is `Some` AND the variant is not Fill (Fill ignores
    /// strength — its mask controls what changes). `None` falls back to
    /// the pipeline default (~0.85). 1.0 ≈ ignore the init image
    /// entirely; 0.0 ≈ no change.
    pub strength: Option<f32>,
    /// v0.15 phase 4: conditioning image for Flux.1-Canny-dev /
    /// Flux.1-Depth-dev. Pre-rendered canny edges or depth map.
    /// Required for the concept variants; ignored otherwise.
    pub concept_conditioning: Option<PathBuf>,
    /// v0.13 phase 4: tiled (MultiDiffusion-style) denoise. When set,
    /// the full canvas is split into overlapping `tile_size`-pixel
    /// windows; each step runs Flux per-tile and blends noise
    /// predictions with a Hann window. Lets Flux produce 2K-4K
    /// outputs without exceeding the model's working resolution per
    /// pass. Rejects ControlNet and Fill in this first cut.
    pub tiled: Option<crate::pipelines::tiled::TiledConfig>,
    /// v0.18 phase 2b: opt-in aspect-bucket snap for Flux Kontext.
    /// When `true` AND `variant.is_kontext()`, the requested
    /// (width, height) is snapped to the nearest of 17
    /// BFL-recommended Kontext resolutions before VAE encoding —
    /// matching diffusers' default behaviour. Ignored on every
    /// other variant.
    pub kontext_bucket: bool,
}

// =====================================================================
// Pipeline: load once, generate many.
// =====================================================================

pub struct LoadRequest {
    pub variant: Variant,
    pub repo: String,
    pub device: Device,
    /// v0.12: Flux LoRAs to merge into the transformer at load time.
    /// Empty for the original Flux behaviour. Supports diffusers PEFT
    /// format only in this phase (see `pipelines::flux_lora`).
    pub loras: Vec<crate::pipelines::lora::ResolvedLora>,
    /// Global multiplier applied on top of each LoRA's per-file scale.
    pub lora_scale: f32,
    /// v0.12: zero or more Flux ControlNets. Empty disables the
    /// ControlNet path; one entry is the v0.12 phase 2b behaviour;
    /// two+ stacks residuals (multi-Flux-ControlNet, v0.12 multi).
    pub controlnets: Vec<FluxControlNetLoad>,
    /// v0.13 phase 1b: when `true`, load T5-XXL as a quantized GGUF
    /// (~3 GB instead of ~10 GB BF16). Combined with `--model
    /// flux-*-gguf` the total Flux footprint drops from ~17 GB to
    /// ~10 GB — fits 12 GB consumer GPUs. Only meaningful when the
    /// transformer itself is quantized (loud-fail otherwise).
    pub quantize_t5: bool,
    /// v0.13 phase 5: see `Request::flux_quant_level`.
    pub flux_quant_level: Option<String>,
    /// v0.13 phase 5: see `Request::t5_quant_level`.
    pub t5_quant_level: Option<String>,
    /// v0.14 phase 3: when `true`, load SigLIP-so400m + the Flux
    /// Redux adapter at load time so per-`generate` calls can pass
    /// `redux_image`. Loading is lazy here — empty when the caller
    /// doesn't intend to use Redux, since SigLIP + adapter add
    /// ~1 GB of memory and a noticeable load delay.
    pub redux: bool,
}

/// v0.13 phase 5: city96's published GGUF quant levels for the Flux
/// transformer. Anything else gets rejected with this list in the
/// error message — saves a long HF round-trip on typos.
///
/// Source: https://huggingface.co/city96/FLUX.1-dev-gguf/tree/main
pub const FLUX_QUANT_LEVELS: &[&str] = &[
    "F16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_1",
    "Q5_0",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_1",
    "Q4_0",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
];

/// v0.13 phase 5: city96's published GGUF quant levels for the T5-XXL
/// encoder. Slightly different set from Flux (no Q4_0/Q4_1/Q5_0/Q5_1,
/// has Q3_K_L).
///
/// Source: https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/tree/main
pub const T5_QUANT_LEVELS: &[&str] = &[
    "F32",
    "F16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q4_K_M",
    "Q4_K_S",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
];

fn validate_quant_level(level: &str, allowed: &[&str], component: &str) -> Result<()> {
    if allowed.iter().any(|l| l.eq_ignore_ascii_case(level)) {
        Ok(())
    } else {
        bail!(
            "{component} quant level '{level}' isn't published by city96. \
             Supported: {}",
            allowed.join(", ")
        )
    }
}

/// Flux ControlNet weight repo + config. The actual model load happens
/// inside `Pipeline::load`. Distinct from `flux_controlnet::Config` so
/// the user-facing API stays narrow.
#[derive(Debug, Clone)]
pub struct FluxControlNetLoad {
    pub repo: String,
    pub file: String,
    pub cfg: crate::pipelines::flux_controlnet::Config,
    /// `controlnet_conditioning_scale` (diffusers default 1.0).
    pub scale: f32,
    /// Union ControlNet mode index. Required when `cfg.num_mode` is
    /// `Some`; ignored otherwise. Specialised CNs leave this `None`.
    pub mode: Option<u32>,
    /// Conditioning image path (pre-rendered canny / depth / etc.).
    /// Each ControlNet in a multi-CN stack carries its own path.
    pub conditioning: Option<PathBuf>,
    /// v0.13 phase 6: gating window in `[0, 1]`. `start=0.0` means the
    /// CN engages from the first step; `end=1.0` means it stays active
    /// to the end. `start=0.0, end=0.4` keeps the CN's structure pull
    /// only in the early high-noise steps — common pattern when you
    /// want geometry from the conditioner but free composition later.
    /// Defaults: `start=0.0, end=1.0` (full schedule).
    pub start: f32,
    pub end: f32,
}

pub struct GenRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    /// v0.12 phase 2b: optional path to a conditioning image. When
    /// the pipeline has a ControlNet loaded AND this is `Some`, the
    /// image is VAE-encoded + packed and threaded into the per-step
    /// denoise. `None` skips the ControlNet pass (residuals come out
    /// to zero) — useful for back-compat callers that don't know
    /// about ControlNet.
    pub conditioning: Option<PathBuf>,
    /// v0.13 phase 2: inpainting inputs for `Variant::FillDev`. Both
    /// fields are required when the loaded model is Fill; ignored
    /// otherwise. `init_image` is the source image; `mask` is a single-
    /// channel image where white (≥128) marks pixels to inpaint and
    /// black leaves them. Both are resized to (`width`, `height`)
    /// before encoding.
    pub init_image: Option<PathBuf>,
    pub mask: Option<PathBuf>,
    /// v0.13 phase 3: img2img strength. See `Request::strength`.
    pub strength: Option<f32>,
    /// v0.15 phase 4: conditioning for concept variants. See
    /// `Request::concept_conditioning`.
    pub concept_conditioning: Option<PathBuf>,
    /// v0.13 phase 4: tiled denoise config. See `Request::tiled`.
    pub tiled: Option<crate::pipelines::tiled::TiledConfig>,
    /// v0.14 phase 3 / 3c: reference images for Flux Redux. See
    /// `Request::redux_images`.
    pub redux_images: Vec<crate::pipelines::flux_redux::ReduxSpec>,
    /// v0.18 phase 2b: opt-in Kontext aspect-bucket snap. See
    /// `Request::kontext_bucket`.
    pub kontext_bucket: bool,
}

pub struct Pipeline {
    pub variant: Variant,
    /// Resolved HF repo id this pipeline was loaded from.
    #[allow(dead_code)]
    pub repo: String,
    device: Device,
    dtype: DType,
    clip_text: sdclip::ClipTextTransformer,
    clip_tok: Tokenizer,
    clip_cfg: sdclip::Config,
    // T5EncoderModel::forward needs &mut self (KV cache), so generate is
    // &mut self too. The scenario loop is sequential so this is fine.
    // v0.13 phase 1b: BF16 or Quantized via the T5Backbone enum.
    t5_enc: T5Backbone,
    t5_tok: Tokenizer,
    flux_model: FluxBackbone,
    ae_model: fae::AutoEncoder,
    /// v0.12 phase 2b + v0.12 multi: zero or more Flux ControlNets.
    /// At denoise time each loaded CN runs once per step with its
    /// own conditioning, mode, and scale; the resulting residuals
    /// are summed per-block before being fed to the main Flux's
    /// `forward_with_residuals`. Empty disables the ControlNet path
    /// entirely.
    controlnets: Vec<LoadedFluxControlNet>,
    /// v0.14 phase 3: optional Redux encoder for image-conditioned
    /// Flux. `Some` when `LoadRequest::redux` was true; per-call
    /// `redux_image` activates conditioning by feeding the encoded
    /// tokens through `encode_prompt`.
    redux_encoder: Option<crate::pipelines::flux_redux::ReduxEncoder>,
}

/// One element of the Pipeline's ControlNet stack — the loaded
/// network plus the user knobs (`scale`, `mode`) it was loaded with.
/// The conditioning image is stored alongside per-`generate` call
/// since it depends on width / height.
/// v0.13: which Flux backbone the pipeline is running on.
/// `Bf16` is plakat's vendored BF16 Flux with the residual hook;
/// `Quantized` is plakat's vendored quantized Flux loaded from a
/// GGUF file (~7 GB for Q4_K_S vs ~24 GB BF16). Both variants expose
/// the same `forward_with_residuals` API, so ControlNet composes on
/// either backbone. v0.13 phase 1e: LoRAs also compose on the
/// quantized backbone — affected Linears are dequantized once at
/// load time, merged with deltas, and substituted as dense; the rest
/// of the model stays 4-bit.
///
/// v0.14 phase 2d: `Nf4` is plakat's vendored NF4 backbone — weights
/// stay packed at 4-bit (bnb / NormalFloat-4) and dequantize per
/// forward call. ~6 GB transformer at inference. Slower than the
/// GGUF backbone (no kernel-fused dequant+matmul) but a real 4×
/// weight-memory savings vs BF16. ControlNet / LoRA composition
/// with NF4 isn't wired yet — Pipeline::load bails for those combos.
pub enum FluxBackbone {
    Bf16(fmodel::Flux),
    Quantized(qmodel::Flux),
    Nf4(crate::pipelines::flux_nf4_inner::Flux),
}

impl FluxBackbone {
    /// v0.15 phase 7b-7: dispatch the per-task runtime LoRA stack to
    /// the underlying backbone. Each Flux variant (BF16 / GGUF / NF4)
    /// implements `apply_loras` with identical semantics — path-keyed
    /// `LoraSpec` maps replace the active slot stack on every
    /// registered LoraLinear. Returns the count of Linears updated.
    pub fn apply_loras(
        &self,
        specs: std::collections::HashMap<
            String,
            Vec<crate::pipelines::lora_linear::LoraSpec>,
        >,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> Result<usize> {
        let r = match self {
            FluxBackbone::Bf16(net) => net.apply_loras(specs, dtype, device),
            FluxBackbone::Quantized(net) => net.apply_loras(specs, dtype, device),
            FluxBackbone::Nf4(net) => net
                .apply_loras(specs, dtype, device)
                .map_err(|e| candle_core::Error::Msg(format!("{e}"))),
        };
        r.map_err(|e| anyhow::anyhow!("Flux apply_loras: {e}"))
    }

    /// v0.15 phase 7b-7: clear every runtime LoRA on the backbone.
    /// Called at end-of-task in scenarios so the next task starts
    /// from the scenario-level merged baseline (no per-task LoRA
    /// bleed across tasks).
    pub fn clear_all_loras(&self) -> Result<()> {
        match self {
            FluxBackbone::Bf16(net) => net
                .clear_all_loras()
                .map_err(|e| anyhow::anyhow!("Flux BF16 clear: {e}")),
            FluxBackbone::Quantized(net) => net
                .clear_all_loras()
                .map_err(|e| anyhow::anyhow!("Flux GGUF clear: {e}")),
            FluxBackbone::Nf4(net) => net
                .clear_all_loras()
                .map_err(|e| anyhow::anyhow!("Flux NF4 clear: {e}")),
        }
    }
}

/// v0.13 phase 1b: T5-XXL encoder backbone. The T5 owns a KV cache
/// internally so both variants take `&mut self` for forward — wrap
/// in an enum so dispatch in `encode_prompt` is a thin match.
pub enum T5Backbone {
    Bf16(t5::T5EncoderModel),
    Quantized(qt5::T5EncoderModel),
}

impl T5Backbone {
    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Bf16(t) => Ok(t.forward(input_ids)?),
            Self::Quantized(t) => Ok(t.forward(input_ids)?),
        }
    }
}

pub struct LoadedFluxControlNet {
    pub net: crate::pipelines::flux_controlnet::FluxControlNet,
    pub scale: f32,
    pub mode: Option<u32>,
    /// Pending conditioning image — resolved here so the spec is
    /// fully self-contained when we hand it to `generate`.
    pub conditioning_path: Option<PathBuf>,
    /// v0.13 phase 6: step-gating window (see `FluxControlNetLoad`).
    pub start: f32,
    pub end: f32,
}

impl LoadedFluxControlNet {
    /// `true` when this CN should contribute residuals at the given
    /// schedule progress (0.0 at the first step, 1.0 just past the
    /// last). Same `[start, end)` half-open convention plakat uses for
    /// SD ControlNet's `active_at`.
    pub fn active_at(&self, progress: f32) -> bool {
        progress >= self.start && progress < self.end
    }
}

impl Pipeline {
    /// v0.13 phase 10: re-tune an already-loaded ControlNet's per-call
    /// parameters between `generate` calls. Scenarios load Union Pro v2
    /// once at startup and then vary `(mode, scale, [start, end])` per
    /// task — this is the bridge for that pattern without forcing a
    /// reload of the ~600 MB CN weights per task.
    ///
    /// Returns an error when `idx` is out of bounds, when `start`/`end`
    /// land outside `[0, 1]`, or when `start >= end`.
    pub fn set_controlnet_call_params(
        &mut self,
        idx: usize,
        mode: Option<u32>,
        scale: f32,
        start: f32,
        end: f32,
    ) -> Result<()> {
        if !(0.0..=1.0).contains(&start) || !(0.0..=1.0).contains(&end) {
            bail!(
                "set_controlnet_call_params: start/end must be in [0, 1] (got {start}, {end})"
            );
        }
        if start >= end {
            bail!(
                "set_controlnet_call_params: start ({start}) must be < end ({end})"
            );
        }
        let n = self.controlnets.len();
        let cn = self
            .controlnets
            .get_mut(idx)
            .ok_or_else(|| anyhow!("ControlNet index {idx} out of bounds (have {n})"))?;
        cn.mode = mode;
        cn.scale = scale;
        cn.start = start;
        cn.end = end;
        Ok(())
    }

    /// `true` when this Pipeline has at least one loaded ControlNet.
    /// Used by callers that build a CN stack only when needed.
    pub fn has_controlnets(&self) -> bool {
        !self.controlnets.is_empty()
    }

    /// Count of loaded ControlNets. Lets per-task callers (e.g.,
    /// scenarios) check how many slots are available before slot-
    /// indexed mutations.
    pub fn controlnet_count(&self) -> usize {
        self.controlnets.len()
    }

    /// v0.15 phase 7b-7: borrow the active FluxBackbone for runtime
    /// LoRA dispatch. Callers (scenarios applying per-task LoRA)
    /// use this to call `apply_loras` / `clear_all_loras` between
    /// `generate` calls without holding `&mut Pipeline`.
    pub fn backbone(&self) -> &FluxBackbone {
        &self.flux_model
    }

    /// v0.15 phase 7b-7: convenience — current runtime dtype the
    /// backbone uses (BF16 on GPU, F32 on CPU). Used by
    /// `apply_loras` to cast LoRA A/B matrices.
    pub fn dtype(&self) -> candle_core::DType {
        self.dtype
    }

    /// v0.15 phase 7b-7: convenience — current device. Used to
    /// build padded LoRA-B tensors.
    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    /// v0.13 phase 11: swap a loaded ControlNet's conditioning image
    /// path between `generate` calls. `None` clears the path so the
    /// CN contributes no residuals on the next call (used when a
    /// task has fewer CNs than the scenario's max slot count).
    pub fn set_controlnet_conditioning(
        &mut self,
        idx: usize,
        path: Option<PathBuf>,
    ) -> Result<()> {
        let n = self.controlnets.len();
        let cn = self
            .controlnets
            .get_mut(idx)
            .ok_or_else(|| anyhow!("ControlNet index {idx} out of bounds (have {n})"))?;
        cn.conditioning_path = path;
        Ok(())
    }

    /// Download + load everything Flux needs. ~33 GB on first run.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        // Flux was trained in BF16 and its transformer's wide intermediates
        // (hidden=3072, intermediate=12288) regularly exceed F16's ±65504
        // range, producing NaN/Inf that propagate to all-black output. BF16
        // has F32's range with F16's memory footprint and is well-supported
        // on CUDA + Metal in candle 0.8.
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::BF16
        };

        // v0.13: detect quantized GGUF mode. city96 / any repo with
        // "gguf" in the id ships only the transformer in GGUF form;
        // AE + text encoders still come from the original BFL repo
        // ("donor"). Phase 1c unblocks ControlNet on the quantized
        // backbone; phase 1e unblocks LoRAs via selective dequant
        // overrides — so both compose now.
        let is_gguf = req.repo.to_lowercase().contains("gguf");
        // v0.14 phase 2d: NF4 detection. "nf4" or "bnb-nf4" in the
        // repo id routes through the NF4 vendor (flux_nf4_inner).
        // Like GGUF, NF4 packs ship only the transformer; AE + text
        // encoders come from the BFL donor.
        let is_nf4 = !is_gguf
            && (req.repo.to_lowercase().contains("nf4")
                || req.repo.to_lowercase().contains("bnb"));
        // Quantized T5 only makes sense when paired with a quantized
        // transformer — running BF16 transformer + Q4 T5 loses T5
        // quality without saving meaningful memory.
        if req.quantize_t5 && !is_gguf {
            bail!(
                "--quantize-t5 requires a GGUF Flux model (e.g. --model flux-dev-gguf). \
                 Pairing quantized T5 with a BF16 transformer wastes T5 quality \
                 without unlocking the memory budget that needs it."
            );
        }
        // v0.14 phase 2d → v0.15 phase 1: NF4 + LoRA composes (v0.14
        // phase 8b) and NF4 + ControlNet composes (v0.15 phase 1 —
        // forward_with_residuals added to the NF4 vendor mirrors the
        // GGUF path). The only remaining NF4 bail is Fill, which
        // needs a distinct loader path (Fill's `img_in` is 384ch and
        // the NF4 pack would need a matching wider quant — none
        // shipped yet).
        if is_nf4 && matches!(req.variant, Variant::FillDev) {
            bail!(
                "NF4 Flux Fill isn't supported. Use BF16 (flux-fill-dev) or GGUF \
                 (flux-fill-dev-gguf) instead."
            );
        }
        // v0.18: no upstream NF4 pack ships for Kontext at time of
        // writing. GGUF is supported via unsloth/FLUX.1-Kontext-dev-GGUF
        // (v0.18 Kontext phase 3) — bail stays for NF4 only.
        if is_nf4 && matches!(req.variant, Variant::KontextDev) {
            bail!(
                "NF4 Flux Kontext isn't supported (no upstream NF4 pack ships yet). \
                 Use BF16 (--model flux-kontext-dev) or GGUF \
                 (--model flux-kontext-dev-gguf) instead."
            );
        }
        let donor_repo: String = if is_gguf || is_nf4 {
            match req.variant {
                Variant::Dev => "black-forest-labs/FLUX.1-dev".to_string(),
                Variant::Schnell => "black-forest-labs/FLUX.1-schnell".to_string(),
                Variant::FillDev => "black-forest-labs/FLUX.1-Fill-dev".to_string(),
                // v0.15 phase 4: no community GGUF / NF4 packs for the
                // concept variants exist yet; the bail in Pipeline::load
                // for is_gguf / is_nf4 + concept variant short-circuits
                // before we reach this match. Keep the arms exhaustive.
                Variant::CannyDev => "black-forest-labs/FLUX.1-Canny-dev".to_string(),
                Variant::DepthDev => "black-forest-labs/FLUX.1-Depth-dev".to_string(),
                // v0.18 Kontext phase 3: BF16 donor for LoRA dequant.
                // Kontext shares the Dev architecture (img_in stays 64ch)
                // so Flux LoRAs that target Dev layer names compose
                // unchanged; the donor's role is to supply the
                // full-precision weights that selective-dequant copies
                // for affected Linear layers.
                Variant::KontextDev => "black-forest-labs/FLUX.1-Kontext-dev".to_string(),
            }
        } else {
            req.repo.clone()
        };

        // ---------- download weights ----------
        let dl = progress::spinner(&format!("Downloading weights for {}", req.repo));
        // Transformer: GGUF when quantized, NF4 safetensors when nf4,
        // BF16 safetensors otherwise.
        let main_path = if is_gguf {
            // v0.13 phase 5: user can override the GGUF quant level
            // (Q2_K..F16). Default Q4_K_S keeps the v0.13 phase 1
            // memory profile.
            let level = req
                .flux_quant_level
                .as_deref()
                .unwrap_or("Q4_K_S")
                .to_string();
            validate_quant_level(&level, FLUX_QUANT_LEVELS, "Flux GGUF")?;
            let stem = match req.variant {
                Variant::Dev => "flux1-dev",
                Variant::Schnell => "flux1-schnell",
                Variant::FillDev => "flux1-fill-dev",
                // No GGUF for concept variants — gated below in
                // is_gguf check, this arm is unreachable in practice.
                Variant::CannyDev => "flux1-canny-dev",
                Variant::DepthDev => "flux1-depth-dev",
                // v0.18 Kontext phase 3: matches unsloth's filename
                // convention in unsloth/FLUX.1-Kontext-dev-GGUF
                // (`flux1-kontext-dev-Q4_K_M.gguf` etc.).
                Variant::KontextDev => "flux1-kontext-dev",
            };
            let gguf_file = format!("{stem}-{level}.gguf");
            crate::hf::download::get_file(&req.repo, &gguf_file)
                .await
                .with_context(|| format!("{} from {}", gguf_file, req.repo))?
        } else if is_nf4 {
            // v0.14 phase 2d: NF4 packs typically ship as a single
            // file at the repo root. lllyasviel's pack is
            // `flux1-dev-bnb-nf4-v2.safetensors`; other community
            // packs vary by name. Try the known-good names in order;
            // fall back to the first .safetensors at the root if
            // none match.
            crate::hf::download::get_first_of(&[
                (&req.repo, "flux1-dev-bnb-nf4-v2.safetensors"),
                (&req.repo, "flux1-dev-bnb-nf4.safetensors"),
                (&req.repo, "flux1-dev-nf4.safetensors"),
                (&req.repo, "diffusion_pytorch_model.safetensors"),
            ])
            .await
            .with_context(|| {
                format!(
                    "locating NF4 Flux pack in {}. Plakat tries the lllyasviel naming \
                     convention (`flux1-dev-bnb-nf4-v2.safetensors` and similar). If \
                     the repo uses a different filename, point `--model` at the full \
                     HF repo id and ensure the pack matches the bitsandbytes-NF4 \
                     layout (per-weight `.absmax` companions, no double quant).",
                    req.repo
                )
            })?
        } else {
            crate::hf::download::get_file(&req.repo, req.variant.main_filename())
                .await
                .with_context(|| format!("{}", req.variant.main_filename()))?
        };
        // AE + text encoders + tokenizers come from the donor repo
        // (= req.repo for BF16 mode, original BFL repo for GGUF mode).
        let ae_path = crate::hf::download::get_file(&donor_repo, "ae.safetensors").await?;

        let clip_weights = crate::hf::download::get_first_of(&[
            (&donor_repo, "text_encoder/model.fp16.safetensors"),
            (&donor_repo, "text_encoder/model.safetensors"),
        ])
        .await?;
        let clip_tokenizer = crate::hf::download::get_first_of(&[
            (&donor_repo, "tokenizer/tokenizer.json"),
            ("openai/clip-vit-large-patch14", "tokenizer.json"),
        ])
        .await?;

        // v0.13 phase 1b: when --quantize-t5 is on, the T5 encoder
        // comes from city96's GGUF mirror; otherwise the standard
        // BF16 sharded safetensors from the donor repo. config.json
        // and the tokenizer come from the donor either way (city96
        // T5 mirror doesn't ship them).
        let t5_gguf_path = if req.quantize_t5 {
            // v0.13 phase 5: user can override the T5 quant level too.
            // Default Q4_K_M keeps the v0.13 phase 1b memory profile.
            let level = req
                .t5_quant_level
                .as_deref()
                .unwrap_or("Q4_K_M")
                .to_string();
            validate_quant_level(&level, T5_QUANT_LEVELS, "T5 GGUF")?;
            let t5_file = format!("t5-v1_1-xxl-encoder-{level}.gguf");
            Some(
                crate::hf::download::get_file("city96/t5-v1_1-xxl-encoder-gguf", &t5_file)
                    .await
                    .with_context(|| format!("downloading T5-XXL GGUF ({level})"))?,
            )
        } else {
            None
        };
        let (t5_shard1, t5_shard2) = if req.quantize_t5 {
            (None, None)
        } else {
            let s1 = crate::hf::download::get_file(
                &donor_repo,
                "text_encoder_2/model-00001-of-00002.safetensors",
            )
            .await?;
            let s2 = crate::hf::download::get_file(
                &donor_repo,
                "text_encoder_2/model-00002-of-00002.safetensors",
            )
            .await?;
            (Some(s1), Some(s2))
        };
        let t5_config_path =
            crate::hf::download::get_file(&donor_repo, "text_encoder_2/config.json").await?;
        let t5_tokenizer = crate::hf::download::get_file(&donor_repo, "tokenizer_2/tokenizer.json")
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
        let t5_enc = if let Some(gguf_path) = t5_gguf_path.as_ref() {
            // qt5 has its own Config type — same JSON schema as t5::Config
            // but a distinct Rust type. Both parse from the same
            // config.json the donor ships.
            let qt5_cfg: qt5::Config = serde_json::from_str(&t5_cfg_str).with_context(
                || format!("parse T5 (quantized) config from {}", t5_config_path.display()),
            )?;
            let qvb = QVarBuilder::from_gguf(gguf_path, &req.device).with_context(
                || format!("loading T5 GGUF from {}", gguf_path.display()),
            )?;
            T5Backbone::Quantized(qt5::T5EncoderModel::load(qvb, &qt5_cfg)?)
        } else {
            let t5_cfg: t5::Config = serde_json::from_str(&t5_cfg_str)
                .with_context(|| format!("parse T5 config from {}", t5_config_path.display()))?;
            let shard1 = t5_shard1.as_ref().expect("BF16 path keeps shard1");
            let shard2 = t5_shard2.as_ref().expect("BF16 path keeps shard2");
            let t5_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[shard1, shard2], dtype, &req.device)?
            };
            T5Backbone::Bf16(t5::T5EncoderModel::load(t5_vb, &t5_cfg)?)
        };
        let t5_tok = Tokenizer::from_file(&t5_tokenizer)
            .map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
        build.finish_with_message("✓ text encoders ready");

        // ---------- merge Flux LoRAs ----------
        // BF16 path (v0.12): merge LoRA deltas into a temporary
        // safetensors file, then mmap that. Defeats nothing because
        // the base is already dense BF16.
        // GGUF path (v0.13 phase 1e): merging into 4-bit storage would
        // either defeat the memory win (rewriting to BF16 of everything)
        // or compound quantization noise (re-quantize after delta). Instead,
        // we dequantize **only** the LoRA-targeted Linears, apply deltas
        // densely, and feed them into the quantized Flux as
        // `QMatMul::Tensor` overrides — the un-targeted ~95% of the
        // model stays 4-bit.
        let (effective_main_path, lora_tmp) = if req.loras.is_empty() || is_gguf {
            (main_path.clone(), None)
        } else {
            let spin = progress::spinner(&format!(
                "Merging {} Flux LoRA(s) into transformer",
                req.loras.len()
            ));
            let tmp = tempfile::Builder::new()
                .prefix("plakat-flux-merged-")
                .suffix(".safetensors")
                .tempfile()?;
            let (modified, total) =
                crate::pipelines::flux_lora::merge_flux_loras_into_weights(
                    &main_path,
                    tmp.path(),
                    &req.loras,
                    req.lora_scale,
                    &req.device,
                )?;
            spin.finish_with_message(format!(
                "✓ Flux LoRA merged ({modified}/{total} target groups)"
            ));
            let p = tmp.path().to_path_buf();
            (p, Some(tmp))
        };
        // Tempfile handle kept alive for the rest of this fn — the
        // mmap below references it. Pipeline holds none of it after
        // load completes (the merged tensors are loaded into RAM by
        // candle's loader, so the file can drop after `new` returns).
        let _lora_tmp = lora_tmp;

        // ---------- load flux + ae ----------
        let load = progress::spinner("Loading transformer + autoencoder");
        let flux_model = if is_gguf {
            let qvb = QVarBuilder::from_gguf(&main_path, &req.device).with_context(
                || format!("loading GGUF transformer from {}", main_path.display()),
            )?;
            // GGUF + LoRAs: build a dense-override map keyed by base
            // path (e.g. "double_blocks.0.img_attn.qkv.weight"). The
            // vendored quantized Flux substitutes a dense `QMatMul::Tensor`
            // for each override and leaves everything else 4-bit.
            let overrides = if req.loras.is_empty() {
                std::sync::Arc::new(std::collections::HashMap::new())
            } else {
                let spin = progress::spinner(&format!(
                    "Merging {} Flux LoRA(s) into quantized transformer",
                    req.loras.len()
                ));
                let (map, modified, total) =
                    crate::pipelines::flux_lora::precompute_quantized_overrides(
                        &qvb,
                        &req.loras,
                        req.lora_scale,
                        &req.device,
                    )?;
                spin.finish_with_message(format!(
                    "✓ Flux LoRA merged onto quantized backbone ({modified}/{total} target groups, \
                     {} dense Linears)",
                    map.len()
                ));
                std::sync::Arc::new(map)
            };
            // Plakat's vendored quantized Flux shares the BF16 vendor's
            // Config type, so the same `Variant::flux_config()` works
            // for both backbones.
            FluxBackbone::Quantized(qmodel::Flux::new_with_loras(
                &req.variant.flux_config(),
                qvb,
                overrides,
            )?)
        } else if is_nf4 {
            // v0.14 phase 2d: NF4 backbone. The pack is a regular
            // safetensors file with quantized weight bytes + per-block
            // absmax. We load every tensor into an Nf4Store and the
            // vendored Flux looks up packed+absmax via path-tracking
            // at each Linear construction site.
            //
            // The lllyasviel pack prefixes every transformer key with
            // `model.diffusion_model.` (ComfyUI convention). We strip
            // that here so the vendor's path matches Flux's BFL-native
            // naming (`img_in.weight`, `double_blocks.0.img_attn.qkv.weight`,
            // etc.).
            let raw_store =
                crate::pipelines::nf4_loader::Nf4Store::from_safetensors(
                    &main_path,
                    &req.device,
                )?;
            // ComfyUI / lllyasviel pack convention: transformer keys
            // are namespaced under `model.diffusion_model.`. Strip
            // that so the vendor's BFL-native paths match.
            let store = raw_store.with_prefix_stripped("model.diffusion_model.")?;
            // v0.14 phase 8b: NF4 + LoRA composition. Same selective
            // dequant strategy as GGUF: dequantize only LoRA-targeted
            // Linears at load, leave the rest 4-bit. Empty LoRA stack
            // → empty overrides map → exact phase-2 behaviour.
            let overrides = if req.loras.is_empty() {
                std::sync::Arc::new(std::collections::HashMap::new())
            } else {
                let spin = progress::spinner(&format!(
                    "Merging {} Flux LoRA(s) into NF4 transformer",
                    req.loras.len()
                ));
                let (map, modified, total) =
                    crate::pipelines::flux_lora::precompute_nf4_overrides(
                        &store,
                        &req.loras,
                        req.lora_scale,
                        &req.variant.flux_config(),
                        &req.device,
                    )?;
                spin.finish_with_message(format!(
                    "✓ Flux LoRA merged onto NF4 backbone ({modified}/{total} target groups, \
                     {} dense Linears)",
                    map.len()
                ));
                std::sync::Arc::new(map)
            };
            FluxBackbone::Nf4(crate::pipelines::flux_nf4_inner::Flux::new_with_loras(
                &req.variant.flux_config(),
                &store,
                dtype,
                overrides,
            )?)
        } else {
            let flux_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&effective_main_path], dtype, &req.device)?
            };
            FluxBackbone::Bf16(fmodel::Flux::new(&req.variant.flux_config(), flux_vb)?)
        };
        let ae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&ae_path], dtype, &req.device)?
        };
        let ae_model = fae::AutoEncoder::new(&req.variant.ae_config(), ae_vb)?;
        load.finish_with_message("✓ models loaded");

        // ---------- Flux ControlNet stack (v0.12 phase 2b + multi) -
        // Each CN in the stack loads its own weights and carries its
        // own scale / mode / conditioning path. Residuals from active
        // CNs sum per-step inside `denoise_with_optional_controlnet`.
        let mut controlnets: Vec<LoadedFluxControlNet> = Vec::with_capacity(req.controlnets.len());
        for (i, cn) in req.controlnets.into_iter().enumerate() {
            let spin = progress::spinner(&format!(
                "Downloading + remapping Flux ControlNet #{} ({}/{})",
                i + 1,
                cn.repo,
                cn.file
            ));
            let net = crate::pipelines::flux_controlnet::load_from_hf(
                &cn.repo,
                &cn.file,
                cn.cfg,
                &req.device,
                dtype,
            )
            .await?;
            spin.finish_with_message(format!("✓ Flux ControlNet #{} ready", i + 1));
            controlnets.push(LoadedFluxControlNet {
                net,
                scale: cn.scale,
                mode: cn.mode,
                conditioning_path: cn.conditioning,
                start: cn.start,
                end: cn.end,
            });
        }

        // v0.14 phase 3: Redux encoder. Only loaded when the caller
        // explicitly opted in (`LoadRequest::redux`) — SigLIP-so400m
        // is ~1.5 GB and the adapter is ~140 MB, and nothing in the
        // standard t2i path needs them. Pipeline::generate bails loud
        // if `redux_image` is set without this loader.
        let redux_encoder = if req.redux {
            let spin = progress::spinner("Loading Flux Redux (SigLIP + adapter)");
            let enc = crate::pipelines::flux_redux::ReduxEncoder::load(
                "google/siglip-so400m-patch14-384",
                "black-forest-labs/FLUX.1-Redux-dev",
                &req.device,
                dtype,
            )
            .await?;
            spin.finish_with_message("✓ Redux ready");
            Some(enc)
        } else {
            None
        };

        Ok(Self {
            variant: req.variant,
            repo: req.repo,
            device: req.device,
            dtype,
            clip_text,
            clip_tok,
            clip_cfg,
            t5_enc,
            t5_tok,
            flux_model,
            ae_model,
            controlnets,
            redux_encoder,
        })
    }

    /// Generate `req.count` images for one prompt. Reuses the loaded models.
    /// `&mut self` because T5's forward maintains a KV cache.
    pub fn generate(&mut self, req: &GenRequest) -> Result<()> {
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps());
        let guidance = req.guidance.unwrap_or_else(|| self.variant.default_guidance());
        // v0.18 phase 2b: opt-in Kontext aspect-bucket snap. When
        // active, the user's (w, h) is rounded to the nearest of the
        // 17 BFL-recommended Kontext resolutions BEFORE the standard
        // multiple-of-16 floor below — both the bucket sizes and the
        // floor are already 16-multiples so the snap is idempotent.
        let (req_width, req_height) =
            if self.variant.is_kontext() && req.kontext_bucket {
                let (sw, sh) = snap_to_kontext_bucket(req.width, req.height);
                if (sw, sh) != (req.width, req.height) {
                    crate::ui::progress::println(&format!(
                        "  kontext-bucket: {}x{} → {sw}x{sh}",
                        req.width, req.height
                    ));
                }
                (sw, sh)
            } else {
                (req.width, req.height)
            };
        let w = (req_width as usize / 16) * 16;
        let h = (req_height as usize / 16) * 16;
        if w == 0 || h == 0 {
            bail!("Flux requires width and height divisible by 16, both ≥ 16");
        }
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        // ---------- tiled-denoise validation (v0.13 phase 4 + 9) ----
        // Tiled Flux composes with ControlNet (v0.13 phase 9) and
        // — as of v0.16 phase 4 — also with Flux.1-Fill-dev: the
        // 2D masked-latent + image-space mask are sliced per tile
        // before packing (see `pack_fill_2d_tile`). Img2img + LoRA
        // + GGUF compose for free.
        if let Some(tcfg) = req.tiled.as_ref() {
            if tcfg.tile_size % 16 != 0 || tcfg.stride % 16 != 0 {
                bail!(
                    "Tiled Flux: --tile-size and --tile-stride must be divisible by 16 \
                     (got tile={} stride={}). Flux's 2×2 patching plus the VAE's 8× \
                     downsample requires a 16-pixel granularity.",
                    tcfg.tile_size, tcfg.stride
                );
            }
            if tcfg.stride == 0 || tcfg.stride > tcfg.tile_size {
                bail!(
                    "Tiled Flux: --tile-stride must be in (0, --tile-size]; \
                     got stride={} tile={}",
                    tcfg.stride, tcfg.tile_size
                );
            }
        }

        // ---------- encode prompt ----------
        let enc = progress::spinner("Encoding prompt");
        let (clip_pooled, mut t5_emb) = self.encode_prompt(&req.prompt)?;
        enc.finish_with_message("✓ prompt encoded");

        // ---------- Flux Redux conditioning (v0.14 phase 3 / 3c) ---
        // When the user supplies one or more `--redux-image` specs,
        // encode each through SigLIP-so400m + the Redux adapter,
        // scale by the spec's weight, and seq-concat the resulting
        // 729 tokens per image onto the T5 embedding. The Flux
        // transformer's `txt` input grows from (1, t5_seq, 4096) to
        // (1, t5_seq + N * 729, 4096); `txt_ids` is regenerated below
        // as zeros of matching length inside `sampling::State::new`,
        // which already builds it from `t5_emb.dim(1)`.
        if !req.redux_images.is_empty() {
            if self.variant.is_fill() {
                bail!(
                    "Flux Redux doesn't compose with Flux.1-Fill-dev (Fill's `img_in` \
                     is 384ch — Redux changes only the text sequence so it works on \
                     the standard Flux variants only)."
                );
            }
            // Soft guardrails on stack depth. Attention is O(seq²), so
            // every Redux image adds 729² ≈ 530k extra attn entries.
            // 4 images: ~8.5M entries on top of T5's ~250k. Beyond
            // that the per-step cost dwarfs everything else and the
            // user usually doesn't realise.
            const REDUX_MAX_IMAGES: usize = 4;
            const REDUX_WARN_THRESHOLD: usize = 2;
            if req.redux_images.len() > REDUX_MAX_IMAGES {
                bail!(
                    "--redux-image cap is {} (got {}). Each image adds 729 attention \
                     tokens to every Flux block — past the cap, per-step cost \
                     dominates and quality usually doesn't improve.",
                    REDUX_MAX_IMAGES,
                    req.redux_images.len()
                );
            }
            if req.redux_images.len() > REDUX_WARN_THRESHOLD {
                tracing::warn!(
                    target: "plakat",
                    "Flux Redux with {} images: txt sequence grows to {} tokens, \
                     attention cost scales quadratically.",
                    req.redux_images.len(),
                    self.variant.t5_seq_len() + 729 * req.redux_images.len(),
                );
            }
            let enc = self.redux_encoder.as_ref().ok_or_else(|| anyhow!(
                "redux_images set but Redux encoder isn't loaded. Build the Pipeline \
                 with `LoadRequest::redux = true` (or run the CLI without --redux-image)."
            ))?;
            let spin = progress::spinner(&format!(
                "Encoding {} Redux reference image(s)",
                req.redux_images.len()
            ));
            let t5_dtype = t5_emb.dtype();
            let mut total_added = 0usize;
            for spec in &req.redux_images {
                if spec.weight == 0.0 {
                    // Weight-0 image is a no-op — skip to save a
                    // SigLIP + adapter forward each generate call.
                    continue;
                }
                let tokens = enc.encode_image_scaled(&spec.path, spec.weight)?;
                let tokens = tokens.to_dtype(t5_dtype)?;
                total_added += tokens.dim(1)?;
                t5_emb = Tensor::cat(&[&t5_emb, &tokens], 1)?;
            }
            spin.finish_with_message(format!(
                "✓ Redux conditioning encoded (+{} tokens across {} image(s))",
                total_added,
                req.redux_images.len()
            ));
        }

        let ae_cfg = self.variant.ae_config();
        let lat_h = (h + 15) / 16;
        let lat_w = (w + 15) / 16;
        let image_seq_len = lat_h * lat_w;

        // ---------- Flux img2img init prep (v0.13 phase 3) ---------
        // For non-Fill variants with an init image, VAE-encode the
        // init once and reuse across the per-count loop. Each
        // generation builds its own start_latent = lerp(init, noise,
        // strength) and runs a truncated schedule starting at t=strength.
        // Fill mode uses `init_image` for its own conditioning path
        // (handled below) so we skip this branch when is_fill().
        let img2img_init: Option<(Tensor, f32)> = if self.variant.is_fill() {
            None
        } else if let Some(init_path) = req.init_image.as_ref() {
            let strength = req.strength.unwrap_or(0.85).clamp(0.0, 1.0);
            if !strength.is_finite() {
                bail!("img2img strength must be finite in [0, 1], got {strength}");
            }
            let spin = progress::spinner("Encoding img2img init image");
            // [-1, 1] domain, then VAE-encode and apply the BFL latent
            // normalization (z - shift) * scale — same convention the
            // standard t2i path's noise sampling produces, so lerp is
            // dimensionally consistent.
            let init_pixels = crate::imaging::preprocess::sd_image_tensor(
                init_path,
                w as u32,
                h as u32,
                &self.device,
                self.dtype,
            )
            .with_context(|| {
                format!("loading img2img init image {}", init_path.display())
            })?;
            let init_z = self.ae_model.encode(&init_pixels)?;
            let init_norm = ((init_z - ae_cfg.shift_factor)? * ae_cfg.scale_factor)?;
            spin.finish_with_message(format!("✓ img2img init encoded (strength {strength:.2})"));
            Some((init_norm, strength))
        } else {
            None
        };

        // ---------- v0.15 phase 4: concept-variant prep ----------
        // Flux.1-Canny-dev / Flux.1-Depth-dev have `img_in` widened to
        // 128 channels = 64 noise + 64 conditioning. The conditioning
        // is VAE-encoded just like a ControlNet conditioning image
        // (and reuses `encode_conditioning_2d` for that). Packed once
        // outside the per-step loop; cat'd into `flux_input` per step
        // alongside the noise tokens.
        //
        // Defensive bails on cross-feature combos that aren't yet
        // validated (tiled, img2img, Redux, Fill, ControlNet). These
        // can be relaxed once we've tested against real workflows.
        // ---------- v0.18 Kontext reference (seq-concat path) ------
        // Build the (packed_tokens, ref_img_ids) tuple once. Unlike
        // Fill / Concept which channel-concat, Kontext extends the
        // sequence dimension and adds matching positional ids with
        // axis 0 = 1 so the model's RoPE can tell reference tokens
        // from noise tokens.
        let kontext_ref_packed: Option<(Tensor, Tensor)> = if self.variant.is_kontext() {
            if req.tiled.is_some() {
                bail!(
                    "Flux Kontext doesn't compose with --tiled in this release \
                     (per-tile reference slicing isn't wired)."
                );
            }
            if req.init_image.is_some() || req.mask.is_some() {
                bail!(
                    "Flux Kontext denoises from pure noise + the reference image. \
                     --init-image / --mask aren't accepted on flux-kontext-dev — \
                     pass the reference via --concept-image PATH instead."
                );
            }
            if !req.redux_images.is_empty() {
                bail!(
                    "Flux Kontext + --redux-image isn't wired yet — both extend \
                     the sequence dimension and may exceed Flux's RoPE budget. \
                     Use one or the other."
                );
            }
            if !self.controlnets.is_empty() {
                bail!(
                    "Flux Kontext + --control-spec isn't wired yet. The reference \
                     image already drives the layout; pairing with a ControlNet \
                     would double-condition. Drop --control-spec or switch to \
                     flux-dev for ControlNet."
                );
            }
            let ref_path = req.concept_conditioning.as_ref().ok_or_else(|| {
                anyhow!(
                    "Flux.1-Kontext-dev requires a reference image — pass it via \
                     --concept-image PATH (the image you want to edit)."
                )
            })?;
            let spin = progress::spinner("Encoding Kontext reference image");
            let pair = self.encode_kontext_reference(ref_path, h, w)?;
            spin.finish_with_message(format!(
                "✓ Kontext reference encoded ({} tokens × 64ch)",
                pair.0.dim(1)?
            ));
            Some(pair)
        } else {
            None
        };

        let concept_cond_packed: Option<Tensor> = if self.variant.is_concept() {
            if req.tiled.is_some() {
                bail!(
                    "Flux concept variants (Canny-dev / Depth-dev) don't compose with \
                     --tiled yet (v0.15 phase 4 deferral — per-tile conditioning slicing \
                     isn't wired)."
                );
            }
            if req.init_image.is_some() || req.mask.is_some() {
                bail!(
                    "Flux concept variants don't accept --init-image / --mask — they \
                     denoise from pure noise + the conditioning map. Use \
                     --concept-image PATH for the canny/depth conditioning."
                );
            }
            if !req.redux_images.is_empty() {
                bail!(
                    "Flux concept variants don't yet compose with --redux-image \
                     (v0.15 phase 4 deferral)."
                );
            }
            if !self.controlnets.is_empty() {
                bail!(
                    "Flux concept variants ship their own canny/depth conditioning \
                     baked in — pairing with --control-spec would double-condition. \
                     Drop --control-spec or switch to flux-dev for ControlNet."
                );
            }
            let cond_path = req.concept_conditioning.as_ref().ok_or_else(|| {
                anyhow!(
                    "Flux.1-{}-dev requires a conditioning map — pass it via \
                     --concept-image PATH.",
                    self.variant.concept_kind().unwrap_or("Canny/Depth")
                )
            })?;
            let spin = progress::spinner(&format!(
                "Encoding {} conditioning",
                self.variant.concept_kind().unwrap_or("concept")
            ));
            let z2d = self.encode_conditioning_2d(cond_path, h, w)?;
            let packed = pack_latent_to_tokens(&z2d)?;
            spin.finish_with_message(format!(
                "✓ {} conditioning encoded ({} tokens × 64ch)",
                self.variant.concept_kind().unwrap_or("concept"),
                packed.dim(1)?
            ));
            Some(packed)
        } else {
            // Kontext consumes its own --concept-image upstream; only
            // warn when the variant accepts neither channel-concat nor
            // seq-concat conditioning.
            if req.concept_conditioning.is_some() && !self.variant.is_kontext() {
                tracing::warn!(
                    target: "plakat",
                    "--concept-image supplied but model is not a concept variant \
                     (Canny-dev / Depth-dev / Kontext-dev) — input ignored."
                );
            }
            None
        };

        // ---------- Flux Fill inpainting prep (v0.13 phase 2) -------
        // Fill-dev's `img_in` takes 384 channels = 64 noise + 64 masked-
        // latent + 256 image-space-mask. The first 64 are filled per
        // step from the noise tensor we're integrating; the trailing
        // 320 are constants computed here from the user's init image +
        // mask. Stays `None` for non-Fill variants — the denoise loop
        // skips the cat.
        //
        // v0.16 phase 4: keep the 2D form alongside the packed full
        // canvas so the tiled denoise can slice per tile.
        // `fill_cond_2d` is `Some` only on Fill; `fill_cond_packed`
        // is the canonical full-canvas packing (320ch) consumed by
        // the non-tiled denoise.
        let fill_cond_2d = if self.variant.is_fill() {
            let init_path = req.init_image.as_ref().ok_or_else(|| {
                anyhow!(
                    "Flux.1-Fill-dev requires --image / init_image — no init image provided."
                )
            })?;
            let mask_path = req.mask.as_ref().ok_or_else(|| {
                anyhow!("Flux.1-Fill-dev requires --mask — no mask image provided.")
            })?;
            let spin = progress::spinner("Encoding init image + mask for Fill");
            let cond = self.encode_fill_conditioning_2d(init_path, mask_path, h, w)?;
            spin.finish_with_message("✓ Fill conditioning encoded");
            Some(cond)
        } else {
            if req.init_image.is_some() || req.mask.is_some() {
                tracing::warn!(
                    target: "plakat",
                    "init_image / mask supplied but model is not Flux.1-Fill-dev — \
                     inputs ignored."
                );
            }
            None
        };
        // Pack the full canvas once for the non-tiled denoise. Tiled
        // packs per-tile inside `denoise_tiled` from `fill_cond_2d`.
        let fill_cond_packed = match fill_cond_2d.as_ref() {
            Some(c) => Some(pack_fill_2d_full(c)?),
            None => None,
        };

        // ---------- ControlNet conditioning prep (v0.12 + multi) ----
        // VAE-encode + pack each loaded ControlNet's conditioning
        // image once. Per-CN paths come from the LoadRequest; the
        // GenRequest's `conditioning` field is a back-compat shim
        // that applies to the first CN if present.
        let mut conditioning_packed: Vec<Option<Tensor>> =
            Vec::with_capacity(self.controlnets.len());
        // v0.13 phase 9: when tiled is on, also keep the 2D conditioning
        // latent alongside the packed-token form. Tiled denoise narrows
        // per tile and packs locally; non-tiled uses `packed` directly
        // and ignores `conditioning_2d`. Memory cost is small (one 2D
        // latent per CN) and only paid when tiled is active.
        let want_2d_cond = req.tiled.is_some();
        let mut conditioning_2d: Vec<Option<Tensor>> = if want_2d_cond {
            Vec::with_capacity(self.controlnets.len())
        } else {
            Vec::new()
        };
        for (i, cn) in self.controlnets.iter().enumerate() {
            // GenRequest.conditioning overrides the LoadRequest path
            // for CN #0 — keeps single-CN callers that haven't been
            // updated to per-CN conditioning paths working.
            let path = if i == 0 {
                req.conditioning
                    .as_deref()
                    .or(cn.conditioning_path.as_deref())
            } else {
                cn.conditioning_path.as_deref()
            };
            match path {
                Some(p) => {
                    let spin = progress::spinner(&format!(
                        "Encoding ControlNet #{} conditioning",
                        i + 1
                    ));
                    let z2d = self.encode_conditioning_2d(p, h, w)?;
                    let packed = pack_latent_to_tokens(&z2d)?;
                    if want_2d_cond {
                        conditioning_2d.push(Some(z2d));
                    }
                    spin.finish_with_message(format!(
                        "✓ ControlNet #{} conditioning encoded",
                        i + 1
                    ));
                    conditioning_packed.push(Some(packed));
                }
                None => {
                    tracing::warn!(
                        target: "plakat",
                        "Flux ControlNet #{} loaded but no conditioning image — \
                         this CN won't contribute residuals.",
                        i + 1
                    );
                    conditioning_packed.push(None);
                    if want_2d_cond {
                        conditioning_2d.push(None);
                    }
                }
            }
        }

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random)
                & (u32::MAX as u64);
            if let Err(e) = self.device.set_seed(seed) {
                tracing::debug!(
                    target: "plakat",
                    "set_seed not supported ({e}); using global RNG"
                );
            }

            // Fresh noise. For img2img, mixed with the (pre-encoded)
            // init latent at t=strength using the standard rectified-
            // flow interpolation: x_t = (1-t)*x_init + t*x_noise.
            let noise = sampling::get_noise(1, h, w, &self.device)?.to_dtype(self.dtype)?;
            let img_2d = match img2img_init.as_ref() {
                Some((init_norm, strength)) => {
                    let s = *strength as f64;
                    ((init_norm * (1.0 - s))? + (&noise * s)?)?
                }
                None => noise,
            };

            let shift = if self.variant.is_dev() {
                Some((image_seq_len, 0.5_f64, 1.15_f64))
            } else {
                None
            };
            let mut timesteps = sampling::get_schedule(steps, shift);
            // img2img: drop schedule entries above `strength` so the
            // loop starts at t≈strength. Prepend `strength` itself so
            // the first window's t_curr matches the noise level the
            // start latent was built at. Mirrors diffusers'
            // `FluxImg2ImgPipeline.get_timesteps`. Without this the
            // model would think the input is fully-noised when it's
            // only partially noised.
            if let Some((_init, strength)) = img2img_init.as_ref() {
                let s = *strength as f64;
                let mut filtered: Vec<f64> =
                    timesteps.iter().copied().filter(|&t| t < s).collect();
                if filtered.is_empty() {
                    filtered.push(0.0);
                }
                let mut new_ts = Vec::with_capacity(filtered.len() + 1);
                new_ts.push(s);
                new_ts.extend(filtered);
                timesteps = new_ts;
            }

            let bar = progress::step_bar(
                (timesteps.len().saturating_sub(1)) as u64,
                &format!("img {}/{}", idx + 1, req.count),
            );

            // v0.13 phase 4: tiled denoise stays in 2D latent form
            // throughout — every per-step forward operates on a tile
            // and the predictions are blended back via Hann window.
            // Standard path packs to tokens once, denoises, unpacks
            // at the end.
            let denoised_2d = if let Some(tcfg) = req.tiled.as_ref() {
                bar.set_message(format!(
                    "tiled flow-match denoise, {steps} steps, seed={seed}"
                ));
                self.denoise_tiled(
                    &img_2d,
                    &t5_emb,
                    &clip_pooled,
                    &timesteps,
                    guidance,
                    tcfg,
                    &conditioning_2d,
                    fill_cond_2d.as_ref(),
                    &bar,
                )?
            } else {
                bar.set_message(format!("flow-match denoise, {steps} steps, seed={seed}"));
                let state = sampling::State::new(&t5_emb, &clip_pooled, &img_2d)?;
                let denoised = self.denoise_with_optional_controlnet(
                    &state,
                    &timesteps,
                    guidance,
                    &conditioning_packed,
                    fill_cond_packed.as_ref(),
                    concept_cond_packed.as_ref(),
                    kontext_ref_packed.as_ref(),
                    &bar,
                )?;
                sampling::unpack(&denoised, h, w)?
            };
            bar.set_position(timesteps.len().saturating_sub(1) as u64);
            bar.finish_with_message("✓ denoised");

            // BFL AE expects: x = decode((z / scale) + shift)
            let pre_decode = ((&denoised_2d / ae_cfg.scale_factor)? + ae_cfg.shift_factor)?;
            let decoded = if let Some(tcfg) = req.tiled.as_ref() {
                // Tiled decode keeps the VAE working at its native scale
                // even for 4K outputs. Latent units = pixel units / 8.
                let lat_tile = (tcfg.tile_size as usize) / 8;
                let lat_stride = (tcfg.stride as usize) / 8;
                crate::pipelines::tiled::tile_decode_2d(
                    &pre_decode,
                    lat_tile,
                    lat_stride,
                    8,
                    |t| Ok(self.ae_model.decode(t)?),
                )?
            } else {
                self.ae_model.decode(&pre_decode)?
            };
            let img_norm = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 0.5)?;
            let img_u8 = (img_norm * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = img_u8.dims3()?;
            let buf = img_u8.flatten_all()?.to_vec1::<u8>()?;

            let out_path = req.out_dir.join(format!("plakat-flux-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
        }
        Ok(())
    }

    /// Encode a single prompt into (clip_pooled, t5_emb).
    /// - clip_pooled: (1, 768)   — CLIP-L pooled at the EOT-token position
    /// - t5_emb:      (1, seq, 4096) — T5-XXL last hidden states
    ///
    /// v0.18 phase 3: T5 hidden states are per-token-row weighted by
    /// the A1111 attention syntax (e.g. `(red:1.4)`). CLIP-L on Flux
    /// is pooled-only — only the EOT position is read out, so per-
    /// token weights don't change the pooled vector and the CLIP-L
    /// path is left untouched.
    fn encode_prompt(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        // v0.19 #5: A1111 BREAK keyword. Flux's T5 has a 256/512-
        // token budget, far past CLIP's 77-token cap — BREAK adds
        // no value here. Strip + warn rather than silently passing
        // a literal "BREAK" through to T5 (which would tokenize
        // it as a normal word).
        let prompt_stripped: String;
        let prompt: &str = if crate::prompt::break_chunks::has_break(prompt) {
            tracing::warn!(
                target: "plakat",
                "BREAK keyword ignored on Flux — T5 has a {}-token budget, \
                 prompt chunking is a CLIP-only workaround. Strip BREAK or \
                 switch to --model sd15 / sd21 / sdxl.",
                self.variant.t5_seq_len()
            );
            prompt_stripped = crate::prompt::break_chunks::strip(prompt);
            prompt_stripped.as_str()
        } else {
            prompt
        };

        // CLIP-L: tokenize to 77, run, pool at EOT.
        let mut clip_ids = self
            .clip_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("CLIP encode: {e}"))?
            .get_ids()
            .to_vec();
        clip_ids.resize(self.clip_cfg.max_position_embeddings, CLIP_EOT);
        let clip_eot_pos = clip_ids.iter().position(|&t| t == CLIP_EOT).unwrap_or(0);
        let clip_ids_t = Tensor::new(clip_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let clip_seq = self.clip_text.forward(&clip_ids_t)?;
        let clip_pooled = clip_seq.i((.., clip_eot_pos, ..))?.to_dtype(self.dtype)?;

        // T5: tokenize to variant.t5_seq_len(), pad with id 0, run encoder.
        let t5_seq_len = self.variant.t5_seq_len();
        let t5_emb = if crate::prompt::a1111::has_attention_syntax(prompt) {
            // v0.18 phase 3: weighted path. Per-token-row broadcast
            // of A1111 attention weights onto T5's hidden states.
            // T5 has no BOS; `</s>` (id 1 in T5-XXL) is EOS; pad is
            // 0. Look the IDs up rather than hardcoding so a future
            // tokenizer variant doesn't silently drift.
            let t5_eos = self
                .t5_tok
                .token_to_id("</s>")
                .ok_or_else(|| anyhow!("T5 tokenizer missing </s>"))?;
            let t5_pad = self.t5_tok.token_to_id("<pad>").unwrap_or(0);
            let wcfg = crate::prompt::weighted_encoding::WeightedTokenConfig {
                tokenizer: &self.t5_tok,
                max_len: t5_seq_len,
                bos_id: None,
                eos_id: t5_eos,
                pad_id: t5_pad,
            };
            let (ids, weights) = crate::prompt::weighted_encoding::tokenize_with_attention(
                &wcfg,
                prompt,
                &self.device,
                self.dtype,
            )?;
            let hidden = self.t5_enc.forward(&ids)?.to_dtype(self.dtype)?;
            let weights = weights.to_dtype(hidden.dtype())?;
            hidden.broadcast_mul(&weights)?
        } else {
            // Fast path — byte-identical to pre-phase-3 behaviour.
            let mut t5_ids = self
                .t5_tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("T5 encode: {e}"))?
                .get_ids()
                .to_vec();
            t5_ids.truncate(t5_seq_len);
            t5_ids.resize(t5_seq_len, 0);
            let t5_ids_t = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            self.t5_enc.forward(&t5_ids_t)?.to_dtype(self.dtype)?
        };
        Ok((clip_pooled, t5_emb))
    }

    /// v0.12 phase 2b → v0.13 phase 9: load + VAE-encode a Flux
    /// ControlNet conditioning image. Returns the **2D latent**
    /// `(1, 16, lh, lw)`. Standard non-tiled callers pack the result
    /// with `pack_latent_to_tokens` immediately; tiled callers keep
    /// the 2D form so they can `.narrow` per tile before packing —
    /// without that, every tile would receive the full-canvas
    /// conditioning and the CN's spatial signal would smear.
    fn encode_conditioning_2d(
        &self,
        path: &std::path::Path,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        // Read pixels in the same `[-1, 1]` normalization the Flux AE
        // was trained on, matching plakat's existing SD `sd_image_tensor`
        // convention. The Flux AE accepts this domain directly.
        let pixels = crate::imaging::preprocess::sd_image_tensor(
            path,
            w as u32,
            h as u32,
            &self.device,
            self.dtype,
        )
        .with_context(|| {
            format!("loading Flux ControlNet conditioning {}", path.display())
        })?;
        // Flux AE expects pre-shift: z = (encode(x) - shift) * scale
        let ae_cfg = self.variant.ae_config();
        let z = self.ae_model.encode(&pixels)?;
        let z = ((z - ae_cfg.shift_factor)? * ae_cfg.scale_factor)?;
        let (_b, c, _lh, _lw) = z.dims4()?;
        if c != 16 {
            anyhow::bail!(
                "Flux AE encoded to {c} channels — expected 16. Conditioning prep aborted."
            );
        }
        Ok(z)
    }

    /// v0.18: encode a Kontext reference image into the
    /// `(packed_tokens, ref_img_ids)` pair that gets sequence-concat'd
    /// onto the noise tokens at each denoise step. Unlike the concept
    /// variants (which channel-concat into a 128ch `img_in`), Kontext
    /// keeps `img_in` at 64 channels and extends the **sequence**
    /// dimension. The reference tokens carry `img_ids[..., 0] = 1` so
    /// the model's RoPE positional encoding can tell them apart from
    /// the noise tokens (which carry `img_ids[..., 0] = 0`).
    ///
    /// Returns `((1, ref_seq, 64), (1, ref_seq, 3))`. Caller cats both
    /// onto the noise tokens / `state.img_ids` per step.
    fn encode_kontext_reference(
        &self,
        path: &std::path::Path,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, Tensor)> {
        let z2d = self.encode_conditioning_2d(path, h, w)?;
        let packed = pack_latent_to_tokens(&z2d)?;
        // Match the layout `sampling::State::new` builds for the noise
        // tokens (axis 0 marker, h_id, w_id, shape (1, lh/2 * lw/2, 3))
        // — but force axis 0 to 1 so the model recognises this half as
        // the reference. Use the SAME dtype as `state.img_ids` so the
        // cat at step time succeeds without explicit promotion.
        let (_b, _c, lh, lw) = z2d.dims4()?;
        let (h2, w2) = (lh / 2, lw / 2);
        let dev = &self.device;
        let ones = Tensor::full(1f32, (h2, w2), dev)?.to_dtype(self.dtype)?;
        let h_ids = Tensor::arange(0u32, h2 as u32, dev)?
            .reshape(((), 1))?
            .broadcast_as((h2, w2))?
            .to_dtype(self.dtype)?;
        let w_ids = Tensor::arange(0u32, w2 as u32, dev)?
            .reshape((1, ()))?
            .broadcast_as((h2, w2))?
            .to_dtype(self.dtype)?;
        let ref_img_ids = Tensor::stack(&[&ones, &h_ids, &w_ids], 2)?
            .reshape((1, h2 * w2, 3))?;
        Ok((packed, ref_img_ids))
    }

    /// v0.12 phase 2b + multi: flow-matching denoise loop. For each
    /// step, every loaded ControlNet that has its conditioning ready
    /// runs once with its own (scale, mode, conditioning) tuple. The
    /// resulting DoubleStream + SingleStream residuals sum per-block
    /// across all active CNs before being fed to the main Flux's
    /// `forward_with_residuals`. Empty CN stack reduces to candle's
    /// stock `sampling::denoise` byte-for-byte.
    ///
    /// v0.18: Kontext reference (`kontext_ref`) seq-concats reference
    /// tokens onto the noise tokens at each step. The full sequence
    /// goes through the DiT; afterwards the reference tail is
    /// stripped from the prediction so only the noise half advances
    /// under flow-matching.
    fn denoise_with_optional_controlnet(
        &self,
        state: &sampling::State,
        timesteps: &[f64],
        guidance: f64,
        conditioning_packed: &[Option<Tensor>],
        fill_cond: Option<&Tensor>,
        concept_cond: Option<&Tensor>,
        kontext_ref: Option<&(Tensor, Tensor)>,
        bar: &indicatif::ProgressBar,
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let dev = state.img.device();
        let guidance_t = Tensor::full(guidance as f32, b_sz, dev)?;
        // `img` is always the 64-channel noise tensor we're integrating
        // over time. Fill mode concatenates `fill_cond` (320ch) to it
        // only for the Flux forward call — CNs still see the noise
        // tensor unchanged, and the output noise prediction is also
        // 64ch (final_layer is independent of img_in's input width).
        let mut img = state.img.clone();
        // Total steps for the per-step progress fraction. timesteps has
        // `num_steps + 1` entries (boundary list), so `num_steps` =
        // `timesteps.windows(2).count()` — denominator for the
        // `[0, 1)` progress signal each CN's `active_at` consumes.
        let num_steps = timesteps.windows(2).count().max(1);
        for (step_i, window) in timesteps.windows(2).enumerate() {
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
            let progress = step_i as f32 / num_steps as f32;

            // Run each loaded CN that has its conditioning ready AND is
            // active at this progress; sum residuals per-block across
            // all of them. CN's img_in is 64ch (stock Flux), so we
            // pass the noise tensor — not the concatenated Fill input
            // — even in Fill mode.
            let mut summed_double: Option<Vec<Tensor>> = None;
            let mut summed_single: Option<Vec<Tensor>> = None;
            for (cn, cond_opt) in self.controlnets.iter().zip(conditioning_packed.iter())
            {
                if !cn.active_at(progress) {
                    continue; // outside this CN's `[start, end)` window
                }
                let cond = match cond_opt.as_ref() {
                    Some(c) => c,
                    None => continue, // CN has no conditioning → skip.
                };
                let (d, s) = cn.net.forward(
                    &img,
                    cond,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    cn.mode,
                    cn.scale,
                )?;
                summed_double = Some(merge_residuals(summed_double, d)?);
                summed_single = Some(merge_residuals(summed_single, s)?);
            }
            let double_r = summed_double.as_deref();
            let single_r = match summed_single.as_ref() {
                Some(v) if !v.is_empty() => Some(v.as_slice()),
                _ => None,
            };

            // Build the Flux-forward input. Standard Flux: img is 64ch
            // noise. Fill: cat 320ch conditioning → 384ch. Concept
            // (Canny-dev / Depth-dev): cat 64ch conditioning → 128ch.
            // Fill + concept are mutually exclusive variants so only
            // one of these branches fires per call.
            let flux_input: Tensor = match (fill_cond, concept_cond) {
                (Some(c), _) => Tensor::cat(&[&img, c], D::Minus1)?,
                (None, Some(c)) => Tensor::cat(&[&img, c], D::Minus1)?,
                (None, None) => img.clone(),
            };

            // v0.18 Kontext: sequence-concat reference tokens onto
            // the noise tokens (and same for img_ids). The reference
            // tokens are constant across steps so the cat result for
            // img_ids could be cached at setup; we still rebuild
            // `expanded_img` per step because the noise half varies.
            // Tracks the noise half's length so we can slice the
            // reference tail off the prediction.
            let noise_seq_len = flux_input.dim(1)?;
            let (expanded_img, expanded_img_ids) = match kontext_ref {
                Some((ref_tokens, ref_ids)) => {
                    let exp_img = Tensor::cat(&[&flux_input, ref_tokens], 1)?;
                    let exp_ids = Tensor::cat(&[&state.img_ids, ref_ids], 1)?;
                    (exp_img, exp_ids)
                }
                None => (flux_input, state.img_ids.clone()),
            };

            // v0.13 phase 1c: both backbones expose the same
            // `forward_with_residuals` signature. ControlNet residuals
            // compose exactly the same way on BF16 and quantized.
            let pred_full = match &self.flux_model {
                FluxBackbone::Bf16(net) => net.forward_with_residuals(
                    &expanded_img,
                    &expanded_img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    double_r,
                    single_r,
                )?,
                FluxBackbone::Quantized(net) => net.forward_with_residuals(
                    &expanded_img,
                    &expanded_img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    double_r,
                    single_r,
                )?,
                // v0.15 phase 1: NF4 vendor now exposes
                // forward_with_residuals (same interleave as GGUF/BF16).
                FluxBackbone::Nf4(net) => net.forward_with_residuals(
                    &expanded_img,
                    &expanded_img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    double_r,
                    single_r,
                )?,
            };
            // Slice off the reference tail (when present). `pred` is the
            // 64ch noise prediction regardless of Fill mode (final_layer
            // outputs the same 64ch). The flow-match step only ever
            // updates the noise tensor.
            let pred = if kontext_ref.is_some() {
                pred_full.narrow(1, 0, noise_seq_len)?
            } else {
                pred_full
            };
            img = (img + pred * (t_prev - t_curr))?;
            bar.set_position(step_i as u64);
        }
        Ok(img)
    }

    /// v0.13 phase 2: build the 320-channel Fill conditioning tensor.
    ///
    /// Output shape: `(1, image_seq_len, 320)` — concatenates with the
    /// 64-channel noise to form Fill's 384-channel `img_in` input.
    ///
    /// Layout per token:
    /// * channels `0..64`: VAE-encoded init image with mask=1 regions
    ///   zeroed, 2x2-patched the same way the noise latent is packed.
    /// * channels `64..320`: image-space mask (1ch × H × W) reshaped
    ///   into 16×16 patches → 256 channels per token. The 16-pixel
    ///   patch size matches Flux's effective per-token receptive field
    ///   (8× VAE downsample × 2× Flux 2x2 patching).
    /// v0.16 phase 4: build the **2D form** of Flux Fill conditioning
    /// — the masked init latent + image-space mask — kept separately
    /// so the tiled denoise can slice each per tile before packing.
    /// Non-tiled callers pack the full canvas via [`pack_fill_2d_full`];
    /// tiled callers slice + pack per tile in `denoise_tiled`.
    fn encode_fill_conditioning_2d(
        &self,
        init_path: &std::path::Path,
        mask_path: &std::path::Path,
        h: usize,
        w: usize,
    ) -> Result<FillConditioning2D> {
        // Load init image at the target resolution, normalized to
        // `[-1, 1]` (the Flux AE input domain).
        let init_pixels = crate::imaging::preprocess::sd_image_tensor(
            init_path,
            w as u32,
            h as u32,
            &self.device,
            self.dtype,
        )
        .with_context(|| format!("loading Fill init image {}", init_path.display()))?;

        // Load mask as a single-channel grayscale at full resolution
        // and binarise: ≥128 → 1.0 (inpaint), else 0.0. Same convention
        // plakat's SD inpaint path uses. Shape: (1, 1, H_img, W_img).
        let mask_img = image::open(mask_path)
            .with_context(|| format!("opening mask {}", mask_path.display()))?
            .to_luma8();
        let mask_img = image::imageops::resize(
            &mask_img,
            w as u32,
            h as u32,
            image::imageops::FilterType::Triangle,
        );
        let mask_data: Vec<f32> = mask_img
            .pixels()
            .map(|p| if p.0[0] >= 128 { 1.0 } else { 0.0 })
            .collect();
        let mask_tensor = Tensor::from_vec(mask_data, (1, 1, h, w), &self.device)?
            .to_dtype(self.dtype)?;

        // Apply mask to init (zero out the mask=1 region) — the model
        // sees the regions to be inpainted as black in the masked
        // latent, which matches BFL's training distribution. Mask is
        // broadcast across the 3 colour channels.
        let one_minus_mask = (Tensor::ones_like(&mask_tensor)? - &mask_tensor)?;
        let masked_pixels = init_pixels.broadcast_mul(&one_minus_mask)?;

        // VAE-encode the masked image into a 16ch latent.
        let ae_cfg = self.variant.ae_config();
        let z = self.ae_model.encode(&masked_pixels)?;
        let z = ((z - ae_cfg.shift_factor)? * ae_cfg.scale_factor)?;
        let (_b, c, lh, lw) = z.dims4()?;
        if c != 16 {
            bail!(
                "Flux AE encoded to {c} channels — expected 16. Fill conditioning aborted."
            );
        }
        if lh % 2 != 0 || lw % 2 != 0 {
            bail!(
                "Flux Fill needs latent dims divisible by 2 (got {lh}x{lw}); image dims \
                 should be divisible by 16."
            );
        }
        if h % 16 != 0 || w % 16 != 0 {
            bail!(
                "Flux Fill needs image dims divisible by 16 (got {h}x{w})."
            );
        }
        Ok(FillConditioning2D {
            masked_latent_2d: z,
            mask_2d: mask_tensor,
        })
    }

    /// v0.13 phase 4: MultiDiffusion-style tiled Flux denoise.
    ///
    /// Each step:
    /// 1. For every overlapping tile in the latent canvas, pack to
    ///    `(1, num_tokens, 64)` tokens, build per-tile `img_ids` that
    ///    reflect the tile's **global** position (so positional
    ///    embeddings agree across tiles), and run Flux forward to get
    ///    a per-tile noise prediction.
    /// 2. Unpack each prediction back to a 2D latent tile, weight it
    ///    by a 2D Hann window, and accumulate into a full-canvas
    ///    noise-prediction buffer plus a matching weight buffer.
    /// 3. Divide accumulator by weights → full-canvas noise prediction.
    /// 4. Standard flow-match update: `latent += pred * (t_prev - t_curr)`.
    ///
    /// The transformer only ever sees `tile_size × tile_size` worth of
    /// tokens per call, so memory cost is bounded by the tile, not the
    /// canvas — same trick the SDXL tiled path uses. Trades wall time
    /// linearly with tile count: a 2048² canvas with 1024² tiles at
    /// 768 stride = 3×3 = 9 forwards per step.
    ///
    /// Composes with: LoRA, GGUF (via the same `FluxBackbone` dispatch
    /// the standard denoise uses), img2img (caller pre-mixes the
    /// init+noise into the canvas), **ControlNet** (v0.13 phase 9 —
    /// each loaded CN's conditioning latent is cropped to the current
    /// tile and packed inside the loop), and — as of v0.16 phase 4 —
    /// **Flux.1-Fill-dev**: the 2D masked latent + image-space mask
    /// are sliced per tile via [`pack_fill_2d_tile`] and concatenated
    /// to the per-tile noise packing before the Flux forward call.
    ///
    /// `conditioning_2d` carries one optional entry per loaded
    /// ControlNet, in the same order as `self.controlnets`. Entries
    /// shape `(1, 16, lh, lw)` (the full-canvas conditioning latent).
    /// `None` entries mean "CN loaded but no conditioning image" — the
    /// CN is silently skipped for every tile.
    ///
    /// `fill_cond_2d` is `Some` only when the loaded variant is
    /// Flux.1-Fill-dev. Layout: `(1, 16, lh, lw)` masked latent +
    /// `(1, 1, lh*8, lw*8)` image-space binary mask. Sliced per
    /// tile in the loop below.
    #[allow(clippy::too_many_arguments)]
    fn denoise_tiled(
        &self,
        canvas_latent: &Tensor,
        t5_emb: &Tensor,
        clip_pooled: &Tensor,
        timesteps: &[f64],
        guidance: f64,
        tile_cfg: &crate::pipelines::tiled::TiledConfig,
        conditioning_2d: &[Option<Tensor>],
        fill_cond_2d: Option<&FillConditioning2D>,
        bar: &indicatif::ProgressBar,
    ) -> Result<Tensor> {
        use crate::pipelines::tiled::{hann_window_2d, tile_positions, TilePos};

        let dev = canvas_latent.device();
        let dtype = canvas_latent.dtype();
        let (b_sz, c, lh, lw) = canvas_latent.dims4()?;
        if c != 16 {
            bail!(
                "denoise_tiled expected a 16-channel Flux latent (got {c}); call after \
                 noise/init prep, before token packing."
            );
        }

        // Tile dims in LATENT units (VAE downsample = 8).
        let tile_lat = (tile_cfg.tile_size as usize) / 8;
        let stride_lat = (tile_cfg.stride as usize) / 8;
        if tile_lat % 2 != 0 {
            bail!(
                "Tiled Flux: --tile-size {} produces an odd-sized latent tile ({} latent \
                 units). Flux's 2x2 patching needs an even latent tile.",
                tile_cfg.tile_size,
                tile_lat
            );
        }
        // Clamp tile dims to canvas — if the user asked for a tile
        // bigger than the canvas, the single-tile fallback below is
        // equivalent to the standard non-tiled denoise (with a Hann
        // window that's just 1 at the centre — no blending needed).
        let tile_lat = tile_lat.min(lh).min(lw);
        let stride_lat = stride_lat.min(tile_lat);

        let positions: Vec<TilePos> = tile_positions(lh, lw, tile_lat, stride_lat);

        // Shared text components (same across tiles, same across steps).
        let txt = t5_emb.clone();
        let txt_ids = Tensor::zeros((b_sz, txt.dim(1)?, 3), dtype, dev)?;
        let vec_ = clip_pooled.clone();

        // Hann window at latent resolution. Shape `(1, 1, tile_lat,
        // tile_lat)`, broadcast-multiplies 16ch noise predictions
        // cleanly.
        let win = hann_window_2d(tile_lat, dev, dtype)?;
        let guidance_t = Tensor::full(guidance as f32, b_sz, dev)?;

        let mut canvas = canvas_latent.clone();
        let num_steps = timesteps.windows(2).count().max(1);

        for (step_i, window) in timesteps.windows(2).enumerate() {
            let (t_curr, t_prev) = match window {
                [a, b] => (*a, *b),
                _ => continue,
            };
            let t_vec = Tensor::full(t_curr as f32, b_sz, dev)?;
            let progress = step_i as f32 / num_steps as f32;

            // Per-step accumulators. `pred_acc` collects weighted noise
            // predictions over the full canvas; `weight_acc` tracks the
            // overlap weight at each latent pixel so we can normalise.
            let mut pred_acc = Tensor::zeros((b_sz, c, lh, lw), dtype, dev)?;
            let mut weight_acc = Tensor::zeros((1, 1, lh, lw), dtype, dev)?;

            for TilePos { y: ty, x: tx, size: sz } in positions.iter().copied() {
                // Extract tile latent: (1, 16, sz, sz).
                let tile = canvas.narrow(2, ty, sz)?.narrow(3, tx, sz)?;

                // Pack to tokens (1, sz/2 * sz/2, 64). Same patching
                // `sampling::State::new` does on the full canvas.
                let h_tokens = sz / 2;
                let w_tokens = sz / 2;
                let tile_packed = pack_latent_to_tokens(&tile)?;

                // Per-tile img_ids that point at the tile's global
                // position. The Flux RoPE positional embedding uses
                // these to compute per-axis frequencies; if every tile
                // claimed (0, 0) as its origin the tiles wouldn't agree
                // on geometry and the blend would smear.
                let h_start = (ty / 2) as u32;
                let w_start = (tx / 2) as u32;
                let zeros_ids = Tensor::zeros((h_tokens, w_tokens), dtype, dev)?;
                let h_ids = Tensor::arange(h_start, h_start + h_tokens as u32, dev)?
                    .reshape((h_tokens, 1))?
                    .broadcast_as((h_tokens, w_tokens))?
                    .to_dtype(dtype)?;
                let w_ids = Tensor::arange(w_start, w_start + w_tokens as u32, dev)?
                    .reshape((1, w_tokens))?
                    .broadcast_as((h_tokens, w_tokens))?
                    .to_dtype(dtype)?;
                let img_ids = Tensor::stack(&[zeros_ids, h_ids, w_ids], 2)?
                    .reshape((1, h_tokens * w_tokens, 3))?
                    .repeat((b_sz, 1, 1))?;

                // v0.13 phase 9: per-tile ControlNet residuals. Each
                // loaded CN that's active at this progress runs once
                // per tile with the tile-cropped + packed conditioning.
                // Residuals sum across CNs the same way the non-tiled
                // path does.
                let mut summed_double: Option<Vec<Tensor>> = None;
                let mut summed_single: Option<Vec<Tensor>> = None;
                for (cn, cond_opt) in self.controlnets.iter().zip(conditioning_2d.iter()) {
                    if !cn.active_at(progress) {
                        continue;
                    }
                    let cond_full = match cond_opt.as_ref() {
                        Some(c) => c,
                        None => continue,
                    };
                    // Crop the 2D conditioning to this tile and pack to
                    // tokens. Same patching the noise tile uses, so the
                    // CN sees a per-tile conditioning aligned with the
                    // per-tile noise.
                    let cond_tile = cond_full.narrow(2, ty, sz)?.narrow(3, tx, sz)?;
                    let cond_packed = pack_latent_to_tokens(&cond_tile)?;
                    let (d, s) = cn.net.forward(
                        &tile_packed,
                        &cond_packed,
                        &img_ids,
                        &txt,
                        &txt_ids,
                        &t_vec,
                        &vec_,
                        Some(&guidance_t),
                        cn.mode,
                        cn.scale,
                    )?;
                    summed_double = Some(merge_residuals(summed_double, d)?);
                    summed_single = Some(merge_residuals(summed_single, s)?);
                }
                let double_r = summed_double.as_deref();
                let single_r = match summed_single.as_ref() {
                    Some(v) if !v.is_empty() => Some(v.as_slice()),
                    _ => None,
                };

                // v0.16 phase 4: per-tile Flux Fill packing. When
                // `fill_cond_2d` is set the loaded variant is
                // Flux.1-Fill-dev — slice the masked-latent + mask
                // to the current tile and cat onto the noise tokens
                // to get the 384-channel input Fill's `img_in`
                // expects. Non-Fill: `flux_input` is the noise
                // tokens unchanged (64ch).
                let flux_input: Tensor = match fill_cond_2d {
                    Some(fc) => {
                        let fill_tile = pack_fill_2d_tile(fc, ty, tx, sz)?;
                        Tensor::cat(&[&tile_packed, &fill_tile], D::Minus1)?
                    }
                    None => tile_packed.clone(),
                };
                let pred = match &self.flux_model {
                    FluxBackbone::Bf16(net) => net.forward_with_residuals(
                        &flux_input,
                        &img_ids,
                        &txt,
                        &txt_ids,
                        &t_vec,
                        &vec_,
                        Some(&guidance_t),
                        double_r,
                        single_r,
                    )?,
                    FluxBackbone::Quantized(net) => net.forward_with_residuals(
                        &flux_input,
                        &img_ids,
                        &txt,
                        &txt_ids,
                        &t_vec,
                        &vec_,
                        Some(&guidance_t),
                        double_r,
                        single_r,
                    )?,
                    // v0.15 phase 1: NF4 + tiled + ControlNet all
                    // compose via forward_with_residuals (same
                    // signature as GGUF/BF16). The per-tile residual
                    // slicing happened earlier in this loop so by
                    // here `double_r`/`single_r` are already in the
                    // (1, tile_tokens, hidden) shape the backbone
                    // expects.
                    FluxBackbone::Nf4(net) => net.forward_with_residuals(
                        &flux_input,
                        &img_ids,
                        &txt,
                        &txt_ids,
                        &t_vec,
                        &vec_,
                        Some(&guidance_t),
                        double_r,
                        single_r,
                    )?,
                };

                // Unpack the (1, n_tokens, 64) prediction back to a
                // 2D latent tile (1, 16, sz, sz). Inverse of the pack
                // above.
                let pred_2d = pred
                    .reshape((b_sz, h_tokens, w_tokens, c, 2, 2))?
                    .permute((0, 3, 1, 4, 2, 5))?
                    .reshape((b_sz, c, sz, sz))?;

                // Hann-weight and add into accumulators.
                let weighted = pred_2d.broadcast_mul(&win)?;
                let pred_region = pred_acc.narrow(2, ty, sz)?.narrow(3, tx, sz)?;
                let pred_updated = (pred_region + &weighted)?;
                pred_acc = pred_acc.slice_assign(
                    &[0..b_sz, 0..c, ty..ty + sz, tx..tx + sz],
                    &pred_updated,
                )?;

                let weight_region = weight_acc.narrow(2, ty, sz)?.narrow(3, tx, sz)?;
                let weight_updated = weight_region.broadcast_add(&win)?;
                weight_acc = weight_acc.slice_assign(
                    &[0..1, 0..1, ty..ty + sz, tx..tx + sz],
                    &weight_updated,
                )?;
            }

            // Normalised per-pixel noise prediction over the whole canvas.
            let pred_canvas = pred_acc.broadcast_div(&weight_acc)?;
            canvas = (canvas + pred_canvas * (t_prev - t_curr))?;
            bar.set_position(step_i as u64);
        }

        Ok(canvas)
    }
}

/// v0.16 phase 4: 2D form of Flux Fill conditioning. Held in 2D
/// (rather than the packed-token form Flux's `img_in` consumes)
/// because the tiled denoise needs to slice both planes per tile
/// before packing — non-tiled callers pack the whole canvas once
/// via [`pack_fill_2d_full`].
///
/// Layout invariants:
/// * `masked_latent_2d.dims4() == (1, 16, lh, lw)` — VAE-encoded init,
///   masked region zeroed in pixel space before encode.
/// * `mask_2d.dims4() == (1, 1, lh*8, lw*8)` — binary image-space
///   mask (1.0 = inpaint, 0.0 = preserve).
struct FillConditioning2D {
    masked_latent_2d: Tensor,
    mask_2d: Tensor,
}

/// v0.16 phase 4: pack the full-canvas Fill conditioning into the
/// 320-channel-per-token form Flux Fill's `img_in` consumes. Mirrors
/// the inline packing the pre-phase-4 `encode_fill_conditioning` did
/// — split out so both the non-tiled denoise (full canvas) and the
/// tiled denoise (per tile via [`pack_fill_2d_tile`]) share the same
/// reshape kernels.
fn pack_fill_2d_full(cond: &FillConditioning2D) -> Result<Tensor> {
    let (_b, c, lh, lw) = cond.masked_latent_2d.dims4()?;
    let masked_packed = cond
        .masked_latent_2d
        .reshape((1, c, lh / 2, 2, lw / 2, 2))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((1, lh / 2 * lw / 2, c * 4))?;
    let (_b, _c, h, w) = cond.mask_2d.dims4()?;
    let mask_packed = cond
        .mask_2d
        .reshape((1, 1, h / 16, 16, w / 16, 16))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((1, h / 16 * w / 16, 16 * 16))?;
    Ok(Tensor::cat(&[&masked_packed, &mask_packed], D::Minus1)?)
}

/// v0.16 phase 4: pack a per-tile slice of Fill conditioning. The
/// tile is specified in **latent units** — same convention the rest
/// of `denoise_tiled` uses (Y/X offset = tile origin in the
/// `(1, 16, lh, lw)` latent; `sz` = tile side in latent units).
///
/// The image-space mask slice runs at 8× the tile coords (since
/// latent = pixel / 8), and packs at 16-pixel granularity (256
/// raw mask values per Flux token).
///
/// Caller must guarantee:
/// * `sz` even (Flux's 2×2 patching), else the packing reshape errors.
/// * `(ty + sz, tx + sz) ≤ (lh, lw)` — tile fits in the canvas.
fn pack_fill_2d_tile(
    cond: &FillConditioning2D,
    ty: usize,
    tx: usize,
    sz: usize,
) -> Result<Tensor> {
    let (_b, c, _lh, _lw) = cond.masked_latent_2d.dims4()?;
    let masked_tile = cond
        .masked_latent_2d
        .narrow(2, ty, sz)?
        .narrow(3, tx, sz)?;
    let masked_packed = masked_tile
        .reshape((1, c, sz / 2, 2, sz / 2, 2))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((1, sz / 2 * sz / 2, c * 4))?;
    // Mask slice in pixel space: 8× the latent tile coords.
    let pixel_y = ty * 8;
    let pixel_x = tx * 8;
    let pixel_sz = sz * 8;
    let mask_tile = cond
        .mask_2d
        .narrow(2, pixel_y, pixel_sz)?
        .narrow(3, pixel_x, pixel_sz)?;
    let mask_packed = mask_tile
        .reshape((1, 1, pixel_sz / 16, 16, pixel_sz / 16, 16))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((1, (pixel_sz / 16) * (pixel_sz / 16), 16 * 16))?;
    Ok(Tensor::cat(&[&masked_packed, &mask_packed], D::Minus1)?)
}

/// Sum two per-block residual lists, padding the shorter one to the
/// v0.13 phase 9: pack a 2D Flux latent `(1, 16, lh, lw)` into the
/// per-token form `(1, lh/2 * lw/2, 64)` that Flux's `img_in` consumes.
/// Same 2×2 patching the upstream `sampling::State::new` does on the
/// noise latent. Used by both `encode_conditioning` (whole canvas) and
/// the tiled denoise loop (per tile).
/// v0.18 phase 2b: the 17 BFL-recommended Kontext resolutions
/// (target_w, target_h). Spans the full 9:21 → 21:9 aspect range
/// at ~1M-token budgets. All entries are multiples of 16 so VAE +
/// Flux 2x2 packing land clean. Order is conventional (tall →
/// square → wide). Matches diffusers' `PREFERRED_KONTEXT_RESOLUTIONS`.
pub const KONTEXT_BUCKETS: &[(u32, u32)] = &[
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
];

/// Snap `(w, h)` to the closest Kontext bucket by aspect ratio.
/// Used when `--kontext-bucket` is set; otherwise the user's
/// requested size flows through unchanged.
pub fn snap_to_kontext_bucket(w: u32, h: u32) -> (u32, u32) {
    debug_assert!(!KONTEXT_BUCKETS.is_empty());
    let target = w as f64 / h.max(1) as f64;
    KONTEXT_BUCKETS
        .iter()
        .copied()
        .min_by(|(aw, ah), (bw, bh)| {
            let a_diff = (*aw as f64 / *ah as f64 - target).abs();
            let b_diff = (*bw as f64 / *bh as f64 - target).abs();
            a_diff
                .partial_cmp(&b_diff)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((1024, 1024))
}

fn pack_latent_to_tokens(z: &Tensor) -> Result<Tensor> {
    let (b, c, lh, lw) = z.dims4()?;
    if lh % 2 != 0 || lw % 2 != 0 {
        anyhow::bail!(
            "pack_latent_to_tokens: latent dims must be even (got {lh}x{lw}); Flux \
             patches by 2x2."
        );
    }
    let packed = z
        .reshape((b, c, lh / 2, 2, lw / 2, 2))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((b, lh / 2 * lw / 2, c * 4))?;
    Ok(packed)
}

/// longer one's length (missing entries contribute zero). Returns the
/// merged Vec; the inputs are consumed.
fn merge_residuals(acc: Option<Vec<Tensor>>, new: Vec<Tensor>) -> Result<Vec<Tensor>> {
    let mut acc = acc.unwrap_or_default();
    for (i, t) in new.into_iter().enumerate() {
        if i < acc.len() {
            acc[i] = (&acc[i] + &t)?;
        } else {
            acc.push(t);
        }
    }
    Ok(acc)
}

// =====================================================================
// Public single-shot entry — preserves the existing API used by t2i::run.
// =====================================================================

pub async fn run(req: Request) -> Result<()> {
    let mut p = Pipeline::load(LoadRequest {
        variant: req.variant,
        repo: req.repo,
        device: req.device,
        loras: req.loras,
        lora_scale: req.lora_scale,
        controlnets: req.controlnets,
        quantize_t5: req.quantize_t5,
        flux_quant_level: req.flux_quant_level,
        t5_quant_level: req.t5_quant_level,
        redux: req.redux,
    })
    .await?;
    p.generate(&GenRequest {
        prompt: req.prompt,
        width: req.width,
        height: req.height,
        count: req.count,
        steps: req.steps,
        guidance: req.guidance,
        seed: req.seed,
        out_dir: req.out_dir,
        conditioning: req.conditioning,
        init_image: req.init_image,
        mask: req.mask,
        strength: req.strength,
        concept_conditioning: req.concept_conditioning,
        tiled: req.tiled,
        redux_images: req.redux_images,
        kontext_bucket: req.kontext_bucket,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // v0.13 phase 5 — quant-level validation.

    #[test]
    fn validates_known_flux_quant_level() {
        assert!(validate_quant_level("Q4_K_S", FLUX_QUANT_LEVELS, "Flux GGUF").is_ok());
        assert!(validate_quant_level("Q8_0", FLUX_QUANT_LEVELS, "Flux GGUF").is_ok());
        assert!(validate_quant_level("F16", FLUX_QUANT_LEVELS, "Flux GGUF").is_ok());
    }

    #[test]
    fn quant_level_case_insensitive() {
        // CLI users sometimes lowercase; the validator should still
        // accept since GGUF filenames are case-preserved by city96 but
        // user typing varies.
        assert!(validate_quant_level("q4_k_s", FLUX_QUANT_LEVELS, "Flux GGUF").is_ok());
    }

    #[test]
    fn rejects_unknown_quant_level() {
        // city96 doesn't publish Q1_K — should bail loud.
        let err = validate_quant_level("Q1_K", FLUX_QUANT_LEVELS, "Flux GGUF").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Q4_K_S"), "error should list supported levels: {msg}");
    }

    #[test]
    fn rejects_t5_only_level_for_flux() {
        // Q3_K_L is published for T5 but not for Flux — should bail
        // when used as the Flux level.
        assert!(validate_quant_level("Q3_K_L", FLUX_QUANT_LEVELS, "Flux GGUF").is_err());
        assert!(validate_quant_level("Q3_K_L", T5_QUANT_LEVELS, "T5 GGUF").is_ok());
    }

    // v0.13 phase 6 — Flux ControlNet step gating. Half-open `[start,
    // end)` matches the SD path so the same `start=0.0:end=0.4` flag
    // string means the same thing on both backbones. Note: we can't
    // construct a `LoadedFluxControlNet` without real CN weights, so
    // these tests cover the gate predicate via a free function with
    // the same body as `LoadedFluxControlNet::active_at`.
    fn active_at_window(start: f32, end: f32, progress: f32) -> bool {
        progress >= start && progress < end
    }

    #[test]
    fn cn_gate_full_window_active_every_step() {
        for i in 0..28 {
            let progress = i as f32 / 28.0;
            assert!(
                active_at_window(0.0, 1.0, progress),
                "default window must include progress={progress}"
            );
        }
    }

    #[test]
    fn cn_gate_early_window_drops_late_steps() {
        // start=0.0, end=0.4 → active for first 40% of steps.
        // 28 steps → first ~11 active, rest inactive.
        let early_active = (0..28)
            .filter(|i| active_at_window(0.0, 0.4, *i as f32 / 28.0))
            .count();
        assert!((10..=12).contains(&early_active), "got {early_active} active steps");
    }

    #[test]
    fn cn_gate_late_window_drops_early_steps() {
        // start=0.6, end=1.0 → active only for the last 40%.
        // step 0 progress=0.0 → inactive; step 27 progress~=0.96 → active.
        assert!(!active_at_window(0.6, 1.0, 0.0));
        assert!(active_at_window(0.6, 1.0, 0.7));
        assert!(active_at_window(0.6, 1.0, 0.95));
        // Right edge is half-open — progress=end must be inactive.
        assert!(!active_at_window(0.6, 1.0, 1.0));
    }

    #[test]
    fn cn_gate_zero_width_window_never_active() {
        // start == end: half-open window is empty.
        assert!(!active_at_window(0.5, 0.5, 0.5));
        assert!(!active_at_window(0.5, 0.5, 0.0));
    }

    // v0.16 phase 4 — Tiled + Flux Fill packing.

    fn dummy_fill_cond_2d(lh: usize, lw: usize) -> FillConditioning2D {
        use candle_core::Device;
        // Distinct values per pixel so slicing bugs would change the
        // packed tensor's content (vs. all-zeros which would mask
        // the off-by-one).
        let total_lat: usize = 1 * 16 * lh * lw;
        let lat: Vec<f32> = (0..total_lat).map(|i| i as f32 * 0.001).collect();
        let masked_latent_2d = Tensor::from_vec(lat, (1, 16, lh, lw), &Device::Cpu).unwrap();
        let h = lh * 8;
        let w = lw * 8;
        let total_mask: usize = h * w;
        let mask: Vec<f32> = (0..total_mask).map(|i| (i % 2) as f32).collect();
        let mask_2d = Tensor::from_vec(mask, (1, 1, h, w), &Device::Cpu).unwrap();
        FillConditioning2D { masked_latent_2d, mask_2d }
    }

    #[test]
    fn pack_fill_2d_full_emits_320_channel_token_tensor() {
        // Full canvas: lh=8, lw=8 → 4×4 tokens (each spans 2 latent
        // units → 16 pixels) → 16 tokens, 64 (latent) + 256 (mask)
        // = 320 channels per token.
        let cond = dummy_fill_cond_2d(8, 8);
        let packed = pack_fill_2d_full(&cond).unwrap();
        let (b, n, c) = packed.dims3().unwrap();
        assert_eq!((b, n, c), (1, 16, 320));
    }

    #[test]
    fn pack_fill_2d_tile_emits_tile_sized_token_tensor() {
        // Latent canvas 16×16 → 8×8 tokens at full canvas. Tile
        // (sz=4 latent units) → 2×2 tokens → 4 tokens per tile.
        let cond = dummy_fill_cond_2d(16, 16);
        let tile = pack_fill_2d_tile(&cond, 0, 0, 4).unwrap();
        let (b, n, c) = tile.dims3().unwrap();
        assert_eq!((b, n, c), (1, 4, 320));
    }

    #[test]
    fn pack_fill_2d_tile_at_offset_matches_full_canvas_slice() {
        // The tile at origin (4, 4) with sz=4 should produce the same
        // 4-token packing as if we'd sliced the full-canvas packed
        // form at the matching token offset. The token grid is
        // lh/2 × lw/2 = 8×8 = 64 tokens; the tile's tokens at offset
        // (2, 2) tile size 2×2 are rows 2-3 × cols 2-3 of that grid.
        let cond = dummy_fill_cond_2d(16, 16);
        let full = pack_fill_2d_full(&cond).unwrap();
        let tile = pack_fill_2d_tile(&cond, 4, 4, 4).unwrap();

        // Extract the matching 4 tokens from `full` by gather: token
        // (row=2, col=2), (2, 3), (3, 2), (3, 3) in row-major.
        let expected_indices = [2 * 8 + 2, 2 * 8 + 3, 3 * 8 + 2, 3 * 8 + 3];
        for (out_i, &full_i) in expected_indices.iter().enumerate() {
            let from_tile: Vec<f32> = tile.i((0, out_i, ..))
                .unwrap()
                .to_vec1()
                .unwrap();
            let from_full: Vec<f32> = full.i((0, full_i, ..))
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(
                from_tile, from_full,
                "tile token {out_i} (canvas token {full_i}) mismatch"
            );
        }
    }

    // v0.18 Kontext phase 2 — reference token + img_ids shape.

    #[test]
    fn pack_latent_to_tokens_matches_kontext_seq_shape() {
        // 16ch VAE latent at 64×64 → 32×32 tokens × 64 channels.
        let z =
            Tensor::zeros((1, 16, 64, 64), DType::F32, &Device::Cpu).unwrap();
        let packed = pack_latent_to_tokens(&z).unwrap();
        assert_eq!(packed.dims(), &[1, 32 * 32, 64]);
    }

    // v0.18 Kontext phase 2b — aspect-bucket snap.

    #[test]
    fn kontext_buckets_are_seventeen_and_multiple_of_sixteen() {
        assert_eq!(KONTEXT_BUCKETS.len(), 17);
        for (w, h) in KONTEXT_BUCKETS {
            assert!(w % 16 == 0, "bucket width {w} not multiple of 16");
            assert!(h % 16 == 0, "bucket height {h} not multiple of 16");
        }
    }

    #[test]
    fn snap_square_to_1024() {
        // 768x768 aspect = 1.0 → closest bucket is (1024, 1024).
        assert_eq!(snap_to_kontext_bucket(768, 768), (1024, 1024));
        assert_eq!(snap_to_kontext_bucket(512, 512), (1024, 1024));
        assert_eq!(snap_to_kontext_bucket(2048, 2048), (1024, 1024));
    }

    #[test]
    fn snap_widescreen_to_landscape_bucket() {
        // 21:9 ≈ 2.33 → closest bucket is (1568, 672) at 2.33.
        let (w, h) = snap_to_kontext_bucket(2100, 900);
        assert_eq!((w, h), (1568, 672));
    }

    #[test]
    fn snap_portrait_to_portrait_bucket() {
        // 9:21 ≈ 0.43 → closest bucket is (672, 1568) at 0.43.
        let (w, h) = snap_to_kontext_bucket(900, 2100);
        assert_eq!((w, h), (672, 1568));
    }

    #[test]
    fn snap_4_3_to_nearest() {
        // 4:3 ≈ 1.33 → closest bucket should be one of the moderate
        // landscape entries; exact bucket depends on which has the
        // closest ratio. (1248, 832) has ratio 1.5; (1184, 880) has
        // ratio ~1.345. Pick whichever is closer.
        let (w, h) = snap_to_kontext_bucket(1600, 1200);
        // Verify the result is *one of* the published buckets, and
        // that its aspect ratio is closer to 4/3 than any other.
        assert!(KONTEXT_BUCKETS.contains(&(w, h)));
        let chosen = w as f64 / h as f64;
        let target = 4.0 / 3.0;
        for (bw, bh) in KONTEXT_BUCKETS {
            let other = *bw as f64 / *bh as f64;
            if (*bw, *bh) == (w, h) {
                continue;
            }
            assert!(
                (chosen - target).abs() <= (other - target).abs(),
                "chosen {w}x{h} (ratio {chosen}) is farther from 4:3 than {bw}x{bh} (ratio {other})"
            );
        }
    }

    #[test]
    fn kontext_ref_img_ids_axis0_is_one() {
        // Replicate the encode_kontext_reference img_ids construction
        // standalone so we can assert the axis-0 marker without
        // touching the VAE. h2 = w2 = 4 → 16 ref tokens.
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let (h2, w2) = (4usize, 4usize);
        let ones = Tensor::full(1f32, (h2, w2), &dev).unwrap().to_dtype(dtype).unwrap();
        let h_ids = Tensor::arange(0u32, h2 as u32, &dev)
            .unwrap()
            .reshape(((), 1))
            .unwrap()
            .broadcast_as((h2, w2))
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let w_ids = Tensor::arange(0u32, w2 as u32, &dev)
            .unwrap()
            .reshape((1, ()))
            .unwrap()
            .broadcast_as((h2, w2))
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let ref_ids = Tensor::stack(&[&ones, &h_ids, &w_ids], 2)
            .unwrap()
            .reshape((1, h2 * w2, 3))
            .unwrap();
        assert_eq!(ref_ids.dims(), &[1, 16, 3]);
        // Every row's axis-0 column must be 1 (the reference marker).
        // Compare against a (1, 16) all-ones tensor.
        let axis0: Vec<f32> = ref_ids
            .narrow(2, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for v in &axis0 {
            assert!((v - 1.0).abs() < 1e-6, "expected axis-0=1 marker, got {v}");
        }
    }
}
