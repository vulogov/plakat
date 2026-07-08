//! Stable Diffusion 3 / 3.5 text-to-image pipeline (v0.14 phase 1a).
//!
//! Architecture (per Stability AI's SD3 paper + the 3.5 model card):
//!
//! * **MMDiT** transformer (candle ships this — `models::mmdit`) —
//!   replaces the SD UNet with a joint text/image diffusion
//!   transformer. SD3.5 Medium uses depth=24 / hidden=1536;
//!   SD3.5 Large uses depth=38 / hidden=2432.
//!
//! * **Triple text encoder**:
//!     - CLIP-L (text_encoder/): 77-token sequence, 768d hidden,
//!       768d pooled.
//!     - CLIP-G (text_encoder_2/): 77-token sequence, 1280d hidden,
//!       1280d pooled (projected via `text_projection`).
//!     - T5-XXL (text_encoder_3/): 256-token sequence, 4096d hidden.
//!
//! * **Conditioning concat** the MMDiT expects:
//!     - `y` (pooled, 2048d) = `[CLIP-G_pooled (1280) || CLIP-L_pooled (768)]`
//!     - `context` (B, 77+t5_seq, 4096) =
//!         `[ pad([CLIP-L_hidden || CLIP-G_hidden], 2048→4096), T5_hidden ]`
//!       — CLIP halves are concatenated along the channel dim (77×2048),
//!       zero-padded to 4096, then T5's 4096-d hidden is appended
//!       along the sequence dim.
//!
//! * **16-channel VAE** (`vae/`) — standard `AutoEncoderKL` with
//!   `latent_channels: 16`, `use_quant_conv: false`. Pixel-space
//!   convention: `[-1, 1]` in, `[-1, 1]` out. Latent normalisation:
//!   `z_norm = (z - shift) * scale` with `scale = 1.5305`,
//!   `shift = 0.0609`. Decode: `decode((z / scale) + shift)`.
//!
//! * **Rectified-flow sampler** — same flow-match update Flux uses
//!   (`x_{t-1} = x_t + pred * (t_prev - t_curr)`) with a different
//!   time-shift transform: SD3 uses `f(t) = shift * t / (1 + (shift - 1) * t)`
//!   over the linear `[0, 1]` schedule, default `shift = 3.0` for
//!   3.5 Medium.
//!
//! * **Classifier-free guidance** — unlike Flux, SD3 *does* use CFG.
//!   We double-batch `[neg, pos]` per step and blend via
//!   `pred = neg + guidance * (pos - neg)`.
//!
//! ## Phase 1a / 8a / v0.15 phase 2 scope
//!
//! * t2i on Sd3Medium, Sd35Medium, Sd35Large, Sd35LargeTurbo.
//! * v0.15 phase 2: img2img + inpaint (RePaint-style) for the full
//!   lineup. VAE-encoded init lerps with fresh noise at `t=strength`;
//!   the timestep schedule is truncated to entries below `strength`
//!   with `strength` itself prepended (same trick `FluxImg2Img` uses).
//!   Inpaint keeps unmasked pixels on the flow trajectory of the init
//!   by per-step blending `latent = mask * latent + (1-mask) *
//!   lerp(init, eps, t_next)` where `eps` is the same noise sample the
//!   start latent was built with.
//!
//! Still deferred (v0.15+): LoRA (phase 3), Canny/Depth-dev (phase 4),
//! tiled (phase 5), ControlNet (phase 6).

use anyhow::{Context, Result, anyhow, bail};
use candle_core::Module;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
// v0.15 phase 6a: route MMDiT through the vendored module
// (`mmdit_inner`) instead of candle's upstream. The vendor exposes
// `forward_with_residuals` for the v0.16 SD3 ControlNet integration.
// Pre-phase-6 callers see identical behaviour — the vendor's
// `forward(...)` delegates to `forward_with_residuals(... None)`.
use crate::pipelines::mmdit_inner as mmdit;
use candle_transformers::models::stable_diffusion::vae as sdvae;
// v1.10.0: SD3.5 Textual-Inversion training needs a differentiable
// forward-from-input-embeds + token-embedding access on T5, which candle's
// `t5::T5EncoderModel` doesn't expose. We use a faithful vendored copy
// (byte-equivalent encoder math + `embed_tokens` / `forward_from_input_embeds`)
// aliased as `t5` so every existing call site (`t5::Config`,
// `T5EncoderModel::load`, `.forward`) is unchanged — inference is identical.
use crate::pipelines::vendored_t5 as t5;

/// Build the MMDiT VarBuilder, transparently remapping a **diffusers**-format
/// transformer checkpoint to plakat's SAI/mmdit naming.
///
/// plakat's MMDiT loader was written for the original SAI single-file layout
/// (`joint_blocks.N.x_block.attn.qkv`, fused QKV). Stability's diffusers
/// checkpoints (what `transformer/diffusion_pytorch_model.safetensors`
/// ships) use `transformer_blocks.N.attn.to_q/to_k/to_v` (split) — so the
/// SAI loader 404s on every tensor. When we detect the diffusers layout we
/// load the tensors eagerly, fuse Q/K/V, rename, and hand the MMDiT a
/// `from_tensors` VarBuilder. SAI-format checkpoints take the mmap path
/// unchanged (byte-identical to before).
fn build_mmdit_vb<'a>(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
    depth: usize,
) -> Result<VarBuilder<'a>> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading MMDiT tensors from {}", path.display()))?;
    if tensors.contains_key("transformer_blocks.0.attn.to_q.weight") {
        let map = remap_diffusers_mmdit(&tensors, depth, dtype)
            .context("remapping diffusers MMDiT → SAI layout")?;
        Ok(VarBuilder::from_tensors(map, dtype, device))
    } else {
        Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? })
    }
}

/// Remap a diffusers SD3/SD3.5 transformer tensor map to plakat's SAI naming.
/// Fuses split Q/K/V into a single `qkv`, renames every tensor, and emits
/// the dual-attention `attn2` block (SD3.5) when present. The final block is
/// `context_pre_only` (no context output proj / MLP) — handled below.
pub(crate) fn remap_diffusers_mmdit(
    d: &std::collections::HashMap<String, Tensor>,
    depth: usize,
    dtype: DType,
) -> Result<std::collections::HashMap<String, Tensor>> {
    use std::collections::HashMap;
    let mut m: HashMap<String, Tensor> = HashMap::new();
    let take = |k: &str| -> Result<Tensor> {
        d.get(k)
            .ok_or_else(|| anyhow!("missing diffusers tensor `{k}`"))?
            .to_dtype(dtype)
            .map_err(Into::into)
    };
    let fuse = |a: &str, b: &str, c: &str| -> Result<Tensor> {
        Ok(Tensor::cat(&[&take(a)?, &take(b)?, &take(c)?], 0)?)
    };

    // ---- top-level ----
    for (sai, dif) in [
        ("x_embedder.proj.weight", "pos_embed.proj.weight"),
        ("x_embedder.proj.bias", "pos_embed.proj.bias"),
        ("pos_embed", "pos_embed.pos_embed"),
        ("t_embedder.mlp.0.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
        ("t_embedder.mlp.0.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
        ("t_embedder.mlp.2.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
        ("t_embedder.mlp.2.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
        ("y_embedder.mlp.0.weight", "time_text_embed.text_embedder.linear_1.weight"),
        ("y_embedder.mlp.0.bias", "time_text_embed.text_embedder.linear_1.bias"),
        ("y_embedder.mlp.2.weight", "time_text_embed.text_embedder.linear_2.weight"),
        ("y_embedder.mlp.2.bias", "time_text_embed.text_embedder.linear_2.bias"),
        ("context_embedder.weight", "context_embedder.weight"),
        ("context_embedder.bias", "context_embedder.bias"),
        ("final_layer.linear.weight", "proj_out.weight"),
        ("final_layer.linear.bias", "proj_out.bias"),
        ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight"),
        ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias"),
    ] {
        m.insert(sai.to_string(), take(dif)?);
    }

    // ---- joint blocks ----
    for i in 0..depth {
        let dp = format!("transformer_blocks.{i}");
        let jp = format!("joint_blocks.{i}");
        let last = i == depth - 1;

        for wb in ["weight", "bias"] {
            // x-stream joint QKV (fused) + context joint QKV (fused).
            m.insert(
                format!("{jp}.x_block.attn.qkv.{wb}"),
                fuse(&format!("{dp}.attn.to_q.{wb}"), &format!("{dp}.attn.to_k.{wb}"), &format!("{dp}.attn.to_v.{wb}"))?,
            );
            m.insert(
                format!("{jp}.context_block.attn.qkv.{wb}"),
                fuse(&format!("{dp}.attn.add_q_proj.{wb}"), &format!("{dp}.attn.add_k_proj.{wb}"), &format!("{dp}.attn.add_v_proj.{wb}"))?,
            );
            m.insert(format!("{jp}.x_block.attn.proj.{wb}"), take(&format!("{dp}.attn.to_out.0.{wb}"))?);
            // x adaLN + MLP.
            m.insert(format!("{jp}.x_block.adaLN_modulation.1.{wb}"), take(&format!("{dp}.norm1.linear.{wb}"))?);
            m.insert(format!("{jp}.x_block.mlp.fc1.{wb}"), take(&format!("{dp}.ff.net.0.proj.{wb}"))?);
            m.insert(format!("{jp}.x_block.mlp.fc2.{wb}"), take(&format!("{dp}.ff.net.2.{wb}"))?);
            m.insert(format!("{jp}.context_block.adaLN_modulation.1.{wb}"), take(&format!("{dp}.norm1_context.linear.{wb}"))?);
            if !last {
                m.insert(format!("{jp}.context_block.attn.proj.{wb}"), take(&format!("{dp}.attn.to_add_out.{wb}"))?);
                m.insert(format!("{jp}.context_block.mlp.fc1.{wb}"), take(&format!("{dp}.ff_context.net.0.proj.{wb}"))?);
                m.insert(format!("{jp}.context_block.mlp.fc2.{wb}"), take(&format!("{dp}.ff_context.net.2.{wb}"))?);
            }
        }
        // QK-norm (per-head RMSNorm scales, weight-only).
        m.insert(format!("{jp}.x_block.attn.ln_q.weight"), take(&format!("{dp}.attn.norm_q.weight"))?);
        m.insert(format!("{jp}.x_block.attn.ln_k.weight"), take(&format!("{dp}.attn.norm_k.weight"))?);
        m.insert(format!("{jp}.context_block.attn.ln_q.weight"), take(&format!("{dp}.attn.norm_added_q.weight"))?);
        m.insert(format!("{jp}.context_block.attn.ln_k.weight"), take(&format!("{dp}.attn.norm_added_k.weight"))?);

        // Dual attention (SD3.5 blocks 0..N): x-stream self-attention.
        if d.contains_key(&format!("{dp}.attn2.to_q.weight")) {
            for wb in ["weight", "bias"] {
                m.insert(
                    format!("{jp}.x_block.attn2.qkv.{wb}"),
                    fuse(&format!("{dp}.attn2.to_q.{wb}"), &format!("{dp}.attn2.to_k.{wb}"), &format!("{dp}.attn2.to_v.{wb}"))?,
                );
                m.insert(format!("{jp}.x_block.attn2.proj.{wb}"), take(&format!("{dp}.attn2.to_out.0.{wb}"))?);
            }
            m.insert(format!("{jp}.x_block.attn2.ln_q.weight"), take(&format!("{dp}.attn2.norm_q.weight"))?);
            m.insert(format!("{jp}.x_block.attn2.ln_k.weight"), take(&format!("{dp}.attn2.norm_k.weight"))?);
        }
    }
    Ok(m)
}
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::sdxl_clip::SdxlClipGTextTransformer;
use crate::ui::progress;

/// CLIP EOT-token id — shared across CLIP-L and CLIP-G in diffusers.
const CLIP_EOT: u32 = 49407;
const CLIP_BOS: u32 = 49406;

/// VAE latent normalisation constants for SD3 / SD3.5. Match the
/// `scaling_factor` / `shift_factor` baked into the diffusers
/// `vae/config.json` for the 16-channel AE.
const VAE_SCALE: f64 = 1.5305;
const VAE_SHIFT: f64 = 0.0609;

/// SD3 / SD3.5 variant.
///
/// * `Sd3Medium` — the original v0.5 Stable Diffusion 3 Medium
///   (June 2024). 2B parameters. Known anatomy issues; SD3.5 is the
///   recommended baseline today.
/// * `Sd35Medium` — SD3.5 Medium (Oct 2024). Same 2.5B-param MMDiT
///   shape as SD3 but with `pos_embed_max_size = 384` (vs 192) so it
///   handles up to 1536² without positional aliasing.
/// * `Sd35Large` — SD3.5 Large. 8B-parameter MMDiT (depth=38). The
///   flagship. ~17 GB BF16 weights.
/// * `Sd35LargeTurbo` — 4-step distillation of Sd35Large. Recommended
///   `guidance: 0.0`, `steps: 4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Sd3Medium,
    Sd35Medium,
    Sd35Large,
    Sd35LargeTurbo,
}

impl Variant {
    fn mmdit_config(self) -> mmdit::Config {
        match self {
            Self::Sd3Medium => mmdit::Config::sd3_medium(),
            Self::Sd35Medium => mmdit::Config::sd3_5_medium(),
            // SD3.5 Large + Turbo share the same MMDiT shape; the
            // turbo distillation only changes the sampling schedule.
            Self::Sd35Large | Self::Sd35LargeTurbo => mmdit::Config::sd3_5_large(),
        }
    }

    /// T5-XXL sequence length budget per the SD3 paper. 256 is the
    /// canonical value Stability used in training across the lineup;
    /// longer prompts get truncated.
    fn t5_seq_len(self) -> usize {
        256
    }

    /// Default time-shift parameter for the rectified-flow schedule.
    /// Diffusers' `FlowMatchEulerDiscreteScheduler` uses 3.0 for SD3.5
    /// Medium at 1024². Sd35Large + Sd35LargeTurbo recommend higher
    /// shift values matching the increased token count of the deeper
    /// transformer; Turbo's 4-step schedule benefits from shift = 1.0
    /// (linear) since the schedule is so short.
    fn default_time_shift(self) -> f64 {
        match self {
            Self::Sd35LargeTurbo => 1.0,
            Self::Sd35Large => 3.0,
            Self::Sd35Medium | Self::Sd3Medium => 3.0,
        }
    }

    pub fn default_guidance(self) -> f64 {
        match self {
            // Sd35LargeTurbo is a distillation that ignores CFG — its
            // training schedule is single-pass (no conditional /
            // unconditional pairing). Per Stability's model card,
            // guidance=0.0 (no CFG) is the recommended sampling.
            Self::Sd35LargeTurbo => 0.0,
            // Sd3Medium / Sd35Medium / Sd35Large all use the same
            // default CFG. Stability publishes a 4.5 floor across the
            // lineup.
            Self::Sd3Medium | Self::Sd35Medium | Self::Sd35Large => 4.5,
        }
    }

    pub fn default_steps(self) -> usize {
        match self {
            // Turbo is a 4-step distillation. Going past 4 typically
            // hurts quality — the distillation collapses the
            // intermediate timesteps.
            Self::Sd35LargeTurbo => 4,
            Self::Sd3Medium | Self::Sd35Medium | Self::Sd35Large => 28,
        }
    }

    /// v0.15 phase 3: MMDiT hidden size = `head_size * depth`. SD3 /
    /// SD3.5-Medium both use 64 * 24 = 1536; SD3.5-Large is
    /// 64 * 38 = 2432. Used by the LoRA resolver for QKV-fusion row
    /// slicing — wrong value silently mis-slices into wrong rows of
    /// the fused QKV tensor, producing scrambled outputs.
    pub fn mmdit_hidden_size(self) -> usize {
        let cfg = self.mmdit_config();
        cfg.head_size * cfg.depth
    }
}

pub struct Request {
    pub prompt: String,
    pub negative: String,
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
    /// v0.15 phase 2: img2img init image. `None` is pure t2i. When set
    /// AND `mask` is `None`, every pixel is re-denoised at `strength`
    /// (img2img). When set AND `mask` is `Some`, only the masked
    /// region re-denoises; unmasked pixels stay on the init's flow
    /// trajectory (RePaint-style inpaint).
    pub init_image: Option<PathBuf>,
    /// v0.15 phase 2: per-image binary or feathered mask. Same
    /// conventions as `imaging::mask::Mask` — white (1.0) = inpaint,
    /// black (0.0) = preserve.
    pub mask: Option<PathBuf>,
    /// v0.15 phase 2: feather radius (pixels) applied to the mask
    /// before downsampling to latent resolution. Soft edges hide
    /// the inpaint↔preserve boundary.
    pub mask_feather: u32,
    /// v0.15 phase 2: when `true`, the loaded mask is inverted
    /// before feathering (handles sources where black = inpaint).
    pub mask_invert: bool,
    /// v0.15 phase 2: img2img / inpaint strength in `[0, 1]`. `None`
    /// defaults to 0.6 for img2img, 1.0 for inpaint (matches Flux +
    /// SD img2img defaults). Ignored when `init_image` is `None`.
    pub strength: Option<f32>,
    /// v0.15 phase 3: PEFT-format SD3 LoRAs to merge at load time.
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    /// v0.15 phase 3: per-LoRA scale multiplier.
    pub lora_scale: f32,
    /// v0.15 phase 5: tiled MultiDiffusion-style denoise. `Some(cfg)`
    /// splits the latent into overlapping `tile_size`-pixel windows
    /// and Hann-blends MMDiT predictions per step. Lets SD3 produce
    /// 4K+ outputs without exceeding `pos_embed_max_size` (192 for
    /// SD3 / SD3.5-Large, 384 for SD3.5-Medium). `None` = single-pass
    /// canvas (phase 1a behaviour).
    pub tiled: Option<crate::pipelines::tiled::TiledConfig>,
    /// Regional prompting: per-region prompts (MultiDiffusion velocity blend).
    /// Empty = single-prompt. Wired from `--region` / the scenario `regions` key.
    pub regions: Vec<crate::pipelines::tiled::RegionSpec>,
    /// v0.16 phase 3e: SD3 ControlNet stack to load. Each entry
    /// carries the InstantX repo + per-instance runtime knobs
    /// (`scale`, `conditioning`, `start`, `end`). Empty Vec means no
    /// CN — byte-identical to the pre-phase-3 schedule. Threaded
    /// through to `LoadRequest.controlnets`, then VAE-encoded once
    /// per `generate` call into the cached per-slot conditioning
    /// latents used in `predict_velocity_full`.
    pub controlnets: Vec<crate::pipelines::sd3_controlnet::Sd3ControlNetLoad>,
    /// v1.10.0: Textual-Inversion embeddings (`--embedding`). Threaded into
    /// `LoadRequest.embeddings`. Empty = no TI (byte-identical encode).
    pub embeddings: Vec<crate::pipelines::embedding::EmbeddingSpec>,
    /// v0.20: output container — see `GenRequest::output_format`.
    pub output_format: crate::imaging::io::OutputFormat,
}

pub struct LoadRequest {
    pub variant: Variant,
    pub repo: String,
    pub device: Device,
    /// v0.15 phase 3: PEFT-format LoRA stack to merge into the MMDiT
    /// weights at load time. Empty = no merge (pre-phase-3 path,
    /// byte-identical). Diffusers SD3 LoRAs use `transformer.` as the
    /// PEFT root; the resolver strips it automatically.
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    /// v0.15 phase 3: per-LoRA scale multiplier applied on top of each
    /// LoRA's own `:scale` suffix. Same semantics as Flux/SD LoRA's
    /// `--lora-scale`. 1.0 is the default.
    pub lora_scale: f32,
    /// v0.16 phase 3: zero or more SD3 ControlNets to preload alongside
    /// the MMDiT backbone. Each load spec carries the InstantX repo +
    /// per-instance runtime knobs (scale, conditioning, step-gating).
    /// Empty disables CN entirely (byte-identical to pre-3 behaviour).
    pub controlnets: Vec<crate::pipelines::sd3_controlnet::Sd3ControlNetLoad>,
    /// v1.10.0: Textual-Inversion embeddings to load (`--embedding`). Each is a
    /// triple file (`clip_l`+`clip_g`+`t5`) trained by `embedding train --base
    /// sd35`. Empty = no TI; the encode path is then byte-identical to the
    /// verified pre-TI behaviour. The trigger is registered as an added token
    /// in each tokenizer and runtime-spliced into the prompt's embeddings.
    pub embeddings: Vec<crate::pipelines::embedding::EmbeddingSpec>,
}

pub struct GenRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    /// v0.15 phase 2: img2img / inpaint surface. See `Request` for
    /// field semantics.
    pub init_image: Option<PathBuf>,
    pub mask: Option<PathBuf>,
    pub mask_feather: u32,
    pub mask_invert: bool,
    pub strength: Option<f32>,
    /// v0.15 phase 5: tiled denoise config. See `Request::tiled`.
    pub tiled: Option<crate::pipelines::tiled::TiledConfig>,
    /// Regional prompting (see `Request::regions`).
    pub regions: Vec<crate::pipelines::tiled::RegionSpec>,
    /// v0.16 phase 3: per-call conditioning override for the loaded
    /// SD3 ControlNets. Indexed parallel to
    /// `LoadRequest::controlnets`. An entry of `None` keeps the path
    /// from the load request (used when one scenario task has no CN
    /// conditioning to swap to). Empty Vec = preserve all paths.
    pub controlnet_conditioning: Vec<Option<PathBuf>>,
    /// v0.20: output container — see `Request::output_format`.
    pub output_format: crate::imaging::io::OutputFormat,
}

pub struct Pipeline {
    pub variant: Variant,
    #[allow(dead_code)]
    pub repo: String,
    device: Device,
    dtype: DType,
    clip_l: crate::pipelines::vendored_clip::ClipTextTransformer,
    clip_l_tok: Tokenizer,
    clip_l_cfg: crate::pipelines::vendored_clip::Config,
    clip_g: SdxlClipGTextTransformer,
    clip_g_tok: Tokenizer,
    t5_enc: t5::T5EncoderModel,
    t5_tok: Tokenizer,
    mmdit_model: mmdit::MMDiT,
    vae: sdvae::AutoEncoderKL,
    /// v0.16 phase 3: loaded SD3 ControlNets. Each entry carries a
    /// `Sd3ControlNet` + the per-instance runtime knobs (scale,
    /// conditioning path, step-gating window). Empty when no CN
    /// was passed in the LoadRequest.
    controlnets: Vec<crate::pipelines::sd3_controlnet::LoadedSd3ControlNet>,
    /// v1.10.0: loaded Textual Inversions, runtime-spliced in `encode_prompt`.
    /// Empty for every non-TI load (the default, verified path).
    tis: Vec<LoadedSd3Ti>,
}

/// A loaded SD3.5 Textual Inversion: the trigger word + its learned vector in
/// each of the three encoders + a strength scale, plus the per-tokenizer
/// added-token id used to locate the trigger in a tokenized prompt. Vectors are
/// in the pipeline dtype/device. See [`Pipeline::encode_prompt_ti`].
struct LoadedSd3Ti {
    /// Kept for diagnostics/logging at load time; lookup uses the per-encoder ids.
    #[allow(dead_code)]
    trigger: String,
    /// Added-token id in clip_l_tok / clip_g_tok / t5_tok respectively (each
    /// tokenizer has its own vocab, so the same trigger gets three ids).
    id_l: u32,
    id_g: u32,
    id_t5: u32,
    clip_l: Tensor, // (1, 768)
    clip_g: Tensor, // (1, 1280)
    t5: Tensor,     // (1, 4096)
    scale: f32,
}

impl Pipeline {
    pub async fn load(req: LoadRequest) -> Result<Self> {
        // BF16 matches Flux's reasoning: F16 range can't hold MMDiT's
        // intermediate activations cleanly; BF16 has F32's exponent
        // range with F16's storage. CUDA + Metal both support BF16 in
        // candle 0.8.
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::BF16
        };

        let dl = progress::spinner(&format!("Downloading weights for {}", req.repo));
        // Diffusers layout: each component lives under its own subdir.
        // Stability also ships `sd3.5_medium.safetensors` as a
        // single-file MMDiT-only artefact, but the diffusers subdirs
        // give us VAE + text encoders + tokenizers in one place.
        let mmdit_path = crate::hf::download::get_first_of(&[
            (&req.repo, "transformer/diffusion_pytorch_model.safetensors"),
            (&req.repo, "sd3.5_medium.safetensors"),
        ])
        .await
        .context("locating SD3 MMDiT weights")?;
        let vae_path =
            crate::hf::download::get_file(&req.repo, "vae/diffusion_pytorch_model.safetensors")
                .await
                .context("downloading SD3 VAE")?;
        let clip_l_w = crate::hf::download::get_first_of(&[
            (&req.repo, "text_encoder/model.fp16.safetensors"),
            (&req.repo, "text_encoder/model.safetensors"),
        ])
        .await
        .context("downloading CLIP-L weights")?;
        let clip_g_w = crate::hf::download::get_first_of(&[
            (&req.repo, "text_encoder_2/model.fp16.safetensors"),
            (&req.repo, "text_encoder_2/model.safetensors"),
        ])
        .await
        .context("downloading CLIP-G weights")?;
        let clip_l_tok_path = crate::hf::download::get_first_of(&[
            (&req.repo, "tokenizer/tokenizer.json"),
            ("openai/clip-vit-large-patch14", "tokenizer.json"),
        ])
        .await?;
        let clip_g_tok_path = crate::hf::download::get_first_of(&[
            (&req.repo, "tokenizer_2/tokenizer.json"),
            (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "tokenizer.json",
            ),
        ])
        .await?;
        // T5 ships sharded; try one common layout, fall back to the
        // single-file path some mirrors use.
        let (t5_shard1, t5_shard2) = {
            let shard1 = crate::hf::download::get_file(
                &req.repo,
                "text_encoder_3/model-00001-of-00002.safetensors",
            )
            .await
            .context("downloading T5-XXL shard 1")?;
            let shard2 = crate::hf::download::get_file(
                &req.repo,
                "text_encoder_3/model-00002-of-00002.safetensors",
            )
            .await
            .context("downloading T5-XXL shard 2")?;
            (shard1, shard2)
        };
        let t5_cfg_path =
            crate::hf::download::get_file(&req.repo, "text_encoder_3/config.json").await?;
        let t5_tok_path =
            crate::hf::download::get_file(&req.repo, "tokenizer_3/spiece.model").await.ok();
        let t5_tok_json = crate::hf::download::get_file(&req.repo, "tokenizer_3/tokenizer.json")
            .await
            .context("downloading T5 tokenizer")?;
        let _ = t5_tok_path; // candle's T5 tokenizer reads tokenizer.json directly
        dl.finish_with_message("✓ weights ready");

        let build = progress::spinner("Loading text encoders");
        // ---------- CLIP-L (no projection — just the hidden + EOT pool) ---
        // v0.32 phase 1: vendored CLIP-L. SDXL CLIP-L = SD3 CLIP-L
        // (77 tokens, 768d, 12 layers). Numerically identical to
        // candle's `sdclip::Config::sdxl()`.
        let clip_l_cfg = crate::pipelines::vendored_clip::Config::sdxl();
        let clip_l = crate::pipelines::vendored_clip::build_clip_transformer(
            &clip_l_cfg,
            &clip_l_w,
            &req.device,
            dtype,
        )?;
        let mut clip_l_tok =
            Tokenizer::from_file(&clip_l_tok_path).map_err(|e| anyhow!("CLIP-L tokenizer: {e}"))?;

        // ---------- CLIP-G (with text_projection for pooled) ----------
        // v0.30 phase 0: vendored CLIP Config for CLIP-G. Numerically
        // identical to candle's `sdclip::Config::sdxl2()`.
        let clip_g_cfg = crate::pipelines::vendored_clip::Config::sdxl2();
        let clip_g_vs = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&clip_g_w], dtype, &req.device)?
        };
        let clip_g = SdxlClipGTextTransformer::new(clip_g_vs, &clip_g_cfg, 1280)?;
        let mut clip_g_tok =
            Tokenizer::from_file(&clip_g_tok_path).map_err(|e| anyhow!("CLIP-G tokenizer: {e}"))?;

        // ---------- T5-XXL ----------
        let t5_cfg_str = std::fs::read_to_string(&t5_cfg_path)
            .with_context(|| format!("read T5 config {}", t5_cfg_path.display()))?;
        let t5_cfg: t5::Config =
            serde_json::from_str(&t5_cfg_str).context("parse T5 config")?;
        let t5_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&t5_shard1, &t5_shard2], dtype, &req.device)?
        };
        let t5_enc = t5::T5EncoderModel::load(t5_vb, &t5_cfg)?;
        let mut t5_tok =
            Tokenizer::from_file(&t5_tok_json).map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
        build.finish_with_message("✓ text encoders ready");

        // ---------- v1.10.0: Textual Inversion (runtime splice) ----------
        // Load each triple TI file, register its trigger as a single added
        // token in all three tokenizers, and stash the learned vectors;
        // `encode_prompt` splices them in when the trigger appears. Gated:
        // empty `embeddings` → no tokenizer mutation, the encode path stays
        // byte-identical to the verified pre-TI behaviour.
        let mut tis: Vec<LoadedSd3Ti> = Vec::new();
        for spec in &req.embeddings {
            let path = crate::pipelines::embedding::resolve(spec).await?;
            let tensors = candle_core::safetensors::load(&path, &req.device)
                .with_context(|| format!("loading SD3 TI {}", path.display()))?;
            let take = |k: &str, dim: usize| -> Result<Tensor> {
                let t = tensors.get(k).ok_or_else(|| {
                    anyhow!(
                        "SD3 TI {} missing `{k}` tensor — not an sd35 triple \
                         embedding (train with `embedding train --base sd35`)",
                        path.display()
                    )
                })?;
                let t = t.to_dtype(dtype)?.to_device(&req.device)?;
                let t = if t.rank() == 1 { t.unsqueeze(0)? } else { t };
                anyhow::ensure!(
                    t.dim(1)? == dim,
                    "SD3 TI `{k}` has dim {} but expected {dim}",
                    t.dim(1)?
                );
                Ok(t.narrow(0, 0, 1)?) // first vector → (1, dim)
            };
            let clip_l_v = take("clip_l", 768)?;
            let clip_g_v = take("clip_g", 1280)?;
            let t5_v = take("t5", 4096)?;
            let trigger = spec.trigger.clone().unwrap_or_else(|| {
                crate::pipelines::embedding::derive_trigger_from_path(&path)
            });
            let reg = |tok: &mut Tokenizer, trig: &str| -> u32 {
                tok.add_tokens(&[tokenizers::AddedToken::from(trig.to_string(), false)]);
                tok.token_to_id(trig).unwrap_or(0)
            };
            let id_l = reg(&mut clip_l_tok, &trigger);
            let id_g = reg(&mut clip_g_tok, &trigger);
            let id_t5 = reg(&mut t5_tok, &trigger);
            tracing::info!(
                target: "plakat",
                "SD3 TI loaded: trigger {:?} (ids L{id_l}/G{id_g}/T5{id_t5}, scale {})",
                trigger, spec.scale
            );
            tis.push(LoadedSd3Ti {
                trigger,
                id_l,
                id_g,
                id_t5,
                clip_l: clip_l_v,
                clip_g: clip_g_v,
                t5: t5_v,
                scale: spec.scale,
            });
        }

        // ---------- v0.15 phase 3: optional LoRA merge ----------
        //
        // When the caller passes any `LoadRequest::loras`, resolve each
        // and merge into a tempfile that replaces `mmdit_path` for the
        // VarBuilder. The tempfile lives in `std::env::temp_dir()`
        // (named uniquely by PID + nanos) and is kept around for the
        // lifetime of the loaded MMDiT — the mmap into VarBuilder
        // needs the bytes on disk. Best-effort cleanup is a deferred
        // concern; the OS sweeps temp_dir periodically.
        //
        // Sd35-Large + Sd35-LargeTurbo use the same hidden_size
        // (64 * 38 = 2432) since the LargeTurbo is just a distillation
        // of the Large transformer.
        let resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> = if req.loras.is_empty() {
            Vec::new()
        } else {
            let lr = progress::spinner(&format!("Resolving {} SD3 LoRA(s)", req.loras.len()));
            let mut v = Vec::with_capacity(req.loras.len());
            for spec in &req.loras {
                v.push(spec.resolve().await?);
            }
            lr.finish_with_message(format!("✓ resolved {} SD3 LoRA file(s)", v.len()));
            v
        };
        let merged_mmdit_path: Option<std::path::PathBuf> = if resolved_loras.is_empty() {
            None
        } else {
            let h = req.variant.mmdit_hidden_size();
            let lr = progress::spinner(&format!(
                "Merging {} SD3 LoRA(s) into MMDiT", resolved_loras.len()
            ));
            let out_path = std::env::temp_dir().join(format!(
                "plakat-sd3-lora-merged-{}-{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            let (n_mod, n_total) =
                crate::pipelines::sd3_lora::merge_sd3_loras_into_weights(
                    &mmdit_path,
                    &out_path,
                    &resolved_loras,
                    req.lora_scale,
                    h,
                    &req.device,
                )?;
            lr.finish_with_message(format!(
                "✓ SD3 LoRA merge → {n_mod}/{n_total} target groups applied"
            ));
            Some(out_path)
        };
        let effective_mmdit_path = merged_mmdit_path.as_ref().unwrap_or(&mmdit_path);

        // ---------- MMDiT + VAE ----------
        let load = progress::spinner("Loading MMDiT + VAE");
        let mmdit_vb = build_mmdit_vb(
            effective_mmdit_path.as_path(),
            dtype,
            &req.device,
            req.variant.mmdit_config().depth,
        )?;
        let mmdit_model = mmdit::MMDiT::new(&req.variant.mmdit_config(), false, mmdit_vb)?;

        // SD3 VAE: 4 down-blocks (128, 256, 512, 512), 2 layers each,
        // 16 latent channels, no quant/post-quant convs (diffusers
        // dropped these for the SD3 AE).
        let vae_cfg = sdvae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 16,
            norm_num_groups: 32,
            use_quant_conv: false,
            use_post_quant_conv: false,
        };
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&vae_path], dtype, &req.device)?
        };
        let vae = sdvae::AutoEncoderKL::new(vae_vb, 3, 3, vae_cfg)?;
        load.finish_with_message("✓ MMDiT + VAE loaded");

        // v0.16 phase 3: preload SD3 ControlNets, one per LoadRequest
        // entry. Each download is ~2-3 GB on cold cache.
        let mut controlnets = Vec::with_capacity(req.controlnets.len());
        for (i, spec) in req.controlnets.iter().enumerate() {
            let cn_spin = progress::spinner(&format!(
                "Loading SD3 ControlNet [{}/{}] {}",
                i + 1,
                req.controlnets.len(),
                spec.repo
            ));
            let net = crate::pipelines::sd3_controlnet::load_from_hf(
                &spec.repo, &spec.file, &spec.cfg, &req.device, dtype,
            )
            .await
            .with_context(|| {
                format!("loading SD3 ControlNet from {}/{}", spec.repo, spec.file)
            })?;
            cn_spin.finish_with_message(format!(
                "✓ SD3 ControlNet loaded ({} joint blocks)",
                net.n_residuals()
            ));
            controlnets.push(crate::pipelines::sd3_controlnet::LoadedSd3ControlNet {
                net,
                scale: spec.scale,
                conditioning_path: spec.conditioning.clone(),
                start: spec.start,
                end: spec.end,
            });
        }

        Ok(Self {
            variant: req.variant,
            repo: req.repo,
            device: req.device,
            dtype,
            clip_l,
            clip_l_tok,
            clip_l_cfg,
            clip_g,
            clip_g_tok,
            t5_enc,
            t5_tok,
            mmdit_model,
            vae,
            controlnets,
            tis,
        })
    }

    /// v0.16 phase 3: are any ControlNets loaded? Cheap check used
    /// by the dispatcher to skip CN-specific code paths when the
    /// scenario / call doesn't use CN.
    pub fn has_controlnets(&self) -> bool {
        !self.controlnets.is_empty()
    }

    /// v0.16 phase 3: number of loaded ControlNet slots. The
    /// dispatcher uses this to validate per-task conditioning lists
    /// against the load-time slot count.
    pub fn n_controlnets(&self) -> usize {
        self.controlnets.len()
    }

    /// v0.16 phase 3: swap the conditioning path for a single CN
    /// slot between `generate` calls. `None` clears the path so the
    /// CN contributes no residuals on the next call (used when a
    /// scenario task has fewer CNs than the scenario's max slot
    /// count). Same shape as `flux::Pipeline::set_controlnet_conditioning`.
    pub fn set_controlnet_conditioning(
        &mut self,
        idx: usize,
        path: Option<PathBuf>,
    ) -> Result<()> {
        let n = self.controlnets.len();
        let cn = self.controlnets.get_mut(idx).ok_or_else(|| {
            anyhow!(
                "sd3::Pipeline::set_controlnet_conditioning: slot {idx} out of \
                 range (have {n} loaded CN(s))"
            )
        })?;
        cn.conditioning_path = path;
        Ok(())
    }

    /// v0.16 phase 3: per-call CN runtime knobs (scale + step-gating
    /// window). Lets the scenario dispatcher set per-task strengths
    /// without re-loading the model.
    pub fn set_controlnet_call_params(
        &mut self,
        idx: usize,
        scale: f32,
        start: f32,
        end: f32,
    ) -> Result<()> {
        let n = self.controlnets.len();
        let cn = self.controlnets.get_mut(idx).ok_or_else(|| {
            anyhow!(
                "sd3::Pipeline::set_controlnet_call_params: slot {idx} out of \
                 range (have {n} loaded CN(s))"
            )
        })?;
        cn.scale = scale;
        cn.start = start;
        cn.end = end;
        Ok(())
    }

    /// v0.16 phase 2: replace the runtime LoRA stack on the MMDiT
    /// backbone. Mirrors `flux::FluxBackbone::apply_loras`. The
    /// scenario dispatcher uses this between tasks to apply each
    /// task's per-task LoRA additions on top of the scenario-merged
    /// baseline; followed by `clear_all_loras()` at end-of-task to
    /// avoid bleed.
    pub fn apply_loras(
        &self,
        specs: std::collections::HashMap<
            String,
            Vec<crate::pipelines::lora_linear::LoraSpec>,
        >,
    ) -> Result<usize> {
        let applied = self
            .mmdit_model
            .apply_loras(specs, self.dtype, &self.device)
            .map_err(|e| anyhow!("SD3 apply_loras: {e}"))?;
        Ok(applied)
    }

    /// v0.16 phase 2: clear every runtime LoRA on the MMDiT.
    pub fn clear_all_loras(&self) -> Result<()> {
        self.mmdit_model
            .clear_all_loras()
            .map_err(|e| anyhow!("SD3 clear_all_loras: {e}"))?;
        Ok(())
    }

    /// v0.16 phase 2: runtime dtype the MMDiT uses (BF16 on GPU,
    /// F32 on CPU). Exposed for the scenario dispatcher's LoRA spec
    /// builder.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// v0.16 phase 2: device the pipeline lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// v0.16 phase 2: variant accessor. Used by the dispatcher to
    /// look up the MMDiT hidden size for LoRA-B row-slice padding.
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// Generate `req.count` images. Reuses the loaded weights across
    /// images; `&mut self` because T5 maintains an internal KV cache.
    pub fn generate(&mut self, req: &GenRequest) -> Result<()> {
        self.generate_hooked(req, None)
    }

    /// As [`generate`](Self::generate) with an optional per-step
    /// [`StepHook`](crate::pipelines::step_hook::StepHook) (RFC TUI-1 §0-R0-3) for
    /// TUI progress + cancellation. `None` is the unchanged CLI path. (Live preview
    /// is SD-only for now — SD3's 16-channel latent needs its own projection;
    /// progress + cancel apply.)
    pub fn generate_hooked(
        &mut self,
        req: &GenRequest,
        mut hook: Option<&mut dyn crate::pipelines::step_hook::StepHook>,
    ) -> Result<()> {
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps());
        let guidance = req.guidance.unwrap_or_else(|| self.variant.default_guidance());
        // MMDiT processes 2×2 patches of a 16-ch latent. With VAE
        // downsample 8, image dims must be multiples of 16 so the
        // latent (H/8 × W/8) is even.
        let w = (req.width as usize / 16) * 16;
        let h = (req.height as usize / 16) * 16;
        if w == 0 || h == 0 {
            bail!("SD3 requires width and height divisible by 16, both ≥ 16");
        }
        // v0.16 phase 10: --tiled now composes with --init-image
        // (img2img + inpaint). The math is straightforward: the
        // start-latent lerp builds a full-canvas `x`, tiled
        // velocity prediction blends per-tile Hann-weighted
        // contributions into a full-canvas `pred`, and the
        // Euler step + the optional RePaint mask blend operate
        // unchanged on the full canvas. ControlNet + tiled is
        // still bail-loud inside `predict_velocity_tiled` (the
        // CN's `pos_embed_input` works on full-canvas latents,
        // not tile slices) — combinations with --init-image are
        // unaffected.
        //
        // Inpaint + tiled note: tile seams can become visible near
        // sharp mask boundaries because the Hann blend smooths
        // across tile edges without knowing about the mask. Use
        // a feathered mask (`--mask-feather`) to smooth the
        // transition.
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        // ---------- encode prompt + negative ----------
        let enc = progress::spinner("Encoding prompt");
        let (pos_y, pos_ctx) = self.encode_prompt(&req.prompt)?;
        let (neg_y, neg_ctx) = self.encode_prompt(&req.negative)?;
        // Batch them into [neg, pos] so the MMDiT forward returns
        // (B=2, 16, H/8, W/8) and we can split for CFG.
        let cfg_y = Tensor::cat(&[&neg_y, &pos_y], 0)?;
        let cfg_ctx = Tensor::cat(&[&neg_ctx, &pos_ctx], 0)?;
        enc.finish_with_message("✓ prompt encoded");

        let lat_h = h / 8;
        let lat_w = w / 8;

        // Regional prompting: encode each region prompt (CFG-batched with the
        // shared negative) + build its bbox mask; the base covers the rest and
        // supplies global coherence. Empty `regions` = the normal single-prompt
        // path (no extra work).
        let mut region_data: Vec<(Tensor, Tensor, Tensor)> = Vec::new();
        for r in &req.regions {
            let (py, pctx) = self.encode_prompt(&r.prompt)?;
            region_data.push((
                Tensor::cat(&[&neg_y, &py], 0)?,
                Tensor::cat(&[&neg_ctx, &pctx], 0)?,
                crate::pipelines::tiled::region_mask(r.bbox, lat_h, lat_w, &self.device, self.dtype)?,
            ));
        }
        let region_base_mask = if region_data.is_empty() {
            None
        } else {
            let mut covered = Tensor::zeros((1usize, 1, lat_h, lat_w), self.dtype, &self.device)?;
            for (_, _, m) in &region_data {
                covered = (covered + m)?;
            }
            let ones = Tensor::ones((1usize, 1, lat_h, lat_w), self.dtype, &self.device)?;
            Some((&ones - &covered)?.clamp(0f32, 1f32)?)
        };
        if region_base_mask.is_some() && req.tiled.is_some() {
            bail!("regions don't compose with tiled hi-res on SD3.5 — use one or the other");
        }

        let time_shift = self.variant.default_time_shift();

        // ---------- v0.15 phase 2: img2img / inpaint prep ----------
        // VAE-encode the init image once; reuse across the count loop.
        // The mask is image-space (HxW); we downsample to latent
        // resolution (H/8 x W/8) inside the loop. RePaint convention:
        // white (1.0) = inpaint (denoise this region), black (0.0) =
        // preserve (snap back to the init's flow trajectory).
        let img2img_init: Option<(Tensor, f32, Option<Tensor>)> =
            if let Some(init_path) = req.init_image.as_ref() {
                let has_mask = req.mask.is_some();
                let strength = req
                    .strength
                    .unwrap_or(if has_mask { 1.0 } else { 0.6 })
                    .clamp(0.0, 1.0);
                if !strength.is_finite() {
                    bail!(
                        "SD3 img2img strength must be finite in [0, 1], got {strength}"
                    );
                }
                let spin = progress::spinner("Encoding SD3 img2img init image");
                let init_pixels = crate::imaging::preprocess::sd_image_tensor(
                    init_path,
                    w as u32,
                    h as u32,
                    &self.device,
                    self.dtype,
                )
                .with_context(|| {
                    format!("loading SD3 init image {}", init_path.display())
                })?;
                // SD3 latent normalisation is the inverse of the decode
                // path's `(x / SCALE) + SHIFT`: `(z - SHIFT) * SCALE`.
                // `encode` returns a `DiagonalGaussianDistribution`;
                // `.sample()` draws one latent sample (matches every
                // other SD-family encode site in the codebase).
                let init_dist = self.vae.encode(&init_pixels)?;
                let init_z = init_dist.sample()?;
                let init_norm = ((init_z - VAE_SHIFT)? * VAE_SCALE)?;

                let mask_lat = if let Some(mask_path) = req.mask.as_ref() {
                    let mut m = crate::imaging::mask::Mask::load(
                        mask_path, w as u32, h as u32,
                    )?;
                    if req.mask_invert {
                        m.invert();
                    }
                    if req.mask_feather > 0 {
                        m.feather(req.mask_feather);
                    }
                    Some(m.to_latent_tensor(&self.device, self.dtype)?)
                } else {
                    None
                };
                spin.finish_with_message(format!(
                    "✓ SD3 init encoded (strength {strength:.2}{})",
                    if has_mask { ", masked" } else { "" }
                ));
                Some((init_norm, strength, mask_lat))
            } else {
                None
            };

        // v0.16 phase 3d: pre-encode the ControlNet conditioning
        // image(s) into VAE latents (one per loaded CN slot). The
        // encode runs once per `generate` call — every step within
        // the count loop reuses the cached latents.
        //
        // Per-call `req.controlnet_conditioning` overrides the
        // load-time path when set (used by the scenario dispatcher
        // to swap conditioning between tasks). An entry of `None`
        // keeps the load-time path; an empty Vec preserves all
        // load-time paths.
        let cn_conditionings: Vec<Option<Tensor>> = if self.controlnets.is_empty() {
            Vec::new()
        } else {
            let mut v = Vec::with_capacity(self.controlnets.len());
            for (i, cn) in self.controlnets.iter().enumerate() {
                let path: Option<&std::path::Path> = req
                    .controlnet_conditioning
                    .get(i)
                    .and_then(|p| p.as_deref())
                    .or(cn.conditioning_path.as_deref());
                match path {
                    Some(p) => {
                        let spin = progress::spinner(&format!(
                            "Encoding SD3 CN[{}/{}] conditioning",
                            i + 1,
                            self.controlnets.len()
                        ));
                        let enc = self.encode_cn_conditioning(p, h, w)?;
                        spin.finish_with_message(format!(
                            "✓ SD3 CN[{}/{}] conditioning encoded",
                            i + 1,
                            self.controlnets.len()
                        ));
                        v.push(Some(enc));
                    }
                    None => {
                        tracing::warn!(
                            target: "plakat",
                            "SD3 CN[{}/{}] loaded but no conditioning image — \
                             this CN won't contribute residuals.",
                            i + 1,
                            self.controlnets.len()
                        );
                        v.push(None);
                    }
                }
            }
            v
        };

        // RFC TUI-1 §0-R0-3: set when a StepHook requests cancellation.
        let mut cancelled = false;
        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random);
            // v0.34 phase 1: device-aware seed prep.
            let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
            if let Err(e) = self.device.set_seed(prepared) {
                tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
            }

            // Fresh per-image noise. Cached for inpaint mode below so
            // the unmasked-region resampling uses the same noise the
            // start latent was built from.
            let eps = Tensor::randn(0f32, 1.0_f32, (1, 16, lat_h, lat_w), &self.device)?
                .to_dtype(self.dtype)?;

            // Initial latent + schedule.
            //
            // **Pure t2i**: x = eps (Gaussian noise), timesteps full
            //   [1.0 → 0.0] linear-with-shift.
            //
            // **img2img / inpaint**: x = lerp(init, eps, strength) using
            //   the rectified-flow interpolation `x_t = (1-t)*init + t*eps`,
            //   then truncate the schedule to entries below strength
            //   and prepend strength itself so the first step's
            //   `t_curr` matches the noise level of the start latent.
            //   Mirrors diffusers' FlowMatchEulerDiscreteScheduler
            //   `get_timesteps` + Flux phase-3 img2img init.
            let (mut x, timesteps) = match img2img_init.as_ref() {
                Some((init_norm, strength, _)) => {
                    let s = *strength as f64;
                    let start = ((init_norm * (1.0 - s))? + (&eps * s)?)?;
                    let ts = build_img2img_timesteps(steps, time_shift, Some(s));
                    (start, ts)
                }
                None => {
                    let ts = build_img2img_timesteps(steps, time_shift, None);
                    (eps.clone(), ts)
                }
            };

            let bar = progress::step_bar(
                (timesteps.len().saturating_sub(1)) as u64,
                &format!("img {}/{}", idx + 1, req.count),
            );
            // v0.16 phase 10: mode tags now compose. `tiled inpaint`
            // / `tiled img2img` surface the new combinations
            // explicitly in the progress bar.
            let mode_tag = sd3_mode_tag(
                req.tiled.is_some(),
                req.init_image.is_some(),
                req.mask.is_some(),
            );
            bar.set_message(format!(
                "{mode_tag} flow-match, {} steps, seed={seed}",
                timesteps.len().saturating_sub(1)
            ));

            for (step_i, window) in timesteps.windows(2).enumerate() {
                let (t_curr, t_prev) = match window {
                    [a, b] => (*a, *b),
                    _ => continue,
                };
                // v0.15 phase 5: dispatch to tiled or single-pass.
                // Both return the post-CFG velocity prediction at
                // full canvas resolution; the Euler step is identical.
                // v0.16 phase 3d: thread the pre-encoded CN
                // conditioning latents + the schedule progress through
                // so the CN forwards can run + step-gating works.
                let num_steps = timesteps.windows(2).count().max(1);
                let progress = step_i as f32 / num_steps as f32;
                let pred = if let Some(base_mask) = region_base_mask.as_ref() {
                    // Regional: blend the base + per-region velocities by masks.
                    self.predict_velocity_regional(
                        &x,
                        t_curr,
                        &cfg_y,
                        &cfg_ctx,
                        base_mask,
                        &region_data,
                        guidance,
                        &cn_conditionings,
                        progress,
                    )?
                } else {
                    match req.tiled.as_ref() {
                        None => self.predict_velocity_full(
                            &x,
                            t_curr,
                            &cfg_y,
                            &cfg_ctx,
                            guidance,
                            &cn_conditionings,
                            progress,
                        )?,
                        Some(cfg) => self.predict_velocity_tiled(
                            &x,
                            t_curr,
                            &cfg_y,
                            &cfg_ctx,
                            guidance,
                            cfg,
                            &cn_conditionings,
                            progress,
                        )?,
                    }
                };
                x = (x + pred * (t_prev - t_curr))?;

                // RePaint-style inpaint blend: after the denoise step
                // brought every pixel to `t_prev`, snap the *unmasked*
                // region back onto the init's flow trajectory at the
                // same noise level. This keeps unmasked content
                // visually anchored to the init while letting the
                // masked region freely re-denoise. Done in latent
                // space so per-step VAE roundtrips aren't needed.
                if let Some((init_norm, _strength, Some(mask_lat))) =
                    img2img_init.as_ref()
                {
                    let init_at_tprev = ((init_norm * (1.0 - t_prev))?
                        + (&eps * t_prev)?)?;
                    // mask*x + (1-mask)*init_at_tprev — broadcasting
                    // mask (1,1,h,w) over the 16 latent channels.
                    let one_minus = (mask_lat.affine(-1.0, 1.0))?;
                    let kept = init_at_tprev.broadcast_mul(&one_minus)?;
                    let edited = x.broadcast_mul(mask_lat)?;
                    x = (edited + kept)?;
                }

                bar.set_position(step_i as u64);
                // RFC TUI-1 §0-R0-3: per-step hook (progress + cancel; no-op on None).
                if crate::pipelines::step_hook::step(
                    &mut hook,
                    step_i,
                    timesteps.len().saturating_sub(1),
                ) == crate::pipelines::step_hook::StepControl::Cancel
                {
                    cancelled = true;
                    break;
                }
            }
            bar.set_position(timesteps.len().saturating_sub(1) as u64);
            bar.finish_with_message(format!("✓ {mode_tag} done"));

            let pre_decode = ((&x / VAE_SCALE)? + VAE_SHIFT)?;
            let decoded = self.vae.decode(&pre_decode)?;
            let img_norm = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 0.5)?;
            let img_u8 = (img_norm * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = img_u8.dims3()?;
            let buf = img_u8.flatten_all()?.to_vec1::<u8>()?;

            let out_path = req
                .out_dir
                .join(format!(
                    "plakat-sd3-{mode_tag}-{seed}.{}",
                    req.output_format.extension()
                ));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
            // RFC TUI-1 §0-R0-3: a cancelled step saved this partial; stop.
            if cancelled {
                break;
            }
        }
        Ok(())
    }

    /// v0.15 phase 5: single-pass post-CFG velocity prediction.
    ///
    /// Builds the `[neg, pos]` double-batch, runs MMDiT once, splits
    /// the two predictions, and blends with `guidance`. Same math as
    /// the inline path in `generate` — factored out so the tiled
    /// dispatch can reuse it inside the per-tile loop.
    ///
    /// v0.16 phase 3d: `cn_conditionings` carries the pre-encoded
    /// (VAE-encoded + normalised) conditioning latents for each
    /// loaded SD3 ControlNet, indexed parallel to `self.controlnets`.
    /// `progress` is the fraction of the denoise schedule in
    /// `[0, 1)`; each CN's `active_at` window gates whether its
    /// residuals contribute. When no CN is active for this step the
    /// forward path is identical to the pre-3d call.
    /// Regional prompting velocity blend: the base velocity plus each region's
    /// velocity (each a full `predict_velocity_full` MMDiT pass), blended by the
    /// bbox masks — the base fills where no region covers. ~(1 + N) passes/step.
    #[allow(clippy::too_many_arguments)]
    fn predict_velocity_regional(
        &self,
        x: &Tensor,
        t_curr: f64,
        base_cfg_y: &Tensor,
        base_cfg_ctx: &Tensor,
        base_mask: &Tensor,
        regions: &[(Tensor, Tensor, Tensor)],
        guidance: f64,
        cn_conditionings: &[Option<Tensor>],
        progress: f32,
    ) -> Result<Tensor> {
        let base_v = self.predict_velocity_full(
            x,
            t_curr,
            base_cfg_y,
            base_cfg_ctx,
            guidance,
            cn_conditionings,
            progress,
        )?;
        let mut acc = base_v.broadcast_mul(base_mask)?;
        let mut weights = base_mask.clone();
        for (cfg_y, cfg_ctx, mask) in regions {
            let v = self.predict_velocity_full(
                x,
                t_curr,
                cfg_y,
                cfg_ctx,
                guidance,
                cn_conditionings,
                progress,
            )?;
            acc = (acc + v.broadcast_mul(mask)?)?;
            weights = (weights + mask)?;
            // Bound peak memory: finish each region's MMDiT pass before the next.
            acc.device().synchronize()?;
        }
        Ok(acc.broadcast_div(&weights)?)
    }

    fn predict_velocity_full(
        &self,
        x: &Tensor,
        t_curr: f64,
        cfg_y: &Tensor,
        cfg_ctx: &Tensor,
        guidance: f64,
        cn_conditionings: &[Option<Tensor>],
        progress: f32,
    ) -> Result<Tensor> {
        let x_doubled = Tensor::cat(&[x, x], 0)?;
        // The flow-match schedule lives in [0, 1], but SD3's MMDiT time
        // embedder was trained on diffusers' timestep convention
        // (`sigma * num_train_timesteps`, i.e. [0, 1000]). Passing the raw
        // [0, 1] sigma gives a wildly wrong sinusoidal embedding → wrong
        // velocity → garbage. (The single-forward reference check can't
        // catch this: it feeds the same `t` to both sides.)
        let t_vec = Tensor::full((t_curr * 1000.0) as f32, 2, &self.device)?;

        // v0.16 phase 3d: build the SD3 CN residual sum for this
        // step. Each active CN forwards once on the doubled batch
        // (the CN's pos_embed_input broadcast-adds the (1,16,h,w)
        // conditioning across both CFG branches). Residuals are
        // scaled per-CN and summed across slots.
        let mut summed_residuals: Option<Vec<Tensor>> = None;
        for (cn, cond_opt) in self.controlnets.iter().zip(cn_conditionings.iter()) {
            if !cn.active_at(progress) {
                continue;
            }
            let cond = match cond_opt.as_ref() {
                Some(c) => c,
                None => continue,
            };
            let res = cn
                .net
                .forward(&x_doubled, cond, cfg_ctx, cfg_y, &t_vec)
                .map_err(|e| anyhow!("SD3 CN forward: {e}"))?;
            let scaled: Vec<Tensor> = res
                .into_iter()
                .map(|t| t * cn.scale as f64)
                .collect::<core::result::Result<_, _>>()
                .map_err(|e| anyhow!("SD3 CN residual scale: {e}"))?;
            summed_residuals = Some(merge_residuals(summed_residuals, scaled)?);
        }
        let residuals = summed_residuals.as_deref();

        let pred_doubled = self
            .mmdit_model
            .forward_with_residuals(&x_doubled, &t_vec, cfg_y, cfg_ctx, None, residuals)?;
        let pred_neg = pred_doubled.i(0..1)?;
        let pred_pos = pred_doubled.i(1..2)?;
        Ok((&pred_neg + ((pred_pos - &pred_neg)? * guidance)?)?)
    }

    /// v0.15 phase 5: tiled MultiDiffusion-style post-CFG velocity
    /// prediction.
    ///
    /// Splits the latent into overlapping `tile_size`-pixel windows
    /// (`stride`-spaced), runs MMDiT per tile, and blends the per-tile
    /// velocity predictions back into a full-canvas tensor with a 2D
    /// Hann window. Output shape matches `x` — the Euler step the
    /// caller applies is identical to the single-pass path.
    ///
    /// Constraints:
    /// * `tile_size` and `stride` must be multiples of 16 (the
    ///   product of VAE downsample 8 × MMDiT patch_size 2).
    /// * Patched tile dim = `tile_latent / 2` must be `<=
    ///   pos_embed_max_size` (384 on SD3.5-Medium, 192 on
    ///   SD3 / SD3.5-Large). For the default 1024-px tile that's 64
    ///   patches per axis — well within either cap.
    ///
    /// When the canvas fits inside one tile, falls back to
    /// `predict_velocity_full` (cheaper, identical output).
    fn predict_velocity_tiled(
        &self,
        x: &Tensor,
        t_curr: f64,
        cfg_y: &Tensor,
        cfg_ctx: &Tensor,
        guidance: f64,
        tcfg: &crate::pipelines::tiled::TiledConfig,
        cn_conditionings: &[Option<Tensor>],
        progress: f32,
    ) -> Result<Tensor> {
        // v0.16 phase 3d: SD3 CN + tiled composition isn't wired
        // (per-tile conditioning slicing would mirror Flux's tiled-CN
        // path but the SD3 CN's `pos_embed_input` operates on the
        // full-canvas latent, not a tile). Bail cleanly so users see
        // a clear message rather than silently-wrong outputs.
        if cn_conditionings.iter().any(|c| c.is_some()) {
            bail!(
                "SD3 tiled denoise doesn't compose with ControlNet yet — drop \
                 `--tiled` or `--control-spec`."
            );
        }
        let _ = progress;
        let (_b, c, lat_h, lat_w) = x.dims4()?;
        // VAE downsample 8 — same factor the rest of the SD3 pipeline
        // uses. Pixel-to-latent conversion for the tile + stride.
        const VAE_FACTOR: usize = 8;
        if tcfg.tile_size as usize % 16 != 0 || tcfg.stride as usize % 16 != 0 {
            bail!(
                "SD3 tiled denoise requires --tile-size and --tile-stride \
                 divisible by 16 (got {} / {})",
                tcfg.tile_size, tcfg.stride
            );
        }
        let tile_lat = (tcfg.tile_size as usize) / VAE_FACTOR;
        let stride_lat = (tcfg.stride as usize) / VAE_FACTOR;
        // Patched-tile dim against MMDiT's pos_embed cap. patch_size=2
        // is the SD3 constant — kept inline rather than reaching into
        // the variant config so the constraint is explicit at the
        // call site.
        let max_patched =
            self.variant.mmdit_config().pos_embed_max_size;
        if tile_lat / 2 > max_patched {
            bail!(
                "SD3 tile_size {}px → patched {} exceeds variant's pos_embed_max_size {} \
                 (drop --tile-size or pick a larger SD3 variant)",
                tcfg.tile_size,
                tile_lat / 2,
                max_patched
            );
        }
        // Single-tile fast path: latent fits within the tile. Skips
        // the Hann blend overhead.
        if lat_h <= tile_lat && lat_w <= tile_lat {
            return self.predict_velocity_full(
                x, t_curr, cfg_y, cfg_ctx, guidance, &[], 0.0,
            );
        }
        let positions = crate::pipelines::tiled::tile_positions(
            lat_h, lat_w, tile_lat, stride_lat,
        );
        let win = crate::pipelines::tiled::hann_window_2d(
            tile_lat,
            &self.device,
            self.dtype,
        )?;
        // Accumulators: weighted velocity sum + scalar weight sum
        // (broadcast over channels at the final divide).
        let mut acc_pred = Tensor::zeros(
            (1, c, lat_h, lat_w),
            self.dtype,
            &self.device,
        )?;
        let mut acc_weight = Tensor::zeros(
            (1, 1, lat_h, lat_w),
            self.dtype,
            &self.device,
        )?;
        for pos in positions.iter() {
            // narrow extracts a tile; both axes use the same size since
            // tile_positions always emits square tiles.
            let x_tile = x.narrow(2, pos.y, pos.size)?.narrow(3, pos.x, pos.size)?;
            let pred_tile = self.predict_velocity_full(
                &x_tile, t_curr, cfg_y, cfg_ctx, guidance, &[], 0.0,
            )?;
            // Weighted contribution: pred_tile * hann broadcast over
            // (B, C, tile, tile).
            let weighted = pred_tile.broadcast_mul(&win)?;
            // Slice the accumulator at the tile position, add, write
            // back. candle 0.8 has no in-place slice update so we
            // narrow → add → reassemble via cat.
            //
            // Approach: build a `(1, c, lat_h, lat_w)` "patch" tensor
            // that's zero everywhere except inside the tile rect; the
            // tile rect holds `weighted`. Adding two equal-shape
            // tensors is the simplest path on candle.
            let patch = pad_tile_to_canvas(
                &weighted, pos.y, pos.x, lat_h, lat_w, self.dtype, &self.device,
            )?;
            acc_pred = (acc_pred + patch)?;
            let w_patch = pad_tile_to_canvas(
                &win, pos.y, pos.x, lat_h, lat_w, self.dtype, &self.device,
            )?;
            acc_weight = (acc_weight + w_patch)?;
        }
        // Normalise by the weight sum. The Hann window has a small
        // positive epsilon at its edges (see tiled::hann_window_2d) so
        // every covered pixel has weight > 0; no NaN guards needed.
        Ok(acc_pred.broadcast_div(&acc_weight)?)
    }

    /// Encode a single prompt into the `(y, context)` pair the MMDiT
    /// forward consumes.
    ///
    /// * `y` — `(1, 2048)` pooled embedding =
    ///   `[CLIP-G_pooled (1280) || CLIP-L_pooled (768)]`.
    /// * `context` — `(1, 77 + t5_seq, 4096)` text hidden states =
    ///   `[ pad([CLIP-L_hidden || CLIP-G_hidden], 2048→4096), T5_hidden ]`.
    /// v0.16 phase 3d: VAE-encode a ControlNet conditioning image
    /// path into the `(1, 16, h/8, w/8)` latent the SD3 CN's
    /// `pos_embed_input.proj` consumes. Same normalisation as the
    /// main pipeline's `init_norm` path:
    ///
    /// ```text
    ///     z_norm = (vae.encode(x).sample() - VAE_SHIFT) * VAE_SCALE
    /// ```
    ///
    /// so the CN sees a conditioning latent in the same numerical
    /// range as the noise it's added to.
    fn encode_cn_conditioning(
        &self,
        path: &std::path::Path,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let pixels = crate::imaging::preprocess::sd_image_tensor(
            path,
            w as u32,
            h as u32,
            &self.device,
            self.dtype,
        )
        .with_context(|| format!("loading SD3 CN conditioning {}", path.display()))?;
        let dist = self.vae.encode(&pixels)?;
        let z = dist.sample()?;
        let z_norm = ((z - VAE_SHIFT)? * VAE_SCALE)?;
        Ok(z_norm)
    }

    /// v0.26 phase 6: public wrapper for the prompt encoder so
    /// `plakat animate` can pre-encode endpoint prompts once and
    /// lerp between them per frame. Returns
    /// `(pooled_y, joint_context)` — same shape as the internal
    /// `encode_prompt`.
    pub fn encode_for_animate(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        self.encode_prompt(prompt)
    }

    /// v0.26 phase 6: single-frame SD3 / SD3.5 inference with
    /// pre-lerped embeddings. Mirrors `flux::Pipeline::animate_frame`.
    ///
    /// Inputs:
    /// - `pos_y` / `pos_ctx`: lerped positive embeddings for this
    ///   frame (from `encode_for_animate(--from)` lerped with
    ///   `encode_for_animate(--to)` at the per-frame `t`).
    /// - `neg_y` / `neg_ctx`: encoded negative prompt (constant
    ///   across frames since `--negative` doesn't lerp).
    /// - `width` / `height`: pixel dims, must be divisible by 16.
    /// - `steps`: per-frame denoise steps (animation typically uses
    ///   fewer than single-image generation).
    /// - `guidance`: CFG scale.
    /// - `seed`: shared seed for every frame — locks the initial
    ///   noise constant so only the prompt-driven trajectory
    ///   varies. Matches v0.20 Flux animate convention.
    ///
    /// Returns `(rgb_bytes, width, height)`. No file I/O — caller
    /// writes the frame.
    ///
    /// Scope cap:
    /// - No ControlNets (animate is text-only morph)
    /// - No tiled denoise (per-frame hi-res is impractical)
    /// - No img2img / mask (per-frame conditioning images are out
    ///   of scope for the morph contract — different design)
    pub fn animate_frame(
        &self,
        pos_y: &Tensor,
        pos_ctx: &Tensor,
        neg_y: &Tensor,
        neg_ctx: &Tensor,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f64,
        seed: u64,
    ) -> Result<(Vec<u8>, u32, u32)> {
        let w = (width as usize / 16) * 16;
        let h = (height as usize / 16) * 16;
        if w == 0 || h == 0 {
            bail!("SD3 animate requires width and height divisible by 16, both ≥ 16");
        }
        let lat_h = h / 8;
        let lat_w = w / 8;

        // [neg, pos] batching for CFG.
        let cfg_y = Tensor::cat(&[neg_y, pos_y], 0)?;
        let cfg_ctx = Tensor::cat(&[neg_ctx, pos_ctx], 0)?;

        // Init noise at the shared seed. v0.34 phase 1: device-aware
        // prep (Metal-only hash for high seeds; identity below 2^32).
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
        if let Err(e) = self.device.set_seed(prepared) {
            tracing::debug!(
                target: "plakat",
                "set_seed not supported ({e}); using global RNG"
            );
        }
        // Verify Tier 2 (env-gated): deterministic LCG init (candle CPU RNG isn't seed-repro).
        let mut x = if std::env::var("PLAKAT_VERIFY_DET_INIT").is_ok() {
            crate::verify::deterministic_latent(16, lat_h, lat_w, &self.device, self.dtype)?
        } else {
            Tensor::randn(0.0f32, 1.0f32, (1, 16, lat_h, lat_w), &self.device)?.to_dtype(self.dtype)?
        };

        // Rectified-flow schedule.
        let time_shift = self.variant.default_time_shift();
        let timesteps = build_img2img_timesteps(steps, time_shift, None);

        // No ControlNets in animate (scope cap).
        let no_cn: Vec<Option<Tensor>> = Vec::new();

        let num_steps = timesteps.windows(2).count().max(1);
        for (step_i, window) in timesteps.windows(2).enumerate() {
            let (t_curr, t_prev) = match window {
                [a, b] => (*a, *b),
                _ => continue,
            };
            let progress = step_i as f32 / num_steps as f32;
            let pred = self.predict_velocity_full(
                &x, t_curr, &cfg_y, &cfg_ctx, guidance, &no_cn, progress,
            )?;
            x = (x + pred * (t_prev - t_curr))?;
        }

        // VAE decode + image-buffer extraction.
        let pre_decode = ((&x / VAE_SCALE)? + VAE_SHIFT)?;
        let decoded = self.vae.decode(&pre_decode)?;
        let img_norm = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 0.5)?;
        let img_u8 = (img_norm * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = img_u8.dims3()?;
        let buf = img_u8.flatten_all()?.to_vec1::<u8>()?;
        Ok((buf, ow as u32, oh as u32))
    }

    /// Capture named intermediate tensors for `plakat verify` Tier 1 (RFC_VERIFY). Additive —
    /// reuses the real `encode_prompt`, so the captured tensor is exactly what generation uses.
    ///
    /// - `pooled_y` — the pooled conditioning `y` fed to the MMDiT, a concat of the two CLIP
    ///   pooled vectors. **The order was the killer SD3 bug** — this capture, compared to the
    ///   diffusers golden, proves plakat's order matches.
    pub fn capture_intermediates(
        &mut self,
        prompt: &str,
        wanted: &std::collections::HashSet<String>,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        let mut out = std::collections::HashMap::new();
        if wanted.contains("pooled_y") {
            let (pooled_y, _joint_context) = self.encode_prompt(prompt)?;
            out.insert("pooled_y".to_string(), pooled_y);
        }
        // T5 caption embedding WITH the padding attention mask (the v2.1 fix). Plain (non-
        // weighted) path — the fixture prompt has no attention syntax — matching what
        // `encode_prompt` runs. Corresponds to diffusers `text_encoder_3(ids, attention_mask)`.
        if wanted.contains("t5.hidden") {
            let t5_seq = self.variant.t5_seq_len();
            let mut ids = self
                .t5_tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("T5 encode: {e}"))?
                .get_ids()
                .to_vec();
            ids.truncate(t5_seq);
            ids.resize(t5_seq, 0);
            let ids_t = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let mask = ids_t.ne(0u32)?.to_dtype(candle_core::DType::F32)?;
            let hidden = self.t5_enc.forward_with_mask(&ids_t, &mask)?.to_dtype(candle_core::DType::F32)?;
            out.insert("t5.hidden".to_string(), hidden);
        }
        // MMDiT joint-block-0 tap: run the embed prologue + joint_blocks[0] on a shared
        // deterministic latent (16-ch) + FIXED timestep + DETERMINISTIC y/context (LCG). The
        // y/context are synthetic (not CLIP/T5) so this isolates the MMDiT joint-block math;
        // the dumper feeds byte-identical inputs via `fixtures.deterministic_tensor`.
        if wanted.contains("mmdit.block0") {
            let cfg = self.variant.mmdit_config();
            // Fixture 512 → 16-ch latent (1,16,64,64). Context seq is arbitrary but must match
            // the dumper — pin it here as the shared contract.
            const CONTEXT_SEQ: usize = 154;
            let latent = crate::verify::deterministic_latent(cfg.in_channels, 64, 64, &self.device, self.dtype)?;
            let y = crate::verify::deterministic_tensor(&[1, cfg.adm_in_channels], 3, &self.device, self.dtype)?;
            let context = crate::verify::deterministic_tensor(
                &[1, CONTEXT_SEQ, cfg.context_embed_size], 2, &self.device, self.dtype,
            )?;
            let t = Tensor::full(500.0f32, (1usize,), &self.device)?.to_dtype(self.dtype)?;
            let b0 = self.mmdit_model.capture_block0(&latent, &t, &y, &context)?;
            out.insert("mmdit.block0".to_string(), b0);
        }
        Ok(out)
    }

    fn encode_prompt(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        // v1.10.0: when any Textual Inversion is loaded, take the splice path
        // (it registers + locates the trigger token and overrides its embedding
        // row in each encoder). Gated so the verified path below is untouched
        // for every non-TI generation.
        if !self.tis.is_empty() {
            return self.encode_prompt_ti(prompt);
        }
        // v0.18: BREAK is a CLIP-77-token-cap workaround. SD3's
        // T5 has a 256-token budget; per-CLIP chunking isn't wired
        // here (the pooled `y` blend assumes single-chunk CLIP-L/G
        // outputs). Strip + warn so users notice rather than getting
        // "BREAK" tokenized as a literal word.
        let prompt_stripped: String;
        let prompt: &str = if crate::prompt::break_chunks::has_break(prompt) {
            tracing::warn!(
                target: "plakat",
                "BREAK keyword ignored on SD3 / SD3.5 — T5 has a 256-token \
                 budget making the CLIP chunk workaround unnecessary, and SD3's \
                 pooled y vector assumes single-chunk CLIP outputs. Strip \
                 BREAK or switch to --model sd15 / sd21 / sdxl."
            );
            prompt_stripped = crate::prompt::break_chunks::strip(prompt);
            prompt_stripped.as_str()
        } else {
            prompt
        };

        // v0.18 phase 3: A1111 attention syntax broadcasts per-token
        // weights onto the three encoders that flow into the SD3
        // cross-attention context (CLIP-L penult, CLIP-G penult,
        // T5 hidden). Pooled outputs (l_pooled at EOT, g_pooled from
        // forward_for_sdxl) stay unweighted — pooling collapses to a
        // single position, so per-token weights have no target there.
        let has_attn = crate::prompt::a1111::has_attention_syntax(prompt);

        // ---------- CLIP-L tokenize (with optional weights) ----------
        let (clip_l_ids_t, clip_l_weights) = if has_attn {
            let wcfg = crate::prompt::weighted_encoding::WeightedTokenConfig {
                tokenizer: &self.clip_l_tok,
                max_len: self.clip_l_cfg.max_position_embeddings,
                bos_id: Some(CLIP_BOS),
                eos_id: CLIP_EOT,
                pad_id: CLIP_EOT,
            };
            let (ids, w) = crate::prompt::weighted_encoding::tokenize_with_attention(
                &wcfg,
                prompt,
                &self.device,
                self.dtype,
            )?;
            (ids, Some(w))
        } else {
            let mut clip_l_ids = self
                .clip_l_tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("CLIP-L encode: {e}"))?
                .get_ids()
                .to_vec();
            clip_l_ids.resize(self.clip_l_cfg.max_position_embeddings, CLIP_EOT);
            let t = Tensor::new(clip_l_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            (t, None)
        };

        // ---------- CLIP-L forward → pooled + penult ----------
        let clip_l_hidden = self.clip_l.forward(&clip_l_ids_t)?;
        // Recover the EOT position from the (possibly weighted) ID
        // tensor. Works for both paths because both place EOS as the
        // first CLIP_EOT in the sequence.
        let clip_l_ids_vec: Vec<u32> = clip_l_ids_t.flatten_all()?.to_vec1()?;
        let clip_l_eot_pos = clip_l_ids_vec
            .iter()
            .position(|&t| t == CLIP_EOT)
            .unwrap_or(0);
        let clip_l_pooled = clip_l_hidden
            .i((.., clip_l_eot_pos, ..))?
            .to_dtype(self.dtype)?;

        // ---------- CLIP-G tokenize (with optional weights) ----------
        let (clip_g_ids_t, clip_g_weights) = if has_attn {
            let wcfg = crate::prompt::weighted_encoding::WeightedTokenConfig {
                tokenizer: &self.clip_g_tok,
                max_len: 77,
                bos_id: Some(CLIP_BOS),
                eos_id: CLIP_EOT,
                pad_id: CLIP_EOT,
            };
            let (ids, w) = crate::prompt::weighted_encoding::tokenize_with_attention(
                &wcfg,
                prompt,
                &self.device,
                self.dtype,
            )?;
            (ids, Some(w))
        } else {
            let mut clip_g_ids = self
                .clip_g_tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("CLIP-G encode: {e}"))?
                .get_ids()
                .to_vec();
            clip_g_ids.resize(77, CLIP_EOT);
            let t = Tensor::new(clip_g_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            (t, None)
        };

        // ---------- CLIP-G forward_for_sdxl → penult + pooled ----------
        let (clip_g_penult, clip_g_pooled) = self.clip_g.forward_for_sdxl(&clip_g_ids_t)?;
        let mut clip_g_penult = clip_g_penult.to_dtype(self.dtype)?;
        let clip_g_pooled = clip_g_pooled.to_dtype(self.dtype)?;
        if let Some(w) = &clip_g_weights {
            clip_g_penult = clip_g_penult.broadcast_mul(&w.to_dtype(self.dtype)?)?;
        }

        // ---------- T5-XXL tokenize (with optional weights) ----------
        let t5_seq = self.variant.t5_seq_len();
        let (t5_ids_t, t5_weights) = if has_attn {
            let t5_eos = self
                .t5_tok
                .token_to_id("</s>")
                .ok_or_else(|| anyhow!("T5 tokenizer missing </s>"))?;
            let t5_pad = self.t5_tok.token_to_id("<pad>").unwrap_or(0);
            let wcfg = crate::prompt::weighted_encoding::WeightedTokenConfig {
                tokenizer: &self.t5_tok,
                max_len: t5_seq,
                bos_id: None,
                eos_id: t5_eos,
                pad_id: t5_pad,
            };
            let (ids, w) = crate::prompt::weighted_encoding::tokenize_with_attention(
                &wcfg,
                prompt,
                &self.device,
                self.dtype,
            )?;
            (ids, Some(w))
        } else {
            let mut t5_ids = self
                .t5_tok
                .encode(prompt, true)
                .map_err(|e| anyhow!("T5 encode: {e}"))?
                .get_ids()
                .to_vec();
            t5_ids.truncate(t5_seq);
            t5_ids.resize(t5_seq, 0);
            let t = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            (t, None)
        };

        // ---------- T5 forward (with pad attention mask) ----------
        // v2.1: mask pad tokens so real tokens don't attend to padding in T5 self-attention
        // (diffusers always passes attention_mask). T5 `<pad>` = id 0, so the mask is simply
        // (ids != 0): 1 for real tokens incl. EOS, 0 for pad. Without it the caption drifts
        // (corr ~0.7 vs correct) — the same bug fixed for PixArt.
        let t5_mask = t5_ids_t.ne(0u32)?.to_dtype(candle_core::DType::F32)?;
        let mut t5_hidden = self.t5_enc.forward_with_mask(&t5_ids_t, &t5_mask)?.to_dtype(self.dtype)?;
        if let Some(w) = &t5_weights {
            t5_hidden = t5_hidden.broadcast_mul(&w.to_dtype(self.dtype)?)?;
        }

        // ---------- Pooled (y) ----------
        // SD3 convention (diffusers): CLIP-L pooled FIRST (768), CLIP-G
        // pooled second (1280) → (1, 2048). The previous order was
        // swapped, which scrambles the y_embedder input (the pooled
        // vector drives adaLN modulation across the whole MMDiT) and
        // prevents the model from ever denoising.
        let y = Tensor::cat(&[&clip_l_pooled, &clip_g_pooled], candle_core::D::Minus1)?;

        // ---------- CLIP-L penultimate (weighted if has_attn) ----------
        // CLIP-L's penultimate hidden state is what SD3 mixes with
        // CLIP-G's penultimate. We grab CLIP-L penultimate by running
        // until layer -2 (matching SDXL's convention).
        let (_clip_l_final, clip_l_penult) = {
            let (final_h, pen_h) =
                crate::pipelines::vendored_clip::ClipTextTransformer::forward_until_encoder_layer(
                    &self.clip_l,
                    &clip_l_ids_t,
                    usize::MAX,
                    -2,
                )?;
            (final_h, pen_h)
        };
        let mut clip_l_penult = clip_l_penult.to_dtype(self.dtype)?;
        if let Some(w) = &clip_l_weights {
            clip_l_penult = clip_l_penult.broadcast_mul(&w.to_dtype(self.dtype)?)?;
        }
        // Concat CLIP halves along channel: (1, 77, 768) + (1, 77, 1280)
        //   → (1, 77, 2048).
        let clip_concat = Tensor::cat(&[&clip_l_penult, &clip_g_penult], candle_core::D::Minus1)?;
        // Pad along the channel dim from 2048 → 4096 with zeros so it
        // can be sequence-concatenated with T5's 4096-d hidden.
        let (b, seq, _clip_ch) = clip_concat.dims3()?;
        let pad =
            Tensor::zeros((b, seq, 4096 - 2048), self.dtype, &self.device)?;
        let clip_padded =
            Tensor::cat(&[&clip_concat, &pad], candle_core::D::Minus1)?;
        // Sequence-concatenate with T5: (1, 77, 4096) + (1, t5_seq, 4096)
        //   → (1, 77+t5_seq, 4096).
        let context = Tensor::cat(&[&clip_padded, &t5_hidden], 1)?;

        Ok((y, context))
    }

    /// Textual-Inversion encode (v1.10.0). Mirrors `encode_prompt`'s triple-
    /// encoder assembly, but each encoder runs from spliced input embeddings:
    /// the trigger token's out-of-vocab id is clamped to a valid row for the
    /// lookup, then that row is overwritten by the learned vector·scale before
    /// the encoder stack runs. (A1111 attention weighting isn't combined with
    /// TI here — a niche overlap; the plain tokenization is used.)
    fn encode_prompt_ti(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let dtype = self.dtype;

        // ---- CLIP-L (pooled at EOT + penultimate) ----
        let mut l_ids: Vec<u32> = self
            .clip_l_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("CLIP-L encode: {e}"))?
            .get_ids()
            .to_vec();
        l_ids.resize(self.clip_l_cfg.max_position_embeddings, CLIP_EOT);
        let l_eot = l_ids.iter().position(|&t| t == CLIP_EOT).unwrap_or(0);
        let l_embeds = self.ti_embeds_clip_l(&l_ids)?;
        let (l_final, l_penult) = self
            .clip_l
            .forward_until_encoder_layer_from_embeds(&l_embeds, usize::MAX, -2)?;
        let l_pooled = l_final.i((.., l_eot, ..))?.to_dtype(dtype)?;
        let l_penult = l_penult.to_dtype(dtype)?;

        // ---- CLIP-G (penultimate + pooled via argmax over clamped ids) ----
        let mut g_ids: Vec<u32> = self
            .clip_g_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("CLIP-G encode: {e}"))?
            .get_ids()
            .to_vec();
        g_ids.resize(77, CLIP_EOT);
        let (g_embeds, g_clamped) = self.ti_embeds_clip_g(&g_ids)?;
        let (g_penult, g_pooled) =
            self.clip_g.forward_for_sdxl_from_embeds(&g_embeds, &g_clamped)?;
        let g_penult = g_penult.to_dtype(dtype)?;
        let g_pooled = g_pooled.to_dtype(dtype)?;

        // ---- T5 ----
        let t5_seq = self.variant.t5_seq_len();
        let mut t_ids: Vec<u32> = self
            .t5_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("T5 encode: {e}"))?
            .get_ids()
            .to_vec();
        t_ids.truncate(t5_seq);
        t_ids.resize(t5_seq, 0);
        let t5_embeds = self.ti_embeds_t5(&t_ids)?;
        let t5_hidden = self.t5_enc.forward_from_input_embeds(&t5_embeds)?.to_dtype(dtype)?;

        // ---- assemble (identical layout to encode_prompt) ----
        let y = Tensor::cat(&[&l_pooled, &g_pooled], candle_core::D::Minus1)?;
        let clip_concat = Tensor::cat(&[&l_penult, &g_penult], candle_core::D::Minus1)?;
        let (b, seq, _ch) = clip_concat.dims3()?;
        let pad = Tensor::zeros((b, seq, 4096 - 2048), dtype, &self.device)?;
        let clip_padded = Tensor::cat(&[&clip_concat, &pad], candle_core::D::Minus1)?;
        let context = Tensor::cat(&[&clip_padded, &t5_hidden], 1)?;
        Ok((y, context))
    }

    /// Build CLIP-L input embeddings with each loaded TI's learned vector
    /// spliced into its trigger position (id matched against `id_l`). The
    /// trigger's out-of-vocab id is looked up as row 0 (a throwaway — the row
    /// is overwritten), so the lookup never goes out of bounds.
    fn ti_embeds_clip_l(&self, ids: &[u32]) -> Result<Tensor> {
        let dim = 768usize;
        let clamped: Vec<u32> = ids
            .iter()
            .map(|&id| if self.tis.iter().any(|t| t.id_l == id) { 0 } else { id })
            .collect();
        let ids_t = Tensor::new(clamped.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut embeds = self.clip_l.embed_tokens(&ids_t)?;
        for (pos, &id) in ids.iter().enumerate() {
            if let Some(ti) = self.tis.iter().find(|t| t.id_l == id) {
                let row = ti.clip_l.affine(ti.scale as f64, 0.0)?
                    .reshape((1, 1, dim))?
                    .to_dtype(embeds.dtype())?;
                embeds = embeds.slice_assign(&[0..1, pos..pos + 1, 0..dim], &row)?;
            }
        }
        Ok(embeds)
    }

    /// CLIP-G counterpart. Returns `(embeds, clamped_ids)` — the clamped ids
    /// (trigger → 0) feed `forward_for_sdxl_from_embeds`'s argmax pooling so
    /// EOT (49407) stays the unique max and the pooled row is correct.
    fn ti_embeds_clip_g(&self, ids: &[u32]) -> Result<(Tensor, Tensor)> {
        let dim = 1280usize;
        let clamped: Vec<u32> = ids
            .iter()
            .map(|&id| if self.tis.iter().any(|t| t.id_g == id) { 0 } else { id })
            .collect();
        let ids_t = Tensor::new(clamped.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut embeds = self.clip_g.embed_tokens(&ids_t)?;
        for (pos, &id) in ids.iter().enumerate() {
            if let Some(ti) = self.tis.iter().find(|t| t.id_g == id) {
                let row = ti.clip_g.affine(ti.scale as f64, 0.0)?
                    .reshape((1, 1, dim))?
                    .to_dtype(embeds.dtype())?;
                embeds = embeds.slice_assign(&[0..1, pos..pos + 1, 0..dim], &row)?;
            }
        }
        Ok((embeds, ids_t))
    }

    /// T5 counterpart (4096d).
    fn ti_embeds_t5(&self, ids: &[u32]) -> Result<Tensor> {
        let dim = 4096usize;
        let clamped: Vec<u32> = ids
            .iter()
            .map(|&id| if self.tis.iter().any(|t| t.id_t5 == id) { 0 } else { id })
            .collect();
        let ids_t = Tensor::new(clamped.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut embeds = self.t5_enc.embed_tokens(&ids_t)?;
        for (pos, &id) in ids.iter().enumerate() {
            if let Some(ti) = self.tis.iter().find(|t| t.id_t5 == id) {
                let row = ti.t5.affine(ti.scale as f64, 0.0)?
                    .reshape((1, 1, dim))?
                    .to_dtype(embeds.dtype())?;
                embeds = embeds.slice_assign(&[0..1, pos..pos + 1, 0..dim], &row)?;
            }
        }
        Ok(embeds)
    }
}

/// Apply the SD3 time-shift transform to a `[0, 1]` linear schedule.
/// Diffusers' `FlowMatchEulerDiscreteScheduler` calls this `mu_t`.
/// `shift = 1.0` is the identity; higher values push more steps into
/// the high-noise region (where the model has more uncertainty to
/// resolve).
/// v0.15 phase 5: place a tile-shaped tensor into a zero-padded
/// canvas-shaped tensor at the given top-left offset. The tile may
/// have a batch dim (1, C, T, T) or no batch dim (1, 1, T, T) for
/// the Hann weight. The returned tensor matches the tile's channel
/// count and the requested canvas spatial size.
///
/// Builds the padded tensor via three `Tensor::cat` calls — top/bot
/// rows of zeros and left/right cols of zeros wrapping the tile.
/// candle 0.8 has no `slice_assign`, so this cat-based approach is
/// the cleanest way to lift a sub-region into a larger canvas.
fn pad_tile_to_canvas(
    tile: &Tensor,
    y: usize,
    x: usize,
    canvas_h: usize,
    canvas_w: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (b, c, th, tw) = tile.dims4()?;
    // Left + right horizontal pads. Both can be width 0; candle's cat
    // handles zero-width inputs fine.
    let left = Tensor::zeros((b, c, th, x), dtype, device)?;
    let right = Tensor::zeros(
        (b, c, th, canvas_w.saturating_sub(x + tw)),
        dtype,
        device,
    )?;
    let row = Tensor::cat(&[&left, tile, &right], 3)?;
    let top = Tensor::zeros((b, c, y, canvas_w), dtype, device)?;
    let bot = Tensor::zeros(
        (b, c, canvas_h.saturating_sub(y + th), canvas_w),
        dtype,
        device,
    )?;
    Ok(Tensor::cat(&[&top, &row, &bot], 2)?)
}

/// v0.16 phase 3d: element-wise sum of two residual lists for the
/// SD3 ControlNet multi-CN composition path. Mirrors the same-named
/// helper in `flux.rs`. When the accumulator is shorter than the new
/// list, the new entries are appended; when longer, missing entries
/// from the new list contribute zero (so the longer list's tail
/// passes through unchanged).
fn merge_residuals(
    acc: Option<Vec<Tensor>>,
    new: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
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

/// v0.16 phase 10: derive the progress-bar mode tag from the SD3
/// pipeline's active feature flags. Pure function so the test suite
/// can pin every combination without needing real weights.
///
/// Truth table:
/// ```text
///   tiled  init  mask  → tag
///   T      T     T     → "tiled inpaint"
///   T      T     F     → "tiled img2img"
///   T      F     _     → "tiled"
///   F      T     T     → "inpaint"
///   F      T     F     → "img2img"
///   F      F     _     → "denoise"
/// ```
fn sd3_mode_tag(tiled: bool, init: bool, mask: bool) -> &'static str {
    match (tiled, init, mask) {
        (true, true, true) => "tiled inpaint",
        (true, true, false) => "tiled img2img",
        (true, false, _) => "tiled",
        (false, true, true) => "inpaint",
        (false, true, false) => "img2img",
        _ => "denoise",
    }
}

fn shift_t(t: f64, shift: f64) -> f64 {
    if shift == 1.0 {
        t
    } else {
        shift * t / (1.0 + (shift - 1.0) * t)
    }
}

/// v0.15 phase 2: build the SD3 img2img schedule — same construction
/// the `generate` loop runs inline, factored out for unit testing.
///
/// Pure t2i (`strength == None`): the full linear `[1.0 → 0.0]`
/// schedule with `shift_t` applied.
///
/// img2img (`strength = Some(s)`): drop schedule entries with
/// `shifted_t >= s` and prepend `s` itself, so the first window's
/// `t_curr` matches the noise level of the `lerp(init, eps, s)`
/// start latent. Mirrors `FluxImg2ImgPipeline.get_timesteps`.
fn build_img2img_timesteps(steps: usize, shift: f64, strength: Option<f64>) -> Vec<f64> {
    let full: Vec<f64> = (0..=steps)
        .map(|v| 1.0 - (v as f64 / steps as f64))
        .map(|t| shift_t(t, shift))
        .collect();
    match strength {
        None => full,
        Some(s) => {
            let mut filtered: Vec<f64> = full.into_iter().filter(|t| *t < s).collect();
            if filtered.is_empty() {
                filtered.push(0.0);
            }
            let mut new_ts = Vec::with_capacity(filtered.len() + 1);
            new_ts.push(s);
            new_ts.extend(filtered);
            new_ts
        }
    }
}

pub async fn run(req: Request) -> Result<()> {
    let output_format = req.output_format;
    let mut p = Pipeline::load(LoadRequest {
        variant: req.variant,
        repo: req.repo,
        device: req.device,
        loras: req.loras,
        lora_scale: req.lora_scale,
        // v0.16 phase 3e: wired through t2i::run from CLI
        // --control-spec flags. Scenarios build sd3::Pipeline
        // directly and bypass `run` (set_controlnet_conditioning /
        // set_controlnet_call_params drive per-task changes).
        controlnets: req.controlnets,
        embeddings: req.embeddings,
    })
    .await?;
    p.generate(&GenRequest {
        prompt: req.prompt,
        negative: req.negative,
        width: req.width,
        height: req.height,
        count: req.count,
        steps: req.steps,
        guidance: req.guidance,
        seed: req.seed,
        out_dir: req.out_dir,
        init_image: req.init_image,
        mask: req.mask,
        mask_feather: req.mask_feather,
        mask_invert: req.mask_invert,
        strength: req.strength,
        tiled: req.tiled,
        regions: req.regions,
        controlnet_conditioning: Vec::new(),
        output_format,
    })
}

// =====================================================================
// `plakat style train` (Phase 1, v0.45.0) — train a style LoRA on a
// folder of images against the SD3.5 MMDiT, saving a diffusers-PEFT
// `.safetensors` loadable via `--lora`. Rectified-flow objective.
//
// Two memory phases (24 GB-safe): encode the images + caption with the
// BF16 pipeline and DROP it, then load the MMDiT in F32 for training.
// =====================================================================

/// Inputs for [`train_style_lora`].
pub struct StyleTrainRequest {
    pub variant: Variant,
    pub repo: String,
    pub device: Device,
    pub images: Vec<std::path::PathBuf>,
    pub trigger: String,
    pub rank: usize,
    pub steps: usize,
    pub lr: f64,
    pub size: u32,
    pub out: std::path::PathBuf,
    /// Explicit checkpoint interval in steps. `None` → ~10 evenly-spaced
    /// (`checkpoint_interval`). `0` is treated as `None`.
    pub checkpoint_every: Option<usize>,
    /// Log a progress line every N steps (min 1).
    pub log_every: usize,
    /// Resume from a checkpoint (a diffusers-PEFT LoRA written by an earlier
    /// run). The fused adapters are reconstructed from it and the step counter
    /// continues from the checkpoint's step up to `steps`. `None` = from scratch.
    pub resume_from: Option<std::path::PathBuf>,
    /// DreamBooth prior preservation: a few generic CLASS images (e.g. other
    /// dogs) trained alongside the subject under `class_prompt`, so the rare
    /// subject token doesn't overfit or collapse the whole class. Empty (the
    /// default) = plain style/subject training, no prior loss — loop unchanged.
    pub class_images: Vec<std::path::PathBuf>,
    /// Class prompt for `class_images` (e.g. "a photo of a dog"). Required when
    /// `class_images` is non-empty.
    pub class_prompt: Option<String>,
    /// Weight on the prior-preservation loss (DreamBooth's λ; typical ~1.0).
    pub prior_weight: f32,
}

/// Train a style LoRA on the MMDiT attention projections; write a
/// diffusers-PEFT safetensors. Rectified flow: `x_σ=(1-σ)x₀+σε`, the
/// model predicts the velocity `v=ε-x₀`.
pub async fn train_style_lora(req: StyleTrainRequest) -> Result<()> {
    use candle_core::Var;
    use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

    let device = req.device.clone();
    let cfg = req.variant.mmdit_config();
    let hidden = cfg.head_size * cfg.depth;

    // --- Phase A: encode images + caption (BF16 pipeline), then drop it.
    tracing::info!(
        "style-train: encoding {} image(s) + caption \"{}\"",
        req.images.len(),
        req.trigger
    );
    let (latents, y, context, class_data, dtype) = {
        let mut pipe = Pipeline::load(LoadRequest {
            variant: req.variant,
            repo: req.repo.clone(),
            device: device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            controlnets: Vec::new(),
            embeddings: Vec::new(),
        })
        .await?;
        let dtype = pipe.dtype;
        let encode_imgs = |pipe: &mut Pipeline, imgs: &[std::path::PathBuf]| -> Result<Vec<Tensor>> {
            let mut v = Vec::with_capacity(imgs.len());
            for img in imgs {
                let px = crate::imaging::preprocess::sd_image_tensor(
                    img.as_path(),
                    req.size,
                    req.size,
                    &device,
                    dtype,
                )?;
                let z = pipe.vae.encode(&px)?.sample()?;
                v.push(((z - VAE_SHIFT)? * VAE_SCALE)?);
            }
            Ok(v)
        };
        let (y, context) = pipe.encode_prompt(&req.trigger)?;
        let latents = encode_imgs(&mut pipe, &req.images)?;
        // DreamBooth prior preservation (optional): encode the class set + its
        // prompt's (y, context) too, while the triple encoder is still loaded.
        let class_data = if req.class_images.is_empty() {
            None
        } else {
            let cp = req.class_prompt.as_deref().ok_or_else(|| {
                anyhow::anyhow!("prior preservation: --class-prompt is required when class images are given")
            })?;
            let (cy, ccontext) = pipe.encode_prompt(cp)?;
            let clats = encode_imgs(&mut pipe, &req.class_images)?;
            Some((clats, cy, ccontext))
        };
        (latents, y, context, class_data, dtype)
    }; // BF16 pipeline (MMDiT + T5 + CLIP + VAE) dropped here → freed

    // --- Phase B: load MMDiT in F32, install trainable adapters.
    tracing::info!("style-train: loading MMDiT (F32) for training");
    let mmdit_path = crate::hf::download::get_first_of(&[
        (&req.repo, "transformer/diffusion_pytorch_model.safetensors"),
        (&req.repo, "sd3.5_medium.safetensors"),
    ])
    .await?;
    // BF16 base (Metal-fast, half the memory of F32); the trainable LoRA
    // adapters stay F32 for stable AdamW (LoraLinear casts at the boundary).
    let vb = build_mmdit_vb(&mmdit_path, dtype, &device, cfg.depth)?;
    let model = mmdit::MMDiT::new(&cfg, false, vb)?;
    let adapters = model.install_train_adapters(req.rank, 1.0, &device)?;
    tracing::info!(
        "style-train: {} trainable attention adapters (rank {})",
        adapters.len(),
        req.rank
    );
    let vars: Vec<Var> = adapters
        .iter()
        .flat_map(|(_, a, b)| [a.clone(), b.clone()])
        .collect();
    let mut opt = AdamW::new(vars.clone(), ParamsAdamW { lr: req.lr, ..Default::default() })?;

    // --- Phase C: rectified-flow training loop.
    let n = latents.len().max(1);
    let mut progress = crate::pipelines::train_progress::TrainProgress::new(
        req.steps,
        req.lr,
        checkpoint_interval(req.checkpoint_every, req.steps),
    );
    // Additive: `start_step` is 0 unless --resume, so the loop is unchanged.
    let start_step = match &req.resume_from {
        Some(ckpt) => {
            load_peft_into_adapters(&adapters, ckpt, &device)?;
            let s = crate::pipelines::sd_train::trainer::parse_resume_step(ckpt)
                .unwrap_or(0)
                .min(req.steps);
            if s >= req.steps {
                bail!(
                    "style-train: --resume checkpoint at step {s} ≥ --steps {}; \
                     raise --steps to continue training",
                    req.steps
                );
            }
            tracing::info!(
                "style-train: resuming from {} at step {s}/{}",
                ckpt.display(),
                req.steps
            );
            s
        }
        None => 0,
    };
    for step in start_step..req.steps {
        let x0 = &latents[step % n];
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &device)?.to_dtype(dtype)?;
        let sigma = 0.05
            + 0.90 * Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] as f64;
        let x_t = ((x0 * (1.0 - sigma))? + (&noise * sigma)?)?;
        let target = (&noise - x0)?;
        // Timestep stays F32 — the embedder requires it and casts to the
        // model dtype internally.
        let t_vec = Tensor::full((sigma * 1000.0) as f32, (1usize,), &device)?;
        let pred = model.forward(&x_t, &t_vec, &y, &context, None)?;
        let loss = (&pred - &target)?.sqr()?.mean_all()?.to_dtype(DType::F32)?;
        // DreamBooth prior preservation: add the class loss on an INDEPENDENT
        // class sample / sigma / noise (same rectified-flow objective), so the
        // rare subject token doesn't overfit or collapse the broader class. No
        // class data → this is plain training and the term is skipped.
        let loss = if let Some((class_lat, cy, ccontext)) = &class_data {
            let cn = class_lat.len().max(1);
            let cx0 = &class_lat[step % cn];
            let cnoise = Tensor::randn(0f32, 1f32, cx0.dims(), &device)?.to_dtype(dtype)?;
            let csigma = 0.05
                + 0.90 * Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] as f64;
            let cx_t = ((cx0 * (1.0 - csigma))? + (&cnoise * csigma)?)?;
            let ctarget = (&cnoise - cx0)?;
            let ct_vec = Tensor::full((csigma * 1000.0) as f32, (1usize,), &device)?;
            let cpred = model.forward(&cx_t, &ct_vec, cy, ccontext, None)?;
            let closs = (&cpred - &ctarget)?.sqr()?.mean_all()?.to_dtype(DType::F32)?;
            (&loss + (closs * req.prior_weight as f64)?)?
        } else {
            loss
        };
        let mut grads = loss.backward()?;
        crate::pipelines::lora_linear::clip_grad_norm(&mut grads, &vars, 1.0)?;
        opt.step(&grads)?;
        if step % req.log_every.max(1) == 0 || step + 1 == req.steps {
            tracing::info!(
                "{}",
                progress.line("style-train", step + 1, loss.to_scalar::<f32>()?)
            );
        }
        // Periodic NUMBERED checkpoint (`<stem>-step<N>`) so a long run can be
        // swept after the fact for the best step — the best LoRA is rarely the
        // last (training over-cooks). Set PLAKAT_TRAIN_SINGLE_FILE=1 to
        // overwrite one file instead. The final save writes plain `--out`.
        if (step + 1) % checkpoint_interval(req.checkpoint_every, req.steps) == 0
            && step + 1 != req.steps
        {
            let ckpt = checkpoint_path(&req.out, step + 1);
            save_peft_lora(&adapters, req.rank, hidden, &ckpt)?;
            tracing::info!("style-train: checkpoint @ step {} → {}", step + 1, ckpt.display());
        }
    }

    // --- Phase D: save diffusers-PEFT safetensors.
    save_peft_lora(&adapters, req.rank, hidden, &req.out)?;
    tracing::info!("style-train: wrote {}", req.out.display());
    tracing::info!("{}", progress.finish("style-train", &req.out));
    Ok(())
}

/// SD 3.5 Textual Inversion: learn a placeholder token vector in EACH of the
/// three text encoders (CLIP-L 768 + CLIP-G 1280 + T5 4096) with the whole
/// model **frozen**. Reproduces SD3's exact triple-encoder conditioning with
/// the trainable vectors spliced into each encoder's init-word slot (a
/// differentiable masked combine → the gradient reaches only those vectors),
/// and backprops the **rectified-flow velocity** loss through the frozen MMDiT.
/// Saves a triple embedding file (`clip_l` + `clip_g` + `t5`), loadable via
/// `--embedding PATH:trigger`.
///
/// **Memory-bound** like the SD3.5 LoRA trainer: CLIP-L + CLIP-G + T5-XXL +
/// MMDiT must all stay resident in one autograd forward — past 24 GB on the
/// canonical checkpoint. The training itself is light (three vectors); the wall
/// is the resident encoders. Keep `--size` modest.
pub async fn train_textual_inversion(
    req: crate::pipelines::ti_train::TiTrainRequest,
) -> Result<()> {
    use crate::pipelines::ti_train::{slot_masks, splice};
    use candle_core::Var;
    use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

    let device = req.device.clone();
    // SD3.5-medium is the only published SD3 checkpoint plakat verifies.
    let pipe = Pipeline::load(LoadRequest {
        variant: Variant::Sd35Medium,
        repo: "stabilityai/stable-diffusion-3.5-medium".to_string(),
        device: device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        controlnets: Vec::new(),
        embeddings: Vec::new(),
    })
    .await?;
    let dtype = pipe.dtype;
    tracing::info!(
        "ti-train(sd35): frozen MMDiT + triple encoders loaded; encoding {} image(s)",
        req.images.len()
    );

    // --- encode images → latents (frozen VAE, SD3 normalization) ---
    let mut latents = Vec::with_capacity(req.images.len());
    for img in &req.images {
        let px = crate::imaging::preprocess::sd_image_tensor(
            img.as_path(),
            req.size,
            req.size,
            &device,
            dtype,
        )?;
        let z = pipe.vae.encode(&px)?.sample()?;
        latents.push(((z - VAE_SHIFT)? * VAE_SCALE)?.to_dtype(dtype)?);
    }
    let n = latents.len().max(1);

    // --- template "a photo of <init>"; per-encoder token slot ---
    let prompt = format!("a photo of {}", req.init_word.trim());

    // CLIP-L ids + slot + EOT position (the pooled row).
    let mut l_ids: Vec<u32> = pipe
        .clip_l_tok
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow!("CLIP-L encode: {e}"))?
        .get_ids()
        .to_vec();
    let l_max = pipe.clip_l_cfg.max_position_embeddings;
    l_ids.resize(l_max, CLIP_EOT);
    let l_eot = l_ids.iter().position(|&t| t == CLIP_EOT).unwrap_or(0);
    if l_eot < 2 {
        bail!(
            "init word {:?} tokenized oddly — pick a simple class word (e.g. 'art', 'toy')",
            req.init_word
        );
    }
    let slot_l = l_eot - 1;
    let ids_l = Tensor::new(l_ids.as_slice(), &device)?.unsqueeze(0)?;

    // CLIP-G ids + slot.
    let mut g_ids: Vec<u32> = pipe
        .clip_g_tok
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow!("CLIP-G encode: {e}"))?
        .get_ids()
        .to_vec();
    g_ids.resize(77, CLIP_EOT);
    let g_eot = g_ids.iter().position(|&t| t == CLIP_EOT).unwrap_or(0);
    let slot_g = g_eot.saturating_sub(1);
    let ids_g = Tensor::new(g_ids.as_slice(), &device)?.unsqueeze(0)?;

    // T5 ids + slot (</s>-terminated, pad 0 to the T5 budget).
    let t5_seq = pipe.variant.t5_seq_len();
    let t5_eos = pipe
        .t5_tok
        .token_to_id("</s>")
        .ok_or_else(|| anyhow!("T5 tokenizer missing </s>"))?;
    let mut t_ids: Vec<u32> = pipe
        .t5_tok
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow!("T5 encode: {e}"))?
        .get_ids()
        .to_vec();
    let t5_eos_pos = t_ids
        .iter()
        .position(|&t| t == t5_eos)
        .unwrap_or(t_ids.len().saturating_sub(1));
    let slot_t5 = t5_eos_pos.saturating_sub(1);
    t_ids.truncate(t5_seq);
    t_ids.resize(t5_seq, 0);
    let ids_t5 = Tensor::new(t_ids.as_slice(), &device)?.unsqueeze(0)?;

    // --- init each placeholder from its encoder's init-word embedding ---
    let ph_l = Var::from_tensor(
        &pipe.clip_l.embed_tokens(&ids_l)?.i((0, slot_l))?.to_dtype(DType::F32)?.unsqueeze(0)?,
    )?;
    let ph_g = Var::from_tensor(
        &pipe.clip_g.embed_tokens(&ids_g)?.i((0, slot_g))?.to_dtype(DType::F32)?.unsqueeze(0)?,
    )?;
    let ph_t5 = Var::from_tensor(
        &pipe.t5_enc.embed_tokens(&ids_t5)?.i((0, slot_t5))?.to_dtype(DType::F32)?.unsqueeze(0)?,
    )?;

    let (mask_l, inv_l) = slot_masks(slot_l, l_max, &device, dtype)?;
    let (mask_g, inv_g) = slot_masks(slot_g, 77, &device, dtype)?;
    let (mask_t5, inv_t5) = slot_masks(slot_t5, t5_seq, &device, dtype)?;

    let mut opt = AdamW::new(
        vec![ph_l.clone(), ph_g.clone(), ph_t5.clone()],
        ParamsAdamW { lr: req.lr, ..Default::default() },
    )?;

    let mut progress =
        crate::pipelines::train_progress::TrainProgress::new(req.steps, req.lr, req.steps + 1);
    tracing::info!(
        "ti-train(sd35): token {:?} init from {:?} (slots L{slot_l}/G{slot_g}/T5{slot_t5}), {} steps",
        req.token,
        req.init_word,
        req.steps
    );

    for step in 0..req.steps {
        // Splice each trainable vector into its encoder, then reproduce SD3's
        // exact triple-encoder conditioning.
        let spliced_l = splice(&pipe.clip_l.embed_tokens(&ids_l)?, &ph_l, &mask_l, &inv_l, dtype)?;
        let (l_final, l_penult) =
            pipe.clip_l.forward_until_encoder_layer_from_embeds(&spliced_l, usize::MAX, -2)?;
        let l_pooled = l_final.i((.., l_eot, ..))?.to_dtype(dtype)?;
        let l_penult = l_penult.to_dtype(dtype)?;

        let spliced_g = splice(&pipe.clip_g.embed_tokens(&ids_g)?, &ph_g, &mask_g, &inv_g, dtype)?;
        let (g_penult, g_pooled) = pipe.clip_g.forward_for_sdxl_from_embeds(&spliced_g, &ids_g)?;
        let g_penult = g_penult.to_dtype(dtype)?;
        let g_pooled = g_pooled.to_dtype(dtype)?;

        let spliced_t5 = splice(&pipe.t5_enc.embed_tokens(&ids_t5)?, &ph_t5, &mask_t5, &inv_t5, dtype)?;
        let t5_hidden = pipe.t5_enc.forward_from_input_embeds(&spliced_t5)?.to_dtype(dtype)?;

        // y = [CLIP-L pooled, CLIP-G pooled] (768+1280=2048) — the SD3 order.
        let y = Tensor::cat(&[&l_pooled, &g_pooled], candle_core::D::Minus1)?;
        // context = [CLIP-L⊕CLIP-G penult → 2048, zero-pad → 4096] ⧺seq T5.
        let clip_concat = Tensor::cat(&[&l_penult, &g_penult], candle_core::D::Minus1)?;
        let (b, seq, _ch) = clip_concat.dims3()?;
        let pad = Tensor::zeros((b, seq, 4096 - 2048), dtype, &device)?;
        let clip_padded = Tensor::cat(&[&clip_concat, &pad], candle_core::D::Minus1)?;
        let context = Tensor::cat(&[&clip_padded, &t5_hidden], 1)?;

        // rectified-flow velocity loss through the frozen MMDiT.
        let x0 = &latents[step % n];
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &device)?.to_dtype(dtype)?;
        let sigma = 0.05
            + 0.90 * Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] as f64;
        let x_t = ((x0 * (1.0 - sigma))? + (&noise * sigma)?)?;
        let target = (&noise - x0)?;
        let t_vec = Tensor::full((sigma * 1000.0) as f32, (1usize,), &device)?;
        let pred = pipe.mmdit_model.forward(&x_t, &t_vec, &y, &context, None)?;
        let loss = (&pred - &target)?.sqr()?.mean_all()?.to_dtype(DType::F32)?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
        if step % req.log_every.max(1) == 0 || step + 1 == req.steps {
            tracing::info!(
                "{}",
                progress.line("ti-train(sd35)", step + 1, loss.to_scalar::<f32>()?)
            );
        }
    }

    // --- save the triple embedding: clip_l (1,768) + clip_g (1,1280) + t5 (1,4096) ---
    if let Some(parent) = req.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut tensors: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    tensors.insert(
        "clip_l".into(),
        ph_l.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?,
    );
    tensors.insert(
        "clip_g".into(),
        ph_g.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?,
    );
    tensors.insert(
        "t5".into(),
        ph_t5.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?,
    );
    crate::pipelines::atomic_safetensors_save(&tensors, &req.out)?;
    tracing::info!(
        "ti-train(sd35): wrote {} (clip_l+clip_g+t5) — use it with  --embedding {}:{}",
        req.out.display(),
        req.out.display(),
        req.token
    );
    tracing::info!("{}", progress.finish("ti-train(sd35)", &req.out));
    Ok(())
}

/// Numbered checkpoint path (`<stem>-step<N>.<ext>`) — see the SD trainer's
/// copy. `PLAKAT_TRAIN_SINGLE_FILE=1` overwrites the plain `--out` instead.
fn checkpoint_path(out: &std::path::Path, step: usize) -> PathBuf {
    if std::env::var_os("PLAKAT_TRAIN_SINGLE_FILE").is_some() {
        return out.to_path_buf();
    }
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("lora");
    let ext = out.extension().and_then(|s| s.to_str()).unwrap_or("safetensors");
    out.with_file_name(format!("{stem}-step{step}.{ext}"))
}

/// Resolve checkpoint interval: explicit `--checkpoint-every` (positive) wins,
/// else ~10 evenly-spaced (min every 30).
fn checkpoint_interval(every: Option<usize>, total_steps: usize) -> usize {
    every
        .filter(|&n| n > 0)
        .unwrap_or_else(|| (total_steps / 10).max(30))
}

/// Write trained MMDiT attention adapters as a diffusers-PEFT LoRA
/// (`lora_A`/`lora_B`/`alpha`). Fused qkv → q/k/v (shared A, sliced B);
/// proj → to_out.0 / to_add_out. attn-only (the SD3 loader skips attn2).
fn save_peft_lora(
    adapters: &[(String, candle_core::Var, candle_core::Var)],
    rank: usize,
    hidden: usize,
    out: &std::path::Path,
) -> Result<()> {
    use std::collections::HashMap;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let alpha = Tensor::new(rank as f32, &Device::Cpu)?;
    for (key, a, b) in adapters {
        // joint_blocks.{i}.{x_block|context_block}.attn.{qkv|proj}.weight
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() < 5 {
            continue;
        }
        let i: usize = match parts[1].parse() {
            Ok(i) => i,
            Err(_) => continue,
        };
        let is_ctx = parts[2] == "context_block";
        let kind = parts[4];
        let a_t = a.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        let b_t = b.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        let base = format!("transformer.transformer_blocks.{i}.attn");
        let subs: &[&str] = match (is_ctx, kind) {
            (false, "qkv") => &["to_q", "to_k", "to_v"],
            (true, "qkv") => &["add_q_proj", "add_k_proj", "add_v_proj"],
            (false, "proj") => &["to_out.0"],
            (true, "proj") => &["to_add_out"],
            _ => &[],
        };
        let mut targets: Vec<(String, Tensor)> = Vec::new();
        if kind == "qkv" {
            for (j, s) in subs.iter().enumerate() {
                targets.push((
                    format!("{base}.{s}"),
                    b_t.narrow(0, j * hidden, hidden)?.contiguous()?,
                ));
            }
        } else if let Some(s) = subs.first() {
            targets.push((format!("{base}.{s}"), b_t.clone()));
        }
        for (name, b_slice) in targets {
            tensors.insert(format!("{name}.lora_A.weight"), a_t.clone());
            tensors.insert(format!("{name}.lora_B.weight"), b_slice);
            tensors.insert(format!("{name}.alpha"), alpha.clone());
        }
    }
    crate::pipelines::atomic_safetensors_save(&tensors, out)?;
    Ok(())
}

/// Load a diffusers-PEFT LoRA checkpoint (written by [`save_peft_lora`]) back
/// into the live fused MMDiT adapters — the inverse of the save: the shared
/// `lora_A` is read once per fused projection and the per-q/k/v `lora_B` slices
/// are concatenated back into the fused B. Used by `--resume`. The slug mapping
/// mirrors `save_peft_lora` exactly (the round-trip test guards against drift).
fn load_peft_into_adapters(
    adapters: &[(String, candle_core::Var, candle_core::Var)],
    path: &std::path::Path,
    device: &Device,
) -> Result<()> {
    let loaded = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading resume checkpoint {}", path.display()))?;
    let get = |name: &str| -> Result<Tensor> {
        loaded
            .get(name)
            .ok_or_else(|| anyhow!("resume: checkpoint missing {name} (rank/base mismatch?)"))
            .cloned()
    };
    for (key, a, b) in adapters {
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() < 5 {
            continue;
        }
        let i: usize = match parts[1].parse() {
            Ok(i) => i,
            Err(_) => continue,
        };
        let is_ctx = parts[2] == "context_block";
        let kind = parts[4];
        let base = format!("transformer.transformer_blocks.{i}.attn");
        let subs: &[&str] = match (is_ctx, kind) {
            (false, "qkv") => &["to_q", "to_k", "to_v"],
            (true, "qkv") => &["add_q_proj", "add_k_proj", "add_v_proj"],
            (false, "proj") => &["to_out.0"],
            (true, "proj") => &["to_add_out"],
            _ => continue,
        };
        // Shared A: read once from the first sub. B: concat the per-sub slices
        // back into the fused adapter (inverse of save's narrow).
        let a_loaded = get(&format!("{base}.{}.lora_A.weight", subs[0]))?;
        let mut b_parts = Vec::with_capacity(subs.len());
        for s in subs {
            b_parts.push(get(&format!("{base}.{s}.lora_B.weight"))?);
        }
        let b_loaded = if b_parts.len() == 1 {
            b_parts.pop().unwrap()
        } else {
            Tensor::cat(&b_parts, 0)?
        };
        a.set(&a_loaded.to_dtype(a.as_tensor().dtype())?)?;
        b.set(&b_loaded.to_dtype(b.as_tensor().dtype())?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peft_lora_save_load_roundtrip() {
        // The fused-qkv split (save) ↔ concat (load) must round-trip, or --resume
        // silently corrupts the adapters. Verified on CPU, no training needed.
        let dev = Device::Cpu;
        let (rank, hidden, in_dim) = (2usize, 4usize, 6usize);
        let mk = |key: &str, b_rows: usize| -> (String, candle_core::Var, candle_core::Var) {
            let a = candle_core::Var::from_tensor(
                &Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap(),
            )
            .unwrap();
            let b = candle_core::Var::from_tensor(
                &Tensor::randn(0f32, 1f32, (b_rows, rank), &dev).unwrap(),
            )
            .unwrap();
            (key.to_string(), a, b)
        };
        let keys = || {
            vec![
                mk("joint_blocks.0.x_block.attn.qkv.weight", 3 * hidden),
                mk("joint_blocks.0.context_block.attn.proj.weight", hidden),
            ]
        };
        let src = keys();
        let tmp = std::env::temp_dir().join("plakat_sd3_peft_roundtrip.safetensors");
        save_peft_lora(&src, rank, hidden, &tmp).unwrap();
        let dst = keys(); // fresh adapters, overwritten by the load
        load_peft_into_adapters(&dst, &tmp, &dev).unwrap();
        let max_abs = |x: &candle_core::Var, y: &candle_core::Var| -> f32 {
            let d: Vec<f32> = (x.as_tensor() - y.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            d.iter().fold(0f32, |m, v| m.max(v.abs()))
        };
        for ((_, sa, sb), (_, da, db)) in src.iter().zip(dst.iter()) {
            assert!(max_abs(sa, da) < 1e-2, "A drift (F16 round-trip)");
            assert!(max_abs(sb, db) < 1e-2, "B drift (F16 round-trip)");
        }
        let _ = std::fs::remove_file(&tmp);
    }

    // v0.14 phase 1a — schedule transform.

    #[test]
    fn shift_t_identity_at_shift_one() {
        for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((shift_t(v, 1.0) - v).abs() < 1e-12);
        }
    }

    #[test]
    fn shift_t_endpoints_fixed() {
        // f(0) = 0, f(1) = 1 for any shift > 0.
        for shift in [1.0, 2.0, 3.0, 5.0, 10.0] {
            assert!((shift_t(0.0, shift) - 0.0).abs() < 1e-12);
            assert!((shift_t(1.0, shift) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn shift_t_compresses_low_end_with_high_shift() {
        // shift > 1: high-noise region gets more density. f(0.5, 3.0)
        // = 3*0.5 / (1 + 2*0.5) = 0.75 — the midpoint of the schedule
        // sits past 0.5 in t-space, meaning more steps cluster near 1.
        assert!((shift_t(0.5, 3.0) - 0.75).abs() < 1e-12);
    }

    // v0.15 phase 2 — img2img schedule truncation.

    #[test]
    fn schedule_unchanged_without_strength() {
        // Pure t2i: full linear schedule with shift_t applied,
        // length = steps + 1, endpoints 1.0 and 0.0.
        let ts = build_img2img_timesteps(4, 1.0, None);
        assert_eq!(ts.len(), 5);
        assert!((ts[0] - 1.0).abs() < 1e-12);
        assert!((ts[4] - 0.0).abs() < 1e-12);
        // shift = 1.0 is identity, so middle entries are linear.
        assert!((ts[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn schedule_truncated_to_strength() {
        // Strength = 0.5: drop the high-noise half and prepend 0.5
        // itself. So the schedule starts at 0.5, walks down, ends 0.0.
        let ts = build_img2img_timesteps(4, 1.0, Some(0.5));
        // Full was [1.0, 0.75, 0.5, 0.25, 0.0]. Filter < 0.5 → [0.25,
        // 0.0]. Prepend 0.5 → [0.5, 0.25, 0.0].
        assert_eq!(ts.len(), 3);
        assert!((ts[0] - 0.5).abs() < 1e-12);
        assert!((ts[1] - 0.25).abs() < 1e-12);
        assert!((ts[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn schedule_strength_zero_falls_back_to_terminal_step() {
        // Strength = 0.0 would filter out every entry; we fall back to
        // a single 0.0 step. Schedule becomes [0.0, 0.0] — a no-op
        // denoise window. The generate loop's `t_prev - t_curr` is
        // zero so no update happens. Safer than panicking on empty.
        let ts = build_img2img_timesteps(4, 1.0, Some(0.0));
        assert_eq!(ts.len(), 2);
        assert!((ts[0] - 0.0).abs() < 1e-12);
        assert!((ts[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn schedule_strength_one_keeps_all_but_one() {
        // Strength = 1.0: filter keeps every entry except the terminal
        // 1.0 itself (since the filter is `t < s`). Prepended 1.0
        // recovers the standard `[1.0, ..., 0.0]` shape — equivalent
        // to pure t2i except built differently.
        let ts = build_img2img_timesteps(4, 1.0, Some(1.0));
        // Full = [1.0, 0.75, 0.5, 0.25, 0.0]. Filter < 1.0 → [0.75,
        // 0.5, 0.25, 0.0]. Prepend 1.0 → [1.0, 0.75, 0.5, 0.25, 0.0].
        assert_eq!(ts.len(), 5);
        assert!((ts[0] - 1.0).abs() < 1e-12);
        assert!((ts[4] - 0.0).abs() < 1e-12);
    }

    // v0.15 phase 5 — tile-to-canvas pad helper.

    #[test]
    fn pad_tile_places_at_origin() {
        // Tile (1, 2, 2, 2) of ones, canvas 4x4. Place at (0, 0).
        // Top-left 2x2 of the canvas should be ones, rest zeros.
        let tile = Tensor::ones((1, 2, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let out = pad_tile_to_canvas(&tile, 0, 0, 4, 4, DType::F32, &Device::Cpu).unwrap();
        let (_b, c, h, w) = out.dims4().unwrap();
        assert_eq!((c, h, w), (2, 4, 4));
        // Channel 0 should have ones in the top-left 2x2 corner.
        let ch0 = out.i(0).unwrap().i(0).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(ch0[0][0], 1.0);
        assert_eq!(ch0[1][1], 1.0);
        assert_eq!(ch0[0][2], 0.0);
        assert_eq!(ch0[2][0], 0.0);
        assert_eq!(ch0[3][3], 0.0);
    }

    #[test]
    fn pad_tile_places_with_offset() {
        let tile = Tensor::ones((1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let out = pad_tile_to_canvas(&tile, 1, 2, 4, 5, DType::F32, &Device::Cpu).unwrap();
        let (_b, c, h, w) = out.dims4().unwrap();
        assert_eq!((c, h, w), (1, 4, 5));
        let ch = out.i(0).unwrap().i(0).unwrap().to_vec2::<f32>().unwrap();
        // The 2x2 ones should land at rows 1-2, cols 2-3.
        assert_eq!(ch[1][2], 1.0);
        assert_eq!(ch[2][3], 1.0);
        assert_eq!(ch[0][2], 0.0); // row above
        assert_eq!(ch[3][2], 0.0); // row below
        assert_eq!(ch[1][1], 0.0); // col left
        assert_eq!(ch[1][4], 0.0); // col right
    }

    #[test]
    fn pad_tile_full_canvas_is_identity() {
        // A tile exactly the canvas size should be returned unchanged.
        let tile = Tensor::randn(0f32, 1.0_f32, (1, 3, 4, 4), &Device::Cpu).unwrap();
        let out = pad_tile_to_canvas(&tile, 0, 0, 4, 4, DType::F32, &Device::Cpu).unwrap();
        let diff = (&tile - &out).unwrap().abs().unwrap().sum_all().unwrap();
        let d: f32 = diff.to_scalar().unwrap();
        assert!(d < 1e-5, "expected identity; got diff {d}");
    }

    #[test]
    fn schedule_respects_shift() {
        // With shift = 3.0, the schedule is non-linear; truncating at
        // a low strength keeps fewer entries because the shift packed
        // more density into the high-noise end.
        let full_steps_count =
            build_img2img_timesteps(8, 3.0, Some(0.5)).len();
        // Should still start at 0.5 and end at 0.0. The intermediate
        // count depends on how many shifted t's fell below 0.5.
        let ts = build_img2img_timesteps(8, 3.0, Some(0.5));
        assert!((ts[0] - 0.5).abs() < 1e-12);
        assert!((ts[ts.len() - 1] - 0.0).abs() < 1e-12);
        // Sanity: result is non-empty and well-formed.
        assert!(full_steps_count >= 2);
    }

    // v0.16 phase 3e — merge_residuals (multi-CN composition).

    fn cpu_tensor(v: &[f32]) -> Tensor {
        Tensor::from_slice(v, (v.len(),), &Device::Cpu).unwrap()
    }

    fn tensor_to_vec(t: &Tensor) -> Vec<f32> {
        t.to_vec1::<f32>().unwrap()
    }

    #[test]
    fn merge_residuals_none_acc_passes_new_through() {
        // First CN's residuals become the seed accumulator. The merge
        // is a no-op pass-through when nothing was there before.
        let new = vec![cpu_tensor(&[1.0, 2.0]), cpu_tensor(&[3.0, 4.0])];
        let merged = merge_residuals(None, new).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(tensor_to_vec(&merged[0]), vec![1.0, 2.0]);
        assert_eq!(tensor_to_vec(&merged[1]), vec![3.0, 4.0]);
    }

    #[test]
    fn merge_residuals_sums_same_length_lists() {
        // Two CNs producing equal-length residual stacks compose by
        // element-wise sum across each block index.
        let a = vec![cpu_tensor(&[1.0, 2.0]), cpu_tensor(&[3.0, 4.0])];
        let b = vec![cpu_tensor(&[10.0, 20.0]), cpu_tensor(&[30.0, 40.0])];
        let merged = merge_residuals(Some(a), b).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(tensor_to_vec(&merged[0]), vec![11.0, 22.0]);
        assert_eq!(tensor_to_vec(&merged[1]), vec![33.0, 44.0]);
    }

    #[test]
    fn merge_residuals_appends_when_new_is_longer() {
        // A longer new-list extends the accumulator. (Won't happen in
        // practice with sibling CNs targeting the same MMDiT — every
        // slot produces num_layers residuals — but the helper handles
        // the case for symmetry with the flux.rs sibling.)
        let a = vec![cpu_tensor(&[1.0])];
        let b = vec![cpu_tensor(&[2.0]), cpu_tensor(&[5.0])];
        let merged = merge_residuals(Some(a), b).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(tensor_to_vec(&merged[0]), vec![3.0]);
        assert_eq!(tensor_to_vec(&merged[1]), vec![5.0]);
    }

    #[test]
    fn merge_residuals_preserves_tail_when_new_is_shorter() {
        // Shorter new-list contributes to the first N entries; the
        // accumulator's tail passes through unchanged.
        let a = vec![cpu_tensor(&[1.0]), cpu_tensor(&[2.0]), cpu_tensor(&[3.0])];
        let b = vec![cpu_tensor(&[10.0])];
        let merged = merge_residuals(Some(a), b).unwrap();
        assert_eq!(merged.len(), 3);
        assert_eq!(tensor_to_vec(&merged[0]), vec![11.0]);
        assert_eq!(tensor_to_vec(&merged[1]), vec![2.0]);
        assert_eq!(tensor_to_vec(&merged[2]), vec![3.0]);
    }

    #[test]
    fn merge_residuals_handles_empty_new() {
        // Defensive: an empty new-list (no CN-residuals contributed
        // this step) leaves the accumulator untouched.
        let a = vec![cpu_tensor(&[1.0, 2.0])];
        let merged = merge_residuals(Some(a), Vec::new()).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(tensor_to_vec(&merged[0]), vec![1.0, 2.0]);
    }

    // v0.16 phase 10 — mode tag truth table for SD3
    // tiled / img2img / inpaint combinations.

    #[test]
    fn mode_tag_tiled_inpaint() {
        assert_eq!(sd3_mode_tag(true, true, true), "tiled inpaint");
    }

    #[test]
    fn mode_tag_tiled_img2img() {
        assert_eq!(sd3_mode_tag(true, true, false), "tiled img2img");
    }

    #[test]
    fn mode_tag_plain_tiled() {
        // Mask without init is ignored — img2img requires init too.
        assert_eq!(sd3_mode_tag(true, false, false), "tiled");
        assert_eq!(sd3_mode_tag(true, false, true), "tiled");
    }

    #[test]
    fn mode_tag_non_tiled_inpaint() {
        assert_eq!(sd3_mode_tag(false, true, true), "inpaint");
    }

    #[test]
    fn mode_tag_non_tiled_img2img() {
        assert_eq!(sd3_mode_tag(false, true, false), "img2img");
    }

    #[test]
    fn mode_tag_plain_denoise() {
        assert_eq!(sd3_mode_tag(false, false, false), "denoise");
        assert_eq!(sd3_mode_tag(false, false, true), "denoise");
    }
}

