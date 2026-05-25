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
use candle_transformers::models::{
    stable_diffusion::clip as sdclip, stable_diffusion::vae as sdvae, t5,
};
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
    /// v0.16 phase 3e: SD3 ControlNet stack to load. Each entry
    /// carries the InstantX repo + per-instance runtime knobs
    /// (`scale`, `conditioning`, `start`, `end`). Empty Vec means no
    /// CN — byte-identical to the pre-phase-3 schedule. Threaded
    /// through to `LoadRequest.controlnets`, then VAE-encoded once
    /// per `generate` call into the cached per-slot conditioning
    /// latents used in `predict_velocity_full`.
    pub controlnets: Vec<crate::pipelines::sd3_controlnet::Sd3ControlNetLoad>,
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
    /// v0.16 phase 3: per-call conditioning override for the loaded
    /// SD3 ControlNets. Indexed parallel to
    /// `LoadRequest::controlnets`. An entry of `None` keeps the path
    /// from the load request (used when one scenario task has no CN
    /// conditioning to swap to). Empty Vec = preserve all paths.
    pub controlnet_conditioning: Vec<Option<PathBuf>>,
}

pub struct Pipeline {
    pub variant: Variant,
    #[allow(dead_code)]
    pub repo: String,
    device: Device,
    dtype: DType,
    clip_l: sdclip::ClipTextTransformer,
    clip_l_tok: Tokenizer,
    clip_l_cfg: sdclip::Config,
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
        let clip_l_cfg = sdclip::Config::sdxl(); // SDXL CLIP-L = SD3 CLIP-L (77 tokens, 768d, 12 layers)
        let clip_l = candle_transformers::models::stable_diffusion::build_clip_transformer(
            &clip_l_cfg,
            &clip_l_w,
            &req.device,
            dtype,
        )?;
        let clip_l_tok =
            Tokenizer::from_file(&clip_l_tok_path).map_err(|e| anyhow!("CLIP-L tokenizer: {e}"))?;

        // ---------- CLIP-G (with text_projection for pooled) ----------
        let clip_g_cfg = sdclip::Config::sdxl2(); // SDXL CLIP-G = SD3 CLIP-G (77 tokens, 1280d, 32 layers)
        let clip_g_vs = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&clip_g_w], dtype, &req.device)?
        };
        let clip_g = SdxlClipGTextTransformer::new(clip_g_vs, &clip_g_cfg, 1280)?;
        let clip_g_tok =
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
        let t5_tok =
            Tokenizer::from_file(&t5_tok_json).map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
        build.finish_with_message("✓ text encoders ready");

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
        let mmdit_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[effective_mmdit_path], dtype, &req.device,
            )?
        };
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

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random)
                & (u32::MAX as u64);
            if let Err(e) = self.device.set_seed(seed) {
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
                let pred = match req.tiled.as_ref() {
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
                .join(format!("plakat-sd3-{mode_tag}-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
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
        let t_vec = Tensor::full(t_curr as f32, 2, &self.device)?;

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

    fn encode_prompt(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
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

        // ---------- T5 forward ----------
        let mut t5_hidden = self.t5_enc.forward(&t5_ids_t)?.to_dtype(self.dtype)?;
        if let Some(w) = &t5_weights {
            t5_hidden = t5_hidden.broadcast_mul(&w.to_dtype(self.dtype)?)?;
        }

        // ---------- Pooled (y) ----------
        // SD3 convention: CLIP-G pooled first (1280), CLIP-L pooled
        // second (768) → (1, 2048).
        let y = Tensor::cat(&[&clip_g_pooled, &clip_l_pooled], candle_core::D::Minus1)?;

        // ---------- CLIP-L penultimate (weighted if has_attn) ----------
        // CLIP-L's penultimate hidden state is what SD3 mixes with
        // CLIP-G's penultimate. We grab CLIP-L penultimate by running
        // until layer -2 (matching SDXL's convention).
        let (_clip_l_final, clip_l_penult) = {
            let (final_h, pen_h) = candle_transformers::models::stable_diffusion::clip::ClipTextTransformer
                ::forward_until_encoder_layer(&self.clip_l, &clip_l_ids_t, usize::MAX, -2)?;
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
        controlnet_conditioning: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
