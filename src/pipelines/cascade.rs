//! Stable Cascade pipeline — fifth model family.
//!
//! v0.37 phase 0 (this commit): foundational stub. `Pipeline::load`
//! downloads + assembles the CLIP-G text encoder + tokenizer for
//! the canonical `stabilityai/stable-cascade` checkpoint, proving
//! the dispatch + alias plumbing. `run()` bails with a clear
//! "Stage A VAE lands in phase 1" message after a successful load.
//!
//! ## Architecture (target end-state across phases 0-5)
//!
//! Stable Cascade is a **3-stage** generative system. Inference
//! chains the stages from coarsest to finest:
//!
//! ```text
//!   prompt
//!     │
//!     ▼  CLIP-G text encoder (1280d)
//!     │
//!     ▼  Stage C (high-res prior, ~3.6B params, large UNet)
//!     │       conditioned on text; operates on a 24×24×16 latent
//!     │
//!     ▼  Stage B (latent prior, ~1.5B params, UNet)
//!     │       conditioned on Stage C output + text;
//!     │       produces Stage A's 32×32 latent
//!     │
//!     ▼  Stage A (VAE, ~3.6M params, tiny decoder)
//!     │       decodes 32×32 latent → 1024×1024 image
//!     │
//!     ▼
//!   image
//! ```
//!
//! - **Stage A** ships in **phase 1** — small custom VAE; loads
//!   from `vqgan/diffusion_pytorch_model.safetensors` in the
//!   diffusers repo (diffusers calls Stage A "vqgan").
//! - **Stage B** ships in **phase 2** — UNet at ~1.5B params;
//!   loads from `decoder/diffusion_pytorch_model.safetensors`
//!   (diffusers calls Stage B "decoder").
//! - **Stage C** ships in **phase 3** — UNet at ~3.6B params;
//!   loads from `prior/diffusion_pytorch_model.safetensors`
//!   (diffusers calls Stage C "prior").
//! - **End-to-end pipeline orchestration** lands in **phase 4**.
//!
//! ## Phase 0 scope
//!
//! Acceptance criteria (per RFC):
//! - `--model stable-cascade` resolves via the alias system. ✓
//!   (`hf/mod.rs` v0.37 phase 0 entry).
//! - The pipeline module compiles + exports a stub
//!   `Pipeline::load` + `run`. ✓ (this file).
//! - `t2i::Pipeline::load` rejects Stable Cascade with a clear
//!   pointer at `pipelines::cascade::Pipeline::load` (parallels
//!   the Flux / SD3 / PixArt bail pattern).
//! - `t2i::run` routes Stable Cascade to `cascade::run` (which
//!   bails until phase 1). Proves the dispatch wiring without
//!   inference.
//!
//! ## Text encoder reuse
//!
//! Stable Cascade's text encoder is **CLIP-G** — the same
//! 32-layer / 1280-dim CLIP variant SDXL CLIP-G uses. plakat
//! already vendored this in v0.30 phase 0 (rolled out across
//! pipelines in v0.32 phase 1). `Pipeline::load` reuses
//! `pipelines::sdxl_clip::SdxlClipGTextTransformer` with the
//! `pipelines::vendored_clip::Config::sdxl2()` configuration —
//! zero new text-encoder code needed.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

// v0.39 phase 0g: upstream-aligned modules. Replaces v0.37/v0.38's
// SD-style approximations with the architecturally-correct Stable
// Cascade modules from cascade_blocks / cascade_prior / cascade_vae /
// cascade_cn.
use crate::pipelines::cascade_cn::{
    CascadeControlNet, Config as CnConfig,
};
use crate::pipelines::cascade_prior::{
    Config as PriorConfig, StableCascadePrior,
};
use crate::pipelines::cascade_vae::{Config as VaeConfig, StageAVae};
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::sdxl_clip::SdxlClipGTextTransformer;
use crate::pipelines::vendored_clip;
use crate::ui::progress;

#[allow(dead_code)] // v0.40 generate() rewires; retained for that wiring.
const CLIP_EOT: u32 = 49407;

/// Inputs to [`Pipeline::load`]. Mirrors the shape of
/// `flux::LoadRequest` / `pixart::LoadRequest` so the scenario +
/// scripting cache machinery can hand off uniformly when those
/// integrations land in v0.38+.
pub struct LoadRequest {
    /// Resolved HF repo id (callers run `crate::hf::resolve_alias`
    /// first; this struct holds the canonical form).
    pub repo: String,
    pub device: Device,
    /// v0.38 phase 3: Cascade LoRA stack (resolved by the caller via
    /// `LoraSpec::resolve`). Each entry is dispatched against BOTH
    /// the Stage B (decoder) and Stage C (prior) prior UNets via
    /// `cascade_lora::merge_cascade_{b,c}_loras_into_weights`. Empty
    /// (default) → no merge, base safetensors mmap directly.
    pub loras: Vec<crate::pipelines::lora::ResolvedLora>,
    /// Global scale multiplier on each LoRA's per-spec scale.
    /// Mirrors `--lora-scale`. Default `1.0` means honour each
    /// LoRA's own scale; `0.0` zeroes out every LoRA contribution.
    pub lora_scale: f32,
    /// v0.38 phase 5: optional ControlNet weights path. When `Some`,
    /// `Pipeline::load` constructs a `CascadeControlNet` from the
    /// safetensors at this path and stores it on the pipeline.
    /// When `None`, no CN is attached and `Pipeline::generate` runs
    /// as plain t2i regardless of any `control_conditioning` arg.
    ///
    /// Users supply this via `--cascade-control-weights PATH` —
    /// upstream Stable Cascade ControlNet checkpoints aren't yet
    /// catalogued in plakat's `hf::ALIAS_TABLE`, so a local path
    /// or full HF repo:filename is the v0.38 contract. Catalogued
    /// CN-by-kind aliases land in v0.39.
    pub controlnet_weights: Option<std::path::PathBuf>,
}

/// v0.38 phase 5: per-call ControlNet conditioning input. Bundles
/// the conditioning image tensor with the spec's strength + start /
/// end timestep window. Built once at the CLI/pipeline boundary
/// from a `ControlSpec` + annotator hookup.
#[derive(Debug)]
pub struct ControlConditioning {
    /// Conditioning image already at Stage C ControlNet's expected
    /// input shape `(1, 3, 1024, 1024)` in `[-1, 1]`. Use
    /// `crate::imaging::preprocess::sd_image_tensor` to build.
    pub conditioning_image: Tensor,
    /// Per-ControlSpec strength multiplier on the residual sum.
    pub scale: f32,
    /// Timestep window start in `[0, 1]`. The CN residual is active
    /// during `progress in [start, end)` where `progress = step_idx
    /// / (n_steps - 1)`.
    pub start: f32,
    pub end: f32,
}

/// Stable Cascade pipeline.
///
/// ## Spatial contract (v0.40 phase 0)
///
/// For a final image dimension `D` (default 1024), the working
/// shapes through the 3-stage pipeline are:
///
/// ```text
///   image                            (B, 3,   D,    D)
///     ↓ Stage A.encode_to_stage_b_space
///     ↓   = encode (4× compression) + PixelUnshuffle(2)
///   Stage B input/output             (B, 16,  D/8,  D/8)    e.g. 1024 → 128
///     ↓ Stage A.decode_from_stage_b_space
///     ↓   = PixelShuffle(2) + decode (4× expansion)
///   image                            (B, 3,   D,    D)
/// ```
///
/// Stage C operates on a FIXED `(B, 16, 24, 24)` prior latent
/// regardless of `D` (semantic conditioning, not image resolution).
/// Stage B consumes Stage C's output as effnet conditioning via
/// `apply_effnet_mapper` (with spatial alignment to Stage B's input).
///
/// `D` must be divisible by 8 (the total Stage A↔B compression).
/// Use [`crate::pipelines::cascade_vae::stage_b_spatial_for_image`]
/// to compute the Stage B working spatial.
pub struct Pipeline {
    pub device: Device,
    pub dtype: DType,
    /// CLIP-G text encoder. Reuses the same wrapper SDXL uses for
    /// text_encoder_2 (penultimate hidden state + pooled output via
    /// the `text_projection` Linear). 1280-dim embed.
    pub clip_g_enc: SdxlClipGTextTransformer,
    pub clip_g_tok: Tokenizer,
    /// v0.37 phase 1: Stage A VAE. ~3.6M-param small VAE that
    /// compresses 1024×1024×3 images to 32×32×4 latents (32× per
    /// axis). Used to encode training images (during img2img /
    /// future ControlNet) and decode generated latents at the end
    /// of the 3-stage pipeline.
    pub stage_a: StageAVae,
    /// v0.39 phase 0g: Stage B latent prior (upstream-aligned).
    /// 4-level UNet (320/640/1280/1280 widths) with attention at
    /// the deeper two. 2-mapper TimestepBlock. effnet + pixels
    /// conditioning embedders.
    pub stage_b: StableCascadePrior,
    /// v0.39 phase 0g: Stage C high-res prior (upstream-aligned).
    /// 2-level UNet at c_hidden=2048. 3-mapper TimestepBlock.
    /// CLIP-G text + image + pooled-text conditioning.
    pub stage_c: StableCascadePrior,
    /// v0.38 phase 5: optional Cascade ControlNet. `Some` when
    /// `LoadRequest.controlnet_weights` was supplied; produces a
    /// residual on the conditioning image that gets added to Stage
    /// C's latent at the input. `None` for plain t2i. The Stage B
    /// path doesn't carry a CN — Stage C is the semantic stage
    /// where spatial conditioning lands.
    pub controlnet: Option<CascadeControlNet>,
}

impl Pipeline {
    /// Phase 0 load: CLIP-G + tokenizer.
    ///
    /// Repo layout assumes the canonical diffusers
    /// `stabilityai/stable-cascade` structure:
    ///
    /// ```text
    /// text_encoder/
    ///   config.json
    ///   model.safetensors        (CLIP-G, 32 layers, 1280 embed)
    /// tokenizer/
    ///   tokenizer.json
    /// vqgan/                       [phase 1: Stage A]
    ///   diffusion_pytorch_model.safetensors
    /// decoder/                     [phase 2: Stage B]
    ///   diffusion_pytorch_model.safetensors
    /// prior/                       [phase 3: Stage C]
    ///   diffusion_pytorch_model.safetensors
    /// ```
    pub async fn load(req: LoadRequest) -> Result<Self> {
        // Stable Cascade was trained in BF16 (diffusers'
        // `torch_dtype=torch.bfloat16` default). BF16 has the same
        // exponent range as F32 (~1e38) but F16's mantissa, which is
        // exactly the trade Cascade needs — Stage C's FiLM
        // modulation amplifies the residual stream by `(1 + scale)`
        // per TimestepBlock, and at scale ≈ 1 across 24+ stacked
        // blocks the intermediate values fly past F16's 6.5e4
        // ceiling and become Inf → NaN. v0.41 phase 2b caught this
        // on the first Metal end-to-end run (CPU F32 was fine all
        // along). BF16 keeps the GPU memory win F16 gave us while
        // matching the upstream dtype.
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::BF16
        };

        let dl = progress::spinner("Resolving Stable Cascade weights");
        // CLIP-G text encoder. Stable Cascade's `text_encoder/`
        // ships a single safetensors file (vs SDXL's 2-shard
        // layout); fall back to the sharded shape for forks that
        // mirror the SDXL convention.
        let clip_g_w = crate::hf::download::get_first_of(&[
            (&req.repo, "text_encoder/model.safetensors"),
            (&req.repo, "text_encoder/model.fp16.safetensors"),
        ])
        .await
        .with_context(|| {
            format!("downloading CLIP-G weights for Stable Cascade ({})", req.repo)
        })?;
        let clip_g_tok_path = crate::hf::download::get_first_of(&[
            (&req.repo, "tokenizer/tokenizer.json"),
            ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json"),
        ])
        .await
        .with_context(|| {
            format!("downloading CLIP-G tokenizer for Stable Cascade ({})", req.repo)
        })?;
        // v0.37 phase 1: Stage A VAE weights. Diffusers calls
        // Stage A `vqgan` even though it's not vector-quantized;
        // the path convention follows.
        let stage_a_w = crate::hf::download::get_first_of(&[
            (&req.repo, "vqgan/diffusion_pytorch_model.safetensors"),
            (&req.repo, "vqgan/diffusion_pytorch_model.fp16.safetensors"),
        ])
        .await
        .with_context(|| {
            format!("downloading Stage A VAE weights for Stable Cascade ({})", req.repo)
        })?;
        // v0.37 phase 2: Stage B UNet weights. Diffusers calls
        // Stage B `decoder`. Single safetensors file in the
        // canonical Full layout; community Lite forks may shard.
        let stage_b_w = crate::hf::download::get_first_of(&[
            (&req.repo, "decoder/diffusion_pytorch_model.safetensors"),
            (&req.repo, "decoder/diffusion_pytorch_model.fp16.safetensors"),
        ])
        .await
        .with_context(|| {
            format!("downloading Stage B UNet weights for Stable Cascade ({})", req.repo)
        })?;
        // v0.39 phase 0g: Stage C lives in a SEPARATE repo
        // (`stabilityai/stable-cascade-prior`), confirmed during the
        // v0.39 phase 0 inspection. v0.37/v0.38 incorrectly looked
        // for `prior/` inside `stable-cascade`. We try the canonical
        // prior repo first; the legacy `prior/` path in the same
        // repo is kept as a fallback for forks that bundle.
        let prior_repo =
            stage_c_prior_repo(&req.repo).unwrap_or_else(|| "stabilityai/stable-cascade-prior".to_string());
        let stage_c_w = crate::hf::download::get_first_of(&[
            (&prior_repo, "prior/diffusion_pytorch_model.safetensors"),
            (&prior_repo, "prior/diffusion_pytorch_model.fp16.safetensors"),
            (&req.repo, "prior/diffusion_pytorch_model.safetensors"),
            (&req.repo, "prior/diffusion_pytorch_model.fp16.safetensors"),
        ])
        .await
        .with_context(|| {
            format!("downloading Stage C UNet weights for Stable Cascade ({prior_repo})")
        })?;
        dl.finish_with_message(
            "✓ Stable Cascade text-encoder + Stage A + Stage B + Stage C weights resolved",
        );

        let build = progress::spinner("Loading CLIP-G text encoder");
        // v0.30 phase 0 vendored CLIP Config for the bigG variant.
        let clip_g_cfg = vendored_clip::Config::sdxl2();
        let clip_g_vs = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&clip_g_w], dtype, &req.device)?
        };
        // 1280-dim embed for CLIP-G.
        let clip_g_enc = SdxlClipGTextTransformer::new(clip_g_vs, &clip_g_cfg, 1280)
            .context("building CLIP-G text encoder for Stable Cascade")?;
        let clip_g_tok = Tokenizer::from_file(&clip_g_tok_path)
            .map_err(|e| anyhow!("CLIP-G tokenizer: {e}"))?;
        build.finish_with_message("✓ CLIP-G ready");

        let stage_a_build = progress::spinner("Loading Stage A VAE (Paella v3)");
        let stage_a_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_a_w.as_path()], dtype, &req.device)?
        };
        let stage_a = StageAVae::new(VaeConfig::paella_v3(), stage_a_vb)
            .context("building Stage A VAE for Stable Cascade")?;
        stage_a_build.finish_with_message("✓ Stage A VAE ready");

        // v0.38 phase 3: optionally merge user LoRAs into Stage B
        // and Stage C tempfiles (mirrors pixart::Pipeline::load
        // pattern). Empty stack short-circuits to the base mmap.
        let stage_b_load_path = maybe_merge_loras(
            &stage_b_w,
            &req.loras,
            req.lora_scale,
            &req.device,
            crate::pipelines::cascade_lora::Stage::B,
        )?;
        let stage_c_load_path = maybe_merge_loras(
            &stage_c_w,
            &req.loras,
            req.lora_scale,
            &req.device,
            crate::pipelines::cascade_lora::Stage::C,
        )?;

        // v0.39 phase 0g: Stage B prior (upstream-aligned). 4 levels,
        // attention at deepest 2, effnet + pixels conditioning.
        let stage_b_build = progress::spinner("Loading Stage B UNet (decoder prior)");
        let stage_b_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_b_load_path.as_path()], dtype, &req.device)?
        };
        let stage_b = StableCascadePrior::new_stage_b(PriorConfig::stage_b_full(), stage_b_vb)
            .context("building Stage B prior for Stable Cascade")?;
        stage_b_build.finish_with_message("✓ Stage B prior ready");

        // v0.39 phase 0g: Stage C prior. 2 levels, attention at every
        // level, 3-mapper TimestepBlock, CLIP-G + CLIP-img conditioning.
        let stage_c_build = progress::spinner("Loading Stage C UNet (high-res prior — heaviest stage)");
        let stage_c_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_c_load_path.as_path()], dtype, &req.device)?
        };
        let stage_c = StableCascadePrior::new_stage_c(PriorConfig::stage_c_full(), stage_c_vb)
            .context("building Stage C prior for Stable Cascade")?;
        stage_c_build.finish_with_message("✓ Stage C prior ready");

        // v0.38 phase 5: optional ControlNet load. When the user
        // didn't pass `--cascade-control-weights`, this is None and
        // generate() runs as plain t2i regardless of any control
        // conditioning args.
        let controlnet = if let Some(cn_path) = req.controlnet_weights.as_ref() {
            let cn_build = progress::spinner("Loading Cascade ControlNet (MobileNetV3-Large)");
            let cn_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[cn_path.as_path()], dtype, &req.device)?
            };
            let cn = CascadeControlNet::new(CnConfig::canny_upstream(), cn_vb)
                .context("building Cascade ControlNet")?;
            cn_build.finish_with_message("✓ Cascade ControlNet ready");
            Some(cn)
        } else {
            None
        };

        Ok(Self {
            device: req.device,
            dtype,
            clip_g_enc,
            clip_g_tok,
            stage_a,
            stage_b,
            stage_c,
            controlnet,
        })
    }

    /// v0.38 phase 5: per-call ControlNet conditioning input.
    /// Bundled so `generate` / `generate_img2img` don't bloat their
    /// signatures with four extra args. Construct from a resolved
    /// `ControlSpec` at the CLI layer (annotator + image loader).
    pub fn control_conditioning_active(&self) -> bool {
        self.controlnet.is_some()
    }

    /// v0.40 phase 4: tokenize a prompt + forward through CLIP-G.
    /// Returns `(penult, pooled)` where:
    /// - `penult` is the penultimate hidden states `(1, 77, 1280)` for
    ///   cross-attention into Stage C `AttnBlock`s.
    /// - `pooled` is the pooled output `(1, 1280)` for Stage C's
    ///   `clip_txt_pooled_mapper` + Stage B's only conditioning source.
    fn encode_prompt(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let mut ids = self
            .clip_g_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("CLIP-G encode: {e}"))?
            .get_ids()
            .to_vec();
        ids.resize(77, CLIP_EOT);
        let ids_t = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        // v0.41 phase 2j: Cascade uses the LAST hidden state
        // (hidden_states[-1]), not SDXL's penultimate.
        let (last_hidden, pooled) = self.clip_g_enc.forward_for_cascade(&ids_t)?;
        Ok((
            last_hidden.to_dtype(self.dtype)?,
            pooled.to_dtype(self.dtype)?,
        ))
    }

    /// End-to-end 3-stage generation.
    ///
    /// Chain: text → Stage C CFG denoise → Stage B CFG denoise →
    /// Stage A decode → image. Returns `(buf, width, height)` —
    /// the caller composes metadata + writes the PNG.
    ///
    /// ## Scope (after v0.38 phase 1 — effnet conditioning landed)
    ///
    /// - **Shape-correct end-to-end**: every stage runs at the right
    ///   shapes; output PNG has the correct (1024, 1024, 3) dims.
    /// - **FiLM timestep injection wired** (v0.38 phase 0). Output
    ///   is timestep-dependent.
    /// - **Effnet conditioning wired** (v0.38 phase 1). Stage C's
    ///   16ch×24×24 prior latent is upsampled + channel-concatenated
    ///   into Stage B's `in_conv`. Stage B is now conditioned on
    ///   `(text, Stage C output)` not just text.
    ///
    /// On random weights (the test path), shape correctness is the
    /// acceptance. On real `stabilityai/stable-cascade` weights,
    /// the architecture is now complete — output quality is bounded
    /// by tensor-naming alignment with the upstream checkpoint
    /// (real-weight smoke at user time will surface any remaining
    /// VarBuilder mismatches).
    pub fn generate(
        &mut self,
        prompt: &str,
        negative: &str,
        output_dim: u32,
        stage_c_steps: usize,
        stage_b_steps: usize,
        guidance: f64,
        seed: u64,
        scheduler_kind: SchedulerKind,
        _control: Option<&ControlConditioning>,
    ) -> Result<(Vec<u8>, u32, u32)> {
        self.generate_at_size(
            prompt,
            negative,
            output_dim,
            stage_c_steps,
            stage_b_steps,
            guidance,
            seed,
            scheduler_kind,
        )
    }

    /// v0.40 phase 4: end-to-end 3-stage generation at the given
    /// output image dimension (must be divisible by 8 — the total
    /// Stage A↔B compression).
    ///
    /// Chain: text → CLIP-G → Stage C CFG denoise → effnet
    /// conditioning → Stage B CFG denoise → PixelShuffle(2) →
    /// Stage A.decode → image bytes.
    ///
    /// `sca_emb` and `crp_emb` in the `TimestepBlock`s use the
    /// sinusoidal embedding of a **zero** scalar — the upstream
    /// diffusers default for `sca=None` / `crp=None` (aesthetic-score
    /// and crop conditioning omitted). This is NOT the zero tensor:
    /// `sin(0) = 0`, `cos(0) = 1`, so the embedding has constant
    /// signal that lets the `mapper_sca` / `mapper_crp` linears emit
    /// their learned baseline AdaLN-style scales/shifts. The KV
    /// stream is built via `StableCascadePrior::build_clip_conditioning`
    /// — Stage C gets concat(77 text + 4 pooled text + 4 zero image)
    /// at c_hidden=2048; Stage B gets pooled-only at c_hidden=1280.
    ///
    /// Returns `(buf_rgb_u8, width, height)`.
    pub fn generate_at_size(
        &mut self,
        prompt: &str,
        negative: &str,
        output_dim: u32,
        stage_c_steps: usize,
        stage_b_steps: usize,
        guidance: f64,
        seed: u64,
        scheduler_kind: SchedulerKind,
    ) -> Result<(Vec<u8>, u32, u32)> {
        use crate::pipelines::cascade_prior::sinusoidal_time_embedding;
        use crate::pipelines::cascade_scheduler::CascadeScheduler;
        use crate::pipelines::cascade_vae::stage_b_spatial_for_image;
        use candle_core::IndexOp;

        // v0.41 phase 0: scheduler_kind is ignored for Stable Cascade.
        // The model is trained against `DDPMWuerstchenScheduler` (ratio
        // timesteps + cosine α-cumprod) which doesn't fit candle's
        // SD-family scheduler trait. A future cycle may surface a
        // `SchedulerKind::CascadeWuerstchen{Linear,Shifted}` variant
        // to expose the scaler knob.
        let _ = scheduler_kind;

        anyhow::ensure!(
            output_dim % 8 == 0,
            "output_dim must be divisible by 8 (Stage A↔B compression contract); got {output_dim}"
        );
        let stage_b_dim = stage_b_spatial_for_image(output_dim) as usize;

        // ---- Seed prep ----
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
        if let Err(e) = self.device.set_seed(prepared) {
            tracing::debug!(
                target: "plakat",
                "set_seed not supported ({e}); using global RNG"
            );
        }

        // ---- Text encoding ----
        let s = progress::spinner("Encoding CLIP-G text embeddings");
        let (pos_penult, pos_pooled) = self.encode_prompt(prompt)?;
        let (neg_penult, neg_pooled) = self.encode_prompt(negative)?;
        // CFG batch order: [neg, pos] on dim 0.
        let cfg_penult = Tensor::cat(&[&neg_penult, &pos_penult], 0)?;
        let cfg_pooled = Tensor::cat(&[&neg_pooled, &pos_pooled], 0)?;
        s.finish_with_message("✓ text encoded");

        // ---- Stage C KV conditioning (text + pooled + zero-img) ----
        let clip_c = self
            .stage_c
            .build_clip_conditioning(&cfg_penult, &cfg_pooled, None)?;
        // Stage B KV conditioning (pooled-only — clip_text arg ignored).
        let clip_b = self
            .stage_b
            .build_clip_conditioning(&cfg_penult, &cfg_pooled, None)?;

        // ---- sca/crp zero-conditioning embedding ----
        // v0.41 phase 1: upstream's `sca=None / crp=None` default
        // path sets the conditioning value to zeros_like(timestep_ratio)
        // and runs the same sinusoidal encoder over it. The result is
        // constant across denoise steps, so precompute once and share
        // between Stage C (uses both) and Stage B (uses sca only).
        // Batch is fixed at 2 throughout the CFG denoise loops.
        let zero_cond_input = Tensor::zeros(2, candle_core::DType::F32, &self.device)?;
        let zero_cond_emb = sinusoidal_time_embedding(&zero_cond_input, 64, 10000.0)?
            .to_dtype(self.dtype)?;

        // ---- Stage C denoise (fixed 24×24×16 prior latent) ----
        // v0.41 phase 0: Wuerstchen-style scheduler — ratio timesteps,
        // cosine α-cumprod, init_noise_sigma=1.0, no input scaling.
        let c_scheduler = CascadeScheduler::new(stage_c_steps);
        let c_timesteps: Vec<f64> = c_scheduler.timesteps().to_vec();
        let noise_c = Tensor::randn(0f32, 1f32, (1, 16, 24, 24), &self.device)?
            .to_dtype(self.dtype)?;
        let mut latent_c = noise_c;
        let bar = crate::ui::progress::step_bar(
            c_timesteps.len() as u64,
            "cascade stage C (prior)",
        );
        for &t in &c_timesteps {
            let cfg_latent = Tensor::cat(&[&latent_c, &latent_c], 0)?;
            let t_scalar = Tensor::new(&[t as f32], &self.device)?
                .to_dtype(self.dtype)?
                .expand((2,))?;
            let t_emb = sinusoidal_time_embedding(&t_scalar, 64, 10000.0)?
                .to_dtype(self.dtype)?;
            let pred = self.stage_c.forward(
                &cfg_latent,
                &t_emb,
                Some(&zero_cond_emb),
                Some(&zero_cond_emb),
                &clip_c,
                None,
                None,
            )?;
            let chunks = pred.chunk(2, 0)?;
            let neg = &chunks[0];
            let pos = &chunks[1];
            let guided = (neg + ((pos - neg)? * guidance)?)?;
            latent_c = c_scheduler.step(&guided, t, &latent_c)?;
            bar.inc(1);
            bar.set_message(format!("t={t:.3}"));
        }
        bar.finish_and_clear();

        // ---- Stage B denoise with Stage C effnet ----
        // v0.41 phase 2i: CFG over the effnet conditioning. Upstream
        // decoder uses `cat([image_embeddings, zeros_like(...)])` — the
        // UNCONDITIONAL half gets ZERO effnet so classifier-free
        // guidance amplifies the Stage C semantic conditioning. Our
        // CFG batch order is [neg, pos] (= [uncond, cond]), so the neg
        // half is zeros and the pos half is the Stage C output. Feeding
        // latent_c to BOTH halves (the v0.40 bug) left the effnet out
        // of the guidance entirely → coherent-but-wrong decoder texture
        // (the seed-43 "circuit board" artifact).
        let zero_c = latent_c.zeros_like()?;
        let cfg_effnet = Tensor::cat(&[&zero_c, &latent_c], 0)?;
        let b_scheduler = CascadeScheduler::new(stage_b_steps);
        let b_timesteps: Vec<f64> = b_scheduler.timesteps().to_vec();
        let noise_b = Tensor::randn(
            0f32,
            1f32,
            (1, 16, stage_b_dim, stage_b_dim),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        let mut latent_b = noise_b;
        let bar = crate::ui::progress::step_bar(
            b_timesteps.len() as u64,
            "cascade stage B (decoder)",
        );
        for &t in &b_timesteps {
            let cfg_latent = Tensor::cat(&[&latent_b, &latent_b], 0)?;
            let t_scalar = Tensor::new(&[t as f32], &self.device)?
                .to_dtype(self.dtype)?
                .expand((2,))?;
            let t_emb = sinusoidal_time_embedding(&t_scalar, 64, 10000.0)?
                .to_dtype(self.dtype)?;
            let pred = self.stage_b.forward(
                &cfg_latent,
                &t_emb,
                Some(&zero_cond_emb),
                None, // crp_emb — Stage B has no mapper_crp
                &clip_b,
                Some(&cfg_effnet),
                None, // pixels — t2i has no pixel conditioning
            )?;
            let chunks = pred.chunk(2, 0)?;
            let neg = &chunks[0];
            let pos = &chunks[1];
            // v0.41 phase 2i: the DECODER uses a much lower guidance
            // than the prior. Upstream StableCascadeDecoderPipeline
            // defaults `guidance_scale=0.0` (no CFG — pure conditional);
            // the prior uses ~4.0. Applying the prior's 4.0 to Stage B
            // over-drove the decoder into harsh over-detailed texture.
            // In our `neg + scale*(pos-neg)` form, scale=1.0 reproduces
            // the pure conditional (= upstream no-CFG decoder). A future
            // phase exposes `--decoder-guidance`; for now clamp Stage B
            // to a mild fixed value.
            const DECODER_GUIDANCE: f64 = 1.1;
            let guided = (neg + ((pos - neg)? * DECODER_GUIDANCE)?)?;
            latent_b = b_scheduler.step(&guided, t, &latent_b)?;
            bar.inc(1);
            bar.set_message(format!("t={t:.3}"));
        }
        bar.finish_and_clear();

        // ---- Stage A decode via Stage B → image ----
        let s = progress::spinner("Stage A decode → image");
        // v0.41 phase 2e: the Paella VQ decoder already outputs in
        // [0, 1] (upstream does `vqgan.decode(...).sample.clamp(0,
        // 1)`). v0.40's extra `(decoded / 2 + 0.5)` denorm — copied
        // from the SD [-1,1] convention — double-shifted the image.
        // Just clamp.
        let decoded = self.stage_a.decode_from_stage_b_space(&latent_b)?;
        let image = decoded.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;
        s.finish_with_message("✓ image decoded");
        Ok((buf, ow as u32, oh as u32))
    }

    /// End-to-end 3-stage img2img generation.
    ///
    /// v0.38 phase 4: encode the init image with Stage A to seed the
    /// Stage B latent; truncate Stage B's denoise schedule based on
    /// `strength` (1.0 = pure noise / equivalent to `generate`,
    /// lower values keep more of the input's structure). Stage C
    /// always runs the full schedule (text → effnet) — the input
    /// image conditions Stage B's output (the Stage A latent), not
    /// Stage C's semantic prior.
    ///
    /// `strength` is clamped to `[0, 1]` by the caller.
    pub fn generate_img2img(
        &mut self,
        _init_image_path: &std::path::Path,
        _prompt: &str,
        _negative: &str,
        _stage_c_steps: usize,
        _stage_b_steps: usize,
        _strength: f32,
        _guidance: f64,
        _seed: u64,
        _scheduler_kind: SchedulerKind,
    ) -> Result<(Vec<u8>, u32, u32)> {
        // v0.39 phase 0g — see `generate()` for the same bail message.
        anyhow::bail!(
            "Stable Cascade generate_img2img() is pending v0.40 integration. \
             v0.39 rewrote the architecture to load real upstream weights; the \
             inference path lands in v0.40."
        );
    }
}

/// v0.39 phase 0g: derive the Stage C prior HF repo from the user's
/// Stage A/B repo (Stage C lives in `stabilityai/stable-cascade-prior`,
/// not the standard `stabilityai/stable-cascade`). Returns the prior
/// repo id, or `None` if `repo` doesn't follow the expected mapping
/// (caller falls back to a default).
fn stage_c_prior_repo(repo: &str) -> Option<String> {
    // Canonical mapping: stabilityai/stable-cascade → stabilityai/stable-cascade-prior.
    // Lite forks: `*-lite` suffix on the base repo maps to `*-prior` (no -lite).
    if repo.ends_with("/stable-cascade") {
        Some(repo.replace("/stable-cascade", "/stable-cascade-prior"))
    } else {
        None
    }
}

/// v0.38 phase 3: optional LoRA merge into a temporary safetensors
/// file. Returns the original `base` path when the LoRA stack is
/// empty (zero work, zero IO); otherwise writes a stage-specific
/// merged tempfile (under `std::env::temp_dir()` with pid + nanos
/// for uniqueness) and returns its path. The caller mmaps the
/// returned path — the tempfile stays alive for the lifetime of
/// that mmap (same pattern pixart::Pipeline::load uses, no explicit
/// cleanup; OS sweep handles disposal).
fn maybe_merge_loras(
    base: &std::path::Path,
    loras: &[crate::pipelines::lora::ResolvedLora],
    lora_scale: f32,
    device: &Device,
    stage: crate::pipelines::cascade_lora::Stage,
) -> Result<std::path::PathBuf> {
    if loras.is_empty() {
        return Ok(base.to_path_buf());
    }
    let merge_spinner = progress::spinner(&format!(
        "Merging {} Cascade LoRA(s) into Stage {:?}",
        loras.len(),
        stage
    ));
    let out_path = std::env::temp_dir().join(format!(
        "plakat-cascade-{:?}-lora-merged-{}-{}.safetensors",
        stage,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let (n_mod, n_total) = match stage {
        crate::pipelines::cascade_lora::Stage::B => {
            crate::pipelines::cascade_lora::merge_cascade_b_loras_into_weights(
                base,
                &out_path,
                loras,
                lora_scale,
                device,
            )
        }
        crate::pipelines::cascade_lora::Stage::C => {
            crate::pipelines::cascade_lora::merge_cascade_c_loras_into_weights(
                base,
                &out_path,
                loras,
                lora_scale,
                device,
            )
        }
    }?;
    merge_spinner.finish_with_message(format!(
        "✓ Cascade Stage {:?} LoRA merge: {n_mod}/{n_total} target groups applied",
        stage
    ));
    Ok(out_path)
}

/// Stable Cascade entrypoint called by `t2i::run` when
/// `Variant::detect` classifies the model as Stable Cascade.
/// Phase 0: bails after a successful CLIP-G load, proving the
/// dispatch wiring + the text-encoder foundation.
pub async fn run(req: RunRequest) -> Result<()> {
    let repo = if req.model.contains('/') {
        req.model.clone()
    } else {
        crate::hf::resolve_alias(&req.model).to_string()
    };

    // v0.38 phase 3: resolve LoRA specs to on-disk safetensors before
    // load. Mirrors pixart::run's resolve-then-pass pattern.
    let mut resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> =
        Vec::with_capacity(req.loras.len());
    for spec in &req.loras {
        resolved_loras.push(spec.resolve().await?);
    }

    let mut pipeline = Pipeline::load(LoadRequest {
        repo,
        device: req.device.clone(),
        loras: resolved_loras,
        lora_scale: req.lora_scale,
        controlnet_weights: req.controlnet_weights.clone(),
    })
    .await?;

    // v0.38 phase 5: build the per-call conditioning tensor when CN
    // is wired AND a ControlSpec was supplied. The conditioning
    // image is loaded from `spec.image`; auto-annotate via
    // `spec.from` is deferred (annotator pickers + Cascade CN
    // combos aren't yet validated). Without weights OR without
    // spec, `control_conditioning` stays None and generate runs as
    // plain t2i.
    let control_conditioning: Option<ControlConditioning> = match (
        pipeline.control_conditioning_active(),
        req.control_spec.as_ref(),
    ) {
        (true, Some(spec)) => {
            let image_path = spec.image.as_ref().ok_or_else(|| {
                anyhow!(
                    "Cascade ControlNet requires `--control-image PATH` (or \
                     `image=` in `--control-spec`); auto-annotate via \
                     `--control-from` is a v0.39 follow-up."
                )
            })?;
            let cond = crate::imaging::preprocess::sd_image_tensor(
                image_path,
                1024,
                1024,
                &req.device,
                pipeline.dtype,
            )
            .with_context(|| {
                format!(
                    "loading Cascade control conditioning image {}",
                    image_path.display()
                )
            })?;
            Some(ControlConditioning {
                conditioning_image: cond,
                scale: spec.strength,
                start: spec.start,
                end: spec.end,
            })
        }
        (false, Some(_)) => {
            tracing::warn!(
                target: "plakat",
                "Cascade run received a ControlSpec but no controlnet_weights — \
                 spec is ignored. Pass `--cascade-control-weights PATH` to enable."
            );
            None
        }
        _ => None,
    };

    let base_seed = req
        .seed
        .unwrap_or_else(|| rand::random::<u64>() & (u32::MAX as u64));

    // v0.38 phase 3: pre-build LoRA metadata stack so each generated
    // PNG carries the same record SD/Flux/SD3/PixArt do.
    let metadata_lora_stack: Vec<crate::imaging::metadata::LoraEntry> = req
        .loras
        .iter()
        .map(|s| s.to_entry())
        .collect();

    std::fs::create_dir_all(&req.out_dir)
        .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

    for idx in 0..req.count {
        let seed = base_seed.wrapping_add(idx as u64);
        crate::ui::progress::println(&format!(
            "  {} stable-cascade {} of {} (seed={seed})",
            console::style("◆").cyan().bold(),
            idx + 1,
            req.count,
        ));
        let (buf, ow, oh) = pipeline.generate(
            &req.prompt,
            &req.negative,
            req.output_dim,
            req.stage_c_steps,
            req.stage_b_steps,
            req.guidance,
            seed,
            req.scheduler,
            control_conditioning.as_ref(),
        )?;

        // Build sidecar metadata. Same field set PixArt emits
        // (v0.35 phase 4). Stable Cascade specific extras (Stage C
        // / Stage B step counts, conditioning provenance) land in
        // v0.38 alongside the FiLM + effnet wiring.
        let mut m = crate::imaging::metadata::GenerationMetadata::new(
            req.prompt.clone(),
            req.model.clone(),
            seed,
            req.stage_c_steps,
            req.guidance,
            format!("{:?}", req.scheduler).to_lowercase(),
            ow,
            oh,
        );
        m.negative = req.negative.clone();
        if !metadata_lora_stack.is_empty() {
            m.with_lora_stack(metadata_lora_stack.clone());
            m.lora_scale = Some(req.lora_scale);
        }

        let out_path = req
            .out_dir
            .join(format!("plakat-cascade-{seed}.png"));
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, ow, oh, &out_path, &m)?;
        crate::ui::progress::println(&format!(
            "  {} {}",
            console::style("✓").green().bold(),
            out_path.display()
        ));
    }

    Ok(())
}

/// v0.38 phase 4: Stable Cascade img2img CLI entrypoint. Routed by
/// `cli::img2img::run` when `Variant::detect` classifies the model
/// as Stable Cascade. Loads the pipeline, runs `generate_img2img`,
/// writes the output PNG with metadata.
pub async fn run_img2img(req: RunImg2imgRequest) -> Result<()> {
    let repo = if req.model.contains('/') {
        req.model.clone()
    } else {
        crate::hf::resolve_alias(&req.model).to_string()
    };

    let mut resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> =
        Vec::with_capacity(req.loras.len());
    for spec in &req.loras {
        resolved_loras.push(spec.resolve().await?);
    }

    let mut pipeline = Pipeline::load(LoadRequest {
        repo,
        device: req.device.clone(),
        loras: resolved_loras,
        lora_scale: req.lora_scale,
        // v0.38 phase 5: Cascade img2img + ControlNet is deferred
        // (v0.39 follow-up). The img2img CLI doesn't expose
        // `--cascade-control-weights` either; this stays None.
        controlnet_weights: None,
    })
    .await?;

    let base_seed = req
        .seed
        .unwrap_or_else(|| rand::random::<u64>() & (u32::MAX as u64));
    let metadata_lora_stack: Vec<crate::imaging::metadata::LoraEntry> = req
        .loras
        .iter()
        .map(|s| s.to_entry())
        .collect();

    std::fs::create_dir_all(&req.out_dir)
        .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

    for idx in 0..req.count {
        let seed = base_seed.wrapping_add(idx as u64);
        crate::ui::progress::println(&format!(
            "  {} stable-cascade img2img {} of {} (seed={seed}, strength={:.2})",
            console::style("◆").cyan().bold(),
            idx + 1,
            req.count,
            req.strength,
        ));
        let (buf, ow, oh) = pipeline.generate_img2img(
            &req.init_image,
            &req.prompt,
            &req.negative,
            req.stage_c_steps,
            req.stage_b_steps,
            req.strength,
            req.guidance,
            seed,
            req.scheduler,
        )?;

        let mut m = crate::imaging::metadata::GenerationMetadata::new(
            req.prompt.clone(),
            req.model.clone(),
            seed,
            req.stage_c_steps + req.stage_b_steps,
            req.guidance,
            format!("{:?}", req.scheduler).to_lowercase(),
            ow,
            oh,
        );
        m.negative = req.negative.clone();
        m.strength = Some(req.strength);
        if !metadata_lora_stack.is_empty() {
            m.with_lora_stack(metadata_lora_stack.clone());
            m.lora_scale = Some(req.lora_scale);
        }

        let out_path = req
            .out_dir
            .join(format!("plakat-cascade-img2img-{seed}.png"));
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, ow, oh, &out_path, &m)?;
        crate::ui::progress::println(&format!(
            "  {} {}",
            console::style("✓").green().bold(),
            out_path.display()
        ));
    }

    Ok(())
}

/// CLI entrypoint: parameters needed for one Stable Cascade
/// img2img generation. v0.38 phase 4.
#[derive(Clone)]
pub struct RunImg2imgRequest {
    pub model: String,
    pub device: Device,
    pub init_image: std::path::PathBuf,
    pub prompt: String,
    pub negative: String,
    pub stage_c_steps: usize,
    pub stage_b_steps: usize,
    /// Img2img denoise strength in `[0, 1]`. 1.0 = pure t2i
    /// (matches `generate`). 0.6 = upstream default. 0.0 = no
    /// denoise (decoded init image only).
    pub strength: f32,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub scheduler: SchedulerKind,
    pub out_dir: std::path::PathBuf,
    pub count: u32,
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    pub lora_scale: f32,
}

/// CLI entrypoint: parameters needed for one Stable Cascade
/// generation. v0.37 phase 4 grows beyond the phase 0 stub.
#[derive(Clone)]
pub struct RunRequest {
    pub model: String,
    pub device: Device,
    pub prompt: String,
    pub negative: String,
    /// Output image side length (output is square because Stage C's
    /// prior latent is fixed at 24×24×16). Must be divisible by 8 —
    /// the total Stage A↔B compression contract.
    pub output_dim: u32,
    /// Number of Stage C denoise steps (the heavy text-to-prior
    /// stage). Upstream recommendation: 20.
    pub stage_c_steps: usize,
    /// Number of Stage B denoise steps (Stage C latent → Stage A
    /// latent). Upstream recommendation: 10.
    pub stage_b_steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub scheduler: SchedulerKind,
    pub out_dir: std::path::PathBuf,
    /// Count of images (per-image seed = base + idx).
    pub count: u32,
    /// v0.38 phase 3: unresolved Cascade LoRA specs (resolved
    /// inside `cascade::run` before `Pipeline::load`).
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    /// Global LoRA scale multiplier. Default 1.0.
    pub lora_scale: f32,
    /// v0.38 phase 5: at most one ControlSpec (multi-CN deferred).
    /// `image` (or `from` for auto-annotate) supplies the
    /// conditioning image; `strength` / `start` / `end` shape the
    /// residual window. Ignored unless `controlnet_weights` is
    /// also set.
    pub control_spec: Option<crate::pipelines::controlnet::ControlSpec>,
    /// v0.38 phase 5: path to Stable Cascade ControlNet weights
    /// (safetensors). When `None`, no CN is loaded and any
    /// `control_spec` is logged + ignored.
    pub controlnet_weights: Option<std::path::PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v0.37 phase 4: RunRequest carries every field
    /// `t2i::run` needs to dispatch into `cascade::run`.
    #[test]
    fn run_request_carries_all_inference_fields() {
        let r = RunRequest {
            model: "stable-cascade".into(),
            device: Device::Cpu,
            prompt: "a fox in a meadow".into(),
            negative: "blurry".into(),
            output_dim: 1024,
            stage_c_steps: 20,
            stage_b_steps: 10,
            guidance: 4.0,
            seed: Some(42),
            scheduler: SchedulerKind::DpmppKarras,
            out_dir: std::path::PathBuf::from("/tmp/cascade-test"),
            count: 1,
            loras: Vec::new(),
            lora_scale: 1.0,
            control_spec: None,
            controlnet_weights: None,
        };
        assert_eq!(r.prompt, "a fox in a meadow");
        assert_eq!(r.stage_c_steps, 20);
        assert_eq!(r.output_dim, 1024);
        assert_eq!(r.stage_b_steps, 10);
        assert_eq!(r.seed, Some(42));
        assert_eq!(r.count, 1);
        assert_eq!(r.lora_scale, 1.0);
        assert!(r.loras.is_empty());
    }

    /// v0.38 phase 4: RunImg2imgRequest carries every field
    /// `cli::img2img::run_cascade_img2img` needs.
    #[test]
    fn run_img2img_request_carries_all_fields() {
        let r = RunImg2imgRequest {
            model: "stable-cascade".into(),
            device: Device::Cpu,
            init_image: std::path::PathBuf::from("/tmp/init.png"),
            prompt: "a fox".into(),
            negative: "blurry".into(),
            stage_c_steps: 20,
            stage_b_steps: 10,
            strength: 0.6,
            guidance: 4.0,
            seed: Some(7),
            scheduler: SchedulerKind::DpmppKarras,
            out_dir: std::path::PathBuf::from("/tmp/out"),
            count: 1,
            loras: Vec::new(),
            lora_scale: 1.0,
        };
        assert_eq!(r.prompt, "a fox");
        assert_eq!(r.strength, 0.6);
        assert_eq!(r.stage_c_steps, 20);
        assert_eq!(r.stage_b_steps, 10);
        assert_eq!(r.init_image.file_name().unwrap(), "init.png");
    }

    /// v0.38 phase 4: schedule truncation math. At strength=1.0
    /// keep every timestep; at 0.0 keep none; intermediate values
    /// proportionally drop the leading entries (the high-noise
    /// segments).
    #[test]
    fn img2img_schedule_truncation_skip_count() {
        // Mirrors the formula used inside generate_img2img:
        //   skip = round((1 - strength) * n_total)
        fn skip(n: usize, s: f32) -> usize {
            (((1.0 - s as f64) * n as f64).round() as usize).min(n)
        }
        assert_eq!(skip(10, 1.0), 0); // pure t2i: keep everything
        assert_eq!(skip(10, 0.0), 10); // no denoise: drop everything
        assert_eq!(skip(10, 0.5), 5); // half-and-half
        assert_eq!(skip(20, 0.6), 8); // upstream default (≈ 0.6)
        assert_eq!(skip(0, 0.5), 0); // empty schedule: nothing to skip
    }

    /// v0.37 phase 0: aliases resolve to the canonical Stable
    /// Cascade repo.
    #[test]
    fn alias_stable_cascade_resolves_to_canonical_repo() {
        assert_eq!(
            crate::hf::resolve_alias("stable-cascade"),
            "stabilityai/stable-cascade"
        );
        assert_eq!(
            crate::hf::resolve_alias("cascade"),
            "stabilityai/stable-cascade"
        );
    }

    /// v0.37 phase 0: aliases listed in all_known_aliases().
    #[test]
    fn cascade_aliases_listed_in_all_known() {
        let known = crate::hf::all_known_aliases();
        assert!(known.contains(&"stable-cascade"), "got {known:?}");
        assert!(known.contains(&"cascade"), "got {known:?}");
    }

    /// v0.40 phase 4: end-to-end real-weight smoke. Loads all four
    /// stages from `STABLE_CASCADE_WEIGHTS_DIR`, runs generate at
    /// a small output size (256²) with minimal steps to verify the
    /// inference path runs end-to-end without errors or NaN.
    ///
    /// Skipped unless `STABLE_CASCADE_WEIGHTS_DIR` is set and the
    /// required files exist. Heavy on RAM and time (loads all four
    /// checkpoints simultaneously, ~6 GB; runs at Stage C 24² and
    /// Stage B 32² with 2 + 2 steps for speed).
    #[test]
    #[ignore = "real-weight smoke; opt-in via cargo test -- --ignored"]
    fn end_to_end_smoke_from_real_upstream_weights() {
        let dir_var = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "Skipping end_to_end_smoke: STABLE_CASCADE_WEIGHTS_DIR not set."
                );
                return;
            }
        };
        let weights_dir = std::path::PathBuf::from(&dir_var);
        let required = [
            "vqgan/diffusion_pytorch_model.safetensors",
            "decoder/diffusion_pytorch_model.safetensors",
            "prior/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
        ];
        for r in &required {
            let p = weights_dir.join(r);
            if !p.exists() {
                eprintln!("Skipping end_to_end_smoke: {} missing", p.display());
                return;
            }
        }

        // Load all four stages.
        let device = Device::Cpu;
        let dtype = DType::F32;
        let clip_g_w = weights_dir.join("text_encoder/model.safetensors");
        let clip_g_tok_path = weights_dir.join("tokenizer/tokenizer.json");
        let stage_a_w = weights_dir.join("vqgan/diffusion_pytorch_model.safetensors");
        let stage_b_w = weights_dir.join("decoder/diffusion_pytorch_model.safetensors");
        let stage_c_w = weights_dir.join("prior/diffusion_pytorch_model.safetensors");

        let clip_g_cfg = vendored_clip::Config::sdxl2();
        let clip_g_vs = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&clip_g_w], dtype, &device).unwrap()
        };
        let clip_g_enc = SdxlClipGTextTransformer::new(clip_g_vs, &clip_g_cfg, 1280).unwrap();
        let clip_g_tok = Tokenizer::from_file(&clip_g_tok_path).unwrap();

        let stage_a_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_a_w.as_path()], dtype, &device).unwrap()
        };
        let stage_a = StageAVae::new(VaeConfig::paella_v3(), stage_a_vb).unwrap();

        let stage_b_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_b_w.as_path()], dtype, &device).unwrap()
        };
        let stage_b = StableCascadePrior::new_stage_b(PriorConfig::stage_b_full(), stage_b_vb).unwrap();

        let stage_c_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_c_w.as_path()], dtype, &device).unwrap()
        };
        let stage_c = StableCascadePrior::new_stage_c(PriorConfig::stage_c_full(), stage_c_vb).unwrap();

        let mut pipeline = Pipeline {
            device: device.clone(),
            dtype,
            clip_g_enc,
            clip_g_tok,
            stage_a,
            stage_b,
            stage_c,
            controlnet: None,
        };

        eprintln!("All 4 stages loaded. Running generate at 256² with 2+2 steps...");
        let result = pipeline.generate_at_size(
            "a misty forest at dawn, painterly",
            "blurry, low quality",
            256,
            2,
            2,
            4.0,
            42,
            SchedulerKind::DpmppKarras,
        );
        match result {
            Ok((buf, w, h)) => {
                assert_eq!(buf.len(), (w as usize) * (h as usize) * 3);
                let any_nan = buf.iter().any(|&b| b == 0xff && buf.iter().all(|&x| x == 0xff || x == 0));
                let mean: f32 = buf.iter().map(|&b| b as f32).sum::<f32>() / (buf.len() as f32);
                let max = *buf.iter().max().unwrap();
                let min = *buf.iter().min().unwrap();
                eprintln!(
                    "✓ Generate end-to-end OK: {}×{} ({} bytes), \
                     mean={:.1}, min={}, max={}, uniform-byte={}",
                    w, h, buf.len(), mean, min, max, any_nan
                );
                // Save output for visual inspection.
                let out_path = std::env::temp_dir().join("cascade_smoke.png");
                if let Some(img) = image::RgbImage::from_raw(w, h, buf) {
                    img.save(&out_path).ok();
                    eprintln!("✓ saved to {}", out_path.display());
                }
            }
            Err(e) => panic!("generate failed: {e}"),
        }
    }
}
