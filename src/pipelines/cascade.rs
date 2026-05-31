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
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::pipelines::cascade_stage_a::{Config as StageAConfig, StageAVae};
use crate::pipelines::sdxl_clip::SdxlClipGTextTransformer;
use crate::pipelines::vendored_clip;
use crate::ui::progress;

/// Inputs to [`Pipeline::load`]. Mirrors the shape of
/// `flux::LoadRequest` / `pixart::LoadRequest` so the scenario +
/// scripting cache machinery can hand off uniformly when those
/// integrations land in v0.38+.
pub struct LoadRequest {
    /// Resolved HF repo id (callers run `crate::hf::resolve_alias`
    /// first; this struct holds the canonical form).
    pub repo: String,
    pub device: Device,
}

/// Stable Cascade pipeline.
///
/// Phase 0 shipped CLIP-G + tokenizer. v0.37 phase 1 adds Stage A
/// VAE (the small Paella-v3 design for image ↔ latent mapping).
/// Stage B and Stage C land in phases 2 / 3.
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
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
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
        dl.finish_with_message("✓ Stable Cascade text-encoder + Stage A weights resolved");

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

        let stage_a_build = progress::spinner("Loading Stage A VAE");
        let stage_a_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_a_w.as_path()], dtype, &req.device)?
        };
        let stage_a = StageAVae::new(StageAConfig::paella_v3(), stage_a_vb)
            .context("building Stage A VAE for Stable Cascade")?;
        stage_a_build.finish_with_message("✓ Stage A VAE ready");

        Ok(Self {
            device: req.device,
            dtype,
            clip_g_enc,
            clip_g_tok,
            stage_a,
        })
    }
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

    let pipeline = Pipeline::load(LoadRequest {
        repo,
        device: req.device.clone(),
    })
    .await?;

    tracing::info!(
        target: "plakat",
        "Stable Cascade phase 1: CLIP-G + Stage A loaded (dtype={:?}). \
         Stage B lands in v0.37 phase 2; Stage C in phase 3; 3-stage orchestration in phase 4.",
        pipeline.dtype
    );

    anyhow::bail!(
        "Stable Cascade inference is not yet implemented — phase 1 ships the \
         CLIP-G text encoder + Stage A VAE foundation (this load succeeded). \
         Stage B latent prior lands in v0.37 phase 2; Stage C high-res prior \
         in phase 3; 3-stage orchestration (text → Stage C → Stage B → Stage A \
         → image) in phase 4. Track progress against \
         `Documentation/RFC_v0.37_STABLE_CASCADE.md`."
    )
}

/// Minimal request shape Stable Cascade's phase 0 stub consumes.
/// Will grow additively in later phases (prompt, negative, steps,
/// guidance, seed, scheduler, width, height — same fields t2i /
/// sd3 / flux / pixart requests carry).
#[derive(Clone)]
pub struct RunRequest {
    pub model: String,
    pub device: Device,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub `RunRequest` carries the minimum fields required to
    /// dispatch from t2i::run. Later phases extend without renaming.
    #[test]
    fn run_request_carries_model_and_device() {
        let r = RunRequest {
            model: "stable-cascade".into(),
            device: Device::Cpu,
        };
        assert_eq!(r.model, "stable-cascade");
        matches!(r.device, Device::Cpu);
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
}
