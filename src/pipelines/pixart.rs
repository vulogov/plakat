//! PixArt Sigma pipeline — fourth model family.
//!
//! v0.35 phase 0 (this commit): foundational stub. Pipeline::load
//! downloads and assembles the T5-XXL text encoder + SDXL-shape
//! VAE for the canonical `PixArt-Σ-XL-2-1024-MS` checkpoint, but
//! the DiT-XL/2 backbone is NOT yet implemented. `run()` bails
//! with a clear "DiT inference lands in phase 1" message.
//!
//! Architecture (target end-state across phases 0-4):
//!
//! * **DiT-XL/2 backbone** (~600M params). PixArt-Σ adds KV-
//!   compression on top of PixArt-α — sparse cross-attention to
//!   the T5 sequence. Phase 1.
//! * **T5-XXL text encoder** (~4.7B params). Same T5 we ship for
//!   SD3 + Flux today (sourced from `candle_transformers::models::t5`).
//!   Load pattern mirrors sd3.rs verbatim. **Phase 0.**
//! * **SD-family KL-VAE** (~330 MB). Reused via the v0.34 phase 3
//!   Arc-cache mechanism — mixed-kind scenarios with SDXL+PixArt
//!   share one VAE handle. **Phase 0.**
//! * **DPM++ sampler** (PixArt-Σ's published recommendation).
//!   Phase 2.
//!
//! Acceptance criteria for v0.35 phase 0 (per RFC):
//! - `--model pixart` resolves via the alias system. ✓ (`hf/mod.rs`).
//! - The pipeline module compiles + exports a stub `Pipeline::load`.
//!   ✓ (this file).
//! - `t2i::Pipeline::load` rejects PixArt with a clear pointer at
//!   `pipelines::pixart` (parallels the Flux + SD3 bail pattern).
//! - `t2i::run` routes PixArt to `pixart::run` (which bails until
//!   phase 1). Proves the dispatch wiring without inference.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{StableDiffusionConfig, vae::AutoEncoderKL};
use candle_transformers::models::t5;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::ui::progress;

/// Inputs to [`Pipeline::load`]. Mirrors the shape of
/// `sd3::LoadRequest` / `flux::LoadRequest` so the scenario +
/// scripting cache machinery can hand off uniformly.
pub struct LoadRequest {
    /// Resolved HF repo id (callers run `crate::hf::resolve_alias`
    /// first; this struct holds the canonical form).
    pub repo: String,
    pub device: Device,
    /// v0.34 phase 3 mechanism: pre-built VAE shared with t2i's
    /// scenario-level cache. `Some` reuses; `None` builds fresh.
    pub vae_cache: Option<Arc<AutoEncoderKL>>,
}

/// PixArt Sigma pipeline.
///
/// Phase 0 ships T5 + VAE only. The DiT backbone field will land
/// in phase 1; this struct grows additively (no field rename).
pub struct Pipeline {
    pub device: Device,
    pub dtype: DType,
    /// T5-XXL text encoder — same `candle_transformers` type SD3
    /// and Flux use. PixArt feeds the full T5 hidden states (not
    /// the pooled output) into the DiT cross-attention; no CLIP
    /// branch (unlike SD3's CLIP-L + CLIP-G + T5 trio).
    pub t5_enc: t5::T5EncoderModel,
    pub t5_tok: Tokenizer,
    /// SD-family KL-VAE, Arc-shared via the v0.34 phase 3 cache.
    /// Same VAE used by SDXL; PixArt-Σ inherits its 8× downsample +
    /// 4 latent channels.
    pub vae: Arc<AutoEncoderKL>,
}

impl Pipeline {
    /// Phase 0 load: T5 + VAE.
    ///
    /// Repo layout assumes the canonical diffusers
    /// `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` structure:
    ///
    /// ```text
    /// text_encoder/
    ///   config.json
    ///   model-00001-of-00003.safetensors
    ///   model-00002-of-00003.safetensors
    ///   model-00003-of-00003.safetensors
    /// tokenizer/
    ///   tokenizer.json
    /// vae/
    ///   diffusion_pytorch_model.safetensors
    /// transformer/
    ///   diffusion_pytorch_model.safetensors    [phase 1 — not loaded yet]
    /// ```
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        let dl = progress::spinner("Resolving PixArt Sigma weights");
        // T5 ships as 3 shards in the canonical Sigma checkpoint.
        let t5_shard1 = crate::hf::download::get_file(
            &req.repo,
            "text_encoder/model-00001-of-00003.safetensors",
        )
        .await
        .context("downloading T5-XXL shard 1 for PixArt")?;
        let t5_shard2 = crate::hf::download::get_file(
            &req.repo,
            "text_encoder/model-00002-of-00003.safetensors",
        )
        .await
        .context("downloading T5-XXL shard 2 for PixArt")?;
        let t5_shard3 = crate::hf::download::get_file(
            &req.repo,
            "text_encoder/model-00003-of-00003.safetensors",
        )
        .await
        .context("downloading T5-XXL shard 3 for PixArt")?;
        let t5_cfg_path = crate::hf::download::get_file(&req.repo, "text_encoder/config.json")
            .await
            .context("downloading T5 config for PixArt")?;
        let t5_tok_path = crate::hf::download::get_file(&req.repo, "tokenizer/tokenizer.json")
            .await
            .context("downloading T5 tokenizer for PixArt")?;
        let vae_path = crate::hf::download::get_file(
            &req.repo,
            "vae/diffusion_pytorch_model.safetensors",
        )
        .await
        .context("downloading VAE weights for PixArt")?;
        dl.finish_with_message("✓ PixArt weights resolved");

        let build = progress::spinner("Loading T5-XXL text encoder");
        let t5_cfg_str = std::fs::read_to_string(&t5_cfg_path)
            .with_context(|| format!("read T5 config {}", t5_cfg_path.display()))?;
        let t5_cfg: t5::Config =
            serde_json::from_str(&t5_cfg_str).context("parse T5 config (PixArt)")?;
        let t5_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[&t5_shard1, &t5_shard2, &t5_shard3],
                dtype,
                &req.device,
            )?
        };
        let t5_enc = t5::T5EncoderModel::load(t5_vb, &t5_cfg)
            .context("building T5-XXL encoder for PixArt")?;
        let t5_tok =
            Tokenizer::from_file(&t5_tok_path).map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
        build.finish_with_message("✓ T5-XXL ready");

        // v0.34 phase 3 cache passthrough. SDXL VAE config is the
        // closest match — PixArt inherits the same 8× downsample +
        // 4-channel latent shape, and the diffusers checkpoint
        // weights load through the SDXL VAE accessor.
        let vae_build = progress::spinner("Loading PixArt VAE");
        let cfg = StableDiffusionConfig::sdxl(None, None, None);
        let vae = match req.vae_cache {
            Some(arc) => {
                tracing::info!(
                    target: "plakat",
                    "PixArt: reusing cached VAE (skipping {} build)",
                    vae_path.display()
                );
                arc
            }
            None => Arc::new(cfg.build_vae(&vae_path, &req.device, dtype)?),
        };
        vae_build.finish_with_message("✓ VAE ready");

        Ok(Self {
            device: req.device,
            dtype,
            t5_enc,
            t5_tok,
            vae,
        })
    }
}

/// PixArt entrypoint called by `t2i::run` when `Variant::detect`
/// classifies the model as PixArt. Phase 0: bails after a
/// successful pipeline load, proving the dispatch wiring + the T5
/// + VAE foundation.
pub async fn run(req: RunRequest) -> Result<()> {
    let repo = if req.model.contains('/') {
        req.model.clone()
    } else {
        crate::hf::resolve_alias(&req.model).to_string()
    };

    let pipeline = Pipeline::load(LoadRequest {
        repo,
        device: req.device.clone(),
        vae_cache: None, // v0.35 phase 0: no cross-kind cache wiring yet
    })
    .await?;

    // Sanity touch on the loaded pieces so the compiler can't
    // optimise the load away (and so the user sees PixArt actually
    // landed before the phase 1 bail).
    tracing::info!(
        target: "plakat",
        "PixArt phase 0: T5 + VAE loaded (dtype={:?}). DiT inference lands in v0.35 phase 1.",
        pipeline.dtype
    );

    anyhow::bail!(
        "PixArt Sigma DiT inference is not yet implemented — phase 0 ships \
         the T5 + VAE foundation (this load succeeded). The DiT-XL/2 backbone \
         + denoising loop land in v0.35 phase 1. Track progress against \
         `Documentation/RFC_v0.35_PIXART_SIGMA.md`."
    )
}

/// Minimal request shape PixArt's phase 0 stub consumes. Will
/// grow additively in later phases (prompt, negative, steps,
/// guidance, seed, scheduler — same fields t2i / sd3 / flux
/// requests carry).
#[derive(Clone)]
pub struct RunRequest {
    pub model: String,
    pub device: Device,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub `RunRequest` has the minimum fields required to dispatch.
    /// Phase 1 will add prompt/seed/etc; this guards the shape.
    #[test]
    fn run_request_carries_model_and_device() {
        let r = RunRequest {
            model: "pixart".into(),
            device: Device::Cpu,
        };
        assert_eq!(r.model, "pixart");
        matches!(r.device, Device::Cpu);
    }

    #[test]
    fn alias_pixart_resolves_to_sigma_repo() {
        assert_eq!(
            crate::hf::resolve_alias("pixart"),
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        );
        assert_eq!(
            crate::hf::resolve_alias("pixart-sigma"),
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        );
        assert_eq!(
            crate::hf::resolve_alias("pixart-1024"),
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        );
    }

    #[test]
    fn pixart_aliases_listed_in_all_known() {
        let known = crate::hf::all_known_aliases();
        assert!(known.contains(&"pixart"), "got {known:?}");
        assert!(known.contains(&"pixart-sigma"), "got {known:?}");
        assert!(known.contains(&"pixart-1024"), "got {known:?}");
    }
}
