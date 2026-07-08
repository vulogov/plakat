//! PixArt Sigma pipeline — fourth model family.
//!
//! v0.35 phase 2: end-to-end inference. `Pipeline::load` assembles
//! T5-XXL + DiT-XL/2 + VAE; `run` executes the standard CFG
//! denoise loop and saves the resulting PNG. Output target is the
//! canonical `PixArt-Σ-XL-2-1024-MS` checkpoint.
//!
//! Pipeline composition:
//!
//! * **T5-XXL text encoder** (~4.7B params) — sourced from
//!   `candle_transformers::models::t5`. Same `T5EncoderModel` SD3
//!   uses.
//! * **DiT-XL/2 backbone** (~600M params) — vendored in
//!   `pipelines::pixart_dit` (v0.35 phase 1, v0.36 phase 3 added
//!   KV-compression). adaLN-single + per-block scale_shift_table.
//! * **SD-family KL-VAE** — Arc-shared via the v0.34 phase 3 cache.
//! * **DPM++ sampler** via `pipelines::scheduler` (PixArt-Σ's
//!   recommendation). v0.36 phase 4: LCM scheduler composes too
//!   when paired with a PixArt LCM-LoRA (see below).
//! * **Seed plumbing** through `pipelines::seeds::prepare_seed`
//!   (v0.34 phase 1 chokepoint).
//!
//! ## v0.36 phase 4: LCM with PixArt
//!
//! PixArt-Σ-LCM is not published as a standalone checkpoint, but
//! 2-step (or 4-step) PixArt generation composes from existing
//! infrastructure today:
//!
//! ```bash
//! plakat generate "a misty forest at dawn" \
//!     --model pixart \
//!     --lora civitai:NNNNNN:1.0 \   # PixArt LCM-LoRA
//!     --scheduler lcm --steps 4 --guidance 1.5
//! ```
//!
//! Mechanism:
//! - `SchedulerKind::Lcm` (v0.28 phase 1) routes through
//!   `lcm_scheduler::LcmSchedulerConfig` inside `scheduler::build`
//!   — already wired into PixArt's denoise loop at line ~271.
//! - PixArt LoRA merge (v0.35 phase 4) handles the LCM-LoRA via
//!   the diffusers PEFT format — no PixArt-specific changes
//!   needed.
//! - Recommended hyperparameters mirror the SD-family LCM-LoRA
//!   pattern: 4 steps + guidance 1.5 (the LCM-LoRA was distilled
//!   under those conditions).
//!
//! ## What's NOT here (deferred to v0.37+)
//!
//! - **Native PixArt-α-LCM checkpoint integration**
//!   (`PixArt-alpha/PixArt-LCM-XL-2-1024-MS`). α-LCM uses the
//!   PixArt-α architecture, which lacks the Σ-specific
//!   `resolution_embedder` + `aspect_ratio_embedder` inside
//!   `adaln_single.emb`. Loading α weights into the Σ DiT would
//!   fail at the VarBuilder for those missing tensors. Supporting
//!   α-LCM requires an α/Σ architectural fork in `pixart_dit`
//!   (`Config::sigma_conditioning: bool` flag + optional
//!   embedders) — well-scoped but deferred to keep v0.36 tight.
//!   The early bail in `Pipeline::load` (below) surfaces this
//!   clearly when a user passes an α repo path.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{StableDiffusionConfig, vae::AutoEncoderKL};
// v2.1: route PixArt's T5 through the vendored copy (drop-in for candle's `t5`, proven
// equivalent by `vendored_t5::tests::vendored_matches_candle_t5`). Needed for the padding
// attention mask + an immutable `&self` encode — candle's `T5EncoderModel::forward` is `&mut`
// and takes no mask, so captions were encoded WITHOUT masking pad tokens (a real bug).
use crate::pipelines::vendored_t5 as t5;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::pipelines::pixart_dit::{Config as DitConfig, PixArtSigmaXL};
use crate::pipelines::scheduler::{SchedulerKind, build_pixart as build_scheduler};
use crate::ui::progress;

/// Inputs to [`Pipeline::load`]. Mirrors the shape of
/// `sd3::LoadRequest` / `flux::LoadRequest`.
pub struct LoadRequest {
    pub repo: String,
    pub device: Device,
    /// v0.34 phase 3 mechanism: pre-built VAE shared with t2i's
    /// scenario-level cache.
    pub vae_cache: Option<Arc<AutoEncoderKL>>,
    /// v0.35 phase 4: PixArt LoRA stack. Resolved by the caller via
    /// `LoraSpec::resolve`. Merged into the DiT safetensors at load
    /// time via `pixart_lora::merge_pixart_loras_into_weights`.
    pub loras: Vec<crate::pipelines::lora::ResolvedLora>,
    /// Global scale multiplier on each LoRA's per-spec scale (the
    /// `--lora-scale` flag semantics).
    pub lora_scale: f32,
}

/// PixArt Sigma pipeline.
pub struct Pipeline {
    pub device: Device,
    pub dtype: DType,
    /// T5-XXL text encoder. `&mut self` required for forward.
    pub t5_enc: t5::T5EncoderModel,
    pub t5_tok: Tokenizer,
    /// DiT-XL/2 backbone.
    pub dit: PixArtSigmaXL,
    /// Architecture config. Held alongside `dit` so generate can
    /// read `out_channels` / `max_caption_tokens` without unwrapping
    /// the model.
    pub dit_cfg: DitConfig,
    /// SD-family KL-VAE, Arc-shared via the v0.34 phase 3 cache.
    pub vae: Arc<AutoEncoderKL>,
    /// SD config used to build the VAE. Carries vae_scale_factor for
    /// the decode step.
    sd_cfg: StableDiffusionConfig,
}

/// v0.36 phase 4: verify the repo is a PixArt-Σ checkpoint (not
/// α / α-LCM). The Σ DiT requires `adaln_single.emb.resolution_
/// embedder` + `aspect_ratio_embedder` weights that α checkpoints
/// don't ship. Detection: every Σ checkpoint published by
/// `PixArt-alpha` carries the substring `Sigma` (case-insensitive)
/// in the repo path. PixArt-α + PixArt-LCM (the α-distilled LCM)
/// do NOT.
///
/// Returns `Ok(())` for Σ repos (and unrecognised paths — best-
/// effort; users can override by passing their own fork's full
/// HF id). Bails on `PixArt-LCM` / `PixArt-XL-2-*` strings that
/// match α with the LCM-LoRA composition pointer.
fn is_pixart_sigma_repo(repo: &str) -> Result<()> {
    let r = repo.to_lowercase();
    let has_sigma = r.contains("sigma");
    if has_sigma {
        return Ok(());
    }
    let looks_like_alpha = r.contains("pixart-xl-2") || r.contains("pixart-lcm");
    if looks_like_alpha {
        anyhow::bail!(
            "PixArt-α / α-LCM checkpoints are not yet supported — \
             this plakat build loads only PixArt-Σ DiT weights, which \
             carry Σ-specific resolution + aspect_ratio embedders that \
             α checkpoints don't ship.\n\n\
             For LCM-style 2-step / 4-step PixArt generation, compose \
             PixArt-Σ with a PixArt LCM-LoRA instead:\n\n  \
             plakat generate \"...\" --model pixart \\\n    \
             --lora civitai:NNNNNN:1.0 \\\n    \
             --scheduler lcm --steps 4 --guidance 1.5\n\n\
             Native α-LCM checkpoint integration is a v0.37+ deferral \
             (requires an α/Σ architectural fork in `pixart_dit`)."
        );
    }
    Ok(())
}

impl Pipeline {
    /// v0.35 phase 2: full load. Downloads T5 (3 shards) + DiT +
    /// VAE from the canonical diffusers layout, builds each module,
    /// returns the assembled pipeline.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        // v0.36 phase 4: detect PixArt-α (non-Σ) repos and bail
        // early with a pointer at the LCM-LoRA composition path.
        // Plakat's DiT loads Σ-specific `adaln_single.emb.
        // resolution_embedder` + `aspect_ratio_embedder` tensors
        // which α checkpoints (including α-LCM) don't ship —
        // VarBuilder would surface this as an opaque missing-key
        // error mid-load. Better to fail fast at the boundary.
        is_pixart_sigma_repo(&req.repo)?;

        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        let dl = progress::spinner("Resolving PixArt Sigma weights");
        // T5-XXL ships sharded, but the repo has re-sharded over time
        // (3 → 2 shards), so discover the shard set from the index rather
        // than hardcoding the count. Falls back to a single-file encoder
        // if the repo has no index.
        let t5_shards: Vec<std::path::PathBuf> = match crate::hf::download::get_file(
            &req.repo,
            "text_encoder/model.safetensors.index.json",
        )
        .await
        {
            Ok(index_path) => {
                let idx_str = std::fs::read_to_string(&index_path)
                    .context("read T5 shard index for PixArt")?;
                let idx: serde_json::Value = serde_json::from_str(&idx_str)
                    .context("parse T5 shard index for PixArt")?;
                let mut names: Vec<String> = idx
                    .get("weight_map")
                    .and_then(|m| m.as_object())
                    .map(|m| m.values().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                names.sort();
                names.dedup();
                if names.is_empty() {
                    anyhow::bail!("T5 shard index for PixArt has no weight_map entries");
                }
                let mut paths = Vec::with_capacity(names.len());
                for (i, name) in names.iter().enumerate() {
                    paths.push(
                        crate::hf::download::get_file(&req.repo, &format!("text_encoder/{name}"))
                            .await
                            .with_context(|| {
                                format!("downloading T5-XXL shard {} ({name}) for PixArt", i + 1)
                            })?,
                    );
                }
                paths
            }
            Err(_) => vec![crate::hf::download::get_file(
                &req.repo,
                "text_encoder/model.safetensors",
            )
            .await
            .context("downloading T5-XXL encoder for PixArt")?],
        };
        let t5_cfg_path = crate::hf::download::get_file(&req.repo, "text_encoder/config.json")
            .await
            .context("downloading T5 config for PixArt")?;
        // The Sigma repo dropped tokenizer.json (sentencepiece-only now).
        // The T5-v1.1 vocab is identical across every T5 model, so any
        // flan-t5 tokenizer.json is a drop-in for the fast tokenizer.
        let t5_tok_path = crate::hf::download::get_first_of(&[
            (req.repo.as_str(), "tokenizer/tokenizer.json"),
            ("google/flan-t5-base", "tokenizer.json"),
        ])
        .await
        .context("downloading T5 tokenizer for PixArt")?;
        // PixArt-Σ uses the SDXL VAE, whose decoder overflows F16 →
        // all-black on Metal/CUDA. Swap in madebyollin's F16-stable
        // retrained drop-in for non-CPU (exactly as SdCore does for SDXL);
        // CPU keeps the repo VAE at F32, where the stock VAE is fine.
        let vae_path = if matches!(req.device, Device::Cpu) {
            crate::hf::download::get_file(&req.repo, "vae/diffusion_pytorch_model.safetensors")
                .await
                .context("downloading VAE weights for PixArt")?
        } else {
            const VAE_FIX_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";
            crate::hf::download::get_first_of(&[
                (VAE_FIX_REPO, "diffusion_pytorch_model.safetensors"),
                (VAE_FIX_REPO, "sdxl_vae.safetensors"),
                (VAE_FIX_REPO, "sdxl.vae.safetensors"),
            ])
            .await
            .context(
                "downloading the SDXL fp16-fix VAE for PixArt-Σ \
                 (its stock VAE produces black images in F16)",
            )?
        };
        let dit_path = crate::hf::download::get_file(
            &req.repo,
            "transformer/diffusion_pytorch_model.safetensors",
        )
        .await
        .context("downloading DiT transformer weights for PixArt")?;
        dl.finish_with_message("✓ PixArt weights resolved");

        // v0.35 phase 4: merge LoRAs into a tempfile that replaces
        // `dit_path` for the VarBuilder. Mirrors the SD3 / Flux /
        // SD-family pattern (`std::env::temp_dir()` + PID + nanos
        // for uniqueness; OS sweep handles cleanup — same trade-off
        // those pipelines make to keep the tempfile alive for the
        // lifetime of the mmap).
        let dit_load_path: std::path::PathBuf = if req.loras.is_empty() {
            dit_path.clone()
        } else {
            let merge_spinner = progress::spinner(&format!(
                "Merging {} PixArt LoRA(s) into DiT", req.loras.len()
            ));
            let out_path = std::env::temp_dir().join(format!(
                "plakat-pixart-lora-merged-{}-{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            let (n_mod, n_total) =
                crate::pipelines::pixart_lora::merge_pixart_loras_into_weights(
                    &dit_path,
                    &out_path,
                    &req.loras,
                    req.lora_scale,
                    &req.device,
                )?;
            merge_spinner.finish_with_message(format!(
                "✓ PixArt LoRA merge: {n_mod}/{n_total} target groups applied"
            ));
            out_path
        };

        let build = progress::spinner("Loading T5-XXL text encoder");
        let t5_cfg_str = std::fs::read_to_string(&t5_cfg_path)
            .with_context(|| format!("read T5 config {}", t5_cfg_path.display()))?;
        let t5_cfg: t5::Config =
            serde_json::from_str(&t5_cfg_str).context("parse T5 config (PixArt)")?;
        // T5-XXL overflows F16 — its FFN activations exceed F16's ~65k
        // ceiling → inf caption embeddings (HF's T5 clamps for f16; candle's
        // does not). Run T5 in BF16 (same 9.4 GB footprint as F16, but
        // F32-range so no overflow) on non-CPU; CPU keeps F32.
        let t5_dtype = if matches!(req.device, Device::Cpu) {
            dtype
        } else {
            DType::BF16
        };
        let t5_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&t5_shards, t5_dtype, &req.device)?
        };
        let t5_enc = t5::T5EncoderModel::load(t5_vb, &t5_cfg)
            .context("building T5-XXL encoder for PixArt")?;
        let t5_tok =
            Tokenizer::from_file(&t5_tok_path).map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
        build.finish_with_message("✓ T5-XXL ready");

        let dit_build = progress::spinner("Loading DiT-XL/2 backbone");
        // v0.36 phase 2: pick the right config from the repo path.
        // 512-MS shares the architecture with 1024-MS — only the
        // sample_size differs (informational, see `pixart_dit::
        // Config::sigma_xl_512` doc).
        let dit_cfg = DitConfig::for_pixart_repo(&req.repo);
        // The DiT overflows F16 (activations exceed F16's ~65k ceiling →
        // inf → NaN → all-black on Metal). Run it in F32 for numerical
        // stability; T5 stays F16 to fit memory. ~2.4 GB extra, fits 24 GB.
        let dit_dtype = DType::F32;
        let dit_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dit_load_path.as_path()],
                dit_dtype,
                &req.device,
            )?
        };
        let dit = PixArtSigmaXL::new(dit_cfg.clone(), dit_vb)
            .context("building DiT-XL/2 from PixArt checkpoint")?;
        dit_build.finish_with_message("✓ DiT-XL/2 ready");

        let vae_build = progress::spinner("Loading PixArt VAE");
        let sd_cfg = StableDiffusionConfig::sdxl(None, None, None);
        let vae = match req.vae_cache {
            Some(arc) => {
                tracing::info!(
                    target: "plakat",
                    "PixArt: reusing cached VAE (skipping {} build)",
                    vae_path.display()
                );
                arc
            }
            None => Arc::new(sd_cfg.build_vae(&vae_path, &req.device, dtype)?),
        };
        vae_build.finish_with_message("✓ VAE ready");

        Ok(Self {
            device: req.device,
            dtype,
            t5_enc,
            t5_tok,
            dit,
            dit_cfg,
            vae,
            sd_cfg,
        })
    }

    /// Capture named intermediate tensors for `plakat verify` Tier 1 (RFC_VERIFY). Additive.
    ///
    /// - `dit.pos_embed` — the 2D sin-cos patch positional embedding for `(width, height)`,
    ///   computed exactly as the DiT forward does. **The pos-embed scaling (H/W half-swap +
    ///   `base_size`/interpolation) was a real DiT bug**; comparing this to the diffusers
    ///   golden pins the formula. Prompt-independent.
    pub fn capture_intermediates(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
        wanted: &std::collections::HashSet<String>,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        let mut out = std::collections::HashMap::new();
        // T5 caption embedding WITH the padding attention mask (the v2.1 fix). Corresponds to
        // diffusers `text_encoder(ids, attention_mask=mask)[0]`. Real prompt (not deterministic)
        // — this is where the missing-mask bug lived (real tokens attending to pad).
        if wanted.contains("t5.hidden") {
            out.insert("t5.hidden".to_string(), self.encode_prompt(prompt)?.0);
        }
        if wanted.contains("dit.pos_embed") {
            let cfg = &self.dit_cfg;
            let (lh, lw) = ((height / 8) as usize, (width / 8) as usize);
            let (gh, gw) = self.dit.patch_embed.grid_dims(lh, lw);
            let interp = (((cfg.sample_size * cfg.patch_size) as f32) / 64.0).floor().max(1.0);
            let pe = crate::pipelines::pixart_dit::build_2d_sincos_pos_embed(
                cfg.hidden_size,
                gh,
                gw,
                cfg.sample_size,
                interp,
                &self.device,
                self.dtype,
            )?;
            out.insert("dit.pos_embed".to_string(), pe);
        }
        // DiT block-0 tap: run patch-embed + adaLN + caption-proj + block[0] on a shared
        // deterministic latent + FIXED timestep + DETERMINISTIC caption (LCG) + a DETERMINISTIC
        // caption MASK (first half real, second half "pad"). The caption is synthetic (not T5)
        // so this isolates the DiT block math; the mask exercises the v2.1 cross-attention pad
        // masking (image tokens must not attend to the masked keys). The dumper feeds the
        // byte-identical caption + mask (encoder_attention_mask on the diffusers side).
        if wanted.contains("dit.block0") {
            let cfg = &self.dit_cfg;
            let (lh, lw) = ((height / 8) as usize, (width / 8) as usize);
            let latent = crate::verify::deterministic_latent(4, lh, lw, &self.device, self.dtype)?;
            // Deterministic T5-caption stand-in (seed 2): (1, max_caption_tokens, caption_channels).
            let caption = crate::verify::deterministic_tensor(
                &[1, cfg.max_caption_tokens, cfg.caption_channels], 2, &self.device, self.dtype,
            )?;
            // Deterministic caption mask: 1 for the first half, 0 for the second (synthetic pad).
            let half = cfg.max_caption_tokens / 2;
            let mut m: Vec<f32> = vec![1.0; half];
            m.resize(cfg.max_caption_tokens, 0.0);
            let cap_mask = Tensor::new(m.as_slice(), &self.device)?.unsqueeze(0)?.to_dtype(self.dtype)?;
            let timestep = Tensor::full(500.0f32, (1usize,), &self.device)?.to_dtype(self.dtype)?;
            // Σ micro-conditioning at the fixture resolution (square → aspect 1). Matches the
            // real forward's `res` (1,2)=[h,w] and `asp` (1,2)=[1,1].
            let res = Tensor::new(&[height as f32, width as f32], &self.device)?
                .reshape((1, 2))?.to_dtype(self.dtype)?;
            let asp = Tensor::new(&[1.0f32, 1.0f32], &self.device)?.reshape((1, 2))?.to_dtype(self.dtype)?;
            let b0 = self.dit.capture_block0(&latent, &timestep, &caption, &res, &asp, Some(&cap_mask))?;
            out.insert("dit.block0".to_string(), b0);
        }
        Ok(out)
    }

    /// Tokenize a prompt + forward through T5 **with a padding attention mask**. Returns
    /// `(1, max_caption_tokens, 4096)` right-padded with zeros to the training sequence length.
    ///
    /// The mask (`1` for the real tokens incl. EOS, `0` for padding) is the fix for the
    /// caption-without-mask bug: without it T5 self-attention lets real tokens attend to the
    /// (many) pad tokens, drifting the real-token embeddings to corr ~0.7 vs the correct
    /// masked output — matching diffusers, which always passes `attention_mask`.
    /// Returns `(t5_hidden (1, max_tokens, 4096), attention_mask (1, max_tokens))`. The mask
    /// (F32, 1 real / 0 pad) is reused for the DiT cross-attention (image tokens must not
    /// attend to pad-position caption keys either — diffusers `encoder_attention_mask`).
    fn encode_prompt(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let max_tokens = self.dit_cfg.max_caption_tokens;
        let mut ids = self
            .t5_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("T5 encode: {e}"))?
            .get_ids()
            .to_vec();
        ids.truncate(max_tokens);
        let real = ids.len(); // real tokens incl. EOS, before pad
        ids.resize(max_tokens, 0);
        let ids_t = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        // attention_mask: 1.0 for [0, real), 0.0 for pad — F32 like the T5 forward.
        let mut mask: Vec<f32> = vec![1.0; real];
        mask.resize(max_tokens, 0.0);
        let mask_t = Tensor::new(mask.as_slice(), &self.device)?.unsqueeze(0)?;
        // Keep the T5 output in F32 (the DiT runs F32 and the BF16→F16
        // round-trip would re-overflow the large embedding values).
        let hidden = self.t5_enc.forward_with_mask(&ids_t, &mask_t)?.to_dtype(DType::F32)?;
        Ok((hidden, mask_t))
    }

    /// End-to-end CFG denoise loop + VAE decode. Returns the raw
    /// RGB u8 buffer + (width, height) so the caller can compose
    /// metadata and write through `save_rgb_u8_with_metadata`.
    pub fn generate(
        &mut self,
        prompt: &str,
        negative: &str,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f64,
        seed: u64,
        scheduler_kind: SchedulerKind,
        hook: &mut Option<&mut dyn crate::pipelines::step_hook::StepHook>,
    ) -> Result<(Vec<u8>, u32, u32)> {
        anyhow::ensure!(
            width % 8 == 0 && height % 8 == 0,
            "PixArt requires width + height divisible by 8 (got {width}×{height})"
        );

        // v0.34 phase 1: device-aware seed prep.
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
        if let Err(e) = self.device.set_seed(prepared) {
            tracing::debug!(
                target: "plakat",
                "set_seed not supported ({e}); using global RNG"
            );
        }

        // ---- T5 encoding for CFG (positive + negative). ----
        let s = progress::spinner("Encoding T5 caption embeddings");
        let (pos_caption, pos_mask) = self.encode_prompt(prompt)?;
        let (neg_caption, neg_mask) = self.encode_prompt(negative)?;
        s.finish_with_message("✓ captions ready");

        // ---- Scheduler. ----
        let mut scheduler = build_scheduler(scheduler_kind, &self.sd_cfg, steps)?;
        let timesteps = scheduler.timesteps().to_vec();

        // ---- Initial noise. ----
        let lh = (height / 8) as usize;
        let lw = (width / 8) as usize;
        let init_sigma = scheduler.init_noise_sigma();
        let noise = Tensor::randn(0f32, 1f32, (1, 4, lh, lw), &self.device)?
            .to_dtype(self.dtype)?;
        let mut latents = (noise * init_sigma)?;

        // ---- Resolution + aspect conditioning (Σ-specific). ----
        // diffusers passes raw pixel dims for `resolution`; aspect is
        // `(1.0, height/width)` to match upstream.
        let res = Tensor::new(&[height as f32, width as f32], &self.device)?
            .reshape((1, 2))?
            .to_dtype(self.dtype)?;
        let asp = Tensor::new(&[1.0_f32, (height as f32) / (width as f32)], &self.device)?
            .reshape((1, 2))?
            .to_dtype(self.dtype)?;
        // CFG batch: replicate [neg, pos] along batch.
        let res_cfg = Tensor::cat(&[&res, &res], 0)?;
        let asp_cfg = Tensor::cat(&[&asp, &asp], 0)?;
        let caption_cfg = Tensor::cat(&[&neg_caption, &pos_caption], 0)?;
        // v2.1: caption cross-attention mask (CFG-batched like the caption) — image tokens
        // don't attend to pad-position caption keys (matches diffusers encoder_attention_mask).
        let mask_cfg = Tensor::cat(&[&neg_mask, &pos_mask], 0)?;

        // ---- Denoise loop. ----
        let bar = crate::ui::progress::step_bar(timesteps.len() as u64, "pixart");
        let n_steps = timesteps.len();
        for (step_i, &t) in timesteps.iter().enumerate() {
            let scaled = scheduler.scale_model_input(latents.clone(), t)?;
            // Replicate along batch for CFG: (2, 4, lh, lw).
            let scaled_cfg = Tensor::cat(&[&scaled, &scaled], 0)?;
            let t_tensor = Tensor::new(&[t as f32], &self.device)?
                .to_dtype(self.dtype)?
                .expand((2,))?;
            // The DiT runs in F32 (see load); cast inputs up and the
            // prediction back down to the pipeline dtype for the scheduler.
            let pred = self
                .dit
                .forward(
                    &scaled_cfg.to_dtype(DType::F32)?,
                    &t_tensor.to_dtype(DType::F32)?,
                    &caption_cfg.to_dtype(DType::F32)?,
                    &res_cfg.to_dtype(DType::F32)?,
                    &asp_cfg.to_dtype(DType::F32)?,
                    Some(&mask_cfg.to_dtype(DType::F32)?),
                )?
                .to_dtype(self.dtype)?;
            // learn_sigma=True → first 4 channels are noise; the
            // log-variance half is discarded (standard inference path).
            let noise_pred = pred.narrow(1, 0, 4)?;
            let chunks = noise_pred.chunk(2, 0)?;
            let neg = &chunks[0];
            let pos = &chunks[1];
            let guided = (neg + ((pos - neg)? * guidance)?)?;
            latents = scheduler.step(&guided, t, &latents)?;
            bar.inc(1);
            bar.set_message(format!("t={t}"));
            // RFC TUI-1 §0-R0-3: per-step hook (progress + cancel; no-op on None).
            // On Cancel, decode + return the partial; the caller stops the count loop.
            if crate::pipelines::step_hook::step(hook, step_i, n_steps)
                == crate::pipelines::step_hook::StepControl::Cancel
            {
                break;
            }
        }
        bar.finish_and_clear();

        // ---- VAE decode. ----
        // PixArt-Σ uses the SDXL VAE, whose latent-space scaling factor
        // is 0.13025 (per the repo's vae/config.json) — NOT the 0.18215
        // SD 1.5/2.1 constant.
        let _ = &self.sd_cfg; // kept on the struct for phase 3+ uses
        let vae_scale: f64 = 0.13025;
        let s = progress::spinner("Decoding latents → image");
        let decoded = self.vae.decode(&(&latents / vae_scale)?)?;
        let image = ((decoded / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?
            .to_dtype(DType::U8)?
            .i(0)?
            .permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;
        s.finish_with_message("✓ image decoded");

        Ok((buf, ow as u32, oh as u32))
    }
}

/// CLI entrypoint: parameters needed for one PixArt generation.
#[derive(Clone)]
pub struct RunRequest {
    pub model: String,
    pub device: Device,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub scheduler: SchedulerKind,
    pub out_dir: std::path::PathBuf,
    /// Count of images (per-image seed = base + idx).
    pub count: u32,
    /// v0.35 phase 4: LoRA stack (resolved or unresolved). `run()`
    /// resolves any unresolved specs before passing to
    /// `Pipeline::load`.
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    pub lora_scale: f32,
}

pub async fn run(req: RunRequest) -> Result<()> {
    run_hooked(req, None).await
}

/// As [`run`] with an optional per-step [`StepHook`](crate::pipelines::step_hook::StepHook)
/// (RFC TUI-1 §0-R0-3) for TUI progress + cancellation. `None` = the CLI path.
pub async fn run_hooked(
    req: RunRequest,
    mut hook: Option<&mut dyn crate::pipelines::step_hook::StepHook>,
) -> Result<()> {
    let repo = if req.model.contains('/') {
        req.model.clone()
    } else {
        crate::hf::resolve_alias(&req.model).to_string()
    };

    // v0.35 phase 4: resolve LoRA specs (local / hub / civitai) before
    // load. Mirrors the SD3 / Flux resolve-then-load pattern.
    let resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> = if req.loras.is_empty() {
        Vec::new()
    } else {
        let s = progress::spinner(&format!("Resolving {} PixArt LoRA(s)", req.loras.len()));
        let mut v = Vec::with_capacity(req.loras.len());
        for spec in &req.loras {
            v.push(spec.resolve().await?);
        }
        s.finish_with_message(format!("✓ resolved {} PixArt LoRA file(s)", v.len()));
        v
    };

    let mut pipeline = Pipeline::load(LoadRequest {
        repo,
        device: req.device.clone(),
        vae_cache: None, // v0.35 phase 2: scenario VAE-cache wiring lands in v0.36
        loras: resolved_loras,
        lora_scale: req.lora_scale,
    })
    .await?;

    let base_seed = req
        .seed
        .unwrap_or_else(|| rand::random::<u64>() & (u32::MAX as u64));

    std::fs::create_dir_all(&req.out_dir)
        .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

    // v0.35 phase 4: re-resolve loras once for metadata population.
    // Cheap on the second call — Civitai / HF cache short-circuits;
    // local LoraSpec::resolve is a path-exists check.
    let metadata_lora_stack: Vec<crate::imaging::metadata::LoraEntry> = req
        .loras
        .iter()
        .map(|s| s.to_entry())
        .collect();

    for idx in 0..req.count {
        let seed = base_seed.wrapping_add(idx as u64);
        crate::ui::progress::println(&format!(
            "  {} pixart {} of {} (seed={seed})",
            console::style("◆").cyan().bold(),
            idx + 1,
            req.count,
        ));
        let (buf, ow, oh) = pipeline.generate(
            &req.prompt,
            &req.negative,
            req.width,
            req.height,
            req.steps,
            req.guidance,
            seed,
            req.scheduler,
            &mut hook,
        )?;

        // Build sidecar metadata. PixArt now emits the full v0.34
        // phase 0 schema (model + size + steps + scheduler + LoRA
        // stack with source kind per entry). Other PixArt-specific
        // fields (Σ resolution/aspect conditioning, T5 sequence
        // length used) land in v0.36+ alongside non-t2i metadata
        // build-out.
        let mut m = crate::imaging::metadata::GenerationMetadata::new(
            req.prompt.clone(),
            req.model.clone(),
            seed,
            req.steps,
            req.guidance,
            format!("{:?}", req.scheduler).to_lowercase(),
            req.width,
            req.height,
        );
        m.negative = req.negative.clone();
        if !metadata_lora_stack.is_empty() {
            m.with_lora_stack(metadata_lora_stack.clone());
            m.lora_scale = Some(req.lora_scale);
        }

        let out_path = req.out_dir.join(format!("plakat-pixart-{seed}.png"));
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, ow, oh, &out_path, &m)?;
        crate::ui::progress::println(&format!(
            "  {} {}",
            console::style("✓").green().bold(),
            out_path.display()
        ));
        // RFC TUI-1 §0-R0-3: a cancelled denoise saved this partial; stop.
        if crate::pipelines::step_hook::is_cancelled(&hook) {
            break;
        }
    }

    Ok(())
}

// =====================================================================
// Style / subject LoRA training (v1.10.0).
//
// Mirrors `sd3::train_style_lora` (also a DiT) but with the PixArt-Σ
// objective: **DDPM ε-prediction** (NOT rectified-flow velocity), the
// Σ resolution/aspect conditioning, BF16 T5 (done in Phase A's encode),
// and the SDXL VAE's 0.13025 latent scale. The trained adapters are
// the same per-block attention projections the `pixart_lora` merge path
// already loads, so the output `.safetensors` is a plain diffusers-PEFT
// file usable via `--lora`.
// =====================================================================

/// PixArt-Σ style/subject LoRA training request. Field-parallel with
/// `sd3::StyleTrainRequest` so the CLI dispatch is uniform.
pub struct StyleTrainRequest {
    /// HF repo (resolved). The Σ checkpoint whose DiT is fine-tuned.
    pub repo: String,
    pub device: Device,
    pub images: Vec<std::path::PathBuf>,
    pub trigger: String,
    pub rank: usize,
    pub steps: usize,
    pub lr: f64,
    pub size: u32,
    pub out: std::path::PathBuf,
    pub checkpoint_every: Option<usize>,
    pub log_every: usize,
    pub resume_from: Option<std::path::PathBuf>,
    /// DreamBooth prior preservation (optional).
    pub class_images: Vec<std::path::PathBuf>,
    pub class_prompt: Option<String>,
    pub prior_weight: f32,
}

/// PixArt-Σ's β-schedule → cumulative ᾱ (length 1000). Σ uses the
/// diffusers default **linear** betas (β_start 1e-4 → β_end 2e-2),
/// matching the inference scheduler — not SD's scaled-linear.
fn pixart_alphas_cumprod() -> Vec<f64> {
    let (n, bs, be) = (1000usize, 0.0001f64, 0.02f64);
    let mut acc = 1.0;
    (0..n)
        .map(|i| {
            let beta = bs + (be - bs) * (i as f64 / (n - 1) as f64);
            acc *= 1.0 - beta;
            acc
        })
        .collect()
}

/// Numbered checkpoint path (`<stem>-step<N>.<ext>`), mirroring the SD /
/// SD3 trainers. `PLAKAT_TRAIN_SINGLE_FILE=1` overwrites `--out` instead.
fn ckpt_path(out: &std::path::Path, step: usize) -> std::path::PathBuf {
    if std::env::var_os("PLAKAT_TRAIN_SINGLE_FILE").is_some() {
        return out.to_path_buf();
    }
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("lora");
    let ext = out.extension().and_then(|s| s.to_str()).unwrap_or("safetensors");
    out.with_file_name(format!("{stem}-step{step}.{ext}"))
}

/// ~10 evenly-spaced checkpoints (min every 30) unless `--checkpoint-every`.
fn ckpt_interval(every: Option<usize>, total_steps: usize) -> usize {
    every.filter(|&n| n > 0).unwrap_or_else(|| (total_steps / 10).max(30))
}

/// SDXL-VAE latent scale used at both encode (train) and decode (infer).
const PIXART_VAE_SCALE: f64 = 0.13025;

/// Write trained DiT attention adapters as a diffusers-PEFT LoRA. PixArt
/// has no fused QKV, so each registry key (`transformer_blocks.{i}.
/// attn{1,2}.{to_q,to_k,to_v,to_out.0}.weight`) maps directly to a
/// `transformer.<leaf>.lora_A/lora_B/alpha` triple — exactly the logical
/// names `pixart_lora::resolve_target` accepts.
fn save_pixart_peft_lora(
    adapters: &[(String, Var, Var)],
    rank: usize,
    out: &std::path::Path,
) -> Result<()> {
    use std::collections::HashMap;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let alpha = Tensor::new(rank as f32, &Device::Cpu)?;
    for (key, a, b) in adapters {
        let leaf = key.strip_suffix(".weight").unwrap_or(key);
        let base = format!("transformer.{leaf}");
        let a_t = a.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        let b_t = b.as_tensor().to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        tensors.insert(format!("{base}.lora_A.weight"), a_t);
        tensors.insert(format!("{base}.lora_B.weight"), b_t);
        tensors.insert(format!("{base}.alpha"), alpha.clone());
    }
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    crate::pipelines::atomic_safetensors_save(&tensors, out)
        .with_context(|| format!("writing PixArt LoRA {}", out.display()))?;
    Ok(())
}

/// Load a PixArt-PEFT checkpoint back into the live adapter Vars (resume).
/// Inverse of `save_pixart_peft_lora` — direct per-leaf key match.
fn load_pixart_peft_into_adapters(
    adapters: &[(String, Var, Var)],
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
        let leaf = key.strip_suffix(".weight").unwrap_or(key);
        let base = format!("transformer.{leaf}");
        let a_loaded = get(&format!("{base}.lora_A.weight"))?;
        let b_loaded = get(&format!("{base}.lora_B.weight"))?;
        a.set(&a_loaded.to_dtype(a.as_tensor().dtype())?)?;
        b.set(&b_loaded.to_dtype(b.as_tensor().dtype())?)?;
    }
    Ok(())
}

/// `plakat style train --base pixart`: fine-tune a PixArt-Σ style (or,
/// with `--class-dir`, subject) LoRA. Phase A encodes the images +
/// trigger with the full pipeline (T5 BF16 + VAE) then drops it; Phase B
/// reloads only the DiT in F32 with trainable adapters; Phase C runs the
/// DDPM-ε loop; Phase D writes diffusers-PEFT safetensors.
///
/// **Memory-bound on 24 GB.** Phase A's peak is dominated by T5-XXL
/// (4.7 B): loading it transiently holds the mmap'd source *and* the
/// in-memory copy, and on Metal a unified-memory duplicate too — pushing
/// the footprint past 32 GB on the canonical Σ checkpoint. On a 24 GB box
/// this swap-thrashes rather than cleanly OOM-ing (the kernel keeps
/// swapping, so [`crate::memwatch::MemoryGuard`]'s sustained-CRITICAL
/// signal never trips). The code path is correct and the same family as
/// the verified SD3.5 trainer; the showcase run wants ≥ 36 GB unified or a
/// CUDA box. Same memory class as SD3.5 DreamBooth (carried debt).
pub async fn train_style_lora(req: StyleTrainRequest) -> Result<()> {
    use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

    let device = req.device.clone();
    let tag = "pixart-style-train";

    // --- Phase A: encode images + caption(s) with the full pipeline, drop it.
    tracing::info!(
        "{tag}: encoding {} image(s) + caption \"{}\"",
        req.images.len(),
        req.trigger
    );
    let (latents, caption, class_data) = {
        let mut pipe = Pipeline::load(LoadRequest {
            repo: req.repo.clone(),
            device: device.clone(),
            vae_cache: None,
            loras: Vec::new(),
            lora_scale: 1.0,
        })
        .await?;
        let pdtype = pipe.dtype;
        let encode_imgs = |pipe: &mut Pipeline,
                           imgs: &[std::path::PathBuf]|
         -> Result<Vec<Tensor>> {
            let mut v = Vec::with_capacity(imgs.len());
            for img in imgs {
                let px = crate::imaging::preprocess::sd_image_tensor(
                    img.as_path(),
                    req.size,
                    req.size,
                    &device,
                    pdtype,
                )?;
                // SDXL VAE has no shift; scale by 0.13025 into DiT latent
                // space, then up to F32 (the DiT trains in F32).
                let z = pipe.vae.encode(&px)?.sample()?;
                v.push((z * PIXART_VAE_SCALE)?.to_dtype(DType::F32)?);
            }
            Ok(v)
        };
        let caption = pipe.encode_prompt(&req.trigger)?.0; // (1, max_tokens, 4096) F32
        let latents = encode_imgs(&mut pipe, &req.images)?;
        let class_data = if req.class_images.is_empty() {
            None
        } else {
            let cp = req.class_prompt.as_deref().ok_or_else(|| {
                anyhow!("prior preservation: --class-prompt is required when class images are given")
            })?;
            let ccap = pipe.encode_prompt(cp)?.0;
            let clats = encode_imgs(&mut pipe, &req.class_images)?;
            Some((clats, ccap))
        };
        (latents, caption, class_data)
    }; // T5 (BF16) + DiT + VAE dropped here → freed

    // --- Phase B: reload the DiT in F32, install trainable adapters.
    tracing::info!("{tag}: loading DiT (F32) for training");
    let dit_path = crate::hf::download::get_file(
        &req.repo,
        "transformer/diffusion_pytorch_model.safetensors",
    )
    .await
    .context("downloading DiT transformer weights for PixArt training")?;
    let dit_cfg = DitConfig::for_pixart_repo(&req.repo);
    let dit_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[dit_path.as_path()],
            DType::F32,
            &device,
        )?
    };
    let dit = PixArtSigmaXL::new(dit_cfg, dit_vb).context("building DiT for training")?;
    let adapters = dit.install_train_adapters(req.rank, 1.0, &device)?;
    tracing::info!(
        "{tag}: {} trainable attention adapters (rank {})",
        adapters.len(),
        req.rank
    );
    let vars: Vec<Var> = adapters
        .iter()
        .flat_map(|(_, a, b)| [a.clone(), b.clone()])
        .collect();
    let mut opt = AdamW::new(vars.clone(), ParamsAdamW { lr: req.lr, ..Default::default() })?;

    // Σ resolution + aspect conditioning. Square training → asp (1, h/w=1).
    let res = Tensor::new(&[req.size as f32, req.size as f32], &device)?
        .reshape((1, 2))?;
    let asp = Tensor::new(&[1.0_f32, 1.0_f32], &device)?.reshape((1, 2))?;

    // --- Phase C: DDPM-ε loop. x_t = √ᾱ·x0 + √(1-ᾱ)·ε; predict ε (first 4 ch).
    let abar = pixart_alphas_cumprod();
    let n = latents.len().max(1);
    let interval = ckpt_interval(req.checkpoint_every, req.steps);
    let start_step = match &req.resume_from {
        Some(ckpt) => {
            load_pixart_peft_into_adapters(&adapters, ckpt, &device)?;
            let s = crate::pipelines::sd_train::trainer::parse_resume_step(ckpt)
                .unwrap_or(0)
                .min(req.steps);
            if s >= req.steps {
                bail!(
                    "{tag}: --resume checkpoint at step {s} ≥ --steps {}; \
                     raise --steps to continue training",
                    req.steps
                );
            }
            tracing::info!("{tag}: resuming from {} at step {s}/{}", ckpt.display(), req.steps);
            s
        }
        None => 0,
    };
    let mut progress =
        crate::pipelines::train_progress::TrainProgress::new(req.steps, req.lr, interval);

    let eps_loss = |dit: &PixArtSigmaXL,
                    x0: &Tensor,
                    cap: &Tensor|
     -> Result<Tensor> {
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &device)?;
        let t = (Tensor::rand(0f32, 1f32, (1usize,), &device)?.to_vec1::<f32>()?[0] * 999.0)
            as usize;
        let a = abar[t];
        let x_t = ((x0 * a.sqrt())? + (&noise * (1.0 - a).sqrt())?)?;
        let t_vec = Tensor::full(t as f32, (1usize,), &device)?;
        // Training forwards without a caption mask (short trigger prompt; the LoRA/DreamBooth
        // trainers are a separate path). Inference masks — see `generate`.
        let pred = dit.forward(&x_t, &t_vec, cap, &res, &asp, None)?;
        // learn_sigma=True → first 4 channels are the ε prediction.
        let eps = pred.narrow(1, 0, 4)?;
        Ok((&eps - &noise)?.sqr()?.mean_all()?)
    };

    for step in start_step..req.steps {
        let x0 = &latents[step % n];
        let mut loss = eps_loss(&dit, x0, &caption)?;
        // DreamBooth prior preservation on an independent class sample.
        if let Some((class_lat, ccap)) = &class_data {
            let cn = class_lat.len().max(1);
            let closs = eps_loss(&dit, &class_lat[step % cn], ccap)?;
            loss = (&loss + (closs * req.prior_weight as f64)?)?;
        }
        let mut grads = loss.backward()?;
        crate::pipelines::lora_linear::clip_grad_norm(&mut grads, &vars, 1.0)?;
        opt.step(&grads)?;
        if step % req.log_every.max(1) == 0 || step + 1 == req.steps {
            tracing::info!("{}", progress.line(tag, step + 1, loss.to_scalar::<f32>()?));
        }
        if (step + 1) % interval == 0 && step + 1 != req.steps {
            let ckpt = ckpt_path(&req.out, step + 1);
            save_pixart_peft_lora(&adapters, req.rank, &ckpt)?;
            tracing::info!("{tag}: checkpoint @ step {} → {}", step + 1, ckpt.display());
        }
    }

    // --- Phase D: save diffusers-PEFT safetensors.
    save_pixart_peft_lora(&adapters, req.rank, &req.out)?;
    tracing::info!("{tag}: wrote {}", req.out.display());
    tracing::info!("{}", progress.finish(tag, &req.out));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// v0.36 phase 2: 512-MS alias resolves to the smaller checkpoint.
    #[test]
    fn alias_pixart_512_resolves_to_sigma_512_repo() {
        assert_eq!(
            crate::hf::resolve_alias("pixart-512"),
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        );
        assert_eq!(
            crate::hf::resolve_alias("pixart-sigma-512"),
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        );
    }

    /// v0.36 phase 3: 2K-MS alias resolves to the heavyweight
    /// checkpoint with KV-compression.
    #[test]
    fn alias_pixart_2k_resolves_to_sigma_2k_repo() {
        assert_eq!(
            crate::hf::resolve_alias("pixart-2k"),
            "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS"
        );
        assert_eq!(
            crate::hf::resolve_alias("pixart-sigma-2k"),
            "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS"
        );
    }

    // v0.36 phase 4: LCM-LoRA composition path + α-LCM rejection.

    /// Σ repos pass `is_pixart_sigma_repo`. Mixed-case + the three
    /// shipped variants (1024 / 512 / 2K) all succeed.
    #[test]
    fn is_pixart_sigma_repo_accepts_sigma_variants() {
        for repo in [
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
            "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
            "PIXART-ALPHA/PIXART-SIGMA-XL-2-1024-MS",
            // A community fork that follows the Σ naming convention.
            "user/some-sigma-finetune",
        ] {
            is_pixart_sigma_repo(repo).unwrap_or_else(|e| {
                panic!("{repo}: expected Ok, got {e}");
            });
        }
    }

    /// α / α-LCM repo paths bail with the LCM-LoRA composition
    /// pointer. The error text references `--lora` + `--scheduler
    /// lcm` so users get the exact recipe.
    #[test]
    fn is_pixart_sigma_repo_bails_on_alpha_lcm() {
        for alpha_repo in [
            "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            "PixArt-alpha/PixArt-XL-2-512x512",
        ] {
            let err = is_pixart_sigma_repo(alpha_repo).unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("PixArt-α"),
                "{alpha_repo}: error should reference PixArt-α, got: {msg}"
            );
            assert!(
                msg.contains("--lora") && msg.contains("--scheduler lcm"),
                "{alpha_repo}: error should point at LCM-LoRA composition, got: {msg}"
            );
            assert!(
                msg.contains("v0.37"),
                "{alpha_repo}: error should mention the v0.37 deferral, got: {msg}"
            );
        }
    }

    /// Unrecognised repo paths default to Ok (best-effort — users
    /// with their own forks shouldn't get spurious bails). The
    /// VarBuilder will surface mismatches downstream.
    #[test]
    fn is_pixart_sigma_repo_passes_unrecognised_paths() {
        is_pixart_sigma_repo("user/my-fork").unwrap();
        is_pixart_sigma_repo("local/path/to/model").unwrap();
    }

    /// LCM scheduler composes with PixArt's SD config — same build
    /// path SD-family / Flux / SD3 use. The scheduler trait's
    /// timesteps + init_noise_sigma are reachable. This pins the
    /// LCM-LoRA composition path: any PixArt run with
    /// `--scheduler lcm --steps 4` produces a valid scheduler.
    #[test]
    fn lcm_scheduler_composes_with_pixart_sd_config() {
        let sd_cfg = StableDiffusionConfig::sdxl(None, None, None);
        let scheduler =
            build_scheduler(SchedulerKind::Lcm, &sd_cfg, 4).expect(
                "LCM scheduler must build with PixArt's SD config and 4 steps",
            );
        let timesteps = scheduler.timesteps();
        assert_eq!(
            timesteps.len(),
            4,
            "LCM scheduler at --steps 4 must produce 4 timesteps",
        );
        // init_noise_sigma is a finite positive number (the
        // scheduler picks the right value for LCM-LoRA distillation).
        let sigma = scheduler.init_noise_sigma();
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "LCM init_noise_sigma must be finite + positive, got {sigma}"
        );
    }

    #[test]
    fn pixart_aliases_listed_in_all_known() {
        let known = crate::hf::all_known_aliases();
        assert!(known.contains(&"pixart"), "got {known:?}");
        assert!(known.contains(&"pixart-sigma"), "got {known:?}");
        assert!(known.contains(&"pixart-1024"), "got {known:?}");
        assert!(known.contains(&"pixart-512"), "got {known:?}");
        assert!(known.contains(&"pixart-sigma-512"), "got {known:?}");
        assert!(known.contains(&"pixart-2k"), "got {known:?}");
        assert!(known.contains(&"pixart-sigma-2k"), "got {known:?}");
    }

    /// v0.36 phase 2: variant detection routes both 1024 and 512
    /// repo strings through `Variant::PixArt` — the dispatch
    /// branch is shared because the architecture is identical.
    #[test]
    fn variant_detect_covers_both_pixart_sizes() {
        use crate::pipelines::t2i::Variant;
        assert_eq!(
            Variant::detect("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"),
            Variant::PixArt
        );
        assert_eq!(
            Variant::detect("PixArt-alpha/PixArt-Sigma-XL-2-512-MS"),
            Variant::PixArt
        );
    }

    #[test]
    fn run_request_carries_all_inference_fields() {
        let r = RunRequest {
            model: "pixart".into(),
            device: Device::Cpu,
            prompt: "a fox".into(),
            negative: "".into(),
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 4.5,
            seed: Some(42),
            scheduler: SchedulerKind::DpmppKarras,
            out_dir: std::path::PathBuf::from("/tmp/pixart-test"),
            count: 1,
            loras: Vec::new(),
            lora_scale: 1.0,
        };
        assert_eq!(r.prompt, "a fox");
        assert_eq!(r.width, 1024);
        assert_eq!(r.seed, Some(42));
        assert_eq!(r.count, 1);
        matches!(r.scheduler, SchedulerKind::DpmppKarras);
        assert!(r.loras.is_empty());
        assert_eq!(r.lora_scale, 1.0);
    }
}
