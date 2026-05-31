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
//!   `pipelines::pixart_dit` (v0.35 phase 1). adaLN-single + per-
//!   block scale_shift_table; KV-compression deferred to v0.36+
//!   (only used by the 2K-MS variant).
//! * **SD-family KL-VAE** — Arc-shared via the v0.34 phase 3 cache.
//! * **DPM++ sampler** via `pipelines::scheduler` (PixArt-Σ's
//!   recommendation).
//! * **Seed plumbing** through `pipelines::seeds::prepare_seed`
//!   (v0.34 phase 1 chokepoint).

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{StableDiffusionConfig, vae::AutoEncoderKL};
use candle_transformers::models::t5;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::pipelines::pixart_dit::{Config as DitConfig, PixArtSigmaXL};
use crate::pipelines::scheduler::{SchedulerKind, build as build_scheduler};
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

impl Pipeline {
    /// v0.35 phase 2: full load. Downloads T5 (3 shards) + DiT +
    /// VAE from the canonical diffusers layout, builds each module,
    /// returns the assembled pipeline.
    pub async fn load(req: LoadRequest) -> Result<Self> {
        let dtype = if matches!(req.device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        let dl = progress::spinner("Resolving PixArt Sigma weights");
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

        let dit_build = progress::spinner("Loading DiT-XL/2 backbone");
        let dit_cfg = DitConfig::sigma_xl_1024();
        let dit_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dit_load_path.as_path()],
                dtype,
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

    /// Tokenize a prompt + forward through T5. Returns `(1,
    /// max_caption_tokens, 4096)` left-padded with zeros to match
    /// the model's training-time sequence length.
    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor> {
        let max_tokens = self.dit_cfg.max_caption_tokens;
        let mut ids = self
            .t5_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("T5 encode: {e}"))?
            .get_ids()
            .to_vec();
        ids.truncate(max_tokens);
        ids.resize(max_tokens, 0);
        let ids_t = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden = self.t5_enc.forward(&ids_t)?.to_dtype(self.dtype)?;
        Ok(hidden)
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
        let pos_caption = self.encode_prompt(prompt)?;
        let neg_caption = self.encode_prompt(negative)?;
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

        // ---- Denoise loop. ----
        let bar = crate::ui::progress::step_bar(timesteps.len() as u64, "pixart");
        for &t in &timesteps {
            let scaled = scheduler.scale_model_input(latents.clone(), t)?;
            // Replicate along batch for CFG: (2, 4, lh, lw).
            let scaled_cfg = Tensor::cat(&[&scaled, &scaled], 0)?;
            let t_tensor = Tensor::new(&[t as f32], &self.device)?
                .to_dtype(self.dtype)?
                .expand((2,))?;
            let pred = self.dit.forward(
                &scaled_cfg,
                &t_tensor,
                &caption_cfg,
                &res_cfg,
                &asp_cfg,
            )?;
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
        }
        bar.finish_and_clear();

        // ---- VAE decode. ----
        // PixArt-Σ shares the SDXL VAE; latent-space scale is 0.18215
        // (same constant SD 1.5 / 2.1 / SDXL / SD3 use).
        let _ = &self.sd_cfg; // kept on the struct for phase 3+ uses
        let vae_scale: f64 = 0.18215;
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
    }

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

    #[test]
    fn pixart_aliases_listed_in_all_known() {
        let known = crate::hf::all_known_aliases();
        assert!(known.contains(&"pixart"), "got {known:?}");
        assert!(known.contains(&"pixart-sigma"), "got {known:?}");
        assert!(known.contains(&"pixart-1024"), "got {known:?}");
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
