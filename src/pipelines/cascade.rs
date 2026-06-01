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
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::pipelines::cascade_stage_a::{Config as StageAConfig, StageAVae};
use crate::pipelines::cascade_unet::{Config as UnetConfig, StableCascadeUnet};
use crate::pipelines::scheduler::{SchedulerKind, build as build_scheduler};
use crate::pipelines::sdxl_clip::SdxlClipGTextTransformer;
use crate::pipelines::vendored_clip;
use crate::ui::progress;

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
    /// v0.37 phase 2: Stage B latent prior. ~1.5B-param UNet that
    /// takes Stage C's output + text conditioning and produces
    /// Stage A's latent. Variant-aware (Full vs Lite) — selected
    /// from the alias at load time.
    pub stage_b: StableCascadeUnet,
    /// v0.37 phase 3: Stage C high-res prior. ~3.6B-param UNet —
    /// the headline model. Text → 24×24×16 super-compressed prior
    /// latent. Variant-aware (Full vs Lite) — selected from the
    /// alias alongside Stage B.
    pub stage_c: StableCascadeUnet,
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
        // v0.37 phase 3: Stage C UNet weights. Diffusers calls
        // Stage C `prior`. This is the heaviest single file in the
        // pipeline (~3.6B params for the Full variant).
        let stage_c_w = crate::hf::download::get_first_of(&[
            (&req.repo, "prior/diffusion_pytorch_model.safetensors"),
            (&req.repo, "prior/diffusion_pytorch_model.fp16.safetensors"),
        ])
        .await
        .with_context(|| {
            format!("downloading Stage C UNet weights for Stable Cascade ({})", req.repo)
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

        let stage_a_build = progress::spinner("Loading Stage A VAE");
        let stage_a_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_a_w.as_path()], dtype, &req.device)?
        };
        let stage_a = StageAVae::new(StageAConfig::paella_v3(), stage_a_vb)
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

        // v0.37 phase 2: Stage B. `stage_b_for_alias` picks Full or
        // Lite based on the resolved repo path (substring "lite").
        let stage_b_build = progress::spinner("Loading Stage B UNet");
        let stage_b_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_b_load_path.as_path()], dtype, &req.device)?
        };
        let stage_b_cfg = UnetConfig::stage_b_for_alias(&req.repo);
        let stage_b = StableCascadeUnet::new(stage_b_cfg, stage_b_vb)
            .context("building Stage B UNet for Stable Cascade")?;
        stage_b_build.finish_with_message("✓ Stage B UNet ready");

        // v0.37 phase 3: Stage C. Same `stage_c_for_alias` Lite-vs-
        // Full routing rule as Stage B (substring "lite" → Lite).
        let stage_c_build = progress::spinner("Loading Stage C UNet (heaviest stage)");
        let stage_c_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[stage_c_load_path.as_path()], dtype, &req.device)?
        };
        let stage_c_cfg = UnetConfig::stage_c_for_alias(&req.repo);
        let stage_c = StableCascadeUnet::new(stage_c_cfg, stage_c_vb)
            .context("building Stage C UNet for Stable Cascade")?;
        stage_c_build.finish_with_message("✓ Stage C UNet ready");

        Ok(Self {
            device: req.device,
            dtype,
            clip_g_enc,
            clip_g_tok,
            stage_a,
            stage_b,
            stage_c,
        })
    }

    /// Tokenize a prompt + forward through CLIP-G. Returns the
    /// penultimate hidden states `(1, 77, 1280)` for cross-attn
    /// (matches the SDXL CLIP-G convention).
    ///
    /// v0.37 phase 4 scope (still current): pooled output is NOT
    /// used (Stable Cascade Stage C also consumes pooled text, but
    /// the upstream code conditions on it inside its `Effnet`
    /// embedding which is v0.38 phase 1 follow-through). For now
    /// we feed the penult only.
    fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let mut ids = self
            .clip_g_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("CLIP-G encode: {e}"))?
            .get_ids()
            .to_vec();
        ids.resize(77, CLIP_EOT);
        let ids_t = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let (penult, _pooled) = self.clip_g_enc.forward_for_sdxl(&ids_t)?;
        Ok(penult.to_dtype(self.dtype)?)
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
        stage_c_steps: usize,
        stage_b_steps: usize,
        guidance: f64,
        seed: u64,
        scheduler_kind: SchedulerKind,
    ) -> Result<(Vec<u8>, u32, u32)> {
        // v0.34 phase 1: device-aware seed prep.
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
        if let Err(e) = self.device.set_seed(prepared) {
            tracing::debug!(
                target: "plakat",
                "set_seed not supported ({e}); using global RNG"
            );
        }

        // ---- Text encoding for CFG (positive + negative). ----
        let s = progress::spinner("Encoding CLIP-G text embeddings");
        let pos_text = self.encode_prompt(prompt)?;
        let neg_text = self.encode_prompt(negative)?;
        let cfg_text = Tensor::cat(&[&neg_text, &pos_text], 0)?; // (2, 77, 1280)
        s.finish_with_message("✓ text encoded");

        // SD config carrier for scheduler::build (PixArt uses the
        // same trick — schedulers take the SD config but only need
        // its timestep schedule + sigma machinery).
        let sd_cfg = candle_transformers::models::stable_diffusion::StableDiffusionConfig::sdxl(
            None, None, None,
        );

        // ---- Stage C denoise: text → 24×24×16 prior latent. ----
        let mut c_scheduler = build_scheduler(scheduler_kind, &sd_cfg, stage_c_steps)?;
        let c_timesteps = c_scheduler.timesteps().to_vec();
        let c_init_sigma = c_scheduler.init_noise_sigma();
        let noise_c = Tensor::randn(0f32, 1f32, (1, 16, 24, 24), &self.device)?
            .to_dtype(self.dtype)?;
        let mut latent_c = (noise_c * c_init_sigma)?;
        let bar = crate::ui::progress::step_bar(c_timesteps.len() as u64, "cascade stage C");
        for &t in &c_timesteps {
            let scaled = c_scheduler.scale_model_input(latent_c.clone(), t)?;
            let cfg_latent = Tensor::cat(&[&scaled, &scaled], 0)?;
            let t_tensor = Tensor::new(&[t as f32], &self.device)?
                .to_dtype(self.dtype)?
                .expand((2,))?;
            let pred = self.stage_c.forward(&cfg_latent, &t_tensor, &cfg_text)?;
            let chunks = pred.chunk(2, 0)?;
            let neg = &chunks[0];
            let pos = &chunks[1];
            let guided = (neg + ((pos - neg)? * guidance)?)?;
            latent_c = c_scheduler.step(&guided, t, &latent_c)?;
            bar.inc(1);
            bar.set_message(format!("t={t}"));
        }
        bar.finish_and_clear();

        // ---- Stage B denoise: (text + Stage C effnet) →
        //      32×32×4 Stage A latent. ----
        //
        // v0.38 phase 1: Stage B is now conditioned on Stage C's
        // 16ch×24×24 prior latent ("effnet" conditioning) on top
        // of text. The effnet tensor is duplicated across the CFG
        // batch dim (upstream applies CFG to text only — effnet is
        // identical for positive and negative branches).
        let cfg_effnet = Tensor::cat(&[&latent_c, &latent_c], 0)?;
        let mut b_scheduler = build_scheduler(scheduler_kind, &sd_cfg, stage_b_steps)?;
        let b_timesteps = b_scheduler.timesteps().to_vec();
        let b_init_sigma = b_scheduler.init_noise_sigma();
        let noise_b = Tensor::randn(0f32, 1f32, (1, 4, 32, 32), &self.device)?
            .to_dtype(self.dtype)?;
        let mut latent_b = (noise_b * b_init_sigma)?;
        let bar = crate::ui::progress::step_bar(b_timesteps.len() as u64, "cascade stage B");
        for &t in &b_timesteps {
            let scaled = b_scheduler.scale_model_input(latent_b.clone(), t)?;
            let cfg_latent = Tensor::cat(&[&scaled, &scaled], 0)?;
            let t_tensor = Tensor::new(&[t as f32], &self.device)?
                .to_dtype(self.dtype)?
                .expand((2,))?;
            let pred = self.stage_b.forward_with_effnet(
                &cfg_latent,
                &t_tensor,
                &cfg_text,
                &cfg_effnet,
            )?;
            let chunks = pred.chunk(2, 0)?;
            let neg = &chunks[0];
            let pos = &chunks[1];
            let guided = (neg + ((pos - neg)? * guidance)?)?;
            latent_b = b_scheduler.step(&guided, t, &latent_b)?;
            bar.inc(1);
            bar.set_message(format!("t={t}"));
        }
        bar.finish_and_clear();

        // ---- Stage A decode: 32×32×4 latent → 1024×1024×3 image. ----
        let s = progress::spinner("Decoding latent → image (Stage A)");
        let decoded = self.stage_a.decode(&latent_b)?;
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
        init_image_path: &std::path::Path,
        prompt: &str,
        negative: &str,
        stage_c_steps: usize,
        stage_b_steps: usize,
        strength: f32,
        guidance: f64,
        seed: u64,
        scheduler_kind: SchedulerKind,
    ) -> Result<(Vec<u8>, u32, u32)> {
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
        if let Err(e) = self.device.set_seed(prepared) {
            tracing::debug!(
                target: "plakat",
                "set_seed not supported ({e}); using global RNG"
            );
        }

        // ---- Text encoding (CFG positive + negative). ----
        let s = progress::spinner("Encoding CLIP-G text embeddings");
        let pos_text = self.encode_prompt(prompt)?;
        let neg_text = self.encode_prompt(negative)?;
        let cfg_text = Tensor::cat(&[&neg_text, &pos_text], 0)?;
        s.finish_with_message("✓ text encoded");

        // ---- Init image encode through Stage A → (1, 4, 32, 32). ----
        // Stage A expects (1, 3, 1024, 1024) input in [-1, 1] — the
        // canonical 32× compression target. `sd_image_tensor` does
        // exactly that normalization.
        let s = progress::spinner("Encoding init image through Stage A");
        let init_pixels = crate::imaging::preprocess::sd_image_tensor(
            init_image_path,
            1024,
            1024,
            &self.device,
            self.dtype,
        )
        .with_context(|| {
            format!(
                "loading Cascade init image {}",
                init_image_path.display()
            )
        })?;
        let y_init = self.stage_a.encode(&init_pixels)?;
        s.finish_with_message("✓ Stage A encoded init");

        // Scheduler carrier (same SDXL config the Stage B/C denoise
        // loops use — schedulers only need the timestep schedule).
        let sd_cfg = candle_transformers::models::stable_diffusion::StableDiffusionConfig::sdxl(
            None, None, None,
        );

        // ---- Stage C denoise: text → 24×24×16 prior latent. ----
        // Full schedule regardless of strength — img2img conditions
        // Stage B output (the Stage A latent), not Stage C.
        let mut c_scheduler = build_scheduler(scheduler_kind, &sd_cfg, stage_c_steps)?;
        let c_timesteps = c_scheduler.timesteps().to_vec();
        let c_init_sigma = c_scheduler.init_noise_sigma();
        let noise_c = Tensor::randn(0f32, 1f32, (1, 16, 24, 24), &self.device)?
            .to_dtype(self.dtype)?;
        let mut latent_c = (noise_c * c_init_sigma)?;
        let bar = crate::ui::progress::step_bar(c_timesteps.len() as u64, "cascade stage C");
        for &t in &c_timesteps {
            let scaled = c_scheduler.scale_model_input(latent_c.clone(), t)?;
            let cfg_latent = Tensor::cat(&[&scaled, &scaled], 0)?;
            let t_tensor = Tensor::new(&[t as f32], &self.device)?
                .to_dtype(self.dtype)?
                .expand((2,))?;
            let pred = self.stage_c.forward(&cfg_latent, &t_tensor, &cfg_text)?;
            let chunks = pred.chunk(2, 0)?;
            let neg = &chunks[0];
            let pos = &chunks[1];
            let guided = (neg + ((pos - neg)? * guidance)?)?;
            latent_c = c_scheduler.step(&guided, t, &latent_c)?;
            bar.inc(1);
            bar.set_message(format!("t={t}"));
        }
        bar.finish_and_clear();

        // ---- Stage B denoise: truncated schedule starting from
        //      add_noise(y_init, noise, t_start). ----
        let cfg_effnet = Tensor::cat(&[&latent_c, &latent_c], 0)?;
        let mut b_scheduler = build_scheduler(scheduler_kind, &sd_cfg, stage_b_steps)?;
        let b_timesteps = b_scheduler.timesteps().to_vec();
        // Truncate: drop the first `(1 - strength) * len` schedule
        // entries. At strength=1.0 we keep them all (matches the
        // `generate` path); at strength=0.0 we drop everything and
        // emit the input verbatim through Stage A decode.
        let n_total = b_timesteps.len();
        let skip = ((1.0 - strength as f64) * n_total as f64).round() as usize;
        let skip = skip.min(n_total);
        let kept = &b_timesteps[skip..];

        let mut latent_b = if let Some(&t_start) = kept.first() {
            let noise_b = Tensor::randn(0f32, 1f32, y_init.shape(), &self.device)?
                .to_dtype(self.dtype)?;
            b_scheduler.add_noise(&y_init, noise_b, t_start)?
        } else {
            // strength == 0: skip Stage B entirely, decode y_init
            // straight through Stage A. Matches "no denoise" semantics
            // SD3 / Flux img2img already document.
            y_init.clone()
        };

        if !kept.is_empty() {
            let bar = crate::ui::progress::step_bar(
                kept.len() as u64,
                "cascade stage B (img2img)",
            );
            for &t in kept {
                let scaled = b_scheduler.scale_model_input(latent_b.clone(), t)?;
                let cfg_latent = Tensor::cat(&[&scaled, &scaled], 0)?;
                let t_tensor = Tensor::new(&[t as f32], &self.device)?
                    .to_dtype(self.dtype)?
                    .expand((2,))?;
                let pred = self.stage_b.forward_with_effnet(
                    &cfg_latent,
                    &t_tensor,
                    &cfg_text,
                    &cfg_effnet,
                )?;
                let chunks = pred.chunk(2, 0)?;
                let neg = &chunks[0];
                let pos = &chunks[1];
                let guided = (neg + ((pos - neg)? * guidance)?)?;
                latent_b = b_scheduler.step(&guided, t, &latent_b)?;
                bar.inc(1);
                bar.set_message(format!("t={t}"));
            }
            bar.finish_and_clear();
        }

        // ---- Stage A decode: 32×32×4 → 1024×1024×3. ----
        let s = progress::spinner("Decoding latent → image (Stage A)");
        let decoded = self.stage_a.decode(&latent_b)?;
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
    })
    .await?;

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
            req.stage_c_steps,
            req.stage_b_steps,
            req.guidance,
            seed,
            req.scheduler,
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
            stage_c_steps: 20,
            stage_b_steps: 10,
            guidance: 4.0,
            seed: Some(42),
            scheduler: SchedulerKind::DpmppKarras,
            out_dir: std::path::PathBuf::from("/tmp/cascade-test"),
            count: 1,
            loras: Vec::new(),
            lora_scale: 1.0,
        };
        assert_eq!(r.prompt, "a fox in a meadow");
        assert_eq!(r.stage_c_steps, 20);
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
}
