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
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    flux::{autoencoder as fae, sampling},
    stable_diffusion::clip as sdclip,
    t5,
};
// v0.12 phase 2a: use plakat's vendored Flux model (with the
// residual-aware forward hook) instead of candle's upstream
// flux::model. The vendored type is byte-identical to upstream when
// no residuals are passed, so the existing Flux generation path
// behaves the same.
use crate::pipelines::flux_inner as fmodel;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::ui::progress;

const CLIP_EOT: u32 = 49407;

#[derive(Clone, Copy, Debug)]
pub enum Variant {
    Schnell,
    Dev,
}

impl Variant {
    pub fn is_dev(self) -> bool {
        matches!(self, Self::Dev)
    }
    fn main_filename(self) -> &'static str {
        match self {
            Self::Schnell => "flux1-schnell.safetensors",
            Self::Dev => "flux1-dev.safetensors",
        }
    }
    fn t5_seq_len(self) -> usize {
        match self {
            Self::Schnell => 256,
            Self::Dev => 512,
        }
    }
    fn flux_config(self) -> fmodel::Config {
        match self {
            Self::Schnell => fmodel::Config::schnell(),
            Self::Dev => fmodel::Config::dev(),
        }
    }
    fn ae_config(self) -> fae::Config {
        match self {
            Self::Schnell => fae::Config::schnell(),
            Self::Dev => fae::Config::dev(),
        }
    }
    pub fn default_guidance(self) -> f64 {
        match self {
            Self::Schnell => 1.0,
            Self::Dev => 3.5,
        }
    }
    pub fn default_steps(self) -> usize {
        match self {
            Self::Schnell => 4,
            Self::Dev => 28,
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
    /// v0.12 phase 2b: optional Flux ControlNet weight repo + config +
    /// conditioning-image path. `None` runs Flux without a ControlNet.
    pub controlnet: Option<FluxControlNetLoad>,
    pub conditioning: Option<PathBuf>,
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
    /// v0.12 phase 2b: optional Flux ControlNet.
    pub controlnet: Option<FluxControlNetLoad>,
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
    t5_enc: t5::T5EncoderModel,
    t5_tok: Tokenizer,
    flux_model: fmodel::Flux,
    ae_model: fae::AutoEncoder,
    /// v0.12 phase 2b: optional Flux ControlNet + the conditioning
    /// image (already VAE-encoded + packed to the 64-d token shape).
    /// `controlnet_scale` is diffusers `controlnet_conditioning_scale`.
    controlnet: Option<crate::pipelines::flux_controlnet::FluxControlNet>,
    controlnet_scale: f32,
}

impl Pipeline {
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

        // ---------- download weights ----------
        let dl = progress::spinner(&format!("Downloading weights for {}", req.repo));
        let main_path = crate::hf::download::get_file(&req.repo, req.variant.main_filename())
            .await
            .with_context(|| format!("{}", req.variant.main_filename()))?;
        let ae_path = crate::hf::download::get_file(&req.repo, "ae.safetensors").await?;

        let clip_weights = crate::hf::download::get_first_of(&[
            (&req.repo, "text_encoder/model.fp16.safetensors"),
            (&req.repo, "text_encoder/model.safetensors"),
        ])
        .await?;
        let clip_tokenizer = crate::hf::download::get_first_of(&[
            (&req.repo, "tokenizer/tokenizer.json"),
            ("openai/clip-vit-large-patch14", "tokenizer.json"),
        ])
        .await?;

        let t5_shard1 = crate::hf::download::get_file(
            &req.repo,
            "text_encoder_2/model-00001-of-00002.safetensors",
        )
        .await?;
        let t5_shard2 = crate::hf::download::get_file(
            &req.repo,
            "text_encoder_2/model-00002-of-00002.safetensors",
        )
        .await?;
        let t5_config_path =
            crate::hf::download::get_file(&req.repo, "text_encoder_2/config.json").await?;
        let t5_tokenizer = crate::hf::download::get_file(&req.repo, "tokenizer_2/tokenizer.json")
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
        let t5_cfg: t5::Config = serde_json::from_str(&t5_cfg_str)
            .with_context(|| format!("parse T5 config from {}", t5_config_path.display()))?;
        let t5_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&t5_shard1, &t5_shard2], dtype, &req.device)?
        };
        let t5_enc = t5::T5EncoderModel::load(t5_vb, &t5_cfg)?;
        let t5_tok = Tokenizer::from_file(&t5_tokenizer)
            .map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
        build.finish_with_message("✓ text encoders ready");

        // ---------- merge Flux LoRAs (v0.12) ----------
        // When the caller supplied LoRAs, we merge them into the Flux
        // transformer safetensors first (writing to a temp file) and
        // then load the merged file. Same pattern plakat uses for SD
        // LoRA merging into the UNet — keeps candle's Flux loader
        // unchanged.
        let (effective_main_path, lora_tmp) = if req.loras.is_empty() {
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
        let flux_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&effective_main_path], dtype, &req.device)?
        };
        let flux_model = fmodel::Flux::new(&req.variant.flux_config(), flux_vb)?;
        let ae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&ae_path], dtype, &req.device)?
        };
        let ae_model = fae::AutoEncoder::new(&req.variant.ae_config(), ae_vb)?;
        load.finish_with_message("✓ models loaded");

        // ---------- Flux ControlNet (v0.12 phase 2b) ----------
        let (controlnet, controlnet_scale) = match req.controlnet {
            Some(cn) => {
                let spin = progress::spinner(&format!(
                    "Downloading + remapping Flux ControlNet {}/{}",
                    cn.repo, cn.file
                ));
                let net = crate::pipelines::flux_controlnet::load_from_hf(
                    &cn.repo,
                    &cn.file,
                    cn.cfg,
                    &req.device,
                    dtype,
                )
                .await?;
                spin.finish_with_message("✓ Flux ControlNet ready");
                (Some(net), cn.scale)
            }
            None => (None, 1.0),
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
            controlnet,
            controlnet_scale,
        })
    }

    /// Generate `req.count` images for one prompt. Reuses the loaded models.
    /// `&mut self` because T5's forward maintains a KV cache.
    pub fn generate(&mut self, req: &GenRequest) -> Result<()> {
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps());
        let guidance = req.guidance.unwrap_or_else(|| self.variant.default_guidance());
        let w = (req.width as usize / 16) * 16;
        let h = (req.height as usize / 16) * 16;
        if w == 0 || h == 0 {
            bail!("Flux requires width and height divisible by 16, both ≥ 16");
        }
        std::fs::create_dir_all(&req.out_dir)
            .with_context(|| format!("creating output dir {}", req.out_dir.display()))?;

        // ---------- encode prompt ----------
        let enc = progress::spinner("Encoding prompt");
        let (clip_pooled, t5_emb) = self.encode_prompt(&req.prompt)?;
        enc.finish_with_message("✓ prompt encoded");

        let ae_cfg = self.variant.ae_config();
        let lat_h = (h + 15) / 16;
        let lat_w = (w + 15) / 16;
        let image_seq_len = lat_h * lat_w;

        // ---------- ControlNet conditioning prep (v0.12 phase 2b) ----
        // VAE-encode the conditioning image (if any) and pack to the
        // same `(1, image_seq_len, 64)` token shape the main image
        // tokens use. Done once per `generate()` call — the same
        // conditioning is reused at every denoise step.
        let conditioning_packed: Option<Tensor> = match (
            self.controlnet.as_ref(),
            req.conditioning.as_deref(),
        ) {
            (Some(_), Some(path)) => {
                let spin = progress::spinner("Encoding ControlNet conditioning");
                let packed = self.encode_conditioning(path, h, w)?;
                spin.finish_with_message("✓ conditioning encoded");
                Some(packed)
            }
            (Some(_), None) => {
                tracing::warn!(
                    target: "plakat",
                    "Flux ControlNet loaded but no conditioning image supplied — \
                     running the pipeline without residuals."
                );
                None
            }
            _ => None,
        };

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

            let img = sampling::get_noise(1, h, w, &self.device)?.to_dtype(self.dtype)?;
            let state = sampling::State::new(&t5_emb, &clip_pooled, &img)?;

            let shift = if self.variant.is_dev() {
                Some((image_seq_len, 0.5_f64, 1.15_f64))
            } else {
                None
            };
            let timesteps = sampling::get_schedule(steps, shift);

            let bar = progress::step_bar(
                (timesteps.len().saturating_sub(1)) as u64,
                &format!("img {}/{}", idx + 1, req.count),
            );
            bar.set_message(format!("flow-match denoise, {steps} steps, seed={seed}"));

            // v0.12 phase 2b: custom denoise loop. When the pipeline
            // has a ControlNet AND a conditioning image, run the
            // ControlNet each step to produce DoubleStream residuals,
            // then run Flux with those residuals. Otherwise this is
            // the same flow-matching integration candle's
            // `sampling::denoise` does, just inlined.
            let denoised = self.denoise_with_optional_controlnet(
                &state,
                &timesteps,
                guidance,
                conditioning_packed.as_ref(),
                &bar,
            )?;
            bar.set_position(timesteps.len().saturating_sub(1) as u64);
            bar.finish_with_message("✓ denoised");

            let unpacked = sampling::unpack(&denoised, h, w)?;
            // BFL AE expects: x = decode((z / scale) + shift)
            let pre_decode = ((&unpacked / ae_cfg.scale_factor)? + ae_cfg.shift_factor)?;
            let decoded = self.ae_model.decode(&pre_decode)?;
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
    fn encode_prompt(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
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
        let mut t5_ids = self
            .t5_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("T5 encode: {e}"))?
            .get_ids()
            .to_vec();
        t5_ids.truncate(t5_seq_len);
        t5_ids.resize(t5_seq_len, 0);
        let t5_ids_t = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let t5_emb = self.t5_enc.forward(&t5_ids_t)?.to_dtype(self.dtype)?;
        Ok((clip_pooled, t5_emb))
    }

    /// v0.12 phase 2b: load + VAE-encode + pack a Flux ControlNet
    /// conditioning image. Output shape `(1, image_seq_len, 64)` —
    /// same as Flux's `State::new` img packing, ready to flow into
    /// `FluxControlNet::forward`.
    fn encode_conditioning(
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
        // Pack 16-channel latent to (1, image_seq_len, 64) — the
        // same pixel-unshuffle + flatten dance State::new does.
        let (_bsz, c, lh, lw) = z.dims4()?;
        if c != 16 {
            anyhow::bail!(
                "Flux AE encoded to {c} channels — expected 16. Conditioning prep aborted."
            );
        }
        let packed = z
            .reshape((1, c, lh / 2, 2, lw / 2, 2))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((1, lh / 2 * lw / 2, c * 4))?;
        Ok(packed)
    }

    /// v0.12 phase 2b: flow-matching denoise loop that runs the
    /// ControlNet per step (when present + conditioning supplied)
    /// and threads its residuals into the main Flux's
    /// `forward_with_residuals`. Identical to candle's
    /// `sampling::denoise` when no ControlNet is engaged.
    fn denoise_with_optional_controlnet(
        &self,
        state: &sampling::State,
        timesteps: &[f64],
        guidance: f64,
        conditioning_packed: Option<&Tensor>,
        bar: &indicatif::ProgressBar,
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let dev = state.img.device();
        let guidance_t = Tensor::full(guidance as f32, b_sz, dev)?;
        let mut img = state.img.clone();
        for (step_i, window) in timesteps.windows(2).enumerate() {
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
            // ControlNet residuals (DoubleStream only in this phase).
            let residuals: Option<Vec<Tensor>> = match (
                self.controlnet.as_ref(),
                conditioning_packed,
            ) {
                (Some(net), Some(cond)) => Some(net.forward(
                    &img,
                    cond,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    self.controlnet_scale,
                )?),
                _ => None,
            };
            let pred = self.flux_model.forward_with_residuals(
                &img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &t_vec,
                &state.vec,
                Some(&guidance_t),
                residuals.as_deref(),
                None,
            )?;
            img = (img + pred * (t_prev - t_curr))?;
            bar.set_position(step_i as u64);
        }
        Ok(img)
    }
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
        controlnet: req.controlnet,
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
    })
}
