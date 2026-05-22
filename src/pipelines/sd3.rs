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
//! ## Phase 1a scope
//!
//! Just t2i on `sd35-medium`. LoRA / GGUF / ControlNet / img2img all
//! land in subsequent phases. Sd35Large + Sd35LargeTurbo + Sd3Medium
//! variants land in phase 1b.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::Module;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    mmdit, stable_diffusion::clip as sdclip, stable_diffusion::vae as sdvae, t5,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::pipelines::sdxl_clip::SdxlClipGTextTransformer;
use crate::ui::progress;

/// CLIP EOT-token id — shared across CLIP-L and CLIP-G in diffusers.
const CLIP_EOT: u32 = 49407;

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
    fn mmdit_config(self) -> mmdit::model::Config {
        match self {
            Self::Sd3Medium => mmdit::model::Config::sd3_medium(),
            Self::Sd35Medium => mmdit::model::Config::sd3_5_medium(),
            // SD3.5 Large + Turbo share the same MMDiT shape; the
            // turbo distillation only changes the sampling schedule.
            Self::Sd35Large | Self::Sd35LargeTurbo => mmdit::model::Config::sd3_5_large(),
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
}

pub struct LoadRequest {
    pub variant: Variant,
    pub repo: String,
    pub device: Device,
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
    mmdit_model: mmdit::model::MMDiT,
    vae: sdvae::AutoEncoderKL,
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

        // ---------- MMDiT + VAE ----------
        let load = progress::spinner("Loading MMDiT + VAE");
        let mmdit_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&mmdit_path], dtype, &req.device)?
        };
        let mmdit_model = mmdit::model::MMDiT::new(&req.variant.mmdit_config(), false, mmdit_vb)?;

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
        })
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

        for idx in 0..req.count {
            let seed = req
                .seed
                .map(|s| s + idx as u64)
                .unwrap_or_else(rand::random)
                & (u32::MAX as u64);
            if let Err(e) = self.device.set_seed(seed) {
                tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
            }

            // Initial latent: pure Gaussian noise, (1, 16, lat_h, lat_w).
            let mut x = Tensor::randn(0f32, 1.0_f32, (1, 16, lat_h, lat_w), &self.device)?
                .to_dtype(self.dtype)?;

            // Linear timestep schedule [1.0 → 0.0] with the SD3 time
            // shift transform applied.
            let timesteps: Vec<f64> = (0..=steps)
                .map(|v| 1.0 - (v as f64 / steps as f64))
                .map(|t| shift_t(t, time_shift))
                .collect();

            let bar = progress::step_bar(
                (timesteps.len().saturating_sub(1)) as u64,
                &format!("img {}/{}", idx + 1, req.count),
            );
            bar.set_message(format!("flow-match denoise, {steps} steps, seed={seed}"));

            for (step_i, window) in timesteps.windows(2).enumerate() {
                let (t_curr, t_prev) = match window {
                    [a, b] => (*a, *b),
                    _ => continue,
                };
                // Double-batch [neg, pos] so the model forward
                // produces both directions in one call.
                let x_doubled = Tensor::cat(&[&x, &x], 0)?;
                // MMDiT timestep convention: scalar 0..1 broadcast to
                // batch. Pass the current t per batch row.
                let t_vec = Tensor::full(t_curr as f32, 2, &self.device)?;
                let pred_doubled =
                    self.mmdit_model
                        .forward(&x_doubled, &t_vec, &cfg_y, &cfg_ctx, None)?;
                let pred_neg = pred_doubled.i(0..1)?;
                let pred_pos = pred_doubled.i(1..2)?;
                let pred = (&pred_neg + ((pred_pos - &pred_neg)? * guidance)?)?;
                x = (x + pred * (t_prev - t_curr))?;
                bar.set_position(step_i as u64);
            }
            bar.set_position(timesteps.len().saturating_sub(1) as u64);
            bar.finish_with_message("✓ denoised");

            // VAE decode: undo the latent normalisation, decode,
            // convert to RGB u8.
            let pre_decode = ((&x / VAE_SCALE)? + VAE_SHIFT)?;
            let decoded = self.vae.decode(&pre_decode)?;
            let img_norm = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 0.5)?;
            let img_u8 = (img_norm * 255.0)?
                .to_dtype(DType::U8)?
                .i(0)?
                .permute((1, 2, 0))?;
            let (oh, ow, _) = img_u8.dims3()?;
            let buf = img_u8.flatten_all()?.to_vec1::<u8>()?;

            let out_path = req.out_dir.join(format!("plakat-sd3-{seed}.png"));
            crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, &out_path)?;
            crate::ui::progress::println(&format!("→ {}", out_path.display()));
        }
        Ok(())
    }

    /// Encode a single prompt into the `(y, context)` pair the MMDiT
    /// forward consumes.
    ///
    /// * `y` — `(1, 2048)` pooled embedding =
    ///   `[CLIP-G_pooled (1280) || CLIP-L_pooled (768)]`.
    /// * `context` — `(1, 77 + t5_seq, 4096)` text hidden states =
    ///   `[ pad([CLIP-L_hidden || CLIP-G_hidden], 2048→4096), T5_hidden ]`.
    fn encode_prompt(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        // ---------- CLIP-L ----------
        let mut clip_l_ids = self
            .clip_l_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("CLIP-L encode: {e}"))?
            .get_ids()
            .to_vec();
        clip_l_ids.resize(self.clip_l_cfg.max_position_embeddings, CLIP_EOT);
        let clip_l_eot_pos = clip_l_ids.iter().position(|&t| t == CLIP_EOT).unwrap_or(0);
        let clip_l_ids_t =
            Tensor::new(clip_l_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        // Full hidden states from CLIP-L (forward returns
        // `final_layer_norm` output).
        let clip_l_hidden = self.clip_l.forward(&clip_l_ids_t)?;
        let clip_l_pooled = clip_l_hidden
            .i((.., clip_l_eot_pos, ..))?
            .to_dtype(self.dtype)?;

        // ---------- CLIP-G (penultimate hidden + pooled via projection) ---
        let mut clip_g_ids = self
            .clip_g_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("CLIP-G encode: {e}"))?
            .get_ids()
            .to_vec();
        // CLIP-G uses the same 77-token budget as CLIP-L.
        clip_g_ids.resize(77, CLIP_EOT);
        let clip_g_ids_t =
            Tensor::new(clip_g_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let (clip_g_penult, clip_g_pooled) = self.clip_g.forward_for_sdxl(&clip_g_ids_t)?;
        let clip_g_penult = clip_g_penult.to_dtype(self.dtype)?;
        let clip_g_pooled = clip_g_pooled.to_dtype(self.dtype)?;

        // ---------- T5-XXL ----------
        let mut t5_ids = self
            .t5_tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("T5 encode: {e}"))?
            .get_ids()
            .to_vec();
        let t5_seq = self.variant.t5_seq_len();
        t5_ids.truncate(t5_seq);
        t5_ids.resize(t5_seq, 0);
        let t5_ids_t = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let t5_hidden = self.t5_enc.forward(&t5_ids_t)?.to_dtype(self.dtype)?;

        // ---------- Pooled (y) ----------
        // SD3 convention: CLIP-G pooled first (1280), CLIP-L pooled
        // second (768) → (1, 2048).
        let y = Tensor::cat(&[&clip_g_pooled, &clip_l_pooled], candle_core::D::Minus1)?;

        // ---------- Context ----------
        // CLIP-L's penultimate hidden state is what SD3 mixes with
        // CLIP-G's penultimate. We grab CLIP-L penultimate by running
        // until layer -2 (matching SDXL's convention).
        let (_clip_l_final, clip_l_penult) = {
            let (final_h, pen_h) = candle_transformers::models::stable_diffusion::clip::ClipTextTransformer
                ::forward_until_encoder_layer(&self.clip_l, &clip_l_ids_t, usize::MAX, -2)?;
            (final_h, pen_h)
        };
        let clip_l_penult = clip_l_penult.to_dtype(self.dtype)?;
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
fn shift_t(t: f64, shift: f64) -> f64 {
    if shift == 1.0 {
        t
    } else {
        shift * t / (1.0 + (shift - 1.0) * t)
    }
}

pub async fn run(req: Request) -> Result<()> {
    let mut p = Pipeline::load(LoadRequest {
        variant: req.variant,
        repo: req.repo,
        device: req.device,
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
}
