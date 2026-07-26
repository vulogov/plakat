//! Sana 1.6B (1024px) — the sixth model family (ROADMAP_4.5.0).
//!
//! Assembles the three verified components into an end-to-end text-to-image pipeline:
//!   * [`dc_ae`](super::dc_ae) — the DC-AE deep-compression autoencoder (32×, 32-ch), F32.
//!   * [`vendored_gemma2`](super::vendored_gemma2) — the Gemma-2-2B text encoder (+ Sana's CHI recipe).
//!   * [`sana_dit`](super::sana_dit) — the Linear-DiT, sampled with a flow-matching (Euler) loop.
//!
//! Precision: the DiT + Gemma run in the model's native **BF16** on GPU (F32 on CPU); the DC-AE
//! stays **F32** (its ReLU-linear attention is not self-normalizing → F16/BF16 NaNs). The DiT's own
//! linear-attention reduction is an internal F32 island regardless of weight dtype.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::pipelines::scheduler::SchedulerKind;

const MAX_SEQ: usize = 300;
const LATENT_CH: usize = 32;
const DCAE_SCALE: f64 = 0.41407;
const FLOW_SHIFT: f64 = 3.0;
/// Sana's "complex human instruction" — string-prepended to every prompt (Sana was trained with
/// it; alignment degrades without it). Verbatim from diffusers `SanaPipeline`.
const CHI: &[&str] = &[
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
];

/// A Sana text-to-image run. Mirrors [`pixart::RunRequest`](crate::pipelines::pixart::RunRequest).
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
    pub count: u32,
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    pub lora_scale: f32,
}

/// `shift·t / (1 + (shift−1)·t)` — the flow-matching sigma time-shift (diffusers `mu_t`; identical
/// to `sd3::shift_t`).
fn shift_t(t: f64, shift: f64) -> f64 {
    if shift == 1.0 { t } else { shift * t / (1.0 + (shift - 1.0) * t) }
}

/// FlowMatchEuler sigma schedule for `steps` inference steps, plus a terminal `0.0`.
/// `sigmas[i] · 1000` is the DiT timestep. Matches diffusers `FlowMatchEulerDiscreteScheduler`,
/// which applies the shift **twice**: once at init (the schedule floor becomes `shift_t(1/1000)`),
/// then again over the `linspace(1, floor)` in `set_timesteps`.
fn flow_sigmas(steps: usize, shift: f64) -> Vec<f64> {
    let floor = shift_t(1.0 / 1000.0, shift);
    let mut sig: Vec<f64> = (0..steps)
        .map(|i| {
            let lin = 1.0 - (i as f64) * (1.0 - floor) / ((steps - 1) as f64);
            shift_t(lin, shift)
        })
        .collect();
    sig.push(0.0);
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference from diffusers FlowMatchEulerDiscreteScheduler(shift=3.0).set_timesteps(20).
    #[test]
    fn flow_sigmas_match_diffusers() {
        let want = [
            1.0, 0.981875, 0.962386, 0.941373, 0.918652, 0.894003, 0.867172, 0.837855, 0.805689,
            0.770239, 0.730974, 0.687244, 0.63824, 0.582948, 0.520074, 0.447944, 0.364352, 0.26633,
            0.149787, 0.008929, 0.0,
        ];
        let got = flow_sigmas(20, 3.0);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "sigma[{i}]: got {g}, want {w}");
        }
    }
}

/// Download a component's safetensors shards from `repo/subfolder` (single file or a sharded
/// `{base}.safetensors.index.json`). `base` is `diffusion_pytorch_model` (transformer/vae) or
/// `model` (text_encoder).
async fn download_shards(repo: &str, subfolder: &str, base: &str) -> Result<Vec<std::path::PathBuf>> {
    let single = format!("{subfolder}/{base}.safetensors");
    if let Ok(p) = crate::hf::download::get_file(repo, &single).await {
        return Ok(vec![p]);
    }
    let index_name = format!("{subfolder}/{base}.safetensors.index.json");
    let index = crate::hf::download::get_file(repo, &index_name)
        .await
        .with_context(|| format!("Sana {subfolder}: neither {single} nor its index found"))?;
    let json: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index)?)?;
    let mut names: Vec<String> = json["weight_map"]
        .as_object()
        .context("index weight_map")?
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        out.push(crate::hf::download::get_file(repo, &format!("{subfolder}/{n}")).await?);
    }
    Ok(out)
}

/// The loaded Sana pipeline: DC-AE (F32) + Gemma-2 encoder + Linear-DiT + tokenizer.
///
/// `gemma` is an `Option` so the encoder (~5 GB) can be **freed after encoding** — the prompt is
/// fixed across the `--count` loop, so we encode once up front, drop Gemma, then denoise. This
/// keeps peak residency to DiT + DC-AE (relieves tight 24 GB Metal without a co-residence hit).
pub struct Pipeline {
    vae: super::dc_ae::AutoencoderDc,
    gemma: Option<super::vendored_gemma2::Model>,
    dit: Option<super::sana_dit::SanaTransformer>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dtype: DType,
    chi_tokens: usize,
}

impl Pipeline {
    pub async fn load(model: &str, device: Device) -> Result<Self> {
        let repo = crate::hf::resolve_alias(model).to_string();
        // DiT + Gemma run in BF16 on GPU (native), F32 on CPU. DC-AE always F32.
        let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };

        // Tokenizer (Gemma).
        let tok_path = crate::hf::download::get_file(&repo, "tokenizer/tokenizer.json")
            .await
            .context("Sana tokenizer.json")?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("loading Gemma tokenizer: {e}"))?;

        // DC-AE (F32).
        let vae_shards = download_shards(&repo, "vae", "diffusion_pytorch_model").await?;
        let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&vae_shards, DType::F32, &device)? };
        let vae = super::dc_ae::AutoencoderDc::load(vae_vb, DCAE_SCALE)?;

        // Gemma-2-2B encoder.
        let te_shards = download_shards(&repo, "text_encoder", "model").await?;
        let te_cfg_path = crate::hf::download::get_file(&repo, "text_encoder/config.json").await?;
        let gcfg: super::vendored_gemma2::Config =
            serde_json::from_reader(std::fs::File::open(&te_cfg_path)?).context("gemma config")?;
        let te_vb = unsafe { VarBuilder::from_mmaped_safetensors(&te_shards, dtype, &device)? };
        let gemma = super::vendored_gemma2::Model::new(false, &gcfg, te_vb)?;

        // Linear-DiT.
        let dit_shards = download_shards(&repo, "transformer", "diffusion_pytorch_model").await?;
        let dit_vb = unsafe { VarBuilder::from_mmaped_safetensors(&dit_shards, dtype, &device)? };
        let dit = super::sana_dit::SanaTransformer::load(dit_vb)?;

        let chi_tokens = tokenizer
            .encode(CHI.join("\n"), true)
            .map_err(|e| anyhow::anyhow!("tokenizing CHI: {e}"))?
            .len();

        Ok(Self { vae, gemma: Some(gemma), dit: Some(dit), tokenizer, device, dtype, chi_tokens })
    }

    /// Encode a prompt → `(embeds (1,300,2304), mask (1,300))` per Sana's `_get_gemma_prompt_embeds`
    /// + the `encode_prompt` `[0]+last-299` re-slice. `embeds`/`mask` are in the DiT dtype.
    fn encode_prompt(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let full = format!("{}{}", CHI.join("\n"), prompt);
        let max_len = self.chi_tokens + MAX_SEQ - 2;
        let enc = self
            .tokenizer
            .encode(full, true)
            .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
        // right-pad / truncate to max_len (Gemma pad id = 0).
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let mut mask: Vec<f32> = vec![1.0; ids.len()];
        ids.truncate(max_len);
        mask.truncate(max_len);
        while ids.len() < max_len {
            ids.push(0);
            mask.push(0.0);
        }
        let ids_t = Tensor::from_vec(ids.clone(), (1, max_len), &self.device)?;
        let mask_t = Tensor::from_vec(mask.clone(), (1, max_len), &self.device)?;
        // Gemma hidden states (all positions), then re-slice [0] + last 299.
        let gemma = self.gemma.as_mut().context("Sana text encoder already freed")?;
        let hidden = gemma.forward_hidden(&ids_t, Some(&mask_t))?;
        let bos = hidden.narrow(1, 0, 1)?;
        let tail = hidden.narrow(1, max_len - (MAX_SEQ - 1), MAX_SEQ - 1)?;
        let embeds = Tensor::cat(&[bos, tail], 1)?.to_dtype(self.dtype)?;
        // re-slice the mask the same way.
        let mut rmask = Vec::with_capacity(MAX_SEQ);
        rmask.push(mask[0]);
        rmask.extend_from_slice(&mask[max_len - (MAX_SEQ - 1)..]);
        let rmask_t = Tensor::from_vec(rmask, (1, MAX_SEQ), &self.device)?.to_dtype(self.dtype)?;
        Ok((embeds, rmask_t))
    }

    /// Encode the positive + negative prompts into a CFG caption batch `[uncond, cond]` and mask,
    /// then **free the Gemma encoder** (unused for the rest of the run). Call once per run.
    fn encode(&mut self, prompt: &str, negative: &str) -> Result<(Tensor, Tensor)> {
        let (pos, pos_m) = self.encode_prompt(prompt)?;
        let (neg, neg_m) = self.encode_prompt(negative)?;
        let caption = Tensor::cat(&[&neg, &pos], 0)?; // [uncond, cond]
        let mask = Tensor::cat(&[&neg_m, &pos_m], 0)?;
        self.gemma = None; // drop the ~5 GB encoder before the denoise loop
        Ok((caption, mask))
    }

    /// Run the flow-matching denoise loop → a raw latent `(1,32,h/32,w/32)`. Uses the DiT (must
    /// still be loaded). A per-step progress bar mirrors the other pipelines.
    fn denoise(&self, caption: &Tensor, mask: &Tensor, w: u32, h: u32, steps: usize, guidance: f64) -> Result<Tensor> {
        let (lw, lh) = (w as usize / 32, h as usize / 32);
        let dit = self.dit.as_ref().context("Sana DiT already freed")?;
        let sigmas = flow_sigmas(steps, FLOW_SHIFT);
        let mut latent = Tensor::randn(0f32, 1f32, (1, LATENT_CH, lh, lw), &self.device)?.to_dtype(DType::F32)?;

        let pb = indicatif::ProgressBar::new(steps as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template("  {spinner:.cyan} denoise [{bar:30.cyan/blue}] {pos}/{len}  {elapsed}")
                .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        for i in 0..steps {
            let t = sigmas[i] * 1000.0;
            let lat_in = Tensor::cat(&[&latent, &latent], 0)?.to_dtype(self.dtype)?; // (2,32,lh,lw)
            let ts = Tensor::from_vec(vec![t as f32; 2], 2, &self.device)?;
            let v = dit.forward(&lat_in, caption, &ts, Some(mask))?.to_dtype(DType::F32)?;
            let v_uncond = v.narrow(0, 0, 1)?;
            let v_text = v.narrow(0, 1, 1)?;
            let v = (&v_uncond + ((v_text - &v_uncond)? * guidance)?)?; // CFG
            let dt = sigmas[i + 1] - sigmas[i];
            latent = (latent + (v * dt)?)?; // Euler flow-match step
            pb.set_position((i + 1) as u64);
        }
        pb.finish_and_clear();
        Ok(latent)
    }

    /// Free the DiT (~3.3 GB) — call after all denoise loops, before the memory-heavy F32 decode.
    fn free_dit(&mut self) {
        self.dit = None;
    }

    /// DC-AE decode a raw latent → packed RGB `u8` `(H·W·3)`. Only needs the VAE (DiT freed).
    fn decode(&self, latent: &Tensor, w: u32, h: u32) -> Result<Vec<u8>> {
        let z = (latent / DCAE_SCALE)?;
        let img = {
            use super::dc_ae::ImageVae;
            self.vae.decode(&z)?
        };
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?.permute((1, 2, 0))?.flatten_all()?; // (H,W,3)
        let _ = (w, h);
        Ok(img.to_vec1::<u8>()?)
    }
}

use candle_core::IndexOp;

/// Run a Sana text-to-image generation.
pub async fn run(req: RunRequest) -> Result<()> {
    if !req.loras.is_empty() {
        bail!("Sana LoRA is not wired yet (base t2i first — ROADMAP_4.5.0). Drop --loras.");
    }
    if req.width % 32 != 0 || req.height % 32 != 0 {
        bail!("Sana output must be a multiple of 32 (DC-AE is 32× compression); got {}x{}.", req.width, req.height);
    }
    let steps = if req.steps == 0 { 20 } else { req.steps };
    let guidance = if req.guidance <= 0.0 { 4.5 } else { req.guidance };

    let mut pipeline = Pipeline::load(&req.model, req.device.clone()).await?;
    // Any OOM in encode / denoise / decode is caught and decorated with Sana-specific mitigations
    // (smaller --size, --device cpu) rather than crashing with a raw candle Metal error.
    generate_all(&mut pipeline, &req, steps, guidance)
        .map_err(|e| crate::error_hints::decorate_oom(e, crate::error_hints::OomContext::Sana))
}

/// Encode → denoise (all seeds) → free the DiT → decode + save. Separated from `run` so its errors
/// (notably Metal buffer OOM) can be decorated with mitigations before surfacing.
fn generate_all(pipeline: &mut Pipeline, req: &RunRequest, steps: usize, guidance: f64) -> Result<()> {
    // Encode the prompt once (fixed across the count loop), then free the ~5 GB Gemma encoder.
    let (caption, mask) = pipeline.encode(&req.prompt, &req.negative)?;
    let count = req.count.max(1);
    let (w, h) = (req.width, req.height);

    // Denoise every seed first (DiT resident), collecting the small latents.
    let mut latents: Vec<(u64, Tensor)> = Vec::with_capacity(count as usize);
    for idx in 0..count {
        let seed = req.seed.unwrap_or(42).wrapping_add(idx as u64);
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &req.device);
        let _ = req.device.set_seed(prepared);
        crate::ui::progress::println(&format!("  sana {} of {} (seed={seed})", idx + 1, count));
        latents.push((seed, pipeline.denoise(&caption, &mask, w, h, steps, guidance)?));
    }
    // Free the DiT (~3.3 GB) before the memory-heavy F32 DC-AE decode (avoids Metal buffer OOM).
    pipeline.free_dit();

    for (seed, latent) in &latents {
        let buf = pipeline.decode(latent, w, h)?;
        let mut m = crate::imaging::metadata::GenerationMetadata::new(
            &req.prompt, &req.model, *seed, steps, guidance, "flow-euler", w, h,
        );
        m.negative = req.negative.clone();
        let out_path = req.out_dir.join(format!("plakat-sana-{seed}.png"));
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, w, h, &out_path, &m)?;
        crate::ui::progress::println(&format!("  → {}", out_path.display()));
    }
    Ok(())
}
