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
    /// img2img: optional init image + denoise strength (0 = keep init, 1 = full txt2img).
    pub init_image: Option<std::path::PathBuf>,
    pub strength: Option<f32>,
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

/// DPM++ flow sigma schedule (Sana's *default* `DPMSolverMultistepScheduler` with
/// `use_flow_sigmas`): `σ = 1 − α`, `α = linspace(1, 1/1000, N+1)`, shifted, flipped, drop-last,
/// terminal 0. Returns N+1 sigmas. Differs from FlowMatchEuler's schedule.
fn dpm_flow_sigmas(steps: usize, shift: f64) -> Vec<f64> {
    let n = steps;
    let mut sig: Vec<f64> = (0..=n)
        .map(|i| {
            let alpha = 1.0 - (i as f64) * (1.0 - 1.0 / 1000.0) / (n as f64);
            let s = 1.0 - alpha; // ascending 0..0.999
            shift_t(s, shift)
        })
        .rev() // descending, n+1 values
        .collect();
    sig.pop(); // drop the last (smallest) → n values
    sig.push(0.0); // terminal sigma (final_sigmas_type = "zero")
    sig
}

/// The Sana sampler. **DPM++ 2M multistep (flow)** is the default — Sana's shipped scheduler,
/// higher quality than the 4.5 FlowMatchEuler, which stays available via `--scheduler euler`.
enum SanaSched {
    /// FlowMatchEuler (4.5): `latent += (σ_next − σ)·v`.
    Euler { sigmas: Vec<f64> },
    /// DPM++ 2M multistep with flow sigmas + `flow_prediction` (x0 = sample − σ·v). `prev_x0`
    /// carries the previous step's converted output for the 2nd-order update.
    Dpm { sigmas: Vec<f64>, prev_x0: Option<Tensor> },
}

impl SanaSched {
    fn new(scheduler: SchedulerKind, steps: usize) -> Self {
        // Only an explicit euler request opts out of the DPM++ default.
        let want_euler = matches!(scheduler, SchedulerKind::Euler | SchedulerKind::EulerA);
        if want_euler {
            SanaSched::Euler { sigmas: flow_sigmas(steps, FLOW_SHIFT) }
        } else {
            SanaSched::Dpm { sigmas: dpm_flow_sigmas(steps, FLOW_SHIFT), prev_x0: None }
        }
    }
    /// The sigma at step index `i`.
    fn sigma(&self, i: usize) -> f64 {
        match self {
            SanaSched::Euler { sigmas } | SanaSched::Dpm { sigmas, .. } => sigmas[i],
        }
    }
    /// The DiT timestep for step `i` (`floor(σ_i · 1000)`, matching diffusers).
    fn timestep(&self, i: usize) -> f64 {
        let s = match self {
            SanaSched::Euler { sigmas } | SanaSched::Dpm { sigmas, .. } => sigmas[i],
        };
        (s * 1000.0).floor()
    }
    /// One step: advance `sample` given the DiT velocity `v` at step `i` → the next latent.
    fn step(&mut self, v: &Tensor, i: usize, sample: &Tensor) -> Result<Tensor> {
        match self {
            SanaSched::Euler { sigmas } => {
                let dt = sigmas[i + 1] - sigmas[i];
                Ok((sample + (v * dt)?)?)
            }
            SanaSched::Dpm { sigmas, prev_x0 } => {
                let n = sigmas.len() - 1;
                let sigma_s0 = sigmas[i]; // current
                let sigma_t = sigmas[i + 1]; // next
                // flow_prediction: x0 = sample − σ·v.
                let x0 = (sample - (v * sigma_s0)?)?;
                // flow: alpha = 1 − σ, lambda = ln(alpha) − ln(σ).
                let lam = |s: f64| (1.0 - s).ln() - s.ln();
                let (alpha_t, lam_t, lam_s0) = (1.0 - sigma_t, lam(sigma_t), lam(sigma_s0));
                let h = lam_t - lam_s0;
                let coef = alpha_t * ((-h).exp() - 1.0); // at σ_t=0: alpha_t=1, exp(-∞)=0 → −1
                let ratio = sigma_t / sigma_s0;
                // First-order (DDIM-like) on step 0 and the last step (lower_order_final);
                // second-order midpoint otherwise.
                let out = if i == 0 || i == n - 1 || prev_x0.is_none() {
                    ((sample * ratio)? - (&x0 * coef)?)?
                } else {
                    let m1 = prev_x0.as_ref().unwrap();
                    let sigma_s1 = sigmas[i - 1];
                    let h0 = lam_s0 - lam(sigma_s1);
                    let r0 = h0 / h;
                    let d1 = ((&x0 - m1)? * (1.0 / r0))?; // (1/r0)(m0 − m1)
                    (((sample * ratio)? - (&x0 * coef)?)? - (d1 * (0.5 * coef))?)?
                };
                *prev_x0 = Some(x0);
                Ok(out)
            }
        }
    }
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

    // Reference from diffusers DPMSolverMultistepScheduler(use_flow_sigmas, flow_shift=3.0,
    // final_sigmas_type="zero").set_timesteps(20) — Sana's default scheduler.
    #[test]
    fn dpm_flow_sigmas_match_diffusers() {
        let want = [
            0.999666, 0.982419, 0.963941, 0.944094, 0.922722, 0.89964, 0.874635, 0.847457, 0.81781,
            0.78534, 0.749625, 0.710152, 0.666296, 0.617284, 0.562148, 0.499667, 0.428265, 0.345888,
            0.249792, 0.13624, 0.0,
        ];
        let got = dpm_flow_sigmas(20, 3.0);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "dpm sigma[{i}]: got {g}, want {w}");
        }
    }

    /// Verify the DPM++ 2M flow **step** (x0-conversion + 1st/2nd-order update) against a diffusers
    /// trajectory (`tools/reference/sana_dpm_dump.py`) — fixed velocities, no model. Skips if the
    /// goldens aren't present (they're gitignored; regenerate with the dumper).
    #[test]
    fn dpm_step_matches_diffusers() {
        use candle_core::Device;
        let path = "tools/reference/out/sana-dpm/goldens.safetensors";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let dev = Device::Cpu;
        let g = candle_core::safetensors::load(path, &dev).unwrap();
        let mut sched = SanaSched::Dpm { sigmas: dpm_flow_sigmas(20, FLOW_SHIFT), prev_x0: None };
        let mut latent = g["init"].clone();
        for i in 0..20 {
            let v = &g[&format!("v{i}")];
            latent = sched.step(v, i, &latent).unwrap();
            let want = &g[&format!("out{i}")];
            let max_abs = (&latent - want)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_vec0::<f32>()
                .unwrap();
            assert!(max_abs < 1e-3, "dpm step {i}: max_abs {max_abs}");
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

    /// DC-AE-encode an init image → a **model-space** latent `z0·scaling_factor` for img2img.
    fn encode_init(&self, path: &std::path::Path, w: u32, h: u32) -> Result<Tensor> {
        use super::dc_ae::ImageVae;
        let px = crate::imaging::preprocess::sd_image_tensor(path, w, h, &self.device, DType::F32)?;
        let px = if px.dims().len() == 3 { px.unsqueeze(0)? } else { px };
        let z0 = self.vae.encode(&px)?; // raw latent (deterministic)
        Ok((z0 * DCAE_SCALE)?) // model space (the denoise operates here)
    }

    /// Run the denoise loop → a raw latent `(1,32,h/32,w/32)`. Uses the DiT (must still be loaded).
    /// `scheduler` picks DPM++ 2M flow (default) vs FlowMatchEuler. With `init = Some((z0, strength))`
    /// this is **img2img**: start from a partially-noised init over a strength-trimmed schedule.
    #[allow(clippy::too_many_arguments)]
    fn denoise(&self, caption: &Tensor, mask: &Tensor, w: u32, h: u32, steps: usize, guidance: f64, scheduler: SchedulerKind, init: Option<(&Tensor, f32)>) -> Result<Tensor> {
        let (lw, lh) = (w as usize / 32, h as usize / 32);
        let dit = self.dit.as_ref().context("Sana DiT already freed")?;
        let sched = SanaSched::new(scheduler, steps);
        let noise = Tensor::randn(0f32, 1f32, (1, LATENT_CH, lh, lw), &self.device)?.to_dtype(DType::F32)?;
        // txt2img: start at step 0 from pure noise. img2img: start at the strength point from a
        // flow-noised init `x_σ = (1-σ)·z0 + σ·noise`.
        let (start, mut latent) = match init {
            None => (0usize, noise),
            Some((z0, strength)) => {
                let s = (strength.clamp(0.0, 1.0) as f64).max(0.0);
                let init_steps = (s * steps as f64).round() as usize;
                let start = steps.saturating_sub(init_steps).min(steps.saturating_sub(1));
                let sigma = sched.sigma(start);
                let x = ((z0 * (1.0 - sigma))? + (&noise * sigma)?)?;
                (start, x)
            }
        };
        let mut sched = sched;

        let pb = indicatif::ProgressBar::new((steps - start) as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template("  {spinner:.cyan} denoise [{bar:30.cyan/blue}] {pos}/{len}  {elapsed}")
                .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        for i in start..steps {
            let t = sched.timestep(i);
            let lat_in = Tensor::cat(&[&latent, &latent], 0)?.to_dtype(self.dtype)?; // (2,32,lh,lw)
            let ts = Tensor::from_vec(vec![t as f32; 2], 2, &self.device)?;
            let v = dit.forward(&lat_in, caption, &ts, Some(mask))?.to_dtype(DType::F32)?;
            let v_uncond = v.narrow(0, 0, 1)?;
            let v_text = v.narrow(0, 1, 1)?;
            let v = (&v_uncond + ((v_text - &v_uncond)? * guidance)?)?; // CFG
            latent = sched.step(&v, i, &latent)?;
            pb.set_position((i - start + 1) as u64);
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

    // img2img: DC-AE-encode the init once (fixed across the count loop).
    let init_z = match &req.init_image {
        Some(p) => Some(pipeline.encode_init(p, w, h)?),
        None => None,
    };
    let strength = req.strength.unwrap_or(0.6);

    // Denoise every seed first (DiT resident), collecting the small latents.
    let mut latents: Vec<(u64, Tensor)> = Vec::with_capacity(count as usize);
    for idx in 0..count {
        let seed = req.seed.unwrap_or(42).wrapping_add(idx as u64);
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &req.device);
        let _ = req.device.set_seed(prepared);
        crate::ui::progress::println(&format!("  sana {} of {} (seed={seed})", idx + 1, count));
        let init = init_z.as_ref().map(|z| (z, strength));
        latents.push((seed, pipeline.denoise(&caption, &mask, w, h, steps, guidance, req.scheduler, init)?));
    }
    // Free the DiT (~3.3 GB) before the memory-heavy F32 DC-AE decode (avoids Metal buffer OOM).
    pipeline.free_dit();

    let sched_label = match req.scheduler {
        SchedulerKind::Euler | SchedulerKind::EulerA => "flow-euler",
        _ => "dpm++2m-flow",
    };
    for (seed, latent) in &latents {
        let buf = pipeline.decode(latent, w, h)?;
        let mut m = crate::imaging::metadata::GenerationMetadata::new(
            &req.prompt, &req.model, *seed, steps, guidance, sched_label, w, h,
        );
        m.negative = req.negative.clone();
        let out_path = req.out_dir.join(format!("plakat-sana-{seed}.png"));
        crate::imaging::io::save_rgb_u8_with_metadata(&buf, w, h, &out_path, &m)?;
        crate::ui::progress::println(&format!("  → {}", out_path.display()));
    }
    Ok(())
}
