//! Kandinsky 5.0 T2I Lite — plakat's eighth model family (RFC KANDINSKY-1).
//!
//! **Through P3: it generates.** Phase 0 registered the surface (alias, variant and dispatch, capability
//! row, the native resolution buckets, the family-scoped flags); P1 added the conditioning and the VAE
//! (`kandinsky_text`); P2 the 6B flow-matching DiT (`kandinsky_dit`). This module is the run itself:
//!
//! * **Staged residency** (RFC §10.1). The three heavy pieces are never resident together: the text
//!   encoders (≈14 GB) are loaded, every prompt is encoded, and they are dropped; then the DiT (≈12 GB)
//!   denoises every image and is dropped; then the VAE decodes. `--keep-encoders` holds the encoders
//!   to the end instead.
//! * **Batching** (RFC §7.4). [`run_jobs`] takes any number of jobs and encodes each distinct
//!   prompt / negative pair once, up front, so a batch pays for the encoder swap once.
//! * **Sampling** (RFC §8). Flow-matching Euler on [`sigmas`]; the DiT's timestep is `sigma · 1000`; CFG
//!   is two forwards (the branches differ in length) and is skipped at `guidance ≤ 1`.
//!
//! txt2img only: img2img and inpaint are P5, the quantized tier P4.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::kandinsky_dit::Dit;
use super::kandinsky_text::{Embeds, TextEncoders, Vae};
use super::step_hook::{self, StepControl, StepHook};

/// The seven native resolution buckets (W×H) the model was trained at (RFC §4.3).
pub const BUCKETS: [(u32, u32); 7] = [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)];

/// Pipeline defaults (RFC §4.3).
pub const DEFAULT_STEPS: usize = 50;
pub const DEFAULT_GUIDANCE: f64 = 3.5;
/// Prompt tokens kept after the template (RFC §7.1); the reference requires `< 1024`.
pub const DEFAULT_MAX_SEQ: usize = 512;
pub const MAX_SEQ_CAP: usize = 1023;
/// Peak resident GB at BF16 with staged residency — encode, release the encoders, denoise (RFC §10.1).
pub const STAGED_PEAK_GB: f64 = 15.0;

/// Whether the family's pipeline exists yet (it does, since P3): `doctor --capability` gives a real
/// verdict instead of `pending`.
pub const PIPELINE_READY: bool = true;

/// Whether a `--model` alias or repo id names this family. The substring is unambiguous across
/// plakat's alias and repo surface; a short alias that does not carry it (`k5`) is resolved to its
/// repo first.
pub fn is_kandinsky(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("kandinsky") || crate::hf::resolve_alias(&m).to_lowercase().contains("kandinsky")
}

/// Snap a requested size to the nearest native bucket: by aspect ratio first (compared in log space,
/// so 2:1 and 1:2 are equally far from square), then by area.
pub fn snap_bucket(w: u32, h: u32) -> (u32, u32) {
    let want = (w.max(1) as f64 / h.max(1) as f64).ln();
    let area = w as f64 * h as f64;
    let mut best = BUCKETS[0];
    let mut best_key = (f64::MAX, f64::MAX);
    for &(bw, bh) in &BUCKETS {
        let d_ratio = ((bw as f64 / bh as f64).ln() - want).abs();
        let d_area = ((bw as f64 * bh as f64) - area).abs();
        if d_ratio < best_key.0 - 1e-9 || ((d_ratio - best_key.0).abs() <= 1e-9 && d_area < best_key.1) {
            best = (bw, bh);
            best_key = (d_ratio, d_area);
        }
    }
    best
}

/// `--size-exact`: an off-bucket size must still divide by 16 (the 8× VAE and the 2×2 patch).
pub fn check_exact(w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 || w % 16 != 0 || h % 16 != 0 {
        anyhow::bail!("--size-exact needs both dimensions divisible by 16 (the 8× VAE and the DiT's 2×2 patch); got {w}x{h}");
    }
    Ok(())
}

/// What a path the family does not serve yet says: txt2img is in; img2img and inpaint are P5.
pub fn txt2img_only(what: &str) -> anyhow::Error {
    anyhow::anyhow!("Kandinsky 5 does text-to-image only so far (RFC KANDINSKY-1): {what} is not wired for it yet. Use `plakat generate --model kandinsky5`.")
}

/// The sigma schedule the family samples on: diffusers' `FlowMatchEulerDiscreteScheduler` at
/// `shift = 5.0` (the repo's `scheduler_config.json`), `steps + 1` values ending at 0. The DiT's
/// timestep is `sigma · 1000`.
pub const SCHEDULER_SHIFT: f64 = 5.0;
pub fn sigmas(steps: usize) -> Vec<f64> {
    super::sana::flow_sigmas(steps, SCHEDULER_SHIFT)
}

/// One image to make.
#[derive(Debug, Clone)]
pub struct Job {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    /// Where the PNG goes.
    pub out_path: PathBuf,
}

/// What a batch shares.
#[derive(Debug, Clone)]
pub struct Settings {
    pub model: String,
    pub device: Device,
    /// Prompt tokens kept after the template (`--max-seq`).
    pub max_seq: usize,
    /// Hold the text encoders through the denoise instead of dropping them (`--keep-encoders`).
    pub keep_encoders: bool,
    /// The Qwen tower from a Q4_K_M GGUF (`--quantize-qwen`).
    pub quantize_qwen: bool,
    /// The DiT's block weights in NF4 (`--dit-nf4`).
    pub dit_nf4: bool,
}

/// `plakat generate`'s request: `count` images of one prompt, seeds counting up from `seed`.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub model: String,
    pub device: Device,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    /// 0 = the family default.
    pub steps: usize,
    /// ≤ 0 = the family default; ≤ 1 runs without CFG.
    pub guidance: f64,
    pub seed: Option<u64>,
    pub out_dir: PathBuf,
    pub count: u32,
    pub max_seq: usize,
    pub keep_encoders: bool,
    /// The Qwen tower from a Q4_K_M GGUF (`--quantize-qwen`).
    pub quantize_qwen: bool,
    /// The DiT's block weights in NF4 (`--dit-nf4`).
    pub dit_nf4: bool,
}

/// Flow-matching Euler from `noise` (NCHW `(1, 16, H/8, W/8)`, pure noise at sigma 1) to the clean
/// latent. `neg` is the unconditional branch: `None` runs without CFG. The text stream is
/// time-modulated, so nothing is cached across steps (trap T8).
pub fn denoise(dit: &Dit, pos: &Embeds, neg: Option<&Embeds>, noise: &Tensor, steps: usize, guidance: f64, hook: &mut Option<&mut dyn StepHook>, label: &str) -> Result<Tensor> {
    denoise_observed(dit, pos, neg, noise, steps, guidance, hook, label, &mut |_, _| {})
}

/// [`denoise`], showing `observe` the latent going into each step (a parity run compares them).
#[allow(clippy::too_many_arguments)]
fn denoise_observed(dit: &Dit, pos: &Embeds, neg: Option<&Embeds>, noise: &Tensor, steps: usize, guidance: f64, hook: &mut Option<&mut dyn StepHook>, label: &str, observe: &mut dyn FnMut(usize, &Tensor)) -> Result<Tensor> {
    if steps < 2 {
        anyhow::bail!("Kandinsky 5 needs at least 2 steps (got {steps}); the default is {DEFAULT_STEPS}");
    }
    let sig = sigmas(steps);
    let mut x = noise.to_dtype(DType::F32)?;
    let bar = crate::ui::progress::step_bar(steps as u64, label);
    for i in 0..steps {
        if step_hook::step(hook, i, steps) == StepControl::Cancel || step_hook::is_cancelled(hook) {
            bar.abandon();
            anyhow::bail!("cancelled at step {i} of {steps}");
        }
        observe(i, &x);
        let t = sig[i] * 1000.0;
        let mut v = dit.forward(&x, &pos.qwen, &pos.pooled, t)?;
        if let Some(neg) = neg {
            let v_uncond = dit.forward(&x, &neg.qwen, &neg.pooled, t)?;
            v = (&v_uncond + ((&v - &v_uncond)? * guidance)?)?;
        }
        x = (x + (v * (sig[i + 1] - sig[i]))?)?;
        bar.inc(1);
    }
    bar.finish_and_clear();
    Ok(x)
}

/// A decoded image `(1, 3, H, W)` in `[-1, 1]` as packed RGB8.
fn to_rgb8(image: &Tensor) -> Result<Vec<u8>> {
    let (_b, _c, h, w) = image.dims4()?;
    let px: Vec<f32> = image.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let plane = h * w;
    Ok((0..plane).flat_map(|i| [0, 1, 2].map(|c| ((px[c * plane + i] + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)).collect())
}

/// Latent tiles of the fallback decode: 64 latent px (512 image px) with a quarter overlap.
const TILE_LATENT: usize = 64;
const TILE_STRIDE: usize = 48;

/// Decode a latent in blended tiles.
pub fn decode_tiled(vae: &Vae, latent: &Tensor) -> Result<Tensor> {
    crate::pipelines::tiled::tile_decode_2d(latent, TILE_LATENT, TILE_STRIDE, 8, |tile| vae.decode(tile))
}

/// Whether to trade time and decode fidelity for memory: a host with under 24 GB of RAM, or
/// `PLAKAT_K5_LOW_MEMORY=1` (`=0` to refuse). It shrinks the DiT's attention chunks and decodes in tiles;
/// with the quantized tier that is what holds a 1024² run under 7 GB (RFC §10.2).
pub fn low_memory() -> bool {
    static LOW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LOW.get_or_init(|| match std::env::var("PLAKAT_K5_LOW_MEMORY").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => crate::hw::total_ram_gb() < 24.0,
    })
}

/// Where the VAE decodes. On Metal that is the CPU: candle's whole-image decode takes 38 GB there at
/// 1024² against 10 GB on the CPU (30 s, and within 1e-5 of the reference's image), and its tiles 11 GB
/// against 3 GB.
pub fn decode_device(base: &Device) -> Result<Device> {
    Ok(if base.is_metal() { Device::Cpu } else { stage_device(base)? })
}

/// Decode a latent on the VAE's device. Whole, with blended tiles as the out-of-memory fallback — or
/// in tiles at once on a [`low_memory`] host, where the whole decode's 10 GB is not there to take: the
/// tiles need 3 GB, take twice as long and land 36 dB from the whole decode.
/// `PLAKAT_K5_VAE_TILED=1` forces the tiles and `=0` the whole decode.
pub fn decode(vae: &Vae, latent: &Tensor) -> Result<Tensor> {
    let tiles = match std::env::var("PLAKAT_K5_VAE_TILED").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => low_memory(),
    };
    if tiles {
        return decode_tiled(vae, latent);
    }
    match vae.decode(latent) {
        Ok(image) => Ok(image),
        Err(e) if crate::error_hints::looks_like_oom(&format!("{e:#}")) => {
            crate::ui::progress::println("kandinsky5: the whole-image VAE decode did not fit; decoding in tiles");
            decode_tiled(vae, latent)
        }
        Err(e) => Err(e),
    }
}

/// A device for one stage of the run. On Metal it is a fresh handle with its own buffer pools, so that
/// dropping the stage's tensors returns its memory at once. On the shared handle it does not: candle
/// sweeps the pool the weights sit in only when a command buffer fills (so the encoders outlived the
/// DiT's load, and the DiT the decode), and never sweeps the pool its intermediates sit in.
/// Nothing on a stage's device may outlive the stage — what crosses over goes through the CPU.
pub fn stage_device(base: &Device) -> Result<Device> {
    Ok(if base.is_metal() { Device::new_metal(0)? } else { base.clone() })
}

/// Run a batch with staged residency: encode every distinct prompt, release the encoders, denoise every
/// job, release the DiT, decode and save. Returns the files written, in job order.
pub async fn run_jobs(settings: &Settings, jobs: &[Job]) -> Result<Vec<PathBuf>> {
    run_jobs_inner(settings, jobs).await.map_err(|e| crate::error_hints::decorate_oom(e, crate::error_hints::OomContext::Kandinsky5))
}

async fn run_jobs_inner(settings: &Settings, jobs: &[Job]) -> Result<Vec<PathBuf>> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    for j in jobs {
        check_exact(j.width, j.height)?;
    }
    let repo = crate::hf::resolve_alias(&settings.model).to_string();
    let println = |m: String| crate::ui::progress::println(&m);

    // Stage 1 — text. Each distinct prompt and negative is encoded once.
    let t0 = std::time::Instant::now();
    let spin = crate::ui::progress::spinner("Loading the Kandinsky 5 text encoders (Qwen2.5-VL text tower + CLIP-L)");
    // The quantized tower runs on the CPU on Metal: 5.7 GB there against 7.9 GB on the GPU, the same
    // embeddings (per-token cosine 0.994 against 0.995), and a prompt takes two seconds either way.
    let text_device = if settings.quantize_qwen && settings.device.is_metal() { Device::Cpu } else { stage_device(&settings.device)? };
    let mut encoders = TextEncoders::load_with(&repo, &text_device, settings.quantize_qwen).await.context("loading the Kandinsky 5 text encoders")?;
    spin.finish_with_message(format!("✓ text encoders loaded in {:.1}s", t0.elapsed().as_secs_f64()));
    let t1 = std::time::Instant::now();
    let mut embeds: std::collections::HashMap<String, Embeds> = std::collections::HashMap::new();
    for j in jobs {
        let wanted = std::iter::once(&j.prompt).chain((j.guidance > 1.0).then_some(&j.negative));
        for text in wanted {
            if !embeds.contains_key(text) {
                let mut e = encoders.encode(text, settings.max_seq)?;
                (e.qwen, e.pooled) = (e.qwen.to_device(&Device::Cpu)?, e.pooled.to_device(&Device::Cpu)?);
                if let Some(d) = &e.dropped {
                    println(format!("kandinsky5: the prompt ran past --max-seq {}; dropped: “{}”", settings.max_seq, d.trim()));
                }
                embeds.insert(text.clone(), e);
            }
        }
    }
    println(format!("kandinsky5: encoded {} text(s) in {:.1}s", embeds.len(), t1.elapsed().as_secs_f64()));
    // Staged residency: the encoders are gone before the DiT is loaded, unless asked to stay.
    let kept = if settings.keep_encoders { Some((encoders, text_device)) } else { drop((encoders, text_device)); None };

    // Stage 2 — denoise every job with the DiT resident.
    let t2 = std::time::Instant::now();
    let spin = crate::ui::progress::spinner("Loading the Kandinsky 5 DiT");
    let dit_device = stage_device(&settings.device)?;
    let device = &dit_device;
    let dit = Dit::load_with(&repo, device, settings.dit_nf4).await.context("loading the Kandinsky 5 DiT")?;
    spin.finish_with_message(format!("✓ DiT loaded in {:.1}s", t2.elapsed().as_secs_f64()));
    let mut latents = Vec::with_capacity(jobs.len());
    for (n, j) in jobs.iter().enumerate() {
        let _ = device.set_seed(crate::pipelines::seeds::prepare_seed(j.seed, device));
        let noise = Tensor::randn(0f32, 1f32, (1, dit.cfg.in_visual_dim, (j.height / 8) as usize, (j.width / 8) as usize), device)?;
        let neg = (j.guidance > 1.0).then(|| &embeds[&j.negative]);
        println(format!("  kandinsky5 {} of {} (seed={}, {}x{}, {} steps, guidance {})", n + 1, jobs.len(), j.seed, j.width, j.height, j.steps, j.guidance));
        let t = std::time::Instant::now();
        let mut nohook: Option<&mut dyn StepHook> = None;
        let latent = denoise(&dit, &embeds[&j.prompt], neg, &noise, j.steps, j.guidance, &mut nohook, "denoise")?;
        let secs = t.elapsed().as_secs_f64();
        println(format!("  denoised in {secs:.1}s ({:.2}s a step)", secs / j.steps as f64));
        latents.push(latent.to_device(&Device::Cpu)?);
    }
    // The DiT is released before the F32 decode, which is the other memory peak.
    drop((dit, embeds, dit_device));

    // Stage 3 — decode and save.
    let device = &decode_device(&settings.device)?;
    let vae = Vae::load(&repo, device).await?;
    let mut written = Vec::with_capacity(jobs.len());
    for (j, latent) in jobs.iter().zip(&latents) {
        let image = decode(&vae, &latent.to_device(device)?)?;
        let mut m = crate::imaging::metadata::GenerationMetadata::new(&j.prompt, &settings.model, j.seed, j.steps, j.guidance, "flow-euler", j.width, j.height);
        m.negative = j.negative.clone();
        let bucket = if BUCKETS.contains(&(j.width, j.height)) { format!("{}x{}", j.width, j.height) } else { "exact".to_string() };
        m.extras.extend([("family".to_string(), "kandinsky5".to_string()), ("max_seq".to_string(), settings.max_seq.to_string()), ("bucket".to_string(), bucket)]);
        if let Some(dir) = j.out_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        crate::imaging::io::save_rgb_u8_with_metadata(&to_rgb8(&image)?, j.width, j.height, &j.out_path, &m)?;
        println(format!("  → {}", j.out_path.display()));
        written.push(j.out_path.clone());
    }
    drop(kept);
    Ok(written)
}

/// `plakat generate --model kandinsky5`.
pub async fn run(req: RunRequest) -> Result<()> {
    let steps = if req.steps == 0 { DEFAULT_STEPS } else { req.steps };
    let guidance = if req.guidance <= 0.0 { DEFAULT_GUIDANCE } else { req.guidance };
    let first = req.seed.unwrap_or(42);
    let jobs: Vec<Job> = (0..req.count.max(1) as u64)
        .map(|i| {
            let seed = first.wrapping_add(i);
            Job { prompt: req.prompt.clone(), negative: req.negative.clone(), width: req.width, height: req.height, steps, guidance, seed, out_path: req.out_dir.join(format!("plakat-kandinsky5-{seed}.png")) }
        })
        .collect();
    let settings = Settings { model: req.model.clone(), device: req.device.clone(), max_seq: if req.max_seq == 0 { DEFAULT_MAX_SEQ } else { req.max_seq }, keep_encoders: req.keep_encoders, quantize_qwen: req.quantize_qwen, dit_nf4: req.dit_nf4 };
    run_jobs(&settings, &jobs).await.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::t2i::Variant;

    #[test]
    fn the_aliases_resolve_to_the_family() {
        for a in ["kandinsky5", "kandinsky", "k5", "kandinsky5-lite", "kandinsky5-pretrain"] {
            let repo = crate::hf::resolve_alias(a);
            assert!(repo.starts_with("kandinskylab/Kandinsky-5.0-T2I-Lite-"), "{a} → {repo}");
            assert_eq!(Variant::detect(repo), Variant::Kandinsky5, "{a}");
            // The dispatch detects from what the user typed, not the resolved repo — `k5` included.
            assert_eq!(Variant::detect(a), Variant::Kandinsky5, "{a}");
            assert!(is_kandinsky(a) && is_kandinsky(&a.to_uppercase()), "{a}");
        }
        assert_eq!(Variant::detect("kandinsky5"), Variant::Kandinsky5);
        // No other family is caught by the substring.
        for other in ["sd15", "sdxl", "flux-dev", "sd35-medium", "pixart", "stable-cascade", "sana"] {
            assert!(!Variant::detect(other).is_kandinsky(), "{other}");
            assert!(!is_kandinsky(crate::hf::resolve_alias(other)), "{other}");
        }
    }

    #[test]
    fn a_size_snaps_to_its_nearest_bucket() {
        assert_eq!(snap_bucket(1200, 800), (1280, 768)); // the RFC's example
        assert_eq!(snap_bucket(512, 512), (1024, 1024));
        assert_eq!(snap_bucket(720, 1280), (768, 1280));
        assert_eq!(snap_bucket(1920, 800), (1408, 640));
        assert_eq!(snap_bucket(1000, 1250), (896, 1152));
        for &(w, h) in &BUCKETS {
            assert_eq!(snap_bucket(w, h), (w, h), "a bucket is its own nearest");
            assert!(check_exact(w, h).is_ok());
            // The largest bucket stays inside the 128-position RoPE table (RFC §6.2).
            assert!(w / 16 <= 128 && h / 16 <= 128);
        }
    }

    #[test]
    fn the_sigma_schedule_runs_from_one_to_zero() {
        let s = sigmas(DEFAULT_STEPS);
        assert_eq!(s.len(), DEFAULT_STEPS + 1);
        assert!((s[0] - 1.0).abs() < 1e-9, "starts at pure noise: {}", s[0]);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s.windows(2).all(|w| w[1] < w[0]), "strictly decreasing");
        // shift 5 keeps the schedule in the high-noise region longer than the unshifted line.
        assert!(s[DEFAULT_STEPS / 2] > 0.5);
    }

    #[test]
    fn the_sigmas_are_the_reference_schedulers() {
        // diffusers' FlowMatchEulerDiscreteScheduler(shift = 5.0).set_timesteps(50), as dumped by
        // tools/kandinsky_dump.py --stage p2 (RFC §8.1).
        let s = sigmas(50);
        for (i, want) in [(0, 1.0), (1, 0.9958716), (2, 0.9916046), (24, 0.840241), (25, 0.8290321), (26, 0.8171927), (47, 0.1928036), (48, 0.1148195), (49, 0.0244141), (50, 0.0)] {
            assert!((s[i] - want).abs() < 2e-6, "sigma[{i}]: got {}, want {want}", s[i]);
        }
    }

    #[test]
    fn a_decoded_image_packs_to_rgb8() {
        // (1, 3, 1, 2): pixel 0 is (−1, 0, 1) → (0, 128, 255); pixel 1 is out of range and clamps.
        let t = Tensor::new(&[-1f32, 2.0, 0.0, -3.0, 1.0, 0.5], &Device::Cpu).unwrap().reshape((1, 3, 1, 2)).unwrap();
        assert_eq!(to_rgb8(&t).unwrap(), [0, 128, 255, 255, 0, 191]);
    }

    #[test]
    fn too_few_steps_are_refused_before_any_work() {
        assert!(txt2img_only("img2img").to_string().contains("img2img"));
        assert!(PIPELINE_READY);
        assert!(DEFAULT_STEPS >= 2 && DEFAULT_GUIDANCE > 1.0);
    }

    /// P3's parity gate (RFC §12, stages 7–8): the whole loop from the reference's own noise and
    /// embeddings to the final latent, and its decode. Needs a `tools/kandinsky_dump.py --stage p2` dump
    /// (with its `p1.safetensors`) in `PLAKAT_PARITY_DIR`; 100 DiT forwards at the dump's size:
    /// `PLAKAT_PARITY_DIR=<dir> cargo test --release --features metal --lib kandinsky_parity_p3 -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn kandinsky_parity_p3() {
        let dir = std::path::PathBuf::from(std::env::var("PLAKAT_PARITY_DIR").expect("set PLAKAT_PARITY_DIR to a tools/kandinsky_dump.py output directory"));
        let meta: serde_json::Value = serde_json::from_reader(std::fs::File::open(dir.join("meta_p2.json")).unwrap()).unwrap();
        let device = crate::device::select(&std::env::var("PLAKAT_PARITY_DEVICE").unwrap_or_else(|_| "auto".into())).unwrap();
        let repo = meta["repo"].as_str().unwrap();
        let (steps, guidance) = (meta["steps"].as_u64().unwrap() as usize, meta["guidance"].as_f64().unwrap());
        let p1 = candle_core::safetensors::load(dir.join("p1.safetensors"), &device).unwrap();
        let p2 = candle_core::safetensors::load(dir.join("p2.safetensors"), &Device::Cpu).unwrap();
        let embeds = |tag: &str| Embeds { qwen: p1[&format!("qwen_hidden_{tag}")].clone(), pooled: p1[&format!("clip_pooled_{tag}")].clone(), ids: Vec::new(), dropped: None };
        let (pos, neg) = (embeds("pos"), embeds("neg"));
        let flat = |t: &Tensor| -> Vec<f32> { t.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap() };
        let cosine = |a: &Tensor, b: &Tensor| -> f64 {
            let (a, b) = (flat(a), flat(b));
            let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |x: &[f32]| x.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(&a) * n(&b))
        };
        // PSNR on a [0, 1] scale, both images clamped to the displayable range.
        let psnr = |a: &Tensor, b: &Tensor| -> f64 {
            let (a, b) = (flat(a), flat(b));
            let mse: f64 = a.iter().zip(&b).map(|(x, y)| ((x.clamp(-1.0, 1.0) - y.clamp(-1.0, 1.0)) as f64 / 2.0).powi(2)).sum::<f64>() / a.len() as f64;
            assert!(mse.is_finite());
            -10.0 * mse.max(1e-12).log10()
        };

        let dit = Dit::load(repo, &device).await.unwrap();
        let exact = dit.compute() == DType::F32;
        let t = std::time::Instant::now();
        let mut nohook: Option<&mut dyn StepHook> = None;
        // The trajectory: the latent going into each tapped step against the reference's.
        let mut observe = |i: usize, x: &Tensor| {
            if let Some(want) = p2.get(&format!("x_step{i}")) {
                println!("latent into step {i:>2}: cosine {:.6}", cosine(x, want));
            }
        };
        let latent = denoise_observed(&dit, &pos, Some(&neg), &p2["noise"].to_device(&device).unwrap(), steps, guidance, &mut nohook, "parity", &mut observe).unwrap();
        println!("{steps} steps with CFG in {:.0}s on {device:?}, compute {:?}", t.elapsed().as_secs_f64(), dit.compute());
        drop(dit);
        let vae_device = decode_device(&device).unwrap();
        let vae = Vae::load(repo, &vae_device).await.unwrap();
        // Stage 7: the final latent. Stage 8: the image — ours against the reference's, and (to tell
        // the VAE from the loop) our decode of the REFERENCE's latent against the reference's image.
        let c = cosine(&latent, &p2["final_latent"]);
        let image = decode(&vae, &latent.to_device(&vae_device).unwrap()).unwrap();
        let (db, db_vae) = (psnr(&image, &p2["decoded"]), psnr(&decode(&vae, &p2["final_latent"].to_device(&vae_device).unwrap()).unwrap(), &p2["decoded"]));
        println!("final latent cosine {c:.6} · image PSNR {db:.1} dB (the VAE alone, on the reference's latent: {db_vae:.1} dB)");
        if let Some(out) = std::env::var_os("PLAKAT_PARITY_OUT") {
            let (h, w) = (image.dim(2).unwrap() as u32, image.dim(3).unwrap() as u32);
            image::RgbImage::from_raw(w, h, to_rgb8(&image).unwrap()).unwrap().save(&out).unwrap();
        }
        let (min_cos, min_db) = if exact { (0.999, 40.0) } else { (0.99, 32.0) };
        assert!(c >= min_cos, "final latent cosine {c}");
        assert!(db >= min_db, "image PSNR {db} dB");
    }

    /// P4's first measurements (RFC §10.2), against a P2 reference dump: the quantized Qwen tower on the
    /// CPU and on the GPU, and one NF4 DiT forward. Prints; asserts nothing.
    /// `PLAKAT_PARITY_DIR=<dir> cargo test --release --features metal --lib kandinsky_p4_probe -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn kandinsky_p4_probe() {
        let dir = std::path::PathBuf::from(std::env::var("PLAKAT_PARITY_DIR").unwrap());
        let meta: serde_json::Value = serde_json::from_reader(std::fs::File::open(dir.join("meta_p2.json")).unwrap()).unwrap();
        let p1meta: serde_json::Value = serde_json::from_reader(std::fs::File::open(dir.join("meta.json")).unwrap()).unwrap();
        let p1 = candle_core::safetensors::load(dir.join("p1.safetensors"), &Device::Cpu).unwrap();
        let p2 = candle_core::safetensors::load(dir.join("p2.safetensors"), &Device::Cpu).unwrap();
        let repo = meta["repo"].as_str().unwrap();
        let base = crate::device::select("auto").unwrap();
        let gb = || crate::memwatch::footprint_gb().unwrap_or(0.0);
        let flat = |t: &Tensor| -> Vec<f32> { t.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap() };
        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let n = |x: &[f32]| x.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        // Mean and worst per-token cosine of `(1, L, H)` hidden states.
        let per_token = |got: &Tensor, want: &Tensor| -> (f64, f64) {
            let (g, w, h) = (flat(got), flat(want), want.dim(2).unwrap());
            let each: Vec<f64> = g.chunks(h).zip(w.chunks(h)).map(|(a, b)| cos(a, b)).collect();
            (each.iter().sum::<f64>() / each.len() as f64, each.iter().cloned().fold(1.0, f64::min))
        };
        if std::env::var("PLAKAT_P4_SKIP_TEXT").is_err() {
            for on_cpu in [true, false] {
                let device = if on_cpu { Device::Cpu } else { stage_device(&base).unwrap() };
                let (before, t) = (gb(), std::time::Instant::now());
                let mut enc = TextEncoders::load_with(repo, &device, true).await.unwrap();
                let (loaded, t1) = (t.elapsed().as_secs_f64(), std::time::Instant::now());
                for tag in ["pos", "neg"] {
                    let text = p1meta[if tag == "pos" { "prompt" } else { "negative" }].as_str().unwrap();
                    let e = enc.encode(text, DEFAULT_MAX_SEQ).unwrap();
                    let (mean, worst) = per_token(&e.qwen, &p1[&format!("qwen_hidden_{tag}")]);
                    println!("Q4 Qwen on {}: {tag} per-token cosine mean {mean:.4} worst {worst:.4}", if on_cpu { "CPU" } else { "GPU" });
                }
                println!("Q4 Qwen on {}: loaded in {loaded:.1}s, two texts in {:.1}s, +{:.2} GB", if on_cpu { "CPU" } else { "GPU" }, t1.elapsed().as_secs_f64(), gb() - before);
                drop((enc, device));
                std::thread::sleep(std::time::Duration::from_secs(3));
            }
        }
        let device = stage_device(&base).unwrap();
        let (before, t) = (gb(), std::time::Instant::now());
        let dit = Dit::load_with(repo, &device, true).await.unwrap();
        println!("NF4 DiT: loaded in {:.1}s, +{:.2} GB", t.elapsed().as_secs_f64(), gb() - before);
        for i in 0..2 {
            let t = std::time::Instant::now();
            let v = dit.forward(&p2["noise"], &p1["qwen_hidden_pos"], &p1["clip_pooled_pos"], 500.0).unwrap();
            let c = cos(&flat(&v), &flat(&p2["velocity_500"]));
            println!("NF4 DiT: forward {i} in {:.1}s, velocity cosine {c:.6}, footprint {:.2} GB", t.elapsed().as_secs_f64(), gb());
        }
    }

    /// P4's acceptance (RFC §10.2). The RFC names a "64-prompt verify corpus" that does not exist in
    /// the tree, so the prompts are the 8 × 8 grid below.
    /// 1. The Q4 tower against the BF16 one on all 64: mean per-token cosine ≥ 0.98.
    /// 2. CLIP adherence of the quantized tier (Q4 + NF4) against the full one, on the first
    ///    `PLAKAT_P4_IMAGES` prompts (default 4) at `PLAKAT_P4_STEPS` steps (default 20), same seeds:
    ///    the mean drops by ≤ 2 %. 64 images a tier at 50 steps is a day of this machine; set the two
    ///    variables for the whole thing. Images land in `PLAKAT_P4_OUT`.
    /// `PLAKAT_P4_OUT=<dir> cargo test --release --features metal --lib kandinsky_p4_acceptance -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn kandinsky_p4_acceptance() {
        let subjects = ["a red fox", "an old fisherman mending a net", "a glass teapot", "a lighthouse on a cliff", "two children flying a kite", "a steam locomotive", "a bowl of ripe figs", "a snowy owl in flight"];
        let settings = ["at dawn in soft golden light", "in heavy rain at night, neon reflections", "as a watercolor painting", "in a studio, black background, product photo", "in thick fog, muted colors", "as a detailed pencil sketch", "under a starry sky, long exposure", "in a sunlit meadow, shallow depth of field"];
        let prompts: Vec<String> = subjects.iter().flat_map(|a| settings.iter().map(move |b| format!("{a} {b}"))).collect();
        let base = crate::device::select("auto").unwrap();
        let repo = crate::hf::resolve_alias("kandinsky5").to_string();
        let flat = |t: &Tensor| -> Vec<f32> { t.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap() };
        let mut failed = Vec::new();

        if std::env::var("PLAKAT_P4_SKIP_TEXT").is_err() {
            let encode_all = |mut enc: TextEncoders| -> Vec<(Vec<f32>, usize)> { prompts.iter().map(|p| enc.encode(p, DEFAULT_MAX_SEQ).map(|e| (flat(&e.qwen), e.qwen.dim(2).unwrap())).unwrap()).collect() };
            let dense = {
                let device = stage_device(&base).unwrap();
                encode_all(TextEncoders::load(&repo, &device).await.unwrap())
            };
            let quant = encode_all(TextEncoders::load_with(&repo, &Device::Cpu, true).await.unwrap());
            let mut per_prompt = Vec::new();
            for ((d, h), (q, _)) in dense.iter().zip(&quant) {
                let each: Vec<f64> = d.chunks(*h).zip(q.chunks(*h)).map(|(a, b)| {
                    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
                    let n = |x: &[f32]| x.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                    dot / (n(a) * n(b))
                }).collect();
                per_prompt.push(each.iter().sum::<f64>() / each.len() as f64);
            }
            let mean = per_prompt.iter().sum::<f64>() / per_prompt.len() as f64;
            let worst = per_prompt.iter().cloned().fold(1.0, f64::min);
            println!("Q4 against BF16 tower, {} prompts: per-token cosine mean {mean:.4}, worst prompt {worst:.4}", prompts.len());
            if mean < 0.98 {
                failed.push(format!("encoder cosine {mean}"));
            }
        }

        let out = std::path::PathBuf::from(std::env::var("PLAKAT_P4_OUT").expect("set PLAKAT_P4_OUT to a directory for the images"));
        let n: usize = std::env::var("PLAKAT_P4_IMAGES").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let steps: usize = std::env::var("PLAKAT_P4_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
        // Spread over the grid rather than the first subject's row.
        let chosen: Vec<&String> = (0..n).map(|i| &prompts[(i * 9) % prompts.len()]).collect();
        let mut files = Vec::new();
        for (tier, quantized) in [("full", false), ("quant", true)] {
            let settings = Settings { model: "kandinsky5".into(), device: base.clone(), max_seq: DEFAULT_MAX_SEQ, keep_encoders: false, quantize_qwen: quantized, dit_nf4: quantized };
            let jobs: Vec<Job> = chosen.iter().enumerate().map(|(i, p)| Job { prompt: (*p).clone(), negative: String::new(), width: 1024, height: 1024, steps, guidance: DEFAULT_GUIDANCE, seed: 42 + i as u64, out_path: out.join(tier).join(format!("{i}.png")) }).collect();
            let t = std::time::Instant::now();
            files.push(run_jobs(&settings, &jobs).await.unwrap());
            println!("{tier}: {n} images at {steps} steps in {:.0}s", t.elapsed().as_secs_f64());
        }
        let aes = crate::pipelines::aesthetic::AestheticScorer::load(&base).await.unwrap();
        let clip = crate::pipelines::clip_adherence::ClipAdherence::load(&base).await.unwrap();
        let score = |files: &[PathBuf]| -> Vec<f32> { files.iter().zip(&chosen).map(|(f, p)| clip.adherence(&aes.image_embedding(f).unwrap(), p).unwrap()).collect() };
        let (full, quant) = (score(&files[0]), score(&files[1]));
        for (i, p) in chosen.iter().enumerate() {
            println!("adherence {:.4} full, {:.4} quantized — {p}", full[i], quant[i]);
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let drop = (mean(&full) - mean(&quant)) / mean(&full);
        println!("CLIP adherence: full {:.4}, quantized {:.4}, drop {:.1} %", mean(&full), mean(&quant), drop * 100.0);
        if drop > 0.02 {
            failed.push(format!("adherence drop {drop}"));
        }
        assert!(failed.is_empty(), "P4 acceptance failed: {failed:#?}");
    }

    /// What each way of decoding costs and how far it lands from the reference's image:
    /// `PLAKAT_PARITY_DIR=<dir> cargo test --release --features metal --lib kandinsky_decode_probe -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn kandinsky_decode_probe() {
        let dir = std::path::PathBuf::from(std::env::var("PLAKAT_PARITY_DIR").unwrap());
        let p2 = candle_core::safetensors::load(dir.join("p2.safetensors"), &Device::Cpu).unwrap();
        let repo = crate::hf::resolve_alias("kandinsky5").to_string();
        let base = crate::device::select("auto").unwrap();
        let gb = || crate::memwatch::footprint_gb().unwrap_or(0.0);
        let want = p2["decoded"].to_dtype(DType::F32).unwrap();
        let ways: [(&str, bool, Option<(usize, usize)>); 5] = [("CPU, whole", true, None), ("GPU, tiles 64/48", false, Some((64, 48))), ("GPU, tiles 64/32", false, Some((64, 32))), ("GPU, tiles 96/64", false, Some((96, 64))), ("CPU, tiles 64/48", true, Some((64, 48)))];
        for (name, on_cpu, tiles) in ways {
            let device = if on_cpu { Device::Cpu } else { stage_device(&base).unwrap() };
            let vae = Vae::load(&repo, &device).await.unwrap();
            let z = p2["final_latent"].to_device(&device).unwrap();
            let (before, t) = (gb(), std::time::Instant::now());
            let peak = std::sync::Arc::new(std::sync::Mutex::new(0f64));
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let watch = {
                let (peak, stop) = (peak.clone(), stop.clone());
                std::thread::spawn(move || {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let now = crate::memwatch::footprint_gb().unwrap_or(0.0);
                        let mut p = peak.lock().unwrap();
                        *p = p.max(now);
                        drop(p);
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                })
            };
            let image = match tiles {
                Some((tile, stride)) => crate::pipelines::tiled::tile_decode_2d(&z, tile, stride, 8, |x| vae.decode(x)).unwrap(),
                None => vae.decode(&z).unwrap(),
            }
            .to_device(&Device::Cpu)
            .unwrap();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            watch.join().unwrap();
            let mse = (&image - &want).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
            println!("{name:<18} {:>5.1}s  +{:>5.2} GB  PSNR {:.1} dB", t.elapsed().as_secs_f64(), *peak.lock().unwrap() - before, 10.0 * (4.0 / mse.max(1e-12)).log10());
            drop((vae, z, image, device));
            std::thread::sleep(std::time::Duration::from_secs(3));
        }
    }

    /// Where the memory goes, stage by stage (the process footprint, which counts GPU buffers):
    /// `cargo test --release --features metal --lib kandinsky_memory_probe -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn kandinsky_memory_probe() {
        let base = crate::device::select("auto").unwrap();
        let repo = crate::hf::resolve_alias("kandinsky5").to_string();
        let mem = |what: &str| println!("{:>6.2} GB  {what}", crate::memwatch::footprint_gb().unwrap_or(0.0));
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu).unwrap();
        // Metal returns memory a little after the last reference goes; watch it settle.
        let settle = |what: &str| {
            let t = std::time::Instant::now();
            let mut last = crate::memwatch::footprint_gb().unwrap_or(0.0);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let now = crate::memwatch::footprint_gb().unwrap_or(0.0);
                if (last - now).abs() < 0.05 || t.elapsed().as_secs() > 20 {
                    break;
                }
                last = now;
            }
            println!("{:>6.2} GB  {what} (settled in {:.1}s)", crate::memwatch::footprint_gb().unwrap_or(0.0), t.elapsed().as_secs_f64());
        };
        mem("start");
        // `PLAKAT_PROBE_QUANT=1` probes the quantized tier, as `run_jobs` stages it.
        let quant = std::env::var("PLAKAT_PROBE_QUANT").is_ok();
        // The highest footprint since the last call, sampled every 20 ms.
        let high = std::sync::Arc::new(std::sync::Mutex::new(0f64));
        {
            let high = high.clone();
            std::thread::spawn(move || loop {
                let now = crate::memwatch::footprint_gb().unwrap_or(0.0);
                let mut h = high.lock().unwrap();
                *h = h.max(now);
                drop(h);
                std::thread::sleep(std::time::Duration::from_millis(20));
            });
        }
        let peak = |what: &str| println!("{:>6.2} GB  peak during {what}", std::mem::take(&mut *high.lock().unwrap()));
        let device = if quant && base.is_metal() { Device::Cpu } else { stage_device(&base).unwrap() };
        let mut enc = TextEncoders::load_with(&repo, &device, quant).await.unwrap();
        mem("encoders loaded");
        let mut pos = enc.encode("A red fox sitting in fresh snow at dawn", DEFAULT_MAX_SEQ).unwrap();
        let mut neg = enc.encode("", DEFAULT_MAX_SEQ).unwrap();
        for e in [&mut pos, &mut neg] {
            (e.qwen, e.pooled) = (cpu(&e.qwen), cpu(&e.pooled));
        }
        mem("two texts encoded");
        drop((enc, device));
        mem("encoders dropped");
        settle("encoders dropped");
        peak("the text stage");
        let device = stage_device(&base).unwrap();
        let dit = Dit::load_with(&repo, &device, quant).await.unwrap();
        mem("DiT loaded");
        peak("the DiT's load");
        let size = std::env::var("PLAKAT_PROBE_SIZE").unwrap_or_else(|_| "1024x1024".into());
        let (w, h) = size.split_once('x').map(|(w, h)| (w.parse::<usize>().unwrap(), h.parse::<usize>().unwrap())).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, dit.cfg.in_visual_dim, h / 8, w / 8), &device).unwrap();
        let mut nohook: Option<&mut dyn StepHook> = None;
        // One tensor left on a stage's device pins that device's whole pool, hence the inner scope.
        let t_denoise = std::time::Instant::now();
        let latent = {
            let on_gpu = denoise_observed(&dit, &pos, Some(&neg), &noise, 3, 3.5, &mut nohook, "probe", &mut |i, _| mem(&format!("into step {i}"))).unwrap();
            println!("          {:.1}s a step", t_denoise.elapsed().as_secs_f64() / 3.0);
            cpu(&on_gpu)
        };
        mem(&format!("denoised at {w}x{h}"));
        peak("the denoise");
        drop((dit, pos, neg, noise, device));
        mem("DiT dropped");
        settle("DiT dropped");
        // The two decodes, each on its own device and read back, so the footprint is what they touched.
        let mut images = Vec::new();
        for tiled in [true, false] {
            let device = stage_device(&base).unwrap();
            let vae = Vae::load(&repo, &device).await.unwrap();
            let t = std::time::Instant::now();
            let z = latent.to_device(&device).unwrap();
            images.push(cpu(&if tiled { decode_tiled(&vae, &z) } else { vae.decode(&z) }.unwrap()));
            drop(z);
            mem(&format!("decoded, {} ({:.1}s)", if tiled { "in tiles" } else { "whole" }, t.elapsed().as_secs_f64()));
            drop((vae, device));
            settle("VAE dropped");
        }
        let mse = (&images[0] - &images[1]).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
        println!("tiles against the whole decode: PSNR {:.1} dB (range 2)", 10.0 * (4.0 / mse).log10());
    }

    #[test]
    fn an_exact_size_must_divide_by_sixteen() {
        assert!(check_exact(1040, 720).is_ok());
        assert!(check_exact(1000, 720).is_err());
        assert!(check_exact(0, 720).is_err());
    }
}
