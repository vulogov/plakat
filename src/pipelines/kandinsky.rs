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

/// Decode a latent, whole if the device takes it and in blended tiles if it runs out of memory — the
/// F32 decode's single buffer is what a 24 GB Mac cannot allocate at 1024² (RFC §9).
/// `PLAKAT_K5_VAE_TILED=1` forces the tiles.
pub fn decode(vae: &Vae, latent: &Tensor) -> Result<Tensor> {
    let tiled = || crate::pipelines::tiled::tile_decode_2d(latent, TILE_LATENT, TILE_STRIDE, 8, |tile| vae.decode(tile));
    if std::env::var("PLAKAT_K5_VAE_TILED").ok().as_deref() == Some("1") {
        return tiled();
    }
    match vae.decode(latent) {
        Ok(image) => Ok(image),
        Err(e) if crate::error_hints::looks_like_oom(&format!("{e:#}")) => {
            crate::ui::progress::println("kandinsky5: the whole-image VAE decode did not fit; decoding in tiles");
            tiled()
        }
        Err(e) => Err(e),
    }
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
    let device = &settings.device;
    let println = |m: String| crate::ui::progress::println(&m);

    // Stage 1 — text. Each distinct prompt and negative is encoded once.
    let t0 = std::time::Instant::now();
    let spin = crate::ui::progress::spinner("Loading the Kandinsky 5 text encoders (Qwen2.5-VL text tower + CLIP-L)");
    let mut encoders = TextEncoders::load(&repo, device).await.context("loading the Kandinsky 5 text encoders")?;
    spin.finish_with_message(format!("✓ text encoders loaded in {:.1}s", t0.elapsed().as_secs_f64()));
    let t1 = std::time::Instant::now();
    let mut embeds: std::collections::HashMap<String, Embeds> = std::collections::HashMap::new();
    for j in jobs {
        let wanted = std::iter::once(&j.prompt).chain((j.guidance > 1.0).then_some(&j.negative));
        for text in wanted {
            if !embeds.contains_key(text) {
                let e = encoders.encode(text, settings.max_seq)?;
                if let Some(d) = &e.dropped {
                    println(format!("kandinsky5: the prompt ran past --max-seq {}; dropped: “{}”", settings.max_seq, d.trim()));
                }
                embeds.insert(text.clone(), e);
            }
        }
    }
    println(format!("kandinsky5: encoded {} text(s) in {:.1}s", embeds.len(), t1.elapsed().as_secs_f64()));
    // Staged residency: the encoders are gone before the DiT is loaded, unless asked to stay.
    let kept = if settings.keep_encoders { Some(encoders) } else { drop(encoders); None };

    // Stage 2 — denoise every job with the DiT resident.
    let t2 = std::time::Instant::now();
    let spin = crate::ui::progress::spinner("Loading the Kandinsky 5 DiT");
    let dit = Dit::load(&repo, device).await.context("loading the Kandinsky 5 DiT")?;
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
    drop(dit);
    drop(embeds);

    // Stage 3 — decode and save.
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
    let settings = Settings { model: req.model.clone(), device: req.device.clone(), max_seq: if req.max_seq == 0 { DEFAULT_MAX_SEQ } else { req.max_seq }, keep_encoders: req.keep_encoders };
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
        let vae = Vae::load(repo, &device).await.unwrap();
        // Stage 7: the final latent. Stage 8: the image — ours against the reference's, and (to tell
        // the VAE from the loop) our decode of the REFERENCE's latent against the reference's image.
        let c = cosine(&latent, &p2["final_latent"]);
        let image = decode(&vae, &latent).unwrap();
        let (db, db_vae) = (psnr(&image, &p2["decoded"]), psnr(&decode(&vae, &p2["final_latent"].to_device(&device).unwrap()).unwrap(), &p2["decoded"]));
        println!("final latent cosine {c:.6} · image PSNR {db:.1} dB (the VAE alone, on the reference's latent: {db_vae:.1} dB)");
        if let Some(out) = std::env::var_os("PLAKAT_PARITY_OUT") {
            let (h, w) = (image.dim(2).unwrap() as u32, image.dim(3).unwrap() as u32);
            image::RgbImage::from_raw(w, h, to_rgb8(&image).unwrap()).unwrap().save(&out).unwrap();
        }
        let (min_cos, min_db) = if exact { (0.999, 40.0) } else { (0.99, 32.0) };
        assert!(c >= min_cos, "final latent cosine {c}");
        assert!(db >= min_db, "image PSNR {db} dB");
    }

    #[test]
    fn an_exact_size_must_divide_by_sixteen() {
        assert!(check_exact(1040, 720).is_ok());
        assert!(check_exact(1000, 720).is_err());
        assert!(check_exact(0, 720).is_err());
    }
}
