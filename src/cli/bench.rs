//! `plakat bench` — a real, reproducible per-model generation benchmark.
//!
//! **Phase 0 of the 2.4 performance pass: the ruler.** Every optimization in the pass must show
//! a before/after number from this harness. It runs a real generation and decomposes the
//! wall-clock into **load / encode+first-step / per-step / VAE-decode tail / total**, and tracks
//! **peak RSS** via a background sampler. Reproducible (fixed prompt + seed).
//!
//! SD family for now (`sd15` / `sd21` / `sdxl` / `pony` / `sdxl-turbo`); PixArt / SD3.5 / Cascade
//! / Flux land in later phases (each has its own `Pipeline::load` + `generate_hooked`).
//!
//! ```text
//! plakat bench sdxl --size 1024x1024 --steps 25 --repeat 3
//! plakat bench sd15 --json          # machine-readable, for the perf CI gate
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::step_hook::{StepControl, StepHook};

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// Model alias to benchmark (SD family for now: sd15 / sd21 / sdxl / pony / sdxl-turbo).
    #[arg(default_value = "sd15")]
    pub model: String,
    /// Device: auto / metal / cuda / cpu.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Output size, `WxH`.
    #[arg(long, default_value = "512x512")]
    pub size: String,
    /// Denoise steps.
    #[arg(long, default_value_t = 20)]
    pub steps: usize,
    /// Guidance scale.
    #[arg(long, default_value_t = 7.5)]
    pub guidance: f64,
    /// Repeat the timed generation K times; report the best (min total) — warms caches, drops
    /// the cold-start outlier. Model load is timed once (cold).
    #[arg(long, default_value_t = 1)]
    pub repeat: usize,
    /// Emit JSON instead of a table (for the perf CI gate).
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

/// Records a monotonic timestamp at each denoise-step boundary. `on_step` fires once per step
/// (after that step's compute), so consecutive stamps bound per-step latency.
#[derive(Default)]
struct TimingHook {
    stamps: Vec<Instant>,
}

impl StepHook for TimingHook {
    fn on_step(&mut self, _step: usize, _total: usize) -> StepControl {
        self.stamps.push(Instant::now());
        StepControl::Continue
    }
}

/// One timed run's decomposition, milliseconds.
#[derive(Clone)]
struct Sample {
    load_ms: f64,
    encode_first_ms: f64, // gen-start → first step: setup + text-encode + first-step compute
    per_step_ms: f64,     // mean inter-step latency
    steps_span_ms: f64,   // first step → last step
    vae_tail_ms: f64,     // last step → gen return: last-step compute + VAE decode + save
    gen_ms: f64,          // the whole generate call
    total_ms: f64,        // load + gen
    peak_rss_gb: f64,
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .with_context(|| format!("--size must be WxH (got {s:?})"))?;
    Ok((w.trim().parse().context("width")?, h.trim().parse().context("height")?))
}

/// The pipeline family a model alias maps to (each has its own load + hook-capable generate).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Family {
    Sd,
    PixArt,
    Sd3,
    Cascade,
    Flux,
}

fn family_of(model: &str) -> Family {
    let m = model.to_lowercase();
    if m.contains("flux") {
        Family::Flux
    } else if m.contains("cascade") {
        Family::Cascade
    } else if m.contains("sd3") || m.contains("sd35") {
        Family::Sd3
    } else if m.contains("pixart") {
        Family::PixArt
    } else {
        Family::Sd
    }
}

fn sd3_variant(model: &str) -> crate::pipelines::sd3::Variant {
    use crate::pipelines::sd3::Variant;
    let m = model.to_lowercase();
    if m.contains("large") && m.contains("turbo") {
        Variant::Sd35LargeTurbo
    } else if m.contains("large") {
        Variant::Sd35Large
    } else if m.contains("sd3-medium") || m == "sd3" {
        Variant::Sd3Medium
    } else {
        Variant::Sd35Medium
    }
}

/// Run `repeat` timed generations, decomposing each from the per-step stamps `gen` records.
fn time_runs<F>(load_ms: f64, repeat: usize, peak: &AtomicU64, mut gen_fn: F) -> Result<Vec<Sample>>
where
    F: FnMut(&mut TimingHook) -> Result<()>,
{
    let mut samples = Vec::with_capacity(repeat.max(1));
    for _ in 0..repeat.max(1) {
        let mut hook = TimingHook::default();
        let t_gen = Instant::now();
        gen_fn(&mut hook)?;
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;
        samples.push(decompose(load_ms, gen_ms, t_gen, &hook.stamps, peak.load(Ordering::Relaxed)));
    }
    Ok(samples)
}

pub async fn run(args: BenchArgs) -> Result<()> {
    let device = crate::device::select(&args.device)?;
    let (width, height) = parse_size(&args.size)?;
    let family = family_of(&args.model);

    // Background peak-RSS sampler (poll every 40ms; store max bytes).
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let sampler = {
        let (stop, peak) = (stop.clone(), peak.clone());
        std::thread::spawn(move || {
            use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate};
            let pid = Pid::from_u32(std::process::id());
            let mut sys = sysinfo::System::new();
            while !stop.load(Ordering::Relaxed) {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    true,
                    ProcessRefreshKind::everything(),
                );
                if let Some(p) = sys.process(pid) {
                    peak.fetch_max(p.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(40));
            }
        })
    };

    let tmp = std::env::temp_dir().join(format!("plakat-bench-{}", std::process::id()));
    let prompt = "a portrait of a red fox in a sunlit forest, detailed fur";
    let repo = crate::hf::resolve_alias(&args.model).to_string();

    // ---- load (timed, cold) + generate (timed), dispatched by family ----
    let samples = match family {
        Family::Sd => {
            let t = Instant::now();
            let pipeline = crate::pipelines::t2i::Pipeline::load(crate::pipelines::t2i::LoadRequest {
                model: args.model.clone(),
                device: device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                use_refiner: false,
                embeddings: Vec::new(),
                vae_cache: None,
            })
            .await
            .with_context(|| format!("loading {:?}", args.model))?;
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            time_runs(load_ms, args.repeat, &peak, |hook| {
                let req = crate::pipelines::t2i::GenRequest {
                    prompt: prompt.into(),
                    negative: "blurry".into(),
                    width,
                    height,
                    count: 1,
                    steps: args.steps,
                    guidance: args.guidance,
                    seed: Some(42),
                    out_dir: tmp.clone(),
                    scheduler: SchedulerKind::default(),
                    refine: None,
                    refine_strength: 0.3,
                    refiner_frac: None,
                    clip_skip: 1,
                    metadata: None,
                    preview_every: None,
                    preview_size: None,
                    output_format: crate::imaging::io::OutputFormat::Png,
                };
                pipeline.generate_hooked(&req, &[], Some(hook))
            })?
        }
        Family::PixArt => {
            let t = Instant::now();
            let mut pipeline = crate::pipelines::pixart::Pipeline::load(
                crate::pipelines::pixart::LoadRequest {
                    repo: repo.clone(),
                    device: device.clone(),
                    vae_cache: None,
                    loras: Vec::new(),
                    lora_scale: 1.0,
                },
            )
            .await
            .with_context(|| format!("loading PixArt {:?}", args.model))?;
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            time_runs(load_ms, args.repeat, &peak, |hook| {
                let mut opt: Option<&mut dyn StepHook> = Some(hook);
                pipeline
                    .generate(prompt, "blurry", width, height, args.steps, args.guidance, 42,
                        SchedulerKind::default(), &mut opt)
                    .map(|_| ())
            })?
        }
        Family::Sd3 => {
            let t = Instant::now();
            let mut pipeline = crate::pipelines::sd3::Pipeline::load(crate::pipelines::sd3::LoadRequest {
                variant: sd3_variant(&args.model),
                repo: repo.clone(),
                device: device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                controlnets: Vec::new(),
                embeddings: Vec::new(),
            })
            .await
            .with_context(|| format!("loading SD3 {:?}", args.model))?;
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            time_runs(load_ms, args.repeat, &peak, |hook| {
                let req = crate::pipelines::sd3::GenRequest {
                    prompt: prompt.into(),
                    negative: "blurry".into(),
                    width,
                    height,
                    count: 1,
                    steps: Some(args.steps),
                    guidance: Some(args.guidance),
                    seed: Some(42),
                    out_dir: tmp.clone(),
                    init_image: None,
                    mask: None,
                    mask_feather: 0,
                    mask_invert: false,
                    strength: None,
                    tiled: None,
                    regions: Vec::new(),
                    controlnet_conditioning: Vec::new(),
                    output_format: crate::imaging::io::OutputFormat::Png,
                };
                pipeline.generate_hooked(&req, Some(hook))
            })?
        }
        Family::Cascade => {
            let t = Instant::now();
            let mut pipeline = crate::pipelines::cascade::Pipeline::load(
                crate::pipelines::cascade::LoadRequest {
                    repo: repo.clone(),
                    device: device.clone(),
                    loras: Vec::new(),
                    lora_scale: 1.0,
                    controlnet_weights: None,
                    image_encoder_weights: None,
                },
            )
            .await
            .with_context(|| format!("loading Cascade {:?}", args.model))?;
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            // Cascade output is square (Stage-C prior is fixed 24²×16). Split the step budget
            // Stage-C 2/3 : Stage-B 1/3 (the CLI default). The hook fires across both stages.
            let stage_c = (args.steps * 2).div_ceil(3).max(1);
            let stage_b = args.steps.saturating_sub(stage_c).max(1);
            time_runs(load_ms, args.repeat, &peak, |hook| {
                let mut opt: Option<&mut dyn StepHook> = Some(hook);
                pipeline
                    .generate(prompt, "blurry", width, stage_c, stage_b, args.guidance, 1.1, 42,
                        SchedulerKind::default(), None, &mut opt)
                    .map(|_| ())
            })?
        }
        Family::Flux => {
            stop.store(true, Ordering::Relaxed);
            let _ = sampler.join();
            anyhow::bail!(
                "plakat bench defers Flux: it can't be tested faithfully on this hardware \
                 (GGUF-quantized Flux is broken on Metal — candle 0.10.2 kernel bug — and \
                 BF16 Flux is too large to bench meaningfully here). Wired: SD / PixArt / SD3 / Cascade."
            );
        }
    };

    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let _ = std::fs::remove_dir_all(&tmp);

    // Best = min total (drops the cold outlier when --repeat > 1).
    let best = samples
        .iter()
        .min_by(|a, b| a.total_ms.total_cmp(&b.total_ms))
        .expect("at least one sample")
        .clone();

    report(&args, &device, width, height, &best);
    Ok(())
}

/// Split a run into phases from the per-step timestamps.
fn decompose(load_ms: f64, gen_ms: f64, t_gen: Instant, stamps: &[Instant], peak_bytes: u64) -> Sample {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let encode_first_ms = stamps.first().map(|s| ms(s.duration_since(t_gen))).unwrap_or(0.0);
    let (steps_span_ms, per_step_ms) = if stamps.len() >= 2 {
        let span = ms(stamps.last().unwrap().duration_since(*stamps.first().unwrap()));
        (span, span / (stamps.len() - 1) as f64)
    } else {
        (0.0, 0.0)
    };
    // VAE tail = whole gen minus (start → last step).
    let vae_tail_ms = stamps
        .last()
        .map(|s| gen_ms - ms(s.duration_since(t_gen)))
        .unwrap_or(0.0)
        .max(0.0);
    Sample {
        load_ms,
        encode_first_ms,
        per_step_ms,
        steps_span_ms,
        vae_tail_ms,
        gen_ms,
        total_ms: load_ms + gen_ms,
        peak_rss_gb: peak_bytes as f64 / 1e9,
    }
}

fn report(args: &BenchArgs, device: &candle_core::Device, w: u32, h: u32, s: &Sample) {
    let dev = format!("{device:?}");
    if args.json {
        println!(
            "{{\"model\":{:?},\"device\":{:?},\"width\":{w},\"height\":{h},\"steps\":{},\
             \"load_ms\":{:.1},\"encode_first_ms\":{:.1},\"per_step_ms\":{:.2},\
             \"steps_span_ms\":{:.1},\"vae_tail_ms\":{:.1},\"gen_ms\":{:.1},\
             \"total_ms\":{:.1},\"peak_rss_gb\":{:.2}}}",
            args.model, dev, args.steps, s.load_ms, s.encode_first_ms, s.per_step_ms,
            s.steps_span_ms, s.vae_tail_ms, s.gen_ms, s.total_ms, s.peak_rss_gb,
        );
        return;
    }
    println!("plakat bench — {} @ {} · {w}x{h} · {} steps", args.model, dev, args.steps);
    println!("  load (cold)     {:>9.1} ms", s.load_ms);
    println!("  encode + first  {:>9.1} ms", s.encode_first_ms);
    println!("  per step (mean) {:>9.2} ms  ({} steps, {:.1} ms span)", s.per_step_ms, args.steps, s.steps_span_ms);
    println!("  VAE decode tail {:>9.1} ms", s.vae_tail_ms);
    println!("  generate        {:>9.1} ms", s.gen_ms);
    println!("  ── total        {:>9.1} ms", s.total_ms);
    println!("  peak RSS        {:>9.2} GB", s.peak_rss_gb);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_of_dispatches_by_alias() {
        assert_eq!(family_of("sd15"), Family::Sd);
        assert_eq!(family_of("sdxl"), Family::Sd);
        assert_eq!(family_of("pixart"), Family::PixArt);
        assert_eq!(family_of("sd35-medium"), Family::Sd3);
        assert_eq!(family_of("sd3-medium"), Family::Sd3);
        assert_eq!(family_of("stable-cascade"), Family::Cascade);
        assert_eq!(family_of("flux-schnell"), Family::Flux);
    }

    #[test]
    fn sd3_variant_maps_aliases() {
        use crate::pipelines::sd3::Variant;
        assert!(matches!(sd3_variant("sd35-medium"), Variant::Sd35Medium));
        assert!(matches!(sd3_variant("sd35-large"), Variant::Sd35Large));
        assert!(matches!(sd3_variant("sd35-large-turbo"), Variant::Sd35LargeTurbo));
        assert!(matches!(sd3_variant("sd3-medium"), Variant::Sd3Medium));
    }

    #[test]
    fn parse_size_accepts_wxh() {
        assert_eq!(parse_size("512x768").unwrap(), (512, 768));
        assert_eq!(parse_size("1024X1024").unwrap(), (1024, 1024));
        assert!(parse_size("512").is_err());
    }

    #[test]
    fn decompose_splits_phases_from_stamps() {
        let t0 = Instant::now();
        // Three steps at +100, +200, +300ms; gen took 350ms total.
        let stamps = vec![
            t0 + Duration::from_millis(100),
            t0 + Duration::from_millis(200),
            t0 + Duration::from_millis(300),
        ];
        let s = decompose(500.0, 350.0, t0, &stamps, 8_000_000_000);
        assert!((s.encode_first_ms - 100.0).abs() < 1.0);
        assert!((s.per_step_ms - 100.0).abs() < 1.0); // (300-100)/2
        assert!((s.vae_tail_ms - 50.0).abs() < 1.0); // 350 - 300
        assert!((s.total_ms - 850.0).abs() < 1.0); // 500 + 350
        assert!((s.peak_rss_gb - 8.0).abs() < 0.01);
    }
}
