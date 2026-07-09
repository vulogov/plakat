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

pub async fn run(args: BenchArgs) -> Result<()> {
    let device = crate::device::select(&args.device)?;
    let (width, height) = parse_size(&args.size)?;

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

    // ---- load (timed, cold) ----
    let t_load = Instant::now();
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
    .with_context(|| {
        format!("loading {:?} for bench (SD family only for now)", args.model)
    })?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let tmp = std::env::temp_dir().join(format!("plakat-bench-{}", std::process::id()));

    // ---- generate (timed, repeatable) ----
    let mut samples = Vec::with_capacity(args.repeat.max(1));
    for _ in 0..args.repeat.max(1) {
        let mut hook = TimingHook::default();
        let req = crate::pipelines::t2i::GenRequest {
            prompt: "a portrait of a red fox in a sunlit forest, detailed fur".into(),
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
        let t_gen = Instant::now();
        pipeline.generate_hooked(&req, &[], Some(&mut hook))?;
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;
        samples.push(decompose(load_ms, gen_ms, t_gen, &hook.stamps, peak.load(Ordering::Relaxed)));
    }

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
