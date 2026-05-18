//! Microbenchmark for `clip_image_tensor` — measures the per-call cost
//! of the CLIP-H image preprocess in isolation (no model load, no
//! detection, no network).
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example bench_preprocess -- <PATH-TO-IMAGE> [ITERATIONS]
//! ```
//!
//! Defaults to 1000 iterations. Prints total + mean + min + max in
//! microseconds.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_preprocess <PATH-TO-IMAGE> [ITERATIONS]");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let iters: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    // CPU device — keeps measurement focused on the CPU preprocess loop
    // rather than including any GPU-transfer overhead.
    let device = Device::Cpu;

    // Warm-up: one untimed call. Touches the disk page cache and the
    // image decoder's first-time-allocations.
    plakat::imaging::preprocess::clip_image_tensor(&path, 224, &device, DType::F32)
        .context("warm-up clip_image_tensor failed")?;

    let mut samples = Vec::with_capacity(iters as usize);
    let total_start = Instant::now();
    for _ in 0..iters {
        let t0 = Instant::now();
        let _t = plakat::imaging::preprocess::clip_image_tensor(&path, 224, &device, DType::F32)?;
        samples.push(t0.elapsed().as_micros() as u64);
    }
    let total = total_start.elapsed();

    samples.sort_unstable();
    let n = samples.len() as u64;
    let sum: u64 = samples.iter().sum();
    let mean = sum / n;
    let min = samples[0];
    let max = *samples.last().unwrap();
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() as f64 * 0.95) as usize];

    println!("clip_image_tensor x{} on {}", iters, path.display());
    println!("  total: {:.2} ms", total.as_secs_f64() * 1000.0);
    println!("  mean:  {} µs", mean);
    println!("  p50:   {} µs", p50);
    println!("  p95:   {} µs", p95);
    println!("  min:   {} µs", min);
    println!("  max:   {} µs", max);

    Ok(())
}
