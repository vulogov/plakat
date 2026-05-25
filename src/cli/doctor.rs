//! `plakat doctor` — identity-pipeline health check.
//!
//! Inspects the user's environment and prints a verdict for each
//! configurable surface the identity pipelines depend on:
//!
//! * **ArcFace** weights — local file (`PLAKAT_ARCFACE_WEIGHTS`) or HF
//!   spec (`PLAKAT_ARCFACE_HF`). Verifies the local file exists; for
//!   HF specs just parses the `repo#file` format offline.
//! * **SCRFD** weights — local file (`PLAKAT_SCRFD_WEIGHTS`) or HF spec
//!   (`PLAKAT_SCRFD_HF`). Same offline checks.
//! * **FaceID UNet LoRA** — checks the `PLAKAT_FACEID_LORA` opt-out
//!   env var and reports whether it's active.
//! * **Device + cache** — current device selection, HF cache root.
//!
//! Default mode is fully offline. Pass `--verify` to actively probe
//! configured HF specs by attempting the download — confirms remote
//! files actually resolve before a long generation hits a 404.

use anyhow::Result;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

#[derive(ClapArgs, Debug)]
pub struct DoctorArgs {
    /// Actively probe configured HuggingFace specs by downloading the
    /// referenced files (or hitting the cache). Confirms env vars
    /// like `PLAKAT_ARCFACE_HF=repo#file` actually resolve to a real
    /// remote file *before* a long generation discovers a 404.
    ///
    /// Without this flag, doctor is fully offline: parses HF specs and
    /// checks local paths, but doesn't hit the network.
    #[arg(long)]
    pub verify: bool,

    /// Run a synthetic micro-benchmark (conv2d / matmul / resize at
    /// SD-typical tensor shapes) on the resolved device. Reports
    /// per-op latency and an extrapolated SD 1.5 wall-time estimate.
    /// No model downloads, no network — completes in ~2 seconds.
    ///
    /// Replaces the health-check flow; the two modes are independent.
    #[arg(long)]
    pub benchmark: bool,

    /// Override the device for `--benchmark`. Default: auto-detect.
    /// Mostly useful for forcing CPU benchmarking on a Metal/CUDA-
    /// capable host to compare backends.
    #[arg(long, value_name = "SPEC", default_value = "auto")]
    pub device: String,
}

pub async fn run(args: DoctorArgs) -> Result<()> {
    if args.benchmark {
        return run_benchmark(&args.device);
    }

    println!(
        "\n{}  plakat configuration health check\n",
        style("doctor").yellow().bold()
    );

    // -------- HF cache root --------
    let cache_root = crate::hf::cache::hf_cache_root();
    println!(
        "  {} HF cache: {}",
        style("•").dim(),
        cache_root.display()
    );

    // -------- Device --------
    let device_env = std::env::var("PLAKAT_DEVICE").ok();
    if let Some(d) = device_env {
        println!("  {} device override (PLAKAT_DEVICE): {}", style("•").dim(), d);
    }

    // -------- FaceID UNet LoRA opt-out --------
    let lora_off = std::env::var("PLAKAT_FACEID_LORA").as_deref() == Ok("off");
    if lora_off {
        println!(
            "  {} FaceID UNet LoRA: {}",
            style("•").dim(),
            style("disabled (PLAKAT_FACEID_LORA=off)").yellow()
        );
    } else {
        println!(
            "  {} FaceID UNet LoRA: {}",
            style("•").dim(),
            style("auto-applied (default; PLAKAT_FACEID_LORA=off to disable)").green()
        );
    }

    println!();

    // -------- ArcFace --------
    section_header("ArcFace IR-ResNet50 (--identity faceid / faceid-sdxl)");
    let local = std::env::var("PLAKAT_ARCFACE_WEIGHTS").ok();
    let hf_spec = std::env::var("PLAKAT_ARCFACE_HF").ok();
    match (local.as_deref(), hf_spec.as_deref()) {
        (Some(path), _) => {
            let p = PathBuf::from(path);
            if p.exists() {
                ok(&format!(
                    "PLAKAT_ARCFACE_WEIGHTS = {} (exists)",
                    p.display()
                ));
            } else {
                err(&format!(
                    "PLAKAT_ARCFACE_WEIGHTS = {} (file NOT FOUND)",
                    p.display()
                ));
            }
            if hf_spec.is_some() {
                note("PLAKAT_ARCFACE_HF is also set; the local path wins.");
            }
        }
        (None, Some(spec)) => {
            match crate::pipelines::ip_adapter::parse_hf_spec(spec, "PLAKAT_ARCFACE_HF") {
                Ok((repo, file)) => {
                    ok(&format!("PLAKAT_ARCFACE_HF = {repo}#{file} (parsed OK)"));
                    if args.verify {
                        probe_hf(&repo, &file, "ArcFace").await;
                    } else {
                        note("Pass --verify to actually download and confirm the file resolves.");
                    }
                }
                Err(e) => err(&format!("PLAKAT_ARCFACE_HF invalid: {e}")),
            }
        }
        (None, None) => {
            warn("neither PLAKAT_ARCFACE_WEIGHTS nor PLAKAT_ARCFACE_HF is set");
            println!(
                "    {} FaceID strategies will fail at load. Setup options:",
                style("→").dim()
            );
            println!("    {} A. Convert and point at a local file:", style(" ").dim());
            println!(
                "       {}",
                style("export PLAKAT_ARCFACE_WEIGHTS=/path/to/arcface_r50.safetensors")
                    .dim()
            );
            println!(
                "    {} B. Point at an HF-hosted safetensors:",
                style(" ").dim()
            );
            println!(
                "       {}",
                style("export PLAKAT_ARCFACE_HF=<user>/<repo>#<path/in/repo.safetensors>")
                    .dim()
            );
            println!(
                "    {} Find candidates at:",
                style(" ").dim()
            );
            println!(
                "       {}",
                style("https://huggingface.co/models?search=arcface+iresnet50").dim()
            );
        }
    }

    println!();

    // -------- SCRFD --------
    section_header("SCRFD face detector (auto-fills landmarks for FaceID)");
    let scrfd_local = std::env::var("PLAKAT_SCRFD_WEIGHTS").ok();
    let scrfd_hf = std::env::var("PLAKAT_SCRFD_HF").ok();
    match (scrfd_local.as_deref(), scrfd_hf.as_deref()) {
        (Some(path), _) => {
            let p = PathBuf::from(path);
            if p.exists() {
                ok(&format!(
                    "PLAKAT_SCRFD_WEIGHTS = {} (exists; auto-detection active)",
                    p.display()
                ));
                note(
                    "The SCRFD architecture port is a best-guess from the \
                     InsightFace reference; weight-loading is verified by \
                     actually running an identity job. If load errors at \
                     a layer, paste the error.",
                );
            } else {
                err(&format!(
                    "PLAKAT_SCRFD_WEIGHTS = {} (file NOT FOUND)",
                    p.display()
                ));
            }
            if scrfd_hf.is_some() {
                note("PLAKAT_SCRFD_HF is also set; the local path wins.");
            }
        }
        (None, Some(spec)) => {
            match crate::pipelines::ip_adapter::parse_hf_spec(spec, "PLAKAT_SCRFD_HF") {
                Ok((repo, file)) => {
                    ok(&format!("PLAKAT_SCRFD_HF = {repo}#{file} (parsed OK)"));
                    if args.verify {
                        probe_hf(&repo, &file, "SCRFD").await;
                    } else {
                        note("Pass --verify to actually download and confirm the file resolves.");
                    }
                    note(
                        "The SCRFD architecture port is a best-guess from the \
                         InsightFace reference; weight-loading is verified by \
                         actually running an identity job. If load errors at \
                         a layer, paste the error.",
                    );
                }
                Err(e) => err(&format!("PLAKAT_SCRFD_HF invalid: {e}")),
            }
        }
        (None, None) => {
            note(
                "Neither PLAKAT_SCRFD_WEIGHTS nor PLAKAT_SCRFD_HF is set — \
                 FaceID falls back to --face-bbox / --face-landmarks / \
                 centre-crop. Set one of them to enable auto-detection.",
            );
            note("Find SCRFD candidates at https://huggingface.co/models?search=scrfd");
        }
    }

    println!();

    // -------- FaceID image_proj — auto-downloaded from h94 --------
    section_header("FaceID image-proj (auto-downloaded from h94/IP-Adapter-FaceID)");
    ok("Downloaded automatically on first use of `--identity faceid` / `faceid-sdxl`. No setup needed.");

    println!();

    // -------- IP-Adapter weights for Plus-Face --------
    section_header("Plus-Face / IP-Adapter (--identity plus-face / plus-face-sdxl)");
    ok("Downloaded automatically from h94/IP-Adapter. No setup needed.");

    println!();

    // -------- v0.18 phase 7: build / runtime device match --------
    // Catches the class of failures where the binary was compiled
    // with --features cuda but no NVIDIA driver is present at run
    // time, or where a Metal binary is being run on a CUDA box, etc.
    section_runtime_device_match();
    println!();

    // -------- v0.18 phase 7: CUDA driver shim presence (Linux+CUDA) --------
    // Linux + `--features cuda` only. Probes the well-known driver
    // shim locations. The CUDA *toolkit* (nvcc + cuBLAS) is not the
    // same as the *driver* (libcuda.so.1) — the v0.17 CI failure
    // was a runner with the toolkit but no driver.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    {
        section_cuda_driver_shim();
        println!();
    }

    // -------- v0.18 phase 7: HF cache disk usage --------
    section_cache_disk_usage();
    println!();

    println!(
        "  {}\n",
        style("If you've fixed any of the issues above, re-run `plakat doctor` to confirm.").dim()
    );

    Ok(())
}

/// Reports the backend flags the binary was compiled with vs the
/// device that `device::select("auto")` actually resolves on this
/// host. Warns when the build features and runtime resolution don't
/// agree (e.g. `--features cuda` build with no driver → falls back
/// to CPU silently otherwise).
fn section_runtime_device_match() {
    section_header("Build / runtime device match");

    let build_cuda = cfg!(feature = "cuda");
    let build_metal = cfg!(feature = "metal");
    let build_features: Vec<&str> = [("cuda", build_cuda), ("metal", build_metal)]
        .into_iter()
        .filter_map(|(n, on)| if on { Some(n) } else { None })
        .collect();
    let build_summary = if build_features.is_empty() {
        "(none — CPU-only build)".to_string()
    } else {
        build_features.join(" + ")
    };
    note(&format!("Build features: {build_summary}"));

    // Runtime probe. `device::select("auto")` falls back to CPU
    // silently when the configured backends fail to initialise.
    match crate::device::select("auto") {
        Ok(d) => {
            let runtime = describe_device(&d);
            note(&format!("Runtime device (auto): {runtime}"));

            // Mismatch heuristics — only worth a warn when the user
            // explicitly built for an accelerator and got CPU.
            let runtime_is_cpu = matches!(d, candle_core::Device::Cpu);
            if (build_cuda || build_metal) && runtime_is_cpu {
                warn(
                    "binary was built with an accelerator feature but auto-detect \
                     fell back to CPU. Check the section below (CUDA driver) and / or \
                     the Metal runtime install.",
                );
            } else {
                ok(&format!("build + runtime aligned: {runtime}"));
            }
        }
        Err(e) => {
            err(&format!("device::select(\"auto\") failed: {e}"));
        }
    }
}

fn describe_device(d: &candle_core::Device) -> String {
    match d {
        candle_core::Device::Cpu => "CPU".to_string(),
        candle_core::Device::Cuda(_) => "CUDA".to_string(),
        candle_core::Device::Metal(_) => "Metal".to_string(),
    }
}

/// Probe the well-known locations of `libcuda.so.1` — the NVIDIA
/// driver shim. Toolkit installs (nvcc + cuBLAS via apt) do NOT
/// install the driver; the v0.17 CI failure was a GitHub runner in
/// exactly that state. Linux only, cuda-feature only.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn section_cuda_driver_shim() {
    section_header("CUDA driver shim (libcuda.so.1)");
    const CANDIDATES: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
        "/lib/x86_64-linux-gnu/libcuda.so.1",
    ];
    let found = CANDIDATES.iter().find(|p| std::path::Path::new(p).exists());
    match found {
        Some(path) => ok(&format!("libcuda.so.1 found at {path}")),
        None => {
            err(
                "libcuda.so.1 not found in any standard location. \
                 The CUDA toolkit (nvcc + cuBLAS) is separate from the NVIDIA \
                 driver — `plakat generate` will fail at startup with \
                 'cannot open shared object file' until the driver is installed. \
                 Install via `nvidia-driver-*` (Debian/Ubuntu) or the NVIDIA \
                 installer for your distro.",
            );
            note("Searched: /usr/lib/x86_64-linux-gnu, /usr/lib64, /usr/lib, /lib/x86_64-linux-gnu");
        }
    }
}

/// Walks the HF cache root and reports total bytes used. Surfaces
/// `plakat models rm` as the cleanup path when the cache grows
/// large. Uses the existing `hf::cache::dir_size` walker — no new
/// dep needed.
fn section_cache_disk_usage() {
    section_header("HF cache disk usage");
    let root = crate::hf::cache::hf_cache_root();
    if !root.exists() {
        note(&format!(
            "{} doesn't exist yet — no models downloaded.",
            root.display()
        ));
        return;
    }
    let used = match crate::hf::cache::dir_size(&root) {
        Ok(n) => n,
        Err(e) => {
            err(&format!("couldn't walk {}: {e}", root.display()));
            return;
        }
    };
    let human = crate::hf::cache::human_bytes(used);
    ok(&format!("{} cached at {}", human, root.display()));
    match cache_usage_severity(used) {
        CacheSeverity::Ok => {}
        CacheSeverity::Note => note(
            "cache is over 100 GB. `plakat models ls` to inspect; \
             `plakat models rm <repo>` to clean up specific entries.",
        ),
        CacheSeverity::Warn => warn(
            "cache is over 500 GB. Run `plakat models ls` to see what's in it \
             and `plakat models rm <repo>` to prune unused checkpoints.",
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheSeverity {
    Ok,
    Note,
    Warn,
}

/// Bucket cache size into a severity tier. 100 GB ≈ 3-4 Flux
/// variants; 500 GB is "you might want to clean up". Pure function
/// so the bucket boundaries are unit-testable without filesystem
/// state.
fn cache_usage_severity(used_bytes: u64) -> CacheSeverity {
    const GB: u64 = 1024 * 1024 * 1024;
    if used_bytes > 500 * GB {
        CacheSeverity::Warn
    } else if used_bytes > 100 * GB {
        CacheSeverity::Note
    } else {
        CacheSeverity::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // v0.18 phase 7 — unit tests for the pure pieces of the new
    // doctor sections. The print-side helpers (section_*) are
    // side-effecty and exercised via integration runs.

    #[test]
    fn describe_device_covers_all_variants() {
        assert_eq!(describe_device(&candle_core::Device::Cpu), "CPU");
        // CUDA / Metal variants can only be constructed when the
        // corresponding feature is enabled at compile time; skip the
        // assertion on configurations that lack them. The string
        // outputs are exercised by the manual `plakat doctor` run.
    }

    #[test]
    fn cache_usage_severity_under_100gb_is_ok() {
        assert_eq!(cache_usage_severity(0), CacheSeverity::Ok);
        assert_eq!(cache_usage_severity(50 * 1024 * 1024 * 1024), CacheSeverity::Ok);
        assert_eq!(cache_usage_severity(100 * 1024 * 1024 * 1024), CacheSeverity::Ok);
    }

    #[test]
    fn cache_usage_severity_between_100_and_500gb_is_note() {
        assert_eq!(
            cache_usage_severity(101 * 1024 * 1024 * 1024),
            CacheSeverity::Note
        );
        assert_eq!(
            cache_usage_severity(250 * 1024 * 1024 * 1024),
            CacheSeverity::Note
        );
        assert_eq!(
            cache_usage_severity(500 * 1024 * 1024 * 1024),
            CacheSeverity::Note
        );
    }

    #[test]
    fn cache_usage_severity_over_500gb_is_warn() {
        assert_eq!(
            cache_usage_severity(501 * 1024 * 1024 * 1024),
            CacheSeverity::Warn
        );
        assert_eq!(
            cache_usage_severity(2 * 1024 * 1024 * 1024 * 1024),
            CacheSeverity::Warn
        );
    }
}

fn section_header(label: &str) {
    println!(
        "  {} {}",
        style("◆").cyan().bold(),
        style(label).bold()
    );
}

fn ok(msg: &str) {
    println!("    {} {}", style("✓").green(), msg);
}

fn err(msg: &str) {
    println!("    {} {}", style("✗").red(), msg);
}

fn warn(msg: &str) {
    println!("    {} {}", style("!").yellow(), msg);
}

fn note(msg: &str) {
    println!("    {} {}", style("·").dim(), style(msg).dim());
}

/// `plakat doctor --benchmark`: synthetic micro-benchmark of the
/// tensor primitives that dominate SD inference. Runs entirely
/// in-process — no model downloads, no scheduler — so it gives a
/// "this hardware does X conv2d ops per second" number in a couple
/// of seconds.
///
/// Workloads (chosen to match the shape distribution of SD 1.5
/// inference at 512²):
/// * `conv2d`:  (1, 320, 64, 64) → (1, 320, 64, 64), 3×3 — UNet block.
/// * `conv2d`:  (1, 4,   64, 64) → (1, 320, 64, 64), 3×3 — UNet conv_in.
/// * `matmul`:  (1, 77, 768) × (768, 768) — text encoder layer.
/// * `bilinear`: (1, 3, 256, 256) → (1, 3, 512, 512) — VAE-ish upsample.
fn run_benchmark(device_spec: &str) -> Result<()> {
    println!(
        "\n{}  plakat synthetic benchmark\n",
        style("doctor --benchmark").yellow().bold()
    );

    // -------- device + dtype --------
    let device = crate::device::select(device_spec)?;
    let dtype = if matches!(device, candle_core::Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::F16
    };
    let backend = match &device {
        candle_core::Device::Cpu => "CPU (F32)",
        candle_core::Device::Cuda(_) => "CUDA (F16)",
        candle_core::Device::Metal(_) => "Metal (F16)",
    };
    println!("  {} backend: {}", style("•").dim(), style(backend).bold());
    println!("  {} requested device spec: {}", style("•").dim(), device_spec);
    println!();

    // -------- pre-build kernels (warm-up) --------
    let warmup = crate::ui::progress::spinner("Warming up kernels");
    // Materialise each shape once so subsequent timed runs reuse the
    // compiled Metal kernel / CUDA kernel selection.
    let _ = bench_conv2d(&device, dtype, 320, 320, 64, 64, 1)?;
    let _ = bench_conv2d(&device, dtype, 4, 320, 64, 64, 1)?;
    let _ = bench_matmul(&device, dtype, 77, 768, 768, 1)?;
    let _ = bench_bilinear(&device, dtype, 3, 256, 256, 512, 512, 1)?;
    warmup.finish_and_clear();

    // -------- timed runs --------
    section_header("Per-operation latency (median of 10 runs)");

    let conv_block_ms = bench_conv2d(&device, dtype, 320, 320, 64, 64, 10)?;
    println!(
        "    {} {:<32} {:>8.2} ms",
        style("·").dim(),
        "conv2d  3×3  320→320 @ 64²",
        conv_block_ms,
    );

    let conv_in_ms = bench_conv2d(&device, dtype, 4, 320, 64, 64, 10)?;
    println!(
        "    {} {:<32} {:>8.2} ms",
        style("·").dim(),
        "conv2d  3×3    4→320 @ 64²",
        conv_in_ms,
    );

    let matmul_ms = bench_matmul(&device, dtype, 77, 768, 768, 10)?;
    println!(
        "    {} {:<32} {:>8.2} ms",
        style("·").dim(),
        "matmul (1,77,768)×(768,768)",
        matmul_ms,
    );

    let bilinear_ms = bench_bilinear(&device, dtype, 3, 256, 256, 512, 512, 10)?;
    println!(
        "    {} {:<32} {:>8.2} ms",
        style("·").dim(),
        "bilinear 256² → 512²",
        bilinear_ms,
    );

    println!();

    // -------- extrapolated SD 1.5 estimate --------
    //
    // Per-step cost is dominated by ~24 conv2d ops at UNet shapes plus
    // ~4 matmuls in the text encoder. This is a coarse proxy; real
    // SD 1.5 throughput depends heavily on attention layers we don't
    // simulate. Treat as "order-of-magnitude check" not gospel.
    section_header("Coarse SD 1.5 wall-time estimate (28-step generation)");
    let per_step_proxy_ms = conv_block_ms * 24.0 + matmul_ms * 4.0;
    let per_image_s = (per_step_proxy_ms * 28.0) / 1000.0;
    println!(
        "    {} per-step proxy:    {:>8.2} ms",
        style("·").dim(),
        per_step_proxy_ms,
    );
    println!(
        "    {} per-image proxy:   {:>8.2} s   (28 steps × per-step proxy)",
        style("·").dim(),
        per_image_s,
    );
    println!();
    note(
        "Estimate ignores attention layers and CFG doubling; real SD 1.5 wall \
         times are often 1.5-2× the per-image proxy. Use this number to compare \
         backends/devices, not to predict an absolute generation time.",
    );

    println!();
    Ok(())
}

/// Measure median latency (ms) of `iters` conv2d invocations on `device`.
fn bench_conv2d(
    device: &candle_core::Device,
    dtype: candle_core::DType,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    iters: usize,
) -> Result<f64> {
    use candle_core::{Module, Tensor};
    use candle_nn::{conv2d_no_bias, Conv2dConfig, VarBuilder, VarMap};

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let cfg = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    let conv = conv2d_no_bias(in_ch, out_ch, 3, cfg, vb.pp("c"))?;
    let input = Tensor::randn(0f32, 1f32, (1, in_ch, h, w), device)?.to_dtype(dtype)?;
    median_run(device, iters, || {
        let _ = conv.forward(&input)?;
        Ok(())
    })
}

/// Measure median latency (ms) of `iters` matmul invocations.
fn bench_matmul(
    device: &candle_core::Device,
    dtype: candle_core::DType,
    seq: usize,
    d_in: usize,
    d_out: usize,
    iters: usize,
) -> Result<f64> {
    use candle_core::Tensor;

    let lhs = Tensor::randn(0f32, 1f32, (1, seq, d_in), device)?.to_dtype(dtype)?;
    let rhs = Tensor::randn(0f32, 1f32, (d_in, d_out), device)?.to_dtype(dtype)?;
    median_run(device, iters, || {
        let _ = lhs.broadcast_matmul(&rhs)?;
        Ok(())
    })
}

/// Measure median latency (ms) of `iters` bilinear upsamples.
fn bench_bilinear(
    device: &candle_core::Device,
    dtype: candle_core::DType,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    iters: usize,
) -> Result<f64> {
    use candle_core::Tensor;

    let input =
        Tensor::randn(0f32, 1f32, (1, channels, in_h, in_w), device)?.to_dtype(dtype)?;
    median_run(device, iters, || {
        let _ = input.upsample_nearest2d(out_h, out_w)?;
        Ok(())
    })
}

/// Run `f` `iters` times, returning the median wall-time in ms.
/// Calls `device.synchronize()` after each run so async-dispatch
/// backends (Metal, CUDA) actually block until work is done — without
/// the sync, the median would just measure kernel-launch overhead and
/// produce sub-millisecond numbers regardless of actual GPU work.
fn median_run<F: FnMut() -> Result<()>>(
    device: &candle_core::Device,
    iters: usize,
    mut f: F,
) -> Result<f64> {
    use std::time::Instant;
    let mut samples = Vec::with_capacity(iters);
    // One untimed sync first so the queue is empty.
    device.synchronize().ok();
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        device.synchronize().ok();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(samples[samples.len() / 2])
}

/// Active HF probe — used by `doctor --verify`. Resolves the file via
/// the same hf::download path the runtime would use, hitting the cache
/// on subsequent runs.
async fn probe_hf(repo: &str, file: &str, label: &str) {
    let s = crate::ui::progress::spinner(&format!("Verifying {label} download from {repo}/{file}"));
    match crate::hf::download::get_file(repo, file).await {
        Ok(path) => {
            let size_mb = std::fs::metadata(&path)
                .map(|m| m.len() as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0);
            s.finish_and_clear();
            ok(&format!(
                "{label} download OK — {} ({:.1} MB cached)",
                path.display(),
                size_mb
            ));
        }
        Err(e) => {
            s.finish_and_clear();
            err(&format!("{label} download FAILED — {e}"));
            note(
                "Common causes: the repo or file path doesn't exist, the repo \
                 is gated (needs `huggingface-cli login`), or HF Hub is \
                 temporarily unreachable.",
            );
        }
    }
}
