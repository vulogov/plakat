//! `plakat doctor` — environment health check.
//!
//! Inspects the user's environment and prints a verdict for each
//! configurable surface plakat depends on:
//!
//! * **ArcFace** weights — local file (`PLAKAT_ARCFACE_WEIGHTS`) or HF
//!   spec (`PLAKAT_ARCFACE_HF`). Verifies the local file exists; for
//!   HF specs just parses the `repo#file` format offline.
//! * **SCRFD** weights — local file (`PLAKAT_SCRFD_WEIGHTS`) or HF spec
//!   (`PLAKAT_SCRFD_HF`). Same offline checks.
//! * **FaceID UNet LoRA** — checks the `PLAKAT_FACEID_LORA` opt-out
//!   env var and reports whether it's active.
//! * **Device + cache** — build / runtime device alignment, HF cache
//!   root, disk usage.
//! * **v0.30 ffmpeg** — version probe for the v0.28 video output
//!   formats and v0.30 `--control-spec video=` per-frame video CN
//!   input decode.
//! * **v0.30 API keys** — presence (never value) of `HF_TOKEN` /
//!   `HUGGING_FACE_HUB_TOKEN` (gated HF repos) and
//!   `CIVITAI_API_KEY` (Civitai rate limits + gated assets).
//!
//! Default mode is fully offline. Pass `--verify` to actively probe
//! configured HF specs by attempting the download — confirms remote
//! files actually resolve before a long generation hits a 404.
//! `--json` emits a structured report covering the device, CUDA
//! driver, cache, ffmpeg, and API-key sections for CI consumption.

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
    #[arg(help_heading = "Checks", long)]
    pub verify: bool,

    /// Run a synthetic micro-benchmark (conv2d / matmul / resize at
    /// SD-typical tensor shapes) on the resolved device. Reports
    /// per-op latency and an extrapolated SD 1.5 wall-time estimate.
    /// No model downloads, no network — completes in ~2 seconds.
    ///
    /// Replaces the health-check flow; the two modes are independent.
    #[arg(help_heading = "Checks", long)]
    pub benchmark: bool,

    /// Override the device for `--benchmark`. Default: auto-detect.
    /// Mostly useful for forcing CPU benchmarking on a Metal/CUDA-
    /// capable host to compare backends.
    #[arg(long, value_name = "SPEC", default_value = "auto")]
    pub device: String,

    /// v0.19: emit a structured JSON report instead of the
    /// human-friendly section blocks. Covers the v0.18 health
    /// checks (build / runtime device match, libcuda driver
    /// shim, HF cache disk usage) plus the plakat version.
    /// FaceID env-var sections stay human-only — they're
    /// configuration inspection, not health checks.
    ///
    /// Designed for CI / scripting; pipe through `jq` for
    /// structured queries. Mutually exclusive with `--benchmark`.
    #[arg(help_heading = "Checks", long, default_value_t = false, conflicts_with = "benchmark")]
    pub json: bool,

    /// v0.33 phase 3: walk every RNG-touching code path across
    /// pipelines and surface its determinism guarantee. Output
    /// covers SD t2i / AnimateDiff / SDXL / SD3 / Flux / stylize /
    /// portrait + scheduler / wildcard / enhancer paths. Surfaces
    /// known gaps (Metal u32 seed truncation, VAE-encode placement
    /// needing verification, non-deterministic --enhance-temp >
    /// 0.0) without fixing them — fixes that need deep changes
    /// defer to v0.34+.
    ///
    /// Combine with `--json` for a machine-readable report.
    /// Mutually exclusive with `--benchmark`.
    #[arg(help_heading = "Checks", long = "reproducibility-check", default_value_t = false, conflicts_with = "benchmark")]
    pub reproducibility_check: bool,

    /// v0.46: report which supported models can run on THIS hardware.
    /// Probes RAM + backend + a conservative memory budget, then for each
    /// model derives its weight size — from the on-disk HF cache (exact),
    /// or, when combined with `--verify`, the HF API (no download) for
    /// models you haven't pulled yet — and judges runs / tight / won't-fit,
    /// naming the tuning lever that helps. `--device` picks which backend to
    /// probe. Combine with `--json` for a machine-readable report.
    #[arg(help_heading = "Checks", long, default_value_t = false, conflicts_with = "benchmark")]
    pub capability: bool,
}

pub async fn run(args: DoctorArgs) -> Result<()> {
    if args.benchmark {
        return run_benchmark(&args.device);
    }
    if args.capability {
        return run_capability(&args).await;
    }
    if args.reproducibility_check {
        return run_reproducibility_check(args.json);
    }
    if args.json {
        return run_json();
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

    // -------- v0.30 phase 4: ffmpeg presence --------
    section_ffmpeg();
    println!();

    // -------- v0.30 phase 4: API keys probe --------
    section_api_keys();
    println!();

    // -------- v5.0.0: persona pipeline --------
    section_persona();

    // -------- v6.0.0: bookart pipeline --------
    section_bookart();
    println!();

    println!(
        "  {}\n",
        style("If you've fixed any of the issues above, re-run `plakat doctor` to confirm.").dim()
    );

    Ok(())
}

/// v5.0.0: the `plakat persona` pipeline (RFC PERSONA-1). Reports which stages are weights-free (run
/// anywhere) vs weight-backed, the identity tiers, and the shared weights the generative stages reuse.
fn section_persona() {
    section_header("persona (5.0.0 — controllable synthetic-person composition)");
    ok("weights-free (run anywhere, no GPU): new · lint · show · interview · geometry · diff");
    note("weight-backed: cast · render · verify · composite · repair · bake · calibrate --from");
    note(
        "identity tiers (§11.4): A = IP-Adapter-Plus-Face (sd15/sd21/sdxl) · B = face-swap bridge \
         (universal, every family) · C = baked TI/LoRA (`persona bake`). `render --tier auto` picks A \
         where an adapter exists, else B.",
    );
    note(
        "reuses the SCRFD detector + ArcFace + inswapper + restore-faces (the swap bridge) and the \
         PIPNet-98 aligner (auto-downloaded from vulogov98/plakat-persona); the same FaceID/SCRFD env \
         vars as the sections above apply.",
    );
    note("landmark topology = WFLW-98; calibration tables ship provisional (see Documentation/PERSONA.md).");
    note("body identity is face-only (§11.7); the neutral lexicon is binding (§7.4/§23.3). No age gate — apparent_age grounds the render like any other model.");
}

/// v6.0.0: the `plakat bookart` pipeline (RFC BOOKART-1) — transparent, print-ready B/W book ornaments.
fn section_bookart() {
    section_header("bookart (6.0.0 — controllable B/W book-ornament composition)");
    ok("weights-free (run anywhere, no GPU): new · lint · show · diff · edit · blend · proof · render (procedural tier)");
    note("weight-backed: render/kit/manuscript on the diffusion & composite tiers (sd15) · kit coherence (CLIP)");
    note(
        "render tiers (§5.3): procedural = vector-native geometric ornament (border/rosette/divider/corner, \
         ZERO weights) · diffusion = pictorial via sd15 + origin LoRA · composite = procedural frame + diffusion inlay.",
    );
    note(
        "origins russian/english/japanese ship as sd15 LoRAs auto-resolved from vulogov98/plakat-bookart \
         (<origin>-sd15.safetensors); `generic` is the LoRA-free line-art fallback so every origin×technique works.",
    );
    note("output contract: primary = transparent, exact-page-sized PNG (luminance-native alpha, DPI in pHYs). Procedural SVG is born-vector (always available); diffusion/composite SVG is a raster trace.");
    note("integration (6.1): scenario `type: bookart` · compile `type: bookart` · Bund `plakat.bookart.*` · library `plakat::api::BookArt` · `render/illustrate --import <album>` (recipe sidecar + tEXt).");
    if cfg!(feature = "bookart-trace") {
        ok("raster→SVG trace ON: `bookart vectorize <raster>` + `render/illustrate --svg` on the diffusion/composite tiers (vtracer).");
    } else {
        note("raster→SVG trace OFF: `bookart vectorize` + pixel-tier `--svg` need `--features bookart-trace` (pulls an extra image stack; procedural `--svg` is always on).");
    }
    note("flagship: `kit` (a coherent matched set) + `manuscript` (a whole book's per-chapter ornaments). See Documentation/BOOKART.md.");
}

/// v0.30 phase 4: ffmpeg presence + version. Required by `plakat
/// animate --format mp4|webm` (output encoding) and by the new v0.30
/// phase 2 `--control-spec ...video=PATH` (input decode). `Frames`
/// and `gif` output formats work without ffmpeg.
fn section_ffmpeg() {
    section_header("ffmpeg (video output + v0.30 control-video decode)");
    match crate::imaging::video::ffmpeg_version() {
        Ok(v) => ok(&format!("ffmpeg present: {v}")),
        Err(_) => {
            warn(
                "ffmpeg not found on PATH. Without it: `plakat animate --format mp4|webm` \
                 bails, and v0.30 `--control-spec ...video=PATH` per-frame video CN can't \
                 decode. `--format frames` and `--format gif` still work.",
            );
            note("Install: macOS `brew install ffmpeg` / Ubuntu `apt install ffmpeg` / Windows `scoop install ffmpeg`.");
        }
    }
}

/// v0.30 phase 4: probe optional API key env vars. Reports presence
/// only — NEVER prints the value, even truncated. Token absence is
/// fine for non-gated workflows; the goal is to surface "I set
/// CIVITAI_API_KEY but the shell didn't export it" before a 401 hits
/// in the middle of a long generation.
fn section_api_keys() {
    section_header("API tokens (optional — gated repos + Civitai rate limits)");
    let hf_set = std::env::var("HF_TOKEN").map(|v| !v.is_empty()).unwrap_or(false);
    let hf_legacy_set = std::env::var("HUGGING_FACE_HUB_TOKEN")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if hf_set || hf_legacy_set {
        let which = if hf_set && hf_legacy_set {
            "HF_TOKEN + HUGGING_FACE_HUB_TOKEN"
        } else if hf_set {
            "HF_TOKEN"
        } else {
            "HUGGING_FACE_HUB_TOKEN"
        };
        ok(&format!("HuggingFace token: present ({which})"));
    } else {
        note(
            "No HF_TOKEN set. Public repos still work; gated ones (Flux Dev, \
             SD3 Medium, ...) bail at download time. Set via \
             `export HF_TOKEN=hf_...`.",
        );
    }

    let civitai_set = std::env::var("CIVITAI_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if civitai_set {
        ok("Civitai token: present (CIVITAI_API_KEY)");
    } else {
        note(
            "No CIVITAI_API_KEY set. Anonymous downloads work for ungated assets \
             but you'll hit rate limits faster; gated NSFW-toggle assets bail. \
             Set via `export CIVITAI_API_KEY=...`.",
        );
    }
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

    // v0.19 — doctor --json structured report.

    #[test]
    fn json_report_serializes_with_version() {
        let report = collect_report();
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains(&format!("\"version\":\"{}\"", env!("CARGO_PKG_VERSION"))));
    }

    #[test]
    fn json_report_device_section_is_consistent() {
        let report = collect_report();
        // Both `build_features` and `runtime` are populated.
        // `aligned` must be `false` only when build features
        // exist but runtime resolved to CPU.
        if !report.device.build_features.is_empty() {
            if report.device.runtime == "CPU" {
                assert!(!report.device.aligned, "built-for-accel + CPU should be misaligned");
            }
        } else {
            // CPU-only build → always aligned (trivially).
            assert!(report.device.aligned);
        }
    }

    #[test]
    fn json_report_cuda_driver_only_applicable_on_linux_cuda() {
        let report = collect_report();
        let expected_applicable =
            cfg!(all(feature = "cuda", target_os = "linux"));
        assert_eq!(report.cuda_driver.applicable, expected_applicable);
        if !expected_applicable {
            assert!(report.cuda_driver.candidates.is_empty());
            assert!(report.cuda_driver.found_at.is_none());
        }
    }

    #[test]
    fn json_report_cache_severity_matches_internal_bucket() {
        let report = collect_report();
        let expected = match report.cache.used_bytes {
            n if n > 500 * 1024 * 1024 * 1024 => "warn",
            n if n > 100 * 1024 * 1024 * 1024 => "note",
            _ => "ok",
        };
        assert_eq!(report.cache.severity, expected);
    }

    #[test]
    fn json_report_roundtrips_through_serde() {
        let report = collect_report();
        let json = serde_json::to_string_pretty(&report).unwrap();
        // Verify we get valid JSON (parse-back).
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
        // And the top-level keys are present.
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"device\""));
        assert!(json.contains("\"cuda_driver\""));
        assert!(json.contains("\"cache\""));
        // v0.30 phase 4 sections.
        assert!(json.contains("\"ffmpeg\""));
        assert!(json.contains("\"api_keys\""));
    }

    // v0.30 phase 4 — ffmpeg + API keys.

    #[test]
    fn ffmpeg_report_internal_consistency() {
        let r = collect_ffmpeg_report();
        // `present` must align with `version.is_some()` — the two
        // fields can't disagree.
        assert_eq!(r.present, r.version.is_some());
    }

    #[test]
    fn api_keys_report_serializes_without_leaking_values() {
        // Even when tokens are present in the test env, the JSON
        // serialization must only carry booleans — never the value.
        // Set a synthetic value, generate JSON, confirm the value
        // string doesn't appear anywhere in the output.
        // SAFETY: setting env vars in a test is racy across threads
        // running other env-dependent tests; we use a value unlikely
        // to collide with any real fixture string.
        const SECRET: &str = "do-not-leak-this-token-XKCD-2347";
        unsafe {
            std::env::set_var("HF_TOKEN", SECRET);
        }
        let report = collect_report();
        let json = serde_json::to_string(&report).unwrap();
        assert!(report.api_keys.hf_token_set);
        assert!(
            !json.contains(SECRET),
            "doctor JSON leaked the HF_TOKEN value"
        );
        unsafe {
            std::env::remove_var("HF_TOKEN");
        }
    }

    #[test]
    fn api_keys_report_empty_string_counts_as_absent() {
        // Shells that `export FOO=` (no value) leave the env var
        // present but empty. Doctor should treat that as "not set"
        // rather than "set" — otherwise users get spurious "token
        // present" reports.
        unsafe {
            std::env::set_var("CIVITAI_API_KEY", "");
        }
        let report = collect_api_keys_report();
        assert!(
            !report.civitai_token_set,
            "empty CIVITAI_API_KEY should report as absent"
        );
        unsafe {
            std::env::remove_var("CIVITAI_API_KEY");
        }
    }

    // -----------------------------------------------------------------
    // v0.33 phase 3 — reproducibility audit shape.
    // -----------------------------------------------------------------

    #[test]
    fn repro_audit_covers_every_major_pipeline() {
        let rows = audit_rows();
        // Each major surface should have at least one row pointing
        // at it. Use substring matching against `pipeline` to stay
        // robust against future copy edits.
        let must_cover = [
            "t2i",
            "AnimateDiff (SD 1.5)",
            "AnimateDiff (SDXL",
            "SD3",
            "Flux",
            "Stylize",
            "Portrait",
            "img2img",
            "wildcards",
            "enhancer",
        ];
        for needle in must_cover {
            let hit = rows.iter().any(|r| r.pipeline.contains(needle));
            assert!(hit, "audit table missing coverage for `{needle}`");
        }
    }

    #[test]
    fn repro_guarantee_symbols_non_empty() {
        // Every symbol is non-empty so the human table never
        // falls back to whitespace. The `label()` method was
        // removed in v0.34 phase 1 (was dead code).
        use ReproGuarantee::*;
        assert!(!Guaranteed.symbol().is_empty());
        assert!(!GuaranteedMetalU32.symbol().is_empty());
        assert!(!NeedsVerification.symbol().is_empty());
        assert!(!NonDeterministic.symbol().is_empty());
    }

    #[test]
    fn repro_report_serializes_with_expected_top_level_keys() {
        let r = ReproReport::collect();
        let json = serde_json::to_string_pretty(&r).unwrap();
        assert!(json.contains("\"plakat_version\""));
        assert!(json.contains("\"generated_at\""));
        assert!(json.contains("\"warnings\""));
        assert!(json.contains("\"rows\""));
        // Rows expose the documented field set.
        assert!(json.contains("\"pipeline\""));
        assert!(json.contains("\"code_path\""));
        assert!(json.contains("\"file_line\""));
        assert!(json.contains("\"guarantee\""));
        assert!(json.contains("\"note\""));
    }

    #[test]
    fn repro_report_warnings_cover_top_gaps() {
        let r = ReproReport::collect();
        let joined = r.warnings.join(" || ");
        // The --seed N requirement survives every fix — it's
        // architectural (rand::random fallback when omitted).
        assert!(joined.contains("--seed N"));
        // Enhancer + remote-LLM gaps also persist (out of plakat's
        // control by design).
        assert!(joined.contains("enhance"));
        assert!(joined.contains("Gemini") || joined.contains("DeepSeek"));
    }

    #[test]
    fn repro_audit_classifies_at_least_one_non_deterministic_path() {
        let rows = audit_rows();
        let any_non_det = rows
            .iter()
            .any(|r| r.guarantee == ReproGuarantee::NonDeterministic);
        assert!(
            any_non_det,
            "audit table must surface at least one NonDeterministic path \
             (rand::random() fallback when --seed is omitted, or remote LLMs)"
        );
    }

    #[test]
    fn repro_audit_v034_no_metal_u32_or_needs_verification() {
        // v0.34 phase 1 closed both gaps via `seeds::prepare_seed`
        // and the VAE-encode set_seed reorder. Asserting absence is
        // the regression lock: any future change that re-introduces
        // either tier will fail this test.
        let rows = audit_rows();
        for r in &rows {
            assert_ne!(
                r.guarantee,
                ReproGuarantee::GuaranteedMetalU32,
                "row {} should be Guaranteed after v0.34 phase 1: {}",
                r.pipeline,
                r.note
            );
            assert_ne!(
                r.guarantee,
                ReproGuarantee::NeedsVerification,
                "row {} should be Guaranteed after v0.34 phase 1: {}",
                r.pipeline,
                r.note
            );
        }
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
// ============================================================
// v0.19: --json output. Structured equivalent of the v0.18
// section blocks (device match, libcuda driver, HF cache disk).
// ============================================================

#[derive(serde::Serialize)]
struct DoctorReport {
    /// plakat version that produced this report. Lets CI pin
    /// expectations against specific releases.
    version: String,
    device: DeviceReport,
    cuda_driver: CudaDriverReport,
    cache: CacheReport,
    /// v0.30 phase 4: ffmpeg presence + version. Required for
    /// MP4/WebM output and the v0.30 `--control-spec video=` input
    /// decode.
    ffmpeg: FfmpegReport,
    /// v0.30 phase 4: presence of optional API tokens. Reports
    /// boolean presence ONLY — never the value.
    api_keys: ApiKeysReport,
}

#[derive(serde::Serialize)]
struct DeviceReport {
    /// `["cuda", "metal"]` or empty (`[]`) for the CPU-only build.
    build_features: Vec<String>,
    /// `"CPU"` / `"CUDA"` / `"Metal"` — whatever
    /// `device::select("auto")` resolved on this host.
    runtime: String,
    /// `true` when the build features explain the runtime device,
    /// `false` when a built-for-accelerator binary fell back to CPU.
    aligned: bool,
}

#[derive(serde::Serialize)]
struct CudaDriverReport {
    /// `true` iff we actually probed (Linux + `--features cuda`).
    /// On non-Linux or non-CUDA builds the section doesn't apply.
    applicable: bool,
    /// Resolved `libcuda.so.1` path on success, `null` otherwise.
    found_at: Option<String>,
    /// Paths searched (always populated when `applicable: true`).
    candidates: Vec<String>,
}

#[derive(serde::Serialize)]
struct CacheReport {
    root: String,
    /// `false` when the cache root doesn't exist yet (no models
    /// downloaded). `used_bytes` is `0` in that case.
    exists: bool,
    used_bytes: u64,
    used_human: String,
    /// `"ok" | "note" | "warn"` matching the human section's
    /// threshold output. Stable enum string for CI assertions.
    severity: &'static str,
}

#[derive(serde::Serialize)]
struct FfmpegReport {
    /// `true` when `ffmpeg -version` resolved successfully.
    present: bool,
    /// First line of `ffmpeg -version` output (e.g. `"ffmpeg version
    /// 6.1.1 ..."`), or `null` when not present.
    version: Option<String>,
}

#[derive(serde::Serialize)]
struct ApiKeysReport {
    /// `true` iff `HF_TOKEN` env var is set + non-empty.
    hf_token_set: bool,
    /// `true` iff legacy `HUGGING_FACE_HUB_TOKEN` env var is set + non-empty.
    hf_legacy_token_set: bool,
    /// `true` iff `CIVITAI_API_KEY` env var is set + non-empty.
    civitai_token_set: bool,
}

fn collect_report() -> DoctorReport {
    DoctorReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        device: collect_device_report(),
        cuda_driver: collect_cuda_driver_report(),
        cache: collect_cache_report(),
        ffmpeg: collect_ffmpeg_report(),
        api_keys: collect_api_keys_report(),
    }
}

fn collect_ffmpeg_report() -> FfmpegReport {
    match crate::imaging::video::ffmpeg_version() {
        Ok(v) => FfmpegReport {
            present: true,
            version: Some(v),
        },
        Err(_) => FfmpegReport {
            present: false,
            version: None,
        },
    }
}

fn collect_api_keys_report() -> ApiKeysReport {
    let hf = std::env::var("HF_TOKEN").map(|v| !v.is_empty()).unwrap_or(false);
    let hf_legacy = std::env::var("HUGGING_FACE_HUB_TOKEN")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let civitai = std::env::var("CIVITAI_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    ApiKeysReport {
        hf_token_set: hf,
        hf_legacy_token_set: hf_legacy,
        civitai_token_set: civitai,
    }
}

fn collect_device_report() -> DeviceReport {
    let build_features: Vec<String> = [
        ("cuda", cfg!(feature = "cuda")),
        ("metal", cfg!(feature = "metal")),
    ]
    .into_iter()
    .filter_map(|(n, on)| if on { Some(n.to_string()) } else { None })
    .collect();
    let (runtime, aligned) = match crate::device::select("auto") {
        Ok(d) => {
            let label = describe_device(&d);
            let runtime_is_cpu = matches!(d, candle_core::Device::Cpu);
            let built_for_accel = !build_features.is_empty();
            // If the binary was built CPU-only, alignment trivially holds.
            // If it was built for an accelerator and we got CPU, that's a
            // silent fallback — surface as misaligned.
            let aligned = !(built_for_accel && runtime_is_cpu);
            (label, aligned)
        }
        // device::select("auto") shouldn't fail (auto always returns
        // CPU as a worst-case), but if it does, flag as unaligned.
        Err(_) => ("unresolved".to_string(), false),
    };
    DeviceReport {
        build_features,
        runtime,
        aligned,
    }
}

fn collect_cuda_driver_report() -> CudaDriverReport {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    {
        const CANDIDATES: &[&str] = &[
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib/libcuda.so.1",
            "/lib/x86_64-linux-gnu/libcuda.so.1",
        ];
        let found = CANDIDATES
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .map(|s| s.to_string());
        return CudaDriverReport {
            applicable: true,
            found_at: found,
            candidates: CANDIDATES.iter().map(|s| s.to_string()).collect(),
        };
    }
    #[cfg(not(all(feature = "cuda", target_os = "linux")))]
    {
        CudaDriverReport {
            applicable: false,
            found_at: None,
            candidates: Vec::new(),
        }
    }
}

fn collect_cache_report() -> CacheReport {
    let root = crate::hf::cache::hf_cache_root();
    if !root.exists() {
        return CacheReport {
            root: root.display().to_string(),
            exists: false,
            used_bytes: 0,
            used_human: "0 B".to_string(),
            severity: "ok",
        };
    }
    let used = crate::hf::cache::dir_size(&root).unwrap_or(0);
    let severity = match cache_usage_severity(used) {
        CacheSeverity::Ok => "ok",
        CacheSeverity::Note => "note",
        CacheSeverity::Warn => "warn",
    };
    CacheReport {
        root: root.display().to_string(),
        exists: true,
        used_bytes: used,
        used_human: crate::hf::cache::human_bytes(used),
        severity,
    }
}

fn run_json() -> Result<()> {
    let report = collect_report();
    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| anyhow::anyhow!("serializing doctor report: {e}"))?;
    println!("{json}");
    Ok(())
}

// =====================================================================
// v0.46: model capability report (--capability).
// =====================================================================

async fn run_capability(args: &DoctorArgs) -> Result<()> {
    let device = crate::device::select(&args.device)
        .map_err(|e| anyhow::anyhow!("selecting device {:?}: {e}", args.device))?;
    let report = crate::capability::build(crate::hw::probe(&device), args.verify).await;
    if args.json {
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| anyhow::anyhow!("serializing capability report: {e}"))?;
        println!("{json}");
        return Ok(());
    }
    render_capability(&report);
    Ok(())
}

fn render_capability(r: &crate::capability::CapabilityReport) {
    use console::style;
    let hw = &r.hardware;

    println!("\n  {}\n", style("text-to-image hardware").bold());
    println!(
        "    backend   {} · features {}",
        style(&hw.backend).cyan(),
        if hw.features.is_empty() { "—".to_string() } else { hw.features.join("+") }
    );
    println!("    memory    {}", hw.budget_label());
    println!(
        "    pressure  {} · ~{:.1} GB free now (the OOM guard's signal)",
        match crate::hw::mem_pressure() {
            crate::hw::Pressure::Normal => "normal",
            crate::hw::Pressure::Warning => "⚠ warning",
            crate::hw::Pressure::Critical => "⛔ critical",
            crate::hw::Pressure::Unknown => "n/a",
        },
        crate::hw::available_ram_gb(),
    );
    println!("    cpu       {} cores · {} · {}", hw.cpu_cores, hw.arch, hw.os);

    println!("\n  {}\n", style("model capability").bold());
    for m in &r.models {
        let sym = match m.verdict.as_str() {
            "runs" => style("✓").green(),
            "tight" => style("!").yellow(),
            "wont-fit" => style("✗").red(),
            "blocked" => style("⊘").red(),
            "gated" => style("·").yellow(),
            _ => style("·").dim(),
        };
        let verdict = match m.verdict.as_str() {
            "wont-fit" => "won't fit",
            other => other,
        };
        let size = match (m.weight_gb, m.size_source.as_str()) {
            (Some(w), "cache") => format!("{w:.0} GB"),
            (Some(w), _) => format!("≤{w:.0} GB"),
            _ => "—".to_string(),
        };
        let note = m
            .tuning
            .clone()
            .unwrap_or_else(|| format!("{}²", m.native_res));
        println!(
            "    {} {:<13} {:>7}  {:<9}  {}",
            sym,
            m.model,
            size,
            verdict,
            style(note).dim()
        );
    }

    println!(
        "\n  {}",
        style("sizes: exact from disk cache · ≤ HF upper-bound estimate · — unknown").dim()
    );
    let any_unknown = r.models.iter().any(|m| m.size_source == "unknown");
    if any_unknown {
        println!(
            "  {}",
            style("add --verify to fetch sizes for models you haven't downloaded").dim()
        );
    }

    // Few-step `--fast` presets (bundled LoRA + scheduler + steps/guidance), grouped by the
    // base family they accelerate. Derived from the canonical `flux_fast::PRESETS` table so
    // new presets appear here automatically.
    use crate::pipelines::flux_fast::{FastTarget, PRESETS};
    println!("\n  {}\n", style("few-step presets (--fast)").bold());
    for (label, target) in [("Flux", FastTarget::Flux), ("SDXL", FastTarget::Sdxl), ("SD 1.5", FastTarget::Sd15)] {
        let names: Vec<&str> = PRESETS
            .iter()
            .filter(|p| std::mem::discriminant(&p.target) == std::mem::discriminant(&target))
            .map(|p| p.name)
            .collect();
        if !names.is_empty() {
            println!("    {:<8} {}", label, style(names.join(" · ")).cyan());
        }
    }
    println!(
        "  {}",
        style("e.g. plakat generate \"…\" --model sdxl --fast lightning-sdxl-8").dim()
    );

    println!();
}

// =====================================================================
// v0.33 phase 3: reproducibility audit.
// =====================================================================

/// Determinism guarantee tier for one RNG-touching code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ReproGuarantee {
    /// Fully deterministic when `--seed N` is set. No backend-
    /// specific gotchas.
    Guaranteed,
    /// Deterministic when `--seed N` is set, BUT the seed gets
    /// truncated to `u32` on Metal — so user seeds > 2^32 lose
    /// entropy silently. Documented in v0.33 phase 3 audit.
    GuaranteedMetalU32,
    /// Plumbing looks correct on the surface but hasn't been
    /// exercised under a hostile test — could harbour a silent
    /// gap. Listed for v0.34 verification.
    NeedsVerification,
    /// Never deterministic. Either consumes thread_rng / system
    /// entropy, or runs without an enclosing seed call.
    NonDeterministic,
}

impl ReproGuarantee {
    fn symbol(self) -> &'static str {
        match self {
            Self::Guaranteed => "✓",
            Self::GuaranteedMetalU32 => "⚠",
            Self::NeedsVerification => "?",
            Self::NonDeterministic => "✗",
        }
    }
}

/// One row of the reproducibility audit table — a single
/// RNG-touching code path with its determinism classification.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReproRow {
    pub pipeline: &'static str,
    pub code_path: &'static str,
    pub file_line: &'static str,
    pub guarantee: ReproGuarantee,
    pub note: &'static str,
}

/// The static survey table. Derived from the v0.33 phase 3 audit
/// (recorded in `Documentation/RFC_v0.33_PRODUCTION_POLISH.md`).
/// v0.34 phase 1: Metal-u32 rows flipped to Guaranteed via
/// `pipelines::seeds::prepare_seed`; img2img + stylize VAE-encode
/// placement bugs fixed (set_seed now runs before the encode).
/// Each row is hand-curated; future cycles update it when seed
/// plumbing changes.
pub fn audit_rows() -> Vec<ReproRow> {
    vec![
        ReproRow {
            pipeline: "t2i (SD-family)",
            code_path: "Pipeline::run pre-loop latent randn",
            file_line: "pipelines/t2i.rs:~1225",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: `seeds::prepare_seed` mixes full u64 entropy into the Metal backend's u32 RNG init via SplitMix64. Seeds <2^32 unchanged; seeds >=2^32 no longer collide.",
        },
        ReproRow {
            pipeline: "AnimateDiff (SD 1.5)",
            code_path: "denoise_window noise init + FreeNoise pre-gen",
            file_line: "pipelines/animatediff.rs:~340",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: `seeds::prepare_seed` applied at per-window + FreeNoise sites. Per-window seed for v0.27 path, full-length pre-gen for v0.32 FreeNoise.",
        },
        ReproRow {
            pipeline: "AnimateDiff (SDXL beta)",
            code_path: "denoise_window noise init",
            file_line: "pipelines/animatediff.rs:~1075",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: same fix as SD 1.5; FreeNoise path identical.",
        },
        ReproRow {
            pipeline: "SD3 / SD3.5",
            code_path: "Pipeline::run init latents",
            file_line: "pipelines/sd3.rs:~813",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: `seeds::prepare_seed` before randn. Falls back to rand::random() when --seed absent.",
        },
        ReproRow {
            pipeline: "Flux (BF16 / GGUF / NF4)",
            code_path: "sampling::get_noise",
            file_line: "pipelines/flux.rs:~1565",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: `seeds::prepare_seed` before candle's `flux::sampling::get_noise`.",
        },
        ReproRow {
            pipeline: "PixArt Sigma (DiT-XL/2)",
            code_path: "Pipeline::generate noise init",
            file_line: "pipelines/pixart.rs:~200",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.35 phase 2: `seeds::prepare_seed` runs before `device.set_seed()` + `Tensor::randn`. CFG denoise loop is fully deterministic with --seed N.",
        },
        ReproRow {
            pipeline: "Stable Cascade (3-stage)",
            code_path: "Pipeline::generate Stage C + Stage B noise init",
            file_line: "pipelines/cascade.rs:~290",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.37 phase 4: `seeds::prepare_seed` runs once before BOTH Stage C and Stage B `Tensor::randn` calls (Stage A is one-shot decode). Same --seed N produces byte-identical output. Numerical correctness on real weights is v0.38 follow-through (FiLM time injection + effnet conditioning); reproducibility holds either way.",
        },
        ReproRow {
            pipeline: "Stylize (SD 1.5)",
            code_path: "init noise + VAE encode",
            file_line: "pipelines/stylize.rs:~250",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: set_seed moved BEFORE VAE encode (init_dist.sample is RNG-touching). `seeds::prepare_seed` replaces the old u32 mask.",
        },
        ReproRow {
            pipeline: "Portrait (FaceID / Plus-Face)",
            code_path: "Pipeline::generate t2i + inpaint blend",
            file_line: "pipelines/portrait.rs:~420 + ~751",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: all 4 set_seed sites now route through `seeds::prepare_seed`.",
        },
        ReproRow {
            pipeline: "img2img / inpaint",
            code_path: "init latents from VAE-encoded init image",
            file_line: "pipelines/img2img.rs:~185",
            guarantee: ReproGuarantee::Guaranteed,
            note: "v0.34 phase 1: per-iter set_seed inserted BEFORE vae_encode_image_file (the inner dist.sample is RNG-touching).",
        },
        ReproRow {
            pipeline: "LCM scheduler",
            code_path: "per-step re-noise",
            file_line: "pipelines/lcm_scheduler.rs:~222",
            guarantee: ReproGuarantee::Guaranteed,
            note: "Reuses parent pipeline's set_seed RNG stream — deterministic when parent seeded (which v0.34 phase 1 guarantees on Metal).",
        },
        ReproRow {
            pipeline: "Prompt wildcards",
            code_path: "expand() RNG from --seed",
            file_line: "src/prompt/wildcards.rs",
            guarantee: ReproGuarantee::Guaranteed,
            note: "StdRng seeded from --seed at CLI layer (cli/generate.rs:~1555).",
        },
        ReproRow {
            pipeline: "Prompt enhancer (local LLM)",
            code_path: "LogitsProcessor sampling",
            file_line: "src/llm/enhancer.rs:~189",
            guarantee: ReproGuarantee::Guaranteed,
            note: "Greedy (--enhance-temp 0.0, default) → deterministic. --enhance-temp > 0.0 sampling honours --seed but is inherently stochastic.",
        },
        ReproRow {
            pipeline: "Prompt enhancer (DeepSeek / Gemini)",
            code_path: "HTTP request",
            file_line: "src/llm/deepseek.rs + gemini.rs",
            guarantee: ReproGuarantee::NonDeterministic,
            note: "Remote LLM call — server-side sampling is outside plakat's control.",
        },
        ReproRow {
            pipeline: "Any pipeline (no --seed flag)",
            code_path: "Fallback rand::random()",
            file_line: "(multiple)",
            guarantee: ReproGuarantee::NonDeterministic,
            note: "Pipelines fall back to `rand::random()` for seed when --seed is omitted. Reproducibility REQUIRES --seed N.",
        },
    ]
}

/// v0.33 phase 3: full reproducibility report as one struct. JSON
/// serialisation matches the `--json` output.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReproReport {
    pub plakat_version: String,
    pub generated_at: String,
    /// Top-level warnings that summarise the most-common gaps.
    /// Shown above the table in human output so users hit them
    /// without scanning every row.
    pub warnings: Vec<&'static str>,
    pub rows: Vec<ReproRow>,
}

impl ReproReport {
    pub fn collect() -> Self {
        Self {
            plakat_version: env!("CARGO_PKG_VERSION").to_string(),
            generated_at: format!("{:?}", std::time::SystemTime::now()),
            warnings: vec![
                "Reproducibility REQUIRES `--seed N`. Without it, pipelines fall back to `rand::random()` and produce different output each run.",
                "Prompt enhancer with `--enhance-temp > 0.0` honours `--seed N` but local LLM sampling is inherently stochastic. Use `--enhance-temp 0.0` (default) for deterministic enhancement.",
                "DeepSeek / Gemini providers are remote — server-side sampling is outside plakat's control. No reproducibility guarantee for those.",
            ],
            rows: audit_rows(),
        }
    }
}

fn run_reproducibility_check(as_json: bool) -> Result<()> {
    let report = ReproReport::collect();
    if as_json {
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| anyhow::anyhow!("serializing reproducibility report: {e}"))?;
        println!("{json}");
        return Ok(());
    }

    println!(
        "\n{}  reproducibility audit (v0.33 phase 3 baseline + v0.34 phase 1 fixes)\n",
        style("doctor --reproducibility-check").yellow().bold()
    );

    section_header("Top warnings");
    for w in &report.warnings {
        println!("    {} {}", style("!").yellow(), w);
    }
    println!();

    section_header("Per-pipeline determinism table");
    println!(
        "    {:^7} {:<26} {:<40} {}",
        style("status").dim(),
        style("pipeline").dim(),
        style("code path").dim(),
        style("note").dim(),
    );
    for row in &report.rows {
        let sym = match row.guarantee {
            ReproGuarantee::Guaranteed => style(row.guarantee.symbol()).green(),
            ReproGuarantee::GuaranteedMetalU32 => style(row.guarantee.symbol()).yellow(),
            ReproGuarantee::NeedsVerification => style(row.guarantee.symbol()).cyan(),
            ReproGuarantee::NonDeterministic => style(row.guarantee.symbol()).red(),
        };
        println!(
            "    {sym:^7} {:<26} {:<40} {}",
            row.pipeline,
            row.code_path,
            style(row.note).dim(),
        );
    }
    println!();
    println!(
        "    {} legend: {}=GUARANTEED  {}=Metal-u32-truncation  {}=NEEDS-VERIFICATION  {}=NON-DETERMINISTIC",
        style("·").dim(),
        style("✓").green(),
        style("⚠").yellow(),
        style("?").cyan(),
        style("✗").red(),
    );
    println!();
    Ok(())
}

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
