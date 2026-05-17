//! `plakat doctor` — Phase 4 health check.
//!
//! Inspects the user's environment and prints a verdict for each
//! configurable surface that Phase 4 exposes:
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
}

pub async fn run(args: DoctorArgs) -> Result<()> {
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
                    "Weight-loading verification is the user-iteration step \
                     in Phase 4c.4 — the architecture is a best-guess from \
                     the InsightFace reference. If load errors at a layer, \
                     paste the error.",
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
                        "Weight-loading verification is the user-iteration step \
                         in Phase 4c.4 — the architecture is a best-guess from \
                         the InsightFace reference. If load errors at a layer, \
                         paste the error.",
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
    println!(
        "  {}\n",
        style("If you've fixed any of the issues above, re-run `plakat doctor` to confirm.").dim()
    );

    Ok(())
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
