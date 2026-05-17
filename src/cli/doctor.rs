//! `plakat doctor` — Phase 4 health check.
//!
//! Inspects the user's environment and prints a verdict for each
//! configurable surface that Phase 4 exposes:
//!
//! * **ArcFace** weights — local file (`PLAKAT_ARCFACE_WEIGHTS`) or HF
//!   spec (`PLAKAT_ARCFACE_HF`). Verifies the local file exists; for
//!   HF specs just parses the `repo#file` format (no network call —
//!   `doctor` stays offline by design).
//! * **SCRFD** weights — `PLAKAT_SCRFD_WEIGHTS` local file. SCRFD HF
//!   download isn't shipped yet; that lands when the FaceIdEncoder
//!   constructor goes async.
//! * **FaceID UNet LoRA** — checks the `PLAKAT_FACEID_LORA` opt-out
//!   env var and reports whether it's active.
//! * **Device + cache** — current device selection, HF cache root.
//!
//! All checks are read-only and **don't** download or load any model.
//! Run this before kicking off a long generation to verify setup.

use anyhow::Result;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

#[derive(ClapArgs, Debug)]
pub struct DoctorArgs {}

pub async fn run(_args: DoctorArgs) -> Result<()> {
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
                    ok(&format!(
                        "PLAKAT_ARCFACE_HF = {repo}#{file} (parsed OK; download deferred to first use)"
                    ));
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
        }
    }

    println!();

    // -------- SCRFD --------
    section_header("SCRFD face detector (auto-fills landmarks for FaceID)");
    let scrfd = std::env::var("PLAKAT_SCRFD_WEIGHTS").ok();
    match scrfd.as_deref() {
        Some(path) => {
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
        }
        None => {
            note(
                "PLAKAT_SCRFD_WEIGHTS not set — FaceID falls back to \
                 --face-bbox / --face-landmarks / centre-crop. \
                 SCRFD HF auto-download isn't shipped yet.",
            );
        }
    }

    println!();

    // -------- FaceID image_proj — auto-downloaded from h94 --------
    section_header("FaceID image-proj (auto-downloaded from h94/IP-Adapter)");
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
