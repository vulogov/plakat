use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;

use crate::pipelines::adetailer;

/// Restore degraded faces in existing images: SCRFD-detect each face, diffusion-refine the crop at a
/// gentle strength (identity-preserving), and feather-composite it back. The standalone form of
/// `generate --adetailer` — run it on photos, upscales (pairs with `upscale --diffusion`), or any
/// image with low-fidelity faces. SD 1.5 / SDXL. Needs SCRFD weights (`PLAKAT_SCRFD_WEIGHTS` /
/// `PLAKAT_SCRFD_HF`, the same detector `--portrait` uses).
#[derive(ClapArgs, Debug)]
pub struct RestoreFacesArgs {
    /// Images and/or directories to restore (edited in place). Directories are scanned
    /// non-recursively for `.png` / `.jpg` / `.jpeg` / `.webp`.
    #[arg(value_name = "PATH", required = true)]
    pub inputs: Vec<PathBuf>,

    /// SD model for the face refinement pass.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// img2img strength on each face crop. 0.3–0.5 crisps detail while preserving identity/colour;
    /// higher can change the face.
    #[arg(long, default_value_t = 0.4)]
    pub strength: f32,

    /// Bbox expansion (0.25 = +25% each side) — more context = smoother blend, less face resolution.
    #[arg(long, default_value_t = 0.25)]
    pub padding: f32,

    /// Feather fraction for the composite seam (outer 0.25 fades 1→0).
    #[arg(long, default_value_t = 0.25)]
    pub feather: f32,

    /// SCRFD confidence threshold — faces below this are skipped.
    #[arg(long, default_value_t = 0.5)]
    pub confidence: f32,

    /// Working resolution of the face pass (square; 512 = SD 1.5 native, 1024 = SDXL).
    #[arg(long = "working-size", default_value_t = 512)]
    pub working_size: u32,
}

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp"];

fn collect_images(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            for entry in std::fs::read_dir(p).with_context(|| format!("reading dir {}", p.display()))? {
                let path = entry?.path();
                let is_img = path.is_file()
                    && path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
                        .unwrap_or(false);
                if is_img {
                    out.push(path);
                }
            }
        } else if p.is_file() {
            out.push(p.clone());
        } else {
            anyhow::bail!("no such file or directory: {}", p.display());
        }
    }
    out.sort();
    Ok(out)
}

pub async fn run(args: RestoreFacesArgs, device: Device) -> Result<()> {
    let files = collect_images(&args.inputs)?;
    anyhow::ensure!(!files.is_empty(), "no images found in the given paths");

    let cfg = adetailer::Config {
        model: args.model.clone(),
        strength: args.strength,
        padding: args.padding,
        feather: args.feather,
        confidence: args.confidence,
        working_size: args.working_size,
        device,
        ..adetailer::Config::defaults()
    };

    let n = adetailer::refine_files(&cfg, &files, None)
        .await
        .context("restoring faces")?;

    println!(
        "{}  restored {} face(s) across {} image(s)",
        style("✓").green(),
        n,
        files.len()
    );
    Ok(())
}
