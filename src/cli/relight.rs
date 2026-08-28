//! `plakat relight` — IC-Light relighting of a foreground subject.
//!
//! Given a subject image and a text prompt describing the desired
//! lighting, IC-Light ("Imposing Consistent Light", lllyasviel)
//! re-illuminates the subject. The subject is matted off its
//! background, composited onto neutral grey, and used as a foreground
//! condition for an SD 1.5 UNet whose input conv has been widened to 8
//! channels with the IC-Light offset weights merged in.

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

use crate::pipelines::ic_light;

#[derive(ClapArgs, Debug)]
pub struct RelightArgs {
    /// Subject image to relight. Its background is matted away
    /// automatically (U2Net) before conditioning. Not required with `--list-lights`.
    #[arg(help_heading = "Relight", value_name = "SUBJECT", required_unless_present = "list_lights")]
    pub subject: Option<PathBuf>,

    /// Prompt describing the desired lighting / scene (e.g. "warm sunset light from the left, golden
    /// hour"). Optional when `--light <preset>` is given (the user prompt is appended to the preset).
    #[arg(help_heading = "Prompt & text", long, required_unless_present_any = ["light", "list_lights"])]
    pub prompt: Option<String>,

    /// Negative prompt (added to any preset negative).
    #[arg(help_heading = "Prompt & text", long, default_value = "")]
    pub negative: String,

    /// A named lighting preset — `key-left` / `key-right` / `top` / `rim` / `softbox` / `golden-hour` /
    /// `sunset` / `moonlight` / `candlelight` / `neon` / `overcast` (see `--list-lights`). RELIGHT-1.
    #[arg(help_heading = "Relight", long)]
    pub light: Option<String>,

    /// List the lighting presets and exit.
    #[arg(help_heading = "Relight", long = "list-lights", default_value_t = false)]
    pub list_lights: bool,

    /// Override the light direction with a custom gradient angle (degrees; 0 = left, 90 = top, 180 =
    /// right, 270 = bottom). Overrides the preset's backdrop.
    #[arg(help_heading = "Relight", long = "light-angle", value_name = "DEG")]
    pub light_angle: Option<f32>,

    /// Output size: `N` (square) or `WxH` (e.g. 512x768).
    #[arg(help_heading = "Size & output", long, default_value = "512")]
    pub size: String,

    /// Denoise steps.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 25)]
    pub steps: usize,

    /// Classifier-free guidance scale. IC-Light works best at low CFG.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 2.0)]
    pub guidance: f64,

    /// Seed (omit for a random seed).
    #[arg(help_heading = "Model & sampler", long)]
    pub seed: Option<u64>,

    /// Output directory (or file path ending in an image extension).
    #[arg(help_heading = "Size & output", long, default_value = "./")]
    pub out: PathBuf,

    /// `--import <album>` / `--import-move`: land the relit image in a photo album.
    #[command(flatten)]
    pub import: crate::cli::import::ImportArgs,
}

pub async fn run(args: RelightArgs, device: Device) -> Result<()> {
    // --list-lights: print the preset table and exit (no model load).
    if args.list_lights {
        println!("\n  {} — `relight --light <name>`", console::style("lighting presets").bold());
        for p in ic_light::light_presets() {
            println!("  {:<12} {}", console::style(p.name).green(), console::style(p.prompt).dim());
        }
        println!();
        return Ok(());
    }

    let subject = args.subject.as_ref().context("a SUBJECT image is required")?;
    let (width, height) = parse_size(&args.size)?;
    let seed = args.seed.unwrap_or_else(rand::random);

    // Resolve the lighting: a `--light` preset (prompt + negative + backdrop), with the user `--prompt`
    // appended, and `--light-angle` overriding the direction. RELIGHT-1 P1/P2.
    let preset = match &args.light {
        Some(name) => Some(*ic_light::light_preset(name).with_context(|| {
            format!("unknown --light preset `{name}` — see `relight --list-lights`")
        })?),
        None => None,
    };
    let user_prompt = args.prompt.clone().unwrap_or_default();
    let prompt = match &preset {
        Some(p) if user_prompt.trim().is_empty() => p.prompt.to_string(),
        Some(p) => format!("{}, {}", p.prompt, user_prompt.trim()),
        None => user_prompt,
    };
    let negative = match &preset {
        Some(p) if args.negative.trim().is_empty() => p.negative.to_string(),
        Some(p) => format!("{}, {}", p.negative, args.negative.trim()),
        None => args.negative.clone(),
    };
    let backdrop = match args.light_angle {
        Some(deg) => ic_light::Backdrop::Directional(deg),
        None => preset.map(|p| p.backdrop).unwrap_or(ic_light::Backdrop::Flat),
    };

    let pipeline = ic_light::Pipeline::load(device).await?;
    let (buf, w, h) = pipeline.relight(
        subject,
        &prompt,
        &negative,
        width,
        height,
        args.steps,
        args.guidance,
        seed,
        backdrop,
    )?;

    // Resolve the output path: a directory gets a generated filename;
    // an explicit image path is used as-is.
    let out_path = if args
        .out
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp"))
        .unwrap_or(false)
    {
        args.out.clone()
    } else {
        std::fs::create_dir_all(&args.out)
            .with_context(|| format!("creating output dir {}", args.out.display()))?;
        args.out.join(format!("plakat-relight-{seed}.png"))
    };
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    crate::imaging::io::save_rgb_u8(&buf, w, h, &out_path)?;
    println!(
        "{}  relit {} ({}×{}, seed {})",
        style("✓").green(),
        subject.display(),
        w,
        h,
        seed,
    );
    println!("→ {}", out_path.display());
    Ok(())
}

/// Parse `--size`: either `N` (square `N×N`) or `WxH`.
fn parse_size(s: &str) -> Result<(u32, u32)> {
    let s = s.trim();
    if let Some((w, h)) = s.split_once(['x', 'X']) {
        let w: u32 = w
            .trim()
            .parse()
            .with_context(|| format!("--size width in {s:?}"))?;
        let h: u32 = h
            .trim()
            .parse()
            .with_context(|| format!("--size height in {s:?}"))?;
        Ok((w, h))
    } else {
        let n: u32 = s.parse().with_context(|| format!("--size {s:?} (expected N or WxH)"))?;
        Ok((n, n))
    }
}
