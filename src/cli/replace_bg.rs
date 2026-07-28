//! `plakat replace-bg` — swap an image's background while keeping the subject.
//!
//! 1. Matte the subject off its background with U2Net (`matting::matte` → foreground + alpha).
//! 2. Get a new background: `--bg-image PATH` (composited as-is) or generate one from `--prompt`
//!    via txt2img at the subject's dimensions.
//! 3. Alpha-composite the subject over the new background (the alpha is feathered for a soft edge).
//!    The subject pixels are preserved exactly; only the background changes.

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use image::{imageops::FilterType, GrayImage, RgbImage};
use std::path::PathBuf;

use crate::pipelines::scheduler::SchedulerKind;

#[derive(ClapArgs, Debug)]
pub struct ReplaceBgArgs {
    /// Path to the source image (subject on some background). Any format the `image` crate reads.
    pub input: PathBuf,

    /// Prompt describing the NEW background to generate. Ignored when `--bg-image` is given.
    #[arg(help_heading = "Prompt & text", long, default_value = "")]
    pub prompt: String,

    /// Negative prompt for the generated background.
    #[arg(help_heading = "Prompt & text", long, default_value = "")]
    pub negative: String,

    /// Composite the subject over this image instead of generating a background. Resized to the
    /// subject's dimensions. Mutually exclusive with generating from `--prompt`.
    #[arg(help_heading = "Background", long = "bg-image", value_name = "PATH")]
    pub bg_image: Option<PathBuf>,

    /// Feather radius (px) on the subject's matte edge — softens the composite seam.
    #[arg(help_heading = "Background", long = "edge-feather", default_value_t = 2, value_name = "PX")]
    pub edge_feather: u32,

    /// Model for background generation. Defaults to `sdxl`.
    #[arg(help_heading = "Model & sampler", long, default_value = "sdxl")]
    pub model: String,

    /// Denoising steps for the generated background.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 28)]
    pub steps: usize,

    /// Base seed for the generated background.
    #[arg(help_heading = "Model & sampler", long)]
    pub seed: Option<u64>,

    /// Scheduler for background generation.
    #[arg(help_heading = "Model & sampler", long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// Output directory.
    #[arg(help_heading = "Size & output", long, default_value = "./out")]
    pub out: PathBuf,

    /// `--import <album>` / `--import-move`: land the result in a photo album.
    #[command(flatten)]
    pub import: crate::cli::import::ImportArgs,
}

pub async fn run(args: ReplaceBgArgs, device: Device) -> Result<()> {
    // 1. Matte the subject.
    let (fg, mut alpha) = crate::pipelines::matting::matte(&args.input, &device)
        .await
        .with_context(|| format!("matting subject from {}", args.input.display()))?;
    let (w, h) = (fg.width(), fg.height());
    if args.edge_feather > 0 {
        alpha = feather(&alpha, args.edge_feather);
    }

    // 2. Get the new background at the subject's dimensions.
    let bg: RgbImage = match args.bg_image.as_ref() {
        Some(p) => {
            let img = image::open(p)
                .with_context(|| format!("opening --bg-image {}", p.display()))?
                .to_rgb8();
            image::imageops::resize(&img, w, h, FilterType::Lanczos3)
        }
        None => {
            if args.prompt.trim().is_empty() {
                anyhow::bail!("replace-bg needs a new background: pass --prompt \"…\" or --bg-image PATH");
            }
            generate_background(&args, w, h, device).await?
        }
    };

    // 3. Alpha-composite the subject over the background.
    let mut out = bg;
    for y in 0..h {
        for x in 0..w {
            let a = alpha.get_pixel(x, y).0[0] as f32 / 255.0;
            let f = fg.get_pixel(x, y).0;
            let b = out.get_pixel(x, y).0;
            let blend = |fi: u8, bi: u8| -> u8 {
                (fi as f32 * a + bi as f32 * (1.0 - a)).round().clamp(0.0, 255.0) as u8
            };
            out.put_pixel(x, y, image::Rgb([blend(f[0], b[0]), blend(f[1], b[1]), blend(f[2], b[2])]));
        }
    }

    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating output dir {}", args.out.display()))?;
    let seed = args.seed.unwrap_or(0);
    let out_path = args.out.join(format!("plakat-replacebg-{seed}.png"));
    out.save(&out_path)
        .with_context(|| format!("saving {}", out_path.display()))?;
    crate::ui::progress::println(&format!("✓ replaced background ({w}×{h})  →  {}", out_path.display()));
    Ok(())
}

/// Generate a background via txt2img at (roughly) the subject's dims, then resize to exactly (w,h).
/// Snaps the generation size to a multiple of 8 (SD constraint); the exact fit comes from the resize.
async fn generate_background(args: &ReplaceBgArgs, w: u32, h: u32, device: Device) -> Result<RgbImage> {
    let snap = |n: u32| -> u32 { (n / 8).max(1) * 8 };
    let (gw, gh) = (snap(w), snap(h));
    let tmp = tempfile::Builder::new().prefix("plakat-replacebg-").tempdir()?;
    let mut req = crate::pipelines::t2i::Request::simple(
        args.prompt.clone(),
        args.model.clone(),
        gw,
        gh,
        args.steps,
        args.seed,
        device,
        tmp.path().to_path_buf(),
    );
    req.negative = args.negative.clone();
    req.scheduler = args.scheduler;
    crate::ui::progress::println(&format!("Generating background {gw}×{gh} with {} …", args.model));
    crate::pipelines::t2i::run(req).await.context("generating replace-bg background")?;

    // Read back the produced PNG (count = 1).
    let png = std::fs::read_dir(tmp.path())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .context("background generation produced no PNG")?;
    let img = image::open(&png)?.to_rgb8();
    Ok(image::imageops::resize(&img, w, h, FilterType::Lanczos3))
}

/// Feather a matte edge with a Gaussian blur of `radius` (soft alpha ramp at the seam).
fn feather(alpha: &GrayImage, radius: u32) -> GrayImage {
    image::imageops::blur(alpha, radius as f32)
}
