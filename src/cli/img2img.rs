use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
use crate::pipelines::img2img;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;

/// `plakat img2img <INPUT> --prompt "..."` — re-imagine an existing
/// image at a chosen denoise strength. Supply `--mask` to restrict
/// the changes to a region (inpaint).
#[derive(ClapArgs, Debug)]
pub struct Img2ImgArgs {
    /// Path to the source image. Any format the `image` crate reads.
    pub input: PathBuf,

    /// Text prompt describing the desired output.
    #[arg(long)]
    pub prompt: String,

    /// Negative prompt (things to discourage).
    #[arg(long, default_value = "")]
    pub negative: String,

    /// Optional inpaint mask. When set, only mask=white pixels are
    /// re-painted; mask=black pixels are preserved. Grayscale, RGB
    /// (luminance), or RGBA (alpha channel) all accepted.
    #[arg(long, value_name = "PATH")]
    pub mask: Option<PathBuf>,

    /// Feather radius (pixels) applied to the mask edge. Softens
    /// the inpaint↔preserve transition. Only meaningful with --mask.
    #[arg(long = "mask-feather", default_value_t = 8, value_name = "PX")]
    pub mask_feather: u32,

    /// Invert the mask polarity (treat black as inpaint instead of
    /// white). Use when your mask source uses the opposite convention.
    #[arg(long = "mask-invert", default_value_t = false)]
    pub mask_invert: bool,

    /// img2img strength in [0, 1]. 0.0 = no change, 1.0 = full
    /// re-noise + denoise inside the mask. Default differs by mode:
    /// 0.6 for img2img (whole image), 1.0 for inpaint (--mask set).
    #[arg(long, value_name = "F")]
    pub strength: Option<f32>,

    /// Model: alias (sd15, sd21, sdxl, sdxl-turbo) or any HF repo id.
    /// Flux is not supported by img2img.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// Output size, e.g. 512x512. If absent, the input's dimensions
    /// are snapped to a multiple of 8 (VAE requirement) and used.
    #[arg(long)]
    pub size: Option<Size>,

    /// Number of variations to generate from the same input. Each
    /// gets a fresh seed.
    #[arg(long, short = 'n', default_value_t = 1)]
    pub count: u32,

    /// Denoising steps.
    #[arg(long, default_value_t = 28)]
    pub steps: usize,

    /// Classifier-free guidance scale.
    #[arg(long, default_value_t = 7.5)]
    pub guidance: f64,

    /// Base seed. Subsequent --count outputs use seed+1, seed+2, ...
    /// If omitted, a random seed is picked.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Scheduler. `default` follows the model's preferred scheduler.
    #[arg(long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// LoRA spec(s). Same grammar as `plakat generate --loras`.
    #[arg(long = "loras", value_delimiter = ',')]
    pub loras: Vec<LoraSpec>,

    /// LoRA weight scale multiplier.
    #[arg(long, default_value_t = 1.0)]
    pub lora_scale: f32,

    /// Output directory. Files land as
    /// `plakat-img2img-<seed>.png` or `plakat-inpaint-<seed>.png`.
    #[arg(long, default_value = "./out")]
    pub out: PathBuf,
}

pub async fn run(args: Img2ImgArgs, device: Device) -> Result<()> {
    // Strength: 0.6 for img2img, 1.0 for inpaint when not explicit.
    let strength = args
        .strength
        .unwrap_or_else(|| if args.mask.is_some() { 1.0 } else { 0.6 });
    if !(0.0..=1.0).contains(&strength) || !strength.is_finite() {
        anyhow::bail!("strength must be finite in [0, 1], got {strength}");
    }

    // Working resolution: explicit --size > input dims snapped to /8.
    let (width, height) = match args.size {
        Some(s) => (s.w, s.h),
        None => detect_input_size(&args.input)?,
    };
    if width % 8 != 0 || height % 8 != 0 {
        anyhow::bail!(
            "working size {width}x{height} must be a multiple of 8 (VAE constraint); \
             pass --size to override",
        );
    }

    let req = img2img::Request {
        prompt: args.prompt,
        negative: args.negative,
        model: args.model,
        device,
        loras: args.loras,
        lora_scale: args.lora_scale,
        input: args.input,
        mask: args.mask,
        mask_feather: args.mask_feather,
        mask_invert: args.mask_invert,
        width,
        height,
        count: args.count,
        steps: args.steps,
        guidance: args.guidance,
        scheduler: args.scheduler,
        strength,
        seed: args.seed,
        out_dir: args.out,
    };

    img2img::run(req).await
}

/// Read the input's actual dimensions and round each axis DOWN to
/// the nearest multiple of 8 (the VAE downsample factor). Avoids
/// silently introducing fractional-pixel resizes the user didn't
/// ask for.
fn detect_input_size(path: &std::path::Path) -> Result<(u32, u32)> {
    let (w, h) = image::image_dimensions(path)
        .with_context(|| format!("reading dimensions of {}", path.display()))?;
    let snap = |x: u32| (x / 8) * 8;
    let sw = snap(w).max(8);
    let sh = snap(h).max(8);
    Ok((sw, sh))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn detect_input_size_snaps_to_eight() {
        let img = RgbImage::from_pixel(513, 800, image::Rgb([0, 0, 0]));
        let tmp = std::env::temp_dir().join("plakat_img2img_size_test.png");
        img.save(&tmp).unwrap();
        let (w, h) = detect_input_size(&tmp).unwrap();
        // 513 → 512 (rounded down), 800 stays at 800.
        assert_eq!((w, h), (512, 800));
    }
}
