use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

use crate::imaging::upscale::Method;

#[derive(ClapArgs, Debug)]
pub struct UpscaleArgs {
    /// Input image.
    #[arg(long = "in", value_name = "IN")]
    pub input: PathBuf,

    /// Output image. Extension determines format (.png, .jpg, .webp).
    #[arg(long, value_name = "OUT")]
    pub out: PathBuf,

    /// Scale factor for classical filters (e.g. 2 for 2×, 4 for 4×). Ignored
    /// for ML methods — their scale is fixed by the model.
    #[arg(long, default_value_t = 2.0)]
    pub scale: f32,

    /// Method:
    ///   nearest | bilinear | bicubic | lanczos                    (classical)
    ///   real-esrgan-x2 | real-esrgan-x4 | real-esrgan-anime-x4    (ML, RRDBNet)
    #[arg(long, default_value = "lanczos")]
    pub method: Method,
}

pub async fn run(args: UpscaleArgs, device: Device) -> Result<()> {
    let (w, h, nw, nh) = if args.method.is_ml() {
        crate::imaging::upscale::ml_upscale(&args.input, &args.out, args.method, &device).await?
    } else {
        crate::imaging::upscale::upscale(&args.input, &args.out, args.scale, args.method)?
    };
    let effective_scale = args.method.native_scale().unwrap_or(args.scale);
    println!(
        "{}  {}×{} → {}×{}  ({:.2}×, {:?})",
        style("✓").green(),
        w,
        h,
        nw,
        nh,
        effective_scale,
        args.method,
    );
    println!("→ {}", args.out.display());
    Ok(())
}
