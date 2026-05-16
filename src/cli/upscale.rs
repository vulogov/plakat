use anyhow::Result;
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

    /// Scale factor (e.g. 2 for 2×, 4 for 4×). Non-integer values OK.
    #[arg(long, default_value_t = 2.0)]
    pub scale: f32,

    /// Resampling filter: nearest | bilinear | bicubic | lanczos
    #[arg(long, default_value = "lanczos")]
    pub method: Method,
}

pub async fn run(args: UpscaleArgs) -> Result<()> {
    let (w, h, nw, nh) =
        crate::imaging::upscale::upscale(&args.input, &args.out, args.scale, args.method)?;
    println!(
        "{}  {}×{} → {}×{}  ({:.2}×, {:?})",
        style("✓").green(),
        w,
        h,
        nw,
        nh,
        args.scale,
        args.method,
    );
    println!("→ {}", args.out.display());
    Ok(())
}
