use anyhow::Result;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

#[derive(ClapArgs, Debug)]
pub struct TransparentArgs {
    /// Input image.
    #[arg(long = "in", value_name = "IN")]
    pub input: PathBuf,

    /// Output image. Use a .png (or .webp) extension to preserve alpha.
    #[arg(long, value_name = "OUT")]
    pub out: PathBuf,

    /// Per-channel max diff tolerance against the upper-left pixel
    /// (0 = exact match; 10 absorbs JPEG noise; 30–50 broadens to anti-aliased edges).
    #[arg(long, default_value_t = 0)]
    pub tolerance: u8,
}

pub async fn run(args: TransparentArgs) -> Result<()> {
    let r =
        crate::imaging::transparent::make_transparent(&args.input, &args.out, args.tolerance)?;
    let pct = 100.0 * r.transparent_pixels as f64 / r.total_pixels.max(1) as f64;
    println!(
        "{}  key #{:02x}{:02x}{:02x}  •  {}×{}  •  {}/{} pixels transparent ({:.1}%)",
        style("✓").green(),
        r.key_rgb[0],
        r.key_rgb[1],
        r.key_rgb[2],
        r.width,
        r.height,
        r.transparent_pixels,
        r.total_pixels,
        pct,
    );
    println!("→ {}", args.out.display());
    Ok(())
}
