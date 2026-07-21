use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

#[derive(ClapArgs, Debug)]
pub struct TransparentArgs {
    /// Input image.
    #[arg(help_heading = "Cutout", long = "in", value_name = "IN")]
    pub input: PathBuf,

    /// Output image. Use a .png (or .webp) extension to preserve alpha.
    #[arg(help_heading = "Size & output", long, value_name = "OUT")]
    pub out: PathBuf,

    /// Per-channel tolerance for the background flood-fill — the max diff
    /// between adjacent pixels as it grows from the image corners. 0 = exact;
    /// ~10 absorbs render/JPEG noise on a flat backdrop; ~20–32 follows
    /// gradients / soft shadows; too high leaks into the subject.
    #[arg(help_heading = "Cutout", long, default_value_t = 10)]
    pub tolerance: u8,

    /// Crop the output to the subject's non-transparent bounding box — useful
    /// when the cut-out feeds a compositor (e.g. the artefact library) that
    /// scales by frame size, so a centred subject isn't left tiny.
    #[arg(help_heading = "Cutout", long, default_value_t = false)]
    pub crop: bool,

    /// Smart, content-aware cut-out: a salient-object model (U2Net) predicts the
    /// foreground matte directly from image content — no chroma backdrop needed,
    /// works on photoreal / painted subjects on ANY background. Overrides the
    /// corner flood-fill (`--tolerance` is ignored). Downloads the model once.
    #[arg(help_heading = "Cutout", long, default_value_t = false)]
    pub matte: bool,
}

pub async fn run(args: TransparentArgs, device: Device) -> Result<()> {
    if args.matte {
        crate::pipelines::matting::cutout(&args.input, &args.out, args.crop, &device).await?;
        println!(
            "{}  smart matte cut-out  •  {}",
            style("✓").green(),
            args.out.display()
        );
        return Ok(());
    }

    let r = crate::imaging::transparent::make_transparent(
        &args.input,
        &args.out,
        args.tolerance,
        args.crop,
    )?;
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
