use anyhow::{Result, anyhow};
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

use crate::pipelines::sam::PointPrompt;

#[derive(ClapArgs, Debug)]
pub struct SegmentArgs {
    /// Input image to segment.
    #[arg(long = "in", value_name = "IN")]
    pub input: PathBuf,

    /// Output mask (PNG). White = selected region, black = excluded — feed it
    /// straight to `img2img --mask` (inpaint) or any `--mask` consumer.
    #[arg(long, value_name = "OUT")]
    pub out: PathBuf,

    /// A prompt point, repeatable. `X,Y` selects (foreground); append `:bg` to
    /// exclude a region. Coords are normalized `0..1` by default, or pixels if
    /// any value exceeds 1 (e.g. `--point 0.5,0.4` or `--point 512,400`). Click
    /// the object to select it; add `:bg` points to carve away over-selection.
    #[arg(long = "point", value_name = "X,Y[:bg]", required = true)]
    pub points: Vec<String>,

    /// Invert the mask — select everything EXCEPT the prompted object (handy
    /// for "change the background, keep the subject").
    #[arg(long, default_value_t = false)]
    pub invert: bool,

    /// Grow the selection by N pixels (dilate) before output. For inpaint edits,
    /// a small margin (~8–12) keeps the repaint off the subject's fringe — the
    /// cure for rope/halo artefacts at the mask boundary. Applied before
    /// `--invert`, so it always expands the *subject*.
    #[arg(long, default_value_t = 0, value_name = "PX")]
    pub grow: u32,

    /// Feather the mask edge by N pixels (gaussian) for a soft inpaint blend
    /// instead of a hard seam.
    #[arg(long, default_value_t = 0, value_name = "PX")]
    pub feather: u32,
}

/// Parse `X,Y`, `X,Y:fg`, or `X,Y:bg` into a point prompt (default foreground).
fn parse_point(s: &str) -> Result<PointPrompt> {
    let (coords, foreground) = match s.rsplit_once(':') {
        Some((c, "bg")) => (c, false),
        Some((c, "fg")) => (c, true),
        Some((_, tag)) => {
            return Err(anyhow!(
                "bad point '{s}': suffix ':{tag}' must be ':fg' or ':bg'"
            ));
        }
        None => (s, true),
    };
    let (xs, ys) = coords
        .split_once(',')
        .ok_or_else(|| anyhow!("bad point '{s}': expected 'X,Y'"))?;
    let x: f64 = xs
        .trim()
        .parse()
        .map_err(|_| anyhow!("bad point '{s}': X ('{xs}') is not a number"))?;
    let y: f64 = ys
        .trim()
        .parse()
        .map_err(|_| anyhow!("bad point '{s}': Y ('{ys}') is not a number"))?;
    Ok(PointPrompt { x, y, foreground })
}

pub async fn run(args: SegmentArgs, device: Device) -> Result<()> {
    let points: Vec<PointPrompt> = args
        .points
        .iter()
        .map(|s| parse_point(s))
        .collect::<Result<_>>()?;

    crate::pipelines::sam::segment(
        &args.input,
        &args.out,
        &points,
        args.invert,
        args.grow,
        args.feather,
        &device,
    )
    .await?;

    let n = points.len();
    println!(
        "{}  segmented ({} point{}{})  •  {}",
        style("✓").green(),
        n,
        if n == 1 { "" } else { "s" },
        if args.invert { ", inverted" } else { "" },
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_fg_and_explicit_tags() {
        let p = parse_point("0.5,0.4").unwrap();
        assert!((p.x - 0.5).abs() < 1e-9 && (p.y - 0.4).abs() < 1e-9);
        assert!(p.foreground, "no suffix → foreground");
        assert!(parse_point("0.5,0.4:fg").unwrap().foreground);
        assert!(!parse_point("0.5,0.4:bg").unwrap().foreground);
    }

    #[test]
    fn pixel_coords_pass_through() {
        // The pipeline normalizes pixel coords later; the parser keeps them as-is.
        let p = parse_point("512,400").unwrap();
        assert_eq!((p.x, p.y), (512.0, 400.0));
        assert!(p.foreground);
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_point("0.5").is_err(), "missing comma");
        assert!(parse_point("a,b").is_err(), "non-numeric");
        assert!(parse_point("0.5,0.4:xy").is_err(), "bad suffix");
    }
}
