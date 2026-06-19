use anyhow::{Context, Result, anyhow};
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
    /// Optional when `--depth-band` is given.
    #[arg(long = "point", value_name = "X,Y[:bg]")]
    pub points: Vec<String>,

    /// Select by DEPTH band instead of (or together with) points: `LO,HI` in
    /// normalized depth `0..1` where **1.0 = nearest, 0.0 = farthest**. So
    /// `0.6,1.0` masks the foreground, `0.0,0.3` the far background. Uses
    /// Depth-Anything-V2 (downloaded once). Combine with `--point` to intersect
    /// (this object, but only where it's near). Pass at least one of
    /// `--point` / `--depth-band`.
    #[arg(long = "depth-band", value_name = "LO,HI")]
    pub depth_band: Option<String>,

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

/// Parse `LO,HI` into a normalized depth band, validating `0 ≤ lo < hi ≤ 1`.
fn parse_band(s: &str) -> Result<(f32, f32)> {
    let (ls, hs) = s
        .split_once(',')
        .ok_or_else(|| anyhow!("bad --depth-band '{s}': expected 'LO,HI'"))?;
    let lo: f32 = ls
        .trim()
        .parse()
        .map_err(|_| anyhow!("bad --depth-band '{s}': LO ('{ls}') is not a number"))?;
    let hi: f32 = hs
        .trim()
        .parse()
        .map_err(|_| anyhow!("bad --depth-band '{s}': HI ('{hs}') is not a number"))?;
    if !(0.0..=1.0).contains(&lo) || !(0.0..=1.0).contains(&hi) {
        anyhow::bail!("bad --depth-band '{s}': LO/HI must be in 0..1 (1.0 = nearest, 0.0 = farthest)");
    }
    if lo >= hi {
        anyhow::bail!("bad --depth-band '{s}': need LO < HI (got {lo},{hi})");
    }
    Ok((lo, hi))
}

pub async fn run(args: SegmentArgs, device: Device) -> Result<()> {
    let points: Vec<PointPrompt> = args
        .points
        .iter()
        .map(|s| parse_point(s))
        .collect::<Result<_>>()?;
    let band = args.depth_band.as_deref().map(parse_band).transpose()?;

    if points.is_empty() && band.is_none() {
        anyhow::bail!("no selection: pass --point X,Y and/or --depth-band LO,HI");
    }

    // Build whichever source mask(s) are requested (original resolution); when
    // both are given, intersect them; then run the shared post-processing.
    let point_mask = if points.is_empty() {
        None
    } else {
        Some(crate::pipelines::sam::build_selection_mask(&args.input, &points, &device).await?)
    };
    let depth_mask = if let Some((lo, hi)) = band {
        let (w, h) = image::image_dimensions(&args.input)
            .with_context(|| format!("reading dimensions of {}", args.input.display()))?;
        let depth = crate::pipelines::depth::DepthPipeline::load(device.clone())
            .await?
            .depth_map(&args.input, w, h)?;
        Some(crate::pipelines::sam::depth_band_to_mask(&depth, w, h, lo, hi))
    } else {
        None
    };

    let mask = match (point_mask, depth_mask) {
        (Some(p), Some(d)) => crate::pipelines::sam::intersect_masks(&p, &d),
        (Some(p), None) => p,
        (None, Some(d)) => d,
        (None, None) => unreachable!("validated at least one source above"),
    };
    crate::pipelines::sam::finish_mask(mask, args.invert, args.grow, args.feather, &args.out)?;

    let mut srcs = Vec::new();
    if !points.is_empty() {
        srcs.push(format!(
            "{} point{}",
            points.len(),
            if points.len() == 1 { "" } else { "s" }
        ));
    }
    if let Some((lo, hi)) = band {
        srcs.push(format!("depth {lo}–{hi}"));
    }
    println!(
        "{}  segmented ({}{})  •  {}",
        style("✓").green(),
        srcs.join(" ∩ "),
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

    #[test]
    fn parses_and_validates_depth_band() {
        assert_eq!(parse_band("0.6,1.0").unwrap(), (0.6, 1.0));
        assert_eq!(parse_band(" 0.0 , 0.3 ").unwrap(), (0.0, 0.3));
        assert!(parse_band("0.6").is_err(), "missing comma");
        assert!(parse_band("a,1").is_err(), "non-numeric");
        assert!(parse_band("0.8,0.2").is_err(), "lo >= hi");
        assert!(parse_band("0.5,0.5").is_err(), "lo == hi");
        assert!(parse_band("-0.1,0.5").is_err(), "below 0");
        assert!(parse_band("0.5,1.5").is_err(), "above 1");
    }
}
