//! `plakat compose` — declarative layered-scene compositing.
//!
//! Stacks image layers (z-order = array order) onto a canvas: a layer with no
//! `at` fills the canvas (a background), each placed layer is positioned
//! (9-grid name or `x,y` fractions), scaled (fraction of canvas width), and
//! alpha-composited with optional opacity. **No GPU** — it composes existing
//! image assets (RGBA cutouts from `plakat transparent` / the artefact library,
//! or any image). `generate:` and inline `matte:` layers (which would need the
//! GPU) are a planned follow-up — pre-render / pre-matte those for now.
//!
//! Paths in the scene file resolve relative to the scene file's directory.

use anyhow::{Context, Result, anyhow, bail};
use clap::Args as ClapArgs;
use console::style;
use image::{DynamicImage, Rgba, RgbaImage};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(ClapArgs, Debug)]
pub struct ComposeArgs {
    /// Scene file (HJSON): `{ size: "WxH", out: "scene.png", layers: [...] }`.
    #[arg(value_name = "SCENE")]
    pub scene: PathBuf,
}

#[derive(Deserialize, Debug)]
struct Scene {
    /// Canvas size, `"WxH"` (e.g. `"1024x1024"`).
    size: String,
    /// Output path (relative to the scene file). `.png`/`.webp` keep alpha.
    out: PathBuf,
    layers: Vec<Layer>,
}

#[derive(Deserialize, Debug)]
struct Layer {
    /// Image file for this layer (an RGBA cutout, or any image).
    load: PathBuf,
    /// Placement: a 9-grid name (`center`, `top_left`, `bottom_right`, …) or
    /// `"x,y"` fractions in `[0,1]`. Omit → the layer fills the canvas.
    #[serde(default)]
    at: Option<String>,
    /// Width as a fraction of the canvas width (height keeps aspect). Default
    /// `1.0` for a placed layer; ignored for a background fill.
    #[serde(default)]
    scale: Option<f32>,
    /// Layer opacity in `[0,1]` (default `1.0`).
    #[serde(default)]
    opacity: Option<f32>,
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X', '*'])
        .ok_or_else(|| anyhow!("size must be \"WxH\", got {s:?}"))?;
    Ok((
        w.trim().parse().with_context(|| format!("size width {w:?}"))?,
        h.trim().parse().with_context(|| format!("size height {h:?}"))?,
    ))
}

/// 9-grid name or `"x,y"` fractions → `(ax, ay)` in `[0,1]`: the layer's own
/// point `(ax·lw, ay·lh)` is pinned to canvas `(ax·W, ay·H)`, so corners sit
/// flush and `center` centers — and the layer always stays on-canvas.
fn parse_at(s: &str) -> Result<(f32, f32)> {
    if let Some((x, y)) = s.split_once(',') {
        if let (Ok(x), Ok(y)) = (x.trim().parse::<f32>(), y.trim().parse::<f32>()) {
            return Ok((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)));
        }
    }
    match s.trim().to_lowercase().replace('-', "_").as_str() {
        "top_left" => Ok((0.0, 0.0)),
        "top_center" | "top_centre" | "top" => Ok((0.5, 0.0)),
        "top_right" => Ok((1.0, 0.0)),
        "center_left" | "centre_left" | "left" => Ok((0.0, 0.5)),
        "center" | "centre" | "middle" => Ok((0.5, 0.5)),
        "center_right" | "centre_right" | "right" => Ok((1.0, 0.5)),
        "bottom_left" => Ok((0.0, 1.0)),
        "bottom_center" | "bottom_centre" | "bottom" => Ok((0.5, 1.0)),
        "bottom_right" => Ok((1.0, 1.0)),
        other => bail!("bad 'at' {other:?}: use a 9-grid name or \"x,y\" fractions"),
    }
}

/// Over-blend `overlay` onto `canvas` at top-left `(ox, oy)` with `opacity`
/// (`out = ov·a + base·(1-a)`, `a = overlay_alpha · opacity`).
fn composite(canvas: &mut RgbaImage, overlay: &RgbaImage, ox: i64, oy: i64, opacity: f32) {
    let (cw, ch) = (canvas.width() as i64, canvas.height() as i64);
    let (ow, oh) = (overlay.width() as i64, overlay.height() as i64);
    let (x0, y0) = (ox.max(0), oy.max(0));
    let (x1, y1) = ((ox + ow).min(cw), (oy + oh).min(ch));
    if x0 >= x1 || y0 >= y1 {
        return; // entirely off-canvas
    }
    for y in y0..y1 {
        for x in x0..x1 {
            let ov = overlay.get_pixel((x - ox) as u32, (y - oy) as u32).0;
            let a = (ov[3] as f32) * opacity / 255.0;
            if a <= 0.0 {
                continue;
            }
            let cv = canvas.get_pixel(x as u32, y as u32).0;
            let inv = 1.0 - a;
            let r = (ov[0] as f32 * a + cv[0] as f32 * inv).round() as u8;
            let g = (ov[1] as f32 * a + cv[1] as f32 * inv).round() as u8;
            let b = (ov[2] as f32 * a + cv[2] as f32 * inv).round() as u8;
            let ao = ((cv[3] as f32) * inv + 255.0 * a).round() as u8;
            canvas.put_pixel(x as u32, y as u32, Rgba([r, g, b, ao]));
        }
    }
}

fn compose_scene(scene: &Scene, base_dir: &Path) -> Result<RgbaImage> {
    let (cw, ch) = parse_size(&scene.size)?;
    let mut canvas = RgbaImage::new(cw, ch);
    for (i, layer) in scene.layers.iter().enumerate() {
        let path = if layer.load.is_absolute() {
            layer.load.clone()
        } else {
            base_dir.join(&layer.load)
        };
        let img = image::open(&path)
            .with_context(|| format!("layer {i}: opening {}", path.display()))?
            .to_rgba8();
        let opacity = layer.opacity.unwrap_or(1.0).clamp(0.0, 1.0);
        match &layer.at {
            None => {
                // Background: stretch to the canvas, composite at the origin.
                let bg = image::imageops::resize(&img, cw, ch, image::imageops::FilterType::Lanczos3);
                composite(&mut canvas, &bg, 0, 0, opacity);
            }
            Some(at) => {
                let (ax, ay) = parse_at(at).with_context(|| format!("layer {i}"))?;
                let scale = layer.scale.unwrap_or(1.0).clamp(0.001, 4.0);
                let lw = ((cw as f32) * scale).round().max(1.0) as u32;
                let lh = ((lw as f32) * (img.height() as f32 / img.width().max(1) as f32))
                    .round()
                    .max(1.0) as u32;
                let scaled = image::imageops::resize(&img, lw, lh, image::imageops::FilterType::Lanczos3);
                let ox = (ax * (cw as f32 - lw as f32)).round() as i64;
                let oy = (ay * (ch as f32 - lh as f32)).round() as i64;
                composite(&mut canvas, &scaled, ox, oy, opacity);
            }
        }
    }
    Ok(canvas)
}

pub async fn run(args: ComposeArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.scene)
        .with_context(|| format!("reading scene {}", args.scene.display()))?;
    let scene: Scene = deser_hjson::from_str(&text)
        .with_context(|| format!("parsing scene {}", args.scene.display()))?;
    if scene.layers.is_empty() {
        bail!("compose: scene has no layers");
    }
    let base_dir = args.scene.parent().unwrap_or_else(|| Path::new("."));
    let canvas = compose_scene(&scene, base_dir)?;

    let out = if scene.out.is_absolute() {
        scene.out.clone()
    } else {
        base_dir.join(&scene.out)
    };
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if matches!(ext.as_str(), "png" | "webp") {
        canvas.save(&out)?;
    } else {
        DynamicImage::ImageRgba8(canvas).to_rgb8().save(&out)?;
    }
    println!(
        "{}  composed {} layer(s) → {}",
        style("✓").green(),
        scene.layers.len(),
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_size() {
        assert_eq!(parse_size("1024x768").unwrap(), (1024, 768));
        assert_eq!(parse_size("512 X 512").unwrap(), (512, 512));
        assert!(parse_size("nope").is_err());
    }

    #[test]
    fn parses_at_grid_and_custom() {
        assert_eq!(parse_at("center").unwrap(), (0.5, 0.5));
        assert_eq!(parse_at("top-left").unwrap(), (0.0, 0.0));
        assert_eq!(parse_at("bottom_right").unwrap(), (1.0, 1.0));
        assert_eq!(parse_at("0.25,0.75").unwrap(), (0.25, 0.75));
        assert!(parse_at("nowhere").is_err());
    }

    #[test]
    fn composite_over_blends_and_clips() {
        let mut canvas = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 255]));
        // Opaque red 2×2 at (1,1).
        let overlay = RgbaImage::from_pixel(2, 2, Rgba([255, 0, 0, 255]));
        composite(&mut canvas, &overlay, 1, 1, 1.0);
        assert_eq!(canvas.get_pixel(1, 1).0, [255, 0, 0, 255], "covered pixel is red");
        assert_eq!(canvas.get_pixel(0, 0).0, [0, 0, 0, 255], "outside untouched");
        // Half-opacity over black → ~128 red.
        let mut c2 = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255]));
        composite(&mut c2, &overlay, 0, 0, 0.5);
        let p = c2.get_pixel(0, 0).0;
        assert!((p[0] as i32 - 128).abs() <= 1 && p[1] == 0, "50% red over black ≈ 128");
        // Fully off-canvas is a no-op.
        let mut c3 = RgbaImage::from_pixel(2, 2, Rgba([7, 7, 7, 255]));
        composite(&mut c3, &overlay, 10, 10, 1.0);
        assert_eq!(c3.get_pixel(0, 0).0, [7, 7, 7, 255]);
    }
}
