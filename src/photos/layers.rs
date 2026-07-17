//! Layer compositing (RFC PHOTOS-1 Phase 8) — non-destructive image layers over a base image.
//!
//! A layer stack overlays one or more images on top of a base with independent position, scale,
//! opacity, blend mode, and z-order. Layers are the interactive front-end to the same compositing
//! the `plakat compose` command performs. The stack lives on the image's `album.hjson` record
//! (`layers`) so it survives sessions and stays editable; **flatten** bakes the composite into a new
//! derivative file (a `_layered.png` variant — the base is never modified), mirroring the ML-edit
//! variant model in [`super::mledit`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{DynamicImage, RgbImage};

use super::hjson::LayerEntry;

/// How a layer's pixels combine with what's beneath it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blend {
    Normal,
    Multiply,
    Screen,
    Overlay,
}

impl Blend {
    pub fn as_str(self) -> &'static str {
        match self {
            Blend::Normal => "normal",
            Blend::Multiply => "multiply",
            Blend::Screen => "screen",
            Blend::Overlay => "overlay",
        }
    }

    pub fn parse(s: &str) -> Blend {
        match s {
            "multiply" => Blend::Multiply,
            "screen" => Blend::Screen,
            "overlay" => Blend::Overlay,
            _ => Blend::Normal,
        }
    }

    /// Next mode in the cycle (for the interactive `b` key).
    pub fn cycle(self) -> Blend {
        match self {
            Blend::Normal => Blend::Multiply,
            Blend::Multiply => Blend::Screen,
            Blend::Screen => Blend::Overlay,
            Blend::Overlay => Blend::Normal,
        }
    }

    /// Combine a base channel `b` and layer channel `l` (both 0..1) under this mode.
    fn mix(self, b: f32, l: f32) -> f32 {
        match self {
            Blend::Normal => l,
            Blend::Multiply => b * l,
            Blend::Screen => 1.0 - (1.0 - b) * (1.0 - l),
            Blend::Overlay => {
                if b < 0.5 {
                    2.0 * b * l
                } else {
                    1.0 - 2.0 * (1.0 - b) * (1.0 - l)
                }
            }
        }
    }
}

/// One overlaid layer. `x`/`y` are the top-left position as a fraction of the base dimensions;
/// `scale` is the layer width as a fraction of the base width (aspect preserved); `opacity` 0..1.
#[derive(Debug, Clone)]
pub struct Layer {
    pub src: PathBuf,
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub opacity: f32,
    pub blend: Blend,
}

impl Layer {
    /// A fresh layer at half the base width, offset a quarter in, fully opaque, normal blend.
    pub fn new(src: PathBuf) -> Layer {
        Layer { src, x: 0.25, y: 0.25, scale: 0.5, opacity: 1.0, blend: Blend::Normal }
    }

    /// Short one-line label for the HUD / status line.
    pub fn label(&self) -> String {
        let name = self.src.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        format!(
            "{name}  {:.0}%  op{:.0}  {}",
            self.scale * 100.0,
            self.opacity * 100.0,
            self.blend.as_str()
        )
    }

    /// Serialise for `album.hjson`, storing the source relative to `album` when it lives under it
    /// (portable across moves), else absolute.
    pub fn to_entry(&self, album: &Path) -> LayerEntry {
        let src = self
            .src
            .strip_prefix(album)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| self.src.clone());
        LayerEntry {
            src: src.to_string_lossy().into_owned(),
            x: self.x,
            y: self.y,
            scale: self.scale,
            opacity: self.opacity,
            blend: self.blend.as_str().to_string(),
        }
    }

    /// Rebuild a layer from a stored entry, resolving a relative source against `album`.
    pub fn from_entry(e: &LayerEntry, album: &Path) -> Layer {
        let p = PathBuf::from(&e.src);
        let src = if p.is_absolute() { p } else { album.join(p) };
        Layer { src, x: e.x, y: e.y, scale: e.scale, opacity: e.opacity, blend: Blend::parse(&e.blend) }
    }
}

/// Composite `layers` (bottom-to-top) over an opaque `base`. When `marker` is `Some(i)`, a bright
/// border is drawn around that layer's placement — used by the interactive preview to show the
/// active layer; pass `None` for a clean flatten. Missing / unreadable sources are skipped.
pub fn composite(base: &DynamicImage, layers: &[Layer], marker: Option<usize>) -> DynamicImage {
    let mut canvas: RgbImage = base.to_rgb8();
    let (bw, bh) = (canvas.width(), canvas.height());
    for (i, layer) in layers.iter().enumerate() {
        let Ok(src) = image::open(&layer.src) else { continue };
        let lw = ((layer.scale.max(0.0) * bw as f32).round() as u32).max(1);
        // Fit the layer to `lw` wide, aspect preserved (u32::MAX height = width is the limit).
        let src = src.resize(lw, u32::MAX, image::imageops::FilterType::Lanczos3).to_rgba8();
        let (sw, sh) = (src.width(), src.height());
        let ox = (layer.x * bw as f32).round() as i32;
        let oy = (layer.y * bh as f32).round() as i32;
        let op = layer.opacity.clamp(0.0, 1.0);
        for (px, py, p) in src.enumerate_pixels() {
            let (cx, cy) = (ox + px as i32, oy + py as i32);
            if cx < 0 || cy < 0 || cx >= bw as i32 || cy >= bh as i32 {
                continue;
            }
            let a = (p.0[3] as f32 / 255.0) * op; // source alpha × layer opacity
            if a <= 0.0 {
                continue;
            }
            let bp = canvas.get_pixel_mut(cx as u32, cy as u32);
            for c in 0..3 {
                let bch = bp.0[c] as f32 / 255.0;
                let lch = p.0[c] as f32 / 255.0;
                let mixed = layer.blend.mix(bch, lch);
                let out = bch * (1.0 - a) + mixed * a;
                bp.0[c] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        if marker == Some(i) {
            draw_border(&mut canvas, ox, oy, sw, sh);
        }
    }
    DynamicImage::ImageRgb8(canvas)
}

/// Draw a 2-px cyan rectangle around `[ox, oy, ox+w, oy+h]`, clamped to the canvas.
fn draw_border(canvas: &mut RgbImage, ox: i32, oy: i32, w: u32, h: u32) {
    let c = image::Rgb([0u8, 220, 255]);
    let (x1, y1) = (ox + w as i32 - 1, oy + h as i32 - 1);
    for t in 0..2 {
        for x in ox..=x1 {
            set(canvas, x, oy + t, c);
            set(canvas, x, y1 - t, c);
        }
        for y in oy..=y1 {
            set(canvas, ox + t, y, c);
            set(canvas, x1 - t, y, c);
        }
    }
}

fn set(canvas: &mut RgbImage, x: i32, y: i32, c: image::Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < canvas.width() && (y as u32) < canvas.height() {
        canvas.put_pixel(x as u32, y as u32, c);
    }
}

/// Flatten the stack over the image at `base_path`, saving the composite as a new deduped
/// `<stem>_layered.png` in `album`. Returns the output filename (the base is untouched).
pub fn flatten(base_path: &Path, album: &Path, layers: &[Layer]) -> Result<String> {
    let base = image::open(base_path).with_context(|| format!("opening {}", base_path.display()))?;
    let out = composite(&base, layers, None);
    let dest = dest_path(album, base_path);
    out.save(&dest).with_context(|| format!("writing {}", dest.display()))?;
    Ok(dest.file_name().unwrap_or_default().to_string_lossy().into_owned())
}

/// `<album>/<stem>_layered.png`, suffixing `-2`, `-3`, … on a collision.
fn dest_path(album: &Path, input: &Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let cand = album.join(format!("{stem}_layered.png"));
    if !cand.exists() {
        return cand;
    }
    for i in 2..10_000 {
        let c = album.join(format!("{stem}_layered-{i}.png"));
        if !c.exists() {
            return c;
        }
    }
    album.join(format!("{stem}_layered-dup.png"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn solid_rgb(w: u32, h: u32, px: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(w, h, image::Rgb(px)))
    }

    #[test]
    fn blend_math_is_correct() {
        // Multiply of mid-greys, screen brighter than either, overlay of white with black = black-ish.
        assert!((Blend::Multiply.mix(0.5, 0.5) - 0.25).abs() < 1e-6);
        assert!((Blend::Screen.mix(0.5, 0.5) - 0.75).abs() < 1e-6);
        assert!((Blend::Normal.mix(0.2, 0.9) - 0.9).abs() < 1e-6);
        // Overlay: base<0.5 branch = 2*b*l.
        assert!((Blend::Overlay.mix(0.25, 0.5) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn blend_str_roundtrips_and_cycles() {
        for b in [Blend::Normal, Blend::Multiply, Blend::Screen, Blend::Overlay] {
            assert_eq!(Blend::parse(b.as_str()), b);
        }
        // The cycle visits all four and returns home.
        let mut b = Blend::Normal;
        for _ in 0..4 {
            b = b.cycle();
        }
        assert_eq!(b, Blend::Normal);
    }

    #[test]
    fn opaque_full_scale_normal_layer_replaces_base() {
        let base = solid_rgb(10, 10, [0, 0, 0]);
        let dir = std::env::temp_dir().join(format!("plakat-layers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lp = dir.join("red.png");
        solid_rgb(10, 10, [255, 0, 0]).save(&lp).unwrap();
        // A full-frame, fully-opaque, normal layer covering the whole base → base becomes red.
        let l = Layer { src: lp, x: 0.0, y: 0.0, scale: 1.0, opacity: 1.0, blend: Blend::Normal };
        let out = composite(&base, &[l], None).to_rgb8();
        assert_eq!(out.get_pixel(5, 5).0, [255, 0, 0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn half_opacity_blends_halfway() {
        let base = solid_rgb(4, 4, [0, 0, 0]);
        let dir = std::env::temp_dir().join(format!("plakat-layers-op-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lp = dir.join("white.png");
        DynamicImage::ImageRgba8(ImageBuffer::from_pixel(4, 4, Rgba([255, 255, 255, 255])))
            .save(&lp)
            .unwrap();
        let l = Layer { src: lp, x: 0.0, y: 0.0, scale: 1.0, opacity: 0.5, blend: Blend::Normal };
        let out = composite(&base, &[l], None).to_rgb8();
        // 0*(1-0.5) + 1*0.5 = 0.5 → ~128.
        let v = out.get_pixel(1, 1).0[0];
        assert!((120..=135).contains(&v), "got {v}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entry_roundtrip_relative_and_absolute() {
        let album = PathBuf::from("/tmp/album");
        // In-album source stores relative; out-of-album stays absolute.
        let inside = Layer::new(album.join("cutout.png"));
        let e = inside.to_entry(&album);
        assert_eq!(e.src, "cutout.png");
        assert_eq!(Layer::from_entry(&e, &album).src, album.join("cutout.png"));

        let outside = Layer { src: PathBuf::from("/other/logo.png"), ..Layer::new(PathBuf::new()) };
        let e2 = outside.to_entry(&album);
        assert_eq!(e2.src, "/other/logo.png");
        assert_eq!(Layer::from_entry(&e2, &album).src, PathBuf::from("/other/logo.png"));
    }

    #[test]
    fn flatten_writes_a_deduped_variant() {
        let dir = std::env::temp_dir().join(format!("plakat-layers-flat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("shot.png");
        solid_rgb(8, 8, [10, 20, 30]).save(&base).unwrap();
        let name = flatten(&base, &dir, &[]).unwrap();
        assert_eq!(name, "shot_layered.png");
        assert!(dir.join(&name).exists());
        // A second flatten dedups.
        let name2 = flatten(&base, &dir, &[]).unwrap();
        assert_eq!(name2, "shot_layered-2.png");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
