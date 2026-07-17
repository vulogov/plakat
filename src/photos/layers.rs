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

use super::hjson::{LayerEntry, MaskEntry};

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

/// A shape-mask outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    Rect,
    Ellipse,
}

/// Per-layer mask — multiplies the layer's alpha before compositing so only part of it shows.
/// Freehand painting isn't possible on a terminal graphics protocol, so masks are parametric:
/// a feathered shape region, or an external grayscale matte.
#[derive(Debug, Clone)]
pub enum Mask {
    None,
    /// A rectangle/ellipse region in fractions of the **layer** (x,y top-left, w,h), soft-edged by
    /// `feather` (fraction of the region), `invert` to hide the region instead of showing it.
    Shape { kind: ShapeKind, x: f32, y: f32, w: f32, h: f32, feather: f32, invert: bool },
    /// Luminance matte from an external image (resized to the layer): white shows, black hides.
    Image { src: PathBuf, invert: bool },
}

impl Mask {
    /// A centred feathered shape covering most of the layer — the default when a shape mask is added.
    pub fn centered_shape(kind: ShapeKind) -> Mask {
        Mask::Shape { kind, x: 0.05, y: 0.05, w: 0.9, h: 0.9, feather: 0.15, invert: false }
    }

    pub fn is_some(&self) -> bool {
        !matches!(self, Mask::None)
    }

    /// Short tag for the HUD label.
    fn tag(&self) -> &'static str {
        match self {
            Mask::None => "",
            Mask::Shape { kind: ShapeKind::Ellipse, .. } => "◯mask",
            Mask::Shape { kind: ShapeKind::Rect, .. } => "▭mask",
            Mask::Image { .. } => "▦matte",
        }
    }

    /// Alpha multiplier (0..1) for a layer-local fraction coordinate `(fx, fy)` in `[0,1]`. `matte`
    /// is the pre-decoded luma buffer for [`Mask::Image`] (any size; sampled by fraction).
    fn coverage(&self, fx: f32, fy: f32, matte: Option<&image::GrayImage>) -> f32 {
        match self {
            Mask::None => 1.0,
            Mask::Shape { kind, x, y, w, h, feather, invert } => {
                let a = shape_coverage(*kind, fx, fy, *x, *y, *w, *h, *feather);
                if *invert { 1.0 - a } else { a }
            }
            Mask::Image { invert, .. } => {
                let a = matte
                    .map(|m| {
                        let mx = ((fx * m.width() as f32) as u32).min(m.width().saturating_sub(1));
                        let my = ((fy * m.height() as f32) as u32).min(m.height().saturating_sub(1));
                        m.get_pixel(mx, my).0[0] as f32 / 255.0
                    })
                    .unwrap_or(1.0);
                if *invert { 1.0 - a } else { a }
            }
        }
    }
}

/// Soft coverage (0..1) of a rect/ellipse region `[x, y, w, h]` (layer fractions) at fraction
/// `(fx, fy)`, ramped over `feather` (fraction of the smaller region side).
fn shape_coverage(kind: ShapeKind, fx: f32, fy: f32, x: f32, y: f32, w: f32, h: f32, feather: f32) -> f32 {
    let (cx, cy) = (x + w / 2.0, y + h / 2.0);
    let (rx, ry) = ((w / 2.0).max(1e-4), (h / 2.0).max(1e-4));
    match kind {
        ShapeKind::Ellipse => {
            // Normalised radius: 0 at centre, 1 on the ellipse edge.
            let r = (((fx - cx) / rx).powi(2) + ((fy - cy) / ry).powi(2)).sqrt();
            if feather <= 1e-4 {
                if r <= 1.0 { 1.0 } else { 0.0 }
            } else {
                ((1.0 - r) / feather).clamp(0.0, 1.0)
            }
        }
        ShapeKind::Rect => {
            // Distance inside the nearest edge, normalised by feather × the smaller half-side.
            let d = (fx - x).min(x + w - fx).min(fy - y).min(y + h - fy);
            if d < 0.0 {
                return 0.0;
            }
            let ramp = (feather * rx.min(ry)).max(1e-4);
            (d / ramp).clamp(0.0, 1.0)
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
    pub mask: Mask,
}

impl Layer {
    /// A fresh layer at half the base width, offset a quarter in, fully opaque, normal blend, no mask.
    pub fn new(src: PathBuf) -> Layer {
        Layer { src, x: 0.25, y: 0.25, scale: 0.5, opacity: 1.0, blend: Blend::Normal, mask: Mask::None }
    }

    /// Short one-line label for the HUD / status line.
    pub fn label(&self) -> String {
        let name = self.src.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let mask = if self.mask.is_some() { format!("  {}", self.mask.tag()) } else { String::new() };
        format!(
            "{name}  {:.0}%  op{:.0}  {}{mask}",
            self.scale * 100.0,
            self.opacity * 100.0,
            self.blend.as_str()
        )
    }

    /// Serialise for `album.hjson`, storing the source relative to `album` when it lives under it
    /// (portable across moves), else absolute.
    pub fn to_entry(&self, album: &Path) -> LayerEntry {
        LayerEntry {
            src: rel_src(&self.src, album),
            x: self.x,
            y: self.y,
            scale: self.scale,
            opacity: self.opacity,
            blend: self.blend.as_str().to_string(),
            mask: mask_to_entry(&self.mask, album),
        }
    }

    /// Rebuild a layer from a stored entry, resolving a relative source against `album`.
    pub fn from_entry(e: &LayerEntry, album: &Path) -> Layer {
        Layer {
            src: resolve_src(&e.src, album),
            x: e.x,
            y: e.y,
            scale: e.scale,
            opacity: e.opacity,
            blend: Blend::parse(&e.blend),
            mask: mask_from_entry(e.mask.as_ref(), album),
        }
    }
}

/// Store a source path relative to `album` when it lives under it (portable), else absolute.
fn rel_src(src: &Path, album: &Path) -> String {
    src.strip_prefix(album)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| src.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn resolve_src(s: &str, album: &Path) -> PathBuf {
    let p = PathBuf::from(s);
    if p.is_absolute() { p } else { album.join(p) }
}

fn mask_to_entry(mask: &Mask, album: &Path) -> Option<MaskEntry> {
    match mask {
        Mask::None => None,
        Mask::Shape { kind, x, y, w, h, feather, invert } => Some(MaskEntry {
            kind: match kind {
                ShapeKind::Rect => "rect",
                ShapeKind::Ellipse => "ellipse",
            }
            .to_string(),
            x: *x,
            y: *y,
            w: *w,
            h: *h,
            feather: *feather,
            invert: *invert,
            src: String::new(),
        }),
        Mask::Image { src, invert } => Some(MaskEntry {
            kind: "image".to_string(),
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
            feather: 0.0,
            invert: *invert,
            src: rel_src(src, album),
        }),
    }
}

fn mask_from_entry(e: Option<&MaskEntry>, album: &Path) -> Mask {
    let Some(m) = e else { return Mask::None };
    match m.kind.as_str() {
        "ellipse" => Mask::Shape {
            kind: ShapeKind::Ellipse,
            x: m.x,
            y: m.y,
            w: m.w,
            h: m.h,
            feather: m.feather,
            invert: m.invert,
        },
        "rect" => Mask::Shape {
            kind: ShapeKind::Rect,
            x: m.x,
            y: m.y,
            w: m.w,
            h: m.h,
            feather: m.feather,
            invert: m.invert,
        },
        "image" => Mask::Image { src: resolve_src(&m.src, album), invert: m.invert },
        _ => Mask::None,
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
        // Pre-decode an image matte once, at the layer size (sampled by fraction below).
        let matte = match &layer.mask {
            Mask::Image { src, .. } => image::open(src)
                .ok()
                .map(|m| m.resize_exact(sw.max(1), sh.max(1), image::imageops::FilterType::Triangle).to_luma8()),
            _ => None,
        };
        for (px, py, p) in src.enumerate_pixels() {
            let (cx, cy) = (ox + px as i32, oy + py as i32);
            if cx < 0 || cy < 0 || cx >= bw as i32 || cy >= bh as i32 {
                continue;
            }
            let cov = layer.mask.coverage(px as f32 / sw as f32, py as f32 / sh as f32, matte.as_ref());
            let a = (p.0[3] as f32 / 255.0) * op * cov; // source alpha × layer opacity × mask
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
        let l = Layer { src: lp, x: 0.0, y: 0.0, scale: 1.0, opacity: 1.0, blend: Blend::Normal, mask: Mask::None };
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
        let l = Layer { src: lp, x: 0.0, y: 0.0, scale: 1.0, opacity: 0.5, blend: Blend::Normal, mask: Mask::None };
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
    fn ellipse_mask_is_opaque_at_centre_and_clear_at_corner() {
        // Full-layer centred ellipse: centre fully covered, a corner fully outside.
        let m = Mask::centered_shape(ShapeKind::Ellipse);
        assert!((m.coverage(0.5, 0.5, None) - 1.0).abs() < 1e-3, "centre should be 1");
        assert_eq!(m.coverage(0.98, 0.98, None), 0.0, "corner should be 0");
        // Invert flips it.
        let inv = match m {
            Mask::Shape { kind, x, y, w, h, feather, .. } => {
                Mask::Shape { kind, x, y, w, h, feather, invert: true }
            }
            _ => unreachable!(),
        };
        assert!(inv.coverage(0.5, 0.5, None) < 1e-3, "inverted centre should be ~0");
    }

    #[test]
    fn offcentre_ellipse_mask_tracks_its_position() {
        // An ellipse mask placed in the top-left quadrant is opaque at its own centre and clear at
        // the layer centre — i.e. position is honoured, not forced to the middle.
        let m = Mask::Shape { kind: ShapeKind::Ellipse, x: 0.0, y: 0.0, w: 0.4, h: 0.4, feather: 0.1, invert: false };
        assert!((m.coverage(0.2, 0.2, None) - 1.0).abs() < 1e-3, "own centre should be 1");
        assert_eq!(m.coverage(0.5, 0.5, None), 0.0, "layer centre should be outside");
    }

    #[test]
    fn image_matte_uses_luminance() {
        // A matte that is white on the left half, black on the right.
        let mut matte = image::GrayImage::new(10, 4);
        for (x, _y, p) in matte.enumerate_pixels_mut() {
            p.0[0] = if x < 5 { 255 } else { 0 };
        }
        let m = Mask::Image { src: PathBuf::new(), invert: false };
        assert!((m.coverage(0.1, 0.5, Some(&matte)) - 1.0).abs() < 1e-3);
        assert_eq!(m.coverage(0.9, 0.5, Some(&matte)), 0.0);
    }

    #[test]
    fn mask_entry_roundtrips() {
        let album = PathBuf::from("/tmp/album");
        // Shape mask.
        let mut l = Layer::new(album.join("l.png"));
        l.mask = Mask::Shape { kind: ShapeKind::Rect, x: 0.1, y: 0.2, w: 0.6, h: 0.5, feather: 0.2, invert: true };
        let back = Layer::from_entry(&l.to_entry(&album), &album);
        match back.mask {
            Mask::Shape { kind: ShapeKind::Rect, x, feather, invert, .. } => {
                assert!((x - 0.1).abs() < 1e-6 && (feather - 0.2).abs() < 1e-6 && invert);
            }
            _ => panic!("lost shape mask"),
        }
        // Image matte, in-album → relative.
        l.mask = Mask::Image { src: album.join("matte.png"), invert: false };
        let e = l.to_entry(&album);
        assert_eq!(e.mask.as_ref().unwrap().src, "matte.png");
        match Layer::from_entry(&e, &album).mask {
            Mask::Image { src, .. } => assert_eq!(src, album.join("matte.png")),
            _ => panic!("lost image mask"),
        }
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
