//! T1 pixel editing (RFC PHOTOS-1 Phase 3) — non-destructive, replayable parametric edits.
//!
//! Edits never mutate your originals irreversibly: on the first edit of an image, the pristine file
//! is copied into a hidden `.plakat_edits/` backup (skipped by the library walk), and the visible
//! file is *re-derived* from that backup by replaying the whole edit list. Because every rebuild
//! starts from the original, chaining ten edits costs one re-encode from the pristine bytes, not ten
//! generational ones. Undo drops the last edit and rebuilds; revert clears them and restores the
//! original. The edit list lives in the image's `album.hjson` record (`edits`), so it survives across
//! sessions and travels with the album.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::DynamicImage;

use super::hjson::EditEntry;

/// One replayable pixel operation. Serialised to/from an [`EditEntry`] for `album.hjson`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditOp {
    RotateCw,
    RotateCcw,
    Rotate180,
    FlipH,
    FlipV,
    Grayscale,
    /// Brightness delta (image `brighten`, ±/step).
    Brightness(i32),
    /// Contrast delta (image `adjust_contrast`, ±/step).
    Contrast(i32),
    // --- Tonal / colour adjustments (amount in ~[-100, 100], applied per-step & chainable) ---
    /// Overall exposure in stops (multiplicative light).
    Exposure(i32),
    /// Brilliance — adaptive: lifts shadows/mids and recovers highlights for richer depth.
    Brilliance(i32),
    /// Brighten/darken the highlight tones.
    Highlights(i32),
    /// Brighten/darken the mid tones.
    Midrange(i32),
    /// Brighten/darken the shadow tones.
    Shadows(i32),
    /// Black point — remap the darkest tones (raise = crush blacks, lower = lift).
    Blackpoint(i32),
    /// Global saturation (chroma scale; -100 = greyscale).
    Saturation(i32),
    /// Vibrance — saturation weighted toward the least-saturated pixels.
    Vibrance(i32),
    /// Warmth — colour temperature (blue ↔ orange).
    Warmth(i32),
    /// Tint — green ↔ magenta balance.
    Tint(i32),
    /// Definition — midtone local contrast (clarity / structure).
    Definition(i32),
    /// Sharpen (positive) / soften (negative) via unsharp mask.
    Sharpen(i32),
    /// Noise reduction (edge-softening blend; positive only).
    NoiseReduction(i32),
    /// One-tap auto-enhance: per-channel histogram stretch (auto levels + auto colour balance).
    AutoEnhance,
    /// Straighten: rotate by `degrees` about the centre, auto-cropping the empty corners.
    Straighten(i32),
    /// Vignette: darken (positive) / lighten (negative) the frame edges radially.
    Vignette(i32),
    /// Levels: map input `[black, white]` (0..255) to full range with a midtone `gamma` (×100:
    /// 100 = 1.00). `gamma > 1` brightens the mids.
    Levels { black: i32, white: i32, gamma: i32 },
    /// Centered square (1:1) crop.
    CropSquare,
    /// Centered crop to aspect ratio `w:h` (largest fitting rect).
    CropAspect { w: u32, h: u32 },
    /// Free-form crop: a rectangle in [0,1] fractions of the image (x, y, width, height).
    Crop { x: f32, y: f32, w: f32, h: f32 },
    /// Centered crop to exactly `w`×`h` pixels (clamped to the image).
    CropPx { w: u32, h: u32 },
    /// Resize to fit within `w`×`h` pixels, preserving aspect (Lanczos).
    Resize { w: u32, h: u32 },
}

impl EditOp {
    /// Apply this op to `img`, returning the transformed image.
    pub fn apply(self, img: DynamicImage) -> DynamicImage {
        match self {
            EditOp::RotateCw => img.rotate90(),
            EditOp::RotateCcw => img.rotate270(),
            EditOp::Rotate180 => img.rotate180(),
            EditOp::FlipH => img.fliph(),
            EditOp::FlipV => img.flipv(),
            EditOp::Grayscale => img.grayscale(),
            EditOp::Brightness(v) => img.brighten(v),
            EditOp::Contrast(v) => img.adjust_contrast(v as f32),
            EditOp::Exposure(v) => adjust::exposure(&img, v),
            EditOp::Brilliance(v) => adjust::brilliance(&img, v),
            EditOp::Highlights(v) => adjust::tone(&img, v, adjust::Region::High),
            EditOp::Midrange(v) => adjust::tone(&img, v, adjust::Region::Mid),
            EditOp::Shadows(v) => adjust::tone(&img, v, adjust::Region::Shadow),
            EditOp::Blackpoint(v) => adjust::blackpoint(&img, v),
            EditOp::Saturation(v) => adjust::saturation(&img, v),
            EditOp::Vibrance(v) => adjust::vibrance(&img, v),
            EditOp::Warmth(v) => adjust::warmth(&img, v),
            EditOp::Tint(v) => adjust::tint(&img, v),
            EditOp::Definition(v) => adjust::definition(&img, v),
            EditOp::Sharpen(v) => adjust::sharpen(&img, v),
            EditOp::NoiseReduction(v) => adjust::denoise(&img, v),
            EditOp::AutoEnhance => adjust::auto_enhance(&img),
            EditOp::Straighten(tenths) => straighten(&img, tenths as f32 / 10.0),
            EditOp::Vignette(v) => adjust::vignette(&img, v),
            EditOp::Levels { black, white, gamma } => adjust::levels(&img, black, white, gamma as f32 / 100.0),
            EditOp::CropSquare => centered_aspect(&img, 1, 1),
            EditOp::CropAspect { w, h } => centered_aspect(&img, w, h),
            EditOp::Crop { x, y, w, h } => {
                let (iw, ih) = (img.width() as f32, img.height() as f32);
                let cx = (x.clamp(0.0, 1.0) * iw) as u32;
                let cy = (y.clamp(0.0, 1.0) * ih) as u32;
                let cw = ((w.clamp(0.0, 1.0) * iw) as u32).max(1).min(img.width() - cx.min(img.width() - 1));
                let ch = ((h.clamp(0.0, 1.0) * ih) as u32).max(1).min(img.height() - cy.min(img.height() - 1));
                img.crop_imm(cx, cy, cw, ch)
            }
            EditOp::CropPx { w, h } => {
                let (iw, ih) = (img.width(), img.height());
                let cw = w.clamp(1, iw);
                let ch = h.clamp(1, ih);
                img.crop_imm((iw - cw) / 2, (ih - ch) / 2, cw, ch)
            }
            EditOp::Resize { w, h } => {
                img.resize(w.max(1), h.max(1), image::imageops::FilterType::Lanczos3)
            }
        }
    }

    /// A short human label for the status line / edit menu.
    pub fn label(self) -> String {
        match self {
            EditOp::RotateCw => "rotate ⟳".into(),
            EditOp::RotateCcw => "rotate ⟲".into(),
            EditOp::Rotate180 => "rotate 180°".into(),
            EditOp::FlipH => "flip H".into(),
            EditOp::FlipV => "flip V".into(),
            EditOp::Grayscale => "grayscale".into(),
            EditOp::Brightness(_) => "brightness".into(),
            EditOp::Contrast(_) => "contrast".into(),
            EditOp::Exposure(_) => "exposure".into(),
            EditOp::Brilliance(_) => "brilliance".into(),
            EditOp::Highlights(_) => "highlights".into(),
            EditOp::Midrange(_) => "midrange".into(),
            EditOp::Shadows(_) => "shadows".into(),
            EditOp::Blackpoint(_) => "black point".into(),
            EditOp::Saturation(_) => "saturation".into(),
            EditOp::Vibrance(_) => "vibrance".into(),
            EditOp::Warmth(_) => "warmth".into(),
            EditOp::Tint(_) => "tint".into(),
            EditOp::Definition(_) => "definition".into(),
            EditOp::Sharpen(_) => "sharpen".into(),
            EditOp::NoiseReduction(_) => "noise reduction".into(),
            EditOp::AutoEnhance => "auto-enhance".into(),
            EditOp::Straighten(t) => format!("straighten {:.1}°", t as f32 / 10.0),
            EditOp::Vignette(_) => "vignette".into(),
            EditOp::Levels { black, white, gamma } => {
                format!("levels {black}/{white}·γ{:.2}", gamma as f32 / 100.0)
            }
            EditOp::CropSquare => "crop 1:1".into(),
            EditOp::CropAspect { w, h } => format!("crop {w}:{h}"),
            EditOp::Crop { .. } => "crop (free-form)".into(),
            EditOp::CropPx { w, h } => format!("crop {w}×{h}px"),
            EditOp::Resize { w, h } => format!("resize ≤{w}×{h}px"),
        }
    }

    /// Serialise to an `album.hjson` edit-log entry.
    pub fn to_entry(self) -> EditEntry {
        let mut params = std::collections::HashMap::new();
        let op = match self {
            EditOp::RotateCw => "rotate_cw",
            EditOp::RotateCcw => "rotate_ccw",
            EditOp::Rotate180 => "rotate_180",
            EditOp::FlipH => "flip_h",
            EditOp::FlipV => "flip_v",
            EditOp::Grayscale => "grayscale",
            EditOp::Brightness(v) => {
                params.insert("value".into(), serde_json::json!(v));
                "brightness"
            }
            EditOp::Contrast(v) => {
                params.insert("value".into(), serde_json::json!(v));
                "contrast"
            }
            EditOp::Exposure(v) => val_op(&mut params, v, "exposure"),
            EditOp::Brilliance(v) => val_op(&mut params, v, "brilliance"),
            EditOp::Highlights(v) => val_op(&mut params, v, "highlights"),
            EditOp::Midrange(v) => val_op(&mut params, v, "midrange"),
            EditOp::Shadows(v) => val_op(&mut params, v, "shadows"),
            EditOp::Blackpoint(v) => val_op(&mut params, v, "blackpoint"),
            EditOp::Saturation(v) => val_op(&mut params, v, "saturation"),
            EditOp::Vibrance(v) => val_op(&mut params, v, "vibrance"),
            EditOp::Warmth(v) => val_op(&mut params, v, "warmth"),
            EditOp::Tint(v) => val_op(&mut params, v, "tint"),
            EditOp::Definition(v) => val_op(&mut params, v, "definition"),
            EditOp::Sharpen(v) => val_op(&mut params, v, "sharpen"),
            EditOp::NoiseReduction(v) => val_op(&mut params, v, "noise_reduction"),
            EditOp::AutoEnhance => "auto_enhance",
            EditOp::Straighten(v) => val_op(&mut params, v, "straighten"),
            EditOp::Vignette(v) => val_op(&mut params, v, "vignette"),
            EditOp::Levels { black, white, gamma } => {
                params.insert("black".into(), serde_json::json!(black));
                params.insert("white".into(), serde_json::json!(white));
                params.insert("gamma".into(), serde_json::json!(gamma));
                "levels"
            }
            EditOp::CropSquare => "crop_square",
            EditOp::CropAspect { w, h } => {
                params.insert("w".into(), serde_json::json!(w));
                params.insert("h".into(), serde_json::json!(h));
                "crop_aspect"
            }
            EditOp::Crop { x, y, w, h } => {
                params.insert("x".into(), serde_json::json!(x));
                params.insert("y".into(), serde_json::json!(y));
                params.insert("w".into(), serde_json::json!(w));
                params.insert("h".into(), serde_json::json!(h));
                "crop"
            }
            EditOp::CropPx { w, h } => {
                params.insert("w".into(), serde_json::json!(w));
                params.insert("h".into(), serde_json::json!(h));
                "crop_px"
            }
            EditOp::Resize { w, h } => {
                params.insert("w".into(), serde_json::json!(w));
                params.insert("h".into(), serde_json::json!(h));
                "resize"
            }
        };
        EditEntry { op: op.to_string(), params, ts: None }
    }

    /// Parse a bare op tag into an op — the vocabulary the natural-language / free-form command
    /// pipeline maps onto. Directional adjustment verbs (`sharpen`, `warmer`, `brighter`, …) carry a
    /// sensible default amount so a single command makes a visible change; structural ops
    /// (`rotate_cw`, `grayscale`, `crop_square`, …) fall through to [`from_entry`].
    pub fn from_tag(tag: &str) -> Option<EditOp> {
        const S: i32 = 22; // default adjustment step for a one-shot command
        // `straighten:N` carries its angle in **degrees** in the tag (the NL pipeline emits it this
        // way); the op stores tenths-of-a-degree internally.
        if let Some(rest) = tag.strip_prefix("straighten:") {
            return rest.trim().parse::<f32>().ok().map(|d| EditOp::Straighten((d * 10.0).round() as i32));
        }
        // `levels:BLACK,WHITE,GAMMA` (e.g. levels:16,235,1.1) — gamma is a float, stored ×100.
        if let Some(rest) = tag.strip_prefix("levels:") {
            let parts: Vec<&str> = rest.split(&[',', '/', ' ']).filter(|s| !s.is_empty()).collect();
            if let [b, w, g] = parts.as_slice() {
                return Some(EditOp::Levels {
                    black: b.trim().parse().ok()?,
                    white: w.trim().parse().ok()?,
                    gamma: (g.trim().parse::<f32>().ok()? * 100.0).round() as i32,
                });
            }
            return None;
        }
        let op = match tag {
            "auto_enhance" | "enhance" | "auto" => EditOp::AutoEnhance,
            "vignette" | "vignette_dark" => EditOp::Vignette(30),
            "vignette_light" | "lighten_edges" => EditOp::Vignette(-30),
            "sharpen" => EditOp::Sharpen(S),
            "soften" | "blur" => EditOp::Sharpen(-S),
            "denoise" | "noise_reduction" | "reduce_noise" => EditOp::NoiseReduction(S + 8),
            "definition" | "clarity" => EditOp::Definition(S),
            "less_definition" => EditOp::Definition(-S),
            "brighter" | "brighten" => EditOp::Brightness(20),
            "darker" | "darken" => EditOp::Brightness(-20),
            "more_contrast" | "contrast_up" => EditOp::Contrast(18),
            "less_contrast" | "contrast_down" => EditOp::Contrast(-18),
            "exposure_up" | "overexpose" => EditOp::Exposure(S),
            "exposure_down" | "underexpose" => EditOp::Exposure(-S),
            "brilliance" | "brilliance_up" => EditOp::Brilliance(S),
            "brilliance_down" => EditOp::Brilliance(-S),
            "highlights_up" => EditOp::Highlights(S),
            "highlights_down" | "recover_highlights" => EditOp::Highlights(-S),
            "midrange_up" => EditOp::Midrange(S),
            "midrange_down" => EditOp::Midrange(-S),
            "shadows_up" | "lift_shadows" => EditOp::Shadows(S),
            "shadows_down" => EditOp::Shadows(-S),
            "blackpoint_up" | "crush_blacks" => EditOp::Blackpoint(S),
            "blackpoint_down" | "lift_blacks" => EditOp::Blackpoint(-S),
            "saturate" | "more_saturation" | "saturation_up" => EditOp::Saturation(S),
            "desaturate" | "less_saturation" | "saturation_down" => EditOp::Saturation(-S),
            "vibrant" | "vibrance_up" | "more_vibrance" => EditOp::Vibrance(S),
            "vibrance_down" | "less_vibrance" => EditOp::Vibrance(-S),
            "warmer" | "warmth_up" => EditOp::Warmth(S),
            "cooler" | "warmth_down" => EditOp::Warmth(-S),
            "tint_magenta" | "tint_up" => EditOp::Tint(S),
            "tint_green" | "tint_down" => EditOp::Tint(-S),
            _ => return Self::from_entry(&EditEntry {
                op: tag.to_string(),
                params: Default::default(),
                ts: None,
            }),
        };
        Some(op)
    }

    /// Parse an `album.hjson` edit-log entry back into an op (unknown ops → `None`, skipped on replay).
    pub fn from_entry(e: &EditEntry) -> Option<EditOp> {
        let val = || e.params.get("value").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let u = |k: &str| e.params.get(k).and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let fr = |k: &str| e.params.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let iv = |k: &str, d: i32| e.params.get(k).and_then(|v| v.as_i64()).unwrap_or(d as i64) as i32;
        Some(match e.op.as_str() {
            "rotate_cw" => EditOp::RotateCw,
            "rotate_ccw" => EditOp::RotateCcw,
            "rotate_180" => EditOp::Rotate180,
            "flip_h" => EditOp::FlipH,
            "flip_v" => EditOp::FlipV,
            "grayscale" => EditOp::Grayscale,
            "brightness" => EditOp::Brightness(val()),
            "contrast" => EditOp::Contrast(val()),
            "exposure" => EditOp::Exposure(val()),
            "brilliance" => EditOp::Brilliance(val()),
            "highlights" => EditOp::Highlights(val()),
            "midrange" => EditOp::Midrange(val()),
            "shadows" => EditOp::Shadows(val()),
            "blackpoint" => EditOp::Blackpoint(val()),
            "saturation" => EditOp::Saturation(val()),
            "vibrance" => EditOp::Vibrance(val()),
            "warmth" => EditOp::Warmth(val()),
            "tint" => EditOp::Tint(val()),
            "definition" => EditOp::Definition(val()),
            "sharpen" => EditOp::Sharpen(val()),
            "noise_reduction" => EditOp::NoiseReduction(val()),
            "auto_enhance" => EditOp::AutoEnhance,
            "straighten" => EditOp::Straighten(val()),
            "vignette" => EditOp::Vignette(val()),
            "levels" => EditOp::Levels { black: iv("black", 0), white: iv("white", 255), gamma: iv("gamma", 100) },
            "crop_square" => EditOp::CropSquare,
            "crop_aspect" => EditOp::CropAspect { w: u("w"), h: u("h") },
            "crop" => EditOp::Crop { x: fr("x"), y: fr("y"), w: fr("w"), h: fr("h") },
            "crop_px" => EditOp::CropPx { w: u("w"), h: u("h") },
            "resize" => EditOp::Resize { w: u("w"), h: u("h") },
            _ => return None,
        })
    }
}

/// Serialise a value-carrying op: insert its `value` param and return the op tag.
fn val_op(params: &mut std::collections::HashMap<String, serde_json::Value>, v: i32, tag: &'static str) -> &'static str {
    params.insert("value".into(), serde_json::json!(v));
    tag
}

/// Tonal / colour adjustment pixel math (RFC PHOTOS-1 Phase 3, extended set). Everything works on
/// normalised sRGB channels in `[0,1]`; `amount` is roughly `[-100, 100]` mapped to `t = amount/100`.
mod adjust {
    use image::{DynamicImage, Rgb, RgbImage};

    /// Which tonal band a region adjustment targets.
    #[derive(Clone, Copy)]
    pub enum Region {
        Shadow,
        Mid,
        High,
    }

    fn luma(r: f32, g: f32, b: f32) -> f32 {
        0.299 * r + 0.587 * g + 0.114 * b
    }
    fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
    fn enc(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    /// Map every pixel through `f(r,g,b) -> [r,g,b]` (normalised).
    fn map_rgb(img: &DynamicImage, f: impl Fn(f32, f32, f32) -> [f32; 3]) -> DynamicImage {
        let mut rgb = img.to_rgb8();
        for p in rgb.pixels_mut() {
            let o = f(p.0[0] as f32 / 255.0, p.0[1] as f32 / 255.0, p.0[2] as f32 / 255.0);
            p.0 = [enc(o[0]), enc(o[1]), enc(o[2])];
        }
        DynamicImage::ImageRgb8(rgb)
    }

    pub fn exposure(img: &DynamicImage, amount: i32) -> DynamicImage {
        let f = 2f32.powf(amount as f32 / 100.0); // ±1 stop at the extremes
        map_rgb(img, |r, g, b| [r * f, g * f, b * f])
    }

    /// Brighten/darken a tonal band, weighting the shift by the band's luma membership.
    pub fn tone(img: &DynamicImage, amount: i32, region: Region) -> DynamicImage {
        let t = amount as f32 / 100.0;
        map_rgb(img, |r, g, b| {
            let y = luma(r, g, b);
            let w = match region {
                Region::Shadow => 1.0 - smoothstep(0.0, 0.5, y),
                Region::High => smoothstep(0.5, 1.0, y),
                Region::Mid => (1.0 - (y - 0.5).abs() * 2.0).clamp(0.0, 1.0),
            };
            let d = t * 0.5 * w;
            [r + d, g + d, b + d]
        })
    }

    /// Adaptive richness: lift shadows & mids, gently recover highlights.
    pub fn brilliance(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = amount as f32 / 100.0;
        map_rgb(img, |r, g, b| {
            let y = luma(r, g, b);
            let wsh = 1.0 - smoothstep(0.0, 0.5, y);
            let wmid = (1.0 - (y - 0.5).abs() * 2.0).clamp(0.0, 1.0);
            let whi = smoothstep(0.5, 1.0, y);
            let d = t * (0.45 * wsh + 0.2 * wmid - 0.25 * whi);
            [r + d, g + d, b + d]
        })
    }

    /// Remap the black point: positive crushes blacks, negative lifts them.
    pub fn blackpoint(img: &DynamicImage, amount: i32) -> DynamicImage {
        let b0 = (amount as f32 / 100.0) * 0.5; // [-0.5, 0.5]
        let denom = (1.0 - b0).max(0.05);
        map_rgb(img, |r, g, b| [(r - b0) / denom, (g - b0) / denom, (b - b0) / denom])
    }

    pub fn saturation(img: &DynamicImage, amount: i32) -> DynamicImage {
        let f = 1.0 + amount as f32 / 100.0; // -100 → greyscale, +100 → 2×
        map_rgb(img, |r, g, b| {
            let y = luma(r, g, b);
            [y + (r - y) * f, y + (g - y) * f, y + (b - y) * f]
        })
    }

    /// Saturation weighted toward the least-saturated pixels (protects already-vivid colour).
    pub fn vibrance(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = amount as f32 / 100.0;
        map_rgb(img, |r, g, b| {
            let y = luma(r, g, b);
            let s = r.max(g).max(b) - r.min(g).min(b); // current saturation 0..1
            let f = 1.0 + t * (1.0 - s);
            [y + (r - y) * f, y + (g - y) * f, y + (b - y) * f]
        })
    }

    /// Colour temperature: warmer boosts red / cuts blue, cooler the reverse.
    pub fn warmth(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = amount as f32 / 100.0;
        map_rgb(img, |r, g, b| [r * (1.0 + 0.2 * t), g, b * (1.0 - 0.2 * t)])
    }

    /// Green ↔ magenta balance.
    pub fn tint(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = amount as f32 / 100.0;
        map_rgb(img, |r, g, b| [r * (1.0 + 0.1 * t), g * (1.0 - 0.2 * t), b * (1.0 + 0.1 * t)])
    }

    /// Blur `img` and combine base + blurred per pixel via `f(base, blurred, luma)`.
    fn spatial(img: &DynamicImage, sigma: f32, f: impl Fn([f32; 3], [f32; 3], f32) -> [f32; 3]) -> DynamicImage {
        let rgb = img.to_rgb8();
        let blurred = image::imageops::blur(&rgb, sigma);
        let (w, h) = (rgb.width(), rgb.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let bp = rgb.get_pixel(x, y).0;
                let bl = blurred.get_pixel(x, y).0;
                let base = [bp[0] as f32 / 255.0, bp[1] as f32 / 255.0, bp[2] as f32 / 255.0];
                let blf = [bl[0] as f32 / 255.0, bl[1] as f32 / 255.0, bl[2] as f32 / 255.0];
                let yy = luma(base[0], base[1], base[2]);
                let o = f(base, blf, yy);
                out.put_pixel(x, y, Rgb([enc(o[0]), enc(o[1]), enc(o[2])]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Unsharp mask: positive sharpens, negative softens.
    pub fn sharpen(img: &DynamicImage, amount: i32) -> DynamicImage {
        let k = amount as f32 / 100.0 * 1.5;
        spatial(img, 1.2, |base, bl, _| std::array::from_fn(|i| base[i] + k * (base[i] - bl[i])))
    }

    /// Midtone local contrast (clarity): large-radius unsharp, weighted to the mids.
    pub fn definition(img: &DynamicImage, amount: i32) -> DynamicImage {
        let k = amount as f32 / 100.0;
        spatial(img, 6.0, |base, bl, y| {
            let wmid = (1.0 - (y - 0.5).abs() * 2.0).clamp(0.0, 1.0);
            std::array::from_fn(|i| base[i] + k * wmid * (base[i] - bl[i]))
        })
    }

    /// Edge-softening noise reduction: blend toward a small blur (positive only).
    pub fn denoise(img: &DynamicImage, amount: i32) -> DynamicImage {
        let s = (amount as f32 / 100.0).clamp(0.0, 1.0);
        spatial(img, 1.4, |base, bl, _| std::array::from_fn(|i| base[i] * (1.0 - s) + bl[i] * s))
    }

    /// Vignette: multiply each pixel by a radial falloff — positive `amount` darkens the frame edges
    /// (a smooth ramp from the centre out to the corners), negative lightens them.
    pub fn vignette(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = amount as f32 / 100.0;
        let mut rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as f32, rgb.height() as f32);
        let (cx, cy) = (w / 2.0, h / 2.0);
        let maxr = (cx * cx + cy * cy).sqrt().max(1.0);
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt() / maxr; // 0 at centre, 1 at the corners
            let f = (1.0 - t * smoothstep(0.4, 1.0, r)).clamp(0.0, 2.0);
            for c in 0..3 {
                p.0[c] = enc(p.0[c] as f32 / 255.0 * f);
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Levels: remap the input range `[black, white]` (0..255) to the full range with a midtone
    /// `gamma` (>1 brightens the mids). The classic levels tool — the core of a tone curve.
    pub fn levels(img: &DynamicImage, black: i32, white: i32, gamma: f32) -> DynamicImage {
        let b = (black as f32 / 255.0).clamp(0.0, 1.0);
        let w = (white as f32 / 255.0).clamp(0.0, 1.0);
        let span = (w - b).max(1e-3);
        let inv_g = 1.0 / gamma.clamp(0.05, 20.0);
        map_rgb(img, |r, g, bl| {
            let f = |c: f32| ((c - b) / span).clamp(0.0, 1.0).powf(inv_g);
            [f(r), f(g), f(bl)]
        })
    }

    /// Auto-enhance: stretch each channel between its 0.5 % / 99.5 % percentiles — auto levels plus
    /// auto colour balance (independent channel stretch neutralises casts) in one pass.
    pub fn auto_enhance(img: &DynamicImage) -> DynamicImage {
        let mut rgb = img.to_rgb8();
        let n = (rgb.width() * rgb.height()) as u64;
        if n == 0 {
            return img.clone();
        }
        let mut hist = [[0u64; 256]; 3];
        for p in rgb.pixels() {
            for c in 0..3 {
                hist[c][p.0[c] as usize] += 1;
            }
        }
        let clip = (n as f64 * 0.005) as u64; // ignore the extreme 0.5 % tails
        let (mut lo, mut hi) = ([0f32; 3], [255f32; 3]);
        for c in 0..3 {
            let (mut acc, mut l) = (0u64, 0usize);
            while l < 255 {
                acc += hist[c][l];
                if acc > clip {
                    break;
                }
                l += 1;
            }
            let (mut acc2, mut h) = (0u64, 255usize);
            while h > 0 {
                acc2 += hist[c][h];
                if acc2 > clip {
                    break;
                }
                h -= 1;
            }
            lo[c] = l as f32;
            hi[c] = (h.max(l + 1)) as f32;
        }
        for p in rgb.pixels_mut() {
            for c in 0..3 {
                let v = (p.0[c] as f32 - lo[c]) / (hi[c] - lo[c]);
                p.0[c] = enc(v);
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }
}

/// Largest axis-aligned rectangle (w, h) that fits inside a `w0`×`h0` rectangle rotated by `a`
/// radians — the classic `rotatedRectWithMaxArea`, used to auto-crop a straighten.
fn max_inner_rect(w0: f32, h0: f32, a: f32) -> (f32, f32) {
    if w0 <= 0.0 || h0 <= 0.0 {
        return (0.0, 0.0);
    }
    let (sin, cos) = (a.sin().abs(), a.cos().abs());
    let width_longer = w0 >= h0;
    let (long, short) = if width_longer { (w0, h0) } else { (h0, w0) };
    if short <= 2.0 * sin * cos * long || (sin - cos).abs() < 1e-10 {
        let x = 0.5 * short;
        let (a1, a2) = (sin.max(1e-6), cos.max(1e-6));
        if width_longer { (x / a1, x / a2) } else { (x / a2, x / a1) }
    } else {
        let cos2 = cos * cos - sin * sin;
        (((w0 * cos - h0 * sin) / cos2).abs(), ((h0 * cos - w0 * sin) / cos2).abs())
    }
}

/// Bilinear sample of `src` at (possibly fractional) `(x, y)`, edge-clamped.
fn bilinear(src: &image::RgbImage, x: f32, y: f32) -> image::Rgb<u8> {
    let (w, h) = (src.width() as i32, src.height() as i32);
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |xx: i32, yy: i32| src.get_pixel(xx.clamp(0, w - 1) as u32, yy.clamp(0, h - 1) as u32).0;
    let (p00, p10, p01, p11) = (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1));
    let mut o = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        o[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
    }
    image::Rgb(o)
}

/// Rotate `img` by `degrees` about its centre (bilinear), auto-cropped to the largest inner rect so
/// there are no empty corners. Positive = counter-clockwise.
fn straighten(img: &DynamicImage, degrees: f32) -> DynamicImage {
    if degrees.abs() < 1e-3 {
        return img.clone();
    }
    let src = img.to_rgb8();
    let (w, h) = (src.width() as f32, src.height() as f32);
    let a = degrees.to_radians();
    let (sin, cos) = (a.sin(), a.cos());
    let (cw, ch) = max_inner_rect(w, h, a);
    let (ow, oh) = ((cw.floor() as u32).max(1).min(src.width()), (ch.floor() as u32).max(1).min(src.height()));
    let (cx, cy) = (w / 2.0, h / 2.0);
    let mut out = image::RgbImage::new(ow, oh);
    for oy in 0..oh {
        let ry = oy as f32 - oh as f32 / 2.0;
        for ox in 0..ow {
            let rx = ox as f32 - ow as f32 / 2.0;
            // Source point for this rotated-frame offset: R(-a)·(rx, ry) about the source centre.
            let sx = cx + rx * cos + ry * sin;
            let sy = cy - rx * sin + ry * cos;
            out.put_pixel(ox, oy, bilinear(&src, sx, sy));
        }
    }
    DynamicImage::ImageRgb8(out)
}

/// Largest centered rectangle of `img` with aspect ratio `aw:ah`.
fn centered_aspect(img: &DynamicImage, aw: u32, ah: u32) -> DynamicImage {
    let (iw, ih) = (img.width(), img.height());
    if aw == 0 || ah == 0 {
        return img.clone();
    }
    let target = aw as f64 / ah as f64;
    let (cw, ch) = if (iw as f64 / ih as f64) > target {
        (((ih as f64 * target).round() as u32).min(iw), ih) // too wide → cap width
    } else {
        (iw, ((iw as f64 / target).round() as u32).min(ih)) // too tall → cap height
    };
    let (cw, ch) = (cw.max(1), ch.max(1));
    img.crop_imm((iw - cw) / 2, (ih - ch) / 2, cw, ch)
}

/// Replay `ops` over a pristine `original`, returning the fully-edited image.
pub fn replay(original: &DynamicImage, ops: &[EditOp]) -> DynamicImage {
    let mut img = original.clone();
    for op in ops {
        img = op.apply(img);
    }
    img
}

fn backup_dir(album: &Path) -> PathBuf {
    album.join(".plakat_edits")
}

/// Path of the pristine backup for `filename` in `album` (hidden `.plakat_edits/`).
pub fn backup_path(album: &Path, filename: &str) -> PathBuf {
    backup_dir(album).join(filename)
}

/// Copy the current file into the hidden backup, once, before the first edit.
pub fn ensure_backup(album: &Path, filename: &str) -> Result<()> {
    let bak = backup_path(album, filename);
    if !bak.exists() {
        std::fs::create_dir_all(backup_dir(album))?;
        std::fs::copy(album.join(filename), &bak)
            .with_context(|| format!("backing up {filename}"))?;
    }
    Ok(())
}

/// Re-derive the visible file from the pristine backup + the full `ops` list. An empty `ops`
/// restores the original and removes the backup (a full revert).
pub fn rebuild_file(album: &Path, filename: &str, ops: &[EditOp]) -> Result<()> {
    let bak = backup_path(album, filename);
    let target = album.join(filename);
    if ops.is_empty() {
        if bak.exists() {
            std::fs::copy(&bak, &target).with_context(|| format!("restoring {filename}"))?;
            let _ = std::fs::remove_file(&bak);
        }
        return Ok(());
    }
    let original = image::open(&bak).with_context(|| format!("reading backup for {filename}"))?;
    let out = replay(&original, ops);
    out.save(&target).with_context(|| format!("writing edited {filename}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn img(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |x, y| Rgb([x as u8, y as u8, 0])))
    }

    #[test]
    fn rotate_swaps_dimensions_and_is_reversible() {
        let src = img(6, 4);
        let cw = EditOp::RotateCw.apply(src.clone());
        assert_eq!((cw.width(), cw.height()), (4, 6));
        // 4× rotate-cw returns to the original pixels.
        let round = EditOp::RotateCw.apply(EditOp::RotateCw.apply(EditOp::RotateCw.apply(cw)));
        assert_eq!(round.to_rgb8(), src.to_rgb8());
    }

    #[test]
    fn crop_square_is_centered_square() {
        let out = EditOp::CropSquare.apply(img(10, 4));
        assert_eq!((out.width(), out.height()), (4, 4));
    }

    #[test]
    fn entry_roundtrips_every_op() {
        for op in [
            EditOp::RotateCw, EditOp::RotateCcw, EditOp::Rotate180, EditOp::FlipH, EditOp::FlipV,
            EditOp::Grayscale, EditOp::Brightness(15), EditOp::Contrast(-10), EditOp::CropSquare,
            EditOp::CropAspect { w: 16, h: 9 },
            EditOp::Crop { x: 0.1, y: 0.2, w: 0.5, h: 0.4 },
            EditOp::CropPx { w: 800, h: 600 },
            EditOp::Resize { w: 1024, h: 768 },
            EditOp::Exposure(20), EditOp::Brilliance(-15), EditOp::Highlights(30),
            EditOp::Midrange(-20), EditOp::Shadows(25), EditOp::Blackpoint(10),
            EditOp::Saturation(-40), EditOp::Vibrance(35), EditOp::Warmth(15),
            EditOp::Tint(-25), EditOp::Definition(20), EditOp::Sharpen(-22),
            EditOp::NoiseReduction(30), EditOp::Vignette(40),
            EditOp::Levels { black: 16, white: 235, gamma: 110 },
        ] {
            assert_eq!(EditOp::from_entry(&op.to_entry()), Some(op));
        }
        // Unknown op → None (skipped, not a crash).
        let unknown = EditEntry { op: "warp_drive".into(), params: Default::default(), ts: None };
        assert_eq!(EditOp::from_entry(&unknown), None);
    }

    #[test]
    fn aspect_and_freeform_crop_dimensions() {
        // 16:9 centered crop of a 4:3 image → full width, reduced height.
        let out = EditOp::CropAspect { w: 16, h: 9 }.apply(img(400, 300));
        assert_eq!(out.width(), 400);
        assert_eq!(out.height(), 225); // 400 * 9/16
        // Free-form: middle 50%×40% of a 200×100 image.
        let ff = EditOp::Crop { x: 0.25, y: 0.3, w: 0.5, h: 0.4 }.apply(img(200, 100));
        assert_eq!((ff.width(), ff.height()), (100, 40));
        // Exact centered pixel crop (clamped to the image).
        let cp = EditOp::CropPx { w: 120, h: 80 }.apply(img(400, 300));
        assert_eq!((cp.width(), cp.height()), (120, 80));
        // Resize fits within the box preserving aspect: 400×300 into 200×200 → 200×150.
        let rz = EditOp::Resize { w: 200, h: 200 }.apply(img(400, 300));
        assert_eq!((rz.width(), rz.height()), (200, 150));
    }

    #[test]
    fn adjustments_move_pixels_the_expected_way() {
        // A flat mid-grey 8×8 image: exposure up brightens, saturation -100 keeps grey, warmth up
        // pushes red above blue.
        let grey = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([128, 128, 128])));
        let up = EditOp::Exposure(50).apply(grey.clone()).to_rgb8();
        assert!(up.get_pixel(4, 4).0[0] > 128, "exposure up should brighten");

        let desat = EditOp::Saturation(-100).apply(grey.clone()).to_rgb8().get_pixel(4, 4).0;
        assert!(desat[0].abs_diff(desat[2]) <= 1, "greyscale stays neutral");

        let warm = EditOp::Warmth(60).apply(grey.clone()).to_rgb8().get_pixel(4, 4).0;
        assert!(warm[0] > warm[2], "warmth up: red over blue");

        // Sharpen touches an edge: a black/white split should get a stronger contrast at the seam.
        let mut edge = ImageBuffer::from_pixel(8, 8, Rgb([120u8, 120, 120]));
        for y in 0..8 {
            for x in 4..8 {
                edge.put_pixel(x, y, Rgb([160, 160, 160]));
            }
        }
        let sharp = EditOp::Sharpen(80).apply(DynamicImage::ImageRgb8(edge.clone())).to_rgb8();
        let base_lo = 120i32;
        assert!((sharp.get_pixel(3, 4).0[0] as i32) <= base_lo, "sharpen darkens the dark side of the edge");
    }

    #[test]
    fn from_tag_directional_verbs() {
        assert_eq!(EditOp::from_tag("sharpen"), Some(EditOp::Sharpen(22)));
        assert_eq!(EditOp::from_tag("warmer"), Some(EditOp::Warmth(22)));
        assert_eq!(EditOp::from_tag("desaturate"), Some(EditOp::Saturation(-22)));
        assert_eq!(EditOp::from_tag("brighter"), Some(EditOp::Brightness(20)));
        assert_eq!(EditOp::from_tag("auto"), Some(EditOp::AutoEnhance));
        assert_eq!(EditOp::from_tag("vignette"), Some(EditOp::Vignette(30)));
        assert_eq!(EditOp::from_tag("vignette_light"), Some(EditOp::Vignette(-30)));
        assert_eq!(
            EditOp::from_tag("levels:16,235,1.1"),
            Some(EditOp::Levels { black: 16, white: 235, gamma: 110 })
        );
        // Tag angle is in degrees; the op stores tenths (5° → 50, 2.5° → 25).
        assert_eq!(EditOp::from_tag("straighten:5"), Some(EditOp::Straighten(50)));
        assert_eq!(EditOp::from_tag("straighten:-2.5"), Some(EditOp::Straighten(-25)));
        // Structural tags still resolve through from_entry.
        assert_eq!(EditOp::from_tag("grayscale"), Some(EditOp::Grayscale));
        assert_eq!(EditOp::from_tag("warp_drive"), None);
    }

    #[test]
    fn auto_enhance_stretches_a_flat_image() {
        // A low-contrast gradient (values 80..120) should stretch toward the full 0..255 range.
        let src = DynamicImage::ImageRgb8(ImageBuffer::from_fn(40, 4, |x, _| {
            Rgb([(80 + x) as u8, (80 + x) as u8, (80 + x) as u8])
        }));
        let out = EditOp::AutoEnhance.apply(src).to_rgb8();
        assert!(out.get_pixel(0, 0).0[0] < 20, "darkest stretched down");
        assert!(out.get_pixel(39, 0).0[0] > 235, "brightest stretched up");
    }

    #[test]
    fn straighten_autocrops_and_stays_valid() {
        let out = EditOp::Straighten(100).apply(img(200, 120)); // 10.0°
        // Auto-crop → smaller than the source, non-empty, no all-black corners from the rotation.
        assert!(out.width() > 0 && out.height() > 0);
        assert!(out.width() < 200 && out.height() < 120);
        let noop = EditOp::Straighten(0).apply(img(20, 20));
        assert_eq!((noop.width(), noop.height()), (20, 20));
    }

    #[test]
    fn vignette_darkens_corners_not_centre() {
        let flat = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(60, 60, Rgb([200, 200, 200])));
        let out = EditOp::Vignette(60).apply(flat).to_rgb8();
        let centre = out.get_pixel(30, 30).0[0];
        let corner = out.get_pixel(0, 0).0[0];
        assert!(centre >= 198, "centre stays bright ({centre})");
        assert!(corner < 160, "corner darkened ({corner})");
    }

    #[test]
    fn levels_black_and_white_points_clip() {
        // A 0..255 gradient. Black=64, white=192, gamma=1 → below 64 clips to 0, above 192 to 255.
        let grad = DynamicImage::ImageRgb8(ImageBuffer::from_fn(256, 1, |x, _| Rgb([x as u8; 3])));
        let out = EditOp::Levels { black: 64, white: 192, gamma: 100 }.apply(grad).to_rgb8();
        assert_eq!(out.get_pixel(30, 0).0[0], 0, "below black → 0");
        assert_eq!(out.get_pixel(220, 0).0[0], 255, "above white → 255");
        let mid = out.get_pixel(128, 0).0[0];
        assert!((120..=135).contains(&mid), "midpoint maps to ~mid ({mid})");
    }

    #[test]
    fn backup_replay_and_full_revert() {
        let dir = std::env::temp_dir().join(format!("plakat-edit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = "p.png";
        img(8, 6).save(dir.join(name)).unwrap();

        // First edit: back up + rebuild (rotate cw → 6×8).
        ensure_backup(&dir, name).unwrap();
        rebuild_file(&dir, name, &[EditOp::RotateCw]).unwrap();
        let after = image::open(dir.join(name)).unwrap();
        assert_eq!((after.width(), after.height()), (6, 8));
        assert!(backup_path(&dir, name).exists());

        // Revert (empty ops): original restored, backup gone.
        rebuild_file(&dir, name, &[]).unwrap();
        let back = image::open(dir.join(name)).unwrap();
        assert_eq!((back.width(), back.height()), (8, 6));
        assert!(!backup_path(&dir, name).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
