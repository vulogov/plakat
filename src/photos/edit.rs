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
    /// Keystone / perspective: correct converging lines by a trapezoidal warp. `axis` 0 = vertical
    /// (converging verticals), 1 = horizontal; `amount` −100..100.
    Keystone { axis: i32, amount: i32 },
    /// Vignette: darken (positive) / lighten (negative) the frame edges radially.
    Vignette(i32),
    /// Levels: map input `[black, white]` (0..255) to full range with a midtone `gamma` (×100:
    /// 100 = 1.00). `gamma > 1` brightens the mids.
    Levels { black: i32, white: i32, gamma: i32 },
    /// Dehaze: cut the washed-out veil (contrast + saturation lift).
    Dehaze(i32),
    /// Rotate all hues by `degrees`.
    HueRotate(i32),
    /// Split-tone: warm highlights / cool shadows (positive), or the inverse (negative).
    SplitTone(i32),
    /// Selective colour: change the saturation of pixels near `hue` (degrees) by `sat` (−100..100).
    SelectiveColor { hue: i32, sat: i32 },
    /// Film grain: add deterministic monochrome noise (positive only).
    Grain(i32),
    /// Median despeckle: blend toward a 3×3 per-channel median (positive only).
    Despeckle(i32),
    /// Graduated ND: darken (positive) / lighten (negative) a linear gradient from an edge
    /// (`dir`: 0 top, 1 bottom, 2 left, 3 right — strongest at that edge, fading to the middle).
    GradND { dir: i32, strength: i32 },
    /// Radial dodge/burn: darken the centre (positive = burn) or lighten it (negative = dodge).
    Radial(i32),
    /// Tone curve: output values (0..255) at the five input points 0/64/128/192/255, linearly
    /// interpolated into a 256-entry LUT applied to R, G, B.
    Curve { pts: [i32; 5] },
    /// CLAHE — contrast-limited adaptive histogram equalization, blended by `amount` (0..100).
    Clahe(i32),
    /// Invert (photo negative).
    Invert,
    /// Sepia tone.
    Sepia,
    /// Duotone: map luma to a two-colour gradient (deep indigo → warm cream).
    Duotone,
    /// Posterize: reduce the number of tonal levels. `strength` 0 = none … 100 = 2 levels.
    Posterize(i32),
    /// Solarize: invert tones above a threshold. `strength` 0 = none … 100 = classic (mid).
    Solarize(i32),
    /// Threshold to black & white at luma `level` (0..255).
    Threshold(i32),
    /// Oil-paint / oilify filter — `style` 1..10 selects a brush/palette preset; `strength` 0..100
    /// blends with the original.
    OilPaint { style: i32, strength: i32 },
    /// Pencil-sketch (grayscale colour-dodge of a blurred inverse); `strength` 0..100.
    PencilSketch(i32),
    /// Cartoon / comic (colour quantise + dark ink edges); `strength` 0..100.
    Cartoon(i32),
    /// Watercolour — `style` 1..10 selects a wash/palette preset; `strength` 0..100.
    Watercolor { style: i32, strength: i32 },
    /// Ink / traditional painting — `style`: 1 European ink · 2 Japanese sumi-e · 3 Chinese wash ·
    /// 4 Russian icon (tempera); `strength` 0..100.
    Ink { style: i32, strength: i32 },
    /// Emboss (directional gradient, grey relief); `strength` 0..100.
    Emboss(i32),
    /// Gaussian blur / soft focus; `strength` 0..100 → blur radius.
    Blur(i32),
    /// Bloom / glow (screen-blend the blurred highlights back); `strength` 0..100.
    Bloom(i32),
    /// Charcoal drawing (dark edge strokes on paper); `strength` 0..100.
    Charcoal(i32),
    /// Halftone / newsprint dots; `strength` 0..100 blends with the original.
    Halftone(i32),
    /// False colour — `style`: 1 thermal · 2 infrared · 3 night-vision; `strength` 0..100.
    FalseColor { style: i32, strength: i32 },
    /// Pixelate / mosaic. `strength` 0 = none … 100 = large blocks.
    Pixelate(i32),
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
            EditOp::Keystone { axis, amount } => keystone(&img, axis, amount),
            EditOp::Vignette(v) => adjust::vignette(&img, v),
            EditOp::Levels { black, white, gamma } => adjust::levels(&img, black, white, gamma as f32 / 100.0),
            EditOp::Dehaze(v) => adjust::dehaze(&img, v),
            EditOp::HueRotate(v) => adjust::hue_rotate(&img, v),
            EditOp::SplitTone(v) => adjust::split_tone(&img, v),
            EditOp::SelectiveColor { hue, sat } => adjust::selective_color(&img, hue, sat),
            EditOp::Grain(v) => adjust::grain(&img, v),
            EditOp::Despeckle(v) => adjust::despeckle(&img, v),
            EditOp::GradND { dir, strength } => adjust::grad_nd(&img, dir, strength),
            EditOp::Radial(v) => adjust::radial(&img, v),
            EditOp::Curve { pts } => adjust::curve(&img, pts),
            EditOp::Clahe(v) => adjust::clahe(&img, v),
            EditOp::Invert => adjust::invert(&img),
            EditOp::Sepia => adjust::sepia(&img),
            EditOp::Duotone => adjust::duotone(&img),
            EditOp::Posterize(v) => adjust::posterize(&img, v),
            EditOp::Solarize(v) => adjust::solarize(&img, v),
            EditOp::Threshold(v) => adjust::threshold(&img, v),
            EditOp::OilPaint { style, strength } => {
                let f = adjust::oil_paint(&img, style);
                adjust::blend(&img, f, strength)
            }
            EditOp::PencilSketch(s) => {
                let f = adjust::pencil_sketch(&img);
                adjust::blend(&img, f, s)
            }
            EditOp::Cartoon(s) => {
                let f = adjust::cartoon(&img);
                adjust::blend(&img, f, s)
            }
            EditOp::Watercolor { style, strength } => {
                let f = adjust::watercolor(&img, style);
                adjust::blend(&img, f, strength)
            }
            EditOp::Ink { style, strength } => {
                let f = adjust::ink(&img, style);
                adjust::blend(&img, f, strength)
            }
            EditOp::Emboss(s) => {
                let f = adjust::emboss(&img);
                adjust::blend(&img, f, s)
            }
            EditOp::Blur(s) => adjust::gaussian(&img, s),
            EditOp::Bloom(s) => adjust::bloom(&img, s),
            EditOp::Charcoal(s) => {
                let f = adjust::charcoal(&img);
                adjust::blend(&img, f, s)
            }
            EditOp::Halftone(s) => {
                let f = adjust::halftone(&img);
                adjust::blend(&img, f, s)
            }
            EditOp::FalseColor { style, strength } => {
                let f = adjust::false_color(&img, style);
                adjust::blend(&img, f, strength)
            }
            EditOp::Pixelate(v) => adjust::pixelate(&img, v),
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

    /// The single scalar amount for slider-style adjustments (`None` for structural / multi-param
    /// ops, which use a prompt or their own mode instead).
    pub fn scalar(self) -> Option<i32> {
        Some(match self {
            EditOp::Brightness(v) | EditOp::Contrast(v) | EditOp::Exposure(v) | EditOp::Brilliance(v)
            | EditOp::Highlights(v) | EditOp::Midrange(v) | EditOp::Shadows(v) | EditOp::Blackpoint(v)
            | EditOp::Saturation(v) | EditOp::Vibrance(v) | EditOp::Warmth(v) | EditOp::Tint(v)
            | EditOp::Definition(v) | EditOp::Sharpen(v) | EditOp::NoiseReduction(v) | EditOp::Dehaze(v)
            | EditOp::HueRotate(v) | EditOp::SplitTone(v) | EditOp::Vignette(v) | EditOp::Grain(v)
            | EditOp::Despeckle(v) | EditOp::Radial(v) | EditOp::Clahe(v) | EditOp::Posterize(v)
            | EditOp::Solarize(v) | EditOp::Pixelate(v) | EditOp::PencilSketch(v) | EditOp::Cartoon(v)
            | EditOp::Emboss(v) | EditOp::Blur(v) | EditOp::Bloom(v) | EditOp::Charcoal(v)
            | EditOp::Halftone(v) => v,
            EditOp::GradND { strength, .. }
            | EditOp::OilPaint { strength, .. }
            | EditOp::Watercolor { strength, .. }
            | EditOp::Ink { strength, .. }
            | EditOp::FalseColor { strength, .. } => strength,
            EditOp::Keystone { amount, .. } => amount,
            _ => return None,
        })
    }

    /// Replace a scalar op's amount (identity for non-scalar ops).
    pub fn with_scalar(self, v: i32) -> EditOp {
        match self {
            EditOp::Brightness(_) => EditOp::Brightness(v),
            EditOp::Contrast(_) => EditOp::Contrast(v),
            EditOp::Exposure(_) => EditOp::Exposure(v),
            EditOp::Brilliance(_) => EditOp::Brilliance(v),
            EditOp::Highlights(_) => EditOp::Highlights(v),
            EditOp::Midrange(_) => EditOp::Midrange(v),
            EditOp::Shadows(_) => EditOp::Shadows(v),
            EditOp::Blackpoint(_) => EditOp::Blackpoint(v),
            EditOp::Saturation(_) => EditOp::Saturation(v),
            EditOp::Vibrance(_) => EditOp::Vibrance(v),
            EditOp::Warmth(_) => EditOp::Warmth(v),
            EditOp::Tint(_) => EditOp::Tint(v),
            EditOp::Definition(_) => EditOp::Definition(v),
            EditOp::Sharpen(_) => EditOp::Sharpen(v),
            EditOp::NoiseReduction(_) => EditOp::NoiseReduction(v),
            EditOp::Dehaze(_) => EditOp::Dehaze(v),
            EditOp::HueRotate(_) => EditOp::HueRotate(v),
            EditOp::SplitTone(_) => EditOp::SplitTone(v),
            EditOp::Vignette(_) => EditOp::Vignette(v),
            EditOp::Grain(_) => EditOp::Grain(v),
            EditOp::Despeckle(_) => EditOp::Despeckle(v),
            EditOp::Radial(_) => EditOp::Radial(v),
            EditOp::Clahe(_) => EditOp::Clahe(v),
            EditOp::Posterize(_) => EditOp::Posterize(v),
            EditOp::Solarize(_) => EditOp::Solarize(v),
            EditOp::Pixelate(_) => EditOp::Pixelate(v),
            EditOp::GradND { dir, .. } => EditOp::GradND { dir, strength: v },
            EditOp::Keystone { axis, .. } => EditOp::Keystone { axis, amount: v },
            EditOp::PencilSketch(_) => EditOp::PencilSketch(v),
            EditOp::Cartoon(_) => EditOp::Cartoon(v),
            EditOp::Emboss(_) => EditOp::Emboss(v),
            EditOp::Blur(_) => EditOp::Blur(v),
            EditOp::Bloom(_) => EditOp::Bloom(v),
            EditOp::Charcoal(_) => EditOp::Charcoal(v),
            EditOp::Halftone(_) => EditOp::Halftone(v),
            EditOp::OilPaint { style, .. } => EditOp::OilPaint { style, strength: v },
            EditOp::Watercolor { style, .. } => EditOp::Watercolor { style, strength: v },
            EditOp::Ink { style, .. } => EditOp::Ink { style, strength: v },
            EditOp::FalseColor { style, .. } => EditOp::FalseColor { style, strength: v },
            other => other,
        }
    }

    /// `(min, max, step)` for the slider — most tonal/colour ops are bipolar ±100; hue is ±180;
    /// the effect-only ops (grain/denoise/despeckle/dehaze) are one-sided 0..100.
    pub fn scalar_range(self) -> (i32, i32, i32) {
        match self {
            EditOp::HueRotate(_) => (-180, 180, 5),
            EditOp::NoiseReduction(_) | EditOp::Grain(_) | EditOp::Despeckle(_) | EditOp::Dehaze(_)
            | EditOp::Clahe(_) | EditOp::Posterize(_) | EditOp::Solarize(_) | EditOp::Pixelate(_)
            | EditOp::PencilSketch(_) | EditOp::Cartoon(_) | EditOp::Emboss(_) | EditOp::Blur(_)
            | EditOp::Bloom(_) | EditOp::Charcoal(_) | EditOp::Halftone(_) | EditOp::OilPaint { .. }
            | EditOp::Watercolor { .. } | EditOp::Ink { .. } | EditOp::FalseColor { .. } => (0, 100, 5),
            _ => (-100, 100, 5),
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
            EditOp::Keystone { axis, .. } => {
                format!("keystone {}", if axis == 0 { "vertical" } else { "horizontal" })
            }
            EditOp::Vignette(_) => "vignette".into(),
            EditOp::Levels { black, white, gamma } => {
                format!("levels {black}/{white}·γ{:.2}", gamma as f32 / 100.0)
            }
            EditOp::Dehaze(_) => "dehaze".into(),
            EditOp::HueRotate(d) => format!("hue {d:+}°"),
            EditOp::SplitTone(_) => "split-tone".into(),
            EditOp::SelectiveColor { hue, sat } => format!("selective colour {hue}°·{sat:+}"),
            EditOp::Grain(_) => "film grain".into(),
            EditOp::Despeckle(_) => "despeckle".into(),
            EditOp::GradND { dir, .. } => {
                format!("grad ND {}", ["top", "bottom", "left", "right"].get(dir as usize).unwrap_or(&"top"))
            }
            EditOp::Radial(_) => "radial dodge/burn".into(),
            EditOp::Curve { .. } => "curves".into(),
            EditOp::Clahe(_) => "CLAHE".into(),
            EditOp::Invert => "invert".into(),
            EditOp::Sepia => "sepia".into(),
            EditOp::Duotone => "duotone".into(),
            EditOp::Posterize(_) => "posterize".into(),
            EditOp::Solarize(_) => "solarize".into(),
            EditOp::Threshold(_) => "threshold".into(),
            EditOp::OilPaint { style, .. } => format!("oil paint {}", style.clamp(1, 10)),
            EditOp::PencilSketch(_) => "pencil sketch".into(),
            EditOp::Cartoon(_) => "cartoon".into(),
            EditOp::Watercolor { style, .. } => format!("watercolour {}", style.clamp(1, 10)),
            EditOp::Ink { style, .. } => format!("ink ({})", adjust::ink_name(style)),
            EditOp::Emboss(_) => "emboss".into(),
            EditOp::Blur(_) => "blur (soft focus)".into(),
            EditOp::Bloom(_) => "bloom / glow".into(),
            EditOp::Charcoal(_) => "charcoal".into(),
            EditOp::Halftone(_) => "halftone".into(),
            EditOp::FalseColor { style, .. } => format!("false colour ({})", adjust::false_color_name(style)),
            EditOp::Pixelate(_) => "pixelate".into(),
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
            EditOp::Keystone { axis, amount } => {
                params.insert("axis".into(), serde_json::json!(axis));
                params.insert("amount".into(), serde_json::json!(amount));
                "keystone"
            }
            EditOp::Vignette(v) => val_op(&mut params, v, "vignette"),
            EditOp::Levels { black, white, gamma } => {
                params.insert("black".into(), serde_json::json!(black));
                params.insert("white".into(), serde_json::json!(white));
                params.insert("gamma".into(), serde_json::json!(gamma));
                "levels"
            }
            EditOp::Dehaze(v) => val_op(&mut params, v, "dehaze"),
            EditOp::HueRotate(v) => val_op(&mut params, v, "hue_rotate"),
            EditOp::SplitTone(v) => val_op(&mut params, v, "split_tone"),
            EditOp::SelectiveColor { hue, sat } => {
                params.insert("hue".into(), serde_json::json!(hue));
                params.insert("sat".into(), serde_json::json!(sat));
                "selective_color"
            }
            EditOp::Grain(v) => val_op(&mut params, v, "grain"),
            EditOp::Despeckle(v) => val_op(&mut params, v, "despeckle"),
            EditOp::GradND { dir, strength } => {
                params.insert("dir".into(), serde_json::json!(dir));
                params.insert("strength".into(), serde_json::json!(strength));
                "grad_nd"
            }
            EditOp::Radial(v) => val_op(&mut params, v, "radial"),
            EditOp::Curve { pts } => {
                for (i, v) in pts.iter().enumerate() {
                    params.insert(format!("p{i}"), serde_json::json!(v));
                }
                "curve"
            }
            EditOp::Clahe(v) => val_op(&mut params, v, "clahe"),
            EditOp::Invert => "invert",
            EditOp::Sepia => "sepia",
            EditOp::Duotone => "duotone",
            EditOp::Posterize(v) => val_op(&mut params, v, "posterize"),
            EditOp::Solarize(v) => val_op(&mut params, v, "solarize"),
            EditOp::Threshold(v) => val_op(&mut params, v, "threshold"),
            EditOp::OilPaint { style, strength } => style_op(&mut params, style, strength, "oil_paint"),
            EditOp::PencilSketch(v) => val_op(&mut params, v, "pencil_sketch"),
            EditOp::Cartoon(v) => val_op(&mut params, v, "cartoon"),
            EditOp::Watercolor { style, strength } => style_op(&mut params, style, strength, "watercolor"),
            EditOp::Ink { style, strength } => style_op(&mut params, style, strength, "ink"),
            EditOp::Emboss(v) => val_op(&mut params, v, "emboss"),
            EditOp::Blur(v) => val_op(&mut params, v, "blur"),
            EditOp::Bloom(v) => val_op(&mut params, v, "bloom"),
            EditOp::Charcoal(v) => val_op(&mut params, v, "charcoal"),
            EditOp::Halftone(v) => val_op(&mut params, v, "halftone"),
            EditOp::FalseColor { style, strength } => style_op(&mut params, style, strength, "false_color"),
            EditOp::Pixelate(v) => val_op(&mut params, v, "pixelate"),
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
        // `hue:DEGREES` and `selcolor:HUE,SAT`.
        if let Some(rest) = tag.strip_prefix("hue:") {
            return rest.trim().parse::<i32>().ok().map(EditOp::HueRotate);
        }
        if let Some(rest) = tag.strip_prefix("selcolor:") {
            let p: Vec<&str> = rest.split(&[',', ' ']).filter(|s| !s.is_empty()).collect();
            if let [h, s] = p.as_slice() {
                return Some(EditOp::SelectiveColor { hue: h.trim().parse().ok()?, sat: s.trim().parse().ok()? });
            }
            return None;
        }
        let op = match tag {
            "auto_enhance" | "enhance" | "auto" => EditOp::AutoEnhance,
            "vignette" | "vignette_dark" => EditOp::Vignette(30),
            "vignette_light" | "lighten_edges" => EditOp::Vignette(-30),
            "dehaze" | "defog" => EditOp::Dehaze(35),
            "split_tone" | "split_tone_warm" => EditOp::SplitTone(35),
            "split_tone_cool" => EditOp::SplitTone(-35),
            "grain" | "film_grain" | "add_grain" => EditOp::Grain(30),
            "despeckle" | "median" | "remove_speckle" => EditOp::Despeckle(55),
            "burn" => EditOp::Radial(30),
            "dodge" => EditOp::Radial(-30),
            "grad_nd" | "graduated_nd" | "nd_grad" => EditOp::GradND { dir: 0, strength: 30 },
            "keystone" | "fix_verticals" | "keystone_v" => EditOp::Keystone { axis: 0, amount: 30 },
            "keystone_h" => EditOp::Keystone { axis: 1, amount: 30 },
            "clahe" | "equalize" | "adaptive_contrast" => EditOp::Clahe(60),
            "invert" | "negative" => EditOp::Invert,
            "sepia" => EditOp::Sepia,
            "duotone" => EditOp::Duotone,
            "posterize" => EditOp::Posterize(50),
            "solarize" => EditOp::Solarize(50),
            "threshold" | "black_white" => EditOp::Threshold(128),
            "oil_paint" | "oilify" | "oil" => EditOp::OilPaint { style: 3, strength: 100 },
            "pencil_sketch" | "sketch" | "pencil" => EditOp::PencilSketch(100),
            "cartoon" | "comic" => EditOp::Cartoon(100),
            "watercolor" | "watercolour" => EditOp::Watercolor { style: 5, strength: 100 },
            "european_ink" | "ink" => EditOp::Ink { style: 1, strength: 100 },
            "japanese_ink" | "sumie" | "sumi_e" => EditOp::Ink { style: 2, strength: 100 },
            "chinese_ink" | "ink_wash" | "shanshui" => EditOp::Ink { style: 3, strength: 100 },
            "russian_icon" | "icon" | "tempera" => EditOp::Ink { style: 4, strength: 100 },
            "emboss" => EditOp::Emboss(100),
            "blur" | "soft_focus" | "gaussian_blur" => EditOp::Blur(50),
            "bloom" | "glow" | "orton" => EditOp::Bloom(60),
            "charcoal" => EditOp::Charcoal(100),
            "halftone" | "newsprint" => EditOp::Halftone(100),
            "thermal" | "false_color" => EditOp::FalseColor { style: 1, strength: 100 },
            "infrared" => EditOp::FalseColor { style: 2, strength: 100 },
            "night_vision" | "nightvision" => EditOp::FalseColor { style: 3, strength: 100 },
            "pixelate" | "mosaic" | "pixelize" => EditOp::Pixelate(40),
            "pop_reds" | "pop_red" | "boost_reds" => EditOp::SelectiveColor { hue: 0, sat: 45 },
            "mute_reds" | "mute_red" => EditOp::SelectiveColor { hue: 0, sat: -55 },
            "pop_greens" | "pop_green" | "boost_greens" => EditOp::SelectiveColor { hue: 120, sat: 45 },
            "mute_greens" | "mute_green" => EditOp::SelectiveColor { hue: 120, sat: -55 },
            "pop_blues" | "pop_blue" | "boost_blues" => EditOp::SelectiveColor { hue: 240, sat: 45 },
            "mute_blues" | "mute_blue" => EditOp::SelectiveColor { hue: 240, sat: -55 },
            "sharpen" => EditOp::Sharpen(S),
            "soften" => EditOp::Sharpen(-S),
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
            "keystone" => EditOp::Keystone { axis: iv("axis", 0), amount: iv("amount", 0) },
            "vignette" => EditOp::Vignette(val()),
            "levels" => EditOp::Levels { black: iv("black", 0), white: iv("white", 255), gamma: iv("gamma", 100) },
            "dehaze" => EditOp::Dehaze(val()),
            "hue_rotate" => EditOp::HueRotate(val()),
            "split_tone" => EditOp::SplitTone(val()),
            "selective_color" => EditOp::SelectiveColor { hue: iv("hue", 0), sat: iv("sat", 0) },
            "grain" => EditOp::Grain(val()),
            "despeckle" => EditOp::Despeckle(val()),
            "grad_nd" => EditOp::GradND { dir: iv("dir", 0), strength: iv("strength", 0) },
            "radial" => EditOp::Radial(val()),
            "curve" => EditOp::Curve {
                pts: [iv("p0", 0), iv("p1", 64), iv("p2", 128), iv("p3", 192), iv("p4", 255)],
            },
            "clahe" => EditOp::Clahe(val()),
            "invert" => EditOp::Invert,
            "sepia" => EditOp::Sepia,
            "duotone" => EditOp::Duotone,
            "posterize" => EditOp::Posterize(val()),
            "solarize" => EditOp::Solarize(val()),
            "threshold" => EditOp::Threshold(val()),
            "oil_paint" => EditOp::OilPaint { style: iv("style", 3), strength: iv("strength", 100) },
            "pencil_sketch" => EditOp::PencilSketch(iv("value", 100)),
            "cartoon" => EditOp::Cartoon(iv("value", 100)),
            "watercolor" => EditOp::Watercolor { style: iv("style", 5), strength: iv("strength", 100) },
            "ink" => EditOp::Ink { style: iv("style", 1), strength: iv("strength", 100) },
            "emboss" => EditOp::Emboss(iv("value", 100)),
            "blur" => EditOp::Blur(iv("value", 50)),
            "bloom" => EditOp::Bloom(iv("value", 60)),
            "charcoal" => EditOp::Charcoal(iv("value", 100)),
            "halftone" => EditOp::Halftone(iv("value", 100)),
            "false_color" => EditOp::FalseColor { style: iv("style", 1), strength: iv("strength", 100) },
            "pixelate" => EditOp::Pixelate(val()),
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

/// Serialise a `{style, strength}` filter op.
fn style_op(
    params: &mut std::collections::HashMap<String, serde_json::Value>,
    style: i32,
    strength: i32,
    tag: &'static str,
) -> &'static str {
    params.insert("style".into(), serde_json::json!(style));
    params.insert("strength".into(), serde_json::json!(strength));
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

    /// RGB (0..1) → HSV (h in 0..360, s/v in 0..1).
    fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;
        let h = if d < 1e-6 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / d) % 6.0)
        } else if max == g {
            60.0 * ((b - r) / d + 2.0)
        } else {
            60.0 * ((r - g) / d + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        (h, if max < 1e-6 { 0.0 } else { d / max }, max)
    }

    /// HSV → RGB (0..1).
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
        let h = h.rem_euclid(360.0);
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r, g, b) = match (h / 60.0) as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        [r + m, g + m, b + m]
    }

    /// Dehaze: cut the washed-out veil — contrast around mid-grey plus a saturation lift, scaled by
    /// `amount`. Reads as pulling a flat, hazy frame back to punch.
    pub fn dehaze(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = (amount as f32 / 100.0).clamp(-1.0, 1.0);
        map_rgb(img, |r, g, b| {
            let ct = 1.0 + t * 0.8; // contrast around 0.5
            let cr = |c: f32| (c - 0.5) * ct + 0.5;
            let (r, g, b) = (cr(r), cr(g), cr(b));
            let y = luma(r, g, b);
            let sf = 1.0 + t * 0.5; // saturation lift
            [y + (r - y) * sf, y + (g - y) * sf, y + (b - y) * sf]
        })
    }

    /// Rotate every hue by `degrees`.
    pub fn hue_rotate(img: &DynamicImage, degrees: i32) -> DynamicImage {
        let d = degrees as f32;
        map_rgb(img, |r, g, b| {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            hsv_to_rgb(h + d, s, v)
        })
    }

    /// Split-tone: warm the highlights and cool the shadows (positive), or the inverse (negative).
    pub fn split_tone(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = amount as f32 / 100.0;
        map_rgb(img, |r, g, b| {
            let y = luma(r, g, b);
            let whi = smoothstep(0.5, 1.0, y);
            let wsh = 1.0 - smoothstep(0.0, 0.5, y);
            let dr = t * (0.15 * whi - 0.10 * wsh);
            let db = t * (0.15 * wsh - 0.12 * whi);
            [r + dr, g, b + db]
        })
    }

    /// Selective colour: scale the saturation of pixels whose hue is near `hue` (degrees) by `sat`
    /// (−100..100), with a smooth ±60° band so the shift is confined to that colour family.
    pub fn selective_color(img: &DynamicImage, hue: i32, sat: i32) -> DynamicImage {
        let target = hue as f32;
        let amt = sat as f32 / 100.0;
        map_rgb(img, |r, g, b| {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let mut dh = (h - target).abs() % 360.0;
            if dh > 180.0 {
                dh = 360.0 - dh;
            }
            let w = 1.0 - smoothstep(20.0, 80.0, dh); // full within 20°, gone by 80°
            hsv_to_rgb(h, (s * (1.0 + amt * w)).clamp(0.0, 1.0), v)
        })
    }

    /// Film grain: add deterministic monochrome noise (same pattern on replay — seeded by position).
    pub fn grain(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = (amount as f32 / 100.0).clamp(0.0, 1.0) * 0.18;
        let mut rgb = img.to_rgb8();
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            // Cheap integer hash of (x, y) → noise in [-1, 1]; deterministic, so replay matches.
            let hsh = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)).wrapping_mul(83_492_791);
            let n = ((hsh >> 8) & 0xffff) as f32 / 32_768.0 - 1.0;
            let d = n * t;
            for c in 0..3 {
                p.0[c] = enc(p.0[c] as f32 / 255.0 + d);
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Median despeckle: blend toward a 3×3 per-channel median (edge-clamped) by `amount` (0..100).
    pub fn despeckle(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = (amount as f32 / 100.0).clamp(0.0, 1.0);
        let src = img.to_rgb8();
        let (w, h) = (src.width() as i32, src.height() as i32);
        let mut out = RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let base = src.get_pixel(x as u32, y as u32).0;
                let mut px = [0u8; 3];
                for c in 0..3 {
                    let mut win = [0u8; 9];
                    let mut k = 0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let sx = (x + dx).clamp(0, w - 1) as u32;
                            let sy = (y + dy).clamp(0, h - 1) as u32;
                            win[k] = src.get_pixel(sx, sy).0[c];
                            k += 1;
                        }
                    }
                    win.sort_unstable();
                    let med = win[4] as f32;
                    px[c] = enc((base[c] as f32 * (1.0 - t) + med * t) / 255.0);
                }
                out.put_pixel(x as u32, y as u32, Rgb(px));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Graduated ND: a linear tonal gradient strongest at one edge (`dir`: 0 top/1 bottom/2 left/
    /// 3 right), fading to nothing by the middle. Positive darkens (a grad-ND filter for skies),
    /// negative lightens.
    pub fn grad_nd(img: &DynamicImage, dir: i32, strength: i32) -> DynamicImage {
        let t = (strength as f32 / 100.0).clamp(-1.0, 1.0);
        let mut rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as f32, rgb.height() as f32);
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            let frac = match dir {
                0 => y as f32 / h.max(1.0),         // top
                1 => 1.0 - y as f32 / h.max(1.0),   // bottom
                2 => x as f32 / w.max(1.0),         // left
                _ => 1.0 - x as f32 / w.max(1.0),   // right
            };
            let wgt = (1.0 - frac * 2.0).clamp(0.0, 1.0); // full at the edge, 0 by the centre
            let f = (1.0 - t * wgt).clamp(0.0, 2.0);
            for c in 0..3 {
                p.0[c] = enc(p.0[c] as f32 / 255.0 * f);
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Radial dodge/burn: a centred radial adjustment — positive darkens (burns) the centre, negative
    /// lightens (dodges) it, fading out toward the edges.
    pub fn radial(img: &DynamicImage, strength: i32) -> DynamicImage {
        let t = (strength as f32 / 100.0).clamp(-1.0, 1.0);
        let mut rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as f32, rgb.height() as f32);
        let (cx, cy) = (w / 2.0, h / 2.0);
        let maxr = (cx * cx + cy * cy).sqrt().max(1.0);
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt() / maxr;
            let wgt = 1.0 - smoothstep(0.0, 0.75, r); // strong at the centre
            let f = (1.0 - t * wgt).clamp(0.0, 2.0);
            for c in 0..3 {
                p.0[c] = enc(p.0[c] as f32 / 255.0 * f);
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Tone curve: build a 256-entry LUT by linear interpolation between the five output points
    /// (at inputs 0/64/128/192/255) and map R, G, B through it.
    pub fn curve(img: &DynamicImage, pts: [i32; 5]) -> DynamicImage {
        let xs = [0i32, 64, 128, 192, 255];
        let mut lut = [0u8; 256];
        for (i, l) in lut.iter_mut().enumerate() {
            let iv = i as i32;
            let mut seg = 0;
            while seg < 3 && iv > xs[seg + 1] {
                seg += 1;
            }
            let (x0, x1) = (xs[seg], xs[seg + 1]);
            let (y0, y1) = (pts[seg] as f32, pts[seg + 1] as f32);
            let t = if x1 > x0 { (iv - x0) as f32 / (x1 - x0) as f32 } else { 0.0 };
            *l = (y0 + t * (y1 - y0)).clamp(0.0, 255.0) as u8;
        }
        let mut rgb = img.to_rgb8();
        for p in rgb.pixels_mut() {
            for c in 0..3 {
                p.0[c] = lut[p.0[c] as usize];
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// CLAHE: contrast-limited adaptive histogram equalization on the luma channel (8×8 tiles,
    /// clipped histograms, bilinearly-interpolated mappings), colour preserved by a luma ratio, then
    /// blended with the original by `amount` (0..100).
    pub fn clahe(img: &DynamicImage, amount: i32) -> DynamicImage {
        let t = (amount as f32 / 100.0).clamp(0.0, 1.0);
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        if t <= 0.0 || w < 8 || h < 8 {
            return DynamicImage::ImageRgb8(rgb);
        }
        const N: usize = 8; // tiles per axis
        let (tw, th) = (w.div_ceil(N), h.div_ceil(N));
        let mut lum = vec![0u8; w * h];
        for (i, p) in rgb.pixels().enumerate() {
            lum[i] = luma(p[0] as f32, p[1] as f32, p[2] as f32).round().clamp(0.0, 255.0) as u8;
        }
        // Per-tile clipped-CDF mapping.
        let clip = ((tw * th) as f32 / 256.0 * 4.0).max(1.0) as u32;
        let mut maps = vec![[0u8; 256]; N * N];
        for ty in 0..N {
            for tx in 0..N {
                let (x0, y0) = (tx * tw, ty * th);
                let (x1, y1) = ((x0 + tw).min(w), (y0 + th).min(h));
                let mut hist = [0u32; 256];
                for y in y0..y1 {
                    for x in x0..x1 {
                        hist[lum[y * w + x] as usize] += 1;
                    }
                }
                let mut excess = 0u32;
                for b in hist.iter_mut() {
                    if *b > clip {
                        excess += *b - clip;
                        *b = clip;
                    }
                }
                let add = excess / 256;
                for b in hist.iter_mut() {
                    *b += add;
                }
                let total = hist.iter().sum::<u32>().max(1);
                let map = &mut maps[ty * N + tx];
                let mut cdf = 0u32;
                for i in 0..256 {
                    cdf += hist[i];
                    map[i] = (cdf * 255 / total).min(255) as u8;
                }
            }
        }
        // Bilinear blend of the four surrounding tile mappings per pixel.
        let mut out = rgb.clone();
        for y in 0..h {
            let fy = (y as f32 / th as f32) - 0.5;
            let ty0 = fy.floor().clamp(0.0, (N - 1) as f32) as usize;
            let ty1 = (ty0 + 1).min(N - 1);
            let dy = (fy - ty0 as f32).clamp(0.0, 1.0);
            for x in 0..w {
                let fx = (x as f32 / tw as f32) - 0.5;
                let tx0 = fx.floor().clamp(0.0, (N - 1) as f32) as usize;
                let tx1 = (tx0 + 1).min(N - 1);
                let dx = (fx - tx0 as f32).clamp(0.0, 1.0);
                let l = lum[y * w + x] as usize;
                let (a, b) = (maps[ty0 * N + tx0][l] as f32, maps[ty0 * N + tx1][l] as f32);
                let (c, d) = (maps[ty1 * N + tx0][l] as f32, maps[ty1 * N + tx1][l] as f32);
                let newl = (a * (1.0 - dx) + b * dx) * (1.0 - dy) + (c * (1.0 - dx) + d * dx) * dy;
                let oldl = lum[y * w + x] as f32;
                let ratio = if oldl > 1.0 { newl / oldl } else { 1.0 };
                let p = out.get_pixel_mut(x as u32, y as u32);
                for ch in 0..3 {
                    let orig = p.0[ch] as f32;
                    p.0[ch] = (orig * (1.0 - t) + (orig * ratio).clamp(0.0, 255.0) * t) as u8;
                }
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Invert (photo negative).
    pub fn invert(img: &DynamicImage) -> DynamicImage {
        map_rgb(img, |r, g, b| [1.0 - r, 1.0 - g, 1.0 - b])
    }

    /// Sepia tone (the classic luminance-preserving warm matrix).
    pub fn sepia(img: &DynamicImage) -> DynamicImage {
        map_rgb(img, |r, g, b| {
            [
                0.393 * r + 0.769 * g + 0.189 * b,
                0.349 * r + 0.686 * g + 0.168 * b,
                0.272 * r + 0.534 * g + 0.131 * b,
            ]
        })
    }

    /// Duotone: map luma to a two-colour gradient (deep indigo shadows → warm cream highlights).
    pub fn duotone(img: &DynamicImage) -> DynamicImage {
        let lo = [0.12, 0.10, 0.24];
        let hi = [1.0, 0.90, 0.71];
        map_rgb(img, |r, g, b| {
            let y = luma(r, g, b);
            std::array::from_fn(|i| lo[i] + (hi[i] - lo[i]) * y)
        })
    }

    /// Posterize: quantise each channel to N tonal levels. `strength` 0 → 256 (identity) … 100 → 2.
    pub fn posterize(img: &DynamicImage, strength: i32) -> DynamicImage {
        let levels = (256.0 - (strength.clamp(0, 100) as f32 / 100.0) * 254.0).round().max(2.0);
        let n = levels - 1.0;
        map_rgb(img, |r, g, b| [(r * n).round() / n, (g * n).round() / n, (b * n).round() / n])
    }

    /// Solarize: invert channel values above a threshold. `strength` 0 → none … 100 → mid (classic).
    pub fn solarize(img: &DynamicImage, strength: i32) -> DynamicImage {
        let thr = 1.0 - strength.clamp(0, 100) as f32 / 200.0; // 1.0 .. 0.5
        let s = |c: f32| if c > thr { 1.0 - c } else { c };
        map_rgb(img, |r, g, b| [s(r), s(g), s(b)])
    }

    /// Threshold to black & white at luma `level` (0..255).
    pub fn threshold(img: &DynamicImage, level: i32) -> DynamicImage {
        let t = level.clamp(0, 255) as f32 / 255.0;
        map_rgb(img, |r, g, b| {
            let v = if luma(r, g, b) >= t { 1.0 } else { 0.0 };
            [v, v, v]
        })
    }

    /// Oilify core: for each pixel, over a `radius` window with `bins` intensity buckets, paint the
    /// most-common bucket's average colour.
    fn oilify(rgb: &RgbImage, radius: i32, bins: usize) -> RgbImage {
        let (w, h) = (rgb.width() as i32, rgb.height() as i32);
        let bins = bins.clamp(4, 32);
        let mut out = RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let mut cnt = vec![0u32; bins];
                let mut sum = vec![[0u32; 3]; bins];
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let sx = (x + dx).clamp(0, w - 1) as u32;
                        let sy = (y + dy).clamp(0, h - 1) as u32;
                        let p = rgb.get_pixel(sx, sy).0;
                        let l = luma(p[0] as f32, p[1] as f32, p[2] as f32) as usize;
                        let b = (l * bins / 256).min(bins - 1);
                        cnt[b] += 1;
                        for c in 0..3 {
                            sum[b][c] += p[c] as u32;
                        }
                    }
                }
                let best = (0..bins).max_by_key(|&b| cnt[b]).unwrap_or(0);
                let n = cnt[best].max(1);
                out.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([(sum[best][0] / n) as u8, (sum[best][1] / n) as u8, (sum[best][2] / n) as u8]),
                );
            }
        }
        out
    }

    /// Oil-paint style presets: `(radius, bins, saturation×100 delta)` — ten distinct brush looks.
    fn oil_style(v: i32) -> (i32, usize, i32) {
        const S: [(i32, usize, i32); 10] = [
            (2, 24, 5), (3, 20, 12), (3, 14, 22), (4, 18, 0), (4, 10, 30),
            (5, 16, 15), (2, 12, 40), (5, 8, 25), (3, 28, -12), (6, 12, 10),
        ];
        S[(v.clamp(1, 10) - 1) as usize]
    }

    /// Oil paint, style 1..10.
    pub fn oil_paint(img: &DynamicImage, style: i32) -> DynamicImage {
        let (r, bins, sat) = oil_style(style);
        let painted = DynamicImage::ImageRgb8(oilify(&img.to_rgb8(), r, bins));
        if sat != 0 { saturation(&painted, sat) } else { painted }
    }

    /// Pencil sketch: colour-dodge the grayscale by a blurred inverse of itself.
    pub fn pencil_sketch(img: &DynamicImage) -> DynamicImage {
        let g = img.to_luma8();
        let mut inv = g.clone();
        for p in inv.pixels_mut() {
            p.0[0] = 255 - p.0[0];
        }
        let bl = image::imageops::blur(&inv, 8.0);
        let (w, h) = (g.width(), g.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = g.get_pixel(x, y).0[0] as f32;
                let b = bl.get_pixel(x, y).0[0] as f32;
                let v = if b >= 255.0 { 255.0 } else { (a * 255.0 / (255.0 - b)).min(255.0) };
                let v = v as u8;
                out.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Sobel edge magnitude (0..1) over the luma of `rgb`.
    fn edge_mag(rgb: &RgbImage, x: u32, y: u32) -> f32 {
        let (w, h) = (rgb.width() as i32, rgb.height() as i32);
        let l = |xx: i32, yy: i32| {
            let p = rgb.get_pixel(xx.clamp(0, w - 1) as u32, yy.clamp(0, h - 1) as u32).0;
            luma(p[0] as f32, p[1] as f32, p[2] as f32)
        };
        let (x, y) = (x as i32, y as i32);
        let gx = l(x + 1, y - 1) + 2.0 * l(x + 1, y) + l(x + 1, y + 1)
            - l(x - 1, y - 1) - 2.0 * l(x - 1, y) - l(x - 1, y + 1);
        let gy = l(x - 1, y + 1) + 2.0 * l(x, y + 1) + l(x + 1, y + 1)
            - l(x - 1, y - 1) - 2.0 * l(x, y - 1) - l(x + 1, y - 1);
        ((gx * gx + gy * gy).sqrt() / 255.0).min(1.0)
    }

    /// Cartoon / comic: quantise the (blurred) colours and lay dark ink on the strong edges.
    pub fn cartoon(img: &DynamicImage) -> DynamicImage {
        let sm = image::imageops::blur(&img.to_rgb8(), 1.5);
        let (w, h) = (sm.width(), sm.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = sm.get_pixel(x, y).0;
                // Posterize each channel to 6 levels.
                let q = |c: u8| ((c as f32 / 255.0 * 5.0).round() / 5.0 * 255.0) as u8;
                let edge = edge_mag(&sm, x, y);
                let px = if edge > 0.28 { [20, 20, 20] } else { [q(p[0]), q(p[1]), q(p[2])] };
                out.put_pixel(x, y, Rgb(px));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Watercolour core: edge-preserving smoothing + reduced palette (`levels`) + soft dark edges
    /// (`edge`), on an optionally `warm`-tinted, `sat`-scaled base.
    fn watercolor_core(img: &DynamicImage, blur: f32, levels: f32, edge: f32, sat: i32, warm: i32) -> DynamicImage {
        let mut base = despeckle(img, 100); // median smoothing
        if sat != 0 {
            base = saturation(&base, sat);
        }
        if warm != 0 {
            base = warmth(&base, warm);
        }
        let sm = image::imageops::blur(&base.to_rgb8(), blur.max(0.3));
        let (w, h) = (sm.width(), sm.height());
        let n = (levels - 1.0).max(1.0);
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = sm.get_pixel(x, y).0;
                let q = |c: u8| (c as f32 / 255.0 * n).round() / n;
                let e = edge_mag(&sm, x, y);
                let dark = 1.0 - edge * (e - 0.2).clamp(0.0, 1.0);
                out.put_pixel(x, y, Rgb([enc(q(p[0]) * dark), enc(q(p[1]) * dark), enc(q(p[2]) * dark)]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Watercolour style presets: `(blur, palette levels, edge darkening, saturation, warmth)`.
    fn watercolor_style(v: i32) -> (f32, f32, f32, i32, i32) {
        const S: [(f32, f32, f32, i32, i32); 10] = [
            (1.0, 10.0, 0.5, 10, 4), (1.5, 8.0, 0.4, 20, 8), (2.0, 6.0, 0.6, 15, 0),
            (0.8, 12.0, 0.7, 5, -6), (2.5, 7.0, 0.3, 25, 12), (1.2, 9.0, 0.55, 12, 6),
            (3.0, 5.0, 0.45, 30, -4), (1.8, 11.0, 0.35, 8, 10), (1.0, 6.0, 0.8, 18, 2),
            (2.2, 8.0, 0.5, 22, 16),
        ];
        S[(v.clamp(1, 10) - 1) as usize]
    }

    /// Watercolour, style 1..10.
    pub fn watercolor(img: &DynamicImage, style: i32) -> DynamicImage {
        let (b, l, e, s, w) = watercolor_style(style);
        watercolor_core(img, b, l, e, s, w)
    }

    /// Short name for an ink/paint style.
    pub fn ink_name(v: i32) -> &'static str {
        match v {
            1 => "European",
            2 => "Japanese sumi-e",
            3 => "Chinese wash",
            4 => "Russian icon",
            _ => "ink",
        }
    }

    /// Blend a filtered result back over the original by `strength` (0 = original … 100 = filtered).
    /// Both must share dimensions (all filters preserve them).
    pub fn blend(orig: &DynamicImage, filtered: DynamicImage, strength: i32) -> DynamicImage {
        let t = strength.clamp(0, 100) as f32 / 100.0;
        if t >= 1.0 {
            return filtered;
        }
        if t <= 0.0 {
            return orig.clone();
        }
        let o = orig.to_rgb8();
        let mut f = filtered.to_rgb8();
        for (fp, op) in f.pixels_mut().zip(o.pixels()) {
            for c in 0..3 {
                fp.0[c] = (op.0[c] as f32 * (1.0 - t) + fp.0[c] as f32 * t) as u8;
            }
        }
        DynamicImage::ImageRgb8(f)
    }

    fn to_gray(img: &DynamicImage) -> DynamicImage {
        map_rgb(img, |r, g, b| {
            let y = luma(r, g, b);
            [y, y, y]
        })
    }

    /// Ink / traditional-painting styles (non-AI combinations of tone, edges, wash, and tint).
    pub fn ink(img: &DynamicImage, style: i32) -> DynamicImage {
        match style {
            // European pen-and-ink: high-contrast black lines on white (edges over a bright wash).
            1 => {
                let rgb = img.to_rgb8();
                let (w, h) = (rgb.width(), rgb.height());
                let mut out = RgbImage::new(w, h);
                for y in 0..h {
                    for x in 0..w {
                        let p = rgb.get_pixel(x, y).0;
                        let yl = luma(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0;
                        let e = edge_mag(&rgb, x, y);
                        // Strong edges → ink; dark tones get sparse hatching; else paper white.
                        let v = if e > 0.22 || (yl < 0.28 && (x + y) % 5 == 0) { 0.0 } else { 1.0 };
                        out.put_pixel(x, y, Rgb([enc(v), enc(v), enc(v)]));
                    }
                }
                DynamicImage::ImageRgb8(out)
            }
            // Japanese sumi-e: soft high-key grey ink, bold dark accents, warm paper.
            2 => {
                let g = to_gray(img);
                let g = brightness_f(&g, 0.12); // high key
                let sm = DynamicImage::ImageRgb8(image::imageops::blur(&g.to_rgb8(), 1.6));
                let inked = map_rgb(&sm, |r, _, _| {
                    let v = if r < 0.32 { r * 0.4 } else { r }; // deepen the darkest strokes
                    [v, v, v]
                });
                warmth(&paper_tint(&inked, [1.0, 0.98, 0.92]), 6)
            }
            // Chinese ink wash (shan-shui): soft misty grey gradients, low contrast, paper tone.
            3 => {
                let g = to_gray(img);
                let sm = DynamicImage::ImageRgb8(image::imageops::blur(&g.to_rgb8(), 2.4));
                let washed = map_rgb(&sm, |r, _, _| {
                    let v = (r - 0.5) * 0.7 + 0.55; // gentle contrast, lifted
                    [v, v, v]
                });
                paper_tint(&washed, [0.99, 0.98, 0.94])
            }
            // Russian icon / egg-tempera: warm ochre/gold palette, flat regions, dark outlines.
            _ => {
                let warm = warmth(img, 22);
                let sm = image::imageops::blur(&warm.to_rgb8(), 1.2);
                let (w, h) = (sm.width(), sm.height());
                let mut out = RgbImage::new(w, h);
                let n = 4.0f32; // flat, posterised regions
                for y in 0..h {
                    for x in 0..w {
                        let p = sm.get_pixel(x, y).0;
                        let q = |c: u8| (c as f32 / 255.0 * n).round() / n;
                        let yl = luma(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0;
                        let e = edge_mag(&sm, x, y);
                        let mut px = [q(p[0]), q(p[1]), q(p[2])];
                        // Gold in the highlights.
                        if yl > 0.72 {
                            px = [px[0].max(0.85), px[1].max(0.72), px[2].max(0.35)];
                        }
                        // Dark ochre outlines.
                        if e > 0.3 {
                            px = [0.16, 0.11, 0.06];
                        }
                        out.put_pixel(x, y, Rgb([enc(px[0]), enc(px[1]), enc(px[2])]));
                    }
                }
                // A slight vignette for the icon frame feel.
                vignette(&DynamicImage::ImageRgb8(out), 20)
            }
        }
    }

    /// Multiply toward a paper-tone RGB (subtle warm/cream cast).
    fn paper_tint(img: &DynamicImage, tint: [f32; 3]) -> DynamicImage {
        map_rgb(img, |r, g, b| [r * tint[0], g * tint[1], b * tint[2]])
    }

    /// Brightness as a normalised additive lift (for internal filter use).
    fn brightness_f(img: &DynamicImage, d: f32) -> DynamicImage {
        map_rgb(img, |r, g, b| [r + d, g + d, b + d])
    }

    /// Emboss: a directional luma gradient rendered as grey relief.
    pub fn emboss(img: &DynamicImage) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let (iw, ih) = (w as i32, h as i32);
        let l = |xx: i32, yy: i32| {
            let p = rgb.get_pixel(xx.clamp(0, iw - 1) as u32, yy.clamp(0, ih - 1) as u32).0;
            luma(p[0] as f32, p[1] as f32, p[2] as f32)
        };
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (xi, yi) = (x as i32, y as i32);
                let v = (128.0 + (l(xi - 1, yi - 1) - l(xi + 1, yi + 1))).clamp(0.0, 255.0) as u8;
                out.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Pixelate / mosaic: average each block. `strength` 0 = none … 100 = large blocks.
    pub fn pixelate(img: &DynamicImage, strength: i32) -> DynamicImage {
        let block = (1 + strength.clamp(0, 100) * 40 / 100) as u32;
        if block <= 1 {
            return img.clone();
        }
        let mut rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        for by in (0..h).step_by(block as usize) {
            for bx in (0..w).step_by(block as usize) {
                let (x1, y1) = ((bx + block).min(w), (by + block).min(h));
                let mut sum = [0u64; 3];
                let mut n = 0u64;
                for y in by..y1 {
                    for x in bx..x1 {
                        let p = rgb.get_pixel(x, y).0;
                        for c in 0..3 {
                            sum[c] += p[c] as u64;
                        }
                        n += 1;
                    }
                }
                let n = n.max(1);
                let avg = [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8];
                for y in by..y1 {
                    for x in bx..x1 {
                        rgb.put_pixel(x, y, Rgb(avg));
                    }
                }
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Gaussian blur / soft focus; `strength` 0..100 → sigma.
    pub fn gaussian(img: &DynamicImage, strength: i32) -> DynamicImage {
        let s = strength.clamp(0, 100);
        if s == 0 {
            return img.clone();
        }
        DynamicImage::ImageRgb8(image::imageops::blur(&img.to_rgb8(), s as f32 / 100.0 * 8.0))
    }

    /// Bloom / glow: screen-blend the blurred bright areas back over the image.
    pub fn bloom(img: &DynamicImage, strength: i32) -> DynamicImage {
        let t = strength.clamp(0, 100) as f32 / 100.0;
        if t <= 0.0 {
            return img.clone();
        }
        let rgb = img.to_rgb8();
        // Bright mask: keep only the highlights.
        let mut bright = rgb.clone();
        for p in bright.pixels_mut() {
            for c in 0..3 {
                let v = p.0[c] as f32 / 255.0;
                p.0[c] = enc(((v - 0.6).max(0.0) / 0.4).min(1.0));
            }
        }
        let glow = image::imageops::blur(&bright, 6.0);
        let mut out = rgb.clone();
        for (op, gp) in out.pixels_mut().zip(glow.pixels()) {
            for c in 0..3 {
                let a = op.0[c] as f32 / 255.0;
                let b = (gp.0[c] as f32 / 255.0) * t;
                op.0[c] = enc(1.0 - (1.0 - a) * (1.0 - b)); // screen
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Charcoal: dark edge strokes over a light paper, tone-shaded.
    pub fn charcoal(img: &DynamicImage) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = rgb.get_pixel(x, y).0;
                let yl = luma(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0;
                let e = edge_mag(&rgb, x, y);
                // Paper white, darkened by edges and (a little) by tone.
                let v = (1.0 - e * 1.3 - (1.0 - yl) * 0.25).clamp(0.0, 1.0);
                out.put_pixel(x, y, Rgb([enc(v), enc(v), enc(v)]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Halftone / newsprint: grayscale dots whose size tracks local darkness.
    pub fn halftone(img: &DynamicImage) -> DynamicImage {
        const CELL: u32 = 5;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        // Per-cell mean luma.
        let mut out = RgbImage::new(w, h);
        for cy in (0..h).step_by(CELL as usize) {
            for cx in (0..w).step_by(CELL as usize) {
                let (x1, y1) = ((cx + CELL).min(w), (cy + CELL).min(h));
                let mut sum = 0.0f32;
                let mut n = 0.0f32;
                for y in cy..y1 {
                    for x in cx..x1 {
                        let p = rgb.get_pixel(x, y).0;
                        sum += luma(p[0] as f32, p[1] as f32, p[2] as f32);
                        n += 1.0;
                    }
                }
                let dark = 1.0 - (sum / n.max(1.0)) / 255.0;
                let r = dark.sqrt() * (CELL as f32 * 0.75); // dot radius from darkness
                let (mx, my) = (cx as f32 + CELL as f32 / 2.0, cy as f32 + CELL as f32 / 2.0);
                for y in cy..y1 {
                    for x in cx..x1 {
                        let d = ((x as f32 - mx).powi(2) + (y as f32 - my).powi(2)).sqrt();
                        let v = if d < r { 0.0 } else { 1.0 };
                        out.put_pixel(x, y, Rgb([enc(v), enc(v), enc(v)]));
                    }
                }
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    pub fn false_color_name(v: i32) -> &'static str {
        match v {
            1 => "thermal",
            2 => "infrared",
            _ => "night-vision",
        }
    }

    /// Sample a colour ramp (stops evenly spaced) at `t` in 0..1.
    fn ramp(stops: &[[f32; 3]], t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
        let i = (t.floor() as usize).min(stops.len() - 2);
        let f = t - i as f32;
        std::array::from_fn(|c| stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f)
    }

    /// False-colour maps: 1 thermal, 2 infrared, 3 night-vision.
    pub fn false_color(img: &DynamicImage, style: i32) -> DynamicImage {
        match style {
            1 => {
                let stops = [[0.0, 0.0, 0.1], [0.2, 0.0, 0.5], [0.8, 0.0, 0.2], [1.0, 0.6, 0.0], [1.0, 1.0, 0.8]];
                map_rgb(img, |r, g, b| ramp(&stops, luma(r, g, b)))
            }
            2 => {
                // Infrared: foliage → warm pink/white, sky → deep blue (swap + boost red on greens).
                map_rgb(img, |r, g, b| {
                    let ir = (g * 1.2).min(1.0); // vegetation glows
                    [ir, (r * 0.6 + b * 0.2), (b * 0.7).min(1.0)]
                })
            }
            _ => {
                // Night-vision: green monochrome, boosted, with faint scan darkening.
                map_rgb(img, |r, g, b| {
                    let y = (luma(r, g, b) * 1.4).min(1.0);
                    [y * 0.1, y, y * 0.1]
                })
            }
        }
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

/// Keystone / perspective correction: a trapezoidal warp (bilinear, edge-clamped so there are no
/// black wedges). `axis` 0 scales width per row (vertical keystone — converging verticals), 1 scales
/// height per column (horizontal keystone); `amount` −100..100.
fn keystone(img: &DynamicImage, axis: i32, amount: i32) -> DynamicImage {
    let k = (amount as f32 / 100.0).clamp(-1.0, 1.0);
    if k.abs() < 1e-4 {
        return img.clone();
    }
    let src = img.to_rgb8();
    let (w, h) = (src.width() as f32, src.height() as f32);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let mut out = image::RgbImage::new(src.width(), src.height());
    for oy in 0..src.height() {
        let ny = (oy as f32 - cy) / h; // ~[-0.5, 0.5]
        for ox in 0..src.width() {
            let nx = (ox as f32 - cx) / w;
            let (sx, sy) = if axis == 0 {
                let scale = (1.0 + k * ny * 2.0).max(0.1); // width scale varies with the row
                (cx + (ox as f32 - cx) / scale, oy as f32)
            } else {
                let scale = (1.0 + k * nx * 2.0).max(0.1); // height scale varies with the column
                (ox as f32, cy + (oy as f32 - cy) / scale)
            };
            out.put_pixel(ox, oy, bilinear(&src, sx, sy));
        }
    }
    DynamicImage::ImageRgb8(out)
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
            EditOp::Dehaze(30), EditOp::HueRotate(45), EditOp::SplitTone(-20),
            EditOp::SelectiveColor { hue: 120, sat: 45 }, EditOp::Grain(30), EditOp::Despeckle(55),
            EditOp::GradND { dir: 1, strength: 40 }, EditOp::Radial(-30),
            EditOp::Curve { pts: [0, 50, 128, 205, 255] }, EditOp::Clahe(60),
            EditOp::Keystone { axis: 0, amount: 30 }, EditOp::Keystone { axis: 1, amount: -20 },
            EditOp::Invert, EditOp::Sepia, EditOp::Duotone,
            EditOp::Posterize(50), EditOp::Solarize(40), EditOp::Threshold(128),
            EditOp::OilPaint { style: 3, strength: 80 }, EditOp::PencilSketch(60), EditOp::Cartoon(70),
            EditOp::Watercolor { style: 5, strength: 90 }, EditOp::Ink { style: 2, strength: 100 },
            EditOp::Emboss(50), EditOp::Pixelate(40), EditOp::Blur(60), EditOp::Bloom(70),
            EditOp::Charcoal(80), EditOp::Halftone(90), EditOp::FalseColor { style: 1, strength: 100 },
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
        assert_eq!(EditOp::from_tag("dehaze"), Some(EditOp::Dehaze(35)));
        assert_eq!(EditOp::from_tag("hue:45"), Some(EditOp::HueRotate(45)));
        assert_eq!(EditOp::from_tag("grain"), Some(EditOp::Grain(30)));
        assert_eq!(EditOp::from_tag("despeckle"), Some(EditOp::Despeckle(55)));
        assert_eq!(EditOp::from_tag("pop_blues"), Some(EditOp::SelectiveColor { hue: 240, sat: 45 }));
        assert_eq!(EditOp::from_tag("selcolor:120,-50"), Some(EditOp::SelectiveColor { hue: 120, sat: -50 }));
        // Tag angle is in degrees; the op stores tenths (5° → 50, 2.5° → 25).
        assert_eq!(EditOp::from_tag("straighten:5"), Some(EditOp::Straighten(50)));
        assert_eq!(EditOp::from_tag("straighten:-2.5"), Some(EditOp::Straighten(-25)));
        // Structural tags still resolve through from_entry.
        assert_eq!(EditOp::from_tag("grayscale"), Some(EditOp::Grayscale));
        assert_eq!(EditOp::from_tag("warp_drive"), None);
    }

    #[test]
    fn scalar_helpers_roundtrip_and_range() {
        // Scalar ops expose + replace their amount; structural ops don't.
        assert_eq!(EditOp::Brightness(15).scalar(), Some(15));
        assert_eq!(EditOp::Warmth(0).with_scalar(-40), EditOp::Warmth(-40));
        assert_eq!(EditOp::HueRotate(0).with_scalar(90).scalar(), Some(90));
        assert_eq!(EditOp::Grayscale.scalar(), None);
        assert_eq!(EditOp::Levels { black: 0, white: 255, gamma: 100 }.scalar(), None);
        // Ranges: hue is ±180, effect ops are one-sided, tonal ops bipolar ±100.
        assert_eq!(EditOp::HueRotate(0).scalar_range(), (-180, 180, 5));
        assert_eq!(EditOp::NoiseReduction(0).scalar_range().0, 0);
        assert_eq!(EditOp::Brightness(0).scalar_range(), (-100, 100, 5));
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
    fn keystone_warps_and_is_noop_at_zero() {
        let src = img(40, 40);
        let noop = EditOp::Keystone { axis: 0, amount: 0 }.apply(src.clone()).to_rgb8();
        assert_eq!(noop, src.to_rgb8(), "0 amount is identity");
        // A vertical keystone keeps dims but changes pixels; the centre row stays put.
        let out = EditOp::Keystone { axis: 0, amount: 60 }.apply(src.clone()).to_rgb8();
        assert_eq!((out.width(), out.height()), (40, 40));
        assert_ne!(out, src.to_rgb8());
        assert_eq!(EditOp::from_tag("keystone"), Some(EditOp::Keystone { axis: 0, amount: 30 }));
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
    fn hue_rotate_360_is_identity_and_grain_is_deterministic() {
        let src = DynamicImage::ImageRgb8(ImageBuffer::from_fn(16, 16, |x, y| {
            Rgb([(x * 15) as u8, (y * 15) as u8, 90])
        }));
        // A full turn returns (near) the original colours.
        let back = EditOp::HueRotate(360).apply(src.clone()).to_rgb8();
        let orig = src.to_rgb8();
        for (a, b) in orig.pixels().zip(back.pixels()) {
            for c in 0..3 {
                assert!(a.0[c].abs_diff(b.0[c]) <= 2, "hue 360° ~ identity");
            }
        }
        // Grain is deterministic → replaying it twice is byte-identical.
        let g1 = EditOp::Grain(40).apply(src.clone()).to_rgb8();
        let g2 = EditOp::Grain(40).apply(src.clone()).to_rgb8();
        assert_eq!(g1, g2, "grain must be replay-stable");
        assert_ne!(g1, src.to_rgb8(), "grain changes pixels");
    }

    #[test]
    fn despeckle_removes_a_lone_hot_pixel() {
        let mut buf = ImageBuffer::from_pixel(9, 9, Rgb([100u8, 100, 100]));
        buf.put_pixel(4, 4, Rgb([255, 255, 255])); // a single speckle in a flat field
        let out = EditOp::Despeckle(100).apply(DynamicImage::ImageRgb8(buf)).to_rgb8();
        assert_eq!(out.get_pixel(4, 4).0, [100, 100, 100], "median wipes the lone speckle");
    }

    #[test]
    fn stylize_filters_run_and_change_pixels() {
        let src = DynamicImage::ImageRgb8(ImageBuffer::from_fn(24, 24, |x, y| {
            Rgb([(x * 10) as u8, (y * 10) as u8, ((x + y) * 5) as u8])
        }));
        let base = src.to_rgb8();
        for op in [
            EditOp::OilPaint { style: 1, strength: 100 }, EditOp::OilPaint { style: 7, strength: 100 },
            EditOp::PencilSketch(100), EditOp::Cartoon(100),
            EditOp::Watercolor { style: 1, strength: 100 }, EditOp::Watercolor { style: 9, strength: 100 },
            EditOp::Ink { style: 1, strength: 100 }, EditOp::Ink { style: 2, strength: 100 },
            EditOp::Ink { style: 3, strength: 100 }, EditOp::Ink { style: 4, strength: 100 },
            EditOp::Emboss(100), EditOp::Pixelate(50), EditOp::Blur(80), EditOp::Bloom(80),
            EditOp::Charcoal(100), EditOp::Halftone(100),
            EditOp::FalseColor { style: 1, strength: 100 }, EditOp::FalseColor { style: 2, strength: 100 },
            EditOp::FalseColor { style: 3, strength: 100 },
        ] {
            let out = op.apply(src.clone()).to_rgb8();
            assert_eq!((out.width(), out.height()), (24, 24), "{} keeps dims", op.label());
            assert_ne!(out, base, "{} should change pixels", op.label());
        }
        // Pixelate at strength 0 and any filter at strength 0 are identity (blend = original).
        assert_eq!(EditOp::Pixelate(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::Cartoon(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::OilPaint { style: 5, strength: 0 }.apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::from_tag("sketch"), Some(EditOp::PencilSketch(100)));
    }

    #[test]
    fn creative_looks_behave() {
        let mid = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([100, 150, 200])));
        // Invert is its own inverse.
        let inv = EditOp::Invert.apply(mid.clone()).to_rgb8();
        assert_eq!(inv.get_pixel(0, 0).0, [155, 105, 55]);
        assert_eq!(EditOp::Invert.apply(EditOp::Invert.apply(mid.clone())).to_rgb8(), mid.to_rgb8());
        // Threshold → pure black or white.
        let th = EditOp::Threshold(128).apply(mid.clone()).to_rgb8().get_pixel(0, 0).0;
        assert!(th == [0, 0, 0] || th == [255, 255, 255]);
        // Posterize at full strength → 2 levels per channel (each channel is 0 or 255).
        let po = EditOp::Posterize(100).apply(mid.clone()).to_rgb8().get_pixel(0, 0).0;
        assert!(po.iter().all(|&c| c == 0 || c == 255), "got {po:?}");
        // Strength 0 posterize / solarize are (near) identity.
        assert_eq!(EditOp::Posterize(0).apply(mid.clone()).to_rgb8(), mid.to_rgb8());
        assert_eq!(EditOp::Solarize(0).apply(mid.clone()).to_rgb8(), mid.to_rgb8());
        // Sepia warms it: red channel ends up ≥ blue.
        let se = EditOp::Sepia.apply(mid).to_rgb8().get_pixel(0, 0).0;
        assert!(se[0] >= se[2]);
        assert_eq!(EditOp::from_tag("negative"), Some(EditOp::Invert));
    }

    #[test]
    fn curve_lut_maps_endpoints_and_identity() {
        // Identity curve leaves a gradient unchanged; a lifted-mid curve brightens the midtone.
        let grad = DynamicImage::ImageRgb8(ImageBuffer::from_fn(256, 1, |x, _| Rgb([x as u8; 3])));
        let id = EditOp::Curve { pts: [0, 64, 128, 192, 255] }.apply(grad.clone()).to_rgb8();
        assert_eq!(id.get_pixel(128, 0).0[0], 128);
        assert_eq!(id.get_pixel(255, 0).0[0], 255);
        let lift = EditOp::Curve { pts: [0, 64, 180, 192, 255] }.apply(grad).to_rgb8();
        assert!(lift.get_pixel(128, 0).0[0] > 150, "mid lifted");
    }

    #[test]
    fn clahe_changes_a_low_contrast_image_and_is_noop_at_zero() {
        let src = DynamicImage::ImageRgb8(ImageBuffer::from_fn(32, 32, |x, y| {
            Rgb([(100 + (x ^ y) % 40) as u8; 3]) // low-contrast texture
        }));
        let out = EditOp::Clahe(80).apply(src.clone()).to_rgb8();
        assert_ne!(out, src.to_rgb8(), "CLAHE should change a flat image");
        assert_eq!(EditOp::Clahe(0).apply(src.clone()).to_rgb8(), src.to_rgb8(), "0 = no-op");
        assert_eq!(EditOp::from_tag("clahe"), Some(EditOp::Clahe(60)));
    }

    #[test]
    fn grad_nd_darkens_its_edge_and_radial_hits_the_centre() {
        let flat = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(20, 20, Rgb([200, 200, 200])));
        // Grad ND from the top darkens the top row, leaves the bottom (past the midpoint) untouched.
        let g = EditOp::GradND { dir: 0, strength: 60 }.apply(flat.clone()).to_rgb8();
        assert!(g.get_pixel(10, 0).0[0] < 190, "top edge darkened");
        assert_eq!(g.get_pixel(10, 19).0[0], 200, "bottom untouched");
        // Radial burn darkens the centre more than the corner.
        let r = EditOp::Radial(60).apply(flat).to_rgb8();
        assert!(r.get_pixel(10, 10).0[0] < 180, "centre burned");
        assert!(r.get_pixel(10, 10).0[0] < r.get_pixel(0, 0).0[0], "centre darker than corner");
        assert_eq!(EditOp::from_tag("dodge"), Some(EditOp::Radial(-30)));
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
