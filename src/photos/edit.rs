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
#[derive(Debug, Clone, PartialEq)]
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
    /// Selective luminance (HSL per-band): brighten / darken pixels near `hue` by `lum` (−100..100),
    /// gated by saturation so neutral tones aren't shifted.
    SelectiveLum { hue: i32, lum: i32 },
    /// Gray-point white balance (eyedropper): sample the pixel at (`x`, `y`) in per-mille of the
    /// image and scale each channel so that pixel becomes neutral grey. Replayable (re-samples the
    /// pristine image on rebuild).
    GrayPointWB { x: i32, y: i32 },
    /// Film grain: add deterministic monochrome noise (positive only).
    Grain(i32),
    /// Weight-free **de-slop** (RFC QUALITY-8 P5): the naturalize *Photo* recipe scaled by
    /// strength (0–100). Softens the AI/computer "slop" tells — oversaturation, plastic
    /// smoothness, flat contrast — while preserving the original. Deterministic → replay-safe.
    Naturalize(i32),
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
    /// Kelvin white balance — `amount` −100..100 (positive = warmer / lower Kelvin).
    Kelvin(i32),
    /// Gradient map — recolour by luma through a preset ramp; `style` 1 warm · 2 cyanotype · 3 fire ·
    /// 4 teal-orange; `strength` 0..100.
    GradientMap { style: i32, strength: i32 },
    /// Cross-hatch (ink hatching by tone); `strength` 0..100.
    Crosshatch(i32),
    /// Crystallize (Voronoi): flat-fill jittered polygonal cells with their average colour (a
    /// stained-glass / low-poly look); `strength` 0..100 sets the cell size (0 = identity).
    Crystallize(i32),
    /// Bilateral denoise: edge-preserving smoothing (a 5×5 range+spatial weighted average) that wipes
    /// noise while keeping edges crisp — cleaner than the median despeckle. `strength` 0..100.
    Bilateral(i32),
    /// Tilt-shift / miniature: in-focus band + blur toward top/bottom + saturation pop. `strength` 0..100.
    TiltShift(i32),
    /// Motion blur: directional streak at `angle`° (0 = horizontal); `strength` 0..100 = length.
    MotionBlur { angle: i32, strength: i32 },
    /// Zoom blur: radial streak from the centre; `strength` 0..100.
    ZoomBlur(i32),
    /// Spin blur: rotational streak about the centre; `strength` 0..100.
    SpinBlur(i32),
    /// Channel-mixer black & white: weighted mono via `r/g/b` channel weights (normalised by sum).
    ChannelMixerBW { r: i32, g: i32, b: i32 },
    /// Film-negative conversion: invert + per-channel auto-stretch (removes the orange C-41 mask).
    FilmNegative,
    /// Lens distortion correction: radial warp; `amount` > 0 fixes barrel, < 0 fixes pincushion (−100..100).
    LensDistort(i32),
    /// Chromatic-aberration removal: rescale R/B channels radially to remove colour fringing; `strength` 0..100.
    ChromaticAberration(i32),
    /// Local (masked) adjustment: the base adjustment `adjust` (0 exposure · 1 brightness · 2 contrast
    /// · 3 saturation · 4 warmth · 5 vibrance · 6 definition · 7 blur) applied by `amount` through a
    /// mask — `shape` 0 linear (from edge `dir` 0-3) or 1 radial (`dir` 1 = edges). Slider = `amount`.
    LocalAdjust { adjust: i32, amount: i32, shape: i32, dir: i32 },
    /// Brush (masked) adjustment: like [`EditOp::LocalAdjust`] but the mask is **painted** — a union of
    /// soft dabs collected in the pick-mode. `adjust`/`amount` match `LocalAdjust`; each dab is
    /// `[x, y, radius]` in per-mille (x/y of the image, radius of the min dimension), so it replays
    /// exactly on the pristine original. Slider = `amount`.
    BrushAdjust { adjust: i32, amount: i32, dabs: Vec<[i32; 3]> },
    /// Retouch (from the pick-mode): coordinates are per-mille of the image, radius per-mille of the
    /// min dimension — so each replays exactly on the pristine original.
    /// Spot heal: fill a disc from its surroundings (dust / blemish removal).
    SpotHeal { x: i32, y: i32, radius: i32 },
    /// Clone stamp: copy a disc from source `(sx, sy)` to destination `(dx, dy)`.
    Clone { sx: i32, sy: i32, dx: i32, dy: i32, radius: i32 },
    /// Red-eye removal: neutralise the red pupil glare inside a disc.
    RedEye { x: i32, y: i32, radius: i32 },
    /// Dodge / burn brush: lighten (`amount` > 0) or darken a soft disc.
    DodgeBurn { x: i32, y: i32, radius: i32, amount: i32 },
    /// 4-point perspective rectify: warp the picked quad (TL, TR, BR, BL per-mille) to fill the frame.
    Perspective4 { pts: [[i32; 2]; 4] },
    /// Face polish (auto-retouch): edge-preserving skin smoothing limited to detected face regions —
    /// the mask normally painted by hand, here supplied by the SCRFD face detector. `strength` 0..100;
    /// `faces` holds up to 6 ellipses `(cx, cy, rx, ry)` in per-mille of the image dims (filled once at
    /// creation, so replay is pure geometry — no model reload); `n` is the valid count (`0` = identity).
    FacePolish { strength: i32, faces: [[i32; 4]; 6], n: i32 },
    /// "Better sky" — build a soft sky mask (no AI/no manual masking) and apply a polarizer-like
    /// enhancement (deepen & saturate blue, lift cloud contrast) weighted by it; `amount` 0..100.
    EnhanceSky(i32),
    /// Auto white balance (gray-world): neutralise a colour cast by scaling each channel toward the
    /// overall grey; `amount` 0..100 blends toward the correction.
    AutoWhiteBalance(i32),
    /// Watermark / caption burn-in: draw `text` in the lower-right corner. `font` is an optional path
    /// to a TrueType/OpenType file (else the built-in bitmap font). A **replayable** edit — re-rendered
    /// on rebuild; a missing font file just falls back to the default font (never breaks the replay).
    Watermark { text: String, font: Option<String> },
    /// Apply a `.cube` 3D LUT from `path` (a film-look colour grade). Replayable; a missing / invalid
    /// file is a no-op (identity) so the edit stack never breaks.
    Lut { path: String },
    /// Pixelate / mosaic. `strength` 0 = none … 100 = large blocks.
    Pixelate(i32),
    /// Border / letterbox: pad to aspect `aspect_w:aspect_h` (0,0 = even frame) with `mode` 0 black,
    /// 1 white, 2 blur-extend.
    Border { aspect_w: i32, aspect_h: i32, mode: i32 },
    /// Circle crop: keep a centred circle, fill outside (`mode` 0 black, 1 white).
    CropCircle(i32),
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
            EditOp::SelectiveLum { hue, lum } => adjust::selective_lum(&img, hue, lum),
            EditOp::GrayPointWB { x, y } => adjust::gray_point_wb(&img, x, y),
            EditOp::Grain(v) => adjust::grain(&img, v),
            EditOp::Naturalize(v) => naturalize_op(&img, v),
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
            EditOp::Kelvin(v) => adjust::kelvin(&img, v),
            EditOp::GradientMap { style, strength } => {
                let f = adjust::gradient_map(&img, style);
                adjust::blend(&img, f, strength)
            }
            EditOp::Crosshatch(s) => {
                let f = adjust::crosshatch(&img);
                adjust::blend(&img, f, s)
            }
            EditOp::FacePolish { strength, faces, n } => {
                let k = (n.max(0) as usize).min(faces.len());
                adjust::face_polish(&img, strength, &faces[..k])
            }
            EditOp::Crystallize(v) => adjust::crystallize(&img, v),
            EditOp::Bilateral(v) => adjust::bilateral(&img, v),
            EditOp::TiltShift(v) => adjust::tilt_shift(&img, v),
            EditOp::MotionBlur { angle, strength } => adjust::motion_blur(&img, angle, strength),
            EditOp::ZoomBlur(v) => adjust::zoom_blur(&img, v),
            EditOp::SpinBlur(v) => adjust::spin_blur(&img, v),
            EditOp::ChannelMixerBW { r, g, b } => adjust::channel_mixer_bw(&img, r, g, b),
            EditOp::FilmNegative => adjust::film_negative(&img),
            EditOp::LensDistort(v) => lens_distort(&img, v),
            EditOp::ChromaticAberration(v) => chromatic_aberration(&img, v),
            EditOp::LocalAdjust { adjust, amount, shape, dir } => local_adjust(&img, adjust, amount, shape, dir),
            EditOp::BrushAdjust { adjust, amount, dabs } => brush_adjust(&img, adjust, amount, &dabs),
            EditOp::SpotHeal { x, y, radius } => adjust::spot_heal(&img, x, y, radius),
            EditOp::Clone { sx, sy, dx, dy, radius } => adjust::clone_stamp(&img, sx, sy, dx, dy, radius),
            EditOp::RedEye { x, y, radius } => adjust::red_eye(&img, x, y, radius),
            EditOp::DodgeBurn { x, y, radius, amount } => adjust::dodge_burn(&img, x, y, radius, amount),
            EditOp::Perspective4 { pts } => super::homography::rectify(&img, pts),
            EditOp::EnhanceSky(v) => adjust::enhance_sky(&img, v),
            EditOp::AutoWhiteBalance(v) => adjust::auto_white_balance(&img, v),
            EditOp::Watermark { text, font } => watermark(&img, &text, font.as_deref()),
            EditOp::Lut { path } => match super::lut::load_cube(std::path::Path::new(&path)) {
                Ok(l) => super::lut::apply(&img, &l),
                Err(_) => img,
            },
            EditOp::Pixelate(v) => adjust::pixelate(&img, v),
            EditOp::Border { aspect_w, aspect_h, mode } => border(&img, aspect_w, aspect_h, mode),
            EditOp::CropCircle(mode) => crop_circle(&img, mode),
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
    pub fn scalar(&self) -> Option<i32> {
        Some(match self {
            EditOp::Brightness(v) | EditOp::Contrast(v) | EditOp::Exposure(v) | EditOp::Brilliance(v)
            | EditOp::Highlights(v) | EditOp::Midrange(v) | EditOp::Shadows(v) | EditOp::Blackpoint(v)
            | EditOp::Saturation(v) | EditOp::Vibrance(v) | EditOp::Warmth(v) | EditOp::Tint(v)
            | EditOp::Definition(v) | EditOp::Sharpen(v) | EditOp::NoiseReduction(v) | EditOp::Dehaze(v)
            | EditOp::HueRotate(v) | EditOp::SplitTone(v) | EditOp::Vignette(v) | EditOp::Grain(v)
            | EditOp::Naturalize(v)
            | EditOp::Despeckle(v) | EditOp::Radial(v) | EditOp::Clahe(v) | EditOp::Posterize(v)
            | EditOp::Solarize(v) | EditOp::Pixelate(v) | EditOp::PencilSketch(v) | EditOp::Cartoon(v)
            | EditOp::Emboss(v) | EditOp::Blur(v) | EditOp::Bloom(v) | EditOp::Charcoal(v)
            | EditOp::Halftone(v) | EditOp::Kelvin(v) | EditOp::Crosshatch(v)
            | EditOp::EnhanceSky(v) | EditOp::AutoWhiteBalance(v) | EditOp::Crystallize(v)
            | EditOp::Bilateral(v) | EditOp::TiltShift(v) | EditOp::ZoomBlur(v) | EditOp::SpinBlur(v)
            | EditOp::LensDistort(v) | EditOp::ChromaticAberration(v) => *v,
            EditOp::GradND { strength, .. }
            | EditOp::OilPaint { strength, .. }
            | EditOp::Watercolor { strength, .. }
            | EditOp::Ink { strength, .. }
            | EditOp::FalseColor { strength, .. }
            | EditOp::GradientMap { strength, .. }
            | EditOp::FacePolish { strength, .. }
            | EditOp::MotionBlur { strength, .. } => *strength,
            EditOp::LocalAdjust { amount, .. } => *amount,
            EditOp::BrushAdjust { amount, .. } => *amount,
            EditOp::Keystone { amount, .. } => *amount,
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
            EditOp::Naturalize(_) => EditOp::Naturalize(v),
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
            EditOp::Kelvin(_) => EditOp::Kelvin(v),
            EditOp::Crosshatch(_) => EditOp::Crosshatch(v),
            EditOp::Crystallize(_) => EditOp::Crystallize(v),
            EditOp::Bilateral(_) => EditOp::Bilateral(v),
            EditOp::TiltShift(_) => EditOp::TiltShift(v),
            EditOp::ZoomBlur(_) => EditOp::ZoomBlur(v),
            EditOp::SpinBlur(_) => EditOp::SpinBlur(v),
            EditOp::MotionBlur { angle, .. } => EditOp::MotionBlur { angle, strength: v },
            EditOp::LensDistort(_) => EditOp::LensDistort(v),
            EditOp::ChromaticAberration(_) => EditOp::ChromaticAberration(v),
            EditOp::LocalAdjust { adjust, shape, dir, .. } => EditOp::LocalAdjust { adjust, amount: v, shape, dir },
            EditOp::BrushAdjust { adjust, dabs, .. } => EditOp::BrushAdjust { adjust, amount: v, dabs },
            EditOp::EnhanceSky(_) => EditOp::EnhanceSky(v),
            EditOp::AutoWhiteBalance(_) => EditOp::AutoWhiteBalance(v),
            EditOp::FacePolish { faces, n, .. } => EditOp::FacePolish { strength: v, faces, n },
            EditOp::GradientMap { style, .. } => EditOp::GradientMap { style, strength: v },
            other => other,
        }
    }

    /// `(min, max, step)` for the slider — most tonal/colour ops are bipolar ±100; hue is ±180;
    /// the effect-only ops (grain/denoise/despeckle/dehaze) are one-sided 0..100.
    pub fn scalar_range(&self) -> (i32, i32, i32) {
        match self {
            EditOp::HueRotate(_) => (-180, 180, 5),
            EditOp::NoiseReduction(_) | EditOp::Grain(_) | EditOp::Naturalize(_) | EditOp::Despeckle(_) | EditOp::Dehaze(_)
            | EditOp::Clahe(_) | EditOp::Posterize(_) | EditOp::Solarize(_) | EditOp::Pixelate(_)
            | EditOp::PencilSketch(_) | EditOp::Cartoon(_) | EditOp::Emboss(_) | EditOp::Blur(_)
            | EditOp::Bloom(_) | EditOp::Charcoal(_) | EditOp::Halftone(_) | EditOp::OilPaint { .. }
            | EditOp::Watercolor { .. } | EditOp::Ink { .. } | EditOp::FalseColor { .. }
            | EditOp::Crosshatch(_) | EditOp::GradientMap { .. }
            | EditOp::EnhanceSky(_) | EditOp::AutoWhiteBalance(_) | EditOp::Crystallize(_)
            | EditOp::Bilateral(_) | EditOp::FacePolish { .. }
            | EditOp::TiltShift(_) | EditOp::ZoomBlur(_) | EditOp::SpinBlur(_)
            | EditOp::MotionBlur { .. } | EditOp::ChromaticAberration(_) => (0, 100, 5),
            EditOp::LocalAdjust { adjust: 7, .. } => (0, 100, 5), // local blur is one-sided
            EditOp::BrushAdjust { adjust: 7, .. } => (0, 100, 5), // brushed blur is one-sided
            _ => (-100, 100, 5),
        }
    }

    /// A short human label for the status line / edit menu.
    pub fn label(&self) -> String {
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
            EditOp::Straighten(t) => format!("straighten {:.1}°", *t as f32 / 10.0),
            EditOp::Keystone { axis, .. } => {
                format!("keystone {}", if *axis == 0 { "vertical" } else { "horizontal" })
            }
            EditOp::Vignette(_) => "vignette".into(),
            EditOp::Levels { black, white, gamma } => {
                format!("levels {black}/{white}·γ{:.2}", *gamma as f32 / 100.0)
            }
            EditOp::Dehaze(_) => "dehaze".into(),
            EditOp::HueRotate(d) => format!("hue {d:+}°"),
            EditOp::SplitTone(_) => "split-tone".into(),
            EditOp::SelectiveColor { hue, sat } => format!("selective colour {hue}°·{sat:+}"),
            EditOp::SelectiveLum { hue, lum } => format!("selective lum {hue}°·{lum:+}"),
            EditOp::GrayPointWB { .. } => "gray-point WB".into(),
            EditOp::Grain(_) => "film grain".into(),
            EditOp::Naturalize(_) => "naturalize (de-slop)".into(),
            EditOp::Despeckle(_) => "despeckle".into(),
            EditOp::GradND { dir, .. } => {
                format!("grad ND {}", ["top", "bottom", "left", "right"].get(*dir as usize).unwrap_or(&"top"))
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
            EditOp::OilPaint { style, .. } => format!("oil paint {}", (*style).clamp(1, 10)),
            EditOp::PencilSketch(_) => "pencil sketch".into(),
            EditOp::Cartoon(_) => "cartoon".into(),
            EditOp::Watercolor { style, .. } => format!("watercolour {}", (*style).clamp(1, 10)),
            EditOp::Ink { style, .. } => format!("ink ({})", adjust::ink_name(*style)),
            EditOp::Emboss(_) => "emboss".into(),
            EditOp::Blur(_) => "blur (soft focus)".into(),
            EditOp::Bloom(_) => "bloom / glow".into(),
            EditOp::Charcoal(_) => "charcoal".into(),
            EditOp::Halftone(_) => "halftone".into(),
            EditOp::FalseColor { style, .. } => format!("false colour ({})", adjust::false_color_name(*style)),
            EditOp::Kelvin(_) => "Kelvin white balance".into(),
            EditOp::GradientMap { .. } => "gradient map".into(),
            EditOp::Crosshatch(_) => "cross-hatch".into(),
            EditOp::Crystallize(_) => "crystallize".into(),
            EditOp::Bilateral(_) => "bilateral denoise".into(),
            EditOp::TiltShift(_) => "tilt-shift".into(),
            EditOp::MotionBlur { angle, .. } => format!("motion blur {angle}°"),
            EditOp::ZoomBlur(_) => "zoom blur".into(),
            EditOp::SpinBlur(_) => "spin blur".into(),
            EditOp::ChannelMixerBW { .. } => "B&W channel mixer".into(),
            EditOp::FilmNegative => "film negative".into(),
            EditOp::LensDistort(_) => "lens distortion".into(),
            EditOp::ChromaticAberration(_) => "chromatic aberration".into(),
            EditOp::LocalAdjust { adjust, shape, dir, .. } => {
                let m = if *shape == 1 {
                    if *dir == 1 { "radial-edges" } else { "radial" }
                } else {
                    ["top", "bottom", "left", "right"].get(*dir as usize).copied().unwrap_or("top")
                };
                format!("local {} ({m})", local_adjust_name(*adjust))
            }
            EditOp::BrushAdjust { adjust, dabs, .. } => {
                format!("brush {} ({} dab{})", local_adjust_name(*adjust), dabs.len(), if dabs.len() == 1 { "" } else { "s" })
            }
            EditOp::SpotHeal { .. } => "spot heal".into(),
            EditOp::Clone { .. } => "clone stamp".into(),
            EditOp::RedEye { .. } => "red-eye removal".into(),
            EditOp::DodgeBurn { amount, .. } => if *amount >= 0 { "dodge".into() } else { "burn".into() },
            EditOp::Perspective4 { .. } => "perspective rectify".into(),
            EditOp::FacePolish { .. } => "face polish".into(),
            EditOp::EnhanceSky(_) => "enhance sky".into(),
            EditOp::AutoWhiteBalance(_) => "auto white balance".into(),
            EditOp::Watermark { .. } => "watermark".into(),
            EditOp::Lut { .. } => "LUT grade".into(),
            EditOp::Pixelate(_) => "pixelate".into(),
            EditOp::Border { aspect_w, aspect_h, .. } => {
                if *aspect_w <= 0 || *aspect_h <= 0 {
                    "border (frame)".into()
                } else {
                    format!("letterbox {aspect_w}:{aspect_h}")
                }
            }
            EditOp::CropCircle(_) => "circle crop".into(),
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
            EditOp::SelectiveLum { hue, lum } => {
                params.insert("hue".into(), serde_json::json!(hue));
                params.insert("lum".into(), serde_json::json!(lum));
                "selective_lum"
            }
            EditOp::GrayPointWB { x, y } => {
                params.insert("x".into(), serde_json::json!(x));
                params.insert("y".into(), serde_json::json!(y));
                "gray_point_wb"
            }
            EditOp::Grain(v) => val_op(&mut params, v, "grain"),
            EditOp::Naturalize(v) => val_op(&mut params, v, "naturalize"),
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
            EditOp::Kelvin(v) => val_op(&mut params, v, "kelvin"),
            EditOp::GradientMap { style, strength } => style_op(&mut params, style, strength, "gradient_map"),
            EditOp::Crosshatch(v) => val_op(&mut params, v, "crosshatch"),
            EditOp::Crystallize(v) => val_op(&mut params, v, "crystallize"),
            EditOp::Bilateral(v) => val_op(&mut params, v, "bilateral"),
            EditOp::TiltShift(v) => val_op(&mut params, v, "tilt_shift"),
            EditOp::ZoomBlur(v) => val_op(&mut params, v, "zoom_blur"),
            EditOp::SpinBlur(v) => val_op(&mut params, v, "spin_blur"),
            EditOp::MotionBlur { angle, strength } => {
                params.insert("angle".into(), serde_json::json!(angle));
                params.insert("strength".into(), serde_json::json!(strength));
                "motion_blur"
            }
            EditOp::ChannelMixerBW { r, g, b } => {
                params.insert("r".into(), serde_json::json!(r));
                params.insert("g".into(), serde_json::json!(g));
                params.insert("b".into(), serde_json::json!(b));
                "channel_mixer_bw"
            }
            EditOp::FilmNegative => "film_negative",
            EditOp::LensDistort(v) => val_op(&mut params, v, "lens_distort"),
            EditOp::ChromaticAberration(v) => val_op(&mut params, v, "chromatic_aberration"),
            EditOp::LocalAdjust { adjust, amount, shape, dir } => {
                params.insert("adjust".into(), serde_json::json!(adjust));
                params.insert("amount".into(), serde_json::json!(amount));
                params.insert("shape".into(), serde_json::json!(shape));
                params.insert("dir".into(), serde_json::json!(dir));
                "local_adjust"
            }
            EditOp::BrushAdjust { adjust, amount, dabs } => {
                params.insert("adjust".into(), serde_json::json!(adjust));
                params.insert("amount".into(), serde_json::json!(amount));
                params.insert("dabs".into(), serde_json::json!(dabs));
                "brush_adjust"
            }
            EditOp::SpotHeal { x, y, radius } => {
                params.insert("x".into(), serde_json::json!(x));
                params.insert("y".into(), serde_json::json!(y));
                params.insert("radius".into(), serde_json::json!(radius));
                "spot_heal"
            }
            EditOp::Clone { sx, sy, dx, dy, radius } => {
                params.insert("sx".into(), serde_json::json!(sx));
                params.insert("sy".into(), serde_json::json!(sy));
                params.insert("dx".into(), serde_json::json!(dx));
                params.insert("dy".into(), serde_json::json!(dy));
                params.insert("radius".into(), serde_json::json!(radius));
                "clone"
            }
            EditOp::RedEye { x, y, radius } => {
                params.insert("x".into(), serde_json::json!(x));
                params.insert("y".into(), serde_json::json!(y));
                params.insert("radius".into(), serde_json::json!(radius));
                "red_eye"
            }
            EditOp::DodgeBurn { x, y, radius, amount } => {
                params.insert("x".into(), serde_json::json!(x));
                params.insert("y".into(), serde_json::json!(y));
                params.insert("radius".into(), serde_json::json!(radius));
                params.insert("amount".into(), serde_json::json!(amount));
                "dodge_burn"
            }
            EditOp::Perspective4 { pts } => {
                for (i, p) in pts.iter().enumerate() {
                    params.insert(format!("p{i}"), serde_json::json!(p));
                }
                "perspective4"
            }
            EditOp::EnhanceSky(v) => val_op(&mut params, v, "enhance_sky"),
            EditOp::AutoWhiteBalance(v) => val_op(&mut params, v, "auto_wb"),
            EditOp::Watermark { text, font } => {
                params.insert("text".into(), serde_json::json!(text));
                if let Some(f) = font {
                    params.insert("font".into(), serde_json::json!(f));
                }
                "watermark"
            }
            EditOp::Lut { path } => {
                params.insert("path".into(), serde_json::json!(path));
                "lut"
            }
            EditOp::FacePolish { strength, faces, n } => {
                params.insert("strength".into(), serde_json::json!(strength));
                params.insert("n".into(), serde_json::json!(n));
                for (i, f) in faces.iter().enumerate() {
                    params.insert(format!("f{i}"), serde_json::json!(f));
                }
                "face_polish"
            }
            EditOp::Pixelate(v) => val_op(&mut params, v, "pixelate"),
            EditOp::Border { aspect_w, aspect_h, mode } => {
                params.insert("aspect_w".into(), serde_json::json!(aspect_w));
                params.insert("aspect_h".into(), serde_json::json!(aspect_h));
                params.insert("mode".into(), serde_json::json!(mode));
                "border"
            }
            EditOp::CropCircle(v) => val_op(&mut params, v, "crop_circle"),
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
            "naturalize" | "deslop" | "de_slop" => EditOp::Naturalize(60),
            "despeckle" | "median" | "remove_speckle" => EditOp::Despeckle(55),
            "burn" => EditOp::Radial(30),
            "dodge" => EditOp::Radial(-30),
            "grad_nd" | "graduated_nd" | "nd_grad" => EditOp::GradND { dir: 0, strength: 30 },
            "keystone" | "fix_verticals" | "keystone_v" => EditOp::Keystone { axis: 0, amount: 30 },
            "border" | "frame" => EditOp::Border { aspect_w: 0, aspect_h: 0, mode: 1 },
            "circle" | "circle_crop" => EditOp::CropCircle(0),
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
            "kelvin" | "temperature" | "white_balance" => EditOp::Kelvin(0),
            "gradient_map" | "gradientmap" => EditOp::GradientMap { style: 1, strength: 100 },
            "crosshatch" | "hatch" => EditOp::Crosshatch(100),
            "crystallize" | "voronoi" | "low_poly" | "stained_glass" => EditOp::Crystallize(100),
            "bilateral" | "denoise_edge" | "smart_denoise" => EditOp::Bilateral(80),
            "gray_point_wb" | "eyedropper" | "neutral_point" => EditOp::GrayPointWB { x: 500, y: 500 },
            "tilt_shift" | "tiltshift" | "miniature" => EditOp::TiltShift(70),
            "motion_blur" | "motion" => EditOp::MotionBlur { angle: 0, strength: 60 },
            "zoom_blur" | "zoom" => EditOp::ZoomBlur(60),
            "spin_blur" | "spin" => EditOp::SpinBlur(60),
            "bw_mixer" | "channel_mixer" | "mono_mixer" => EditOp::ChannelMixerBW { r: 60, g: 30, b: 10 },
            "film_negative" | "c41" | "scan_negative" => EditOp::FilmNegative,
            "lens_distort" | "distortion" | "defish" => EditOp::LensDistort(0),
            "chromatic_aberration" | "defringe" | "ca" => EditOp::ChromaticAberration(60),
            "enhance_sky" | "better_sky" | "sky" => EditOp::EnhanceSky(100),
            "auto_wb" | "auto_white_balance" | "gray_world" => EditOp::AutoWhiteBalance(100),
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
            "selective_lum" => EditOp::SelectiveLum { hue: iv("hue", 0), lum: iv("lum", 0) },
            "gray_point_wb" => EditOp::GrayPointWB { x: iv("x", 500), y: iv("y", 500) },
            "grain" => EditOp::Grain(val()),
            "naturalize" => EditOp::Naturalize(val()),
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
            "kelvin" => EditOp::Kelvin(val()),
            "gradient_map" => EditOp::GradientMap { style: iv("style", 1), strength: iv("strength", 100) },
            "crosshatch" => EditOp::Crosshatch(iv("value", 100)),
            "crystallize" => EditOp::Crystallize(iv("value", 100)),
            "bilateral" => EditOp::Bilateral(iv("value", 100)),
            "tilt_shift" => EditOp::TiltShift(iv("value", 100)),
            "zoom_blur" => EditOp::ZoomBlur(iv("value", 100)),
            "spin_blur" => EditOp::SpinBlur(iv("value", 100)),
            "motion_blur" => EditOp::MotionBlur { angle: iv("angle", 0), strength: iv("strength", 100) },
            "channel_mixer_bw" => EditOp::ChannelMixerBW { r: iv("r", 40), g: iv("g", 40), b: iv("b", 20) },
            "film_negative" => EditOp::FilmNegative,
            "lens_distort" => EditOp::LensDistort(val()),
            "chromatic_aberration" => EditOp::ChromaticAberration(iv("value", 60)),
            "local_adjust" => EditOp::LocalAdjust {
                adjust: iv("adjust", 0),
                amount: iv("amount", 0),
                shape: iv("shape", 0),
                dir: iv("dir", 0),
            },
            "brush_adjust" => {
                let dabs = e
                    .params
                    .get("dabs")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|d| d.as_array())
                            .map(|d| {
                                let g = |i: usize| d.get(i).and_then(|x| x.as_i64()).unwrap_or(0) as i32;
                                [g(0), g(1), g(2)]
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                EditOp::BrushAdjust { adjust: iv("adjust", 0), amount: iv("amount", 0), dabs }
            }
            "spot_heal" => EditOp::SpotHeal { x: iv("x", 500), y: iv("y", 500), radius: iv("radius", 60) },
            "clone" => EditOp::Clone {
                sx: iv("sx", 400), sy: iv("sy", 500), dx: iv("dx", 600), dy: iv("dy", 500), radius: iv("radius", 60),
            },
            "red_eye" => EditOp::RedEye { x: iv("x", 500), y: iv("y", 500), radius: iv("radius", 40) },
            "dodge_burn" => EditOp::DodgeBurn {
                x: iv("x", 500), y: iv("y", 500), radius: iv("radius", 120), amount: iv("amount", 30),
            },
            "perspective4" => {
                let mut pts = [[0i32; 2]; 4];
                for (i, pt) in pts.iter_mut().enumerate() {
                    if let Some(arr) = e.params.get(&format!("p{i}")).and_then(|v| v.as_array()) {
                        pt[0] = arr.first().and_then(|x| x.as_i64()).unwrap_or(0) as i32;
                        pt[1] = arr.get(1).and_then(|y| y.as_i64()).unwrap_or(0) as i32;
                    }
                }
                EditOp::Perspective4 { pts }
            }
            "enhance_sky" => EditOp::EnhanceSky(iv("value", 100)),
            "auto_wb" => EditOp::AutoWhiteBalance(iv("value", 100)),
            "watermark" => EditOp::Watermark {
                text: e.params.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                font: e.params.get("font").and_then(|v| v.as_str()).map(String::from),
            },
            "lut" => EditOp::Lut {
                path: e.params.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            },
            "face_polish" => {
                let mut faces = [[0i32; 4]; 6];
                for (i, face) in faces.iter_mut().enumerate() {
                    if let Some(arr) = e.params.get(&format!("f{i}")).and_then(|v| v.as_array()) {
                        for (j, slot) in face.iter_mut().enumerate() {
                            *slot = arr.get(j).and_then(|x| x.as_i64()).unwrap_or(0) as i32;
                        }
                    }
                }
                EditOp::FacePolish { strength: iv("strength", 100), n: iv("n", 0), faces }
            }
            "pixelate" => EditOp::Pixelate(val()),
            "border" => EditOp::Border {
                aspect_w: iv("aspect_w", 0),
                aspect_h: iv("aspect_h", 0),
                mode: iv("mode", 1),
            },
            "crop_circle" => EditOp::CropCircle(val()),
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

    /// Selective luminance (HSL per-band): brighten (`lum > 0`) / darken pixels near `hue`, gated by
    /// both hue proximity and saturation so neutral tones stay put.
    pub fn selective_lum(img: &DynamicImage, hue: i32, lum: i32) -> DynamicImage {
        let target = hue as f32;
        let amt = lum as f32 / 100.0;
        map_rgb(img, |r, g, b| {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let mut dh = (h - target).abs() % 360.0;
            if dh > 180.0 {
                dh = 360.0 - dh;
            }
            let wh = 1.0 - smoothstep(20.0, 80.0, dh);
            let ws = smoothstep(0.12, 0.35, s); // ignore near-grey pixels
            let nv = (v + amt * wh * ws * 0.5).clamp(0.0, 1.0);
            hsv_to_rgb(h, s, nv)
        })
    }

    /// Gray-point white balance: sample the pixel at (`x`, `y`) per-mille and scale each channel so it
    /// becomes neutral grey (its own luma). A grey point picked on a neutral surface removes the cast.
    pub fn gray_point_wb(img: &DynamicImage, x: i32, y: i32) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let px = ((x.clamp(0, 1000) as u32 * (w - 1).max(1)) / 1000).min(w - 1);
        let py = ((y.clamp(0, 1000) as u32 * (h - 1).max(1)) / 1000).min(h - 1);
        let s = rgb.get_pixel(px, py).0;
        let (sr, sg, sb) = (s[0] as f32, s[1] as f32, s[2] as f32);
        let gray = (sr + sg + sb) / 3.0;
        // Gains that map the sampled colour to its grey; guard against a black/blown sample.
        let gain = |c: f32| if c < 1.0 { 1.0 } else { (gray / c).clamp(0.3, 3.0) };
        let (kr, kg, kb) = (gain(sr), gain(sg), gain(sb));
        map_rgb(img, |r, g, b| [r * kr, g * kg, b * kb])
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

    /// Build a per-pixel mask (0..1) for a local adjustment. `shape` 0 = linear gradient from edge
    /// `dir` (0 top · 1 bottom · 2 left · 3 right); `shape` 1 = centred radial (`dir` 1 inverts it to
    /// favour the edges, like a vignette region).
    pub fn local_mask(img: &DynamicImage, shape: i32, dir: i32) -> Vec<f32> {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let (wf, hf) = ((w.max(1) - 1) as f32, (h.max(1) - 1) as f32);
        let mut m = vec![0f32; (w * h) as usize];
        for y in 0..h {
            let fy = y as f32 / hf;
            for x in 0..w {
                let fx = x as f32 / wf;
                let v = if shape == 1 {
                    let d = (((fx - 0.5) * 2.0).powi(2) + ((fy - 0.5) * 2.0).powi(2)).sqrt();
                    let base = 1.0 - smoothstep(0.2, 0.95, d);
                    if dir == 1 { 1.0 - base } else { base }
                } else {
                    match dir {
                        1 => smoothstep(0.1, 0.9, fy),       // from bottom
                        2 => 1.0 - smoothstep(0.1, 0.9, fx), // from left
                        3 => smoothstep(0.1, 0.9, fx),       // from right
                        _ => 1.0 - smoothstep(0.1, 0.9, fy), // from top
                    }
                };
                m[(y * w + x) as usize] = v.clamp(0.0, 1.0);
            }
        }
        m
    }

    /// Painted-brush mask: the union of soft circular dabs. Each dab is `[x, y, radius]` in per-mille
    /// (x/y of the width/height, radius of the min dimension). The falloff is a smooth cosine from the
    /// centre (full) to the edge (zero); overlapping dabs take the max. Returns a `w·h` mask in [0,1].
    pub fn brush_mask(img: &DynamicImage, dabs: &[[i32; 3]]) -> Vec<f32> {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let mut m = vec![0f32; (w * h) as usize];
        let mind = w.min(h).max(1) as f32;
        for d in dabs {
            let cx = d[0] as f32 / 1000.0 * w as f32;
            let cy = d[1] as f32 / 1000.0 * h as f32;
            let r = (d[2] as f32 / 1000.0 * mind).max(1.0);
            // Bound the work to the dab's box.
            let (x0, x1) = ((cx - r).floor().max(0.0) as u32, (cx + r).ceil().min(w as f32) as u32);
            let (y0, y1) = ((cy - r).floor().max(0.0) as u32, (cy + r).ceil().min(h as f32) as u32);
            for y in y0..y1 {
                for x in x0..x1 {
                    let dist = (((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt()) / r;
                    if dist < 1.0 {
                        // Cosine falloff: 1 at centre, 0 at the edge (smooth, no hard rim).
                        let v = 0.5 * (1.0 + (dist * std::f32::consts::PI).cos());
                        let idx = (y * w + x) as usize;
                        if v > m[idx] {
                            m[idx] = v;
                        }
                    }
                }
            }
        }
        m
    }

    /// Blend `adjusted` over `orig` per-pixel by `mask` (same-size RGB): `out = orig·(1−m) + adjusted·m`.
    pub fn blend_masked(orig: &DynamicImage, adjusted: &DynamicImage, mask: &[f32]) -> DynamicImage {
        let o = orig.to_rgb8();
        let a = adjusted.to_rgb8();
        let (w, h) = (o.width(), o.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let m = mask.get((y * w + x) as usize).copied().unwrap_or(0.0);
                let op = o.get_pixel(x, y).0;
                let ap = a.get_pixel(x, y).0;
                out.put_pixel(x, y, Rgb(std::array::from_fn(|c| (op[c] as f32 * (1.0 - m) + ap[c] as f32 * m) as u8)));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Tilt-shift / miniature: keep a horizontal band in focus, blur increasingly toward the top and
    /// bottom, and add a saturation/contrast pop (the "toy model" look). `strength` 0..100.
    pub fn tilt_shift(img: &DynamicImage, strength: i32) -> DynamicImage {
        let s = strength.clamp(0, 100);
        if s == 0 {
            return img.clone();
        }
        let t = s as f32 / 100.0;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let blurred = image::imageops::blur(&rgb, 1.0 + t * 8.0);
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            let fy = if h > 1 { y as f32 / (h - 1) as f32 } else { 0.5 };
            let m = smoothstep(0.12, 0.34, (fy - 0.5).abs()); // in-focus band ±12 %, full blur by 34 %
            for x in 0..w {
                let o = rgb.get_pixel(x, y).0;
                let b = blurred.get_pixel(x, y).0;
                let base: [f32; 3] = std::array::from_fn(|i| o[i] as f32 * (1.0 - m) + b[i] as f32 * m);
                // Saturation pop for the miniature effect.
                let yl = luma(base[0], base[1], base[2]);
                let k = 1.0 + 0.22 * t;
                out.put_pixel(x, y, Rgb(std::array::from_fn(|i| enc((yl + (base[i] - yl) * k) / 255.0))));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Average `taps` colour samples along a path: `pos(i)` returns the sample coordinate for tap `i`
    /// (centred at 0). Edge-clamped bilinear-free (nearest) sampling — cheap, fine for creative blurs.
    fn path_blur(rgb: &RgbImage, taps: i32, pos: impl Fn(u32, u32, i32) -> (f32, f32)) -> DynamicImage {
        let (w, h) = (rgb.width(), rgb.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0f32; 3];
                let mut n = 0f32;
                for i in -taps..=taps {
                    let (sx, sy) = pos(x, y, i);
                    let px = (sx.round() as i32).clamp(0, w as i32 - 1) as u32;
                    let py = (sy.round() as i32).clamp(0, h as i32 - 1) as u32;
                    let p = rgb.get_pixel(px, py).0;
                    acc[0] += p[0] as f32;
                    acc[1] += p[1] as f32;
                    acc[2] += p[2] as f32;
                    n += 1.0;
                }
                out.put_pixel(x, y, Rgb(std::array::from_fn(|c| (acc[c] / n) as u8)));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Motion blur: directional streak at `angle` degrees; `strength` 0..100 sets the length.
    pub fn motion_blur(img: &DynamicImage, angle: i32, strength: i32) -> DynamicImage {
        let s = strength.clamp(0, 100);
        if s == 0 {
            return img.clone();
        }
        let rgb = img.to_rgb8();
        let taps = (1.0 + s as f32 / 100.0 * (rgb.width().min(rgb.height()) as f32 / 40.0)).round() as i32;
        let rad = (angle as f32).to_radians();
        let (dx, dy) = (rad.cos(), rad.sin());
        path_blur(&rgb, taps.max(1), |x, y, i| (x as f32 + i as f32 * dx, y as f32 + i as f32 * dy))
    }

    /// Zoom blur: radial streak toward / from the centre; `strength` 0..100 sets the reach.
    pub fn zoom_blur(img: &DynamicImage, strength: i32) -> DynamicImage {
        let s = strength.clamp(0, 100);
        if s == 0 {
            return img.clone();
        }
        let rgb = img.to_rgb8();
        let (cx, cy) = (rgb.width() as f32 / 2.0, rgb.height() as f32 / 2.0);
        let amt = s as f32 / 100.0 * 0.18;
        let taps = 8;
        path_blur(&rgb, taps, |x, y, i| {
            let f = 1.0 + (i as f32 / taps as f32) * amt;
            (cx + (x as f32 - cx) * f, cy + (y as f32 - cy) * f)
        })
    }

    /// Spin blur: rotational streak about the centre; `strength` 0..100 sets the arc.
    pub fn spin_blur(img: &DynamicImage, strength: i32) -> DynamicImage {
        let s = strength.clamp(0, 100);
        if s == 0 {
            return img.clone();
        }
        let rgb = img.to_rgb8();
        let (cx, cy) = (rgb.width() as f32 / 2.0, rgb.height() as f32 / 2.0);
        let max_a = s as f32 / 100.0 * 0.20; // radians at the extreme tap
        let taps = 8;
        path_blur(&rgb, taps, |x, y, i| {
            let a = (i as f32 / taps as f32) * max_a;
            let (vx, vy) = (x as f32 - cx, y as f32 - cy);
            (cx + vx * a.cos() - vy * a.sin(), cy + vx * a.sin() + vy * a.cos())
        })
    }

    /// Channel-mixer black & white: a weighted mono conversion. Weights `wr/wg/wb` (any integers) are
    /// normalised by their sum — e.g. a red-heavy mix darkens skies, a green-heavy mix flatters skin.
    pub fn channel_mixer_bw(img: &DynamicImage, wr: i32, wg: i32, wb: i32) -> DynamicImage {
        let sum = (wr + wg + wb).abs().max(1) as f32;
        let (fr, fg, fb) = (wr as f32 / sum, wg as f32 / sum, wb as f32 / sum);
        map_rgb(img, |r, g, b| {
            let y = (fr * r + fg * g + fb * b).clamp(0.0, 1.0);
            [y, y, y]
        })
    }

    /// Film-negative conversion: invert, then per-channel auto-stretch — the independent channel
    /// stretch removes the orange C-41 base mask, turning a scanned negative into a positive.
    pub fn film_negative(img: &DynamicImage) -> DynamicImage {
        auto_enhance(&invert(img))
    }

    /// Map per-mille `(x, y)` + per-mille `radius` (of the min dimension) to pixel geometry.
    fn disc_geom(w: u32, h: u32, x: i32, y: i32, radius: i32) -> (f32, f32, f32) {
        let cx = x as f32 / 1000.0 * (w.max(1) - 1) as f32;
        let cy = y as f32 / 1000.0 * (h.max(1) - 1) as f32;
        let r = (radius.max(1) as f32 / 1000.0 * w.min(h) as f32).max(1.0);
        (cx, cy, r)
    }

    /// Spot heal: fill a disc at `(x, y)` (per-mille) of `radius` by interpolating the colour from its
    /// four boundary points — removes dust / blemishes on reasonably smooth areas. Feathered edge.
    pub fn spot_heal(img: &DynamicImage, x: i32, y: i32, radius: i32) -> DynamicImage {
        let mut rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let (cx, cy, r) = disc_geom(w, h, x, y, radius);
        let samp = |xx: f32, yy: f32| super::bilinear(&rgb, xx, yy).0;
        let (bt, bb, bl, br) = (samp(cx, cy - r), samp(cx, cy + r), samp(cx - r, cy), samp(cx + r, cy));
        let (x0, y0) = ((cx - r).floor().max(0.0) as u32, (cy - r).floor().max(0.0) as u32);
        let (x1, y1) = (((cx + r).ceil() as u32).min(w - 1), ((cy + r).ceil() as u32).min(h - 1));
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (dx, dy) = (px as f32 - cx, py as f32 - cy);
                let d = (dx * dx + dy * dy).sqrt() / r;
                if d >= 1.0 {
                    continue;
                }
                let fx = ((px as f32 - (cx - r)) / (2.0 * r)).clamp(0.0, 1.0);
                let fy = ((py as f32 - (cy - r)) / (2.0 * r)).clamp(0.0, 1.0);
                let feather = smoothstep(1.0, 0.8, d); // 1 in the middle, fades to 0 at the rim
                let o = rgb.get_pixel(px, py).0;
                let np = std::array::from_fn(|c| {
                    let fill = (bl[c] as f32 * (1.0 - fx) + br[c] as f32 * fx + bt[c] as f32 * (1.0 - fy) + bb[c] as f32 * fy) / 2.0;
                    (o[c] as f32 * (1.0 - feather) + fill * feather) as u8
                });
                rgb.put_pixel(px, py, Rgb(np));
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Clone stamp: copy a disc from source `(sx, sy)` to destination `(dx, dy)` (all per-mille) with a
    /// feathered edge.
    pub fn clone_stamp(img: &DynamicImage, sx: i32, sy: i32, dx: i32, dy: i32, radius: i32) -> DynamicImage {
        let src = img.to_rgb8();
        let mut rgb = src.clone();
        let (w, h) = (rgb.width(), rgb.height());
        let (dcx, dcy, r) = disc_geom(w, h, dx, dy, radius);
        let (scx, scy, _) = disc_geom(w, h, sx, sy, radius);
        let (x0, y0) = ((dcx - r).floor().max(0.0) as u32, (dcy - r).floor().max(0.0) as u32);
        let (x1, y1) = (((dcx + r).ceil() as u32).min(w - 1), ((dcy + r).ceil() as u32).min(h - 1));
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (ox, oy) = (px as f32 - dcx, py as f32 - dcy);
                let d = (ox * ox + oy * oy).sqrt() / r;
                if d >= 1.0 {
                    continue;
                }
                let s = super::bilinear(&src, scx + ox, scy + oy).0;
                let feather = smoothstep(1.0, 0.75, d);
                let o = rgb.get_pixel(px, py).0;
                rgb.put_pixel(px, py, Rgb(std::array::from_fn(|c| (o[c] as f32 * (1.0 - feather) + s[c] as f32 * feather) as u8)));
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Red-eye removal: within a disc, desaturate + darken pixels where red strongly dominates.
    pub fn red_eye(img: &DynamicImage, x: i32, y: i32, radius: i32) -> DynamicImage {
        let mut rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let (cx, cy, r) = disc_geom(w, h, x, y, radius);
        let (x0, y0) = ((cx - r).floor().max(0.0) as u32, (cy - r).floor().max(0.0) as u32);
        let (x1, y1) = (((cx + r).ceil() as u32).min(w - 1), ((cy + r).ceil() as u32).min(h - 1));
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (dx, dy) = (px as f32 - cx, py as f32 - cy);
                if (dx * dx + dy * dy).sqrt() >= r {
                    continue;
                }
                let p = rgb.get_pixel(px, py).0;
                let (rr, gg, bb) = (p[0] as f32, p[1] as f32, p[2] as f32);
                // Redness: red clearly above the green/blue average → the pupil glare.
                if rr > (gg + bb) * 0.7 + 20.0 {
                    let g = ((gg + bb) / 2.0 * 0.6) as u8;
                    rgb.put_pixel(px, py, Rgb([g, g, g]));
                }
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }

    /// Dodge / burn brush: lighten (`amount` > 0) or darken a disc with a soft radial falloff.
    pub fn dodge_burn(img: &DynamicImage, x: i32, y: i32, radius: i32, amount: i32) -> DynamicImage {
        let mut rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let (cx, cy, r) = disc_geom(w, h, x, y, radius);
        let amt = amount.clamp(-100, 100) as f32 / 100.0;
        let (x0, y0) = ((cx - r).floor().max(0.0) as u32, (cy - r).floor().max(0.0) as u32);
        let (x1, y1) = (((cx + r).ceil() as u32).min(w - 1), ((cy + r).ceil() as u32).min(h - 1));
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (dx, dy) = (px as f32 - cx, py as f32 - cy);
                let d = (dx * dx + dy * dy).sqrt() / r;
                if d >= 1.0 {
                    continue;
                }
                let wgt = smoothstep(1.0, 0.0, d) * amt * 0.6; // soft, full at centre
                let o = rgb.get_pixel(px, py).0;
                rgb.put_pixel(px, py, Rgb(std::array::from_fn(|c| enc(o[c] as f32 / 255.0 + wgt))));
            }
        }
        DynamicImage::ImageRgb8(rgb)
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

    /// Approximate blackbody RGB (0..1) for a colour temperature in Kelvin (Tanner Helland).
    fn kelvin_rgb(kelvin: f32) -> [f32; 3] {
        let t = kelvin.clamp(1000.0, 40000.0) / 100.0;
        let r = if t <= 66.0 { 1.0 } else { (1.292936 * (t - 60.0).powf(-0.1332047)).clamp(0.0, 1.0) };
        let g = if t <= 66.0 {
            (0.3900816 * t.ln() - 0.6318414).clamp(0.0, 1.0)
        } else {
            (1.129891 * (t - 60.0).powf(-0.0755148)).clamp(0.0, 1.0)
        };
        let b = if t >= 66.0 {
            1.0
        } else if t <= 19.0 {
            0.0
        } else {
            (0.5432068 * (t - 10.0).ln() - 1.196254).clamp(0.0, 1.0)
        };
        [r, g, b]
    }

    /// Kelvin white balance: `amount` −100..100 (positive = warmer / lower Kelvin), applied as
    /// per-channel gains relative to a 6500 K neutral.
    pub fn kelvin(img: &DynamicImage, amount: i32) -> DynamicImage {
        let temp = (6500.0 - amount as f32 * 42.0).clamp(1500.0, 12000.0);
        let target = kelvin_rgb(temp);
        let neutral = kelvin_rgb(6500.0);
        let gain: [f32; 3] = std::array::from_fn(|i| target[i] / neutral[i].max(1e-3));
        map_rgb(img, |r, g, b| [r * gain[0], g * gain[1], b * gain[2]])
    }

    /// Gradient map: recolour by luma through a preset ramp.
    pub fn gradient_map(img: &DynamicImage, style: i32) -> DynamicImage {
        let stops: &[[f32; 3]] = match style {
            2 => &[[0.02, 0.05, 0.16], [0.12, 0.34, 0.58], [0.72, 0.9, 1.0]], // cyanotype
            3 => &[[0.0, 0.0, 0.0], [0.55, 0.08, 0.0], [1.0, 0.55, 0.0], [1.0, 1.0, 0.82]], // fire
            4 => &[[0.05, 0.15, 0.2], [0.18, 0.42, 0.48], [0.9, 0.6, 0.32], [1.0, 0.86, 0.62]], // teal-orange
            _ => &[[0.14, 0.09, 0.05], [0.6, 0.45, 0.3], [1.0, 0.96, 0.86]], // warm
        };
        map_rgb(img, |r, g, b| ramp(stops, luma(r, g, b)))
    }

    /// Cross-hatch: black ink lines whose density tracks tone (diagonal, anti-diagonal, then axes).
    pub fn crosshatch(img: &DynamicImage) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = rgb.get_pixel(x, y).0;
                let yl = luma(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0;
                let (xi, yi) = (x as i32, y as i32);
                let ink = (yl < 0.85 && (xi + yi).rem_euclid(6) == 0)
                    || (yl < 0.62 && (xi - yi).rem_euclid(6) == 0)
                    || (yl < 0.4 && yi.rem_euclid(6) == 0)
                    || (yl < 0.2 && xi.rem_euclid(6) == 0);
                let v = if ink { 0.1 } else { 1.0 };
                out.put_pixel(x, y, Rgb([enc(v), enc(v), enc(v)]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// "Better sky": a NO-AI, no-manual-mask sky enhancer. Builds a soft per-pixel sky mask from a
    /// vertical prior (sky sits up top) combined with blue-dominance OR bright/near-neutral (overcast)
    /// colour, then applies a polarizer-like effect — saturate & slightly deepen the blue, nudging it
    /// richer — weighted by `mask × amount/100`. At `amount = 0` the mask is zero → byte-identical.
    pub fn enhance_sky(img: &DynamicImage, amount: i32) -> DynamicImage {
        let strength = amount.clamp(0, 100) as f32 / 100.0;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            let fy = if h > 1 { y as f32 / (h - 1) as f32 } else { 0.0 };
            // Vertical prior: full weight in the top ~35 %, fading to 0 by ~90 % down.
            let wy = 1.0 - smoothstep(0.35, 0.9, fy);
            for x in 0..w {
                let p = rgb.get_pixel(x, y).0;
                let (r, g, b) = (p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
                let yl = luma(r, g, b);
                // Blue sky: blue channel clearly above the red/green mean.
                let blue = smoothstep(0.01, 0.16, b - (r + g) * 0.5);
                // Overcast / hazy sky: bright and near-neutral.
                let mx = r.max(g).max(b);
                let mn = r.min(g).min(b);
                let sat = if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
                let bright = smoothstep(0.72, 0.97, yl) * (1.0 - smoothstep(0.15, 0.4, sat));
                let mask = (wy * blue.max(bright)).clamp(0.0, 1.0) * strength;
                // Polarizer: push chroma out from luma, deepen a touch, keep the blue rich.
                let k = 1.0 + 0.45 * mask;
                let deepen = 1.0 - 0.12 * mask;
                let nr = (yl + (r - yl) * k) * deepen;
                let ng = (yl + (g - yl) * k) * deepen;
                let nb = ((yl + (b - yl) * k) * deepen + 0.05 * mask).min(1.0);
                out.put_pixel(x, y, Rgb([enc(nr), enc(ng), enc(nb)]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Gray-world auto white balance: scale each channel so its mean moves toward the overall grey,
    /// neutralising a colour cast. `amount` 0..100 blends channel gains toward the full correction
    /// (0 = identity).
    pub fn auto_white_balance(img: &DynamicImage, amount: i32) -> DynamicImage {
        let rgb = img.to_rgb8();
        let n = rgb.width() as f64 * rgb.height() as f64;
        if n == 0.0 {
            return DynamicImage::ImageRgb8(rgb);
        }
        let (mut sr, mut sg, mut sb) = (0f64, 0f64, 0f64);
        for p in rgb.pixels() {
            sr += p.0[0] as f64;
            sg += p.0[1] as f64;
            sb += p.0[2] as f64;
        }
        let (mr, mg, mb) = (sr / n, sg / n, sb / n);
        let gray = (mr + mg + mb) / 3.0;
        let t = amount.clamp(0, 100) as f32 / 100.0;
        let gain = |m: f64| -> f32 {
            if m < 1.0 { 1.0 } else { (1.0 - t) + t * (gray / m) as f32 }
        };
        let (kr, kg, kb) = (gain(mr), gain(mg), gain(mb));
        map_rgb(img, |r, g, b| [r * kr, g * kg, b * kb])
    }

    /// Crystallize (Voronoi): partition the image into jittered polygonal cells (a grid of seed
    /// points nudged by a deterministic hash) and flat-fill each cell with its average colour — a
    /// stained-glass / low-poly look. `strength` 0..100 sets the cell size (0 = identity). Fully
    /// deterministic (grid + hash jitter), so it replays byte-identically.
    pub fn crystallize(img: &DynamicImage, strength: i32) -> DynamicImage {
        let s = strength.clamp(0, 100);
        if s <= 0 {
            return img.clone();
        }
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let maxcell = (w.min(h) as f32 / 12.0).max(3.0);
        let cell = (2.0 + (s as f32 / 100.0) * maxcell).max(2.0);
        let cols = ((w as f32 / cell).ceil() as u32).max(1);
        let rows = ((h as f32 / cell).ceil() as u32).max(1);
        // Deterministic integer hash → per-seed jitter (replay-stable, no RNG).
        let hash = |a: u32, b: u32| -> u32 {
            let mut x = a.wrapping_mul(374761393).wrapping_add(b.wrapping_mul(668265263));
            x = (x ^ (x >> 13)).wrapping_mul(1274126177);
            x ^ (x >> 16)
        };
        let seed_pos = |gx: u32, gy: u32| -> (f32, f32) {
            let jx = (hash(gx, gy) & 0xffff) as f32 / 65535.0 - 0.5;
            let jy = (hash(gx.wrapping_add(7), gy.wrapping_add(31)) & 0xffff) as f32 / 65535.0 - 0.5;
            ((gx as f32 + 0.5 + jx) * cell, (gy as f32 + 0.5 + jy) * cell)
        };
        let ncells = (cols * rows) as usize;
        let mut sum = vec![[0u64; 3]; ncells];
        let mut cnt = vec![0u32; ncells];
        let mut cell_of = vec![0u32; (w * h) as usize];
        for y in 0..h {
            let gy = ((y as f32 / cell) as u32).min(rows - 1);
            for x in 0..w {
                let gx = ((x as f32 / cell) as u32).min(cols - 1);
                // Nearest seed among the 3×3 neighbouring grid cells (jitter stays within one cell).
                let mut best = gy * cols + gx;
                let mut bestd = f32::MAX;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let sx = gx as i32 + dx;
                        let sy = gy as i32 + dy;
                        if sx < 0 || sy < 0 || sx >= cols as i32 || sy >= rows as i32 {
                            continue;
                        }
                        let (px, py) = seed_pos(sx as u32, sy as u32);
                        let d = (px - x as f32).powi(2) + (py - y as f32).powi(2);
                        if d < bestd {
                            bestd = d;
                            best = sy as u32 * cols + sx as u32;
                        }
                    }
                }
                let idx = (y * w + x) as usize;
                cell_of[idx] = best;
                let p = rgb.get_pixel(x, y).0;
                let c = best as usize;
                sum[c][0] += p[0] as u64;
                sum[c][1] += p[1] as u64;
                sum[c][2] += p[2] as u64;
                cnt[c] += 1;
            }
        }
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = cell_of[(y * w + x) as usize] as usize;
                let n = cnt[c].max(1) as u64;
                out.put_pixel(x, y, Rgb([(sum[c][0] / n) as u8, (sum[c][1] / n) as u8, (sum[c][2] / n) as u8]));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Bilateral denoise: an edge-preserving 5×5 filter — each pixel is a weighted average of its
    /// neighbours, weighted by both spatial distance and luma similarity, so flat noise is averaged
    /// out while edges (large luma jumps) are preserved. `strength` 0..100 widens the range kernel
    /// (smooths across bigger differences) and blends toward the result; 0 = identity.
    pub fn bilateral(img: &DynamicImage, strength: i32) -> DynamicImage {
        let s = strength.clamp(0, 100);
        if s <= 0 {
            return img.clone();
        }
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let t = s as f32 / 100.0;
        // Precompute the 5×5 spatial (Gaussian) weights.
        const R: i32 = 2;
        let sigma_s = 2.0f32;
        let mut sw = [[0f32; 5]; 5];
        for dy in -R..=R {
            for dx in -R..=R {
                sw[(dy + R) as usize][(dx + R) as usize] =
                    (-((dx * dx + dy * dy) as f32) / (2.0 * sigma_s * sigma_s)).exp();
            }
        }
        // Range sigma grows with strength: 12 (subtle) → 60 (aggressive) luma units.
        let sigma_r = 12.0 + t * 48.0;
        let inv_2sr2 = 1.0 / (2.0 * sigma_r * sigma_r);
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = rgb.get_pixel(x, y).0;
                let cl = luma(c[0] as f32, c[1] as f32, c[2] as f32);
                let (mut acc, mut wsum) = ([0f32; 3], 0f32);
                for dy in -R..=R {
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    for dx in -R..=R {
                        let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                        let p = rgb.get_pixel(nx, ny).0;
                        let pl = luma(p[0] as f32, p[1] as f32, p[2] as f32);
                        let dr = pl - cl;
                        let wgt = sw[(dy + R) as usize][(dx + R) as usize] * (-dr * dr * inv_2sr2).exp();
                        acc[0] += p[0] as f32 * wgt;
                        acc[1] += p[1] as f32 * wgt;
                        acc[2] += p[2] as f32 * wgt;
                        wsum += wgt;
                    }
                }
                let inv = 1.0 / wsum.max(1e-6);
                // Blend the filtered value toward the original by `t` (so low strength stays subtle).
                let px = std::array::from_fn(|i| {
                    let f = acc[i] * inv;
                    (c[i] as f32 * (1.0 - t) + f * t).round().clamp(0.0, 255.0) as u8
                });
                out.put_pixel(x, y, Rgb(px));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    /// Face polish: edge-preserving skin smoothing limited to the detected face `ellipses`
    /// (`(cx, cy, rx, ry)` in per-mille of the image), blended by `strength` 0..100. A selective
    /// Gaussian — smooth low-detail skin (cheeks/forehead) while keeping high-detail edges (eyes,
    /// lips, hair, glasses) crisp — so it reads as a retouch, not a blur. At `strength = 0` or with no
    /// faces it returns the input unchanged.
    pub fn face_polish(img: &DynamicImage, strength: i32, ellipses: &[[i32; 4]]) -> DynamicImage {
        let s = strength.clamp(0, 100) as f32 / 100.0;
        if s <= 0.0 || ellipses.is_empty() {
            return img.clone();
        }
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        // Smoothing radius scales with the image so it reads the same at any resolution.
        let sigma = (w.min(h) as f32 / 200.0).clamp(1.5, 6.0);
        let blurred = image::imageops::blur(&rgb, sigma);
        let (wf, hf) = (w as f32, h as f32);
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            let py = y as f32 / hf;
            for x in 0..w {
                let px = x as f32 / wf;
                // Soft face mask = max over the ellipses of a smooth radial falloff (full inside,
                // fading to 0 just past the ellipse edge).
                let mut mask = 0f32;
                for e in ellipses {
                    let (cx, cy) = (e[0] as f32 / 1000.0, e[1] as f32 / 1000.0);
                    let (rx, ry) = (e[2] as f32 / 1000.0, e[3] as f32 / 1000.0);
                    if rx <= 0.0 || ry <= 0.0 {
                        continue;
                    }
                    let dx = (px - cx) / rx;
                    let dy = (py - cy) / ry;
                    let d = (dx * dx + dy * dy).sqrt(); // 1.0 on the ellipse boundary
                    let fm = 1.0 - smoothstep(0.7, 1.05, d);
                    if fm > mask {
                        mask = fm;
                    }
                }
                let o = rgb.get_pixel(x, y).0;
                if mask <= 0.0 {
                    out.put_pixel(x, y, Rgb(o));
                    continue;
                }
                let b = blurred.get_pixel(x, y).0;
                // Edge preservation: keep the original where local detail (orig vs blurred luma) is
                // high — that's eyes, lips, hairlines — and smooth only the flat skin.
                let od = luma(o[0] as f32, o[1] as f32, o[2] as f32);
                let bd = luma(b[0] as f32, b[1] as f32, b[2] as f32);
                let detail = (od - bd).abs() / 255.0;
                let keep_edge = smoothstep(0.06, 0.16, detail);
                let wgt = mask * s * (1.0 - keep_edge);
                let mix = |a: u8, c: u8| enc((a as f32 / 255.0) * (1.0 - wgt) + (c as f32 / 255.0) * wgt);
                out.put_pixel(x, y, Rgb([mix(o[0], b[0]), mix(o[1], b[1]), mix(o[2], b[2])]));
            }
        }
        DynamicImage::ImageRgb8(out)
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

/// Burn `text` into the lower-right of `img`. `font` is an optional TrueType/OpenType path; a failed
/// load (or `None`) falls back to the built-in bitmap font. Non-destructive at the pixel level — the
/// caller keeps the original; this returns a new image. Empty text is a no-op.
fn watermark(img: &DynamicImage, text: &str, font: Option<&str>) -> DynamicImage {
    if text.trim().is_empty() {
        return img.clone();
    }
    if let Some(fp) = font {
        // Best-effort: a bad path just leaves the previously-loaded / default font in place.
        let _ = crate::map::labels::shaped::load_font(std::path::Path::new(fp));
    }
    let mut rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let scale = (h / 45).max(1);
    let margin = (h / 40).max(6) as i32;
    let approx_w = text.chars().count() as u32 * 6 * scale; // bitmap-ish width estimate
    let x = (w as i32 - approx_w as i32 - margin).max(margin);
    let y = (h as i32 - (8 * scale) as i32 - margin).max(0);
    crate::map::labels::draw_text_haloed(&mut rgb, x, y, text, scale, [255, 255, 255], [0, 0, 0]);
    DynamicImage::ImageRgb8(rgb)
}

/// Map a local-adjust `adjust` index to the base tonal/colour op it drives (reusing all the existing
/// adjustment logic). Kept in sync with `local_adjust_name`.
fn local_base_op(adjust: i32, amount: i32) -> EditOp {
    match adjust {
        1 => EditOp::Brightness(amount),
        2 => EditOp::Contrast(amount),
        3 => EditOp::Saturation(amount),
        4 => EditOp::Warmth(amount),
        5 => EditOp::Vibrance(amount),
        6 => EditOp::Definition(amount),
        7 => EditOp::Blur(amount),
        _ => EditOp::Exposure(amount),
    }
}

/// Human name for a local-adjust index.
fn local_adjust_name(adjust: i32) -> &'static str {
    match adjust {
        1 => "brightness",
        2 => "contrast",
        3 => "saturation",
        4 => "warmth",
        5 => "vibrance",
        6 => "definition",
        7 => "blur",
        _ => "exposure",
    }
}

/// Local (masked) adjustment: apply the base adjustment globally, then blend it back through a linear
/// or radial mask — a Lightroom-style graduated / radial local adjustment for *any* tonal/colour op.
fn local_adjust(img: &DynamicImage, adjust: i32, amount: i32, shape: i32, dir: i32) -> DynamicImage {
    if amount == 0 {
        return img.clone();
    }
    let adjusted = local_base_op(adjust, amount).apply(img.clone());
    if adjusted.width() != img.width() || adjusted.height() != img.height() {
        return adjusted; // shouldn't happen (these ops keep dims) — fail open
    }
    let mask = adjust::local_mask(img, shape, dir);
    adjust::blend_masked(img, &adjusted, &mask)
}

/// Brushed (painted-mask) adjustment: apply the base adjustment globally, then blend it back through
/// the union of the painted dabs. Identity when `amount == 0` or no dabs were painted.
fn brush_adjust(img: &DynamicImage, adjust: i32, amount: i32, dabs: &[[i32; 3]]) -> DynamicImage {
    if amount == 0 || dabs.is_empty() {
        return img.clone();
    }
    let adjusted = local_base_op(adjust, amount).apply(img.clone());
    if adjusted.width() != img.width() || adjusted.height() != img.height() {
        return adjusted; // these ops keep dims — fail open if not
    }
    let mask = adjust::brush_mask(img, dabs);
    adjust::blend_masked(img, &adjusted, &mask)
}

/// Lens distortion correction: a radial warp about the centre — `amount` > 0 corrects **barrel**
/// (bulging) distortion, < 0 corrects **pincushion**. `r_src = r · (1 + k·r²)` with `r` normalised to
/// the corner. Bilinear, edge-clamped. `amount` −100..100.
fn lens_distort(img: &DynamicImage, amount: i32) -> DynamicImage {
    let k = amount as f32 / 100.0 * 0.35;
    if k.abs() < 1e-4 {
        return img.clone();
    }
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let (cx, cy) = ((w - 1) as f32 / 2.0, (h - 1) as f32 / 2.0);
    let norm = (cx * cx + cy * cy).sqrt().max(1.0);
    let mut out = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (vx, vy) = (x as f32 - cx, y as f32 - cy);
            let r = (vx * vx + vy * vy).sqrt() / norm;
            let f = 1.0 + k * r * r;
            out.put_pixel(x, y, bilinear(&rgb, cx + vx * f, cy + vy * f));
        }
    }
    DynamicImage::ImageRgb8(out)
}

/// Chromatic-aberration removal: rescale the red and blue channels radially about the centre (green
/// fixed) to pull colour fringes back into register. `strength` 0..100 sets the correction; dial it
/// until the coloured edges disappear.
fn chromatic_aberration(img: &DynamicImage, strength: i32) -> DynamicImage {
    let s = strength.clamp(0, 100);
    if s == 0 {
        return img.clone();
    }
    let k = s as f32 / 100.0 * 0.008;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let (cx, cy) = ((w - 1) as f32 / 2.0, (h - 1) as f32 / 2.0);
    let mut out = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (vx, vy) = (x as f32 - cx, y as f32 - cy);
            let r = bilinear(&rgb, cx + vx * (1.0 - k), cy + vy * (1.0 - k)).0[0];
            let g = rgb.get_pixel(x, y).0[1];
            let b = bilinear(&rgb, cx + vx * (1.0 + k), cy + vy * (1.0 + k)).0[2];
            out.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    DynamicImage::ImageRgb8(out)
}

/// Weight-free **de-slop** (RFC QUALITY-8 P5): run the naturalize *Photo* recipe scaled by
/// `strength` (0–100 → 0.0–1.0) over the whole image. Delegates to the shared [`crate::naturalize`]
/// core (gray-world WB + auto-levels + vibrance + unsharp + variance-gated micro-texture + a gentle
/// desaturating grade) so the photo manager and the CLI/`plakat ui` tab stay in lock-step.
/// Deterministic → replay-safe through the edit pipeline.
fn naturalize_op(img: &DynamicImage, strength: i32) -> DynamicImage {
    let s = (strength.clamp(0, 100) as f32) / 100.0;
    if s == 0.0 {
        return img.clone();
    }
    let mut p = crate::naturalize::Preset::Photo.params();
    p.grain *= s;
    p.aberration *= s;
    p.vignette *= s;
    p.bloom *= s;
    p.desaturate *= s;
    p.warm *= s;
    p.defocus *= s;
    p.polish *= s;
    p.micro *= s;
    DynamicImage::ImageRgb8(crate::naturalize::apply(&img.to_rgb8(), &p))
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

/// Border / letterbox: pad the image onto a larger canvas. `aw:ah` sets the target aspect (0,0 = an
/// even frame ~6 %); `mode` 0 black, 1 white, 2 a blurred, scaled copy of the image.
fn border(img: &DynamicImage, aw: i32, ah: i32, mode: i32) -> DynamicImage {
    use image::{imageops, Rgb, RgbImage};
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let (cw, ch) = if aw <= 0 || ah <= 0 {
        let b = (w.min(h) as f32 * 0.06).round() as u32;
        (w + 2 * b, h + 2 * b)
    } else {
        let a = aw as f32 / ah as f32;
        if (w as f32 / h as f32) > a {
            (w, (w as f32 / a).round() as u32)
        } else {
            ((h as f32 * a).round() as u32, h)
        }
    };
    let (cw, ch) = (cw.max(w), ch.max(h));
    let mut canvas: RgbImage = match mode {
        2 => imageops::blur(&imageops::resize(&rgb, cw, ch, imageops::FilterType::Triangle), 20.0),
        1 => RgbImage::from_pixel(cw, ch, Rgb([255, 255, 255])),
        _ => RgbImage::from_pixel(cw, ch, Rgb([0, 0, 0])),
    };
    imageops::overlay(&mut canvas, &rgb, ((cw - w) / 2) as i64, ((ch - h) / 2) as i64);
    DynamicImage::ImageRgb8(canvas)
}

/// Circle crop: keep a centred circle (radius = half the shorter side), fill outside with `mode`
/// 0 black / 1 white.
fn crop_circle(img: &DynamicImage, mode: i32) -> DynamicImage {
    let mut rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as f32, rgb.height() as f32);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let r = (w.min(h) / 2.0).max(1.0);
    let fill = if mode == 1 { [255u8, 255, 255] } else { [0, 0, 0] };
    for (x, y, p) in rgb.enumerate_pixels_mut() {
        if ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() > r {
            p.0 = fill;
        }
    }
    DynamicImage::ImageRgb8(rgb)
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
        img = op.clone().apply(img);
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
            EditOp::Border { aspect_w: 16, aspect_h: 9, mode: 0 }, EditOp::CropCircle(1),
            EditOp::Invert, EditOp::Sepia, EditOp::Duotone,
            EditOp::Posterize(50), EditOp::Solarize(40), EditOp::Threshold(128),
            EditOp::OilPaint { style: 3, strength: 80 }, EditOp::PencilSketch(60), EditOp::Cartoon(70),
            EditOp::Watercolor { style: 5, strength: 90 }, EditOp::Ink { style: 2, strength: 100 },
            EditOp::Emboss(50), EditOp::Pixelate(40), EditOp::Blur(60), EditOp::Bloom(70),
            EditOp::Charcoal(80), EditOp::Halftone(90), EditOp::FalseColor { style: 1, strength: 100 },
            EditOp::Kelvin(-30), EditOp::GradientMap { style: 3, strength: 80 }, EditOp::Crosshatch(90),
            EditOp::EnhanceSky(70), EditOp::AutoWhiteBalance(85), EditOp::Crystallize(60), EditOp::Bilateral(70),
            EditOp::SelectiveLum { hue: 240, lum: -40 }, EditOp::GrayPointWB { x: 500, y: 300 },
            EditOp::TiltShift(70), EditOp::MotionBlur { angle: 45, strength: 60 }, EditOp::ZoomBlur(50),
            EditOp::SpinBlur(40), EditOp::ChannelMixerBW { r: 75, g: 20, b: 5 }, EditOp::FilmNegative,
            EditOp::LensDistort(-40), EditOp::ChromaticAberration(60),
            EditOp::LocalAdjust { adjust: 4, amount: 30, shape: 0, dir: 2 },
            EditOp::SpotHeal { x: 400, y: 300, radius: 60 },
            EditOp::Clone { sx: 300, sy: 400, dx: 600, dy: 400, radius: 55 },
            EditOp::RedEye { x: 450, y: 350, radius: 40 },
            EditOp::DodgeBurn { x: 500, y: 500, radius: 120, amount: -35 },
            EditOp::Perspective4 { pts: [[80, 90], [910, 70], [940, 930], [60, 950]] },
            EditOp::Watermark { text: "© plakat".into(), font: None },
            EditOp::Watermark { text: "shot on film".into(), font: Some("/fonts/Foo.ttf".into()) },
            EditOp::Lut { path: "/grades/teal.cube".into() },
            EditOp::FacePolish {
                strength: 65,
                faces: [[500, 400, 120, 150], [300, 350, 90, 110], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                n: 2,
            },
        ] {
            assert_eq!(EditOp::from_entry(&op.clone().to_entry()), Some(op));
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
    fn border_letterboxes_and_circle_crop_masks() {
        // 16:9 letterbox of a 100×100 image → wider canvas, same height, black bars on the sides.
        let out = EditOp::Border { aspect_w: 16, aspect_h: 9, mode: 0 }.apply(img(100, 100)).to_rgb8();
        assert_eq!(out.height(), 100);
        assert!(out.width() > 100, "canvas widened to 16:9");
        assert_eq!(out.get_pixel(1, 50).0, [0, 0, 0], "left bar is black");
        // Circle crop (white): a corner is white, the centre is the original pixel.
        let cc = EditOp::CropCircle(1).apply(img(40, 40)).to_rgb8();
        assert_eq!(cc.get_pixel(0, 0).0, [255, 255, 255], "corner outside the circle");
        assert_ne!(cc.get_pixel(20, 20).0, [255, 255, 255], "centre kept");
        assert_eq!(EditOp::from_tag("circle"), Some(EditOp::CropCircle(0)));
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
    fn naturalize_op_is_replay_stable_and_a_noop_at_zero() {
        // A textured image so the weight-free pass has something to work on.
        let src = DynamicImage::ImageRgb8(ImageBuffer::from_fn(24, 24, |x, y| {
            Rgb([((x * 11) % 256) as u8, ((y * 7) % 256) as u8, (((x + y) * 5) % 256) as u8])
        }));
        // Deterministic → replaying is byte-identical (safe through the edit pipeline).
        let a = EditOp::Naturalize(60).apply(src.clone()).to_rgb8();
        let b = EditOp::Naturalize(60).apply(src.clone()).to_rgb8();
        assert_eq!(a, b, "naturalize must be replay-stable");
        assert_ne!(a, src.to_rgb8(), "naturalize at 60 changes pixels");
        // Strength 0 is an exact no-op (the identity guard).
        assert_eq!(EditOp::Naturalize(0).apply(src.clone()).to_rgb8(), src.to_rgb8(), "0 = identity");
        // Round-trips through the serialised EditEntry form.
        let e = EditOp::Naturalize(60).to_entry();
        assert_eq!(EditOp::from_entry(&e), Some(EditOp::Naturalize(60)), "entry round-trip");
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
            EditOp::GradientMap { style: 1, strength: 100 }, EditOp::GradientMap { style: 3, strength: 100 },
            EditOp::Crosshatch(100), EditOp::Crystallize(80), EditOp::Bilateral(90),
            EditOp::TiltShift(80), EditOp::MotionBlur { angle: 0, strength: 80 }, EditOp::ZoomBlur(70),
            EditOp::SpinBlur(60), EditOp::ChannelMixerBW { r: 75, g: 20, b: 5 }, EditOp::FilmNegative,
            EditOp::LensDistort(60), EditOp::ChromaticAberration(80),
            EditOp::LocalAdjust { adjust: 0, amount: 50, shape: 0, dir: 0 },
            EditOp::LocalAdjust { adjust: 3, amount: -40, shape: 1, dir: 1 },
        ] {
            let label = op.label();
            let out = op.apply(src.clone()).to_rgb8();
            assert_eq!((out.width(), out.height()), (24, 24), "{label} keeps dims");
            assert_ne!(out, base, "{label} should change pixels");
        }
        // Pixelate at strength 0 and any filter at strength 0 are identity (blend = original).
        assert_eq!(EditOp::Pixelate(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::Cartoon(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::Crystallize(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::Bilateral(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::TiltShift(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::ZoomBlur(0).apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::MotionBlur { angle: 0, strength: 0 }.apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::OilPaint { style: 5, strength: 0 }.apply(src.clone()).to_rgb8(), base);
        assert_eq!(EditOp::from_tag("sketch"), Some(EditOp::PencilSketch(100)));
        // Kelvin at 0 is (near) identity; warming shifts red above blue on grey.
        let grey = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([128, 128, 128])));
        let warm = EditOp::Kelvin(60).apply(grey.clone()).to_rgb8().get_pixel(0, 0).0;
        assert!(warm[0] >= warm[2], "warmer: red ≥ blue");
    }

    #[test]
    fn enhance_sky_masks_and_deepens_blue_top() {
        // Blue sky up top, green grass at the bottom — the enhancer should touch the sky, not the grass.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(20, 20, |_x, y| {
            if y < 8 { Rgb([120u8, 160, 235]) } else { Rgb([60u8, 130, 50]) }
        }));
        let out = EditOp::EnhanceSky(100).apply(img.clone()).to_rgb8();
        let sky = out.get_pixel(10, 1).0;
        let grass = out.get_pixel(10, 18).0;
        assert_ne!(sky, [120, 160, 235], "sky pixel changed");
        assert!(sky[2] >= 235, "blue kept rich / deepened");
        assert_eq!(grass, [60, 130, 50], "grass (low, non-blue) untouched");
        // Strength 0 → byte-identical.
        assert_eq!(EditOp::EnhanceSky(0).apply(img.clone()).to_rgb8(), img.to_rgb8());
    }

    #[test]
    fn local_adjust_applies_through_the_mask_only() {
        // A flat grey; local exposure +60 through a top gradient should lift the TOP much more than
        // the bottom (which the mask barely reaches). Amount 0 is identity.
        let grey = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(20, 40, Rgb([120u8, 120, 120])));
        let out = EditOp::LocalAdjust { adjust: 0, amount: 60, shape: 0, dir: 0 }.apply(grey.clone()).to_rgb8();
        let top = out.get_pixel(10, 1).0[0];
        let bottom = out.get_pixel(10, 38).0[0];
        assert!(top > 120, "top lifted by the local exposure: {top}");
        assert!(top > bottom + 20, "gradient localises the effect: top {top} >> bottom {bottom}");
        assert_eq!(EditOp::LocalAdjust { adjust: 0, amount: 0, shape: 0, dir: 0 }.apply(grey.clone()).to_rgb8(), grey.to_rgb8());
    }

    #[test]
    fn brush_adjust_only_affects_painted_dabs() {
        // Flat grey; a single dab centred near the top-left lifts exposure there but leaves a far
        // corner untouched. No dabs or amount 0 → identity. Serde round-trips the dab list.
        let grey = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(40, 40, Rgb([120u8, 120, 120])));
        let dabs = vec![[250, 250, 200]]; // x=25%, y=25%, r=20% of min dim
        let op = EditOp::BrushAdjust { adjust: 0, amount: 80, dabs: dabs.clone() };
        let out = op.clone().apply(grey.clone()).to_rgb8();
        let inside = out.get_pixel(10, 10).0[0]; // under the dab
        let outside = out.get_pixel(38, 38).0[0]; // opposite corner, unpainted
        assert!(inside > 130, "dab lifted exposure: {inside}");
        assert_eq!(outside, 120, "far corner untouched: {outside}");
        // Empty dabs and amount 0 are identity.
        assert_eq!(
            EditOp::BrushAdjust { adjust: 0, amount: 80, dabs: vec![] }.apply(grey.clone()).to_rgb8(),
            grey.to_rgb8()
        );
        assert_eq!(
            EditOp::BrushAdjust { adjust: 0, amount: 0, dabs: dabs.clone() }.apply(grey.clone()).to_rgb8(),
            grey.to_rgb8()
        );
        // Serde round-trip preserves the dabs.
        assert_eq!(EditOp::from_entry(&op.clone().to_entry()), Some(op));
    }

    #[test]
    fn retouch_ops_heal_dodge_and_redeye() {
        // Spot heal fills a red blemish on a grey field with the surrounding grey.
        let mut buf = ImageBuffer::from_pixel(60, 60, Rgb([130u8, 130, 130]));
        for y in 27..33 {
            for x in 27..33 {
                buf.put_pixel(x, y, Rgb([230, 20, 20]));
            }
        }
        let img = DynamicImage::ImageRgb8(buf);
        let healed = EditOp::SpotHeal { x: 500, y: 500, radius: 120 }.apply(img.clone()).to_rgb8();
        let c = healed.get_pixel(30, 30).0;
        assert!(c[0] < 180 && c[1] > 80, "blemish healed toward grey (was 230,20,20): {c:?}");

        // Dodge lightens the centre; burn darkens it.
        let grey = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(40, 40, Rgb([120u8, 120, 120])));
        let dodged = EditOp::DodgeBurn { x: 500, y: 500, radius: 300, amount: 60 }.apply(grey.clone()).to_rgb8();
        let burned = EditOp::DodgeBurn { x: 500, y: 500, radius: 300, amount: -60 }.apply(grey.clone()).to_rgb8();
        assert!(dodged.get_pixel(20, 20).0[0] > 120, "dodge lightened");
        assert!(burned.get_pixel(20, 20).0[0] < 120, "burn darkened");

        // Red-eye neutralises a red disc.
        let eye = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(30, 30, Rgb([220u8, 30, 30])));
        let fixed = EditOp::RedEye { x: 500, y: 500, radius: 600 }.apply(eye).to_rgb8().get_pixel(15, 15).0;
        assert!(fixed[0] < 120 && fixed[0] == fixed[1], "red-eye neutralised to grey: {fixed:?}");
    }

    #[test]
    fn watermark_edit_is_replayable_and_draws_text() {
        // Watermark is now a normal replayable op: it burns text (changes pixels) and its serialised
        // form round-trips. Empty text is identity.
        let white = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(120, 60, Rgb([255, 255, 255])));
        let out = EditOp::Watermark { text: "PLAKAT".into(), font: None }.apply(white.clone()).to_rgb8();
        assert_ne!(out, white.to_rgb8(), "watermark drew pixels");
        assert_eq!(EditOp::Watermark { text: "".into(), font: None }.apply(white.clone()).to_rgb8(), white.to_rgb8());
        // A LUT pointing at a missing file replays as identity (never breaks the stack).
        assert_eq!(EditOp::Lut { path: "/no/such.cube".into() }.apply(white.clone()).to_rgb8(), white.to_rgb8());
    }

    #[test]
    fn face_polish_smooths_only_inside_the_face_ellipse() {
        // A noisy field: a face ellipse centred at (0.5,0.5) should get smoothed; a far corner not.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(40, 40, |x, y| {
            let n = (((x * 7 + y * 13) % 2) as u8) * 40; // 0/40 checker → high-frequency "skin texture"
            Rgb([150 + n, 120 + n, 110 + n])
        }));
        let faces = [[500, 500, 250, 300], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        let out = EditOp::FacePolish { strength: 100, faces, n: 1 }.apply(img.clone()).to_rgb8();
        let base = img.to_rgb8();
        assert_ne!(out.get_pixel(20, 20).0, base.get_pixel(20, 20).0, "centre of face smoothed");
        assert_eq!(out.get_pixel(0, 0).0, base.get_pixel(0, 0).0, "corner outside the face untouched");
        // Strength 0 or no faces → byte-identical.
        assert_eq!(EditOp::FacePolish { strength: 0, faces, n: 1 }.apply(img.clone()).to_rgb8(), base);
        assert_eq!(EditOp::FacePolish { strength: 100, faces, n: 0 }.apply(img.clone()).to_rgb8(), base);
    }

    #[test]
    fn auto_white_balance_neutralises_a_cast() {
        // A red-cast grey: means R>G>B. Gray-world should pull them together.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([180u8, 120, 90])));
        let out = EditOp::AutoWhiteBalance(100).apply(img.clone()).to_rgb8().get_pixel(0, 0).0;
        let spread_in = 180i32 - 90;
        let spread_out = out[0].max(out[1]).max(out[2]) as i32 - out[0].min(out[1]).min(out[2]) as i32;
        assert!(spread_out < spread_in, "cast reduced: {spread_out} < {spread_in}");
        // Strength 0 → identity.
        assert_eq!(EditOp::AutoWhiteBalance(0).apply(img.clone()).to_rgb8(), img.to_rgb8());
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
