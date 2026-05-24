//! Alpha-blend a resolved artefact onto a base RGBA image.
//!
//! The compositor:
//!
//! 1. Loads the artefact PNG (or other image-crate format).
//! 2. If the loaded image has no alpha channel, runs the existing
//!    `imaging::transparent` corner chroma-key as an automatic
//!    fallback (transparency tolerance 10 — absorbs JPEG noise and
//!    most anti-aliased edges without eating the artefact's edges).
//! 3. Computes the on-canvas target rect from the zone, the
//!    artefact's intrinsic aspect ratio, the scale fraction, the
//!    anchor, and the offset.
//! 4. Resizes the artefact to that rect (bilinear filter; same trade-
//!    off rationale as in `imaging::preprocess`).
//! 5. Alpha-composites pixel-by-pixel with optional global alpha
//!    multiplier and optional horizontal flip.
//!
//! No diffusion involved. v2's inpaint blend will run after this
//! pass to smooth edges and unify palette.

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};

use super::runtime::ResolvedArtefact;

/// Bilinear-quality preprocess filter (same trade-off as
/// `imaging::preprocess::PREPROCESS_FILTER`). Bicubic is barely
/// distinguishable on artefacts that get further refined by stylize
/// passes; bilinear is ~2× faster.
const RESIZE_FILTER: image::imageops::FilterType = image::imageops::FilterType::Triangle;

/// Tolerance used when auto-chroma-keying an artefact PNG that lacks
/// an alpha channel. Absorbs JPEG noise (~5) and modest anti-aliasing
/// (~10) without eating the artefact's own edges.
const AUTO_CHROMA_TOLERANCE: u8 = 10;

/// Composite every resolved artefact onto `base`, in order. The
/// order is the z-order — later artefacts cover earlier ones in
/// overlap regions.
///
/// Mutates `base` in place. Returns when every artefact has been
/// applied; errors fail the call early without partial state recovery
/// (so the caller can choose whether to retry).
pub fn composite_resolved(base: &mut DynamicImage, resolved: &[ResolvedArtefact]) -> Result<()> {
    if resolved.is_empty() {
        return Ok(());
    }
    // Promote the base to RGBA8 if it isn't already, so the blend math
    // operates on a single representation. Most plakat outputs are
    // RGB8; this is a one-time conversion.
    let mut canvas: RgbaImage = base.to_rgba8();
    for r in resolved {
        composite_one(&mut canvas, r)
            .with_context(|| format!("compositing artefact {:?}", r.artefact.name))?;
    }
    *base = DynamicImage::ImageRgba8(canvas);
    Ok(())
}

fn composite_one(canvas: &mut RgbaImage, r: &ResolvedArtefact) -> Result<()> {
    // 1. Load the artefact image (decoded).
    let loaded = image::open(&r.artefact.path)
        .with_context(|| format!("opening {}", r.artefact.path.display()))?;

    // 2. Ensure alpha. If the source has no alpha, auto-chroma-key
    //    the upper-left corner to alpha=0 (reuses the existing
    //    `plakat transparent` feature's in-memory variant).
    let mut rgba: RgbaImage = ensure_alpha(loaded);

    // 3. Compute the target rect on the canvas.
    let (tx0, ty0, tw, th) = compute_target_rect(r, rgba.width(), rgba.height(), canvas);

    // Guard: if the target rect is zero-sized, skip (degenerate).
    if tw == 0 || th == 0 {
        return Ok(());
    }

    // 4. Resize artefact to (tw, th).
    if rgba.width() != tw || rgba.height() != th {
        rgba = image::imageops::resize(&rgba, tw, th, RESIZE_FILTER);
    }

    // 5. Optional horizontal flip.
    if r.flip {
        rgba = image::imageops::flip_horizontal(&rgba);
    }

    // 6. Alpha-composite.
    alpha_composite_in_place(canvas, &rgba, tx0, ty0, r.alpha);
    Ok(())
}

/// Promote any input image to RGBA8. If the source has no alpha
/// channel, runs corner-chroma-key as the auto-fallback so the
/// "background color" becomes transparent.
fn ensure_alpha(loaded: DynamicImage) -> RgbaImage {
    let has_alpha = matches!(
        loaded.color(),
        image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F
            | image::ColorType::La8 | image::ColorType::La16
    );
    let rgba = loaded.to_rgba8();
    if has_alpha {
        rgba
    } else {
        let (out, _hit, _key) =
            crate::imaging::transparent::chroma_key_image(rgba, AUTO_CHROMA_TOLERANCE);
        out
    }
}

/// Compute the on-canvas target rect for this artefact placement.
/// Returns `(x0, y0, width, height)` — the rect where the artefact
/// will be drawn (already clamped to canvas bounds).
fn compute_target_rect(
    r: &ResolvedArtefact,
    src_w: u32,
    src_h: u32,
    canvas: &RgbaImage,
) -> (i64, i64, u32, u32) {
    let zone = r.zone;
    let zone_w = zone.width() as f32;
    let zone_h = zone.height() as f32;

    // Target height = zone height × scale fraction. Aspect-preserving
    // width derived from artefact's intrinsic aspect ratio.
    let target_h = (zone_h * r.scale_fraction).round().max(1.0);
    let aspect = src_w as f32 / src_h as f32;
    let target_w = (target_h * aspect).round().max(1.0);

    // Anchor point on the artefact (fractional, 0..=1).
    let (ax, ay) = (r.anchor.x, r.anchor.y);

    // Anchor point in zone coordinates: maps the artefact's anchor
    // point onto a specific spot in the zone. Without an offset, the
    // anchor lands at the zone's centre × the same anchor fractions —
    // so a bottom-center-anchored artefact bottom lands at the zone's
    // bottom-center, a center-anchored artefact center lands at the
    // zone's center, etc.
    //
    // With an offset, the anchor point is shifted by `offset.x` of
    // the zone width and `offset.y` of the zone height.
    let zone_anchor_x = zone.x0 as f32 + ax * zone_w + r.offset[0] * zone_w;
    let zone_anchor_y = zone.y0 as f32 + ay * zone_h + r.offset[1] * zone_h;

    // Top-left of the target rect = zone anchor − artefact anchor offset.
    let mut x0 = (zone_anchor_x - ax * target_w).round() as i64;
    let mut y0 = (zone_anchor_y - ay * target_h).round() as i64;

    let canvas_w = canvas.width() as i64;
    let canvas_h = canvas.height() as i64;

    // Clamp to canvas bounds (preserving target size when possible).
    let max_x0 = canvas_w - target_w as i64;
    let max_y0 = canvas_h - target_h as i64;
    if x0 < 0 {
        x0 = 0;
    } else if x0 > max_x0 {
        x0 = max_x0.max(0);
    }
    if y0 < 0 {
        y0 = 0;
    } else if y0 > max_y0 {
        y0 = max_y0.max(0);
    }

    // If the artefact is wider/taller than the canvas, shrink the
    // target rect to fit (rare on natural inputs; safety net).
    let avail_w = (canvas_w - x0).max(0) as u32;
    let avail_h = (canvas_h - y0).max(0) as u32;
    let tw = (target_w as u32).min(avail_w);
    let th = (target_h as u32).min(avail_h);

    (x0, y0, tw, th)
}

/// Pixel-by-pixel alpha-over composite. `overlay` is the resized
/// artefact RGBA; `(ox, oy)` is its top-left on `canvas`; `alpha_mul`
/// scales every overlay pixel's alpha channel.
fn alpha_composite_in_place(
    canvas: &mut RgbaImage,
    overlay: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    ox: i64,
    oy: i64,
    alpha_mul: f32,
) {
    let cw = canvas.width() as i64;
    let ch = canvas.height() as i64;
    let ow = overlay.width() as i64;
    let oh = overlay.height() as i64;

    // Compute the intersection of overlay rect with canvas rect.
    let x_start = ox.max(0);
    let y_start = oy.max(0);
    let x_end = (ox + ow).min(cw);
    let y_end = (oy + oh).min(ch);
    if x_start >= x_end || y_start >= y_end {
        return; // entirely off-canvas
    }

    for y in y_start..y_end {
        for x in x_start..x_end {
            // Overlay-local coords.
            let ox_local = (x - ox) as u32;
            let oy_local = (y - oy) as u32;
            let ov = overlay.get_pixel(ox_local, oy_local).0;
            let a = (ov[3] as f32) * alpha_mul / 255.0;
            if a <= 0.0 {
                continue; // fully transparent — skip the read+write
            }
            let cv = canvas.get_pixel(x as u32, y as u32).0;
            // Standard over-blend: out = ov*a + base*(1-a)
            let inv = 1.0 - a;
            let r = (ov[0] as f32 * a + cv[0] as f32 * inv).round() as u8;
            let g = (ov[1] as f32 * a + cv[1] as f32 * inv).round() as u8;
            let b = (ov[2] as f32 * a + cv[2] as f32 * inv).round() as u8;
            // Alpha: preserve canvas alpha but never decrease it (the
            // composited result is at least as opaque as the canvas
            // was).
            let a_out = ((cv[3] as f32) * inv + 255.0 * a).round() as u8;
            canvas.put_pixel(x as u32, y as u32, Rgba([r, g, b, a_out]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artefacts::anchor::Anchor;
    use crate::artefacts::library::Artefact;
    use crate::artefacts::zones::{Rect, ZoneRef};
    use std::path::PathBuf;

    fn fake_resolved(canvas_w: u32, canvas_h: u32, scale_frac: f32, anchor: Anchor) -> ResolvedArtefact {
        ResolvedArtefact {
            artefact: Artefact {
                name: "test".to_string(),
                category: "test".to_string(),
                path: PathBuf::from("/tmp/nonexistent.png"),
                natural_zone: ZoneRef::full(),
                natural_size_pct: scale_frac, // not used directly here
                anchor,
                license: None,
                license_url: None,
                tags: vec![],
            },
            zone: Rect {
                x0: 0,
                y0: 0,
                x1: canvas_w,
                y1: canvas_h,
            },
            scale_fraction: scale_frac,
            offset: [0.0, 0.0],
            anchor,
            flip: false,
            alpha: 1.0,
        }
    }

    #[test]
    fn target_rect_centered_anchor() {
        let r = fake_resolved(800, 400, 0.5, Anchor::CENTER);
        let canvas = RgbaImage::new(800, 400);
        // 100×100 source, scale 0.5 of 400 height = 200 → aspect 1.0
        // → 200×200. Centered in (0,0)-(800,400): top-left at (300, 100).
        let (x, y, w, h) = compute_target_rect(&r, 100, 100, &canvas);
        assert_eq!((x, y, w, h), (300, 100, 200, 200));
    }

    #[test]
    fn target_rect_bottom_anchor() {
        let r = fake_resolved(800, 400, 0.5, Anchor::BOTTOM_CENTER);
        let canvas = RgbaImage::new(800, 400);
        // 200×200 target; anchor bottom-center → artefact bottom lands
        // at zone bottom (y=400), artefact center-x lands at zone
        // center-x (x=400). Top-left = (300, 200).
        let (x, y, w, h) = compute_target_rect(&r, 100, 100, &canvas);
        assert_eq!((x, y, w, h), (300, 200, 200, 200));
    }

    #[test]
    fn target_rect_clamps_when_off_canvas() {
        // Anchor bottom_right on a zone that's the full canvas, but
        // with a positive offset that would push the artefact off the
        // right edge → should clamp back.
        let mut r = fake_resolved(800, 400, 0.5, Anchor::BOTTOM_RIGHT);
        r.offset = [0.5, 0.0]; // would push right edge to 1200
        let canvas = RgbaImage::new(800, 400);
        let (x, y, w, h) = compute_target_rect(&r, 100, 100, &canvas);
        // Should clamp so artefact fits inside canvas: x + w ≤ 800
        assert!(x + w as i64 <= 800, "clamped rect overflows: ({x},{y},{w},{h})");
        assert!(y + h as i64 <= 400);
    }

    #[test]
    fn alpha_composite_blends_correctly() {
        // Canvas: solid red. Overlay: solid blue with alpha=128 over
        // a 50×50 region. Result: top-left 50×50 should be a 50/50
        // red/blue mix (approximately).
        let mut canvas = RgbaImage::from_pixel(100, 100, Rgba([255, 0, 0, 255]));
        let overlay = RgbaImage::from_pixel(50, 50, Rgba([0, 0, 255, 128]));
        alpha_composite_in_place(&mut canvas, &overlay, 0, 0, 1.0);

        let blended = canvas.get_pixel(10, 10).0;
        // a = 128/255 ≈ 0.502; r ≈ 255 * 0.498 ≈ 127; b ≈ 255 * 0.502 ≈ 128
        assert!((blended[0] as i32 - 127).abs() <= 1, "red wrong: {blended:?}");
        assert!(blended[1] < 2, "green should stay 0: {blended:?}");
        assert!((blended[2] as i32 - 128).abs() <= 1, "blue wrong: {blended:?}");

        // Outside the overlay region, canvas should be untouched.
        let unblended = canvas.get_pixel(80, 80).0;
        assert_eq!(unblended, [255, 0, 0, 255]);
    }

    #[test]
    fn alpha_composite_skips_fully_transparent_pixels() {
        let mut canvas = RgbaImage::from_pixel(50, 50, Rgba([100, 100, 100, 255]));
        let overlay = RgbaImage::from_pixel(20, 20, Rgba([0, 0, 0, 0])); // fully transparent
        alpha_composite_in_place(&mut canvas, &overlay, 10, 10, 1.0);
        // Canvas should be entirely unchanged.
        for (_, _, px) in canvas.enumerate_pixels() {
            assert_eq!(px.0, [100, 100, 100, 255]);
        }
    }
}
