//! The compositing pass (RFC §8.4). Runs after generation, before scoring:
//!
//! ```text
//! rendered image + realised landmark set (FaceMetrics)
//!   → for each detail, in deterministic z-order (fields → areal → linear → piercing → worn jewelry):
//!       resolve its anchor through the REALISED landmarks (not the requested ones)
//!       cull if out of frame / behind the visible surface (reported, not silently dropped)
//!       generate or load its overlay; scale by the realised face width
//!       estimate local light direction from the face's own shading
//!       alpha-composite
//!   → union the affected regions into one mask (for optional harmonisation)
//! ```
//!
//! Resolving anchors through the *realised* landmarks is the crux (§8.4): it puts the mole below the
//! eye that exists, not the eye that was specified. Deterministic + byte-stable (no harmonisation here);
//! harmonisation is the one stochastic step and is layered on top by the CLI.

use super::overlay::{self, Light};
use crate::persona::scorecard::FaceMetrics;
use crate::persona::spec::{Color, Mark, PersonaSpec};
use image::{GrayImage, Luma, RgbImage};

/// A detail that could not be placed (§8.4/§8.5) — reported so the scorecard doesn't penalise it.
#[derive(Debug, Clone)]
pub struct Culled {
    pub kind: String,
    pub reason: String,
}

/// Result of the compositing pass.
#[derive(Debug, Clone)]
pub struct CompositeResult {
    pub image: RgbImage,
    /// Union of the affected regions (feathered), for an optional harmonisation img2img.
    pub mask: GrayImage,
    pub culled: Vec<Culled>,
    pub placed: usize,
    /// The estimated scene light (also reported to the user).
    pub light: Light,
}

/// Map a crop-normalised point to full-image pixels via the crop origin (§ the `crop_origin` field).
fn full_px(m: &FaceMetrics, p: (f32, f32)) -> (f32, f32) {
    (m.crop_origin.0 as f32 + p.0 * m.crop.width() as f32, m.crop_origin.1 as f32 + p.1 * m.crop.height() as f32)
}

/// Median sRGB over a small disc of the crop — a robust skin sample.
fn crop_disc_rgb(m: &FaceMetrics, cx: f32, cy: f32, r: f32) -> [u8; 3] {
    let (w, h) = (m.crop.width() as f32, m.crop.height() as f32);
    let (mut rs, mut gs, mut bs, mut n) = (0u32, 0u32, 0u32, 0u32);
    let (x0, y0) = ((cx - r).max(0.0) as u32, (cy - r).max(0.0) as u32);
    let (x1, y1) = ((cx + r).min(w - 1.0) as u32, (cy + r).min(h - 1.0) as u32);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let p = m.crop.get_pixel(x, y).0;
            rs += p[0] as u32;
            gs += p[1] as u32;
            bs += p[2] as u32;
            n += 1;
        }
    }
    let n = n.max(1);
    [(rs / n) as u8, (gs / n) as u8, (bs / n) as u8]
}

fn luma(c: [u8; 3]) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
}

/// Estimate the scene light from the face's own shading: a brightness gradient across the cheeks
/// (horizontal) and forehead↔chin (vertical). Falls back to the top-left default if the face is flat.
fn estimate_light(m: &FaceMetrics) -> Light {
    use crate::persona::geometry::topology::*;
    let (cw, ch) = (m.crop.width() as f32, m.crop.height() as f32);
    let at = |i: usize| (m.landmarks[i].0 * cw, m.landmarks[i].1 * ch);
    let r = (m.face_w * cw * 0.07).max(4.0);
    let cheek = |pupil: usize, corner: usize| {
        let (px, py) = at(pupil);
        let (mx, my) = at(corner);
        crop_disc_rgb(m, (px + mx) / 2.0, (py + my) / 2.0, r)
    };
    let right = luma(cheek(PUPIL_RIGHT, MOUTH_CORNER_RIGHT));
    let left = luma(cheek(PUPIL_LEFT, MOUTH_CORNER_LEFT));
    let (fx, fy) = at(51); // nose-bridge top ≈ mid-forehead-ish
    let forehead = luma(crop_disc_rgb(m, fx, fy - m.face_h * ch * 0.15, r));
    let (chx, chy) = at(CHIN);
    let chin = luma(crop_disc_rgb(m, chx, chy, r));
    let gx = left - right; // brighter left cheek → light from image-left (dx<0)
    let gy = chin - forehead; // brighter chin → light from below (dy>0)
    let dx = -gx;
    let dy = gy;
    let n = (dx * dx + dy * dy).sqrt();
    if n < 6.0 {
        Light::default()
    } else {
        Light { dx: dx / n, dy: dy / n }
    }
}

/// sRGB for a spec `Color` (named or Lab), with a fallback if unset/unknown.
fn color_rgb(c: Option<&Color>, fallback: [u8; 3]) -> [u8; 3] {
    match c {
        Some(Color::Named(n)) => named_rgb(n).unwrap_or(fallback),
        Some(Color::Lab { lab }) => lab_to_rgb(*lab),
        None => fallback,
    }
}

fn named_rgb(n: &str) -> Option<[u8; 3]> {
    Some(match n {
        "brown" | "dark-brown" => [90, 58, 42],
        "black" => [40, 34, 32],
        "red" | "pink" => [180, 90, 90],
        "tan" | "light-brown" => [140, 100, 74],
        "blue" | "blue-black" => [60, 60, 90],
        "white" | "pale" => [225, 210, 200],
        _ => return overlay_named(n),
    })
}
fn overlay_named(n: &str) -> Option<[u8; 3]> {
    // fall through to the jewelry palettes for shared names (e.g. a "gold" tattoo tint).
    let m = overlay::metal_colour(n);
    if m != [200, 204, 210] {
        Some(m)
    } else {
        None
    }
}

/// CIELAB (D65) → sRGB u8.
fn lab_to_rgb(lab: [f32; 3]) -> [u8; 3] {
    let (l, a, b) = (lab[0], lab[1], lab[2]);
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let inv = |t: f32| {
        let t3 = t * t * t;
        if t3 > 0.008856 { t3 } else { (t - 16.0 / 116.0) / 7.787 }
    };
    let (x, y, z) = (inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883);
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let bl = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    let gamma = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let v = if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        (v * 255.0).round().clamp(0.0, 255.0) as u8
    };
    [gamma(r), gamma(g), gamma(bl)]
}

/// Alpha-composite an overlay centred at `(cx, cy)` full-image pixels; stamp its footprint into `mask`.
fn stamp(img: &mut RgbImage, mask: &mut GrayImage, ov: &image::RgbaImage, cx: f32, cy: f32) {
    let (ow, oh) = ov.dimensions();
    let (ox, oy) = (cx - ow as f32 / 2.0, cy - oh as f32 / 2.0);
    let (iw, ih) = (img.width(), img.height());
    for (x, y, p) in ov.enumerate_pixels() {
        let a = p.0[3] as f32 / 255.0;
        if a <= 0.0 {
            continue;
        }
        let (dx, dy) = (ox + x as f32, oy + y as f32);
        if dx < 0.0 || dy < 0.0 || dx >= iw as f32 || dy >= ih as f32 {
            continue;
        }
        let (dx, dy) = (dx as u32, dy as u32);
        let b = img.get_pixel(dx, dy).0;
        let mix = |i: usize| (p.0[i] as f32 * a + b[i] as f32 * (1.0 - a)).round() as u8;
        img.put_pixel(dx, dy, image::Rgb([mix(0), mix(1), mix(2)]));
        mask.put_pixel(dx, dy, Luma([255]));
    }
}

/// Resolve a mark's anchor through the realised landmarks → full-image pixels (§8.4). `None` if the
/// mark has no anchor or the anchor names an unknown region.
fn mark_anchor_px(mark: &Mark, m: &FaceMetrics) -> Option<(f32, f32)> {
    let a = mark.anchor.as_ref()?;
    let p = crate::persona::scorecard::resolve_anchor(a, m)?; // crop-normalised
    Some(full_px(m, p))
}

/// z-order class (§8.9): lower composites first (further back).
fn z_order(mark: &Mark) -> u8 {
    match mark.kind.as_deref() {
        Some("freckles") | Some("pockmarks") | Some("mottling") => 0, // skin fields
        Some("birthmark") | Some("mole") | Some("beauty-mark") => 1,  // areal
        Some("scar") => 2,                                            // linear
        _ => 1,
    }
}

/// The compositing pass. Deterministic given `(base, spec, m, seed)`; no harmonisation.
pub fn composite_details(base: &RgbImage, spec: &PersonaSpec, m: &FaceMetrics, seed: u64) -> CompositeResult {
    let mut img = base.clone();
    let mut mask = GrayImage::from_pixel(base.width(), base.height(), Luma([0]));
    let mut culled = Vec::new();
    let mut placed = 0usize;
    let light = estimate_light(m);
    let face_px = (m.face_w * m.crop.width() as f32).max(8.0);
    // a representative skin tone (right cheek), for scar/relief colour ramps.
    let (cw, ch) = (m.crop.width() as f32, m.crop.height() as f32);
    let skin = {
        use crate::persona::geometry::topology::{MOUTH_CORNER_RIGHT, PUPIL_RIGHT};
        let (px, py) = (m.landmarks[PUPIL_RIGHT].0 * cw, m.landmarks[PUPIL_RIGHT].1 * ch);
        let (mx, my) = (m.landmarks[MOUTH_CORNER_RIGHT].0 * cw, m.landmarks[MOUTH_CORNER_RIGHT].1 * ch);
        crop_disc_rgb(m, (px + mx) / 2.0, (py + my) / 2.0, (m.face_w * cw * 0.06).max(3.0))
    };

    // Collect + sort marks by z-order (stable by original index within a class).
    let empty = Vec::new();
    let marks = spec.marks.as_ref().unwrap_or(&empty);
    let mut order: Vec<usize> = (0..marks.len()).collect();
    order.sort_by_key(|&i| (z_order(&marks[i]), i));

    const DETAIL_CAP: usize = 24;
    if marks.len() > DETAIL_CAP {
        culled.push(Culled { kind: "*".into(), reason: format!("{} marks exceed the cap {DETAIL_CAP}; extras skipped", marks.len()) });
    }

    for &i in order.iter().take(DETAIL_CAP) {
        let mark = &marks[i];
        let kind = mark.kind.as_deref().unwrap_or("mark");
        let mseed = seed ^ (0x1000 + i as u64).wrapping_mul(0x9E3779B97F4A7C15);

        // distributional fields: region → disc mask → freckle field (no point anchor).
        if matches!(kind, "freckles" | "pockmarks" | "mottling") {
            let Some(region) = mark.region.as_deref().or(Some("right-cheek")) else { continue };
            let Some(centre) = region_centre_px(region, m) else {
                culled.push(Culled { kind: kind.into(), reason: format!("unknown region `{region}`") });
                continue;
            };
            let rad = face_px * 0.28;
            let fmask = disc_mask(base.width(), base.height(), centre, rad);
            let colour = color_rgb(mark.color.as_ref(), [150, 95, 70]);
            let field = overlay::freckle_field(&fmask, mark.density.unwrap_or(0.5), colour, mseed);
            stamp(&mut img, &mut mask, &field, 0.0 + field.width() as f32 / 2.0, field.height() as f32 / 2.0);
            placed += 1;
            continue;
        }

        // positional marks: need an anchor.
        let Some((cx, cy)) = mark_anchor_px(mark, m) else {
            culled.push(Culled { kind: kind.into(), reason: "no resolvable anchor".into() });
            continue;
        };
        if cx < 0.0 || cy < 0.0 || cx >= base.width() as f32 || cy >= base.height() as f32 {
            culled.push(Culled { kind: kind.into(), reason: "anchor outside the frame".into() });
            continue;
        }
        let size = (mark.size.unwrap_or(0.04) * face_px).max(4.0);
        let ov = match kind {
            "scar" => {
                let len = (mark.length.unwrap_or(0.12) * face_px).max(6.0) as u32;
                let wid = (mark.width.unwrap_or(0.012) * face_px).max(2.0) as u32;
                overlay::scar(len, wid, mark.orientation.unwrap_or(0.4), mark.maturity.unwrap_or(0.6), mark.relief.unwrap_or(0.5), skin, light)
            }
            "birthmark" => {
                let colour = color_rgb(mark.color.as_ref(), [120, 85, 70]);
                overlay::birthmark(size as u32, mark.aspect.unwrap_or(1.0), mark.edge.as_deref().unwrap_or("soft"), mark.intensity.unwrap_or(0.7), colour, mseed)
            }
            // mole / beauty-mark / default
            _ => {
                let colour = color_rgb(mark.color.as_ref(), [90, 58, 42]);
                overlay::mole(size as u32, colour, mark.raised.unwrap_or(0.5), light)
            }
        };
        stamp(&mut img, &mut mask, &ov, cx, cy);
        placed += 1;
    }

    // feather the union mask a little so an optional harmonisation extends past each overlay (§8.4).
    feather(&mut mask, 2);
    CompositeResult { image: img, mask, culled, placed, light }
}

/// The full-image pixel centre of a named region (for distributional fields).
fn region_centre_px(region: &str, m: &FaceMetrics) -> Option<(f32, f32)> {
    let idxs = crate::persona::geometry::topology::named_region(region)?;
    let n = idxs.len() as f32;
    let (bx, by) = idxs.iter().fold((0.0, 0.0), |(ax, ay), &i| (ax + m.landmarks[i].0 / n, ay + m.landmarks[i].1 / n));
    Some(full_px(m, (bx, by)))
}

fn disc_mask(w: u32, h: u32, centre: (f32, f32), r: f32) -> GrayImage {
    let mut mask = GrayImage::from_pixel(w, h, Luma([0]));
    let (x0, y0) = ((centre.0 - r).max(0.0) as u32, (centre.1 - r).max(0.0) as u32);
    let (x1, y1) = ((centre.0 + r).min((w - 1) as f32) as u32, (centre.1 + r).min((h - 1) as f32) as u32);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if ((x as f32 - centre.0).powi(2) + (y as f32 - centre.1).powi(2)).sqrt() <= r {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
    }
    mask
}

/// Dilate the mask by `r` pixels (box) — a cheap feather for the harmonisation region.
fn feather(mask: &mut GrayImage, r: i32) {
    let (w, h) = mask.dimensions();
    let src = mask.clone();
    for y in 0..h {
        for x in 0..w {
            if src.get_pixel(x, y).0[0] > 0 {
                continue;
            }
            let mut near = false;
            'o: for dy in -r..=r {
                for dx in -r..=r {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h && src.get_pixel(nx as u32, ny as u32).0[0] > 0 {
                        near = true;
                        break 'o;
                    }
                }
            }
            if near {
                mask.put_pixel(x, y, Luma([128]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrips_a_known_colour() {
        // mid-grey-ish: L50 a0 b0 → equal channels around 119.
        let rgb = lab_to_rgb([53.39, 0.0, 0.0]);
        assert!((rgb[0] as i32 - rgb[1] as i32).abs() <= 2 && (rgb[1] as i32 - rgb[2] as i32).abs() <= 2);
        assert!(rgb[0] > 100 && rgb[0] < 140);
    }

    #[test]
    fn named_colours_resolve() {
        assert_eq!(named_rgb("brown"), Some([90, 58, 42]));
        assert!(named_rgb("gold").is_some()); // shared with the jewelry palette
        assert!(named_rgb("not-a-colour").is_none());
    }

    // A synthetic FaceMetrics from the mean template — lets the compositing pass be exercised (and its
    // determinism pinned) WITHOUT the SCRFD/PIPNet weights. This is the P3 corpus.
    fn synthetic_metrics(dim: u32) -> FaceMetrics {
        use crate::persona::geometry::{mean_template, topology::*};
        let lm: Vec<(f32, f32)> = mean_template(false).to_vec();
        let fx0 = CONTOUR.clone().map(|i| lm[i].0).fold(f32::INFINITY, f32::min);
        let fx1 = CONTOUR.clone().map(|i| lm[i].0).fold(f32::NEG_INFINITY, f32::max);
        let fy0 = CONTOUR.clone().map(|i| lm[i].1).fold(f32::INFINITY, f32::min);
        let fy1 = CONTOUR.map(|i| lm[i].1).fold(f32::NEG_INFINITY, f32::max);
        FaceMetrics {
            interpupillary_over_facewidth: 0.4,
            mouth_over_facewidth: 0.4,
            face_aspect: 1.3,
            landmarks: lm,
            face_w: fx1 - fx0,
            face_h: fy1 - fy0,
            detection_score: 1.0,
            crop: image::RgbImage::from_pixel(dim, dim, image::Rgb([205, 165, 140])),
            crop_origin: (0, 0),
        }
    }

    #[test]
    fn compositing_is_byte_stable_on_a_synthetic_face() {
        use crate::persona::spec::PersonaSpec;
        let spec = PersonaSpec::from_hjson(
            "{ schema: \"persona/1\"\n marks: [ { kind: \"mole\"\n anchor: { region: \"left-cheek\" }\n size: 0.04 }, { kind: \"scar\"\n anchor: { region: \"forehead-centre\" }\n length: 0.12\n maturity: 0.3 } ] }",
        )
        .unwrap();
        let m = synthetic_metrics(256);
        let base = image::RgbImage::from_pixel(256, 256, image::Rgb([205, 165, 140]));
        let r = composite_details(&base, &spec, &m, 99);
        assert_eq!(r.placed, 2);
        assert!(r.culled.is_empty());
        let mut acc: u64 = 1469598103934665603;
        for p in r.image.pixels() {
            for &c in &p.0 {
                acc = (acc ^ c as u64).wrapping_mul(1099511628211);
            }
        }
        assert_eq!(acc, 14264266837389475466, "composite corpus changed — update the golden intentionally");
    }
}
