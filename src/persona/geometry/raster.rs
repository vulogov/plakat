//! Conditioning-map rasteriser (RFC §10.3). From a realised landmark set the engine rasterises, on
//! demand, the maps a downstream ControlNet / repair / preview step consumes. Pure Rust, no weights,
//! **byte-stable on-box** (aliased integer drawing via `imageproc`, no RNG) — the §5.2 contract.
//!
//! Phase-G emphasis (see `Documentation/PERSONA_GATING.md`): the mesh map is an SD1.5/2.1 Tier-A
//! bonus; the cross-family value is the depth / region-mask / detail-overlay outputs. Every map here
//! is a pure function of `(landmarks, size)` (+ small style/param args).
//!
//! | fn | map | consumer |
//! |---|---|---|
//! | [`mesh_map`] | feature polylines/loops | face-landmark ControlNet (per-family style) |
//! | [`wireframe`] | thin lines + points | TUI / SD mesh CN |
//! | [`depth_proxy`] | smoothed grayscale relief | depth ControlNet |
//! | [`face_skeleton`] | OpenPose head keypoints | OpenPose-union CN (cross-family) |
//! | [`region_mask`] | filled feature polygon | targeted repair / regional prompt / compositing |
//! | [`dentition_hint`] | tooth arch in the aperture | mouth-region conditioning (§8.7) |
//! | [`detail_overlay`] | markers at anchors | preview + compositor plan (expanded in P3) |

use super::template::Template;
use super::topology::*;
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_polygon_mut};
use imageproc::point::Point;

/// Per-family drawing convention for the mesh map. `Generic` = feature polylines in white; `MediaPipe`
/// = the coloured feature convention the SD1.5/2.1 `ControlNetMediaPipeFace` checkpoint was trained on
/// (approximated by per-feature hues here; the exact 468-pt tessellation is out of scope for WFLW-98).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshStyle {
    Generic,
    MediaPipe,
}

fn px(pt: (f32, f32), w: u32, h: u32) -> (f32, f32) {
    (pt.0 * (w - 1) as f32, pt.1 * (h - 1) as f32)
}

fn ipoint(pt: (f32, f32), w: u32, h: u32) -> Point<i32> {
    let (x, y) = px(pt, w, h);
    Point::new(x.round() as i32, y.round() as i32)
}

/// Draw a feature group's polyline (closing the loop if `closed`) with the given colour + thickness.
fn draw_group(img: &mut RgbImage, lm: &Template, g: &DrawGroup, colour: Rgb<u8>, thick: i32) {
    let (w, h) = (img.width(), img.height());
    let idxs: Vec<usize> = g.range.clone().collect();
    for pair in idxs.windows(2) {
        thick_line(img, px(lm[pair[0]], w, h), px(lm[pair[1]], w, h), colour, thick);
    }
    if g.closed && idxs.len() > 2 {
        thick_line(img, px(lm[idxs[idxs.len() - 1]], w, h), px(lm[idxs[0]], w, h), colour, thick);
    }
}

/// A thick line = a few parallel offset lines (imageproc has only 1-px segments). Byte-stable.
fn thick_line(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), colour: Rgb<u8>, thick: i32) {
    let r = (thick - 1).max(0) / 2;
    for oy in -r..=r {
        for ox in -r..=r {
            draw_line_segment_mut(img, (a.0 + ox as f32, a.1 + oy as f32), (b.0 + ox as f32, b.1 + oy as f32), colour);
        }
    }
}

/// Hue per feature group for the coloured styles.
fn feature_colour(name: &str) -> Rgb<u8> {
    match name {
        "contour" => Rgb([80, 170, 255]),
        "brow-right" | "brow-left" => Rgb([255, 190, 90]),
        "nose-bridge" | "nose-base" => Rgb([150, 255, 130]),
        "eye-right" | "eye-left" => Rgb([255, 120, 190]),
        "lip-outer" => Rgb([255, 90, 90]),
        "lip-inner" => Rgb([190, 70, 200]),
        _ => Rgb([255, 255, 255]),
    }
}

/// Feature polylines/loops on black. `MediaPipe` colours per feature; `Generic` draws white.
pub fn mesh_map(lm: &Template, size: u32, style: MeshStyle) -> RgbImage {
    let mut img = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
    let thick = (size / 256).max(1) as i32 * 2 + 1;
    for g in DRAW_GROUPS {
        let colour = match style {
            MeshStyle::Generic => Rgb([255, 255, 255]),
            MeshStyle::MediaPipe => feature_colour(g.name),
        };
        draw_group(&mut img, lm, g, colour, thick);
    }
    // pupils as small dots
    for &p in &[PUPIL_RIGHT, PUPIL_LEFT] {
        let (x, y) = px(lm[p], size, size);
        draw_filled_circle_mut(&mut img, (x.round() as i32, y.round() as i32), thick, Rgb([255, 255, 255]));
    }
    img
}

/// Thin white wireframe + landmark points (TUI preview / SD mesh CN).
pub fn wireframe(lm: &Template, size: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
    for g in DRAW_GROUPS {
        draw_group(&mut img, lm, g, Rgb([200, 200, 200]), 1);
    }
    for &(x, y) in lm.iter() {
        let (px_, py_) = px((x, y), size, size);
        img.put_pixel(px_.round().clamp(0.0, (size - 1) as f32) as u32, py_.round().clamp(0.0, (size - 1) as f32) as u32, Rgb([255, 255, 255]));
    }
    img
}

/// A per-region depth profile → a smoothed grayscale relief (crude but directionally correct, §10.3).
/// Splats hand-authored per-feature depth bumps and normalises within the face-oval mask; the nose
/// ridge is brightest (nearest), the outer contour darkest. `relief` adds raised-detail contribution.
pub fn depth_proxy(lm: &Template, size: u32) -> GrayImage {
    let face = region_mask(lm, size, Region::Face);
    // accumulate a float depth field
    let n = (size * size) as usize;
    let mut depth = vec![0.0f32; n];
    let at = |x: f32, y: f32| px((x, y), size, size);

    // broad face dome (centre of the face box, nearest at the middle).
    let (fc_x, fc_y) = at(0.5, lm[NOSE_TIP].1);
    let face_r = size as f32 * 0.55;
    // nose ridge: a bright line down the bridge to the tip.
    let ridge: Vec<(f32, f32)> = (NOSE_BRIDGE_TOP..=NOSE_TIP).map(|i| at(lm[i].0, lm[i].1)).collect();

    for y in 0..size {
        for x in 0..size {
            let idx = (y * size + x) as usize;
            if face.get_pixel(x, y).0[0] == 0 {
                continue; // outside the face
            }
            let (fx, fy) = (x as f32, y as f32);
            // dome falloff
            let dd = ((fx - fc_x).powi(2) + (fy - fc_y).powi(2)).sqrt() / face_r;
            let mut d = (1.0 - dd).clamp(0.0, 1.0) * 0.6;
            // nose ridge bump
            let ridge_d = ridge.iter().map(|&(rx, ry)| ((fx - rx).powi(2) + (fy - ry).powi(2)).sqrt()).fold(f32::INFINITY, f32::min);
            d += (1.0 - (ridge_d / (size as f32 * 0.06)).min(1.0)) * 0.4;
            // eye sockets recess slightly
            for &p in &[PUPIL_RIGHT, PUPIL_LEFT] {
                let (ex, ey) = at(lm[p].0, lm[p].1);
                let ed = ((fx - ex).powi(2) + (fy - ey).powi(2)).sqrt() / (size as f32 * 0.05);
                d -= (1.0 - ed.min(1.0)) * 0.18;
            }
            depth[idx] = d.max(0.0);
        }
    }
    // normalise within the mask and emit
    let maxd = depth.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let mut out = GrayImage::from_pixel(size, size, Luma([0]));
    for y in 0..size {
        for x in 0..size {
            let idx = (y * size + x) as usize;
            if face.get_pixel(x, y).0[0] != 0 {
                out.put_pixel(x, y, Luma([(depth[idx] / maxd * 255.0).round() as u8]));
            }
        }
    }
    out
}

/// OpenPose head keypoints (nose, both eyes, both "ears" ≈ contour temples) drawn in the OpenPose
/// colour convention, with the connecting bones — the cross-family OpenPose-union CN path (§10.3).
pub fn face_skeleton(lm: &Template, size: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
    let s = size as i32;
    let kp = |i: usize| {
        let (x, y) = px(lm[i], size, size);
        (x, y)
    };
    // OpenPose-18 head subset: 0 nose, 14 R-eye, 15 L-eye, 16 R-ear, 17 L-ear.
    let nose = kp(NOSE_TIP);
    let reye = kp(PUPIL_RIGHT);
    let leye = kp(PUPIL_LEFT);
    let rear = kp(0); // contour right temple ≈ ear
    let lear = kp(32);
    let bones = [
        (nose, reye, Rgb([255, 0, 85])),
        (nose, leye, Rgb([255, 0, 170])),
        (reye, rear, Rgb([170, 0, 255])),
        (leye, lear, Rgb([85, 0, 255])),
    ];
    let thick = (size / 256).max(1) as i32 * 2 + 1;
    for (a, b, c) in bones {
        thick_line(&mut img, a, b, c, thick);
    }
    let dot = (thick + 2).max(3);
    for (pt, c) in [
        (nose, Rgb([255, 0, 0])),
        (reye, Rgb([255, 85, 0])),
        (leye, Rgb([255, 170, 0])),
        (rear, Rgb([0, 255, 0])),
        (lear, Rgb([0, 255, 85])),
    ] {
        draw_filled_circle_mut(&mut img, (pt.0.round().clamp(0.0, (s - 1) as f32) as i32, pt.1.round() as i32), dot, c);
    }
    img
}

/// A named feature region for masking / repair (§10.3/§12.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Face,
    EyeRight,
    EyeLeft,
    Mouth,
    Nose,
    BrowRight,
    BrowLeft,
}

fn polygon(lm: &Template, size: u32, idxs: impl Iterator<Item = usize>) -> Vec<Point<i32>> {
    let mut pts: Vec<Point<i32>> = idxs.map(|i| ipoint(lm[i], size, size)).collect();
    // imageproc's draw_polygon_mut requires the first != last vertex.
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    pts
}

/// A filled binary mask (255 inside, 0 outside) for a feature region.
pub fn region_mask(lm: &Template, size: u32, region: Region) -> GrayImage {
    let mut img = GrayImage::from_pixel(size, size, Luma([0]));
    let white = Luma([255u8]);
    let poly: Vec<Point<i32>> = match region {
        // face = contour, closed across the top through the brows (37 → 46) so the forehead is inside.
        Region::Face => {
            let mut v: Vec<usize> = CONTOUR.collect();
            v.extend([46usize, 37]); // up over the brows and back
            polygon(lm, size, v.into_iter())
        }
        Region::EyeRight => polygon(lm, size, MASK_EYE_RIGHT),
        Region::EyeLeft => polygon(lm, size, MASK_EYE_LEFT),
        Region::Mouth => polygon(lm, size, MASK_MOUTH),
        Region::Nose => polygon(lm, size, [51usize, 55, 57, 59].into_iter()),
        Region::BrowRight => brow_polygon(lm, size, BROW_RIGHT),
        Region::BrowLeft => brow_polygon(lm, size, BROW_LEFT),
    };
    if poly.len() >= 3 {
        draw_polygon_mut(&mut img, &poly, white);
    }
    img
}

/// A brow is an open arc; thicken it into a thin filled band for masking.
fn brow_polygon(lm: &Template, size: u32, range: std::ops::Range<usize>) -> Vec<Point<i32>> {
    let (w, h) = (size, size);
    let mut top: Vec<Point<i32>> = range.clone().map(|i| ipoint(lm[i], w, h)).collect();
    let mut bot: Vec<Point<i32>> = range.rev().map(|i| {
        let (x, y) = px(lm[i], w, h);
        Point::new(x.round() as i32, (y + size as f32 * 0.03).round() as i32)
    }).collect();
    top.append(&mut bot);
    top
}

/// Per-tooth arch inside the (open-mouth) inner-lip contour (§8.7). `count` upper teeth are drawn as
/// vertical divisions across the top of the aperture. A no-op-ish thin line if the mouth is closed.
pub fn dentition_hint(lm: &Template, size: u32, upper_teeth: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
    let (x0, _y0, x1, y1, ytop) = inner_lip_bounds(lm, size);
    let arch_y = ytop + (y1 - ytop) * 0.15;
    let colour = Rgb([220, 220, 220]);
    // aperture outline
    thick_line(&mut img, (x0, arch_y), (x1, arch_y), colour, 2);
    let n = upper_teeth.max(1);
    for k in 0..=n {
        let t = k as f32 / n as f32;
        let x = x0 + (x1 - x0) * t;
        thick_line(&mut img, (x, arch_y), (x, y1), colour, 1);
    }
    img
}

/// (x0, y0, x1, y1, y_top_of_aperture) of the inner lip in pixels.
fn inner_lip_bounds(lm: &Template, size: u32) -> (f32, f32, f32, f32, f32) {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in LIP_INNER {
        let (x, y) = px(lm[i], size, size);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    (x0, y0, x1, y1, y0)
}

/// Render detail markers at the given crop-normalised anchor positions (preview + compositor plan,
/// §10.3). Minimal here — the P3 detail subsystem drives shapes/relief/light; this just plots them.
pub fn detail_overlay(size: u32, marks: &[(f32, f32)]) -> RgbImage {
    let mut img = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
    let r = (size / 128).max(2) as i32;
    for &(mx, my) in marks {
        let (x, y) = px((mx, my), size, size);
        draw_filled_circle_mut(&mut img, (x.round() as i32, y.round() as i32), r, Rgb([255, 60, 60]));
    }
    img
}

#[cfg(test)]
mod tests {
    use super::super::template::mean_template;
    use super::*;

    fn nonblack(img: &RgbImage) -> usize {
        img.pixels().filter(|p| p.0 != [0, 0, 0]).count()
    }
    fn white_px(img: &GrayImage) -> usize {
        img.pixels().filter(|p| p.0[0] > 0).count()
    }

    #[test]
    fn maps_are_the_right_size_and_nonempty() {
        let lm = mean_template(false);
        for style in [MeshStyle::Generic, MeshStyle::MediaPipe] {
            let m = mesh_map(&lm, 256, style);
            assert_eq!(m.dimensions(), (256, 256));
            assert!(nonblack(&m) > 100, "mesh should draw something");
        }
        assert!(nonblack(&wireframe(&lm, 256)) > 100);
        assert!(nonblack(&face_skeleton(&lm, 256)) > 50);
    }

    #[test]
    fn region_masks_are_filled_and_ordered_by_area() {
        let lm = mean_template(false);
        let face = white_px(&region_mask(&lm, 256, Region::Face));
        let eye = white_px(&region_mask(&lm, 256, Region::EyeRight));
        let mouth = white_px(&region_mask(&lm, 256, Region::Mouth));
        assert!(eye > 0 && mouth > 0 && face > 0, "masks non-empty");
        assert!(face > mouth && mouth > eye, "face > mouth > eye area: {face} {mouth} {eye}");
        // right-eye mask sits left of the left-eye mask (centroid x).
        let cx = |r| {
            let m = region_mask(&lm, 256, r);
            let (mut sx, mut n) = (0u64, 0u64);
            for (x, _y, p) in m.enumerate_pixels() {
                if p.0[0] > 0 {
                    sx += x as u64;
                    n += 1;
                }
            }
            sx as f64 / n.max(1) as f64
        };
        assert!(cx(Region::EyeRight) < cx(Region::EyeLeft));
    }

    #[test]
    fn depth_is_brightest_on_the_nose_ridge() {
        let lm = mean_template(false);
        let d = depth_proxy(&lm, 256);
        let sample = |i: usize| {
            let (x, y) = px(lm[i], 256, 256);
            d.get_pixel(x as u32, y as u32).0[0]
        };
        assert!(sample(NOSE_TIP) > sample(0), "nose brighter than the jaw contour");
        assert!(sample(52) > 100, "bridge is bright");
    }

    #[test]
    fn dentition_only_meaningful_on_open_mouth() {
        let closed = dentition_hint(&mean_template(false), 256, 6);
        let open = dentition_hint(&mean_template(true), 256, 6);
        // the open aperture spans more vertical pixels → more drawn ink.
        assert!(nonblack(&open) > nonblack(&closed), "open mouth draws a taller tooth arch");
    }

    #[test]
    fn maps_are_byte_stable() {
        let lm = mean_template(false);
        let m = mesh_map(&lm, 128, MeshStyle::MediaPipe);
        let mut acc: u64 = 1469598103934665603;
        for p in m.pixels() {
            for &c in &p.0 {
                acc = (acc ^ c as u64).wrapping_mul(1099511628211);
            }
        }
        assert_eq!(acc, 10743564549321961617, "mesh raster changed — update the golden intentionally");
    }
}
