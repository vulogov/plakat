//! Figure geometry (RFC §10.4). A distinct engine, same discipline as the face: pure, deterministic,
//! byte-stable, no weights. `figure` scalars resolve to a parametric **silhouette** and a **body-pose
//! skeleton** (18-keypoint OpenPose convention, reusing the crate's `LIMB_SEQ`/`LIMB_COLORS` so the
//! output feeds the existing pose-conditioning path), plus a **body anchor-site** vocabulary for
//! body-sited details (forearm tattoos, wrist jewelry, throat pendants, nape piercings; §8.5).
//!
//! **Honest scope (§10.4/§11.7):** body conditioning is materially weaker than face conditioning. The
//! calibration pass (P4) will grade most `figure` attributes below `strong`, and any UI must not
//! present figure sliders with the same authority as facial ones. This module produces a usable soft
//! mask + skeleton; it does not claim precise anthropometry.

use crate::pipelines::openpose_post::{LIMB_COLORS, LIMB_SEQ, NUM_KEYPOINTS};
use image::{GrayImage, Luma, Rgb, RgbImage};

/// Somatotype (lexicon `figure.build`). Drives base shoulder/waist width + limb mass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Build {
    Ectomorph,
    Mesomorph,
    Endomorph,
}

impl Build {
    pub fn parse(s: &str) -> Option<Build> {
        Some(match s {
            "ectomorph" | "slim" | "lean" => Build::Ectomorph,
            "mesomorph" | "athletic" | "average" => Build::Mesomorph,
            "endomorph" | "heavy" | "stocky" => Build::Endomorph,
            _ => return None,
        })
    }
    /// (shoulder-width, waist/hip-width, limb-mass) multipliers.
    fn factors(self) -> (f32, f32, f32) {
        match self {
            Build::Ectomorph => (0.88, 0.82, 0.80),
            Build::Mesomorph => (1.05, 0.92, 1.05),
            Build::Endomorph => (1.02, 1.20, 1.30),
        }
    }
}

/// Resolved figure parameters (mixed units — height in cm, the rest `[0,1]` scalars centred at 0.5).
#[derive(Debug, Clone, Copy)]
pub struct FigureParams {
    pub height_cm: f32,
    pub build: Build,
    /// Shoulder↔waist taper: high = broad shoulders / narrow waist (V), low = the reverse.
    pub shoulder_waist: f32,
    /// Limb length relative to torso.
    pub limb_length: f32,
    /// Muscularity / mass → limb + torso silhouette thickness.
    pub musculature: f32,
}

impl Default for FigureParams {
    fn default() -> Self {
        FigureParams { height_cm: 170.0, build: Build::Mesomorph, shoulder_waist: 0.5, limb_length: 0.5, musculature: 0.5 }
    }
}

/// A resolved figure: 18 keypoints in the normalised figure box `[0,1]²` (head top, feet bottom) plus
/// the params it came from (silhouette widths + anchors derive from both).
#[derive(Debug, Clone)]
pub struct Figure {
    pub joints: [[f32; 2]; NUM_KEYPOINTS],
    pub params: FigureParams,
}

// Canonical standing front-facing skeleton (normalised, symmetric about x=0.5). Indices are the
// OpenPose-18 ordering (see openpose_post): 0 nose,1 neck,2/5 R/L shoulder,3/6 elbow,4/7 wrist,
// 8/11 hip,9/12 knee,10/13 ankle,14/15 R/L eye,16/17 R/L ear.
const CANON: [[f32; 2]; NUM_KEYPOINTS] = [
    [0.500, 0.090], // 0 nose
    [0.500, 0.160], // 1 neck
    [0.410, 0.180], // 2 R shoulder
    [0.375, 0.300], // 3 R elbow
    [0.360, 0.420], // 4 R wrist
    [0.590, 0.180], // 5 L shoulder
    [0.625, 0.300], // 6 L elbow
    [0.640, 0.420], // 7 L wrist
    [0.450, 0.470], // 8 R hip
    [0.450, 0.700], // 9 R knee
    [0.450, 0.960], // 10 R ankle
    [0.550, 0.470], // 11 L hip
    [0.550, 0.700], // 12 L knee
    [0.550, 0.960], // 13 L ankle
    [0.472, 0.075], // 14 R eye
    [0.528, 0.075], // 15 L eye
    [0.452, 0.085], // 16 R ear
    [0.548, 0.085], // 17 L ear
];

fn dev(v: f32) -> f32 {
    (v.clamp(0.0, 1.0) - 0.5) * 2.0 // [0,1] → [-1,1]
}

/// Resolve figure params into a scaled skeleton. `seed` is reserved for future stance jitter; the
/// skeleton itself is currently deterministic in the params alone (front stance).
pub fn resolve_figure(p: &FigureParams, _seed: u64) -> Figure {
    let (sh_f, hip_f, _mass) = p.build.factors();
    let sw = dev(p.shoulder_waist);
    let ll = dev(p.limb_length);
    // taller → slightly smaller head-to-body ratio: nudge the neck/shoulders up a touch.
    let tall = ((p.height_cm - 170.0) / 40.0).clamp(-1.0, 1.0);

    let mut j = CANON;
    let widen = |x: f32, k: f32| 0.5 + (x - 0.5) * k;
    // shoulders
    let shoulder_k = sh_f * (1.0 + sw * 0.18);
    for &i in &[2usize, 5] {
        j[i][0] = widen(j[i][0], shoulder_k);
        j[i][1] -= tall * 0.01;
    }
    // hips / waist
    let hip_k = hip_f * (1.0 - sw * 0.15);
    for &i in &[8usize, 11] {
        j[i][0] = widen(j[i][0], hip_k);
    }
    // arms: elbow/wrist follow the shoulder x, and extend in y with limb_length.
    for (elb, wri, sho) in [(3usize, 4usize, 2usize), (6, 7, 5)] {
        j[elb][0] = j[sho][0] + (CANON[elb][0] - CANON[sho][0]);
        j[wri][0] = j[sho][0] + (CANON[wri][0] - CANON[sho][0]);
        let ext = 1.0 + ll * 0.12;
        j[elb][1] = j[1][1] + (CANON[elb][1] - CANON[1][1]) * ext;
        j[wri][1] = j[1][1] + (CANON[wri][1] - CANON[1][1]) * ext;
    }
    // legs: knee/ankle follow the hip x and extend with limb_length.
    for (hip, kne, ank) in [(8usize, 9usize, 10usize), (11, 12, 13)] {
        j[kne][0] = j[hip][0];
        j[ank][0] = j[hip][0];
        let ext = 1.0 + ll * 0.10 + tall * 0.03;
        j[kne][1] = j[hip][1] + (CANON[kne][1] - CANON[hip][1]) * ext;
        j[ank][1] = j[hip][1] + (CANON[ank][1] - CANON[hip][1]) * ext;
    }
    // clamp everything into the box.
    for pt in j.iter_mut() {
        pt[0] = pt[0].clamp(0.0, 1.0);
        pt[1] = pt[1].clamp(0.0, 1.0);
    }
    Figure { joints: j, params: *p }
}

/// A body anchor site → its point on the resolved skeleton (§10.4/§8.5). `x` is image-right.
/// Hand sites are flagged experimental per §8.5 (past the wrist, no wrist-relative orientation).
pub fn figure_anchor(fig: &Figure, name: &str) -> Option<[f32; 2]> {
    let j = &fig.joints;
    let mid = |a: usize, b: usize| [(j[a][0] + j[b][0]) / 2.0, (j[a][1] + j[b][1]) / 2.0];
    Some(match name {
        "right-shoulder" => j[2],
        "left-shoulder" => j[5],
        "right-upper-arm" => mid(2, 3),
        "left-upper-arm" => mid(5, 6),
        "right-forearm" => mid(3, 4),
        "left-forearm" => mid(6, 7),
        "right-wrist" => j[4],
        "left-wrist" => j[7],
        "throat" | "neck" => [j[1][0], (j[0][1] + j[1][1]) / 2.0],
        "nape" => j[1], // back of neck ≈ neck joint in a front render
        "sternum" | "chest" => [j[1][0], j[1][1] + (mid(8, 11)[1] - j[1][1]) * 0.35],
        "navel" => [mid(8, 11)[0], j[1][1] + (mid(8, 11)[1] - j[1][1]) * 0.85],
        "right-thigh" => mid(8, 9),
        "left-thigh" => mid(11, 12),
        // experimental (§8.5): hand ≈ a short extension past the wrist along the forearm.
        "right-hand" => extend(j[3], j[4], 0.4),
        "left-hand" => extend(j[6], j[7], 0.4),
        _ => return None,
    })
}

fn extend(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [b[0] + (b[0] - a[0]) * t, b[1] + (b[1] - a[1]) * t]
}

/// The advertised body anchor vocabulary (`figure_anchor` keys). Hand sites are experimental.
pub const BODY_ANCHOR_VOCAB: &[&str] = &[
    "right-shoulder", "left-shoulder", "right-upper-arm", "left-upper-arm",
    "right-forearm", "left-forearm", "right-wrist", "left-wrist",
    "throat", "nape", "sternum", "chest", "navel", "right-thigh", "left-thigh",
    "right-hand", "left-hand",
];

fn px(pt: [f32; 2], w: u32, h: u32) -> [f32; 2] {
    [pt[0] * (w - 1) as f32, pt[1] * (h - 1) as f32]
}

/// OpenPose-18 skeleton on black, in the crate's `LIMB_SEQ`/`LIMB_COLORS` convention (§10.4 — feeds
/// the existing pose-conditioning path).
pub fn figure_skeleton_map(fig: &Figure, width: u32, height: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));
    let thick = (width.min(height) / 200).max(1) as i32 * 2 + 1;
    for (limb, &[a, b]) in LIMB_SEQ.iter().enumerate() {
        let pa = px(fig.joints[a], width, height);
        let pb = px(fig.joints[b], width, height);
        thick_line(&mut img, pa, pb, Rgb(LIMB_COLORS[limb]), thick);
    }
    for (i, &pt) in fig.joints.iter().enumerate() {
        let p = px(pt, width, height);
        let c = LIMB_COLORS[i.min(LIMB_COLORS.len() - 1)];
        disc(&mut img, p, thick + 1, Rgb(c));
    }
    img
}

/// A soft body silhouette mask (255 inside) — union of part capsules whose thickness scales with
/// build + musculature (§10.4). Usable as a soft mask / TUI preview.
pub fn silhouette_mask(fig: &Figure, width: u32, height: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(width, height, Luma([0]));
    let (_sh, _hp, mass) = fig.params.build.factors();
    let muscle = 0.7 + dev(fig.params.musculature) * 0.35 + (mass - 1.0) * 0.5;
    let unit = width.min(height) as f32;
    let j = |i: usize| px(fig.joints[i], width, height);

    // head (disc at the nose; a degenerate capsule = a filled circle in the gray mask)
    let head_r = unit * 0.055 * muscle.max(0.8);
    capsule(&mut img, j(0), j(0), head_r * 1.4, 255);
    // torso: neck→shoulders→hips as capsules + a filled quad
    let torso_r = unit * 0.10 * muscle;
    capsule(&mut img, j(2), j(8), torso_r, 255);
    capsule(&mut img, j(5), j(11), torso_r, 255);
    capsule(&mut img, j(2), j(5), torso_r, 255);
    capsule(&mut img, j(8), j(11), torso_r * 0.9, 255);
    fill_quad(&mut img, [j(2), j(5), j(11), j(8)], 255);
    // limbs
    let arm_r = unit * 0.045 * muscle;
    let leg_r = unit * 0.06 * muscle;
    for [a, b, r] in [
        [2.0, 3.0, arm_r], [3.0, 4.0, arm_r * 0.85],
        [5.0, 6.0, arm_r], [6.0, 7.0, arm_r * 0.85],
        [8.0, 9.0, leg_r], [9.0, 10.0, leg_r * 0.8],
        [11.0, 12.0, leg_r], [12.0, 13.0, leg_r * 0.8],
    ] {
        capsule(&mut img, j(a as usize), j(b as usize), r, 255);
    }
    img
}

// --- self-contained byte-stable raster primitives (grayscale + rgb) ---

fn disc(img: &mut RgbImage, c: [f32; 2], r: i32, colour: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let (cx, cy) = (c[0].round() as i32, c[1].round() as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let (x, y) = (cx + dx, cy + dy);
                if x >= 0 && y >= 0 && x < w && y < h {
                    img.put_pixel(x as u32, y as u32, colour);
                }
            }
        }
    }
}

fn thick_line(img: &mut RgbImage, a: [f32; 2], b: [f32; 2], colour: Rgb<u8>, thick: i32) {
    let steps = ((b[0] - a[0]).hypot(b[1] - a[1]).ceil() as i32).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        disc(img, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], (thick - 1).max(0) / 2, colour);
    }
}

/// Fill a capsule (thick segment) into a grayscale mask by distance-to-segment.
fn capsule(img: &mut GrayImage, a: [f32; 2], b: [f32; 2], r: f32, val: u8) {
    let (w, h) = (img.width(), img.height());
    let (x0, y0) = ((a[0].min(b[0]) - r).max(0.0) as u32, (a[1].min(b[1]) - r).max(0.0) as u32);
    let (x1, y1) = (((a[0].max(b[0]) + r).min((w - 1) as f32)) as u32, ((a[1].max(b[1]) + r).min((h - 1) as f32)) as u32);
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = (dx * dx + dy * dy).max(1e-6);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (px_, py_) = (x as f32, y as f32);
            let t = (((px_ - a[0]) * dx + (py_ - a[1]) * dy) / len2).clamp(0.0, 1.0);
            let (cx, cy) = (a[0] + dx * t, a[1] + dy * t);
            if (px_ - cx).hypot(py_ - cy) <= r {
                img.put_pixel(x, y, Luma([val]));
            }
        }
    }
}

/// Scanline-fill a convex quad into a grayscale mask.
fn fill_quad(img: &mut GrayImage, q: [[f32; 2]; 4], val: u8) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let ymin = q.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
    let ymax = q.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).ceil().min((h - 1) as f32) as i32;
    for y in ymin..=ymax {
        let mut xs: Vec<f32> = Vec::new();
        for k in 0..4 {
            let (p, np) = (q[k], q[(k + 1) % 4]);
            let (y0, y1) = (p[1], np[1]);
            let yf = y as f32;
            if (y0 <= yf && yf < y1) || (y1 <= yf && yf < y0) {
                let t = (y as f32 - y0) / (y1 - y0);
                xs.push(p[0] + (np[0] - p[0]) * t);
            }
        }
        if xs.len() >= 2 {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (mut x0, x1) = (xs[0].round() as i32, xs[xs.len() - 1].round() as i32);
            x0 = x0.max(0);
            for x in x0..=x1.min(w - 1) {
                img.put_pixel(x as u32, y as u32, Luma([val]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_parsing_and_defaults() {
        assert_eq!(Build::parse("athletic"), Some(Build::Mesomorph));
        assert_eq!(Build::parse("nope"), None);
        assert_eq!(FigureParams::default().height_cm, 170.0);
    }

    #[test]
    fn skeleton_is_bounded_and_symmetric_at_neutral() {
        let f = resolve_figure(&FigureParams::default(), 0);
        for pt in f.joints.iter() {
            assert!((0.0..=1.0).contains(&pt[0]) && (0.0..=1.0).contains(&pt[1]));
        }
        // neutral figure is left-right symmetric: R/L shoulder mirror about 0.5.
        assert!((f.joints[2][0] - (1.0 - f.joints[5][0])).abs() < 1e-5);
        assert!((f.joints[10][0] - (1.0 - f.joints[13][0])).abs() < 1e-5, "ankles mirror");
    }

    #[test]
    fn broad_shoulders_widen_the_shoulder_span() {
        let base = resolve_figure(&FigureParams::default(), 0);
        let mut p = FigureParams::default();
        p.shoulder_waist = 1.0;
        p.build = Build::Mesomorph;
        let broad = resolve_figure(&p, 0);
        let span = |f: &Figure| f.joints[5][0] - f.joints[2][0];
        assert!(span(&broad) > span(&base), "V-taper widens shoulders");
    }

    #[test]
    fn longer_limbs_drop_the_ankles() {
        let short = resolve_figure(&FigureParams { limb_length: 0.0, ..Default::default() }, 0);
        let long = resolve_figure(&FigureParams { limb_length: 1.0, ..Default::default() }, 0);
        assert!(long.joints[10][1] > short.joints[10][1], "longer legs → lower ankle");
    }

    #[test]
    fn silhouette_and_skeleton_render_nonempty() {
        let f = resolve_figure(&FigureParams::default(), 0);
        let sil = silhouette_mask(&f, 256, 384);
        let ink = sil.pixels().filter(|p| p.0[0] > 0).count();
        assert!(ink > 1000, "silhouette should fill a body");
        let skel = figure_skeleton_map(&f, 256, 384);
        assert!(skel.pixels().filter(|p| p.0 != [0, 0, 0]).count() > 200);
    }

    #[test]
    fn body_anchors_all_resolve_and_are_ordered() {
        let f = resolve_figure(&FigureParams::default(), 0);
        for &name in BODY_ANCHOR_VOCAB {
            assert!(figure_anchor(&f, name).is_some(), "{name} missing");
        }
        assert!(figure_anchor(&f, "not-a-site").is_none());
        // throat above sternum above navel (increasing y).
        let y = |n: &str| figure_anchor(&f, n).unwrap()[1];
        assert!(y("throat") < y("sternum") && y("sternum") < y("navel"));
        // right forearm left of left forearm.
        assert!(figure_anchor(&f, "right-forearm").unwrap()[0] < figure_anchor(&f, "left-forearm").unwrap()[0]);
    }
}
