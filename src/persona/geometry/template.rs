//! The **mean template** (RFC §10.1): a canonical, front-facing, neutral-expression WFLW-98 landmark
//! configuration in the normalised face-box, plus an **open-mouth variant** (selected by the
//! manifestation gate when dentition geometry is needed, §8.7).
//!
//! Hand-authored, not fitted to a dataset — for the licensing + reproducibility reasons in §10.5. The
//! authoring lives in readable parametric code (anatomical bands + arcs), exactly as the `map` track's
//! bitmap font is a committed, inspectable, license-free asset. The output is byte-stable on-box
//! (pinned by `mean_template_is_byte_stable`), so it functions as a committed asset while remaining
//! diffable as code.
//!
//! Coordinates: face-box `[0,1]²`, `x` right, `y` down, symmetric about `x = 0.5` (see `topology`).

use super::topology::*;
use std::f32::consts::PI;

/// A landmark point in the normalised face-box.
pub type Point = (f32, f32);

/// The 98-point mean template.
pub type Template = [Point; NUM_LANDMARKS];

// --- Anatomical bands (image-y, top = 0). Hand-authored from published proportion ranges. ---
const TEMPLE_Y: f32 = 0.22; // contour endpoints at the temples
const CHIN_Y: f32 = 0.99;
const BROW_Y: f32 = 0.30;
const EYE_Y: f32 = 0.40; // pupil line
const NOSE_TOP_Y: f32 = 0.33; // bridge root, just below the glabella
const NOSE_TIP_Y: f32 = 0.60;
const MOUTH_Y: f32 = 0.755;

// Half-widths (from centre).
const CHEEK_HALF: f32 = 0.47; // widest face point (cheekbone / temple)
const CHIN_HALF: f32 = 0.19;
const BROW_OUTER_X: f32 = 0.36; // ±from centre → x 0.14 / 0.86
const BROW_INNER_X: f32 = 0.10; // ±from centre → x 0.40 / 0.60
const BROW_ARCH: f32 = 0.035;
const NOSE_HALF: f32 = 0.10; // nostril wing half-width
const NOSE_WING_RISE: f32 = 0.015;
const EYE_CX: f32 = 0.16; // eye-centre offset from face centre → 0.34 / 0.66
const EYE_HALF_W: f32 = 0.085;
const EYE_HALF_H: f32 = 0.033;
const MOUTH_HALF: f32 = 0.155;
const UPPER_LIP: f32 = 0.028; // outer top rise above the mouth line
const LOWER_LIP: f32 = 0.045; // outer bottom drop below the mouth line
const CUPID_DIP: f32 = 0.010; // cupid's-bow centre dip
const INNER_HALF_W: f32 = 0.105;
const INNER_HALF_H_CLOSED: f32 = 0.010;
const INNER_HALF_H_OPEN: f32 = 0.038; // dentition-visible aperture (opens downward, see INNER_OPEN_DROP)
const INNER_OPEN_DROP: f32 = 0.020; // aperture centre drop when the mouth opens

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Build the mean template. `open` selects the open-mouth variant (wider inner-lip aperture + a
/// slightly dropped lower outer lip) used when the mouth manifests teeth (§8.7).
pub fn mean_template(open: bool) -> Template {
    let mut p: Template = [(0.5, 0.5); NUM_LANDMARKS];

    // --- Contour 0..=32: ear-to-ear jaw arc, symmetric about 0.5. ---
    for i in CONTOUR {
        let phi = PI * (1.0 - i as f32 / 32.0); // π (right temple) → 0 (left temple), π/2 at chin
        let s = phi.sin().max(0.0); // sin(0)/sin(π) are tiny negatives in f32 → clamp before powf
        // half-width: widest at the temples (phi=0,π), narrowest at the chin (phi=π/2)
        let hw = CHIN_HALF + (CHEEK_HALF - CHIN_HALF) * (1.0 - s.powf(1.3));
        let x = 0.5 + phi.cos() * hw;
        let y = TEMPLE_Y + (CHIN_Y - TEMPLE_Y) * s.powf(0.85);
        p[i] = (x, y);
    }

    // --- Brows: single arched arc per brow (outer→inner right, inner→outer left). ---
    for k in 0..9 {
        let t = k as f32 / 8.0;
        let arch = (t * PI).sin(); // 0..1..0, peak mid-brow
        let y = BROW_Y - arch * BROW_ARCH;
        // right brow 33..=41: outer (small x) → inner (toward centre)
        p[BROW_RIGHT.start + k] = (lerp(0.5 - BROW_OUTER_X, 0.5 - BROW_INNER_X, t), y);
        // left brow 42..=50: inner (toward centre) → outer (large x); mirrors the right brow
        p[BROW_LEFT.start + k] = (lerp(0.5 + BROW_INNER_X, 0.5 + BROW_OUTER_X, t), y);
    }

    // --- Nose: straight centreline bridge 51..=54, then the base/nostril line 55..=59. ---
    for k in 0..4 {
        let t = k as f32 / 3.0;
        p[NOSE.start + k] = (0.5, lerp(NOSE_TOP_Y, NOSE_TIP_Y - 0.045, t));
    }
    for k in 0..5 {
        let t = k as f32 / 4.0;
        let x = lerp(0.5 - NOSE_HALF, 0.5 + NOSE_HALF, t);
        // centre (k=2 → index 57 = tip) is lowest; wings rise
        let rise = NOSE_WING_RISE * (1.0 - (t * PI).sin());
        p[55 + k] = (x, NOSE_TIP_Y - rise);
    }

    // --- Eyes: 8-pt almond ellipse per eye, outer corner first. ---
    for (eye, cx, inner_on_left) in [
        (EYE_RIGHT.start, 0.5 - EYE_CX, false),
        (EYE_LEFT.start, 0.5 + EYE_CX, true),
    ] {
        for k in 0..8 {
            // start at angle π (image-left point) and sweep; for the right eye that left point is the
            // outer corner (index 60), for the left eye it is the inner corner (index 68).
            let ang = PI + k as f32 * (2.0 * PI / 8.0);
            let x = cx + ang.cos() * EYE_HALF_W;
            let y = EYE_Y + ang.sin() * EYE_HALF_H;
            p[eye + k] = (x, y);
        }
        let _ = inner_on_left; // documented above; layout is symmetric by construction
    }
    p[PUPIL_RIGHT] = (0.5 - EYE_CX, EYE_Y);
    p[PUPIL_LEFT] = (0.5 + EYE_CX, EYE_Y);

    // --- Mouth outer lip 76..=87 (closed loop): top arc right→left (76..=82), bottom back (83..=87). ---
    let (rc_x, lc_x) = (0.5 - MOUTH_HALF, 0.5 + MOUTH_HALF);
    let lower_drop = if open { LOWER_LIP + 0.02 } else { LOWER_LIP };
    for k in 0..7 {
        let t = k as f32 / 6.0; // 76 (right corner) → 82 (left corner)
        let x = lerp(rc_x, lc_x, t);
        // upper-lip vermilion: rises to two peaks with a cupid's-bow centre dip
        let rise = UPPER_LIP * (t * PI).sin() - CUPID_DIP * (2.0 * t * PI).sin().max(0.0);
        p[LIP_OUTER.start + k] = (x, MOUTH_Y - rise);
    }
    for k in 0..5 {
        let t = (k as f32 + 1.0) / 6.0; // interior points left→right between the corners
        let x = lerp(lc_x, rc_x, t);
        let drop = lower_drop * (t * PI).sin();
        p[83 + k] = (x, MOUTH_Y + drop);
    }

    // --- Inner lip 88..=95 (aperture ellipse): 8 pts, right inner corner first. ---
    // An open mouth opens *downward* (the jaw drops), so the aperture centre shifts down and its top
    // stays below the outer upper-lip edge — otherwise it pokes above the outer lip and trips the
    // validity clamp (§10.2). Closed mouths keep a thin symmetric aperture at the mouth line.
    let ih = if open { INNER_HALF_H_OPEN } else { INNER_HALF_H_CLOSED };
    let iy = if open { MOUTH_Y + INNER_OPEN_DROP } else { MOUTH_Y };
    for k in 0..8 {
        let ang = PI + k as f32 * (2.0 * PI / 8.0);
        let x = 0.5 + ang.cos() * INNER_HALF_W;
        let y = iy + ang.sin() * ih;
        p[LIP_INNER.start + k] = (x, y);
    }

    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn template_is_bounded_and_finite() {
        for open in [false, true] {
            for &(x, y) in mean_template(open).iter() {
                assert!(x.is_finite() && y.is_finite());
                assert!((0.0..=1.0).contains(&x), "x {x} out of box");
                assert!((0.0..=1.0).contains(&y), "y {y} out of box");
            }
        }
    }

    #[test]
    fn template_is_left_right_symmetric() {
        // Mirror pairs must reflect about x = 0.5 with equal y.
        let p = mean_template(false);
        let mirror = |a: usize, b: usize| {
            assert!(approx(p[a].0, 1.0 - p[b].0), "x mirror {a}/{b}: {} vs {}", p[a].0, p[b].0);
            assert!(approx(p[a].1, p[b].1), "y mirror {a}/{b}");
        };
        // contour: i ↔ 32-i (chin 16 self-mirrors)
        for i in 0..16 {
            mirror(i, 32 - i);
        }
        assert!(approx(p[CHIN].0, 0.5));
        // brows (right runs outer→inner, left inner→outer, so k mirrors 8-k), eyes, pupils, corners
        for k in 0..9 {
            mirror(BROW_RIGHT.start + k, BROW_LEFT.start + (8 - k));
        }
        mirror(PUPIL_RIGHT, PUPIL_LEFT);
        mirror(EYE_RIGHT_OUTER, EYE_LEFT_OUTER);
        mirror(MOUTH_CORNER_RIGHT, MOUTH_CORNER_LEFT);
        // nose centreline is on the axis
        for i in NOSE_BRIDGE_TOP..=54 {
            assert!(approx(p[i].0, 0.5), "bridge {i} off-axis");
        }
    }

    #[test]
    fn anatomical_ordering_holds() {
        let p = mean_template(false);
        // brows above eyes above nose-tip above mouth above chin (increasing y).
        assert!(p[37].1 < p[PUPIL_RIGHT].1, "brow below eye");
        assert!(p[PUPIL_RIGHT].1 < p[NOSE_TIP].1, "eye below nose");
        assert!(p[NOSE_TIP].1 < p[MOUTH_CORNER_RIGHT].1, "nose below mouth");
        assert!(p[MOUTH_CORNER_RIGHT].1 < p[CHIN].1, "mouth below chin");
        // right-side features sit left of centre, left-side right of centre.
        assert!(p[PUPIL_RIGHT].0 < 0.5 && p[PUPIL_LEFT].0 > 0.5);
        assert!(p[EYE_RIGHT_OUTER].0 < p[EYE_RIGHT_INNER].0, "right eye outer left of inner");
    }

    #[test]
    fn open_mouth_widens_the_aperture() {
        let (closed, open) = (mean_template(false), mean_template(true));
        let aperture = |t: &Template| {
            let top = (LIP_INNER.start..LIP_INNER.end).map(|i| t[i].1).fold(f32::INFINITY, f32::min);
            let bot = (LIP_INNER.start..LIP_INNER.end).map(|i| t[i].1).fold(f32::NEG_INFINITY, f32::max);
            bot - top
        };
        assert!(aperture(&open) > aperture(&closed) * 3.0, "open mouth should open the inner lip");
        // and the closed inner lip must sit inside the outer lip vertically.
        let inner_top = (LIP_INNER.start..LIP_INNER.end).map(|i| closed[i].1).fold(f32::INFINITY, f32::min);
        assert!(inner_top > closed[79].1, "inner lip above outer upper-lip vermilion");
    }

    #[test]
    fn mean_template_is_byte_stable() {
        // Golden: a checksum over the quantised template. Any authoring change is a deliberate,
        // reviewable diff to this number (topology v1 is frozen).
        let p = mean_template(false);
        let mut acc: u64 = 1469598103934665603; // FNV-1a offset
        for &(x, y) in p.iter() {
            for v in [x, y] {
                let q = (v * 100_000.0).round() as i64 as u64;
                acc = (acc ^ q).wrapping_mul(1099511628211);
            }
        }
        assert_eq!(acc, 12757080545314864755, "mean template changed — update the golden intentionally");
    }
}
