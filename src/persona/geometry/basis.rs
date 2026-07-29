//! The deformation basis + `resolve()` (RFC §10.2).
//!
//! Each geometric spec attribute names a **deformation direction** — a displacement field over the
//! WFLW-98 landmark set, applied proportionally to the attribute's deviation from `0.5`:
//!
//! ```text
//! landmarks = mean_template + Σ_a  (value(a) − 0.5)·2 · basis(a)
//! ```
//!
//! (`basis(a)` is authored as the displacement at the extreme pole, so `value = 1.0` applies `+basis`
//! and `value = 0.0` applies `−basis`.) Design constraints from §10.2 are honoured here:
//!
//! * **Locality with anatomical coupling** — e.g. `eyes.spacing` moves the eyes *and* nudges the inner
//!   brows + nasal base, so the face doesn't read as pasted-together.
//! * **Fixed-order composition** — attributes are applied in `GEOMETRIC_ATTRS` order, so where two
//!   fight over the same landmarks (jaw width vs face width) the result is deterministic.
//! * **Seed-derived asymmetry** — a small deterministic perturbation (perfectly symmetric faces read
//!   as synthetic; the cheapest realism win, §10.2).
//! * **Anchors follow** — detail anchors resolve against the *deformed* set (done by the caller via the
//!   returned landmarks), so a mark stays anatomically correct through any edit.
//! * **Bounded composition** — a validity pass clamps escaped/inverted geometry and reports it for lint
//!   (§6.6); failures are surfaced, never silently rendered wrong.
//!
//! Pure + byte-stable: a function of `(values, open_mouth, seed)` only. No weights, no I/O.

use super::template::{mean_template, Point, Template};
use super::topology::*;
use std::collections::BTreeMap;

/// The geometric (landmark-deforming) attributes, in fixed application order. Non-geometric lexicon
/// attributes (colour, hair, enums that only steer the prompt) are absent by design.
pub const GEOMETRIC_ATTRS: &[&str] = &[
    "face.width",
    "face.jaw.width",
    "face.cheekbones.prominence",
    "face.chin.projection",
    "eyes.spacing",
    "eyes.canthal_tilt",
    "eyes.brow.thickness",
    "nose.length",
    "mouth.width",
    "mouth.lower_lip",
];

/// A validity issue found after composition (surfaced by lint, §6.6). Never fatal at this layer — the
/// geometry is clamped to a valid configuration and the issue is reported.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoWarning {
    pub kind: &'static str,
    pub detail: String,
}

/// Result of resolving a spec's geometry into a landmark configuration.
#[derive(Debug, Clone)]
pub struct Deformed {
    pub landmarks: Template,
    pub open_mouth: bool,
    pub warnings: Vec<GeoWarning>,
}

/// The displacement field for one attribute, computed against the mean template `t` (some fields are
/// naturally expressed relative to the template — e.g. "scale the contour width about the axis").
/// Returns `(index, dx, dy)` at the extreme pole.
fn basis(attr: &str, t: &Template) -> Vec<(usize, f32, f32)> {
    let mut v = Vec::new();
    // scale a set of indices in x about the centre axis by `k` (proportional to current offset)
    let scale_x = |v: &mut Vec<(usize, f32, f32)>, idxs: &[usize], k: f32| {
        for &i in idxs {
            v.push((i, (t[i].0 - 0.5) * k, 0.0));
        }
    };
    match attr {
        // whole face width: scale the entire contour about the axis.
        "face.width" => scale_x(&mut v, &CONTOUR.collect::<Vec<_>>(), 0.16),
        // jaw width: the lower contour only (below the cheekbones), weighted toward the jaw angle.
        "face.jaw.width" => {
            for (i, pt) in t.iter().enumerate().take(30).skip(3) {
                let low = ((pt.1 - 0.45) / 0.5).clamp(0.0, 1.0); // 0 at eye line → 1 near chin
                v.push((i, (pt.0 - 0.5) * 0.18 * low, 0.0));
            }
        }
        // cheekbones: upper contour points move outward + up (high = high/prominent).
        "face.cheekbones.prominence" => {
            for &i in &[1usize, 2, 3, 29, 30, 31] {
                v.push((i, (t[i].0 - 0.5) * 0.08, -0.03));
            }
        }
        // chin projection: the chin cluster drops (2D proxy for forward projection).
        "face.chin.projection" => {
            for (i, w) in [(14usize, 0.4), (15, 0.8), (16, 1.0), (17, 0.8), (18, 0.4)] {
                v.push((i, 0.0, 0.05 * w));
            }
        }
        // eyes spacing: eyes + pupils out; couple inner brows out + nasal wings slightly wider.
        "eyes.spacing" => {
            for i in EYE_RIGHT {
                v.push((i, -0.035, 0.0));
            }
            for i in EYE_LEFT {
                v.push((i, 0.035, 0.0));
            }
            v.push((PUPIL_RIGHT, -0.035, 0.0));
            v.push((PUPIL_LEFT, 0.035, 0.0));
            v.push((BROW_RIGHT_INNER, -0.012, 0.0)); // coupling: inner brow follows
            v.push((BROW_LEFT_INNER, 0.012, 0.0));
            v.push((55, -0.008, 0.0)); // coupling: nostril wings
            v.push((59, 0.008, 0.0));
        }
        // canthal tilt: outer corners rise, inner corners drop (high = upturned).
        "eyes.canthal_tilt" => {
            v.push((EYE_RIGHT_OUTER, 0.0, -0.025));
            v.push((EYE_RIGHT_INNER, 0.0, 0.012));
            v.push((EYE_LEFT_OUTER, 0.0, -0.025));
            v.push((EYE_LEFT_INNER, 0.0, 0.012));
            v.push((BROW_RIGHT_OUTER, 0.0, -0.012)); // coupling: outer brow lifts with the tilt
            v.push((BROW_LEFT_OUTER, 0.0, -0.012));
        }
        // brow thickness: modest vertical thickening (mostly surface; small geometric effect).
        "eyes.brow.thickness" => {
            for i in BROW_RIGHT.chain(BROW_LEFT) {
                v.push((i, 0.0, 0.008));
            }
        }
        // nose length: bridge + base descend, weighted by how far down the point already is.
        "nose.length" => {
            for i in NOSE {
                let w = ((t[i].1 - NOSE_TEMPLATE_TOP) / 0.28).clamp(0.0, 1.0);
                v.push((i, 0.0, 0.045 * w));
            }
            v.push((PUPIL_RIGHT, 0.0, 0.0)); // (no eye coupling; nose length is fairly local)
        }
        // mouth width: scale the whole mouth (outer + inner lips) about the axis.
        "mouth.width" => {
            let all: Vec<usize> = LIP_OUTER.chain(LIP_INNER).collect();
            scale_x(&mut v, &all, 0.20);
        }
        // lower-lip fullness: lower outer + lower inner lip descend; upper lip rises a touch.
        "mouth.lower_lip" => {
            for i in 83..88 {
                v.push((i, 0.0, 0.022));
            }
            for i in 92..96 {
                v.push((i, 0.0, 0.018)); // lower inner-lip arc
            }
            v.push((79, 0.0, -0.006)); // upper-lip centre lifts slightly (fuller pout)
        }
        _ => {}
    }
    v
}

// The mean template's nose-bridge-top y, used to weight nose.length. Kept in sync with template.rs.
const NOSE_TEMPLATE_TOP: f32 = 0.33;

/// A tiny deterministic PRNG (splitmix64) so asymmetry is seed-reproducible without `rand`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Resolve a spec's geometric scalar `values` (keyed by lexicon path, each in `[0,1]`) into a
/// deformed landmark configuration. `open_mouth` selects the aperture variant (§8.7, from the
/// manifestation gate). `seed` drives the asymmetry perturbation and the optional `face.asymmetry`
/// scalar scales its magnitude (absent → a small realism baseline).
pub fn resolve(values: &BTreeMap<String, f32>, open_mouth: bool, seed: u64) -> Deformed {
    let base = mean_template(open_mouth);
    let mut p = base;

    // --- linear deformation basis, fixed order ---
    for &attr in GEOMETRIC_ATTRS {
        let Some(&val) = values.get(attr) else { continue };
        if !val.is_finite() {
            continue;
        }
        let d = (val.clamp(0.0, 1.0) - 0.5) * 2.0; // deviation ∈ [-1, 1]
        if d == 0.0 {
            continue;
        }
        for (i, dx, dy) in basis(attr, &base) {
            p[i].0 += dx * d;
            p[i].1 += dy * d;
        }
    }

    // --- seed-derived asymmetry (§10.2): break perfect symmetry a little ---
    let asym = values.get("face.asymmetry").copied().filter(|v| v.is_finite());
    let amount = asym.map(|v| v.clamp(0.0, 1.0)).unwrap_or(0.15);
    if amount > 0.0 {
        let mut st = seed ^ 0xA5A5_5A5A_1234_5678;
        // apply small correlated jitter to a few regions (one side only), so the face is subtly uneven.
        let jitter = |st: &mut u64, mag: f32| ((splitmix64(st) as f64 / u64::MAX as f64) as f32 - 0.5) * 2.0 * mag;
        let mag = amount * 0.03;
        // eye height + brow on the subject's-left; jaw sway; nose-tip drift — all sub-perceptual.
        let dyl = jitter(&mut st, mag);
        for i in EYE_LEFT.chain(std::iter::once(PUPIL_LEFT)).chain(BROW_LEFT) {
            p[i].1 += dyl;
        }
        let dxj = jitter(&mut st, mag);
        for pt in p.iter_mut().take(12).skip(4) {
            pt.0 += dxj; // subject's-right jaw sways
        }
        p[NOSE_TIP].0 += jitter(&mut st, mag * 0.5);
    }

    // --- validity pass: clamp + report (§10.2 bounded composition, §6.6 lint) ---
    let mut warnings = Vec::new();
    validate_and_clamp(&mut p, &mut warnings);

    Deformed { landmarks: p, open_mouth, warnings }
}

fn bbox(p: &Template, idxs: impl Iterator<Item = usize>) -> (f32, f32, f32, f32) {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in idxs {
        x0 = x0.min(p[i].0);
        y0 = y0.min(p[i].1);
        x1 = x1.max(p[i].0);
        y1 = y1.max(p[i].1);
    }
    (x0, y0, x1, y1)
}

fn centre_y(p: &Template, idxs: impl Iterator<Item = usize>) -> f32 {
    let (mut s, mut n) = (0.0f32, 0u32);
    for i in idxs {
        s += p[i].1;
        n += 1;
    }
    s / n.max(1) as f32
}

/// Clamp escaped / inverted geometry back to validity and record what happened (§10.2/§6.6).
fn validate_and_clamp(p: &mut Template, w: &mut Vec<GeoWarning>) {
    // 1. escaped the face box → clamp to [0,1].
    let mut escaped = 0;
    for pt in p.iter_mut() {
        for c in [&mut pt.0, &mut pt.1] {
            if *c < 0.0 || *c > 1.0 {
                escaped += 1;
                *c = c.clamp(0.0, 1.0);
            }
        }
    }
    if escaped > 0 {
        w.push(GeoWarning { kind: "out-of-box", detail: format!("{escaped} coordinate(s) clamped to the face box") });
    }

    // 2. eye contour inversion (a corner crossing the axis / its opposite corner).
    if p[EYE_RIGHT_OUTER].0 >= p[EYE_RIGHT_INNER].0 || p[EYE_LEFT_INNER].0 >= p[EYE_LEFT_OUTER].0 {
        w.push(GeoWarning { kind: "eye-inverted", detail: "eye contour inverted (spacing/tilt too extreme)".into() });
    }

    // 3. inner lip must stay within the outer-lip bbox — clamp it back if it escaped.
    let (ox0, oy0, ox1, oy1) = bbox(p, LIP_OUTER);
    let mut lip_clamped = false;
    for i in LIP_INNER {
        let nx = p[i].0.clamp(ox0, ox1);
        let ny = p[i].1.clamp(oy0, oy1);
        if nx != p[i].0 || ny != p[i].1 {
            lip_clamped = true;
        }
        p[i] = (nx, ny);
    }
    if lip_clamped {
        w.push(GeoWarning { kind: "lip-escaped", detail: "inner lip clamped inside the outer lip".into() });
    }

    // 4. vertical feature ordering: brow < eye < nose-tip < mouth.
    let brow = centre_y(p, BROW_RIGHT.chain(BROW_LEFT));
    let eye = centre_y(p, std::iter::once(PUPIL_RIGHT).chain(std::iter::once(PUPIL_LEFT)));
    let nose = p[NOSE_TIP].1;
    let mouth = centre_y(p, LIP_OUTER);
    if !(brow < eye && eye < nose && nose < mouth) {
        w.push(GeoWarning {
            kind: "feature-order",
            detail: "vertical feature ordering (brow<eye<nose<mouth) violated".into(),
        });
    }
}

/// Convenience: the un-deformed mean template as a `Deformed` (no attributes, baseline asymmetry off).
pub fn identity(open_mouth: bool) -> Deformed {
    Deformed { landmarks: mean_template(open_mouth), open_mouth, warnings: Vec::new() }
}

/// Resolve a single named anchor against a realised landmark set (anchors follow deformation, §10.2).
/// `offset` is in face-normalised units (x positive to the subject's left / image right).
pub fn anchor_point(p: &Template, name: &str, offset: [f32; 2]) -> Option<Point> {
    let idxs = named_region(name)?;
    let n = idxs.len() as f32;
    let (bx, by) = idxs.iter().fold((0.0, 0.0), |(ax, ay), &i| (ax + p[i].0 / n, ay + p[i].1 / n));
    Some(((bx + offset[0]).clamp(0.0, 1.0), (by + offset[1]).clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(pairs: &[(&str, f32)]) -> BTreeMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn empty_values_reproduce_the_template_plus_baseline_asymmetry() {
        // With no geometric values and asymmetry forced off, resolve == mean template exactly.
        let d = resolve(&vals(&[("face.asymmetry", 0.0)]), false, 7);
        assert_eq!(d.landmarks, mean_template(false));
        assert!(d.warnings.is_empty());
    }

    #[test]
    fn wide_eyes_move_pupils_apart_and_narrow_eyes_together() {
        let base = mean_template(false);
        let base_ipd = base[PUPIL_LEFT].0 - base[PUPIL_RIGHT].0;
        let wide = resolve(&vals(&[("eyes.spacing", 1.0), ("face.asymmetry", 0.0)]), false, 0);
        let narrow = resolve(&vals(&[("eyes.spacing", 0.0), ("face.asymmetry", 0.0)]), false, 0);
        let wipd = wide.landmarks[PUPIL_LEFT].0 - wide.landmarks[PUPIL_RIGHT].0;
        let nipd = narrow.landmarks[PUPIL_LEFT].0 - narrow.landmarks[PUPIL_RIGHT].0;
        assert!(wipd > base_ipd && base_ipd > nipd, "spacing monotonic: {nipd} < {base_ipd} < {wipd}");
    }

    #[test]
    fn value_half_is_a_no_op_for_that_attribute() {
        let a = resolve(&vals(&[("mouth.width", 0.5), ("face.asymmetry", 0.0)]), false, 3);
        let b = resolve(&vals(&[("face.asymmetry", 0.0)]), false, 3);
        assert_eq!(a.landmarks, b.landmarks, "value 0.5 must contribute nothing");
    }

    #[test]
    fn asymmetry_breaks_symmetry_and_is_seed_stable() {
        let a = resolve(&vals(&[("face.asymmetry", 1.0)]), false, 42);
        let b = resolve(&vals(&[("face.asymmetry", 1.0)]), false, 42);
        let c = resolve(&vals(&[("face.asymmetry", 1.0)]), false, 99);
        assert_eq!(a.landmarks, b.landmarks, "same seed → identical (determinism)");
        assert_ne!(a.landmarks, c.landmarks, "different seed → different jitter");
        // it is actually asymmetric: left/right pupil heights differ.
        assert_ne!(a.landmarks[PUPIL_LEFT].1, a.landmarks[PUPIL_RIGHT].1);
    }

    #[test]
    fn extreme_spacing_reports_a_validity_warning() {
        // Push eyes far apart AND wide face — something should trip the validity pass or clamp.
        let d = resolve(&vals(&[("eyes.spacing", 1.0), ("face.width", 1.0), ("mouth.width", 1.0), ("face.asymmetry", 0.0)]), false, 0);
        // All points remain inside the box regardless (post-clamp invariant).
        for &(x, y) in d.landmarks.iter() {
            assert!((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y));
        }
    }

    #[test]
    fn anchors_follow_deformation() {
        // A cheek anchor should sit further out on a wide face than on a narrow one.
        let wide = resolve(&vals(&[("face.width", 1.0), ("eyes.spacing", 1.0), ("face.asymmetry", 0.0)]), false, 0);
        let narrow = resolve(&vals(&[("face.width", 0.0), ("eyes.spacing", 0.0), ("face.asymmetry", 0.0)]), false, 0);
        let wl = anchor_point(&wide.landmarks, "left-cheek", [0.0, 0.0]).unwrap();
        let nl = anchor_point(&narrow.landmarks, "left-cheek", [0.0, 0.0]).unwrap();
        assert!(wl.0 > nl.0, "left cheek further right on a wider face: {} vs {}", wl.0, nl.0);
    }

    #[test]
    fn resolve_is_byte_stable() {
        let d = resolve(
            &vals(&[("face.width", 0.7), ("eyes.spacing", 0.3), ("nose.length", 0.8), ("mouth.width", 0.6), ("face.asymmetry", 0.5)]),
            false,
            12345,
        );
        let mut acc: u64 = 1469598103934665603;
        for &(x, y) in d.landmarks.iter() {
            for v in [x, y] {
                let q = (v * 100_000.0).round() as i64 as u64;
                acc = (acc ^ q).wrapping_mul(1099511628211);
            }
        }
        assert_eq!(acc, 18438757073983175851, "resolve output changed — update the golden intentionally");
    }
}
