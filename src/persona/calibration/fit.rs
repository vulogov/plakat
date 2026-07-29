//! Response-curve fitting, inverse pre-distortion, and grade derivation (RFC §13.2/§13.3) — the pure
//! math half of calibration. No I/O, no weights, byte-stable; testable with synthetic sweep data.
//!
//! A **response curve** is the empirical transfer function `requested → realised` for one geometric
//! attribute on one family, both expressed in the normalised `[0,1]` scalar space (the realised metric
//! is mapped back to a scalar via the family prior + spread, §13.1). Two things come off it:
//!
//! * **pre-distortion** (§13.2): the compiler feeds `f⁻¹(r)` so that a request of `r` *lands* at `r`
//!   rather than at `f(r)`. Where the curve is flat / non-monotone the attribute is uncontrollable and
//!   pre-distortion is a no-op (there is nothing to correct toward).
//! * **grade** (§13.3): derived from the fitted slope, monotonicity, and seed-variance — never asserted.

/// Controllability grade (§7.3/§13.3). Measured from the curve, not opinion. Mirrors the lexicon's
/// `control` weights so a per-family grade can override the lexicon default consistently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    Strong,
    Moderate,
    Weak,
    Experimental,
}

impl Grade {
    pub fn as_str(self) -> &'static str {
        match self {
            Grade::Strong => "strong",
            Grade::Moderate => "moderate",
            Grade::Weak => "weak",
            Grade::Experimental => "experimental",
        }
    }
    pub fn parse(s: &str) -> Option<Grade> {
        Some(match s {
            "strong" => Grade::Strong,
            "moderate" => Grade::Moderate,
            "weak" => Grade::Weak,
            "experimental" => Grade::Experimental,
            _ => return None,
        })
    }
    /// Salience / scorecard weight (§9.2/§12.2) — identical to `lexicon::control_weight`.
    pub fn weight(self) -> f32 {
        match self {
            Grade::Strong => 1.0,
            Grade::Moderate => 0.7,
            Grade::Weak => 0.4,
            Grade::Experimental => 0.2,
        }
    }
}

/// A fitted monotone response curve for one attribute on one family.
#[derive(Debug, Clone)]
pub struct ResponseCurve {
    /// Monotone-enforced sample points `(requested, realised)` in `[0,1]`, sorted by `requested`.
    pub samples: Vec<(f32, f32)>,
    /// Fitted slope (least-squares over the samples) — the effect size.
    pub slope: f32,
    /// Mean per-step seed-variance of the realised metric (in normalised units).
    pub variance: f32,
    /// The derived controllability grade.
    pub grade: Grade,
}

/// Fit a response curve from raw sweep samples: `(requested, realised)` pairs (realised already
/// normalised to `[0,1]`), plus the mean per-step seed variance. Enforces monotonicity (isotonic
/// cumulative-max — a family cannot make an attribute run backwards) and derives the grade.
pub fn fit(mut pairs: Vec<(f32, f32)>, variance: f32) -> ResponseCurve {
    pairs.retain(|(a, b)| a.is_finite() && b.is_finite());
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // isotonic-lite: clamp each realised to be >= the previous (monotone non-decreasing).
    let mut running = f32::NEG_INFINITY;
    let samples: Vec<(f32, f32)> = pairs
        .iter()
        .map(|&(a, b)| {
            let m = b.max(running);
            running = m;
            (a.clamp(0.0, 1.0), m.clamp(0.0, 1.0))
        })
        .collect();

    let slope = least_squares_slope(&samples);
    // "true" monotonicity of the RAW data (before isotonic clamping) — how often it went backwards.
    let mut inversions = 0usize;
    for w in pairs.windows(2) {
        if w[1].1 < w[0].1 - 1e-4 {
            inversions += 1;
        }
    }
    let monotone_frac = if pairs.len() > 1 { 1.0 - inversions as f32 / (pairs.len() - 1) as f32 } else { 1.0 };
    let grade = grade_from(slope, monotone_frac, variance);
    ResponseCurve { samples, slope, variance, grade }
}

fn least_squares_slope(s: &[(f32, f32)]) -> f32 {
    let n = s.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for &(x, y) in s {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let d = n * sxx - sx * sx;
    if d.abs() < 1e-9 {
        0.0
    } else {
        (n * sxy - sx * sy) / d
    }
}

/// Grade from the fitted shape (§13.3). Thresholds are committed constants.
pub fn grade_from(slope: f32, monotone_frac: f32, variance: f32) -> Grade {
    // an ideal identity transfer has slope 1.0. Effect size = how much of that slope is realised.
    if slope < 0.12 || monotone_frac < 0.5 {
        return Grade::Experimental; // no reliable / no monotone effect
    }
    if slope >= 0.55 && monotone_frac >= 0.9 && variance <= 0.03 {
        Grade::Strong
    } else if slope >= 0.30 && monotone_frac >= 0.7 && variance <= 0.06 {
        Grade::Moderate
    } else {
        Grade::Weak
    }
}

/// Evaluate the curve: realised metric for a `requested` value (piecewise-linear interpolation).
pub fn eval(curve: &ResponseCurve, requested: f32) -> f32 {
    interp(&curve.samples, requested.clamp(0.0, 1.0))
}

/// Pre-distort a requested value through the curve inverse (§13.2): returns the value to actually feed
/// the model so the *realised* result lands at `requested`. A no-op for uncontrollable attributes
/// (`experimental`) — there is nothing to correct toward — and clamped to `[0,1]`.
pub fn predistort(curve: &ResponseCurve, requested: f32) -> f32 {
    if curve.grade == Grade::Experimental || curve.samples.len() < 2 {
        return requested;
    }
    // invert: interpolate on (realised → requested). Samples are monotone in both axes.
    let inv: Vec<(f32, f32)> = curve.samples.iter().map(|&(a, b)| (b, a)).collect();
    interp(&inv, requested.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Piecewise-linear interpolation of `x` over sorted-by-first `pts` (clamped at the ends).
fn interp(pts: &[(f32, f32)], x: f32) -> f32 {
    if pts.is_empty() {
        return x;
    }
    if x <= pts[0].0 {
        return pts[0].1;
    }
    if x >= pts[pts.len() - 1].0 {
        return pts[pts.len() - 1].1;
    }
    for w in pts.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x >= x0 && x <= x1 {
            let t = if (x1 - x0).abs() < 1e-9 { 0.0 } else { (x - x0) / (x1 - x0) };
            return y0 + (y1 - y0) * t;
        }
    }
    pts[pts.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples_from<F: Fn(f32) -> f32>(f: F, n: usize) -> Vec<(f32, f32)> {
        (0..n).map(|i| { let x = i as f32 / (n - 1) as f32; (x, f(x)) }).collect()
    }

    #[test]
    fn identity_curve_is_strong_and_predistort_is_a_no_op() {
        let c = fit(samples_from(|x| x, 9), 0.0);
        assert_eq!(c.grade, Grade::Strong);
        assert!((c.slope - 1.0).abs() < 1e-3);
        for r in [0.0, 0.3, 0.7, 1.0] {
            assert!((predistort(&c, r) - r).abs() < 1e-3, "identity predistort at {r}");
        }
    }

    #[test]
    fn compressed_curve_predistorts_outward() {
        // realised = 0.5 + 0.4*(x-0.5): a family that under-shoots the extremes.
        let c = fit(samples_from(|x| 0.5 + 0.4 * (x - 0.5), 9), 0.0);
        // requesting 0.9 realises only 0.66 without correction; predistort must ask for MORE than 0.9.
        assert!(eval(&c, 0.9) < 0.7);
        assert!(predistort(&c, 0.9) > 0.9, "predistort expands to hit the target");
        // ...but the achievable max is capped at 1.0.
        assert!(predistort(&c, 0.9) <= 1.0);
    }

    #[test]
    fn flat_curve_is_experimental_and_predistort_passes_through() {
        let c = fit(samples_from(|_| 0.5, 9), 0.0);
        assert_eq!(c.grade, Grade::Experimental);
        assert!((predistort(&c, 0.8) - 0.8).abs() < 1e-6, "no correction when uncontrollable");
    }

    #[test]
    fn grade_reflects_slope_monotonicity_variance() {
        assert_eq!(grade_from(1.0, 1.0, 0.0), Grade::Strong);
        assert_eq!(grade_from(0.4, 0.8, 0.05), Grade::Moderate);
        assert_eq!(grade_from(0.2, 0.7, 0.05), Grade::Weak);
        assert_eq!(grade_from(0.05, 1.0, 0.0), Grade::Experimental); // too little effect
        assert_eq!(grade_from(0.8, 0.3, 0.0), Grade::Experimental); // non-monotone
        // high variance demotes a strong slope.
        assert_eq!(grade_from(0.8, 1.0, 0.1), Grade::Weak);
    }

    #[test]
    fn isotonic_repairs_a_backward_step() {
        // a dip in the middle is clamped monotone; grade drops for the non-monotonicity.
        let pairs = vec![(0.0, 0.0), (0.25, 0.4), (0.5, 0.2), (0.75, 0.6), (1.0, 0.9)];
        let c = fit(pairs, 0.02);
        // samples never decrease.
        for w in c.samples.windows(2) {
            assert!(w[1].1 >= w[0].1 - 1e-6);
        }
        assert_ne!(c.grade, Grade::Strong, "a backward step is not strong");
    }

    #[test]
    fn grade_weights_match_the_lexicon() {
        assert_eq!(Grade::Strong.weight(), 1.0);
        assert_eq!(Grade::Moderate.weight(), 0.7);
        assert_eq!(Grade::Weak.weight(), 0.4);
        assert_eq!(Grade::Experimental.weight(), 0.2);
    }
}
