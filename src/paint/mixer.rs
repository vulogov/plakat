//! The mixture solver (RFC PAINT-1 §8.3): given a target colour and a palette, find the combination of **at
//! most three** pigments whose Kubelka-Munk mixture is closest to the target in perceptual ΔE. The
//! three-pigment cap is not a performance limit — it is how a painter actually mixes, and it generates colour
//! harmony for free (every load on the canvas is reachable from the same few pigments).
//!
//! The search is an exhaustive scan over pigment combinations with a concentration grid per combination:
//! deterministic, and cheap at the palette sizes in use (4–8 pigments). A per-call cache keyed on the
//! quantised target keeps repeated loads (thousands of strokes) fast; the solver itself is pure.

use crate::paint::color::{self, Srgb};
use crate::paint::palette::Palette;
use crate::paint::pigment::{self, Pigment};

/// Concentration grid divisions for the 2- and 3-pigment searches. 16 → 1/16 steps.
const GRID: usize = 16;

/// A solved brush load: which palette pigments, in what proportions, and how close it lands.
#[derive(Clone, Debug)]
pub struct Mixture {
    /// Indices into the palette's `pigments`.
    pub pigments: Vec<usize>,
    /// Matching non-negative weights (proportions; they sum to 1).
    pub weights: Vec<f32>,
    /// The achieved colour.
    pub srgb: Srgb,
    /// Perceptual distance to the requested target (CIE76).
    pub delta_e: f32,
}

impl Mixture {
    /// The concrete pigments (resolved against a palette), for handing to [`pigment::mix`].
    pub fn resolve(&self, palette: &Palette) -> Vec<Pigment> {
        self.pigments.iter().map(|&i| palette.pigments[i]).collect()
    }
}

fn eval(palette: &Palette, idx: &[usize], weights: &[f32], target: color::Lab) -> (Srgb, f32) {
    let pigs: Vec<Pigment> = idx.iter().map(|&i| palette.pigments[i]).collect();
    let srgb = pigment::mix(&pigs, weights);
    (srgb, color::delta_e76(color::srgb_to_lab(srgb), target))
}

/// Solve for the best ≤`max_pigments` (capped at 3) mixture of `palette` matching `target`.
///
/// The scan is exhaustive (every combination × the concentration grid) and was the painter's whole cost: ~24k
/// Kubelka-Munk evaluations a solve on a 12-pigment palette, each building vectors and re-deriving every
/// pigment's K/S. It now runs on per-pigment constants computed once and evaluates each candidate on the stack,
/// in the same order with the same comparisons and the same arithmetic — the answer is bit-identical to the
/// original scan (kept as `solve_mixture_reference` for the test that proves it).
pub fn solve_mixture(palette: &Palette, target: Srgb, max_pigments: usize) -> Mixture {
    let cap = max_pigments.clamp(1, 3);
    let tlab = color::srgb_to_lab(target);
    let n = palette.pigments.len();
    let ksl: Vec<[f32; 3]> = palette.pigments.iter().map(|p| p.ks()).collect();
    let tint: Vec<f32> = palette.pigments.iter().map(|p| p.tinting()).collect();
    // The running best: (ΔE, which pigments, how many, their weights, the colour). Strictly-less wins, so on an
    // exact tie the earlier candidate stays — as before.
    let mut best_de = f32::INFINITY;
    let mut best_idx = [0usize; 3];
    let mut best_len = 1usize;
    let mut best_w = [1.0f32, 0.0, 0.0];
    let mut best_srgb: Srgb = palette.pigments.first().map(|p| p.masstone).unwrap_or([128, 128, 128]);
    let mut consider = |idx: [usize; 3], len: usize, w: [f32; 3]| {
        let mut k3 = [[0f32; 3]; 3];
        let mut t3 = [0f32; 3];
        for c in 0..len {
            k3[c] = ksl[idx[c]];
            t3[c] = tint[idx[c]];
        }
        // `pigment::mix` sums the (non-negative) concentrations first; the same sum, the same order.
        let total: f32 = w[..len].iter().copied().map(|c| c.max(0.0)).sum();
        let srgb = color::linear_to_srgb(pigment::mix_linear_pre(&k3[..len], &t3[..len], &w[..len], total));
        let de = color::delta_e76(color::srgb_to_lab(srgb), tlab);
        if de < best_de {
            best_de = de;
            best_idx = idx;
            best_len = len;
            best_w = w;
            best_srgb = srgb;
        }
    };

    // 1 pigment.
    for i in 0..n {
        consider([i, 0, 0], 1, [1.0, 0.0, 0.0]);
    }
    // 2 pigments — scan the mixing ratio.
    if cap >= 2 {
        for i in 0..n {
            for j in (i + 1)..n {
                for s in 1..GRID {
                    let t = s as f32 / GRID as f32;
                    consider([i, j, 0], 2, [t, 1.0 - t, 0.0]);
                }
            }
        }
    }
    // 3 pigments — scan the barycentric interior.
    if cap >= 3 {
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for a in 1..GRID {
                        for b in 1..(GRID - a) {
                            let (wa, wb) = (a as f32 / GRID as f32, b as f32 / GRID as f32);
                            let wc = 1.0 - wa - wb;
                            if wc > 0.0 {
                                consider([i, j, k], 3, [wa, wb, wc]);
                            }
                        }
                    }
                }
            }
        }
    }
    drop(consider);
    let mut best = Mixture { pigments: best_idx[..best_len].to_vec(), weights: best_w[..best_len].to_vec(), srgb: best_srgb, delta_e: best_de };

    // Local refinement (coordinate descent) of the winning combination's weights — the coarse grid can't add a
    // whisker of a strong tinter like black, so it lands close; this dials the proportions in precisely.
    let idx = best.pigments.clone();
    let mut w = best.weights.clone();
    let mut step = 1.0 / GRID as f32;
    while step > 1e-4 {
        let mut improved = false;
        for c in 0..w.len() {
            for d in [step, -step] {
                let mut cand = w.clone();
                cand[c] = (cand[c] + d).max(0.0);
                let s: f32 = cand.iter().sum();
                if s <= 0.0 {
                    continue;
                }
                for x in cand.iter_mut() {
                    *x /= s;
                }
                let (srgb, de) = eval(palette, &idx, &cand, tlab);
                if de + 1e-6 < best.delta_e {
                    best = Mixture { pigments: idx.clone(), weights: cand.clone(), srgb, delta_e: de };
                    w = cand;
                    improved = true;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    best
}

/// The original solver, kept verbatim for the test that proves the fast scan is bit-identical to it.
#[cfg(test)]
fn solve_mixture_reference(palette: &Palette, target: Srgb, max_pigments: usize) -> Mixture {
    let cap = max_pigments.clamp(1, 3);
    let tlab = color::srgb_to_lab(target);
    let n = palette.pigments.len();
    let mut best = Mixture { pigments: vec![0], weights: vec![1.0], srgb: palette.pigments.first().map(|p| p.masstone).unwrap_or([128, 128, 128]), delta_e: f32::INFINITY };
    let mut consider = |idx: Vec<usize>, weights: Vec<f32>| {
        let (srgb, de) = eval(palette, &idx, &weights, tlab);
        if de < best.delta_e {
            best = Mixture { pigments: idx, weights, srgb, delta_e: de };
        }
    };
    for i in 0..n {
        consider(vec![i], vec![1.0]);
    }
    if cap >= 2 {
        for i in 0..n {
            for j in (i + 1)..n {
                for s in 1..GRID {
                    let t = s as f32 / GRID as f32;
                    consider(vec![i, j], vec![t, 1.0 - t]);
                }
            }
        }
    }
    if cap >= 3 {
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for a in 1..GRID {
                        for b in 1..(GRID - a) {
                            let (wa, wb) = (a as f32 / GRID as f32, b as f32 / GRID as f32);
                            let wc = 1.0 - wa - wb;
                            if wc > 0.0 {
                                consider(vec![i, j, k], vec![wa, wb, wc]);
                            }
                        }
                    }
                }
            }
        }
    }
    drop(consider);
    let idx = best.pigments.clone();
    let mut w = best.weights.clone();
    let mut step = 1.0 / GRID as f32;
    while step > 1e-4 {
        let mut improved = false;
        for c in 0..w.len() {
            for d in [step, -step] {
                let mut cand = w.clone();
                cand[c] = (cand[c] + d).max(0.0);
                let s: f32 = cand.iter().sum();
                if s <= 0.0 {
                    continue;
                }
                for x in cand.iter_mut() {
                    *x /= s;
                }
                let (srgb, de) = eval(palette, &idx, &cand, tlab);
                if de + 1e-6 < best.delta_e {
                    best = Mixture { pigments: idx.clone(), weights: cand.clone(), srgb, delta_e: de };
                    w = cand;
                    improved = true;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    #[test]
    fn the_fast_scan_is_bit_identical_to_the_original_solver() {
        // Every accepted render was mixed by the original scan; the allocation-free one must give the same
        // pigments, the same weights and the same colour, to the bit, across palettes and a spread of targets.
        let mut n = 0;
        for pal in [&palette::ZORN, &palette::EARTH, &palette::SPLIT_PRIMARY] {
            for r in (0..256).step_by(51) {
                for g in (0..256).step_by(85) {
                    for b in (0..256).step_by(63) {
                        let t = [r as u8, g as u8, b as u8];
                        for cap in [1, 2, 3] {
                            let a = solve_mixture(pal, t, cap);
                            let x = solve_mixture_reference(pal, t, cap);
                            assert_eq!(a.pigments, x.pigments, "{} {t:?} cap {cap}", pal.name);
                            assert_eq!(a.weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), x.weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{} {t:?} cap {cap}", pal.name);
                            assert_eq!(a.srgb, x.srgb);
                            assert_eq!(a.delta_e.to_bits(), x.delta_e.to_bits());
                            n += 1;
                        }
                    }
                }
            }
        }
        assert!(n > 500);
    }

    #[test]
    fn solves_a_reachable_colour_closely() {
        // Build the target by mixing known Zorn pigments, so it is guaranteed inside the gamut, then check the
        // solver recovers it perceptually (a warm flesh half-tone: ochre + white + a whisker of black).
        let target = crate::paint::pigment::mix(&palette::ZORN.pigments[..], &[0.5, 0.0, 0.08, 0.42]);
        let m = solve_mixture(&palette::ZORN, target, 3);
        assert!(m.delta_e < 3.0, "close match (ΔE {}): {:?} → {:?} vs target {:?}", m.delta_e, m.pigments, m.srgb, target);
        assert!(m.pigments.len() <= 3 && !m.pigments.is_empty());
        assert!((m.weights.iter().sum::<f32>() - 1.0).abs() < 1e-4, "weights sum to 1");
    }

    #[test]
    fn is_deterministic() {
        let a = solve_mixture(&palette::SPLIT_PRIMARY, [90, 140, 70], 3);
        let b = solve_mixture(&palette::SPLIT_PRIMARY, [90, 140, 70], 3);
        assert_eq!(a.srgb, b.srgb);
        assert_eq!(a.pigments, b.pigments);
    }

    #[test]
    fn respects_the_pigment_cap() {
        let m1 = solve_mixture(&palette::EARTH, [130, 90, 60], 1);
        assert_eq!(m1.pigments.len(), 1, "cap 1 → single pigment");
        let m2 = solve_mixture(&palette::EARTH, [130, 90, 60], 2);
        assert!(m2.pigments.len() <= 2);
        // A richer mix is never worse than a poorer one.
        let m3 = solve_mixture(&palette::EARTH, [130, 90, 60], 3);
        assert!(m3.delta_e <= m1.delta_e + 1e-3, "3 pigments ≤ 1 pigment error");
    }

    #[test]
    fn out_of_gamut_still_returns_the_nearest() {
        // A saturated cyan is out of the Zorn (warm) gamut — the solver returns the closest reachable colour.
        let m = solve_mixture(&palette::ZORN, [0, 200, 220], 3);
        assert!(m.delta_e.is_finite() && !m.pigments.is_empty(), "nearest-in-gamut, not a panic");
    }
}
