//! Pigments and Kubelka-Munk mixing (RFC PAINT-1 §8.3).
//!
//! Mixing is subtractive, not an RGB lerp: an RGB average of blue and yellow gives grey, Kubelka-Munk gives
//! green, and that difference is most of why paint reads as paint. This is a single-constant KM model applied
//! per RGB channel (three "bands") — pragmatic and dependency-free, and it reproduces the key subtractive
//! behaviour. A full spectral (many-band) upgrade can replace the internals without touching callers.
//!
//! Each pigment is described by its opaque **masstone** (the colour of the pure pigment at full hiding). From
//! that we derive a per-channel `K/S` (absorption over scattering); a mixture's `K/S` is the
//! concentration-weighted average of its components', and the reflectance is recovered by inverting the KM
//! equation. White and black are ordinary pigments — white simply has a very low `K/S`.

use crate::paint::color::{self, LinRgb, Srgb};

/// A named pigment, defined by its opaque masstone in sRGB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pigment {
    pub name: &'static str,
    pub masstone: Srgb,
}

/// Clamp reflectance away from 0 and 1 so `K/S` stays finite at the extremes.
fn clamp_r(r: f32) -> f32 {
    r.clamp(0.003, 0.997)
}

/// Per-channel `K/S` for a reflectance triple: `K/S = (1 − R)² / (2R)` (Kubelka-Munk, complete hiding).
fn ks_from_linear(r: LinRgb) -> [f32; 3] {
    let mut ks = [0f32; 3];
    for i in 0..3 {
        let r = clamp_r(r[i]);
        ks[i] = (1.0 - r) * (1.0 - r) / (2.0 * r);
    }
    ks
}

/// Reflectance from `K/S`: `R = 1 + K/S − √((K/S)² + 2·K/S)`.
fn linear_from_ks(ks: [f32; 3]) -> LinRgb {
    let mut r = [0f32; 3];
    for i in 0..3 {
        let k = ks[i].max(0.0);
        r[i] = (1.0 + k - (k * k + 2.0 * k).sqrt()).clamp(0.0, 1.0);
    }
    r
}

/// How hard a pigment's K/S magnitude is damped when it MIXES (per unit of `TINT_DAMP`). Single-constant KM
/// makes `K/S` blow up as reflectance → 0, so a dark pigment (black, a deep earth) has a huge K/S and, weighted
/// only by concentration, swamps a mixture into mud — the "everything goes grey-black" tell. Damping a pigment's
/// mixing weight in proportion to its K/S counteracts that: a dark pigment still darkens, but no longer overrides
/// the hue of everything it touches, so mixtures keep their colour. Tuned against real scenes.
const TINT_DAMP: f32 = 0.03;

impl Pigment {
    /// The pigment's per-channel `K/S`, from its masstone.
    pub fn ks(&self) -> [f32; 3] {
        ks_from_linear(color::srgb_to_linear(self.masstone))
    }

    /// Per-pigment TINTING STRENGTH for MIXING (0..1]: the mixing weight multiplier that damps the single-constant
    /// K/S blow-up (see [`TINT_DAMP`]). Darker/higher-K/S pigments get a lower weight so they stop muddying every
    /// mixture; light, low-K/S pigments stay ~1. A SOLO pigment is unaffected (the weight cancels in the average),
    /// so a pigment painted alone still reads as its masstone.
    pub fn tinting(&self) -> f32 {
        let ks = self.ks();
        let mean = (ks[0] + ks[1] + ks[2]) / 3.0;
        1.0 / (1.0 + TINT_DAMP * mean)
    }
}

/// Mix pigments in the given (non-negative) concentrations under Kubelka-Munk, returning the resulting sRGB.
/// Concentrations are relative — the mixture is their **weighted average** in `K/S`, so scaling them all by a
/// constant leaves the colour unchanged (it is a proportion, not an amount). A zero-total or empty mix returns
/// mid-grey rather than panicking.
pub fn mix(pigments: &[Pigment], concentrations: &[f32]) -> Srgb {
    color::linear_to_srgb(mix_linear(pigments, concentrations))
}

/// As [`mix`] but returning linear reflectance (for downstream colour math without a double conversion).
pub fn mix_linear(pigments: &[Pigment], concentrations: &[f32]) -> LinRgb {
    let n = pigments.len().min(concentrations.len());
    let total: f32 = concentrations[..n].iter().copied().map(|c| c.max(0.0)).sum();
    if n == 0 || total <= 0.0 {
        return color::srgb_to_linear([128, 128, 128]); // a defined, safe mid-grey
    }
    // Weight each pigment by concentration × its TINTING STRENGTH, so a dark high-K/S pigment stops swamping the
    // mixture into mud. Accumulate the weighted K/S and the weight sum in one pass, then normalise — so it stays a
    // proportion (scaling all concentrations leaves colour unchanged) and a solo pigment's weight cancels (its
    // masstone is preserved). Allocation-free (hot path: called per pixel).
    // The per-pigment constants are pure functions of the masstone; gathered here, mixed in `mix_linear_pre` (the
    // solver precomputes them once per palette and calls the same core — identical arithmetic, so its answers
    // are the same to the bit). Stack arrays up to 16 pigments (every built-in and derived palette), else a Vec.
    if n <= 16 {
        let mut ksl = [[0f32; 3]; 16];
        let mut tint = [0f32; 16];
        for i in 0..n {
            ksl[i] = pigments[i].ks();
            tint[i] = pigments[i].tinting();
        }
        mix_linear_pre(&ksl[..n], &tint[..n], &concentrations[..n], total)
    } else {
        let ksl: Vec<[f32; 3]> = pigments[..n].iter().map(|p| p.ks()).collect();
        let tint: Vec<f32> = pigments[..n].iter().map(|p| p.tinting()).collect();
        mix_linear_pre(&ksl, &tint, &concentrations[..n], total)
    }
}

/// The mixing core of [`mix_linear`] on precomputed per-pigment `K/S` and tinting strength, `total` being the
/// concentrations' (non-negative) sum. Allocation-free; the solver's hot path.
pub fn mix_linear_pre(ksl: &[[f32; 3]], tint: &[f32], concentrations: &[f32], total: f32) -> LinRgb {
    let n = ksl.len().min(tint.len()).min(concentrations.len());
    if n == 0 || total <= 0.0 {
        return color::srgb_to_linear([128, 128, 128]);
    }
    let mut ks = [0f32; 3];
    let mut wsum = 0f32;
    for i in 0..n {
        let w = (concentrations[i].max(0.0) / total) * tint[i];
        let p = ksl[i];
        for c in 0..3 {
            ks[c] += w * p[c];
        }
        wsum += w;
    }
    if wsum <= 0.0 {
        return color::srgb_to_linear([128, 128, 128]);
    }
    for c in 0..3 {
        ks[c] /= wsum;
    }
    linear_from_ks(ks)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLUE: Pigment = Pigment { name: "ultramarine", masstone: [30, 50, 180] };
    const YELLOW: Pigment = Pigment { name: "cadmium-yellow", masstone: [240, 220, 30] };
    const WHITE: Pigment = Pigment { name: "titanium-white", masstone: [252, 251, 248] };

    #[test]
    fn blue_plus_yellow_is_green_not_grey() {
        // The canonical Kubelka-Munk demonstration: subtractive mixing yields green.
        let mixed = mix(&[BLUE, YELLOW], &[1.0, 1.0]);
        assert!(mixed[1] > mixed[0] && mixed[1] > mixed[2], "green channel dominates: {mixed:?}");
        // …whereas a naive RGB average would be a desaturated grey-purple (G is NOT the max there).
        let rgb_avg = [(30 + 240) / 2, (50 + 220) / 2, (180 + 30) / 2];
        assert!(!(rgb_avg[1] > rgb_avg[0] && rgb_avg[1] > rgb_avg[2]), "RGB average is not green: {rgb_avg:?}");
    }

    #[test]
    fn concentration_is_a_proportion_not_an_amount() {
        let a = mix(&[BLUE, YELLOW], &[1.0, 1.0]);
        let b = mix(&[BLUE, YELLOW], &[5.0, 5.0]);
        assert_eq!(a, b, "scaling all concentrations leaves the colour unchanged");
    }

    #[test]
    fn white_lightens_a_mixture() {
        let base = color::srgb_to_lab(mix(&[BLUE, YELLOW], &[1.0, 1.0]));
        let tinted = color::srgb_to_lab(mix(&[BLUE, YELLOW, WHITE], &[1.0, 1.0, 2.0]));
        assert!(tinted.l > base.l + 5.0, "adding white raises L*: {} → {}", base.l, tinted.l);
    }

    #[test]
    fn a_single_pigment_returns_near_its_masstone() {
        let just_blue = mix(&[BLUE], &[1.0]);
        let d = color::delta_e76(color::srgb_to_lab(just_blue), color::srgb_to_lab(BLUE.masstone));
        assert!(d < 12.0, "one pigment ≈ its masstone (ΔE {d})");
    }

    #[test]
    fn empty_or_zero_mix_is_safe() {
        assert_eq!(mix(&[], &[]), [128, 128, 128]);
        assert_eq!(mix(&[BLUE], &[0.0]), [128, 128, 128]);
    }
}
