//! Colour science for the paint engine — sRGB ↔ linear ↔ XYZ ↔ CIELAB and ΔE, dependency-free so the whole
//! paint module builds under `--no-default-features` (the `palette` crate is `fractals`-feature-gated).
//!
//! The mixer (§8.3) solves each brush load as the pigment combination minimising perceptual ΔE against a
//! target colour, so a correct sRGB↔Lab path is load-bearing, not cosmetic.

/// An sRGB triple in `[0,255]`.
pub type Srgb = [u8; 3];
/// A linear-light RGB triple in `[0,1]`.
pub type LinRgb = [f32; 3];

/// One sRGB channel (0–255) to linear light `[0,1]` (IEC 61966-2-1).
pub fn srgb_channel_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// One linear-light channel `[0,1]` to sRGB (0–255), rounded and clamped.
pub fn linear_channel_to_srgb(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let s = if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}

/// sRGB triple → linear-light RGB.
pub fn srgb_to_linear(c: Srgb) -> LinRgb {
    [srgb_channel_to_linear(c[0]), srgb_channel_to_linear(c[1]), srgb_channel_to_linear(c[2])]
}

/// Linear-light RGB → sRGB triple.
pub fn linear_to_srgb(l: LinRgb) -> Srgb {
    [linear_channel_to_srgb(l[0]), linear_channel_to_srgb(l[1]), linear_channel_to_srgb(l[2])]
}

/// A CIELAB colour (D65 white point).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

// sRGB (D65) → XYZ matrix, and the D65 white point.
const XR: [f32; 3] = [0.4124564, 0.3575761, 0.1804375];
const YR: [f32; 3] = [0.2126729, 0.7151522, 0.0721750];
const ZR: [f32; 3] = [0.0193339, 0.1191920, 0.9503041];
const XN: f32 = 0.95047;
const YN: f32 = 1.0;
const ZN: f32 = 1.08883;

fn lab_f(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D * D * D {
        t.cbrt()
    } else {
        t / (3.0 * D * D) + 4.0 / 29.0
    }
}

/// Linear-light RGB → CIELAB.
pub fn linear_to_lab(rgb: LinRgb) -> Lab {
    let x = XR[0] * rgb[0] + XR[1] * rgb[1] + XR[2] * rgb[2];
    let y = YR[0] * rgb[0] + YR[1] * rgb[1] + YR[2] * rgb[2];
    let z = ZR[0] * rgb[0] + ZR[1] * rgb[1] + ZR[2] * rgb[2];
    let (fx, fy, fz) = (lab_f(x / XN), lab_f(y / YN), lab_f(z / ZN));
    Lab { l: 116.0 * fy - 16.0, a: 500.0 * (fx - fy), b: 200.0 * (fy - fz) }
}

/// sRGB → CIELAB.
pub fn srgb_to_lab(c: Srgb) -> Lab {
    linear_to_lab(srgb_to_linear(c))
}

/// CIE76 colour difference (Euclidean in Lab). Adequate for mixture ranking; CIEDE2000 can replace it later
/// without touching callers.
pub fn delta_e76(a: Lab, b: Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

/// Rec.601 relative luminance of a linear-RGB triple `[0,1]` — used for value (notan) work.
pub fn linear_luma(rgb: LinRgb) -> f32 {
    0.2126729 * rgb[0] + 0.7151522 * rgb[1] + 0.0721750 * rgb[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_roundtrips() {
        for c in [0u8, 1, 18, 64, 128, 200, 255] {
            let back = linear_channel_to_srgb(srgb_channel_to_linear(c));
            assert!((back as i32 - c as i32).abs() <= 1, "roundtrip {c} → {back}");
        }
    }

    #[test]
    fn known_lab_values() {
        // White and black anchor the L axis.
        let white = srgb_to_lab([255, 255, 255]);
        assert!((white.l - 100.0).abs() < 0.5 && white.a.abs() < 1.0 && white.b.abs() < 1.0, "white ≈ L100: {white:?}");
        let black = srgb_to_lab([0, 0, 0]);
        assert!(black.l.abs() < 0.5, "black ≈ L0: {black:?}");
        // Pure red is high L*, strongly +a*.
        let red = srgb_to_lab([255, 0, 0]);
        assert!(red.a > 60.0 && red.b > 30.0, "red is +a +b: {red:?}");
    }

    #[test]
    fn delta_e_is_zero_for_identical_and_grows_with_distance() {
        let a = srgb_to_lab([120, 80, 60]);
        assert_eq!(delta_e76(a, a), 0.0);
        let near = srgb_to_lab([122, 82, 62]);
        let far = srgb_to_lab([20, 200, 240]);
        assert!(delta_e76(a, near) < delta_e76(a, far), "closer colour → smaller ΔE");
    }
}
