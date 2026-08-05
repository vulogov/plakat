//! The tileability + PBR-validity scorecard (RFC TEXTURE-1 §12). Measures a [`Material`] so quality is
//! **falsifiable**: does it tile? is the normal a valid tangent-space map? is the albedo delit? All
//! pure, weight-free — the `bookart` `verify` analog. Drives `render --attempts N` rejection sampling.

use crate::texture::derive::Material;
use image::{GrayImage, RgbImage};

/// A tile join ≤ this ratio of the interior counts as seamless.
pub const SEAM_MAX: f32 = 1.5;
/// At least this fraction of normal texels must be valid unit vectors with +Z.
pub const NORMAL_VALID_MIN: f32 = 0.99;
/// Low-frequency albedo luminance std above this suggests baked lighting (a warning, not a hard fail).
pub const FLATNESS_MAX: f32 = 0.14;

#[derive(Debug, Clone)]
pub struct Scorecard {
    /// Edge-wrap seam of the albedo, x and y — join discontinuity / interior. ~1 = tiles like itself.
    pub tileability_x: f32,
    pub tileability_y: f32,
    /// Fraction of normal texels that are ~unit-length with +Z.
    pub normal_valid: f32,
    /// Low-frequency luminance std of the albedo (a delit proxy).
    pub albedo_flatness: f32,
    /// All channels share one resolution.
    pub consistent: bool,
    /// Whether the metallic / roughness maps carry spatial structure (a composite material) vs are flat
    /// (a single-class material — for which a uniform map is *correct*, not a defect). See notes.
    pub metallic_structured: bool,
    pub roughness_structured: bool,
    pub passes: bool,
    pub notes: Vec<String>,
}

/// Std-dev of a gray map in `[0,1]` and its mean — the flat-vs-structured probe. Std above a small
/// epsilon means the channel varies spatially (a composite material); at/below it the map is uniform.
fn map_stats(g: &GrayImage) -> (f32, f32) {
    let vals: Vec<f32> = g.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
    (std, mean)
}
/// A channel with std below this is "flat" (uniform / single-class).
pub const STRUCTURE_MIN_STD: f32 = 0.02;

fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

/// Edge-wrap seam ratio on one axis of a luma image: the join (col/row 0 vs last) discontinuity
/// relative to the interior adjacent difference. ~1 = seamless; ≫1 = a seam.
fn seam_ratio(luma: &GrayImage, horizontal: bool) -> f32 {
    let (w, h) = luma.dimensions();
    let g = |x: u32, y: u32| luma.get_pixel(x, y).0[0] as f32 / 255.0;
    let (mut join, mut interior) = (Vec::new(), Vec::new());
    if horizontal {
        for y in 0..h {
            join.push(g(0, y) - g(w - 1, y));
            for x in 1..w {
                interior.push(g(x, y) - g(x - 1, y));
            }
        }
    } else {
        for x in 0..w {
            join.push(g(x, 0) - g(x, h - 1));
            for y in 1..h {
                interior.push(g(x, y) - g(x, y - 1));
            }
        }
    }
    rms(&join) / rms(&interior).max(1e-6)
}

fn to_luma(a: &RgbImage) -> GrayImage {
    let mut g = GrayImage::new(a.width(), a.height());
    for (x, y, p) in a.enumerate_pixels() {
        let l = 0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32;
        g.put_pixel(x, y, image::Luma([l.round() as u8]));
    }
    g
}

/// Fraction of normal texels that decode to a ~unit vector with +Z (a valid tangent-space normal).
fn normal_validity(n: &RgbImage) -> f32 {
    let mut ok = 0u64;
    let total = (n.width() * n.height()) as u64;
    for p in n.pixels() {
        let v = [p.0[0], p.0[1], p.0[2]].map(|c| c as f32 / 255.0 * 2.0 - 1.0);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if (len - 1.0).abs() < 0.08 && v[2] > 0.0 {
            ok += 1;
        }
    }
    ok as f32 / total.max(1) as f32
}

/// Low-frequency luminance std of the albedo (baked-lighting proxy): downsample hard, measure spread.
fn low_freq_std(a: &RgbImage) -> f32 {
    let luma = to_luma(a);
    let small = image::imageops::resize(&luma, 8, 8, image::imageops::FilterType::Triangle);
    let vals: Vec<f32> = small.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt()
}

/// Score a material against the tileability + PBR-validity probes.
pub fn score(m: &Material) -> Scorecard {
    let dims = m.albedo.dimensions();
    let consistent = [
        m.normal.dimensions(),
        (m.height.width(), m.height.height()),
        (m.roughness.width(), m.roughness.height()),
        (m.metallic.width(), m.metallic.height()),
        (m.ao.width(), m.ao.height()),
    ]
    .iter()
    .all(|d| *d == dims);

    let luma = to_luma(&m.albedo);
    let tileability_x = seam_ratio(&luma, true);
    let tileability_y = seam_ratio(&luma, false);
    let normal_valid = normal_validity(&m.normal);
    let albedo_flatness = low_freq_std(&m.albedo);

    let mut notes = Vec::new();
    if tileability_x > SEAM_MAX {
        notes.push(format!("tileability-x {tileability_x:.2} > {SEAM_MAX} — a vertical seam"));
    }
    if tileability_y > SEAM_MAX {
        notes.push(format!("tileability-y {tileability_y:.2} > {SEAM_MAX} — a horizontal seam"));
    }
    if normal_valid < NORMAL_VALID_MIN {
        notes.push(format!("normal-valid {normal_valid:.3} < {NORMAL_VALID_MIN} — malformed normal map"));
    }
    if !consistent {
        notes.push("channels differ in resolution".into());
    }
    if albedo_flatness > FLATNESS_MAX {
        notes.push(format!("albedo-flatness {albedo_flatness:.3} > {FLATNESS_MAX} — baked lighting? (try delight)"));
    }

    // Flat-vs-structured (A4): a uniform metallic/roughness is CORRECT for a single-class material — say
    // so, so a flat map reads as a decision, not a bug. Structure means a composite material.
    let (met_std, met_mean) = map_stats(&m.metallic);
    let (rgh_std, _) = map_stats(&m.roughness);
    let metallic_structured = met_std >= STRUCTURE_MIN_STD;
    let roughness_structured = rgh_std >= STRUCTURE_MIN_STD;
    if metallic_structured {
        notes.push("metallic is structured — a composite (metal + dielectric) material".into());
    } else {
        let kind = if met_mean > 0.5 { "white = metal/conductor" } else { "black = dielectric" };
        notes.push(format!("metallic is uniform ({kind}) — correct for a single-class material, not a defect"));
    }
    if !roughness_structured {
        notes.push("roughness is uniform — correct for a single-class material".into());
    }

    // Flatness/structure are advisory; the hard gate is tiling + normal validity + consistency.
    let passes = tileability_x <= SEAM_MAX && tileability_y <= SEAM_MAX && normal_valid >= NORMAL_VALID_MIN && consistent;
    Scorecard { tileability_x, tileability_y, normal_valid, albedo_flatness, consistent, metallic_structured, roughness_structured, passes, notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::compile::ChannelSource;
    use image::{Rgb, RgbImage};

    fn tiled_albedo() -> RgbImage {
        // A smooth radial-ish pattern that already wraps (built from sines) → should score seamless.
        RgbImage::from_fn(48, 48, |x, y| {
            let u = (x as f32 / 48.0 * std::f32::consts::TAU).sin();
            let v = (y as f32 / 48.0 * std::f32::consts::TAU).sin();
            let c = ((u + v) * 0.25 + 0.5).clamp(0.0, 1.0);
            Rgb([(c * 200.0) as u8, (c * 180.0) as u8, (c * 150.0) as u8])
        })
    }

    #[test]
    fn a_wrapping_material_passes() {
        let m = Material::derive(tiled_albedo(), None, 1.0, true, 1.0, &ChannelSource::FromAlbedo, &ChannelSource::Scalar(0.0));
        let sc = score(&m);
        assert!(sc.tileability_x <= SEAM_MAX && sc.tileability_y <= SEAM_MAX, "seams: x={} y={}", sc.tileability_x, sc.tileability_y);
        assert!(sc.normal_valid >= NORMAL_VALID_MIN, "normal_valid {}", sc.normal_valid);
        assert!(sc.consistent && sc.passes, "{:?}", sc.notes);
    }

    #[test]
    fn a_seam_is_flagged() {
        // A left-half-dark / right-half-bright albedo has a hard vertical seam at the wrap.
        let a = RgbImage::from_fn(48, 48, |x, _| if x < 24 { Rgb([20, 20, 20]) } else { Rgb([220, 220, 220]) });
        let m = Material::derive(a, None, 1.0, true, 1.0, &ChannelSource::FromAlbedo, &ChannelSource::Scalar(0.0));
        let sc = score(&m);
        assert!(sc.tileability_x > SEAM_MAX, "expected a vertical seam, got {}", sc.tileability_x);
        assert!(!sc.passes);
    }
}
