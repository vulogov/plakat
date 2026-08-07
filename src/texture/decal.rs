//! Track B (ROADMAP 6.5.0) — **decals**: alpha-masked overlay materials (a rust streak, crack, sign)
//! that layer onto a base material. A decal is a [`Material`] + an **opacity** mask; `apply` stamps it
//! onto a base PBR set — alpha-blending albedo/roughness/metallic/height and blending the normal via
//! **Reoriented Normal Mapping** (G0.1-validated) so decal detail rides the base slope, not flattens it.
//! Weight-free.

use crate::texture::derive::Material;
use image::{GrayImage, Luma, Rgb, RgbImage};

/// A decal = a material + an opacity (alpha) mask (white = opaque decal, black = transparent).
pub struct Decal {
    pub material: Material,
    pub opacity: GrayImage,
}

/// Placement of a decal onto a base: centre in normalised `[0,1]` base coords, `scale` as a fraction of
/// the base edge, `rotate` in degrees, `tile` to repeat the decal across the whole base.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub cx: f32,
    pub cy: f32,
    pub scale: f32,
    pub rotate_deg: f32,
    pub tile: bool,
}

impl Default for Placement {
    fn default() -> Self {
        Self { cx: 0.5, cy: 0.5, scale: 0.5, rotate_deg: 0.0, tile: false }
    }
}

// --- B2: Reoriented Normal Mapping (the G0.1 formula) ---------------------------------------------

/// RNM blend of a `detail` tangent-space normal over a `base` normal (both `[0,1]` RGB). Returns the
/// unit-length blended normal so decal detail sits on the base surface. (Validated in
/// `examples/texture_rnm_probe.rs`: flat detail → base unchanged, detail amplitude preserved.)
pub fn rnm(base: Rgb<u8>, detail: Rgb<u8>) -> [f32; 3] {
    let b = base.0.map(|c| c as f32 / 255.0);
    let d = detail.0.map(|c| c as f32 / 255.0);
    let t = [b[0] * 2.0 - 1.0, b[1] * 2.0 - 1.0, b[2] * 2.0];
    let u = [d[0] * -2.0 + 1.0, d[1] * -2.0 + 1.0, d[2] * 2.0 - 1.0];
    let dt = t[0] * u[0] + t[1] * u[1] + t[2] * u[2];
    let r = [t[0] * dt / t[2] - u[0], t[1] * dt / t[2] - u[1], t[2] * dt / t[2] - u[2]];
    let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-9);
    [r[0] / l, r[1] / l, r[2] / l]
}

fn dec_n(p: Rgb<u8>) -> [f32; 3] {
    [0, 1, 2].map(|i| p.0[i] as f32 / 255.0 * 2.0 - 1.0)
}
fn enc_n(v: [f32; 3]) -> Rgb<u8> {
    Rgb([0, 1, 2].map(|i| ((v[i] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8))
}
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t).round().clamp(0.0, 255.0) as u8
}

// --- B1: procedural opacity shapes ----------------------------------------------------------------

fn hash2(x: u32, y: u32, salt: u32) -> f32 {
    let mut h = x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263)).wrapping_add(salt.wrapping_mul(2246822519));
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}

/// A procedural opacity mask (white = opaque). `kind`: `circle` | `ring` | `stripe` | `splatter` | `crack`.
pub fn opacity_shape(kind: &str, size: u32) -> GrayImage {
    let s = size.max(1);
    GrayImage::from_fn(s, s, |x, y| {
        let (u, v) = (x as f32 / s as f32, y as f32 / s as f32);
        let (dx, dy) = (u - 0.5, v - 0.5);
        let r = (dx * dx + dy * dy).sqrt();
        let a = match kind.to_ascii_lowercase().as_str() {
            "circle" | "dot" => if r < 0.45 { 1.0 } else { 0.0 },
            "ring" => if (r - 0.38).abs() < 0.07 { 1.0 } else { 0.0 },
            "stripe" => if ((u + v) * 6.0).fract() < 0.5 && r < 0.48 { 1.0 } else { 0.0 },
            "splatter" => {
                // a few soft blobs
                let mut a = 0.0f32;
                for i in 0..7 {
                    let (bx, by) = (hash2(i, 0, 1), hash2(i, 0, 2));
                    let br = 0.05 + 0.09 * hash2(i, 0, 3);
                    let d = ((u - bx).powi(2) + (v - by).powi(2)).sqrt();
                    if d < br {
                        a = a.max(1.0 - d / br);
                    }
                }
                a
            }
            "crack" => {
                // radial branching cracks: opaque near a few angular spokes, thinning outward
                let ang = dy.atan2(dx);
                let spoke = (ang * 3.0).sin().abs();
                if r < 0.48 && spoke < 0.12 - 0.15 * r + 0.06 * hash2(x, y, 5) {
                    1.0
                } else {
                    0.0
                }
            }
            _ => if r < 0.45 { 1.0 } else { 0.0 }, // default = circle
        };
        Luma([(a.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

// --- B3: apply a decal onto a base material -------------------------------------------------------

/// Bilinear-ish (nearest) sample of a decal channel at decal-UV `[0,1)`; `None` if outside and not tiling.
fn duv(bx: u32, by: u32, bw: u32, bh: u32, p: &Placement) -> Option<(f32, f32)> {
    let (cx, cy) = (p.cx * bw as f32, p.cy * bh as f32);
    let size = (p.scale.max(1e-3)) * bw as f32; // decal edge in base px
    let (ox, oy) = (bx as f32 + 0.5 - cx, by as f32 + 0.5 - cy);
    let th = -p.rotate_deg.to_radians();
    let (c, s) = (th.cos(), th.sin());
    let (rx, ry) = (ox * c - oy * s, ox * s + oy * c);
    let (mut u, mut v) = ((rx + size / 2.0) / size, (ry + size / 2.0) / size);
    if p.tile {
        u = u.rem_euclid(1.0);
        v = v.rem_euclid(1.0);
    } else if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
        return None;
    }
    Some((u, v))
}

fn sample_gray(g: &GrayImage, u: f32, v: f32) -> u8 {
    let (w, h) = g.dimensions();
    g.get_pixel(((u * w as f32) as u32).min(w - 1), ((v * h as f32) as u32).min(h - 1)).0[0]
}
fn sample_rgb(g: &RgbImage, u: f32, v: f32) -> Rgb<u8> {
    let (w, h) = g.dimensions();
    *g.get_pixel(((u * w as f32) as u32).min(w - 1), ((v * h as f32) as u32).min(h - 1))
}

/// Composite `decal` onto `base` at `placement`. Alpha-blends albedo/roughness/metallic/height, blends
/// the normal via RNM (weighted by opacity), then re-derives AO from the new height. Base tiles are
/// preserved where the decal is transparent.
pub fn apply(base: &Material, decal: &Decal, placement: &Placement, ao_strength: f32) -> Material {
    let (w, h) = base.albedo.dimensions();
    let mut albedo = base.albedo.clone();
    let mut normal = base.normal.clone();
    let mut roughness = base.roughness.clone();
    let mut metallic = base.metallic.clone();
    let mut height = base.height.clone();

    for by in 0..h {
        for bx in 0..w {
            let Some((u, v)) = duv(bx, by, w, h, placement) else { continue };
            let a = sample_gray(&decal.opacity, u, v) as f32 / 255.0;
            if a <= 0.003 {
                continue;
            }
            // albedo / roughness / metallic / height: alpha blend
            let da = sample_rgb(&decal.material.albedo, u, v);
            let ba = albedo.get_pixel(bx, by).0;
            albedo.put_pixel(bx, by, Rgb([lerp_u8(ba[0], da[0], a), lerp_u8(ba[1], da[1], a), lerp_u8(ba[2], da[2], a)]));
            roughness.put_pixel(bx, by, Luma([lerp_u8(roughness.get_pixel(bx, by).0[0], sample_gray(&decal.material.roughness, u, v), a)]));
            metallic.put_pixel(bx, by, Luma([lerp_u8(metallic.get_pixel(bx, by).0[0], sample_gray(&decal.material.metallic, u, v), a)]));
            height.put_pixel(bx, by, Luma([lerp_u8(height.get_pixel(bx, by).0[0], sample_gray(&decal.material.height, u, v), a)]));
            // normal: RNM(base, decal), then lerp base→RNM by opacity, renormalised
            let base_n = *normal.get_pixel(bx, by);
            let rn = rnm(base_n, sample_rgb(&decal.material.normal, u, v));
            let bn = dec_n(base_n);
            let mixed = [0, 1, 2].map(|i| bn[i] * (1.0 - a) + rn[i] * a);
            let l = (mixed[0] * mixed[0] + mixed[1] * mixed[1] + mixed[2] * mixed[2]).sqrt().max(1e-9);
            normal.put_pixel(bx, by, enc_n([mixed[0] / l, mixed[1] / l, mixed[2] / l]));
        }
    }
    // AO follows the new height (a decal that adds relief casts/receives cavity).
    let ao = crate::texture::derive::ao_from_height(&height, ao_strength);
    Material { albedo, height, normal, roughness, metallic, ao, anisotropy: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::compile::ChannelSource;

    fn mat(fill: u8) -> Material {
        Material::derive(RgbImage::from_pixel(64, 64, Rgb([fill, fill, fill])), None, 1.0, true, 1.0, &ChannelSource::Scalar(0.5), &ChannelSource::Scalar(0.0))
    }

    #[test]
    fn rnm_flat_detail_returns_base() {
        // Flat detail (straight-up normal) over any base → the base normal unchanged (the G0.1 property).
        let base = enc_n([-0.5, 0.2, 1.0]);
        let flat = enc_n([0.0, 0.0, 1.0]);
        let r = rnm(base, flat);
        let b = dec_n(base);
        let bl = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let bn = [b[0] / bl, b[1] / bl, b[2] / bl];
        assert!((0..3).all(|i| (r[i] - bn[i]).abs() < 0.01), "flat detail → base: {r:?} vs {bn:?}");
    }

    #[test]
    fn apply_stamps_the_decal_where_opaque_and_leaves_base_elsewhere() {
        let base = mat(60);
        let decal_mat = mat(220);
        let mut opacity = GrayImage::new(64, 64);
        // opaque only in the centre quarter
        for y in 24..40 {
            for x in 24..40 {
                opacity.put_pixel(x, y, Luma([255]));
            }
        }
        let d = Decal { material: decal_mat, opacity };
        let out = apply(&base, &d, &Placement { cx: 0.5, cy: 0.5, scale: 1.0, rotate_deg: 0.0, tile: false }, 1.0);
        assert!((out.albedo.get_pixel(32, 32).0[0] as i32 - 220).abs() <= 4, "centre = decal");
        assert!((out.albedo.get_pixel(4, 4).0[0] as i32 - 60).abs() <= 4, "corner = base (untouched)");
    }

    #[test]
    fn opacity_shapes_produce_a_mask() {
        for k in ["circle", "ring", "stripe", "splatter", "crack"] {
            let m = opacity_shape(k, 48);
            assert!(m.pixels().any(|p| p.0[0] > 128), "{k} produced some opaque pixels");
            assert!(m.pixels().any(|p| p.0[0] < 128), "{k} produced some transparent pixels");
        }
    }
}
