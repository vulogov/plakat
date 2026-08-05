//! Channel derivation (RFC TEXTURE-1 §8) — the **weight-free** half. From an albedo (+ optionally a
//! height), derive the rest of the PBR set with pure, deterministic, **circular** image ops so every
//! derived map **tiles**: height (luminance), normal (circular Sobel → tangent-space, the G0.4-proven
//! math), AO (circular cavity), and roughness/metallic (from-albedo heuristics or a flat scalar).

use crate::texture::compile::ChannelSource;
use image::{GrayImage, Luma, Rgb, RgbImage};

/// The full PBR channel set, in memory.
pub struct Material {
    pub albedo: RgbImage,
    pub height: GrayImage,
    pub normal: RgbImage,
    pub roughness: GrayImage,
    pub metallic: GrayImage,
    pub ao: GrayImage,
}

impl Material {
    /// Derive the full set from an albedo (+ optional supplied height) and the resolved channel sources.
    /// Any `Prompt`/generation source falls back to the weight-free path here (the generation passes are
    /// B4/B5); the caller decides when to generate instead.
    pub fn derive(
        albedo: RgbImage,
        height: Option<GrayImage>,
        normal_strength: f32,
        opengl: bool,
        ao_strength: f32,
        roughness: &ChannelSource,
        metallic: &ChannelSource,
    ) -> Material {
        let height = height.unwrap_or_else(|| height_from_albedo(&albedo));
        let normal = normal_from_height(&height, normal_strength, opengl);
        let ao = ao_from_height(&height, ao_strength);
        let roughness = match roughness {
            ChannelSource::Scalar(v) => flat_map(albedo.width(), albedo.height(), *v),
            ChannelSource::Auto => roughness_auto(&albedo),
            _ => roughness_from_albedo(&albedo),
        };
        let metallic = match metallic {
            ChannelSource::Scalar(v) => flat_map(albedo.width(), albedo.height(), *v),
            ChannelSource::Auto => metallic_auto(&albedo),
            _ => metallic_from_albedo(&albedo),
        };
        Material { albedo, height, normal, roughness, metallic, ao }
    }

    /// Look up a channel by its canonical name as a savable `DynamicImage`.
    pub fn channel(&self, name: &str) -> Option<image::DynamicImage> {
        use image::DynamicImage::{ImageLuma8, ImageRgb8};
        Some(match name {
            "albedo" => ImageRgb8(self.albedo.clone()),
            "normal" => ImageRgb8(self.normal.clone()),
            "height" => ImageLuma8(self.height.clone()),
            "roughness" => ImageLuma8(self.roughness.clone()),
            "metallic" => ImageLuma8(self.metallic.clone()),
            "ao" => ImageLuma8(self.ao.clone()),
            _ => return None,
        })
    }

    /// Write the requested channel maps as PNGs into `dir` (the B1 basic writer; ORM/naming/glTF/preview
    /// are B2's `export`).
    pub fn write_channels(&self, dir: &std::path::Path, maps: &[String]) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        for m in maps {
            if let Some(img) = self.channel(m) {
                img.save(dir.join(format!("{m}.png")))?;
            }
        }
        Ok(())
    }
}

// --- primitives -----------------------------------------------------------------------------------

/// Rec.601 luma as a `[0,1]` sample.
fn luma01(p: &Rgb<u8>) -> f32 {
    (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0
}

fn flat_map(w: u32, h: u32, v: f32) -> GrayImage {
    GrayImage::from_pixel(w, h, Luma([(v.clamp(0.0, 1.0) * 255.0).round() as u8]))
}

/// A wrapped (circular) sample — the key to tileable derivation.
fn wrap(x: i32, n: u32) -> u32 {
    (((x % n as i32) + n as i32) % n as i32) as u32
}

/// Height from albedo: luminance, lightly circular-blurred + contrast-stretched (brighter = higher).
pub fn height_from_albedo(albedo: &RgbImage) -> GrayImage {
    let (w, h) = albedo.dimensions();
    let mut g = GrayImage::new(w, h);
    for (x, y, p) in albedo.enumerate_pixels() {
        g.put_pixel(x, y, Luma([(luma01(p) * 255.0).round() as u8]));
    }
    let g = circular_box_blur(&g, 1);
    autocontrast(&g)
}

/// Circular tangent-space normal from a height map (RFC §8, G0.4). `strength` scales the slope; `opengl`
/// = +Y (else DirectX -Y). Encoded to `[0,1]` RGB.
pub fn normal_from_height(height: &GrayImage, strength: f32, opengl: bool) -> RgbImage {
    let (w, h) = height.dimensions();
    let at = |x: i32, y: i32| height.get_pixel(wrap(x, w), wrap(y, h)).0[0] as f32 / 255.0;
    // Gain so a default strength=1 yields visible normals from typical texture contrast.
    let s = strength * 6.0;
    let mut out = RgbImage::new(w, h);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)
                - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1)) / 8.0;
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)
                - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1)) / 8.0;
            let ny = if opengl { -gy } else { gy };
            let (nx, ny, nz) = normalize(-gx * s, ny * s, 1.0);
            out.put_pixel(x as u32, y as u32, Rgb([enc(nx), enc(ny), enc(nz)]));
        }
    }
    out
}

/// Ambient occlusion from height: a cheap circular cavity — darker where the texel sits below its local
/// (circular-blurred) neighbourhood.
pub fn ao_from_height(height: &GrayImage, strength: f32) -> GrayImage {
    let blur = circular_box_blur(height, 4);
    let (w, h) = height.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let hv = height.get_pixel(x, y).0[0] as f32 / 255.0;
            let bv = blur.get_pixel(x, y).0[0] as f32 / 255.0;
            let occ = (bv - hv).max(0.0) * strength * 3.0; // below-neighbourhood → occluded
            let ao = (1.0 - occ).clamp(0.0, 1.0);
            out.put_pixel(x, y, Luma([(ao * 255.0).round() as u8]));
        }
    }
    out
}

/// Roughness from albedo (heuristic): darker + less saturated → rougher. `[0.2, 0.95]`.
pub fn roughness_from_albedo(albedo: &RgbImage) -> GrayImage {
    let (w, h) = albedo.dimensions();
    let mut out = GrayImage::new(w, h);
    for (x, y, p) in albedo.enumerate_pixels() {
        let l = luma01(p);
        let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let sat = if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
        let rough = (0.9 - 0.5 * l - 0.2 * sat).clamp(0.2, 0.95);
        out.put_pixel(x, y, Luma([(rough * 255.0).round() as u8]));
    }
    out
}

/// Metallic from albedo (conservative heuristic): only very desaturated *and* bright reads as metal;
/// most materials are dielectric.
pub fn metallic_from_albedo(albedo: &RgbImage) -> GrayImage {
    let (w, h) = albedo.dimensions();
    let mut out = GrayImage::new(w, h);
    for (x, y, p) in albedo.enumerate_pixels() {
        let l = luma01(p);
        let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let sat = if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
        let metal = if sat < 0.12 && l > 0.5 { ((l - 0.5) * 1.6).clamp(0.0, 1.0) } else { 0.0 };
        out.put_pixel(x, y, Luma([(metal * 255.0).round() as u8]));
    }
    out
}

/// Soft per-pixel metal-ness in `[0,1]` — low saturation AND bright → metal. Graded ramps (not a cliff)
/// so the region vote is smooth. The seed the `auto` region-vote smooths (G0.A).
fn metal_soft(albedo: &RgbImage) -> GrayImage {
    let (w, h) = albedo.dimensions();
    let mut out = GrayImage::new(w, h);
    for (x, y, p) in albedo.enumerate_pixels() {
        let l = luma01(p);
        let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let sat = if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
        // Saturation is the real metal↔dielectric separator in a flat albedo: raw metal is near-grey
        // (measured steel sat ≈ 0.01), while grey DIELECTRICS (stone, concrete) still carry sat ≈ 0.1–0.2.
        // So gate hard on VERY-low saturation; luma need only be mid (steel measured ≈ 0.56, not bright).
        // The region-vote then collapses any residual scattered grey speckle on a dielectric to nothing.
        let sat_term = ((0.05 - sat) / 0.03).clamp(0.0, 1.0); // 1 at sat≤0.02, 0 by sat 0.05
        let lum_term = ((l - 0.40) / 0.12).clamp(0.0, 1.0); // 1 at luma≥0.52, 0 below 0.40
        out.put_pixel(x, y, Luma([((sat_term * lum_term) * 255.0).round() as u8]));
    }
    out
}

/// The region-vote radius for a `w×h` image (G0.A used r=8 on 256px → ~size/32), clamped sane.
fn vote_radius(w: u32, h: u32) -> i32 {
    ((w.min(h) / 32) as i32).clamp(4, 48)
}

/// **Spatially-coherent metallic** (`metallic: "auto"`, 6.4.0 / G0.A). Soft per-pixel metal-ness →
/// **circular region-vote** (a separable circular box = isotropic majority) → threshold. Gives a
/// composite material (rusted iron, gilded frame, chipped paint) a clean, structured metal mask where
/// the per-pixel heuristic leaves speckle; the circular window keeps it tileable. On a single-class
/// material it correctly collapses to a flat mask (all-metal → white, all-dielectric → black).
pub fn metallic_auto(albedo: &RgbImage) -> GrayImage {
    let (w, h) = albedo.dimensions();
    let soft = metal_soft(albedo);
    let voted = circular_box_blur(&soft, vote_radius(w, h)); // majority over the neighbourhood
    let mut out = GrayImage::new(w, h);
    for (x, y, p) in voted.enumerate_pixels() {
        out.put_pixel(x, y, Luma([if p.0[0] >= 128 { 255 } else { 0 }])); // ≥50% → metal
    }
    out
}

/// **Spatially-coherent roughness** (`roughness: "auto"`, 6.4.0). The from-albedo per-pixel roughness,
/// region-smoothed (a smaller circular box) so distinct regions (wet/dry, polished/matte) read as
/// coherent patches rather than pixel noise, AND pulled smoother inside detected-metal regions (bare
/// metal is less rough than the surrounding dielectric). Circular → tileable.
pub fn roughness_auto(albedo: &RgbImage) -> GrayImage {
    let (w, h) = albedo.dimensions();
    let base = roughness_from_albedo(albedo);
    let coherent = circular_box_blur(&base, vote_radius(w, h) / 2); // region coherence
    let metal = metallic_auto(albedo);
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = coherent.get_pixel(x, y).0[0] as f32 / 255.0;
            let m = metal.get_pixel(x, y).0[0] as f32 / 255.0;
            let r = r * (1.0 - 0.45 * m); // metal regions read smoother
            out.put_pixel(x, y, Luma([(r * 255.0).round() as u8]));
        }
    }
    out
}

/// Weight-free **delighting** (RFC §9): divide out the low-frequency illumination so the albedo is
/// flat-lit (no baked gradient/shadow), preserving colour + detail. A circular low-pass keeps it
/// tileable. The texture-appropriate delight — IC-Light is subject-oriented (G0.3); the flat-lighting
/// prompt + this homomorphic flatten are the primary path.
pub fn flatten_lighting(albedo: &RgbImage) -> RgbImage {
    let (w, h) = albedo.dimensions();
    // low-frequency luminance (the baked lighting).
    let mut luma = GrayImage::new(w, h);
    for (x, y, p) in albedo.enumerate_pixels() {
        luma.put_pixel(x, y, Luma([(luma01(p) * 255.0).round() as u8]));
    }
    let r = ((w.min(h) / 12) as i32).max(2);
    let low = circular_box_blur(&luma, r);
    let target: f32 = low.pixels().map(|p| p.0[0] as f32).sum::<f32>() / (w * h) as f32 / 255.0;
    let mut out = RgbImage::new(w, h);
    for (x, y, p) in albedo.enumerate_pixels() {
        let l = (low.get_pixel(x, y).0[0] as f32 / 255.0).max(0.02);
        let s = (target / l).clamp(0.4, 2.5); // divide out the low-freq lighting
        out.put_pixel(x, y, Rgb([(p.0[0] as f32 * s).clamp(0.0, 255.0) as u8, (p.0[1] as f32 * s).clamp(0.0, 255.0) as u8, (p.0[2] as f32 * s).clamp(0.0, 255.0) as u8]));
    }
    out
}

fn normalize(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let m = (x * x + y * y + z * z).sqrt().max(1e-9);
    (x / m, y / m, z / m)
}

/// Encode a `[-1,1]` normal component to `[0,255]`.
fn enc(v: f32) -> u8 {
    ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

/// A separable **circular** box blur of radius `r` (wraps at the edges, so the blur is tileable).
fn circular_box_blur(g: &GrayImage, r: i32) -> GrayImage {
    let (w, h) = g.dimensions();
    let n = (2 * r + 1) as f32;
    // horizontal
    let mut tmp = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for dx in -r..=r {
                acc += g.get_pixel(wrap(x as i32 + dx, w), y).0[0] as f32;
            }
            tmp.put_pixel(x, y, Luma([(acc / n).round() as u8]));
        }
    }
    // vertical
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for dy in -r..=r {
                acc += tmp.get_pixel(x, wrap(y as i32 + dy, h)).0[0] as f32;
            }
            out.put_pixel(x, y, Luma([(acc / n).round() as u8]));
        }
    }
    out
}

/// 1st/99th-percentile contrast stretch (a faint height field → full range).
fn autocontrast(g: &GrayImage) -> GrayImage {
    let mut hist = [0u32; 256];
    for p in g.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    let n = (g.width() * g.height()) as f32;
    let clip = (n * 0.01) as u32;
    let (mut lo, mut acc) = (0usize, 0u32);
    while lo < 255 && acc < clip {
        acc += hist[lo];
        lo += 1;
    }
    let (mut hi, mut acc2) = (255usize, 0u32);
    while hi > 0 && acc2 < clip {
        acc2 += hist[hi];
        hi -= 1;
    }
    if hi <= lo + 4 {
        return g.clone();
    }
    let (lo, hi) = (lo as f32, hi as f32);
    let mut out = g.clone();
    for p in out.pixels_mut() {
        p.0[0] = (((p.0[0] as f32 - lo) / (hi - lo)) * 255.0).clamp(0.0, 255.0).round() as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_height_gives_a_flat_normal() {
        // A constant height → every normal points straight up → RGB (0.5,0.5,1.0).
        let h = GrayImage::from_pixel(16, 16, Luma([128]));
        let n = normal_from_height(&h, 1.0, true);
        for p in n.pixels() {
            // A constant height → straight-up normal → RGB ≈ (0.5,0.5,1.0). (The X/Y channels sit on the
            // 127/128 rounding boundary because the Sobel of a constant field is 0 ± fp-noise.)
            assert!((p.0[0] as i32 - 128).abs() <= 1 && (p.0[1] as i32 - 128).abs() <= 1 && p.0[2] == 255, "flat normal ≈ (0.5,0.5,1.0), got {:?}", p.0);
        }
    }

    #[test]
    fn a_ramp_normal_tilts_the_right_way_and_flips_for_directx() {
        // A height ramp increasing in +x: the surface normal tilts toward -x (nx < 0.5 in RGB).
        let mut h = GrayImage::new(32, 8);
        for y in 0..8 {
            for x in 0..32 {
                h.put_pixel(x, y, Luma([(x as f32 / 31.0 * 255.0) as u8]));
            }
        }
        let ogl = normal_from_height(&h, 1.0, true);
        let mid = ogl.get_pixel(16, 4).0;
        assert!(mid[0] < 120, "normal should tilt -x (R<0.5): {mid:?}");
        // DirectX flips green; a +x ramp has ~zero y-slope so green stays ~128 either way — assert the
        // convention plumbs through on a y-ramp instead.
        let mut hy = GrayImage::new(8, 32);
        for y in 0..32 {
            for x in 0..8 {
                hy.put_pixel(x, y, Luma([(y as f32 / 31.0 * 255.0) as u8]));
            }
        }
        let g_ogl = normal_from_height(&hy, 1.0, true).get_pixel(4, 16).0[1];
        let g_dx = normal_from_height(&hy, 1.0, false).get_pixel(4, 16).0[1];
        assert!((g_ogl as i32 - 128).signum() != (g_dx as i32 - 128).signum(), "OpenGL/DirectX Y must flip");
    }

    #[test]
    fn derived_maps_are_deterministic_and_sized() {
        let albedo = RgbImage::from_fn(24, 24, |x, y| Rgb([(x * 8) as u8, (y * 8) as u8, 100]));
        let a = Material::derive(albedo.clone(), None, 1.0, true, 1.0, &ChannelSource::FromAlbedo, &ChannelSource::Scalar(0.0));
        let b = Material::derive(albedo, None, 1.0, true, 1.0, &ChannelSource::FromAlbedo, &ChannelSource::Scalar(0.0));
        assert_eq!(a.normal.as_raw(), b.normal.as_raw(), "deterministic");
        assert_eq!(a.normal.dimensions(), (24, 24));
        assert_eq!(a.metallic.get_pixel(0, 0).0[0], 0, "scalar-0 metallic is a flat 0 map");
        assert!(a.roughness.pixels().any(|p| p.0[0] != a.roughness.get_pixel(0, 0).0[0]), "from-albedo roughness varies");
    }

    #[test]
    fn metallic_auto_is_region_coherent_and_tiles() {
        // Half bare steel (bright, grey), half rust (saturated orange). `auto` should give a clean
        // two-region mask — steel white, rust black — not per-pixel speckle. Circular window → tiles.
        let a = RgbImage::from_fn(64, 64, |x, _| {
            if x < 32 { Rgb([180, 182, 185]) } else { Rgb([150, 70, 30]) }
        });
        let m = Material::derive(a, None, 1.0, true, 1.0, &ChannelSource::Auto, &ChannelSource::Auto);
        // steel side → metal (white), rust side → dielectric (black), away from the boundary.
        assert!(m.metallic.get_pixel(8, 32).0[0] > 200, "bare-steel region should read metal");
        assert_eq!(m.metallic.get_pixel(56, 32).0[0], 0, "rust region should read dielectric");
        // binary mask (region-vote thresholded): only 0 or 255.
        assert!(m.metallic.pixels().all(|p| p.0[0] == 0 || p.0[0] == 255), "auto metallic is a clean mask");
        // roughness auto: metal side smoother than a naive from-albedo would give (pulled down).
        assert!(m.roughness.get_pixel(8, 32).0[0] < m.roughness.get_pixel(56, 32).0[0], "metal reads smoother");
    }

    #[test]
    fn metallic_auto_collapses_flat_for_a_single_class_material() {
        // An all-dielectric (saturated) albedo → auto metallic collapses to a flat black mask (correct).
        let a = RgbImage::from_fn(48, 48, |x, y| Rgb([40 + (x % 8) as u8 * 4, 120 + (y % 6) as u8 * 3, 30]));
        let m = Material::derive(a, None, 1.0, true, 1.0, &ChannelSource::FromAlbedo, &ChannelSource::Auto);
        assert!(m.metallic.pixels().all(|p| p.0[0] == 0), "a single-class dielectric → flat black metallic");
    }
}
