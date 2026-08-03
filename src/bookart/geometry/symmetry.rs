//! The symmetry engine (RFC BOOKART-1 §6.3) — a *geometric guarantee* diffusion cannot hold. Given a
//! finished ornament (transparent RGBA) and a symmetry spec, it enforces the symmetry by replicating
//! across the fundamental domain: **bilateral** (mirror about the vertical mid-axis) unions the two
//! halves; **radial:N** unions the N rotations about the centre; **frieze**/**none** pass through.
//!
//! "Union" = per output pixel, keep the highest-alpha contributor (ink from either side), so the result
//! is exactly symmetric (`α(x) == α(w−1−x)` for bilateral) while preserving the strongest linework. This
//! is applied deliberately by the render path (not silently in the finisher) so the scorecard can still
//! *measure* whether symmetry was achieved.

use image::{Rgba, RgbaImage};

/// Apply the symmetry named by a spec string (`bilateral` | `radial:N` | `frieze:…` | `none`).
pub fn symmetrize(img: &RgbaImage, symmetry: &str) -> RgbaImage {
    match symmetry.split(':').next().unwrap_or(symmetry) {
        "bilateral" => bilateral(img),
        "radial" => {
            let n = symmetry.split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
            radial(img, n)
        }
        _ => img.clone(), // frieze (→ border assembly, B3) / none
    }
}

/// Mirror about the vertical mid-axis, unioning ink.
fn bilateral(img: &RgbaImage) -> RgbaImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let a = img.get_pixel(x, y).0;
            let b = img.get_pixel(w - 1 - x, y).0;
            out.put_pixel(x, y, if a[3] >= b[3] { Rgba(a) } else { Rgba(b) });
        }
    }
    out
}

/// Bilinear RGBA sample; out-of-bounds → transparent.
fn sample(img: &RgbaImage, fx: f32, fy: f32) -> [f32; 4] {
    let (w, h) = (img.width() as i32, img.height() as i32);
    if fx < 0.0 || fy < 0.0 || fx > (w - 1) as f32 || fy > (h - 1) as f32 {
        return [0.0; 4];
    }
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let g = |x: i32, y: i32| img.get_pixel(x as u32, y as u32).0;
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let mut out = [0f32; 4];
    for c in 0..4 {
        let top = lerp(g(x0, y0)[c] as f32, g(x1, y0)[c] as f32, tx);
        let bot = lerp(g(x0, y1)[c] as f32, g(x1, y1)[c] as f32, tx);
        out[c] = lerp(top, bot, ty);
    }
    out
}

/// N-fold rotational union about the centre.
fn radial(img: &RgbaImage, n: u32) -> RgbaImage {
    let n = n.clamp(1, 24);
    let (w, h) = (img.width(), img.height());
    let (cx, cy) = ((w as f32 - 1.0) / 2.0, (h as f32 - 1.0) / 2.0);
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let mut best = [0f32; 4];
            for k in 0..n {
                let ang = -(k as f32) * std::f32::consts::TAU / n as f32;
                let (c, s) = (ang.cos(), ang.sin());
                let p = sample(img, cx + dx * c - dy * s, cy + dx * s + dy * c);
                if p[3] >= best[3] {
                    best = p;
                }
            }
            out.put_pixel(x, y, Rgba([best[0] as u8, best[1] as u8, best[2] as u8, best[3] as u8]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asym() -> RgbaImage {
        // ink only on the left third → strongly asymmetric
        let mut img = RgbaImage::from_pixel(32, 32, Rgba([0, 0, 0, 0]));
        for y in 4..28 {
            for x in 2..10 {
                img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        img
    }

    fn bilateral_rms(img: &RgbaImage) -> f32 {
        let (w, h) = (img.width(), img.height());
        let (mut acc, mut n) = (0.0f64, 0u64);
        for y in 0..h {
            for x in 0..w / 2 {
                let d = (img.get_pixel(x, y).0[3] as f64 - img.get_pixel(w - 1 - x, y).0[3] as f64) / 255.0;
                acc += d * d;
                n += 1;
            }
        }
        (acc / n.max(1) as f64).sqrt() as f32
    }

    #[test]
    fn bilateral_makes_it_exactly_symmetric() {
        let before = bilateral_rms(&asym());
        let after = bilateral_rms(&symmetrize(&asym(), "bilateral"));
        assert!(before > 0.4, "control should be asymmetric ({before})");
        assert_eq!(after, 0.0, "bilateral must be exact");
    }

    #[test]
    fn radial_unions_all_folds() {
        let sym = symmetrize(&asym(), "radial:4");
        // the left-ink blob is now replicated to (at least) the 3 other quadrants → far more ink.
        let ink0 = asym().pixels().filter(|p| p.0[3] > 128).count();
        let ink4 = sym.pixels().filter(|p| p.0[3] > 128).count();
        assert!(ink4 > ink0 * 2, "radial:4 should replicate the fundamental domain ({ink0} -> {ink4})");
    }

    #[test]
    fn none_is_passthrough() {
        assert_eq!(symmetrize(&asym(), "none").as_raw(), asym().as_raw());
    }
}
