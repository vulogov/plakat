//! The seamless engine primitives (RFC TEXTURE-1 §7, ROADMAP B3). The building blocks that make a
//! generation tileable, kept **self-contained + tested** here so B4 can apply them in the render loop
//! without touching the shared, corr-1.0 generation stack:
//!
//!   * [`circular_pad2d`] — wrap-pad the last two dims (the tileable-conv primitive, from the G0.1 probe).
//!   * [`roll2d`] — a circular shift of a latent; per-step rolling spreads the zero-pad conv seam across
//!     the tile so no fixed location is ever the boundary (tileable diffusion without conv surgery).
//!   * [`SeamlessConv2d`] — a circular-padded convolution, for the eventual vendored circular ResNet
//!     (the G0.1 finding) should measurement (G0.2) show per-step rolling leaves too large a residual.
//!   * [`feather_seam`] — a weight-free hairline blend across the wrap boundary (the G0.2 seam-repair
//!     fallback for any residual VAE-decode seam).
//!
//! Which axes tile is carried by [`Axes`]; a trim sheet tiles one axis.

use candle_core::{Module, Result, Tensor, D};
use image::RgbImage;

/// Which axes should tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axes {
    Both,
    X,
    Y,
}

impl Axes {
    pub fn parse(s: &str) -> Axes {
        match s.to_ascii_lowercase().as_str() {
            "x" => Axes::X,
            "y" => Axes::Y,
            _ => Axes::Both,
        }
    }
    fn pads(self, p: usize) -> (usize, usize) {
        match self {
            Axes::Both => (p, p),
            Axes::X => (p, 0),
            Axes::Y => (0, p),
        }
    }
}

/// Circular ("wrap") pad the last two dims by `(px, py)` (width, height). candle's `Conv2d` only
/// zero-pads, so this wraps the input and the caller convolves with `padding: 0`.
pub fn circular_pad2d(t: &Tensor, px: usize, py: usize) -> Result<Tensor> {
    let mut t = t.clone();
    if px > 0 {
        let w = t.dim(D::Minus1)?;
        let left = t.narrow(D::Minus1, w - px, px)?;
        let right = t.narrow(D::Minus1, 0, px)?;
        t = Tensor::cat(&[&left, &t, &right], D::Minus1)?;
    }
    if py > 0 {
        let h = t.dim(D::Minus2)?;
        let top = t.narrow(D::Minus2, h - py, py)?;
        let bot = t.narrow(D::Minus2, 0, py)?;
        t = Tensor::cat(&[&top, &t, &bot], D::Minus2)?;
    }
    Ok(t)
}

/// Circular shift of the last two dims by `(dx, dy)` — `out[.., y, x] = in[.., y-dy, x-dx]` (wrapping).
/// Rolling the latent by a different offset each denoise step spreads the boundary so the zero-pad
/// convs never persistently see the tile edge → the decode tiles, with **no** change to the model.
pub fn roll2d(t: &Tensor, dx: i64, dy: i64) -> Result<Tensor> {
    let t = roll_dim(t, D::Minus1, dx)?;
    roll_dim(&t, D::Minus2, dy)
}

fn roll_dim(t: &Tensor, dim: D, shift: i64) -> Result<Tensor> {
    let n = t.dim(dim)? as i64;
    if n == 0 {
        return Ok(t.clone());
    }
    let s = ((shift % n) + n) % n; // normalised right-shift in [0, n)
    if s == 0 {
        return Ok(t.clone());
    }
    let s = s as usize;
    let n = n as usize;
    // out = cat([ last s , first n-s ])
    let tail = t.narrow(dim, n - s, s)?;
    let head = t.narrow(dim, 0, n - s)?;
    Tensor::cat(&[&tail, &head], dim)
}

/// A circular-padded 2-D convolution: wrap-pad by the kernel's implied padding, then conv with
/// `padding: 0`. For the vendored circular ResNet (B3 escalation path, per G0.1).
pub struct SeamlessConv2d {
    inner: candle_nn::Conv2d,
    padding: usize,
    axes: Axes,
}

impl SeamlessConv2d {
    /// Wrap a `candle_nn::Conv2d` whose config was built with `padding: 0`. `padding` is the wrap amount
    /// (usually `kernel/2`).
    pub fn new(inner: candle_nn::Conv2d, padding: usize, axes: Axes) -> Self {
        Self { inner, padding, axes }
    }
}

impl Module for SeamlessConv2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (px, py) = self.axes.pads(self.padding);
        let xs = circular_pad2d(xs, px, py)?;
        self.inner.forward(&xs)
    }
}

/// Smoothstep (C¹) ease `3t²−2t³` on a clamped `t∈[0,1]` — a softer ramp than linear, so a feathered
/// band has no hard edges where it starts/stops. (RFC SEAMS-1 P1.)
#[inline]
pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// **Seam score** (measure-first, G0.2/SEAMS-1): mean per-channel intensity jump *across* the wrap
/// boundary, normalised by the interior step baseline. `≈1` ⇒ the boundary is as smooth as the interior
/// (seamless); larger ⇒ a visible seam. Pure — drives auto-band selection and the P1 tests.
pub fn seam_score(img: &RgbImage, axes: Axes) -> f32 {
    let (w, h) = img.dimensions();
    let (mut cross, mut base, mut nc, mut nb) = (0.0f64, 0.0f64, 0u64, 0u64);
    let diff = |a: &image::Rgb<u8>, b: &image::Rgb<u8>| (0..3).map(|c| (a.0[c] as f64 - b.0[c] as f64).abs()).sum::<f64>();
    if matches!(axes, Axes::Both | Axes::X) && w >= 3 {
        for y in 0..h {
            cross += diff(img.get_pixel(0, y), img.get_pixel(w - 1, y)); // across the wrap
            base += diff(img.get_pixel(0, y), img.get_pixel(1, y)); // interior baseline
            nc += 1;
            nb += 1;
        }
    }
    if matches!(axes, Axes::Both | Axes::Y) && h >= 3 {
        for x in 0..w {
            cross += diff(img.get_pixel(x, 0), img.get_pixel(x, h - 1));
            base += diff(img.get_pixel(x, 0), img.get_pixel(x, 1));
            nc += 1;
            nb += 1;
        }
    }
    if nc == 0 {
        return 0.0;
    }
    let cross = cross / nc as f64;
    let base = (base / nb.max(1) as f64).max(1.0); // guard a flat image
    (cross / base) as f32
}

/// **Frequency-aware** hairline seam repair across the wrap boundary (RFC SEAMS-1 P1). Instead of
/// cross-fading raw pixels (which blurs detail and can leave a tonal ramp), estimate the *low-frequency*
/// tone on each edge (a short `k`-px inward mean) and add a **smooth, half-magnitude offset** that decays
/// (smoothstep) inward over `band` px so the two edges' tone **meets** at the seam — while every high
/// frequency (the actual texture detail) is preserved untouched. `band` px each side; `axes` picks the
/// boundary.
pub fn feather_seam(img: &RgbImage, band: u32, axes: Axes) -> RgbImage {
    let (w, h) = img.dimensions();
    let mut out = img.clone();
    if band == 0 {
        return out;
    }
    let k = (band / 2).clamp(1, 8); // low-freq tone window: match tone, not detail
    if matches!(axes, Axes::Both | Axes::X) && w > 2 * band {
        for y in 0..h {
            let (mut el, mut er) = ([0f32; 3], [0f32; 3]);
            for i in 0..k {
                let (l, r) = (img.get_pixel(i, y), img.get_pixel(w - 1 - i, y));
                for c in 0..3 {
                    el[c] += l.0[c] as f32;
                    er[c] += r.0[c] as f32;
                }
            }
            let d = [0, 1, 2].map(|c| (el[c] - er[c]) / k as f32 * 0.5); // half each side → they meet
            for x in 0..band {
                let ramp = smoothstep(1.0 - (x as f32 + 0.5) / band as f32); // 1 at seam → 0 inward
                let pl = out.get_pixel_mut(x, y);
                for c in 0..3 {
                    pl.0[c] = (pl.0[c] as f32 - d[c] * ramp).clamp(0.0, 255.0) as u8;
                }
                let pr = out.get_pixel_mut(w - 1 - x, y);
                for c in 0..3 {
                    pr.0[c] = (pr.0[c] as f32 + d[c] * ramp).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    if matches!(axes, Axes::Both | Axes::Y) && h > 2 * band {
        for x in 0..w {
            let (mut et, mut eb) = ([0f32; 3], [0f32; 3]);
            for i in 0..k {
                let (tp, bt) = (out.get_pixel(x, i), out.get_pixel(x, h - 1 - i));
                for c in 0..3 {
                    et[c] += tp.0[c] as f32;
                    eb[c] += bt.0[c] as f32;
                }
            }
            let d = [0, 1, 2].map(|c| (et[c] - eb[c]) / k as f32 * 0.5);
            for y in 0..band {
                let ramp = smoothstep(1.0 - (y as f32 + 0.5) / band as f32);
                let ptp = out.get_pixel_mut(x, y);
                for c in 0..3 {
                    ptp.0[c] = (ptp.0[c] as f32 - d[c] * ramp).clamp(0.0, 255.0) as u8;
                }
                let pbt = out.get_pixel_mut(x, h - 1 - y);
                for c in 0..3 {
                    pbt.0[c] = (pbt.0[c] as f32 + d[c] * ramp).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    out
}

/// **Mirror-tile** (RFC SEAMS-1 P1, `mode: "mirror"`): reflect the image so opposite edges are identical
/// by construction — the boundary is guaranteed seamless (`seam_score ≈ 1`) with zero blending. Reads
/// better than a wrap for many organic/fabric textures (at the cost of a mirror-symmetric pattern). Halves
/// then reflects along each selected axis, so the result is the same size and tiles exactly.
pub fn mirror_tile(img: &RgbImage, axes: Axes) -> RgbImage {
    let (w, h) = img.dimensions();
    RgbImage::from_fn(w, h, |x, y| {
        // fold the second half back onto a mirror of the first — edges meet their own reflection.
        let sx = if matches!(axes, Axes::Both | Axes::X) && x >= w - x - 1 { w - 1 - x } else { x };
        let sy = if matches!(axes, Axes::Both | Axes::Y) && y >= h - y - 1 { h - 1 - y } else { y };
        *img.get_pixel(sx.min(w - 1), sy.min(h - 1))
    })
}

/// Make a **non-tileable photo** tileable (image-to-material, B6): the classic *offset-and-heal*. Roll
/// by half so the discontinuous edges move to the interior — the boundary is now the photo's continuous
/// centre, so it tiles — then feather the resulting central cross (narrow band, strongest at the seam)
/// to hide the moved discontinuity while preserving most texture.
pub fn make_tileable(img: &RgbImage, band: u32, axes: Axes) -> RgbImage {
    let (w, h) = img.dimensions();
    let (cx, cy) = (w / 2, h / 2);
    // roll by half → the discontinuous outer edges move to the interior central cross.
    let mut out = RgbImage::from_fn(w, h, |x, y| *img.get_pixel((x + cx) % w, (y + cy) % h));
    let src = out.clone();
    let k = (band / 2).clamp(1, 8) as i64; // low-freq tone window either side of the central seam
    // Frequency-aware heal (RFC SEAMS-1 P1): match the low-frequency tone across the moved seam with a
    // smoothstep offset decaying outward, preserving the photo's high-frequency texture.
    if matches!(axes, Axes::Both | Axes::X) && band > 0 {
        for y in 0..h {
            let (mut el, mut er) = ([0f32; 3], [0f32; 3]);
            for i in 0..k {
                let l = src.get_pixel((cx as i64 - 1 - i).rem_euclid(w as i64) as u32, y);
                let r = src.get_pixel((cx as i64 + i).rem_euclid(w as i64) as u32, y);
                for c in 0..3 {
                    el[c] += l.0[c] as f32;
                    er[c] += r.0[c] as f32;
                }
            }
            let d = [0, 1, 2].map(|c| (el[c] - er[c]) / k as f32 * 0.5);
            for dx in 1..=band.min(cx) {
                let ramp = smoothstep(1.0 - (dx as f32 - 0.5) / band as f32); // 1 at seam → 0 outward
                let pl = out.get_pixel_mut(cx - dx, y);
                for c in 0..3 {
                    pl.0[c] = (src.get_pixel(cx - dx, y).0[c] as f32 + d[c] * ramp).clamp(0.0, 255.0) as u8;
                }
                if cx + dx < w {
                    let pr = out.get_pixel_mut(cx + dx, y);
                    for c in 0..3 {
                        pr.0[c] = (src.get_pixel(cx + dx, y).0[c] as f32 - d[c] * ramp).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    let after_x = out.clone();
    if matches!(axes, Axes::Both | Axes::Y) && band > 0 {
        for x in 0..w {
            let (mut et, mut eb) = ([0f32; 3], [0f32; 3]);
            for i in 0..k {
                let tp = after_x.get_pixel(x, (cy as i64 - 1 - i).rem_euclid(h as i64) as u32);
                let bt = after_x.get_pixel(x, (cy as i64 + i).rem_euclid(h as i64) as u32);
                for c in 0..3 {
                    et[c] += tp.0[c] as f32;
                    eb[c] += bt.0[c] as f32;
                }
            }
            let d = [0, 1, 2].map(|c| (et[c] - eb[c]) / k as f32 * 0.5);
            for dy in 1..=band.min(cy) {
                let ramp = smoothstep(1.0 - (dy as f32 - 0.5) / band as f32);
                let ptp = out.get_pixel_mut(x, cy - dy);
                for c in 0..3 {
                    ptp.0[c] = (after_x.get_pixel(x, cy - dy).0[c] as f32 + d[c] * ramp).clamp(0.0, 255.0) as u8;
                }
                if cy + dy < h {
                    let pbt = out.get_pixel_mut(x, cy + dy);
                    for c in 0..3 {
                        pbt.0[c] = (after_x.get_pixel(x, cy + dy).0[c] as f32 - d[c] * ramp).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn circular_pad_wraps_exactly() {
        let cpu = Device::Cpu;
        let t = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 2, 2), &cpu).unwrap();
        let p = circular_pad2d(&t, 1, 1).unwrap();
        assert_eq!(p.dims4().unwrap(), (1, 1, 4, 4));
        let v = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![4., 3., 4., 3., 2., 1., 2., 1., 4., 3., 4., 3., 2., 1., 2., 1.]);
    }

    #[test]
    fn per_axis_pad() {
        let cpu = Device::Cpu;
        let t = Tensor::ones((1, 1, 4, 6), candle_core::DType::F32, &cpu).unwrap();
        let (px, py) = Axes::X.pads(2);
        let p = circular_pad2d(&t, px, py).unwrap();
        assert_eq!(p.dims4().unwrap(), (1, 1, 4, 10), "X-only pads width, not height");
    }

    #[test]
    fn roll_is_circular_and_invertible() {
        let cpu = Device::Cpu;
        // 1×1×1×4 = [0,1,2,3]; roll dx=1 → [3,0,1,2]; roll back dx=-1 → original.
        let t = Tensor::from_vec(vec![0f32, 1., 2., 3.], (1, 1, 1, 4), &cpu).unwrap();
        let r = roll2d(&t, 1, 0).unwrap();
        assert_eq!(r.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![3., 0., 1., 2.]);
        let back = roll2d(&r, -1, 0).unwrap();
        assert_eq!(back.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![0., 1., 2., 3.]);
    }

    #[test]
    fn make_tileable_kills_a_gradient_seam() {
        // A left→right BRIGHTNESS RAMP has a hard wrap seam (col 0 black vs col w-1 white). Boundary
        // feathering can't fix an interior-driven ramp; offset-and-heal does.
        let img = RgbImage::from_fn(64, 16, |x, _| {
            let v = (x as f32 / 63.0 * 255.0) as u8;
            image::Rgb([v, v, v])
        });
        let seam = |im: &RgbImage| -> i32 {
            let (w, h) = im.dimensions();
            (0..h).map(|y| (im.get_pixel(0, y).0[0] as i32 - im.get_pixel(w - 1, y).0[0] as i32).abs()).sum()
        };
        let before = seam(&img);
        let after = seam(&make_tileable(&img, 8, Axes::X));
        assert!(after * 4 < before, "offset-and-heal should crush the ramp seam ({before} → {after})");
    }

    #[test]
    fn feather_reduces_a_seam() {
        // A left-dark / right-bright image has a hard vertical wrap seam; feathering shrinks it.
        let img = RgbImage::from_fn(32, 8, |x, _| if x < 16 { image::Rgb([10, 10, 10]) } else { image::Rgb([240, 240, 240]) });
        let seam = |im: &RgbImage| -> i32 {
            let (w, h) = im.dimensions();
            (0..h).map(|y| (im.get_pixel(0, y).0[0] as i32 - im.get_pixel(w - 1, y).0[0] as i32).abs()).sum()
        };
        let before = seam(&img);
        let after = seam(&feather_seam(&img, 4, Axes::X));
        assert!(after < before, "feather should reduce the wrap seam ({before} → {after})");
    }

    #[test]
    fn smoothstep_is_eased_and_monotonic() {
        assert_eq!(smoothstep(0.0), 0.0);
        assert_eq!(smoothstep(1.0), 1.0);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-6, "symmetric midpoint");
        let mut prev = -1.0;
        for i in 0..=20 {
            let v = smoothstep(i as f32 / 20.0);
            assert!(v >= prev, "monotonic non-decreasing");
            prev = v;
        }
        assert_eq!(smoothstep(-3.0), 0.0, "clamps below");
        assert_eq!(smoothstep(3.0), 1.0, "clamps above");
    }

    #[test]
    fn seam_score_measures_and_feather_lowers_it() {
        // Left-dark / right-bright → a big wrap seam; the frequency-aware feather must lower seam_score.
        let img = RgbImage::from_fn(48, 12, |x, _| if x < 24 { image::Rgb([20, 40, 60]) } else { image::Rgb([210, 190, 170]) });
        let raw = seam_score(&img, Axes::X);
        let fixed = seam_score(&feather_seam(&img, 8, Axes::X), Axes::X);
        assert!(raw > 3.0, "raw image has a visible seam (score {raw:.2})");
        assert!(fixed < raw, "feather lowers the seam score ({raw:.2} → {fixed:.2})");
    }

    #[test]
    fn mirror_tile_is_exactly_seamless() {
        // An asymmetric gradient: mirror-tiling makes opposite edges identical by construction.
        let img = RgbImage::from_fn(40, 16, |x, y| image::Rgb([(x * 5) as u8, (y * 9) as u8, 90]));
        let m = mirror_tile(&img, Axes::Both);
        let (w, h) = m.dimensions();
        for y in 0..h {
            assert_eq!(m.get_pixel(0, y), m.get_pixel(w - 1, y), "left edge == right edge");
        }
        for x in 0..w {
            assert_eq!(m.get_pixel(x, 0), m.get_pixel(x, h - 1), "top edge == bottom edge");
        }
        assert!(seam_score(&m, Axes::Both) < 1.5, "mirror boundary is as smooth as the interior");
    }

    #[test]
    fn feather_preserves_interior_high_frequency() {
        // A high-frequency checker away from the seam must be untouched (freq-aware = tone-only offset).
        let img = RgbImage::from_fn(64, 8, |x, y| { let v = if (x + y) % 2 == 0 { 200 } else { 40 }; image::Rgb([v, v, v]) });
        let out = feather_seam(&img, 6, Axes::X);
        // column at the centre (far from either wrap edge) is unchanged.
        for y in 0..8 {
            assert_eq!(out.get_pixel(32, y), img.get_pixel(32, y), "interior detail preserved");
        }
    }
}
