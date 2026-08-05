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

/// A weight-free hairline blend across the wrap boundary (columns/rows near the seam) — cleans a small
/// residual VAE-decode seam (G0.2 fallback). `band` px each side; `axes` selects which boundary.
pub fn feather_seam(img: &RgbImage, band: u32, axes: Axes) -> RgbImage {
    let (w, h) = img.dimensions();
    let mut out = img.clone();
    let blend = |a: u8, b: u8, t: f32| ((a as f32) * (1.0 - t) + (b as f32) * t).round() as u8;
    if matches!(axes, Axes::Both | Axes::X) && w > 2 * band {
        // cross-fade the first/last `band` columns toward their wrapped partner so col 0 ≈ col w-1.
        for x in 0..band {
            let t = 0.5 * (1.0 - (x as f32 + 0.5) / band as f32); // 0.5 at the seam → 0 inward
            for y in 0..h {
                let (l, r) = (*img.get_pixel(x, y), *img.get_pixel(w - 1 - x, y));
                let pl = out.get_pixel_mut(x, y);
                for c in 0..3 {
                    pl.0[c] = blend(l.0[c], r.0[c], t);
                }
                let pr = out.get_pixel_mut(w - 1 - x, y);
                for c in 0..3 {
                    pr.0[c] = blend(r.0[c], l.0[c], t);
                }
            }
        }
    }
    if matches!(axes, Axes::Both | Axes::Y) && h > 2 * band {
        for y in 0..band {
            let t = 0.5 * (1.0 - (y as f32 + 0.5) / band as f32);
            for x in 0..w {
                let (tp, bt) = (*out.get_pixel(x, y), *out.get_pixel(x, h - 1 - y));
                let ptp = out.get_pixel_mut(x, y);
                for c in 0..3 {
                    ptp.0[c] = blend(tp.0[c], bt.0[c], t);
                }
                let pbt = out.get_pixel_mut(x, h - 1 - y);
                for c in 0..3 {
                    pbt.0[c] = blend(bt.0[c], tp.0[c], t);
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
}
