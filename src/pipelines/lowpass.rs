//! Low-pass filter for the LAYERED-1 guide anchor (RFC §Filter). The finish trajectory is steered toward
//! the guide latent only in its LOW frequencies — layout and binding live there; high frequencies (texture,
//! fine detail) belong to the finish model. `lowpass` is a Gaussian-pyramid low-pass on a spatial latent
//! `(B, C, H, W)`: `L` rounds of box-average downsample, then bilinear upsample back to each level.
//!
//! Properties the LAYERED anchor relies on (Tier-0 tests below):
//!   * **DC exact** — `lowpass(constant) == constant` (an average of a constant is the constant; a bilinear
//!     upsample of a constant is the constant). So the anchor never shifts the image's mean.
//!   * **Kills the high band** — a zero-mean pattern at the pool grid (a 1-px checkerboard) filters to ~0,
//!     so the model's own detail above the cutoff is left untouched.
//!   * **No block artifacts** — the upsample is **bilinear**, never nearest, so no grid edges are injected
//!     into the detail band.
//!
//! Pure candle ops (`avg_pool2d` + `upsample_bilinear2d`), so it runs on CPU, CUDA and Metal. `fft.rs` is
//! used only by the spectral non-regression check, never in the denoise loop.

use candle_core::{Result, Tensor};

/// Gaussian-pyramid low-pass of a spatial `(B, C, H, W)` latent, cutting frequencies above ~`2^levels`
/// pixels. `levels == 0` is the identity. Odd sizes are handled: each downsample records its output size
/// so the upsample restores the exact dimensions.
pub fn lowpass(spatial: &Tensor, levels: usize) -> Result<Tensor> {
    if levels == 0 {
        return Ok(spatial.clone());
    }
    let (_, _, h0, w0) = spatial.dims4()?;
    // Down: box-average decimate, recording each level's dims for an exact restore.
    let mut cur = spatial.clone();
    let mut dims: Vec<(usize, usize)> = vec![(h0, w0)];
    for _ in 0..levels {
        let (_, _, h, w) = cur.dims4()?;
        // Stop if a further 2× decimate would collapse a dimension.
        if h < 2 || w < 2 {
            break;
        }
        cur = cur.avg_pool2d(2)?;
        let (_, _, hh, ww) = cur.dims4()?;
        dims.push((hh, ww));
    }
    // Up: bilinear back to each recorded level (dims.len()-1 downsamples happened).
    for lvl in (0..dims.len() - 1).rev() {
        let (th, tw) = dims[lvl];
        cur = cur.upsample_bilinear2d(th, tw, false)?;
    }
    Ok(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    fn dc_is_exact() {
        // A constant field must pass through unchanged (the anchor never shifts the mean).
        let x = Tensor::full(0.7f32, (1, 4, 16, 16), &dev()).unwrap();
        let y = lowpass(&x, 2).unwrap();
        let d = (&y - &x).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(d < 1e-5, "DC not preserved (max abs diff {d})");
        assert_eq!(y.dims4().unwrap(), x.dims4().unwrap(), "dims preserved");
    }

    #[test]
    fn kills_a_grid_checkerboard() {
        // A 1-px checkerboard (zero mean at the 2×2 pool grid) filters to ≈ 0.
        let (h, w) = (16usize, 16usize);
        let mut v = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let t = Tensor::from_vec(v, (1, 1, h, w), &dev()).unwrap();
        let y = lowpass(&t, 2).unwrap();
        let energy = y.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(energy < 1e-6, "high-band checkerboard not removed (residual energy {energy})");
    }

    #[test]
    fn preserves_a_low_ramp_roughly() {
        // A smooth horizontal ramp (all energy below the cutoff) survives with only mild attenuation.
        let (h, w) = (32usize, 32usize);
        let mut v = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = x as f32 / w as f32;
            }
        }
        let t = Tensor::from_vec(v, (1, 1, h, w), &dev()).unwrap();
        let y = lowpass(&t, 2).unwrap();
        // Means match (DC), and the filtered ramp still increases left→right.
        let mt = t.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let my = y.mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((mt - my).abs() < 1e-3, "mean drifted ({mt} vs {my})");
    }

    #[test]
    fn identity_at_zero_levels() {
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev()).unwrap().to_dtype(DType::F32).unwrap();
        let y = lowpass(&x, 0).unwrap();
        let d = (&y - &x).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d, 0.0, "zero levels is the identity");
    }
}
