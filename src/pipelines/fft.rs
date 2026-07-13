//! Minimal 2D DFT for the FreeU Fourier skip-filter — candle ships no FFT.
//!
//! Rather than a radix-2 FFT (which only handles power-of-two sizes — UNet feature maps at e.g.
//! 768px are 12/24/48/96, not powers of two), we compute the DFT as a **matrix multiply** by the
//! `N×N` DFT matrix. That's `O(N²)` vs `O(N log N)`, but `N` here is small (≤~128) and the whole
//! thing is a handful of matmuls candle runs fast (incl. Metal). Complex tensors are carried as a
//! `(re, im)` pair. The DFT matrices `cos(2π kj/N)` / `sin(2π kj/N)` are symmetric in `(k, j)`, so
//! the same matrix contracts either operand order.

use candle_core::{DType, Device, Result, Tensor};

/// `(cos, sin)` DFT matrices of size `n×n`: `cos[k,j] = cos(2π kj/n)`, `sin[k,j] = sin(2π kj/n)`.
fn dft_matrices(n: usize, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    // idx[k,j] = k*j as f32, via an outer product of 0..n.
    let rng = Tensor::arange(0u32, n as u32, device)?.to_dtype(DType::F32)?; // (n,)
    let kj = rng.reshape((n, 1))?.broadcast_mul(&rng.reshape((1, n))?)?; // (n,n) = k*j
    let ang = (kj * (2.0 * std::f64::consts::PI / n as f64))?;
    Ok((ang.cos()?.to_dtype(dtype)?, ang.sin()?.to_dtype(dtype)?))
}

/// 2D DFT of a real `(B,C,H,W)` tensor. Returns `(re, im)`, each `(B,C,H,W)`.
/// `X[k,l] = Σ_{h,w} x[h,w] · exp(-2πi(kh/H + lw/W))`.
fn fft2_real(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_b, _c, h, w) = x.dims4()?;
    let (cos_w, sin_w) = dft_matrices(w, x.device(), x.dtype())?;
    let (cos_h, sin_h) = dft_matrices(h, x.device(), x.dtype())?;
    // Transform along W (last dim): exp(-iθ) = cos − i·sin, input real.
    let re = x.broadcast_matmul(&cos_w)?; // Σ_j x·cos
    let im = x.broadcast_matmul(&sin_w)?.neg()?; // −Σ_j x·sin
    // Transform along H: bring H to the last dim, complex matmul, restore.
    let re_t = re.transpose(2, 3)?.contiguous()?; // (B,C,W,H)
    let im_t = im.transpose(2, 3)?.contiguous()?;
    // (a−ib)(re+iim): re' = cos·re + sin·im ; im' = cos·im − sin·re  (matrices symmetric).
    let re2 = (re_t.broadcast_matmul(&cos_h)? + im_t.broadcast_matmul(&sin_h)?)?;
    let im2 = (im_t.broadcast_matmul(&cos_h)? - re_t.broadcast_matmul(&sin_h)?)?;
    Ok((re2.transpose(2, 3)?.contiguous()?, im2.transpose(2, 3)?.contiguous()?))
}

/// Inverse 2D DFT, returning the **real part** of `(1/HW) Σ X[k,l] exp(+2πi(...))` — the filtered
/// image. (For our use the imaginary part is negligible; we only need the real reconstruction.)
fn ifft2_real(re: &Tensor, im: &Tensor) -> Result<Tensor> {
    let (_b, _c, h, w) = re.dims4()?;
    let (cos_h, sin_h) = dft_matrices(h, re.device(), re.dtype())?;
    let (cos_w, sin_w) = dft_matrices(w, re.device(), re.dtype())?;
    // Inverse along H (last after transpose): exp(+iθ) = cos + i·sin.
    let re_t = re.transpose(2, 3)?.contiguous()?;
    let im_t = im.transpose(2, 3)?.contiguous()?;
    // (a+ib)(re+iim): re' = cos·re − sin·im ; im' = cos·im + sin·re.
    let re_h = (re_t.broadcast_matmul(&cos_h)? - im_t.broadcast_matmul(&sin_h)?)?;
    let im_h = (im_t.broadcast_matmul(&cos_h)? + re_t.broadcast_matmul(&sin_h)?)?;
    let re_h = re_h.transpose(2, 3)?.contiguous()?;
    let im_h = im_h.transpose(2, 3)?.contiguous()?;
    // Inverse along W; only the real part is needed: re'' = cos·re − sin·im.
    let re_final = (re_h.broadcast_matmul(&cos_w)? - im_h.broadcast_matmul(&sin_w)?)?;
    Ok((re_final / (h as f64 * w as f64))?)
}

/// FreeU's Fourier skip-filter (diffusers `fourier_filter`): suppress the **low-frequency** centre
/// of `x`'s spectrum by `scale`, leaving high frequencies intact. `threshold` is the half-width of
/// the low-freq box (diffusers default 1). Low frequencies in an unshifted spectrum live at the
/// corners (indices near 0 and near N), so we build the mask directly there — no fftshift needed.
pub fn fourier_filter(x: &Tensor, threshold: usize, scale: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    // FFT is done in F32 for numeric headroom, then cast back.
    let xf = x.to_dtype(DType::F32)?;
    let (h, w) = (xf.dim(2)?, xf.dim(3)?);
    let (re, im) = fft2_real(&xf)?;
    // Per-axis low-freq indicator: 1.0 in [0,threshold) ∪ [N-threshold, N), else 0.
    let axis_lowfreq = |n: usize| -> Result<Tensor> {
        let mut v = vec![0f32; n];
        for (i, e) in v.iter_mut().enumerate() {
            if i < threshold || i + threshold >= n {
                *e = 1.0;
            }
        }
        Ok(Tensor::from_vec(v, n, x.device())?)
    };
    let low_h = axis_lowfreq(h)?.reshape((h, 1))?;
    let low_w = axis_lowfreq(w)?.reshape((1, w))?;
    // mask = 1 everywhere, = scale where BOTH axes are low-freq (the corner boxes).
    let both_low = low_h.broadcast_mul(&low_w)?; // (H,W), 1 in the low-freq box else 0
    let mask = (both_low.affine(scale - 1.0, 1.0)?).reshape((1, 1, h, w))?; // 1 + (scale−1)·box
    let re = re.broadcast_mul(&mask)?;
    let im = im.broadcast_mul(&mask)?;
    Ok(ifft2_real(&re, &im)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round-trip: scale=1 leaves the spectrum untouched → iFFT(FFT(x)) == x. This validates both
    // the forward and inverse transforms (incl. non-power-of-two sizes) in one shot.
    #[test]
    fn fourier_filter_identity_roundtrip() {
        let d = Device::Cpu;
        for (h, w) in [(8usize, 8usize), (12, 24), (7, 13)] {
            let x = Tensor::randn(0f32, 1.0, (1, 2, h, w), &d).unwrap();
            let y = fourier_filter(&x, 1, 1.0).unwrap();
            let err = (y - &x).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(err < 1e-3, "roundtrip err {err} at {h}x{w}");
        }
    }

    // scale=0 zeroes the DC term, so the filtered map's mean drops toward ~0 while high-freq
    // structure (variance) survives — the qualitative FreeU behaviour.
    #[test]
    fn fourier_filter_suppresses_dc() {
        let d = Device::Cpu;
        let x = (Tensor::randn(0f32, 1.0, (1, 1, 16, 16), &d).unwrap() + 5.0).unwrap(); // mean ~5
        let y = fourier_filter(&x, 1, 0.0).unwrap();
        let my = y.mean_all().unwrap().to_scalar::<f32>().unwrap().abs();
        assert!(my < 0.5, "DC not suppressed: |mean| {my}");
    }
}
