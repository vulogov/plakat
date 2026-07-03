//! Pure-Rust tensor comparators for Tier 1 (per-module correctness). Compare a plakat
//! intermediate against a golden reference tensor and decide pass/fail against per-tensor
//! thresholds. Backend-independent: both operands are flattened to f32 on CPU and the
//! statistics computed in f64, so a golden captured in F32 (diffusers) compares cleanly to
//! a plakat capture in BF16/F16.

use anyhow::{Result, bail};
use candle_core::{DType, Tensor};

/// Similarity statistics between a candidate tensor and a golden reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorStats {
    /// Pearson correlation of the flattened values (catches structural/layout errors even
    /// when magnitudes are scaled — the primary signal for the silent-noise class).
    pub corr: f64,
    /// Cosine similarity (direction agreement, scale-invariant).
    pub cosine: f64,
    /// Max absolute element-wise difference (catches magnitude drift correlation misses).
    pub max_abs: f64,
    /// Whether the shapes matched (a shape mismatch is an immediate fail).
    pub shape_match: bool,
}

/// Per-tensor pass thresholds (from the manifest).
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    pub corr_min: f64,
    pub max_abs: f64,
}

impl TensorStats {
    /// A comparison passes when the shapes match, correlation is at least `corr_min`, and
    /// the max abs difference is within `max_abs`.
    pub fn passes(&self, t: &Thresholds) -> bool {
        self.shape_match && self.corr >= t.corr_min && self.max_abs <= t.max_abs
    }
}

/// Flatten a tensor to an f32 CPU vector regardless of dtype/device.
fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

/// Compare `candidate` against `golden`. A shape mismatch short-circuits to a definite fail
/// (`shape_match = false`, `corr = 0`, `max_abs = inf`).
pub fn compare(candidate: &Tensor, golden: &Tensor) -> Result<TensorStats> {
    if candidate.dims() != golden.dims() {
        return Ok(TensorStats { corr: 0.0, cosine: 0.0, max_abs: f64::INFINITY, shape_match: false });
    }
    let a = to_f32_vec(candidate)?;
    let b = to_f32_vec(golden)?;
    if a.len() != b.len() {
        bail!("flattened lengths differ ({} vs {}) despite matching dims", a.len(), b.len());
    }
    Ok(stats_from_slices(&a, &b))
}

/// The pure statistics kernel — split out so it can be unit-tested without tensors.
fn stats_from_slices(a: &[f32], b: &[f32]) -> TensorStats {
    let n = a.len();
    if n == 0 {
        // Two empty tensors are trivially identical.
        return TensorStats { corr: 1.0, cosine: 1.0, max_abs: 0.0, shape_match: true };
    }
    let (mut sa, mut sb) = (0f64, 0f64);
    for i in 0..n {
        sa += a[i] as f64;
        sb += b[i] as f64;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64); // for correlation (mean-centered)
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64); // for cosine (raw)
    let mut max_abs = 0f64;
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        let (dx, dy) = (x - ma, y - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_abs = max_abs.max((x - y).abs());
    }
    // Correlation: if either side is constant (zero variance), define corr as 1.0 when the
    // means also match (identical constants) else 0.0 — avoids a NaN from 0/0.
    let corr = if va == 0.0 || vb == 0.0 {
        if (ma - mb).abs() <= f64::EPSILON && va == 0.0 && vb == 0.0 { 1.0 } else { 0.0 }
    } else {
        cov / (va.sqrt() * vb.sqrt())
    };
    let cosine = if na == 0.0 || nb == 0.0 {
        if na == 0.0 && nb == 0.0 { 1.0 } else { 0.0 }
    } else {
        dot / (na.sqrt() * nb.sqrt())
    };
    TensorStats { corr, cosine, max_abs, shape_match: true }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn t(v: &[f32]) -> Tensor {
        Tensor::new(v, &Device::Cpu).unwrap()
    }

    #[test]
    fn identical_tensors_are_perfect() {
        let s = compare(&t(&[1., 2., 3., 4.]), &t(&[1., 2., 3., 4.])).unwrap();
        assert!(s.shape_match && s.max_abs == 0.0);
        assert!((s.corr - 1.0).abs() < 1e-9 && (s.cosine - 1.0).abs() < 1e-9);
        assert!(s.passes(&Thresholds { corr_min: 0.999, max_abs: 1e-6 }));
    }

    #[test]
    fn scaled_correlates_but_differs_in_magnitude() {
        // 2× scale: perfect correlation + cosine, but max_abs is large.
        let s = compare(&t(&[2., 4., 6., 8.]), &t(&[1., 2., 3., 4.])).unwrap();
        assert!((s.corr - 1.0).abs() < 1e-9, "scale preserves correlation");
        assert!(s.max_abs >= 3.9, "magnitude drift shows up in max_abs: {}", s.max_abs);
        // Correlation-only would wrongly pass; the max_abs bound catches it.
        assert!(!s.passes(&Thresholds { corr_min: 0.99, max_abs: 0.1 }));
    }

    #[test]
    fn shape_mismatch_is_an_immediate_fail() {
        let s = compare(&t(&[1., 2., 3.]), &t(&[1., 2., 3., 4.])).unwrap();
        assert!(!s.shape_match && s.max_abs.is_infinite());
        assert!(!s.passes(&Thresholds { corr_min: 0.0, max_abs: f64::INFINITY }));
    }

    #[test]
    fn anticorrelated_and_reordered_are_caught() {
        // Reversed order → low/negative correlation → fails a high corr floor.
        let s = compare(&t(&[1., 2., 3., 4.]), &t(&[4., 3., 2., 1.])).unwrap();
        assert!(s.corr < 0.0, "reversed data anti-correlates: {}", s.corr);
        assert!(!s.passes(&Thresholds { corr_min: 0.99, max_abs: 100.0 }));
    }

    #[test]
    fn dtype_and_device_are_normalized() {
        // A BF16 candidate vs an F32 golden still compares (both flattened to f32).
        let cand = t(&[1., 2., 3., 4.]).to_dtype(DType::BF16).unwrap();
        let s = compare(&cand, &t(&[1., 2., 3., 4.])).unwrap();
        assert!((s.corr - 1.0).abs() < 1e-3 && s.max_abs < 0.05, "bf16 rounding within tol");
    }
}
