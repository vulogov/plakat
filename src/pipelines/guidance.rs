//! Shared guidance post-processing — currently **CFG-rescale** (guidance rescale).
//!
//! High classifier-free-guidance scales push the guided prediction's statistics away from the
//! conditional's, over-exposing/over-saturating the image (the "Common Diffusion Noise Schedules
//! and Sample Steps Are Flawed" finding). CFG-rescale corrects this by rescaling the guided
//! prediction back toward the conditional prediction's per-sample standard deviation, then blending
//! that correction in by a factor `phi`:
//!
//! ```text
//! x_rescaled = x_cfg * std(x_cond) / std(x_cfg)
//! x_final    = phi * x_rescaled + (1 - phi) * x_cfg
//! ```
//!
//! `phi = 0` is exact CFG (no-op); `phi ≈ 0.7` is the paper's sweet spot. Opt-in via the
//! `PLAKAT_CFG_RESCALE` env (set by the `--guidance-rescale` CLI flag), so every pipeline that
//! routes its CFG blend through [`cfg_rescale`] honors one uniform knob — mirrors the PAG pattern.

use std::sync::OnceLock;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// The active CFG-rescale factor `phi` from `PLAKAT_CFG_RESCALE` (0 = off, the default).
pub fn cfg_rescale_phi() -> f64 {
    std::env::var("PLAKAT_CFG_RESCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(0.0)
}

/// FreeU backbone/skip factors `(b1, b2, s1, s2)` when enabled, else `None`. Read from
/// `PLAKAT_FREEU` (set by `--freeu` / `--freeu-params`): any value enables with the SD1.5 defaults
/// `(1.2, 1.4, 0.9, 0.2)`; a 4-value CSV `b1,b2,s1,s2` overrides (SDXL wants `1.3,1.4,0.9,0.2`).
/// `b*` boost the low-res backbone features; `s*` suppress the skip connections' low frequencies.
pub fn freeu_params() -> Option<(f64, f64, f64, f64)> {
    let v = std::env::var("PLAKAT_FREEU").ok()?;
    let parts: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if parts.len() == 4 {
        Some((parts[0], parts[1], parts[2], parts[3]))
    } else {
        Some((1.2, 1.4, 0.9, 0.2))
    }
}

/// Per-sample standard deviation over every non-batch dim, shape `(b, 1)`.
fn std_over_features(t: &Tensor) -> Result<Tensor> {
    let b = t.dim(0)?;
    let flat = t.reshape((b, ()))?; // (b, features)
    let mean = flat.mean_keepdim(1)?; // (b, 1)
    let var = flat.broadcast_sub(&mean)?.sqr()?.mean_keepdim(1)?; // (b, 1)
    Ok(var.sqrt()?)
}

/// Apply CFG-rescale to a guided prediction. `cfg_pred` is the CFG output
/// (`uncond + scale·(cond − uncond)`); `cond_pred` is the conditional prediction whose statistics
/// we rescale toward. Returns `cfg_pred` unchanged when the knob is off (`phi = 0`). Both tensors
/// must share shape `(b, …)`; std is computed per batch sample so it composes with batched CFG.
pub fn cfg_rescale(cfg_pred: &Tensor, cond_pred: &Tensor) -> Result<Tensor> {
    let phi = cfg_rescale_phi();
    if phi <= 0.0 {
        return Ok(cfg_pred.clone());
    }
    let b = cfg_pred.dim(0)?;
    let std_cond = std_over_features(cond_pred)?; // (b, 1)
    let std_cfg = std_over_features(cfg_pred)?; // (b, 1)
    // Guard a zero-std (flat) prediction so the ratio stays finite.
    let eps = (std_cfg.zeros_like()? + 1e-12)?;
    let factor = std_cond.broadcast_div(&std_cfg.broadcast_add(&eps)?)?; // (b, 1)
    // Broadcast the per-sample factor over the remaining dims.
    let mut shape = vec![b];
    shape.extend(std::iter::repeat(1).take(cfg_pred.rank() - 1));
    let factor = factor.reshape(shape)?;
    let rescaled = cfg_pred.broadcast_mul(&factor)?;
    Ok(((rescaled * phi)? + (cfg_pred * (1.0 - phi))?)?)
}

/// Dynamic-thresholding percentile from `PLAKAT_DYNTHRESH` (0 = off; ~99.5 = Imagen default).
pub fn dynthresh_percentile() -> f64 {
    std::env::var("PLAKAT_DYNTHRESH")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(0.0)
}

/// Standard SD ᾱ (alphas_cumprod) over 1000 train steps — scaled-linear betas
/// `(0.00085 → 0.012)`, byte-matching candle's `DDIMSchedulerConfig` default. Computed once.
fn sd_alphas_cumprod() -> &'static Vec<f64> {
    static A: OnceLock<Vec<f64>> = OnceLock::new();
    A.get_or_init(|| {
        let (n, bs, be) = (1000usize, 0.00085f64, 0.012f64);
        let mut acc = 1.0;
        (0..n)
            .map(|i| {
                let f = i as f64 / (n - 1) as f64;
                let beta = (bs.sqrt() + f * (be.sqrt() - bs.sqrt())).powi(2);
                acc *= 1.0 - beta;
                acc
            })
            .collect()
    })
}

/// Imagen dynamic thresholding of a predicted `x0` sample: per batch sample, take `s =
/// max(percentile(|x0|), 1.0)`, clamp `x0` to `[-s, s]`, and rescale by `1/s`. Compresses the
/// dynamic range so high-CFG saturation is pulled back without a hard static clip. The percentile
/// is computed on CPU (a cheap sort over the small latent) — the schedulers already run on CPU.
fn dynamic_threshold_sample(x0: &Tensor, percentile: f64) -> Result<Tensor> {
    let b = x0.dim(0)?;
    let flat = x0.reshape((b, ()))?.abs()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let mut s_vals = Vec::with_capacity(b);
    for row in &flat {
        let mut v = row.clone();
        v.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (((percentile / 100.0) * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
        s_vals.push(v[idx].max(1.0));
    }
    let mut shape = vec![b];
    shape.extend(std::iter::repeat(1).take(x0.rank() - 1));
    let s = Tensor::from_vec(s_vals, shape, x0.device())?.to_dtype(x0.dtype())?;
    let clamped = x0.broadcast_minimum(&s)?.broadcast_maximum(&s.neg()?)?;
    Ok(clamped.broadcast_div(&s)?)
}

/// Dynamic-threshold an **epsilon** prediction: reconstruct `x0` from `(sample, noise_pred)` at
/// `timestep` via the SD schedule, threshold it, and re-derive `epsilon`. No-op when the knob is
/// off or `is_epsilon` is false (v-prediction / flow-matching aren't handled). `sample` is the raw
/// latent `x_t` fed to `scheduler.step`.
pub fn apply_dynamic_threshold(
    noise_pred: &Tensor,
    sample: &Tensor,
    timestep: usize,
    is_epsilon: bool,
) -> Result<Tensor> {
    let p = dynthresh_percentile();
    if p <= 0.0 || !is_epsilon {
        return Ok(noise_pred.clone());
    }
    let ac = sd_alphas_cumprod();
    let abar = ac[timestep.min(ac.len() - 1)];
    let sa = abar.sqrt();
    let soma = (1.0 - abar).sqrt();
    // x0 = (x_t − √(1−ᾱ)·ε) / √ᾱ
    let x0 = ((sample - (noise_pred * soma)?)? / sa)?;
    let x0 = dynamic_threshold_sample(&x0, p)?;
    // ε' = (x_t − √ᾱ·x0) / √(1−ᾱ)
    Ok(((sample - (x0 * sa)?)? / soma)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // One serial test: the knob is a process-global env, so splitting into two `#[test]`s would
    // race (cargo runs them on parallel threads). Cover off→identity and phi=1→full-rescale here.
    #[test]
    fn cfg_rescale_off_and_full() {
        let d = Device::Cpu;
        unsafe { std::env::remove_var("PLAKAT_CFG_RESCALE") };

        // Off (unset) → the guided prediction is returned untouched.
        let cfg = Tensor::randn(0f32, 1.0, (2, 4, 8, 8), &d).unwrap();
        let cond = Tensor::randn(0f32, 1.0, (2, 4, 8, 8), &d).unwrap();
        let out = cfg_rescale(&cfg, &cond).unwrap();
        let diff = (out - &cfg).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0, "phi=0 must be identity");

        // phi=1 → the output std matches the conditional's (full rescale).
        unsafe { std::env::set_var("PLAKAT_CFG_RESCALE", "1.0") };
        let cfg = (Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &d).unwrap() * 3.0).unwrap(); // std ~3
        let cond = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &d).unwrap(); // std ~1
        let out = cfg_rescale(&cfg, &cond).unwrap();
        let s_out = std_over_features(&out).unwrap().to_vec2::<f32>().unwrap()[0][0];
        let s_cond = std_over_features(&cond).unwrap().to_vec2::<f32>().unwrap()[0][0];
        unsafe { std::env::remove_var("PLAKAT_CFG_RESCALE") };
        assert!((s_out - s_cond).abs() < 1e-4, "out std {s_out} vs cond std {s_cond}");
    }

    #[test]
    fn dynamic_threshold_compresses_outliers() {
        // A bulk of small values plus one big outlier: percentile≈95 → s≈max(0.2,1)=1, so the
        // outlier is clamped and the whole sample compressed to ≤1 in magnitude.
        let d = Device::Cpu;
        let mut data = vec![0.2f32; 63];
        data.push(9.0);
        let x0 = Tensor::from_vec(data, (1, 1, 8, 8), &d).unwrap();
        let out = dynamic_threshold_sample(&x0, 95.0).unwrap();
        let mx = out.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(mx <= 1.0 + 1e-5, "outlier not clamped: max {mx}");
    }

    #[test]
    fn dynthresh_off_and_non_epsilon_are_noops() {
        let d = Device::Cpu;
        let eps = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &d).unwrap();
        let x_t = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &d).unwrap();
        let unchanged = |o: Tensor| (o - &eps).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        unsafe { std::env::remove_var("PLAKAT_DYNTHRESH") };
        assert_eq!(unchanged(apply_dynamic_threshold(&eps, &x_t, 500, true).unwrap()), 0.0);
        unsafe { std::env::set_var("PLAKAT_DYNTHRESH", "99.5") };
        // Guarded off for non-epsilon predictions.
        assert_eq!(unchanged(apply_dynamic_threshold(&eps, &x_t, 500, false).unwrap()), 0.0);
        unsafe { std::env::remove_var("PLAKAT_DYNTHRESH") };
    }
}
