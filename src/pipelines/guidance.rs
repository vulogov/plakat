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

use anyhow::Result;
use candle_core::Tensor;

/// The active CFG-rescale factor `phi` from `PLAKAT_CFG_RESCALE` (0 = off, the default).
pub fn cfg_rescale_phi() -> f64 {
    std::env::var("PLAKAT_CFG_RESCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(0.0)
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
}
