//! v0.41 phase 0: Wuerstchen-style ratio-timestep scheduler for
//! Stable Cascade.
//!
//! Upstream Stable Cascade trains against `DDPMWuerstchenScheduler`
//! from diffusers — timesteps are float ratios in `[0, 1]` (NOT
//! integer 0-1000 like SD-family schedulers), and the noise schedule
//! is a cosine α-cumprod curve computed analytically from the ratio.
//!
//! plakat's pre-v0.41 generate used candle's SDXL-style DDPM, which
//! fed Stable Cascade timestep distributions it wasn't trained on.
//! v0.40's end-to-end smoke shipped "structurally valid but visually
//! noisy" output partly because of this mismatch.
//!
//! ## Algorithm
//!
//! - **Timesteps**: `linspace(1.0, 0.0, N+1)[:-1]` — N float ratios
//!   decreasing from 1.0 to just above 0.0.
//! - **`alpha_cumprod(t)`**: cosine schedule with shift `s = 0.008`
//!   (Improved DDPM convention):
//!
//!   ```text
//!     α_cumprod(t) = cos((t + s) / (1 + s) · π/2)² / α_cumprod(0)
//!   ```
//!
//!   where `α_cumprod(0) = cos(s/(1+s) · π/2)²` is the normalization
//!   constant. Clamped to `[1e-4, 1 - 1e-4]` to avoid div-by-zero
//!   at the schedule endpoints.
//!
//! - **`step(model_output, t, t_prev, sample)`**: standard DDPM
//!   reverse step:
//!
//!   ```text
//!     α     = α_cumprod(t) / α_cumprod(t_prev)
//!     μ     = (1 / √α) · (sample - (1-α)/√(1-α_cumprod(t)) · model_output)
//!     σ     = √((1-α) · (1-α_cumprod(t_prev)) / (1-α_cumprod(t)))
//!     next  = μ + σ · ε   (ε ~ N(0, 1); skip noise for final step)
//!   ```
//!
//! - **`init_noise_sigma`**: 1.0 (Wuerstchen latents start from
//!   standard normal noise, no extra scaling).
//!
//! - **`scale_model_input`**: identity (no per-step scaling).

use anyhow::Result;
use candle_core::Tensor;

/// Wuerstchen-style scheduler for Stable Cascade.
///
/// One instance per stage (Stage C / Stage B) — each stage runs its
/// own denoise loop with its own step count.
pub struct CascadeScheduler {
    timesteps: Vec<f64>,
    /// Cosine schedule shift parameter. Upstream default 0.008.
    s: f64,
    /// `cos(s / (1 + s) · π/2)²` — normalization constant for the
    /// cosine schedule so that `alpha_cumprod(0) = 1`.
    init_alpha_cumprod: f64,
}

impl CascadeScheduler {
    /// Construct a scheduler for the given step count.
    ///
    /// `num_inference_steps` is the number of denoise iterations
    /// (e.g., 20 for Stage C, 10 for Stage B). The returned scheduler
    /// produces that many timesteps via
    /// `linspace(1.0, 0.0, num_inference_steps + 1)[:-1]`.
    pub fn new(num_inference_steps: usize) -> Self {
        let s = 0.008;
        let init_alpha_cumprod = (s / (1.0 + s) * std::f64::consts::PI * 0.5).cos().powi(2);
        let n = num_inference_steps;
        // linspace(1.0, 0.0, n+1) — n+1 points evenly spaced from 1.0
        // to 0.0 inclusive; drop the last (0.0) to get n timesteps.
        let mut timesteps = Vec::with_capacity(n);
        if n > 0 {
            for i in 0..n {
                let frac = i as f64 / n as f64; // 0.0 ≤ frac < 1.0
                let t = 1.0 - frac;
                timesteps.push(t);
            }
        }
        Self {
            timesteps,
            s,
            init_alpha_cumprod,
        }
    }

    /// The ratio timesteps to iterate over in the denoise loop.
    /// Length = `num_inference_steps`, decreasing from ≤ 1.0 toward 0.
    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// Number of denoise steps.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Cosine α-cumprod at ratio timestep `t`. Clamped to
    /// `[1e-4, 1 - 1e-4]` to keep the step math finite at endpoints.
    pub fn alpha_cumprod(&self, t: f64) -> f64 {
        let raw = ((t + self.s) / (1.0 + self.s) * std::f64::consts::PI * 0.5)
            .cos()
            .powi(2)
            / self.init_alpha_cumprod;
        raw.clamp(1e-4, 1.0 - 1e-4)
    }

    /// The previous timestep for `t` — `max(t - 1/N, 0)`.
    /// Used by `step()` to compute the α at the next denoise level.
    pub fn prev_timestep(&self, t: f64) -> f64 {
        let n = self.timesteps.len();
        if n == 0 {
            0.0
        } else {
            (t - 1.0 / n as f64).max(0.0)
        }
    }

    /// `init_noise_sigma = 1.0` — Wuerstchen latents start from
    /// standard normal noise without extra scaling.
    pub fn init_noise_sigma(&self) -> f64 {
        1.0
    }

    /// Identity — no per-step input scaling in the Wuerstchen
    /// formulation (unlike SD DDPM where the scaler is √(1 + σ²)).
    pub fn scale_model_input(&self, sample: Tensor, _t: f64) -> Result<Tensor> {
        Ok(sample)
    }

    /// One DDPM reverse step. `model_output` is the noise prediction
    /// from the UNet; `sample` is the current noisy latent.
    ///
    /// Returns the next-step latent (or the final clean latent if
    /// `t` is the last in the schedule).
    pub fn step(
        &self,
        model_output: &Tensor,
        t: f64,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let t_prev = self.prev_timestep(t);
        let alpha_cumprod = self.alpha_cumprod(t);
        let alpha_cumprod_prev = self.alpha_cumprod(t_prev);
        let alpha = alpha_cumprod / alpha_cumprod_prev;

        // μ = (1 / √α) · (sample - (1-α)/√(1-α_cumprod) · model_output)
        let one_minus_alpha = 1.0 - alpha;
        let one_minus_alpha_cumprod = 1.0 - alpha_cumprod;
        let pred_scale = one_minus_alpha / one_minus_alpha_cumprod.sqrt();
        let pred_term = model_output.affine(pred_scale, 0.0)?;
        let inner = sample.sub(&pred_term)?;
        let mu_scale = 1.0 / alpha.sqrt();
        let mu = inner.affine(mu_scale, 0.0)?;

        // For the final step (t_prev == 0 → α_cumprod_prev clamped at
        // 1 - 1e-4), the noise term is vanishingly small. Skip the
        // noise add when t_prev == 0 — matches upstream's "if t > 0"
        // guard.
        if t_prev <= 0.0 {
            return Ok(mu);
        }

        let one_minus_alpha_cumprod_prev = 1.0 - alpha_cumprod_prev;
        let sigma =
            (one_minus_alpha * one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod).sqrt();
        let noise = Tensor::randn(0f32, 1f32, sample.shape(), sample.device())?
            .to_dtype(sample.dtype())?;
        let noise_term = noise.affine(sigma, 0.0)?;
        Ok(mu.add(&noise_term)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn timesteps_decrease_from_one_to_just_above_zero() {
        let s = CascadeScheduler::new(20);
        let ts = s.timesteps();
        assert_eq!(ts.len(), 20);
        assert!(ts[0] <= 1.0 && ts[0] > 0.5, "first timestep near 1.0");
        assert!(ts[19] > 0.0 && ts[19] < 0.1, "last timestep just above 0");
        for i in 1..20 {
            assert!(ts[i] < ts[i - 1], "timesteps should be decreasing");
        }
    }

    #[test]
    fn alpha_cumprod_monotonically_increases_as_t_decreases() {
        // At t=1: α_cumprod is small (lots of noise still in latent).
        // At t=0: α_cumprod ≈ 1 (no noise left, clean signal).
        let s = CascadeScheduler::new(20);
        let a_one = s.alpha_cumprod(1.0);
        let a_half = s.alpha_cumprod(0.5);
        let a_zero = s.alpha_cumprod(0.0);
        assert!(a_one < a_half, "α(1) < α(0.5) (got {a_one} vs {a_half})");
        assert!(a_half < a_zero, "α(0.5) < α(0) (got {a_half} vs {a_zero})");
        assert_eq!(a_zero, 1.0 - 1e-4, "α(0) is clamped at the upper bound");
    }

    #[test]
    fn alpha_cumprod_clamped_to_bounded_range() {
        let s = CascadeScheduler::new(20);
        for &t in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
            let a = s.alpha_cumprod(t);
            assert!(a >= 1e-4 && a <= 1.0 - 1e-4, "α({t}) = {a} out of range");
        }
    }

    #[test]
    fn prev_timestep_clamps_at_zero() {
        let s = CascadeScheduler::new(10);
        // t=0.05 with N=10 → prev = 0.05 - 0.1 = -0.05 → clamp to 0.
        assert_eq!(s.prev_timestep(0.05), 0.0);
        // Mid-range: t=0.5, N=10 → prev = 0.4.
        let p = s.prev_timestep(0.5);
        assert!((p - 0.4).abs() < 1e-10, "got {p}");
    }

    #[test]
    fn scale_model_input_is_identity() {
        let s = CascadeScheduler::new(10);
        let x = Tensor::randn(0f32, 1f32, (1, 4, 4, 4), &Device::Cpu).unwrap();
        let y = s.scale_model_input(x.clone(), 0.5).unwrap();
        let diff = (&x - &y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "scale_model_input should be identity (got {diff})");
    }

    #[test]
    fn step_preserves_shape() {
        let s = CascadeScheduler::new(10);
        let device = Device::Cpu;
        let sample = Tensor::randn(0f32, 1f32, (1, 16, 24, 24), &device).unwrap();
        let model_output = Tensor::randn(0f32, 1f32, (1, 16, 24, 24), &device).unwrap();
        let next = s.step(&model_output, 0.5, &sample).unwrap();
        assert_eq!(next.dims(), sample.dims());
    }

    #[test]
    fn step_at_final_timestep_skips_noise_term() {
        // At the last scheduler timestep, prev_t clamps to 0 and the
        // step should produce μ exactly (no stochastic noise add).
        // Verify determinism: two calls at the same t should produce
        // the same result.
        let s = CascadeScheduler::new(4);
        let device = Device::Cpu;
        let sample = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        let model_output = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        // Use the final timestep from the schedule.
        let t_final = *s.timesteps().last().unwrap();
        let r1 = s.step(&model_output, t_final, &sample).unwrap();
        let r2 = s.step(&model_output, t_final, &sample).unwrap();
        let diff = (&r1 - &r2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-6,
            "final step should be deterministic (no noise add), got max diff {diff}"
        );
    }

    #[test]
    fn step_mid_schedule_is_stochastic() {
        // Mid-schedule, σ > 0 so the noise term varies the output.
        let s = CascadeScheduler::new(20);
        let device = Device::Cpu;
        let sample = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        let model_output = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        let r1 = s.step(&model_output, 0.5, &sample).unwrap();
        let r2 = s.step(&model_output, 0.5, &sample).unwrap();
        let diff = (&r1 - &r2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-4, "mid-schedule step should add noise (got {diff})");
    }

    #[test]
    fn step_preserves_dtype() {
        let s = CascadeScheduler::new(10);
        let device = Device::Cpu;
        let sample = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        let model_output = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        // Both should be F32.
        assert_eq!(sample.dtype(), DType::F32);
        let next = s.step(&model_output, 0.5, &sample).unwrap();
        assert_eq!(next.dtype(), DType::F32);
    }
}
