//! Three additional schedulers, all sharing candle 0.8's `Scheduler` trait:
//!
//!   * Deterministic Euler — Euler-Ancestral with the noise-injection term
//!     removed. Same composition across runs for a given seed.
//!   * Heun — second-order predictor-corrector. Two model evaluations per
//!     "effective" step, but typically beats Euler at the same number of
//!     model calls.
//!   * DDPM (wrapped) — candle ships a `DDPMScheduler` that doesn't
//!     implement the `Scheduler` trait. We add a thin adapter.
//!
//! All three are pure F32 math (no F64 `powf`), so they work on Metal as
//! well as CUDA / CPU.

use candle_core::{Error, Result, Tensor};
use candle_transformers::models::stable_diffusion::ddpm::{DDPMScheduler, DDPMSchedulerConfig};
use candle_transformers::models::stable_diffusion::schedulers::{
    BetaSchedule, PredictionType, Scheduler, SchedulerConfig, TimestepSpacing,
};
use candle_transformers::models::stable_diffusion::utils::{interp, linspace};

// =====================================================================
// Shared helpers — beta schedule → alphas_cumprod → sigmas list
// =====================================================================

fn alphas_cumprod_for(
    beta_schedule: BetaSchedule,
    beta_start: f64,
    beta_end: f64,
    train_timesteps: usize,
) -> Result<Vec<f64>> {
    let betas = match beta_schedule {
        BetaSchedule::ScaledLinear => {
            linspace(beta_start.sqrt(), beta_end.sqrt(), train_timesteps)?.sqr()?
        }
        BetaSchedule::Linear => linspace(beta_start, beta_end, train_timesteps)?,
        BetaSchedule::SquaredcosCapV2 => candle_core::bail!(
            "SquaredcosCapV2 betas aren't supported here — pass Linear or ScaledLinear"
        ),
    };
    let betas = betas.to_vec1::<f64>()?;
    let mut alphas_cumprod = Vec::with_capacity(betas.len());
    for &beta in betas.iter() {
        let alpha = 1.0 - beta;
        alphas_cumprod.push(alpha * *alphas_cumprod.last().unwrap_or(&1.0));
    }
    Ok(alphas_cumprod)
}

fn pick_timesteps(
    spacing: TimestepSpacing,
    train_timesteps: usize,
    inference_steps: usize,
    steps_offset: usize,
) -> Result<Vec<usize>> {
    let step_ratio = train_timesteps / inference_steps;
    Ok(match spacing {
        TimestepSpacing::Leading => (0..inference_steps)
            .map(|s| s * step_ratio + steps_offset)
            .rev()
            .collect(),
        TimestepSpacing::Trailing => std::iter::successors(Some(train_timesteps), |n| {
            if *n > step_ratio {
                Some(n - step_ratio)
            } else {
                None
            }
        })
        .map(|n| n - 1)
        .collect(),
        TimestepSpacing::Linspace => linspace(0.0, (train_timesteps - 1) as f64, inference_steps)?
            .to_vec1::<f64>()?
            .iter()
            .map(|&f| f as usize)
            .rev()
            .collect(),
    })
}

/// Interpolate `sigmas[0..]` (indexed at `0,1,2,...`) at the requested timestep
/// positions, append a final 0.0 sentinel. Matches diffusers' convention.
fn sigmas_at(timesteps: &[usize], alphas_cumprod: &[f64]) -> Vec<f64> {
    let sigmas: Vec<f64> = alphas_cumprod
        .iter()
        .map(|&a| ((1.0 - a) / a).sqrt())
        .collect();
    let xs: Vec<f64> = (0..sigmas.len()).map(|i| i as f64).collect();
    let ts: Vec<f64> = timesteps.iter().map(|&t| t as f64).collect();
    let mut out = interp(&ts, &xs, &sigmas);
    out.push(0.0);
    out
}

// =====================================================================
// Deterministic Euler scheduler.
// =====================================================================

#[derive(Debug, Clone, Copy)]
pub struct EulerSchedulerConfig {
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub steps_offset: usize,
    pub prediction_type: PredictionType,
    pub train_timesteps: usize,
    pub timestep_spacing: TimestepSpacing,
}

impl Default for EulerSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            steps_offset: 1,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
            timestep_spacing: TimestepSpacing::Leading,
        }
    }
}

impl SchedulerConfig for EulerSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(EulerScheduler::new(inference_steps, *self)?))
    }
}

#[derive(Debug, Clone)]
pub struct EulerScheduler {
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    config: EulerSchedulerConfig,
}

impl EulerScheduler {
    fn new(inference_steps: usize, config: EulerSchedulerConfig) -> Result<Self> {
        let timesteps = pick_timesteps(
            config.timestep_spacing,
            config.train_timesteps,
            inference_steps,
            config.steps_offset,
        )?;
        let alphas_cumprod = alphas_cumprod_for(
            config.beta_schedule,
            config.beta_start,
            config.beta_end,
            config.train_timesteps,
        )?;
        let sigmas = sigmas_at(&timesteps, &alphas_cumprod);
        let init_noise_sigma = sigmas
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| if a > b { a } else { b });
        Ok(Self {
            timesteps,
            sigmas,
            init_noise_sigma,
            config,
        })
    }
}

impl Scheduler for EulerScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        match self.config.timestep_spacing {
            TimestepSpacing::Trailing | TimestepSpacing::Linspace => self.init_noise_sigma,
            TimestepSpacing::Leading => (self.init_noise_sigma.powi(2) + 1.0).sqrt(),
        }
    }

    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> Result<Tensor> {
        let idx = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| Error::Msg(format!("Euler: timestep {timestep} not in schedule")))?;
        let sigma = self.sigmas[idx];
        sample / (sigma.powi(2) + 1.0).sqrt()
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        let idx = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| Error::Msg(format!("Euler: timestep {timestep} not in schedule")))?;
        let sigma = self.sigmas[idx];
        original + (noise * sigma)?
    }

    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let idx = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| Error::Msg(format!("Euler: timestep {timestep} not in schedule")))?;
        let sigma = self.sigmas[idx];
        let sigma_next = self.sigmas[idx + 1];

        let pred_x0 = match self.config.prediction_type {
            PredictionType::Epsilon => (sample - (model_output * sigma))?,
            PredictionType::VPrediction => {
                ((model_output * (-sigma / (sigma.powi(2) + 1.0).sqrt()))?
                    + (sample / (sigma.powi(2) + 1.0))?)?
            }
            PredictionType::Sample => candle_core::bail!("Euler: prediction_type=Sample unsupported"),
        };

        // dx/dσ = (sample - x_pred) / σ; step by dt = σ_next - σ.
        let derivative = ((sample - pred_x0)? / sigma)?;
        let dt = sigma_next - sigma;
        sample + (derivative * dt)?
    }
}

// =====================================================================
// Heun scheduler — second-order predictor-corrector.
//
// For N "effective" inference steps, this returns 2N-1 timesteps (each
// inner step appears twice: once for the predictor, once for the
// corrector). The caller's loop runs the UNet `2N - 1` times. Quality
// generally beats Euler at the same number of model calls; not as cheap
// as Euler at the same `--steps` value.
// =====================================================================

#[derive(Debug, Clone, Copy)]
pub struct HeunSchedulerConfig {
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub prediction_type: PredictionType,
    pub train_timesteps: usize,
}

impl Default for HeunSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

impl SchedulerConfig for HeunSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(HeunScheduler::new(inference_steps, *self)?))
    }
}

#[derive(Debug, Clone)]
pub struct HeunScheduler {
    /// Interleaved timesteps: `[t0, t1, t1, t2, t2, …, t_{N-1}, t_{N-1}]`.
    /// Length 2N - 1; the first element appears once, the rest twice.
    timesteps: Vec<usize>,
    /// Sigma sequence aligned to unique timesteps + a trailing 0.0.
    /// Length N + 1.
    sigmas: Vec<f64>,
    /// Map from each (interleaved) timesteps[i] → sigma index in `sigmas`.
    /// The same logical step appears at sigma index k for both predictor
    /// (state=first_order) and corrector (state=second_order).
    /// Specifically: timesteps[2i-1] and timesteps[2i] both map to sigma index i+1.
    init_noise_sigma: f64,
    config: HeunSchedulerConfig,
    // ---- state machine ----
    /// Position in `timesteps` of the upcoming step (used to detect predictor
    /// vs corrector by parity).
    next_step_idx: usize,
    /// Previous-step derivative — saved during the predictor, used during
    /// the corrector.
    prev_derivative: Option<Tensor>,
    /// Sample at start of the current logical step — saved during the
    /// predictor, used during the corrector.
    saved_sample: Option<Tensor>,
    /// dt used by the predictor — reused by the corrector.
    saved_dt: f64,
}

impl HeunScheduler {
    fn new(inference_steps: usize, config: HeunSchedulerConfig) -> Result<Self> {
        if inference_steps < 2 {
            candle_core::bail!("Heun needs at least 2 inference steps");
        }
        let unique: Vec<usize> = linspace(0.0, (config.train_timesteps - 1) as f64, inference_steps)?
            .to_vec1::<f64>()?
            .into_iter()
            .map(|f| f as usize)
            .rev()
            .collect();
        let alphas_cumprod = alphas_cumprod_for(
            config.beta_schedule,
            config.beta_start,
            config.beta_end,
            config.train_timesteps,
        )?;
        let sigmas = sigmas_at(&unique, &alphas_cumprod);

        // Interleave: first element once, rest twice.
        let mut timesteps = Vec::with_capacity(2 * unique.len() - 1);
        timesteps.push(unique[0]);
        for &t in &unique[1..] {
            timesteps.push(t);
            timesteps.push(t);
        }

        let init_noise_sigma = sigmas
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| if a > b { a } else { b });

        Ok(Self {
            timesteps,
            sigmas,
            init_noise_sigma,
            config,
            next_step_idx: 0,
            prev_derivative: None,
            saved_sample: None,
            saved_dt: 0.0,
        })
    }

    /// Map `next_step_idx` to the (round, is_first_order) pair.
    ///
    /// Interleaved timesteps for N rounds: [t0, t1, t1, t2, t2, …, t_{N-1}, t_{N-1}]
    /// has length 2N-1. Round k has its predictor at index 2k (or 0 for k=0),
    /// its corrector at index 2k+1. So:
    ///   round = idx / 2
    ///   is_first_order = (idx % 2 == 0)
    fn position(&self) -> (usize, bool) {
        let i = self.next_step_idx;
        (i / 2, i % 2 == 0)
    }
}

impl Scheduler for HeunScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        (self.init_noise_sigma.powi(2) + 1.0).sqrt()
    }

    fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Result<Tensor> {
        let (sigma_idx, _) = self.position();
        let sigma = self.sigmas[sigma_idx];
        sample / (sigma.powi(2) + 1.0).sqrt()
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        // Use the sigma at the first occurrence of this timestep.
        let sigma_idx = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .map(|i| if i == 0 { 0 } else { (i + 1) / 2 })
            .unwrap_or(0);
        let sigma = self.sigmas[sigma_idx];
        original + (noise * sigma)?
    }

    fn step(&mut self, model_output: &Tensor, _timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let (sigma_idx, is_first_order) = self.position();
        let sigma = self.sigmas[sigma_idx];
        let sigma_next = self.sigmas[sigma_idx + 1];

        // Predicted x0 from epsilon (or v-prediction).
        let sigma_for_pred = if is_first_order { sigma } else { sigma_next };
        let pred_x0 = match self.config.prediction_type {
            PredictionType::Epsilon => (sample - (model_output * sigma_for_pred))?,
            PredictionType::VPrediction => ((model_output
                * (-sigma_for_pred / (sigma_for_pred.powi(2) + 1.0).sqrt()))?
                + (sample / (sigma_for_pred.powi(2) + 1.0))?)?,
            PredictionType::Sample => candle_core::bail!("Heun: prediction_type=Sample unsupported"),
        };

        let result = if is_first_order {
            // Predictor (Euler step) — also save state for the corrector.
            let derivative = ((sample - &pred_x0)? / sigma)?;
            let dt = sigma_next - sigma;
            self.prev_derivative = Some(derivative.clone());
            self.saved_sample = Some(sample.clone());
            self.saved_dt = dt;
            (sample + (derivative * dt)?)?
        } else {
            // Corrector — uses the current `sample` (= post-predictor output)
            // for the new derivative, averages it with the saved
            // pre-predictor derivative, then integrates from the saved
            // pre-predictor sample.
            let derivative_new = ((sample - &pred_x0)? / sigma_next)?;
            let prev = self
                .prev_derivative
                .as_ref()
                .ok_or_else(|| Error::Msg("Heun corrector without prev_derivative".to_string()))?;
            let avg = ((prev + &derivative_new)? * 0.5_f64)?;
            let saved = self
                .saved_sample
                .as_ref()
                .ok_or_else(|| Error::Msg("Heun corrector without saved_sample".to_string()))?;
            let out = (saved + (avg * self.saved_dt)?)?;
            // Reset for the next predictor.
            self.prev_derivative = None;
            self.saved_sample = None;
            self.saved_dt = 0.0;
            out
        };
        self.next_step_idx += 1;
        Ok(result)
    }
}

// =====================================================================
// DDPM (wrap candle's implementation in the Scheduler trait)
// =====================================================================

pub use candle_transformers::models::stable_diffusion::ddpm::DDPMSchedulerConfig as DdpmConfig;

#[derive(Debug, Clone)]
pub struct DdpmConfigWrap(pub DDPMSchedulerConfig);

impl SchedulerConfig for DdpmConfigWrap {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(DdpmSchedulerWrap {
            inner: DDPMScheduler::new(inference_steps, self.0.clone())?,
        }))
    }
}

pub struct DdpmSchedulerWrap {
    inner: DDPMScheduler,
}

impl Scheduler for DdpmSchedulerWrap {
    fn timesteps(&self) -> &[usize] {
        self.inner.timesteps()
    }
    fn init_noise_sigma(&self) -> f64 {
        self.inner.init_noise_sigma()
    }
    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> Result<Tensor> {
        Ok(self.inner.scale_model_input(sample, timestep))
    }
    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        self.inner.add_noise(original, noise, timestep)
    }
    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        self.inner.step(model_output, timestep, sample)
    }
}
