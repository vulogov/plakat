//! LCM (Latent Consistency Model) scheduler.
//!
//! Implements the consistency-function sampling from
//! Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images
//! with Few-Step Inference" (2023), mirroring the diffusers `LCMScheduler`.
//!
//! The scheduler is what makes LCM-LoRA actually work at 4–8 steps. Used
//! against the same UNet weights at the same step count, DDIM gives muddy
//! output while LCM gives crisp output — the difference is which timesteps
//! are visited and how the consistency function maps model output to the
//! denoised sample at each step.
//!
//! Math, per step:
//!
//!   c_skip = sigma_data² / (scaled_t² + sigma_data²)
//!   c_out  = scaled_t / sqrt(scaled_t² + sigma_data²)
//!   x0_hat = (sample − sqrt(β_t) · ε) / sqrt(α_t)        # epsilon prediction
//!   denoised = c_out · x0_hat + c_skip · sample          # consistency fn
//!
//!   if not last step:
//!       prev_sample = sqrt(α_{t-1}) · denoised + sqrt(β_{t-1}) · noise
//!   else:
//!       prev_sample = denoised

use candle_core::{Result, Tensor};
use candle_transformers::models::stable_diffusion::schedulers::{
    BetaSchedule, PredictionType, Scheduler, SchedulerConfig,
};
use candle_transformers::models::stable_diffusion::utils::linspace;

#[derive(Debug, Clone, Copy)]
pub struct LcmSchedulerConfig {
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub train_timesteps: usize,
    /// LCM's "original inference steps" parameter — the size of the
    /// pre-strided timestep pool. Default 50 matches diffusers and the
    /// LCM-LoRA training recipe.
    pub original_inference_steps: usize,
    /// `sigma_data` constant in the consistency function. 0.5 matches LCM
    /// distillation.
    pub sigma_data: f64,
    /// Multiplier applied to the raw timestep before computing c_skip / c_out.
    /// 10.0 matches diffusers; LCM-LoRAs are trained with this value.
    pub timestep_scaling: f64,
    pub prediction_type: PredictionType,
}

impl Default for LcmSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            original_inference_steps: 50,
            sigma_data: 0.5,
            timestep_scaling: 10.0,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

impl SchedulerConfig for LcmSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(LcmScheduler::new(inference_steps, *self)?))
    }
}

#[derive(Debug, Clone)]
pub struct LcmScheduler {
    config: LcmSchedulerConfig,
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
}

impl LcmScheduler {
    fn new(inference_steps: usize, config: LcmSchedulerConfig) -> Result<Self> {
        // ---- alphas_cumprod from beta schedule ----
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => linspace(
                config.beta_start.sqrt(),
                config.beta_end.sqrt(),
                config.train_timesteps,
            )?
            .sqr()?,
            BetaSchedule::Linear => {
                linspace(config.beta_start, config.beta_end, config.train_timesteps)?
            }
            BetaSchedule::SquaredcosCapV2 => {
                candle_core::bail!(
                    "LCM scheduler doesn't support SquaredcosCapV2 betas \
                     (LCM-LoRAs are trained with ScaledLinear)"
                );
            }
        };
        let betas = betas.to_vec1::<f64>()?;
        let mut alphas_cumprod: Vec<f64> = Vec::with_capacity(betas.len());
        for &beta in betas.iter() {
            let alpha = 1.0 - beta;
            let prev = *alphas_cumprod.last().unwrap_or(&1.0);
            alphas_cumprod.push(alpha * prev);
        }

        // ---- LCM-specific timestep selection ----
        // Stride the training schedule into a pool of `original_inference_steps`
        // candidate timesteps, then pick `inference_steps` of them.
        let k = config.train_timesteps / config.original_inference_steps;
        if k == 0 {
            candle_core::bail!(
                "LCM: train_timesteps ({}) must be ≥ original_inference_steps ({})",
                config.train_timesteps,
                config.original_inference_steps
            );
        }
        // [k-1, 2k-1, ..., original_inference_steps·k - 1], reversed (high → low)
        let lcm_origin: Vec<usize> = (1..=config.original_inference_steps)
            .rev()
            .map(|i| i * k - 1)
            .collect();
        let lcm_len = lcm_origin.len();
        if inference_steps == 0 || inference_steps > lcm_len {
            candle_core::bail!(
                "LCM: inference_steps {} not in 1..={}; the original LCM \
                 schedule has only {} timesteps to choose from",
                inference_steps,
                lcm_len,
                lcm_len
            );
        }
        // floor(linspace(0, lcm_len, num=inference_steps, endpoint=False))
        let timesteps: Vec<usize> = (0..inference_steps)
            .map(|i| {
                let f = (i as f64) * (lcm_len as f64) / (inference_steps as f64);
                let idx = (f.floor() as usize).min(lcm_len - 1);
                lcm_origin[idx]
            })
            .collect();

        Ok(Self {
            config,
            timesteps,
            alphas_cumprod,
        })
    }
}

impl Scheduler for LcmScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        1.0
    }

    fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Result<Tensor> {
        Ok(sample)
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        let t = timestep.min(self.alphas_cumprod.len() - 1);
        let sqrt_alpha = self.alphas_cumprod[t].sqrt();
        let sqrt_one_minus = (1.0 - self.alphas_cumprod[t]).sqrt();
        (original * sqrt_alpha)? + (noise * sqrt_one_minus)?
    }

    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        // Locate the current position in the schedule. The caller iterates
        // `self.timesteps()` from a snapshot, so it's safe to assume timestep
        // is one of ours.
        let idx = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("timestep {timestep} not in LCM schedule"))
            })?;
        let is_last = idx + 1 >= self.timesteps.len();
        let prev_t = if is_last {
            timestep
        } else {
            self.timesteps[idx + 1]
        };
        let t_safe = timestep.min(self.alphas_cumprod.len() - 1);
        let prev_safe = prev_t.min(self.alphas_cumprod.len() - 1);

        let alpha_prod_t = self.alphas_cumprod[t_safe];
        let alpha_prod_t_prev = self.alphas_cumprod[prev_safe];
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Boundary scalings — the consistency function.
        let scaled_t = (timestep as f64) * self.config.timestep_scaling;
        let sigma_data_sq = self.config.sigma_data * self.config.sigma_data;
        let c_skip = sigma_data_sq / (scaled_t * scaled_t + sigma_data_sq);
        let c_out = scaled_t / (scaled_t * scaled_t + sigma_data_sq).sqrt();

        // Predict x0 from the model output.
        let pred_x0 = match self.config.prediction_type {
            PredictionType::Epsilon => {
                ((sample - (model_output * beta_prod_t.sqrt())?)?
                    * (1.0 / alpha_prod_t.sqrt()))?
            }
            PredictionType::VPrediction => {
                ((sample * alpha_prod_t.sqrt())? - (model_output * beta_prod_t.sqrt())?)?
            }
            PredictionType::Sample => model_output.clone(),
        };

        // Apply the consistency function.
        let denoised = ((&pred_x0 * c_out)? + (sample * c_skip)?)?;

        // Last step: return the denoised sample directly.
        if is_last {
            return Ok(denoised);
        }
        // Else re-noise to the next timestep (the "multistep inference" trick
        // — adds stochasticity that improves few-step LCM quality).
        let noise =
            Tensor::randn(0f32, 1f32, denoised.shape(), denoised.device())?.to_dtype(denoised.dtype())?;
        let prev_sample = ((&denoised * alpha_prod_t_prev.sqrt())?
            + (noise * beta_prod_t_prev.sqrt())?)?;
        Ok(prev_sample)
    }
}
