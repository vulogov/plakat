//! Scheduler selection for the t2i pipeline.
//!
//! Wraps candle's stable_diffusion scheduler configs behind a simple enum that
//! the CLI can parse. The candle Scheduler trait is the runtime interface; the
//! Config types differ per algorithm but all implement `SchedulerConfig::build`.

use anyhow::{Result, anyhow};
use candle_core::Device;
use candle_transformers::models::stable_diffusion::{
    StableDiffusionConfig,
    ddim::DDIMSchedulerConfig,
    euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig,
    schedulers::{BetaSchedule, PredictionType, Scheduler, SchedulerConfig},
    uni_pc::{
        CorrectorConfiguration, ExponentialSigmaSchedule, KarrasSigmaSchedule, SigmaSchedule,
        UniPCSchedulerConfig,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub enum SchedulerKind {
    /// Use the SD variant's built-in default (DDIM for SD 1.5/2.1/SDXL,
    /// Euler-Ancestral for SDXL-Turbo).
    #[default]
    Default,
    Ddim,
    EulerA,
    /// UniPC corrector with default Karras sigmas. Predictor-corrector
    /// behaviour — generally smoother at low step counts.
    UniPc,
    /// DPM-Solver++ 2M Karras — multistep without the UniPC corrector.
    /// Tends to render slightly crisper edges than `unipc` at the same step
    /// count; widely considered a "safe default" in A1111 / ComfyUI.
    DpmppKarras,
    /// UniPC with exponential sigma schedule instead of Karras. Different
    /// noise-step distribution; sometimes better for very low step counts.
    UniPcExp,
    /// LCM — Latent Consistency Model scheduler. Designed for LCM-LoRAs at
    /// 4–8 steps. Pure F32 arithmetic, so works on Metal/CUDA/CPU. Requires
    /// `steps ≤ original_inference_steps` (50 by default).
    Lcm,
    /// Deterministic Euler — Euler-Ancestral without the noise injection.
    /// Reproducible across runs given a seed. Works on Metal/CUDA/CPU.
    Euler,
    /// Euler with **trailing** timestep spacing (diffusers `timestep_spacing="trailing"`).
    /// This is the schedule **SDXL-Lightning** is distilled for — leading spacing wrecks its
    /// few-step (2–8) output. Otherwise identical to `Euler`. Works on Metal/CUDA/CPU.
    EulerTrailing,
    /// Heun second-order predictor-corrector. Two UNet evaluations per
    /// "effective" step. Higher quality at the same number of model calls;
    /// approximately 2× wall time per `--steps` value vs Euler.
    Heun,
    /// DDPM — the original reference. Slow (typically `--steps` close to
    /// the training schedule); mainly useful as a baseline. Works on
    /// Metal/CUDA/CPU.
    Ddpm,
}

impl std::str::FromStr for SchedulerKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "default" => Self::Default,
            "ddim" => Self::Ddim,
            "euler-a" | "euler_a" | "eulera" | "euler-ancestral" => Self::EulerA,
            "unipc" | "uni-pc" => Self::UniPc,
            "dpm++" | "dpm-solver++" | "dpmpp" | "dpmpp-2m" | "dpm++2m" | "dpmpp-karras" => {
                Self::DpmppKarras
            }
            "unipc-exp" | "unipc-exponential" => Self::UniPcExp,
            "lcm" | "lcm-scheduler" => Self::Lcm,
            "euler" | "euler-discrete" | "euler-deterministic" => Self::Euler,
            "euler-trailing" | "euler-discrete-trailing" | "lightning" => Self::EulerTrailing,
            "heun" | "heun-discrete" => Self::Heun,
            "ddpm" => Self::Ddpm,
            other => {
                return Err(anyhow!(
                    "unknown scheduler {other:?} (try: default | ddim | euler-a | euler | \
                     euler-trailing | heun | unipc | dpmpp-2m | unipc-exp | lcm | ddpm)"
                ));
            }
        })
    }
}

/// Refuse known-broken combinations early so the user gets a useful message
/// instead of a cryptic mid-inference error.
pub fn check_device_support(_kind: SchedulerKind, _device: &Device) -> Result<()> {
    // v2.4: the UniPC / DPM-Solver++ family (UniPc / DpmppKarras / UniPcExp) used to be rejected
    // on Metal — their solver-coefficient math builds F64 tensors on the sample's device, and
    // candle 0.10.2 has no F64 Metal backend. They now run on Metal via `CpuHopScheduler`, which
    // routes the (tiny) per-step scheduler math through the CPU. So every scheduler is supported
    // on every device; this stays as a no-op hook in case a future scheduler needs gating.
    Ok(())
}

/// Build a Scheduler for `steps` steps. PredictionType=Epsilon is the trained
/// type for SD 1.5 / 2.1 / SDXL / SDXL-Turbo; we don't yet expose v-prediction.
/// Wraps a candle scheduler whose internals use **F64 tensor ops** (the UniPC / DPM-Solver++
/// family: they build F64 solver-coefficient tensors on the *sample's* device) so it runs on
/// **Metal**, where candle 0.10.2 has no F64 backend. Every tensor method is routed through the
/// CPU — where F64 works — and the result is moved back to the caller's device. The latent is
/// tiny (~64–256 KB), so the round-trip is negligible next to the denoiser forward (seconds).
/// On a CPU device the hop is a no-op. This unblocks DPM++ 2M Karras / UniPC on Apple Silicon.
struct CpuHopScheduler {
    inner: Box<dyn Scheduler>,
}

impl CpuHopScheduler {
    fn to_cpu(t: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        t.to_device(&Device::Cpu)
    }
}

impl Scheduler for CpuHopScheduler {
    fn timesteps(&self) -> &[usize] {
        self.inner.timesteps()
    }
    fn init_noise_sigma(&self) -> f64 {
        self.inner.init_noise_sigma()
    }
    fn add_noise(
        &self,
        original: &candle_core::Tensor,
        noise: candle_core::Tensor,
        timestep: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        let dev = original.device().clone();
        let out = self
            .inner
            .add_noise(&Self::to_cpu(original)?, noise.to_device(&Device::Cpu)?, timestep)?;
        out.to_device(&dev)
    }
    fn scale_model_input(
        &self,
        sample: candle_core::Tensor,
        timestep: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        let dev = sample.device().clone();
        let out = self.inner.scale_model_input(sample.to_device(&Device::Cpu)?, timestep)?;
        out.to_device(&dev)
    }
    fn step(
        &mut self,
        model_output: &candle_core::Tensor,
        timestep: usize,
        sample: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        let dev = sample.device().clone();
        let out = self
            .inner
            .step(&Self::to_cpu(model_output)?, timestep, &Self::to_cpu(sample)?)?;
        out.to_device(&dev)
    }
}

pub fn build(
    kind: SchedulerKind,
    cfg: &StableDiffusionConfig,
    steps: usize,
) -> Result<Box<dyn Scheduler>> {
    Ok(match kind {
        SchedulerKind::Default => cfg.build_scheduler(steps)?,
        SchedulerKind::Ddim => DDIMSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?,
        SchedulerKind::EulerA => EulerAncestralDiscreteSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?,
        // UniPC with corrector + DPM-Solver++ algorithm + Karras sigmas (default).
        SchedulerKind::UniPc => Box::new(CpuHopScheduler {
            inner: UniPCSchedulerConfig {
                prediction_type: PredictionType::Epsilon,
                ..Default::default()
            }
            .build(steps)?,
        }),
        // DPM-Solver++ 2M Karras — corrector disabled, otherwise UniPC defaults.
        // (candle's `UniPCSchedulerConfig` doesn't expose an algorithm_type
        // toggle even though the AlgorithmType enum exists — the corrector
        // on/off is the meaningful behavior difference.)
        SchedulerKind::DpmppKarras => Box::new(CpuHopScheduler {
            inner: UniPCSchedulerConfig {
                prediction_type: PredictionType::Epsilon,
                corrector: CorrectorConfiguration::Disabled,
                sigma_schedule: SigmaSchedule::Karras(KarrasSigmaSchedule::default()),
                ..Default::default()
            }
            .build(steps)?,
        }),
        // UniPC with exponential sigma schedule (less common but sometimes useful).
        SchedulerKind::UniPcExp => Box::new(CpuHopScheduler {
            inner: UniPCSchedulerConfig {
                prediction_type: PredictionType::Epsilon,
                sigma_schedule: SigmaSchedule::Exponential(ExponentialSigmaSchedule::default()),
                ..Default::default()
            }
            .build(steps)?,
        }),
        // LCM — consistency-function sampling, designed for LCM-LoRAs.
        SchedulerKind::Lcm => crate::pipelines::lcm_scheduler::LcmSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?,
        // Deterministic Euler.
        SchedulerKind::Euler => crate::pipelines::extra_schedulers::EulerSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?,
        // Euler with trailing timestep spacing — SDXL-Lightning's distilled schedule.
        SchedulerKind::EulerTrailing => crate::pipelines::extra_schedulers::EulerSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            timestep_spacing:
                candle_transformers::models::stable_diffusion::schedulers::TimestepSpacing::Trailing,
            ..Default::default()
        }
        .build(steps)?,
        // Heun second-order predictor-corrector.
        SchedulerKind::Heun => crate::pipelines::extra_schedulers::HeunSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?,
        // DDPM — wrap candle's standalone implementation.
        SchedulerKind::Ddpm => crate::pipelines::extra_schedulers::DdpmConfigWrap(
            crate::pipelines::extra_schedulers::DdpmConfig {
                prediction_type: PredictionType::Epsilon,
                ..Default::default()
            },
        )
        .build(steps)?,
    })
}

/// AnimateDiff scheduler. The motion adapters were fine-tuned on the
/// **linear** beta schedule, NOT SD's default `scaled_linear`. Sampling
/// a motion model with scaled_linear mismatches the noise schedule the
/// motion modules were trained on and corrupts the frames (the spatial
/// path alone is fine — it's the motion modules that are schedule-
/// sensitive). So the Default kind maps to a linear-beta DDIM here;
/// explicit scheduler choices pass through unchanged.
pub fn build_animate(
    kind: SchedulerKind,
    cfg: &StableDiffusionConfig,
    steps: usize,
) -> Result<Box<dyn Scheduler>> {
    match kind {
        SchedulerKind::Default => Ok(DDIMSchedulerConfig {
            beta_schedule: BetaSchedule::Linear,
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?),
        other => build(other, cfg, steps),
    }
}

/// PixArt scheduler. PixArt-α/Σ train with the IDDPM **linear** beta
/// schedule `beta_start=0.0001, beta_end=0.02` — NOT SD's `scaled_linear`
/// (0.00085..0.012). Sampling the DiT on the SD schedule mismatches its
/// training noise levels and the sampler fails to denoise (pure noise
/// out, even with finite latents). Default maps to a DDIM with PixArt's
/// betas; explicit scheduler choices pass through unchanged.
pub fn build_pixart(
    kind: SchedulerKind,
    cfg: &StableDiffusionConfig,
    steps: usize,
) -> Result<Box<dyn Scheduler>> {
    match kind {
        SchedulerKind::Default => Ok(DDIMSchedulerConfig {
            beta_start: 0.0001,
            beta_end: 0.02,
            beta_schedule: BetaSchedule::Linear,
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?),
        other => build(other, cfg, steps),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::extra_schedulers::EulerSchedulerConfig;
    use candle_transformers::models::stable_diffusion::schedulers::{
        SchedulerConfig, TimestepSpacing,
    };

    #[test]
    fn from_str_parses_euler_trailing() {
        assert!(matches!(
            "euler-trailing".parse::<SchedulerKind>().unwrap(),
            SchedulerKind::EulerTrailing
        ));
        assert!(matches!(
            "lightning".parse::<SchedulerKind>().unwrap(),
            SchedulerKind::EulerTrailing
        ));
    }

    // SDXL-Lightning needs trailing spacing: the first timestep must be the last train step
    // (999 for a 1000-step schedule), and the schedule must differ from the leading default.
    #[test]
    fn euler_trailing_uses_trailing_spacing() {
        let steps = 8;
        let leading = EulerSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)
        .unwrap();
        let trailing = EulerSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            timestep_spacing: TimestepSpacing::Trailing,
            ..Default::default()
        }
        .build(steps)
        .unwrap();
        assert_eq!(trailing.timesteps()[0], 999);
        assert_ne!(leading.timesteps(), trailing.timesteps());
    }
}
