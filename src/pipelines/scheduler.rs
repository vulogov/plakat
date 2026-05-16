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
    schedulers::{PredictionType, Scheduler, SchedulerConfig},
    uni_pc::UniPCSchedulerConfig,
};

#[derive(Clone, Copy, Debug, Default)]
pub enum SchedulerKind {
    /// Use the SD variant's built-in default (DDIM for SD 1.5/2.1/SDXL,
    /// Euler-Ancestral for SDXL-Turbo).
    #[default]
    Default,
    Ddim,
    EulerA,
    UniPc,
}

impl std::str::FromStr for SchedulerKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "default" => Self::Default,
            "ddim" => Self::Ddim,
            "euler-a" | "euler_a" | "eulera" | "euler-ancestral" | "euler" => Self::EulerA,
            "unipc" | "uni-pc" | "dpm++" | "dpm-solver++" | "dpmpp" => Self::UniPc,
            other => {
                return Err(anyhow!(
                    "unknown scheduler {other:?} (try: default | ddim | euler-a | unipc)"
                ));
            }
        })
    }
}

/// Refuse known-broken combinations early so the user gets a useful message
/// instead of a cryptic mid-inference error.
pub fn check_device_support(kind: SchedulerKind, device: &Device) -> Result<()> {
    if matches!(kind, SchedulerKind::UniPc) && device.is_metal() {
        return Err(anyhow!(
            "UniPC / DPM++ uses F64 ops candle's Metal backend doesn't implement. \
             Use --scheduler euler-a (works on Metal) or --device cpu."
        ));
    }
    Ok(())
}

/// Build a Scheduler for `steps` steps. PredictionType=Epsilon is the trained
/// type for SD 1.5 / 2.1 / SDXL / SDXL-Turbo; we don't yet expose v-prediction.
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
        SchedulerKind::UniPc => UniPCSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?,
    })
}
