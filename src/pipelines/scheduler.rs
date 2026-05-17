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
}

impl std::str::FromStr for SchedulerKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "default" => Self::Default,
            "ddim" => Self::Ddim,
            "euler-a" | "euler_a" | "eulera" | "euler-ancestral" | "euler" => Self::EulerA,
            "unipc" | "uni-pc" => Self::UniPc,
            "dpm++" | "dpm-solver++" | "dpmpp" | "dpmpp-2m" | "dpm++2m" | "dpmpp-karras" => {
                Self::DpmppKarras
            }
            "unipc-exp" | "unipc-exponential" => Self::UniPcExp,
            other => {
                return Err(anyhow!(
                    "unknown scheduler {other:?} (try: default | ddim | euler-a | unipc | \
                     dpmpp-2m | unipc-exp)"
                ));
            }
        })
    }
}

/// Refuse known-broken combinations early so the user gets a useful message
/// instead of a cryptic mid-inference error.
pub fn check_device_support(kind: SchedulerKind, device: &Device) -> Result<()> {
    if matches!(
        kind,
        SchedulerKind::UniPc | SchedulerKind::DpmppKarras | SchedulerKind::UniPcExp
    ) && device.is_metal()
    {
        return Err(anyhow!(
            "{kind:?} uses F64 ops candle's Metal backend doesn't implement \
             (all variants of the UniPC / DPM-Solver++ family share the same \
             scheduler class). Use --scheduler euler-a (works on Metal) or \
             --device cpu."
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
        // UniPC with corrector + DPM-Solver++ algorithm + Karras sigmas (default).
        SchedulerKind::UniPc => UniPCSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        }
        .build(steps)?,
        // DPM-Solver++ 2M Karras — corrector disabled, otherwise UniPC defaults.
        // (candle's `UniPCSchedulerConfig` doesn't expose an algorithm_type
        // toggle even though the AlgorithmType enum exists — the corrector
        // on/off is the meaningful behavior difference.)
        SchedulerKind::DpmppKarras => UniPCSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            corrector: CorrectorConfiguration::Disabled,
            sigma_schedule: SigmaSchedule::Karras(KarrasSigmaSchedule::default()),
            ..Default::default()
        }
        .build(steps)?,
        // UniPC with exponential sigma schedule (less common but sometimes useful).
        SchedulerKind::UniPcExp => UniPCSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            sigma_schedule: SigmaSchedule::Exponential(ExponentialSigmaSchedule::default()),
            ..Default::default()
        }
        .build(steps)?,
    })
}
