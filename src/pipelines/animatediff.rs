//! v0.26 phase 5: AnimateDiff pipeline scaffolding.
//!
//! This module ties together the four phase-1-through-4 pieces:
//!
//! - [`super::motion_adapter::MotionAdapter`] — V3 weights + config
//! - [`super::motion_module::MotionAdapterModules`] — built temporal
//!   transformers per UNet block
//! - [`super::sd15_motion_unet::Sd15MotionUNet`] — vendored UNet
//!   with motion splice at block-output boundaries
//! - Motion LoRA composition via
//!   [`super::motion_adapter::MotionAdapter::load_v3_with_motion_loras`]
//!
//! The `AnimateDiffPipeline` type is the assembly point. Phase 5
//! ships the **scaffolding**:
//!
//! - The pipeline struct definition
//! - An async `load_v3` constructor that materializes the
//!   motion stack (downloads + assembles)
//! - A `generate(...)` method **stub** that bails with a clear
//!   v0.26.1 deferral message
//!
//! ## Why the inference dispatch is deferred
//!
//! Per [`Documentation/RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md`] §12,
//! phase 5 closeout is the documented cycle-cut decision point.
//! The infrastructure (phases 1-4) ships solid: motion adapter
//! loads, modules build, vendored UNet runs, motion LoRAs merge.
//! What's NOT yet in scope:
//!
//! 1. Loading SD 1.5 text encoder + VAE alongside the motion
//!    stack. Mostly straightforward (existing
//!    [`super::sd_core`] helpers cover it) but adds ~150 LOC
//!    of plumbing.
//! 2. N-frame scheduler loop — initialize `(N_FRAMES, 4, H/8, W/8)`
//!    latents at a shared seed; loop over scheduler timesteps;
//!    call `Sd15MotionUNet::forward_with_motion` per step;
//!    update latents. ~80 LOC.
//! 3. Per-frame VAE decode + image save loop. ~50 LOC.
//! 4. **Quality validation** — does block-boundary motion
//!    produce recognizable AnimateDiff output? Without 1-3
//!    landed, can't measure. This is the actual cycle-cut
//!    signal.
//!
//! v0.26.1 (or v0.27 if AnimateDiff slips per the cycle-cut)
//! finishes this. v0.26.0 ships every other carry from the
//! v0.25 list cleanly and bundles the AnimateDiff infrastructure
//! as "ready to wire."

use anyhow::Result;
use candle_core::Device;

use super::lora::LoraSpec;
use super::motion_adapter::MotionAdapter;
use super::motion_module::MotionAdapterModules;

/// Loaded AnimateDiff stack: motion adapter + per-block modules.
/// Phase 5 ships just the assembly; SD 1.5 components (text
/// encoder, VAE, vendored UNet) join in v0.26.1 / v0.27 alongside
/// the inference loop.
pub struct AnimateDiffPipeline {
    /// The downloaded V3 adapter weights + parsed config.
    /// Phase 6+ pipeline assembly takes ownership of this slot
    /// when it builds [`Sd15MotionUNet`].
    pub adapter: MotionAdapter,
    /// Per-UNet-block temporal transformers. 16 modules for V3
    /// SD 1.5 (8 down × 2 layers + 8 up × 2 layers, no mid).
    pub modules: MotionAdapterModules,
    /// Echoed back for caller convenience — same as
    /// `adapter.config.motion_max_seq_length`.
    pub max_frames: usize,
}

impl AnimateDiffPipeline {
    /// Load the AnimateDiff V3 stack with optional motion LoRAs.
    /// Network-required on first run (downloads ~1.4 GB motion
    /// adapter + each motion LoRA); cache-hits subsequently.
    ///
    /// `device` + `dtype` are passed through to the module
    /// construction. `motion_loras` may be empty.
    /// `motion_lora_scale` is the multiplier applied to every
    /// LoRA's per-spec scale at merge time (matches the
    /// `--lora-scale` convention from v0.17).
    pub async fn load_v3(
        device: &Device,
        dtype: candle_core::DType,
        motion_loras: &[LoraSpec],
        motion_lora_scale: f32,
    ) -> Result<Self> {
        let adapter = if motion_loras.is_empty() {
            MotionAdapter::load_v3().await?
        } else {
            MotionAdapter::load_v3_with_motion_loras(
                motion_loras,
                motion_lora_scale,
                device,
            )
            .await?
        };
        let modules = adapter.build_modules(device, dtype)?;
        let max_frames = adapter.config.motion_max_seq_length;
        Ok(Self {
            adapter,
            modules,
            max_frames,
        })
    }

    /// **DEFERRED to v0.26.1**: end-to-end AnimateDiff inference.
    ///
    /// When called today, this bails with a clear message
    /// pointing the user at the carries that DID ship in
    /// v0.26.0. Phase 5 scope cap per RFC §12 — the motion stack
    /// is fully loadable + tested, but the SD 1.5 component
    /// assembly + N-frame scheduler loop + VAE decode loop are
    /// queued for v0.26.1 (or v0.27 if cycle-cut invokes per the
    /// risk register).
    pub async fn generate(
        &self,
        _prompt: &str,
        _negative: &str,
        _frames: usize,
        _seed: u64,
        _width: u32,
        _height: u32,
    ) -> Result<()> {
        anyhow::bail!(
            "AnimateDiff inference dispatch lands in v0.26.1. v0.26.0 ships:\n\
             - Motion-adapter loader (phase 1)\n\
             - Temporal-attention modules (phase 2)\n\
             - Vendored SD 1.5 UNet with motion splice (phase 3)\n\
             - Motion LoRA composition (phase 4)\n\
             - --animatediff + --motion-lora + --format CLI flags (phase 5)\n\
             - SD3/SD3.5 animate (phase 6)\n\
             - All v0.25 carries (phases 7-12)\n\n\
             See RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md §12 for the\n\
             cycle-cut decision. Track v0.26.1 release notes for\n\
             the inference-dispatch closure."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Network-required end-to-end test: builds the full
    /// AnimateDiff V3 stack, asserts 16 motion modules + the
    /// expected max-frames value. Cost ~1.4 GB on first run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    #[ignore]
    async fn load_v3_full_stack() {
        let pipeline = AnimateDiffPipeline::load_v3(
            &Device::Cpu,
            candle_core::DType::F32,
            &[],
            1.0,
        )
        .await
        .expect("load V3 stack");
        assert_eq!(pipeline.modules.modules.len(), 16);
        assert_eq!(pipeline.max_frames, 32);
    }

    // Note: the `generate()` deferral message is verified by the
    // integration smoke through `cli::animate::run` (phase 5
    // dispatch + CLI test). A unit test here would need to
    // hand-construct an AnimateDiffPipeline, but MotionAdapter
    // has a private field (tensor_layout) by design — the only
    // legitimate constructor is `load_v3` which is network-
    // required. Testing the deferral via the CLI dispatch is the
    // more honest signal anyway.
}
