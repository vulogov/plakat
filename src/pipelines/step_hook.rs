//! `StepHook` — a per-denoise-step callback for the samplers (RFC TUI-1 §0-R0-3).
//!
//! Before this, every denoise loop drove an `indicatif` progress bar directly and
//! had no way to (a) report progress to a non-terminal consumer, (b) be cancelled
//! mid-run, or (c) emit an intermediate preview to memory. The TUI (progressive
//! preview §15, `Ctrl-C` cancellation §14) needs all three.
//!
//! A `StepHook` is an OPTIONAL collaborator: `Pipeline::generate` keeps its exact
//! CLI behaviour (indicatif bar, file previews via `preview_every`) and passes
//! `None`; the TUI calls `generate_hooked` with `Some(hook)`. The hook is called
//! once per step. Returning [`StepControl::Cancel`] stops the sampler at the next
//! step boundary (never mid-step). When [`StepHook::wants_preview`] is true for a
//! step, the sampler hands the hook a cheap latent→RGB projection (microseconds,
//! the same projection the file-preview path uses — not a full VAE decode).

/// What the sampler should do after a step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepControl {
    /// Continue to the next denoise step.
    Continue,
    /// Stop the sampler at this step boundary; the partial latents are decoded
    /// and saved (the "use last preview as output" path).
    Cancel,
}

/// A per-step observer threaded through the denoise loops. `Send` so a hook can
/// be moved into the `spawn_blocking` generation worker.
pub trait StepHook: Send {
    /// Called once per denoise step with the 0-based `step` index and the `total`
    /// step count. Return [`StepControl::Cancel`] to abort at this boundary.
    fn on_step(&mut self, step: usize, total: usize) -> StepControl;

    /// Whether the hook wants an intermediate preview at this step. Default: never
    /// (the CLI uses the file-preview path, not this). When `true`, the sampler
    /// projects the current latent to RGB and calls [`on_preview`](Self::on_preview).
    fn wants_preview(&self, _step: usize, _total: usize) -> bool {
        false
    }

    /// Receives a cheap latent→RGB projection of the current step (only when
    /// `wants_preview` returned `true`). Reduced fidelity vs a VAE decode — it is
    /// the live-denoise preview, not the final image.
    fn on_preview(&mut self, _step: usize, _image: image::RgbImage) {}

    /// Whether cancellation has been requested. Lets a sampler whose denoise loop
    /// lives in a helper (returning partial latents on `Cancel`) tell its caller to
    /// stop the surrounding per-image loop too. Default `false` (the CLI never
    /// cancels); `ChannelHook` reads its shared flag.
    fn is_cancelled(&self) -> bool {
        false
    }

    /// LAYERED-1 §S3: called after the sampler has produced the latent for the next
    /// level, giving the hook a chance to STEER it (e.g. low-frequency substitution
    /// toward a guide latent). `space` is the family's [`NoiseSpace`] so the hook can
    /// forward-noise its guide to the current level. Return `Some(latent)` to replace
    /// the running latent; the default returns `None` (no change), so every existing
    /// hook is unaffected and layered-off output is byte-identical.
    fn refine_latent(
        &mut self,
        _step: usize,
        _total: usize,
        _space: &dyn crate::pipelines::noise_space::NoiseSpace,
        _latent: &candle_core::Tensor,
    ) -> candle_core::Result<Option<candle_core::Tensor>> {
        Ok(None)
    }
}

/// A hook that does nothing and never cancels — a stand-in where a `StepHook` is
/// needed but no observation is wanted.
pub struct NoopHook;

impl StepHook for NoopHook {
    fn on_step(&mut self, _step: usize, _total: usize) -> StepControl {
        StepControl::Continue
    }
}

/// `Option<&mut dyn StepHook>` is what the samplers actually take. These helpers
/// keep the call sites terse and tolerate the `None` (CLI) case.
pub(crate) fn step(hook: &mut Option<&mut dyn StepHook>, step: usize, total: usize) -> StepControl {
    match hook {
        Some(h) => h.on_step(step, total),
        None => StepControl::Continue,
    }
}

pub(crate) fn wants_preview(hook: &Option<&mut dyn StepHook>, step: usize, total: usize) -> bool {
    matches!(hook, Some(h) if h.wants_preview(step, total))
}

pub(crate) fn preview(hook: &mut Option<&mut dyn StepHook>, step: usize, image: image::RgbImage) {
    if let Some(h) = hook {
        h.on_preview(step, image);
    }
}

pub(crate) fn is_cancelled(hook: &Option<&mut dyn StepHook>) -> bool {
    matches!(hook, Some(h) if h.is_cancelled())
}

/// LAYERED-1: apply the hook's optional latent refinement at a denoise step, returning the (possibly
/// replaced) latent. Mirrors [`step`] — tolerant of the `None` (CLI) case, in which the latent is returned
/// unchanged, so every existing call site stays byte-identical until a `refine_latent` override is present.
// Wired into each family's denoise loop by the P0 call-site commit; kept `pub(crate)` and covered by a
// Tier-0 no-op test until then.
#[allow(dead_code)]
pub(crate) fn refine(
    hook: &mut Option<&mut dyn StepHook>,
    step: usize,
    total: usize,
    space: &dyn crate::pipelines::noise_space::NoiseSpace,
    latent: candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    match hook {
        Some(h) => Ok(h.refine_latent(step, total, space, &latent)?.unwrap_or(latent)),
        None => Ok(latent),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records every step it sees and can be told to cancel after N steps.
    struct Recorder {
        seen: Vec<(usize, usize)>,
        cancel_after: Option<usize>,
        previews: Vec<usize>,
        preview_every: usize,
    }
    impl Recorder {
        fn new() -> Self {
            Self { seen: vec![], cancel_after: None, previews: vec![], preview_every: 0 }
        }
    }
    impl StepHook for Recorder {
        fn on_step(&mut self, step: usize, total: usize) -> StepControl {
            self.seen.push((step, total));
            match self.cancel_after {
                Some(n) if step >= n => StepControl::Cancel,
                _ => StepControl::Continue,
            }
        }
        fn wants_preview(&self, step: usize, _total: usize) -> bool {
            self.preview_every > 0 && (step + 1) % self.preview_every == 0
        }
        fn on_preview(&mut self, step: usize, _image: image::RgbImage) {
            self.previews.push(step);
        }
    }

    /// Drive a hook the way a sampler would over `total` steps, honouring cancel.
    fn drive(hook: &mut dyn StepHook, total: usize) -> usize {
        let mut completed = 0;
        for s in 0..total {
            if hook.on_step(s, total) == StepControl::Cancel {
                break;
            }
            if hook.wants_preview(s, total) {
                hook.on_preview(s, image::RgbImage::new(1, 1));
            }
            completed += 1;
        }
        completed
    }

    #[test]
    fn noop_runs_all_steps() {
        assert_eq!(drive(&mut NoopHook, 10), 10);
    }

    #[test]
    fn records_every_step() {
        let mut r = Recorder::new();
        drive(&mut r, 5);
        assert_eq!(r.seen, vec![(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]);
    }

    #[test]
    fn cancel_stops_at_the_boundary() {
        let mut r = Recorder::new();
        r.cancel_after = Some(3);
        let completed = drive(&mut r, 10);
        // step 3 returns Cancel → steps 0,1,2 completed, loop breaks before 3.
        assert_eq!(completed, 3);
        assert_eq!(r.seen.last(), Some(&(3, 10)));
    }

    #[test]
    fn preview_fires_on_the_requested_cadence() {
        let mut r = Recorder::new();
        r.preview_every = 2;
        drive(&mut r, 6);
        assert_eq!(r.previews, vec![1, 3, 5]); // steps where (step+1) % 2 == 0
    }

    #[test]
    fn option_helpers_tolerate_none() {
        let mut none: Option<&mut dyn StepHook> = None;
        assert_eq!(step(&mut none, 0, 1), StepControl::Continue);
        assert!(!wants_preview(&none, 0, 1));
        preview(&mut none, 0, image::RgbImage::new(1, 1)); // no panic
    }

    /// LAYERED-1 Tier-0: with no `refine_latent` override, the `refine` helper returns the latent
    /// UNCHANGED — the guarantee that layered-off denoise loops are byte-identical.
    #[test]
    fn refine_is_a_noop_without_override() {
        use crate::pipelines::noise_space::{FlowSpace, LatentGeometry};
        let d = candle_core::Device::Cpu;
        let space = FlowSpace::new(vec![0.0], LatentGeometry { v: 8, u: 8, pool_levels: 2 });
        let latent = candle_core::Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &d).unwrap();
        // None hook (CLI path).
        let mut none: Option<&mut dyn StepHook> = None;
        let out = refine(&mut none, 0, 10, &space, latent.clone()).unwrap();
        let d0 = (&out - &latent).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d0, 0.0, "None hook leaves the latent unchanged");
        // A hook that overrides only on_step (default refine_latent) also leaves it unchanged.
        let mut r = Recorder::new();
        let mut some: Option<&mut dyn StepHook> = Some(&mut r);
        let out2 = refine(&mut some, 0, 10, &space, latent.clone()).unwrap();
        let d1 = (&out2 - &latent).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d1, 0.0, "default refine_latent leaves the latent unchanged");
    }
}
