//! The generation→UI message channel and the `ChannelHook` bridge (RFC TUI-1
//! §14–§15). A generation runs on a blocking worker; it talks to the TUI event
//! loop only through [`GenMessage`]s on an `mpsc` channel that the loop drains
//! non-blocking each tick. [`ChannelHook`] is the [`StepHook`] that turns the
//! samplers' per-step calls into `Progress`/`Preview` messages and reads a shared
//! cancellation flag — so the same denoise loop serves the CLI (indicatif) and the
//! TUI (this channel) with no sampler change.
//!
//! Lives in `pipelines` (not behind the `ui` feature) because it is generation
//! infrastructure with no `ratatui` dependency — pure `std`/`image`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use super::step_hook::{StepControl, StepHook};

/// A message from a running generation to the UI. `Progress`/`Preview` come from
/// the [`ChannelHook`] inside the sampler; `Done`/`Error` are sent by the worker
/// once `generate` returns.
pub enum GenMessage {
    /// One denoise step elapsed. `elapsed`/`steps_per_sec` are wall-clock since
    /// the hook started, for the status bar's ETA.
    Progress { step: u32, total: u32, elapsed: Duration, steps_per_sec: f32 },
    /// A cheap intermediate latent→RGB preview (reduced fidelity; live-denoise,
    /// not the final image). Only emitted on the configured cadence.
    Preview { step: u32, image: image::RgbImage },
    /// The generation finished. `cancelled` = a partial saved via `Ctrl-C`.
    Done { output: PathBuf, cancelled: bool },
    /// The generation failed.
    Error { message: String },
}

/// A shared, clonable cancellation flag. The UI sets it; the [`ChannelHook`] reads
/// it between steps. (An `Arc<AtomicBool>` rather than a new `tokio-util` dep —
/// the check is a single relaxed load per step.)
#[derive(Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }
    /// Request cancellation; the sampler stops at its next step boundary.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A [`StepHook`] that forwards progress + previews to a [`GenMessage`] channel and
/// honours a [`CancelFlag`]. Constructed by the worker for each generation.
pub struct ChannelHook {
    tx: Sender<GenMessage>,
    cancel: CancelFlag,
    /// Emit a `Preview` every N steps (0 = never). Mirrors `preview_every_n_steps`.
    preview_every: usize,
    started: Instant,
}

impl ChannelHook {
    pub fn new(tx: Sender<GenMessage>, cancel: CancelFlag, preview_every: usize) -> Self {
        Self { tx, cancel, preview_every, started: Instant::now() }
    }
}

impl StepHook for ChannelHook {
    fn on_step(&mut self, step: usize, total: usize) -> StepControl {
        let elapsed = self.started.elapsed();
        let secs = elapsed.as_secs_f32();
        let steps_per_sec = if secs > 0.0 { (step as f32 + 1.0) / secs } else { 0.0 };
        // A closed receiver (UI gone) is not fatal — the generation keeps running
        // and finishes; the worker discovers the closed channel on Done.
        let _ = self.tx.send(GenMessage::Progress {
            step: step as u32,
            total: total as u32,
            elapsed,
            steps_per_sec,
        });
        if self.cancel.is_cancelled() {
            StepControl::Cancel
        } else {
            StepControl::Continue
        }
    }

    fn wants_preview(&self, step: usize, _total: usize) -> bool {
        self.preview_every > 0 && (step + 1) % self.preview_every == 0
    }

    fn on_preview(&mut self, step: usize, image: image::RgbImage) {
        let _ = self.tx.send(GenMessage::Preview { step: step as u32, image });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    /// Drive a hook the way the sampler does, honouring cancel.
    fn drive(hook: &mut dyn StepHook, total: usize) -> usize {
        let mut done = 0;
        for s in 0..total {
            if hook.on_step(s, total) == StepControl::Cancel {
                break;
            }
            if hook.wants_preview(s, total) {
                hook.on_preview(s, image::RgbImage::new(1, 1));
            }
            done += 1;
        }
        done
    }

    #[test]
    fn forwards_progress_for_every_step() {
        let (tx, rx) = channel();
        let mut hook = ChannelHook::new(tx, CancelFlag::new(), 0);
        drive(&mut hook, 4);
        drop(hook); // close the sender so rx.iter() ends
        let progress: Vec<_> = rx
            .iter()
            .filter_map(|m| match m {
                GenMessage::Progress { step, total, .. } => Some((step, total)),
                _ => None,
            })
            .collect();
        assert_eq!(progress, vec![(0, 4), (1, 4), (2, 4), (3, 4)]);
    }

    #[test]
    fn cancel_flag_stops_at_the_boundary() {
        let (tx, _rx) = channel();
        let cancel = CancelFlag::new();
        let mut hook = ChannelHook::new(tx, cancel.clone(), 0);
        cancel.cancel();
        // First step already sees the flag → cancels immediately.
        assert_eq!(drive(&mut hook, 10), 0);
    }

    #[test]
    fn emits_preview_on_cadence() {
        let (tx, rx) = channel();
        let mut hook = ChannelHook::new(tx, CancelFlag::new(), 2);
        drive(&mut hook, 6);
        drop(hook);
        let previews: Vec<u32> = rx
            .iter()
            .filter_map(|m| match m {
                GenMessage::Preview { step, .. } => Some(step),
                _ => None,
            })
            .collect();
        assert_eq!(previews, vec![1, 3, 5]);
    }

    #[test]
    fn survives_a_closed_receiver() {
        let (tx, rx) = channel();
        let mut hook = ChannelHook::new(tx, CancelFlag::new(), 0);
        drop(rx); // UI went away
        // Sends fail silently; generation continues without panicking.
        assert_eq!(drive(&mut hook, 3), 3);
    }
}
