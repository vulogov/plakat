//! Rich style-training progress, shared by the SD (`trainer.rs`) and SD 3.5
//! (`sd3.rs`) trainers. Three things the bare `step N/M loss X` line lacked:
//!
//! * **timing** (humantime) — elapsed, ETA, and per-step rate, so a long run is
//!   schedulable;
//! * **an EMA-smoothed loss** — flow-matching / DDPM loss is per-step *noise*
//!   (a random timestep each step); the EMA is the only readable signal;
//! * **a rough "imprint phase"** from effective learning (`lr × steps`) — style
//!   LoRAs over-cook, so this flags warming-up vs sweet-spot vs over-cook to
//!   help a continue/abort call. HEURISTIC ONLY — rendering a checkpoint is the
//!   sole real quality check.

use std::path::Path;
use std::time::{Duration, Instant};

pub(crate) struct TrainProgress {
    start: Instant,
    total: usize,
    lr: f64,
    ckpt_interval: usize,
    ema: Option<f32>,
}

impl TrainProgress {
    pub(crate) fn new(total: usize, lr: f64, ckpt_interval: usize) -> Self {
        Self {
            start: Instant::now(),
            total,
            lr,
            ckpt_interval: ckpt_interval.max(1),
            ema: None,
        }
    }

    /// One progress line for `step` (1-based steps completed) with raw `loss`.
    /// Updates the EMA. `tag` prefixes the line (e.g. `sd-style-train`).
    pub(crate) fn line(&mut self, tag: &str, step: usize, loss: f32) -> String {
        let ema = match self.ema {
            Some(e) => 0.2 * loss + 0.8 * e,
            None => loss,
        };
        self.ema = Some(ema);

        let done = step.max(1);
        let elapsed = self.start.elapsed();
        let per_step = elapsed / done as u32;
        let eta = per_step * self.total.saturating_sub(done) as u32;
        let pct = step * 100 / self.total.max(1);
        let phase = phase_label(self.lr * step as f64);

        let next = (step / self.ckpt_interval + 1) * self.ckpt_interval;
        let next = if next < self.total {
            format!(" · next ckpt @{next}")
        } else {
            String::new()
        };

        format!(
            "{}  {step}/{} ({pct}%) · {} elapsed, eta {} ({:.1}s/step) · loss {loss:.3} ema {ema:.3} · {phase}{next}",
            tag,
            self.total,
            secs(elapsed),
            secs(eta),
            per_step.as_secs_f64(),
        )
    }

    /// Closing summary. `tag` prefixes the line.
    pub(crate) fn finish(&self, tag: &str, out: &Path) -> String {
        let ckpts = self.total.saturating_sub(1) / self.ckpt_interval;
        let stem = out.file_name().and_then(|s| s.to_str()).unwrap_or("out");
        format!(
            "{}  done: {} steps in {} · {ckpts} numbered checkpoint(s) + {stem} — render each and pick the best (the last step is rarely it)",
            tag,
            self.total,
            secs(self.start.elapsed()),
        )
    }
}

/// Humantime-formatted duration rounded to whole seconds ("12m 4s").
fn secs(d: Duration) -> humantime::FormattedDuration {
    humantime::format_duration(Duration::from_secs(d.as_secs()))
}

/// Rough "imprint phase" from effective learning (`lr × steps`). The corpus
/// sweet spots land around `0.013–0.024` (sd15 lr 2e-4 ×120; sd35 lr 1.5e-4 ×90).
/// HEURISTIC — only a rendered checkpoint confirms quality.
fn phase_label(effective: f64) -> &'static str {
    if effective < 0.010 {
        "warming up (heuristic: likely under-baked)"
    } else if effective <= 0.035 {
        "sweet-spot window (heuristic: render a checkpoint)"
    } else {
        "over-cook risk (heuristic: likely past the peak)"
    }
}
