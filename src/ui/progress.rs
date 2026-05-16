//! Progress bars and spinners. A process-wide `MultiProgress` is shared by
//! every module so spinners from concurrent stages (hf downloads, model
//! loading, denoising loops) don't collide horizontally on the terminal.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::OnceLock;
use std::time::Duration;

static MULTI: OnceLock<MultiProgress> = OnceLock::new();

fn shared() -> &'static MultiProgress {
    MULTI.get_or_init(MultiProgress::new)
}

/// Add a new step-counted bar to the shared MultiProgress.
pub fn step_bar(total: u64, label: &str) -> ProgressBar {
    let pb = shared().add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:>12} [{bar:30.cyan/blue}] {pos}/{len} {wide_msg} {elapsed_precise}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_prefix(label.to_string());
    pb
}

/// Add a new spinner to the shared MultiProgress.
pub fn spinner(msg: &str) -> ProgressBar {
    let pb = shared().add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(Duration::from_millis(80));
    pb.set_message(msg.to_string());
    pb
}
