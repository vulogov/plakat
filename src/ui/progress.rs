//! Progress bars and spinners. A process-wide `MultiProgress` is shared by
//! every module so spinners from concurrent stages (hf downloads, model
//! loading, denoising loops) don't collide horizontally on the terminal.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static MULTI: OnceLock<MultiProgress> = OnceLock::new();

/// When set, all bars are hidden and `println` is a no-op. The `plakat ui` TUI
/// owns the terminal (raw mode + alternate screen), so any stray bar/println from
/// a background model load or denoise loop would scribble over the UI. The TUI
/// flips this on at startup; the CLI leaves it off.
static QUIET: AtomicBool = AtomicBool::new(false);

/// Suppress all terminal progress output process-wide (set by `plakat ui`).
pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

fn shared() -> &'static MultiProgress {
    MULTI.get_or_init(MultiProgress::new)
}

/// Add a new step-counted bar to the shared MultiProgress.
pub fn step_bar(total: u64, label: &str) -> ProgressBar {
    if is_quiet() {
        return ProgressBar::hidden();
    }
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

/// Print a line that won't be clobbered by active bars. On a TTY this routes
/// through the shared MultiProgress so the text lands above active bars and
/// the bars re-render below it. On a piped stream (where indicatif suppresses
/// everything), falls back to `println!` so log captures still see it.
pub fn println(msg: &str) {
    if is_quiet() {
        return;
    }
    if std::io::stderr().is_terminal() {
        let _ = shared().println(msg);
    } else {
        println!("{msg}");
    }
}

/// v0.16 phase 7: bytes-counted progress bar for file downloads.
/// Same shared MultiProgress as `step_bar` / `spinner`. `total` is
/// the content length in bytes; the label is rendered as the prefix.
pub fn bytes_bar(total: u64, label: &str) -> ProgressBar {
    if is_quiet() {
        return ProgressBar::hidden();
    }
    let pb = shared().add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:>12} [{bar:30.cyan/blue}] {bytes}/{total_bytes} {wide_msg} {elapsed_precise}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_prefix(label.to_string());
    pb
}

/// Add a new spinner to the shared MultiProgress.
pub fn spinner(msg: &str) -> ProgressBar {
    if is_quiet() {
        return ProgressBar::hidden();
    }
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
