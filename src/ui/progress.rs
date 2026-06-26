//! Progress bars and spinners. A process-wide `MultiProgress` is shared by every
//! module so concurrent stages (hf downloads, model loading, denoise loops) don't
//! collide on the terminal.
//!
//! In the `plakat ui` TUI the same `MultiProgress` is *rerouted* into a channel via
//! [`install_tui_sink`]: instead of drawing to the terminal (which the TUI owns),
//! indicatif renders into [`ChannelTerm`], and the TUI shows the captured lines in
//! its "Output" pane. This means EVERY pipeline's progress — load, download, the
//! denoise `step_bar` (a real `pos/len` bar), scenario runs — appears in the UI
//! with zero per-pipeline instrumentation. The CLI is unchanged.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle, TermLike};
use std::io::{self, IsTerminal};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

static MULTI: OnceLock<MultiProgress> = OnceLock::new();
static SINK: Mutex<Option<Sender<String>>> = Mutex::new(None);

/// Reroute the shared progress into a channel (the TUI calls this once, before any
/// bar is created) and return the receiver to drain each event-loop tick. After
/// this, all bars/spinners/`println` render into the channel, not the terminal.
pub fn install_tui_sink() -> Receiver<String> {
    let (tx, rx) = channel();
    *SINK.lock().unwrap() = Some(tx);
    let _ = shared(); // force-init the MultiProgress with the term_like target now
    rx
}

fn tui_mode() -> bool {
    SINK.lock().unwrap().is_some()
}

fn shared() -> &'static MultiProgress {
    MULTI.get_or_init(|| match SINK.lock().unwrap().clone() {
        Some(tx) => MultiProgress::with_draw_target(ProgressDrawTarget::term_like(Box::new(
            ChannelTerm { tx: Mutex::new(tx) },
        ))),
        None => MultiProgress::new(),
    })
}

/// An indicatif `TermLike` that forwards rendered lines to the TUI sink instead of
/// a terminal. Cursor moves / clears are no-ops; each non-empty, ANSI-stripped line
/// (a bar frame, a spinner frame, a `println`) is sent to the channel. The TUI
/// dedupes consecutive same-label frames so a live bar updates in place.
#[derive(Debug)]
struct ChannelTerm {
    tx: Mutex<Sender<String>>,
}

impl ChannelTerm {
    fn emit(&self, s: &str) {
        let clean = console::strip_ansi_codes(s).trim().to_string();
        if !clean.is_empty() {
            if let Ok(tx) = self.tx.lock() {
                let _ = tx.send(clean);
            }
        }
    }
}

impl TermLike for ChannelTerm {
    fn width(&self) -> u16 {
        100
    }
    fn move_cursor_up(&self, _n: usize) -> io::Result<()> {
        Ok(())
    }
    fn move_cursor_down(&self, _n: usize) -> io::Result<()> {
        Ok(())
    }
    fn move_cursor_right(&self, _n: usize) -> io::Result<()> {
        Ok(())
    }
    fn move_cursor_left(&self, _n: usize) -> io::Result<()> {
        Ok(())
    }
    fn write_line(&self, s: &str) -> io::Result<()> {
        self.emit(s);
        Ok(())
    }
    fn write_str(&self, s: &str) -> io::Result<()> {
        self.emit(s);
        Ok(())
    }
    fn clear_line(&self) -> io::Result<()> {
        Ok(())
    }
    fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

/// Add a new step-counted bar to the shared MultiProgress (the denoise loops use
/// this — it renders a real `[bar] pos/len` that the TUI captures verbatim).
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

/// Print a line above the active bars. In TUI mode this routes through the shared
/// MultiProgress (→ the capture channel); on a CLI TTY it lands above the bars; on
/// a piped stream it falls back to `println!`.
pub fn println(msg: &str) {
    if tui_mode() {
        let _ = shared().println(msg);
    } else if std::io::stderr().is_terminal() {
        let _ = shared().println(msg);
    } else {
        println!("{msg}");
    }
}

/// v0.16 phase 7: bytes-counted progress bar for file downloads.
pub fn bytes_bar(total: u64, label: &str) -> ProgressBar {
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
