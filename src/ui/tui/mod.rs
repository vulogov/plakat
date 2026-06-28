//! `plakat ui` — the terminal user interface (RFC TUI-1).
//!
//! Lives under the existing `ui` module (CLI progress/logging) as `ui::tui`, gated
//! behind the default-on `ui` Cargo feature. This Phase-1 entry point wires the
//! `plakat ui` subcommand and the terminal graphics-capability check; the workspace
//! resolver, app/event loop, and screens land in subsequent Phase-1 increments.

pub mod app;
pub mod memory;
pub mod output;
pub mod screens;
pub mod services;
pub mod workspace;

use anyhow::Result;
use ratatui_image::picker::{Picker, ProtocolType};

/// Arguments for `plakat ui` (parsed by clap in the subcommand router).
#[derive(Debug, clap::Args)]
pub struct UiArgs {
    /// Workspace directory. Created (with a short wizard) if it has no
    /// `plakat-workspace.hjson`. Defaults to the nearest workspace at/above the
    /// current directory, else creates one here.
    #[arg(long)]
    pub workspace: Option<std::path::PathBuf>,

    /// Open directly to a screen: `chat` | `models` | `scenarios` | `history`
    /// | `lora` | `people` | `prompts` | `canvas`.
    #[arg(long)]
    pub screen: Option<String>,

    /// A scenario / prompt file to pre-load.
    pub file: Option<std::path::PathBuf>,
}

/// Entry point dispatched from the `plakat ui` subcommand. Detects terminal
/// graphics support (exiting cleanly with guidance if absent), then launches the
/// TUI. Phase 1: detection + a placeholder until the app loop lands.
pub fn run(args: UiArgs) -> Result<()> {
    use std::io::IsTerminal;
    let picker = check_terminal_support()?;
    // Resolve (or create, via the wizard) the workspace before any raw mode.
    let cwd = std::env::current_dir()?;
    let interactive = std::io::stdin().is_terminal();
    let ws = workspace::resolve_or_create(args.workspace, &cwd, interactive)?;
    // The TUI owns the terminal. (1) Reroute all indicatif progress (load, download,
    // the denoise pos/len bar, scenario runs) into a channel rendered in the Output
    // pane instead of the alternate screen; (2) redirect stderr to a per-workspace
    // log file so `tracing` / stray eprintln don't scribble over the UI.
    let progress_rx = crate::ui::progress::install_tui_sink();
    #[cfg(unix)]
    let _stderr_guard = StderrGuard::redirect_to(&ws.cache_dir().join("ui.log"));
    // The model thread loads on the app's existing multi-thread runtime; select the
    // default device (Metal/CUDA/CPU) up front so loads land on the GPU.
    let device = crate::device::select("auto")?;
    let rt = tokio::runtime::Handle::current();
    app::App::new(ws, picker, device, rt, progress_rx).run()
}

/// RAII guard that redirects process stderr (fd 2) to a file for the TUI's
/// lifetime — so `tracing` / stray `eprintln!` from background work land in a log
/// instead of corrupting the alternate screen. Restores the original stderr on drop.
#[cfg(unix)]
struct StderrGuard {
    saved_fd: i32,
}

#[cfg(unix)]
impl StderrGuard {
    fn redirect_to(path: &std::path::Path) -> Option<Self> {
        use std::os::unix::io::AsRawFd;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).ok()?;
        // SAFETY: dup/dup2/close on valid fds; STDERR keeps the file's open
        // description after dup2, so `file` may drop.
        let saved_fd = unsafe { libc::dup(libc::STDERR_FILENO) };
        if saved_fd < 0 {
            return None;
        }
        unsafe {
            libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO);
        }
        Some(Self { saved_fd })
    }
}

#[cfg(unix)]
impl Drop for StderrGuard {
    fn drop(&mut self) {
        // SAFETY: restore the saved stderr and release the dup.
        unsafe {
            libc::dup2(self.saved_fd, libc::STDERR_FILENO);
            libc::close(self.saved_fd);
        }
    }
}

/// Detect a usable pixel graphics protocol. Returns the `Picker` (used later to
/// render images) or a guidance error when only half-blocks / nothing is
/// available. `Picker` is `Copy`, so reading `protocol_type()` keeps it intact.
pub fn check_terminal_support() -> Result<Picker> {
    let picker = Picker::from_query_stdio().map_err(|_| no_graphics_error())?;
    match picker.protocol_type() {
        // Half-blocks is the no-real-graphics fallback — insufficient for plakat.
        ProtocolType::Halfblocks => Err(no_graphics_error()),
        ProtocolType::Kitty | ProtocolType::Iterm2 | ProtocolType::Sixel => Ok(picker),
    }
}

/// The "no graphics support" error, with the same guidance the RFC specifies.
fn no_graphics_error() -> anyhow::Error {
    let term = std::env::var("TERM_PROGRAM")
        .or_else(|_| std::env::var("TERM"))
        .unwrap_or_else(|_| "unknown".into());
    anyhow::anyhow!(
        "plakat ui requires a terminal with graphics support.\n\n\
         Supported terminals:\n\
         \x20 macOS:   Kitty, iTerm2, WezTerm, Ghostty\n\
         \x20 Linux:   Kitty, WezTerm, foot, any Sixel-capable terminal\n\
         \x20 SSH:     Kitty (via kitten ssh), iTerm2 (forwards TERM_PROGRAM)\n\n\
         Your terminal: {term} (no graphics protocol detected)\n\n\
         For image-free operation, use the CLI:\n\
         \x20 plakat generate \"your prompt\"\n\
         \x20 plakat scenario file.hjson"
    )
}
