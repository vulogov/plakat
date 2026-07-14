//! Filesystem watcher (RFC PHOTOS-1 §23). A background `notify` watcher signals the UI when image
//! files are created / removed / renamed anywhere under the library root; the event loop coalesces
//! the signals (500 ms debounce) and rescans. Keeps the library live as generations / card imports
//! land, without a manual refresh.

use std::path::Path;
use std::sync::mpsc::{channel, Receiver};

use anyhow::{Context, Result};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use super::library;

/// A live watch: the receiver ticks `()` on each relevant change. Hold `_watcher` alive for the
/// duration of the watch (dropping it stops the notifications).
pub struct Watch {
    pub rx: Receiver<()>,
    _watcher: RecommendedWatcher,
}

/// Start watching `root` recursively. Emits `()` on create/remove/rename/modify of a supported
/// image file (HJSON / sidecar churn is ignored). No in-thread debounce — the caller coalesces.
pub fn spawn(root: &Path) -> Result<Watch> {
    let (tx, rx) = channel();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(ev) = res {
            let relevant = matches!(
                ev.kind,
                EventKind::Create(_) | EventKind::Remove(_) | EventKind::Modify(_)
            );
            if relevant && ev.paths.iter().any(|p| library::is_image(p)) {
                let _ = tx.send(());
            }
        }
    })
    .context("creating filesystem watcher")?;
    watcher
        .watch(root, RecursiveMode::Recursive)
        .with_context(|| format!("watching {}", root.display()))?;
    Ok(Watch { rx, _watcher: watcher })
}
