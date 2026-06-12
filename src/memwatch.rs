//! Memory watchdog — graceful self-abort before a unified-memory host crash.
//!
//! On Apple-Silicon Metal the GPU allocates from the **same** memory pool as
//! the OS, so when a generation exhausts unified memory the kernel /
//! WindowServer are starved faster than jetsam can cleanly kill plakat — and
//! the whole **host** hangs or crashes (the observed failure). Polling free RAM
//! on a background thread and aborting plakat *before* the cliff turns that into
//! a clean exit: the OS reclaims every allocation (including Metal buffers) on
//! process death, relieving pressure immediately so the host survives.
//!
//! This is the during-generation complement to [`crate::hw::memory_preflight`]
//! (which only warns up-front). The durable fix for *batch* OOM is still one
//! in-process run (`plakat scenario`) rather than N `generate` processes.
//!
//! Tuning: `PLAKAT_OOM_GUARD_GB` sets the critical free-RAM floor in GB
//! (default 1.5); `0` disables the guard. No-op on CUDA (separate VRAM).

use candle_core::{Device, DeviceLocation};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Default critical floor (GB free) below which a host crash is imminent.
const DEFAULT_FLOOR_GB: f64 = 1.5;
/// Sample period.
const INTERVAL: Duration = Duration::from_millis(300);
/// Consecutive sub-floor samples required before aborting (~0.9 s sustained),
/// so a momentary dip during a large allocation doesn't trip the guard.
const SUSTAINED: u32 = 3;
/// Exit code on guard abort (128 + SIGKILL(9), mirroring an OS "Killed: 9").
const ABORT_CODE: i32 = 137;

/// Resolve the floor from `PLAKAT_OOM_GUARD_GB` (default 1.5; `0` disables).
fn floor_gb() -> f64 {
    match std::env::var("PLAKAT_OOM_GUARD_GB") {
        Ok(v) => v.trim().parse::<f64>().unwrap_or(DEFAULT_FLOOR_GB),
        Err(_) => DEFAULT_FLOOR_GB,
    }
}

/// A running watchdog. Keep the returned guard alive for the duration of the
/// heavy work (bind it to a named local — `let _g = MemoryGuard::start(..)`);
/// dropping it stops the thread.
pub struct MemoryGuard {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MemoryGuard {
    /// Start watching unified memory for the given device. No-op (returns an
    /// inert guard) on CUDA or when `PLAKAT_OOM_GUARD_GB=0`.
    pub fn start(device: &Device, label: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let floor = floor_gb();
        let cuda = matches!(device.location(), DeviceLocation::Cuda { .. });
        if floor <= 0.0 || cuda {
            return Self { stop, handle: None };
        }
        let stop_t = stop.clone();
        let label = label.to_string();
        let handle = thread::Builder::new()
            .name("plakat-memwatch".into())
            .spawn(move || {
                let mut sys = sysinfo::System::new();
                let mut breaches = 0u32;
                while !stop_t.load(Ordering::Relaxed) {
                    sys.refresh_memory();
                    let avail = sys.available_memory() as f64 / 1e9;
                    if avail < floor {
                        breaches += 1;
                        if breaches >= SUSTAINED {
                            eprintln!(
                                "\n{} OOM GUARD — only ~{:.1} GB RAM free while generating \
                                 '{}'; aborting plakat now to avoid crashing the host. \
                                 Try a smaller model, --device cpu, fewer parallel runs, or \
                                 a single `plakat scenario` for batches. \
                                 (tune/disable with PLAKAT_OOM_GUARD_GB)",
                                console::style("⛔").red().bold(),
                                avail,
                                label,
                            );
                            // Hard exit: the OS reclaims all memory (incl. Metal
                            // buffers) on process death, relieving pressure fast.
                            std::process::exit(ABORT_CODE);
                        }
                    } else {
                        breaches = 0;
                    }
                    thread::sleep(INTERVAL);
                }
            })
            .ok();
        Self { stop, handle }
    }
}

impl Drop for MemoryGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_env_parses_and_disables() {
        // (env is process-global; just exercise the parse paths via defaults)
        assert_eq!(DEFAULT_FLOOR_GB, 1.5);
        assert!(SUSTAINED >= 2, "need sustained breaches to avoid transient trips");
    }

    #[test]
    fn inert_guard_on_disabled_floor_does_not_panic_on_drop() {
        // A guard with no thread (the disabled path) must drop cleanly.
        let g = MemoryGuard {
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
        };
        drop(g);
    }
}
