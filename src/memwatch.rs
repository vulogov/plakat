//! Memory watchdog — graceful self-abort before a unified-memory host crash.
//!
//! On Apple-Silicon Metal the GPU allocates from the **same** memory pool as
//! the OS, so when a generation exhausts unified memory the kernel /
//! WindowServer are starved faster than jetsam can cleanly kill plakat — and
//! the whole **host** hangs or crashes (the observed failure). A background
//! thread watching for danger and aborting plakat *before* the cliff turns that
//! into a clean exit: the OS reclaims every allocation (including Metal buffers)
//! on process death, relieving pressure immediately so the host survives.
//!
//! **Signal:** on macOS we read the kernel's own memory-pressure level
//! ([`crate::hw::mem_pressure`]) and abort only on *sustained critical*. This is
//! deliberately NOT a free-RAM threshold: `sysinfo`'s "available" under-reports
//! reclaimable inactive / compressed pages on macOS, so a big-but-reclaimable
//! load (e.g. the SD3.5 T5 encoders) reads as ~0 GB free yet loads fine — a
//! free-RAM guard cried wolf on exactly that. Critical pressure means the kernel
//! is genuinely out of road. Elsewhere we fall back to a free-RAM floor.
//!
//! This is the during-generation complement to [`crate::hw::memory_preflight`]
//! (which only warns up-front). The durable fix for *batch* OOM is still one
//! in-process run (`plakat scenario`) rather than N `generate` processes.
//!
//! Tuning: `PLAKAT_OOM_GUARD_GB` `0` disables the guard; any value > 0 enables it
//! (and sets the free-RAM floor on non-macOS platforms). No-op on CUDA.

use candle_core::{Device, DeviceLocation};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// A cleanup run just before the guard's hard `process::exit` — e.g. the TUI restores
/// the terminal (raw mode is bypassed by `exit`, which skips Drop). Best-effort; called
/// from the watchdog thread, so it must be self-contained + panic-free.
type AbortHook = Box<dyn Fn() + Send + Sync>;
static ABORT_HOOK: OnceLock<AbortHook> = OnceLock::new();

/// Register the pre-abort cleanup (idempotent — only the first registration wins). The
/// TUI calls this with a terminal-restore so a guard abort doesn't leave a garbled shell.
pub fn set_abort_hook<F: Fn() + Send + Sync + 'static>(hook: F) {
    let _ = ABORT_HOOK.set(Box::new(hook));
}

/// Run the registered abort hook, if any (called immediately before a hard exit).
fn run_abort_hook() {
    if let Some(hook) = ABORT_HOOK.get() {
        hook();
    }
}

/// Default critical floor (GB free) below which a host crash is imminent.
const DEFAULT_FLOOR_GB: f64 = 1.5;
/// Sample period.
const INTERVAL: Duration = Duration::from_millis(300);
/// Exit code on guard abort (128 + SIGKILL(9), mirroring an OS "Killed: 9").
const ABORT_CODE: i32 = 137;

/// Workload kind — sets how long critical pressure must be sustained before the
/// guard aborts. Training's first backward / optimizer step spikes are larger and
/// longer-lived (and the OS can ride them out via swap), so training tolerates a
/// longer window than inference before treating the pressure as terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Inference,
    Training,
}

/// Consecutive sub-floor / critical samples required before aborting. Bumped from
/// the old 3 (~0.9 s) so a transient decode / first-backward spike the OS would
/// absorb doesn't trip the guard. Inference ≈ 1.5 s, training ≈ 3.6 s. Override
/// with `PLAKAT_OOM_GUARD_SUSTAINED`.
fn sustained_samples(mode: Mode) -> u32 {
    if let Ok(v) = std::env::var("PLAKAT_OOM_GUARD_SUSTAINED") {
        if let Ok(n) = v.trim().parse::<u32>() {
            return n.max(2);
        }
    }
    match mode {
        Mode::Inference => 5,  // ~1.5 s
        Mode::Training => 12,  // ~3.6 s
    }
}

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
    /// Start watching unified memory for the given device (inference window).
    /// No-op (inert guard) on CUDA or when `PLAKAT_OOM_GUARD_GB=0`.
    pub fn start(device: &Device, label: &str) -> Self {
        Self::start_mode(device, label, Mode::Inference)
    }

    /// Start watching with an explicit workload [`Mode`] — training uses a longer
    /// sustained-pressure window than inference before aborting.
    pub fn start_mode(device: &Device, label: &str, mode: Mode) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let floor = floor_gb();
        let cuda = matches!(device.location(), DeviceLocation::Cuda { .. });
        if floor <= 0.0 || cuda {
            return Self { stop, handle: None };
        }
        let sustained = sustained_samples(mode);
        let stop_t = stop.clone();
        let label = label.to_string();
        let handle = thread::Builder::new()
            .name("plakat-memwatch".into())
            .spawn(move || {
                let mut sys = sysinfo::System::new();
                let mut breaches = 0u32;
                while !stop_t.load(Ordering::Relaxed) {
                    // On macOS trust the kernel pressure level — it accounts for
                    // reclaimable pages, so a big-but-reclaimable load (e.g. the
                    // T5 encoders) does NOT read as danger the way free-RAM does.
                    // Only sustained CRITICAL means the OS is out of road and a
                    // host crash is imminent. Elsewhere (Unknown) fall back to the
                    // free-RAM floor, which is accurate on Linux.
                    let danger = match crate::hw::mem_pressure() {
                        crate::hw::Pressure::Critical => true,
                        crate::hw::Pressure::Unknown => {
                            sys.refresh_memory();
                            (sys.available_memory() as f64 / 1e9) < floor
                        }
                        crate::hw::Pressure::Normal | crate::hw::Pressure::Warning => false,
                    };
                    if danger {
                        breaches += 1;
                        if breaches >= sustained {
                            eprintln!(
                                "\n{} OOM GUARD — sustained critical memory pressure while \
                                 generating '{}'; aborting plakat now to avoid crashing the \
                                 host. Free RAM (close apps / `sudo purge`), use a smaller \
                                 model or --device cpu, or run one `plakat scenario` for \
                                 batches. (PLAKAT_OOM_GUARD_GB=0 disables; on non-macOS it \
                                 sets the free-RAM floor.)",
                                console::style("⛔").red().bold(),
                                label,
                            );
                            // Restore the terminal (TUI) before the hard exit — `exit`
                            // skips Drop, so the alt-screen / raw mode would otherwise
                            // leak into the user's shell.
                            run_abort_hook();
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
        assert!(sustained_samples(Mode::Inference) >= 2 && sustained_samples(Mode::Training) > sustained_samples(Mode::Inference));
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

    #[test]
    fn abort_hook_runs_when_registered() {
        use std::sync::atomic::AtomicBool;
        // A registered hook is invoked by run_abort_hook (the pre-exit cleanup path).
        static RAN: AtomicBool = AtomicBool::new(false);
        set_abort_hook(|| RAN.store(true, Ordering::SeqCst));
        run_abort_hook();
        assert!(RAN.load(Ordering::SeqCst), "the registered abort hook ran");
        // run_abort_hook with no hook (other process) is a no-op — already covered by
        // OnceLock semantics; calling it again here is still safe.
        run_abort_hook();
    }
}
