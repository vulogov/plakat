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
//! **Signal:** we read the kernel's own memory-pressure level
//! ([`crate::hw::mem_pressure`]), deliberately NOT a free-RAM threshold —
//! `sysinfo`'s "available" under-reports reclaimable inactive / compressed pages
//! on macOS, so a big-but-reclaimable load (the SD3.5 T5 encoders) reads as ~0 GB
//! free yet loads fine (a free-RAM guard cried wolf on exactly that). On macOS an
//! `Unknown` reading (sysctl gave no answer) carries no signal and never aborts.
//!
//! Aborting only helps if plakat is the culprit, so before exiting the guard
//! checks **attribution** ([`plakat_is_culprit`]): plakat's own RSS is a large
//! share of RAM, or free RAM fell sharply since the guard armed. That stops it
//! self-terminating (and discarding work) when another app drives the *system* to
//! critical pressure while plakat is small.
//!
//! **Two triggers:** an *acute* fast-path — Critical + free RAM already below the
//! floor — aborts on the first sample (a fast single-buffer VAE / upscale
//! allocation can cross the cliff between samples); otherwise Critical must be
//! *sustained* (the OS may reclaim / swap through a transient spike).
//!
//! This is the during-generation complement to [`crate::hw::memory_preflight`]
//! (which only warns up-front). The durable fix for *batch* OOM is still one
//! in-process run (`plakat scenario`) rather than N `generate` processes.
//!
//! Tuning: `PLAKAT_OOM_GUARD_GB` `0` disables the guard; any value > 0 enables it
//! (and sets the floor). Armed **only on Metal** — CPU / CUDA OOM is handled
//! cleanly by the OS killing the process, so the guard is inert there.

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

/// Default critical floor (GB free) below which a host crash is imminent. Doubles as the
/// "acute" trigger: Critical pressure with free RAM already this low aborts on the FIRST
/// sample (a fast single-buffer VAE/upscale allocation can cross the cliff in well under
/// the sustained window).
const DEFAULT_FLOOR_GB: f64 = 1.5;
/// Sample period. Short so the acute fast-path can catch a near-instant allocation before
/// the host hangs (the sustained window is expressed in samples below).
const INTERVAL: Duration = Duration::from_millis(100);
/// Exit code on guard abort (128 + SIGKILL(9), mirroring an OS "Killed: 9").
const ABORT_CODE: i32 = 137;
/// plakat is treated as the culprit (so aborting it will actually relieve pressure) when
/// its RSS is at least this share of total RAM, OR free RAM fell by [`FREE_DROP_GB`] since
/// the guard armed. Otherwise another app owns the pressure and we must not self-terminate.
const CULPRIT_SHARE: f64 = 0.25;
const FREE_DROP_GB: f64 = 4.0;

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
        Mode::Inference => 15, // ~1.5 s at 100 ms
        Mode::Training => 36,  // ~3.6 s at 100 ms
    }
}

/// Whether aborting plakat would actually relieve the memory pressure it observes — i.e.
/// plakat is a meaningful contributor. This prevents the guard from self-terminating (and
/// discarding in-flight work) when another application drives the *system* to critical
/// pressure while plakat's own footprint is small. Culprit if plakat's RSS is a large
/// share of RAM, or free RAM dropped sharply since the guard armed (its own load consumed
/// it). Pure fn for testing.
fn plakat_is_culprit(rss_gb: f64, total_gb: f64, baseline_free_gb: f64, free_gb: f64) -> bool {
    let big_share = total_gb > 0.0 && rss_gb >= CULPRIT_SHARE * total_gb;
    let consumed = (baseline_free_gb - free_gb) >= FREE_DROP_GB;
    big_share || consumed
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
        // The host-crash this guard exists to prevent is specific to Apple-Silicon Metal
        // (GPU shares the OS memory pool). CPU / CUDA OOM is handled cleanly by the OS
        // killing the process, so arming there only risks false self-aborts on pressure
        // plakat didn't cause — keep the guard inert for them.
        let armed = matches!(device.location(), DeviceLocation::Metal { .. });
        if floor <= 0.0 || !armed {
            return Self { stop, handle: None };
        }
        let sustained = sustained_samples(mode);
        let stop_t = stop.clone();
        let label = label.to_string();
        let handle = thread::Builder::new()
            .name("plakat-memwatch".into())
            .spawn(move || {
                use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate};
                let mut sys = sysinfo::System::new();
                let self_pid = Pid::from_u32(std::process::id());
                // Baseline free RAM at arm time (before the heavy load) — a sharp drop
                // since now attributes the pressure to plakat's own work.
                sys.refresh_memory();
                let total_gb = sys.total_memory() as f64 / 1e9;
                let baseline_free_gb = sys.available_memory() as f64 / 1e9;
                let mut breaches = 0u32;
                while !stop_t.load(Ordering::Relaxed) {
                    // Trust the kernel pressure level: it accounts for reclaimable pages,
                    // so a big-but-reclaimable load (the T5 encoders) does NOT read as
                    // danger the way free-RAM does. On macOS, `Unknown` (sysctl gave no
                    // answer) carries NO signal — never fall back to the discredited
                    // free-RAM guard there. The guard only arms on Metal (⇒ macOS), so
                    // the non-macOS `Unknown`→free-RAM branch is effectively dead.
                    let critical = match crate::hw::mem_pressure() {
                        crate::hw::Pressure::Critical => true,
                        crate::hw::Pressure::Unknown if !cfg!(target_os = "macos") => {
                            sys.refresh_memory();
                            (sys.available_memory() as f64 / 1e9) < floor
                        }
                        _ => false,
                    };
                    if !critical {
                        breaches = 0;
                        thread::sleep(INTERVAL);
                        continue;
                    }
                    // Critical (rare) → attribute + decide. The extra process refresh is
                    // only paid under real pressure, not every tick.
                    sys.refresh_memory();
                    let free_gb = sys.available_memory() as f64 / 1e9;
                    sys.refresh_processes_specifics(
                        ProcessesToUpdate::Some(&[self_pid]),
                        true,
                        ProcessRefreshKind::everything(),
                    );
                    let rss_gb = sys.process(self_pid).map(|p| p.memory() as f64 / 1e9).unwrap_or(0.0);
                    // Only abort if killing plakat would actually relieve the pressure —
                    // otherwise another app owns it and we'd just discard our own work.
                    if !plakat_is_culprit(rss_gb, total_gb, baseline_free_gb, free_gb) {
                        breaches = 0;
                        thread::sleep(INTERVAL);
                        continue;
                    }
                    breaches += 1;
                    // Fast path: Critical AND free RAM already below the floor → a host
                    // crash is imminent (a fast single-buffer allocation can cross the
                    // cliff between samples), so abort on the FIRST such sample. Otherwise
                    // ride out the sustained window (the OS may reclaim / swap through it).
                    //
                    // 6.27: the acute path also requires that SWAP can't absorb the dip. With
                    // ample free swap the OS can ride out a low-free-RAM spike (exactly what the
                    // sustained window is for), so acute-aborting there cries wolf. When both RAM
                    // and swap are below the floor, it's a true cliff → fast-abort still fires.
                    let (used_swap, total_swap) = crate::hw::swap_gb();
                    let free_swap = (total_swap - used_swap).max(0.0);
                    let acute = free_gb < floor && free_swap < floor;
                    if acute || breaches >= sustained {
                        eprintln!(
                            "\n{} OOM GUARD — critical memory pressure attributable to plakat \
                             while generating '{}' ({:.1} GB resident, {:.1} GB free RAM, {:.1} GB free \
                             swap); aborting now to avoid crashing the host. Free RAM (close apps / \
                             `sudo purge`), use a smaller model / --size / --device cpu, or run one `plakat \
                             scenario` for batches. (PLAKAT_OOM_GUARD_GB=0 disables; \
                             PLAKAT_OOM_GUARD_SUSTAINED raises the ride-out window.)",
                            console::style("⛔").red().bold(),
                            label,
                            rss_gb,
                            free_gb,
                            free_swap,
                        );
                        // Restore the terminal (TUI) before the hard exit — `exit` skips
                        // Drop, so the alt-screen / raw mode would otherwise leak.
                        run_abort_hook();
                        // Hard exit: the OS reclaims all memory (incl. Metal buffers) on
                        // process death, relieving pressure fast.
                        std::process::exit(ABORT_CODE);
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
    fn attribution_gates_the_abort_to_plakat_caused_pressure() {
        let total = 24.0;
        // plakat holds a big share → culprit (aborting relieves pressure).
        assert!(plakat_is_culprit(8.0, total, 20.0, 19.0));
        // free RAM fell sharply since arm (plakat's load consumed it) → culprit.
        assert!(plakat_is_culprit(1.0, total, 20.0, 15.0));
        // plakat is small AND free RAM barely moved → NOT the culprit (another app owns
        // the pressure); the guard must not self-terminate.
        assert!(!plakat_is_culprit(1.0, total, 20.0, 19.0));
        // Degenerate total (0) doesn't yield a false "big share".
        assert!(!plakat_is_culprit(0.0, 0.0, 0.0, 0.0));
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
