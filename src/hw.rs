//! Hardware probe for `plakat doctor --capability`.
//!
//! Detects the system's text-to-image-relevant hardware: total RAM, the
//! active backend (Metal / CUDA / CPU), a **conservative** usable memory
//! budget for a model's weights + activations, CPU cores, and OS/arch.
//!
//! Budget heuristic (per the design decision): on unified-memory backends
//! (Apple-Silicon Metal, or CPU) the GPU shares system RAM, so we assume
//! only `RESERVE_FRACTION` of total RAM is realistically usable for a model
//! (the rest goes to the OS, the framework, and headroom). CUDA VRAM is not
//! yet queried directly — the RAM-based figure is a proxy there (flagged in
//! the report).
use candle_core::{Device, DeviceLocation};
use serde::Serialize;

/// Fraction of total RAM we treat as usable for model weights + activations
/// on a unified-memory machine. Deliberately conservative.
const RESERVE_FRACTION: f64 = 0.75;

#[derive(Debug, Clone, Serialize)]
pub struct HardwareReport {
    /// Active backend: `metal` | `cuda` | `cpu`.
    pub backend: String,
    /// GPU features compiled into this binary (`metal`, `cuda`).
    pub features: Vec<String>,
    /// Total system RAM (GB, decimal).
    pub total_ram_gb: f64,
    /// Conservative usable budget for a model (GB) = total RAM × 0.75 on
    /// unified-memory backends.
    pub budget_gb: f64,
    /// True when `budget_gb` is a RAM-based proxy rather than real VRAM
    /// (i.e. CUDA, where we don't yet query device memory).
    pub budget_is_proxy: bool,
    /// Logical CPU cores.
    pub cpu_cores: usize,
    pub os: String,
    pub arch: String,
    /// Coarse memory tier the box lands in: `8 GB` | `16 GB` | `32 GB` |
    /// `64 GB+`.
    pub tier: String,
}

/// Probe the host for the given (already-selected) device.
pub fn probe(device: &Device) -> HardwareReport {
    let total_ram_gb = total_ram_bytes() as f64 / 1e9;
    let (backend, budget_is_proxy) = match device.location() {
        DeviceLocation::Cpu => ("cpu", false),
        DeviceLocation::Metal { .. } => ("metal", false),
        DeviceLocation::Cuda { .. } => ("cuda", true),
    };
    let mut features = Vec::new();
    if cfg!(feature = "metal") {
        features.push("metal".to_string());
    }
    if cfg!(feature = "cuda") {
        features.push("cuda".to_string());
    }
    HardwareReport {
        backend: backend.to_string(),
        features,
        total_ram_gb,
        budget_gb: total_ram_gb * RESERVE_FRACTION,
        budget_is_proxy,
        cpu_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        tier: tier_for(total_ram_gb).to_string(),
    }
}

fn total_ram_bytes() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory() // bytes (sysinfo >= 0.30)
}

/// RAM the OS reports as available for new allocations right now (GB). Unlike
/// `total_memory`, this reflects current pressure from other processes / file
/// cache — the figure that actually predicts an OOM kill of a fresh load.
pub fn available_ram_gb() -> f64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory() as f64 / 1e9
}

/// Coarse memory-pressure level. On macOS this is the **kernel's own** signal
/// (`kern.memorystatus_vm_pressure_level`) which — unlike "free RAM" — already
/// accounts for the reclaimable inactive / compressed / cached pages the OS
/// frees to satisfy a load. (sysinfo's `available_memory` under-reports these on
/// macOS, so a healthy idle box can read ~0 GB "free" with GBs reclaimable —
/// that mismatch is what made an earlier free-RAM guard cry wolf.) Elsewhere it
/// is `Unknown`; callers fall back to a free-RAM figure, which is accurate on
/// Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    Normal,
    Warning,
    Critical,
    Unknown,
}

#[cfg(target_os = "macos")]
pub fn mem_pressure() -> Pressure {
    let mut val: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let rc = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&mut val as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Pressure::Unknown;
    }
    // dispatch memory-pressure flags: 1 = normal, 2 = warn, 4 = critical.
    match val {
        v if v >= 4 => Pressure::Critical,
        2 | 3 => Pressure::Warning,
        1 => Pressure::Normal,
        _ => Pressure::Unknown,
    }
}

#[cfg(not(target_os = "macos"))]
pub fn mem_pressure() -> Pressure {
    Pressure::Unknown
}

/// Pre-load memory preflight (recommendation #3). Large diffusion weights need
/// several–tens of GB resident; on a box already under pressure the load is
/// silently killed by the OS ("Killed: 9"). This converts that into an
/// actionable, up-front **warning** — never fatal.
///
/// Uses the kernel pressure level on macOS (so it does NOT cry wolf when
/// "available" looks low but GBs are reclaimable — the common idle state); on
/// other platforms it falls back to a free-RAM floor. No-op on CUDA and when
/// `PLAKAT_NO_PREFLIGHT` is set. The durable fix for batch OOM is one in-process
/// run (`plakat scenario`), not this check.
pub fn memory_preflight(device: &Device, model_label: &str) {
    if std::env::var_os("PLAKAT_NO_PREFLIGHT").is_some() {
        return;
    }
    if matches!(device.location(), DeviceLocation::Cuda { .. }) {
        return;
    }
    let warn = |reason: String| {
        eprintln!(
            "{} {reason} — loading '{}' may be killed by the OS (\"Killed: 9\"). \
             Free up memory (close apps / `sudo purge`), pick a smaller model, or use \
             --device cpu. For batches, run one `plakat scenario` (loads the model once) \
             instead of N separate `generate` calls. Silence with PLAKAT_NO_PREFLIGHT=1.",
            console::style("⚠").yellow().bold(),
            model_label,
        );
    };
    match mem_pressure() {
        Pressure::Critical => warn("system under CRITICAL memory pressure".to_string()),
        Pressure::Warning => warn("system under memory pressure".to_string()),
        Pressure::Normal => {} // healthy — low "available" on macOS is not a risk
        Pressure::Unknown => {
            // Linux / older: free RAM is a sound proxy here.
            let total = total_ram_bytes() as f64 / 1e9;
            let avail = available_ram_gb();
            let floor = 6.0_f64.min(total * 0.5);
            if avail < floor {
                warn(format!("low free RAM: ~{avail:.1} GB of {total:.0} GB"));
            }
        }
    }
}

fn tier_for(ram_gb: f64) -> &'static str {
    match ram_gb.round() as u64 {
        0..=9 => "8 GB",
        10..=20 => "16 GB",
        21..=40 => "32 GB",
        _ => "64 GB+",
    }
}

impl HardwareReport {
    /// One-line tier label for the budget, e.g. `16 GB unified · ~12 GB budget`.
    pub fn budget_label(&self) -> String {
        format!(
            "{:.0} GB {} · ~{:.0} GB budget{}",
            self.total_ram_gb,
            if self.backend == "cuda" { "RAM" } else { "unified" },
            self.budget_gb,
            if self.budget_is_proxy {
                " (RAM proxy — verify against VRAM)"
            } else {
                ""
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_cpu_is_sane() {
        let r = probe(&Device::Cpu);
        assert_eq!(r.backend, "cpu");
        assert!(r.total_ram_gb > 0.5, "should detect some RAM");
        assert!((r.budget_gb - r.total_ram_gb * 0.75).abs() < 1e-6);
        assert!(r.cpu_cores >= 1);
        assert!(!r.tier.is_empty());
    }

    #[test]
    fn tiers_bucket() {
        assert_eq!(tier_for(8.0), "8 GB");
        assert_eq!(tier_for(16.0), "16 GB");
        assert_eq!(tier_for(24.0), "32 GB");
        assert_eq!(tier_for(128.0), "64 GB+");
    }

    #[test]
    fn mem_pressure_returns_a_known_variant() {
        // Must not panic and must yield a defined level. On macOS this exercises
        // the sysctl path; elsewhere it's Unknown.
        let p = mem_pressure();
        assert!(matches!(
            p,
            Pressure::Normal | Pressure::Warning | Pressure::Critical | Pressure::Unknown
        ));
        #[cfg(not(target_os = "macos"))]
        assert_eq!(p, Pressure::Unknown);
    }
}
