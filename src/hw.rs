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
}
