//! `plakat verify` — model-correctness harness (RFC_VERIFY.md).
//!
//! **Phase 0** lands the framework + **Tier 0**: structural / determinism invariants that
//! need NO external data (no weights, no network) — so `plakat verify` runs green offline
//! and in CI. Higher tiers (per-module correctness vs golden tensors from HF; end-to-end
//! perceptual) arrive in later phases.
//!
//! Self-containment invariant: this is pure Rust. It never shells out to python/diffusers;
//! its only external touch (in later tiers) is Hugging Face, exactly like model weights.

pub mod capture;
pub mod compare;
pub mod fixtures;
pub mod golden;
pub mod manifest;
pub mod tier0;
pub mod tier1;

pub use capture::{CaptureBag, TensorTap};

use anyhow::Result;

/// Build a DETERMINISTIC latent `(1, channels, lh, lw)` via a tiny LCG — NOT a seeded RNG.
/// candle's and torch's RNGs diverge for the same seed, so a seeded latent can't correspond
/// between plakat and the diffusers golden. This pure-integer generator is reproduced
/// byte-for-byte in `tools/reference/fixtures.py::deterministic_latent`, so both sides decode
/// the SAME input. Values land in `[-1, 1)`.
pub fn deterministic_latent(
    channels: usize,
    lh: usize,
    lw: usize,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<candle_core::Tensor> {
    deterministic_tensor(&[1, channels, lh, lw], 1, device, dtype)
}

/// Shared LCG value stream — the byte-identical source both `deterministic_latent` and any
/// deterministic conditioning input (caption / context) draw from, mirrored on the Python
/// side by `fixtures.deterministic_tensor`. Values in [-1, 1). `seed` selects an independent
/// stream (latent = 1) so distinct inputs to the same forward don't alias.
pub fn deterministic_tensor(
    dims: &[usize],
    seed: u64,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<candle_core::Tensor> {
    let n: usize = dims.iter().product();
    let mut x: u64 = seed;
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        x = (x.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff_ffff;
        vals.push((x % 2000) as f32 / 1000.0 - 1.0);
    }
    Ok(candle_core::Tensor::from_vec(vals, dims.to_vec(), device)?.to_dtype(dtype)?)
}

/// Outcome of a single check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    /// Not run (e.g. a tier that needs downloads, when only Tier 0 was requested).
    Skip,
}

impl Status {
    fn glyph(self) -> &'static str {
        match self {
            Status::Pass => "✓",
            Status::Fail => "✗",
            Status::Skip => "–",
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "fail",
            Status::Skip => "skip",
        }
    }
}

/// One verification check's result.
#[derive(Clone, Debug)]
pub struct Check {
    /// Stable identifier, e.g. `tier0.cfg_batch_layout` or `tier1.sd15.clip_l.penultimate`.
    pub name: String,
    pub tier: u8,
    pub status: Status,
    /// One-line human detail (what was checked / why it failed).
    pub detail: String,
}

impl Check {
    pub fn pass(name: impl Into<String>, tier: u8, detail: impl Into<String>) -> Self {
        Self { name: name.into(), tier, status: Status::Pass, detail: detail.into() }
    }
    pub fn fail(name: impl Into<String>, tier: u8, detail: impl Into<String>) -> Self {
        Self { name: name.into(), tier, status: Status::Fail, detail: detail.into() }
    }
    pub fn skip(name: impl Into<String>, tier: u8, detail: impl Into<String>) -> Self {
        Self { name: name.into(), tier, status: Status::Skip, detail: detail.into() }
    }
    /// From a `Result<()>` producer: `Ok` → pass with `ok_detail`, `Err` → fail with the error.
    pub fn from_result(name: impl Into<String>, tier: u8, ok_detail: &str, r: Result<()>) -> Self {
        match r {
            Ok(()) => Check::pass(name, tier, ok_detail),
            Err(e) => Check::fail(name, tier, format!("{e:#}")),
        }
    }
}

/// A run's accumulated results.
#[derive(Default)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    pub fn push(&mut self, c: Check) {
        self.checks.push(c);
    }
    pub fn any_failed(&self) -> bool {
        self.checks.iter().any(|c| c.status == Status::Fail)
    }
    fn counts(&self) -> (usize, usize, usize) {
        let mut p = 0;
        let mut f = 0;
        let mut s = 0;
        for c in &self.checks {
            match c.status {
                Status::Pass => p += 1,
                Status::Fail => f += 1,
                Status::Skip => s += 1,
            }
        }
        (p, f, s)
    }

    /// Human-readable summary to stdout.
    fn emit_text(&self) {
        for c in &self.checks {
            println!("  {} [tier {}] {} — {}", c.status.glyph(), c.tier, c.name, c.detail);
        }
        let (p, f, s) = self.counts();
        println!("\n{p} passed · {f} failed · {s} skipped");
    }

    /// Machine-readable JSON (for CI gating). Hand-rolled to avoid pulling serde into this
    /// path; the shape is stable: `{ "summary": {...}, "checks": [ {...} ] }`.
    fn emit_json(&self) {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");
        let (p, f, s) = self.counts();
        let mut out = String::new();
        out.push_str(&format!(
            "{{\n  \"summary\": {{ \"passed\": {p}, \"failed\": {f}, \"skipped\": {s} }},\n  \"checks\": [\n"
        ));
        for (i, c) in self.checks.iter().enumerate() {
            let comma = if i + 1 < self.checks.len() { "," } else { "" };
            out.push_str(&format!(
                "    {{ \"name\": \"{}\", \"tier\": {}, \"status\": \"{}\", \"detail\": \"{}\" }}{comma}\n",
                c.name,
                c.tier,
                c.status.as_str(),
                esc(&c.detail),
            ));
        }
        out.push_str("  ]\n}");
        println!("{out}");
    }
}

/// What `plakat verify` runs.
pub struct VerifyConfig {
    /// Run only this tier (0/1/2); `None` = all applicable.
    pub tier: Option<u8>,
    /// Restrict Tier 1+ to a single model alias; `None` = the pilot set.
    pub model: Option<String>,
    /// Local golden source: `<dir>/<model>/<fixture>/{manifest.json, goldens.safetensors}`.
    /// When set, Tier 1 loads the model and actually compares; when `None`, it reports
    /// coverage (skips). HF-dataset fetch replaces this default in a later phase.
    pub golden_dir: Option<std::path::PathBuf>,
    /// Device Tier 1 loads models on (Metal/CUDA/CPU).
    pub device: candle_core::Device,
    pub json: bool,
}

/// Execute verification. Returns `Ok(())` only when nothing failed.
pub async fn run(cfg: &VerifyConfig) -> Result<()> {
    let mut report = Report::default();
    let want = |t: u8| cfg.tier.map(|x| x == t).unwrap_or(true);

    if want(0) {
        tier0::run(&mut report);
    }
    // Tier 1 — per-module correctness. Load each model + compare its captured intermediates
    // against goldens from the local `--golden-dir` or (default) the HF dataset. Missing
    // goldens / un-instrumented families skip cleanly.
    if want(1) {
        let src = cfg.golden_dir.as_deref();
        for model in tier1::models(cfg) {
            for c in tier1::run_model(&model, src, &cfg.device).await {
                report.push(c);
            }
        }
    }
    // Tier 2 — end-to-end perceptual gate (phase 3).
    if want(2) {
        report.push(Check::skip(
            "tier2.end_to_end_perceptual",
            2,
            "golden-image comparison not yet available (RFC_VERIFY phase 3)",
        ));
    }

    if cfg.json {
        report.emit_json();
    } else {
        report.emit_text();
    }

    if report.any_failed() {
        anyhow::bail!("verification failed ({} check(s))", report.counts().1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_latent_is_reproducible_shaped_and_bounded() {
        let dev = candle_core::Device::Cpu;
        let a = deterministic_latent(4, 8, 8, &dev, candle_core::DType::F32).unwrap();
        let b = deterministic_latent(4, 8, 8, &dev, candle_core::DType::F32).unwrap();
        assert_eq!(a.dims(), &[1, 4, 8, 8]);
        let (va, vb) = (a.flatten_all().unwrap().to_vec1::<f32>().unwrap(), b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        assert_eq!(va, vb, "same LCG → identical latent (the cross-language contract)");
        assert!(va.iter().all(|&v| (-1.0..1.0).contains(&v)), "values in [-1, 1)");
        // First LCG value: x = (1*1103515245 + 12345) & 0x7fffffff = 1103527590;
        // 1103527590 % 2000 = 1590 → 1590/1000 - 1 = 0.59. Pins the Python mirror.
        assert!((va[0] - 0.59).abs() < 1e-6, "first value pins the shared LCG: {}", va[0]);
    }
}
