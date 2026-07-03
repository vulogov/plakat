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
pub mod manifest;
pub mod tier0;
pub mod tier1;

pub use capture::{CaptureBag, TensorTap};

use anyhow::Result;

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
    pub json: bool,
}

/// Execute verification. Returns `Ok(())` only when nothing failed.
pub fn run(cfg: &VerifyConfig) -> Result<()> {
    let mut report = Report::default();
    let want = |t: u8| cfg.tier.map(|x| x == t).unwrap_or(true);

    if want(0) {
        tier0::run(&mut report);
    }
    // Tier 1 — per-module correctness. The comparison engine is complete; capture-point
    // wiring (phase 1b) + hosted goldens (phase 2) make it verify real models.
    if want(1) {
        tier1::run(&mut report, cfg);
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
