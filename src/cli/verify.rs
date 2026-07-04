//! `plakat verify` — run the model-correctness harness (RFC_VERIFY.md).
//!
//! Phase 0 ships Tier 0 (structural / determinism, zero downloads). Higher tiers report as
//! skipped until their golden data lands. The command is pure Rust — no python/diffusers.

use anyhow::Result;

#[derive(clap::Args, Debug)]
pub struct VerifyArgs {
    /// Run only this tier: 0 = structural/determinism (no downloads),
    /// 1 = per-module correctness, 2 = end-to-end perceptual. Omit for all applicable.
    #[arg(long)]
    pub tier: Option<u8>,

    /// Restrict Tier 1+ correctness checks to a single model alias (e.g. `sd15`).
    /// Omit to cover the pilot set.
    #[arg(long)]
    pub model: Option<String>,

    /// Local golden source for Tier 1 (authored by `tools/reference/dump.py`), laid out as
    /// `<dir>/<model>/<fixture>/{manifest.json, goldens.safetensors}`. Omit to only report
    /// coverage (Tier 1 loads models + compares only when this is set).
    #[arg(long)]
    pub golden_dir: Option<std::path::PathBuf>,

    /// Device for Tier 1 model loads: `auto` (default), `metal`, `cuda`, `cpu`.
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Emit a machine-readable JSON report (for CI gating) instead of text.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

pub async fn run(args: VerifyArgs) -> Result<()> {
    if !args.json {
        println!("plakat verify — correctness harness (RFC_VERIFY.md)\n");
    }
    let device = crate::device::select(&args.device)?;
    crate::verify::run(&crate::verify::VerifyConfig {
        tier: args.tier,
        model: args.model,
        golden_dir: args.golden_dir,
        device,
        json: args.json,
    })
    .await
}
