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

    /// Emit a machine-readable JSON report (for CI gating) instead of text.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

pub async fn run(args: VerifyArgs) -> Result<()> {
    if !args.json {
        println!("plakat verify — correctness harness (RFC_VERIFY.md)\n");
    }
    crate::verify::run(&crate::verify::VerifyConfig { tier: args.tier, json: args.json })
}
