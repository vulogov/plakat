//! v0.21: `plakat run SCRIPT.bund` — evaluate a Bund script with
//! the `plakat.*` host words registered.
//!
//! Phase 1 ships file evaluation only; the `--repl` flag (decision
//! #6, RFC §8) lands in phase 7 once the host-word surface is
//! stable. Until then `--repl` parses but bails up front with a
//! clear "not yet wired" message so the CLI shape is fixed early.

use anyhow::{Context, Result, anyhow};
use candle_core::Device;
use clap::Args;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to a `.bund` script file. Required when `--repl` is
    /// off; ignored when `--repl` is on.
    #[arg(value_name = "SCRIPT", required_unless_present = "repl")]
    pub script: Option<PathBuf>,

    /// Output directory passed to host words that produce images
    /// (`plakat.save`, the implicit auto-saves in `plakat.generate`).
    /// Defaults to `./out`.
    #[arg(long, default_value = "./out")]
    pub out: PathBuf,

    /// v0.21 phase 7 placeholder. Parses but bails up front today.
    #[arg(long, default_value_t = false)]
    pub repl: bool,
}

pub async fn run(args: RunArgs, device: Device) -> Result<()> {
    if args.repl {
        anyhow::bail!(
            "`plakat run --repl` lands in v0.21 phase 7 (RFC \
             §9). For now use `plakat run SCRIPT.bund`."
        );
    }

    let script = args
        .script
        .as_ref()
        .ok_or_else(|| anyhow!("expected SCRIPT path (clap usually enforces this)"))?;

    crate::scripting::ScriptCtx::init(device, args.out.clone())
        .with_context(|| "initialising script context")?;

    crate::scripting::eval_file(script)
        .with_context(|| format!("running script {}", script.display()))?;

    Ok(())
}
