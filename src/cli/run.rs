//! v0.21: `plakat run SCRIPT.bund` — evaluate a Bund script with
//! the `plakat.*` host words registered.
//!
//! Two modes:
//!   * **File** (default) — `plakat run path.bund` reads + evals
//!     the file as a single string. Exits on completion.
//!   * **REPL** (v0.21 phase 7) — `plakat run --repl` starts an
//!     interactive line editor against a persistent Bund. Stack
//!     state, named lambdas, and `plakat.config.set` knobs all
//!     survive across lines. See [`crate::scripting::repl`].

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
    #[arg(help_heading = "Size & output", long, default_value = "./out")]
    pub out: PathBuf,

    /// v0.21 phase 7: start an interactive REPL instead of evaling
    /// a file. The `SCRIPT` positional is ignored when this is on.
    #[arg(long, default_value_t = false)]
    pub repl: bool,
}

pub async fn run(args: RunArgs, device: Device) -> Result<()> {
    crate::scripting::ScriptCtx::init(device, args.out.clone())
        .with_context(|| "initialising script context")?;

    if args.repl {
        // The REPL is interactive + blocking — we want it to own
        // the calling thread until the user quits. `spawn_blocking`
        // would let the rest of the runtime keep going, but plakat
        // CLI is one-shot anyway and the REPL is the foreground
        // work, so running it directly is fine.
        return crate::scripting::repl::run();
    }

    let script = args
        .script
        .as_ref()
        .ok_or_else(|| anyhow!("expected SCRIPT path (clap usually enforces this)"))?;

    crate::scripting::eval_file(script)
        .with_context(|| format!("running script {}", script.display()))?;

    Ok(())
}
