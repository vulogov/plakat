//! v0.21: Bund scripting language integration.
//!
//! Embeds [`bundcore`] so users can drive plakat from a stack-based
//! script (`.bund` file) instead of (or alongside) the CLI
//! subcommands. The script invokes host words namespaced under
//! `plakat.*` which call into the same pipelines `plakat generate`
//! / `img2img` / `portrait` / `upscale` use.
//!
//! Architectural decisions are locked in
//! `Documentation/RFC_v0.21_BUND_SCRIPTING.md`. The relevant ones
//! for this module:
//!
//! * **Embed crate**: `bundcore = "=0.7.0"` — pinned exact.
//!   The `bund` binary crate has no `[lib]` target; `bundcore`
//!   carries the parser + `eval()` wrapper.
//! * **Stdlib strategy**: build our own VM via [`VM::new`] and
//!   register **only** `plakat.*` words. Skip `Bund::new()` so
//!   filesystem / network / shell / sudo stdlib words can't reach
//!   user scripts by construction.
//! * **State sharing**: [`VMInlineFn`] is a bare `fn` pointer, not a
//!   closure. Host words reach plakat state through a process-wide
//!   [`OnceLock<RwLock<ScriptCtx>>`] singleton (see [`ctx`]).
//! * **Async bridge**: bundcore is fully synchronous; plakat
//!   pipelines are `async`. Each host word that hits a pipeline does
//!   `tokio::task::block_in_place(|| Handle::current().block_on(...))`.
//!   [`plakat_run`] must therefore execute on a multi-threaded
//!   tokio runtime — `cli::run` dispatches that way already.

use anyhow::{Context, Result, anyhow};
use bundcore::bundcore::Bund;
use std::path::Path;

pub mod ctx;
pub mod helpers;
pub mod words;

pub use ctx::{ScriptCtx, with_ctx, with_ctx_mut};

/// Build a fresh plakat-flavoured Bund instance.
///
/// `Bund::new()` calls `init_lib()` which iterates `bundcore::STDLIB`
/// — a `lazy_static! Mutex<HashMap>` populated only by the `bund`
/// binary crate (which plakat does **not** depend on). With no
/// stdlib registrants, `init_lib()` is a no-op and the resulting VM
/// holds only the `rust_multistackvm` primitives (arithmetic, stack
/// ops, lambdas, control flow). We then register only `plakat.*`
/// words. Net effect of decision #2 (RFC §8): no filesystem,
/// network, shell, or sudo stdlib words can reach user scripts by
/// construction.
pub fn build_plakat_bund() -> Result<Bund> {
    let mut bund = Bund::new();
    words::register_plakat_words(&mut bund.vm)
        .map_err(|e| anyhow!("registering plakat host words: {e}"))?;
    Ok(bund)
}

/// Evaluate a plakat-flavoured Bund script string. Caller is
/// responsible for [`ScriptCtx::init`] before invoking; host words
/// look the context up unconditionally and will bail loud if it's
/// missing.
pub fn eval(source: &str) -> Result<()> {
    let mut bund = build_plakat_bund()?;
    bund.eval(source)
        .map_err(|e| anyhow!("bund eval failed: {e}"))?;
    Ok(())
}

/// Convenience: read `path` and eval its contents.
pub fn eval_file(path: &Path) -> Result<()> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading script {}", path.display()))?;
    eval(&source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// v0.21 phase 1 smoke test: VM build + register + eval a
    /// trivial script that exercises `plakat.echo`. Verifies the
    /// entire integration shape end-to-end:
    ///
    /// * `bundcore` linked + parses
    /// * `VM::new()` returns a usable VM
    /// * `register_plakat_words` registers without collision
    /// * `ScriptCtx::init` + the singleton are reachable from a
    ///   host fn
    /// * The async bridge inside `plakat.echo` resolves (we drive
    ///   eval from a multi-threaded tokio runtime here, matching
    ///   the production CLI dispatch)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn echo_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        ScriptCtx::init(Device::Cpu, tmp.path().to_path_buf()).unwrap();
        // The simplest script that exercises a host word: push a
        // string, call echo, drop the result.
        eval("\"hello from phase 1\" plakat.echo drop").unwrap();
    }
}
