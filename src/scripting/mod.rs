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
pub mod script_entry;
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
    ///
    /// **Test isolation note**: ScriptCtx is a process-wide
    /// singleton (OnceLock) and can only be initialised once per
    /// process. All tests that need a context must serialise
    /// through one shared init — see `with_singleton_ctx` below.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn echo_round_trip() {
        with_singleton_ctx(|| {
            // The simplest script that exercises a host word: push a
            // string, call echo, drop the result.
            eval("\"hello from phase 1\" plakat.echo drop").unwrap();
        });
    }

    /// v0.21 phase 2 round-trip: eval a script that uses
    /// `plakat.save` against an image pre-stuffed into the context
    /// (so we don't have to run the SD pipeline in unit tests).
    /// Exercises the full pull/handle/save path the host word
    /// implements, end-to-end via `eval()`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn save_round_trip_via_eval() {
        with_singleton_ctx(|| {
            // Stuff a tiny image into the context + remember the
            // out_dir so we can assert the save landed.
            let (handle, out_dir) = with_ctx_mut(|ctx| {
                let img = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(4, 4, image::Rgb([1, 2, 3])),
                );
                (ctx.push_image(img), ctx.out_dir.clone())
            })
            .unwrap();

            // Script: push handle, push path, call save. The
            // out_dir prefix gets prepended for relative paths.
            let script = format!(
                "{handle} \"phase2-save-test.png\" plakat.save"
            );
            eval(&script).unwrap();

            assert!(out_dir.join("phase2-save-test.png").exists());
        });
    }

    /// Test-helper: serialises every test that needs the singleton
    /// context behind one shared init. Subsequent calls re-use the
    /// already-init'd singleton; only the *first* test through the
    /// gate pays the init cost. Each test runs its body inside the
    /// mutex so they don't trample each other's state on `out_dir`
    /// or `images`.
    fn with_singleton_ctx<R>(body: impl FnOnce() -> R) -> R {
        use std::sync::Mutex;
        static GATE: Mutex<()> = Mutex::new(());
        let _g = GATE.lock().unwrap();
        if ctx::CTX.get().is_none() {
            let tmp = tempfile::tempdir().unwrap();
            // Leak the tempdir to keep the path alive for the
            // singleton's lifetime — only a single tempdir per
            // process, harmless.
            let path = tmp.path().to_path_buf();
            std::mem::forget(tmp);
            ScriptCtx::init(Device::Cpu, path).unwrap();
        }
        body()
    }
}
