//! v0.21 phase 1: `plakat.echo`.
//!
//! ```bund
//! "hello" plakat.echo    \ pushes "[out=/tmp/foo] hello" onto the stack
//! ```
//!
//! Phase 1's whole point is to validate the integration shape with
//! the smallest meaningful host word. `plakat.echo`:
//!
//! * pulls one string off the stack ([`helpers::pull`] + [`value_to_string`])
//! * reads from the [`ScriptCtx`] singleton ([`with_ctx`])
//! * goes through the async bridge (`block_in_place` +
//!   `Handle::current().block_on`) even though the work itself is
//!   trivial — proves the pattern compiles and runs from a tokio
//!   worker thread
//! * pushes a new string back via [`helpers::push`]
//!
//! All four boilerplate points exercised. Phase 2 (`plakat.load` +
//! `plakat.generate` + `plakat.save`) copies this template and
//! swaps the trivial async body for a real pipeline call.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.echo";

pub fn plakat_echo(vm: &mut VM) -> BundResult<'_> {
    do_plakat_echo(vm).map_err(to_bund_err)
}

fn do_plakat_echo(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let msg_v = pull(vm, TAG)?;
    let msg = value_to_string(msg_v, "msg", TAG)?;

    // Read the script context. Phase 1 only uses `out_dir` to
    // demonstrate the singleton plumbing; later phases will read
    // device + loaded pipelines.
    let out_dir_str = with_ctx(|ctx| format!("{}", ctx.out_dir.display()))?;

    // The async bridge. Trivial body (`yield_now` is a noop
    // round-trip through the tokio scheduler), but the pattern is
    // exactly what phase 2's `plakat.generate` will use:
    //
    //   let result = block_in_place(|| handle.block_on(async {
    //       pipeline.generate(&request).await
    //   }));
    //
    // Bailing here if no tokio runtime is in scope makes the
    // multi-threaded-runtime requirement explicit at the call site
    // rather than at script time.
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: no tokio runtime in scope (eval must run on a \
             multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let echoed = tokio::task::block_in_place(|| {
        handle.block_on(async {
            tokio::task::yield_now().await;
            format!("[out={out_dir_str}] {msg}")
        })
    });

    push(vm, Value::from_string(echoed));
    Ok(vm)
}
