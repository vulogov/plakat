//! `plakat.generate ( prompt -- handle )`.
//!
//! Renders one image with the cached pipeline (set by
//! `plakat.load`), stores it in `ScriptCtx.images`, and pushes
//! the 1-based integer handle onto the stack.
//!
//! v0.22 phase 1: the cached pipeline is reused across calls,
//! so consecutive `plakat.generate` invocations pay zero
//! model-load cost.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};
use crate::scripting::script_entry;

const TAG: &str = "plakat.generate";

pub fn plakat_generate(vm: &mut VM) -> BundResult<'_> {
    do_plakat_generate(vm).map_err(to_bund_err)
}

fn do_plakat_generate(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;

    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let img = script_entry::generate_one(ctx, &prompt)?;
        Ok(ctx.push_image(img))
    })??;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int}"
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
