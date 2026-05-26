//! v0.21 phase 2: `plakat.generate ( prompt -- handle )`.
//!
//! Renders one image with the model recorded by `plakat.load`,
//! stores it in `ScriptCtx.images`, and pushes the 1-based
//! integer handle onto the stack so subsequent words
//! (`plakat.save`, future `plakat.upscale`) can address it.
//!
//! Bails if `plakat.load` hasn't been called first — better a
//! clear error than a default model surprising the user.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
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

    // Pull (model, device, config) out of the context up front so
    // we don't hold the read lock across the async/blocking work.
    // Clone config because generate_one borrows it across the
    // tokio block.
    let (model, device, config) = with_ctx(|ctx| {
        (
            ctx.loaded_model.clone(),
            ctx.device.clone(),
            ctx.config.clone(),
        )
    })?;
    let model = model.ok_or_else(|| {
        anyhow::anyhow!(
            "{TAG}: no model loaded. Call `\"sd15\" plakat.load` (or \
             your model of choice) before `plakat.generate`."
        )
    })?;

    // Async bridge. Pattern is identical to `plakat.echo` —
    // `cli::run::run` already runs us on a multi-threaded tokio
    // runtime, so `Handle::try_current()` always returns Ok here.
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: no tokio runtime in scope (eval must run on a \
             multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let img = tokio::task::block_in_place(|| {
        handle.block_on(script_entry::generate_one(&model, &prompt, device, &config))
    })?;

    let handle_int = with_ctx_mut(|ctx| ctx.push_image(img))?;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} via model {model:?}"
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
