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
        // v0.26 phase 8: build A1111-compatible metadata so
        // plakat.save / plakat.metadata.write can attach the
        // sidecar + tEXt chunk. Snapshot only the always-known
        // fields here — optional bits (loras, controls, etc.)
        // ship in v0.26.1.
        let model = ctx
            .loaded_model()
            .unwrap_or("unknown")
            .to_string();
        let scheduler = format!("{:?}", ctx.config.scheduler).to_lowercase();
        let (w, h) = (img.width(), img.height());
        let mut meta = crate::imaging::metadata::GenerationMetadata::new(
            prompt.clone(),
            model,
            ctx.config.seed.unwrap_or(0),
            ctx.config.steps,
            ctx.config.guidance,
            scheduler,
            w,
            h,
        );
        meta.negative = ctx.config.negative.clone();
        Ok(ctx.push_image_with_metadata(img, meta))
    })??;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int}"
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
