//! `plakat.load ( model-alias -- )`.
//!
//! Loads the SD-family pipeline for `model-alias` into the
//! `ScriptCtx` cache. Subsequent `plakat.generate` / `img2img` /
//! `portrait` calls reuse the cached pipeline.
//!
//! **v0.22 phase 1 change**: in v0.21, `plakat.load` just
//! recorded the alias and the per-image words triggered the
//! actual load. v0.22 makes `plakat.load` do the load now, so
//! the script's "load up front" cost is explicit + amortised
//! across all subsequent calls. Calling twice with the same
//! alias is a no-op (cache hit); calling with a different
//! alias drops the previous pipeline (RAII-freeing GPU memory)
//! and loads the new one.
//!
//! v0.21 compat is relaxed per RFC decision #7: the timing of
//! the load shifts but the user-visible behaviour ("the right
//! model gets used for subsequent words") is unchanged.

use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.load";

pub fn plakat_load(vm: &mut VM) -> BundResult<'_> {
    do_plakat_load(vm).map_err(to_bund_err)
}

fn do_plakat_load(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let alias_v = pull(vm, TAG)?;
    let alias = value_to_string(alias_v, "model", TAG)?;

    // v0.22 phase 2: unified family dispatch. ensure_loaded
    // picks SD-family vs Flux automatically; SD3 still bails
    // with the "phase 3" pointer.
    with_ctx_mut(|ctx| -> anyhow::Result<()> {
        ctx.ensure_loaded(&alias)?;
        Ok(())
    })??;
    tracing::info!(target: "plakat", "{TAG}: cached pipeline for {alias:?}");
    Ok(vm)
}
