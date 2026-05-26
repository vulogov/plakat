//! v0.21 phase 2: `plakat.load ( model-alias -- )`.
//!
//! Records the model alias the script wants to use. Subsequent
//! `plakat.generate` calls render against this model. Idempotent
//! by design — calling twice with the same alias is a no-op;
//! calling with a different alias overwrites the previous one.
//!
//! Phase 2 doesn't preload the pipeline; the alias is just
//! stored, and `t2i::run` does its own load on every `generate`
//! call. Phase 4 (`plakat.img2img`) will likely introduce a
//! pipeline cache to avoid paying the load cost three times in
//! a row. For now, scripts that need throughput can stick to
//! `cli::generate` directly.

use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, require_depth, to_bund_err, value_to_string,
};
use crate::scripting::script_entry;

const TAG: &str = "plakat.load";

pub fn plakat_load(vm: &mut VM) -> BundResult<'_> {
    do_plakat_load(vm).map_err(to_bund_err)
}

fn do_plakat_load(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let alias_v = pull(vm, TAG)?;
    let alias = value_to_string(alias_v, "model", TAG)?;
    script_entry::validate_supported_for_phase_2(&alias)?;
    with_ctx_mut(|ctx| {
        ctx.loaded_model = Some(alias.clone());
    })?;
    tracing::info!(target: "plakat", "{TAG}: loaded model {alias:?}");
    Ok(vm)
}
