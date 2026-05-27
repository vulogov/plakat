//! `plakat.portrait ( prompt -- handle )`.
//!
//! Identity-preserving portrait via the cached pipeline's
//! IP-Adapter-Plus-Face identity encoder. v0.22 phase 1 picks
//! the identity at cache-load time based on the model alias:
//! SD 1.5 → PlusFace, SDXL → PlusFaceSdxl, SD 2.1 → no
//! identity (no shipped Plus-Face checkpoint — `plakat.portrait`
//! bails at generate time on sd21).
//!
//! **v0.24 phase 1 change**: photos no longer come in as a stack
//! arg. Push photos onto the dedicated stack with
//! `plakat.portrait.photo.add ( path-or-handle weight -- )`
//! before calling `plakat.portrait`. This is the
//! collection-namespace pattern shared by `plakat.lora.*` and
//! `plakat.controlnet.*`; it lets users build up multi-photo
//! identity blends with explicit per-photo weights.
//!
//! Single-photo migration from v0.23:
//!
//! ```bund
//! // v0.23:
//! "alice.jpg" plakat.portrait
//! // v0.24:
//! "alice.jpg" 1.0 plakat.portrait.photo.add
//! plakat.portrait
//! ```

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};
use crate::scripting::script_entry;

const TAG: &str = "plakat.portrait";

pub fn plakat_portrait(vm: &mut VM) -> BundResult<'_> {
    do_plakat_portrait(vm).map_err(to_bund_err)
}

fn do_plakat_portrait(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;

    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let img = script_entry::portrait_one(ctx, &prompt)?;
        Ok(ctx.push_image(img))
    })??;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int}"
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
