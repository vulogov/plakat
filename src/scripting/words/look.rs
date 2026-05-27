//! v0.25 phase 8: `plakat.look.*` host words.
//!
//! ```bund
//! "watercolor" plakat.look.apply         // pick by name
//! plakat.look.list                       // ( -- l_1 ... l_n n )
//! plakat.look.clear                      // forget the active look
//! ```
//!
//! State lives on [`ScriptCtx`] as `look_name`. The actual apply
//! happens lazily inside [`script_entry::generate_one`] (and the
//! img2img / portrait equivalents) — same split-of-concerns as
//! `plakat.style.*`. Trigger words from discovered LoRAs are
//! prepended to the prompt via the dedup-aware
//! `style::prepend_trigger`.
//!
//! **Cache invalidation**: `.apply` and `.clear` drop the
//! cached SD pipeline slots via `ctx.mark_loras_changed`. A new
//! look may trigger auto-LoRA discovery which would push a fresh
//! LoRA onto the stack at load time — the v0.22 invalidation
//! pattern applies.
//!
//! **Discovery**: gated by `ctx.config.offline_discovery`
//! (`plakat.config.set "offline_discovery" "true"`). Default
//! online. Discovery fires only when `ctx.loras` is empty at
//! generate time — user-passed LoRAs always win.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

// ---- plakat.look.apply ( name -- ) -------------------------------

const APPLY_TAG: &str = "plakat.look.apply";

pub fn plakat_look_apply(vm: &mut VM) -> BundResult<'_> {
    do_plakat_look_apply(vm).map_err(to_bund_err)
}

fn do_plakat_look_apply(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, APPLY_TAG)?;
    let name_v = pull(vm, APPLY_TAG)?;
    let name = value_to_string(name_v, "name", APPLY_TAG)?;
    if name.is_empty() {
        anyhow::bail!("{APPLY_TAG}: look name can't be empty");
    }
    // Validate against the bundled catalog at set-time so typos
    // surface immediately, not at generate.
    let cat = crate::preset::Catalog::load_default(crate::preset::Kind::Look)
        .map_err(|e| anyhow::anyhow!("{APPLY_TAG}: loading look catalog: {e}"))?;
    if cat.find(&name).is_none() {
        anyhow::bail!(
            "{APPLY_TAG}: unknown look {name:?} (try one of: {})",
            cat.names().join(", ")
        );
    }
    with_ctx_mut(|ctx| {
        ctx.look_name = Some(name.clone());
        // Discovery may push a fresh LoRA at next generate.
        ctx.mark_loras_changed();
    })?;
    tracing::info!(target: "plakat", "{APPLY_TAG}: look_name = {name:?}");
    Ok(vm)
}

// ---- plakat.look.clear ( -- ) ------------------------------------

const CLEAR_TAG: &str = "plakat.look.clear";

pub fn plakat_look_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_look_clear(vm).map_err(to_bund_err)
}

fn do_plakat_look_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let was_set = with_ctx_mut(|ctx| {
        let was = ctx.look_name.is_some();
        ctx.look_name = None;
        if was {
            ctx.mark_loras_changed();
        }
        was
    })?;
    tracing::info!(
        target: "plakat",
        "{CLEAR_TAG}: look state cleared (was active: {was_set})"
    );
    Ok(vm)
}

// ---- plakat.look.list ( -- l_1 ... l_n n ) -----------------------

const LIST_TAG: &str = "plakat.look.list";

pub fn plakat_look_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_look_list(vm).map_err(to_bund_err)
}

fn do_plakat_look_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let cat = crate::preset::Catalog::load_default(crate::preset::Kind::Look)
        .map_err(|e| anyhow::anyhow!("{LIST_TAG}: loading look catalog: {e}"))?;
    let names: Vec<String> = cat.entries.iter().map(|e| e.name.clone()).collect();
    let n = names.len();
    for name in names {
        push(vm, Value::from_string(name));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(target: "plakat", "{LIST_TAG}: pushed {n} look name(s) + depth");
    // Touch `with_ctx` to anchor the rwlock invariant the rest of
    // the words rely on (no-op read).
    with_ctx(|_| ()).map_err(|e| anyhow::anyhow!("{LIST_TAG}: ctx unavailable: {e}"))?;
    Ok(vm)
}
