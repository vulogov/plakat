//! v0.25 phase 8: `plakat.genre.*` host words.
//!
//! ```bund
//! "anime" plakat.genre.apply             // pick by name
//! plakat.genre.list                      // ( -- g_1 ... g_n n )
//! plakat.genre.clear                     // forget the active genre
//! ```
//!
//! State lives on [`ScriptCtx`] as `genre_name`. Independent axis
//! from [`super::look`] — they compose additively at generate
//! time. Same cache invalidation + discovery gating as
//! `plakat.look.*` (see that module's docs for the details).

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

// ---- plakat.genre.apply ( name -- ) ------------------------------

const APPLY_TAG: &str = "plakat.genre.apply";

pub fn plakat_genre_apply(vm: &mut VM) -> BundResult<'_> {
    do_plakat_genre_apply(vm).map_err(to_bund_err)
}

fn do_plakat_genre_apply(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, APPLY_TAG)?;
    let name_v = pull(vm, APPLY_TAG)?;
    let name = value_to_string(name_v, "name", APPLY_TAG)?;
    if name.is_empty() {
        anyhow::bail!("{APPLY_TAG}: genre name can't be empty");
    }
    let cat = crate::preset::Catalog::load_default(crate::preset::Kind::Genre)
        .map_err(|e| anyhow::anyhow!("{APPLY_TAG}: loading genre catalog: {e}"))?;
    if cat.find(&name).is_none() {
        anyhow::bail!(
            "{APPLY_TAG}: unknown genre {name:?} (try one of: {})",
            cat.names().join(", ")
        );
    }
    with_ctx_mut(|ctx| {
        ctx.genre_name = Some(name.clone());
        ctx.mark_loras_changed();
    })?;
    tracing::info!(target: "plakat", "{APPLY_TAG}: genre_name = {name:?}");
    Ok(vm)
}

// ---- plakat.genre.clear ( -- ) -----------------------------------

const CLEAR_TAG: &str = "plakat.genre.clear";

pub fn plakat_genre_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_genre_clear(vm).map_err(to_bund_err)
}

fn do_plakat_genre_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let was_set = with_ctx_mut(|ctx| {
        let was = ctx.genre_name.is_some();
        ctx.genre_name = None;
        if was {
            ctx.mark_loras_changed();
        }
        was
    })?;
    tracing::info!(
        target: "plakat",
        "{CLEAR_TAG}: genre state cleared (was active: {was_set})"
    );
    Ok(vm)
}

// ---- plakat.genre.list ( -- g_1 ... g_n n ) ----------------------

const LIST_TAG: &str = "plakat.genre.list";

pub fn plakat_genre_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_genre_list(vm).map_err(to_bund_err)
}

fn do_plakat_genre_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let cat = crate::preset::Catalog::load_default(crate::preset::Kind::Genre)
        .map_err(|e| anyhow::anyhow!("{LIST_TAG}: loading genre catalog: {e}"))?;
    let names: Vec<String> = cat.entries.iter().map(|e| e.name.clone()).collect();
    let n = names.len();
    for name in names {
        push(vm, Value::from_string(name));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(target: "plakat", "{LIST_TAG}: pushed {n} genre name(s) + depth");
    with_ctx(|_| ()).map_err(|e| anyhow::anyhow!("{LIST_TAG}: ctx unavailable: {e}"))?;
    Ok(vm)
}
