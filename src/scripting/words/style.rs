//! v0.23 phase 4: `plakat.style.*` host words.
//!
//! ```bund
//! "poster-bold" plakat.style.apply         // pick by id
//! "./ref.jpg"   plakat.style.detect        // pick by photo at next generate
//! plakat.style.list                        // ( -- s_1 … s_n n ) — catalog ids
//! plakat.style.clear                       // forget the active style
//! ```
//!
//! State lives on [`ScriptCtx`]: `style_id` (set by `.apply`) and
//! `style_ref` (set by `.detect`). Resolution against the catalog
//! happens lazily inside [`script_entry::generate_one`] when the
//! SD-family alias + the active style are both known — same
//! split-of-concerns as the CLI's `--style` / `--style-ref` flags
//! (resolution runs after model selection, before generate).
//!
//! **Cache invalidation**: all four words drop the cached SD
//! pipeline slots via `ctx.mark_loras_changed`. Style resolution
//! produces a different LoRA stack at load time (catalog LoRAs
//! override user LoRAs, per CLI behaviour), so the v0.22 LoRA
//! invalidation pattern applies.
//!
//! **Family scope (v0.23 phase 4)**: SD-family only. The style
//! catalog's `ModelEntry` map can hold entries for any
//! `BaseModel`; whether a given style resolves on a non-SD family
//! depends on the catalog. `plakat.generate` on Flux + SD3 bails
//! when style state is set (mirrors the CLI behaviour — style
//! integration on those backbones isn't wired in the runtime yet).

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    collect_images_in_dir, BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

// ---- plakat.style.train ( images-dir trigger out -- out ) --------
//
// Train a style LoRA from a folder of images. The base model is the one loaded via
// `plakat.load` (family auto-detected). Uses sensible training defaults (rank 16, 800 steps,
// lr 1e-4, 512px) — for full control use the `plakat style train` CLI. Pushes the output path
// back so it can be chained (e.g. into `plakat.lora.add`).

const TRAIN_TAG: &str = "plakat.style.train";

pub fn plakat_style_train(vm: &mut VM) -> BundResult<'_> {
    do_plakat_style_train(vm).map_err(to_bund_err)
}

fn do_plakat_style_train(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 3, TRAIN_TAG)?;
    // Top pops first: out, then trigger, then images-dir.
    let out_v = pull(vm, TRAIN_TAG)?;
    let trigger_v = pull(vm, TRAIN_TAG)?;
    let dir_v = pull(vm, TRAIN_TAG)?;
    let out = value_to_string(out_v, "out", TRAIN_TAG)?;
    let trigger = value_to_string(trigger_v, "trigger", TRAIN_TAG)?;
    let dir = value_to_string(dir_v, "images-dir", TRAIN_TAG)?;
    let images = collect_images_in_dir(&dir, TRAIN_TAG)?;

    let (model, device) = with_ctx(|ctx| (ctx.loaded_model().map(|s| s.to_string()), ctx.device.clone()))?;
    let model = model.ok_or_else(|| {
        anyhow::anyhow!("{TRAIN_TAG}: no model loaded. Call \"sd15\" plakat.load before {TRAIN_TAG}.")
    })?;

    let handle = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{TRAIN_TAG}: no tokio runtime in scope. {e}"))?;
    tokio::task::block_in_place(|| {
        handle.block_on(crate::api::style_train(
            model,
            images,
            trigger,
            16,           // rank
            800,          // steps
            1e-4,         // lr
            512,          // size
            PathBuf::from(&out),
            25,           // log_every
            device,
        ))
    })?;
    tracing::info!(target: "plakat", "{TRAIN_TAG}: wrote {out}");
    push(vm, Value::from_string(out));
    Ok(vm)
}

// ---- plakat.style.apply ( id -- ) --------------------------------

const APPLY_TAG: &str = "plakat.style.apply";

pub fn plakat_style_apply(vm: &mut VM) -> BundResult<'_> {
    do_plakat_style_apply(vm).map_err(to_bund_err)
}

fn do_plakat_style_apply(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, APPLY_TAG)?;
    let id_v = pull(vm, APPLY_TAG)?;
    let id = value_to_string(id_v, "id", APPLY_TAG)?;
    if id.is_empty() {
        anyhow::bail!("{APPLY_TAG}: style id can't be empty");
    }
    with_ctx_mut(|ctx| {
        ctx.style_id = Some(id.clone());
        // Style change → load-time LoRA stack changes → drop cache.
        ctx.mark_loras_changed();
    })?;
    tracing::info!(target: "plakat", "{APPLY_TAG}: style_id = {id:?}");
    Ok(vm)
}

// ---- plakat.style.detect ( photo -- ) ----------------------------

const DETECT_TAG: &str = "plakat.style.detect";

pub fn plakat_style_detect(vm: &mut VM) -> BundResult<'_> {
    do_plakat_style_detect(vm).map_err(to_bund_err)
}

fn do_plakat_style_detect(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, DETECT_TAG)?;
    let path_v = pull(vm, DETECT_TAG)?;
    let path_str = value_to_string(path_v, "photo", DETECT_TAG)?;
    if path_str.is_empty() {
        anyhow::bail!("{DETECT_TAG}: photo path can't be empty");
    }
    let path = PathBuf::from(&path_str);
    with_ctx_mut(|ctx| {
        ctx.style_ref = Some(path.clone());
        ctx.mark_loras_changed();
    })?;
    tracing::info!(
        target: "plakat",
        "{DETECT_TAG}: style_ref = {} (detection runs at next plakat.generate)",
        path.display()
    );
    Ok(vm)
}

// ---- plakat.style.clear ( -- ) -----------------------------------

const CLEAR_TAG: &str = "plakat.style.clear";

pub fn plakat_style_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_style_clear(vm).map_err(to_bund_err)
}

fn do_plakat_style_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let was_set = with_ctx_mut(|ctx| {
        let was = ctx.style_id.is_some() || ctx.style_ref.is_some();
        ctx.style_id = None;
        ctx.style_ref = None;
        if was {
            ctx.mark_loras_changed();
        }
        was
    })?;
    tracing::info!(
        target: "plakat",
        "{CLEAR_TAG}: style state cleared (was active: {was_set})"
    );
    Ok(vm)
}

// ---- plakat.style.list ( -- s_1 … s_n n ) ------------------------

const LIST_TAG: &str = "plakat.style.list";

pub fn plakat_style_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_style_list(vm).map_err(to_bund_err)
}

fn do_plakat_style_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let (catalog_dir, device) = with_ctx(|ctx| {
        let dir = if ctx.config.style_catalog.is_empty() {
            std::path::PathBuf::from("assets/style_catalog")
        } else {
            std::path::PathBuf::from(&ctx.config.style_catalog)
        };
        (dir, ctx.device.clone())
    })?;
    let catalog = crate::style::StyleCatalog::load(&catalog_dir, &device)
        .map_err(|e| anyhow::anyhow!("{LIST_TAG}: loading catalog from {}: {e}", catalog_dir.display()))?;
    let ids: Vec<String> = catalog.order.clone();
    let n = ids.len();
    for id in ids {
        push(vm, Value::from_string(id));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(target: "plakat", "{LIST_TAG}: pushed {n} style id(s) + depth");
    Ok(vm)
}
