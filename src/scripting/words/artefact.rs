//! v0.22 phase 9: `plakat.artefact.*` host words.
//!
//! ```bund
//! "oak"                       plakat.artefact.add
//! "sun@sky/right:0.8"         plakat.artefact.add
//! plakat.artefact.blend.enable
//! 0.35 "artefact_blend_strength" plakat.config.set
//! "sd15" plakat.load
//! "a forest clearing" plakat.generate
//! ```
//!
//! Stack grammar:
//!
//! | Word | Stack effect |
//! |---|---|
//! | `( spec -- )` | `plakat.artefact.add`: parse `NAME[@ZONE[:SCALE]]`, push to artefact stack |
//! | `( -- )` | `plakat.artefact.clear`: drop stack |
//! | `( -- list )` | `plakat.artefact.list`: push display strings + count |
//! | `( -- )` | `plakat.artefact.blend.enable`: set `ctx.artefact_blend_enabled = true` |
//! | `( -- )` | `plakat.artefact.blend.disable`: reset it |
//!
//! Spec grammar matches the CLI's `--artefact` flag exactly —
//! `ArtefactSpec::from_str`. Full-object overrides (offset / anchor
//! / flip / alpha) aren't expressible in CLI shorthand and remain
//! HJSON-only; v0.22 phase 9 ships the shorthand surface only.
//!
//! Three bundled Category-B config keys live alongside this
//! namespace via `plakat.config.set`:
//!
//! - `artefact_library` (string path, default empty → CLI default
//!   `assets/artefact_library`)
//! - `artefact_blend_strength` (float [0, 1], default 0.3)
//! - `artefact_smart_zones` (bool, default false)
//!
//! **Family scope (v0.22 phase 9)**: SD-family only. The blend
//! pass uses `portrait::Pipeline::blend_latents_one`; the compose
//! pass is family-agnostic but bundled with blend behind the
//! same gate to keep "either both or neither". Flux + SD3
//! `plakat.generate` bail when the artefact stack is non-empty.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::str::FromStr;

use crate::artefacts::ArtefactSpec;
use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

// ---- plakat.artefact.add ( spec -- ) -----------------------------

const ADD_TAG: &str = "plakat.artefact.add";

pub fn plakat_artefact_add(vm: &mut VM) -> BundResult<'_> {
    do_plakat_artefact_add(vm).map_err(to_bund_err)
}

fn do_plakat_artefact_add(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, ADD_TAG)?;
    let spec_v = pull(vm, ADD_TAG)?;
    let spec_str = value_to_string(spec_v, "spec", ADD_TAG)?;
    let spec = ArtefactSpec::from_str(&spec_str).map_err(|e| {
        anyhow::anyhow!("{ADD_TAG}: parsing spec {spec_str:?}: {e}")
    })?;
    let depth = with_ctx_mut(|ctx| {
        ctx.artefacts.push(spec);
        ctx.artefacts.len()
    })?;
    tracing::info!(
        target: "plakat",
        "{ADD_TAG}: pushed {spec_str:?} (stack now {depth} artefact(s))"
    );
    Ok(vm)
}

// ---- plakat.artefact.clear ( -- ) --------------------------------

const CLEAR_TAG: &str = "plakat.artefact.clear";

pub fn plakat_artefact_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_artefact_clear(vm).map_err(to_bund_err)
}

fn do_plakat_artefact_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| ctx.artefacts.clear())?;
    tracing::info!(target: "plakat", "{CLEAR_TAG}: stack drained");
    Ok(vm)
}

// ---- plakat.artefact.list ( -- s_1 ... s_n n ) -------------------

const LIST_TAG: &str = "plakat.artefact.list";

pub fn plakat_artefact_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_artefact_list(vm).map_err(to_bund_err)
}

fn do_plakat_artefact_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    // Display each spec as a roundtripped shorthand string:
    // `NAME`, `NAME@ZONE`, or `NAME@ZONE:SCALE`. Matches the
    // `--artefact` grammar so users can copy-paste a list entry
    // back into another script (or the CLI) and get the same
    // spec.
    let entries: Vec<String> = with_ctx_mut(|ctx| {
        ctx.artefacts.iter().map(format_spec).collect()
    })?;
    let n = entries.len();
    for entry in entries {
        push(vm, Value::from_string(entry));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(target: "plakat", "{LIST_TAG}: pushed {n} entries + depth");
    Ok(vm)
}

fn format_spec(s: &ArtefactSpec) -> String {
    let zone = s.zone.as_ref().map(|z| z.display());
    match (zone, s.scale) {
        (Some(z), Some(sc)) => format!("{}@{z}:{sc}", s.name),
        (Some(z), None) => format!("{}@{z}", s.name),
        (None, _) => s.name.clone(),
    }
}

// ---- plakat.artefact.blend.enable / disable ----------------------

const BLEND_ENABLE_TAG: &str = "plakat.artefact.blend.enable";
const BLEND_DISABLE_TAG: &str = "plakat.artefact.blend.disable";

pub fn plakat_artefact_blend_enable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_artefact_blend_enable(vm).map_err(to_bund_err)
}

fn do_plakat_artefact_blend_enable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| ctx.artefact_blend_enabled = true)?;
    tracing::info!(target: "plakat", "{BLEND_ENABLE_TAG}: ON");
    Ok(vm)
}

pub fn plakat_artefact_blend_disable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_artefact_blend_disable(vm).map_err(to_bund_err)
}

fn do_plakat_artefact_blend_disable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| ctx.artefact_blend_enabled = false)?;
    tracing::info!(target: "plakat", "{BLEND_DISABLE_TAG}: OFF");
    Ok(vm)
}
