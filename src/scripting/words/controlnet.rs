//! v0.22 phase 5: `plakat.controlnet.*` host words.
//!
//! ControlNet stacking via the same collection-state pattern
//! `plakat.lora.*` uses. Three add-style words match the CLI's
//! three input shapes:
//!
//! | Word | Stack effect | CLI equivalent |
//! |---|---|---|
//! | `plakat.controlnet.add` | `( kind image-path -- )` | `--control KIND --control-image PATH` |
//! | `plakat.controlnet.annotate` | `( kind from-path -- )` | `--control KIND --control-from PATH` |
//! | `plakat.controlnet.spec` | `( spec-string -- )` | `--control-spec STR` (full grammar) |
//! | `plakat.controlnet.clear` | `( -- )` | (drop the stack) |
//! | `plakat.controlnet.list` | `( -- ...entries depth )` | (introspection) |
//!
//! `spec-string` accepts the full v0.21 `--control-spec` grammar:
//! `KIND[:option=value]*` where options are `image=PATH`,
//! `from=PATH`, `strength=F`, `start=F`, `end=F`.
//!
//! **Family scope (v0.22 phase 5)**: SD-family only. ControlNet on
//! the cached SD-family `portrait::Pipeline` flows through
//! `Pipeline::generate(req, &control_reqs)` at *generate* time —
//! no cache invalidation needed. Flux + SD3 ControlNet need
//! load-time `FluxControlNetLoad` / `Sd3ControlNetLoad` setup that
//! doesn't fit phase 5's scope; `plakat.generate` / `plakat.img2img`
//! on Flux + SD3 bail loud when the controlnet stack is non-empty.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::str::FromStr;

use crate::pipelines::controlnet::{ControlKind, ControlSpec};
use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

// ---- plakat.controlnet.add ( kind image -- ) ---------------------

const ADD_TAG: &str = "plakat.controlnet.add";

pub fn plakat_controlnet_add(vm: &mut VM) -> BundResult<'_> {
    do_plakat_controlnet_add(vm).map_err(to_bund_err)
}

fn do_plakat_controlnet_add(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, ADD_TAG)?;
    // Top pops first: image path. Then kind.
    let image_v = pull(vm, ADD_TAG)?;
    let kind_v = pull(vm, ADD_TAG)?;
    let kind = parse_kind(value_to_string(kind_v, "kind", ADD_TAG)?, ADD_TAG)?;
    let image = value_to_string(image_v, "image", ADD_TAG)?;
    let spec = ControlSpec {
        kind,
        image: Some(std::path::PathBuf::from(image)),
        from: None,
        strength: 1.0,
        start: 0.0,
        end: 1.0,
    };
    push_controlnet(spec, ADD_TAG)?;
    Ok(vm)
}

// ---- plakat.controlnet.annotate ( kind from -- ) -----------------

const ANNOTATE_TAG: &str = "plakat.controlnet.annotate";

pub fn plakat_controlnet_annotate(vm: &mut VM) -> BundResult<'_> {
    do_plakat_controlnet_annotate(vm).map_err(to_bund_err)
}

fn do_plakat_controlnet_annotate(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, ANNOTATE_TAG)?;
    let from_v = pull(vm, ANNOTATE_TAG)?;
    let kind_v = pull(vm, ANNOTATE_TAG)?;
    let kind = parse_kind(value_to_string(kind_v, "kind", ANNOTATE_TAG)?, ANNOTATE_TAG)?;
    let from = value_to_string(from_v, "from", ANNOTATE_TAG)?;
    let spec = ControlSpec {
        kind,
        image: None,
        from: Some(std::path::PathBuf::from(from)),
        strength: 1.0,
        start: 0.0,
        end: 1.0,
    };
    push_controlnet(spec, ANNOTATE_TAG)?;
    Ok(vm)
}

// ---- plakat.controlnet.spec ( spec-string -- ) -------------------

const SPEC_TAG: &str = "plakat.controlnet.spec";

pub fn plakat_controlnet_spec(vm: &mut VM) -> BundResult<'_> {
    do_plakat_controlnet_spec(vm).map_err(to_bund_err)
}

fn do_plakat_controlnet_spec(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, SPEC_TAG)?;
    let spec_v = pull(vm, SPEC_TAG)?;
    let spec_str = value_to_string(spec_v, "spec", SPEC_TAG)?;
    let spec = ControlSpec::from_str(&spec_str).map_err(|e| {
        anyhow::anyhow!(
            "{SPEC_TAG}: parsing spec {spec_str:?}: {e}. \
             Grammar: KIND[:option=value]* where KIND is depth | \
             canny | openpose | lineart | softedge and options are \
             image=PATH | from=PATH | strength=F | start=F | end=F"
        )
    })?;
    push_controlnet(spec, SPEC_TAG)?;
    Ok(vm)
}

// ---- plakat.controlnet.clear ( -- ) ------------------------------

const CLEAR_TAG: &str = "plakat.controlnet.clear";

pub fn plakat_controlnet_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_controlnet_clear(vm).map_err(to_bund_err)
}

fn do_plakat_controlnet_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        ctx.controlnets.clear();
        // v0.23 phase 6: Flux + SD3 bake the CN stack at load time;
        // drop those slots so the next generate reloads without
        // CN. SD-family slots are left intact (per-call CN).
        ctx.mark_controlnets_changed();
    })?;
    tracing::info!(target: "plakat", "{CLEAR_TAG}: stack drained");
    Ok(vm)
}

// ---- plakat.controlnet.list ( -- ...entries depth ) --------------

const LIST_TAG: &str = "plakat.controlnet.list";

pub fn plakat_controlnet_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_controlnet_list(vm).map_err(to_bund_err)
}

fn do_plakat_controlnet_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let entries: Vec<String> = with_ctx_mut(|ctx| {
        ctx.controlnets.iter().map(format_spec).collect()
    })?;
    let n = entries.len();
    for entry in entries {
        push(vm, Value::from_string(entry));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(target: "plakat", "{LIST_TAG}: pushed {n} entries + depth");
    Ok(vm)
}

// ---- helpers -----------------------------------------------------

fn parse_kind(kind_str: String, tag: &str) -> anyhow::Result<ControlKind> {
    ControlKind::from_str(&kind_str).map_err(|e| {
        anyhow::anyhow!(
            "{tag}: parsing kind {kind_str:?}: {e}. \
             Supported: depth, canny, openpose, lineart, softedge"
        )
    })
}

fn push_controlnet(spec: ControlSpec, tag: &str) -> anyhow::Result<()> {
    let kind_slug = spec.kind.slug();
    let desc = format_spec(&spec);
    with_ctx_mut(|ctx| {
        ctx.controlnets.push(spec);
        // v0.23 phase 6: Flux + SD3 bake the CN stack at load time.
        ctx.mark_controlnets_changed();
    })?;
    tracing::info!(
        target: "plakat",
        "{tag}: pushed {kind_slug} controlnet ({desc}); stack now {} CN(s)",
        with_ctx_mut(|ctx| ctx.controlnets.len()).unwrap_or(0)
    );
    Ok(())
}

/// Compact human-readable form for `.list`. Mirrors the CLI
/// `--control-spec` grammar so users can copy a `.list` entry
/// back into a script verbatim.
fn format_spec(spec: &ControlSpec) -> String {
    let mut parts = vec![spec.kind.slug().to_string()];
    if let Some(p) = &spec.image {
        parts.push(format!("image={}", p.display()));
    }
    if let Some(p) = &spec.from {
        parts.push(format!("from={}", p.display()));
    }
    if (spec.strength - 1.0).abs() > 1e-6 {
        parts.push(format!("strength={}", spec.strength));
    }
    if spec.start > 0.0 {
        parts.push(format!("start={}", spec.start));
    }
    if (spec.end - 1.0).abs() > 1e-6 {
        parts.push(format!("end={}", spec.end));
    }
    parts.join(":")
}
