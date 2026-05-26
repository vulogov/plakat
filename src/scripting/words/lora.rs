//! v0.22 phase 4: `plakat.lora.*` host words.
//!
//! ```bund
//! "civitai:2595428" 0.7 plakat.lora.add
//! "./my.safetensors" 1.0 plakat.lora.add
//! 0.5 "lora_scale" plakat.config.set      // global multiplier
//! "sd15" plakat.load                       // loads with the stack
//!
//! plakat.lora.list                         // → list of "spec:scale" strings
//! plakat.lora.clear                        // drops the stack
//! ```
//!
//! Mutations invalidate the cache per RFC §7 (the "defer the merge
//! to next generate" pattern). `plakat.lora.add` / `clear` both
//! call `ctx.mark_loras_changed()` which drops `ctx.loaded`; the
//! next `plakat.generate` (or any `ensure_loaded`) rebuilds the
//! pipeline with the current LoRA set merged in.
//!
//! Stack grammar (Forth-flavoured):
//!
//! | Word | Effect |
//! |---|---|
//! | `( spec scale -- )` | `plakat.lora.add`: pop scale (top), pop spec (bottom), push to LoRA stack |
//! | `( -- )` | `plakat.lora.clear`: drop entire stack |
//! | `( -- list )` | `plakat.lora.list`: push current stack as a list of `"spec:scale"` strings |
//!
//! `spec` is the same grammar `--lora` accepts on the CLI:
//! `path/to.safetensors`, `civitai:NNNNNN`, `civitai-version:NNNNNN`,
//! HF `repo#file`, etc. The `:scale` suffix on the string is
//! redundant when you pass an explicit scale arg — the stack scale
//! overrides any `:scale` baked into the spec string.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::str::FromStr;

use crate::pipelines::lora::{CivitaiIdKind, LoraSource, LoraSpec};
use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_float, value_to_string,
};

/// Format a `LoraSource` for the REPL `.list` listing. Mirrors
/// the CLI's resolved-LoRA display strings: local path basename;
/// `repo#file` or `repo` for HF specs; `civitai:NNN` or
/// `civitai-version:NNN` (with optional `#file`) for Civitai.
fn format_source(src: &LoraSource) -> String {
    match src {
        LoraSource::Local(p) => p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("(local)")
            .to_string(),
        LoraSource::Hub { repo, file, .. } => match file {
            Some(f) => format!("{repo}#{f}"),
            None => repo.clone(),
        },
        LoraSource::Civitai { id_kind, file } => {
            let base = match id_kind {
                CivitaiIdKind::Model(n) => format!("civitai:{n}"),
                CivitaiIdKind::Version(n) => format!("civitai-version:{n}"),
            };
            match file {
                Some(f) => format!("{base}#{f}"),
                None => base,
            }
        }
    }
}

// ---- plakat.lora.add ( spec scale -- ) ---------------------------

const ADD_TAG: &str = "plakat.lora.add";

pub fn plakat_lora_add(vm: &mut VM) -> BundResult<'_> {
    do_plakat_lora_add(vm).map_err(to_bund_err)
}

fn do_plakat_lora_add(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, ADD_TAG)?;
    // Top pops first: scale. Then spec.
    let scale_v = pull(vm, ADD_TAG)?;
    let spec_v = pull(vm, ADD_TAG)?;
    let scale = value_to_float(scale_v, "scale", ADD_TAG)? as f32;
    let spec_str = value_to_string(spec_v, "spec", ADD_TAG)?;

    if !scale.is_finite() || scale < 0.0 {
        anyhow::bail!(
            "{ADD_TAG}: scale must be finite and >= 0 (got {scale})"
        );
    }

    // Parse via FromStr so the script-side grammar matches the
    // CLI: paths, civitai shorthand, HF repo#file. Any `:scale`
    // suffix in the spec gets overridden by the explicit scale
    // arg from the stack — same precedence the CLI uses.
    let mut spec = LoraSpec::from_str(&spec_str)
        .map_err(|e| anyhow::anyhow!("{ADD_TAG}: parsing spec {spec_str:?}: {e}"))?;
    spec.scale = scale;

    with_ctx_mut(|ctx| {
        ctx.loras.push(spec);
        ctx.mark_loras_changed();
    })?;
    tracing::info!(
        target: "plakat",
        "{ADD_TAG}: pushed {spec_str:?} @ scale {scale} (stack now {} LoRA(s))",
        with_ctx_mut(|ctx| ctx.loras.len()).unwrap_or(0)
    );
    Ok(vm)
}

// ---- plakat.lora.clear ( -- ) ------------------------------------

const CLEAR_TAG: &str = "plakat.lora.clear";

pub fn plakat_lora_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_lora_clear(vm).map_err(to_bund_err)
}

fn do_plakat_lora_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        let n = ctx.loras.len();
        ctx.loras.clear();
        ctx.mark_loras_changed();
        n
    })?;
    tracing::info!(target: "plakat", "{CLEAR_TAG}: stack drained");
    Ok(vm)
}

// ---- plakat.lora.list ( -- list ) --------------------------------

const LIST_TAG: &str = "plakat.lora.list";

pub fn plakat_lora_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_lora_list(vm).map_err(to_bund_err)
}

fn do_plakat_lora_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    // Read the current LoRA stack + format each entry as a
    // human-readable `"<display>:<scale>"` string. We don't push
    // a Bund list (rust_dynamic LIST is awkward to inspect at
    // the REPL) — instead, push one string per entry. Scripts
    // that want to count use `.s` to see the depth.
    //
    // The display form mirrors the CLI's resolved-LoRA output:
    // path basename for local LoRAs, `civitai:NNNN` for civitai
    // specs, `repo#file` for HF specs.
    let entries: Vec<String> = with_ctx_mut(|ctx| {
        ctx.loras
            .iter()
            .map(|s| format!("{}:{}", format_source(&s.source), s.scale))
            .collect()
    })?;
    let n = entries.len();
    for entry in entries {
        push(vm, Value::from_string(entry));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(target: "plakat", "{LIST_TAG}: pushed {n} entries + depth");
    Ok(vm)
}
