//! v0.24 phase 5: `plakat.embedding.*` Textual Inversion namespace.
//!
//! Collection-style namespace mirroring `plakat.lora.*`. Each
//! entry is an [`EmbeddingSpec`] — path/HF-repo + optional
//! trigger + optional scale. Specs parse via
//! `EmbeddingSpec::FromStr` (same grammar as `--embedding`):
//!
//! ```text
//! path/to/foo.safetensors              -> path, no trigger, scale 1.0
//! path/to/foo.safetensors:mytrigger    -> path, trigger, scale 1.0
//! foo.safetensors:mytrigger:0.7        -> path, trigger, scale 0.7
//! foo.safetensors:0.7                  -> path, no trigger, scale 0.7
//! repo/user                            -> HF repo, no trigger
//! ```
//!
//! ```bund
//! "civitai:123:trigger:0.8"        plakat.embedding.add
//! "./my-ti.safetensors:foo"        plakat.embedding.add
//! "./my-ti.safetensors"            plakat.embedding.add
//! plakat.embedding.list             // ( -- s_1 ... s_n n )
//! plakat.embedding.clear
//! ```
//!
//! **Effective only on `plakat.generate`'s SdT2i path.** TI
//! embeddings live on `t2i::LoadRequest.embeddings`;
//! `portrait::Pipeline` doesn't take embeddings, so
//! `plakat.img2img` + `plakat.portrait` silently ignore the
//! stack (matches the CLI — `cli::img2img` and `cli::portrait`
//! don't expose `--embedding` either).
//!
//! **Cache invalidation**: mutations call `mark_loras_changed`
//! because embeddings are baked in at load time. The next
//! `plakat.generate` reloads with the updated stack.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::str::FromStr;

use crate::pipelines::embedding::EmbeddingSpec;
use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

// ---- plakat.embedding.add ( spec -- ) ----------------------------

const ADD_TAG: &str = "plakat.embedding.add";

pub fn plakat_embedding_add(vm: &mut VM) -> BundResult<'_> {
    do_plakat_embedding_add(vm).map_err(to_bund_err)
}

fn do_plakat_embedding_add(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, ADD_TAG)?;
    let spec_v = pull(vm, ADD_TAG)?;
    let spec_str = value_to_string(spec_v, "spec", ADD_TAG)?;
    if spec_str.is_empty() {
        anyhow::bail!("{ADD_TAG}: spec can't be empty");
    }
    let spec = EmbeddingSpec::from_str(&spec_str).map_err(|e| {
        anyhow::anyhow!("{ADD_TAG}: parsing spec {spec_str:?}: {e}")
    })?;
    let desc = format_spec(&spec);
    let depth = with_ctx_mut(|ctx| {
        ctx.embeddings.push(spec);
        // TI is load-time: same invalidation as LoRA mutations.
        ctx.mark_loras_changed();
        ctx.embeddings.len()
    })?;
    tracing::info!(
        target: "plakat",
        "{ADD_TAG}: pushed {desc} (stack now {depth} embedding(s))"
    );
    Ok(vm)
}

// ---- plakat.embedding.clear ( -- ) -------------------------------

const CLEAR_TAG: &str = "plakat.embedding.clear";

pub fn plakat_embedding_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_embedding_clear(vm).map_err(to_bund_err)
}

fn do_plakat_embedding_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let was_set = with_ctx_mut(|ctx| {
        let was = !ctx.embeddings.is_empty();
        ctx.embeddings.clear();
        if was {
            ctx.mark_loras_changed();
        }
        was
    })?;
    tracing::info!(
        target: "plakat",
        "{CLEAR_TAG}: stack drained (was active: {was_set})"
    );
    Ok(vm)
}

// ---- plakat.embedding.list ( -- s_1 … s_n n ) --------------------

const LIST_TAG: &str = "plakat.embedding.list";

pub fn plakat_embedding_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_embedding_list(vm).map_err(to_bund_err)
}

fn do_plakat_embedding_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let entries: Vec<String> = with_ctx_mut(|ctx| {
        ctx.embeddings.iter().map(format_spec).collect()
    })?;
    let n = entries.len();
    for entry in entries {
        push(vm, Value::from_string(entry));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(target: "plakat", "{LIST_TAG}: pushed {n} entries + depth");
    Ok(vm)
}

/// Round-trip an `EmbeddingSpec` back into a `--embedding`-style
/// display string for `plakat.embedding.list` + log lines.
fn format_spec(s: &EmbeddingSpec) -> String {
    match (&s.trigger, s.scale) {
        (Some(t), x) if (x - 1.0).abs() < 1e-6 => {
            format!("{}:{}", s.source, t)
        }
        (Some(t), x) => format!("{}:{}:{}", s.source, t, x),
        (None, x) if (x - 1.0).abs() < 1e-6 => s.source.clone(),
        (None, x) => format!("{}:{}", s.source, x),
    }
}
