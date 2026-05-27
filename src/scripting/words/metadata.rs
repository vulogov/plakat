//! v0.24 phase 7: `plakat.metadata.read ( path -- … )`.
//!
//! Reads the JSON sidecar plakat writes alongside every
//! generated PNG (the structured form of the A1111
//! `parameters` tEXt chunk). Pushes every populated field as a
//! `(key, value)` pair of strings + a count.
//!
//! Write is deferred to v0.25 (per RFC §6 Q3) — gated on
//! `plakat.save` itself attaching sidecars.
//!
//! ## Stack effect
//!
//! ```text
//! ( path -- k_1 v_1 k_2 v_2 … k_n v_n n )
//! ```
//!
//! Top of stack is `n` (the pair count). Below are `n` pairs:
//! each pair is `( … key value )` with the value on top. Pop the
//! count, then loop `n` times popping `v` then `k`.
//!
//! Both keys and values are strings; numeric fields are
//! stringified at push time. Empty `negative`, empty `loras`,
//! and `None`-valued optional fields are skipped — the count
//! reflects only the present fields.
//!
//! Required fields (always present): `prompt`, `model`, `seed`,
//! `steps`, `guidance`, `scheduler`, `width`, `height`,
//! `generator`.
//!
//! Optional fields (pushed only when set): `negative`, `loras`,
//! `lora_scale`, `clip_skip`, `controls`, `refiner_frac`,
//! `mode`, `strength`, plus any `extras` (per-key push).
//!
//! ## Failure modes
//!
//! - Path doesn't exist → bail.
//! - PNG has no sidebar JSON sidecar (`<path>.json`) → bail
//!   with a pointer at `plakat metadata` (the CLI subcommand
//!   can still print whatever's in the A1111 tEXt chunk).
//!   Future enhancement (v0.25+): fall back to parsing the
//!   A1111 string directly.
//!
//! ```bund
//! "fox.png" plakat.metadata.read     // ( … k_n v_n n )
//! plakat.echo                         // prints n
//! ```

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.metadata.read";

pub fn plakat_metadata_read(vm: &mut VM) -> BundResult<'_> {
    do_plakat_metadata_read(vm).map_err(to_bund_err)
}

fn do_plakat_metadata_read(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let path_v = pull(vm, TAG)?;
    let path_str = value_to_string(path_v, "path", TAG)?;
    if path_str.is_empty() {
        anyhow::bail!("{TAG}: path can't be empty");
    }
    let path = PathBuf::from(&path_str);

    // Locate the JSON sidecar: same stem, `.json` extension.
    // The CLI's `plakat save` (and the t2i / portrait pipelines'
    // save_image) write this alongside every PNG unless
    // --no-metadata is set.
    let sidecar = path.with_extension("json");
    if !sidecar.exists() {
        anyhow::bail!(
            "{TAG}: no JSON sidecar at {} (the PNG might've been written \
             with --no-metadata, or it's not a plakat output). Use the CLI \
             `plakat metadata {}` to inspect the A1111 tEXt chunk if there \
             is one.",
            sidecar.display(),
            path.display()
        );
    }

    let json_text = std::fs::read_to_string(&sidecar).map_err(|e| {
        anyhow::anyhow!("{TAG}: reading {}: {e}", sidecar.display())
    })?;
    let md: crate::imaging::metadata::GenerationMetadata =
        serde_json::from_str(&json_text).map_err(|e| {
            anyhow::anyhow!(
                "{TAG}: parsing {}: {e}. The file exists but doesn't \
                 deserialize into GenerationMetadata — it may be a \
                 different schema version or hand-edited.",
                sidecar.display()
            )
        })?;

    // Build the (key, value) pair list. Required fields first,
    // then optional fields, then extras.
    let mut pairs: Vec<(String, String)> = Vec::new();
    pairs.push(("prompt".into(), md.prompt.clone()));
    if !md.negative.is_empty() {
        pairs.push(("negative".into(), md.negative.clone()));
    }
    pairs.push(("model".into(), md.model.clone()));
    pairs.push(("seed".into(), md.seed.to_string()));
    pairs.push(("steps".into(), md.steps.to_string()));
    pairs.push(("guidance".into(), md.guidance.to_string()));
    pairs.push(("scheduler".into(), md.scheduler.clone()));
    pairs.push(("width".into(), md.width.to_string()));
    pairs.push(("height".into(), md.height.to_string()));
    if !md.loras.is_empty() {
        pairs.push(("loras".into(), md.loras.join(",")));
    }
    if let Some(v) = md.lora_scale {
        pairs.push(("lora_scale".into(), v.to_string()));
    }
    if let Some(v) = md.clip_skip {
        pairs.push(("clip_skip".into(), v.to_string()));
    }
    if !md.controls.is_empty() {
        pairs.push(("controls".into(), md.controls.join(",")));
    }
    if let Some(v) = md.refiner_frac {
        pairs.push(("refiner_frac".into(), v.to_string()));
    }
    if let Some(s) = &md.mode {
        pairs.push(("mode".into(), s.clone()));
    }
    if let Some(v) = md.strength {
        pairs.push(("strength".into(), v.to_string()));
    }
    pairs.push(("generator".into(), md.generator.clone()));
    for (k, v) in &md.extras {
        pairs.push((k.clone(), v.clone()));
    }

    // Push pairs onto the stack in the documented order:
    // for each pair, push k first then v, so the value lands
    // on top of its key. The user pops n, then loops popping
    // (v, k) repeatedly.
    let n = pairs.len();
    for (k, v) in &pairs {
        push(vm, Value::from_string(k.clone()));
        push(vm, Value::from_string(v.clone()));
    }
    push(vm, Value::from_int(n as i64));
    tracing::info!(
        target: "plakat",
        "{TAG}: pushed {n} field pair(s) from {}",
        sidecar.display()
    );
    Ok(vm)
}
