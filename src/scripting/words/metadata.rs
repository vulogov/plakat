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

// =====================================================================
// v0.26 phase 8: plakat.metadata.write ( handle path -- )
// =====================================================================

const WRITE_TAG: &str = "plakat.metadata.write";

/// `plakat.metadata.write ( handle path -- )` — re-attach the
/// metadata from an in-memory image handle to an existing file
/// at `path`. Writes the JSON sidecar (`<name>.json`) AND
/// re-encodes the PNG with the A1111 `parameters` tEXt chunk.
///
/// Use case: a script generates an image, saves it once, then
/// edits or upscales the result and wants the new file to carry
/// the same provenance. Or an external tool produces a PNG and
/// the script wants to attach plakat metadata to it.
///
/// Stack effect: `( handle path -- )`. Top pops first (path).
///
/// Bails when:
/// - The handle has no metadata attached (the rendering path
///   didn't populate it).
/// - The file at `path` doesn't exist or can't be read.
/// - The file isn't a PNG (sidecar still writes; tEXt is PNG-only).
///
/// Relative paths resolve against `ScriptCtx.out_dir`.
pub fn plakat_metadata_write(vm: &mut VM) -> BundResult<'_> {
    do_plakat_metadata_write(vm).map_err(to_bund_err)
}

fn do_plakat_metadata_write(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, WRITE_TAG)?;
    let path_v = pull(vm, WRITE_TAG)?;
    let handle_v = pull(vm, WRITE_TAG)?;
    let path_str = value_to_string(path_v, "path", WRITE_TAG)?;
    let handle = crate::scripting::helpers::value_to_int(
        handle_v, "handle", WRITE_TAG,
    )?;
    if path_str.is_empty() {
        anyhow::bail!("{WRITE_TAG}: path can't be empty");
    }

    // Pull out_dir + the metadata once.
    let (out_dir, meta_clone) =
        crate::scripting::ctx::with_ctx(|ctx| {
            let meta = ctx.metadata_at(handle)?.cloned();
            Ok::<_, anyhow::Error>((ctx.out_dir.clone(), meta))
        })??;
    let meta = meta_clone.ok_or_else(|| {
        anyhow::anyhow!(
            "{WRITE_TAG}: handle {handle} has no metadata attached. \
             The image was registered without metadata (e.g. via an \
             older rendering path) — call plakat.generate / .img2img / \
             .portrait / .stylize / .outpaint on it first."
        )
    })?;

    let path: PathBuf = {
        let p = PathBuf::from(&path_str);
        if p.is_absolute() {
            p
        } else {
            out_dir.join(p)
        }
    };
    if !path.exists() {
        anyhow::bail!(
            "{WRITE_TAG}: target file doesn't exist: {}. Pass the path \
             of an already-saved image (use plakat.save first if needed).",
            path.display()
        );
    }

    // JSON sidecar — write next to the file.
    let json_path = path.with_extension("json");
    let json = meta
        .to_json_pretty()
        .map_err(|e| anyhow::anyhow!("{WRITE_TAG}: serializing metadata: {e}"))?;
    std::fs::write(&json_path, json).map_err(|e| {
        anyhow::anyhow!("{WRITE_TAG}: writing sidecar {}: {e}", json_path.display())
    })?;

    // PNG tEXt re-attach. Only PNG files get the embedded chunk;
    // other formats (WebP / JPEG) just get the sidecar.
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    if ext.as_deref() == Some("png") {
        // Round-trip through DynamicImage to re-encode the PNG
        // with the tEXt chunk. The save_rgb_u8_with_metadata
        // helper does both sidecar + tEXt; we already wrote the
        // sidecar so this is for the tEXt only — and the helper
        // overwriting the sidecar with the same bytes is fine.
        let img = image::open(&path).map_err(|e| {
            anyhow::anyhow!(
                "{WRITE_TAG}: reading {} for re-encode: {e}",
                path.display()
            )
        })?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        crate::imaging::io::save_rgb_u8_with_metadata(
            rgb.as_raw(),
            w,
            h,
            &path,
            &meta,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "{WRITE_TAG}: re-encoding PNG with tEXt {}: {e}",
                path.display()
            )
        })?;
    }

    tracing::info!(
        target: "plakat",
        "{WRITE_TAG}: handle {handle} metadata re-attached to {}",
        path.display()
    );
    Ok(vm)
}
