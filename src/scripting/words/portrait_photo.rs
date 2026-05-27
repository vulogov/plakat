//! v0.24 phase 1: `plakat.portrait.photo.*` collection namespace.
//!
//! Multi-photo identity-blending stack for `plakat.portrait`.
//! Each entry carries a photo path + optional weight. Weights
//! normalize to sum-to-1 at request-build time
//! (`ip_adapter::normalize_photo_weights`). `None`-weighted
//! entries auto-fill the remainder evenly — same semantics as
//! the CLI's `--photo PATH:WEIGHT` repeatable flag.
//!
//! | Word | Stack effect |
//! |---|---|
//! | `plakat.portrait.photo.add` | `( path-or-handle weight -- )` |
//! | `plakat.portrait.photo.clear` | `( -- )` |
//! | `plakat.portrait.photo.list` | `( -- s_1 … s_n n )` |
//!
//! `path-or-handle` accepts the same two shapes as
//! `plakat.img2img`'s input: string path (passes straight
//! through) or integer image handle (materialized to a
//! tempfile bound to the script's lifetime). Handle-based
//! photos: the tempfile is leaked into the script's
//! `_handle_tempfiles` list on ScriptCtx so it survives
//! until the script ends.
//!
//! `weight` accepts the special value `-1.0` to mean "auto-fill"
//! (the v0.21 CLI default when a `--photo PATH` has no `:weight`
//! suffix). Positive weights pass through; negative-but-not-(-1)
//! values bail with a clear error.
//!
//! ```bund
//! "alice.jpg" 1.0  plakat.portrait.photo.add   // explicit weight 1.0
//! "bob.jpg"   0.5  plakat.portrait.photo.add   // explicit weight 0.5
//! "carol.jpg" -1.0 plakat.portrait.photo.add   // auto-fill
//! plakat.portrait.photo.list                   // "( s_1 s_2 s_3 3 )"
//! "a group portrait" plakat.portrait
//! ```
//!
//! **No cache invalidation** — photos are per-call on the
//! SD-family path; `plakat.portrait.photo.*` mutations don't
//! drop the cached pipeline (same as `plakat.controlnet.*` on
//! SD-family).

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::pipelines::ip_adapter::WeightedPhoto;
use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_float, value_to_string,
};

// ---- plakat.portrait.photo.add ( path-or-handle weight -- ) ------

const ADD_TAG: &str = "plakat.portrait.photo.add";

pub fn plakat_portrait_photo_add(vm: &mut VM) -> BundResult<'_> {
    do_plakat_portrait_photo_add(vm).map_err(to_bund_err)
}

fn do_plakat_portrait_photo_add(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, ADD_TAG)?;
    // Top pops first: weight.
    let weight_v = pull(vm, ADD_TAG)?;
    let photo_v = pull(vm, ADD_TAG)?;
    let weight_f = value_to_float(weight_v, "weight", ADD_TAG)? as f32;

    // Validate weight: positive = explicit, -1.0 = auto-fill,
    // anything else negative is a bug.
    let weight = if (weight_f - (-1.0)).abs() < 1e-6 {
        None
    } else if weight_f.is_finite() && weight_f >= 0.0 {
        Some(weight_f)
    } else {
        anyhow::bail!(
            "{ADD_TAG}: weight must be -1.0 (auto-fill) or >= 0 (got {weight_f})"
        );
    };

    // Resolve photo: string path or integer handle.
    let path = match photo_v.dt {
        types::STRING => {
            let s = value_to_string(photo_v, "photo", ADD_TAG)?;
            std::path::PathBuf::from(s)
        }
        types::INTEGER => {
            // Materialize the handle to a tempfile and persist it
            // via tempfile::TempPath::keep — the path is owned by
            // the OS until the script process exits. (Bund scripts
            // are one-shot per process, so leaking these is fine.)
            let handle = photo_v.cast_int().unwrap_or(0);
            let img = with_ctx_mut(|ctx| ctx.image_at(handle).cloned())??;
            let tmp = tempfile::Builder::new()
                .prefix("plakat-script-portrait-photo-")
                .suffix(".png")
                .tempfile()
                .map_err(|e| {
                    anyhow::anyhow!("{ADD_TAG}: tempfile for handle {handle}: {e}")
                })?;
            img.save(tmp.path()).map_err(|e| {
                anyhow::anyhow!(
                    "{ADD_TAG}: writing handle {handle} to tempfile: {e}"
                )
            })?;
            // Detach the tempfile so the path stays valid; the
            // OS reclaims at process exit. `tempfile::NamedTempFile::keep`
            // returns `(File, PathBuf)`; we only need the path.
            let (_, kept_path) = tmp.keep().map_err(|e| {
                anyhow::anyhow!("{ADD_TAG}: detaching tempfile: {e}")
            })?;
            kept_path
        }
        _ => {
            anyhow::bail!(
                "{ADD_TAG}: photo must be a string path or an integer handle \
                 (got rust_dynamic dt = {})",
                photo_v.dt
            );
        }
    };

    let spec = WeightedPhoto {
        path: path.clone(),
        weight,
    };
    let depth = with_ctx_mut(|ctx| {
        ctx.portrait_photos.push(spec);
        ctx.portrait_photos.len()
    })?;
    tracing::info!(
        target: "plakat",
        "{ADD_TAG}: pushed {} (weight {:?}); stack now {depth} photo(s)",
        path.display(),
        weight
    );
    Ok(vm)
}

// ---- plakat.portrait.photo.clear ( -- ) --------------------------

const CLEAR_TAG: &str = "plakat.portrait.photo.clear";

pub fn plakat_portrait_photo_clear(vm: &mut VM) -> BundResult<'_> {
    do_plakat_portrait_photo_clear(vm).map_err(to_bund_err)
}

fn do_plakat_portrait_photo_clear(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| ctx.portrait_photos.clear())?;
    tracing::info!(target: "plakat", "{CLEAR_TAG}: stack drained");
    Ok(vm)
}

// ---- plakat.portrait.photo.list ( -- s_1 … s_n n ) ---------------

const LIST_TAG: &str = "plakat.portrait.photo.list";

pub fn plakat_portrait_photo_list(vm: &mut VM) -> BundResult<'_> {
    do_plakat_portrait_photo_list(vm).map_err(to_bund_err)
}

fn do_plakat_portrait_photo_list(vm: &mut VM) -> anyhow::Result<&mut VM> {
    let entries: Vec<String> = with_ctx_mut(|ctx| {
        ctx.portrait_photos
            .iter()
            .map(|p| {
                let name = p
                    .path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("(?)");
                match p.weight {
                    Some(w) => format!("{name}:{w}"),
                    None => format!("{name}:auto"),
                }
            })
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
