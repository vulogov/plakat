//! v0.24 phase 6: `plakat.stylize ( subject style -- handle )`.
//!
//! IP-Adapter style transfer: SD 1.5 base + IP-Adapter Plus
//! image-encoder + a separate UNet pass that conditions on the
//! reference image's CLIP-H embedding. No prompt — the CLI
//! `plakat stylize` subcommand doesn't take one either; style
//! transfer is image-driven.
//!
//! ## Stack effect
//!
//! `( subject style -- handle )`. Both args accept the same two
//! shapes as `plakat.img2img`:
//! - string path: filesystem path read directly.
//! - integer handle: image handle materialised to a tempfile
//!   bound to this stack frame.
//!
//! Strength is read from `ctx.config.strength` (default 0.75).
//! Steps from `ctx.config.steps`. Seed from `ctx.config.seed`.
//!
//! **SD 1.5 only.** SDXL / Flux / SD3 bail with a clear pointer
//! at the underlying `stylize::Pipeline::load` checkpoint check.
//!
//! ```bund
//! "sd15" plakat.load
//! 0.35 "strength" plakat.config.set     // face-preserving preset
//! "alice.jpg" "renaissance.jpg" plakat.stylize
//!   "alice-renaissance.png" plakat.save
//! ```
//!
//! ## Caching
//!
//! No cache slot in v0.24 phase 6 — each `plakat.stylize` loads
//! SD 1.5 + IP-Adapter fresh (~5 GB total). Scripts that call
//! stylize once per run pay the load once; scripts that loop
//! pay each iteration. Caching is a v0.25+ optimisation.

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.stylize";

pub fn plakat_stylize(vm: &mut VM) -> BundResult<'_> {
    do_plakat_stylize(vm).map_err(to_bund_err)
}

fn do_plakat_stylize(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Top pops first: style, then subject.
    let style_v = pull(vm, TAG)?;
    let subject_v = pull(vm, TAG)?;

    // Validate arg shapes (string or int) + paths (non-empty)
    // BEFORE touching ctx state. Bad-arg errors fire before
    // the no-model-loaded gate so users see the most specific
    // error first.
    let (_subject_tmp_guard, subject_path) = resolve_image_arg(subject_v, "subject")?;
    let (_style_tmp_guard, style_path) = resolve_image_arg(style_v, "style")?;

    // Snapshot config + alias for the async `stylize::run` call.
    let (alias, strength, steps, seed, device) = with_ctx(|ctx| {
        (
            ctx.loaded_model().map(|s| s.to_string()),
            ctx.config.strength,
            ctx.config.steps,
            ctx.config.seed,
            ctx.device.clone(),
        )
    })?;
    let alias = alias.ok_or_else(|| {
        anyhow::anyhow!(
            "{TAG}: no model loaded. Call \"sd15\" plakat.load before {TAG}."
        )
    })?;
    // Quick early bail for obvious non-SD-1.5 aliases — the
    // underlying `stylize::Pipeline::load` will also bail, but
    // catching here gives a tighter error.
    let alias_lower = alias.to_lowercase();
    if alias_lower.contains("xl") || alias_lower.contains("flux") || alias_lower.contains("sd3") {
        anyhow::bail!(
            "{TAG}: stylize is SD 1.5 only (got {alias:?}). Call \
             \"sd15\" plakat.load before {TAG}."
        );
    }

    // Output to a tempfile bound to this stack frame; we read
    // the rendered PNG back into a DynamicImage and push the
    // handle.
    let out_tmp = tempfile::Builder::new()
        .prefix("plakat-script-stylize-")
        .suffix(".png")
        .tempfile()
        .map_err(|e| anyhow::anyhow!("{TAG}: creating output tempfile: {e}"))?;
    let out_path = out_tmp.path().to_path_buf();

    let req = crate::pipelines::stylize::Request {
        input: subject_path.clone(),
        reference: style_path.clone(),
        out: out_path.clone(),
        strength,
        model: alias.clone(),
        steps,
        seed,
        device,
    };

    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: no tokio runtime in scope (eval must run on a \
             multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    tokio::task::block_in_place(|| {
        handle.block_on(crate::pipelines::stylize::run(req))
    })
    .map_err(|e| anyhow::anyhow!("{TAG}: stylize::run failed: {e}"))?;

    // Read the rendered PNG back + register a handle.
    let img = image::open(&out_path).map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: reading rendered PNG {}: {e}",
            out_path.display()
        )
    })?;
    let handle_int = with_ctx_mut(|ctx| ctx.push_image(img))?;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} (subject={}, style={}, strength={strength})",
        subject_path.display(),
        style_path.display()
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}

/// Resolve a stack arg that's either a string path or an int
/// handle to a `PathBuf` + an optional tempfile guard (None when
/// the input was a string).
fn resolve_image_arg(
    v: rust_dynamic::value::Value,
    field: &str,
) -> anyhow::Result<(Option<tempfile::NamedTempFile>, PathBuf)> {
    match v.dt {
        types::STRING => {
            let s = value_to_string(v, field, TAG)?;
            if s.is_empty() {
                anyhow::bail!("{TAG}: {field} path can't be empty");
            }
            Ok((None, PathBuf::from(s)))
        }
        types::INTEGER => {
            let handle = v.cast_int().unwrap_or(0);
            let img = with_ctx_mut(|ctx| ctx.image_at(handle).cloned())??;
            let tmp = tempfile::Builder::new()
                .prefix(&format!("plakat-script-stylize-{field}-"))
                .suffix(".png")
                .tempfile()
                .map_err(|e| {
                    anyhow::anyhow!("{TAG}: tempfile for {field} handle {handle}: {e}")
                })?;
            img.save(tmp.path()).map_err(|e| {
                anyhow::anyhow!(
                    "{TAG}: writing {field} handle {handle} to tempfile: {e}"
                )
            })?;
            let path = tmp.path().to_path_buf();
            Ok((Some(tmp), path))
        }
        _ => anyhow::bail!(
            "{TAG}: {field} must be a string path or an integer handle \
             (got rust_dynamic dt = {})",
            v.dt
        ),
    }
}
