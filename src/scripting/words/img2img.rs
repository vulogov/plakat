//! v0.21 phase 4: `plakat.img2img ( prompt input -- handle )`.
//!
//! Re-imagines an existing image at the script's current `strength`
//! config. The `input` arg accepts two shapes:
//!
//! 1. **Filesystem path** (string) — read directly from disk.
//! 2. **Image handle** (integer) — look up `ScriptCtx.images[handle-1]`,
//!    materialise to a tempfile, pass that path to the pipeline.
//!
//! The handle-reuse path is the load-bearing piece of phase 4 — it
//! lets scripts compose `plakat.generate → plakat.img2img` without
//! a round-trip through disk:
//!
//! ```bund
//! "sd15" plakat.load
//! 0.6 "strength" plakat.config.set
//! "a fox in a meadow" plakat.generate          \ leaves handle 1
//! "a fox in a meadow, painterly oil"           \ refinement prompt
//!   1                                          \ reuse handle
//!   plakat.img2img                             \ leaves handle 2
//!   "fox-refined.png" plakat.save
//! ```
//!
//! The input image is **not** consumed (the handle stays in the
//! registry), so a script can img2img the same source through
//! multiple variations.

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};
use crate::scripting::script_entry;

const TAG: &str = "plakat.img2img";

pub fn plakat_img2img(vm: &mut VM) -> BundResult<'_> {
    do_plakat_img2img(vm).map_err(to_bund_err)
}

fn do_plakat_img2img(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Top pops first: input (path or handle).
    let input_v = pull(vm, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;

    // Pull context up front so we don't hold the lock across the
    // async work. The handle→tempfile materialisation also happens
    // here, before the async block: the tempfile guard must outlive
    // the pipeline call below.
    let (model, device, config) = with_ctx(|ctx| {
        (
            ctx.loaded_model.clone(),
            ctx.device.clone(),
            ctx.config.clone(),
        )
    })?;
    let model = model.ok_or_else(|| {
        anyhow::anyhow!(
            "{TAG}: no model loaded. Call `\"sd15\" plakat.load` (or \
             your model of choice) before `plakat.img2img`."
        )
    })?;

    // Resolve the input. Two paths:
    //   STRING  → use the string as a filesystem path
    //   INTEGER → look up the image handle, write to tempfile,
    //             use that path. Tempfile guard binds to this
    //             stack frame so it outlives img2img_one's
    //             pipeline read.
    let _tempfile_guard;
    let input_path: std::path::PathBuf = match input_v.dt {
        types::STRING => {
            let s = value_to_string(input_v, "input", TAG)?;
            std::path::PathBuf::from(s)
        }
        types::INTEGER => {
            let handle = input_v.cast_int().unwrap_or(0);
            let img = with_ctx(|ctx| ctx.image_at(handle).cloned())??;
            let tmp = tempfile::Builder::new()
                .prefix("plakat-script-handle-")
                .suffix(".png")
                .tempfile()
                .map_err(|e| {
                    anyhow::anyhow!("{TAG}: creating tempfile for handle {handle}: {e}")
                })?;
            img.save(tmp.path()).map_err(|e| {
                anyhow::anyhow!(
                    "{TAG}: writing handle {handle} to tempfile: {e}"
                )
            })?;
            let path = tmp.path().to_path_buf();
            _tempfile_guard = tmp; // keep the file alive until fn exit
            path
        }
        _ => {
            anyhow::bail!(
                "{TAG}: input must be a string path or an integer handle \
                 (got rust_dynamic dt = {})",
                input_v.dt
            );
        }
    };

    // Async bridge identical to plakat.generate's.
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: no tokio runtime in scope (eval must run on a \
             multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let img = tokio::task::block_in_place(|| {
        handle.block_on(script_entry::img2img_one(
            &model,
            &prompt,
            &input_path,
            device,
            &config,
        ))
    })?;

    let handle_int = with_ctx_mut(|ctx| ctx.push_image(img))?;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} via model {model:?} from {}",
        input_path.display()
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
