//! v0.21 phase 5: `plakat.portrait ( prompt photo -- handle )`.
//!
//! Renders an identity-preserving portrait via IP-Adapter-Plus-Face.
//! Same two input shapes as `plakat.img2img`:
//!
//! 1. **Filesystem path** (string) — read directly from disk.
//! 2. **Image handle** (integer) — materialise from
//!    `ScriptCtx.images[handle-1]` to a tempfile.
//!
//! Identity strategy is auto-picked from the loaded model:
//! SD 1.5 → `PlusFace`, SDXL → `PlusFaceSdxl`. SD 2.1 bails with
//! a model-suggestion message (no shipped Plus-Face checkpoint).
//!
//! Phase 5 MVP per RFC §5.1: single reference photo only.
//! FaceID variants + multi-photo blends + manual landmarks are
//! deferred to v0.22.
//!
//! Strength knob: `plakat.config.set "face_strength"` in `[0, 1]`
//! (default 0.8). 1.0 = image tokens carry full weight; 0.0
//! collapses portrait into a text-only generate.
//!
//! ```bund
//! "sdxl" plakat.load
//! "a renaissance oil portrait, ornate frame" "./me.jpg" plakat.portrait
//!   "portrait.png" plakat.save
//! ```

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};
use crate::scripting::script_entry;

const TAG: &str = "plakat.portrait";

pub fn plakat_portrait(vm: &mut VM) -> BundResult<'_> {
    do_plakat_portrait(vm).map_err(to_bund_err)
}

fn do_plakat_portrait(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Top pops first: photo (path or handle).
    let photo_v = pull(vm, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;

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
             `\"sdxl\" plakat.load`) before `plakat.portrait`."
        )
    })?;

    // Same handle-vs-path dispatch as plakat.img2img — keep the
    // tempfile guard bound to this frame so the materialised PNG
    // outlives the pipeline read inside portrait_one.
    let _tempfile_guard;
    let photo_path: std::path::PathBuf = match photo_v.dt {
        types::STRING => {
            let s = value_to_string(photo_v, "photo", TAG)?;
            std::path::PathBuf::from(s)
        }
        types::INTEGER => {
            let handle = photo_v.cast_int().unwrap_or(0);
            let img = with_ctx(|ctx| ctx.image_at(handle).cloned())??;
            let tmp = tempfile::Builder::new()
                .prefix("plakat-script-portrait-photo-")
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
            _tempfile_guard = tmp;
            path
        }
        _ => {
            anyhow::bail!(
                "{TAG}: photo must be a string path or an integer handle \
                 (got rust_dynamic dt = {})",
                photo_v.dt
            );
        }
    };

    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: no tokio runtime in scope (eval must run on a \
             multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let img = tokio::task::block_in_place(|| {
        handle.block_on(script_entry::portrait_one(
            &model,
            &prompt,
            &photo_path,
            device,
            &config,
        ))
    })?;

    let handle_int = with_ctx_mut(|ctx| ctx.push_image(img))?;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} via model {model:?} from {}",
        photo_path.display()
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
