//! `plakat.img2img ( prompt input -- handle )`.
//!
//! Re-imagine an existing image. `input` is either a string
//! filesystem path or an integer handle to a prior generation;
//! handles materialise to a tempfile bound to this host-fn's
//! stack frame so the file outlives the pipeline read.
//!
//! v0.22 phase 1: uses the cached pipeline — no model reload.

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
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

    // Materialise integer handles to a tempfile guarded by this
    // stack frame; pass strings straight through. We need the
    // input_path resolved before entering with_ctx_mut so the
    // tempfile's lifetime extends past the pipeline read.
    let (_tempfile_guard, input_path): (
        Option<tempfile::NamedTempFile>,
        std::path::PathBuf,
    ) = match input_v.dt {
        types::STRING => {
            let s = value_to_string(input_v, "input", TAG)?;
            (None, std::path::PathBuf::from(s))
        }
        types::INTEGER => {
            let handle = input_v.cast_int().unwrap_or(0);
            let img = with_ctx_mut(|ctx| {
                ctx.image_at(handle).cloned()
            })??;
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
            (Some(tmp), path)
        }
        _ => {
            anyhow::bail!(
                "{TAG}: input must be a string path or an integer handle \
                 (got rust_dynamic dt = {})",
                input_v.dt
            );
        }
    };

    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let img = script_entry::img2img_one(ctx, &prompt, &input_path)?;
        Ok(ctx.push_image(img))
    })??;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} from {}",
        input_path.display()
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
