//! `plakat.portrait ( prompt photo -- handle )`.
//!
//! Identity-preserving portrait via the cached pipeline's
//! IP-Adapter-Plus-Face identity encoder. v0.22 phase 1 picks
//! the identity at cache-load time based on the model alias:
//! SD 1.5 → PlusFace, SDXL → PlusFaceSdxl, SD 2.1 → no
//! identity (no shipped Plus-Face checkpoint — `plakat.portrait`
//! bails at generate time on sd21).
//!
//! Same two input shapes as `plakat.img2img`: string path or
//! integer handle.

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
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
    let photo_v = pull(vm, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;

    let (_tempfile_guard, photo_path): (
        Option<tempfile::NamedTempFile>,
        std::path::PathBuf,
    ) = match photo_v.dt {
        types::STRING => {
            let s = value_to_string(photo_v, "photo", TAG)?;
            (None, std::path::PathBuf::from(s))
        }
        types::INTEGER => {
            let handle = photo_v.cast_int().unwrap_or(0);
            let img = with_ctx_mut(|ctx| {
                ctx.image_at(handle).cloned()
            })??;
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
            (Some(tmp), path)
        }
        _ => {
            anyhow::bail!(
                "{TAG}: photo must be a string path or an integer handle \
                 (got rust_dynamic dt = {})",
                photo_v.dt
            );
        }
    };

    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let img = script_entry::portrait_one(ctx, &prompt, &photo_path)?;
        Ok(ctx.push_image(img))
    })??;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} from {}",
        photo_path.display()
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
