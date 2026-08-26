//! 6.21.0 (RFC FACESWAP-3 S3) — `plakat.faceswap` scripting word. Swap the largest face in a scene with
//! the identity from a source face photo, and push the result as an image handle.
//!
//!   `plakat.faceswap ( scene source out -- handle )`   swap `source`'s face into `scene` → `out`, push it

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

pub fn plakat_faceswap(vm: &mut VM) -> BundResult<'_> {
    do_faceswap(vm).map_err(to_bund_err)
}

fn do_faceswap(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.faceswap";
    require_depth(vm, 3, TAG)?;
    let out_path = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let source = value_to_string(pull(vm, TAG)?, "source", TAG)?;
    let scene = value_to_string(pull(vm, TAG)?, "scene", TAG)?;
    let handle = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let img = crate::scripting::script_entry::faceswap_one(ctx, &scene, &source)?;
        img.to_rgb8().save(&out_path).with_context(|| format!("{TAG}: writing {out_path}"))?;
        Ok(ctx.push_image(img))
    })??;
    tracing::info!(target: "plakat", "{TAG}: {scene} ← {source} → {out_path} → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}
