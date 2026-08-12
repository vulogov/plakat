//! 6.10.0 — `plakat.naturalize` scripting word. Apply the analog naturalize post-pass to an image and
//! push the result as an image handle.
//!
//!   `plakat.naturalize ( in-path spec out-path -- handle )`   naturalize `in` by `spec` → `out`, push it

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

pub fn plakat_naturalize(vm: &mut VM) -> BundResult<'_> {
    do_naturalize(vm).map_err(to_bund_err)
}

fn do_naturalize(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.naturalize";
    require_depth(vm, 3, TAG)?;
    let out_path = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let spec = value_to_string(pull(vm, TAG)?, "spec", TAG)?;
    let in_path = value_to_string(pull(vm, TAG)?, "in-path", TAG)?;
    let params = crate::naturalize::from_spec(&spec);
    let img = image::open(&in_path).with_context(|| format!("{TAG}: reading {in_path}"))?.to_rgb8();
    let out = crate::naturalize::apply(&img, &params);
    out.save(&out_path).with_context(|| format!("{TAG}: writing {out_path}"))?;
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(out)))?;
    tracing::info!(target: "plakat", "{TAG}: {in_path} [{spec}] → {out_path} → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}
