//! `plakat.segment ( subject x y -- handle )` — SAM/MobileSAM point segmentation.
//!
//! Segment the subject at pixel `(x, y)` and push a handle to the binary mask (255 = selected,
//! at the input's resolution) — ready to feed `plakat.inpaint`. `subject` is a string path or
//! image handle. Self-contained (loads MobileSAM); no `plakat.load` required. For richer control
//! (multiple points, depth bands, grow/feather) use the `plakat segment` CLI.
//!
//! ```bund
//! "photo.png" 256.0 340.0 plakat.segment       // mask around the point
//!   // ... feed the handle into an inpaint mask ...
//! ```

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, resolve_image_arg, to_bund_err, value_to_float,
};

const TAG: &str = "plakat.segment";

pub fn plakat_segment(vm: &mut VM) -> BundResult<'_> {
    do_plakat_segment(vm).map_err(to_bund_err)
}

fn do_plakat_segment(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 3, TAG)?;
    // Top pops first: y, then x, then subject.
    let y_v = pull(vm, TAG)?;
    let x_v = pull(vm, TAG)?;
    let subject_v = pull(vm, TAG)?;
    let y = value_to_float(y_v, "y", TAG)?;
    let x = value_to_float(x_v, "x", TAG)?;
    let (_subject_guard, subject_path) = resolve_image_arg(subject_v, "subject", TAG)?;

    let device = with_ctx(|ctx| ctx.device.clone())?;

    let out_tmp = tempfile::Builder::new()
        .prefix("plakat-script-segment-")
        .suffix(".png")
        .tempfile()
        .map_err(|e| anyhow::anyhow!("{TAG}: creating output tempfile: {e}"))?;
    let out_path = out_tmp.path().to_path_buf();

    let points = vec![crate::pipelines::sam::PointPrompt { x, y, foreground: true }];
    let handle = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{TAG}: no tokio runtime in scope. {e}"))?;
    tokio::task::block_in_place(|| {
        handle.block_on(crate::pipelines::sam::segment(
            &subject_path,
            &out_path,
            &points,
            false, // invert
            0,     // grow
            0,     // feather
            &device,
        ))
    })?;

    let img = image::open(&out_path)
        .map_err(|e| anyhow::anyhow!("{TAG}: reading mask {}: {e}", out_path.display()))?;
    let handle_int = with_ctx_mut(|ctx| ctx.push_image(img))?;
    tracing::info!(target: "plakat", "{TAG}: mask handle {handle_int} at ({x},{y})");
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
