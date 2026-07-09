//! `plakat.transparent ( subject -- handle )` — U2Net cut-out to a transparent background.
//!
//! Matte the salient subject of `subject` (string path or image handle) and push a handle to
//! the RGBA result (the CLI `plakat transparent`). Feed the handle to `plakat.relight`,
//! `plakat.save` (to a `.png`/`.webp`), or an artefact stack. Self-contained (loads U2Net); no
//! `plakat.load` required.
//!
//! ```bund
//! "photo.jpg" plakat.transparent "cutout.png" plakat.save
//! ```

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, resolve_image_arg, to_bund_err,
};

const TAG: &str = "plakat.transparent";

pub fn plakat_transparent(vm: &mut VM) -> BundResult<'_> {
    do_plakat_transparent(vm).map_err(to_bund_err)
}

fn do_plakat_transparent(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let subject_v = pull(vm, TAG)?;
    let (_subject_guard, subject_path) = resolve_image_arg(subject_v, "subject", TAG)?;

    let device = with_ctx(|ctx| ctx.device.clone())?;

    // matting::cutout writes an RGBA PNG (alpha needs a real container); read it back as a handle.
    let out_tmp = tempfile::Builder::new()
        .prefix("plakat-script-transparent-")
        .suffix(".png")
        .tempfile()
        .map_err(|e| anyhow::anyhow!("{TAG}: creating output tempfile: {e}"))?;
    let out_path = out_tmp.path().to_path_buf();

    let handle = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{TAG}: no tokio runtime in scope. {e}"))?;
    tokio::task::block_in_place(|| {
        handle.block_on(crate::pipelines::matting::cutout(&subject_path, &out_path, false, &device))
    })?;

    let img = image::open(&out_path)
        .map_err(|e| anyhow::anyhow!("{TAG}: reading cut-out {}: {e}", out_path.display()))?;
    let handle_int = with_ctx_mut(|ctx| ctx.push_image(img))?;
    tracing::info!(target: "plakat", "{TAG}: cut out handle {handle_int}");
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
