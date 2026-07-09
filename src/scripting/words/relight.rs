//! `plakat.relight ( subject prompt -- handle )` — IC-Light re-illumination.
//!
//! Relight a cut-out subject under a described lighting condition (the CLI `plakat relight`).
//! `subject` is a string path or an image handle (ideally an RGBA cut-out — see
//! `plakat.transparent`); `prompt` is the lighting/scene description. Size, steps, guidance,
//! seed, and negative are read from `plakat.config.set`. IC-Light is self-contained (loads its
//! own SD 1.5 + IC-Light weights), so no `plakat.load` is required.
//!
//! ```bund
//! "subject.png" plakat.transparent            // cut out first
//!   "warm sunset light from the left" plakat.relight
//!   "relit.png" plakat.save
//! ```

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, resolve_image_arg, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.relight";

pub fn plakat_relight(vm: &mut VM) -> BundResult<'_> {
    do_plakat_relight(vm).map_err(to_bund_err)
}

fn do_plakat_relight(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Top pops first: prompt, then subject.
    let prompt_v = pull(vm, TAG)?;
    let subject_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;
    let (_subject_guard, subject_path) = resolve_image_arg(subject_v, "subject", TAG)?;

    let (device, negative, width, height, steps, guidance, seed) = with_ctx(|ctx| {
        (
            ctx.device.clone(),
            ctx.config.negative.clone(),
            ctx.config.width,
            ctx.config.height,
            ctx.config.steps,
            ctx.config.guidance,
            ctx.config.seed.unwrap_or(0),
        )
    })?;

    let handle = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{TAG}: no tokio runtime in scope. {e}"))?;
    let (pixels, w, h) = tokio::task::block_in_place(|| {
        handle.block_on(async {
            let pipe = crate::pipelines::ic_light::Pipeline::load(device)
                .await
                .map_err(|e| anyhow::anyhow!("{TAG}: loading IC-Light: {e}"))?;
            pipe.relight(&subject_path, &prompt, &negative, width, height, steps, guidance, seed)
        })
    })?;

    let rgb = image::RgbImage::from_raw(w, h, pixels)
        .ok_or_else(|| anyhow::anyhow!("{TAG}: relit image buffer size mismatch"))?;
    let handle_int = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(rgb)))?;
    tracing::info!(target: "plakat", "{TAG}: rendered handle {handle_int} ({w}x{h})");
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
