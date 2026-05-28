//! v0.21 phase 6: `plakat.upscale ( handle scale -- handle )`.
//!
//! Resize a registered image. Two flavours, dispatched by the
//! type of the `scale` arg:
//!
//! - **Integer** (2 or 4) — Lanczos-3 resize, pure-CPU. Pre-v0.26
//!   behaviour, preserved for backwards-compat.
//! - **String** (v0.26 phase 9) — Real-ESRGAN ML upscaler.
//!   Accepted values:
//!     - `"real-esrgan-x2"` (RealESRGAN x2plus, general)
//!     - `"real-esrgan-x4"` (RealESRGAN x4plus, general)
//!     - `"real-esrgan-anime-x4"` (RealESRGAN x4plus-anime-6B,
//!       anime / illustration targets)
//!   First-time use downloads ~17-65 MB of weights from HF;
//!   cached afterward.
//!
//! The source handle is **not** consumed; the upscaled image is
//! pushed as a new handle. Same lifetime contract as
//! `plakat.img2img` / `plakat.portrait`: the source stays
//! addressable for additional saves or downstream chaining.
//!
//! ```bund
//! "sdxl" plakat.load
//! "a fox in a meadow" plakat.generate     // handle 1, 1024x1024
//!   2 plakat.upscale                       // Lanczos x2 → handle 2
//!   "fox-2k.png" plakat.save
//! 1 "real-esrgan-x4" plakat.upscale        // ML x4 → handle 3
//!   "fox-4k.png" plakat.save
//! ```
//!
//! ML path: async (model load is network + compute). Requires a
//! tokio runtime in scope — same constraint as `plakat.generate`.
//! Lanczos path: pure CPU, no async bridge.

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_int,
};

const TAG: &str = "plakat.upscale";

pub fn plakat_upscale(vm: &mut VM) -> BundResult<'_> {
    do_plakat_upscale(vm).map_err(to_bund_err)
}

fn do_plakat_upscale(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Top pops first: scale. Then handle.
    let scale_v = pull(vm, TAG)?;
    let handle_v = pull(vm, TAG)?;
    let handle = value_to_int(handle_v, "handle", TAG)?;

    // Dispatch on scale arg type.
    match scale_v.dt {
        types::INTEGER => do_lanczos(vm, handle, scale_v.cast_int().unwrap_or(0)),
        types::STRING => {
            let method_str = scale_v
                .cast_string()
                .map_err(|e| anyhow::anyhow!("{TAG}: scale string conversion failed: {e}"))?;
            do_ml(vm, handle, &method_str)
        }
        _ => anyhow::bail!(
            "{TAG}: scale must be an integer (2/4 for Lanczos) OR a string \
             (real-esrgan-x2 / real-esrgan-x4 / real-esrgan-anime-x4 for ML \
             upscaling). Got rust_dynamic dt = {}",
            scale_v.dt
        ),
    }
}

/// Lanczos x2 / x4 — pre-v0.26 behaviour. Pure CPU, no async.
fn do_lanczos<'vm>(
    vm: &'vm mut VM,
    handle: i64,
    scale: i64,
) -> anyhow::Result<&'vm mut VM> {
    if !matches!(scale, 2 | 4) {
        anyhow::bail!(
            "{TAG}: integer scale must be 2 or 4 (got {scale}). For other \
             scales, use the Real-ESRGAN string variants (real-esrgan-x2 / \
             real-esrgan-x4 / real-esrgan-anime-x4)."
        );
    }
    let src = with_ctx(|ctx| ctx.image_at(handle).cloned())??;
    let (w, h) = (src.width(), src.height());
    let nw = w
        .checked_mul(scale as u32)
        .ok_or_else(|| anyhow::anyhow!("{TAG}: width overflow ({w} × {scale})"))?;
    let nh = h
        .checked_mul(scale as u32)
        .ok_or_else(|| anyhow::anyhow!("{TAG}: height overflow ({h} × {scale})"))?;

    let upscaled = src.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3);
    let new_handle = with_ctx_mut(|ctx| ctx.push_image(upscaled))?;
    tracing::info!(
        target: "plakat",
        "{TAG}: handle {handle} ({w}x{h}) → handle {new_handle} ({nw}x{nh}) Lanczos x{scale}"
    );
    push(vm, Value::from_int(new_handle));
    Ok(vm)
}

/// v0.26 phase 9: Real-ESRGAN ML upscaling. Uses the existing
/// `imaging::upscale::ml_upscale` helper via tempfile round-trip —
/// the helper is file-path based (uses
/// `EsrganPipeline::upscale_file`). Round-trip cost is one PNG
/// encode + one decode per call; small compared to the ML
/// inference itself.
fn do_ml<'vm>(
    vm: &'vm mut VM,
    handle: i64,
    method_str: &str,
) -> anyhow::Result<&'vm mut VM> {
    use std::str::FromStr;

    let method = crate::imaging::upscale::Method::from_str(method_str)
        .map_err(|e| anyhow::anyhow!("{TAG}: parsing scale {method_str:?}: {e}"))?;
    if !method.is_ml() {
        anyhow::bail!(
            "{TAG}: scale {method_str:?} parses to {method:?} which isn't \
             an ML method. For Lanczos, pass an integer (2 or 4) instead."
        );
    }

    // Snapshot ctx state needed for the call.
    let (src, device) = with_ctx(|ctx| {
        let img = ctx.image_at(handle).cloned()?;
        Ok::<_, anyhow::Error>((img, ctx.device.clone()))
    })??;
    let (w, h) = (src.width(), src.height());

    // Write source to a tempfile; ml_upscale reads/writes files.
    // Tempfiles drop at scope-exit; OS reclaims.
    let in_tmp = tempfile::Builder::new()
        .prefix("plakat-script-upscale-in-")
        .suffix(".png")
        .tempfile()
        .map_err(|e| anyhow::anyhow!("{TAG}: source tempfile: {e}"))?;
    src.save(in_tmp.path()).map_err(|e| {
        anyhow::anyhow!("{TAG}: writing source tempfile: {e}")
    })?;

    let out_tmp = tempfile::Builder::new()
        .prefix("plakat-script-upscale-out-")
        .suffix(".png")
        .tempfile()
        .map_err(|e| anyhow::anyhow!("{TAG}: dest tempfile: {e}"))?;
    let out_path = out_tmp.path().to_path_buf();

    let in_path = in_tmp.path().to_path_buf();
    let rt_handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: ML upscale requires a tokio runtime in scope (eval must \
             run on a multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let (_iw, _ih, ow, oh) = tokio::task::block_in_place(|| {
        rt_handle.block_on(crate::imaging::upscale::ml_upscale(
            &in_path, &out_path, method, &device,
        ))
    })?;

    // Read the upscaled PNG back + register as new handle.
    let upscaled = image::open(&out_path).map_err(|e| {
        anyhow::anyhow!("{TAG}: reading upscaled tempfile: {e}")
    })?;
    let new_handle = with_ctx_mut(|ctx| ctx.push_image(upscaled))?;
    tracing::info!(
        target: "plakat",
        "{TAG}: handle {handle} ({w}x{h}) → handle {new_handle} ({ow}x{oh}) {method:?}"
    );
    push(vm, Value::from_int(new_handle));
    Ok(vm)
}
