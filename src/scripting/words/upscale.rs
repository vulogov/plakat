//! v0.21 phase 6: `plakat.upscale ( handle scale -- handle )`.
//!
//! Lanczos-3 resize of a registered image. Scale must be 2 or 4
//! (matches the spec in RFC §5.1). Real-ESRGAN ML upscaling on
//! the standalone `plakat.upscale` word is still deferred (v0.25+
//! carry; `plakat.hires` already exposes ML upscalers via
//! `hires_upscaler` since v0.22 phase 8).
//!
//! The source handle is **not** consumed; the upscaled image is
//! pushed as a new handle. Same lifetime contract as
//! `plakat.img2img` / `plakat.portrait`: the source stays
//! addressable for additional saves or downstream chaining.
//!
//! ```bund
//! "sdxl" plakat.load
//! "a fox in a meadow" plakat.generate  // handle 1, 1024x1024
//!   2 plakat.upscale                    // handle 2, 2048x2048
//!   "fox-2k.png" plakat.save
//!   4 plakat.upscale                    // wait — this upscales
//!                                       // the 1024 source, not
//!                                       // the 2048; handle 1 is
//!                                       // still on the stack
//!                                       // first. Use the chain
//!                                       // explicitly:
//! 1 4 plakat.upscale "fox-4k.png" plakat.save
//! ```
//!
//! No async bridge needed — the resize is pure CPU + image-crate
//! work. Unlike generate / img2img / portrait, this word doesn't
//! require a tokio runtime in scope.

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
    let scale = value_to_int(scale_v, "scale", TAG)?;
    let handle = value_to_int(handle_v, "handle", TAG)?;

    if !matches!(scale, 2 | 4) {
        anyhow::bail!(
            "{TAG}: scale must be 2 or 4 (got {scale}). v0.21 phase 6 ships \
             Lanczos x2/x4 only; arbitrary scales + ML upscaling land in v0.22."
        );
    }

    // Look up + clone the source. Cloning here lets us drop the
    // read lock before the resize; the resize itself can take
    // measurable wall time for SDXL-sized inputs (1024² → 4096²
    // Lanczos isn't free) and we don't want to hold a read lock
    // across it.
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
