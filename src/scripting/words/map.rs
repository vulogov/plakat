//! MAP-4 — `plakat.map.*` scripting words. Render a map from a committed `MapSpec`
//! into an image handle, so a Bund script can produce maps alongside generated
//! images (then `plakat.save` writes them).
//!
//! `plakat.map.render ( spec-path style -- handle )` — the deterministic linework
//! map (no GPU). The painted SD path lives behind the `--map-render-sd` CLI / the
//! `type: map` scenario task; scripting stays on the GPU-free render to match the
//! image-handle model.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

const TAG: &str = "plakat.map.render";

pub fn plakat_map_render(vm: &mut VM) -> BundResult<'_> {
    do_plakat_map_render(vm).map_err(to_bund_err)
}

fn do_plakat_map_render(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Stack: bottom = spec-path (pushed first), top = style (pushed second).
    let style_v = pull(vm, TAG)?;
    let path_v = pull(vm, TAG)?;
    let style_s = value_to_string(style_v, "style", TAG)?;
    let path_s = value_to_string(path_v, "spec-path", TAG)?;

    let style = crate::map::render::Style::named(&style_s)?;
    let text = std::fs::read_to_string(&path_s)
        .with_context(|| format!("{TAG}: reading map spec {path_s}"))?;
    let spec: crate::map::spec::MapSpec =
        serde_json::from_str(&text).with_context(|| format!("{TAG}: parsing MapSpec {path_s}"))?;

    let handle = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let seed = ctx.config.seed.unwrap_or(42);
        let img = crate::map::render::render(&spec, seed, style)?;
        Ok(ctx.push_image(image::DynamicImage::ImageRgb8(img)))
    })??;

    tracing::info!(target: "plakat", "{TAG}: rendered {path_s} ({style_s}) → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}
