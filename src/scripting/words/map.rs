//! MAP-4/5 — `plakat.map.*` scripting words. Render a map from a committed
//! `MapSpec` into an image handle, so a Bund script can produce maps alongside
//! generated images (then `plakat.save` writes them).
//!
//!   `plakat.map.layout  ( style  -- )`   set the town plan (radial|grid|organic|none)
//!   `plakat.map.erosion ( amount -- )`   set natural-feature erosion (0..>1)
//!   `plakat.map.render  ( spec-path style -- handle )`   render → image handle
//!
//! These mirror `--map-urban-layout` / `--map-erosion`. The painted SD path lives
//! behind the `--map-render-sd` CLI / the `type: map` scenario task; scripting
//! stays on the GPU-free render to match the image-handle model.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_float, value_to_string};

const TAG: &str = "plakat.map.render";

/// `plakat.map.layout ( style -- )` — set the town street plan override.
pub fn plakat_map_layout(vm: &mut VM) -> BundResult<'_> {
    do_plakat_map_layout(vm).map_err(to_bund_err)
}

fn do_plakat_map_layout(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, "plakat.map.layout")?;
    let style = value_to_string(pull(vm, "plakat.map.layout")?, "style", "plakat.map.layout")?;
    with_ctx_mut(|ctx| {
        // `none`/`auto` clears the override (back to spec / inference).
        ctx.map_layout = if matches!(style.to_ascii_lowercase().as_str(), "none" | "auto" | "") {
            None
        } else {
            Some(style.clone())
        };
    })?;
    tracing::info!(target: "plakat", "plakat.map.layout: {style}");
    Ok(vm)
}

/// `plakat.map.erosion ( amount -- )` — set the natural-feature erosion override.
pub fn plakat_map_erosion(vm: &mut VM) -> BundResult<'_> {
    do_plakat_map_erosion(vm).map_err(to_bund_err)
}

fn do_plakat_map_erosion(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, "plakat.map.erosion")?;
    let amount = value_to_float(pull(vm, "plakat.map.erosion")?, "amount", "plakat.map.erosion")? as f32;
    with_ctx_mut(|ctx| {
        ctx.map_erosion = Some(amount);
    })?;
    tracing::info!(target: "plakat", "plakat.map.erosion: {amount}");
    Ok(vm)
}

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
    let mut spec: crate::map::spec::MapSpec =
        serde_json::from_str(&text).with_context(|| format!("{TAG}: parsing MapSpec {path_s}"))?;

    let handle = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        // Apply the script's map overrides (set via plakat.map.layout / .erosion).
        if let Some(l) = &ctx.map_layout {
            spec.urban.get_or_insert_with(Default::default).layout = Some(l.clone());
        }
        if let Some(e) = ctx.map_erosion {
            spec.terrain.erosion = Some(e);
        }
        let seed = ctx.config.seed.unwrap_or(42);
        // Kind-routed: a `urban` spec renders the town map, else the geographic map.
        let img = crate::map::render_map_image(&spec, seed, style)?;
        Ok(ctx.push_image(image::DynamicImage::ImageRgb8(img)))
    })??;

    tracing::info!(target: "plakat", "{TAG}: rendered {path_s} ({style_s}) → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}
