//! MAP-4/5 — `plakat.map.*` scripting words. Render a map from a committed
//! `MapSpec` into an image handle, so a Bund script can produce maps alongside
//! generated images (then `plakat.save` writes them).
//!
//!   `plakat.map.layout  ( style  -- )`   set the town plan (radial|grid|organic|none)
//!   `plakat.map.erosion ( amount -- )`   set natural-feature erosion (0..>1)
//!   `plakat.map.render  ( spec-path style -- handle )`   linework map (no GPU)
//!   `plakat.map.paint   ( spec-path style -- handle )`   SD-painted map (GPU)
//!
//! These mirror `--map-urban-layout` / `--map-erosion` / `--map-render(-sd)`.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
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

/// Load a `MapSpec` and apply the script's layout/erosion overrides.
fn load_spec_with_overrides(path: &str) -> anyhow::Result<crate::map::spec::MapSpec> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading map spec {path}"))?;
    let mut spec: crate::map::spec::MapSpec =
        serde_json::from_str(&text).with_context(|| format!("parsing MapSpec {path}"))?;
    with_ctx(|ctx| {
        if let Some(l) = &ctx.map_layout {
            spec.urban.get_or_insert_with(Default::default).layout = Some(l.clone());
        }
        if let Some(e) = ctx.map_erosion {
            spec.terrain.erosion = Some(e);
        }
    })?;
    Ok(spec)
}

/// `plakat.map.paint ( spec-path style -- handle )` — SD-painted map (img2img +
/// Canny, the `render_sd` path) → image handle. Requires a GPU build.
pub fn plakat_map_paint(vm: &mut VM) -> BundResult<'_> {
    do_plakat_map_paint(vm).map_err(to_bund_err)
}

fn do_plakat_map_paint(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const PTAG: &str = "plakat.map.paint";
    require_depth(vm, 2, PTAG)?;
    let style_s = value_to_string(pull(vm, PTAG)?, "style", PTAG)?;
    let path_s = value_to_string(pull(vm, PTAG)?, "spec-path", PTAG)?;
    let style = crate::map::render::Style::named(&style_s)?;
    let spec = load_spec_with_overrides(&path_s)?;

    let (device, seed) = with_ctx(|ctx| (ctx.device.clone(), ctx.config.seed.unwrap_or(42)))?;
    let opts = crate::map::render_sd::SdOptions::default();
    let out = std::env::temp_dir().join(format!("plakat-map-paint-{seed}.png"));

    // render_sd is async; bridge it the same way the cascade/control words do.
    let rt = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{PTAG}: no tokio runtime: {e}"))?;
    tokio::task::block_in_place(|| {
        rt.block_on(crate::map::render_sd::render_sd(&spec, seed, style, &opts, device, &out))
    })
    .context("plakat.map.paint: SD render")?;

    let img = image::open(&out).with_context(|| format!("{PTAG}: reading {}", out.display()))?.to_rgb8();
    let _ = std::fs::remove_file(&out);
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))?;
    tracing::info!(target: "plakat", "{PTAG}: painted {path_s} ({style_s}) → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.map.tiles ( spec-path style out-dir -- count )` — 1.14.0-B: slice the
/// world into seamless tiles (`tile_r{R}_c{C}.png` + `world.png`) over the spec's
/// `tile_grid`, into `out-dir`. Pushes the tile count. Mirrors `--map-render-tiles`.
pub fn plakat_map_tiles(vm: &mut VM) -> BundResult<'_> {
    do_plakat_map_tiles(vm).map_err(to_bund_err)
}

fn do_plakat_map_tiles(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TTAG: &str = "plakat.map.tiles";
    require_depth(vm, 3, TTAG)?;
    // Stack bottom→top: spec-path, style, out-dir.
    let dir_s = value_to_string(pull(vm, TTAG)?, "out-dir", TTAG)?;
    let style_s = value_to_string(pull(vm, TTAG)?, "style", TTAG)?;
    let path_s = value_to_string(pull(vm, TTAG)?, "spec-path", TTAG)?;

    let style = crate::map::render::Style::named(&style_s)?;
    let spec = load_spec_with_overrides(&path_s)?;
    let seed = with_ctx(|ctx| ctx.config.seed.unwrap_or(42))?;

    let n = crate::map::save_world_tiles(&spec, seed, style, std::path::Path::new(&dir_s))
        .with_context(|| format!("{TTAG}: tiling {path_s} into {dir_s}"))?;
    tracing::info!(target: "plakat", "{TTAG}: {path_s} ({style_s}) → {n} tile(s) + world.png in {dir_s}");
    push(vm, Value::from_int(n as i64));
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
    let spec = load_spec_with_overrides(&path_s)?;

    let handle = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let seed = ctx.config.seed.unwrap_or(42);
        // Kind-routed: a `urban` spec renders the town map, else the geographic map.
        let img = crate::map::render_map_image(&spec, seed, style)?;
        Ok(ctx.push_image(image::DynamicImage::ImageRgb8(img)))
    })??;

    tracing::info!(target: "plakat", "{TAG}: rendered {path_s} ({style_s}) → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}
