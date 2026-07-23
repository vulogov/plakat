//! 4.3-ecosystem — `plakat.fractal.*` scripting words. Render a fractal (Track A,
//! pure-CPU) or paint one (Track B, GPU) from a Bund script into an image handle, so a
//! script can produce fractals alongside generated images and hand them to the existing
//! `plakat.save` / `plakat.relight` / `plakat.upscale` words.
//!
//!   `plakat.fractal.size    ( w h -- )`                        output-size override
//!   `plakat.fractal.render  ( spec-source -- handle )`         Track-A render (no GPU)
//!   `plakat.fractal.compose ( spec-source mode rows cols -- handle )`  grid contact sheet
//!   `plakat.fractal.paint   ( spec-source -- handle )`         Track-B AI paint (GPU)
//!   `plakat.fractal.animate ( spec-source mode frames fps out-path -- out-path )`  video/gif
//!
//! `spec-source` is anything `--control-fractal` accepts: a spec `.json`/`.hjson` path, a
//! `kind` / `kind:preset` shorthand (`flame`, `ifs:barnsley-fern`, `raymarch:menger`), or
//! prose. `mode` mirrors `--fractal-compose` / `--fractal-animate`.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::fractals::{FractalSpec, control_source};
use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

/// Coerce a stack `Value` to an `i64` via bund's native type conversion
/// (`Value::conv`), so a numeric **string** (`"256"`), float, or integer all work.
/// Bund scripts routinely quote scalars — the `plakat.config.*` words take strings —
/// so the fractal words accept whichever spelling the author reaches for.
fn value_to_int_lenient(v: Value, field: &str, tag: &str) -> anyhow::Result<i64> {
    v.conv(rust_dynamic::types::INTEGER)
        .map_err(|e| anyhow::anyhow!("{tag}: arg {field:?} must be an integer ({e})"))?
        .cast_int()
        .map_err(|e| anyhow::anyhow!("{tag}: arg {field:?} must be an integer ({e})"))
}

/// Resolve a `spec-source` string and apply the script's size override (if any).
fn resolve_spec(source: &str) -> anyhow::Result<FractalSpec> {
    let mut spec = control_source::resolve(source)
        .with_context(|| format!("resolving fractal source {source:?}"))?;
    if let Some((w, h)) = with_ctx(|ctx| ctx.fractal_size)? {
        spec.width = w;
        spec.height = h;
    }
    Ok(spec)
}

/// `plakat.fractal.size ( w h -- )` — set the output-size override for later
/// `plakat.fractal.*` words. `0 0` clears it (back to each spec's own size).
pub fn plakat_fractal_size(vm: &mut VM) -> BundResult<'_> {
    do_plakat_fractal_size(vm).map_err(to_bund_err)
}

fn do_plakat_fractal_size(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.fractal.size";
    require_depth(vm, 2, TAG)?;
    // Stack bottom→top: w, h.
    let h = value_to_int_lenient(pull(vm, TAG)?, "h", TAG)?;
    let w = value_to_int_lenient(pull(vm, TAG)?, "w", TAG)?;
    with_ctx_mut(|ctx| {
        ctx.fractal_size = if w <= 0 || h <= 0 { None } else { Some((w as u32, h as u32)) };
    })?;
    tracing::info!(target: "plakat", "{TAG}: {w}x{h}");
    Ok(vm)
}

/// `plakat.fractal.render ( spec-source -- handle )` — Track-A CPU render → image handle.
pub fn plakat_fractal_render(vm: &mut VM) -> BundResult<'_> {
    do_plakat_fractal_render(vm).map_err(to_bund_err)
}

fn do_plakat_fractal_render(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.fractal.render";
    require_depth(vm, 1, TAG)?;
    let source = value_to_string(pull(vm, TAG)?, "spec-source", TAG)?;
    let spec = resolve_spec(&source)?;
    let r = crate::fractals::render_spec(&spec)
        .with_context(|| format!("{TAG}: rendering {source:?}"))?;
    let img = image::RgbImage::from_raw(r.width, r.height, r.pixels)
        .context("fractal render produced a malformed buffer")?;
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))?;
    tracing::info!(target: "plakat", "{TAG}: {source:?} ({}x{}) → handle {handle}", r.width, r.height);
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.fractal.compose ( spec-source mode rows cols -- handle )` — grid contact
/// sheet (Track-A). `mode` = julia-sweep | zoom-grid | palette-grid | variation-sweep.
pub fn plakat_fractal_compose(vm: &mut VM) -> BundResult<'_> {
    do_plakat_fractal_compose(vm).map_err(to_bund_err)
}

fn do_plakat_fractal_compose(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.fractal.compose";
    require_depth(vm, 4, TAG)?;
    // Stack bottom→top: spec-source, mode, rows, cols.
    let cols = value_to_int_lenient(pull(vm, TAG)?, "cols", TAG)?;
    let rows = value_to_int_lenient(pull(vm, TAG)?, "rows", TAG)?;
    let mode_s = value_to_string(pull(vm, TAG)?, "mode", TAG)?;
    let source = value_to_string(pull(vm, TAG)?, "spec-source", TAG)?;
    if rows < 1 || cols < 1 {
        anyhow::bail!("{TAG}: rows/cols must be >= 1 (got {rows}x{cols})");
    }
    let mode = crate::fractals::compose::ComposeMode::parse(&mode_s)?;
    let spec = resolve_spec(&source)?;
    let silent = crate::fractals::progress::silent();
    let r = crate::fractals::compose::compose(&spec, mode, rows as u32, cols as u32, &silent)
        .with_context(|| format!("{TAG}: composing {source:?}"))?;
    let img = image::RgbImage::from_raw(r.width, r.height, r.pixels)
        .context("fractal compose produced a malformed buffer")?;
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))?;
    tracing::info!(target: "plakat", "{TAG}: {source:?} {rows}x{cols} {mode_s} → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.fractal.paint ( spec-source -- handle )` — render Track A, then run the AI
/// paint pass (txt2img / ControlNet, per the spec's `ai` block) → image handle. GPU.
pub fn plakat_fractal_paint(vm: &mut VM) -> BundResult<'_> {
    do_plakat_fractal_paint(vm).map_err(to_bund_err)
}

fn do_plakat_fractal_paint(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.fractal.paint";
    require_depth(vm, 1, TAG)?;
    let source = value_to_string(pull(vm, TAG)?, "spec-source", TAG)?;
    let mut spec = resolve_spec(&source)?;
    spec.ai.enabled = true;
    let device = with_ctx(|ctx| ctx.device.clone())?;

    let scratch = tempfile::tempdir().context("fractal paint scratch dir")?;
    let base = scratch.path().join("fractal-base.png");
    let painted = scratch.path().join("fractal-painted.png");
    crate::fractals::render_to_file(&spec, &base)
        .with_context(|| format!("{TAG}: rendering base for {source:?}"))?;

    // run_ai_pass is async; bridge it the way the map/cascade words do.
    let rt = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{TAG}: no tokio runtime: {e}"))?;
    tokio::task::block_in_place(|| {
        rt.block_on(crate::fractals::ai_pass::run_ai_pass(&spec, &base, &painted, device))
    })
    .with_context(|| format!("{TAG}: AI paint pass for {source:?}"))?;

    let img = image::open(&painted)
        .with_context(|| format!("{TAG}: reading painted {}", painted.display()))?
        .to_rgb8();
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))?;
    tracing::info!(target: "plakat", "{TAG}: painted {source:?} → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.fractal.animate ( spec-source mode frames fps out-path -- out-path )` — render
/// an animation straight to a file (`.mp4`/`.webm` need ffmpeg, `.gif` never does). `mode`
/// = zoom | julia-sweep | param-sweep. Pushes the written path back for chaining.
pub fn plakat_fractal_animate(vm: &mut VM) -> BundResult<'_> {
    do_plakat_fractal_animate(vm).map_err(to_bund_err)
}

fn do_plakat_fractal_animate(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.fractal.animate";
    require_depth(vm, 5, TAG)?;
    // Stack bottom→top: spec-source, mode, frames, fps, out-path.
    let out_s = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let fps = value_to_int_lenient(pull(vm, TAG)?, "fps", TAG)?;
    let frames = value_to_int_lenient(pull(vm, TAG)?, "frames", TAG)?;
    let mode_s = value_to_string(pull(vm, TAG)?, "mode", TAG)?;
    let source = value_to_string(pull(vm, TAG)?, "spec-source", TAG)?;
    if frames < 2 {
        anyhow::bail!("{TAG}: frames must be >= 2 (got {frames})");
    }
    if fps < 1 {
        anyhow::bail!("{TAG}: fps must be >= 1 (got {fps})");
    }
    let mode = crate::fractals::animation::AnimMode::parse(&mode_s)?;
    let spec = resolve_spec(&source)?;
    let out = std::path::PathBuf::from(&out_s);
    let silent = crate::fractals::progress::silent();
    crate::fractals::animation::render_animation(&spec, mode, frames as u32, fps as u32, &out, &silent)
        .with_context(|| format!("{TAG}: animating {source:?} → {out_s}"))?;
    tracing::info!(target: "plakat", "{TAG}: {source:?} {mode_s} {frames}f@{fps} → {out_s}");
    push(vm, Value::from_string(out_s));
    Ok(vm)
}
