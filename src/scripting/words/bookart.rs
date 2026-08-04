//! 6.1.0 (A4) — `plakat.bookart.*` scripting words. Render a book ornament from a Bund
//! script into an image handle, so a script can produce transparent, page-sized B/W
//! ornaments alongside generated images and hand them to the existing `plakat.save` /
//! `plakat.metadata.write` / `plakat.upscale` words.
//!
//!   `plakat.bookart.origin     ( origin    -- )`          origin override for `.illustrate`
//!   `plakat.bookart.technique  ( technique -- )`          technique override for `.illustrate`
//!   `plakat.bookart.render     ( spec-path -- handle )`   render a BookArtSpec file (any tier)
//!   `plakat.bookart.illustrate ( prompt    -- handle )`   prose → a diffusion B/W plate (GPU)
//!
//! The pushed handle is the **transparent, exactly-page-sized** RGBA ornament — it flows
//! straight into `plakat.save` (keeps the alpha). `model` / `seed` / `steps` come from the
//! shared `plakat.config.*` state (the diffusion/composite tiers are GPU; `procedural` specs
//! render with no model). SVG is a file-only artefact (no image handle) — use the CLI
//! `bookart render --svg` for that.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::bookart::render::{RenderOpts, render_spec};
use crate::bookart::spec::{BookArtSpec, Ornament, Page};
use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

/// Read the shared `plakat.config.*` render knobs (model / seed / steps) into a `RenderOpts`.
fn opts_from_config() -> anyhow::Result<RenderOpts> {
    with_ctx(|ctx| RenderOpts {
        model: ctx
            .loaded_model()
            .map(str::to_string)
            .unwrap_or_else(|| "sd15".to_string()),
        seed: ctx.config.seed.unwrap_or(0),
        steps: ctx.config.steps,
        svg: false,
        attempts: 1,
        font: None,
    })
}

/// Bridge the async render core the same way the map/cascade/fractal words do.
fn render_blocking(spec: &BookArtSpec, opts: &RenderOpts, tag: &str) -> anyhow::Result<image::RgbaImage> {
    let rt = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{tag}: no tokio runtime: {e}"))?;
    let rendered = tokio::task::block_in_place(|| rt.block_on(render_spec(spec, opts)))
        .with_context(|| format!("{tag}: rendering ornament"))?;
    Ok(rendered.page)
}

/// `plakat.bookart.origin ( origin -- )` — set the origin override for `.illustrate`.
/// `none`/`auto`/`""` clears it (back to `generic`). Mirrors `bookart illustrate --origin`.
pub fn plakat_bookart_origin(vm: &mut VM) -> BundResult<'_> {
    do_plakat_bookart_origin(vm).map_err(to_bund_err)
}

fn do_plakat_bookart_origin(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.bookart.origin";
    require_depth(vm, 1, TAG)?;
    let origin = value_to_string(pull(vm, TAG)?, "origin", TAG)?;
    with_ctx_mut(|ctx| {
        ctx.bookart_origin = match origin.to_ascii_lowercase().as_str() {
            "none" | "auto" | "" => None,
            _ => Some(origin.clone()),
        };
    })?;
    tracing::info!(target: "plakat", "{TAG}: {origin}");
    Ok(vm)
}

/// `plakat.bookart.technique ( technique -- )` — set the technique override for `.illustrate`.
/// `none`/`auto`/`""` clears it (back to `line`). Mirrors `bookart illustrate --technique`.
pub fn plakat_bookart_technique(vm: &mut VM) -> BundResult<'_> {
    do_plakat_bookart_technique(vm).map_err(to_bund_err)
}

fn do_plakat_bookart_technique(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.bookart.technique";
    require_depth(vm, 1, TAG)?;
    let tech = value_to_string(pull(vm, TAG)?, "technique", TAG)?;
    with_ctx_mut(|ctx| {
        ctx.bookart_technique = match tech.to_ascii_lowercase().as_str() {
            "none" | "auto" | "" => None,
            _ => Some(tech.clone()),
        };
    })?;
    tracing::info!(target: "plakat", "{TAG}: {tech}");
    Ok(vm)
}

/// `plakat.bookart.render ( spec-path -- handle )` — render a BookArtSpec file (any tier) →
/// a transparent, page-sized ornament image handle. `procedural` specs need no GPU.
pub fn plakat_bookart_render(vm: &mut VM) -> BundResult<'_> {
    do_plakat_bookart_render(vm).map_err(to_bund_err)
}

fn do_plakat_bookart_render(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.bookart.render";
    require_depth(vm, 1, TAG)?;
    let path = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;
    let spec = BookArtSpec::load(std::path::Path::new(&path))
        .with_context(|| format!("{TAG}: loading spec {path}"))?;
    let opts = opts_from_config()?;
    let page = render_blocking(&spec, &opts, TAG)?;
    let (w, h) = page.dimensions();
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgba8(page)))?;
    tracing::info!(target: "plakat", "{TAG}: {path} ({w}x{h}) → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.bookart.illustrate ( prompt -- handle )` — synthesise a diffusion-tier B/W plate
/// from prose (a frontispiece/spot) → a transparent, page-sized ornament image handle. GPU.
/// Honours the `plakat.bookart.origin` / `.technique` overrides (else `generic` / `line`).
pub fn plakat_bookart_illustrate(vm: &mut VM) -> BundResult<'_> {
    do_plakat_bookart_illustrate(vm).map_err(to_bund_err)
}

fn do_plakat_bookart_illustrate(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.bookart.illustrate";
    require_depth(vm, 1, TAG)?;
    let prompt = value_to_string(pull(vm, TAG)?, "prompt", TAG)?;
    let (origin, technique) = with_ctx(|ctx| {
        (
            ctx.bookart_origin.clone().unwrap_or_else(|| "generic".into()),
            ctx.bookart_technique.clone().unwrap_or_else(|| "line".into()),
        )
    })?;
    // Mirror `run_illustrate`: a single diffusion-tier frontispiece plate.
    let spec = BookArtSpec {
        schema: Some(crate::bookart::SCHEMA_VERSION.into()),
        origin: Some(origin.clone()),
        technique: Some(technique.clone()),
        page: Some(Page { size: Some("a5".into()), ..Default::default() }),
        ornament: Some(Ornament {
            kind: Some("frontispiece".into()),
            tier: Some("diffusion".into()),
            prompt: Some(prompt.clone()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let opts = opts_from_config()?;
    let page = render_blocking(&spec, &opts, TAG)?;
    let (w, h) = page.dimensions();
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgba8(page)))?;
    tracing::info!(
        target: "plakat",
        "{TAG}: {prompt:?} ({origin}/{technique}, {w}x{h}) → handle {handle}"
    );
    push(vm, Value::from_int(handle));
    Ok(vm)
}
