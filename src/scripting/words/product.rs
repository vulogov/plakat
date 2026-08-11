//! 6.9.0 (P4) — `plakat.product.*` scripting words. Render a studio packshot / catalog sheet from a Bund
//! script and push the finished image as a handle, so a script can produce product-shots alongside
//! generated images and hand them to `plakat.save` / `plakat.upscale`.
//!
//!   `plakat.product.render ( spec-path out-path -- handle )`   subject → sweep + grounding → a packshot
//!   `plakat.product.sheet  ( spec-path out-path -- handle )`   main + variants → a labelled contact sheet

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

fn block_on<F: std::future::Future>(f: F, tag: &str) -> anyhow::Result<F::Output> {
    let rt = tokio::runtime::Handle::try_current().map_err(|e| anyhow::anyhow!("{tag}: no tokio runtime: {e}"))?;
    Ok(tokio::task::block_in_place(|| rt.block_on(f)))
}

fn push_image(path: &std::path::Path, tag: &str) -> anyhow::Result<i64> {
    let img = image::open(path).with_context(|| format!("{tag}: reading {}", path.display()))?.to_rgb8();
    with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))
}

/// `plakat.product.render ( spec-path out-path -- handle )`.
pub fn plakat_product_render(vm: &mut VM) -> BundResult<'_> {
    do_render(vm).map_err(to_bund_err)
}

fn do_render(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.product.render";
    require_depth(vm, 2, TAG)?;
    let out_path = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let spec_path = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;
    let spec = crate::product::ProductSpec::load(std::path::Path::new(&spec_path)).with_context(|| format!("{TAG}: loading {spec_path}"))?;
    let out = std::path::PathBuf::from(&out_path);
    let opts = crate::product::render::RenderOpts { relight: spec.lighting.is_some(), ..Default::default() };
    block_on(crate::product::render::render_spec(&spec, &out, &opts), TAG)?.with_context(|| format!("{TAG}: rendering {spec_path}"))?;
    let handle = push_image(&out, TAG)?;
    tracing::info!(target: "plakat", "{TAG}: {spec_path} → {out_path} → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.product.sheet ( spec-path out-path -- handle )`.
pub fn plakat_product_sheet(vm: &mut VM) -> BundResult<'_> {
    do_sheet(vm).map_err(to_bund_err)
}

fn do_sheet(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.product.sheet";
    require_depth(vm, 2, TAG)?;
    let out_path = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let spec_path = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;
    let spec = crate::product::ProductSpec::load(std::path::Path::new(&spec_path)).with_context(|| format!("{TAG}: loading {spec_path}"))?;
    let out = std::path::PathBuf::from(&out_path);
    let opts = crate::product::render::RenderOpts { relight: spec.lighting.is_some(), ..Default::default() };
    block_on(crate::product::render::render_sheet(&spec, &opts, &out), TAG)?.with_context(|| format!("{TAG}: sheeting {spec_path}"))?;
    let handle = push_image(&out, TAG)?;
    tracing::info!(target: "plakat", "{TAG}: {spec_path} → {out_path} → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}
