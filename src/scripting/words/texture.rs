//! 6.3.0 (B7) — `plakat.texture.*` scripting words. Synthesise a seamless PBR material from a Bund
//! script and push its lit **preview** as an image handle, so a script can produce materials alongside
//! generated images and hand the preview to `plakat.save` / `plakat.upscale`.
//!
//!   `plakat.texture.render  ( spec-path out-dir -- handle )`   render a TextureSpec → material dir, push preview
//!   `plakat.texture.from    ( image out-dir    -- handle )`   image-to-material, push preview
//!   `plakat.texture.preview ( mat-dir          -- handle )`   re-render the lit preview of a material dir
//!
//! The full material set (albedo/normal/roughness/metallic/height/AO + ORM + manifest) is written to
//! the directory; the pushed handle is the preview (a viewable RGB image).

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

fn block_on<F: std::future::Future>(f: F, tag: &str) -> anyhow::Result<F::Output> {
    let rt = tokio::runtime::Handle::try_current().map_err(|e| anyhow::anyhow!("{tag}: no tokio runtime: {e}"))?;
    Ok(tokio::task::block_in_place(|| rt.block_on(f)))
}

/// Load `<dir>/preview.png` (or render it) and push it as an image handle.
fn push_preview(dir: &std::path::Path, tag: &str) -> anyhow::Result<i64> {
    let p = dir.join("preview.png");
    let img = image::open(&p).with_context(|| format!("{tag}: reading {}", p.display()))?.to_rgb8();
    with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))
}

/// `plakat.texture.render ( spec-path out-dir -- handle )`.
pub fn plakat_texture_render(vm: &mut VM) -> BundResult<'_> {
    do_render(vm).map_err(to_bund_err)
}

fn do_render(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.texture.render";
    require_depth(vm, 2, TAG)?;
    let out_dir = value_to_string(pull(vm, TAG)?, "out-dir", TAG)?;
    let spec_path = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;
    let spec = crate::texture::TextureSpec::load(std::path::Path::new(&spec_path)).with_context(|| format!("{TAG}: loading {spec_path}"))?;
    let out = std::path::PathBuf::from(&out_dir);
    block_on(crate::texture::render::render_material(&spec, &out, &Default::default()), TAG)?.with_context(|| format!("{TAG}: rendering {spec_path}"))?;
    let handle = push_preview(&out, TAG)?;
    tracing::info!(target: "plakat", "{TAG}: {spec_path} → {out_dir} → preview handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.texture.from ( image out-dir -- handle )` — image-to-material.
pub fn plakat_texture_from(vm: &mut VM) -> BundResult<'_> {
    do_from(vm).map_err(to_bund_err)
}

fn do_from(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.texture.from";
    require_depth(vm, 2, TAG)?;
    let out_dir = value_to_string(pull(vm, TAG)?, "out-dir", TAG)?;
    let image = value_to_string(pull(vm, TAG)?, "image", TAG)?;
    let spec = crate::texture::TextureSpec { from_image: Some(image.clone()), ..Default::default() };
    let out = std::path::PathBuf::from(&out_dir);
    block_on(crate::texture::render::render_material(&spec, &out, &Default::default()), TAG)?.with_context(|| format!("{TAG}: {image}"))?;
    let handle = push_preview(&out, TAG)?;
    tracing::info!(target: "plakat", "{TAG}: {image} → {out_dir} → preview handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.texture.preview ( mat-dir -- handle )` — re-render the lit preview of a material directory.
pub fn plakat_texture_preview(vm: &mut VM) -> BundResult<'_> {
    do_preview(vm).map_err(to_bund_err)
}

fn do_preview(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.texture.preview";
    require_depth(vm, 1, TAG)?;
    let dir = value_to_string(pull(vm, TAG)?, "mat-dir", TAG)?;
    let d = std::path::Path::new(&dir);
    let albedo = image::open(d.join("albedo.png")).with_context(|| format!("{TAG}: needs albedo.png in {dir}"))?.to_rgb8();
    let (w, h) = albedo.dimensions();
    let gray = |n: &str, def: u8| image::open(d.join(n)).ok().map(|i| i.to_luma8()).unwrap_or_else(|| image::GrayImage::from_pixel(w, h, image::Luma([def])));
    let normal = image::open(d.join("normal.png")).ok().map(|i| i.to_rgb8()).unwrap_or_else(|| image::RgbImage::from_pixel(w, h, image::Rgb([128, 128, 255])));
    let m = crate::texture::Material { albedo, height: gray("height.png", 128), normal, roughness: gray("roughness.png", 153), metallic: gray("metallic.png", 0), ao: gray("ao.png", 255) };
    let img = crate::texture::preview::render(&m, crate::texture::Shape::Sphere, 512);
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))?;
    tracing::info!(target: "plakat", "{TAG}: {dir} → preview handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}
