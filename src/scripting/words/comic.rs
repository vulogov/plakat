//! 6.8.0 (P4) — `plakat.comic.*` scripting words. Build a multi-panel comic page from a Bund script and
//! push the finished **page** as an image handle, so a script can produce comic pages alongside generated
//! images and hand the page to `plakat.save` / `plakat.upscale`.
//!
//!   `plakat.comic.render ( spec-path out-path             -- handle )`  full render (scene art + balloons)
//!   `plakat.comic.letter ( spec-path panels-dir out-path  -- handle )`  weight-free: composite supplied panels + letter
//!   `plakat.comic.layout ( spec-path out-path             -- handle )`  weight-free: placeholder page (grid only)
//!
//! The page PNG (and its `panels.json` sidecar) is written to `out-path`; the pushed handle is the page.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::Context;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

fn block_on<F: std::future::Future>(f: F, tag: &str) -> anyhow::Result<F::Output> {
    let rt = tokio::runtime::Handle::try_current().map_err(|e| anyhow::anyhow!("{tag}: no tokio runtime: {e}"))?;
    Ok(tokio::task::block_in_place(|| rt.block_on(f)))
}

/// Load a saved page PNG and push it as an image handle.
fn push_page(path: &std::path::Path, tag: &str) -> anyhow::Result<i64> {
    let img = image::open(path).with_context(|| format!("{tag}: reading {}", path.display()))?.to_rgb8();
    with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))
}

/// `plakat.comic.render ( spec-path out-path -- handle )` — the full flagship (needs a model).
pub fn plakat_comic_render(vm: &mut VM) -> BundResult<'_> {
    do_render(vm).map_err(to_bund_err)
}

fn do_render(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.comic.render";
    require_depth(vm, 2, TAG)?;
    let out_path = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let spec_path = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;
    let spec = crate::comic::ComicSpec::load(std::path::Path::new(&spec_path)).with_context(|| format!("{TAG}: loading {spec_path}"))?;
    let out = std::path::PathBuf::from(&out_path);
    let opts = crate::comic::render::RenderOpts { letter: true, ..Default::default() };
    block_on(crate::comic::render::render_spec(&spec, &out, &opts), TAG)?.with_context(|| format!("{TAG}: rendering {spec_path}"))?;
    let handle = push_page(&out, TAG)?;
    tracing::info!(target: "plakat", "{TAG}: {spec_path} → {out_path} → page handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.comic.letter ( spec-path panels-dir out-path -- handle )` — weight-free compose + letter.
pub fn plakat_comic_letter(vm: &mut VM) -> BundResult<'_> {
    do_letter(vm).map_err(to_bund_err)
}

fn do_letter(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.comic.letter";
    require_depth(vm, 3, TAG)?;
    let out_path = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let panels_dir = value_to_string(pull(vm, TAG)?, "panels-dir", TAG)?;
    let spec_path = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;
    let spec = crate::comic::ComicSpec::load(std::path::Path::new(&spec_path)).with_context(|| format!("{TAG}: loading {spec_path}"))?;
    let plan = crate::comic::layout::resolve(&spec);
    let imgs = load_panels(&panels_dir, spec.panels.len());
    let mut page = crate::comic::page::compose(&plan, &imgs);
    crate::comic::page::letter(&mut page, &plan, &spec);
    let out = std::path::PathBuf::from(&out_path);
    page.save(&out).with_context(|| format!("{TAG}: writing {out_path}"))?;
    std::fs::write(out.with_extension("panels.json"), crate::comic::page::panels_json(&plan)).ok();
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(page)))?;
    tracing::info!(target: "plakat", "{TAG}: {spec_path} + {panels_dir} → {out_path} → page handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// `plakat.comic.layout ( spec-path out-path -- handle )` — weight-free placeholder page (grid only).
pub fn plakat_comic_layout(vm: &mut VM) -> BundResult<'_> {
    do_layout(vm).map_err(to_bund_err)
}

fn do_layout(vm: &mut VM) -> anyhow::Result<&mut VM> {
    const TAG: &str = "plakat.comic.layout";
    require_depth(vm, 2, TAG)?;
    let out_path = value_to_string(pull(vm, TAG)?, "out-path", TAG)?;
    let spec_path = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;
    let spec = crate::comic::ComicSpec::load(std::path::Path::new(&spec_path)).with_context(|| format!("{TAG}: loading {spec_path}"))?;
    let plan = crate::comic::layout::resolve(&spec);
    let empty: Vec<Option<image::DynamicImage>> = vec![None; spec.panels.len().max(1)];
    let page = crate::comic::page::compose(&plan, &empty);
    let out = std::path::PathBuf::from(&out_path);
    page.save(&out).with_context(|| format!("{TAG}: writing {out_path}"))?;
    std::fs::write(out.with_extension("panels.json"), crate::comic::page::panels_json(&plan)).ok();
    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(page)))?;
    tracing::info!(target: "plakat", "{TAG}: {spec_path} → {out_path} → page handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// Load panel images from a directory (sorted by name → panel order), sized to the spec's panel count.
fn load_panels(dir: &str, n: usize) -> Vec<Option<image::DynamicImage>> {
    let mut imgs: Vec<Option<image::DynamicImage>> = vec![None; n.max(1)];
    let Ok(rd) = std::fs::read_dir(dir) else { return imgs };
    let mut files: Vec<std::path::PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(), Some("png" | "jpg" | "jpeg" | "webp")))
        .collect();
    files.sort();
    for (i, f) in files.iter().enumerate() {
        if i >= imgs.len() {
            break;
        }
        imgs[i] = image::open(f).ok();
    }
    imgs
}
