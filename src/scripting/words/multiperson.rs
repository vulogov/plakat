//! 1.14.0-A — `plakat.multiperson` scripting word. Place specific people into a
//! generated scene from a Bund script, dispatching the SAME
//! `pipelines::multiperson::run` the CLI + scenario surfaces use (byte-for-byte
//! parity, like the `plakat.map.*` words).
//!
//!   `plakat.multiperson ( spec-path -- handle )`
//!
//! The spec file (JSON / HJSON) is a self-contained [`MultipersonScriptSpec`]:
//! the scene prompt + placed `people` + an inline `personas` table mapping each
//! name to a reference photo. The composed scene is read back as an image handle
//! (then `plakat.save` writes it).

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use anyhow::{Context, anyhow};

use crate::pipelines::multiperson::scenario_task::{MultipersonScriptSpec, build_request};
use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{BundResult, pull, push, require_depth, to_bund_err, value_to_string};

const TAG: &str = "plakat.multiperson";

/// `plakat.multiperson ( spec-path -- handle )` — compose a people-in-scene image.
pub fn plakat_multiperson(vm: &mut VM) -> BundResult<'_> {
    do_plakat_multiperson(vm).map_err(to_bund_err)
}

fn do_plakat_multiperson(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let path_s = value_to_string(pull(vm, TAG)?, "spec-path", TAG)?;

    let text = std::fs::read_to_string(&path_s)
        .with_context(|| format!("{TAG}: reading multiperson spec {path_s}"))?;
    let spec: MultipersonScriptSpec = deser(&text)
        .with_context(|| format!("{TAG}: parsing multiperson spec {path_s}"))?;

    let (device, seed, model) = with_ctx(|ctx| {
        (
            ctx.device.clone(),
            ctx.config.seed,
            ctx.loaded_model().unwrap_or("sdxl").to_string(),
        )
    })?;

    // A composed scene lands in a temp dir; we read the first output back as a
    // handle and clean up, mirroring `plakat.map.paint`.
    let out_dir = std::env::temp_dir().join(format!("plakat-multiperson-{}", seed.unwrap_or(42)));
    let _ = std::fs::remove_dir_all(&out_dir);

    let mut task = spec.task.clone();
    if task.seed.is_none() {
        task.seed = seed;
    }
    let req = build_request(
        &task,
        |name| spec.resolve(name),
        out_dir.clone(),
        device,
        &model,
        false,
    )?;

    let rt = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow!("{TAG}: no tokio runtime: {e}"))?;
    tokio::task::block_in_place(|| rt.block_on(crate::pipelines::multiperson::run(req)))
        .with_context(|| format!("{TAG}: composing {path_s}"))?;

    let img_path = first_image(&out_dir)
        .ok_or_else(|| anyhow!("{TAG}: run produced no image in {}", out_dir.display()))?;
    let img = image::open(&img_path)
        .with_context(|| format!("{TAG}: reading {}", img_path.display()))?
        .to_rgb8();
    let _ = std::fs::remove_dir_all(&out_dir);

    let handle = with_ctx_mut(|ctx| ctx.push_image(image::DynamicImage::ImageRgb8(img)))?;
    tracing::info!(target: "plakat", "{TAG}: composed {path_s} → handle {handle}");
    push(vm, Value::from_int(handle));
    Ok(vm)
}

/// Parse the spec as HJSON when the `compile`/HJSON path is built, else JSON.
/// JSON is a subset of HJSON, so plain JSON specs always parse.
fn deser(text: &str) -> anyhow::Result<MultipersonScriptSpec> {
    serde_json::from_str(text).map_err(|e| anyhow!("{e}"))
}

/// First PNG in `dir` (shallow), lexically lowest so it's deterministic.
fn first_image(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut pngs: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("png"))
        .collect();
    pngs.sort();
    pngs.into_iter().next()
}
