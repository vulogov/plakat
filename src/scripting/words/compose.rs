//! `plakat.compose ( scene-path -- )` — run a layered-composition scene file.
//!
//! Composite a scene described by an HJSON file (the CLI `plakat compose`): `load`/`matte`/
//! `generate` layers stacked onto a canvas. The output is written to the `out:` path declared
//! inside the scene file (this word is terminal — it consumes the path and produces the file on
//! disk, not a handle, because the scene decides its own output location).
//!
//! ```bund
//! "poster.hjson" plakat.compose
//! ```

use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::ctx::with_ctx;
use crate::scripting::helpers::{
    BundResult, pull, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.compose";

pub fn plakat_compose(vm: &mut VM) -> BundResult<'_> {
    do_plakat_compose(vm).map_err(to_bund_err)
}

fn do_plakat_compose(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let scene_v = pull(vm, TAG)?;
    let scene = value_to_string(scene_v, "scene", TAG)?;
    if scene.is_empty() {
        anyhow::bail!("{TAG}: scene path can't be empty");
    }

    let device = with_ctx(|ctx| ctx.device.clone())?;

    let handle = tokio::runtime::Handle::try_current()
        .map_err(|e| anyhow::anyhow!("{TAG}: no tokio runtime in scope. {e}"))?;
    tokio::task::block_in_place(|| {
        handle.block_on(crate::cli::compose::run(
            crate::cli::compose::ComposeArgs { scene: PathBuf::from(&scene) },
            device,
        ))
    })?;

    tracing::info!(target: "plakat", "{TAG}: composed scene {scene}");
    Ok(vm)
}
