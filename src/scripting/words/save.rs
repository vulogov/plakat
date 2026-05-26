//! v0.21 phase 2: `plakat.save ( handle path -- )`.
//!
//! Writes the image at `handle` to `path`. The path is resolved
//! relative to `ScriptCtx.out_dir` if it's not absolute, so scripts
//! can use short names (`"fox.png"`) without worrying about the
//! caller's CWD. The handle is **not** consumed — the image stays
//! in the registry so a script can save the same image to multiple
//! paths or chain it through later words (`plakat.upscale` in
//! phase 6).

use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::ctx::with_ctx;
use crate::scripting::helpers::{
    BundResult, pull, require_depth, to_bund_err, value_to_int, value_to_string,
};

const TAG: &str = "plakat.save";

pub fn plakat_save(vm: &mut VM) -> BundResult<'_> {
    do_plakat_save(vm).map_err(to_bund_err)
}

fn do_plakat_save(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Stack order: bottom = handle (pushed first by generate),
    // top = path (pushed second by the script). Top pops first.
    let path_v = pull(vm, TAG)?;
    let handle_v = pull(vm, TAG)?;
    let path_str = value_to_string(path_v, "path", TAG)?;
    let handle = value_to_int(handle_v, "handle", TAG)?;

    // Resolve relative paths against ctx.out_dir; absolute paths
    // pass through. Reading the lock once here means we don't
    // hold it across the image write.
    let (out_dir, image_clone) = with_ctx(|ctx| {
        let img = ctx.image_at(handle)?;
        Ok::<_, anyhow::Error>((ctx.out_dir.clone(), img.clone()))
    })??;
    let path: PathBuf = {
        let p = PathBuf::from(&path_str);
        if p.is_absolute() { p } else { out_dir.join(p) }
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!(
                "{TAG}: creating parent dir {}: {e}",
                parent.display()
            )
        })?;
    }
    image_clone.save(&path).map_err(|e| {
        anyhow::anyhow!("{TAG}: writing {}: {e}", path.display())
    })?;
    tracing::info!(
        target: "plakat",
        "{TAG}: handle {handle} → {}",
        path.display()
    );
    Ok(vm)
}
