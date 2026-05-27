//! v0.21 phase 2: `plakat.save ( handle path -- )`.
//!
//! Writes the image at `handle` to `path`. The path is resolved
//! relative to `ScriptCtx.out_dir` if it's not absolute, so scripts
//! can use short names (`"fox.png"`) without worrying about the
//! caller's CWD. The handle is **not** consumed — the image stays
//! in the registry so a script can save the same image to multiple
//! paths or chain it through later words (`plakat.upscale` in
//! phase 6).
//!
//! ## v0.26 phase 8: metadata writes
//!
//! When the image's handle has [`GenerationMetadata`] attached
//! (the rendering paths populate this), `plakat.save` routes
//! through [`crate::imaging::io::save_rgb_u8_with_metadata`]
//! which writes:
//!   - The PNG with an A1111-compatible `parameters` tEXt chunk
//!   - A `<name>.json` sidecar with the structured metadata
//!
//! When no metadata is attached (e.g. images loaded from disk
//! via TBD future words, or rendering paths that don't yet
//! populate it), `plakat.save` falls back to the plain
//! [`DynamicImage::save`] path — byte-identical to the v0.21
//! behaviour.

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
    // pass through. Read everything once so we don't hold the
    // lock across the image write.
    let (out_dir, image_clone, metadata_clone) = with_ctx(|ctx| {
        let img = ctx.image_at(handle)?;
        let meta = ctx.metadata_at(handle)?.cloned();
        Ok::<_, anyhow::Error>((ctx.out_dir.clone(), img.clone(), meta))
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

    if let Some(meta) = metadata_clone {
        // v0.26 phase 8: metadata-aware path. Writes the PNG
        // with the A1111 `parameters` tEXt chunk PLUS the JSON
        // sidecar. For non-PNG extensions, the tEXt chunk is
        // skipped (per save_rgb_u8_with_metadata's extension-
        // routing) but the sidecar still lands.
        let rgb = image_clone.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        crate::imaging::io::save_rgb_u8_with_metadata(
            rgb.as_raw(),
            w,
            h,
            &path,
            &meta,
        )
        .map_err(|e| {
            anyhow::anyhow!("{TAG}: writing {} with metadata: {e}", path.display())
        })?;
        tracing::info!(
            target: "plakat",
            "{TAG}: handle {handle} → {} (with metadata sidecar)",
            path.display()
        );
    } else {
        image_clone.save(&path).map_err(|e| {
            anyhow::anyhow!("{TAG}: writing {}: {e}", path.display())
        })?;
        tracing::info!(
            target: "plakat",
            "{TAG}: handle {handle} → {}",
            path.display()
        );
    }
    Ok(vm)
}
