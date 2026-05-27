//! v0.23 phase 5: `plakat.inpaint ( prompt input mask -- handle )`.
//!
//! Mask-guided img2img. `input` is either a filesystem path or
//! an image handle (same grammar as `plakat.img2img`). `mask` is
//! a filesystem path to a grayscale / RGB / RGBA image — white
//! pixels mark inpaint regions, black pixels are preserved.
//! `config.mask_feather` (px) softens the edge;
//! `config.mask_invert` flips the polarity if the mask source
//! uses the opposite convention. Both knobs were declared in
//! v0.22 phase 11; v0.23 phase 5 ships the mask path arg that
//! actually fires them.
//!
//! ```bund
//! "stained glass window in the wall"  "./photo.png"  "./mask.png"
//!     plakat.inpaint
//!     "result.png" plakat.save
//! ```
//!
//! Stack effect: `( prompt input mask -- handle )`. Top pops
//! first: `mask`, then `input`, then `prompt`.
//!
//! SD-family only in v0.23 phase 5 — Flux inpaint requires the
//! `flux-fill-dev` variant with load-time channel-concat wiring
//! on `img_in` (not in scope here; workaround: use the CLI's
//! `plakat img2img --model flux-fill-dev --mask MASK`). SD3
//! supports native RePaint-style inpaint, which IS wired (mask
//! threads through `sd3::GenRequest.mask`).

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};
use crate::scripting::script_entry;

const TAG: &str = "plakat.inpaint";

pub fn plakat_inpaint(vm: &mut VM) -> BundResult<'_> {
    do_plakat_inpaint(vm).map_err(to_bund_err)
}

fn do_plakat_inpaint(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 3, TAG)?;
    // Top pops first: mask, then input, then prompt.
    let mask_v = pull(vm, TAG)?;
    let input_v = pull(vm, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;
    let mask_str = value_to_string(mask_v, "mask", TAG)?;
    if mask_str.is_empty() {
        anyhow::bail!("{TAG}: mask path can't be empty");
    }
    let mask_path = std::path::PathBuf::from(mask_str);

    // Materialise integer handles to a tempfile (same pattern as
    // plakat.img2img). Mask is path-only in v0.23 phase 5 — no
    // handle-form is supported for masks (RGB-or-grayscale source
    // semantics make handle-forms ambiguous; revisit if real
    // scripts ask for it).
    let (_tempfile_guard, input_path): (
        Option<tempfile::NamedTempFile>,
        std::path::PathBuf,
    ) = match input_v.dt {
        types::STRING => {
            let s = value_to_string(input_v, "input", TAG)?;
            (None, std::path::PathBuf::from(s))
        }
        types::INTEGER => {
            let handle = input_v.cast_int().unwrap_or(0);
            let img = with_ctx_mut(|ctx| ctx.image_at(handle).cloned())??;
            let tmp = tempfile::Builder::new()
                .prefix("plakat-script-inpaint-handle-")
                .suffix(".png")
                .tempfile()
                .map_err(|e| {
                    anyhow::anyhow!("{TAG}: creating tempfile for handle {handle}: {e}")
                })?;
            img.save(tmp.path()).map_err(|e| {
                anyhow::anyhow!("{TAG}: writing handle {handle} to tempfile: {e}")
            })?;
            let path = tmp.path().to_path_buf();
            (Some(tmp), path)
        }
        _ => {
            anyhow::bail!(
                "{TAG}: input must be a string path or an integer handle \
                 (got rust_dynamic dt = {})",
                input_v.dt
            );
        }
    };

    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        let img = script_entry::inpaint_one(ctx, &prompt, &input_path, &mask_path)?;
        Ok(ctx.push_image(img))
    })??;
    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} from {} (mask {})",
        input_path.display(),
        mask_path.display()
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
