//! v0.24 phase 4: `plakat.outpaint ( prompt input expand-spec -- handle )`.
//!
//! Extend an image past its borders. Thin wrapper over the
//! inpaint flow (`plakat.inpaint`): the host word builds a
//! replicate-edge canvas + a single-channel mask and dispatches
//! to `script_entry::inpaint_one`.
//!
//! Mirrors the CLI's `plakat outpaint` subcommand. Same
//! snap-multiple rule: Flux models snap padding to multiples of
//! 16; everything else snaps to 8. The input image must already
//! be snap-aligned — bails loud otherwise.
//!
//! ## Stack effect
//!
//! ```text
//! ( prompt input expand-spec -- handle )
//! ```
//!
//! - **prompt** (string): describes the *whole* expanded scene.
//! - **input** (string | int): source image. String is a path;
//!   integer is an image handle (materialised to a tempfile).
//! - **expand-spec** (string): one of these grammars:
//!   - `"expand=384"` — all four sides get 384 px.
//!   - `"left=512,right=512"` — per-side, missing sides default
//!     to 0. Components: `left`, `right`, `top`, `bottom`.
//!   - At least one side must be > 0.
//!
//! ## Family scope
//!
//! Routes through `inpaint_one`, so the family rules match
//! `plakat.inpaint`: SD-family + SD3 work; Flux bails (use
//! `flux-fill-dev` via the CLI for Flux outpaint).
//!
//! ```bund
//! "wide mountain valley, panorama" "photo.jpg" "left=512,right=512"
//!     plakat.outpaint
//!     "expanded.png" plakat.save
//! ```

use rust_dynamic::types;
use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::ctx::{with_ctx, with_ctx_mut};
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};
use crate::scripting::script_entry;

const TAG: &str = "plakat.outpaint";

pub fn plakat_outpaint(vm: &mut VM) -> BundResult<'_> {
    do_plakat_outpaint(vm).map_err(to_bund_err)
}

fn do_plakat_outpaint(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 3, TAG)?;
    // Top pops first: expand-spec, then input, then prompt.
    let spec_v = pull(vm, TAG)?;
    let input_v = pull(vm, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;
    let spec_str = value_to_string(spec_v, "expand-spec", TAG)?;
    let (left, right, top, bottom) = parse_expand_spec(&spec_str)?;

    // Determine snap-multiple from the loaded model. Flux uses
    // 16; SD-family + SD3 use 8.
    let alias = with_ctx(|ctx| ctx.loaded_model().map(|s| s.to_string()))?;
    let alias = alias.ok_or_else(|| {
        anyhow::anyhow!(
            "{TAG}: no model loaded. Call \"sd15\" plakat.load (or another \
             alias) before {TAG}."
        )
    })?;
    let snap = if alias.to_lowercase().contains("flux") {
        16
    } else {
        8
    };
    let snap_up = |n: u32| -> u32 {
        if n == 0 {
            0
        } else {
            n.div_ceil(snap) * snap
        }
    };
    let left = snap_up(left);
    let right = snap_up(right);
    let top = snap_up(top);
    let bottom = snap_up(bottom);

    // Resolve input: string path or integer handle. Handle goes
    // to a tempfile that lives as long as this stack frame.
    let (_input_tmp_guard, input_path): (
        Option<tempfile::NamedTempFile>,
        PathBuf,
    ) = match input_v.dt {
        types::STRING => {
            let s = value_to_string(input_v, "input", TAG)?;
            (None, PathBuf::from(s))
        }
        types::INTEGER => {
            let handle = input_v.cast_int().unwrap_or(0);
            let img = with_ctx_mut(|ctx| ctx.image_at(handle).cloned())??;
            let tmp = tempfile::Builder::new()
                .prefix("plakat-script-outpaint-handle-")
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

    // Load the input + check snap alignment.
    let input_img = image::open(&input_path)
        .map_err(|e| anyhow::anyhow!("{TAG}: opening input {}: {e}", input_path.display()))?;
    let input_rgb = input_img.to_rgb8();
    let (in_w, in_h) = image::GenericImageView::dimensions(&input_img);
    if in_w % snap != 0 || in_h % snap != 0 {
        anyhow::bail!(
            "{TAG}: input image is {in_w}x{in_h}, not divisible by {snap} \
             (the model's VAE / patch constraint). Resize the input to a \
             multiple of {snap} before outpainting."
        );
    }

    let new_w = in_w + left + right;
    let new_h = in_h + top + bottom;

    // Build canvas + mask using the CLI's existing helpers. Both
    // are pub(crate) in cli::outpaint.
    let canvas = crate::cli::outpaint::build_replicate_canvas(
        &input_rgb, left, top, new_w, new_h,
    );
    let mask = crate::cli::outpaint::build_outpaint_mask(
        in_w, in_h, left, top, new_w, new_h,
    );

    // Persist canvas + mask to a tempdir bound to this stack
    // frame. inpaint_one reads them by path.
    let work_tmp = tempfile::Builder::new()
        .prefix("plakat-script-outpaint-")
        .tempdir()
        .map_err(|e| anyhow::anyhow!("{TAG}: creating outpaint tempdir: {e}"))?;
    let canvas_path = work_tmp.path().join("canvas.png");
    let mask_path = work_tmp.path().join("mask.png");
    canvas
        .save(&canvas_path)
        .map_err(|e| anyhow::anyhow!("{TAG}: writing canvas: {e}"))?;
    mask.save(&mask_path)
        .map_err(|e| anyhow::anyhow!("{TAG}: writing mask: {e}"))?;

    // Outpaint = inpaint with the replicated canvas + mask.
    // inpaint_one honours the per-call `mask_feather` /
    // `mask_invert` knobs from config; outpaint typically wants
    // a small feather (~8 px default), which is already the
    // CLI's outpaint default and matches `cli::outpaint::run`.
    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        // Force `size_explicit` to the new canvas dims so
        // build_t2i_gen_request (or img2img request) doesn't
        // snap to default. Save the prior values to restore.
        let prev_size_explicit = ctx.config.size_explicit;
        let prev_w = ctx.config.width;
        let prev_h = ctx.config.height;
        ctx.config.size_explicit = true;
        ctx.config.width = new_w;
        ctx.config.height = new_h;

        let result = script_entry::inpaint_one(ctx, &prompt, &canvas_path, &mask_path);

        // Restore prior size config — outpaint's canvas dims are
        // a per-call concern, not a persistent ctx mutation.
        ctx.config.size_explicit = prev_size_explicit;
        ctx.config.width = prev_w;
        ctx.config.height = prev_h;

        let img = result?;
        Ok(ctx.push_image(img))
    })??;

    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int} ({in_w}x{in_h} -> {new_w}x{new_h}; \
         padding L{left} R{right} T{top} B{bottom})"
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}

/// Parse the expand-spec grammar. Supported forms:
/// - `"expand=N"` — all four sides.
/// - `"left=L,right=R,top=T,bottom=B"` — per-side; missing sides
///   default to 0.
/// Returns `(left, right, top, bottom)`. Bails if all zero (the
/// CLI bails the same way) or if any component fails to parse.
fn parse_expand_spec(spec: &str) -> anyhow::Result<(u32, u32, u32, u32)> {
    let spec = spec.trim();
    if spec.is_empty() {
        anyhow::bail!(
            "{TAG}: expand-spec can't be empty. Examples: \"expand=384\" \
             or \"left=512,right=512\"."
        );
    }
    let mut expand_val: Option<u32> = None;
    let mut left: Option<u32> = None;
    let mut right: Option<u32> = None;
    let mut top: Option<u32> = None;
    let mut bottom: Option<u32> = None;
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (key, val) = part.split_once('=').ok_or_else(|| {
            anyhow::anyhow!(
                "{TAG}: expand-spec part {part:?} must be `key=value` \
                 (e.g. `left=512`)"
            )
        })?;
        let val: u32 = val.trim().parse().map_err(|e| {
            anyhow::anyhow!("{TAG}: expand-spec {part:?}: {e}")
        })?;
        match key.trim().to_lowercase().as_str() {
            "expand" => expand_val = Some(val),
            "left" => left = Some(val),
            "right" => right = Some(val),
            "top" => top = Some(val),
            "bottom" => bottom = Some(val),
            other => anyhow::bail!(
                "{TAG}: unknown expand-spec key {other:?} (accepted: expand, \
                 left, right, top, bottom)"
            ),
        }
    }
    // Mutual exclusion: `expand=N` can't combine with any per-side key.
    let has_per_side =
        left.is_some() || right.is_some() || top.is_some() || bottom.is_some();
    if expand_val.is_some() && has_per_side {
        anyhow::bail!(
            "{TAG}: `expand=N` is mutually exclusive with the per-side keys."
        );
    }
    let (l, r, t, b) = if let Some(e) = expand_val {
        (e, e, e, e)
    } else {
        (
            left.unwrap_or(0),
            right.unwrap_or(0),
            top.unwrap_or(0),
            bottom.unwrap_or(0),
        )
    };
    if l == 0 && r == 0 && t == 0 && b == 0 {
        anyhow::bail!(
            "{TAG}: needs at least one of left/right/top/bottom (or expand) > 0."
        );
    }
    Ok((l, r, t, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_expand_all_four_sides() {
        assert_eq!(parse_expand_spec("expand=384").unwrap(), (384, 384, 384, 384));
    }

    #[test]
    fn parse_expand_per_side() {
        assert_eq!(
            parse_expand_spec("left=512,right=256").unwrap(),
            (512, 256, 0, 0)
        );
        assert_eq!(
            parse_expand_spec("top=128,bottom=128,left=64,right=64").unwrap(),
            (64, 64, 128, 128)
        );
    }

    #[test]
    fn parse_expand_rejects_empty() {
        assert!(parse_expand_spec("").is_err());
        assert!(parse_expand_spec("   ").is_err());
    }

    #[test]
    fn parse_expand_rejects_all_zero() {
        let err = parse_expand_spec("left=0,right=0").unwrap_err();
        assert!(format!("{err}").contains("> 0"));
    }

    #[test]
    fn parse_expand_rejects_unknown_key() {
        let err = parse_expand_spec("middle=128").unwrap_err();
        assert!(format!("{err}").contains("unknown expand-spec key"));
    }

    #[test]
    fn parse_expand_rejects_expand_with_per_side() {
        let err = parse_expand_spec("expand=128,left=64").unwrap_err();
        assert!(format!("{err}").contains("mutually exclusive"));
    }
}
