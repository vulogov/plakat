//! v0.36 phase 1: `plakat.pixart ( prompt -- handle )`.
//!
//! Renders one image with the PixArt-Σ pipeline cached on
//! `ScriptCtx`, stores it in `ScriptCtx.images`, and pushes the
//! 1-based integer handle onto the stack. Mirrors `plakat.generate`'s
//! single-image shape — width / height / steps / guidance / seed /
//! scheduler / negative all come from `ctx.config`; LoRAs come from
//! `ctx.loras` (merged at PixArt load time per v0.35 phase 4).
//!
//! ## Stack effect
//!
//! `( prompt -- handle )`. Same as `plakat.generate`.
//!
//! ## Usage
//!
//! ```bund
//! "pixart" plakat.load
//! "1024" "width" plakat.config.set
//! "1024" "height" plakat.config.set
//! "20" "steps" plakat.config.set
//! "4.5" "guidance" plakat.config.set
//! "a misty forest at dawn, painterly" plakat.pixart
//! // → integer image handle on the stack
//! ```
//!
//! ## Configurable knobs
//!
//! Standard config keys honoured: `width`, `height`, `steps`,
//! `guidance`, `seed`, `negative`, `scheduler`. PixArt-Σ
//! conditioning (resolution + aspect_ratio) computed from
//! width/height inside `pixart::Pipeline::generate`.
//!
//! ## Scope
//!
//! PixArt-Σ only (the loaded alias must be one of `pixart` /
//! `pixart-sigma` / `pixart-1024` / the canonical
//! `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` repo path).
//! Non-PixArt aliases bail loud with a pointer at `plakat.generate`.
//!
//! The pipeline is cached on `ScriptCtx.loaded_pixart`; multi-call
//! scripts amortise the ~12 GB cold load. Alias change drops; LoRA
//! stack mutation drops via [`ScriptCtx::mark_loras_changed`].

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.pixart";

pub fn plakat_pixart(vm: &mut VM) -> BundResult<'_> {
    do_plakat_pixart(vm).map_err(to_bund_err)
}

fn do_plakat_pixart(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;
    if prompt.is_empty() {
        anyhow::bail!("{TAG}: prompt can't be empty");
    }

    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        // Variant check: the loaded model must be PixArt. Mirrors
        // the SD-family / Flux / SD3 family detection done inside
        // plakat.animate, but for PixArt only.
        let alias = ctx.loaded_model().ok_or_else(|| {
            anyhow::anyhow!(
                "{TAG}: no model loaded. Call \"pixart\" plakat.load before \
                 {TAG}."
            )
        })?;
        let resolved_for_detect = if alias.contains('/') {
            alias.to_string()
        } else {
            crate::hf::resolve_alias(alias).to_string()
        };
        let variant = crate::pipelines::t2i::Variant::detect(&resolved_for_detect);
        if !variant.is_pixart() {
            anyhow::bail!(
                "{TAG}: loaded model {alias:?} resolves to {variant:?}, not PixArt. \
                 Use `plakat.generate` for SD-family / SD3 / Flux models, or \
                 call `\"pixart\" plakat.load` first."
            );
        }

        // Snapshot config-driven sampler knobs (mirrors the v0.21
        // pattern other words use). Width/height divisibility check
        // matches pixart::Pipeline::generate's own bail.
        let prompt_owned = prompt.clone();
        let width = ctx.config.width.max(8);
        let height = ctx.config.height.max(8);
        anyhow::ensure!(
            width.is_multiple_of(8) && height.is_multiple_of(8),
            "{TAG}: width/height must be divisible by 8 (got {width}x{height})"
        );
        let steps = ctx.config.steps;
        let guidance = ctx.config.guidance;
        let scheduler = ctx.config.scheduler;
        let negative = ctx.config.negative.clone();
        // Seed: explicit when given, otherwise random per call.
        let seed = ctx
            .config
            .seed
            .unwrap_or_else(rand::random::<u64>);

        // get_or_load + generate. The borrow of `pipeline` is
        // released before push_image_with_metadata mutates ctx.
        let (buf, ow, oh) = {
            let pipeline = ctx.get_or_load_pixart()?;
            pipeline.generate(
                &prompt_owned,
                &negative,
                width,
                height,
                steps,
                guidance,
                seed,
                scheduler,
            )?
        };

        // Convert (buf, w, h) → DynamicImage so it fits ctx.images.
        let rgb = image::RgbImage::from_raw(ow, oh, buf).ok_or_else(|| {
            anyhow::anyhow!(
                "{TAG}: VAE decode produced unexpected byte length"
            )
        })?;
        let img = image::DynamicImage::ImageRgb8(rgb);

        // Build sidecar metadata matching pixart::run (v0.35 phase 4
        // shape). LoRA stack populated from ctx.loras via the v0.34
        // phase 0 LoraSpec::to_entry path.
        let model = ctx
            .loaded_model()
            .unwrap_or("pixart")
            .to_string();
        let mut meta = crate::imaging::metadata::GenerationMetadata::new(
            prompt_owned.clone(),
            model,
            seed,
            steps,
            guidance,
            format!("{scheduler:?}").to_lowercase(),
            width,
            height,
        );
        meta.negative = negative;
        if !ctx.loras.is_empty() {
            let stack: Vec<crate::imaging::metadata::LoraEntry> = ctx
                .loras
                .iter()
                .map(|s| s.to_entry())
                .collect();
            meta.with_lora_stack(stack);
            meta.lora_scale = Some(ctx.config.lora_scale);
        }
        Ok(ctx.push_image_with_metadata(img, meta))
    })??;

    tracing::info!(
        target: "plakat",
        "{TAG}: rendered handle {handle_int}"
    );
    push(vm, Value::from_int(handle_int));
    Ok(vm)
}
