//! v0.38 phase 2: `plakat.cascade ( prompt -- handle )`.
//!
//! Renders one image with the Stable Cascade 3-stage pipeline
//! cached on `ScriptCtx`, stores it in `ScriptCtx.images`, and
//! pushes the 1-based integer handle onto the stack. Mirrors
//! `plakat.pixart`'s shape but with Cascade-specific knobs.
//!
//! ## Stack effect
//!
//! `( prompt -- handle )`. Same as `plakat.generate` /
//! `plakat.pixart`.
//!
//! ## Usage
//!
//! ```bund
//! "stable-cascade" plakat.load
//! "20" "stage_c_steps" plakat.config.set
//! "10" "stage_b_steps" plakat.config.set
//! "4.0" "guidance" plakat.config.set
//! "a misty forest at dawn" plakat.cascade
//! // → integer image handle on the stack
//! ```
//!
//! ## Configurable knobs
//!
//! - `stage_c_steps` / `stage_b_steps`: Cascade-specific. When
//!   either is unset (`None`), it's derived from the standard
//!   `steps` key via the upstream 2/3 + 1/3 split.
//! - `guidance`, `seed`, `negative`, `scheduler`: shared with
//!   `plakat.generate` / `plakat.pixart`.
//! - `width` / `height`: ignored. Stable Cascade is fixed
//!   1024×1024 (Stage A's design ratio); arbitrary sizes are a
//!   future cycle's work.
//!
//! ## Scope
//!
//! Stable Cascade only (the loaded alias must resolve to a
//! Cascade variant via [`crate::pipelines::t2i::Variant::detect`]).
//! Non-Cascade aliases bail loud with a pointer at the
//! appropriate word (`plakat.generate`, `plakat.pixart`).
//!
//! The pipeline is cached on `ScriptCtx.loaded_cascade`;
//! multi-call scripts amortise the ~14 GB cold load (CLIP-G +
//! Stage A + Stage B + Stage C). Alias change drops; no LoRA
//! invalidation yet (Cascade LoRA support is v0.38 phase 3).

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.cascade";

pub fn plakat_cascade(vm: &mut VM) -> BundResult<'_> {
    do_plakat_cascade(vm).map_err(to_bund_err)
}

fn do_plakat_cascade(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;
    if prompt.is_empty() {
        anyhow::bail!("{TAG}: prompt can't be empty");
    }

    let handle_int = with_ctx_mut(|ctx| -> anyhow::Result<i64> {
        // Variant check: the loaded model must be Stable Cascade.
        // Mirrors the PixArt / SD-family / Flux / SD3 family
        // detection done inside `plakat.pixart`.
        let alias = ctx.loaded_model().ok_or_else(|| {
            anyhow::anyhow!(
                "{TAG}: no model loaded. Call \"stable-cascade\" plakat.load \
                 before {TAG}."
            )
        })?;
        let resolved_for_detect = if alias.contains('/') {
            alias.to_string()
        } else {
            crate::hf::resolve_alias(alias).to_string()
        };
        let variant = crate::pipelines::t2i::Variant::detect(&resolved_for_detect);
        if !variant.is_cascade() {
            anyhow::bail!(
                "{TAG}: loaded model {alias:?} resolves to {variant:?}, not \
                 Stable Cascade. Use `plakat.generate` for SD-family / SD3 / \
                 Flux models, `plakat.pixart` for PixArt-Σ, or call \
                 `\"stable-cascade\" plakat.load` first."
            );
        }

        // Snapshot config-driven knobs. Stable Cascade has TWO
        // step budgets (Stage C heavy, Stage B refines); honour
        // explicit overrides else split the unified `steps` key
        // 2/3 + 1/3 the same way the CLI dispatch does
        // (t2i.rs:2073 / scenario.rs:3477).
        let prompt_owned = prompt.clone();
        let total_steps = ctx.config.steps.max(1);
        let stage_c_steps = ctx
            .config
            .stage_c_steps
            .unwrap_or_else(|| (total_steps * 2).div_ceil(3).max(1));
        let stage_b_steps = ctx.config.stage_b_steps.unwrap_or_else(|| {
            total_steps.saturating_sub(stage_c_steps).max(1)
        });
        let guidance = ctx.config.guidance;
        let scheduler = ctx.config.scheduler;
        let negative = ctx.config.negative.clone();
        let seed = ctx.config.seed.unwrap_or_else(rand::random::<u64>);

        // get_or_load + generate. The borrow of `pipeline` is
        // released before push_image_with_metadata mutates ctx.
        let (buf, ow, oh) = {
            let pipeline = ctx.get_or_load_cascade()?;
            pipeline.generate(
                &prompt_owned,
                &negative,
                stage_c_steps,
                stage_b_steps,
                guidance,
                seed,
                scheduler,
            )?
        };

        // Convert (buf, w, h) → DynamicImage so it fits ctx.images.
        let rgb = image::RgbImage::from_raw(ow, oh, buf).ok_or_else(|| {
            anyhow::anyhow!(
                "{TAG}: Stage A decode produced unexpected byte length"
            )
        })?;
        let img = image::DynamicImage::ImageRgb8(rgb);

        // Build sidecar metadata. Mirrors cascade::run's shape
        // (v0.37 phase 4). `steps` field carries the combined
        // total (stage_c + stage_b) so prompt-info viewers display
        // a single number; the split is preserved in the per-stage
        // tracking inside the pipeline.
        let model = ctx.loaded_model().unwrap_or("stable-cascade").to_string();
        let mut meta = crate::imaging::metadata::GenerationMetadata::new(
            prompt_owned.clone(),
            model,
            seed,
            stage_c_steps + stage_b_steps,
            guidance,
            format!("{scheduler:?}").to_lowercase(),
            ow,
            oh,
        );
        meta.negative = negative;
        // v0.38 phase 3: emit Cascade LoRA stack metadata. The LoRAs
        // themselves merged into Stage B + Stage C tempfiles at load
        // time; this just records what was used per-image.
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
