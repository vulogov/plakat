//! v0.22 phase 6: `plakat.refiner.*` host words.
//!
//! Two state-toggle words plus three config keys (declared in
//! [`super::super::config`]):
//!
//! | Word | Stack effect |
//! |---|---|
//! | `plakat.refiner.enable` | `( -- )` — set `ctx.refiner_enabled = true` |
//! | `plakat.refiner.disable` | `( -- )` — reset it |
//!
//! Plus `plakat.config.set` keys:
//! - `refine_steps` (int) — same-model polish pass step count
//! - `refine_strength` (float [0,1]) — polish denoise strength
//! - `refiner_frac` (float [0,1]) — fraction at which the SDXL refiner UNet takes over
//!
//! **Wiring status (v0.22 phase 6 + v0.23 phase 2)**:
//! - **Same-model polish** (`refine_steps` + `refine_strength`):
//!   wired in v0.22. SD-family only; Flux + SD3 don't have an
//!   equivalent polish path.
//! - **SDXL refiner UNet** (`plakat.refiner.enable` +
//!   `refiner_frac`): wired in v0.23 phase 2. The v0.23 phase 1
//!   SdT2i cache slot holds an optional refiner-UNet from
//!   t2i::Pipeline; ScriptCtx::get_or_load_sd_t2i sets
//!   `use_refiner` from `ctx.refiner_enabled`. SDXL-only; non-SDXL
//!   aliases silently downgrade with a warn (same behaviour as
//!   the CLI's `--refiner` flag).
//!
//! Cache invalidation: the SDXL refiner is a load-time pipeline
//! feature (the refiner-UNet weights are mmapped at load), so
//! mutating `refiner_enabled` invalidates the cache the same way
//! `plakat.lora.*` mutations do. The same-model polish doesn't
//! need invalidation — it runs in `pipeline.generate` directly
//! from the GenRequest.

use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, to_bund_err};

const ENABLE_TAG: &str = "plakat.refiner.enable";
const DISABLE_TAG: &str = "plakat.refiner.disable";

pub fn plakat_refiner_enable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_refiner_enable(vm).map_err(to_bund_err)
}

fn do_plakat_refiner_enable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        let was = ctx.refiner_enabled;
        ctx.refiner_enabled = true;
        if !was {
            // Toggle changed → invalidate cache so the next
            // generate would (in a future cycle) reload with the
            // refiner UNet attached.
            ctx.mark_loras_changed();
        }
    })?;
    tracing::info!(target: "plakat", "{ENABLE_TAG}: SDXL refiner toggle ON");
    Ok(vm)
}

pub fn plakat_refiner_disable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_refiner_disable(vm).map_err(to_bund_err)
}

fn do_plakat_refiner_disable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        let was = ctx.refiner_enabled;
        ctx.refiner_enabled = false;
        if was {
            ctx.mark_loras_changed();
        }
    })?;
    tracing::info!(target: "plakat", "{DISABLE_TAG}: SDXL refiner toggle OFF");
    Ok(vm)
}
