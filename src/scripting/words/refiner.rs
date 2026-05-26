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
//! **Wiring status (v0.22 phase 6)**:
//! - **Same-model polish** (`refine_steps` + `refine_strength`):
//!   wired today. The portrait::Pipeline cache exposes
//!   `GenRequest.refine` + `refine_strength` which
//!   `script_entry::build_gen_request` populates from these
//!   config keys. SD-family only; Flux + SD3 don't have an
//!   equivalent polish path.
//! - **SDXL refiner UNet** (`plakat.refiner.enable` +
//!   `refiner_frac`): the toggle exists today but loading the
//!   actual refiner UNet requires switching the SD-family cache
//!   from `portrait::Pipeline` to `t2i::Pipeline` (the only
//!   pipeline that holds an optional `refiner_unet`). That's a
//!   v0.23 refactor. Setting `refiner_enabled = true` and then
//!   calling `plakat.generate` bails with a clear deferral
//!   message + the workaround (use the CLI's `--refiner` for
//!   now, or `plakat.refiner.disable`).
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
