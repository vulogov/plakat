//! v0.22 phase 8: `plakat.hires.*` host words.
//!
//! Hires fix is a post-process: after `plakat.generate` /
//! `plakat.img2img` / `plakat.portrait` renders at the model's
//! trained resolution, hires-fix upscales the result classically
//! or via Real-ESRGAN, then img2img-refines the upscale at
//! moderate strength to preserve composition while crisping
//! detail. Two state-toggle words plus four bundled
//! Category-B config keys:
//!
//! | Word | Stack effect |
//! |---|---|
//! | `plakat.hires.enable` | `( -- )` — set `ctx.hires_enabled = true` |
//! | `plakat.hires.disable` | `( -- )` — reset it |
//!
//! Config keys (all `plakat.config.set`):
//! - `hires_scale` (float (1, 4], default 2.0) — upscale factor
//! - `hires_strength` (float [0, 1], default 0.5) — refine img2img strength
//! - `hires_upscaler` (string, default "lanczos") — same grammar as
//!   `plakat upscale --method`
//! - `hires_steps` (int (0, 500], default = main `steps`) — refine step count
//!
//! **Family scope (v0.22 phase 8)**: SD-family only. Hires fix
//! requires an SD img2img pipeline for the refine pass. Flux + SD3
//! `plakat.generate` bail when `hires_enabled` is `true` with a
//! clear "SD-family only" message.
//!
//! No cache invalidation — hires fix is a per-call post-process.
//! The cached pipeline's `core()` (`Arc<SdCore>`) is reused inside
//! `refine_files` so no second model load happens.

use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, to_bund_err};

const ENABLE_TAG: &str = "plakat.hires.enable";
const DISABLE_TAG: &str = "plakat.hires.disable";

pub fn plakat_hires_enable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_hires_enable(vm).map_err(to_bund_err)
}

fn do_plakat_hires_enable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        ctx.hires_enabled = true;
    })?;
    tracing::info!(target: "plakat", "{ENABLE_TAG}: post-process ON");
    Ok(vm)
}

pub fn plakat_hires_disable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_hires_disable(vm).map_err(to_bund_err)
}

fn do_plakat_hires_disable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        ctx.hires_enabled = false;
    })?;
    tracing::info!(target: "plakat", "{DISABLE_TAG}: post-process OFF");
    Ok(vm)
}
