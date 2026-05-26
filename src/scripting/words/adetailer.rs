//! v0.22 phase 7: `plakat.adetailer.*` host words.
//!
//! ADetailer is a post-process — after `plakat.generate` /
//! `plakat.img2img` renders the main image, SCRFD detects face
//! bboxes and re-runs an img2img pass on each face crop, then
//! feather-composites the refined faces back. Two state-toggle
//! words plus six bundled Category-B config keys:
//!
//! | Word | Stack effect |
//! |---|---|
//! | `plakat.adetailer.enable` | `( -- )` — set `ctx.adetailer_enabled = true` |
//! | `plakat.adetailer.disable` | `( -- )` — reset it |
//!
//! Config keys (all `plakat.config.set`):
//! - `adetailer_strength` (float [0, 1], default 0.4)
//! - `adetailer_padding` (float [0, 1], default 0.25)
//! - `adetailer_feather` (float [0, 1], default 0.25)
//! - `adetailer_confidence` (float [0, 1], default 0.5)
//! - `adetailer_size` (int, multiple of 8, default 512)
//! - `adetailer_prompt` (string, default "detailed face, sharp focus, high quality")
//!
//! **Family scope (v0.22 phase 7)**: SD-family only. ADetailer
//! requires SCRFD face detection + an SD img2img pipeline for
//! the face crops. Flux + SD3 don't have an equivalent
//! post-process; `plakat.generate` on those families bails when
//! adetailer is enabled with a clear "SD-family only" message.
//!
//! **Requires SCRFD weights**: `PLAKAT_SCRFD_WEIGHTS` (local
//! safetensors path) or `PLAKAT_SCRFD_HF` (HF `repo#file` spec)
//! env var must be set. The bail message from
//! `adetailer::refine_files` lists both options when neither is
//! configured.
//!
//! No cache invalidation — ADetailer is a per-call
//! post-process. The cached pipeline's `core()` (`Arc<SdCore>`)
//! is reused inside `refine_files` so no second model load
//! happens.

use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, to_bund_err};

const ENABLE_TAG: &str = "plakat.adetailer.enable";
const DISABLE_TAG: &str = "plakat.adetailer.disable";

pub fn plakat_adetailer_enable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_adetailer_enable(vm).map_err(to_bund_err)
}

fn do_plakat_adetailer_enable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        ctx.adetailer_enabled = true;
    })?;
    tracing::info!(target: "plakat", "{ENABLE_TAG}: post-process ON");
    Ok(vm)
}

pub fn plakat_adetailer_disable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_adetailer_disable(vm).map_err(to_bund_err)
}

fn do_plakat_adetailer_disable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        ctx.adetailer_enabled = false;
    })?;
    tracing::info!(target: "plakat", "{DISABLE_TAG}: post-process OFF");
    Ok(vm)
}
