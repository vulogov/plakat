//! v1.0: `plakat.tiled.*` host words — tiled hi-res generation for the SD-family
//! `plakat.generate`, mirroring the CLI `--tiled`. The UNet only ever sees tiles
//! of `tile_size` × `tile_size` (blended with a Hann window), so you can render
//! well above a model's trained resolution without the OOM or repetition of a
//! single huge pass.
//!
//! | Word | Stack effect |
//! |---|---|
//! | `plakat.tiled.enable` | `( -- )` — set `ctx.config.tiled = true` |
//! | `plakat.tiled.disable` | `( -- )` — reset it |
//!
//! Config keys (`plakat.config.set`):
//! - `tile_size` (px, default 1024) — the per-tile render size
//! - `tile_stride` (px, multiple of 8 and ≤ `tile_size`, default 768) — tile step
//!
//! **Scope:** the SD-family `plakat.generate` path. Doesn't compose with
//! ControlNet (the tiled denoise has no conditioning slot) — `plakat.generate`
//! bails if both are set. A per-call generate option; no cache invalidation.

use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{BundResult, to_bund_err};

const ENABLE_TAG: &str = "plakat.tiled.enable";
const DISABLE_TAG: &str = "plakat.tiled.disable";

pub fn plakat_tiled_enable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_tiled_enable(vm).map_err(to_bund_err)
}

fn do_plakat_tiled_enable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        ctx.config.tiled = true;
    })?;
    tracing::info!(target: "plakat", "{ENABLE_TAG}: tiled hi-res ON");
    Ok(vm)
}

pub fn plakat_tiled_disable(vm: &mut VM) -> BundResult<'_> {
    do_plakat_tiled_disable(vm).map_err(to_bund_err)
}

fn do_plakat_tiled_disable(vm: &mut VM) -> anyhow::Result<&mut VM> {
    with_ctx_mut(|ctx| {
        ctx.config.tiled = false;
    })?;
    tracing::info!(target: "plakat", "{DISABLE_TAG}: tiled hi-res OFF");
    Ok(vm)
}
