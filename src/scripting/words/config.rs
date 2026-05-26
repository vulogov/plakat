//! v0.21 phase 3: `plakat.config.set ( value key -- )`.
//!
//! Sets one knob on the script's accumulated [`GenerationConfig`].
//! Stack order: bottom = value, top = key string. Top pops first.
//!
//! Keys + expected value types (rejected with a helpful message
//! otherwise):
//!
//! | Key | Type | Notes |
//! |---|---|---|
//! | `steps` | int | > 0 |
//! | `guidance` | float\|int | finite |
//! | `seed` | int | >= 0; persistent across calls |
//! | `width` | int | > 0, /8, ≤ 4096 |
//! | `height` | int | same as width |
//! | `negative` | string | passthrough |
//! | `scheduler` | string | one of default\|ddim\|euler-a\|… |
//!
//! Example:
//!
//! ```bund
//! "sdxl" plakat.load
//! 40   "steps"     plakat.config.set
//! 7.5  "guidance"  plakat.config.set
//! 1024 "width"     plakat.config.set
//! 1024 "height"    plakat.config.set
//! 42   "seed"      plakat.config.set
//! "blurry, low quality" "negative" plakat.config.set
//! "euler-a" "scheduler" plakat.config.set
//!
//! "a fox in a meadow" plakat.generate "fox.png" plakat.save
//! ```

use rust_dynamic::types;
use rust_multistackvm::multistackvm::VM;

use crate::scripting::ctx::with_ctx_mut;
use crate::scripting::helpers::{
    BundResult, pull, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.config.set";

pub fn plakat_config_set(vm: &mut VM) -> BundResult<'_> {
    do_plakat_config_set(vm).map_err(to_bund_err)
}

fn do_plakat_config_set(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 2, TAG)?;
    // Top of stack popped first: the key.
    let key_v = pull(vm, TAG)?;
    let value_v = pull(vm, TAG)?;
    let key = value_to_string(key_v, "key", TAG)?;

    // Dispatch on the value's underlying dynamic type rather than
    // forcing every value through a string round-trip. Integer
    // values can apply cleanly to int keys; float values can apply
    // to float keys (or to int keys with a fractional check); only
    // string keys (negative / scheduler) round-trip via the
    // string path. The `dt` field on `Value` carries the type tag.
    with_ctx_mut(|ctx| match value_v.dt {
        types::INTEGER => ctx.config.set_int(&key, value_v.cast_int().unwrap_or(0)),
        types::FLOAT => ctx
            .config
            .set_float(&key, value_v.cast_float().unwrap_or(0.0)),
        types::STRING => {
            let s = value_v.cast_string().unwrap_or_default();
            ctx.config.set_str(&key, &s)
        }
        _ => Err(anyhow::anyhow!(
            "{TAG}: value for key {key:?} must be int, float, or string \
             (got rust_dynamic dt = {})",
            value_v.dt
        )),
    })??;
    tracing::info!(target: "plakat", "{TAG}: {key} updated");
    Ok(vm)
}
