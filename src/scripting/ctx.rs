//! v0.21: process-global script context.
//!
//! Host words registered into bundcore can't capture closures
//! (`VMInlineFn` is a bare `fn` pointer). They reach plakat state
//! via the [`CTX`] singleton — the same pattern blackInkhaven uses
//! for its `ADAM` VM and `ACTIVE_STORE` project handle.
//!
//! Phase 1 carries only `device` + `out_dir`; phase 2 will add a
//! lazy-loaded `HashMap<String, LoadedPipeline>` so scripts can
//! reuse a loaded model across calls without paying the model-load
//! cost per `plakat.generate`.

use anyhow::{Result, anyhow};
use candle_core::Device;
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

/// Process-wide script context. Holds the device + output dir +
/// (phase 2+) the cache of loaded pipelines.
///
/// One script per process by construction — bundcore's VM has no
/// per-eval isolation and the singleton can only be written once.
pub struct ScriptCtx {
    pub device: Device,
    pub out_dir: PathBuf,
    // Phase 2: lazily-loaded pipelines keyed by `--model` alias.
    // pub loaded: HashMap<String, LoadedPipeline>,
    // Phase 2: ScriptCtx.last_image for the handle-reuse contract.
    // pub last_image: Option<DynamicImage>,
}

impl ScriptCtx {
    /// Initialise the singleton. Called once at the top of
    /// `cli::run::run` after the CLI device selection lands. A
    /// second call after the first is a hard error — bundcore
    /// can't run two scripts concurrently in one process.
    pub fn init(device: Device, out_dir: PathBuf) -> Result<()> {
        std::fs::create_dir_all(&out_dir).map_err(|e| {
            anyhow!("creating script output dir {}: {e}", out_dir.display())
        })?;
        CTX.set(RwLock::new(ScriptCtx { device, out_dir }))
            .map_err(|_| anyhow!("ScriptCtx already initialised"))
    }
}

/// The singleton. Using `std::sync::RwLock` to keep the dep
/// footprint flat; phase-1's contention story is "one host word
/// at a time on one thread" so the lighter parking_lot variant
/// wouldn't pay back.
pub(crate) static CTX: OnceLock<RwLock<ScriptCtx>> = OnceLock::new();

/// Borrow the script context for a read. Bails if [`ScriptCtx::init`]
/// hasn't run yet — host words always need a context.
pub fn with_ctx<R>(f: impl FnOnce(&ScriptCtx) -> R) -> Result<R> {
    let lock = CTX
        .get()
        .ok_or_else(|| anyhow!("ScriptCtx not initialised — was `plakat run` invoked?"))?;
    let guard = lock
        .read()
        .map_err(|e| anyhow!("ScriptCtx read lock poisoned: {e}"))?;
    Ok(f(&guard))
}

/// Borrow the script context for a write.
pub fn with_ctx_mut<R>(f: impl FnOnce(&mut ScriptCtx) -> R) -> Result<R> {
    let lock = CTX
        .get()
        .ok_or_else(|| anyhow!("ScriptCtx not initialised — was `plakat run` invoked?"))?;
    let mut guard = lock
        .write()
        .map_err(|e| anyhow!("ScriptCtx write lock poisoned: {e}"))?;
    Ok(f(&mut guard))
}
