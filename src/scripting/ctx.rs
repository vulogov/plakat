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

use crate::scripting::config::GenerationConfig;

/// Process-wide script context. Holds the device + output dir +
/// the in-script image registry + the active model alias.
///
/// One script per process by construction — bundcore's VM has no
/// per-eval isolation and the singleton can only be written once.
pub struct ScriptCtx {
    pub device: Device,
    pub out_dir: PathBuf,
    /// v0.21 phase 2: model alias the script most recently set
    /// via `plakat.load`. `plakat.generate` uses this. `None`
    /// means no model has been loaded yet — `plakat.generate`
    /// bails with a clear message.
    pub loaded_model: Option<String>,
    /// v0.21 phase 2: rendered images, addressable by the integer
    /// handle pushed onto the stack by `plakat.generate`. Index =
    /// handle (1-based — handle 0 is reserved as "no image").
    /// Phase 2 keeps every rendered image in memory for the
    /// script's lifetime; if scripts ever start producing hundreds
    /// of images we'll revisit (e.g. spill to disk).
    pub images: Vec<image::DynamicImage>,
    /// v0.21 phase 3: generation knobs the script accumulates via
    /// `plakat.config.set`. Persistent across calls within one
    /// script. Read by [`super::script_entry::generate_one`] when
    /// building the `t2i::Request`.
    pub config: GenerationConfig,
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
        CTX.set(RwLock::new(ScriptCtx {
            device,
            out_dir,
            loaded_model: None,
            images: Vec::new(),
            config: GenerationConfig::default(),
        }))
        .map_err(|_| anyhow!("ScriptCtx already initialised"))
    }

    /// v0.21 phase 2: register a rendered image and return the
    /// 1-based handle the script will see. Caller is responsible
    /// for serialising mutation through [`with_ctx_mut`].
    pub fn push_image(&mut self, img: image::DynamicImage) -> i64 {
        self.images.push(img);
        self.images.len() as i64
    }

    /// v0.21 phase 2: look up an image by its script-visible
    /// handle. Bails on unknown handles + on the reserved 0
    /// handle.
    pub fn image_at(&self, handle: i64) -> Result<&image::DynamicImage> {
        if handle <= 0 {
            return Err(anyhow!(
                "image handle must be >= 1 (got {handle}); handle 0 is reserved"
            ));
        }
        let idx = handle as usize - 1;
        self.images.get(idx).ok_or_else(|| {
            anyhow!(
                "image handle {handle} not found (only {} image(s) rendered so far)",
                self.images.len()
            )
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn mk_ctx() -> ScriptCtx {
        ScriptCtx {
            device: Device::Cpu,
            out_dir: std::env::temp_dir(),
            loaded_model: None,
            images: Vec::new(),
            config: GenerationConfig::default(),
        }
    }

    fn mk_image(r: u8) -> DynamicImage {
        let mut img = RgbImage::new(2, 2);
        for p in img.pixels_mut() {
            *p = image::Rgb([r, 0, 0]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn push_image_returns_one_based_handle() {
        let mut ctx = mk_ctx();
        assert_eq!(ctx.push_image(mk_image(10)), 1);
        assert_eq!(ctx.push_image(mk_image(20)), 2);
        assert_eq!(ctx.push_image(mk_image(30)), 3);
    }

    #[test]
    fn image_at_returns_the_pushed_image() {
        let mut ctx = mk_ctx();
        let h = ctx.push_image(mk_image(99));
        let got = ctx.image_at(h).unwrap();
        // pixel (0,0) should be (99, 0, 0)
        let rgb = got.to_rgb8();
        let p = rgb.get_pixel(0, 0);
        assert_eq!(p.0, [99, 0, 0]);
    }

    #[test]
    fn image_at_zero_bails_with_reserved_message() {
        let ctx = mk_ctx();
        let err = ctx.image_at(0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("reserved"), "got {msg}");
    }

    #[test]
    fn image_at_negative_bails() {
        let ctx = mk_ctx();
        assert!(ctx.image_at(-1).is_err());
    }

    #[test]
    fn image_at_unknown_handle_includes_rendered_count() {
        let mut ctx = mk_ctx();
        ctx.push_image(mk_image(1));
        let err = ctx.image_at(99).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("99"), "got {msg}");
        // The diagnostic mentions the rendered count so users can
        // tell whether they're addressing a future handle vs a
        // typo.
        assert!(msg.contains("1 image"), "got {msg}");
    }
}
