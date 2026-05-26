//! v0.21: Bund scripting language integration.
//!
//! Embeds [`bundcore`] so users can drive plakat from a stack-based
//! script (`.bund` file) instead of (or alongside) the CLI
//! subcommands. The script invokes host words namespaced under
//! `plakat.*` which call into the same pipelines `plakat generate`
//! / `img2img` / `portrait` / `upscale` use.
//!
//! Architectural decisions are locked in
//! `Documentation/RFC_v0.21_BUND_SCRIPTING.md`. The relevant ones
//! for this module:
//!
//! * **Embed crate**: `bundcore = "=0.7.0"` — pinned exact.
//!   The `bund` binary crate has no `[lib]` target; `bundcore`
//!   carries the parser + `eval()` wrapper.
//! * **Stdlib strategy**: build our own VM via [`VM::new`] and
//!   register **only** `plakat.*` words. Skip `Bund::new()` so
//!   filesystem / network / shell / sudo stdlib words can't reach
//!   user scripts by construction.
//! * **State sharing**: [`VMInlineFn`] is a bare `fn` pointer, not a
//!   closure. Host words reach plakat state through a process-wide
//!   [`OnceLock<RwLock<ScriptCtx>>`] singleton (see [`ctx`]).
//! * **Async bridge**: bundcore is fully synchronous; plakat
//!   pipelines are `async`. Each host word that hits a pipeline does
//!   `tokio::task::block_in_place(|| Handle::current().block_on(...))`.
//!   [`plakat_run`] must therefore execute on a multi-threaded
//!   tokio runtime — `cli::run` dispatches that way already.

use anyhow::{Context, Result, anyhow};
use bundcore::bundcore::Bund;
use std::path::Path;

pub mod config;
pub mod ctx;
pub mod helpers;
pub mod script_entry;
pub mod words;

pub use ctx::{ScriptCtx, with_ctx, with_ctx_mut};

/// Build a fresh plakat-flavoured Bund instance.
///
/// `Bund::new()` calls `init_lib()` which iterates `bundcore::STDLIB`
/// — a `lazy_static! Mutex<HashMap>` populated only by the `bund`
/// binary crate (which plakat does **not** depend on). With no
/// stdlib registrants, `init_lib()` is a no-op and the resulting VM
/// holds only the `rust_multistackvm` primitives (arithmetic, stack
/// ops, lambdas, control flow). We then register only `plakat.*`
/// words. Net effect of decision #2 (RFC §8): no filesystem,
/// network, shell, or sudo stdlib words can reach user scripts by
/// construction.
pub fn build_plakat_bund() -> Result<Bund> {
    let mut bund = Bund::new();
    words::register_plakat_words(&mut bund.vm)
        .map_err(|e| anyhow!("registering plakat host words: {e}"))?;
    Ok(bund)
}

/// Evaluate a plakat-flavoured Bund script string. Caller is
/// responsible for [`ScriptCtx::init`] before invoking; host words
/// look the context up unconditionally and will bail loud if it's
/// missing.
pub fn eval(source: &str) -> Result<()> {
    let mut bund = build_plakat_bund()?;
    bund.eval(source)
        .map_err(|e| anyhow!("bund eval failed: {e}"))?;
    Ok(())
}

/// Convenience: read `path` and eval its contents.
pub fn eval_file(path: &Path) -> Result<()> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading script {}", path.display()))?;
    eval(&source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// v0.21 phase 1 smoke test: VM build + register + eval a
    /// trivial script that exercises `plakat.echo`. Verifies the
    /// entire integration shape end-to-end:
    ///
    /// * `bundcore` linked + parses
    /// * `VM::new()` returns a usable VM
    /// * `register_plakat_words` registers without collision
    /// * `ScriptCtx::init` + the singleton are reachable from a
    ///   host fn
    /// * The async bridge inside `plakat.echo` resolves (we drive
    ///   eval from a multi-threaded tokio runtime here, matching
    ///   the production CLI dispatch)
    ///
    /// **Test isolation note**: ScriptCtx is a process-wide
    /// singleton (OnceLock) and can only be initialised once per
    /// process. All tests that need a context must serialise
    /// through one shared init — see `with_singleton_ctx` below.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn echo_round_trip() {
        with_singleton_ctx(|| {
            // The simplest script that exercises a host word: push a
            // string, call echo, drop the result.
            eval("\"hello from phase 1\" plakat.echo drop").unwrap();
        });
    }

    /// v0.21 phase 2 round-trip: eval a script that uses
    /// `plakat.save` against an image pre-stuffed into the context
    /// (so we don't have to run the SD pipeline in unit tests).
    /// Exercises the full pull/handle/save path the host word
    /// implements, end-to-end via `eval()`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn save_round_trip_via_eval() {
        with_singleton_ctx(|| {
            // Stuff a tiny image into the context + remember the
            // out_dir so we can assert the save landed.
            let (handle, out_dir) = with_ctx_mut(|ctx| {
                let img = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(4, 4, image::Rgb([1, 2, 3])),
                );
                (ctx.push_image(img), ctx.out_dir.clone())
            })
            .unwrap();

            // Script: push handle, push path, call save. The
            // out_dir prefix gets prepended for relative paths.
            let script = format!(
                "{handle} \"phase2-save-test.png\" plakat.save"
            );
            eval(&script).unwrap();

            assert!(out_dir.join("phase2-save-test.png").exists());
        });
    }

    /// v0.21 phase 3 round-trip: eval a script that mutates several
    /// `plakat.config.set` knobs (int, float, string keys) and
    /// verify the GenerationConfig reflects them. Doesn't run a
    /// real generation; just proves the host word + value dispatch
    /// + GenerationConfig::set_* compose end-to-end through eval.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn config_set_round_trip_via_eval() {
        with_singleton_ctx(|| {
            // Wipe to defaults so this test doesn't depend on
            // others' state.
            with_ctx_mut(|ctx| {
                ctx.config = config::GenerationConfig::default();
            })
            .unwrap();
            // Three pushes of (value, key) pairs covering each
            // value-type branch (int, float, string).
            let script = r#"
                50    "steps"      plakat.config.set
                3.5   "guidance"   plakat.config.set
                "blurry" "negative" plakat.config.set
                "euler-a" "scheduler" plakat.config.set
                42    "seed"       plakat.config.set
            "#;
            eval(script).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.config.steps, 50);
                assert!((ctx.config.guidance - 3.5).abs() < 1e-9);
                assert_eq!(ctx.config.negative, "blurry");
                assert!(matches!(
                    ctx.config.scheduler,
                    crate::pipelines::scheduler::SchedulerKind::EulerA
                ));
                assert_eq!(ctx.config.seed, Some(42));
            })
            .unwrap();
        });
    }

    /// v0.21 phase 4: handle-reuse path. We can't run the real
    /// img2img pipeline in CI (needs SD weights), but we can pin
    /// the "bail with helpful message" surface for two failure
    /// modes that have to keep working as the codebase evolves:
    ///
    /// 1. img2img with no model loaded → "no model loaded" pointer
    ///    fires from inside the host word, before any tempfile
    ///    materialisation happens.
    /// 2. img2img with an invalid input type (neither string nor
    ///    int) → typed error from the dispatch arm.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn img2img_no_model_loaded_bails_via_eval() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded_model = None;
            })
            .unwrap();
            let err = eval(
                "\"a fox\" \"/tmp/does-not-matter.png\" plakat.img2img",
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no model loaded"), "got {msg}");
        });
    }

    /// v0.21 phase 4: handle-vs-path dispatch — when the script
    /// passes a handle that doesn't exist, the lookup fails inside
    /// the word with the same "image handle N not found" message
    /// `ctx.image_at` produces. Pins that the int dispatch arm
    /// routes through `image_at` (not silently treating the int as
    /// a path string).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn img2img_unknown_handle_bails_with_image_at_message() {
        with_singleton_ctx(|| {
            // Pretend a model is loaded so we get past the load gate
            // and into the handle resolution.
            with_ctx_mut(|ctx| {
                ctx.loaded_model = Some("sd15".to_string());
                // Reset the image registry so the handle is
                // genuinely unknown (other tests may have pushed).
                ctx.images.clear();
            })
            .unwrap();
            let err = eval("\"a fox\" 999 plakat.img2img").unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("image handle 999"), "got {msg}");
        });
    }

    /// v0.21 phase 6: upscale round-trip. Stuff a hand-crafted
    /// 16×16 image into the registry, eval `1 2 plakat.upscale`,
    /// verify the new handle's image is 32×32. Unlike phases 4 +
    /// 5, this one runs the **real** transform in CI (no SD
    /// weights involved — pure image-crate Lanczos resize).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upscale_round_trip_via_eval() {
        with_singleton_ctx(|| {
            let src_handle = with_ctx_mut(|ctx| {
                ctx.images.clear();
                let img = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(16, 16, image::Rgb([7, 7, 7])),
                );
                ctx.push_image(img)
            })
            .unwrap();
            assert_eq!(src_handle, 1);

            // 1 2 plakat.upscale → new handle (2). Bund prints the
            // returned handle but we don't read the stack here; we
            // verify via the registry instead.
            eval("1 2 plakat.upscale drop").unwrap();

            with_ctx(|ctx| {
                assert_eq!(ctx.images.len(), 2, "expected upscaled image to land at handle 2");
                let dst = &ctx.images[1];
                assert_eq!(dst.width(), 32);
                assert_eq!(dst.height(), 32);
                // Source should still be addressable (handle reuse contract).
                let src = &ctx.images[0];
                assert_eq!(src.width(), 16);
                assert_eq!(src.height(), 16);
            })
            .unwrap();
        });
    }

    /// v0.21 phase 6: bad scale bails through eval with the
    /// "scale must be 2 or 4" message.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upscale_invalid_scale_bails_via_eval() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.images.clear();
                let img = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(8, 8, image::Rgb([0, 0, 0])),
                );
                ctx.push_image(img);
            })
            .unwrap();
            let err = eval("1 3 plakat.upscale").unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("scale must be 2 or 4"), "got {msg}");
            assert!(msg.contains("v0.22"), "got {msg}");
        });
    }

    /// v0.21 phase 6: unknown handle bails through the
    /// `ctx.image_at` lookup, same as img2img / portrait.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upscale_unknown_handle_bails_via_eval() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.images.clear();
            })
            .unwrap();
            let err = eval("999 2 plakat.upscale").unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("image handle 999"), "got {msg}");
        });
    }

    /// v0.21 phase 5: portrait gate — no model loaded bails with
    /// the "Call \"sdxl\" plakat.load" pointer (note the SDXL
    /// suggestion in the message, not just sd15).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn portrait_no_model_loaded_bails_via_eval() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded_model = None;
            })
            .unwrap();
            let err = eval(
                "\"a portrait\" \"/tmp/me.jpg\" plakat.portrait",
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no model loaded"), "got {msg}");
            // The portrait-specific pointer recommends sdxl too,
            // not just sd15 — pin that so a future tweak doesn't
            // silently drop the SDXL hint.
            assert!(msg.contains("sdxl"), "got {msg}");
        });
    }

    /// v0.21 phase 5: unknown handle for the photo arg surfaces
    /// the same image_at error img2img uses. Confirms the int
    /// dispatch arm shares the lookup path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn portrait_unknown_handle_bails_with_image_at_message() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded_model = Some("sd15".to_string());
                ctx.images.clear();
            })
            .unwrap();
            let err = eval("\"a portrait\" 999 plakat.portrait").unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("image handle 999"), "got {msg}");
        });
    }

    /// v0.21 phase 3: an unknown key surfaces a clear error from
    /// inside eval. Exercise the failure mode end-to-end so we
    /// know the helpful error message actually reaches user
    /// scripts and isn't swallowed by bundcore.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn config_set_unknown_key_bails_via_eval() {
        with_singleton_ctx(|| {
            let err = eval(
                "1 \"definitely-not-a-real-key\" plakat.config.set",
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("unknown key"), "got {msg}");
            assert!(msg.contains("steps"), "got {msg}");
        });
    }

    /// Test-helper: serialises every test that needs the singleton
    /// context behind one shared init. Subsequent calls re-use the
    /// already-init'd singleton; only the *first* test through the
    /// gate pays the init cost. Each test runs its body inside the
    /// mutex so they don't trample each other's state on `out_dir`
    /// or `images`.
    fn with_singleton_ctx<R>(body: impl FnOnce() -> R) -> R {
        use std::sync::Mutex;
        static GATE: Mutex<()> = Mutex::new(());
        let _g = GATE.lock().unwrap();
        if ctx::CTX.get().is_none() {
            let tmp = tempfile::tempdir().unwrap();
            // Leak the tempdir to keep the path alive for the
            // singleton's lifetime — only a single tempdir per
            // process, harmless.
            let path = tmp.path().to_path_buf();
            std::mem::forget(tmp);
            ScriptCtx::init(Device::Cpu, path).unwrap();
        }
        body()
    }
}
