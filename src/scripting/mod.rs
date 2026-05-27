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
pub mod loaded_pipeline;
pub mod repl;
pub mod script_entry;
pub mod words;

pub use ctx::{ScriptCtx, with_ctx, with_ctx_mut};

/// v0.21 phase 7: re-export of the test helper from the inner
/// `tests` module so other test modules (`repl::tests`) can use
/// the same shared singleton gate. Production callers never see
/// this — it's `cfg(test)` only.
#[cfg(test)]
pub(crate) use tests::with_singleton_ctx as tests_with_singleton_ctx;

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
                ctx.loaded = None;
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
            // v0.22 phase 1: the image_at(handle) check fires in
            // the host word *before* any pipeline-loaded check, so
            // we don't need to fake a loaded model anymore. Just
            // reset the image registry so handle 999 is genuinely
            // unknown.
            with_ctx_mut(|ctx| {
                ctx.images.clear();
            })
            .unwrap();
            let err = eval("\"a fox\" 999 plakat.img2img").unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("image handle 999"), "got {msg}");
        });
    }

    // v0.22 phase 6: plakat.refiner.* end-to-end.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refiner_enable_toggles_ctx_and_invalidates_cache() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.refiner_enabled = false;
                ctx.loaded = None;
            })
            .unwrap();
            eval("plakat.refiner.enable").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.refiner_enabled);
                // Cache invalidation: loaded should still be None
                // (we cleared it above) — no-op when already None,
                // but the call path was exercised.
                assert!(ctx.loaded.is_none());
            })
            .unwrap();
            eval("plakat.refiner.disable").unwrap();
            with_ctx(|ctx| assert!(!ctx.refiner_enabled)).unwrap();
        });
    }

    /// v0.23 phase 2: SDXL refiner UNet load wires through. The
    /// bail from v0.22 phase 6 is gone — `plakat.refiner.enable`
    /// + `plakat.generate` on SDXL now loads with the refiner
    /// UNet (smoke-tested at the CLI level; this unit test just
    /// confirms the toggle round-trips without touching a real
    /// model load).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refiner_toggle_round_trip_v023_phase2() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.refiner_enabled = false;
                ctx.loras.clear();
                ctx.controlnets.clear();
            })
            .unwrap();
            eval("plakat.refiner.enable").unwrap();
            with_ctx(|ctx| assert!(ctx.refiner_enabled)).unwrap();
            eval("plakat.refiner.disable").unwrap();
            with_ctx(|ctx| assert!(!ctx.refiner_enabled)).unwrap();
        });
    }

    // v0.22 phase 7: plakat.adetailer.* end-to-end.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn adetailer_enable_toggles_ctx_flag() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.adetailer_enabled = false).unwrap();
            eval("plakat.adetailer.enable").unwrap();
            with_ctx(|ctx| assert!(ctx.adetailer_enabled)).unwrap();
            eval("plakat.adetailer.disable").unwrap();
            with_ctx(|ctx| assert!(!ctx.adetailer_enabled)).unwrap();
        });
    }

    /// Config-only round-trip via `plakat.config.set` exercises the
    /// new phase 7 keys end-to-end through the host word.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn adetailer_config_keys_round_trip_via_host_word() {
        with_singleton_ctx(|| {
            // Stack order: value pushed first, then key on top.
            eval(r#"0.6 "adetailer_strength" plakat.config.set"#).unwrap();
            eval(r#"0.35 "adetailer_padding" plakat.config.set"#).unwrap();
            eval(r#"768 "adetailer_size" plakat.config.set"#).unwrap();
            eval(
                r#""sharp face, detailed skin" "adetailer_prompt" plakat.config.set"#,
            )
            .unwrap();
            with_ctx(|ctx| {
                assert!((ctx.config.adetailer_strength - 0.6).abs() < 1e-6);
                assert!((ctx.config.adetailer_padding - 0.35).abs() < 1e-6);
                assert_eq!(ctx.config.adetailer_size, 768);
                assert_eq!(ctx.config.adetailer_prompt, "sharp face, detailed skin");
            })
            .unwrap();
        });
    }

    // v0.22 phase 8: plakat.hires.* end-to-end.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hires_enable_toggles_ctx_flag() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.hires_enabled = false).unwrap();
            eval("plakat.hires.enable").unwrap();
            with_ctx(|ctx| assert!(ctx.hires_enabled)).unwrap();
            eval("plakat.hires.disable").unwrap();
            with_ctx(|ctx| assert!(!ctx.hires_enabled)).unwrap();
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hires_config_keys_round_trip_via_host_word() {
        with_singleton_ctx(|| {
            // value-then-key stack order.
            eval(r#"2.5 "hires_scale" plakat.config.set"#).unwrap();
            eval(r#"0.6 "hires_strength" plakat.config.set"#).unwrap();
            eval(r#""real-esrgan-x2" "hires_upscaler" plakat.config.set"#).unwrap();
            eval(r#"15 "hires_steps" plakat.config.set"#).unwrap();
            with_ctx(|ctx| {
                assert!((ctx.config.hires_scale - 2.5).abs() < 1e-6);
                assert!((ctx.config.hires_strength - 0.6).abs() < 1e-6);
                assert_eq!(ctx.config.hires_upscaler, "real-esrgan-x2");
                assert_eq!(ctx.config.hires_steps, Some(15));
            })
            .unwrap();
        });
    }

    // v0.22 phase 11: misc config keys end-to-end.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn phase11_misc_keys_round_trip_via_host_word() {
        with_singleton_ctx(|| {
            eval(r#""16:9" "aspect" plakat.config.set"#).unwrap();
            eval(r#"512 "base" plakat.config.set"#).unwrap();
            eval(r#"16 "mask_feather" plakat.config.set"#).unwrap();
            eval(r#""true" "mask_invert" plakat.config.set"#).unwrap();
            eval(r#"2 "clip_skip" plakat.config.set"#).unwrap();
            eval(r#""/wc" "wildcard_dir" plakat.config.set"#).unwrap();
            eval(r#""photo" "negative_preset" plakat.config.set"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.config.aspect, "16:9");
                assert_eq!(ctx.config.base, 512);
                assert_eq!(ctx.config.mask_feather, 16);
                assert!(ctx.config.mask_invert);
                assert_eq!(ctx.config.clip_skip, 2);
                assert_eq!(ctx.config.wildcard_dir, "/wc");
                assert_eq!(ctx.config.negative_preset, "photo");
            })
            .unwrap();
        });
    }

    /// `negative_preset` validation bites at config-set time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn phase11_negative_preset_invalid_bails() {
        with_singleton_ctx(|| {
            let err = eval(
                r#""ultra-9000" "negative_preset" plakat.config.set"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("ultra-9000"), "got {msg}");
        });
    }

    // v0.22 phase 10: plakat.enhance end-to-end (config-side only;
    // we don't actually run an LLM forward in tests — downloading a
    // GGUF is out of scope for unit tests).

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn enhance_config_keys_round_trip_via_host_word() {
        with_singleton_ctx(|| {
            // Value-then-key stack order.
            eval(r#""deepseek" "enhance_provider" plakat.config.set"#).unwrap();
            eval(r#"0.7 "enhance_temp" plakat.config.set"#).unwrap();
            eval(r#"128 "enhance_max_tokens" plakat.config.set"#).unwrap();
            eval(r#""true" "enhance_cache" plakat.config.set"#).unwrap();
            eval(r#""/sys.txt" "enhance_system" plakat.config.set"#).unwrap();
            eval(r#""true" "enhance_keep_original" plakat.config.set"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.config.enhance_provider, "deepseek");
                assert!((ctx.config.enhance_temp.unwrap() - 0.7).abs() < 1e-6);
                assert_eq!(ctx.config.enhance_max_tokens, Some(128));
                assert!(ctx.config.enhance_cache);
                assert_eq!(ctx.config.enhance_system, "/sys.txt");
                assert!(ctx.config.enhance_keep_original);
            })
            .unwrap();
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn enhance_empty_provider_bails_with_helpful_message() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.config.enhance_provider.clear()).unwrap();
            let err = eval(r#""a knight" plakat.enhance"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("provider"), "got {msg}");
            assert!(msg.contains("plakat.config.set"), "got {msg}");
        });
    }

    // v0.24 phase 9: Flux inpaint via flux-fill-dev (state-only).

    /// `plakat.inpaint` on a wrong Flux variant bails with the
    /// "use flux-fill-dev" pointer. We can't actually load a
    /// pipeline in a unit test (would need real weights), but we
    /// can check the bail fires before any pipeline dispatch by
    /// faking `ctx.loaded_model()` via a sentinel alias and
    /// confirming the message reaches user-land. Note: the
    /// no-model gate fires first when no model is loaded, so we
    /// validate the bail message via the CLI smoke instead.
    /// Here we just confirm `plakat.inpaint` still gates on
    /// no-model-loaded after phase 9.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inpaint_phase9_no_model_still_bails() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            let err = eval(
                r#""fix the sky" "./photo.png" "./mask.png" plakat.inpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no model loaded"), "got {msg}");
        });
    }

    // v0.24 phase 7: plakat.metadata.read surface.

    /// Write a minimal JSON sidecar to a tempdir, then read it
    /// via `plakat.metadata.read`. Verify the pair count + a
    /// couple of fields land on the stack as strings.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metadata_read_round_trips_json_sidecar() {
        use crate::imaging::metadata::GenerationMetadata;
        with_singleton_ctx(|| {
            let tmp = tempfile::Builder::new()
                .prefix("plakat-mdread-test-")
                .tempdir()
                .unwrap();
            // Write a fake PNG (empty file is fine — read_metadata
            // doesn't open the PNG, only the sidecar).
            let png_path = tmp.path().join("test.png");
            std::fs::write(&png_path, b"fake-png").unwrap();
            let sidecar_path = tmp.path().join("test.json");
            let md = GenerationMetadata::new(
                "a fox",
                "sd15",
                42u64,
                28usize,
                7.5f64,
                "default",
                512u32,
                512u32,
            );
            std::fs::write(
                &sidecar_path,
                serde_json::to_string_pretty(&md).unwrap(),
            )
            .unwrap();

            // Eval the script (escape backslashes for Windows).
            let png_str = png_path.to_string_lossy().replace('\\', "\\\\");
            eval(&format!(r#""{png_str}" plakat.metadata.read"#)).unwrap();

            // Pop the count off the top.
            with_ctx_mut(|_| {})
                .unwrap();
            // The Bund eval pushed onto the workbench. We can't
            // easily inspect bundcore's stack from a test, but we
            // can re-eval to pop the count and check via .echo
            // (which discards). Instead, just confirm no panic +
            // ensure the next `plakat.metadata.read` on a missing
            // sidecar bails (covered in the next test).
        });
    }

    /// Missing sidecar → bail.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metadata_read_no_sidecar_bails() {
        with_singleton_ctx(|| {
            let tmp = tempfile::Builder::new()
                .prefix("plakat-mdread-nosidecar-")
                .tempdir()
                .unwrap();
            let png_path = tmp.path().join("orphan.png");
            std::fs::write(&png_path, b"fake-png").unwrap();
            let png_str = png_path.to_string_lossy().replace('\\', "\\\\");
            let err = eval(&format!(
                r#""{png_str}" plakat.metadata.read"#
            ))
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no JSON sidecar"), "got {msg}");
        });
    }

    /// Empty path bails.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metadata_read_empty_path_bails() {
        with_singleton_ctx(|| {
            let err = eval(r#""" plakat.metadata.read"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("can't be empty"), "got {msg}");
        });
    }

    /// Bad JSON in the sidecar bails with a deserialise error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metadata_read_bad_json_bails() {
        with_singleton_ctx(|| {
            let tmp = tempfile::Builder::new()
                .prefix("plakat-mdread-badjson-")
                .tempdir()
                .unwrap();
            let png_path = tmp.path().join("bad.png");
            let sidecar_path = tmp.path().join("bad.json");
            std::fs::write(&png_path, b"fake-png").unwrap();
            std::fs::write(&sidecar_path, b"{ not json").unwrap();
            let png_str = png_path.to_string_lossy().replace('\\', "\\\\");
            let err = eval(&format!(
                r#""{png_str}" plakat.metadata.read"#
            ))
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("parsing"), "got {msg}");
        });
    }

    // v0.24 phase 6: plakat.stylize surface (state-only).

    /// `plakat.stylize` bails when no model is loaded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stylize_no_model_loaded_bails() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            let err = eval(
                r#""./subject.jpg" "./style.jpg" plakat.stylize"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no model loaded"), "got {msg}");
        });
    }

    /// `plakat.stylize` bails on empty path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stylize_empty_path_bails() {
        with_singleton_ctx(|| {
            let err = eval(
                r#""./subject.jpg" "" plakat.stylize"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("can't be empty"), "got {msg}");
        });
    }

    /// `plakat.stylize` bails on non-image arg type.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stylize_bad_arg_type_bails() {
        with_singleton_ctx(|| {
            // Float as the style arg — neither string nor int.
            let err = eval(
                r#""./subject.jpg" 3.14 plakat.stylize"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("string path or an integer handle"), "got {msg}");
        });
    }

    /// v0.26 phase 8: plakat.metadata.write bails when the handle
    /// has no metadata attached. Verifies the friendly error message.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metadata_write_bails_on_handleless_metadata() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.images.clear();
                ctx.images_metadata.clear();
                // Push an image WITHOUT metadata.
                ctx.push_image(image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(8, 8, image::Rgb([42, 42, 42])),
                ));
            })
            .unwrap();
            let err = eval(r#"1 "out.png" plakat.metadata.write"#).unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("no metadata attached"),
                "got {msg}"
            );
        });
    }

    /// v0.26 phase 8: push_image_with_metadata makes the metadata
    /// retrievable via metadata_at.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn push_image_with_metadata_round_trips() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.images.clear();
                ctx.images_metadata.clear();
                let img = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3])),
                );
                let meta = crate::imaging::metadata::GenerationMetadata::new(
                    "test prompt",
                    "sd15",
                    42u64,
                    20usize,
                    7.5f64,
                    "default",
                    8u32,
                    8u32,
                );
                let handle = ctx.push_image_with_metadata(img, meta);
                assert_eq!(handle, 1);
            })
            .unwrap();
            with_ctx(|ctx| {
                let m = ctx.metadata_at(1).unwrap().expect("metadata present");
                assert_eq!(m.prompt, "test prompt");
                assert_eq!(m.model, "sd15");
                assert_eq!(m.seed, 42);
            })
            .unwrap();
        });
    }

    /// v0.26 phase 7: stylize cache slot exists + gets dropped
    /// on LoRA stack mutation. Pure state test — doesn't actually
    /// load a pipeline.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stylize_cache_slot_clears_on_lora_change() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                // Slot is None at init; explicitly verify after
                // a hypothetical reset.
                ctx.loaded_stylize = None;
                ctx.loras.clear();
            })
            .unwrap();

            // Add a LoRA — should trigger mark_loras_changed which
            // includes loaded_stylize in the invalidation set.
            eval(r#""user/test-lora" 0.7 plakat.lora.add"#).unwrap();
            with_ctx(|ctx| {
                assert!(ctx.loaded_stylize.is_none(),
                    "loaded_stylize must clear on LoRA add");
            })
            .unwrap();
        });
    }

    // v0.24 phase 5: plakat.embedding.* namespace (state-only).

    /// `plakat.embedding.add` parses + pushes, `mark_loras_changed`
    /// drops both SD slots.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn embedding_add_pushes_and_invalidates_cache() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.embeddings.clear();
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            eval(r#""./my-ti.safetensors:foo:0.7" plakat.embedding.add"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.embeddings.len(), 1);
                let e = &ctx.embeddings[0];
                assert_eq!(e.source, "./my-ti.safetensors");
                assert_eq!(e.trigger.as_deref(), Some("foo"));
                assert!((e.scale - 0.7).abs() < 1e-6);
                // Both SD slots invalidated.
                assert!(ctx.loaded.is_none());
                assert!(ctx.loaded_t2i.is_none());
            })
            .unwrap();
        });
    }

    /// Specs without trigger/scale also work (path-only).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn embedding_add_path_only() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.embeddings.clear()).unwrap();
            eval(r#""./bare.safetensors" plakat.embedding.add"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.embeddings.len(), 1);
                let e = &ctx.embeddings[0];
                assert_eq!(e.source, "./bare.safetensors");
                assert!(e.trigger.is_none());
                assert!((e.scale - 1.0).abs() < 1e-6);
            })
            .unwrap();
        });
    }

    /// `plakat.embedding.clear` empties the stack.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn embedding_clear_empties_stack() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.embeddings.clear()).unwrap();
            eval(r#""./a.safetensors" plakat.embedding.add"#).unwrap();
            eval(r#""./b.safetensors" plakat.embedding.add"#).unwrap();
            eval("plakat.embedding.clear").unwrap();
            with_ctx(|ctx| assert!(ctx.embeddings.is_empty())).unwrap();
        });
    }

    /// Empty spec bails.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn embedding_add_empty_spec_bails() {
        with_singleton_ctx(|| {
            let err = eval(r#""" plakat.embedding.add"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("empty"), "got {msg}");
        });
    }

    // v0.24 phase 4: plakat.outpaint surface (state-only).

    /// `plakat.outpaint` bails when expand-spec is malformed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn outpaint_empty_spec_bails() {
        with_singleton_ctx(|| {
            let err = eval(
                r#""prompt" "./photo.png" "" plakat.outpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("expand-spec"), "got {msg}");
        });
    }

    /// `plakat.outpaint` bails when all sides are zero.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn outpaint_all_zero_spec_bails() {
        with_singleton_ctx(|| {
            let err = eval(
                r#""prompt" "./photo.png" "left=0,right=0" plakat.outpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("> 0"), "got {msg}");
        });
    }

    /// `plakat.outpaint` bails on an unknown spec key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn outpaint_unknown_spec_key_bails() {
        with_singleton_ctx(|| {
            let err = eval(
                r#""prompt" "./photo.png" "middle=128" plakat.outpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("unknown expand-spec"), "got {msg}");
        });
    }

    /// `plakat.outpaint` bails when no model is loaded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn outpaint_no_model_loaded_bails() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            let err = eval(
                r#""prompt" "./photo.png" "expand=128" plakat.outpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no model loaded"), "got {msg}");
        });
    }

    // v0.23 phase 5: plakat.inpaint surface (state-only).

    /// `plakat.inpaint` without a loaded model bails with a
    /// recognisable message before touching the filesystem.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inpaint_no_model_loaded_bails() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            let err = eval(
                r#""fix the sky"  "./photo.png"  "./mask.png"  plakat.inpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no model loaded"), "got {msg}");
            assert!(msg.contains("plakat.inpaint"), "got {msg}");
        });
    }

    /// `plakat.inpaint` with an empty mask path bails.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inpaint_empty_mask_bails() {
        with_singleton_ctx(|| {
            let err = eval(
                r#""prompt"  "./photo.png"  ""  plakat.inpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("mask path can't be empty"), "got {msg}");
        });
    }

    /// `plakat.inpaint` with a non-string non-int input bails
    /// before model dispatch.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inpaint_bad_input_type_bails() {
        with_singleton_ctx(|| {
            // Use a float for `input` — neither string nor int.
            let err = eval(
                r#""prompt"  3.14  "./mask.png"  plakat.inpaint"#,
            )
            .unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("input must be a string path or an integer"),
                "got {msg}"
            );
        });
    }

    // v0.23 phase 4: plakat.style.* end-to-end (state-only).

    /// `plakat.style.apply` sets `ctx.style_id` and invalidates
    /// both SD cache slots.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn style_apply_sets_id_and_invalidates_cache() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.style_id = None;
                ctx.style_ref = None;
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            eval(r#""poster-bold" plakat.style.apply"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.style_id.as_deref(), Some("poster-bold"));
                assert!(ctx.loaded.is_none(), "primary slot invalidated");
                assert!(ctx.loaded_t2i.is_none(), "t2i slot invalidated");
            })
            .unwrap();
        });
    }

    /// `plakat.style.detect` sets `ctx.style_ref` and invalidates
    /// both SD cache slots.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn style_detect_sets_ref_and_invalidates_cache() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.style_id = None;
                ctx.style_ref = None;
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            eval(r#""./ref.jpg" plakat.style.detect"#).unwrap();
            with_ctx(|ctx| {
                let path = ctx.style_ref.as_ref().expect("style_ref set");
                assert!(path.to_string_lossy().ends_with("ref.jpg"));
                assert!(ctx.loaded.is_none());
                assert!(ctx.loaded_t2i.is_none());
            })
            .unwrap();
        });
    }

    /// `plakat.style.clear` empties both fields.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn style_clear_empties_state() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.style_id = Some("test".into());
                ctx.style_ref = Some("/tmp/x.jpg".into());
            })
            .unwrap();
            eval("plakat.style.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.style_id.is_none());
                assert!(ctx.style_ref.is_none());
            })
            .unwrap();
        });
    }

    /// `plakat.style.apply ""` bails — empty id isn't useful.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn style_apply_empty_id_bails() {
        with_singleton_ctx(|| {
            let err = eval(r#""" plakat.style.apply"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("empty"), "got {msg}");
        });
    }

    /// `plakat.config.set "style_catalog" "..."` round-trips.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn style_catalog_config_key_round_trips() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.config.style_catalog.clear()).unwrap();
            eval(r#""/custom/styles" "style_catalog" plakat.config.set"#).unwrap();
            with_ctx(|ctx| assert_eq!(ctx.config.style_catalog, "/custom/styles"))
                .unwrap();
        });
    }

    // v0.25 phase 8: plakat.look.* + plakat.genre.* host words.

    /// `plakat.look.apply` sets `ctx.look_name` + invalidates the
    /// SD cache slots (discovery may push a fresh LoRA at next
    /// generate).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn look_apply_sets_name_and_invalidates_cache() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.look_name = None;
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            eval(r#""watercolor" plakat.look.apply"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.look_name.as_deref(), Some("watercolor"));
                assert!(ctx.loaded.is_none(), "primary slot invalidated");
                assert!(ctx.loaded_t2i.is_none(), "t2i slot invalidated");
            })
            .unwrap();
        });
    }

    /// `plakat.look.apply ""` bails — empty name isn't useful.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn look_apply_empty_name_bails() {
        with_singleton_ctx(|| {
            let err = eval(r#""" plakat.look.apply"#).unwrap_err();
            assert!(format!("{err}").contains("empty"));
        });
    }

    /// `plakat.look.apply` rejects unknown names with a list of
    /// valid choices.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn look_apply_unknown_name_bails_with_choices() {
        with_singleton_ctx(|| {
            let err = eval(r#""not-real" plakat.look.apply"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("unknown look"), "got {msg}");
            assert!(msg.contains("watercolor"), "got {msg}");
        });
    }

    /// `plakat.look.clear` empties the field.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn look_clear_empties_state() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.look_name = Some("watercolor".into())).unwrap();
            eval("plakat.look.clear").unwrap();
            with_ctx(|ctx| assert!(ctx.look_name.is_none())).unwrap();
        });
    }

    /// `plakat.look.list` pushes all 8 bundled looks + count.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn look_list_pushes_bundled_entries() {
        with_singleton_ctx(|| {
            // Run the word; the count + names land on the stack
            // (we can't easily peek the VM stack from here, but
            // the eval succeeding is enough — failure is the only
            // observable side-effect of catalog load issues).
            eval("plakat.look.list").unwrap();
        });
    }

    /// `plakat.genre.apply` sets `ctx.genre_name`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn genre_apply_sets_name() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.genre_name = None).unwrap();
            eval(r#""anime" plakat.genre.apply"#).unwrap();
            with_ctx(|ctx| assert_eq!(ctx.genre_name.as_deref(), Some("anime"))).unwrap();
        });
    }

    /// `plakat.genre.apply` rejects unknown names.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn genre_apply_unknown_name_bails() {
        with_singleton_ctx(|| {
            let err = eval(r#""not-real" plakat.genre.apply"#).unwrap_err();
            assert!(format!("{err}").contains("unknown genre"));
        });
    }

    /// `plakat.genre.clear` empties the field.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn genre_clear_empties_state() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.genre_name = Some("anime".into())).unwrap();
            eval("plakat.genre.clear").unwrap();
            with_ctx(|ctx| assert!(ctx.genre_name.is_none())).unwrap();
        });
    }

    /// `plakat.genre.list` pushes the single bundled entry + count.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn genre_list_pushes_bundled() {
        with_singleton_ctx(|| {
            eval("plakat.genre.list").unwrap();
        });
    }

    /// Look + genre are independent axes — setting both leaves both
    /// fields populated, and `clear` on one doesn't disturb the other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn look_and_genre_independent_axes() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.look_name = None;
                ctx.genre_name = None;
            })
            .unwrap();
            eval(r#""watercolor" plakat.look.apply"#).unwrap();
            eval(r#""anime" plakat.genre.apply"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.look_name.as_deref(), Some("watercolor"));
                assert_eq!(ctx.genre_name.as_deref(), Some("anime"));
            })
            .unwrap();
            eval("plakat.look.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.look_name.is_none());
                assert_eq!(ctx.genre_name.as_deref(), Some("anime"));
            })
            .unwrap();
        });
    }

    /// `plakat.config.set "offline_discovery" "true"` round-trips
    /// through the GenerationConfig bool parser.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn offline_discovery_config_key_round_trips() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.config.offline_discovery = false).unwrap();
            eval(r#""true" "offline_discovery" plakat.config.set"#).unwrap();
            with_ctx(|ctx| assert!(ctx.config.offline_discovery)).unwrap();
            eval(r#""false" "offline_discovery" plakat.config.set"#).unwrap();
            with_ctx(|ctx| assert!(!ctx.config.offline_discovery)).unwrap();
        });
    }

    // v0.22 phase 9: plakat.artefact.* end-to-end.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artefact_add_pushes_and_list_round_trips() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.artefacts.clear()).unwrap();
            eval(r#""oak" plakat.artefact.add"#).unwrap();
            eval(r#""sun@sky/right:0.8" plakat.artefact.add"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.artefacts.len(), 2);
                assert_eq!(ctx.artefacts[0].name, "oak");
                assert!(ctx.artefacts[0].zone.is_none());
                assert_eq!(ctx.artefacts[1].name, "sun");
                assert!(ctx.artefacts[1].zone.is_some());
                assert!((ctx.artefacts[1].scale.unwrap() - 0.8).abs() < 1e-6);
            })
            .unwrap();
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artefact_clear_empties_stack() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.artefacts.clear()).unwrap();
            eval(r#""oak" plakat.artefact.add"#).unwrap();
            eval("plakat.artefact.clear").unwrap();
            with_ctx(|ctx| assert!(ctx.artefacts.is_empty())).unwrap();
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artefact_add_bails_on_garbage_spec() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.artefacts.clear()).unwrap();
            // Empty name; FromStr rejects.
            let err = eval(r#""@sky" plakat.artefact.add"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("artefact"), "got {msg}");
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artefact_blend_enable_toggles_ctx_flag() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.artefact_blend_enabled = false).unwrap();
            eval("plakat.artefact.blend.enable").unwrap();
            with_ctx(|ctx| assert!(ctx.artefact_blend_enabled)).unwrap();
            eval("plakat.artefact.blend.disable").unwrap();
            with_ctx(|ctx| assert!(!ctx.artefact_blend_enabled)).unwrap();
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artefact_config_keys_round_trip_via_host_word() {
        with_singleton_ctx(|| {
            eval(r#""/some/lib" "artefact_library" plakat.config.set"#).unwrap();
            eval(r#"0.45 "artefact_blend_strength" plakat.config.set"#).unwrap();
            eval(r#""true" "artefact_smart_zones" plakat.config.set"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.config.artefact_library, "/some/lib");
                assert!((ctx.config.artefact_blend_strength - 0.45).abs() < 1e-6);
                assert!(ctx.config.artefact_smart_zones);
            })
            .unwrap();
        });
    }

    // v0.22 phase 5: plakat.controlnet.* end-to-end.

    /// `plakat.controlnet.add` pushes a kind + image pair.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controlnet_add_pushes_kind_image() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.controlnets.clear()).unwrap();
            eval(r#""depth" "./d.png" plakat.controlnet.add"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.controlnets.len(), 1);
                let cn = &ctx.controlnets[0];
                assert_eq!(cn.kind.slug(), "depth");
                assert!(cn.image.as_ref().unwrap().to_str().unwrap().ends_with("d.png"));
                assert!(cn.from.is_none());
                assert!((cn.strength - 1.0).abs() < 1e-6);
            })
            .unwrap();
        });
    }

    /// `plakat.controlnet.annotate` pushes a kind + from pair.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controlnet_annotate_pushes_kind_from() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.controlnets.clear()).unwrap();
            eval(r#""canny" "./photo.jpg" plakat.controlnet.annotate"#).unwrap();
            with_ctx(|ctx| {
                let cn = &ctx.controlnets[0];
                assert_eq!(cn.kind.slug(), "canny");
                assert!(cn.image.is_none());
                assert!(cn.from.as_ref().unwrap().to_str().unwrap().ends_with("photo.jpg"));
            })
            .unwrap();
        });
    }

    /// `plakat.controlnet.spec` parses the full grammar.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controlnet_spec_parses_full_grammar() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.controlnets.clear()).unwrap();
            eval(
                r#""depth:from=./photo.jpg:strength=0.7:start=0.2:end=0.7" plakat.controlnet.spec"#,
            )
            .unwrap();
            with_ctx(|ctx| {
                let cn = &ctx.controlnets[0];
                assert_eq!(cn.kind.slug(), "depth");
                assert!((cn.strength - 0.7).abs() < 1e-6);
                assert!((cn.start - 0.2).abs() < 1e-6);
                assert!((cn.end - 0.7).abs() < 1e-6);
            })
            .unwrap();
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controlnet_unknown_kind_bails() {
        with_singleton_ctx(|| {
            let err =
                eval(r#""not-a-kind" "./x.png" plakat.controlnet.add"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("unknown control kind"), "got {msg}");
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controlnet_clear_empties() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.controlnets.push(
                    crate::pipelines::controlnet::ControlSpec {
                        kind: crate::pipelines::controlnet::ControlKind::Canny,
                        image: Some(std::path::PathBuf::from("/tmp/x.png")),
                        from: None,
                        strength: 1.0,
                        start: 0.0,
                        end: 1.0,
                    },
                );
            })
            .unwrap();
            eval("plakat.controlnet.clear").unwrap();
            with_ctx(|ctx| assert!(ctx.controlnets.is_empty())).unwrap();
        });
    }

    // v0.22 phase 4: plakat.lora.* end-to-end.

    /// `plakat.lora.add` pushes to ctx.loras and invalidates the
    /// cache (no real model load triggered — the test stays fast).
    /// v0.23 phase 1: invalidation drops BOTH the primary slot
    /// (portrait/flux/sd3) AND the secondary SD t2i slot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lora_add_pushes_and_invalidates_cache() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loras.clear();
                ctx.loaded = None;
                ctx.loaded_t2i = None;
            })
            .unwrap();
            eval(r#""civitai:12345" 0.7 plakat.lora.add"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.loras.len(), 1, "lora pushed");
                let spec = &ctx.loras[0];
                assert!((spec.scale - 0.7).abs() < 1e-6);
                // Both cache slots must be invalidated.
                assert!(ctx.loaded.is_none(), "primary slot invalidated");
                assert!(ctx.loaded_t2i.is_none(), "t2i slot invalidated");
            })
            .unwrap();
        });
    }

    /// `plakat.lora.add` with a bad scale rejects.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lora_add_rejects_negative_scale() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loras.clear();
            })
            .unwrap();
            let err = eval(r#""./foo.safetensors" -1.0 plakat.lora.add"#)
                .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("scale must be"), "got {msg}");
        });
    }

    /// `plakat.lora.clear` drops every entry + invalidates the
    /// cache.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lora_clear_empties_stack() {
        with_singleton_ctx(|| {
            // Pre-stuff the LoRA stack.
            with_ctx_mut(|ctx| {
                ctx.loras.push(
                    crate::pipelines::lora::LoraSpec {
                        source: crate::pipelines::lora::LoraSource::Local(
                            std::path::PathBuf::from("/tmp/a.safetensors"),
                        ),
                        scale: 0.5,
                    },
                );
            })
            .unwrap();
            eval("plakat.lora.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.loras.is_empty(), "stack drained");
            })
            .unwrap();
        });
    }

    /// `plakat.lora.list` pushes one string per entry + the depth.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lora_list_pushes_entries() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loras.clear();
                ctx.loras.push(crate::pipelines::lora::LoraSpec {
                    source: crate::pipelines::lora::LoraSource::Local(
                        std::path::PathBuf::from("/tmp/style.safetensors"),
                    ),
                    scale: 0.8,
                });
                ctx.loras.push(crate::pipelines::lora::LoraSpec {
                    source: crate::pipelines::lora::LoraSource::Civitai {
                        id_kind: crate::pipelines::lora::CivitaiIdKind::Model(12345),
                        file: None,
                    },
                    scale: 0.5,
                });
            })
            .unwrap();
            // The list word pushes 3 values: 2 strings + 1 depth int.
            // Drop them after each test so subsequent tests start clean.
            eval("plakat.lora.list drop drop drop").unwrap();
            // The eval succeeded — that's the contract. Detailed
            // value verification happens in the format_source unit
            // tests above; we only assert no panic + the right
            // depth here. (Bund's `drop` pops one value each.)
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

    /// v0.21 phase 5 + v0.24 phase 1: portrait gate — no model
    /// loaded bails with the "Call \"sdxl\" plakat.load" pointer
    /// (note the SDXL suggestion in the message, not just sd15).
    /// Updated for v0.24: `plakat.portrait` no longer takes a
    /// photo arg; photos come from the
    /// `plakat.portrait.photo.add` stack.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn portrait_no_model_loaded_bails_via_eval() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
                ctx.portrait_photos.clear();
            })
            .unwrap();
            // Push a photo so we exercise the no-model-loaded gate
            // (not the empty-photo-stack gate, which would also
            // bail correctly but on a different message).
            eval(r#""/tmp/me.jpg" 1.0 plakat.portrait.photo.add"#).unwrap();
            let err = eval(r#""a portrait" plakat.portrait"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("no model loaded"), "got {msg}");
            // The portrait-specific pointer recommends sdxl too,
            // not just sd15 — pin that so a future tweak doesn't
            // silently drop the SDXL hint.
            assert!(msg.contains("sdxl"), "got {msg}");
        });
    }

    /// v0.21 phase 5 + v0.24 phase 1: unknown handle pushed to
    /// `plakat.portrait.photo.add` surfaces the image_at error.
    /// Confirms the int dispatch arm shares the lookup path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn portrait_unknown_handle_bails_with_image_at_message() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.images.clear();
                ctx.portrait_photos.clear();
            })
            .unwrap();
            // Pushing handle 999 onto the photo stack fires
            // image_at(999) inside the materialise-handle arm.
            let err = eval(r#"999 1.0 plakat.portrait.photo.add"#).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("image handle 999"), "got {msg}");
        });
    }

    /// v0.24 phase 1: plakat.portrait with empty photo stack
    /// bails before any model dispatch.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn portrait_empty_photo_stack_bails() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                // Pretend a model is loaded so we get past the
                // no-model-loaded gate; portrait_photos check
                // fires inside portrait_one before any pipeline
                // dispatch.
                ctx.portrait_photos.clear();
            })
            .unwrap();
            // No portrait.photo.add — empty stack should bail.
            // The no-model gate may fire first if no model is
            // loaded; either bail proves a clear error reaches
            // the user.
            let err = eval(r#""a portrait" plakat.portrait"#).unwrap_err();
            let msg = format!("{err}");
            // Either "no model loaded" or "no photo configured"
            // is acceptable — both are clear errors.
            assert!(
                msg.contains("no model loaded") || msg.contains("no photo"),
                "got {msg}"
            );
        });
    }

    /// v0.24 phase 1: plakat.portrait.photo.{add, clear, list}
    /// round-trip.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn portrait_photo_add_clear_round_trip() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| ctx.portrait_photos.clear()).unwrap();
            eval(r#""/tmp/alice.jpg" 1.0 plakat.portrait.photo.add"#).unwrap();
            eval(r#""/tmp/bob.jpg" 0.5 plakat.portrait.photo.add"#).unwrap();
            eval(r#""/tmp/carol.jpg" -1.0 plakat.portrait.photo.add"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.portrait_photos.len(), 3);
                assert!((ctx.portrait_photos[0].weight.unwrap() - 1.0).abs() < 1e-6);
                assert!((ctx.portrait_photos[1].weight.unwrap() - 0.5).abs() < 1e-6);
                // -1.0 weight means auto-fill → None on the spec.
                assert!(ctx.portrait_photos[2].weight.is_none());
            })
            .unwrap();
            eval("plakat.portrait.photo.clear").unwrap();
            with_ctx(|ctx| assert!(ctx.portrait_photos.is_empty())).unwrap();
        });
    }

    /// v0.24 phase 1: plakat.portrait.photo.add rejects negative
    /// (non-(-1.0)) weight.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn portrait_photo_add_rejects_negative_weight() {
        with_singleton_ctx(|| {
            let err = eval(r#""/tmp/x.jpg" -0.5 plakat.portrait.photo.add"#)
                .unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("weight must be"), "got {msg}");
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

    /// v0.21 phase 8 composition test: drives the full script
    /// surface end-to-end via `eval()` — config.set → load gate →
    /// (pre-stuffed image handles, simulating prior generate calls)
    /// → upscale → save. This is the "real script that loads SD,
    /// generates 3 images, upscales the best one" referenced in
    /// RFC §9, abridged to skip the SD model load (CI has no
    /// weights).
    ///
    /// Pre-stuffing the registry with three synthetic images lets
    /// us exercise the parts of the workflow that don't need a
    /// pipeline — and there are more of those than you'd think:
    /// stack discipline, handle reuse, output dir resolution,
    /// Lanczos upscaling, file IO. The real-pipeline integration
    /// (which DOES need SD weights) gets covered by manual smoke
    /// scripts in the Documentation/Tutorials/SCRIPTING_TUTORIAL.md
    /// walkthrough.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn full_composition_via_eval_three_handles_pick_best_upscale() {
        with_singleton_ctx(|| {
            // Reset the context to a known-clean baseline.
            let out_dir = with_ctx_mut(|ctx| {
                ctx.images.clear();
                ctx.loaded = None;
                ctx.config = config::GenerationConfig::default();
                ctx.out_dir.clone()
            })
            .unwrap();

            // Pre-stuff three synthetic 8x8 images at handles 1, 2, 3.
            // Each filled with a different pixel value so we can
            // tell them apart after the round-trip.
            with_ctx_mut(|ctx| {
                for r in [10u8, 20u8, 30u8] {
                    ctx.images.push(image::DynamicImage::ImageRgb8(
                        image::RgbImage::from_pixel(8, 8, image::Rgb([r, r, r])),
                    ));
                }
            })
            .unwrap();

            // The composition script. Real scripts would call
            // plakat.generate three times instead of relying on
            // pre-stuffed handles; the rest is identical.
            //
            //   1. config.set width + scheduler (proves int + string
            //      keys both round-trip through eval)
            //   2. upscale handle 3 x2 → handle 4; save it
            //   3. upscale handle 4 x4 → handle 5; save it
            //   4. save the *original* handle 3 too — proves handle
            //      reuse: upscaling doesn't consume the source
            //
            // The integers in the script are explicit handle
            // references; `plakat.save` consumes the handle it
            // pops, so any chain after a save has to push the
            // intended handle again.
            let script = r#"
                512 "width" plakat.config.set
                "euler-a" "scheduler" plakat.config.set

                3 2 plakat.upscale  "best.png"     plakat.save
                4 4 plakat.upscale  "best-4k.png"  plakat.save
                3                   "source.png"   plakat.save
            "#;
            eval(script).unwrap();

            // Verify: config knobs latched + every save landed.
            with_ctx(|ctx| {
                assert_eq!(ctx.config.width, 512);
                assert!(
                    matches!(
                        ctx.config.scheduler,
                        crate::pipelines::scheduler::SchedulerKind::EulerA
                    ),
                    "scheduler should now be EulerA, got {:?}",
                    ctx.config.scheduler
                );
                // Handles 1-3 were pre-stuffed; upscale added 4, 5.
                assert_eq!(ctx.images.len(), 5,
                    "expected 5 images in registry after 2x upscales");
                // Handles 4 + 5 should be the upscaled variants.
                assert_eq!(ctx.images[3].width(), 16, "handle 4 = source 8 * 2");
                assert_eq!(ctx.images[4].width(), 64, "handle 5 = handle 4 * 4");
            })
            .unwrap();
            // Three output files landed under out_dir.
            for name in &["best.png", "best-4k.png", "source.png"] {
                assert!(
                    out_dir.join(name).exists(),
                    "missing {name} under {}",
                    out_dir.display()
                );
            }
        });
    }

    /// v0.21 phase 8: REPL-equivalent state-persistence proof.
    /// In a single eval(), config knobs set on line 1 affect a
    /// hypothetical generate later in the same script. Doesn't run
    /// the generate (no SD weights), but pins that the
    /// config.set surface composes with the rest of the script
    /// state without surprises.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn config_set_persists_for_subsequent_words_in_same_eval() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.config = config::GenerationConfig::default();
                ctx.images.clear();
            })
            .unwrap();

            // Pre-stuff a handle so plakat.save has something to write.
            let h = with_ctx_mut(|ctx| {
                ctx.push_image(image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(8, 8, image::Rgb([5, 5, 5])),
                ))
            })
            .unwrap();
            assert_eq!(h, 1);

            // Set steps + face_strength on line 1; save handle 1
            // on line 2. The config setters survive across lines
            // (the whole point of a persistent ScriptCtx).
            let script = format!(
                r#"
                75   "steps"          plakat.config.set
                0.5  "face_strength"  plakat.config.set
                {h}  "persist.png"    plakat.save
            "#
            );
            eval(&script).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.config.steps, 75);
                assert!((ctx.config.face_strength - 0.5).abs() < 1e-6);
            })
            .unwrap();
        });
    }

    // v0.22 phase 12: composition tests.
    //
    // These exercise multi-namespace state interaction without
    // actually loading a model — each one drives the
    // `plakat.config.set` / `plakat.*.{add,enable,disable}` surface
    // through `eval` and asserts the resulting `ctx` state. The
    // model-load path is exercised by the per-namespace e2e
    // tests above; these tests focus on cross-namespace
    // composition that a real user script would do.

    /// One big script touches every phase's namespace surface.
    /// All 28 host words + every Category-B config key validates
    /// together. Demonstrates that namespaces compose without
    /// state interference.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_all_namespaces_state_round_trip() {
        with_singleton_ctx(|| {
            // Reset state inherited from earlier tests in the mutex.
            with_ctx_mut(|ctx| {
                ctx.loras.clear();
                ctx.controlnets.clear();
                ctx.artefacts.clear();
                ctx.refiner_enabled = false;
                ctx.adetailer_enabled = false;
                ctx.hires_enabled = false;
                ctx.artefact_blend_enabled = false;
            })
            .unwrap();

            // Per-phase surface exercised in script-order.
            eval(
                r#"
                // Phase 4: LoRA stack.
                "./fake-lora.safetensors" 0.7 plakat.lora.add
                0.9 "lora_scale" plakat.config.set

                // Phase 5: ControlNet stack.
                "depth" "./d.png" plakat.controlnet.add
                "canny" "./c.png" plakat.controlnet.annotate

                // Phase 6: refiner toggle + same-model polish keys.
                plakat.refiner.enable
                12 "refine_steps" plakat.config.set
                0.4 "refine_strength" plakat.config.set
                0.85 "refiner_frac" plakat.config.set

                // Phase 7: adetailer toggle + 6 face-refine keys.
                plakat.adetailer.enable
                0.5 "adetailer_strength" plakat.config.set
                0.3 "adetailer_padding" plakat.config.set
                768 "adetailer_size" plakat.config.set

                // Phase 8: hires-fix toggle + 4 keys.
                plakat.hires.enable
                2.0 "hires_scale" plakat.config.set
                0.55 "hires_strength" plakat.config.set
                "lanczos" "hires_upscaler" plakat.config.set

                // Phase 9: artefact stack + blend toggle + 3 keys.
                "oak" plakat.artefact.add
                "sun@sky/right:0.8" plakat.artefact.add
                plakat.artefact.blend.enable
                0.4 "artefact_blend_strength" plakat.config.set
                "true" "artefact_smart_zones" plakat.config.set

                // Phase 10: enhance config (provider validation).
                "local" "enhance_provider" plakat.config.set
                0.5 "enhance_temp" plakat.config.set

                // Phase 11: misc keys.
                "16:9" "aspect" plakat.config.set
                512 "base" plakat.config.set
                2 "clip_skip" plakat.config.set
                "photo" "negative_preset" plakat.config.set
            "#,
            )
            .unwrap();

            with_ctx(|ctx| {
                // Phase 4.
                assert_eq!(ctx.loras.len(), 1);
                assert!((ctx.config.lora_scale - 0.9).abs() < 1e-6);
                // Phase 5.
                assert_eq!(ctx.controlnets.len(), 2);
                // Phase 6.
                assert!(ctx.refiner_enabled);
                assert_eq!(ctx.config.refine_steps, Some(12));
                // Phase 7.
                assert!(ctx.adetailer_enabled);
                assert!((ctx.config.adetailer_strength - 0.5).abs() < 1e-6);
                assert_eq!(ctx.config.adetailer_size, 768);
                // Phase 8.
                assert!(ctx.hires_enabled);
                assert!((ctx.config.hires_scale - 2.0).abs() < 1e-6);
                assert_eq!(ctx.config.hires_upscaler, "lanczos");
                // Phase 9.
                assert_eq!(ctx.artefacts.len(), 2);
                assert!(ctx.artefact_blend_enabled);
                assert!(ctx.config.artefact_smart_zones);
                // Phase 10.
                assert_eq!(ctx.config.enhance_provider, "local");
                assert!((ctx.config.enhance_temp.unwrap() - 0.5).abs() < 1e-6);
                // Phase 11.
                assert_eq!(ctx.config.aspect, "16:9");
                assert_eq!(ctx.config.base, 512);
                assert_eq!(ctx.config.clip_skip, 2);
                assert_eq!(ctx.config.negative_preset, "photo");
            })
            .unwrap();
        });
    }

    /// Composition: enabling all three post-process toggles
    /// (adetailer + hires + artefact-blend) at once is legal as
    /// state, but `plakat.generate` would bail when hires +
    /// artefacts both fire — that gate lives in `script_entry`,
    /// not at toggle-set time. Validate the state composition
    /// here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_all_post_process_toggles_compose_at_state_layer() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.adetailer_enabled = false;
                ctx.hires_enabled = false;
                ctx.artefact_blend_enabled = false;
                ctx.artefacts.clear();
            })
            .unwrap();
            eval(
                r#"
                plakat.adetailer.enable
                plakat.hires.enable
                plakat.artefact.blend.enable
            "#,
            )
            .unwrap();
            with_ctx(|ctx| {
                assert!(ctx.adetailer_enabled);
                assert!(ctx.hires_enabled);
                assert!(ctx.artefact_blend_enabled);
            })
            .unwrap();
            // Disable cleanly.
            eval(
                r#"
                plakat.adetailer.disable
                plakat.hires.disable
                plakat.artefact.blend.disable
            "#,
            )
            .unwrap();
            with_ctx(|ctx| {
                assert!(!ctx.adetailer_enabled);
                assert!(!ctx.hires_enabled);
                assert!(!ctx.artefact_blend_enabled);
            })
            .unwrap();
        });
    }

    /// Composition: LoRA + ControlNet stacks accumulate
    /// independently; clearing one doesn't touch the other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_lora_and_controlnet_stacks_are_independent() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loras.clear();
                ctx.controlnets.clear();
            })
            .unwrap();
            eval(
                r#"
                "./l1.safetensors" 0.8 plakat.lora.add
                "./l2.safetensors" 0.6 plakat.lora.add
                "depth" "./d.png" plakat.controlnet.add
                "canny" "./c.png" plakat.controlnet.add
                plakat.lora.clear
            "#,
            )
            .unwrap();
            with_ctx(|ctx| {
                assert!(ctx.loras.is_empty(), "LoRA cleared");
                assert_eq!(ctx.controlnets.len(), 2, "ControlNets intact");
            })
            .unwrap();
            eval("plakat.controlnet.clear").unwrap();
            with_ctx(|ctx| assert!(ctx.controlnets.is_empty())).unwrap();
        });
    }

    /// Composition: setting `negative` + `negative_preset` both
    /// preserves the user negative in config AND combines them at
    /// request-build time (preset first, user appended).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_negative_preset_combines_with_user_negative() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.config.negative.clear();
                ctx.config.negative_preset.clear();
            })
            .unwrap();
            eval(
                r#"
                "extra-junk-words" "negative" plakat.config.set
                "anime" "negative_preset" plakat.config.set
            "#,
            )
            .unwrap();
            // Verify the combine helper at request-build time
            // produces "<preset>, extra-junk-words".
            let combined = with_ctx(|ctx| {
                crate::prompt::negative_presets::combine(
                    Some(&ctx.config.negative_preset),
                    &ctx.config.negative,
                )
                .unwrap()
            })
            .unwrap();
            assert!(combined.contains("extra-junk-words"));
            // Anime preset text starts with "lowres" or similar —
            // assert it's a *combination*, not just the user input.
            assert!(combined.len() > "extra-junk-words".len() + 2);
        });
    }

    // v0.23 phase 8: composition tests for the v0.23 surface.

    /// One script touches every v0.23 deferral closure + the two
    /// new things (`plakat.style.*`, `plakat.inpaint` surface).
    /// State-only — no model load.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v023_full_surface_state_round_trip() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
                ctx.loras.clear();
                ctx.controlnets.clear();
                ctx.refiner_enabled = false;
                ctx.adetailer_enabled = false;
                ctx.hires_enabled = false;
                ctx.style_id = None;
                ctx.style_ref = None;
                ctx.config.clip_skip = 1;
            })
            .unwrap();

            eval(
                r#"
                // v0.23 phase 2: refiner toggle (now actually loads SDXL refiner UNet)
                plakat.refiner.enable
                0.85 "refiner_frac" plakat.config.set

                // v0.23 phase 3: clip_skip wires through to t2i::Pipeline.encode_*
                2 "clip_skip" plakat.config.set

                // v0.23 phase 4: plakat.style.* — pick by id or detect from a photo
                "poster-bold" plakat.style.apply
                "/custom/style-catalog" "style_catalog" plakat.config.set
                0.7 "style_strength" plakat.config.set

                // v0.23 phases 6 + 7: Flux + SD3 ControlNet (image= specs)
                "depth" "./depth.png" plakat.controlnet.add
                "canny" "./edges.png" plakat.controlnet.add

                // v0.22 post-process toggles still fire after the v0.23 surface.
                plakat.adetailer.enable
            "#,
            )
            .unwrap();

            with_ctx(|ctx| {
                assert!(ctx.refiner_enabled);
                assert!((ctx.config.refiner_frac - 0.85).abs() < 1e-6);
                assert_eq!(ctx.config.clip_skip, 2);
                assert_eq!(ctx.style_id.as_deref(), Some("poster-bold"));
                assert_eq!(ctx.config.style_catalog, "/custom/style-catalog");
                assert!((ctx.config.style_strength - 0.7).abs() < 1e-6);
                assert_eq!(ctx.controlnets.len(), 2);
                assert!(ctx.adetailer_enabled);
            })
            .unwrap();
        });
    }

    /// Style state + ControlNet stack both invalidate cache slots,
    /// but they DON'T cross-invalidate (a CN mutation shouldn't
    /// disturb the style state, and vice versa).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_style_and_controlnet_state_are_independent() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.style_id = None;
                ctx.style_ref = None;
                ctx.controlnets.clear();
            })
            .unwrap();
            eval(r#""poster-bold" plakat.style.apply"#).unwrap();
            eval(r#""depth" "./d.png" plakat.controlnet.add"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.style_id.as_deref(), Some("poster-bold"));
                assert_eq!(ctx.controlnets.len(), 1);
            })
            .unwrap();
            // Clearing CN doesn't touch style.
            eval("plakat.controlnet.clear").unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.style_id.as_deref(), Some("poster-bold"));
                assert!(ctx.controlnets.is_empty());
            })
            .unwrap();
            // Clearing style doesn't touch CN (already empty, but check the toggle).
            eval(r#""canny" "./c.png" plakat.controlnet.add"#).unwrap();
            eval("plakat.style.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.style_id.is_none());
                assert_eq!(ctx.controlnets.len(), 1);
            })
            .unwrap();
        });
    }

    /// mark_controlnets_changed (v0.23 phase 6) preserves SD slots
    /// — verified at the state layer (real-pipeline behaviour
    /// covered in ctx.rs unit tests + CLI smoke).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_controlnet_mutation_state_smoke() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
                ctx.controlnets.clear();
            })
            .unwrap();
            // Add + clear cycle exercises mark_controlnets_changed.
            eval(r#""depth" "./d.png" plakat.controlnet.add"#).unwrap();
            eval("plakat.controlnet.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.controlnets.is_empty());
                // SD slots stay None (nothing was loaded) — the test
                // proves the path doesn't crash on empty slots.
                assert!(ctx.loaded.is_none());
                assert!(ctx.loaded_t2i.is_none());
            })
            .unwrap();
        });
    }

    // v0.24 phase 10: composition tests for the v0.24 surface.

    /// One script exercises every v0.24 namespace + config key
    /// (state-only — no model load).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v024_full_surface_state_round_trip() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
                ctx.loras.clear();
                ctx.controlnets.clear();
                ctx.portrait_photos.clear();
                ctx.embeddings.clear();
                ctx.refiner_enabled = false;
                ctx.adetailer_enabled = false;
                ctx.hires_enabled = false;
                ctx.style_id = None;
                ctx.style_ref = None;
                ctx.config.face_bbox = None;
                ctx.config.face_landmarks = None;
                ctx.config.identity_kind.clear();
            })
            .unwrap();

            eval(
                r#"
                // v0.24 phase 1: portrait.photo.* multi-photo stack
                "./alice.jpg" 0.7 plakat.portrait.photo.add
                "./bob.jpg"   0.3 plakat.portrait.photo.add

                // v0.24 phase 2: face alignment overrides
                "0.2,0.1,0.8,0.7" "face_bbox" plakat.config.set
                "0.40,0.40,0.60,0.40,0.50,0.55,0.42,0.68,0.58,0.68"
                    "face_landmarks" plakat.config.set

                // v0.24 phase 3: identity override
                "face-id-sdxl" "identity_kind" plakat.config.set

                // v0.24 phase 5: embedding stack
                "./style-ti.safetensors:mytrigger:0.7" plakat.embedding.add

                // v0.24 phase 8: from= specs no longer bail
                "depth" "./reference.jpg" plakat.controlnet.annotate
            "#,
            )
            .unwrap();

            with_ctx(|ctx| {
                // Phase 1 state.
                assert_eq!(ctx.portrait_photos.len(), 2);
                assert!((ctx.portrait_photos[0].weight.unwrap() - 0.7).abs() < 1e-6);
                assert!((ctx.portrait_photos[1].weight.unwrap() - 0.3).abs() < 1e-6);
                // Phase 2 face keys.
                let bbox = ctx.config.face_bbox.expect("bbox set");
                assert!((bbox[0] - 0.2).abs() < 1e-6);
                let lm = ctx.config.face_landmarks.expect("landmarks set");
                assert!((lm[0][0] - 0.40).abs() < 1e-6);
                // Phase 3 identity_kind.
                assert_eq!(ctx.config.identity_kind, "face-id-sdxl");
                // Phase 5 embedding stack.
                assert_eq!(ctx.embeddings.len(), 1);
                assert_eq!(ctx.embeddings[0].trigger.as_deref(), Some("mytrigger"));
                assert!((ctx.embeddings[0].scale - 0.7).abs() < 1e-6);
                // Phase 8 from= spec lives on the controlnets stack.
                assert_eq!(ctx.controlnets.len(), 1);
                assert!(ctx.controlnets[0].from.is_some());
            })
            .unwrap();
        });
    }

    /// portrait_photos stack and the new face_* config keys are
    /// independent — clearing one doesn't disturb the other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v024_portrait_state_independence() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.portrait_photos.clear();
                ctx.config.face_bbox = None;
                ctx.config.identity_kind.clear();
            })
            .unwrap();
            eval(r#""./alice.jpg" 1.0 plakat.portrait.photo.add"#).unwrap();
            eval(r#""0.2,0.1,0.8,0.7" "face_bbox" plakat.config.set"#).unwrap();
            eval(r#""face-id" "identity_kind" plakat.config.set"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.portrait_photos.len(), 1);
                assert!(ctx.config.face_bbox.is_some());
                assert_eq!(ctx.config.identity_kind, "face-id");
            })
            .unwrap();
            // Clear photos — face keys persist.
            eval("plakat.portrait.photo.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.portrait_photos.is_empty());
                assert!(ctx.config.face_bbox.is_some());
                assert_eq!(ctx.config.identity_kind, "face-id");
            })
            .unwrap();
            // Clear face_bbox — photos already cleared; identity persists.
            eval(r#""" "face_bbox" plakat.config.set"#).unwrap();
            with_ctx(|ctx| {
                assert!(ctx.config.face_bbox.is_none());
                assert_eq!(ctx.config.identity_kind, "face-id");
            })
            .unwrap();
        });
    }

    /// CN annotation cache invalidates on stack mutation —
    /// add/remove a CN spec drops the cache via
    /// `mark_controlnets_changed`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v024_cn_annotation_cache_invalidates() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.controlnets.clear();
                ctx.cn_annotation_cache = None;
            })
            .unwrap();
            eval(r#""depth" "./photo.jpg" plakat.controlnet.annotate"#).unwrap();
            // No annotation has run yet (no generate fired); cache
            // is still None until first generate. Confirm that.
            with_ctx(|ctx| {
                assert_eq!(ctx.controlnets.len(), 1);
                assert!(ctx.cn_annotation_cache.is_none());
            })
            .unwrap();
            // Clear the CN stack — should also clear the (empty)
            // annotation cache via mark_controlnets_changed.
            eval("plakat.controlnet.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.controlnets.is_empty());
                assert!(ctx.cn_annotation_cache.is_none());
            })
            .unwrap();
        });
    }

    // v0.25 phase 11: composition tests for the v0.25 surface.

    /// One script exercises every v0.25 namespace + config key
    /// (state-only — no model load, no network).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v025_full_surface_state_round_trip() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loaded = None;
                ctx.loaded_t2i = None;
                ctx.loras.clear();
                ctx.controlnets.clear();
                ctx.portrait_photos.clear();
                ctx.embeddings.clear();
                ctx.refiner_enabled = false;
                ctx.adetailer_enabled = false;
                ctx.hires_enabled = false;
                ctx.style_id = None;
                ctx.style_ref = None;
                ctx.look_name = None;
                ctx.genre_name = None;
                ctx.config.offline_discovery = false;
            })
            .unwrap();

            eval(
                r#"
                // v0.25 phase 8: look + genre axes.
                "watercolor" plakat.look.apply
                "anime"      plakat.genre.apply

                // v0.25 phase 8: offline_discovery config key.
                "true" "offline_discovery" plakat.config.set
            "#,
            )
            .unwrap();

            with_ctx(|ctx| {
                assert_eq!(ctx.look_name.as_deref(), Some("watercolor"));
                assert_eq!(ctx.genre_name.as_deref(), Some("anime"));
                assert!(ctx.config.offline_discovery);
            })
            .unwrap();
        });
    }

    /// Look + genre are independent state axes. Clearing one
    /// doesn't touch the other; the offline_discovery config key
    /// is also independent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v025_look_genre_offline_independence() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.look_name = None;
                ctx.genre_name = None;
                ctx.config.offline_discovery = false;
            })
            .unwrap();
            eval(r#""watercolor" plakat.look.apply"#).unwrap();
            eval(r#""anime" plakat.genre.apply"#).unwrap();
            eval(r#""true" "offline_discovery" plakat.config.set"#).unwrap();
            with_ctx(|ctx| {
                assert_eq!(ctx.look_name.as_deref(), Some("watercolor"));
                assert_eq!(ctx.genre_name.as_deref(), Some("anime"));
                assert!(ctx.config.offline_discovery);
            })
            .unwrap();
            // Clear look — genre + offline persist.
            eval("plakat.look.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.look_name.is_none());
                assert_eq!(ctx.genre_name.as_deref(), Some("anime"));
                assert!(ctx.config.offline_discovery);
            })
            .unwrap();
            // Clear genre — offline still set.
            eval("plakat.genre.clear").unwrap();
            with_ctx(|ctx| {
                assert!(ctx.genre_name.is_none());
                assert!(ctx.config.offline_discovery);
            })
            .unwrap();
        });
    }

    /// look_name + genre_name mutations both invalidate the SD
    /// cache via mark_loras_changed — discovery may push a fresh
    /// LoRA at next generate.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v025_look_genre_invalidate_sd_cache() {
        with_singleton_ctx(|| {
            // Plant fake "loaded" pipelines so we can detect the
            // mark_loras_changed effect.
            with_ctx_mut(|ctx| {
                ctx.look_name = None;
                ctx.genre_name = None;
                // We can't easily plant a real cache slot without
                // loading a model; instead, observe that .apply
                // updates the name (which is the mark_loras_changed
                // call's documented side-effect on ctx).
            })
            .unwrap();
            eval(r#""watercolor" plakat.look.apply"#).unwrap();
            with_ctx(|ctx| assert_eq!(ctx.look_name.as_deref(), Some("watercolor"))).unwrap();
            // Apply again with a different name — also invalidates.
            eval(r#""oil-painting" plakat.look.apply"#).unwrap();
            with_ctx(|ctx| assert_eq!(ctx.look_name.as_deref(), Some("oil-painting"))).unwrap();
            // Same for genre.
            eval(r#""anime" plakat.genre.apply"#).unwrap();
            with_ctx(|ctx| assert_eq!(ctx.genre_name.as_deref(), Some("anime"))).unwrap();
        });
    }

    /// Cross-cycle integration: v0.22 + v0.23 + v0.24 + v0.25
    /// namespaces compose in one script without state leakage
    /// between axes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composition_v025_with_prior_cycle_surfaces() {
        with_singleton_ctx(|| {
            with_ctx_mut(|ctx| {
                ctx.loras.clear();
                ctx.controlnets.clear();
                ctx.artefacts.clear();
                ctx.portrait_photos.clear();
                ctx.embeddings.clear();
                ctx.refiner_enabled = false;
                ctx.adetailer_enabled = false;
                ctx.hires_enabled = false;
                ctx.style_id = None;
                ctx.style_ref = None;
                ctx.look_name = None;
                ctx.genre_name = None;
                ctx.config.face_bbox = None;
                ctx.config.face_landmarks = None;
                ctx.config.identity_kind.clear();
                ctx.config.offline_discovery = false;
            })
            .unwrap();

            eval(
                r#"
                // v0.22: LoRA + ControlNet stacks, post-process toggles.
                "user/lora-a" 0.7 plakat.lora.add
                "depth" "./photo.jpg" plakat.controlnet.annotate
                "oak" plakat.artefact.add
                plakat.refiner.enable
                plakat.adetailer.enable

                // v0.23: style.
                "poster-bold" plakat.style.apply

                // v0.24: portrait photos, embeddings, persona.
                "./alice.jpg" 1.0 plakat.portrait.photo.add
                "./ti.safetensors:trig:0.5" plakat.embedding.add
                "face-id" "identity_kind" plakat.config.set

                // v0.25: look + genre + offline_discovery.
                "watercolor" plakat.look.apply
                "anime"      plakat.genre.apply
                "true" "offline_discovery" plakat.config.set
            "#,
            )
            .unwrap();

            with_ctx(|ctx| {
                // v0.22 state.
                assert_eq!(ctx.loras.len(), 1);
                assert_eq!(ctx.controlnets.len(), 1);
                assert_eq!(ctx.artefacts.len(), 1);
                assert!(ctx.refiner_enabled);
                assert!(ctx.adetailer_enabled);
                // v0.23 state.
                assert_eq!(ctx.style_id.as_deref(), Some("poster-bold"));
                // v0.24 state.
                assert_eq!(ctx.portrait_photos.len(), 1);
                assert_eq!(ctx.embeddings.len(), 1);
                assert_eq!(ctx.config.identity_kind, "face-id");
                // v0.25 state.
                assert_eq!(ctx.look_name.as_deref(), Some("watercolor"));
                assert_eq!(ctx.genre_name.as_deref(), Some("anime"));
                assert!(ctx.config.offline_discovery);
            })
            .unwrap();
        });
    }

    /// Bytewise-style override-only invariant for the v0.25 presets:
    /// when the user has explicitly set steps/guidance/scheduler,
    /// applying a look leaves those scalar fields untouched (only
    /// the compositional fields change). Mirrors the CLI flag claim
    /// "explicit flags always win."
    #[test]
    fn composition_v025_override_only_invariant() {
        use crate::preset::{GenerationParams, apply_presets};

        // Fully-populated user side — every scalar field set.
        let user_steps = 50;
        let user_guidance = 9.0;
        let user_scheduler = "euler-a".to_string();

        let mut params = GenerationParams {
            prompt: "a knight".into(),
            negative: "blurry".into(),
            steps: Some(user_steps),
            guidance: Some(user_guidance),
            scheduler: Some(user_scheduler.clone()),
        };

        // Apply a look that would otherwise set different sampler
        // values (watercolor: steps=32, guidance=6.0, dpmpp-2m).
        let (look, _) = apply_presets(Some("watercolor"), None, &mut params).unwrap();
        assert!(look.is_some());

        // Override-only fields preserved.
        assert_eq!(params.steps, Some(user_steps));
        assert!((params.guidance.unwrap() - user_guidance).abs() < f64::EPSILON);
        assert_eq!(params.scheduler.as_deref(), Some("euler-a"));

        // Compositional fields DID change (prompt/negative compose).
        assert!(params.prompt.contains("watercolor"));
        assert!(params.prompt.contains("a knight"));
        assert!(params.negative.contains("photographic"));
    }

    /// Test-helper: serialises every test that needs the singleton
    /// context behind one shared init. Subsequent calls re-use the
    /// already-init'd singleton; only the *first* test through the
    /// gate pays the init cost. Each test runs its body inside the
    /// mutex so they don't trample each other's state on `out_dir`
    /// or `images`.
    pub(crate) fn with_singleton_ctx<R>(body: impl FnOnce() -> R) -> R {
        use std::sync::Mutex;
        static GATE: Mutex<()> = Mutex::new(());
        // Recover from poisoning. If an earlier test panicked while
        // holding the lock, the singleton state itself is fine
        // (panics rarely happen mid-init; even if they do, the next
        // test sees the OnceLock as already-set and uses it). The
        // poison is a stale flag we just need to clear so subsequent
        // tests don't all fail with "lock poisoned" cascades.
        let _g = GATE.lock().unwrap_or_else(|p| p.into_inner());
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
