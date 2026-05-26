//! v0.22 phase 1: the cached pipeline behind [`ScriptCtx`].
//!
//! v0.21 reloaded the model on every `plakat.generate` /
//! `plakat.img2img` / `plakat.portrait`. v0.22 caches the loaded
//! pipeline by alias so subsequent calls reuse it — a single SDXL
//! load (~10s on a 4090) amortises across a script that produces
//! dozens of images.
//!
//! Phase 1 ships only the SD-family variant. Phases 2-3 will add
//! `Flux(flux::Pipeline)` and `Sd3(sd3::Pipeline)` and lift the
//! [`super::script_entry::validate_supported_for_phase_2`] gate.

use crate::pipelines::portrait;

/// The active pipeline, cached by `(alias, pipeline)` in
/// [`super::ctx::ScriptCtx::loaded`].
///
/// We hold a [`portrait::Pipeline`] for the SD-family case
/// because it generalises across the three image-producing host
/// words:
///
/// * `plakat.generate` → `portrait::Pipeline::generate` with
///   empty `photos` (pure text-to-image; the identity encoder
///   is loaded but produces no tokens when photos is empty).
/// * `plakat.img2img` → `img2img::run_with_pipeline` borrows the
///   same loaded pipeline.
/// * `plakat.portrait` → `portrait::Pipeline::generate` with one
///   reference photo.
///
/// Identity encoder is loaded conditionally at cache-creation
/// time based on the alias:
/// * `sd15` / `sdxl` / `sdxl-turbo` → `PlusFace` / `PlusFaceSdxl`
/// * `sd21` → `None` (no shipped Plus-Face SD 2.1 checkpoint)
///
/// `plakat.portrait` against an `sd21`-loaded pipeline bails at
/// generate time with the v0.21 "no identity encoder" message —
/// same behaviour as v0.21.
pub enum LoadedPipeline {
    SdFamily(portrait::Pipeline),
    // Phase 2: Flux(crate::pipelines::flux::Pipeline)
    // Phase 3: Sd3(crate::pipelines::sd3::Pipeline)
}
