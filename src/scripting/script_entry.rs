//! Thin façade over plakat's pipelines for the `plakat.*` host
//! words. Cache-aware as of v0.22 phase 1.
//!
//! In v0.21 every image-producing word called `t2i::run` /
//! `img2img::run` / `portrait::run`, which each reloaded the
//! model. v0.22 phase 1 caches the loaded pipeline in
//! [`ScriptCtx::loaded`] and reuses it across calls — a single
//! SDXL load amortises across a whole script.
//!
//! Architectural choice (RFC §7): cache a `portrait::Pipeline`
//! because it generalises across the three image-producing
//! words. Phases 2-3 will add `flux::Pipeline` and
//! `sd3::Pipeline` variants and lift the SD-family gate.
//!
//! All three `*_one` functions are sync as of phase 1. The
//! pipeline calls themselves are sync; the model-load happens
//! inside [`ScriptCtx::get_or_load_sd_family`] which uses
//! `tokio::task::block_in_place` internally. The img2img path
//! is the one remaining async caller (`img2img::run_with_pipeline`
//! is async); it bridges via `block_in_place` here.

use anyhow::{Context, Result, anyhow, bail};
use image::DynamicImage;
use std::path::{Path, PathBuf};

use crate::pipelines::{flux, ip_adapter::WeightedPhoto, portrait, sd3, t2i};
use crate::scripting::ctx::ScriptCtx;
use crate::scripting::loaded_pipeline::PipelineFamily;

/// v0.22 phase 3: legacy gate kept for back-compat with v0.21 tests.
/// New code should use [`ScriptCtx::ensure_loaded`] which handles
/// the family dispatch itself.
///
/// As of phase 3 every family plakat knows about is supported:
/// SD-family, Flux, and SD3 / SD3.5. The function still exists for
/// callers that want to validate before invoking the cache — it
/// just returns `Ok` for everything now.
pub fn validate_supported_for_phase_2(_model: &str) -> Result<()> {
    Ok(())
}

/// Pick the per-family default size used when the script hasn't
/// set width / height explicitly. SD 1.5 / 2.1 → 512²;
/// SDXL / SDXL-Turbo → 1024²; Flux → 1024²; SD3 / SD3.5 → 1024².
/// Reads the alias on `ctx.loaded`.
fn default_size_for_loaded(ctx: &ScriptCtx) -> (u32, u32) {
    let alias = ctx
        .loaded_model()
        .expect("default_size called without a loaded pipeline");
    let resolved = if alias.contains('/') {
        alias.to_string()
    } else {
        crate::hf::resolve_alias(alias).to_string()
    };
    let variant = t2i::Variant::detect(&resolved);
    if variant.is_flux() || variant.is_xl() || variant.is_sd3() {
        (1024, 1024)
    } else {
        (512, 512)
    }
}

/// v0.22 phase 5: resolve the script's ControlNet stack to
/// `Vec<OwnedControl>` for the SD-family path. `pipeline.generate`
/// wants `&[ControlRequest]` that borrow from owned data; the
/// caller binds the returned vec to a stack-frame local and
/// builds the requests via [`controlnets_to_requests`].
///
/// `fallback_input` is the source image to auto-annotate when a
/// `ControlSpec` has neither `image=` nor `from=` set. `None`
/// for generate (a missing input bails); `Some(path)` for
/// img2img (uses the source image).
///
/// Empty `ctx.controlnets` short-circuits to an empty vec
/// without paying any HF download cost.
fn resolve_sd_controlnets(
    ctx: &ScriptCtx,
    alias: &str,
    width: u32,
    height: u32,
    fallback_input: Option<&Path>,
) -> Result<Vec<crate::pipelines::controlnet::OwnedControl>> {
    if ctx.controlnets.is_empty() {
        return Ok(Vec::new());
    }
    let dtype = if matches!(ctx.device, candle_core::Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::F16
    };
    let specs = ctx.controlnets.clone();
    let device = ctx.device.clone();
    let model = alias.to_string();
    let fallback = fallback_input.map(|p| p.to_path_buf());
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow!(
            "plakat.controlnet: no tokio runtime in scope (eval must run on \
             a multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    tokio::task::block_in_place(|| {
        handle.block_on(crate::pipelines::controlnet::load_control_stack(
            &specs,
            &model,
            width,
            height,
            &device,
            dtype,
            fallback.as_deref(),
        ))
    })
    .context("loading ControlNet stack for script generate")
}

/// v0.22 phase 5: borrow each `OwnedControl` into a
/// `ControlRequest` matching `pipeline.generate`'s arg shape.
/// Cheap — `conditioning.clone()` is an Arc bump.
fn controlnets_to_requests<'a>(
    owned: &'a [crate::pipelines::controlnet::OwnedControl],
) -> Vec<crate::pipelines::controlnet::ControlRequest<'a>> {
    owned
        .iter()
        .map(|o| crate::pipelines::controlnet::ControlRequest {
            net: &o.net,
            conditioning: o.conditioning.clone(),
            strength: o.strength,
            start: o.start,
            end: o.end,
        })
        .collect()
}

/// v0.22 phase 3: build the TiledConfig if the script enabled
/// tiled denoise, else None. Shared across Flux + SD3.
fn tiled_cfg_from(ctx: &ScriptCtx) -> Option<crate::pipelines::tiled::TiledConfig> {
    if ctx.config.tiled {
        Some(crate::pipelines::tiled::TiledConfig {
            tile_size: ctx.config.tile_size,
            stride: ctx.config.tile_stride,
        })
    } else {
        None
    }
}

/// v0.22 phase 2: build a `flux::GenRequest` from the script's
/// config. Most fields map straight across from `GenerationConfig`;
/// Flux-specific knobs come from the D-keys (kontext_bucket,
/// fast applies at the pipeline level so isn't here).
fn build_flux_gen_request(
    ctx: &ScriptCtx,
    prompt: &str,
    out_dir: PathBuf,
    init_image: Option<PathBuf>,
) -> flux::GenRequest {
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        default_size_for_loaded(ctx)
    };
    flux::GenRequest {
        prompt: prompt.to_string(),
        width,
        height,
        count: 1,
        steps: Some(ctx.config.steps),
        // Honour user-set guidance; non-default (7.5) is suspicious
        // on Flux but we pass it through. The user can call
        // `plakat.config.set "guidance" 3.5` to pin the BFL default.
        guidance: Some(ctx.config.guidance),
        seed: ctx.config.seed,
        out_dir,
        conditioning: None,
        init_image,
        mask: None,
        strength: Some(ctx.config.strength),
        concept_conditioning: None,
        tiled: tiled_cfg_from(ctx),
        redux_images: Vec::new(),
        kontext_bucket: ctx.config.kontext_bucket,
        output_format: crate::imaging::io::OutputFormat::Png,
    }
}

/// v0.22 phase 3: build an `sd3::GenRequest` from the script's
/// config. SD3 lacks SD-family's `face_strength` + Flux's
/// `kontext_bucket`; it has its own `mask_feather` + `mask_invert`
/// not yet exposed at the script layer (v0.23 once
/// `plakat.inpaint` lands).
fn build_sd3_gen_request(
    ctx: &ScriptCtx,
    prompt: &str,
    out_dir: PathBuf,
    init_image: Option<PathBuf>,
) -> sd3::GenRequest {
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        default_size_for_loaded(ctx)
    };
    sd3::GenRequest {
        prompt: prompt.to_string(),
        negative: ctx.config.negative.clone(),
        width,
        height,
        count: 1,
        steps: Some(ctx.config.steps),
        guidance: Some(ctx.config.guidance),
        seed: ctx.config.seed,
        out_dir,
        init_image,
        mask: None,
        mask_feather: 0,
        mask_invert: false,
        strength: Some(ctx.config.strength),
        tiled: tiled_cfg_from(ctx),
        controlnet_conditioning: Vec::new(),
        output_format: crate::imaging::io::OutputFormat::Png,
    }
}

/// Build a `portrait::GenRequest` from the script's accumulated
/// `GenerationConfig`. Shared across all three image-producing
/// host words; only `prompt` + `photos` + `out_dir` differ
/// per-call.
fn build_gen_request(
    ctx: &ScriptCtx,
    prompt: &str,
    photos: Vec<WeightedPhoto>,
    out_dir: PathBuf,
) -> portrait::GenRequest {
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        default_size_for_loaded(ctx)
    };
    portrait::GenRequest {
        prompt: prompt.to_string(),
        negative: ctx.config.negative.clone(),
        photos,
        width,
        height,
        count: 1,
        steps: ctx.config.steps,
        guidance: ctx.config.guidance,
        seed: ctx.config.seed,
        out_dir,
        scheduler: ctx.config.scheduler,
        // v0.22 phase 6: same-model polish refine pass. `None`
        // when the script never set refine_steps (== v0.21
        // behaviour); `Some(N)` runs N extra denoise steps at
        // `refine_strength` after the main loop.
        refine: ctx.config.refine_steps,
        refine_strength: ctx.config.refine_strength,
        face_strength: ctx.config.face_strength,
        face_bbox: None,
        face_landmarks: None,
    }
}

/// Locate the single PNG `pipeline.generate` writes into `dir`
/// and load it as a [`DynamicImage`]. Pipelines name their
/// outputs `plakat-<seed>.png`, `plakat-portrait-<seed>.png`,
/// or `plakat-img2img-<seed>.png`; we don't try to predict the
/// filename — we just grab the single PNG file.
fn read_rendered_png(dir: &Path) -> Result<DynamicImage> {
    let path = find_rendered_png(dir)?;
    image::open(&path)
        .with_context(|| format!("reading rendered PNG {}", path.display()))
}

/// Same locator as [`read_rendered_png`] but returns the path
/// so callers (e.g. ADetailer post-process) can operate on the
/// file in place before the final load.
fn find_rendered_png(dir: &Path) -> Result<PathBuf> {
    let entry = std::fs::read_dir(dir)
        .with_context(|| format!("reading tempdir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .find(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            anyhow!(
                "pipeline produced no PNG in {} — pipeline may have \
                 silently failed",
                dir.display()
            )
        })?;
    Ok(entry.path())
}

/// v0.22 phase 8: snapshot of hires-fix inputs that can be built
/// *before* the cached pipeline is borrowed (same rationale as
/// [`AdetailerArgs`]).
struct HiresArgs {
    enabled: bool,
    cfg: crate::pipelines::hires_fix::Config,
}

impl HiresArgs {
    fn from_ctx(ctx: &ScriptCtx, alias: &str, prompt: &str) -> Result<Self> {
        use std::str::FromStr;
        let upscaler = crate::imaging::upscale::Method::from_str(
            &ctx.config.hires_upscaler,
        )
        .with_context(|| {
            format!(
                "plakat.hires: hires_upscaler {:?} not recognised",
                ctx.config.hires_upscaler
            )
        })?;
        let cfg = crate::pipelines::hires_fix::Config {
            model: alias.to_string(),
            loras: ctx.loras.clone(),
            lora_scale: ctx.config.lora_scale,
            prompt: prompt.to_string(),
            negative: ctx.config.negative.clone(),
            scale: ctx.config.hires_scale,
            upscaler,
            strength: ctx.config.hires_strength,
            steps: ctx.config.hires_steps.unwrap_or(ctx.config.steps),
            guidance: ctx.config.guidance as f64,
            scheduler: ctx.config.scheduler,
            device: ctx.device.clone(),
        };
        Ok(Self {
            enabled: ctx.hires_enabled,
            cfg,
        })
    }
}

/// v0.22 phase 8: run hires-fix on the rendered PNG in place.
/// Reuses the cached `portrait::Pipeline`'s `SdCore` so no second
/// model load happens. Caller is responsible for the family check
/// (Flux + SD3 bail loud before reaching here).
fn apply_hires_sd(
    hcfg: &crate::pipelines::hires_fix::Config,
    rendered: &Path,
    pipeline: &portrait::Pipeline,
) -> Result<()> {
    let shared_core = Some(pipeline.core());
    let files = vec![rendered.to_path_buf()];
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow!(
            "plakat.hires: no tokio runtime in scope (eval must run on \
             a multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let n = tokio::task::block_in_place(|| {
        handle.block_on(crate::pipelines::hires_fix::refine_files(
            hcfg, &files, shared_core,
        ))
    })
    .context("hires_fix::refine_files (plakat post-process)")?;
    tracing::info!(
        target: "plakat",
        "plakat.hires: refined {n} file(s) on {}",
        rendered.display()
    );
    Ok(())
}

/// v0.22 phase 7: snapshot of ADetailer inputs that can be built
/// *before* the cached pipeline is borrowed. Keeps the post-process
/// out of the `ctx` borrow scope (the cached pipeline mutably
/// borrows `ctx`, so we can't reach back into `ctx.config` while
/// holding that borrow).
struct AdetailerArgs {
    enabled: bool,
    cfg: crate::pipelines::adetailer::Config,
}

impl AdetailerArgs {
    fn from_ctx(ctx: &ScriptCtx, alias: &str) -> Self {
        let mut cfg = crate::pipelines::adetailer::Config::defaults();
        cfg.model = alias.to_string();
        cfg.device = ctx.device.clone();
        cfg.prompt = ctx.config.adetailer_prompt.clone();
        cfg.negative = ctx.config.negative.clone();
        cfg.strength = ctx.config.adetailer_strength;
        cfg.working_size = ctx.config.adetailer_size;
        cfg.steps = ctx.config.steps;
        cfg.guidance = ctx.config.guidance as f64;
        cfg.scheduler = ctx.config.scheduler;
        cfg.confidence = ctx.config.adetailer_confidence;
        cfg.padding = ctx.config.adetailer_padding;
        cfg.feather = ctx.config.adetailer_feather;
        Self {
            enabled: ctx.adetailer_enabled,
            cfg,
        }
    }
}

/// v0.22 phase 7: run ADetailer on the rendered PNG in place.
/// Reuses the cached `portrait::Pipeline`'s `SdCore` so no second
/// model load happens. Caller is responsible for the family check
/// (Flux + SD3 bail loud before reaching here).
fn apply_adetailer_sd(
    acfg: &crate::pipelines::adetailer::Config,
    rendered: &Path,
    pipeline: &portrait::Pipeline,
) -> Result<()> {
    let shared_core = Some(pipeline.core());
    let files = vec![rendered.to_path_buf()];
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow!(
            "plakat.adetailer: no tokio runtime in scope (eval must run on \
             a multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let n = tokio::task::block_in_place(|| {
        handle.block_on(crate::pipelines::adetailer::refine_files(
            acfg, &files, shared_core,
        ))
    })
    .context("adetailer::refine_files (plakat post-process)")?;
    tracing::info!(
        target: "plakat",
        "plakat.adetailer: refined {n} face(s) on {}",
        rendered.display()
    );
    Ok(())
}

/// v0.22 phase 2: render one image. Dispatches on the loaded
/// pipeline's family — SD path uses `portrait::Pipeline.generate`
/// with empty photos; Flux path uses `flux::Pipeline.generate`
/// with no init_image.
pub fn generate_one(ctx: &mut ScriptCtx, prompt: &str) -> Result<DynamicImage> {
    let alias = ctx
        .loaded_model()
        .ok_or_else(|| {
            anyhow!(
                "plakat.generate: no model loaded. Call \"sd15\" plakat.load \
                 (or another supported alias) before plakat.generate."
            )
        })?
        .to_string();

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-gen-")
        .tempdir()
        .context("creating tempdir for plakat.generate output")?;
    let tmp_path = tmp.path().to_path_buf();

    match PipelineFamily::detect(&alias) {
        PipelineFamily::SdFamily => {
            if ctx.refiner_enabled {
                bail!(
                    "plakat.generate: SDXL refiner from scripts is deferred \
                     to v0.23 — the cached `portrait::Pipeline` doesn't \
                     hold the refiner UNet slot. Workarounds: call \
                     `plakat.refiner.disable` (same-model polish via \
                     `refine_steps`/`refine_strength` still works), or use \
                     `plakat generate --refiner` from the CLI directly."
                );
            }
            let req = build_gen_request(ctx, prompt, Vec::new(), tmp_path.clone());
            // v0.22 phase 5: resolve the script's controlnets to
            // OwnedControl + ControlRequest before borrowing the
            // pipeline. The owned data lives on this frame for the
            // pipeline.generate call's lifetime.
            let control_owned =
                resolve_sd_controlnets(ctx, &alias, req.width, req.height, None)?;
            let control_reqs = controlnets_to_requests(&control_owned);
            // v0.22 phase 7+8: snapshot post-process inputs *before*
            // the cached-pipeline borrow so they can run while we
            // still hold the pipeline reference. Hires fix runs
            // first (upscales + refines composition); ADetailer
            // runs second (refines faces at the higher resolution).
            let adargs = AdetailerArgs::from_ctx(ctx, &alias);
            let hargs = HiresArgs::from_ctx(ctx, &alias, prompt)?;
            let pipeline = ctx.get_or_load_sd_family(&alias)?;
            pipeline.generate(&req, &control_reqs)
                .context("portrait::Pipeline::generate (plakat.generate SD path)")?;
            if hargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_hires_sd(&hargs.cfg, &rendered, pipeline)?;
            }
            if adargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_adetailer_sd(&adargs.cfg, &rendered, pipeline)?;
            }
        }
        PipelineFamily::Flux => {
            if !ctx.controlnets.is_empty() {
                bail!(
                    "plakat.generate: ControlNet on Flux isn't wired in v0.22 \
                     phase 5 (Flux CN needs load-time setup; deferred to v0.23). \
                     Call plakat.controlnet.clear before plakat.generate on Flux."
                );
            }
            if ctx.adetailer_enabled {
                bail!(
                    "plakat.generate: ADetailer is SD-family only in v0.22 \
                     phase 7 — SCRFD + the face img2img pass require an SD \
                     backbone. Call plakat.adetailer.disable before \
                     plakat.generate on Flux."
                );
            }
            if ctx.hires_enabled {
                bail!(
                    "plakat.generate: hires-fix is SD-family only in v0.22 \
                     phase 8 — the refine pass needs an SD img2img \
                     pipeline. Call plakat.hires.disable before \
                     plakat.generate on Flux."
                );
            }
            let req = build_flux_gen_request(ctx, prompt, tmp_path.clone(), None);
            let pipeline = ctx.get_or_load_flux(&alias)?;
            pipeline.generate(&req)
                .context("flux::Pipeline::generate (plakat.generate Flux path)")?;
        }
        PipelineFamily::Sd3 => {
            if !ctx.controlnets.is_empty() {
                bail!(
                    "plakat.generate: ControlNet on SD3 isn't wired in v0.22 \
                     phase 5 (SD3 CN needs load-time setup; deferred to v0.23). \
                     Call plakat.controlnet.clear before plakat.generate on SD3."
                );
            }
            if ctx.adetailer_enabled {
                bail!(
                    "plakat.generate: ADetailer is SD-family only in v0.22 \
                     phase 7 — SCRFD + the face img2img pass require an SD \
                     backbone. Call plakat.adetailer.disable before \
                     plakat.generate on SD3."
                );
            }
            if ctx.hires_enabled {
                bail!(
                    "plakat.generate: hires-fix is SD-family only in v0.22 \
                     phase 8 — the refine pass needs an SD img2img \
                     pipeline. Call plakat.hires.disable before \
                     plakat.generate on SD3."
                );
            }
            let req = build_sd3_gen_request(ctx, prompt, tmp_path.clone(), None);
            let pipeline = ctx.get_or_load_sd3(&alias)?;
            pipeline.generate(&req)
                .context("sd3::Pipeline::generate (plakat.generate SD3 path)")?;
        }
    }
    read_rendered_png(&tmp_path)
}

/// v0.22 phase 1: render one img2img image. `input_path` may be
/// any filesystem path the script provides (or a tempfile from a
/// handle materialisation in the host word).
pub fn img2img_one(
    ctx: &mut ScriptCtx,
    prompt: &str,
    input_path: &Path,
) -> Result<DynamicImage> {
    let alias = ctx
        .loaded_model()
        .ok_or_else(|| {
            anyhow!(
                "plakat.img2img: no model loaded. Call \"sd15\" plakat.load \
                 (or another supported alias) before plakat.img2img."
            )
        })?
        .to_string();

    // Working size: explicit config wins; else input image dims
    // snapped to /8 (downward). Read config + dims first, then
    // borrow the pipeline mutably.
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        let dims = image::image_dimensions(input_path).with_context(|| {
            format!(
                "reading dimensions of {} for plakat.img2img working size",
                input_path.display()
            )
        })?;
        let (w, h) = dims;
        ((w / 8) * 8, (h / 8) * 8)
    };
    if width == 0 || height == 0 {
        bail!(
            "plakat.img2img: working size {width}x{height} collapsed to 0 \
             after /8 snap. Input image is too small (< 8 pixels on a side)."
        );
    }

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-i2i-")
        .tempdir()
        .context("creating tempdir for plakat.img2img output")?;
    let tmp_path = tmp.path().to_path_buf();

    match PipelineFamily::detect(&alias) {
        PipelineFamily::SdFamily => {
            let req = crate::pipelines::img2img::Request {
                prompt: prompt.to_string(),
                negative: ctx.config.negative.clone(),
                model: alias.clone(),
                device: ctx.device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                input: input_path.to_path_buf(),
                mask: None,
                mask_feather: 0,
                mask_invert: false,
                width,
                height,
                count: 1,
                steps: ctx.config.steps,
                guidance: ctx.config.guidance,
                scheduler: ctx.config.scheduler,
                strength: ctx.config.strength,
                seed: ctx.config.seed,
                out_dir: tmp_path.clone(),
                // v0.22 phase 5: ControlNet stack flows through
                // img2img::Request.controls. img2img::run_with_pipeline
                // resolves the specs internally.
                controls: ctx.controlnets.clone(),
            };
            // v0.22 phase 7+8: post-process snapshots before
            // pipeline borrow. Hires runs before ADetailer.
            let adargs = AdetailerArgs::from_ctx(ctx, &alias);
            let hargs = HiresArgs::from_ctx(ctx, &alias, prompt)?;
            let pipeline = ctx.get_or_load_sd_family(&alias)?;
            // run_with_pipeline is async; bridge via block_in_place.
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "plakat.img2img: no tokio runtime in scope (eval must \
                     run on a multi-threaded runtime). Underlying error: {e}"
                )
            })?;
            tokio::task::block_in_place(|| {
                handle.block_on(crate::pipelines::img2img::run_with_pipeline(
                    pipeline, &req,
                ))
            })
            .context("img2img::run_with_pipeline (plakat.img2img SD path)")?;
            if hargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_hires_sd(&hargs.cfg, &rendered, pipeline)?;
            }
            if adargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_adetailer_sd(&adargs.cfg, &rendered, pipeline)?;
            }
        }
        PipelineFamily::Flux => {
            if !ctx.controlnets.is_empty() {
                bail!(
                    "plakat.img2img: ControlNet on Flux isn't wired in v0.22 \
                     phase 5 (deferred to v0.23). Call plakat.controlnet.clear \
                     before plakat.img2img on Flux."
                );
            }
            if ctx.adetailer_enabled {
                bail!(
                    "plakat.img2img: ADetailer is SD-family only in v0.22 \
                     phase 7. Call plakat.adetailer.disable before \
                     plakat.img2img on Flux."
                );
            }
            if ctx.hires_enabled {
                bail!(
                    "plakat.img2img: hires-fix is SD-family only in v0.22 \
                     phase 8. Call plakat.hires.disable before \
                     plakat.img2img on Flux."
                );
            }
            // Flux img2img threads `init_image` + `strength` through
            // the same flux::GenRequest used for text-to-image.
            // Working size override: width / height become the
            // current values (size_explicit OR snapped input dims).
            let mut req = build_flux_gen_request(
                ctx,
                prompt,
                tmp_path.clone(),
                Some(input_path.to_path_buf()),
            );
            req.width = width;
            req.height = height;
            let pipeline = ctx.get_or_load_flux(&alias)?;
            pipeline.generate(&req)
                .context("flux::Pipeline::generate (plakat.img2img Flux path)")?;
        }
        PipelineFamily::Sd3 => {
            if !ctx.controlnets.is_empty() {
                bail!(
                    "plakat.img2img: ControlNet on SD3 isn't wired in v0.22 \
                     phase 5 (deferred to v0.23). Call plakat.controlnet.clear \
                     before plakat.img2img on SD3."
                );
            }
            if ctx.adetailer_enabled {
                bail!(
                    "plakat.img2img: ADetailer is SD-family only in v0.22 \
                     phase 7. Call plakat.adetailer.disable before \
                     plakat.img2img on SD3."
                );
            }
            if ctx.hires_enabled {
                bail!(
                    "plakat.img2img: hires-fix is SD-family only in v0.22 \
                     phase 8. Call plakat.hires.disable before \
                     plakat.img2img on SD3."
                );
            }
            // SD3 img2img: GenRequest has init_image + strength
            // built-in, same shape as Flux. Working size honours
            // the snapped input dims.
            let mut req = build_sd3_gen_request(
                ctx,
                prompt,
                tmp_path.clone(),
                Some(input_path.to_path_buf()),
            );
            req.width = width;
            req.height = height;
            let pipeline = ctx.get_or_load_sd3(&alias)?;
            pipeline.generate(&req)
                .context("sd3::Pipeline::generate (plakat.img2img SD3 path)")?;
        }
    }
    read_rendered_png(&tmp_path)
}

/// v0.22 phase 1: render one portrait. Uses the cached
/// pipeline's identity encoder; if the pipeline was loaded
/// without one (sd21), `pipeline.generate` bails with the
/// v0.21 "no identity encoder" message.
pub fn portrait_one(
    ctx: &mut ScriptCtx,
    prompt: &str,
    photo_path: &Path,
) -> Result<DynamicImage> {
    let alias = ctx
        .loaded_model()
        .ok_or_else(|| {
            anyhow!(
                "plakat.portrait: no model loaded. Call \"sd15\" plakat.load \
                 (or \"sdxl\" plakat.load) before plakat.portrait."
            )
        })?
        .to_string();

    // Portrait is SD-family-only. Flux + SD3 have no shipped
    // IP-Adapter-Plus-Face checkpoint, so neither can do
    // identity-preserving portraits. Bail loud rather than
    // silently loading the wrong pipeline.
    match PipelineFamily::detect(&alias) {
        PipelineFamily::Flux => bail!(
            "plakat.portrait: Flux has no IP-Adapter-Plus-Face \
             checkpoint (got {alias:?}). Use SD 1.5 / SDXL for \
             identity-preserving portraits in v0.22."
        ),
        PipelineFamily::Sd3 => bail!(
            "plakat.portrait: SD3 / SD3.5 has no IP-Adapter-Plus-Face \
             checkpoint (got {alias:?}). Use SD 1.5 / SDXL for \
             identity-preserving portraits."
        ),
        PipelineFamily::SdFamily => {}
    }

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-portrait-")
        .tempdir()
        .context("creating tempdir for plakat.portrait output")?;
    let photos = vec![WeightedPhoto::single(photo_path.to_path_buf())];
    let mut req = build_gen_request(ctx, prompt, photos, tmp.path().to_path_buf());
    // Override per-family default size for portrait: 3:4
    // is the CLI default. Honour size_explicit override.
    if !ctx.config.size_explicit {
        let (w, h) = default_size_for_loaded(ctx);
        // CLI portrait default is 3:4; for SDXL → 768×1024,
        // for SD 1.5 → 512×768. Map from the square default.
        req.width = w * 3 / 4;
        req.height = h;
        // VAE-snap to /8.
        req.width = (req.width / 8) * 8;
        req.height = (req.height / 8) * 8;
    }
    // Normalize photo weights (the pipeline's invariant).
    crate::pipelines::ip_adapter::normalize_photo_weights(&mut req.photos)?;

    // v0.22 phase 7+8: post-process snapshots before pipeline borrow.
    let adargs = AdetailerArgs::from_ctx(ctx, &alias);
    let hargs = HiresArgs::from_ctx(ctx, &alias, prompt)?;
    let pipeline = ctx.get_or_load_sd_family(&alias)?;
    pipeline.generate(&req, &[])
        .context("portrait::Pipeline::generate (plakat.portrait path)")?;
    if hargs.enabled {
        let rendered = find_rendered_png(tmp.path())?;
        apply_hires_sd(&hargs.cfg, &rendered, pipeline)?;
    }
    if adargs.enabled {
        let rendered = find_rendered_png(tmp.path())?;
        apply_adetailer_sd(&adargs.cfg, &rendered, pipeline)?;
    }
    read_rendered_png(tmp.path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_2_gate_accepts_sd_family_aliases() {
        for alias in &["sd15", "sd21", "sdxl", "sdxl-turbo"] {
            validate_supported_for_phase_2(alias).unwrap_or_else(|e| {
                panic!("alias {alias:?} should be accepted in phase 1: {e}")
            });
        }
    }

    #[test]
    fn phase_2_gate_accepts_flux_aliases() {
        // v0.22 phase 2 lifts the Flux gate. SD3 still bails.
        for alias in &["flux-dev", "flux-schnell", "flux-kontext-dev"] {
            validate_supported_for_phase_2(alias).unwrap_or_else(|e| {
                panic!("Flux alias {alias:?} should pass the gate in v0.22 phase 2: {e}")
            });
        }
    }

    #[test]
    fn phase_2_gate_accepts_sd3_aliases() {
        // v0.22 phase 3 lifts the last family bail. Every family
        // plakat knows now passes validate_supported_for_phase_2.
        for alias in &["sd35-medium", "sd35-large", "sd3-medium"] {
            validate_supported_for_phase_2(alias).unwrap_or_else(|e| {
                panic!("SD3 alias {alias:?} should pass the gate in v0.22 phase 3: {e}")
            });
        }
    }

    #[test]
    fn phase_2_gate_passes_canonical_hf_repos_for_sd_family() {
        validate_supported_for_phase_2(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        )
        .unwrap();
    }
}
