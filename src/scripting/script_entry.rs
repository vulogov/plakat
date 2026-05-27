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
///
/// v0.22 phase 11: when `ctx.config.aspect` is non-empty, the
/// aspect-derived size takes precedence over the family default.
/// `base` is the shorter side; the longer side scales to maintain
/// the ratio, snapped to /8 (VAE).
fn default_size_for_loaded(ctx: &ScriptCtx) -> (u32, u32) {
    if !ctx.config.aspect.is_empty() {
        if let Some(dims) = aspect_to_size(&ctx.config.aspect, ctx.config.base) {
            return dims;
        }
    }
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

/// v0.22 phase 11: parse `W:H` aspect + `base` into `(w, h)`. The
/// shorter side equals `base`; the longer side scales to keep the
/// ratio, then both snap down to a multiple of 8 (VAE). Returns
/// `None` on malformed input (caller falls back to family default).
/// `set_str` already validates at config-set time, so a `None`
/// here means defaults won.
fn aspect_to_size(aspect: &str, base: u32) -> Option<(u32, u32)> {
    let (a, b) = aspect.split_once(':')?;
    let w_ratio: u32 = a.parse().ok()?;
    let h_ratio: u32 = b.parse().ok()?;
    if w_ratio == 0 || h_ratio == 0 {
        return None;
    }
    let (w, h) = if w_ratio >= h_ratio {
        (base * w_ratio / h_ratio, base)
    } else {
        (base, base * h_ratio / w_ratio)
    };
    let snap = |n: u32| (n / 8) * 8;
    Some((snap(w), snap(h)))
}

/// v0.22 phase 11: combine `config.negative_preset` (when set)
/// with the user-provided `config.negative` via
/// `negative_presets::combine`. Returns the user negative
/// unchanged when no preset is configured. A non-resolving
/// preset name (e.g. user-installed preset got removed
/// mid-script) warns and falls back rather than bailing, since
/// `set_str` validated at config-set time.
fn resolve_negative(ctx: &ScriptCtx) -> String {
    let preset = if ctx.config.negative_preset.is_empty() {
        None
    } else {
        Some(ctx.config.negative_preset.as_str())
    };
    match crate::prompt::negative_presets::combine(preset, &ctx.config.negative) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                target: "plakat",
                "plakat.config: negative_preset combine failed ({e}) — \
                 falling back to user negative only"
            );
            ctx.config.negative.clone()
        }
    }
}

/// v0.23 phase 4: resolve the script's active style state (id +
/// ref) against the catalog. Returns `None` when neither is set;
/// otherwise builds a [`StylePrepRequest`] for the current alias
/// and dispatches to [`prepare_style`].
///
/// The async bridge follows the same pattern as the other
/// post-process helpers — `block_in_place` + `block_on` on the
/// current tokio handle. CLIP-H is lazy-loaded inside
/// `prepare_style` only when `style_ref` is set (.detect path);
/// `style_id` alone (.apply path) only needs the catalog JSON +
/// the SD-family per-model entries.
fn resolve_style_for_generate(
    ctx: &ScriptCtx,
    alias: &str,
) -> Result<Option<crate::style::StylePrep>> {
    if ctx.style_id.is_none() && ctx.style_ref.is_none() {
        return Ok(None);
    }
    let catalog_dir = if ctx.config.style_catalog.is_empty() {
        None
    } else {
        Some(std::path::PathBuf::from(&ctx.config.style_catalog))
    };
    let req = crate::style::StylePrepRequest {
        style_ref: ctx.style_ref.as_deref(),
        style_override: ctx.style_id.as_deref(),
        style_strength: ctx.config.style_strength,
        style_catalog: catalog_dir.as_deref(),
        model: alias,
        user_loras_nonempty: !ctx.loras.is_empty(),
        device: &ctx.device,
    };
    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow!(
            "plakat.style: no tokio runtime in scope (eval must run on \
             a multi-threaded runtime). Underlying error: {e}"
        )
    })?;
    let prep = tokio::task::block_in_place(|| {
        handle.block_on(crate::style::prepare_style(req))
    })
    .context("style catalog resolve (plakat.style.* lazy resolve)")?;
    Ok(Some(prep))
}

/// v0.22 phase 11: expand the prompt against `config.wildcard_dir`
/// (`__name__` file wildcards + inline `{a|b|c}` alternation).
/// When `wildcard_dir` is empty, only inline alternation expands.
/// Seed: `config.seed` when set (reproducible), else OS entropy.
fn expand_prompt(ctx: &ScriptCtx, prompt: &str) -> Result<String> {
    use crate::prompt::wildcards;
    let dir = if ctx.config.wildcard_dir.is_empty() {
        None
    } else {
        Some(std::path::PathBuf::from(&ctx.config.wildcard_dir))
    };
    use rand::SeedableRng;
    let mut rng: rand::rngs::StdRng = match ctx.config.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };
    wildcards::expand(prompt, dir.as_deref(), &mut rng)
        .context("expanding wildcards in script prompt")
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
/// not yet exposed at the script layer (v0.23 phase 5 once
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
        negative: resolve_negative(ctx),
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
        negative: resolve_negative(ctx),
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
        // v0.24 phase 2: face_bbox + face_landmarks come from
        // config keys. Read only by the portrait path; ignored
        // by the SD-family generate/img2img paths which pass an
        // empty `photos` vec.
        face_bbox: ctx.config.face_bbox,
        face_landmarks: ctx.config.face_landmarks,
    }
}

/// v0.23 phase 1: build a `t2i::GenRequest` from the script's
/// `GenerationConfig`. Used by `plakat.generate`'s SD-family
/// path. Maps the cross-cutting GenerationConfig fields onto
/// t2i's request shape, which exposes the SD-family extras
/// (`clip_skip`, refiner controls, preview cadence, metadata)
/// that the portrait::GenRequest doesn't carry.
fn build_t2i_gen_request(
    ctx: &ScriptCtx,
    prompt: &str,
    out_dir: PathBuf,
) -> t2i::GenRequest {
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        default_size_for_loaded(ctx)
    };
    t2i::GenRequest {
        prompt: prompt.to_string(),
        negative: resolve_negative(ctx),
        width,
        height,
        count: 1,
        steps: ctx.config.steps,
        guidance: ctx.config.guidance,
        seed: ctx.config.seed,
        out_dir,
        scheduler: ctx.config.scheduler,
        refine: ctx.config.refine_steps,
        refine_strength: ctx.config.refine_strength,
        // v0.23 phase 2: refiner_frac steers the base→refiner
        // schedule split when `ctx.refiner_enabled` triggered the
        // SDXL refiner UNet load.
        refiner_frac: Some(ctx.config.refiner_frac),
        // v0.23 phase 3: clip_skip honoured by t2i::Pipeline's
        // encode_prompt — SD 1.5 / SD 2.1 returns the (N-th from
        // last) CLIP-L hidden state. SDXL / Flux / SD3 ignore
        // (SDXL already uses penultimate by training default).
        clip_skip: ctx.config.clip_skip,
        metadata: None,
        preview_every: None,
        preview_size: None,
        output_format: crate::imaging::io::OutputFormat::Png,
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

/// v0.22 phase 9: snapshot of artefact compose + blend inputs
/// that can be built *before* the cached pipeline is borrowed
/// (same rationale as [`AdetailerArgs`] / [`HiresArgs`]).
///
/// `specs` empty → the whole post-process step is a no-op (matches
/// `composite_onto_files`'s short-circuit). The optional blend
/// pass needs the model alias + LoRA stack to build a
/// `BlendConfig`; we snapshot them too so a later config-key
/// change can't desync the blend with the cached pipeline.
struct ArtefactArgs {
    specs: Vec<crate::artefacts::ArtefactSpec>,
    library_dir: std::path::PathBuf,
    smart_zones: bool,
    blend_enabled: bool,
    blend_cfg: crate::pipelines::artefact_blend::BlendConfig,
}

impl ArtefactArgs {
    fn from_ctx(
        ctx: &ScriptCtx,
        alias: &str,
        prompt: &str,
        image_w: u32,
        image_h: u32,
    ) -> Self {
        let library_dir = if ctx.config.artefact_library.is_empty() {
            std::path::PathBuf::from("assets/artefact_library")
        } else {
            std::path::PathBuf::from(&ctx.config.artefact_library)
        };
        let blend_cfg = crate::pipelines::artefact_blend::BlendConfig {
            model: alias.to_string(),
            device: ctx.device.clone(),
            loras: ctx.loras.clone(),
            lora_scale: ctx.config.lora_scale,
            prompt: prompt.to_string(),
            negative: resolve_negative(ctx),
            image_w,
            image_h,
            steps: ctx.config.steps,
            guidance: ctx.config.guidance as f64,
            scheduler: ctx.config.scheduler,
            strength: ctx.config.artefact_blend_strength,
            feather_px: None,
        };
        Self {
            specs: ctx.artefacts.clone(),
            library_dir,
            smart_zones: ctx.config.artefact_smart_zones,
            blend_enabled: ctx.artefact_blend_enabled,
            blend_cfg,
        }
    }

    fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

/// v0.22 phase 9: run the artefact compositing + optional blend
/// pass on the rendered PNG in place. Caller is responsible for
/// the family check (Flux + SD3 bail before reaching here).
///
/// Smart-zones loads `Depth-Anything-V2-Small` on first use; the
/// runtime block-in-place bridge mirrors the other post-process
/// paths.
fn apply_artefacts_sd(
    args: &ArtefactArgs,
    rendered: &std::path::Path,
    shared_core: std::sync::Arc<crate::pipelines::sd_core::SdCore>,
) -> Result<()> {
    if args.is_empty() {
        return Ok(());
    }
    let canvas_w = args.blend_cfg.image_w;
    let canvas_h = args.blend_cfg.image_h;
    let files = vec![rendered.to_path_buf()];

    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow!(
            "plakat.artefact: no tokio runtime in scope (eval must run on \
             a multi-threaded runtime). Underlying error: {e}"
        )
    })?;

    // Smart-zones: lazy depth-pipeline load. We pay the load cost
    // exactly when the script asked for it; otherwise None and
    // the rigid grid runs (no depth weights download).
    let smart = if args.smart_zones {
        Some(
            tokio::task::block_in_place(|| {
                handle.block_on(crate::pipelines::depth::DepthPipeline::load(
                    args.blend_cfg.device.clone(),
                ))
            })
            .context("loading Depth-Anything-V2-Small for artefact smart-zones")?,
        )
    } else {
        None
    };

    // Alpha composite — synchronous, no model load. Empty spec
    // list is already handled by the early-return above.
    crate::artefacts::composite_onto_files(
        &args.specs,
        &args.library_dir,
        &files,
        canvas_w,
        canvas_h,
        &Default::default(),
        smart.as_ref(),
    )
    .context("artefact compositing (plakat.artefact post-process)")?;

    if args.blend_enabled {
        let blend_shared_core = Some(shared_core.clone());
        // BlendConfig is non-Clone; build a fresh one here from the
        // snapshot. The snapshot's blend_cfg fields are owned strings
        // / cheap to recreate.
        let bcfg = crate::pipelines::artefact_blend::BlendConfig {
            model: args.blend_cfg.model.clone(),
            device: args.blend_cfg.device.clone(),
            loras: args.blend_cfg.loras.clone(),
            lora_scale: args.blend_cfg.lora_scale,
            prompt: args.blend_cfg.prompt.clone(),
            negative: args.blend_cfg.negative.clone(),
            image_w: args.blend_cfg.image_w,
            image_h: args.blend_cfg.image_h,
            steps: args.blend_cfg.steps,
            guidance: args.blend_cfg.guidance,
            scheduler: args.blend_cfg.scheduler,
            strength: args.blend_cfg.strength,
            feather_px: args.blend_cfg.feather_px,
        };
        tokio::task::block_in_place(|| {
            handle.block_on(crate::pipelines::artefact_blend::blend_files(
                bcfg,
                &args.specs,
                &args.library_dir,
                &files,
                &Default::default(),
                None,
                smart.as_ref(),
                blend_shared_core,
            ))
        })
        .context("artefact_blend::blend_files (plakat post-process)")?;
    }
    Ok(())
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
            // v0.22 phase 11: same negative+preset combination as
            // the main t2i request — the hires refine pass sees a
            // consistent negative.
            prompt: prompt.to_string(),
            negative: resolve_negative(ctx),
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
    shared_core: std::sync::Arc<crate::pipelines::sd_core::SdCore>,
) -> Result<()> {
    let shared_core = Some(shared_core);
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
        cfg.negative = resolve_negative(ctx);
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
    shared_core: std::sync::Arc<crate::pipelines::sd_core::SdCore>,
) -> Result<()> {
    let shared_core = Some(shared_core);
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

    // v0.22 phase 11: expand `{a|b|c}` + `__name__` wildcards against
    // `config.wildcard_dir`. Seeded by `config.seed` for reproducibility.
    let prompt_owned = expand_prompt(ctx, prompt)?;
    let prompt = prompt_owned.as_str();

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-gen-")
        .tempdir()
        .context("creating tempdir for plakat.generate output")?;
    let tmp_path = tmp.path().to_path_buf();

    match PipelineFamily::detect(&alias) {
        PipelineFamily::SdFamily => {
            // v0.23 phase 1: SD-family generate uses the t2i slot.
            // v0.23 phase 2: when `ctx.refiner_enabled` is on and the
            // alias is SDXL, the t2i pipeline loads with the official
            // SDXL refiner UNet (~6 GB download on first run); the
            // schedule splits between base + refiner at
            // `refiner_frac` (default 0.8 = last 20% of steps).
            // Non-SDXL aliases silently downgrade with a warn —
            // gating happens inside `get_or_load_sd_t2i`.

            // v0.23 phase 4: resolve the active style (if any)
            // BEFORE borrowing the pipeline. The resolve produces
            // catalog LoRAs that override the user LoRA stack for
            // this load, plus a trigger phrase that prepends to
            // the prompt and negative_extras that append to the
            // negative. We temporarily swap `ctx.loras` so the
            // loader sees the catalog LoRAs; restored after the
            // pipeline borrow releases. CLI parity:
            // `cli::generate::apply_style` does the same overwrite.
            let style_prep = resolve_style_for_generate(ctx, &alias)?;
            let user_loras_snapshot = if style_prep.is_some() {
                Some(ctx.loras.clone())
            } else {
                None
            };
            // Compose the effective prompt + negative from style.
            let (effective_prompt, effective_negative_extras): (String, String) =
                match style_prep.as_ref() {
                    Some(prep) => (
                        crate::style::prepend_trigger(&prep.trigger, prompt),
                        prep.negative_extras.clone(),
                    ),
                    None => (prompt.to_string(), String::new()),
                };
            // Mutate ctx.loras to the style-resolved set (CLI
            // behavior: style overwrites user LoRAs).
            if let Some(prep) = style_prep.as_ref() {
                ctx.loras = crate::style::parse_resolved_loras(prep)
                    .context("parsing resolved style LoRAs into LoraSpec")?;
            }

            let mut req = build_t2i_gen_request(ctx, &effective_prompt, tmp_path.clone());
            // Append style's negative_extras to the request negative.
            if !effective_negative_extras.is_empty() {
                req.negative = crate::style::combine_negative(&req.negative, &effective_negative_extras);
            }
            // v0.22 phase 5: resolve the script's controlnets to
            // OwnedControl + ControlRequest before borrowing the
            // pipeline. The owned data lives on this frame for the
            // pipeline.generate call's lifetime.
            let control_owned =
                resolve_sd_controlnets(ctx, &alias, req.width, req.height, None)?;
            let control_reqs = controlnets_to_requests(&control_owned);
            // v0.22 phase 7-9: post-process snapshots *before* the
            // cached-pipeline borrow. Run order: artefacts (compose
            // + blend) → hires → adetailer. The CLI bails if
            // --hires-fix is combined with --artefact / --artefact-blend;
            // we mirror that gate here.
            let adargs = AdetailerArgs::from_ctx(ctx, &alias);
            let hargs = HiresArgs::from_ctx(ctx, &alias, &effective_prompt)?;
            let aargs = ArtefactArgs::from_ctx(
                ctx, &alias, &effective_prompt, req.width, req.height,
            );
            if !aargs.is_empty() && hargs.enabled {
                bail!(
                    "plakat.generate: hires-fix doesn't compose with \
                     artefacts in v0.22 — the CLI bails the same way. \
                     Call plakat.hires.disable OR plakat.artefact.clear \
                     before plakat.generate."
                );
            }
            // Scope-bound pipeline borrow so we can restore
            // ctx.loras after the generate call returns.
            let shared_core = {
                let pipeline = ctx.get_or_load_sd_t2i(&alias)?;
                pipeline.generate(&req, &control_reqs)
                    .context("t2i::Pipeline::generate (plakat.generate SD path)")?;
                pipeline.core()
            };
            // Restore user LoRA stack now that the pipeline borrow
            // is released. Subsequent generate calls with the same
            // style cache-hit the pipeline (loaded with style
            // LoRAs); the user-visible LoRA stack returns to what
            // the user actually configured.
            if let Some(snap) = user_loras_snapshot {
                ctx.loras = snap;
            }
            // v0.23 phase 1: post-process helpers take Arc<SdCore>
            // directly (pipeline-agnostic) so they work after either
            // a t2i or portrait pipeline produced the image.
            if !aargs.is_empty() {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_artefacts_sd(&aargs, &rendered, shared_core.clone())?;
            }
            if hargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_hires_sd(&hargs.cfg, &rendered, shared_core.clone())?;
            }
            if adargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_adetailer_sd(&adargs.cfg, &rendered, shared_core)?;
            }
        }
        PipelineFamily::Flux => {
            // v0.23 phase 6: Flux ControlNet wires through the cache
            // at load time. `ctx.controlnets` mutations call
            // `mark_controlnets_changed` which drops the Flux slot,
            // so the next plakat.generate reloads with the current
            // CN stack. Image-only specs (plakat.controlnet.add KIND
            // PATH) supported; auto-annotate bails inside the
            // loader.
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
            if !ctx.artefacts.is_empty() {
                bail!(
                    "plakat.generate: artefacts are SD-family only in v0.22 \
                     phase 9 — the optional blend pass uses portrait::Pipeline. \
                     Call plakat.artefact.clear before plakat.generate on Flux."
                );
            }
            if ctx.style_id.is_some() || ctx.style_ref.is_some() {
                bail!(
                    "plakat.generate: plakat.style.* is SD-family only in \
                     v0.23 phase 4 — Flux style integration isn't wired \
                     in the runtime yet. Call plakat.style.clear before \
                     plakat.generate on Flux."
                );
            }
            let req = build_flux_gen_request(ctx, prompt, tmp_path.clone(), None);
            let pipeline = ctx.get_or_load_flux(&alias)?;
            pipeline.generate(&req)
                .context("flux::Pipeline::generate (plakat.generate Flux path)")?;
        }
        PipelineFamily::Sd3 => {
            // v0.23 phase 7: SD3 ControlNet wires through the cache
            // at load time. Same as Flux (phase 6) — image= specs
            // only; mark_controlnets_changed drops the slot on stack
            // mutations.
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
            if !ctx.artefacts.is_empty() {
                bail!(
                    "plakat.generate: artefacts are SD-family only in v0.22 \
                     phase 9 — the optional blend pass uses portrait::Pipeline. \
                     Call plakat.artefact.clear before plakat.generate on SD3."
                );
            }
            if ctx.style_id.is_some() || ctx.style_ref.is_some() {
                bail!(
                    "plakat.generate: plakat.style.* is SD-family only in \
                     v0.23 phase 4 — SD3 style integration isn't wired in \
                     the runtime yet. Call plakat.style.clear before \
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
    img2img_or_inpaint_one(ctx, prompt, input_path, None, "plakat.img2img")
}

/// v0.23 phase 5: inpaint dispatch. Shares the body with
/// `img2img_one` — the only differences are the `mask` arg + the
/// error-message tag.
pub fn inpaint_one(
    ctx: &mut ScriptCtx,
    prompt: &str,
    input_path: &Path,
    mask_path: &Path,
) -> Result<DynamicImage> {
    img2img_or_inpaint_one(ctx, prompt, input_path, Some(mask_path), "plakat.inpaint")
}

/// Shared body for [`img2img_one`] and [`inpaint_one`].
///
/// `mask_path` is `None` for plain img2img, `Some(...)` for
/// inpaint. The SD-family path threads it into
/// `img2img::Request.mask` (the v0.22 phase 11 `mask_feather` /
/// `mask_invert` config keys are honoured only when this is
/// non-None — they were declared then but unreachable until now).
/// Flux + SD3 inpaint require their fill variants
/// (flux-fill-dev / sd3 mmdit native inpaint) — those bail with
/// a clear message in phase 5; full wiring lands when there's a
/// CLI parity gap that needs it.
fn img2img_or_inpaint_one(
    ctx: &mut ScriptCtx,
    prompt: &str,
    input_path: &Path,
    mask_path: Option<&Path>,
    word_tag: &str,
) -> Result<DynamicImage> {
    let alias = ctx
        .loaded_model()
        .ok_or_else(|| {
            anyhow!(
                "{word_tag}: no model loaded. Call \"sd15\" plakat.load \
                 (or another supported alias) before {word_tag}."
            )
        })?
        .to_string();

    // v0.22 phase 11: wildcard expansion before the working-size
    // resolution (size doesn't depend on prompt, but expanding
    // first keeps the seeded-RNG order consistent across paths).
    let prompt_owned = expand_prompt(ctx, prompt)?;
    let prompt = prompt_owned.as_str();

    // Working size: explicit config wins; else input image dims
    // snapped to /8 (downward). Read config + dims first, then
    // borrow the pipeline mutably.
    let (width, height) = if ctx.config.size_explicit {
        (ctx.config.width, ctx.config.height)
    } else {
        let dims = image::image_dimensions(input_path).with_context(|| {
            format!(
                "reading dimensions of {} for {word_tag} working size",
                input_path.display()
            )
        })?;
        let (w, h) = dims;
        ((w / 8) * 8, (h / 8) * 8)
    };
    if width == 0 || height == 0 {
        bail!(
            "{word_tag}: working size {width}x{height} collapsed to 0 \
             after /8 snap. Input image is too small (< 8 pixels on a side)."
        );
    }

    let tmp = tempfile::Builder::new()
        .prefix("plakat-script-i2i-")
        .tempdir()
        .with_context(|| format!("creating tempdir for {word_tag} output"))?;
    let tmp_path = tmp.path().to_path_buf();

    match PipelineFamily::detect(&alias) {
        PipelineFamily::SdFamily => {
            let req = crate::pipelines::img2img::Request {
                prompt: prompt.to_string(),
                negative: resolve_negative(ctx),
                model: alias.clone(),
                device: ctx.device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                input: input_path.to_path_buf(),
                // v0.23 phase 5: mask threads through when plakat.inpaint
                // is the caller. mask_feather / mask_invert (declared
                // v0.22 phase 11) finally have a mask to act on.
                mask: mask_path.map(|p| p.to_path_buf()),
                mask_feather: ctx.config.mask_feather,
                mask_invert: ctx.config.mask_invert,
                width,
                height,
                count: 1,
                steps: ctx.config.steps,
                guidance: ctx.config.guidance,
                scheduler: ctx.config.scheduler,
                strength: ctx.config.strength,
                seed: ctx.config.seed,
                out_dir: tmp_path.clone(),
                controls: ctx.controlnets.clone(),
            };
            // v0.22 phase 7-9: post-process snapshots before pipeline borrow.
            let adargs = AdetailerArgs::from_ctx(ctx, &alias);
            let hargs = HiresArgs::from_ctx(ctx, &alias, prompt)?;
            let aargs = ArtefactArgs::from_ctx(ctx, &alias, prompt, width, height);
            if !aargs.is_empty() && hargs.enabled {
                bail!(
                    "{word_tag}: hires-fix doesn't compose with artefacts \
                     in v0.22. Disable one before {word_tag}."
                );
            }
            let pipeline = ctx.get_or_load_sd_family(&alias)?;
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "{word_tag}: no tokio runtime in scope (eval must \
                     run on a multi-threaded runtime). Underlying error: {e}"
                )
            })?;
            tokio::task::block_in_place(|| {
                handle.block_on(crate::pipelines::img2img::run_with_pipeline(
                    pipeline, &req,
                ))
            })
            .with_context(|| format!("img2img::run_with_pipeline ({word_tag} SD path)"))?;
            let shared_core = pipeline.core();
            if !aargs.is_empty() {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_artefacts_sd(&aargs, &rendered, shared_core.clone())?;
            }
            if hargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_hires_sd(&hargs.cfg, &rendered, shared_core.clone())?;
            }
            if adargs.enabled {
                let rendered = find_rendered_png(&tmp_path)?;
                apply_adetailer_sd(&adargs.cfg, &rendered, shared_core)?;
            }
        }
        PipelineFamily::Flux => {
            // v0.23 phase 6: Flux ControlNet wires through at load
            // time. See `get_or_load_flux` for the resolve path.
            if ctx.adetailer_enabled {
                bail!(
                    "{word_tag}: ADetailer is SD-family only in v0.22 \
                     phase 7. Call plakat.adetailer.disable before \
                     {word_tag} on Flux."
                );
            }
            if ctx.hires_enabled {
                bail!(
                    "{word_tag}: hires-fix is SD-family only in v0.22 \
                     phase 8. Call plakat.hires.disable before \
                     {word_tag} on Flux."
                );
            }
            if !ctx.artefacts.is_empty() {
                bail!(
                    "{word_tag}: artefacts are SD-family only in v0.22 \
                     phase 9. Call plakat.artefact.clear before \
                     {word_tag} on Flux."
                );
            }
            if mask_path.is_some() {
                // v0.23 phase 5: Flux inpaint requires the
                // flux-fill-dev variant + load-time channel-concat
                // wiring on the img_in projection. That's its own
                // refactor; not in scope for phase 5. Bail with a
                // clear pointer.
                bail!(
                    "plakat.inpaint: Flux inpaint requires the \
                     flux-fill-dev variant + per-load setup; not wired \
                     in v0.23 phase 5. Workaround: use the CLI's \
                     `plakat img2img --model flux-fill-dev --mask MASK` \
                     directly, or stay on SD-family in scripts."
                );
            }
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
                .with_context(|| format!("flux::Pipeline::generate ({word_tag} Flux path)"))?;
        }
        PipelineFamily::Sd3 => {
            // v0.23 phase 7: SD3 ControlNet wires through at load
            // time. See `get_or_load_sd3` for the resolve path.
            if ctx.adetailer_enabled {
                bail!(
                    "{word_tag}: ADetailer is SD-family only in v0.22 \
                     phase 7. Call plakat.adetailer.disable before \
                     {word_tag} on SD3."
                );
            }
            if ctx.hires_enabled {
                bail!(
                    "{word_tag}: hires-fix is SD-family only in v0.22 \
                     phase 8. Call plakat.hires.disable before \
                     {word_tag} on SD3."
                );
            }
            if !ctx.artefacts.is_empty() {
                bail!(
                    "{word_tag}: artefacts are SD-family only in v0.22 \
                     phase 9. Call plakat.artefact.clear before \
                     {word_tag} on SD3."
                );
            }
            let mut req = build_sd3_gen_request(
                ctx,
                prompt,
                tmp_path.clone(),
                Some(input_path.to_path_buf()),
            );
            req.width = width;
            req.height = height;
            // v0.23 phase 5: SD3 / SD3.5 support native RePaint
            // inpaint via the mask field. Thread it through when
            // plakat.inpaint is the caller.
            if let Some(p) = mask_path {
                req.mask = Some(p.to_path_buf());
                req.mask_feather = ctx.config.mask_feather;
                req.mask_invert = ctx.config.mask_invert;
            }
            let pipeline = ctx.get_or_load_sd3(&alias)?;
            pipeline.generate(&req)
                .with_context(|| format!("sd3::Pipeline::generate ({word_tag} SD3 path)"))?;
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

    // v0.24 phase 1: photos come from the multi-photo stack
    // populated by `plakat.portrait.photo.add`. Empty stack →
    // bail loudly so users get a clear "add at least one
    // photo first" error.
    if ctx.portrait_photos.is_empty() {
        bail!(
            "plakat.portrait: no photo configured. Push at least one \
             photo onto the portrait stack with `plakat.portrait.photo.add \
             ( path-or-handle weight -- )` before calling plakat.portrait."
        );
    }
    let photos = ctx.portrait_photos.clone();

    // v0.22 phase 11: wildcard expansion (matches generate_one).
    let prompt_owned = expand_prompt(ctx, prompt)?;
    let prompt = prompt_owned.as_str();

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

    // v0.22 phase 7-9: post-process snapshots before pipeline borrow.
    let adargs = AdetailerArgs::from_ctx(ctx, &alias);
    let hargs = HiresArgs::from_ctx(ctx, &alias, prompt)?;
    let aargs = ArtefactArgs::from_ctx(ctx, &alias, prompt, req.width, req.height);
    if !aargs.is_empty() && hargs.enabled {
        bail!(
            "plakat.portrait: hires-fix doesn't compose with artefacts in \
             v0.22. Disable one before plakat.portrait."
        );
    }
    let pipeline = ctx.get_or_load_sd_family(&alias)?;
    pipeline.generate(&req, &[])
        .context("portrait::Pipeline::generate (plakat.portrait path)")?;
    let shared_core = pipeline.core();
    if !aargs.is_empty() {
        let rendered = find_rendered_png(tmp.path())?;
        apply_artefacts_sd(&aargs, &rendered, shared_core.clone())?;
    }
    if hargs.enabled {
        let rendered = find_rendered_png(tmp.path())?;
        apply_hires_sd(&hargs.cfg, &rendered, shared_core.clone())?;
    }
    if adargs.enabled {
        let rendered = find_rendered_png(tmp.path())?;
        apply_adetailer_sd(&adargs.cfg, &rendered, shared_core)?;
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

    // v0.22 phase 11: aspect-derived size + negative-preset combine.

    #[test]
    fn aspect_to_size_landscape_snaps_to_eight() {
        // 16:9 with base 768 → longer side 1365.33 → snap down to 1360.
        let (w, h) = aspect_to_size("16:9", 768).unwrap();
        assert_eq!(h, 768);
        assert_eq!(w % 8, 0);
        assert!(w > h);
    }

    #[test]
    fn aspect_to_size_portrait_returns_base_as_shorter_side() {
        let (w, h) = aspect_to_size("2:3", 512).unwrap();
        assert_eq!(w, 512);
        assert!(h > w);
        assert_eq!(h % 8, 0);
    }

    #[test]
    fn aspect_to_size_square_keeps_base() {
        let (w, h) = aspect_to_size("1:1", 1024).unwrap();
        assert_eq!(w, 1024);
        assert_eq!(h, 1024);
    }

    #[test]
    fn aspect_to_size_malformed_returns_none() {
        assert!(aspect_to_size("garbage", 768).is_none());
        assert!(aspect_to_size("16:0", 768).is_none());
    }

    // v0.23 phase 3: clip_skip + refiner_frac propagation.

    /// `build_t2i_gen_request` carries the config-layer
    /// clip_skip + refiner_frac through to `t2i::GenRequest`. We
    /// dodge the loaded-pipeline requirement by setting
    /// size_explicit = true so default_size_for_loaded isn't
    /// called.
    #[test]
    fn build_t2i_gen_request_carries_clip_skip_and_refiner_frac() {
        let mut ctx = ScriptCtx {
            device: candle_core::Device::Cpu,
            out_dir: std::env::temp_dir(),
            loaded: None,
            loaded_t2i: None,
            images: Vec::new(),
            config: crate::scripting::config::GenerationConfig::default(),
            loras: Vec::new(),
            controlnets: Vec::new(),
            refiner_enabled: false,
            adetailer_enabled: false,
            hires_enabled: false,
            artefacts: Vec::new(),
            artefact_blend_enabled: false,
            style_id: None,
            style_ref: None,
            portrait_photos: Vec::new(),
        };
        ctx.config.size_explicit = true;
        ctx.config.width = 512;
        ctx.config.height = 512;
        ctx.config.clip_skip = 2;
        ctx.config.refiner_frac = 0.85;

        let req = build_t2i_gen_request(&ctx, "a fox", std::env::temp_dir());
        assert_eq!(req.clip_skip, 2);
        assert_eq!(req.refiner_frac, Some(0.85));
        assert_eq!(req.width, 512);
    }
}
