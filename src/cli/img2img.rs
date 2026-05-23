use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
use crate::pipelines::img2img;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;

/// `plakat img2img <INPUT> --prompt "..."` — re-imagine an existing
/// image at a chosen denoise strength. Supply `--mask` to restrict
/// the changes to a region (inpaint).
#[derive(ClapArgs, Debug)]
pub struct Img2ImgArgs {
    /// Path to the source image. Any format the `image` crate reads.
    pub input: PathBuf,

    /// Text prompt describing the desired output.
    #[arg(long)]
    pub prompt: String,

    /// Negative prompt (things to discourage).
    #[arg(long, default_value = "")]
    pub negative: String,

    /// Optional inpaint mask. When set, only mask=white pixels are
    /// re-painted; mask=black pixels are preserved. Grayscale, RGB
    /// (luminance), or RGBA (alpha channel) all accepted.
    #[arg(long, value_name = "PATH")]
    pub mask: Option<PathBuf>,

    /// Feather radius (pixels) applied to the mask edge. Softens
    /// the inpaint↔preserve transition. Only meaningful with --mask.
    #[arg(long = "mask-feather", default_value_t = 8, value_name = "PX")]
    pub mask_feather: u32,

    /// Invert the mask polarity (treat black as inpaint instead of
    /// white). Use when your mask source uses the opposite convention.
    #[arg(long = "mask-invert", default_value_t = false)]
    pub mask_invert: bool,

    /// img2img strength in [0, 1]. 0.0 = no change, 1.0 = full
    /// re-noise + denoise inside the mask. Default differs by mode:
    /// 0.6 for img2img (whole image), 1.0 for inpaint (--mask set).
    #[arg(long, value_name = "F")]
    pub strength: Option<f32>,

    /// Model: alias or any HF repo id. Aliases:
    ///   `sd15` / `sd21` / `sdxl` / `sdxl-turbo` — regular text-to-image
    ///   UNets. Use with `--mask` for RePaint-style masked img2img.
    ///   `sd15-inpaint` / `sdxl-inpaint` — dedicated 9-channel
    ///   inpainting checkpoints. Trained for inpainting, so they
    ///   preserve the unmasked region natively (no RePaint blending).
    ///   `--mask` is required when picking these.
    /// Flux is not supported by img2img.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// Output size, e.g. 512x512. If absent, the input's dimensions
    /// are snapped to a multiple of 8 (VAE requirement) and used.
    #[arg(long)]
    pub size: Option<Size>,

    /// Number of variations to generate from the same input. Each
    /// gets a fresh seed.
    #[arg(long, short = 'n', default_value_t = 1)]
    pub count: u32,

    /// Denoising steps.
    #[arg(long, default_value_t = 28)]
    pub steps: usize,

    /// Classifier-free guidance scale.
    #[arg(long, default_value_t = 7.5)]
    pub guidance: f64,

    /// Base seed. Subsequent --count outputs use seed+1, seed+2, ...
    /// If omitted, a random seed is picked.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Scheduler. `default` follows the model's preferred scheduler.
    #[arg(long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// LoRA spec(s). Same grammar as `plakat generate --loras`.
    #[arg(long = "loras", value_delimiter = ',')]
    pub loras: Vec<LoraSpec>,

    /// LoRA weight scale multiplier.
    #[arg(long, default_value_t = 1.0)]
    pub lora_scale: f32,

    /// Output directory. Files land as
    /// `plakat-img2img-<seed>.png` or `plakat-inpaint-<seed>.png`.
    #[arg(long, default_value = "./out")]
    pub out: PathBuf,

    /// ControlNet conditioner kind (currently `depth`). Composes
    /// with the img2img / inpaint path — the conditioner guides
    /// every denoise step. Conditioning source: `--control-image PATH`
    /// (pre-rendered), `--control-from PATH` (auto-annotate any
    /// image), or **default**: auto-annotate `<INPUT>`.
    #[arg(long = "control", value_name = "KIND")]
    pub control: Option<crate::pipelines::controlnet::ControlKind>,

    /// Pre-rendered conditioning image for `--control`. Mutually
    /// exclusive with `--control-from`. If neither is set on
    /// `img2img`, the `<INPUT>` image is auto-annotated.
    #[arg(long = "control-image", value_name = "PATH", conflicts_with = "control_from")]
    pub control_image: Option<PathBuf>,

    /// **v0.10**: source image to auto-annotate. Runs the matching
    /// annotator for `--control` and uses the result as the
    /// conditioning. Default for `img2img` when neither
    /// `--control-image` nor this flag is set: use `<INPUT>`.
    #[arg(long = "control-from", value_name = "PATH")]
    pub control_from: Option<PathBuf>,

    /// ControlNet residual scale. Default 1.0.
    #[arg(long = "control-strength", default_value_t = 1.0, value_name = "F")]
    pub control_strength: f32,

    /// Timestep window start in `[0, 1]`. Default 0.0.
    #[arg(long = "control-start", default_value_t = 0.0, value_name = "F")]
    pub control_start: f32,

    /// Timestep window end in `[0, 1]`. Default 1.0. Use `0.5` to
    /// disable ControlNet for the back half of the schedule.
    #[arg(long = "control-end", default_value_t = 1.0, value_name = "F")]
    pub control_end: f32,

    /// **v0.11**: full ControlNet spec, repeatable for multi-ControlNet.
    /// See `plakat generate --control-spec` for grammar. Mutually
    /// exclusive with the legacy single-conditioner flags. When a spec
    /// has neither `image=` nor `from=`, the input image is
    /// auto-annotated (img2img-specific default).
    #[arg(
        long = "control-spec",
        value_name = "SPEC",
        conflicts_with_all = [
            "control", "control_image", "control_from",
            "control_strength", "control_start", "control_end",
        ],
    )]
    pub control_specs: Vec<crate::pipelines::controlnet::ControlSpec>,

    // -------- artefact compositing (mirrors `plakat generate`) --------
    /// Composite a named artefact (PNG cutout) into each output image.
    /// Repeatable. Grammar: `NAME[@ZONE[:SCALE]]`. Same as
    /// `plakat generate --artefact` — see that command for examples.
    #[arg(long = "artefact", value_name = "NAME[@ZONE[:SCALE]]")]
    pub artefacts: Vec<crate::artefacts::ArtefactSpec>,

    /// Override the bundled artefact library directory.
    #[arg(long = "artefact-library", value_name = "DIR")]
    pub artefact_library: Option<PathBuf>,

    /// After alpha-compositing artefacts, run a low-strength masked
    /// img2img pass over the artefact zones to soften the seams.
    /// Reuses the SD backbone loaded for the main img2img/inpaint
    /// pass — no second download or model load.
    #[arg(long = "artefact-blend", default_value_t = false)]
    pub artefact_blend: bool,

    /// Blend strength for `--artefact-blend`. 0.0 = no-op, 0.3 is the
    /// recommended default; higher values let the model redraw the
    /// artefact silhouette and can "fix" it into something unrecognisable.
    #[arg(long = "artefact-blend-strength", default_value_t = 0.3, value_name = "F")]
    pub artefact_blend_strength: f32,

    /// Derive artefact zones from each generated image's own depth +
    /// luminance, instead of the bundled rigid grid. Falls back to
    /// the grid if the depth model load fails.
    #[arg(long = "smart-zones", default_value_t = false)]
    pub smart_zones: bool,
}

pub async fn run(args: Img2ImgArgs, device: Device) -> Result<()> {
    // v0.13 phase 2: Flux.1-Fill-dev is an inpainting-only model.
    // When the user picks it via `--model flux-fill-dev`, route to
    // the Flux pipeline instead of the SD img2img path. The Fill
    // model trains on the mask directly, so we don't need RePaint-
    // style strength blending — every step respects the mask.
    //
    // v0.13 phase 3: standard Flux variants (Dev/Schnell) get the
    // rectified-flow img2img path. Same VAE→lerp(init, noise,
    // strength)→truncated-schedule denoise diffusers uses.
    {
        let variant = crate::pipelines::t2i::Variant::detect(&args.model);
        if variant == crate::pipelines::t2i::Variant::FluxFillDev {
            return run_flux_fill(args, device).await;
        }
        if variant.is_flux() {
            return run_flux_img2img(args, device).await;
        }
        // v0.15 phase 2: SD3.x img2img + inpaint. The MMDiT pipeline
        // already supports the rectified-flow lerp + truncated
        // schedule; this dispatch builds the sd3::Request directly,
        // skipping the SD-family LoRA / refiner / ControlNet plumbing
        // that doesn't apply on MMDiT.
        if variant.is_sd3() {
            return run_sd3_img2img(args, device).await;
        }
    }

    // Strength: 0.6 for img2img, 1.0 for inpaint when not explicit.
    let strength = args
        .strength
        .unwrap_or_else(|| if args.mask.is_some() { 1.0 } else { 0.6 });
    if !(0.0..=1.0).contains(&strength) || !strength.is_finite() {
        anyhow::bail!("strength must be finite in [0, 1], got {strength}");
    }

    // Working resolution: explicit --size > input dims snapped to /8.
    let (width, height) = match args.size {
        Some(s) => (s.w, s.h),
        None => detect_input_size(&args.input)?,
    };
    if width % 8 != 0 || height % 8 != 0 {
        anyhow::bail!(
            "working size {width}x{height} must be a multiple of 8 (VAE constraint); \
             pass --size to override",
        );
    }

    // Pre-resolve the seed at the CLI boundary so the artefact
    // compositor knows which output filenames to read back. Behaviour
    // is bit-equivalent to letting the pipeline pick a random one.
    let seed = Some(args.seed.unwrap_or_else(rand::random));

    // Same `mode_tag` rule the pipeline uses so the file names line up.
    let mode_tag = if args.mask.is_some() { "inpaint" } else { "img2img" };
    let file_prefix = format!("plakat-{mode_tag}");

    // Clone the values the artefact-blend step will need before `args`
    // gets partially moved into `img2img::Request`.
    let out_dir = args.out.clone();
    let count = args.count;
    let prompt = args.prompt.clone();
    let negative = args.negative.clone();
    let model = args.model.clone();
    let loras = args.loras.clone();
    let lora_scale = args.lora_scale;
    let scheduler = args.scheduler;
    let steps = args.steps;
    let guidance = args.guidance;

    let req = img2img::Request {
        prompt: args.prompt,
        negative: args.negative,
        model: args.model,
        device: device.clone(),
        loras: args.loras,
        lora_scale: args.lora_scale,
        input: args.input,
        mask: args.mask,
        mask_feather: args.mask_feather,
        mask_invert: args.mask_invert,
        width,
        height,
        count: args.count,
        steps: args.steps,
        guidance: args.guidance,
        scheduler: args.scheduler,
        strength,
        seed,
        out_dir: args.out,
        controls: crate::pipelines::controlnet::resolve_control_specs(
            args.control_specs,
            args.control,
            args.control_image,
            args.control_from,
            args.control_strength,
            args.control_start,
            args.control_end,
        ),
    };

    // Phase 7d/7e pattern: capture the SD backbone img2img loaded so
    // the optional --artefact-blend pass below reuses it instead of
    // paying for a second multi-GB load.
    let shared_core = img2img::run(req).await?;

    // Composite any --artefact flags onto the generated images.
    let library_dir = args
        .artefact_library
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/artefact_library"));

    // Lazily load depth pipeline if --smart-zones. Warn + fall back to
    // the rigid grid on load failure — same pattern as generate.
    let smart_depth = if args.smart_zones && !args.artefacts.is_empty() {
        match crate::pipelines::depth::DepthPipeline::load(device.clone()).await {
            Ok(p) => Some(p),
            Err(e) => {
                crate::ui::progress::println(&format!(
                    "  warn: --smart-zones requested but depth model load failed ({e}). \
                     Falling back to rigid 4×3 grid.",
                ));
                None
            }
        }
    } else {
        None
    };

    crate::artefacts::composite_onto_seed_range(
        &args.artefacts,
        &library_dir,
        &out_dir,
        seed,
        count,
        &file_prefix,
        width,
        height,
        &Default::default(),
        smart_depth.as_ref(),
    )?;

    // Optional masked img2img blend over the artefact zones.
    if args.artefact_blend && !args.artefacts.is_empty() {
        let files: Vec<PathBuf> = (0..count)
            .map(|i| {
                let s = seed.unwrap_or(0).wrapping_add(i as u64);
                out_dir.join(format!("{file_prefix}-{s}.png"))
            })
            .filter(|p| p.exists())
            .collect();
        crate::pipelines::artefact_blend::blend_files(
            crate::pipelines::artefact_blend::BlendConfig {
                model,
                device,
                loras,
                lora_scale,
                prompt,
                negative,
                image_w: width,
                image_h: height,
                steps,
                guidance,
                scheduler,
                strength: args.artefact_blend_strength,
                feather_px: None,
            },
            &args.artefacts,
            &library_dir,
            &files,
            &Default::default(),
            seed,
            smart_depth.as_ref(),
            Some(shared_core),
        )
        .await?;
    }

    Ok(())
}

/// v0.13 phase 2: dispatch `plakat img2img --model flux-fill-dev` to
/// the Flux pipeline. Flux.1-Fill-dev is inpaint-only — the model
/// trains with mask + masked-latent as input channels, so a mask is
/// required and `--strength` doesn't apply (the schedule is the
/// standard Flux flow-match, not RePaint-style partial denoise).
///
/// Unsupported on this path: `--mask-feather`, `--mask-invert`,
/// `--negative` (Flux has no negative-prompt mechanism today),
/// `--scheduler` (Flux uses flow-matching, not the SD schedulers),
/// `--artefact*`, `--control-spec`. Warn-and-ignore rather than bail.
async fn run_flux_fill(args: Img2ImgArgs, device: Device) -> Result<()> {
    use crate::pipelines::flux;

    let mask = args.mask.ok_or_else(|| {
        anyhow::anyhow!(
            "Flux.1-Fill-dev requires --mask. The model is inpaint-only \
             — without a mask there's nothing to vary."
        )
    })?;

    let (width, height) = match args.size {
        Some(s) => (s.w, s.h),
        None => detect_input_size(&args.input)?,
    };
    if width % 16 != 0 || height % 16 != 0 {
        anyhow::bail!(
            "Flux.1-Fill-dev needs dimensions divisible by 16 (got {width}x{height}); \
             pass --size to override.",
        );
    }

    // Warn-and-ignore for SD-specific flags that don't apply.
    if !args.negative.is_empty() {
        crate::ui::progress::println(
            "  warn: --negative ignored for Flux.1-Fill-dev (Flux has no negative-prompt path).",
        );
    }
    if args.mask_feather != 8 || args.mask_invert {
        crate::ui::progress::println(
            "  warn: --mask-feather / --mask-invert ignored for Flux.1-Fill-dev (Fill trains \
             on the raw binary mask).",
        );
    }
    if args.strength.is_some() {
        crate::ui::progress::println(
            "  warn: --strength ignored for Flux.1-Fill-dev (the mask itself controls what \
             changes; --strength is an SD-only RePaint knob).",
        );
    }
    // Resolve LoRAs. Same path the Flux generate flow uses — see
    // `t2i::run` for the canonical pattern. flux_lora's resolver will
    // silently skip SD-format LoRAs at merge time.
    let mut resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> =
        Vec::with_capacity(args.loras.len());
    for spec in &args.loras {
        resolved_loras.push(spec.resolve().await?);
    }

    let fvar = flux::Variant::FillDev;
    let repo = if args.model.contains('/') {
        args.model.clone()
    } else {
        crate::hf::resolve_alias(&args.model).to_string()
    };

    // Flux defaults: 28 steps, guidance ~30 for Fill (BFL recommends
    // much higher CFG than standard Flux). User can still override.
    let steps_opt = if args.steps == 28 { None } else { Some(args.steps) };
    let guidance_opt = if (args.guidance - 7.5).abs() < f64::EPSILON {
        None
    } else {
        Some(args.guidance)
    };

    // v0.14 phase 5: Fill + ControlNet composition. Resolve the
    // user's --control-spec stack into `FluxControlNetLoad` entries
    // with the same Union Pro v2 routing the generate CLI uses. CN
    // residuals add at the 3072-d hidden state (post `img_in`), so
    // Fill's wider input doesn't affect composition — the same Union
    // weights work for both Dev and Fill.
    let resolved_specs = crate::pipelines::controlnet::resolve_control_specs(
        args.control_specs,
        args.control,
        args.control_image,
        args.control_from,
        args.control_strength,
        args.control_start,
        args.control_end,
    );
    // Tempdir for any auto-annotator PNGs. Held alive across
    // `flux::run` below so the pipeline can read the conditioning
    // files at load time.
    let anno_tmp = tempfile::Builder::new()
        .prefix("plakat-flux-fill-anno-")
        .tempdir()
        .context("creating tempdir for Flux Fill ControlNet auto-annotator output")?;
    let anno_dtype = if matches!(device, candle_core::Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::BF16
    };
    let mut flux_controlnets: Vec<flux::FluxControlNetLoad> =
        Vec::with_capacity(resolved_specs.len());
    for (cn_idx, spec) in resolved_specs.iter().enumerate() {
        let cond_path = match (spec.image.as_ref(), spec.from.as_ref()) {
            (Some(p), None) => p.clone(),
            (None, Some(from_path)) => {
                let anno = crate::pipelines::controlnet_annotator::annotate(
                    spec.kind, from_path, width, height, &device, anno_dtype,
                )
                .await
                .with_context(|| {
                    format!(
                        "auto-annotating {} for Flux Fill ControlNet",
                        spec.kind.slug()
                    )
                })?;
                let out_path = anno_tmp
                    .path()
                    .join(format!("cn{cn_idx}-{}.png", spec.kind.slug()));
                crate::pipelines::t2i::write_annotator_tensor_as_png(&anno, &out_path)?;
                out_path
            }
            (Some(_), Some(_)) => {
                anyhow::bail!(
                    "--control-spec {}: image= and from= are mutually exclusive",
                    spec.kind.slug()
                )
            }
            (None, None) => {
                anyhow::bail!(
                    "--control-spec {}: requires image=PATH or from=PATH on Flux",
                    spec.kind.slug()
                )
            }
        };
        let mut cn_load = crate::pipelines::t2i::flux_controlnet_load_for(
            spec.kind, fvar, spec.strength,
        )?;
        cn_load.conditioning = Some(cond_path);
        cn_load.start = spec.start;
        cn_load.end = spec.end;
        flux_controlnets.push(cn_load);
    }

    flux::run(flux::Request {
        prompt: args.prompt,
        variant: fvar,
        repo,
        width,
        height,
        count: args.count,
        steps: steps_opt,
        guidance: guidance_opt,
        seed: args.seed,
        out_dir: args.out,
        device,
        loras: resolved_loras,
        lora_scale: args.lora_scale,
        // v0.14 phase 5: Fill + CN composes. CN sees the 64ch noise
        // tokens (Fill's 384ch concat happens inside the Flux forward
        // only); residuals add at the hidden state level the same way
        // as on standard Flux.
        controlnets: flux_controlnets,
        // Per-CN conditioning lives on each FluxControlNetLoad now.
        conditioning: None,
        quantize_t5: false,
        init_image: Some(args.input),
        mask: Some(mask),
        // Fill's mask drives the denoise — `--strength` doesn't apply.
        strength: None,
        // Tiled denoise + Fill don't compose in this phase.
        tiled: None,
        // Quant level not surfaced on the img2img CLI yet — defaults
        // (Q4_K_S / Q4_K_M) match v0.13 phase 1.
        flux_quant_level: None,
        t5_quant_level: None,
        // Redux + Fill don't compose (different forward shape).
        redux: false,
        redux_images: Vec::new(),
        // Concept variants (Canny-dev / Depth-dev) aren't routed
        // through the img2img CLI — they go via `plakat generate
        // --model flux-canny-dev --concept-image ...`.
        concept_conditioning: None,
    })
    .await?;
    // Tempdir held until after the awaited generate completes —
    // pipeline reads any auto-annotated PNGs at load time, so the
    // files must survive until then. Dropping explicitly here is
    // cosmetic (would happen on scope exit anyway) but documents
    // the intent.
    drop(anno_tmp);
    Ok(())
}

/// v0.13 phase 3: `plakat img2img --model flux-dev|flux-schnell` →
/// rectified-flow img2img. Same shape as the SD path: VAE-encode the
/// init, lerp with fresh noise at `t = strength`, truncate the
/// schedule so the first step starts at `strength`, then run the
/// standard Flux denoise. `--mask` is rejected here (use
/// `--model flux-fill-dev` for masked Flux inpainting).
///
/// Unsupported on this path (warn-and-ignore): `--negative`,
/// `--mask-feather`, `--mask-invert`, `--scheduler`, `--control-spec`,
/// `--artefact*`. Keeps the entry point usable on the same CLI
/// surface SD users already know.
async fn run_flux_img2img(args: Img2ImgArgs, device: Device) -> Result<()> {
    use crate::pipelines::flux;

    if args.mask.is_some() {
        anyhow::bail!(
            "--mask requires --model flux-fill-dev for Flux inpainting (Fill is the only \
             Flux variant trained with mask conditioning). For standard Flux img2img, \
             drop --mask."
        );
    }

    let (width, height) = match args.size {
        Some(s) => (s.w, s.h),
        None => detect_input_size(&args.input)?,
    };
    if width % 16 != 0 || height % 16 != 0 {
        anyhow::bail!(
            "Flux img2img needs dimensions divisible by 16 (got {width}x{height}); \
             pass --size to override.",
        );
    }

    let strength = args.strength.unwrap_or(0.85);
    if !(0.0..=1.0).contains(&strength) || !strength.is_finite() {
        anyhow::bail!("--strength must be finite in [0, 1], got {strength}");
    }

    if !args.negative.is_empty() {
        crate::ui::progress::println(
            "  warn: --negative ignored for Flux (no negative-prompt mechanism).",
        );
    }
    if !args.control_specs.is_empty() || args.control.is_some() {
        crate::ui::progress::println(
            "  warn: --control-spec / --control on the Flux img2img CLI path isn't \
             wired in this phase. The image init still runs; ControlNet is skipped.",
        );
    }

    let mut resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> =
        Vec::with_capacity(args.loras.len());
    for spec in &args.loras {
        resolved_loras.push(spec.resolve().await?);
    }

    let fvar = match crate::pipelines::t2i::Variant::detect(&args.model) {
        crate::pipelines::t2i::Variant::FluxDev => flux::Variant::Dev,
        _ => flux::Variant::Schnell,
    };
    let repo = if args.model.contains('/') {
        args.model.clone()
    } else {
        crate::hf::resolve_alias(&args.model).to_string()
    };

    let steps_opt = if args.steps == 28 { None } else { Some(args.steps) };
    let guidance_opt = if (args.guidance - 7.5).abs() < f64::EPSILON {
        None
    } else {
        Some(args.guidance)
    };

    flux::run(flux::Request {
        prompt: args.prompt,
        variant: fvar,
        repo,
        width,
        height,
        count: args.count,
        steps: steps_opt,
        guidance: guidance_opt,
        seed: args.seed,
        out_dir: args.out,
        device,
        loras: resolved_loras,
        lora_scale: args.lora_scale,
        controlnets: Vec::new(),
        conditioning: None,
        quantize_t5: false,
        init_image: Some(args.input),
        mask: None,
        strength: Some(strength),
        // Tiled img2img on Flux ships via `plakat generate --tiled` later.
        tiled: None,
        flux_quant_level: None,
        t5_quant_level: None,
        // Redux not exposed on img2img CLI (use `plakat generate` for
        // image-conditioned generation).
        redux: false,
        redux_images: Vec::new(),
        // Concept variants aren't routed through img2img.
        concept_conditioning: None,
    })
    .await?;
    Ok(())
}

/// v0.15 phase 2: SD3 / SD3.5 img2img + inpaint dispatch.
///
/// Builds an `sd3::Request` directly from the CLI args and runs it.
/// MMDiT doesn't carry the SD-family extras (refiner / ControlNet /
/// LoRA — those land in later phases), so we explicitly bail when the
/// user passes flags that don't apply on SD3 yet. That's friendlier
/// than silently ignoring `--loras` on an SD3 model.
async fn run_sd3_img2img(args: Img2ImgArgs, device: Device) -> Result<()> {
    use crate::pipelines::{sd3, t2i};

    if args.control.is_some()
        || args.control_image.is_some()
        || args.control_from.is_some()
    {
        anyhow::bail!(
            "--control / --control-image / --control-from aren't wired for SD3 yet \
             (v0.15 phase 6 deferred)."
        );
    }
    // v0.15 phase 3: SD3 + LoRA composes. Resolved at sd3::Pipeline::load
    // via the tempfile merge path; nothing else to gate here.
    let variant = t2i::Variant::detect(&args.model);
    let sd3_variant = match variant {
        t2i::Variant::Sd3Medium => sd3::Variant::Sd3Medium,
        t2i::Variant::Sd35Medium => sd3::Variant::Sd35Medium,
        t2i::Variant::Sd35Large => sd3::Variant::Sd35Large,
        t2i::Variant::Sd35LargeTurbo => sd3::Variant::Sd35LargeTurbo,
        _ => unreachable!("dispatch ensures variant.is_sd3()"),
    };

    let (width, height) = match args.size {
        Some(s) => (s.w, s.h),
        None => detect_input_size(&args.input)?,
    };

    let repo = if args.model.contains('/') {
        args.model.clone()
    } else {
        crate::hf::resolve_alias(&args.model).to_string()
    };

    sd3::run(sd3::Request {
        prompt: args.prompt,
        negative: args.negative,
        variant: sd3_variant,
        repo,
        width,
        height,
        count: args.count,
        // Only forward `steps` / `guidance` if the user moved them off
        // the CLI defaults; otherwise let sd3 pick the per-variant
        // recommended values (Turbo wants 4 steps + guidance 0, Large
        // / Medium want 28 + 4.5, etc.).
        steps: if args.steps == 28 {
            None
        } else {
            Some(args.steps)
        },
        guidance: if (args.guidance - 7.5).abs() < f64::EPSILON {
            None
        } else {
            Some(args.guidance)
        },
        seed: args.seed,
        out_dir: args.out,
        device,
        init_image: Some(args.input),
        mask: args.mask,
        mask_feather: args.mask_feather,
        mask_invert: args.mask_invert,
        strength: args.strength,
        // v0.15 phase 3: SD3 LoRA — merged into the MMDiT tempfile
        // at Pipeline::load. Empty Vec = no merge (byte-identical to
        // the phase-2 behaviour).
        loras: args.loras,
        lora_scale: args.lora_scale,
    })
    .await
}

/// Read the input's actual dimensions and round each axis DOWN to
/// the nearest multiple of 8 (the VAE downsample factor). Avoids
/// silently introducing fractional-pixel resizes the user didn't
/// ask for.
fn detect_input_size(path: &std::path::Path) -> Result<(u32, u32)> {
    let (w, h) = image::image_dimensions(path)
        .with_context(|| format!("reading dimensions of {}", path.display()))?;
    let snap = |x: u32| (x / 8) * 8;
    let sw = snap(w).max(8);
    let sh = snap(h).max(8);
    Ok((sw, sh))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn detect_input_size_snaps_to_eight() {
        let img = RgbImage::from_pixel(513, 800, image::Rgb([0, 0, 0]));
        let tmp = std::env::temp_dir().join("plakat_img2img_size_test.png");
        img.save(&tmp).unwrap();
        let (w, h) = detect_input_size(&tmp).unwrap();
        // 513 → 512 (rounded down), 800 stays at 800.
        assert_eq!((w, h), (512, 800));
    }
}
