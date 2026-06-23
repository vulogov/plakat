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

    /// v0.19: bundled negative-prompt preset. See
    /// `plakat generate --negative-preset` for the full list.
    /// Combined with `--negative` (preset first, user appended).
    #[arg(long = "negative-preset", value_name = "NAME")]
    pub negative_preset: Option<String>,

    /// **v0.25**: art-medium preset (`ink-wash` / `watercolor` /
    /// `oil-painting` / `charcoal` / `pencil` / `chalk-pastel` /
    /// `linocut` / `gouache`). Composes the prompt, suggests
    /// sampler/steps/guidance, and (when `--lora` is empty)
    /// auto-discovers a matching LoRA from Civitai → HF Hub →
    /// local cache. Override-only: explicit flags always win.
    /// Applies to inpaint too (use `--mask`).
    #[arg(long = "look", value_name = "NAME")]
    pub look: Option<String>,

    /// **v0.25**: subject-domain preset (`anime`). Independent
    /// axis from `--look`; composes additively.
    #[arg(long = "genre", value_name = "NAME")]
    pub genre: Option<String>,

    /// **v0.25**: skip remote LoRA discovery for `--look` /
    /// `--genre` (use only cache + local scan).
    #[arg(long, default_value_t = false)]
    pub offline: bool,

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

    /// Output size, e.g. 512x512. Resolution order:
    ///   1. `--size WxH` — explicit, wins over everything.
    ///   2. `--aspect 16:9 --base 1024` — derived ratio + base
    ///      resolution (the longer side becomes `base * ratio`).
    ///   3. (default) the input image's dims snapped to /8.
    /// Multiple-of-8 round-down is applied to (1) and (2) as well —
    /// VAE constraint.
    #[arg(long)]
    pub size: Option<Size>,

    /// v0.18: aspect ratio (e.g. `16:9`, `9:16`, `1:1`, `4:3`)
    /// paired with `--base`. Mutually exclusive with `--size`. When
    /// neither flag is set, the input image's dimensions are used.
    #[arg(long, conflicts_with = "size")]
    pub aspect: Option<String>,

    /// v0.18: base resolution used with `--aspect` (the shorter
    /// side; the longer side becomes `base * ratio`). Ignored when
    /// `--size` or no aspect override is set. SD 1.5 defaults to 512
    /// in the rest of the codebase; we pick 1024 here as the modern
    /// default matching SDXL / Flux / SD3.
    #[arg(long, default_value_t = 1024)]
    pub base: u32,

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

    /// Stable Cascade decoder (Stage B) CFG scale, decoupled from
    /// `--guidance`. Default 1.1. Cascade-only.
    #[arg(long = "decoder-guidance", default_value_t = 1.1)]
    pub decoder_guidance: f64,

    /// Stable Cascade faithful img2img: also condition Stage C's
    /// semantic prior on the init image (CLIP ViT-L/14), not just Stage
    /// B's VAE seed. Pulls the output's *content* toward the init — use
    /// when plain img2img drifts off-subject at higher strengths. Loads
    /// the `image_encoder/` from the Cascade repo. Cascade-only.
    #[arg(long = "faithful", default_value_t = false)]
    pub faithful: bool,

    /// Base seed. Subsequent --count outputs use seed+1, seed+2, ...
    /// If omitted, a random seed is picked.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Scheduler. `default` follows the model's preferred scheduler.
    #[arg(long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// v0.16 phase 5: directory holding `<name>.txt` wildcard files
    /// for `__name__` prompt expansion. Inline `{a|b|c}` alternation
    /// works without this flag. RNG is seeded from `--seed` when set.
    #[arg(long = "wildcard-dir", value_name = "DIR")]
    pub wildcard_dir: Option<PathBuf>,

    /// v0.16 phase 10: tiled MultiDiffusion-style denoise for SD3
    /// img2img / inpaint. Composes with `--tiled` only on SD3
    /// variants — SD 1.5 / SDXL img2img doesn't share the SD3
    /// rectified-flow tile path. Drop `--tiled` on SD-family
    /// models or wait for a follow-up phase.
    #[arg(long = "tiled", default_value_t = false)]
    pub tiled: bool,

    /// v0.16 phase 10: tile side length in pixels. Default 1024.
    /// Must be a multiple of 16 (SD3 patch-2 × VAE-8). See the
    /// SD3 tutorial §7 for per-variant pos_embed_max_size limits.
    #[arg(long = "tile-size", default_value_t = 1024, value_name = "PX")]
    pub tile_size: u32,

    /// v0.16 phase 10: stride between tile origins in pixels.
    /// Default 768 — 256 px overlap with 1024 px tiles. Smaller
    /// stride = more overlap = smoother seams = more compute.
    #[arg(long = "tile-stride", default_value_t = 768, value_name = "PX")]
    pub tile_stride: u32,

    /// LoRA spec(s). Repeatable — same grammar as `plakat generate --lora`.
    #[arg(long = "lora")]
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

    /// v0.18 phase 2: with `--count N > 1`, also write a single
    /// `<prefix>-grid-<base-seed>.png` combining all N outputs in a
    /// near-square layout. Per-image PNGs are written as usual
    /// alongside. The prefix tracks the backbone (e.g.
    /// `plakat-img2img-grid-…`, `plakat-flux-grid-…`,
    /// `plakat-sd3-inpaint-grid-…`).
    #[arg(long = "grid", default_value_t = false)]
    pub grid: bool,

    /// v0.18 phase 2: column count for `--grid`. Default is
    /// `ceil(sqrt(count))` — 4 → 2×2, 6 → 3×2, 9 → 3×3, 16 → 4×4.
    /// Ignored when `--grid` is off.
    #[arg(long = "grid-cols", value_name = "N")]
    pub grid_cols: Option<usize>,

    /// v0.18 phase 2: padding (px) between grid cells. Default 0
    /// (flush). Higher values insert a white border between cells.
    /// Ignored when `--grid` is off.
    #[arg(long = "grid-padding", default_value_t = 0, value_name = "PX")]
    pub grid_padding: u32,

    /// v0.18 phase 2b: on `--model flux-kontext-dev`, snap `--size`
    /// (or the input's native dims) to the closest of 17 BFL-
    /// recommended Kontext resolutions before VAE encoding. Off by
    /// default. Ignored on every other model.
    #[arg(long = "kontext-bucket", default_value_t = false)]
    pub kontext_bucket: bool,
}

pub async fn run(mut args: Img2ImgArgs, device: Device) -> Result<()> {
    // v0.19: resolve --negative-preset first so the combined
    // negative flows into wildcard expansion + every dispatch arm.
    args.negative = crate::prompt::negative_presets::combine(
        args.negative_preset.as_deref(),
        &args.negative,
    )?;

    // v0.16 phase 5: wildcard expansion before dispatching to any
    // model-specific path. Same RNG-seeding rules as the generate
    // CLI — seeded from `--seed` when set, OS entropy otherwise.
    expand_img2img_wildcards(&mut args)?;
    // v0.18: A1111 inline <lora:name[:weight]> extraction. Runs
    // AFTER wildcard expansion so `<lora:{styleA|styleB}>` resolves
    // a concrete name first, BEFORE the dispatch arms so every
    // variant (SD-family, Flux Fill, Flux img2img, Flux Kontext,
    // SD3) sees the same cleaned prompt + LoRA stack.
    if crate::prompt::lora_tags::has_lora_tags(&args.prompt) {
        let (cleaned, extracted) = crate::prompt::lora_tags::extract(&args.prompt)?;
        if !extracted.is_empty() {
            tracing::info!(
                target: "plakat",
                "Extracted {} inline <lora:> tag(s) from img2img prompt",
                extracted.len()
            );
            for ex in extracted.into_iter().rev() {
                args.loras.insert(0, ex.spec);
            }
            args.prompt = cleaned;
        }
    }
    if crate::prompt::lora_tags::has_lora_tags(&args.negative) {
        let (cleaned, _dropped) =
            crate::prompt::lora_tags::extract(&args.negative)?;
        args.negative = cleaned;
    }

    // v0.25 phase 6: --look / --genre presets + auto-LoRA discovery.
    // Lands before the dispatch arms so SD-family / Flux Fill /
    // Flux img2img / SD3 all see the same composed prompt + LoRA
    // stack. Also covers the inpaint variant (img2img --mask) since
    // inpaint is a flag on this same path.
    if args.look.is_some() || args.genre.is_some() {
        use crate::preset::{GenerationParams, apply_presets_with_discovery};
        use crate::pipelines::scheduler::SchedulerKind;
        use std::str::FromStr;

        let mut params = GenerationParams {
            prompt: args.prompt.clone(),
            negative: args.negative.clone(),
            // img2img clap defaults: steps=28, guidance=7.5 (match generate).
            steps: (args.steps != 28).then_some(args.steps),
            guidance: ((args.guidance - 7.5).abs() >= f64::EPSILON)
                .then_some(args.guidance),
            scheduler: (!matches!(args.scheduler, SchedulerKind::Default))
                .then(String::new),
        };
        apply_presets_with_discovery(
            args.look.as_deref(),
            args.genre.as_deref(),
            args.offline,
            crate::preset::discovery::BaseFamily::from_model_arg(&args.model),
            false,
            &mut params,
            &mut args.loras,
        )
        .await?;

        args.prompt = params.prompt;
        args.negative = params.negative;
        if let Some(s) = params.steps {
            args.steps = s;
        }
        if let Some(g) = params.guidance {
            args.guidance = g;
        }
        if let Some(sched) = params.scheduler.filter(|s| !s.is_empty()) {
            args.scheduler =
                SchedulerKind::from_str(&sched).unwrap_or(SchedulerKind::Default);
        }
    }

    // v0.16 phase 10: --tiled is SD3 img2img / inpaint only. SD 1.5
    // / SDXL img2img uses a different (UNet) backbone and doesn't
    // share the rectified-flow tiled path. Flux Fill ignores
    // tiled at the dispatcher boundary too. Catch the combo
    // before either backbone gets a confusing partial config.
    if args.tiled {
        let variant = crate::pipelines::t2i::Variant::detect(&args.model);
        if !variant.is_sd3() {
            anyhow::bail!(
                "`plakat img2img --tiled` is wired for SD3 / SD3.5 variants only. \
                 Got --model {} which routes through the {} backbone — drop \
                 --tiled, or switch to --model sd35-medium / sd35-large for \
                 tiled rectified-flow img2img + inpaint.",
                args.model,
                if variant.is_flux() {
                    "Flux"
                } else {
                    "SD UNet"
                }
            );
        }
    }
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
        // v0.18 phase 2b: Flux.1-Kontext-dev. On `plakat img2img`,
        // the natural mapping is "input is the reference" — the
        // input arg flows into the pipeline's concept_conditioning
        // slot (Kontext's seq-concat path), NOT into init_image
        // (which feeds the rectified-flow lerp used by flux-dev
        // img2img). Routed before the generic is_flux() arm so the
        // Dev img2img path doesn't claim Kontext requests.
        if variant == crate::pipelines::t2i::Variant::FluxKontextDev {
            return run_flux_kontext(args, device).await;
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
        // v0.38 phase 4: Stable Cascade img2img. Routes to
        // `cascade::run_img2img` — Stage A encodes the input image,
        // Stage C runs full text→effnet, Stage B denoises a
        // strength-truncated schedule. Mask / tiled are not yet
        // supported on Cascade; bail loud when those flags combine.
        if variant.is_cascade() {
            if args.mask.is_some() {
                anyhow::bail!(
                    "Stable Cascade img2img doesn't yet support `--mask` \
                     (inpaint). Drop the mask or use SD3 / Flux Fill for \
                     mask-aware inpaint."
                );
            }
            if args.tiled {
                anyhow::bail!(
                    "Stable Cascade img2img doesn't support `--tiled` \
                     (the 3-stage architecture's working resolution is fixed)."
                );
            }
            return run_cascade_img2img(args, device).await;
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
    let (width, height) = resolve_img2img_size(&args)?;
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
    // v0.18 phase 2: grid args captured early — the SD-family path
    // moves `args` into the `img2img::Request` below, so the grid
    // block at the end of run() needs them held aside. (Flux + SD3
    // arms read them directly off the still-owned `args`.)
    let grid_enabled = args.grid;
    let grid_cols = args.grid_cols;
    let grid_padding = args.grid_padding;
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

    // v0.18 phase 2: bundle the per-image SD-family outputs into one
    // grid PNG. Runs LAST so the artefact composite + optional blend
    // are reflected in the grid cells. No-op when --count is 1.
    if grid_enabled {
        if let Some((gw, gh, path)) = crate::imaging::grid::compose_grid_from_seed_range(
            &out_dir,
            &file_prefix,
            seed.unwrap_or(0),
            count,
            grid_cols,
            grid_padding,
        )? {
            crate::ui::progress::println(&format!(
                "✓ grid {gw}x{gh} → {}",
                path.display()
            ));
        }
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
async fn run_flux_fill(mut args: Img2ImgArgs, device: Device) -> Result<()> {
    // v0.18 phase 2: pre-resolve the seed so the post-dispatch grid
    // block can find the per-image PNGs by deterministic filename;
    // capture grid-relevant fields before `args` gets partially moved
    // into the flux::Request below.
    args.seed = Some(args.seed.unwrap_or_else(rand::random));
    let grid_enabled = args.grid;
    let grid_cols = args.grid_cols;
    let grid_padding = args.grid_padding;
    let grid_out_dir = args.out.clone();
    let grid_seed = args.seed.unwrap_or(0);
    let grid_count = args.count;
    use crate::pipelines::flux;

    // Resolve size BEFORE moving args.mask out (the ok_or_else
    // partially moves args, blocking subsequent &args borrows).
    let (width, height) = resolve_img2img_size(&args)?;

    let mask = args.mask.ok_or_else(|| {
        anyhow::anyhow!(
            "Flux.1-Fill-dev requires --mask. The model is inpaint-only \
             — without a mask there's nothing to vary."
        )
    })?;
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
        // v0.18 phase 2b: Kontext bucket only matters for Kontext;
        // these img2img arms don't route through Kontext.
        kontext_bucket: false,
        // v0.20: WebP isn't surfaced at the img2img CLI yet.
        // Defaults to PNG; pipeline-level filename construction
        // honours this field.
        output_format: crate::imaging::io::OutputFormat::Png,
    })
    .await?;
    // Tempdir held until after the awaited generate completes —
    // pipeline reads any auto-annotated PNGs at load time, so the
    // files must survive until then. Dropping explicitly here is
    // cosmetic (would happen on scope exit anyway) but documents
    // the intent.
    drop(anno_tmp);

    // v0.18 phase 2: compose a grid of the per-image flux outputs.
    if grid_enabled {
        if let Some((gw, gh, path)) = crate::imaging::grid::compose_grid_from_seed_range(
            &grid_out_dir,
            "plakat-flux",
            grid_seed,
            grid_count,
            grid_cols,
            grid_padding,
        )? {
            crate::ui::progress::println(&format!(
                "✓ grid {gw}x{gh} → {}",
                path.display()
            ));
        }
    }
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
async fn run_flux_img2img(mut args: Img2ImgArgs, device: Device) -> Result<()> {
    // v0.18 phase 2: pre-resolve the seed + capture grid-relevant
    // fields before `args` gets partially moved into flux::Request.
    args.seed = Some(args.seed.unwrap_or_else(rand::random));
    let grid_enabled = args.grid;
    let grid_cols = args.grid_cols;
    let grid_padding = args.grid_padding;
    let grid_out_dir = args.out.clone();
    let grid_seed = args.seed.unwrap_or(0);
    let grid_count = args.count;
    use crate::pipelines::flux;

    if args.mask.is_some() {
        anyhow::bail!(
            "--mask requires --model flux-fill-dev for Flux inpainting (Fill is the only \
             Flux variant trained with mask conditioning). For standard Flux img2img, \
             drop --mask."
        );
    }

    let (width, height) = resolve_img2img_size(&args)?;
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
        // v0.18 phase 2b: Kontext bucket only matters for Kontext;
        // these img2img arms don't route through Kontext.
        kontext_bucket: false,
        // v0.20: WebP isn't surfaced at the img2img CLI yet.
        // Defaults to PNG; pipeline-level filename construction
        // honours this field.
        output_format: crate::imaging::io::OutputFormat::Png,
    })
    .await?;

    // v0.18 phase 2: compose a grid of the per-image flux outputs.
    if grid_enabled {
        if let Some((gw, gh, path)) = crate::imaging::grid::compose_grid_from_seed_range(
            &grid_out_dir,
            "plakat-flux",
            grid_seed,
            grid_count,
            grid_cols,
            grid_padding,
        )? {
            crate::ui::progress::println(&format!(
                "✓ grid {gw}x{gh} → {}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// v0.18 phase 2b: `plakat img2img --model flux-kontext-dev` → Kontext
/// reference-image editing. The `input` positional becomes the
/// reference image fed into Kontext's seq-concat path (NOT the
/// rectified-flow init that `run_flux_img2img` uses). Prompt
/// describes the edit; output appears alongside the prompt-only
/// generate path's outputs as `plakat-flux-<seed>.png`.
async fn run_flux_kontext(mut args: Img2ImgArgs, device: Device) -> Result<()> {
    use crate::pipelines::flux;

    // Pre-resolve seed + capture grid fields before `args` is moved.
    args.seed = Some(args.seed.unwrap_or_else(rand::random));
    let grid_enabled = args.grid;
    let grid_cols = args.grid_cols;
    let grid_padding = args.grid_padding;
    let grid_out_dir = args.out.clone();
    let grid_seed = args.seed.unwrap_or(0);
    let grid_count = args.count;
    let kontext_bucket = args.kontext_bucket;

    if args.mask.is_some() {
        anyhow::bail!(
            "--mask isn't supported on flux-kontext-dev. Kontext is an editing model, \
             not an inpainter — describe the edit in --prompt and let the model decide \
             where to apply it. For region-restricted edits use --model flux-fill-dev."
        );
    }
    if args.strength.is_some() {
        crate::ui::progress::println(
            "  warn: --strength ignored on flux-kontext-dev (Kontext has no \
             rectified-flow init lerp — the reference flows through the seq-concat \
             path instead).",
        );
    }
    if !args.negative.is_empty() {
        crate::ui::progress::println(
            "  warn: --negative ignored for Flux (no negative-prompt mechanism).",
        );
    }
    if !args.control_specs.is_empty() || args.control.is_some() {
        crate::ui::progress::println(
            "  warn: --control-spec / --control aren't wired on flux-kontext-dev \
             yet — the reference image already drives layout. Skipping.",
        );
    }

    // Working resolution. The reference image is VAE-encoded at the
    // requested (w, h); --kontext-bucket snaps before VAE encoding.
    let (width, height) = resolve_img2img_size(&args)?;
    if width % 16 != 0 || height % 16 != 0 {
        anyhow::bail!(
            "Flux Kontext needs dimensions divisible by 16 (got {width}x{height}); \
             pass --size to override or use --kontext-bucket to auto-snap.",
        );
    }

    let mut resolved_loras: Vec<crate::pipelines::lora::ResolvedLora> =
        Vec::with_capacity(args.loras.len());
    for spec in &args.loras {
        resolved_loras.push(spec.resolve().await?);
    }

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
        variant: flux::Variant::KontextDev,
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
        // Crucial: input goes to concept_conditioning (Kontext's
        // seq-concat path), NOT init_image (which would re-route
        // into the rectified-flow lerp the Dev img2img uses).
        init_image: None,
        mask: None,
        strength: None,
        tiled: None,
        flux_quant_level: None,
        t5_quant_level: None,
        redux: false,
        redux_images: Vec::new(),
        concept_conditioning: Some(args.input),
        kontext_bucket,
        // v0.20: WebP isn't surfaced at the img2img CLI yet.
        output_format: crate::imaging::io::OutputFormat::Png,
    })
    .await?;

    if grid_enabled {
        if let Some((gw, gh, path)) = crate::imaging::grid::compose_grid_from_seed_range(
            &grid_out_dir,
            "plakat-flux",
            grid_seed,
            grid_count,
            grid_cols,
            grid_padding,
        )? {
            crate::ui::progress::println(&format!(
                "✓ grid {gw}x{gh} → {}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// v0.15 phase 2: SD3 / SD3.5 img2img + inpaint dispatch.
///
/// Builds an `sd3::Request` directly from the CLI args and runs it.
/// MMDiT doesn't carry the SD-family extras (refiner / ControlNet /
/// LoRA — those land in later phases), so we explicitly bail when the
/// user passes flags that don't apply on SD3 yet. That's friendlier
/// than silently ignoring `--loras` on an SD3 model.
/// v0.38 phase 4: Stable Cascade img2img CLI runner. Mirrors
/// `run_sd3_img2img` in shape — pulls args into a
/// `cascade::RunImg2imgRequest`, dispatches to `cascade::run_img2img`.
/// Inherits the 2/3 + 1/3 step split from t2i (no dedicated
/// `--stage-c-steps` / `--stage-b-steps` on img2img yet — that's a
/// v0.39 follow-up). Strength default 0.6 matches every other
/// img2img path.
async fn run_cascade_img2img(mut args: Img2ImgArgs, device: Device) -> Result<()> {
    args.seed = Some(args.seed.unwrap_or_else(rand::random));
    let strength = args.strength.unwrap_or(0.6).clamp(0.0, 1.0);
    if !strength.is_finite() {
        anyhow::bail!("Cascade img2img strength must be finite in [0, 1], got {strength}");
    }
    let stage_c_steps = (args.steps * 2).div_ceil(3).max(1);
    let stage_b_steps = args.steps.saturating_sub(stage_c_steps).max(1);
    // Cascade output is square. Derive the dim from --size (width);
    // bail on non-square; default 1024.
    let output_dim = match args.size {
        Some(sz) => {
            anyhow::ensure!(
                sz.w == sz.h,
                "Stable Cascade output is square; got --size {}x{}",
                sz.w, sz.h
            );
            sz.w
        }
        None => 1024,
    };
    // v0.41 phase 4c: a single canny control spec may be supplied
    // (auto-resolves the CN from the repo).
    let control_spec = crate::pipelines::controlnet::resolve_control_specs(
        args.control_specs,
        args.control,
        args.control_image,
        args.control_from,
        args.control_strength,
        args.control_start,
        args.control_end,
    )
    .into_iter()
    .next();
    crate::pipelines::cascade::run_img2img(
        crate::pipelines::cascade::RunImg2imgRequest {
            model: args.model,
            device,
            init_image: args.input,
            prompt: args.prompt,
            negative: args.negative,
            output_dim,
            stage_c_steps,
            stage_b_steps,
            strength,
            guidance: args.guidance,
            decoder_guidance: args.decoder_guidance,
            seed: args.seed,
            scheduler: args.scheduler,
            out_dir: args.out,
            count: args.count,
            loras: args.loras,
            lora_scale: args.lora_scale,
            control_spec,
            controlnet_weights: None,
            faithful: args.faithful,
        },
    )
    .await
}

async fn run_sd3_img2img(mut args: Img2ImgArgs, device: Device) -> Result<()> {
    // v0.18 phase 2: pre-resolve the seed + capture grid-relevant
    // fields (including mask presence for the filename mode tag)
    // before `args` gets partially moved into sd3::Request.
    args.seed = Some(args.seed.unwrap_or_else(rand::random));
    let grid_enabled = args.grid;
    let grid_cols = args.grid_cols;
    let grid_padding = args.grid_padding;
    let grid_out_dir = args.out.clone();
    let grid_seed = args.seed.unwrap_or(0);
    let grid_count = args.count;
    let grid_mode_tag = if args.mask.is_some() { "inpaint" } else { "img2img" };
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

    let (width, height) = resolve_img2img_size(&args)?;

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
        // v0.16 phase 10: SD3 tiled img2img / inpaint. Composes
        // with the rectified-flow init lerp + per-step RePaint
        // mask blend. Drop `--tiled` to get the single-pass path
        // (~25-50 % faster on canvases that fit a single tile).
        tiled: if args.tiled {
            Some(crate::pipelines::tiled::TiledConfig {
                tile_size: args.tile_size,
                stride: args.tile_stride,
            })
        } else {
            None
        },
        regions: Vec::new(),
        // v0.16 phase 3e: SD3 img2img path doesn't take
        // --control-spec — the CN integration lives on the t2i
        // dispatch only. The img2img CLI doesn't surface --control*,
        // so an empty Vec here is the only valid value.
        controlnets: Vec::new(),
        // v1.10.0: img2img CLI doesn't surface --embedding for SD3 yet.
        embeddings: Vec::new(),
        // v0.20: WebP isn't surfaced at the img2img CLI yet — SD3
        // outputs default to PNG. The pipeline layer honours this
        // when set.
        output_format: crate::imaging::io::OutputFormat::Png,
    })
    .await?;

    // v0.18 phase 2: compose a grid of the per-image SD3 outputs.
    // Prefix tracks the pipeline's own filename convention
    // (`plakat-sd3-{img2img|inpaint}-{seed}.png`).
    if grid_enabled {
        let file_prefix = format!("plakat-sd3-{grid_mode_tag}");
        if let Some((gw, gh, path)) = crate::imaging::grid::compose_grid_from_seed_range(
            &grid_out_dir,
            &file_prefix,
            grid_seed,
            grid_count,
            grid_cols,
            grid_padding,
        )? {
            crate::ui::progress::println(&format!(
                "✓ grid {gw}x{gh} → {}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// Read the input's actual dimensions and round each axis DOWN to
/// the nearest multiple of 8 (the VAE downsample factor). Avoids
/// silently introducing fractional-pixel resizes the user didn't
/// ask for.
/// v0.16 phase 5: expand `{a|b|c}` + `__name__` wildcards in the
/// img2img prompt + negative. Shared by every img2img dispatch arm
/// (SD-family, Flux, Flux Fill, SD3) via the single entry point.
fn expand_img2img_wildcards(args: &mut Img2ImgArgs) -> Result<()> {
    use rand::SeedableRng;
    let dir = args.wildcard_dir.as_deref();
    let mut rng: rand::rngs::StdRng = match args.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };
    let new_prompt = crate::prompt::wildcards::expand(&args.prompt, dir, &mut rng)?;
    if new_prompt != args.prompt {
        tracing::info!(
            target: "plakat",
            "Wildcard-expanded prompt: {new_prompt}"
        );
        args.prompt = new_prompt;
    }
    if !args.negative.is_empty() {
        let new_neg = crate::prompt::wildcards::expand(&args.negative, dir, &mut rng)?;
        if new_neg != args.negative {
            tracing::info!(
                target: "plakat",
                "Wildcard-expanded negative: {new_neg}"
            );
            args.negative = new_neg;
        }
    }
    Ok(())
}

fn detect_input_size(path: &std::path::Path) -> Result<(u32, u32)> {
    let (w, h) = image::image_dimensions(path)
        .with_context(|| format!("reading dimensions of {}", path.display()))?;
    let snap = |x: u32| (x / 8) * 8;
    let sw = snap(w).max(8);
    let sh = snap(h).max(8);
    Ok((sw, sh))
}

/// v0.18: img2img size resolution priority — `--size > --aspect
/// + --base > input dims`. Centralises the order so the five
/// dispatch arms (SD-family, Flux Fill, Flux img2img, Flux Kontext,
/// SD3) all behave identically and the `--aspect` flag composes
/// uniformly. The crate-wide `imaging::sizes::resolve` does the
/// `--size > --aspect` decision; we wrap it with the input-fallback
/// branch when both flags are unset.
fn resolve_img2img_size(args: &Img2ImgArgs) -> Result<(u32, u32)> {
    if args.size.is_none() && args.aspect.is_none() {
        return detect_input_size(&args.input);
    }
    crate::imaging::sizes::resolve(args.size, args.aspect.as_deref(), args.base)
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

    // v0.18 — img2img --aspect resolution priority.

    fn mk_args(input: PathBuf) -> Img2ImgArgs {
        Img2ImgArgs {
            input,
            prompt: "test".into(),
            negative: String::new(),
            negative_preset: None,
            look: None,
            genre: None,
            offline: false,
            mask: None,
            mask_feather: 8,
            mask_invert: false,
            strength: None,
            model: "sd15".into(),
            size: None,
            aspect: None,
            base: 1024,
            count: 1,
            steps: 28,
            guidance: 7.5,
            decoder_guidance: 1.1,
            faithful: false,
            seed: None,
            scheduler: crate::pipelines::scheduler::SchedulerKind::Default,
            loras: Vec::new(),
            lora_scale: 1.0,
            out: PathBuf::from("./out"),
            control: None,
            control_image: None,
            control_from: None,
            control_strength: 1.0,
            control_start: 0.0,
            control_end: 1.0,
            control_specs: Vec::new(),
            artefacts: Vec::new(),
            artefact_library: None,
            artefact_blend: false,
            artefact_blend_strength: 0.3,
            smart_zones: false,
            wildcard_dir: None,
            tiled: false,
            tile_size: 1024,
            tile_stride: 768,
            grid: false,
            grid_cols: None,
            grid_padding: 0,
            kontext_bucket: false,
        }
    }

    fn write_test_png(path: &std::path::Path, w: u32, h: u32) {
        RgbImage::from_pixel(w, h, image::Rgb([0, 0, 0])).save(path).unwrap();
    }

    #[test]
    fn resolve_img2img_size_explicit_size_wins() {
        let tmp = std::env::temp_dir().join("plakat_aspect_test_explicit.png");
        write_test_png(&tmp, 1024, 1024);
        let mut args = mk_args(tmp);
        args.size = Some(Size { w: 768, h: 512 });
        args.aspect = Some("16:9".into()); // would normally pick if size unset
        assert_eq!(resolve_img2img_size(&args).unwrap(), (768, 512));
    }

    #[test]
    fn resolve_img2img_size_aspect_when_no_explicit() {
        let tmp = std::env::temp_dir().join("plakat_aspect_test_aspect.png");
        write_test_png(&tmp, 1024, 1024);
        let mut args = mk_args(tmp);
        args.aspect = Some("16:9".into());
        // 16:9 at base=1024 → longer side 1024 * 16/9 ≈ 1820 →
        // snapped to mult of 8 = 1816. Shorter side stays at base.
        let (w, h) = resolve_img2img_size(&args).unwrap();
        assert_eq!((w, h), (1816, 1024));
    }

    #[test]
    fn resolve_img2img_size_falls_back_to_input_dims() {
        let tmp = std::env::temp_dir().join("plakat_aspect_test_fallback.png");
        write_test_png(&tmp, 512, 768);
        let args = mk_args(tmp);
        // Neither --size nor --aspect set → input dims (snapped to 8).
        assert_eq!(resolve_img2img_size(&args).unwrap(), (512, 768));
    }

    #[test]
    fn resolve_img2img_size_portrait_aspect() {
        let tmp = std::env::temp_dir().join("plakat_aspect_test_portrait.png");
        write_test_png(&tmp, 1024, 1024);
        let mut args = mk_args(tmp);
        args.aspect = Some("9:16".into());
        // 9:16 at base=1024 → portrait, height longer.
        let (w, h) = resolve_img2img_size(&args).unwrap();
        assert_eq!((w, h), (1024, 1816));
    }
}
