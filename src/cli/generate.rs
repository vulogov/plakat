use std::path::PathBuf;

use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;

use crate::imaging::sizes::Size;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::t2i;
use crate::style::{
    combine_negative, log_style_prep, parse_resolved_loras, prepare_style, prepend_trigger,
    StylePrepRequest,
};

#[derive(ClapArgs, Debug)]
pub struct GenerateArgs {
    /// Text prompt describing the image.
    pub prompt: String,

    /// Model: alias (sd15, sd21, sdxl, sdxl-turbo, flux-schnell) or any HF repo id.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// Output size, e.g. 768x768. If omitted, use --aspect and --base.
    #[arg(long)]
    pub size: Option<Size>,

    /// Aspect ratio, e.g. 16:9, 1:1, 2:3.
    #[arg(long, conflicts_with = "size")]
    pub aspect: Option<String>,

    /// Base resolution used with --aspect (shorter side).
    #[arg(long, default_value_t = 768)]
    pub base: u32,

    /// Number of images to generate.
    #[arg(long, short = 'n', default_value_t = 1)]
    pub count: u32,

    /// Denoising steps.
    #[arg(long, default_value_t = 28)]
    pub steps: usize,

    /// Classifier-free guidance scale. Use 0.0 for SDXL-Turbo.
    #[arg(long, default_value_t = 7.5)]
    pub guidance: f64,

    /// Negative prompt.
    #[arg(long, default_value = "")]
    pub negative: String,

    /// Random seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Optional prompt enhancer: deepseek | gemini.
    #[arg(long)]
    pub enhance: Option<String>,

    /// Output directory.
    #[arg(long, default_value = "./out")]
    pub out: PathBuf,

    /// LoRA to apply (kohya format). Repeatable. Each value can be:
    ///   - a local path:   `./mylora.safetensors`
    ///   - an HF repo:     `latent-consistency/lcm-lora-sdv1-5` (file auto-picked)
    ///   - an HF repo+file: `civitai/anime#models/style-v1.safetensors`
    /// Optionally append `:SCALE` (e.g. `:0.7`) to weight one LoRA. SD only.
    #[arg(long = "lora")]
    pub loras: Vec<LoraSpec>,

    /// Global multiplier applied to every LoRA's per-file scale.
    #[arg(long, default_value_t = 1.0)]
    pub lora_scale: f32,

    /// Sampler: default | ddim | euler-a | unipc (DPM-Solver++).
    /// Euler-A often improves SD 1.5/SDXL quality at the same step count.
    #[arg(long, default_value = "default")]
    pub scheduler: SchedulerKind,

    /// Add a low-strength img2img polish pass at the end (extra denoise steps
    /// on the generated latents using the SAME base model). Sharpens details
    /// and removes some artifacts. Not the official SDXL refiner.
    #[arg(long, value_name = "STEPS")]
    pub refine: Option<usize>,

    /// Strength of the --refine polish (0.0 = no effect, 1.0 = full re-noise).
    #[arg(long, default_value_t = 0.3)]
    pub refine_strength: f32,

    /// Use the real SDXL refiner UNet (stable-diffusion-xl-refiner-1.0) for
    /// the last fraction of the schedule. SDXL/SDXL-Turbo only. Adds a
    /// ~6 GB download on first run. Independent of --refine; both can be on.
    #[arg(long)]
    pub refiner: bool,

    /// Fraction of the schedule where the refiner takes over (last 1-FRAC).
    /// 0.8 = last 20% of steps run on the refiner.
    #[arg(long, default_value_t = 0.8)]
    pub refiner_frac: f32,

    /// Detect art style from this photo and load the matching LoRAs from
    /// the style catalog. Composes with --style to override the detected
    /// result by name. Conflicts with --lora (catalog LoRAs win, with a
    /// warning).
    #[arg(long, value_name = "PATH")]
    pub style_ref: Option<PathBuf>,

    /// Pick a style by id from the catalog. Bypasses detection when used
    /// alone; overrides the detection result when combined with
    /// --style-ref. See `plakat style list` (when shipped).
    #[arg(long, value_name = "ID")]
    pub style: Option<String>,

    /// Multiplier applied to every catalog LoRA's :scale. 1.0 uses the
    /// catalog's authored scales verbatim. Above ~1.8 most LoRAs start
    /// to degrade the prompt.
    #[arg(long, default_value_t = 1.0)]
    pub style_strength: f32,

    /// Override the bundled style catalog directory.
    #[arg(long, value_name = "DIR")]
    pub style_catalog: Option<PathBuf>,

    /// Composite a named artefact (PNG cutout) into the generated
    /// image. Repeatable. Grammar: `NAME[@ZONE[:SCALE]]`. The artefact
    /// is alpha-blended onto the generated image *after* generation
    /// but *before* any optional stylize/upscale pass.
    ///
    /// Examples:
    ///   --artefact oak                         (natural zone, default scale)
    ///   --artefact oak@middle_plan/left        (override zone)
    ///   --artefact sun@sky/right:0.8           (zone + scale)
    ///
    /// Multiple `--artefact` flags compose left-to-right (z-order
    /// equals flag order). For per-artefact offset / anchor /
    /// flip / alpha overrides, use the scenario `artefacts: [...]`
    /// HJSON form.
    #[arg(long = "artefact", value_name = "NAME[@ZONE[:SCALE]]")]
    pub artefacts: Vec<crate::artefacts::ArtefactSpec>,

    /// Override the bundled artefact library directory.
    #[arg(long, value_name = "DIR")]
    pub artefact_library: Option<PathBuf>,

    /// After alpha-compositing artefacts, run a low-strength masked
    /// img2img blending pass over the artefact zones. Smooths hard
    /// edges and modest lighting mismatches at the cost of one extra
    /// short denoise pass (~2–5 s per image on GPU). Default: off
    /// (v1 alpha-only).
    #[arg(long = "artefact-blend", default_value_t = false)]
    pub artefact_blend: bool,

    /// img2img strength for `--artefact-blend`. 0.0 = no-op,
    /// 1.0 = full re-noise inside the mask. Sweet spot: 0.25–0.4.
    /// Higher values let the model redraw the artefact silhouette.
    #[arg(long = "artefact-blend-strength", default_value_t = 0.3, value_name = "F")]
    pub artefact_blend_strength: f32,

    /// v3: derive artefact zones from the generated image's own
    /// depth + luminance instead of the rigid 4×3 grid. Requires the
    /// Depth-Anything-V2 small checkpoint (~99 MB, downloaded once
    /// and cached). Falls back to the grid with a warning if the
    /// model can't be loaded. Default: off.
    #[arg(long = "smart-zones", default_value_t = false)]
    pub smart_zones: bool,

    /// v0.9: ControlNet conditioner kind. Currently supports
    /// `depth`. Requires `--control-image PATH`. SD 1.5 only;
    /// Flux is unsupported.
    #[arg(long = "control", value_name = "KIND")]
    pub control: Option<crate::pipelines::controlnet::ControlKind>,

    /// Path to the conditioning image (a depth map, edge image,
    /// pose skeleton, etc.). Required when `--control` is set.
    #[arg(long = "control-image", value_name = "PATH")]
    pub control_image: Option<PathBuf>,

    /// Multiplier applied to ControlNet residuals. 0.0 = ignore the
    /// conditioner; 1.0 = full diffusers default; >1.0 over-emphasises
    /// the structure at the cost of prompt adherence. Sweet spot 0.6–1.0.
    #[arg(long = "control-strength", default_value_t = 1.0, value_name = "F")]
    pub control_strength: f32,
}

pub async fn run(mut args: GenerateArgs, device: Device) -> Result<()> {
    // Style detection / resolution runs BEFORE the enhancer so the
    // trigger phrase carries the LoRA's training tokens unaltered.
    if args.style_ref.is_some() || args.style.is_some() {
        apply_style(&mut args, &device).await?;
    }

    if let Some(provider) = args.enhance.clone() {
        let enhanced = crate::prompt::enhance(&provider, &args.prompt).await?;
        tracing::info!(target: "plakat", "Enhanced prompt: {enhanced}");
        args.prompt = enhanced;
    }

    let (width, height) =
        crate::imaging::sizes::resolve(args.size, args.aspect.as_deref(), args.base)?;
    std::fs::create_dir_all(&args.out)?;

    let out_dir = args.out.clone();
    let count = args.count;
    // Resolve the seed at the CLI boundary (rather than letting t2i pick a
    // random one internally) so the artefact compositor knows which output
    // files to read back. Behaviour is bit-equivalent — t2i picks the same
    // seed if given vs. random it would otherwise pick.
    let seed = Some(args.seed.unwrap_or_else(rand::random));

    let prompt = args.prompt.clone();
    let negative = args.negative.clone();
    let model = args.model.clone();
    let loras = args.loras.clone();
    let lora_scale = args.lora_scale;
    let scheduler = args.scheduler;
    let steps = args.steps;
    let guidance = args.guidance;

    t2i::run(t2i::Request {
        prompt: args.prompt,
        negative: args.negative,
        model: args.model,
        width,
        height,
        count: args.count,
        steps: args.steps,
        guidance: args.guidance,
        seed,
        out_dir: args.out,
        device: device.clone(),
        loras: args.loras,
        lora_scale: args.lora_scale,
        scheduler: args.scheduler,
        refine: args.refine,
        refine_strength: args.refine_strength,
        use_refiner: args.refiner,
        refiner_frac: args.refiner_frac,
        control_kind: args.control,
        control_image: args.control_image,
        control_strength: args.control_strength,
    })
    .await?;

    // Composite any --artefact flags onto the generated images. t2i
    // writes `plakat-<seed>.png` files (one per image in `count`).
    let library_dir = args
        .artefact_library
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/artefact_library"));

    // v3: lazily load the depth pipeline if --smart-zones is on.
    // On load failure, warn and continue with the rigid grid.
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
        "plakat",
        width,
        height,
        &Default::default(),
        smart_depth.as_ref(),
    )?;

    // v2: optional masked img2img blend over the artefact zones,
    // smoothing the alpha-composited edges. Skipped when no
    // artefacts were placed.
    if args.artefact_blend && !args.artefacts.is_empty() {
        let files: Vec<PathBuf> = (0..count)
            .map(|i| {
                let s = seed.unwrap_or(0).wrapping_add(i as u64);
                out_dir.join(format!("plakat-{s}.png"))
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
        )
        .await?;
    }
    Ok(())
}

async fn apply_style(args: &mut GenerateArgs, device: &Device) -> Result<()> {
    let n_user_loras = args.loras.len();
    let prep = prepare_style(StylePrepRequest {
        style_ref: args.style_ref.as_deref(),
        style_override: args.style.as_deref(),
        style_strength: args.style_strength,
        style_catalog: args.style_catalog.as_deref(),
        model: &args.model,
        user_loras_nonempty: !args.loras.is_empty(),
        device,
    })
    .await?;

    log_style_prep(&prep, n_user_loras);

    args.loras = parse_resolved_loras(&prep)?;
    args.prompt = prepend_trigger(&prep.trigger, &args.prompt);
    args.negative = combine_negative(&args.negative, &prep.negative_extras);

    Ok(())
}
