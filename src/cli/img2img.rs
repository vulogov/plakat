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

    /// Model: alias (sd15, sd21, sdxl, sdxl-turbo) or any HF repo id.
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
