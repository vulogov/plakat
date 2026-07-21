use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

use crate::imaging::upscale::Method;

#[derive(ClapArgs, Debug)]
pub struct UpscaleArgs {
    /// Input image.
    #[arg(long = "in", value_name = "IN")]
    pub input: PathBuf,

    /// Output image. Extension determines format (.png, .jpg, .webp).
    #[arg(help_heading = "Size & output", long, value_name = "OUT")]
    pub out: PathBuf,

    /// `--import <album>` / `--import-move`: land the upscaled image in a photo album.
    #[command(flatten)]
    pub import: crate::cli::import::ImportArgs,

    /// Scale factor for classical filters (e.g. 2 for 2×, 4 for 4×). Ignored
    /// for ML methods — their scale is fixed by the model.
    #[arg(long, default_value_t = 2.0)]
    pub scale: f32,

    /// Method:
    ///   nearest | bilinear | bicubic | lanczos                    (classical)
    ///   real-esrgan-x2 | real-esrgan-x4 | real-esrgan-anime-x4    (ML, RRDBNet)
    #[arg(long, default_value = "lanczos")]
    pub method: Method,

    /// Diffusion upscale (ControlNet-Tile / SUPIR-lite): pre-upscale then tiled img2img refine,
    /// each tile guided by ControlNet-Tile to hallucinate coherent detail. Uses `--scale`.
    #[arg(long, default_value_t = false)]
    pub diffusion: bool,

    /// [diffusion] SD model (Tile ControlNet is SD 1.5 / SDXL).
    #[arg(help_heading = "Model & sampler", long, default_value = "sd15")]
    pub model: String,

    /// [diffusion] Tile side in px (SD 1.5 → 512, SDXL → 1024).
    #[arg(long, default_value_t = 512)]
    pub tile: u32,

    /// [diffusion] Tile overlap in px (feathered blend).
    #[arg(long, default_value_t = 96)]
    pub overlap: u32,

    /// [diffusion] Per-tile img2img denoise strength. 0.3–0.5 adds detail while preserving
    /// structure; higher invents more (and risks tile drift).
    #[arg(long = "tile-strength", default_value_t = 0.4)]
    pub tile_strength: f32,

    /// [diffusion] ControlNet-Tile residual scale.
    #[arg(long = "cn-strength", default_value_t = 1.0)]
    pub cn_strength: f32,

    /// [diffusion] Denoise steps per tile.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 20)]
    pub steps: usize,

    /// [diffusion] CFG scale.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 6.0)]
    pub guidance: f64,

    /// [diffusion] Prompt steering the detail (kept generic by default).
    #[arg(help_heading = "Prompt & text", long, default_value = "highly detailed, sharp focus, intricate texture")]
    pub prompt: String,

    /// [diffusion] Negative prompt.
    #[arg(help_heading = "Prompt & text", long, default_value = "blurry, lowres, jpeg artifacts, oversmoothed")]
    pub negative: String,

    /// [diffusion] Seed.
    #[arg(help_heading = "Model & sampler", long, default_value_t = 0)]
    pub seed: u64,
}

pub async fn run(args: UpscaleArgs, device: Device) -> Result<()> {
    if args.diffusion {
        return run_diffusion(args, device).await;
    }
    let (w, h, nw, nh) = if args.method.is_ml() {
        // Real-ESRGAN ×4 can blow the Metal single-buffer cap on large inputs;
        // decorate the OOM with a `--device cpu` hint instead of a raw crash.
        crate::imaging::upscale::ml_upscale(&args.input, &args.out, args.method, &device)
            .await
            .map_err(|e| {
                crate::error_hints::decorate_oom(e, crate::error_hints::OomContext::Upscale)
            })?
    } else {
        crate::imaging::upscale::upscale(&args.input, &args.out, args.scale, args.method)?
    };
    let effective_scale = args.method.native_scale().unwrap_or(args.scale);
    println!(
        "{}  {}×{} → {}×{}  ({:.2}×, {:?})",
        style("✓").green(),
        w,
        h,
        nw,
        nh,
        effective_scale,
        args.method,
    );
    println!("→ {}", args.out.display());
    Ok(())
}

/// ControlNet-Tile diffusion upscale (SUPIR-lite). Loads an SD pipeline + the Tile ControlNet and
/// runs the tiled img2img refine.
async fn run_diffusion(args: UpscaleArgs, device: Device) -> Result<()> {
    use crate::pipelines::diffusion_upscale::Options;
    use crate::pipelines::portrait;
    use crate::pipelines::scheduler::SchedulerKind;

    let pipeline = portrait::Pipeline::load(portrait::LoadRequest {
        model: args.model.clone(),
        device: device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        identity: None,
        shared_clip_h: None,
    })
    .await
    .map_err(|e| crate::error_hints::decorate_oom(e, crate::error_hints::OomContext::Upscale))?;

    let opts = Options {
        input: args.input.clone(),
        out_path: args.out.clone(),
        scale: args.scale,
        tile: args.tile,
        overlap: args.overlap.min(args.tile.saturating_sub(8)),
        tile_strength: args.tile_strength,
        cn_strength: args.cn_strength,
        steps: args.steps,
        guidance: args.guidance,
        prompt: args.prompt.clone(),
        negative: args.negative.clone(),
        seed: args.seed,
        scheduler: SchedulerKind::default(),
    };
    pipeline
        .diffusion_upscale(&opts)
        .await
        .map_err(|e| crate::error_hints::decorate_oom(e, crate::error_hints::OomContext::Upscale))?;
    println!("{}  diffusion upscale complete", style("✓").green());
    Ok(())
}
