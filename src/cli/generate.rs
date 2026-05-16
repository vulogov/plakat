use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::t2i;

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
}

pub async fn run(mut args: GenerateArgs, device: Device) -> Result<()> {
    if let Some(provider) = args.enhance.clone() {
        let enhanced = crate::prompt::enhance(&provider, &args.prompt).await?;
        tracing::info!(target: "plakat", "Enhanced prompt: {enhanced}");
        args.prompt = enhanced;
    }

    let (width, height) =
        crate::imaging::sizes::resolve(args.size, args.aspect.as_deref(), args.base)?;
    std::fs::create_dir_all(&args.out)?;

    t2i::run(t2i::Request {
        prompt: args.prompt,
        negative: args.negative,
        model: args.model,
        width,
        height,
        count: args.count,
        steps: args.steps,
        guidance: args.guidance,
        seed: args.seed,
        out_dir: args.out,
        device,
        loras: args.loras,
        lora_scale: args.lora_scale,
        scheduler: args.scheduler,
        refine: args.refine,
        refine_strength: args.refine_strength,
        use_refiner: args.refiner,
        refiner_frac: args.refiner_frac,
    })
    .await
}
