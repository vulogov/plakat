use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, IdentityKind};
use crate::pipelines::scheduler::SchedulerKind;

/// Portrait-tuned defaults — overrideable via flags.
const DEFAULT_NEGATIVE: &str = "deformed face, asymmetric eyes, extra fingers, \
                                cross-eyed, low quality, blurry, watermark, \
                                jpeg artifacts, bad anatomy, cropped head, \
                                disfigured, extra limbs, low resolution";

#[derive(ClapArgs, Debug)]
pub struct PortraitArgs {
    /// Text prompt describing the portrait (lighting, framing, style, etc.).
    pub prompt: String,

    /// Optional reference photo. Provide a head-and-shoulders crop for
    /// best results. Without a photo, runs as a portrait-tuned text-only
    /// generate (3:4 aspect, face/anatomy negatives baked in).
    #[arg(long, value_name = "PATH")]
    pub photo: Option<PathBuf>,

    /// Identity strategy. Phase 1 supports `plus-face` (IP-Adapter-Plus-Face
    /// on SD 1.5). Phase 2 will add FaceID and InstantID.
    #[arg(long, default_value = "plus-face")]
    pub identity: IdentityKind,

    /// Strength of the identity signal (image-token scale). 0.0 = pure
    /// text-driven, 1.0 = full reference influence, >1.0 over-amplifies
    /// the face at the cost of prompt adherence. Ignored without --photo.
    #[arg(long, default_value_t = 0.8)]
    pub face_strength: f32,

    /// Model: alias (sd15) or any HF SD-1.5 repo id. Phase 1 is SD 1.5 only.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// Output size, e.g. 768x1024. If omitted, use --aspect and --base.
    #[arg(long)]
    pub size: Option<Size>,

    /// Aspect ratio, e.g. 3:4 (default for portrait), 1:1, 2:3.
    #[arg(long, conflicts_with = "size", default_value = "3:4")]
    pub aspect: String,

    /// Base resolution used with --aspect (shorter side).
    #[arg(long, default_value_t = 768)]
    pub base: u32,

    /// Number of portraits to generate.
    #[arg(long, short = 'n', default_value_t = 1)]
    pub count: u32,

    /// Denoising steps.
    #[arg(long, default_value_t = 30)]
    pub steps: usize,

    /// Classifier-free guidance scale.
    #[arg(long, default_value_t = 7.0)]
    pub guidance: f64,

    /// Negative prompt. Defaults to a face-and-anatomy fixer baseline;
    /// pass --negative "" to disable.
    #[arg(long)]
    pub negative: Option<String>,

    /// Random seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Optional prompt enhancer: deepseek | gemini.
    #[arg(long)]
    pub enhance: Option<String>,

    /// Output directory.
    #[arg(long, default_value = "./out")]
    pub out: PathBuf,

    /// LoRA to apply. Repeatable. Same syntax as `generate`.
    #[arg(long = "lora")]
    pub loras: Vec<LoraSpec>,

    /// Global multiplier applied to every LoRA's per-file scale.
    #[arg(long, default_value_t = 1.0)]
    pub lora_scale: f32,

    /// Sampler. Defaults to `euler-a` (smoother skin tones than DDIM).
    #[arg(long, default_value = "euler-a")]
    pub scheduler: SchedulerKind,

    /// Add a low-strength img2img polish pass at the end (same model).
    /// Sharpens details and reduces artifacts.
    #[arg(long, value_name = "STEPS")]
    pub refine: Option<usize>,

    /// Strength of the --refine polish (0.0 = none, 1.0 = full re-noise).
    #[arg(long, default_value_t = 0.3)]
    pub refine_strength: f32,
}

pub async fn run(mut args: PortraitArgs, device: Device) -> Result<()> {
    if let Some(provider) = args.enhance.clone() {
        let enhanced = crate::prompt::enhance(&provider, &args.prompt).await?;
        tracing::info!(target: "plakat", "Enhanced prompt: {enhanced}");
        args.prompt = enhanced;
    }

    let (width, height) = crate::imaging::sizes::resolve(
        args.size,
        Some(args.aspect.as_str()),
        args.base,
    )?;
    std::fs::create_dir_all(&args.out)?;

    let negative = args.negative.unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());

    // Identity is only wired when a photo is actually provided. Without one,
    // skipping the identity load avoids a ~50 MB download for callers who
    // just want a portrait-tuned generate.
    let identity = args.photo.as_ref().map(|_| args.identity);

    portrait::run(portrait::Request {
        prompt: args.prompt,
        negative,
        photo: args.photo,
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
        face_strength: args.face_strength,
        identity,
    })
    .await
}
