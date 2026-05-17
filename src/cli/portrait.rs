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

    /// Identity strategy:
    ///   * `plus-face` (default) — IP-Adapter-Plus-Face on SD 1.5
    ///   * `plus-face-sdxl`     — IP-Adapter-Plus-Face on SDXL (vit-h)
    ///   * `faceid`             — IP-Adapter-FaceID on SD 1.5 (ArcFace)
    ///   * `faceid-sdxl`        — IP-Adapter-FaceID on SDXL (ArcFace)
    /// FaceID strategies require ArcFace weights (PLAKAT_ARCFACE_WEIGHTS
    /// or PLAKAT_ARCFACE_HF). Alignment: `--face-landmarks` > SCRFD-
    /// detected landmarks (when configured) > `--face-bbox` > centre-crop.
    /// InstantID is roadmap.
    #[arg(long, default_value = "plus-face")]
    pub identity: IdentityKind,

    /// Strength of the identity signal (image-token scale). 0.0 = pure
    /// text-driven, 1.0 = full reference influence, >1.0 over-amplifies
    /// the face at the cost of prompt adherence. Ignored without --photo.
    #[arg(long, default_value_t = 0.8)]
    pub face_strength: f32,

    /// Optional face bbox in the photo, format `X0,Y0,X1,Y1` (normalised
    /// to [0,1], origin top-left). When set, the photo is cropped to this
    /// region before identity encoding — meaningful for FaceID strategies
    /// where ArcFace was trained on tight face crops. CLIP-H strategies
    /// (`plus-face`, `plus-face-sdxl`) ignore it. Optional SCRFD auto-
    /// detection (PLAKAT_SCRFD_*) can fill this in from any photo.
    ///
    /// Example: `--face-bbox 0.2,0.1,0.8,0.7`.
    #[arg(long, value_name = "X0,Y0,X1,Y1", value_parser = parse_face_bbox)]
    pub face_bbox: Option<[f32; 4]>,

    /// Optional 5-point face landmarks in the photo, format
    /// `LX,LY,RX,RY,NX,NY,MLX,MLY,MRX,MRY` (10 floats normalised to
    /// [0,1], origin top-left). Order is fixed: left_eye, right_eye,
    /// nose, left_mouth_corner, right_mouth_corner (the same order
    /// InsightFace publishes detector outputs in).
    ///
    /// **Takes precedence over `--face-bbox`** when both are passed.
    /// FaceID strategies use this for a proper 5-point similarity-
    /// transform alignment to ArcFace's canonical 112×112 template —
    /// the closest we get to reference-quality identity preservation
    /// today without face auto-detection. CLIP-H strategies
    /// (`plus-face`, `plus-face-sdxl`) ignore it.
    ///
    /// Example (eyes at y=0.40, nose at y=0.55, mouth corners at y=0.68):
    /// `--face-landmarks 0.40,0.40,0.60,0.40,0.50,0.55,0.42,0.68,0.58,0.68`
    ///
    /// Optional SCRFD auto-detection (PLAKAT_SCRFD_*) can auto-fill these
    /// from any photo when no manual landmarks are passed.
    #[arg(long, value_name = "LX,LY,RX,RY,NX,NY,MLX,MLY,MRX,MRY", value_parser = parse_face_landmarks)]
    pub face_landmarks: Option<[[f32; 2]; 5]>,

    /// Model: alias (`sd15`, `sdxl`) or any HF SD-1.5/SDXL repo id. The
    /// `--identity` strategy must target the matching cross_attn_dim
    /// (`plus-face` for SD 1.5, `plus-face-sdxl` for SDXL).
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

/// Parse `X0,Y0,X1,Y1` into a normalised bbox. Validates `[0, 1]` bounds
/// and `x0 < x1`, `y0 < y1`.
fn parse_face_bbox(s: &str) -> std::result::Result<[f32; 4], String> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 4 {
        return Err(format!(
            "expected 4 comma-separated floats `X0,Y0,X1,Y1`, got {} parts",
            parts.len()
        ));
    }
    let mut vals = [0f32; 4];
    for (i, p) in parts.iter().enumerate() {
        vals[i] = p
            .parse::<f32>()
            .map_err(|e| format!("component {i} {p:?}: {e}"))?;
    }
    let [x0, y0, x1, y1] = vals;
    let in_unit = (0.0..=1.0).contains(&x0)
        && (0.0..=1.0).contains(&y0)
        && (0.0..=1.0).contains(&x1)
        && (0.0..=1.0).contains(&y1);
    if !in_unit || x0 >= x1 || y0 >= y1 {
        return Err(format!(
            "bbox [{x0},{y0},{x1},{y1}] must satisfy 0 ≤ x0 < x1 ≤ 1 \
             and 0 ≤ y0 < y1 ≤ 1"
        ));
    }
    Ok(vals)
}

/// Parse `LX,LY,RX,RY,NX,NY,MLX,MLY,MRX,MRY` (10 normalised floats) into
/// the 5-point landmark array. Order matches `LANDMARK_ORDER`:
/// left_eye, right_eye, nose, left_mouth, right_mouth.
fn parse_face_landmarks(s: &str) -> std::result::Result<[[f32; 2]; 5], String> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 10 {
        return Err(format!(
            "expected 10 comma-separated floats \
             `LX,LY,RX,RY,NX,NY,MLX,MLY,MRX,MRY`, got {} parts",
            parts.len()
        ));
    }
    let mut flat = [0f32; 10];
    for (i, p) in parts.iter().enumerate() {
        flat[i] = p
            .parse::<f32>()
            .map_err(|e| format!("component {i} {p:?}: {e}"))?;
        if !(0.0..=1.0).contains(&flat[i]) {
            return Err(format!(
                "landmark component {i} = {} out of range [0, 1]",
                flat[i]
            ));
        }
    }
    Ok([
        [flat[0], flat[1]],
        [flat[2], flat[3]],
        [flat[4], flat[5]],
        [flat[6], flat[7]],
        [flat[8], flat[9]],
    ])
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
        face_bbox: args.face_bbox,
        face_landmarks: args.face_landmarks,
        identity,
    })
    .await
}
