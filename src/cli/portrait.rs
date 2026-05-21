use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, IdentityKind};
use crate::pipelines::scheduler::SchedulerKind;
use crate::style::{
    combine_negative, log_style_prep, parse_resolved_loras, prepare_style, prepend_trigger,
    StylePrepRequest,
};

/// Portrait-tuned defaults — overrideable via flags.
const DEFAULT_NEGATIVE: &str = "deformed face, asymmetric eyes, extra fingers, \
                                cross-eyed, low quality, blurry, watermark, \
                                jpeg artifacts, bad anatomy, cropped head, \
                                disfigured, extra limbs, low resolution";

#[derive(ClapArgs, Debug)]
pub struct PortraitArgs {
    /// Text prompt describing the portrait (lighting, framing, style, etc.).
    pub prompt: String,

    /// Reference photo(s). Repeatable: pass `--photo` multiple times to
    /// merge facial features from several reference photos at the
    /// embedding-space level (not pixel-blending — useful for averaging
    /// multiple photos of the same person, or weighted blending across
    /// look-alikes). Each photo accepts an optional `:WEIGHT` suffix:
    ///
    ///   --photo alice.jpg                    (single, weight ignored)
    ///   --photo alice.jpg --photo bob.jpg    (equal 50/50 merge)
    ///   --photo alice.jpg:0.7 --photo bob.jpg:0.3   (weighted merge)
    ///   --photo alice.jpg:0.8 --photo bob.jpg       (bob auto-fills 0.2)
    ///
    /// Weights are normalized to sum to 1.0. Total identity strength is
    /// independently controlled by `--face-strength`. Without any
    /// `--photo`, runs as a portrait-tuned text-only generate (3:4
    /// aspect, face/anatomy negatives baked in).
    #[arg(long, value_name = "PATH[:WEIGHT]")]
    pub photo: Vec<crate::pipelines::ip_adapter::WeightedPhoto>,

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

    /// Detect art style from this photo and load the matching LoRAs from
    /// the style catalog. The style reference is *separate* from `--photo`
    /// — `--photo` controls identity (who), `--style-ref` controls visual
    /// style (how). Conflicts with --lora (catalog LoRAs win, with a
    /// warning).
    #[arg(long, value_name = "PATH")]
    pub style_ref: Option<PathBuf>,

    /// Pick a style by id from the catalog. Bypasses detection when used
    /// alone; overrides the detection result when combined with
    /// --style-ref.
    #[arg(long, value_name = "ID")]
    pub style: Option<String>,

    /// Multiplier applied to every catalog LoRA's :scale. 1.0 uses the
    /// catalog's authored scales verbatim.
    #[arg(long, default_value_t = 1.0)]
    pub style_strength: f32,

    /// Override the bundled style catalog directory.
    #[arg(long, value_name = "DIR")]
    pub style_catalog: Option<PathBuf>,

    /// Composite a named artefact (PNG cutout from the artefact library)
    /// into the portrait. Repeatable. Grammar: `NAME[@ZONE[:SCALE]]` —
    /// same as `plakat generate`. See `plakat artefact list`.
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
    #[arg(long = "artefact-blend-strength", default_value_t = 0.3, value_name = "F")]
    pub artefact_blend_strength: f32,

    /// v3: derive artefact zones from the generated image's own
    /// depth + luminance instead of the rigid 4×3 grid. See
    /// `Documentation/ARTEFACTS.md` § Smart zones (v3) for cost +
    /// fallback behaviour.
    #[arg(long = "smart-zones", default_value_t = false)]
    pub smart_zones: bool,

    /// ControlNet conditioner kind (currently `depth`). Requires
    /// `--control-image PATH` (pre-rendered) or `--control-from PATH`
    /// (auto-annotate). SD 1.5 only. See `Documentation/CONTROLNET.md`.
    #[arg(long = "control", value_name = "KIND")]
    pub control: Option<crate::pipelines::controlnet::ControlKind>,

    /// Pre-rendered conditioning image (depth map, pose skeleton, ...).
    /// Mutually exclusive with `--control-from`.
    #[arg(long = "control-image", value_name = "PATH", conflicts_with = "control_from")]
    pub control_image: Option<PathBuf>,

    /// **v0.10**: source image to auto-annotate. Runs the matching
    /// annotator for `--control` and uses the result as the
    /// conditioning. Mutually exclusive with `--control-image`.
    #[arg(long = "control-from", value_name = "PATH")]
    pub control_from: Option<PathBuf>,

    /// Multiplier applied to ControlNet residuals. Sweet spot 0.6–1.0.
    #[arg(long = "control-strength", default_value_t = 1.0, value_name = "F")]
    pub control_strength: f32,

    /// v0.10 phase 4: timestep window start [0, 1]. Default 0.0.
    #[arg(long = "control-start", default_value_t = 0.0, value_name = "F")]
    pub control_start: f32,

    /// v0.10 phase 4: timestep window end [0, 1]. Default 1.0. Use
    /// `0.5` to disable ControlNet for the back half of the schedule.
    #[arg(long = "control-end", default_value_t = 1.0, value_name = "F")]
    pub control_end: f32,
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
    // Resolve the effective negative prompt up front. Style detection
    // may augment it via the catalog's negative_extras, which means
    // we need a concrete String to merge into — not Option<String>.
    let mut negative = args.negative.clone().unwrap_or_else(|| DEFAULT_NEGATIVE.to_string());

    // Style detection / resolution runs BEFORE the enhancer so the
    // trigger phrase carries the LoRA's training tokens unaltered.
    if args.style_ref.is_some() || args.style.is_some() {
        apply_style(&mut args, &mut negative, &device).await?;
    }

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

    // Identity is only wired when at least one photo is actually provided.
    // Without any, skipping the identity load avoids a ~50 MB download for
    // callers who just want a portrait-tuned generate.
    let photos = args.photo.clone();
    let identity = if photos.is_empty() {
        None
    } else {
        Some(args.identity)
    };

    let out_dir = args.out.clone();
    let count = args.count;
    // Pre-resolve the seed at the CLI boundary so the artefact compositor /
    // blender know which output files to read back.
    let seed = Some(args.seed.unwrap_or_else(rand::random));
    let artefact_specs = args.artefacts.clone();
    let artefact_library = args
        .artefact_library
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/artefact_library"));

    let prompt = args.prompt.clone();
    let negative_for_blend = negative.clone();
    let model = args.model.clone();
    let loras = args.loras.clone();
    let lora_scale = args.lora_scale;
    let scheduler = args.scheduler;
    let steps = args.steps;
    let guidance = args.guidance;

    portrait::run(portrait::Request {
        prompt: args.prompt,
        negative,
        photos,
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
        face_strength: args.face_strength,
        face_bbox: args.face_bbox,
        face_landmarks: args.face_landmarks,
        identity,
        control_kind: args.control,
        control_image: args.control_image,
        control_from: args.control_from,
        control_strength: args.control_strength,
        control_start: args.control_start,
        control_end: args.control_end,
    })
    .await?;

    // v3: lazily load the depth pipeline if --smart-zones is on.
    let smart_depth = if args.smart_zones && !artefact_specs.is_empty() {
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

    // Composite any --artefact onto the saved portrait file(s).
    crate::artefacts::composite_onto_seed_range(
        &artefact_specs,
        &artefact_library,
        &out_dir,
        seed,
        count,
        "plakat-portrait",
        width,
        height,
        &Default::default(),
        smart_depth.as_ref(),
    )?;

    // v2: optional masked img2img blend over the artefact zones.
    if args.artefact_blend && !artefact_specs.is_empty() {
        let files: Vec<PathBuf> = (0..count)
            .map(|i| {
                let s = seed.unwrap_or(0).wrapping_add(i as u64);
                out_dir.join(format!("plakat-portrait-{s}.png"))
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
                negative: negative_for_blend,
                image_w: width,
                image_h: height,
                steps,
                guidance,
                scheduler,
                strength: args.artefact_blend_strength,
                feather_px: None,
            },
            &artefact_specs,
            &artefact_library,
            &files,
            &Default::default(),
            seed,
            smart_depth.as_ref(),
        )
        .await?;
    }
    Ok(())
}

async fn apply_style(
    args: &mut PortraitArgs,
    negative: &mut String,
    device: &Device,
) -> Result<()> {
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
    *negative = combine_negative(negative, &prep.negative_extras);

    Ok(())
}
