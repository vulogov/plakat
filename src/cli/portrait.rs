use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait::{self, IdentityKind};
use crate::pipelines::scheduler::SchedulerKind;
use crate::style::{
    combine_negative, log_style_prep, parse_resolved_loras, prepare_style_with_session,
    prepend_trigger, StylePrepRequest,
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

    /// v0.19: bundled negative-prompt preset. See
    /// `plakat generate --negative-preset` for the full list.
    /// Combined with `--negative` if both are set; replaces the
    /// portrait DEFAULT_NEGATIVE if `--negative` isn't set.
    #[arg(long = "negative-preset", value_name = "NAME")]
    pub negative_preset: Option<String>,

    /// Random seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Optional prompt enhancer: deepseek | gemini | local |
    /// local:<alias> | auto.
    #[arg(long)]
    pub enhance: Option<String>,

    /// v0.19: see `plakat generate --enhance-system` — same semantics.
    #[arg(long = "enhance-system", value_name = "PATH")]
    pub enhance_system: Option<PathBuf>,

    /// v0.19: see `plakat generate --enhance-temp` — same semantics.
    #[arg(long = "enhance-temp", value_name = "F")]
    pub enhance_temp: Option<f64>,

    /// v0.19: see `plakat generate --enhance-max-tokens` — same semantics.
    #[arg(long = "enhance-max-tokens", value_name = "N")]
    pub enhance_max_tokens: Option<usize>,

    /// v0.19: opt-in disk cache for `--enhance local`. See
    /// `plakat generate --enhance-cache` for full details.
    #[arg(long = "enhance-cache", default_value_t = false)]
    pub enhance_cache: bool,

    /// v0.20: keep the original prompt alongside the enhancer's
    /// rewrite via the SD-family `BREAK` separator. See
    /// `plakat generate --enhance-keep-original` for full
    /// rationale. Portrait is always SD-family, so the flag
    /// applies unconditionally when `--enhance` is set.
    #[arg(long = "enhance-keep-original", default_value_t = false)]
    pub enhance_keep_original: bool,

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

    /// Timestep window start in `[0, 1]`. Default 0.0.
    #[arg(long = "control-start", default_value_t = 0.0, value_name = "F")]
    pub control_start: f32,

    /// Timestep window end in `[0, 1]`. Default 1.0. Use `0.5` to
    /// disable ControlNet for the back half of the schedule.
    #[arg(long = "control-end", default_value_t = 1.0, value_name = "F")]
    pub control_end: f32,

    /// **v0.11**: full ControlNet spec, repeatable for multi-ControlNet.
    /// See `plakat generate --control-spec` for grammar. Mutually
    /// exclusive with the legacy single-conditioner flags.
    #[arg(
        long = "control-spec",
        value_name = "SPEC",
        conflicts_with_all = [
            "control", "control_image", "control_from",
            "control_strength", "control_start", "control_end",
        ],
    )]
    pub control_specs: Vec<crate::pipelines::controlnet::ControlSpec>,

    /// v0.18 phase 2: with `--count N > 1`, also write a single
    /// `plakat-portrait-grid-<base-seed>.png` combining all N
    /// portraits in a near-square layout.
    #[arg(long = "grid", default_value_t = false)]
    pub grid: bool,

    /// v0.18 phase 2: column count for `--grid`. Default is
    /// `ceil(sqrt(count))`. Ignored when `--grid` is off.
    #[arg(long = "grid-cols", value_name = "N")]
    pub grid_cols: Option<usize>,

    /// v0.18 phase 2: padding (px) between grid cells. Default 0.
    /// Ignored when `--grid` is off.
    #[arg(long = "grid-padding", default_value_t = 0, value_name = "PX")]
    pub grid_padding: u32,
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
    // v0.19: --negative-preset takes precedence over DEFAULT_NEGATIVE
    // when `--negative` isn't set; combines with `--negative` when
    // both are set (preset first, user appended).
    let user_negative = args.negative.clone().unwrap_or_default();
    let mut negative = crate::prompt::negative_presets::combine(
        args.negative_preset.as_deref(),
        &user_negative,
    )?;
    if negative.trim().is_empty() {
        negative = DEFAULT_NEGATIVE.to_string();
    }

    // Phase 7f: capture the CLIP-H encoder the style runtime may have
    // lazy-loaded so we can hand it to portrait::Pipeline below — when
    // identity is PlusFace / PlusFaceSdxl, that saves a second ~2.5 GB
    // load. None when style isn't active.
    let mut shared_clip_h: Option<
        std::sync::Arc<crate::pipelines::ip_adapter::ImageEncoder>,
    > = None;

    // Style detection / resolution runs BEFORE the enhancer so the
    // trigger phrase carries the LoRA's training tokens unaltered.
    if args.style_ref.is_some() || args.style.is_some() {
        shared_clip_h = apply_style(&mut args, &mut negative, &device).await?;
    }

    if let Some(provider) = args.enhance.clone() {
        let enhance_args = crate::prompt::EnhanceArgs {
            system_path: args.enhance_system.clone(),
            temperature: args.enhance_temp,
            max_new_tokens: args.enhance_max_tokens,
            cache: args.enhance_cache,
        };
        let original = args.prompt.clone();
        let enhanced =
            crate::prompt::enhance_with_args(&provider, &args.prompt, &enhance_args)
                .await?;
        tracing::info!(target: "plakat", "Enhanced prompt: {enhanced}");
        args.prompt = crate::cli::generate::maybe_keep_original(
            &args.model,
            enhanced,
            &original,
            args.enhance_keep_original,
        );
    }

    // v0.18: A1111 inline <lora:name[:weight]> extraction.
    // Same ordering as generate / img2img: wildcards → enhance →
    // lora-tags → encode. Civitai LoRA prompt cards often include
    // inline tags; portrait prompts inherit the convention.
    if crate::prompt::lora_tags::has_lora_tags(&args.prompt) {
        let (cleaned, extracted) = crate::prompt::lora_tags::extract(&args.prompt)?;
        if !extracted.is_empty() {
            tracing::info!(
                target: "plakat",
                "Extracted {} inline <lora:> tag(s) from portrait prompt",
                extracted.len()
            );
            for ex in extracted.into_iter().rev() {
                args.loras.insert(0, ex.spec);
            }
            args.prompt = cleaned;
        }
    }
    if crate::prompt::lora_tags::has_lora_tags(&negative) {
        let (cleaned, _dropped) =
            crate::prompt::lora_tags::extract(&negative)?;
        negative = cleaned;
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
    // v0.18 phase 2: grid args captured early — `args` gets partially
    // moved into the portrait::Request below.
    let grid_enabled = args.grid;
    let grid_cols = args.grid_cols;
    let grid_padding = args.grid_padding;
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

    // Phase 7e: capture the loaded SD backbone so the optional
    // --artefact-blend pass below can reuse it instead of paying for
    // a second multi-GB model load.
    let shared_core = portrait::run(portrait::Request {
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
        // Phase 7f: hand the style runtime's lazy-loaded CLIP-H to the
        // portrait pipeline so a PlusFace identity reuses the same
        // weights instead of mmapping them a second time. FaceID
        // identities ignore this; the cost when style is inactive is a
        // single Option clone.
        shared_clip_h,
        controls: crate::pipelines::controlnet::resolve_control_specs(
            args.control_specs,
            args.control,
            args.control_image,
            args.control_from,
            args.control_strength,
            args.control_start,
            args.control_end,
        ),
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
            // Phase 7e: reuse the SD backbone loaded for the portrait
            // pass. Identity adapter weights are pipeline-local (not in
            // SdCore), so they correctly don't carry into the blend.
            Some(shared_core),
        )
        .await?;
    }

    // v0.18 phase 2: compose a grid of the per-image portrait outputs.
    // Runs LAST so artefact composite + blend are reflected in cells.
    if grid_enabled {
        if let Some((gw, gh, path)) = crate::imaging::grid::compose_grid_from_seed_range(
            &out_dir,
            "plakat-portrait",
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

/// Returns the CLIP-H encoder the style runtime loaded (if any) so the
/// caller can feed it into a downstream portrait pipeline build via
/// `LoadRequest::shared_clip_h`. `None` when the prep didn't need an
/// encoder (e.g. `--style ID` without a `--style-ref` photo).
async fn apply_style(
    args: &mut PortraitArgs,
    negative: &mut String,
    device: &Device,
) -> Result<Option<std::sync::Arc<crate::pipelines::ip_adapter::ImageEncoder>>> {
    let n_user_loras = args.loras.len();
    let (prep, shared_clip_h) = prepare_style_with_session(StylePrepRequest {
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

    Ok(shared_clip_h)
}
