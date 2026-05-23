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

    /// v0.16 phase 5: directory holding `<name>.txt` wildcard files
    /// for `__name__` prompt expansion. Inline `{a|b|c}` alternation
    /// works without this flag. When set, file wildcards in the
    /// prompt and negative prompt resolve to a random non-empty,
    /// non-comment line. Wildcard RNG is seeded from `--seed` when
    /// set (reproducible expansion) and from the OS RNG otherwise.
    #[arg(long = "wildcard-dir", value_name = "DIR")]
    pub wildcard_dir: Option<PathBuf>,

    /// v0.16 phase 5: CLIP-skip. `1` (default) uses the last hidden
    /// state — diffusers default, byte-identical to pre-v0.16 output.
    /// `2` uses the penultimate hidden state — the Auto1111 / NovelAI
    /// community default for SD 1.5 anime checkpoints (Anything-v3,
    /// AnyLoRA, ...). SD 1.5 / SD 2.1 only — SDXL ignores with a
    /// warning (already uses penultimate by training default).
    /// Flux / SD3 ignore entirely.
    #[arg(long = "clip-skip", default_value_t = 1, value_name = "N")]
    pub clip_skip: usize,

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

    /// ControlNet conditioner kind. v0.10 supports: `depth`. Requires
    /// either `--control-image PATH` (pre-rendered map) or
    /// `--control-from PATH` (auto-annotate any image). SD 1.5 only;
    /// Flux is unsupported.
    #[arg(long = "control", value_name = "KIND")]
    pub control: Option<crate::pipelines::controlnet::ControlKind>,

    /// Path to a pre-rendered conditioning image (a depth map, edge
    /// image, pose skeleton, etc.). Use this when you already have
    /// the annotator output. Mutually exclusive with `--control-from`.
    #[arg(long = "control-image", value_name = "PATH", conflicts_with = "control_from")]
    pub control_image: Option<PathBuf>,

    /// **v0.10**: path to an ordinary image to auto-annotate. Runs
    /// the matching annotator for `--control` (e.g. Depth-Anything-V2
    /// for `depth`) on this image and uses the result as the
    /// conditioning. Mutually exclusive with `--control-image`.
    #[arg(long = "control-from", value_name = "PATH")]
    pub control_from: Option<PathBuf>,

    /// Multiplier applied to ControlNet residuals. 0.0 = ignore the
    /// conditioner; 1.0 = full diffusers default; >1.0 over-emphasises
    /// the structure at the cost of prompt adherence. Sweet spot 0.6–1.0.
    #[arg(long = "control-strength", default_value_t = 1.0, value_name = "F")]
    pub control_strength: f32,

    /// Fractional timestep at which ControlNet becomes active.
    /// Default 0.0 (active from the start). Set e.g. 0.3 to skip
    /// control on the early high-noise steps.
    #[arg(long = "control-start", default_value_t = 0.0, value_name = "F")]
    pub control_start: f32,

    /// Fractional timestep at which ControlNet stops applying.
    /// Default 1.0 (active through to the end). Set e.g. 0.5 to
    /// lock composition early then let the prompt drive the late
    /// texture/atmosphere passes.
    #[arg(long = "control-end", default_value_t = 1.0, value_name = "F")]
    pub control_end: f32,

    /// **v0.11**: full ControlNet spec, repeatable for multi-ControlNet
    /// (depth + canny stacked etc.). Each occurrence stacks one
    /// conditioner; residuals from every active conditioner are summed.
    ///
    /// Grammar: `KIND[:option=value]*` where KIND ∈ {depth, canny} and
    /// options are `image=PATH`, `from=PATH`, `strength=F`, `start=F`,
    /// `end=F`. Examples:
    ///
    ///   --control-spec 'depth:from=in.jpg'
    ///   --control-spec 'canny:image=edges.png:strength=0.5:start=0.2:end=0.7'
    ///
    /// Mutually exclusive with the legacy single-conditioner flags
    /// (`--control`, `--control-image`, etc.). All conditioners in the
    /// stack share the model variant — mixing SD 1.5 / SDXL is not
    /// supported.
    #[arg(
        long = "control-spec",
        value_name = "SPEC",
        conflicts_with_all = [
            "control", "control_image", "control_from",
            "control_strength", "control_start", "control_end",
        ],
    )]
    pub control_specs: Vec<crate::pipelines::controlnet::ControlSpec>,

    /// **v0.12 / v0.13**: tiled hi-res generation. Enables
    /// MultiDiffusion-style overlapping passes — the transformer only
    /// ever sees tiles of `--tile-size` × `--tile-size`, blended
    /// per-step via a 2D Hann window. Lets SDXL or Flux produce 4K+
    /// outputs without exceeding the model's trained working
    /// resolution. Supported on SDXL (v0.12) and Flux (v0.13 phase 4).
    /// Doesn't yet compose with `--control*`, the SDXL refiner, or
    /// Flux.1-Fill-dev.
    #[arg(long = "tiled", default_value_t = false)]
    pub tiled: bool,

    /// Tile side length in pixels. Default 1024 — SDXL's native
    /// working resolution. Must be a multiple of 8 (VAE constraint).
    #[arg(long = "tile-size", default_value_t = 1024, value_name = "PX")]
    pub tile_size: u32,

    /// Stride between tile origins in pixels. Default 768 — gives a
    /// 256 px overlap between adjacent tiles (~25 %). Smaller stride
    /// = more overlap = smoother seams = more compute. Must be a
    /// multiple of 8 and ≤ `--tile-size`.
    #[arg(long = "tile-stride", default_value_t = 768, value_name = "PX")]
    pub tile_stride: u32,

    /// **v0.13 phase 1b**: also quantize the T5-XXL text encoder via
    /// city96's GGUF mirror (Q4_K_M, ~3 GB instead of ~10 GB BF16).
    /// Combined with `--model flux-*-gguf` the total Flux footprint
    /// drops to ~10 GB — fits 12 GB consumer GPUs. Requires a GGUF
    /// transformer (bails loud on BF16 Flux). Ignored for SD-family
    /// models.
    #[arg(long = "quantize-t5", default_value_t = false)]
    pub quantize_t5: bool,

    /// **v0.13 phase 5**: GGUF quant level for the Flux transformer.
    /// Defaults to `Q4_K_S` (~7 GB; v0.13 phase 1 footprint). city96
    /// publishes Q2_K..Q8_0 + F16; pick lower for tighter VRAM, higher
    /// for better quality. Ignored on BF16 Flux (`flux-dev` /
    /// `flux-schnell`) and SD-family models.
    ///
    /// Common picks:
    ///   * `Q3_K_S` (~5.5 GB) — tightest at the cost of noticeable quality drop
    ///   * `Q4_K_S` (~7 GB) — default; balanced
    ///   * `Q5_K_M` (~8.5 GB) — sweeter quality/memory tradeoff
    ///   * `Q8_0`   (~13 GB) — near-BF16 quality at half the memory
    ///   * `F16`    (~24 GB) — equivalent to BF16
    #[arg(long = "quant-level", value_name = "LEVEL")]
    pub quant_level: Option<String>,

    /// **v0.13 phase 5**: GGUF quant level for the T5-XXL encoder.
    /// Defaults to `Q4_K_M` (~3 GB). Only meaningful with
    /// `--quantize-t5`. city96 publishes Q3_K_S..Q8_0 + F16/F32.
    #[arg(long = "t5-quant-level", value_name = "LEVEL")]
    pub t5_quant_level: Option<String>,

    /// **v0.14 phase 6**: Apply a curated distillation-LoRA preset for
    /// fast Flux inference. Each preset bundles a published LoRA +
    /// recommended `--steps` + `--guidance`; the LoRA gets prepended
    /// to your `--loras` stack and the step/guidance defaults are
    /// overridden when you didn't pass them explicitly.
    ///
    /// Supported presets:
    ///   * `hyper-8`     — ByteDance Hyper-FLUX 8-step (CFG-free)
    ///   * `hyper-16`    — ByteDance Hyper-FLUX 16-step (CFG-free)
    ///   * `turbo-alpha` — alimama-creative FLUX.1-Turbo-Alpha 8-step
    ///
    /// ```bash
    /// plakat generate "..." --model flux-dev --fast hyper-8
    /// ```
    ///
    /// Requires a non-Fill Flux variant. NF4 + `--fast` bails (NF4 +
    /// LoRA composition isn't wired in v0.14).
    #[arg(long = "fast", value_name = "PRESET")]
    pub fast: Option<crate::pipelines::flux_fast::FastPresetArg>,

    /// **v0.14 phase 3 / 3c**: Flux Redux reference image. Adds image
    /// conditioning to the standard Flux variants (`flux-dev`,
    /// `flux-schnell`, GGUF, NF4) by encoding the image through
    /// SigLIP-so400m and BFL's Redux adapter, then seq-concatenating
    /// 729 tokens onto the T5 text embedding. Doesn't compose with
    /// `flux-fill-dev` (different `img_in` shape).
    ///
    /// **Repeatable** (v0.14 phase 3c): pass `--redux-image` up to 4
    /// times to stack references. Each entry accepts an optional
    /// `:weight=F.F` suffix that scales its tokens before concat
    /// (default 1.0; 0.0 turns the image off; ≤2.0 typical range).
    ///
    /// ```bash
    /// --redux-image style.png
    /// --redux-image subject.png:weight=0.7 --redux-image pose.png:weight=0.4
    /// ```
    ///
    /// Loading Redux adds ~1.5 GB of memory for SigLIP + the 140 MB
    /// adapter — paid only when this flag is set.
    #[arg(long = "redux-image", value_name = "SPEC")]
    pub redux_images: Vec<crate::pipelines::flux_redux::ReduxSpec>,

    /// Pre-rendered conditioning map for the BFL Flux "concept"
    /// checkpoints (`--model flux-canny-dev` or `flux-depth-dev`). The
    /// path is a canny edge map (for Canny-dev) or depth map (for
    /// Depth-dev) at the target output resolution. The image is
    /// VAE-encoded and concat'd onto the noise tokens at every
    /// denoise step — the model's `img_in` Linear is widened to
    /// 128 channels to consume it.
    ///
    /// Required for the concept variants when `--concept-from` isn't
    /// supplied; ignored on other models. Mutually exclusive with
    /// `--concept-from`.
    #[arg(
        long = "concept-image",
        value_name = "PATH",
        conflicts_with = "concept_from"
    )]
    pub concept_image: Option<PathBuf>,

    /// Auto-annotate this source photo into the conditioning map the
    /// loaded concept variant expects. With `--model flux-canny-dev`
    /// the source is run through the canny edge detector; with
    /// `--model flux-depth-dev` it's run through Depth-Anything-V2.
    /// The annotated PNG is written to a temporary file and fed to
    /// the model the same way `--concept-image` would.
    ///
    /// Mutually exclusive with `--concept-image`. Only valid with
    /// `--model flux-canny-dev` / `flux-depth-dev`.
    #[arg(long = "concept-from", value_name = "PATH")]
    pub concept_from: Option<PathBuf>,

    /// v0.16 phase 6: enable ADetailer-style face refinement. After
    /// the main t2i pass, plakat runs SCRFD on each output image,
    /// then for each detected face: crops an expanded bounding box,
    /// runs img2img on the crop with the same SD model + LoRAs, and
    /// feather-composites the refined crop back onto the original.
    /// Needs SCRFD weights configured via `PLAKAT_SCRFD_WEIGHTS` or
    /// `PLAKAT_SCRFD_HF` (same env vars the FaceID portrait flow
    /// uses). SD 1.5 / SDXL only — Flux / SD3 bail loud.
    #[arg(long = "adetailer", default_value_t = false)]
    pub adetailer: bool,

    /// v0.16 phase 6: img2img strength for the face refinement pass.
    /// `0.4` (default) preserves identity + colour, only crisps
    /// detail. `0.6+` can change the face significantly.
    #[arg(long = "adetailer-strength", default_value_t = 0.4, value_name = "F")]
    pub adetailer_strength: f32,

    /// v0.16 phase 6: bbox expansion factor for the face crop.
    /// `0.25` (default) adds 25% on each side — gives the inpaint
    /// pass enough surrounding context to match colour + skin tone.
    #[arg(long = "adetailer-padding", default_value_t = 0.25, value_name = "F")]
    pub adetailer_padding: f32,

    /// v0.16 phase 6: feather fraction for the composite. `0.25`
    /// fades the outer 25% of the bbox from full opacity → 0 at the
    /// edge. Larger feather = softer seam, smaller = sharper detail
    /// near the edge but more visible boundary.
    #[arg(long = "adetailer-feather", default_value_t = 0.25, value_name = "F")]
    pub adetailer_feather: f32,

    /// v0.16 phase 6: SCRFD confidence threshold. Faces below this
    /// score are skipped. `0.5` is the InsightFace deploy default.
    #[arg(long = "adetailer-confidence", default_value_t = 0.5, value_name = "F")]
    pub adetailer_confidence: f32,

    /// v0.16 phase 6: working resolution for the face img2img pass
    /// (square, snapped to multiples of 8). `512` (default) suits
    /// SD 1.5; `1024` matches SDXL. Larger = more VRAM + slower per
    /// face.
    #[arg(long = "adetailer-size", default_value_t = 512, value_name = "PX")]
    pub adetailer_size: u32,

    /// v0.16 phase 6: optional prompt override for the face pass.
    /// When unset, plakat uses a generic "detailed face, sharp
    /// focus, high quality". Override when you want a specific style
    /// (e.g. "ethereal portrait, soft lighting").
    #[arg(long = "adetailer-prompt", value_name = "STR")]
    pub adetailer_prompt: Option<String>,
}

pub async fn run(mut args: GenerateArgs, device: Device) -> Result<()> {
    // Style detection / resolution runs BEFORE the enhancer so the
    // trigger phrase carries the LoRA's training tokens unaltered.
    if args.style_ref.is_some() || args.style.is_some() {
        apply_style(&mut args, &device).await?;
    }

    // v0.16 phase 5: wildcard expansion. Runs BEFORE the enhancer
    // so the enhancer sees a concrete prompt — `{red|blue}` →
    // `red` first, then "improve this prompt" works. The wildcard
    // RNG is seeded from `--seed` for reproducibility when set.
    expand_prompt_wildcards(&mut args)?;

    if let Some(provider) = args.enhance.clone() {
        let enhanced = crate::prompt::enhance(&provider, &args.prompt).await?;
        tracing::info!(target: "plakat", "Enhanced prompt: {enhanced}");
        args.prompt = enhanced;
    }

    let (width, height) =
        crate::imaging::sizes::resolve(args.size, args.aspect.as_deref(), args.base)?;
    std::fs::create_dir_all(&args.out)?;

    // v0.14 phase 6: apply the `--fast` preset before LoRA / steps /
    // guidance get snapshotted into the t2i Request. Sequencing
    // matters: the preset LoRA must land on the LoRA stack BEFORE
    // the snapshot, and the step / guidance defaults must be
    // overridden only when the user didn't pass them explicitly
    // (clap doesn't give us provenance, so we match against the
    // documented defaults — `steps == 28` and `guidance == 7.5`).
    if let Some(fast) = args.fast.take() {
        let preset = fast.0;
        // Bail loud on incompatible model targets. Detection mirrors
        // t2i::Variant::detect so the failure mode is consistent.
        let m = args.model.to_lowercase();
        if !m.contains("flux") {
            anyhow::bail!(
                "--fast {} requires a Flux model (got --model {:?}). Hyper-FLUX / \
                 FLUX-Turbo LoRAs are Flux-family only.",
                preset.name,
                args.model
            );
        }
        if m.contains("fill") {
            anyhow::bail!(
                "--fast {} doesn't compose with flux-fill-dev. Use the standard \
                 flux-dev model with the distillation LoRA, then handle inpainting \
                 separately.",
                preset.name
            );
        }
        if m.contains("nf4") {
            anyhow::bail!(
                "--fast {} bails on NF4 — NF4 + LoRA composition isn't wired \
                 (deferred from v0.14 phase 2). Use --model flux-dev or \
                 flux-dev-gguf with the preset.",
                preset.name
            );
        }
        // Prepend so the preset LoRA loads BEFORE user LoRAs — user
        // LoRAs override at merge time when keys collide.
        args.loras.insert(0, preset.to_lora_spec());
        if args.steps == 28 {
            args.steps = preset.steps;
        }
        // clap's default for --guidance is 7.5; the preset override
        // only fires when that hasn't been touched.
        if (args.guidance - 7.5).abs() < f64::EPSILON {
            args.guidance = preset.guidance;
        }
        crate::ui::progress::println(&format!(
            "  fast preset '{}': +{} LoRA, steps={}, guidance={}",
            preset.name, preset.lora_repo, args.steps, args.guidance
        ));
    }

    // v0.16 phase 1: auto-annotation for the Flux "concept" variants.
    // When `--concept-from PATH` is set on `--model flux-canny-dev` /
    // `flux-depth-dev`, run the matching annotator (canny or depth)
    // on the source photo, write the result to a tempdir PNG, and
    // hand that path to the downstream pipeline the same way
    // `--concept-image` would.
    //
    // The tempdir must outlive the t2i::run call (the pipeline reads
    // the file inside `Pipeline::generate`), so we hold it in
    // `_concept_anno_tmp` for the rest of this function.
    let _concept_anno_tmp = if let Some(src) = args.concept_from.as_ref() {
        use crate::pipelines::controlnet::ControlKind;
        use crate::pipelines::t2i::Variant as TVariant;
        let variant = TVariant::detect(&args.model);
        if !variant.is_flux_concept() {
            anyhow::bail!(
                "--concept-from requires a Flux concept variant (--model \
                 flux-canny-dev or flux-depth-dev), got --model {:?}",
                args.model
            );
        }
        // Resolve target size: explicit --size wins; otherwise default
        // to 1024² (BFL's reference resolution for the concept models).
        let (anno_w, anno_h) = match &args.size {
            Some(sz) => (sz.w, sz.h),
            None => (1024, 1024),
        };
        // Pick the kind that matches the loaded variant. Canny-dev
        // wants edges; Depth-dev wants depth.
        let kind = if matches!(variant, TVariant::FluxCannyDev) {
            ControlKind::Canny
        } else {
            ControlKind::Depth
        };
        let anno_dtype = if matches!(device, Device::Cpu) {
            candle_core::DType::F32
        } else {
            candle_core::DType::BF16
        };
        let spin = crate::ui::progress::spinner(&format!(
            "Auto-annotating concept-from with {kind:?}"
        ));
        let anno = crate::pipelines::controlnet_annotator::annotate(
            kind, src, anno_w, anno_h, &device, anno_dtype,
        )
        .await?;
        let tmp = tempfile::Builder::new()
            .prefix("plakat-concept-anno-")
            .tempdir()?;
        let out_path = tmp.path().join(format!("concept-{}.png", kind.slug()));
        crate::pipelines::t2i::write_annotator_tensor_as_png(&anno, &out_path)?;
        spin.finish_with_message(format!(
            "✓ auto-annotated to {}", out_path.display()
        ));
        // Promote the auto-annotated PNG into `args.concept_image` so
        // the downstream code path is identical to the pre-rendered
        // case.
        args.concept_image = Some(out_path);
        Some(tmp)
    } else {
        None
    };

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

    // Phase 7d: capture the loaded SD backbone so the optional
    // --artefact-blend pass below can reuse it instead of paying for
    // a second multi-GB model load. `None` is returned when t2i routed
    // through the Flux pipeline — Flux has its own backbone and the
    // blend pass would need to load SD anyway (Flux portraits aren't
    // supported by the blend path).
    let shared_core = t2i::run(t2i::Request {
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
        controls: crate::pipelines::controlnet::resolve_control_specs(
            args.control_specs,
            args.control,
            args.control_image,
            args.control_from,
            args.control_strength,
            args.control_start,
            args.control_end,
        ),
        tiled: if args.tiled {
            Some(crate::pipelines::tiled::TiledConfig {
                tile_size: args.tile_size,
                stride: args.tile_stride,
            })
        } else {
            None
        },
        quantize_t5: args.quantize_t5,
        flux_quant_level: args.quant_level,
        t5_quant_level: args.t5_quant_level,
        redux_images: args.redux_images,
        // v0.15 phase 4: conditioning map for Flux Canny-dev / Depth-dev.
        flux_concept_image: args.concept_image,
        // v0.16 phase 5: CLIP-skip. SD 1.5 / SD 2.1 only.
        clip_skip: args.clip_skip,
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

    // v0.16 phase 6: ADetailer-style face refinement runs BEFORE the
    // artefact composite + blend. Order matters: face refinement is
    // a content fix, artefacts are intentional overlays — running
    // refinement first means the user's stamps land on faces that
    // already look right. The shared_core gets Arc-cloned so the
    // later artefact-blend pass can still consume it.
    if args.adetailer {
        let variant = crate::pipelines::t2i::Variant::detect(&model);
        if variant.is_flux() || variant.is_sd3() {
            anyhow::bail!(
                "--adetailer requires an SD-family model (SD 1.5 / SD 2.1 / SDXL / \
                 SDXL-Turbo). Got --model {} which routes through the {} pipeline. \
                 SD-family models can run the post-t2i face refinement pass; \
                 Flux / SD3 portrait support is a future phase.",
                model,
                if variant.is_flux() { "Flux" } else { "SD3" }
            );
        }
        let files: Vec<PathBuf> = (0..count)
            .map(|i| {
                let s = seed.unwrap_or(0).wrapping_add(i as u64);
                out_dir.join(format!("plakat-{s}.png"))
            })
            .filter(|p| p.exists())
            .collect();
        if !files.is_empty() {
            let adetailer_cfg = crate::pipelines::adetailer::Config {
                model: model.clone(),
                loras: loras.clone(),
                lora_scale,
                prompt: args.adetailer_prompt
                    .clone()
                    .unwrap_or_else(|| {
                        "detailed face, sharp focus, high quality".to_string()
                    }),
                negative: if negative.is_empty() {
                    "lowres, bad anatomy, blurry, deformed".to_string()
                } else {
                    negative.clone()
                },
                strength: args.adetailer_strength,
                working_size: args.adetailer_size,
                steps,
                guidance,
                scheduler,
                confidence: args.adetailer_confidence,
                padding: args.adetailer_padding,
                feather: args.adetailer_feather,
                device: device.clone(),
            };
            let spin = crate::ui::progress::spinner(&format!(
                "Running ADetailer over {} image(s)", files.len()
            ));
            let n = crate::pipelines::adetailer::refine_files(
                &adetailer_cfg,
                &files,
                shared_core.clone(),
            )
            .await?;
            spin.finish_with_message(format!(
                "✓ ADetailer refined {n} face(s) across {} image(s)",
                files.len()
            ));
        }
    }

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
            shared_core,
        )
        .await?;
    }
    Ok(())
}

/// v0.16 phase 5: expand `{a|b|c}` and `__name__` wildcards in both
/// the prompt and negative prompt. The wildcard RNG is seeded from
/// `--seed` when set (so the same seed reproduces the same picks);
/// otherwise OS entropy. `--wildcard-dir` is only required for
/// file wildcards (inline `{a|b|c}` works without it).
fn expand_prompt_wildcards(args: &mut GenerateArgs) -> Result<()> {
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
