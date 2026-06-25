use std::path::PathBuf;

use anyhow::{Context, Result};
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

    /// v0.20: load generation defaults from a JSON sidecar (the
    /// `.json` file plakat writes alongside every output, or any
    /// file in the same `GenerationMetadata` format). CLI flags
    /// that differ from their built-in defaults override the
    /// recipe; flags left at their defaults inherit from the
    /// recipe.
    ///
    /// The positional `prompt` arg ALWAYS comes from the CLI —
    /// the recipe never replaces it. Use `plakat clone PNG` if
    /// you want a full byte-equivalent rerun of an existing
    /// image.
    ///
    /// Common workflow: rerun a previous setup with a new prompt:
    ///
    /// ```bash
    /// plakat generate "a different scene description" \
    ///     --recipe ./out/plakat-42.json
    /// ```
    ///
    /// All scalar fields (model / seed / steps / guidance /
    /// scheduler / size / negative / clip-skip) flow through, plus
    /// the LoRA stack. Pass any CLI flag explicitly to override
    /// the corresponding recipe field.
    #[arg(long, value_name = "PATH")]
    pub recipe: Option<PathBuf>,

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

    /// Stable Cascade Stage C denoising steps. Unset → split
    /// `--steps` 2/3 to Stage C (heavy semantic stage) + 1/3 to
    /// Stage B (refine). When set, takes precedence over `--steps`
    /// for the Stage C count. Cascade-only; ignored on every
    /// non-Cascade model.
    #[arg(long = "stage-c-steps", value_name = "N")]
    pub stage_c_steps: Option<usize>,

    /// Stable Cascade Stage B denoising steps. Unset → derived
    /// from `--steps` (1/3 of total, or `steps - stage_c_steps`
    /// when only `--stage-c-steps` was given). Cascade-only.
    #[arg(long = "stage-b-steps", value_name = "N")]
    pub stage_b_steps: Option<usize>,

    /// Stable Cascade ControlNet weights path (safetensors).
    /// OPTIONAL since v0.41 — when omitted and a `canny` control is
    /// given (`--control-image` / `--control-from` / `--control-spec
    /// canny:…`), the CN is auto-resolved from the model repo's
    /// `controlnet/canny.safetensors`. Set this only to point at a
    /// non-standard checkpoint. Cascade-only; ignored on other models.
    #[arg(long = "cascade-control-weights", value_name = "PATH")]
    pub cascade_control_weights: Option<PathBuf>,

    /// Stable Cascade image variation: condition generation on a
    /// reference image's CLIP ViT-L/14 embedding (unCLIP-style). The
    /// output shares the reference's semantics (subject, palette, mood)
    /// while re-composing it. Combine with a `--prompt` to steer, or
    /// leave the prompt empty to vary on the image alone. Loads the
    /// `image_encoder/` from the Cascade repo on first use.
    /// Cascade-only; ignored on other models.
    #[arg(long = "image-variation", value_name = "PATH")]
    pub image_variation: Option<PathBuf>,

    /// Classifier-free guidance scale. Use 0.0 for SDXL-Turbo.
    #[arg(long, default_value_t = 7.5)]
    pub guidance: f64,

    /// Stable Cascade Stage B (decoder) CFG scale, decoupled from
    /// `--guidance` (which drives the Stage C prior). Upstream's
    /// decoder defaults to ~0 (no CFG); ~1.0 is the pure conditional.
    /// Default 1.1 (mild). Raise toward 2-4 for sharper decoder
    /// detail at the risk of over-saturation. Cascade-only.
    #[arg(long = "decoder-guidance", default_value_t = 1.1)]
    pub decoder_guidance: f64,

    /// Negative prompt.
    #[arg(long, default_value = "")]
    pub negative: String,

    /// v0.19: bundled negative-prompt preset. One of `photo` /
    /// `painting` / `anime` / `cinematic`. When set, the preset's
    /// curated negative is used; if `--negative` is ALSO set, the
    /// two are comma-joined (preset first, user negative appended).
    /// Saves users from copy-pasting the same `blurry, low quality,
    /// watermark, ...` line into every invocation.
    #[arg(long = "negative-preset", value_name = "NAME")]
    pub negative_preset: Option<String>,

    /// Random seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Optional prompt enhancer: deepseek | gemini | local |
    /// local:<alias> | auto.
    #[arg(long)]
    pub enhance: Option<String>,

    /// v0.19: custom system prompt path for the enhancer. Overrides
    /// the built-in "rewrite text-to-image prompts..." system prompt
    /// for all three providers (DeepSeek / Gemini / local). Loaded
    /// from disk on each enhance dispatch.
    #[arg(long = "enhance-system", value_name = "PATH")]
    pub enhance_system: Option<PathBuf>,

    /// v0.19: sampling temperature for `--enhance local`. Default
    /// `0.0` (greedy — reproducible: same prompt = same enhancement).
    /// Bump to `0.5`-`1.0` for variety at the cost of repeatability.
    /// Ignored on the API providers (DeepSeek / Gemini have their own
    /// server-side defaults).
    #[arg(long = "enhance-temp", value_name = "F")]
    pub enhance_temp: Option<f64>,

    /// v0.19: maximum new tokens for `--enhance local`. Default 96.
    /// Higher values let the enhancer write longer prompts at the
    /// cost of decode-loop latency. Ignored on the API providers.
    #[arg(long = "enhance-max-tokens", value_name = "N")]
    pub enhance_max_tokens: Option<usize>,

    /// v0.19: SHA-256 disk cache for the local enhancer. When set,
    /// (alias, system, user, temp, max_tokens) keys an on-disk
    /// lookup at `~/.cache/plakat/enhance/`; cache hits skip the
    /// LLM forward entirely. Cache misses run the model and write
    /// the result on success (refusals + empty output never cache).
    /// Opt-in to avoid stale-hit surprises during system-prompt
    /// iteration. Ignored on the API providers.
    #[arg(long = "enhance-cache", default_value_t = false)]
    pub enhance_cache: bool,

    /// v0.20: keep the original prompt alongside the enhancer's
    /// rewrite by joining them with the SD-family `BREAK`
    /// separator (each chunk gets its own 77-token CLIP slot,
    /// so the original terms keep full attention weight).
    /// SD 1.5 / 2.1 / SDXL only — Flux and SD3 use T5, which
    /// has the token budget to carry both phrasings without
    /// BREAK and where the keyword is a no-op anyway. Ignored
    /// without `--enhance`. Ignored if the prompt was empty.
    #[arg(long = "enhance-keep-original", default_value_t = false)]
    pub enhance_keep_original: bool,

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

    /// v0.30 phase 1: explicit LCM-LoRA mode. Forces the LCM
    /// scheduler and the LCM-LoRA defaults (`--steps 4 --guidance
    /// 1.5`) so a Civitai LCM-LoRA you've named with something
    /// other than `lcm` still gets the right schedule. Without the
    /// flag, plakat auto-detects LCM-LoRAs by matching `lcm` in any
    /// `--lora` source — explicit `--lcm` is the override for the
    /// auto-detect's blind spots. User-supplied `--steps` /
    /// `--guidance` still take precedence — `--lcm --steps 8` runs
    /// at higher quality. SD 1.5 / SDXL.
    #[arg(long = "lcm", default_value_t = false)]
    pub lcm: bool,

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

    /// Regional prompting: a prompted region `"X0,Y0,X1,Y1:prompt"` (coords are
    /// `[0,1]` canvas fractions). Repeatable — each region's prompt applies in
    /// its box, blended over the main prompt for one coherent image. SD 1.5 /
    /// SDXL, native resolution. Not composed with `--tiled` / `--control*`.
    #[arg(long = "region", value_name = "X0,Y0,X1,Y1:PROMPT")]
    pub region: Vec<String>,

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
    #[arg(long = "flux-quant-level", value_name = "LEVEL")]
    pub flux_quant_level: Option<String>,

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

    /// **v0.25**: art-medium preset. Bundles a prompt prefix/suffix,
    /// recommended `--scheduler` / `--steps` / `--guidance`, and
    /// (phase 4+) automatic LoRA discovery matched to the medium.
    /// Built-in: `ink-wash`, `watercolor`, `oil-painting`,
    /// `charcoal`, `pencil`, `chalk-pastel`, `linocut`, `gouache`.
    /// User-extensible via `$CONFIG_DIR/looks/*.json` (phase 9).
    /// Override-only: flags you pass explicitly always win.
    /// Composes with `--genre`, `--style`, `--fast`,
    /// `--negative-preset`.
    #[arg(long = "look", value_name = "NAME")]
    pub look: Option<String>,

    /// **v0.25**: subject-domain preset — independent axis from
    /// `--look`. Built-in: `anime`. User-extensible via
    /// `$CONFIG_DIR/genres/*.json` (phase 9). Combines additively
    /// with `--look` (a watercolor anime composes both).
    #[arg(long = "genre", value_name = "NAME")]
    pub genre: Option<String>,

    /// **v0.25**: skip remote LoRA discovery for `--look` /
    /// `--genre`. Discovery normally chains Civitai → HuggingFace
    /// Hub → local-cache scan; with `--offline`, only the on-disk
    /// discovery cache and the local-cache scan run. Useful for CI
    /// / reproducibility / air-gapped runs. Has no effect when no
    /// `--look` / `--genre` is set.
    #[arg(long, default_value_t = false)]
    pub offline: bool,

    /// **v0.46**: re-rank `--look` / `--genre` Civitai discovery candidates
    /// with a small local LLM that picks the best *style* LoRA and rejects
    /// character/person LoRAs. Generic medium terms (watercolour, pencil)
    /// otherwise match anime-character LoRAs that hijack the subject. Runs the
    /// judge on CPU; falls back to the prompt preset if nothing suitable.
    #[arg(long = "smart-discovery", default_value_t = false)]
    pub smart_discovery: bool,

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

    /// Pre-rendered conditioning map / reference image for the BFL
    /// Flux "concept" + Kontext checkpoints. Path interpretation
    /// depends on `--model`:
    ///
    /// * `flux-canny-dev` — canny edge map at the target resolution
    ///   (channel-concat into a 128ch `img_in`)
    /// * `flux-depth-dev` — depth map at the target resolution
    ///   (channel-concat into a 128ch `img_in`)
    /// * `flux-kontext-dev` (v0.18) — the literal reference image to
    ///   edit (VAE-encoded, sequence-concat onto the noise tokens with
    ///   `img_ids[..., 0] = 1` as the RoPE marker; `img_in` stays at
    ///   64 channels)
    ///
    /// Required on the concept variants when `--concept-from` isn't
    /// supplied; required on Kontext (no auto-annotate equivalent —
    /// Kontext wants the actual reference, not a derived map).
    /// Ignored on other models. Mutually exclusive with `--concept-from`.
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

    /// v0.18 phase 2b: snap `--size` to the closest of 17 BFL-
    /// recommended Kontext resolutions before VAE encoding. Off by
    /// default (the user's `--size` flows through verbatim — surprise-
    /// free for non-Kontext workflows). On `--model flux-kontext-dev`,
    /// the snap matches diffusers' default behaviour and produces the
    /// best-trained-quality outputs. The 17 buckets span 9:21 → 21:9
    /// at ~1M-token budgets, all multiples of 16. Ignored on every
    /// other model.
    #[arg(long = "kontext-bucket", default_value_t = false)]
    pub kontext_bucket: bool,

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

    /// v0.16 phase 8: enable Hires fix workflow. After the t2i pass,
    /// upscales each output by `--hires-scale` and runs img2img at
    /// `--hires-strength` to recover small-scale detail. Standard
    /// mitigation for the "multi-head problem" when sampling SD 1.5
    /// or SDXL above their trained resolution. SD-family only;
    /// Flux / SD3 bail loud.
    #[arg(long = "hires-fix", default_value_t = false)]
    pub hires_fix: bool,

    /// v0.16 phase 8: upscale factor for the hires-fix pass. `2.0`
    /// (default) doubles each axis. Ignored for ML upscalers
    /// (Real-ESRGAN) which use their native fixed scale (2× / 4×).
    #[arg(long = "hires-scale", default_value_t = 2.0, value_name = "F")]
    pub hires_scale: f32,

    /// v0.16 phase 8: img2img strength on the upscaled image. `0.5`
    /// (default) preserves the t2i composition + adds refinement;
    /// `0.7+` allows more reinterpretation.
    #[arg(long = "hires-strength", default_value_t = 0.5, value_name = "F")]
    pub hires_strength: f32,

    /// v0.16 phase 8: upscaler for the hires-fix pass. Accepts the
    /// same tokens as `plakat upscale --method`:
    /// `lanczos | bicubic | bilinear | nearest | real-esrgan-x2 |
    /// real-esrgan-x4 | real-esrgan-anime-x4`.
    /// Classical filters are fast + sharp; Real-ESRGAN reconstructs
    /// high-frequency detail at extra compute cost.
    #[arg(long = "hires-upscaler", default_value = "lanczos", value_name = "MODE")]
    pub hires_upscaler: String,

    /// v0.16 phase 8: optional step-count override for the refine
    /// pass. Defaults to the main `--steps`.
    #[arg(long = "hires-steps", value_name = "N")]
    pub hires_steps: Option<usize>,

    /// v0.16 phase 9: Textual Inversion (embedding) spec. Repeatable.
    /// Format: `PATH_OR_REPO[:trigger][:scale]`.
    ///
    /// Examples:
    ///   `--embedding ./my-style.safetensors`
    ///   `--embedding ./my-style.safetensors:custom-trigger`
    ///   `--embedding ./my-style.safetensors:custom-trigger:0.7`
    ///   `--embedding sd-concepts-library/cat-toy`
    ///
    /// SD 1.5 / SD 2.1 only — SDXL dual-encoder TIs bail loud in
    /// the parser. Use `plakat embedding info PATH` to inspect a
    /// TI file's trigger word + dims before generating.
    ///
    /// Runtime injection works (v0.30): the TI is injected at runtime via the
    /// vendored CLIP path — the corpus `embedding.sh` proves it (baseline vs
    /// +EasyNegative on SD 1.5).
    #[arg(long = "embedding", value_name = "SPEC")]
    pub embeddings: Vec<crate::pipelines::embedding::EmbeddingSpec>,

    /// Disable the v0.17 PNG `parameters` tEXt-chunk metadata + the
    /// sibling `.json` sidecar. By default plakat writes
    /// Auto1111-compatible recipe metadata into every output so
    /// any viewer (A1111 Web UI, Civitai upload, ComfyUI
    /// drag-to-load, sd-prompt-reader, ...) can surface the
    /// prompt, seed, sampler, LoRAs, etc. Pass `--no-metadata` to
    /// get anonymous PNGs identical to pre-v0.17 plakat.
    #[arg(long = "no-metadata", default_value_t = false)]
    pub no_metadata: bool,

    /// v0.19: output image container. `png` (default) writes the
    /// Auto1111-compatible `parameters` tEXt chunk that A1111 /
    /// Civitai / ComfyUI auto-read on drag-and-drop. `webp` ships
    /// ~30% smaller files at perceptually-equivalent quality but
    /// CAN'T carry the tEXt chunk (WebP's EXIF / XMP slots aren't
    /// part of the diffusion-tools metadata convention). The JSON
    /// sidecar is written for both formats — `plakat metadata` and
    /// `plakat clone` work on WebP outputs via the sidecar.
    ///
    /// Currently honoured by the SD-family pipeline (SD 1.5 / SD
    /// 2.1 / SDXL / SDXL-Turbo). Flux / SD3 outputs stay PNG-only
    /// in this release; passing `--format webp` with those models
    /// emits a warning and falls back to PNG.
    #[arg(long, default_value = "png", value_name = "FORMAT")]
    pub format: crate::imaging::io::OutputFormat,

    /// v0.17 phase 4: with `--count N > 1`, also write a single
    /// `plakat-grid-<base-seed>.png` combining all N outputs in a
    /// near-square grid. Per-image PNGs are written as usual
    /// alongside.
    #[arg(long = "grid", default_value_t = false)]
    pub grid: bool,

    /// v0.17 phase 4: column count for `--grid`. Default is
    /// `ceil(sqrt(count))` — 4 → 2×2, 6 → 3×2, 9 → 3×3, 16 → 4×4.
    /// Ignored when `--grid` is off.
    #[arg(long = "grid-cols", value_name = "N")]
    pub grid_cols: Option<usize>,

    /// v0.17 phase 4: padding (px) between grid cells. Default 0
    /// (flush). Higher values insert a white border between cells
    /// for clearer per-cell separation. Ignored when `--grid` is
    /// off.
    #[arg(long = "grid-padding", default_value_t = 0, value_name = "PX")]
    pub grid_padding: u32,

    /// v0.17 phase D: write a low-cost latent-projection preview
    /// PNG every N denoise steps so you can monitor progress
    /// without waiting for the full run. Output goes to
    /// `<out>/plakat-<seed>-preview.png` and is overwritten each
    /// step. `0` disables. Uses the community latent → RGB
    /// projection (microseconds — no VAE decode), so the preview
    /// adds no meaningful runtime cost. SD 1.5 / 2.1 / SDXL /
    /// SDXL-Turbo only; Flux / SD3 ignore.
    #[arg(long = "preview-every", default_value_t = 0, value_name = "N")]
    pub preview_every: u32,

    /// v0.17 phase D: longer-side dimension (px) of the preview
    /// PNG. Default 384. Smaller = faster writes; larger = more
    /// detail in the live preview.
    #[arg(long = "preview-size", default_value_t = 384, value_name = "PX")]
    pub preview_size: u32,
}

pub async fn run(mut args: GenerateArgs, device: Device) -> Result<()> {
    // v0.33 phase 1: actionable hint when --model is a typo'd alias.
    // The check skips org/name shapes (those go through the HF
    // resolver's friendly_error path) and the recipe-load path
    // below (which may replace args.model from the recipe's
    // serialized canonical name).
    let known = crate::hf::all_known_aliases();
    crate::error_hints::hint_unknown_alias(&args.model, &known)?;

    // Memory safety (recs #3 + the host-crash guard): warn up-front if RAM is
    // already tight, then run a watchdog for the whole load+generate that aborts
    // plakat before a unified-memory exhaustion can crash the host. Bound to a
    // named local so it lives until run() returns.
    crate::hw::memory_preflight(&device, &args.model);
    let _mem_guard = crate::memwatch::MemoryGuard::start(&device, &args.model);

    // v0.20: apply --recipe FIRST so subsequent flags + downstream
    // resolution (negative-preset combine, wildcards, enhance,
    // dispatch) operate against the merged config. Recipe fields
    // fill in args left at clap defaults; explicit CLI flags win.
    if let Some(path) = args.recipe.clone() {
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("reading --recipe {}", path.display()))?;
        let recipe: crate::imaging::metadata::GenerationMetadata =
            serde_json::from_str(&json).with_context(|| {
                format!("parsing --recipe JSON {}", path.display())
            })?;
        apply_recipe(&mut args, &recipe)?;
        tracing::info!(
            target: "plakat",
            "Applied recipe {} (model={}, seed={}, loras={})",
            path.display(),
            args.model,
            args.seed.map(|s| s.to_string()).unwrap_or_else(|| "(random)".into()),
            args.loras.len()
        );
    }

    // v0.19: resolve --negative-preset FIRST so the combined
    // negative flows into every downstream step (wildcards on the
    // negative branch, enhance call, encoder dispatch). Bails up
    // front on a typo'd preset name — no point waiting for a
    // 30-second model load to discover the error.
    args.negative = crate::prompt::negative_presets::combine(
        args.negative_preset.as_deref(),
        &args.negative,
    )?;

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
        args.prompt = maybe_keep_original(
            &args.model,
            enhanced,
            &original,
            args.enhance_keep_original,
        );
    }

    // v0.18: A1111 inline <lora:name[:weight]> syntax. Extract
    // tags from both the positive and negative prompts; the
    // negatives are stripped silently (A1111 convention — LoRAs
    // don't apply via the uncond branch). Order: wildcards (above)
    // → enhance (above) → lora-tags (here) → attention syntax (in
    // the encoder). LoRAs land on top of any --lora CLI args.
    if crate::prompt::lora_tags::has_lora_tags(&args.prompt) {
        let (cleaned, extracted) = crate::prompt::lora_tags::extract(&args.prompt)?;
        if !extracted.is_empty() {
            tracing::info!(
                target: "plakat",
                "Extracted {} inline <lora:> tag(s) from prompt",
                extracted.len()
            );
            for ex in extracted.into_iter().rev() {
                // Insert at the front so explicit --lora flags retain
                // their relative order at the END (later entries win
                // on key collision during merge).
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

    let (width, height) =
        crate::imaging::sizes::resolve(args.size, args.aspect.as_deref(), args.base)?;
    crate::imaging::sizes::warn_large_for_metal(width, height, &device);
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
        let m = args.model.to_lowercase();
        match preset.target {
            crate::pipelines::flux_fast::FastTarget::Flux => {
                if !m.contains("flux") {
                    anyhow::bail!(
                        "--fast {} requires a Flux model (got --model {:?}). \
                         Hyper-FLUX / FLUX-Turbo LoRAs are Flux-family only.",
                        preset.name,
                        args.model
                    );
                }
                if m.contains("fill") {
                    anyhow::bail!(
                        "--fast {} doesn't compose with flux-fill-dev. Use the \
                         standard flux-dev model with the distillation LoRA, then \
                         handle inpainting separately.",
                        preset.name
                    );
                }
                if m.contains("nf4") {
                    anyhow::bail!(
                        "--fast {} bails on NF4 — NF4 + LoRA composition isn't \
                         wired. Use --model flux-dev or flux-dev-gguf with the \
                         preset.",
                        preset.name
                    );
                }
            }
            crate::pipelines::flux_fast::FastTarget::Sdxl => {
                // SDXL family covers both `sdxl` and `sdxl-turbo`.
                let is_sdxl = m == "sdxl"
                    || m == "sdxl-turbo"
                    || m.contains("xl-base")
                    || m.contains("sdxl-base")
                    || m.contains("stable-diffusion-xl");
                if !is_sdxl {
                    anyhow::bail!(
                        "--fast {} (Latent Consistency LoRA for SDXL) requires an \
                         SDXL model (got --model {:?}). Use --model sdxl or \
                         --model sdxl-turbo.",
                        preset.name,
                        args.model
                    );
                }
                if args.refiner {
                    anyhow::bail!(
                        "--fast {} doesn't compose with the SDXL refiner — the \
                         refiner runs a non-LCM scheduler on the late steps which \
                         conflicts with the 4-step LCM schedule. Drop --refiner.",
                        preset.name
                    );
                }
            }
            crate::pipelines::flux_fast::FastTarget::Sd15 => {
                // SD 1.5 family — exclude SDXL (which contains "xl") and
                // the inpaint variant (mask + 4-step distillation interact
                // the same way Flux Fill + Hyper does — bail loud).
                let is_sd15 = (m == "sd15"
                    || m == "sd1.5"
                    || m == "sd-1.5"
                    || m.contains("v1-5")
                    || m.contains("stable-diffusion-v1-5"))
                    && !m.contains("xl");
                if !is_sd15 {
                    anyhow::bail!(
                        "--fast {} (Latent Consistency LoRA for SD 1.5) requires an \
                         SD 1.5 model (got --model {:?}). Use --model sd15.",
                        preset.name,
                        args.model
                    );
                }
                if m.contains("inpaint") {
                    anyhow::bail!(
                        "--fast {} doesn't compose with SD 1.5 inpaint — the \
                         mask-driven denoise interacts oddly with the 4-step LCM \
                         schedule. Use --model sd15 and handle inpaint separately.",
                        preset.name
                    );
                }
            }
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
        // v0.17 phase I: presets that target a specific scheduler
        // (e.g. LCM-LoRA → `lcm`) override the user's --scheduler
        // when it's still the default. Explicit non-default
        // `--scheduler` values stay honoured — power users know
        // what they're doing.
        if let Some(sched_hint) = preset.scheduler_hint {
            use crate::pipelines::scheduler::SchedulerKind;
            use std::str::FromStr;
            if matches!(args.scheduler, SchedulerKind::Default) {
                args.scheduler = SchedulerKind::from_str(sched_hint)
                    .unwrap_or(SchedulerKind::Default);
            }
        }
        crate::ui::progress::println(&format!(
            "  fast preset '{}': +{} LoRA, steps={}, guidance={}{}",
            preset.name,
            preset.lora_repo,
            args.steps,
            args.guidance,
            preset
                .scheduler_hint
                .map(|s| format!(", scheduler={s}"))
                .unwrap_or_default(),
        ));
    }

    // v0.25 phases 3–5: --look and --genre presets + auto-LoRA
    // discovery. Runs AFTER --fast so distillation step counts
    // (e.g. hyper-8 → 8 steps) aren't overwritten by a softer
    // medium suggestion. Override-only-if-user-didn't-pass: matches
    // the clap-default-comparison trick the fast preset uses above.
    // The shared helper lives in src/preset/mod.rs so portrait /
    // img2img / outpaint can reuse it (phase 6).
    if args.look.is_some() || args.genre.is_some() {
        use crate::preset::{GenerationParams, apply_presets_with_discovery};
        use std::str::FromStr;

        let mut params = GenerationParams {
            prompt: args.prompt.clone(),
            negative: args.negative.clone(),
            // None == "user didn't touch" → preset may fill.
            steps: (args.steps != 28).then_some(args.steps),
            guidance: ((args.guidance - 7.5).abs() >= f64::EPSILON)
                .then_some(args.guidance),
            // Scheduler is an enum, not a string. Empty-string
            // sentinel means "user passed something non-Default,
            // block preset"; a non-empty string means "preset wrote
            // a name we should parse back into SchedulerKind".
            scheduler: (!matches!(args.scheduler, SchedulerKind::Default))
                .then(String::new),
        };
        apply_presets_with_discovery(
            args.look.as_deref(),
            args.genre.as_deref(),
            args.offline,
            crate::preset::discovery::BaseFamily::from_model_arg(&args.model),
            args.smart_discovery,
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

    // v0.30 phase 1: LCM-LoRA auto-detection.
    apply_lcm_override(&mut args);

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
    // v0.33 phase 1: capture model name before move for the OOM
    // decorator below.
    let model_for_oom = args.model.clone();
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
        regions: args
            .region
            .iter()
            .map(|s| crate::pipelines::tiled::RegionSpec::parse(s))
            .collect::<Result<Vec<_>>>()?,
        quantize_t5: args.quantize_t5,
        flux_quant_level: args.flux_quant_level,
        t5_quant_level: args.t5_quant_level,
        redux_images: args.redux_images,
        // v0.15 phase 4: conditioning map for Flux Canny-dev / Depth-dev.
        flux_concept_image: args.concept_image,
        // v0.18 phase 2b: opt-in Kontext aspect-bucket snap. Only
        // honoured on `--model flux-kontext-dev`; ignored elsewhere.
        kontext_bucket: args.kontext_bucket,
        // v0.16 phase 5: CLIP-skip. SD 1.5 / SD 2.1 only.
        clip_skip: args.clip_skip,
        // v0.16 phase 9: TI specs. sd_core::load bails loud when
        // these are non-empty (candle vocab_size API blocker).
        embeddings: args.embeddings,
        // v0.17 phase 3: embed A1111-compatible PNG metadata +
        // write a sibling JSON sidecar. Default on; --no-metadata
        // flips it off.
        write_metadata: !args.no_metadata,
        // v0.17 phase D: latent-projection preview cadence. `0`
        // (default) → no previews; downstream pipeline treats
        // `Some(0)` the same way.
        preview_every: if args.preview_every > 0 {
            Some(args.preview_every)
        } else {
            None
        },
        preview_size: Some(args.preview_size),
        // v0.19: pass through the --format flag. SD-family
        // pipeline honours; Flux + SD3 fallback below.
        output_format: args.format,
        // v0.33 phase 0 — metadata polish fields. Flow into
        // `GenerationMetadata` for downstream tooling. None when
        // the user didn't supply the corresponding flag.
        look: args.look.clone(),
        genre: args.genre.clone(),
        negative_preset: args.negative_preset.clone(),
        // The structured stacks aren't built from CLI args directly —
        // the t2i pipeline resolves loras / TIs / controls during
        // load and we'd surface the resolved info here. For phase 0
        // we leave them as None; later in the cycle (phase 2 JSON
        // sidecar) the pipeline will fill them from its resolved
        // state at write-metadata time.
        lora_stack: None,
        embedding_stack: None,
        control_stack: None,
        enhancement: None,
        // v0.38 phase 2: Stable Cascade Stage C / Stage B step
        // overrides. Plain `--steps` still works; these refine it
        // when set. Ignored on every non-Cascade pipeline.
        cascade_stage_c_steps: args.stage_c_steps,
        cascade_decoder_guidance: args.decoder_guidance,
        cascade_stage_b_steps: args.stage_b_steps,
        cascade_image_prompt: args.image_variation.clone(),
        // v0.38 phase 5: Cascade ControlNet weights path.
        cascade_controlnet_weights: args.cascade_control_weights,
    })
    .await
    .map_err(|e| {
        // v0.33 phase 1: decorate OOM errors with pipeline-specific
        // mitigation suggestions. Detection is conservative (looks
        // for "out of memory" / "OOM" substrings); unrelated errors
        // pass through unchanged.
        let ctx = if model_for_oom.contains("flux") {
            crate::error_hints::OomContext::Flux
        } else if model_for_oom.contains("xl") {
            crate::error_hints::OomContext::Sdxl
        } else {
            crate::error_hints::OomContext::Sd
        };
        crate::error_hints::decorate_oom(e, ctx)
    })?;

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

    // v0.16 phase 8: Hires fix runs BEFORE ADetailer + artefacts so
    // the face refinement and artefact placement operate on the
    // upscaled image (refined faces, correctly-sized stamps).
    //
    // Gated against `--artefacts` / `--artefact-blend` since those
    // pipelines reference the original t2i dimensions and would
    // misplace stamps after the upscale changes dims. Drop one or
    // the other — Auto1111 has similar mutual-exclusivity quirks.
    if args.hires_fix {
        let variant = crate::pipelines::t2i::Variant::detect(&model);
        if variant.is_flux() || variant.is_sd3() {
            anyhow::bail!(
                "--hires-fix requires an SD-family model (SD 1.5 / SD 2.1 / SDXL / \
                 SDXL-Turbo). Got --model {} which routes through the {} pipeline. \
                 SD-family models can run the post-t2i hires-fix refine pass; Flux \
                 / SD3 already have native tiled paths for high-res output \
                 (--tiled).",
                model,
                if variant.is_flux() { "Flux" } else { "SD3" }
            );
        }
        if !args.artefacts.is_empty() || args.artefact_blend {
            anyhow::bail!(
                "--hires-fix doesn't compose with --artefact / --artefact-blend \
                 yet. The hires upscale changes image dims; the artefact compositor \
                 reads them from the t2i config and would misplace stamps. Drop one \
                 or the other."
            );
        }
        let upscaler: crate::imaging::upscale::Method = args
            .hires_upscaler
            .parse()
            .with_context(|| format!("parsing --hires-upscaler {:?}", args.hires_upscaler))?;
        if upscaler.is_ml() && (args.hires_scale - 2.0).abs() > f32::EPSILON {
            // Honour the user's choice silently if they're using a
            // classical upscaler at non-2x. The ML branch is fixed by
            // the model, so log when the user passed something else.
            tracing::info!(
                target: "plakat",
                "--hires-scale {} ignored for ML upscaler (uses native {}×)",
                args.hires_scale,
                upscaler.native_scale().unwrap_or(0.0),
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
            let hires_cfg = crate::pipelines::hires_fix::Config {
                model: model.clone(),
                loras: loras.clone(),
                lora_scale,
                prompt: prompt.clone(),
                negative: negative.clone(),
                scale: args.hires_scale,
                upscaler,
                strength: args.hires_strength,
                steps: args.hires_steps.unwrap_or(steps),
                guidance,
                scheduler,
                device: device.clone(),
            };
            let spin = crate::ui::progress::spinner(&format!(
                "Hires fix over {} image(s)", files.len()
            ));
            let n = crate::pipelines::hires_fix::refine_files(
                &hires_cfg,
                &files,
                shared_core.clone(),
            )
            .await?;
            spin.finish_with_message(format!(
                "✓ Hires fix refined {n} image(s)"
            ));
        }
    }

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

    // v0.17 phase 4: --grid bundles the per-image outputs into one
    // shareable grid PNG. Runs LAST so artefacts + blend + face
    // refinement are all reflected in the grid cells. No-op when
    // --count is 1 (a 1-cell "grid" is just a copy).
    if args.grid && count > 1 {
        // v0.19: per-image filenames use the user's --format
        // extension (png by default; webp when set). The grid PNG
        // itself stays .png for compatibility — combining N WebP
        // cells into one shareable grid file is the common
        // workflow even when individual cells are WebP-encoded.
        let img_ext = args.format.extension();
        let files: Vec<PathBuf> = (0..count)
            .map(|i| {
                let s = seed.unwrap_or(0).wrapping_add(i as u64);
                out_dir.join(format!("plakat-{s}.{img_ext}"))
            })
            .filter(|p| p.exists())
            .collect();
        if files.len() >= 2 {
            let base_seed = seed.unwrap_or(0);
            let grid_path = out_dir.join(format!("plakat-grid-{base_seed}.png"));
            let spin = crate::ui::progress::spinner(&format!(
                "Composing {}-cell grid", files.len()
            ));
            let (gw, gh) = crate::imaging::grid::write_grid(
                &files,
                &grid_path,
                args.grid_cols,
                args.grid_padding,
            )?;
            spin.finish_with_message(format!(
                "✓ grid {gw}x{gh} → {}",
                grid_path.display()
            ));
        }
    }
    Ok(())
}

/// v0.16 phase 5: expand `{a|b|c}` and `__name__` wildcards in both
/// the prompt and negative prompt. The wildcard RNG is seeded from
/// v0.20: apply a generation recipe (JSON sidecar) as default
/// values for `GenerateArgs`. CLI flags that the user explicitly
/// set (i.e. differ from clap's built-in default) win; flags left
/// at their defaults inherit from the recipe. The positional
/// `prompt` is never overridden — use `plakat clone PNG` for a
/// byte-equivalent rerun.
///
/// Detection of "user set this explicitly" is by value comparison
/// against the known clap defaults declared in the
/// `#[arg(default_value = ...)]` attributes above. Edge case: a
/// user who explicitly passes `--model sd15` (the default) can't
/// be distinguished from a user who didn't pass `--model` at all
/// — in both cases the recipe's `model` field wins. Acceptable
/// for v0.20; matches diffusers' parser-recipe behaviour.
fn apply_recipe(
    args: &mut GenerateArgs,
    recipe: &crate::imaging::metadata::GenerationMetadata,
) -> Result<()> {
    use crate::imaging::sizes::Size;
    use crate::pipelines::lora::LoraSpec;
    use crate::pipelines::scheduler::SchedulerKind;
    use std::str::FromStr;

    // model: default "sd15".
    if args.model == "sd15" {
        args.model = recipe.model.clone();
    }
    // negative: default "".
    if args.negative.is_empty() {
        args.negative = recipe.negative.clone();
    }
    // seed: default None.
    if args.seed.is_none() {
        args.seed = Some(recipe.seed);
    }
    // steps: default 28.
    if args.steps == 28 {
        args.steps = recipe.steps;
    }
    // guidance: default 7.5.
    if (args.guidance - 7.5).abs() < f64::EPSILON {
        args.guidance = recipe.guidance;
    }
    // scheduler: default SchedulerKind::Default.
    if matches!(args.scheduler, SchedulerKind::Default) {
        if let Ok(s) = SchedulerKind::from_str(&recipe.scheduler) {
            args.scheduler = s;
        }
    }
    // size: default None. Apply width/height from recipe.
    if args.size.is_none() && args.aspect.is_none() {
        args.size = Some(Size {
            w: recipe.width,
            h: recipe.height,
        });
    }
    // loras: default empty Vec. Parse each recipe spec.
    if args.loras.is_empty() {
        let mut parsed: Vec<LoraSpec> = Vec::with_capacity(recipe.loras.len());
        for spec in &recipe.loras {
            // Recipe LoRA strings are debug-formatted in the v0.17
            // metadata writer (`format!("{:?}", spec)`). Strip the
            // outer quotes and reparse — best-effort: skip the
            // entry with a warn if it doesn't parse rather than
            // bailing the whole generation.
            let cleaned = spec.trim_matches('"');
            match cleaned.parse::<LoraSpec>() {
                Ok(p) => parsed.push(p),
                Err(e) => tracing::warn!(
                    target: "plakat",
                    "recipe LoRA {spec:?} failed to parse: {e} — skipping"
                ),
            }
        }
        args.loras = parsed;
    }
    // lora_scale: default 1.0.
    if (args.lora_scale - 1.0).abs() < f32::EPSILON {
        if let Some(s) = recipe.lora_scale {
            args.lora_scale = s;
        }
    }
    // clip_skip: default 1.
    if args.clip_skip == 1 {
        if let Some(cs) = recipe.clip_skip {
            args.clip_skip = cs;
        }
    }
    Ok(())
}

/// v0.20: implement `--enhance-keep-original`. When the flag is
/// set on an SD-family model, the enhancer's rewrite and the
/// user's original prompt get joined with the A1111 `BREAK`
/// keyword — each chunk gets its own 77-token CLIP slot, so the
/// original terms keep their full attention weight instead of
/// being diluted by the enhancer's added detail. On Flux / SD3
/// the flag is a no-op with a one-line warn: their T5 text
/// encoder ignores `BREAK` and has the token budget to carry
/// both phrasings without it, so the original prompt isn't at
/// risk of being clipped.
///
/// Empty originals (`""` after the wildcard/style passes) also
/// short-circuit to the enhanced text alone — joining with
/// `BREAK` would produce a trailing-empty chunk that the
/// chunk-splitter would drop anyway.
pub(crate) fn maybe_keep_original(
    model: &str,
    enhanced: String,
    original: &str,
    keep_original: bool,
) -> String {
    if !keep_original || original.trim().is_empty() {
        return enhanced;
    }
    let variant = crate::pipelines::t2i::Variant::detect(model);
    if variant.is_flux() || variant.is_sd3() {
        tracing::warn!(
            target: "plakat",
            "--enhance-keep-original ignored on Flux/SD3 model {:?}: \
             BREAK is CLIP-only, and T5 already has the token budget \
             to carry both phrasings. The enhancer's output is used as-is.",
            model
        );
        return enhanced;
    }
    format!(
        "{enhanced} {break_kw} {original}",
        break_kw = crate::prompt::break_chunks::BREAK_KEYWORD
    )
}

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

/// v0.30 phase 1: apply the LCM-LoRA override to `GenerateArgs`.
///
/// Runs after `--fast` / `--look` / `--genre` preset application so
/// preset-added LoRAs participate. Fires when either:
///   * `--lcm` is explicit, OR
///   * any LoRA in the (possibly preset-extended) stack matches the
///     `lcm` substring heuristic (see `pipelines::lora::is_lcm_lora_spec`).
///
/// Override applies only to args still at their clap defaults
/// (steps=28, guidance=7.5, scheduler=Default) — explicit user
/// values stay honoured. Mirrors the v0.28 `--lcm` flag handling
/// in `cli/animate.rs`.
fn apply_lcm_override(args: &mut GenerateArgs) {
    let lcm_active = args.lcm
        || args
            .loras
            .iter()
            .any(crate::pipelines::lora::is_lcm_lora_spec);
    if !lcm_active {
        return;
    }
    if matches!(args.scheduler, SchedulerKind::Default) {
        args.scheduler = SchedulerKind::Lcm;
    }
    if args.steps == 28 {
        args.steps = 4;
    }
    if (args.guidance - 7.5).abs() < f64::EPSILON {
        args.guidance = 1.5;
    }
    tracing::info!(
        target: "plakat",
        "LCM-LoRA detected (explicit={}): scheduler={:?}, steps={}, guidance={}",
        args.lcm,
        args.scheduler,
        args.steps,
        args.guidance
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::metadata::GenerationMetadata;

    /// v0.20: helper to construct a `GenerateArgs` with clap's
    /// declared defaults. Mirrors the `#[arg(default_value = ...)]`
    /// attributes above. Used by the recipe-override tests.
    fn mk_default_args(prompt: &str) -> GenerateArgs {
        use crate::pipelines::scheduler::SchedulerKind;
        GenerateArgs {
            prompt: prompt.into(),
            recipe: None,
            model: "sd15".into(),
            size: None,
            aspect: None,
            base: 768,
            count: 1,
            steps: 28,
            stage_c_steps: None,
            stage_b_steps: None,
            cascade_control_weights: None,
            image_variation: None,
            guidance: 7.5,
            decoder_guidance: 1.1,
            negative: String::new(),
            negative_preset: None,
            seed: None,
            enhance: None,
            enhance_system: None,
            enhance_temp: None,
            enhance_max_tokens: None,
            enhance_cache: false,
            enhance_keep_original: false,
            wildcard_dir: None,
            clip_skip: 1,
            out: PathBuf::from("./out"),
            loras: Vec::new(),
            lora_scale: 1.0,
            scheduler: SchedulerKind::Default,
            lcm: false,
            refine: None,
            refine_strength: 0.4,
            refiner: false,
            refiner_frac: 0.8,
            style_ref: None,
            style: None,
            style_strength: 0.6,
            style_catalog: None,
            artefacts: Vec::new(),
            artefact_library: None,
            artefact_blend: false,
            artefact_blend_strength: 0.3,
            smart_zones: false,
            control: None,
            control_image: None,
            control_from: None,
            control_strength: 1.0,
            control_start: 0.0,
            control_end: 1.0,
            control_specs: Vec::new(),
            tiled: false,
            region: Vec::new(),
            tile_size: 1024,
            tile_stride: 768,
            quantize_t5: false,
            flux_quant_level: None,
            t5_quant_level: None,
            fast: None,
            look: None,
            genre: None,
            offline: false,
            smart_discovery: false,
            redux_images: Vec::new(),
            concept_image: None,
            concept_from: None,
            kontext_bucket: false,
            adetailer: false,
            adetailer_strength: 0.4,
            adetailer_padding: 0.25,
            adetailer_feather: 0.25,
            adetailer_confidence: 0.5,
            adetailer_size: 512,
            adetailer_prompt: None,
            hires_fix: false,
            hires_scale: 2.0,
            hires_strength: 0.5,
            hires_upscaler: "lanczos".into(),
            hires_steps: None,
            embeddings: Vec::new(),
            no_metadata: false,
            format: crate::imaging::io::OutputFormat::Png,
            grid: false,
            grid_cols: None,
            grid_padding: 0,
            preview_every: 0,
            preview_size: 384,
        }
    }

    fn mk_recipe() -> GenerationMetadata {
        let mut m = GenerationMetadata::new(
            "the recipe's prompt",
            "sdxl",
            12345,
            42,
            6.0,
            "dpmpp-2m",
            1024,
            768,
        );
        m.negative = "blurry".into();
        m.loras = vec!["civitai:777:0.6".into()];
        m.lora_scale = Some(0.5);
        m.clip_skip = Some(2);
        m
    }

    #[test]
    fn recipe_overrides_default_model() {
        let mut args = mk_default_args("user prompt");
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.model, "sdxl");
    }

    #[test]
    fn cli_explicit_model_wins_over_recipe() {
        let mut args = mk_default_args("user prompt");
        args.model = "flux-dev".into(); // user set --model flux-dev
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.model, "flux-dev");
    }

    #[test]
    fn recipe_fills_seed_when_unset() {
        let mut args = mk_default_args("p");
        assert!(args.seed.is_none());
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.seed, Some(12345));
    }

    #[test]
    fn cli_seed_wins_over_recipe() {
        let mut args = mk_default_args("p");
        args.seed = Some(99);
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.seed, Some(99));
    }

    #[test]
    fn recipe_overrides_default_steps_and_guidance() {
        let mut args = mk_default_args("p");
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.steps, 42);
        assert!((args.guidance - 6.0).abs() < 1e-6);
    }

    #[test]
    fn cli_non_default_steps_wins() {
        let mut args = mk_default_args("p");
        args.steps = 50; // user set --steps 50
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.steps, 50);
    }

    #[test]
    fn recipe_fills_size_when_unset() {
        let mut args = mk_default_args("p");
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        let size = args.size.unwrap();
        assert_eq!((size.w, size.h), (1024, 768));
    }

    #[test]
    fn cli_size_wins_over_recipe() {
        use crate::imaging::sizes::Size;
        let mut args = mk_default_args("p");
        args.size = Some(Size { w: 512, h: 512 });
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        let size = args.size.unwrap();
        assert_eq!((size.w, size.h), (512, 512));
    }

    #[test]
    fn recipe_loras_parse_via_lora_spec_from_str() {
        let mut args = mk_default_args("p");
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.loras.len(), 1);
        // Scale was 0.6 in the recipe's "civitai:777:0.6".
        assert!((args.loras[0].scale - 0.6).abs() < 1e-6);
    }

    #[test]
    fn recipe_clip_skip_applies_when_default() {
        let mut args = mk_default_args("p");
        assert_eq!(args.clip_skip, 1);
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.clip_skip, 2);
    }

    #[test]
    fn recipe_negative_applies_when_empty() {
        let mut args = mk_default_args("p");
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.negative, "blurry");
    }

    #[test]
    fn cli_negative_wins_over_recipe() {
        let mut args = mk_default_args("p");
        args.negative = "ugly hands".into();
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        assert_eq!(args.negative, "ugly hands");
    }

    #[test]
    fn recipe_skips_loras_when_cli_has_some() {
        use crate::pipelines::lora::{LoraSource, LoraSpec};
        let mut args = mk_default_args("p");
        args.loras = vec![LoraSpec {
            source: LoraSource::Local(PathBuf::from("foo.safetensors")),
            scale: 1.0,
        }];
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        // CLI LoRA stack untouched.
        assert_eq!(args.loras.len(), 1);
        match &args.loras[0].source {
            LoraSource::Local(p) => {
                assert_eq!(p.to_str().unwrap(), "foo.safetensors")
            }
            other => panic!("expected Local, got {other:?}"),
        }
    }

    #[test]
    fn prompt_is_never_overridden_by_recipe() {
        let mut args = mk_default_args("user's own prompt");
        apply_recipe(&mut args, &mk_recipe()).unwrap();
        // The recipe's prompt field is "the recipe's prompt" but
        // apply_recipe never touches args.prompt.
        assert_eq!(args.prompt, "user's own prompt");
    }

    // v0.20 #7: --enhance-keep-original behaviour.

    #[test]
    fn keep_original_off_returns_enhanced_unchanged() {
        let out = maybe_keep_original(
            "sd15",
            "an enhanced detailed prompt".to_string(),
            "original",
            false,
        );
        assert_eq!(out, "an enhanced detailed prompt");
    }

    #[test]
    fn keep_original_on_sd15_inserts_break() {
        let out = maybe_keep_original(
            "sd15",
            "an enhanced detailed prompt".to_string(),
            "a cat on a couch",
            true,
        );
        assert_eq!(out, "an enhanced detailed prompt BREAK a cat on a couch");
    }

    #[test]
    fn keep_original_on_sdxl_inserts_break() {
        let out = maybe_keep_original(
            "sdxl",
            "enhanced".to_string(),
            "original",
            true,
        );
        assert!(out.contains(" BREAK "), "got {out}");
    }

    #[test]
    fn keep_original_on_flux_is_noop() {
        let out = maybe_keep_original(
            "flux-dev",
            "enhanced".to_string(),
            "original",
            true,
        );
        // No BREAK on Flux — T5 ignores it. Enhanced text only.
        assert_eq!(out, "enhanced");
    }

    #[test]
    fn keep_original_on_sd3_is_noop() {
        let out = maybe_keep_original(
            "sd35-medium",
            "enhanced".to_string(),
            "original",
            true,
        );
        assert_eq!(out, "enhanced");
    }

    #[test]
    fn keep_original_with_empty_original_returns_enhanced() {
        // A wildcard pass that produced an empty original (or the
        // user passed `""`) would yield a trailing-empty chunk after
        // BREAK that the chunk-splitter drops anyway — skip the
        // concat so logs stay clean.
        let out = maybe_keep_original(
            "sd15",
            "enhanced".to_string(),
            "   ",
            true,
        );
        assert_eq!(out, "enhanced");
    }

    #[test]
    fn keep_original_canonical_hf_repo_for_flux_also_noops() {
        // Variant detection runs on the raw `--model` string, so a
        // user-passed canonical repo path should classify the same.
        let out = maybe_keep_original(
            "black-forest-labs/FLUX.1-dev",
            "enhanced".to_string(),
            "original",
            true,
        );
        assert_eq!(out, "enhanced");
    }

    // ------------------------------------------------------------------
    // v0.30 phase 1: LCM-LoRA override.
    // ------------------------------------------------------------------

    use crate::pipelines::lora::LoraSpec;
    use std::str::FromStr;

    #[test]
    fn lcm_override_noops_without_flag_or_matching_lora() {
        let mut args = mk_default_args("a cat");
        apply_lcm_override(&mut args);
        assert_eq!(args.steps, 28);
        assert!((args.guidance - 7.5).abs() < f64::EPSILON);
        assert!(matches!(args.scheduler, SchedulerKind::Default));
    }

    #[test]
    fn lcm_override_explicit_flag_overrides_defaults() {
        let mut args = mk_default_args("a cat");
        args.lcm = true;
        apply_lcm_override(&mut args);
        assert_eq!(args.steps, 4);
        assert!((args.guidance - 1.5).abs() < f64::EPSILON);
        assert!(matches!(args.scheduler, SchedulerKind::Lcm));
    }

    #[test]
    fn lcm_override_detects_via_lora_repo_name() {
        let mut args = mk_default_args("a cat");
        args.loras.push(LoraSpec::from_str("latent-consistency/lcm-lora-sdv1-5").unwrap());
        apply_lcm_override(&mut args);
        assert_eq!(args.steps, 4);
        assert!(matches!(args.scheduler, SchedulerKind::Lcm));
    }

    #[test]
    fn lcm_override_preserves_explicit_steps() {
        let mut args = mk_default_args("a cat");
        args.lcm = true;
        args.steps = 8; // user wants higher quality at 2× cost
        apply_lcm_override(&mut args);
        assert_eq!(args.steps, 8); // not clobbered
        // Scheduler still flipped because it was at Default.
        assert!(matches!(args.scheduler, SchedulerKind::Lcm));
    }

    #[test]
    fn lcm_override_preserves_explicit_guidance() {
        let mut args = mk_default_args("a cat");
        args.lcm = true;
        args.guidance = 3.0;
        apply_lcm_override(&mut args);
        assert!((args.guidance - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn lcm_override_preserves_explicit_scheduler() {
        let mut args = mk_default_args("a cat");
        args.lcm = true;
        args.scheduler = SchedulerKind::EulerA;
        apply_lcm_override(&mut args);
        // User picked Euler-A explicitly; LCM stays off the scheduler
        // slot. (This is a known foot-gun documented in --lcm help —
        // most users will want the implicit LCM scheduler too.)
        assert!(matches!(args.scheduler, SchedulerKind::EulerA));
        // But steps + guidance still flip because they were at default.
        assert_eq!(args.steps, 4);
    }
}
