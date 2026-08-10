//! `scenario` — batch-generate images from an HJSON file that mixes scenes,
//! weather, and per-task prompts. See README for the schema and an example.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::imaging::sizes::Size;
use crate::imaging::upscale::{EsrganPipeline, Method as UpscaleMethod};
use crate::pipelines::flux;
use crate::pipelines::ip_adapter::IdentityKind;
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::portrait;
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::stylize;
use crate::pipelines::t2i::{GenRequest, LoadRequest, Pipeline, Variant};
use crate::style::{
    combine_negative, log_style_prep, parse_resolved_loras, prepend_trigger, StylePrepRequest,
    StyleSession,
};

#[derive(ClapArgs, Debug)]
pub struct ScenarioArgs {
    /// Path to the HJSON scenario file.
    pub file: PathBuf,

    /// Validate, print every task's planned prompts, but skip generation.
    /// Does NOT call the enhancer (no API cost).
    #[arg(help_heading = "Batch run", long)]
    pub dry_run: bool,

    /// v0.17 phase 5: skip tasks whose **every** expected output
    /// PNG already exists. Lets a crashed / Ctrl-C'd scenario
    /// pick up where it left off without restarting from task 0.
    /// Task name + seed-based filenames are checked under
    /// `<out>/<task-name>/`. Mutually exclusive with `--force`.
    #[arg(help_heading = "Batch run", long, default_value_t = false, conflicts_with = "force")]
    pub resume: bool,

    /// v0.17 phase 5: regenerate every task even when outputs
    /// already exist on disk. Default behaviour overwrites
    /// existing files silently — `--force` makes the intent
    /// explicit (and pairs with future safety checks). Mutually
    /// exclusive with `--resume`.
    #[arg(help_heading = "Batch run", long, default_value_t = false, conflicts_with = "resume")]
    pub force: bool,

    /// v0.19: run only the named tasks. Comma-separated list of task
    /// `name:` values from the scenario file. Useful for iterating
    /// on a single task without re-running the whole batch. Tasks
    /// not in the list are silently skipped (no output written).
    /// Composes with `--resume` (a named task already on disk still
    /// gets skipped under resume semantics).
    #[arg(help_heading = "Batch run", long, value_delimiter = ',', value_name = "NAME[,NAME,…]")]
    pub only: Vec<String>,

    /// v0.19: run only the first N tasks (in scenario file order).
    /// Handy for sanity-checking a long batch before launching the
    /// full run. `0` means "no limit" (same as omitting the flag).
    /// Composes with `--only` (the limit applies after the
    /// name-filter; e.g. `--only a,b,c --limit 2` runs `a` and `b`).
    #[arg(help_heading = "Batch run", long, default_value_t = 0, value_name = "N")]
    pub limit: u32,

    /// v0.33 phase 2: write a structured `ScenarioRunSummary` JSON
    /// at the given path after the run completes. The summary
    /// covers the scenario metadata (file, model, out dir,
    /// per-task entries with kind / status / seed, aggregate
    /// success / fail / skipped counts, total wall-clock time).
    /// Useful for CI / automation that needs to know whether a
    /// long scenario landed cleanly without parsing console
    /// output. Writes in dry-run mode too (status fields say
    /// `dry-run` per task) so plan validation has the same
    /// machine-readable shape.
    #[arg(help_heading = "Batch run", long = "json-summary", value_name = "PATH")]
    pub json_summary: Option<PathBuf>,

    /// Programmatic override for the scenario's `out:` dir (not a CLI flag). The
    /// `plakat ui` runner sets this to a path under the workspace `out/` so generated
    /// images land where History scans them, regardless of the scenario's own `out:`.
    #[arg(skip)]
    pub out_override: Option<PathBuf>,
}

// =====================================================================
// v0.33 phase 2: ScenarioRunSummary — JSON write target for the
// `--json-summary PATH` flag.
// =====================================================================

/// One task's outcome record. Populated as the scenario loop
/// dispatches each task; collected into `ScenarioRunSummary.tasks`
/// at the end.
#[derive(Debug, Clone, Serialize)]
pub struct TaskRunRecord {
    pub name: String,
    /// `"generate"` / `"animatediff"`.
    pub kind: String,
    /// `"ok"` / `"skipped"` / `"failed"` / `"dry-run"`.
    pub status: String,
    pub seed: Option<u64>,
    /// Free-form note attached on `failed` / `skipped` so consumers
    /// can pivot without parsing console output. e.g. `"--only
    /// filter excluded"`, `"--limit reached"`, `"--resume cache hit"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// v0.34 phase 2: populated only on `status: "failed"`. Carries
    /// the propagated `anyhow::Error::to_string()` from the
    /// task-dispatch site, so CI consumers don't have to scrape
    /// console output to know WHY a task failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Run-level summary written by `--json-summary PATH`.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioRunSummary {
    pub scenario_file: String,
    pub model: String,
    pub out_dir: String,
    pub total_tasks: usize,
    pub ran: usize,
    pub skipped: usize,
    pub failed: usize,
    pub wall_time_secs: f64,
    pub plakat_version: String,
    pub tasks: Vec<TaskRunRecord>,
}

#[derive(Debug, Deserialize)]
struct ScenarioFile {
    // ---------- global generation parameters ----------
    model: Option<String>,
    device: Option<String>,
    size: Option<String>,
    aspect: Option<String>,
    base: Option<u32>,
    count: Option<u32>,
    steps: Option<usize>,
    guidance: Option<f64>,
    seed: Option<u64>,
    out: Option<PathBuf>,

    #[serde(default)]
    loras: Vec<String>,
    #[serde(rename = "lora-scale")]
    lora_scale: Option<f32>,

    scheduler: Option<String>,
    refine: Option<usize>,
    #[serde(rename = "refine-strength")]
    refine_strength: Option<f32>,

    /// If true (and model is SDXL/SDXL-Turbo) use the real SDXL refiner
    /// UNet for the last fraction of every task's schedule.
    #[serde(default)]
    refiner: bool,
    #[serde(rename = "refiner-frac")]
    refiner_frac: Option<f32>,

    // ---------- v2.7: free-quality guidance bundle (env-driven; scenario-global) ----------
    /// Perturbed-Attention Guidance scale (0/omitted = off). SD 1.5/SDXL + PixArt + SD3. → PLAKAT_PAG_SCALE.
    #[serde(rename = "pag-scale")]
    pag_scale: Option<f64>,
    /// CFG-rescale φ (~0.7). Curbs high-guidance over-exposure. → PLAKAT_CFG_RESCALE.
    #[serde(rename = "guidance-rescale")]
    guidance_rescale: Option<f64>,
    /// FreeU on/off (defaults 1.2,1.4,0.9,0.2). `freeu-params` overrides. → PLAKAT_FREEU.
    #[serde(default)]
    freeu: bool,
    #[serde(rename = "freeu-params")]
    freeu_params: Option<String>,
    /// Dynamic-thresholding percentile (~99.5). Epsilon SD. → PLAKAT_DYNTHRESH.
    #[serde(rename = "dynamic-threshold")]
    dynamic_threshold: Option<f64>,

    // ---------- v0.13 phase 10: GGUF + quant + tiled ----------
    /// v0.13 phase 1b: load T5-XXL as a quantized GGUF when running
    /// Flux GGUF. Requires `model: flux-*-gguf`. Bails loud otherwise.
    #[serde(rename = "quantize-t5", default)]
    quantize_t5: bool,
    /// v0.13 phase 5: Flux GGUF quant level (e.g. `Q5_K_M`, `Q8_0`,
    /// `F16`). `None` falls back to `Q4_K_S`. Only meaningful with a
    /// GGUF Flux model.
    #[serde(rename = "flux-quant-level", default)]
    quant_level: Option<String>,
    /// v0.13 phase 5: T5-XXL GGUF quant level (e.g. `Q5_K_M`, `Q8_0`).
    /// `None` falls back to `Q4_K_S`. Only meaningful with
    /// `quantize-t5: true`.
    #[serde(rename = "t5-quant-level", default)]
    t5_quant_level: Option<String>,
    /// v0.13 phase 4/9: MultiDiffusion-style tiled denoise. When set,
    /// every task in the scenario runs through the tiled path (Flux,
    /// SDXL, SD 1.5/2.1, or SD3). Per-task `tiled:` overrides this.
    #[serde(default)]
    tiled: Option<TiledCfg>,

    /// v0.15 phase 7a: scenario-wide Flux distillation preset (Hyper-FLUX
    /// or FLUX-Turbo) — applied to every Flux task by default. Per-task
    /// `fast:` overrides. Accepts the same preset names as the CLI
    /// `--fast` flag: `hyper-8`, `hyper-16`, `turbo-alpha`. Non-Flux
    /// tasks ignore with a warning.
    #[serde(default)]
    fast: Option<String>,

    /// v0.25: scenario-wide art-medium preset. Bundled choices: `ink-wash`,
    /// `watercolor`, `oil-painting`, `charcoal`, `pencil`, `chalk-pastel`,
    /// `linocut`, `gouache`. Composes the prompt + suggests sampler /
    /// steps / guidance. Per-task `look:` overrides. Override-only:
    /// scenario-level steps / guidance / scheduler / loras take
    /// precedence.
    ///
    /// **Auto-LoRA discovery is NOT active in scenarios** (v0.25
    /// scope). The prompt prefix / suffix / sampler hints still apply;
    /// users who want a specific LoRA should supply `loras: [...]`
    /// at the scenario or task level. Discovery integration is
    /// deferred to v0.26.
    #[serde(default)]
    look: Option<String>,

    /// v0.25: scenario-wide subject-domain preset (`anime`). Independent
    /// axis from `look:`; composes additively. Per-task `genre:`
    /// overrides.
    #[serde(default)]
    genre: Option<String>,

    /// v0.25: skip remote LoRA discovery for `look:` / `genre:` —
    /// effectively a no-op in v0.25 scenarios (discovery isn't wired
    /// here yet) but accepted for forward-compatibility. Per-task
    /// `offline:` overrides.
    #[serde(default)]
    offline: Option<bool>,

    // ---------- v0.29 phase 2: scenario-level animate defaults ----------

    /// v0.29: task dispatch type. `"generate"` (default) runs the
    /// existing t2i/img2img/portrait path. `"animatediff"` (alias
    /// `"animate"`) routes each task through the AnimateDiff pipeline
    /// — every `frames`/`from`/`lcm`/etc. field below becomes
    /// meaningful. Per-task `type:` overrides.
    #[serde(default, rename = "type")]
    task_type: Option<String>,

    /// v0.29 phase 2: scenario-level animate total frame count.
    /// Per-task `frames:` overrides. Default 16 when omitted.
    #[serde(rename = "frames", default)]
    animate_frames: Option<u32>,

    /// v0.29 phase 2: per-window frame count for long-form sliding-
    /// window animate. Per-task `window-size:` overrides. Default 16.
    #[serde(rename = "window-size", default)]
    animate_window_size: Option<u32>,

    /// v0.29 phase 2: cross-fade region for sliding-window animate.
    /// Per-task `window-overlap:` overrides. Default 4.
    #[serde(rename = "window-overlap", default)]
    animate_window_overlap: Option<u32>,

    /// v0.29 phase 2: AnimateLCM 4-step mode toggle. SD 1.5 only.
    /// Per-task `lcm:` overrides. Default false.
    #[serde(default)]
    lcm: Option<bool>,

    /// v0.29 phase 2: motion LoRAs stacked on top of the AnimateDiff
    /// motion adapter. Same `LoraSpec` grammar as the CLI
    /// `--motion-lora` flag. Per-task `motion-lora:` adds on top.
    /// Ignored for non-animate tasks.
    #[serde(rename = "motion-lora", default)]
    motion_loras: Vec<String>,

    /// v0.29 phase 2: global multiplier on each motion-LoRA's
    /// per-spec scale. Default 1.0. Per-task `motion-lora-scale:`
    /// overrides.
    #[serde(rename = "motion-lora-scale", default)]
    motion_lora_scale: Option<f32>,

    /// v0.29 phase 2: animate output format. `frames | gif | mp4 |
    /// webm | all`. Per-task `format:` overrides. Default `frames`.
    /// MP4 / WebM require ffmpeg on `$PATH`.
    #[serde(rename = "format", default)]
    animate_format: Option<String>,

    /// v0.29 phase 2: GIF frame delay in ms (when `format` is `gif`
    /// or `all`). Default 100 (10 fps). Per-task `gif-delay-ms:`
    /// overrides.
    #[serde(rename = "gif-delay-ms", default)]
    animate_gif_delay_ms: Option<u16>,

    // ---------- MAP-4: scenario-level `map` task defaults (per-task overrides) ----------
    /// `--map-spec` path: a committed MapSpec to load (skips the LLM). Per-task wins.
    #[serde(rename = "map-spec", default)]
    map_spec: Option<String>,
    /// `parchment` | `inked` | `blueprint`.
    #[serde(rename = "map-style", default)]
    map_style: Option<String>,
    /// Paint the map with SD (img2img + Canny) instead of the deterministic linework.
    #[serde(rename = "map-paint", default)]
    map_paint: Option<bool>,
    /// `--map-scale` alias / `--map-tiles` `CxR`.
    #[serde(rename = "map-scale", default)]
    map_scale: Option<String>,
    #[serde(rename = "map-tiles", default)]
    map_tiles: Option<String>,
    /// SD backbone + LoRA(s) for the painted path.
    #[serde(rename = "map-sd-model", default)]
    map_sd_model: Option<String>,
    #[serde(rename = "map-sd-lora", default)]
    map_sd_lora: Vec<String>,
    /// LLM provider for the prose→spec parse.
    #[serde(rename = "map-provider", default)]
    map_provider: Option<String>,
    /// MAP-5 town street plan (`radial`/`grid`/`organic`) + MAP-2 natural-feature
    /// erosion (0 smooth … 1 natural … >1 rugged).
    #[serde(rename = "map-layout", default)]
    map_layout: Option<String>,
    #[serde(rename = "map-erosion", default)]
    map_erosion: Option<f32>,
    /// 1.14.0-B: emit the world as a grid of seamless tiles (over `map-tiles`/
    /// `map-scale`) instead of a single `map.png`. Mirrors `--map-render-tiles`.
    #[serde(rename = "map-render-tiles", default)]
    map_render_tiles: Option<bool>,
    /// 1.14.0-D: draw per-tile furniture (frame + grid coordinate + north arrow)
    /// on each tile. Mirrors `--map-tile-furniture`.
    #[serde(rename = "map-tile-furniture", default)]
    map_tile_furniture: Option<bool>,

    /// v0.15 phase 7a / v0.18: scenario-wide conditioning image. Three
    /// roles depending on `model:`:
    ///   * `flux-canny-dev` — canny edge map (channel-concat 128ch img_in)
    ///   * `flux-depth-dev` — depth map (channel-concat 128ch img_in)
    ///   * `flux-kontext-dev` — reference image to edit (seq-concat,
    ///                          img_ids[..., 0] = 1)
    /// Per-task `concept-image:` overrides. Ignored on non-conditioning
    /// models.
    #[serde(rename = "concept-image", default)]
    concept_image: Option<PathBuf>,

    /// v0.18 Kontext phase 4: opt-in aspect-bucket snap. When `true`
    /// AND `model:` is `flux-kontext-dev` / `flux-kontext-dev-gguf`,
    /// `size:` snaps to the closest of 17 BFL-recommended Kontext
    /// resolutions before VAE encoding. Per-task `kontext-bucket:`
    /// overrides.
    #[serde(rename = "kontext-bucket", default)]
    kontext_bucket: Option<bool>,

    // ---------- prompt-assembly fragments ----------
    #[serde(rename = "lora-header", default)]
    lora_header: String,
    #[serde(rename = "lora-footer", default)]
    lora_footer: String,
    #[serde(rename = "prompt-header", default)]
    prompt_header: String,
    #[serde(rename = "prompt-footer", default)]
    prompt_footer: String,

    // Accept both correct + the typo-spelling commonly seen in the wild.
    #[serde(alias = "enchancer")]
    enhancer: Option<String>,

    #[serde(default)]
    negative: String,

    // ---------- style detection / transfer ----------
    /// Detect art style from this photo and load the matching LoRAs
    /// from the catalog. Applied globally to every task. Conflicts
    /// with `loras: [...]` — catalog LoRAs win with a warning.
    #[serde(rename = "style-ref", default)]
    style_ref: Option<PathBuf>,
    /// Pick a style by id from the catalog. Bypasses detection when
    /// used alone; overrides the detection result when combined with
    /// `style-ref`.
    #[serde(default)]
    style: Option<String>,
    /// Multiplier applied to every catalog LoRA's :scale. Defaults to 1.0.
    #[serde(rename = "style-strength", default)]
    style_strength: Option<f32>,
    /// Override the bundled style catalog directory.
    #[serde(rename = "style-catalog", default)]
    style_catalog: Option<PathBuf>,

    // ---------- artefact compositing ----------
    /// Override the bundled artefact library directory. Per-task
    /// `artefacts: [...]` references resolve against this library.
    #[serde(rename = "artefact-library", default)]
    artefact_library: Option<PathBuf>,
    /// Override the default zone grid (normalized `[0, 1]` band
    /// extents). Missing bands fall back to the default 4×3 grid.
    #[serde(default)]
    zones: crate::artefacts::ZoneOverrides,
    /// v2: enable the masked img2img blending pass after alpha
    /// compositing. Set per-task via `artefact-blend: true`, or
    /// scenario-wide here. Default off (v1 alpha-only).
    #[serde(rename = "artefact-blend", default)]
    artefact_blend: bool,
    /// img2img strength for the v2 blend pass. Sweet spot: 0.25–0.4.
    #[serde(rename = "artefact-blend-strength", default)]
    artefact_blend_strength: Option<f32>,
    /// v3: derive zone extents from the generated image's own depth +
    /// luminance instead of the rigid 4×3 grid (with per-band
    /// `zones:` overrides applied where smart resolution comes up
    /// empty). Per-task `smart-zones: true/false` overrides. Default
    /// off.
    #[serde(rename = "smart-zones", default)]
    smart_zones: bool,

    // ---------- post-generate options ----------
    #[serde(default)]
    upscale: UpscaleConfig,

    // ---------- catalogs ----------
    #[serde(default)]
    scene: Vec<NamedPrompt>,
    #[serde(default)]
    weather: Vec<NamedPrompt>,
    /// Named identities tasks can pull in via their own `personas: [name]`
    /// list. Each persona is a reference photo + per-persona portrait
    /// parameters; the task supplies scene/weather/prompt/size/sampler.
    #[serde(default)]
    personas: Vec<PersonaDef>,
    #[serde(default)]
    tasks: Vec<TaskDef>,
}

/// Top-level `personas: [ {...}, ... ]` entry. Identity-defining settings only
/// — task-side concerns (scene, prompt, size) live on `TaskDef`.
/// HJSON shape for one entry in `PersonaDef.photos`. Untagged so HJSON
/// authors can use either `"path:weight"` shorthand strings (matching the
/// CLI `--photo` grammar) or `{ path: "...", weight: 0.7 }` objects.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum PersonaPhoto {
    Shorthand(String),
    Full {
        path: PathBuf,
        #[serde(default)]
        weight: Option<f32>,
    },
}

impl PersonaPhoto {
    fn to_weighted(&self) -> Result<crate::pipelines::ip_adapter::WeightedPhoto> {
        match self {
            Self::Shorthand(s) => s.parse::<crate::pipelines::ip_adapter::WeightedPhoto>(),
            Self::Full { path, weight } => Ok(crate::pipelines::ip_adapter::WeightedPhoto {
                path: path.clone(),
                weight: *weight,
            }),
        }
    }
}

#[derive(Debug, Deserialize)]
struct PersonaDef {
    /// Referenced by `task.personas: [<name>]`. Must be unique within the file.
    name: String,
    /// Single reference photo. Mutually exclusive with `photos`.
    #[serde(default)]
    photo: Option<PathBuf>,
    /// Multiple weighted reference photos for embedding-space merging
    /// (averaging facial features across photos). Mutually exclusive
    /// with `photo`. Each entry accepts either a shorthand
    /// `"path:weight"` string or a `{ path: "...", weight: 0.7 }`
    /// object; weights are normalized to sum to 1.0. Same grammar as
    /// the CLI's repeatable `--photo` flag.
    #[serde(default)]
    photos: Vec<PersonaPhoto>,
    /// Which identity strategy. Defaults to `plus-face`. Other options:
    /// `plus-face-sdxl`, `faceid`, `faceid-sdxl`.
    #[serde(default)]
    identity: Option<String>,
    /// IP-Adapter scale on the image tokens (0..). Defaults to 0.8.
    #[serde(rename = "face-strength", default)]
    face_strength: Option<f32>,
    /// Optional face bbox in the photo, `[x0, y0, x1, y1]` normalised to
    /// `[0, 1]` with origin top-left. Used by FaceID strategies to crop
    /// the photo to the face region before ArcFace embedding; CLIP-H
    /// strategies ignore it. Optional SCRFD auto-detection (PLAKAT_SCRFD_*)
    /// can fill this in from any photo.
    #[serde(rename = "face-bbox", default)]
    face_bbox: Option<[f32; 4]>,
    /// Optional 5-point landmarks in the photo. When set,
    /// FaceID strategies perform a similarity-transform alignment to
    /// ArcFace's canonical 112×112 template — the proper alignment.
    /// Takes precedence over `face-bbox` when both are set.
    ///
    /// Format: `[[x, y]; 5]` normalised, order:
    /// `left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner`.
    /// CLIP-H strategies (`plus-face*`) ignore this field.
    #[serde(rename = "face-landmarks", default)]
    face_landmarks: Option<[[f32; 2]; 5]>,
    /// Optional persona-specific negative prompt (e.g. "no glasses, no beard").
    /// Prepended to the task's effective negative when this persona is
    /// imposed — kept with the persona because it describes the *who*, not
    /// the scene.
    #[serde(default)]
    negative: Option<String>,
}

impl PersonaDef {
    /// Resolve the persona's `photo` (legacy single-ref) and `photos`
    /// (multi-ref) fields into a normalized list of weighted photos
    /// suitable for the portrait pipeline. Validates mutual-exclusion
    /// and that at least one is set.
    fn resolve_photos(&self) -> Result<Vec<crate::pipelines::ip_adapter::WeightedPhoto>> {
        let mut out: Vec<crate::pipelines::ip_adapter::WeightedPhoto> =
            match (&self.photo, self.photos.is_empty()) {
                (Some(_), false) => bail!(
                    "persona {:?}: `photo` and `photos` are mutually exclusive — \
                     use one or the other",
                    self.name
                ),
                (None, true) => bail!(
                    "persona {:?}: missing reference photo (set `photo: <path>` or \
                     `photos: [...]`)",
                    self.name
                ),
                (Some(p), true) => vec![
                    crate::pipelines::ip_adapter::WeightedPhoto::single(p.clone()),
                ],
                (None, false) => self
                    .photos
                    .iter()
                    .map(PersonaPhoto::to_weighted)
                    .collect::<Result<_>>()
                    .with_context(|| format!("parsing persona {:?} photos", self.name))?,
            };
        crate::pipelines::ip_adapter::normalize_photo_weights(&mut out)
            .with_context(|| format!("normalizing persona {:?} photos", self.name))?;
        Ok(out)
    }

    /// Convenience: the first photo path, for log messages / existence
    /// checks. With multi-photo personas we still need to point users
    /// at a representative location; the first one is fine for that.
    fn primary_photo_path(&self) -> Option<PathBuf> {
        if let Some(p) = &self.photo {
            return Some(p.clone());
        }
        self.photos.first().and_then(|pp| match pp {
            PersonaPhoto::Shorthand(s) => s
                .parse::<crate::pipelines::ip_adapter::WeightedPhoto>()
                .ok()
                .map(|w| w.path),
            PersonaPhoto::Full { path, .. } => Some(path.clone()),
        })
    }
}

/// Top-level `upscale: { ... }` section.
#[derive(Debug, Deserialize)]
struct UpscaleConfig {
    /// Enable the post-generate upscale pass.
    #[serde(default, alias = "enabled")]
    upscale: bool,
    /// Scale factor (2× by default).
    #[serde(default = "default_upscale_scale")]
    scale: f32,
    /// Filter: nearest | bilinear | bicubic | lanczos.
    #[serde(default = "default_upscale_method")]
    method: String,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            upscale: false,
            scale: default_upscale_scale(),
            method: default_upscale_method(),
        }
    }
}

fn default_upscale_scale() -> f32 {
    2.0
}
fn default_upscale_method() -> String {
    "lanczos".to_string()
}

#[derive(Debug, Deserialize)]
struct NamedPrompt {
    name: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct TaskDef {
    name: String,
    // `scene`/`weather` style a generate task; `prompt` is its text (or, for a
    // `type: map` task, the world description). All default-empty so a map task —
    // which sources from `map-spec` or `prompt` — needn't carry scene/weather.
    #[serde(default)]
    scene: String,
    #[serde(default)]
    weather: String,
    #[serde(default)]
    prompt: String,

    /// Regional prompting: per-region prompts `"x0,y0,x1,y1:prompt"` (canvas
    /// fractions in `[0,1]`). Each region steers its box, blended over the
    /// task `prompt` (MultiDiffusion). SD 1.5 / SDXL / SD3.5. Empty = off.
    #[serde(default)]
    regions: Vec<String>,

    // ---------- per-task style pass ----------
    /// Optional path to a style reference image. If set, every generated
    /// image for this task is also run through `stylize` (IP-Adapter) using
    /// this image as REF. Original + styled both land in the task directory.
    #[serde(default)]
    style: Option<PathBuf>,
    /// IP-Adapter strength for the style pass (0..1). Higher = more REF.
    #[serde(rename = "style-strength", default)]
    style_strength: Option<f32>,

    // ---------- per-task overrides for global fields ----------
    // When set, override the scenario's global value for THIS task only.
    // Fields not listed here (model, device, loras, lora-scale, enhancer,
    // out, upscale.*) stay global because changing them would force the
    // shared pipeline to reload.
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    aspect: Option<String>,
    #[serde(default)]
    count: Option<u32>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    guidance: Option<f64>,
    /// Per-task seed override. When set, this task uses exactly this seed
    /// (the global seed_offset counter still advances so later tasks are
    /// unaffected).
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    negative: Option<String>,
    #[serde(default)]
    scheduler: Option<String>,
    #[serde(default)]
    refine: Option<usize>,
    #[serde(rename = "refine-strength", default)]
    refine_strength: Option<f32>,
    #[serde(rename = "refiner-frac", default)]
    refiner_frac: Option<f32>,

    /// Personas (from the top-level `personas` list) to impose into this
    /// task's output. Two accepted forms:
    ///
    /// Single persona, whole image:
    ///     personas: [ alice ]
    ///
    /// Multi-persona, region-masked compositing:
    ///     personas: [
    ///         { name: alice, bbox: [0.0, 0.1, 0.45, 0.9] }
    ///         { name: bob,   bbox: [0.55, 0.1, 1.0, 0.9] }
    ///     ]
    ///
    /// `bbox` is `[x0, y0, x1, y1]` normalised to `[0, 1]`. Mixing forms
    /// within one task is rejected at load time.
    #[serde(default)]
    personas: Option<Vec<PersonaRef>>,

    // ---------- per-task catalog-style override ----------
    /// Detect art style from this photo. Fully overrides the scenario's
    /// global `style-ref` / `style` for this task only; the catalog
    /// trigger phrase and `negative_extras` are recomputed against the
    /// scenario's bare `lora-header` and `negative` (NOT against the
    /// global style's effective values).
    ///
    /// Per-task catalog style applies **trigger and negative only** —
    /// LoRAs cannot be swapped per-task because scenarios share a
    /// pre-loaded pipeline. If the resolved style would have required
    /// different LoRAs than the scenario's global set, plakat warns and
    /// proceeds (trigger + negative may still help, but the LoRA's
    /// visual contribution is missing). For per-task LoRA swaps,
    /// split into separate scenarios.
    ///
    /// **Note on naming:** distinct from per-task `style:` (IP-Adapter
    /// REF stylize pass) — different feature.
    #[serde(rename = "style-ref", default)]
    style_ref_catalog: Option<PathBuf>,

    /// Composite named artefacts (PNG cutouts from the library) onto
    /// this task's output. Accepts either CLI-grammar shorthand
    /// strings (`"oak@middle_plan/left"`) or full objects with
    /// per-artefact `offset`, `anchor`, `flip`, `alpha` overrides.
    /// Compositing happens BEFORE any per-task stylize / upscale pass,
    /// so the IP-Adapter stylize re-paints over the composited
    /// artefacts (often a feature: it unifies the palette).
    #[serde(default)]
    artefacts: Vec<crate::artefacts::ArtefactSpecEntry>,

    /// v2: per-task override for the masked img2img blend pass.
    /// `None` inherits the scenario-level `artefact-blend:` field.
    #[serde(rename = "artefact-blend", default)]
    artefact_blend: Option<bool>,

    /// v2: per-task override for blend strength. `None` inherits the
    /// scenario-level `artefact-blend-strength:` (default 0.3).
    #[serde(rename = "artefact-blend-strength", default)]
    artefact_blend_strength: Option<f32>,

    /// v3: per-task override for smart zones. `None` inherits the
    /// scenario-level `smart-zones:` flag.
    #[serde(rename = "smart-zones", default)]
    smart_zones: Option<bool>,

    /// v0.9: ControlNet block. When set, the named conditioner
    /// guides every denoise step for this task. `strength` defaults
    /// to 1.0 if omitted.
    ///
    /// ```hjson
    /// control: { kind: depth, image: ./hint.png, strength: 0.85 }
    /// ```
    /// Mutually exclusive with `controls:` (the Vec form below).
    #[serde(default)]
    control: Option<ControlSpec>,

    /// v0.13 phase 11: multi-ControlNet per task. Each entry is a
    /// `ControlSpec` (same grammar as the singular `control:` block).
    /// Residuals from all entries sum per denoise step.
    ///
    /// ```hjson
    /// controls: [
    ///   { kind: depth, image: ./d.png, strength: 0.8 },
    ///   { kind: canny, auto-from: ./ref.png, strength: 0.5, end: 0.5 },
    /// ]
    /// ```
    /// Mutually exclusive with the singular `control:` block. Both
    /// SD and Flux multi-CN flows compose with the scenario
    /// `tiled:` config (per phase 9).
    #[serde(default)]
    controls: Option<Vec<ControlSpec>>,

    // ---------- v0.13 phase 10/11: img2img / Fill / inpaint inputs ----
    /// v0.13 phase 3 / phase 11: init image. For Flux non-Fill →
    /// img2img (uses `strength`). For Flux Fill or SD inpaint models →
    /// inpaint with `mask`. For plain SD t2i models → RePaint-style
    /// masked img2img if `mask:` is also set, else plain img2img.
    /// Empty by default.
    #[serde(rename = "init-image", default)]
    init_image: Option<PathBuf>,
    /// v0.13 phase 2 / phase 11: inpaint mask (white = inpaint, black
    /// = preserve). Requires `init-image`. Pure t2i tasks ignore.
    #[serde(default)]
    mask: Option<PathBuf>,
    /// v0.13 phase 3: img2img strength in `[0, 1]`. `None` → pipeline
    /// default (Flux 0.85; SD 0.6 for img2img, 1.0 for inpaint).
    #[serde(default)]
    strength: Option<f32>,
    /// v0.13 phase 11: SD inpaint mask feathering (px). Softens the
    /// boundary between preserved and inpainted regions. Default 8.
    /// Ignored on Flux (mask is binarised before the patching pack).
    #[serde(rename = "mask-feather", default)]
    mask_feather: Option<u32>,
    /// v0.13 phase 11: invert mask polarity (treat black as inpaint).
    #[serde(rename = "mask-invert", default)]
    mask_invert: Option<bool>,

    /// v0.13 phase 11: outpaint task — extend the canvas of
    /// `init-image` past its borders, build a mask covering the new
    /// region, and run the inpaint pipeline. Mutually exclusive with
    /// `mask:` (the outpaint block synthesises the mask).
    ///
    /// ```hjson
    /// outpaint: { expand: 256 }                         # all sides
    /// outpaint: { left: 512, right: 512 }               # panoramic
    /// outpaint: { top: 128, bottom: 256 }               # vertical
    /// ```
    #[serde(default)]
    outpaint: Option<OutpaintSpec>,

    /// v0.14 phase 3c: zero or more Flux Redux reference images for
    /// this task. Each string parses as `path` (weight = 1.0) or
    /// `path:weight=0.7`. Only meaningful for Flux variants except
    /// Fill (which has an incompatible `img_in` shape).
    ///
    /// ```hjson
    /// redux-images: [ ./refs/cat1.jpg, ./refs/cat2.jpg:weight=0.6 ]
    /// ```
    ///
    /// Cap of 4 enforced inside `Pipeline::generate`.
    #[serde(rename = "redux-images", default)]
    redux_images: Vec<String>,

    /// v0.15 phase 7a: per-task Flux distillation preset (`hyper-8`,
    /// `hyper-16`, `turbo-alpha`). Overrides the scenario-level `fast:`.
    /// Applies the preset's LoRA + step/guidance defaults the same way
    /// `plakat generate --fast PRESET` does. Bails if combined with a
    /// non-Flux task model.
    #[serde(default)]
    fast: Option<String>,

    /// v0.25: per-task art-medium preset. Overrides scenario-level
    /// `look:`. Same accepted names + semantics as the CLI `--look`
    /// flag.
    #[serde(default)]
    look: Option<String>,

    /// v0.25: per-task subject-domain preset (`anime`). Overrides
    /// scenario-level `genre:`.
    #[serde(default)]
    genre: Option<String>,

    /// v0.25: per-task `--offline` override. `None` inherits the
    /// scenario-level setting.
    #[serde(default)]
    offline: Option<bool>,

    /// v0.15 phase 7a / v0.18: per-task conditioning image. Required
    /// on Flux concept variants (Canny-dev / Depth-dev) and Kontext
    /// (Kontext-dev). Falls back to scenario-level `concept-image:`.
    #[serde(rename = "concept-image", default)]
    concept_image: Option<PathBuf>,

    /// v0.18 Kontext phase 4: per-task Kontext bucket override. When
    /// `Some(true)`, snap `size:` to the closest of 17 BFL Kontext
    /// resolutions before VAE encoding (Kontext models only). Falls
    /// back to the scenario-level `kontext-bucket:` when `None`.
    #[serde(rename = "kontext-bucket", default)]
    kontext_bucket: Option<bool>,

    /// v0.15 phase 7a: per-task prompt enhancement override. Three
    /// forms:
    ///   * absent — inherit scenario-level `enhancer:`
    ///   * `enhance: "deepseek"` — use this provider for this task
    ///   * `enhance: false` — opt out of enhancement entirely for this
    ///     task even when the scenario has one configured
    #[serde(default)]
    enhance: Option<EnhanceCfg>,

    /// v0.15 phase 7a: per-task tiled-denoise override.
    ///   * absent — inherit scenario-level `tiled:`
    ///   * `tiled: false` — force off (e.g. a small portrait task in
    ///     a mostly-4K scenario)
    ///   * `tiled: { size: 1024, stride: 768 }` — override config
    #[serde(default)]
    tiled: Option<TaskTiledCfg>,

    /// v0.15 phase 7b-7: per-task LoRA stack (ADDITIVE on top of
    /// the scenario-level LoRAs). Each string parses via the same
    /// `LoraSpec` grammar the CLI `--lora` flag uses: local path,
    /// HF repo, or `:weight` suffix.
    ///
    /// Supported on Flux (BF16 / GGUF / NF4) and SD3 / SD3.5.
    /// SD-family tasks (`sd15`, `sd21`, `sdxl`, `sdxl-turbo`) bail
    /// loud at dispatch — SD UNet runtime LoRA infrastructure is
    /// deferred (v0.15 phase 7b-6 skeleton).
    #[serde(default)]
    loras: Vec<String>,

    /// v0.15 phase 7b-7: per-task LoRA scale multiplier applied on
    /// top of each LoRA's own `:weight` suffix. `None` defaults to
    /// 1.0. Mirrors `--lora-scale` on the CLI.
    #[serde(rename = "lora-scale", default)]
    lora_scale: Option<f32>,

    // ---------- v0.29 phase 2: per-task animate overrides ----------

    /// v0.29 phase 2: per-task dispatch type override. Same accepted
    /// values as the scenario-level field: `"generate"` (default,
    /// inherits scenario) or `"animatediff"` (also `"animate"`).
    #[serde(default, rename = "type")]
    task_type: Option<String>,

    /// v0.29 phase 2: per-task animate total frames. Overrides
    /// scenario-level `frames:`. No effect on non-animate tasks.
    #[serde(rename = "frames", default)]
    animate_frames: Option<u32>,

    /// v0.29 phase 2: per-task sliding-window size. Overrides
    /// scenario-level `window-size:`.
    #[serde(rename = "window-size", default)]
    animate_window_size: Option<u32>,

    /// v0.29 phase 2: per-task sliding-window overlap. Overrides
    /// scenario-level `window-overlap:`.
    #[serde(rename = "window-overlap", default)]
    animate_window_overlap: Option<u32>,

    /// v0.29 phase 2: per-task AnimateLCM toggle. Overrides
    /// scenario-level `lcm:`. SD 1.5 only.
    #[serde(default)]
    lcm: Option<bool>,

    /// v0.29 phase 2: motion LoRAs ADDED on top of the scenario-
    /// level `motion-lora:` list. Same `LoraSpec` grammar.
    #[serde(rename = "motion-lora", default)]
    motion_loras: Vec<String>,

    /// v0.29 phase 2: per-task motion-LoRA scale multiplier. Overrides
    /// scenario-level `motion-lora-scale:`.
    #[serde(rename = "motion-lora-scale", default)]
    motion_lora_scale: Option<f32>,

    /// v0.29 phase 2: per-task animate output format. Overrides
    /// scenario-level `format:`. Values: `frames | gif | mp4 | webm
    /// | all`.
    #[serde(rename = "format", default)]
    animate_format: Option<String>,

    /// v0.29 phase 2: per-task GIF frame delay in ms. Overrides
    /// scenario-level `gif-delay-ms:`.
    #[serde(rename = "gif-delay-ms", default)]
    animate_gif_delay_ms: Option<u16>,

    // ---------- MAP-4: per-task `map` overrides (the task `prompt` is the description) ----------
    #[serde(rename = "map-spec", default)]
    map_spec: Option<String>,
    #[serde(rename = "map-style", default)]
    map_style: Option<String>,
    #[serde(rename = "map-paint", default)]
    map_paint: Option<bool>,
    #[serde(rename = "map-scale", default)]
    map_scale: Option<String>,
    #[serde(rename = "map-tiles", default)]
    map_tiles: Option<String>,
    #[serde(rename = "map-sd-model", default)]
    map_sd_model: Option<String>,
    #[serde(rename = "map-sd-lora", default)]
    map_sd_lora: Vec<String>,
    #[serde(rename = "map-provider", default)]
    map_provider: Option<String>,
    #[serde(rename = "map-layout", default)]
    map_layout: Option<String>,
    #[serde(rename = "map-erosion", default)]
    map_erosion: Option<f32>,
    #[serde(rename = "map-render-tiles", default)]
    map_render_tiles: Option<bool>,
    #[serde(rename = "map-tile-furniture", default)]
    map_tile_furniture: Option<bool>,

    // ---------- 1.14.0-A: per-task `multiperson` block (a `type: multiperson` task) ----------
    /// The multiperson task body: scene prompt + placed people + identity mode.
    /// Only consulted for `type: multiperson` tasks. `people[].persona` names
    /// refer to the top-level `personas` list (resolved to photos at run time).
    #[serde(default)]
    multiperson: Option<crate::pipelines::multiperson::scenario_task::MultipersonTaskSpec>,

    /// The fractal task body (only consulted for `type: fractal` tasks).
    #[cfg(feature = "fractals")]
    #[serde(default)]
    fractal: Option<crate::fractals::scenario_task::FractalTaskCfg>,

    /// The bookart task body (only consulted for `type: bookart` tasks — 6.1.0 A2).
    #[serde(default)]
    bookart: Option<crate::bookart::scenario_task::BookartTaskCfg>,

    /// The texture task body (only consulted for `type: texture` tasks — 6.3.0 B7).
    #[serde(default)]
    texture: Option<crate::texture::scenario_task::TextureTaskCfg>,

    /// The comic task body (only consulted for `type: comic` tasks — 6.8.0 P4).
    #[serde(default)]
    comic: Option<crate::comic::scenario_task::ComicTaskCfg>,
}

/// v0.15 phase 7a: per-task enhancement override. Accepts a string
/// provider name (`"deepseek"`) or `false` to opt out.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum EnhanceCfg {
    Provider(String),
    Toggle(bool),
}

/// v0.29 phase 2: an animate task's effective config after
/// scenario-level defaults merge with per-task overrides. Computed
/// once per task by [`effective_animate_config`] and threaded into
/// the dispatch (phase 3).
#[derive(Debug, Clone)]
#[allow(dead_code)] // fields consumed by phase 3 dispatch
struct EffectiveAnimateCfg {
    pub frames: u32,
    pub window_size: u32,
    pub window_overlap: u32,
    pub lcm: bool,
    /// LoRA spec strings, scenario list + task list concatenated.
    pub motion_loras: Vec<String>,
    pub motion_lora_scale: f32,
    pub format: crate::imaging::video::Format,
    pub gif_delay_ms: u16,
}

impl EffectiveAnimateCfg {
    /// Validate frame/window/overlap bounds. Mirrors the CLI animate
    /// gate so users see the same diagnostics whether they're driving
    /// from `plakat animate` or `plakat scenario`.
    fn validate(&self, task_name: &str) -> Result<()> {
        const MAX_SEQ: u32 = 32;
        anyhow::ensure!(
            self.frames >= 1,
            "scenario task {task_name:?}: animate frames must be ≥ 1"
        );
        anyhow::ensure!(
            self.window_size >= 1 && self.window_size <= MAX_SEQ,
            "scenario task {task_name:?}: animate window-size {} \
             must be in 1..={MAX_SEQ} (motion_max_seq_length)",
            self.window_size,
        );
        anyhow::ensure!(
            self.window_overlap < self.window_size,
            "scenario task {task_name:?}: animate window-overlap {} \
             must be < window-size {}",
            self.window_overlap,
            self.window_size,
        );
        Ok(())
    }
}

/// v0.29 phase 2: parse a task-type string into a stable enum.
/// Accepts `"generate"` (or omitted) for the existing pipeline path
/// and `"animatediff"` / `"animate"` for the v0.29 animate dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
    Generate,
    Animate,
    Map,
    Multiperson,
    #[cfg(feature = "fractals")]
    Fractal,
    Bookart,
    Texture,
    Comic,
}

impl TaskKind {
    fn from_strs(
        task_level: Option<&str>,
        scenario_level: Option<&str>,
    ) -> Result<Self> {
        let raw = task_level.or(scenario_level).unwrap_or("generate");
        match raw.to_ascii_lowercase().as_str() {
            "generate" | "gen" | "t2i" => Ok(Self::Generate),
            "animatediff" | "animate" => Ok(Self::Animate),
            "map" => Ok(Self::Map),
            "multiperson" | "multi-person" => Ok(Self::Multiperson),
            #[cfg(feature = "fractals")]
            "fractal" | "fractals" => Ok(Self::Fractal),
            "bookart" => Ok(Self::Bookart),
            "texture" => Ok(Self::Texture),
            "comic" => Ok(Self::Comic),
            other => bail!(
                "scenario task type {other:?} not recognised \
                 (expected: generate, animatediff, map, multiperson, fractal, bookart, texture, comic)"
            ),
        }
    }
}

/// v0.31 phase 3: which cached pipeline (if any) the scenario loop
/// should drop before running the next task. Used by the kind-
/// switching evictor to close the v0.29 mixed-kind carry: scenarios
/// that mix `type: generate` and `type: animatediff` used to hold
/// both pipelines simultaneously (~10 GB SD 1.5 / worse on SDXL).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheEviction {
    /// No-op — same kind as last task, or this is the first task.
    None,
    /// Just switched FROM generate TO animate — drop the SD t2i
    /// `pipeline` carrier so the animate load can fit.
    DropT2i,
    /// Just switched FROM animate TO generate — drop the
    /// `animate_sd15` / `animate_sdxl` carriers.
    DropAnimate,
    /// Switching TO a `map` task — its painted render loads its own SD pipeline
    /// internally, so free whatever t2i / animate pipeline was cached first.
    DropAll,
}

/// v0.32 phase 2: pure VAE cache decision. Returns `Some(value.clone())`
/// when the cache holds a matching key, `None` otherwise. Generic
/// over the cached value type so the decision logic is unit-testable
/// without constructing a real `AutoEncoderKL`.
///
/// The real call site uses `T = Arc<AutoEncoderKL>` and the `clone()`
/// produces another Arc-handle to the same VAE — no tensor copies.
fn vae_cache_lookup<T: Clone>(cache: Option<&(String, T)>, model: &str) -> Option<T> {
    cache.filter(|(k, _)| k == model).map(|(_, v)| v.clone())
}

/// Eviction decision for one (last, current) kind pair. Pure
/// function — unit-testable without spinning up real pipelines.
fn evict_decision(last: Option<TaskKind>, current: TaskKind) -> CacheEviction {
    match (last, current) {
        (Some(TaskKind::Generate), TaskKind::Animate) => CacheEviction::DropT2i,
        (Some(TaskKind::Animate), TaskKind::Generate) => CacheEviction::DropAnimate,
        // A map / multiperson / fractal task loads its own SD pipeline(s) internally (or
        // none, for a pure Track-A fractal) — free any cached t2i / animate pipeline first.
        #[cfg(feature = "fractals")]
        (Some(TaskKind::Generate) | Some(TaskKind::Animate), TaskKind::Map | TaskKind::Multiperson | TaskKind::Fractal | TaskKind::Bookart | TaskKind::Texture | TaskKind::Comic) => {
            CacheEviction::DropAll
        }
        #[cfg(not(feature = "fractals"))]
        (Some(TaskKind::Generate) | Some(TaskKind::Animate), TaskKind::Map | TaskKind::Multiperson | TaskKind::Bookart | TaskKind::Texture | TaskKind::Comic) => {
            CacheEviction::DropAll
        }
        // First task (last == None) or same-kind continuation —
        // no eviction needed.
        _ => CacheEviction::None,
    }
}

/// v0.29 phase 2: compute the effective animate config for one task
/// by merging scenario-level defaults with per-task overrides.
/// Scenario `motion-lora` list is the BASE; task `motion-lora` list
/// is APPENDED (same pattern as `loras:`).
/// MAP-4: merge scenario-level + per-task `map` fields into a `MapTaskCfg`. The
/// task `prompt` is the world description; per-task fields override scenario ones.
fn effective_map_config(scenario: &ScenarioFile, task: &TaskDef) -> crate::map::scenario_task::MapTaskCfg {
    let pick = |t: Option<&String>, s: Option<&String>, d: &str| t.or(s).cloned().unwrap_or_else(|| d.to_string());
    let spec = task.map_spec.clone().or_else(|| scenario.map_spec.clone());
    // Per-task LoRAs replace scenario ones when present (mirrors --map-sd-lora).
    let sd_loras = if !task.map_sd_lora.is_empty() {
        task.map_sd_lora.clone()
    } else {
        scenario.map_sd_lora.clone()
    };
    crate::map::scenario_task::MapTaskCfg {
        description: task.prompt.clone(),
        spec_path: spec.map(std::path::PathBuf::from),
        style: pick(task.map_style.as_ref(), scenario.map_style.as_ref(), "parchment"),
        paint: task.map_paint.or(scenario.map_paint).unwrap_or(false),
        provider: pick(task.map_provider.as_ref(), scenario.map_provider.as_ref(), "auto"),
        scale: task.map_scale.clone().or_else(|| scenario.map_scale.clone()),
        tiles: task.map_tiles.clone().or_else(|| scenario.map_tiles.clone()),
        sd_model: pick(task.map_sd_model.as_ref(), scenario.map_sd_model.as_ref(), "sdxl"),
        sd_loras,
        urban_layout: task.map_layout.clone().or_else(|| scenario.map_layout.clone()),
        erosion: task.map_erosion.or(scenario.map_erosion),
        render_tiles: task.map_render_tiles.or(scenario.map_render_tiles).unwrap_or(false),
        render_tile_furniture: task.map_tile_furniture.or(scenario.map_tile_furniture).unwrap_or(false),
        cache: false,
    }
}

fn effective_animate_config(
    scenario: &ScenarioFile,
    task: &TaskDef,
) -> Result<EffectiveAnimateCfg> {
    use std::str::FromStr;

    let frames = task
        .animate_frames
        .or(scenario.animate_frames)
        .unwrap_or(16);
    let window_size = task
        .animate_window_size
        .or(scenario.animate_window_size)
        .unwrap_or(16);
    let window_overlap = task
        .animate_window_overlap
        .or(scenario.animate_window_overlap)
        .unwrap_or(4);
    let lcm = task.lcm.or(scenario.lcm).unwrap_or(false);

    // Motion LoRAs: scenario base + task add-on.
    let mut motion_loras = scenario.motion_loras.clone();
    motion_loras.extend(task.motion_loras.iter().cloned());

    let motion_lora_scale = task
        .motion_lora_scale
        .or(scenario.motion_lora_scale)
        .unwrap_or(1.0);

    let format_str = task
        .animate_format
        .as_deref()
        .or(scenario.animate_format.as_deref())
        .unwrap_or("frames");
    let format = crate::imaging::video::Format::from_str(format_str)
        .with_context(|| {
            format!(
                "scenario task {:?}: animate format {format_str:?} not recognised",
                task.name,
            )
        })?;

    let gif_delay_ms = task
        .animate_gif_delay_ms
        .or(scenario.animate_gif_delay_ms)
        .unwrap_or(100);

    Ok(EffectiveAnimateCfg {
        frames,
        window_size,
        window_overlap,
        lcm,
        motion_loras,
        motion_lora_scale,
        format,
        gif_delay_ms,
    })
}

/// v0.15 phase 7a: per-task tiled override. Accepts a full config
/// block or `false` to force off.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(untagged)]
enum TaskTiledCfg {
    Toggle(bool),
    Override(TiledCfg),
}

/// v0.13 phase 11: outpaint task block. At least one of the four
/// per-side fields (or `expand`) must be > 0. `expand` is shorthand
/// for setting all four sides equally and conflicts with the per-
/// side fields.
#[derive(Debug, Clone, Copy, Deserialize)]
struct OutpaintSpec {
    #[serde(default)]
    left: u32,
    #[serde(default)]
    right: u32,
    #[serde(default)]
    top: u32,
    #[serde(default)]
    bottom: u32,
    /// Shorthand: extend all four sides by this many pixels. When
    /// set, the per-side fields must all be 0.
    #[serde(default)]
    expand: Option<u32>,
}

/// v0.13 phase 11: helper trait — collapse `control:` (singular) and
/// `controls:` (Vec) into a single Vec<&ControlSpec>. Validates
/// mutual exclusion. Returned references live as long as the
/// borrowed task.
fn task_effective_controls(task: &TaskDef) -> Result<Vec<&ControlSpec>> {
    match (task.control.as_ref(), task.controls.as_ref()) {
        (Some(_), Some(_)) => bail!(
            "task {:?}: `control:` (singular) and `controls:` (Vec) are mutually exclusive — pick one form",
            task.name
        ),
        (Some(c), None) => Ok(vec![c]),
        (None, Some(v)) => {
            if v.is_empty() {
                bail!(
                    "task {:?}: `controls: []` is empty — drop the field or supply at least one entry",
                    task.name
                );
            }
            Ok(v.iter().collect())
        }
        (None, None) => Ok(Vec::new()),
    }
}

/// v0.13 phase 11: validate the inpaint / outpaint input combination
/// for a task. Returns the effective `(init_image, mask)` pair after
/// honouring `outpaint:` (which synthesises the mask itself). The
/// mask path returned for outpaint is `None` — the outpaint dispatch
/// generates it from the canvas + padding at run time.
fn task_validate_image_inputs(task: &TaskDef) -> Result<()> {
    if let Some(ospec) = task.outpaint.as_ref() {
        if task.init_image.is_none() {
            bail!(
                "task {:?}: `outpaint:` requires `init-image:` (the canvas to extend)",
                task.name
            );
        }
        if task.mask.is_some() {
            bail!(
                "task {:?}: `outpaint:` synthesises the mask — `mask:` must be unset",
                task.name
            );
        }
        // Touch ospec so the borrow stays around for the bail() above.
        let _ = ospec;
    }
    if task.mask.is_some() && task.init_image.is_none() {
        bail!(
            "task {:?}: `mask:` requires `init-image:` (mask without init-image is meaningless)",
            task.name
        );
    }
    Ok(())
}

impl OutpaintSpec {
    /// Resolve the four per-side amounts after applying `expand` and
    /// validate that at least one side is > 0.
    fn resolved(self) -> Result<(u32, u32, u32, u32)> {
        let (l, r, t, b) = match self.expand {
            Some(n) => {
                if self.left != 0 || self.right != 0 || self.top != 0 || self.bottom != 0 {
                    bail!(
                        "outpaint: `expand:` conflicts with per-side amounts; pick one form"
                    );
                }
                (n, n, n, n)
            }
            None => (self.left, self.right, self.top, self.bottom),
        };
        if l == 0 && r == 0 && t == 0 && b == 0 {
            bail!(
                "outpaint: need at least one of left/right/top/bottom (or expand) > 0"
            );
        }
        Ok((l, r, t, b))
    }
}

/// v0.13 phase 10: scenario-level tiled-denoise config. `size` and
/// `stride` are in **pixels**; the pipelines internally divide by the
/// VAE downsample (8) to get latent units. Same defaults as the CLI
/// flags (1024 / 768) when fields are omitted.
#[derive(Debug, Clone, Copy, Deserialize)]
struct TiledCfg {
    #[serde(default = "default_tile_size")]
    size: u32,
    #[serde(default = "default_tile_stride")]
    stride: u32,
}

fn default_tile_size() -> u32 {
    1024
}
fn default_tile_stride() -> u32 {
    768
}

impl From<TiledCfg> for crate::pipelines::tiled::TiledConfig {
    fn from(c: TiledCfg) -> Self {
        Self {
            tile_size: c.size,
            stride: c.stride,
        }
    }
}

/// Per-task ControlNet configuration. `kind` is a string parsed via
/// the same `FromStr` impl as the CLI `--control`. Conditioning
/// source is either `image` (pre-rendered map) or `auto-from`
/// (image to auto-annotate via the matching annotator). Exactly
/// one must be set. `strength` defaults to 1.0 when omitted.
#[derive(Debug, Clone, Deserialize)]
struct ControlSpec {
    kind: String,
    /// Pre-rendered conditioning image. Mutually exclusive with `auto-from`.
    #[serde(default)]
    image: Option<PathBuf>,
    /// **v0.10**: source image to auto-annotate. Mutually exclusive with `image`.
    #[serde(default, rename = "auto-from")]
    auto_from: Option<PathBuf>,
    /// **v0.30 phase 2**: input video — per-frame extract + annotate.
    /// Only valid for animate-kind tasks. Mutually exclusive with
    /// `image` and `auto-from`.
    #[serde(default)]
    video: Option<PathBuf>,
    #[serde(default)]
    strength: Option<f32>,
    /// Timestep window. `start` defaults to 0.0 when omitted; `end`
    /// defaults to 1.0. Set `end: 0.5` to lock composition early
    /// then release the prompt to drive late texture / atmosphere
    /// refinement.
    #[serde(default)]
    start: Option<f32>,
    #[serde(default)]
    end: Option<f32>,
}

/// One persona reference inside a task. Accepts both the Phase-1
/// bare-name form and the Phase-2 `{name, bbox}` form.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PersonaRef {
    /// `personas: [ alice ]` — single persona over the whole image.
    /// Errors at load when used alongside the bbox form or when `>1` of
    /// these appear in the same task.
    Bare(String),
    /// `personas: [ { name: alice, bbox: [x0,y0,x1,y1] } ]`. Multi-persona
    /// compositing path; allowed even with a single persona (works as a
    /// single-region inpaint, useful for fine framing control).
    Bbox(PersonaBboxRef),
}

#[derive(Debug, Deserialize)]
struct PersonaBboxRef {
    name: String,
    /// `[x0, y0, x1, y1]`. Validated at load: components in `[0, 1]`,
    /// `x0 < x1`, `y0 < y1`. Pixel-space coordinates are derived from
    /// the task's effective `(width, height)` at dispatch time.
    bbox: [f32; 4],
}

impl PersonaRef {
    fn name(&self) -> &str {
        match self {
            Self::Bare(n) => n,
            Self::Bbox(b) => &b.name,
        }
    }
    fn bbox(&self) -> Option<[f32; 4]> {
        match self {
            Self::Bare(_) => None,
            Self::Bbox(b) => Some(b.bbox),
        }
    }
}

/// Parse a scenario file just far enough to report `(task_count, model)` for a
/// listing UI (the `plakat ui` Scenarios screen). Reuses the real `ScenarioFile`
/// deserialize so every HJSON quirk (quoteless `size: 512x512`, comments, the full
/// field set) is handled correctly — a hand-rolled minimal struct trips over
/// unknown quoteless values. Returns an error on a malformed file.
pub fn peek(path: &std::path::Path) -> Result<(usize, String)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let s: ScenarioFile = deser_hjson::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok((s.tasks.len(), s.model.unwrap_or_else(|| "?".to_string())))
}

/// A persona defined in a scenario's top-level `personas:` block, summarised for the
/// People screen (RFC TUI-1 §11 — "read people also from scenario HJSON"). Photo
/// paths are resolved relative to the scenario file's directory.
#[derive(Debug, Clone)]
pub struct PersonaSummary {
    pub name: String,
    pub identity: Option<String>,
    pub face_strength: Option<f32>,
    /// `(path, weight)` reference photos.
    pub photos: Vec<(PathBuf, f32)>,
}

/// Read the `personas:` block of a scenario file. Empty when the file defines none.
pub fn peek_personas(path: &std::path::Path) -> Result<Vec<PersonaSummary>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let s: ScenarioFile = deser_hjson::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut out = Vec::new();
    for p in &s.personas {
        let mut photos: Vec<(PathBuf, f32)> = Vec::new();
        if let Some(ph) = &p.photo {
            photos.push((base.join(ph), 1.0));
        }
        for ph in &p.photos {
            if let Ok(wp) = ph.to_weighted() {
                photos.push((base.join(&wp.path), wp.weight.unwrap_or(1.0)));
            }
        }
        out.push(PersonaSummary {
            name: p.name.clone(),
            identity: p.identity.clone(),
            face_strength: p.face_strength,
            photos,
        });
    }
    Ok(out)
}

/// The ordered task names in a scenario file — for a runner UI to pre-populate a
/// per-task status board before the run starts. Same parser as [`peek`].
pub fn task_names(path: &std::path::Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let s: ScenarioFile = deser_hjson::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(s.tasks.iter().map(|t| t.name.clone()).collect())
}

/// Live per-task progress, emitted by [`run_with_events`] so a UI (the `plakat ui`
/// Scenarios RUNNER board) can show a status board distinct from the flat rerouted
/// log. `index` is the task's position in the scenario; `status` mirrors the
/// `TaskRunRecord.status` strings (`ok` / `failed` / `skipped` / `dry-run`).
#[derive(Debug, Clone)]
pub enum ScenarioEvent {
    Started { total: usize },
    TaskStarted { index: usize, name: String },
    TaskFinished { index: usize, name: String, status: String },
    Finished { ok: usize, failed: usize },
}

/// Best-effort emit (no-op when no sink is wired, e.g. the CLI path).
fn emit(events: &Option<std::sync::mpsc::Sender<ScenarioEvent>>, ev: ScenarioEvent) {
    if let Some(tx) = events {
        let _ = tx.send(ev);
    }
}

/// Run a scenario, dispatching CLI-side (no structured event sink). The
/// task-by-task progress still streams to `ui::progress` as before.
pub async fn run(args: ScenarioArgs) -> Result<()> {
    run_with_events(args, None, None).await
}

/// Run a scenario, optionally streaming [`ScenarioEvent`]s to `events` for a live
/// status board. Passing `None, None` is byte-identical to [`run`].
///
/// `preloaded_sd` is a `(model_alias, Pipeline)` a caller (the TUI) already has resident
/// and offers for reuse. It is used ONLY when the scenario is an all-generate SD-family
/// run whose model matches `model_alias`, has no scenario-level LoRAs, and no refiner —
/// i.e. exactly the vanilla base the runner would otherwise load. In every other case it
/// is dropped up front (before any other pipeline loads), so reuse can never alter output.
pub async fn run_with_events(
    args: ScenarioArgs,
    events: Option<std::sync::mpsc::Sender<ScenarioEvent>>,
    preloaded_sd: Option<(String, Pipeline)>,
) -> Result<()> {
    // `-` reads the scenario from stdin (pipe integration: `plakat compile … --out -
    // | plakat scenario -`).
    let text = if args.file.as_os_str() == "-" {
        use std::io::Read;
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .context("reading scenario from stdin")?;
        s
    } else {
        std::fs::read_to_string(&args.file)
            .with_context(|| format!("reading {}", args.file.display()))?
    };
    let s: ScenarioFile = deser_hjson::from_str(&text)
        .map_err(|e| {
            // v0.33 phase 1: enrich with the surrounding task name
            // when discoverable. Best-effort — falls through to the
            // bare error when the line-number heuristic can't find
            // a task boundary above the failure.
            crate::error_hints::decorate_scenario_parse(
                anyhow::Error::msg(e.to_string()),
                &text,
            )
        })
        .with_context(|| format!("parsing HJSON {}", args.file.display()))?;

    // -------- validate structure --------
    if s.tasks.is_empty() {
        bail!("scenario has no `tasks` to run");
    }

    // v2.7 feature-sync: promote the free-quality guidance-bundle knobs to the env the pipelines
    // read (same mechanism as the `generate` CLI), so scenarios can drive the full quality toolchain.
    // Scenario-global (applies to every task); only set when explicitly requested.
    if let Some(v) = s.pag_scale {
        if v > 0.0 {
            unsafe { std::env::set_var("PLAKAT_PAG_SCALE", v.to_string()) };
        }
    }
    if let Some(v) = s.guidance_rescale {
        if v > 0.0 {
            unsafe { std::env::set_var("PLAKAT_CFG_RESCALE", v.to_string()) };
        }
    }
    if let Some(v) = s.dynamic_threshold {
        if v > 0.0 {
            unsafe { std::env::set_var("PLAKAT_DYNTHRESH", v.to_string()) };
        }
    }
    if let Some(p) = &s.freeu_params {
        unsafe { std::env::set_var("PLAKAT_FREEU", p) };
    } else if s.freeu {
        unsafe { std::env::set_var("PLAKAT_FREEU", "1") };
    }

    // v0.29 phases 2+3: classify each task by kind + validate the
    // animate effective config up-front. Validation here means
    // schema typos (window-size > 32, format: avif) bail before
    // any pipeline load. Phase 3 wires the actual dispatch inside
    // the task loop below. `has_generate_tasks` gates the enhancer
    // requirement: all-animate scenarios don't need an enhancer
    // (animate doesn't run prompt enhancement).
    let mut has_generate_tasks = false;
    for t in &s.tasks {
        let kind = TaskKind::from_strs(
            t.task_type.as_deref(),
            s.task_type.as_deref(),
        )?;
        match kind {
            TaskKind::Animate => {
                let eff = effective_animate_config(&s, t)?;
                eff.validate(&t.name)?;
            }
            TaskKind::Generate => {
                has_generate_tasks = true;
            }
            TaskKind::Map => {
                // Validate the style up front; spec sourcing happens at run time.
                let cfg = effective_map_config(&s, t);
                crate::map::render::Style::named(&cfg.style)?;
            }
            TaskKind::Multiperson => {
                let spec = t.multiperson.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "task {:?} is type `multiperson` but has no `multiperson:` block",
                        t.name
                    )
                })?;
                if spec.people.is_empty() {
                    bail!("task {:?}: multiperson `people` must list at least one persona", t.name);
                }
                // Every referenced persona must exist in the top-level list and
                // any identity override must parse — fail before the model load.
                let known: BTreeSet<&str> = s.personas.iter().map(|p| p.name.as_str()).collect();
                for r in &spec.people {
                    if !known.contains(r.persona.as_str()) {
                        bail!(
                            "task {:?}: multiperson references unknown persona {:?} \
                             (define it in the top-level `personas:` list)",
                            t.name, r.persona
                        );
                    }
                }
                if let Some(id) = &spec.identity {
                    id.parse::<crate::pipelines::ip_adapter::IdentityKind>()
                        .map_err(|e| anyhow::anyhow!("task {:?}: identity {id:?}: {e}", t.name))?;
                }
            }
            #[cfg(feature = "fractals")]
            TaskKind::Fractal => {
                // Validate the spec up front (kind/coloring/presets parse, dims non-zero).
                let cfg = t.fractal.clone().unwrap_or_default();
                crate::fractals::scenario_task::build_spec(&cfg, 0)
                    .with_context(|| format!("task {:?} (fractal)", t.name))?;
            }
            TaskKind::Bookart => {
                // Validate the ornament spec up front (sources + resolves) — before any model load.
                let cfg = t.bookart.clone().unwrap_or_default();
                crate::bookart::scenario_task::validate(&cfg).with_context(|| format!("task {:?} (bookart)", t.name))?;
            }
            TaskKind::Texture => {
                let cfg = t.texture.clone().unwrap_or_default();
                crate::texture::scenario_task::validate(&cfg).with_context(|| format!("task {:?} (texture)", t.name))?;
            }
            TaskKind::Comic => {
                let cfg = t.comic.clone().unwrap_or_default();
                crate::comic::scenario_task::validate(&cfg).with_context(|| format!("task {:?} (comic)", t.name))?;
            }
        }
    }

    let scenes: HashMap<&str, &str> = s
        .scene
        .iter()
        .map(|p| (p.name.as_str(), p.prompt.as_str()))
        .collect();
    let weathers: HashMap<&str, &str> = s
        .weather
        .iter()
        .map(|p| (p.name.as_str(), p.prompt.as_str()))
        .collect();
    for t in &s.tasks {
        // Map / multiperson tasks don't use scene/weather styling — skip the
        // cross-reference check (multiperson carries its own scene prompt).
        let tk = TaskKind::from_strs(t.task_type.as_deref(), s.task_type.as_deref());
        if matches!(tk, Ok(TaskKind::Map | TaskKind::Multiperson)) {
            continue;
        }
        #[cfg(feature = "fractals")]
        if matches!(tk, Ok(TaskKind::Fractal)) {
            continue;
        }
        if matches!(tk, Ok(TaskKind::Bookart | TaskKind::Texture | TaskKind::Comic)) {
            continue;
        }
        if !scenes.contains_key(t.scene.as_str()) {
            bail!("task {:?} references unknown scene {:?}", t.name, t.scene);
        }
        if !weathers.contains_key(t.weather.as_str()) {
            bail!("task {:?} references unknown weather {:?}", t.name, t.weather);
        }
    }

    // -------- personas: validate + index by name --------
    // Build a name → PersonaDef map and pre-flight every field that could
    // fail later (identity-kind parse, photo existence) so the scenario
    // fails before the model load.
    let personas_map: BTreeMap<&str, &PersonaDef> = {
        let mut map: BTreeMap<&str, &PersonaDef> = BTreeMap::new();
        for p in &s.personas {
            if map.contains_key(p.name.as_str()) {
                bail!("duplicate persona name {:?}", p.name);
            }
            // Resolve photo / photos → normalized weighted list. This
            // also enforces mutual-exclusion + "at least one photo set".
            let resolved = p.resolve_photos()?;
            // Verify every referenced photo exists. For multi-photo
            // personas this lets us surface a missing path before the
            // long model load.
            for wp in &resolved {
                if !wp.path.exists() {
                    bail!(
                        "persona {:?}: photo not found at {}",
                        p.name,
                        wp.path.display()
                    );
                }
            }
            if let Some(id) = p.identity.as_deref() {
                // Parse just to validate; the parsed value is recomputed at
                // dispatch time so the persona record stays as written.
                let _: IdentityKind = id
                    .parse()
                    .with_context(|| format!("persona {:?} identity", p.name))?;
            }
            if let Some([x0, y0, x1, y1]) = p.face_bbox {
                let in_unit = (0.0..=1.0).contains(&x0)
                    && (0.0..=1.0).contains(&y0)
                    && (0.0..=1.0).contains(&x1)
                    && (0.0..=1.0).contains(&y1);
                if !in_unit || x0 >= x1 || y0 >= y1 {
                    bail!(
                        "persona {:?} face-bbox {:?} is invalid \
                         (must be [x0,y0,x1,y1] with 0 ≤ x0 < x1 ≤ 1 \
                         and 0 ≤ y0 < y1 ≤ 1)",
                        p.name,
                        p.face_bbox.unwrap(),
                    );
                }
            }
            if let Some(lm) = p.face_landmarks {
                for (i, [x, y]) in lm.iter().enumerate() {
                    if !(0.0..=1.0).contains(x) || !(0.0..=1.0).contains(y) {
                        bail!(
                            "persona {:?} face-landmarks point {} = [{}, {}] \
                             is out of range [0, 1]",
                            p.name,
                            i,
                            x,
                            y
                        );
                    }
                }
            }
            map.insert(p.name.as_str(), p);
        }
        map
    };

    // Validate task → persona references and enforce form-mixing rules.
    for t in &s.tasks {
        if let Some(refs) = &t.personas {
            // Resolve every name first.
            for r in refs {
                if !personas_map.contains_key(r.name()) {
                    let known: Vec<&str> = personas_map.keys().copied().collect();
                    bail!(
                        "task {:?} references unknown persona {:?} (defined: [{}])",
                        t.name,
                        r.name(),
                        known.join(", ")
                    );
                }
            }
            // Validate bbox bounds on every Bbox variant.
            for r in refs {
                if let Some([x0, y0, x1, y1]) = r.bbox() {
                    let inside_unit = (0.0..=1.0).contains(&x0)
                        && (0.0..=1.0).contains(&y0)
                        && (0.0..=1.0).contains(&x1)
                        && (0.0..=1.0).contains(&y1);
                    if !inside_unit || x0 >= x1 || y0 >= y1 {
                        bail!(
                            "task {:?}: persona {:?} bbox {:?} is invalid \
                             (must be [x0,y0,x1,y1] with 0 ≤ x0 < x1 ≤ 1 \
                             and 0 ≤ y0 < y1 ≤ 1)",
                            t.name,
                            r.name(),
                            r.bbox().unwrap(),
                        );
                    }
                }
            }
            // Form-mixing rule: within a single task, every entry must
            // use the bare-name form OR every entry must use the bbox
            // form. No mixing.
            let any_bare = refs.iter().any(|r| matches!(r, PersonaRef::Bare(_)));
            let any_bbox = refs.iter().any(|r| matches!(r, PersonaRef::Bbox(_)));
            if any_bare && any_bbox {
                bail!(
                    "task {:?}: cannot mix bare-name form (`[alice]`) with \
                     bbox form (`[{{name:alice, bbox:[...]}}]`) in the same \
                     task. Pick one. Use bbox for multi-persona compositing; \
                     use bare-name when the persona occupies the whole image.",
                    t.name
                );
            }
            // Bare-form: still capped at 1 (Phase-2 multi-persona requires
            // bboxes; bare-form `[alice, bob]` has no way to place them).
            if any_bare && refs.len() > 1 {
                let names: Vec<&str> = refs.iter().map(|r| r.name()).collect();
                bail!(
                    "task {:?} requests {} personas ({}) in bare-name form. \
                     Multi-persona requires bboxes — convert to \
                     `[{{name:..., bbox:[x0,y0,x1,y1]}}, ...]` form.",
                    t.name,
                    refs.len(),
                    names.join(", "),
                );
            }
        }
    }

    // -------- personas: agree on a single identity strategy --------
    // The portrait pipeline picks its base model (SD 1.5 / SDXL) from the
    // identity strategy used by the personas referenced in this scenario.
    // Mixed-variant scenarios would require loading two portrait pipelines
    // simultaneously (~10 GB+ of resident weights) and aren't supported.
    // Computed once here so the preload below can use it; also used to
    // surface a clear error if the user accidentally mixes strategies.
    let persona_kinds: HashMap<IdentityKind, Vec<&str>> = {
        let mut m: HashMap<IdentityKind, Vec<&str>> = HashMap::new();
        for p in &s.personas {
            let kind: IdentityKind = p
                .identity
                .as_deref()
                .unwrap_or("plus-face")
                .parse()
                .with_context(|| format!("persona {:?} identity", p.name))?;
            m.entry(kind).or_default().push(p.name.as_str());
        }
        m
    };
    let portrait_identity: Option<IdentityKind> = match persona_kinds.len() {
        0 => None,
        1 => Some(*persona_kinds.keys().next().unwrap()),
        _ => {
            let mix: Vec<String> = persona_kinds
                .iter()
                .map(|(k, names)| format!("{k:?} ({})", names.join(", ")))
                .collect();
            bail!(
                "scenario mixes identity strategies across personas: {}. \
                 Pick one strategy per scenario — every persona must share the \
                 same model variant (all SD 1.5 `plus-face`, or all SDXL \
                 `plus-face-sdxl`).",
                mix.join("; "),
            );
        }
    };

    // v0.29 phase 3: all-animate scenarios don't need an enhancer
    // (the enhance step is t2i-only). Default to "local" for the
    // logging/var-passing surface; the enhance cache stays empty
    // for animate tasks regardless.
    let enhancer = match s.enhancer.clone() {
        Some(e) => {
            validate_enhancer_keys(&e)?;
            e
        }
        None if !has_generate_tasks => "local".to_string(),
        None => {
            bail!("scenario requires `enhancer` (deepseek | gemini)");
        }
    };

    // Parse the upscale method now so a bad string fails fast.
    let upscale_method: UpscaleMethod = s
        .upscale
        .method
        .parse()
        .with_context(|| format!("upscale.method = {:?}", s.upscale.method))?;

    // -------- resolve global parameters --------
    let model = s.model.clone().unwrap_or_else(|| "sd15".to_string());
    let device = crate::device::select(s.device.as_deref().unwrap_or("auto"))?;
    // Memory safety for the whole batch (loaded once, looped): warn up-front if
    // RAM is already tight, and run a watchdog that aborts plakat before a
    // unified-memory exhaustion can take the host down.
    crate::hw::memory_preflight(&device, &model);
    let _mem_guard = crate::memwatch::MemoryGuard::start(&device, &model);
    let base = s.base.unwrap_or(768);
    let count = s.count.unwrap_or(1);
    // `steps` / `guidance` start at the user's scenario values; the
    // v0.15 phase 7a `fast:` preset application below may override
    // them when the user didn't explicitly set them (i.e. left them
    // at plakat's documented defaults 28 / 7.5).
    let mut steps = s.steps.unwrap_or(28);
    let mut guidance = s.guidance.unwrap_or(7.5);
    let seed = s.seed.unwrap_or(0);
    // The TUI's `out_override` wins so scenario images land under the workspace `out/`
    // (where History scans); else the scenario's own `out:`; else `./out`.
    let out_root = args
        .out_override
        .clone()
        .or_else(|| s.out.clone())
        .unwrap_or_else(|| PathBuf::from("./out"));
    let lora_scale = s.lora_scale.unwrap_or(1.0);
    let refine_strength = s.refine_strength.unwrap_or(0.3);
    let scheduler: SchedulerKind = match s.scheduler.as_deref() {
        Some(x) => x.parse().with_context(|| format!("scheduler {x:?}"))?,
        None => SchedulerKind::Default,
    };

    let size = match s.size.as_deref() {
        Some(s) => Some(s.parse::<Size>().with_context(|| format!("size {s:?}"))?),
        None => None,
    };
    let (width, height) = crate::imaging::sizes::resolve(size, s.aspect.as_deref(), base)?;

    let mut loras: Vec<LoraSpec> = s
        .loras
        .iter()
        .map(|x| x.parse::<LoraSpec>())
        .collect::<Result<Vec<_>>>()?;

    // -------- v0.15 phase 7a: scenario-level `fast:` preset --------
    // Mirrors `plakat generate --fast PRESET` semantics:
    // * Prepend the preset's distillation LoRA to the scenario LoRA stack
    //   (so user-supplied LoRAs land later and can override on collision)
    // * Override `steps` / `guidance` only when user left them at the
    //   plakat defaults (28 / 7.5).
    // Per-task `fast:` is allowed iff equal to scenario-level — divergent
    // per-task presets need v0.15 phase 7b runtime LoRA.
    if let Some(name) = s.fast.as_deref() {
        let preset_arg: crate::pipelines::flux_fast::FastPresetArg = name
            .parse()
            .with_context(|| format!("scenario fast preset {name:?}"))?;
        let preset = preset_arg.0;
        let m = model.to_lowercase();
        if !m.contains("flux") {
            anyhow::bail!(
                "scenario `fast: {}` requires a Flux model (got model {:?}). \
                 Hyper-FLUX / FLUX-Turbo presets are Flux-family only.",
                preset.name, model
            );
        }
        if m.contains("fill") {
            anyhow::bail!(
                "scenario `fast: {}` doesn't compose with flux-fill-dev — Fill \
                 needs its own forward path.",
                preset.name
            );
        }
        // v0.15 phase 1 unblocked NF4 + LoRA + CN; fast (LoRA-based)
        // composes too. No NF4 bail here, unlike pre-v0.15 generate.rs.
        loras.insert(0, preset.to_lora_spec());
        if s.steps.is_none() {
            steps = preset.steps;
        }
        if s.guidance.is_none() {
            guidance = preset.guidance;
        }
        tracing::info!(
            target: "plakat",
            "scenario fast preset '{}': +{} LoRA, steps={steps}, guidance={guidance}",
            preset.name, preset.lora_repo
        );
        // Per-task `fast:` validation: must match scenario when both
        // are set, and must NOT be set per-task when scenario isn't
        // (the load-once design can't swap presets mid-run).
        for task in &s.tasks {
            if let Some(task_fast) = task.fast.as_deref() {
                if task_fast != name {
                    anyhow::bail!(
                        "task {:?} declares `fast: {task_fast}` but scenario uses \
                         `fast: {name}` — per-task preset swaps require runtime \
                         LoRA (deferred to v0.15 phase 7b). For now, all tasks must \
                         share the scenario preset.",
                        task.name
                    );
                }
            }
        }
    } else {
        // No scenario-level fast — reject per-task fast too. The
        // pre-loaded SdCore/Flux backbone doesn't have the preset's
        // distillation LoRA merged, so applying just the
        // step/guidance overrides per task would silently produce
        // garbage outputs.
        for task in &s.tasks {
            if let Some(task_fast) = task.fast.as_deref() {
                anyhow::bail!(
                    "task {:?} declares `fast: {task_fast}` but scenario has no \
                     scenario-level `fast:` — promote it to the scenario level so \
                     the preset's distillation LoRA is loaded into the pipeline.",
                    task.name
                );
            }
        }
    }

    // -------- v0.13 phase 11: validate per-task fields --------
    // Surface schema errors (mutually-exclusive control/controls,
    // outpaint without init-image, etc.) before we burn time on model
    // loads. Same loop applies the validators across every task.
    // An explicit `count: 0` (scenario or task) would run the generate loop `0..0`,
    // write nothing, and still record `✓ done` — a task that produced no output looking
    // like success. Reject it up front. (Absent = defaults to 1.)
    if s.count == Some(0) {
        anyhow::bail!("scenario `count` must be >= 1 (0 produces no images)");
    }
    for task in &s.tasks {
        if task.count == Some(0) {
            anyhow::bail!("task {:?}: `count` must be >= 1 (0 produces no images)", task.name);
        }
        task_effective_controls(task).with_context(|| {
            format!("validating ControlNet config for task {:?}", task.name)
        })?;
        task_validate_image_inputs(task).with_context(|| {
            format!("validating image inputs for task {:?}", task.name)
        })?;
    }
    // Task outputs land under `safe_name(task.name)`, which collapses every non-alnum char
    // to `_`, so names differing only in punctuation (e.g. "a b" and "a/b") share a dir and
    // — with the same explicit seed — overwrite each other's images. Reject the collision.
    {
        let mut seen = std::collections::HashSet::new();
        for task in &s.tasks {
            let dir = safe_name(&task.name);
            if !seen.insert(dir.clone()) {
                anyhow::bail!(
                    "two tasks map to the same output dir {dir:?} (names differ only in \
                     punctuation) — rename one so their images don't overwrite"
                );
            }
        }
    }

    // -------- style detection / transfer --------
    // A single `StyleSession` is constructed when any task (or the
    // scenario globally) uses style detection. The session shares one
    // CLIP-H encoder load across global + every per-task style-ref;
    // a 5-task scenario where every task has its own style-ref pays
    // for the ~2.5 GB encoder weights exactly once.
    let any_task_style = s.tasks.iter().any(|t| t.style_ref_catalog.is_some());
    let scenario_uses_style = s.style_ref.is_some() || s.style.is_some() || any_task_style;
    let mut style_session: Option<StyleSession> = if scenario_uses_style {
        Some(StyleSession::load(s.style_catalog.as_deref(), device.clone())?)
    } else {
        None
    };

    // Global style block — applied to all tasks that don't have their
    // own per-task style override.
    let mut effective_lora_header = s.lora_header.clone();
    let mut effective_negative = s.negative.clone();
    if s.style_ref.is_some() || s.style.is_some() {
        let n_user_loras = loras.len();
        let session = style_session.as_mut().expect("session created when any style is set");
        let prep = session
            .prepare(StylePrepRequest {
                style_ref: s.style_ref.as_deref(),
                style_override: s.style.as_deref(),
                style_strength: s.style_strength.unwrap_or(1.0),
                style_catalog: None, // session already locked the catalog in
                model: &model,
                user_loras_nonempty: !loras.is_empty(),
                device: &device,
            })
            .await?;

        log_style_prep(&prep, n_user_loras);

        loras = parse_resolved_loras(&prep)?;
        effective_lora_header = prepend_trigger(&prep.trigger, &effective_lora_header);
        effective_negative = combine_negative(&effective_negative, &prep.negative_extras);
    }

    // v0.16 phase 11: preflight check — when running SD-family with
    // per-task LoRA stacks, surface the hint NOW rather than at the
    // first task's apply_loras bail. Includes the "all per-task
    // LoRAs identical → fold into scenario.loras" suggestion when
    // applicable, with the exact YAML to copy.
    sd_per_task_lora_preflight(&s, &model)?;

    // In the `plakat ui` TUI, raw stdout scribbles over the alternate screen — every
    // status line must go through the rerouted progress sink (which falls back to
    // stdout on the CLI). `sout!` is `println!`-shaped but sink-safe.
    macro_rules! sout {
        ($($a:tt)*) => { crate::ui::progress::println(&format!($($a)*)) };
    }

    // -------- execution plan summary --------
    let total_images = (s.tasks.len() as u32) * count;
    sout!(
        "{}  {} task(s) × {} image(s) = {} image(s) to generate",
        style("scenario").yellow().bold(),
        s.tasks.len(),
        count,
        total_images,
    );
    sout!("  model:     {model}");
    sout!("  size:      {width}×{height}");
    sout!("  steps:     {steps}  guidance: {guidance}  scheduler: {scheduler:?}");
    sout!("  out:       {}", out_root.display());
    sout!("  enhancer:  {enhancer}");
    if !loras.is_empty() {
        sout!("  loras:     {} (scale {lora_scale})", loras.len());
    }
    if let Some(r) = s.refine {
        sout!("  refine:    {r} steps × strength {refine_strength}");
    }
    if s.refiner {
        let frac = s.refiner_frac.unwrap_or(0.8);
        sout!(
            "  refiner:   on (switch at {:.0}% of schedule, SDXL only)",
            frac * 100.0
        );
    }
    if s.upscale.upscale {
        let shown = upscale_method.native_scale().unwrap_or(s.upscale.scale);
        sout!(
            "  upscale:   {:.2}× {} (post-stylize if `style` is set, else original)",
            shown, s.upscale.method
        );
    }
    // v0.13 phase 10: print effective tiled / GGUF settings when set.
    if let Some(tcfg) = s.tiled {
        sout!(
            "  tiled:     {}px tiles, stride {}px (MultiDiffusion)",
            tcfg.size, tcfg.stride
        );
    }
    if s.quantize_t5
        || s.quant_level.is_some()
        || s.t5_quant_level.is_some()
    {
        let q = s.quant_level.as_deref().unwrap_or("Q4_K_S");
        if s.quantize_t5 {
            let t5q = s.t5_quant_level.as_deref().unwrap_or("Q4_K_M");
            sout!("  gguf:      Flux={q}, T5={t5q} (quantized T5)");
        } else {
            sout!("  gguf:      Flux={q} (T5 stays BF16)");
        }
    }
    if !s.personas.is_empty() {
        let names: Vec<&str> = s.personas.iter().map(|p| p.name.as_str()).collect();
        let persona_tasks = s
            .tasks
            .iter()
            .filter(|t| t.personas.as_deref().map(|p| !p.is_empty()).unwrap_or(false))
            .count();
        let portrait_label = portrait_identity
            .map(|k| k.label())
            .unwrap_or("(unused — no persona tasks)");
        sout!(
            "  personas:  {} defined [{}], used by {} task(s) — {}",
            s.personas.len(),
            names.join(", "),
            persona_tasks,
            portrait_label,
        );
    }

    if !args.dry_run {
        std::fs::create_dir_all(&out_root)?;
    }

    // -------- preload the Real-ESRGAN model if used --------
    // Without this, every task would re-download + re-build the model.
    let esrgan: Option<EsrganPipeline> =
        if !args.dry_run && s.upscale.upscale && upscale_method.is_ml() {
            Some(EsrganPipeline::load(upscale_method, &device).await?)
        } else {
            None
        };

    // -------- load pipeline (lazy after v0.31 phase 3) --------
    // Three parallel pipeline types; exactly one is populated for
    // non-dry-run runs (SD-family / Flux / SD3-family).
    //
    // v0.31 phase 3: SD-family `pipeline` is now lazy — the
    // initial Option<Pipeline> stays None for all-animate scenarios
    // (no generate tasks → no t2i backbone needed) AND for the
    // first task slot of mixed-kind scenarios (the loop's kind-
    // switching evictor drops it on switch to animate, the first-
    // generate-task path reloads it). For all-generate scenarios,
    // we still pre-load it here so the user sees the load progress
    // before the first task runs (preserves the v0.29 UX).
    let variant = Variant::detect(&model);
    let any_animate_tasks = s.tasks.iter().any(|t| {
        matches!(
            TaskKind::from_strs(t.task_type.as_deref(), s.task_type.as_deref()),
            Ok(TaskKind::Animate)
        )
    });
    let will_use_sd_base = !(args.dry_run
        || variant.is_flux()
        || variant.is_sd3()
        || variant.is_pixart()
        || variant.is_cascade()
        || !has_generate_tasks
        || any_animate_tasks);
    // A handed-off Chat pipeline may be reused only when it IS the exact base this run
    // would load: same model, no scenario-level LoRAs, no refiner (per-task LoRAs then
    // apply identically on top). Anything else → drop it now, before other pipelines load.
    let preloaded_sd = preloaded_sd
        .filter(|(a, _)| can_reuse_sd_pipeline(will_use_sd_base, a, &model, &loras, s.refiner));
    let mut pipeline: Option<Pipeline> = if !will_use_sd_base {
        // Mixed-kind (or all-animate, or non-SD-family) → defer. PixArt
        // and Stable Cascade have their own pre-loaded pipelines
        // (`pixart_pipeline` / `cascade_pipeline`) and must NOT hit the
        // SD-only `load_sd_pipeline_for_scenario`, which bails on them.
        None
    } else if let Some((_, pipe)) = preloaded_sd {
        // Reuse the resident Chat pipeline — no reload.
        crate::ui::progress::println(&format!("  reusing the loaded {model} — no reload"));
        Some(pipe)
    } else {
        // All-generate SD-family scenario — pre-load as before so
        // the user sees the "Loading SD core" spinner up front.
        Some(load_sd_pipeline_for_scenario(&model, &device, &loras, lora_scale, s.refiner, None).await?)
    };
    // v0.16 phase 2: SD3 / SD3.5 backbone loaded once for the whole
    // scenario (previously the scenario fell through to t2i::Pipeline
    // which bailed on SD3). Per-task LoRA dispatches via
    // `sd3::Pipeline::apply_loras` between tasks.
    let mut sd3_pipeline: Option<crate::pipelines::sd3::Pipeline> =
        if args.dry_run || !variant.is_sd3() {
            None
        } else {
            use crate::pipelines::sd3;
            let sd3_variant = match variant {
                Variant::Sd3Medium => sd3::Variant::Sd3Medium,
                Variant::Sd35Medium => sd3::Variant::Sd35Medium,
                Variant::Sd35Large => sd3::Variant::Sd35Large,
                Variant::Sd35LargeTurbo => sd3::Variant::Sd35LargeTurbo,
                _ => unreachable!("is_sd3() implies one of the SD3 variants"),
            };
            let resolved_repo = if model.contains('/') {
                model.clone()
            } else {
                crate::hf::resolve_alias(&model).to_string()
            };
            Some(
                sd3::Pipeline::load(sd3::LoadRequest {
                    variant: sd3_variant,
                    repo: resolved_repo,
                    device: device.clone(),
                    loras: loras.clone(),
                    lora_scale,
                    // SD3 ControlNet wiring into scenarios lands in
                    // a later phase. For now scenarios load with no
                    // SD3 CN slots; the per-task CN spec dispatch
                    // mirrors what the existing Flux scenario path
                    // does (max_flux_controls + scenario-wide
                    // preload).
                    controlnets: Vec::new(),
                    embeddings: Vec::new(),
                })
                .await?,
            )
        };
    // -------- preload the stylize pipeline if any task uses `style` --------
    let any_style = s.tasks.iter().any(|t| t.style.is_some());

    // -------- preload the portrait pipeline if any task uses `personas` ----
    let any_persona = s
        .tasks
        .iter()
        .any(|t| t.personas.as_deref().map(|v| !v.is_empty()).unwrap_or(false));

    // A scenario using BOTH style and personas holds several pipelines co-resident for
    // the whole run — the main model + stylize (SD1.5) + the persona portrait model
    // (SDXL for plus-face-sdxl) + shared CLIP-H — which can exceed a 24 GB budget (they
    // aren't evicted between tasks; a lazy per-kind lifecycle is a future refactor).
    // Warn up front so the OOM is expected + actionable; the memory guard is the backstop.
    if !args.dry_run && any_style && any_persona {
        let msg = "scenario uses both `style` and `personas` — the main model + stylize \
             (SD1.5) + the persona portrait model + shared CLIP-H stay co-resident for the \
             whole run and may exceed unified memory. If it OOMs, split style and persona \
             tasks into separate scenarios, or use a smaller base / --size.";
        tracing::warn!(target: "plakat", "{msg}");
        crate::ui::progress::println(&format!("⚠ {msg}"));
    }

    // Phase 7f: pre-load a single CLIP-H image encoder when both
    // stylize and a Plus-Face portrait identity are going to run.
    // FaceID strategies don't touch CLIP-H, so they don't trigger the
    // share. The shared Arc is then fed into both pipelines' load
    // requests so each skips its own download / mmap.
    let plusface_portrait = any_persona
        && matches!(
            portrait_identity,
            Some(IdentityKind::PlusFace) | Some(IdentityKind::PlusFaceSdxl)
        );
    let shared_clip_h: Option<std::sync::Arc<crate::pipelines::ip_adapter::ImageEncoder>> =
        if !args.dry_run && any_style && plusface_portrait {
            let spinner = crate::ui::progress::spinner(
                "Pre-loading shared CLIP-H image encoder",
            );
            // F32 matches stylize's standalone choice and the portrait
            // identity encoder casts down at encode-time as needed.
            let arc = crate::pipelines::ip_adapter::load_shared_clip_vision(
                &device,
                candle_core::DType::F32,
            )
            .await?;
            spinner.finish_with_message("✓ shared CLIP-H loaded");
            Some(arc)
        } else {
            None
        };

    let stylize_pipeline: Option<stylize::Pipeline> = if !args.dry_run && any_style {
        // stylize is SD 1.5 only — the IP-Adapter projection targets the SD 1.5
        // cross-attention dim (768). The scenario's main `model` can still be
        // anything (we operate on the produced image bytes, not on latents).
        Some(
            stylize::Pipeline::load(stylize::LoadRequest {
                model: "sd15".to_string(),
                device: device.clone(),
                shared_clip_h: shared_clip_h.clone(),
                instantstyle: false,
                style_scale: 1.0,
            })
            .await?,
        )
    } else {
        None
    };

    let portrait_pipeline: Option<portrait::Pipeline> = if !args.dry_run && any_persona {
        // Portrait base model is derived from the scenario's persona
        // identity kind (all personas must agree — validated above). The
        // scenario's main `model` field stays separate: non-persona tasks
        // use that; persona tasks use this portrait pipeline.
        let kind = portrait_identity
            .expect("any_persona implies at least one persona, validation ensures kind agreement");
        let portrait_model = kind.target_variant().to_string();
        Some(
            portrait::Pipeline::load(portrait::LoadRequest {
                model: portrait_model,
                device: device.clone(),
                loras: loras.clone(),
                lora_scale,
                identity: Some(kind),
                // Only Plus-Face identity strategies consume CLIP-H;
                // FaceID strategies ignore this even when set.
                shared_clip_h: shared_clip_h.clone(),
            })
            .await?,
        )
    } else {
        None
    };

    let mut flux_pipeline: Option<flux::Pipeline> = if args.dry_run || !variant.is_flux() {
        None
    } else {
        if !loras.is_empty() {
            crate::ui::progress::println(&format!(
                "  {} ignoring {} LoRA file(s): SD-format LoRAs don't apply to Flux's transformer",
                style("warn:").yellow().bold(),
                loras.len()
            ));
        }
        let fvar = match variant {
            Variant::FluxDev => flux::Variant::Dev,
            Variant::FluxFillDev => flux::Variant::FillDev,
            _ => flux::Variant::Schnell,
        };
        let resolved_repo = if model.contains('/') {
            model.clone()
        } else {
            crate::hf::resolve_alias(&model).to_string()
        };
        // v0.12: Flux LoRAs (PEFT-format) are now supported via
        // flux_lora::merge_flux_loras_into_weights. Resolve the
        // scenario's `loras` here so the merge runs at load time.
        let resolved_flux_loras: Vec<crate::pipelines::lora::ResolvedLora> =
            if loras.is_empty() {
                Vec::new()
            } else {
                let mut v = Vec::with_capacity(loras.len());
                for spec in &loras {
                    v.push(spec.resolve().await?);
                }
                v
            };
        // v0.13 phase 10: pre-load Shakker-Labs Union Pro v2 once if
        // any task in the scenario uses `control:`. Union Pro v2 covers
        // canny/softedge/openpose/depth/lineart via a single weight set
        // — `set_controlnet_call_params` swaps the mode + scale per
        // task. Tasks that have no `control:` simply leave the CN's
        // conditioning empty and it contributes no residuals.
        // v0.13 phase 11: scenario-wide max CN slots = the largest
        // `effective_controls().len()` across all tasks. Load that
        // many Union Pro v2 instances at startup so each per-task
        // multi-CN dispatch has independent slots to mutate. One slot
        // is enough for the common single-CN case; the cost scales
        // linearly with the deepest stack any task asks for.
        let max_flux_controls = s
            .tasks
            .iter()
            .map(|t| task_effective_controls(t).map(|v| v.len()).unwrap_or(0))
            .max()
            .unwrap_or(0);
        let flux_controlnets: Vec<flux::FluxControlNetLoad> = if max_flux_controls > 0 {
            use crate::pipelines::flux_controlnet;
            (0..max_flux_controls)
                .map(|_| flux::FluxControlNetLoad {
                    repo: "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0".to_string(),
                    file: "diffusion_pytorch_model.safetensors".to_string(),
                    cfg: flux_controlnet::Config::shakker_union_pro_v2(),
                    // Per-task: mutated via `set_controlnet_call_params`
                    // / `set_controlnet_conditioning` below.
                    scale: 1.0,
                    mode: None,
                    conditioning: None,
                    start: 0.0,
                    end: 1.0,
                })
                .collect()
        } else {
            Vec::new()
        };
        Some(
            flux::Pipeline::load(flux::LoadRequest {
                variant: fvar,
                repo: resolved_repo,
                device: device.clone(),
                loras: resolved_flux_loras,
                lora_scale,
                controlnets: flux_controlnets,
                // v0.13 phase 10: surface --quantize-t5 + GGUF quant
                // levels at scenario scope (load-time decisions).
                quantize_t5: s.quantize_t5,
                flux_quant_level: s.quant_level.clone(),
                t5_quant_level: s.t5_quant_level.clone(),
                // v0.14 phase 3c: enable the Redux encoder if ANY
                // task in the scenario uses `redux-images:`. Loaded
                // once at scenario startup, reused across tasks.
                redux: s.tasks.iter().any(|t| !t.redux_images.is_empty()),
            })
            .await?,
        )
    };

    // v0.36 phase 0: PixArt-Σ scenario pre-load. Mirrors the SD3 +
    // Flux pattern (load once at scenario start when the model
    // resolves as PixArt; scenarios with no PixArt tasks pay
    // nothing). LoRAs are merged at load time via the diffusers
    // PEFT path (v0.35 phase 4).
    let mut pixart_pipeline: Option<crate::pipelines::pixart::Pipeline> =
        if args.dry_run || !variant.is_pixart() {
            None
        } else {
            let resolved_repo = if model.contains('/') {
                model.clone()
            } else {
                crate::hf::resolve_alias(&model).to_string()
            };
            // Resolve scenario-level LoRAs once for the lifetime of
            // this pipeline. Per-task PixArt LoRA overrides land in
            // the v0.36 phase 2 / 3 variant work — phase 0 keeps the
            // contract tight: scenario-level loras: applies;
            // per-task overrides flow through the SD-style preflight
            // (extended below to recognise PixArt).
            let mut resolved_pixart_loras:
                Vec<crate::pipelines::lora::ResolvedLora> =
                Vec::with_capacity(loras.len());
            for spec in &loras {
                resolved_pixart_loras.push(spec.resolve().await?);
            }
            Some(
                crate::pipelines::pixart::Pipeline::load(
                    crate::pipelines::pixart::LoadRequest {
                        repo: resolved_repo,
                        device: device.clone(),
                        // v0.34 phase 3 mechanism — primed below
                        // (mixed-kind SDXL+PixArt share the VAE the
                        // moment the SDXL t2i path lazy-loads).
                        vae_cache: None,
                        loras: resolved_pixart_loras,
                        lora_scale,
                    },
                )
                .await?,
            )
        };
    // Cache key tracks the user's alias (e.g. "pixart") so subsequent
    // PixArt loads of the same alias hit. Populated only when the
    // pipeline actually loaded.
    let _pixart_pipeline_key: Option<String> = pixart_pipeline
        .as_ref()
        .map(|_| model.clone());

    // v0.37 phase 5: Stable Cascade scenario pre-load. Mirrors the
    // PixArt pattern above (load once at scenario start when
    // variant.is_cascade(); scenarios with no Cascade tasks pay
    // nothing). v0.38 phase 3 wires scenario-level LoRAs at load
    // time (mirrors the PixArt v0.35 phase 4 pattern). Per-task LoRA
    // overrides for Cascade are NOT yet supported — the preflight
    // skips Cascade and the dispatch arm runs against the load-time
    // merged tempfiles. Stage A VAE is its own type (not the SD-
    // family AutoEncoderKL), so the v0.34 phase 3 VAE-cache mechanism
    // doesn't apply — Cascade scenarios don't share VAE with other
    // pipelines.
    let mut cascade_pipeline: Option<crate::pipelines::cascade::Pipeline> =
        if args.dry_run || !variant.is_cascade() {
            None
        } else {
            let resolved_repo = if model.contains('/') {
                model.clone()
            } else {
                crate::hf::resolve_alias(&model).to_string()
            };
            let mut resolved_cascade_loras:
                Vec<crate::pipelines::lora::ResolvedLora> =
                Vec::with_capacity(loras.len());
            for spec in &loras {
                resolved_cascade_loras.push(spec.resolve().await?);
            }
            // v0.41 phase 3: if any task carries a control spec, resolve
            // the canny ControlNet from the model repo so the dispatch
            // arm can inject it. Stable Cascade ships the CN at
            // `controlnet/canny.safetensors`.
            let cascade_has_control = s.tasks.iter().any(|t| {
                task_effective_controls(t).map(|v| !v.is_empty()).unwrap_or(false)
            });
            let cascade_cn_weights = if cascade_has_control {
                Some(
                    crate::hf::download::get_first_of(&[(
                        &resolved_repo,
                        "controlnet/canny.safetensors",
                    )])
                    .await?,
                )
            } else {
                None
            };
            Some(
                crate::pipelines::cascade::Pipeline::load(
                    crate::pipelines::cascade::LoadRequest {
                        repo: resolved_repo,
                        device: device.clone(),
                        loras: resolved_cascade_loras,
                        lora_scale,
                        controlnet_weights: cascade_cn_weights,
                        // v0.42 phase 3: image variation isn't a
                        // scenario knob yet (scoped follow-up).
                        image_encoder_weights: None,
                    },
                )
                .await?,
            )
        };
    let _cascade_pipeline_key: Option<String> = cascade_pipeline
        .as_ref()
        .map(|_| model.clone());

    // -------- enhance prompts up front, deduped + parallelized --------
    // Each task's `pre_refine` string is enhancer-input; under sequential
    // per-task calls this fires N times serially. Pre-loop we dedupe to the
    // set of unique prompts and fire them concurrently (capped), so a 10-task
    // scenario with one slow enhancer goes from ~10 * latency to
    // ~ceil(unique / cap) * latency. Dry-run skips this entirely.
    let enhanced_cache: HashMap<String, String> = if args.dry_run {
        HashMap::new()
    } else {
        // v0.15 phase 7a: skip pre_refines for tasks that opted out
        // via `enhance: false` — those won't consult the cache.
        let pre_refines: Vec<String> = s
            .tasks
            .iter()
            .filter(|t| !matches!(t.enhance, Some(EnhanceCfg::Toggle(false))))
            // Map / multiperson tasks don't run scenario prompt enhancement (they
            // carry no scene/weather and source their own scene prompt).
            .filter(|t| !matches!(
                TaskKind::from_strs(t.task_type.as_deref(), s.task_type.as_deref()),
                Ok(TaskKind::Map | TaskKind::Multiperson)
            ))
            .map(|t| {
                let scene = scenes.get(t.scene.as_str()).copied().unwrap_or("");
                let weather = weathers.get(t.weather.as_str()).copied().unwrap_or("");
                join_parts(&[
                    &s.prompt_header,
                    scene,
                    weather,
                    &t.prompt,
                    &s.prompt_footer,
                ])
            })
            .collect();
        let unique: BTreeSet<String> = pre_refines.iter().cloned().collect();

        crate::ui::progress::println(&format!(
            "  {} enhancing {} unique prompt(s) (from {} task{}) via {}…",
            style("→").cyan().bold(),
            unique.len(),
            s.tasks.len(),
            if s.tasks.len() == 1 { "" } else { "s" },
            enhancer,
        ));

        // Soft cap on concurrent requests to be polite to upstream APIs.
        const MAX_CONCURRENT_ENHANCE: usize = 8;
        let sem = Arc::new(Semaphore::new(MAX_CONCURRENT_ENHANCE));
        let mut joinset: JoinSet<(String, Result<String>)> = JoinSet::new();
        for pre in unique {
            let enhancer_owned = enhancer.clone();
            let pre_owned = pre.clone();
            let sem = sem.clone();
            joinset.spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore not closed");
                let result = crate::prompt::enhance(&enhancer_owned, &pre_owned).await;
                (pre_owned, result)
            });
        }

        let mut cache: HashMap<String, String> = HashMap::with_capacity(joinset.len());
        while let Some(joined) = joinset.join_next().await {
            let (pre, result) = joined.context("enhancer task panicked")?;
            let enhanced = result
                .with_context(|| format!("enhancing prompt {:?}", trim_preview(&pre, 80)))?;
            cache.insert(pre, enhanced);
        }
        cache
    };

    // -------- v3 smart-zones depth pipeline (lazy) --------
    // Load once on first need across the run. Tasks share it so the
    // ~99 MB Depth-Anything-V2 weights are mmap'd just once.
    let mut smart_depth: Option<crate::pipelines::depth::DepthPipeline> = None;
    let mut smart_depth_attempted = false;
    let any_smart = s.smart_zones
        || s.tasks
            .iter()
            .any(|t| t.smart_zones.unwrap_or(false));

    // -------- v0.9 ControlNet cache (lazy, keyed by kind) --------
    // Same lazy pattern as smart_depth. The first task that needs a
    // given ControlKind triggers a download; subsequent tasks reuse
    // the loaded network.
    let mut controlnets: std::collections::HashMap<
        crate::pipelines::controlnet::ControlKind,
        crate::pipelines::controlnet::ControlNet,
    > = std::collections::HashMap::new();
    let cn_dtype = if matches!(device, candle_core::Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::F16
    };
    // v0.10: scenarios run a single model architecture for all
    // tasks, so the ControlNet variant is scenario-wide too.
    let cn_variant = crate::pipelines::controlnet::ControlNetVariant::detect(&model);

    // v0.19: --only filters tasks by name. Validate up-front so a
    // typo bails before the long batch starts. --limit caps the
    // run length post-filter.
    if !args.only.is_empty() {
        let scenario_names: std::collections::HashSet<&str> =
            s.tasks.iter().map(|t| t.name.as_str()).collect();
        for requested in &args.only {
            if !scenario_names.contains(requested.as_str()) {
                anyhow::bail!(
                    "--only {requested:?} not found in scenario {:?}. \
                     Available task names: {}",
                    args.file.display(),
                    s.tasks
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
        }
    }
    let only_set: Option<std::collections::HashSet<&str>> = if args.only.is_empty() {
        None
    } else {
        Some(args.only.iter().map(|s| s.as_str()).collect())
    };

    // -------- main loop --------
    let mut seed_offset: u64 = 0;
    let mut ran_count: u32 = 0;

    // v0.33 phase 2: per-task records for the optional
    // `--json-summary PATH` output. Tracking is cheap (a Vec push
    // per task); only written to disk when the flag is set.
    let mut task_records: Vec<TaskRunRecord> = Vec::with_capacity(s.tasks.len());
    let run_started = std::time::Instant::now();
    // v0.34 phase 2: catch-and-record failures per task. Set when
    // any task records `status: "failed"`. Used at end-of-loop to
    // exit non-zero so CI consumers see a failure exit code AND
    // get a full --json-summary listing every failure.
    let mut any_task_failed = false;

    // v0.34 phase 2: the generate body's async-block wrap returns
    // this enum so the outer match knows whether the body already
    // pushed a record (e.g. dry-run early-exit, --resume cache hit)
    // or whether the OK arm still owes a success record.
    enum GenerateOutcome {
        AlreadyRecorded,
        NeedSuccessRecord,
    }

    // v0.29 phase 3: lazy-loaded AnimateDiff pipelines + cache keys.
    // Initialised on the first animate task encountered. Key format:
    //   SD 1.5: "{alias}:{v3|lcm}:{joined_motion_loras}"
    //   SDXL:   "{alias}:{joined_motion_loras}"
    // Slot changes when key changes (toggling lcm, swapping motion
    // LoRAs, changing base alias). All-generate scenarios pay no
    // animate-pipeline cost.
    let mut animate_sd15: Option<crate::pipelines::animatediff::AnimateDiffPipeline> = None;
    let mut animate_sd15_key: Option<String> = None;
    let mut animate_sdxl: Option<crate::pipelines::animatediff::AnimateDiffSdxlPipeline> = None;
    let mut animate_sdxl_key: Option<String> = None;

    // v0.26 phase 12: scenario-level auto-LoRA discovery cache.
    // Keyed by preset name; the base_family is constant for the
    // whole scenario (scenarios don't override model per-task in
    // v0.26). Per the locked decision Q7 of RFC v0.26:
    // "Discovery cache key already includes the base_model.
    //  Smart-cache across the scenario: first task with
    //  `look: watercolor` fires discovery; tasks 2..100 with the
    //  same look hit the cache."
    let scenario_base_family =
        crate::preset::discovery::BaseFamily::from_model_arg(&model);
    let mut discovery_cache: std::collections::HashMap<
        String,
        Option<crate::preset::discovery::DiscoveredLora>,
    > = std::collections::HashMap::new();

    // v0.31 phase 3: kind-switch evictor state. Tracks the previous
    // task's kind so we can drop the OTHER kind's pipeline before
    // loading the new one. Closes the v0.29 carry where mixed-kind
    // scenarios held both t2i and animate pipelines simultaneously
    // (~10 GB SD 1.5 / much worse on SDXL).
    let mut last_task_kind: Option<TaskKind> = None;

    // v0.32 phase 2: VAE cache. SD-family pipelines hold the VAE
    // behind an `Arc<AutoEncoderKL>` so the scenario runner can keep
    // one shared instance alive across mixed-kind pipeline reloads
    // (t2i ↔ animate). The cache key is the resolved base alias —
    // any reload against the same base hits, reloads against a
    // different base miss. Saves the ~330 MB SDXL VAE rebuild on
    // every kind switch in a mixed-kind scenario.
    let mut vae_cache: Option<(
        String,
        std::sync::Arc<candle_transformers::models::stable_diffusion::vae::AutoEncoderKL>,
    )> = None;
    // Populate the cache from the eager pre-load if it happened
    // (all-generate SD-family scenarios pre-load at the function
    // top). The `pipeline.as_ref()` keeps the field type stable
    // even though the eager-load may have already taken ownership.
    if let Some(p) = pipeline.as_ref() {
        vae_cache = Some((model.clone(), std::sync::Arc::clone(&p.core().vae)));
        tracing::info!(
            target: "plakat",
            "v0.32 phase 2: VAE cache primed from eager pre-load (model={model})"
        );
    }
    // v0.36 phase 0: also prime the VAE cache from a freshly pre-
    // loaded PixArt pipeline. Mixed-kind SDXL+PixArt scenarios
    // (PixArt shares the SDXL VAE) reuse the same Arc across kind
    // switches the moment SDXL t2i lazy-loads later.
    if let Some(p) = pixart_pipeline.as_ref() {
        vae_cache = Some((model.clone(), std::sync::Arc::clone(&p.vae)));
        tracing::info!(
            target: "plakat",
            "v0.36 phase 0: VAE cache primed from PixArt pre-load (model={model})"
        );
    }
    // SD-family lazy reload predicate: same condition used at
    // pre-load time (line ~1635). When false (Flux / SD3 / PixArt /
    // Stable Cascade / dry-run), `pipeline` stays None for the whole
    // loop and never gets touched by the evictor.
    let sd_pipeline_applicable = !args.dry_run
        && !variant.is_flux()
        && !variant.is_sd3()
        && !variant.is_pixart()
        && !variant.is_cascade()
        && has_generate_tasks;

    // Live status-board events (no-op without a sink). `emitted_records` tracks how
    // many terminal `task_records` we've already turned into `TaskFinished` events;
    // we flush at each loop top (and after the loop) so the eight early-`continue`
    // paths — each of which pushes its own record — are all reported.
    emit(&events, ScenarioEvent::Started { total: s.tasks.len() });
    let mut emitted_records = 0usize;

    for (idx, task) in s.tasks.iter().enumerate() {
        // Report any task(s) that finished since the last iteration, then mark this
        // one running. Keyed by the record's name so it survives count drift.
        while emitted_records < task_records.len() {
            let r = &task_records[emitted_records];
            emit(
                &events,
                ScenarioEvent::TaskFinished {
                    index: emitted_records,
                    name: r.name.clone(),
                    status: r.status.clone(),
                },
            );
            emitted_records += 1;
        }
        emit(&events, ScenarioEvent::TaskStarted { index: idx, name: task.name.clone() });

        // v0.19: skip tasks excluded by --only / --limit. The
        // seed_offset advance still happens for skipped tasks so a
        // partial --only run yields the same seeds as the full
        // batch — important for reproducibility when iterating on
        // one task in isolation.
        if let Some(allowed) = &only_set {
            if !allowed.contains(task.name.as_str()) {
                seed_offset += count as u64;
                // v0.33 phase 2: record the skip for --json-summary.
                task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: task
                        .task_type
                        .as_deref()
                        .or(s.task_type.as_deref())
                        .unwrap_or("generate")
                        .to_string(),
                    status: "skipped".to_string(),
                    seed: None,
                    note: Some("--only filter excluded".to_string()),
                    error: None,
                });
                continue;
            }
        }
        if args.limit > 0 && ran_count >= args.limit {
            crate::ui::progress::println(&format!(
                "  {} reached --limit {} — skipping remaining tasks",
                style("(limit)").yellow(),
                args.limit,
            ));
            // v0.33 phase 2: record every unread remaining task as
            // skipped with the --limit reason so the summary has a
            // complete per-task accounting.
            for skipped in &s.tasks[idx..] {
                task_records.push(TaskRunRecord {
                    name: skipped.name.clone(),
                    kind: skipped
                        .task_type
                        .as_deref()
                        .or(s.task_type.as_deref())
                        .unwrap_or("generate")
                        .to_string(),
                    status: "skipped".to_string(),
                    seed: None,
                    note: Some(format!("--limit {} reached", args.limit)),
                    error: None,
                });
            }
            break;
        }
        ran_count += 1;

        // Tolerant lookups: generate/animate tasks are scene-validated up front, so
        // these always hit; a `type: map` task carries no scene/weather and falls
        // through to the map dispatch arm below (which ignores these).
        let scene_prompt = scenes.get(task.scene.as_str()).copied().unwrap_or("");
        let weather_prompt = weathers.get(task.weather.as_str()).copied().unwrap_or("");

        let pre_refine = join_parts(&[
            &s.prompt_header,
            scene_prompt,
            weather_prompt,
            &task.prompt,
            &s.prompt_footer,
        ]);

        // v0.29 phase 3: animate-task dispatch. Runs the AnimateDiff
        // pipeline + writes frames + format output, then advances
        // seed_offset and continues to the next task. Generate-path
        // tasks fall through past this block to the existing
        // pipeline.generate(...) logic below.
        let task_kind = TaskKind::from_strs(
            task.task_type.as_deref(),
            s.task_type.as_deref(),
        )
        .expect("validated up-front");

        // v0.31 phase 3: drop the opposite-kind cached pipeline on
        // switch. Closes the v0.29 carry — mixed-kind scenarios
        // (some `type: generate`, some `type: animatediff`) used to
        // hold BOTH pipelines for the whole run. Now we hold at
        // most one at a time; switching incurs a reload, but peak
        // memory drops by the size of whichever pipeline was just
        // dropped (typically the bigger half wins back ~5-10 GB).
        match evict_decision(last_task_kind, task_kind) {
            CacheEviction::None => {}
            CacheEviction::DropT2i => {
                if pipeline.is_some() {
                    tracing::info!(
                        target: "plakat",
                        "scenario kind switch: dropping cached t2i pipeline before animate task {:?}",
                        task.name,
                    );
                    pipeline = None;
                }
            }
            CacheEviction::DropAnimate => {
                if animate_sd15.is_some() || animate_sdxl.is_some() {
                    tracing::info!(
                        target: "plakat",
                        "scenario kind switch: dropping cached animate pipeline(s) before generate task {:?}",
                        task.name,
                    );
                    animate_sd15 = None;
                    animate_sd15_key = None;
                    animate_sdxl = None;
                    animate_sdxl_key = None;
                }
            }
            CacheEviction::DropAll => {
                if pipeline.is_some() || animate_sd15.is_some() || animate_sdxl.is_some() {
                    tracing::info!(
                        target: "plakat",
                        "scenario kind switch: dropping cached pipeline(s) before map task {:?}",
                        task.name,
                    );
                    pipeline = None;
                    animate_sd15 = None;
                    animate_sd15_key = None;
                    animate_sdxl = None;
                    animate_sdxl_key = None;
                }
            }
        }
        last_task_kind = Some(task_kind);

        if matches!(task_kind, TaskKind::Animate) {
            let task_seed = task.seed.unwrap_or(seed + seed_offset)
                & (u32::MAX as u64);
            let task_out = out_root.join(safe_name(&task.name));
            // v0.34 phase 2: wrap dispatch with catch-and-record.
            // Pre-v0.34 a failure here `?`-aborted the whole scenario.
            // Now we capture + continue + return non-zero at the end.
            let animate_result: Result<()> = async {
                let eff = effective_animate_config(&s, task)?;
                run_animate_task_inline(
                    &s,
                    task,
                    &eff,
                    &pre_refine,
                    idx + 1,
                    task_seed,
                    width,
                    height,
                    &task_out,
                    &args,
                    &device,
                    &model,
                    &mut animate_sd15,
                    &mut animate_sd15_key,
                    &mut animate_sdxl,
                    &mut animate_sdxl_key,
                    &mut vae_cache, // v0.34 phase 3: cross-kind VAE share
                )
                .await?;
                Ok(())
            }.await;
            match animate_result {
                Ok(()) => {
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: "animatediff".to_string(),
                        status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: None,
                    });
                }
                Err(e) => {
                    crate::ui::progress::println(&format!(
                        "  {} task {:?}: {}",
                        style("✗ failed").red().bold(),
                        task.name,
                        e
                    ));
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: "animatediff".to_string(),
                        status: "failed".to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: Some(e.to_string()),
                    });
                    any_task_failed = true;
                }
            }
            seed_offset += count as u64;
            continue;
        }

        // MAP-4: map-task dispatch. Sources the spec (load or LLM-parse) and renders
        // linework (deterministic, no GPU) or paints with SD, to `<out>/<name>/map.png`.
        if matches!(task_kind, TaskKind::Map) {
            let task_seed = task.seed.unwrap_or(seed + seed_offset);
            let task_out = out_root.join(safe_name(&task.name));
            let map_result: Result<()> = async {
                let cfg = effective_map_config(&s, task);
                crate::map::scenario_task::run_map_task(&cfg, task_seed, device.clone(), &task_out, args.dry_run)
                    .await
                    .map(|_| ())
            }
            .await;
            match map_result {
                Ok(()) => task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: "map".to_string(),
                    status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                    seed: Some(task_seed),
                    note: None,
                    error: None,
                }),
                Err(e) => {
                    crate::ui::progress::println(&format!(
                        "  {} task {:?}: {}",
                        style("✗ failed").red().bold(),
                        task.name,
                        e
                    ));
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: "map".to_string(),
                        status: "failed".to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: Some(e.to_string()),
                    });
                    any_task_failed = true;
                }
            }
            seed_offset += count as u64;
            continue;
        }

        // 4.3.0: fractal-task dispatch. Renders Track A / composition / animation (no GPU)
        // or an AI-painted fractal, to `<out>/<name>/fractal.{png,mp4}` (+ `.painted.png`).
        #[cfg(feature = "fractals")]
        if matches!(task_kind, TaskKind::Fractal) {
            let task_seed = task.seed.unwrap_or(seed + seed_offset);
            let task_out = out_root.join(safe_name(&task.name));
            let fractal_result: Result<()> = async {
                let cfg = task.fractal.clone().unwrap_or_default();
                crate::fractals::scenario_task::run_fractal_task(&cfg, task_seed, device.clone(), &task_out, args.dry_run)
                    .await
                    .map(|_| ())
            }
            .await;
            let rec_kind = "fractal".to_string();
            match fractal_result {
                Ok(()) => task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: rec_kind,
                    status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                    seed: Some(task_seed),
                    note: None,
                    error: None,
                }),
                Err(e) => {
                    crate::ui::progress::println(&format!(
                        "  {} task {:?}: {}",
                        style("✗ failed").red().bold(),
                        task.name,
                        e
                    ));
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: rec_kind,
                        status: "failed".to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: Some(e.to_string()),
                    });
                    any_task_failed = true;
                }
            }
            seed_offset += count as u64;
            continue;
        }

        // 6.1.0 A2: bookart-task dispatch. Renders an ornament / kit / manuscript to
        // `<out>/<name>/ornament.png` (+ `ornament.svg` on request) via the shared render core.
        if matches!(task_kind, TaskKind::Bookart) {
            let task_seed = task.seed.unwrap_or(seed + seed_offset);
            let task_out = out_root.join(safe_name(&task.name));
            let bookart_result: Result<()> = async {
                let cfg = task.bookart.clone().unwrap_or_default();
                crate::bookart::scenario_task::run_bookart_task(&cfg, task_seed, device.clone(), &task_out, args.dry_run).await
            }
            .await;
            let rec_kind = "bookart".to_string();
            match bookart_result {
                Ok(()) => task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: rec_kind,
                    status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                    seed: Some(task_seed),
                    note: None,
                    error: None,
                }),
                Err(e) => {
                    crate::ui::progress::println(&format!("  {} task {:?}: {}", style("✗ failed").red().bold(), task.name, e));
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: rec_kind,
                        status: "failed".to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: Some(e.to_string()),
                    });
                    any_task_failed = true;
                }
            }
            seed_offset += count as u64;
            continue;
        }

        // 6.3.0 B7: texture-task dispatch. Renders a seamless PBR material to `<out>/<name>/` via the
        // shared render core.
        if matches!(task_kind, TaskKind::Texture) {
            let task_seed = task.seed.unwrap_or(seed + seed_offset);
            let task_out = out_root.join(safe_name(&task.name));
            let texture_result: Result<()> = async {
                let cfg = task.texture.clone().unwrap_or_default();
                crate::texture::scenario_task::run_texture_task(&cfg, task_seed, device.clone(), &task_out, args.dry_run).await
            }
            .await;
            let rec_kind = "texture".to_string();
            match texture_result {
                Ok(()) => task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: rec_kind,
                    status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                    seed: Some(task_seed),
                    note: None,
                    error: None,
                }),
                Err(e) => {
                    crate::ui::progress::println(&format!("  {} task {:?}: {}", style("✗ failed").red().bold(), task.name, e));
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: rec_kind,
                        status: "failed".to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: Some(e.to_string()),
                    });
                    any_task_failed = true;
                }
            }
            seed_offset += count as u64;
            continue;
        }

        // 6.8.0 P4: comic-task dispatch. Renders a multi-panel comic page to `<out>/<name>/page.png`
        // via the shared render core.
        if matches!(task_kind, TaskKind::Comic) {
            let task_seed = task.seed.unwrap_or(seed + seed_offset);
            let task_out = out_root.join(safe_name(&task.name));
            let comic_result: Result<()> = async {
                let cfg = task.comic.clone().unwrap_or_default();
                crate::comic::scenario_task::run_comic_task(&cfg, task_seed, device.clone(), &task_out, args.dry_run).await
            }
            .await;
            let rec_kind = "comic".to_string();
            match comic_result {
                Ok(()) => task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: rec_kind,
                    status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                    seed: Some(task_seed),
                    note: None,
                    error: None,
                }),
                Err(e) => {
                    crate::ui::progress::println(&format!("  {} task {:?}: {}", style("✗ failed").red().bold(), task.name, e));
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: rec_kind,
                        status: "failed".to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: Some(e.to_string()),
                    });
                    any_task_failed = true;
                }
            }
            seed_offset += count as u64;
            continue;
        }

        // 1.14.0-A: multiperson-task dispatch. Resolves each placed persona to its
        // photos (from the top-level `personas` list) and dispatches the SAME
        // `pipelines::multiperson::run` the CLI uses — byte-for-byte parity.
        if matches!(task_kind, TaskKind::Multiperson) {
            let task_seed = task.seed.unwrap_or(seed + seed_offset);
            let task_out = out_root.join(safe_name(&task.name));
            let mp_result: Result<()> = async {
                let spec = task.multiperson.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("task {:?}: type `multiperson` needs a `multiperson:` block", task.name)
                })?;
                let mut spec = spec.clone();
                // A bare task `seed:` seeds the run when the block omits its own.
                if spec.seed.is_none() {
                    spec.seed = Some(task_seed);
                }
                let req = crate::pipelines::multiperson::scenario_task::build_request(
                    &spec,
                    |name| personas_map.get(name).and_then(|p| p.resolve_photos().ok()),
                    task_out.clone(),
                    device.clone(),
                    &model,
                    args.dry_run,
                )?;
                crate::pipelines::multiperson::run(req).await.map(|_| ())
            }
            .await;
            match mp_result {
                Ok(()) => task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: "multiperson".to_string(),
                    status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                    seed: Some(task_seed),
                    note: None,
                    error: None,
                }),
                Err(e) => {
                    crate::ui::progress::println(&format!(
                        "  {} task {:?}: {}",
                        style("✗ failed").red().bold(),
                        task.name,
                        e
                    ));
                    task_records.push(TaskRunRecord {
                        name: task.name.clone(),
                        kind: "multiperson".to_string(),
                        status: "failed".to_string(),
                        seed: Some(task_seed),
                        note: None,
                        error: Some(e.to_string()),
                    });
                    any_task_failed = true;
                }
            }
            seed_offset += count as u64;
            continue;
        }

        // v0.34 phase 2: lift task_seed before the generate body so
        // both arms of the result match below can reference it.
        let task_seed_for_record = task.seed.unwrap_or(seed + seed_offset);

        // v0.34 phase 2: wrap the entire generate body so per-task
        // failures are captured rather than aborting the scenario.
        // Body runs inside an `async {}` that returns
        // Result<GenerateOutcome>; failures push a failed record +
        // continue past the task. Early-exit paths (dry-run, resume
        // cache hit) push their own record and return AlreadyRecorded.
        let generate_result: anyhow::Result<GenerateOutcome> = async {

        // v0.31 phase 3: lazy reload of the SD-family t2i pipeline.
        // Fires when (a) the scenario is mixed-kind and we just
        // switched from animate back to generate (the evictor above
        // set `pipeline = None`), or (b) the pre-load was skipped
        // because `any_animate_tasks` was true (deferred to first
        // generate task). All-generate scenarios already loaded
        // up front and skip this branch.
        if sd_pipeline_applicable && pipeline.is_none() {
            // v0.32 phase 2: pull from VAE cache if the key matches.
            let cached = vae_cache_lookup(vae_cache.as_ref(), &model);
            if cached.is_some() {
                tracing::info!(
                    target: "plakat",
                    "v0.32 phase 2: VAE cache HIT on lazy SD reload (model={model})"
                );
            }
            let p = load_sd_pipeline_for_scenario(
                &model,
                &device,
                &loras,
                lora_scale,
                s.refiner,
                cached,
            )
            .await?;
            // Cache the VAE from the freshly loaded pipeline. Idempotent
            // if it was already cached (same Arc — cache holds one of
            // the two clones already alive).
            vae_cache = Some((model.clone(), std::sync::Arc::clone(&p.core().vae)));
            pipeline = Some(p);
        }

        crate::ui::progress::println(&format!(
            "\n{} [{}/{}] {} (scene={}, weather={})",
            style("▶").cyan().bold(),
            idx + 1,
            s.tasks.len(),
            style(&task.name).bold(),
            task.scene,
            task.weather,
        ));
        crate::ui::progress::println(&wrap_label("pre-enhance", &pre_refine));

        // -------- per-task style override --------
        // Trigger + negative_extras only — the pipeline pre-loaded its
        // LoRAs at scenario start and can't swap them per task without
        // a full reload. If the per-task style would have wanted
        // different LoRAs, warn loudly so the user knows the LoRAs
        // aren't being applied (trigger phrase alone won't fully
        // produce a style that needs a LoRA).
        let task_overrides_style = task.style_ref_catalog.is_some();
        let task_lora_header: String;
        let task_negative_base: String;
        if task_overrides_style {
            let session = style_session
                .as_mut()
                .expect("session created when any style is set");
            let prep = session
                .prepare(StylePrepRequest {
                    style_ref: task.style_ref_catalog.as_deref(),
                    style_override: None,
                    style_strength: s.style_strength.unwrap_or(1.0),
                    style_catalog: None, // session already locked the catalog in
                    model: &model,
                    // For per-task, the warning's "N user LoRAs overridden"
                    // count is meaningless (global LoRAs are NEVER overridden
                    // per-task) — suppress by passing false.
                    user_loras_nonempty: false,
                    device: &device,
                })
                .await?;

            log_style_prep(&prep, 0);

            // Warn when per-task style would have wanted LoRAs that differ
            // from the global ones currently loaded.
            let task_resolved = parse_resolved_loras(&prep)?;
            if !same_lora_set(&task_resolved, &loras) {
                crate::ui::progress::println(&format!(
                    "  {} per-task style '{}' wants {} LoRA(s); scenarios share \
                     one pipeline so only trigger + negative apply (global LoRAs \
                     stay loaded)",
                    style("⚠").yellow(),
                    prep.picked_style_id,
                    task_resolved.len()
                ));
            }

            // Per-task style applies against the SCENARIO BASE values, not
            // against the global-style-modified ones. Symmetrical with how
            // global style applies against user-authored values.
            task_lora_header = prepend_trigger(&prep.trigger, &s.lora_header);
            task_negative_base = combine_negative(&s.negative, &prep.negative_extras);
        } else {
            task_lora_header = effective_lora_header.clone();
            task_negative_base = effective_negative.clone();
        }

        // v0.15 phase 7a: `enhance: false` on the task skips the
        // enhancement step for this task only — pre_refine carries
        // through unmodified. `enhance: "provider"` must match the
        // scenario-level enhancer (the cache is built once with the
        // scenario provider, so per-task swap requires a wider
        // refactor — deferred).
        let task_skip_enhance =
            matches!(task.enhance, Some(EnhanceCfg::Toggle(false)));
        if let Some(EnhanceCfg::Provider(p)) = task.enhance.as_ref() {
            if p != &enhancer {
                anyhow::bail!(
                    "task {:?}: enhance provider {p:?} differs from scenario \
                     `enhancer: {enhancer}` — per-task provider swap not yet wired. \
                     Use the scenario enhancer or drop the override.",
                    task.name
                );
            }
        }
        let enhanced = if args.dry_run {
            format!("(dry-run; {enhancer} not called)")
        } else if task_skip_enhance {
            pre_refine.clone()
        } else {
            enhanced_cache
                .get(&pre_refine)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "task {:?}: enhanced prompt missing from cache (internal bug)",
                        task.name
                    )
                })?
        };
        crate::ui::progress::println(&wrap_label("enhanced", &enhanced));

        let mut final_prompt = join_parts(&[&task_lora_header, &enhanced, &s.lora_footer]);
        crate::ui::progress::println(&wrap_label("final", &final_prompt));

        if args.dry_run {
            // Show effective per-task values in dry-run so a user can see
            // what overrides are taking effect.
            let dry_count = task.count.unwrap_or(count);
            let dry_seed = task.seed.unwrap_or(seed + seed_offset);
            let dry_task_out = out_root.join(safe_name(&task.name));
            crate::ui::progress::println(&format!(
                "  {} would generate {} image(s) with seeds {}..{} → {}",
                style("(dry-run)").dim(),
                dry_count,
                dry_seed,
                dry_seed + dry_count as u64 - 1,
                dry_task_out.display(),
            ));
            if has_overrides(task) {
                crate::ui::progress::println(&format!(
                    "  {} overrides: {}",
                    style("(dry-run)").dim(),
                    describe_overrides(task)
                ));
            }
            // v0.25 phase 7: surface effective look/genre/offline so users
            // can validate that task / scenario fallbacks resolve to the
            // intended preset before paying the inference cost.
            let dry_look = task.look.as_ref().or(s.look.as_ref());
            let dry_genre = task.genre.as_ref().or(s.genre.as_ref());
            let dry_offline = task.offline.or(s.offline).unwrap_or(false);
            if dry_look.is_some() || dry_genre.is_some() {
                let parts = [
                    dry_look.map(|n| format!("look={n}")),
                    dry_genre.map(|n| format!("genre={n}")),
                    dry_offline.then(|| "offline=true".to_string()),
                ];
                let s_parts: Vec<String> = parts.into_iter().flatten().collect();
                crate::ui::progress::println(&format!(
                    "  {} presets: {}",
                    style("(dry-run)").dim(),
                    s_parts.join(", ")
                ));
            }
            if let Some(refs) = &task.personas {
                for r in refs {
                    let p = personas_map[r.name()];
                    let primary = p.primary_photo_path();
                    let exists = match &primary {
                        Some(path) if path.exists() => "ok",
                        Some(_) => "MISSING",
                        None => "(no photo set)",
                    };
                    let photo_desc = match &primary {
                        Some(path) => {
                            let multi = if !p.photos.is_empty() && p.photos.len() > 1 {
                                format!(" +{} more", p.photos.len() - 1)
                            } else {
                                String::new()
                            };
                            format!("{}{multi}", path.display())
                        }
                        None => String::from("(none)"),
                    };
                    let strength = p.face_strength.unwrap_or(0.8);
                    let bbox_str = match r.bbox() {
                        Some([x0, y0, x1, y1]) => {
                            format!(" bbox=[{x0:.2},{y0:.2},{x1:.2},{y1:.2}]")
                        }
                        None => String::new(),
                    };
                    crate::ui::progress::println(&format!(
                        "  {} would impose persona {:?} via portrait pipeline \
                         (photo {}, strength {:.2}{}, {})",
                        style("(dry-run)").dim(),
                        p.name,
                        photo_desc,
                        strength,
                        bbox_str,
                        exists,
                    ));
                }
            }
            if let Some(style_ref) = &task.style {
                let strength = task.style_strength.unwrap_or(0.6);
                let exists = if style_ref.exists() { "ok" } else { "MISSING" };
                crate::ui::progress::println(&format!(
                    "  {} would stylize each with REF {} (strength {:.2}, {})",
                    style("(dry-run)").dim(),
                    style_ref.display(),
                    strength,
                    exists,
                ));
            }
            if s.upscale.upscale {
                let target = if task.style.is_some() {
                    "styled"
                } else {
                    "original"
                };
                crate::ui::progress::println(&format!(
                    "  {} would upscale the {} image(s) at {:.2}× ({:?})",
                    style("(dry-run)").dim(),
                    target,
                    s.upscale.scale,
                    upscale_method,
                ));
            }
            // v0.33 phase 2: dry-run path bypasses the dispatch
            // wrapper at the bottom of the loop. Record the
            // dry-run outcome here so the summary has a complete
            // per-task accounting (with the actual planned seed,
            // not the bare seed_offset).
            let dry_seed = task.seed.unwrap_or(seed + seed_offset);
            task_records.push(TaskRunRecord {
                name: task.name.clone(),
                kind: task
                    .task_type
                    .as_deref()
                    .or(s.task_type.as_deref())
                    .unwrap_or("generate")
                    .to_string(),
                status: "dry-run".to_string(),
                seed: Some(dry_seed),
                note: None,
                error: None,
            });
            // v0.34 phase 2: was `continue;` outside an async-block.
            // Inside the wrap, return AlreadyRecorded so the outer
            // match skips the success-push. Outer seed_offset advance
            // runs unconditionally after the match.
            return Ok(GenerateOutcome::AlreadyRecorded);
        }

        // -------- effective per-task values (override-or-global) --------
        // Resolution: per-task size/aspect override the global pair.
        let task_size = match task.size.as_deref() {
            Some(s) => Some(s.parse::<Size>().with_context(|| format!("task size {s:?}"))?),
            None => None,
        };
        let (mut eff_w, mut eff_h) = if task_size.is_some() || task.aspect.is_some() {
            crate::imaging::sizes::resolve(
                task_size,
                task.aspect.as_deref(),
                base,
            )?
        } else {
            (width, height)
        };

        // v0.13 phase 11: outpaint pre-processing. When `outpaint:` is
        // set, the canvas + mask are synthesised from `init-image:`
        // and the resolved per-side padding; the working resolution is
        // overridden to the expanded canvas size (the user's `size:`
        // is ignored — outpaint defines its own output dims). The
        // tempdir holds canvas.png + mask.png alive across the
        // generate dispatch.
        let mut eff_init_image: Option<PathBuf> = task.init_image.clone();
        let mut eff_mask: Option<PathBuf> = task.mask.clone();
        let _outpaint_tmp: Option<tempfile::TempDir> = if let Some(ospec) = task.outpaint {
            let init = task
                .init_image
                .as_ref()
                .expect("validated above (outpaint requires init-image)");
            let (l_req, r_req, t_req, b_req) = ospec.resolved()?;
            // Snap padding to the model's VAE / patch constraint —
            // mirrors `cli::outpaint::run`. Flux uses 16, SD uses 8.
            let snap: u32 = if model.to_lowercase().contains("flux") {
                16
            } else {
                8
            };
            let snap_up = |n: u32| if n == 0 { 0 } else { ((n + snap - 1) / snap) * snap };
            let (l, r, t, b) =
                (snap_up(l_req), snap_up(r_req), snap_up(t_req), snap_up(b_req));
            let input_img = image::open(init).with_context(|| {
                format!(
                    "task {:?}: opening outpaint init-image {}",
                    task.name,
                    init.display()
                )
            })?;
            let (in_w, in_h) = {
                use image::GenericImageView;
                input_img.dimensions()
            };
            if in_w % snap != 0 || in_h % snap != 0 {
                bail!(
                    "task {:?}: outpaint init-image is {in_w}x{in_h}, not divisible by {snap} \
                     (the model's VAE / patch constraint). Resize the input to a multiple of \
                     {snap} before outpainting.",
                    task.name
                );
            }
            let in_rgb = input_img.to_rgb8();
            let new_w = in_w + l + r;
            let new_h = in_h + t + b;
            let canvas =
                crate::cli::outpaint::build_replicate_canvas(&in_rgb, l, t, new_w, new_h);
            let mask = crate::cli::outpaint::build_outpaint_mask(in_w, in_h, l, t, new_w, new_h);
            let tmp = tempfile::Builder::new()
                .prefix("plakat-scenario-outpaint-")
                .tempdir()?;
            let cpath = tmp.path().join("canvas.png");
            let mpath = tmp.path().join("mask.png");
            canvas.save(&cpath).with_context(|| {
                format!("task {:?}: writing outpaint canvas", task.name)
            })?;
            mask.save(&mpath).with_context(|| {
                format!("task {:?}: writing outpaint mask", task.name)
            })?;
            eff_w = new_w;
            eff_h = new_h;
            eff_init_image = Some(cpath);
            eff_mask = Some(mpath);
            crate::ui::progress::println(&format!(
                "  outpaint: {in_w}x{in_h} → {new_w}x{new_h} (left={l} right={r} top={t} bottom={b}, snap={snap})"
            ));
            Some(tmp)
        } else {
            None
        };

        let eff_count = task.count.unwrap_or(count);
        let mut eff_steps = task.steps.unwrap_or(steps);
        let mut eff_guidance = task.guidance.unwrap_or(guidance);
        let mut eff_negative = task
            .negative
            .clone()
            .unwrap_or_else(|| task_negative_base.clone());
        let mut eff_scheduler: SchedulerKind = match task.scheduler.as_deref() {
            Some(s) => s.parse().with_context(|| format!("task scheduler {s:?}"))?,
            None => scheduler,
        };
        let eff_refine = task.refine.or(s.refine);
        let eff_refine_strength = task.refine_strength.unwrap_or(refine_strength);
        let eff_refiner_frac = task.refiner_frac.unwrap_or(s.refiner_frac.unwrap_or(0.8));

        // v0.15 phase 7a: scenario↔task sync resolutions. Task wins
        // when set; scenario provides the default; explicit `false`
        // on tiled / enhance forces off even if scenario has it on.
        let eff_concept_image: Option<PathBuf> = task
            .concept_image
            .clone()
            .or_else(|| s.concept_image.clone());
        // v0.18 Kontext phase 4: task override → scenario fallback →
        // false default. Only honoured at Pipeline::generate when the
        // resolved model is Kontext.
        let eff_kontext_bucket: bool =
            task.kontext_bucket.or(s.kontext_bucket).unwrap_or(false);
        let eff_tiled: Option<crate::pipelines::tiled::TiledConfig> = match &task.tiled {
            Some(TaskTiledCfg::Toggle(false)) => None,
            Some(TaskTiledCfg::Toggle(true)) => {
                // `tiled: true` without an explicit block at task scope:
                // inherit the scenario-level config, or fall back to
                // the same defaults the CLI uses (1024 / 768).
                s.tiled.map(Into::into).or(Some(
                    crate::pipelines::tiled::TiledConfig {
                        tile_size: default_tile_size(),
                        stride: default_tile_stride(),
                    },
                ))
            }
            Some(TaskTiledCfg::Override(cfg)) => Some((*cfg).into()),
            None => s.tiled.map(Into::into),
        };
        // `task.enhance` is checked directly at the per-task enhance
        // lookup below — `Toggle(false)` skips the cache, `Provider(p)`
        // must equal scenario.enhancer (validated below).
        //
        // `task.fast` is validated at scenario load (must equal
        // scenario.fast — true per-task swap is v0.15 phase 7b runtime
        // LoRA), so no per-task resolution is needed here.

        // v0.25 phase 7: --look / --genre presets in scenarios. Task
        // wins; scenario provides the default. Auto-LoRA discovery is
        // NOT wired here (the scenario's two-stage LoRA pipeline
        // makes integration non-trivial); the prompt prefix/suffix +
        // sampler hints apply, and users who want a specific LoRA
        // pass `loras:` at the scenario or task level.
        //
        // Override-only-if-user-didn't-pass: if scenario OR task
        // explicitly sets steps / guidance / scheduler, that wins
        // over the preset's recommendation.
        let eff_look = task.look.clone().or_else(|| s.look.clone());
        let eff_genre = task.genre.clone().or_else(|| s.genre.clone());
        let eff_offline = task.offline.or(s.offline).unwrap_or(false);
        // v0.26 phase 12: per-task discovered LoRA spec strings.
        // Filled inside the apply block below when a preset has a
        // lora_query AND no user LoRAs are set (scenario or task
        // level). Appended onto task.loras when handing off to
        // apply_task_loras_for_dispatch.
        let mut discovered_lora_strings: Vec<String> = Vec::new();
        if eff_look.is_some() || eff_genre.is_some() {
            use crate::preset::{GenerationParams, apply_presets};
            use std::str::FromStr;

            let user_set_steps = task.steps.is_some() || s.steps.is_some();
            let user_set_guidance = task.guidance.is_some() || s.guidance.is_some();
            let user_set_scheduler =
                task.scheduler.is_some() || s.scheduler.is_some();

            let mut params = GenerationParams {
                prompt: final_prompt.clone(),
                negative: eff_negative.clone(),
                steps: user_set_steps.then_some(eff_steps),
                guidance: user_set_guidance.then_some(eff_guidance),
                scheduler: user_set_scheduler.then(String::new),
            };
            let (look_spec, genre_spec) = apply_presets(
                eff_look.as_deref(),
                eff_genre.as_deref(),
                &mut params,
            )
            .with_context(|| format!("task {:?}: look/genre apply", task.name))?;

            final_prompt = params.prompt;
            eff_negative = params.negative;
            if let Some(s_steps) = params.steps {
                eff_steps = s_steps;
            }
            if let Some(g) = params.guidance {
                eff_guidance = g;
            }
            if let Some(sched) = params.scheduler.filter(|s| !s.is_empty()) {
                eff_scheduler =
                    SchedulerKind::from_str(&sched).unwrap_or(eff_scheduler);
            }

            if let Some(l) = &look_spec {
                crate::ui::progress::println(&format!(
                    "  look '{}': prompt/negative composed (scenario)",
                    l.name
                ));
            }
            if let Some(g) = &genre_spec {
                crate::ui::progress::println(&format!(
                    "  genre '{}': prompt/negative composed (scenario)",
                    g.name
                ));
            }

            // v0.26 phase 12: scenario auto-LoRA discovery.
            // Gated on:
            //   1. No scenario-level loras (parsed earlier into
            //      `loras: Vec<LoraSpec>` from `s.loras`).
            //   2. No task-level loras (`task.loras` empty).
            //   3. The look or genre carries a `lora_query`.
            // Looks first (more specific), then genre as fallback.
            // Per Q7 the discovery cache is keyed by preset name
            // since the scenario_base_family is constant — 100
            // tasks sharing `look: watercolor` fire one network call.
            //
            // Discovery is async — uses the scenario `run()` async
            // context directly (no block_in_place needed).
            if loras.is_empty() && task.loras.is_empty() && !args.dry_run {
                use crate::pipelines::lora::{CivitaiIdKind, LoraSource};
                let query_source = look_spec
                    .as_ref()
                    .filter(|p| p.lora_query.is_some())
                    .or(genre_spec
                        .as_ref()
                        .filter(|p| p.lora_query.is_some()));
                if let Some(preset) = query_source {
                    let query = preset.lora_query.as_ref().expect("filter Some");
                    // Cache lookup / fill.
                    let discovered: Option<
                        crate::preset::discovery::DiscoveredLora,
                    > = match discovery_cache.get(&preset.name) {
                        Some(opt) => opt.clone(),
                        None => {
                            let opts = crate::preset::discovery::DiscoveryOptions::with_defaults(
                                eff_offline,
                                scenario_base_family,
                            );
                            let result =
                                crate::preset::discovery::discover_lora(
                                    query,
                                    &preset.name,
                                    &opts,
                                )
                                .await;
                            let value = match result {
                                Ok(v) => v,
                                Err(e) => {
                                    tracing::warn!(
                                        target: "plakat",
                                        "scenario discovery failed for {}: {e:#}",
                                        preset.name
                                    );
                                    None
                                }
                            };
                            discovery_cache
                                .insert(preset.name.clone(), value.clone());
                            value
                        }
                    };
                    if let Some(d) = discovered {
                        // Serialize the LoraSpec back to a string
                        // that LoraSpec::from_str will parse — same
                        // grammar as the user-supplied entries in
                        // task.loras: Vec<String>.
                        let spec_str = match &d.spec.source {
                            LoraSource::Civitai { id_kind, .. } => {
                                let id_part = match id_kind {
                                    CivitaiIdKind::Version(n) => {
                                        format!("civitai-version:{n}")
                                    }
                                    CivitaiIdKind::Model(n) => {
                                        format!("civitai:{n}")
                                    }
                                };
                                format!("{id_part}:{:.3}", d.spec.scale)
                            }
                            LoraSource::Hub { repo, .. } => {
                                format!("{repo}:{:.3}", d.spec.scale)
                            }
                            LoraSource::Local(p) => {
                                format!("{}:{:.3}", p.display(), d.spec.scale)
                            }
                        };
                        crate::ui::progress::println(&format!(
                            "  discovered LoRA '{}' (scale={}) for preset '{}'",
                            d.model_name,
                            d.spec.scale,
                            preset.name,
                        ));
                        if !d.trigger_words.is_empty() {
                            let trigger = d.trigger_words.join(", ");
                            final_prompt = crate::style::prepend_trigger(
                                &trigger,
                                &final_prompt,
                            );
                            crate::ui::progress::println(&format!(
                                "  trigger words prepended: {trigger}"
                            ));
                        }
                        discovered_lora_strings.push(spec_str);
                    }
                }
            }
        }

        // Seed: per-task override picks an absolute seed; global path
        // advances seed_offset to keep later tasks reproducible.
        let task_seed = task.seed.unwrap_or(seed + seed_offset);

        let task_out = out_root.join(safe_name(&task.name));

        // v0.17 phase 5: --resume skip. If every expected output
        // PNG for this task already exists on disk, skip the
        // task entirely (no model dispatch, no enhancer call).
        // Backbones write a few prefixes: `plakat-<seed>.png`
        // (SD t2i), `plakat-flux-<seed>.png` (Flux), `plakat-sd3-
        // <seed>.png` (SD3 t2i), `plakat-img2img-<seed>.png` /
        // `plakat-inpaint-<seed>.png` (img2img). We accept any of
        // those prefixes when probing — gives correct skip
        // behaviour for mixed-task scenarios without per-backbone
        // dispatch.
        //
        // `seed_offset` MUST advance by the global `count`, NOT
        // `eff_count`, to match the non-skip path (line 2739) —
        // the global-seed scheme means re-running with the same
        // scenario file produces the same per-task seeds whether
        // any tasks were skipped or not.
        if args.resume && task_outputs_all_present(&task_out, task_seed, eff_count) {
            crate::ui::progress::println(&format!(
                "  ↺ {}: all {} output(s) already on disk — skipping",
                console::style(&task.name).cyan(),
                eff_count,
            ));
            // v0.33 phase 2: --resume cache hit counts as skipped.
            task_records.push(TaskRunRecord {
                name: task.name.clone(),
                kind: "generate".to_string(),
                status: "skipped".to_string(),
                seed: Some(task_seed),
                note: Some("--resume: outputs already present".to_string()),
                error: None,
            });
            // v0.34 phase 2: see note above for the dry-run path.
            return Ok(GenerateOutcome::AlreadyRecorded);
        }

        // Classify the persona configuration for this task.
        //   None        — no personas; regular t2i / flux dispatch.
        //   Single(p)   — Phase-1 form: one bare-name persona, whole image.
        //   Multi(...)  — Phase-2 form: one or more {name, bbox} personas
        //                 routed through region-masked compositing.
        enum TaskPersonas<'a> {
            None,
            Single(&'a PersonaDef),
            Multi(Vec<(&'a PersonaDef, [f32; 4])>),
        }
        let task_persona_mode: TaskPersonas = match task.personas.as_deref() {
            None => TaskPersonas::None,
            Some(refs) if refs.is_empty() => TaskPersonas::None,
            Some(refs) => {
                // Validation guarantees: either all-bare (and len == 1), or
                // all-bbox (len >= 1).
                match refs.first().unwrap() {
                    PersonaRef::Bare(name) => {
                        TaskPersonas::Single(personas_map[name.as_str()])
                    }
                    PersonaRef::Bbox(_) => {
                        let mut v = Vec::with_capacity(refs.len());
                        for r in refs {
                            match r {
                                PersonaRef::Bbox(b) => {
                                    v.push((personas_map[b.name.as_str()], b.bbox));
                                }
                                PersonaRef::Bare(_) => {
                                    unreachable!("form-mixing rejected at load")
                                }
                            }
                        }
                        TaskPersonas::Multi(v)
                    }
                }
            }
        };

        // Filename prefix used by downstream style / upscale passes. Both
        // persona forms output via the portrait pipeline.
        let prefix = match task_persona_mode {
            TaskPersonas::None => {
                if variant.is_flux() { "plakat-flux" } else { "plakat" }
            }
            TaskPersonas::Single(_) | TaskPersonas::Multi(_) => "plakat-portrait",
        };

        // Build a per-persona effective negative (persona-negative prepended
        // to the task's effective negative). Returns the negative string and
        // the persona's effective face_strength.
        let persona_request_for = |persona: &PersonaDef| -> (String, f32) {
            let combined = match persona.negative.as_deref() {
                Some(p_neg) if !p_neg.trim().is_empty() => {
                    if eff_negative.trim().is_empty() {
                        p_neg.to_string()
                    } else {
                        format!("{p_neg}, {eff_negative}")
                    }
                }
                _ => eff_negative.clone(),
            };
            (combined, persona.face_strength.unwrap_or(0.8))
        };

        // -------- v0.9 / v0.13 phase 11 per-task ControlNet plumbing --
        // Resolve the task's effective control list (either the
        // singular `control:` or the multi `controls:` Vec). For each
        // entry, lazy-load the CN weights into the scenario-wide
        // HashMap and prepare its conditioning tensor at the task's
        // working resolution. The result is a parallel Vec of (kind,
        // conditioning_tensor, strength, start, end) tuples consumed
        // by `make_control_reqs` below.
        let task_controls = task_effective_controls(task)?;
        let mut task_cn_resolved: Vec<(
            crate::pipelines::controlnet::ControlKind,
            candle_core::Tensor,
            f32,
            f32,
            f32,
        )> = Vec::with_capacity(task_controls.len());
        for spec in &task_controls {
            let kind: crate::pipelines::controlnet::ControlKind =
                spec.kind.parse().with_context(|| {
                    format!("task {:?}: parsing control.kind {:?}", task.name, spec.kind)
                })?;
            if !controlnets.contains_key(&kind) {
                let net = crate::pipelines::controlnet::ControlNet::load(
                    device.clone(),
                    cn_dtype,
                    kind,
                    cn_variant,
                )
                .await
                .with_context(|| {
                    format!(
                        "task {:?}: loading ControlNet for {:?} ({:?})",
                        task.name, kind, cn_variant,
                    )
                })?;
                controlnets.insert(kind, net);
            }
            let cond = match (spec.image.as_ref(), spec.auto_from.as_ref()) {
                (Some(path), None) => {
                    crate::pipelines::controlnet::prepare_conditioning(
                        path, eff_w, eff_h, &device, cn_dtype,
                    )
                    .with_context(|| {
                        format!(
                            "task {:?}: preparing ControlNet conditioning ({:?})",
                            task.name, kind
                        )
                    })?
                }
                (None, Some(path)) => {
                    crate::pipelines::controlnet_annotator::annotate(
                        kind, path, eff_w, eff_h, &device, cn_dtype,
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "task {:?}: running auto-annotator on {} ({:?})",
                            task.name,
                            path.display(),
                            kind
                        )
                    })?
                }
                (Some(_), Some(_)) => bail!(
                    "task {:?}: control entry must set either `image:` or `auto-from:`, not both",
                    task.name
                ),
                (None, None) => bail!(
                    "task {:?}: control entry requires either `image:` or `auto-from:`",
                    task.name
                ),
            };
            task_cn_resolved.push((
                kind,
                cond,
                spec.strength.unwrap_or(1.0),
                spec.start.unwrap_or(0.0),
                spec.end.unwrap_or(1.0),
            ));
        }
        let make_control_reqs = || -> Vec<crate::pipelines::controlnet::ControlRequest> {
            task_cn_resolved
                .iter()
                .map(|(kind, cond, strength, start, end)| {
                    crate::pipelines::controlnet::ControlRequest {
                        net: controlnets.get(kind).expect("loaded above"),
                        conditioning: cond.clone(),
                        strength: *strength,
                        start: *start,
                        end: *end,
                    }
                })
                .collect()
        };

        match &task_persona_mode {
            // -------- single persona, whole image --------
            TaskPersonas::Single(persona) => {
                let pp = portrait_pipeline
                    .as_ref()
                    .expect("portrait pipeline preloaded when any task uses personas");
                let (combined_negative, face_strength) = persona_request_for(persona);
                let photos = persona.resolve_photos()?;
                let photo_desc = match (photos.first(), photos.len()) {
                    (Some(first), 1) => first.path.display().to_string(),
                    (Some(first), n) => {
                        format!("{} (+{} more, weighted merge)", first.path.display(), n - 1)
                    }
                    _ => String::from("(none)"),
                };
                crate::ui::progress::println(&format!(
                    "  {} persona {} (photo {}, face-strength {:.2})",
                    style("portrait").magenta().bold(),
                    style(&persona.name).bold(),
                    photo_desc,
                    face_strength,
                ));
                pp.generate(&portrait::GenRequest {
                    prompt: final_prompt.clone(),
                    negative: combined_negative,
                    photos,
                    width: eff_w,
                    height: eff_h,
                    count: eff_count,
                    steps: eff_steps,
                    guidance: eff_guidance,
                    seed: Some(task_seed),
                    out_dir: task_out.clone(),
                    scheduler: eff_scheduler,
                    refine: eff_refine,
                    refine_strength: eff_refine_strength,
                    face_strength,
                    face_bbox: persona.face_bbox,
                    face_landmarks: persona.face_landmarks,
                }, &make_control_reqs())?;
            }

            // -------- multi-persona, region-masked compositing --------
            TaskPersonas::Multi(passes) => {
                let pp = portrait_pipeline
                    .as_ref()
                    .expect("portrait pipeline preloaded when any task uses personas");
                std::fs::create_dir_all(&task_out)
                    .with_context(|| format!("creating output dir {}", task_out.display()))?;

                let names_log: Vec<String> = passes
                    .iter()
                    .map(|(p, b)| {
                        format!(
                            "{}@[{:.2},{:.2},{:.2},{:.2}]",
                            p.name, b[0], b[1], b[2], b[3]
                        )
                    })
                    .collect();
                crate::ui::progress::println(&format!(
                    "  {} composite {} persona(s): {}",
                    style("portrait").magenta().bold(),
                    passes.len(),
                    names_log.join(", "),
                ));

                for img_idx in 0..eff_count {
                    let img_seed = (task_seed + img_idx as u64) & (u32::MAX as u64);

                    // Base request: text-only (no photo), same prompt/negative
                    // as the scenario / task.
                    let base_req = portrait::GenRequest {
                        prompt: final_prompt.clone(),
                        negative: eff_negative.clone(),
                        photos: Vec::new(),
                        width: eff_w,
                        height: eff_h,
                        count: 1,
                        steps: eff_steps,
                        guidance: eff_guidance,
                        seed: Some(img_seed),
                        out_dir: task_out.clone(),
                        scheduler: eff_scheduler,
                        refine: None,
                        refine_strength: 0.3,
                        face_strength: 0.0,
                        face_bbox: None,
                        face_landmarks: None,
                    };

                    // Multi-persona: control applies to the base layout pass
                    // only. Each per-persona inpaint pass below skips control
                    // (the persona reference itself drives the local region).
                    let mut latents =
                        pp.generate_latents_one(&base_req, img_seed, &make_control_reqs())?;

                    // Chain one inpaint pass per persona. Each pass uses
                    // a per-persona seed offset so re-running with the same
                    // task_seed yields the same composite.
                    let latent_w = (eff_w as usize) / 8;
                    let latent_h = (eff_h as usize) / 8;
                    for (pass_idx, (persona, bbox)) in passes.iter().enumerate() {
                        let (combined_negative, face_strength) = persona_request_for(persona);
                        let mask = build_persona_mask(
                            *bbox,
                            latent_w,
                            latent_h,
                            &device,
                            pp.latent_dtype(),
                        )?;
                        let pass_photos = persona.resolve_photos()?;
                        let pass_req = portrait::GenRequest {
                            prompt: final_prompt.clone(),
                            negative: combined_negative,
                            photos: pass_photos,
                            width: eff_w,
                            height: eff_h,
                            count: 1,
                            steps: eff_steps,
                            guidance: eff_guidance,
                            seed: Some(img_seed),
                            out_dir: task_out.clone(),
                            scheduler: eff_scheduler,
                            refine: None,
                            refine_strength: 0.3,
                            face_strength,
                            face_bbox: persona.face_bbox,
                    face_landmarks: persona.face_landmarks,
                        };
                        let pass_seed = img_seed
                            .wrapping_add(1)
                            .wrapping_add(pass_idx as u64)
                            & (u32::MAX as u64);
                        latents = pp.inpaint_latents_one(
                            &latents,
                            &mask,
                            &pass_req,
                            pass_seed,
                            &[],
                            // Scenario inpaint uses the portrait pipeline's
                            // SD UNet (RePaint blending) — not SDXL-Inpaint.
                            None,
                        )?;
                    }

                    let out_path = task_out.join(format!("{prefix}-{img_seed}.png"));
                    pp.save_image(&latents, &out_path)?;
                }
            }

            TaskPersonas::None => {
            // -------- regular t2i / flux dispatch (unchanged behaviour) --------
            // v0.47: the SD/Flux scenario arm now embeds per-image recipe
            // metadata (sidecar + `parameters` tEXt chunk) like the Cascade/PixArt
            // arms — so every proof, including artefact-composited ones, self-
            // documents and surfaces in `plakat gallery`. (Closes the v0.17
            // `metadata: None` deferral.)
            let mut t2i_meta = crate::imaging::metadata::GenerationMetadata::new(
                final_prompt.clone(),
                model.clone(),
                task_seed,
                eff_steps,
                eff_guidance,
                format!("{:?}", eff_scheduler).to_lowercase(),
                eff_w,
                eff_h,
            );
            t2i_meta.negative = eff_negative.clone();
            let t2i_lora_entries: Vec<crate::imaging::metadata::LoraEntry> =
                loras.iter().map(|s| s.to_entry()).collect();
            if !t2i_lora_entries.is_empty() {
                t2i_meta.with_lora_stack(t2i_lora_entries);
                t2i_meta.lora_scale = Some(lora_scale);
            }
            if let Ok(specs) = task_effective_controls(task) {
                if let Some(spec) = specs.first() {
                    t2i_meta.with_control_stack(vec![crate::imaging::metadata::ControlEntry {
                        kind: spec.kind.clone(),
                        image: spec.image.as_ref().map(|p| p.display().to_string()),
                        from: spec.auto_from.as_ref().map(|p| p.display().to_string()),
                        video: None,
                        strength: spec.strength.unwrap_or(1.0),
                        start: spec.start.unwrap_or(0.0),
                        end: spec.end.unwrap_or(1.0),
                    }]);
                }
            }
            let eff_regions: Vec<crate::pipelines::tiled::RegionSpec> = task
                .regions
                .iter()
                .map(|r| crate::pipelines::tiled::RegionSpec::parse(r))
                .collect::<Result<_>>()?;
            let gen_req = GenRequest {
                prompt: final_prompt.clone(),
                negative: eff_negative.clone(),
                width: eff_w,
                height: eff_h,
                count: eff_count,
                steps: eff_steps,
                guidance: eff_guidance,
                seed: Some(task_seed),
                out_dir: task_out.clone(),
                scheduler: eff_scheduler,
                refine: eff_refine,
                refine_strength: eff_refine_strength,
                refiner_frac: if s.refiner { Some(eff_refiner_frac) } else { None },
                // v0.16 phase 5: scenarios don't surface per-task
                // clip-skip yet. Default `1` = bit-identical to
                // pre-phase-5 behaviour.
                clip_skip: 1,
                // v0.47: per-image recipe metadata (built above) → the t2i
                // save writes the `parameters` tEXt chunk + JSON sidecar, so
                // scenario proofs are self-documenting and gallery-visible.
                metadata: Some(t2i_meta),
                // v0.17 phase D: scenarios don't surface live-
                // preview cadence — batch runs typically don't
                // need the per-step PNG churn. None = disabled.
                preview_every: None,
                preview_size: None,
                // v0.19: scenarios don't surface --format yet —
                // default to PNG (the v0.17 A1111-compat path).
                // Per-task webp output lands in a follow-up once
                // the scenario schema is extended.
                output_format: crate::imaging::io::OutputFormat::Png,
            };

            // Per-task runtime LoRA. Applied BEFORE the per-task
            // generate, cleared AFTER (so the next task starts from
            // the scenario-merged baseline).
            //
            // Backbone routing:
            //   * Flux (BF16 / GGUF / NF4) → flux_pipeline backbone
            //   * SD3 / SD3.5              → sd3_pipeline
            //   * SD-family                → bails (SD UNet runtime
            //     LoRA support not yet wired)
            //
            // Empty task.loras = no-op for every backbone.
            // v0.26 phase 12: combine task.loras with any
            // auto-discovered LoRAs from the scenario look/genre.
            // When both are empty, skip the dispatch (no-op for
            // every backbone).
            let effective_task_loras: Vec<String> = if discovered_lora_strings.is_empty() {
                task.loras.clone()
            } else {
                let mut combined = task.loras.clone();
                combined.extend(discovered_lora_strings.iter().cloned());
                combined
            };
            let task_lora_applied = if !effective_task_loras.is_empty() {
                apply_task_loras_for_dispatch(
                    task,
                    &effective_task_loras,
                    task.lora_scale.unwrap_or(lora_scale),
                    flux_pipeline.as_mut(),
                    &pipeline,
                    sd3_pipeline.as_mut(),
                    &model,
                    &device,
                )
                .await?
            } else {
                false
            };

            // v0.37 phase 5: Stable Cascade dispatch arm. Routes per-
            // task Cascade tasks through the scenario-cached
            // `cascade::Pipeline`. Single --steps splits 2/3 → Stage C
            // and 1/3 → Stage B, matching t2i::run's CLI dispatch
            // split. Cascade has no scenario-level LoRA wiring in
            // v0.37 (deferred to v0.38). Falls through to PixArt →
            // SD3 → SD/Flux for non-Cascade tasks.
            if let Some(cp) = cascade_pipeline.as_mut() {
                use crate::imaging::metadata::{GenerationMetadata, LoraEntry};
                // Same square / divisible-by-8 contract as t2i::run.
                // Stage C's prior is fixed at 24×24×16; the pipeline
                // can only produce square output, so non-square sizes
                // are a hard error.
                anyhow::ensure!(
                    eff_w == eff_h,
                    "Stable Cascade output is square; task size is {}x{}.",
                    eff_w,
                    eff_h
                );
                anyhow::ensure!(
                    eff_w % 8 == 0,
                    "Stable Cascade output dim must be divisible by 8; got {}.",
                    eff_w
                );
                let cascade_output_dim = eff_w as u32;
                let stage_c_steps = (eff_steps * 2).div_ceil(3).max(1);
                let stage_b_steps = eff_steps.saturating_sub(stage_c_steps).max(1);
                // v0.38 phase 3: scenario-level Cascade LoRA stack
                // for metadata. Merged at load time; per-task
                // overrides are not yet wired (Cascade has no
                // per-task runtime LoRA swap).
                let cascade_lora_entries: Vec<LoraEntry> =
                    loras.iter().map(|s| s.to_entry()).collect();
                // v0.41 phase 3: build the per-task Cascade ControlNet
                // conditioning. The CN is loaded only when some task
                // carries a control spec; here we materialise this
                // task's conditioning image (pre-rendered `image=`;
                // `auto-from` annotation for Cascade is a follow-up).
                let cascade_control: Option<crate::pipelines::cascade::ControlConditioning> =
                    if cp.control_conditioning_active() {
                        let controls = task_effective_controls(task)?;
                        match controls.first() {
                            Some(spec) => {
                                // v0.43: support BOTH `image:` (pre-rendered
                                // edge map) and `auto-from:` (auto-annotate),
                                // mirroring `cascade::run`. Both feed Stage C
                                // the [-1,1] conditioning the CN expects.
                                let cond = if let Some(image_path) = spec.image.as_ref() {
                                    crate::imaging::preprocess::sd_image_tensor(
                                        image_path, 1024, 1024, &device, cp.dtype,
                                    )?
                                } else if let Some(from_path) = spec.auto_from.as_ref() {
                                    let kind: crate::pipelines::controlnet::ControlKind =
                                        spec.kind.parse().with_context(|| {
                                            format!(
                                                "task {:?}: control kind {:?}",
                                                task.name, spec.kind
                                            )
                                        })?;
                                    let edges =
                                        crate::pipelines::controlnet_annotator::annotate(
                                            kind, from_path, 1024, 1024, &device, cp.dtype,
                                        )
                                        .await?;
                                    edges.affine(2.0, -1.0)?
                                } else {
                                    anyhow::bail!(
                                        "task {:?}: Cascade control requires `image:` or \
                                         `auto-from:`",
                                        task.name
                                    );
                                };
                                Some(crate::pipelines::cascade::ControlConditioning {
                                    conditioning_image: cond,
                                    scale: spec.strength.unwrap_or(1.0),
                                    start: spec.start.unwrap_or(0.0),
                                    end: spec.end.unwrap_or(1.0),
                                })
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                for img_idx in 0..eff_count {
                    let img_seed = task_seed.wrapping_add(img_idx as u64);
                    let mut nohook: Option<&mut dyn crate::pipelines::step_hook::StepHook> = None;
                    let (buf, ow, oh) = cp.generate(
                        &final_prompt,
                        &eff_negative,
                        cascade_output_dim,
                        stage_c_steps,
                        stage_b_steps,
                        eff_guidance,
                        // v0.42 phase 0: decoder guidance default; a
                        // scenario-level override is a follow-up.
                        1.1,
                        img_seed,
                        eff_scheduler,
                        cascade_control.as_ref(),
                        &mut nohook,
                    )?;
                    let mut m = GenerationMetadata::new(
                        final_prompt.clone(),
                        model.clone(),
                        img_seed,
                        eff_steps,
                        eff_guidance,
                        format!("{:?}", eff_scheduler).to_lowercase(),
                        ow,
                        oh,
                    );
                    m.negative = eff_negative.clone();
                    if !cascade_lora_entries.is_empty() {
                        m.with_lora_stack(cascade_lora_entries.clone());
                        m.lora_scale = Some(lora_scale);
                    }
                    // v0.43: record the ControlNet so the proof image
                    // self-documents its conditioning (mirrors the CLI /
                    // scripting Cascade paths).
                    if cp.control_conditioning_active() {
                        if let Some(spec) = task_effective_controls(task)?.first() {
                            m.with_control_stack(vec![
                                crate::imaging::metadata::ControlEntry {
                                    kind: spec.kind.clone(),
                                    image: spec
                                        .image
                                        .as_ref()
                                        .map(|p| p.display().to_string()),
                                    from: spec
                                        .auto_from
                                        .as_ref()
                                        .map(|p| p.display().to_string()),
                                    video: None,
                                    strength: spec.strength.unwrap_or(1.0),
                                    start: spec.start.unwrap_or(0.0),
                                    end: spec.end.unwrap_or(1.0),
                                },
                            ]);
                        }
                    }
                    let out_path = task_out
                        .join(format!("plakat-cascade-{img_seed}.png"));
                    crate::imaging::io::save_rgb_u8_with_metadata(
                        &buf,
                        ow,
                        oh,
                        &out_path,
                        &m,
                    )?;
                }
            } else
            // v0.36 phase 0: PixArt dispatch arm. Routes per-task PixArt
            // tasks through the scenario-cached `pixart::Pipeline`.
            // PixArt has no runtime per-task LoRA swap (merge happens
            // at load time per v0.35 phase 4); scenarios with PixArt
            // per-task LoRA overrides are documented as a v0.36 phase
            // 2/3 follow-up. Falls through to the SD3 → SD/Flux match
            // for non-PixArt tasks.
            if let Some(pp) = pixart_pipeline.as_mut() {
                use crate::imaging::metadata::{GenerationMetadata, LoraEntry};
                let pixart_lora_entries: Vec<LoraEntry> = loras
                    .iter()
                    .map(|s| s.to_entry())
                    .collect();
                // PixArt's generate produces one image per call;
                // honour eff_count by stepping the seed per-image.
                for img_idx in 0..eff_count {
                    let img_seed = task_seed.wrapping_add(img_idx as u64);
                    let mut nohook: Option<&mut dyn crate::pipelines::step_hook::StepHook> = None;
                    let (buf, ow, oh) = pp.generate(
                        &final_prompt,
                        &eff_negative,
                        eff_w,
                        eff_h,
                        eff_steps,
                        eff_guidance,
                        img_seed,
                        eff_scheduler,
                        &mut nohook,
                    )?;
                    // Build sidecar metadata. Same field set
                    // `pixart::run` emits (v0.35 phase 4).
                    let mut m = GenerationMetadata::new(
                        final_prompt.clone(),
                        model.clone(),
                        img_seed,
                        eff_steps,
                        eff_guidance,
                        format!("{:?}", eff_scheduler).to_lowercase(),
                        eff_w,
                        eff_h,
                    );
                    m.negative = eff_negative.clone();
                    if !pixart_lora_entries.is_empty() {
                        m.with_lora_stack(pixart_lora_entries.clone());
                        m.lora_scale = Some(lora_scale);
                    }
                    let out_path = task_out
                        .join(format!("plakat-pixart-{img_seed}.png"));
                    crate::imaging::io::save_rgb_u8_with_metadata(
                        &buf,
                        ow,
                        oh,
                        &out_path,
                        &m,
                    )?;
                }
            } else
            // v0.16 phase 2: SD3 dispatch arm. Routes per-task tasks
            // through the scenario-cached `sd3::Pipeline` rather than
            // building a fresh pipeline per task. Keeps SD3-family
            // LoRA / Tiled wiring intact for runtime per-task LoRA.
            // Falls through to the SD/Flux match for non-SD3 tasks.
            if let Some(sp) = sd3_pipeline.as_mut() {
                use crate::pipelines::sd3;
                // Mirrors the field mapping in t2i.rs's SD3 dispatch.
                // Only forwards steps / guidance when the user moved
                // them off plakat's defaults so SD3's variant-specific
                // recommendations stay in play otherwise.
                let sd3_req = sd3::GenRequest {
                    prompt: final_prompt.clone(),
                    negative: eff_negative.clone(),
                    width: eff_w,
                    height: eff_h,
                    count: eff_count,
                    steps: if eff_steps == 28 { None } else { Some(eff_steps) },
                    guidance: if (eff_guidance - 7.5).abs() < f64::EPSILON {
                        None
                    } else {
                        Some(eff_guidance)
                    },
                    seed: Some(task_seed),
                    out_dir: task_out.clone(),
                    init_image: eff_init_image.clone(),
                    mask: eff_mask.clone(),
                    mask_feather: task.mask_feather.unwrap_or(8),
                    mask_invert: task.mask_invert.unwrap_or(false),
                    strength: task.strength,
                    // v0.15 phase 5: tiled denoise. Composes with
                    // pure t2i; img2img + tiled bails inside the SD3
                    // pipeline (mutually-exclusive design).
                    tiled: eff_tiled.clone(),
                    regions: eff_regions.clone(),
                    // v0.16 phase 3: SD3 CN per-call conditioning
                    // overrides. Empty Vec preserves whatever the
                    // load-time conditioning paths were (which is
                    // also empty in scenarios today — SD3 CN scenario
                    // wiring lands in a later phase).
                    controlnet_conditioning: Vec::new(),
                    // v0.20: scenarios don't expose --format yet —
                    // default to PNG (the v0.17 A1111-compat path).
                    output_format: crate::imaging::io::OutputFormat::Png,
                };
                sp.generate(&sd3_req)?;
                if task_lora_applied {
                    sp.clear_all_loras()?;
                    tracing::debug!(
                        target: "plakat",
                        "task {:?}: cleared runtime LoRA stack on SD3 backbone",
                        task.name
                    );
                }
            } else {
            match (&pipeline, flux_pipeline.as_mut()) {
                // SD: reuse the loaded UNet/VAE/CLIP/LoRA across tasks.
                // v0.13 phase 10: dispatch to the tiled SDXL path when
                // the scenario sets `tiled:` (matches `plakat generate
                // --tiled`).
                // v0.13 phase 11: if the task has `init-image:`, hand
                // off to img2img::run (SD img2img / inpaint flow).
                // This currently reloads the SD model per such task —
                // the t2i Pipeline is built for load-once-generate-many,
                // but img2img doesn't share that shape yet. Acceptable
                // for v0.13's batch-rarely-img2img workflows; a follow-
                // up could share the SdCore between paths.
                (Some(p), _) if eff_init_image.is_some() => {
                    if eff_tiled.is_some() {
                        bail!(
                            "task {:?}: --tiled does not yet compose with SD img2img / \
                             inpaint in scenarios. Drop `tiled:` or `init-image:`.",
                            task.name
                        );
                    }
                    // v0.14 phase 7: pass the t2i Pipeline by ref so
                    // `run_sd_img2img_task` can share its SdCore with
                    // the img2img runner. Pre-phase-7 the helper
                    // delegated to `img2img::run` which built its own
                    // ~5GB SD pipeline per task.
                    run_sd_img2img_task(
                        task,
                        &gen_req,
                        &task_controls,
                        &model,
                        &loras,
                        lora_scale,
                        eff_scheduler,
                        &device,
                        eff_init_image.as_ref().unwrap(),
                        eff_mask.as_deref(),
                        p,
                    )
                    .await?;
                }
                (Some(p), _) => {
                    if !eff_regions.is_empty() {
                        crate::pipelines::tiled::check_regional_combo(
                            eff_tiled.is_some(),
                            !make_control_reqs().is_empty(),
                        )?;
                        p.generate_regional(&gen_req, &eff_regions)?;
                    } else {
                        match eff_tiled.as_ref() {
                            Some(tcfg) => p.generate_tiled(&gen_req, tcfg.clone())?,
                            None => p.generate(&gen_req, &make_control_reqs())?,
                        }
                    }
                }
                // Flux: reuse the loaded transformer + AE + T5 + CLIP across tasks.
                (_, Some(fp)) => {
                    if !eff_regions.is_empty() {
                        anyhow::bail!(
                            "task '{}': regions are not supported for Flux yet — \
                             use an SD 1.5 / SDXL / SD3.5 model",
                            task.name
                        );
                    }
                    // Pass `steps` / `guidance` through to Flux only if they
                    // diverge from plakat's generic defaults (28 / 7.5) so
                    // Flux's variant-specific defaults stay in play otherwise.
                    let flux_steps = if eff_steps == 28 { None } else { Some(eff_steps) };
                    let flux_guidance = if (eff_guidance - 7.5).abs() < f64::EPSILON {
                        None
                    } else {
                        Some(eff_guidance)
                    };

                    // v0.13 phase 10/11: per-task Flux ControlNet swap.
                    // The Pipeline holds up to `max_flux_controls`
                    // Union Pro v2 instances pre-loaded at scenario
                    // startup. For each entry in the task's
                    // `effective_controls()` we mutate one slot's
                    // call params + conditioning. Unused slots are
                    // cleared so they contribute no residuals.
                    // `_cn_tmps` outlives the `generate` call so any
                    // auto-annotator PNGs stay readable.
                    //
                    // v0.14 phase 3c: parse the task's Redux specs
                    // here so an invalid spec fails the task with a
                    // clear error before the slow generate kicks in.
                    let task_redux_specs: Vec<
                        crate::pipelines::flux_redux::ReduxSpec,
                    > = task
                        .redux_images
                        .iter()
                        .map(|s| s.parse::<crate::pipelines::flux_redux::ReduxSpec>())
                        .collect::<Result<Vec<_>>>()
                        .with_context(|| {
                            format!("task {:?}: parsing redux-images entries", task.name)
                        })?;
                    let task_flux_controls = task_effective_controls(task)?;
                    let mut _cn_tmps: Vec<tempfile::TempDir> = Vec::new();
                    if fp.has_controlnets() {
                        let slots = fp.controlnet_count();
                        if task_flux_controls.len() > slots {
                            bail!(
                                "task {:?}: {} controls requested but only {} Flux CN slots \
                                 pre-loaded — scenario startup undercount",
                                task.name,
                                task_flux_controls.len(),
                                slots
                            );
                        }
                        for (slot, cspec) in task_flux_controls.iter().enumerate() {
                            let parsed: crate::pipelines::controlnet::ControlKind = cspec
                                .kind
                                .parse()
                                .with_context(|| {
                                    format!(
                                        "task {:?} control[{slot}] kind '{}' not recognised",
                                        task.name, cspec.kind
                                    )
                                })?;
                            // Shakker-Labs Union Pro v2 mode index per kind
                            // — same mapping the CLI dispatch uses.
                            use crate::pipelines::controlnet::ControlKind as CK;
                            let mode = Some(match parsed {
                                CK::Canny | CK::Lineart => 0u32,
                                CK::SoftEdge => 1u32,
                                CK::OpenPose => 2u32,
                                CK::Depth => 3u32,
                                CK::Tile => anyhow::bail!(
                                    "Tile ControlNet is not supported on Flux (SD 1.5/SDXL only)"
                                ),
                            });
                            let scale = cspec.strength.unwrap_or(1.0);
                            let start = cspec.start.unwrap_or(0.0);
                            let end = cspec.end.unwrap_or(1.0);
                            fp.set_controlnet_call_params(slot, mode, scale, start, end)?;
                            let cond_path = match (cspec.image.as_ref(), cspec.auto_from.as_ref()) {
                                (Some(p), None) => p.clone(),
                                (None, Some(from_path)) => {
                                    let anno_dtype = if matches!(device, Device::Cpu) {
                                        candle_core::DType::F32
                                    } else {
                                        candle_core::DType::BF16
                                    };
                                    let anno = crate::pipelines::controlnet_annotator::annotate(
                                        parsed, from_path, eff_w, eff_h, &device, anno_dtype,
                                    )
                                    .await?;
                                    let tmp = tempfile::Builder::new()
                                        .prefix("plakat-scenario-flux-anno-")
                                        .tempdir()?;
                                    let out_path = tmp
                                        .path()
                                        .join(format!("cn{slot}-{}.png", parsed.slug()));
                                    write_flux_anno_png(&anno, &out_path)?;
                                    _cn_tmps.push(tmp);
                                    out_path
                                }
                                (Some(_), Some(_)) => bail!(
                                    "task {:?} control[{slot}]: image and auto-from are mutually exclusive",
                                    task.name
                                ),
                                (None, None) => bail!(
                                    "task {:?} control[{slot}]: requires image or auto-from",
                                    task.name
                                ),
                            };
                            fp.set_controlnet_conditioning(slot, Some(cond_path))?;
                        }
                        // Clear any leftover slots from a previous task
                        // that had more controls than this one.
                        for slot in task_flux_controls.len()..slots {
                            fp.set_controlnet_conditioning(slot, None)?;
                        }
                    }

                    fp.generate(&flux::GenRequest {
                        prompt: gen_req.prompt.clone(),
                        width: gen_req.width,
                        height: gen_req.height,
                        count: gen_req.count,
                        steps: flux_steps,
                        guidance: flux_guidance,
                        seed: gen_req.seed,
                        out_dir: gen_req.out_dir.clone(),
                        // Per-CN conditioning is now set via
                        // `set_controlnet_conditioning` (per-slot)
                        // rather than the singular `req.conditioning`
                        // back-compat shim.
                        conditioning: None,
                        // v0.13 phase 10/11: surface Flux img2img /
                        // Fill / outpaint inputs and tiled denoise at
                        // task scope. `eff_init_image` / `eff_mask`
                        // are the outpaint-synthesised canvas+mask
                        // when `outpaint:` is set, else passthrough.
                        init_image: eff_init_image.clone(),
                        mask: eff_mask.clone(),
                        strength: task.strength,
                        // v0.15 phase 7a: per-task tiled override
                        // (was scenario-global only).
                        tiled: eff_tiled.clone(),
                        // v0.14 phase 3c: per-task Redux. The
                        // task's `redux-images: [...]` is parsed
                        // upstream as `Vec<ReduxSpec>`; pass it
                        // through verbatim. The Pipeline was loaded
                        // with `redux: any_task_has_redux` above so
                        // the encoder is available iff any task uses
                        // it.
                        redux_images: task_redux_specs.clone(),
                        // v0.15 phase 7a: concept-variant conditioning
                        // — `task.concept-image:` overrides scenario
                        // `concept-image:`; bails inside the Flux
                        // pipeline if the loaded variant doesn't
                        // expect this input.
                        concept_conditioning: eff_concept_image.clone(),
                        // v0.18 Kontext phase 4: opt-in 17-bucket
                        // aspect snap. Ignored when the resolved
                        // variant isn't Kontext.
                        kontext_bucket: eff_kontext_bucket,
                        // v0.20: scenarios don't expose --format yet.
                        output_format: crate::imaging::io::OutputFormat::Png,
                    })?;
                }
                // Dry-run path doesn't reach here.
                (None, None) => unreachable!("non-dry-run task without a pipeline"),
            }

            // v0.15 phase 7b-7: clear the runtime LoRA stack after
            // the task generates. Restores every LoraLinear to the
            // scenario-merged baseline so the next task isn't
            // contaminated by this task's deltas. No-op when
            // task.loras was empty.
            if task_lora_applied {
                if let Some(fp) = flux_pipeline.as_mut() {
                    fp.backbone().clear_all_loras()?;
                    tracing::debug!(
                        target: "plakat",
                        "task {:?}: cleared runtime LoRA stack",
                        task.name
                    );
                }
            }
            } // close `else` (v0.16 phase 2 SD3 dispatch branch)
            }
        }

        // -------- artefact compositing (post-generate, pre-stylize) --------
        // Stylize will re-paint over the composited artefacts via IP-Adapter,
        // unifying their palette with the generated scene — that's the
        // reason this step lands before stylize, not after.
        if !task.artefacts.is_empty() {
            let specs: Vec<crate::artefacts::ArtefactSpec> = task
                .artefacts
                .iter()
                .map(crate::artefacts::ArtefactSpecEntry::to_spec)
                .collect::<Result<_>>()
                .with_context(|| format!("task {:?}: parsing artefact specs", task.name))?;
            let library_dir = s
                .artefact_library
                .clone()
                .unwrap_or_else(|| PathBuf::from("assets/artefact_library"));

            // v3: lazily load DepthPipeline on first task that wants it.
            // Subsequent tasks reuse the loaded instance; load failure
            // is sticky for the rest of the run (we don't retry).
            let task_smart = task.smart_zones.unwrap_or(s.smart_zones);
            if task_smart && smart_depth.is_none() && !smart_depth_attempted && any_smart {
                smart_depth_attempted = true;
                match crate::pipelines::depth::DepthPipeline::load(device.clone()).await {
                    Ok(p) => smart_depth = Some(p),
                    Err(e) => {
                        crate::ui::progress::println(&format!(
                            "  {} smart-zones depth load failed ({e}). Falling back \
                             to rigid grid for this run.",
                            style("warn:").yellow().bold(),
                        ));
                    }
                }
            }
            let smart_ref = if task_smart { smart_depth.as_ref() } else { None };

            crate::artefacts::composite_onto_seed_range(
                &specs,
                &library_dir,
                &task_out,
                Some(task_seed),
                eff_count,
                prefix,
                eff_w,
                eff_h,
                &s.zones,
                smart_ref,
            )
            .with_context(|| format!("task {:?}: compositing artefacts", task.name))?;

            // v2: masked img2img blending pass — per-task override
            // falls back to scenario-level toggle.
            let blend_on = task.artefact_blend.unwrap_or(s.artefact_blend);
            if blend_on {
                let strength = task
                    .artefact_blend_strength
                    .or(s.artefact_blend_strength)
                    .unwrap_or(0.3);
                let files: Vec<PathBuf> = (0..eff_count)
                    .map(|i| {
                        let seed = task_seed.wrapping_add(i as u64);
                        task_out.join(format!("{prefix}-{seed}.png"))
                    })
                    .filter(|p| p.exists())
                    .collect();
                crate::pipelines::artefact_blend::blend_files(
                    crate::pipelines::artefact_blend::BlendConfig {
                        model: model.clone(),
                        device: device.clone(),
                        loras: loras.clone(),
                        lora_scale,
                        prompt: final_prompt.clone(),
                        negative: eff_negative.clone(),
                        image_w: eff_w,
                        image_h: eff_h,
                        steps: eff_steps,
                        guidance: eff_guidance,
                        scheduler: eff_scheduler,
                        strength,
                        feather_px: None,
                    },
                    &specs,
                    &library_dir,
                    &files,
                    &s.zones,
                    Some(task_seed),
                    smart_ref,
                    // Phase 7e: reuse the scenario's t2i pipeline core
                    // when present. The blend's BlendConfig.model is
                    // already the scenario's main `model` (same one the
                    // t2i pipeline was loaded with), so the core matches.
                    // For dry-run or Flux scenarios the t2i pipeline is
                    // None — fall back to the blend's own load. (Flux
                    // blends aren't supported anyway; this just keeps
                    // the dry-run path inert.)
                    pipeline.as_ref().map(|p| p.core()),
                )
                .await
                .with_context(|| format!("task {:?}: blending artefacts", task.name))?;
            }
        }

        // Optional post-generate style pass.
        let style_attempted = task.style.is_some();
        if let Some(style_ref) = &task.style {
            if !style_ref.exists() {
                crate::ui::progress::println(&format!(
                    "  {} style reference not found: {} — skipping",
                    style("warn:").yellow().bold(),
                    style_ref.display(),
                ));
            } else if let Some(sp) = stylize_pipeline.as_ref() {
                run_style_pass(
                    sp,
                    style_ref,
                    task.style_strength.unwrap_or(0.6),
                    &task_out,
                    task_seed,
                    eff_count,
                    prefix,
                );
            }
        }

        // Optional post-generate upscale pass.
        // Targets the stylized image when stylize was requested, otherwise the
        // original. Falls back to the original (with a warning) if the styled
        // file isn't on disk (e.g. stylize failed).
        if s.upscale.upscale {
            run_upscale_pass(
                &task_out,
                task_seed,
                eff_count,
                prefix,
                style_attempted,
                s.upscale.scale,
                upscale_method,
                &device,
                esrgan.as_ref(),
            )
            .await;
        }

        Ok(GenerateOutcome::NeedSuccessRecord)
        }.await; // end of v0.34 phase 2 generate body wrap

        match generate_result {
            Ok(GenerateOutcome::AlreadyRecorded) => {
                // Body's early-exit path pushed its own record
                // (dry-run skip / --resume cache hit). Nothing to do.
            }
            Ok(GenerateOutcome::NeedSuccessRecord) => {
                // v0.34 phase 2: generate task reached the end → ok or
                // dry-run. Previously the success push was inline;
                // it's been hoisted into the Ok-arm so the Err-arm can
                // emit a `failed` record carrying e.to_string().
                task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: "generate".to_string(),
                    status: if args.dry_run { "dry-run" } else { "ok" }.to_string(),
                    seed: Some(task_seed_for_record),
                    note: None,
                    error: None,
                });
            }
            Err(e) => {
                crate::ui::progress::println(&format!(
                    "  {} task {:?}: {}",
                    style("✗ failed").red().bold(),
                    task.name,
                    e
                ));
                task_records.push(TaskRunRecord {
                    name: task.name.clone(),
                    kind: "generate".to_string(),
                    status: "failed".to_string(),
                    seed: Some(task_seed_for_record),
                    note: None,
                    error: Some(format!("{e:#}")),
                });
                any_task_failed = true;
            }
        }
        // Global seed_offset always advances by the GLOBAL count so a
        // re-run with the same scenario gives the same global-seed
        // tasks the same composition, regardless of per-task overrides.
        seed_offset += count as u64;
    }

    // Flush the final iteration's terminal record(s) to the status board.
    while emitted_records < task_records.len() {
        let r = &task_records[emitted_records];
        emit(
            &events,
            ScenarioEvent::TaskFinished {
                index: emitted_records,
                name: r.name.clone(),
                status: r.status.clone(),
            },
        );
        emitted_records += 1;
    }
    {
        let failed = task_records.iter().filter(|r| r.status == "failed").count();
        let ok = task_records.len().saturating_sub(failed);
        emit(&events, ScenarioEvent::Finished { ok, failed });
    }

    // v0.18: tag the summary line so dry-run users can tell at
    // a glance that nothing was actually written to disk. Without
    // this, "✓ done N images" misleads — they'd look in the out
    // dir, find it empty, and wonder what happened.
    if args.dry_run {
        sout!(
            "\n{} would have generated {} image(s) across {} task(s) → {} \
             (no files written — drop --dry-run to actually generate)",
            style("(dry-run)").yellow().bold(),
            total_images,
            s.tasks.len(),
            out_root.display()
        );
    } else {
        sout!(
            "\n{} {} task(s), {} image(s) → {}",
            style("✓ done").green().bold(),
            s.tasks.len(),
            total_images,
            out_root.display()
        );
    }

    // v0.34 phase 2: count failures up-front so the post-summary
    // bail message can include it. task_records gets moved into the
    // summary struct below.
    let failed_count = task_records.iter().filter(|r| r.status == "failed").count();

    // v0.33 phase 2: optional structured run summary for CI /
    // automation consumers. Written after the console output so
    // any disk-write failure shows up alongside the existing
    // success line. JSON is canonical so `jq` / Python parsers can
    // ingest it directly.
    if let Some(path) = args.json_summary.as_ref() {
        let ran = task_records
            .iter()
            .filter(|r| r.status == "ok" || r.status == "dry-run")
            .count();
        let skipped = task_records.iter().filter(|r| r.status == "skipped").count();
        let failed = failed_count;
        let summary = ScenarioRunSummary {
            scenario_file: args.file.display().to_string(),
            model: model.clone(),
            out_dir: out_root.display().to_string(),
            total_tasks: s.tasks.len(),
            ran,
            skipped,
            failed,
            wall_time_secs: run_started.elapsed().as_secs_f64(),
            plakat_version: env!("CARGO_PKG_VERSION").to_string(),
            tasks: task_records,
        };
        let json = serde_json::to_string_pretty(&summary)
            .with_context(|| "serialising scenario run summary")?;
        std::fs::write(path, json)
            .with_context(|| format!("writing --json-summary to {}", path.display()))?;
        crate::ui::progress::println(&format!(
            "  {} run summary → {}",
            style("·").dim(),
            path.display()
        ));
    }

    // v0.34 phase 2: if any task failed, exit non-zero. Summary
    // file was already written above (so CI consumers get the full
    // failure breakdown even when the process exits with an error).
    if any_task_failed {
        anyhow::bail!(
            "{failed_count} task(s) failed — see preceding errors or --json-summary for details"
        );
    }

    Ok(())
}

/// v0.16 phase 11: surface SD-family per-task LoRA incompatibility
/// upfront with actionable guidance. Catches three patterns:
///
/// 1. **All task LoRA stacks identical** → suggest folding to
///    scenario-level `loras:` (zero-cost fix, applied once at load).
/// 2. **Stacks vary across tasks** → suggest Flux / SD3.5 switch or
///    splitting the scenario.
/// 3. **SD3 / Flux model** → no-op (those support runtime LoRA).
///
/// Runs ONCE at scenario start, before any model load. Cheap.
fn sd_per_task_lora_preflight(
    s: &ScenarioFile,
    model: &str,
) -> Result<()> {
    use crate::pipelines::t2i::Variant;
    let variant = Variant::detect(model);
    // SD3 + Flux support runtime per-task LoRA; nothing to warn.
    // v0.36 phase 0: PixArt's runtime per-task LoRA dispatch lands
    // alongside the v0.36 phase 2/3 variant work. For phase 0, the
    // scenario load merges scenario-level LoRAs into the tempfile
    // once; per-task PixArt LoRA overrides are tracked the same way
    // the SD-family preflight tracks SD per-task LoRAs and will
    // surface here in a later phase. For now: skip the warning so
    // PixArt scenarios load cleanly.
    // v0.37 phase 5: Stable Cascade also skips. Scenario-level
    // LoRAs land in v0.38 alongside the Cascade LoRA story (deferred
    // from v0.37 per the cycle's locked decision).
    if variant.is_flux()
        || variant.is_sd3()
        || variant.is_pixart()
        || variant.is_cascade()
    {
        return Ok(());
    }
    // Collect tasks that declare per-task loras.
    let with_loras: Vec<&TaskDef> = s
        .tasks
        .iter()
        .filter(|t| !t.loras.is_empty())
        .collect();
    if with_loras.is_empty() {
        return Ok(());
    }
    // Pattern 1: every task with loras has the same stack.
    let first = &with_loras[0].loras;
    let all_same = with_loras.iter().all(|t| t.loras == *first);
    if all_same && with_loras.len() == s.tasks.len() {
        // Every task uses the same loras AND no task omits them.
        // Folding to scenario-level is byte-equivalent.
        crate::ui::progress::println(&format!(
            "  {} every task uses the same per-task `loras:` stack ({} entries). \
             On SD-family models per-task LoRA runtime swap isn't wired \
             (vendor work — see phase 11 notes). The stack is identical across \
             tasks, so it can be folded to scenario-level `loras:` for the same \
             result without the per-task swap requirement:",
            console::style("hint:").yellow().bold(),
            first.len(),
        ));
        crate::ui::progress::println(&format!("    loras: {first:?}"));
        crate::ui::progress::println(&format!(
            "    {} drop the per-task `loras:` from every task block",
            console::style("then").dim(),
        ));
        return Ok(());
    }
    // Pattern 2: stacks vary. Bail clearly upfront so the user
    // doesn't watch model load for 30s before hitting the per-task
    // apply_loras bail.
    anyhow::bail!(
        "scenario uses SD-family model `{model}` with per-task LoRA stacks that \
         vary across {} task(s) — SD UNet runtime per-task LoRA swap isn't \
         wired yet (vendor work; see phase 11). Workarounds:\n  \
         1. Switch to a Flux / SD3.5 model (instant per-task LoRA swap via the \
         runtime stack).\n  \
         2. Split into multiple scenarios, one per LoRA stack — each pays the \
         SD UNet load once.\n  \
         3. If you only need ONE per-task LoRA combination, move it to the \
         scenario-level `loras:` block and drop the per-task overrides.",
        with_loras.len()
    );
}

fn validate_enhancer_keys(enhancer: &str) -> Result<()> {
    let cfg = crate::config::Config::load()?;
    let lower = enhancer.to_lowercase();
    // v0.20 #5: accept the `local` + `local:<alias>` + `auto`
    // providers `prompt::enhance` already supports. Previously
    // scenarios were gated to cloud providers only, which made
    // `plakat init`-generated starters non-runnable without an
    // API key.
    if lower == "local" || lower.starts_with("local:") || lower == "auto" {
        return Ok(());
    }
    match lower.as_str() {
        "deepseek" => {
            if cfg.deepseek_api_key.is_none() {
                bail!(
                    "scenario uses `enhancer: deepseek` but DEEPSEEK_API_KEY \
                     is not set in the environment or ~/.config/plakat/config.toml"
                );
            }
        }
        "gemini" => {
            if cfg.gemini_api_key.is_none() {
                bail!(
                    "scenario uses `enhancer: gemini` but GEMINI_API_KEY \
                     is not set in the environment or ~/.config/plakat/config.toml"
                );
            }
        }
        other => bail!(
            "unknown enhancer {other:?} \
             (expected: deepseek | gemini | local | local:<alias> | auto)"
        ),
    }
    Ok(())
}

/// Compare two LoRA spec lists for equivalence under the per-task
/// style override constraint: scenarios share a pre-loaded pipeline,
/// so we can't swap LoRAs per task. If a per-task style's resolved
/// LoRAs match the currently-loaded set (typically both empty for
/// trigger-only styles), the override is safe; otherwise we warn.
///
/// Compared by spec-string + scale, which is what the user-facing
/// catalog encodes — bit-for-bit `LoraSpec` equality isn't useful
/// because the type doesn't implement `PartialEq`.
fn same_lora_set(a: &[LoraSpec], b: &[LoraSpec]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let key = |s: &LoraSpec| format!("{:?}|{}", s.source, s.scale);
    let mut a_keys: Vec<String> = a.iter().map(key).collect();
    let mut b_keys: Vec<String> = b.iter().map(key).collect();
    a_keys.sort();
    b_keys.sort();
    a_keys == b_keys
}

/// Truncate a string to `max` characters for inclusion in error messages.
/// Appends `…` when truncated. Char-boundary safe.
fn trim_preview(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{}…", truncated)
    }
}

/// Strip leading/trailing whitespace and commas from each part so users can
/// write fragments like `"fantasy art,"` or `, masterpiece` without producing
/// double-commas in the final prompt.
fn join_parts(parts: &[&str]) -> String {
    parts
        .iter()
        .map(|s| s.trim().trim_matches(|c: char| c == ',' || c.is_whitespace()))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Run the IP-Adapter stylize pass on every image produced by a task. The
/// original `plakat-<seed>.png` (or `plakat-flux-<seed>.png`) is preserved;
/// the styled version is written next to it as `…-styled.png`.
///
/// Failures on individual images are logged but don't abort the scenario —
/// you keep the original even if stylization fails (e.g. SDXL output with
/// dims stylize can't handle).
fn run_style_pass(
    pipeline: &stylize::Pipeline,
    ref_path: &std::path::Path,
    strength: f32,
    out_dir: &std::path::Path,
    seed_start: u64,
    count: u32,
    prefix: &str,
) {
    for i in 0..count {
        let seed = (seed_start + i as u64) & (u32::MAX as u64);
        let in_path = out_dir.join(format!("{prefix}-{seed}.png"));
        let out_path = out_dir.join(format!("{prefix}-{seed}-styled.png"));

        if !in_path.exists() {
            crate::ui::progress::println(&format!(
                "  {} expected {} not on disk — stylize skipped",
                style("warn:").yellow().bold(),
                in_path.display(),
            ));
            continue;
        }

        crate::ui::progress::println(&format!(
            "  {} {} (REF {}, strength {:.2})",
            style("stylize").cyan().bold(),
            in_path.display(),
            ref_path.display(),
            strength,
        ));

        let req = stylize::GenRequest {
            input: in_path,
            reference: ref_path.to_path_buf(),
            out: out_path,
            strength,
            steps: 30,
            seed: Some(seed),
            ref_blur: 0.0,
            ref_weight: 1.0,
        };
        if let Err(e) = pipeline.stylize_one(&req) {
            crate::ui::progress::println(&format!(
                "  {} stylize failed: {e}",
                style("warn:").yellow().bold(),
            ));
        }
    }
}

/// Run the classical upscaler on every image produced by a task.
///
/// Target file:
///   - If `style_attempted` and `<task>/plakat-<seed>-styled.png` exists → upscale it.
///   - Else → upscale `<task>/plakat-<seed>.png`.
/// v0.31 phase 3: build a fresh SD-family t2i Pipeline for the
/// scenario. Extracted from the per-scenario load site so the
/// kind-switching evictor (mixed-kind scenarios) can reload the
/// pipeline after the loop drops it on a switch to animate.
///
/// `loras` is the scenario-level LoRA spec stack; per-task LoRA
/// overlays still happen via the v0.18 inline LoRA tag merger.
/// `use_refiner` is the scenario-level `refiner` flag.
///
/// `vae_cache` (v0.32 phase 2): when `Some`, the SD core load reuses
/// the cached `AutoEncoderKL` instead of materializing a fresh one
/// from disk. Mixed-kind scenarios populate the cache from the
/// first SD-family pipeline load and pass it on every subsequent
/// reload, skipping ~330 MB SDXL VAE rebuild per kind switch.
#[allow(clippy::too_many_arguments)]
/// Whether a caller-supplied resident pipeline (for `loaded_model`) can be reused as this
/// scenario's SD base. Safe only when the run actually uses an SD base, the models match,
/// and there are no scenario-level LoRAs or refiner — so the reused pipeline is byte-for-byte
/// the base the runner would have loaded. Per-task LoRAs (applied later) don't affect this.
fn can_reuse_sd_pipeline(
    will_use_sd_base: bool,
    loaded_model: &str,
    scenario_model: &str,
    scenario_loras: &[crate::pipelines::lora::LoraSpec],
    use_refiner: bool,
) -> bool {
    will_use_sd_base && loaded_model == scenario_model && scenario_loras.is_empty() && !use_refiner
}

async fn load_sd_pipeline_for_scenario(
    model: &str,
    device: &candle_core::Device,
    loras: &[crate::pipelines::lora::LoraSpec],
    lora_scale: f32,
    use_refiner: bool,
    vae_cache: Option<std::sync::Arc<candle_transformers::models::stable_diffusion::vae::AutoEncoderKL>>,
) -> Result<Pipeline> {
    Pipeline::load(LoadRequest {
        model: model.to_string(),
        device: device.clone(),
        loras: loras.to_vec(),
        lora_scale,
        use_refiner,
        // v0.16 phase 9 / v0.30 phase 0 follow-up: scenarios still
        // don't surface --embedding. The runtime TI path ships
        // (v0.30 phase 0) but the scenario schema doesn't expose
        // it; that's a v0.33+ candidate.
        embeddings: Vec::new(),
        vae_cache,
    })
    .await
}

///
/// Output is written next to the source with `-upscaled` appended.
async fn run_upscale_pass(
    out_dir: &std::path::Path,
    seed_start: u64,
    count: u32,
    prefix: &str,
    style_attempted: bool,
    scale: f32,
    method: UpscaleMethod,
    device: &candle_core::Device,
    esrgan: Option<&EsrganPipeline>,
) {
    for i in 0..count {
        let seed = (seed_start + i as u64) & (u32::MAX as u64);
        let styled = out_dir.join(format!("{prefix}-{seed}-styled.png"));
        let orig = out_dir.join(format!("{prefix}-{seed}.png"));

        // Pick source per the rule.
        let (source, suffix) = if style_attempted && styled.exists() {
            (styled, "styled-upscaled")
        } else {
            if style_attempted {
                crate::ui::progress::println(&format!(
                    "  {} styled image missing; upscaling original instead",
                    style("warn:").yellow().bold(),
                ));
            }
            (orig, "upscaled")
        };
        if !source.exists() {
            crate::ui::progress::println(&format!(
                "  {} {} not on disk — upscale skipped",
                style("warn:").yellow().bold(),
                source.display(),
            ));
            continue;
        }
        let dest = out_dir.join(format!("{prefix}-{seed}-{suffix}.png"));

        let result = match (method.is_ml(), esrgan) {
            // Cached ESRGAN model — no per-image build cost.
            (true, Some(p)) => p.upscale_file(&source, &dest),
            // ML method but no preloaded pipeline (shouldn't happen in normal
            // scenario flow; fall back to the one-shot path).
            (true, None) => {
                crate::imaging::upscale::ml_upscale(&source, &dest, method, device).await
            }
            (false, _) => crate::imaging::upscale::upscale(&source, &dest, scale, method),
        };
        match result {
            Ok((w, h, nw, nh)) => {
                let shown = method.native_scale().unwrap_or(scale);
                crate::ui::progress::println(&format!(
                    "  {} {} ({}×{} → {}×{}, {:.2}×, {:?})",
                    style("upscale").cyan().bold(),
                    dest.display(),
                    w,
                    h,
                    nw,
                    nh,
                    shown,
                    method,
                ));
            }
            Err(e) => crate::ui::progress::println(&format!(
                "  {} upscale failed for {}: {e}",
                style("warn:").yellow().bold(),
                source.display(),
            )),
        }
    }
}

/// Word-wrap `text` under a labeled line. Continuation lines are indented
/// to line up after the `"  <label>: "` prefix so the result reads as one
/// logical entry. Existing newlines in `text` are treated as whitespace
/// (HJSON multi-line strings carry editor-formatting newlines that aren't
/// semantically meaningful to SD).
///
/// Format:
///     "  pre-enhance: first line of wrapped text up to terminal width"
///     "               second line continues at the same column"
///     "               third line ..."
fn wrap_label(label: &str, text: &str) -> String {
    let cols = terminal_width();
    let prefix_len = 2 + label.chars().count() + 2; // "  " + label + ": "
    let avail = cols.saturating_sub(prefix_len).max(40);
    let indent = " ".repeat(prefix_len);

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= avail {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }

    let label_styled = style(label).dim();
    let mut out = format!("  {label_styled}: {}", lines[0]);
    for line in &lines[1..] {
        out.push('\n');
        out.push_str(&indent);
        out.push_str(line);
    }
    out
}

/// Any non-default per-task field set? Used to decide whether the dry-run
/// should print an "overrides:" line.
fn has_overrides(task: &TaskDef) -> bool {
    task.size.is_some()
        || task.aspect.is_some()
        || task.count.is_some()
        || task.steps.is_some()
        || task.guidance.is_some()
        || task.seed.is_some()
        || task.negative.is_some()
        || task.scheduler.is_some()
        || task.refine.is_some()
        || task.refine_strength.is_some()
        || task.refiner_frac.is_some()
}

fn describe_overrides(task: &TaskDef) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(v) = &task.size {
        parts.push(format!("size={v}"));
    }
    if let Some(v) = &task.aspect {
        parts.push(format!("aspect={v}"));
    }
    if let Some(v) = task.count {
        parts.push(format!("count={v}"));
    }
    if let Some(v) = task.steps {
        parts.push(format!("steps={v}"));
    }
    if let Some(v) = task.guidance {
        parts.push(format!("guidance={v}"));
    }
    if let Some(v) = task.seed {
        parts.push(format!("seed={v}"));
    }
    if task.negative.is_some() {
        parts.push("negative=…".to_string());
    }
    if let Some(v) = &task.scheduler {
        parts.push(format!("scheduler={v}"));
    }
    if let Some(v) = task.refine {
        parts.push(format!("refine={v}"));
    }
    if let Some(v) = task.refine_strength {
        parts.push(format!("refine-strength={v}"));
    }
    if let Some(v) = task.refiner_frac {
        parts.push(format!("refiner-frac={v}"));
    }
    parts.join(", ")
}

fn terminal_width() -> usize {
    console::Term::stdout()
        .size_checked()
        .map(|(_, c)| c as usize)
        .unwrap_or(100)
}

/// Build a `(1, 1, latent_h, latent_w)` mask from a normalised bbox.
///
/// `bbox = [x0, y0, x1, y1]` is in the unit square. Pixel-space bounds
/// are computed against the latent dimensions (not the image dims) so
/// the mask aligns with what the UNet sees. Values are `1.0` inside
/// the bbox, `0.0` outside; the dtype matches the pipeline's latents
/// so `broadcast_mul` works without an extra cast.
///
/// Edges round inward: x0/y0 use `ceil`, x1/y1 use `floor`, so the
/// mask always strictly fits inside the bbox.
fn build_persona_mask(
    bbox: [f32; 4],
    latent_w: usize,
    latent_h: usize,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<candle_core::Tensor> {
    let [x0, y0, x1, y1] = bbox;
    let lw = latent_w as f32;
    let lh = latent_h as f32;
    let xs = (x0 * lw).floor().max(0.0) as usize;
    let ys = (y0 * lh).floor().max(0.0) as usize;
    let xe = (x1 * lw).ceil().min(lw) as usize;
    let ye = (y1 * lh).ceil().min(lh) as usize;

    // Defensive: if the bbox collapses (e.g. a 1-pixel persona slot at
    // 32× latent compression), expand to at least 1×1 so the mask isn't
    // entirely zero (which would make inpaint a no-op).
    let xe = xe.max(xs + 1).min(latent_w);
    let ye = ye.max(ys + 1).min(latent_h);

    let mut buf = vec![0f32; latent_w * latent_h];
    for y in ys..ye {
        for x in xs..xe {
            buf[y * latent_w + x] = 1.0;
        }
    }
    let t = candle_core::Tensor::from_vec(buf, (1, 1, latent_h, latent_w), device)?;
    Ok(t.to_dtype(dtype)?)
}

/// Sanitize a task name for use as a directory.
fn safe_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// v0.17 phase 5: check whether every expected output PNG for
/// a task is already on disk. Probes each of the four prefixes
/// backbones write under (`plakat-`, `plakat-flux-`, `plakat-sd3-`,
/// `plakat-img2img-`, `plakat-inpaint-`, `plakat-portrait-`) — if
/// **any** prefix has all N seeds present, the task is treated as
/// already-generated and `--resume` skips it.
///
/// Per-image seed: `task_seed + i`, masked to `u32::MAX` to match
/// what the SD save sites use.
fn task_outputs_all_present(task_out: &std::path::Path, task_seed: u64, count: u32) -> bool {
    if count == 0 {
        return false;
    }
    if !task_out.exists() {
        return false;
    }
    let prefixes = [
        "plakat-",
        "plakat-flux-",
        "plakat-sd3-",
        "plakat-img2img-",
        "plakat-inpaint-",
        "plakat-portrait-",
    ];
    for prefix in prefixes {
        let all_present = (0..count).all(|i| {
            let seed = (task_seed + i as u64) & (u32::MAX as u64);
            let path = task_out.join(format!("{prefix}{seed}.png"));
            path.exists()
        });
        if all_present {
            return true;
        }
    }
    false
}

/// v0.13 phase 10: write a `(1, 3, H, W)` `[0, 1]` ControlNet annotator
/// tensor as an 8-bit RGB PNG. Mirrors `t2i::write_annotator_tensor_as_png`
/// (kept private there) so scenarios don't take a dependency on
/// internal pipeline code.
/// v0.13 phase 11: dispatch one SD img2img / inpaint task in a
/// scenario by delegating to `img2img::run`. Reloads the SD model
/// per call (img2img doesn't share the t2i Pipeline's load-once-
/// generate-many shape today). Caller has already validated that
/// `task.init_image.is_some()`.
#[allow(clippy::too_many_arguments)]
/// v0.15 phase 7b-7: apply a task's per-task runtime LoRA stack.
///
/// Backbone routing:
/// * Flux (BF16 / GGUF / NF4): resolves user specs via
///   `flux_lora::compute_runtime_specs` and calls
///   `FluxBackbone::apply_loras`.
/// * SD3 / SD3.5: resolves via `sd3_lora::compute_runtime_specs` and
///   calls `sd3::Pipeline::apply_loras`. The scenario caches the
///   `sd3::Pipeline` across tasks so the runtime LoRA stack is the
///   per-task delta on top of the scenario-merged baseline.
/// * SD-family (sd15 / sd21 / sdxl): bails — the SD UNet's Linears
///   aren't yet wrapped as `LoraLinear`. Use scenario-level `loras:`
///   for SD models.
///
/// Returns `true` when a stack was applied (caller clears at
/// end-of-task), `false` when no application happened.
///
/// Returns `Ok(true)` when a stack was successfully applied (caller
/// must clear after the task) and `Ok(false)` when no application
/// happened (e.g. all specs resolved to unknown targets — silent
/// skip with a debug log).
async fn apply_task_loras_for_dispatch(
    task: &TaskDef,
    task_loras: &[String],
    task_lora_scale: f32,
    flux_pipeline: Option<&mut crate::pipelines::flux::Pipeline>,
    sd_pipeline: &Option<crate::pipelines::t2i::Pipeline>,
    sd3_pipeline: Option<&mut crate::pipelines::sd3::Pipeline>,
    model: &str,
    device: &Device,
) -> Result<bool> {
    use crate::pipelines::lora::LoraSpec;
    use crate::pipelines::t2i::Variant as TVariant;

    // Parse + resolve user specs (downloads any HF-hosted LoRAs).
    let parsed: Vec<LoraSpec> = task_loras
        .iter()
        .map(|s| s.parse::<LoraSpec>())
        .collect::<Result<_>>()
        .with_context(|| format!("task {:?}: parsing loras", task.name))?;
    if parsed.is_empty() {
        return Ok(false);
    }
    let mut resolved: Vec<crate::pipelines::lora::ResolvedLora> =
        Vec::with_capacity(parsed.len());
    for spec in &parsed {
        resolved.push(spec.resolve().await.with_context(|| {
            format!("task {:?}: resolving lora {spec:?}", task.name)
        })?);
    }

    let variant = TVariant::detect(model);
    if variant.is_flux() {
        let fp = flux_pipeline.ok_or_else(|| {
            anyhow::anyhow!(
                "task {:?}: declared Flux task LoRAs but no Flux pipeline loaded",
                task.name
            )
        })?;
        let (specs, modified, total) =
            crate::pipelines::flux_lora::compute_runtime_specs(
                &resolved, task_lora_scale, device,
            )?;
        tracing::info!(
            target: "plakat",
            "task {:?}: staging {} per-task Flux runtime LoRA target(s) ({modified}/{total} groups)",
            task.name,
            specs.len()
        );
        let dtype = fp.dtype();
        let applied = fp.backbone().apply_loras(specs, dtype, device)?;
        tracing::debug!(
            target: "plakat",
            "task {:?}: applied {} per-task runtime LoRA target(s) to Flux backbone",
            task.name, applied
        );
        return Ok(true);
    }

    if variant.is_sd3() {
        let sp = sd3_pipeline.ok_or_else(|| {
            anyhow::anyhow!(
                "task {:?}: declared SD3 task LoRAs but no SD3 pipeline loaded",
                task.name
            )
        })?;
        // v0.16 phase 2: per-task SD3 LoRA via the runtime stack.
        // Mirrors the Flux dispatch — `compute_runtime_specs` builds
        // the path-keyed map, then `apply_loras` updates every
        // registered LoraLinear in MMDiT.
        let hidden_size = sp.variant().mmdit_hidden_size();
        let (specs, modified, total) =
            crate::pipelines::sd3_lora::compute_runtime_specs(
                &resolved, task_lora_scale, hidden_size, device,
            )?;
        tracing::info!(
            target: "plakat",
            "task {:?}: staging {} per-task SD3 runtime LoRA target(s) ({modified}/{total} groups)",
            task.name,
            specs.len()
        );
        let applied = sp.apply_loras(specs)?;
        tracing::debug!(
            target: "plakat",
            "task {:?}: applied {} per-task runtime LoRA target(s) to SD3 backbone",
            task.name, applied
        );
        return Ok(true);
    }

    if sd_pipeline.is_some() {
        anyhow::bail!(
            "task {:?}: per-task LoRA on SD-family (SD 1.5 / 2.1 / SDXL) needs a \
             runtime-swappable UNet. The proper fix vendors candle's UNet model so \
             every internal `nn::Linear` can be wrapped as `LoraLinear` (same pattern \
             Flux + SD3 use — see `pipelines::flux_lora` / `pipelines::sd3_lora`). \
             Without that, each per-task LoRA swap would require reloading the UNet \
             from disk (~15-30s overhead per task on SDXL), defeating the runtime \
             stack's purpose. Workarounds:\n  \
             1. Move per-task LoRAs to the scenario's `loras:` block when they're \
             the same across every task — applied once at load time, no per-task \
             cost.\n  \
             2. Switch to Flux / SD3.5 models — they support instant per-task LoRA \
             swap via the runtime LoraLinear stack (v0.15 phase 7b + v0.16 phase 2).\n  \
             3. Split into multiple scenarios, one per LoRA stack — the SD UNet \
             load is a one-time cost per scenario.",
            task.name
        );
    }

    // Unreachable in practice — dry-run paths short-circuit earlier.
    Ok(false)
}

async fn run_sd_img2img_task(
    task: &TaskDef,
    gen_req: &GenRequest,
    task_controls: &[&ControlSpec],
    model: &str,
    loras: &[crate::pipelines::lora::LoraSpec],
    lora_scale: f32,
    scheduler: SchedulerKind,
    device: &Device,
    init_image: &std::path::Path,
    mask: Option<&std::path::Path>,
    sd_pipeline: &Pipeline,
) -> Result<()> {
    use crate::pipelines::{controlnet, img2img, portrait};

    // Default strength matches the CLI flow: 0.6 for img2img, 1.0 for
    // inpaint. Task can override via `strength:`.
    let strength = task
        .strength
        .unwrap_or_else(|| if mask.is_some() { 1.0 } else { 0.6 });
    if !(0.0..=1.0).contains(&strength) || !strength.is_finite() {
        bail!(
            "task {:?}: strength must be finite in [0, 1], got {strength}",
            task.name
        );
    }
    // Translate scenario ControlSpec → CLI/library ControlSpec.
    let mut cli_controls: Vec<controlnet::ControlSpec> = Vec::with_capacity(task_controls.len());
    for spec in task_controls {
        let kind: controlnet::ControlKind = spec.kind.parse().with_context(|| {
            format!(
                "task {:?}: parsing control kind {:?}",
                task.name, spec.kind
            )
        })?;
        cli_controls.push(controlnet::ControlSpec {
            kind,
            image: spec.image.clone(),
            from: spec.auto_from.clone(),
            video: None,
            strength: spec.strength.unwrap_or(1.0),
            start: spec.start.unwrap_or(0.0),
            end: spec.end.unwrap_or(1.0),
        });
    }
    let req = img2img::Request {
        prompt: gen_req.prompt.clone(),
        negative: gen_req.negative.clone(),
        model: model.to_string(),
        device: device.clone(),
        // v0.14 phase 7: the scenario's t2i Pipeline already has these
        // LoRAs merged into the SdCore we're about to reuse. Pass them
        // along for parity with `img2img::run`'s Request shape, but
        // `run_with_pipeline` doesn't re-merge — the pipeline's
        // weights are authoritative.
        loras: loras.to_vec(),
        lora_scale,
        input: init_image.to_path_buf(),
        mask: mask.map(|p| p.to_path_buf()),
        mask_feather: task.mask_feather.unwrap_or(8),
        mask_invert: task.mask_invert.unwrap_or(false),
        width: gen_req.width,
        height: gen_req.height,
        count: gen_req.count,
        steps: gen_req.steps,
        guidance: gen_req.guidance,
        scheduler,
        strength,
        seed: gen_req.seed,
        out_dir: gen_req.out_dir.clone(),
        controls: cli_controls,
    };
    // v0.14 phase 7: share the t2i Pipeline's SdCore with the
    // img2img runner instead of paying for a second multi-GB load
    // per task. `portrait::Pipeline::from_core` is just an `Arc`
    // clone of the existing core (no identity encoder needed for
    // img2img). Pre-phase-7 each scenario img2img task triggered a
    // fresh `portrait::Pipeline::load` inside `img2img::run`.
    let port = portrait::Pipeline::from_core(sd_pipeline.core());
    img2img::run_with_pipeline(&port, &req).await?;
    Ok(())
}

fn write_flux_anno_png(anno: &candle_core::Tensor, out_path: &std::path::Path) -> Result<()> {
    use candle_core::{DType, IndexOp};
    let (b, c, h, w) = anno.dims4()?;
    if b != 1 || c != 3 {
        anyhow::bail!(
            "annotator output expected shape (1, 3, H, W), got ({b}, {c}, {h}, {w})"
        );
    }
    let scaled = (anno * 255.0)?
        .clamp(0f32, 255f32)?
        .to_dtype(DType::U8)?
        .i(0)?
        .permute((1, 2, 0))?;
    let buf = scaled.flatten_all()?.to_vec1::<u8>()?;
    crate::imaging::io::save_rgb_u8(&buf, w as u32, h as u32, out_path)?;
    Ok(())
}

// ================================================================
// v0.29 phase 3: animate-task dispatch.
// ================================================================

#[allow(clippy::too_many_arguments)]
async fn run_animate_task_inline(
    s: &ScenarioFile,
    task: &TaskDef,
    eff: &EffectiveAnimateCfg,
    pre_refine: &str,
    task_pos: usize,
    task_seed: u64,
    width: u32,
    height: u32,
    task_out: &std::path::Path,
    args: &ScenarioArgs,
    device: &candle_core::Device,
    base_alias: &str,
    animate_sd15: &mut Option<crate::pipelines::animatediff::AnimateDiffPipeline>,
    animate_sd15_key: &mut Option<String>,
    animate_sdxl: &mut Option<crate::pipelines::animatediff::AnimateDiffSdxlPipeline>,
    animate_sdxl_key: &mut Option<String>,
    // v0.34 phase 3: scenario-level VAE cache (shared with t2i path
    // via the v0.32 phase 2 mechanism). Pre-load: lookup by alias
    // to skip the ~330 MB SDXL VAE rebuild. Post-load: populate so
    // subsequent t2i loads of the same alias reuse this pipeline's
    // VAE.
    vae_cache: &mut Option<(
        String,
        std::sync::Arc<candle_transformers::models::stable_diffusion::vae::AutoEncoderKL>,
    )>,
) -> Result<()> {
    use crate::pipelines::animatediff::{AnimateDiffPipeline, AnimateDiffSdxlPipeline};
    use crate::pipelines::controlnet::load_control_stack;
    use crate::pipelines::lora::LoraSpec;
    use crate::pipelines::scheduler::SchedulerKind;
    use crate::pipelines::sd_core::SdVariant;

    // Variant detect on the resolved repo path (mirrors animate CLI).
    let resolved = if base_alias.contains('/') {
        base_alias.to_string()
    } else {
        crate::hf::resolve_alias(base_alias).to_string()
    };
    let variant = SdVariant::detect(&resolved);
    if !matches!(variant, SdVariant::Sd15 | SdVariant::Sdxl) {
        bail!(
            "scenario animate task {:?}: model {base_alias:?} resolves to \
             {variant:?} which has no upstream motion adapter. Use sd15 or sdxl.",
            task.name
        );
    }
    if eff.lcm && matches!(variant, SdVariant::Sdxl) {
        bail!(
            "scenario animate task {:?}: lcm=true on SDXL not supported \
             (wangfuyun/AnimateLCM-SDXL isn't publicly available).",
            task.name
        );
    }

    crate::ui::progress::println(&format!(
        "\n{} [{}/{}] {} {} (scene={}, weather={}, frames={}, format={})",
        style("▶").cyan().bold(),
        task_pos,
        s.tasks.len(),
        style(&task.name).bold(),
        style("animate").magenta(),
        task.scene,
        task.weather,
        eff.frames,
        eff.format,
    ));
    crate::ui::progress::println(&wrap_label("prompt", pre_refine));

    // v0.27 phase 5: --resume detects an already-rendered task by
    // the presence of frame-0000.png in the task's out_dir.
    let frame0 = task_out.join("frame-0000.png");
    if args.resume && frame0.exists() {
        crate::ui::progress::println(&format!(
            "  ↺ {}: frame-0000.png already on disk — skipping",
            console::style(&task.name).cyan(),
        ));
        return Ok(());
    }

    if args.dry_run {
        crate::ui::progress::println(&format!(
            "  {} would render {} frames at {}x{} (seed={task_seed}, \
             lcm={}, motion-loras={}, format={}, out={})",
            style("[dry-run]").yellow(),
            eff.frames,
            width,
            height,
            eff.lcm,
            eff.motion_loras.len(),
            eff.format,
            task_out.display(),
        ));
        return Ok(());
    }

    std::fs::create_dir_all(task_out).with_context(|| {
        format!(
            "scenario animate task {:?}: creating out_dir {}",
            task.name,
            task_out.display()
        )
    })?;

    if eff.format.needs_ffmpeg() {
        let v = crate::imaging::video::ffmpeg_version()?;
        tracing::info!(target: "plakat", "ffmpeg detected ({v})");
    }

    let dtype = if matches!(device, candle_core::Device::Cpu) {
        candle_core::DType::F32
    } else {
        candle_core::DType::BF16
    };

    // Parse motion-LoRA specs.
    let motion_lora_specs: Vec<LoraSpec> = eff
        .motion_loras
        .iter()
        .map(|s| {
            s.parse::<LoraSpec>().with_context(|| {
                format!(
                    "scenario animate task {:?}: parsing motion-lora {s:?}",
                    task.name
                )
            })
        })
        .collect::<Result<_>>()?;

    // Cache key encodes everything that changes the loaded pipeline.
    let motion_loras_joined = eff.motion_loras.join("|");
    let mode_tag = if eff.lcm { "lcm" } else { "v3" };

    // Per-task ControlNet stack. Translate scenario ControlSpec
    // (local struct) → pipelines::controlnet::ControlSpec (library
    // type) by parsing the kind string. Animate honours `start`/
    // `end` ramps the same way the t2i path does (passes through
    // to load_control_stack, which builds an active-step window).
    let scenario_controls = task_effective_controls(task)?;
    let mut cli_controls: Vec<crate::pipelines::controlnet::ControlSpec> =
        Vec::with_capacity(scenario_controls.len());
    for spec in &scenario_controls {
        let kind: crate::pipelines::controlnet::ControlKind = spec
            .kind
            .parse()
            .with_context(|| {
                format!(
                    "scenario animate task {:?}: parsing control kind {:?}",
                    task.name, spec.kind
                )
            })?;
        cli_controls.push(crate::pipelines::controlnet::ControlSpec {
            kind,
            image: spec.image.clone(),
            from: spec.auto_from.clone(),
            video: spec.video.clone(),
            strength: spec.strength.unwrap_or(1.0),
            start: spec.start.unwrap_or(0.0),
            end: spec.end.unwrap_or(1.0),
        });
    }
    let controls = if cli_controls.is_empty() {
        Vec::new()
    } else {
        load_control_stack(
            &cli_controls,
            base_alias,
            width,
            height,
            device,
            dtype,
            None,
            Some(eff.frames as usize), // v0.30 phase 2: per-frame video CN
        )
        .await
        .with_context(|| {
            format!(
                "scenario animate task {:?}: loading ControlNet stack",
                task.name
            )
        })?
    };

    // Effective steps / guidance / scheduler. LCM mode applies
    // diffusers-recommended defaults when the user didn't override.
    // We inherit the scenario steps/guidance if the task didn't
    // override; same shape as the t2i path.
    let cfg_steps = task
        .steps
        .or(s.steps)
        .unwrap_or(if eff.lcm { 4 } else { 20 });
    let cfg_guidance = task
        .guidance
        .or(s.guidance)
        .unwrap_or(if eff.lcm { 1.5 } else { 7.5 });
    let cfg_scheduler = if eff.lcm {
        SchedulerKind::Lcm
    } else {
        match task.scheduler.as_deref().or(s.scheduler.as_deref()) {
            Some(name) => name.parse::<SchedulerKind>().with_context(|| {
                format!(
                    "scenario animate task {:?}: scheduler {name:?}",
                    task.name
                )
            })?,
            None => SchedulerKind::Default,
        }
    };
    let negative = task
        .negative
        .clone()
        .unwrap_or_else(|| s.negative.clone());

    // -------- variant-specific dispatch via cache slot --------
    let images: Vec<image::DynamicImage> = match variant {
        SdVariant::Sd15 => {
            let key = format!("sd15:{mode_tag}:{motion_loras_joined}");
            let hit = animate_sd15_key.as_deref() == Some(&key);
            if !hit {
                *animate_sd15 = None;
                // v0.34 phase 3: SD 1.5 animate hard-codes the
                // canonical sd15 base; cache key is "sd15" so it
                // pairs with t2i loads of the same alias.
                let vae_cache_key = "sd15";
                let cached_vae = vae_cache_lookup(vae_cache.as_ref(), vae_cache_key);
                if cached_vae.is_some() {
                    tracing::info!(
                        target: "plakat",
                        "v0.34 phase 3: VAE cache HIT on AnimateDiff SD 1.5 load (key={vae_cache_key})"
                    );
                }
                let p = if eff.lcm {
                    AnimateDiffPipeline::load_animatelcm(
                        device,
                        dtype,
                        &motion_lora_specs,
                        eff.motion_lora_scale,
                        cached_vae,
                        "sd15",
                    )
                    .await
                } else {
                    AnimateDiffPipeline::load_v3(
                        device,
                        dtype,
                        &motion_lora_specs,
                        eff.motion_lora_scale,
                        cached_vae,
                        "sd15",
                    )
                    .await
                }
                .with_context(|| {
                    format!(
                        "scenario animate task {:?}: loading SD 1.5 AnimateDiff stack",
                        task.name
                    )
                })?;
                // v0.34 phase 3: populate cache from freshly loaded
                // animate pipeline so subsequent t2i tasks for sd15
                // reuse this VAE (closing the mixed-kind rebuild gap).
                *vae_cache = Some((vae_cache_key.to_string(), std::sync::Arc::clone(&p.vae)));
                *animate_sd15 = Some(p);
                *animate_sd15_key = Some(key);
            }
            let p = animate_sd15.as_ref().expect("just inserted");
            tracing::info!(
                target: "plakat",
                "scenario animate task {:?}: {} stack — {} modules; \
                 {frames} frames at {width}x{height}, steps={cfg_steps}, \
                 guidance={cfg_guidance:.2}, scheduler={cfg_scheduler:?}, CN={}",
                task.name,
                if eff.lcm { "AnimateLCM" } else { "AnimateDiff V3" },
                p.modules.modules.len(),
                controls.len(),
                frames = eff.frames,
            );
            p.generate_long(
                pre_refine,
                &negative,
                eff.frames as usize,
                eff.window_size as usize,
                eff.window_overlap as usize,
                task_seed,
                width,
                height,
                cfg_steps,
                cfg_guidance,
                cfg_scheduler,
                &controls,
                false, // v0.32 phase 0: FreeNoise opt-in not yet on scenarios
            )?
        }
        SdVariant::Sdxl => {
            let key = format!("{base_alias}:{motion_loras_joined}");
            let hit = animate_sdxl_key.as_deref() == Some(&key);
            if !hit {
                *animate_sdxl = None;
                // v0.34 phase 3: SDXL animate uses the user's
                // base_alias; cache key is `base_alias` so it pairs
                // with SDXL t2i loads of the same alias (closing the
                // mixed-kind rebuild gap from v0.32 phase 2).
                let cached_vae = vae_cache_lookup(vae_cache.as_ref(), base_alias);
                if cached_vae.is_some() {
                    tracing::info!(
                        target: "plakat",
                        "v0.34 phase 3: VAE cache HIT on AnimateDiff SDXL load (key={base_alias})"
                    );
                }
                let p = AnimateDiffSdxlPipeline::load_sdxl_beta(
                    device,
                    dtype,
                    base_alias,
                    &motion_lora_specs,
                    eff.motion_lora_scale,
                    cached_vae,
                )
                .await
                .with_context(|| {
                    format!(
                        "scenario animate task {:?}: loading SDXL AnimateDiff beta stack",
                        task.name
                    )
                })?;
                *vae_cache = Some((base_alias.to_string(), std::sync::Arc::clone(&p.vae)));
                *animate_sdxl = Some(p);
                *animate_sdxl_key = Some(key);
            }
            let p = animate_sdxl.as_ref().expect("just inserted");
            tracing::info!(
                target: "plakat",
                "scenario animate task {:?}: AnimateDiff SDXL beta — {} modules; \
                 {frames} frames at {width}x{height}, steps={cfg_steps}, \
                 guidance={cfg_guidance:.2}, scheduler={cfg_scheduler:?}, CN={}",
                task.name,
                p.modules.modules.len(),
                controls.len(),
                frames = eff.frames,
            );
            p.generate_long(
                pre_refine,
                &negative,
                eff.frames as usize,
                eff.window_size as usize,
                eff.window_overlap as usize,
                task_seed,
                width,
                height,
                cfg_steps,
                cfg_guidance,
                cfg_scheduler,
                &controls,
                false, // v0.32 phase 0: FreeNoise opt-in not yet on scenarios
            )?
        }
        _ => unreachable!("variant gate filtered above"),
    };

    // -------- write per-frame PNGs + metadata --------
    let scheduler_name = format!("{cfg_scheduler:?}").to_lowercase();
    let mode_label = if eff.lcm {
        "animatediff-lcm"
    } else {
        "animatediff"
    };
    let mut frame_paths: Vec<std::path::PathBuf> =
        Vec::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        let frame_path = task_out.join(format!("frame-{i:04}.png"));
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let mut meta = crate::imaging::metadata::GenerationMetadata::new(
            pre_refine.to_string(),
            base_alias.to_string(),
            task_seed,
            cfg_steps,
            cfg_guidance,
            scheduler_name.clone(),
            width,
            height,
        );
        meta.negative = negative.clone();
        meta.mode = Some(mode_label.to_string());
        meta.extras.push((
            "Scenario task".to_string(),
            task.name.clone(),
        ));
        meta.extras.push((
            "AnimateDiff frame".to_string(),
            format!("{i}/{}", images.len()),
        ));
        crate::imaging::io::save_rgb_u8_with_metadata(
            rgb.as_raw(),
            w,
            h,
            &frame_path,
            &meta,
        )?;
        frame_paths.push(frame_path);
    }

    // Format dispatch — matches cli::animate::run_animatediff exactly.
    if eff.format.needs_gif() {
        let gif_path = task_out.join("animation.gif");
        crate::cli::animate::write_gif(&frame_paths, &gif_path, eff.gif_delay_ms)?;
    }
    if eff.format.needs_mp4() || eff.format.needs_webm() {
        let pattern = task_out
            .join("frame-%04d.png")
            .to_string_lossy()
            .to_string();
        let fps = 8u32;
        if eff.format.needs_mp4() {
            crate::imaging::video::frames_to_mp4(
                &pattern,
                &task_out.join("animation.mp4"),
                fps,
            )?;
        }
        if eff.format.needs_webm() {
            crate::imaging::video::frames_to_webm(
                &pattern,
                &task_out.join("animation.webm"),
                fps,
            )?;
        }
    }

    crate::ui::progress::println(&format!(
        "  ✓ wrote {} frame(s) → {} (format={})",
        images.len(),
        task_out.display(),
        eff.format,
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_sd_pipeline_only_for_the_exact_vanilla_base() {
        use crate::pipelines::lora::{LoraSource, LoraSpec};
        let none: &[LoraSpec] = &[];
        // Exact match: same model, no LoRAs, no refiner, and the run uses an SD base.
        assert!(can_reuse_sd_pipeline(true, "sdxl", "sdxl", none, false));
        // Any mismatch forbids reuse (→ safe reload).
        assert!(!can_reuse_sd_pipeline(false, "sdxl", "sdxl", none, false), "non-SD-base run");
        assert!(!can_reuse_sd_pipeline(true, "sdxl", "sd15", none, false), "different model");
        assert!(!can_reuse_sd_pipeline(true, "sdxl", "sdxl", none, true), "refiner differs");
        let with_lora = &[LoraSpec { source: LoraSource::Local("/l/x.safetensors".into()), scale: 0.8 }];
        assert!(!can_reuse_sd_pipeline(true, "sdxl", "sdxl", with_lora, false), "scenario-level LoRA");
    }

    // v0.15 phase 7a — schema parsing for the new task/scenario fields.

    fn parse_task(src: &str) -> TaskDef {
        deser_hjson::from_str::<TaskDef>(src).expect("task parses")
    }

    /// HJSON requires newline-separated keys (commas are optional but
    /// the parser doesn't reliably consume them inline). The common
    /// task fields `name` / `scene` / `weather` / `prompt` are
    /// required by the deser; we set them to placeholders in every
    /// test.
    const COMMON_TASK: &str = r#"
        name: t
        scene: s
        weather: w
        prompt: p"#;

    #[test]
    fn task_parses_fast_preset() {
        let src = format!(r#"{{{COMMON_TASK}
            fast: hyper-8
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.fast.as_deref(), Some("hyper-8"));
    }

    #[test]
    fn task_parses_map_render_tiles() {
        // 1.14.0-B: a map task can request seamless tiled output from automation.
        let src = format!(r#"{{{COMMON_TASK}
            type: map
            map-spec: corpus/map/coastal.spec.json
            map-tiles: 2x2
            map-render-tiles: true
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.map_render_tiles, Some(true));
        assert_eq!(t.map_tiles.as_deref(), Some("2x2"));
        // effective config carries it into the MapTaskCfg the runner dispatches.
        let s: ScenarioFile = deser_hjson::from_str(r#"{ tasks: [] }"#).unwrap();
        let cfg = effective_map_config(&s, &t);
        assert!(cfg.render_tiles);
    }

    #[test]
    fn task_parses_map_tile_furniture() {
        // 1.14.0-D: per-tile furniture flows task → MapTaskCfg.
        let src = format!(r#"{{{COMMON_TASK}
            type: map
            map-spec: corpus/map/realms.hjson
            map-render-tiles: true
            map-tile-furniture: true
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.map_tile_furniture, Some(true));
        let s: ScenarioFile = deser_hjson::from_str(r#"{ tasks: [] }"#).unwrap();
        let cfg = effective_map_config(&s, &t);
        assert!(cfg.render_tile_furniture);
    }

    #[test]
    fn task_parses_multiperson_block() {
        // 1.14.0-A: a `type: multiperson` task carries a `multiperson:` block of
        // scene + placed people that reference top-level personas by name.
        let src = format!(r#"{{{COMMON_TASK}
            type: multiperson
            multiperson: {{
                scene: "two friends at a cafe // watercolor"
                swap: true
                pose: true
                people: [
                    {{
                        persona: alice
                        at: "left closer front"
                    }}
                    {{
                        persona: bob
                        scale: 0.8
                    }}
                ]
            }}
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.task_type.as_deref(), Some("multiperson"));
        let mp = t.multiperson.expect("multiperson block parses");
        assert!(mp.swap && mp.pose);
        assert_eq!(mp.people.len(), 2);
        assert_eq!(mp.people[0].persona, "alice");
        assert_eq!(mp.people[0].at.as_deref(), Some("left closer front"));
        assert_eq!(mp.people[1].scale, Some(0.8));
    }

    #[test]
    fn task_parses_concept_image() {
        let src = format!(r#"{{{COMMON_TASK}
            concept-image: ./edges.png
        }}"#);
        let t = parse_task(&src);
        assert_eq!(
            t.concept_image.as_deref().map(|p| p.to_string_lossy().into_owned()),
            Some("./edges.png".to_string())
        );
    }

    // v0.18 Kontext phase 4 — kontext-bucket HJSON key at task scope.

    #[test]
    fn task_parses_kontext_bucket_true() {
        let src = format!(r#"{{{COMMON_TASK}
            kontext-bucket: true
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.kontext_bucket, Some(true));
    }

    #[test]
    fn task_parses_kontext_bucket_false() {
        let src = format!(r#"{{{COMMON_TASK}
            kontext-bucket: false
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.kontext_bucket, Some(false));
    }

    #[test]
    fn task_kontext_bucket_defaults_to_none() {
        // Task without kontext-bucket: parses None → scenario-level
        // value applies; both unset → false. Use the concept-image
        // template as the "non-empty body" so the HJSON parser is
        // happy.
        let src = format!(r#"{{{COMMON_TASK}
            concept-image: ./ref.png
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.kontext_bucket, None);
    }

    #[test]
    fn task_parses_enhance_false() {
        let src = format!(r#"{{{COMMON_TASK}
            enhance: false
        }}"#);
        let t = parse_task(&src);
        assert!(matches!(t.enhance, Some(EnhanceCfg::Toggle(false))));
    }

    #[test]
    fn task_parses_enhance_provider() {
        let src = format!(r#"{{{COMMON_TASK}
            enhance: deepseek
        }}"#);
        let t = parse_task(&src);
        assert!(matches!(
            t.enhance.as_ref(),
            Some(EnhanceCfg::Provider(p)) if p == "deepseek"
        ));
    }

    #[test]
    fn task_parses_tiled_toggle_false() {
        let src = format!(r#"{{{COMMON_TASK}
            tiled: false
        }}"#);
        let t = parse_task(&src);
        assert!(matches!(t.tiled, Some(TaskTiledCfg::Toggle(false))));
    }

    #[test]
    fn task_parses_tiled_override_block() {
        let src = format!(r#"{{{COMMON_TASK}
            tiled: {{ size: 768, stride: 512 }}
        }}"#);
        let t = parse_task(&src);
        let cfg = match t.tiled {
            Some(TaskTiledCfg::Override(c)) => c,
            other => panic!("expected Override, got {other:?}"),
        };
        assert_eq!(cfg.size, 768);
        assert_eq!(cfg.stride, 512);
    }

    #[test]
    fn task_omitting_new_fields_keeps_them_none() {
        // Backward-compat: every new v0.15 phase 7a field is optional,
        // so an existing scenario task (pre-7a schema) still parses.
        let src = format!("{{{COMMON_TASK}\n}}");
        let t = parse_task(&src);
        assert!(t.fast.is_none());
        assert!(t.concept_image.is_none());
        assert!(t.enhance.is_none());
        assert!(t.tiled.is_none());
    }

    #[test]
    fn scenario_file_parses_fast_at_global() {
        let src = r#"{
            model: flux-dev
            fast: hyper-8
            enhancer: deepseek
            lora-header: ""
        }"#;
        let s = deser_hjson::from_str::<ScenarioFile>(src)
            .expect("scenario parses");
        assert_eq!(s.fast.as_deref(), Some("hyper-8"));
    }

    // v0.25 phase 7 — scenario + per-task look/genre/offline parsing.

    #[test]
    fn scenario_file_parses_look_genre_offline_at_global() {
        let src = r#"{
            model: sdxl
            look: watercolor
            genre: anime
            offline: true
            lora-header: ""
        }"#;
        let s = deser_hjson::from_str::<ScenarioFile>(src)
            .expect("scenario with look/genre parses");
        assert_eq!(s.look.as_deref(), Some("watercolor"));
        assert_eq!(s.genre.as_deref(), Some("anime"));
        assert_eq!(s.offline, Some(true));
    }

    #[test]
    fn scenario_file_look_genre_default_to_none() {
        let src = r#"{
            model: sdxl
            lora-header: ""
        }"#;
        let s = deser_hjson::from_str::<ScenarioFile>(src)
            .expect("scenario parses");
        assert_eq!(s.look, None);
        assert_eq!(s.genre, None);
        assert_eq!(s.offline, None);
    }

    #[test]
    fn task_parses_look_genre_offline() {
        let src = format!(r#"{{{COMMON_TASK}
            look: oil-painting
            genre: anime
            offline: false
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.look.as_deref(), Some("oil-painting"));
        assert_eq!(t.genre.as_deref(), Some("anime"));
        assert_eq!(t.offline, Some(false));
    }

    #[test]
    fn task_omitting_look_genre_is_none() {
        let src = format!(r#"{{{COMMON_TASK}
        }}"#);
        let t = parse_task(&src);
        assert!(t.look.is_none());
        assert!(t.genre.is_none());
        assert!(t.offline.is_none());
    }

    /// Per-task `look:` overrides scenario-level — verified via the
    /// `.or_else(scenario)` resolution. Pure-data test of the
    /// override expression; the actual apply lives in run_one
    /// (covered by phase 11 integration).
    #[test]
    fn task_look_overrides_scenario_look() {
        let scenario_look = Some("watercolor".to_string());
        let task_look = Some("oil-painting".to_string());
        let eff = task_look.clone().or_else(|| scenario_look.clone());
        assert_eq!(eff.as_deref(), Some("oil-painting"));
    }

    /// Task with no look: inherits from scenario.
    #[test]
    fn task_look_unset_inherits_scenario() {
        let scenario_look = Some("watercolor".to_string());
        let task_look: Option<String> = None;
        let eff = task_look.or_else(|| scenario_look.clone());
        assert_eq!(eff.as_deref(), Some("watercolor"));
    }

    // v0.15 phase 7b-7 — per-task LoRA schema parsing.

    #[test]
    fn task_parses_per_task_loras() {
        let src = format!(r#"{{{COMMON_TASK}
            loras: [ "user/repo:0.7", "./local/style.safetensors" ]
            lora-scale: 0.5
        }}"#);
        let t = parse_task(&src);
        assert_eq!(t.loras.len(), 2);
        assert_eq!(t.loras[0], "user/repo:0.7");
        assert_eq!(t.loras[1], "./local/style.safetensors");
        assert_eq!(t.lora_scale, Some(0.5));
    }

    #[test]
    fn task_omitting_loras_is_empty_vec() {
        let src = format!("{{{COMMON_TASK}\n}}");
        let t = parse_task(&src);
        assert!(t.loras.is_empty());
        assert!(t.lora_scale.is_none());
    }

    #[test]
    fn scenario_file_parses_concept_image_at_global() {
        let src = r#"{
            model: flux-canny-dev
            concept-image: ./edges.png
            enhancer: deepseek
            lora-header: ""
        }"#;
        let s = deser_hjson::from_str::<ScenarioFile>(src)
            .expect("scenario parses");
        assert_eq!(
            s.concept_image.as_deref().map(|p| p.to_string_lossy().into_owned()),
            Some("./edges.png".to_string())
        );
    }

    // v0.16 phase 11 — SD per-task LoRA preflight.

    /// Minimal scenario builder via direct struct construction —
    /// HJSON's brace+newline rules make programmatic scenarios
    /// fiddly. The preflight only reads `tasks[*].loras`, so we
    /// fill those + defaults for the rest.
    fn scenario_with_task_loras(task_loras: &[&[&str]]) -> ScenarioFile {
        let tasks: Vec<TaskDef> = task_loras
            .iter()
            .enumerate()
            .map(|(i, ls)| {
                // Parse a minimal valid task via HJSON to pick up
                // serde defaults on every field we don't set, then
                // override `loras` directly.
                let src = format!(r#"{{
            name: t{i}
            scene: s
            weather: w
            prompt: p
        }}"#);
                let mut t: TaskDef = deser_hjson::from_str(&src)
                    .expect("task parses");
                t.loras = ls.iter().map(|s| s.to_string()).collect();
                t
            })
            .collect();
        // Empty scenario picks up all-default fields.
        let mut s: ScenarioFile = deser_hjson::from_str(
            r#"{
            lora-header: ""
        }"#,
        )
        .expect("scenario parses");
        s.tasks = tasks;
        s
    }

    #[test]
    fn preflight_no_per_task_loras_passes() {
        // Tasks declare no per-task LoRAs → preflight is silent.
        let s = scenario_with_task_loras(&[&[], &[], &[]]);
        sd_per_task_lora_preflight(&s, "sd15").expect("no LoRAs → ok");
        sd_per_task_lora_preflight(&s, "sdxl").expect("no LoRAs → ok");
    }

    #[test]
    fn preflight_uniform_per_task_loras_emits_hint_but_passes() {
        // Every task uses the SAME LoRA stack. The preflight prints a
        // hint to fold to scenario-level loras: but doesn't bail —
        // the user's scenario is salvageable by moving the per-task
        // block up; we don't want to block them from doing that
        // themselves.
        let s = scenario_with_task_loras(&[
            &["foo/lora-a:0.7"],
            &["foo/lora-a:0.7"],
            &["foo/lora-a:0.7"],
        ]);
        sd_per_task_lora_preflight(&s, "sd15").expect("uniform stacks → hint only");
    }

    #[test]
    fn preflight_varying_per_task_loras_bails() {
        // Two tasks with different LoRA stacks → bail loud upfront
        // with the three-option workaround.
        let s = scenario_with_task_loras(&[
            &["foo/lora-a:0.7"],
            &["bar/lora-b:0.5"],
        ]);
        let err = sd_per_task_lora_preflight(&s, "sdxl").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("vary across"), "got {msg}");
        assert!(msg.contains("Flux"), "got {msg}");
        assert!(msg.contains("Split"), "got {msg}");
    }

    #[test]
    fn preflight_partial_per_task_loras_bails() {
        // Some tasks declare LoRAs, others don't. Even if the
        // declaring tasks all use the same stack, the asymmetry
        // means we can't fold cleanly — bail.
        let s = scenario_with_task_loras(&[
            &["foo/lora-a:0.7"],
            &[],
            &["foo/lora-a:0.7"],
        ]);
        let err = sd_per_task_lora_preflight(&s, "sd15").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("vary across"), "got {msg}");
    }

    #[test]
    fn preflight_flux_model_skips() {
        // Flux supports runtime per-task LoRA — preflight is a
        // silent no-op even with varying stacks.
        let s = scenario_with_task_loras(&[
            &["foo/lora-a:0.7"],
            &["bar/lora-b:0.5"],
        ]);
        sd_per_task_lora_preflight(&s, "flux-dev").expect("Flux supports runtime LoRA");
    }

    #[test]
    fn preflight_sd3_model_skips() {
        let s = scenario_with_task_loras(&[
            &["foo/lora-a:0.7"],
            &["bar/lora-b:0.5"],
        ]);
        sd_per_task_lora_preflight(&s, "sd35-medium")
            .expect("SD3 supports runtime LoRA");
    }

    /// v0.36 phase 0: PixArt scenarios must skip the SD-style
    /// per-task LoRA bail too. PixArt's runtime per-task LoRA swap
    /// path lands alongside the v0.36 phase 2/3 variant work; for
    /// phase 0, scenario-level LoRAs are merged once at load time
    /// and per-task overrides need to load cleanly (the dispatch arm
    /// will surface unsupported-overrides separately).
    #[test]
    fn preflight_pixart_model_skips() {
        let s = scenario_with_task_loras(&[
            &["foo/lora-a:0.7"],
            &["bar/lora-b:0.5"],
        ]);
        sd_per_task_lora_preflight(&s, "pixart")
            .expect("PixArt scenarios must not bail in the SD-style preflight");
        sd_per_task_lora_preflight(&s, "pixart-sigma")
            .expect("pixart-sigma alias must also skip");
        sd_per_task_lora_preflight(
            &s,
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        )
        .expect("canonical PixArt repo string must skip too");
    }

    /// v0.37 phase 5: Stable Cascade scenarios skip the SD-style
    /// preflight too. Cascade LoRA support is deferred to v0.38;
    /// scenarios with per-task LoRAs need to load cleanly today.
    #[test]
    fn preflight_stable_cascade_model_skips() {
        let s = scenario_with_task_loras(&[
            &["foo/lora-a:0.7"],
            &["bar/lora-b:0.5"],
        ]);
        sd_per_task_lora_preflight(&s, "stable-cascade")
            .expect("Stable Cascade scenarios must not bail in the SD-style preflight");
        sd_per_task_lora_preflight(&s, "cascade")
            .expect("cascade alias must also skip");
        sd_per_task_lora_preflight(&s, "stabilityai/stable-cascade")
            .expect("canonical Stable Cascade repo string must skip too");
    }

    // v0.17 phase 5 — task_outputs_all_present probe.

    fn touch(dir: &std::path::Path, name: &str) {
        std::fs::write(dir.join(name), b"x").unwrap();
    }

    #[test]
    fn resume_probe_returns_false_when_dir_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nonexistent");
        assert!(!task_outputs_all_present(&missing, 1000, 2));
    }

    #[test]
    fn resume_probe_returns_false_when_some_files_missing() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "plakat-1000.png");
        // missing plakat-1001.png
        assert!(!task_outputs_all_present(tmp.path(), 1000, 2));
    }

    #[test]
    fn resume_probe_matches_plakat_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "plakat-1000.png");
        touch(tmp.path(), "plakat-1001.png");
        assert!(task_outputs_all_present(tmp.path(), 1000, 2));
    }

    #[test]
    fn resume_probe_matches_flux_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "plakat-flux-2000.png");
        touch(tmp.path(), "plakat-flux-2001.png");
        assert!(task_outputs_all_present(tmp.path(), 2000, 2));
    }

    #[test]
    fn resume_probe_matches_sd3_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        touch(tmp.path(), "plakat-sd3-3000.png");
        assert!(task_outputs_all_present(tmp.path(), 3000, 1));
    }

    #[test]
    fn resume_probe_zero_count_returns_false() {
        // Defensive: count == 0 means "no expected outputs" — the
        // empty `for i in 0..0` vacuously satisfies all() (returns
        // true for an empty iterator). Skip the task body via the
        // up-front `count == 0` guard instead.
        let tmp = tempfile::tempdir().unwrap();
        assert!(!task_outputs_all_present(tmp.path(), 1000, 0));
    }

    // v0.20 #5: `validate_enhancer_keys` should pass providers
    // that don't require a cloud API key without consulting the
    // env. The previous version of the gate rejected `local`,
    // which made `plakat init`-generated scenarios non-runnable
    // out of the box.

    #[test]
    fn validate_enhancer_keys_accepts_local() {
        validate_enhancer_keys("local").unwrap();
        validate_enhancer_keys("LOCAL").unwrap();
    }

    #[test]
    fn validate_enhancer_keys_accepts_local_alias() {
        validate_enhancer_keys("local:qwen2.5-1.5b").unwrap();
        validate_enhancer_keys("local:smollm2-360m").unwrap();
    }

    #[test]
    fn validate_enhancer_keys_accepts_auto() {
        validate_enhancer_keys("auto").unwrap();
    }

    #[test]
    fn validate_enhancer_keys_rejects_unknown() {
        let err = validate_enhancer_keys("openai-gpt").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown enhancer"), "got {msg}");
        // Error message advertises every supported form so users
        // can self-correct without grepping the source.
        assert!(msg.contains("local"), "got {msg}");
        assert!(msg.contains("auto"), "got {msg}");
    }

    // ================================================================
    // v0.29 phase 2 — animate scenario schema tests.
    // ================================================================

    fn parse_scenario(src: &str) -> ScenarioFile {
        deser_hjson::from_str::<ScenarioFile>(src).expect("scenario parses")
    }

    /// Scenario-level animate defaults parse with the documented keys.
    #[test]
    fn scenario_parses_animate_defaults() {
        let src = r#"{
            model: sd15
            type: animatediff
            frames: 32
            window-size: 16
            window-overlap: 4
            lcm: true
            motion-lora: [ "hf:guoyww/animatediff-motion-lora-zoom-in:0.8" ]
            motion-lora-scale: 0.7
            format: mp4
            gif-delay-ms: 125
        }"#;
        let s = parse_scenario(src);
        assert_eq!(s.task_type.as_deref(), Some("animatediff"));
        assert_eq!(s.animate_frames, Some(32));
        assert_eq!(s.animate_window_size, Some(16));
        assert_eq!(s.animate_window_overlap, Some(4));
        assert_eq!(s.lcm, Some(true));
        assert_eq!(s.motion_loras.len(), 1);
        assert!(
            (s.motion_lora_scale.unwrap() - 0.7).abs() < f32::EPSILON
        );
        assert_eq!(s.animate_format.as_deref(), Some("mp4"));
        assert_eq!(s.animate_gif_delay_ms, Some(125));
    }

    /// Task-level animate overrides parse independently of the
    /// scenario-level fields.
    #[test]
    fn task_parses_animate_overrides() {
        let src = format!(
            r#"{{{COMMON_TASK}
                type: animate
                frames: 48
                window-size: 24
                window-overlap: 8
                lcm: false
                motion-lora: [ "hf:repo:0.5" ]
                motion-lora-scale: 1.2
                format: webm
                gif-delay-ms: 50
            }}"#
        );
        let t = parse_task(&src);
        assert_eq!(t.task_type.as_deref(), Some("animate"));
        assert_eq!(t.animate_frames, Some(48));
        assert_eq!(t.animate_window_size, Some(24));
        assert_eq!(t.animate_window_overlap, Some(8));
        assert_eq!(t.lcm, Some(false));
        assert_eq!(t.motion_loras.len(), 1);
        assert!(
            (t.motion_lora_scale.unwrap() - 1.2).abs() < f32::EPSILON
        );
        assert_eq!(t.animate_format.as_deref(), Some("webm"));
        assert_eq!(t.animate_gif_delay_ms, Some(50));
    }

    /// effective_animate_config merges scenario defaults with task
    /// overrides; motion_loras list APPENDS (matches loras: pattern).
    #[test]
    fn effective_config_merges_scenario_and_task() {
        let s = parse_scenario(
            r#"{
                model: sd15
                type: animatediff
                frames: 16
                lcm: true
                motion-lora: [ "hf:base:0.7" ]
                format: gif
            }"#,
        );
        // Empty task overrides → scenario defaults win.
        let t1 = parse_task(&format!(
            "{{{COMMON_TASK}\n        }}"
        ));
        let eff1 = effective_animate_config(&s, &t1).unwrap();
        assert_eq!(eff1.frames, 16);
        assert!(eff1.lcm);
        assert_eq!(eff1.motion_loras, vec!["hf:base:0.7".to_string()]);
        assert!((eff1.motion_lora_scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(eff1.format, crate::imaging::video::Format::Gif);
        assert_eq!(eff1.gif_delay_ms, 100); // baked default

        // Task overrides win + LoRAs ARE APPENDED.
        let t2 = parse_task(&format!(
            r#"{{{COMMON_TASK}
                frames: 32
                lcm: false
                motion-lora: [ "hf:task:0.5" ]
                format: mp4
            }}"#
        ));
        let eff2 = effective_animate_config(&s, &t2).unwrap();
        assert_eq!(eff2.frames, 32);
        assert!(!eff2.lcm);
        assert_eq!(
            eff2.motion_loras,
            vec!["hf:base:0.7".to_string(), "hf:task:0.5".to_string()]
        );
        assert_eq!(eff2.format, crate::imaging::video::Format::Mp4);
    }

    /// TaskKind dispatch: explicit `animatediff` / `animate` map
    /// to Animate; absent / `generate` / `t2i` map to Generate;
    /// unknown bails.
    #[test]
    fn task_kind_classifies_strings() {
        assert_eq!(
            TaskKind::from_strs(None, None).unwrap(),
            TaskKind::Generate
        );
        assert_eq!(
            TaskKind::from_strs(None, Some("animatediff")).unwrap(),
            TaskKind::Animate
        );
        assert_eq!(
            TaskKind::from_strs(Some("animate"), Some("generate")).unwrap(),
            TaskKind::Animate
        );
        assert_eq!(
            TaskKind::from_strs(Some("t2i"), None).unwrap(),
            TaskKind::Generate
        );
        let err = TaskKind::from_strs(Some("video"), None).unwrap_err();
        assert!(err.to_string().contains("not recognised"), "{err}");
    }

    /// EffectiveAnimateCfg::validate enforces frame/window/overlap
    /// bounds at parse time (before any pipeline load).
    #[test]
    fn effective_config_validate_enforces_bounds() {
        let mut eff = EffectiveAnimateCfg {
            frames: 16,
            window_size: 16,
            window_overlap: 4,
            lcm: false,
            motion_loras: vec![],
            motion_lora_scale: 1.0,
            format: crate::imaging::video::Format::Frames,
            gif_delay_ms: 100,
        };
        assert!(eff.validate("ok").is_ok());

        // Window too large.
        eff.window_size = 64;
        eff.window_overlap = 4;
        let err = eff.validate("oversize").unwrap_err().to_string();
        assert!(err.contains("motion_max_seq_length"), "got {err}");

        // Overlap >= window.
        eff.window_size = 16;
        eff.window_overlap = 16;
        let err = eff.validate("overlap").unwrap_err().to_string();
        assert!(err.contains("window-overlap"), "got {err}");

        // Zero frames.
        eff.window_overlap = 4;
        eff.frames = 0;
        let err = eff.validate("zero").unwrap_err().to_string();
        assert!(err.contains("frames"), "got {err}");
    }

    /// Bad format string surfaces from effective_animate_config with
    /// the task name for context.
    #[test]
    fn effective_config_bad_format_bails_with_task_name() {
        let s = parse_scenario(
            r#"{
                model: sd15
                type: animatediff
                format: avif
            }"#,
        );
        let t = parse_task(&format!("{{{COMMON_TASK}\n        }}"));
        let err = effective_animate_config(&s, &t).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("\"t\""), "task name missing: {msg}");
        assert!(msg.contains("avif"), "format value missing: {msg}");
    }

    // -----------------------------------------------------------------
    // v0.31 phase 3: kind-switch cache evictor decisions.
    // -----------------------------------------------------------------

    #[test]
    fn evict_decision_first_task_no_eviction() {
        // last_kind=None means "this is the first task" — there's
        // nothing cached to evict.
        assert_eq!(evict_decision(None, TaskKind::Generate), CacheEviction::None);
        assert_eq!(evict_decision(None, TaskKind::Animate), CacheEviction::None);
    }

    #[test]
    fn evict_decision_same_kind_no_eviction() {
        // Same-kind continuation must NOT evict — that would
        // reload the pipeline on every task, defeating the cache.
        assert_eq!(
            evict_decision(Some(TaskKind::Generate), TaskKind::Generate),
            CacheEviction::None,
        );
        assert_eq!(
            evict_decision(Some(TaskKind::Animate), TaskKind::Animate),
            CacheEviction::None,
        );
    }

    #[test]
    fn evict_decision_generate_to_animate_drops_t2i() {
        assert_eq!(
            evict_decision(Some(TaskKind::Generate), TaskKind::Animate),
            CacheEviction::DropT2i,
        );
    }

    #[test]
    fn evict_decision_animate_to_generate_drops_animate() {
        assert_eq!(
            evict_decision(Some(TaskKind::Animate), TaskKind::Generate),
            CacheEviction::DropAnimate,
        );
    }

    /// Walk a realistic task sequence and verify the evictions fire
    /// at the right boundaries. This is the "behavioural" check —
    /// not just (last, current) pair decisions but the cumulative
    /// effect across a typical mixed-kind run.
    #[test]
    fn evict_decision_walks_mixed_kind_sequence() {
        use TaskKind::*;
        // Sequence: gen, gen, anim, gen, anim, anim
        let sequence = [Generate, Generate, Animate, Generate, Animate, Animate];
        let mut last: Option<TaskKind> = None;
        let mut decisions: Vec<CacheEviction> = Vec::new();
        for k in &sequence {
            decisions.push(evict_decision(last, *k));
            last = Some(*k);
        }
        // Expected: first task no-op; gen→gen no-op; gen→anim DropT2i;
        // anim→gen DropAnimate; gen→anim DropT2i; anim→anim no-op.
        assert_eq!(
            decisions,
            vec![
                CacheEviction::None,
                CacheEviction::None,
                CacheEviction::DropT2i,
                CacheEviction::DropAnimate,
                CacheEviction::DropT2i,
                CacheEviction::None,
            ],
        );
    }

    /// All-generate scenarios must NEVER evict — the t2i pipeline
    /// stays loaded for the whole run, matching the v0.29 UX.
    #[test]
    fn evict_decision_all_generate_never_evicts() {
        use TaskKind::*;
        let sequence = [Generate, Generate, Generate, Generate];
        let mut last: Option<TaskKind> = None;
        for k in &sequence {
            assert_eq!(evict_decision(last, *k), CacheEviction::None);
            last = Some(*k);
        }
    }

    /// All-animate scenarios must NEVER evict either.
    #[test]
    fn evict_decision_all_animate_never_evicts() {
        use TaskKind::*;
        let sequence = [Animate, Animate, Animate];
        let mut last: Option<TaskKind> = None;
        for k in &sequence {
            assert_eq!(evict_decision(last, *k), CacheEviction::None);
            last = Some(*k);
        }
    }

    // -----------------------------------------------------------------
    // v0.32 phase 2: VAE cache decision logic.
    // -----------------------------------------------------------------

    #[test]
    fn vae_cache_lookup_empty_returns_none() {
        // No cache → nothing to hit on.
        let cache: Option<&(String, String)> = None;
        assert_eq!(vae_cache_lookup(cache, "sdxl"), None);
    }

    #[test]
    fn vae_cache_lookup_matching_key_returns_value() {
        let entry = ("sdxl".to_string(), "VAE_marker".to_string());
        assert_eq!(
            vae_cache_lookup(Some(&entry), "sdxl"),
            Some("VAE_marker".to_string())
        );
    }

    #[test]
    fn vae_cache_lookup_mismatched_key_returns_none() {
        // Cached SDXL VAE; lookup for SD 1.5 must miss so we don't
        // hand out an incompatible VAE.
        let entry = ("sdxl".to_string(), "SDXL_VAE".to_string());
        assert_eq!(vae_cache_lookup(Some(&entry), "sd15"), None);
    }

    #[test]
    fn vae_cache_lookup_case_sensitive() {
        // Aliases pass through resolve_alias before reaching the
        // cache, so case-sensitivity matches reality. A user typing
        // `SDXL` vs `sdxl` would resolve to the same canonical
        // string before lookup; the cache trusts that contract.
        let entry = ("sdxl".to_string(), "v".to_string());
        assert_eq!(vae_cache_lookup(Some(&entry), "SDXL"), None);
    }

    /// Simulates a mixed-kind scenario's cache trace. Walk a
    /// [gen, anim, gen, anim] sequence; each kind boundary either
    /// HITs (key matches) or MISSes (different key). The test
    /// confirms the helper's decision shape across the full sequence.
    #[test]
    fn vae_cache_lookup_walks_mixed_kind_sequence_with_same_model() {
        let model = "sdxl";
        let mut cache: Option<(String, String)> = None;
        // First load — cache miss (empty).
        assert_eq!(vae_cache_lookup(cache.as_ref(), model), None);
        // Populate.
        cache = Some((model.to_string(), "SDXL_VAE_arc".to_string()));
        // Each subsequent reload against the same model — cache HIT.
        for _round in 0..3 {
            assert_eq!(
                vae_cache_lookup(cache.as_ref(), model),
                Some("SDXL_VAE_arc".to_string()),
            );
        }
    }

    #[test]
    fn vae_cache_lookup_walks_model_switch_invalidates() {
        // Scenario switches from SDXL to SD 1.5 mid-run. Cache
        // misses on the new model, the caller is responsible for
        // re-populating with the SD 1.5 VAE on next load.
        let mut cache: Option<(String, String)> = Some(("sdxl".to_string(), "v_xl".to_string()));
        assert_eq!(
            vae_cache_lookup(cache.as_ref(), "sd15"),
            None,
            "model swap must miss cache",
        );
        cache = Some(("sd15".to_string(), "v_15".to_string()));
        assert_eq!(
            vae_cache_lookup(cache.as_ref(), "sd15"),
            Some("v_15".to_string()),
        );
    }

    // -----------------------------------------------------------------
    // v0.33 phase 2: ScenarioRunSummary structure + JSON shape.
    // -----------------------------------------------------------------

    fn mk_summary() -> ScenarioRunSummary {
        ScenarioRunSummary {
            scenario_file: "test.hjson".into(),
            model: "sdxl".into(),
            out_dir: "./out".into(),
            total_tasks: 3,
            ran: 2,
            skipped: 1,
            failed: 0,
            wall_time_secs: 12.345,
            plakat_version: "0.33.0".into(),
            tasks: vec![
                TaskRunRecord {
                    name: "alpha".into(),
                    kind: "generate".into(),
                    status: "ok".into(),
                    seed: Some(42),
                    note: None,
                    error: None,
                },
                TaskRunRecord {
                    name: "beta".into(),
                    kind: "animatediff".into(),
                    status: "ok".into(),
                    seed: Some(100),
                    note: None,
                    error: None,
                },
                TaskRunRecord {
                    name: "gamma".into(),
                    kind: "generate".into(),
                    status: "skipped".into(),
                    seed: None,
                    note: Some("--only filter excluded".into()),
                    error: None,
                },
            ],
        }
    }

    #[test]
    fn summary_serializes_with_expected_top_level_keys() {
        let s = mk_summary();
        let json = serde_json::to_string_pretty(&s).unwrap();
        // Top-level shape.
        assert!(json.contains("\"scenario_file\""));
        assert!(json.contains("\"model\""));
        assert!(json.contains("\"out_dir\""));
        assert!(json.contains("\"total_tasks\""));
        assert!(json.contains("\"ran\""));
        assert!(json.contains("\"skipped\""));
        assert!(json.contains("\"failed\""));
        assert!(json.contains("\"wall_time_secs\""));
        assert!(json.contains("\"plakat_version\""));
        assert!(json.contains("\"tasks\""));
    }

    #[test]
    fn task_record_omits_none_note_field() {
        // `note: None` is `skip_serializing_if = "Option::is_none"`
        // so it stays out of the JSON for clean tasks. Tasks that
        // carry a note for skip/fail reasons emit it.
        let s = mk_summary();
        let json = serde_json::to_string(&s).unwrap();
        // "alpha" has note=None — should not produce a "note" key
        // bound to "null" (or any).
        // We expect note to appear only for `gamma` (the skip).
        let note_occurrences = json.matches("\"note\"").count();
        assert_eq!(note_occurrences, 1, "got {json}");
        assert!(json.contains("\"--only filter excluded\""));
    }

    #[test]
    fn summary_counts_align_with_tasks_array() {
        let s = mk_summary();
        let ran = s.tasks.iter().filter(|t| t.status == "ok").count();
        let skipped = s.tasks.iter().filter(|t| t.status == "skipped").count();
        // Field aggregates match the array — this catches accidental
        // divergence if the loop-side counters and the records get
        // out of sync.
        assert_eq!(ran, s.ran);
        assert_eq!(skipped, s.skipped);
    }

    #[test]
    fn task_record_kind_field_accepts_known_values() {
        // No serde enum — we serialize raw strings so animate tasks
        // can land here regardless of the v0.29 task_type aliases
        // ("animate" / "animatediff").
        let kinds = ["generate", "animatediff"];
        for k in kinds {
            let r = TaskRunRecord {
                name: "t".into(),
                kind: k.into(),
                status: "ok".into(),
                seed: Some(1),
                note: None,
                error: None,
            };
            let json = serde_json::to_string(&r).unwrap();
            assert!(json.contains(&format!("\"{}\"", k)));
        }
    }

    // v0.34 phase 2: error field behavior.

    #[test]
    fn task_record_omits_none_error_field() {
        // `error: None` is `skip_serializing_if = "Option::is_none"`
        // so it stays out of JSON for clean tasks. Mirrors the v0.33
        // phase 2 contract for the `note` field.
        let r = TaskRunRecord {
            name: "alpha".into(),
            kind: "generate".into(),
            status: "ok".into(),
            seed: Some(42),
            note: None,
            error: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("\"error\""), "got {json}");
    }

    #[test]
    fn task_record_serializes_error_field_on_failed_task() {
        let r = TaskRunRecord {
            name: "broken".into(),
            kind: "generate".into(),
            status: "failed".into(),
            seed: Some(7),
            note: None,
            error: Some("VAE encode failed: shape mismatch".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"status\":\"failed\""));
        assert!(json.contains("\"error\":\"VAE encode failed: shape mismatch\""));
    }

    #[test]
    fn summary_with_failed_record_serializes_cleanly() {
        // Full scenario summary carrying one failed record. Verifies
        // the failed-count + error text round-trip together.
        let s = ScenarioRunSummary {
            scenario_file: "broken.hjson".into(),
            model: "sd15".into(),
            out_dir: "./out".into(),
            total_tasks: 2,
            ran: 1,
            skipped: 0,
            failed: 1,
            wall_time_secs: 5.5,
            plakat_version: "0.34.0".into(),
            tasks: vec![
                TaskRunRecord {
                    name: "alpha".into(),
                    kind: "generate".into(),
                    status: "ok".into(),
                    seed: Some(42),
                    note: None,
                    error: None,
                },
                TaskRunRecord {
                    name: "beta".into(),
                    kind: "generate".into(),
                    status: "failed".into(),
                    seed: Some(43),
                    note: None,
                    error: Some("model file not found: ./missing.safetensors".into()),
                },
            ],
        };
        let json = serde_json::to_string_pretty(&s).unwrap();
        assert!(json.contains("\"failed\": 1"));
        assert!(json.contains("\"status\": \"failed\""));
        assert!(json.contains("model file not found"));
        // alpha has no error → field omitted; only one "error" key.
        assert_eq!(json.matches("\"error\":").count(), 1);
    }

    #[test]
    fn out_override_supersedes_the_scenario_out_dir() {
        // The TUI sets `out_override` so images land under the workspace out/ dir (where
        // History scans) regardless of the scenario's own `out:`. Verified via the
        // json-summary's recorded out_dir on a dry-run.
        let d = std::env::temp_dir().join("plakat-scenario-outoverride-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let file = d.join("s.hjson");
        std::fs::write(
            &file,
            r#"{"model":"stable-diffusion-v1-5/stable-diffusion-v1-5","size":"512x512","enhancer":"local","out":"./scenario-own-out","scene":[{"name":"","prompt":"p"}],"weather":[{"name":"","prompt":"c"}],"tasks":[{"name":"alpha","prompt":"a"}]}"#,
        )
        .unwrap();
        let summary = d.join("summary.json");
        let override_dir = d.join("workspace-out").join("scenarios").join("s");
        let args = ScenarioArgs {
            file,
            dry_run: true,
            resume: false,
            force: false,
            only: Vec::new(),
            limit: 0,
            json_summary: Some(summary.clone()),
            out_override: Some(override_dir.clone()),
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(run_with_events(args, None, None)).expect("dry-run should succeed");
        let written = std::fs::read_to_string(&summary).unwrap();
        assert!(written.contains(override_dir.to_str().unwrap()), "summary out_dir should be the override: {written}");
        assert!(!written.contains("scenario-own-out"), "the scenario's own out: must be ignored");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn count_zero_is_rejected_up_front_not_silently_successful() {
        // A `count: 0` task used to run the generate loop `0..0`, write nothing, and still
        // record ✓ done. It must now bail with a clear message (checked on a dry-run, which
        // reaches the up-front validation before any model load).
        let d = std::env::temp_dir().join("plakat-scenario-count0-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let file = d.join("s.hjson");
        std::fs::write(
            &file,
            r#"{"model":"stable-diffusion-v1-5/stable-diffusion-v1-5","size":"512x512","enhancer":"local","scene":[{"name":"","prompt":"p"}],"weather":[{"name":"","prompt":"c"}],"tasks":[{"name":"alpha","prompt":"a","count":0}]}"#,
        )
        .unwrap();
        let args = ScenarioArgs {
            file,
            dry_run: true,
            resume: false,
            force: false,
            only: Vec::new(),
            limit: 0,
            json_summary: None,
            out_override: None,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt.block_on(run_with_events(args, None, None)).unwrap_err();
        assert!(err.to_string().contains("count"), "should reject count:0, got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn run_with_events_emits_per_task_events_on_dry_run() {
        // A dry-run iterates every task but loads no model and hits no network
        // (Variant::detect is pure string matching), so it exercises the live
        // status-board event wiring offline.
        use std::sync::mpsc;
        let d = std::env::temp_dir().join("plakat-scenario-events-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let file = d.join("s.hjson");
        let out = d.join("out");
        // Define an empty-named scene/weather so the tasks' defaults validate.
        std::fs::write(
            &file,
            format!(
                r#"{{"model":"stable-diffusion-v1-5/stable-diffusion-v1-5","size":"512x512","enhancer":"local","out":"{}","scene":[{{"name":"","prompt":"plain"}}],"weather":[{{"name":"","prompt":"clear"}}],"tasks":[{{"name":"alpha","prompt":"a"}},{{"name":"beta","prompt":"b"}}]}}"#,
                out.to_str().unwrap()
            ),
        )
        .unwrap();

        let (tx, rx) = mpsc::channel();
        let args = ScenarioArgs {
            file,
            dry_run: true,
            resume: false,
            force: false,
            only: Vec::new(),
            limit: 0,
            json_summary: None,
            out_override: None,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(run_with_events(args, Some(tx), None)).expect("dry-run should succeed");

        let evs: Vec<ScenarioEvent> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(matches!(evs.first(), Some(ScenarioEvent::Started { total: 2 })), "first is Started{{2}}");
        assert!(matches!(evs.last(), Some(ScenarioEvent::Finished { .. })), "last is Finished");

        let started: Vec<String> = evs
            .iter()
            .filter_map(|e| match e {
                ScenarioEvent::TaskStarted { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(started, vec!["alpha", "beta"]);

        let finished: Vec<(String, String)> = evs
            .iter()
            .filter_map(|e| match e {
                ScenarioEvent::TaskFinished { name, status, .. } => Some((name.clone(), status.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            finished,
            vec![("alpha".into(), "dry-run".into()), ("beta".into(), "dry-run".into())]
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
