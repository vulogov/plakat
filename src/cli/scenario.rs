//! `scenario` — batch-generate images from an HJSON file that mixes scenes,
//! weather, and per-task prompts. See README for the schema and an example.

use anyhow::{Context, Result, anyhow, bail};
use clap::Args as ClapArgs;
use console::style;
use serde::Deserialize;
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
    #[arg(long)]
    pub dry_run: bool,
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
    scene: String,
    weather: String,
    prompt: String,

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

pub async fn run(args: ScenarioArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading {}", args.file.display()))?;
    let s: ScenarioFile = deser_hjson::from_str(&text)
        .with_context(|| format!("parsing HJSON {}", args.file.display()))?;

    // -------- validate structure --------
    if s.tasks.is_empty() {
        bail!("scenario has no `tasks` to run");
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

    let enhancer = s
        .enhancer
        .clone()
        .ok_or_else(|| anyhow!("scenario requires `enhancer` (deepseek | gemini)"))?;
    validate_enhancer_keys(&enhancer)?;

    // Parse the upscale method now so a bad string fails fast.
    let upscale_method: UpscaleMethod = s
        .upscale
        .method
        .parse()
        .with_context(|| format!("upscale.method = {:?}", s.upscale.method))?;

    // -------- resolve global parameters --------
    let model = s.model.clone().unwrap_or_else(|| "sd15".to_string());
    let device = crate::device::select(s.device.as_deref().unwrap_or("auto"))?;
    let base = s.base.unwrap_or(768);
    let count = s.count.unwrap_or(1);
    let steps = s.steps.unwrap_or(28);
    let guidance = s.guidance.unwrap_or(7.5);
    let seed = s.seed.unwrap_or(0);
    let out_root = s.out.clone().unwrap_or_else(|| PathBuf::from("./out"));
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

    // -------- execution plan summary --------
    let total_images = (s.tasks.len() as u32) * count;
    println!(
        "{}  {} task(s) × {} image(s) = {} image(s) to generate",
        style("scenario").yellow().bold(),
        s.tasks.len(),
        count,
        total_images,
    );
    println!("  model:     {model}");
    println!("  size:      {width}×{height}");
    println!("  steps:     {steps}  guidance: {guidance}  scheduler: {scheduler:?}");
    println!("  out:       {}", out_root.display());
    println!("  enhancer:  {enhancer}");
    if !loras.is_empty() {
        println!("  loras:     {} (scale {lora_scale})", loras.len());
    }
    if let Some(r) = s.refine {
        println!("  refine:    {r} steps × strength {refine_strength}");
    }
    if s.refiner {
        let frac = s.refiner_frac.unwrap_or(0.8);
        println!(
            "  refiner:   on (switch at {:.0}% of schedule, SDXL only)",
            frac * 100.0
        );
    }
    if s.upscale.upscale {
        let shown = upscale_method.native_scale().unwrap_or(s.upscale.scale);
        println!(
            "  upscale:   {:.2}× {} (post-stylize if `style` is set, else original)",
            shown, s.upscale.method
        );
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
        println!(
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

    // -------- load pipeline once (skipped for dry-run) --------
    // Two parallel pipeline types; exactly one is populated for non-dry-run.
    let variant = Variant::detect(&model);
    let pipeline: Option<Pipeline> = if args.dry_run || variant.is_flux() {
        None
    } else {
        Some(
            Pipeline::load(LoadRequest {
                model: model.clone(),
                device: device.clone(),
                loras: loras.clone(),
                lora_scale,
                use_refiner: s.refiner,
            })
            .await?,
        )
    };
    // -------- preload the stylize pipeline if any task uses `style` --------
    let any_style = s.tasks.iter().any(|t| t.style.is_some());
    let stylize_pipeline: Option<stylize::Pipeline> = if !args.dry_run && any_style {
        // stylize is SD 1.5 only — the IP-Adapter projection targets the SD 1.5
        // cross-attention dim (768). The scenario's main `model` can still be
        // anything (we operate on the produced image bytes, not on latents).
        Some(
            stylize::Pipeline::load(stylize::LoadRequest {
                model: "sd15".to_string(),
                device: device.clone(),
            })
            .await?,
        )
    } else {
        None
    };

    // -------- preload the portrait pipeline if any task uses `personas` ----
    let any_persona = s
        .tasks
        .iter()
        .any(|t| t.personas.as_deref().map(|v| !v.is_empty()).unwrap_or(false));
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
        let fvar = if variant == Variant::FluxDev {
            flux::Variant::Dev
        } else {
            flux::Variant::Schnell
        };
        let resolved_repo = if model.contains('/') {
            model.clone()
        } else {
            crate::hf::resolve_alias(&model).to_string()
        };
        Some(
            flux::Pipeline::load(flux::LoadRequest {
                variant: fvar,
                repo: resolved_repo,
                device: device.clone(),
            })
            .await?,
        )
    };

    // -------- enhance prompts up front, deduped + parallelized --------
    // Each task's `pre_refine` string is enhancer-input; under sequential
    // per-task calls this fires N times serially. Pre-loop we dedupe to the
    // set of unique prompts and fire them concurrently (capped), so a 10-task
    // scenario with one slow enhancer goes from ~10 * latency to
    // ~ceil(unique / cap) * latency. Dry-run skips this entirely.
    let enhanced_cache: HashMap<String, String> = if args.dry_run {
        HashMap::new()
    } else {
        let pre_refines: Vec<String> = s
            .tasks
            .iter()
            .map(|t| {
                let scene = scenes[t.scene.as_str()];
                let weather = weathers[t.weather.as_str()];
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

    // -------- main loop --------
    let mut seed_offset: u64 = 0;
    for (idx, task) in s.tasks.iter().enumerate() {
        let scene_prompt = scenes[task.scene.as_str()];
        let weather_prompt = weathers[task.weather.as_str()];

        let pre_refine = join_parts(&[
            &s.prompt_header,
            scene_prompt,
            weather_prompt,
            &task.prompt,
            &s.prompt_footer,
        ]);

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

        let enhanced = if args.dry_run {
            format!("(dry-run; {enhancer} not called)")
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

        let final_prompt = join_parts(&[&task_lora_header, &enhanced, &s.lora_footer]);
        crate::ui::progress::println(&wrap_label("final", &final_prompt));

        if args.dry_run {
            // Show effective per-task values in dry-run so a user can see
            // what overrides are taking effect.
            let dry_count = task.count.unwrap_or(count);
            let dry_seed = task.seed.unwrap_or(seed + seed_offset);
            crate::ui::progress::println(&format!(
                "  {} would generate {} image(s) with seeds {}..{}",
                style("(dry-run)").dim(),
                dry_count,
                dry_seed,
                dry_seed + dry_count as u64 - 1,
            ));
            if has_overrides(task) {
                crate::ui::progress::println(&format!(
                    "  {} overrides: {}",
                    style("(dry-run)").dim(),
                    describe_overrides(task)
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
            seed_offset += count as u64;
            continue;
        }

        // -------- effective per-task values (override-or-global) --------
        // Resolution: per-task size/aspect override the global pair.
        let task_size = match task.size.as_deref() {
            Some(s) => Some(s.parse::<Size>().with_context(|| format!("task size {s:?}"))?),
            None => None,
        };
        let (eff_w, eff_h) = if task_size.is_some() || task.aspect.is_some() {
            crate::imaging::sizes::resolve(
                task_size,
                task.aspect.as_deref(),
                base,
            )?
        } else {
            (width, height)
        };

        let eff_count = task.count.unwrap_or(count);
        let eff_steps = task.steps.unwrap_or(steps);
        let eff_guidance = task.guidance.unwrap_or(guidance);
        let eff_negative = task
            .negative
            .clone()
            .unwrap_or_else(|| task_negative_base.clone());
        let eff_scheduler: SchedulerKind = match task.scheduler.as_deref() {
            Some(s) => s.parse().with_context(|| format!("task scheduler {s:?}"))?,
            None => scheduler,
        };
        let eff_refine = task.refine.or(s.refine);
        let eff_refine_strength = task.refine_strength.unwrap_or(refine_strength);
        let eff_refiner_frac = task.refiner_frac.unwrap_or(s.refiner_frac.unwrap_or(0.8));
        // Seed: per-task override picks an absolute seed; global path
        // advances seed_offset to keep later tasks reproducible.
        let task_seed = task.seed.unwrap_or(seed + seed_offset);

        let task_out = out_root.join(safe_name(&task.name));

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
                })?;
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

                    let mut latents = pp.generate_latents_one(&base_req, img_seed)?;

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
                        latents = pp.inpaint_latents_one(&latents, &mask, &pass_req, pass_seed)?;
                    }

                    let out_path = task_out.join(format!("{prefix}-{img_seed}.png"));
                    pp.save_image(&latents, &out_path)?;
                }
            }

            TaskPersonas::None => {
            // -------- regular t2i / flux dispatch (unchanged behaviour) --------
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
            };
            match (&pipeline, flux_pipeline.as_mut()) {
                // SD: reuse the loaded UNet/VAE/CLIP/LoRA across tasks.
                (Some(p), _) => p.generate(&gen_req)?,
                // Flux: reuse the loaded transformer + AE + T5 + CLIP across tasks.
                (_, Some(fp)) => {
                    // Pass `steps` / `guidance` through to Flux only if they
                    // diverge from plakat's generic defaults (28 / 7.5) so
                    // Flux's variant-specific defaults stay in play otherwise.
                    let flux_steps = if eff_steps == 28 { None } else { Some(eff_steps) };
                    let flux_guidance = if (eff_guidance - 7.5).abs() < f64::EPSILON {
                        None
                    } else {
                        Some(eff_guidance)
                    };
                    fp.generate(&flux::GenRequest {
                        prompt: gen_req.prompt.clone(),
                        width: gen_req.width,
                        height: gen_req.height,
                        count: gen_req.count,
                        steps: flux_steps,
                        guidance: flux_guidance,
                        seed: gen_req.seed,
                        out_dir: gen_req.out_dir.clone(),
                    })?;
                }
                // Dry-run path doesn't reach here.
                (None, None) => unreachable!("non-dry-run task without a pipeline"),
            }
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

        // Global seed_offset always advances by the GLOBAL count so a
        // re-run with the same scenario gives the same global-seed
        // tasks the same composition, regardless of per-task overrides.
        seed_offset += count as u64;
    }

    println!(
        "\n{} {} task(s), {} image(s) → {}",
        style("✓ done").green().bold(),
        s.tasks.len(),
        total_images,
        out_root.display()
    );
    Ok(())
}

fn validate_enhancer_keys(enhancer: &str) -> Result<()> {
    let cfg = crate::config::Config::load()?;
    match enhancer.to_lowercase().as_str() {
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
        other => bail!("unknown enhancer {other:?} (expected: deepseek | gemini)"),
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
