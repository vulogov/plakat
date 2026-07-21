//! `plakat multiperson` — place specific personas into a generated scene.
//!
//! Each persona gets a relative location in words (`--at "alice:left closer front"`);
//! omit it and a scene-aware LLM places them naturally. M1 renders via the
//! portrait pipeline (scene base → per-persona inpaint). See
//! `Documentation/RFC_MULTIPERSON_REVIEW.md`.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::Args;
use candle_core::Device;

use crate::pipelines::ip_adapter::{IdentityKind, WeightedPhoto};
use crate::pipelines::multiperson::{self, MultipersonRequest, Person, Placement};

#[derive(Args, Debug)]
pub struct MultipersonArgs {
    /// Scene description (prose). A `// ` splits an inline style clause:
    /// `"three friends at tea // oil painting"`.
    #[arg(value_name = "SCENE")]
    pub scene: String,

    /// A persona: `label:photo[:weight][,photo:weight…]`. Repeatable; min 2.
    /// Multi-photo weighted merge (same as `portrait --photo`).
    #[arg(long = "person", value_name = "LABEL:PHOTO")]
    pub person: Vec<String>,

    /// Relative location for a persona: `label:<position> <distance> <facing>`,
    /// e.g. `"alice:left closer front"`. Axes (order-insensitive): position
    /// left|center-left|center|center-right|right · distance closer|mid|farther
    /// · facing front|side|back. Omit a persona to auto-place them. Repeatable.
    #[arg(long = "at", value_name = "LABEL:LOC")]
    pub at: Vec<String>,

    /// Behavioural prompt for a persona: `label:text`. Repeatable.
    #[arg(long = "person-prompt", value_name = "LABEL:TEXT")]
    pub person_prompt: Vec<String>,

    /// Explicit pixel region for a persona: `label:[x0,y0,x1,y1]` (normalised).
    /// Overrides `--at`. Repeatable.
    #[arg(long = "bbox", value_name = "LABEL:BBOX")]
    pub bbox: Vec<String>,

    /// Per-persona identity strength override: `label:0.85`. Repeatable.
    #[arg(long = "face-strength", value_name = "LABEL:F")]
    pub face_strength: Vec<String>,

    /// Per-persona figure-height scale for `--pose`: `label:0.7` (1.0 = adult,
    /// ~0.7 = a child/teen so they render shorter, not adult-sized). Repeatable.
    #[arg(long = "scale", value_name = "LABEL:F")]
    pub scale: Vec<String>,

    /// Global aesthetic clause (sent to the enhancer, never the scene analyser).
    #[arg(help_heading = "Style & look", long)]
    pub style: Option<String>,

    /// Negative prompt.
    #[arg(help_heading = "Prompt & text", long, default_value = "")]
    pub negative: String,

    /// Base model (SD 1.5 / SDXL family).
    #[arg(help_heading = "Model & sampler", long, default_value = "sd15")]
    pub model: String,

    /// Identity strategy (all personas share one): plus-face | plus-face-sdxl |
    /// faceid | faceid-sdxl.
    #[arg(long, default_value = "plus-face")]
    pub identity: IdentityKind,

    /// LLM provider for scene-aware auto-placement: deepseek | gemini | local |
    /// local:<alias> | none (geometric). Only used for un-pinned personas.
    #[arg(long = "layout-provider", default_value = "none")]
    pub layout_provider: String,

    /// Optional prompt enhancer for the scene base: deepseek | gemini | local.
    #[arg(help_heading = "Prompt & text", long)]
    pub enhancer: Option<String>,

    /// Output size, `WxH` or `N` (square).
    #[arg(help_heading = "Size & output", long, default_value = "768x768")]
    pub size: String,

    #[arg(help_heading = "Model & sampler", long, default_value_t = 30)]
    pub steps: usize,

    #[arg(help_heading = "Model & sampler", long, default_value_t = 7.5)]
    pub guidance: f64,

    #[arg(help_heading = "Model & sampler", long)]
    pub seed: Option<u64>,

    #[arg(help_heading = "Size & output", long, default_value_t = 1)]
    pub count: u32,

    #[arg(help_heading = "Model & sampler", long, default_value = "default")]
    pub scheduler: String,

    #[arg(help_heading = "Size & output", long, default_value = "./")]
    pub out: PathBuf,

    /// `--import <album>` / `--import-move`: land the composed scene in a photo album.
    #[command(flatten)]
    pub import: crate::cli::import::ImportArgs,

    /// Enable the identity face-refinement pass: detect each rendered face (SCRFD)
    /// and lightly repaint the crop conditioned on that persona's photos. Uses a
    /// LOW-strength masked repaint that preserves the face's scale/framing, so it
    /// nudges likeness without distorting. Face boxes are clamped to each
    /// persona's region. OFF by default while it's validated.
    #[arg(long = "face-refine")]
    pub face_refine: bool,

    /// IP-Adapter identity scale for the face-refinement pass (how hard to push
    /// the reference likeness). Only used with `--face-refine`.
    #[arg(long = "refine-identity", default_value_t = 0.85)]
    pub refine_face_strength: f32,

    /// Denoise strength of the face-refinement repaint, 0..1. Low (~0.3) keeps the
    /// existing face's framing and just nudges identity/detail; higher redraws it.
    /// Only used with `--face-refine`.
    #[arg(long = "refine-strength", default_value_t = 0.35)]
    pub refine_denoise: f32,

    /// **Composite** mode (recommended for true identity, widest model coverage):
    /// generate the scene background with ANY text-to-image model, matte each
    /// persona's photo (U2Net — no face model), and place them at their `--at`
    /// positions. Identity is the actual photo, so it's exact and model-agnostic.
    /// Use a photo (or any portrait) on a plain/light background for the cleanest
    /// cut-out. Add `--harmonize` to img2img-blend the composite into the scene.
    #[arg(long = "composite")]
    pub composite: bool,

    /// With `--composite`: **relight each person** to the scene's lighting
    /// (IC-Light) before placing them — matches light direction/colour so they
    /// belong in the scene instead of looking pasted. The real integration step.
    #[arg(long = "relight")]
    pub relight: bool,

    /// With `--composite`: run a light img2img pass over the finished composite so
    /// the placed people share the scene's lighting/style (less collage-like).
    /// Strength 0..1 (low keeps identity; ~0.3 is a good blend).
    #[arg(long = "harmonize", value_name = "STRENGTH", num_args = 0..=1, default_missing_value = "0.3")]
    pub harmonize: Option<f32>,

    /// Pin each figure's position + pose with a synthetic **OpenPose ControlNet**
    /// (one skeleton per persona region) during scene generation. Fixes the
    /// persona↔figure binding so the right identity lands on the right figure.
    /// Use with `--swap`.
    #[arg(long = "pose")]
    pub pose: bool,

    /// After `--swap`, run a light ADetailer-style detail pass on the swapped
    /// faces to sharpen small scene faces. OFF by default — it's a low-strength
    /// img2img and can slightly drift the swapped identity.
    #[arg(long = "restore-faces")]
    pub restore_faces: bool,

    /// Use **face-swap** for identity (secondary): generate one coherent scene,
    /// then swap each detected face with that persona's source identity (SCRFD +
    /// ArcFace + inswapper). Far stronger likeness than the IP-Adapter region
    /// path. Needs the inswapper / ArcFace weights (auto-resolved or via
    /// `PLAKAT_INSWAPPER_WEIGHTS` / `PLAKAT_ARCFACE_WEIGHTS`).
    #[arg(long = "swap")]
    pub swap: bool,

    /// Resolve placement (+ print the plan) without loading models or generating.
    #[arg(long = "dry-run")]
    pub dry_run: bool,
}

/// Split `"label:rest"` on the first colon.
fn split_label(s: &str, flag: &str) -> Result<(String, String)> {
    s.split_once(':')
        .map(|(l, r)| (l.trim().to_string(), r.trim().to_string()))
        .with_context(|| format!("--{flag} {s:?} must be LABEL:VALUE"))
}

fn parse_bbox(s: &str) -> Result<[f32; 4]> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    let nums: Vec<f32> = inner
        .split(',')
        .map(|t| t.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("bbox {s:?}: expected [x0,y0,x1,y1] floats"))?;
    if nums.len() != 4 {
        bail!("bbox {s:?}: expected 4 numbers, got {}", nums.len());
    }
    let b = [nums[0], nums[1], nums[2], nums[3]];
    if !(b[0] < b[2] && b[1] < b[3] && b.iter().all(|&v| (0.0..=1.0).contains(&v))) {
        bail!("bbox {s:?}: need x0<x1, y0<y1, all in [0,1]");
    }
    Ok(b)
}

pub async fn run(args: MultipersonArgs, device: Device) -> Result<()> {
    // ── people ──
    if args.person.is_empty() {
        bail!("multiperson needs at least one --person entry");
    }
    let mut people: Vec<Person> = Vec::with_capacity(args.person.len());
    for spec in &args.person {
        let (label, photospec) = split_label(spec, "person")?;
        let photos: Vec<WeightedPhoto> = photospec
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| WeightedPhoto::from_str(t.trim()))
            .collect::<Result<_, _>>()
            .with_context(|| format!("--person {spec:?}: bad photo spec"))?;
        if photos.is_empty() {
            bail!("--person {spec:?}: no photo given");
        }
        for ph in &photos {
            if !ph.path.exists() {
                bail!("person {label:?}: photo not found at {}", ph.path.display());
            }
        }
        if people.iter().any(|p| p.label == label) {
            bail!("duplicate person label {label:?}");
        }
        people.push(Person {
            label,
            photos,
            placement: None,
            bbox: None,
            prompt: None,
            face_strength: None,
            face_bbox: None,
            face_landmarks: None,
            scale: None,
        });
    }

    let find = |label: &str, people: &mut Vec<Person>, flag: &str| -> Result<usize> {
        people
            .iter()
            .position(|p| p.label == label)
            .with_context(|| {
                let known: Vec<&str> = people.iter().map(|p| p.label.as_str()).collect();
                format!("--{flag} {label:?} — no person {label:?} (defined: {})", known.join(", "))
            })
    };

    for spec in &args.at {
        let (label, loc) = split_label(spec, "at")?;
        let i = find(&label, &mut people, "at")?;
        people[i].placement = Some(Placement::parse(&loc)?);
    }
    for spec in &args.bbox {
        let (label, b) = split_label(spec, "bbox")?;
        let i = find(&label, &mut people, "bbox")?;
        people[i].bbox = Some(parse_bbox(&b)?);
    }
    for spec in &args.person_prompt {
        let (label, text) = split_label(spec, "person-prompt")?;
        let i = find(&label, &mut people, "person-prompt")?;
        people[i].prompt = Some(text);
    }
    for spec in &args.face_strength {
        let (label, v) = split_label(spec, "face-strength")?;
        let i = find(&label, &mut people, "face-strength")?;
        let f: f32 = v.parse().with_context(|| format!("face-strength {spec:?}: not a number"))?;
        if !(0.0..=1.0).contains(&f) {
            bail!("face-strength {spec:?}: must be in [0,1]");
        }
        people[i].face_strength = Some(f);
    }
    for spec in &args.scale {
        let (label, v) = split_label(spec, "scale")?;
        let i = find(&label, &mut people, "scale")?;
        let f: f32 = v.parse().with_context(|| format!("scale {spec:?}: not a number"))?;
        if !(0.4..=1.0).contains(&f) {
            bail!("scale {spec:?}: must be in [0.4, 1.0] (1.0 = adult height, ~0.7 = child)");
        }
        people[i].scale = Some(f);
    }

    // ── size ──
    let (width, height) = parse_size(&args.size)?;
    let scheduler = crate::pipelines::scheduler::SchedulerKind::from_str(&args.scheduler)
        .map_err(|e| anyhow::anyhow!("--scheduler {:?}: {e}", args.scheduler))?;

    multiperson::run(MultipersonRequest {
        scene: args.scene,
        people,
        model: args.model,
        identity: args.identity,
        style: args.style,
        negative: args.negative,
        layout_provider: args.layout_provider,
        enhancer: args.enhancer,
        width,
        height,
        steps: args.steps,
        guidance: args.guidance,
        seed: args.seed,
        count: args.count,
        out_dir: args.out,
        scheduler,
        device,
        dry_run: args.dry_run,
        composite: args.composite,
        relight: args.relight,
        harmonize: args.harmonize,
        pose: args.pose,
        swap: args.swap,
        restore_faces: args.restore_faces,
        refine_faces: args.face_refine,
        refine_face_strength: args.refine_face_strength,
        refine_denoise: args.refine_denoise,
    })
    .await
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let s = s.trim().to_ascii_lowercase();
    if let Some((w, h)) = s.split_once('x') {
        Ok((w.trim().parse()?, h.trim().parse()?))
    } else {
        let n: u32 = s.parse().with_context(|| format!("--size {s:?}: expected WxH or N"))?;
        Ok((n, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bbox_and_rejects_bad() {
        assert_eq!(parse_bbox("[0.1,0.2,0.5,0.8]").unwrap(), [0.1, 0.2, 0.5, 0.8]);
        assert!(parse_bbox("[0.5,0.2,0.3,0.8]").is_err()); // x0>x1
        assert!(parse_bbox("[0,0,2,1]").is_err()); // >1
    }

    #[test]
    fn parses_size_forms() {
        assert_eq!(parse_size("1024x768").unwrap(), (1024, 768));
        assert_eq!(parse_size("512").unwrap(), (512, 512));
    }

    #[test]
    fn split_label_splits_first_colon() {
        let (l, r) = split_label("alice:left closer", "at").unwrap();
        assert_eq!(l, "alice");
        assert_eq!(r, "left closer");
    }
}
