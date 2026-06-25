//! Scenario / scripting integration for `multiperson` — the `type: multiperson`
//! task and the `plakat.multiperson` word both build a [`MultipersonTaskSpec`]
//! and resolve it (via [`build_request`]) into the same
//! [`MultipersonRequest`](super::MultipersonRequest) the CLI uses, so all three
//! surfaces dispatch the identical pipeline.

use anyhow::{Context, Result, bail};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::pipelines::ip_adapter::{IdentityKind, WeightedPhoto};
use crate::pipelines::scheduler::SchedulerKind;

use super::{MultipersonRequest, Person, Placement};

/// One persona placed into the scene, by name (the persona is defined once at the
/// scenario level / passed alongside in scripting) plus its relative location.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersonRef {
    /// Persona name (resolved to photos by the caller).
    pub persona: String,
    /// Relative location, e.g. `"left closer front"`. Omit → auto-placed.
    #[serde(default)]
    pub at: Option<String>,
    /// Per-persona behavioural prompt.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Figure-height scale for `--pose` (child < 1.0).
    #[serde(default)]
    pub scale: Option<f32>,
}

/// A `multiperson` task / word: the scene + the placed people + the identity mode.
/// Mirrors the CLI flags; everything serde-defaults so existing scenarios are
/// untouched.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MultipersonTaskSpec {
    /// Scene prompt (a `// ` splits an inline style clause).
    pub scene: String,
    /// The placed people (≥1).
    pub people: Vec<PersonRef>,
    /// Identity path: `swap` (face-swap), else `composite` (matted photos).
    #[serde(default)]
    pub swap: bool,
    #[serde(default)]
    pub composite: bool,
    /// Pin figures with a synthetic OpenPose ControlNet (swap path).
    #[serde(default)]
    pub pose: bool,
    /// Light img2img harmonise over the composite (composite path).
    #[serde(default)]
    pub harmonize: Option<f32>,
    /// Detail pass on swapped faces (swap path).
    #[serde(default, rename = "restore-faces")]
    pub restore_faces: bool,
    /// Identity strategy for the (non-swap) inpaint path.
    #[serde(default)]
    pub identity: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub guidance: Option<f64>,
    #[serde(default)]
    pub negative: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// The scripting / standalone form of a multiperson spec: a [`MultipersonTaskSpec`]
/// plus an inline `personas` table (name → reference photo path), so a Bund script
/// can describe a whole people-in-scene composition in one self-contained file.
/// (The scenario surface instead resolves personas from its top-level `personas`
/// block, so it doesn't embed this table.)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MultipersonScriptSpec {
    #[serde(flatten)]
    pub task: MultipersonTaskSpec,
    /// `name → reference photo path` (one photo per persona for the script form).
    #[serde(default)]
    pub personas: std::collections::BTreeMap<String, PathBuf>,
}

impl MultipersonScriptSpec {
    /// Resolve a persona name to its single weighted photo from the inline table.
    pub fn resolve(&self, name: &str) -> Option<Vec<WeightedPhoto>> {
        self.personas
            .get(name)
            .map(|p| vec![WeightedPhoto::single(p.clone())])
    }
}

/// Build a [`MultipersonRequest`] from the spec. `resolve_persona` maps a persona
/// name to its weighted photos (the scenario resolves these from its `personas`
/// block; scripting passes a closure over its own persona table). `defaults`
/// supplies the scenario/CLI-level model + size fallbacks.
pub fn build_request(
    spec: &MultipersonTaskSpec,
    resolve_persona: impl Fn(&str) -> Option<Vec<WeightedPhoto>>,
    out_dir: PathBuf,
    device: Device,
    default_model: &str,
    dry_run: bool,
) -> Result<MultipersonRequest> {
    if spec.people.is_empty() {
        bail!("multiperson task: `people` must list at least one persona");
    }
    let mut people = Vec::with_capacity(spec.people.len());
    for r in &spec.people {
        let photos = resolve_persona(&r.persona)
            .with_context(|| format!("multiperson task: unknown persona {:?}", r.persona))?;
        if photos.is_empty() {
            bail!("multiperson task: persona {:?} has no photo", r.persona);
        }
        let placement = match &r.at {
            Some(s) => Some(Placement::parse(s)?),
            None => None,
        };
        people.push(Person {
            label: r.persona.clone(),
            photos,
            placement,
            bbox: None,
            prompt: r.prompt.clone(),
            face_strength: None,
            face_bbox: None,
            face_landmarks: None,
            scale: r.scale,
        });
    }

    // Defaults mirror the `plakat multiperson` CLI flags exactly, so an identical
    // explicit spec yields an identical request on all three surfaces (parity).
    let (width, height) = parse_size(spec.size.as_deref().unwrap_or("768x768"))?;
    let identity: IdentityKind = match &spec.identity {
        Some(s) => s.parse().map_err(|e| anyhow::anyhow!("identity {s:?}: {e}"))?,
        None => IdentityKind::PlusFace,
    };

    Ok(MultipersonRequest {
        scene: spec.scene.clone(),
        people,
        model: spec.model.clone().unwrap_or_else(|| default_model.to_string()),
        identity,
        style: spec.style.clone(),
        negative: spec.negative.clone().unwrap_or_default(),
        layout_provider: "none".to_string(),
        enhancer: None,
        width,
        height,
        steps: spec.steps.unwrap_or(30),
        guidance: spec.guidance.unwrap_or(7.5),
        seed: spec.seed,
        count: 1,
        out_dir,
        scheduler: SchedulerKind::default(),
        device,
        dry_run,
        composite: spec.composite,
        relight: false,
        harmonize: spec.harmonize,
        pose: spec.pose,
        swap: spec.swap,
        restore_faces: spec.restore_faces,
        refine_faces: false,
        refine_face_strength: 0.85,
        refine_denoise: 0.35,
    })
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let s = s.trim().to_ascii_lowercase();
    if let Some((w, h)) = s.split_once('x') {
        Ok((w.trim().parse()?, h.trim().parse()?))
    } else {
        let n: u32 = s.parse().with_context(|| format!("size {s:?}: expected WxH or N"))?;
        Ok((n, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn photo(name: &str) -> Vec<WeightedPhoto> {
        vec![WeightedPhoto { path: Path::new(name).to_path_buf(), weight: None }]
    }

    #[test]
    fn builds_request_from_spec() {
        let spec = MultipersonTaskSpec {
            scene: "two friends at a cafe // watercolor".into(),
            people: vec![
                PersonRef { persona: "alice".into(), at: Some("left closer front".into()), prompt: None, scale: None },
                PersonRef { persona: "bob".into(), at: None, prompt: Some("smiling".into()), scale: Some(0.7) },
            ],
            swap: true,
            pose: true,
            ..MultipersonTaskSpec {
                scene: String::new(), people: vec![], swap: false, composite: false, pose: false,
                harmonize: None, restore_faces: false, identity: None, model: None, size: None,
                steps: None, guidance: None, negative: None, style: None, seed: None,
            }
        };
        let req = build_request(
            &spec,
            |n| Some(photo(n)),
            Path::new("/tmp").to_path_buf(),
            candle_core::Device::Cpu,
            "sdxl",
            true,
        )
        .unwrap();
        assert_eq!(req.people.len(), 2);
        assert!(req.swap && req.pose);
        assert_eq!(req.people[1].scale, Some(0.7));
        assert!(req.people[0].placement.is_some());
        assert!(req.people[1].placement.is_none()); // auto-placed
    }

    #[test]
    fn defaults_mirror_the_cli_flags() {
        // An all-default spec must reproduce the `plakat multiperson` CLI defaults
        // (768x768, plus-face, guidance 7.5, 30 steps) so the three surfaces agree.
        let spec = MultipersonTaskSpec {
            scene: "x".into(),
            people: vec![PersonRef { persona: "a".into(), at: None, prompt: None, scale: None }],
            swap: false, composite: false, pose: false, harmonize: None, restore_faces: false,
            identity: None, model: None, size: None, steps: None, guidance: None,
            negative: None, style: None, seed: None,
        };
        let req = build_request(&spec, |n| Some(photo(n)), "/tmp".into(), candle_core::Device::Cpu, "sd15", true).unwrap();
        assert_eq!((req.width, req.height), (768, 768));
        assert_eq!(req.steps, 30);
        assert!((req.guidance - 7.5).abs() < 1e-9);
        assert!(matches!(req.identity, IdentityKind::PlusFace));
        assert_eq!(req.model, "sd15");
    }

    #[test]
    fn scenario_and_script_forms_build_identical_requests() {
        // Same explicit inputs through the scenario-form resolver (a persona table)
        // and the scripting-form resolver (the inline `personas` map) must produce
        // byte-identical requests — the parity contract behind all three surfaces.
        let task = MultipersonTaskSpec {
            scene: "two friends // ink".into(),
            people: vec![
                PersonRef { persona: "alice".into(), at: Some("left".into()), prompt: None, scale: None },
                PersonRef { persona: "bob".into(), at: None, prompt: Some("waving".into()), scale: Some(0.8) },
            ],
            swap: true, composite: false, pose: true, harmonize: None, restore_faces: true,
            identity: None, model: Some("sdxl".into()), size: Some("1024x768".into()),
            steps: Some(24), guidance: Some(6.0), negative: Some("blurry".into()),
            style: None, seed: Some(7),
        };
        // scenario form: a map<name, photos>
        let scen = |n: &str| Some(photo(n));
        let a = build_request(&task, scen, "/tmp/o".into(), candle_core::Device::Cpu, "sd15", true).unwrap();
        // script form: inline personas table → same photos
        let script = MultipersonScriptSpec {
            task: task.clone(),
            personas: [("alice".to_string(), Path::new("alice").to_path_buf()),
                       ("bob".to_string(), Path::new("bob").to_path_buf())]
                .into_iter().collect(),
        };
        let b = build_request(&task, |n| script.resolve(n), "/tmp/o".into(), candle_core::Device::Cpu, "sd15", true).unwrap();

        assert_eq!(a.scene, b.scene);
        assert_eq!(a.model, b.model);
        assert_eq!((a.width, a.height), (b.width, b.height));
        assert_eq!(a.steps, b.steps);
        assert_eq!(a.guidance, b.guidance);
        assert_eq!(a.swap, b.swap);
        assert_eq!(a.pose, b.pose);
        assert_eq!(a.restore_faces, b.restore_faces);
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.negative, b.negative);
        assert_eq!(a.people.len(), b.people.len());
        for (pa, pb) in a.people.iter().zip(&b.people) {
            assert_eq!(pa.label, pb.label);
            assert_eq!(pa.scale, pb.scale);
            assert_eq!(pa.prompt, pb.prompt);
            assert_eq!(pa.placement.is_some(), pb.placement.is_some());
            assert_eq!(pa.photos.len(), pb.photos.len());
        }
    }

    #[test]
    fn empty_people_errors() {
        let spec = MultipersonTaskSpec {
            scene: "x".into(), people: vec![], swap: false, composite: false, pose: false,
            harmonize: None, restore_faces: false, identity: None, model: None, size: None,
            steps: None, guidance: None, negative: None, style: None, seed: None,
        };
        assert!(build_request(&spec, |_| None, "/tmp".into(), candle_core::Device::Cpu, "sdxl", true).is_err());
    }
}
