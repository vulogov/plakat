//! LAYERED-1 lint + the **size classifier** (RFC §Size classes, §Lint rules). Weight-free, no network, no
//! GPU — safe to gate CI. Classifies each layer (anchored / hinted / lifted) for the chosen FINISH family
//! and reports schema/geometry problems (duplicate ids, invalid or ambiguous boxes, empty prompts, relation
//! words that suggest a merge, a global medium that leaked into a layer prompt).

use crate::compile::{classify_model, ModelFamily};
use crate::layered::plan::{self, Layer, LayerPlan};
use crate::pipelines::noise_space::LatentGeometry;

/// Size class of a layer for the finish family (RFC §Size classes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// `s ≥ k·2^L·v` — gets a guide anchor + verify + repair.
    Anchored,
    /// `u ≤ s < k·2^L·v` — named in the finish prompt, verify/repair, no anchor.
    Hinted,
    /// `s < u` — named in the finish prompt, structure only via the S5 lift.
    Lifted,
}

impl Class {
    pub fn label(self) -> &'static str {
        match self {
            Class::Anchored => "anchored",
            Class::Hinted => "hinted",
            Class::Lifted => "lifted",
        }
    }
}

/// The latent geometry the classifier + guide use for a finish family — matches the per-family `NoiseSpace`
/// impls wired in P0: `v` = VAE/latent pixel size, `u` = denoised-unit size, `L` = low-pass pool depth.
pub fn geometry_for_family(f: ModelFamily) -> LatentGeometry {
    match f {
        // 8× VAE UNet families — unit = an 8-px cell.
        ModelFamily::Sd15 | ModelFamily::Sdxl | ModelFamily::Unknown => LatentGeometry { v: 8, u: 8, pool_levels: 2 },
        // 8× VAE DiT families — unit = a 2×2-patchified token (16 px).
        ModelFamily::Sd3 | ModelFamily::Flux | ModelFamily::PixArt => LatentGeometry { v: 8, u: 16, pool_levels: 2 },
        // Sana: 32× DC-AE, 32-px token, shallow pyramid.
        ModelFamily::Sana => LatentGeometry { v: 32, u: 32, pool_levels: 1 },
        // Cascade stage C: coarse ~42× grid, no pyramid.
        ModelFamily::Cascade => LatentGeometry { v: 42, u: 42, pool_levels: 0 },
    }
}

/// Geometry for a finish MODEL name (via the compile family classifier).
pub fn geometry_for_model(model: &str) -> LatentGeometry {
    geometry_for_family(classify_model(model))
}

/// The anchor threshold in pixels: `k · 2^L · v` with `k = 2` (RFC §Size classes).
fn anchor_threshold(g: &LatentGeometry) -> u32 {
    (2u32 * (1u32 << g.pool_levels) * g.v as u32).max(1)
}

/// Classify a layer's shorter box side (output px) against the finish geometry.
pub fn classify_size(short_side_px: u32, g: &LatentGeometry) -> Class {
    if short_side_px >= anchor_threshold(g) {
        Class::Anchored
    } else if short_side_px >= g.u as u32 {
        Class::Hinted
    } else {
        Class::Lifted
    }
}

/// The resolved class of a layer: an explicit `class:` override wins; else classify by size.
pub fn layer_class(layer: &Layer, g: &LatentGeometry, out_w: u32, out_h: u32) -> Class {
    if let Some(c) = layer.class.as_deref() {
        match c.trim().to_ascii_lowercase().as_str() {
            "anchored" | "anchor" => return Class::Anchored,
            "hinted" | "hint" => return Class::Hinted,
            "lifted" | "lift" => return Class::Lifted,
            _ => {}
        }
    }
    classify_size(plan::box_short_side_px(layer, out_w, out_h), g)
}

/// A lint finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warn,
    Info,
}

#[derive(Clone, Debug)]
pub struct Issue {
    pub severity: Severity,
    pub message: String,
}

impl Issue {
    fn err(m: impl Into<String>) -> Self {
        Self { severity: Severity::Error, message: m.into() }
    }
    fn warn(m: impl Into<String>) -> Self {
        Self { severity: Severity::Warn, message: m.into() }
    }
    fn info(m: impl Into<String>) -> Self {
        Self { severity: Severity::Info, message: m.into() }
    }
}

/// Relation verbs whose presence in a layer prompt suggests the subject interacts with another layer (which
/// should then be ONE layer, not split) — RFC's split rule, checked as a warning.
const RELATION_WORDS: &[&str] =
    &["holding", "wearing", "riding", "carrying", "sitting on", "on top of", "leaning on", "standing on", "hugging"];

fn boxes_overlap(a: &[f32; 4], b: &[f32; 4]) -> bool {
    !(a[2] <= b[0] || b[2] <= a[0] || a[3] <= b[1] || b[3] <= a[1])
}

/// Lint a plan for the given finish geometry + output size. Returns findings (errors first is the caller's
/// job); a per-layer class report is included as `Info`.
pub fn lint(p: &LayerPlan, g: &LatentGeometry, out_w: u32, out_h: u32) -> Vec<Issue> {
    let mut issues = Vec::new();

    if p.layers.is_empty() {
        issues.push(Issue::err("plan has no layers"));
    }

    // Unique ids.
    let mut seen = std::collections::HashSet::new();
    for l in &p.layers {
        if l.id.trim().is_empty() {
            issues.push(Issue::err("a layer has an empty id"));
        } else if !seen.insert(l.id.trim().to_ascii_lowercase()) {
            issues.push(Issue::err(format!("duplicate layer id {:?}", l.id)));
        }
    }

    // Boxes + prompts + class report.
    for l in &p.layers {
        let b = plan::layer_box(l);
        if !(b[0] < b[2] && b[1] < b[3]) {
            issues.push(Issue::err(format!("layer {:?}: box has non-positive area ({b:?})", l.id)));
        }
        if l.prompt.as_deref().map(|s| s.trim().is_empty()).unwrap_or(true) {
            issues.push(Issue::err(format!("layer {:?}: empty prompt", l.id)));
        }
        // Relation words → suggest a merge.
        if let Some(prompt) = &l.prompt {
            let lp = prompt.to_ascii_lowercase();
            if let Some(w) = RELATION_WORDS.iter().find(|w| lp.contains(**w)) {
                issues.push(Issue::warn(format!(
                    "layer {:?} prompt contains {:?} — if it interacts with another layer's subject, keep them in ONE layer",
                    l.id, w
                )));
            }
            // Global medium leaked into a layer prompt (it's stripped from drafts anyway).
            if let Some(medium) = &p.global.medium {
                let m = medium.trim().to_ascii_lowercase();
                if !m.is_empty() && lp.contains(&m) {
                    issues.push(Issue::warn(format!(
                        "layer {:?}: the global medium {:?} appears in the layer prompt — it's stripped from drafts; drop it",
                        l.id, medium
                    )));
                }
            }
        }
        let c = layer_class(l, g, out_w, out_h);
        issues.push(Issue::info(format!("layer {:?}: {} ({}px shorter side)", l.id, c.label(), plan::box_short_side_px(l, out_w, out_h))));
    }

    // Ambiguous z-order: two layers at the same depth with overlapping boxes.
    for (i, a) in p.layers.iter().enumerate() {
        for b in p.layers.iter().skip(i + 1) {
            if (plan::layer_depth(a) - plan::layer_depth(b)).abs() < 1e-4 && boxes_overlap(&plan::layer_box(a), &plan::layer_box(b)) {
                issues.push(Issue::warn(format!(
                    "layers {:?} and {:?} share a depth and overlap — the z-order is ambiguous; give them distinct depths",
                    a.id, b.id
                )));
            }
        }
    }

    issues
}

/// True when the plan has no `Error`-level findings.
pub fn is_clean(issues: &[Issue]) -> bool {
    !issues.iter().any(|i| i.severity == Severity::Error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered::plan::parse;

    fn sd() -> LatentGeometry {
        geometry_for_model("sdxl")
    }

    #[test]
    fn classifier_thresholds() {
        let g = sd(); // v8, u8, L2 → anchor at 2·4·8 = 64
        assert_eq!(classify_size(64, &g), Class::Anchored);
        assert_eq!(classify_size(63, &g), Class::Hinted);
        assert_eq!(classify_size(8, &g), Class::Hinted);
        assert_eq!(classify_size(6, &g), Class::Lifted);
        // Sana anchors from 128; 6px is lifted everywhere.
        let sana = geometry_for_model("sana");
        assert_eq!(classify_size(64, &sana), Class::Hinted, "64px is only hinted on Sana");
        assert_eq!(classify_size(128, &sana), Class::Anchored);
        assert_eq!(classify_size(6, &sana), Class::Lifted);
    }

    #[test]
    fn lint_catches_dups_empties_and_overlap() {
        let text = r#"{
          layers: [
            { id: "a", prompt: "a cat", box: [0.0,0.0,0.5,0.5], depth: 0.5 }
            { id: "a", prompt: "", box: [0.4,0.4,0.9,0.9], depth: 0.5 }
          ]
        }"#;
        let p = parse(text).unwrap();
        let issues = lint(&p, &sd(), 1024, 1024);
        let errs: Vec<_> = issues.iter().filter(|i| i.severity == Severity::Error).collect();
        assert!(errs.iter().any(|i| i.message.contains("duplicate layer id")), "dup id: {issues:?}");
        assert!(errs.iter().any(|i| i.message.contains("empty prompt")), "empty prompt");
        assert!(issues.iter().any(|i| i.severity == Severity::Warn && i.message.contains("z-order")), "overlap+same depth");
        assert!(!is_clean(&issues), "has errors");
    }

    #[test]
    fn lint_warns_on_relation_and_leaked_medium() {
        let text = r#"{
          global: { medium: "oil painting" }
          layers: [
            { id: "boy", prompt: "a boy holding a red umbrella, oil painting", box: [0.1,0.1,0.5,0.9], depth: 0.3 }
          ]
        }"#;
        let p = parse(text).unwrap();
        let issues = lint(&p, &sd(), 1024, 1024);
        assert!(issues.iter().any(|i| i.message.contains("holding")), "relation warn: {issues:?}");
        assert!(issues.iter().any(|i| i.message.contains("global medium")), "leaked medium warn");
        assert!(issues.iter().any(|i| i.severity == Severity::Info && i.message.contains("anchored")), "class report");
        assert!(is_clean(&issues), "warnings only, no errors");
    }
}
