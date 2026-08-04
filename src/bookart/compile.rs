//! Layer 1 — the resolver (RFC BOOKART-1 §5.2). `resolve(spec) → RenderPlan` is a **pure, byte-stable**
//! function: it fills every unset field from the lexicon defaults, decides the render tier, builds the
//! diffusion prompt/negative (with the anti-text guard baked in — the persona lesson), and resolves the
//! print canvas. No weights, no I/O, no RNG. Golden-tested.

use crate::bookart::geometry::{resolve_page, PageResolved};
use crate::bookart::lexicon;
use crate::bookart::spec::{BookArtSpec, Ornament};

/// The anti-text guard (§9 / RFC §2.1.5): an SD-UNet scrawls fake letterforms into ornament space,
/// which is fatal for book ornament. Always in the negative on the diffusion/composite tiers.
pub const ANTI_TEXT: &str = "text, letters, words, watermark, signature, caption, gibberish writing";
/// Keep the render actually black-and-white line ornament, not a tinted grey photo.
pub const ANTI_COLOR: &str = "color, colour, photograph, grey wash, gradient shading, soft focus, 3d render, blurry";

/// A fully-resolved plan for one ornament: everything a downstream phase needs, deterministic.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RenderPlan {
    pub schema_ok: bool,
    pub origin: String,
    pub technique: String,
    pub motif: Vec<String>,
    pub ornament_kind: String,
    pub tier: String, // resolved concrete tier (auto → procedural|diffusion|composite)
    pub symmetry: String,
    pub page: PageResolved,
    pub transparent: bool,
    pub transparency_mode: String,
    /// Edge fade for vignette/spot art, `[0,1]` (from `ornament.fade`).
    pub fade: f32,
    pub binariser: String,
    pub ink_color: String,
    pub ink_weight: f32,
    pub tint: String,
    pub formats: Vec<String>, // png always first
    /// Diffusion/composite prompt (empty for a pure `procedural` ornament).
    pub prompt: String,
    /// Diffusion/composite negative (anti-text + anti-colour), empty for pure `procedural`.
    pub negative: String,
}

fn resolve_tier(orn: &Ornament, kind: &str) -> String {
    match orn.tier.as_deref().unwrap_or("auto") {
        "auto" => lexicon::default_tier(kind).to_string(),
        t => t.to_string(),
    }
}

/// Build the ornament's diffusion prompt from origin scaffold + technique cue + type + motif.
fn build_prompt(origin: &str, technique: &str, kind: &str, motif: &[String], orn: &Ornament) -> String {
    let (scaffold, _, _) = lexicon::origin_scaffold_dyn(origin);
    let tech = lexicon::technique_prompt(technique);
    let subject = orn.prompt.clone().unwrap_or_else(|| format!("a {kind} ornament"));
    let motif_clause = if motif.is_empty() { String::new() } else { format!(", featuring {}", motif.join(" and ")) };
    // Subject-first, then style scaffolds, then the hard B/W + plain-background anchor.
    format!("{subject}{motif_clause}, {scaffold}, {tech}, black and white, on a plain white background")
}

/// Resolve a single-ornament spec to a byte-stable [`RenderPlan`].
pub fn resolve(spec: &BookArtSpec) -> RenderPlan {
    let orn = spec.ornament_or_default();
    let kind = orn.kind.clone().unwrap_or_else(|| "divider".into());

    let origin = spec.origin.clone().unwrap_or_else(|| "generic".into());
    let (_, default_tech, default_motifs) = lexicon::origin_scaffold_dyn(&origin);
    let technique = spec.technique.clone().unwrap_or(default_tech);

    // motif: per-ornament override, else top-level, else the origin's defaults.
    let motif = orn
        .motif
        .clone()
        .or_else(|| spec.motif.clone())
        .unwrap_or(default_motifs);

    let tier = resolve_tier(&orn, &kind);
    let symmetry = orn.symmetry.clone().unwrap_or_else(|| lexicon::default_symmetry(&kind).to_string());
    let page = resolve_page(spec.page.as_ref());

    let ink = spec.ink.clone().unwrap_or_default();
    let transparency_mode = ink.transparency.clone().unwrap_or_else(|| "luminance".into());
    let ink_color = ink.color.clone().unwrap_or_else(|| "black".into());
    let ink_weight = ink.weight.unwrap_or(0.6).clamp(0.0, 1.0);
    let binariser = lexicon::technique_binariser(&technique).to_string();

    let transparent = spec.transparent.unwrap_or(true);
    let out = spec.output.clone().unwrap_or_default();
    let tint = out.tint.clone().unwrap_or_else(|| ink_color.clone());
    // png is always first (primary); keep any opt-in extras, de-duped, order-stable.
    let mut formats = vec!["png".to_string()];
    if let Some(fs) = out.formats {
        for f in fs {
            if f != "png" && !formats.contains(&f) {
                formats.push(f);
            }
        }
    }

    // Prompt only for the tiers that sample; a pure procedural ornament needs none.
    let (prompt, negative) = if tier == "procedural" {
        (String::new(), String::new())
    } else {
        (build_prompt(&origin, &technique, &kind, &motif, &orn), format!("{ANTI_COLOR}, {ANTI_TEXT}"))
    };

    RenderPlan {
        schema_ok: spec.schema.as_deref().map(|s| s == crate::bookart::spec::SCHEMA_VERSION).unwrap_or(true),
        origin,
        technique,
        motif,
        ornament_kind: kind,
        tier,
        symmetry,
        page,
        transparent,
        transparency_mode,
        fade: orn.fade.unwrap_or(0.0).clamp(0.0, 1.0),
        binariser,
        ink_color,
        ink_weight,
        tint,
        formats,
        prompt,
        negative,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_spec_resolves_to_a_procedural_divider() {
        let plan = resolve(&BookArtSpec::default());
        assert_eq!(plan.ornament_kind, "divider");
        assert_eq!(plan.tier, "procedural"); // divider is geometric
        assert_eq!(plan.origin, "generic");
        assert_eq!(plan.symmetry, "bilateral");
        assert_eq!(plan.transparent, true);
        assert_eq!(plan.transparency_mode, "luminance");
        assert_eq!(plan.formats, vec!["png"]);
        assert!(plan.prompt.is_empty(), "procedural tier needs no prompt");
    }

    #[test]
    fn golden_russian_woodcut_headpiece() {
        let spec = BookArtSpec::from_hjson(
            r#"{"schema":"bookart/1","origin":"russian","technique":"woodcut","motif":["firebird","oak-leaf"],"page":{"size":"a5","dpi":300},"ornament":{"type":"headpiece"}}"#,
        )
        .unwrap();
        let plan = resolve(&spec);
        assert_eq!(plan.tier, "composite"); // headpiece defaults to composite
        assert_eq!(plan.symmetry, "bilateral");
        assert_eq!(plan.binariser, "threshold-bold");
        assert_eq!((plan.page.w_px, plan.page.h_px), (1748, 2480));
        assert_eq!(
            plan.prompt,
            "a headpiece ornament, featuring firebird and oak-leaf, in the tradition of Russian folk book illustration, Bilibin lubok ornament, bold woodcut, high contrast, black and white, on a plain white background"
        );
        assert_eq!(plan.negative, "color, colour, photograph, grey wash, gradient shading, soft focus, 3d render, blurry, text, letters, words, watermark, signature, caption, gibberish writing");
    }

    #[test]
    fn resolve_is_deterministic() {
        let spec = BookArtSpec::from_hjson(r#"{"origin":"english","ornament":{"type":"vignette","prompt":"a wolf in a forest"}}"#).unwrap();
        assert_eq!(resolve(&spec), resolve(&spec));
        // vignette is pictorial → diffusion, prompt present.
        assert_eq!(resolve(&spec).tier, "diffusion");
        assert!(resolve(&spec).prompt.starts_with("a wolf in a forest"));
    }

    #[test]
    fn png_always_first_extras_kept() {
        let spec = BookArtSpec::from_hjson(r#"{"output":{"formats":["svg","png","pdf"]}}"#).unwrap();
        assert_eq!(resolve(&spec).formats, vec!["png", "svg", "pdf"]);
    }
}
