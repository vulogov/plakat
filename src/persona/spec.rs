//! `PersonaSpec` v1 — the on-disk schema for a persona (RFC PERSONA-1 §6.2).
//!
//! Design constraints from the RFC:
//!   * P4 — partial specs are valid: **every field is optional**; an absent field is *unknown*.
//!   * P5 — the HJSON file is the source of truth; loading must never fail on a well-formed-but-partial
//!     spec, so enums are carried as `String` (validated later by lint, with nearest-match suggestions)
//!     rather than hard serde enums that would reject unknown values.
//!   * Scalars are `Option<f32>` in `[0,1]` — `None` = unknown, distinct from an explicit `0.5` (§6.4).
//!
//! Loading is `deser_hjson` (the same crate scenario/album specs use). Deliberately permissive: unknown
//! *keys* are ignored (forward-compatible), unknown enum *values* load fine and are caught by lint.

use serde::Deserialize;

/// A colour: a lexicon enum name (`hazel`) or an exact CIELAB triple (`{ lab: [L, a, b] }`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Color {
    Lab { lab: [f32; 3] },
    Named(String),
}

/// A landmark-relative anchor for a localized detail (§8.2). Either an explicit `landmark` + `offset`,
/// or a `region` shorthand (resolved to the region centroid + seeded jitter).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Anchor {
    pub landmark: Option<String>,
    pub offset: Option<[f32; 2]>,
    pub region: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Identity {
    pub name: Option<String>,
    pub display_name: Option<String>,
    pub apparent_age: Option<u32>,
    pub sex: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Jaw {
    pub angle: Option<f32>,
    pub width: Option<f32>,
    pub definition: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Chin {
    pub projection: Option<f32>,
    pub width: Option<f32>,
    pub cleft: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Cheekbones {
    pub height: Option<f32>,
    pub prominence: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Forehead {
    pub height: Option<f32>,
    pub slope: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Face {
    pub shape: Option<String>,
    pub width: Option<f32>,
    pub jaw: Option<Jaw>,
    pub chin: Option<Chin>,
    pub cheekbones: Option<Cheekbones>,
    pub forehead: Option<Forehead>,
    pub temples: Option<f32>,
    pub asymmetry: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Brow {
    pub thickness: Option<f32>,
    pub arch: Option<String>,
    pub length: Option<f32>,
    pub spacing: Option<f32>,
    pub color: Option<Color>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Lashes {
    pub length: Option<f32>,
    pub density: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct UnderEye {
    pub hollow: Option<f32>,
    pub lines: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Eyes {
    pub color: Option<Color>,
    /// `none`, or `{ left: <color>, right: <color> }` — carried loosely; lint checks consistency.
    pub heterochromia: Option<serde_json::Value>,
    pub shape: Option<String>,
    pub size: Option<f32>,
    pub spacing: Option<f32>,
    pub canthal_tilt: Option<f32>,
    pub hood: Option<f32>,
    pub sclera_show: Option<f32>,
    pub lashes: Option<Lashes>,
    pub brow: Option<Brow>,
    pub under_eye: Option<UnderEye>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct NoseBridge {
    pub width: Option<f32>,
    pub height: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct NoseTip {
    pub projection: Option<f32>,
    pub rotation: Option<f32>,
    pub width: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Nostrils {
    pub width: Option<f32>,
    pub flare: Option<f32>,
    pub visibility: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Nose {
    pub profile: Option<String>,
    pub length: Option<f32>,
    pub bridge: Option<NoseBridge>,
    pub tip: Option<NoseTip>,
    pub nostrils: Option<Nostrils>,
    pub columella: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Philtrum {
    pub length: Option<f32>,
    pub depth: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Mouth {
    pub width: Option<f32>,
    pub upper_lip: Option<f32>,
    pub lower_lip: Option<f32>,
    pub cupids_bow: Option<String>,
    pub corners: Option<f32>,
    pub philtrum: Option<Philtrum>,
    pub lip_texture: Option<f32>,
    /// `auto` (derive from skin) or an enum / lab colour.
    pub lip_color: Option<Color>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ToothFeature {
    pub kind: Option<String>,
    pub tooth: Option<String>,
    pub size: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Teeth {
    pub visibility: Option<String>,
    pub alignment: Option<String>,
    pub diastema: Option<f32>,
    pub shade: Option<f32>,
    pub shade_uniformity: Option<f32>,
    pub size: Option<f32>,
    pub proportion: Option<f32>,
    pub gum_show: Option<f32>,
    pub wear: Option<f32>,
    pub features: Option<Vec<ToothFeature>>,
    pub appliance: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Ears {
    pub size: Option<f32>,
    pub protrusion: Option<f32>,
    pub lobe: Option<String>,
    pub shape: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SkinLines {
    pub forehead: Option<f32>,
    pub nasolabial: Option<f32>,
    pub crows_feet: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Flush {
    pub region: Option<String>,
    pub intensity: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Skin {
    pub tone: Option<String>,
    pub undertone: Option<String>,
    pub texture: Option<f32>,
    pub complexion: Option<f32>,
    pub lines: Option<SkinLines>,
    pub pores: Option<f32>,
    pub flush: Option<Flush>,
}

/// A localized mark (§8) — a flat, permissive union over the mark kinds (freckles / mole / scar /
/// birthmark / tattoo / …). `kind` selects; the applicable fields are populated. Extra/unknown fields
/// are ignored so new kinds need no code here.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Mark {
    pub kind: Option<String>,
    // distributional (freckles/pockmark fields)
    pub region: Option<String>,
    pub density: Option<f32>,
    // positional
    pub anchor: Option<Anchor>,
    pub size: Option<f32>,
    pub color: Option<Color>,
    // mole
    pub raised: Option<f32>,
    pub hairs: Option<bool>,
    // scar
    pub form: Option<String>,
    pub length: Option<f32>,
    pub width: Option<f32>,
    pub orientation: Option<f32>,
    pub maturity: Option<f32>,
    pub relief: Option<f32>,
    pub hair_interruption: Option<bool>,
    // birthmark
    pub aspect: Option<f32>,
    pub edge: Option<String>,
    pub intensity: Option<f32>,
    // tattoo
    pub motif: Option<String>,
    pub age: Option<f32>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Piercing {
    pub site: Option<String>,
    pub count: Option<u32>,
    pub gauge: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct JewelryItem {
    pub kind: Option<String>,
    pub site: Option<String>,
    pub style: Option<String>,
    pub size: Option<f32>,
    pub metal: Option<String>,
    pub stone: Option<String>,
    pub length: Option<f32>,
    pub frame: Option<String>,
    pub thickness: Option<f32>,
    pub tint: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Jewelry {
    pub identity_locked: Option<bool>,
    pub items: Option<Vec<JewelryItem>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Hairline {
    pub height: Option<f32>,
    pub shape: Option<String>,
    pub recession: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Hair {
    pub color: Option<Color>,
    pub color_variation: Option<f32>,
    pub greying: Option<f32>,
    pub length: Option<String>,
    pub texture: Option<String>,
    pub density: Option<f32>,
    pub style: Option<String>,
    pub hairline: Option<Hairline>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct FacialHair {
    pub style: Option<String>,
    pub density: Option<f32>,
    pub color: Option<Color>,
    pub length: Option<f32>,
    pub greying: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Neck {
    pub length: Option<f32>,
    pub thickness: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Hands {
    pub size: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Figure {
    pub height_cm: Option<f32>,
    pub build: Option<String>,
    pub weight_impression: Option<f32>,
    pub shoulders: Option<f32>,
    pub waist: Option<f32>,
    pub limb_length: Option<f32>,
    pub neck: Option<Neck>,
    pub hands: Option<Hands>,
    pub posture: Option<String>,
    pub musculature: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Defaults {
    pub expression: Option<String>,
    pub gaze: Option<String>,
    pub framing: Option<String>,
    pub lighting: Option<String>,
    pub wardrobe: Option<String>,
    /// `all` | `none` | a list of item indices — carried loosely.
    pub jewelry_worn: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Provenance {
    pub method: Option<String>,
    pub lexicon_version: Option<String>,
    pub derived_from: Option<String>,
}

/// The full persona spec. Every section is optional (P4). `schema` is the only field the loader
/// requires by convention; a spec with no `schema:` is treated as `persona/1` with a lint warning.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct PersonaSpec {
    pub schema: Option<String>,
    pub identity: Option<Identity>,
    pub face: Option<Face>,
    pub eyes: Option<Eyes>,
    pub nose: Option<Nose>,
    pub mouth: Option<Mouth>,
    pub teeth: Option<Teeth>,
    pub ears: Option<Ears>,
    pub skin: Option<Skin>,
    /// Absent = unknown (author never said); `Some(vec![])` = asserted "no marks" (§6.4).
    pub marks: Option<Vec<Mark>>,
    pub piercings: Option<Vec<Piercing>>,
    pub jewelry: Option<Jewelry>,
    pub hair: Option<Hair>,
    pub facial_hair: Option<FacialHair>,
    pub figure: Option<Figure>,
    pub defaults: Option<Defaults>,
    pub provenance: Option<Provenance>,
}

/// Current schema major version.
pub const SCHEMA_VERSION: u32 = 1;

impl PersonaSpec {
    /// Load a spec from an HJSON string. Permissive: partial specs load; unknown keys are ignored.
    pub fn from_hjson(text: &str) -> Result<Self, deser_hjson::Error> {
        deser_hjson::from_str(text)
    }

    /// Load from a file path.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading persona spec {}: {e}", path.display()))?;
        Self::from_hjson(&text)
            .map_err(|e| anyhow::anyhow!("parsing persona spec {}: {e}", path.display()))
    }

    /// The declared schema major version (`persona/N`), or `None` if `schema:` is absent/malformed.
    pub fn schema_version(&self) -> Option<u32> {
        self.schema
            .as_deref()
            .and_then(|s| s.strip_prefix("persona/"))
            .and_then(|n| n.trim().parse().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_spec_loads() {
        // P4: a spec with nothing but a schema line must load. (Quoteless HJSON strings run to EOL, so
        // the value must be alone on its line — `persona/1` then a newline, not `persona/1 }`.)
        let s = PersonaSpec::from_hjson("{\n  schema: persona/1\n}\n").unwrap();
        assert_eq!(s.schema_version(), Some(1));
        assert!(s.identity.is_none());
        assert!(s.marks.is_none()); // absent = unknown
    }

    #[test]
    fn partial_spec_loads_with_unknown_keys() {
        // Unknown keys are ignored (forward-compatible); partial sections load. HJSON is brace-based
        // (not indentation-based); quoteless strings run to end-of-line, so one key per line.
        let s = PersonaSpec::from_hjson(
            "{\n  schema: persona/1\n  identity: {\n    name: alice\n    apparent_age: 34\n  }\n\
             eyes: {\n    spacing: 0.62\n    color: hazel\n    future_key: 9\n  }\n  marks: []\n}\n",
        )
        .unwrap();
        assert_eq!(s.identity.unwrap().name.as_deref(), Some("alice"));
        assert_eq!(s.eyes.unwrap().spacing, Some(0.62));
        assert_eq!(s.marks.unwrap().len(), 0); // asserted empty, distinct from absent
    }

    #[test]
    fn color_accepts_name_or_lab() {
        let named = PersonaSpec::from_hjson("{\n  eyes: {\n    color: hazel\n  }\n}\n").unwrap();
        assert!(matches!(named.eyes.unwrap().color, Some(Color::Named(_))));
        let lab = PersonaSpec::from_hjson("{\n  eyes: {\n    color: {\n      lab: [32, 8, 12]\n    }\n  }\n}\n").unwrap();
        assert!(matches!(lab.eyes.unwrap().color, Some(Color::Lab { .. })));
    }

    #[test]
    fn marks_union_populates_by_kind() {
        let s = PersonaSpec::from_hjson(
            "{\n  marks: [\n    {\n      kind: mole\n      anchor: {\n        landmark: left-cheek\n\
             offset: [0.02, -0.03]\n      }\n      size: 0.18\n    }\n    {\n      kind: scar\n\
             form: linear\n      length: 0.16\n      orientation: 68\n      maturity: 0.8\n    }\n  ]\n}\n",
        )
        .unwrap();
        let m = s.marks.unwrap();
        assert_eq!(m[0].kind.as_deref(), Some("mole"));
        assert_eq!(m[0].anchor.as_ref().unwrap().landmark.as_deref(), Some("left-cheek"));
        assert_eq!(m[1].kind.as_deref(), Some("scar"));
        assert_eq!(m[1].length, Some(0.16));
    }
}
