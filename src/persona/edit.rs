//! Attribute-class edit diffing (RFC §6.5). Every persona leaf carries a **class** that determines
//! what an edit invalidates — so the TUI/CLI can report, before saving, *"this change is structural
//! and will invalidate 4 references and 2 adapters"* and make the re-cast opt-in. Pure + deterministic
//! (a function of two HJSON documents + the lexicon); no weights.
//!
//! | Class | Edit invalidates | Repair strategy |
//! |---|---|---|
//! | Structural | the reference set + every baked adapter | full re-cast |
//! | Surface | nothing structural | targeted inpaint / recomposite over the existing set (§12.4) |
//! | Detail | nothing | recomposite only (§8.4) — the cheapest class |
//! | Presentation | nothing | per-render override only |

use crate::persona::lexicon::Lexicon;
use anyhow::Result;
use serde_json::Value;
use std::collections::BTreeMap;

/// The attribute class (§6.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Structural,
    Surface,
    Detail,
    Presentation,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Structural => "structural",
            Class::Surface => "surface",
            Class::Detail => "detail",
            Class::Presentation => "presentation",
        }
    }
    /// What changing an attribute of this class invalidates + the repair strategy.
    pub fn invalidation(self) -> &'static str {
        match self {
            Class::Structural => "invalidates the reference set + baked adapters → full re-cast",
            Class::Surface => "targeted inpaint / recomposite over the existing references",
            Class::Detail => "recomposite only (milliseconds)",
            Class::Presentation => "per-render override only",
        }
    }
}

/// Strip array indices from a flattened path so it can match a lexicon entry:
/// `marks.0.kind` → `marks.kind`, `jewelry.items.1.metal` → `jewelry.items.metal`.
fn canonical(path: &str) -> String {
    path.split('.').filter(|seg| seg.parse::<usize>().is_err()).collect::<Vec<_>>().join(".")
}

/// The class of a leaf `path` (§6.5). Path rules for the detail/presentation collections win over the
/// lexicon; scalar/enum attributes fall back to the lexicon's `class`, then a structural-keyword
/// heuristic for paths the skeleton lexicon does not yet cover.
pub fn class_of(path: &str, lex: &Lexicon) -> Class {
    let c = canonical(path);
    // detail collections — the cheapest class (recomposite only).
    if c.starts_with("marks") || c.starts_with("teeth.features") {
        return Class::Detail;
    }
    // worn jewelry is presentation; `identity_locked` toggling it is a surface config change.
    if c == "jewelry.identity_locked" {
        return Class::Surface;
    }
    if c.starts_with("jewelry") {
        return Class::Presentation;
    }
    // piercing sites are durable → surface.
    if c.starts_with("piercings") {
        return Class::Surface;
    }
    // render presentation.
    if c.starts_with("defaults") || c.starts_with("provenance") {
        return Class::Presentation;
    }
    // teeth: alignment/proportion/size are dentition (structural); shade/wear/etc. are surface.
    if c.starts_with("teeth") {
        return if matches!(c.as_str(), "teeth.alignment" | "teeth.proportion" | "teeth.size") {
            Class::Structural
        } else {
            Class::Surface
        };
    }
    // the lexicon carries the authoritative class for the scalar/enum morphology attributes.
    if let Some(e) = lex.get(&c) {
        return match e.class.as_str() {
            "structural" => Class::Structural,
            "detail" => Class::Detail,
            _ => Class::Surface,
        };
    }
    // heuristic fallback for paths not in the skeleton lexicon.
    const STRUCTURAL_KEYS: &[&str] = &[
        "shape", "width", "spacing", "jaw", "chin", "cheekbone", "bridge", "projection", "canthal",
        "proportion", "alignment", "height_cm", "build", "shoulders", "waist", "limb", "forehead",
        "temples", "asymmetry",
    ];
    if STRUCTURAL_KEYS.iter().any(|k| c.contains(k)) {
        Class::Structural
    } else {
        Class::Surface
    }
}

/// How a leaf changed between two specs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    Added,
    Removed,
    Changed,
}

/// One changed leaf + its class + old/new values.
#[derive(Debug, Clone)]
pub struct ChangedAttr {
    pub path: String,
    pub kind: ChangeKind,
    pub class: Class,
    pub old: Option<String>,
    pub new: Option<String>,
}

/// Flatten a JSON value into `path → scalar-string` leaves (objects recurse by key, arrays by index).
fn flatten(v: &Value, prefix: &str, out: &mut BTreeMap<String, String>) {
    match v {
        Value::Object(m) => {
            for (k, val) in m {
                let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(val, &p, out);
            }
        }
        Value::Array(a) => {
            for (i, val) in a.iter().enumerate() {
                flatten(val, &format!("{prefix}.{i}"), out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.to_string());
        }
    }
}

fn flatten_hjson(text: &str) -> Result<BTreeMap<String, String>> {
    let v: Value = deser_hjson::from_str(text)?;
    let mut out = BTreeMap::new();
    flatten(&v, "", &mut out);
    Ok(out)
}

/// Diff two persona HJSON documents into classified changed leaves (§6.5).
pub fn diff(old_hjson: &str, new_hjson: &str, lex: &Lexicon) -> Result<Vec<ChangedAttr>> {
    let old = flatten_hjson(old_hjson)?;
    let new = flatten_hjson(new_hjson)?;
    let mut paths: Vec<&String> = old.keys().chain(new.keys()).collect();
    paths.sort();
    paths.dedup();
    let mut out = Vec::new();
    for p in paths {
        let (o, n) = (old.get(p), new.get(p));
        let kind = match (o, n) {
            (Some(a), Some(b)) if a == b => continue,
            (Some(_), Some(_)) => ChangeKind::Changed,
            (None, Some(_)) => ChangeKind::Added,
            (Some(_), None) => ChangeKind::Removed,
            (None, None) => continue,
        };
        out.push(ChangedAttr { path: p.clone(), kind, class: class_of(p, lex), old: o.cloned(), new: n.cloned() });
    }
    Ok(out)
}

/// A per-class tally of a diff + what it invalidates.
#[derive(Debug, Clone, Default)]
pub struct DiffSummary {
    pub structural: usize,
    pub surface: usize,
    pub detail: usize,
    pub presentation: usize,
}

impl DiffSummary {
    /// A structural change is the only one that invalidates the reference set + adapters (§6.5).
    pub fn invalidates_references(&self) -> bool {
        self.structural > 0
    }
    pub fn total(&self) -> usize {
        self.structural + self.surface + self.detail + self.presentation
    }
}

/// Tally a diff by class.
pub fn summarize(changes: &[ChangedAttr]) -> DiffSummary {
    let mut s = DiffSummary::default();
    for c in changes {
        match c.class {
            Class::Structural => s.structural += 1,
            Class::Surface => s.surface += 1,
            Class::Detail => s.detail += 1,
            Class::Presentation => s.presentation += 1,
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> Lexicon {
        Lexicon::skeleton()
    }

    #[test]
    fn class_of_by_rules_and_lexicon() {
        let l = lex();
        assert_eq!(class_of("eyes.color", &l), Class::Surface); // lexicon surface
        assert_eq!(class_of("eyes.spacing", &l), Class::Structural); // lexicon structural
        assert_eq!(class_of("marks.0.kind", &l), Class::Detail);
        assert_eq!(class_of("jewelry.items.0.metal", &l), Class::Presentation);
        assert_eq!(class_of("jewelry.identity_locked", &l), Class::Surface);
        assert_eq!(class_of("piercings.0.site", &l), Class::Surface);
        assert_eq!(class_of("defaults.expression", &l), Class::Presentation);
        assert_eq!(class_of("teeth.alignment", &l), Class::Structural);
        assert_eq!(class_of("teeth.shade", &l), Class::Surface);
        // heuristic fallback for a non-lexicon structural path.
        assert_eq!(class_of("figure.height_cm", &l), Class::Structural);
    }

    #[test]
    fn diff_classifies_a_surface_and_a_structural_change() {
        let old = "{ schema: \"persona/1\"\n eyes: { color: \"hazel\", spacing: 0.5 } }";
        let new = "{ schema: \"persona/1\"\n eyes: { color: \"blue\", spacing: 0.8 } }";
        let d = diff(old, new, &lex()).unwrap();
        let by = |p: &str| d.iter().find(|c| c.path == p).unwrap();
        assert_eq!(by("eyes.color").class, Class::Surface);
        assert_eq!(by("eyes.color").kind, ChangeKind::Changed);
        assert_eq!(by("eyes.spacing").class, Class::Structural);
        let s = summarize(&d);
        assert!(s.invalidates_references(), "the spacing change is structural");
        assert_eq!(s.structural, 1);
        assert_eq!(s.surface, 1);
    }

    #[test]
    fn detail_only_edit_does_not_invalidate_references() {
        let old = "{ schema: \"persona/1\"\n marks: [ { kind: \"mole\" } ] }";
        let new = "{ schema: \"persona/1\"\n marks: [ { kind: \"scar\" } ] }";
        let d = diff(old, new, &lex()).unwrap();
        let s = summarize(&d);
        assert!(!s.invalidates_references(), "a mark edit is recomposite-only");
        assert_eq!(s.detail, 1);
    }

    #[test]
    fn added_and_removed_are_detected() {
        let old = "{ schema: \"persona/1\"\n eyes: { color: \"hazel\" } }";
        let new = "{ schema: \"persona/1\"\n eyes: { color: \"hazel\" }\n hair: { color: \"auburn\" } }";
        let d = diff(old, new, &lex()).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].path, "hair.color");
        assert_eq!(d[0].kind, ChangeKind::Added);
    }
}
