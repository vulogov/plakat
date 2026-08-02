//! Layer 0 validation (RFC BOOKART-1 §15 `lint`). Pure, no weights, no network: schema, vocabulary
//! (with nearest-match suggestions), numeric ranges, page validity, and structural contradictions.
//! `lint` exits non-zero on any [`Level::Error`] so it can gate CI.

use crate::bookart::geometry::page::{named_size_mm, SIZE_VOCAB};
use crate::bookart::lexicon;
use crate::bookart::spec::{BookArtSpec, Ornament};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Error,
    Warn,
    Info,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub level: Level,
    pub path: String,
    pub message: String,
}

impl Finding {
    fn err(path: &str, message: String) -> Self {
        Finding { level: Level::Error, path: path.into(), message }
    }
    fn warn(path: &str, message: String) -> Self {
        Finding { level: Level::Warn, path: path.into(), message }
    }
}

/// Validate a vocabulary field: unknown value → Warn with a nearest-match suggestion (never a hard
/// error — unknown enums still render via defaults; the point is to catch typos).
fn check_vocab(findings: &mut Vec<Finding>, path: &str, value: Option<&str>, vocab: &[&str]) {
    if let Some(v) = value {
        // `radial:8` / `frieze:p4` — validate the head only.
        let head = v.split(':').next().unwrap_or(v);
        if !vocab.contains(&head) {
            let hint = lexicon::nearest(head, vocab).map(|s| format!(" (did you mean `{s}`?)")).unwrap_or_default();
            findings.push(Finding::warn(path, format!("unknown value `{v}`{hint}; known: {}", vocab.join(", "))));
        }
    }
}

fn check_unit(findings: &mut Vec<Finding>, path: &str, value: Option<f32>) {
    if let Some(x) = value {
        if !(0.0..=1.0).contains(&x) {
            findings.push(Finding::err(path, format!("must be in [0,1], got {x}")));
        }
    }
}

fn lint_ornament(findings: &mut Vec<Finding>, path: &str, orn: &Ornament) {
    check_vocab(findings, &format!("{path}.type"), orn.kind.as_deref(), lexicon::ORNAMENTS);
    check_vocab(findings, &format!("{path}.symmetry"), orn.symmetry.as_deref(), lexicon::SYMMETRIES);
    check_vocab(findings, &format!("{path}.tier"), orn.tier.as_deref(), lexicon::TIERS);
    check_unit(findings, &format!("{path}.taper"), orn.taper);
    check_unit(findings, &format!("{path}.fade"), orn.fade);
    if orn.kind.as_deref() == Some("initial") && orn.glyph.is_none() && orn.glyphs.is_none() {
        findings.push(Finding::warn(path, "an `initial` with no `glyph`/`glyphs` has no letter to build around (§6.5)".into()));
    }
}

/// Lint a spec. Returns all findings (Error/Warn/Info), most-severe usefully first is the caller's job.
pub fn lint(spec: &BookArtSpec) -> Vec<Finding> {
    let mut f = Vec::new();

    // schema
    if let Some(s) = &spec.schema {
        if s != crate::bookart::spec::SCHEMA_VERSION {
            f.push(Finding::warn("schema", format!("expected `{}`, got `{s}`", crate::bookart::spec::SCHEMA_VERSION)));
        }
    }

    // top-level vocabulary
    check_vocab(&mut f, "origin", spec.origin.as_deref(), lexicon::ORIGINS);
    check_vocab(&mut f, "technique", spec.technique.as_deref(), lexicon::TECHNIQUES);

    // ink
    if let Some(ink) = &spec.ink {
        check_vocab(&mut f, "ink.transparency", ink.transparency.as_deref(), lexicon::TRANSPARENCY_MODES);
        check_unit(&mut f, "ink.weight", ink.weight);
    }

    // page
    if let Some(p) = &spec.page {
        if let Some(size) = &p.size {
            if size != "custom" && named_size_mm(size).is_none() {
                let hint = lexicon::nearest(size, SIZE_VOCAB).map(|s| format!(" (did you mean `{s}`?)")).unwrap_or_default();
                f.push(Finding::warn("page.size", format!("unknown size `{size}`{hint}; known: {}", SIZE_VOCAB.join(", "))));
            }
            if size == "custom" && p.custom.as_ref().map(|c| c.w_mm.is_none() || c.h_mm.is_none()).unwrap_or(true) {
                f.push(Finding::err("page.custom", "size `custom` needs `custom: { w_mm, h_mm }`".into()));
            }
        }
        if let Some(dpi) = p.dpi {
            if !(72..=1200).contains(&dpi) {
                f.push(Finding::warn("page.dpi", format!("{dpi} DPI is unusual; expected 72..1200 (print default 300)")));
            }
        }
    }

    // structure: an ornament or a kit (not neither, not both)
    match (&spec.ornament, &spec.kit) {
        (None, None) => f.push(Finding::warn("ornament", "no `ornament` and no `kit` — defaults to a `divider` (§6.1)".into())),
        (Some(_), Some(_)) => f.push(Finding::err("ornament", "a spec has either a single `ornament` or a `kit`, not both".into())),
        _ => {}
    }
    if let Some(orn) = &spec.ornament {
        lint_ornament(&mut f, "ornament", orn);
    }
    if let Some(kit) = &spec.kit {
        for (i, orn) in kit.ornaments.iter().flatten().enumerate() {
            lint_ornament(&mut f, &format!("kit.ornaments[{i}]"), orn);
        }
    }

    f
}

/// True if any finding is an error (the CLI exits non-zero on this).
pub fn has_errors(findings: &[Finding]) -> bool {
    findings.iter().any(|x| x.level == Level::Error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_spec_has_no_errors() {
        let spec = BookArtSpec::from_hjson(r#"{"origin":"russian","technique":"woodcut","ornament":{"type":"headpiece"}}"#).unwrap();
        let f = lint(&spec);
        assert!(!has_errors(&f), "{f:?}");
    }

    #[test]
    fn typo_origin_warns_with_suggestion() {
        let spec = BookArtSpec::from_hjson(r#"{"origin":"russsian","ornament":{"type":"divider"}}"#).unwrap();
        let f = lint(&spec);
        assert!(f.iter().any(|x| x.path == "origin" && x.message.contains("russian")));
        assert!(!has_errors(&f)); // vocab typos are warnings, not errors
    }

    #[test]
    fn out_of_range_taper_errors() {
        let spec = BookArtSpec::from_hjson(r#"{"ornament":{"type":"tailpiece","taper":1.8}}"#).unwrap();
        assert!(has_errors(&lint(&spec)));
    }

    #[test]
    fn ornament_and_kit_together_errors() {
        let spec = BookArtSpec::from_hjson(r#"{"ornament":{"type":"divider"},"kit":{"ornaments":[{"type":"headpiece"}]}}"#).unwrap();
        assert!(has_errors(&lint(&spec)));
    }

    #[test]
    fn custom_size_needs_dimensions() {
        let spec = BookArtSpec::from_hjson(r#"{"page":{"size":"custom"},"ornament":{"type":"divider"}}"#).unwrap();
        assert!(has_errors(&lint(&spec)));
    }
}
