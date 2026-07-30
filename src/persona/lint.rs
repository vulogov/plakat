//! `plakat persona lint` — validate a spec without weights or network (RFC §6.6).
//!
//! This first slice covers what needs no lexicon: schema version, scalar ranges (a core set), and
//! specific contradictions. Full lexicon-driven validation — unknown
//! enum values with nearest-match suggestions, complete range coverage, budget/controllability/
//! occlusion/manifestation warnings — lands with the lexicon (next P0 task). `lint` returns a non-zero
//! exit on any error so it can gate CI.

use super::spec::PersonaSpec;

/// Severity of a lint finding. Errors make `lint` exit non-zero; warnings/info do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub level: Level,
    pub path: String,
    pub message: String,
}

impl Finding {
    fn err(path: &str, msg: impl Into<String>) -> Self {
        Finding { level: Level::Error, path: path.into(), message: msg.into() }
    }
    fn warn(path: &str, msg: impl Into<String>) -> Self {
        Finding { level: Level::Warning, path: path.into(), message: msg.into() }
    }
    #[allow(dead_code)] // used by the manifestation / budget findings the lexicon adds next
    fn info(path: &str, msg: impl Into<String>) -> Self {
        Finding { level: Level::Info, path: path.into(), message: msg.into() }
    }
}

/// Run all lint checks. Returns findings ordered errors-first.
pub fn lint(spec: &PersonaSpec) -> Vec<Finding> {
    let mut f = Vec::new();
    schema(spec, &mut f);
    ranges(spec, &mut f);
    contradictions(spec, &mut f);
    f.sort_by_key(|x| match x.level {
        Level::Error => 0,
        Level::Warning => 1,
        Level::Info => 2,
    });
    f
}

/// True if any finding is an error.
pub fn has_errors(findings: &[Finding]) -> bool {
    findings.iter().any(|x| x.level == Level::Error)
}

fn schema(spec: &PersonaSpec, f: &mut Vec<Finding>) {
    match spec.schema_version() {
        None => f.push(Finding::warn("schema", "missing or malformed `schema:` — assuming persona/1")),
        Some(v) if v > super::SCHEMA_VERSION => f.push(Finding::err(
            "schema",
            format!("schema persona/{v} is newer than this build (persona/{}); upgrade plakat", super::SCHEMA_VERSION),
        )),
        Some(_) => {}
    }
}

/// Range-check the `[0,1]` scalars. This covers the core structural/surface scalars; the lexicon adds
/// complete coverage + the `[0.05,0.95]` beyond-controllability warning.
fn ranges(spec: &PersonaSpec, f: &mut Vec<Finding>) {
    let mut chk = |path: &str, v: Option<f32>| {
        if let Some(x) = v {
            if !(0.0..=1.0).contains(&x) {
                f.push(Finding::err(path, format!("scalar {x} out of range [0,1]")));
            } else if !(0.05..=0.95).contains(&x) {
                f.push(Finding::warn(path, format!("scalar {x} is beyond typical controllability")));
            }
        }
    };
    if let Some(fc) = &spec.face {
        chk("face.width", fc.width);
        chk("face.temples", fc.temples);
        chk("face.asymmetry", fc.asymmetry);
        if let Some(j) = &fc.jaw {
            chk("face.jaw.angle", j.angle);
            chk("face.jaw.width", j.width);
        }
        if let Some(c) = &fc.chin {
            chk("face.chin.projection", c.projection);
            chk("face.chin.width", c.width);
        }
        if let Some(cb) = &fc.cheekbones {
            chk("face.cheekbones.height", cb.height);
            chk("face.cheekbones.prominence", cb.prominence);
        }
        if let Some(fh) = &fc.forehead {
            chk("face.forehead.height", fh.height);
            chk("face.forehead.slope", fh.slope);
        }
    }
    if let Some(e) = &spec.eyes {
        chk("eyes.size", e.size);
        chk("eyes.spacing", e.spacing);
        chk("eyes.canthal_tilt", e.canthal_tilt);
        chk("eyes.hood", e.hood);
        chk("eyes.sclera_show", e.sclera_show);
        if let Some(b) = &e.brow {
            chk("eyes.brow.thickness", b.thickness);
            chk("eyes.brow.length", b.length);
            chk("eyes.brow.spacing", b.spacing);
        }
    }
    if let Some(n) = &spec.nose {
        chk("nose.length", n.length);
        chk("nose.columella", n.columella);
        if let Some(b) = &n.bridge {
            chk("nose.bridge.width", b.width);
            chk("nose.bridge.height", b.height);
        }
        if let Some(t) = &n.tip {
            chk("nose.tip.projection", t.projection);
            chk("nose.tip.rotation", t.rotation);
            chk("nose.tip.width", t.width);
        }
    }
    if let Some(m) = &spec.mouth {
        chk("mouth.width", m.width);
        chk("mouth.upper_lip", m.upper_lip);
        chk("mouth.lower_lip", m.lower_lip);
        chk("mouth.corners", m.corners);
        chk("mouth.lip_texture", m.lip_texture);
    }
    if let Some(t) = &spec.teeth {
        chk("teeth.diastema", t.diastema);
        chk("teeth.shade", t.shade);
        chk("teeth.gum_show", t.gum_show);
        chk("teeth.wear", t.wear);
    }
    if let Some(s) = &spec.skin {
        chk("skin.texture", s.texture);
        chk("skin.complexion", s.complexion);
        chk("skin.pores", s.pores);
    }
    if let Some(fig) = &spec.figure {
        chk("figure.weight_impression", fig.weight_impression);
        chk("figure.shoulders", fig.shoulders);
        chk("figure.waist", fig.waist);
        chk("figure.limb_length", fig.limb_length);
        chk("figure.musculature", fig.musculature);
    }
}

fn contradictions(spec: &PersonaSpec, f: &mut Vec<Finding>) {
    // facial_hair: none with positive density/length.
    if let Some(fh) = &spec.facial_hair {
        if fh.style.as_deref() == Some("none") {
            if fh.density.is_some_and(|d| d > 0.0) {
                f.push(Finding::err("facial_hair", "style `none` with density > 0"));
            }
            if fh.length.is_some_and(|l| l > 0.0) {
                f.push(Finding::err("facial_hair", "style `none` with length > 0"));
            }
        }
    }
    // heterochromia set alongside a single scalar eye colour.
    if let Some(e) = &spec.eyes {
        let het = e.heterochromia.as_ref().is_some_and(|v| !v.is_null() && v.as_str() != Some("none"));
        if het && e.color.is_some() {
            f.push(Finding::warn("eyes", "heterochromia set alongside a single `eyes.color`; the pair is ambiguous"));
        }
    }
    // teeth: alignment diastema with a zero gap.
    if let Some(t) = &spec.teeth {
        if t.alignment.as_deref() == Some("diastema") && t.diastema.is_some_and(|d| d == 0.0) {
            f.push(Finding::err("teeth", "alignment `diastema` with diastema: 0.0"));
        }
    }
    // marks authored but empty vs absent is fine; a mark with neither anchor nor region is unplaceable.
    if let Some(marks) = &spec.marks {
        for (i, m) in marks.iter().enumerate() {
            let distributional = m.region.is_some();
            let positional = m.anchor.as_ref().is_some_and(|a| a.landmark.is_some() || a.region.is_some());
            if !distributional && !positional && m.kind.as_deref() != Some("freckles") {
                f.push(Finding::warn(
                    &format!("marks[{i}]"),
                    "mark has neither an anchor nor a region — it cannot be placed",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(s: &str) -> PersonaSpec {
        PersonaSpec::from_hjson(s).unwrap()
    }

    #[test]
    fn clean_spec_has_no_errors() {
        let s = load("{\n  schema: persona/1\n  identity: {\n    name: alice\n    apparent_age: 34\n  }\n  eyes: {\n    spacing: 0.62\n  }\n}\n");
        let f = lint(&s);
        assert!(!has_errors(&f), "{f:?}");
    }

    #[test]
    fn out_of_range_scalar_is_error() {
        let s = load("{\n  eyes: {\n    spacing: 1.4\n  }\n}\n");
        let f = lint(&s);
        assert!(has_errors(&f));
        assert!(f.iter().any(|x| x.path == "eyes.spacing" && x.level == Level::Error));
    }

    #[test]
    fn facial_hair_contradiction() {
        let s = load("{\n  facial_hair: {\n    style: none\n    density: 0.5\n  }\n}\n");
        assert!(has_errors(&lint(&s)));
    }

    #[test]
    fn diastema_zero_gap_contradiction() {
        let s = load("{\n  teeth: {\n    alignment: diastema\n    diastema: 0.0\n  }\n}\n");
        assert!(has_errors(&lint(&s)));
    }

    #[test]
    fn missing_schema_warns_not_errors() {
        let s = load("{\n  identity: {\n    name: bob\n    apparent_age: 40\n  }\n}\n");
        let f = lint(&s);
        assert!(!has_errors(&f));
        assert!(f.iter().any(|x| x.path == "schema" && x.level == Level::Warning));
    }
}
