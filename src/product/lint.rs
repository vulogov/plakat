//! Validation (RFC PRODUCT-1) — pure, no weights: schema, vocabulary, and a subject source. Warnings
//! guide; a missing subject is the one hard error (nothing to shoot).

use super::spec::ProductSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Level {
    Error,
    Warn,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub level: Level,
    pub path: String,
    pub message: String,
}
impl Finding {
    fn warn(path: &str, m: impl Into<String>) -> Self {
        Self { level: Level::Warn, path: path.into(), message: m.into() }
    }
    fn err(path: &str, m: impl Into<String>) -> Self {
        Self { level: Level::Error, path: path.into(), message: m.into() }
    }
}

const BG: &[&str] = &["white", "grey-sweep", "gradient", "scene"];
const RIG: &[&str] = &["three-point", "softbox", "beauty", "rim", "hard", "flat"];
const ANGLE: &[&str] = &["eye", "hero", "top", "three-quarter"];
const SHADOW: &[&str] = &["soft", "hard", "none"];
const REFLECTION: &[&str] = &["gloss", "mirror", "none"];

pub fn lint(spec: &ProductSpec) -> Vec<Finding> {
    let mut f = Vec::new();
    if let Some(s) = &spec.schema {
        if s != super::SCHEMA_VERSION {
            f.push(Finding::warn("schema", format!("`{s}` != this build's `{}`", super::SCHEMA_VERSION)));
        }
    }
    // a subject source is required to shoot something.
    let subj = spec.subject.clone().unwrap_or_default();
    let has_source = [&subj.image, &subj.photo, &subj.prompt].iter().any(|o| o.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false));
    if !has_source {
        f.push(Finding::err("subject", "no subject — set `image` (a cutout), `photo`, or `prompt`"));
    }
    if subj.photo.is_some() || subj.prompt.is_some() {
        f.push(Finding::warn("subject", "`photo`/`prompt` need a model (matte / generate) — the P1 weight-free path uses `image` (a transparent cutout)"));
    }
    if let Some(c) = &spec.canvas {
        if let Some(bg) = &c.bg {
            let head = bg.split(':').next().unwrap_or(bg);
            if !BG.contains(&head.to_ascii_lowercase().as_str()) {
                f.push(Finding::warn("canvas.bg", format!("unknown `{bg}`; known: {}", BG.join(", "))));
            }
            if bg.eq_ignore_ascii_case("scene") && spec.scene.as_ref().and_then(|s| s.prompt.as_deref()).unwrap_or("").trim().is_empty() {
                f.push(Finding::warn("canvas.bg", "`scene` needs `scene.prompt` (and a model — P3)"));
            }
        }
    }
    if let Some(l) = &spec.lighting {
        if let Some(r) = &l.rig {
            if !RIG.contains(&r.to_ascii_lowercase().as_str()) {
                f.push(Finding::warn("lighting.rig", format!("unknown `{r}`; known: {}", RIG.join(", "))));
            }
        }
        f.push(Finding::warn("lighting", "relighting needs a model (IC-Light) — P2; ignored on the P1 weight-free path"));
    }
    if let Some(c) = &spec.camera {
        if let Some(a) = &c.angle {
            if !ANGLE.contains(&a.to_ascii_lowercase().as_str()) {
                f.push(Finding::warn("camera.angle", format!("unknown `{a}`; known: {}", ANGLE.join(", "))));
            }
        }
    }
    if let Some(g) = &spec.ground {
        if let Some(s) = &g.shadow {
            if !SHADOW.contains(&s.to_ascii_lowercase().as_str()) {
                f.push(Finding::warn("ground.shadow", format!("unknown `{s}`; known: {}", SHADOW.join(", "))));
            }
        }
        if let Some(r) = &g.reflection {
            if !REFLECTION.contains(&r.to_ascii_lowercase().as_str()) {
                f.push(Finding::warn("ground.reflection", format!("unknown `{r}`; known: {}", REFLECTION.join(", "))));
            }
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_missing_subject_and_unknown_vocab() {
        let s = ProductSpec::from_hjson(r#"{ canvas: { bg: "chartreuse" }, ground: { shadow: "fuzzy" } }"#).unwrap();
        let f = lint(&s);
        assert!(f.iter().any(|x| x.level == Level::Error && x.path == "subject"), "missing subject is an error");
        assert!(f.iter().any(|x| x.message.contains("chartreuse")), "unknown bg flagged");
        assert!(f.iter().any(|x| x.message.contains("fuzzy")), "unknown shadow flagged");
    }

    #[test]
    fn a_cutout_spec_is_clean() {
        let s = ProductSpec::from_hjson(r#"{ subject: { image: "x.png" }, canvas: { bg: "grey-sweep" }, ground: { shadow: "soft", reflection: "gloss" } }"#).unwrap();
        assert!(!lint(&s).iter().any(|x| x.level == Level::Error));
    }
}
