//! Layer-0 validation (RFC TEXTURE-1 §13 `lint`). Pure, no weights, no network: schema, vocabulary
//! (with nearest-match suggestions), numeric ranges, and structural contradictions. `lint` exits
//! non-zero on any [`Level::Error`] so it can gate CI.

use crate::texture::compile::ALL_MAPS;
use crate::texture::spec::TextureSpec;

pub const SEAMLESS_MODES: &[&str] = &["circular", "offset", "none"];
pub const SEAMLESS_AXES: &[&str] = &["both", "x", "y"];
pub const UPSCALES: &[&str] = &["none", "2k", "4k"];
pub const NORMAL_Y: &[&str] = &["opengl", "directx"];
pub const NAMINGS: &[&str] = &["plakat", "unity", "unreal"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    fn err(path: &str, message: String) -> Self {
        Finding { level: Level::Error, path: path.into(), message }
    }
    fn warn(path: &str, message: String) -> Self {
        Finding { level: Level::Warn, path: path.into(), message }
    }
}

/// Unknown enum value → Warn with a nearest-match suggestion (never a hard error — unknown values still
/// resolve via defaults; the point is to catch typos).
fn check_vocab(out: &mut Vec<Finding>, path: &str, value: Option<&str>, vocab: &[&str]) {
    if let Some(v) = value {
        if !vocab.contains(&v) {
            let hint = nearest(v, vocab).map(|s| format!(" (did you mean `{s}`?)")).unwrap_or_default();
            out.push(Finding::warn(path, format!("unknown value `{v}`{hint}; known: {}", vocab.join(", "))));
        }
    }
}

fn check_unit(out: &mut Vec<Finding>, path: &str, v: &Option<serde_json::Value>) {
    if let Some(serde_json::Value::Number(n)) = v {
        let x = n.as_f64().unwrap_or(0.0);
        if !(0.0..=1.0).contains(&x) {
            out.push(Finding::err(path, format!("scalar must be in [0,1], got {x}")));
        }
    }
}

/// Lint a spec → findings (Errors gate CI).
pub fn lint(spec: &TextureSpec) -> Vec<Finding> {
    let mut f = Vec::new();

    if let Some(s) = &spec.schema {
        if s != super::SCHEMA_VERSION {
            f.push(Finding::warn("schema", format!("`{s}` != this build's `{}` — newer keys may be ignored", super::SCHEMA_VERSION)));
        }
    }
    if spec.material.as_deref().map(str::trim).unwrap_or("").is_empty() && spec.from_image.is_none() {
        f.push(Finding::warn("material", "no `material` prompt and no `from_image` — resolves to a neutral flat material".into()));
    }
    if let Some(sm) = &spec.seamless {
        check_vocab(&mut f, "seamless.mode", sm.mode.as_deref(), SEAMLESS_MODES);
        check_vocab(&mut f, "seamless.axes", sm.axes.as_deref(), SEAMLESS_AXES);
    }
    if let Some(ch) = &spec.channels {
        check_vocab(&mut f, "channels.normal_y", ch.normal_y.as_deref(), NORMAL_Y);
        check_unit(&mut f, "channels.roughness", &ch.roughness);
        check_unit(&mut f, "channels.metallic", &ch.metallic);
        if let Some(ns) = ch.normal_strength {
            if !(0.0..=8.0).contains(&ns) {
                f.push(Finding::err("channels.normal_strength", format!("must be in [0,8], got {ns}")));
            }
        }
    }
    if let Some(p) = &spec.page {
        check_vocab(&mut f, "page.upscale", p.upscale.as_deref(), UPSCALES);
        if let Some(sz) = p.size {
            if !(64..=8192).contains(&sz) {
                f.push(Finding::err("page.size", format!("must be in [64,8192], got {sz}")));
            }
        }
    }
    if let Some(e) = &spec.export {
        check_vocab(&mut f, "export.naming", e.naming.as_deref(), NAMINGS);
        if let Some(maps) = &e.maps {
            for m in maps {
                if !ALL_MAPS.contains(&m.as_str()) {
                    let hint = nearest(m, ALL_MAPS).map(|s| format!(" (did you mean `{s}`?)")).unwrap_or_default();
                    f.push(Finding::warn("export.maps", format!("unknown map `{m}`{hint}; known: {}", ALL_MAPS.join(", "))));
                }
            }
        }
    }
    f
}

/// Nearest vocabulary entry by case-insensitive edit distance (typo suggestions).
pub fn nearest<'a>(word: &str, vocab: &[&'a str]) -> Option<&'a str> {
    let w = word.to_ascii_lowercase();
    vocab
        .iter()
        .map(|&c| (levenshtein(&w, &c.to_ascii_lowercase()), c))
        .filter(|&(d, c)| d <= 3.max(c.len() / 2))
        .min_by_key(|&(d, _)| d)
        .map(|(_, c)| c)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn errs(spec: &TextureSpec) -> Vec<Finding> {
        lint(spec).into_iter().filter(|f| f.level == Level::Error).collect()
    }

    #[test]
    fn bare_spec_has_no_errors() {
        assert!(errs(&TextureSpec::default()).is_empty());
    }

    #[test]
    fn out_of_range_is_an_error() {
        let s = TextureSpec::from_hjson(r#"{ page: { size: 40 }, channels: { roughness: 1.7 } }"#).unwrap();
        let e = errs(&s);
        assert_eq!(e.len(), 2, "{e:?}");
    }

    #[test]
    fn unknown_enum_warns_with_suggestion() {
        let s = TextureSpec::from_hjson(r#"{ seamless: { mode: "circualr" } }"#).unwrap();
        let ws: Vec<_> = lint(&s).into_iter().filter(|f| f.level == Level::Warn).collect();
        assert!(ws.iter().any(|w| w.message.contains("did you mean `circular`")), "{ws:?}");
        assert!(errs(&s).is_empty(), "a typo'd enum is a warning, not an error");
    }
}
