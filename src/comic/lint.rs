//! Layer-0 validation (RFC COMIC-1) — pure, no weights: schema, vocabulary, and cross-references (a
//! panel/balloon naming a cast member that doesn't exist). Warnings guide; nothing here is a hard failure
//! except a schema mismatch signalled loudly.

use super::spec::ComicSpec;

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

const SIZES: &[&str] = &["us-letter", "letter", "a4", "a5", "tabloid", "ledger", "square", "custom"];
const READING: &[&str] = &["ltr", "rtl"];
const KINDS: &[&str] = &["speech", "thought", "shout", "caption"];

pub fn lint(spec: &ComicSpec) -> Vec<Finding> {
    let mut f = Vec::new();
    if let Some(s) = &spec.schema {
        if s != super::SCHEMA_VERSION {
            f.push(Finding::warn("schema", format!("`{s}` != this build's `{}`", super::SCHEMA_VERSION)));
        }
    }
    if let Some(p) = &spec.page {
        if let Some(sz) = &p.size {
            if !SIZES.contains(&sz.to_ascii_lowercase().as_str()) {
                f.push(Finding::warn("page.size", format!("unknown size `{sz}`; known: {}", SIZES.join(", "))));
            }
            if sz.eq_ignore_ascii_case("custom") && (p.w_in.is_none() || p.h_in.is_none()) {
                f.push(Finding::err("page.size", "custom size needs page.w_in and page.h_in"));
            }
        }
    }
    if let Some(r) = &spec.reading {
        if !READING.contains(&r.to_ascii_lowercase().as_str()) {
            f.push(Finding::warn("reading", format!("unknown `{r}`; expected ltr|rtl")));
        }
    }
    if spec.panels.is_empty() {
        f.push(Finding::warn("panels", "no panels — resolves to a single empty page"));
    }
    // grid cell count vs panel count.
    if let Some(rows) = spec.layout.as_ref().and_then(|l| l.rows.as_ref()) {
        let cells: usize = rows.iter().map(|r| r.len()).sum();
        if cells != spec.panels.len() {
            f.push(Finding::warn("layout", format!("{cells} grid cell(s) but {} panel(s) — extra cells are empty / extra panels are dropped", spec.panels.len())));
        }
    }
    // cross-refs: panel chars + balloon `by` must name a cast member.
    let cast: std::collections::HashSet<&str> = spec.cast.iter().map(|c| c.name.as_str()).collect();
    for (i, panel) in spec.panels.iter().enumerate() {
        for c in &panel.chars {
            if !cast.contains(c.as_str()) {
                f.push(Finding::warn(&format!("panels[{i}].chars"), format!("`{c}` is not in the cast")));
            }
        }
        for (b, balloon) in panel.balloons.iter().enumerate() {
            if let Some(by) = &balloon.by {
                if !cast.contains(by.as_str()) {
                    f.push(Finding::warn(&format!("panels[{i}].balloons[{b}].by"), format!("`{by}` is not in the cast (no tail target)")));
                }
            }
            if let Some(k) = &balloon.kind {
                if !KINDS.contains(&k.to_ascii_lowercase().as_str()) {
                    f.push(Finding::warn(&format!("panels[{i}].balloons[{b}].kind"), format!("unknown `{k}`; known: {}", KINDS.join(", "))));
                }
            }
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_spec_has_no_errors_and_flags_unknown_cast() {
        let s = ComicSpec::from_hjson(
            r#"{ page: { size: "a4" }, reading: "ltr", cast: [{name:"mika"}],
                 panels: [ { chars: ["mika"] }, { chars: ["ghost"], balloons: [{ by: "ghost", say: "boo", kind: "whisper" }] } ] }"#,
        )
        .unwrap();
        let f = lint(&s);
        assert!(!f.iter().any(|x| x.level == Level::Error));
        assert!(f.iter().any(|x| x.message.contains("ghost")), "unknown cast flagged");
        assert!(f.iter().any(|x| x.message.contains("whisper")), "unknown balloon kind flagged");
    }
}
