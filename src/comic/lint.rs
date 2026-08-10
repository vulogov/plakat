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
    // Multi-page (6.8.1): validate across every logical page. `panel_path` names findings by page.
    let pages = spec.logical_pages();
    let multi = spec.pages.len() > 1 || !spec.pages.is_empty();
    if pages.iter().all(|p| p.panels.is_empty()) {
        f.push(Finding::warn("panels", "no panels — resolves to a single empty page"));
    }
    let cast: std::collections::HashSet<&str> = spec.cast.iter().map(|c| c.name.as_str()).collect();
    for (pi, lp) in pages.iter().enumerate() {
        let pfx = |field: &str| if multi { format!("pages[{pi}].{field}") } else { field.to_string() };
        // grid cell count vs panel count for this page.
        if let Some(rows) = lp.layout.as_ref().and_then(|l| l.rows.as_ref()) {
            let cells: usize = rows.iter().map(|r| r.len()).sum();
            if cells != lp.panels.len() {
                f.push(Finding::warn(&pfx("layout"), format!("{cells} grid cell(s) but {} panel(s) — extra cells are empty / extra panels are dropped", lp.panels.len())));
            }
        }
        // cross-refs: panel chars + balloon `by` must name a cast member. (`@scene` refs already expanded.)
        for (i, panel) in lp.panels.iter().enumerate() {
            for c in &panel.chars {
                if !cast.contains(c.as_str()) {
                    f.push(Finding::warn(&pfx(&format!("panels[{i}].chars")), format!("`{c}` is not in the cast")));
                }
            }
            for (b, balloon) in panel.balloons.iter().enumerate() {
                if let Some(by) = &balloon.by {
                    if !cast.contains(by.as_str()) {
                        f.push(Finding::warn(&pfx(&format!("panels[{i}].balloons[{b}].by")), format!("`{by}` is not in the cast (no tail target)")));
                    }
                }
                if let Some(k) = &balloon.kind {
                    if !KINDS.contains(&k.to_ascii_lowercase().as_str()) {
                        f.push(Finding::warn(&pfx(&format!("panels[{i}].balloons[{b}].kind")), format!("unknown `{k}`; known: {}", KINDS.join(", "))));
                    }
                }
            }
        }
    }
    // unresolved `@scene` references (against the raw panels, before expansion).
    let raw_pages: Vec<&Vec<super::spec::Panel>> = if spec.pages.is_empty() { vec![&spec.panels] } else { spec.pages.iter().map(|p| &p.panels).collect() };
    for panels in raw_pages {
        for panel in panels {
            if let Some(s) = panel.scene.as_deref() {
                if let Some(key) = s.strip_prefix('@') {
                    if !spec.scenes.contains_key(key.trim()) {
                        f.push(Finding::warn("scene", format!("`@{}` is not in the `scenes` library", key.trim())));
                    }
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
