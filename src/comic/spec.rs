//! The `ComicSpec` — the HJSON a comic page is authored from (RFC COMIC-1 §"The ComicSpec"). Permissive
//! serde like `BookArtSpec` / `PersonaSpec`: every field optional (a bare `{}` resolves to a neutral
//! single-panel page), enums carried as strings (lint catches typos, not a hard failure), unknown keys
//! ignored (forward-compatible).

use serde::Deserialize;

pub const SCHEMA_VERSION: &str = "comic/1";

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ComicSpec {
    pub schema: Option<String>,
    pub page: Option<Page>,
    /// `ltr` (western, default) | `rtl` (manga).
    pub reading: Option<String>,
    pub layout: Option<Layout>,
    pub cast: Vec<CastMember>,
    pub panels: Vec<Panel>,
    /// Diffusion base for the per-panel scene art (P3).
    pub model: Option<String>,
    pub seed: Option<u64>,
    pub steps: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Page {
    /// Named (`us-letter`/`a4`/`a5`/`tabloid`/`square`) or `custom` (with `w_in`/`h_in`).
    pub size: Option<String>,
    pub dpi: Option<u32>,
    /// Gutter between panels, in px at the page DPI.
    pub gutter: Option<u32>,
    /// Panel border stroke width, px.
    pub border: Option<u32>,
    /// Page background colour name (`white` default) or `r,g,b`.
    pub bg: Option<String>,
    pub w_in: Option<f32>,
    pub h_in: Option<f32>,
}

/// The panel grid: `rows` of relative-width cells. `[[1,1],[1],[1,1,1]]` = 2 | 1-wide | 3 panels. Absent →
/// auto-grid the panels into a near-square.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Layout {
    pub rows: Option<Vec<Vec<f32>>>,
    /// Per-row relative heights (default: equal).
    pub row_heights: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct CastMember {
    pub name: String,
    /// A `PersonaSpec` path → a *specific* recurring identity (the reason a comic needs persona).
    pub persona: Option<String>,
    /// Or a stable text description (seed-locked) when no persona.
    pub describe: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Panel {
    /// The scene prompt (P3 generates it).
    pub scene: Option<String>,
    /// Cast names appearing in this panel.
    pub chars: Vec<String>,
    /// A narration/caption box (no tail).
    pub caption: Option<String>,
    pub balloons: Vec<Balloon>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Balloon {
    /// The speaking cast member's name (for the tail).
    pub by: Option<String>,
    pub say: Option<String>,
    /// A placement hint (`auto` default) — `top-left`/`top-right`/… bias the placer.
    pub at: Option<String>,
    /// `speech` (default) | `thought` | `shout` | `caption`.
    pub kind: Option<String>,
}

impl ComicSpec {
    pub fn from_hjson(text: &str) -> Result<Self, deser_hjson::Error> {
        deser_hjson::from_str(text)
    }
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Self::from_hjson(&text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_and_full_specs_parse() {
        assert!(ComicSpec::from_hjson("{}").unwrap().panels.is_empty());
        let s = ComicSpec::from_hjson(
            r#"{
                schema: "comic/1"
                page: { size: "us-letter", dpi: 300, gutter: 24, border: 6 }
                reading: "ltr"
                layout: { rows: [[1,1],[1],[1,1,1]] }
                cast: [ { name: "mika", persona: "mika.hjson" }, { name: "bot", describe: "a brass robot" } ]
                panels: [
                    { scene: "a neon alley", caption: "3 a.m." }
                    { scene: "mika crouched", chars: ["mika"], balloons: [ { by: "mika", say: "Did you hear that?", at: "top-left" } ] }
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(s.reading.as_deref(), Some("ltr"));
        assert_eq!(s.layout.unwrap().rows.unwrap()[2], vec![1.0, 1.0, 1.0]);
        assert_eq!(s.cast.len(), 2);
        assert_eq!(s.panels[1].balloons[0].say.as_deref(), Some("Did you hear that?"));
    }
}
