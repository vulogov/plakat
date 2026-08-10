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
    /// Inherit a base spec (the shared "world" — cast/style/model/page) from another file (6.8.1). The
    /// path is resolved relative to this spec's directory. See [`ComicSpec::load`].
    pub extends: Option<String>,
    pub page: Option<Page>,
    /// `ltr` (western, default) | `rtl` (manga).
    pub reading: Option<String>,
    pub layout: Option<Layout>,
    pub cast: Vec<CastMember>,
    /// The panels of a **single-page** comic. For a **multi-page** comic use [`pages`](Self::pages)
    /// instead; the two are mutually exclusive (a non-empty `pages` wins).
    pub panels: Vec<Panel>,
    /// A **multi-page** comic (6.8.1): each entry is one page's layout + panels. The top-level
    /// `cast`/`style`/`model`/`seed`/`page` are the shared world that propagates to every page.
    pub pages: Vec<PageSpec>,
    /// A named scene library (6.8.1): a panel `scene: "@alley"` resolves to `scenes["alley"]`, so a
    /// setting recurs across pages by reference. Deterministic (BTree).
    pub scenes: std::collections::BTreeMap<String, String>,
    /// Diffusion base for the per-panel scene art (P3).
    pub model: Option<String>,
    /// A shared art-style suffix appended to every panel prompt, so the whole page reads as one hand
    /// (P3). Absent → a sensible comic-book default.
    pub style: Option<String>,
    /// M2 style-lock: a LoRA (path or HF repo) applied to *every* panel of *every* page, so the whole
    /// book holds one look beyond the text style. Absent → no LoRA.
    pub style_lora: Option<String>,
    /// Scale for [`style_lora`](Self::style_lora) (default 0.8).
    pub style_lora_scale: Option<f32>,
    pub seed: Option<u64>,
    pub steps: Option<usize>,
}

/// One page of a multi-page comic (6.8.1): its own layout + panels; the rest is the shared world.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct PageSpec {
    /// An optional label (for `comic show` / filenames).
    pub name: Option<String>,
    /// This page's grid; absent → the spec's top-level `layout`, else auto-grid.
    pub layout: Option<Layout>,
    /// This page's reading direction; absent → the spec's top-level `reading`.
    pub reading: Option<String>,
    pub panels: Vec<Panel>,
}

/// A resolved logical page: the panels (with `@scene` refs expanded) + the effective layout/reading.
#[derive(Debug, Clone)]
pub struct LogicalPage {
    pub name: Option<String>,
    pub layout: Option<Layout>,
    pub reading: Option<String>,
    pub panels: Vec<Panel>,
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
    /// M2 reference-lock: an explicit reference face image. When set, the character's face is locked to
    /// this image on every panel (face-swap); absent → a canonical portrait is rendered once from the
    /// persona/describe and used as the reference.
    pub reference: Option<String>,
    /// Opt this character out of the face-lock (still description-consistent). Default: locked.
    pub lock: Option<bool>,
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
    /// 6.8.2 D2: a label for this panel so another panel can reuse its rendered art (`reuse: "@id"`).
    pub id: Option<String>,
    /// 6.8.2 D2: render this panel as an **exact copy** of a labelled panel's art (`"@id"`), book-wide —
    /// an establishing shot that repeats identically instead of re-generating from a recurring `@scene`.
    pub reuse: Option<String>,
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

    /// Load a spec, applying `extends:` inheritance (6.8.1). The base is resolved relative to `path`'s
    /// directory; this spec's fields override the base's (see [`merge_over`](Self::merge_over)).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Self::load_depth(path, 0)
    }

    fn load_depth(path: &std::path::Path, depth: usize) -> anyhow::Result<Self> {
        if depth > 8 {
            anyhow::bail!("comic: `extends` chain too deep (cycle?) at {}", path.display());
        }
        let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let mut spec = Self::from_hjson(&text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;
        if let Some(base_ref) = spec.extends.clone() {
            let base_path = path.parent().unwrap_or_else(|| std::path::Path::new(".")).join(&base_ref);
            let base = Self::load_depth(&base_path, depth + 1).map_err(|e| anyhow::anyhow!("{}: extends {base_ref:?}: {e}", path.display()))?;
            spec = spec.merge_over(base);
        }
        Ok(spec)
    }

    /// Overlay `self` on top of `base`: `self`'s set scalars win; `cast`/`scenes` merge by name (self
    /// wins on collision); `panels`/`pages` replace the base's only when `self` provides them.
    pub fn merge_over(self, base: Self) -> Self {
        let mut cast = base.cast;
        for c in self.cast {
            if let Some(slot) = cast.iter_mut().find(|b| b.name == c.name) {
                *slot = c;
            } else {
                cast.push(c);
            }
        }
        let mut scenes = base.scenes;
        scenes.extend(self.scenes);
        ComicSpec {
            schema: self.schema.or(base.schema),
            extends: None,
            page: self.page.or(base.page),
            reading: self.reading.or(base.reading),
            layout: self.layout.or(base.layout),
            cast,
            panels: if self.panels.is_empty() { base.panels } else { self.panels },
            pages: if self.pages.is_empty() { base.pages } else { self.pages },
            scenes,
            model: self.model.or(base.model),
            style: self.style.or(base.style),
            style_lora: self.style_lora.or(base.style_lora),
            style_lora_scale: self.style_lora_scale.or(base.style_lora_scale),
            seed: self.seed.or(base.seed),
            steps: self.steps.or(base.steps),
        }
    }

    /// Expand a panel `scene` that starts with `@` against the [`scenes`](Self::scenes) library; other
    /// values pass through unchanged.
    pub fn resolve_scene(&self, scene: &str) -> String {
        if let Some(key) = scene.strip_prefix('@') {
            if let Some(v) = self.scenes.get(key.trim()) {
                return v.clone();
            }
        }
        scene.to_string()
    }

    /// The comic as an ordered list of logical pages, each with `@scene` refs expanded. A multi-page spec
    /// (`pages` non-empty) yields those; otherwise a single page from the top-level `layout`/`panels`.
    pub fn logical_pages(&self) -> Vec<LogicalPage> {
        let expand = |panels: &[Panel]| -> Vec<Panel> {
            panels
                .iter()
                .map(|p| {
                    let mut p = p.clone();
                    if let Some(s) = &p.scene {
                        p.scene = Some(self.resolve_scene(s));
                    }
                    p
                })
                .collect()
        };
        if !self.pages.is_empty() {
            self.pages
                .iter()
                .map(|pg| LogicalPage {
                    name: pg.name.clone(),
                    layout: pg.layout.clone().or_else(|| self.layout.clone()),
                    reading: pg.reading.clone().or_else(|| self.reading.clone()),
                    panels: expand(&pg.panels),
                })
                .collect()
        } else {
            vec![LogicalPage { name: None, layout: self.layout.clone(), reading: self.reading.clone(), panels: expand(&self.panels) }]
        }
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

    #[test]
    fn single_page_spec_yields_one_logical_page() {
        let s = ComicSpec::from_hjson(r#"{ panels: [{scene:"a"},{scene:"b"}] }"#).unwrap();
        let pages = s.logical_pages();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].panels.len(), 2);
    }

    #[test]
    fn multi_page_shares_world_and_expands_scenes() {
        let s = ComicSpec::from_hjson(
            r#"{
                cast: [{name:"mira", describe:"a woman in a red scarf"}]
                style: "noir"
                scenes: { alley: "a rain-slick neon alley" }
                pages: [
                    { layout: {rows:[[1,1]]}, panels: [ {scene:"@alley", chars:["mira"]}, {scene:"a rooftop"} ] }
                    { reading: "rtl", panels: [ {scene:"@alley"} ] }
                ]
            }"#,
        )
        .unwrap();
        let pages = s.logical_pages();
        assert_eq!(pages.len(), 2, "two pages");
        // `@alley` expanded on both pages.
        assert_eq!(pages[0].panels[0].scene.as_deref(), Some("a rain-slick neon alley"));
        assert_eq!(pages[1].panels[0].scene.as_deref(), Some("a rain-slick neon alley"));
        // page 2 inherits nothing for layout (auto) but overrides reading.
        assert_eq!(pages[1].reading.as_deref(), Some("rtl"));
        // the shared world lives at the top level.
        assert_eq!(s.cast.len(), 1);
        assert_eq!(s.style.as_deref(), Some("noir"));
    }

    #[test]
    fn extends_merges_base_world_with_child_pages() {
        let base = ComicSpec::from_hjson(r#"{ cast:[{name:"mira",describe:"red scarf"}], style:"noir", model:"sdxl", seed:7 }"#).unwrap();
        let child = ComicSpec::from_hjson(r#"{ style:"bright", pages:[ {panels:[{scene:"a"}]} ] }"#).unwrap();
        let merged = child.merge_over(base);
        assert_eq!(merged.cast.len(), 1, "cast inherited");
        assert_eq!(merged.model.as_deref(), Some("sdxl"), "model inherited");
        assert_eq!(merged.seed, Some(7), "seed inherited");
        assert_eq!(merged.style.as_deref(), Some("bright"), "child overrides style");
        assert_eq!(merged.logical_pages().len(), 1, "child pages win");
    }
}
