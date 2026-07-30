//! The persona lexicon (RFC §7) — data, not code. Maps each prompt-bearing attribute to its class,
//! section, and phrasing, so the resolver + emitters are template-driven and can never drift from the
//! schema. P0 ships a **skeleton** (the highest-signal attributes); it grows one HJSON entry at a time.
//!
//! The skeleton is embedded (`include_str!`) so it is byte-stable and needs no runtime file.

use serde::Deserialize;
use std::collections::HashMap;

/// One lexicon entry (P0 subset of §7.1). See `assets/persona/lexicon.hjson` for the field docs.
#[derive(Debug, Clone, Deserialize)]
pub struct LexEntry {
    pub class: String,   // structural | surface | detail
    pub section: String, // anatomical group
    pub kind: String,    // scalar | enum | color
    #[serde(default)]
    pub low: Option<String>,
    #[serde(default)]
    pub high: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    /// Per-value phrasing for enums that need it (e.g. `skin.tone` fitzpatrick → neutral tone words,
    /// §7.4). Takes precedence over `template`; an empty string means "emit nothing for this value".
    #[serde(default)]
    pub values: Option<HashMap<String, String>>,
    #[serde(default)]
    pub none_negative: Option<String>,
    #[serde(default)]
    pub control: Option<String>,
    // --- interview question-graph fields (RFC §17.3); all optional, defaults derived from kind/class ---
    /// The question text. Defaults to a "<section> — <leaf>" phrasing.
    #[serde(default)]
    pub ask: Option<String>,
    /// The answer widget (§17.4). Defaults from `kind` (scalar→scalar, enum→select, color→color).
    #[serde(default)]
    pub widget: Option<String>,
    /// The depth at which the question first appears: `quick` | `standard` | `full`. Defaults from
    /// `class` (structural→quick, surface→standard, detail→full).
    #[serde(default)]
    pub depth: Option<String>,
    /// A within-section ordering hint (lower first). Defaults to 0.
    #[serde(default)]
    pub order: Option<i32>,
    /// A total condition on the answer log; the question is skipped when it evaluates false (§17.3).
    #[serde(default)]
    pub when: Option<String>,
}

impl LexEntry {
    /// The interview widget kind for this entry (§17.4), derived from `kind` when unset.
    pub fn widget_kind(&self) -> &str {
        if let Some(w) = &self.widget {
            return w;
        }
        match self.kind.as_str() {
            "scalar" => "scalar",
            "color" => "color",
            _ => "select",
        }
    }
    /// The question text (§17.3), derived when unset.
    pub fn question_text(&self, path: &str) -> String {
        if let Some(a) = &self.ask {
            return a.clone();
        }
        let leaf = path.rsplit('.').next().unwrap_or(path).replace('_', " ");
        format!("{} — {}", self.section, leaf)
    }
    /// The depth tier (0 quick, 1 standard, 2 full), from `depth` or `class`.
    pub fn depth_tier(&self) -> u8 {
        match self.depth.as_deref() {
            Some("quick") => 0,
            Some("standard") => 1,
            Some("full") => 2,
            _ => match self.class.as_str() {
                "structural" => 0,
                "surface" => 1,
                _ => 2,
            },
        }
    }
}

impl LexEntry {
    /// Salience weight from the controllability grade (§9.2). Default `moderate` when unspecified.
    pub fn control_weight(&self) -> f32 {
        match self.control.as_deref() {
            Some("strong") => 1.0,
            Some("moderate") | None => 0.7,
            Some("weak") => 0.4,
            Some("experimental") => 0.2,
            Some(_) => 0.7,
        }
    }

    /// Class weight (§9.2): structural attributes dominate the budget, then surface, then detail.
    pub fn class_weight(&self) -> f32 {
        match self.class.as_str() {
            "structural" => 1.0,
            "surface" => 0.7,
            "detail" => 0.5,
            _ => 0.7,
        }
    }
}

/// The loaded lexicon: attribute path → entry.
#[derive(Debug, Clone)]
pub struct Lexicon {
    pub entries: HashMap<String, LexEntry>,
    pub version: String,
}

impl Lexicon {
    /// Load the embedded skeleton lexicon (byte-stable, no I/O).
    pub fn skeleton() -> Self {
        let src = include_str!("../../assets/persona/lexicon.hjson");
        let entries: HashMap<String, LexEntry> =
            deser_hjson::from_str(src).expect("embedded persona lexicon must parse");
        Lexicon { entries, version: "1.0".to_string() }
    }

    pub fn get(&self, path: &str) -> Option<&LexEntry> {
        self.entries.get(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_loads_and_covers_the_sections() {
        let lex = Lexicon::skeleton();
        assert!(lex.entries.len() >= 12, "skeleton should cover the core attributes");
        for path in ["eyes.spacing", "eyes.color", "hair.color", "nose.profile", "facial_hair.style"] {
            assert!(lex.get(path).is_some(), "missing lexicon entry {path}");
        }
        // scalar entries carry poles; enum/color carry a template.
        assert!(lex.get("eyes.spacing").unwrap().low.is_some());
        assert!(lex.get("hair.color").unwrap().template.is_some());
        // weights resolve.
        assert_eq!(lex.get("eyes.spacing").unwrap().control_weight(), 1.0); // strong
        assert_eq!(lex.get("eyes.spacing").unwrap().class_weight(), 1.0); // structural
    }
}
