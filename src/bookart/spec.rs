//! `BookArtSpec` v1 — the on-disk schema for a book ornament (RFC BOOKART-1 §6.1).
//!
//! Same design constraints as `PersonaSpec`: **every field is optional** (a partial spec is valid),
//! enums are carried as `String` (unknown values load fine and are caught by `lint` with nearest-match
//! suggestions rather than a hard serde failure), and loading is `deser_hjson`. Deliberately permissive:
//! unknown *keys* are ignored (forward-compatible). Nothing here is weights-bearing or does I/O beyond
//! reading the file — the resolver, geometry, finisher and render tiers live in sibling modules.

use serde::Deserialize;

pub const SCHEMA_VERSION: &str = "bookart/1";

/// A single book-ornament spec (or a kit — see [`Kit`]).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct BookArtSpec {
    pub schema: Option<String>,
    /// Illustration tradition preset: `russian` | `english` | `japanese` | `american` | `european` |
    /// `chinese` | `generic`.
    pub origin: Option<String>,
    /// Drawing method: `line` | `woodcut` | `engraving` | `stipple` | `cross-hatch` | `silhouette` |
    /// `ink-wash` | `scratchboard`.
    pub technique: Option<String>,
    /// The shared decorative motif(s) — threaded into prompts and (later) generator params.
    pub motif: Option<Vec<String>>,
    pub ink: Option<Ink>,
    pub page: Option<Page>,
    /// Emit a transparent PNG (the default output contract). `false` keeps the paper opaque.
    pub transparent: Option<bool>,
    pub output: Option<Output>,
    /// A single ornament. Mutually-informative with [`Kit`]; a spec has one or the other.
    pub ornament: Option<Ornament>,
    /// A coherent matched set (RFC §10; generation lands in B7 — the schema is defined now).
    pub kit: Option<Kit>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Ink {
    /// `black` (default) | `sepia` | `#rrggbb`. Recolours the ink; alpha is unchanged (§7.4).
    pub color: Option<String>,
    /// Stroke/coverage weight in `[0,1]`; drives the technique binariser.
    pub weight: Option<f32>,
    /// Transparency model: `luminance` (default — ink darkness = opacity) | `threshold` | `matte` |
    /// `fade`.
    pub transparency: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Page {
    /// Named size (`a4`/`a5`/`a6`/`b5`/`letter`/`legal`/`trade`/`mass-market`) or `custom`.
    pub size: Option<String>,
    pub dpi: Option<u32>,
    /// `portrait` (default) | `landscape`.
    pub orientation: Option<String>,
    pub bleed_mm: Option<f32>,
    pub margins: Option<Margins>,
    pub gutter_mm: Option<f32>,
    pub custom: Option<CustomSize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Margins {
    pub top: Option<f32>,
    pub bottom: Option<f32>,
    pub inner: Option<f32>,
    pub outer: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct CustomSize {
    pub w_mm: Option<f32>,
    pub h_mm: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Output {
    /// Output formats. `png` is always emitted (primary); `svg`/`eps`/`pdf` are opt-in (§7.5).
    pub formats: Option<Vec<String>>,
    /// Ink tint applied to the transparent result (`black` default).
    pub tint: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Ornament {
    /// The ornament vocabulary key: `headpiece` | `tailpiece` | `initial` | `corner` | `border` |
    /// `divider` | `fleuron` | `dinkus` | `vignette` | `frontispiece` | `colophon` | `endpaper` |
    /// `marginalia`. Carried as `type:` in HJSON.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// `bilateral` | `radial:N` | `frieze:GROUP` | `none`.
    pub symmetry: Option<String>,
    /// Render tier override: `auto` (default) | `procedural` | `diffusion` | `composite`.
    pub tier: Option<String>,
    /// Pictorial prompt (diffusion / composite inlay).
    pub prompt: Option<String>,
    /// Procedural scaffold family for a `composite` frame (e.g. `filigree`).
    pub frame: Option<String>,
    /// Tailpiece taper in `[0,1]` (0 = a band, 1 = a point).
    pub taper: Option<f32>,
    /// A single decorated initial's letter (any script).
    pub glyph: Option<String>,
    /// A glyph-set name for an initial *series* (e.g. `cyrillic-upper`, `latin-upper`).
    pub glyphs: Option<String>,
    /// Initial cell height in text lines.
    pub lines: Option<u32>,
    /// Corner replication count (typically 4).
    pub places: Option<u32>,
    /// Edge fade for vignette/spot art, `[0,1]`.
    pub fade: Option<f32>,
    /// Per-ornament motif override (falls back to the top-level `motif`).
    pub motif: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Kit {
    /// Base seed; each ornament derives a deterministic per-item seed (§10).
    pub seed: Option<u64>,
    pub ornaments: Option<Vec<Ornament>>,
}

impl BookArtSpec {
    /// Parse from an HJSON string (permissive — unknown keys ignored, unknown enum values tolerated).
    pub fn from_hjson(text: &str) -> Result<Self, deser_hjson::Error> {
        deser_hjson::from_str(text)
    }

    /// Load + parse a spec from disk.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading bookart spec {}: {e}", path.display()))?;
        Self::from_hjson(&text).map_err(|e| anyhow::anyhow!("parsing bookart spec {}: {e}", path.display()))
    }

    /// The resolved ornament for a single-ornament spec: the explicit `ornament`, else a default
    /// (`divider`) so a bare `{}` still resolves to *something* renderable.
    pub fn ornament_or_default(&self) -> Ornament {
        self.ornament.clone().unwrap_or_else(|| Ornament { kind: Some("divider".into()), ..Default::default() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_spec_loads() {
        let s = BookArtSpec::from_hjson("{}").unwrap();
        assert!(s.origin.is_none());
        assert_eq!(s.ornament_or_default().kind.as_deref(), Some("divider"));
    }

    #[test]
    fn partial_spec_with_unknown_keys() {
        // Unknown keys ignored; `type` maps to kind; enums load as strings.
        let s = BookArtSpec::from_hjson(
            r#"{"schema":"bookart/1","origin":"russian","technique":"woodcut","motif":["firebird","oak-leaf"],"ornament":{"type":"headpiece","symmetry":"bilateral"},"wibble":3}"#,
        )
        .unwrap();
        assert_eq!(s.origin.as_deref(), Some("russian"));
        assert_eq!(s.motif.as_ref().unwrap().len(), 2);
        assert_eq!(s.ornament.as_ref().unwrap().kind.as_deref(), Some("headpiece"));
        assert_eq!(s.ornament.as_ref().unwrap().symmetry.as_deref(), Some("bilateral"));
    }

    #[test]
    fn kit_schema_parses() {
        let s = BookArtSpec::from_hjson(
            r#"{"origin":"english","kit":{"seed":42,"ornaments":[{"type":"headpiece"},{"type":"tailpiece","taper":0.6}]}}"#,
        )
        .unwrap();
        let k = s.kit.unwrap();
        assert_eq!(k.seed, Some(42));
        assert_eq!(k.ornaments.unwrap().len(), 2);
    }
}
