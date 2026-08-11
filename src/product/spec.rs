//! The `ProductSpec` — the HJSON a studio product-shot is authored from (RFC PRODUCT-1). Permissive serde
//! like `TextureSpec` / `ComicSpec`: every field optional (a bare `{}` + a supplied cutout resolves to a
//! neutral white-sweep packshot), enums carried as strings (lint catches typos), unknown keys ignored.

use serde::Deserialize;

pub const SCHEMA_VERSION: &str = "product/1";

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ProductSpec {
    pub schema: Option<String>,
    pub subject: Option<Subject>,
    pub canvas: Option<Canvas>,
    /// Only consulted when `canvas.bg` is `scene` (P3): a generated environment.
    pub scene: Option<Scene>,
    /// Relight rig (P2). Absent → the subject keeps its own light (weight-free).
    pub lighting: Option<Lighting>,
    pub camera: Option<Camera>,
    pub ground: Option<Ground>,
    /// Extra angles for a catalog contact sheet (`product sheet`, P3).
    pub variants: Vec<Variant>,
    pub model: Option<String>,
    pub seed: Option<u64>,
    pub steps: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Subject {
    /// A transparent **cutout** PNG — used as-is (composited, pixel-exact). The P1 weight-free path.
    pub image: Option<String>,
    /// …or a **photo** to matte (U2Net) first (P2).
    pub photo: Option<String>,
    /// …or a **prompt** to generate then matte (P2).
    pub prompt: Option<String>,
    /// Fraction of the canvas height the product fills (default 0.7).
    pub scale: Option<f32>,
    /// Where the product sits: `bottom` (grounded, default) | `center`.
    pub anchor: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Canvas {
    /// `square` (default) | `portrait` | `landscape` — sets the aspect; `px` sets the long side.
    pub size: Option<String>,
    pub px: Option<u32>,
    /// `white` (default) | `grey-sweep` | `gradient:<top>,<bottom>` | `scene`.
    pub bg: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Scene {
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Lighting {
    /// `three-point` | `softbox` | `beauty` | `rim` | `hard` | `flat`.
    pub rig: Option<String>,
    /// Dominant light direction — `top-left` (default) | `top` | `top-right` | `left` | `right`.
    pub key_dir: Option<String>,
    pub intensity: Option<f32>,
    /// -1 cool … +1 warm.
    pub warmth: Option<f32>,
    /// Free-text override fed straight to IC-Light (P2).
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Camera {
    /// `eye` (default) | `hero` (low) | `top` (flatlay) | `three-quarter`.
    pub angle: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Ground {
    /// `soft` (default) | `hard` (a directional cast rake) | `none`.
    pub shadow: Option<String>,
    pub softness: Option<f32>,
    /// `gloss` (dim floor reflection, default) | `mirror` | `none`.
    pub reflection: Option<String>,
    pub falloff: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Variant {
    pub image: Option<String>,
    pub label: Option<String>,
}

impl ProductSpec {
    pub fn from_hjson(text: &str) -> Result<Self, deser_hjson::Error> {
        deser_hjson::from_str(text)
    }
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Self::from_hjson(&text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))
    }
    /// The subject source path for the P1 weight-free path (a cutout), if any.
    pub fn subject_image(&self) -> Option<&str> {
        self.subject.as_ref().and_then(|s| s.image.as_deref()).filter(|s| !s.trim().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_and_full_specs_parse() {
        assert!(ProductSpec::from_hjson("{}").unwrap().subject.is_none());
        let s = ProductSpec::from_hjson(
            r#"{
                schema: "product/1"
                subject: { image: "shoe.png", scale: 0.7, anchor: "bottom" }
                canvas: { size: "square", px: 1024, bg: "white" }
                lighting: { rig: "three-point", key_dir: "top-left" }
                camera: { angle: "three-quarter" }
                ground: { shadow: "soft", reflection: "gloss" }
                variants: [ { image: "shoe_side.png", label: "side" } ]
            }"#,
        )
        .unwrap();
        assert_eq!(s.subject_image(), Some("shoe.png"));
        assert_eq!(s.canvas.unwrap().px, Some(1024));
        assert_eq!(s.ground.unwrap().shadow.as_deref(), Some("soft"));
        assert_eq!(s.variants.len(), 1);
    }
}
