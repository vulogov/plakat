//! The `TextureSpec` — the HJSON document a `plakat texture` material is authored from (RFC TEXTURE-1
//! §5). Permissive serde exactly like `BookArtSpec` / `PersonaSpec`: **every field is optional** (a bare
//! `{}` resolves to a neutral 1K material), enums are carried as strings (unknown values load and are
//! caught by `lint`, not a hard failure), and unknown *keys* are ignored (forward-compatible).

use serde::Deserialize;

/// The spec schema version this build understands.
pub const SCHEMA_VERSION: &str = "texture/1";

/// A full material spec. All fields optional.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TextureSpec {
    pub schema: Option<String>,
    /// The prompt (text-to-material).
    pub material: Option<String>,
    /// Image-to-material: a photo → a tileable PBR set (crop-to-tileable + delight).
    pub from_image: Option<String>,

    pub seamless: Option<Seamless>,
    pub channels: Option<Channels>,
    /// Delight the albedo (flatten baked lighting) via IC-Light. Default true.
    pub delight: Option<bool>,
    pub page: Option<Page>,
    pub export: Option<Export>,

    /// Diffusion base for the generation passes (default `sdxl`).
    pub model: Option<String>,
    pub seed: Option<u64>,
    pub steps: Option<usize>,
}

/// Seamless-tiling control (RFC §7).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Seamless {
    /// `circular` (default) | `offset` | `mirror` | `none`.
    pub mode: Option<String>,
    /// `both` (default) | `x` | `y` — a trim sheet tiles one axis.
    pub axes: Option<String>,
}

/// Per-channel controls (RFC §8). `roughness`/`metallic`/`height` accept a **scalar**, the string
/// `"from-albedo"`, or a `"<prompt>"` — carried as a raw JSON value and interpreted at resolve time.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Channels {
    /// `auto` (depth-CN pass) | `from-albedo` | `"<prompt>"`.
    pub height: Option<String>,
    /// scalar `[0,1]` | `"from-albedo"` | `"<prompt>"`.
    pub roughness: Option<serde_json::Value>,
    /// scalar `[0,1]` | `"from-albedo"` | `"<prompt>"`.
    pub metallic: Option<serde_json::Value>,
    /// Slope gain when deriving normal from height.
    pub normal_strength: Option<f32>,
    pub ao_strength: Option<f32>,
    /// `opengl` (+Y, default) | `directx` (-Y).
    pub normal_y: Option<String>,
    /// C1 (6.4.0): anisotropy strength `[0,1]` for brushed/grained metals (0 = isotropic, default).
    /// Emits an anisotropy flow+strength map and stretches the preview highlight along the grain.
    pub anisotropy: Option<f32>,
    /// Grain direction in degrees. Omit for `auto` (dominant grain direction from the height's
    /// structure tensor). Only used when `anisotropy > 0`.
    pub anisotropy_angle: Option<f32>,
}

/// The raster target (RFC §5). The name echoes `bookart`'s `page`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Page {
    /// Native square generation size (default 1024).
    pub size: Option<u32>,
    /// `none` (default) | `2k` | `4k` — tiled, tileability-preserving.
    pub upscale: Option<String>,
    /// Run the seam scorecard (default true).
    pub tiling_check: Option<bool>,
}

/// Export controls (RFC §11).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Export {
    /// Which channel maps to write (default: the full set).
    pub maps: Option<Vec<String>>,
    /// Also write the packed ORM (R=AO, G=roughness, B=metallic). Default true.
    pub orm: Option<bool>,
    /// Also write a glTF 2.0 material. Default false.
    pub gltf: Option<bool>,
    /// Channel filename convention: `plakat` (default) | `unity` | `unreal`.
    pub naming: Option<String>,
    /// Render the lit preview. Default true.
    pub preview: Option<bool>,
}

impl TextureSpec {
    /// Parse from an HJSON string. NB (bookart gotcha): `deser_hjson` quoteless strings run to EOL, so
    /// inline object/array string values in a spec must be JSON-quoted.
    pub fn from_hjson(text: &str) -> Result<Self, deser_hjson::Error> {
        deser_hjson::from_str(text)
    }

    /// Load + parse a spec file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Self::from_hjson(&text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_spec_parses() {
        let s = TextureSpec::from_hjson("{}").unwrap();
        assert!(s.material.is_none() && s.seamless.is_none());
    }

    #[test]
    fn full_spec_parses_including_scalar_or_string_channels() {
        let s = TextureSpec::from_hjson(
            r#"{
                schema: "texture/1"
                material: "mossy cobblestone"
                seamless: { mode: "circular", axes: "both" }
                channels: { height: "auto", roughness: 0.6, metallic: "from-albedo", normal_strength: 1.2, normal_y: "opengl" }
                delight: true
                page: { size: 1024, upscale: "2k" }
                export: { maps: ["albedo","normal"], orm: true, naming: "unreal" }
                model: "sdxl"  seed: 7  steps: 28
            }"#,
        )
        .unwrap();
        assert_eq!(s.material.as_deref(), Some("mossy cobblestone"));
        let ch = s.channels.unwrap();
        assert_eq!(ch.roughness.unwrap().as_f64(), Some(0.6)); // scalar
        assert_eq!(ch.metallic.unwrap().as_str(), Some("from-albedo")); // string
        assert_eq!(s.page.unwrap().upscale.as_deref(), Some("2k"));
        assert_eq!(s.export.unwrap().naming.as_deref(), Some("unreal"));
    }

    #[test]
    fn unknown_keys_are_ignored() {
        // Forward-compatible: a future key doesn't break today's parser.
        let s = TextureSpec::from_hjson(r#"{ material: "brick", future_knob: 3 }"#).unwrap();
        assert_eq!(s.material.as_deref(), Some("brick"));
    }
}
