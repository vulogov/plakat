//! The deterministic resolver (RFC TEXTURE-1 §6). A `TextureSpec` → a byte-stable [`RenderPlan`]:
//! fill defaults, resolve the seamless mode, the channel sources, the export plan, and the compiled
//! diffusion prompt/negative (flat-lighting + tileable anchors baked in). Pure — no weights, no I/O.

use crate::texture::spec::TextureSpec;
use serde::{Deserialize, Serialize};

/// How a non-albedo channel is produced (RFC §8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChannelSource {
    /// A flat scalar map at this value (`[0,1]`).
    Scalar(f32),
    /// Derived from the albedo by a luminance/heuristic pass.
    FromAlbedo,
    /// A dedicated albedo-conditioned diffusion pass with this prompt.
    Prompt(String),
}

/// How the height channel is produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HeightSource {
    /// A depth-ControlNet pass conditioned on the albedo.
    Auto,
    /// A fast luminance→height from the albedo.
    FromAlbedo,
    /// A dedicated diffusion pass with this prompt.
    Prompt(String),
}

/// A fully-resolved plan for one material — everything the render/derive/export phases need.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderPlan {
    pub schema_ok: bool,
    /// The albedo prompt (empty for a pure image-to-material job).
    pub material: String,
    pub from_image: Option<String>,

    pub seamless_mode: String, // circular | offset | none
    pub seamless_axes: String, // both | x | y

    pub size: u32,
    pub upscale: String, // none | 2k | 4k
    pub tiling_check: bool,

    pub height: HeightSource,
    pub roughness: ChannelSource,
    pub metallic: ChannelSource,
    pub normal_strength: f32,
    pub ao_strength: f32,
    pub normal_y: String, // opengl | directx

    pub delight: bool,

    pub maps: Vec<String>, // which channel maps to write, in order
    pub orm: bool,
    pub gltf: bool,
    pub naming: String, // plakat | unity | unreal
    pub preview: bool,

    pub model: String,
    pub seed: u64,
    pub steps: usize,

    /// The compiled albedo diffusion prompt (material + flat-light + tileable anchors).
    pub prompt: String,
    /// The compiled negative (anti-lighting-bake + anti-seam anchors).
    pub negative: String,
}

/// The full PBR channel set, in canonical order.
pub const ALL_MAPS: &[&str] = &["albedo", "normal", "roughness", "metallic", "height", "ao"];

const FLAT_ANCHOR: &str = "flat even lighting, top-down orthographic, a seamless tileable texture, material sample, no shadows, no highlights, uniform";
const NEG_ANCHOR: &str = "cast shadow, hard shadow, directional light, vignette, gradient lighting, specular highlight, seam, border, frame, perspective, object, watermark";

/// Interpret a scalar-or-string channel value (`0.6` | `"from-albedo"` | `"<prompt>"`).
fn channel_source(v: Option<&serde_json::Value>, default: f32) -> ChannelSource {
    match v {
        Some(serde_json::Value::Number(n)) => ChannelSource::Scalar(n.as_f64().unwrap_or(default as f64) as f32),
        Some(serde_json::Value::String(s)) if s.eq_ignore_ascii_case("from-albedo") => ChannelSource::FromAlbedo,
        Some(serde_json::Value::String(s)) => ChannelSource::Prompt(s.clone()),
        _ => ChannelSource::Scalar(default),
    }
}

fn height_source(h: Option<&str>) -> HeightSource {
    match h {
        None | Some("auto") => HeightSource::Auto,
        Some("from-albedo") => HeightSource::FromAlbedo,
        Some(p) => HeightSource::Prompt(p.to_string()),
    }
}

/// Resolve a spec to a byte-stable [`RenderPlan`].
pub fn resolve(spec: &TextureSpec) -> RenderPlan {
    let seamless = spec.seamless.clone().unwrap_or_default();
    let ch = spec.channels.clone().unwrap_or_default();
    let page = spec.page.clone().unwrap_or_default();
    let exp = spec.export.clone().unwrap_or_default();

    let material = spec.material.clone().unwrap_or_default();
    let prompt = if material.trim().is_empty() {
        String::new() // pure image-to-material: no albedo prompt
    } else {
        format!("{material}, {FLAT_ANCHOR}")
    };

    // Only the canonical map names survive, in canonical order; unknown entries dropped (lint warns).
    let maps = match &exp.maps {
        Some(m) => ALL_MAPS.iter().filter(|k| m.iter().any(|x| x == *k)).map(|s| s.to_string()).collect(),
        None => ALL_MAPS.iter().map(|s| s.to_string()).collect(),
    };

    RenderPlan {
        schema_ok: spec.schema.as_deref().map(|s| s == super::SCHEMA_VERSION).unwrap_or(true),
        material,
        from_image: spec.from_image.clone(),
        seamless_mode: seamless.mode.clone().unwrap_or_else(|| "circular".into()),
        seamless_axes: seamless.axes.clone().unwrap_or_else(|| "both".into()),
        size: page.size.unwrap_or(1024).clamp(64, 8192),
        upscale: page.upscale.clone().unwrap_or_else(|| "none".into()),
        tiling_check: page.tiling_check.unwrap_or(true),
        height: height_source(ch.height.as_deref()),
        roughness: channel_source(ch.roughness.as_ref(), 0.6),
        metallic: channel_source(ch.metallic.as_ref(), 0.0),
        normal_strength: ch.normal_strength.unwrap_or(1.0).clamp(0.0, 8.0),
        ao_strength: ch.ao_strength.unwrap_or(1.0).clamp(0.0, 4.0),
        normal_y: ch.normal_y.clone().unwrap_or_else(|| "opengl".into()),
        delight: spec.delight.unwrap_or(true),
        maps,
        orm: exp.orm.unwrap_or(true),
        gltf: exp.gltf.unwrap_or(false),
        naming: exp.naming.clone().unwrap_or_else(|| "plakat".into()),
        preview: exp.preview.unwrap_or(true),
        model: spec.model.clone().unwrap_or_else(|| "sdxl".into()),
        seed: spec.seed.unwrap_or(0),
        steps: spec.steps.unwrap_or(28),
        prompt,
        negative: NEG_ANCHOR.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_spec_resolves_to_a_neutral_1k_material() {
        let p = resolve(&TextureSpec::default());
        assert_eq!(p.size, 1024);
        assert_eq!(p.seamless_mode, "circular");
        assert_eq!(p.seamless_axes, "both");
        assert_eq!(p.height, HeightSource::Auto);
        assert_eq!(p.roughness, ChannelSource::Scalar(0.6));
        assert_eq!(p.metallic, ChannelSource::Scalar(0.0));
        assert_eq!(p.normal_y, "opengl");
        assert!(p.delight && p.orm && p.preview && !p.gltf);
        assert_eq!(p.maps.len(), 6);
        assert!(p.prompt.is_empty(), "no material → no albedo prompt");
    }

    #[test]
    fn resolve_is_deterministic_and_compiles_the_prompt() {
        let spec = TextureSpec::from_hjson(
            r#"{ material: "rusted iron plate", channels: { roughness: "from-albedo", metallic: 1.0 }, seed: 3 }"#,
        )
        .unwrap();
        assert_eq!(resolve(&spec), resolve(&spec));
        let p = resolve(&spec);
        assert_eq!(p.roughness, ChannelSource::FromAlbedo);
        assert_eq!(p.metallic, ChannelSource::Scalar(1.0));
        assert!(p.prompt.starts_with("rusted iron plate, "));
        assert!(p.prompt.contains("seamless tileable"), "flat/tileable anchor baked in");
        assert!(p.negative.contains("shadow") && p.negative.contains("seam"));
    }

    #[test]
    fn channels_can_be_a_prompt() {
        let spec = TextureSpec::from_hjson(r#"{ material: "brick", channels: { height: "deep mortar grooves", roughness: "worn patches" } }"#).unwrap();
        let p = resolve(&spec);
        assert_eq!(p.height, HeightSource::Prompt("deep mortar grooves".into()));
        assert_eq!(p.roughness, ChannelSource::Prompt("worn patches".into()));
    }

    #[test]
    fn maps_are_filtered_to_canonical_order() {
        let spec = TextureSpec::from_hjson(r#"{ export: { maps: ["normal","bogus","albedo"] } }"#).unwrap();
        let p = resolve(&spec);
        assert_eq!(p.maps, vec!["albedo".to_string(), "normal".to_string()]); // canonical order, bogus dropped
    }
}
