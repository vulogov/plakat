//! Engine-ready export (RFC TEXTURE-1 §11). Packs the derived [`Material`] into a material directory:
//! the channel PNGs under a naming convention, the **ORM** pack (R=AO, G=roughness, B=metallic — the
//! glTF/Unreal layout), an optional **glTF 2.0** material, a lit **preview**, and a `material.json`
//! manifest (recipe + channel list + scorecard + a stable spec-hash). Pure, weight-free.

use crate::texture::compile::{ChannelSource, HeightSource, RenderPlan};
use crate::texture::derive::Material;
use crate::texture::preview::{self, Shape};
use crate::texture::scorecard::Scorecard;
use anyhow::{Context, Result};
use image::{GrayImage, Rgb, RgbImage};
use serde::Serialize;

/// Pack AO / roughness / metallic into one RGB image (R=AO, G=roughness, B=metallic) — the glTF
/// `metallicRoughnessTexture` (GB) + `occlusionTexture` (R) layout, and Unreal's ORM.
pub fn orm_pack(m: &Material) -> RgbImage {
    let (w, h) = m.ao.dimensions();
    let g = |img: &GrayImage, x: u32, y: u32| img.get_pixel(x.min(img.width() - 1), y.min(img.height() - 1)).0[0];
    RgbImage::from_fn(w, h, |x, y| Rgb([g(&m.ao, x, y), g(&m.roughness, x, y), g(&m.metallic, x, y)]))
}

/// The on-disk filename for a channel under a naming convention.
pub fn channel_filename(map: &str, naming: &str) -> String {
    match naming {
        "unity" => match map {
            "albedo" => "albedo_MainTex.png",
            "normal" => "normal_BumpMap.png",
            "metallic" => "metallic_Metallic.png",
            "ao" => "ao_Occlusion.png",
            "height" => "height_Parallax.png",
            "orm" => "orm.png",
            other => return format!("{other}.png"),
        },
        "unreal" => match map {
            "albedo" => "T_BaseColor.png",
            "normal" => "T_Normal.png",
            "roughness" => "T_Roughness.png",
            "metallic" => "T_Metallic.png",
            "height" => "T_Height.png",
            "ao" => "T_AO.png",
            "orm" => "T_ORM.png",
            other => return format!("T_{other}.png"),
        },
        _ => return format!("{map}.png"), // plakat
    }
    .to_string()
}

#[derive(Serialize)]
struct ScoreSummary {
    tileability_x: f32,
    tileability_y: f32,
    normal_valid: f32,
    albedo_flatness: f32,
    passes: bool,
}

#[derive(Serialize)]
struct Manifest {
    schema: String,
    material: String,
    seamless: String,
    size: u32,
    naming: String,
    normal_y: String,
    maps: Vec<String>,
    orm: Option<String>,
    gltf: Option<String>,
    preview: Option<String>,
    spec_hash: String,
    scorecard: ScoreSummary,
    generator: String,
}

/// A stable FNV-1a fingerprint of the plan identity (the bookart spec-hash pattern).
fn spec_hash(plan: &RenderPlan) -> String {
    let rough = match &plan.roughness {
        ChannelSource::Scalar(v) => format!("s{v:.3}"),
        ChannelSource::FromAlbedo => "albedo".into(),
        ChannelSource::Auto => "auto".into(),
        ChannelSource::Prompt(p) => format!("p:{p}"),
    };
    let height = match &plan.height {
        HeightSource::Auto => "auto".into(),
        HeightSource::FromAlbedo => "albedo".into(),
        HeightSource::Prompt(p) => format!("p:{p}"),
    };
    let id = format!(
        "{}|{}|{}|{}|{}|{:.2}|{}|{}",
        plan.material, plan.seamless_mode, plan.seamless_axes, plan.size, height, plan.normal_strength, rough, plan.model
    );
    let mut hh: u64 = 0xcbf29ce484222325;
    for b in id.as_bytes() {
        hh ^= *b as u64;
        hh = hh.wrapping_mul(0x100000001b3);
    }
    format!("{hh:016x}")
}

/// A minimal glTF 2.0 document describing the material (baseColor / normal / ORM). References the
/// naming-convention filenames written alongside it. Material-only (no mesh) — a drop-in for import.
fn gltf_document(naming: &str) -> String {
    let f = |m: &str| channel_filename(m, naming);
    format!(
        r#"{{
  "asset": {{ "version": "2.0", "generator": "plakat texture {ver}" }},
  "images": [
    {{ "uri": "{albedo}" }},
    {{ "uri": "{normal}" }},
    {{ "uri": "{orm}" }}
  ],
  "samplers": [ {{ "wrapS": 10497, "wrapT": 10497 }} ],
  "textures": [
    {{ "source": 0, "sampler": 0 }},
    {{ "source": 1, "sampler": 0 }},
    {{ "source": 2, "sampler": 0 }}
  ],
  "materials": [
    {{
      "name": "plakat_material",
      "pbrMetallicRoughness": {{
        "baseColorTexture": {{ "index": 0 }},
        "metallicRoughnessTexture": {{ "index": 2 }}
      }},
      "normalTexture": {{ "index": 1 }},
      "occlusionTexture": {{ "index": 2 }}
    }}
  ]
}}
"#,
        ver = env!("CARGO_PKG_VERSION"),
        albedo = f("albedo"),
        normal = f("normal"),
        orm = f("orm"),
    )
}

/// Write the full material directory: channel PNGs (named), ORM, preview, glTF, and `material.json`.
pub fn write_material(m: &Material, plan: &RenderPlan, sc: &Scorecard, dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut written = Vec::new();
    for map in &plan.maps {
        if let Some(img) = m.channel(map) {
            let name = channel_filename(map, &plan.naming);
            img.save(dir.join(&name)).with_context(|| format!("writing {name}"))?;
            written.push(name);
        }
    }
    let orm_name = if plan.orm {
        let name = channel_filename("orm", &plan.naming);
        orm_pack(m).save(dir.join(&name)).with_context(|| format!("writing {name}"))?;
        Some(name)
    } else {
        None
    };
    let preview_name = if plan.preview {
        let img = preview::render(m, Shape::Sphere, 512);
        img.save(dir.join("preview.png")).context("writing preview.png")?;
        Some("preview.png".to_string())
    } else {
        None
    };
    let gltf_name = if plan.gltf {
        std::fs::write(dir.join("material.gltf"), gltf_document(&plan.naming)).context("writing material.gltf")?;
        Some("material.gltf".to_string())
    } else {
        None
    };

    let manifest = Manifest {
        schema: super::SCHEMA_VERSION.to_string(),
        material: plan.material.clone(),
        seamless: format!("{}:{}", plan.seamless_mode, plan.seamless_axes),
        size: plan.size,
        naming: plan.naming.clone(),
        normal_y: plan.normal_y.clone(),
        maps: written,
        orm: orm_name,
        gltf: gltf_name,
        preview: preview_name,
        spec_hash: spec_hash(plan),
        scorecard: ScoreSummary {
            tileability_x: sc.tileability_x,
            tileability_y: sc.tileability_y,
            normal_valid: sc.normal_valid,
            albedo_flatness: sc.albedo_flatness,
            passes: sc.passes,
        },
        generator: format!("plakat {}", env!("CARGO_PKG_VERSION")),
    };
    std::fs::write(dir.join("material.json"), serde_json::to_string_pretty(&manifest)?).context("writing material.json")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::{compile, TextureSpec};

    #[test]
    fn orm_packs_ao_rough_metal_into_rgb() {
        let albedo = RgbImage::from_pixel(8, 8, Rgb([100, 100, 100]));
        let m = Material::derive(albedo, None, 1.0, true, 1.0, &ChannelSource::Scalar(0.7), &ChannelSource::Scalar(0.3));
        let orm = orm_pack(&m);
        let p = orm.get_pixel(4, 4).0;
        assert_eq!(p[1], 179, "G = roughness 0.7 → 179"); // 0.7*255 = 178.5 → 179
        assert_eq!(p[2], 77, "B = metallic 0.3 → 77"); // 0.3*255 = 76.5 → 77
    }

    #[test]
    fn naming_conventions_map_channels() {
        assert_eq!(channel_filename("albedo", "plakat"), "albedo.png");
        assert_eq!(channel_filename("normal", "unity"), "normal_BumpMap.png");
        assert_eq!(channel_filename("albedo", "unreal"), "T_BaseColor.png");
        assert_eq!(channel_filename("orm", "unreal"), "T_ORM.png");
    }

    #[test]
    fn write_material_produces_the_full_dir_and_valid_json_gltf() {
        let dir = std::env::temp_dir().join(format!("plakat-tex-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let spec = TextureSpec::from_hjson(r#"{ material: "brick", export: { gltf: true, naming: "unreal" } }"#).unwrap();
        let plan = compile::resolve(&spec);
        let albedo = RgbImage::from_fn(16, 16, |x, y| Rgb([(x * 16) as u8, (y * 16) as u8, 120]));
        let m = Material::derive(albedo, None, 1.0, true, 1.0, &plan.roughness, &plan.metallic);
        let sc = crate::texture::scorecard::score(&m);
        write_material(&m, &plan, &sc, &dir).unwrap();
        assert!(dir.join("T_BaseColor.png").exists() && dir.join("T_ORM.png").exists());
        assert!(dir.join("preview.png").exists());
        // glTF + manifest parse as JSON.
        let gltf: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("material.gltf")).unwrap()).unwrap();
        assert_eq!(gltf["asset"]["version"], "2.0");
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("material.json")).unwrap()).unwrap();
        assert_eq!(man["naming"], "unreal");
        assert_eq!(man["spec_hash"].as_str().unwrap().len(), 16);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
