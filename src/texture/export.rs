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
/// `metallicRoughnessTexture` (GB) + `occlusionTexture` (R) layout, and Unreal's / Godot's ORM.
pub fn orm_pack(m: &Material) -> RgbImage {
    let (w, h) = m.ao.dimensions();
    let g = |img: &GrayImage, x: u32, y: u32| img.get_pixel(x.min(img.width() - 1), y.min(img.height() - 1)).0[0];
    RgbImage::from_fn(w, h, |x, y| Rgb([g(&m.ao, x, y), g(&m.roughness, x, y), g(&m.metallic, x, y)]))
}

/// Pack the **Unity HDRP mask map** — a DIFFERENT convention from ORM: `R=metallic, G=ambient-occlusion,
/// B=detail-mask (neutral 128), A=smoothness (= 255 − roughness)`. (6.6.0 / G0.1.)
pub fn mask_map_pack(m: &Material) -> image::RgbaImage {
    let (w, h) = m.metallic.dimensions();
    let g = |img: &GrayImage, x: u32, y: u32| img.get_pixel(x.min(img.width() - 1), y.min(img.height() - 1)).0[0];
    image::RgbaImage::from_fn(w, h, |x, y| {
        image::Rgba([g(&m.metallic, x, y), g(&m.ao, x, y), 128, 255 - g(&m.roughness, x, y)])
    })
}

/// A material export target (6.6.0). Each pins a filename convention + how the packed channels lay out —
/// the conventions genuinely differ per engine and a wrong pack fails *silently* in-engine, so they live
/// in ONE place (this enum), verified by `packing_conventions_are_locked`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Raw plakat names, ORM pack, no material doc.
    Plakat,
    /// glTF 2.0 material doc; ORM (metallicRoughness = GB, occlusion = R).
    Gltf,
    /// Unreal `T_*` names; ORM.
    Unreal,
    /// Godot names; ORM.
    Godot,
    /// Unity HDRP; the **mask map** pack (R=metal/G=AO/B=detail/A=smoothness), NOT ORM.
    UnityHdrp,
    /// MaterialX `.mtlx` `standard_surface`; separate channel textures.
    MaterialX,
}

/// Which packed texture an engine consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing {
    /// R=AO, G=roughness, B=metallic (glTF / Unreal / Godot).
    Orm,
    /// R=metallic, G=AO, B=detail, A=smoothness (Unity HDRP).
    MaskMap,
}

impl Engine {
    pub fn parse(s: &str) -> Option<Engine> {
        Some(match s.to_ascii_lowercase().as_str() {
            "plakat" => Engine::Plakat,
            "gltf" | "gltf2" => Engine::Gltf,
            "unreal" => Engine::Unreal,
            "godot" => Engine::Godot,
            "unity-hdrp" | "hdrp" | "unity_hdrp" => Engine::UnityHdrp,
            "materialx" | "mtlx" => Engine::MaterialX,
            _ => return None,
        })
    }
    /// The filename convention key (`channel_filename`'s `naming`).
    pub fn naming(&self) -> &'static str {
        match self {
            Engine::Unreal => "unreal",
            Engine::UnityHdrp => "unity",
            _ => "plakat",
        }
    }
    /// The packed-map convention this engine expects.
    pub fn packing(&self) -> Packing {
        match self {
            Engine::UnityHdrp => Packing::MaskMap,
            _ => Packing::Orm,
        }
    }
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

/// A complete glTF 2.0 material document (baseColor / metallic-roughness = ORM GB / occlusion = ORM R
/// with strength / normal with scale). References the naming-convention filenames written alongside it.
/// When the material has an anisotropy map, emits the **`KHR_materials_anisotropy`** extension (its
/// texture is our flow map: RG = tangent-space direction, B = strength — the KHR convention). Material-
/// only (no mesh) — a drop-in for import. Built via `serde_json` so it's always valid.
fn gltf_document(naming: &str, has_anisotropy: bool) -> String {
    use serde_json::json;
    let f = |m: &str| channel_filename(m, naming);
    // images/textures 0=albedo 1=normal 2=orm (+ 3=anisotropy when present).
    let mut images = vec![json!({ "uri": f("albedo") }), json!({ "uri": f("normal") }), json!({ "uri": f("orm") })];
    let mut textures = vec![
        json!({ "source": 0, "sampler": 0 }),
        json!({ "source": 1, "sampler": 0 }),
        json!({ "source": 2, "sampler": 0 }),
    ];
    let mut material = json!({
        "name": "plakat_material",
        "pbrMetallicRoughness": {
            "baseColorTexture": { "index": 0 },
            "metallicRoughnessTexture": { "index": 2 }
        },
        "normalTexture": { "index": 1, "scale": 1.0 },
        "occlusionTexture": { "index": 2, "strength": 1.0 }
    });
    let mut extensions_used: Vec<&str> = Vec::new();
    if has_anisotropy {
        images.push(json!({ "uri": f("anisotropy") }));
        textures.push(json!({ "source": 3, "sampler": 0 }));
        material["extensions"] = json!({
            "KHR_materials_anisotropy": {
                "anisotropyStrength": 1.0,
                "anisotropyRotation": 0.0,
                "anisotropyTexture": { "index": 3 }
            }
        });
        extensions_used.push("KHR_materials_anisotropy");
    }
    let mut doc = json!({
        "asset": { "version": "2.0", "generator": format!("plakat texture {}", env!("CARGO_PKG_VERSION")) },
        "images": images,
        "samplers": [ { "wrapS": 10497, "wrapT": 10497 } ],
        "textures": textures,
        "materials": [ material ]
    });
    if !extensions_used.is_empty() {
        doc["extensionsUsed"] = json!(extensions_used);
    }
    serde_json::to_string_pretty(&doc).unwrap_or_default()
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
    // C1: the anisotropy flow+strength map is an optional extra (not in the canonical 6) — write it when
    // present. Consumed by engine anisotropy workflows (e.g. glTF KHR_materials_anisotropy).
    if m.anisotropy.is_some() {
        if let Some(img) = m.channel("anisotropy") {
            let name = channel_filename("anisotropy", &plan.naming);
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
        std::fs::write(dir.join("material.gltf"), gltf_document(&plan.naming, m.anisotropy.is_some())).context("writing material.gltf")?;
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
    fn packing_conventions_are_locked() {
        use image::Luma;
        // Distinct constant channels so the channel ORDER is unambiguous: AO=50, roughness=100, metal=200.
        let flat = |v: u8| GrayImage::from_pixel(8, 8, Luma([v]));
        let m = Material {
            albedo: RgbImage::from_pixel(8, 8, Rgb([10, 20, 30])),
            normal: RgbImage::from_pixel(8, 8, Rgb([128, 128, 255])),
            height: flat(64),
            roughness: flat(100),
            metallic: flat(200),
            ao: flat(50),
            anisotropy: None,
        };
        // ORM (glTF metallicRoughness=GB, occlusion=R · Unreal/Godot): R=AO, G=rough, B=metal.
        let orm = orm_pack(&m).get_pixel(4, 4).0;
        assert_eq!(orm, [50, 100, 200], "ORM = R:AO G:rough B:metal");
        // Unity HDRP mask map: R=metal, G=AO, B=detail(128), A=smoothness(=255-rough).
        let mm = mask_map_pack(&m).get_pixel(4, 4).0;
        assert_eq!(mm, [200, 50, 128, 155], "HDRP mask = R:metal G:AO B:detail A:smoothness(255-100)");
        // The Engine table routes each target to the right packing + naming (one source of truth).
        assert_eq!(Engine::parse("unity-hdrp"), Some(Engine::UnityHdrp));
        assert_eq!(Engine::UnityHdrp.packing(), Packing::MaskMap);
        assert_eq!(Engine::Gltf.packing(), Packing::Orm);
        assert_eq!(Engine::Unreal.packing(), Packing::Orm);
        assert_eq!(Engine::Unreal.naming(), "unreal");
        assert_eq!(Engine::UnityHdrp.naming(), "unity");
        assert_eq!(Engine::Gltf.naming(), "plakat");
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
        // A complete glTF: metallicRoughness + occlusion(strength) + normal(scale), no anisotropy ext here.
        let mat = &gltf["materials"][0];
        assert_eq!(mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"], 2, "metallicRoughness = ORM");
        assert_eq!(mat["occlusionTexture"]["strength"], 1.0);
        assert_eq!(mat["normalTexture"]["scale"], 1.0);
        assert!(gltf.get("extensionsUsed").is_none(), "no anisotropy → no extensions");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gltf_emits_khr_anisotropy_when_the_material_has_a_flow_map() {
        let mut m = Material::derive(
            RgbImage::from_pixel(16, 16, Rgb([180, 180, 185])),
            None, 1.0, true, 1.0, &ChannelSource::Scalar(0.3), &ChannelSource::Scalar(1.0),
        );
        // give it an anisotropy flow map (as `render` does for a brushed metal)
        m.anisotropy = Some(RgbImage::from_pixel(16, 16, Rgb([255, 128, 217])));
        let dir = std::env::temp_dir().join(format!("plakat-tex-aniso-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut plan = compile::resolve(&TextureSpec::default());
        plan.gltf = true;
        let sc = crate::texture::scorecard::score(&m);
        write_material(&m, &plan, &sc, &dir).unwrap();
        assert!(dir.join("anisotropy.png").exists(), "anisotropy channel written");
        let gltf: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("material.gltf")).unwrap()).unwrap();
        assert_eq!(gltf["extensionsUsed"][0], "KHR_materials_anisotropy");
        let ext = &gltf["materials"][0]["extensions"]["KHR_materials_anisotropy"];
        assert_eq!(ext["anisotropyTexture"]["index"], 3, "the 4th texture is the flow map");
        assert_eq!(ext["anisotropyStrength"], 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
