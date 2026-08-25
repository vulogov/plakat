//! The render router (RFC TEXTURE-1 §6, ROADMAP B4) — the weights-bearing half. Resolve → generate a
//! flat, tileable **albedo** → make it seamless (post-hoc feather; a per-step latent roll is the
//! escalation) → optional IC-Light **delight** → **derive** the rest of the PBR set (B1) → **measure**
//! (B1 scorecard) → **export** the engine-ready material directory (B2).

use crate::texture::compile::{self, RenderPlan};
use crate::texture::derive::Material;
use crate::texture::seamless::{self, Axes};
use crate::texture::spec::TextureSpec;
use crate::texture::{export, scorecard, Scorecard};
use anyhow::{Context, Result};
use console::style;

/// Knobs for a texture render.
#[derive(Debug, Clone)]
pub struct RenderOpts {
    /// Diffusion rejection sampling: try up to N seeds, keep the first that clears the scorecard.
    pub attempts: u32,
    /// Override the spec's `page.upscale` (`none`/`2k`/`4k`).
    pub upscale: Option<String>,
    /// A3: a hand-painted metallic / roughness mask PNG to use verbatim (overrides derivation).
    pub metallic_ref: Option<std::path::PathBuf>,
    pub roughness_ref: Option<std::path::PathBuf>,
    /// 6.6.0: an engine export preset (`gltf`/`unreal`/`unity-hdrp`/`godot`/`materialx`/`plakat`) applied
    /// to the output (naming + packing + material doc).
    pub engine: Option<String>,
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self { attempts: 1, upscale: None, metallic_ref: None, roughness_ref: None, engine: None }
    }
}

/// Load a hand-painted mask PNG and resize to `w×h` (the A3 `--*-ref` override).
fn load_mask_ref(path: &std::path::Path, w: u32, h: u32) -> Result<image::GrayImage> {
    let g = image::open(path).with_context(|| format!("opening mask {}", path.display()))?.to_luma8();
    Ok(if g.dimensions() == (w, h) {
        g
    } else {
        image::imageops::resize(&g, w, h, image::imageops::FilterType::Lanczos3)
    })
}

fn upscale_target(s: &str) -> Option<u32> {
    match s.to_ascii_lowercase().as_str() {
        "2k" => Some(2048),
        "4k" => Some(4096),
        _ => None,
    }
}

/// Wrap-pad an RGB image by `p` px (circular) — so a Lanczos upscale of a tileable albedo stays tileable.
fn circular_pad_rgb(a: &image::RgbImage, p: u32) -> image::RgbImage {
    let (w, h) = a.dimensions();
    image::RgbImage::from_fn(w + 2 * p, h + 2 * p, |x, y| {
        let sx = ((x as i64 - p as i64).rem_euclid(w as i64)) as u32;
        let sy = ((y as i64 - p as i64).rem_euclid(h as i64)) as u32;
        *a.get_pixel(sx, sy)
    })
}

/// A **tileability-preserving** Lanczos upscale to `target`² (circular-pad → resize → crop) — local
/// resampling keeps the wrap; Real-ESRGAN is avoided here (it tiles internally + hallucinates, breaking
/// the seam).
fn upscale_tileable(a: &image::RgbImage, target: u32) -> image::RgbImage {
    let (w, h) = a.dimensions();
    if target <= w {
        return a.clone();
    }
    let pad = (w / 16).max(8);
    let padded = circular_pad_rgb(a, pad);
    let scale = target as f32 / w as f32;
    let (pw, ph) = (((w + 2 * pad) as f32 * scale).round() as u32, ((h + 2 * pad) as f32 * scale).round() as u32);
    let up = image::imageops::resize(&padded, pw, ph, image::imageops::FilterType::Lanczos3);
    let cp = (pad as f32 * scale).round() as u32;
    image::imageops::crop_imm(&up, cp, cp, target, target).to_image()
}

/// Height via Depth-Anything-V2 on the albedo (the `height: auto` path) — brighter (closer) = raised.
async fn height_auto(albedo: &image::RgbImage) -> Result<image::GrayImage> {
    let device = crate::api::device("auto")?;
    let (w, h) = albedo.dimensions();
    let tmp = std::env::temp_dir().join(format!("plakat_tex_h_{w}x{h}.png"));
    albedo.save(&tmp)?;
    let depth = crate::pipelines::depth::DepthPipeline::load(device).await.context("loading depth model")?;
    let d = depth.depth_map(&tmp, w, h).context("depth estimation");
    let _ = std::fs::remove_file(&tmp);
    let d = d?;
    // Depth gives the smooth macro relief but a low-contrast (flat-normal) field; add the luminance
    // **high-pass** for crisp per-feature micro-detail so the derived normal has both macro tilt + bite.
    let luma: Vec<f32> = albedo.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0).collect();
    let mut luma_img = image::GrayImage::new(w, h);
    for (i, p) in luma_img.pixels_mut().enumerate() {
        p.0[0] = (luma[i] * 255.0) as u8;
    }
    let low = image::imageops::blur(&luma_img, (w.min(h) as f32 / 48.0).max(1.0));
    let mut g = image::GrayImage::new(w, h);
    for (i, p) in g.pixels_mut().enumerate() {
        let hp = luma[i] - low.get_pixel((i as u32) % w, (i as u32) / w).0[0] as f32 / 255.0;
        let v = (d[i] + 0.6 * hp).clamp(0.0, 1.0);
        p.0[0] = (v * 255.0).round() as u8;
    }
    Ok(g)
}

fn to_rgb(img: &crate::api::Image) -> Result<image::RgbImage> {
    image::RgbImage::from_raw(img.width(), img.height(), img.pixels().to_vec())
        .context("generated albedo had a malformed buffer")
}

/// Generate (or load) the albedo, then make it seamless.
async fn albedo_for(plan: &RenderPlan, seed: u64) -> Result<image::RgbImage> {
    let is_photo = plan.from_image.is_some();
    let mut albedo = if let Some(path) = &plan.from_image {
        // Image-to-material (B6): load + centre-square-crop + resize to the working size.
        println!("  {} image-to-material from {path}", style("→").cyan());
        let img = image::open(path).with_context(|| format!("reading {path}"))?.to_rgb8();
        let s = img.width().min(img.height());
        let (ox, oy) = ((img.width() - s) / 2, (img.height() - s) / 2);
        image::imageops::resize(&image::imageops::crop_imm(&img, ox, oy, s, s).to_image(), plan.size, plan.size, image::imageops::FilterType::Lanczos3)
    } else {
        println!("  {} albedo {}² · {} steps · seed {seed}…", style("→").cyan(), plan.size, plan.steps);
        let imgs = crate::api::Generate::new(&plan.model)
            .prompt(&plan.prompt)
            .negative(&plan.negative)
            .size(plan.size, plan.size)
            .steps(plan.steps)
            .seed(seed)
            .run()
            .await
            .context("albedo generation")?;
        to_rgb(imgs.first().context("generation produced no image")?)?
    };
    // Seamless. A generated albedo is already near-tileable (flat/tileable prompt) → a boundary feather.
    // A photo isn't → the offset-and-heal `make_tileable` (roll + central-cross feather).
    if plan.seamless_mode == "mirror" {
        // Mirror-tile (SEAMS-1 P1): opposite edges identical by construction → guaranteed seamless, no
        // blend smear. Trades a mirror-symmetric pattern for a perfect boundary (good for organic/fabric).
        let axes = Axes::parse(&plan.seamless_axes);
        albedo = seamless::mirror_tile(&albedo, axes);
    } else if plan.seamless_mode != "none" {
        let axes = Axes::parse(&plan.seamless_axes);
        albedo = if is_photo {
            // Offset-and-heal the interior seams, then a light boundary feather for any residual.
            let t = seamless::make_tileable(&albedo, (plan.size / 12).max(8), axes);
            seamless::feather_seam(&t, (plan.size / 24).max(4), axes)
        } else {
            // ADAPTIVE feather (B1, measure-first): a wide feather removes a seam but SOFTENS a band —
            // visible as a faint low-detail stripe on high-frequency materials. Since the flat/tileable
            // prompt already makes the albedo near-tileable, measure the RAW seam and use only as wide a
            // band as it needs — narrow when it's already near-seamless (least smear), wide only when a
            // real seam demands it. (The dormant per-step latent-roll / vendored circular ResNet remain
            // the escalation for a hypothetical material a feather truly can't tile — G0.B / 6.3 G0.1.)
            let raw = raw_seam(&albedo, axes);
            let band = if raw < 1.3 {
                (plan.size / 96).max(3) // already near-tileable → a thin hairline erase, minimal smear
            } else if raw < 2.5 {
                (plan.size / 48).max(4)
            } else {
                (plan.size / 24).max(4) // a genuine seam → the full feather
            };
            seamless::feather_seam(&albedo, band, axes)
        };
    }
    Ok(albedo)
}

/// The larger of the x/y edge-wrap seam ratios of an image's luma (join discontinuity / interior),
/// restricted to the seamed axes — drives the adaptive feather band. ~1 = already tiles.
fn raw_seam(img: &image::RgbImage, axes: Axes) -> f32 {
    let (w, h) = img.dimensions();
    let l = |x: u32, y: u32| {
        let p = img.get_pixel(x, y).0;
        (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0
    };
    let ratio = |join: &[f32], interior: &[f32]| -> f32 {
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
        rms(join) / rms(interior).max(1e-6)
    };
    let mut worst = 0.0f32;
    if matches!(axes, Axes::Both | Axes::X) {
        let (mut j, mut i) = (Vec::new(), Vec::new());
        for y in 0..h {
            j.push(l(0, y) - l(w - 1, y));
            for x in 1..w {
                i.push(l(x, y) - l(x - 1, y));
            }
        }
        worst = worst.max(ratio(&j, &i));
    }
    if matches!(axes, Axes::Both | Axes::Y) {
        let (mut j, mut i) = (Vec::new(), Vec::new());
        for x in 0..w {
            j.push(l(x, 0) - l(x, h - 1));
            for y in 1..h {
                i.push(l(x, y) - l(x, y - 1));
            }
        }
        worst = worst.max(ratio(&j, &i));
    }
    worst
}

/// Render a spec to a material directory. Returns the scorecard of the written material.
pub async fn render_material(spec: &TextureSpec, out: &std::path::Path, opts: &RenderOpts) -> Result<Scorecard> {
    let mut plan = compile::resolve(spec);
    if let Some(e) = &opts.engine {
        crate::texture::export::Engine::parse(e)
            .ok_or_else(|| anyhow::anyhow!("unknown engine `{e}`; known: gltf, unreal, unity-hdrp, godot, materialx, plakat"))?
            .apply_to_plan(&mut plan);
    }
    let tries = opts.attempts.max(1);
    let (mut best_m, mut best_sc, mut fewest) = (None, None, usize::MAX);

    let upscale = opts.upscale.clone().unwrap_or_else(|| plan.upscale.clone());
    for i in 0..tries {
        let mut albedo = albedo_for(&plan, plan.seed + i as u64).await?;
        if plan.delight {
            albedo = crate::texture::derive::flatten_lighting(&albedo); // weight-free homomorphic delight
        }
        // Tiled, tileability-preserving upscale BEFORE derivation, so the detail maps carry full res.
        if let Some(target) = upscale_target(&upscale) {
            println!("  {} tiled upscale → {target}² (tileability-preserving)", style("↳").cyan());
            albedo = upscale_tileable(&albedo, target);
        }
        // Height: `auto` = a Depth-Anything pass on the albedo (real relief); else derived from luma.
        let height = if matches!(plan.height, crate::texture::compile::HeightSource::Auto) {
            match height_auto(&albedo).await {
                Ok(h) => Some(h),
                Err(e) => {
                    println!("  {} depth-height skipped ({e:#}) — deriving height from luminance", style("·").yellow());
                    None
                }
            }
        } else {
            None
        };
        let mut m = Material::derive(
            albedo,
            height,
            plan.normal_strength,
            plan.normal_y == "opengl",
            plan.ao_strength,
            &plan.roughness,
            &plan.metallic,
        );
        // A3: a supplied mask overrides the derived channel verbatim (scored as the final channel).
        let (mw, mh) = m.albedo.dimensions();
        if let Some(p) = &opts.metallic_ref {
            m.metallic = load_mask_ref(p, mw, mh)?;
        }
        if let Some(p) = &opts.roughness_ref {
            m.roughness = load_mask_ref(p, mw, mh)?;
        }
        let sc = scorecard::score(&m);
        let n_issues = sc.notes.len();
        if sc.passes {
            if tries > 1 {
                println!("  {} scorecard PASS on attempt {}/{}", style("✓").green(), i + 1, tries);
            }
            best_m = Some(m);
            best_sc = Some(sc);
            break;
        }
        if n_issues < fewest {
            fewest = n_issues;
            best_m = Some(m);
            best_sc = Some(sc);
        }
        if tries > 1 {
            println!("  {} attempt {}/{} FAIL ({n_issues} issue(s)), retrying…", style("·").yellow(), i + 1, tries);
        }
    }

    let m = best_m.context("no material rendered")?;
    let sc = best_sc.unwrap();
    export::write_material(&m, &plan, &sc, out)?;
    Ok(sc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn raw_seam_low_for_a_tiling_image_high_for_a_split() {
        // A high-frequency field at INTEGER periods (so it wraps) → strong interior detail, tiny join →
        // a cleanly low seam, like a real texture. A left-dark/right-bright split → a hard vertical seam.
        let tiling = RgbImage::from_fn(48, 48, |x, y| {
            let u = (x as f32 / 48.0 * std::f32::consts::TAU * 6.0).sin();
            let v = (y as f32 / 48.0 * std::f32::consts::TAU * 6.0).cos();
            let c = ((u * v) * 0.4 + 0.5).clamp(0.0, 1.0);
            Rgb([(c * 200.0) as u8, (c * 180.0) as u8, (c * 150.0) as u8])
        });
        let split = RgbImage::from_fn(48, 48, |x, _| if x < 24 { Rgb([20, 20, 20]) } else { Rgb([220, 220, 220]) });
        let s_tile = raw_seam(&tiling, Axes::Both);
        let s_split = raw_seam(&split, Axes::Both);
        // A wrapping texture reads near the tiles bound (→ a narrow feather); a hard split reads as a big
        // seam (→ the full feather). The adaptive band keys off exactly this separation.
        assert!(s_tile < 1.5, "a wrapping image should read near-tileable, got {s_tile}");
        assert!(s_split > 2.5, "a hard split should read as a big seam, got {s_split}");
        assert!(s_split > s_tile * 1.8, "split seam should dominate the tiling seam: {s_split} vs {s_tile}");
    }
}
