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
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self { attempts: 1 }
    }
}

fn to_rgb(img: &crate::api::Image) -> Result<image::RgbImage> {
    image::RgbImage::from_raw(img.width(), img.height(), img.pixels().to_vec())
        .context("generated albedo had a malformed buffer")
}

/// Generate (or load) the albedo, then make it seamless.
async fn albedo_for(plan: &RenderPlan, seed: u64) -> Result<image::RgbImage> {
    let mut albedo = if let Some(path) = &plan.from_image {
        // Image-to-material (B6 refines the crop-to-tileable); B4 loads + squares it.
        let img = image::open(path).with_context(|| format!("reading {path}"))?.to_rgb8();
        let s = img.width().min(img.height());
        image::imageops::resize(&image::imageops::crop_imm(&img, 0, 0, s, s).to_image(), plan.size, plan.size, image::imageops::FilterType::Lanczos3)
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
    // Seamless: a boundary feather (B4). The flat/tileable prompt keeps the field low-frequency so the
    // wrap is gentle; a per-step latent roll is the escalation if the scorecard residual demands it.
    if plan.seamless_mode != "none" {
        let band = (plan.size / 24).max(4);
        albedo = seamless::feather_seam(&albedo, band, Axes::parse(&plan.seamless_axes));
    }
    Ok(albedo)
}

/// Render a spec to a material directory. Returns the scorecard of the written material.
pub async fn render_material(spec: &TextureSpec, out: &std::path::Path, opts: &RenderOpts) -> Result<Scorecard> {
    let plan = compile::resolve(spec);
    let tries = opts.attempts.max(1);
    let (mut best_m, mut best_sc, mut fewest) = (None, None, usize::MAX);

    for i in 0..tries {
        let mut albedo = albedo_for(&plan, plan.seed + i as u64).await?;
        if plan.delight {
            albedo = crate::texture::derive::flatten_lighting(&albedo); // weight-free homomorphic delight
        }
        let m = Material::derive(
            albedo,
            None,
            plan.normal_strength,
            plan.normal_y == "opengl",
            plan.ao_strength,
            &plan.roughness,
            &plan.metallic,
        );
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
