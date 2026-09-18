//! LAYERED-1 S4 — **repair**. For each failing (or explicitly named) anchored/hinted layer, re-assert the
//! subject with a masked img2img pass: a white box (feathered by the sampler) marks the region to regenerate
//! toward the layer's prompt while the rest of the finish is preserved. Repairs run SEQUENTIALLY — each
//! layer's result feeds the next — so multiple subjects don't fight over the same denoise.
//!
//! The mask builder is pure (offline-tested); the img2img pass renders, so a repair run is a diffusion job.

use anyhow::{Context, Result};
use image::{GrayImage, Luma};

use crate::layered::plan::{self, Layer, LayerPlan};

/// A HARD white-inside-box mask (`255` = regenerate here, `0` = keep). The sampler feathers the edge, so this
/// stays a crisp rectangle.
pub fn box_mask(bbox: [f32; 4], w: u32, h: u32) -> GrayImage {
    let x0 = (bbox[0] * w as f32).round().clamp(0.0, w as f32) as u32;
    let y0 = (bbox[1] * h as f32).round().clamp(0.0, h as f32) as u32;
    let x1 = (bbox[2] * w as f32).round().clamp(0.0, w as f32) as u32;
    let y1 = (bbox[3] * h as f32).round().clamp(0.0, h as f32) as u32;
    let mut m = GrayImage::new(w, h);
    for y in y0..y1.min(h) {
        for x in x0..x1.min(w) {
            m.put_pixel(x, y, Luma([255]));
        }
    }
    m
}

/// The prompt used to repair a layer: its own prompt, joined with the anchored global look (palette + light +
/// medium — at repair time the medium DOES apply, unlike the drafts).
pub fn repair_prompt(l: &Layer, plan: &LayerPlan) -> String {
    let g = &plan.global;
    [l.prompt.as_deref(), g.medium.as_deref(), g.palette.as_deref(), g.light.as_deref()]
        .into_iter()
        .flatten()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Options for [`repair`].
pub struct RepairOpts {
    pub model: String,
    pub out: std::path::PathBuf,
    /// Inpaint strength `[0,1]` — how far the box may drift from the current image. Default ~0.6.
    pub strength: f32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    pub mask_feather: u32,
    /// The `--device` spec (so the repair runs on the caller's device).
    pub device: String,
}

/// Repair the named layers in `image`, in order, each with a masked img2img pass toward its prompt. Writes
/// the final image to `o.out`. (Renders — one diffusion pass per target.)
pub async fn repair(image: &std::path::Path, plan: &LayerPlan, targets: &[String], o: &RepairOpts) -> Result<()> {
    let (w, h) = image::image_dimensions(image).with_context(|| format!("reading {}", image.display()))?;
    let tmp = tempfile::Builder::new().prefix("plakat-layered-repair-").tempdir().context("repair scratch dir")?;
    let mut current = image.to_path_buf();

    let mut n = 0usize;
    for id in targets {
        let Some(l) = plan.layers.iter().find(|l| &l.id == id) else { continue };
        let mask_path = tmp.path().join(format!("mask_{id}.png"));
        box_mask(plan::layer_box(l), w, h).save(&mask_path).with_context(|| format!("saving mask {}", mask_path.display()))?;

        let images = crate::api::Img2img::new(&o.model, current.clone())
            .prompt(repair_prompt(l, plan))
            .mask(&mask_path)
            .mask_feather(o.mask_feather)
            .strength(o.strength)
            .steps(o.steps)
            .guidance(o.guidance)
            .seed(o.seed)
            .device(&o.device)
            .run()
            .await
            .with_context(|| format!("repairing layer {id:?}"))?;
        let img = images.into_iter().next().ok_or_else(|| anyhow::anyhow!("repair produced no image for {id:?}"))?;
        let step_path = tmp.path().join(format!("repaired_{n}.png"));
        img.save(&step_path)?;
        current = step_path;
        n += 1;
    }

    // Copy the final image out (even if no targets ran, so the command is a well-defined pass-through).
    if let Some(parent) = o.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::copy(&current, &o.out).with_context(|| format!("writing {}", o.out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered::plan::Global;

    #[test]
    fn box_mask_is_white_inside_black_outside() {
        let m = box_mask([0.25, 0.25, 0.75, 0.75], 128, 128);
        assert_eq!(m.get_pixel(64, 64).0[0], 255, "centre is inpaint (white)");
        assert_eq!(m.get_pixel(4, 4).0[0], 0, "corner is preserved (black)");
        // Rough area check: ~1/4 of the pixels white.
        let white = m.pixels().filter(|p| p.0[0] > 127).count();
        let frac = white as f32 / (128.0 * 128.0);
        assert!((frac - 0.25).abs() < 0.05, "≈25% white (got {frac})");
    }

    #[test]
    fn repair_prompt_adds_medium_and_look() {
        let plan = LayerPlan {
            global: Global { palette: Some("muted autumn".into()), light: Some("overcast".into()), medium: Some("oil painting".into()) },
            ..Default::default()
        };
        let l = Layer { id: "a".into(), prompt: Some("a boy in a raincoat".into()), ..Default::default() };
        let p = repair_prompt(&l, &plan);
        assert!(p.starts_with("a boy in a raincoat"));
        assert!(p.contains("oil painting"), "medium DOES apply at repair (unlike drafts)");
        assert!(p.contains("muted autumn") && p.contains("overcast"));
    }
}
