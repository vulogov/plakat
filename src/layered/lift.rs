//! LAYERED-1 S5 — **lift**. A lifted-class subject is too small to anchor (shorter box side `< u`), so its
//! structure comes from a dedicated pass: crop the box, regenerate the subject at a workable resolution
//! (img2img toward the layer's prompt), downscale, and composite it back into the box with a feathered edge.
//!
//! The crop + feathered composite are pure (offline-tested); the img2img regenerate renders, so a lift run is
//! a diffusion job.

use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};

use crate::layered::lint::{layer_class, Class};
use crate::layered::plan::{self, Layer, LayerPlan};
use crate::pipelines::noise_space::LatentGeometry;

/// A layer box in output pixels, clamped, at least 1×1.
fn box_px(l: &Layer, w: u32, h: u32) -> (u32, u32, u32, u32) {
    let b = plan::layer_box(l);
    let x0 = (b[0] * w as f32).round().clamp(0.0, w as f32 - 1.0) as u32;
    let y0 = (b[1] * h as f32).round().clamp(0.0, h as f32 - 1.0) as u32;
    let x1 = ((b[2] * w as f32).round().clamp(0.0, w as f32) as u32).max(x0 + 1);
    let y1 = ((b[3] * h as f32).round().clamp(0.0, h as f32) as u32).max(y0 + 1);
    (x0, y0, x1 - x0, y1 - y0)
}

/// Crop `img` to a layer's box → the crop and its `(x0, y0)` origin.
pub fn crop_box(img: &RgbImage, l: &Layer, w: u32, h: u32) -> (RgbImage, u32, u32) {
    let (x0, y0, cw, ch) = box_px(l, w, h);
    (image::imageops::crop_imm(img, x0, y0, cw, ch).to_image(), x0, y0)
}

/// Composite `patch` back into `base` at `(x0, y0)`, blending over a `feather`-pixel border so the seam
/// disappears. `patch` is expected to already be the box's size.
pub fn composite_back(base: &RgbImage, patch: &RgbImage, x0: u32, y0: u32, feather: u32) -> RgbImage {
    let mut out = base.clone();
    let (pw, ph) = (patch.width(), patch.height());
    let f = feather.max(1) as f32;
    for py in 0..ph {
        for px in 0..pw {
            let (bx, by) = (x0 + px, y0 + py);
            if bx >= out.width() || by >= out.height() {
                continue;
            }
            // Alpha ramps 0→1 over `feather` px in from each patch edge.
            let d_edge = px.min(pw.saturating_sub(1) - px).min(py).min(ph.saturating_sub(1) - py) as f32;
            let a = (d_edge / f).clamp(0.0, 1.0);
            let bg = out.get_pixel(bx, by).0;
            let fg = patch.get_pixel(px, py).0;
            let mix = |i: usize| (a * fg[i] as f32 + (1.0 - a) * bg[i] as f32).round() as u8;
            out.put_pixel(bx, by, image::Rgb([mix(0), mix(1), mix(2)]));
        }
    }
    out
}

/// A workable regenerate size for a small crop: scale so the short side ≈ `target`, keep aspect, ×8.
pub fn work_size(cw: u32, ch: u32, target: u32) -> (u32, u32) {
    let scale = target as f32 / cw.min(ch).max(1) as f32;
    let round8 = |v: f32| (((v / 8.0).round() as i64 * 8).clamp(256, 1024)) as u32;
    (round8(cw as f32 * scale), round8(ch as f32 * scale))
}

/// Options for [`lift`].
pub struct LiftOpts {
    pub model: String,
    pub out: std::path::PathBuf,
    /// img2img strength for the regenerate (default ~0.7 — redraw the subject, keep the framing).
    pub strength: f32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    /// The short-side working resolution the crop is upscaled to before regenerating (default 384).
    pub work: u32,
    /// Feather (px) for the composite-back seam.
    pub feather: u32,
    /// The `--device` spec.
    pub device: String,
}

/// The prompt used to lift a layer: its own prompt + the global look (medium applies here, as at repair).
pub fn lift_prompt(l: &Layer, plan: &LayerPlan) -> String {
    super::repair::repair_prompt(l, plan)
}

/// Lift every lifted-class layer in `image`: crop → regenerate at `work` res → composite back. Writes the
/// final image to `o.out`. (Renders — one img2img pass per lifted layer.)
pub async fn lift(image: &std::path::Path, plan: &LayerPlan, geom: &LatentGeometry, o: &LiftOpts) -> Result<()> {
    let (w, h) = image::image_dimensions(image).with_context(|| format!("reading {}", image.display()))?;
    let mut current = image::open(image).with_context(|| format!("loading {}", image.display()))?.to_rgb8();
    let tmp = tempfile::Builder::new().prefix("plakat-layered-lift-").tempdir().context("lift scratch dir")?;

    for l in &plan.layers {
        if layer_class(l, geom, w, h) != Class::Lifted || l.prompt.as_deref().map(str::trim).unwrap_or("").is_empty() {
            continue;
        }
        let (crop, x0, y0) = crop_box(&current, l, w, h);
        let (cw, ch) = (crop.width(), crop.height());
        let (ww, wh) = work_size(cw, ch, o.work);
        let up = image::imageops::resize(&crop, ww, wh, FilterType::Lanczos3);
        let up_path = tmp.path().join(format!("crop_{}.png", l.id));
        up.save(&up_path).with_context(|| format!("saving crop {}", up_path.display()))?;

        let images = crate::api::Img2img::new(&o.model, up_path.clone())
            .prompt(lift_prompt(l, plan))
            .strength(o.strength)
            .steps(o.steps)
            .guidance(o.guidance)
            .seed(o.seed)
            .device(&o.device)
            .run()
            .await
            .with_context(|| format!("lifting layer {:?}", l.id))?;
        let regen = images.into_iter().next().ok_or_else(|| anyhow::anyhow!("lift produced no image for {:?}", l.id))?;
        let regen = RgbImage::from_raw(regen.width(), regen.height(), regen.pixels().to_vec()).ok_or_else(|| anyhow::anyhow!("lift buffer size mismatch"))?;
        let patch = image::imageops::resize(&regen, cw, ch, FilterType::Lanczos3);
        current = composite_back(&current, &patch, x0, y0, o.feather);
    }

    if let Some(parent) = o.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    current.save(&o.out).with_context(|| format!("writing {}", o.out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn crop_box_extracts_the_region() {
        let mut img = RgbImage::from_pixel(100, 100, Rgb([0, 0, 0]));
        for y in 25..75 {
            for x in 25..75 {
                img.put_pixel(x, y, Rgb([255, 0, 0]));
            }
        }
        let l = Layer { id: "s".into(), bbox: Some([0.25, 0.25, 0.75, 0.75]), ..Default::default() };
        let (crop, x0, y0) = crop_box(&img, &l, 100, 100);
        assert_eq!((x0, y0), (25, 25));
        assert_eq!(crop.dimensions(), (50, 50));
        assert_eq!(crop.get_pixel(25, 25).0, [255, 0, 0], "crop is the red region");
    }

    #[test]
    fn composite_back_blends_center_and_preserves_far_background() {
        let base = RgbImage::from_pixel(64, 64, Rgb([0, 0, 0]));
        let patch = RgbImage::from_pixel(32, 32, Rgb([255, 255, 255]));
        let out = composite_back(&base, &patch, 16, 16, 4);
        // Patch centre → fully the patch; far background untouched.
        assert_eq!(out.get_pixel(32, 32).0, [255, 255, 255], "patch centre wins");
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0], "far background preserved");
        // The patch edge is feathered (a blend, not a hard 255 or 0).
        let edge = out.get_pixel(16, 32).0[0];
        assert!(edge < 255, "edge is blended, not fully the patch (got {edge})");
    }

    #[test]
    fn work_size_keeps_aspect_and_multiple_of_8() {
        let (ww, wh) = work_size(20, 40, 384);
        assert_eq!(ww % 8, 0);
        assert_eq!(wh % 8, 0);
        assert!(wh > ww, "taller crop stays taller");
        assert!(ww >= 256, "clamped up to the minimum working size");
    }
}
