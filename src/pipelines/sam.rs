//! Segment Anything (MobileSAM) — prompt-driven selection → binary mask.
//!
//! Wraps candle-transformers' `segment_anything` in its TinyViT/MobileSAM
//! variant (`Sam::new_tiny`). Point prompts (foreground / background clicks)
//! produce a binary mask that feeds plakat's existing `--mask` consumers
//! (inpaint / img2img), so "select → remove / replace" composes from pieces
//! plakat already owns rather than a bespoke pipeline.
//!
//! Model facts that shape this wrapper (from the candle port):
//! - `Sam::preprocess` expects the image in **0-255** scale (it subtracts the
//!   0-255 ImageNet mean/std), and it **only pads** to `IMAGE_SIZE` (1024) —
//!   it bails if a side exceeds 1024. So we resize longest-side to 1024 and
//!   build the tensor without dividing by 255.
//! - Point coords are **normalized [0,1]** (the decoder scales by the input
//!   dims), so a resize doesn't move the prompts.
//! - `forward(..)` returns mask **logits** cropped to the (resized) input;
//!   threshold at 0 (the model's mask threshold) for a binary selection.
//!
//! Weights: `mobile_sam-tiny-vitt.safetensors` (~40 MB, ungated) in candle's
//! key layout. Resolution mirrors the U2Net matte loader
//! (`PLAKAT_SAM_WEIGHTS` → plakat cache → HF mirror).

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam::{IMAGE_SIZE, Sam};
use image::{GrayImage, Luma, RgbImage};
use std::path::Path;

/// Candle-layout MobileSAM safetensors (TinyViT encoder + SAM decoder).
const WEIGHTS_FILE: &str = "mobile_sam-tiny-vitt.safetensors";
/// Primary mirror (redistributed, ungated) — same convention as the U2Net matte.
const SAM_REPO: &str = "vulogov98/mobile-sam";
/// Fallback: candle author's ungated repo, so this works even pre-mirror.
const SAM_FALLBACK_REPO: &str = "lmz/candle-sam";

/// A point prompt: image coords + whether it marks foreground (include) or
/// background (exclude). Coords are normalized [0,1] **or** pixel values —
/// `segment` auto-detects pixel coords (any value > 1) and normalizes them.
#[derive(Debug, Clone, Copy)]
pub struct PointPrompt {
    pub x: f64,
    pub y: f64,
    pub foreground: bool,
}

/// Resolve the MobileSAM weights: `PLAKAT_SAM_WEIGHTS` (a safetensors path)
/// wins, else a locally-cached file, else download from the HF mirror (with
/// candle's repo as a fallback).
async fn sam_weights_path() -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var("PLAKAT_SAM_WEIGHTS") {
        return Ok(p.into());
    }
    let base = std::env::var("HOME").unwrap_or_default();
    let local = std::path::PathBuf::from(base)
        .join(".cache/plakat/mobile-sam")
        .join(WEIGHTS_FILE);
    if local.exists() {
        return Ok(local);
    }
    match crate::hf::download::get_file(SAM_REPO, WEIGHTS_FILE).await {
        Ok(p) => Ok(p),
        Err(primary) => crate::hf::download::get_file(SAM_FALLBACK_REPO, WEIGHTS_FILE)
            .await
            .with_context(|| {
                format!(
                    "downloading MobileSAM weights — {SAM_REPO}/{WEIGHTS_FILE} failed \
                     ({primary}), and the {SAM_FALLBACK_REPO} fallback also failed"
                )
            }),
    }
}

/// Resize so the longest side is `IMAGE_SIZE` (SAM's fixed input), keeping
/// aspect. SAM pads the rest, so images already ≤ 1024 are passed as-is.
fn resize_longest(img: &RgbImage) -> (RgbImage, u32, u32) {
    let (w, h) = (img.width(), img.height());
    if w.max(h) <= IMAGE_SIZE as u32 {
        return (img.clone(), w, h);
    }
    let scale = IMAGE_SIZE as f32 / w.max(h) as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let r = image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle);
    (r, nw, nh)
}

/// Build a `(3, h, w)` f32 tensor in **0-255** scale (CHW planar) — SAM's
/// `preprocess` does the ImageNet normalization itself.
fn to_tensor(img: &RgbImage, device: &Device) -> Result<Tensor> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let plane = h * w;
    let mut data = vec![0f32; 3 * plane];
    for (i, p) in img.pixels().enumerate() {
        data[i] = p.0[0] as f32;
        data[plane + i] = p.0[1] as f32;
        data[2 * plane + i] = p.0[2] as f32;
    }
    Ok(Tensor::from_vec(data, (3, h, w), device)?)
}

/// Segment an image with point prompts → a binary mask PNG (255 = selected,
/// 0 = excluded), sized to the original image. The mask is the SAM
/// multimask-best (highest predicted IoU), which gives the cleanest single
/// object for a single click while still honoring extra refine points.
pub async fn segment(
    in_path: &Path,
    out_path: &Path,
    points: &[PointPrompt],
    invert: bool,
    device: &Device,
) -> Result<()> {
    if points.is_empty() {
        return Err(anyhow!(
            "no prompt: pass at least one --point X,Y (append :bg to exclude a region)"
        ));
    }

    let weights = sam_weights_path().await?;
    let img = image::open(in_path)
        .with_context(|| format!("opening {}", in_path.display()))?
        .to_rgb8();
    let (w0, h0) = (img.width(), img.height());
    let (resized, rw, rh) = resize_longest(&img);

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device)
            .context("loading MobileSAM safetensors")?
    };
    let sam = Sam::new_tiny(vb).context("building MobileSAM (Sam::new_tiny)")?;

    // Normalize prompts to [0,1]. If any coord exceeds 1 we treat the whole
    // set as pixel coords (predictable; no mixed interpretation).
    let pixel_mode = points.iter().any(|p| p.x > 1.0 || p.y > 1.0);
    let pts: Vec<(f64, f64, bool)> = points
        .iter()
        .map(|p| {
            if pixel_mode {
                (p.x / w0 as f64, p.y / h0 as f64, p.foreground)
            } else {
                (p.x, p.y, p.foreground)
            }
        })
        .collect();

    let x = to_tensor(&resized, device)?;
    // multimask_output=true → 3 candidate masks + IoU; pick the best.
    let (masks, iou) = sam.forward(&x, &pts, true).context("SAM forward")?;
    let (n_masks, _mh, _mw) = masks.dims3()?;
    let best = {
        let ious: Vec<f32> = iou.flatten_all()?.to_vec1()?;
        ious.iter()
            .take(n_masks)
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let sel = masks.i(best)?; // (rh, rw) logits

    // Threshold at 0 (the model's mask threshold) → binary, at the resized
    // resolution, then nearest-resize back to the true original size.
    let vals: Vec<f32> = sel.flatten_all()?.to_vec1()?;
    let mut g = GrayImage::new(rw, rh);
    for (i, &v) in vals.iter().enumerate() {
        let on = (v > 0.0) ^ invert;
        g.put_pixel((i as u32) % rw, (i as u32) / rw, Luma([if on { 255 } else { 0 }]));
    }
    let mask = if (rw, rh) != (w0, h0) {
        image::imageops::resize(&g, w0, h0, image::imageops::FilterType::Nearest)
    } else {
        g
    };

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    mask.save(out_path)
        .with_context(|| format!("writing mask {}", out_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_longest_is_noop_when_within_bounds() {
        let img = RgbImage::new(800, 600);
        let (r, w, h) = resize_longest(&img);
        assert_eq!((w, h), (800, 600));
        assert_eq!((r.width(), r.height()), (800, 600));
    }

    #[test]
    fn resize_longest_caps_at_image_size_keeping_aspect() {
        let img = RgbImage::new(2048, 1024);
        let (r, w, h) = resize_longest(&img);
        assert_eq!(w, IMAGE_SIZE as u32, "longest side capped to 1024");
        assert_eq!(h, 512, "aspect ratio preserved");
        assert_eq!((r.width(), r.height()), (1024, 512));
    }
}
