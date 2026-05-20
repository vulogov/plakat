//! Monocular depth estimation via Depth-Anything-V2 (vit_small variant).
//!
//! Used by the v3 smart-zones feature: a depth map of the generated
//! image lets the compositor pick zones that follow the actual painted
//! horizon / sky / foreground instead of the rigid 4×3 grid.
//!
//! Only the small variant (~99 MB) is wired in. The base / large
//! checkpoints are higher quality but materially heavier — for
//! "approximately where is the sky" purposes the small model is more
//! than enough, and the cost has to stay subordinate to the
//! generation pass that produced the image.
//!
//! Inference flow:
//!
//! 1. Load image, resize to 518 × 518, normalize with ImageNet stats.
//! 2. Forward through DinoV2 (vit_small) → DPT head → raw depth
//!    `(1, 1, H, W)`.
//! 3. Min-max normalize to `[0, 1]` (the model emits relative depth,
//!    not metric).
//! 4. Bilinear-resize back to the original image resolution.
//!
//! **Convention.** Larger values = closer to the camera. Sky / far
//! objects → values near 0; foreground → values near 1. This is
//! Depth-Anything's relative-inverse-depth output (already monotonic
//! in that direction after the head's ReLU).

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;
use std::path::Path;
use std::sync::Arc;

/// Model input resolution (square). Hardcoded to 518 because the
/// candle DinoV2 module's positional embedding is built at
/// `IMG_SIZE = 518` and patch size 14.
const INPUT_SIZE: u32 = 518;

/// Lazy-loaded Depth-Anything-V2 pipeline. Reuse one instance across
/// many images in the same run — model load is the dominant cost.
pub struct DepthPipeline {
    model: DepthAnythingV2,
    device: Device,
    dtype: DType,
}

impl DepthPipeline {
    /// Download (or hit the cache for) the small variant's safetensors
    /// and instantiate the model. `device` is whatever the caller's
    /// generation pass is using.
    ///
    /// Tries a couple of community safetensors mirrors in order. The
    /// official `depth-anything/Depth-Anything-V2-Small` repo ships
    /// `.pth` files only, which candle can't mmap directly.
    pub async fn load(device: Device) -> Result<Self> {
        // The depth model itself isn't heavy in memory; F32 keeps things
        // simple across CPU and accelerator devices.
        let dtype = DType::F32;

        // Two-stage weight fetch:
        // 1. DinoV2 vit_small backbone.
        // 2. The DepthAnything DPT head (often bundled together in the
        //    community safetensors releases).
        //
        // We try a few known community-converted repos in order. If
        // none are reachable, propagate the error — the caller will
        // warn and fall back to the rigid grid.
        let combined_candidates: &[(&str, &str)] = &[
            // Combined backbone + head conversions.
            (
                "jeromekoo/depth-anything-v2-safetensors-small",
                "depth_anything_v2_vits.safetensors",
            ),
            (
                "Bingsu/depth-anything-v2-safetensors",
                "depth_anything_v2_vits.safetensors",
            ),
            (
                "MackinationsAi/Depth-Anything-V2_Safetensors",
                "depth_anything_v2_vits.safetensors",
            ),
        ];
        let weights_path = crate::hf::download::get_first_of(combined_candidates)
            .await
            .context(
                "downloading Depth-Anything-V2 small safetensors. Tried community \
                 mirrors of the official `depth-anything/Depth-Anything-V2-Small` \
                 (which only ships .pth). Smart zones cannot run without these.",
            )?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[&weights_path],
                dtype,
                &device,
            )?
        };

        let dino_cfg = DepthAnythingV2Config::vit_small();
        let backbone = dinov2::vit_small(vb.pp("pretrained"))?;
        let model = DepthAnythingV2::new(Arc::new(backbone), dino_cfg, vb)?;

        Ok(Self { model, device, dtype })
    }

    /// Estimate per-pixel relative depth for the image at `path`.
    ///
    /// Returns a `(image_h, image_w)` f32 buffer in `[0, 1]`. The
    /// buffer is row-major; index `[y * w + x]`. Larger = closer.
    ///
    /// `out_w`/`out_h` specify the resolution to upsample to (almost
    /// always the caller's generation resolution — we want the map
    /// at the same scale as the image we'll be reading zones from).
    pub fn depth_map(&self, path: &Path, out_w: u32, out_h: u32) -> Result<Vec<f32>> {
        let pixels = preprocess_image(path, INPUT_SIZE, &self.device, self.dtype)
            .with_context(|| format!("preprocessing {} for depth", path.display()))?;
        let raw = self
            .model
            .forward(&pixels)
            .context("Depth-Anything-V2 forward")?;
        let normalized = normalize_min_max(&raw)?;
        let resized = bilinear_resize(&normalized, out_w as usize, out_h as usize)?;
        Ok(resized)
    }
}

/// Load + resize + normalize one image to the model's expected input
/// shape `(1, 3, INPUT_SIZE, INPUT_SIZE)`. ImageNet mean/std
/// normalisation (consistent with DinoV2's training).
fn preprocess_image(
    path: &Path,
    size: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &img,
        size,
        size,
        image::imageops::FilterType::Triangle,
    );

    // ImageNet stats.
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let n = (size as usize) * (size as usize);
    let mut buf: Vec<f32> = vec![0.0; 3 * n];
    let raw = resized.as_raw();
    let (r_dst, rest) = buf.split_at_mut(n);
    let (g_dst, b_dst) = rest.split_at_mut(n);
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        r_dst[i] = (chunk[0] as f32 / 255.0 - mean[0]) / std[0];
        g_dst[i] = (chunk[1] as f32 / 255.0 - mean[1]) / std[1];
        b_dst[i] = (chunk[2] as f32 / 255.0 - mean[2]) / std[2];
    }
    let t = Tensor::from_vec(buf, (1, 3, size as usize, size as usize), device)?
        .to_dtype(dtype)?;
    Ok(t)
}

/// Min-max normalise `(1, 1, H, W)` (or `(1, H, W)`) into `[0, 1]`
/// as a flat Vec<f32>. Output stays at the input resolution.
fn normalize_min_max(t: &Tensor) -> Result<(Vec<f32>, usize, usize)> {
    // Squeeze leading singleton dims to land on (H, W).
    let mut x = t.clone();
    while x.dims().len() > 2 {
        x = x.i(0)?;
    }
    let (h, w) = x.dims2()?;
    let v: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let (mn, mx) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    let span = (mx - mn).max(1e-6);
    let out: Vec<f32> = v.iter().map(|&x| (x - mn) / span).collect();
    Ok((out, h, w))
}

/// Bilinear resample a flat `(in_h * in_w)` f32 buffer (we know the
/// model output dims, so the caller passes them via `normalize_min_max`)
/// to `(out_h * out_w)`.
fn bilinear_resize(
    buf_with_dims: &(Vec<f32>, usize, usize),
    out_w: usize,
    out_h: usize,
) -> Result<Vec<f32>> {
    let (src, in_h, in_w) = buf_with_dims;
    let in_h = *in_h;
    let in_w = *in_w;
    if out_w == in_w && out_h == in_h {
        return Ok(src.clone());
    }
    if in_w == 0 || in_h == 0 {
        return Err(anyhow!("bilinear_resize: empty input"));
    }
    let mut dst = vec![0f32; out_w * out_h];
    let sx = (in_w as f32 - 1.0) / (out_w.max(1) as f32 - 1.0).max(1.0);
    let sy = (in_h as f32 - 1.0) / (out_h.max(1) as f32 - 1.0).max(1.0);
    for y in 0..out_h {
        let fy = y as f32 * sy;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;
        for x in 0..out_w {
            let fx = x as f32 * sx;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = fx - x0 as f32;
            let a = src[y0 * in_w + x0];
            let b = src[y0 * in_w + x1];
            let c = src[y1 * in_w + x0];
            let d = src[y1 * in_w + x1];
            let top = a + wx * (b - a);
            let bot = c + wx * (d - c);
            dst[y * out_w + x] = top + wy * (bot - top);
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_resize_identity_passes_through() {
        let src = (vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let out = bilinear_resize(&src, 2, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bilinear_resize_2x_upscale_interpolates() {
        // 2×2 source: [0, 0; 0, 1] → 3×3 dst should put 1 at bottom-right
        // and roughly 0.5 at the centre.
        let src = (vec![0.0, 0.0, 0.0, 1.0], 2, 2);
        let out = bilinear_resize(&src, 3, 3).unwrap();
        // Top-left: 0.
        assert!(out[0].abs() < 1e-5);
        // Bottom-right: 1.
        assert!((out[8] - 1.0).abs() < 1e-5);
        // Centre: 0.25 (bilinear average).
        assert!((out[4] - 0.25).abs() < 1e-5);
    }
}
