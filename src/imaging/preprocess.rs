//! Image preprocessing for CLIP vision encoders and SD VAE input.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::path::Path;

// OpenAI CLIP normalization (RGB).
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_1];

/// Filter used for shrinking arbitrary inputs to the small fixed sizes the
/// encoders consume (CLIP 224², ArcFace 112², VAE 512/768²). Bilinear is
/// fast and visually indistinguishable from bicubic at these sizes —
/// downstream models compress to latent/embedding space and don't see the
/// difference. Output-quality paths (Real-ESRGAN, the final image
/// `--upscale`) keep CatmullRom / Lanczos.
const PREPROCESS_FILTER: FilterType = FilterType::Triangle;

/// Load an image, shorter-side-resize then center-crop to `size`x`size`,
/// CLIP-normalize. Returns (1, 3, size, size) in `dtype` on `device`.
pub fn clip_image_tensor(
    path: &Path,
    size: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::open(path)?.to_rgb8();
    let img = DynamicImage::ImageRgb8(img);
    let (w, h) = img.dimensions();

    // Shorter side → `size`, keep aspect.
    let (rw, rh) = if w < h {
        let s = size;
        (s, ((h as f32) * (s as f32) / (w as f32)).round() as u32)
    } else {
        let s = size;
        (((w as f32) * (s as f32) / (h as f32)).round() as u32, s)
    };
    let resized = img.resize_exact(rw, rh, PREPROCESS_FILTER);
    let cx = rw.saturating_sub(size) / 2;
    let cy = rh.saturating_sub(size) / 2;
    let cropped = resized.crop_imm(cx, cy, size, size).to_rgb8();

    // Channel-first, normalized. We iterate over packed RGB bytes once and
    // scatter into channel-separated slices — single pass over the source
    // buffer, no per-pixel `get_pixel` indirection.
    let n = size as usize;
    let n_pixels = n * n;
    let mut data: Vec<f32> = vec![0.0f32; 3 * n_pixels];
    let raw = cropped.as_raw();
    let (r_dst, rest) = data.split_at_mut(n_pixels);
    let (g_dst, b_dst) = rest.split_at_mut(n_pixels);
    let inv_r = 1.0 / (255.0 * CLIP_STD[0]);
    let inv_g = 1.0 / (255.0 * CLIP_STD[1]);
    let inv_b = 1.0 / (255.0 * CLIP_STD[2]);
    let off_r = CLIP_MEAN[0] / CLIP_STD[0];
    let off_g = CLIP_MEAN[1] / CLIP_STD[1];
    let off_b = CLIP_MEAN[2] / CLIP_STD[2];
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        r_dst[i] = chunk[0] as f32 * inv_r - off_r;
        g_dst[i] = chunk[1] as f32 * inv_g - off_g;
        b_dst[i] = chunk[2] as f32 * inv_b - off_b;
    }

    let t = Tensor::from_vec(data, (1, 3, n, n), device)?.to_dtype(dtype)?;
    Ok(t)
}

/// Load an image, resize exactly to `w`x`h`, normalize to [-1, 1].
/// Returns (1, 3, h, w) for the SD VAE encoder.
pub fn sd_image_tensor(
    path: &Path,
    w: u32,
    h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(&img, w, h, PREPROCESS_FILTER);

    // Same channel-first scatter as CLIP, but normalized to [-1, 1] for VAE.
    let total = (w as usize) * (h as usize);
    let mut data: Vec<f32> = vec![0.0f32; 3 * total];
    let raw = resized.as_raw();
    let (r_dst, rest) = data.split_at_mut(total);
    let (g_dst, b_dst) = rest.split_at_mut(total);
    let scale = 1.0 / 127.5;
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        r_dst[i] = chunk[0] as f32 * scale - 1.0;
        g_dst[i] = chunk[1] as f32 * scale - 1.0;
        b_dst[i] = chunk[2] as f32 * scale - 1.0;
    }

    let t = Tensor::from_vec(data, (1, 3, h as usize, w as usize), device)?.to_dtype(dtype)?;
    Ok(t)
}
