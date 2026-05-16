//! Image preprocessing for CLIP vision encoders and SD VAE input.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::path::Path;

// OpenAI CLIP normalization (RGB).
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_1];

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
    let resized = img.resize_exact(rw, rh, FilterType::CatmullRom);
    let cx = rw.saturating_sub(size) / 2;
    let cy = rh.saturating_sub(size) / 2;
    let cropped = resized.crop_imm(cx, cy, size, size).to_rgb8();

    // Channel-first, normalized.
    let n = size as usize;
    let mut data: Vec<f32> = Vec::with_capacity(3 * n * n);
    for c in 0..3usize {
        for y in 0..n {
            for x in 0..n {
                let px = cropped.get_pixel(x as u32, y as u32).0[c];
                data.push((px as f32 / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c]);
            }
        }
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
    let resized = image::imageops::resize(&img, w, h, FilterType::CatmullRom);
    let total = (w as usize) * (h as usize);
    let mut data: Vec<f32> = Vec::with_capacity(3 * total);
    for c in 0..3usize {
        for y in 0..h {
            for x in 0..w {
                let px = resized.get_pixel(x, y).0[c];
                data.push(px as f32 / 127.5 - 1.0);
            }
        }
    }
    let t = Tensor::from_vec(data, (1, 3, h as usize, w as usize), device)?.to_dtype(dtype)?;
    Ok(t)
}
