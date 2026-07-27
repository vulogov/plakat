//! Mask loading + feathering for inpaint passes.
//!
//! A mask is a per-pixel value in `[0.0, 1.0]`:
//!
//! * `1.0` = re-paint this pixel (inside the inpaint region).
//! * `0.0` = preserve this pixel (outside the mask).
//!
//! Mask images on disk can come in any of three forms; this module
//! normalises them:
//!
//! | Source format | How we interpret it |
//! |---|---|
//! | Grayscale (`L8`)        | Brightness directly: `value / 255`. |
//! | RGB (no alpha)          | Luminance: `Y = 0.299R + 0.587G + 0.114B`, then `/255`. |
//! | RGBA                    | Alpha channel: `alpha / 255`. |
//!
//! After loading, the mask is resized to the working dimensions
//! (typically the image we're inpainting), optionally inverted, and
//! optionally feathered via separable box blur.
//!
//! For the latent-space mask that the denoise pipeline consumes, see
//! [`to_latent_tensor`] — it average-pools by the VAE downsample
//! factor (8 for SD 1.5 / SDXL) and returns a `(1, 1, h/8, w/8)`
//! tensor at the pipeline's dtype.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::imageops::FilterType;
use std::path::Path;

/// VAE downsample factor used by every SD-family pipeline plakat
/// supports. The latent grid is `image_w/8 × image_h/8`.
pub const VAE_FACTOR: u32 = 8;

/// A normalised inpaint mask at full image resolution.
///
/// Values are in `[0.0, 1.0]`. Indexing is row-major:
/// `pixels[y * width + height]`.
#[derive(Debug, Clone)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
}

impl Mask {
    /// Load a mask image from disk, resize to `(w, h)`, and normalise
    /// to `[0, 1]`. RGB sources are flattened by luminance; RGBA by
    /// the alpha channel; grayscale is taken as-is.
    pub fn load(path: &Path, w: u32, h: u32) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("opening mask {}", path.display()))?;
        let resized = img.resize_exact(w, h, FilterType::Triangle);
        let pixels = match resized.color() {
            image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => {
                resized
                    .to_rgba8()
                    .pixels()
                    .map(|p| p[3] as f32 / 255.0)
                    .collect()
            }
            image::ColorType::L8 | image::ColorType::L16 => resized
                .to_luma8()
                .pixels()
                .map(|p| p[0] as f32 / 255.0)
                .collect(),
            // Default: convert anything else (Rgb8, Rgb16, RgbF32, etc.)
            // to luma and read luminance. `image::DynamicImage::to_luma8`
            // applies Rec. 601 weights, which is what we want.
            _ => resized
                .to_luma8()
                .pixels()
                .map(|p| p[0] as f32 / 255.0)
                .collect(),
        };
        Ok(Self {
            width: w,
            height: h,
            pixels,
        })
    }

    /// Flip mask polarity: every value becomes `1.0 - value`. Useful
    /// when the source convention is reversed (black = inpaint).
    pub fn invert(&mut self) {
        for v in self.pixels.iter_mut() {
            *v = 1.0 - *v;
        }
    }

    /// Apply a separable box blur of `radius` pixels for soft edges.
    /// Zero radius is a no-op. Edge-clamped (see `box_blur_inplace`): a
    /// region saturated at the image border stays saturated, so only true
    /// interior boundaries soften.
    pub fn feather(&mut self, radius: u32) {
        if radius == 0 {
            return;
        }
        box_blur_inplace(
            &mut self.pixels,
            self.width as usize,
            self.height as usize,
            radius as usize,
        );
    }

    /// Build a solid `1.0` mask of size `(w, h)` — what `plakat img2img`
    /// uses when no `--mask` is provided. The denoise touches every
    /// pixel.
    pub fn solid_one(w: u32, h: u32) -> Self {
        Self {
            width: w,
            height: h,
            pixels: vec![1.0; (w as usize) * (h as usize)],
        }
    }

    /// Encode as a `(1, 1, h/8, w/8)` tensor at `dtype` on `device`.
    /// Average-pools by the VAE factor; the latent grid never
    /// receives sub-pixel mask information, so the pool is fine.
    pub fn to_latent_tensor(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        self.to_latent_tensor_factor(VAE_FACTOR as usize, device, dtype)
    }

    /// Like [`to_latent_tensor`] but for a caller-supplied spatial
    /// downsample `factor`. SD-family VAEs are 8×; DC-AE (Sana) is 32×.
    pub fn to_latent_tensor_factor(
        &self,
        factor: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let iw = self.width as usize;
        let ih = self.height as usize;
        if iw == 0 || ih == 0 {
            anyhow::bail!("empty mask");
        }
        if factor == 0 {
            anyhow::bail!("mask downsample factor must be non-zero");
        }
        let latent_w = iw / factor;
        let latent_h = ih / factor;
        if latent_w == 0 || latent_h == 0 {
            anyhow::bail!(
                "mask {iw}x{ih} too small to downsample by {factor}",
            );
        }
        let mut latent = vec![0f32; latent_w * latent_h];
        let norm = 1.0 / (factor * factor) as f32;
        for ly in 0..latent_h {
            for lx in 0..latent_w {
                let mut sum = 0f32;
                for ky in 0..factor {
                    let sy = ly * factor + ky;
                    if sy >= ih {
                        continue;
                    }
                    for kx in 0..factor {
                        let sx = lx * factor + kx;
                        if sx >= iw {
                            continue;
                        }
                        sum += self.pixels[sy * iw + sx];
                    }
                }
                latent[ly * latent_w + lx] = sum * norm;
            }
        }
        let t = Tensor::from_vec(latent, (1, 1, latent_h, latent_w), device)?;
        Ok(t.to_dtype(dtype)?)
    }
}

/// Separable box blur of `radius` (horizontal pass, then vertical).
/// **Edge-clamped**: each output divides the running sum by the count of
/// *in-bounds* pixels in its window, not the full kernel. So a region that's
/// saturated at the image border stays saturated — only true interior
/// boundaries soften. Outpaint depends on this: its mask is white right up
/// to the canvas edge, and dividing by the full kernel there (the old
/// behaviour) faded it, leaving the new strip half-inpainted as a dark band.
fn box_blur_inplace(buf: &mut [f32], w: usize, h: usize, radius: usize) {
    if radius == 0 || w == 0 || h == 0 {
        return;
    }
    let mut scratch = vec![0f32; buf.len()];

    // Horizontal.
    for y in 0..h {
        let row_start = y * w;
        let mut sum = 0f32;
        for x in 0..radius.min(w) {
            sum += buf[row_start + x];
        }
        for x in 0..w {
            let add = x + radius;
            if add < w {
                sum += buf[row_start + add];
            }
            if x > radius {
                sum -= buf[row_start + x - radius - 1];
            }
            let lo = x.saturating_sub(radius);
            let hi = (x + radius).min(w - 1);
            scratch[row_start + x] = sum / (hi - lo + 1) as f32;
        }
    }

    // Vertical.
    for x in 0..w {
        let mut sum = 0f32;
        for y in 0..radius.min(h) {
            sum += scratch[y * w + x];
        }
        for y in 0..h {
            let add = y + radius;
            if add < h {
                sum += scratch[add * w + x];
            }
            if y > radius {
                sum -= scratch[(y - radius - 1) * w + x];
            }
            let lo = y.saturating_sub(radius);
            let hi = (y + radius).min(h - 1);
            buf[y * w + x] = sum / (hi - lo + 1) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb, Rgba};

    #[test]
    fn solid_one_is_all_ones() {
        let m = Mask::solid_one(8, 4);
        assert_eq!(m.pixels.len(), 32);
        assert!(m.pixels.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn invert_swaps_polarity() {
        let mut m = Mask::solid_one(2, 2);
        m.pixels = vec![0.0, 0.25, 0.75, 1.0];
        m.invert();
        assert_eq!(m.pixels, vec![1.0, 0.75, 0.25, 0.0]);
    }

    #[test]
    fn feather_softens_edges() {
        let mut m = Mask {
            width: 32,
            height: 32,
            pixels: vec![0f32; 32 * 32],
        };
        // 16×16 block in the centre.
        for y in 8..24 {
            for x in 8..24 {
                m.pixels[y * 32 + x] = 1.0;
            }
        }
        m.feather(4);
        // Centre saturated.
        assert!(m.pixels[16 * 32 + 16] > 0.95);
        // Inside boundary drops below saturation.
        assert!(m.pixels[8 * 32 + 8] < 0.5 && m.pixels[8 * 32 + 8] > 0.0);
        // Far corner ~ zero.
        assert!(m.pixels[0] < 0.05);
    }

    #[test]
    fn load_grayscale_reads_brightness() {
        let mut img = image::GrayImage::new(4, 4);
        // Half black, half white columns.
        for y in 0..4 {
            for x in 0..4 {
                let val = if x < 2 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, Luma([val]));
            }
        }
        let tmp = std::env::temp_dir().join("plakat_mask_test_l.png");
        img.save(&tmp).unwrap();
        let m = Mask::load(&tmp, 4, 4).unwrap();
        // Left half ~ 0, right half ~ 1.
        assert!(m.pixels[0] < 0.05);
        assert!(m.pixels[3] > 0.95);
    }

    #[test]
    fn load_rgba_reads_alpha() {
        let mut img = image::RgbaImage::new(4, 4);
        // All red, alpha varies.
        for y in 0..4 {
            for x in 0..4 {
                let a = if x < 2 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, Rgba([255, 0, 0, a]));
            }
        }
        let tmp = std::env::temp_dir().join("plakat_mask_test_rgba.png");
        img.save(&tmp).unwrap();
        let m = Mask::load(&tmp, 4, 4).unwrap();
        // Red channel ignored; alpha determines mask.
        assert!(m.pixels[0] < 0.05);
        assert!(m.pixels[3] > 0.95);
    }

    #[test]
    fn load_rgb_uses_luminance() {
        let mut img = image::RgbImage::new(4, 4);
        // Left half pure black, right half pure white.
        for y in 0..4 {
            for x in 0..4 {
                let v = if x < 2 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let tmp = std::env::temp_dir().join("plakat_mask_test_rgb.png");
        img.save(&tmp).unwrap();
        let m = Mask::load(&tmp, 4, 4).unwrap();
        assert!(m.pixels[0] < 0.05);
        assert!(m.pixels[3] > 0.95);
    }

    #[test]
    fn to_latent_tensor_pools_by_eight() {
        let m = Mask::solid_one(64, 32);
        let t = m
            .to_latent_tensor(&Device::Cpu, DType::F32)
            .expect("latent tensor");
        assert_eq!(t.dims(), &[1, 1, 32 / 8, 64 / 8]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn to_latent_tensor_rejects_too_small() {
        let m = Mask::solid_one(4, 4);
        assert!(m.to_latent_tensor(&Device::Cpu, DType::F32).is_err());
    }

    #[test]
    fn to_latent_tensor_factor_pools_by_thirtytwo() {
        // DC-AE / Sana inpaint path: 32× downsample.
        let m = Mask::solid_one(128, 64);
        let t = m
            .to_latent_tensor_factor(32, &Device::Cpu, DType::F32)
            .expect("latent tensor");
        assert_eq!(t.dims(), &[1, 1, 64 / 32, 128 / 32]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn to_latent_tensor_factor_rejects_zero() {
        let m = Mask::solid_one(64, 64);
        assert!(m.to_latent_tensor_factor(0, &Device::Cpu, DType::F32).is_err());
    }
}
