//! Monocular surface normals via Marigold-Normals (`prs-eth/marigold-normals-v1-1`).
//!
//! Used by `plakat paint --medium durer --normals auto`: a normal map tells the burin which way every
//! surface faces, so its cuts can wrap a form — the one thing a flat picture's values and edges cannot say.
//!
//! Marigold is Stable Diffusion 2 repurposed: the same VAE, the same UNet with EIGHT input channels (the
//! picture's latent beside the latent being denoised) and an empty prompt. Every block is candle's own SD 2.1
//! one, so this module is only the loop:
//!
//! 1. Resize the picture so its long side is 768 (the model's processing resolution), sides multiples of 8,
//!    and MIRROR it outward by a margin on every side: the model reads a picture's border as a wall turning
//!    away, and with the margin that wall falls outside the picture and is cropped off.
//! 2. VAE-encode it → the image latent.
//! 3. From seeded noise, four DDIM steps (v-prediction, trailing timesteps, zero terminal SNR), the UNet seeing
//!    `[image latent, normals latent]` at each.
//! 4. VAE-decode the normals latent → three channels, normalised to unit vectors, resized to the size asked.
//!
//! **Convention.** What this module returns is in IMAGE space: `x` right, `y` DOWN, `z` toward the viewer.
//! The model (and a normal-map PNG, see [`to_png`] / [`from_png`]) has `y` UP — green up, the OpenGL habit.
//!
//! The starting noise is plakat's own seeded draw, so the same picture gives the same map on every run.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};
use image::{Rgb, RgbImage};
use std::path::Path;

const REPO: &str = "prs-eth/marigold-normals-v1-1";
const UNET: &str = "unet/diffusion_pytorch_model.fp16.safetensors";
const VAE: &str = "vae/diffusion_pytorch_model.fp16.safetensors";
const TEXT: &str = "text_encoder/model.fp16.safetensors";
/// The long side the model works at, and the mirrored margin added round the picture before it is read.
const PROCESS: u32 = 768;
const MARGIN: u32 = 64;
const STEPS: usize = 4;
const VAE_SCALE: f64 = 0.18215;
const TRAIN_STEPS: usize = 1000;
/// The noise the map starts from: one fixed draw, so a picture has one map.
const SEED: u64 = 2024;

pub struct NormalsPipeline {
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
    /// The empty prompt's embedding, `(1, 2, 1024)`.
    empty: Tensor,
    device: Device,
}

impl NormalsPipeline {
    pub async fn load(device: Device) -> Result<Self> {
        let unet = crate::hf::download::get_file(REPO, UNET).await.context("fetching the Marigold-Normals UNet")?;
        let vae = crate::hf::download::get_file(REPO, VAE).await.context("fetching the Marigold-Normals VAE")?;
        let text = crate::hf::download::get_file(REPO, TEXT).await.context("fetching the Marigold-Normals text encoder")?;
        let cfg = StableDiffusionConfig::v2_1(None, None, None);
        // The prompt is empty and never changes: encode it once and let the encoder go. Marigold does not pad
        // it — begin, end, nothing else; the encoder is causal, so those two positions of a padded row are the
        // same two.
        let empty = {
            let clip = stable_diffusion::build_clip_transformer(&cfg.clip, &text, &device, DType::F32)?;
            let mut ids = vec![0u32; cfg.clip.max_position_embeddings];
            ids[0] = 49406;
            ids[1] = 49407;
            let ids = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            candle_core::Module::forward(&clip, &ids)?.narrow(1, 0, 2)?.contiguous()?
        };
        let vae = cfg.build_vae(&vae, &device, DType::F32)?;
        let unet = cfg.build_unet(&unet, &device, 8, false, DType::F32)?;
        Ok(Self { unet, vae, empty, device })
    }

    /// The picture's normals at `out_w × out_h`, row-major, unit vectors in image space (`y` down).
    pub fn normal_map(&self, path: &Path, out_w: u32, out_h: u32) -> Result<Vec<[f32; 3]>> {
        let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
        let k = PROCESS as f32 / img.width().max(img.height()) as f32;
        let side = |v: u32| (((v as f32 * k / 8.0).round() as u32).max(8)) * 8;
        let (w, h) = (side(img.width()), side(img.height()));
        let small = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
        let (iw, ih) = (w as usize, h as usize);
        let (w, h) = (iw + 2 * MARGIN as usize, ih + 2 * MARGIN as usize);
        // Folded back on itself at every edge.
        let fold = |v: i64, n: usize| -> usize {
            let n = n as i64;
            let v = v.rem_euclid(2 * n);
            (if v < n { v } else { 2 * n - 1 - v }) as usize
        };
        let mut px = vec![0f32; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                let p = small.get_pixel(fold(x as i64 - MARGIN as i64, iw) as u32, fold(y as i64 - MARGIN as i64, ih) as u32);
                for c in 0..3 {
                    px[c * w * h + y * w + x] = p.0[c] as f32 / 127.5 - 1.0;
                }
            }
        }
        let px = Tensor::from_vec(px, (1, 3, h, w), &self.device)?;
        let image_latent = (self.vae.encode(&px)?.sample()? * VAE_SCALE)?;

        let alphas = alphas_cumprod();
        let mut latent = crate::pipelines::kandinsky::seeded_noise(SEED, 4, h / 8, w / 8, &self.device)?.to_dtype(DType::F32)?;
        for i in 0..STEPS {
            // Trailing timesteps: 999, 749, 499, 249.
            let t = (TRAIN_STEPS as f64 - (i * TRAIN_STEPS) as f64 / STEPS as f64).round() as usize - 1;
            let prev = t as i64 - (TRAIN_STEPS / STEPS) as i64;
            let (a, a_prev) = (alphas[t], if prev >= 0 { alphas[prev as usize] } else { alphas[0] });
            let v = self.unet.forward(&Tensor::cat(&[&image_latent, &latent], 1)?, t as f64, &self.empty)?;
            // v-prediction: the clean latent and the noise it was mixed with, then the mix one step back.
            let x0 = ((&latent * a.sqrt())? - (&v * (1.0 - a).sqrt())?)?;
            let eps = ((&v * a.sqrt())? + (&latent * (1.0 - a).sqrt())?)?;
            latent = ((x0 * a_prev.sqrt())? + (eps * (1.0 - a_prev).sqrt())?)?;
        }
        let out = self.vae.decode(&(latent / VAE_SCALE)?)?.clamp(-1f32, 1f32)?;
        let out = out.i(0)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let n = w * h;
        // Read inside the margin only: the picture's own normals, the mirrored border's thrown away.
        let at = |x: usize, y: usize| -> [f32; 3] {
            let i = (y + MARGIN as usize) * w + x + MARGIN as usize;
            // The model's `y` is up; the sheet's is down.
            [out[i], -out[n + i], out[2 * n + i]]
        };
        let (w, h) = (iw, ih);
        let mut map = Vec::with_capacity((out_w * out_h) as usize);
        for y in 0..out_h as usize {
            for x in 0..out_w as usize {
                let (fx, fy) = ((x as f32 + 0.5) / out_w as f32 * w as f32 - 0.5, (y as f32 + 0.5) / out_h as f32 * h as f32 - 0.5);
                let (fx, fy) = (fx.clamp(0.0, w as f32 - 1.0), fy.clamp(0.0, h as f32 - 1.0));
                let (x0, y0) = (fx as usize, fy as usize);
                let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
                let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
                let mut v = [0f32; 3];
                for j in 0..3 {
                    v[j] = (a[j] * (1.0 - tx) + b[j] * tx) * (1.0 - ty) + (c[j] * (1.0 - tx) + d[j] * tx) * ty;
                }
                map.push(unit(v));
            }
        }
        Ok(map)
    }
}

fn unit(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0, 0.0, 1.0] }
}

/// The scheduler's cumulative alphas: scaled-linear betas, rescaled so the last step is pure noise.
fn alphas_cumprod() -> Vec<f64> {
    let (b0, b1) = (0.00085f64.sqrt(), 0.012f64.sqrt());
    let mut prod = 1.0f64;
    let mut root: Vec<f64> = (0..TRAIN_STEPS)
        .map(|i| {
            let beta = (b0 + (b1 - b0) * i as f64 / (TRAIN_STEPS - 1) as f64).powi(2);
            prod *= 1.0 - beta;
            prod.sqrt()
        })
        .collect();
    let (first, last) = (root[0], root[TRAIN_STEPS - 1]);
    for r in root.iter_mut() {
        *r = (*r - last) * first / (first - last);
    }
    root.into_iter().map(|r| r * r).collect()
}

/// A normal map as a picture, the usual way: `rgb = n · ½ + ½`, with green UP.
pub fn to_png(map: &[[f32; 3]], w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let n = map[(y * w + x) as usize];
        let c = |v: f32| ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
        Rgb([c(n[0]), c(-n[1]), c(n[2])])
    })
}

/// A normal-map picture (green up) read back into image-space unit vectors.
pub fn from_png(img: &RgbImage) -> Vec<[f32; 3]> {
    img.pixels()
        .map(|p| {
            let c = |v: u8| v as f32 / 127.5 - 1.0;
            unit([c(p.0[0]), -c(p.0[1]), c(p.0[2])])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_ends_in_pure_noise() {
        let a = alphas_cumprod();
        assert_eq!(a.len(), TRAIN_STEPS);
        assert!(a[TRAIN_STEPS - 1].abs() < 1e-12, "zero terminal SNR");
        assert!((a[0] - 0.99915).abs() < 1e-4, "the first step is untouched ({})", a[0]);
        assert!(a.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn a_normal_map_survives_the_picture() {
        let map = vec![unit([0.3, -0.5, 0.8]), [0.0, 0.0, 1.0], unit([-0.7, 0.2, 0.4]), unit([0.0, 0.9, 0.3])];
        let back = from_png(&to_png(&map, 2, 2));
        for (a, b) in map.iter().zip(&back) {
            assert!((0..3).all(|j| (a[j] - b[j]).abs() < 0.02), "{a:?} came back {b:?}");
        }
    }
}
