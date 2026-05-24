//! Cheap latent → RGB projection for live previews during
//! denoise. Multiplies the 4-channel SD / SDXL latent by a
//! 4×3 community-derived matrix to land in approximate RGB
//! space. This is far cheaper than running the full VAE decode
//! (microseconds vs hundreds of milliseconds) at the cost of
//! visual fidelity — colours are recognisable, edges are
//! somewhat blurry. Good enough for "is the generation going
//! the right direction" feedback at low cost.
//!
//! The matrix is the same one Auto1111's "approx" preview path
//! and ComfyUI's "latent2rgb" node use. It's not exact (the real
//! VAE decode is non-linear); it's a known approximation that
//! correlates well with the post-VAE output in colour and gross
//! structure.

use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use image::{ImageBuffer, RgbImage};
use std::path::Path;

/// Community-derived projection from a 4-channel SD/SDXL latent
/// to RGB. Each row maps one latent channel to its
/// `[r, g, b]` contribution; the output is the sum across rows.
///
/// Source: this matrix has been in widespread use across A1111,
/// ComfyUI, and InvokeAI since ~2022. The exact provenance is
/// community lore — it was tuned empirically against a sample
/// of denoised latents + their VAE outputs.
pub const LATENT_RGB_FACTORS_SD: [[f32; 3]; 4] = [
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473],
];

/// Project a `(1, 4, h, w)` SD/SDXL latent to a CPU RGB
/// `ImageBuffer`. Output is at the latent's spatial resolution
/// (1/8 of the final image dims); caller resizes if needed.
///
/// The projection runs on CPU after copying the latent down —
/// the per-step VAE decode is *the* expensive thing we're
/// avoiding, so a 64×64 float matmul on CPU is fine.
pub fn project_latent_sd_to_rgb(latent: &Tensor) -> Result<RgbImage> {
    let dims = latent.dims4()?;
    let (b, c, h, w) = dims;
    if c != 4 {
        anyhow::bail!(
            "latent preview expects 4 channels (SD/SDXL), got {c} (Flux/SD3 use 16 \
             — preview path not wired for those variants)"
        );
    }
    if b != 1 {
        anyhow::bail!("latent preview expects batch 1, got {b}");
    }
    // Move to CPU + f32 for the projection math.
    let cpu = latent.to_device(&candle_core::Device::Cpu)?;
    let cpu = cpu.to_dtype(candle_core::DType::F32)?;
    // (1, 4, h, w) → (4, h*w)
    let flat: Vec<f32> = cpu
        .i(0)?
        .reshape(((), h * w))?
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let mut rgb = vec![0u8; h * w * 3];
    for pixel_idx in 0..(h * w) {
        let mut r = 0.0f32;
        let mut g = 0.0f32;
        let mut bch = 0.0f32;
        for ch in 0..4 {
            let v = flat[ch * h * w + pixel_idx];
            let f = LATENT_RGB_FACTORS_SD[ch];
            r += v * f[0];
            g += v * f[1];
            bch += v * f[2];
        }
        // The projection lands in roughly [-1, 1]; remap to
        // [0, 255] via the standard `+1)/2 * 255` rescale used
        // by every SD viewer. Clip to handle tail outliers from
        // the early-noise steps.
        rgb[pixel_idx * 3] = remap_to_u8(r);
        rgb[pixel_idx * 3 + 1] = remap_to_u8(g);
        rgb[pixel_idx * 3 + 2] = remap_to_u8(bch);
    }
    ImageBuffer::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow::anyhow!("preview rgb buffer alloc"))
}

fn remap_to_u8(v: f32) -> u8 {
    let scaled = ((v + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0);
    scaled.round() as u8
}

/// Project + upscale + save in one call. Used by the denoise
/// loop's preview hook. `target_dim` is the longer side of the
/// preview PNG (the projected RGB is upscaled with Lanczos to
/// match). The same path keeps getting overwritten so the user
/// can `open` the preview once and watch it evolve.
pub fn write_latent_preview_sd(
    latent: &Tensor,
    out_path: &Path,
    target_dim: u32,
) -> Result<()> {
    let raw = project_latent_sd_to_rgb(latent)?;
    let (lw, lh) = raw.dimensions();
    let (out_w, out_h) = if lw >= lh {
        let ratio = target_dim as f32 / lw as f32;
        (target_dim, ((lh as f32) * ratio).round() as u32)
    } else {
        let ratio = target_dim as f32 / lh as f32;
        (((lw as f32) * ratio).round() as u32, target_dim)
    };
    let resized = if out_w == lw && out_h == lh {
        raw
    } else {
        image::imageops::resize(&raw, out_w, out_h, image::imageops::FilterType::Lanczos3)
    };
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating preview dir {}", parent.display()))?;
        }
    }
    // Direct save via the `image` crate — preview PNGs intentionally
    // skip the v0.17 phase 3 metadata embedding (they're scratch
    // files, not final outputs).
    resized
        .save(out_path)
        .with_context(|| format!("writing preview {}", out_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn projects_zero_latent_to_mid_grey() {
        // (1, 4, 2, 2) of zeros. After projection: r=g=b=0;
        // remap_to_u8(0) = 128 (mid-grey, since +1)/2 * 255 = 127.5
        // rounded up).
        let latent = Tensor::zeros((1, 4, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let rgb = project_latent_sd_to_rgb(&latent).unwrap();
        assert_eq!(rgb.dimensions(), (2, 2));
        assert_eq!(rgb.get_pixel(0, 0).0, [128, 128, 128]);
        assert_eq!(rgb.get_pixel(1, 1).0, [128, 128, 128]);
    }

    #[test]
    fn projects_first_channel_positive_warm_tones() {
        // Latent with channel 0 = 1.0, others 0. Projection:
        // r = 0.298, g = 0.207, b = 0.208 → warm tone.
        let mut buf = vec![0f32; 4 * 2 * 2];
        for i in 0..4 {
            buf[i] = 1.0; // channel 0, pixels 0..4
        }
        let latent = Tensor::from_vec(buf, (1, 4, 2, 2), &Device::Cpu).unwrap();
        let rgb = project_latent_sd_to_rgb(&latent).unwrap();
        let px = rgb.get_pixel(0, 0).0;
        // (0.298+1)/2 * 255 ≈ 165 (warm-ish red), green slightly
        // less, blue slightly less.
        assert!(px[0] > px[1], "red > green");
        assert!(px[0] > px[2], "red > blue");
        assert!(px[0] > 140 && px[0] < 180, "red around mid-warm, got {}", px[0]);
    }

    #[test]
    fn rejects_wrong_channel_count() {
        // Flux/SD3 latents are 16ch — explicit error message.
        let latent =
            Tensor::zeros((1, 16, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let err = project_latent_sd_to_rgb(&latent).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("4 channels"), "got {msg}");
        assert!(msg.contains("16"), "got {msg}");
    }

    #[test]
    fn rejects_batch_gt_1() {
        let latent =
            Tensor::zeros((2, 4, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let err = project_latent_sd_to_rgb(&latent).unwrap_err();
        assert!(format!("{err}").contains("batch 1"));
    }

    #[test]
    fn write_preview_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("preview.png");
        // 8×8 latent → 64-px preview at target_dim=64.
        let latent = Tensor::zeros((1, 4, 8, 8), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        write_latent_preview_sd(&latent, &out, 64).unwrap();
        let read = image::open(&out).unwrap();
        assert_eq!(read.width(), 64);
        assert_eq!(read.height(), 64);
    }

    #[test]
    fn write_preview_rectangular_latent_keeps_aspect() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("preview.png");
        // 16-wide × 8-tall latent → 256-wide preview (target_dim=256).
        let latent = Tensor::zeros((1, 4, 8, 16), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        write_latent_preview_sd(&latent, &out, 256).unwrap();
        let read = image::open(&out).unwrap();
        assert_eq!(read.width(), 256);
        assert_eq!(read.height(), 128);
    }

    #[test]
    fn remap_clips_outliers() {
        // Early-step latents can have values past ±2. Clip without
        // wrapping.
        assert_eq!(remap_to_u8(5.0), 255);
        assert_eq!(remap_to_u8(-5.0), 0);
        assert_eq!(remap_to_u8(0.0), 128);
        assert_eq!(remap_to_u8(1.0), 255);
        assert_eq!(remap_to_u8(-1.0), 0);
    }
}
