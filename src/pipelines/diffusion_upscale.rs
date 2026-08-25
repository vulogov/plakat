//! ControlNet-Tile diffusion upscaling (SUPIR-lite).
//!
//! Coherent 512 → 2K/4K with hallucinated detail. The flow:
//!   1. **Pre-upscale** the input `scale×` (Lanczos) — the "blurry" large image.
//!   2. **Tiled img2img** at the model's native tile size, moderate strength, each tile conditioned
//!      by **ControlNet-Tile** on *its own* blurry content → adds detail while staying faithful to
//!      the tile's structure (no drift, no seams of invented content).
//!   3. **Feathered overlap blend** of the decoded tiles back into the full canvas.
//!
//! ControlNet-Tile is SD 1.5 / SDXL. Runs on the standard `t2i::Pipeline` (own SD UNet default) via
//! its `blend_latents_one` img2img primitive + a per-tile `ControlRequest`.

use anyhow::{Context, Result};
use candle_core::Tensor;
use image::RgbImage;

use crate::pipelines::controlnet::{ControlKind, ControlNet, ControlNetVariant, ControlRequest};
use crate::pipelines::portrait::{GenRequest, Pipeline};
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::sd_core::SdVariant;

/// Tiled diffusion-upscale parameters.
pub struct Options {
    pub input: std::path::PathBuf,
    pub out_path: std::path::PathBuf,
    /// Upscale factor over the input (e.g. 2.0, 4.0).
    pub scale: f32,
    /// Tile side in pixels (model working res; SD 1.5 → 512).
    pub tile: u32,
    /// Overlap between adjacent tiles in pixels (feathered).
    pub overlap: u32,
    /// img2img denoise strength per tile (0.3–0.5 keeps structure; higher invents more).
    pub tile_strength: f32,
    /// ControlNet-Tile residual scale.
    pub cn_strength: f32,
    pub steps: usize,
    pub guidance: f64,
    pub prompt: String,
    pub negative: String,
    pub seed: u64,
    pub scheduler: SchedulerKind,
}

/// Tile origins along one axis: strided starts, with the last tile pinned to the far edge so the
/// whole extent is covered even when `total` isn't a multiple of the stride.
fn tile_origins(total: u32, tile: u32, stride: u32) -> Vec<u32> {
    if total <= tile {
        return vec![0];
    }
    let mut v = Vec::new();
    let mut x = 0u32;
    loop {
        if x + tile >= total {
            v.push(total - tile);
            break;
        }
        v.push(x);
        x += stride.max(1);
    }
    v.dedup();
    v
}

/// Separable feather weights (tile×tile): a **smoothstep** ramp over `overlap` px at each border, 1.0 in
/// the interior. Blending decoded tiles by this weight hides seams in the overlap regions; the C¹
/// smoothstep (vs a linear ramp) removes the faint weight-slope discontinuity at the overlap edge.
/// (RFC SEAMS-1 P2.)
fn feather_map(tile: u32, overlap: u32) -> Vec<f32> {
    let ramp = |i: u32| -> f32 {
        if overlap == 0 {
            return 1.0;
        }
        let d = i.min(tile - 1 - i); // distance to nearest edge
        let r = ((d as f32 + 0.5) / overlap as f32).clamp(0.0, 1.0);
        (r * r * (3.0 - 2.0 * r)).clamp(0.05, 1.0) // smoothstep, floored for the final divide guard
    };
    let mut w = vec![0f32; (tile * tile) as usize];
    for y in 0..tile {
        let wy = ramp(y);
        for x in 0..tile {
            w[(y * tile + x) as usize] = ramp(x) * wy;
        }
    }
    w
}

/// Per-channel mean colour offset to ADD to a decoded tile so it matches the already-placed canvas over
/// their overlap (RFC SEAMS-1 P2 — cross-tile colour match). Averaged over the tile pixels whose canvas
/// cell already has weight (`wsum>0`), then clamped so a genuinely different tile is reduced, not warped.
/// `[0;3]` when there's no overlap yet (the first tile placed at any row/col).
fn tile_color_offset(canvas: &[f32], wsum: &[f32], cw: usize, ox: u32, oy: u32, rgb: &[u8], tile: u32) -> [f32; 3] {
    let (mut off, mut n) = ([0f64; 3], 0u64);
    for ty in 0..tile {
        for tx in 0..tile {
            let cidx = (oy + ty) as usize * cw + (ox + tx) as usize;
            if wsum[cidx] > 1e-6 {
                let sidx = ((ty * tile + tx) * 3) as usize;
                for c in 0..3 {
                    off[c] += canvas[cidx * 3 + c] as f64 / wsum[cidx] as f64 - rgb[sidx + c] as f64;
                }
                n += 1;
            }
        }
    }
    if n == 0 {
        return [0.0; 3];
    }
    [0, 1, 2].map(|c| ((off[c] / n as f64) as f32).clamp(-24.0, 24.0))
}

/// Pack an RGB image crop into a `(1,3,H,W)` tensor. `signed` → `[-1,1]` (VAE input); else `[0,1]`
/// (ControlNet conditioning).
fn img_to_chw(
    img: &RgbImage,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    signed: bool,
) -> Result<Tensor> {
    let (w, h) = img.dimensions();
    let total = (w * h) as usize;
    let mut buf = vec![0f32; 3 * total];
    let (r, rest) = buf.split_at_mut(total);
    let (g, b) = rest.split_at_mut(total);
    for (i, px) in img.pixels().enumerate() {
        let f = |c: u8| if signed { c as f32 / 255.0 * 2.0 - 1.0 } else { c as f32 / 255.0 };
        r[i] = f(px[0]);
        g[i] = f(px[1]);
        b[i] = f(px[2]);
    }
    Ok(Tensor::from_vec(buf, (1, 3, h as usize, w as usize), device)?.to_dtype(dtype)?)
}

impl Pipeline {
    /// ControlNet-Tile diffusion upscale. Loads the Tile ControlNet once, refines every tile via
    /// img2img + Tile conditioning, and feathers the decoded tiles into the output.
    pub async fn diffusion_upscale(&self, opts: &Options) -> Result<()> {
        let core = self.core();
        let (device, dtype) = (core.device.clone(), core.dtype);

        // 1. Pre-upscale (Lanczos) to the target size.
        let src = image::open(&opts.input)
            .with_context(|| format!("opening {}", opts.input.display()))?
            .to_rgb8();
        let (sw, sh) = src.dimensions();
        let (tw, th) = (
            ((sw as f32 * opts.scale).round() as u32).max(opts.tile),
            ((sh as f32 * opts.scale).round() as u32).max(opts.tile),
        );
        let big = image::imageops::resize(&src, tw, th, image::imageops::FilterType::Lanczos3);
        // SEAMS-1 P7: a light pre-sharpen so the Lanczos base isn't mushy going into the refine —
        // ControlNet-Tile then has crisper structure to lock onto (esp. on flat/soft inputs), which reads
        // as more real detail after diffusion. Gentle (sigma 1.2, threshold 3) so it doesn't ring.
        let big = image::imageops::unsharpen(&big, 1.2, 3);
        crate::ui::progress::println(&format!(
            "diffusion-upscale: {sw}×{sh} → {tw}×{th} (tile {}, overlap {}, strength {:.2}, pre-sharpen)",
            opts.tile, opts.overlap, opts.tile_strength
        ));

        // 2. Load the Tile ControlNet once (SD 1.5 / SDXL by the loaded variant).
        let cn_variant = match core.variant {
            SdVariant::Sdxl => ControlNetVariant::Sdxl,
            _ => ControlNetVariant::Sd15,
        };
        let net = ControlNet::load(device.clone(), dtype, ControlKind::Tile, cn_variant)
            .await
            .context("loading ControlNet-Tile weights")?;

        // 3. Tile grid + accumulation canvases.
        let stride = opts.tile.saturating_sub(opts.overlap).max(1);
        let xs = tile_origins(tw, opts.tile, stride);
        let ys = tile_origins(th, opts.tile, stride);
        let feather = feather_map(opts.tile, opts.overlap);
        let (cw, ch) = (tw as usize, th as usize);
        let mut canvas = vec![0f32; cw * ch * 3];
        let mut wsum = vec![0f32; cw * ch];
        let mask = Tensor::ones((1, 1, (opts.tile / 8) as usize, (opts.tile / 8) as usize), dtype, &device)?;
        let n_tiles = xs.len() * ys.len();

        let mut done = 0usize;
        for &oy in &ys {
            for &ox in &xs {
                done += 1;
                crate::ui::progress::println(&format!("  tile {done}/{n_tiles} @ ({ox},{oy})"));
                let crop = image::imageops::crop_imm(&big, ox, oy, opts.tile, opts.tile).to_image();

                // Encode + condition on the same blurry tile.
                let pixels = img_to_chw(&crop, &device, dtype, true)?;
                let base_latents = self.vae_encode_pixels(&pixels)?;
                let conditioning = img_to_chw(&crop, &device, dtype, false)?;
                let control = ControlRequest {
                    net: &net,
                    conditioning,
                    strength: opts.cn_strength,
                    start: 0.0,
                    end: 1.0,
                };
                let gen_req = GenRequest {
                    prompt: opts.prompt.clone(),
                    negative: opts.negative.clone(),
                    photos: Vec::new(),
                    width: opts.tile,
                    height: opts.tile,
                    count: 1,
                    steps: opts.steps,
                    guidance: opts.guidance,
                    seed: Some(opts.seed.wrapping_add(done as u64)),
                    out_dir: std::env::temp_dir(),
                    scheduler: opts.scheduler,
                    refine: None,
                    refine_strength: 0.0,
                    face_strength: 0.0,
                    face_bbox: None,
                    face_landmarks: None,
                };
                let latents = self.blend_latents_one(
                    &base_latents,
                    &mask,
                    &gen_req,
                    opts.tile_strength,
                    opts.seed.wrapping_add(done as u64),
                    &[control],
                    None,
                    None,
                )?;
                let (rgb, rw, rh) = self.decode_to_rgb8(&latents)?;
                debug_assert_eq!((rw, rh), (opts.tile, opts.tile));

                // P2 cross-tile colour/tone match (RFC SEAMS-1): each tile denoises independently, so its
                // mean exposure can drift from its neighbours — the feather then blends two exposures into
                // a visible seam. Measure the per-channel mean offset against the ALREADY-placed canvas over
                // this tile's overlap region and shift the tile to match (clamped, so a genuinely different
                // tile is reduced, not warped). The first tile has no overlap → no shift; later tiles chain
                // to it.
                let off = tile_color_offset(&canvas, &wsum, cw, ox, oy, &rgb, opts.tile);

                // Feathered accumulate into the canvas (with the colour-match offset applied).
                for ty in 0..rh {
                    for tx in 0..rw {
                        let w = feather[(ty * opts.tile + tx) as usize];
                        let (gx, gy) = ((ox + tx) as usize, (oy + ty) as usize);
                        let cidx = gy * cw + gx;
                        let sidx = ((ty * rw + tx) * 3) as usize;
                        for c in 0..3 {
                            canvas[cidx * 3 + c] += (rgb[sidx + c] as f32 + off[c]).clamp(0.0, 255.0) * w;
                        }
                        wsum[cidx] += w;
                    }
                }
            }
        }

        // 4. Normalise + write.
        let mut out = vec![0u8; cw * ch * 3];
        for i in 0..cw * ch {
            let w = wsum[i].max(1e-6);
            for c in 0..3 {
                out[i * 3 + c] = (canvas[i * 3 + c] / w).round().clamp(0.0, 255.0) as u8;
            }
        }
        if let Some(parent) = opts.out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        crate::imaging::io::save_rgb_u8(&out, tw, th, &opts.out_path)?;
        crate::ui::progress::println(&format!("→ {}", opts.out_path.display()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feather_map_is_smoothstep_and_monotonic_to_interior() {
        let (tile, overlap) = (16u32, 4u32);
        let w = feather_map(tile, overlap);
        // Interior weight is 1.0.
        assert!((w[(8 * tile + 8) as usize] - 1.0).abs() < 1e-6, "interior = 1.0");
        // Along the interior row (y=8, wy=1.0) the separable weight IS the 1-D ramp — monotone inward.
        let row = |x: u32| w[(8 * tile + x) as usize];
        assert!(row(0) <= row(1) && row(1) <= row(2) && row(2) <= row(3), "ramp rises inward");
        assert!((row(0) - 0.05).abs() < 1e-6, "edge floored at 0.05");
        // Every weight is a product of two floored ramps → strictly positive, at most 1.0.
        assert!(w.iter().all(|&v| v > 0.0 && v <= 1.0 + 1e-6), "weights in (0, 1]");
        // overlap 0 → all ones (no feather).
        assert!(feather_map(8, 0).iter().all(|&v| v == 1.0));
    }

    #[test]
    fn tile_color_offset_matches_a_drifted_tile_and_is_clamped() {
        // 4×4 canvas, one 2×2 tile already placed (value 200) at (0,0), full weight.
        let (cw, ch, tile) = (4usize, 4usize, 2u32);
        let mut canvas = vec![0f32; cw * ch * 3];
        let mut wsum = vec![0f32; cw * ch];
        for gy in 0..2 {
            for gx in 0..2 {
                let c = gy * cw + gx;
                for k in 0..3 {
                    canvas[c * 3 + k] = 200.0;
                }
                wsum[c] = 1.0;
            }
        }
        // A new tile placed at (0,0) whose pixels are darker (120) → offset should be +80, clamped to +24.
        let rgb_dark = vec![120u8; (tile * tile * 3) as usize];
        let off = tile_color_offset(&canvas, &wsum, cw, 0, 0, &rgb_dark, tile);
        assert_eq!(off, [24.0, 24.0, 24.0], "raw +80 clamped to +24 (lift toward the neighbour)");
        // A tile placed with NO overlap (all wsum there is 0) → no shift.
        let off_none = tile_color_offset(&vec![0f32; cw * ch * 3], &vec![0f32; cw * ch], cw, 2, 2, &rgb_dark, tile);
        assert_eq!(off_none, [0.0, 0.0, 0.0], "no overlap → no colour shift");
    }
}
