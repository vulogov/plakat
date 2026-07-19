//! Multi-shot composites (non-AI): **HDR exposure fusion** and **focus stacking** over a burst /
//! bracket. Both take a set of same-scene frames (resized to the first's dimensions — no
//! registration, so they suit tripod / steady-burst sequences) and merge them per-pixel:
//!
//! - **Exposure fusion** (Mertens, single-scale): blend bracketed exposures weighted by local
//!   contrast × saturation × well-exposedness — recovers shadow + highlight detail without tone-maps.
//! - **Focus stacking**: pick each pixel from whichever frame is locally sharpest — extends
//!   depth-of-field across a focus bracket.

use anyhow::{ensure, Result};
use image::{imageops, GrayImage, RgbImage};

/// Resize every frame to `(w, h)` (the first frame's size) so they can be merged pixel-wise.
fn normalise(imgs: &[RgbImage]) -> (u32, u32, Vec<RgbImage>) {
    let (w, h) = (imgs[0].width().max(1), imgs[0].height().max(1));
    let norm = imgs
        .iter()
        .map(|i| {
            if i.dimensions() == (w, h) {
                i.clone()
            } else {
                imageops::resize(i, w, h, imageops::FilterType::Lanczos3)
            }
        })
        .collect();
    (w, h, norm)
}

/// Per-pixel absolute Laplacian (a local-contrast / sharpness measure) of a grayscale frame.
fn laplacian_abs(gray: &GrayImage) -> Vec<f32> {
    let (w, h) = (gray.width(), gray.height());
    let at = |x: i32, y: i32| gray.get_pixel(x.clamp(0, w as i32 - 1) as u32, y.clamp(0, h as i32 - 1) as u32).0[0] as f32;
    let mut out = vec![0f32; (w * h) as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let c = at(x, y);
            out[(y as u32 * w + x as u32) as usize] =
                (4.0 * c - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1)).abs();
        }
    }
    out
}

/// Well-exposedness of a normalised channel value: a Gaussian centred on 0.5 (mid-tone).
fn well_exposed(v: f32) -> f32 {
    let d = v - 0.5;
    (-(d * d) / (2.0 * 0.2 * 0.2)).exp()
}

/// HDR exposure fusion (single-scale Mertens): merge bracketed exposures into one well-exposed image.
pub fn exposure_fusion(imgs: &[RgbImage]) -> Result<RgbImage> {
    ensure!(imgs.len() >= 2, "need 2+ exposures to fuse");
    let (w, h, norm) = normalise(imgs);
    let npx = (w * h) as usize;

    // Per-frame weight = (contrast + ε)·(saturation + ε)·(well-exposedness + ε); then normalise the
    // weights across frames at each pixel.
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(norm.len());
    let mut wsum = vec![0f32; npx];
    for im in &norm {
        let lap = laplacian_abs(&imageops::grayscale(im));
        let mut wm = vec![0f32; npx];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let p = im.get_pixel(x, y).0;
                let (r, g, b) = (p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
                let mean = (r + g + b) / 3.0;
                let sat = (((r - mean).powi(2) + (g - mean).powi(2) + (b - mean).powi(2)) / 3.0).sqrt();
                let we = well_exposed(r) * well_exposed(g) * well_exposed(b);
                let wgt = (lap[idx] / 255.0 + 0.02) * (sat + 0.02) * (we + 0.02);
                wm[idx] = wgt;
                wsum[idx] += wgt;
            }
        }
        weights.push(wm);
    }

    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let ws = wsum[idx].max(1e-6);
            let mut acc = [0f32; 3];
            for (k, im) in norm.iter().enumerate() {
                let wgt = weights[k][idx] / ws;
                let p = im.get_pixel(x, y).0;
                for c in 0..3 {
                    acc[c] += p[c] as f32 * wgt;
                }
            }
            out.put_pixel(x, y, image::Rgb([acc[0] as u8, acc[1] as u8, acc[2] as u8]));
        }
    }
    Ok(out)
}

/// Focus stacking: for each pixel take the frame that is locally sharpest (blurred Laplacian, so the
/// selection follows regions rather than single noisy pixels) — extends depth-of-field.
pub fn focus_stack(imgs: &[RgbImage]) -> Result<RgbImage> {
    ensure!(imgs.len() >= 2, "need 2+ frames to stack");
    let (w, h, norm) = normalise(imgs);
    // Sharpness map per frame, smoothed so the per-pixel choice is spatially coherent.
    let sharp: Vec<Vec<f32>> = norm
        .iter()
        .map(|im| {
            let lap = laplacian_abs(&imageops::grayscale(im));
            let g = GrayImage::from_raw(w, h, lap.iter().map(|v| v.min(255.0) as u8).collect()).unwrap();
            imageops::blur(&g, 2.0).pixels().map(|p| p.0[0] as f32).collect()
        })
        .collect();

    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let mut best = 0usize;
            let mut best_s = -1.0f32;
            for (k, s) in sharp.iter().enumerate() {
                if s[idx] > best_s {
                    best_s = s[idx];
                    best = k;
                }
            }
            out.put_pixel(x, y, *norm[best].get_pixel(x, y));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn exposure_fusion_pulls_a_dark_and_bright_pair_toward_the_middle() {
        // A dark frame (40) and a bright frame (215) → the fused mid-tone should sit between them.
        let dark = RgbImage::from_pixel(16, 16, Rgb([40, 40, 40]));
        let bright = RgbImage::from_pixel(16, 16, Rgb([215, 215, 215]));
        let out = exposure_fusion(&[dark, bright]).unwrap();
        let v = out.get_pixel(8, 8).0[0];
        assert!(v > 60 && v < 200, "fused value {v} is between the two exposures");
        assert_eq!(out.dimensions(), (16, 16));
    }

    #[test]
    fn focus_stack_picks_the_sharp_region_from_each_frame() {
        // Frame A: a sharp edge in the left half, flat right. Frame B: the opposite. The stack should
        // take the sharp side from each — so neither flat region wins.
        let a = RgbImage::from_fn(20, 20, |x, _| if x < 10 && x % 2 == 0 { Rgb([0, 0, 0]) } else if x < 10 { Rgb([255, 255, 255]) } else { Rgb([128, 128, 128]) });
        let b = RgbImage::from_fn(20, 20, |x, _| if x >= 10 && x % 2 == 0 { Rgb([0, 0, 0]) } else if x >= 10 { Rgb([255, 255, 255]) } else { Rgb([128, 128, 128]) });
        let out = focus_stack(&[a, b]).unwrap();
        assert_eq!(out.dimensions(), (20, 20));
        // Left half should come from A (sharp there → not flat 128); right half from B.
        assert_ne!(out.get_pixel(2, 10).0, [128, 128, 128], "left picked the sharp frame A");
        assert_ne!(out.get_pixel(16, 10).0, [128, 128, 128], "right picked the sharp frame B");
    }
}
