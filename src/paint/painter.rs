//! The coarse-to-fine painter (RFC PAINT-1 P0) — and the filter-gate it exists to test.
//!
//! Given a reference IMAGE, paint it stroke by stroke under PAINT-1's hard constraints: a finite, inviolable
//! stroke **budget**; a limited **palette** under Kubelka-Munk mixing; a **minimum brush size**; and strokes
//! that follow an **orientation field** and are laid with the bristle brush (deposit + pickup). Crucially the
//! reference each brush layer paints from is BLURRED proportionally to the brush size, so a large brush has no
//! detail to trace — the constraints, not a matching objective, decide when a pass is done.
//!
//! This is the P0 make-or-break: if the output reads as a painterly filter with the budget and palette
//! constraints active, the core thesis is wrong. [`traceability`] measures how close the output is to the
//! full-resolution reference — a filter scores high, a painting lower (structure kept, surface invented).

use image::{imageops, RgbImage};

use crate::paint::canvas::Canvas;
use crate::paint::color::{self, Srgb};
use crate::paint::mixer;
use crate::paint::palette::Palette;
use crate::paint::stroke::{BrushConfig, Stroke};

/// Parameters for a paint-from-image run.
#[derive(Clone, Debug)]
pub struct PaintParams {
    pub palette: Palette,
    /// Total stroke budget across all layers — inviolable.
    pub budget: usize,
    /// Brush radii, coarse → fine. A radius below `min_brush` is skipped.
    pub brush_sizes: Vec<f32>,
    /// The smallest brush allowed, so the finest pass still cannot chase pixel detail.
    pub min_brush: f32,
    /// Charge multiplier for a stroke's load (how much paint the brush holds vs its footprint).
    pub charge: f32,
    pub seed: u64,
    pub brush: BrushConfig,
}

impl PaintParams {
    /// A sensible default over a palette at a stroke budget.
    pub fn new(palette: Palette, budget: usize) -> Self {
        Self { palette, budget, brush_sizes: vec![28.0, 14.0, 7.0], min_brush: 4.0, charge: 6.0, seed: 42, brush: BrushConfig::default() }
    }
}

/// The result of a paint run.
pub struct PaintResult {
    pub canvas: Canvas,
    /// How many strokes were actually laid (≤ budget).
    pub strokes: usize,
}

fn luma_map(img: &RgbImage) -> Vec<f32> {
    img.pixels().map(|p| color::linear_luma(color::srgb_to_linear(p.0))).collect()
}

/// Sobel gradient of a luma map → `(gx, gy, magnitude)` per pixel.
fn sobel(luma: &[f32], w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let (wi, hi) = (w as i32, h as i32);
    let at = |x: i32, y: i32| -> f32 {
        let x = x.clamp(0, wi - 1) as usize;
        let y = y.clamp(0, hi - 1) as usize;
        luma[y * w as usize + x]
    };
    let mut gx = vec![0f32; luma.len()];
    let mut gy = vec![0f32; luma.len()];
    for y in 0..hi {
        for x in 0..wi {
            let i = (y as usize) * w as usize + x as usize;
            gx[i] = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1);
            gy[i] = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1);
        }
    }
    (gx, gy)
}

/// The isophote (stroke) direction at a pixel: perpendicular to the luminance gradient. In a flat region the
/// gradient vanishes, so we fall back to horizontal.
fn stroke_dir(gx: f32, gy: f32) -> [f32; 2] {
    let m = (gx * gx + gy * gy).sqrt();
    if m < 1e-4 {
        [1.0, 0.0]
    } else {
        // perpendicular to (gx,gy) is (-gy,gx)
        [-gy / m, gx / m]
    }
}

/// Deterministic per-index jitter in `[-0.5,0.5]` (a hashed LCG — no RNG dependency, reproducible).
fn jitter(seed: u64, k: u64) -> f32 {
    let mut z = seed.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z as f32 / u64::MAX as f32) - 0.5
}

/// sRGB distance (linear-RGB Euclidean) — cheap, for the placement error map.
fn rgb_dist(a: Srgb, b: Srgb) -> f32 {
    let (la, lb) = (color::srgb_to_linear(a), color::srgb_to_linear(b));
    ((la[0] - lb[0]).powi(2) + (la[1] - lb[1]).powi(2) + (la[2] - lb[2]).powi(2)).sqrt()
}

/// Grow a stroke in ONE direction (`sign` = +1 forward, −1 backward) from the seed along the orientation
/// field, ending when the reference colour drifts too far from the stroke's colour or the half-length cap hits.
fn grow_half(x0: f32, y0: f32, sign: f32, radius: f32, gx: &[f32], gy: &[f32], reference: &RgbImage, color0: Srgb) -> Vec<[f32; 2]> {
    let (w, h) = (reference.width(), reference.height());
    let max_len = (radius * 2.5).max(radius + 1.0);
    let step = (radius * 0.6).max(1.0);
    let mut pts = Vec::new();
    let mut last = [0f32, 0f32];
    let (mut x, mut y) = (x0, y0);
    let mut travelled = 0.0;
    while travelled < max_len {
        let (ix, iy) = (x.round().clamp(0.0, w as f32 - 1.0) as usize, y.round().clamp(0.0, h as f32 - 1.0) as usize);
        let i = iy * w as usize + ix;
        let mut d = stroke_dir(gx[i], gy[i]);
        d = [d[0] * sign, d[1] * sign];
        // Keep the direction from flipping 180° between steps.
        if travelled > 0.0 && last[0] * d[0] + last[1] * d[1] < 0.0 {
            d = [-d[0], -d[1]];
        }
        x += d[0] * step;
        y += d[1] * step;
        if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
            break;
        }
        let here = reference.get_pixel(x as u32, y as u32).0;
        if rgb_dist(here, color0) > 0.18 && travelled > step {
            break;
        }
        pts.push([x, y]);
        last = d;
        travelled += step;
    }
    pts
}

/// Grow a stroke through the seed in BOTH directions (Hertzmann), so a seed mid-feature paints the whole
/// isophote it sits on, not just the half below it.
fn grow_path(x0: f32, y0: f32, radius: f32, gx: &[f32], gy: &[f32], reference: &RgbImage, color0: Srgb) -> Vec<[f32; 2]> {
    let mut back = grow_half(x0, y0, -1.0, radius, gx, gy, reference, color0);
    back.reverse();
    let fwd = grow_half(x0, y0, 1.0, radius, gx, gy, reference, color0);
    back.push([x0, y0]);
    back.extend(fwd);
    back
}

/// Paint a reference image under the PAINT-1 constraints, returning the canvas and the stroke count.
pub fn paint_from_image(input: &RgbImage, p: &PaintParams) -> PaintResult {
    let (w, h) = (input.width(), input.height());
    let mut canvas = Canvas::white(w, h, p.palette, 0.85);
    let n = p.palette.pigments.len();
    // Mixture cache keyed on the quantised reference colour — thousands of strokes sample similar colours.
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let mut mixture_for = |target: Srgb, palette: &Palette, charge: f32| -> Vec<f32> {
        let key = ((target[0] as u32 >> 2) << 12) | ((target[1] as u32 >> 2) << 6) | (target[2] as u32 >> 2);
        let base = cache.entry(key).or_insert_with(|| {
            let m = mixer::solve_mixture(palette, target, 3);
            let mut v = vec![0f32; n];
            for (&idx, &w) in m.pigments.iter().zip(m.weights.iter()) {
                v[idx] = w;
            }
            v
        });
        base.iter().map(|c| c * charge).collect()
    };

    let mut placed = 0usize;
    let mut k = 0u64;
    let sizes: Vec<f32> = p.brush_sizes.iter().copied().filter(|&r| r >= p.min_brush).collect();
    for (layer, &radius) in sizes.iter().enumerate() {
        // The coarsest layer is a block-in: it covers the whole canvas so no white ground survives. Later
        // layers only restate where the canvas is still wrong.
        let block_in = layer == 0;
        if placed >= p.budget {
            break;
        }
        // The reference this layer paints from is blurred ∝ the brush — a coarse brush has no detail to trace.
        let reference = imageops::blur(input, radius * 0.7);
        let luma = luma_map(&reference);
        let (gx, gy) = sobel(&luma, w, h);
        let grid = (radius * 0.9).max(1.5);

        let cols = ((w as f32) / grid).ceil() as u32;
        let rows = ((h as f32) / grid).ceil() as u32;
        for gyi in 0..rows {
            for gxi in 0..cols {
                if placed >= p.budget {
                    break;
                }
                k += 1;
                let jx = jitter(p.seed, k) * grid;
                let jy = jitter(p.seed, k.wrapping_add(1)) * grid;
                let cx = (gxi as f32 + 0.5) * grid + jx;
                let cy = (gyi as f32 + 0.5) * grid + jy;
                if cx < 0.0 || cy < 0.0 || cx >= w as f32 || cy >= h as f32 {
                    continue;
                }
                let (ix, iy) = (cx as u32, cy as u32);
                let target = reference.get_pixel(ix, iy).0;
                // Later layers only restate where the canvas is still notably wrong; the block-in covers all.
                if !block_in && rgb_dist(canvas.color_at(ix, iy), target) < 0.06 {
                    continue;
                }
                let load = mixture_for(target, &p.palette, p.charge);
                let path = grow_path(cx, cy, radius, &gx, &gy, &reference, target);
                let s = Stroke { path, width0: radius, width1: (radius * 0.6).max(p.min_brush * 0.6), load, pressure: 1.0, wetness: 1.0 };
                s.rasterize(&mut canvas, &p.brush);
                placed += 1;
            }
        }
    }
    PaintResult { canvas, strokes: placed }
}

/// Traceability (RFC PAINT-1 §12.1): the mean linear-luma correlation between a painted image and its
/// reference at full resolution. High (→1) means the painting traced the reference (a filter); a real painting
/// keeps the structure but invents the surface, so it correlates moderately, not perfectly.
pub fn traceability(painted: &RgbImage, reference: &RgbImage) -> f32 {
    if painted.dimensions() != reference.dimensions() {
        return f32::NAN;
    }
    let a = luma_map(painted);
    let b = luma_map(reference);
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    fn gradient_img(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, _y| {
            let t = x as f32 / w as f32;
            image::Rgb([(40.0 + 180.0 * t) as u8, (60.0 + 120.0 * t) as u8, (30.0 + 60.0 * t) as u8])
        })
    }

    #[test]
    fn painter_respects_the_budget_and_is_deterministic() {
        let img = gradient_img(64, 48);
        let mut p = PaintParams::new(palette::EARTH, 120);
        p.brush_sizes = vec![16.0, 8.0];
        let a = paint_from_image(&img, &p);
        assert!(a.strokes <= 120, "budget honoured ({} ≤ 120)", a.strokes);
        assert!(a.strokes > 0, "some strokes laid");
        let b = paint_from_image(&img, &p);
        assert_eq!(a.canvas.to_image().into_raw(), b.canvas.to_image().into_raw(), "deterministic");
    }

    #[test]
    fn painting_keeps_structure_without_perfectly_tracing() {
        // The painted output should correlate with the reference (structure survives) but NOT perfectly
        // (surface is invented, budget/palette constrain it) — the filter-gate signal.
        let img = gradient_img(80, 60);
        let mut p = PaintParams::new(palette::EARTH, 300);
        p.brush_sizes = vec![20.0, 10.0];
        let out = paint_from_image(&img, &p).canvas.to_image();
        let tr = traceability(&out, &img);
        assert!(tr > 0.5, "structure survives (corr {tr})");
        assert!(tr < 0.999, "not a pixel-perfect trace (corr {tr})");
    }
}
