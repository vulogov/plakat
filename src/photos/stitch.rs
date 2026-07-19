//! Non-AI image compositing: panorama stitching + collage. These are deterministic layout
//! operations — images are normalised and placed edge-to-edge (panorama) or in a padded grid
//! (collage). There is no feature detection / alignment (that would be the AI-adjacent path); this
//! suits pre-cropped strips, tripod sequences, and quick contact-style composites.

use anyhow::{ensure, Result};
use image::{imageops, RgbImage};

/// Panorama layout.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PanoMode {
    Horizontal,
    Vertical,
    Grid,
    /// Horizontal with **overlap registration**: adjacent frames are aligned by cross-correlation
    /// (translation only — best for tripod pans) and the seam is cross-faded.
    HorizontalAligned,
    /// Vertical with overlap registration.
    VerticalAligned,
    /// Feature-matched **homography** stitch: corrects rotation/perspective, not just translation
    /// (the "true" panorama). Falls back to edge-to-edge for frames it can't register.
    Homography,
}

impl PanoMode {
    pub fn from_i32(v: i32) -> PanoMode {
        match v {
            1 => PanoMode::Vertical,
            2 => PanoMode::Grid,
            3 => PanoMode::HorizontalAligned,
            4 => PanoMode::VerticalAligned,
            5 => PanoMode::Homography,
            _ => PanoMode::Horizontal,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            PanoMode::Horizontal => "horizontal",
            PanoMode::Vertical => "vertical",
            PanoMode::Grid => "grid",
            PanoMode::HorizontalAligned => "horizontal (aligned)",
            PanoMode::VerticalAligned => "vertical (aligned)",
            PanoMode::Homography => "homography",
        }
    }
}

/// Edge-to-edge horizontal concatenation of two frames at their common (min) height — the fallback
/// when a smarter stitch (homography / overlap alignment) can't register a pair.
pub(crate) fn concat_h(a: &RgbImage, b: &RgbImage) -> RgbImage {
    let h = a.height().min(b.height()).max(1);
    let pa = resize_to_h(a, h);
    let pb = resize_to_h(b, h);
    let mut canvas = RgbImage::new(pa.width() + pb.width(), h);
    imageops::overlay(&mut canvas, &pa, 0, 0);
    imageops::overlay(&mut canvas, &pb, pa.width() as i64, 0);
    canvas
}

fn resize_to_h(img: &RgbImage, h: u32) -> RgbImage {
    let w = ((img.width() as f32 * h as f32 / img.height().max(1) as f32).round() as u32).max(1);
    imageops::resize(img, w, h, imageops::FilterType::Lanczos3)
}
fn resize_to_w(img: &RgbImage, w: u32) -> RgbImage {
    let h = ((img.height() as f32 * w as f32 / img.width().max(1) as f32).round() as u32).max(1);
    imageops::resize(img, w, h, imageops::FilterType::Lanczos3)
}

/// Stitch `imgs` into a panorama: horizontal (common height, placed left→right), vertical (common
/// width, top→bottom), or an edge-to-edge grid.
pub fn panorama(imgs: &[RgbImage], mode: PanoMode) -> Result<RgbImage> {
    ensure!(imgs.len() >= 2, "need at least 2 images for a panorama");
    match mode {
        PanoMode::Horizontal => {
            let h = imgs.iter().map(|i| i.height()).min().unwrap().max(1);
            let parts: Vec<RgbImage> = imgs.iter().map(|i| resize_to_h(i, h)).collect();
            let total_w: u32 = parts.iter().map(|p| p.width()).sum();
            let mut canvas = RgbImage::new(total_w.max(1), h);
            let mut x = 0i64;
            for p in &parts {
                imageops::overlay(&mut canvas, p, x, 0);
                x += p.width() as i64;
            }
            Ok(canvas)
        }
        PanoMode::Vertical => {
            let w = imgs.iter().map(|i| i.width()).min().unwrap().max(1);
            let parts: Vec<RgbImage> = imgs.iter().map(|i| resize_to_w(i, w)).collect();
            let total_h: u32 = parts.iter().map(|p| p.height()).sum();
            let mut canvas = RgbImage::new(w, total_h.max(1));
            let mut y = 0i64;
            for p in &parts {
                imageops::overlay(&mut canvas, p, 0, y);
                y += p.height() as i64;
            }
            Ok(canvas)
        }
        // An edge-to-edge grid (equal cells, no gaps).
        PanoMode::Grid => crate::imaging::grid::compose_images(imgs, None, 0),
        PanoMode::HorizontalAligned => {
            let h = imgs.iter().map(|i| i.height()).min().unwrap().max(1);
            let parts: Vec<RgbImage> = imgs.iter().map(|i| resize_to_h(i, h)).collect();
            Ok(stitch_aligned_h(&parts))
        }
        // Vertical = rotate the frames 90°, stitch horizontally-aligned, rotate the result back.
        PanoMode::VerticalAligned => {
            let rot: Vec<RgbImage> = imgs.iter().map(|i| imageops::rotate90(i)).collect();
            let refs: Vec<&RgbImage> = rot.iter().collect();
            let w = refs.iter().map(|i| i.height()).min().unwrap().max(1);
            let parts: Vec<RgbImage> = rot.iter().map(|i| resize_to_h(i, w)).collect();
            Ok(imageops::rotate270(&stitch_aligned_h(&parts)))
        }
        // Full feature-matched homography stitch (rotation/perspective; edge-to-edge fallback).
        PanoMode::Homography => Ok(super::homography::stitch(imgs)),
    }
}

/// Estimate the horizontal overlap of two same-height frames by translation registration: slide
/// `b`'s left edge over `a`'s right edge and pick the `(overlap_cols, dy)` that minimises the mean
/// absolute luma difference in the overlap. Returns `None` when no candidate beats the no-overlap
/// baseline confidently (→ caller falls back to edge-to-edge). Works on a downscaled luma copy for
/// speed; the search is bounded to `≤ 45 %` overlap and `± h/12` vertical drift.
fn estimate_overlap_h(a: &RgbImage, b: &RgbImage) -> Option<(u32, i32)> {
    let (aw, h) = (a.width(), a.height());
    let bw = b.width();
    // Downscale factor so the correlation strip is cheap (~cap the working height at 120 px).
    let ds = (h / 120).max(1);
    let luma = |im: &RgbImage| -> (Vec<f32>, u32, u32) {
        let (w2, h2) = ((im.width() / ds).max(1), (im.height() / ds).max(1));
        let small = imageops::resize(im, w2, h2, imageops::FilterType::Triangle);
        let v: Vec<f32> = small.pixels().map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect();
        (v, w2, h2)
    };
    let (la, law, lah) = luma(a);
    let (lb, lbw, _lbh) = luma(b);
    let at = |v: &[f32], w: u32, x: u32, y: u32| v[(y * w + x) as usize];

    let max_ov = ((law.min(lbw)) as f32 * 0.45) as i32;
    let min_ov = 2.max(law as i32 / 20);
    let max_dy = (lah as i32 / 12).max(1);
    let mut best: Option<(f32, u32, i32)> = None; // (score, overlap_small, dy_small)
    for ov in min_ov..=max_ov {
        for dy in -max_dy..=max_dy {
            let mut sum = 0f32;
            let mut n = 0u32;
            // Sample the overlap region on a coarse lattice (step 2) — plenty for a translation fit.
            let mut y = 0u32;
            while (y as i32) < lah as i32 {
                let ay = y as i32;
                let by = y as i32 - dy;
                if by >= 0 && by < lah as i32 && ay < lah as i32 {
                    let mut xo = 0i32;
                    while xo < ov {
                        let ax = law as i32 - ov + xo;
                        if ax >= 0 {
                            sum += (at(&la, law, ax as u32, ay as u32) - at(&lb, lbw, xo as u32, by as u32)).abs();
                            n += 1;
                        }
                        xo += 2;
                    }
                }
                y += 2;
            }
            if n > 16 {
                let score = sum / n as f32;
                if best.map(|(s, ..)| score < s).unwrap_or(true) {
                    best = Some((score, ov as u32, dy));
                }
            }
        }
    }
    // Confidence: the aligned overlap must be clearly flatter than a mid-grey baseline (~ random
    // overlap ≈ 40–60 luma MAD). Reject weak matches so unrelated frames fall back to concatenation.
    let (score, ov_s, dy_s) = best?;
    if score > 22.0 {
        return None;
    }
    let overlap = (ov_s * ds).min(aw.min(bw).saturating_sub(1));
    Some((overlap.max(1), dy_s * ds as i32))
}

/// Stitch same-height frames left→right, aligning each to the running canvas by translation
/// registration and cross-fading the overlap. Frames that don't register confidently are placed
/// edge-to-edge (the old behaviour), so a pile of unrelated shots never gets mangled.
fn stitch_aligned_h(parts: &[RgbImage]) -> RgbImage {
    if parts.is_empty() {
        return RgbImage::new(1, 1);
    }
    let h = parts[0].height();
    let mut canvas = parts[0].clone();
    for b in &parts[1..] {
        let (overlap, dy) = estimate_overlap_h(&canvas, b).unwrap_or((0, 0));
        let cw = canvas.width();
        let new_w = cw + b.width() - overlap;
        // Vertical span: b is shifted by dy, so the canvas may need to grow up/down.
        let top = dy.min(0); // ≤ 0 → b extends above
        let bottom = (dy + b.height() as i32).max(h as i32);
        let new_h = (bottom - top) as u32;
        let mut out = RgbImage::from_pixel(new_w, new_h, image::Rgb([245, 245, 245]));
        let cy = (-top) as i64; // where the old canvas sits vertically
        imageops::overlay(&mut out, &canvas, 0, cy);
        // Place b, cross-fading across the overlap columns.
        let bx0 = cw as i64 - overlap as i64;
        let by0 = cy + dy as i64;
        for by in 0..b.height() {
            let oy = by0 + by as i64;
            if oy < 0 || oy >= new_h as i64 {
                continue;
            }
            for bx in 0..b.width() {
                let ox = bx0 + bx as i64;
                if ox < 0 || ox >= new_w as i64 {
                    continue;
                }
                let src = b.get_pixel(bx, by).0;
                let px = if (bx as u32) < overlap && ox < cw as i64 {
                    // Linear cross-fade: 0 at the seam start (keep canvas) → 1 at overlap end (take b).
                    let t = (bx as f32 + 1.0) / (overlap as f32 + 1.0);
                    let dst = out.get_pixel(ox as u32, oy as u32).0;
                    [
                        (dst[0] as f32 * (1.0 - t) + src[0] as f32 * t) as u8,
                        (dst[1] as f32 * (1.0 - t) + src[1] as f32 * t) as u8,
                        (dst[2] as f32 * (1.0 - t) + src[2] as f32 * t) as u8,
                    ]
                } else {
                    src
                };
                out.put_pixel(ox as u32, oy as u32, image::Rgb(px));
            }
        }
        canvas = out;
    }
    canvas
}

/// A collage: a padded grid on a light background (`pad` px gaps), auto-columned.
pub fn collage(imgs: &[RgbImage], pad: u32) -> Result<RgbImage> {
    ensure!(!imgs.is_empty(), "no images for a collage");
    crate::imaging::grid::compose_images(imgs, None, pad)
}

/// A **mosaic / scrapbook** collage: a *justified-rows* layout (like a Flickr gallery). Images keep
/// their aspect ratios and are packed left→right into rows, each row then scaled so its images fill
/// the canvas width exactly — so cell sizes **vary** (a wide panorama gets a big cell, a portrait a
/// narrow one) instead of the uniform grid `collage` produces. `target_w` is the output width,
/// `row_h` the nominal row height (the packing target before justification), `pad` the gap. Fully
/// deterministic.
pub fn mosaic(imgs: &[RgbImage], target_w: u32, row_h: u32, pad: u32) -> Result<RgbImage> {
    ensure!(!imgs.is_empty(), "no images for a mosaic");
    let target_w = target_w.max(64);
    let row_h = row_h.clamp(48, target_w);
    let aspect = |im: &RgbImage| im.width() as f32 / im.height().max(1) as f32;

    // Greedily group images into rows: add until the aspect-scaled widths overflow the canvas.
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_w = 0f32;
    for (i, im) in imgs.iter().enumerate() {
        cur.push(i);
        cur_w += aspect(im) * row_h as f32 + pad as f32;
        if cur_w >= target_w as f32 {
            rows.push(std::mem::take(&mut cur));
            cur_w = 0.0;
        }
    }
    if !cur.is_empty() {
        rows.push(cur);
    }

    // Justify each row: pick the height so its scaled widths + gaps fill `target_w` exactly.
    let mut placements: Vec<(usize, u32, u32, u32, u32)> = Vec::new(); // (idx, x, y, w, h)
    let mut y = pad;
    for row in &rows {
        let sum_asp: f32 = row.iter().map(|&i| aspect(&imgs[i])).sum::<f32>().max(1e-3);
        let inner = (target_w as f32 - pad as f32 * (row.len() as f32 + 1.0)).max(1.0);
        let h = (inner / sum_asp).round().clamp(1.0, target_w as f32) as u32;
        let mut x = pad;
        for &i in row {
            let w = (aspect(&imgs[i]) * h as f32).round().max(1.0) as u32;
            placements.push((i, x, y, w, h));
            x += w + pad;
        }
        y += h + pad;
    }

    let mut canvas = RgbImage::from_pixel(target_w, y.max(1), image::Rgb([245, 245, 245]));
    for (i, x, py, w, h) in placements {
        let scaled = imageops::resize(&imgs[i], w, h, imageops::FilterType::Lanczos3);
        imageops::overlay(&mut canvas, &scaled, x as i64, py as i64);
    }
    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn solid(w: u32, h: u32, v: u8) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb([v, v, v]))
    }

    #[test]
    fn horizontal_sums_width_and_normalises_height() {
        let a = solid(40, 20, 10);
        let b = solid(30, 40, 200); // taller → scaled to common height 20 → width 15
        let out = panorama(&[a, b], PanoMode::Horizontal).unwrap();
        assert_eq!(out.height(), 20);
        assert_eq!(out.width(), 40 + 15);
    }

    #[test]
    fn vertical_sums_height() {
        let out = panorama(&[solid(20, 30, 10), solid(40, 30, 20)], PanoMode::Vertical).unwrap();
        // Common width = 20; second (40×30) → 20×15. Heights 30 + 15.
        assert_eq!(out.width(), 20);
        assert_eq!(out.height(), 30 + 15);
    }

    #[test]
    fn collage_and_grid_produce_a_canvas() {
        let imgs = [solid(20, 20, 10), solid(20, 20, 20), solid(20, 20, 30)];
        assert!(collage(&imgs, 6).unwrap().width() > 0);
        assert!(panorama(&imgs, PanoMode::Grid).unwrap().width() > 0);
    }

    #[test]
    fn aligned_panorama_recovers_a_known_overlap() {
        // A 160-wide textured source; two 80-wide windows offset by 50 → a 30 px (~37 %) overlap.
        // The aligner should register it and produce a canvas ~130 wide (not 160 = plain concat).
        let src = RgbImage::from_fn(160, 40, |x, y| {
            let v = (((x * 5) ^ (y * 3)) % 200 + 20) as u8;
            Rgb([v, v.wrapping_add(30), v.wrapping_add(60)])
        });
        let left = imageops::crop_imm(&src, 0, 0, 80, 40).to_image();
        let right = imageops::crop_imm(&src, 50, 0, 80, 40).to_image();
        let out = panorama(&[left, right], PanoMode::HorizontalAligned).unwrap();
        // Recovered overlap ≈ 30 → width ≈ 130; allow a few px of search slack.
        assert!((out.width() as i32 - 130).abs() <= 6, "aligned width {} ≈ 130", out.width());
        assert_eq!(out.height(), 40);
    }

    #[test]
    fn aligned_panorama_falls_back_for_unrelated_frames() {
        // Two unrelated flat frames don't register → edge-to-edge concatenation (full width).
        let out = panorama(&[solid(50, 30, 20), solid(50, 30, 210)], PanoMode::HorizontalAligned).unwrap();
        assert_eq!(out.width(), 100);
    }

    #[test]
    fn mosaic_fills_the_target_width_with_varied_cells() {
        // A wide, a tall, and a square image → a justified mosaic exactly `target_w` wide.
        let imgs = [solid(80, 20, 10), solid(20, 60, 20), solid(30, 30, 30)];
        let out = mosaic(&imgs, 400, 100, 8).unwrap();
        assert_eq!(out.width(), 400, "canvas is exactly the target width");
        assert!(out.height() > 0);
    }
}
