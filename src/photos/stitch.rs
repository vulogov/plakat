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
}

impl PanoMode {
    pub fn from_i32(v: i32) -> PanoMode {
        match v {
            1 => PanoMode::Vertical,
            2 => PanoMode::Grid,
            _ => PanoMode::Horizontal,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            PanoMode::Horizontal => "horizontal",
            PanoMode::Vertical => "vertical",
            PanoMode::Grid => "grid",
        }
    }
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
    }
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
    fn mosaic_fills_the_target_width_with_varied_cells() {
        // A wide, a tall, and a square image → a justified mosaic exactly `target_w` wide.
        let imgs = [solid(80, 20, 10), solid(20, 60, 20), solid(30, 30, 30)];
        let out = mosaic(&imgs, 400, 100, 8).unwrap();
        assert_eq!(out.width(), 400, "canvas is exactly the target width");
        assert!(out.height() > 0);
    }
}
