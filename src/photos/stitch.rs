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
}
