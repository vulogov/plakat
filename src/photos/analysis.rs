//! View analysis (RFC PHOTOS-1 Phase 6): a luma histogram + exposure and focus stats for the image
//! on screen. Pure functions over a decoded (thumbnail-sized) image, so they're fast and testable;
//! the parent module renders the results as a panel in the image view (`H`).

use image::DynamicImage;

/// Number of histogram buckets (fits a terminal panel).
pub const BINS: usize = 64;

/// Computed analysis for one image.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// Luma histogram, `BINS` buckets over 0..=255.
    pub hist: [u32; BINS],
    pub width: u32,
    pub height: u32,
    /// Mean luma 0..255.
    pub mean: f32,
    /// Fraction of pixels that are blown highlights (luma ≥ 250).
    pub clip_high: f32,
    /// Fraction of pixels that are crushed shadows (luma ≤ 5).
    pub clip_low: f32,
    /// Focus/sharpness measure — variance of the Laplacian (higher = sharper / better focus).
    pub sharpness: f32,
}

/// Rec. 601 luma of an 8-bit RGB pixel.
fn luma(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Analyze `img` (any size — pass a thumbnail for speed): histogram + exposure + focus.
pub fn analyze(img: &DynamicImage) -> Analysis {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let mut hist = [0u32; BINS];
    let mut sum = 0.0f64;
    let mut clip_high = 0u64;
    let mut clip_low = 0u64;
    // Precompute a luma plane for the Laplacian pass.
    let mut lum = vec![0f32; (w * h) as usize];
    for (i, px) in rgb.pixels().enumerate() {
        let y = luma(px[0], px[1], px[2]);
        lum[i] = y;
        sum += y as f64;
        let yi = y as u8;
        hist[(yi as usize) * BINS / 256] += 1;
        if yi >= 250 {
            clip_high += 1;
        }
        if yi <= 5 {
            clip_low += 1;
        }
    }
    let n = (w * h).max(1) as f32;
    let sharpness = laplacian_variance(&lum, w, h);
    Analysis {
        hist,
        width: w,
        height: h,
        mean: (sum as f32) / n,
        clip_high: clip_high as f32 / n,
        clip_low: clip_low as f32 / n,
        sharpness,
    }
}

/// Variance of the 4-neighbour Laplacian over the luma plane — a standard focus measure.
fn laplacian_variance(lum: &[f32], w: u32, h: u32) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let (w, h) = (w as usize, h as usize);
    let at = |x: usize, y: usize| lum[y * w + x];
    let mut vals: Vec<f32> = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            vals.push(lap);
        }
    }
    let m = vals.iter().sum::<f32>() / vals.len() as f32;
    vals.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / vals.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn flat(v: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(32, 32, Rgb([v, v, v])))
    }

    #[test]
    fn flat_image_has_no_sharpness_and_one_hist_spike() {
        let a = analyze(&flat(128));
        assert!(a.sharpness < 1e-3, "flat image is not sharp: {}", a.sharpness);
        assert!((a.mean - 128.0).abs() < 1.0);
        // All pixels land in one bucket.
        assert_eq!(a.hist.iter().filter(|&&c| c > 0).count(), 1);
        assert_eq!(a.clip_high, 0.0);
        assert_eq!(a.clip_low, 0.0);
    }

    #[test]
    fn clipping_is_detected() {
        assert_eq!(analyze(&flat(255)).clip_high, 1.0);
        assert_eq!(analyze(&flat(0)).clip_low, 1.0);
    }

    #[test]
    fn edges_raise_sharpness() {
        // A vertical black/white split has strong edges → high Laplacian variance.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(32, 32, |x, _| {
            if x < 16 { Rgb([0, 0, 0]) } else { Rgb([255, 255, 255]) }
        }));
        assert!(analyze(&img).sharpness > analyze(&flat(128)).sharpness + 100.0);
    }
}
