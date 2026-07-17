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
    /// Mean R, G, B (0..255) — the colour balance / cast.
    pub channel_mean: [f32; 3],
    /// Fraction of pixels in shadows / midtones / highlights (luma < 85 / 85–170 / ≥ 170) — the
    /// lighting balance / tonal distribution.
    pub zones: [f32; 3],
    /// Per-channel histograms (R, G, B), `BINS` buckets each.
    pub hist_rgb: [[u32; BINS]; 3],
    /// The most common colours (representative RGB), most-frequent first — a rough palette.
    pub dominant: Vec<[u8; 3]>,
    /// Luma waveform: `[row][col]` counts, row 0 = brightest, `col` spans image width.
    pub waveform: [[u16; WCOLS]; WROWS],
    /// RGB parade: a waveform per channel (R, G, B).
    pub parade: [[[u16; WCOLS]; WROWS]; 3],
}

/// Waveform-scope resolution (columns across the frame × tonal rows).
pub const WCOLS: usize = 30;
pub const WROWS: usize = 6;

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
    let mut chan = [0.0f64; 3];
    let mut zones = [0u64; 3]; // shadows / mids / highlights
    let mut hist_rgb = [[0u32; BINS]; 3];
    let mut waveform = [[0u16; WCOLS]; WROWS];
    let mut parade = [[[0u16; WCOLS]; WROWS]; 3];
    // Dominant-colour buckets: 4 bits/channel (12-bit key) → count + colour sum, for a rough palette.
    let mut buckets: std::collections::HashMap<u16, (u32, [u64; 3])> = std::collections::HashMap::new();
    // Precompute a luma plane for the Laplacian pass.
    let mut lum = vec![0f32; (w * h) as usize];
    for (i, px) in rgb.pixels().enumerate() {
        let y = luma(px[0], px[1], px[2]);
        lum[i] = y;
        sum += y as f64;
        for c in 0..3 {
            chan[c] += px[c] as f64;
            hist_rgb[c][(px[c] as usize) * BINS / 256] += 1;
        }
        let yi = y as u8;
        hist[(yi as usize) * BINS / 256] += 1;
        if yi >= 250 {
            clip_high += 1;
        }
        if yi <= 5 {
            clip_low += 1;
        }
        zones[if yi < 85 { 0 } else if yi < 170 { 1 } else { 2 }] += 1;
        let key = (((px[0] >> 4) as u16) << 8) | (((px[1] >> 4) as u16) << 4) | (px[2] >> 4) as u16;
        let e = buckets.entry(key).or_insert((0, [0; 3]));
        e.0 += 1;
        for c in 0..3 {
            e.1[c] += px[c] as u64;
        }
        // Waveform / parade: column = x across the frame, row = tonal band (row 0 = brightest).
        let wc = (i % (w as usize)) * WCOLS / (w as usize).max(1);
        let wr = |v: u8| (255 - v as usize) * WROWS / 256;
        waveform[wr(yi)][wc] = waveform[wr(yi)][wc].saturating_add(1);
        for c in 0..3 {
            parade[c][wr(px[c])][wc] = parade[c][wr(px[c])][wc].saturating_add(1);
        }
    }
    // Top colours: most-populated buckets, averaged to a representative RGB.
    let mut ranked: Vec<(u32, [u64; 3])> = buckets.into_values().collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0));
    let dominant: Vec<[u8; 3]> = ranked
        .iter()
        .take(6)
        .map(|(cnt, s)| {
            let c = (*cnt).max(1) as u64;
            [(s[0] / c) as u8, (s[1] / c) as u8, (s[2] / c) as u8]
        })
        .collect();
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
        channel_mean: [chan[0] as f32 / n, chan[1] as f32 / n, chan[2] as f32 / n],
        zones: [zones[0] as f32 / n, zones[1] as f32 / n, zones[2] as f32 / n],
        hist_rgb,
        dominant,
        waveform,
        parade,
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
    fn channel_stats_and_dominant_colour() {
        // A solid teal image → channel means match, one dominant colour close to it.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(32, 32, Rgb([20, 160, 160])));
        let a = analyze(&img);
        assert!((a.channel_mean[0] - 20.0).abs() < 1.0);
        assert!((a.channel_mean[1] - 160.0).abs() < 1.0);
        assert!(!a.dominant.is_empty());
        let d = a.dominant[0];
        assert!(d[1] > 140 && d[2] > 140 && d[0] < 40, "dominant ≈ teal, got {d:?}");
        // Per-channel histograms each hold every pixel.
        for c in 0..3 {
            assert_eq!(a.hist_rgb[c].iter().sum::<u32>(), 32 * 32);
        }
        // Waveform + each parade channel account for every pixel.
        assert_eq!(a.waveform.iter().flatten().map(|&v| v as u32).sum::<u32>(), 32 * 32);
        for c in 0..3 {
            assert_eq!(a.parade[c].iter().flatten().map(|&v| v as u32).sum::<u32>(), 32 * 32);
        }
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
