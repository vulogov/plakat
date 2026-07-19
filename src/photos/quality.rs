//! Non-AI image-quality scoring for offline auto-culling: a **sharpness** score (variance of the
//! Laplacian — blurry frames score low) and a **brightness** mean (to flag badly under/over-exposed
//! shots). Both are cheap and model-free — the offline complement to the LAION aesthetic cull.

use image::DynamicImage;

/// Sharpness = variance of the Laplacian on a fixed 256×256 grayscale (higher = sharper). The fixed
/// size makes the score comparable across images regardless of their resolution.
pub fn sharpness(img: &DynamicImage) -> f32 {
    let g = img.resize_exact(256, 256, image::imageops::FilterType::Triangle).to_luma8();
    let (w, h) = (256i32, 256i32);
    let at = |x: i32, y: i32| g.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32)[0] as f32;
    let n = (w * h) as f32;
    let mut sum = 0f32;
    let mut sum2 = 0f32;
    for y in 0..h {
        for x in 0..w {
            let l = 4.0 * at(x, y) - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1);
            sum += l;
            sum2 += l * l;
        }
    }
    let mean = sum / n;
    sum2 / n - mean * mean
}

/// Mean brightness (0..255) on a downscale — for the exposure check.
pub fn brightness(img: &DynamicImage) -> f32 {
    let g = img.resize_exact(64, 64, image::imageops::FilterType::Triangle).to_luma8();
    g.pixels().map(|p| p[0] as f32).sum::<f32>() / (64.0 * 64.0)
}

/// Why a frame was culled (for the status summary).
#[derive(Debug, PartialEq, Eq)]
pub enum Cull {
    Keep,
    Soft,       // too blurry
    Underexposed,
    Overexposed,
}

/// Judge one image against the sharpness floor + exposure bounds.
pub fn judge(sharp: f32, bright: f32, min_sharp: f32) -> Cull {
    if bright < 22.0 {
        Cull::Underexposed
    } else if bright > 236.0 {
        Cull::Overexposed
    } else if sharp < min_sharp {
        Cull::Soft
    } else {
        Cull::Keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn sharp_edges_score_higher_than_a_blur() {
        let sharp = DynamicImage::ImageRgb8(ImageBuffer::from_fn(128, 128, |x, _| {
            let v = if (x / 4) % 2 == 0 { 0 } else { 255 };
            Rgb([v, v, v])
        }));
        let flat = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(128, 128, Rgb([128, 128, 128])));
        assert!(sharpness(&sharp) > sharpness(&flat) + 100.0);
        assert!(sharpness(&flat) < 1.0, "a flat image has ~zero Laplacian variance");
    }

    #[test]
    fn judge_flags_exposure_and_blur() {
        assert_eq!(judge(500.0, 128.0, 40.0), Cull::Keep);
        assert_eq!(judge(5.0, 128.0, 40.0), Cull::Soft);
        assert_eq!(judge(500.0, 10.0, 40.0), Cull::Underexposed);
        assert_eq!(judge(500.0, 250.0, 40.0), Cull::Overexposed);
    }
}
