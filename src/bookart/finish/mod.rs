//! The B/W finisher (RFC BOOKART-1 §7). The stage between render and output: a raw grayscale/colour
//! render → technique **binarisation** (§7.1) → **transparency** (§7.2) → a transparent RGBA ornament.
//! Pure and deterministic (matte transparency is the one exception — it needs U2Net and is wired at
//! render time; here it falls back to `luminance`). Page-sizing onto the print canvas is B2.

pub mod alpha;
pub mod binarize;
pub mod canvas;
pub mod vector;

use crate::bookart::compile::RenderPlan;
use image::{GrayImage, Luma, RgbImage, RgbaImage};

/// Rec.601 luma.
pub fn to_luma(img: &RgbImage) -> GrayImage {
    let mut g = GrayImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        let l = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        g.put_pixel(x, y, Luma([l.round() as u8]));
    }
    g
}

/// Finish a raw render into a transparent ornament, per the resolved plan: `binarise → transparency`.
pub fn finish_ornament(raw: &RgbImage, plan: &RenderPlan) -> RgbaImage {
    let g = to_luma(raw);
    let b = binarize::binarise(&g, &plan.binariser, plan.ink_weight);
    let tint = alpha::parse_tint(&plan.tint);
    alpha::to_transparent(&b, &plan.transparency_mode, tint, plan.fade)
}

/// Finish an already-clean **procedural** line-art grayscale: skip binarisation (it's born 1-bit-clean)
/// and go straight to transparency + tint, preserving the antialiased strokes.
pub fn finish_procedural(gray: &GrayImage, plan: &RenderPlan) -> RgbaImage {
    let tint = alpha::parse_tint(&plan.tint);
    alpha::to_transparent(gray, &plan.transparency_mode, tint, plan.fade)
}

pub use alpha::{luminance_alpha, parse_tint, to_transparent};
pub use binarize::{binarise, otsu};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bookart::{compile::resolve, BookArtSpec};

    #[test]
    fn finish_yields_transparent_ornament() {
        // a raw render: warm-tinted "black" ink line on an off-white page
        let mut raw = RgbImage::from_pixel(32, 32, image::Rgb([248, 246, 240]));
        for y in 0..32 {
            raw.put_pixel(16, y, image::Rgb([40, 30, 22])); // a warm-black vertical line
        }
        let plan = resolve(&BookArtSpec::default()); // divider → xdog + luminance + black tint
        let rgba = finish_ornament(&raw, &plan);
        assert_eq!(rgba.dimensions(), (32, 32));
        // paper is transparent; the ink column has opaque, neutral-black pixels.
        assert_eq!(rgba.get_pixel(0, 0).0[3], 0, "paper transparent");
        assert!(rgba.pixels().any(|p| p.0[3] > 200 && p.0[0] < 20), "no opaque neutral ink");
    }
}
