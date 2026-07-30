//! Tier-1 preview rendering (RFC §17.5): the CPU-cheap geometry wireframe rasterised to **braille**
//! glyphs so it draws in any terminal — "a strictly better degradation story than a placeholder box,
//! and it means the entire structural + detail half of the interview is usable over a plain SSH
//! session." Pure + deterministic (a function of the landmarks); no GPU, no graphics protocol.

use crate::persona::geometry::{self, Template};

/// The 8 braille dot bits for a 2×4 cell, indexed `[col][row]` (col 0..2, row 0..4). Base `U+2800`.
const DOT: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];

/// Render a grayscale image to braille text: `cols × rows` character cells, each covering a 2×4 pixel
/// block (so the sampled grid is `2·cols × 4·rows`). A pixel lights its dot when above `threshold`.
pub fn gray_to_braille(img: &image::GrayImage, cols: u32, rows: u32, threshold: u8) -> Vec<String> {
    let (iw, ih) = (img.width().max(1), img.height().max(1));
    let (gw, gh) = (cols * 2, rows * 4);
    let mut out = Vec::with_capacity(rows as usize);
    for cy in 0..rows {
        let mut line = String::with_capacity(cols as usize);
        for cx in 0..cols {
            let mut mask = 0u32;
            for (dc, dcol) in DOT.iter().enumerate() {
                for (dr, &bit) in dcol.iter().enumerate() {
                    // sample the source pixel this dot maps to.
                    let gx = cx * 2 + dc as u32;
                    let gy = cy * 4 + dr as u32;
                    let sx = (gx * iw / gw).min(iw - 1);
                    let sy = (gy * ih / gh).min(ih - 1);
                    if img.get_pixel(sx, sy).0[0] > threshold {
                        mask |= bit;
                    }
                }
            }
            line.push(char::from_u32(0x2800 + mask).unwrap_or(' '));
        }
        out.push(line);
    }
    out
}

/// Render a resolved landmark set to a braille wireframe preview (`cols × rows` cells). Draws the
/// geometry wireframe at a working resolution then maps it to braille.
pub fn wireframe_braille(lm: &Template, cols: u32, rows: u32) -> Vec<String> {
    // work at a resolution matched to the braille grid (2×4 px per cell), square-ish for the face.
    let px = (rows * 4).max(cols * 2).max(32);
    let rgb = geometry::wireframe(lm, px);
    let gray = image::DynamicImage::ImageRgb8(rgb).to_luma8();
    gray_to_braille(&gray, cols, rows, 40)
}

/// Convenience: the mean-template wireframe preview (the interview's starting face).
pub fn mean_wireframe_braille(cols: u32, rows: u32) -> Vec<String> {
    wireframe_braille(&geometry::mean_template(false), cols, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_to_braille_dimensions_and_glyphs() {
        // a fully-lit image → every cell is the full braille block (U+28FF).
        let img = image::GrayImage::from_pixel(16, 16, image::Luma([255]));
        let out = gray_to_braille(&img, 4, 2, 40);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chars().count(), 4);
        assert!(out[0].chars().all(|c| c == '\u{28FF}'), "all dots set");
        // a fully-black image → blank braille (U+2800).
        let black = image::GrayImage::from_pixel(16, 16, image::Luma([0]));
        let ob = gray_to_braille(&black, 4, 2, 40);
        assert!(ob[0].chars().all(|c| c == '\u{2800}'));
    }

    #[test]
    fn wireframe_braille_is_nonempty_and_stable() {
        let a = mean_wireframe_braille(30, 16);
        assert_eq!(a.len(), 16);
        // the face draws *something* (not all-blank rows).
        let lit: usize = a.iter().map(|l| l.chars().filter(|&c| c != '\u{2800}').count()).sum();
        assert!(lit > 20, "wireframe should light a good number of dots, got {lit}");
        // deterministic.
        assert_eq!(a, mean_wireframe_braille(30, 16));
    }
}
