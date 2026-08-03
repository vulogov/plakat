//! The coherent **kit** (RFC BOOKART-1 §10, flagship pt 1). A matched *set* of ornaments sharing one
//! origin/technique (one hand), one motif DNA, and one seed lineage — the ornament analog of persona's
//! cast reference set. This module holds the pure helpers (crop-to-content, contact sheet, coherence
//! math, per-ornament seed lineage); the CLI orchestrates generation + the CLIP style-coherence probe.

use image::{imageops, imageops::FilterType, Rgb, RgbImage, RgbaImage};

/// Deterministic per-ornament seed from the kit base seed + index (splitmix64 mix, so adjacent
/// ornaments don't share a near-identical noise field).
pub fn ornament_seed(base: u64, index: usize) -> u64 {
    let mut z = base.wrapping_add((index as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Crop a transparent page to its ink bounding box (so a small vignette and a full border are compared
/// on their *content*, not the mostly-empty page).
pub fn crop_to_content(page: &RgbaImage) -> RgbaImage {
    let (w, h) = (page.width(), page.height());
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if page.get_pixel(x, y).0[3] > 10 {
                any = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if !any {
        return page.clone();
    }
    imageops::crop_imm(page, x0, y0, x1 - x0 + 1, y1 - y0 + 1).to_image()
}

/// A square white cell with the ornament composited (ink over white), aspect-fit to 90% of the cell.
pub fn thumb_on_white(rgba: &RgbaImage, cell: u32) -> RgbImage {
    let (w, h) = (rgba.width() as f32, rgba.height() as f32);
    let s = (cell as f32 * 0.9) / w.max(h).max(1.0);
    let (tw, th) = ((w * s).max(1.0) as u32, (h * s).max(1.0) as u32);
    let small = imageops::resize(rgba, tw, th, FilterType::Lanczos3);
    let mut out = RgbImage::from_pixel(cell, cell, Rgb([255, 255, 255]));
    let (ox, oy) = (((cell - tw.min(cell)) / 2) as i64, ((cell - th.min(cell)) / 2) as i64);
    for (x, y, p) in small.enumerate_pixels() {
        let (dx, dy) = (ox + x as i64, oy + y as i64);
        if dx >= 0 && dy >= 0 && (dx as u32) < cell && (dy as u32) < cell {
            let a = p.0[3] as f32 / 255.0;
            let base = out.get_pixel(dx as u32, dy as u32).0;
            let mix = |c: u8, ink: u8| (ink as f32 * a + c as f32 * (1.0 - a)) as u8;
            out.put_pixel(dx as u32, dy as u32, Rgb([mix(base[0], p.0[0]), mix(base[1], p.0[1]), mix(base[2], p.0[2])]));
        }
    }
    out
}

/// Tile square cells into a padded grid contact sheet.
pub fn contact_sheet(cells: &[RgbImage], cols: u32) -> RgbImage {
    if cells.is_empty() {
        return RgbImage::from_pixel(1, 1, Rgb([255, 255, 255]));
    }
    let cell = cells[0].width();
    let cols = cols.max(1);
    let rows = (cells.len() as u32).div_ceil(cols);
    let pad = (cell / 20).max(4);
    let mut sheet = RgbImage::from_pixel(cols * cell + (cols + 1) * pad, rows * cell + (rows + 1) * pad, Rgb([248, 248, 248]));
    for (i, c) in cells.iter().enumerate() {
        let (col, row) = (i as u32 % cols, i as u32 / cols);
        imageops::overlay(&mut sheet, c, (pad + col * (cell + pad)) as i64, (pad + row * (cell + pad)) as i64);
    }
    sheet
}

/// Cosine of two L2-normalised embeddings.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Min + mean pairwise cosine across a set of embeddings — the kit style-coherence (min = the least
/// similar pair, the "does this read as one hand?" check; mean = overall consistency).
pub fn pairwise_min_mean(embs: &[Vec<f32>]) -> (f32, f32) {
    if embs.len() < 2 {
        return (1.0, 1.0);
    }
    let (mut min, mut sum, mut n) = (f32::INFINITY, 0.0f32, 0u32);
    for i in 0..embs.len() {
        for j in (i + 1)..embs.len() {
            let c = cosine(&embs[i], &embs[j]);
            min = min.min(c);
            sum += c;
            n += 1;
        }
    }
    (min, sum / n.max(1) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn seed_lineage_is_deterministic_and_distinct() {
        assert_eq!(ornament_seed(42, 0), ornament_seed(42, 0));
        assert_ne!(ornament_seed(42, 0), ornament_seed(42, 1));
    }

    #[test]
    fn crop_tightens_to_ink() {
        let mut page = RgbaImage::from_pixel(100, 100, Rgba([0, 0, 0, 0]));
        for y in 40..60 {
            for x in 30..50 {
                page.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        let c = crop_to_content(&page);
        assert_eq!(c.dimensions(), (20, 20));
    }

    #[test]
    fn contact_sheet_grid_dims() {
        let cells: Vec<RgbImage> = (0..5).map(|_| RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]))).collect();
        let sheet = contact_sheet(&cells, 3); // 5 cells, 3 cols → 2 rows
        let pad = (64u32 / 20).max(4);
        assert_eq!(sheet.width(), 3 * 64 + 4 * pad);
        assert_eq!(sheet.height(), 2 * 64 + 3 * pad);
    }

    #[test]
    fn coherence_of_identical_is_one() {
        let e = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        let (min, mean) = pairwise_min_mean(&e);
        assert!((min - 1.0).abs() < 1e-6 && (mean - 1.0).abs() < 1e-6);
    }
}
