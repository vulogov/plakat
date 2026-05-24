//! Compose N PNGs into a single grid PNG. Used by
//! `plakat generate --grid` to bundle a `--count N` sweep into
//! one shareable image alongside the individual files.
//!
//! Grid layout: `cols` columns and `ceil(N / cols)` rows. The
//! default `cols` is `ceil(sqrt(N))` — produces near-square
//! output for typical small Ns (4 → 2×2, 6 → 3×2, 9 → 3×3,
//! 16 → 4×4). All input images are assumed to share dimensions;
//! the grid uses the dims of the first input. Optional padding
//! between cells.

use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;

/// Pick a sensible column count for `n` cells. `ceil(sqrt(n))`
/// gives near-square layouts: 4→2, 5..9→3, 10..16→4, etc.
pub fn default_columns(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    (n as f64).sqrt().ceil() as usize
}

/// Compose `inputs` into one RGB image. `cols` controls the
/// column count; pass `None` to use [`default_columns`]. `padding`
/// is the gap (in pixels) between cells — `0` for flush. The
/// padding colour matches the conventional A1111 grid: pure white
/// (255, 255, 255).
pub fn compose_paths(
    inputs: &[std::path::PathBuf],
    cols: Option<usize>,
    padding: u32,
) -> Result<RgbImage> {
    if inputs.is_empty() {
        anyhow::bail!("compose_paths: no input images");
    }
    // Load every input as RGB8.
    let images: Vec<RgbImage> = inputs
        .iter()
        .map(|p| {
            image::open(p)
                .with_context(|| format!("opening {}", p.display()))
                .map(|img| img.to_rgb8())
        })
        .collect::<Result<Vec<_>>>()?;
    compose_images(&images, cols, padding)
}

/// Same as [`compose_paths`] but accepts already-loaded RGB
/// buffers. Useful for in-process composition where reads are
/// avoidable.
pub fn compose_images(
    images: &[RgbImage],
    cols: Option<usize>,
    padding: u32,
) -> Result<RgbImage> {
    if images.is_empty() {
        anyhow::bail!("compose_images: empty input");
    }
    let cell_w = images[0].width();
    let cell_h = images[0].height();
    // Sanity: warn (not bail) if dims differ — the artefact-blend
    // pass can change widths by ±1px in some edge cases. Use the
    // first image's dims as canonical and Lanczos-resize stragglers
    // on the fly so the grid stays aligned.
    let cols = cols.unwrap_or_else(|| default_columns(images.len()));
    let cols = cols.max(1);
    let rows = images.len().div_ceil(cols);

    let total_w = (cell_w * cols as u32) + padding * (cols as u32 + 1);
    let total_h = (cell_h * rows as u32) + padding * (rows as u32 + 1);

    let mut canvas: RgbImage =
        ImageBuffer::from_pixel(total_w, total_h, Rgb([255u8, 255, 255]));

    for (idx, img) in images.iter().enumerate() {
        let r = idx / cols;
        let c = idx % cols;
        let x = padding + c as u32 * (cell_w + padding);
        let y = padding + r as u32 * (cell_h + padding);
        let drawn: RgbImage = if img.width() == cell_w && img.height() == cell_h {
            img.clone()
        } else {
            image::imageops::resize(img, cell_w, cell_h, image::imageops::FilterType::Lanczos3)
        };
        image::imageops::overlay(&mut canvas, &drawn, x as i64, y as i64);
    }
    Ok(canvas)
}

/// Convenience: compose + save to `out_path` in one call.
pub fn write_grid(
    inputs: &[std::path::PathBuf],
    out_path: &Path,
    cols: Option<usize>,
    padding: u32,
) -> Result<(u32, u32)> {
    let grid = compose_paths(inputs, cols, padding)?;
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    grid.save(out_path)
        .with_context(|| format!("writing grid PNG to {}", out_path.display()))?;
    Ok((grid.width(), grid.height()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_columns_picks_near_square() {
        assert_eq!(default_columns(1), 1);
        assert_eq!(default_columns(2), 2);
        assert_eq!(default_columns(4), 2);
        assert_eq!(default_columns(6), 3);
        assert_eq!(default_columns(9), 3);
        assert_eq!(default_columns(10), 4);
        assert_eq!(default_columns(16), 4);
        assert_eq!(default_columns(25), 5);
    }

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        ImageBuffer::from_pixel(w, h, Rgb(rgb))
    }

    #[test]
    fn compose_4_images_2x2_no_padding() {
        let images = vec![
            solid(2, 2, [255, 0, 0]),
            solid(2, 2, [0, 255, 0]),
            solid(2, 2, [0, 0, 255]),
            solid(2, 2, [255, 255, 0]),
        ];
        let g = compose_images(&images, None, 0).unwrap();
        assert_eq!(g.dimensions(), (4, 4));
        // top-left = red, top-right = green, bottom-left = blue,
        // bottom-right = yellow.
        assert_eq!(g.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(g.get_pixel(3, 0).0, [0, 255, 0]);
        assert_eq!(g.get_pixel(0, 3).0, [0, 0, 255]);
        assert_eq!(g.get_pixel(3, 3).0, [255, 255, 0]);
    }

    #[test]
    fn compose_with_padding_inserts_white_border() {
        let images = vec![
            solid(2, 2, [255, 0, 0]),
            solid(2, 2, [0, 255, 0]),
            solid(2, 2, [0, 0, 255]),
            solid(2, 2, [255, 255, 0]),
        ];
        let g = compose_images(&images, Some(2), 1).unwrap();
        // 2 cells of width 2 + 3 padding strips of width 1 = 7.
        assert_eq!(g.dimensions(), (7, 7));
        // Corners are padding (white).
        assert_eq!(g.get_pixel(0, 0).0, [255, 255, 255]);
        // First cell top-left after the leading 1px padding.
        assert_eq!(g.get_pixel(1, 1).0, [255, 0, 0]);
    }

    #[test]
    fn compose_5_images_picks_3_wide_2_tall() {
        let images: Vec<_> = (0..5).map(|i| solid(2, 2, [i * 50, 0, 0])).collect();
        let g = compose_images(&images, None, 0).unwrap();
        // default_columns(5) == 3 → 3 cols × 2 rows × 2px cells = 6×4.
        assert_eq!(g.dimensions(), (6, 4));
        // Last cell (idx=4) at row=1, col=1 → top-left of cell at (2,2).
        assert_eq!(g.get_pixel(2, 2).0, [200, 0, 0]);
    }

    #[test]
    fn compose_with_explicit_cols_3() {
        let images: Vec<_> = (0..6).map(|_| solid(4, 4, [128, 128, 128])).collect();
        let g = compose_images(&images, Some(3), 0).unwrap();
        // 3 cols, 6 / 3 = 2 rows → 12×8.
        assert_eq!(g.dimensions(), (12, 8));
    }

    #[test]
    fn compose_resizes_mismatched_input() {
        // First image is 4×4, second is 2×2 — the second should be
        // upscaled to 4×4 to fit the grid cell.
        let images = vec![solid(4, 4, [255, 0, 0]), solid(2, 2, [0, 255, 0])];
        let g = compose_images(&images, None, 0).unwrap();
        // default_columns(2) == 2 → 2 cols × 1 row × 4px = 8×4.
        assert_eq!(g.dimensions(), (8, 4));
        // Second cell now 4×4 of green.
        assert_eq!(g.get_pixel(4, 0).0, [0, 255, 0]);
        assert_eq!(g.get_pixel(7, 3).0, [0, 255, 0]);
    }

    #[test]
    fn compose_empty_bails() {
        let err = compose_images(&[], None, 0).unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[test]
    fn write_grid_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        // Save two solid PNGs first.
        let a = tmp.path().join("a.png");
        let b = tmp.path().join("b.png");
        solid(2, 2, [255, 0, 0]).save(&a).unwrap();
        solid(2, 2, [0, 0, 255]).save(&b).unwrap();
        let out = tmp.path().join("grid.png");
        let (w, h) = write_grid(&[a, b], &out, None, 0).unwrap();
        assert_eq!((w, h), (4, 2));
        // Round-trip read.
        let g = image::open(&out).unwrap().to_rgb8();
        assert_eq!(g.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(g.get_pixel(2, 0).0, [0, 0, 255]);
    }
}
