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
use std::path::{Path, PathBuf};

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

    // Compute canvas dims in u64 and bail past a ceiling — a big grid of large cells can
    // exceed u32 (release wrap → a truncated/garbled canvas rather than an error).
    let total_w_u64 = cell_w as u64 * cols as u64 + padding as u64 * (cols as u64 + 1);
    let total_h_u64 = cell_h as u64 * rows as u64 + padding as u64 * (rows as u64 + 1);
    const MAX_GRID_DIM: u64 = 32_768;
    anyhow::ensure!(
        total_w_u64 <= MAX_GRID_DIM && total_h_u64 <= MAX_GRID_DIM,
        "grid canvas {total_w_u64}×{total_h_u64} exceeds the {MAX_GRID_DIM}px limit \
         (too many / too large cells)"
    );
    let (total_w, total_h) = (total_w_u64 as u32, total_h_u64 as u32);

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

/// Higher-level helper used by every CLI subcommand that supports
/// `--grid`. Scans `out_dir` for files matching
/// `{file_prefix}-{seed}.png` over `base_seed..base_seed+count`,
/// composes them, and writes `{file_prefix}-grid-{base_seed}.png`.
///
/// Returns `Ok(None)` when the grid isn't worth writing (count < 2
/// or fewer than 2 input files actually exist on disk). Returns
/// `Ok(Some((width, height, grid_path)))` on success so the caller
/// can log dimensions + the final path.
pub fn compose_grid_from_seed_range(
    out_dir: &Path,
    file_prefix: &str,
    base_seed: u64,
    count: u32,
    cols: Option<usize>,
    padding: u32,
) -> Result<Option<(u32, u32, PathBuf)>> {
    if count < 2 {
        return Ok(None);
    }
    let files: Vec<PathBuf> = (0..count)
        .map(|i| {
            let s = base_seed.wrapping_add(i as u64);
            out_dir.join(format!("{file_prefix}-{s}.png"))
        })
        .filter(|p| p.exists())
        .collect();
    if files.len() < 2 {
        return Ok(None);
    }
    let grid_path = out_dir.join(format!("{file_prefix}-grid-{base_seed}.png"));
    let (gw, gh) = write_grid(&files, &grid_path, cols, padding)?;
    Ok(Some((gw, gh, grid_path)))
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

    // v0.18 phase 2 — compose_grid_from_seed_range helper used by
    // the img2img / portrait / outpaint CLI dispatch arms.

    #[test]
    fn compose_grid_from_seed_range_writes_when_all_files_present() {
        let tmp = tempfile::tempdir().unwrap();
        // Mint three solid PNGs at the expected seed-derived paths.
        for (i, rgb) in [[255, 0, 0], [0, 255, 0], [0, 0, 255]].iter().enumerate() {
            let s = 100u64 + i as u64;
            let p = tmp.path().join(format!("plakat-img2img-{s}.png"));
            solid(4, 4, *rgb).save(&p).unwrap();
        }
        let result = compose_grid_from_seed_range(
            tmp.path(),
            "plakat-img2img",
            100,
            3,
            None,
            0,
        )
        .unwrap();
        let (w, h, path) = result.expect("grid composed");
        // default_columns(3) == 2 → 2×2 cells of 4px = 8×8.
        assert_eq!((w, h), (8, 8));
        assert_eq!(path, tmp.path().join("plakat-img2img-grid-100.png"));
        assert!(path.exists());
    }

    #[test]
    fn compose_grid_from_seed_range_skips_when_count_below_two() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("plakat-portrait-42.png");
        solid(4, 4, [128; 3]).save(&p).unwrap();
        let result = compose_grid_from_seed_range(
            tmp.path(),
            "plakat-portrait",
            42,
            1,
            None,
            0,
        )
        .unwrap();
        assert!(result.is_none(), "count=1 should skip");
    }

    #[test]
    fn compose_grid_from_seed_range_skips_when_fewer_than_two_files_exist() {
        let tmp = tempfile::tempdir().unwrap();
        // Only one of the three expected files actually exists.
        let p = tmp.path().join("plakat-flux-7.png");
        solid(4, 4, [255; 3]).save(&p).unwrap();
        let result = compose_grid_from_seed_range(
            tmp.path(),
            "plakat-flux",
            7,
            3,
            None,
            0,
        )
        .unwrap();
        assert!(result.is_none(), "single existing file should skip");
    }
}
