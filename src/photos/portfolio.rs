//! Watermark + portfolio export (RFC PHOTOS-1 Phase 5). A *portfolio* is a shareable bundle: copies
//! of the selected images — optionally down-sized and text-watermarked — plus a contact-sheet grid.
//! Create-only, like [`super::export`]; reuses the built-in bitmap font ([`crate::map::labels`]) and
//! the grid composer ([`crate::imaging::grid`]).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::RgbImage;

/// Stamp a haloed text watermark into the bottom-right corner of `img` (white text, dark halo, sized
/// proportionally so it reads on any image).
pub fn watermark(img: &mut RgbImage, text: &str) {
    if text.is_empty() {
        return;
    }
    let scale = (img.width().max(img.height()) / 90).max(2);
    let (tw, th) = (crate::map::labels::text_width(text, scale), crate::map::labels::text_height(scale));
    let margin = scale * 6;
    let x = img.width().saturating_sub(tw + margin) as i32;
    let y = img.height().saturating_sub(th + margin) as i32;
    crate::map::labels::draw_text_haloed(img, x, y, text, scale, [255, 255, 255], [0, 0, 0]);
}

/// Export `files` as a portfolio into `dest`: each image is copied (down-sized to `max_px` longer
/// side if set, watermarked with `mark` if set), then a `_contact_sheet.png` grid is written.
/// Returns the number of images exported. Best-effort per file.
pub fn export(files: &[PathBuf], dest: &Path, mark: Option<&str>, max_px: Option<u32>) -> Result<usize> {
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    let mut written: Vec<PathBuf> = Vec::new();
    for f in files {
        match one(f, dest, mark, max_px) {
            Ok(p) => written.push(p),
            Err(e) => crate::ui::progress::println(&format!("  portfolio skipped {}: {e:#}", f.display())),
        }
    }
    if !written.is_empty() {
        let sheet = dest.join("_contact_sheet.png");
        let _ = crate::imaging::grid::write_grid(&written, &sheet, None, 4);
    }
    Ok(written.len())
}

fn one(src: &Path, dest: &Path, mark: Option<&str>, max_px: Option<u32>) -> Result<PathBuf> {
    let img = image::open(src).with_context(|| format!("reading {}", src.display()))?;
    let img = match max_px {
        Some(px) if img.width().max(img.height()) > px => {
            img.resize(px, px, image::imageops::FilterType::Lanczos3)
        }
        _ => img,
    };
    let mut rgb = img.to_rgb8();
    if let Some(t) = mark {
        watermark(&mut rgb, t);
    }
    let out = dedup_name(dest, src);
    rgb.save(&out).with_context(|| format!("writing {}", out.display()))?;
    Ok(out)
}

/// `<dest>/<stem>.png` (portfolio copies are re-encoded PNG), suffixing `-2`, `-3`, … on collision.
fn dedup_name(dest: &Path, src: &Path) -> PathBuf {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let cand = dest.join(format!("{stem}.png"));
    if !cand.exists() {
        return cand;
    }
    for i in 2..10_000 {
        let c = dest.join(format!("{stem}-{i}.png"));
        if !c.exists() {
            return c;
        }
    }
    dest.join(format!("{stem}-dup.png"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    #[test]
    fn watermark_stamps_pixels_without_panicking() {
        let mut img = ImageBuffer::from_pixel(400, 300, Rgb([20, 20, 20]));
        watermark(&mut img, "© PLAKAT");
        // Some pixels in the bottom-right region are now brighter than the flat background.
        let brightened = (250..300).any(|y| (300..400).any(|x| img.get_pixel(x, y)[0] > 100));
        assert!(brightened, "watermark drew visible text");
        // Empty text is a no-op.
        let mut flat = ImageBuffer::from_pixel(64, 64, Rgb([20, 20, 20]));
        watermark(&mut flat, "");
        assert!(flat.pixels().all(|p| p[0] == 20));
    }

    #[test]
    fn export_writes_copies_and_a_contact_sheet() {
        let base = std::env::temp_dir().join(format!("plakat-portfolio-{}", std::process::id()));
        let src = base.join("src");
        let dst = base.join("out");
        std::fs::create_dir_all(&src).unwrap();
        let files: Vec<PathBuf> = (0..3)
            .map(|i| {
                let p = src.join(format!("img{i}.png"));
                DynamicImage::ImageRgb8(ImageBuffer::from_pixel(200, 150, Rgb([i * 40, 10, 10]))).save(&p).unwrap();
                p
            })
            .collect();
        let n = export(&files, &dst, Some("© test"), Some(100)).unwrap();
        assert_eq!(n, 3);
        assert!(dst.join("img0.png").exists());
        assert!(dst.join("_contact_sheet.png").exists());
        // Down-sized to ≤100.
        let out = image::open(dst.join("img0.png")).unwrap();
        assert!(out.width().max(out.height()) <= 100);
        let _ = std::fs::remove_dir_all(&base);
    }
}
