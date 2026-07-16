//! Export selected images out of the library (RFC PHOTOS-1 Phase 5).
//!
//! Copies files into a destination directory, optionally downscaling so the longer side is ≤ a max
//! dimension (handy for sharing / web). Non-destructive: the album copies stay put. Name collisions
//! in the destination are suffixed `-2`, `-3`, …

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Export `files` into `dest` (created if missing). When `max_px` is set, each image is decoded and
/// downscaled so its longer side is ≤ `max_px` (images already smaller are copied verbatim);
/// otherwise the raw file is copied. Returns the number exported. Best-effort per file.
pub fn export(files: &[PathBuf], dest: &Path, max_px: Option<u32>) -> Result<usize> {
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    let mut n = 0;
    for f in files {
        match export_one(f, dest, max_px) {
            Ok(()) => n += 1,
            Err(e) => crate::ui::progress::println(&format!("  export skipped {}: {e:#}", f.display())),
        }
    }
    Ok(n)
}

fn export_one(src: &Path, dest: &Path, max_px: Option<u32>) -> Result<()> {
    let out = dedup_name(dest, src);
    match max_px {
        Some(px) => {
            let img = image::open(src).with_context(|| format!("reading {}", src.display()))?;
            let scaled = if img.width().max(img.height()) > px {
                img.resize(px, px, image::imageops::FilterType::Lanczos3)
            } else {
                img
            };
            scaled.save(&out).with_context(|| format!("writing {}", out.display()))?;
        }
        None => {
            std::fs::copy(src, &out).with_context(|| format!("copying {}", src.display()))?;
        }
    }
    Ok(())
}

/// `<dest>/<filename>`, suffixing `-2`, `-3`, … on a collision.
fn dedup_name(dest: &Path, src: &Path) -> PathBuf {
    let file = src.file_name().and_then(|n| n.to_str()).unwrap_or("image.png");
    let cand = dest.join(file);
    if !cand.exists() {
        return cand;
    }
    let stem = Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let ext = Path::new(file).extension().and_then(|e| e.to_str()).unwrap_or("png");
    for i in 2..10_000 {
        let c = dest.join(format!("{stem}-{i}.{ext}"));
        if !c.exists() {
            return c;
        }
    }
    dest.join(format!("{stem}-dup.{ext}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    #[test]
    fn copies_and_downscales_and_dedups() {
        let base = std::env::temp_dir().join(format!("plakat-export-{}", std::process::id()));
        let src = base.join("src");
        let dst = base.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        let big = src.join("photo.png");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(200, 100, Rgb([1, 2, 3]))).save(&big).unwrap();

        // Downscale to longer-side ≤ 50.
        assert_eq!(export(&[big.clone()], &dst, Some(50)).unwrap(), 1);
        let out = image::open(dst.join("photo.png")).unwrap();
        assert!(out.width().max(out.height()) <= 50, "downscaled to {}x{}", out.width(), out.height());

        // A second export of the same name dedups.
        export(&[big], &dst, None).unwrap();
        assert!(dst.join("photo-2.png").exists());
        let _ = std::fs::remove_dir_all(&base);
    }
}
