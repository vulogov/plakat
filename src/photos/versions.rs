//! Explicit per-image version snapshots (RFC PHOTOS-1). Frozen copies of an image's state kept in a
//! hidden `.plakat_versions/<stem>/` directory (skipped by the library walk), numbered `1.<ext>`,
//! `2.<ext>`, …. Save = copy the current file to the next number; restore = copy a number back over
//! the current file. Independent of the T1 edit log — a version is a full snapshot, not a diff.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// The version store directory for `filename` within `album`.
fn store(album: &Path, filename: &str) -> PathBuf {
    let stem = Path::new(filename).file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    album.join(".plakat_versions").join(stem)
}

fn ext_of(filename: &str) -> String {
    Path::new(filename).extension().and_then(|e| e.to_str()).unwrap_or("png").to_string()
}

/// Existing version numbers for `filename` in `album`, ascending.
pub fn list(album: &Path, filename: &str) -> Vec<u32> {
    let mut v: Vec<u32> = std::fs::read_dir(store(album, filename))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            e.path().file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<u32>().ok())
        })
        .collect();
    v.sort_unstable();
    v
}

/// Save the current file as the next version; returns the new version number.
pub fn snapshot(album: &Path, filename: &str) -> Result<u32> {
    let dir = store(album, filename);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let next = list(album, filename).last().map_or(1, |n| n + 1);
    std::fs::copy(album.join(filename), dir.join(format!("{next}.{}", ext_of(filename))))
        .with_context(|| format!("snapshotting {filename}"))?;
    Ok(next)
}

/// Path of version `n` for `filename` in `album`.
pub fn version_path(album: &Path, filename: &str, n: u32) -> PathBuf {
    store(album, filename).join(format!("{n}.{}", ext_of(filename)))
}

/// Restore version `n` over the current file.
pub fn restore(album: &Path, filename: &str, n: u32) -> Result<()> {
    let src = version_path(album, filename, n);
    anyhow::ensure!(src.exists(), "version {n} not found");
    std::fs::copy(&src, album.join(filename)).with_context(|| format!("restoring v{n} of {filename}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    #[test]
    fn snapshot_list_restore() {
        let dir = std::env::temp_dir().join(format!("plakat-ver-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = "photo.png";
        let save = |v: u8| DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([v, v, v]))).save(dir.join(name)).unwrap();

        save(10);
        assert_eq!(snapshot(&dir, name).unwrap(), 1); // v1 = grey 10
        save(20);
        assert_eq!(snapshot(&dir, name).unwrap(), 2); // v2 = grey 20
        assert_eq!(list(&dir, name), vec![1, 2]);

        // Change the live file, then restore v1 → back to grey 10.
        save(99);
        restore(&dir, name, 1).unwrap();
        let back = image::open(dir.join(name)).unwrap().to_rgb8();
        assert_eq!(back.get_pixel(0, 0)[0], 10);

        assert!(restore(&dir, name, 7).is_err()); // missing version
        let _ = std::fs::remove_dir_all(&dir);
    }
}
