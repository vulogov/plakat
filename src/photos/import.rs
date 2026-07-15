//! `--import` — land generated images in a photo album (RFC PHOTOS-IMPORT).
//!
//! Copies (or moves) each output + its `.json` sidecar into the album, reads its
//! [`GenerationMetadata`] from the sidecar, and records it in `album.hjson` so the collection
//! manager shows the image already curated with its prompt / seed / params. Best-effort per file.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::hjson;
use crate::imaging::io::sidecar_path;
use crate::imaging::metadata::GenerationMetadata;

/// Import `files` into `album` (created if missing). Returns the count imported. `move_files` moves
/// instead of copying (leaving only the album copy).
pub fn import_outputs(album: &Path, files: &[PathBuf], move_files: bool) -> Result<usize> {
    if files.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(album).with_context(|| format!("creating album {}", album.display()))?;
    // Re-read immediately before writing so a concurrently-open `plakat photos` isn't clobbered.
    let mut meta = hjson::read_album(album)?;
    let mut n = 0usize;
    for src in files {
        match import_one(album, src, move_files, &mut meta) {
            Ok(()) => n += 1,
            Err(e) => {
                crate::ui::progress::println(&format!("  import skipped {}: {e:#}", src.display()));
            }
        }
    }
    hjson::write_album(album, &meta)?;
    Ok(n)
}

fn import_one(album: &Path, src: &Path, move_files: bool, meta: &mut hjson::AlbumMeta) -> Result<()> {
    anyhow::ensure!(src.is_file(), "not a file");
    // Read the generation params from the sidecar BEFORE moving the file.
    let gen_meta = read_gen_meta(src);

    let dest = dedup_dest(album, src);
    transfer(src, &dest, move_files)?;
    // Carry the `.json` sidecar alongside (best-effort).
    let (side_src, side_dst) = (sidecar_path(src), sidecar_path(&dest));
    if side_src.exists() {
        let _ = transfer(&side_src, &side_dst, move_files);
    }

    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .context("output has no filename")?
        .to_string();
    let rec = meta.images.entry(name).or_default();
    if let Some(g) = &gen_meta {
        rec.score = rec.score.or(g.score); // carry the aesthetic score if the sidecar had one
    }
    rec.generation = gen_meta;
    Ok(())
}

/// Read `GenerationMetadata` from a PNG's `.json` sidecar (the structured form plakat writes).
fn read_gen_meta(png: &Path) -> Option<GenerationMetadata> {
    let text = std::fs::read_to_string(sidecar_path(png)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Destination path in `album` for `src`, suffixing `-2`, `-3`, … on a name collision.
fn dedup_dest(album: &Path, src: &Path) -> PathBuf {
    let file = src.file_name().and_then(|n| n.to_str()).unwrap_or("image.png");
    let candidate = album.join(file);
    if !candidate.exists() {
        return candidate;
    }
    let stem = Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let ext = Path::new(file).extension().and_then(|e| e.to_str()).unwrap_or("png");
    for i in 2..10_000 {
        let c = album.join(format!("{stem}-{i}.{ext}"));
        if !c.exists() {
            return c;
        }
    }
    album.join(format!("{stem}-dup.{ext}"))
}

/// Copy or move `src` → `dest` (move falls back to copy+remove across filesystems).
fn transfer(src: &Path, dest: &Path, move_files: bool) -> Result<()> {
    if move_files {
        if std::fs::rename(src, dest).is_ok() {
            return Ok(());
        }
        std::fs::copy(src, dest).with_context(|| format!("copying {}", src.display()))?;
        let _ = std::fs::remove_file(src);
    } else {
        std::fs::copy(src, dest).with_context(|| format!("copying {}", src.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    #[test]
    fn imports_with_gen_params_and_dedups() {
        let base = std::env::temp_dir().join(format!("plakat-import-{}", std::process::id()));
        let out = base.join("out");
        let album = base.join("album");
        std::fs::create_dir_all(&out).unwrap();
        let png = out.join("plakat-7.png");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([1, 2, 3]))).save(&png).unwrap();
        // A sidecar with gen params + score.
        let meta = GenerationMetadata { score: Some(6.5), ..GenerationMetadata::new("p", "sdxl", 42, 20, 7.0, "ddim", 512, 512) };
        std::fs::write(sidecar_path(&png), meta.to_json_pretty().unwrap()).unwrap();

        let n = import_outputs(&album, &[png.clone()], false).unwrap();
        assert_eq!(n, 1);
        let am = hjson::read_album(&album).unwrap();
        let rec = am.images.get("plakat-7.png").expect("record");
        assert_eq!(rec.generation.as_ref().unwrap().seed, 42);
        assert_eq!(rec.score, Some(6.5));
        assert!(album.join("plakat-7.png").exists());
        assert!(album.join("plakat-7.png.json").exists()); // sidecar carried

        // Second import of the same name dedups.
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(8, 8, Rgb([4, 5, 6]))).save(&png).unwrap();
        import_outputs(&album, &[png], false).unwrap();
        assert!(album.join("plakat-7-2.png").exists());

        let _ = std::fs::remove_dir_all(&base);
    }
}
