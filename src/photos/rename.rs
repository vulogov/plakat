//! Batch rename planning (RFC PHOTOS-1 Phase 5). Pure: turns a pattern + a list of files into the
//! new filenames, so it's fully testable without touching disk. The actual on-disk rename + record
//! migration lives in the parent module (album-local, two-phase to avoid intra-set collisions).
//!
//! Pattern grammar: a run of `#` is replaced by the 1-based sequence number, zero-padded to the run
//! length (`trip_###` → `trip_001`, `trip_002`, …). With no `#`, `-N` is appended (padded to the
//! digit-width of the count). The original file's extension is always preserved.

use std::path::{Path, PathBuf};

use super::{edit, hjson};

/// Execute a rename `plan` (`old_path → new_filename`) in `dir`. Two-phase (→ hidden temp → final)
/// so an intra-set name swap (`a→b` while `b→c`) can't clobber; each image's `album.hjson` record
/// and its pristine edit backup migrate under the *actual* final name (deduped vs unrelated existing
/// files). Returns the count renamed. Album-local — pure of any TUI state, so it's directly testable.
pub fn apply(dir: &Path, plan: Vec<(PathBuf, String)>, meta: &mut hjson::AlbumMeta) -> usize {
    // Phase 1: everything to a unique hidden temp (dotfile → invisible to the walk / listing).
    let mut staged: Vec<(PathBuf, String, String)> = Vec::new();
    for (i, (old, new_name)) in plan.into_iter().enumerate() {
        let Some(old_name) = old.file_name().and_then(|n| n.to_str()).map(String::from) else {
            continue;
        };
        let tmp = dir.join(format!(".plakat_rn_{i}"));
        if std::fs::rename(&old, &tmp).is_ok() {
            staged.push((tmp, old_name, new_name));
        }
    }
    // Phase 2: temp → final (deduped), migrating the record + edit backup under the final name.
    let mut n = 0;
    for (tmp, old_name, want) in staged {
        let dest = dedup_in_dir(dir, &want);
        if std::fs::rename(&tmp, &dest).is_err() {
            continue;
        }
        let final_name = dest.file_name().and_then(|f| f.to_str()).unwrap_or(&want).to_string();
        if let Some(rec) = meta.images.remove(&old_name) {
            meta.images.insert(final_name.clone(), rec);
        }
        let (bak_old, bak_new) = (edit::backup_path(dir, &old_name), edit::backup_path(dir, &final_name));
        if bak_old.exists() {
            let _ = std::fs::rename(&bak_old, &bak_new);
        }
        n += 1;
    }
    n
}

/// `<dir>/<name>`, suffixing `-2`, `-3`, … before the extension on a collision.
fn dedup_in_dir(dir: &Path, name: &str) -> PathBuf {
    let cand = dir.join(name);
    if !cand.exists() {
        return cand;
    }
    let stem = Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let ext = Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("png");
    for i in 2..10_000 {
        let c = dir.join(format!("{stem}-{i}.{ext}"));
        if !c.exists() {
            return c;
        }
    }
    dir.join(format!("{stem}-dup.{ext}"))
}

/// First run of `#` in `pattern` as `(start, len)`, if any.
fn hash_run(pattern: &str) -> Option<(usize, usize)> {
    let bytes = pattern.as_bytes();
    let start = bytes.iter().position(|&b| b == b'#')?;
    let len = bytes[start..].iter().take_while(|&&b| b == b'#').count();
    Some((start, len))
}

/// Map each file to its new filename (extension preserved). 1-based numbering in input order.
pub fn plan(files: &[PathBuf], pattern: &str) -> Vec<(PathBuf, String)> {
    let width_default = files.len().to_string().len().max(1);
    let run = hash_run(pattern);
    files
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let n = i + 1;
            let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("");
            let stem = match run {
                Some((start, len)) => {
                    let num = format!("{n:0len$}");
                    format!("{}{}{}", &pattern[..start], num, &pattern[start + len..])
                }
                None => format!("{pattern}-{n:0width_default$}"),
            };
            let name = if ext.is_empty() { stem } else { format!("{stem}.{ext}") };
            (f.clone(), name)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> Vec<PathBuf> {
        vec![PathBuf::from("a.jpg"), PathBuf::from("b.png"), PathBuf::from("c.JPG")]
    }

    #[test]
    fn hash_run_padded_and_ext_preserved() {
        let out = plan(&files(), "trip_##");
        let names: Vec<&str> = out.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["trip_01.jpg", "trip_02.png", "trip_03.JPG"]);
    }

    #[test]
    fn no_hash_appends_padded_number() {
        // 3 files → width 1.
        let out = plan(&files(), "photo");
        assert_eq!(out[0].1, "photo-1.jpg");
        assert_eq!(out[2].1, "photo-3.JPG");
    }

    #[test]
    fn embedded_hash_run_keeps_prefix_and_suffix() {
        let out = plan(&[PathBuf::from("x.webp")], "IMG_###_final");
        assert_eq!(out[0].1, "IMG_001_final.webp");
    }

    #[test]
    fn apply_moves_files_records_and_backups() {
        use image::{DynamicImage, ImageBuffer, Rgb};
        let base = std::env::temp_dir().join(format!("plakat-rn-apply-{}", std::process::id()));
        let dir = base.join("Album");
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["a.png", "b.png"] {
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([1, 2, 3])))
                .save(dir.join(name))
                .unwrap();
        }
        // a.png has a rating record and a pristine edit backup; b.png has neither.
        let mut meta = hjson::AlbumMeta::default();
        meta.images.insert("a.png".into(), hjson::ImageRecord { rating: 5, ..Default::default() });
        edit::ensure_backup(&dir, "a.png").unwrap();

        let files = vec![dir.join("a.png"), dir.join("b.png")];
        let n = apply(&dir, plan(&files, "trip_##"), &mut meta);

        assert_eq!(n, 2);
        assert!(dir.join("trip_01.png").exists() && dir.join("trip_02.png").exists());
        assert!(!dir.join("a.png").exists() && !dir.join("b.png").exists());
        // record migrated a.png → trip_01.png (input order preserved)
        assert_eq!(meta.images.get("trip_01.png").map(|r| r.rating), Some(5));
        assert!(!meta.images.contains_key("a.png"));
        // edit backup migrated too
        assert!(edit::backup_path(&dir, "trip_01.png").exists());
        assert!(!edit::backup_path(&dir, "a.png").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn apply_dedups_against_an_unrelated_existing_file() {
        use image::{DynamicImage, ImageBuffer, Rgb};
        let base = std::env::temp_dir().join(format!("plakat-rn-dedup-{}", std::process::id()));
        let dir = base.join("Album");
        std::fs::create_dir_all(&dir).unwrap();
        let save = |name: &str| {
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([1, 2, 3])))
                .save(dir.join(name))
                .unwrap();
        };
        save("src.png");
        save("out.png"); // an unrelated file that the target name collides with
        let mut meta = hjson::AlbumMeta::default();
        let n = apply(&dir, vec![(dir.join("src.png"), "out.png".into())], &mut meta);
        assert_eq!(n, 1);
        assert!(dir.join("out.png").exists()); // untouched
        assert!(dir.join("out-2.png").exists()); // renamed src → deduped
        let _ = std::fs::remove_dir_all(&base);
    }
}
