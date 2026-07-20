//! Derived library index (RFC PHOTOS 3.12 — "scale").
//!
//! `album.hjson` stays the authoritative, human-editable source of truth. This module builds a
//! **derived, rebuildable** index over the whole library — one entry per image (its path, source
//! album, and cached [`ImageRecord`]) — so smart albums and search run over an in-memory structure
//! instead of re-reading + re-parsing every `album.hjson` on each build. The index is:
//!
//! - **incremental** — [`LibraryIndex::sync`] re-reads only the albums whose `album.hjson` or
//!   directory changed (by `(mtime, len)` stamp); untouched albums are skipped, removed ones dropped;
//! - **persisted** — [`LibraryIndex::load`] / [`save`](LibraryIndex::save) snapshot to a serde-JSON
//!   file under the XDG cache (keyed by the library root), so a cold start skips the full walk;
//! - **non-authoritative** — delete the snapshot and it rebuilds from `album.hjson`; shared-volume
//!   writes are picked up via the stamps (nothing bypasses the three-way merge).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::hjson::{self, ImageRecord};

/// A cheap change stamp: `(mtime_secs, len)`.
type Stamp = (u64, u64);

/// One indexed image: its path, the album it lives in, and its curation record (absent → `None`).
#[derive(Clone, Serialize, Deserialize)]
pub struct Entry {
    pub path: PathBuf,
    pub album: PathBuf,
    pub rec: Option<ImageRecord>,
}

/// The derived index: all entries plus the per-album stamps used to sync incrementally.
#[derive(Default, Serialize, Deserialize)]
pub struct LibraryIndex {
    pub entries: Vec<Entry>,
    /// `album.hjson` stamp per album dir.
    hjson_stamps: HashMap<PathBuf, Stamp>,
    /// Directory stamp per album dir (catches images added/removed without an `album.hjson` change).
    dir_stamps: HashMap<PathBuf, Stamp>,
}

fn file_stamp(p: &Path) -> Stamp {
    std::fs::metadata(p)
        .ok()
        .map(|m| {
            let secs = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (secs, m.len())
        })
        .unwrap_or((0, 0))
}

impl LibraryIndex {
    /// Reconcile the index with the given album directories, re-reading only what changed. Returns
    /// `true` if anything was added, removed, or updated.
    pub fn sync(&mut self, dirs: &[PathBuf]) -> bool {
        let present: HashSet<&PathBuf> = dirs.iter().collect();
        let before = self.entries.len();
        self.entries.retain(|e| present.contains(&e.album));
        self.hjson_stamps.retain(|d, _| present.contains(d));
        self.dir_stamps.retain(|d, _| present.contains(d));
        let mut changed = self.entries.len() != before;

        for dir in dirs {
            let hs = file_stamp(&dir.join(hjson::ALBUM_FILE));
            let ds = file_stamp(dir);
            if self.hjson_stamps.get(dir) == Some(&hs) && self.dir_stamps.get(dir) == Some(&ds) {
                continue; // this album is unchanged since the last sync
            }
            // Rebuild just this album's entries.
            self.entries.retain(|e| &e.album != dir);
            let meta = hjson::read_album(dir).unwrap_or_default();
            if let Ok(rd) = std::fs::read_dir(dir) {
                let mut imgs: Vec<PathBuf> = rd
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && super::library::is_image(p))
                    .collect();
                imgs.sort();
                for p in imgs {
                    let rec = p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .and_then(|n| meta.images.get(n))
                        .cloned();
                    self.entries.push(Entry { path: p, album: dir.clone(), rec });
                }
            }
            self.hjson_stamps.insert(dir.clone(), hs);
            self.dir_stamps.insert(dir.clone(), ds);
            changed = true;
        }
        changed
    }

    /// All entries as `(path, album, record)` — the shape `collect_library` returns.
    pub fn rows(&self) -> Vec<(PathBuf, PathBuf, Option<ImageRecord>)> {
        self.entries.iter().map(|e| (e.path.clone(), e.album.clone(), e.rec.clone())).collect()
    }

    /// Direct image count per **synced** album directory (including 0 for a synced-but-empty album),
    /// for refreshing the tree badges from the index without re-scanning.
    pub fn counts(&self) -> HashMap<PathBuf, usize> {
        let mut m: HashMap<PathBuf, usize> = self.dir_stamps.keys().map(|d| (d.clone(), 0)).collect();
        for e in &self.entries {
            *m.entry(e.album.clone()).or_insert(0) += 1;
        }
        m
    }

    /// Snapshot path for a library root: `<cache>/plakat/photos/index/<sha256(root)>.json`.
    pub fn snapshot_path(root: &Path) -> PathBuf {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(root.to_string_lossy().as_bytes());
        let hex = format!("{:x}", h.finalize());
        super::loader::thumb_cache_dir()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(std::env::temp_dir)
            .join("index")
            .join(format!("{hex}.json"))
    }

    /// Load the persisted snapshot for `root` (empty index if none / unreadable).
    pub fn load(root: &Path) -> LibraryIndex {
        std::fs::read_to_string(Self::snapshot_path(root))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Persist the snapshot for `root` (best-effort).
    pub fn save(&self, root: &Path) {
        let path = Self::snapshot_path(root);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string(self) {
            let _ = std::fs::write(path, json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    fn img_at(p: &Path) {
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([1u8, 2, 3]))).save(p).unwrap();
    }

    #[test]
    fn syncs_incrementally_and_snapshots() {
        let root = std::env::temp_dir().join(format!("plakat-index-{}", std::process::id()));
        let a = root.join("a");
        let b = root.join("b");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        img_at(&a.join("1.png"));
        img_at(&a.join("2.png"));
        img_at(&b.join("3.png"));
        // Album a rates 1.png.
        let mut meta = hjson::AlbumMeta::default();
        meta.images.insert("1.png".into(), hjson::ImageRecord { rating: 5, ..Default::default() });
        hjson::write_album(&a, &meta).unwrap();

        let dirs = vec![a.clone(), b.clone()];
        let mut idx = LibraryIndex::default();
        assert!(idx.sync(&dirs), "first sync populates");
        assert_eq!(idx.entries.len(), 3);
        let rated = idx.entries.iter().find(|e| e.path.ends_with("1.png")).unwrap();
        assert_eq!(rated.rec.as_ref().map(|r| r.rating), Some(5));

        // A second sync with no changes is a no-op.
        assert!(!idx.sync(&dirs), "unchanged → no work");

        // Add an image to b → only b re-reads, entry count grows.
        img_at(&b.join("4.png"));
        assert!(idx.sync(&dirs));
        assert_eq!(idx.entries.len(), 4);

        // Remove album b from the set → its entries drop.
        assert!(idx.sync(&[a.clone()]));
        assert_eq!(idx.entries.len(), 2);
        assert!(idx.entries.iter().all(|e| e.album == a));

        // Per-album counts reflect the synced state (a is 2 after b was dropped).
        let counts = idx.counts();
        assert_eq!(counts.get(&a).copied(), Some(2));

        // Snapshot round-trips.
        idx.save(&root);
        let loaded = LibraryIndex::load(&root);
        assert_eq!(loaded.entries.len(), 2);
        let _ = std::fs::remove_file(LibraryIndex::snapshot_path(&root));
        let _ = std::fs::remove_dir_all(&root);
    }
}
