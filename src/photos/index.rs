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

/// Aggregate library statistics ([`LibraryIndex::stats`]).
#[derive(Default)]
pub struct IndexStats {
    pub total: usize,
    pub albums: usize,
    /// Count of records at each rating 0..=5.
    pub rating: [usize; 6],
    pub flagged: usize,
    pub rejected: usize,
    pub tagged: usize,
    pub with_gps: usize,
    pub avg_score: Option<f64>,
    /// Top cameras by frequency (up to 5).
    pub top_cameras: Vec<(String, usize)>,
    /// Capture-year histogram (ascending year).
    pub years: Vec<(String, usize)>,
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

    /// Incrementally refresh one album's entries in place from a just-saved `meta` + its known image
    /// file list — no directory scan. Records the album.hjson stamp as current (so the next sync won't
    /// re-read it for this edit) but **leaves the directory stamp** so a later file add/remove still
    /// re-syncs. Used after an in-app edit so the index reflects it immediately.
    pub fn update_album(&mut self, dir: &Path, image_files: &[PathBuf], meta: &super::hjson::AlbumMeta) {
        self.entries.retain(|e| e.album != dir);
        for p in image_files {
            let rec = p.file_name().and_then(|n| n.to_str()).and_then(|n| meta.images.get(n)).cloned();
            self.entries.push(Entry { path: p.clone(), album: dir.to_path_buf(), rec });
        }
        self.hjson_stamps.insert(dir.to_path_buf(), file_stamp(&dir.join(hjson::ALBUM_FILE)));
    }

    /// All entries as `(path, album, record)` — the shape `collect_library` returns.
    pub fn rows(&self) -> Vec<(PathBuf, PathBuf, Option<ImageRecord>)> {
        self.entries.iter().map(|e| (e.path.clone(), e.album.clone(), e.rec.clone())).collect()
    }

    /// Filter entries in place by `(filename, record) → keep`, cloning **only the matches** (a smart
    /// album matching 200 of 50k images clones 200 rows, not 50k). Returns the `collect_library` shape.
    pub fn filter(
        &self,
        pred: impl Fn(&str, Option<&ImageRecord>) -> bool,
    ) -> Vec<(PathBuf, PathBuf, Option<ImageRecord>)> {
        self.entries
            .iter()
            .filter_map(|e| {
                let fname = e.path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                pred(fname, e.rec.as_ref())
                    .then(|| (e.path.clone(), e.album.clone(), e.rec.clone()))
            })
            .collect()
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

    /// Aggregate library statistics computed in one pass over the index (fast at any scale).
    pub fn stats(&self) -> IndexStats {
        let mut s = IndexStats { total: self.entries.len(), albums: self.dir_stamps.len(), ..Default::default() };
        let mut cameras: HashMap<String, usize> = HashMap::new();
        let mut years: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        let mut score_sum = 0.0;
        let mut score_n = 0usize;
        for e in &self.entries {
            // An image with no record is simply unrated (rating 0), so the histogram sums to `total`.
            let Some(r) = &e.rec else {
                s.rating[0] += 1;
                continue;
            };
            s.rating[(r.rating.min(5)) as usize] += 1;
            if r.flagged {
                s.flagged += 1;
            }
            if r.rejected {
                s.rejected += 1;
            }
            if !r.tags.is_empty() {
                s.tagged += 1;
            }
            if let Some(sc) = r.score {
                score_sum += sc;
                score_n += 1;
            }
            if let Some(ex) = &r.exif {
                if ex.gps_lat.is_some() && ex.gps_lon.is_some() {
                    s.with_gps += 1;
                }
                if let Some(cam) = ex.camera_model.clone().or_else(|| ex.camera_make.clone()) {
                    *cameras.entry(cam).or_insert(0) += 1;
                }
                if let Some(d) = &ex.date_taken {
                    if d.len() >= 4 && d[..4].chars().all(|c| c.is_ascii_digit()) {
                        *years.entry(d[..4].to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
        s.avg_score = (score_n > 0).then(|| score_sum / score_n as f64);
        let mut cams: Vec<(String, usize)> = cameras.into_iter().collect();
        cams.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        cams.truncate(5);
        s.top_cameras = cams;
        s.years = years.into_iter().collect();
        s
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

    /// The CLIP vector store lives beside the JSON snapshot as a compact **binary** sidecar (768 f32
    /// per image would bloat JSON), so the whole library's embeddings load in one read at startup
    /// rather than walking every album's `.plakat_clip`.
    fn vec_path(root: &Path) -> PathBuf {
        Self::snapshot_path(root).with_extension("vec")
    }

    /// Persist the CLIP embedding store (path → (mtime, unit vector)) for `root`. Best-effort.
    pub fn save_vectors(root: &Path, vectors: &super::visual_search::Cache) {
        let path = Self::vec_path(root);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut buf = Vec::with_capacity(12 + vectors.len() * (768 + 24));
        buf.extend_from_slice(b"PKIVEC2\n"); // v2 = int8-quantized embeddings
        buf.extend_from_slice(&(vectors.len() as u32).to_le_bytes());
        for (p, (mt, e)) in vectors {
            let name = p.to_string_lossy();
            let nb = name.as_bytes();
            if nb.len() > u16::MAX as usize || e.q.len() > u16::MAX as usize {
                continue;
            }
            buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            buf.extend_from_slice(nb);
            buf.extend_from_slice(&mt.to_le_bytes());
            buf.extend_from_slice(&e.scale.to_le_bytes());
            buf.extend_from_slice(&(e.q.len() as u16).to_le_bytes());
            buf.extend(e.q.iter().map(|&b| b as u8));
        }
        let _ = std::fs::write(path, buf);
    }

    /// Load the CLIP embedding store for `root` (empty on absence / corruption — never fatal).
    pub fn load_vectors(root: &Path) -> super::visual_search::Cache {
        let mut out = super::visual_search::Cache::new();
        let Ok(b) = std::fs::read(Self::vec_path(root)) else { return out };
        if b.len() < 12 || &b[..8] != b"PKIVEC2\n" {
            return out;
        }
        let count = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
        let mut o = 12;
        for _ in 0..count {
            let get = |o: usize, n: usize| b.get(o..o + n);
            let Some(nlb) = get(o, 2) else { break };
            let nl = u16::from_le_bytes(nlb.try_into().unwrap()) as usize;
            o += 2;
            let Some(nameb) = get(o, nl) else { break };
            let Ok(name) = std::str::from_utf8(nameb) else { break };
            let name = name.to_string();
            o += nl;
            let Some(mtb) = get(o, 8) else { break };
            let mt = u64::from_le_bytes(mtb.try_into().unwrap());
            o += 8;
            let Some(sb) = get(o, 4) else { break };
            let scale = f32::from_le_bytes(sb.try_into().unwrap());
            o += 4;
            let Some(db) = get(o, 2) else { break };
            let dim = u16::from_le_bytes(db.try_into().unwrap()) as usize;
            o += 2;
            let Some(qb) = get(o, dim) else { break };
            let q: Vec<i8> = qb.iter().map(|&x| x as i8).collect();
            o += dim;
            out.insert(PathBuf::from(name), (mt, super::visual_search::Embedding { scale, q }));
        }
        out
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

        // filter() clones only the matches (here: the one 5-star record).
        let rated = idx.filter(|_, rec| rec.map(|r| r.rating).unwrap_or(0) >= 5);
        assert_eq!(rated.len(), 1);
        assert!(rated[0].0.ends_with("1.png"));

        // update_album: reflect an in-app edit (rate 2.png = 3) into the index without a re-scan,
        // and record the album.hjson stamp so a follow-up sync doesn't re-read it.
        let mut m2 = hjson::read_album(&a).unwrap_or_default();
        m2.images.insert("2.png".into(), hjson::ImageRecord { rating: 3, ..Default::default() });
        hjson::write_album(&a, &m2).unwrap();
        let files = vec![a.join("1.png"), a.join("2.png")];
        idx.update_album(&a, &files, &m2);
        let two = idx.entries.iter().find(|e| e.path.ends_with("2.png")).unwrap();
        assert_eq!(two.rec.as_ref().map(|r| r.rating), Some(3), "edit reflected immediately");
        assert!(!idx.sync(&[a.clone()]), "stamp bumped → no re-read needed");

        // stats(): 2 images in album a, one rated 5, one now rated 3.
        let st = idx.stats();
        assert_eq!(st.total, 2);
        assert_eq!(st.albums, 1);
        assert_eq!(st.rating[5], 1);
        assert_eq!(st.rating[3], 1);
        assert_eq!(st.rating[0], 0);

        // Snapshot round-trips.
        idx.save(&root);
        let loaded = LibraryIndex::load(&root);
        assert_eq!(loaded.entries.len(), 2);
        let _ = std::fs::remove_file(LibraryIndex::snapshot_path(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn vector_sidecar_roundtrips() {
        let root = std::env::temp_dir().join(format!("plakat-idxvec-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        use super::super::visual_search::quantize;
        let mut vecs = super::super::visual_search::Cache::new();
        vecs.insert(root.join("a/1.png"), (111, quantize(&vec![0.5_f32; 768])));
        vecs.insert(root.join("b/2.png"), (222, quantize(&vec![-0.25_f32; 768])));
        LibraryIndex::save_vectors(&root, &vecs);
        let back = LibraryIndex::load_vectors(&root);
        assert_eq!(back.len(), 2);
        let (mt, e) = back.get(&root.join("a/1.png")).expect("vector present");
        assert_eq!(*mt, 111);
        assert_eq!(e.q.len(), 768);
        assert!((e.scale * e.q[0] as f32 - 0.5).abs() < 0.02);
        // Corrupt file → empty, not a panic.
        std::fs::write(LibraryIndex::vec_path(&root), b"junk").unwrap();
        assert!(LibraryIndex::load_vectors(&root).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
