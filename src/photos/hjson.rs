//! Library storage model (RFC PHOTOS-1 §4): `folder.hjson` / `album.hjson`.
//!
//! Per-album HJSON is the manager's data layer — there is no separate index. Records are sparse and
//! all fields default, so old files parse and new fields are additive. Writes are atomic
//! (`.album.hjson.tmp` → rename).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const ALBUM_FILE: &str = "album.hjson";
pub const FOLDER_FILE: &str = "folder.hjson";

/// A saved filter query (root `folder.hjson`) — a library-wide "smart album" (RFC §8): its `query`
/// is evaluated across every album on open, collecting the matching images into one grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartAlbum {
    pub name: String,
    pub query: String,
}

/// A named, reusable edit stack (a saved sequence of pixel edits) — applied to any image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditPreset {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ops: Vec<EditEntry>,
}

/// `folder.hjson` — display name + persisted UI state for a non-leaf directory.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FolderMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Persisted expand state (sub-dir names that are open in the tree).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expanded: Vec<String>,
    /// Thumbnail worker count (root only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thumb_workers: Option<usize>,
    /// Library-wide saved searches (root folder only). Shown as ★ entries at the top of the tree.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub smart_albums: Vec<SmartAlbum>,
    /// Reusable edit presets (root folder only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edit_presets: Vec<EditPreset>,
}

/// EXIF read once on first scan (RFC §4.3). All optional — cameras vary.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ExifRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_taken: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_make: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focal_length_mm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aperture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shutter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iso: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gps_lat: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gps_lon: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width_px: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height_px: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<u16>,
}

/// One append-only edit-log entry (op + free-form params + timestamp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditEntry {
    pub op: String,
    #[serde(default, flatten)]
    pub params: HashMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ts: Option<String>,
}

/// One composite layer (Phase 8): an image overlaid on the base with position/scale/opacity/blend.
/// `src` is relative to the album when the source lives under it, else an absolute path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerEntry {
    pub src: String,
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    #[serde(default = "one_f32")]
    pub scale: f32,
    #[serde(default = "one_f32")]
    pub opacity: f32,
    #[serde(default = "normal_blend")]
    pub blend: String,
    /// Optional per-layer mask (shape region or image matte).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<MaskEntry>,
}

/// A per-layer mask: a feathered shape region (`rect`/`ellipse`) or an image matte (`image`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskEntry {
    /// `rect` | `ellipse` | `image`.
    pub kind: String,
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    #[serde(default = "one_f32")]
    pub w: f32,
    #[serde(default = "one_f32")]
    pub h: f32,
    #[serde(default)]
    pub feather: f32,
    #[serde(default)]
    pub invert: bool,
    /// Matte source (relative to the album or absolute) for `kind = "image"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub src: String,
}

fn one_f32() -> f32 {
    1.0
}
fn normal_blend() -> String {
    "normal".to_string()
}

/// Per-image record. Key in `AlbumMeta.images` = filename (no path). Sparse — only set fields
/// are stored.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exif: Option<ExifRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// IPTC-style creator / rights (edited in the manager; stored non-destructively).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copyright: Option<String>,
    /// 0–5 (0 = unrated).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub rating: u8,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// red|yellow|green|blue|purple.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub flagged: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub rejected: bool,
    /// Derivative variants (`_upscale.png`, `_analyze_gen.png`, …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<String>,
    /// LAION aesthetic score (carried over from the gen sidecar / `rank`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Generation parameters when this image was produced by plakat (`--import`): prompt, model,
    /// seed, steps, guidance, loras, … (the standard `GenerationMetadata`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<crate::imaging::metadata::GenerationMetadata>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edits: Vec<EditEntry>,
    /// Composite layer stack (Phase 8): images overlaid on the base, flattened into a `_layered.png`
    /// variant on demand. Non-destructive — the stack stays editable across sessions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<LayerEntry>,
}

/// `album.hjson` — album metadata + sparse per-image records.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AlbumMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Album-level tags (distinct from per-image tags).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Cover image filename (null → first alphabetical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<String>,
    /// date-asc|date-desc|name-asc|name-desc|manual.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thumb_size: Option<u32>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub images: HashMap<String, ImageRecord>,
}

fn is_zero(v: &u8) -> bool {
    *v == 0
}
fn is_false(v: &bool) -> bool {
    !*v
}

/// Read `<dir>/album.hjson` (empty default if absent).
pub fn read_album(dir: &Path) -> Result<AlbumMeta> {
    let p = dir.join(ALBUM_FILE);
    if !p.exists() {
        return Ok(AlbumMeta::default());
    }
    let text = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    deser_hjson::from_str(&text).with_context(|| format!("parsing {}", p.display()))
}

/// Read `<dir>/folder.hjson` (empty default if absent).
pub fn read_folder(dir: &Path) -> Result<FolderMeta> {
    let p = dir.join(FOLDER_FILE);
    if !p.exists() {
        return Ok(FolderMeta::default());
    }
    let text = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    deser_hjson::from_str(&text).with_context(|| format!("parsing {}", p.display()))
}

/// Atomically write `meta` to `<dir>/<file>` via a `.tmp` sibling + rename.
fn write_atomic<T: Serialize>(dir: &Path, file: &str, meta: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(meta).context("serialising HJSON")?;
    let tmp = dir.join(format!(".{file}.tmp"));
    std::fs::write(&tmp, json).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, dir.join(file)).with_context(|| format!("renaming into {}", dir.display()))?;
    Ok(())
}

pub fn write_album(dir: &Path, meta: &AlbumMeta) -> Result<()> {
    write_atomic(dir, ALBUM_FILE, meta)
}

// ---- Concurrency-safe (shared-volume) merge -----------------------------------------------------

/// Two records (or their absence) differ. Compared structurally via JSON so we don't need `PartialEq`
/// on every nested type (e.g. the external `GenerationMetadata`).
fn record_changed(a: Option<&ImageRecord>, b: Option<&ImageRecord>) -> bool {
    let to_val = |r: Option<&ImageRecord>| r.map(|x| serde_json::to_value(x).unwrap_or(serde_json::Value::Null));
    to_val(a) != to_val(b)
}

/// Three-way merge of album metadata for safe concurrent access on a **shared volume** (Dropbox / NFS),
/// where cross-machine file locks are unreliable. `baseline` is what this instance originally loaded,
/// `ours` is its current in-memory copy, `disk` is the latest on-disk copy (possibly changed by another
/// instance). The result keeps **our** value for every record/field we actually changed (vs baseline)
/// and the **disk** value for everything we didn't touch — so a concurrent instance editing *other*
/// images is never clobbered. Same-record conflicts resolve last-writer-wins (ours, since we're saving).
pub fn merge_album(baseline: &AlbumMeta, ours: &AlbumMeta, disk: &AlbumMeta) -> AlbumMeta {
    let mut merged = disk.clone();

    // Album-level scalar fields: take ours only where we changed them relative to baseline.
    if ours.name != baseline.name {
        merged.name = ours.name.clone();
    }
    if ours.description != baseline.description {
        merged.description = ours.description.clone();
    }
    if ours.tags != baseline.tags {
        merged.tags = ours.tags.clone();
    }
    if ours.cover != baseline.cover {
        merged.cover = ours.cover.clone();
    }
    if ours.sort != baseline.sort {
        merged.sort = ours.sort.clone();
    }
    if ours.thumb_size != baseline.thumb_size {
        merged.thumb_size = ours.thumb_size.clone();
    }

    // Per-image records: for every key we know about, apply our change (add / update / delete) if we
    // made one; otherwise leave whatever disk has (records only on disk stay via the clone above).
    let keys: HashSet<&String> = ours.images.keys().chain(baseline.images.keys()).collect();
    for k in keys {
        let ours_r = ours.images.get(k);
        if record_changed(ours_r, baseline.images.get(k)) {
            match ours_r {
                Some(r) => {
                    merged.images.insert(k.clone(), r.clone());
                }
                None => {
                    merged.images.remove(k); // we deleted it → respect that
                }
            }
        }
    }
    merged
}

/// Concurrency-safe write of the open album: re-read the current on-disk copy, [`merge_album`] our
/// changes onto it, then write atomically. Returns the merged metadata (the caller adopts it as its
/// new in-memory + baseline state so subsequent diffs are relative to it).
pub fn write_album_merged(dir: &Path, baseline: &AlbumMeta, ours: &AlbumMeta) -> Result<AlbumMeta> {
    let disk = read_album(dir).unwrap_or_default();
    let merged = merge_album(baseline, ours, &disk);
    write_album(dir, &merged)?;
    Ok(merged)
}

pub fn write_folder(dir: &Path, meta: &FolderMeta) -> Result<()> {
    write_atomic(dir, FOLDER_FILE, meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn album_roundtrips_atomically() {
        let dir = std::env::temp_dir().join(format!("plakat-photos-hjson-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut meta = AlbumMeta { name: Some("Iceland".into()), ..Default::default() };
        let mut rec = ImageRecord { rating: 5, flagged: true, ..Default::default() };
        rec.tags = vec!["waterfall".into()];
        rec.score = Some(6.2);
        meta.images.insert("IMG_1.jpg".into(), rec);
        write_album(&dir, &meta).unwrap();

        let back = read_album(&dir).unwrap();
        assert_eq!(back.name.as_deref(), Some("Iceland"));
        let r = &back.images["IMG_1.jpg"];
        assert_eq!(r.rating, 5);
        assert!(r.flagged);
        assert_eq!(r.tags, vec!["waterfall".to_string()]);
        assert_eq!(r.score, Some(6.2));
        // No leftover .tmp.
        assert!(!dir.join(".album.hjson.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn full_image_record_roundtrips() {
        // Lock the whole record shape — every field a curated + generated + edited image accrues.
        let dir = std::env::temp_dir().join(format!("plakat-fullrec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut rec = ImageRecord {
            title: Some("Fox".into()),
            rating: 4,
            tags: vec!["wildlife".into(), "winter".into()],
            color_label: Some("green".into()),
            caption: Some("a red fox in snow".into()),
            notes: Some("golden hour".into()),
            flagged: true,
            score: Some(6.8),
            variants: vec!["fox_upscale.png".into()],
            generation: Some(crate::imaging::metadata::GenerationMetadata::new(
                "a red fox", "sdxl", 42, 20, 7.0, "ddim", 1024, 1024,
            )),
            ..Default::default()
        };
        rec.edits.push(EditEntry { op: "rotate_cw".into(), params: Default::default(), ts: None });
        let mut m = AlbumMeta::default();
        m.images.insert("fox.png".into(), rec);
        write_album(&dir, &m).unwrap();

        let back = read_album(&dir).unwrap();
        let r = &back.images["fox.png"];
        assert_eq!(r.title.as_deref(), Some("Fox"));
        assert_eq!(r.rating, 4);
        assert_eq!(r.tags, vec!["wildlife".to_string(), "winter".into()]);
        assert_eq!(r.color_label.as_deref(), Some("green"));
        assert_eq!(r.caption.as_deref(), Some("a red fox in snow"));
        assert_eq!(r.notes.as_deref(), Some("golden hour"));
        assert!(r.flagged);
        assert_eq!(r.score, Some(6.8));
        assert_eq!(r.variants, vec!["fox_upscale.png".to_string()]);
        assert_eq!(r.generation.as_ref().map(|g| g.seed), Some(42));
        assert_eq!(r.edits.len(), 1);
        assert_eq!(r.edits[0].op, "rotate_cw");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn rec(rating: u8, tags: &[&str]) -> ImageRecord {
        ImageRecord { rating, tags: tags.iter().map(|s| s.to_string()).collect(), ..Default::default() }
    }

    #[test]
    fn merge_keeps_concurrent_edits_to_different_images() {
        // Baseline: two images, unrated. This instance rated A; another instance (disk) tagged B.
        let mut baseline = AlbumMeta::default();
        baseline.images.insert("a.jpg".into(), rec(0, &[]));
        baseline.images.insert("b.jpg".into(), rec(0, &[]));

        let mut ours = baseline.clone();
        ours.images.insert("a.jpg".into(), rec(5, &[])); // we rated A

        let mut disk = baseline.clone();
        disk.images.insert("b.jpg".into(), rec(0, &["keeper"])); // other instance tagged B

        let merged = merge_album(&baseline, &ours, &disk);
        assert_eq!(merged.images["a.jpg"].rating, 5, "our rating on A survives");
        assert_eq!(merged.images["b.jpg"].tags, vec!["keeper".to_string()], "other instance's tag on B survives");
    }

    #[test]
    fn merge_respects_additions_deletions_and_conflicts() {
        let mut baseline = AlbumMeta::default();
        baseline.images.insert("keep.jpg".into(), rec(0, &[]));
        baseline.images.insert("gone.jpg".into(), rec(3, &[]));
        baseline.images.insert("conflict.jpg".into(), rec(1, &[]));

        // Ours: added new.jpg, deleted gone.jpg, bumped conflict.jpg to 4.
        let mut ours = baseline.clone();
        ours.images.insert("new.jpg".into(), rec(2, &[]));
        ours.images.remove("gone.jpg");
        ours.images.insert("conflict.jpg".into(), rec(4, &[]));

        // Disk (another instance): added other.jpg, set conflict.jpg to 5.
        let mut disk = baseline.clone();
        disk.images.insert("other.jpg".into(), rec(1, &[]));
        disk.images.insert("conflict.jpg".into(), rec(5, &[]));

        let merged = merge_album(&baseline, &ours, &disk);
        assert_eq!(merged.images["new.jpg"].rating, 2, "our addition present");
        assert_eq!(merged.images["other.jpg"].rating, 1, "their addition present");
        assert!(!merged.images.contains_key("gone.jpg"), "our deletion respected");
        assert_eq!(merged.images["conflict.jpg"].rating, 4, "same-record conflict → ours (we're saving)");
        assert!(merged.images.contains_key("keep.jpg"), "untouched record kept");
    }

    #[test]
    fn merge_album_level_fields_and_untouched_defer_to_disk() {
        let baseline = AlbumMeta { name: Some("Trip".into()), sort: Some("name-asc".into()), ..Default::default() };
        // We changed the sort; another instance renamed the album on disk.
        let ours = AlbumMeta { name: Some("Trip".into()), sort: Some("date-desc".into()), ..Default::default() };
        let disk = AlbumMeta { name: Some("Iceland Trip".into()), sort: Some("name-asc".into()), ..Default::default() };
        let merged = merge_album(&baseline, &ours, &disk);
        assert_eq!(merged.sort.as_deref(), Some("date-desc"), "our sort change wins");
        assert_eq!(merged.name.as_deref(), Some("Iceland Trip"), "their rename (we didn't touch name) survives");
    }

    #[test]
    fn write_album_merged_persists_the_merge() {
        let dir = std::env::temp_dir().join(format!("plakat-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Baseline on disk.
        let mut baseline = AlbumMeta::default();
        baseline.images.insert("a.jpg".into(), rec(0, &[]));
        baseline.images.insert("b.jpg".into(), rec(0, &[]));
        write_album(&dir, &baseline).unwrap();
        // Another instance tags B on disk while we hold the baseline.
        let mut disk = baseline.clone();
        disk.images.insert("b.jpg".into(), rec(0, &["keeper"]));
        write_album(&dir, &disk).unwrap();
        // We rated A and save via the merge.
        let mut ours = baseline.clone();
        ours.images.insert("a.jpg".into(), rec(5, &[]));
        let merged = write_album_merged(&dir, &baseline, &ours).unwrap();
        assert_eq!(merged.images["a.jpg"].rating, 5);
        assert_eq!(merged.images["b.jpg"].tags, vec!["keeper".to_string()]);
        // And it's on disk.
        let back = read_album(&dir).unwrap();
        assert_eq!(back.images["a.jpg"].rating, 5);
        assert_eq!(back.images["b.jpg"].tags, vec!["keeper".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn folder_smart_albums_roundtrip() {
        let dir = std::env::temp_dir().join(format!("plakat-photos-folder-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fm = FolderMeta {
            smart_albums: vec![
                SmartAlbum { name: "Keepers".into(), query: "rating>=4 -rejected".into() },
                SmartAlbum { name: "AI".into(), query: "ai".into() },
            ],
            ..Default::default()
        };
        write_folder(&dir, &fm).unwrap();
        let back = read_folder(&dir).unwrap();
        assert_eq!(back.smart_albums.len(), 2);
        assert_eq!(back.smart_albums[0].name, "Keepers");
        assert_eq!(back.smart_albums[0].query, "rating>=4 -rejected");
        // An empty smart_albums list serialises away (sparse file).
        let empty = read_folder(&std::env::temp_dir()).unwrap_or_default();
        assert!(empty.smart_albums.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
