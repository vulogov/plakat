//! Library walk + classification (RFC PHOTOS-1 §4.1, §7).
//!
//! A directory is a **Folder** if it contains only sub-directories, an **Album** if it holds at
//! least one supported image. No HJSON is required — classification is derived from contents; the
//! `folder.hjson` / `album.hjson` files (see [`crate::photos::hjson`]) are written lazily on the
//! first user action.

use std::path::{Path, PathBuf};

/// Image extensions the library recognises (standard raster + camera RAW). Lower-case, no dot.
pub const IMAGE_EXTS: &[&str] = &[
    // standard raster
    "png", "jpg", "jpeg", "webp", "gif", "tiff", "tif", "bmp", "tga", "qoi", "ppm", "pgm", "pbm",
    "pnm", "ff", "ico", "avif", "heic", "heif", "hif", // HEIF family (external decode fallback)
    // camera RAW (rawloader)
    "arw", "srf", "sr2", "nef", "nrw", "cr2", "crw", "raf", "orf", "rw2", "pef", "srw", "erf",
    "kdc", "dcr", "dng", "iiq", "mos", "3fr", "mef", "ari", "raw",
];

/// Is `path` a supported image by extension?
pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Is `ext` (lower-case, no dot) a camera-RAW format handled by `rawloader`?
pub fn is_raw_ext(ext: &str) -> bool {
    matches!(
        ext,
        "arw" | "srf" | "sr2" | "nef" | "nrw" | "cr2" | "crw" | "raf" | "orf" | "rw2" | "pef"
            | "srw" | "erf" | "kdc" | "dcr" | "dng" | "iiq" | "mos" | "3fr" | "mef" | "ari" | "raw"
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Folder,
    Album,
    SmartAlbum,
}

/// A node in the library tree.
#[derive(Clone, Debug)]
pub struct LibraryNode {
    pub path: PathBuf,
    pub kind: NodeKind,
    /// Display name — dir basename here; overridden by HJSON `name` when loaded.
    pub name: String,
    /// Number of supported images directly in this node (albums only; 0 for folders).
    pub image_count: usize,
    pub children: Vec<LibraryNode>,
}

impl LibraryNode {
    /// Total images in this subtree (this node + all descendants).
    pub fn total_images(&self) -> usize {
        self.image_count + self.children.iter().map(|c| c.total_images()).sum::<usize>()
    }
}

/// Count supported image files directly inside `dir` (non-recursive). Hidden derivatives
/// (`*_mask.png`) and sidecars are not images, so they don't count.
fn count_images(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_file() && is_image(&e.path()))
        .count()
}

/// Classify a single directory from its contents (RFC §4.1): an Album if it holds any image, else
/// a Folder. (An empty dir is a Folder.)
pub fn classify(dir: &Path) -> NodeKind {
    if count_images(dir) > 0 {
        NodeKind::Album
    } else {
        NodeKind::Folder
    }
}

/// Recursively walk `root` into a [`LibraryNode`] tree. Folders recurse; albums are leaves (their
/// images are loaded lazily by the grid pane, not here). Symlinks are not followed. Directories
/// beginning with `.` are skipped.
pub fn walk(root: &Path) -> std::io::Result<LibraryNode> {
    let basename = root.file_name().and_then(|n| n.to_str()).unwrap_or("/").to_string();

    // One read_dir per directory: count images + collect child dirs in a single pass (the cold-start
    // bottleneck was three separate scans per directory). `file_type()` avoids a stat per entry where
    // the platform provides it.
    let mut images = 0usize;
    let mut child_dirs: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(root)?.flatten() {
        let p = e.path();
        let (is_dir, is_file) = match e.file_type() {
            Ok(t) => (t.is_dir(), t.is_file()),
            Err(_) => (p.is_dir(), p.is_file()),
        };
        if is_file {
            if is_image(&p) {
                images += 1;
            }
        } else if is_dir {
            let hidden = p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with('.')).unwrap_or(false);
            if !hidden {
                child_dirs.push(p);
            }
        }
    }
    // Album = holds images. Folder = only sub-dirs (or empty). A dir with BOTH images and subdirs
    // is treated as an Album (its own images shown; subdirs still walked as children).
    let kind = if images > 0 { NodeKind::Album } else { NodeKind::Folder };

    // Display name comes from the HJSON metadata `name` (album.hjson for albums, folder.hjson for
    // folders), falling back to the directory basename. This is a label only — the directory is
    // never renamed by setting it.
    let meta_name = if kind == NodeKind::Album {
        super::hjson::read_album(root).ok().and_then(|m| m.name)
    } else {
        super::hjson::read_folder(root).ok().and_then(|m| m.name)
    };
    let name = meta_name.filter(|n| !n.trim().is_empty()).unwrap_or(basename);

    child_dirs.sort();
    let mut children = Vec::with_capacity(child_dirs.len());
    for sub in child_dirs {
        children.push(walk(&sub)?);
    }

    Ok(LibraryNode { path: root.to_path_buf(), kind, name, image_count: images, children })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_and_counts() {
        let tmp = std::env::temp_dir().join(format!("plakat-photos-lib-{}", std::process::id()));
        let album = tmp.join("2024-trip");
        let sub = tmp.join("2025");
        std::fs::create_dir_all(&album).unwrap();
        std::fs::create_dir_all(sub.join("studio")).unwrap();
        std::fs::write(album.join("IMG_1.jpg"), b"x").unwrap();
        std::fs::write(album.join("IMG_2.png"), b"x").unwrap();
        std::fs::write(album.join("note.txt"), b"x").unwrap(); // not an image

        assert_eq!(classify(&album), NodeKind::Album);
        assert_eq!(classify(&sub), NodeKind::Folder);

        let tree = walk(&tmp).unwrap();
        assert_eq!(tree.kind, NodeKind::Folder);
        assert_eq!(tree.total_images(), 2);
        let trip = tree.children.iter().find(|c| c.name == "2024-trip").unwrap();
        assert_eq!(trip.kind, NodeKind::Album);
        assert_eq!(trip.image_count, 2);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn mixed_album_reports_direct_and_recursive_counts_separately() {
        // A directory with BOTH loose images AND sub-album directories (the workbench `take` layout):
        // its own `image_count` is the direct files, while `total_images()` folds in the sub-albums.
        // The tree badge uses the former for albums so `[direct]` matches what opening it shows.
        let tmp = std::env::temp_dir().join(format!("plakat-lib-mixed-{}", std::process::id()));
        let test = tmp.join("test");
        std::fs::create_dir_all(&test).unwrap();
        std::fs::write(test.join("a.png"), b"x").unwrap();
        std::fs::write(test.join("b.png"), b"x").unwrap();
        for sub in ["a", "a-2", "b"] {
            let d = test.join(format!("shot_{sub}"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("hi.png"), b"x").unwrap();
        }
        let node = walk(&test).unwrap();
        assert_eq!(node.kind, NodeKind::Album);
        assert_eq!(node.image_count, 2, "direct images (what the grid shows)");
        assert_eq!(node.total_images(), 5, "recursive incl. the 3 sub-albums");
        assert_eq!(node.children.len(), 3, "three sub-albums are children");
        assert!(node.children.iter().all(|c| c.image_count == 1));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn display_name_prefers_album_metadata_then_falls_back() {
        let tmp = std::env::temp_dir().join(format!("plakat-lib-name-{}", std::process::id()));
        let named = tmp.join("2024-07-14");
        let plain = tmp.join("misc");
        std::fs::create_dir_all(&named).unwrap();
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(named.join("IMG_1.jpg"), b"x").unwrap();
        std::fs::write(plain.join("IMG_2.jpg"), b"x").unwrap();
        // Give one album a metadata display name; leave the other without.
        super::super::hjson::write_album(
            &named,
            &super::super::hjson::AlbumMeta { name: Some("Summer Trip".into()), ..Default::default() },
        )
        .unwrap();

        let tree = walk(&tmp).unwrap();
        assert!(tree.children.iter().any(|c| c.name == "Summer Trip"), "metadata name shown");
        assert!(tree.children.iter().any(|c| c.name == "misc"), "fallback to dir basename");
        assert!(!tree.children.iter().any(|c| c.name == "2024-07-14"), "dir name overridden by metadata");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn raw_and_image_detection() {
        assert!(is_image(Path::new("a.ARW")));
        assert!(is_image(Path::new("a.png")));
        assert!(!is_image(Path::new("a.txt")));
        assert!(is_raw_ext("nef"));
        assert!(!is_raw_ext("png"));
    }
}
