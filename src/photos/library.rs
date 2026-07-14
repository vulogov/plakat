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
    "pnm", "ff", "ico", "avif", // camera RAW (rawloader)
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

/// Does `dir` contain at least one sub-directory?
fn has_subdirs(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.path().is_dir())
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
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("/")
        .to_string();

    let images = count_images(root);
    let subdirs = has_subdirs(root);
    // Album = holds images. Folder = only sub-dirs (or empty). A dir with BOTH images and subdirs
    // is treated as an Album (its own images shown; subdirs still walked as children).
    let kind = if images > 0 { NodeKind::Album } else { NodeKind::Folder };

    let mut children = Vec::new();
    if subdirs {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(root)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && !p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with('.'))
                        .unwrap_or(false)
            })
            .collect();
        entries.sort();
        for sub in entries {
            children.push(walk(&sub)?);
        }
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
    fn raw_and_image_detection() {
        assert!(is_image(Path::new("a.ARW")));
        assert!(is_image(Path::new("a.png")));
        assert!(!is_image(Path::new("a.txt")));
        assert!(is_raw_ext("nef"));
        assert!(!is_raw_ext("png"));
    }
}
