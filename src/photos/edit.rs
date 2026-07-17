//! T1 pixel editing (RFC PHOTOS-1 Phase 3) — non-destructive, replayable parametric edits.
//!
//! Edits never mutate your originals irreversibly: on the first edit of an image, the pristine file
//! is copied into a hidden `.plakat_edits/` backup (skipped by the library walk), and the visible
//! file is *re-derived* from that backup by replaying the whole edit list. Because every rebuild
//! starts from the original, chaining ten edits costs one re-encode from the pristine bytes, not ten
//! generational ones. Undo drops the last edit and rebuilds; revert clears them and restores the
//! original. The edit list lives in the image's `album.hjson` record (`edits`), so it survives across
//! sessions and travels with the album.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::DynamicImage;

use super::hjson::EditEntry;

/// One replayable pixel operation. Serialised to/from an [`EditEntry`] for `album.hjson`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditOp {
    RotateCw,
    RotateCcw,
    Rotate180,
    FlipH,
    FlipV,
    Grayscale,
    /// Brightness delta (image `brighten`, ±/step).
    Brightness(i32),
    /// Contrast delta (image `adjust_contrast`, ±/step).
    Contrast(i32),
    /// Centered square (1:1) crop.
    CropSquare,
    /// Centered crop to aspect ratio `w:h` (largest fitting rect).
    CropAspect { w: u32, h: u32 },
    /// Free-form crop: a rectangle in [0,1] fractions of the image (x, y, width, height).
    Crop { x: f32, y: f32, w: f32, h: f32 },
}

impl EditOp {
    /// Apply this op to `img`, returning the transformed image.
    pub fn apply(self, img: DynamicImage) -> DynamicImage {
        match self {
            EditOp::RotateCw => img.rotate90(),
            EditOp::RotateCcw => img.rotate270(),
            EditOp::Rotate180 => img.rotate180(),
            EditOp::FlipH => img.fliph(),
            EditOp::FlipV => img.flipv(),
            EditOp::Grayscale => img.grayscale(),
            EditOp::Brightness(v) => img.brighten(v),
            EditOp::Contrast(v) => img.adjust_contrast(v as f32),
            EditOp::CropSquare => centered_aspect(&img, 1, 1),
            EditOp::CropAspect { w, h } => centered_aspect(&img, w, h),
            EditOp::Crop { x, y, w, h } => {
                let (iw, ih) = (img.width() as f32, img.height() as f32);
                let cx = (x.clamp(0.0, 1.0) * iw) as u32;
                let cy = (y.clamp(0.0, 1.0) * ih) as u32;
                let cw = ((w.clamp(0.0, 1.0) * iw) as u32).max(1).min(img.width() - cx.min(img.width() - 1));
                let ch = ((h.clamp(0.0, 1.0) * ih) as u32).max(1).min(img.height() - cy.min(img.height() - 1));
                img.crop_imm(cx, cy, cw, ch)
            }
        }
    }

    /// A short human label for the status line / edit menu.
    pub fn label(self) -> String {
        match self {
            EditOp::RotateCw => "rotate ⟳".into(),
            EditOp::RotateCcw => "rotate ⟲".into(),
            EditOp::Rotate180 => "rotate 180°".into(),
            EditOp::FlipH => "flip H".into(),
            EditOp::FlipV => "flip V".into(),
            EditOp::Grayscale => "grayscale".into(),
            EditOp::Brightness(_) => "brightness".into(),
            EditOp::Contrast(_) => "contrast".into(),
            EditOp::CropSquare => "crop 1:1".into(),
            EditOp::CropAspect { w, h } => format!("crop {w}:{h}"),
            EditOp::Crop { .. } => "crop (free-form)".into(),
        }
    }

    /// Serialise to an `album.hjson` edit-log entry.
    pub fn to_entry(self) -> EditEntry {
        let mut params = std::collections::HashMap::new();
        let op = match self {
            EditOp::RotateCw => "rotate_cw",
            EditOp::RotateCcw => "rotate_ccw",
            EditOp::Rotate180 => "rotate_180",
            EditOp::FlipH => "flip_h",
            EditOp::FlipV => "flip_v",
            EditOp::Grayscale => "grayscale",
            EditOp::Brightness(v) => {
                params.insert("value".into(), serde_json::json!(v));
                "brightness"
            }
            EditOp::Contrast(v) => {
                params.insert("value".into(), serde_json::json!(v));
                "contrast"
            }
            EditOp::CropSquare => "crop_square",
            EditOp::CropAspect { w, h } => {
                params.insert("w".into(), serde_json::json!(w));
                params.insert("h".into(), serde_json::json!(h));
                "crop_aspect"
            }
            EditOp::Crop { x, y, w, h } => {
                params.insert("x".into(), serde_json::json!(x));
                params.insert("y".into(), serde_json::json!(y));
                params.insert("w".into(), serde_json::json!(w));
                params.insert("h".into(), serde_json::json!(h));
                "crop"
            }
        };
        EditEntry { op: op.to_string(), params, ts: None }
    }

    /// Parse a bare op tag (`rotate_cw`, `grayscale`, `crop_square`, …) into an op. For the
    /// param-less ops used by the NL pipeline; value ops default their value to 0.
    pub fn from_tag(tag: &str) -> Option<EditOp> {
        Self::from_entry(&EditEntry { op: tag.to_string(), params: Default::default(), ts: None })
    }

    /// Parse an `album.hjson` edit-log entry back into an op (unknown ops → `None`, skipped on replay).
    pub fn from_entry(e: &EditEntry) -> Option<EditOp> {
        let val = || e.params.get("value").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let u = |k: &str| e.params.get(k).and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let fr = |k: &str| e.params.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        Some(match e.op.as_str() {
            "rotate_cw" => EditOp::RotateCw,
            "rotate_ccw" => EditOp::RotateCcw,
            "rotate_180" => EditOp::Rotate180,
            "flip_h" => EditOp::FlipH,
            "flip_v" => EditOp::FlipV,
            "grayscale" => EditOp::Grayscale,
            "brightness" => EditOp::Brightness(val()),
            "contrast" => EditOp::Contrast(val()),
            "crop_square" => EditOp::CropSquare,
            "crop_aspect" => EditOp::CropAspect { w: u("w"), h: u("h") },
            "crop" => EditOp::Crop { x: fr("x"), y: fr("y"), w: fr("w"), h: fr("h") },
            _ => return None,
        })
    }
}

/// Largest centered rectangle of `img` with aspect ratio `aw:ah`.
fn centered_aspect(img: &DynamicImage, aw: u32, ah: u32) -> DynamicImage {
    let (iw, ih) = (img.width(), img.height());
    if aw == 0 || ah == 0 {
        return img.clone();
    }
    let target = aw as f64 / ah as f64;
    let (cw, ch) = if (iw as f64 / ih as f64) > target {
        (((ih as f64 * target).round() as u32).min(iw), ih) // too wide → cap width
    } else {
        (iw, ((iw as f64 / target).round() as u32).min(ih)) // too tall → cap height
    };
    let (cw, ch) = (cw.max(1), ch.max(1));
    img.crop_imm((iw - cw) / 2, (ih - ch) / 2, cw, ch)
}

/// Replay `ops` over a pristine `original`, returning the fully-edited image.
pub fn replay(original: &DynamicImage, ops: &[EditOp]) -> DynamicImage {
    let mut img = original.clone();
    for op in ops {
        img = op.apply(img);
    }
    img
}

fn backup_dir(album: &Path) -> PathBuf {
    album.join(".plakat_edits")
}

/// Path of the pristine backup for `filename` in `album` (hidden `.plakat_edits/`).
pub fn backup_path(album: &Path, filename: &str) -> PathBuf {
    backup_dir(album).join(filename)
}

/// Copy the current file into the hidden backup, once, before the first edit.
pub fn ensure_backup(album: &Path, filename: &str) -> Result<()> {
    let bak = backup_path(album, filename);
    if !bak.exists() {
        std::fs::create_dir_all(backup_dir(album))?;
        std::fs::copy(album.join(filename), &bak)
            .with_context(|| format!("backing up {filename}"))?;
    }
    Ok(())
}

/// Re-derive the visible file from the pristine backup + the full `ops` list. An empty `ops`
/// restores the original and removes the backup (a full revert).
pub fn rebuild_file(album: &Path, filename: &str, ops: &[EditOp]) -> Result<()> {
    let bak = backup_path(album, filename);
    let target = album.join(filename);
    if ops.is_empty() {
        if bak.exists() {
            std::fs::copy(&bak, &target).with_context(|| format!("restoring {filename}"))?;
            let _ = std::fs::remove_file(&bak);
        }
        return Ok(());
    }
    let original = image::open(&bak).with_context(|| format!("reading backup for {filename}"))?;
    let out = replay(&original, ops);
    out.save(&target).with_context(|| format!("writing edited {filename}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn img(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |x, y| Rgb([x as u8, y as u8, 0])))
    }

    #[test]
    fn rotate_swaps_dimensions_and_is_reversible() {
        let src = img(6, 4);
        let cw = EditOp::RotateCw.apply(src.clone());
        assert_eq!((cw.width(), cw.height()), (4, 6));
        // 4× rotate-cw returns to the original pixels.
        let round = EditOp::RotateCw.apply(EditOp::RotateCw.apply(EditOp::RotateCw.apply(cw)));
        assert_eq!(round.to_rgb8(), src.to_rgb8());
    }

    #[test]
    fn crop_square_is_centered_square() {
        let out = EditOp::CropSquare.apply(img(10, 4));
        assert_eq!((out.width(), out.height()), (4, 4));
    }

    #[test]
    fn entry_roundtrips_every_op() {
        for op in [
            EditOp::RotateCw, EditOp::RotateCcw, EditOp::Rotate180, EditOp::FlipH, EditOp::FlipV,
            EditOp::Grayscale, EditOp::Brightness(15), EditOp::Contrast(-10), EditOp::CropSquare,
            EditOp::CropAspect { w: 16, h: 9 },
            EditOp::Crop { x: 0.1, y: 0.2, w: 0.5, h: 0.4 },
        ] {
            assert_eq!(EditOp::from_entry(&op.to_entry()), Some(op));
        }
        // Unknown op → None (skipped, not a crash).
        let unknown = EditEntry { op: "warp_drive".into(), params: Default::default(), ts: None };
        assert_eq!(EditOp::from_entry(&unknown), None);
    }

    #[test]
    fn aspect_and_freeform_crop_dimensions() {
        // 16:9 centered crop of a 4:3 image → full width, reduced height.
        let out = EditOp::CropAspect { w: 16, h: 9 }.apply(img(400, 300));
        assert_eq!(out.width(), 400);
        assert_eq!(out.height(), 225); // 400 * 9/16
        // Free-form: middle 50%×40% of a 200×100 image.
        let ff = EditOp::Crop { x: 0.25, y: 0.3, w: 0.5, h: 0.4 }.apply(img(200, 100));
        assert_eq!((ff.width(), ff.height()), (100, 40));
    }

    #[test]
    fn backup_replay_and_full_revert() {
        let dir = std::env::temp_dir().join(format!("plakat-edit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = "p.png";
        img(8, 6).save(dir.join(name)).unwrap();

        // First edit: back up + rebuild (rotate cw → 6×8).
        ensure_backup(&dir, name).unwrap();
        rebuild_file(&dir, name, &[EditOp::RotateCw]).unwrap();
        let after = image::open(dir.join(name)).unwrap();
        assert_eq!((after.width(), after.height()), (6, 8));
        assert!(backup_path(&dir, name).exists());

        // Revert (empty ops): original restored, backup gone.
        rebuild_file(&dir, name, &[]).unwrap();
        let back = image::open(dir.join(name)).unwrap();
        assert_eq!((back.width(), back.height()), (8, 6));
        assert!(!backup_path(&dir, name).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
