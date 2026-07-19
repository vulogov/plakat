//! T2 ML editing (RFC PHOTOS-1 Phase 4) — the op/job types for the cursor-image ML edits. The heavy
//! work runs on the photos-local resident worker ([`super::mlworker`]), which holds the pipelines
//! across ops and reports progress + cancel back inline (no TUI suspend). Unlike the T1 pixel edits
//! ([`super::edit`]) these load a model; the output is always a *new* image (the source is never
//! modified) recorded as a variant. This module now just defines the ops + the output-path helper.

use std::path::{Path, PathBuf};

/// A T2 edit to run on an image. Prompt-driven ops carry their prompt; all produce a new file.
#[derive(Clone, Debug)]
pub enum MlOp {
    /// Real-ESRGAN ML upscale (×4).
    Upscale,
    /// img2img transform under a prompt (SDXL by default).
    Img2img { prompt: String },
    /// IC-Light relight under a lighting prompt.
    Relight { prompt: String },
}

impl MlOp {
    pub fn label(&self) -> String {
        match self {
            MlOp::Upscale => "ML upscale ×4".into(),
            MlOp::Img2img { prompt } => format!("img2img: {prompt}"),
            MlOp::Relight { prompt } => format!("relight: {prompt}"),
        }
    }

    pub(crate) fn suffix(&self) -> &'static str {
        match self {
            MlOp::Upscale => "upscale",
            MlOp::Img2img { .. } => "img2img",
            MlOp::Relight { .. } => "relight",
        }
    }
}

/// A queued ML edit: the operation, its source image, and the album to land the result in.
#[derive(Clone, Debug)]
pub struct MlJob {
    pub op: MlOp,
    pub input: PathBuf,
    pub album: PathBuf,
}

/// `<album>/<stem>_<suffix>.png`, suffixing `-2`, `-3`, … on a collision.
pub(crate) fn dest_path(album: &Path, input: &Path, suffix: &str) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let base = format!("{stem}_{suffix}");
    let cand = album.join(format!("{base}.png"));
    if !cand.exists() {
        return cand;
    }
    for i in 2..10_000 {
        let c = album.join(format!("{base}-{i}.png"));
        if !c.exists() {
            return c;
        }
    }
    album.join(format!("{base}-dup.png"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dest_path_dedups() {
        let dir = std::env::temp_dir().join(format!("plakat-mledit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("IMG_1.jpg");
        let first = dest_path(&dir, &input, "upscale");
        assert_eq!(first.file_name().unwrap().to_str().unwrap(), "IMG_1_upscale.png");
        std::fs::write(&first, b"x").unwrap();
        let second = dest_path(&dir, &input, "upscale");
        assert_eq!(second.file_name().unwrap().to_str().unwrap(), "IMG_1_upscale-2.png");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
