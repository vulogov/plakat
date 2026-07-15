//! `--import` support shared across the image-producing commands (RFC PHOTOS-IMPORT).
//!
//! Any command that writes an image can gain `--import <ALBUM>` by flattening [`ImportArgs`] into its
//! clap struct and calling [`ImportArgs::apply`] on the output paths. The actual album write lives in
//! [`crate::photos::import`] (behind `--features photos`); without that feature `--import` errors with
//! a build hint, so the flag is inert rather than silently ignored.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args as ClapArgs;

/// Flattened `--import` / `--import-move` flags shared by the generation commands.
#[derive(ClapArgs, Debug, Clone, Default)]
pub struct ImportArgs {
    /// Import each output into this photo album (RFC PHOTOS-IMPORT): the file (+ its `.json`
    /// sidecar) is copied into the album directory and its generation parameters are recorded in
    /// the album's `album.hjson`, so it shows up already curated in `plakat photos`. The album is
    /// created if it doesn't exist. Requires a build with `--features photos`.
    #[arg(long = "import", value_name = "ALBUM_DIR")]
    pub import: Option<PathBuf>,

    /// With `--import`, MOVE each output into the album instead of copying (leaves only the album
    /// copy). No effect without `--import`.
    #[arg(long = "import-move", default_value_t = false)]
    pub import_move: bool,
}

impl ImportArgs {
    /// Import `files` into the configured album (no-op if `--import` wasn't given). Best-effort per
    /// file; logs a one-line summary.
    pub fn apply(&self, files: &[PathBuf]) -> Result<()> {
        import_outputs(self.import.as_deref(), files, self.import_move)
    }

    /// Fail fast (before any expensive generation) if `--import` was given on a binary built without
    /// the `photos` feature. No-op when `--import` is absent or the feature is present.
    pub fn validate(&self) -> Result<()> {
        if self.import.is_some() {
            #[cfg(not(feature = "photos"))]
            anyhow::bail!(
                "--import requires a build with the `photos` feature — rebuild with \
                 `cargo build --features photos` (or install the photos-enabled binary)."
            );
        }
        Ok(())
    }
}

/// Image files directly in `dir` (non-recursive) as a set — used to diff a command's output
/// directory before/after it runs, so `--import` picks up exactly this run's new images.
pub fn image_snapshot(dir: &Path) -> HashSet<PathBuf> {
    let exts = ["png", "jpg", "jpeg", "webp"];
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| exts.contains(&e.to_ascii_lowercase().as_str()))
                    .unwrap_or(false)
        })
        .collect()
}

/// Run a command's `fut`, then import whatever new image files it produced under `out` into the
/// `--import` album. `out` may be a directory (batch commands) or a single output file path (its
/// parent dir is snapshotted). No-op wrapper when `--import` wasn't given.
pub async fn run_with_import<F>(import: ImportArgs, out: PathBuf, fut: F) -> Result<()>
where
    F: std::future::Future<Output = Result<()>>,
{
    if import.import.is_none() {
        return fut.await;
    }
    import.validate()?; // fail before the work if photos isn't compiled in
    // A path with an image extension is a file → snapshot its parent; otherwise treat it as a dir.
    let img_ext = out
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp"))
        .unwrap_or(false);
    let dir = if img_ext {
        out.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
    } else {
        out.clone()
    };
    let before = image_snapshot(&dir);
    fut.await?;
    let new_files: Vec<PathBuf> = image_snapshot(&dir).difference(&before).cloned().collect();
    import.apply(&new_files)
}

#[cfg(all(test, feature = "photos"))]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    #[tokio::test]
    async fn run_with_import_lands_new_dir_outputs_in_album() {
        let base = std::env::temp_dir().join(format!("plakat-rwi-{}", std::process::id()));
        let out = base.join("out");
        let album = base.join("Album");
        std::fs::create_dir_all(&out).unwrap();
        // A pre-existing image that must NOT be imported (only this run's new files).
        let old = out.join("preexisting.png");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([9, 9, 9]))).save(&old).unwrap();

        let import = ImportArgs { import: Some(album.clone()), import_move: false };
        let out_dir = out.clone();
        run_with_import(import, out.clone(), async move {
            // Simulate a command writing one new output into the out dir.
            let p = out_dir.join("plakat-1.png");
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([1, 2, 3]))).save(&p).unwrap();
            Ok(())
        })
        .await
        .unwrap();

        let am = crate::photos::hjson::read_album(&album).unwrap();
        assert!(am.images.contains_key("plakat-1.png"), "new output imported");
        assert!(!am.images.contains_key("preexisting.png"), "pre-existing file not imported");
        assert!(album.join("plakat-1.png").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn run_with_import_noop_without_flag() {
        let base = std::env::temp_dir().join(format!("plakat-rwi-noop-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let import = ImportArgs::default(); // no --import
        let ran = std::cell::Cell::new(false);
        run_with_import(import, base.join("x.png"), async {
            ran.set(true);
            Ok(())
        })
        .await
        .unwrap();
        assert!(ran.get(), "wrapped future still runs when --import is absent");
        let _ = std::fs::remove_dir_all(&base);
    }
}

/// Import `files` into the photo `album`, recording each output's generation params in the album's
/// HJSON. No-op when `album` is `None`. Errors (with a build hint) if plakat was built without
/// `--features photos`.
pub fn import_outputs(album: Option<&Path>, files: &[PathBuf], move_files: bool) -> Result<()> {
    let Some(album) = album else { return Ok(()) };
    if files.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "photos")]
    {
        let n = crate::photos::import::import_outputs(album, files, move_files)?;
        crate::ui::progress::println(&format!(
            "  --import: {n} image(s) → {}",
            album.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "photos"))]
    {
        let _ = move_files;
        anyhow::bail!(
            "--import {} requires a build with the `photos` feature — rebuild with \
             `cargo build --features photos` (or install the photos-enabled binary).",
            album.display()
        );
    }
}
