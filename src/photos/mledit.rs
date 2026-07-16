//! T2 ML editing (RFC PHOTOS-1 Phase 4) — dispatch the cursor image to an existing plakat pipeline
//! via the stable [`crate::api`] builders, landing a new derivative in the album.
//!
//! Unlike the T1 pixel edits ([`super::edit`]) these load a model and take minutes, so they run with
//! the TUI **suspended** (see `run_ml_job` in the parent module): the alternate screen is dropped so
//! the pipeline's normal progress bars show on the real terminal, then the manager resumes and picks
//! up the new file. The output is a *new* image (the source is never modified) recorded as a variant.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::api::{self, UpscaleMethod};

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

    fn suffix(&self) -> &'static str {
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

impl MlJob {
    /// Default model for the prompt-driven ops. SDXL gives the best img2img/relight quality; heavy on
    /// 24 GB but the manager runs one job at a time with everything else freed.
    const MODEL: &'static str = "sdxl";

    /// Run the job (async), saving the result into the album under a deduped `<stem>_<op>.png`.
    /// Returns the output path.
    pub async fn run(&self) -> Result<PathBuf> {
        let out = dest_path(&self.album, &self.input, self.op.suffix());
        match &self.op {
            MlOp::Upscale => {
                let img = api::Upscale::new(&self.input)
                    .method(UpscaleMethod::RealEsrganX4)
                    .device("auto")
                    .run()
                    .await?;
                img.save(&out)?;
            }
            MlOp::Img2img { prompt } => {
                let imgs = api::Img2img::new(Self::MODEL, &self.input)
                    .prompt(prompt.clone())
                    .strength(0.5)
                    .device("auto")
                    .run()
                    .await?;
                let img = imgs.into_iter().next().ok_or_else(|| anyhow::anyhow!("no output"))?;
                img.save(&out)?;
            }
            MlOp::Relight { prompt } => {
                let img = api::Relight::new(&self.input)
                    .prompt(prompt.clone())
                    .device("auto")
                    .run()
                    .await?;
                img.save(&out)?;
            }
        }
        Ok(out)
    }
}

/// `<album>/<stem>_<suffix>.png`, suffixing `-2`, `-3`, … on a collision.
fn dest_path(album: &Path, input: &Path, suffix: &str) -> PathBuf {
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
