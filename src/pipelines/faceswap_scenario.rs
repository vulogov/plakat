//! Scenario `type: faceswap` task (RFC FACESWAP-4 P2). Swap a source face into a scene as one step of a
//! scenario pipeline. Reuses the [`FaceSwapper`] engine; single-face (by index), mirroring the CLI's core.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use serde::Deserialize;

/// A `type: faceswap` task body: swap `source`'s identity into `scene`'s `face`-th face.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct FaceswapTaskCfg {
    /// Scene image whose face to replace.
    pub scene: Option<String>,
    /// Source face photo (identity to swap in).
    pub source: Option<String>,
    /// Which detected face to swap (0 = largest). Default 0.
    pub face: Option<usize>,
}

/// Validate up front (before any model load): scene + source are given and exist.
pub fn validate(cfg: &FaceswapTaskCfg) -> Result<()> {
    let scene = cfg.scene.as_deref().filter(|s| !s.trim().is_empty()).context("faceswap task needs `scene`")?;
    let source = cfg.source.as_deref().filter(|s| !s.trim().is_empty()).context("faceswap task needs `source`")?;
    anyhow::ensure!(Path::new(scene).exists(), "faceswap task: scene {scene} not found");
    anyhow::ensure!(Path::new(source).exists(), "faceswap task: source {source} not found");
    Ok(())
}

/// Run the swap → `<out_dir>/faceswap.png`.
pub async fn run_faceswap_task(cfg: &FaceswapTaskCfg, device: Device, out_dir: &Path, dry_run: bool) -> Result<()> {
    validate(cfg)?;
    let (scene, source) = (cfg.scene.clone().unwrap(), cfg.source.clone().unwrap());
    let face = cfg.face.unwrap_or(0);
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out = out_dir.join("faceswap.png");
    if dry_run {
        return Ok(());
    }
    let swapper = super::faceswap::FaceSwapper::load_resolved(&device, DType::F32)
        .await
        .context("loading face-swap models")?;
    let faces = swapper.detect(Path::new(&scene)).context("detecting faces")?;
    anyhow::ensure!(!faces.is_empty(), "faceswap task: no face detected in {scene}");
    anyhow::ensure!(face < faces.len(), "faceswap task: face {face} out of range — {} detected", faces.len());
    let latent = swapper.source_latent(Path::new(&source)).context("embedding the source face")?;
    let img = image::open(&scene).with_context(|| format!("reading {scene}"))?.to_rgb8();
    let swapped = swapper.swap_into(&img, faces[face].landmarks, &latent).context("face swap")?;
    swapped.save(&out).with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_requires_scene_and_source() {
        // Empty cfg → error naming the missing field.
        assert!(validate(&FaceswapTaskCfg::default()).is_err());
        let only_scene = FaceswapTaskCfg { scene: Some("a.png".into()), source: None, face: None };
        assert!(validate(&only_scene).is_err(), "source required");
        // A blank string counts as missing.
        let blank = FaceswapTaskCfg { scene: Some("  ".into()), source: Some("b.png".into()), face: None };
        assert!(validate(&blank).is_err(), "blank scene rejected");
    }
}
