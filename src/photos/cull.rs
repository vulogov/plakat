//! Aesthetic auto-cull (Track A): rank images by the LAION aesthetic predictor and keep the top N.
//! The scorer holds a CLIP ViT-L/14 vision tower + the LAION MLP resident, so it loads once and
//! scores each image in turn. The manager uses the ranking to flag the keepers and reject the rest —
//! purely metadata (non-destructive), undoable via the curation snapshot.

use std::path::PathBuf;

use anyhow::Result;
use candle_core::Device;

use crate::pipelines::aesthetic::AestheticScorer;

/// Score every path by aesthetic quality, returning `(path, score)` sorted best-first. `progress(done,
/// total)` is called per image; unreadable images are skipped (not fatal).
pub async fn rank(
    device: &Device,
    paths: Vec<PathBuf>,
    progress: impl Fn(usize, usize),
) -> Result<Vec<(PathBuf, f32)>> {
    let scorer = AestheticScorer::load(device).await?;
    let total = paths.len();
    let mut scored: Vec<(PathBuf, f32)> = Vec::with_capacity(total);
    for (i, p) in paths.into_iter().enumerate() {
        progress(i + 1, total);
        if let Ok(s) = scorer.score_path(&p) {
            scored.push((p, s));
        }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}
