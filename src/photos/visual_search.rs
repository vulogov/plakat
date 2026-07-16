//! CLIP visual search (RFC PHOTOS-1 Phase 7): rank the library by how well each image matches a
//! free-text query in CLIP's joint image/text space — "find images that *look like* this".
//!
//! The per-image embedding is the expensive part, so it's cached in-session (path → (mtime, vector));
//! a repeat or refined search only pays the model load + query embed + a dot product per image. The
//! session cache is the seed of the derived vector store the RFC's storage note calls for.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::Device;

use crate::pipelines::clip_embed::{cosine, ClipEmbedder};

/// In-session image-embedding cache: path → (file mtime secs, unit vector).
pub type Cache = HashMap<PathBuf, (u64, Vec<f32>)>;

fn mtime_secs(p: &Path) -> u64 {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Rank `items` (image path + its source-album dir) by CLIP similarity to `query`, best first.
/// Reuses/updates `cache`, embedding only images not already cached at their current mtime. Returns
/// `(ranked (path, dir, score), updated cache)`. `progress(done, total)` is called per image.
pub async fn search(
    device: &Device,
    items: Vec<(PathBuf, PathBuf)>,
    query: &str,
    mut cache: Cache,
    progress: impl Fn(usize, usize),
) -> Result<(Vec<(PathBuf, PathBuf, f32)>, Cache)> {
    let embedder = ClipEmbedder::load(device).await?;
    let q = embedder.embed_text(query)?;
    let total = items.len();
    let mut scored: Vec<(PathBuf, PathBuf, f32)> = Vec::with_capacity(total);
    for (i, (path, dir)) in items.into_iter().enumerate() {
        progress(i + 1, total);
        let mt = mtime_secs(&path);
        let vec = match cache.get(&path) {
            Some((m, v)) if *m == mt => v.clone(),
            _ => match embedder.embed_image(&path) {
                Ok(v) => {
                    cache.insert(path.clone(), (mt, v.clone()));
                    v
                }
                Err(_) => continue, // skip unreadable images
            },
        };
        scored.push((path, dir, cosine(&q, &vec)));
    }
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok((scored, cache))
}
