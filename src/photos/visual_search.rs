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

/// Image-embedding cache: path → (file mtime secs, unit vector). Held in-session and persisted
/// per-album to a hidden `.plakat_clip` file so visual search is fast after the first run.
pub type Cache = HashMap<PathBuf, (u64, Vec<f32>)>;

const CACHE_FILE: &str = ".plakat_clip";
const CACHE_MAGIC: &[u8; 8] = b"PKCLIP1\n";
const DIM: usize = 768;

/// Load persisted embeddings for images under each of `dirs` (its hidden `.plakat_clip`), keyed by
/// absolute image path. Corrupt / wrong-dimension entries are skipped, never fatal.
pub fn load_cache(dirs: &[PathBuf]) -> Cache {
    let mut cache = Cache::new();
    for dir in dirs {
        let _ = read_dir_cache(dir, &mut cache);
    }
    cache
}

fn read_dir_cache(dir: &Path, cache: &mut Cache) -> Option<()> {
    let bytes = std::fs::read(dir.join(CACHE_FILE)).ok()?;
    if bytes.len() < 12 || &bytes[..8] != CACHE_MAGIC {
        return None;
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let mut o = 12;
    for _ in 0..count {
        if o + 2 > bytes.len() {
            break;
        }
        let nlen = u16::from_le_bytes(bytes[o..o + 2].try_into().ok()?) as usize;
        o += 2;
        let name = std::str::from_utf8(bytes.get(o..o + nlen)?).ok()?.to_string();
        o += nlen;
        let mtime = u64::from_le_bytes(bytes.get(o..o + 8)?.try_into().ok()?);
        o += 8;
        let vec_bytes = bytes.get(o..o + DIM * 4)?;
        o += DIM * 4;
        let vec: Vec<f32> =
            vec_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        cache.insert(dir.join(name), (mtime, vec));
    }
    Some(())
}

/// Persist `cache`, grouped by parent directory, to each dir's `.plakat_clip`. Best-effort.
pub fn save_cache(cache: &Cache) {
    let mut by_dir: HashMap<&Path, Vec<(&str, u64, &Vec<f32>)>> = HashMap::new();
    for (path, (mt, v)) in cache {
        if v.len() != DIM {
            continue;
        }
        if let (Some(dir), Some(name)) = (path.parent(), path.file_name().and_then(|n| n.to_str())) {
            by_dir.entry(dir).or_default().push((name, *mt, v));
        }
    }
    for (dir, entries) in by_dir {
        let mut buf = Vec::with_capacity(12 + entries.len() * (DIM * 4 + 24));
        buf.extend_from_slice(CACHE_MAGIC);
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (name, mt, v) in entries {
            buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&mt.to_le_bytes());
            for f in v {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        let _ = std::fs::write(dir.join(CACHE_FILE), buf);
    }
}

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

/// Proactively embed every image in `items` that isn't already cached at its current mtime, returning
/// the updated cache. **Lazy model load** — if everything is already embedded, the model never loads
/// (the whole call is offline). This pre-persists the vector store so the first visual search is fast.
pub async fn embed_all(
    device: &Device,
    items: Vec<(PathBuf, PathBuf)>,
    mut cache: Cache,
    progress: impl Fn(usize, usize),
) -> Result<Cache> {
    let mut embedder: Option<ClipEmbedder> = None;
    let total = items.len();
    for (i, (path, _dir)) in items.into_iter().enumerate() {
        progress(i + 1, total);
        let mt = mtime_secs(&path);
        if matches!(cache.get(&path), Some((m, _)) if *m == mt) {
            continue; // already embedded at this version
        }
        if embedder.is_none() {
            embedder = Some(ClipEmbedder::load(device).await?);
        }
        if let Ok(v) = embedder.as_ref().unwrap().embed_image(&path) {
            cache.insert(path, (mt, v));
        }
    }
    Ok(cache)
}

/// CLIP semantic lookalike: rank `items` by embedding similarity to the image at `query_path`.
/// **Lazy model load** — the CLIP model is only loaded if *some* embedding isn't already cached, so a
/// fully-cached library ranks entirely offline (no model, no cache disk).
pub async fn search_by_image(
    device: &Device,
    items: Vec<(PathBuf, PathBuf)>,
    query_path: &Path,
    mut cache: Cache,
    progress: impl Fn(usize, usize),
) -> Result<(Vec<(PathBuf, PathBuf, f32)>, Cache)> {
    let mut embedder: Option<ClipEmbedder> = None;

    // Query embedding (cache-first; loads the model only on a miss).
    let qmt = mtime_secs(query_path);
    let q = match cache.get(query_path) {
        Some((m, v)) if *m == qmt => v.clone(),
        _ => {
            let e = ClipEmbedder::load(device).await?;
            let v = e.embed_image(query_path)?;
            cache.insert(query_path.to_path_buf(), (qmt, v.clone()));
            embedder = Some(e);
            v
        }
    };

    let total = items.len();
    let mut scored: Vec<(PathBuf, PathBuf, f32)> = Vec::with_capacity(total);
    for (i, (path, dir)) in items.into_iter().enumerate() {
        progress(i + 1, total);
        let mt = mtime_secs(&path);
        let vec = match cache.get(&path) {
            Some((m, v)) if *m == mt => v.clone(),
            _ => {
                if embedder.is_none() {
                    embedder = Some(ClipEmbedder::load(device).await?);
                }
                match embedder.as_ref().unwrap().embed_image(&path) {
                    Ok(v) => {
                        cache.insert(path.clone(), (mt, v.clone()));
                        v
                    }
                    Err(_) => continue,
                }
            }
        };
        scored.push((path, dir, cosine(&q, &vec)));
    }
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok((scored, cache))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_roundtrips_and_skips_corruption() {
        let base = std::env::temp_dir().join(format!("plakat-clipcache-{}", std::process::id()));
        let dir = base.join("Album");
        std::fs::create_dir_all(&dir).unwrap();

        let mut cache = Cache::new();
        cache.insert(dir.join("a.png"), (1000, vec![0.1_f32; DIM]));
        cache.insert(dir.join("b.png"), (2000, vec![0.2_f32; DIM]));
        // Wrong-dimension entry must not be written.
        cache.insert(dir.join("bad.png"), (3000, vec![0.5_f32; 10]));
        save_cache(&cache);
        assert!(dir.join(CACHE_FILE).exists());

        let back = load_cache(&[dir.clone()]);
        assert_eq!(back.len(), 2, "the 10-dim entry was skipped");
        let (mt, v) = back.get(&dir.join("a.png")).expect("a.png cached");
        assert_eq!(*mt, 1000);
        assert_eq!(v.len(), DIM);
        assert!((v[0] - 0.1).abs() < 1e-6);

        // A truncated/garbage file loads as empty, not a panic.
        std::fs::write(dir.join(CACHE_FILE), b"not a cache").unwrap();
        assert!(load_cache(&[dir.clone()]).is_empty());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn embed_all_is_offline_when_fully_cached() {
        // When every image is already embedded at its current mtime, embed_all loads no model
        // (so no device/network) and returns the cache unchanged.
        let dir = std::env::temp_dir().join(format!("plakat-embedall-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut cache = Cache::new();
        let mut items = Vec::new();
        for name in ["a.png", "b.png"] {
            let p = dir.join(name);
            std::fs::write(&p, b"x").unwrap();
            cache.insert(p.clone(), (mtime_secs(&p), vec![0.1_f32; DIM]));
            items.push((p, dir.clone()));
        }
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt
            .block_on(embed_all(&Device::Cpu, items, cache, |_, _| {}))
            .expect("fully-cached embed_all is offline");
        assert_eq!(out.len(), 2, "cache returned unchanged, model never loaded");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
