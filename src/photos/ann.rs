//! Approximate-nearest-neighbour index for visual search at scale (RFC PHOTOS 3.13).
//!
//! Wraps a pure-Rust **HNSW** ([`instant_distance`]) over the library's CLIP vectors so a
//! visual / semantic search is **sub-linear** instead of a full cosine scan — the win shows on very
//! large libraries (≥ ~1M images) and it makes incremental growth cheap. The index is *derived*:
//! rebuilt from the (int8-quantized) vector cache, never the source of truth.

use std::path::PathBuf;

use instant_distance::{Builder, HnswMap, Point, Search};

use super::visual_search::{qdot, Cache, Embedding};

/// A CLIP embedding as an HNSW point. Distance is `1 − cosine` (angular-ish; lower = more similar),
/// with `qdot` approximating cosine over the unit vectors.
#[derive(Clone)]
struct EmbPoint(Embedding);

impl Point for EmbPoint {
    fn distance(&self, other: &Self) -> f32 {
        1.0 - qdot(&self.0, &other.0)
    }
}

/// An in-memory HNSW index over the vector cache. Not persisted — building from N vectors is
/// O(N log N), cheap next to the embedding cost, so it's rebuilt when the cache size changes.
pub struct AnnIndex {
    map: HnswMap<EmbPoint, PathBuf>,
    len: usize,
}

impl AnnIndex {
    /// Build the index from every embedding in `cache`. `None` when the cache is empty.
    pub fn build(cache: &Cache) -> Option<AnnIndex> {
        if cache.is_empty() {
            return None;
        }
        let (points, values): (Vec<EmbPoint>, Vec<PathBuf>) =
            cache.iter().map(|(p, (_, e))| (EmbPoint(e.clone()), p.clone())).unzip();
        let map = Builder::default().build(points, values);
        Some(AnnIndex { map, len: cache.len() })
    }

    /// How many vectors the index was built over.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The top-`k` nearest image paths to `query`, each with a similarity score (higher = closer).
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<(PathBuf, f32)> {
        let mut s = Search::default();
        let qp = EmbPoint(query.clone());
        self.map
            .search(&qp, &mut s)
            .take(k)
            .map(|item| (item.value.clone(), 1.0 - item.distance))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photos::visual_search::quantize;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn ann_finds_the_nearest_vector() {
        // Three distinct directions; a query near the first should rank it top.
        let mut cache = Cache::new();
        let a = unit((0..768).map(|i| if i < 256 { 1.0 } else { 0.0 }).collect());
        let b = unit((0..768).map(|i| if (256..512).contains(&i) { 1.0 } else { 0.0 }).collect());
        let c = unit((0..768).map(|i| if i >= 512 { 1.0 } else { 0.0 }).collect());
        cache.insert(PathBuf::from("a.png"), (1, quantize(&a)));
        cache.insert(PathBuf::from("b.png"), (1, quantize(&b)));
        cache.insert(PathBuf::from("c.png"), (1, quantize(&c)));

        let idx = AnnIndex::build(&cache).expect("built");
        assert_eq!(idx.len(), 3);
        // Query = a slightly perturbed toward b, but still closest to a.
        let q: Vec<f32> = unit((0..768).map(|i| if i < 256 { 1.0 } else if i < 300 { 0.1 } else { 0.0 }).collect());
        let hits = idx.search(&quantize(&q), 3);
        assert!(!hits.is_empty());
        assert!(hits[0].0.ends_with("a.png"), "nearest is a.png, got {:?}", hits[0].0);
        // Empty cache → no index.
        assert!(AnnIndex::build(&Cache::new()).is_none());
    }
}
