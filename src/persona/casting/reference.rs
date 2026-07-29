//! The identity reference set (RFC §11.3) — how a cast persona is stored and how its identity coherence
//! is measured. References are images plus precomputed ArcFace embeddings + landmarks + metadata, so
//! downstream operations never re-detect. The **coherence** math (pairwise cosine, centroid, threshold)
//! is pure and deterministic — testable without weights (the embeddings come from ArcFace at cast time).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Minimum pairwise ArcFace cosine for a set to be one person (§11.1). Below this the cast produced
/// several different people and must be re-run with tighter conditioning.
pub const COHERENCE_THRESHOLD: f32 = 0.50;

/// One stored reference (§11.3). Carries everything downstream needs so nothing is re-detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub id: usize,
    /// Image path, relative to the reference-set directory.
    pub image: PathBuf,
    pub seed: u64,
    pub model: String,
    /// View (§11.2): `frontal` / `three-quarter-left` / `three-quarter-right` / `profile`.
    pub view: String,
    pub expression: String,
    pub conditioning_hash: String,
    pub detail_plan_hash: String,
    /// ArcFace embedding (unit-normalised).
    pub embedding: Vec<f32>,
    /// Scorecard aggregate against the spec (`None` if not scored).
    pub score: Option<f32>,
    /// LAION aesthetic score (secondary sort key).
    pub aesthetic: Option<f32>,
    /// Cosine to the set centroid — the default reference weight (§11.3). Filled at set assembly.
    #[serde(default)]
    pub centroid_cosine: f32,
}

/// Identity-coherence measurement over a reference set (§11.1/§11.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coherence {
    /// Unit-normalised mean embedding.
    pub centroid: Vec<f32>,
    /// Mean of the off-diagonal pairwise cosines.
    pub mean_cosine: f32,
    /// Worst (minimum) pairwise cosine — the set is only as coherent as its worst pair.
    pub min_cosine: f32,
    /// `min_cosine >= threshold`.
    pub passes: bool,
    pub threshold: f32,
    /// Full pairwise cosine matrix (row-major), for the report.
    pub matrix: Vec<Vec<f32>>,
}

/// A stored, cast reference set (§11.3). Serialised as `reference_set.json` alongside the images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSet {
    pub persona: String,
    pub model: String,
    pub tier: String,
    pub references: Vec<Reference>,
    pub coherence: Coherence,
}

/// Cosine similarity of two (assumed finite) vectors. 0 if either is degenerate.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-12);
    (dot / d).clamp(-1.0, 1.0)
}

/// Unit-normalised mean of a set of embeddings (the identity centroid, §11.3).
pub fn centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let mut acc = vec![0.0f32; dim];
    for e in embeddings {
        if e.len() != dim {
            continue;
        }
        for (a, &v) in acc.iter_mut().zip(e) {
            *a += v;
        }
    }
    let norm = acc.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    acc.iter().map(|v| v / norm).collect()
}

/// Compute the coherence of a set of embeddings against `threshold` (§11.1).
pub fn compute_coherence(embeddings: &[Vec<f32>], threshold: f32) -> Coherence {
    let n = embeddings.len();
    let centroid = centroid(embeddings);
    let mut matrix = vec![vec![1.0f32; n]; n];
    let (mut sum, mut count, mut min) = (0.0f32, 0u32, 1.0f32);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let c = cosine(&embeddings[i], &embeddings[j]);
            matrix[i][j] = c;
            if i < j {
                sum += c;
                count += 1;
                min = min.min(c);
            }
        }
    }
    let mean_cosine = if count > 0 { sum / count as f32 } else { 1.0 };
    // a single reference is trivially "coherent" with itself.
    let min_cosine = if count > 0 { min } else { 1.0 };
    Coherence { centroid, mean_cosine, min_cosine, passes: min_cosine >= threshold, threshold, matrix }
}

impl ReferenceSet {
    /// Assemble a set from cast references: compute coherence + fill each reference's centroid cosine.
    pub fn assemble(persona: &str, model: &str, tier: &str, mut references: Vec<Reference>, threshold: f32) -> ReferenceSet {
        let embeddings: Vec<Vec<f32>> = references.iter().map(|r| r.embedding.clone()).collect();
        let coherence = compute_coherence(&embeddings, threshold);
        for r in references.iter_mut() {
            r.centroid_cosine = cosine(&r.embedding, &coherence.centroid);
        }
        // most-representative first (§11.3 default weighting).
        references.sort_by(|a, b| b.centroid_cosine.partial_cmp(&a.centroid_cosine).unwrap_or(std::cmp::Ordering::Equal));
        ReferenceSet { persona: persona.into(), model: model.into(), tier: tier.into(), references, coherence }
    }

    /// The directory's manifest path.
    pub fn manifest_path(dir: &Path) -> PathBuf {
        dir.join("reference_set.json")
    }

    /// Write the manifest (images are written separately by the caster).
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(Self::manifest_path(dir), json).context("writing reference_set.json")?;
        Ok(())
    }

    /// Load a manifest.
    pub fn load(dir: &Path) -> Result<ReferenceSet> {
        let text = std::fs::read_to_string(Self::manifest_path(dir)).with_context(|| format!("reading {}", Self::manifest_path(dir).display()))?;
        serde_json::from_str(&text).context("parsing reference_set.json")
    }

    /// The centroid-weighted "most representative" reference (the canonical face for swapping).
    pub fn canonical(&self) -> Option<&Reference> {
        self.references.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic near-identical embeddings: a base + tiny per-index perturbation.
    fn emb(base: f32, jitter: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| base + jitter * ((i % 7) as f32 - 3.0)).collect();
        let n = raw.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        raw.iter().map(|v| v / n).collect()
    }

    #[test]
    fn cosine_and_centroid_basics() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &c).abs() < 1e-6);
        let cen = centroid(&[a.clone(), c.clone()]);
        assert!((cen.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-4, "centroid is unit-norm");
    }

    #[test]
    fn coherent_set_passes_outlier_fails() {
        let dim = 32;
        let tight: Vec<Vec<f32>> = (0..4).map(|i| emb(0.6, 0.002 * i as f32, dim)).collect();
        let coh = compute_coherence(&tight, COHERENCE_THRESHOLD);
        assert!(coh.passes, "near-identical faces cohere ({:.3})", coh.min_cosine);
        assert!(coh.mean_cosine > 0.9);

        // now inject a different person (very different direction).
        let mut mixed = tight.clone();
        mixed.push(emb(0.6, 0.9, dim));
        let coh2 = compute_coherence(&mixed, COHERENCE_THRESHOLD);
        assert!(coh2.min_cosine < coh.min_cosine, "the outlier drops the worst pair");
    }

    #[test]
    fn single_reference_is_trivially_coherent() {
        let coh = compute_coherence(&[emb(0.5, 0.0, 16)], COHERENCE_THRESHOLD);
        assert!(coh.passes);
        assert_eq!(coh.min_cosine, 1.0);
    }

    #[test]
    fn assemble_sorts_by_centroid_and_round_trips() {
        let dim = 16;
        let refs: Vec<Reference> = (0..3)
            .map(|i| Reference {
                id: i,
                image: format!("ref_{i}.png").into(),
                seed: i as u64,
                model: "sdxl".into(),
                view: "frontal".into(),
                expression: "neutral".into(),
                conditioning_hash: "h".into(),
                detail_plan_hash: "d".into(),
                embedding: emb(0.6, 0.01 * i as f32, dim),
                score: Some(0.8),
                aesthetic: Some(5.0),
                centroid_cosine: 0.0,
            })
            .collect();
        let set = ReferenceSet::assemble("alice", "sdxl", "B", refs, COHERENCE_THRESHOLD);
        // centroid cosines are filled + sorted descending.
        assert!(set.references[0].centroid_cosine >= set.references[1].centroid_cosine);
        assert!(set.canonical().is_some());
        let json = serde_json::to_string(&set).unwrap();
        let back: ReferenceSet = serde_json::from_str(&json).unwrap();
        assert_eq!(back.references.len(), 3);
        assert_eq!(back.persona, "alice");
    }
}
