//! Identity encoding **quality score** for the People library (RFC §11 ENCODING). A
//! person's reference photos should depict the *same* face consistently; if they don't
//! (different people, wildly different angles, low quality), any identity strategy will
//! produce a muddy result. We quantify that with the **mean pairwise cosine similarity**
//! of the references' ArcFace embeddings: 1.0 = identical, ~0.4+ = clearly the same
//! person across poses, low/negative = inconsistent refs.
//!
//! Reuses the existing face stack (SCRFD detect + 5-point align + ArcFace IR-ResNet50) —
//! the same components the face-swap / FaceID paths verified against onnxruntime. Loads
//! only the detector + ArcFace (no inswapper generator), so it's light enough to run on
//! demand from the UI.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use std::path::Path;

use crate::pipelines::face_models::{FaceAlignment, IResnet50, prepare_face_tensor};
use crate::pipelines::scrfd::{SCRFDConfig, SCRFDDetector};

/// The outcome of scoring an identity's references.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Mean pairwise cosine similarity of the per-ref ArcFace embeddings, in `[-1, 1]`.
    /// `1.0` for a single usable ref (nothing to compare against — treated as ideal).
    pub score: f32,
    /// How many references yielded a detectable, embeddable face.
    pub faces: usize,
    /// How many references were supplied.
    pub total: usize,
}

/// SCRFD + ArcFace, just enough to embed faces for the quality score.
pub struct IdentityScorer {
    detector: SCRFDDetector,
    arcface: IResnet50,
    device: Device,
    dtype: DType,
}

impl IdentityScorer {
    /// Resolve + load SCRFD and ArcFace weights (the same env vars / default repos the
    /// face-swap path uses). Errors if no face detector is configured.
    pub async fn load_resolved(device: &Device) -> Result<Self> {
        let dtype = DType::F32;
        let scrfd_path = crate::pipelines::scrfd::resolve_scrfd_weights()
            .await?
            .context(
                "identity quality needs a face detector — set PLAKAT_SCRFD_WEIGHTS / \
                 PLAKAT_SCRFD_HF (or rely on the default SCRFD download)",
            )?;
        let detector = SCRFDDetector::load(&scrfd_path, SCRFDConfig::default(), device, dtype)?;
        let arcface_path = resolve_arcface_weights().await?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[arcface_path], dtype, device)?
        };
        let arcface = IResnet50::new(vb).context("loading ArcFace for identity quality")?;
        Ok(Self { detector, arcface, device: device.clone(), dtype })
    }

    /// The unit-norm ArcFace embedding (512-d) of the primary face in `photo`, or `None`
    /// when no face is detected.
    pub fn embed(&self, photo: &Path) -> Result<Option<Vec<f32>>> {
        let Some(landmarks) = self.detector.detect_primary_normalised(photo)? else {
            return Ok(None);
        };
        let tensor = prepare_face_tensor(photo, FaceAlignment::Landmarks(landmarks), &self.device, self.dtype)?;
        let emb = self.arcface.forward(&tensor)?; // (1, 512), unit-norm
        let v = emb.flatten_all()?.to_vec1::<f32>()?;
        Ok(Some(v))
    }

    /// Score an identity from its reference photos: embed each, then the mean pairwise
    /// cosine similarity across the embeddings.
    pub fn score(&self, photos: &[std::path::PathBuf]) -> Result<QualityReport> {
        let total = photos.len();
        let mut embs = Vec::new();
        for p in photos {
            if let Some(e) = self.embed(p)? {
                embs.push(e);
            }
        }
        Ok(QualityReport { score: mean_pairwise_cosine(&embs), faces: embs.len(), total })
    }
}

/// Cosine similarity of two equal-length vectors (each already unit-norm → a dot
/// product, but we normalize defensively).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Mean pairwise cosine similarity over a set of embeddings. `0` embeddings → `0.0`
/// (unknown); exactly one usable embedding → `1.0` (a single ref is internally
/// consistent — nothing to disagree with).
pub fn mean_pairwise_cosine(embs: &[Vec<f32>]) -> f32 {
    match embs.len() {
        0 => 0.0,
        1 => 1.0,
        n => {
            let mut sum = 0.0;
            let mut pairs = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    sum += cosine(&embs[i], &embs[j]);
                    pairs += 1;
                }
            }
            sum / pairs as f32
        }
    }
}

/// Resolve the ArcFace weights (mirrors the face-swap resolver: env override → HF spec →
/// default repo).
async fn resolve_arcface_weights() -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var("PLAKAT_ARCFACE_WEIGHTS") {
        let path = std::path::PathBuf::from(&p);
        anyhow::ensure!(path.exists(), "PLAKAT_ARCFACE_WEIGHTS {p} does not exist");
        return Ok(path);
    }
    let (repo, file) = if let Ok(spec) = std::env::var("PLAKAT_ARCFACE_HF") {
        crate::pipelines::ip_adapter::parse_hf_spec(&spec, "PLAKAT_ARCFACE_HF")?
    } else {
        (
            crate::pipelines::faceswap::DEFAULT_ARCFACE_REPO.to_string(),
            crate::pipelines::faceswap::DEFAULT_ARCFACE_FILE.to_string(),
        )
    };
    crate::hf::download::get_file(&repo, &file)
        .await
        .with_context(|| format!("downloading ArcFace weights from {repo}/{file}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pairwise_cosine_edges() {
        assert_eq!(mean_pairwise_cosine(&[]), 0.0);
        assert_eq!(mean_pairwise_cosine(&[vec![1.0, 0.0]]), 1.0);
        // Two identical unit vectors → 1.0.
        assert!((mean_pairwise_cosine(&[vec![1.0, 0.0], vec![1.0, 0.0]]) - 1.0).abs() < 1e-6);
        // Orthogonal → 0.0.
        assert!(mean_pairwise_cosine(&[vec![1.0, 0.0], vec![0.0, 1.0]]).abs() < 1e-6);
        // Opposite → -1.0.
        assert!((mean_pairwise_cosine(&[vec![1.0, 0.0], vec![-1.0, 0.0]]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pairwise_cosine_averages_all_pairs() {
        // Three vectors [1,0],[1,0],[0,1] → pairs cos = (1, 0, 0) → mean 1/3.
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!((mean_pairwise_cosine(&embs) - 1.0 / 3.0).abs() < 1e-6);
    }
}
