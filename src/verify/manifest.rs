//! Tier 1 manifest: the per-`(model, fixture)` description of which intermediate tensors to
//! compare and at what thresholds. Authored offline (Phase 2) and frozen on HF; loaded here
//! by the pure-Rust verifier. See `RFC_VERIFY.md`.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::compare::Thresholds;

/// One golden tensor's expected shape + pass thresholds.
#[derive(Clone, Debug, Deserialize)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub corr_min: f64,
    pub max_abs: f64,
}

impl TensorSpec {
    pub fn thresholds(&self) -> Thresholds {
        Thresholds { corr_min: self.corr_min, max_abs: self.max_abs }
    }
}

/// The manifest for one `(model, fixture)`.
#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    /// Model alias the goldens were captured for (e.g. `sd15`).
    pub model: String,
    /// Model revision the goldens correspond to (optional; documents provenance).
    #[serde(default)]
    pub model_revision: String,
    /// Fixture id (the deterministic input the tensors were captured under).
    pub fixture: String,
    /// plakat architecture version — bump when a module's shape/semantics change, so stale
    /// goldens are rejected rather than silently mis-compared.
    #[serde(default)]
    pub plakat_arch: String,
    /// Where the goldens came from: `diffusers==X` (correctness oracle) or `plakat@<sha>`
    /// (regression baseline).
    #[serde(default)]
    pub provenance: String,
    /// Named intermediate tensors → their spec. Keys match the capture-point names the
    /// pipelines emit (e.g. `clip_l.penultimate`, `unet.mid`, `vae.decoded`).
    pub tensors: HashMap<String, TensorSpec>,
}

impl Manifest {
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("parsing verify manifest JSON")
    }
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("reading manifest {}", path.display()))?;
        Self::from_json(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "model": "sd15",
        "model_revision": "abc123",
        "fixture": "portrait_v1",
        "plakat_arch": "sd_core@1",
        "provenance": "diffusers==0.27.2",
        "tensors": {
            "clip_l.penultimate": { "shape": [1, 77, 768], "corr_min": 0.999, "max_abs": 0.02 },
            "unet.mid":           { "shape": [1, 1280, 8, 8], "corr_min": 0.99, "max_abs": 0.05 }
        }
    }"#;

    #[test]
    fn parses_a_manifest_and_thresholds() {
        let m = Manifest::from_json(SAMPLE).unwrap();
        assert_eq!(m.model, "sd15");
        assert_eq!(m.fixture, "portrait_v1");
        assert_eq!(m.tensors.len(), 2);
        let clip = &m.tensors["clip_l.penultimate"];
        assert_eq!(clip.shape, vec![1, 77, 768]);
        let th = clip.thresholds();
        assert!((th.corr_min - 0.999).abs() < 1e-12 && (th.max_abs - 0.02).abs() < 1e-12);
    }

    #[test]
    fn optional_fields_default() {
        // A minimal manifest (no revision/arch/provenance) still parses.
        let m = Manifest::from_json(
            r#"{ "model": "sdxl", "fixture": "f", "tensors": {} }"#,
        )
        .unwrap();
        assert_eq!(m.model, "sdxl");
        assert!(m.model_revision.is_empty() && m.plakat_arch.is_empty());
    }
}
