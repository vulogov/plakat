//! Tier 1 — per-module correctness: run a model on a fixture, capture named intermediates
//! (via `TensorTap`), and compare each against a golden reference tensor from a manifest.
//!
//! This module lands the **complete comparison engine** (`compare_against_goldens`) plus
//! golden loading — all self-contained and unit-tested. Wiring the actual capture points
//! into the pipelines and authoring/hosting the goldens are Phase 1b / Phase 2, so the CLI
//! path (`run`) reports honestly when a model isn't instrumented or has no goldens yet.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use super::compare::compare;
use super::manifest::Manifest;
use super::{Check, Report, VerifyConfig};

/// Load a goldens safetensors (name → tensor) onto CPU.
pub fn load_goldens(path: &Path) -> Result<HashMap<String, Tensor>> {
    candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading goldens {}", path.display()))
}

/// Compare captured intermediates against the manifest's goldens. Emits one `Check` per
/// manifest tensor: a missing capture (pipeline not instrumented for it) or a missing/
/// wrong-shape golden fails; otherwise the correlation + max-abs verdict decides.
pub fn compare_against_goldens(
    manifest: &Manifest,
    captured: &HashMap<String, Tensor>,
    goldens: &HashMap<String, Tensor>,
) -> Vec<Check> {
    let mut checks = Vec::with_capacity(manifest.tensors.len());
    // Stable order (manifests are HashMaps) so reports/CI diffs are deterministic.
    let mut names: Vec<&String> = manifest.tensors.keys().collect();
    names.sort();
    for name in names {
        let spec = &manifest.tensors[name];
        let check_name = format!("tier1.{}.{}", manifest.model, name);
        let golden = match goldens.get(name) {
            Some(g) => g,
            None => {
                checks.push(Check::fail(check_name, 1, format!("golden tensor {name:?} missing from the safetensors")));
                continue;
            }
        };
        // Shape sanity vs the manifest (a stale golden / arch drift shows here).
        if golden.dims() != spec.shape.as_slice() {
            checks.push(Check::fail(
                check_name,
                1,
                format!("golden {name:?} shape {:?} != manifest {:?}", golden.dims(), spec.shape),
            ));
            continue;
        }
        let cand = match captured.get(name) {
            Some(c) => c,
            None => {
                checks.push(Check::fail(
                    check_name,
                    1,
                    format!("no capture for {name:?} — is the pipeline instrumented for it?"),
                ));
                continue;
            }
        };
        match compare(cand, golden) {
            Ok(stats) => {
                let th = spec.thresholds();
                if stats.passes(&th) {
                    checks.push(Check::pass(
                        check_name,
                        1,
                        format!("corr {:.5} · max_abs {:.4} (≥{:.4} / ≤{:.4})", stats.corr, stats.max_abs, th.corr_min, th.max_abs),
                    ));
                } else {
                    checks.push(Check::fail(
                        check_name,
                        1,
                        format!("corr {:.5} / max_abs {:.4} vs thresholds corr≥{:.4} max_abs≤{:.4}", stats.corr, stats.max_abs, th.corr_min, th.max_abs),
                    ));
                }
            }
            Err(e) => checks.push(Check::fail(check_name, 1, format!("{e:#}"))),
        }
    }
    checks
}

/// CLI Tier-1 entry. Resolves goldens for the requested model(s); until the pipelines are
/// instrumented (Phase 1b) and the goldens are authored + hosted (Phase 2), this reports the
/// state honestly rather than pretending to verify. The comparison ENGINE above is complete
/// and tested — capture points plug straight into it.
pub fn run(report: &mut Report, cfg: &VerifyConfig) {
    let models: Vec<String> = match &cfg.model {
        Some(m) => vec![m.clone()],
        // The Phase-2 pilot set; listed so `verify --tier 1` is explicit about coverage.
        None => ["sd15", "sdxl", "sd35-medium", "pixart", "stable-cascade", "animatediff"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    for model in models {
        report.push(Check::skip(
            format!("tier1.{model}"),
            1,
            "no golden tensors yet (authored + hosted in RFC_VERIFY phase 2) and pipeline capture points land in phase 1b — comparison engine is ready",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::manifest::Manifest;
    use crate::verify::Status;

    fn tensor(shape: (usize, usize), fill: f32) -> Tensor {
        Tensor::full(fill, shape, &Device::Cpu).unwrap()
    }

    fn manifest(corr_min: f64, max_abs: f64) -> Manifest {
        Manifest::from_json(&format!(
            r#"{{ "model": "sd15", "fixture": "f", "tensors": {{
                "clip.penultimate": {{ "shape": [2, 3], "corr_min": {corr_min}, "max_abs": {max_abs} }}
            }} }}"#
        ))
        .unwrap()
    }

    #[test]
    fn end_to_end_pass_when_capture_matches_golden() {
        let m = manifest(0.999, 0.01);
        let g = [("clip.penultimate".to_string(), tensor((2, 3), 1.0))].into_iter().collect();
        // A capture that matches the golden within tolerance.
        let cap = [("clip.penultimate".to_string(), tensor((2, 3), 1.0))].into_iter().collect();
        let checks = compare_against_goldens(&m, &cap, &g);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].status, Status::Pass, "detail: {}", checks[0].detail);
    }

    #[test]
    fn end_to_end_fail_on_magnitude_and_missing_and_shape() {
        let m = manifest(0.999, 0.01);
        // Wrong magnitude (corr still 1.0 for a constant… but constants: use a gradient).
        let golden: HashMap<_, _> =
            [("clip.penultimate".to_string(), Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu).unwrap())]
                .into_iter()
                .collect();
        // Off by a large additive shift → max_abs blows the 0.01 bound.
        let cap: HashMap<_, _> =
            [("clip.penultimate".to_string(), Tensor::new(&[[1f32, 2., 3.], [4., 5., 7.]], &Device::Cpu).unwrap())]
                .into_iter()
                .collect();
        let checks = compare_against_goldens(&m, &cap, &golden);
        assert_eq!(checks[0].status, Status::Fail, "max_abs 1.0 must exceed 0.01");

        // Missing capture → fail.
        let empty: HashMap<String, Tensor> = HashMap::new();
        assert_eq!(compare_against_goldens(&m, &empty, &golden)[0].status, Status::Fail);

        // Missing golden → fail.
        assert_eq!(compare_against_goldens(&m, &cap, &empty)[0].status, Status::Fail);

        // Wrong-shape golden → fail (manifest says [2,3]).
        let bad_shape: HashMap<_, _> =
            [("clip.penultimate".to_string(), tensor((2, 4), 1.0))].into_iter().collect();
        assert_eq!(compare_against_goldens(&m, &cap, &bad_shape)[0].status, Status::Fail);
    }

    #[test]
    fn goldens_round_trip_through_safetensors() {
        // Prove the load path: write a goldens safetensors, load it, compare.
        let dir = std::env::temp_dir().join(format!("plakat-verify-golden-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("goldens.safetensors");
        let mut save: HashMap<String, Tensor> = HashMap::new();
        save.insert("clip.penultimate".to_string(), tensor((2, 3), 1.0));
        candle_core::safetensors::save(&save, &path).unwrap();

        let goldens = load_goldens(&path).unwrap();
        let m = manifest(0.999, 0.01);
        let cap = [("clip.penultimate".to_string(), tensor((2, 3), 1.0))].into_iter().collect();
        assert_eq!(compare_against_goldens(&m, &cap, &goldens)[0].status, Status::Pass);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
