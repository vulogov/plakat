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

/// Map a detected t2i variant to the sd3 loader's variant (mirrors the UI's `sd3_variant`).
fn sd3_variant(v: crate::pipelines::t2i::Variant) -> crate::pipelines::sd3::Variant {
    use crate::pipelines::{sd3, t2i::Variant};
    match v {
        Variant::Sd3Medium => sd3::Variant::Sd3Medium,
        Variant::Sd35Medium => sd3::Variant::Sd35Medium,
        Variant::Sd35Large => sd3::Variant::Sd35Large,
        Variant::Sd35LargeTurbo => sd3::Variant::Sd35LargeTurbo,
        _ => sd3::Variant::Sd35Medium,
    }
}

/// The model set for Tier 1: an explicit `--model`, else the pilot set.
pub fn models(cfg: &VerifyConfig) -> Vec<String> {
    match &cfg.model {
        Some(m) => vec![m.clone()],
        None => ["sd15", "sdxl", "sd35-medium", "pixart", "stable-cascade", "animatediff"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    }
}

/// CLI Tier-1 entry WITHOUT a golden source — report coverage honestly (the comparison
/// engine is ready; goldens + capture wiring land in later phases).
pub fn run(report: &mut Report, cfg: &VerifyConfig) {
    for model in models(cfg) {
        report.push(Check::skip(
            format!("tier1.{model}"),
            1,
            "no --golden-dir given; supply authored goldens to verify (engine ready)",
        ));
    }
}

/// Run a model against local goldens: load the pipeline, capture the fixture's intermediates,
/// and compare. Only SD-family models are instrumented so far (via `t2i::Pipeline`); others
/// report a skip. Goldens live at `<golden_dir>/<model>/<fixture>/{manifest.json,
/// goldens.safetensors}`.
pub async fn run_model(model: &str, golden_dir: &Path, device: &Device) -> Vec<Check> {
    // Resolve the manifest (pilot fixture) + goldens.
    let fixture = "portrait_v1";
    let dir = golden_dir.join(model).join(fixture);
    let manifest = match Manifest::from_file(&dir.join("manifest.json")) {
        Ok(m) => m,
        Err(_) => {
            return vec![Check::skip(
                format!("tier1.{model}"),
                1,
                format!("no goldens at {} (author with tools/reference/dump.py)", dir.display()),
            )];
        }
    };
    let goldens = match load_goldens(&dir.join("goldens.safetensors")) {
        Ok(g) => g,
        Err(e) => return vec![Check::fail(format!("tier1.{model}"), 1, format!("{e:#}"))],
    };
    let fx = match crate::verify::fixtures::get(&manifest.fixture) {
        Some(f) => f,
        None => {
            return vec![Check::fail(
                format!("tier1.{model}"),
                1,
                format!("unknown fixture {:?} — add it to src/verify/fixtures.rs", manifest.fixture),
            )];
        }
    };

    // Dispatch by family to the right instrumented pipeline. Each capture is additive and
    // reuses the real encode/forward internals. A load failure → clean skip (weights /
    // gating), a capture failure → fail.
    let wanted: std::collections::HashSet<String> = manifest.tensors.keys().cloned().collect();
    // AnimateDiff isn't alias-loadable (a base + motion adapter, loaded via flags), so it
    // doesn't fit this alias dispatch. Its headline bug — the CFG BLOCKED batch layout — is
    // already guarded structurally in Tier 0 (`tier0.cfg_batch_layout`); a numerical
    // motion-module tap needs threading into the flag-based load path (a follow-up).
    if model.contains("animatediff") {
        return vec![Check::skip(
            format!("tier1.{model}"),
            1,
            "AnimateDiff CFG layout is guarded in Tier 0; a numerical motion tap needs the flag-based load path (follow-up)",
        )];
    }
    let variant = crate::pipelines::t2i::Variant::detect(model);
    let load_skip = |e: anyhow::Error| vec![Check::skip(format!("tier1.{model}"), 1, format!("load failed (weights/gating?): {e:#}"))];
    let cap_fail = |e: anyhow::Error| vec![Check::fail(format!("tier1.{model}"), 1, format!("capture failed: {e:#}"))];

    let captured = if variant.is_sd3() {
        match crate::pipelines::sd3::Pipeline::load(crate::pipelines::sd3::LoadRequest {
            variant: sd3_variant(variant),
            repo: crate::hf::resolve_alias(model).to_string(),
            device: device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            controlnets: Vec::new(),
            embeddings: Vec::new(),
        })
        .await
        {
            Ok(mut p) => match p.capture_intermediates(fx.prompt, &wanted) {
                Ok(c) => c,
                Err(e) => return cap_fail(e),
            },
            Err(e) => return load_skip(e),
        }
    } else if variant.is_pixart() {
        match crate::pipelines::pixart::Pipeline::load(crate::pipelines::pixart::LoadRequest {
            repo: crate::hf::resolve_alias(model).to_string(),
            device: device.clone(),
            vae_cache: None,
            loras: Vec::new(),
            lora_scale: 1.0,
        })
        .await
        {
            Ok(p) => match p.capture_intermediates(fx.width, fx.height, &wanted) {
                Ok(c) => c,
                Err(e) => return cap_fail(e),
            },
            Err(e) => return load_skip(e),
        }
    } else if variant.is_cascade() {
        match crate::pipelines::cascade::Pipeline::load(crate::pipelines::cascade::LoadRequest {
            repo: crate::hf::resolve_alias(model).to_string(),
            device: device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            controlnet_weights: None,
            image_encoder_weights: None,
        })
        .await
        {
            Ok(p) => match p.capture_intermediates(fx.prompt, &wanted) {
                Ok(c) => c,
                Err(e) => return cap_fail(e),
            },
            Err(e) => return load_skip(e),
        }
    } else {
        // SD-family (SD 1.5 / 2.1 / SDXL) via t2i.
        match crate::pipelines::t2i::Pipeline::load(crate::pipelines::t2i::LoadRequest {
            model: model.to_string(),
            device: device.clone(),
            loras: Vec::new(),
            lora_scale: 1.0,
            use_refiner: false,
            embeddings: Vec::new(),
            vae_cache: None,
        })
        .await
        {
            Ok(p) => match p.capture_intermediates(fx.prompt, fx.width, fx.height, &wanted) {
                Ok(c) => c,
                Err(e) => return cap_fail(e),
            },
            Err(e) => return load_skip(e),
        }
    };

    compare_against_goldens(&manifest, &captured, &goldens)
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
