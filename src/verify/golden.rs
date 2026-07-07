//! Golden-source resolution for Tier 1: a local `--golden-dir` (authored by
//! `tools/reference/dump.py`) or the Hugging Face **dataset** repo (frozen, the allowed
//! external — RFC_VERIFY). Both yield the two files `run_model` needs: `manifest.json` +
//! `goldens.safetensors`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Default HF dataset repo hosting the golden artifacts. Override with the
/// `PLAKAT_VERIFY_DATASET` env var (e.g. to point at a fork or a staging repo).
const DEFAULT_DATASET: &str = "vulogov98/plakat-verify";

/// The dataset repo to fetch goldens from.
pub fn dataset_repo() -> String {
    std::env::var("PLAKAT_VERIFY_DATASET").unwrap_or_else(|_| DEFAULT_DATASET.to_string())
}

/// The dataset-relative path for one golden file, e.g. `sd15/portrait_v1/manifest.json`.
/// Pure — the layout contract shared with `tools/reference/dump.py`'s output.
pub fn dataset_file_path(model: &str, fixture: &str, file: &str) -> String {
    format!("{model}/{fixture}/{file}")
}

/// Resolve `(manifest_path, goldens_path)` for `(model, fixture)`. With `local_dir` set,
/// reads `<dir>/<model>/<fixture>/…`; otherwise fetches from the HF dataset (cached like
/// model weights). Errors when the goldens don't exist (caller maps that to a skip).
pub async fn resolve_golden_files(
    model: &str,
    fixture: &str,
    local_dir: Option<&Path>,
) -> Result<(PathBuf, PathBuf)> {
    if let Some(dir) = local_dir {
        let d = dir.join(model).join(fixture);
        let manifest = d.join("manifest.json");
        let goldens = d.join("goldens.safetensors");
        anyhow::ensure!(
            manifest.exists(),
            "no goldens at {} (author with `tools/reference/dump.py --model {model}`)",
            d.display()
        );
        return Ok((manifest, goldens));
    }
    let repo = dataset_repo();
    let manifest = crate::hf::download::get_dataset_file(&repo, &dataset_file_path(model, fixture, "manifest.json"))
        .await
        .with_context(|| format!("fetching goldens for {model}/{fixture} from HF dataset {repo}"))?;
    let goldens = crate::hf::download::get_dataset_file(&repo, &dataset_file_path(model, fixture, "goldens.safetensors"))
        .await
        .with_context(|| format!("fetching goldens.safetensors for {model}/{fixture} from {repo}"))?;
    Ok((manifest, goldens))
}

/// Resolve the Tier-2 golden PNG for `(model, fixture)` — the frozen regression reference.
/// Local `--golden-dir/<model>/<fixture>/golden.png` or the HF dataset. Errors → caller skips.
pub async fn resolve_golden_image(
    model: &str,
    fixture: &str,
    local_dir: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(dir) = local_dir {
        let p = dir.join(model).join(fixture).join("golden.png");
        anyhow::ensure!(p.exists(), "no golden image at {} (author with the Tier-2 freeze step)", p.display());
        return Ok(p);
    }
    let repo = dataset_repo();
    crate::hf::download::get_dataset_file(&repo, &dataset_file_path(model, fixture, "golden.png"))
        .await
        .with_context(|| format!("fetching golden.png for {model}/{fixture} from HF dataset {repo}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_file_path_layout() {
        assert_eq!(dataset_file_path("sd15", "portrait_v1", "manifest.json"), "sd15/portrait_v1/manifest.json");
        assert_eq!(dataset_file_path("stable-cascade", "portrait_v1", "goldens.safetensors"), "stable-cascade/portrait_v1/goldens.safetensors");
    }

    #[tokio::test]
    async fn local_dir_resolves_paths_and_errors_when_absent() {
        let dir = std::env::temp_dir().join(format!("plakat-golden-resolve-{}", std::process::id()));
        let d = dir.join("sd15").join("portrait_v1");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("manifest.json"), "{}").unwrap();
        std::fs::write(d.join("goldens.safetensors"), "").unwrap();
        let (m, g) = resolve_golden_files("sd15", "portrait_v1", Some(&dir)).await.unwrap();
        assert!(m.ends_with("sd15/portrait_v1/manifest.json") && g.ends_with("sd15/portrait_v1/goldens.safetensors"));
        // A model with no local goldens errors (→ caller skips).
        assert!(resolve_golden_files("sdxl", "portrait_v1", Some(&dir)).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
