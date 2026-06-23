//! Repository introspection: query the HF tree API to report what's there
//! without downloading it.

use anyhow::{Result, anyhow, bail};
use console::style;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
}

async fn list_tree(repo: &str) -> Result<Vec<TreeEntry>> {
    let resolved = crate::hf::resolve_alias(repo);
    let url = reqwest::Url::parse_with_params(
        &format!("https://huggingface.co/api/models/{resolved}/tree/main"),
        &[("recursive", "true")],
    )?;
    let resp = reqwest::Client::builder()
        .user_agent("plakat/0.1")
        .build()?
        .get(url)
        .send()
        .await?
        .error_for_status()
        .map_err(|e| anyhow!("HF tree API for {resolved}: {e}"))?;
    let entries: Vec<TreeEntry> = resp.json().await?;
    Ok(entries)
}

/// List every FILE path in a repo (recursive) via the HF tree API. Used by
/// `pull_all` so a pull works for ANY repo layout (SD/SDXL diffusers, PixArt /
/// SD3 `transformer/`, Cascade, single-file, sharded) instead of guessing a
/// fixed SD-filename list. Surfaces gated/missing repos with an actionable
/// message rather than a bare per-file 404.
pub(crate) async fn repo_files(repo: &str) -> Result<Vec<String>> {
    let resolved = crate::hf::resolve_alias(repo);
    let url = reqwest::Url::parse_with_params(
        &format!("https://huggingface.co/api/models/{resolved}/tree/main"),
        &[("recursive", "true")],
    )?;
    let resp = reqwest::Client::builder()
        .user_agent("plakat/0.1")
        .build()?
        .get(url)
        .send()
        .await?;
    let status = resp.status();
    if matches!(
        status,
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
    ) {
        bail!(
            "{resolved} is GATED — accept its terms at https://huggingface.co/{resolved} \
             and set HF_TOKEN (https://huggingface.co/settings/tokens), then retry."
        );
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        bail!(
            "{resolved} was not found on HuggingFace (moved, renamed, or removed). \
             Check https://huggingface.co/{resolved}"
        );
    }
    let resp = resp
        .error_for_status()
        .map_err(|e| anyhow!("HF tree API for {resolved}: {e}"))?;
    let entries: Vec<TreeEntry> = resp.json().await?;
    Ok(entries
        .into_iter()
        .filter(|e| e.kind == "file")
        .map(|e| e.path)
        .collect())
}

/// Mirror of the pipelines' fp16-preferred fetch logic. One slot per
/// "component"; the first matching candidate wins per slot.
const SD_PRIORITY: &[(&str, &str)] = &[
    ("text_encoder/", "text_encoder/model.fp16.safetensors"),
    ("text_encoder/", "text_encoder/model.safetensors"),
    ("text_encoder_2/", "text_encoder_2/model.fp16.safetensors"),
    ("text_encoder_2/", "text_encoder_2/model.safetensors"),
    ("unet/", "unet/diffusion_pytorch_model.fp16.safetensors"),
    ("unet/", "unet/diffusion_pytorch_model.safetensors"),
    ("vae/", "vae/diffusion_pytorch_model.fp16.safetensors"),
    ("vae/", "vae/diffusion_pytorch_model.safetensors"),
];
const SD_TINY: &[&str] = &["tokenizer/tokenizer.json", "tokenizer_2/tokenizer.json"];

/// Flux (BFL-native single-file weights + diffusers text encoders).
const FLUX_FIXED: &[&str] = &[
    "ae.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.safetensors",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer_2/tokenizer.json",
    "text_encoder_2/config.json",
];

fn estimate_plakat_download(entries: &[TreeEntry]) -> u64 {
    use std::collections::HashSet;
    let files: HashSet<&str> = entries
        .iter()
        .filter(|e| e.kind == "file")
        .map(|e| e.path.as_str())
        .collect();
    let mut size_map: BTreeMap<&str, u64> = BTreeMap::new();
    for e in entries.iter().filter(|e| e.kind == "file") {
        size_map.insert(&e.path, e.size);
    }

    // Flux repos have flux1-{schnell,dev}.safetensors at root.
    let flux_main = ["flux1-schnell.safetensors", "flux1-dev.safetensors"]
        .iter()
        .find(|f| files.contains(*f as &str))
        .copied();
    if let Some(main) = flux_main {
        let mut total = *size_map.get(main).unwrap_or(&0);
        let mut text_encoder_picked = false;
        for f in FLUX_FIXED {
            if f.starts_with("text_encoder/model") {
                if text_encoder_picked {
                    continue;
                }
                if let Some(&s) = size_map.get(*f) {
                    total += s;
                    text_encoder_picked = true;
                }
                continue;
            }
            if let Some(&s) = size_map.get(*f) {
                total += s;
            }
        }
        return total;
    }

    // Otherwise SD-like layout.
    let mut picked: BTreeMap<&str, u64> = BTreeMap::new();
    for (component, candidate) in SD_PRIORITY {
        if picked.contains_key(component) {
            continue;
        }
        if files.contains(candidate) {
            picked.insert(component, *size_map.get(candidate).unwrap_or(&0));
        }
    }
    let mut total: u64 = picked.values().sum();
    for f in SD_TINY {
        if let Some(&s) = size_map.get(*f) {
            total += s;
        }
    }
    total
}

pub async fn print_size(repo: &str) -> Result<()> {
    let entries = list_tree(repo).await?;
    let files: Vec<&TreeEntry> = entries.iter().filter(|e| e.kind == "file").collect();
    if files.is_empty() {
        println!("(no files found in {repo})");
        return Ok(());
    }

    let mut by_dir: BTreeMap<String, (u64, usize)> = BTreeMap::new();
    let mut total = 0u64;
    let mut has_fp16 = false;
    for e in &files {
        total += e.size;
        if e.path.contains(".fp16.") {
            has_fp16 = true;
        }
        let dir = if e.path.contains('/') {
            e.path.split('/').next().unwrap_or("(root)").to_string()
        } else {
            "(root)".to_string()
        };
        let slot = by_dir.entry(dir).or_default();
        slot.0 += e.size;
        slot.1 += 1;
    }

    let resolved = crate::hf::resolve_alias(repo);
    println!(
        "{}  {}",
        style("Repo:").yellow().bold(),
        style(resolved).bold()
    );
    println!(
        "{}  {}  ({} files)",
        style("Total on Hub:").yellow().bold(),
        style(crate::hf::cache::human_bytes(total)).bold(),
        files.len(),
    );
    println!();
    println!("{}", style("Breakdown by directory:").bold());
    for (d, (size, count)) in &by_dir {
        println!(
            "  {:<22}  {:>10}  ({} file{})",
            d,
            crate::hf::cache::human_bytes(*size),
            count,
            if *count == 1 { "" } else { "s" }
        );
    }

    let dl_size = estimate_plakat_download(&entries);
    println!();
    println!(
        "{}  {}",
        style("plakat would download:").yellow().bold(),
        style(crate::hf::cache::human_bytes(dl_size)).bold().green()
    );
    if has_fp16 {
        println!(
            "  {}",
            style("fp16 variants available — plakat prefers them automatically").dim()
        );
    }
    Ok(())
}
