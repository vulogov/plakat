use anyhow::{Result, anyhow};
use console::style;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    downloads: u64,
    #[serde(default, rename = "pipeline_tag")]
    pipeline: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default, rename = "trendingScore")]
    trending_score: Option<f32>,
}

fn http() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("plakat/0.1")
        .build()?)
}

/// One HF model hit for a UI (the `plakat ui` LoRA Hub HUGGINGFACE tab).
#[derive(Debug, Clone)]
pub struct HfHit {
    pub id: String,
    pub downloads: u64,
    pub pipeline: String,
}

/// Search HF models by `query`, newest-downloaded first. Returns the hits for a UI
/// to render (the CLI's [`print_search`] formats the same data to stdout).
///
/// Two-stage pre-filter: stage A is a **LoRA-tag-filtered** query (`filter=lora`) for
/// precision — most HF text-search hits for a style term are full checkpoints, not
/// adapters; stage B is the **plain search** for recall. Stage-A hits come first, then
/// any stage-B hits the tag missed fill the remaining slots (deduped by id). If the
/// tag-filtered call fails, stage B alone still returns results.
pub async fn search_models(query: &str, limit: usize) -> Result<Vec<HfHit>> {
    let client = http()?;
    // Stage A — tag-filtered (LoRA adapters). Best-effort: an error/empty result just
    // means stage B carries the search.
    let tagged = fetch_models(&client, query, Some("lora"), limit).await.unwrap_or_default();
    // Stage B — plain search (recall). This is the authoritative call; its failure is
    // the function's failure.
    let plain = fetch_models(&client, query, None, limit).await?;
    Ok(merge_hits(tagged, plain, limit))
}

/// One HF `/api/models` call → `HfHit`s. `tag` adds a `filter=<tag>` narrowing.
async fn fetch_models(client: &reqwest::Client, query: &str, tag: Option<&str>, limit: usize) -> Result<Vec<HfHit>> {
    let limit_s = limit.to_string();
    let mut params: Vec<(&str, &str)> = vec![
        ("search", query),
        ("limit", &limit_s),
        ("sort", "downloads"),
        ("direction", "-1"),
    ];
    if let Some(t) = tag {
        params.push(("filter", t));
    }
    let url = reqwest::Url::parse_with_params("https://huggingface.co/api/models", &params)?;
    let resp: Vec<ModelInfo> = client.get(url).send().await?.error_for_status()?.json().await?;
    Ok(resp
        .into_iter()
        .map(|m| HfHit { id: m.id, downloads: m.downloads, pipeline: m.pipeline.unwrap_or_default() })
        .collect())
}

/// Merge two stages (LoRA-tagged first, then plain) into one list: stage-A order is
/// preserved, stage-B hits whose id isn't already present fill the rest, capped at
/// `limit`.
fn merge_hits(primary: Vec<HfHit>, secondary: Vec<HfHit>, limit: usize) -> Vec<HfHit> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(limit);
    for h in primary.into_iter().chain(secondary) {
        if out.len() >= limit {
            break;
        }
        if seen.insert(h.id.clone()) {
            out.push(h);
        }
    }
    out
}

/// Download the LoRA `.safetensors` from an HF repo and COPY it into `dest_dir`
/// (the workspace `loras/`) so the LoRA Hub LOCAL tab picks it up. Picks the
/// smallest `.safetensors` in the repo (LoRAs are far smaller than base weights),
/// avoiding accidentally pulling a full checkpoint. Returns the copied path.
pub async fn download_lora_into(repo: &str, dest_dir: &std::path::Path) -> Result<std::path::PathBuf> {
    let files = crate::hf::info::repo_files(repo)
        .await
        .map_err(|e| anyhow!("listing {repo}: {e}"))?;
    let cand: Vec<&String> = files.iter().filter(|f| f.ends_with(".safetensors")).collect();
    if cand.is_empty() {
        return Err(anyhow!("{repo} has no .safetensors file"));
    }
    // Prefer a file whose name hints "lora"; else the first.
    let file = cand
        .iter()
        .find(|f| f.to_lowercase().contains("lora"))
        .copied()
        .unwrap_or(cand[0]);

    let cached = crate::hf::download::get_file(repo, file)
        .await
        .map_err(|e| anyhow!("downloading {repo}/{file}: {e}"))?;

    std::fs::create_dir_all(dest_dir)
        .map_err(|e| anyhow!("creating {}: {e}", dest_dir.display()))?;
    // Name it after the repo so collisions across repos don't clobber.
    let stem = repo.replace('/', "__");
    let dest = dest_dir.join(format!("{stem}.safetensors"));
    std::fs::copy(&cached, &dest).map_err(|e| anyhow!("copying into {}: {e}", dest.display()))?;
    Ok(dest)
}

pub async fn print_search(query: &str, limit: usize) -> Result<()> {
    let url = reqwest::Url::parse_with_params(
        "https://huggingface.co/api/models",
        &[
            ("search", query),
            ("limit", &limit.to_string()),
            ("sort", "downloads"),
            ("direction", "-1"),
        ],
    )?;
    let resp: Vec<ModelInfo> = http()?.get(url).send().await?.error_for_status()?.json().await?;
    print_table(&resp, query, false);
    Ok(())
}

/// Recommendations restricted to `pipeline_tag=text-to-image`.
/// `sort` accepts: downloads | likes | trending | recent.
pub async fn print_recommend(query: Option<&str>, sort: &str, limit: usize) -> Result<()> {
    let sort_key = match sort.to_lowercase().as_str() {
        "downloads" => "downloads",
        "likes" => "likes",
        "trending" | "trend" => "trendingScore",
        "recent" | "modified" | "lastmodified" => "lastModified",
        other => {
            return Err(anyhow!(
                "unknown sort {other:?} (expected: downloads | likes | trending | recent)"
            ));
        }
    };
    let limit_s = limit.to_string();
    let mut params: Vec<(&str, &str)> = vec![
        ("pipeline_tag", "text-to-image"),
        ("sort", sort_key),
        ("direction", "-1"),
        ("limit", &limit_s),
        ("full", "true"),
    ];
    if let Some(q) = query.filter(|s| !s.is_empty()) {
        params.push(("search", q));
    }
    let url = reqwest::Url::parse_with_params("https://huggingface.co/api/models", &params)?;
    let resp: Vec<ModelInfo> = http()?.get(url).send().await?.error_for_status()?.json().await?;
    let label = query.unwrap_or("text-to-image");
    print_table(&resp, label, sort_key == "trendingScore");
    Ok(())
}

fn print_table(items: &[ModelInfo], label: &str, show_trending: bool) {
    if items.is_empty() {
        println!("(no matches for {label:?})");
        return;
    }
    println!(
        "{}  {:<55}  {:>10}  {:>6}  {:>9}  {}",
        style(" ").dim(),
        style("repo").bold(),
        style("↓dl").bold(),
        style("♥").bold(),
        style(if show_trending { "trend" } else { "modified" }).bold(),
        style("pipeline").bold(),
    );
    for m in items {
        let trail = if show_trending {
            m.trending_score
                .map(|t| format!("{t:>9.2}"))
                .unwrap_or_else(|| "        -".to_string())
        } else {
            m.last_modified
                .as_deref()
                .map(|s| s.get(..10).unwrap_or(s).to_string())
                .unwrap_or_else(|| "        -".to_string())
        };
        println!(
            "{}  {:<55}  {:>10}  {:>6}  {:>9}  {}",
            style("•").cyan(),
            style(&m.id).bold(),
            m.downloads,
            m.likes,
            trail,
            style(m.pipeline.as_deref().unwrap_or("")).dim(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, dl: u64) -> HfHit {
        HfHit { id: id.into(), downloads: dl, pipeline: "text-to-image".into() }
    }

    #[test]
    fn merge_keeps_lora_stage_first_then_fills_and_dedups() {
        let tagged = vec![hit("a/lora-1", 100), hit("a/lora-2", 50)];
        let plain = vec![hit("a/lora-2", 50), hit("b/checkpoint", 999), hit("c/other", 10)];
        let merged = merge_hits(tagged, plain, 10);
        let ids: Vec<&str> = merged.iter().map(|h| h.id.as_str()).collect();
        // LoRA-tagged hits lead; the duplicate (lora-2) appears once; recall fills rest.
        assert_eq!(ids, vec!["a/lora-1", "a/lora-2", "b/checkpoint", "c/other"]);
    }

    #[test]
    fn merge_respects_the_limit() {
        let tagged = vec![hit("a", 1), hit("b", 1)];
        let plain = vec![hit("c", 1), hit("d", 1)];
        let merged = merge_hits(tagged, plain, 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].id, "a");
    }

    #[test]
    fn merge_with_empty_tagged_stage_is_plain_search() {
        // Stage A failed/empty → stage B alone, order preserved.
        let merged = merge_hits(vec![], vec![hit("x", 5), hit("y", 4)], 10);
        assert_eq!(merged.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), vec!["x", "y"]);
    }
}
