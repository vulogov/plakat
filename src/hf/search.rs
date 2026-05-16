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
