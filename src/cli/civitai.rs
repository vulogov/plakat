//! v0.16 phase 7: `plakat civitai` subcommand.
//!
//! Surface:
//!
//! ```text
//! plakat civitai search QUERY [--type TYPE] [--limit N] [--page P]
//! plakat civitai info  REF_OR_URL
//! plakat civitai download REF_OR_URL [--file NAME]
//! ```
//!
//! REF can be:
//! * a bare integer model ID (`123456`)
//! * a `civitai:123456` shorthand
//! * a full URL (`https://civitai.com/models/123456...`)
//! * a download URL (`https://civitai.com/api/download/models/789`)

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;

use crate::civitai::api::{self, AssetType};
use crate::civitai::download;

#[derive(Args, Debug)]
pub struct CivitaiArgs {
    #[command(subcommand)]
    pub cmd: CivitaiCmd,
}

#[derive(Subcommand, Debug)]
pub enum CivitaiCmd {
    /// Search Civitai by free-text query. Returns a table of
    /// matching models with their top version's base model + file
    /// name so the user can pick one to download.
    Search(SearchArgs),
    /// Show one model or version's details — versions, base model,
    /// trigger words, files.
    Info(InfoArgs),
    /// Download one asset into the local cache. Prints the absolute
    /// path on success.
    Download(DownloadArgs),
}

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Search query — matches name, tags, description.
    pub query: String,
    /// Filter to one asset type:
    /// `lora | checkpoint | ti | controlnet | vae | locon | hypernetwork`
    /// (synonyms accepted — `ckpt`, `embedding`, ...).
    #[arg(long = "type", value_name = "TYPE")]
    pub asset_type: Option<String>,
    /// Result page size (1..=100). Default 10.
    #[arg(long, default_value_t = 10, value_name = "N")]
    pub limit: u32,
    /// Page number (1-indexed). Default 1.
    ///
    /// Civitai's API uses page-based pagination when browsing by
    /// type (`--page 2 --type lora`) and cursor-based pagination
    /// when searching by query (`--page 2 --type lora "watercolor"`
    /// walks the cursor chain once from page 1). Deep paging with
    /// a query string costs one round-trip per intermediate page —
    /// refine the query instead of paging past ~5 for typical use.
    #[arg(long, default_value_t = 1, value_name = "P")]
    pub page: u32,
    /// When set, include NSFW results in the output. Default: filter
    /// them out post-fetch.
    #[arg(long = "include-nsfw", default_value_t = false)]
    pub include_nsfw: bool,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Model/version reference. See module docs for accepted forms.
    pub reference: String,
}

#[derive(Args, Debug)]
pub struct DownloadArgs {
    /// Model/version reference. See module docs for accepted forms.
    pub reference: String,
    /// Explicit file name within the version. When unset, picks the
    /// primary file (Civitai marks one per version) or the first.
    #[arg(long, value_name = "NAME")]
    pub file: Option<String>,
}

pub async fn run(args: CivitaiArgs) -> Result<()> {
    match args.cmd {
        CivitaiCmd::Search(a) => run_search(a).await,
        CivitaiCmd::Info(a) => run_info(a).await,
        CivitaiCmd::Download(a) => run_download(a).await,
    }
}

async fn run_search(args: SearchArgs) -> Result<()> {
    let asset_type = args
        .asset_type
        .as_deref()
        .map(|s| s.parse::<AssetType>())
        .transpose()
        .context("parsing --type")?;
    let resp = api::search(&args.query, asset_type, args.limit, args.page).await?;
    let items: Vec<_> = resp
        .items
        .into_iter()
        .filter(|m| args.include_nsfw || !m.nsfw)
        .collect();

    if items.is_empty() {
        println!(
            "(no matches for {:?}{}; page {})",
            args.query,
            asset_type
                .map(|t| format!(" type={}", t.as_query()))
                .unwrap_or_default(),
            args.page
        );
        return Ok(());
    }

    println!(
        "{} {} match(es) {}",
        style("•").cyan(),
        style(items.len()).bold(),
        resp.metadata
            .total_items
            .map(|n| format!("(total: {n})"))
            .unwrap_or_default(),
    );
    for m in &items {
        let version = m.model_versions.first();
        let base = version
            .and_then(|v| v.base_model.as_deref())
            .unwrap_or("?");
        let trigger = version
            .map(|v| v.trained_words.join(", "))
            .unwrap_or_default();
        let creator = m
            .creator
            .as_ref()
            .map(|c| c.username.as_str())
            .unwrap_or("?");
        let dl = m.stats.download_count;
        println!(
            "{}  {}  {}  {}",
            style(format!("{}", m.id)).bold(),
            style(&m.name).cyan(),
            style(format!("[{}]", m.asset_type)).dim(),
            style(format!("by @{creator}")).dim(),
        );
        println!(
            "  base={}  {}downloads={}",
            style(base).yellow(),
            if !trigger.is_empty() {
                format!("triggers=({}) ", style(&trigger).green())
            } else {
                String::new()
            },
            dl,
        );
        if let Some(v) = version {
            for f in &v.files {
                let marker = if f.primary {
                    style("★").yellow().to_string()
                } else {
                    style("·").dim().to_string()
                };
                println!(
                    "  {} {} ({})",
                    marker,
                    f.name,
                    crate::hf::cache::human_bytes((f.size_kb * 1024.0) as u64),
                );
            }
        }
    }
    Ok(())
}

async fn run_info(args: InfoArgs) -> Result<()> {
    let (model_id, version_id) = api::parse_ref(&args.reference)?;
    match (model_id, version_id) {
        (Some(m), None) => {
            let model = api::get_model(m).await?;
            print_model(&model);
        }
        (_, Some(v)) => {
            let ver = api::get_version(v).await?;
            if let Some(m) = model_id {
                println!("{} model {} → version {}", style("→").cyan(), m, ver.id);
            }
            print_version(&ver);
        }
        (None, None) => unreachable!("parse_ref guarantees at least one Some"),
    }
    Ok(())
}

fn print_model(m: &api::Model) {
    println!(
        "{}  {}  {}",
        style(format!("{}", m.id)).bold(),
        style(&m.name).cyan(),
        style(format!("[{}]", m.asset_type)).dim(),
    );
    if let Some(c) = &m.creator {
        println!("  by @{}", c.username);
    }
    println!(
        "  downloads={}  thumbs_up={}  rating={:.2}",
        m.stats.download_count, m.stats.thumbs_up_count, m.stats.rating
    );
    if !m.tags.is_empty() {
        println!("  tags: {}", m.tags.join(", "));
    }
    for v in m.model_versions.iter().take(5) {
        println!(
            "  {} {} (id={}, base={})",
            style("•").yellow(),
            v.name,
            v.id,
            v.base_model.as_deref().unwrap_or("?"),
        );
        if !v.trained_words.is_empty() {
            println!("    triggers: {}", v.trained_words.join(", "));
        }
        for f in &v.files {
            let marker = if f.primary {
                style("★").yellow().to_string()
            } else {
                style("·").dim().to_string()
            };
            println!(
                "    {} {} ({})",
                marker,
                f.name,
                crate::hf::cache::human_bytes((f.size_kb * 1024.0) as u64),
            );
        }
    }
    if m.model_versions.len() > 5 {
        println!("  (+{} more versions)", m.model_versions.len() - 5);
    }
}

fn print_version(v: &api::ModelVersion) {
    println!(
        "{} {}",
        style(format!("version {}", v.id)).bold(),
        style(&v.name).cyan()
    );
    println!("  base: {}", v.base_model.as_deref().unwrap_or("?"));
    if !v.trained_words.is_empty() {
        println!("  triggers: {}", v.trained_words.join(", "));
    }
    for f in &v.files {
        let marker = if f.primary {
            style("★").yellow().to_string()
        } else {
            style("·").dim().to_string()
        };
        println!(
            "  {} {} ({})",
            marker,
            f.name,
            crate::hf::cache::human_bytes((f.size_kb * 1024.0) as u64),
        );
    }
}

async fn run_download(args: DownloadArgs) -> Result<()> {
    let (model_id, version_id) = api::parse_ref(&args.reference)?;
    if model_id.is_none() && version_id.is_none() {
        anyhow::bail!("couldn't parse Civitai reference {:?}", args.reference);
    }
    let result = download::download_version(model_id, version_id, args.file.as_deref()).await?;
    if result.cache_hit {
        println!(
            "{} {} (cached)",
            style("✓").green(),
            result.path.display(),
        );
    } else {
        println!(
            "{} {} ({})",
            style("✓").green(),
            result.path.display(),
            crate::hf::cache::human_bytes(result.bytes_written),
        );
    }
    println!(
        "{} drop this path into --lora or --model PATH",
        style("→").cyan()
    );
    Ok(())
}
