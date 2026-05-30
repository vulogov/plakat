//! v0.16 phase 7: `plakat civitai` subcommand.
//!
//! Surface:
//!
//! ```text
//! plakat civitai search QUERY [--type TYPE] [--limit N] [--page P]
//! plakat civitai info  REF_OR_URL
//! plakat civitai download REF_OR_URL [--file NAME]
//! plakat civitai sync  USERNAME --out DIR [--type TYPE] [--limit N] [--dry-run]
//! ```
//!
//! REF can be:
//! * a bare integer model ID (`123456`)
//! * a `civitai:123456` shorthand
//! * a full URL (`https://civitai.com/models/123456...`)
//! * a download URL (`https://civitai.com/api/download/models/789`)
//!
//! v0.31 phase 1 (swap): `sync` bulk-downloads a creator's full
//! library. Walks the Civitai API's username pagination, picks each
//! model's primary version + primary file, and lands a copy in
//! `--out DIR` alongside the standard plakat cache. Idempotent on
//! rerun (files that already exist at DIR are skipped).

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
    /// v0.31: bulk-download a Civitai creator's full library.
    /// Walks pagination, picks each model's primary version + file,
    /// lands copies in `--out DIR`. Idempotent on rerun.
    Sync(SyncArgs),
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

#[derive(Args, Debug)]
pub struct SyncArgs {
    /// Civitai username to mirror. Case-sensitive (matches the URL
    /// path on civitai.com/user/<USERNAME>).
    pub username: String,
    /// Destination directory. Created if missing. Existing files
    /// are skipped (idempotent rerun).
    #[arg(long, value_name = "DIR")]
    pub out: std::path::PathBuf,
    /// Filter to one asset type:
    /// `lora | checkpoint | ti | controlnet | vae | locon | hypernetwork`
    /// (synonyms accepted). Omitted = all the user's models.
    #[arg(long = "type", value_name = "TYPE")]
    pub asset_type: Option<String>,
    /// Cap total models synced. Useful for testing — `--limit 3`
    /// downloads at most three. Default: unlimited.
    #[arg(long, value_name = "N")]
    pub limit: Option<usize>,
    /// Preview the plan without downloading anything. Lists each
    /// model + file + target path.
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,
}

pub async fn run(args: CivitaiArgs) -> Result<()> {
    match args.cmd {
        CivitaiCmd::Search(a) => run_search(a).await,
        CivitaiCmd::Info(a) => run_info(a).await,
        CivitaiCmd::Download(a) => run_download(a).await,
        CivitaiCmd::Sync(a) => run_sync(a).await,
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

/// v0.31 phase 1 (swap): bulk-download a creator's library.
///
/// Walks Civitai's username pagination via [`api::list_by_username`],
/// then for each model calls the same [`download::download_version`]
/// the single-asset path uses. Files land in plakat's standard
/// Civitai cache as a side effect (so subsequent `--lora` references
/// can hit them) and a copy is placed in `args.out` with the
/// version's primary file name.
///
/// Idempotent on rerun: each step checks both the cache hit AND the
/// destination's existing file, skipping when both are present.
async fn run_sync(args: SyncArgs) -> Result<()> {
    let asset_type = args
        .asset_type
        .as_deref()
        .map(|s| s.parse::<AssetType>())
        .transpose()
        .context("parsing --type")?;

    let spin = crate::ui::progress::spinner(&format!(
        "Listing {}'s Civitai library{}",
        args.username,
        asset_type
            .map(|t| format!(" ({})", t.as_query()))
            .unwrap_or_default()
    ));
    let models = api::list_by_username(&args.username, asset_type, 100).await?;
    spin.finish_with_message(format!(
        "✓ {} found {} model(s)",
        args.username,
        models.len()
    ));

    if models.is_empty() {
        println!(
            "(no models for @{}{})",
            args.username,
            asset_type
                .map(|t| format!(" type={}", t.as_query()))
                .unwrap_or_default(),
        );
        return Ok(());
    }

    let take = args.limit.unwrap_or(models.len()).min(models.len());
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    let mut ok_count = 0usize;
    let mut skip_count = 0usize;
    let mut fail_count = 0usize;
    for (idx, m) in models.iter().take(take).enumerate() {
        let version = match m.model_versions.first() {
            Some(v) => v,
            None => {
                println!(
                    "{} [{}/{}] {} (id {}) — no versions, skipping",
                    style("·").dim(),
                    idx + 1,
                    take,
                    m.name,
                    m.id,
                );
                skip_count += 1;
                continue;
            }
        };
        // Pick the version's first file matching the asset's
        // primary-file convention. If Civitai marks one as primary,
        // honour it; otherwise fall back to the first file.
        let primary = version
            .files
            .iter()
            .find(|f| f.primary)
            .or_else(|| version.files.first());
        let file = match primary {
            Some(f) => f,
            None => {
                println!(
                    "{} [{}/{}] {} v{} — no files, skipping",
                    style("·").dim(),
                    idx + 1,
                    take,
                    m.name,
                    version.id,
                );
                skip_count += 1;
                continue;
            }
        };
        let target = args.out.join(&file.name);
        if target.exists() {
            // Quick idempotency check: file already in DIR, skip
            // the download entirely. No size verification — the
            // user owns this DIR and we don't second-guess.
            println!(
                "{} [{}/{}] {} → {} (already present)",
                style("✓").green().dim(),
                idx + 1,
                take,
                m.name,
                target.display(),
            );
            skip_count += 1;
            continue;
        }
        if args.dry_run {
            println!(
                "{} [{}/{}] {} v{} → {} ({:.1} MB)",
                style("·").dim(),
                idx + 1,
                take,
                m.name,
                version.id,
                target.display(),
                file.size_kb / 1024.0,
            );
            continue;
        }
        println!(
            "{} [{}/{}] {} v{} ({:.1} MB)",
            style("⬇").cyan(),
            idx + 1,
            take,
            m.name,
            version.id,
            file.size_kb / 1024.0,
        );
        let dl_result =
            download::download_version(Some(m.id), Some(version.id), Some(&file.name)).await;
        match dl_result {
            Ok(result) => {
                // Copy from cache to args.out. fs::copy clobbers an
                // existing destination, but we already checked it
                // doesn't exist above.
                match std::fs::copy(&result.path, &target) {
                    Ok(bytes) => {
                        println!(
                            "  {} {} ({})",
                            style("✓").green(),
                            target.display(),
                            crate::hf::cache::human_bytes(bytes),
                        );
                        ok_count += 1;
                    }
                    Err(e) => {
                        println!(
                            "  {} copy from cache failed: {e}",
                            style("✗").red(),
                        );
                        fail_count += 1;
                    }
                }
            }
            Err(e) => {
                println!("  {} {e}", style("✗").red());
                fail_count += 1;
            }
        }
    }

    println!();
    println!(
        "{} sync done: {} downloaded, {} skipped, {} failed (out={})",
        style("·").dim(),
        ok_count,
        skip_count,
        fail_count,
        args.out.display(),
    );
    if fail_count > 0 {
        anyhow::bail!("{fail_count} download(s) failed");
    }
    Ok(())
}
