use anyhow::Result;
use console::style;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Process-wide cache override, set by the CLI `--cache-dir` flag.
static CACHE_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

pub fn set_override(p: PathBuf) {
    let _ = CACHE_DIR_OVERRIDE.set(p);
}

/// Resolve the HF cache root. Order:
///   1. CLI `--cache-dir` (set via set_override)
///   2. PLAKAT_CACHE_DIR env (also bound to the CLI flag)
///   3. HUGGINGFACE_HUB_CACHE env
///   4. HF_HOME env (with `/hub` suffix)
///   5. Default: ~/.cache/huggingface/hub
pub fn hf_cache_root() -> PathBuf {
    if let Some(p) = CACHE_DIR_OVERRIDE.get() {
        return p.clone();
    }
    if let Ok(p) = std::env::var("PLAKAT_CACHE_DIR") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    directories::BaseDirs::new()
        .map(|b| {
            b.home_dir()
                .join(".cache")
                .join("huggingface")
                .join("hub")
        })
        .unwrap_or_else(|| PathBuf::from(".hf-cache"))
}

pub fn list() -> Result<()> {
    let root = hf_cache_root();
    if !root.exists() {
        println!("(cache empty: {})", root.display());
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(&root)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    let mut total = 0u64;
    let mut shown = 0usize;
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(repo) = name.strip_prefix("models--") {
            let repo = repo.replace("--", "/");
            let size = dir_size(&entry.path()).unwrap_or(0);
            total = total.saturating_add(size);
            shown += 1;
            println!(
                "{}  {}  {}",
                style("•").cyan(),
                style(repo).bold(),
                style(human_bytes(size)).dim()
            );
        }
    }
    if shown == 0 {
        println!("(no cached models in {})", root.display());
    } else {
        println!(
            "{}  {} model(s), {} total",
            style("Σ").yellow(),
            shown,
            style(human_bytes(total)).bold()
        );
    }
    Ok(())
}

/// Remove one repo. Returns Ok(true) if removed, Ok(false) if skipped/not cached.
pub fn remove_one(repo: &str, assume_yes: bool) -> Result<bool> {
    let canonical = crate::hf::resolve_alias(repo);
    let dir = hf_cache_root().join(format!("models--{}", canonical.replace('/', "--")));
    if !dir.exists() {
        println!("{}  not cached", style(canonical).dim());
        return Ok(false);
    }
    let size = dir_size(&dir).unwrap_or(0);
    println!(
        "{} {}  ({})",
        style("?").yellow(),
        style(canonical).bold(),
        style(human_bytes(size)).dim()
    );
    if !assume_yes {
        print!("  remove? [y/N] ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        let ans = line.trim().to_lowercase();
        if !(ans == "y" || ans == "yes") {
            println!("  {}", style("skipped").dim());
            return Ok(false);
        }
    }
    std::fs::remove_dir_all(&dir)?;
    println!("  {}", style("removed").red());
    Ok(true)
}

pub fn remove_many(repos: &[String], assume_yes: bool) -> Result<()> {
    let mut removed = 0usize;
    for r in repos {
        if remove_one(r, assume_yes)? {
            removed += 1;
        }
    }
    println!("{} model(s) removed", removed);
    Ok(())
}

pub fn dir_size(p: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    let entries = match std::fs::read_dir(p) {
        Ok(it) => it,
        Err(_) => return Ok(0),
    };
    for entry in entries.flatten() {
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            total = total.saturating_add(dir_size(&entry.path()).unwrap_or(0));
        } else {
            total = total.saturating_add(meta.len());
        }
    }
    Ok(total)
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.1} {}", UNITS[u])
}
