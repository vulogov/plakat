use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Args as ClapArgs;
use console::style;

use crate::pipelines::aesthetic::AestheticScorer;

#[derive(ClapArgs, Debug)]
pub struct RankArgs {
    /// Images and/or directories to score. Directories are scanned (non-recursively) for
    /// `.png` / `.jpg` / `.jpeg` / `.webp`.
    #[arg(value_name = "PATH", required = true)]
    pub inputs: Vec<PathBuf>,

    /// Print only the top-N (after sorting by descending aesthetic score).
    #[arg(long, value_name = "N")]
    pub top: Option<usize>,

    /// Emit JSON (`[{"path":…,"score":…}]`) instead of the aligned table.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp"];

fn is_image(p: &std::path::Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Expand inputs into a flat list of image files (dirs scanned non-recursively).
fn collect_images(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            for entry in std::fs::read_dir(p).with_context(|| format!("reading dir {}", p.display()))? {
                let path = entry?.path();
                if path.is_file() && is_image(&path) {
                    out.push(path);
                }
            }
        } else if p.is_file() {
            out.push(p.clone());
        } else {
            anyhow::bail!("no such file or directory: {}", p.display());
        }
    }
    out.sort();
    Ok(out)
}

pub async fn run(args: RankArgs, device: Device) -> Result<()> {
    let files = collect_images(&args.inputs)?;
    anyhow::ensure!(!files.is_empty(), "no images found in the given paths");

    let scorer = AestheticScorer::load(&device)
        .await
        .context("loading the aesthetic scorer (CLIP ViT-L/14 + LAION predictor)")?;

    let mut scored: Vec<(PathBuf, f32)> = Vec::with_capacity(files.len());
    for f in &files {
        let s = scorer
            .score_path(f)
            .with_context(|| format!("scoring {}", f.display()))?;
        scored.push((f.clone(), s));
    }
    // Descending by score (best first).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(n) = args.top {
        scored.truncate(n);
    }

    if args.json {
        let items: Vec<String> = scored
            .iter()
            .map(|(p, s)| format!("  {{\"path\": {:?}, \"score\": {:.4}}}", p.display().to_string(), s))
            .collect();
        println!("[\n{}\n]", items.join(",\n"));
    } else {
        for (rank, (p, s)) in scored.iter().enumerate() {
            let tag = if rank == 0 { style("★").yellow().to_string() } else { " ".to_string() };
            println!("{tag} {:6.3}  {}", s, p.display());
        }
    }
    Ok(())
}
