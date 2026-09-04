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
    #[arg(help_heading = "Ranking", value_name = "PATH", required = true)]
    pub inputs: Vec<PathBuf>,

    /// Print only the top-N (after sorting by descending aesthetic score).
    #[arg(help_heading = "Ranking", long, value_name = "N")]
    pub top: Option<usize>,

    /// Emit JSON (`[{"path":…,"score":…}]`) instead of the aligned table.
    #[arg(help_heading = "Ranking", long, default_value_t = false)]
    pub json: bool,

    /// Also write each score into the image's `.json` metadata sidecar (the collection manager's
    /// sort key). No-op for images without a sidecar.
    #[arg(help_heading = "Ranking", long, default_value_t = false)]
    pub write: bool,

    /// Rank by the weight-free **AI-tell** score (0..1, lower = more human-looking) instead of the
    /// LAION aesthetic score — least-AI-looking first. No model download. `--write` records `ai_tell`.
    #[arg(help_heading = "Ranking", long, default_value_t = false)]
    pub ai_tells: bool,

    /// For each ranked image, also print a suggested `naturalize:` spec that would nudge it toward
    /// reading more natural (derived from its measured tells — a starting point, not a detector-beater).
    /// Implies `--ai-tells`. Images already reading natural get "no change".
    #[arg(help_heading = "Ranking", long, default_value_t = false)]
    pub suggest: bool,

    /// With `--suggest`: treat inputs as photographs (grain/micro texture) instead of the default
    /// hand-media (watercolor strokes + paper). Only affects the texture family of the suggestion.
    #[arg(help_heading = "Ranking", long, default_value_t = false)]
    pub photo: bool,
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

    // AI-tell ranking is weight-free — analyze each image (0..1, lower = more human-looking) and sort
    // ASCENDING (least AI first). No scorer / device load. `--suggest` implies this path.
    if args.ai_tells || args.suggest {
        let mut scored: Vec<(PathBuf, crate::naturalize::Analysis)> = Vec::with_capacity(files.len());
        for f in &files {
            let img = image::open(f)
                .with_context(|| format!("opening {}", f.display()))?
                .to_rgb8();
            scored.push((f.clone(), crate::naturalize::analyze(&img)));
        }
        // Ascending (least-AI first).
        scored.sort_by(|a, b| a.1.ai_tell.partial_cmp(&b.1.ai_tell).unwrap_or(std::cmp::Ordering::Equal));
        if args.write {
            for (p, a) in &scored {
                let _ = crate::imaging::io::patch_sidecar_ai_tell(p, a.ai_tell as f64);
            }
        }
        if let Some(n) = args.top {
            scored.truncate(n);
        }
        let is_art = !args.photo;
        if args.json {
            let items: Vec<String> = scored
                .iter()
                .map(|(p, a)| {
                    let sug = if args.suggest {
                        crate::naturalize::suggest_spec(a, is_art)
                            .map(|s| format!(", \"suggest\": {s:?}"))
                            .unwrap_or_else(|| ", \"suggest\": null".to_string())
                    } else {
                        String::new()
                    };
                    format!("  {{\"path\": {:?}, \"ai_tell\": {:.4}{sug}}}", p.display().to_string(), a.ai_tell)
                })
                .collect();
            println!("[\n{}\n]", items.join(",\n"));
        } else {
            for (rank, (p, a)) in scored.iter().enumerate() {
                let tag = if rank == 0 { style("★").green().to_string() } else { " ".to_string() };
                println!("{tag} {:6.3}  {}", a.ai_tell, p.display());
                if args.suggest {
                    match crate::naturalize::suggest_spec(a, is_art) {
                        Some(spec) => {
                            let why = crate::naturalize::suggest_reasons(a);
                            println!("        {} naturalize: {}", style("↳ suggest").cyan(), spec);
                            if !why.is_empty() {
                                println!("          {}", style(format!("({why})")).dim());
                            }
                        }
                        None => println!("        {}", style("↳ already natural — no change").dim()),
                    }
                }
            }
        }
        return Ok(());
    }

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

    // Persist scores into sidecars before truncating (so every scored image is recorded, not just
    // the printed top-N).
    if args.write {
        for (p, s) in &scored {
            let _ = crate::imaging::io::patch_sidecar_score(p, *s as f64);
        }
    }

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
