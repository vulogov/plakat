//! `plakat style` subcommand — detect art style from a reference photo.
//!
//! Spike scope: only the `detect` operation. The `list` / `show` /
//! `probe` operations sketched in the design land once the catalog is
//! filled in with real LoRA mappings and trigger phrases.

use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::{DType, Device};
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use console::style;

use crate::pipelines::ip_adapter::{ImageEncoder, IPA_REPO};
use crate::style::{detect_style, encode_reference_photo, StyleCatalog};

#[derive(ClapArgs, Debug)]
pub struct StyleArgs {
    #[command(subcommand)]
    pub op: StyleOp,

    /// Override the bundled style catalog directory.
    #[arg(long, value_name = "DIR", global = true)]
    pub catalog: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum StyleOp {
    /// Detect art style from a photo. Prints top-K matches; doesn't generate.
    Detect(DetectArgs),
}

#[derive(ClapArgs, Debug)]
pub struct DetectArgs {
    /// Reference photo to detect style from.
    pub photo: PathBuf,

    /// Number of top matches to show.
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Text)]
    pub format: OutFormat,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum OutFormat {
    Text,
    Json,
}

pub async fn run(args: StyleArgs, device: Device) -> Result<()> {
    let catalog_dir = args
        .catalog
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/style_catalog"));

    match args.op {
        StyleOp::Detect(a) => detect_cmd(a, &catalog_dir, device).await,
    }
}

async fn detect_cmd(args: DetectArgs, catalog_dir: &Path, device: Device) -> Result<()> {
    let catalog = StyleCatalog::load(catalog_dir, &device)?;
    catalog.assert_encoder("clip-h-laion2b")?;

    let weights =
        crate::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors").await?;
    let encoder = ImageEncoder::load(&weights, &device, DType::F32)?;

    let emb = encode_reference_photo(&encoder, &args.photo, &device)?;
    let result = detect_style(&catalog, &emb, args.top_k)?;

    match args.format {
        OutFormat::Text => print_text(&result),
        OutFormat::Json => print_json(&result)?,
    }
    Ok(())
}

fn print_text(result: &crate::style::DetectionResult) {
    let picked = result.picked.as_deref();

    match (picked, result.ambiguous) {
        (Some(id), false) => {
            let top = &result.top[0];
            println!(
                "Detected: {} ({:.4}) {}",
                style(id).bold().cyan(),
                top.score,
                style("[picked]").green()
            );
        }
        (Some(id), true) => {
            let top = &result.top[0];
            println!(
                "Detected: {} ({:.4}) {}",
                style(id).bold().yellow(),
                top.score,
                style("[ambiguous]").yellow()
            );
            if let Some(runner_up) = result.top.get(1) {
                println!(
                    "Runner-up: {} ({:.4})",
                    style(&runner_up.style_id).bold(),
                    runner_up.score
                );
            }
        }
        (None, _) => {
            println!(
                "{}",
                style("Detected: (none above min_confidence)").red().bold()
            );
            if let Some(top) = result.top.first() {
                println!(
                    "Closest: {} ({:.4})",
                    style(&top.style_id).bold(),
                    top.score
                );
            }
        }
    }

    println!();
    println!("Top {}:", result.top.len());
    for (i, m) in result.top.iter().enumerate() {
        let marker = if Some(m.style_id.as_str()) == picked {
            style("✓ picked").green().to_string()
        } else {
            String::new()
        };
        println!(
            "  {}. {:<20} {:.4}  {}",
            i + 1,
            m.style_id,
            m.score,
            marker
        );
    }
}

fn print_json(result: &crate::style::DetectionResult) -> Result<()> {
    let value = serde_json::json!({
        "picked": result.picked,
        "ambiguous": result.ambiguous,
        "top": result.top.iter().map(|m| serde_json::json!({
            "style_id": m.style_id,
            "display_name": m.display_name,
            "score": m.score,
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
