//! `plakat inspect` — dump safetensors tensor metadata.
//!
//! Lists every tensor name + shape + dtype in a safetensors file.
//! Useful when a weight load fails and you want to see what's actually
//! in the file vs what the model expected.

use anyhow::{Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

#[derive(ClapArgs, Debug)]
pub struct InspectArgs {
    /// Path to the .safetensors file to inspect.
    pub file: PathBuf,

    /// Filter tensor names by substring (case-sensitive). Useful for
    /// large files like UNets where the full key list is overwhelming.
    /// Example: `--filter bn1` to find all BatchNorm-named tensors.
    #[arg(long)]
    pub filter: Option<String>,

    /// Limit number of entries printed (default: all). Combine with
    /// `--filter` for tighter output on huge files.
    #[arg(long)]
    pub limit: Option<usize>,
}

pub async fn run(args: InspectArgs) -> Result<()> {
    let st = unsafe { MmapedSafetensors::new(&args.file) }
        .with_context(|| format!("opening safetensors {}", args.file.display()))?;

    // candle exposes `tensors()` as `Vec<(String, TensorView)>` where
    // TensorView is from a private candle alias. Project to plain types
    // (String, Vec<usize>, String-for-dtype) immediately so we don't
    // name the private alias anywhere.
    let mut names: Vec<(String, Vec<usize>, String)> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec(), format!("{:?}", view.dtype())))
        .collect();
    names.sort_by(|a, b| a.0.cmp(&b.0));

    let total = names.len();
    let filtered: Vec<_> = match args.filter.as_deref() {
        Some(needle) => names.into_iter().filter(|(n, _, _)| n.contains(needle)).collect(),
        None => names,
    };
    let to_show: usize = args.limit.unwrap_or(filtered.len()).min(filtered.len());

    println!(
        "\n{}  {}",
        style("inspect").yellow().bold(),
        args.file.display()
    );
    println!(
        "  {} total tensors, showing {}{}",
        style(total).bold(),
        to_show,
        args.filter
            .as_ref()
            .map(|f| format!(" (filter: {f:?})"))
            .unwrap_or_default(),
    );
    println!();

    println!(
        "  {:<60}  {:<10}  shape",
        style("name").dim(),
        style("dtype").dim()
    );
    println!("  {}", style("─".repeat(80)).dim());

    for (name, shape, dtype) in filtered.iter().take(to_show) {
        println!("  {name:<60}  {dtype:<10}  {shape:?}");
    }

    if filtered.len() > to_show {
        println!(
            "\n  {} {} more tensor(s) hidden by --limit",
            style("…").dim(),
            filtered.len() - to_show
        );
    }
    println!();
    Ok(())
}
