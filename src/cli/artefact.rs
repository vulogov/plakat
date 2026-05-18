//! `plakat artefact` subcommand — inspect the artefact library.
//!
//! Mirrors the shape of `plakat style {list, show, init, probe}` so
//! users who know that surface find this one familiar.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use console::style;

use crate::artefacts::ArtefactLibrary;

const ARTEFACT_LIBRARY_DEFAULT: &str = "assets/artefact_library";

#[derive(ClapArgs, Debug)]
pub struct ArtefactArgs {
    #[command(subcommand)]
    pub op: ArtefactOp,

    /// Override the bundled artefact library directory.
    #[arg(long, value_name = "DIR", global = true)]
    pub library: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum ArtefactOp {
    /// List every artefact in the library, one per row.
    List(ListArgs),
    /// Show full info for one artefact: category, natural zone,
    /// natural size, anchor, license, file path.
    Show(ShowArgs),
}

#[derive(ClapArgs, Debug)]
pub struct ListArgs {
    /// Filter by category prefix.
    #[arg(long, value_name = "CAT")]
    pub category: Option<String>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Text)]
    pub format: OutFormat,
}

#[derive(ClapArgs, Debug)]
pub struct ShowArgs {
    /// Artefact name (use `plakat artefact list` for available names).
    pub name: String,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Text)]
    pub format: OutFormat,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum OutFormat {
    Text,
    Json,
}

pub async fn run(args: ArtefactArgs) -> Result<()> {
    let library_dir = args
        .library
        .clone()
        .unwrap_or_else(|| PathBuf::from(ARTEFACT_LIBRARY_DEFAULT));

    match args.op {
        ArtefactOp::List(a) => list_cmd(a, &library_dir),
        ArtefactOp::Show(a) => show_cmd(a, &library_dir),
    }
}

fn list_cmd(args: ListArgs, library_dir: &Path) -> Result<()> {
    let lib = match ArtefactLibrary::load(library_dir) {
        Ok(l) => l,
        Err(e) => {
            // Library missing or unreadable. Give the user a clear path forward
            // rather than a stack trace.
            if !library_dir.exists() {
                return Err(anyhow!(
                    "artefact library not found at {}. Bundle still being assembled — \
                     pass `--library <DIR>` to point at your own library, or wait \
                     for the bundled set.",
                    library_dir.display()
                ));
            }
            return Err(e).context("loading artefact library");
        }
    };

    let entries: Vec<&crate::artefacts::Artefact> = lib
        .order
        .iter()
        .filter_map(|n| lib.artefacts.get(n))
        .filter(|a| match &args.category {
            Some(cat) => a.category.starts_with(cat.as_str()),
            None => true,
        })
        .collect();

    match args.format {
        OutFormat::Text => print_list_text(&entries),
        OutFormat::Json => print_list_json(&entries)?,
    }
    Ok(())
}

fn print_list_text(entries: &[&crate::artefacts::Artefact]) {
    if entries.is_empty() {
        println!("(library is empty)");
        return;
    }
    let name_w = entries.iter().map(|a| a.name.len()).max().unwrap_or(2).max(4);
    let cat_w = entries
        .iter()
        .map(|a| a.category.len())
        .max()
        .unwrap_or(2)
        .max(8);

    println!(
        "{:<name_w$}  {:<cat_w$}  {:<14}  {:<6}  {}",
        "Name",
        "Category",
        "Natural zone",
        "Size%",
        "Anchor",
        name_w = name_w,
        cat_w = cat_w,
    );
    println!(
        "{}  {}  {}  {}  {}",
        "─".repeat(name_w),
        "─".repeat(cat_w),
        "─".repeat(14),
        "─".repeat(6),
        "─".repeat(20),
    );
    for a in entries {
        let anchor_desc = format!("({:.2},{:.2})", a.anchor.x, a.anchor.y);
        println!(
            "{:<name_w$}  {:<cat_w$}  {:<14}  {:<6.2}  {}",
            a.name,
            a.category,
            a.natural_zone.display(),
            a.natural_size_pct,
            anchor_desc,
            name_w = name_w,
            cat_w = cat_w,
        );
    }
    println!();
    println!("{} artefact(s) total.", entries.len());
}

fn print_list_json(entries: &[&crate::artefacts::Artefact]) -> Result<()> {
    let arr: Vec<serde_json::Value> = entries
        .iter()
        .map(|a| {
            serde_json::json!({
                "name": a.name,
                "category": a.category,
                "natural_zone": a.natural_zone.display(),
                "natural_size_pct": a.natural_size_pct,
                "anchor": { "x": a.anchor.x, "y": a.anchor.y },
                "license": a.license,
                "license_url": a.license_url,
                "tags": a.tags,
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&arr)?);
    Ok(())
}

fn show_cmd(args: ShowArgs, library_dir: &Path) -> Result<()> {
    let lib = ArtefactLibrary::load(library_dir)
        .with_context(|| format!("loading artefact library {}", library_dir.display()))?;
    let a = lib.get(&args.name)?;

    match args.format {
        OutFormat::Text => print_show_text(a),
        OutFormat::Json => {
            let json = serde_json::json!({
                "name": a.name,
                "category": a.category,
                "path": a.path,
                "natural_zone": a.natural_zone.display(),
                "natural_size_pct": a.natural_size_pct,
                "anchor": { "x": a.anchor.x, "y": a.anchor.y },
                "license": a.license,
                "license_url": a.license_url,
                "tags": a.tags,
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
    }
    Ok(())
}

fn print_show_text(a: &crate::artefacts::Artefact) {
    println!("{}:            {}", style("Name").bold(), a.name);
    println!("{}:        {}", style("Category").bold(), a.category);
    println!("{}:            {}", style("Path").bold(), a.path.display());
    println!(
        "{}:    {}",
        style("Natural zone").bold(),
        a.natural_zone.display()
    );
    println!(
        "{}: {:.2} (fraction of zone height)",
        style("Natural size").bold(),
        a.natural_size_pct
    );
    println!(
        "{}:          (x={:.2}, y={:.2}) — fraction of artefact's own size",
        style("Anchor").bold(),
        a.anchor.x,
        a.anchor.y
    );
    if let Some(lic) = &a.license {
        println!("{}:         {}", style("License").bold(), lic);
    }
    if let Some(url) = &a.license_url {
        println!("{}:     {}", style("License URL").bold(), url);
    }
    if !a.tags.is_empty() {
        println!("{}:            {}", style("Tags").bold(), a.tags.join(", "));
    }
}
