//! `plakat map` — generate a fantasy map from a prose world description.
//!
//! MAP-1 ships the LLM **parse** stage: prose → `MapSpec v2` JSON. The geometry
//! engine (MAP-2) and renderer (MAP-3+) follow; until then `--map-dump-spec`
//! writes the parsed spec and the default output prints it.

use anyhow::{Context, Result, bail};
use clap::Args as ClapArgs;
use console::style;
use std::path::PathBuf;

use crate::map::{self, parser::ParseOpts, spec::MapSpec};

#[derive(ClapArgs, Debug)]
pub struct MapArgs {
    /// Prose world description (optional when `--map-spec` is given).
    #[arg(value_name = "DESCRIPTION", default_value = "")]
    pub description: String,

    /// Tile grid `CxR` (1x1 … 8x8, non-square ok). Overrides `--map-scale`.
    #[arg(long = "map-tiles", value_name = "CxR")]
    pub tiles: Option<String>,

    /// Named scale: block|district|settlement|city|vicinity|coast|region|inland-sea|hemisphere.
    #[arg(long = "map-scale", value_name = "NAME")]
    pub scale: Option<String>,

    /// LLM provider for parsing (reuses the `--enhance` stack).
    #[arg(long = "map-provider", default_value = "auto")]
    pub provider: String,

    /// Override the built-in geographic-parser system prompt (file).
    #[arg(long = "map-system", value_name = "PATH")]
    pub system: Option<PathBuf>,

    /// SHA-256 disk cache of parsed specs (`~/.cache/plakat/map/`).
    #[arg(long = "map-cache", default_value_t = false)]
    pub cache: bool,

    /// Load a pre-written `MapSpec` JSON and skip the LLM entirely.
    #[arg(long = "map-spec", value_name = "PATH")]
    pub spec: Option<PathBuf>,

    /// Write the parsed `MapSpec` JSON to PATH (`-` = stdout).
    #[arg(long = "map-dump-spec", value_name = "PATH")]
    pub dump_spec: Option<PathBuf>,
}

pub async fn run(args: MapArgs) -> Result<()> {
    let (grid, tier) = map::resolve_scale(args.tiles.as_deref(), args.scale.as_deref())?;

    // Source the spec: load (skip LLM) or parse the description.
    let spec: MapSpec = if let Some(p) = &args.spec {
        let text = std::fs::read_to_string(p).with_context(|| format!("reading --map-spec {}", p.display()))?;
        let mut m: MapSpec = serde_json::from_str(&text)
            .with_context(|| format!("parsing MapSpec {}", p.display()))?;
        if m.version != map::spec::SPEC_VERSION {
            tracing::warn!(target: "plakat", "map: spec version {} (expected {}) — loading best-effort", m.version, map::spec::SPEC_VERSION);
        }
        if let Some(g) = grid {
            m.tile_grid = g;
        }
        if let Some(t) = tier {
            m.scale_tier = t;
        }
        m
    } else {
        if args.description.trim().is_empty() {
            bail!("provide a world description, or load one with --map-spec PATH");
        }
        let system_override = match &args.system {
            Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading --map-system {}", p.display()))?),
            None => None,
        };
        map::parser::parse(
            &args.description,
            &ParseOpts {
                provider: args.provider.clone(),
                system_override,
                tile_grid: grid,
                scale_tier: tier,
                cache: args.cache,
            },
        )
        .await?
    };

    let json = serde_json::to_string_pretty(&spec)?;
    match &args.dump_spec {
        Some(p) if p.as_os_str() != "-" => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(p, format!("{json}\n")).with_context(|| format!("writing {}", p.display()))?;
            println!(
                "{}  map spec → {}  ({} landmark(s), {}x{} tiles, tier {})",
                style("✓").green(),
                p.display(),
                spec.landmarks.len(),
                spec.tile_grid.cols,
                spec.tile_grid.rows,
                spec.scale_tier
            );
        }
        _ => {
            println!("{json}");
            if args.dump_spec.is_none() {
                eprintln!(
                    "{}  MAP-1 ships the spec + parser; the geometry engine + renderer land in MAP-2+. \
                     Save with --map-dump-spec, or feed it back via --map-spec.",
                    style("note:").dim()
                );
            }
        }
    }
    Ok(())
}
