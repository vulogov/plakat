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

    /// Seed for the geometry engine (deterministic: same spec + seed → same map).
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// MAP-2: write the full-canvas tectonic heightmap PNG (L0+L1).
    #[arg(long = "map-dump-heightmap", value_name = "PATH")]
    pub dump_heightmap: Option<PathBuf>,

    /// MAP-2: write the river network over the terrain (L2 hydraulics).
    #[arg(long = "map-dump-rivers", value_name = "PATH")]
    pub dump_rivers: Option<PathBuf>,
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

    let mut did_dump = false;

    // MAP-2 geometry dumps (share the canvas + heightfield).
    if args.dump_heightmap.is_some() || args.dump_rivers.is_some() {
        let canvas = map::engine::GeoCanvas::from_spec(&spec, args.seed);
        let hf = map::engine::HeightField::generate(&spec, &canvas);
        if let Some(p) = &args.dump_heightmap {
            hf.save_gray_png(p)?;
            println!(
                "{}  heightmap → {}  ({}x{}, seed {})",
                style("✓").green(),
                p.display(),
                canvas.width,
                canvas.height,
                args.seed
            );
            did_dump = true;
        }
        if let Some(p) = &args.dump_rivers {
            let hydro = map::hydrology::Hydrology::compute(&hf, map::hydrology::DEFAULT_RIVER_THRESHOLD);
            hydro.render_overlay(&hf, p)?;
            println!(
                "{}  rivers → {}  ({} channel(s), seed {})",
                style("✓").green(),
                p.display(),
                hydro.rivers.len(),
                args.seed
            );
            did_dump = true;
        }
    }

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
            did_dump = true;
        }
        Some(_) => {
            println!("{json}"); // --map-dump-spec -
            did_dump = true;
        }
        None => {}
    }

    // No explicit dump → print the spec (the MAP-1 deliverable) + a pointer.
    if !did_dump {
        println!("{json}");
        eprintln!(
            "{}  the geometry render (linework / tiled-SD) lands in MAP-3+. \
             --map-dump-spec saves the spec; --map-dump-heightmap writes the L0+L1 heightmap.",
            style("note:").dim()
        );
    }
    Ok(())
}
