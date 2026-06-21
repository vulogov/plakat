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

    /// MAP-2: write land/sea + coastline (L3).
    #[arg(long = "map-dump-coast", value_name = "PATH")]
    pub dump_coast: Option<PathBuf>,

    /// MAP-2: write the biome map (L4).
    #[arg(long = "map-dump-biome", value_name = "PATH")]
    pub dump_biome: Option<PathBuf>,

    /// MAP-2: write resolved landmarks placed at their anchors (L5).
    #[arg(long = "map-dump-landmarks", value_name = "PATH")]
    pub dump_landmarks: Option<PathBuf>,

    /// MAP-2: write the road network (+ rivers, landmarks) (L6).
    #[arg(long = "map-dump-roads", value_name = "PATH")]
    pub dump_roads: Option<PathBuf>,

    /// MAP-2: write the assembled feature overlay — the complete composited map
    /// (biome + coast + rivers + roads + landmarks) (L7).
    #[arg(long = "map-dump-features", value_name = "PATH")]
    pub dump_features: Option<PathBuf>,

    /// MAP-5: write the urban street graph (U0) — wall, gates, arterials, ring
    /// road, minor grid — for a city/town-scale spec.
    #[arg(long = "map-dump-streets", value_name = "PATH")]
    pub dump_streets: Option<PathBuf>,

    /// MAP-3: render the complete styled, labelled map (the headline output:
    /// terrain + coast + rivers + roads + labelled landmarks + compass/scale/legend).
    #[arg(long = "map-render", value_name = "PATH")]
    pub render: Option<PathBuf>,

    /// MAP-3: cartographic style for `--map-render`: parchment|inked|blueprint.
    #[arg(long = "map-style", default_value = "parchment")]
    pub style: String,

    /// MAP-3b: export the map geometry as GeoJSON (coast/rivers/roads/landmarks,
    /// normalized 0–1 north-up).
    #[arg(long = "map-export-geojson", value_name = "PATH")]
    pub export_geojson: Option<PathBuf>,

    /// MAP-3b: export the map as a standalone SVG (scalable linework + labels).
    #[arg(long = "map-export-svg", value_name = "PATH")]
    pub export_svg: Option<PathBuf>,

    /// MAP-6: render a **painted** map via SD img2img + Canny ControlNet over the
    /// styled base, then re-composite labels. Requires a GPU build (downloads the
    /// model on first use). The styled-base conditioning is deterministic.
    #[arg(long = "map-render-sd", value_name = "PATH")]
    pub render_sd: Option<PathBuf>,

    /// MAP-6: just write the deterministic SD conditioning base (no GPU) — the
    /// styled map with no labels, the img2img init + Canny source.
    #[arg(long = "map-dump-conditioning", value_name = "PATH")]
    pub dump_conditioning: Option<PathBuf>,

    /// MAP-6: SD backbone for `--map-render-sd` (any plakat model: sdxl, sd15,
    /// sd21, sdxl-turbo, an HF repo, …).
    #[arg(long = "map-sd-model", default_value = "sdxl")]
    pub sd_model: String,

    /// MAP-6: LoRA for the painted render (repeatable; HF `org/name[:scale]`,
    /// `civitai:ID`, or a local path). `none` forces no LoRA. When unset, SDXL-
    /// family models default to the fantasy-map style LoRA, others to none.
    #[arg(long = "map-sd-lora", value_name = "SPEC")]
    pub sd_lora: Vec<String>,

    /// MAP-6: LoRA scale for `--map-sd-lora`.
    #[arg(long = "map-sd-lora-scale", default_value_t = 0.9)]
    pub sd_lora_scale: f32,

    /// MAP-6: img2img strength (how far the paint moves from the base geometry).
    #[arg(long = "map-sd-strength", default_value_t = 0.55)]
    pub sd_strength: f32,

    /// MAP-6: SD sampling steps.
    #[arg(long = "map-sd-steps", default_value_t = 28)]
    pub sd_steps: usize,

    /// MAP-6: SD guidance scale.
    #[arg(long = "map-sd-guidance", default_value_t = 6.5)]
    pub sd_guidance: f64,

    /// MAP-6: skip the label/furniture re-composite (raw painted output).
    #[arg(long = "map-sd-raw", default_value_t = false)]
    pub sd_raw: bool,

    /// MAP-6: tile size (px) for the multi-tile paint. A canvas larger than this
    /// paints in overlapping, feather-blended tiles (memory-safe for big maps).
    #[arg(long = "map-sd-tile", default_value_t = 1024)]
    pub sd_tile: u32,

    /// MAP-6: tile origin stride (px); smaller = more overlap = smoother seams.
    #[arg(long = "map-sd-tile-stride", default_value_t = 768)]
    pub sd_tile_stride: u32,
}

pub async fn run(args: MapArgs, device_spec: &str) -> Result<()> {
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
    if args.dump_heightmap.is_some()
        || args.dump_rivers.is_some()
        || args.dump_coast.is_some()
        || args.dump_biome.is_some()
        || args.dump_landmarks.is_some()
        || args.dump_roads.is_some()
        || args.dump_features.is_some()
    {
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
            let hydro = map::hydrology::Hydrology::compute(&hf, map::hydrology::DEFAULT_RIVER_THRESHOLD, map::coastline::DEFAULT_SEA_LEVEL);
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
        // L3/L4/L5/L6 share the coastline (biome + resolver + roads read its fields).
        if args.dump_coast.is_some()
            || args.dump_biome.is_some()
            || args.dump_landmarks.is_some()
            || args.dump_roads.is_some()
            || args.dump_features.is_some()
        {
            let coast = map::coastline::Coastline::compute(&hf, map::coastline::DEFAULT_SEA_LEVEL);
            if let Some(p) = &args.dump_coast {
                coast.render_overlay(&hf, p)?;
                println!(
                    "{}  coastline → {}  ({:.0}% land, seed {})",
                    style("✓").green(),
                    p.display(),
                    coast.land_fraction() * 100.0,
                    args.seed
                );
                did_dump = true;
            }
            if let Some(p) = &args.dump_biome {
                let bm = map::biome::BiomeMap::compute(&spec, &hf, &coast, args.seed);
                bm.save_png(p)?;
                println!("{}  biome map → {}  (seed {})", style("✓").green(), p.display(), args.seed);
                did_dump = true;
            }
            // L7: the assembled feature overlay (the complete composited map).
            if let Some(p) = &args.dump_features {
                let biome = map::biome::BiomeMap::compute(&spec, &hf, &coast, args.seed);
                let hydro = map::hydrology::Hydrology::compute(&hf, map::hydrology::DEFAULT_RIVER_THRESHOLD, map::coastline::DEFAULT_SEA_LEVEL);
                let lms = map::resolver::resolve_landmarks(&spec, &hf, &hydro, &coast)?;
                let roads = map::roads::build_roads(&spec, &hf, &coast, &hydro, &lms);
                map::composite::save_features(&hf, &coast, &biome, &hydro, &lms, &roads, p)?;
                println!(
                    "{}  feature overlay → {}  (full map: {} landmark(s), {} road(s), seed {})",
                    style("✓").green(),
                    p.display(),
                    lms.len(),
                    roads.len(),
                    args.seed
                );
                did_dump = true;
            }
            // L5/L6 share the hydrology + resolved landmarks.
            if args.dump_landmarks.is_some() || args.dump_roads.is_some() {
                let hydro = map::hydrology::Hydrology::compute(&hf, map::hydrology::DEFAULT_RIVER_THRESHOLD, map::coastline::DEFAULT_SEA_LEVEL);
                let lms = map::resolver::resolve_landmarks(&spec, &hf, &hydro, &coast)?;
                if let Some(p) = &args.dump_landmarks {
                    map::resolver::render_overlay(&hf, &coast, &lms, p)?;
                    println!(
                        "{}  landmarks → {}  ({} placed, seed {})",
                        style("✓").green(),
                        p.display(),
                        lms.len(),
                        args.seed
                    );
                    did_dump = true;
                }
                if let Some(p) = &args.dump_roads {
                    let roads = map::roads::build_roads(&spec, &hf, &coast, &hydro, &lms);
                    map::roads::render_overlay(&hf, &coast, &hydro, &lms, &roads, p)?;
                    println!(
                        "{}  roads → {}  ({} road(s), seed {})",
                        style("✓").green(),
                        p.display(),
                        roads.len(),
                        args.seed
                    );
                    did_dump = true;
                }
            }
        }
    }

    // MAP-5: the urban street graph (U0).
    if let Some(p) = &args.dump_streets {
        let canvas = map::engine::GeoCanvas::from_spec(&spec, args.seed);
        let sg = map::urban::StreetGraph::generate(&spec, &canvas);
        sg.render_overlay(p)?;
        let (nodes, edges) = sg.stats();
        println!(
            "{}  streets → {}  ({} junction(s), {} segment(s), {} gate(s), seed {})",
            style("✓").green(),
            p.display(),
            nodes,
            edges,
            sg.gates.len(),
            args.seed
        );
        did_dump = true;
    }

    // MAP-3/MAP-5: the complete labelled map. `render_map_image` routes urban specs
    // (a `urban` block) to the town renderer, geographic specs to the linework map.
    if let Some(p) = &args.render {
        let rstyle = map::render::Style::named(&args.style)?;
        map::save_map_image(&spec, args.seed, rstyle, p)?;
        let kind = if spec.urban.is_some() { "town map" } else { "map" };
        println!(
            "{}  {} → {}  ({} landmark(s), style {}, seed {})",
            style("✓").green(),
            kind,
            p.display(),
            spec.landmarks.len(),
            args.style,
            args.seed
        );
        did_dump = true;
    }

    // MAP-3b: vector export (GeoJSON / SVG) — same geometry, scalable.
    if args.export_geojson.is_some() || args.export_svg.is_some() {
        let vm = map::export::VectorMap::build(&spec, args.seed)?;
        if let Some(p) = &args.export_geojson {
            map::export::save(&vm, &spec, p)?;
            println!("{}  GeoJSON → {}  ({} landmark(s), seed {})", style("✓").green(), p.display(), vm.landmarks.len(), args.seed);
            did_dump = true;
        }
        if let Some(p) = &args.export_svg {
            map::export::save(&vm, &spec, p)?;
            println!("{}  SVG → {}  ({} coast ring(s), seed {})", style("✓").green(), p.display(), vm.coast_rings.len(), args.seed);
            did_dump = true;
        }
    }

    // MAP-6: the deterministic SD conditioning base (no GPU).
    if let Some(p) = &args.dump_conditioning {
        let rstyle = map::render::Style::named(&args.style)?;
        map::render_sd::save_conditioning(&spec, args.seed, rstyle, p)?;
        println!("{}  conditioning → {}  (styled base, no labels, seed {})", style("✓").green(), p.display(), args.seed);
        did_dump = true;
    }

    // MAP-6: the painted SD render (GPU). img2img + Canny ControlNet over the base.
    if let Some(p) = &args.render_sd {
        let rstyle = map::render::Style::named(&args.style)?;
        // LoRA resolution: explicit --map-sd-lora wins (`none` → none); else the
        // model's default (fantasy-map for SDXL-family, none otherwise).
        let loras: Vec<String> = if args.sd_lora.is_empty() {
            map::render_sd::default_loras_for_model(&args.sd_model)
        } else if args.sd_lora.iter().any(|s| s.eq_ignore_ascii_case("none")) {
            Vec::new()
        } else {
            args.sd_lora.clone()
        };
        let opts = map::render_sd::SdOptions {
            model: args.sd_model.clone(),
            loras,
            lora_scale: args.sd_lora_scale,
            strength: args.sd_strength,
            steps: args.sd_steps,
            guidance: args.sd_guidance,
            control_strength: 0.9,
            raw: args.sd_raw,
            tile_size: args.sd_tile,
            tile_stride: args.sd_tile_stride,
        };
        let device = crate::device::select(device_spec)?;
        let lora_note = if opts.loras.is_empty() { "no LoRA".to_string() } else { opts.loras.join("+") };
        println!(
            "{}  painting map with {} ({})…",
            style("→").cyan(),
            args.sd_model,
            lora_note
        );
        map::render_sd::render_sd(&spec, args.seed, rstyle, &opts, device, p).await?;
        println!(
            "{}  painted map → {}  ({}, {}, seed {})",
            style("✓").green(),
            p.display(),
            args.sd_model,
            lora_note,
            args.seed
        );
        did_dump = true;
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

    // No explicit dump → print the spec + a pointer to the headline render.
    if !did_dump {
        println!("{json}");
        eprintln!(
            "{}  --map-render PATH writes the styled, labelled linework map \
             (--map-style parchment|inked|blueprint); --map-render-sd PATH paints it \
             with SD (img2img + Canny ControlNet, --map-sd-model/--map-sd-lora); \
             --map-dump-spec saves the spec.",
            style("note:").dim()
        );
    }
    Ok(())
}
