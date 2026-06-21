//! `plakat map` — scale-aware procedural fantasy maps. MAP-1 ships the spec +
//! the LLM geographic parser; the geometry engine (MAP-2+) and the renderer
//! (MAP-3/linework, MAP-6/tiled-SD) follow. See `Documentation/RFC_MAP_COMPILE_PLAN.md`.

pub mod biome;
pub mod cache;
pub mod coastline;
pub mod composite;
pub mod engine;
pub mod export;
pub mod hydrology;
pub mod labels;
pub mod parser;
pub mod render;
pub mod render_sd;
pub mod resolver;
pub mod scenario_task;
pub mod roads;
pub mod spec;
pub mod urban;

use anyhow::{Context, Result, bail};
use spec::TileGrid;

/// Render the linework map image for `(spec, seed)`, routing by kind: a spec with
/// an `urban` block renders the **town map** (streets/blocks/wall/gates/piers), any
/// other spec the **geographic** linework map. The single entry every surface uses
/// (CLI `--map-render`, the scenario `map` task, `plakat.map.render`) so all render
/// the same image. Deterministic, no GPU.
pub fn render_map_image(spec: &spec::MapSpec, seed: u64, style: render::Style) -> Result<image::RgbImage> {
    if spec.urban.is_some() {
        let canvas = engine::GeoCanvas::from_spec(spec, seed);
        Ok(urban::StreetGraph::generate(spec, &canvas).render_town(spec))
    } else {
        render::render(spec, seed, style)
    }
}

/// Render + write the map PNG (kind-routed, see [`render_map_image`]).
pub fn save_map_image(spec: &spec::MapSpec, seed: u64, style: render::Style, path: &std::path::Path) -> Result<()> {
    let img = render_map_image(spec, seed, style)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(path).with_context(|| format!("writing map {}", path.display()))
}

/// Resolve a `--map-scale` alias → (scale_tier, default tile grid).
pub fn scale_alias(name: &str) -> Option<(u8, TileGrid)> {
    let g = |cols, rows| TileGrid { cols, rows };
    Some(match name.to_ascii_lowercase().replace('_', "-").as_str() {
        "block" => (10, g(1, 1)),
        "district" => (11, g(2, 2)),
        "settlement" | "town" => (12, g(4, 4)),
        "city" => (0, g(1, 1)),
        "vicinity" => (1, g(2, 2)),
        "coast" => (2, g(3, 3)),
        "region" => (3, g(4, 4)),
        "inland-sea" => (4, g(6, 6)),
        "hemisphere" => (5, g(8, 8)),
        _ => return None,
    })
}

/// Parse a `CxR` tile grid (e.g. `"4x2"`), validating 1..=8 on each axis.
pub fn parse_tiles(s: &str) -> Result<TileGrid> {
    let (c, r) = s
        .split_once(['x', 'X', '*'])
        .ok_or_else(|| anyhow::anyhow!("--map-tiles must be CxR (e.g. 4x4), got {s:?}"))?;
    let cols: u32 = c.trim().parse().map_err(|_| anyhow::anyhow!("--map-tiles cols {c:?} not a number"))?;
    let rows: u32 = r.trim().parse().map_err(|_| anyhow::anyhow!("--map-tiles rows {r:?} not a number"))?;
    if !(1..=8).contains(&cols) || !(1..=8).contains(&rows) {
        bail!("--map-tiles {s}: cols/rows must each be 1..=8");
    }
    Ok(TileGrid { cols, rows })
}

/// Resolve grid + tier from `--map-tiles` / `--map-scale`. `--map-tiles` wins for
/// the grid (keeping the alias's tier if both are given).
pub fn resolve_scale(tiles: Option<&str>, scale: Option<&str>) -> Result<(Option<TileGrid>, Option<u8>)> {
    let mut grid = None;
    let mut tier = None;
    if let Some(s) = scale {
        let (t, g) = scale_alias(s).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown --map-scale {s:?} \
                 (block|district|settlement|town|city|vicinity|coast|region|inland-sea|hemisphere)"
            )
        })?;
        grid = Some(g);
        tier = Some(t);
    }
    if let Some(t) = tiles {
        grid = Some(parse_tiles(t)?);
    }
    Ok((grid, tier))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_aliases_map_to_tier_and_grid() {
        assert_eq!(scale_alias("city"), Some((0, TileGrid { cols: 1, rows: 1 })));
        assert_eq!(scale_alias("region"), Some((3, TileGrid { cols: 4, rows: 4 })));
        assert_eq!(scale_alias("inland-sea"), Some((4, TileGrid { cols: 6, rows: 6 })));
        assert_eq!(scale_alias("inland_sea"), Some((4, TileGrid { cols: 6, rows: 6 })));
        assert_eq!(scale_alias("settlement"), Some((12, TileGrid { cols: 4, rows: 4 })));
        assert!(scale_alias("galaxy").is_none());
    }

    #[test]
    fn parses_tile_grids() {
        assert_eq!(parse_tiles("4x4").unwrap(), TileGrid { cols: 4, rows: 4 });
        assert_eq!(parse_tiles("4x2").unwrap(), TileGrid { cols: 4, rows: 2 });
        assert!(parse_tiles("9x1").is_err(), "out of 1..8");
        assert!(parse_tiles("nope").is_err());
    }

    #[test]
    fn tiles_override_scale_grid_but_keep_tier() {
        let (grid, tier) = resolve_scale(Some("2x2"), Some("region")).unwrap();
        assert_eq!(grid, Some(TileGrid { cols: 2, rows: 2 }), "tiles win for the grid");
        assert_eq!(tier, Some(3), "alias tier retained");
        // scale alone:
        let (g2, t2) = resolve_scale(None, Some("city")).unwrap();
        assert_eq!(g2, Some(TileGrid { cols: 1, rows: 1 }));
        assert_eq!(t2, Some(0));
        // neither:
        assert_eq!(resolve_scale(None, None).unwrap(), (None, None));
    }
}
