//! MAP-2 **L4** — biome assignment. A per-pixel decision over elevation, the L3
//! distance-to-sea, a latitude proxy, the map climate, and the spec's `regions`
//! → a biome → an RGB colour (RFC Appendix B palette). Pure + deterministic.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use noise::{NoiseFn, Perlin};
use std::path::Path;

use super::coastline::Coastline;
use super::engine::{HeightField, resolve_simple};
use super::spec::MapSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Biome {
    Sea,
    Beach,
    CoastalPlain,
    Grassland,
    TemperateForest,
    BorealForest,
    TropicalForest,
    Tundra,
    AlpineSnow,
    DesertSandy,
    DesertRocky,
    Savanna,
    Wetland,
    MountainHigh,
    MountainExtreme,
    Volcanic,
}

impl Biome {
    /// RFC Appendix B palette.
    pub fn rgb(self) -> [u8; 3] {
        match self {
            Biome::Sea => [0xa8, 0xc8, 0xe8],
            Biome::Beach => [0xe8, 0xdd, 0xb8],
            Biome::CoastalPlain => [0xd4, 0xe8, 0xb8],
            Biome::Grassland => [0xc8, 0xdc, 0x98],
            Biome::TemperateForest => [0x78, 0xa8, 0x58],
            Biome::BorealForest => [0x4a, 0x78, 0x48],
            Biome::TropicalForest => [0x38, 0x88, 0x38],
            Biome::Tundra => [0xb8, 0xc8, 0xc8],
            Biome::AlpineSnow => [0xe8, 0xe8, 0xe8],
            Biome::DesertSandy => [0xe8, 0xc8, 0x78],
            Biome::DesertRocky => [0xc8, 0xa8, 0x58],
            Biome::Savanna => [0xc8, 0xb8, 0x78],
            Biome::Wetland => [0x78, 0x88, 0x58],
            Biome::MountainHigh => [0x88, 0x88, 0x78],
            Biome::MountainExtreme => [0xc8, 0xc8, 0xb8],
            Biome::Volcanic => [0x5a, 0x4a, 0x42],
        }
    }

    /// Map a spec region's `biome` string → a Biome (best-effort).
    pub fn from_region_str(s: &str) -> Option<Biome> {
        Some(match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "grassland" | "plains" => Biome::Grassland,
            "temperate_forest" | "forest" => Biome::TemperateForest,
            "boreal_forest" | "taiga" => Biome::BorealForest,
            "tropical_forest" | "jungle" | "rainforest" => Biome::TropicalForest,
            "tundra" => Biome::Tundra,
            "desert" | "desert_sandy" | "sand" => Biome::DesertSandy,
            "desert_rocky" | "badlands" => Biome::DesertRocky,
            "savanna" => Biome::Savanna,
            "wetland" | "swamp" | "marsh" => Biome::Wetland,
            "volcanic" | "lava" | "burnlands" => Biome::Volcanic,
            "alpine" | "snow" => Biome::AlpineSnow,
            "coastal_plain" => Biome::CoastalPlain,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BiomeMap {
    pub width: u32,
    pub height: u32,
    pub biome: Vec<Biome>,
}

/// A resolved region: center (px), biome, and a coverage-derived influence radius.
struct RegionField {
    cx: f32,
    cy: f32,
    biome: Biome,
    radius: f32,
}

impl BiomeMap {
    pub fn compute(spec: &MapSpec, hf: &HeightField, coast: &Coastline, seed: u64) -> BiomeMap {
        let (w, h) = (hf.width, hf.height);
        let extent = w.max(h) as f32;
        // Each region influences a disc whose radius grows with its `coverage`
        // (so a small region doesn't swallow the map via a Voronoi bisector).
        let regions: Vec<RegionField> = spec
            .regions
            .iter()
            .filter_map(|r| {
                let c = resolve_simple(&r.anchor)?;
                let b = Biome::from_region_str(&r.biome)?;
                let cov = r.coverage.clamp(0.05, 1.0);
                Some(RegionField { cx: c.0 * w as f32, cy: c.1 * h as f32, biome: b, radius: cov.sqrt() * 0.7 * extent })
            })
            .collect();

        // Wiggle region boundaries with low-frequency noise so they aren't straight.
        let perlin = Perlin::new(seed as u32);
        let jitter = (w.min(h) as f32) * 0.09;
        let freq = 6.0 / extent as f64;

        let mut biome = vec![Biome::Sea; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                if coast.sea[i] {
                    biome[i] = Biome::Sea;
                    continue;
                }
                let elev = hf.data[i];
                let cdist = coast.coast_dist[i];
                let lat = y as f32 / h.max(1) as f32; // 0 = north (cold), 1 = south (warm)
                let jx = x as f32 + perlin.get([x as f64 * freq, y as f64 * freq]) as f32 * jitter;
                let jy = y as f32 + perlin.get([x as f64 * freq + 31.7, y as f64 * freq + 11.3]) as f32 * jitter;
                let base = region_at(&regions, jx, jy)
                    .unwrap_or_else(|| climate_default(spec.climate.as_deref(), lat));
                biome[i] = classify(elev, cdist, base);
            }
        }
        BiomeMap { width: w, height: h, biome }
    }

    pub fn save_png(&self, path: &Path) -> Result<()> {
        let (w, h) = (self.width, self.height);
        let mut img = RgbImage::new(w, h);
        for (i, b) in self.biome.iter().enumerate() {
            img.put_pixel((i as u32) % w, (i as u32) / w, Rgb(b.rgb()));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        img.save(path).with_context(|| format!("writing biome map {}", path.display()))
    }
}

/// Elevation + coast overrides on top of the base climate/region biome:
/// peaks → snow/mountain, shoreline → beach/coastal-plain, else the base.
fn classify(elev: f32, coast_dist: f32, base: Biome) -> Biome {
    if elev > 0.9 {
        Biome::AlpineSnow
    } else if elev > 0.8 {
        Biome::MountainExtreme
    } else if elev > 0.7 {
        Biome::MountainHigh
    } else if coast_dist < 0.03 {
        Biome::Beach
    } else if coast_dist < 0.07 {
        Biome::CoastalPlain
    } else {
        base
    }
}

/// The nearest region whose influence disc contains the cell, or None (→ the
/// climate default fills the gaps between regions).
fn region_at(regions: &[RegionField], x: f32, y: f32) -> Option<Biome> {
    regions
        .iter()
        .filter(|r| ((r.cx - x).powi(2) + (r.cy - y).powi(2)).sqrt() <= r.radius)
        .min_by(|a, b| {
            let da = (a.cx - x).powi(2) + (a.cy - y).powi(2);
            let db = (b.cx - x).powi(2) + (b.cy - y).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.biome)
}

/// Default biome when the spec names no regions — from climate + latitude.
fn climate_default(climate: Option<&str>, lat: f32) -> Biome {
    let c = climate.unwrap_or("temperate").to_ascii_lowercase();
    if c.contains("arid") || c.contains("desert") {
        Biome::DesertSandy
    } else if c.contains("tropical") {
        Biome::TropicalForest
    } else if c.contains("arctic") || c.contains("polar") || c.contains("tundra") {
        Biome::Tundra
    } else if lat < 0.2 {
        Biome::BorealForest
    } else if lat > 0.85 {
        Biome::Grassland
    } else {
        Biome::TemperateForest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::coastline::{Coastline, DEFAULT_SEA_LEVEL};
    use crate::map::engine::GeoCanvas;
    use crate::map::spec::{Anchor, MapSpec, MountainRange, RegionSpec};

    fn island_biomes() -> BiomeMap {
        let mut spec = MapSpec::minimal("The Isle", 2, 2, 1);
        spec.climate = Some("temperate maritime".into());
        spec.terrain.mountain_ranges.push(MountainRange {
            id: "spine".into(),
            name: None,
            anchor: Anchor::Cardinal { position: "center".into() },
            orientation: "north-south".into(),
            length_fraction: 0.6,
            height: "extreme".into(),
        });
        spec.regions.push(RegionSpec {
            id: "west".into(),
            name: None,
            biome: "temperate_forest".into(),
            anchor: Anchor::Cardinal { position: "west".into() },
            coverage: 0.5,
            political: None,
        });
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        BiomeMap::compute(&spec, &hf, &coast, 42)
    }

    #[test]
    fn palette_covers_every_biome() {
        for b in [Biome::Sea, Biome::AlpineSnow, Biome::Volcanic, Biome::TemperateForest] {
            assert_ne!(b.rgb(), [0, 0, 0]);
        }
        assert_eq!(Biome::from_region_str("temperate_forest"), Some(Biome::TemperateForest));
        assert_eq!(Biome::from_region_str("volcanic"), Some(Biome::Volcanic));
        assert_eq!(Biome::from_region_str("nonsense"), None);
    }

    #[test]
    fn island_biomes_are_plausible() {
        let bm = island_biomes();
        let (w, h) = (bm.width, bm.height);
        // Offshore is sea; the central peak is snow/mountain (extreme ridge).
        assert_eq!(bm.biome[0], Biome::Sea);
        let center = bm.biome[((h / 2) * w + w / 2) as usize];
        assert!(matches!(center, Biome::AlpineSnow | Biome::MountainExtreme | Biome::MountainHigh), "got {center:?}");
        // Some forest/beach exists on the island.
        assert!(bm.biome.iter().any(|&b| b == Biome::Beach));
        assert!(bm.biome.iter().any(|&b| matches!(b, Biome::TemperateForest)));
    }

    #[test]
    fn biome_is_deterministic() {
        assert_eq!(island_biomes().biome, island_biomes().biome);
    }
}
