//! MAP-2 **L7** — conditioning assembly. Composite every prior layer into one
//! **feature overlay**: biome colours + a darkened coastline + rivers + roads
//! (+ bridges) + landmark markers. This is the fully-assembled geometry — the
//! image the 1.5.0 linework render styles + labels, and the basis for the MAP-6
//! tiled-SD ControlNet conditioning. Pure + deterministic.

use anyhow::Result;
use image::{Rgb, RgbImage};
use std::path::Path;

use super::biome::BiomeMap;
use super::coastline::Coastline;
use super::engine::HeightField;
use super::hydrology::Hydrology;
use super::resolver::{draw_marker, marker_rgb, ResolvedLandmark};
use super::roads::RoadGeom;

/// Composite the layers into the feature overlay (the complete map, pre-styling).
pub fn assemble(
    hf: &HeightField,
    coast: &Coastline,
    biome: &BiomeMap,
    hydro: &Hydrology,
    landmarks: &[ResolvedLandmark],
    roads: &[RoadGeom],
) -> RgbImage {
    let (w, h) = (hf.width, hf.height);
    let mut img = RgbImage::new(w, h);

    // 1) biome base.
    for (i, b) in biome.biome.iter().enumerate() {
        img.put_pixel((i as u32) % w, (i as u32) / w, Rgb(b.rgb()));
    }
    // 2) coastline (a darkened outline at the land/sea boundary).
    for y in 0..h {
        for x in 0..w {
            if coast.is_coast(x, y) {
                img.put_pixel(x, y, Rgb([0x3a, 0x2a, 0x18]));
            }
        }
    }
    // 3) rivers over the biome.
    for &(x, y) in hydro.rivers.iter().flatten() {
        img.put_pixel(x, y, Rgb([0x4a, 0x86, 0xb8]));
    }
    // 4) roads + bridges.
    for r in roads {
        for &(x, y) in &r.path {
            plot_thick(&mut img, x, y, Rgb([0x6e, 0x46, 0x22]));
        }
        for &(x, y) in &r.bridges {
            plot_thick(&mut img, x, y, Rgb([0x28, 0x1a, 0x10]));
        }
    }
    // 5) landmark markers (top).
    for lm in landmarks {
        draw_marker(&mut img, lm.x as i32, lm.y as i32, marker_rgb(&lm.kind));
    }
    img
}

/// Assemble + write the feature overlay PNG.
pub fn save_features(
    hf: &HeightField,
    coast: &Coastline,
    biome: &BiomeMap,
    hydro: &Hydrology,
    landmarks: &[ResolvedLandmark],
    roads: &[RoadGeom],
    path: &Path,
) -> Result<()> {
    let img = assemble(hf, coast, biome, hydro, landmarks, roads);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(path).map_err(|e| anyhow::anyhow!("writing feature overlay {}: {e}", path.display()))
}

fn plot_thick(img: &mut RgbImage, x: u32, y: u32, c: Rgb<u8>) {
    let (w, h) = (img.width(), img.height());
    for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
        let (nx, ny) = (x + dx, y + dy);
        if nx < w && ny < h {
            img.put_pixel(nx, ny, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::biome::{Biome, BiomeMap};
    use crate::map::coastline::{Coastline, DEFAULT_SEA_LEVEL};
    use crate::map::engine::GeoCanvas;
    use crate::map::hydrology::{Hydrology, DEFAULT_RIVER_THRESHOLD};
    use crate::map::resolver::resolve_landmarks;
    use crate::map::roads::build_roads;
    use crate::map::spec::MapSpec;

    #[test]
    fn assembled_overlay_has_every_layer() {
        let spec: MapSpec =
            serde_json::from_str(include_str!("../../corpus/map/island.spec.json")).unwrap();
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let hydro = Hydrology::compute(&hf, DEFAULT_RIVER_THRESHOLD, DEFAULT_SEA_LEVEL);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        let biome = BiomeMap::compute(&spec, &hf, &coast, 42);
        let lms = resolve_landmarks(&spec, &hf, &hydro, &coast).unwrap();
        let roads = build_roads(&spec, &hf, &coast, &hydro, &lms);

        let img = assemble(&hf, &coast, &biome, &hydro, &lms, &roads);
        assert_eq!(img.dimensions(), (hf.width, hf.height));

        // Sea biome shows; a city marker colour shows somewhere; rivers blue show.
        let has = |px: [u8; 3]| img.pixels().any(|p| p.0 == px);
        assert!(has(Biome::Sea.rgb()), "sea biome present");
        assert!(has([0xd0, 0x30, 0x30]), "a city marker present");
        assert!(has([0x4a, 0x86, 0xb8]), "rivers present");

        // Deterministic.
        let img2 = assemble(&hf, &coast, &biome, &hydro, &lms, &roads);
        assert!(img.as_raw() == img2.as_raw());
    }
}
