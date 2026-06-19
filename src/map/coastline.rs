//! MAP-2 **L3** — coastline. A sea level splits the (landmass-shaped) heightmap
//! into land/sea; the land/sea mask + a distance-to-sea field feed biome (L4) and
//! the coast-relative anchors (L5). Pure, deterministic over the heightmap.
//!
//! Proper marching-squares coastline *polylines* (for a clean drawn coast) are a
//! render-time refinement; L3 ships the mask + coast cells the engine needs.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use std::collections::VecDeque;
use std::path::Path;

use super::engine::HeightField;

/// Elevation below this (on the normalized, landmass-shaped heightmap) is sea.
pub const DEFAULT_SEA_LEVEL: f32 = 0.22;

#[derive(Debug, Clone)]
pub struct Coastline {
    pub width: u32,
    pub height: u32,
    pub sea_level: f32,
    /// `true` = sea cell.
    pub sea: Vec<bool>,
    /// Distance from each cell to the nearest sea, normalized to [0,1] (0 at the
    /// coast). Feeds L4 biome (coastal vs interior).
    pub coast_dist: Vec<f32>,
}

impl Coastline {
    pub fn compute(hf: &HeightField, sea_level: f32) -> Coastline {
        let (w, h) = (hf.width, hf.height);
        let sea: Vec<bool> = hf.data.iter().map(|&e| e < sea_level).collect();
        let coast_dist = distance_to_sea(&sea, w, h);
        Coastline { width: w, height: h, sea_level, sea, coast_dist }
    }

    /// A land cell touching the sea (8-connected) — the coastline.
    pub fn is_coast(&self, x: u32, y: u32) -> bool {
        let i = (y * self.width + x) as usize;
        if self.sea[i] {
            return false;
        }
        for (dx, dy) in NEIGHBORS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                continue;
            }
            if self.sea[(ny as u32 * self.width + nx as u32) as usize] {
                return true;
            }
        }
        false
    }

    /// Land area as a fraction of the canvas (sanity / topology hint).
    pub fn land_fraction(&self) -> f32 {
        let land = self.sea.iter().filter(|&&s| !s).count();
        land as f32 / self.sea.len().max(1) as f32
    }

    /// Land (grayscale by elevation) + sea (blue, darker where deeper) + a dark
    /// coastline. Deterministic encode.
    pub fn render_overlay(&self, hf: &HeightField, path: &Path) -> Result<()> {
        let (w, h) = (self.width, self.height);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let px = if self.is_coast(x, y) {
                    Rgb([0x4a, 0x35, 0x1c]) // dark coast line
                } else if self.sea[i] {
                    // deeper sea → darker blue
                    let depth = ((self.sea_level - hf.data[i]) / self.sea_level.max(1e-3)).clamp(0.0, 1.0);
                    let shade = 1.0 - 0.5 * depth;
                    Rgb([(0x3c as f32 * shade) as u8, (0x8d as f32 * shade) as u8, (0xc8 as f32 * shade) as u8])
                } else {
                    // land: light terrain by elevation
                    let g = (120.0 + hf.data[i] * 135.0).clamp(0.0, 255.0) as u8;
                    Rgb([g, (g as f32 * 0.96) as u8, (g as f32 * 0.82) as u8])
                };
                img.put_pixel(x, y, px);
            }
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        img.save(path).with_context(|| format!("writing coastline {}", path.display()))
    }
}

const NEIGHBORS: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

/// Multi-source BFS distance from every cell to the nearest sea cell, normalized.
fn distance_to_sea(sea: &[bool], w: u32, h: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut dist = vec![u32::MAX; n];
    let mut q: VecDeque<usize> = VecDeque::new();
    for (i, &s) in sea.iter().enumerate() {
        if s {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = ((i as u32) % w, (i as u32) / w);
        for (dx, dy) in NEIGHBORS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = (ny as u32 * w + nx as u32) as usize;
            if dist[j] == u32::MAX {
                dist[j] = dist[i] + 1;
                q.push_back(j);
            }
        }
    }
    let max = dist.iter().filter(|&&d| d != u32::MAX).copied().max().unwrap_or(1).max(1) as f32;
    dist.iter().map(|&d| if d == u32::MAX { 1.0 } else { d as f32 / max }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::engine::GeoCanvas;
    use crate::map::spec::MapSpec;

    fn island_coast() -> (HeightField, Coastline) {
        // A bare island (tier 1 → radial landmass falloff) gives a ring of sea.
        let spec = MapSpec::minimal("The Isle", 2, 2, 1);
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let cl = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        (hf, cl)
    }

    #[test]
    fn island_has_sea_and_land() {
        let (_, cl) = island_coast();
        let lf = cl.land_fraction();
        assert!(lf > 0.1 && lf < 0.95, "island should be part land, part sea: {lf}");
        // corners are offshore.
        let (w, h) = (cl.width, cl.height);
        assert!(cl.sea[0], "top-left corner is sea");
        assert!(cl.sea[((h - 1) * w + (w - 1)) as usize], "bottom-right corner is sea");
        // center is land.
        assert!(!cl.sea[((h / 2) * w + w / 2) as usize], "center is land");
    }

    #[test]
    fn coast_distance_zero_at_sea_rising_inland() {
        let (_, cl) = island_coast();
        let (w, h) = (cl.width, cl.height);
        assert_eq!(cl.coast_dist[0], 0.0, "sea cell has distance 0");
        // center is the farthest-inland → larger coast distance than a near-edge cell.
        let center = cl.coast_dist[((h / 2) * w + w / 2) as usize];
        let near_edge = cl.coast_dist[((h / 2) * w + w / 8) as usize];
        assert!(center > near_edge, "interior is farther from the sea");
    }

    #[test]
    fn coastline_is_deterministic() {
        let (_, a) = island_coast();
        let (_, b) = island_coast();
        assert_eq!(a.sea, b.sea);
        assert_eq!(a.coast_dist, b.coast_dist);
    }
}
