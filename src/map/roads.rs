//! MAP-2 **L6** — infrastructure (roads). Each spec road routes between two
//! resolved landmarks via **Dijkstra on a terrain cost grid** (sea impassable,
//! mountains expensive, river crossings penalised), and a **bridge** is recorded
//! wherever a road crosses a river. Pure + deterministic over the prior layers.

use anyhow::Result;
use image::{Rgb, RgbImage};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;

use super::coastline::Coastline;
use super::engine::HeightField;
use super::hydrology::Hydrology;
use super::resolver::{draw_marker, marker_rgb, ResolvedLandmark};
use super::spec::MapSpec;

const DX: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
/// Heap key fixed-point scale (costs are O(10²–10³) over a path; u64 has room).
const KEY: f32 = 64.0;
const IMPASSABLE: f32 = 1.0e6;

#[derive(Debug, Clone)]
pub struct RoadGeom {
    pub id: String,
    pub path: Vec<(u32, u32)>,
    pub bridges: Vec<(u32, u32)>,
}

/// Route every spec road between its from/to landmarks.
pub fn build_roads(
    spec: &MapSpec,
    hf: &HeightField,
    coast: &Coastline,
    hydro: &Hydrology,
    landmarks: &[ResolvedLandmark],
) -> Vec<RoadGeom> {
    let (w, h) = (hf.width, hf.height);
    let river: HashSet<usize> =
        hydro.rivers.iter().flatten().map(|&(x, y)| (y * w + x) as usize).collect();
    let cost = build_cost(hf, coast, &river);
    let pos: HashMap<&str, (u32, u32)> =
        landmarks.iter().map(|l| (l.id.as_str(), (l.x as u32, l.y as u32))).collect();

    let mut roads = Vec::new();
    for r in &spec.infrastructure.roads {
        let (Some(&a), Some(&b)) = (pos.get(r.from.as_str()), pos.get(r.to.as_str())) else {
            continue; // endpoint not a placed landmark
        };
        if let Some(path) = dijkstra(&cost, w, h, a, b) {
            let bridges = path.iter().copied().filter(|&(x, y)| river.contains(&((y * w + x) as usize))).collect();
            roads.push(RoadGeom { id: r.id.clone(), path, bridges });
        }
    }
    roads
}

/// Traversal cost per cell: sea impassable, mountains steep, rivers a crossing tax.
fn build_cost(hf: &HeightField, coast: &Coastline, river: &HashSet<usize>) -> Vec<f32> {
    hf.data
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            if coast.sea[i] {
                IMPASSABLE
            } else {
                let mut c = 1.0 + e * 5.0; // gentle preference for lowland
                if e > 0.72 {
                    c += 30.0; // mountains are very costly → roads route around
                }
                if river.contains(&i) {
                    c += 6.0; // cross rivers reluctantly (→ short perpendicular crossings)
                }
                c
            }
        })
        .collect()
}

/// 8-connected Dijkstra on the cost grid; returns the cell path start→goal.
fn dijkstra(cost: &[f32], w: u32, h: u32, start: (u32, u32), goal: (u32, u32)) -> Option<Vec<(u32, u32)>> {
    let n = (w * h) as usize;
    let s = (start.1 * w + start.0) as usize;
    let g = (goal.1 * w + goal.0) as usize;
    if cost[s] >= IMPASSABLE || cost[g] >= IMPASSABLE {
        return None;
    }
    let mut dist = vec![f32::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    dist[s] = 0.0;
    heap.push(Reverse((0, s)));

    while let Some(Reverse((dk, u))) = heap.pop() {
        if u == g {
            break;
        }
        if dk > (dist[u] * KEY) as u64 + 1 {
            continue; // stale heap entry
        }
        let (ux, uy) = ((u as u32) % w, (u as u32) / w);
        for d in 0..8 {
            let nx = ux as i32 + DX[d];
            let ny = uy as i32 + DY[d];
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let v = (ny as u32 * w + nx as u32) as usize;
            if cost[v] >= IMPASSABLE {
                continue;
            }
            let step = if d % 2 == 0 { 1.0 } else { std::f32::consts::SQRT_2 };
            let nd = dist[u] + (cost[u] + cost[v]) * 0.5 * step;
            if nd < dist[v] {
                dist[v] = nd;
                prev[v] = u;
                heap.push(Reverse(((nd * KEY) as u64, v)));
            }
        }
    }

    if dist[g].is_infinite() {
        return None;
    }
    let mut path = vec![g];
    let mut c = g;
    while c != s {
        c = prev[c];
        if c == usize::MAX {
            return None;
        }
        path.push(c);
    }
    path.reverse();
    Some(path.into_iter().map(|i| ((i as u32) % w, (i as u32) / w)).collect())
}

/// Terrain + sea + rivers + roads (+ bridges) + landmark markers.
pub fn render_overlay(
    hf: &HeightField,
    coast: &Coastline,
    hydro: &Hydrology,
    landmarks: &[ResolvedLandmark],
    roads: &[RoadGeom],
    path: &Path,
) -> Result<()> {
    let (w, h) = (hf.width, hf.height);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let px = if coast.sea[i] {
                Rgb([0x9c, 0xbc, 0xdc])
            } else {
                let g = (140.0 + hf.data[i] * 110.0).clamp(0.0, 255.0) as u8;
                Rgb([g, (g as f32 * 0.96) as u8, (g as f32 * 0.82) as u8])
            };
            img.put_pixel(x, y, px);
        }
    }
    for &(x, y) in hydro.rivers.iter().flatten() {
        img.put_pixel(x, y, Rgb([0x3c, 0x8d, 0xc8]));
    }
    for r in roads {
        for &(x, y) in &r.path {
            plot_thick(&mut img, x, y, Rgb([0x8a, 0x5a, 0x2a])); // road brown
        }
        for &(x, y) in &r.bridges {
            plot_thick(&mut img, x, y, Rgb([0x30, 0x20, 0x14])); // bridge
        }
    }
    for lm in landmarks {
        draw_marker(&mut img, lm.x as i32, lm.y as i32, marker_rgb(&lm.kind));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(path).map_err(|e| anyhow::anyhow!("writing road overlay {}: {e}", path.display()))
}

/// Plot a 2px-ish dot so thin roads read at a glance.
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
    use crate::map::coastline::{Coastline, DEFAULT_SEA_LEVEL};
    use crate::map::engine::{GeoCanvas, HeightField};
    use crate::map::hydrology::{Hydrology, DEFAULT_RIVER_THRESHOLD};
    use crate::map::resolver::resolve_landmarks;

    fn build() -> Vec<RoadGeom> {
        let spec: MapSpec = serde_json::from_str(include_str!("../../corpus/map/island.spec.json")).unwrap();
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let hydro = Hydrology::compute(&hf, DEFAULT_RIVER_THRESHOLD);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        let lms = resolve_landmarks(&spec, &hf, &hydro, &coast).unwrap();
        build_roads(&spec, &hf, &coast, &hydro, &lms)
    }

    #[test]
    fn salt_road_connects_its_endpoints() {
        let roads = build();
        assert_eq!(roads.len(), 1, "the island spec has one road");
        let r = &roads[0];
        assert_eq!(r.id, "salt_road");
        assert!(r.path.len() > 1, "road has a routed path");
    }

    #[test]
    fn roads_avoid_the_sea() {
        let spec: MapSpec = serde_json::from_str(include_str!("../../corpus/map/island.spec.json")).unwrap();
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        for r in build() {
            for &(x, y) in &r.path {
                assert!(!coast.sea[(y * hf.width + x) as usize], "road entered the sea at {x},{y}");
            }
        }
    }

    #[test]
    fn roads_are_deterministic() {
        assert_eq!(
            build().iter().map(|r| r.path.clone()).collect::<Vec<_>>(),
            build().iter().map(|r| r.path.clone()).collect::<Vec<_>>()
        );
    }
}
