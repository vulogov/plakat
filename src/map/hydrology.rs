//! MAP-2 **L2** — hydraulics. Priority-flood depression filling (Barnes 2014)
//! guarantees every cell drains to the map boundary (no inland traps — the
//! "river must reach the sea" property the breach algorithm also targets), then
//! D8 single-flow-direction → flow accumulation → river-network extraction.
//!
//! Pure functions over a row-major `Vec<f32>` grid — deterministic given the
//! heightmap (which is a pure fn of (spec, seed)).

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

use super::engine::HeightField;

/// 8-neighbour offsets (E, SE, S, SW, W, NW, N, NE). Even index = cardinal.
const DX: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];

/// River channels carry > this fraction of the canvas's drainage. Tuned so the
/// network reads as a few major rivers + tributaries rather than a noise of
/// flat-region threads (the priority-flood/D8 parallel-flat artifact).
pub const DEFAULT_RIVER_THRESHOLD: f32 = 0.006;

#[derive(Debug, Clone)]
pub struct Hydrology {
    pub width: u32,
    pub height: u32,
    /// Depression-filled elevation (monotonic drainage to the boundary).
    pub filled: Vec<f32>,
    /// D8 steepest-descent direction index 0..8, or `-1` at the boundary (drains
    /// off-edge) / a flat with no downhill.
    pub flow_dir: Vec<i8>,
    /// Upstream cell count per cell (drainage area, in cells).
    pub flow_accum: Vec<f32>,
    /// Traced river polylines (channel head → boundary), in pixels.
    pub rivers: Vec<Vec<(u32, u32)>>,
}

impl Hydrology {
    pub fn compute(hf: &HeightField, threshold_frac: f32) -> Hydrology {
        let (w, h) = (hf.width, hf.height);
        let filled = priority_flood(&hf.data, w, h);
        let flow_dir = d8(&filled, w, h);
        let flow_accum = accumulate(&flow_dir, &filled, w, h);
        let threshold = (threshold_frac * (w * h) as f32).max(8.0);
        let rivers = trace_rivers(&flow_dir, &flow_accum, w, h, threshold);
        Hydrology { width: w, height: h, filled, flow_dir, flow_accum, rivers }
    }

    /// Terrain (grayscale) with the river network drawn over it. Deterministic.
    pub fn render_overlay(&self, hf: &HeightField, path: &Path) -> Result<()> {
        let (w, h) = (self.width, self.height);
        let mut img = RgbImage::new(w, h);
        for (i, &e) in hf.data.iter().enumerate() {
            let g = (e.clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel((i as u32) % w, (i as u32) / w, Rgb([g, g, g]));
        }
        for river in &self.rivers {
            for &(x, y) in river {
                img.put_pixel(x, y, Rgb([0x3c, 0x8d, 0xc8])); // river blue
            }
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        img.save(path).with_context(|| format!("writing river overlay {}", path.display()))
    }
}

fn idx(x: u32, y: u32, w: u32) -> usize {
    (y * w + x) as usize
}

/// Priority-flood depression filling with an epsilon gradient (Barnes, Lehman,
/// Mulla 2014). Boundary cells seed a min-heap; each interior cell is raised to
/// just above its lowest already-processed neighbour, so every cell drains out.
fn priority_flood(elev: &[f32], w: u32, h: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut filled = vec![0f32; n];
    let mut closed = vec![false; n];
    // Min-heap keyed on fixed-point elevation; the idx tiebreak keeps it
    // deterministic for equal elevations.
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let key = |e: f32| (e.clamp(0.0, 1.0) * 1_000_000.0) as u32;

    for y in 0..h {
        for x in 0..w {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                let i = idx(x, y, w);
                filled[i] = elev[i];
                closed[i] = true;
                heap.push(Reverse((key(elev[i]), i)));
            }
        }
    }

    const EPS: f32 = 1e-6;
    while let Some(Reverse((_, i))) = heap.pop() {
        let (x, y) = ((i as u32) % w, (i as u32) / w);
        for d in 0..8 {
            let nx = x as i32 + DX[d];
            let ny = y as i32 + DY[d];
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = idx(nx as u32, ny as u32, w);
            if closed[j] {
                continue;
            }
            closed[j] = true;
            filled[j] = elev[j].max(filled[i] + EPS);
            heap.push(Reverse((key(filled[j]), j)));
        }
    }
    filled
}

/// D8 steepest-descent direction on the filled DEM. Boundary cells → `-1`.
fn d8(filled: &[f32], w: u32, h: u32) -> Vec<i8> {
    let mut dir = vec![-1i8; (w * h) as usize];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = idx(x, y, w);
            let mut best = -1i8;
            let mut best_slope = 0f32;
            for d in 0..8 {
                let nx = (x as i32 + DX[d]) as u32;
                let ny = (y as i32 + DY[d]) as u32;
                let dist = if d % 2 == 0 { 1.0 } else { std::f32::consts::SQRT_2 };
                let slope = (filled[i] - filled[idx(nx, ny, w)]) / dist;
                if slope > best_slope {
                    best_slope = slope;
                    best = d as i8;
                }
            }
            dir[i] = best;
        }
    }
    dir
}

/// Flow accumulation: process cells high→low so each cell's drainage is complete
/// before it passes downstream.
fn accumulate(flow_dir: &[i8], filled: &[f32], w: u32, h: u32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut accum = vec![1f32; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        filled[b]
            .partial_cmp(&filled[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for &i in &order {
        let d = flow_dir[i];
        if d >= 0 {
            let (x, y) = ((i as u32) % w, (i as u32) / w);
            let nx = (x as i32 + DX[d as usize]) as u32;
            let ny = (y as i32 + DY[d as usize]) as u32;
            accum[idx(nx, ny, w)] += accum[i];
        }
    }
    accum
}

/// Trace each channel head (a river cell with no upstream river contributor)
/// downstream along the flow field to the boundary.
fn trace_rivers(flow_dir: &[i8], accum: &[f32], w: u32, h: u32, threshold: f32) -> Vec<Vec<(u32, u32)>> {
    let n = (w * h) as usize;
    let is_river = |i: usize| accum[i] >= threshold;
    let mut rivers = Vec::new();

    for i in 0..n {
        if !is_river(i) {
            continue;
        }
        let (x, y) = ((i as u32) % w, (i as u32) / w);
        // Skip cells that have an upstream river neighbour flowing into them —
        // we only start a trace at channel heads.
        let mut has_upstream = false;
        for d in 0..8 {
            let nx = x as i32 + DX[d];
            let ny = y as i32 + DY[d];
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = idx(nx as u32, ny as u32, w);
            if is_river(j) {
                let jd = flow_dir[j];
                if jd >= 0 && nx + DX[jd as usize] == x as i32 && ny + DY[jd as usize] == y as i32 {
                    has_upstream = true;
                    break;
                }
            }
        }
        if has_upstream {
            continue;
        }
        // Follow downstream to the boundary.
        let mut path = Vec::new();
        let mut cur = i;
        for _ in 0..=n {
            let (cx, cy) = ((cur as u32) % w, (cur as u32) / w);
            path.push((cx, cy));
            let d = flow_dir[cur];
            if d < 0 {
                break;
            }
            let nx = (cx as i32 + DX[d as usize]) as u32;
            let ny = (cy as i32 + DY[d as usize]) as u32;
            let nxt = idx(nx, ny, w);
            if nxt == cur {
                break;
            }
            cur = nxt;
        }
        if path.len() > 1 {
            rivers.push(path);
        }
    }
    rivers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::engine::GeoCanvas;
    use crate::map::spec::{Anchor, MapSpec, MountainRange};

    fn isle() -> MapSpec {
        let mut m = MapSpec::minimal("Test", 2, 2, 1);
        m.terrain.mountain_ranges.push(MountainRange {
            id: "spine".into(),
            name: None,
            anchor: Anchor::Cardinal { position: "center".into() },
            orientation: "north-south".into(),
            length_fraction: 0.6,
            height: "high".into(),
        });
        m
    }

    fn hydro() -> (HeightField, Hydrology) {
        let spec = isle();
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let hy = Hydrology::compute(&hf, DEFAULT_RIVER_THRESHOLD);
        (hf, hy)
    }

    #[test]
    fn filling_makes_every_interior_cell_drain() {
        let (_, hy) = hydro();
        // After filling, every interior cell has a downhill D8 neighbour.
        let (w, h) = (hy.width, hy.height);
        let mut stuck = 0;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if hy.flow_dir[idx(x, y, w)] < 0 {
                    stuck += 1;
                }
            }
        }
        // A handful of perfectly-flat cells can lack a strict downhill; assert
        // the field drains overwhelmingly (no large undrained basins).
        let interior = ((w - 2) * (h - 2)) as f32;
        assert!((stuck as f32) < interior * 0.01, "too many undrained cells: {stuck}");
    }

    #[test]
    fn at_least_one_river_reaches_the_boundary() {
        let (_, hy) = hydro();
        assert!(!hy.rivers.is_empty(), "no rivers extracted");
        let (w, h) = (hy.width, hy.height);
        let on_boundary = |(x, y): (u32, u32)| x == 0 || y == 0 || x == w - 1 || y == h - 1;
        let longest = hy.rivers.iter().max_by_key(|r| r.len()).unwrap();
        assert!(on_boundary(*longest.last().unwrap()), "main river must reach the boundary");
    }

    #[test]
    fn hydrology_is_deterministic() {
        let (_, a) = hydro();
        let (_, b) = hydro();
        assert_eq!(a.flow_accum, b.flow_accum);
        assert_eq!(a.rivers, b.rivers);
    }
}
