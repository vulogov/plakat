//! MAP-5 **U0** — the urban street graph. A city/town plan as a `petgraph`
//! undirected graph: a centre, a wall ring with gates, **arterials** radiating
//! centre→gate, a **ring road** just inside the wall, and a **minor-street grid**
//! filling the interior — all clipped to the wall. Pure fn of (spec, seed):
//! nodes/edges insert in a fixed order so the graph (and its render) is byte-stable.
//!
//! Named streets/gates in the spec label the generated arterials by bearing; the
//! graph is the substrate U1 (blocks) and the urban anchor resolver build on.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use petgraph::graph::{NodeIndex, UnGraph};
use std::path::Path;

use super::engine::GeoCanvas;
use super::spec::{MapSpec, UrbanSpec};

/// A street-graph node: a position in pixels + what it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Junction {
    pub x: f32,
    pub y: f32,
    pub kind: JunctionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JunctionKind {
    Center,
    Gate,
    Wall,
    Street,
}

/// A street segment's class (drives width + label priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreetClass {
    Arterial,
    Ring,
    Minor,
}

/// A city block — a rectangular parcel between minor streets, inside the wall.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    /// Top-left + size in pixels (the parcel, inset from the street centrelines).
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// The generated urban street graph + the canvas it lives on.
pub struct StreetGraph {
    pub width: u32,
    pub height: u32,
    pub g: UnGraph<Junction, StreetClass>,
    /// Gate nodes in spec order (centre→gate arterials), for labels + anchors.
    pub gates: Vec<NodeIndex>,
    /// The wall ring polygon (pixel points), closed; empty if the town is unwalled.
    pub wall: Vec<(f32, f32)>,
    pub center: (f32, f32),
    /// Wall radius in pixels (the built-up extent), even when unwalled.
    pub radius: f32,
    /// Minor-grid lattice spacing (px) — the block size before street inset.
    grid_spacing: f32,
    /// In-bounds lattice cells `(gx, gy)` whose 4 corners are all inside the wall —
    /// these become U1 blocks. Sorted (raster order) for determinism.
    grid_cells: Vec<(i32, i32)>,
}

impl StreetGraph {
    /// Build the street graph for `(spec, seed)`. Uses `spec.urban` when present,
    /// else a sensible default town (round wall, four cardinal gates).
    pub fn generate(spec: &MapSpec, canvas: &GeoCanvas) -> StreetGraph {
        let (w, h) = (canvas.width, canvas.height);
        let default_urban = UrbanSpec::default();
        let urban = spec.urban.as_ref().unwrap_or(&default_urban);

        let center = (w as f32 * 0.5, h as f32 * 0.5);
        let half = (w.min(h) as f32) * 0.5;
        let wall_radius_frac = urban.wall.as_ref().map(|r| r.radius).unwrap_or(0.72);
        let radius = half * wall_radius_frac.clamp(0.3, 0.95);
        let square = urban.wall.as_ref().map(|r| r.shape == "square").unwrap_or(false);

        let mut g: UnGraph<Junction, StreetClass> = UnGraph::new_undirected();
        let c_node = g.add_node(Junction { x: center.0, y: center.1, kind: JunctionKind::Center });

        // Gate bearings: spec gates (in order) or four cardinals as a default.
        let gate_bearings: Vec<f32> = if urban.gates.is_empty() {
            vec![0.0, 90.0, 180.0, 270.0] // N, E, S, W (degrees clockwise from north)
        } else {
            urban.gates.iter().map(|gt| bearing_deg(&gt.bearing)).collect()
        };

        // Gate nodes on the wall + an arterial centre→gate for each.
        let mut gates = Vec::new();
        for &deg in &gate_bearings {
            let (gx, gy) = ring_point(center, radius, deg, square);
            let gnode = g.add_node(Junction { x: gx, y: gy, kind: JunctionKind::Gate });
            g.add_edge(c_node, gnode, StreetClass::Arterial);
            gates.push(gnode);
        }

        // Wall ring polygon + ring-road nodes just inside it, connected in a loop.
        let ring_r = radius * 0.86;
        let n_ring = if square { 4 } else { 24 };
        let mut wall: Vec<(f32, f32)> = Vec::with_capacity(n_ring + 1);
        let mut ring_nodes: Vec<NodeIndex> = Vec::with_capacity(n_ring);
        for k in 0..n_ring {
            let deg = 360.0 * k as f32 / n_ring as f32;
            wall.push(ring_point(center, radius, deg, square));
            let (rx, ry) = ring_point(center, ring_r, deg, square);
            ring_nodes.push(g.add_node(Junction { x: rx, y: ry, kind: JunctionKind::Wall }));
        }
        if !wall.is_empty() {
            wall.push(wall[0]); // close the polygon
        }
        for k in 0..ring_nodes.len() {
            g.add_edge(ring_nodes[k], ring_nodes[(k + 1) % ring_nodes.len()], StreetClass::Ring);
        }

        // Minor-street grid: a square lattice clipped to the wall, each lattice point
        // a node, 4-connected to in-bounds neighbours. Deterministic raster order.
        let spacing = (radius / 5.0).max(12.0);
        let steps = (radius / spacing).floor() as i32;
        let mut grid: std::collections::HashMap<(i32, i32), NodeIndex> = std::collections::HashMap::new();
        let inside = |x: f32, y: f32| -> bool {
            let (dx, dy) = (x - center.0, y - center.1);
            if square {
                dx.abs() <= ring_r && dy.abs() <= ring_r
            } else {
                (dx * dx + dy * dy).sqrt() <= ring_r
            }
        };
        for gy in -steps..=steps {
            for gx in -steps..=steps {
                let x = center.0 + gx as f32 * spacing;
                let y = center.1 + gy as f32 * spacing;
                if inside(x, y) {
                    let n = g.add_node(Junction { x, y, kind: JunctionKind::Street });
                    grid.insert((gx, gy), n);
                }
            }
        }
        for gy in -steps..=steps {
            for gx in -steps..=steps {
                if let Some(&n) = grid.get(&(gx, gy)) {
                    if let Some(&e) = grid.get(&(gx + 1, gy)) {
                        g.add_edge(n, e, StreetClass::Minor);
                    }
                    if let Some(&s) = grid.get(&(gx, gy + 1)) {
                        g.add_edge(n, s, StreetClass::Minor);
                    }
                }
            }
        }

        // U1 substrate: lattice cells whose four corners are all inside the wall
        // become blocks. Raster order (sorted) → deterministic.
        let mut grid_cells: Vec<(i32, i32)> = Vec::new();
        for gy in -steps..=steps {
            for gx in -steps..=steps {
                let corners = [(gx, gy), (gx + 1, gy), (gx, gy + 1), (gx + 1, gy + 1)];
                if corners.iter().all(|c| grid.contains_key(c)) {
                    grid_cells.push((gx, gy));
                }
            }
        }

        StreetGraph {
            width: w,
            height: h,
            g,
            gates,
            wall,
            center,
            radius,
            grid_spacing: spacing,
            grid_cells,
        }
    }

    /// Node count (junctions) and edge count (street segments).
    pub fn stats(&self) -> (usize, usize) {
        (self.g.node_count(), self.g.edge_count())
    }

    /// U1 — the city blocks: each interior grid cell, inset from the street
    /// centrelines so the streets show between parcels. Deterministic order.
    pub fn blocks(&self) -> Vec<Block> {
        let inset = (self.grid_spacing * 0.16).clamp(1.0, 6.0);
        self.grid_cells
            .iter()
            .map(|&(gx, gy)| {
                let x0 = self.center.0 + gx as f32 * self.grid_spacing + inset;
                let y0 = self.center.1 + gy as f32 * self.grid_spacing + inset;
                let side = self.grid_spacing - 2.0 * inset;
                Block { x: x0, y: y0, w: side, h: side }
            })
            .collect()
    }

    /// Render the street graph: parchment ground, wall ring, ring road, arterials,
    /// minor grid, gate markers. Deterministic.
    pub fn render_overlay(&self, path: &Path) -> Result<()> {
        let (w, h) = (self.width, self.height);
        let mut img = RgbImage::from_pixel(w, h, Rgb([0xe9, 0xdb, 0xbf]));

        // U1 block parcels (built-up infill) under the streets.
        for b in self.blocks() {
            fill_rect(&mut img, b.x, b.y, b.w, b.h, [0xd8, 0xc6, 0xa2]);
        }

        // Edges by class: minor (thin, pale), ring (medium), arterial (bold).
        for e in self.g.edge_indices() {
            let (a, b) = self.g.edge_endpoints(e).unwrap();
            let (pa, pb) = (self.g[a], self.g[b]);
            let (color, thick) = match self.g[e] {
                StreetClass::Minor => ([0xc2, 0xb2, 0x90], 0),
                StreetClass::Ring => ([0x8a, 0x6a, 0x42], 1),
                StreetClass::Arterial => ([0x6e, 0x46, 0x22], 1),
            };
            draw_line(&mut img, pa.x, pa.y, pb.x, pb.y, color, thick);
        }
        // Wall ring (dark ink) over the streets.
        for seg in self.wall.windows(2) {
            draw_line(&mut img, seg[0].0, seg[0].1, seg[1].0, seg[1].1, [0x3a, 0x2a, 0x18], 1);
        }
        // Gate markers (red) + the centre (dark).
        for &gn in &self.gates {
            let j = self.g[gn];
            disc(&mut img, j.x as i32, j.y as i32, 3, [0xd0, 0x30, 0x30]);
        }
        disc(&mut img, self.center.0 as i32, self.center.1 as i32, 3, [0x28, 0x1a, 0x10]);

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        img.save(path).with_context(|| format!("writing street graph {}", path.display()))
    }
}

/// Cardinal/intercardinal bearing word → degrees clockwise from north.
fn bearing_deg(s: &str) -> f32 {
    match s.to_ascii_lowercase().replace([' ', '_'], "-").as_str() {
        "north" | "n" => 0.0,
        "northeast" | "north-east" | "ne" => 45.0,
        "east" | "e" => 90.0,
        "southeast" | "south-east" | "se" => 135.0,
        "south" | "s" => 180.0,
        "southwest" | "south-west" | "sw" => 225.0,
        "west" | "w" => 270.0,
        "northwest" | "north-west" | "nw" => 315.0,
        _ => 0.0,
    }
}

/// A point on the wall ring at `deg` (clockwise from north), round or square.
fn ring_point(center: (f32, f32), radius: f32, deg: f32, square: bool) -> (f32, f32) {
    let rad = (deg - 90.0).to_radians(); // 0°=north → -y
    let (cs, sn) = (rad.cos(), rad.sin());
    if square {
        // Project onto the square: scale so the larger axis component hits `radius`.
        let m = cs.abs().max(sn.abs()).max(1e-6);
        (center.0 + radius * cs / m, center.1 + radius * sn / m)
    } else {
        (center.0 + radius * cs, center.1 + radius * sn)
    }
}

/// Bresenham line with an optional 1px thickening (`thick`=1 → 2px-ish).
fn draw_line(img: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, c: [u8; 3], thick: i32) {
    let (mut x0, mut y0) = (x0.round() as i32, y0.round() as i32);
    let (x1, y1) = (x1.round() as i32, y1.round() as i32);
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let mut err = dx + dy;
    loop {
        for ty in 0..=thick {
            for tx in 0..=thick {
                put(img, x0 + tx, y0 + ty, c);
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn fill_rect(img: &mut RgbImage, x: f32, y: f32, w: f32, h: f32, c: [u8; 3]) {
    let (x0, y0) = (x.round() as i32, y.round() as i32);
    let (x1, y1) = ((x + w).round() as i32, (y + h).round() as i32);
    for py in y0..y1 {
        for px in x0..x1 {
            put(img, px, py, c);
        }
    }
}

fn disc(img: &mut RgbImage, cx: i32, cy: i32, r: i32, c: [u8; 3]) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                put(img, cx + dx, cy + dy, c);
            }
        }
    }
}

fn put(img: &mut RgbImage, x: i32, y: i32, c: [u8; 3]) {
    if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
        img.put_pixel(x as u32, y as u32, Rgb(c));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::spec::MapSpec;

    fn town() -> MapSpec {
        // A walled town at the district scale.
        let src = r#"{
          "version": 2, "name": "Saltmere Town", "scale_tier": 11,
          "tile_grid": { "cols": 2, "rows": 2 },
          "terrain": {}, "water": {},
          "urban": {
            "wall": { "name": "The Old Wall", "shape": "round", "radius": 0.72 },
            "gates": [
              { "id": "north_gate", "name": "North Gate", "bearing": "north" },
              { "id": "harbor_gate", "name": "Harbor Gate", "bearing": "south" },
              { "id": "east_gate", "name": "East Gate", "bearing": "east" }
            ],
            "streets": [ { "id": "high_st", "name": "High Street", "kind": "arterial", "bearing": "north" } ],
            "districts": [ { "id": "market", "name": "Market Square",
              "anchor": { "kind": "cardinal", "position": "center" } } ]
          }
        }"#;
        serde_json::from_str(src).unwrap()
    }

    fn graph() -> StreetGraph {
        let spec = town();
        let canvas = GeoCanvas::from_spec(&spec, 7);
        StreetGraph::generate(&spec, &canvas)
    }

    #[test]
    fn builds_a_connected_town_graph() {
        let sg = graph();
        let (nodes, edges) = sg.stats();
        assert!(nodes > 50, "a town has many junctions (got {nodes})");
        assert!(edges > nodes, "more street segments than junctions");
        assert_eq!(sg.gates.len(), 3, "three spec gates → three arterials");
        assert!(sg.wall.len() > 3 && sg.wall.first() == sg.wall.last(), "wall is a closed ring");
    }

    #[test]
    fn arterials_reach_the_gates_on_the_wall() {
        let sg = graph();
        // Each gate sits at ~wall radius from the centre.
        for &gn in &sg.gates {
            let j = sg.g[gn];
            let d = ((j.x - sg.center.0).powi(2) + (j.y - sg.center.1).powi(2)).sqrt();
            assert!((d - sg.radius).abs() < 1.0, "gate on the wall (d={d}, r={})", sg.radius);
        }
    }

    #[test]
    fn blocks_are_inside_the_wall_and_deterministic() {
        let sg = graph();
        let blocks = sg.blocks();
        assert!(blocks.len() > 10, "a town has many blocks (got {})", blocks.len());
        // Every block centre is within the ring road radius of the centre.
        let ring_r = sg.radius * 0.86;
        for b in &blocks {
            let (cx, cy) = (b.x + b.w * 0.5, b.y + b.h * 0.5);
            let d = ((cx - sg.center.0).powi(2) + (cy - sg.center.1).powi(2)).sqrt();
            assert!(d <= ring_r, "block centre inside the wall (d={d}, r={ring_r})");
            assert!(b.w > 0.0 && b.h > 0.0, "block has positive area");
        }
        // Deterministic.
        let a: Vec<_> = graph().blocks().iter().map(|b| (b.x, b.y, b.w, b.h)).collect();
        let c: Vec<_> = graph().blocks().iter().map(|b| (b.x, b.y, b.w, b.h)).collect();
        assert_eq!(a, c);
    }

    #[test]
    fn generation_is_deterministic() {
        let a = graph();
        let b = graph();
        assert_eq!(a.stats(), b.stats());
        // Node positions identical in order.
        let pa: Vec<_> = a.g.node_indices().map(|i| (a.g[i].x, a.g[i].y)).collect();
        let pb: Vec<_> = b.g.node_indices().map(|i| (b.g[i].x, b.g[i].y)).collect();
        assert_eq!(pa, pb);
    }
}
