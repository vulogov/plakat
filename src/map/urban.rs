//! MAP-5 — the urban street graph. A **medieval radio-concentric** town plan: a
//! market centre, concentric **ring roads**, **radial** streets, **arterials**
//! punching centre→gate, and an irregular **wall**. Everything is noise-warped —
//! ring radii, spoke angles, and the streets *curve* (real walled cities grew
//! organically; a square grid would be nonsense). A `petgraph` `UnGraph` holds the
//! topology (for blocks + anchors); the render bends each segment with the same
//! noise field. Pure fn of (spec, seed): fixed insert order → byte-stable.

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use noise::{NoiseFn, Perlin};
use petgraph::graph::{NodeIndex, UnGraph};
use std::f32::consts::TAU;
use std::path::Path;

use super::engine::GeoCanvas;
use super::resolver::ResolvedLandmark;
use super::spec::{Anchor, MapSpec, UrbanSpec};

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
    Ring,
    Radial,
}

/// A street segment's class (drives width + curve amplitude).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreetClass {
    Arterial,
    Ring,
    Minor,
}

/// A city block — an annular-sector parcel (4 corners) between rings + radials.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub corners: [(f32, f32); 4],
}

/// The street-plan style. Picked from `urban.layout`, else inferred from context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutStyle {
    /// Medieval radio-concentric: a market centre, ring roads, radials. Walled towns.
    Radial,
    /// Planned orthogonal grid (Roman castrum / colonial). Plains, planned towns.
    Grid,
    /// Irregular winding lanes (no global axis). Hill towns, old organic growth.
    Organic,
}

impl LayoutStyle {
    /// Parse an explicit `urban.layout` value (with aliases).
    fn parse(s: &str) -> Option<LayoutStyle> {
        match s.to_ascii_lowercase().as_str() {
            "radial" | "concentric" | "medieval" | "radiocentric" => Some(LayoutStyle::Radial),
            "grid" | "orthogonal" | "planned" | "roman" | "colonial" => Some(LayoutStyle::Grid),
            "organic" | "irregular" | "maze" | "winding" => Some(LayoutStyle::Organic),
            _ => None,
        }
    }

    /// Resolve the layout: explicit `urban.layout` wins, else infer from context —
    /// **mountainous → organic**, **walled → radial**, **plains/unwalled → grid**.
    fn resolve(spec: &MapSpec, urban: &UrbanSpec) -> LayoutStyle {
        if let Some(explicit) = urban.layout.as_deref().and_then(LayoutStyle::parse) {
            return explicit;
        }
        let elev = spec.terrain.dominant_elevation.to_ascii_lowercase();
        if elev.contains("mountain") || elev.contains("hill") || elev.contains("rugged") {
            LayoutStyle::Organic
        } else if urban.wall.is_some() {
            LayoutStyle::Radial // a walled town grew medieval-style around its core
        } else if elev.contains("plain") || elev.contains("lowland") || elev.contains("flat") {
            LayoutStyle::Grid
        } else {
            LayoutStyle::Radial
        }
    }
}

/// The generated urban street graph + the canvas it lives on.
pub struct StreetGraph {
    pub width: u32,
    pub height: u32,
    pub g: UnGraph<Junction, StreetClass>,
    pub layout: LayoutStyle,
    /// Gate nodes in spec order (centre→gate arterials), for labels + anchors.
    pub gates: Vec<NodeIndex>,
    /// The wall polygon (pixel points), closed; empty if the town is unwalled.
    pub wall: Vec<(f32, f32)>,
    pub center: (f32, f32),
    /// Built-up extent in pixels (the nominal wall radius before perturbation).
    pub radius: f32,
    /// U2 — waterfront unit vector toward open water (`None` = inland town).
    pub water_dir: Option<(f32, f32)>,
    /// U2 — piers: `(base, tip)` pixel points extending from the waterline.
    pub piers: Vec<((f32, f32), (f32, f32))>,
    /// Noise seed — the render uses it to curve street segments the same way.
    seed: u32,
    /// Per-layout street-curve amplitude scale (0 = straight grid, 1 = radial, …).
    curve_scale: f32,
    /// U1 block parcels (precomputed per layout).
    block_list: Vec<Block>,
}

/// Number of concentric ring roads + angular spokes (the organic lattice).
const N_RINGS: usize = 4;
const N_SPOKES: usize = 18;
/// Ring radii as fractions of the built-up extent (market core → outer ring road).
const RING_FRAC: [f32; N_RINGS] = [0.18, 0.42, 0.66, 0.86];

impl StreetGraph {
    /// Build the town graph for `(spec, seed)`. The street plan is `urban.layout`
    /// (explicit) or inferred from context — see [`LayoutStyle::resolve`]. Uses
    /// `spec.urban` when present, else a default walled town.
    pub fn generate(spec: &MapSpec, canvas: &GeoCanvas) -> StreetGraph {
        let (w, h) = (canvas.width, canvas.height);
        let default_urban = UrbanSpec::default();
        let urban = spec.urban.as_ref().unwrap_or(&default_urban);
        let seed = canvas.seed as u32;
        let perlin = Perlin::new(seed);
        let noise = |a: f32, b: f32| perlin.get([a as f64, b as f64]) as f32; // ~[-1,1]
        let layout = LayoutStyle::resolve(spec, urban);

        let center = (w as f32 * 0.5, h as f32 * 0.5);
        let half = (w.min(h) as f32) * 0.5;
        let radius = half * urban.wall.as_ref().map(|r| r.radius).unwrap_or(0.72).clamp(0.3, 0.95);

        // Organic wall radius at screen-angle θ: low-frequency lobes so the outline
        // bulges + dents like a real city. A grid town keeps a tamer, near-square wall.
        let wall_amp = if layout == LayoutStyle::Grid { 0.04 } else { 0.10 };
        let wall_r = |theta: f32| -> f32 {
            let (c, s) = (theta.cos(), theta.sin());
            let lobe = noise(c * 1.4 + 11.0, s * 1.4 + 11.0);
            let fine = noise(c * 5.0 + 3.0, s * 5.0 + 3.0) * 0.4;
            radius * (1.0 + wall_amp * lobe + 0.03 * fine)
        };

        let mut g: UnGraph<Junction, StreetClass> = UnGraph::new_undirected();
        let c_node = g.add_node(Junction { x: center.0, y: center.1, kind: JunctionKind::Center });

        // Streets + blocks differ by layout; wall/gates/water/piers are shared.
        let (block_list, curve_scale) = match layout {
            LayoutStyle::Radial => Self::build_radial(&mut g, c_node, center, &noise, &wall_r),
            LayoutStyle::Grid => Self::build_lattice(&mut g, center, radius, &noise, &wall_r, 0.0, 0.0),
            LayoutStyle::Organic => Self::build_lattice(&mut g, center, radius, &noise, &wall_r, 0.55, 1.4),
        };

        // Gates on the wall + a bold arterial centre→gate.
        let gate_bearings: Vec<f32> = if urban.gates.is_empty() {
            vec![0.0, 90.0, 180.0, 270.0]
        } else {
            urban.gates.iter().map(|gt| bearing_deg(&gt.bearing)).collect()
        };
        let mut gates = Vec::new();
        for &deg in &gate_bearings {
            let theta = (deg - 90.0).to_radians();
            let r = wall_r(theta);
            let p = (center.0 + r * theta.cos(), center.1 + r * theta.sin());
            let gnode = g.add_node(Junction { x: p.0, y: p.1, kind: JunctionKind::Gate });
            g.add_edge(c_node, gnode, StreetClass::Arterial);
            gates.push(gnode);
        }

        // Wall polygon (dense).
        let n_wall = 72;
        let mut wall: Vec<(f32, f32)> = Vec::with_capacity(n_wall + 1);
        for k in 0..n_wall {
            let theta = TAU * k as f32 / n_wall as f32;
            let r = wall_r(theta);
            wall.push((center.0 + r * theta.cos(), center.1 + r * theta.sin()));
        }
        if !wall.is_empty() {
            wall.push(wall[0]);
        }

        // U2 — waterfront + piers on the named edge.
        let water_dir = urban.waterfront.as_deref().map(dir_vec).filter(|d| *d != (0.0, 0.0));
        let mut piers = Vec::new();
        if let Some(d) = water_dir {
            let perp = (-d.1, d.0);
            let shore = (center.0 + d.0 * radius * 1.05, center.1 + d.1 * radius * 1.05);
            for p in &urban.piers {
                let t = (p.position.clamp(0.0, 1.0) - 0.5) * 2.0 * radius * 0.8;
                let base = (shore.0 + perp.0 * t, shore.1 + perp.1 * t);
                let tip = (base.0 + d.0 * radius * 0.22, base.1 + d.1 * radius * 0.22);
                piers.push((base, tip));
            }
        }

        StreetGraph {
            width: w,
            height: h,
            g,
            layout,
            gates,
            wall,
            center,
            radius,
            water_dir,
            piers,
            seed,
            curve_scale,
            block_list,
        }
    }

    /// Radio-concentric streets (rings + radials), noise-jittered. Returns
    /// `(blocks, curve_scale)`.
    fn build_radial(
        g: &mut UnGraph<Junction, StreetClass>,
        c_node: NodeIndex,
        center: (f32, f32),
        noise: &dyn Fn(f32, f32) -> f32,
        wall_r: &dyn Fn(f32) -> f32,
    ) -> (Vec<Block>, f32) {
        let spoke_step = TAU / N_SPOKES as f32;
        let mut nodes: Vec<Vec<NodeIndex>> = Vec::with_capacity(N_RINGS);
        let mut rings: Vec<Vec<(f32, f32)>> = Vec::with_capacity(N_RINGS);
        for (i, &frac) in RING_FRAC.iter().enumerate() {
            let mut row_n = Vec::with_capacity(N_SPOKES);
            let mut row_p = Vec::with_capacity(N_SPOKES);
            for j in 0..N_SPOKES {
                let ang_jit = noise(i as f32 * 1.7 + 1.0, j as f32 * 0.9) * spoke_step * 0.30;
                let theta = j as f32 * spoke_step + ang_jit;
                let rad_jit = 1.0 + noise(i as f32 * 0.8 + 7.0, j as f32 * 0.6 + 2.0) * 0.10;
                let r = frac * wall_r(theta) * rad_jit;
                let p = (center.0 + r * theta.cos(), center.1 + r * theta.sin());
                row_n.push(g.add_node(Junction { x: p.0, y: p.1, kind: JunctionKind::Ring }));
                row_p.push(p);
            }
            nodes.push(row_n);
            rings.push(row_p);
            if i == 0 {
                for &n in &nodes[0] {
                    g.add_edge(c_node, n, StreetClass::Minor);
                }
            }
        }
        for i in 0..N_RINGS {
            for j in 0..N_SPOKES {
                g.add_edge(nodes[i][j], nodes[i][(j + 1) % N_SPOKES], StreetClass::Ring);
                if i + 1 < N_RINGS {
                    g.add_edge(nodes[i][j], nodes[i + 1][j], StreetClass::Minor);
                }
            }
        }
        let mut blocks = Vec::new();
        for i in 0..rings.len().saturating_sub(1) {
            for j in 0..N_SPOKES {
                let jn = (j + 1) % N_SPOKES;
                blocks.push(inset_quad([rings[i][j], rings[i][jn], rings[i + 1][jn], rings[i + 1][j]]));
            }
        }
        (blocks, 1.0)
    }

    /// A square lattice clipped to the wall, each node domain-warped by `warp`
    /// (0 = straight grid, high = organic winding lanes). Returns `(blocks, curve_scale)`.
    fn build_lattice(
        g: &mut UnGraph<Junction, StreetClass>,
        center: (f32, f32),
        radius: f32,
        noise: &dyn Fn(f32, f32) -> f32,
        wall_r: &dyn Fn(f32) -> f32,
        warp: f32,
        curve_scale: f32,
    ) -> (Vec<Block>, f32) {
        let spacing = (radius / 5.0).max(12.0);
        let steps = (radius / spacing).floor() as i32;
        // True if a point is inside the wall (sample wall_r at its angle).
        let inside = |x: f32, y: f32| -> bool {
            let (dx, dy) = (x - center.0, y - center.1);
            let theta = dy.atan2(dx);
            (dx * dx + dy * dy).sqrt() <= wall_r(theta) * 0.97
        };
        let warped = |gx: i32, gy: i32| -> (f32, f32) {
            let bx = center.0 + gx as f32 * spacing;
            let by = center.1 + gy as f32 * spacing;
            if warp <= 0.0 {
                (bx, by)
            } else {
                let ox = noise(gx as f32 * 0.6 + 2.0, gy as f32 * 0.6) * warp * spacing;
                let oy = noise(gx as f32 * 0.6 + 9.0, gy as f32 * 0.6 + 5.0) * warp * spacing;
                (bx + ox, by + oy)
            }
        };
        let mut grid: std::collections::HashMap<(i32, i32), (NodeIndex, (f32, f32))> = std::collections::HashMap::new();
        for gy in -steps..=steps {
            for gx in -steps..=steps {
                let p = warped(gx, gy);
                if inside(p.0, p.1) {
                    let n = g.add_node(Junction { x: p.0, y: p.1, kind: JunctionKind::Radial });
                    grid.insert((gx, gy), (n, p));
                }
            }
        }
        for gy in -steps..=steps {
            for gx in -steps..=steps {
                if let Some(&(n, _)) = grid.get(&(gx, gy)) {
                    if let Some(&(e, _)) = grid.get(&(gx + 1, gy)) {
                        g.add_edge(n, e, StreetClass::Minor);
                    }
                    if let Some(&(s, _)) = grid.get(&(gx, gy + 1)) {
                        g.add_edge(n, s, StreetClass::Minor);
                    }
                }
            }
        }
        let mut blocks = Vec::new();
        for gy in -steps..=steps {
            for gx in -steps..=steps {
                let corners = [(gx, gy), (gx + 1, gy), (gx + 1, gy + 1), (gx, gy + 1)];
                let pts: Option<Vec<(f32, f32)>> = corners.iter().map(|c| grid.get(c).map(|&(_, p)| p)).collect();
                if let Some(p) = pts {
                    blocks.push(inset_quad([p[0], p[1], p[2], p[3]]));
                }
            }
        }
        (blocks, curve_scale)
    }

    /// Node count (junctions) and edge count (street segments).
    pub fn stats(&self) -> (usize, usize) {
        (self.g.node_count(), self.g.edge_count())
    }

    /// U1 — the city block parcels (precomputed per layout).
    pub fn blocks(&self) -> &[Block] {
        &self.block_list
    }

    /// Render the raw street graph (parchment ground + blocks + curved streets + wall
    /// + gates). Deterministic.
    pub fn render_overlay(&self, path: &Path) -> Result<()> {
        let img = self.paint(None);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        img.save(path).with_context(|| format!("writing street graph {}", path.display()))
    }

    /// Paint the town: water, block parcels, curved streets, wall, gates, piers, and
    /// (when `spec` is given) landmark markers + labels + title + frame.
    fn paint(&self, spec: Option<&MapSpec>) -> RgbImage {
        let (w, h) = (self.width, self.height);
        let perlin = Perlin::new(self.seed);
        let curve = |a: f32, b: f32| perlin.get([a as f64, b as f64]) as f32;
        let mut img = RgbImage::from_pixel(w, h, Rgb([0xe9, 0xdb, 0xbf]));

        // Water on the waterfront side (a half-plane beyond the shore line).
        if let Some(d) = self.water_dir {
            let shore = (self.center.0 + d.0 * self.radius * 1.05, self.center.1 + d.1 * self.radius * 1.05);
            for y in 0..h {
                for x in 0..w {
                    if (x as f32 - shore.0) * d.0 + (y as f32 - shore.1) * d.1 > 0.0 {
                        img.put_pixel(x, y, Rgb([0xbc, 0xcb, 0xcf]));
                    }
                }
            }
        }
        // Block parcels (built-up infill).
        for b in self.blocks() {
            fill_quad(&mut img, &b.corners, [0xd8, 0xc6, 0xa2]);
        }
        // Streets — each segment bent by the noise field so nothing runs straight.
        for e in self.g.edge_indices() {
            let (a, b) = self.g.edge_endpoints(e).unwrap();
            let (pa, pb) = (self.g[a], self.g[b]);
            let (color, thick, amp) = match self.g[e] {
                StreetClass::Minor => ([0xc2, 0xb2, 0x90], 0, 0.16),
                StreetClass::Ring => ([0x8a, 0x6a, 0x42], 1, 0.13),
                StreetClass::Arterial => ([0x6e, 0x46, 0x22], 1, 0.05),
            };
            draw_curved(&mut img, (pa.x, pa.y), (pb.x, pb.y), color, thick, amp * self.curve_scale, &curve);
        }
        // Piers over the water.
        for &(base, tip) in &self.piers {
            draw_line(&mut img, base.0, base.1, tip.0, tip.1, [0x6e, 0x46, 0x22], 1);
        }
        // Wall ring (organic, dark ink).
        for seg in self.wall.windows(2) {
            draw_line(&mut img, seg[0].0, seg[0].1, seg[1].0, seg[1].1, [0x3a, 0x2a, 0x18], 1);
        }
        // Spec landmarks at urban anchors.
        if let Some(spec) = spec {
            for lm in self.resolve_landmarks(spec) {
                super::resolver::draw_marker(&mut img, lm.x as i32, lm.y as i32, super::resolver::marker_rgb(&lm.kind));
            }
        }
        // Gate markers + centre.
        for &gn in &self.gates {
            let j = self.g[gn];
            disc(&mut img, j.x as i32, j.y as i32, 3, [0xd0, 0x30, 0x30]);
        }
        disc(&mut img, self.center.0 as i32, self.center.1 as i32, 3, [0x28, 0x1a, 0x10]);

        // Labels + furniture (only on the full town render).
        if let Some(spec) = spec {
            let ink = [0x3a, 0x2a, 0x18];
            let paper = [0xe9, 0xdb, 0xbf];
            for l in self.feature_labels(spec) {
                let tw = super::labels::text_width(&l.text, 1) as i32;
                let lx = (l.x as i32 - tw / 2).clamp(2, w as i32 - tw - 2);
                let ly = (l.y as i32 + 5).clamp(2, h as i32 - 10);
                super::labels::draw_text_haloed(&mut img, lx, ly, &l.text, 1, ink, paper);
            }
            for lm in self.resolve_landmarks(spec) {
                super::labels::draw_text_haloed(&mut img, lm.x as i32 + 6, lm.y as i32 - 2, &lm.name, 1, ink, paper);
            }
            let scale = 2;
            let tw = super::labels::text_width(&spec.name, scale) as i32;
            let th = super::labels::text_height(scale) as i32;
            let (bx0, by0) = ((w as i32 - tw) / 2 - 5, 8);
            fill_rect(&mut img, bx0 as f32, by0 as f32, (tw + 10) as f32, (th + 10) as f32, paper);
            rect_outline(&mut img, bx0, by0, bx0 + tw + 10, by0 + th + 10, ink);
            super::labels::draw_text(&mut img, bx0 + 5, by0 + 5, &spec.name, scale, ink);
            rect_outline(&mut img, 3, 3, w as i32 - 4, h as i32 - 4, ink);
            rect_outline(&mut img, 6, 6, w as i32 - 7, h as i32 - 7, ink);
        }
        img
    }

    /// MAP-5 — the complete labelled **town map** (the urban counterpart to the
    /// geographic `render::render`). Deterministic.
    pub fn render_town(&self, spec: &MapSpec) -> RgbImage {
        self.paint(Some(spec))
    }

    // ── urban anchor resolver ────────────────────────────────────────────────

    /// Resolve the spec's landmarks against the urban geometry (U0–U2). Handles the
    /// urban [`Anchor`] variants plus cardinal/canvas fallbacks; unresolvable anchors
    /// drop (no abort).
    pub fn resolve_landmarks(&self, spec: &MapSpec) -> Vec<ResolvedLandmark> {
        let urban = spec.urban.as_ref();
        spec.landmarks
            .iter()
            .filter_map(|lm| {
                self.resolve_anchor(&lm.anchor, urban).map(|(x, y)| ResolvedLandmark {
                    id: lm.id.clone(),
                    name: lm.name.clone(),
                    kind: lm.kind.clone(),
                    x,
                    y,
                })
            })
            .collect()
    }

    fn gate_pos(&self, id: &str, urban: Option<&UrbanSpec>) -> Option<(f32, f32)> {
        let idx = urban?.gates.iter().position(|g| g.id == id)?;
        let n = self.gates.get(idx)?;
        Some((self.g[*n].x, self.g[*n].y))
    }

    fn district_pos(&self, id: &str, urban: Option<&UrbanSpec>) -> Option<(f32, f32)> {
        let d = urban?.districts.iter().find(|d| d.id == id)?;
        let (fx, fy) = super::engine::resolve_simple(&d.anchor)?;
        Some((fx * self.width as f32, fy * self.height as f32))
    }

    fn resolve_anchor(&self, a: &Anchor, urban: Option<&UrbanSpec>) -> Option<(f32, f32)> {
        match a {
            Anchor::CityCenter => Some(self.center),
            Anchor::AtGate { gate } => self.gate_pos(gate, urban),
            Anchor::InDistrict { district } => self.district_pos(district, urban),
            Anchor::PierTip { pier } => {
                let idx = urban?.piers.iter().position(|p| &p.id == pier)?;
                self.piers.get(idx).map(|&(_, tip)| tip)
            }
            Anchor::AlongStreet { street, position } => {
                let st = urban?.streets.iter().find(|s| &s.id == street)?;
                let want = bearing_deg(&st.bearing);
                let gi = urban?.gates.iter().position(|g| (bearing_deg(&g.bearing) - want).abs() < 1.0)?;
                let gn = self.gates.get(gi)?;
                let t = position.clamp(0.0, 1.0);
                Some((self.center.0 + (self.g[*gn].x - self.center.0) * t, self.center.1 + (self.g[*gn].y - self.center.1) * t))
            }
            Anchor::OnWall { position } => {
                let theta = position.clamp(0.0, 1.0) * TAU;
                // Sample the wall polygon at the nearest vertex angle.
                let idx = ((theta / TAU) * (self.wall.len().saturating_sub(1)) as f32) as usize;
                self.wall.get(idx).copied()
            }
            Anchor::NearestIntersection { .. } | Anchor::AtStation { .. } => Some(self.center),
            other => super::engine::resolve_simple(other).map(|(x, y)| (x * self.width as f32, y * self.height as f32)),
        }
    }

    fn feature_labels(&self, spec: &MapSpec) -> Vec<UrbanLabel> {
        let mut labels = Vec::new();
        if let Some(u) = &spec.urban {
            for (i, gt) in u.gates.iter().enumerate() {
                if let (Some(name), Some(n)) = (&gt.name, self.gates.get(i)) {
                    labels.push(UrbanLabel { text: name.clone(), x: self.g[*n].x, y: self.g[*n].y });
                }
            }
            for d in &u.districts {
                if let (Some(name), Some(p)) = (&d.name, self.district_pos(&d.id, Some(u))) {
                    labels.push(UrbanLabel { text: name.clone(), x: p.0, y: p.1 });
                }
            }
            for (i, p) in u.piers.iter().enumerate() {
                if let (Some(name), Some(&(_, tip))) = (&p.name, self.piers.get(i)) {
                    labels.push(UrbanLabel { text: name.clone(), x: tip.0, y: tip.1 });
                }
            }
        }
        labels
    }
}

struct UrbanLabel {
    text: String,
    x: f32,
    y: f32,
}

// ── drawing primitives ───────────────────────────────────────────────────────

/// Draw a street as a noise-bent polyline between `a` and `b`: intermediate points
/// are displaced perpendicular to the segment by `amp · len · noise(midpoint)`, so
/// the street curves organically (and consistently — same field everywhere).
fn draw_curved(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), c: [u8; 3], thick: i32, amp: f32, noise: &dyn Fn(f32, f32) -> f32) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    let (nx, ny) = (-dy / len, dx / len); // unit perpendicular
    let steps = 6;
    let mut prev = a;
    for k in 1..=steps {
        let t = k as f32 / steps as f32;
        let base = (a.0 + dx * t, a.1 + dy * t);
        // Zero displacement at the endpoints (t=0,1), max in the middle.
        let bend = (t * std::f32::consts::PI).sin() * amp * len;
        let off = noise(base.0 * 0.05 + 1.3, base.1 * 0.05 + 4.7) * bend;
        let p = (base.0 + nx * off, base.1 + ny * off);
        draw_line(img, prev.0, prev.1, p.0, p.1, c, thick);
        prev = p;
    }
}

/// Bresenham line with optional 1px thickening.
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

/// Shrink a quad toward its centroid (so streets show between block parcels).
fn inset_quad(quad: [(f32, f32); 4]) -> Block {
    let cx = quad.iter().map(|p| p.0).sum::<f32>() / 4.0;
    let cy = quad.iter().map(|p| p.1).sum::<f32>() / 4.0;
    let f = |p: (f32, f32)| (p.0 + (cx - p.0) * 0.22, p.1 + (cy - p.1) * 0.22);
    Block { corners: [f(quad[0]), f(quad[1]), f(quad[2]), f(quad[3])] }
}

/// Scanline-fill a convex quad (the 4 corners in order).
fn fill_quad(img: &mut RgbImage, q: &[(f32, f32); 4], c: [u8; 3]) {
    let ys: Vec<f32> = q.iter().map(|p| p.1).collect();
    let y0 = ys.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
    let y1 = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil().min(img.height() as f32 - 1.0) as i32;
    for y in y0..=y1 {
        let yf = y as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::new();
        for k in 0..4 {
            let (p, q2) = (q[k], q[(k + 1) % 4]);
            if (p.1 <= yf && q2.1 > yf) || (q2.1 <= yf && p.1 > yf) {
                let t = (yf - p.1) / (q2.1 - p.1);
                xs.push(p.0 + t * (q2.0 - p.0));
            }
        }
        if xs.len() >= 2 {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (xa, xb) = (xs[0].round() as i32, xs[xs.len() - 1].round() as i32);
            for x in xa..=xb {
                put(img, x, y, c);
            }
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

fn rect_outline(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 3]) {
    for x in x0..=x1 {
        put(img, x, y0, c);
        put(img, x, y1, c);
    }
    for y in y0..=y1 {
        put(img, x0, y, c);
        put(img, x1, y, c);
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

/// Cardinal word → unit vector (pixel space, y grows downward → north is -y).
fn dir_vec(s: &str) -> (f32, f32) {
    let theta = (bearing_deg(s) - 90.0).to_radians();
    (theta.cos(), theta.sin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::spec::MapSpec;

    fn town() -> MapSpec {
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
    fn builds_a_radioconcentric_town() {
        let sg = graph();
        let (nodes, edges) = sg.stats();
        assert!(nodes > 50, "a town has many junctions (got {nodes})");
        assert!(edges > nodes, "more street segments than junctions");
        assert_eq!(sg.gates.len(), 3, "three spec gates → three arterials");
        assert!(sg.wall.len() > 3 && sg.wall.first() == sg.wall.last(), "wall is a closed ring");
    }

    #[test]
    fn wall_is_irregular_not_a_circle() {
        let sg = graph();
        // The wall radius varies around the ring (noise lobes) — not constant.
        let rs: Vec<f32> = sg
            .wall
            .iter()
            .map(|&(x, y)| ((x - sg.center.0).powi(2) + (y - sg.center.1).powi(2)).sqrt())
            .collect();
        let (min, max) = (rs.iter().cloned().fold(f32::INFINITY, f32::min), rs.iter().cloned().fold(0.0, f32::max));
        assert!(max - min > sg.radius * 0.08, "wall radius varies (organic), got spread {}", max - min);
    }

    #[test]
    fn blocks_are_inside_the_wall_and_deterministic() {
        let sg = graph();
        assert!(sg.blocks().len() > 10, "a town has many blocks (got {})", sg.blocks().len());
        for b in sg.blocks() {
            for &(x, y) in &b.corners {
                let d = ((x - sg.center.0).powi(2) + (y - sg.center.1).powi(2)).sqrt();
                assert!(d <= sg.radius * 1.05, "block corner inside the wall (d={d}, r={})", sg.radius);
            }
        }
        let (a, b) = (graph(), graph());
        let pa: Vec<_> = a.blocks().iter().map(|x| x.corners).collect::<Vec<_>>();
        let pb: Vec<_> = b.blocks().iter().map(|x| x.corners).collect::<Vec<_>>();
        assert_eq!(pa, pb);
    }

    #[test]
    fn urban_anchors_resolve_to_their_features() {
        let spec = town();
        let canvas = GeoCanvas::from_spec(&spec, 7);
        let sg = StreetGraph::generate(&spec, &canvas);
        assert_eq!(sg.resolve_anchor(&Anchor::CityCenter, spec.urban.as_ref()).unwrap(), sg.center);
        assert!(sg.resolve_anchor(&Anchor::AtGate { gate: "north_gate".into() }, spec.urban.as_ref()).is_some());
        assert!(sg.resolve_anchor(&Anchor::InDistrict { district: "market".into() }, spec.urban.as_ref()).is_some());
        assert!(sg.resolve_anchor(&Anchor::AtGate { gate: "nope".into() }, spec.urban.as_ref()).is_none());
    }

    #[test]
    fn town_render_is_deterministic() {
        let sg = graph();
        let spec = town();
        let a = sg.render_town(&spec);
        let b = sg.render_town(&spec);
        assert!(a.as_raw() == b.as_raw(), "town render must be byte-stable");
        assert_eq!(a.dimensions(), (sg.width, sg.height));
    }

    #[test]
    fn generation_is_deterministic() {
        let a = graph();
        let b = graph();
        assert_eq!(a.stats(), b.stats());
        let pa: Vec<_> = a.g.node_indices().map(|i| (a.g[i].x, a.g[i].y)).collect();
        let pb: Vec<_> = b.g.node_indices().map(|i| (b.g[i].x, b.g[i].y)).collect();
        assert_eq!(pa, pb);
    }
}
