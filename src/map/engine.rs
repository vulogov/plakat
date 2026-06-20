//! MAP-2 geometry engine. **L0** canvas + **L1** tectonic heightmap (this slice);
//! L2 hydraulics … L7 conditioning follow. Everything here is a **pure function
//! of (spec, seed)** — render twice, get identical bytes (the corpus invariant).

use anyhow::{Context, Result};
use image::{GrayImage, Luma};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use std::path::Path;

use super::spec::{Anchor, MapSpec};

/// Working geometry resolution per tile (the heightmap grid; the SD render
/// upsamples per tile later). Kept modest so CPU dumps are fast + deterministic.
const GEO_PX_PER_TILE: u32 = 256;
/// Cap the total working resolution (an 8×8 map would otherwise be 2048²).
const GEO_MAX: u32 = 2048;

/// L0 — the geometry canvas: working resolution + the parameters every layer reads.
#[derive(Debug, Clone)]
pub struct GeoCanvas {
    pub width: u32,
    pub height: u32,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub seed: u64,
    pub noise_octaves: usize,
}

impl GeoCanvas {
    pub fn from_spec(spec: &MapSpec, seed: u64) -> Self {
        let cols = spec.tile_grid.cols.max(1);
        let rows = spec.tile_grid.rows.max(1);
        let width = (cols * GEO_PX_PER_TILE).min(GEO_MAX);
        let height = (rows * GEO_PX_PER_TILE).min(GEO_MAX);
        // Bigger extents carry more octaves of detail.
        let noise_octaves = (5 + cols.max(rows) as usize).min(10);
        GeoCanvas { width, height, tile_cols: cols, tile_rows: rows, seed, noise_octaves }
    }
}

/// A normalized [0,1] elevation field, row-major `width*height`.
#[derive(Debug, Clone)]
pub struct HeightField {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl HeightField {
    /// L1 — fBm base elevation + a gaussian ridge per mountain range.
    pub fn generate(spec: &MapSpec, canvas: &GeoCanvas) -> Self {
        let (w, h) = (canvas.width, canvas.height);
        let fbm = Fbm::<Perlin>::new(canvas.seed as u32)
            .set_octaves(canvas.noise_octaves)
            .set_frequency(2.5)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        let mut data = vec![0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let nx = x as f64 / w as f64;
                let ny = y as f64 / h as f64;
                // fBm returns ~[-1,1]; shift to [0,1].
                data[(y * w + x) as usize] = (fbm.get([nx, ny]) as f32) * 0.5 + 0.5;
            }
        }

        // Mountain ranges: oriented gaussian ridges at their resolved anchor.
        for range in &spec.terrain.mountain_ranges {
            if let Some((cx, cy)) = resolve_simple(&range.anchor) {
                add_ridge(
                    &mut data,
                    w,
                    h,
                    cx,
                    cy,
                    &range.orientation,
                    range.length_fraction.max(0.25),
                    height_amp(&range.height),
                );
            }
        }

        // Shape the landmass so the map actually has coasts (island radial taper /
        // sea-positioned edges) — this also makes rivers drain into the sea.
        apply_landmass_shape(&mut data, spec, w, h);

        normalize(&mut data);
        HeightField { width: w, height: h, data }
    }

    /// Write a normalized grayscale PNG (white = high). Deterministic encode.
    pub fn save_gray_png(&self, path: &Path) -> Result<()> {
        let mut img = GrayImage::new(self.width, self.height);
        for (i, &v) in self.data.iter().enumerate() {
            let p = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel((i as u32) % self.width, (i as u32) / self.width, Luma([p]));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        img.save(path).with_context(|| format!("writing heightmap {}", path.display()))
    }
}

/// Resolve the anchors the early layers need (terrain/biome placement): cardinal
/// + canvas → normalized (x, y) in [0,1]. Anchors that depend on later layers
/// (rivers, coastline) return None here and are resolved by the full Layer-5
/// resolver.
pub fn resolve_simple(anchor: &Anchor) -> Option<(f32, f32)> {
    match anchor {
        Anchor::Canvas { x, y } => Some((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))),
        Anchor::Cardinal { position } => cardinal_xy(position),
        _ => None,
    }
}

/// 9-grid cardinal name → normalized (x, y) in [0,1].
fn cardinal_xy(pos: &str) -> Option<(f32, f32)> {
    let p = pos.to_ascii_lowercase().replace([' ', '-'], "_");
    Some(match p.as_str() {
        "north" | "top" => (0.5, 0.15),
        "south" | "bottom" => (0.5, 0.85),
        "east" | "right" => (0.85, 0.5),
        "west" | "left" => (0.15, 0.5),
        "center" | "centre" | "middle" => (0.5, 0.5),
        "northeast" | "north_east" => (0.8, 0.2),
        "northwest" | "north_west" => (0.2, 0.2),
        "southeast" | "south_east" => (0.8, 0.8),
        "southwest" | "south_west" => (0.2, 0.8),
        _ => return None,
    })
}

/// Orientation name → unit direction the ridge runs along.
fn orient_dir(orientation: &str) -> (f32, f32) {
    let o = orientation.to_ascii_lowercase().replace([' ', '_'], "-");
    let inv = 1.0 / 2f32.sqrt();
    match o.as_str() {
        "north-south" | "n-s" | "vertical" => (0.0, 1.0),
        "east-west" | "e-w" | "horizontal" => (1.0, 0.0),
        "northeast" | "ne" | "northeast-southwest" => (inv, -inv),
        "northwest" | "nw" | "northwest-southeast" => (-inv, -inv),
        _ => (0.0, 1.0), // default to N-S
    }
}

/// `"low"|"moderate"|"high"|"extreme"` → ridge elevation amplitude.
fn height_amp(height: &str) -> f32 {
    match height.to_ascii_lowercase().as_str() {
        "low" => 0.15,
        "high" => 0.5,
        "extreme" => 0.7,
        _ => 0.3, // moderate / unspecified
    }
}

/// Add a smooth oriented ridge centered at normalized (cx, cy): an anisotropic
/// gaussian, narrow across the axis (`sigma_perp`) and long along it
/// (`sigma_along`) — so it tapers smoothly at both ends (no hard rectangle).
#[allow(clippy::too_many_arguments)]
fn add_ridge(data: &mut [f32], w: u32, h: u32, cx: f32, cy: f32, orientation: &str, length_frac: f32, amp: f32) {
    let (dx, dy) = orient_dir(orientation);
    let cxp = cx * w as f32;
    let cyp = cy * h as f32;
    let sigma_perp = (w.min(h) as f32 * 0.05).max(1.0);
    let sigma_along = (length_frac * 0.5 * w.max(h) as f32).max(sigma_perp);
    let (two_sp2, two_sa2) = (2.0 * sigma_perp * sigma_perp, 2.0 * sigma_along * sigma_along);
    // Skip beyond ~3.5σ in either direction (negligible contribution, much faster).
    let reach_perp = 3.5 * sigma_perp;
    let reach_along = 3.5 * sigma_along;
    for y in 0..h {
        for x in 0..w {
            let ox = x as f32 - cxp;
            let oy = y as f32 - cyp;
            let along = ox * dx + oy * dy;
            let perp = ox * -dy + oy * dx;
            if along.abs() > reach_along || perp.abs() > reach_perp {
                continue;
            }
            data[(y * w + x) as usize] += amp * (-(perp * perp) / two_sp2 - (along * along) / two_sa2).exp();
        }
    }
}

fn idx(x: u32, y: u32, w: u32) -> usize {
    (y * w + x) as usize
}

/// Smoothstep: 0 below `e0`, 1 above `e1`, smooth between.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Is this map island-like (terrain surrounded by water)?
fn is_island(spec: &MapSpec) -> bool {
    let n = spec.name.to_ascii_lowercase();
    spec.scale_tier <= 1
        || n.contains("isle")
        || n.contains("island")
        || spec.water.seas.iter().any(|s| s.enclosed)
}

/// Multiply the elevation by a landmass mask so the map has coasts: island maps
/// taper radially to sea; otherwise the sea-positioned edges are lowered.
fn apply_landmass_shape(data: &mut [f32], spec: &MapSpec, w: u32, h: u32) {
    if is_island(spec) {
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let land_r = (cx * cx + cy * cy).sqrt() * 0.6; // land reaches ~60% toward the corners
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt() / land_r; // 0 center … 1 at the coast
                let mask = 1.0 - smoothstep(0.78, 1.12, r); // 1 inland, → 0 offshore
                data[idx(x, y, w)] *= mask;
            }
        }
    } else {
        for sea in &spec.water.seas {
            lower_sea_edge(data, w, h, &sea.position);
        }
    }
}

/// Lower the named edge toward sea (a smooth ramp from the edge inward).
fn lower_sea_edge(data: &mut [f32], w: u32, h: u32, position: &str) {
    let p = position.to_ascii_lowercase();
    let depth = 0.22; // ramp depth as a fraction of the extent
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w.max(1) as f32, y as f32 / h.max(1) as f32);
            // distance into the map from the named edge, 0 at edge → 1 deep inland.
            let d = match () {
                _ if p.contains("west") => fx / depth,
                _ if p.contains("east") => (1.0 - fx) / depth,
                _ if p.contains("north") => fy / depth,
                _ if p.contains("south") => (1.0 - fy) / depth,
                _ => continue,
            };
            data[idx(x, y, w)] *= smoothstep(0.0, 1.0, d);
        }
    }
}

/// Min-max normalize a field to [0,1] in place.
fn normalize(data: &mut [f32]) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in data.iter() {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = (hi - lo).max(1e-6);
    for v in data.iter_mut() {
        *v = (*v - lo) / span;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::spec::{MapSpec, MountainRange};

    fn isle() -> MapSpec {
        let mut m = MapSpec::minimal("Test", 2, 2, 1);
        m.terrain.dominant_elevation = "mountainous".into();
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

    #[test]
    fn canvas_resolution_from_tiles() {
        let c = GeoCanvas::from_spec(&isle(), 42);
        assert_eq!((c.width, c.height), (512, 512));
        let big = GeoCanvas::from_spec(&MapSpec::minimal("X", 8, 8, 5), 1);
        assert_eq!((big.width, big.height), (2048, 2048), "capped at GEO_MAX");
    }

    #[test]
    fn heightmap_is_deterministic_and_normalized() {
        let spec = isle();
        let c = GeoCanvas::from_spec(&spec, 42);
        let a = HeightField::generate(&spec, &c);
        let b = HeightField::generate(&spec, &c);
        assert_eq!(a.data, b.data, "pure fn of (spec, seed)");
        let (lo, hi) = a.data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(lo <= 1e-4 && (hi - 1.0).abs() <= 1e-4, "normalized to [0,1]: {lo}..{hi}");
    }

    #[test]
    fn ridge_raises_the_center_seam() {
        let spec = isle();
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        // A vertical N-S ridge through center → the center column averages higher
        // than the far-left column.
        let col_mean = |cx: u32| -> f32 {
            (0..hf.height).map(|y| hf.data[(y * hf.width + cx) as usize]).sum::<f32>() / hf.height as f32
        };
        assert!(col_mean(hf.width / 2) > col_mean(hf.width / 12), "ridge lifts the center");
    }

    #[test]
    fn different_seeds_differ() {
        let spec = isle();
        let a = HeightField::generate(&spec, &GeoCanvas::from_spec(&spec, 1));
        let b = HeightField::generate(&spec, &GeoCanvas::from_spec(&spec, 2));
        assert_ne!(a.data, b.data);
    }
}
