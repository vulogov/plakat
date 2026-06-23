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

        // Erosion / irregularity strength (0 = smooth, 1 = natural default, >1 rugged).
        let erosion = spec.terrain.erosion.unwrap_or(1.0).clamp(0.0, 4.0);

        // Mountain ranges: oriented gaussian ridges at their resolved anchor.
        for (i, range) in spec.terrain.mountain_ranges.iter().enumerate() {
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
                    canvas.seed as u32 ^ (i as u32).wrapping_mul(0x9e37),
                    erosion,
                );
            }
        }

        // Shape the landmass so the map actually has coasts (island radial taper /
        // sea-positioned edges) — this also makes rivers drain into the sea.
        apply_landmass_shape(&mut data, spec, w, h, canvas.seed as u32, erosion);

        normalize(&mut data);

        // Lakes: carve a smooth basin below sea level at each lake's anchor, so the
        // existing coast/biome/hydrology/render pipeline realizes it as water (blue
        // fill + a shoreline ring; rivers drain into it). Done after `normalize` so
        // the basin floor sits at a known absolute depth under the sea level.
        apply_lakes(&mut data, spec, w, h, canvas.seed as u32, erosion);

        // Plateaus / mesas: flat-topped tablelands with steep scarp edges.
        // Before canyons, so a rift can cut into a plateau. Empty → no-op.
        apply_plateaus(&mut data, spec, w, h, canvas.seed as u32, erosion);

        // Dry canyons (rift valleys): narrow oriented trenches carved into the
        // terrain, floor kept ABOVE sea level so they read as deep gorges, not
        // water. Empty `rift_valleys` → no-op (byte-identical to pre-canyon).
        apply_canyons(&mut data, spec, w, h, canvas.seed as u32, erosion);

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

/// Add an oriented ridge centered at normalized (cx, cy): an anisotropic gaussian,
/// narrow across the axis and long along it — but the **ridgeline wanders** (noise
/// displaces it perpendicular along its length) and the **crest height varies**
/// (noise-modulated amplitude), so a real eroded range, not a smooth oval.
#[allow(clippy::too_many_arguments)]
fn add_ridge(data: &mut [f32], w: u32, h: u32, cx: f32, cy: f32, orientation: &str, length_frac: f32, amp: f32, seed: u32, erosion: f32) {
    let (dx, dy) = orient_dir(orientation);
    let cxp = cx * w as f32;
    let cyp = cy * h as f32;
    let sigma_perp = (w.min(h) as f32 * 0.05).max(1.0);
    let sigma_along = (length_frac * 0.5 * w.max(h) as f32).max(sigma_perp);
    let two_sp2 = 2.0 * sigma_perp * sigma_perp;
    let two_sa2 = 2.0 * sigma_along * sigma_along;
    let reach_perp = 3.5 * sigma_perp + sigma_perp * 2.0; // extra room for the wander
    let reach_along = 3.5 * sigma_along;
    let perlin = Perlin::new(seed);
    // Ridgeline offset + crest modulation along the axis, both scaled by `erosion`
    // (0 → a smooth straight oval, 1 → the natural default).
    let wander = |t: f32| perlin.get([(t * 0.012) as f64, 4.0]) as f32 * sigma_perp * 1.6 * erosion;
    let crest = |t: f32| 1.0 - erosion * (0.35 - 0.35 * (perlin.get([(t * 0.02) as f64, 11.0]) as f32 * 0.5 + 0.5));
    for y in 0..h {
        for x in 0..w {
            let ox = x as f32 - cxp;
            let oy = y as f32 - cyp;
            let along = ox * dx + oy * dy;
            let perp = ox * -dy + oy * dx - wander(along); // bend the ridgeline
            if along.abs() > reach_along || perp.abs() > reach_perp {
                continue;
            }
            let g = (-(perp * perp) / two_sp2 - (along * along) / two_sa2).exp();
            data[(y * w + x) as usize] += amp * crest(along) * g;
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
/// taper radially to sea; otherwise the sea-positioned edges are lowered. The
/// coastline is **multi-scale noise-warped** — big lobes (bays + peninsulas),
/// headlands, and a ragged edge — so a real, eroded shore rather than a circle.
fn apply_landmass_shape(data: &mut [f32], spec: &MapSpec, w: u32, h: u32, seed: u32, erosion: f32) {
    let perlin = Perlin::new(seed.wrapping_add(0x5417));
    // Periodic-around-the-circle coastal radius modulation at screen-angle θ:
    // multiple octaves so the shore is irregular at several scales. `erosion` scales
    // the noise terms (0 → a perfect circle, 1 → the natural default).
    let coast_mod = |theta: f32| -> f32 {
        let (c, s) = (theta.cos() as f64, theta.sin() as f64);
        let n = |fx: f64, dx: f64, dy: f64| perlin.get([c * fx + dx, s * fx + dy]) as f32;
        1.0 + erosion * (0.30 * n(1.2, 2.0, 2.0) + 0.15 * n(3.1, 7.0, 1.0) + 0.06 * n(7.3, 4.0, 9.0))
    };
    if is_island(spec) {
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let land_r = (cx * cx + cy * cy).sqrt() * 0.6; // land reaches ~60% toward the corners
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                // Push the effective coast in/out by angle → bays + peninsulas.
                let r = (dx * dx + dy * dy).sqrt() / land_r / coast_mod(dy.atan2(dx)).max(0.4);
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

/// Carve a smooth lake basin (below sea level) at each spec lake's anchor. The
/// floor sits at `LAKE_FLOOR` (< `coastline::DEFAULT_SEA_LEVEL`) so the lake reads as
/// water everywhere downstream; a smooth `smoothstep` edge gives it a shoreline.
fn apply_lakes(data: &mut [f32], spec: &MapSpec, w: u32, h: u32, seed: u32, erosion: f32) {
    /// Lake floor depth (must be below `coastline::DEFAULT_SEA_LEVEL` = 0.22).
    const LAKE_FLOOR: f32 = 0.10;
    let extent = w.min(h) as f32;
    for (i, lake) in spec.water.lakes.iter().enumerate() {
        let Some((fx, fy)) = resolve_simple(&lake.anchor) else { continue };
        let (cx, cy) = (fx * w as f32, fy * h as f32);
        let r = lake_radius(&lake.size) * extent;
        let reach = r * 1.6;
        // Wander the shore by direction (scaled by erosion) so the lake is a
        // natural, lobed basin — not a perfect disc. Two octaves: big bays +
        // a finer ragged edge. `erosion = 0` → the old circular lake.
        let perlin = Perlin::new(seed ^ (i as u32).wrapping_mul(0x68e3));
        let (x0, y0) = ((cx - reach).max(0.0) as u32, (cy - reach).max(0.0) as u32);
        let (x1, y1) = ((cx + reach).min(w as f32) as u32, (cy + reach).min(h as f32) as u32);
        for y in y0..y1 {
            for x in x0..x1 {
                let (ox, oy) = (x as f32 - cx, y as f32 - cy);
                let ang = oy.atan2(ox);
                let (ca, sa) = (ang.cos() as f64, ang.sin() as f64);
                let wob = (perlin.get([ca * 1.6, sa * 1.6]) as f32 * 0.26
                    + perlin.get([ca * 4.3, sa * 4.3]) as f32 * 0.12)
                    * erosion;
                let d = (ox * ox + oy * oy).sqrt() / r + wob;
                // 0 in the deep centre → 1 at/outside the shore (land untouched).
                let m = smoothstep(0.72, 1.0, d);
                let id = idx(x, y, w);
                data[id] = LAKE_FLOOR * (1.0 - m) + data[id] * m;
            }
        }
    }
}

/// Carve **dry canyons** (rift valleys): narrow oriented trenches whose floor
/// stays ABOVE `coastline::DEFAULT_SEA_LEVEL` (0.22) so they read as deep dry
/// gorges, not water. Mirrors `add_ridge`'s oriented, erosion-wandered profile
/// but SUBTRACTS toward a floor — and only ever lowers terrain (a canyon over a
/// valley doesn't fill it). Done after `normalize` so the floor is absolute.
fn apply_canyons(data: &mut [f32], spec: &MapSpec, w: u32, h: u32, seed: u32, erosion: f32) {
    for (i, rift) in spec.terrain.rift_valleys.iter().enumerate() {
        let Some((fx, fy)) = resolve_simple(&rift.anchor) else { continue };
        let (cx, cy) = (fx * w as f32, fy * h as f32);
        let (dx, dy) = orient_dir(&rift.orientation);
        let len = if rift.length_fraction > 0.0 { rift.length_fraction } else { 0.45 };
        let floor = canyon_floor(&rift.size);
        let sigma_perp = (w.min(h) as f32 * 0.018).max(1.0); // narrow (a gorge)
        let sigma_along = (len * 0.5 * w.max(h) as f32).max(sigma_perp);
        let two_sp2 = 2.0 * sigma_perp * sigma_perp;
        let two_sa2 = 2.0 * sigma_along * sigma_along;
        let reach_perp = 3.5 * sigma_perp + sigma_perp * 2.0;
        let reach_along = 3.5 * sigma_along;
        let perlin = Perlin::new(seed ^ (i as u32).wrapping_mul(0x85eb));
        // Wander the canyon line so it isn't a straight slot (scaled by erosion).
        let wander = |t: f32| perlin.get([(t * 0.012) as f64, 7.0]) as f32 * sigma_perp * 1.8 * erosion;
        for y in 0..h {
            for x in 0..w {
                let ox = x as f32 - cx;
                let oy = y as f32 - cy;
                let along = ox * dx + oy * dy;
                let perp = ox * -dy + oy * dx - wander(along);
                if along.abs() > reach_along || perp.abs() > reach_perp {
                    continue;
                }
                let g = (-(perp * perp) / two_sp2 - (along * along) / two_sa2).exp();
                let id = idx(x, y, w);
                // Carve toward the floor, but never raise terrain (gorge, not fill).
                let target = floor.min(data[id]);
                data[id] = data[id] * (1.0 - g) + target * g;
            }
        }
    }
}

/// Realize **plateaus / mesas**: flat-topped tablelands with steep scarp edges.
/// The core is forced to a flat `top` elevation; a `smoothstep` scarp ramps down
/// to the surrounding terrain; outside the reach the terrain is untouched. The
/// rim is erosion-wandered so the mesa isn't a perfect disc. Done after
/// `normalize` (absolute elevations). Empty `plateaus` → no-op.
fn apply_plateaus(data: &mut [f32], spec: &MapSpec, w: u32, h: u32, seed: u32, erosion: f32) {
    let extent = w.min(h) as f32;
    for (i, plat) in spec.terrain.plateaus.iter().enumerate() {
        let Some((fx, fy)) = resolve_simple(&plat.anchor) else { continue };
        let (cx, cy) = (fx * w as f32, fy * h as f32);
        let r = plateau_radius(&plat.size, plat.length_fraction) * extent;
        let top = plateau_top(&plat.size);
        let reach = r * 1.6;
        let perlin = Perlin::new(seed ^ (i as u32).wrapping_mul(0xc2b2));
        let (x0, y0) = ((cx - reach).max(0.0) as u32, (cy - reach).max(0.0) as u32);
        let (x1, y1) = ((cx + reach).min(w as f32) as u32, (cy + reach).min(h as f32) as u32);
        for y in y0..y1 {
            for x in x0..x1 {
                let ox = x as f32 - cx;
                let oy = y as f32 - cy;
                // Wander the rim radius by direction (scaled by erosion).
                let ang = oy.atan2(ox);
                let wob = perlin.get([ang.cos() as f64 * 1.7, ang.sin() as f64 * 1.7]) as f32
                    * 0.22
                    * erosion;
                let d = (ox * ox + oy * oy).sqrt() / r + wob;
                // 0 across the flat core → 1 past the scarp (land untouched).
                let m = smoothstep(0.70, 1.0, d);
                let id = idx(x, y, w);
                // Flat top across the core, scarp ramp to terrain at the rim.
                let blended = top * (1.0 - m) + data[id] * m;
                // A mesa rises above its surroundings — never lower a higher peak.
                data[id] = blended.max(data[id]);
            }
        }
    }
}

/// Plateau `size` word → flat-top elevation (absolute, above sea 0.22, below
/// mountain peaks ~1.0).
fn plateau_top(size: &str) -> f32 {
    match size.to_ascii_lowercase().as_str() {
        "small" => 0.55,
        "large" | "great" => 0.68,
        _ => 0.60, // moderate / unspecified
    }
}

/// Plateau radius as a fraction of the shorter extent (explicit
/// `length_fraction` wins, else a per-size default).
fn plateau_radius(size: &str, length_fraction: f32) -> f32 {
    if length_fraction > 0.0 {
        return (length_fraction * 0.5).clamp(0.03, 0.4);
    }
    match size.to_ascii_lowercase().as_str() {
        "small" => 0.08,
        "large" | "great" => 0.16,
        _ => 0.12,
    }
}

/// Rift-valley / canyon `size` word → absolute floor depth. All stay above
/// `DEFAULT_SEA_LEVEL` (0.22) so the canyon is dry; deeper words cut closer.
fn canyon_floor(size: &str) -> f32 {
    match size.to_ascii_lowercase().as_str() {
        "shallow" => 0.40,
        "deep" => 0.28,
        "chasm" | "abyss" | "great" => 0.24, // just above sea → dry but dramatic
        _ => 0.33, // moderate / unspecified
    }
}

/// Lake `size` word → radius as a fraction of the canvas's shorter extent.
fn lake_radius(size: &str) -> f32 {
    match size.to_ascii_lowercase().as_str() {
        "large" | "great" => 0.11,
        "small" | "tarn" | "pond" => 0.05,
        _ => 0.075, // medium / unspecified
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
    fn a_spec_lake_becomes_water_at_its_anchor() {
        use crate::map::spec::LakeSpec;
        let mut spec = isle();
        spec.water.lakes.push(LakeSpec {
            id: "tarn".into(),
            name: Some("Blue Tarn".into()),
            anchor: Anchor::Cardinal { position: "center".into() },
            size: "large".into(),
            endorheic: true,
        });
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        // The lake centre (canvas centre) is below sea level (water)…
        let (lx, ly) = (hf.width / 2, hf.height / 2);
        assert!(hf.data[(ly * hf.width + lx) as usize] < 0.22, "lake centre is sub-sea-level (water)");
        // …whereas without the lake that cell is the mountain massif (land).
        let dry = HeightField::generate(&isle(), &c);
        assert!(dry.data[(ly * dry.width + lx) as usize] >= 0.22, "no lake → centre is land");
    }

    #[test]
    fn a_rift_valley_carves_a_dry_canyon() {
        use crate::map::spec::NamedRegion;
        let mut spec = isle();
        spec.terrain.rift_valleys.push(NamedRegion {
            id: "gorge".into(),
            name: Some("Deep Gorge".into()),
            anchor: Anchor::Cardinal { position: "center".into() },
            orientation: "north-south".into(),
            length_fraction: 0.5,
            size: "deep".into(),
        });
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let dry = HeightField::generate(&isle(), &c);
        let i = ((hf.height / 2) * hf.width + hf.width / 2) as usize;
        // The canyon carves the central massif down…
        assert!(hf.data[i] < dry.data[i] - 0.05, "canyon lowers the centre ({} vs {})", hf.data[i], dry.data[i]);
        // …but its floor stays ABOVE sea level — a DRY gorge, not a lake.
        assert!(hf.data[i] >= 0.22, "dry canyon floor above sea level: {}", hf.data[i]);
        // Determinism holds.
        assert_eq!(hf.data, HeightField::generate(&spec, &c).data);
    }

    #[test]
    fn a_plateau_raises_a_flat_tableland() {
        use crate::map::spec::NamedRegion;
        let mut spec = MapSpec::minimal("Plains", 2, 2, 3); // no mountain ranges
        spec.terrain.plateaus.push(NamedRegion {
            id: "mesa".into(),
            name: Some("High Mesa".into()),
            anchor: Anchor::Cardinal { position: "center".into() },
            orientation: String::new(),
            length_fraction: 0.0,
            size: "large".into(),
        });
        let c = GeoCanvas::from_spec(&spec, 7);
        let hf = HeightField::generate(&spec, &c);
        let i = ((hf.height / 2) * hf.width + hf.width / 2) as usize;
        // Core sits at the flat plateau top (≈0.68 "large"); never below it.
        assert!(hf.data[i] >= 0.66, "plateau core at the flat top: {}", hf.data[i]);
        // And it never lowers the underlying plain.
        let plain = MapSpec::minimal("Plains", 2, 2, 3);
        let dry = HeightField::generate(&plain, &GeoCanvas::from_spec(&plain, 7));
        assert!(hf.data[i] >= dry.data[i], "plateau never lowers terrain");
    }

    #[test]
    fn erosion_controls_coastline_irregularity() {
        // Smooth (0) vs natural (default/1) vs rugged (2.5) give different terrain,
        // and each is still deterministic.
        let mut smooth = isle();
        smooth.terrain.erosion = Some(0.0);
        let mut rugged = isle();
        rugged.terrain.erosion = Some(2.5);
        let c = GeoCanvas::from_spec(&isle(), 42);
        let hs = HeightField::generate(&smooth, &c).data;
        let hn = HeightField::generate(&isle(), &c).data; // None → default 1.0
        let hr = HeightField::generate(&rugged, &c).data;
        assert!(hs != hn, "erosion 0 differs from the natural default");
        assert!(hr != hn, "erosion 2.5 differs from the natural default");
        // Determinism per level.
        assert_eq!(hr, HeightField::generate(&rugged, &c).data);
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
