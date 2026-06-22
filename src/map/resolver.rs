//! MAP-2 **L5** — the landmark anchor resolver. Turns each landmark's typed
//! [`Anchor`] into a pixel position by resolving it against the layers that now
//! exist: L2 rivers (`mouth_of`/`source_of`/`delta`), L3 coastline
//! (`natural_harbor`/`coast_nearest`/`shore_nearest`), terrain (`range_slope`/
//! `pass_between`), and other resolved landmarks (`bearing`).
//!
//! Resolution is a fixpoint over the dependency graph: base anchors first, then
//! `bearing`-style anchors whose `from` has been resolved, until stable. A pass
//! with unresolved-but-no-progress means a cycle → error (never a silent miss).

use anyhow::{Result, bail};
use image::{Rgb, RgbImage};
use std::collections::HashMap;
use std::path::Path;

use super::coastline::Coastline;
use super::engine::{HeightField, resolve_simple};
use super::hydrology::Hydrology;
use super::spec::{Anchor, LandmarkKind, MapSpec};

/// A landmark placed at a pixel position.
#[derive(Debug, Clone)]
pub struct ResolvedLandmark {
    pub id: String,
    pub name: String,
    pub kind: LandmarkKind,
    pub x: f32,
    pub y: f32,
}

type Pt = (f32, f32);

/// Precomputed feature reference points (ranges/regions centers, river
/// mouths/sources) + the coast/river geometry the anchors snap to.
struct Context {
    w: f32,
    h: f32,
    coast_points: Vec<Pt>,
    river_cells: Vec<Pt>,
    ranges: HashMap<String, Pt>,
    regions: HashMap<String, Pt>,
    river_mouth: HashMap<String, Pt>,
    river_source: HashMap<String, Pt>,
}

/// Build the feature-reference context: coast/river cells, range/region centers, and
/// each river's resolved source + mouth (mouth snapped to the coast). Shared by the
/// landmark resolver and the named-river matcher.
fn build_context(spec: &MapSpec, hf: &HeightField, hydro: &Hydrology, coast: &Coastline) -> Context {
    let (w, h) = (hf.width as f32, hf.height as f32);

    let mut coast_points = Vec::new();
    for y in 0..coast.height {
        for x in 0..coast.width {
            if coast.is_coast(x, y) {
                coast_points.push((x as f32, y as f32));
            }
        }
    }
    let river_cells: Vec<Pt> = hydro.rivers.iter().flatten().map(|&(x, y)| (x as f32, y as f32)).collect();

    let base = |a: &Anchor| -> Option<Pt> { resolve_simple(a).map(|(x, y)| (x * w, y * h)) };
    let mut ranges = HashMap::new();
    for r in &spec.terrain.mountain_ranges {
        if let Some(p) = base(&r.anchor) {
            ranges.insert(r.id.clone(), p);
        }
    }
    let mut regions = HashMap::new();
    for r in &spec.regions {
        if let Some(p) = base(&r.anchor) {
            regions.insert(r.id.clone(), p);
        }
    }

    let mut ctx = Context {
        w,
        h,
        coast_points,
        river_cells,
        ranges,
        regions,
        river_mouth: HashMap::new(),
        river_source: HashMap::new(),
    };
    for r in &spec.water.rivers {
        if let Some(p) = ctx.resolve_simple_or_feature(&r.mouth) {
            ctx.river_mouth.insert(r.id.clone(), ctx.nearest_coast(p).unwrap_or(p));
        }
        if let Some(p) = ctx.resolve_simple_or_feature(&r.source) {
            ctx.river_source.insert(r.id.clone(), p);
        }
    }
    ctx
}

/// L2 — match each **named** spec river to the traced channel whose mouth is nearest
/// the river's resolved mouth, so labels + GeoJSON ids follow the *intended*
/// watercourse (not just the longest channel). Greedy + unique (a channel is claimed
/// by at most one river). Returns `river_id → channel index` into `hydro.rivers`.
pub fn match_rivers_to_channels(
    spec: &MapSpec,
    hf: &HeightField,
    hydro: &Hydrology,
    coast: &Coastline,
) -> std::collections::HashMap<String, usize> {
    let ctx = build_context(spec, hf, hydro, coast);
    // A channel's mouth is its last (downstream-most) cell.
    let chan_mouths: Vec<Pt> = hydro
        .rivers
        .iter()
        .map(|c| c.last().map(|&(x, y)| (x as f32, y as f32)).unwrap_or((0.0, 0.0)))
        .collect();
    let mut used = vec![false; chan_mouths.len()];
    let mut out = std::collections::HashMap::new();
    // Assign in spec order; each river takes its nearest still-free channel.
    for r in &spec.water.rivers {
        let Some(&mouth) = ctx.river_mouth.get(&r.id) else { continue };
        let best = chan_mouths
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .min_by(|(_, a), (_, b)| {
                let da = (a.0 - mouth.0).powi(2) + (a.1 - mouth.1).powi(2);
                let db = (b.0 - mouth.0).powi(2) + (b.1 - mouth.1).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        if let Some(i) = best {
            used[i] = true;
            out.insert(r.id.clone(), i);
        }
    }
    out
}

/// Resolve all the spec's landmarks to pixel positions.
pub fn resolve_landmarks(
    spec: &MapSpec,
    hf: &HeightField,
    hydro: &Hydrology,
    coast: &Coastline,
) -> Result<Vec<ResolvedLandmark>> {
    let ctx = build_context(spec, hf, hydro, coast);

    // Fixpoint over landmarks.
    let mut resolved: HashMap<String, Pt> = HashMap::new();
    let mut pending: Vec<usize> = (0..spec.landmarks.len()).collect();
    loop {
        let mut progressed = false;
        let mut still = Vec::new();
        for &i in &pending {
            let lm = &spec.landmarks[i];
            if let Some(p) = ctx.resolve(&lm.anchor, &resolved) {
                resolved.insert(lm.id.clone(), p);
                progressed = true;
            } else {
                still.push(i);
            }
        }
        pending = still;
        if pending.is_empty() {
            break;
        }
        if !progressed {
            let ids: Vec<&str> = pending.iter().map(|&i| spec.landmarks[i].id.as_str()).collect();
            bail!(
                "map: could not resolve landmark anchor(s) {:?} — a dependency cycle or a \
                 reference to an undefined feature",
                ids
            );
        }
    }

    Ok(spec
        .landmarks
        .iter()
        .filter_map(|lm| {
            resolved.get(&lm.id).map(|&(x, y)| ResolvedLandmark {
                id: lm.id.clone(),
                name: lm.name.clone(),
                kind: lm.kind.clone(),
                x,
                y,
            })
        })
        .collect())
}

impl Context {
    /// Cardinal/canvas, plus range_slope / region_interior via the precomputed
    /// feature centers. Used to resolve river endpoints (no landmark deps).
    fn resolve_simple_or_feature(&self, a: &Anchor) -> Option<Pt> {
        match a {
            Anchor::Canvas { .. } | Anchor::Cardinal { .. } => {
                resolve_simple(a).map(|(x, y)| (x * self.w, y * self.h))
            }
            Anchor::RangeSlope { range, facing } => {
                let c = *self.ranges.get(range)?;
                let (dx, dy) = direction_vec(facing);
                Some((c.0 + dx * self.w * 0.12, c.1 + dy * self.h * 0.12))
            }
            Anchor::RegionInterior { region } => self.regions.get(region).copied(),
            _ => None,
        }
    }

    /// Full anchor resolution. `resolved` holds already-placed landmarks. Returns
    /// None when the anchor depends on something not yet resolved (defer).
    fn resolve(&self, a: &Anchor, resolved: &HashMap<String, Pt>) -> Option<Pt> {
        match a {
            Anchor::Canvas { .. } | Anchor::Cardinal { .. } | Anchor::RangeSlope { .. } | Anchor::RegionInterior { .. } => {
                self.resolve_simple_or_feature(a)
            }
            Anchor::MouthOf { river } | Anchor::Delta { river } => self.river_mouth.get(river).copied(),
            Anchor::SourceOf { river } => self.river_source.get(river).copied(),
            Anchor::Confluence { river_a, river_b } => {
                Some(midpoint(*self.river_mouth.get(river_a)?, *self.river_mouth.get(river_b)?))
            }
            Anchor::PassBetween { range_a, range_b } => {
                Some(midpoint(*self.ranges.get(range_a)?, *self.ranges.get(range_b)?))
            }
            Anchor::Bearing { from, direction, distance, constraint } => {
                let origin = self.lookup(from, resolved)?;
                let (dx, dy) = direction_vec(direction);
                let dist = distance_px(distance) * self.w.max(self.h);
                let p = (origin.0 + dx * dist, origin.1 + dy * dist);
                Some(self.apply_constraint(p, constraint.as_ref()))
            }
            Anchor::CoastNearest { from } | Anchor::ShoreNearest { from, .. } => {
                let p = self.lookup(from, resolved)?;
                self.nearest_coast(p)
            }
            Anchor::NaturalHarbor { near } => {
                let p = self.lookup(near, resolved)?;
                self.nearest_coast(p)
            }
            _ => None, // urban variants → Layer 6b (MAP-5)
        }
    }

    /// Resolve an id reference: a placed landmark, else a feature (range/region
    /// center, river mouth).
    fn lookup(&self, id: &str, resolved: &HashMap<String, Pt>) -> Option<Pt> {
        resolved
            .get(id)
            .or_else(|| self.ranges.get(id))
            .or_else(|| self.regions.get(id))
            .or_else(|| self.river_mouth.get(id))
            .copied()
    }

    fn apply_constraint(&self, p: Pt, constraint: Option<&super::spec::AnchorConstraint>) -> Pt {
        use super::spec::AnchorConstraint::*;
        match constraint {
            Some(Coastline) => self.nearest_coast(p).unwrap_or(p),
            Some(River | Navigable) => nearest(&self.river_cells, p).unwrap_or(p),
            _ => p, // RidgeLine / RoadNearest snapping deferred
        }
    }

    fn nearest_coast(&self, p: Pt) -> Option<Pt> {
        nearest(&self.coast_points, p)
    }
}

fn nearest(points: &[Pt], p: Pt) -> Option<Pt> {
    points
        .iter()
        .min_by(|a, b| {
            let da = (a.0 - p.0).powi(2) + (a.1 - p.1).powi(2);
            let db = (b.0 - p.0).powi(2) + (b.1 - p.1).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
}

fn midpoint(a: Pt, b: Pt) -> Pt {
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

/// Distance word → fraction of the canvas extent.
fn distance_px(distance: &str) -> f32 {
    match distance.to_ascii_lowercase().as_str() {
        "adjacent" => 0.04,
        "near" => 0.10,
        "moderate" => 0.20,
        "far" => 0.35,
        "distant" => 0.50,
        _ => 0.15,
    }
}

/// Cardinal direction → unit vector (y grows downward → north is -y).
fn direction_vec(direction: &str) -> Pt {
    let d = direction.to_ascii_lowercase().replace([' ', '_'], "-");
    let inv = std::f32::consts::FRAC_1_SQRT_2;
    match d.as_str() {
        "north" => (0.0, -1.0),
        "south" => (0.0, 1.0),
        "east" => (1.0, 0.0),
        "west" => (-1.0, 0.0),
        "northeast" | "north-east" => (inv, -inv),
        "northwest" | "north-west" => (-inv, -inv),
        "southeast" | "south-east" => (inv, inv),
        "southwest" | "south-west" => (-inv, inv),
        _ => (0.0, 0.0),
    }
}

/// Marker colour by kind.
pub fn marker_rgb(kind: &LandmarkKind) -> [u8; 3] {
    match kind {
        LandmarkKind::City => [0xd0, 0x30, 0x30],
        LandmarkKind::Town | LandmarkKind::Village => [0xe0, 0x80, 0x20],
        LandmarkKind::Port => [0x20, 0x60, 0xc0],
        LandmarkKind::Fortress | LandmarkKind::Castle => [0x50, 0x50, 0x58],
        LandmarkKind::Lighthouse => [0xe0, 0xc0, 0x20],
        LandmarkKind::Temple | LandmarkKind::Oracle => [0xb0, 0x40, 0xc0],
        _ => [0xf0, 0xf0, 0xf0],
    }
}

/// Draw the resolved landmarks as coloured discs over the coastline render.
pub fn render_overlay(
    hf: &HeightField,
    coast: &Coastline,
    landmarks: &[ResolvedLandmark],
    path: &Path,
) -> Result<()> {
    let (w, h) = (hf.width, hf.height);
    // Background: muted terrain + sea (same palette as the coast dump).
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
    for lm in landmarks {
        draw_marker(&mut img, lm.x as i32, lm.y as i32, marker_rgb(&lm.kind));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(path).map_err(|e| anyhow::anyhow!("writing landmark overlay {}: {e}", path.display()))
}

/// A filled disc (r=4) with a dark outline.
pub fn draw_marker(img: &mut RgbImage, cx: i32, cy: i32, color: [u8; 3]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for dy in -5..=5 {
        for dx in -5..=5 {
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= w || y >= h {
                continue;
            }
            let d2 = dx * dx + dy * dy;
            if d2 <= 16 {
                img.put_pixel(x as u32, y as u32, Rgb(color));
            } else if d2 <= 25 {
                img.put_pixel(x as u32, y as u32, Rgb([0x20, 0x18, 0x10])); // outline
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::coastline::{Coastline, DEFAULT_SEA_LEVEL};
    use crate::map::engine::GeoCanvas;
    use crate::map::hydrology::{Hydrology, DEFAULT_RIVER_THRESHOLD};

    fn island() -> MapSpec {
        // Load the committed corpus spec so the test tracks the real anchors.
        let src = include_str!("../../corpus/map/island.spec.json");
        serde_json::from_str(src).unwrap()
    }

    fn resolve() -> (MapSpec, Vec<ResolvedLandmark>) {
        let spec = island();
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let hydro = Hydrology::compute(&hf, DEFAULT_RIVER_THRESHOLD, DEFAULT_SEA_LEVEL);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        let lms = resolve_landmarks(&spec, &hf, &hydro, &coast).unwrap();
        (spec, lms)
    }

    #[test]
    fn all_island_landmarks_resolve() {
        let (spec, lms) = resolve();
        assert_eq!(lms.len(), spec.landmarks.len(), "every landmark placed");
        // ids match.
        for lm in &spec.landmarks {
            assert!(lms.iter().any(|r| r.id == lm.id), "missing {}", lm.id);
        }
    }

    #[test]
    fn port_at_river_mouth_is_on_the_coast() {
        let (_, lms) = resolve();
        let c = GeoCanvas::from_spec(&island(), 42);
        let hf = HeightField::generate(&island(), &c);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        // Saltmere (mouth_of ash) should sit near the coastline, not inland.
        let saltmere = lms.iter().find(|l| l.id == "saltmere").unwrap();
        let cd = coast.coast_dist[(saltmere.y as u32 * coast.width + saltmere.x as u32) as usize];
        assert!(cd < 0.15, "river-mouth port should be coastal (coast_dist={cd})");
    }

    #[test]
    fn resolution_is_deterministic() {
        let (_, a) = resolve();
        let (_, b) = resolve();
        let pa: Vec<_> = a.iter().map(|l| (l.id.clone(), l.x, l.y)).collect();
        let pb: Vec<_> = b.iter().map(|l| (l.id.clone(), l.x, l.y)).collect();
        assert_eq!(pa, pb);
    }

    #[test]
    fn named_river_matches_the_channel_at_its_mouth() {
        let spec = island();
        let c = GeoCanvas::from_spec(&spec, 42);
        let hf = HeightField::generate(&spec, &c);
        let hydro = Hydrology::compute(&hf, DEFAULT_RIVER_THRESHOLD, DEFAULT_SEA_LEVEL);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        let m = match_rivers_to_channels(&spec, &hf, &hydro, &coast);
        // The Ashflow (mouth: cardinal southwest) is matched to a channel.
        let &ci = m.get("ash").expect("ash matched to a channel");
        let chan = &hydro.rivers[ci];
        let &(mx, my) = chan.last().unwrap();
        // Its mouth sits in the south-west quadrant (x small, y large).
        assert!(
            (mx as f32) < hf.width as f32 * 0.6 && (my as f32) > hf.height as f32 * 0.4,
            "ash channel mouth is toward the SW ({mx},{my})"
        );
        // Deterministic + unique.
        assert_eq!(match_rivers_to_channels(&spec, &hf, &hydro, &coast).get("ash"), Some(&ci));
    }
}
