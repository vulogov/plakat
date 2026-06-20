//! MAP-3b — vector export. The geometry engine already holds the map as points
//! and polylines (landmarks, rivers, roads) + a land/sea mask; this serializes
//! them to **GeoJSON** (`serde_json`) and **SVG** (string assembly) — no new dep,
//! and a pure function of (spec, seed) like every other map artifact.
//!
//! Coordinates are normalized to `[0,1]`, x east / **y north** (pixel-y flipped,
//! so the export is up-is-north regardless of the raster's top-left origin).

use anyhow::Result;
use std::fmt::Write as _;
use std::path::Path;

use super::coastline::{Coastline, DEFAULT_SEA_LEVEL};
use super::engine::{GeoCanvas, HeightField};
use super::hydrology::{Hydrology, DEFAULT_RIVER_THRESHOLD};
use super::resolver::{resolve_landmarks, ResolvedLandmark};
use super::roads::{build_roads, RoadGeom};
use super::spec::MapSpec;

/// The map's vector geometry, in pixel space + the canvas size for normalization.
pub struct VectorMap {
    pub width: u32,
    pub height: u32,
    pub coast_rings: Vec<Vec<(u32, u32)>>,
    pub rivers: Vec<Vec<(u32, u32)>>,
    pub roads: Vec<RoadGeom>,
    pub landmarks: Vec<ResolvedLandmark>,
}

impl VectorMap {
    /// Build the vector geometry for `(spec, seed)`.
    pub fn build(spec: &MapSpec, seed: u64) -> Result<VectorMap> {
        let canvas = GeoCanvas::from_spec(spec, seed);
        let hf = HeightField::generate(spec, &canvas);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        let hydro = Hydrology::compute(&hf, DEFAULT_RIVER_THRESHOLD, DEFAULT_SEA_LEVEL);
        let landmarks = resolve_landmarks(spec, &hf, &hydro, &coast)?;
        let roads = build_roads(spec, &hf, &coast, &hydro, &landmarks);
        let coast_rings = trace_coast_rings(&coast);
        Ok(VectorMap {
            width: hf.width,
            height: hf.height,
            coast_rings,
            rivers: hydro.rivers,
            roads,
            landmarks,
        })
    }

    /// Normalize a pixel to `[0,1]`, y flipped to north-up.
    fn norm(&self, x: u32, y: u32) -> (f64, f64) {
        (x as f64 / self.width as f64, 1.0 - y as f64 / self.height as f64)
    }
}

// ── GeoJSON ──────────────────────────────────────────────────────────────────

/// A `FeatureCollection` string: coast polygons, river + road LineStrings,
/// landmark Points (with `name`/`kind` properties).
pub fn to_geojson(vm: &VectorMap, spec: &MapSpec) -> String {
    use serde_json::{json, Value};

    let line = |pts: &[(u32, u32)]| -> Value {
        Value::Array(pts.iter().map(|&(x, y)| {
            let (nx, ny) = vm.norm(x, y);
            json!([round6(nx), round6(ny)])
        }).collect())
    };

    let mut features: Vec<Value> = Vec::new();

    for (i, ring) in vm.coast_rings.iter().enumerate() {
        features.push(json!({
            "type": "Feature",
            "properties": { "class": "coastline", "id": format!("coast_{i}") },
            "geometry": { "type": "LineString", "coordinates": line(ring) }
        }));
    }
    for (rv, chan) in spec.water.rivers.iter().zip(vm.rivers.iter()) {
        features.push(json!({
            "type": "Feature",
            "properties": { "class": "river", "id": rv.id, "name": rv.name },
            "geometry": { "type": "LineString", "coordinates": line(chan) }
        }));
    }
    // Any unnamed extra channels still export (no spec row).
    for (i, chan) in vm.rivers.iter().enumerate().skip(spec.water.rivers.len()) {
        features.push(json!({
            "type": "Feature",
            "properties": { "class": "river", "id": format!("channel_{i}") },
            "geometry": { "type": "LineString", "coordinates": line(chan) }
        }));
    }
    for r in &vm.roads {
        features.push(json!({
            "type": "Feature",
            "properties": { "class": "road", "id": r.id },
            "geometry": { "type": "LineString", "coordinates": line(&r.path) }
        }));
    }
    for lm in &vm.landmarks {
        let (nx, ny) = vm.norm(lm.x as u32, lm.y as u32);
        features.push(json!({
            "type": "Feature",
            "properties": { "class": "landmark", "id": lm.id, "name": lm.name, "kind": lm.kind.as_str() },
            "geometry": { "type": "Point", "coordinates": [round6(nx), round6(ny)] }
        }));
    }

    let fc = json!({
        "type": "FeatureCollection",
        "name": spec.name,
        "properties": { "scale_tier": spec.scale_tier, "crs": "normalized-0to1-north-up" },
        "features": features
    });
    serde_json::to_string_pretty(&fc).unwrap_or_else(|_| "{}".into())
}

// ── SVG ──────────────────────────────────────────────────────────────────────

/// A standalone SVG (viewBox = the canvas px). Coast filled land, blue rivers,
/// brown roads, landmark dots + `<text>` labels — scalable for print / editing.
pub fn to_svg(vm: &VectorMap, spec: &MapSpec) -> String {
    let (w, h) = (vm.width, vm.height);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
         viewBox=\"0 0 {w} {h}\" font-family=\"serif\">"
    );
    let _ = writeln!(s, "  <rect width=\"{w}\" height=\"{h}\" fill=\"#bccbcf\"/>");

    // Coast rings as filled land polygons (pixel space — SVG y is already top-down).
    for ring in &vm.coast_rings {
        if ring.len() < 3 {
            continue;
        }
        let pts: String = ring.iter().map(|&(x, y)| format!("{x},{y} ")).collect();
        let _ = writeln!(s, "  <polygon points=\"{}\" fill=\"#e9dbbf\" stroke=\"#3a2a18\" stroke-width=\"1.5\"/>", pts.trim());
    }
    // Rivers.
    for chan in &vm.rivers {
        let _ = writeln!(s, "  <polyline points=\"{}\" fill=\"none\" stroke=\"#46728c\" stroke-width=\"1.5\"/>", polyline_pts(chan));
    }
    // Roads (dashed).
    for r in &vm.roads {
        let _ = writeln!(s, "  <polyline points=\"{}\" fill=\"none\" stroke=\"#7a4620\" stroke-width=\"1.5\" stroke-dasharray=\"4 2\"/>", polyline_pts(&r.path));
    }
    // Landmarks: a dot + a label.
    for lm in &vm.landmarks {
        let (x, y) = (lm.x as i32, lm.y as i32);
        let _ = writeln!(s, "  <circle cx=\"{x}\" cy=\"{y}\" r=\"3\" fill=\"#3a2a18\"/>");
        let _ = writeln!(
            s,
            "  <text x=\"{}\" y=\"{}\" font-size=\"9\" fill=\"#3a2a18\">{}</text>",
            x + 5, y + 3, xml_escape(&lm.name)
        );
    }
    let _ = writeln!(s, "  <text x=\"{}\" y=\"18\" font-size=\"16\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#3a2a18\">{}</text>", w / 2, xml_escape(&spec.name));
    s.push_str("</svg>\n");
    s
}

/// Write GeoJSON or SVG by extension (`.svg` → SVG, else GeoJSON).
pub fn save(vm: &VectorMap, spec: &MapSpec, path: &Path) -> Result<()> {
    let body = if path.extension().map(|e| e.eq_ignore_ascii_case("svg")).unwrap_or(false) {
        to_svg(vm, spec)
    } else {
        to_geojson(vm, spec)
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, body).map_err(|e| anyhow::anyhow!("writing {}: {e}", path.display()))
}

// ── Coast contour tracing ────────────────────────────────────────────────────

/// Trace the land/sea boundary into closed pixel rings via Moore-neighbour
/// contour following. Each connected land blob touching the sea yields one ring;
/// rings are emitted in raster-scan start order (deterministic).
fn trace_coast_rings(coast: &Coastline) -> Vec<Vec<(u32, u32)>> {
    let (w, h) = (coast.width as i32, coast.height as i32);
    let land = |x: i32, y: i32| -> bool {
        x >= 0 && y >= 0 && x < w && y < h && !coast.sea[(y * w + x) as usize]
    };
    // A boundary land cell: land with a sea (or off-grid) 4-neighbour.
    let is_boundary = |x: i32, y: i32| -> bool {
        land(x, y) && (!land(x + 1, y) || !land(x - 1, y) || !land(x, y + 1) || !land(x, y - 1))
    };

    let mut visited = vec![false; (w * h) as usize];
    let mut rings = Vec::new();
    // Moore 8-neighbourhood, clockwise from east.
    const N8: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

    for sy in 0..h {
        for sx in 0..w {
            if !is_boundary(sx, sy) || visited[(sy * w + sx) as usize] {
                continue;
            }
            // Walk the contour starting at (sx,sy). Bound the steps so a pathological
            // case can't loop forever.
            let mut ring = Vec::new();
            let (mut cx, mut cy) = (sx, sy);
            let mut dir = 0usize; // came-from search direction
            let max_steps = (w * h * 2) as usize;
            for _ in 0..max_steps {
                if !visited[(cy * w + cx) as usize] {
                    visited[(cy * w + cx) as usize] = true;
                    ring.push((cx as u32, cy as u32));
                }
                // Search neighbours clockwise for the next boundary cell.
                let mut found = false;
                for k in 0..8 {
                    let nd = (dir + k) % 8;
                    let (nx, ny) = (cx + N8[nd].0, cy + N8[nd].1);
                    if is_boundary(nx, ny) {
                        cx = nx;
                        cy = ny;
                        // Resume the next search "behind" the incoming edge.
                        dir = (nd + 6) % 8;
                        found = true;
                        break;
                    }
                }
                if !found || (cx == sx && cy == sy && ring.len() > 2) {
                    break;
                }
            }
            if ring.len() >= 3 {
                rings.push(ring);
            }
        }
    }
    rings
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn polyline_pts(pts: &[(u32, u32)]) -> String {
    pts.iter().map(|&(x, y)| format!("{x},{y} ")).collect::<String>().trim_end().to_string()
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn island() -> MapSpec {
        serde_json::from_str(include_str!("../../corpus/map/island.spec.json")).unwrap()
    }

    #[test]
    fn geojson_has_every_feature_class() {
        let vm = VectorMap::build(&island(), 42).unwrap();
        let gj = to_geojson(&vm, &island());
        let v: serde_json::Value = serde_json::from_str(&gj).unwrap();
        assert_eq!(v["type"], "FeatureCollection");
        let classes: Vec<&str> = v["features"].as_array().unwrap().iter()
            .map(|f| f["properties"]["class"].as_str().unwrap()).collect();
        assert!(classes.contains(&"coastline"), "coastline exported");
        assert!(classes.contains(&"river"), "river exported");
        assert!(classes.contains(&"road"), "road exported");
        assert!(classes.contains(&"landmark"), "landmark exported");
        // The named port survives with its properties.
        let port = v["features"].as_array().unwrap().iter()
            .find(|f| f["properties"]["id"] == "saltmere").unwrap();
        assert_eq!(port["properties"]["kind"], "port");
        assert_eq!(port["geometry"]["type"], "Point");
    }

    #[test]
    fn coast_rings_are_closed_loops() {
        let vm = VectorMap::build(&island(), 42).unwrap();
        assert!(!vm.coast_rings.is_empty(), "island has a coastline");
        let main = vm.coast_rings.iter().max_by_key(|r| r.len()).unwrap();
        assert!(main.len() > 50, "main island ring is substantial ({} cells)", main.len());
    }

    #[test]
    fn svg_is_well_formed_and_escaped() {
        let vm = VectorMap::build(&island(), 42).unwrap();
        let svg = to_svg(&vm, &island());
        assert!(svg.starts_with("<svg"));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("polygon"), "coast polygon present");
        assert!(svg.contains("Saltmere"), "landmark label present");
        // û in "Vethûn" is fine in UTF-8; & would be escaped if present.
        assert!(!svg.contains(" & "), "raw ampersands escaped");
    }

    #[test]
    fn export_is_deterministic() {
        let a = to_geojson(&VectorMap::build(&island(), 42).unwrap(), &island());
        let b = to_geojson(&VectorMap::build(&island(), 42).unwrap(), &island());
        assert_eq!(a, b);
    }
}
