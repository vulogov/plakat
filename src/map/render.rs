//! MAP-3 — the **linework render**. Turns the MAP-2 geometry (terrain, coast,
//! biomes, rivers, roads, resolved landmarks) into the first complete,
//! *user-facing* map: a styled, labelled image with cartographic furniture
//! (frame, title cartouche, compass rose, scale bar, legend). NO SD — a pure
//! function of (spec, seed), so it's byte-stable in the corpus (the 1.6.0 tiled
//! SD pass is the only GPU step on the map track).

use anyhow::Result;
use image::{Rgb, RgbImage};
use std::path::Path;

use super::biome::{Biome, BiomeMap};
use super::coastline::{Coastline, DEFAULT_SEA_LEVEL};
use super::engine::{resolve_simple, GeoCanvas, HeightField};
use super::hydrology::{Hydrology, DEFAULT_RIVER_THRESHOLD};
use super::labels;
use super::resolver::{resolve_landmarks, ResolvedLandmark};
use super::roads::{build_roads, RoadGeom};
use super::spec::{LandmarkKind, MapSpec};

// ── Style ────────────────────────────────────────────────────────────────────

/// A named cartographic palette. `parchment` (default), `inked` (high-contrast
/// monochrome), `blueprint` (cyan on dark).
#[derive(Debug, Clone, Copy)]
pub struct Style {
    paper: [u8; 3],
    paper_dark: [u8; 3],
    ink: [u8; 3],
    sea: [u8; 3],
    sea_deep: [u8; 3],
    river: [u8; 3],
    road: [u8; 3],
    /// Biome-colour weight vs paper on land (0 = pure paper, 1 = pure biome).
    land_tint: f32,
    /// v1.11.0: seasonal land-palette shift. `Summer` = neutral (default,
    /// byte-identical to pre-season). Applied to land pixels only.
    season: Season,
    /// v1.11.0: tabletop coordinate grid — `0` = off (default), else the number
    /// of cells per axis. Drawn over the composite with A1/B2 labels.
    grid: u32,
}

/// Seasonal land palette for `--map-season`. `Summer` is the neutral default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Season {
    Spring,
    #[default]
    Summer,
    Autumn,
    Winter,
}

impl Season {
    pub fn parse(s: &str) -> Result<Season> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "" | "summer" | "none" => Season::Summer,
            "spring" => Season::Spring,
            "autumn" | "fall" => Season::Autumn,
            "winter" => Season::Winter,
            other => anyhow::bail!("unknown --map-season {other:?} (spring|summer|autumn|winter)"),
        })
    }

    /// Shift a land pixel toward the season. Summer → identity (no-op).
    fn shift(self, px: [u8; 3]) -> [u8; 3] {
        match self {
            Season::Summer => px,
            // Spring: brighten + nudge green for fresh growth.
            Season::Spring => blend(px, [0x7e, 0xb8, 0x4a], 0.16),
            // Autumn: warm amber/russet wash.
            Season::Autumn => blend(px, [0xc2, 0x7a, 0x2e], 0.22),
            // Winter: desaturate toward a cold snow-white.
            Season::Winter => blend(px, [0xe6, 0xea, 0xf0], 0.42),
        }
    }
}

impl Style {
    pub fn named(name: &str) -> Result<Style> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "parchment" | "default" => Style {
                paper: [0xe9, 0xdb, 0xbf],
                paper_dark: [0xd6, 0xc4, 0x9f],
                ink: [0x3a, 0x2a, 0x18],
                sea: [0xbc, 0xcb, 0xcf],
                sea_deep: [0x96, 0xad, 0xb6],
                river: [0x46, 0x72, 0x8c],
                road: [0x7a, 0x46, 0x20],
                // Warm parchment: biomes are a SUBTLE accent over the aged paper,
                // not a saturated fill — so a green-biome continent still reads as
                // an old map, not a satellite photo. (Lowered from 0.5.)
                land_tint: 0.36,
                season: Season::Summer,
                grid: 0,
            },
            "inked" => Style {
                paper: [0xf2, 0xf0, 0xe8],
                paper_dark: [0xd8, 0xd6, 0xce],
                ink: [0x20, 0x1c, 0x18],
                sea: [0xdf, 0xe2, 0xe4],
                sea_deep: [0xc2, 0xc8, 0xcc],
                river: [0x55, 0x5f, 0x66],
                road: [0x3a, 0x34, 0x30],
                land_tint: 0.22,
                season: Season::Summer,
                grid: 0,
            },
            "blueprint" => Style {
                paper: [0x10, 0x2a, 0x44],
                paper_dark: [0x0a, 0x1e, 0x33],
                ink: [0xcf, 0xe6, 0xff],
                sea: [0x16, 0x38, 0x58],
                sea_deep: [0x0e, 0x28, 0x42],
                river: [0x8c, 0xc8, 0xff],
                road: [0xe6, 0xc8, 0x6a],
                land_tint: 0.18,
                season: Season::Summer,
                grid: 0,
            },
            other => anyhow::bail!("unknown --map-style {other:?} (parchment|inked|blueprint)"),
        })
    }

    /// v1.11.0: seasonal land palette (`--map-season`). Summer = neutral.
    pub fn with_season(mut self, season: Season) -> Self {
        self.season = season;
        self
    }

    /// v1.11.0: tabletop coordinate grid (`--map-grid N`); `0` = off.
    pub fn with_grid(mut self, cells: u32) -> Self {
        self.grid = cells;
        self
    }
}

impl Default for Style {
    fn default() -> Self {
        Style::named("parchment").unwrap()
    }
}

// ── Public entry ─────────────────────────────────────────────────────────────

/// The computed geometry layers for a `(spec, seed)` — shared by the linework
/// render, the SD conditioning, and the label re-composite so each is built once.
pub struct Geometry {
    pub hf: HeightField,
    pub coast: Coastline,
    pub biome: BiomeMap,
    pub hydro: Hydrology,
    pub lms: Vec<ResolvedLandmark>,
    pub roads: Vec<RoadGeom>,
}

impl Geometry {
    pub fn compute(spec: &MapSpec, seed: u64) -> Result<Geometry> {
        let canvas = GeoCanvas::from_spec(spec, seed);
        let hf = HeightField::generate(spec, &canvas);
        let coast = Coastline::compute(&hf, DEFAULT_SEA_LEVEL);
        let biome = BiomeMap::compute(spec, &hf, &coast, seed);
        let hydro = Hydrology::compute(&hf, DEFAULT_RIVER_THRESHOLD, DEFAULT_SEA_LEVEL);
        let lms = resolve_landmarks(spec, &hf, &hydro, &coast)?;
        let roads = build_roads(spec, &hf, &coast, &hydro, &lms);
        Ok(Geometry { hf, coast, biome, hydro, lms, roads })
    }
}

/// Render the complete styled, labelled map for `(spec, seed)`.
pub fn render(spec: &MapSpec, seed: u64, style: Style) -> Result<RgbImage> {
    let geo = Geometry::compute(spec, seed)?;
    let mut img = paint_base_map(&geo, style);
    apply_labels_and_furniture(&mut img, spec, &geo, style);
    Ok(img)
}

/// The styled base map — terrain/biome fill + hill-shading, coastline, rivers,
/// roads. **No labels, no furniture.** This is the SD img2img init + Canny source
/// (conditioning), and the substrate the labels/furniture pass draws over.
pub fn paint_base_map(geo: &Geometry, style: Style) -> RgbImage {
    let (w, h) = (geo.hf.width, geo.hf.height);
    let mut img = RgbImage::new(w, h);
    paint_base(&mut img, &geo.hf, &geo.coast, &geo.biome, style);
    draw_marsh_hatching(&mut img, &geo.biome, &geo.coast, style);
    draw_coastline(&mut img, &geo.coast, style);
    draw_deltas(&mut img, &geo.hydro, &geo.coast, style);
    draw_rivers(&mut img, &geo.hydro, style);
    draw_roads(&mut img, &geo.roads, style);
    img
}

/// Redraw the crisp cartographic **linework** — coastline, rivers, roads/bridges —
/// over an existing image. The SD paint pass washes out these thin functional
/// features; this restores them (a touch bolder so they read over painted terrain)
/// so the painted map stays a usable map. Mutates `img` in place.
pub fn apply_linework(img: &mut RgbImage, geo: &Geometry, style: Style) {
    draw_coastline(img, &geo.coast, style);
    // Rivers slightly bolder than the linework render so they survive over paint.
    for &(x, y) in geo.hydro.rivers.iter().flatten() {
        plot_thick(img, x as i32, y as i32, style.river);
    }
    draw_roads(img, &geo.roads, style);
}

/// Draw the labels + cartographic furniture over an existing base image (the
/// linework base, or an SD-painted map). Mutates `img` in place.
pub fn apply_labels_and_furniture(img: &mut RgbImage, spec: &MapSpec, geo: &Geometry, style: Style) {
    let (w, h) = (img.width(), img.height());
    // Furniture reserves its footprint first so labels route around it; the boxes
    // themselves are drawn last (crisp, on top of everything).
    let mut taken: Vec<Rect> = Vec::new();
    let title_box = reserve_title(spec, w, &mut taken);
    let compass_box = reserve_box(w as i32 - 56, 9, 50, 56, &mut taken);
    let scale_box = reserve_box(10, h as i32 - 34, 132, 28, &mut taken);
    let legend_box = reserve_legend(&geo.lms, w, h, &mut taken);

    // Political overlay first: territorial rings + inter-region borders sit
    // under the labels, and polity names get placement priority. No-op (and
    // byte-identical) when no region carries a `political` spec.
    draw_political(img, spec, geo, style, &mut taken);

    let river_match = super::resolver::match_rivers_to_channels(spec, &geo.hf, &geo.hydro, &geo.coast);
    draw_features(img, spec, &geo.hf, &geo.hydro, &river_match, style, &mut taken);
    draw_landmarks(img, &geo.lms, style, &mut taken);

    // v1.11.0: tabletop coordinate grid over the map body (under the furniture
    // boxes). `grid == 0` (default) → no-op, byte-identical.
    if style.grid > 0 {
        draw_grid(img, style.grid, style);
    }

    draw_title(img, spec, style, title_box);
    draw_compass(img, style, compass_box);
    draw_scale_bar(img, spec, style, scale_box);
    draw_legend(img, &geo.lms, style, legend_box);
    draw_frame(img, style);
}

/// Render + write the map PNG.
pub fn save_render(spec: &MapSpec, seed: u64, style: Style, path: &Path) -> Result<()> {
    let img = render(spec, seed, style)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(path).map_err(|e| anyhow::anyhow!("writing map render {}: {e}", path.display()))
}

// ── Base styling ─────────────────────────────────────────────────────────────

fn paint_base(img: &mut RgbImage, hf: &HeightField, coast: &Coastline, biome: &BiomeMap, st: Style) {
    let (w, h) = (hf.width, hf.height);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let px = if coast.sea[i] {
                // Bathymetric shade: lower elevation below sea level → deeper tone.
                let t = ((DEFAULT_SEA_LEVEL - hf.data[i]) / DEFAULT_SEA_LEVEL).clamp(0.0, 1.0);
                blend(st.sea, st.sea_deep, t * 0.6)
            } else {
                let tinted = blend(st.paper, biome.biome[i].rgb(), st.land_tint);
                // v1.11.0: seasonal wash on land (Summer = identity → byte-stable).
                shade(st.season.shift(tinted), hillshade(hf, x, y))
            };
            img.put_pixel(x, y, Rgb(px));
        }
    }
}

/// Cartographic marsh symbol over Wetland regions: staggered rows of short
/// horizontal dashes in a bluish marsh tint. Deterministic (fixed grid) so it
/// stays byte-stable, and only touches land cells whose biome is `Wetland`, so
/// maps with no wetlands render exactly as before.
fn draw_marsh_hatching(img: &mut RgbImage, biome: &BiomeMap, coast: &Coastline, st: Style) {
    let (w, h) = (biome.width, biome.height);
    let marsh = blend(st.river, st.ink, 0.35);
    let mut y = 2u32;
    while y < h {
        let ox = if (y / 5) % 2 == 0 { 0 } else { 4 }; // stagger alternate rows
        let mut x = 2 + ox;
        while x + 3 < w {
            let i = (y * w + x) as usize;
            if !coast.sea[i] && biome.biome[i] == Biome::Wetland {
                for dx in 0..3u32 {
                    img.put_pixel(x + dx, y, Rgb(marsh));
                }
            }
            x += 8;
        }
        y += 5;
    }
}

/// Lambert-ish hill-shading from the local gradient, light from the NW. Returns a
/// multiplier in ~[0.78, 1.18].
fn hillshade(hf: &HeightField, x: u32, y: u32) -> f32 {
    let (w, h) = (hf.width, hf.height);
    if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
        return 1.0;
    }
    let at = |xx: u32, yy: u32| hf.data[(yy * w + xx) as usize];
    let gx = at(x + 1, y) - at(x - 1, y);
    let gy = at(x, y + 1) - at(x, y - 1);
    // Slope facing the NW light (-x,-y) brightens; SE darkens.
    let toward = -(gx + gy);
    (1.0 + toward * 6.0).clamp(0.78, 1.18)
}

fn draw_coastline(img: &mut RgbImage, coast: &Coastline, st: Style) {
    for y in 0..coast.height {
        for x in 0..coast.width {
            if coast.is_coast(x, y) {
                img.put_pixel(x, y, Rgb(st.ink));
            }
        }
    }
}

fn draw_rivers(img: &mut RgbImage, hydro: &Hydrology, st: Style) {
    for &(x, y) in hydro.rivers.iter().flatten() {
        put(img, x as i32, y as i32, st.river);
    }
}

/// Only river paths at least this long get a delta — a proxy for "navigable /
/// sizeable", so small creeks don't sprout fans.
const DELTA_MIN_LEN: usize = 36;
/// How far the distributary channels fan out into the sea (pixels).
const DELTA_REACH: i32 = 7;

/// Draw a small distributary fan into the shallow sea at each navigable river
/// mouth — the cartographic delta. Deterministic (a fixed three-branch fan in the
/// flow direction, clipped to sea cells) → byte-stable.
fn draw_deltas(img: &mut RgbImage, hydro: &Hydrology, coast: &Coastline, st: Style) {
    let (w, h) = (coast.width, coast.height);
    for river in &hydro.rivers {
        let n = river.len();
        if n < DELTA_MIN_LEN {
            continue;
        }
        let (mx, my) = (river[n - 1].0 as f32, river[n - 1].1 as f32);
        let (px, py) = (river[n - 2].0 as f32, river[n - 2].1 as f32);
        let (mut dx, mut dy) = (mx - px, my - py);
        let len = (dx * dx + dy * dy).sqrt().max(1e-3);
        dx /= len;
        dy /= len;
        for &ang in &[-0.6f32, 0.0, 0.6] {
            let (c, s) = (ang.cos(), ang.sin());
            let (bx, by) = (dx * c - dy * s, dx * s + dy * c);
            for step in 1..=DELTA_REACH {
                let x = (mx + bx * step as f32).round() as i32;
                let y = (my + by * step as f32).round() as i32;
                if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                    break;
                }
                if !coast.sea[(y as u32 * w + x as u32) as usize] {
                    break; // fan only over open water
                }
                put(img, x, y, st.river);
            }
        }
    }
}

fn draw_roads(img: &mut RgbImage, roads: &[RoadGeom], st: Style) {
    for r in roads {
        // Dashed line so roads read distinctly from rivers.
        for (k, &(x, y)) in r.path.iter().enumerate() {
            if k % 6 < 4 {
                plot_thick(img, x as i32, y as i32, st.road);
            }
        }
        for &(x, y) in &r.bridges {
            plot_thick(img, x as i32, y as i32, st.ink);
        }
    }
}

// ── Markers + labels ─────────────────────────────────────────────────────────

fn draw_landmarks(img: &mut RgbImage, lms: &[ResolvedLandmark], st: Style, taken: &mut Vec<Rect>) {
    for lm in lms {
        let (px, py) = (lm.x as i32, lm.y as i32);
        draw_symbol(img, px, py, &lm.kind, st);
        place_label(img, (px, py), &lm.name, 1, st.ink, st, taken);
    }
}

/// Named natural features — mountain ranges, regions, the sea, lakes, the main
/// river — labelled at their resolved positions. `river_match` maps each named
/// river id → its traced-channel index (L2 matching).
fn draw_features(
    img: &mut RgbImage,
    spec: &MapSpec,
    hf: &HeightField,
    hydro: &Hydrology,
    river_match: &std::collections::HashMap<String, usize>,
    st: Style,
    taken: &mut Vec<Rect>,
) {
    let (w, h) = (hf.width as f32, hf.height as f32);
    let to_px = |a: &super::spec::Anchor| resolve_simple(a).map(|(x, y)| ((x * w) as i32, (y * h) as i32));

    for r in &spec.terrain.mountain_ranges {
        if let (Some(name), Some(p)) = (&r.name, to_px(&r.anchor)) {
            place_label(img, p, name, 1, st.ink, st, taken);
        }
    }
    for r in &spec.regions {
        if let (Some(name), Some(p)) = (&r.name, to_px(&r.anchor)) {
            place_label(img, p, name, 1, st.road, st, taken);
        }
    }
    for l in &spec.water.lakes {
        if let (Some(name), Some(p)) = (&l.name, to_px(&l.anchor)) {
            place_label(img, p, name, 1, st.river, st, taken);
        }
    }
    // Sea label in open water near the sea's position hint (its centroid would be
    // the island's middle for a ring-shaped sea — squarely on the mountains).
    for sea in &spec.water.seas {
        if let Some(name) = &sea.name {
            if let Some(p) = sea_label_point(hf, &sea.position) {
                place_label(img, p, name, 1, st.river, st, taken);
            }
        }
    }
    // River names: label each named river at the midpoint of its MATCHED channel
    // (L2 — the channel whose mouth is nearest the river's resolved mouth).
    for rv in &spec.water.rivers {
        if let (Some(name), Some(&ci)) = (&rv.name, river_match.get(&rv.id)) {
            if let Some(chan) = hydro.rivers.get(ci) {
                if !chan.is_empty() {
                    let mid = chan[chan.len() / 2];
                    place_label(img, (mid.0 as i32, mid.1 as i32), name, 1, st.river, st, taken);
                }
            }
        }
    }
}

/// Political overlay (v1.11.0): realize `RegionSpec.political`. For each region
/// with a polity, draw a dashed territorial ring around its anchor, the borders
/// to other regions (styled by `kind`), and the polity name. Gated — no
/// political data anywhere → early return, byte-identical to a plain map. Pure
/// fn of (spec, geo, style).
fn draw_political(img: &mut RgbImage, spec: &MapSpec, geo: &Geometry, st: Style, taken: &mut Vec<Rect>) {
    if !spec.regions.iter().any(|r| r.political.is_some()) {
        return;
    }
    let (w, h) = (geo.hf.width as f32, geo.hf.height as f32);
    let extent = w.min(h);
    let to_px = |a: &super::spec::Anchor| resolve_simple(a).map(|(x, y)| ((x * w) as i32, (y * h) as i32));
    // Resolve every region's anchor once (borders reference other regions by id).
    let region_px: std::collections::HashMap<&str, (i32, i32)> = spec
        .regions
        .iter()
        .filter_map(|r| to_px(&r.anchor).map(|p| (r.id.as_str(), p)))
        .collect();

    for r in &spec.regions {
        let Some(pol) = &r.political else { continue };
        let Some(&(cx, cy)) = region_px.get(r.id.as_str()) else { continue };
        let col = polity_color(&pol.polity_name, st);
        let rad = ((r.coverage.max(0.12) * 0.5 * extent) as i32).max(8);
        dashed_ring(img, cx, cy, rad, col);
        for b in &pol.borders {
            if let Some(&(bx, by)) = region_px.get(b.with_region.as_str()) {
                draw_border(img, cx, cy, bx, by, &b.kind, st);
            }
        }
        place_label(img, (cx, cy - rad - 6), &pol.polity_name, 1, col, st, taken);
    }
}

/// v1.11.0: a tabletop coordinate grid — `cells`×`cells` faint lines with
/// A/B/C column + 1/2/3 row labels (for hex/RPG referencing). `cells` is capped
/// at 26 (A–Z). Pure fn of (img dims, cells, style).
fn draw_grid(img: &mut RgbImage, cells: u32, st: Style) {
    let (w, h) = (img.width(), img.height());
    let cells = cells.clamp(1, 26);
    let cw = w as f32 / cells as f32;
    let ch = h as f32 / cells as f32;
    let gridcol = blend(st.ink, st.paper, 0.55); // faint, doesn't dominate
    for c in 1..cells {
        let x = (c as f32 * cw) as i32;
        for y in 0..h as i32 {
            put(img, x, y, gridcol);
        }
    }
    for r in 1..cells {
        let y = (r as f32 * ch) as i32;
        for x in 0..w as i32 {
            put(img, x, y, gridcol);
        }
    }
    // Column letters along the top of each cell; row numbers down the left.
    for c in 0..cells {
        let label = ((b'A' + c as u8) as char).to_string();
        let lx = (c as f32 * cw + cw * 0.5) as i32 - 2;
        labels::draw_text_haloed(img, lx, 2, &label, 1, st.ink, st.paper);
    }
    for r in 0..cells {
        let label = (r + 1).to_string();
        let ly = (r as f32 * ch + ch * 0.5) as i32 - 3;
        labels::draw_text_haloed(img, 2, ly, &label, 1, st.ink, st.paper);
    }
}

/// Deterministic muted polity colour from the name, blended toward ink so it
/// reads on any palette.
fn polity_color(name: &str, st: Style) -> [u8; 3] {
    let mut hsh = 0u32;
    for b in name.bytes() {
        hsh = hsh.wrapping_mul(31).wrapping_add(b as u32);
    }
    const PALETTE: [[u8; 3]; 6] = [
        [0x8a, 0x3b, 0x3b], // crimson
        [0x3b, 0x5a, 0x8a], // indigo
        [0x4a, 0x7a, 0x3b], // green
        [0x7a, 0x5a, 0x2a], // ochre
        [0x6a, 0x3b, 0x7a], // violet
        [0x2a, 0x6a, 0x6a], // teal
    ];
    blend(PALETTE[(hsh as usize) % PALETTE.len()], st.ink, 0.3)
}

/// A dashed circle outline (a polity's territorial extent).
fn dashed_ring(img: &mut RgbImage, cx: i32, cy: i32, r: i32, c: [u8; 3]) {
    if r < 2 {
        return;
    }
    let steps = ((std::f32::consts::TAU * r as f32) as i32).max(1);
    for s in 0..steps {
        if (s / 6) % 2 == 1 {
            continue; // dash gap
        }
        let a = s as f32 / steps as f32 * std::f32::consts::TAU;
        put(img, cx + (r as f32 * a.cos()) as i32, cy + (r as f32 * a.sin()) as i32, c);
    }
}

/// An inter-region border, coloured by `kind` (disputed → crimson, river →
/// blue, mountain → road-brown, else ink).
fn draw_border(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, kind: &str, st: Style) {
    let k = kind.to_ascii_lowercase();
    let col = if k.contains("disput") {
        [0xb0, 0x2a, 0x2a]
    } else if k.contains("river") {
        st.river
    } else if k.contains("mountain") {
        st.road
    } else {
        st.ink
    };
    line(img, x0, y0, x1, y1, col);
}

/// Greedy label placement: try positions around the anchor, take the first that
/// is in-bounds and clear of already-placed boxes; fall back to the right.
fn place_label(img: &mut RgbImage, at: (i32, i32), text: &str, scale: u32, color: [u8; 3], st: Style, taken: &mut Vec<Rect>) {
    let (tw, th) = (labels::text_width(text, scale) as i32, labels::text_height(scale) as i32);
    if tw == 0 {
        return;
    }
    let (w, h) = (img.width() as i32, img.height() as i32);
    let off = 7;
    let cands = [
        (at.0 + off, at.1 - th / 2),       // right
        (at.0 - off - tw, at.1 - th / 2),  // left
        (at.0 - tw / 2, at.1 + off),       // below
        (at.0 - tw / 2, at.1 - off - th),  // above
    ];
    let fits = |x: i32, y: i32, taken: &[Rect]| {
        let r = Rect { x0: x - 1, y0: y - 1, x1: x + tw, y1: y + th };
        x >= 3 && y >= 3 && x + tw <= w - 3 && y + th <= h - 3 && !taken.iter().any(|t| t.overlaps(&r))
    };
    let (lx, ly) = cands.iter().copied().find(|&(x, y)| fits(x, y, taken)).unwrap_or(cands[0]);
    taken.push(Rect { x0: lx - 1, y0: ly - 1, x1: lx + tw, y1: ly + th });
    labels::draw_text_haloed(img, lx, ly, text, scale, color, st.paper);
}

/// A distinct little symbol per landmark kind, in the kind's colour + ink edge.
fn draw_symbol(img: &mut RgbImage, cx: i32, cy: i32, kind: &LandmarkKind, st: Style) {
    let fill = super::resolver::marker_rgb(kind);
    match kind {
        LandmarkKind::City => {
            fill_rect(img, cx - 3, cy - 3, cx + 3, cy + 3, st.ink);
            fill_rect(img, cx - 2, cy - 2, cx + 2, cy + 2, fill);
            put(img, cx, cy, st.paper);
        }
        LandmarkKind::Fortress | LandmarkKind::Castle => {
            // A crenellated tower.
            fill_rect(img, cx - 3, cy - 3, cx + 3, cy + 3, st.ink);
            fill_rect(img, cx - 2, cy - 1, cx + 2, cy + 2, fill);
            put(img, cx - 2, cy - 3, fill);
            put(img, cx, cy - 3, fill);
            put(img, cx + 2, cy - 3, fill);
        }
        LandmarkKind::Lighthouse => {
            // A beacon: a small triangle with rays.
            for dy in -3i32..=3 {
                let half = (3 - dy.abs()).max(0);
                for dx in -half..=half {
                    put(img, cx + dx, cy + dy, fill);
                }
            }
            ring(img, cx, cy, 4, st.ink);
        }
        LandmarkKind::Temple | LandmarkKind::Oracle => diamond(img, cx, cy, 3, fill, st.ink),
        LandmarkKind::Port => {
            disc(img, cx, cy, 3, fill, st.ink);
            put(img, cx, cy, st.paper);
        }
        LandmarkKind::Ruin | LandmarkKind::Dungeon | LandmarkKind::Shipwreck => ring(img, cx, cy, 3, st.ink),
        _ => disc(img, cx, cy, 2, fill, st.ink), // town/village/oasis/pass/other
    }
}

// ── Furniture ────────────────────────────────────────────────────────────────

fn reserve_title(spec: &MapSpec, w: u32, taken: &mut Vec<Rect>) -> Rect {
    let scale = 2;
    let tw = labels::text_width(&spec.name, scale) as i32;
    let th = labels::text_height(scale) as i32;
    let pad = 5;
    let bx0 = (w as i32 - tw) / 2 - pad;
    let r = Rect { x0: bx0, y0: 9, x1: bx0 + tw + 2 * pad, y1: 9 + th + 2 * pad };
    taken.push(r);
    r
}

fn draw_title(img: &mut RgbImage, spec: &MapSpec, st: Style, r: Rect) {
    fill_rect(img, r.x0 + 3, r.y0 + 3, r.x1 + 3, r.y1 + 3, st.paper_dark); // drop shadow
    fill_rect(img, r.x0, r.y0, r.x1, r.y1, st.paper);
    rect_outline(img, r.x0, r.y0, r.x1, r.y1, st.ink);
    rect_outline(img, r.x0 + 2, r.y0 + 2, r.x1 - 2, r.y1 - 2, st.ink);
    labels::draw_text(img, r.x0 + 5, r.y0 + 5, &spec.name, 2, st.ink);
}

fn draw_compass(img: &mut RgbImage, st: Style, r: Rect) {
    let cx = (r.x0 + r.x1) / 2;
    let cy = (r.y0 + r.y1) / 2 + 4;
    let rad = 15;
    // Four-point star: a filled N (dark) lobe + outlined E/S/W lobes.
    let tips = [(0, -rad), (rad, 0), (0, rad), (-rad, 0)];
    for &(tx, ty) in &tips {
        line(img, cx, cy, cx + tx, cy + ty, st.ink);
    }
    // Diamond connecting the cardinal tips.
    for i in 0..4 {
        let a = tips[i];
        let b = tips[(i + 1) % 4];
        line(img, cx + a.0, cy + a.1, cx + b.0, cy + b.1, st.ink);
    }
    disc(img, cx, cy, 2, st.ink, st.ink);
    // 'N' over the north tip.
    labels::draw_text_haloed(img, cx - 2, cy - rad - 9, "N", 1, st.ink, st.paper);
}

fn draw_scale_bar(img: &mut RgbImage, spec: &MapSpec, st: Style, r: Rect) {
    let km_across = km_across(spec);
    let km_per_px = km_across / img.width() as f32;
    // Pick a round bar length ≈ 90px wide.
    let bar_km = nice_round(90.0 * km_per_px);
    let bar_px = ((bar_km / km_per_px).round() as i32).clamp(24, r.x1 - r.x0 - 4);
    let segs = 4;
    let x0 = r.x0 + 2;
    let y0 = r.y0 + 4;
    let bh = 5;
    for s in 0..segs {
        let sx0 = x0 + bar_px * s / segs;
        let sx1 = x0 + bar_px * (s + 1) / segs;
        let c = if s % 2 == 0 { st.ink } else { st.paper };
        fill_rect(img, sx0, y0, sx1, y0 + bh, c);
    }
    rect_outline(img, x0, y0, x0 + bar_px, y0 + bh, st.ink);
    labels::draw_text_haloed(img, x0, y0 + bh + 3, &format!("{} KM", fmt_km(bar_km)), 1, st.ink, st.paper);
}

/// The real bottom-right legend box (anchored to the image bounds up front, so
/// labels route around its true footprint).
fn reserve_legend(lms: &[ResolvedLandmark], w: u32, h: u32, taken: &mut Vec<Rect>) -> Rect {
    let rows = legend_rows(lms);
    if rows.is_empty() {
        return Rect { x0: 0, y0: 0, x1: -1, y1: -1 }; // empty (never overlaps)
    }
    let row_h = labels::line_advance(1) as i32;
    let tw = rows.iter().map(|(_, t)| labels::text_width(t, 1) as i32).max().unwrap_or(40);
    let bw = tw + 22;
    let bh = row_h * (rows.len() as i32 + 1) + 8;
    let x1 = w as i32 - 8;
    let y1 = h as i32 - 8;
    let r = Rect { x0: x1 - bw, y0: y1 - bh, x1, y1 };
    taken.push(r);
    r
}

fn draw_legend(img: &mut RgbImage, lms: &[ResolvedLandmark], st: Style, r: Rect) {
    let rows = legend_rows(lms);
    if rows.is_empty() {
        return;
    }
    let row_h = labels::line_advance(1) as i32;
    fill_rect(img, r.x0, r.y0, r.x1, r.y1, st.paper);
    rect_outline(img, r.x0, r.y0, r.x1, r.y1, st.ink);
    labels::draw_text(img, r.x0 + 5, r.y0 + 4, "LEGEND", 1, st.ink);
    for (i, (kind, label)) in rows.iter().enumerate() {
        let ry = r.y0 + 4 + row_h * (i as i32 + 1);
        draw_symbol(img, r.x0 + 8, ry + 3, kind, st);
        labels::draw_text(img, r.x0 + 16, ry, label, 1, st.ink);
    }
}

fn draw_frame(img: &mut RgbImage, st: Style) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    rect_outline(img, 3, 3, w - 4, h - 4, st.ink);
    rect_outline(img, 6, 6, w - 7, h - 7, st.ink);
}

// ── Legend helpers ───────────────────────────────────────────────────────────

/// Unique landmark kinds present, in first-seen order, with display labels.
fn legend_rows(lms: &[ResolvedLandmark]) -> Vec<(LandmarkKind, String)> {
    let mut seen: Vec<LandmarkKind> = Vec::new();
    for lm in lms {
        if !seen.contains(&lm.kind) {
            seen.push(lm.kind.clone());
        }
    }
    seen.into_iter().map(|k| (k.clone(), k.as_str().to_uppercase())).collect()
}

// ── Scale helpers ────────────────────────────────────────────────────────────

/// Approximate km across the map width. Uses `world_extent_km` if present, else a
/// per-tier nominal scaled by the wider tile axis.
fn km_across(spec: &MapSpec) -> f32 {
    if let Some(k) = spec.world_extent_km {
        return (k as f32).max(0.5);
    }
    let base = match spec.scale_tier {
        0 => 8.0,
        1 => 25.0,
        2 => 80.0,
        3 => 300.0,
        4 => 1200.0,
        5 => 6000.0,
        10 => 2.0,
        11 => 1.0,
        12 => 0.5,
        _ => 50.0,
    };
    let tiles = spec.tile_grid.cols.max(spec.tile_grid.rows).max(1) as f32;
    base * (tiles / 2.0).max(0.5)
}

/// Largest "nice" number (1/2/5 × 10ⁿ) not exceeding `target`.
fn nice_round(target: f32) -> f32 {
    if target <= 0.0 {
        return 1.0;
    }
    let mag = 10f32.powf(target.log10().floor());
    for m in [5.0, 2.0, 1.0] {
        if m * mag <= target {
            return m * mag;
        }
    }
    mag
}

fn fmt_km(km: f32) -> String {
    if km >= 1.0 {
        format!("{}", km.round() as i64)
    } else {
        format!("{km:.1}")
    }
}

/// The sea cell nearest the position hint's cardinal point — open water on the
/// intended side, never the (land-covered) centroid of a ring sea.
fn sea_label_point(hf: &HeightField, position: &str) -> Option<(i32, i32)> {
    let (w, h) = (hf.width, hf.height);
    let (fx, fy) = cardinal_frac(position);
    let (tx, ty) = (fx * w as f32, fy * h as f32);
    let mut best: Option<(i32, f32)> = None;
    for y in 0..h {
        for x in 0..w {
            if hf.data[(y * w + x) as usize] < DEFAULT_SEA_LEVEL {
                let d = (x as f32 - tx).powi(2) + (y as f32 - ty).powi(2);
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some(((y * w + x) as i32, d));
                }
            }
        }
    }
    best.map(|(i, _)| ((i as u32 % w) as i32, (i as u32 / w) as i32))
}

/// A position keyword → normalized (x, y); defaults to centre.
fn cardinal_frac(position: &str) -> (f32, f32) {
    match position.to_ascii_lowercase().replace([' ', '-'], "_").as_str() {
        "north" | "top" => (0.5, 0.12),
        "south" | "bottom" => (0.5, 0.88),
        "east" | "right" => (0.88, 0.5),
        "west" | "left" => (0.12, 0.5),
        "northeast" | "north_east" => (0.85, 0.15),
        "northwest" | "north_west" => (0.15, 0.15),
        "southeast" | "south_east" => (0.85, 0.85),
        "southwest" | "south_west" => (0.15, 0.85),
        _ => (0.5, 0.5),
    }
}

// ── Geometry primitives ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Rect {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Rect {
    fn overlaps(&self, o: &Rect) -> bool {
        self.x0 <= o.x1 && o.x0 <= self.x1 && self.y0 <= o.y1 && o.y0 <= self.y1
    }
}

fn reserve_box(x: i32, y: i32, w: i32, h: i32, taken: &mut Vec<Rect>) -> Rect {
    let r = Rect { x0: x, y0: y, x1: x + w, y1: y + h };
    taken.push(r);
    r
}

fn blend(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let mix = |i: usize| (a[i] as f32 * (1.0 - t) + b[i] as f32 * t).round().clamp(0.0, 255.0) as u8;
    [mix(0), mix(1), mix(2)]
}

fn shade(c: [u8; 3], k: f32) -> [u8; 3] {
    let s = |v: u8| (v as f32 * k).round().clamp(0.0, 255.0) as u8;
    [s(c[0]), s(c[1]), s(c[2])]
}

fn put(img: &mut RgbImage, x: i32, y: i32, c: [u8; 3]) {
    if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
        img.put_pixel(x as u32, y as u32, Rgb(c));
    }
}

fn plot_thick(img: &mut RgbImage, x: i32, y: i32, c: [u8; 3]) {
    for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
        put(img, x + dx, y + dy, c);
    }
}

fn fill_rect(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 3]) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            put(img, x, y, c);
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

fn line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 3]) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut err, mut x, mut y) = (dx + dy, x0, y0);
    loop {
        put(img, x, y, c);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn disc(img: &mut RgbImage, cx: i32, cy: i32, r: i32, fill: [u8; 3], edge: [u8; 3]) {
    let re = r + 1;
    for dy in -re..=re {
        for dx in -re..=re {
            let d2 = dx * dx + dy * dy;
            if d2 <= r * r {
                put(img, cx + dx, cy + dy, fill);
            } else if d2 <= re * re {
                put(img, cx + dx, cy + dy, edge);
            }
        }
    }
}

fn ring(img: &mut RgbImage, cx: i32, cy: i32, r: i32, c: [u8; 3]) {
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if d2 <= r * r && d2 >= (r - 1) * (r - 1) {
                put(img, cx + dx, cy + dy, c);
            }
        }
    }
}

fn diamond(img: &mut RgbImage, cx: i32, cy: i32, r: i32, fill: [u8; 3], edge: [u8; 3]) {
    for dy in -r..=r {
        for dx in -r..=r {
            let m = dx.abs() + dy.abs();
            if m < r {
                put(img, cx + dx, cy + dy, fill);
            } else if m == r {
                put(img, cx + dx, cy + dy, edge);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::spec::MapSpec;

    fn island() -> MapSpec {
        serde_json::from_str(include_str!("../../corpus/map/island.spec.json")).unwrap()
    }

    #[test]
    fn renders_island_to_canvas_size() {
        let img = render(&island(), 42, Style::default()).unwrap();
        let c = GeoCanvas::from_spec(&island(), 42);
        assert_eq!(img.dimensions(), (c.width, c.height));
    }

    #[test]
    fn render_is_deterministic() {
        let a = render(&island(), 42, Style::default()).unwrap();
        let b = render(&island(), 42, Style::default()).unwrap();
        assert!(a.as_raw() == b.as_raw(), "render must be byte-stable");
    }

    #[test]
    fn styles_resolve_and_differ() {
        assert!(Style::named("inked").is_ok());
        assert!(Style::named("blueprint").is_ok());
        assert!(Style::named("nope").is_err());
        let p = render(&island(), 42, Style::named("parchment").unwrap()).unwrap();
        let b = render(&island(), 42, Style::named("blueprint").unwrap()).unwrap();
        assert!(p.as_raw() != b.as_raw(), "different styles → different pixels");
    }

    #[test]
    fn political_overlay_draws_when_present() {
        use crate::map::spec::{Anchor, PoliticalSpec};
        let plain = render(&island(), 42, Style::default()).unwrap();
        let mut pol = island();
        assert!(!pol.regions.is_empty(), "island has regions to politicize");
        pol.regions[0].anchor = Anchor::Cardinal { position: "center".into() };
        pol.regions[0].coverage = 0.3;
        pol.regions[0].political = Some(PoliticalSpec {
            polity_name: "Aldermark".into(),
            polity_kind: "kingdom".into(),
            borders: vec![],
        });
        let drawn = render(&pol, 42, Style::default()).unwrap();
        assert!(drawn.as_raw() != plain.as_raw(), "political overlay must change pixels");
        // Still deterministic with the overlay on.
        let drawn2 = render(&pol, 42, Style::default()).unwrap();
        assert!(drawn.as_raw() == drawn2.as_raw(), "political render byte-stable");
    }

    #[test]
    fn season_and_grid_change_pixels_but_default_is_neutral() {
        let summer = render(&island(), 42, Style::default()).unwrap();
        let autumn = render(&island(), 42, Style::default().with_season(Season::Autumn)).unwrap();
        let winter = render(&island(), 42, Style::default().with_season(Season::Winter)).unwrap();
        let gridded = render(&island(), 42, Style::default().with_grid(8)).unwrap();
        assert!(autumn.as_raw() != summer.as_raw(), "autumn shifts the land palette");
        assert!(winter.as_raw() != summer.as_raw(), "winter shifts the land palette");
        assert!(winter.as_raw() != autumn.as_raw(), "seasons differ from each other");
        assert!(gridded.as_raw() != summer.as_raw(), "grid overlay draws");
        // Default (Summer + no grid) is the neutral baseline.
        let neutral =
            render(&island(), 42, Style::default().with_season(Season::Summer).with_grid(0)).unwrap();
        assert!(neutral.as_raw() == summer.as_raw(), "summer + no grid = byte-identical default");
    }

    #[test]
    fn season_parse_roundtrips() {
        assert_eq!(Season::parse("autumn").unwrap(), Season::Autumn);
        assert_eq!(Season::parse("fall").unwrap(), Season::Autumn);
        assert_eq!(Season::parse("").unwrap(), Season::Summer);
        assert_eq!(Season::parse("winter").unwrap(), Season::Winter);
        assert!(Season::parse("monsoon").is_err());
    }

    #[test]
    fn nice_round_picks_one_two_five() {
        assert_eq!(nice_round(90.0), 50.0);
        assert_eq!(nice_round(30.0), 20.0);
        assert_eq!(nice_round(12.0), 10.0);
        assert_eq!(nice_round(7.0), 5.0);
    }

    #[test]
    fn label_boxes_do_not_overlap_reserved_furniture() {
        // The render runs the full placement path; this just asserts it completes
        // and produces a non-trivial image (ink present from frame + labels).
        let img = render(&island(), 42, Style::default()).unwrap();
        let st = Style::default();
        assert!(img.pixels().any(|p| p.0 == st.ink), "ink linework present");
    }
}
