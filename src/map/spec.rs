//! `MapSpec v2` — the structured description a `plakat map` is generated from.
//! The LLM parser (MAP-1) produces this from prose; the geometry engine (MAP-2+)
//! consumes it. Every position is a typed [`Anchor`] (a spatial *relationship*),
//! never a raw pixel coordinate — that's what makes the spec LLM-emittable and
//! scale-independent.
//!
//! The urban extension (`LandmarkSpec.urban: Option<UrbanSpec>`) lands in MAP-5;
//! MAP-1 ships the geographic schema + the full [`Anchor`] enum (urban variants
//! included so the type contract is stable).

use serde::{Deserialize, Serialize};

/// Current schema version. Written into every spec; checked on load.
pub const SPEC_VERSION: u32 = 2;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MapSpec {
    pub version: u32,
    pub name: String,
    /// 0–5 geographic (city → hemisphere), 10–12 urban (U0–U2).
    pub scale_tier: u8,
    pub tile_grid: TileGrid,
    #[serde(default)]
    pub world_extent_km: Option<f64>,
    #[serde(default)]
    pub climate: Option<String>,
    #[serde(default)]
    pub era: Option<String>,
    /// BCP-47 language for labels (`ar`, `ru`, `zh`, `en`).
    #[serde(default)]
    pub language: Option<String>,
    pub terrain: TerrainSpec,
    pub water: WaterSpec,
    #[serde(default)]
    pub regions: Vec<RegionSpec>,
    #[serde(default)]
    pub landmarks: Vec<LandmarkSpec>,
    #[serde(default)]
    pub infrastructure: InfrastructureSpec,
    /// MAP-5: the urban fabric (streets, districts, wall, gates, waterfront). Present
    /// for city/town-scale maps (scale tiers 10–12); `None` for geographic maps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub urban: Option<UrbanSpec>,
    #[serde(default)]
    pub bund_hooks: Option<Vec<String>>,
}

impl MapSpec {
    /// A minimal valid spec — the parser's last-resort fallback (never abort).
    pub fn minimal(name: impl Into<String>, cols: u32, rows: u32, tier: u8) -> Self {
        MapSpec {
            version: SPEC_VERSION,
            name: name.into(),
            scale_tier: tier,
            tile_grid: TileGrid { cols, rows },
            world_extent_km: None,
            climate: None,
            era: None,
            language: None,
            terrain: TerrainSpec::default(),
            water: WaterSpec::default(),
            regions: Vec::new(),
            landmarks: Vec::new(),
            infrastructure: InfrastructureSpec::default(),
            urban: None,
            bund_hooks: None,
        }
    }
}

// ── Urban fabric (MAP-5) ─────────────────────────────────────────────────────

/// A city/town plan: a wall ring + gates, named streets + districts, an optional
/// waterfront with piers, and a station. The urban engine (U0–U2) is a pure fn of
/// `(spec, seed)`; named streets/gates/districts label the generated geometry.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct UrbanSpec {
    /// Street-plan style: `"radial"` (medieval radio-concentric — rings + radials),
    /// `"grid"` (planned orthogonal — Roman/colonial), or `"organic"` (irregular
    /// winding lanes). When omitted, inferred from context (walled → radial,
    /// mountainous → organic, plains → grid). Aliases: `concentric`/`medieval`,
    /// `orthogonal`/`planned`, `irregular`/`maze`.
    #[serde(default)]
    pub layout: Option<String>,
    /// A city wall enclosing the built-up area. `None` = an open (unwalled) town.
    #[serde(default)]
    pub wall: Option<WallRing>,
    #[serde(default)]
    pub gates: Vec<GateSpec>,
    #[serde(default)]
    pub streets: Vec<StreetSpec>,
    #[serde(default)]
    pub districts: Vec<UrbanDistrict>,
    /// The edge the city meets water on (`"south"`, `"west"`, …). `None` = inland.
    #[serde(default)]
    pub waterfront: Option<String>,
    #[serde(default)]
    pub piers: Vec<PierSpec>,
    #[serde(default)]
    pub station: Option<NamedPoint>,
}

/// The city wall: a ring at `radius` (fraction of the canvas half-extent) around
/// the centre. `shape` is `"round"` (default) or `"square"`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WallRing {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_wall_shape")]
    pub shape: String,
    #[serde(default = "default_wall_radius")]
    pub radius: f32,
}

fn default_wall_shape() -> String {
    "round".into()
}
fn default_wall_radius() -> f32 {
    0.7
}

/// A gate piercing the wall at a cardinal `bearing` (an arterial runs centre→gate).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GateSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// `"north"`, `"southeast"`, … — where the gate sits on the wall.
    #[serde(default)]
    pub bearing: String,
}

/// A named street. `kind` (`"arterial"` / `"minor"`) + `bearing` match it onto the
/// generated graph (an arterial radiates centre→gate at its bearing).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreetSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub bearing: String,
}

/// A named district, placed at a cardinal/canvas `anchor` with optional `character`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UrbanDistrict {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub anchor: Anchor,
    #[serde(default)]
    pub character: String,
}

/// A pier extending from the waterfront at `position` (0..1 along the water edge).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PierSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub position: f32,
}

/// A named point anchored like a landmark (used for the station).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NamedPoint {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub anchor: Anchor,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    pub cols: u32,
    pub rows: u32,
}

// ── Anchors ──────────────────────────────────────────────────────────────────

/// A typed spatial relationship — how an object is positioned relative to other
/// features. The geometry engine's Layer 5 resolves these (topological sort over
/// the dependency graph).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Anchor {
    Canvas { x: f32, y: f32 },
    Cardinal { position: String },
    MouthOf { river: String },
    SourceOf { river: String },
    Confluence { river_a: String, river_b: String },
    Bearing {
        from: String,
        direction: String,
        distance: String,
        #[serde(default)]
        constraint: Option<AnchorConstraint>,
    },
    CoastNearest { from: String },
    PassBetween { range_a: String, range_b: String },
    RegionInterior { region: String },
    ShoreNearest { water: String, from: String },
    RangeSlope { range: String, facing: String },
    NaturalHarbor { near: String },
    Delta { river: String },
    // ── urban-scale variants (resolved by Layer 6b, MAP-5) ──
    AlongWaterfront { waterfront: String, position: f32 },
    PierTip { pier: String },
    AlongStreet { street: String, position: f32 },
    NearestIntersection { near: String },
    OnWall { position: f32 },
    AtGate { gate: String },
    AtStation { station: String },
    BlockFace { street: String, side: String, position: f32 },
    InDistrict { district: String },
    CityCenter,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnchorConstraint {
    Coastline,
    River,
    RidgeLine,
    RoadNearest,
    Navigable,
}

// ── Terrain ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TerrainSpec {
    #[serde(default)]
    pub dominant_elevation: String,
    #[serde(default)]
    pub mountain_ranges: Vec<MountainRange>,
    #[serde(default)]
    pub plateaus: Vec<NamedRegion>,
    #[serde(default)]
    pub rift_valleys: Vec<NamedRegion>,
    /// Erosion / irregularity strength for natural features: `0.0` = smooth/idealized
    /// (circular coasts, oval ranges), `1.0` = natural (default), `>1.0` = rugged
    /// (ragged coasts, wandering ridgelines). `None` = the 1.0 default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub erosion: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MountainRange {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub anchor: Anchor,
    /// Orientation, e.g. `"north-south"`, `"east-west"`, `"northeast"`.
    #[serde(default)]
    pub orientation: String,
    #[serde(default)]
    pub length_fraction: f32,
    /// `"low"` / `"moderate"` / `"high"` / `"extreme"`.
    #[serde(default)]
    pub height: String,
}

/// A generic anchored named region (plateau, rift valley, …).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NamedRegion {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub anchor: Anchor,
}

// ── Water ────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WaterSpec {
    #[serde(default)]
    pub seas: Vec<SeaSpec>,
    #[serde(default)]
    pub rivers: Vec<RiverSpec>,
    #[serde(default)]
    pub lakes: Vec<LakeSpec>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SeaSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Cardinal-ish hint, e.g. `"west"`, `"south"`, `"center"`.
    #[serde(default)]
    pub position: String,
    #[serde(default)]
    pub enclosed: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RiverSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub source: Anchor,
    pub mouth: Anchor,
    #[serde(default)]
    pub tributaries: Vec<String>,
    #[serde(default)]
    pub navigable: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LakeSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub anchor: Anchor,
    #[serde(default)]
    pub size: String,
    #[serde(default)]
    pub endorheic: bool,
}

// ── Regions / polities ───────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegionSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub biome: String,
    pub anchor: Anchor,
    #[serde(default)]
    pub coverage: f32,
    #[serde(default)]
    pub political: Option<PoliticalSpec>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoliticalSpec {
    pub polity_name: String,
    #[serde(default)]
    pub polity_kind: String,
    #[serde(default)]
    pub borders: Vec<BorderSpec>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BorderSpec {
    pub with_region: String,
    /// `"mountain"`, `"river"`, `"disputed"`, …
    #[serde(default)]
    pub kind: String,
}

// ── Landmarks ────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LandmarkSpec {
    pub id: String,
    pub name: String,
    pub kind: LandmarkKind,
    pub anchor: Anchor,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub artefact: Option<String>,
    // `urban: Option<UrbanSpec>` (MAP-5) preserved as opaque JSON for now so a
    // spec with urban data round-trips cleanly before the engine exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub urban: Option<serde_json::Value>,
}

/// Known landmark kinds with a string fallback (LLM output is forgiving).
/// Serialized as a plain snake_case string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LandmarkKind {
    City,
    Town,
    Village,
    Port,
    Fortress,
    Castle,
    Ruin,
    Temple,
    Oasis,
    Pass,
    Lighthouse,
    Shipwreck,
    Dungeon,
    Oracle,
    Other(String),
}

impl LandmarkKind {
    pub fn as_str(&self) -> &str {
        match self {
            LandmarkKind::City => "city",
            LandmarkKind::Town => "town",
            LandmarkKind::Village => "village",
            LandmarkKind::Port => "port",
            LandmarkKind::Fortress => "fortress",
            LandmarkKind::Castle => "castle",
            LandmarkKind::Ruin => "ruin",
            LandmarkKind::Temple => "temple",
            LandmarkKind::Oasis => "oasis",
            LandmarkKind::Pass => "pass",
            LandmarkKind::Lighthouse => "lighthouse",
            LandmarkKind::Shipwreck => "shipwreck",
            LandmarkKind::Dungeon => "dungeon",
            LandmarkKind::Oracle => "oracle",
            LandmarkKind::Other(s) => s,
        }
    }

    pub fn from_label(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "city" => LandmarkKind::City,
            "town" => LandmarkKind::Town,
            "village" => LandmarkKind::Village,
            "port" => LandmarkKind::Port,
            "fortress" => LandmarkKind::Fortress,
            "castle" => LandmarkKind::Castle,
            "ruin" => LandmarkKind::Ruin,
            "temple" => LandmarkKind::Temple,
            "oasis" => LandmarkKind::Oasis,
            "pass" => LandmarkKind::Pass,
            "lighthouse" => LandmarkKind::Lighthouse,
            "shipwreck" => LandmarkKind::Shipwreck,
            "dungeon" => LandmarkKind::Dungeon,
            "oracle" => LandmarkKind::Oracle,
            other => LandmarkKind::Other(other.to_string()),
        }
    }
}

impl Serialize for LandmarkKind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for LandmarkKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(LandmarkKind::from_label(&s))
    }
}

// ── Infrastructure ───────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct InfrastructureSpec {
    #[serde(default)]
    pub roads: Vec<RoadSpec>,
    #[serde(default)]
    pub walls: Vec<WallSpec>,
    #[serde(default)]
    pub bridges: Vec<BridgeSpec>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoadSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Landmark ids.
    pub from: String,
    pub to: String,
    /// `"road"`, `"highway"`, `"trail"`, `"sea-lane"`.
    #[serde(default)]
    pub kind: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WallSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub from: Anchor,
    pub to: Anchor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Road id + river id the bridge crosses.
    #[serde(default)]
    pub road: Option<String>,
    #[serde(default)]
    pub river: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_spec_round_trips() {
        let m = MapSpec::minimal("Test Isle", 1, 1, 0);
        let json = serde_json::to_string(&m).unwrap();
        let back: MapSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.version, SPEC_VERSION);
        assert_eq!(back.name, "Test Isle");
        assert_eq!(back.tile_grid, TileGrid { cols: 1, rows: 1 });
    }

    #[test]
    fn anchor_tagged_serialization() {
        let a = Anchor::Bearing {
            from: "iron_capital".into(),
            direction: "east".into(),
            distance: "moderate".into(),
            constraint: Some(AnchorConstraint::Coastline),
        };
        let j = serde_json::to_value(&a).unwrap();
        assert_eq!(j["kind"], "bearing");
        assert_eq!(j["constraint"], "coastline");
        let back: Anchor = serde_json::from_value(j).unwrap();
        assert_eq!(back, a);
        // MouthOf + the unit variant.
        assert_eq!(serde_json::to_value(Anchor::MouthOf { river: "r".into() }).unwrap()["kind"], "mouth_of");
        assert_eq!(serde_json::to_value(Anchor::CityCenter).unwrap()["kind"], "city_center");
    }

    #[test]
    fn landmark_kind_string_with_fallback() {
        assert_eq!(serde_json::to_value(LandmarkKind::City).unwrap(), serde_json::json!("city"));
        assert_eq!(
            serde_json::from_value::<LandmarkKind>(serde_json::json!("fortress")).unwrap(),
            LandmarkKind::Fortress
        );
        // Unknown kind → Other, round-trips.
        let k: LandmarkKind = serde_json::from_value(serde_json::json!("monastery")).unwrap();
        assert_eq!(k, LandmarkKind::Other("monastery".into()));
        assert_eq!(serde_json::to_value(&k).unwrap(), serde_json::json!("monastery"));
    }

    #[test]
    fn full_geographic_spec_parses() {
        // A representative hand-written spec exercises every top-level section.
        let src = r#"{
          "version": 2, "name": "The Iron Coast", "scale_tier": 2,
          "tile_grid": { "cols": 3, "rows": 3 },
          "terrain": { "dominant_elevation": "hilly",
            "mountain_ranges": [ { "id": "spine", "name": "The Spine",
              "anchor": { "kind": "cardinal", "position": "north" },
              "orientation": "east-west", "height": "high" } ] },
          "water": { "rivers": [ { "id": "iron", "name": "Iron River",
              "source": { "kind": "source_of", "river": "iron" },
              "mouth": { "kind": "cardinal", "position": "southwest" },
              "navigable": true } ],
            "seas": [ { "id": "west_sea", "position": "west" } ] },
          "regions": [ { "id": "heartland", "biome": "temperate_forest",
              "anchor": { "kind": "cardinal", "position": "center" } } ],
          "landmarks": [ { "id": "ironhold", "name": "Ironhold", "kind": "city",
              "anchor": { "kind": "mouth_of", "river": "iron" } },
            { "id": "watchtower", "name": "The Watch", "kind": "fortress",
              "anchor": { "kind": "natural_harbor", "near": "ironhold" } } ],
          "infrastructure": { "roads": [ { "id": "kingsroad",
              "from": "ironhold", "to": "watchtower", "kind": "highway" } ] }
        }"#;
        let m: MapSpec = serde_json::from_str(src).unwrap();
        assert_eq!(m.name, "The Iron Coast");
        assert_eq!(m.landmarks.len(), 2);
        assert_eq!(m.landmarks[0].kind, LandmarkKind::City);
        assert!(matches!(m.landmarks[0].anchor, Anchor::MouthOf { .. }));
        assert_eq!(m.water.rivers[0].id, "iron");
        // Round-trips.
        let j = serde_json::to_string(&m).unwrap();
        let _: MapSpec = serde_json::from_str(&j).unwrap();
    }
}
