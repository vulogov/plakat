//! MAP-1: prose world description → [`MapSpec`] via the LLM provider stack
//! (`prompt::complete`, the same `--enhance` providers). Robust by design: it
//! strips markdown fences, retries with a stricter prompt, and falls back to a
//! minimal spec rather than ever aborting.

use anyhow::Result;

use super::cache;
use super::spec::{MapSpec, SPEC_VERSION, TileGrid};

/// Inputs for [`parse`].
pub struct ParseOpts {
    pub provider: String,
    /// Override the built-in system prompt (`--map-system`).
    pub system_override: Option<String>,
    /// `--map-tiles` / `--map-scale`: when set, the grid is fixed (the LLM is told
    /// not to infer it) and the result is forced to match.
    pub tile_grid: Option<TileGrid>,
    pub scale_tier: Option<u8>,
    /// `--map-cache`: SHA-256 disk cache of the parsed JSON.
    pub cache: bool,
}

/// Parse a prose description into a `MapSpec`.
pub async fn parse(description: &str, opts: &ParseOpts) -> Result<MapSpec> {
    let system = opts
        .system_override
        .clone()
        .unwrap_or_else(|| build_system(opts.tile_grid, opts.scale_tier));
    let eargs = crate::prompt::EnhanceArgs::default();
    let key = cache::key(&[&opts.provider, &system, description]);

    if opts.cache {
        if let Some(hit) = cache::lookup(&key) {
            if let Ok(m) = parse_json(&hit) {
                tracing::info!(target: "plakat", "map: spec cache hit");
                return Ok(finalize(m, opts, description));
            }
        }
    }

    // Stage 1: parse the model's first response.
    if let Ok(text) = crate::prompt::complete(&opts.provider, &system, description, &eargs).await {
        if let Ok(m) = parse_json(&text) {
            if opts.cache {
                cache::store(&key, &extract_json(&text));
            }
            return Ok(finalize(m, opts, description));
        }
        // Stage 2: retry with a stricter "JSON only" instruction.
        let strict = format!(
            "{system}\n\nCRITICAL: respond with ONLY the JSON object — no markdown fences, \
             no commentary before or after."
        );
        if let Ok(t2) = crate::prompt::complete(&opts.provider, &strict, description, &eargs).await {
            if let Ok(m) = parse_json(&t2) {
                if opts.cache {
                    cache::store(&key, &extract_json(&t2));
                }
                return Ok(finalize(m, opts, description));
            }
        }
    }

    // Stage 3: never abort — a minimal spec (cardinal anchors, no rivers/infra).
    tracing::warn!(target: "plakat", "map: LLM parse failed; falling back to a minimal spec");
    Ok(finalize(minimal_for(description, opts), opts, description))
}

/// Apply CLI overrides (grid / tier) and normalize the version.
fn finalize(mut m: MapSpec, opts: &ParseOpts, description: &str) -> MapSpec {
    m.version = SPEC_VERSION;
    if m.name.trim().is_empty() {
        m.name = auto_name(description);
    }
    if let Some(g) = opts.tile_grid {
        m.tile_grid = g;
    }
    if let Some(t) = opts.scale_tier {
        m.scale_tier = t;
    }
    m
}

fn minimal_for(description: &str, opts: &ParseOpts) -> MapSpec {
    let g = opts.tile_grid.unwrap_or(TileGrid { cols: 1, rows: 1 });
    MapSpec::minimal(auto_name(description), g.cols, g.rows, opts.scale_tier.unwrap_or(0))
}

fn auto_name(description: &str) -> String {
    let words: Vec<&str> = description.split_whitespace().take(5).collect();
    if words.is_empty() {
        "Unnamed Map".to_string()
    } else {
        words.join(" ")
    }
}

/// Pull a JSON object out of an LLM reply: prefer a fenced block, else the span
/// from the first `{` to the last `}`.
pub fn extract_json(text: &str) -> String {
    let t = text.trim();
    if let Some(start) = t.find("```") {
        let after = &t[start + 3..];
        // skip an optional language tag on the fence line.
        let after = after.strip_prefix("json").unwrap_or(after);
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    match (t.find('{'), t.rfind('}')) {
        (Some(a), Some(b)) if b > a => t[a..=b].to_string(),
        _ => t.to_string(),
    }
}

fn parse_json(text: &str) -> Result<MapSpec> {
    Ok(serde_json::from_str(&extract_json(text))?)
}

/// The built-in geographic-parser system prompt. Describes the schema + anchor
/// types and the JSON-only contract.
fn build_system(grid: Option<TileGrid>, tier: Option<u8>) -> String {
    let scale_line = match (grid, tier) {
        (Some(g), _) => format!(
            "The tile grid is FIXED at {cols}x{rows} — set tile_grid to exactly that and do not infer it.",
            cols = g.cols,
            rows = g.rows
        ),
        (None, Some(t)) => format!("Use scale_tier {t}; choose a tile_grid that suits it."),
        (None, None) => "Infer scale_tier (0=city … 5=hemisphere; 10–12 urban) and a suitable tile_grid (cols×rows, 1–8 each) from the description.".to_string(),
    };
    format!(
        "{SCHEMA_PREAMBLE}\n\n{scale_line}\n\n{RULES}"
    )
}

const SCHEMA_PREAMBLE: &str = r#"You are a fantasy cartographer. Convert the user's world description into a
MapSpec v2 JSON object. Output ONLY the JSON — no markdown, no commentary.

Top-level shape:
{
  "version": 2,
  "name": "<map title>",
  "scale_tier": <0-5 geographic | 10-12 urban>,
  "tile_grid": { "cols": <1-8>, "rows": <1-8> },
  "climate": "<optional>", "era": "<optional>", "language": "<optional BCP-47>",
  "terrain": { "dominant_elevation": "flat|hilly|mountainous|...",
               "mountain_ranges": [ { "id","name","anchor","orientation","height" } ] },
  "water": { "seas": [ { "id","name","position","enclosed" } ],
             "rivers": [ { "id","name","source":<anchor>,"mouth":<anchor>,"navigable" } ],
             "lakes": [ { "id","name","anchor","size","endorheic" } ] },
  "regions": [ { "id","name","biome","anchor","coverage" } ],
  "landmarks": [ { "id","name","kind":"city|town|port|fortress|castle|ruin|temple|...",
                   "anchor":<anchor> } ],
  "infrastructure": { "roads": [ { "id","from":<landmark id>,"to":<landmark id>,"kind" } ],
                      "walls": [], "bridges": [] }
}

For a CITY/TOWN map (scale_tier 10-12), also add an "urban" block:
  "urban": {
    "layout": "radial|grid|organic",   // radial=medieval walled, grid=planned, organic=hill town
    "wall": { "name","shape":"round|square","radius":<0.3-0.95> },   // omit for an open town
    "gates": [ { "id","name","bearing":"north|east|south|west|..." } ],
    "streets": [ { "id","name","kind":"arterial|minor","bearing":"<cardinal>" } ],
    "districts": [ { "id","name","anchor":<anchor>,"character" } ],
    "waterfront": "<edge, e.g. south>",   // omit if inland
    "piers": [ { "id","name","position":<0-1 along the waterfront> } ]
  }
Landmarks may use urban anchors: {"kind":"city_center"}, {"kind":"at_gate","gate":"<id>"},
  {"kind":"in_district","district":"<id>"}, {"kind":"pier_tip","pier":"<id>"},
  {"kind":"along_street","street":"<id>","position":<0-1>}, {"kind":"on_wall","position":<0-1>}.

POSITIONS ARE ANCHORS, never pixel coordinates. An anchor is one of:
  { "kind":"cardinal", "position":"north|south|east|west|center|northeast|..." }
  { "kind":"mouth_of", "river":"<river id>" }      { "kind":"source_of","river":"<id>" }
  { "kind":"confluence","river_a":"<id>","river_b":"<id>" }
  { "kind":"bearing","from":"<id>","direction":"east","distance":"adjacent|near|moderate|far|distant",
    "constraint":"coastline|river|ridge_line|road_nearest|navigable" }   (constraint optional)
  { "kind":"coast_nearest","from":"<id>" }   { "kind":"natural_harbor","near":"<id>" }
  { "kind":"pass_between","range_a":"<id>","range_b":"<id>" }
  { "kind":"shore_nearest","water":"<id>","from":"<id>" }   { "kind":"delta","river":"<id>" }
  { "kind":"region_interior","region":"<id>" }   { "kind":"range_slope","range":"<id>","facing":"east" }"#;

const RULES: &str = r#"Rules:
- Give every feature a unique lowercase snake_case "id". Reference features by id.
- A river's source is usually high terrain; its mouth is a coast/sea/lake.
- Place cities plausibly: ports on coasts/river-mouths, fortresses on passes/headlands.
- Prefer relational anchors (mouth_of, bearing, natural_harbor) over raw cardinals.
- For a town map, pick a layout that fits: medieval/walled → radial, planned/colonial/
  plains → grid, hill/mountain/old → organic. Omit "layout" to auto-infer from context.
- Keep it consistent: don't reference an id you didn't define.
- Output strictly valid JSON for the schema above and nothing else."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_handles_fences_and_prose() {
        assert_eq!(extract_json("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(extract_json("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(extract_json("Here you go:\n{\"a\":1}\nDone."), "{\"a\":1}");
        assert_eq!(extract_json("{\"a\":1}"), "{\"a\":1}");
    }

    #[test]
    fn parses_a_fenced_minimal_spec() {
        let reply = "Sure!\n```json\n{ \"version\":2, \"name\":\"Isle\", \"scale_tier\":0, \
            \"tile_grid\":{\"cols\":1,\"rows\":1}, \"terrain\":{\"dominant_elevation\":\"hilly\"}, \
            \"water\":{}, \"infrastructure\":{} }\n```\n";
        let m = parse_json(reply).unwrap();
        assert_eq!(m.name, "Isle");
    }

    #[test]
    fn build_system_reflects_grid_override() {
        let sys = build_system(Some(TileGrid { cols: 4, rows: 2 }), None);
        assert!(sys.contains("FIXED at 4x2"));
        assert!(build_system(None, None).contains("Infer scale_tier"));
    }

    #[test]
    fn auto_name_from_description() {
        assert_eq!(auto_name("a tropical island kingdom with a volcano"), "a tropical island kingdom with");
    }
}
