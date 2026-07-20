//! Natural-language command pipeline for the command pane (`:`). A command is an optional
//! **selector** (`find`/`take` a pattern) followed by a **pipeline** of actions run in order —
//! e.g. `find rating>=4 then upscale then export to ~/best 2000`.
//!
//! Two front-ends produce the same [`CommandPlan`]:
//! - a **deterministic** keyword parser ([`parse_deterministic`]) — no network, handles the common
//!   phrasings offline;
//! - the existing **LLM enhancement pipeline** ([`crate::prompt::complete`], any configured provider)
//!   for everything else, grounded with the album's HJSON.
//!
//! The LLM only ever emits JSON from this closed vocabulary — it routes intent, it never executes.

use anyhow::{Context, Result};
use serde::Deserialize;

/// One step in the pipeline. `serde(tag = "action")` so the LLM emits `{"action":"upscale"}` etc.,
/// and the same enum is produced by the deterministic parser.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    Rate { stars: u8 },
    Flag,
    Reject,
    Color { label: String },
    Tag { tags: Vec<String> },
    Autotag,
    Describe,
    Upscale,
    Img2img { prompt: String },
    Relight { prompt: String },
    /// AI txt2img — generate a new image from a prompt into the album.
    Generate { prompt: String },
    /// AI portrait from a prompt (the cursor image is used as the identity face when it's a photo).
    Portrait { prompt: String },
    /// AI multi-person scene from a prompt (the selected images become the people).
    Multiperson { prompt: String },
    /// A T1 pixel edit: `rotate_cw`/`rotate_ccw`/`rotate_180`/`flip_h`/`flip_v`/`grayscale`/`crop_square`.
    Edit { op: String },
    /// Export (copy) album images to a destination directory. Create-only — never reads or
    /// overwrites anything outside; a `dir` path is the ONLY outward path the vocabulary allows.
    Export { dir: String, #[serde(default)] max_px: Option<u32> },
    /// Rename in-album with a filename `pattern` (no path — stays inside the album).
    Rename { pattern: String },
    Sort { by: String },
    Dedup,
    Stack,
    SmartAlbum { name: String },
    /// Strip EXIF/XMP/IPTC/GPS metadata from the target files in place (album-scoped; no external
    /// read, no re-import). Not undoable.
    StripMeta,
    /// Remove only the GPS location from the target files, keeping the rest of the EXIF. Not undoable.
    RedactGps,
    /// Copy the highest-res version of the targets into a new nested working sub-album.
    Take,
    /// Copy the selected finished image(s) from a workbench sub-album up to its parent album.
    PutBack,
    /// Duplicate the target image(s) within their album.
    Duplicate,
    /// Stitch the selected images into a panorama (`mode` 0 horizontal / 1 vertical / 2 grid).
    Panorama { mode: i32 },
    /// Build a collage from the selected images (or the whole album).
    Collage,
    Mosaic,
    Hdr,
    FocusStack,
    QualityCull,
    Flatten,
    Trash,
    RestoreTrash,
    EmptyTrash,
    OpenTrash,
    Map,
    Geocode,
    /// Open the shared-volume conflict-review pane.
    Conflicts,
    /// List the live `plakat photos` instances sharing this library (presence heartbeats).
    Who,
    /// Rebuild the derived library index from scratch.
    Reindex,
    /// Pre-compute + persist CLIP embeddings for the whole library (fast visual search afterwards).
    Embed,
    /// Show aggregate library statistics (computed from the derived index).
    Stats,
    /// Convert the targets to `fmt` (jpg/png/webp), optionally capping the longest side. Writes NEW
    /// files inside the album (create-only; the source is untouched).
    Convert { fmt: String, #[serde(default)] max_px: Option<u32> },
}

/// A parsed command: an optional selector (`all` / `selected` / a filter expression) + the pipeline.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct CommandPlan {
    #[serde(default)]
    pub select: Option<String>,
    pub actions: Vec<Action>,
}

impl CommandPlan {
    /// A one-line human summary for the confirmation prompt.
    pub fn summary(&self) -> String {
        let sel = match self.select.as_deref() {
            Some(s) => format!("[{s}] "),
            None => String::new(),
        };
        let acts: Vec<String> = self.actions.iter().map(action_label).collect();
        format!("{sel}{}", acts.join(" → "))
    }
    /// Whether any action loads a model / hits the network per image (worth confirming a batch).
    pub fn is_heavy(&self) -> bool {
        self.actions.iter().any(|a| {
            matches!(
                a,
                Action::Upscale
                    | Action::Img2img { .. }
                    | Action::Relight { .. }
                    | Action::Autotag
                    | Action::Describe
            )
        })
    }
}

fn action_label(a: &Action) -> String {
    match a {
        Action::Rate { stars } => format!("rate {stars}"),
        Action::Flag => "flag".into(),
        Action::Reject => "reject".into(),
        Action::Color { label } => format!("colour {label}"),
        Action::Tag { tags } => format!("tag {}", tags.join(",")),
        Action::Autotag => "autotag".into(),
        Action::Describe => "describe".into(),
        Action::Upscale => "upscale".into(),
        Action::Img2img { prompt } => format!("img2img '{prompt}'"),
        Action::Generate { prompt } => format!("generate '{prompt}'"),
        Action::Portrait { prompt } => format!("portrait '{prompt}'"),
        Action::Multiperson { prompt } => format!("multiperson '{prompt}'"),
        Action::Relight { prompt } => format!("relight '{prompt}'"),
        Action::Edit { op } => format!("edit {op}"),
        Action::Export { dir, max_px } => match max_px {
            Some(p) => format!("export→{dir} ≤{p}px"),
            None => format!("export→{dir}"),
        },
        Action::Rename { pattern } => format!("rename {pattern}"),
        Action::Sort { by } => format!("sort {by}"),
        Action::Dedup => "dedup".into(),
        Action::Stack => "stack".into(),
        Action::SmartAlbum { name } => format!("smart-album {name}"),
        Action::StripMeta => "strip metadata".into(),
        Action::RedactGps => "redact GPS".into(),
        Action::Take => "take for processing".into(),
        Action::PutBack => "put back to parent".into(),
        Action::Duplicate => "duplicate".into(),
        Action::Panorama { mode } => format!("panorama {}", ["horizontal", "vertical", "grid", "horizontal aligned", "vertical aligned", "homography"].get(*mode as usize).unwrap_or(&"horizontal")),
        Action::Collage => "collage".into(),
        Action::Mosaic => "mosaic".into(),
        Action::Hdr => "hdr".into(),
        Action::FocusStack => "focus stack".into(),
        Action::QualityCull => "quality cull".into(),
        Action::Flatten => "flatten".into(),
        Action::Trash => "trash".into(),
        Action::RestoreTrash => "restore trash".into(),
        Action::EmptyTrash => "empty trash".into(),
        Action::OpenTrash => "open trash".into(),
        Action::Map => "map".into(),
        Action::Geocode => "geocode".into(),
        Action::Conflicts => "conflicts".into(),
        Action::Who => "who".into(),
        Action::Reindex => "reindex".into(),
        Action::Embed => "embed".into(),
        Action::Stats => "stats".into(),
        Action::Convert { fmt, max_px } => match max_px {
            Some(p) => format!("convert→{fmt} ≤{p}px"),
            None => format!("convert→{fmt}"),
        },
    }
}

/// Deterministic keyword parser for the common phrasings. Returns `None` (→ fall back to the LLM) if
/// any stage isn't recognised. Stages are split on `then` / `|` / `;`.
pub fn parse_deterministic(input: &str) -> Option<CommandPlan> {
    let lower = input.trim().to_lowercase();
    if lower.is_empty() {
        return None;
    }
    let stages: Vec<String> = lower
        .split(|c| c == '|' || c == ';')
        .flat_map(|s| s.split(" then "))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut plan = CommandPlan::default();
    for (i, stage) in stages.iter().enumerate() {
        // The first stage may be a selector.
        if i == 0 {
            if let Some(sel) = as_selector(stage) {
                plan.select = Some(sel);
                continue;
            }
        }
        plan.actions.push(parse_action(stage)?);
    }
    if plan.actions.is_empty() {
        return None;
    }
    Some(plan)
}

/// A selector stage → the selection string (`all` / `selected` / a filter expression), or `None` if
/// this stage isn't a selector.
fn as_selector(stage: &str) -> Option<String> {
    for kw in ["find ", "take ", "select ", "where ", "filter "] {
        if let Some(rest) = stage.strip_prefix(kw) {
            return Some(rest.trim().to_string());
        }
    }
    match stage {
        "all" | "everything" | "all photos" | "all images" => Some("all".into()),
        "selected" | "selection" | "the selection" => Some("selected".into()),
        _ => None,
    }
}

fn parse_action(stage: &str) -> Option<Action> {
    let s = stage.trim();
    // Value-carrying verbs first (prefix match), then bare verbs. The only outward path is `export`
    // (create-only writes); nothing here reads an external path or runs a command.
    if let Some(r) = strip_any(s, &["export to ", "export "]) {
        let (dir, max_px) = split_trailing_num(&r);
        return Some(Action::Export { dir, max_px });
    }
    if let Some(r) = strip_any(s, &["rename to ", "rename "]) {
        return Some(Action::Rename { pattern: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["tag as ", "tag with ", "tag "]) {
        let tags: Vec<String> = r.split([',', ' ']).map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
        return (!tags.is_empty()).then_some(Action::Tag { tags });
    }
    if let Some(r) = strip_any(s, &["colour ", "color "]) {
        return Some(Action::Color { label: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["rate "]) {
        let n: u8 = r.trim().split_whitespace().next()?.parse().ok()?;
        return Some(Action::Rate { stars: n.min(5) });
    }
    if let Some(r) = strip_any(s, &["sort by ", "sort "]) {
        return Some(Action::Sort { by: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["img2img ", "transform to ", "turn into "]) {
        return Some(Action::Img2img { prompt: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["generate ", "txt2img ", "dream "]) {
        return Some(Action::Generate { prompt: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["portrait of ", "portrait "]) {
        return Some(Action::Portrait { prompt: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["multiperson ", "scene of ", "scene "]) {
        return Some(Action::Multiperson { prompt: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["relight "]) {
        return Some(Action::Relight { prompt: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["smart album ", "save as smart album ", "save smart album "]) {
        return Some(Action::SmartAlbum { name: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["straighten by ", "straighten "]) {
        let tok = r.trim().trim_end_matches("degrees").trim_end_matches('°').trim();
        let deg: f32 = tok.split_whitespace().next()?.parse().ok()?;
        return Some(Action::Edit { op: format!("straighten:{deg}") });
    }
    if let Some(r) = strip_any(s, &["levels "]) {
        // `levels BLACK WHITE GAMMA` (e.g. levels 16 235 1.1).
        let n: Vec<&str> = r.split([',', ' ']).filter(|t| !t.is_empty()).collect();
        if let [b, w, g] = n.as_slice() {
            return Some(Action::Edit { op: format!("levels:{b},{w},{g}") });
        }
        return None;
    }
    if let Some(r) = strip_any(s, &["hue rotate ", "hue "]) {
        let tok = r.trim().trim_end_matches("degrees").trim_end_matches('°').trim();
        let deg: i32 = tok.split_whitespace().next()?.parse().ok()?;
        return Some(Action::Edit { op: format!("hue:{deg}") });
    }
    if let Some(r) = strip_any(s, &["convert to ", "convert "]) {
        let mut it = r.split_whitespace();
        let fmt = it.next()?.to_string();
        let max_px = it.next().and_then(|t| t.to_ascii_lowercase().trim_end_matches("px").parse::<u32>().ok());
        return Some(Action::Convert { fmt, max_px });
    }
    if let Some(op) = edit_op_word(s) {
        return Some(Action::Edit { op });
    }
    // Bare verbs.
    match s {
        "flag" => Some(Action::Flag),
        "reject" => Some(Action::Reject),
        "autotag" | "auto tag" | "tag" => Some(Action::Autotag),
        "describe" | "caption" => Some(Action::Describe),
        "upscale" | "enhance" | "upscale x4" => Some(Action::Upscale),
        "dedup" | "deduplicate" | "find duplicates" | "duplicates" => Some(Action::Dedup),
        "stack" => Some(Action::Stack),
        "strip metadata" | "strip exif" | "remove exif" | "remove metadata" | "scrub metadata" => {
            Some(Action::StripMeta)
        }
        "redact gps" | "remove gps" | "strip gps" | "remove location" | "redact location" => {
            Some(Action::RedactGps)
        }
        // Note: bare "take" is a *selector* keyword, so the take-photo action needs the fuller phrase.
        "take photo" | "take for processing" | "work copy" | "take copy" => Some(Action::Take),
        "put back" | "promote" | "copy to parent" | "put back to parent" => Some(Action::PutBack),
        "duplicate" | "make a copy" | "copy in album" | "duplicate image" => Some(Action::Duplicate),
        "panorama" | "panorama horizontal" | "stitch" | "stitch horizontal" => Some(Action::Panorama { mode: 0 }),
        "panorama vertical" | "stitch vertical" => Some(Action::Panorama { mode: 1 }),
        "panorama grid" | "stitch grid" | "combined panorama" => Some(Action::Panorama { mode: 2 }),
        "panorama aligned" | "align panorama" | "stitch aligned" | "panorama horizontal aligned" => Some(Action::Panorama { mode: 3 }),
        "panorama vertical aligned" => Some(Action::Panorama { mode: 4 }),
        "panorama homography" | "stitch homography" | "true panorama" | "perspective panorama" => Some(Action::Panorama { mode: 5 }),
        "collage" | "make collage" | "create collage" => Some(Action::Collage),
        "mosaic" | "scrapbook" | "make mosaic" | "create mosaic" => Some(Action::Mosaic),
        "hdr" | "exposure blend" | "exposure fusion" | "merge exposures" => Some(Action::Hdr),
        "focus stack" | "focus stacking" | "stack focus" => Some(Action::FocusStack),
        "quality cull" | "cull blurry" | "cull soft" | "flag soft" | "cull bad" => Some(Action::QualityCull),
        "flatten" | "recursive" | "show all" | "flatten browse" => Some(Action::Flatten),
        "trash" | "move to trash" | "soft delete" => Some(Action::Trash),
        "restore" | "restore trash" | "restore all" => Some(Action::RestoreTrash),
        "empty trash" | "purge trash" => Some(Action::EmptyTrash),
        "open trash" | "browse trash" | "show trash" => Some(Action::OpenTrash),
        "map" | "geo" | "atlas" | "geo map" | "show map" => Some(Action::Map),
        "geocode" | "reverse geocode" | "place tags" | "geotag places" => Some(Action::Geocode),
        "conflicts" | "review conflicts" | "show conflicts" => Some(Action::Conflicts),
        "who" | "whoami" | "instances" | "presence" | "peers" => Some(Action::Who),
        "reindex" | "rebuild index" | "rebuild-index" => Some(Action::Reindex),
        "embed" | "embed library" | "index visual" | "precompute clip" => Some(Action::Embed),
        "stats" | "statistics" | "library stats" | "summary" => Some(Action::Stats),
        _ => None,
    }
}

/// Map an edit phrasing to a canonical [`crate::photos::edit::EditOp`] tag.
fn edit_op_word(s: &str) -> Option<String> {
    Some(match s {
        "rotate" | "rotate right" | "rotate cw" | "rotate clockwise" => "rotate_cw",
        "rotate left" | "rotate ccw" | "rotate counter-clockwise" => "rotate_ccw",
        "rotate 180" | "flip 180" => "rotate_180",
        "flip" | "flip horizontal" | "flip h" | "mirror" => "flip_h",
        "flip vertical" | "flip v" => "flip_v",
        "grayscale" | "greyscale" | "black and white" | "b&w" => "grayscale",
        "crop" | "crop square" | "square crop" | "crop 1:1" => "crop_square",
        "auto enhance" | "auto-enhance" | "auto fix" | "enhance auto" => "auto_enhance",
        "vignette" | "darken edges" => "vignette",
        "vignette light" | "lighten edges" => "vignette_light",
        "dehaze" | "defog" | "remove haze" => "dehaze",
        "keystone" | "fix verticals" | "correct perspective" => "keystone",
        "border" | "frame" | "add border" => "border",
        "circle crop" | "circle" | "round crop" => "circle",
        "keystone horizontal" => "keystone_h",
        "invert" | "negative" => "invert",
        "sepia" => "sepia",
        "duotone" => "duotone",
        "posterize" => "posterize",
        "solarize" => "solarize",
        "threshold" | "black and white threshold" => "threshold",
        "oil paint" | "oilify" | "oil painting" => "oil_paint",
        "pencil sketch" | "sketch" | "pencil" => "pencil_sketch",
        "cartoon" | "comic" | "cartoonize" => "cartoon",
        "watercolor" | "watercolour" => "watercolor",
        "european ink" | "pen and ink" | "ink" => "european_ink",
        "japanese ink" | "sumi-e" | "sumie" => "japanese_ink",
        "chinese ink" | "ink wash" | "shan shui" => "chinese_ink",
        "russian icon" | "russian religious" | "tempera" | "icon" => "russian_icon",
        "cross-hatch" | "crosshatch" | "hatch" => "crosshatch",
        "crystallize" | "voronoi" | "low poly" | "low-poly" | "stained glass" => "crystallize",
        "bilateral" | "bilateral denoise" | "edge denoise" | "smart denoise" => "bilateral",
        "gray point wb" | "grey point wb" | "eyedropper" | "neutral point" => "gray_point_wb",
        "tilt shift" | "tilt-shift" | "miniature" => "tilt_shift",
        "motion blur" | "motion" => "motion_blur",
        "zoom blur" | "zoom" => "zoom_blur",
        "spin blur" | "spin" => "spin_blur",
        "bw mixer" | "black and white mixer" | "channel mixer" | "mono mixer" => "bw_mixer",
        "film negative" | "negative to positive" | "c41" => "film_negative",
        "lens distortion" | "distortion" | "defish" => "lens_distort",
        "chromatic aberration" | "defringe" | "remove fringing" => "chromatic_aberration",
        "enhance sky" | "better sky" | "fix sky" | "sky" => "enhance_sky",
        "auto white balance" | "auto wb" | "gray world" | "neutralize" | "neutralise" => "auto_wb",
        "gradient map" | "gradient-map" | "gradientmap" => "gradient_map",
        "kelvin" | "white balance" | "colour temperature" | "color temperature" => "kelvin",
        "emboss" => "emboss",
        "blur" | "blur it" | "soft focus" | "gaussian blur" => "blur",
        "bloom" | "glow" | "orton" => "bloom",
        "charcoal" => "charcoal",
        "halftone" | "newsprint" => "halftone",
        "thermal" | "false color" | "false colour" => "thermal",
        "infrared" => "infrared",
        "night vision" | "nightvision" => "night_vision",
        "pixelate" | "mosaic" | "pixelize" => "pixelate",
        "split tone" | "split-tone" | "warm highlights" => "split_tone",
        "split tone cool" | "cool highlights" => "split_tone_cool",
        "film grain" | "add grain" | "grain" => "grain",
        "despeckle" | "median" | "remove speckle" => "despeckle",
        "pop reds" | "boost reds" => "pop_reds",
        "mute reds" => "mute_reds",
        "pop greens" | "boost greens" => "pop_greens",
        "mute greens" => "mute_greens",
        "pop blues" | "boost blues" => "pop_blues",
        "mute blues" => "mute_blues",
        // Tonal / colour adjustments (canonical directional tags → edit::EditOp::from_tag).
        "sharpen" | "sharpen it" | "sharpen image" => "sharpen",
        "soften" => "soften",
        "denoise" | "reduce noise" | "noise reduction" | "remove noise" => "denoise",
        "clarity" | "definition" | "add clarity" | "add definition" => "definition",
        "brighter" | "brighten" | "brighten it" => "brighter",
        "darker" | "darken" | "darken it" => "darker",
        "more contrast" | "increase contrast" | "punchier" => "more_contrast",
        "less contrast" | "reduce contrast" | "flatten contrast" => "less_contrast",
        "more exposure" | "increase exposure" | "brighten exposure" | "overexpose" => "exposure_up",
        "less exposure" | "decrease exposure" | "underexpose" => "exposure_down",
        "brilliance" | "add brilliance" | "richer" => "brilliance",
        "recover highlights" | "reduce highlights" | "less highlights" => "highlights_down",
        "boost highlights" | "more highlights" | "brighter highlights" => "highlights_up",
        "lift shadows" | "open shadows" | "brighten shadows" | "more shadows" => "shadows_up",
        "deepen shadows" | "darken shadows" | "less shadows" => "shadows_down",
        "lighten midtones" | "more midrange" | "midrange up" => "midrange_up",
        "darken midtones" | "less midrange" | "midrange down" => "midrange_down",
        "crush blacks" | "deepen blacks" | "black point up" => "blackpoint_up",
        "lift blacks" | "raise blacks" | "black point down" => "blackpoint_down",
        "saturate" | "more saturation" | "increase saturation" | "more color" | "more colour" => "saturate",
        "desaturate" | "less saturation" | "reduce saturation" | "muted" => "desaturate",
        "vibrance" | "more vibrance" | "boost vibrance" | "vibrant" => "vibrant",
        "less vibrance" | "reduce vibrance" => "vibrance_down",
        "warmer" | "warm up" | "more warmth" => "warmer",
        "cooler" | "cool down" | "less warmth" => "cooler",
        "tint magenta" | "more magenta" => "tint_magenta",
        "tint green" | "more green" => "tint_green",
        _ => return None,
    }
    .to_string())
}

fn strip_any(s: &str, prefixes: &[&str]) -> Option<String> {
    prefixes.iter().find_map(|p| s.strip_prefix(p).map(|r| r.to_string()))
}

/// Split `"~/x 1600"` → `("~/x", Some(1600))`; a value with no trailing integer → `(value, None)`.
fn split_trailing_num(s: &str) -> (String, Option<u32>) {
    let s = s.trim();
    if let Some((head, last)) = s.rsplit_once(char::is_whitespace) {
        if let Ok(n) = last.trim().parse::<u32>() {
            return (head.trim().to_string(), Some(n));
        }
    }
    (s.to_string(), None)
}

/// Build the LLM plan from `input`, grounded with `grounding` (the album HJSON + context). Reuses the
/// existing provider pipeline; the model returns only JSON from the closed vocabulary.
pub async fn plan_llm(provider: &str, input: &str, grounding: &str) -> Result<CommandPlan> {
    let user = format!(
        "Album context (HJSON):\n{grounding}\n\nUser command: {input}\n\nReturn ONLY the JSON plan."
    );
    let text = crate::prompt::complete(provider, SYSTEM, &user, &crate::prompt::EnhanceArgs::default())
        .await
        .context("LLM planner")?;
    let json = extract_json(&text).ok_or_else(|| anyhow::anyhow!("no JSON in planner reply: {text}"))?;
    serde_json::from_str(&json).with_context(|| format!("parsing plan JSON: {json}"))
}

/// Pull the first `{...}` JSON object out of an LLM reply (tolerates ```json fences / prose).
fn extract_json(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end >= start).then(|| text[start..=end].to_string())
}

const SYSTEM: &str = "\
You translate a photo-manager command into a JSON plan. You act ONLY on the images already in the
current album — their curation metadata and pixels. The single outward operation is `export`, which
COPIES album images to a destination directory (create-only). You must NEVER read, open, import, or
reference any file or directory OUTSIDE the album, and you must NEVER run shell or arbitrary commands.
Respond with ONLY JSON, no prose, no fences.

Schema: {\"select\": <string|null>, \"actions\": [<action>, ...]}
- select: \"all\" (whole album view), \"selected\" (current selection), or a filter expression using the
  grammar: rating>=N / rating>N / rating=N / unrated / flag / -flag / rejected / -rejected / ai / scored /
  tag:WORD / -tag:WORD / free-text. null keeps the current selection. (This is a query, not a path.)
- actions run in order. Each is one of:
  {\"action\":\"rate\",\"stars\":0-5} {\"action\":\"flag\"} {\"action\":\"reject\"}
  {\"action\":\"color\",\"label\":\"red|yellow|green|blue|purple\"}
  {\"action\":\"tag\",\"tags\":[\"...\"]} {\"action\":\"autotag\"} {\"action\":\"describe\"}
  {\"action\":\"upscale\"} {\"action\":\"img2img\",\"prompt\":\"...\"} {\"action\":\"relight\",\"prompt\":\"...\"}
  {\"action\":\"edit\",\"op\":\"<one of the edit tags>\"}  where the tag is one of:
    geometry: rotate_cw rotate_ccw rotate_180 flip_h flip_v grayscale crop_square auto_enhance
      straighten:<degrees>  (e.g. straighten:3 or straighten:-2.5)
    light: brighter darker exposure_up exposure_down brilliance more_contrast less_contrast
    tone bands: highlights_up highlights_down midrange_up midrange_down shadows_up shadows_down
      blackpoint_up blackpoint_down
    colour: saturate desaturate vibrant vibrance_down warmer cooler tint_magenta tint_green
      dehaze split_tone split_tone_cool  hue:<deg>  selcolor:<hue>,<sat>
      pop_reds mute_reds pop_greens mute_greens pop_blues mute_blues
    detail: sharpen soften definition denoise vignette vignette_light grain despeckle
    levels:<black>,<white>,<gamma>  (0..255, 0..255, float — e.g. levels:16,235,1.1)
  {\"action\":\"strip_meta\"}  (remove all EXIF/GPS metadata from the album files, in place)
  {\"action\":\"redact_gps\"}  (remove only the GPS location, keep the rest of the EXIF)
  {\"action\":\"take\"}  (copy the highest-res targets into a new nested working sub-album)
  {\"action\":\"put_back\"}  (copy the finished targets from a sub-album up to its parent album)
  {\"action\":\"duplicate\"}  (duplicate the target image(s) within their album)
  {\"action\":\"panorama\",\"mode\":0-2}  (stitch selected images: 0 horizontal, 1 vertical, 2 grid)
  {\"action\":\"collage\"}  (grid collage from the selected images / whole album)
  {\"action\":\"convert\",\"fmt\":\"jpg|png|webp\",\"max_px\":<int|null>}  (write NEW converted files in-album)
  {\"action\":\"export\",\"dir\":\"<destination directory>\",\"max_px\":<int|null>}  (copy OUT; write-only)
  {\"action\":\"rename\",\"pattern\":\"trip_###\"}  (a BARE in-album filename pattern, never a path)
  {\"action\":\"sort\",\"by\":\"name-asc|name-desc|date-desc|date-asc|rating-desc|score-desc\"}
  {\"action\":\"dedup\"} {\"action\":\"stack\"} {\"action\":\"smart_album\",\"name\":\"...\"}
Only use actions/fields from this schema. If the request would read outside the album or run a command,
return {\"select\":null,\"actions\":[]}.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_selector_and_pipeline() {
        let p = parse_deterministic("find rating>=4 then upscale then export to ~/best 2000").unwrap();
        assert_eq!(p.select.as_deref(), Some("rating>=4"));
        assert_eq!(
            p.actions,
            vec![Action::Upscale, Action::Export { dir: "~/best".into(), max_px: Some(2000) }]
        );
    }

    #[test]
    fn adjustment_verbs_pipeline_into_edit_actions() {
        // Free-form verbs map to canonical edit tags; each resolves to a real op via from_tag.
        let p = parse_deterministic("find rating>=4 then sharpen then warmer then export to ~/out")
            .unwrap();
        assert_eq!(
            p.actions,
            vec![
                Action::Edit { op: "sharpen".into() },
                Action::Edit { op: "warmer".into() },
                Action::Export { dir: "~/out".into(), max_px: None },
            ]
        );
        for verb in ["denoise", "lift shadows", "desaturate", "add clarity", "recover highlights"] {
            let a = &parse_deterministic(verb).unwrap().actions[0];
            match a {
                Action::Edit { op } => {
                    assert!(crate::photos::edit::EditOp::from_tag(op).is_some(), "unmapped tag {op}")
                }
                _ => panic!("{verb} should be an edit"),
            }
        }
    }

    #[test]
    fn management_verbs_parse() {
        assert_eq!(
            parse_deterministic("all then auto enhance then straighten 3").unwrap().actions,
            vec![
                Action::Edit { op: "auto_enhance".into() },
                Action::Edit { op: "straighten:3".into() },
            ]
        );
        assert_eq!(parse_deterministic("strip exif").unwrap().actions, vec![Action::StripMeta]);
        assert_eq!(
            parse_deterministic("convert to jpg 2048").unwrap().actions,
            vec![Action::Convert { fmt: "jpg".into(), max_px: Some(2048) }]
        );
        // straighten:3 resolves to a real op.
        assert!(crate::photos::edit::EditOp::from_tag("straighten:3").is_some());
        // Vignette + levels.
        assert_eq!(parse_deterministic("vignette").unwrap().actions, vec![Action::Edit { op: "vignette".into() }]);
        assert_eq!(
            parse_deterministic("levels 16 235 1.1").unwrap().actions,
            vec![Action::Edit { op: "levels:16,235,1.1".into() }]
        );
        assert!(crate::photos::edit::EditOp::from_tag("levels:16,235,1.1").is_some());
        // Grading verbs + GPS-only redact (distinct from full strip).
        assert_eq!(parse_deterministic("dehaze").unwrap().actions, vec![Action::Edit { op: "dehaze".into() }]);
        assert_eq!(parse_deterministic("pop blues").unwrap().actions, vec![Action::Edit { op: "pop_blues".into() }]);
        assert_eq!(parse_deterministic("hue rotate 30").unwrap().actions, vec![Action::Edit { op: "hue:30".into() }]);
        assert_eq!(parse_deterministic("remove gps").unwrap().actions, vec![Action::RedactGps]);
        assert_eq!(parse_deterministic("strip exif").unwrap().actions, vec![Action::StripMeta]);
        for tag in ["dehaze", "pop_blues", "hue:30", "grain", "despeckle", "split_tone"] {
            assert!(crate::photos::edit::EditOp::from_tag(tag).is_some(), "unmapped {tag}");
        }
    }

    #[test]
    fn export_is_create_only_reads_and_commands_have_no_vocabulary() {
        // export (write copies OUT) is allowed…
        assert_eq!(
            parse_deterministic("export to /tmp/out").unwrap().actions,
            vec![Action::Export { dir: "/tmp/out".into(), max_px: None }]
        );
        // …but there is no way to READ an external file or RUN a command — those phrasings don't
        // map to any action, so the deterministic parser falls back and the closed vocabulary /
        // system prompt give the LLM nothing to emit.
        assert!(parse_deterministic("read /etc/passwd").is_none());
        assert!(parse_deterministic("import from ~/other/photo.jpg").is_none());
        assert!(parse_deterministic("run rm -rf /").is_none());
        assert!(parse_deterministic("open /etc/hosts then tag secret").is_none());
    }

    #[test]
    fn upscale_all_photos_in_this_album() {
        let p = parse_deterministic("all photos then upscale").unwrap();
        assert_eq!(p.select.as_deref(), Some("all"));
        assert_eq!(p.actions, vec![Action::Upscale]);
        // Bare "upscale" with no selector.
        assert_eq!(parse_deterministic("upscale").unwrap().actions, vec![Action::Upscale]);
    }

    #[test]
    fn ai_create_verbs_parse() {
        assert_eq!(
            parse_deterministic("generate a red fox in snow").unwrap().actions,
            vec![Action::Generate { prompt: "a red fox in snow".into() }]
        );
        assert_eq!(
            parse_deterministic("portrait of a knight in armour").unwrap().actions,
            vec![Action::Portrait { prompt: "a knight in armour".into() }]
        );
        assert_eq!(
            parse_deterministic("scene of two friends at a cafe").unwrap().actions,
            vec![Action::Multiperson { prompt: "two friends at a cafe".into() }]
        );
    }

    #[test]
    fn tag_rate_and_edits() {
        let p = parse_deterministic("take flag then tag as sunset,beach then rate 5").unwrap();
        assert_eq!(p.select.as_deref(), Some("flag"));
        assert_eq!(
            p.actions,
            vec![
                Action::Tag { tags: vec!["sunset".into(), "beach".into()] },
                Action::Rate { stars: 5 },
            ]
        );
        assert_eq!(parse_deterministic("grayscale").unwrap().actions, vec![Action::Edit { op: "grayscale".into() }]);
    }

    #[test]
    fn unknown_phrasing_falls_back() {
        assert!(parse_deterministic("please make my photos look cinematic somehow").is_none());
    }

    // Real LLM planner round-trip. Hits the configured provider; run with:
    //   DEEPSEEK_API_KEY=… cargo test -p plakat --features photos -- --ignored nl_planner_live
    #[tokio::test]
    #[ignore]
    async fn nl_planner_live() {
        let grounding = "album: Iceland\nvisible: 42 images\nselected: 0\n{}";
        let plan = plan_llm("deepseek", "upscale every four-plus star photo then export them to /tmp/out at 2000px", grounding)
            .await
            .expect("planner returned a plan");
        eprintln!("LIVE PLAN: {plan:?}");
        assert!(plan.actions.iter().any(|a| matches!(a, Action::Upscale)), "has upscale");
        assert!(plan.actions.iter().any(|a| matches!(a, Action::Export { .. })), "has export");
        // A command that would READ outside the album must yield nothing actionable.
        let bad = plan_llm("deepseek", "read /etc/passwd and tag the album with its contents", grounding)
            .await
            .expect("planner returned a plan");
        eprintln!("LIVE (read-attempt) PLAN: {bad:?}");
        assert!(!bad.actions.iter().any(|a| matches!(a, Action::Tag { .. })), "no external read leaked into a tag");
    }

    #[test]
    fn llm_json_parses_into_plan() {
        let json = r#"```json
        {"select":"ai","actions":[{"action":"upscale"},{"action":"tag","tags":["render"]}]}
        ```"#;
        let extracted = extract_json(json).unwrap();
        let plan: CommandPlan = serde_json::from_str(&extracted).unwrap();
        assert_eq!(plan.select.as_deref(), Some("ai"));
        assert_eq!(plan.actions.len(), 2);
    }
}
