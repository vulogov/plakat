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
    if let Some(r) = strip_any(s, &["relight "]) {
        return Some(Action::Relight { prompt: r.trim().to_string() });
    }
    if let Some(r) = strip_any(s, &["smart album ", "save as smart album ", "save smart album "]) {
        return Some(Action::SmartAlbum { name: r.trim().to_string() });
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
        "grayscale" | "greyscale" | "black and white" | "b&w" | "desaturate" => "grayscale",
        "crop" | "crop square" | "square crop" | "crop 1:1" => "crop_square",
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
  {\"action\":\"edit\",\"op\":\"rotate_cw|rotate_ccw|rotate_180|flip_h|flip_v|grayscale|crop_square\"}
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
