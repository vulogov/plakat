//! LAYERED-1 P4 — the **planner**. An LLM decomposes a complex prose description into a layer plan: the
//! global look, the backdrop, and a small set of INDEPENDENT subject layers, each with a placement + size.
//! The result is written as an editable plan HJSON (the same shape [`super::plan`] parses and [`super::lint`]
//! checks), so the author can tweak it before rendering.
//!
//! The prompt-building + HJSON assembly are pure (offline-tested); only the single `llm::enhance` call needs
//! a model. Nothing scene-specific is baked into the system prompt — every attribute is derived from the
//! user's description. Reuses the `--enhance` provider stack (deepseek / gemini / local GGUF).

use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;

use crate::layered::plan::{self, LayerPlan};

const SYSTEM: &str = "\
You decompose an image description into a LAYERED PLAN for a text-to-image renderer. A plan has a global \
look, a backdrop (the environment), and a small set of INDEPENDENT subject layers.

Rules:
- Put each INDEPENDENT subject (a person, an animal, a distinct object) in its OWN layer. Subjects that \
physically interact (holding hands, one riding another, an embrace) stay in ONE layer.
- The backdrop is the setting / environment ONLY — never a subject.
- Give each layer a `place` (position + distance in words) and a `size` — NOT pixel coordinates.
- Extract the overall colour palette, lighting, and medium/technique into `global`; do not repeat them in \
the per-layer prompts.
- `prompt` (top level) is the whole-scene finish prompt, in the requested style.
- Keep 1 to 6 subject layers; fewer, well-separated subjects render best.

Output ONLY a JSON object (no prose, no markdown fences):
{\"palette\":\"...\",\"light\":\"...\",\"medium\":\"...\",\"prompt\":\"...\",\"backdrop\":\"...\",\
\"layers\":[{\"id\":\"short-id\",\"prompt\":\"one subject, full detail\",\"place\":\"center-left mid\",\
\"size\":\"medium\",\"depth\":0.3}]}

place: a position (left | center-left | center | center-right | right) plus a distance (closer | mid | \
farther). size: small | medium | large. depth: 0.0 (nearest) .. 1.0 (farthest), consistent with distance.";

/// The LLM's plan, before defaults + id assignment.
#[derive(Deserialize, Default)]
pub struct PlanJson {
    pub palette: Option<String>,
    pub light: Option<String>,
    pub medium: Option<String>,
    pub prompt: Option<String>,
    pub backdrop: Option<String>,
    #[serde(default)]
    pub layers: Vec<LayerJson>,
}

#[derive(Deserialize, Default)]
pub struct LayerJson {
    pub id: Option<String>,
    pub prompt: String,
    pub place: Option<String>,
    pub size: Option<String>,
    pub depth: Option<f32>,
    #[serde(rename = "box")]
    pub bbox: Option<[f32; 4]>,
}

/// A kebab-case id from a layer prompt's first couple of words.
fn slug(prompt: &str) -> String {
    let s: String = prompt
        .split_whitespace()
        .filter(|w| !matches!(w.to_ascii_lowercase().as_str(), "a" | "an" | "the"))
        .take(2)
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        "subject".into()
    } else {
        s
    }
}

/// Assign a stable, unique id to every layer that lacks one (deriving from its prompt), de-duplicating.
pub fn assign_ids(layers: &mut [LayerJson]) {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for l in layers.iter_mut() {
        let base = l.id.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| slug(&l.prompt));
        let mut id = base.clone();
        let mut k = 2;
        while !seen.insert(id.clone()) {
            id = format!("{base}-{k}");
            k += 1;
        }
        l.id = Some(id);
    }
}

/// Escape a string for embedding in the double-quoted HJSON we emit.
fn esc(s: &str) -> String {
    s.trim().replace('"', "'").replace('\n', " ")
}

/// Assemble a plan HJSON from the LLM's [`PlanJson`] with sensible defaults. Ids are assigned first.
pub fn to_hjson(mut pj: PlanJson, w: u32, h: u32, draft_model: &str) -> String {
    assign_ids(&mut pj.layers);
    let mut out = String::new();
    out.push_str("{\n  version: 1\n");
    out.push_str(&format!("  size: \"{w}x{h}\"\n"));
    out.push_str("  global: {\n");
    out.push_str(&format!("    palette: \"{}\"\n", esc(pj.palette.as_deref().unwrap_or(""))));
    out.push_str(&format!("    light:   \"{}\"\n", esc(pj.light.as_deref().unwrap_or(""))));
    out.push_str(&format!("    medium:  \"{}\"\n", esc(pj.medium.as_deref().unwrap_or(""))));
    out.push_str("  }\n");
    out.push_str(&format!("  prompt: \"{}\"\n", esc(pj.prompt.as_deref().unwrap_or(""))));
    out.push_str(&format!("  backdrop: {{ prompt: \"{}\", weight: 0.6, window: 0.25 }}\n", esc(pj.backdrop.as_deref().unwrap_or(""))));
    out.push_str("  layers: [\n");
    for l in &pj.layers {
        let id = l.id.as_deref().unwrap_or("subject");
        let depth = l.depth.unwrap_or(0.4).clamp(0.0, 1.0);
        if let Some(b) = l.bbox {
            out.push_str(&format!(
                "    {{ id: \"{}\", prompt: \"{}\", box: [{:.3}, {:.3}, {:.3}, {:.3}], depth: {:.2} }}\n",
                esc(id), esc(&l.prompt), b[0], b[1], b[2], b[3], depth
            ));
        } else {
            let place = esc(l.place.as_deref().unwrap_or("center mid"));
            let size = esc(l.size.as_deref().unwrap_or("medium"));
            out.push_str(&format!(
                "    {{ id: \"{}\", prompt: \"{}\", place: \"{}\", size: \"{}\", depth: {:.2} }}\n",
                esc(id), esc(&l.prompt), place, size, depth
            ));
        }
    }
    out.push_str("  ]\n");
    out.push_str(&format!("  draft: {{ model: \"{}\", seed: 7 }}\n}}\n", esc(draft_model)));
    out
}

/// Object variant of the analyser's JSON extractor: strip markdown fences, then take the outermost `{...}`.
fn extract_json_object(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix("```json").or_else(|| s.strip_prefix("```")).unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b > a => s[a..=b].to_string(),
        _ => s.trim().to_string(),
    }
}

/// Decompose `prose` into a plan HJSON via the layout LLM `provider`. Returns the HJSON text (parseable by
/// [`plan::parse`]); the caller lints + writes it.
pub async fn plan_prose(prose: &str, w: u32, h: u32, draft_model: &str, provider: &str, device: &Device, seed: u64) -> Result<String> {
    let user = format!("Description: {prose}\nTarget size: {w}x{h}\nReturn the JSON object.");
    // `ollama` / `ollama:<model>` route to a local Ollama server (a bigger model than the in-process GGUF
    // aliases); any other provider is a local GGUF alias run in-process on `device`.
    let pl = provider.to_lowercase();
    let raw = if pl == "ollama" || pl.starts_with("ollama:") {
        let model = if pl.starts_with("ollama:") { &provider["ollama:".len()..] } else { crate::prompt::ollama::DEFAULT_MODEL };
        crate::prompt::ollama::enhance_with_system_model(model, SYSTEM, &user).await.context("planner via Ollama")?
    } else {
        let opts = crate::llm::EnhanceOpts { seed, temperature: 0.0, max_new_tokens: 768 };
        crate::llm::enhance(provider, device.clone(), SYSTEM, &user, opts).await.map_err(|e| anyhow::anyhow!("planner LLM: {e}"))?
    };
    let json = extract_json_object(&raw);
    let pj: PlanJson = serde_json::from_str(&json).with_context(|| format!("parsing the planner JSON (raw: {})", raw.chars().take(240).collect::<String>()))?;
    anyhow::ensure!(!pj.layers.is_empty(), "the planner produced no subject layers — rephrase the description, or write the plan by hand");
    let hjson = to_hjson(pj, w, h, draft_model);
    // Validate it round-trips through the real parser before handing it back.
    let _: LayerPlan = plan::parse(&hjson).context("the assembled plan did not parse")?;
    Ok(hjson)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered::lint;

    fn pj() -> PlanJson {
        PlanJson {
            palette: Some("muted autumn".into()),
            light: Some("soft overcast".into()),
            medium: Some("oil painting".into()),
            prompt: Some("two friends walking in the rain".into()),
            backdrop: Some("a cobblestone square".into()),
            layers: vec![
                LayerJson { prompt: "a boy in a yellow raincoat".into(), place: Some("center-left mid".into()), size: Some("large".into()), depth: Some(0.3), ..Default::default() },
                LayerJson { prompt: "a girl with a red umbrella".into(), place: Some("center-right mid".into()), size: Some("large".into()), depth: Some(0.3), ..Default::default() },
            ],
        }
    }

    #[test]
    fn to_hjson_is_a_valid_lintable_plan() {
        let hjson = to_hjson(pj(), 1216, 832, "sdxl-lightning");
        let plan = plan::parse(&hjson).expect("parses");
        assert_eq!(plan.layers.len(), 2);
        assert_eq!(plan.global.medium.as_deref(), Some("oil painting"));
        assert_eq!(plan.size.as_deref(), Some("1216x832"));
        // Lint should not error (it may warn).
        let geom = lint::geometry_for_model("sdxl");
        assert!(lint::is_clean(&lint::lint(&plan, &geom, 1216, 832)) || lint::lint(&plan, &geom, 1216, 832).iter().all(|i| i.severity != lint::Severity::Error));
    }

    #[test]
    fn ids_are_assigned_and_unique() {
        let mut layers = vec![
            LayerJson { prompt: "a red fox".into(), ..Default::default() },
            LayerJson { prompt: "a red fox".into(), ..Default::default() }, // same slug → deduped
            LayerJson { id: Some("hero".into()), prompt: "a knight".into(), ..Default::default() },
        ];
        assign_ids(&mut layers);
        let ids: Vec<&str> = layers.iter().map(|l| l.id.as_deref().unwrap()).collect();
        assert_eq!(ids[0], "red-fox");
        assert_eq!(ids[1], "red-fox-2", "duplicate slug is disambiguated");
        assert_eq!(ids[2], "hero", "explicit id kept");
    }

    #[test]
    fn slug_drops_articles_and_nonalnum() {
        assert_eq!(slug("a ceramic bowl of pears"), "ceramic-bowl");
        assert_eq!(slug("The Old Lighthouse!"), "old-lighthouse");
        assert_eq!(slug("!!!"), "subject");
    }

    #[test]
    fn extract_json_object_strips_fences_and_prose() {
        let raw = "Sure:\n```json\n{\"prompt\":\"x\",\"layers\":[]}\n```\nhope that helps";
        assert_eq!(extract_json_object(raw), "{\"prompt\":\"x\",\"layers\":[]}");
    }

    #[test]
    fn explicit_box_layer_emits_box() {
        let mut p = pj();
        p.layers[0].bbox = Some([0.1, 0.2, 0.5, 0.9]);
        let hjson = to_hjson(p, 1024, 1024, "sdxl");
        assert!(hjson.contains("box: [0.100, 0.200, 0.500, 0.900]"), "explicit box emitted: {hjson}");
        let plan = plan::parse(&hjson).unwrap();
        assert_eq!(plan.layers[0].bbox, Some([0.1, 0.2, 0.5, 0.9]));
    }
}
