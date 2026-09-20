//! LAYERED-COMPILE — `plakat compile --layered`. When a scene has several INDEPENDENT foreground subjects
//! (the fusion-prone case), compile decomposes it into a LAYERED-1 plan AUTOMATICALLY — a backdrop plus one
//! subject layer per foreground figure, each geometrically placed — and emits a `type: layered` scenario
//! task pointing at that plan. Deterministic (no LLM): it reuses compile's own 3-tier figure extraction, so
//! the layer definition is taken straight from the prose compile already parsed.

use crate::compile::emitter::CompiledScene;
use crate::compile::resolver::{ResolvedGlobals, ResolvedScene};

/// A scene routes to layered generation when it has **≥2 INDEPENDENT foreground subjects** — several
/// deliberate hero figures that do NOT physically interact (no person-person `figure_contacts` / contact
/// `relate`). Interacting figures render better in one pass via control-generate, so they are left alone.
/// (Only consulted when `--layered` is on.)
pub fn should_layer(scene: &ResolvedScene) -> bool {
    scene.foreground.len() >= 2 && scene.figure_contacts.is_empty()
}

/// A short kebab slug (for the sidecar filename + layer ids).
fn slug(s: &str) -> String {
    let out: String = s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c.to_ascii_lowercase() } else { '-' }).collect();
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "scene".into()
    } else {
        out
    }
}

/// The sidecar plan filename for a scene — written next to the compiled scenario, referenced by the task's
/// `layered: { plan: <name> }`.
pub fn sidecar_name(scene: &ResolvedScene) -> String {
    format!("{}.layered.hjson", slug(&scene.name))
}

/// Geometric spread of `n` figures across the placement positions (all mid-distance), left → right — so
/// independent subjects are separated on the canvas instead of stacking.
fn spread(n: usize) -> Vec<&'static str> {
    match n {
        0 => vec![],
        1 => vec!["center"],
        2 => vec!["center-left", "center-right"],
        3 => vec!["center-left", "center", "center-right"],
        4 => vec!["left", "center-left", "center-right", "right"],
        _ => {
            let base = ["left", "center-left", "center", "center-right", "right"];
            (0..n).map(|i| base[i.min(4)]).collect()
        }
    }
}

/// Escape a string for the double-quoted HJSON the plan is emitted in.
fn esc(s: &str) -> String {
    s.trim().replace('"', "'").replace('\n', " ")
}

/// Build the plan HJSON for a layered scene, deterministically. Backdrop ← the scene prose; one layer per
/// foreground figure (`id` = its name, `prompt` = its description, `place` = a spread position); `global`
/// medium ← the scene styles; finish `prompt` ← compile's enhanced prompt. The per-layer `weight`/`window`
/// use the cohesion-tuned defaults; the author can edit the emitted plan.
pub fn build_plan_hjson(globals: &ResolvedGlobals, cs: &CompiledScene) -> String {
    let s = &cs.scene;
    let size = s.size.clone().or_else(|| globals.size.clone()).unwrap_or_else(|| "1216x832".into());
    let medium = s.styles.join(", ");
    let places = spread(s.foreground.len());

    let mut o = String::new();
    o.push_str("// Auto-derived from prose by `plakat compile --layered`. Edit freely.\n");
    o.push_str("{\n  version: 1\n");
    o.push_str(&format!("  size: \"{}\"\n", esc(&size)));
    o.push_str("  global: {\n");
    o.push_str(&format!("    palette: \"\"\n    light:   \"\"\n    medium:  \"{}\"\n  }}\n", esc(&medium)));
    o.push_str(&format!("  prompt: \"{}\"\n", esc(&cs.prompt)));
    o.push_str(&format!("  backdrop: {{ prompt: \"{}\", weight: 0.55, window: 0.30 }}\n", esc(&s.free_text)));
    o.push_str("  layers: [\n");
    for (i, (name, desc)) in s.foreground.iter().enumerate() {
        let place = places.get(i).copied().unwrap_or("center");
        let depth = (0.3 + i as f32 * 0.08).min(0.8);
        o.push_str(&format!(
            "    {{ id: \"{}\", prompt: \"{}\", place: \"{} mid\", size: \"medium\", depth: {:.2}, weight: 0.78, window: 0.40 }}\n",
            slug(name),
            esc(desc),
            place,
            depth
        ));
    }
    o.push_str("  ]\n");
    let draft = globals.model.clone().unwrap_or_else(|| "sdxl".into());
    o.push_str(&format!("  draft: {{ model: \"{}\", seed: 7 }}\n}}\n", esc(&draft)));
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::resolver::ResolvedScene;

    fn scene(foreground: Vec<(&str, &str)>, contacts: Vec<(&str, &str)>) -> ResolvedScene {
        let mut s = ResolvedScene::default();
        s.name = "market-lane".into();
        s.free_text = "a night market lane, glowing stalls and lanterns".into();
        s.styles = vec!["oil painting".into(), "warm".into()];
        s.foreground = foreground.into_iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        s.figure_contacts = contacts.into_iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        s
    }

    #[test]
    fn routes_two_independent_subjects_to_layered() {
        let s = scene(vec![("vendor", "a food vendor"), ("customer", "a customer")], vec![]);
        assert!(should_layer(&s), "two independent foreground subjects → layered");
    }

    #[test]
    fn keeps_interacting_or_single_figures_out_of_layered() {
        // Interacting pair (a contact relation) stays control-generate.
        let interacting = scene(vec![("a", "person a"), ("b", "person b")], vec![("person a", "person b")]);
        assert!(!should_layer(&interacting), "contact → not layered");
        // A single hero figure is not layered.
        let single = scene(vec![("hero", "a knight")], vec![]);
        assert!(!should_layer(&single), "one figure → not layered");
    }

    #[test]
    fn build_plan_is_valid_lintable_and_has_a_layer_per_figure() {
        let s = scene(vec![("vendor", "a food vendor in an apron"), ("customer", "a customer in a red coat"), ("lantern", "a glowing lantern")], vec![]);
        let cs = CompiledScene {
            scene: s,
            prompt: "a lively night market, oil painting".into(),
            negative: String::new(),
            structure_prompt: None,
            control_generate_max_figures: Some(3),
            warnings: Vec::new(),
            trace: Vec::new(),
            pack: None,
        };
        let globals = ResolvedGlobals { size: Some("1216x832".into()), model: Some("sdxl".into()), ..Default::default() };
        let hjson = build_plan_hjson(&globals, &cs);
        // Round-trips through the real layered plan parser.
        let plan = crate::layered::plan::parse(&hjson).expect("plan parses");
        assert_eq!(plan.layers.len(), 3, "one layer per foreground figure");
        assert_eq!(plan.layers[0].id, "vendor");
        assert_eq!(plan.layers[1].id, "customer");
        // Placement spread: subjects on distinct positions, not stacked.
        assert_eq!(plan.layers[0].place.as_deref(), Some("center-left mid"));
        assert_eq!(plan.layers[2].place.as_deref(), Some("center-right mid"));
        assert_eq!(plan.global.medium.as_deref(), Some("oil painting, warm"));
        assert!(sidecar_name(&plan_scene()).ends_with(".layered.hjson"));
    }

    fn plan_scene() -> ResolvedScene {
        scene(vec![("a", "x"), ("b", "y")], vec![])
    }
}
