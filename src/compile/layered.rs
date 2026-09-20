//! LAYERED-COMPILE — `plakat compile --layered`. When a scene has several INDEPENDENT foreground subjects
//! (the fusion-prone case), compile decomposes it into a LAYERED-1 plan AUTOMATICALLY — a backdrop plus one
//! subject layer per foreground figure, each geometrically placed — and emits a `type: layered` scenario
//! task pointing at that plan. Deterministic (no LLM): it reuses compile's own 3-tier figure extraction, so
//! the layer definition is taken straight from the prose compile already parsed.

use crate::compile::emitter::CompiledScene;
use crate::compile::resolver::{ResolvedGlobals, ResolvedScene};

/// A scene routes to layered generation when it has **≥2 INDEPENDENT foreground subjects** — several
/// deliberate hero figures/objects that do NOT physically interact (no person-person `figure_contacts` /
/// contact `relate`). Interacting figures render better in one pass via control-generate, so they are left
/// alone. Robustness: only DISTINCT, NON-EMPTY subject descriptions count — a degenerate foreground (a blank
/// entry, or the same subject listed twice) is not a multi-subject scene. (Only consulted when `--layered`.)
pub fn should_layer(scene: &ResolvedScene) -> bool {
    scene.figure_contacts.is_empty() && distinct_subjects(&scene.foreground) >= 2
}

/// Count the distinct, non-empty subject descriptions in a foreground list (case/whitespace-insensitive), so
/// an empty entry or a duplicate doesn't inflate the subject count.
fn distinct_subjects(foreground: &[(String, String)]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for (_, desc) in foreground {
        let key = desc.split_whitespace().collect::<Vec<_>>().join(" ").to_ascii_lowercase();
        if !key.is_empty() {
            seen.insert(key);
        }
    }
    seen.len()
}

/// Positional cue words that mean a subject is ELEVATED / not standing on the ground — so its box is raised
/// into the upper canvas and made smaller, instead of sharing the figures' ground-line. Generic spatial
/// language (not scene-specific): a hanging lantern, an overhead sign, a bird in the sky.
const ELEVATED_CUES: &[&str] = &["overhead", "above", "hanging", "hung", "suspended", "ceiling", "sky", "aloft", "up high", "high up", "floating", "in the air", "airborne"];

/// True when a subject's description reads as elevated (see [`ELEVATED_CUES`]).
fn is_elevated(desc: &str) -> bool {
    let d = desc.to_ascii_lowercase();
    ELEVATED_CUES.iter().any(|c| d.contains(c))
}

/// A depth override from explicit, UNAMBIGUOUS prose cues: farther for background/distance language, nearer
/// for foreground/close-up language. Deliberately conservative — bare "behind"/"in front" are positional
/// (behind a *cart*, in front of a *stall*), not scene depth, so they're excluded to avoid mis-ordering.
/// `None` = no cue (fall back to the gentle by-index ordering).
fn depth_cue(desc: &str) -> Option<f32> {
    let d = desc.to_ascii_lowercase();
    if ["in the background", "background", "in the distance", "distant", "far away", "far-off", "receding"].iter().any(|c| d.contains(c)) {
        Some(0.72)
    } else if ["in the foreground", "foreground", "close-up", "closest", "nearest", "up front"].iter().any(|c| d.contains(c)) {
        Some(0.18)
    } else {
        None
    }
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

/// The shared ground-line: standing figures rest their box bottoms here, so grounding shadows line up on one
/// floor instead of each subject floating at its own level.
const GROUND_Y: f32 = 0.96;
/// A standing figure's box height (fraction of canvas) — tall, from the ground-line up.
const FIGURE_H: f32 = 0.62;
/// An elevated subject's box: a smaller square in the upper canvas.
const ELEVATED_TOP: f32 = 0.06;
const ELEVATED_H: f32 = 0.26;

/// Lay out `n` subjects as **non-overlapping boxes** across the canvas, left → right. Ground subjects share a
/// baseline ([`GROUND_Y`]) as tall figure boxes; `elevated[i]` subjects are raised into the upper canvas as
/// smaller boxes centred in their column. `canvas_aspect` (w/h) widens the horizontal gutters on a wide
/// poster. Returns `[x0,y0,x1,y1]` fractions per subject.
fn layer_boxes(n: usize, elevated: &[bool], canvas_aspect: f32) -> Vec<[f32; 4]> {
    if n == 0 {
        return Vec::new();
    }
    // Columns with gutters; a wider canvas gets a little more air between subjects (capped so it stays sane).
    let gutter = (0.04 * canvas_aspect.clamp(0.6, 2.0)).min(0.09);
    let col_w = (1.0 - gutter * (n as f32 + 1.0)) / n as f32;
    let fill = 0.92_f32; // subjects don't quite fill their column (breathing room, guarantees no overlap)
    (0..n)
        .map(|i| {
            let col_x0 = gutter + i as f32 * (col_w + gutter);
            let cx = col_x0 + col_w / 2.0;
            let is_up = elevated.get(i).copied().unwrap_or(false);
            let bw = (col_w * fill).min(if is_up { col_w * 0.8 } else { col_w * fill });
            let x0 = (cx - bw / 2.0).clamp(0.0, 1.0);
            let x1 = (cx + bw / 2.0).clamp(0.0, 1.0);
            let (y0, y1) = if is_up {
                (ELEVATED_TOP, (ELEVATED_TOP + ELEVATED_H).min(1.0))
            } else {
                ((GROUND_Y - FIGURE_H).max(0.0), GROUND_Y)
            };
            [x0, y0, x1, y1]
        })
        .collect()
}

/// Escape a string for the double-quoted HJSON the plan is emitted in.
fn esc(s: &str) -> String {
    s.trim().replace('"', "'").replace('\n', " ")
}

/// Parse a `WxH` size into a canvas aspect (w/h), defaulting to 1216×832 ≈ 1.46 when unparseable.
fn size_aspect(size: &str) -> f32 {
    let mut it = size.split(['x', 'X', '*']);
    match (it.next().and_then(|w| w.trim().parse::<f32>().ok()), it.next().and_then(|h| h.trim().parse::<f32>().ok())) {
        (Some(w), Some(h)) if w > 0.0 && h > 0.0 => w / h,
        _ => 1216.0 / 832.0,
    }
}

/// Build the plan HJSON for a layered scene, deterministically. Backdrop ← the scene prose; one layer per
/// foreground subject (`id` = its name, `prompt` = its description). Each subject gets an explicit,
/// **non-overlapping** `box`: ground subjects share a baseline as tall figure boxes, subjects the prose marks
/// as elevated (overhead/hanging/…) are raised into the upper canvas. `depth` comes from explicit prose cues
/// (background/foreground) when present, else a gentle back-to-front by index. `global` medium ← scene styles;
/// finish `prompt` ← compile's enhanced prompt. The `weight`/`window` use the cohesion-tuned defaults; the
/// author can edit the emitted plan.
pub fn build_plan_hjson(globals: &ResolvedGlobals, cs: &CompiledScene) -> String {
    let s = &cs.scene;
    let size = s.size.clone().or_else(|| globals.size.clone()).unwrap_or_else(|| "1216x832".into());
    let medium = s.styles.join(", ");
    let elevated: Vec<bool> = s.foreground.iter().map(|(_, d)| is_elevated(d)).collect();
    let boxes = layer_boxes(s.foreground.len(), &elevated, size_aspect(&size));

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
        let b = boxes.get(i).copied().unwrap_or([0.3, 0.34, 0.7, 0.96]);
        let depth = depth_cue(desc).unwrap_or_else(|| (0.3 + i as f32 * 0.08).min(0.8));
        o.push_str(&format!(
            "    {{ id: \"{}\", prompt: \"{}\", box: [{:.3}, {:.3}, {:.3}, {:.3}], depth: {:.2}, weight: 0.78, window: 0.40 }}\n",
            slug(name),
            esc(desc),
            b[0], b[1], b[2], b[3],
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
        // Explicit boxes: left → right, non-overlapping, and figures share the ground-line.
        let b0 = plan.layers[0].bbox.expect("box emitted");
        let b1 = plan.layers[1].bbox.unwrap();
        let b2 = plan.layers[2].bbox.unwrap();
        assert!(b0[2] <= b1[0] + 1e-4 && b1[2] <= b2[0] + 1e-4, "boxes left→right, non-overlapping: {b0:?} {b1:?} {b2:?}");
        assert!((b0[3] - b1[3]).abs() < 1e-4, "the two ground figures share a baseline");
        // The "glowing lantern" has no elevation cue, so it stays a ground figure here (see the elevation test).
        assert_eq!(plan.global.medium.as_deref(), Some("oil painting, warm"));
        assert!(sidecar_name(&plan_scene()).ends_with(".layered.hjson"));
    }

    #[test]
    fn elevated_cue_raises_a_subject_and_depth_cue_orders_it() {
        let s = scene(vec![("vendor", "a food vendor on the cobblestones"), ("lantern", "a paper lantern hanging overhead"), ("moon", "a full moon in the distant background")], vec![]);
        let cs = CompiledScene {
            scene: s,
            prompt: "a night market".into(),
            negative: String::new(),
            structure_prompt: None,
            control_generate_max_figures: Some(3),
            warnings: Vec::new(),
            trace: Vec::new(),
            pack: None,
        };
        let globals = ResolvedGlobals { size: Some("1216x832".into()), model: Some("sdxl".into()), ..Default::default() };
        let plan = crate::layered::plan::parse(&build_plan_hjson(&globals, &cs)).expect("parses");
        let vendor = plan.layers[0].bbox.unwrap();
        let lantern = plan.layers[1].bbox.unwrap();
        assert!(lantern[3] < vendor[3], "the overhead lantern sits above the vendor's ground-line");
        assert!(lantern[1] < 0.2, "elevated box starts near the top: {lantern:?}");
        // "distant background" pushes the moon far; "on the cobblestones" leaves the vendor at the by-index depth.
        assert!(plan.layers[2].depth.unwrap() > 0.6, "background cue → far depth");
    }

    #[test]
    fn degenerate_foreground_does_not_route_to_layered() {
        // Two entries, same subject → one distinct subject → not layered.
        assert!(!should_layer(&scene(vec![("a", "a knight"), ("b", "A Knight")], vec![])), "duplicate subject is one subject");
        // A blank second entry doesn't count.
        assert!(!should_layer(&scene(vec![("a", "a knight"), ("b", "   ")], vec![])), "blank entry doesn't count");
    }

    fn plan_scene() -> ResolvedScene {
        scene(vec![("a", "x"), ("b", "y")], vec![])
    }
}
