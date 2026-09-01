//! Assemble the LLM inputs from a [`ResolvedScene`]: the positive *user* text
//! (header + description + footer) and the family-aware *system* prompts for the
//! positive and negative calls. With `--no-enhance` the assembled user text is
//! the final prompt verbatim (deterministic).

use super::ModelFamily;
use super::resolver::ResolvedScene;

/// Tidy a comma-joined prompt: collapse repeated commas/spaces, trim stray
/// leading/trailing punctuation. Deterministic and idempotent.
pub fn clean(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_sep = true; // suppress leading separators
    for tok in s.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        if !last_was_sep {
            out.push_str(", ");
        }
        out.push_str(t);
        last_was_sep = false;
    }
    out
}

/// The positive *user* text from an explicit body (possibly a translation): header, the resolved
/// `composition:` (reusable components), the body prose, then footer — comma-joined and cleaned.
/// 6.26.x extends the old header+body+footer order by inserting the composition BEFORE the prose
/// (compose, then prose); everything else (header/footer/persona/style) is unchanged.
pub fn assemble_with_body(scene: &ResolvedScene, body: &str) -> String {
    let joined = [
        scene.header.as_str(),
        scene.composition_text.as_str(),
        body.trim(),
        scene.footer.as_str(),
    ]
    .join(", ");
    clean(&joined)
}

/// The positive *user* text from the scene's own free text. Verbatim prompt
/// under `--no-enhance`, LLM input otherwise.
pub fn assemble_input(scene: &ResolvedScene) -> String {
    assemble_with_body(scene, scene.free_text.trim())
}

const POSITIVE_BASE: &str = "\
You are an expert prompt engineer for text-to-image diffusion models. Rewrite the \
user's scene description into a single optimized image-generation prompt for the SPECIFIC \
target model described below. Output ONLY the prompt — no preamble, no quotation marks, no \
markdown, no explanation.\n\
Optimize for the target model while preserving the description:\n\
- Keep EVERY distinct subject and element the user describes — all people/characters (and \
their stated variety, e.g. men, women, and children of different ages), objects, materials, \
and colours. Never collapse a described group of people into a single word, and never silently \
drop an element. If the description is richer than the model's token budget, TIGHTEN the wording \
(drop function words, merge into keyword clusters) so everything fits — do NOT omit content.\n\
- When the description includes living subjects (people, animals), name them CONCRETELY (e.g. \
'men, women and children walking', not the vague 'townsfolk') and place them near the FRONT of \
the prompt, ahead of the scenery — models drop subjects that are vague or buried after a long \
environment description.\n\
- Honour the STYLE DIRECTION exactly as given — render in the style the user asked for and do \
NOT substitute a different style or medium. Carry the user's style words into the prompt and put \
them up front so they anchor the image. If no style is given, do not invent one.\n\
- Do not invent unrelated elements.";

const FAMILY_SD15: &str = "\
TARGET MODEL: Stable Diffusion 1.5 family (CLIP text encoder, 77-token limit).\n\
- Output comma-separated short phrases and keywords, not full sentences.\n\
- Keep under ~75 tokens. Front-load subject and style.\n\
- Order: [style], [subject], [composition], [lighting], [quality boosters].\n\
- Quality boosters allowed: masterpiece, best quality, highly detailed, sharp focus.";

const FAMILY_SDXL: &str = "\
TARGET MODEL: Stable Diffusion XL (dual CLIP, ~150-token effective range).\n\
- Mix natural language with keyword clusters. Aim 60-150 tokens.\n\
- Order: [style], [detailed subject], [environment/atmosphere], [lighting], [technical].\n\
- Artist references and medium descriptions work well.";

const FAMILY_FLUX: &str = "\
TARGET MODEL: Flux (transformer, guidance distillation, no CLIP token limit).\n\
- Write natural-language prose, not comma-separated tokens. Aim 80-200 tokens.\n\
- Do NOT use SD-style quality boosters (no effect). Be specific about spatial \
relationships, sizes, and positions.";

fn family_section(f: ModelFamily) -> &'static str {
    match f {
        ModelFamily::Sd15 | ModelFamily::Unknown => FAMILY_SD15,
        ModelFamily::Sdxl => FAMILY_SDXL,
        ModelFamily::Flux => FAMILY_FLUX,
    }
}

/// The upper end of a family's *effective* prompt budget in tokens — beyond this, later prompt
/// content is increasingly ignored at render time (CLIP truncates hard at 77; SDXL's dual CLIP
/// stretches the useful range; Flux's T5 is far larger). Used to WARN, never to truncate.
pub fn family_token_budget(f: ModelFamily) -> usize {
    match f {
        ModelFamily::Sd15 | ModelFamily::Unknown => 77,
        ModelFamily::Sdxl => 150,
        ModelFamily::Flux => 300,
    }
}

/// Rough CLIP/T5 token estimate of a prompt (a warning heuristic, not exact): whitespace- and
/// comma-separated pieces scaled by ~1.3 for sub-word splitting. Deterministic.
pub fn estimate_tokens(text: &str) -> usize {
    let words = text.split(|c: char| c.is_whitespace() || c == ',').filter(|s| !s.is_empty()).count();
    (words * 4).div_ceil(3)
}

/// Whether the user's STYLE DIRECTION survived into the generated prompt — style-agnostic: it
/// checks that at least one *significant* word from any style value appears (case-insensitively)
/// in the prompt, whatever the style is. Returns `true` when no style was given (nothing to check).
/// Catches the "enhancer dropped the style the user asked for" failure without assuming any
/// particular style (photographic, illustration, or otherwise).
pub fn style_survived(styles: &[String], prompt: &str) -> bool {
    // Short/function words carry no style signal; ignore them so a match means the real style term.
    const STOP: [&str; 12] = [
        "the", "a", "an", "of", "in", "and", "or", "with", "for", "to", "style", "art",
    ];
    let pl = prompt.to_lowercase();
    let mut had_signal = false;
    for s in styles {
        for w in s.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
            if w.len() < 3 || STOP.contains(&w) {
                continue;
            }
            had_signal = true;
            if pl.contains(w) {
                return true; // at least one style word carried through
            }
        }
    }
    // If there were style words to check and none appeared, the style was dropped → false.
    // If there was no style (or only stop-words), there's nothing to enforce → true.
    !had_signal
}

/// Post-generation diligence warnings for one compiled scene (6.26.2) — surfaced to the user
/// instead of silently accepting a prompt that overflows the model's budget or drops the style
/// they asked for. `styles` are the scene's STYLE DIRECTION values; `prompt` is the final positive.
pub fn scene_warnings(
    name: &str,
    styles: &[String],
    prompt: &str,
    family: ModelFamily,
    enhanced: bool,
) -> Vec<String> {
    let mut w = Vec::new();
    let est = estimate_tokens(prompt);
    let budget = family_token_budget(family);
    if est > budget {
        w.push(format!(
            "scene '{name}': prompt ~{est} tokens exceeds the ~{budget}-token effective range for {} — later elements may be ignored at render; tighten the description or split the scene",
            family.label()
        ));
    }
    // The style-drop check only makes sense when the enhancer ran — `--no-enhance` keeps the
    // user's verbatim text (which never had the STYLE DIRECTION injected in the first place).
    if enhanced && !styles.is_empty() && !style_survived(styles, prompt) {
        let asked = styles.join(", ");
        w.push(format!(
            "scene '{name}': the STYLE you asked for ('{asked}') did not appear in the prompt — the enhancer may have dropped it (try a stronger --provider, or --no-enhance to keep your wording)"
        ));
    }
    w
}

/// Build the positive-call system prompt: base + style injections + the
/// (already-loaded) persona fragments + the family-specific section.
/// `base_override` replaces [`POSITIVE_BASE`] when `--compile-system` is given.
pub fn positive_system(
    scene: &ResolvedScene,
    base_override: Option<&str>,
    persona_fragments: &[String],
) -> String {
    let mut s = String::new();
    s.push_str(base_override.unwrap_or(POSITIVE_BASE));
    for style in &scene.styles {
        s.push_str("\n\nSTYLE DIRECTION: ");
        s.push_str(style);
    }
    for frag in persona_fragments {
        s.push_str("\n\nPERSONA: ");
        s.push_str(frag.trim());
    }
    s.push_str("\n\n");
    s.push_str(family_section(scene.family));
    s
}

const NEGATIVE_BASE: &str = "\
You write the NEGATIVE prompt for a text-to-image model: a comma-separated list of \
things to avoid, given the POSITIVE prompt the user will render. Output ONLY \
comma-separated terms — no preamble, no sentences, no markdown.";

const NEG_FAMILY_FLUX: &str = "\
TARGET: Flux. Guidance distillation makes negatives largely ineffective — output a \
SHORT negative (10-20 tokens) with only critical content exclusions. Do NOT list \
quality terms; they have no effect.";

const NEG_FAMILY_SD: &str = "\
TARGET: Stable Diffusion. Produce ~30-50 tokens covering quality defects \
(blurry, low quality, artifacts), anatomy errors, and unwanted content.";

/// Build the negative-call system prompt. Seed terms (the merged `negative:`
/// values) are injected as a hard must-include instruction.
pub fn negative_system(scene: &ResolvedScene) -> String {
    let mut s = String::from(NEGATIVE_BASE);
    s.push_str("\n\n");
    s.push_str(match scene.family {
        ModelFamily::Flux => NEG_FAMILY_FLUX,
        _ => NEG_FAMILY_SD,
    });
    let seeds = scene.negative_seeds.trim();
    if !seeds.is_empty() {
        s.push_str("\n\nThe following terms MUST appear in your output verbatim: ");
        s.push_str(seeds);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(header: &str, body: &str, footer: &str, family: ModelFamily) -> ResolvedScene {
        ResolvedScene {
            header: header.into(),
            free_text: body.into(),
            footer: footer.into(),
            family,
            ..Default::default()
        }
    }

    #[test]
    fn style_survived_is_style_agnostic() {
        // A carried-through style word → survived (any style, not just photographic).
        assert!(style_survived(&["cinematic photography".into()], "cinematic street scene, dusk"));
        assert!(style_survived(&["watercolor illustration".into()], "a watercolor of a harbour"));
        // Style words wholly absent from the prompt → dropped.
        assert!(!style_survived(&["cinematic photography".into()], "a detailed illustration of a street"));
        // No style given → nothing to enforce (true).
        assert!(style_survived(&[], "anything"));
        // Only stop-words in the style → nothing enforceable (true).
        assert!(style_survived(&["in the style of".into()], "unrelated prompt"));
    }

    #[test]
    fn scene_warnings_flag_budget_and_dropped_style() {
        // Over-budget prompt warns (SD15 ~77 tokens).
        let long = "word ".repeat(80);
        let w = scene_warnings("s", &[], &long, ModelFamily::Sd15, true);
        assert!(w.iter().any(|m| m.contains("exceeds")), "budget warned: {w:?}");
        // Enhanced + style dropped → style warning; verbatim (enhanced=false) → no style warning.
        let dropped = scene_warnings("s", &["cinematic photography".into()], "a plain illustration", ModelFamily::Sdxl, true);
        assert!(dropped.iter().any(|m| m.contains("STYLE you asked for")), "style warned: {dropped:?}");
        let verbatim = scene_warnings("s", &["cinematic photography".into()], "a plain illustration", ModelFamily::Sdxl, false);
        assert!(verbatim.is_empty(), "no style warning under --no-enhance: {verbatim:?}");
        // Style carried through → no warning.
        let ok = scene_warnings("s", &["cinematic photography".into()], "cinematic photography of a street", ModelFamily::Sdxl, true);
        assert!(ok.is_empty(), "clean: {ok:?}");
    }

    #[test]
    fn clean_collapses_commas_and_spaces() {
        assert_eq!(clean(", foo,,  bar ,"), "foo, bar");
        assert_eq!(clean("a, b, c"), "a, b, c");
        assert_eq!(clean(""), "");
    }

    #[test]
    fn assemble_joins_header_body_footer() {
        let s = scene("wide shot,", "a frozen tundra", "8k, photoreal", ModelFamily::Sdxl);
        assert_eq!(assemble_input(&s), "wide shot, a frozen tundra, 8k, photoreal");
    }

    #[test]
    fn positive_system_carries_family_and_styles() {
        let mut s = scene("", "x", "", ModelFamily::Flux);
        s.styles = vec!["impressionist".into()];
        let sys = positive_system(&s, None, &["a grizzled sea captain".to_string()]);
        assert!(sys.contains("Flux"));
        assert!(sys.contains("STYLE DIRECTION: impressionist"));
        assert!(sys.contains("PERSONA: a grizzled sea captain"));
    }

    #[test]
    fn negative_system_injects_seed_terms() {
        let mut s = scene("", "x", "", ModelFamily::Sdxl);
        s.negative_seeds = "blurry, watermark".into();
        let sys = negative_system(&s);
        assert!(sys.contains("MUST appear"));
        assert!(sys.contains("blurry, watermark"));
        // Flux negative profile differs.
        let mut f = scene("", "x", "", ModelFamily::Flux);
        f.negative_seeds = String::new();
        assert!(negative_system(&f).contains("largely ineffective"));
    }
}
