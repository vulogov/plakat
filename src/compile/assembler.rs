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
- PRESERVE attention-weight syntax EXACTLY: `(phrase:1.5)`, `(phrase)`, `[phrase]`. Keep the \
parentheses/brackets and the number unchanged — if the phrase is in another language, translate the \
words INSIDE but keep the weight wrapper (e.g. `(Оранжевое солнце:1.5)` → `(orange sun:1.5)`). These \
are deliberate emphasis controls, not prose — never drop or flatten them.\n\
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

const FAMILY_SD3: &str = "\
TARGET MODEL: Stable Diffusion 3 / 3.5 (T5-XXL + dual CLIP, ~256-token range).\n\
- Write natural-language prose describing the scene, NOT comma-separated booru tags. Aim 80-220 tokens.\n\
- Do NOT use SD1.5 quality boosters (masterpiece, best quality, sharp focus — the T5 encoder ignores them).\n\
- Be specific and concrete about composition and spatial layout (what is where), materials, and lighting; a \
short style phrase up front is fine, then describe the scene as flowing prose.";

fn family_section(f: ModelFamily) -> &'static str {
    match f {
        ModelFamily::Sd15 | ModelFamily::Unknown => FAMILY_SD15,
        ModelFamily::Sdxl => FAMILY_SDXL,
        ModelFamily::Sd3 => FAMILY_SD3,
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
        // SD3/3.5 route the full prompt through T5-XXL (256-token context) alongside the CLIP pair.
        ModelFamily::Sd3 => 256,
        ModelFamily::Flux => 300,
    }
}

/// Rough CLIP/T5 token estimate of a prompt (a warning heuristic, not exact): whitespace- and
/// comma-separated pieces scaled by ~1.3 for sub-word splitting. Deterministic.
pub fn estimate_tokens(text: &str) -> usize {
    let words = text.split(|c: char| c.is_whitespace() || c == ',').filter(|s| !s.is_empty()).count();
    (words * 4).div_ceil(3)
}

/// Count attention-weight spans `(…:number)` in a prompt — used to warn when the enhancer drops the
/// user's deliberate emphasis (`(term:1.5)`) during translate/rewrite.
pub fn weight_span_count(text: &str) -> usize {
    let mut n = 0;
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut open_at = 0usize;
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'(' => {
                if depth == 0 {
                    open_at = i;
                }
                depth += 1;
            }
            b')' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    // A top-level `(...)` group with a `:<digit>` inside is a weighted span.
                    let inner = &text[open_at + 1..i];
                    if let Some(colon) = inner.rfind(':') {
                        if inner[colon + 1..].trim().chars().next().is_some_and(|d| d.is_ascii_digit()) {
                            n += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    n
}

/// Extract every top-level attention-weight span `(phrase:N)` as `(phrase, weight)`, in order.
/// Mirrors [`weight_span_count`]'s scan but returns the inner phrase (trimmed) and parsed weight, so the
/// compiler can re-inject the user's deliberate emphasis deterministically after the enhancer (which may
/// flatten it). Only explicit `:number` spans are returned — bare `(...)` prose parentheticals are ignored.
pub fn extract_weight_spans(text: &str) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut open_at = 0usize;
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'(' => {
                if depth == 0 {
                    open_at = i;
                }
                depth += 1;
            }
            b')' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    let inner = &text[open_at + 1..i];
                    if let Some(colon) = inner.rfind(':') {
                        let (phrase, w) = (inner[..colon].trim(), inner[colon + 1..].trim());
                        if let Ok(weight) = w.parse::<f32>() {
                            if !phrase.is_empty() {
                                out.push((phrase.to_string(), weight));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Rewrite every top-level attention-weight span `(phrase:N)` by replacing it with `f(phrase, weight)`.
/// Bare `(...)` parentheticals (no trailing `:number`) are left untouched. The parens/colon/number are
/// ASCII, so the slices are always on char boundaries even with non-ASCII phrases.
pub fn rewrite_weight_spans(text: &str, mut f: impl FnMut(&str, f32) -> String) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut depth = 0i32;
    let mut open_at = 0usize;
    let mut copied = 0usize; // byte index in `text` up to which we've emitted into `out`
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'(' => {
                if depth == 0 {
                    open_at = i;
                }
                depth += 1;
            }
            b')' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    let inner = &text[open_at + 1..i];
                    if let Some(colon) = inner.rfind(':') {
                        if let Ok(w) = inner[colon + 1..].trim().parse::<f32>() {
                            let phrase = inner[..colon].trim();
                            if !phrase.is_empty() {
                                out.push_str(&text[copied..open_at]);
                                out.push_str(&f(phrase, w));
                                copied = i + 1;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out.push_str(&text[copied..]);
    out
}

/// Remove the attention-weight wrapper from every top-level `(phrase:N)` span, keeping the inner phrase:
/// `roofs (голубого цвета:1.5)` → `roofs голубого цвета`.
pub fn strip_weight_spans(text: &str) -> String {
    rewrite_weight_spans(text, |phrase, _| phrase.to_string())
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
            "scene '{name}': the STYLE you asked for ('{asked}') did not appear in the prompt — the enhancer may have dropped it (try a different --compile-provider; note --no-enhance keeps your wording but also skips translation)"
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

/// A curated, GENERIC quality negative — terms that reliably reduce artifacts (resolution, anatomy,
/// framing, watermarks) WITHOUT excluding scene content. We never invent scene-specific / content
/// exclusions (an LLM asked to write a negative hallucinates "weapons, blood, dark mood…" and can even
/// run away into repetition); the user's explicit `negative:` seeds carry any content intent.
pub const QUALITY_NEGATIVE: &str = "lowres, low quality, worst quality, jpeg artifacts, blurry, \
out of focus, bad anatomy, deformed, disfigured, mutated, extra limbs, extra fingers, fused fingers, \
missing fingers, poorly drawn hands, poorly drawn face, bad proportions, duplicate, cropped, \
out of frame, watermark, signature, text, logo";

/// Deterministic auto-negative: the user's `negative:` seeds (verbatim, first) plus the curated
/// [`QUALITY_NEGATIVE`] — deduped case-insensitively (first occurrence wins, order preserved) and capped
/// so it can never run away. Flux ignores negatives (guidance distillation), so it gets seeds only.
pub fn auto_negative(scene: &ResolvedScene) -> String {
    if matches!(scene.family, ModelFamily::Flux) {
        return scene.negative_seeds.trim().to_string();
    }
    merge_negative_terms(&[&scene.negative_seeds, QUALITY_NEGATIVE], 40)
}

/// Merge comma-separated negative term-lists in order, dedup case-insensitively (keep first), cap to
/// `max_terms`. The cap is the runaway guard.
pub fn merge_negative_terms(parts: &[&str], max_terms: usize) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for part in parts {
        for term in part.split(',') {
            let t = term.trim();
            if t.is_empty() {
                continue;
            }
            if seen.insert(t.to_lowercase()) {
                out.push(t.to_string());
                if out.len() >= max_terms {
                    return out.join(", ");
                }
            }
        }
    }
    out.join(", ")
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
    fn rewrite_weight_spans_substitutes_inline() {
        // Translate-in-place: replace the phrase, keep the weight + position.
        let map = |p: &str, w: f32| {
            let en = match p {
                "голубого цвета" => "blue",
                other => other,
            };
            format!("({en}:{w})")
        };
        assert_eq!(
            rewrite_weight_spans("Крыши (голубого цвета:1.5) tiled", map),
            "Крыши (blue:1.5) tiled"
        );
        // Bare parentheticals untouched; multiple spans handled.
        assert_eq!(
            rewrite_weight_spans("(a:1) and (b:2), a (note)", |p, w| format!("[{p}={w}]")),
            "[a=1] and [b=2], a (note)"
        );
    }

    #[test]
    fn strip_weight_spans_keeps_phrase_drops_wrapper() {
        assert_eq!(
            strip_weight_spans("roofs (blue tile:1.5) and (orange sun:2) here"),
            "roofs blue tile and orange sun here"
        );
        // Non-ASCII inner phrases slice safely; bare parentheticals are left intact.
        assert_eq!(strip_weight_spans("Крыши (голубого цвета:1.5)"), "Крыши голубого цвета");
        assert_eq!(strip_weight_spans("a (plain) note"), "a (plain) note");
        // Round-trips with extract: stripped text has no weighted spans left.
        assert_eq!(weight_span_count(&strip_weight_spans("(a:1.5), (b:2), plain")), 0);
    }

    #[test]
    fn extract_weight_spans_returns_phrase_and_weight() {
        let spans = extract_weight_spans("(cobblestone street:1.5), plain, (green sky: 1.4) and (orange sun:2)");
        assert_eq!(
            spans,
            vec![
                ("cobblestone street".to_string(), 1.5),
                ("green sky".to_string(), 1.4),
                ("orange sun".to_string(), 2.0),
            ]
        );
        // Bare parentheticals / brackets are ignored (only explicit `:number`).
        assert!(extract_weight_spans("a (parenthetical) aside, [de-emphasis]").is_empty());
    }

    #[test]
    fn weight_span_count_counts_weighted_parens() {
        assert_eq!(weight_span_count("(cobblestone street:1.5), (orange sun:1.5), plain text"), 2);
        assert_eq!(weight_span_count("(a green sky: 1.4) and (tall floor:1.5)"), 2);
        // Non-weighted parens / brackets aren't counted.
        assert_eq!(weight_span_count("a (parenthetical) note, [deemphasis], plain"), 0);
        assert_eq!(weight_span_count("no weights here at all"), 0);
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
    fn auto_negative_is_deterministic_seeds_plus_quality() {
        let mut s = scene("", "x", "", ModelFamily::Sdxl);
        s.negative_seeds = "blurry, watermark".into();
        let neg = auto_negative(&s);
        // Seeds come first (verbatim), quality terms follow.
        assert!(neg.starts_with("blurry, watermark"), "seeds first: {neg}");
        assert!(neg.contains("bad anatomy") && neg.contains("lowres"), "quality terms present");
        // Deduped: 'blurry' (a seed AND a quality term) appears once.
        assert_eq!(neg.matches("blurry").count(), 1, "deduped: {neg}");
        // Capped — never a runaway.
        assert!(neg.split(',').count() <= 40);
        // Flux gets seeds only (guidance distillation ignores negatives).
        let mut f = scene("", "x", "", ModelFamily::Flux);
        f.negative_seeds = "blurry, watermark".into();
        assert_eq!(auto_negative(&f), "blurry, watermark");
    }

    #[test]
    fn merge_negative_terms_caps_a_runaway() {
        // A degenerate repeated list collapses to its unique terms, capped.
        let runaway = "wrong, wrong, wrong, all, none, all, none, all";
        assert_eq!(merge_negative_terms(&[runaway], 40), "wrong, all, none");
        assert_eq!(merge_negative_terms(&["a, b, c, d, e"], 3), "a, b, c");
    }
}
