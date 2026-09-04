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
You are an expert prompt engineer for text-to-image diffusion models. The user has written their \
scene description as best they can; YOUR JOB is to turn it into the single prompt that produces the \
BEST POSSIBLE IMAGE on the SPECIFIC target model described below — apply your expert knowledge of that \
model's strengths, quirks, and what it actually responds to. Honour the user's intent (their subjects, \
their style, their key elements); optimise everything else — wording, order, technical phrasing — for \
image quality. Output ONLY the prompt — no preamble, no quotation marks, no markdown, no explanation.\n\
Rules (an expert respects the user's intent and never makes the result worse):\n\
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
- PRESERVE spatial and contact RELATIONSHIPS between objects exactly as described — what rests ON, is \
HELD BY, is ATTACHED TO, is IN FRONT OF / BEHIND / NEXT TO / UNDER what — and render them as physically \
coherent: objects touch, rest, and connect as stated (a vehicle sits ON its track; a basket is HELD in \
the hand; a lamp is BESIDE, not merged into, the foliage). The source may be in ANY language — keep the \
relationship intact through translation. Do NOT invent relationships the user did not state.\n\
- DO NO HARM — enhancement must never make the image HARDER to render well than the user's own wording. \
Do NOT add new subjects, people, animals, objects, or actions the user did not state. Do NOT invent \
unrelated elements. Do NOT pad with atmospheric or mood filler ('bathed in golden light', 'a dreamlike \
glow', 'bustling with life', 'casting warm light across the scene') — such phrases add tokens and dilute \
the model's attention without adding a concrete subject, degrading quality. Keep the scene's complexity \
exactly as the user set it (never more crowded). Prefer the FEWEST words that convey the scene faithfully \
for the target model: clarify wording, fix word order, and translate — but never inflate.";

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

const FAMILY_CASCADE: &str = "\
TARGET MODEL: Stable Cascade (Würstchen v3, CLIP text encoders).\n\
- Descriptive natural language with keyword clusters; aim 40-120 tokens. Front-load subject and style.\n\
- Attention-weight `(term:N)` syntax is NOT honoured by this model — rely on clear description, concrete \
detail and word order for emphasis, never on weights.\n\
- Order: [style], [subject], [composition], [lighting], [mood].";

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
        ModelFamily::Cascade => FAMILY_CASCADE,
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
        // Cascade's CLIP text encoders take a moderate prompt — more forgiving than SD15's hard 77.
        ModelFamily::Cascade => 120,
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

/// Lowercase the first character of a phrase for mid-sentence use, UNLESS it looks like a proper noun /
/// acronym (word 2+ chars, all-uppercase, e.g. "T5"). Keeps a translated `(High first floor:1.5)` from
/// reading as a broken mid-sentence capital in the reinforcement list.
fn lower_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_uppercase() => {
            // Don't lowercase an all-caps acronym (first word ≥2 chars, every letter uppercase).
            let first_word = s.split_whitespace().next().unwrap_or("");
            let is_acronym = first_word.len() >= 2 && first_word.chars().all(|c| c.is_uppercase() || !c.is_alphabetic());
            if is_acronym {
                s.to_string()
            } else {
                c.to_lowercase().collect::<String>() + chars.as_str()
            }
        }
        _ => s.to_string(),
    }
}

/// Join phrases as a readable English list: `a`, `a and b`, `a, b and c`.
fn join_and(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [a] => a.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

/// **SD3/Flux prose reinforcement** (6.27): the T5-driven families honour descriptive prose far more than
/// numeric `(term:N)` weights, so a heavily-weighted concept often still loses to strong priors. Restate the
/// weighted concepts as a short prose intensifier clause (bucketed by weight) that we append to the prompt —
/// the inline `(term:N)` weights stay for the CLIP encoders, and this makes the emphasis actually land on
/// T5. Returns the clause to append, or None for non-T5 families / no weighted spans. Dedups by phrase.
pub fn prose_reinforcement(prompt: &str, family: ModelFamily) -> Option<String> {
    // SD3/Flux honour prose >> weights; Cascade honours NO numeric weights at all — so all three get the
    // prose restatement (for Cascade it's the ONLY way to emphasise, since `(term:N)` does nothing there).
    if !matches!(family, ModelFamily::Sd3 | ModelFamily::Flux | ModelFamily::Cascade) {
        return None;
    }
    let (mut strong, mut moderate, mut faint) = (Vec::new(), Vec::new(), Vec::new());
    let mut seen = std::collections::HashSet::new();
    for (phrase, w) in extract_weight_spans(prompt) {
        let p = lower_first(phrase.trim().trim_end_matches(['.', ',']));
        if p.is_empty() || !seen.insert(p.to_lowercase()) {
            continue;
        }
        if w >= 1.5 {
            strong.push(p);
        } else if w >= 1.15 {
            moderate.push(p);
        } else if w < 0.9 {
            faint.push(p);
        }
        // weights in [0.9, 1.15) are ~neutral — no prose nudge.
    }
    let mut parts = Vec::new();
    if !strong.is_empty() {
        parts.push(format!("Prominent and clearly visible in the scene: {}.", join_and(&strong)));
    }
    if !moderate.is_empty() {
        parts.push(format!("Clearly present: {}.", join_and(&moderate)));
    }
    if !faint.is_empty() {
        parts.push(format!("Only subtle and understated: {}.", join_and(&faint)));
    }
    (!parts.is_empty()).then(|| parts.join(" "))
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

/// System prompt for the HYBRID negative: the LLM adds a FEW scene-specific DEFECT terms only. Strict
/// rules keep it from the old failure modes (runaway, negating content the user wants). The output is
/// still capped/deduped downstream, and any term echoing the positive prompt is stripped as a hard guard.
pub fn negative_scene_system() -> &'static str {
    "You add a FEW extra NEGATIVE-prompt terms for a text-to-image model, tailored to ONE scene. Given the \
     POSITIVE prompt, output a SHORT comma-separated list — AT MOST 10 terms — of the QUALITY / ANATOMY / \
     TECHNICAL DEFECT terms most worth suppressing FOR THIS SCENE. Examples: a crowd → 'cloned faces, \
     duplicate people, merged bodies'; visible hands → 'extra fingers, fused fingers'; architecture → \
     'warped perspective, crooked walls'; a vehicle/machine → 'melted metal, asymmetric wheels'. STRICT \
     RULES: (1) ONLY defects and rendering artifacts — NEVER exclude any content, colour, subject, mood, \
     style, medium, or setting the scene contains or wants (never negate the sky, the sun, colours, the \
     people, the medium). (2) Do NOT repeat generic terms like 'blurry, low quality, watermark' — those are \
     added separately. (3) At most 10 terms, no repetition. Output ONLY the comma-separated terms, nothing \
     else."
}

/// Hard guard for the hybrid negative: drop any negative term whose text appears in the POSITIVE prompt, so
/// an LLM suggestion can never suppress content the user actually asked for (e.g. 'green sky' / 'orange sun').
pub fn strip_terms_in_positive(neg_terms: &str, positive: &str) -> String {
    let pl = positive.to_lowercase();
    neg_terms
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty() && !pl.contains(&t.to_lowercase()))
        .collect::<Vec<_>>()
        .join(", ")
}

// ─────────────────────────── weight-free relationship grounding (6.28) ───────────────────────────
// A generic pass: when a POSITIVE prompt describes how objects relate (one resting on / held by / in
// front of another), push against the universal failure class (things float, detach, or merge instead
// of touching as described) — object-agnostic, never scene-specific. Improves "tram on rails", "lamp not
// merging into foliage", "basket held by the hand" alike, from the prompt, with no model and no sketch.

/// Spatial/contact **relationship markers**. Curated to be specific enough to avoid firing on incidental
/// prepositions ("on a sunny day"): mostly multi-word contact/placement cues + spatial arrangements.
/// Case-insensitive substring match. English (this runs after `translate:`).
pub const RELATIONSHIP_MARKERS: &[&str] = &[
    // contact / support
    "on top of", "stands on", "standing on", "sits on", "sitting on", "seated on", "rests on",
    "resting on", "lying on", "placed on", "set on", "mounted on", "perched on", "atop",
    "attached to", "connected to", "fastened to", "tied to", "coupled to", "leaning against",
    "leaning on", "hanging from", "held by", "holding", "carrying", "riding",
    // spatial arrangement
    "in front of", "next to", "beside", "underneath", "on either side of", "at the edge of",
    "in the middle of", "surrounded by",
];

/// Generic **relationship-violation** negatives — the universal failure class when a scene describes
/// objects in relation. Object-agnostic by design: "not floating / not detached", never "tram off rails".
pub const RELATIONSHIP_NEGATIVE: &str = "floating objects, disconnected, detached, hovering, levitating, \
merging objects, fused together, clipping through, incoherent placement, misaligned, not touching, \
overlapping incorrectly";

/// Whether a POSITIVE prompt describes any object-to-object spatial/contact relationship. Drives the
/// weight-free relationship pass. Check this on the ORIGINAL prompt before appending the grounding clause
/// (the clause itself contains markers).
pub fn has_relationships(positive: &str) -> bool {
    let p = positive.to_lowercase();
    RELATIONSHIP_MARKERS.iter().any(|m| p.contains(m))
}

/// A short, generic positive clause reinforcing coherent object placement — for prose families (SD3 /
/// Flux / Cascade, which read natural language and have the budget). Object-agnostic; DO NO HARM (one
/// sentence, no filler). `None` for weight-capable families (SD1.5/SDXL) where the tight budget makes the
/// negative side alone the right call.
pub fn relationship_reinforcement(family: ModelFamily) -> Option<&'static str> {
    // Purely AFFIRMATIVE — no "not floating" negation (models mishandle negation in a positive prompt,
    // and it would collide with the strip-terms-in-positive guard). The violations live in the negative.
    matches!(family, ModelFamily::Sd3 | ModelFamily::Flux | ModelFamily::Cascade).then_some(
        "The described objects sit in clear, physically coherent spatial relationships — touching, \
         resting on, and connected exactly as stated, each correctly placed and firmly grounded.",
    )
}

/// Map a `relate:` verb to an English relationship phrase for the grounding clause. Small, extensible
/// vocabulary; an unknown verb passes through literally (`the A <verb> the B`), so the directive never
/// breaks on a word we didn't anticipate. Hyphens are normalized to spaces (`in-front-of` == `in front of`).
pub fn relation_phrase(verb: &str) -> String {
    match verb.trim().to_lowercase().replace('-', " ").as_str() {
        "on" | "on top of" | "onto" | "atop" | "sitting on" | "standing on" | "resting on" => "rests on".into(),
        "under" | "underneath" | "beneath" | "below" => "is beneath".into(),
        "above" | "over" => "is above".into(),
        "in front of" | "before" => "is in front of".into(),
        "behind" => "is behind".into(),
        "next to" | "beside" | "by" => "stands beside".into(),
        "near" => "is near".into(),
        "holding" | "holds" | "carrying" => "holds".into(),
        "held by" => "is held by".into(),
        "attached to" | "coupled to" | "connected to" | "fastened to" | "tied to" => "is coupled to".into(),
        "inside" | "in" | "within" => "is inside".into(),
        "leaning on" | "leaning against" | "against" => "leans against".into(),
        "riding" | "rides" => "rides on".into(),
        "surrounded by" => "is surrounded by".into(),
        other => other.to_string(),
    }
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
    fn prose_reinforcement_buckets_by_weight_for_t5_families() {
        let p = "a street, (orange sun:1.5), (green sky:1.3), (faint smoke:0.8), (cobblestone:1.5)";
        let r = prose_reinforcement(p, ModelFamily::Sd3).unwrap();
        // Strong (>=1.5): orange sun + cobblestone, as a natural list.
        assert!(r.contains("Prominent and clearly visible in the scene: orange sun and cobblestone."), "got: {r}");
        // Moderate (1.15–1.5): green sky.
        assert!(r.contains("Clearly present: green sky."), "got: {r}");
        // De-emphasis (<0.9): faint smoke.
        assert!(r.contains("Only subtle and understated: faint smoke."), "got: {r}");
        // Cascade also reinforces (it honours NO numeric weights, so prose is the only lever).
        assert!(prose_reinforcement(p, ModelFamily::Cascade).is_some());
        // Weight-honouring CLIP families get nothing; no weights → nothing.
        assert!(prose_reinforcement(p, ModelFamily::Sdxl).is_none());
        assert!(prose_reinforcement("no weights here", ModelFamily::Flux).is_none());
        // Source-cased phrases are lowercased for mid-sentence use; acronyms are kept.
        let r2 = prose_reinforcement("(High first floor:1.5), (T5 label:1.5)", ModelFamily::Sd3).unwrap();
        assert!(r2.contains("high first floor"), "lowercased: {r2}");
        assert!(r2.contains("T5 label"), "acronym kept: {r2}");
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
    fn relationship_pass_fires_on_relations_not_incidental_prepositions() {
        // Genuine object relationships fire.
        assert!(has_relationships("a tram standing on the rails beside the pavilion"));
        assert!(has_relationships("a woman holding a basket in front of the shop"));
        assert!(has_relationships("a lamp mounted on a post"));
        // Incidental prepositions / temporal "on" do NOT fire.
        assert!(!has_relationships("a quiet street on a sunny day, muted palette"));
        assert!(!has_relationships("an impressionist painting of a harbor at dawn"));
        // Prose families get a grounding clause; weight-capable families get the negative side only.
        assert!(relationship_reinforcement(ModelFamily::Sd3).is_some());
        assert!(relationship_reinforcement(ModelFamily::Flux).is_some());
        assert!(relationship_reinforcement(ModelFamily::Sdxl).is_none());
        assert!(relationship_reinforcement(ModelFamily::Sd15).is_none());
        // The generic negatives merge + dedup cleanly on top of an existing negative.
        let merged = merge_negative_terms(&["blurry, floating objects", RELATIONSHIP_NEGATIVE], 48);
        assert!(merged.contains("detached") && merged.contains("merging objects"));
        assert_eq!(merged.matches("floating objects").count(), 1, "deduped: {merged}");
        // relate: verb vocabulary — known verbs map to phrases; hyphen == space; unknown passes through.
        assert_eq!(relation_phrase("on"), "rests on");
        assert_eq!(relation_phrase("in-front-of"), "is in front of");
        assert_eq!(relation_phrase("next to"), "stands beside");
        assert_eq!(relation_phrase("held by"), "is held by");
        assert_eq!(relation_phrase("dangling above"), "dangling above"); // unknown → literal
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
    fn strip_terms_in_positive_protects_wanted_content() {
        // The hard guard: an LLM negative can never suppress content the positive asked for.
        let positive = "a street with a bright green sky and an orange sun, five children";
        let neg = "cloned faces, green sky, duplicate people, orange sun, extra fingers";
        // 'green sky' and 'orange sun' echo the positive → dropped; the real defects stay.
        assert_eq!(
            strip_terms_in_positive(neg, positive),
            "cloned faces, duplicate people, extra fingers"
        );
    }

    #[test]
    fn merge_negative_terms_caps_a_runaway() {
        // A degenerate repeated list collapses to its unique terms, capped.
        let runaway = "wrong, wrong, wrong, all, none, all, none, all";
        assert_eq!(merge_negative_terms(&[runaway], 40), "wrong, all, none");
        assert_eq!(merge_negative_terms(&["a, b, c, d, e"], 3), "a, b, c");
    }
}
