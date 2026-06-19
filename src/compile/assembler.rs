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

/// The positive *user* text from an explicit body (possibly a translation):
/// header, body, footer — comma-joined and cleaned.
pub fn assemble_with_body(scene: &ResolvedScene, body: &str) -> String {
    let joined = [scene.header.as_str(), body.trim(), scene.footer.as_str()].join(", ");
    clean(&joined)
}

/// The positive *user* text from the scene's own free text. Verbatim prompt
/// under `--no-enhance`, LLM input otherwise.
pub fn assemble_input(scene: &ResolvedScene) -> String {
    assemble_with_body(scene, scene.free_text.trim())
}

const POSITIVE_BASE: &str = "\
You are an expert prompt engineer for text-to-image diffusion models. Rewrite the \
user's scene description into a single optimized image-generation prompt. Output \
ONLY the prompt — no preamble, no quotation marks, no markdown, no explanation. \
Preserve the user's subject and intent; do not invent unrelated elements.";

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
