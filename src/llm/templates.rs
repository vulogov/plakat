//! Chat-template formatting + output sanitizer for the local
//! prompt enhancer.
//!
//! Each model family has its own role-marker convention. Qwen2 uses
//! ChatML (`<|im_start|>` / `<|im_end|>`). SmolLM2 uses Llama 3-style
//! special tokens. We format the prompt at module boundary and
//! sanitize the raw model output back into a plain-text rewritten
//! prompt before handing it to the diffusion encoder.

use crate::llm::aliases::Family;

/// Build the formatted chat string for the given family. Includes
/// the trailing "assistant\n" role marker so the model starts
/// generating the rewritten prompt immediately.
pub fn format(family: Family, system: &str, user: &str) -> String {
    match family {
        Family::Qwen2 => format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
        Family::Llama => format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
    }
}

/// Sanitize the raw decoder output back into a plain-text enhanced
/// prompt. Strips role tags, surrounding quotes, and a handful of
/// known refusal prefixes. Empty / refusal output returns `None`
/// so the caller can fall back to the user's original prompt
/// rather than feeding a refusal into the diffusion encoder.
pub fn sanitize(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_end_matches("<|im_end|>")
        .trim_end_matches("<|endoftext|>")
        .trim_end_matches("<|eot_id|>")
        .trim();
    // Strip surrounding quotes (some models like to "wrap their
    // answer in quotes" despite the system prompt telling them not
    // to). Both straight ASCII and the curly variants.
    let trimmed = trimmed
        .trim_start_matches(|c: char| c == '"' || c == '\'' || c == '\u{201C}')
        .trim_end_matches(|c: char| c == '"' || c == '\'' || c == '\u{201D}')
        .trim();
    if trimmed.is_empty() {
        return None;
    }
    // Reject common refusal prefixes — feeding a refusal into the
    // diffusion encoder pollutes the output.
    const REFUSAL_PREFIXES: &[&str] = &[
        "i cannot",
        "i can't",
        "i'm sorry",
        "i am sorry",
        "i am unable",
        "i'm unable",
        "as an ai",
        "i don't feel comfortable",
        "i won't",
    ];
    let lower = trimmed.to_lowercase();
    for prefix in REFUSAL_PREFIXES {
        if lower.starts_with(prefix) {
            return None;
        }
    }
    // Strip a "Here's the rewritten prompt:" / "Rewritten prompt:"
    // preamble if the model decided to add one despite the system
    // prompt's "no preamble" instruction.
    const PREAMBLE_PREFIXES: &[&str] = &[
        "here's the rewritten prompt:",
        "here is the rewritten prompt:",
        "rewritten prompt:",
        "rewritten:",
        "here's a rewritten version:",
        "sure,",
        "sure!",
        "here you go:",
    ];
    let mut result = trimmed.to_string();
    for prefix in PREAMBLE_PREFIXES {
        if result.to_lowercase().starts_with(prefix) {
            result = result[prefix.len()..].trim().to_string();
            break;
        }
    }
    // After preamble strip, re-trim quotes (some models do
    // "Rewritten: \"<actual prompt>\"").
    let result = result
        .trim_start_matches(|c: char| c == '"' || c == '\'' || c == '\u{201C}')
        .trim_end_matches(|c: char| c == '"' || c == '\'' || c == '\u{201D}')
        .trim()
        .to_string();
    if result.is_empty() { None } else { Some(result) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_qwen2_round_trips() {
        let out = format(Family::Qwen2, "be terse", "a fox in grass");
        assert!(out.starts_with("<|im_start|>system\nbe terse<|im_end|>"));
        assert!(out.contains("<|im_start|>user\na fox in grass<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn sanitize_strips_im_end_tag() {
        let out = sanitize("a (watercolor:1.3) fox<|im_end|>");
        assert_eq!(out.as_deref(), Some("a (watercolor:1.3) fox"));
    }

    #[test]
    fn sanitize_strips_surrounding_quotes() {
        let out = sanitize("\"a watercolor fox\"");
        assert_eq!(out.as_deref(), Some("a watercolor fox"));
        let out = sanitize("'a fox'");
        assert_eq!(out.as_deref(), Some("a fox"));
    }

    #[test]
    fn sanitize_strips_preamble() {
        let cases = [
            "Here's the rewritten prompt: a watercolor fox",
            "Rewritten: a watercolor fox",
            "Sure! a watercolor fox",
            "Here you go: a watercolor fox",
        ];
        for c in cases {
            assert_eq!(sanitize(c).as_deref(), Some("a watercolor fox"));
        }
    }

    #[test]
    fn sanitize_preamble_then_quotes() {
        // Combined preamble + surrounding-quotes pattern that some
        // models love.
        let out = sanitize("Rewritten prompt: \"a watercolor fox\"");
        assert_eq!(out.as_deref(), Some("a watercolor fox"));
    }

    #[test]
    fn sanitize_rejects_refusals() {
        assert!(sanitize("I cannot help with that request.").is_none());
        assert!(sanitize("I'm sorry, but I can't.").is_none());
        assert!(sanitize("As an AI, I don't generate prompts.").is_none());
    }

    #[test]
    fn sanitize_rejects_empty() {
        assert!(sanitize("").is_none());
        assert!(sanitize("   ").is_none());
        assert!(sanitize("<|im_end|>").is_none());
    }

    #[test]
    fn sanitize_passes_clean_output() {
        let out = sanitize("a brutalist whale poster, watercolor, dramatic light");
        assert_eq!(
            out.as_deref(),
            Some("a brutalist whale poster, watercolor, dramatic light")
        );
    }
}
