//! A1111-style inline `<lora:NAME[:weight]>` prompt tags.
//!
//! Civitai LoRA cards embed these directly in their example prompts;
//! before this module, users had to manually translate each tag into
//! a separate `--lora` CLI arg. Now plakat extracts the tags from
//! the prompt at the CLI boundary, parses each into a [`LoraSpec`]
//! (reusing the v0.17 grammar — local paths, HF repos, `civitai:NNN`,
//! `civitai-version:NNN`), and prepends them onto the user's
//! `--lora` stack. The cleaned prompt (tags removed) flows on to the
//! encoder.
//!
//! Grammar:
//!
//! ```text
//! <lora:NAME>                     # weight = 1.0
//! <lora:NAME:WEIGHT>              # explicit weight
//! ```
//!
//! Where `NAME[:WEIGHT]` is anything `LoraSpec::from_str` accepts:
//!
//! ```text
//! <lora:myfile.safetensors:0.7>
//! <lora:author/repo>
//! <lora:author/repo#file.safetensors:0.5>
//! <lora:civitai:12345>
//! <lora:civitai:12345#file.safetensors:0.8>
//! <lora:civitai-version:67890:0.6>
//! ```
//!
//! Ordering at the CLI:
//!
//! ```text
//! wildcards → lora-tags → attention syntax → encode
//! ```
//!
//! Wildcards expand first so `<lora:{styleA|styleB}>` resolves a
//! single LoRA name via the wildcard pick before this module sees
//! it. Attention syntax runs AFTER so the cleaned (no-tag) prompt
//! is what the per-token weight broadcast operates on.
//!
//! Unbalanced `<lora:` (no closing `>`) is treated as a literal —
//! same robustness contract `prompt::a1111::parse` uses. No
//! `ok_or_else` bail, no surprise error during a long batch.
//!
//! Negative prompts: `<lora:>` tags in `--negative` are a no-op
//! (extracted + dropped, but the resulting `LoraSpec`s aren't fed
//! back into the LoRA stack — A1111 strips them silently and so do
//! we). The cleaned negative is what reaches the encoder.

use anyhow::{Context, Result};
use std::ops::Range;

use crate::pipelines::lora::LoraSpec;

/// One LoRA reference extracted from a prompt. The `span` is the
/// byte range in the original prompt where the tag appeared —
/// preserved for diagnostics; not needed for the hot path.
#[derive(Debug, Clone)]
pub struct ExtractedLora {
    pub spec: LoraSpec,
    pub span: Range<usize>,
}

/// Cheap pre-check so callers can skip the parse on prompts without
/// any LoRA tags. Mirrors `a1111::has_attention_syntax`.
pub fn has_lora_tags(prompt: &str) -> bool {
    prompt.contains("<lora:")
}

/// Extract all `<lora:...>` tags from `prompt`. Returns
/// `(cleaned_prompt, specs)`:
/// - `cleaned_prompt`: original with each tag substring removed.
///   No spacing fix (A1111 behaviour — adjacent spaces are left
///   alone; the user's prompt formatter wins).
/// - `specs`: parsed [`ExtractedLora`] per tag, in source order.
///
/// An unparseable inner spec bails (with the offending substring +
/// the underlying parse error). An unbalanced `<lora:` with no
/// closing `>` is treated as a literal and skipped, NOT bailed —
/// matches A1111's permissive behaviour for copy-paste mistakes.
pub fn extract(prompt: &str) -> Result<(String, Vec<ExtractedLora>)> {
    let mut cleaned = String::with_capacity(prompt.len());
    let mut specs: Vec<ExtractedLora> = Vec::new();
    let mut cursor = 0;
    let bytes = prompt.as_bytes();

    while cursor < bytes.len() {
        // Look for the next `<lora:` literal from the cursor.
        let needle = b"<lora:";
        let start = (cursor..bytes.len())
            .find(|&i| bytes[i..].starts_with(needle));
        let start = match start {
            Some(s) => s,
            None => {
                // No more tags — append the rest verbatim.
                cleaned.push_str(&prompt[cursor..]);
                break;
            }
        };
        // Look for the matching closing `>`. We don't support
        // nested tags (A1111 doesn't either).
        let close_rel = bytes[start + needle.len()..]
            .iter()
            .position(|&b| b == b'>');
        let close = match close_rel {
            Some(rel) => start + needle.len() + rel,
            None => {
                // Unbalanced `<lora:` — treat as literal, append
                // through the rest of the prompt, stop searching.
                cleaned.push_str(&prompt[cursor..]);
                break;
            }
        };

        // Inner content between `<lora:` and `>`.
        let inner = &prompt[start + needle.len()..close];
        if inner.is_empty() {
            anyhow::bail!(
                "empty <lora:> tag at byte offset {start} — name is required"
            );
        }
        let spec: LoraSpec = inner.parse().with_context(|| {
            format!(
                "parsing <lora:{}> at byte offset {start}",
                inner
            )
        })?;
        if !(spec.scale.is_finite() && (0.0..=2.0).contains(&spec.scale)) {
            tracing::warn!(
                target: "plakat",
                "<lora:{inner}> weight {} is outside the usual [0, 2] band; \
                 keeping it but result may overcook",
                spec.scale
            );
        }

        // Append everything before the tag, drop the tag itself.
        cleaned.push_str(&prompt[cursor..start]);
        specs.push(ExtractedLora {
            spec,
            span: start..close + 1, // include the closing `>`
        });
        cursor = close + 1;
    }

    Ok((cleaned, specs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::lora::LoraSource;

    #[test]
    fn has_lora_tags_fast_path() {
        assert!(!has_lora_tags("plain prompt"));
        assert!(!has_lora_tags("a (red:1.5) fox")); // attention, not lora
        assert!(has_lora_tags("a fox <lora:foo>"));
        assert!(has_lora_tags("<lora:civitai:12345>"));
    }

    #[test]
    fn extract_single_tag_default_weight() {
        let (cleaned, specs) =
            extract("a fox <lora:myfile.safetensors>").unwrap();
        assert_eq!(cleaned, "a fox ");
        assert_eq!(specs.len(), 1);
        // default scale is 1.0 from LoraSpec::from_str.
        assert!((specs[0].spec.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn extract_single_tag_explicit_weight() {
        let (cleaned, specs) =
            extract("a fox <lora:myfile.safetensors:0.7> in grass").unwrap();
        assert_eq!(cleaned, "a fox  in grass");
        assert_eq!(specs.len(), 1);
        assert!((specs[0].spec.scale - 0.7).abs() < 1e-6);
    }

    #[test]
    fn extract_civitai_shorthand() {
        let (cleaned, specs) =
            extract("watercolor <lora:civitai:12345:0.8> style").unwrap();
        assert_eq!(cleaned, "watercolor  style");
        assert_eq!(specs.len(), 1);
        assert!((specs[0].spec.scale - 0.8).abs() < 1e-6);
        match &specs[0].spec.source {
            LoraSource::Civitai { .. } => {} // ok
            other => panic!("expected Civitai source, got {other:?}"),
        }
    }

    #[test]
    fn extract_multiple_tags() {
        let (cleaned, specs) = extract(
            "a fox <lora:style1:0.5> in grass <lora:style2:0.3>",
        )
        .unwrap();
        assert_eq!(cleaned, "a fox  in grass ");
        assert_eq!(specs.len(), 2);
        assert!((specs[0].spec.scale - 0.5).abs() < 1e-6);
        assert!((specs[1].spec.scale - 0.3).abs() < 1e-6);
    }

    #[test]
    fn extract_preserves_source_order() {
        let (_, specs) = extract(
            "<lora:civitai:111> mid <lora:civitai:222> end <lora:civitai:333>",
        )
        .unwrap();
        assert_eq!(specs.len(), 3);
        let ids: Vec<_> = specs
            .iter()
            .filter_map(|e| match &e.spec.source {
                LoraSource::Civitai {
                    id_kind: crate::pipelines::lora::CivitaiIdKind::Model(id),
                    ..
                } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![111u64, 222, 333]);
    }

    #[test]
    fn extract_no_tags_returns_prompt_verbatim() {
        let (cleaned, specs) = extract("a plain prompt with no tags").unwrap();
        assert_eq!(cleaned, "a plain prompt with no tags");
        assert!(specs.is_empty());
    }

    #[test]
    fn unbalanced_tag_is_treated_as_literal() {
        // No closing `>` — A1111 leaves it in the prompt rather
        // than bailing. Same here.
        let (cleaned, specs) = extract("a fox <lora:bad oops").unwrap();
        assert_eq!(cleaned, "a fox <lora:bad oops");
        assert!(specs.is_empty());
    }

    #[test]
    fn empty_inner_bails() {
        let err = extract("a fox <lora:>").unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[test]
    fn span_byte_offsets_are_correct() {
        let prompt = "a fox <lora:civitai:12345:0.7> in grass";
        let (_, specs) = extract(prompt).unwrap();
        assert_eq!(specs.len(), 1);
        // Tag starts at byte 6 ("a fox " is 6 bytes), spans through
        // the closing `>`.
        assert_eq!(specs[0].span.start, 6);
        assert_eq!(&prompt[specs[0].span.clone()], "<lora:civitai:12345:0.7>");
    }

    #[test]
    fn tag_at_start_of_prompt() {
        let (cleaned, specs) =
            extract("<lora:foo:0.5> a fox").unwrap();
        assert_eq!(cleaned, " a fox");
        assert_eq!(specs.len(), 1);
    }

    #[test]
    fn tag_at_end_of_prompt() {
        let (cleaned, specs) = extract("a fox <lora:foo:0.5>").unwrap();
        assert_eq!(cleaned, "a fox ");
        assert_eq!(specs.len(), 1);
    }
}
