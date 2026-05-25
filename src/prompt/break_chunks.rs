//! A1111 `BREAK` keyword — chunk prompts past CLIP's 77-token cap.
//!
//! SD-family CLIP encoders truncate at 77 tokens. A long prompt
//! gets clipped silently after ~70 words. The A1111 convention:
//! the literal word `BREAK` (case-sensitive, word-boundary delimited)
//! splits the prompt into chunks; each chunk is tokenized + encoded
//! into its own 77-token CLIP context; the resulting hidden states
//! are sequence-concatenated before the UNet's cross-attention
//! consumes them.
//!
//! ```text
//! "a brutalist whale poster BREAK watercolor on rough paper"
//!   → ["a brutalist whale poster", "watercolor on rough paper"]
//!   → each chunk: encode_with_attention → (1, 77, 768)
//!   → concat along seq dim → (1, 154, 768)
//! ```
//!
//! Cross-attention has no max sequence length — the K/V tensors
//! just get longer, and every UNet query attends to every text
//! token. The CFG branch (negative prompt) is chunked independently
//! and padded with empty chunks if its chunk count is smaller so
//! cond and uncond reach the same total length.
//!
//! Scope:
//!   * SD 1.5 / 2.1 — single CLIP, single-encoder chunk path.
//!   * SDXL — dual CLIP (L + G); both encoders chunk independently;
//!     pooled output comes from chunk 0 only (A1111 convention).
//!   * Flux / SD3 — T5 has a 256 / 512-token budget; BREAK is moot.
//!     Strip + warn at the encoder dispatch.
//!
//! Word-boundary matching is conservative: `BREAKING`, `BREAKDOWN`,
//! `BREAKERS_v1` (a hypothetical LoRA name) all stay intact as
//! single words. Only the bare token `BREAK` surrounded by
//! non-alphanumeric characters (or string boundaries) triggers a
//! split.

/// The A1111 keyword. Case-sensitive — `Break` / `break` are NOT
/// split (mirrors A1111's `\bBREAK\b` regex).
pub const BREAK_KEYWORD: &str = "BREAK";

/// Cheap pre-check for the hot path. `false` lets every prompt
/// without BREAK skip the chunk-split + per-chunk encode overhead.
pub fn has_break(prompt: &str) -> bool {
    word_bounded_positions(prompt).next().is_some()
}

/// Split `prompt` on word-bounded BREAK keywords. Returns the
/// individual chunks (trimmed of surrounding whitespace). A prompt
/// without BREAK returns a single-chunk vec containing the input
/// verbatim — the caller can iterate uniformly.
///
/// Empty chunks (e.g. `"prompt BREAK BREAK other"`) are dropped —
/// matches A1111.
pub fn split(prompt: &str) -> Vec<String> {
    let positions: Vec<usize> = word_bounded_positions(prompt).collect();
    if positions.is_empty() {
        return vec![prompt.to_string()];
    }
    let mut chunks: Vec<String> = Vec::with_capacity(positions.len() + 1);
    let mut last = 0;
    for &pos in &positions {
        let chunk = prompt[last..pos].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        last = pos + BREAK_KEYWORD.len();
    }
    let tail = prompt[last..].trim();
    if !tail.is_empty() {
        chunks.push(tail.to_string());
    }
    if chunks.is_empty() {
        // All chunks were empty (e.g. "BREAK BREAK") — return one
        // empty string so the caller's "encode each chunk" loop
        // still produces a valid (empty) sequence.
        vec![String::new()]
    } else {
        chunks
    }
}

/// Strip BREAK keywords from `prompt` without splitting. Used by
/// the Flux / SD3 encoder dispatch where T5's wider token budget
/// makes BREAK a no-op (with a warn to the user).
pub fn strip(prompt: &str) -> String {
    let positions: Vec<usize> = word_bounded_positions(prompt).collect();
    if positions.is_empty() {
        return prompt.to_string();
    }
    let mut out = String::with_capacity(prompt.len());
    let mut last = 0;
    for &pos in &positions {
        out.push_str(&prompt[last..pos]);
        last = pos + BREAK_KEYWORD.len();
    }
    out.push_str(&prompt[last..]);
    // Collapse any double-space the stripped keyword left behind.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Iterator yielding byte positions of `BREAK` substrings that are
/// surrounded by non-word characters (or string boundaries).
///
/// "Word char" = ASCII alphanumeric + underscore (matches Rust's
/// `char::is_alphanumeric` ∪ `_`). UTF-8 word boundaries are out of
/// scope for v0.19 #5 — A1111 itself uses an ASCII `\b`.
fn word_bounded_positions(prompt: &str) -> impl Iterator<Item = usize> + '_ {
    let bytes = prompt.as_bytes();
    let needle = BREAK_KEYWORD.as_bytes();
    let needle_len = needle.len();
    (0..bytes.len().saturating_sub(needle_len - 1)).filter_map(move |i| {
        if &bytes[i..i + needle_len] != needle {
            return None;
        }
        if i > 0 && is_word_byte(bytes[i - 1]) {
            return None;
        }
        if i + needle_len < bytes.len() && is_word_byte(bytes[i + needle_len])
        {
            return None;
        }
        Some(i)
    })
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_break_negative_cases() {
        assert!(!has_break("plain prompt"));
        assert!(!has_break(""));
        assert!(!has_break("BREAKING news"));
        assert!(!has_break("BREAKDOWN"));
        assert!(!has_break("BREAKERS_v1 lora"));
        assert!(!has_break("breakfast")); // case-sensitive
        assert!(!has_break("the great escape was a BREAKER"));
    }

    #[test]
    fn has_break_positive_cases() {
        assert!(has_break("a BREAK b"));
        assert!(has_break("BREAK at the start"));
        assert!(has_break("at the end BREAK"));
        assert!(has_break("a\nBREAK\nb"));
        assert!(has_break("a,BREAK,b")); // non-word delimiters
    }

    #[test]
    fn split_no_break_returns_single_chunk() {
        let chunks = split("a plain prompt");
        assert_eq!(chunks, vec!["a plain prompt".to_string()]);
    }

    #[test]
    fn split_two_chunks() {
        let chunks = split("a brutalist whale poster BREAK watercolor on paper");
        assert_eq!(
            chunks,
            vec![
                "a brutalist whale poster".to_string(),
                "watercolor on paper".to_string(),
            ]
        );
    }

    #[test]
    fn split_three_chunks() {
        let chunks = split("first BREAK second BREAK third");
        assert_eq!(
            chunks,
            vec!["first".to_string(), "second".to_string(), "third".to_string()]
        );
    }

    #[test]
    fn split_does_not_match_breaking() {
        let chunks = split("BREAKING news from BREAKERS_v1");
        assert_eq!(chunks, vec!["BREAKING news from BREAKERS_v1".to_string()]);
    }

    #[test]
    fn split_drops_empty_chunks() {
        // Adjacent BREAKs or leading/trailing BREAK produce empty
        // strings; we drop them to match A1111.
        let chunks = split("BREAK actual content BREAK");
        assert_eq!(chunks, vec!["actual content".to_string()]);
    }

    #[test]
    fn split_all_empty_returns_one_empty() {
        // Pathological case: "BREAK BREAK" → every chunk empty;
        // return a single empty string so the encoder loop still
        // runs (producing a single zero-content CLIP encode).
        let chunks = split("BREAK BREAK");
        assert_eq!(chunks, vec![String::new()]);
    }

    #[test]
    fn split_trims_chunk_whitespace() {
        let chunks = split("   a   BREAK   b   ");
        assert_eq!(chunks, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn strip_removes_break_and_collapses_spaces() {
        assert_eq!(
            strip("a brutalist whale BREAK watercolor on paper"),
            "a brutalist whale watercolor on paper"
        );
        assert_eq!(strip("plain prompt"), "plain prompt");
        assert_eq!(strip("BREAK at start"), "at start");
        assert_eq!(strip("end here BREAK"), "end here");
    }

    #[test]
    fn strip_preserves_word_containing_break() {
        // Same word-boundary contract as split.
        assert_eq!(
            strip("BREAKING news from BREAKERS_v1"),
            "BREAKING news from BREAKERS_v1"
        );
    }
}
