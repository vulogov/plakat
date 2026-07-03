//! A1111-style prompt attention syntax: `(token:1.2)`, `[token]`,
//! nested parens for compounded emphasis, and `\(` / `\[` escapes.
//!
//! Recognized grammar:
//!
//! | Token | Weight |
//! |---|---|
//! | `(text)` | `* 1.1` (parens = default emphasis) |
//! | `(text:F)` | `* F` (explicit float, e.g. `(text:1.5)`) |
//! | `[text]` | `* 1/1.1 ≈ 0.909` (brackets = default de-emphasis) |
//! | `[text:F]` | `* F` (explicit float — same semantic as parens) |
//! | `((text))` | nested → `* 1.1 * 1.1 = 1.21` |
//! | `\(`, `\)`, `\[`, `\]`, `\:` | escaped — literal punctuation |
//!
//! The output is a flat `Vec<(String, f32)>` of text segments with
//! their per-segment weight. Each segment will tokenize independently
//! at the encoder boundary, and every output token from that segment
//! carries the segment's weight. The encoder integration applies the
//! weights by scaling each token's CLIP hidden-state row by the
//! corresponding weight (the "scale hidden states" implementation
//! A1111 originally shipped; simpler than mean-preserving variants
//! and produces visible effect with no normalization surprises).
//!
//! Unbalanced brackets are treated as literals — same robustness
//! contract the wildcard parser uses. A prompt with a stray `(`
//! gets passed through unweighted instead of erroring; mismatches
//! are common enough in copy-pasted Civitai prompts that bailing
//! would mostly annoy users.

/// One parsed segment: a literal text fragment with the weight that
/// should multiply every CLIP hidden-state row produced by tokenizing
/// + encoding that fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedSegment {
    pub text: String,
    pub weight: f32,
}

/// Default emphasis bump applied by bare parens / brackets. Matches
/// A1111: `1.1` for parens (so `((x))` → `1.21`), `1/1.1 ≈ 0.909` for
/// brackets.
pub const ATTENTION_MULTIPLIER: f32 = 1.1;

/// Max `(…)`/`[…]` nesting the weight parser will recurse into. Beyond this the inner
/// text is kept literal — a guard against a stack overflow from pathological nesting in
/// an untrusted prompt (e.g. downloaded PNG metadata / community prompt packs). Real
/// prompts nest a handful of levels; 64 is far beyond legitimate use.
const MAX_NEST_DEPTH: usize = 64;

/// Parse an A1111-style prompt into a sequence of weighted segments.
/// Adjacent segments with the same weight are coalesced so the
/// caller doesn't tokenize trivially-fragmented prompts.
///
/// Returns at minimum a single segment with weight `1.0` (the whole
/// prompt unweighted) for prompts without any attention syntax —
/// the no-op contract that lets the encoder integration call this
/// unconditionally without checking for emptiness first.
pub fn parse(prompt: &str) -> Vec<WeightedSegment> {
    let segments = parse_inner(prompt);
    coalesce_adjacent(segments)
}

/// Parse without coalescing. Exposed for tests so the raw bracket /
/// paren structure can be asserted independently of the coalesce
/// step.
fn parse_inner(prompt: &str) -> Vec<WeightedSegment> {
    let chars: Vec<char> = prompt.chars().collect();
    let mut out = Vec::new();
    let mut weight_stack: Vec<f32> = vec![1.0];
    let mut buf = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // Escape sequences come first so a `\(` in the input is a
        // literal `(`, not a group opener.
        if c == '\\' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if matches!(next, '(' | ')' | '[' | ']' | ':' | '\\') {
                buf.push(next);
                i += 2;
                continue;
            }
        }
        match c {
            '(' | '[' => {
                flush_segment(&mut out, &mut buf, *weight_stack.last().unwrap());
                let is_paren = c == '(';
                // Look ahead for an explicit `:WEIGHT)` closer to
                // decide whether this group is bare (paren default
                // = 1.1, bracket default = 1/1.1) or explicit.
                let group_end = find_matching_close(&chars, i, c);
                let close_char = if is_paren { ')' } else { ']' };
                match group_end {
                    Some(end) => {
                        let (inner, explicit_weight) =
                            split_inner_and_weight(&chars[i + 1..end]);
                        let factor = match explicit_weight {
                            Some(w) => w,
                            None => {
                                if is_paren {
                                    ATTENTION_MULTIPLIER
                                } else {
                                    1.0 / ATTENTION_MULTIPLIER
                                }
                            }
                        };
                        let new_weight = weight_stack.last().unwrap() * factor;
                        weight_stack.push(new_weight);
                        let inner_segments = parse_inner_with_initial_weight(
                            &inner.iter().collect::<String>(),
                            new_weight,
                            1,
                        );
                        out.extend(inner_segments);
                        weight_stack.pop();
                        i = end + 1;
                        let _ = close_char;
                        continue;
                    }
                    None => {
                        // Unbalanced open — treat as literal.
                        buf.push(c);
                        i += 1;
                    }
                }
            }
            ')' | ']' => {
                // Stray close — literal. Same robustness contract.
                buf.push(c);
                i += 1;
            }
            _ => {
                buf.push(c);
                i += 1;
            }
        }
    }
    flush_segment(&mut out, &mut buf, *weight_stack.last().unwrap());
    out
}

fn flush_segment(out: &mut Vec<WeightedSegment>, buf: &mut String, weight: f32) {
    if !buf.is_empty() {
        out.push(WeightedSegment {
            text: std::mem::take(buf),
            weight,
        });
    }
}

/// Parse a nested group's inner content. Identical to `parse_inner`
/// but seeds the weight stack with the caller's current weight so
/// nested parens compound. (The outer `parse_inner` always seeds
/// with `1.0`; this helper lets a `(a (b) c)` group give `b` a
/// weight of `1.1 * 1.1` rather than just `1.1`.)
fn parse_inner_with_initial_weight(prompt: &str, initial: f32, depth: usize) -> Vec<WeightedSegment> {
    let chars: Vec<char> = prompt.chars().collect();
    let mut out = Vec::new();
    let mut weight_stack: Vec<f32> = vec![initial];
    let mut buf = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if matches!(next, '(' | ')' | '[' | ']' | ':' | '\\') {
                buf.push(next);
                i += 2;
                continue;
            }
        }
        match c {
            '(' | '[' => {
                flush_segment(&mut out, &mut buf, *weight_stack.last().unwrap());
                let is_paren = c == '(';
                let group_end = find_matching_close(&chars, i, c);
                match group_end {
                    Some(end) => {
                        let (inner, explicit_weight) =
                            split_inner_and_weight(&chars[i + 1..end]);
                        let factor = match explicit_weight {
                            Some(w) => w,
                            None => {
                                if is_paren {
                                    ATTENTION_MULTIPLIER
                                } else {
                                    1.0 / ATTENTION_MULTIPLIER
                                }
                            }
                        };
                        let new_weight = weight_stack.last().unwrap() * factor;
                        weight_stack.push(new_weight);
                        let mut inner_text: String = inner.iter().collect();
                        if depth >= MAX_NEST_DEPTH {
                            // Too deep — stop recursing (stack-overflow guard). Keep the
                            // inner text as one literal segment at the accumulated weight.
                            flush_segment(&mut out, &mut inner_text, new_weight);
                        } else {
                            let inner_segments =
                                parse_inner_with_initial_weight(&inner_text, new_weight, depth + 1);
                            out.extend(inner_segments);
                        }
                        weight_stack.pop();
                        i = end + 1;
                        continue;
                    }
                    None => {
                        buf.push(c);
                        i += 1;
                    }
                }
            }
            ')' | ']' => {
                buf.push(c);
                i += 1;
            }
            _ => {
                buf.push(c);
                i += 1;
            }
        }
    }
    flush_segment(&mut out, &mut buf, *weight_stack.last().unwrap());
    out
}

/// Find the index of the `)` or `]` that closes the opener at
/// `start`, accounting for nested parens / brackets. `\(` and `\[`
/// inside the group don't change depth.
fn find_matching_close(chars: &[char], start: usize, opener: char) -> Option<usize> {
    let closer = match opener {
        '(' => ')',
        '[' => ']',
        _ => return None,
    };
    let mut depth = 0usize;
    let mut i = start;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() {
            i += 2;
            continue;
        }
        if c == opener {
            depth += 1;
        } else if c == closer {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Split a group's body into `(inner_chars, explicit_weight)`.
/// If the body ends with `:FLOAT`, that float becomes the weight
/// and the inner is the body up to the colon. Otherwise the
/// whole body is the inner with `None` weight (caller applies the
/// default 1.1 or 1/1.1).
fn split_inner_and_weight(body: &[char]) -> (Vec<char>, Option<f32>) {
    // Find the last unescaped `:` not inside a nested group.
    let mut depth_p = 0i32;
    let mut depth_b = 0i32;
    let mut colon_pos: Option<usize> = None;
    let mut i = 0;
    while i < body.len() {
        let c = body[i];
        if c == '\\' && i + 1 < body.len() {
            i += 2;
            continue;
        }
        match c {
            '(' => depth_p += 1,
            ')' => depth_p -= 1,
            '[' => depth_b += 1,
            ']' => depth_b -= 1,
            ':' if depth_p == 0 && depth_b == 0 => {
                colon_pos = Some(i);
            }
            _ => {}
        }
        i += 1;
    }
    if let Some(pos) = colon_pos {
        let trailing: String = body[pos + 1..].iter().collect();
        if let Ok(w) = trailing.trim().parse::<f32>() {
            if w.is_finite() {
                let inner = body[..pos].to_vec();
                return (inner, Some(w));
            }
        }
    }
    (body.to_vec(), None)
}

/// Merge consecutive segments with the same weight. Reduces
/// downstream tokenization cost and avoids segment-boundary
/// fragmentation that would split a single English word across
/// two tokenizer passes.
fn coalesce_adjacent(segs: Vec<WeightedSegment>) -> Vec<WeightedSegment> {
    let mut out: Vec<WeightedSegment> = Vec::with_capacity(segs.len());
    for s in segs {
        if let Some(last) = out.last_mut() {
            if (last.weight - s.weight).abs() < f32::EPSILON {
                last.text.push_str(&s.text);
                continue;
            }
        }
        out.push(s);
    }
    out
}

/// Convenience: `true` when the prompt contains no A1111 attention
/// syntax at all (no `(`, `)`, `[`, `]`, or `\`). Cheap pre-check
/// for the encoder integration's hot path — when this returns true,
/// the encoder can run its standard non-weighted forward without
/// going through the per-segment tokenization.
pub fn has_attention_syntax(prompt: &str) -> bool {
    prompt.chars().any(|c| matches!(c, '(' | ')' | '[' | ']' | '\\'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deeply_nested_groups_do_not_overflow_the_stack() {
        // Pathological nesting from an untrusted prompt (community pack / PNG metadata)
        // used to recurse once per level → SIGABRT. The depth cap keeps it bounded.
        let n = 50_000;
        let prompt = format!("{}core{}", "(".repeat(n), ")".repeat(n));
        let out = parse(&prompt); // must return, not blow the stack
        assert!(out.iter().any(|s| s.text.contains("core")), "inner content survives as a segment");
    }

    fn segs(prompt: &str) -> Vec<(String, f32)> {
        parse(prompt)
            .into_iter()
            .map(|s| (s.text, s.weight))
            .collect()
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn plain_prompt_passes_through_at_weight_one() {
        let out = segs("a red fox");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "a red fox");
        assert!(approx_eq(out[0].1, 1.0));
    }

    #[test]
    fn empty_prompt_returns_empty_vec() {
        assert_eq!(parse("").len(), 0);
    }

    #[test]
    fn bare_parens_apply_default_emphasis() {
        let out = segs("(red)");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "red");
        assert!(approx_eq(out[0].1, ATTENTION_MULTIPLIER));
    }

    #[test]
    fn bare_brackets_apply_default_de_emphasis() {
        let out = segs("[blue]");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "blue");
        assert!(approx_eq(out[0].1, 1.0 / ATTENTION_MULTIPLIER));
    }

    #[test]
    fn explicit_weight_in_parens() {
        let out = segs("(red:1.5)");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "red");
        assert!(approx_eq(out[0].1, 1.5));
    }

    #[test]
    fn explicit_weight_in_brackets_treated_as_literal_weight() {
        // A1111: `[x:0.7]` uses the literal 0.7, not inverse.
        let out = segs("[red:0.7]");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "red");
        assert!(approx_eq(out[0].1, 0.7));
    }

    #[test]
    fn nested_parens_compound_weights() {
        // ((red)) → 1.1 * 1.1 = 1.21
        let out = segs("((red))");
        assert_eq!(out.len(), 1);
        assert!(approx_eq(out[0].1, 1.21), "got {:?}", out);
    }

    #[test]
    fn nested_paren_with_explicit_weight() {
        // ((red:1.5)) → 1.5 * 1.1 = 1.65
        let out = segs("((red:1.5))");
        assert_eq!(out.len(), 1);
        assert!(approx_eq(out[0].1, 1.65), "got {:?}", out);
    }

    #[test]
    fn mixed_weighted_and_unweighted_segments() {
        let out = segs("a (red:1.5) fox");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, "a ");
        assert!(approx_eq(out[0].1, 1.0));
        assert_eq!(out[1].0, "red");
        assert!(approx_eq(out[1].1, 1.5));
        assert_eq!(out[2].0, " fox");
        assert!(approx_eq(out[2].1, 1.0));
    }

    #[test]
    fn adjacent_same_weight_segments_coalesce() {
        // Two bare-paren groups in a row at the same weight should
        // produce two separate segments (different group origins
        // but same weight) — and the coalesce pass merges them.
        let out = segs("(a)(b)");
        // Both at weight 1.1 → merged into one segment "ab".
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "ab");
        assert!(approx_eq(out[0].1, ATTENTION_MULTIPLIER));
    }

    #[test]
    fn escaped_parens_are_literal() {
        let out = segs("a \\(literal paren\\) fox");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "a (literal paren) fox");
        assert!(approx_eq(out[0].1, 1.0));
    }

    #[test]
    fn escaped_brackets_are_literal() {
        let out = segs("\\[a sketch\\]");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "[a sketch]");
    }

    #[test]
    fn unbalanced_open_paren_is_literal() {
        // No matching `)` — emit as literal.
        let out = segs("a ( red fox");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "a ( red fox");
    }

    #[test]
    fn unbalanced_close_paren_is_literal() {
        let out = segs("a ) fox");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "a ) fox");
    }

    #[test]
    fn explicit_weight_with_whitespace() {
        // Civitai prompts often have `(red : 1.5)` — accept it.
        let out = segs("(red : 1.5 )");
        assert_eq!(out.len(), 1);
        // Note: the inner text gets the surrounding whitespace
        // preserved (before the colon split-point).
        assert_eq!(out[0].0, "red ");
        assert!(approx_eq(out[0].1, 1.5));
    }

    #[test]
    fn deep_nesting() {
        // (((red))) → 1.1^3 = 1.331
        let out = segs("(((red)))");
        assert!(approx_eq(out[0].1, 1.331), "got {:?}", out);
    }

    #[test]
    fn paren_inside_brackets() {
        // [(red:1.5)] → 1.5 * (1/1.1) ≈ 1.3636
        let out = segs("[(red:1.5)]");
        assert!(approx_eq(out[0].1, 1.5 / ATTENTION_MULTIPLIER), "got {:?}", out);
    }

    #[test]
    fn has_attention_syntax_detection() {
        assert!(!has_attention_syntax("plain prompt"));
        assert!(has_attention_syntax("a (red) fox"));
        assert!(has_attention_syntax("[blue]"));
        assert!(has_attention_syntax("\\(escaped\\)"));
    }

    #[test]
    fn malformed_weight_falls_back_to_default() {
        // `(red:abc)` — non-numeric weight string. A1111 treats
        // this as a literal `:abc` in the text and applies the
        // default 1.1 emphasis. We match.
        let out = segs("(red:abc)");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "red:abc");
        assert!(approx_eq(out[0].1, ATTENTION_MULTIPLIER));
    }

    #[test]
    fn realistic_civitai_prompt() {
        // Lifted from a real Civitai LoRA card.
        let out = segs("masterpiece, best quality, (1girl:1.2), (red hair:1.3), [low quality]");
        // Five segments: leading literal text, "1girl" at 1.2,
        // separator, "red hair" at 1.3, separator, "low quality" at
        // 1/1.1.
        // Coalesce: leading + first separator may not merge
        // because they wrap weighted groups; what we want to
        // verify is the weighted words got their weights right.
        let red_hair = out.iter().find(|(t, _)| t == "red hair").unwrap();
        assert!(approx_eq(red_hair.1, 1.3));
        let one_girl = out.iter().find(|(t, _)| t == "1girl").unwrap();
        assert!(approx_eq(one_girl.1, 1.2));
        let low_q = out.iter().find(|(t, _)| t == "low quality").unwrap();
        assert!(approx_eq(low_q.1, 1.0 / ATTENTION_MULTIPLIER));
    }
}
