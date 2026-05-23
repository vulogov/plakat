//! v0.16 phase 5: prompt wildcard expansion.
//!
//! Two syntaxes the Auto1111 / ComfyUI / Forge communities all share:
//!
//! 1. **Inline alternation**: `{red|blue|green}` picks one of the
//!    three pipe-separated options at random. Nestable:
//!    `{a {b|c}|d}` expands to either `a b`, `a c`, or `d`.
//!    Whitespace around the pipes is preserved (`{ red | blue }`
//!    yields `" red "` or `" blue "`).
//!
//! 2. **File wildcards**: `__colors__` reads `<dir>/colors.txt` and
//!    picks a uniformly-random non-empty, non-comment (`#`) line.
//!    Lines may themselves contain wildcards (inline or file) —
//!    expansion recurses with a depth cap.
//!
//! Expansion is randomised via a `rand::Rng` so callers can seed
//! reproducibly (the t2i CLI threads its `--seed` into the
//! wildcard RNG, so the same seed reproduces the same expansion).
//!
//! Recursion is depth-bounded (`MAX_DEPTH = 8`). At the cap the
//! remaining wildcards are left as literal tokens — better than a
//! hang or a stack overflow on a self-referential wildcard file
//! like `__synonyms__` containing `__synonyms__`.

use anyhow::{Context, Result, bail};
use rand::Rng;
use rand::seq::SliceRandom;
use std::path::Path;

const MAX_DEPTH: usize = 8;

/// Expand both inline `{a|b|c}` alternation AND `__file__` wildcards
/// in `prompt`. `wildcard_dir` is the directory file wildcards
/// resolve against (typically `wildcards/` next to the user's
/// config). Pass `rng` so the caller controls seeding; the t2i CLI
/// feeds the same `--seed` it uses for noise generation.
///
/// Returns the literal prompt unchanged when there are no
/// wildcards — no allocation, no IO.
pub fn expand<R: Rng + ?Sized>(
    prompt: &str,
    wildcard_dir: Option<&Path>,
    rng: &mut R,
) -> Result<String> {
    if !has_wildcards(prompt) {
        return Ok(prompt.to_string());
    }
    expand_inner(prompt, wildcard_dir, rng, 0)
}

/// `true` iff `s` contains at least one wildcard token. Cheap check
/// — skips the full expander when there's nothing to do.
fn has_wildcards(s: &str) -> bool {
    // Either `{...|...}` (open + at least one pipe within an unclosed
    // group), or `__name__` (paired underscores around a non-empty
    // identifier). Cheap scan over bytes; full parsing happens only
    // when these markers are present.
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            return true;
        }
        if bytes[i] == b'_' && i + 1 < bytes.len() && bytes[i + 1] == b'_' {
            return true;
        }
        i += 1;
    }
    false
}

fn expand_inner<R: Rng + ?Sized>(
    prompt: &str,
    wildcard_dir: Option<&Path>,
    rng: &mut R,
    depth: usize,
) -> Result<String> {
    if depth >= MAX_DEPTH {
        // Bail soft: leave the prompt as-is rather than recursing
        // unboundedly on a self-referential wildcard file.
        tracing::warn!(
            target: "plakat",
            "wildcard expansion hit MAX_DEPTH={MAX_DEPTH} \
             (likely a self-referential file wildcard). Leaving \
             remaining wildcards as literals."
        );
        return Ok(prompt.to_string());
    }
    // Inline `{...}` expansion first — choices can name file
    // wildcards that get resolved on the next pass.
    let after_inline = expand_inline(prompt, rng)?;
    let after_files = expand_files(&after_inline, wildcard_dir, rng)?;
    // If the second pass introduced new wildcards (file content with
    // inline alternation or further file refs), recurse.
    if has_wildcards(&after_files) && after_files != prompt {
        expand_inner(&after_files, wildcard_dir, rng, depth + 1)
    } else {
        Ok(after_files)
    }
}

/// Replace each top-level `{a|b|c}` group with a random choice.
/// Nested groups are handled by depth-tracking the brace level —
/// only `|` at depth 1 splits the current group. A `{` with no
/// matching `}` is left literal (no bail — robustness over
/// strictness, same way A1111 handles malformed wildcards).
fn expand_inline<R: Rng + ?Sized>(s: &str, rng: &mut R) -> Result<String> {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '{' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        // Find matching `}` accounting for nesting. If no match,
        // emit the `{` literally and continue.
        let group_end = match find_group_end(&chars, i) {
            Some(e) => e,
            None => {
                out.push('{');
                i += 1;
                continue;
            }
        };
        // Split body on top-level `|`s.
        let body: String = chars[i + 1..group_end].iter().collect();
        let options = split_top_level(&body);
        let pick = options
            .choose(rng)
            .ok_or_else(|| anyhow::anyhow!("empty wildcard group {{}}"))?;
        // Recurse so nested `{...}` inside the picked option also
        // expand. Pure recursion on `expand_inline` only — file
        // wildcards are handled by the outer `expand_inner` pass.
        out.push_str(&expand_inline(pick, rng)?);
        i = group_end + 1;
    }
    Ok(out)
}

/// Index of the `}` matching the `{` at `start`, accounting for
/// nesting. `None` if unmatched.
fn find_group_end(chars: &[char], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in chars.iter().enumerate().skip(start) {
        match *c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split `body` on `|` at brace depth 0. Brace depth is tracked so
/// `{a|b}|c` inside a group splits into `["{a|b}", "c"]` (two
/// options) not `["{a", "b}", "c"]`.
fn split_top_level(body: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let bytes = body.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' if depth > 0 => depth -= 1,
            b'|' if depth == 0 => {
                out.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&body[start..]);
    out
}

/// Replace each `__name__` with a random non-empty, non-comment line
/// from `<wildcard_dir>/name.txt`. Names allow letters / digits /
/// `-` / `_`. Missing files bail loud (a wildcard the user explicitly
/// named is almost certainly meant to exist; silent fallthrough
/// would surface mid-generate as a weird literal in the prompt).
///
/// When `wildcard_dir` is `None`, leaves `__name__` tokens as
/// literals. Lets pipelines call `expand` unconditionally without
/// requiring a path setup when the user isn't using file wildcards.
fn expand_files<R: Rng + ?Sized>(
    s: &str,
    wildcard_dir: Option<&Path>,
    rng: &mut R,
) -> Result<String> {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Look for `__` followed by an identifier and another `__`.
        // The identifier accepts `[a-zA-Z0-9-]` plus single `_`
        // (single `_` mid-name is allowed for names like
        // `warm_colors`; a doubled `__` always closes the wildcard,
        // never widens the name).
        if i + 1 < chars.len()
            && chars[i] == '_'
            && chars[i + 1] == '_'
        {
            let name_start = i + 2;
            let mut j = name_start;
            while j < chars.len() {
                let c = chars[j];
                if c.is_ascii_alphanumeric() || c == '-' {
                    j += 1;
                } else if c == '_' {
                    // Doubled `_` → end of wildcard. Single `_` →
                    // part of the name; consume just this one.
                    if j + 1 < chars.len() && chars[j + 1] == '_' {
                        break;
                    }
                    j += 1;
                } else {
                    break;
                }
            }
            if j + 1 < chars.len()
                && j > name_start
                && chars[j] == '_'
                && chars[j + 1] == '_'
            {
                let name: String = chars[name_start..j].iter().collect();
                let replacement = match wildcard_dir {
                    Some(dir) => pick_wildcard_line(dir, &name, rng)
                        .with_context(|| format!("expanding wildcard __{name}__"))?,
                    None => {
                        // No directory configured — leave the token
                        // as a literal. The user is responsible for
                        // wiring `--wildcard-dir` if they want this.
                        out.push_str(&format!("__{name}__"));
                        i = j + 2;
                        continue;
                    }
                };
                out.push_str(&replacement);
                i = j + 2;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    Ok(out)
}

/// Read `<dir>/<name>.txt`, filter blank + comment lines, and pick
/// one uniformly at random.
fn pick_wildcard_line<R: Rng + ?Sized>(
    dir: &Path,
    name: &str,
    rng: &mut R,
) -> Result<String> {
    let path = dir.join(format!("{name}.txt"));
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("reading wildcard file {}", path.display()))?;
    let lines: Vec<&str> = contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if lines.is_empty() {
        bail!(
            "wildcard file {} has no usable lines (all blank or `#` comments)",
            path.display()
        );
    }
    let pick = *lines
        .choose(rng)
        .expect("non-empty lines vector — checked above");
    Ok(pick.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn rng() -> StdRng {
        // Deterministic for assertions.
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn no_wildcards_passes_through_unchanged() {
        let mut r = rng();
        let out = expand("a red fox", None, &mut r).unwrap();
        assert_eq!(out, "a red fox");
    }

    #[test]
    fn inline_alternation_picks_one() {
        let mut r = rng();
        let out = expand("a {red|blue|green} fox", None, &mut r).unwrap();
        // Whatever the RNG picks, it must be one of the three options
        // wrapped in the surrounding context.
        let valid = ["a red fox", "a blue fox", "a green fox"];
        assert!(
            valid.contains(&out.as_str()),
            "got {out:?}, expected one of {valid:?}"
        );
    }

    #[test]
    fn inline_alternation_handles_whitespace_in_options() {
        let mut r = rng();
        let out = expand("{ red | blue }", None, &mut r).unwrap();
        assert!(out == " red " || out == " blue ", "got {out:?}");
    }

    #[test]
    fn inline_nested_alternation() {
        // Nested: each pick may itself be a group. Outer chooses
        // between "{a|b}" and "c"; if it picks the group, inner
        // chooses between "a" and "b".
        let mut r = rng();
        for _ in 0..20 {
            let out = expand("{ {a|b}|c}", None, &mut r).unwrap();
            assert!(
                matches!(out.as_str(), " a" | " b" | "c"),
                "got {out:?}"
            );
        }
    }

    #[test]
    fn inline_unmatched_brace_is_literal() {
        let mut r = rng();
        // No closing `}` → leave the `{` literal, don't bail.
        let out = expand("a { red fox", None, &mut r).unwrap();
        assert_eq!(out, "a { red fox");
    }

    #[test]
    fn file_wildcard_picks_line() {
        let mut r = rng();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("colors.txt"),
            "red\nblue\ngreen\n# comment line\n\n",
        )
        .unwrap();
        let out = expand("a __colors__ fox", Some(dir.path()), &mut r).unwrap();
        let valid = ["a red fox", "a blue fox", "a green fox"];
        assert!(
            valid.contains(&out.as_str()),
            "got {out:?}, expected one of {valid:?}"
        );
    }

    #[test]
    fn file_wildcard_skips_blank_and_comments() {
        let mut r = rng();
        let dir = tempfile::tempdir().unwrap();
        // Only `red` is a real line; the rest are filtered.
        std::fs::write(
            dir.path().join("single.txt"),
            "# comment\n\n   \nred\n",
        )
        .unwrap();
        let out = expand("__single__", Some(dir.path()), &mut r).unwrap();
        assert_eq!(out, "red");
    }

    #[test]
    fn file_wildcard_without_dir_left_as_literal() {
        let mut r = rng();
        let out = expand("a __colors__ fox", None, &mut r).unwrap();
        // No dir configured → token passes through literally so the
        // user notices and wires --wildcard-dir.
        assert_eq!(out, "a __colors__ fox");
    }

    #[test]
    fn file_wildcard_missing_file_bails() {
        let mut r = rng();
        let dir = tempfile::tempdir().unwrap();
        let err = expand("__nonexistent__", Some(dir.path()), &mut r).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("nonexistent"), "got {msg}");
    }

    #[test]
    fn file_wildcard_all_blank_bails() {
        let mut r = rng();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.txt"), "# a\n\n   \n# b\n").unwrap();
        let err = expand("__empty__", Some(dir.path()), &mut r).unwrap_err();
        // `{:#}` walks the anyhow source chain — the outer context
        // is "expanding wildcard __empty__"; the inner is "no
        // usable lines".
        let msg = format!("{err:#}");
        assert!(msg.contains("no usable lines"), "got {msg}");
    }

    #[test]
    fn file_wildcard_with_inline_alternation_in_line() {
        // Wildcard file contains an inline `{a|b}` group — expander
        // should recurse and resolve it.
        let mut r = rng();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("mix.txt"), "{ruby|crimson} red\n").unwrap();
        let out = expand("__mix__", Some(dir.path()), &mut r).unwrap();
        assert!(
            matches!(out.as_str(), "ruby red" | "crimson red"),
            "got {out:?}"
        );
    }

    #[test]
    fn self_referential_file_terminates_at_max_depth() {
        // A wildcard file that references itself would loop forever
        // without the depth cap. Verify the cap fires + leaves the
        // remaining `__loop__` as a literal.
        let mut r = rng();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("loop.txt"), "__loop__\n").unwrap();
        // Should not hang or stack-overflow; should return without
        // panicking, with some `__loop__` remaining in output.
        let out = expand("__loop__", Some(dir.path()), &mut r).unwrap();
        assert!(out.contains("__loop__"), "got {out:?}");
    }

    #[test]
    fn ident_chars_accept_dash_and_underscore() {
        // Wildcard names like `__warm-colors__` or `__warm_colors__`
        // are common in published Auto1111 wildcard packs.
        let mut r = rng();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("warm-colors.txt"), "amber\n").unwrap();
        std::fs::write(dir.path().join("warm_colors.txt"), "ruby\n").unwrap();
        let out = expand("__warm-colors__ __warm_colors__", Some(dir.path()), &mut r).unwrap();
        assert_eq!(out, "amber ruby");
    }
}
