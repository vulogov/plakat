//! A1111-style **prompt scheduling** + **alternation** (6.25.0 P2).
//!
//! Two syntaxes, both spelled with square brackets — resolved to a per-step effective
//! prompt string *before* the weighted encoder ([`super::a1111`]) runs:
//!
//! * **Scheduling** `[from:to:when]` — use `from` until `when`, then `to`. `when` is an
//!   integer step, or a float in `(0,1]` read as a fraction of the total step count.
//!   `[to:when]` (empty `from`) inserts `to` after `when`; `[from::when]` removes `from`
//!   after `when`.
//! * **Alternation** `[a|b|c]` — cycle every step: step 0→a, 1→b, 2→c, 3→a, …
//!
//! A **bare** `[x]` (no top-level `:` or `|`) is *de-emphasis*, not a schedule — it is left
//! untouched so [`super::a1111`] can apply the `1/1.1` bracket weight. That's the whole point
//! of resolving scheduling first and weighting second: the two `[...]` meanings don't collide.
//!
//! The scanner respects `\[`, `\]`, `\(`, `\)` escapes and `[...]`/`(...)` nesting when it
//! splits a group's fields, so a weighted or nested branch (`[a (b:1.2):c:0.5]`) parses right.

/// Max bracket nesting the resolver will recurse through before treating deeper groups as
/// literal — a guard against pathological input (downloaded PNG metadata, prompt packs).
const MAX_DEPTH: usize = 32;

/// Does `prompt` contain any scheduling (`[a:b:N]`) or alternation (`[a|b]`) syntax?
/// Bare de-emphasis `[x]` returns `false`.
pub fn has_schedule(prompt: &str) -> bool {
    top_level_groups(prompt).iter().any(|g| classify(&g.inner).is_some())
}

/// The effective prompt at `step` (0-indexed) of `total` steps. Resolves every scheduling /
/// alternation group to its active branch and recurses into that branch; leaves bare `[x]`
/// de-emphasis and `(x:1.2)` weights untouched for the downstream encoder.
pub fn prompt_at_step(prompt: &str, step: usize, total: usize) -> String {
    resolve(prompt, step, total.max(1), 0)
}

/// Distinct effective prompts across all `total` steps plus a `step → index` map, so a caller
/// encodes each unique prompt **once** and selects per step. For a prompt with no scheduling
/// this is `(vec![prompt], vec![0; total])` — one encode, byte-identical to the old path.
pub fn schedule(prompt: &str, total: usize) -> (Vec<String>, Vec<usize>) {
    let total = total.max(1);
    let mut prompts: Vec<String> = Vec::new();
    let mut idx_per_step: Vec<usize> = Vec::with_capacity(total);
    for step in 0..total {
        let p = prompt_at_step(prompt, step, total);
        let idx = match prompts.iter().position(|q| q == &p) {
            Some(i) => i,
            None => {
                prompts.push(p);
                prompts.len() - 1
            }
        };
        idx_per_step.push(idx);
    }
    (prompts, idx_per_step)
}

// ── internals ──

struct Group {
    /// Byte range of the whole `[...]` in the source (inclusive of the brackets).
    start: usize,
    end: usize,
    /// The content between the brackets.
    inner: String,
}

/// A resolved schedule/alternation group.
enum Kind {
    /// `[from:to:when]` → (from, to, when_step).
    Schedule { from: String, to: String, when: WhenSpec },
    /// `[a|b|c]` → the alternatives.
    Alternate(Vec<String>),
}

enum WhenSpec {
    Step(usize),
    Frac(f32),
}

impl WhenSpec {
    fn step(&self, total: usize) -> usize {
        match *self {
            WhenSpec::Step(s) => s,
            WhenSpec::Frac(f) => (f * total as f32).round() as usize,
        }
    }
}

/// Recursively resolve every schedule/alternation group in `s` at `step`.
fn resolve(s: &str, step: usize, total: usize, depth: usize) -> String {
    if depth >= MAX_DEPTH {
        return s.to_string();
    }
    let groups = top_level_groups(s);
    if groups.is_empty() {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0;
    let mut any = false;
    for g in &groups {
        out.push_str(&s[cursor..g.start]);
        cursor = g.end;
        match classify(&g.inner) {
            Some(kind) => {
                any = true;
                let branch = active_branch(&kind, step, total);
                // Recurse: the chosen branch may itself contain scheduling.
                out.push_str(&resolve(&branch, step, total, depth + 1));
            }
            // Not a schedule (bare `[x]` de-emphasis) — keep the brackets verbatim.
            None => out.push_str(&s[g.start..g.end]),
        }
    }
    out.push_str(&s[cursor..]);
    if any { out } else { s.to_string() }
}

/// The active alternative of a resolved group at `step`.
fn active_branch(kind: &Kind, step: usize, total: usize) -> String {
    match kind {
        Kind::Schedule { from, to, when } => {
            if step < when.step(total) { from.clone() } else { to.clone() }
        }
        Kind::Alternate(parts) => {
            if parts.is_empty() {
                String::new()
            } else {
                parts[step % parts.len()].clone()
            }
        }
    }
}

/// Decide whether a group's inner content is a schedule or alternation (or `None` = bare
/// de-emphasis). Alternation (top-level `|`) takes priority; then scheduling (a trailing
/// numeric `when` field after top-level `:`).
fn classify(inner: &str) -> Option<Kind> {
    // Alternation: any top-level `|`.
    let alt = split_top_level(inner, '|');
    if alt.len() > 1 {
        return Some(Kind::Alternate(alt));
    }
    // Scheduling: split on top-level `:`; the LAST field must be a numeric `when`.
    let fields = split_top_level(inner, ':');
    if fields.len() < 2 {
        return None;
    }
    let when = parse_when(fields.last().unwrap().trim())?;
    let (from, to) = match fields.len() {
        2 => (String::new(), fields[0].clone()),         // [to:when]
        3 => (fields[0].clone(), fields[1].clone()),     // [from:to:when] / [from::when]
        _ => return None,                                 // too many colons → not a schedule
    };
    Some(Kind::Schedule { from, to, when })
}

/// Parse a `when` field: an integer step (`10`) or a float fraction in `(0,1]` (`0.4`).
fn parse_when(s: &str) -> Option<WhenSpec> {
    if s.is_empty() {
        return None;
    }
    if let Ok(i) = s.parse::<usize>() {
        return Some(WhenSpec::Step(i));
    }
    if let Ok(f) = s.parse::<f32>() {
        if f > 0.0 && f <= 1.0 {
            return Some(WhenSpec::Frac(f));
        }
    }
    None
}

/// Split `inner` on `delim` at bracket/paren depth 0, honouring `\` escapes. Escaped
/// delimiters and delimiters inside nested `[]`/`()` are not split points.
fn split_top_level(inner: &str, delim: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            cur.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => {
                cur.push(ch);
                escaped = true;
            }
            '[' | '(' => {
                depth += 1;
                cur.push(ch);
            }
            ']' | ')' => {
                depth -= 1;
                cur.push(ch);
            }
            c if c == delim && depth == 0 => {
                parts.push(cur.clone());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    parts.push(cur);
    parts
}

/// Extract the top-level `[...]` groups of `s` (depth-0 brackets only), honouring `\` escapes.
fn top_level_groups(s: &str) -> Vec<Group> {
    let mut groups = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut escaped = false;
    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ']' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        let inner = s[start + 1..i].to_string();
                        groups.push(Group { start, end: i + 1, inner });
                    }
                }
            }
            _ => {}
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_schedule_and_alternation_not_deemphasis() {
        assert!(has_schedule("a [cat:tiger:0.4] in snow"));
        assert!(has_schedule("a [red|blue] car"));
        assert!(has_schedule("[tiger:10]"));      // insert-after
        assert!(has_schedule("[cat::0.5]"));      // remove-after
        assert!(!has_schedule("a [blurry] mess")); // bare de-emphasis
        assert!(!has_schedule("plain prompt"));
        assert!(!has_schedule("(sharp:1.3)"));     // weighting, not scheduling
    }

    #[test]
    fn from_to_switch_at_fraction() {
        // 10 steps, switch at 0.4 → step 4. Before: cat; from step 4: tiger.
        assert_eq!(prompt_at_step("a [cat:tiger:0.4] in snow", 0, 10), "a cat in snow");
        assert_eq!(prompt_at_step("a [cat:tiger:0.4] in snow", 3, 10), "a cat in snow");
        assert_eq!(prompt_at_step("a [cat:tiger:0.4] in snow", 4, 10), "a tiger in snow");
        assert_eq!(prompt_at_step("a [cat:tiger:0.4] in snow", 9, 10), "a tiger in snow");
    }

    #[test]
    fn integer_when_and_insert_remove_forms() {
        // [to:when] inserts `to` after step `when`.
        assert_eq!(prompt_at_step("a [tiger:2] cub", 1, 8), "a  cub");
        assert_eq!(prompt_at_step("a [tiger:2] cub", 2, 8), "a tiger cub");
        // [from::when] removes `from` after `when`.
        assert_eq!(prompt_at_step("a [cat::2] cub", 1, 8), "a cat cub");
        assert_eq!(prompt_at_step("a [cat::2] cub", 2, 8), "a  cub");
    }

    #[test]
    fn alternation_cycles_each_step() {
        assert_eq!(prompt_at_step("a [red|blue] car", 0, 4), "a red car");
        assert_eq!(prompt_at_step("a [red|blue] car", 1, 4), "a blue car");
        assert_eq!(prompt_at_step("a [red|blue] car", 2, 4), "a red car");
    }

    #[test]
    fn bare_deemphasis_and_weights_pass_through() {
        // No scheduling → returned verbatim so a1111.rs can weight it.
        assert_eq!(prompt_at_step("a [blurry] (sharp:1.3) fox", 3, 10), "a [blurry] (sharp:1.3) fox");
    }

    #[test]
    fn nested_schedule_resolves() {
        // Inner schedule inside the chosen branch resolves too.
        let p = "[a:[b|c]:0.5]";
        assert_eq!(prompt_at_step(p, 0, 10), "a");        // before 0.5 → "a"
        assert_eq!(prompt_at_step(p, 5, 10), "c");        // after → [b|c], step 5 → index 1 → c
        assert_eq!(prompt_at_step(p, 6, 10), "b");        // step 6 → index 0 → b
    }

    #[test]
    fn schedule_dedupes_encodes() {
        // A single-switch prompt yields exactly two distinct prompts across the schedule.
        let (prompts, idx) = schedule("a [cat:tiger:0.5] fox", 10);
        assert_eq!(prompts.len(), 2);
        assert_eq!(idx.len(), 10);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[9], 1);
        // A plain prompt yields one encode.
        let (p2, i2) = schedule("a plain fox", 6);
        assert_eq!(p2.len(), 1);
        assert!(i2.iter().all(|&x| x == 0));
    }

    #[test]
    fn escaped_brackets_are_literal() {
        assert!(!has_schedule(r"a \[not a schedule\]"));
    }
}
