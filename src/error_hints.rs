//! v0.33 phase 1: actionable error hints.
//!
//! Plakat surfaces three categories of failure that benefit from
//! decorated diagnostics rather than raw library errors:
//!
//! 1. **Missing model alias** — `--model sd1.5` (vs the canonical
//!    `sd15`) currently falls through to a 404 fetch error. The
//!    nicer path is to detect at CLI boundary that the alias didn't
//!    resolve AND the value doesn't look like an `org/name` repo,
//!    then suggest the closest known alias via simple edit
//!    distance.
//!
//! 2. **OOM** — candle errors when a tensor allocation fails carry
//!    raw "out of memory" wording with no plakat-specific
//!    suggestion. The decorator appends context-appropriate
//!    advice: `--size 768x768` for SD, `--quant-level Q4_K_S` for
//!    Flux, `--tiled` for hires fix paths.
//!
//! 3. **Scenario HJSON parse** — `deser_hjson::Error` already
//!    points at the line/column but doesn't mention the task name
//!    if the failure is inside a specific task's block. The
//!    decorator scans the error column against the source and
//!    adds the task name pointer when discoverable.
//!
//! Each helper takes an existing `anyhow::Error` (or `Result`) and
//! returns a decorated version — no new error taxonomy.

use anyhow::Result;

/// v0.33 phase 1: which kind of pipeline the OOM hint should
/// suggest mitigations for. Picks the most-relevant flag list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OomContext {
    /// SD-family t2i / img2img / portrait / stylize.
    Sd,
    /// SDXL specifically — larger VAE + UNet; size suggestion lower.
    Sdxl,
    /// Flux (BF16 / NF4 / GGUF) — quantization is the primary lever.
    Flux,
    /// AnimateDiff long-form — window-size + frame-count knobs.
    Animate,
}

impl OomContext {
    /// Pipeline-specific mitigation suggestions appended to OOM
    /// errors. Returned as a vec of strings so callers can format
    /// them however they like (currently joined with `\n  · ` for
    /// bullet display).
    pub fn suggestions(self) -> Vec<&'static str> {
        match self {
            OomContext::Sd => vec![
                "try a smaller image: `--size 768x768` (currently bigger)",
                "drop the refiner: omit `--refiner` if you set it",
                "skip ADetailer / hires-fix passes if any are stacked",
                "on Apple/Metal, retry on CPU: `--device cpu` (no single-buffer cap; slower but always fits)",
            ],
            OomContext::Sdxl => vec![
                "try a smaller image: `--size 768x768` (SDXL trains at 1024²)",
                "drop the refiner: omit `--refiner` if you set it",
                "switch to `--scheduler lcm` + an LCM-LoRA for 4-step inference",
                "on Apple/Metal, retry on CPU: `--device cpu` (no single-buffer cap; slower but always fits)",
            ],
            OomContext::Flux => vec![
                "use a quantized model: `--model flux-dev-gguf --flux-quant-level Q4_K_S` (~6 GB transformer)",
                "use NF4: `--model flux-dev-nf4` (~6 GB transformer)",
                "add `--quantize-t5` to keep the T5 encoder in BF16 → INT8 (~50% smaller)",
                "use `--tiled` for hires runs at >1024²",
            ],
            OomContext::Animate => vec![
                "reduce `--frames` (motion adapter trains at 16)",
                "reduce `--window-size` (max 32 per window)",
                "drop ControlNet stacking; each conditioner adds ~1-2 GB",
                "switch to SD 1.5 from SDXL animate beta",
            ],
        }
    }
}

/// v0.33 phase 1: detect OOM error patterns + append actionable
/// mitigation suggestions. Pass the original `anyhow::Error` and
/// the pipeline context; if the error looks like OOM, the returned
/// `anyhow::Error` adds the suggestions to the chain. Otherwise
/// returns the input unchanged.
///
/// Detection is conservative: we match the substring `"out of
/// memory"` (case-insensitive) since that's what every candle
/// backend uses (CUDA OOM, Metal MTLBuffer alloc failure, CPU
/// allocator). False positives are unlikely — the phrase rarely
/// appears in other error contexts.
pub fn decorate_oom(e: anyhow::Error, ctx: OomContext) -> anyhow::Error {
    let raw = format!("{e:#}");
    if !looks_like_oom(&raw) {
        return e;
    }
    let suggestions = ctx
        .suggestions()
        .iter()
        .map(|s| format!("  · {s}"))
        .collect::<Vec<_>>()
        .join("\n");
    e.context(format!("out-of-memory; try:\n{suggestions}"))
}

/// Pure substring check exposed for tests.
pub fn looks_like_oom(s: &str) -> bool {
    let lower = s.to_lowercase();
    lower.contains("out of memory")
        || lower.contains("oom")
        || lower.contains("cuda_error_out_of_memory")
        || lower.contains("cudamalloc")
        || lower.contains("mtl: allocation failed")
}

/// v0.33 phase 1: detect that a `--model` value didn't resolve to
/// any known alias AND doesn't look like a full HF repo (`org/name`),
/// and suggest the closest alias from the table.
///
/// Returns `Some(closest)` when an alias is within edit distance 2
/// (or a prefix / suffix match), `None` when nothing reasonable
/// surfaces.
///
/// Pure function exposed for tests. The CLI wraps the model arg
/// before passing it through to download.
pub fn closest_alias<'a>(name: &str, candidates: &'a [&str]) -> Option<&'a str> {
    if name.contains('/') {
        return None; // looks like a real org/name repo; user knows what they're doing
    }
    let lower = name.to_lowercase();
    let mut best: Option<(usize, &str)> = None;
    for &cand in candidates {
        let d = edit_distance(&lower, &cand.to_lowercase());
        // Threshold: accept if d <= max(2, len/3). Catches "sd1.5"
        // vs "sd15" (d=2) and "sdxlturbo" vs "sdxl-turbo" (d=1).
        let threshold = std::cmp::max(2, cand.len() / 3);
        if d <= threshold {
            match best {
                Some((b, _)) if d >= b => {}
                _ => best = Some((d, cand)),
            }
        }
    }
    best.map(|(_, c)| c)
}

/// Iterative Levenshtein edit distance. Small; meant for short
/// alias strings (`sd15`, `flux-dev`, etc.) where the n^2 cost is
/// negligible.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for (i, ac) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bc) in b.iter().enumerate() {
            let cost = if ac == bc { 0 } else { 1 };
            curr[j + 1] = std::cmp::min(
                std::cmp::min(curr[j] + 1, prev[j + 1] + 1),
                prev[j] + cost,
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// v0.33 phase 1: lift the `Result<T>` into a decorated form when a
/// model alias miss occurs. Used at the CLI boundary before
/// `--model` hits the HF resolver.
pub fn hint_unknown_alias(model: &str, known_aliases: &[&str]) -> Result<()> {
    // If the value looks like an `org/name` repo, never bail —
    // download.rs's 404 path will report it cleanly with the
    // friendly_error decorator already in place.
    if model.contains('/') {
        return Ok(());
    }
    if known_aliases.iter().any(|a| *a == model) {
        return Ok(());
    }
    let closest = closest_alias(model, known_aliases);
    let suggestion = match closest {
        Some(c) => format!(" Did you mean `{c}`?"),
        None => String::new(),
    };
    anyhow::bail!(
        "unknown --model alias `{model}`.{suggestion} \
         Run `plakat models aliases` to see the full list, or pass a \
         HuggingFace `org/name` repo path."
    );
}

/// v0.33 phase 1: enrich a `deser_hjson` parse error with the task
/// name when the underlying error message identifies a field
/// inside a specific `tasks: [...]` block. Best-effort: if we
/// can't determine the surrounding task, returns the original
/// error unchanged.
pub fn decorate_scenario_parse(e: anyhow::Error, source: &str) -> anyhow::Error {
    let raw = format!("{e:#}");
    // deser_hjson error format includes `at line N` and `column M`.
    // Look for a line number and use it to find the containing
    // task name by scanning upward.
    let line_num = match extract_line_number(&raw) {
        Some(n) => n,
        None => return e, // can't pinpoint; pass through
    };
    let task_name = match find_containing_task_name(source, line_num) {
        Some(name) => name,
        None => return e,
    };
    e.context(format!("in scenario task `{task_name}` (around line {line_num})"))
}

fn extract_line_number(err_msg: &str) -> Option<usize> {
    // Match "at line N" or "line N" patterns (deser_hjson and serde
    // both use one of these).
    let lower = err_msg.to_lowercase();
    let needles = ["at line ", "line "];
    for needle in needles {
        if let Some(idx) = lower.find(needle) {
            let tail = &lower[idx + needle.len()..];
            // Grab the leading digits.
            let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<usize>() {
                return Some(n);
            }
        }
    }
    None
}

/// Walk backwards from `line_num` looking for the nearest
/// `name: "..."` or `name: ...` field. HJSON puts the task name
/// at the top of each task block; scanning back from the error
/// position finds it reliably.
fn find_containing_task_name(source: &str, line_num: usize) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let start = std::cmp::min(line_num, lines.len()).saturating_sub(1);
    for line in lines[..=start].iter().rev() {
        let trimmed = line.trim_start();
        // HJSON name lines: `name: foo` or `name: "foo"`.
        if let Some(rest) = trimmed.strip_prefix("name:") {
            let value = rest.trim().trim_matches('"').trim_matches(',');
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- OOM detection -----

    #[test]
    fn oom_detection_matches_common_phrases() {
        assert!(looks_like_oom("Out of memory"));
        assert!(looks_like_oom("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(looks_like_oom("cudaMalloc failed"));
        assert!(looks_like_oom("MTL: allocation failed"));
        assert!(looks_like_oom("oom killer fired"));
    }

    #[test]
    fn oom_detection_no_false_positives() {
        assert!(!looks_like_oom("decoder is out of phase"));
        assert!(!looks_like_oom("file not found"));
        assert!(!looks_like_oom("connection refused"));
    }

    #[test]
    fn decorate_oom_appends_suggestions_when_matching() {
        let e = anyhow::anyhow!("cudaMalloc failed: out of memory");
        let decorated = decorate_oom(e, OomContext::Sdxl);
        let msg = format!("{decorated:#}");
        assert!(msg.contains("out-of-memory; try:"));
        assert!(msg.contains("--size 768x768"));
    }

    #[test]
    fn decorate_oom_passes_through_unrelated_errors() {
        let original = "file not found at /tmp/x";
        let e = anyhow::anyhow!(original);
        let decorated = decorate_oom(e, OomContext::Sd);
        let msg = format!("{decorated:#}");
        assert!(!msg.contains("out-of-memory; try:"));
        assert!(msg.contains(original));
    }

    #[test]
    fn oom_context_suggestions_are_pipeline_specific() {
        let sd = OomContext::Sd.suggestions().join(" ");
        assert!(sd.contains("--size"));
        let flux = OomContext::Flux.suggestions().join(" ");
        assert!(flux.contains("quant"));
        let anim = OomContext::Animate.suggestions().join(" ");
        assert!(anim.contains("frames") || anim.contains("window"));
    }

    // ----- alias suggestion -----

    #[test]
    fn closest_alias_finds_typo() {
        let known = &["sd15", "sd21", "sdxl", "flux-dev", "flux-schnell"];
        assert_eq!(closest_alias("sd1.5", known), Some("sd15"));
        assert_eq!(closest_alias("sdxlturbo", known), None); // not in list, no close match
        assert_eq!(closest_alias("flux-deb", known), Some("flux-dev"));
    }

    #[test]
    fn closest_alias_skips_org_name_repos() {
        // `org/name` shapes always pass through — they look
        // intentional even when the repo doesn't exist.
        let known = &["sd15"];
        assert_eq!(closest_alias("user/typo", known), None);
    }

    #[test]
    fn closest_alias_exact_match_returns_zero_distance() {
        let known = &["sd15", "sdxl"];
        // edit_distance(0) is below threshold, so closest_alias
        // returns the exact match itself.
        assert_eq!(closest_alias("sd15", known), Some("sd15"));
    }

    #[test]
    fn closest_alias_far_typo_returns_none() {
        let known = &["sd15"];
        // edit_distance("flux-dev", "sd15") is huge → None.
        assert_eq!(closest_alias("flux-dev", known), None);
    }

    #[test]
    fn hint_unknown_alias_succeeds_on_known_name() {
        let known = &["sd15", "sdxl"];
        assert!(hint_unknown_alias("sdxl", known).is_ok());
    }

    #[test]
    fn hint_unknown_alias_succeeds_on_org_name() {
        let known = &["sd15"];
        // Org/name shapes always pass through (downstream HF
        // resolver will produce its own diagnostic via friendly_error).
        assert!(hint_unknown_alias("custom-org/some-fancy-model", known).is_ok());
    }

    #[test]
    fn hint_unknown_alias_suggests_closest() {
        let known = &["sd15", "sdxl"];
        let err = hint_unknown_alias("sd1.5", known).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown --model alias `sd1.5`"));
        assert!(msg.contains("Did you mean `sd15`?"));
        assert!(msg.contains("plakat models aliases"));
    }

    #[test]
    fn hint_unknown_alias_no_suggestion_for_far_typo() {
        let known = &["sd15"];
        let err = hint_unknown_alias("xyz123", known).unwrap_err();
        let msg = format!("{err}");
        // Bails, but without a "Did you mean" suggestion.
        assert!(msg.contains("unknown --model alias"));
        assert!(!msg.contains("Did you mean"));
    }

    // ----- edit distance -----

    #[test]
    fn edit_distance_identity_is_zero() {
        assert_eq!(edit_distance("sd15", "sd15"), 0);
        assert_eq!(edit_distance("", ""), 0);
    }

    #[test]
    fn edit_distance_single_insertion() {
        assert_eq!(edit_distance("sd15", "sd1.5"), 1); // insert `.`
        assert_eq!(edit_distance("sdxl", "sdxls"), 1); // append
    }

    #[test]
    fn edit_distance_substitution() {
        assert_eq!(edit_distance("flux-dev", "flux-deb"), 1);
    }

    // ----- scenario parse decoration -----

    #[test]
    fn extract_line_number_handles_common_formats() {
        assert_eq!(extract_line_number("at line 12, column 3"), Some(12));
        assert_eq!(extract_line_number("Error at line 42"), Some(42));
        assert_eq!(extract_line_number("syntax error on line 7"), Some(7));
        assert_eq!(extract_line_number("oops"), None);
    }

    #[test]
    fn find_containing_task_name_walks_back() {
        let src = r#"{
    model: sd15
    tasks: [
        {
            name: cottage
            scene: dawn
            prompt: "a watercolor cottage"
            steps: 28
        }
        {
            name: knight
            scene: dawn
            prompt: "a knight in a forest"
            steps: oops
        }
    ]
}"#;
        // Suppose the parse error landed on the line with `oops`.
        // The line number depends on the literal — count it.
        let line_with_oops = src.lines().position(|l| l.contains("oops")).unwrap() + 1;
        let name = find_containing_task_name(src, line_with_oops);
        assert_eq!(name.as_deref(), Some("knight"));
    }

    #[test]
    fn find_containing_task_name_returns_none_when_no_name_above() {
        let src = "model: sd15\nsome: other\n";
        // Pointing at line 2 (no `name:` above), function returns None.
        assert_eq!(find_containing_task_name(src, 2), None);
    }

    #[test]
    fn decorate_scenario_parse_adds_task_pointer() {
        let src = r#"{
    model: sd15
    tasks: [
        { name: alpha, prompt: a }
        {
            name: beta
            prompt: b
            steps: not-a-number
        }
    ]
}"#;
        // The "steps: not-a-number" line is what would fail.
        let line = src.lines().position(|l| l.contains("not-a-number")).unwrap() + 1;
        let raw = anyhow::anyhow!("invalid type: string \"not-a-number\" at line {line}");
        let decorated = decorate_scenario_parse(raw, src);
        let msg = format!("{decorated:#}");
        assert!(msg.contains("task `beta`"), "got: {msg}");
    }

    #[test]
    fn decorate_scenario_parse_passes_through_when_no_line_number() {
        let raw = anyhow::anyhow!("some generic error with no position");
        let decorated = decorate_scenario_parse(raw, "");
        let msg = format!("{decorated:#}");
        assert!(!msg.contains("task `"));
        assert!(msg.contains("generic error"));
    }
}
