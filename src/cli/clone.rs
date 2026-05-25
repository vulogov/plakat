//! `plakat clone PNG` — read a generated PNG's metadata and print
//! a re-runnable `plakat generate` shell command.
//!
//! Pairs with `plakat metadata` (v0.18, read-only inspection):
//! `metadata` shows you the structured recipe; `clone` translates
//! that recipe into the CLI invocation that would re-create the
//! image (modulo wildcard re-rolls and other entropy sources).
//!
//! Two input sources, in priority order:
//!
//! 1. JSON sidecar at `<stem>.json` — written by plakat v0.17+.
//!    Carries every flag we'd need to synthesise the command
//!    losslessly.
//! 2. Auto1111 `parameters` PNG tEXt chunk. Best-effort parse of
//!    the common fields (prompt, negative, Steps / Sampler / CFG
//!    scale / Seed / Size / Model / ClipSkip). Civitai uploads +
//!    A1111 Web UI outputs land here.
//!
//! Output is always `plakat generate ...` regardless of the
//! original mode. img2img / inpaint / animate clones lose the
//! input-image / mask / animate-endpoint context (we don't carry
//! those into the recipe); a note in the output flags this.
//!
//! Shell-safe single-quote escaping handles prompts containing
//! arbitrary punctuation.

use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::imaging::metadata::GenerationMetadata;

#[derive(Args, Debug)]
pub struct CloneArgs {
    /// PNG to clone. The sibling `<stem>.json` sidecar is
    /// preferred when present; falls back to parsing the Auto1111
    /// `parameters` PNG tEXt chunk.
    pub path: PathBuf,

    /// Emit the command as a single line (no `\` line-breaks)
    /// — useful when piping into another shell or
    /// `xargs -I {}`. Default is the indented multi-line form
    /// that's easy to read + edit in place.
    #[arg(long, default_value_t = false)]
    pub one_line: bool,
}

pub async fn run(args: CloneArgs) -> Result<()> {
    if !args.path.exists() {
        anyhow::bail!("{}: no such file", args.path.display());
    }

    // Try the JSON sidecar first — it's the structured source.
    let sidecar = args.path.with_extension("json");
    let meta = if sidecar.exists() {
        let json = std::fs::read_to_string(&sidecar)
            .with_context(|| format!("reading {}", sidecar.display()))?;
        serde_json::from_str::<GenerationMetadata>(&json)
            .with_context(|| format!("parsing JSON sidecar {}", sidecar.display()))?
    } else {
        // Fall back to A1111 PNG tEXt chunk.
        match crate::imaging::io::read_parameters_chunk(&args.path)? {
            Some(text) => parse_a1111(&text).ok_or_else(|| {
                anyhow::anyhow!(
                    "{} has a `parameters` tEXt chunk but it doesn't parse as \
                     Auto1111 format. Try `plakat metadata {}` to inspect the raw \
                     content.",
                    args.path.display(),
                    args.path.display()
                )
            })?,
            None => anyhow::bail!(
                "{} has neither a JSON sidecar at {} nor a `parameters` PNG tEXt \
                 chunk. Not a plakat / A1111 / Civitai output, or written with \
                 --no-metadata.",
                args.path.display(),
                sidecar.display()
            ),
        }
    };

    let cmd = synthesize(&meta, args.one_line);
    println!("{cmd}");
    // Flag mode mismatch so users know `plakat generate` won't
    // reproduce an img2img / animate exactly.
    if let Some(mode) = meta.mode.as_deref() {
        if mode != "t2i" {
            eprintln!(
                "  note: original mode was {mode:?}; clone emits `plakat generate` \
                 (the input image / mask / animate endpoints aren't part of the \
                 recipe and aren't recoverable from the PNG alone)."
            );
        }
    }
    Ok(())
}

/// Build the shell command. `one_line` controls whether flags are
/// `\`-folded across multiple lines (default) or joined into a
/// single line.
fn synthesize(meta: &GenerationMetadata, one_line: bool) -> String {
    let mut parts: Vec<String> = vec!["plakat generate".to_string()];
    parts.push(shell_quote(&meta.prompt));
    if !meta.negative.is_empty() {
        parts.push(format!("--negative {}", shell_quote(&meta.negative)));
    }
    parts.push(format!("--model {}", shell_quote(&meta.model)));
    parts.push(format!("--seed {}", meta.seed));
    parts.push(format!("--size {}x{}", meta.width, meta.height));
    if meta.steps != 28 {
        parts.push(format!("--steps {}", meta.steps));
    }
    if (meta.guidance - 7.5).abs() > 1e-6 {
        parts.push(format!("--guidance {}", meta.guidance));
    }
    let sched_lower = meta.scheduler.to_lowercase();
    if sched_lower != "default" {
        parts.push(format!("--scheduler {sched_lower}"));
    }
    for lora in &meta.loras {
        parts.push(format!("--lora {}", shell_quote(lora)));
    }
    if let Some(scale) = meta.lora_scale {
        if (scale - 1.0).abs() > 1e-6 {
            parts.push(format!("--lora-scale {scale}"));
        }
    }
    if let Some(cs) = meta.clip_skip {
        if cs > 1 {
            parts.push(format!("--clip-skip {cs}"));
        }
    }

    if one_line {
        parts.join(" ")
    } else {
        // Indent + line-break each flag for readability. The
        // first two parts (`plakat generate "prompt"`) stay on
        // the first line; everything else gets its own line.
        let mut s = format!("{} {}", parts[0], parts[1]);
        for p in &parts[2..] {
            s.push_str(" \\\n    ");
            s.push_str(p);
        }
        s
    }
}

/// Shell single-quote escape. Single-quote wraps the value;
/// embedded single quotes use the standard `'\''` pattern.
/// Robust for arbitrary content (no shell metacharacter escapes
/// happen inside single quotes).
fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            // Close, escape, re-open.
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Minimal Auto1111 `parameters` parser. Maps the three-line
/// A1111 layout into a `GenerationMetadata`. Best-effort — A1111's
/// format is free-form and values can contain commas; we extract
/// what we can and leave the rest as defaults.
///
/// Returns `None` only when the input is empty / doesn't look
/// like A1111 format at all.
pub fn parse_a1111(text: &str) -> Option<GenerationMetadata> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }
    // Line 1: prompt.
    let prompt = lines[0].trim().to_string();
    if prompt.is_empty() {
        return None;
    }
    // Optional "Negative prompt: <text>" line.
    let mut negative = String::new();
    let mut param_line_idx = 1;
    if let Some(neg_line) = lines.get(1) {
        if let Some(rest) = neg_line.strip_prefix("Negative prompt:") {
            negative = rest.trim().to_string();
            param_line_idx = 2;
        }
    }
    // Remaining lines: comma-separated `key: value` pairs. Join
    // and parse — multi-line key:value tails are uncommon.
    let kv_text = lines[param_line_idx..].join(", ");
    let kvs = parse_kv_pairs(&kv_text);

    let mut meta = GenerationMetadata::new(
        prompt,
        kvs.get("Model").cloned().unwrap_or_else(|| "sd15".to_string()),
        kvs.get("Seed").and_then(|v| v.parse().ok()).unwrap_or(0),
        kvs.get("Steps").and_then(|v| v.parse().ok()).unwrap_or(28),
        kvs.get("CFG scale").and_then(|v| v.parse().ok()).unwrap_or(7.5),
        kvs.get("Sampler").cloned().unwrap_or_else(|| "default".to_string()),
        kvs.get("Size")
            .and_then(|v| parse_size_wxh(v).map(|(w, _h)| w))
            .unwrap_or(512),
        kvs.get("Size")
            .and_then(|v| parse_size_wxh(v).map(|(_w, h)| h))
            .unwrap_or(512),
    );
    meta.negative = negative;
    if let Some(cs) = kvs.get("Clip skip").and_then(|v| v.parse().ok()) {
        meta.clip_skip = Some(cs);
    }
    Some(meta)
}

/// Parse `key: value, key2: value2` pairs. Values can contain
/// commas; we use the next `KEY:` pattern as the splitter rather
/// than blindly splitting on commas.
fn parse_kv_pairs(text: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    // Simple state machine: walk char-by-char. Detect `<word>: `
    // (capitalised word followed by `: `) as a new key. Stash
    // the previous (key, value) pair when one's found.
    let mut current_key: Option<String> = None;
    let mut current_val = String::new();
    let mut word_buf = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // Detect ", " followed by a capital-letter sequence ending
        // in `: ` — the start of the next key.
        if c == ',' && i + 1 < chars.len() && chars[i + 1] == ' ' {
            // Look ahead for `: `.
            let mut j = i + 2;
            let mut potential_key = String::new();
            while j < chars.len() && (chars[j].is_alphabetic() || chars[j] == ' ') {
                potential_key.push(chars[j]);
                j += 1;
            }
            let potential_key = potential_key.trim().to_string();
            if j + 1 < chars.len()
                && chars[j] == ':'
                && chars[j + 1] == ' '
                && !potential_key.is_empty()
            {
                // Commit the current pair, start the new one.
                if let Some(k) = current_key.take() {
                    out.insert(k, current_val.trim().to_string());
                }
                current_key = Some(potential_key);
                current_val.clear();
                i = j + 2;
                continue;
            }
        }
        // First key on the line.
        if current_key.is_none() {
            if c == ':' && i + 1 < chars.len() && chars[i + 1] == ' ' {
                current_key = Some(word_buf.trim().to_string());
                word_buf.clear();
                i += 2;
                continue;
            }
            word_buf.push(c);
            i += 1;
            continue;
        }
        current_val.push(c);
        i += 1;
    }
    if let Some(k) = current_key.take() {
        out.insert(k, current_val.trim().to_string());
    }
    out
}

fn parse_size_wxh(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split(['x', 'X']).collect();
    if parts.len() != 2 {
        return None;
    }
    Some((parts[0].trim().parse().ok()?, parts[1].trim().parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_meta() -> GenerationMetadata {
        let mut m = GenerationMetadata::new(
            "a fox in tall grass",
            "sd15",
            42,
            28,
            7.5,
            "euler-a",
            512,
            768,
        );
        m.negative = "blurry".into();
        m
    }

    #[test]
    fn shell_quote_round_trips_simple() {
        assert_eq!(shell_quote("hello"), "'hello'");
        assert_eq!(shell_quote("with spaces"), "'with spaces'");
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        // "don't" → 'don'\''t'
        assert_eq!(shell_quote("don't"), "'don'\\''t'");
    }

    #[test]
    fn shell_quote_preserves_double_quotes_unchanged() {
        // Double quotes inside single-quoted bash strings are
        // literal — no escape needed.
        assert_eq!(shell_quote("a \"quote\""), "'a \"quote\"'");
    }

    #[test]
    fn synthesize_minimal_meta_emits_required_flags() {
        let m = mk_meta();
        let cmd = synthesize(&m, true);
        assert!(cmd.starts_with("plakat generate "));
        assert!(cmd.contains("'a fox in tall grass'"));
        assert!(cmd.contains("--negative 'blurry'"));
        assert!(cmd.contains("--model 'sd15'"));
        assert!(cmd.contains("--seed 42"));
        assert!(cmd.contains("--size 512x768"));
        assert!(cmd.contains("--scheduler euler-a"));
    }

    #[test]
    fn synthesize_omits_default_steps_and_guidance() {
        let m = mk_meta();
        let cmd = synthesize(&m, true);
        // 28 steps + 7.5 guidance are the GenerateArgs defaults —
        // suppress them from the synthesised command.
        assert!(!cmd.contains("--steps"));
        assert!(!cmd.contains("--guidance"));
    }

    #[test]
    fn synthesize_emits_non_default_steps() {
        let mut m = mk_meta();
        m.steps = 50;
        let cmd = synthesize(&m, true);
        assert!(cmd.contains("--steps 50"));
    }

    #[test]
    fn synthesize_emits_loras_in_order() {
        let mut m = mk_meta();
        m.loras = vec!["civitai:111".into(), "civitai:222:0.5".into()];
        let cmd = synthesize(&m, true);
        let a = cmd.find("civitai:111").unwrap();
        let b = cmd.find("civitai:222:0.5").unwrap();
        assert!(a < b, "lora order not preserved in {cmd}");
    }

    #[test]
    fn synthesize_emits_clip_skip_above_one() {
        let mut m = mk_meta();
        m.clip_skip = Some(2);
        let cmd = synthesize(&m, true);
        assert!(cmd.contains("--clip-skip 2"));
        m.clip_skip = Some(1);
        let cmd2 = synthesize(&m, true);
        assert!(!cmd2.contains("--clip-skip"), "clip-skip 1 should be suppressed");
    }

    #[test]
    fn synthesize_multiline_indents_each_flag() {
        let m = mk_meta();
        let cmd = synthesize(&m, false);
        assert!(cmd.contains(" \\\n    --negative"));
        assert!(cmd.contains(" \\\n    --model"));
    }

    #[test]
    fn parse_a1111_minimal_three_line() {
        let text = "a fox\nNegative prompt: blurry\nSteps: 28, Sampler: euler-a, \
                    CFG scale: 7.5, Seed: 42, Size: 512x768, Model: sd15";
        let m = parse_a1111(text).unwrap();
        assert_eq!(m.prompt, "a fox");
        assert_eq!(m.negative, "blurry");
        assert_eq!(m.seed, 42);
        assert_eq!(m.steps, 28);
        assert_eq!((m.width, m.height), (512, 768));
        assert_eq!(m.model, "sd15");
        assert_eq!(m.scheduler, "euler-a");
    }

    #[test]
    fn parse_a1111_no_negative_line() {
        let text = "a fox\nSteps: 20, Sampler: ddim, CFG scale: 6, Seed: 7, \
                    Size: 768x768, Model: sdxl";
        let m = parse_a1111(text).unwrap();
        assert_eq!(m.prompt, "a fox");
        assert!(m.negative.is_empty());
        assert_eq!(m.steps, 20);
        assert_eq!(m.model, "sdxl");
    }

    #[test]
    fn parse_a1111_empty_returns_none() {
        assert!(parse_a1111("").is_none());
        assert!(parse_a1111("   ").is_none());
    }

    #[test]
    fn parse_a1111_clip_skip_field() {
        let text =
            "a fox\nSteps: 28, Sampler: euler-a, CFG scale: 7.5, Seed: 1, \
             Size: 512x512, Model: sd15, Clip skip: 2";
        let m = parse_a1111(text).unwrap();
        assert_eq!(m.clip_skip, Some(2));
    }

    #[tokio::test]
    async fn run_bails_when_file_missing() {
        let args = CloneArgs {
            path: PathBuf::from("/tmp/plakat-clone-nope.png"),
            one_line: false,
        };
        let err = run(args).await.unwrap_err();
        assert!(format!("{err}").contains("no such file"));
    }
}
