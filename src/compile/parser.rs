//! `prompts.txt` block parser. Blank-line-separated blocks; each block is
//! free-text lines + `key: value` command lines. The first block is the global
//! block iff it has no free text. `#`-prefixed lines are comments anywhere.

use anyhow::{Context, Result, bail};

/// One parsed block: ordered commands (key, value — duplicates preserved) and
/// free-text lines (joined later with a space).
#[derive(Debug, Clone, Default)]
pub struct Block {
    pub commands: Vec<(String, String)>,
    pub free_text: Vec<String>,
    /// 1-based source line of the block's first line (for diagnostics).
    pub line_start: usize,
}

impl Block {
    /// All values for a command key, in occurrence order.
    pub fn values<'a>(&'a self, key: &str) -> impl Iterator<Item = &'a str> {
        self.commands
            .iter()
            .filter(move |(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn has_free_text(&self) -> bool {
        !self.free_text.is_empty()
    }

    fn is_empty(&self) -> bool {
        self.commands.is_empty() && self.free_text.is_empty()
    }
}

/// A parsed document: an optional global block + the scene blocks.
#[derive(Debug, Clone, Default)]
pub struct Document {
    pub global: Option<Block>,
    pub scenes: Vec<Block>,
}

/// Try to read a line as a command. Returns `(key, value)` when the line is
/// `ident: value` with `ident = [A-Za-z_][A-Za-z0-9_-]*` at column 0; else None
/// (the line is free text). Matches the RFC's `^([a-zA-Z_][a-zA-Z0-9_-]*):` rule.
pub fn parse_command_line(line: &str) -> Option<(String, String)> {
    let colon = line.find(':')?;
    let key = &line[..colon];
    if key.is_empty() {
        return None;
    }
    let mut chars = key.chars();
    let first = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return None;
    }
    let value = line[colon + 1..].trim().to_string();
    Some((key.to_string(), value))
}

/// **`@include` pre-pass** (RFC FACESWAP-4 C4; globs + params 6.22 E2): inline any line of the form
/// `@include <path> [key=value …]` with the contents of that file (relative to `base`), recursively
/// (depth-guarded). The path may **glob** with a single `*` in the final component (`scenes/*.txt`,
/// sorted); trailing `key=value` params substitute `${key}` in the included text. Runs before the block
/// parse. `@include "quoted path"` is accepted.
pub fn expand_includes(input: &str, base: &std::path::Path, depth: usize) -> Result<String> {
    if depth > 16 {
        bail!("compile: @include nested too deeply (>16 — a cycle?)");
    }
    let mut out = String::new();
    for line in input.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("@include ").or_else(|| t.strip_prefix("@include\t")) {
            let (path, params) = parse_include_args(rest);
            let joined = base.join(&path);
            let files = if path.contains('*') { glob_final_star(&joined)? } else { vec![joined] };
            if files.is_empty() {
                bail!("compile: @include {path} matched no files");
            }
            for full in files {
                let content = std::fs::read_to_string(&full)
                    .with_context(|| format!("compile: @include {}", full.display()))?;
                let content = apply_include_params(&content, &params);
                let child_base = full.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| base.to_path_buf());
                let expanded = expand_includes(&content, &child_base, depth + 1)?;
                out.push_str(&expanded);
                if !expanded.ends_with('\n') {
                    out.push('\n');
                }
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(out)
}

/// Split an `@include` argument tail into `(path, [key=value…])`. The path may be `"quoted"` (allowing
/// spaces); everything after it is `key=value` params.
fn parse_include_args(rest: &str) -> (String, Vec<(String, String)>) {
    let rest = rest.trim();
    let (path, remainder) = if let Some(r) = rest.strip_prefix('"') {
        match r.find('"') {
            Some(end) => (r[..end].to_string(), &r[end + 1..]),
            None => (r.to_string(), ""),
        }
    } else {
        let mut it = rest.splitn(2, char::is_whitespace);
        (it.next().unwrap_or("").to_string(), it.next().unwrap_or(""))
    };
    let params = remainder
        .split_whitespace()
        .filter_map(|tok| tok.split_once('=').map(|(k, v)| (k.to_string(), v.trim_matches('"').to_string())))
        .collect();
    (path, params)
}

/// Substitute `${key}` occurrences with the param values.
fn apply_include_params(content: &str, params: &[(String, String)]) -> String {
    let mut s = content.to_string();
    for (k, v) in params {
        s = s.replace(&format!("${{{k}}}"), v);
    }
    s
}

/// Minimal glob: a single `*` in the FINAL path component (`dir/*.txt`, `dir/scene-*`, `dir/*`). Returns
/// the matching files, sorted. No external dep.
fn glob_final_star(pat: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let dir = pat.parent().unwrap_or_else(|| std::path::Path::new("."));
    let fname = pat.file_name().and_then(|f| f.to_str()).unwrap_or("*");
    let (prefix, suffix) = fname.split_once('*').unwrap_or((fname, ""));
    let mut matches: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("compile: @include glob dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .filter(|p| {
            let n = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
            n.len() >= prefix.len() + suffix.len() && n.starts_with(prefix) && n.ends_with(suffix)
        })
        .collect();
    matches.sort();
    Ok(matches)
}

/// Parse a whole `prompts.txt` string.
pub fn parse(input: &str) -> Result<Document> {
    // Split into blocks on runs of blank lines; drop comment lines first but
    // keep line numbers honest for diagnostics.
    let mut blocks: Vec<Block> = Vec::new();
    let mut cur = Block::default();
    let mut cur_started = false;

    for (i, raw) in input.lines().enumerate() {
        let lineno = i + 1;
        let trimmed = raw.trim_end();
        // Comments: a line whose first non-space char is '#'.
        if trimmed.trim_start().starts_with('#') {
            continue;
        }
        if trimmed.trim().is_empty() {
            // Blank line → block boundary (collapses consecutive blanks).
            if cur_started && !cur.is_empty() {
                blocks.push(std::mem::take(&mut cur));
            }
            cur = Block::default();
            cur_started = false;
            continue;
        }
        if !cur_started {
            cur.line_start = lineno;
            cur_started = true;
        }
        match parse_command_line(trimmed) {
            Some((k, v)) => cur.commands.push((k, v)),
            None => cur.free_text.push(trimmed.trim().to_string()),
        }
    }
    if cur_started && !cur.is_empty() {
        blocks.push(cur);
    }

    if blocks.is_empty() {
        bail!("compile: no blocks found (empty prompts.txt?)");
    }

    // First block is global iff it has commands and NO free text. Otherwise the
    // document has no global block and every block (including the first) is a scene.
    let mut doc = Document::default();
    let first_is_global = !blocks[0].has_free_text() && !blocks[0].commands.is_empty();
    let mut iter = blocks.into_iter();
    if first_is_global {
        doc.global = Some(iter.next().unwrap());
    }
    doc.scenes = iter.collect();

    if doc.scenes.is_empty() {
        bail!("compile: no scene blocks (a scene block needs free-text description)");
    }
    // A scene block must carry some free text — a commands-only block past the
    // global slot is almost always a misplaced global / a missing blank line.
    // Exception (MAP-4): a `type: map` task renders from its spec, not a prompt, so
    // a map block (or any block when the global declares `type: map`) needs none.
    let global_spec_task = doc.global.as_ref().is_some_and(|g| declares_map_task(g) || declares_bookart_task(g) || declares_texture_task(g) || declares_comic_task(g) || declares_product_task(g) || declares_faceswap_task(g) || declares_fractal_task(g));
    for (i, s) in doc.scenes.iter().enumerate() {
        if !s.has_free_text() && !global_spec_task && !declares_map_task(s) && !declares_bookart_task(s) && !declares_texture_task(s) && !declares_comic_task(s) && !declares_product_task(s) && !declares_faceswap_task(s) && !declares_fractal_task(s) {
            bail!(
                "compile: scene block #{} (line {}) has commands but no description text — \
                 a stray blank line, or a global block not placed first?",
                i + 1,
                s.line_start
            );
        }
    }
    Ok(doc)
}

/// Does this block declare a `map` task (so it may omit a prose description)?
fn declares_map_task(b: &Block) -> bool {
    b.commands.iter().any(|(k, v)| {
        (k == "type" && v.eq_ignore_ascii_case("map")) || k.starts_with("map-")
    })
}

/// Does this block declare a `bookart` task (6.1.0 A3)? Prose is optional — a procedural ornament has
/// no prompt.
fn declares_bookart_task(b: &Block) -> bool {
    b.commands.iter().any(|(k, v)| {
        (k == "type" && v.eq_ignore_ascii_case("bookart")) || k.starts_with("bookart-")
    })
}

/// Does this block declare a `texture` task (6.3.0 B7)? Prose is optional — an image-to-material task
/// (`texture-from`) needs no material prompt.
fn declares_texture_task(b: &Block) -> bool {
    b.commands.iter().any(|(k, v)| {
        (k == "type" && v.eq_ignore_ascii_case("texture")) || k.starts_with("texture-")
    })
}

/// Does this block declare a `comic` task (6.8.0 P4)? Prose is optional — a comic renders from its
/// `ComicSpec`, not a page prompt.
fn declares_comic_task(b: &Block) -> bool {
    b.commands.iter().any(|(k, v)| {
        (k == "type" && v.eq_ignore_ascii_case("comic")) || k.starts_with("comic-")
    })
}

/// Does this block declare a `product` task (6.9.0 P4)? Prose (if any) is the subject prompt.
fn declares_product_task(b: &Block) -> bool {
    b.commands.iter().any(|(k, v)| {
        (k == "type" && v.eq_ignore_ascii_case("product")) || k.starts_with("product-")
    })
}

/// Does this block declare a `faceswap` task (6.22.0 FACESWAP-4)? It renders from `scene` + `source`,
/// not a text prompt — so, like the other spec-tasks, a faceswap block needs no free text.
fn declares_faceswap_task(b: &Block) -> bool {
    b.commands.iter().any(|(k, v)| {
        (k == "type" && (v.eq_ignore_ascii_case("faceswap") || v.eq_ignore_ascii_case("face-swap"))) || k.starts_with("faceswap-")
    })
}

/// Does this block declare a `fractal` task (6.22.0 D1)? A fractal renders from its spec, not a prompt.
fn declares_fractal_task(b: &Block) -> bool {
    b.commands.iter().any(|(k, v)| {
        (k == "type" && (v.eq_ignore_ascii_case("fractal") || v.eq_ignore_ascii_case("fractals"))) || k.starts_with("fractal-")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_command_vs_free_text() {
        assert_eq!(parse_command_line("model: sdxl"), Some(("model".into(), "sdxl".into())));
        assert_eq!(parse_command_line("header:wide shot,"), Some(("header".into(), "wide shot,".into())));
        // Free text: leading space, internal colon, or non-ident key.
        assert!(parse_command_line("A vast frozen tundra").is_none());
        assert!(parse_command_line("He said: hello").is_none(), "space in key");
        assert!(parse_command_line("  model: x").is_none(), "leading space = free text");
        assert!(parse_command_line(":nope").is_none());
    }

    #[test]
    fn parses_global_and_scenes() {
        let doc = parse(
            "# a comment\nmodel: sdxl\nstyle: cinematic\n\nheader: wide shot,\nA frozen tundra.\nfooter: 8k\n\nA harbour at dawn.\nseed: 42\n",
        )
        .unwrap();
        let g = doc.global.as_ref().unwrap();
        assert_eq!(g.values("model").collect::<Vec<_>>(), vec!["sdxl"]);
        assert_eq!(doc.scenes.len(), 2);
        assert_eq!(doc.scenes[0].free_text, vec!["A frozen tundra."]);
        assert_eq!(doc.scenes[0].values("header").collect::<Vec<_>>(), vec!["wide shot,"]);
        assert_eq!(doc.scenes[1].values("seed").collect::<Vec<_>>(), vec!["42"]);
    }

    #[test]
    fn faceswap_block_needs_no_free_text() {
        // C1 (FACESWAP-4): a `type: faceswap` block renders from scene+source, not a prompt — it must
        // parse without a description (previously rejected as "commands but no description").
        // A global block first, then the faceswap scene block (commands only, no description).
        let doc = parse("model: sdxl\n\nname: swap\ntype: faceswap\nfaceswap-scene: a.png\nfaceswap-source: b.png\n").unwrap();
        assert_eq!(doc.scenes.len(), 1);
        assert!(doc.scenes[0].free_text.is_empty());
        assert_eq!(doc.scenes[0].values("type").collect::<Vec<_>>(), vec!["faceswap"]);
    }

    #[test]
    fn expand_includes_inlines_and_leaves_plain_text() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("body.txt"), "A frozen tundra.\nseed: 7\n").unwrap();
        let input = "model: sdxl\n\n@include body.txt\n";
        let out = expand_includes(input, dir.path(), 0).unwrap();
        assert!(out.contains("A frozen tundra."), "included body inlined");
        assert!(out.contains("seed: 7"));
        assert!(out.contains("model: sdxl"), "surrounding lines kept");
        assert!(!out.contains("@include"), "the directive is consumed");
        // A missing include is a clear error.
        assert!(expand_includes("@include nope.txt\n", dir.path(), 0).is_err());
    }

    #[test]
    fn expand_includes_globs_and_substitutes_params() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "scene ${who} A\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "scene ${who} B\n").unwrap();
        std::fs::write(dir.path().join("skip.md"), "not matched\n").unwrap();
        // Glob `*.txt` (sorted) + a `who=` param substituted into `${who}`.
        let out = expand_includes("@include *.txt who=alice\n", dir.path(), 0).unwrap();
        assert!(out.contains("scene alice A"), "a.txt inlined + param: {out}");
        assert!(out.contains("scene alice B"), "b.txt inlined too (glob)");
        assert!(!out.contains("not matched"), "*.txt didn't match the .md");
        assert!(!out.contains("${who}"), "param fully substituted");
        // Position: a.txt (sorted) before b.txt.
        assert!(out.find("A").unwrap() < out.find("B").unwrap(), "sorted");
    }

    #[test]
    fn first_block_with_free_text_is_a_scene_not_global() {
        // No global block: the first block has description text.
        let doc = parse("model: sd15\nA lone tower on a cliff.\n").unwrap();
        assert!(doc.global.is_none());
        assert_eq!(doc.scenes.len(), 1);
        assert_eq!(doc.scenes[0].values("model").collect::<Vec<_>>(), vec!["sd15"]);
    }

    #[test]
    fn collapses_multiple_blank_lines() {
        let doc = parse("First scene text.\n\n\n\nSecond scene text.\n").unwrap();
        assert!(doc.global.is_none());
        assert_eq!(doc.scenes.len(), 2);
    }

    #[test]
    fn rejects_commands_only_scene() {
        // A second block with commands but no description is an error.
        let err = parse("A real scene.\n\nseed: 5\ncount: 2\n").unwrap_err();
        assert!(err.to_string().contains("no description"), "got: {err}");
    }
}
