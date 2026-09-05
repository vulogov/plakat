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
    // `.` is allowed so namespaced keys like `component.stall` parse as commands (6.26.x). A
    // leading `.` is still rejected (first-char check above), and lines with spaces before the
    // colon stay free text.
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.') {
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

/// A multiline-value fence, if `s` is exactly one (HJSON-style `'''` or `"""`).
fn multiline_fence(s: &str) -> Option<&'static str> {
    match s.trim() {
        "'''" => Some("'''"),
        "\"\"\"" => Some("\"\"\""),
        _ => None,
    }
}

/// Collect a fenced multiline value from `lines[start..]` until the closing `fence` line. Interior lines
/// (including blank ones, which are NOT block boundaries inside a fence) are trimmed and joined with a
/// single space, so the value stays a clean one-line string for the emitted HJSON. Returns the value and
/// the index of the line AFTER the closing fence.
fn collect_multiline(lines: &[&str], start: usize, fence: &str) -> Result<(String, usize)> {
    let mut parts: Vec<String> = Vec::new();
    let mut j = start;
    while j < lines.len() {
        if lines[j].trim() == fence {
            let val = parts.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
            return Ok((val, j + 1));
        }
        parts.push(lines[j].trim().to_string());
        j += 1;
    }
    bail!("compile: unterminated multiline value — missing closing {fence}");
}

/// Parse a whole `prompts.txt` string.
pub fn parse(input: &str) -> Result<Document> {
    // Split into blocks on runs of blank lines; drop comment lines first but
    // keep line numbers honest for diagnostics.
    let mut blocks: Vec<Block> = Vec::new();
    let mut cur = Block::default();
    let mut cur_started = false;

    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let lineno = i + 1;
        let raw = lines[i];
        let trimmed = raw.trim_end();
        // Comments: a line whose first non-space char is '#'.
        if trimmed.trim_start().starts_with('#') {
            i += 1;
            continue;
        }
        if trimmed.trim().is_empty() {
            // Blank line → block boundary (collapses consecutive blanks).
            if cur_started && !cur.is_empty() {
                blocks.push(std::mem::take(&mut cur));
            }
            cur = Block::default();
            cur_started = false;
            i += 1;
            continue;
        }
        if !cur_started {
            cur.line_start = lineno;
            cur_started = true;
        }
        match parse_command_line(trimmed) {
            Some((k, v)) => {
                // Multiline value (HJSON-style): `key: '''` opens a fence on the same line, or `key:` with an
                // empty value opens it on the NEXT line (`'''` alone). Everything up to the closing fence is
                // the value — command-looking or blank interior lines are literal, not parsed. This is what
                // lets a `component.<name>:` span several lines.
                if let Some(fence) = multiline_fence(&v) {
                    let (val, next) = collect_multiline(&lines, i + 1, fence)?;
                    cur.commands.push((k, val));
                    i = next;
                    continue;
                } else if v.is_empty() {
                    if let Some(fence) = lines.get(i + 1).and_then(|l| multiline_fence(l)) {
                        let (val, next) = collect_multiline(&lines, i + 2, fence)?;
                        cur.commands.push((k, val));
                        i = next;
                        continue;
                    }
                    cur.commands.push((k, v));
                } else {
                    cur.commands.push((k, v));
                }
            }
            None => cur.free_text.push(trimmed.trim().to_string()),
        }
        i += 1;
    }
    if cur_started && !cur.is_empty() {
        blocks.push(cur);
    }

    if blocks.is_empty() {
        bail!("compile: no blocks found (empty prompts.txt?)");
    }

    // Global fragments vs scenes. A block that is ONLY commands — no free-text prose, no `composition:`,
    // no `relate:`, and no spec-task directive (map/bookart/…) — is a GLOBAL FRAGMENT: `model:`/`style:`/
    // `naturalize:`/`component.*` config that carries no scene of its own. Merge EVERY such fragment into
    // one global block; the rest are scenes. This makes `@include` robust — global/component fragments can
    // be split across blank-line boundaries (a blank line at the end of an included file, several component
    // files) without any of them becoming a description-less "scene". (Previously only the FIRST block could
    // be global, so a component/global fragment in block 2+ errored.)
    let mut doc = Document::default();
    let mut global = Block::default();
    let mut scenes: Vec<Block> = Vec::new();
    for b in blocks {
        if is_global_fragment(&b) {
            if global.line_start == 0 {
                global.line_start = b.line_start;
            }
            global.commands.extend(b.commands);
        } else {
            scenes.push(b);
        }
    }
    if !global.commands.is_empty() {
        doc.global = Some(global);
    }
    doc.scenes = scenes;

    if doc.scenes.is_empty() {
        bail!("compile: no scene blocks (a scene block needs free-text description)");
    }
    // A scene block must carry some free text — a commands-only block past the
    // global slot is almost always a misplaced global / a missing blank line.
    // Exception (MAP-4): a `type: map` task renders from its spec, not a prompt, so
    // a map block (or any block when the global declares `type: map`) needs none.
    let global_spec_task = doc.global.as_ref().is_some_and(|g| declares_map_task(g) || declares_bookart_task(g) || declares_texture_task(g) || declares_comic_task(g) || declares_product_task(g) || declares_faceswap_task(g) || declares_fractal_task(g));
    for (i, s) in doc.scenes.iter().enumerate() {
        // A `composition:` also gives the block content (its prompt comes from components), so a
        // composition-only block is valid — free text is optional (compose, then prose).
        if !s.has_free_text() && !declares_composition(s) && !declares_relations(s) && !global_spec_task && !declares_map_task(s) && !declares_bookart_task(s) && !declares_texture_task(s) && !declares_comic_task(s) && !declares_product_task(s) && !declares_faceswap_task(s) && !declares_fractal_task(s) {
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

/// Does this block declare a `composition:` (so its prompt comes from components — no prose needed)?
fn declares_composition(b: &Block) -> bool {
    b.values("composition").next().is_some()
}

/// Does this block declare a `relate:` (6.28)? Its prompt gets a grounding clause built from the related
/// components' descriptions, so — like `composition:` — free text is optional (relate, then optional prose).
fn declares_relations(b: &Block) -> bool {
    b.values("relate").next().is_some()
}

/// A GLOBAL FRAGMENT: commands only, with nothing that makes it a renderable scene (no prose, no
/// `composition:`, no `relate:`, no spec-task). These are `model:`/`style:`/`naturalize:`/`component.*`
/// config that `@include` may split across blank-line boundaries; they all merge into the global block.
fn is_global_fragment(b: &Block) -> bool {
    !b.has_free_text()
        && !declares_composition(b)
        && !declares_relations(b)
        && !declares_map_task(b)
        && !declares_bookart_task(b)
        && !declares_texture_task(b)
        && !declares_comic_task(b)
        && !declares_product_task(b)
        && !declares_faceswap_task(b)
        && !declares_fractal_task(b)
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
    fn multiline_value_fence_on_next_line() {
        // `key:` then a `'''` line — the user's form. Interior command-looking / blank lines are literal;
        // the block stays global (the fenced lines are the VALUE, not free text).
        let doc = parse(
            "model: sd35\ncomponent.street:\n'''\n(cobblestone:1.5) a clean street\nwith flower boxes\n'''\n\nA scene.\n",
        )
        .unwrap();
        let g = doc.global.as_ref().unwrap();
        assert_eq!(
            g.values("component.street").collect::<Vec<_>>(),
            vec!["(cobblestone:1.5) a clean street with flower boxes"]
        );
        assert!(g.free_text.is_empty(), "fenced lines are the value, not free text");
        assert_eq!(doc.scenes.len(), 1);
    }

    #[test]
    fn multiline_value_fence_inline_and_unterminated() {
        // `key: '''` opens the fence on the same line.
        let doc = parse("component.x: '''\nline one\nline two\n'''\n\nA scene.\n").unwrap();
        assert_eq!(doc.scenes.len(), 1);
        // (component.x lives in the first, global-less... actually first block has the component + no free
        // text → it's global.)
        let g = doc.global.as_ref().unwrap();
        assert_eq!(g.values("component.x").collect::<Vec<_>>(), vec!["line one line two"]);
        // A missing closing fence is a clear error.
        assert!(parse("component.x:\n'''\nno close here\n").is_err());
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
    fn global_fragments_merge_across_blank_lines() {
        // The @include shape: global config, a component block, and more component blocks are separated by
        // blank lines (from included files) — they ALL merge into the global block; only the composition/
        // relate/prose block is a scene. (Previously a component block in slot 2+ errored as a scene.)
        let doc = parse(
            "model: sd35\nstyle: impressionist\n\n\
             component.sky: a clear sky\n\n\
             component.sun: a low sun\n\n\
             composition: component.sky, component.sun\nrelate: sun on sky\nsome prose.\n",
        )
        .unwrap();
        assert_eq!(doc.scenes.len(), 1, "only the composition/relate block is a scene");
        let g = doc.global.as_ref().expect("global merged from all fragments");
        assert_eq!(g.values("model").collect::<Vec<_>>(), vec!["sd35"]);
        assert!(g.commands.iter().any(|(k, _)| k == "component.sky"));
        assert!(g.commands.iter().any(|(k, _)| k == "component.sun"));
    }

    #[test]
    fn relate_block_needs_no_free_text() {
        // 6.28: a scene block whose content is `relate:` (a grounding clause built from components) is
        // valid without free-text prose — like `composition:`. Global block first, then the relate scene.
        let doc = parse(
            "model: sd35\ncomponent.tram: a tram\ncomponent.rails: rails\n\nrelate: tram on rails\n",
        )
        .unwrap();
        assert_eq!(doc.scenes.len(), 1);
        assert!(doc.scenes[0].free_text.is_empty());
        assert_eq!(doc.scenes[0].values("relate").collect::<Vec<_>>(), vec!["tram on rails"]);
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
    fn commands_only_block_merges_into_global() {
        // 6.28: a commands-only block (no prose/composition/relate/spec-task) is a GLOBAL FRAGMENT — it
        // merges into the global block rather than erroring as a description-less scene. (Scene-specific
        // directives stay attached to their scene when written in the SAME block, no blank line between.)
        let doc = parse("A real scene.\n\nseed: 5\ncount: 2\n").unwrap();
        assert_eq!(doc.scenes.len(), 1);
        assert!(doc.scenes[0].has_free_text());
        let g = doc.global.as_ref().expect("trailing commands merged to global");
        assert_eq!(g.values("seed").collect::<Vec<_>>(), vec!["5"]);
    }
}
