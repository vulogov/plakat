//! `prompts.txt` block parser. Blank-line-separated blocks; each block is
//! free-text lines + `key: value` command lines. The first block is the global
//! block iff it has no free text. `#`-prefixed lines are comments anywhere.

use anyhow::{Result, bail};

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
    let global_spec_task = doc.global.as_ref().is_some_and(|g| declares_map_task(g) || declares_bookart_task(g));
    for (i, s) in doc.scenes.iter().enumerate() {
        if !s.has_free_text() && !global_spec_task && !declares_map_task(s) && !declares_bookart_task(s) {
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
