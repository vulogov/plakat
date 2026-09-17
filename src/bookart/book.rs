//! The CAPSTONE: assemble a whole typeset BOOK as one compilable Typst file from a Markdown manuscript —
//! a title page (a `bookart title-page` artifact, `#include`d), chapter openers (an optional headpiece,
//! `CHAPTER N`, the title), body prose with a raised decorated initial + small-caps opening on each
//! chapter's first paragraph, running heads + page folios, an optional tailpiece per chapter, and a
//! colophon. Weight-free typography; the only images are the ornaments/plates the caller supplies.
//!
//! The manuscript is plain **Markdown**:
//!   * `# Title`  opens a **chapter** (text before the first one is front matter);
//!   * `## Head`  is an in-chapter **section head**;
//!   * `> quote`  lines are a **blockquote**;
//!   * `***` / `* * *` on its own line is a **scene break** (an asterism, or a divider ornament);
//!   * blank lines separate paragraphs; `*italic*`/`**bold**` (and `_`/`__`) are honoured inline.

use crate::bookart::typst::{esc, mm};

/// A body block within a chapter (or the front matter).
pub enum Block {
    /// A paragraph of prose (inline emphasis honoured).
    Para(String),
    /// An in-chapter section head (`##`).
    Section(String),
    /// A blockquote (`>` lines), each string a paragraph.
    Quote(Vec<String>),
    /// A scene break (`***`).
    SceneBreak,
}

/// One parsed chapter.
pub struct Chapter {
    pub title: String,
    pub blocks: Vec<Block>,
}

/// Parse a Markdown manuscript into (front-matter blocks, chapters).
pub fn parse_manuscript(md: &str) -> (Vec<Block>, Vec<Chapter>) {
    let mut front: Vec<Block> = Vec::new();
    let mut chapters: Vec<Chapter> = Vec::new();
    let mut para = String::new();
    let mut quote: Vec<String> = Vec::new();

    // Flush the pending paragraph / quote into the current target (front, else the last chapter).
    fn target<'a>(front: &'a mut Vec<Block>, chapters: &'a mut Vec<Chapter>) -> &'a mut Vec<Block> {
        match chapters.last_mut() {
            Some(ch) => &mut ch.blocks,
            None => front,
        }
    }
    macro_rules! flush {
        () => {{
            if !para.trim().is_empty() {
                target(&mut front, &mut chapters).push(Block::Para(para.trim().to_string()));
            }
            para.clear();
            if !quote.is_empty() {
                target(&mut front, &mut chapters).push(Block::Quote(std::mem::take(&mut quote)));
            }
        }};
    }

    for raw in md.lines() {
        let line = raw.trim_end();
        let t = line.trim_start();
        let scene = {
            let s: String = t.chars().filter(|c| !c.is_whitespace()).collect();
            !s.is_empty() && (s.chars().all(|c| c == '*') && s.len() >= 3 || s == "⁂")
        };
        if let Some(rest) = t.strip_prefix("# ") {
            flush!();
            chapters.push(Chapter { title: rest.trim().to_string(), blocks: Vec::new() });
        } else if let Some(rest) = t.strip_prefix("### ").or_else(|| t.strip_prefix("## ")) {
            flush!();
            target(&mut front, &mut chapters).push(Block::Section(rest.trim().to_string()));
        } else if scene {
            flush!();
            target(&mut front, &mut chapters).push(Block::SceneBreak);
        } else if let Some(q) = t.strip_prefix("> ").or_else(|| if t == ">" { Some("") } else { None }) {
            // A blockquote line: flush any open paragraph, accumulate quote paragraphs.
            if !para.trim().is_empty() {
                target(&mut front, &mut chapters).push(Block::Para(para.trim().to_string()));
                para.clear();
            }
            if q.trim().is_empty() {
                if let Some(last) = quote.last_mut() {
                    if !last.is_empty() {
                        quote.push(String::new());
                    }
                }
            } else {
                match quote.last_mut() {
                    Some(last) if !last.is_empty() => {
                        last.push(' ');
                        last.push_str(q.trim());
                    }
                    _ => quote.push(q.trim().to_string()),
                }
            }
        } else if line.trim().is_empty() {
            flush!();
        } else {
            if !quote.is_empty() {
                target(&mut front, &mut chapters).push(Block::Quote(std::mem::take(&mut quote)));
            }
            if !para.is_empty() {
                para.push(' ');
            }
            para.push_str(line.trim());
        }
    }
    flush!();
    (front, chapters)
}

/// A roman-ish chapter numeral (kept simple: 1–39 as roman, else the arabic number).
fn roman(mut n: u32) -> String {
    if n == 0 || n > 39 {
        return n.to_string();
    }
    let table = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")];
    let mut out = String::new();
    for (v, s) in table {
        while n >= v {
            out.push_str(s);
            n -= v;
        }
    }
    out
}

/// Split a paragraph into (first char, small-caps lead words, remainder) for the decorated opening.
fn split_opening(p: &str) -> (String, String, String) {
    let mut chars = p.chars();
    let first = chars.next().map(|c| c.to_string()).unwrap_or_default();
    let rest: String = chars.collect();
    let mut lead_end = rest.len();
    let mut words = 0;
    for (i, c) in rest.char_indices() {
        if c == ' ' {
            words += 1;
            if words >= 4 || i >= 28 {
                lead_end = i;
                break;
            }
        }
    }
    let (lead, tail) = rest.split_at(lead_end.min(rest.len()));
    (first, lead.to_string(), tail.to_string())
}

/// Options for [`book_typ`].
pub struct BookOpts<'a> {
    pub w_mm: f32,
    pub h_mm: f32,
    /// Running-head text (usually the book title), small-caps; empty = no head.
    pub running_head: &'a str,
    /// An `#include`-able title-page `.typ` (basename beside the book), rendered as the first leaf.
    pub title_page: Option<&'a str>,
    /// A headpiece image basename placed atop every chapter opener (a kit ornament).
    pub headpiece: Option<&'a str>,
    /// A tailpiece image basename centred at the end of every chapter.
    pub tailpiece: Option<&'a str>,
    /// A divider image basename for scene breaks (else a typographic asterism).
    pub divider: Option<&'a str>,
    /// A colophon line (small-caps, centred on the final leaf); empty = no colophon.
    pub colophon: &'a str,
}

/// Emit the whole book as one Typst source string.
pub fn book_typ(front: &[Block], chapters: &[Chapter], opts: &BookOpts) -> String {
    let mut s = String::new();
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n");
    s.push_str("// plakat bookart — a whole typeset BOOK (title page · chapters · folios · colophon).\n");
    s.push_str("// Compile:  typst compile <this-file>.typ\n");
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n\n");

    s.push_str(&format!("#set page(\n  width: {}mm, height: {}mm,\n", mm(opts.w_mm), mm(opts.h_mm)));
    s.push_str("  margin: (inside: 22mm, outside: 18mm, top: 20mm, bottom: 22mm),\n");
    s.push_str("  footer: context align(center, text(size: 9.5pt)[#counter(page).display()]),\n");
    if !opts.running_head.trim().is_empty() {
        s.push_str(&format!(
            "  header: context {{ if counter(page).get().first() > 1 {{ align(center, text(size: 8.5pt, tracking: 0.08em)[#smallcaps(\"{}\")]) }} }},\n",
            esc(opts.running_head),
        ));
    }
    s.push_str(")\n");
    s.push_str("#set text(size: 11pt)\n");
    s.push_str("#set par(leading: 0.72em, justify: true, first-line-indent: 1.4em)\n\n");

    if let Some(tp) = opts.title_page {
        s.push_str(&format!("#include \"{}\"\n", esc(tp)));
        s.push_str("#counter(page).update(1)\n\n");
    }
    // Front matter (no raised initial).
    let mut first_para_pending = false;
    for b in front {
        emit_block(&mut s, b, &mut first_para_pending, opts.divider);
    }

    for (i, ch) in chapters.iter().enumerate() {
        s.push_str("#pagebreak(weak: true)\n");
        s.push_str("#{\n  set align(center)\n  v(6%)\n");
        if let Some(hp) = opts.headpiece {
            s.push_str(&format!("  image(\"{}\", width: 26%)\n  v(0.6em)\n", esc(hp)));
        }
        s.push_str(&format!("  text(size: 11pt, tracking: 0.14em)[#smallcaps(\"Chapter {}\")]\n  v(0.5em)\n", roman(i as u32 + 1)));
        s.push_str(&format!("  text(size: 19pt, weight: \"bold\")[#upper(\"{}\")]\n", esc(&ch.title)));
        s.push_str("  v(1.6em)\n}\n");
        s.push_str("#set par(first-line-indent: 0pt)\n");

        // The first paragraph of the chapter gets the raised initial.
        first_para_pending = true;
        for b in &ch.blocks {
            emit_block(&mut s, b, &mut first_para_pending, opts.divider);
        }
        if let Some(tp) = opts.tailpiece {
            s.push_str(&format!("#v(1.2em)\n#align(center, image(\"{}\", width: 16%))\n\n", esc(tp)));
        }
    }

    if !opts.colophon.trim().is_empty() {
        s.push_str("#pagebreak(weak: true)\n");
        s.push_str(&format!(
            "#align(center + horizon, text(size: 9.5pt, tracking: 0.06em)[#smallcaps(\"{}\")])\n",
            esc(opts.colophon),
        ));
    }
    s
}

/// Emit one body block. `first_para_pending` (set true at a chapter's start) gives the first paragraph a
/// raised decorated initial + small-caps opening, then resets normal indentation.
fn emit_block(s: &mut String, b: &Block, first_para_pending: &mut bool, divider: Option<&str>) {
    match b {
        Block::Para(text) => {
            if *first_para_pending {
                *first_para_pending = false;
                let (first, lead, tail) = split_opening(text);
                s.push_str(&format!(
                    "#par[#text(size: 2.6em, weight: \"bold\")[{}]#h(0.04em)#smallcaps[{}]{}]\n\n",
                    esc_markup(&first),
                    inline(&lead),
                    inline(&tail),
                ));
                s.push_str("#set par(first-line-indent: 1.4em)\n");
            } else {
                s.push_str(&format!("#par[{}]\n\n", inline(text)));
            }
        }
        Block::Section(text) => {
            *first_para_pending = false;
            s.push_str(&format!(
                "#v(1.2em)\n#align(center, text(size: 11pt, tracking: 0.12em)[#smallcaps(\"{}\")])\n#v(0.6em)\n#set par(first-line-indent: 0pt)\n",
                esc(text),
            ));
        }
        Block::Quote(paras) => {
            s.push_str("#v(0.5em)\n#pad(left: 2.5em, right: 2.5em)[#{\n  set text(size: 10.5pt, style: \"italic\")\n  set par(first-line-indent: 0pt)\n");
            for (i, p) in paras.iter().enumerate() {
                if p.trim().is_empty() {
                    continue;
                }
                if i > 0 {
                    s.push_str("  parbreak()\n");
                }
                s.push_str(&format!("  [{}]\n", inline(p)));
            }
            s.push_str("}]\n#v(0.5em)\n");
        }
        Block::SceneBreak => {
            *first_para_pending = false;
            match divider {
                Some(img) => s.push_str(&format!("#v(0.6em)\n#align(center, image(\"{}\", width: 13%))\n#v(0.6em)\n", esc(img))),
                None => s.push_str("#v(0.7em)\n#align(center, text(size: 11pt, tracking: 0.5em)[\\*\\*\\*])\n#v(0.7em)\n"),
            }
        }
    }
}

/// Parse inline emphasis (`*`/`_` → emph, `**`/`__` → strong) and escape the literal runs — returns Typst
/// markup. Non-nested delimiters that never close are treated as literal text.
fn inline(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        match find_delim(rest) {
            Some((idx, delim, strong)) => {
                out.push_str(&esc_markup(&rest[..idx]));
                let after = &rest[idx + delim.len()..];
                if let Some(close) = after.find(delim) {
                    let inner = &after[..close];
                    if strong {
                        out.push_str(&format!("#strong[{}]", inline(inner)));
                    } else {
                        out.push_str(&format!("#emph[{}]", inline(inner)));
                    }
                    rest = &after[close + delim.len()..];
                } else {
                    out.push_str(&esc_markup(delim));
                    rest = after;
                }
            }
            None => {
                out.push_str(&esc_markup(rest));
                break;
            }
        }
    }
    out
}

/// Find the earliest emphasis delimiter in `s`: `(byte_index, "**"|"__"|"*"|"_", is_strong)`.
fn find_delim(s: &str) -> Option<(usize, &'static str, bool)> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'*' || b[i] == b'_' {
            let two = i + 1 < b.len() && b[i + 1] == b[i];
            return Some(match (b[i], two) {
                (b'*', true) => (i, "**", true),
                (b'*', false) => (i, "*", false),
                (_, true) => (i, "__", true),
                (_, false) => (i, "_", false),
            });
        }
        i += 1;
    }
    None
}

/// Escape a run of prose for Typst MARKUP context — backslash-escapes the characters Typst treats as
/// syntax, so plain manuscript text renders verbatim (no accidental emphasis/headings/links).
fn esc_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if matches!(c, '\\' | '#' | '[' | ']' | '*' | '_' | '`' | '$' | '<' | '>' | '@' | '=' | '~') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paras(bs: &[Block]) -> Vec<String> {
        bs.iter().filter_map(|b| if let Block::Para(p) = b { Some(p.clone()) } else { None }).collect()
    }

    #[test]
    fn parses_front_matter_and_chapters() {
        let md = "A dedication.\n\n# The Departure\n\nFirst para of one.\n\nSecond para.\n\n# The Storm\n\nOnly para.";
        let (front, chapters) = parse_manuscript(md);
        assert_eq!(paras(&front), vec!["A dedication."]);
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "The Departure");
        assert_eq!(paras(&chapters[0].blocks), vec!["First para of one.", "Second para."]);
        assert_eq!(chapters[1].title, "The Storm");
    }

    #[test]
    fn parses_sections_quotes_and_scene_breaks() {
        let md = "# One\n\nOpening.\n\n## A Section\n\nMore.\n\n***\n\nAfter break.\n\n> A quoted line\n> continued.";
        let (_front, chapters) = parse_manuscript(md);
        let b = &chapters[0].blocks;
        assert!(matches!(b[0], Block::Para(_)));
        assert!(matches!(&b[1], Block::Section(t) if t == "A Section"));
        assert!(matches!(b[3], Block::SceneBreak));
        assert!(matches!(&b[5], Block::Quote(q) if q[0] == "A quoted line continued."));
    }

    #[test]
    fn inline_emphasis() {
        assert_eq!(inline("a *b* c"), "a #emph[b] c");
        assert_eq!(inline("a **b** c"), "a #strong[b] c");
        assert_eq!(inline("plain # text"), "plain \\# text");
        assert_eq!(inline("un*closed"), "un\\*closed");
    }

    #[test]
    fn roman_numerals() {
        assert_eq!(roman(1), "I");
        assert_eq!(roman(4), "IV");
        assert_eq!(roman(23), "XXIII");
        assert_eq!(roman(40), "40");
    }

    #[test]
    fn emits_a_whole_book() {
        let (front, chapters) = parse_manuscript("# The Departure\n\nThe *wind* rose over Plymouth Sound.\n\n***\n\nAt dawn.");
        let opts = BookOpts {
            w_mm: 148.0,
            h_mm: 210.0,
            running_head: "The Open Sea",
            title_page: Some("01-title.typ"),
            headpiece: Some("rosette.png"),
            tailpiece: Some("dinkus.png"),
            divider: None,
            colophon: "Set in Libertinus.",
        };
        let out = book_typ(&front, &chapters, &opts);
        assert!(out.contains("#include \"01-title.typ\""), "includes title page");
        assert!(out.contains("counter(page).display()"), "folios");
        assert!(out.contains(r#"smallcaps("Chapter I")"#), "chapter numeral");
        assert!(out.contains("size: 2.6em"), "raised initial");
        assert!(out.contains("#emph[wind]"), "inline emphasis in body");
        assert!(out.contains(r#"[\*\*\*]"#), "asterism scene break");
        assert!(out.contains(r#"image("dinkus.png""#), "tailpiece");
        assert!(out.contains(r#"smallcaps("Set in Libertinus.")"#), "colophon");
    }
}
