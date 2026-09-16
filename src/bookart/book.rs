//! The CAPSTONE: assemble a whole typeset BOOK as one compilable Typst file from a Markdown manuscript —
//! a title page (a `bookart title-page` artifact, `#include`d), chapter openers (an optional headpiece,
//! `CHAPTER N`, the title), body prose with a raised decorated initial + small-caps opening on each
//! chapter's first paragraph, running heads + page folios, an optional tailpiece per chapter, and a
//! colophon. Weight-free typography; the only images are the ornaments/plates the caller supplies.
//!
//! The manuscript is plain Markdown: a line beginning `#`/`##` starts a chapter (its text is the title);
//! blank lines separate paragraphs; anything before the first heading is front matter.

use crate::bookart::typst::{esc, mm};

/// One parsed chapter.
pub struct Chapter {
    pub title: String,
    pub paragraphs: Vec<String>,
}

/// Parse a Markdown manuscript into (front-matter paragraphs, chapters). A `#`/`##` line opens a chapter.
pub fn parse_manuscript(md: &str) -> (Vec<String>, Vec<Chapter>) {
    let mut front: Vec<String> = Vec::new();
    let mut chapters: Vec<Chapter> = Vec::new();
    let mut cur_para = String::new();
    let flush = |para: &mut String, front: &mut Vec<String>, chapters: &mut Vec<Chapter>| {
        let p = para.trim();
        if !p.is_empty() {
            match chapters.last_mut() {
                Some(ch) => ch.paragraphs.push(p.to_string()),
                None => front.push(p.to_string()),
            }
        }
        para.clear();
    };
    for raw in md.lines() {
        let line = raw.trim_end();
        let heading = line.trim_start();
        if let Some(rest) = heading.strip_prefix("## ").or_else(|| heading.strip_prefix("# ")) {
            flush(&mut cur_para, &mut front, &mut chapters);
            chapters.push(Chapter { title: rest.trim().to_string(), paragraphs: Vec::new() });
        } else if line.trim().is_empty() {
            flush(&mut cur_para, &mut front, &mut chapters);
        } else {
            if !cur_para.is_empty() {
                cur_para.push(' ');
            }
            cur_para.push_str(line.trim());
        }
    }
    flush(&mut cur_para, &mut front, &mut chapters);
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
    // A short lead (up to the 4th word or ~28 chars) is set in small caps.
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
    /// A colophon line (small-caps, centred on the final leaf); empty = no colophon.
    pub colophon: &'a str,
}

/// Emit the whole book as one Typst source string.
pub fn book_typ(front: &[String], chapters: &[Chapter], opts: &BookOpts) -> String {
    let mut s = String::new();
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n");
    s.push_str("// plakat bookart — a whole typeset BOOK (title page · chapters · folios · colophon).\n");
    s.push_str("// Compile:  typst compile <this-file>.typ\n");
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n\n");

    // Page: book margins, a centred folio in the footer, a small-caps running head.
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

    // Title page (rendered from a bookart title-page artifact), then front matter.
    if let Some(tp) = opts.title_page {
        s.push_str(&format!("#include \"{}\"\n", esc(tp)));
        s.push_str("#counter(page).update(1)\n\n");
    }
    for para in front {
        s.push_str(&format!("#par[{}]\n\n", markup(para)));
    }

    // Chapters.
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

        for (pi, para) in ch.paragraphs.iter().enumerate() {
            if pi == 0 {
                // A raised decorated initial + a small-caps opening for the first paragraph.
                let (first, lead, tail) = split_opening(para);
                s.push_str(&format!(
                    "#par[#text(size: 2.6em, weight: \"bold\")[{}]#h(0.04em)#smallcaps[{}]{}]\n\n",
                    markup(&first),
                    markup(&lead),
                    markup(&tail),
                ));
                s.push_str("#set par(first-line-indent: 1.4em)\n");
            } else {
                s.push_str(&format!("#par[{}]\n\n", markup(para)));
            }
        }
        if let Some(tp) = opts.tailpiece {
            s.push_str(&format!("#v(1.2em)\n#align(center, image(\"{}\", width: 16%))\n\n", esc(tp)));
        }
    }

    // Colophon.
    if !opts.colophon.trim().is_empty() {
        s.push_str("#pagebreak(weak: true)\n");
        s.push_str(&format!(
            "#align(center + horizon, text(size: 9.5pt, tracking: 0.06em)[#smallcaps(\"{}\")])\n",
            esc(opts.colophon),
        ));
    }
    s
}

/// Escape a run of prose for Typst MARKUP context — backslash-escapes the characters Typst would otherwise
/// treat as syntax, so plain manuscript text renders verbatim (no accidental emphasis/headings/links).
fn markup(text: &str) -> String {
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

    #[test]
    fn parses_front_matter_and_chapters() {
        let md = "A dedication.\n\n# The Departure\n\nFirst para of one.\n\nSecond para.\n\n## The Storm\n\nOnly para.";
        let (front, chapters) = parse_manuscript(md);
        assert_eq!(front, vec!["A dedication."]);
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "The Departure");
        assert_eq!(chapters[0].paragraphs, vec!["First para of one.", "Second para."]);
        assert_eq!(chapters[1].title, "The Storm");
        assert_eq!(chapters[1].paragraphs, vec!["Only para."]);
    }

    #[test]
    fn roman_numerals() {
        assert_eq!(roman(1), "I");
        assert_eq!(roman(4), "IV");
        assert_eq!(roman(9), "IX");
        assert_eq!(roman(23), "XXIII");
        assert_eq!(roman(40), "40");
    }

    #[test]
    fn emits_a_whole_book() {
        let (front, chapters) = parse_manuscript("# The Departure\n\nThe wind rose over Plymouth Sound that morning.");
        let opts = BookOpts {
            w_mm: 148.0,
            h_mm: 210.0,
            running_head: "The Open Sea",
            title_page: Some("01-title.typ"),
            headpiece: Some("rosette.png"),
            tailpiece: Some("dinkus.png"),
            colophon: "Set in Libertinus.",
        };
        let out = book_typ(&front, &chapters, &opts);
        assert!(out.contains("#include \"01-title.typ\""), "includes title page:\n{out}");
        assert!(out.contains("counter(page).display()"), "folios");
        assert!(out.contains(r#"smallcaps("The Open Sea")"#), "running head");
        assert!(out.contains(r#"smallcaps("Chapter I")"#), "chapter numeral");
        assert!(out.contains(r#"upper("The Departure")"#), "chapter title");
        assert!(out.contains("size: 2.6em"), "raised initial");
        assert!(out.contains(r#"image("rosette.png""#), "headpiece");
        assert!(out.contains(r#"image("dinkus.png""#), "tailpiece");
        assert!(out.contains(r#"smallcaps("Set in Libertinus.")"#), "colophon");
    }

    #[test]
    fn markup_escapes_syntax() {
        assert_eq!(markup("a #b [c] *d* _e_"), "a \\#b \\[c\\] \\*d\\* \\_e\\_");
    }
}
