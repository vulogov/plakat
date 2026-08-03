//! Manuscript-aware ornament sets (RFC BOOKART-1 §11, flagship pt 2). Parse a book's chapter structure
//! and emit a *matched* set — a seed-varied headpiece per chapter (a variation of the shared motif), a
//! tailpiece per chapter, and a frontispiece — all in one hand. This module holds the pure pieces:
//! chapter parsing and the optional LaTeX include emission; the CLI drives generation via `do_render`.

/// A parsed chapter: its title and the first alphabetic letter (for a decorated initial).
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    pub title: String,
    pub first_letter: String,
}

/// Parse chapter titles from a manuscript. If any line is a Markdown heading (`#`/`##`/`###`), only
/// headings are chapters; otherwise every non-empty line is a chapter title (a plain list).
pub fn parse_chapters(text: &str) -> Vec<Chapter> {
    let has_md = text.lines().any(|l| l.trim_start().starts_with('#'));
    let titles: Vec<String> = if has_md {
        text.lines()
            .filter_map(|l| {
                let t = l.trim_start();
                t.starts_with('#').then(|| t.trim_start_matches('#').trim().to_string())
            })
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        text.lines().map(|l| l.trim().to_string()).filter(|s| !s.is_empty()).collect()
    };
    titles
        .into_iter()
        .map(|title| {
            let first_letter = title.chars().find(|c| c.is_alphabetic()).map(|c| c.to_uppercase().to_string()).unwrap_or_default();
            Chapter { title, first_letter }
        })
        .collect()
}

/// A LaTeX include file mapping each asset to an `\includegraphics`, grouped by chapter — a drop-in for
/// a book's preamble (`\input{includes.tex}`), correct-DPI PNGs placed at text width.
pub fn latex_includes(frontispiece: &str, chapters: &[(String, String, String)]) -> String {
    // chapters: (chapter title, headpiece file, tailpiece file)
    let mut s = String::from("% plakat bookart — manuscript ornament includes (\\input this file).\n");
    s.push_str(&format!("\\newcommand{{\\bookartFrontispiece}}{{\\includegraphics[width=\\textwidth]{{{frontispiece}}}}}\n"));
    for (i, (title, head, tail)) in chapters.iter().enumerate() {
        let n = i + 1;
        s.push_str(&format!("% Chapter {n}: {title}\n"));
        s.push_str(&format!("\\newcommand{{\\bookartHeadpiece{n}}}{{\\includegraphics[width=\\textwidth]{{{head}}}}}\n"));
        s.push_str(&format!("\\newcommand{{\\bookartTailpiece{n}}}{{\\includegraphics[width=0.4\\textwidth]{{{tail}}}}}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_markdown_headings() {
        let ch = parse_chapters("# The Firebird\n\nsome prose\n\n## Into the Forest\n### sub\n");
        assert_eq!(ch.len(), 3);
        assert_eq!(ch[0].title, "The Firebird");
        assert_eq!(ch[0].first_letter, "T");
        assert_eq!(ch[1].title, "Into the Forest");
    }

    #[test]
    fn parses_plain_list_when_no_headings() {
        let ch = parse_chapters("Chapter One\nChapter Two\n\nChapter Three\n");
        assert_eq!(ch.len(), 3);
        assert_eq!(ch[2].title, "Chapter Three");
        assert_eq!(ch[0].first_letter, "C");
    }

    #[test]
    fn first_letter_skips_numbers_and_punct() {
        let ch = parse_chapters("1. Того царства\n");
        assert_eq!(ch[0].first_letter, "Т");
    }

    #[test]
    fn latex_has_a_command_per_chapter() {
        let tex = latex_includes("frontispiece.png", &[("Ch A".into(), "ch01_h.png".into(), "ch01_t.png".into())]);
        assert!(tex.contains("\\bookartFrontispiece"));
        assert!(tex.contains("\\bookartHeadpiece1"));
        assert!(tex.contains("ch01_h.png"));
    }
}
