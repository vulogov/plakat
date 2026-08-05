//! 6.1.0 (B6): EPUB manuscript input — parse a book's chapter structure from an `.epub` so
//! `bookart manuscript book.epub` produces the same per-chapter ornament set as the Markdown / plain
//! list inputs. Behind the `epub` feature (`zip`). An EPUB is a ZIP: `META-INF/container.xml` → the OPF
//! package → the NCX / nav TOC. We pull chapter titles from the TOC in reading order. XML is extracted
//! with light tag-scanning (no XML-parser dep) — EPUB TOCs are regular enough, and a missed title just
//! falls back to a generic "Chapter N".

#![cfg(feature = "epub")]

use crate::bookart::manuscript::{chapter_from_title, Chapter};
use anyhow::{Context, Result};
use std::io::Read;

/// Parse chapter titles from an EPUB's TOC (NCX `navMap`, else EPUB3 `nav`, else spine doc titles).
pub fn parse_epub_chapters(path: &std::path::Path) -> Result<Vec<Chapter>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).with_context(|| format!("reading EPUB (zip) {}", path.display()))?;

    // 1. container.xml → the OPF package path.
    let container = read_entry(&mut zip, "META-INF/container.xml").context("EPUB missing META-INF/container.xml")?;
    let opf_path = attr_value(&container, "rootfile", "full-path").context("EPUB container.xml has no rootfile")?;

    // 2. OPF → the NCX / nav href (resolved relative to the OPF's directory).
    let opf = read_entry(&mut zip, &opf_path).with_context(|| format!("reading OPF {opf_path}"))?;
    let base = opf_path.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();

    // Prefer the NCX (EPUB2); fall back to the EPUB3 nav document.
    let titles = ncx_titles(&mut zip, &opf, &base)
        .or_else(|| nav_titles(&mut zip, &opf, &base))
        .filter(|t| !t.is_empty());

    let titles = match titles {
        Some(t) => t,
        None => spine_titles(&mut zip, &opf, &base).context("EPUB: no TOC (NCX/nav) and no spine titles found")?,
    };
    anyhow::ensure!(!titles.is_empty(), "EPUB: parsed zero chapters");
    Ok(titles.into_iter().map(chapter_from_title).collect())
}

/// NCX `navMap` → `navPoint`/`navLabel`/`text` in document order.
fn ncx_titles(zip: &mut zip::ZipArchive<std::fs::File>, opf: &str, base: &str) -> Option<Vec<String>> {
    // manifest item with an `.ncx` href (media-type application/x-dtbncx+xml).
    let ncx_href = manifest_href_ending(opf, ".ncx")?;
    let ncx = read_entry(zip, &format!("{base}{ncx_href}")).ok()?;
    let titles = inner_texts(&ncx, "text");
    (!titles.is_empty()).then_some(titles)
}

/// EPUB3 nav document (`properties="nav"`) → the `<a>` link texts of its `toc` nav.
fn nav_titles(zip: &mut zip::ZipArchive<std::fs::File>, opf: &str, base: &str) -> Option<Vec<String>> {
    // Find the manifest item whose properties contain `nav`.
    let nav_href = manifest_nav_href(opf)?;
    let nav = read_entry(zip, &format!("{base}{nav_href}")).ok()?;
    let titles = inner_texts(&nav, "a");
    (!titles.is_empty()).then_some(titles)
}

/// Last resort: each spine document's `<title>` (or a generic label), in reading order.
fn spine_titles(zip: &mut zip::ZipArchive<std::fs::File>, opf: &str, base: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for (i, idref) in spine_idrefs(opf).into_iter().enumerate() {
        if let Some(href) = manifest_href_for_id(opf, &idref) {
            let title = read_entry(zip, &format!("{base}{href}"))
                .ok()
                .and_then(|doc| inner_texts(&doc, "title").into_iter().next())
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| format!("Chapter {}", i + 1));
            out.push(title);
        }
    }
    Ok(out)
}

// --- tiny XML/zip helpers (no XML-parser dep) -----------------------------------------------------

fn read_entry(zip: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<String> {
    let mut f = zip.by_name(name).with_context(|| format!("EPUB entry {name} not found"))?;
    let mut s = String::new();
    f.read_to_string(&mut s).with_context(|| format!("reading EPUB entry {name}"))?;
    Ok(s)
}

/// The value of `attr` on the first `<tag ...>` element that has it.
fn attr_value(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let mut from = 0;
    while let Some(i) = xml[from..].find(&format!("<{tag}")) {
        let start = from + i;
        let end = xml[start..].find('>').map(|e| start + e)?;
        let elem = &xml[start..end];
        if let Some(v) = one_attr(elem, attr) {
            return Some(v);
        }
        from = end;
    }
    None
}

/// Value of `attr="..."` within a single element string.
fn one_attr(elem: &str, attr: &str) -> Option<String> {
    let key = format!("{attr}=\"");
    let a = elem.find(&key)? + key.len();
    let b = elem[a..].find('"')? + a;
    Some(elem[a..b].to_string())
}

/// The href of a manifest `<item>` whose href ends with `suffix`.
fn manifest_href_ending(opf: &str, suffix: &str) -> Option<String> {
    each_item(opf).find_map(|item| {
        let href = one_attr(item, "href")?;
        href.ends_with(suffix).then_some(href)
    })
}

/// The href of the manifest `<item>` whose `properties` mention `nav`.
fn manifest_nav_href(opf: &str) -> Option<String> {
    each_item(opf).find_map(|item| {
        one_attr(item, "properties").filter(|p| p.split_whitespace().any(|w| w == "nav"))?;
        one_attr(item, "href")
    })
}

/// The href of the manifest `<item id="...">`.
fn manifest_href_for_id(opf: &str, id: &str) -> Option<String> {
    each_item(opf).find_map(|item| {
        (one_attr(item, "id").as_deref() == Some(id)).then(|| one_attr(item, "href"))?
    })
}

/// The `idref`s of the spine `<itemref>`s, in order.
fn spine_idrefs(opf: &str) -> Vec<String> {
    let spine = match (opf.find("<spine"), opf.find("</spine>")) {
        (Some(a), Some(b)) if b > a => &opf[a..b],
        _ => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = spine[from..].find("<itemref") {
        let start = from + i;
        let end = spine[start..].find('>').map(|e| start + e).unwrap_or(spine.len());
        if let Some(idref) = one_attr(&spine[start..end], "idref") {
            out.push(idref);
        }
        from = end;
    }
    out
}

/// Iterate the manifest `<item .../>` element strings.
fn each_item(opf: &str) -> impl Iterator<Item = &str> {
    let manifest = match (opf.find("<manifest"), opf.find("</manifest>")) {
        (Some(a), Some(b)) if b > a => &opf[a..b],
        _ => "",
    };
    ItemIter { s: manifest, pos: 0 }
}

struct ItemIter<'a> {
    s: &'a str,
    pos: usize,
}
impl<'a> Iterator for ItemIter<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        let i = self.s[self.pos..].find("<item")?;
        let start = self.pos + i;
        let end = self.s[start..].find('>').map(|e| start + e + 1).unwrap_or(self.s.len());
        self.pos = end;
        Some(&self.s[start..end])
    }
}

/// All `<tag>inner</tag>` inner texts (trimmed, tags stripped), in document order.
fn inner_texts(xml: &str, tag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (open, close) = (format!("<{tag}"), format!("</{tag}>"));
    let mut from = 0;
    while let Some(i) = xml[from..].find(&open) {
        let start = from + i;
        let Some(gt) = xml[start..].find('>').map(|e| start + e + 1) else { break };
        let Some(c) = xml[gt..].find(&close).map(|e| gt + e) else { break };
        let text = strip_tags(&xml[gt..c]).trim().to_string();
        if !text.is_empty() {
            out.push(unescape(&text));
        }
        from = c + close.len();
    }
    out
}

/// Remove any nested markup, collapsing whitespace.
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = (depth - 1i32).max(0),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn unescape(s: &str) -> String {
    s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncx_inner_texts_in_order() {
        let ncx = r#"<navMap>
            <navPoint><navLabel><text>The Firebird</text></navLabel></navPoint>
            <navPoint><navLabel><text>Vasilisa &amp; the Wolf</text></navLabel></navPoint>
        </navMap>"#;
        let t = inner_texts(ncx, "text");
        assert_eq!(t, vec!["The Firebird".to_string(), "Vasilisa & the Wolf".to_string()]);
    }

    #[test]
    fn manifest_and_container_scanning() {
        let container = r#"<container><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#;
        assert_eq!(attr_value(container, "rootfile", "full-path").as_deref(), Some("OEBPS/content.opf"));
        let opf = r#"<manifest>
            <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
            <item id="nav" href="nav.xhtml" properties="nav" media-type="application/xhtml+xml"/>
        </manifest>"#;
        assert_eq!(manifest_href_ending(opf, ".ncx").as_deref(), Some("toc.ncx"));
        assert_eq!(manifest_nav_href(opf).as_deref(), Some("nav.xhtml"));
    }

    #[test]
    fn strip_nested_markup() {
        assert_eq!(strip_tags("<a href=\"x\">Chapter <b>One</b></a>"), "Chapter One");
    }

    #[test]
    fn malformed_epub_errors_not_panics() {
        // B2: a non-zip / missing file must return Err (the CLI shows it), never panic.
        let dir = std::env::temp_dir().join(format!("plakat-epub-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bogus = dir.join("not.epub");
        std::fs::write(&bogus, b"this is not a zip file").unwrap();
        assert!(parse_epub_chapters(&bogus).is_err(), "garbage epub should error");
        assert!(parse_epub_chapters(&dir.join("missing.epub")).is_err(), "missing epub should error");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
