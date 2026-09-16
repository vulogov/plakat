//! Book / chapter TITLE PAGES as compilable Typst — the historical letterpress hierarchy (mixed sizes,
//! small-caps, tracked display type, rules, an imprint at the foot, an optional ornamental border or
//! emblem), authored in a small HJSON. Weight-free: the typography IS the artefact (no diffusion).
//!
//! Generic over a `style:` — **`letterpress`** (the antique look in the examples) is the default; more
//! styles can slot in beside it. The output is a reusable Typst artifact: `#let title-page = { … }` +
//! a preview, so it compiles standalone (`--verify`) AND `#import`s into a book. Shares the `bookart typst`
//! geometry (`Margins`, `clear_window`) so a title page can sit inside a measured border window.

use crate::bookart::typst::{esc, mm, Margins};
use serde::Deserialize;

/// The HJSON a title page is authored in.
#[derive(Deserialize, Default)]
pub struct TitlePageSpec {
    /// Typographic style; `letterpress` (default) = the historical hierarchy.
    #[serde(default)]
    pub style: Option<String>,
    /// Page-size name (`a5`/`a4`/`b5`/…); the CLI `--page` overrides it.
    #[serde(default)]
    pub page: Option<String>,
    /// Optional full-page ornamental FRAME image; the type is fitted inside its measured clear window.
    #[serde(default)]
    pub border: Option<String>,
    /// A plain ruled box at `rule` pt when there is no border image (0/absent = none).
    #[serde(default)]
    pub rule: Option<f32>,
    /// Body serif font family (a Typst font name). Absent = the document default serif.
    #[serde(default)]
    pub font: Option<String>,
    /// Historical typography (old-style figures + historical ligatures). CLI `--historical` also sets it.
    #[serde(default)]
    pub historical: Option<bool>,
    /// The vertical stack of lines, top to bottom.
    #[serde(default)]
    pub lines: Vec<TitleLine>,
}

/// One line/block of the title page.
#[derive(Deserialize, Default, Clone)]
pub struct TitleLine {
    /// `series | title | subtitle | part | subchapter | author | note | epigraph | imprint | rule | ornament | image | space`.
    pub role: String,
    /// The text (text roles); `\n` splits it into stacked centred lines.
    #[serde(default)]
    pub text: Option<String>,
    /// An image reference (`image` / `ornament` roles) — a basename beside the `.typ`.
    #[serde(default)]
    pub src: Option<String>,
    /// Size override: pt for text roles, em of blank space for `space`, or width-% for `image`.
    /// `width` is accepted as an alias (natural on `image`/`ornament` lines).
    #[serde(default, alias = "width")]
    pub size: Option<f32>,
}

/// Emit-time options: a global size `scale` (1.0 = as authored; `--fit` lowers it to fit one page) and
/// `historical` typography (old-style figures + historical ligatures). Kept as a struct so future style
/// work can extend it without churning every call site.
#[derive(Clone, Copy)]
pub struct Emit {
    /// Multiplies every type size and image width (1.0 = as authored).
    pub scale: f32,
    /// Old-style figures + historical ligatures (weight-free antique feel).
    pub historical: bool,
    /// The typographic style (role table + rule weight).
    pub style: Style,
}

impl Default for Emit {
    fn default() -> Self {
        Self { scale: 1.0, historical: false, style: Style::Letterpress }
    }
}

/// A typographic style — the same roles, a different hand. Weight-free: the look comes purely from size,
/// case, weight, tracking, italic and spacing (no display fonts required).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// Dense antique book type — bold upper display, small-caps series, the default.
    Letterpress,
    /// Copperplate / engraved-atlas: regular-weight wide-tracked caps, italic subtitles & author, airy.
    Engraved,
    /// Minimal contemporary: regular weight, as-authored case, tiny wide-tracked labels, lots of air.
    Modern,
    /// Victorian playbill / poster: everything heavy bold upper, big size jumps, tight leading, thick rules.
    Playbill,
}

impl Style {
    /// Resolve a spec/CLI `style:` name (permissive; unknown → letterpress).
    pub fn from_name(s: &str) -> Style {
        match s.trim().to_lowercase().replace('_', "-").as_str() {
            "engraved" | "engraved-atlas" | "copperplate" | "atlas" => Style::Engraved,
            "modern" | "modern-minimal" | "minimal" | "contemporary" => Style::Modern,
            "playbill" | "poster" | "victorian" | "broadside" => Style::Playbill,
            _ => Style::Letterpress,
        }
    }

    /// The role table: `(pt, transform, weight, italic, tracking_em, space_after_em)`.
    /// `transform`: `'u'` upper, `'s'` small-caps, `'n'` none (as authored).
    fn role(self, role: &str) -> Option<(f32, char, &'static str, bool, f32, f32)> {
        match self {
            Style::Letterpress => letterpress(role),
            Style::Engraved => engraved(role),
            Style::Modern => modern(role),
            Style::Playbill => playbill(role),
        }
    }

    /// Rule stroke weight (pt) for this style's `rule` lines.
    fn rule_stroke(self) -> f32 {
        match self {
            Style::Letterpress => 0.5,
            Style::Engraved => 0.3,
            Style::Modern => 0.4,
            Style::Playbill => 1.4,
        }
    }
}

/// The `letterpress` role table: `(pt, transform, weight, italic, tracking_em, space_after_em)`.
/// `transform`: `'u'` upper, `'s'` small-caps, `'n'` none.
fn letterpress(role: &str) -> Option<(f32, char, &'static str, bool, f32, f32)> {
    Some(match role {
        "series" => (13.0, 's', "regular", false, 0.06, 0.7),
        "title" => (30.0, 'u', "bold", false, 0.02, 0.5),
        "subtitle" => (17.0, 'u', "regular", false, 0.03, 0.6),
        "part" => (15.0, 'u', "bold", false, 0.04, 0.5),
        // A SUBCHAPTER / section mark — subordinate to `part`: smaller, small-caps, generously tracked.
        "subchapter" => (12.0, 's', "regular", false, 0.09, 0.45),
        "author" => (15.0, 'n', "regular", false, 0.05, 0.6),
        "note" => (10.0, 's', "regular", false, 0.02, 0.6),
        "epigraph" => (11.0, 'n', "regular", true, 0.0, 0.6),
        "imprint" => (11.0, 's', "regular", false, 0.03, 0.25),
        _ => return None,
    })
}

/// Copperplate / engraved-atlas: regular weight, wide-tracked roman caps for the title, italic for the
/// series / subtitle / author / notes — the elegant, airy engraved look.
fn engraved(role: &str) -> Option<(f32, char, &'static str, bool, f32, f32)> {
    Some(match role {
        "series" => (12.0, 's', "regular", true, 0.08, 0.8),
        "title" => (26.0, 'u', "regular", false, 0.12, 0.7),
        "subtitle" => (15.0, 'n', "regular", true, 0.02, 0.7),
        "part" => (13.0, 's', "regular", false, 0.06, 0.6),
        "subchapter" => (11.0, 's', "regular", true, 0.08, 0.5),
        "author" => (14.0, 'n', "regular", true, 0.03, 0.7),
        "note" => (10.0, 'n', "regular", true, 0.0, 0.6),
        "epigraph" => (11.0, 'n', "regular", true, 0.0, 0.6),
        "imprint" => (10.0, 's', "regular", false, 0.06, 0.3),
        _ => return None,
    })
}

/// Minimal contemporary: regular weight, as-authored case for the display line, tiny wide-tracked labels
/// for the supporting lines, and generous spacing — quiet, lots of white.
fn modern(role: &str) -> Option<(f32, char, &'static str, bool, f32, f32)> {
    Some(match role {
        "series" => (10.0, 'u', "regular", false, 0.25, 1.0),
        "title" => (28.0, 'n', "regular", false, 0.0, 0.8),
        "subtitle" => (14.0, 'n', "regular", false, 0.0, 0.7),
        "part" => (11.0, 'u', "regular", false, 0.2, 0.7),
        "subchapter" => (10.0, 'u', "regular", false, 0.2, 0.5),
        "author" => (13.0, 'n', "regular", false, 0.0, 0.8),
        "note" => (10.0, 'n', "regular", false, 0.0, 0.7),
        "epigraph" => (11.0, 'n', "regular", true, 0.0, 0.7),
        "imprint" => (9.0, 'u', "regular", false, 0.2, 0.4),
        _ => return None,
    })
}

/// Victorian playbill / poster: everything heavy bold upper, big size jumps, tight leading — maximum ink.
fn playbill(role: &str) -> Option<(f32, char, &'static str, bool, f32, f32)> {
    Some(match role {
        "series" => (16.0, 'u', "bold", false, 0.02, 0.4),
        "title" => (44.0, 'u', "bold", false, 0.0, 0.35),
        "subtitle" => (22.0, 'u', "bold", false, 0.02, 0.4),
        "part" => (18.0, 'u', "bold", false, 0.03, 0.4),
        "subchapter" => (14.0, 'u', "bold", false, 0.05, 0.35),
        "author" => (18.0, 'u', "bold", false, 0.03, 0.45),
        "note" => (12.0, 'u', "regular", false, 0.02, 0.45),
        "epigraph" => (13.0, 'n', "regular", true, 0.0, 0.5),
        "imprint" => (14.0, 'u', "bold", false, 0.02, 0.3),
        _ => return None,
    })
}

/// Emit one styled text block (possibly multi-line on `\n`) into `s`.
fn push_text(s: &mut String, style: &(f32, char, &str, bool, f32, f32), text: &str, size_override: Option<f32>, scale: f32) {
    let (dpt, transform, weight, italic, tracking, space) = *style;
    let pt = size_override.unwrap_or(dpt) * scale;
    for (i, raw) in text.split('\n').enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if i > 0 {
            s.push_str("  v(0.15em)\n");
        }
        // A string literal wrapped in the requested transform, then set into `text(...)`.
        let inner = match transform {
            'u' => format!("upper(\"{}\")", esc(line)),
            's' => format!("smallcaps(\"{}\")", esc(line)),
            _ => format!("\"{}\"", esc(line)),
        };
        let mut args = format!("size: {}pt, weight: \"{}\"", trim_pt(pt), weight);
        if tracking > 0.0 {
            args.push_str(&format!(", tracking: {}em", tracking));
        }
        if italic {
            args.push_str(", style: \"italic\"");
        }
        s.push_str(&format!("  text({args})[#{inner}]\n"));
    }
    if space > 0.0 {
        s.push_str(&format!("  v({space}em)\n"));
    }
}

fn trim_pt(v: f32) -> String {
    let r = (v * 100.0).round() / 100.0;
    let mut s = format!("{r}");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    s
}

/// Build the full Typst source for a title page.
///
/// `text_margin` is the type box (already fitted to the border's clear window when a border is present);
/// `border` is `(ref, placement_margins)` for the full-page frame; `rule_pt` draws a plain ruled box when
/// there is no border image. `lines`' image `src`s must already be basenames beside the `.typ`.
pub fn title_page_typst(
    w_mm: f32,
    h_mm: f32,
    text_margin: &Margins,
    border: Option<(&str, &Margins)>,
    rule_pt: f32,
    font: Option<&str>,
    lines: &[TitleLine],
    emit: Emit,
) -> String {
    let scale = if emit.scale.is_finite() && emit.scale > 0.05 { emit.scale } else { 1.0 };
    let mut s = String::new();
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n");
    s.push_str("// plakat bookart — an old-style (letterpress) TITLE PAGE, compilable to PDF.\n");
    s.push_str("//   • hierarchical centred type: series / title / subtitle / part / author / note / imprint\n");
    s.push_str("//   • `title-page` is reusable: compile this file directly, or #import it into your book.\n");
    s.push_str("// Preview:  typst compile <this-file>.typ\n");
    s.push_str("// In a book: #import \"<this-file>.typ\": title-page   then   #title-page\n");
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n\n");

    s.push_str(&format!("#let page-width  = {}mm\n", mm(w_mm)));
    s.push_str(&format!("#let page-height = {}mm\n", mm(h_mm)));
    if let Some((bref, _)) = border {
        s.push_str(&format!("#let border-image = \"{}\"\n", esc(bref)));
    }
    s.push('\n');

    s.push_str("#let title-page = {\n");
    s.push_str("  set page(\n");
    s.push_str("    width: page-width, height: page-height,\n");
    s.push_str(&format!(
        "    margin: (top: {}mm, bottom: {}mm, left: {}mm, right: {}mm),\n",
        mm(text_margin.top),
        mm(text_margin.bottom),
        mm(text_margin.left),
        mm(text_margin.right),
    ));
    if let Some((_, bm)) = border {
        s.push_str("    background: place(top + left, dx: ");
        s.push_str(&format!("{}mm, dy: {}mm, image(border-image, ", mm(bm.left), mm(bm.top)));
        s.push_str(&format!(
            "width: page-width - {}mm - {}mm, height: page-height - {}mm - {}mm, fit: \"stretch\")),\n",
            mm(bm.left),
            mm(bm.right),
            mm(bm.top),
            mm(bm.bottom),
        ));
    } else if rule_pt > 0.0 {
        s.push_str(&format!(
            "    background: place(top + left, dx: {}mm, dy: {}mm, rect(width: page-width - {}mm - {}mm, \
             height: page-height - {}mm - {}mm, stroke: {}pt + black)),\n",
            mm((text_margin.left - 6.0).max(2.0)),
            mm((text_margin.top - 6.0).max(2.0)),
            mm((text_margin.left - 6.0).max(2.0)),
            mm((text_margin.right - 6.0).max(2.0)),
            mm((text_margin.top - 6.0).max(2.0)),
            mm((text_margin.bottom - 6.0).max(2.0)),
            trim_pt(rule_pt),
        ));
    }
    s.push_str("  )\n");
    // Base size scales with `--fit` so em-based spacers/rules shrink with the type. Historical typography
    // (old-style figures + historical ligatures) is weight-free and font-driven — harmless where unsupported.
    let base_pt = trim_pt(12.0 * scale);
    let hist = if emit.historical { ", number-type: \"old-style\", features: (hlig: 1)" } else { "" };
    match font {
        Some(f) => s.push_str(&format!("  set text(font: \"{}\", size: {base_pt}pt{hist})\n", esc(f))),
        None => s.push_str(&format!("  set text(size: {base_pt}pt{hist})\n")),
    }
    s.push_str("  set par(leading: 0.7em, justify: false)\n");
    s.push_str("  set align(center)\n\n");

    // The imprint block is collected and placed at the FOOT of the page; everything else flows from the top.
    let mut foot = String::new();
    for line in lines {
        let role = line.role.trim().to_lowercase();
        match role.as_str() {
            "imprint" => {
                if let Some(sty) = emit.style.role("imprint") {
                    if let Some(t) = &line.text {
                        push_text(&mut foot, &sty, t, line.size, scale);
                    }
                }
            }
            "rule" => {
                let len = line.size.unwrap_or(26.0).clamp(5.0, 100.0);
                s.push_str(&format!(
                    "  v(0.25em)\n  line(length: {}%, stroke: {}pt + black)\n  v(0.4em)\n",
                    trim_pt(len),
                    trim_pt(emit.style.rule_stroke()),
                ));
            }
            "space" => {
                s.push_str(&format!("  v({}em)\n", trim_pt(line.size.unwrap_or(1.0))));
            }
            "ornament" | "image" => {
                if let Some(src) = &line.src {
                    let w = line.size.unwrap_or(if role == "ornament" { 18.0 } else { 60.0 }).clamp(1.0, 100.0);
                    // Images scale with `--fit` too (a plate's height is what overflows a page).
                    let unit = if role == "ornament" {
                        format!("{}mm", trim_pt(line.size.unwrap_or(18.0) * scale))
                    } else {
                        format!("{}%", trim_pt((w * scale).clamp(1.0, 100.0)))
                    };
                    s.push_str(&format!("  v(0.3em)\n  image(\"{}\", width: {})\n  v(0.3em)\n", esc(src), unit));
                }
            }
            other => {
                if let (Some(sty), Some(t)) = (emit.style.role(other), &line.text) {
                    push_text(&mut s, &sty, t, line.size, scale);
                }
                // Unknown roles are silently skipped (permissive, like the rest of bookart).
            }
        }
    }
    if !foot.trim().is_empty() {
        // A flexible spacer pushes the imprint to the FOOT within the flow — so it never overlaps the
        // content (an absolute `place(bottom)` collides when the type box is short).
        s.push_str("\n  v(1fr)\n");
        s.push_str(&foot);
    }
    s.push_str("}\n\n");
    s.push_str("// Preview — compile this file directly; #import takes only `title-page`.\n");
    s.push_str("#title-page\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(v: f32) -> Margins {
        Margins { top: v, bottom: v, left: v, right: v }
    }

    #[test]
    fn letterpress_hierarchy_and_imprint_foot() {
        let lines = vec![
            TitleLine { role: "series".into(), text: Some("Educational manuals".into()), ..Default::default() },
            TitleLine { role: "title".into(), text: Some("Historical Grammar".into()), ..Default::default() },
            TitleLine { role: "part".into(), text: Some("Part I. Etymology".into()), ..Default::default() },
            TitleLine { role: "author".into(), text: Some("F. Buslaev".into()), ..Default::default() },
            TitleLine { role: "imprint".into(), text: Some("Moscow.\n1858".into()), ..Default::default() },
        ];
        let out = title_page_typst(148.0, 210.0, &m(24.0), None, 0.0, None, &lines, Emit::default());
        assert!(out.contains("#let title-page = {"), "reusable function:\n{out}");
        assert!(out.contains("#title-page"), "preview call");
        assert!(out.contains("upper(\"Historical Grammar\")"), "title uppercased");
        assert!(out.contains("weight: \"bold\""), "title bold");
        assert!(out.contains("smallcaps(\"Educational manuals\")"), "series small-caps");
        // Imprint is pushed to the FOOT by a flexible spacer, split on the newline into two centred lines.
        assert!(out.contains("v(1fr)"), "imprint pushed to the foot");
        assert!(out.contains("smallcaps(\"Moscow.\")") && out.contains("smallcaps(\"1858\")"), "imprint lines");
        assert!(!out.contains("background:"), "no border / no rule → no background");
    }

    #[test]
    fn border_fits_the_type_and_ornaments_place() {
        let lines = vec![
            TitleLine { role: "ornament".into(), src: Some("fleuron.png".into()), size: Some(20.0), ..Default::default() },
            TitleLine { role: "title".into(), text: Some("Bibliotheca".into()), size: Some(26.0), ..Default::default() },
            TitleLine { role: "rule".into(), ..Default::default() },
        ];
        let out = title_page_typst(148.0, 210.0, &m(30.0), Some(("frame.png", &m(12.0))), 0.0, Some("Libertinus Serif"), &lines, Emit::default());
        assert!(out.contains("#let border-image = \"frame.png\""), "border ref:\n{out}");
        assert!(out.contains("background: place(top + left, dx: 12mm"), "border placed at its margin");
        assert!(out.contains("set text(font: \"Libertinus Serif\""), "font applied");
        assert!(out.contains("image(\"fleuron.png\", width: 20mm)"), "ornament image");
        assert!(out.contains("size: 26pt"), "title size override");
        assert!(out.contains("line(length: 26%"), "default rule");
    }

    #[test]
    fn subchapter_is_subordinate_smallcaps() {
        let lines = vec![
            TitleLine { role: "subchapter".into(), text: Some("Section the First".into()), ..Default::default() },
            TitleLine { role: "title".into(), text: Some("The Harbour".into()), size: Some(20.0), ..Default::default() },
        ];
        let out = title_page_typst(148.0, 210.0, &m(24.0), None, 0.0, None, &lines, Emit::default());
        // Small-caps, generously tracked, and smaller than a `part` (which is 15pt bold) — a subordinate mark.
        assert!(out.contains("smallcaps(\"Section the First\")"), "subchapter is small-caps:\n{out}");
        assert!(out.contains("tracking: 0.09em"), "subchapter is generously tracked");
        assert!(out.contains("size: 12pt"), "subchapter default size 12pt");
    }

    #[test]
    fn width_aliases_size_for_images() {
        // On image/ornament lines `width` is the natural key; it must feed the same `size` field
        // (width-% for images) rather than being silently dropped to the 60% default.
        let line: TitleLine = deser_hjson::from_str(r#"{ role: "image", src: "e.png", width: 24 }"#).unwrap();
        assert_eq!(line.size, Some(24.0), "width populates size");
        let out = title_page_typst(148.0, 210.0, &m(15.0), None, 0.0, None, &[line], Emit::default());
        assert!(out.contains(r#"image("e.png", width: 24%)"#), "image honours authored width:\n{out}");
    }

    #[test]
    fn scale_shrinks_type_and_images() {
        let lines = vec![
            TitleLine { role: "title".into(), text: Some("Navigation".into()), ..Default::default() },
            TitleLine { role: "image".into(), src: Some("e.png".into()), size: Some(40.0), ..Default::default() },
        ];
        let out = title_page_typst(148.0, 210.0, &m(15.0), None, 0.0, None, &lines, Emit { scale: 0.5, historical: false, style: Style::Letterpress });
        assert!(out.contains("set text(size: 6pt"), "base size scaled 12→6:\n{out}");
        assert!(out.contains("size: 15pt"), "title 30pt scaled to 15pt");
        assert!(out.contains(r#"image("e.png", width: 20%)"#), "image 40% scaled to 20%");
    }

    #[test]
    fn historical_adds_old_style_and_ligatures() {
        let lines = vec![TitleLine { role: "title".into(), text: Some("MDCCXLI".into()), ..Default::default() }];
        let out = title_page_typst(148.0, 210.0, &m(15.0), None, 0.0, None, &lines, Emit { scale: 1.0, historical: true, style: Style::Letterpress });
        assert!(out.contains(r#"number-type: "old-style""#), "old-style figures:\n{out}");
        assert!(out.contains("features: (hlig: 1)"), "historical ligatures");
        let plain = title_page_typst(148.0, 210.0, &m(15.0), None, 0.0, None, &lines, Emit::default());
        assert!(!plain.contains("number-type"), "off by default");
    }

    #[test]
    fn styles_change_the_hand() {
        let lines = vec![
            TitleLine { role: "title".into(), text: Some("Navigation".into()), ..Default::default() },
            TitleLine { role: "rule".into(), ..Default::default() },
        ];
        let emit = |style| title_page_typst(148.0, 210.0, &m(20.0), None, 0.0, None, &lines, Emit { scale: 1.0, historical: false, style });
        // Letterpress: bold upper 30pt, 0.5pt rule.
        let lp = emit(Style::Letterpress);
        assert!(lp.contains("size: 30pt, weight: \"bold\"") && lp.contains(r#"upper("Navigation")"#), "letterpress:\n{lp}");
        assert!(lp.contains("stroke: 0.5pt"), "letterpress rule");
        // Engraved: regular weight, wide tracking.
        let en = emit(Style::Engraved);
        assert!(en.contains("size: 26pt, weight: \"regular\"") && en.contains("tracking: 0.12em"), "engraved:\n{en}");
        // Modern: as-authored case (no upper()), regular.
        let md = emit(Style::Modern);
        assert!(md.contains("size: 28pt, weight: \"regular\"") && md.contains(r##"[#"Navigation"]"##), "modern keeps case:\n{md}");
        // Playbill: huge bold, thick rule.
        let pb = emit(Style::Playbill);
        assert!(pb.contains("size: 44pt, weight: \"bold\"") && pb.contains("stroke: 1.4pt"), "playbill:\n{pb}");
        assert!(Style::from_name("Poster") == Style::Playbill, "alias resolves");
    }
}
