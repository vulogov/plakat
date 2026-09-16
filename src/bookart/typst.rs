//! Typst artifact emission — a page where TEXT is the main character, framed by a border it can't overlap.
//!
//! This is *book art*, not art: the text is the subject and the ornament frames it. So the emitter never
//! crams text into a fixed border. Instead it MEASURES the border's real inner clear window — the largest
//! rectangle free of ink, corner motifs included ([`clear_window`]) — and fits the text box to it, so a
//! generated page never places text over the ornament ("do not generate what doesn't fit").
//!
//! Geometry:
//!   * the four page **margins** set the gap between the page edge and the border;
//!   * the **border is sized to the margin box** (`page − margins`) and drawn behind everything;
//!   * the **text box is the border's measured clear window**, so content flows only where the ornament
//!     leaves room — multi-page safe, because a Typst `background` repeats on every page;
//!   * `#place-on-page(dx, dy, ...)` still drops a single element at an absolute spot when you want one.
//!
//! The emitted `.typ` compiles to a PDF with `typst compile` (the CLI `--verify` flag). This module is the
//! pure builder + the pure measurement — no I/O — so both are gate-tested offline; the CLI handler loads
//! the border pixels, derives the ink mask, and drives compilation.

/// Per-side lengths in mm — reused for both the border placement (page edge → border) and the text box
/// (page edge → text).
pub struct Margins {
    pub top: f32,
    pub bottom: f32,
    pub left: f32,
    pub right: f32,
}

/// Page geometry + the border reference + the two boxes: where the border sits, and where text may go.
pub struct TypstPage {
    pub w_mm: f32,
    pub h_mm: f32,
    /// The border image AS REFERENCED from the `.typ` — a bare basename (the CLI copies it beside the file).
    pub border_ref: String,
    /// Border placement: page edge → border. The border is sized to `page − border`.
    pub border: Margins,
    /// Text box: page edge → text. Set from the border's measured clear window so text never overlaps it.
    pub text: Margins,
}

/// A restrained book frame: thin rules at the margin box + a small ornament at each of the four corners
/// (one image, mirrored so the corners match). The frame never dominates — the text keeps the interior.
pub struct CornerFrame {
    pub w_mm: f32,
    pub h_mm: f32,
    /// The corner ornament AS REFERENCED from the `.typ` (basename); placed once per corner, mirrored.
    pub corner_ref: String,
    /// Size of each corner ornament, in mm.
    pub corner_mm: f32,
    /// Frame rule thickness, in pt (0 → no connecting rules, corners only).
    pub rule_pt: f32,
    /// Page edge → frame (per side).
    pub margin: Margins,
}

/// Optional content placed inside the text box, so the artifact compiles to a non-empty PDF out of the box.
#[derive(Default)]
pub struct Placement {
    /// A centred heading at the top of the text box.
    pub title: Option<String>,
    /// Body markup (Typst syntax allowed). `None` → a `lorem` paragraph so the PDF is never blank.
    pub body: Option<String>,
    /// An image AS REFERENCED from the `.typ` (basename), placed centred under the body.
    pub image_ref: Option<String>,
    /// Placed-image width as a percentage of the text-box width (clamped 1–100).
    pub image_width_pct: u32,
}

/// The inner clear window of a border image, as fractions `[0,1]` of its width/height: the largest centred
/// rectangle free of ink. `ink(x, y)` reports whether a pixel is part of the ornament. Corner motifs
/// (rosettes) are included — the returned rectangle clears them too, so text fitted to it never overlaps
/// the border. If ink crosses the centre (not a frame), falls back to a modest 10% inset.
pub fn clear_window(w: u32, h: u32, ink: impl Fn(u32, u32) -> bool) -> (f32, f32, f32, f32) {
    if w < 4 || h < 4 {
        return (0.0, 0.0, 1.0, 1.0);
    }
    let (cx, cy) = (w / 2, h / 2);
    // Seed each side from the CENTRE CROSS — the middle row/column never crosses a corner motif, so it
    // reads the plain rule thickness (the full-width top/bottom rules would otherwise pollute a naive scan).
    let mut left = (0..cx).rev().find(|&x| ink(x, cy)).map(|x| x + 1).unwrap_or(0);
    let mut right = (cx..w).find(|&x| ink(x, cy)).unwrap_or(w).saturating_sub(1).max(left);
    let mut top = (0..cy).rev().find(|&y| ink(cx, y)).map(|y| y + 1).unwrap_or(0);
    let mut bottom = (cy..h).find(|&y| ink(cx, y)).unwrap_or(h).saturating_sub(1).max(top);
    // Iterate inward: within the current clear band, push each side past the innermost ink on that side —
    // this clears CORNER MOTIFS (rosettes) that reach beyond the rules into the band. Top/bottom are cleared
    // FIRST so corner motifs are escaped VERTICALLY, preserving the full text WIDTH — a book page wants a
    // wide text column, not a tall thin one.
    for _ in 0..6 {
        let prev = (left, top, right, bottom);
        let (mut nt, mut nb) = (0u32, h - 1);
        for x in (left + 1)..right {
            if let Some(y) = (0..cy).rev().find(|&y| ink(x, y)) {
                nt = nt.max(y + 1);
            }
            if let Some(y) = (cy..h).find(|&y| ink(x, y)) {
                nb = nb.min(y.saturating_sub(1));
            }
        }
        top = nt;
        bottom = nb.max(nt);
        let (mut nl, mut nr) = (0u32, w - 1);
        for y in (top + 1)..bottom {
            if let Some(x) = (0..cx).rev().find(|&x| ink(x, y)) {
                nl = nl.max(x + 1);
            }
            if let Some(x) = (cx..w).find(|&x| ink(x, y)) {
                nr = nr.min(x.saturating_sub(1));
            }
        }
        left = nl;
        right = nr.max(nl);
        if (left, top, right, bottom) == prev {
            break;
        }
    }
    let fx0 = left as f32 / w as f32;
    let fy0 = top as f32 / h as f32;
    let fx1 = (right as f32 + 1.0) / w as f32;
    let fy1 = (bottom as f32 + 1.0) / h as f32;
    // Inversion (ink through the centre) or a hairline window → a safe modest inset.
    if fx1 - fx0 < 0.2 || fy1 - fy0 < 0.2 {
        return (0.1, 0.1, 0.9, 0.9);
    }
    (fx0, fy0, fx1, fy1)
}

/// Given the border placement (page edge → border, mm), the page size, the measured clear-window fractions,
/// and a safety inset, return the text-box margins (page edge → text, mm) that keep text inside the window.
pub fn text_margins_from_window(
    w_mm: f32,
    h_mm: f32,
    border: &Margins,
    window: (f32, f32, f32, f32),
    safety_mm: f32,
) -> Margins {
    let bw = (w_mm - border.left - border.right).max(0.0);
    let bh = (h_mm - border.top - border.bottom).max(0.0);
    let (fx0, fy0, fx1, fy1) = window;
    Margins {
        left: border.left + fx0 * bw + safety_mm,
        right: border.right + (1.0 - fx1) * bw + safety_mm,
        top: border.top + fy0 * bh + safety_mm,
        bottom: border.bottom + (1.0 - fy1) * bh + safety_mm,
    }
}

/// Escape a string for a Typst `"..."` string literal (backslash + double-quote).
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Format a length in mm without trailing zeros: `148.0 → "148"`, `138.5 → "138.5"`.
fn mm(v: f32) -> String {
    let r = (v * 100.0).round() / 100.0;
    let mut s = format!("{r}");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    s
}

/// Build the full Typst source: a border sized to the margins, a text box fitted to its clear window, and
/// the placement API + seed content.
pub fn typst_artifact(page: &TypstPage, place: &Placement) -> String {
    let mut s = String::new();
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n");
    s.push_str("// plakat bookart — a bordered book page (TEXT is the subject), compilable + reusable.\n");
    s.push_str("//   • the border is sized to the margin box (page − margins), drawn behind everything\n");
    s.push_str("//   • the text box is fitted to the border's measured clear window — text never overlaps it\n");
    s.push_str("//   • `book-page` is a template: `set page(background: …)` puts the border on EVERY page,\n");
    s.push_str("//     so Typst paginates your text across as many pages as it needs.\n");
    s.push_str("// Preview:  typst compile <this-file>.typ\n");
    s.push_str("// In a book: #import \"<this-file>.typ\": book-page   then   #show: book-page\n");
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n\n");

    s.push_str(&format!("#let page-width   = {}mm\n", mm(page.w_mm)));
    s.push_str(&format!("#let page-height  = {}mm\n\n", mm(page.h_mm)));

    s.push_str("// Border placement (page edge → border); the border is sized to page − these.\n");
    s.push_str(&format!("#let border-top    = {}mm\n", mm(page.border.top)));
    s.push_str(&format!("#let border-bottom = {}mm\n", mm(page.border.bottom)));
    s.push_str(&format!("#let border-left   = {}mm\n", mm(page.border.left)));
    s.push_str(&format!("#let border-right  = {}mm\n", mm(page.border.right)));
    s.push_str(&format!("#let border-image  = \"{}\"\n\n", esc(&page.border_ref)));

    s.push_str("// Text box (page edge → text), fitted to the border's measured clear window.\n");
    s.push_str(&format!("#let text-top    = {}mm\n", mm(page.text.top)));
    s.push_str(&format!("#let text-bottom = {}mm\n", mm(page.text.bottom)));
    s.push_str(&format!("#let text-left   = {}mm\n", mm(page.text.left)));
    s.push_str(&format!("#let text-right  = {}mm\n\n", mm(page.text.right)));

    s.push_str("// Apply to a WHOLE book with `#show: book-page`. `set page(background: …)` repeats the border\n");
    s.push_str("// on EVERY page, so Typst paginates your text across as many pages as it needs — no page-by-page.\n");
    s.push_str("#let book-page(doc) = {\n");
    s.push_str("  set page(\n");
    s.push_str("    width: page-width, height: page-height,\n");
    s.push_str("    // Text flows only inside the border's clear window.\n");
    s.push_str("    margin: (top: text-top, bottom: text-bottom, left: text-left, right: text-right),\n");
    s.push_str("    // The border is SIZED TO THE MARGIN BOX (page − border-*) and drawn behind everything.\n");
    s.push_str("    background: place(top + left, dx: border-left, dy: border-top, image(\n");
    s.push_str("      border-image,\n");
    s.push_str("      width: page-width - border-left - border-right,\n");
    s.push_str("      height: page-height - border-top - border-bottom,\n");
    s.push_str("      fit: \"stretch\",\n");
    s.push_str("    )),\n");
    s.push_str("  )\n");
    s.push_str("  set text(size: 11pt)\n");
    s.push_str("  set par(justify: true, leading: 0.62em)\n");
    s.push_str("  doc\n");
    s.push_str("}\n\n");

    push_place_and_preview(&mut s, place);
    s
}

/// Emit `place-on-page`, the import/usage note, `#show: book-page`, and the preview content — shared by both
/// frame styles. Compiling the file directly renders the preview; `#import`-ing it takes only `book-page`.
fn push_place_and_preview(s: &mut String, place: &Placement) {
    s.push_str("// Absolute placement over the page (importable alongside book-page).\n");
    s.push_str(
        "#let place-on-page(dx: 0pt, dy: 0pt, alignment: top + left, body) = \
         place(alignment, dx: dx, dy: dy, body)\n\n",
    );
    s.push_str("// Put a run of text INSIDE A BOX you can size, pad, fill and align — pair it with\n");
    s.push_str("// place-on-page for a caption, a label, a pull-quote, or a titled panel over the page.\n");
    s.push_str("#let text-box(body, width: auto, height: auto, inset: 6pt, alignment: left, fill: none, stroke: none) = box(\n");
    s.push_str("  width: width, height: height, inset: inset, fill: fill, stroke: stroke,\n");
    s.push_str("  align(alignment, body),\n");
    s.push_str(")\n\n");
    s.push_str("// ── Preview below — compile this file directly to see it; IGNORED when imported. ──────\n");
    s.push_str("// In your book:  #import \"this-file.typ\": book-page   then   #show: book-page\n");
    s.push_str("#show: book-page\n");
    push_content(s, place);
}

/// Emit the content block (title / body / image) shared by both frame styles.
fn push_content(s: &mut String, place: &Placement) {
    if let Some(t) = &place.title {
        s.push_str(&format!("#align(center)[= {t}]\n\n"));
    }
    match place.body.as_deref().map(str::trim).filter(|b| !b.is_empty()) {
        Some(b) => {
            s.push_str(b);
            s.push('\n');
        }
        None => s.push_str("#lorem(60)\n"),
    }
    if let Some(img) = &place.image_ref {
        let w = place.image_width_pct.clamp(1, 100);
        s.push_str(&format!("\n#align(center)[#image(\"{}\", width: {w}%)]\n", esc(img)));
    }
}

/// Build a Typst page with a RESTRAINED corner frame: thin rules at the margin box + a small matching
/// ornament at each of the four corners (one image, mirrored). Text takes the whole interior — book art
/// where the text is the subject and the frame does not dominate.
pub fn typst_corner_frame(f: &CornerFrame, place: &Placement) -> String {
    let mut s = String::new();
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n");
    s.push_str("// plakat bookart — a book page with a RESTRAINED corner frame, compilable + reusable.\n");
    s.push_str("//   • thin rules at the margin box + a small ornament at each corner (mirrored to match)\n");
    s.push_str("//   • the frame never dominates — text takes the whole interior (text is the subject)\n");
    s.push_str("//   • `book-page` is a template: the frame repeats on EVERY page, so Typst paginates freely.\n");
    s.push_str("// Preview:  typst compile <this-file>.typ\n");
    s.push_str("// In a book: #import \"<this-file>.typ\": book-page   then   #show: book-page\n");
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n\n");

    s.push_str(&format!("#let page-width   = {}mm\n", mm(f.w_mm)));
    s.push_str(&format!("#let page-height  = {}mm\n", mm(f.h_mm)));
    s.push_str(&format!("#let margin-top    = {}mm\n", mm(f.margin.top)));
    s.push_str(&format!("#let margin-bottom = {}mm\n", mm(f.margin.bottom)));
    s.push_str(&format!("#let margin-left   = {}mm\n", mm(f.margin.left)));
    s.push_str(&format!("#let margin-right  = {}mm\n", mm(f.margin.right)));
    s.push_str(&format!("#let corner-size   = {}mm\n", mm(f.corner_mm)));
    s.push_str(&format!("#let rule-weight   = {}pt\n", mm(f.rule_pt)));
    s.push_str(&format!("#let corner-image  = \"{}\"\n\n", esc(&f.corner_ref)));

    // One SQUARE tile per corner, placed at exact (dx, dy) from the page's top-left and mirrored in place
    // (scale origin = centre, so the flip never overflows the tile box) — all four corners match.
    s.push_str("#let corner(dx, dy, sx, sy) = place(top + left, dx: dx, dy: dy,\n");
    s.push_str("  scale(x: sx, y: sy, reflow: false, image(corner-image, width: corner-size, height: corner-size)))\n\n");

    s.push_str("// Apply to a WHOLE book with `#show: book-page`. `set page(background: …)` repeats the frame\n");
    s.push_str("// on EVERY page, so Typst paginates your text across as many pages as it needs — no page-by-page.\n");
    s.push_str("#let book-page(doc) = {\n");
    s.push_str("  set page(\n");
    s.push_str("    width: page-width, height: page-height,\n");
    s.push_str("    // Text takes the interior; the frame lives in the margin + a corner-sized inset.\n");
    s.push_str("    margin: (\n");
    s.push_str("      top: margin-top + corner-size, bottom: margin-bottom + corner-size,\n");
    s.push_str("      left: margin-left + corner-size, right: margin-right + corner-size,\n");
    s.push_str("    ),\n");
    s.push_str("    background: {\n");
    s.push_str("      // thin frame rules at the margin box\n");
    s.push_str("      place(top + left, dx: margin-left, dy: margin-top, rect(\n");
    s.push_str("        width: page-width - margin-left - margin-right,\n");
    s.push_str("        height: page-height - margin-top - margin-bottom,\n");
    s.push_str("        stroke: if rule-weight > 0pt { rule-weight + black } else { none },\n");
    s.push_str("      ))\n");
    s.push_str("      // one square tile per corner, MIRRORED in place so all four match\n");
    s.push_str("      corner(margin-left, margin-top, 100%, 100%)\n");
    s.push_str("      corner(page-width - margin-right - corner-size, margin-top, -100%, 100%)\n");
    s.push_str("      corner(margin-left, page-height - margin-bottom - corner-size, 100%, -100%)\n");
    s.push_str("      corner(page-width - margin-right - corner-size, page-height - margin-bottom - corner-size, -100%, -100%)\n");
    s.push_str("    },\n");
    s.push_str("  )\n");
    s.push_str("  set text(size: 11pt)\n");
    s.push_str("  set par(justify: true, leading: 0.62em)\n");
    s.push_str("  doc\n");
    s.push_str("}\n\n");

    push_place_and_preview(&mut s, place);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(m: f32) -> Margins {
        Margins { top: m, bottom: m, left: m, right: m }
    }

    /// A synthetic border: an ink frame `band` px thick, plus (optionally) a solid corner blob of radius
    /// `corner` that reaches further inward than the frame.
    fn framed_ink(w: u32, h: u32, band: u32, corner: u32) -> impl Fn(u32, u32) -> bool {
        move |x, y| {
            let edge = x < band || y < band || x >= w - band || y >= h - band;
            let tl = x < corner && y < corner;
            edge || tl
        }
    }

    #[test]
    fn clear_window_of_a_plain_frame() {
        let (fx0, fy0, fx1, fy1) = clear_window(100, 100, framed_ink(100, 100, 10, 0));
        // A 10px band on a 100px image → clear window ~[0.1, 0.9].
        assert!((fx0 - 0.1).abs() < 0.02, "left {fx0}");
        assert!((fy0 - 0.1).abs() < 0.02, "top {fy0}");
        assert!((fx1 - 0.9).abs() < 0.02, "right {fx1}");
        assert!((fy1 - 0.9).abs() < 0.02, "bottom {fy1}");
    }

    #[test]
    fn corner_motif_is_cleared() {
        // A big top-left corner blob spanning [0,30)×[0,30). The clear window must exclude it — pushed in on
        // at least one axis (the greedy clears whichever side it reaches first; either is overlap-free).
        let (fx0, fy0, fx1, fy1) = clear_window(100, 100, framed_ink(100, 100, 10, 30));
        assert!(fx0 >= 0.29 || fy0 >= 0.29, "corner not cleared: ({fx0},{fy0})-({fx1},{fy1})");
        assert!(fx1 - fx0 > 0.4 && fy1 - fy0 > 0.4, "window collapsed: ({fx0},{fy0})-({fx1},{fy1})");
    }

    #[test]
    fn text_margins_fit_the_window() {
        // A5, border 12mm all round → border box 124×186. Window [0.1,0.9] + 2mm safety.
        let t = text_margins_from_window(148.0, 210.0, &uniform(12.0), (0.1, 0.1, 0.9, 0.9), 2.0);
        // left = 12 + 0.1*124 + 2 = 26.4 ; top = 12 + 0.1*186 + 2 = 32.6
        assert!((t.left - 26.4).abs() < 0.1, "left {}", t.left);
        assert!((t.top - 32.6).abs() < 0.1, "top {}", t.top);
        assert!((t.right - 26.4).abs() < 0.1, "right {}", t.right);
    }

    #[test]
    fn artifact_has_border_and_text_boxes_and_placement_api() {
        let page = TypstPage {
            w_mm: 148.0,
            h_mm: 210.0,
            border_ref: "border.png".into(),
            border: uniform(12.0),
            text: Margins { top: 32.6, bottom: 32.6, left: 26.4, right: 26.4 },
        };
        let out = typst_artifact(&page, &Placement::default());
        assert!(out.contains("#let border-left   = 12mm"), "border placement:\n{out}");
        assert!(out.contains("#let text-left   = 26.4mm"), "text box fitted to the window");
        assert!(out.contains("margin: (top: text-top"), "page margin = text box");
        assert!(out.contains("dx: border-left, dy: border-top"), "border placed at the margin box");
        assert!(out.contains("width: page-width - border-left - border-right"), "border sized to page − margins");
        // Reusable template: book-page wraps set page, applied to the whole doc via #show.
        assert!(out.contains("#let book-page(doc) = {"), "book-page template function");
        assert!(out.contains("#show: book-page"), "applied to the preview");
        assert!(out.contains("#let place-on-page("), "absolute placement function");
        assert!(out.contains("#let text-box(body,"), "text-box helper function");
        assert!(out.contains("#lorem(60)"), "default body");
    }

    #[test]
    fn corner_frame_places_four_mirrored_corners() {
        let f = CornerFrame {
            w_mm: 148.0,
            h_mm: 210.0,
            corner_ref: "corner.png".into(),
            corner_mm: 20.0,
            rule_pt: 0.6,
            margin: uniform(12.0),
        };
        let out = typst_corner_frame(&f, &Placement::default());
        assert!(out.contains("#let corner-size   = 20mm"), "corner size:\n{out}");
        assert!(out.contains("#let corner-image  = \"corner.png\""), "corner reference");
        // Four corners at exact coordinates, mirrored in place so they match: (+,+), (−,+), (+,−), (−,−).
        assert!(out.contains("corner(margin-left, margin-top, 100%, 100%)"), "top-left:\n{out}");
        assert!(out.contains("corner(page-width - margin-right - corner-size, margin-top, -100%, 100%)"), "top-right");
        assert!(out.contains("corner(margin-left, page-height - margin-bottom - corner-size, 100%, -100%)"), "bottom-left");
        assert!(out.contains("-100%, -100%)"), "bottom-right mirror-xy");
        // Thin rule frame + text keeps the interior (margin = margin + corner-size).
        assert!(out.contains("stroke: if rule-weight > 0pt"), "thin rule frame");
        assert!(out.contains("top: margin-top + corner-size"), "text takes the interior");
        // Same reusable template + helpers.
        assert!(out.contains("#let book-page(doc) = {"), "book-page template function");
        assert!(out.contains("#show: book-page"), "applied to the preview");
        assert!(out.contains("#let text-box(body,"), "text-box helper function");
        assert!(out.contains("#lorem(60)"), "default body");
    }

    #[test]
    fn placement_folds_in_title_body_and_image() {
        let page = TypstPage {
            w_mm: 148.0,
            h_mm: 210.0,
            border_ref: "b.png".into(),
            border: uniform(12.0),
            text: uniform(30.0),
        };
        let place = Placement {
            title: Some("Chapter One".into()),
            body: Some("First line.\nSecond line.".into()),
            image_ref: Some("plate.png".into()),
            image_width_pct: 45,
        };
        let out = typst_artifact(&page, &place);
        assert!(out.contains("#align(center)[= Chapter One]"), "title:\n{out}");
        assert!(out.contains("First line.\nSecond line."), "body flows verbatim");
        assert!(out.contains("#image(\"plate.png\", width: 45%)"), "placed image");
        assert!(!out.contains("#lorem"), "explicit body replaces the default");
    }
}
