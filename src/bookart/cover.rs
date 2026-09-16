//! Book COVER / dust-jacket layout as compilable Typst — the three panels of a wrap (back · spine · front)
//! on one wide sheet, with the **spine width computed from the page count** (× paper caliper + boards), and
//! optional flaps. Reuses the `title-page` type engine: each panel is a styled stack of the same roles
//! (`title`/`subtitle`/`author`/`imprint`/`ornament`/`image`/…), so a cover shares its book's hand.
//!
//! Weight-free: the typography IS the artefact. Fold lines are drawn as light dashed guides so the emitted
//! `.typ` is a usable printer's layout.

use crate::bookart::titlepage::{render_stack, Emit, TitleLine};
use crate::bookart::typst::{esc, mm};
use serde::Deserialize;

/// The HJSON a cover is authored in.
#[derive(Deserialize, Default)]
pub struct CoverSpec {
    /// Typographic style (`letterpress`/`engraved`/`modern`/`playbill`); CLI `--style` overrides.
    #[serde(default)]
    pub style: Option<String>,
    /// Trim (page) size name (`a5`/`b5`/…); CLI `--page` overrides.
    #[serde(default)]
    pub page: Option<String>,
    /// Historical typography (old-style figures + historical ligatures). CLI `--historical` also sets it.
    #[serde(default)]
    pub historical: Option<bool>,
    /// Page count — drives the spine width (leaves × caliper). CLI `--pages` overrides.
    #[serde(default)]
    pub pages: Option<u32>,
    /// Paper caliper in mm per PAGE (default 0.06 ≈ 80–90 gsm text). CLI `--paper` overrides.
    #[serde(default)]
    pub paper: Option<f32>,
    /// Extra spine width in mm for hardcover boards (added to the paper bulk). Default 0 (paperback).
    #[serde(default)]
    pub board: Option<f32>,
    /// Explicit spine width in mm — overrides the page-count computation when set.
    #[serde(default)]
    pub spine_mm: Option<f32>,
    /// Flap width in mm (0/absent = a plain paperback cover with no flaps). CLI `--flap` overrides.
    #[serde(default)]
    pub flap: Option<f32>,
    /// Optional front-cover background image (a rendered ornament/plate), fitted behind the front type.
    #[serde(default)]
    pub border: Option<String>,
    /// Front-cover lines (title / author / an emblem …).
    #[serde(default)]
    pub front: Vec<TitleLine>,
    /// Spine lines (title / author), set small and rotated to read top-to-bottom.
    #[serde(default)]
    pub spine: Vec<TitleLine>,
    /// Back-cover lines (blurb `note`/`epigraph`, an `imprint` at the foot …).
    #[serde(default)]
    pub back: Vec<TitleLine>,
}

/// Computed spine width in mm from a page count, paper caliper (mm/page) and board thickness. Clamped to a
/// sane minimum so a thin book still gets a printable spine.
pub fn spine_width_mm(pages: u32, paper_mm_per_page: f32, board_mm: f32) -> f32 {
    (pages as f32 * paper_mm_per_page.max(0.0) + board_mm.max(0.0)).max(3.0)
}

/// The panel geometry of a laid-flat cover, left → right on the OUTSIDE: back flap · back · spine · front ·
/// front flap. All in mm.
pub struct CoverLayout {
    pub trim_w: f32,
    pub trim_h: f32,
    pub spine_w: f32,
    pub flap_w: f32,
}

impl CoverLayout {
    pub fn total_w(&self) -> f32 {
        2.0 * self.flap_w + 2.0 * self.trim_w + self.spine_w
    }
    fn x_back(&self) -> f32 {
        self.flap_w
    }
    fn x_spine(&self) -> f32 {
        self.flap_w + self.trim_w
    }
    fn x_front(&self) -> f32 {
        self.flap_w + self.trim_w + self.spine_w
    }
}

/// A front/back panel: a fixed box with the styled stack centred vertically (a `v(1fr)` above and below).
/// `bg` optionally places a full-bleed image behind the type (the front-cover ornament).
fn panel(dx: f32, w: f32, h: f32, inset: f32, base_pt: &str, hist: &str, bg: Option<&str>, body: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!("  place(top + left, dx: {}mm, dy: 0mm, box(width: {}mm, height: {}mm)[\n", mm(dx), mm(w), mm(h)));
    if let Some(img) = bg {
        // A full-panel background image (behind the type), stretched to the panel.
        s.push_str(&format!(
            "    #place(top + left, image(\"{}\", width: {}mm, height: {}mm, fit: \"cover\"))\n",
            esc(img),
            mm(w),
            mm(h),
        ));
    }
    s.push_str(&format!("    #box(width: {}mm, height: {}mm, inset: {}mm)[#{{\n", mm(w), mm(h), mm(inset)));
    s.push_str(&format!("      set text(size: {base_pt}pt{hist})\n"));
    s.push_str("      set par(leading: 0.7em, justify: false)\n");
    s.push_str("      set align(center)\n");
    s.push_str("      v(1fr)\n");
    for line in body.lines() {
        if line.trim().is_empty() {
            continue;
        }
        s.push_str("    ");
        s.push_str(line);
        s.push('\n');
    }
    s.push_str("      v(1fr)\n");
    s.push_str("    }]\n");
    s.push_str("  ])\n");
    s
}

/// A dashed vertical fold/edge guide at `x` (mm), full page height.
fn fold(x: f32, h: f32) -> String {
    format!(
        "  place(top + left, dx: {}mm, dy: 0mm, line(length: {}mm, angle: 90deg, \
         stroke: (paint: luma(60%), thickness: 0.3pt, dash: \"dashed\")))\n",
        mm(x),
        mm(h),
    )
}

/// Build the full Typst source for a cover / dust jacket.
pub fn cover_typst(layout: &CoverLayout, front: &[TitleLine], spine: &[TitleLine], back: &[TitleLine], front_bg: Option<&str>, emit: Emit) -> String {
    let scale = if emit.scale.is_finite() && emit.scale > 0.05 { emit.scale } else { 1.0 };
    let base = trim(12.0 * scale);
    let hist = if emit.historical { ", number-type: \"old-style\", features: (hlig: 1)" } else { "" };
    let total_w = layout.total_w();
    let h = layout.trim_h;
    let has_flaps = layout.flap_w > 0.01;

    let mut s = String::new();
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n");
    s.push_str("// plakat bookart — a book COVER / dust jacket, laid flat (back · spine · front), compilable.\n");
    s.push_str(&format!(
        "//   trim {:.0}×{:.0} mm · spine {:.1} mm · {}· total {:.0}×{:.0} mm\n",
        layout.trim_w,
        layout.trim_h,
        layout.spine_w,
        if has_flaps { format!("flaps {:.0} mm ", layout.flap_w) } else { String::new() },
        total_w,
        h,
    ));
    s.push_str("//   `cover` is reusable: compile this file directly, or #import it.\n");
    s.push_str("// ─────────────────────────────────────────────────────────────────────\n\n");

    s.push_str("#let cover = {\n");
    s.push_str(&format!("  set page(width: {}mm, height: {}mm, margin: 0mm)\n", mm(total_w), mm(h)));

    // Fold / edge guides (dashed) at every panel boundary.
    if has_flaps {
        s.push_str(&fold(layout.flap_w, h));
        s.push_str(&fold(layout.total_w() - layout.flap_w, h));
    }
    s.push_str(&fold(layout.x_spine(), h));
    s.push_str(&fold(layout.x_front(), h));

    // Back panel.
    s.push_str(&panel(layout.x_back(), layout.trim_w, h, 14.0, &base, hist, None, &render_stack(back, emit)));

    // Front panel (with optional background ornament).
    s.push_str(&panel(layout.x_front(), layout.trim_w, h, 14.0, &base, hist, front_bg, &render_stack(front, emit)));

    // Spine panel — type rotated to read top-to-bottom, centred along the spine height.
    if !spine.is_empty() {
        let spine_body = render_stack(spine, emit);
        s.push_str(&format!("  place(top + left, dx: {}mm, dy: 0mm, box(width: {}mm, height: {}mm)[\n", mm(layout.x_spine()), mm(layout.spine_w), mm(h)));
        s.push_str("    #set align(center + horizon)\n");
        s.push_str(&format!("    #rotate(90deg, reflow: true, box(width: {}mm)[#{{\n", mm(h - 24.0)));
        s.push_str(&format!("      set text(size: {base}pt{hist})\n"));
        s.push_str("      set par(leading: 0.7em, justify: false)\n");
        s.push_str("      set align(center)\n");
        for line in spine_body.lines() {
            if line.trim().is_empty() {
                continue;
            }
            s.push_str("    ");
            s.push_str(line);
            s.push('\n');
        }
        s.push_str("    }])\n");
        s.push_str("  ])\n");
    }

    s.push_str("}\n\n");
    s.push_str("// Preview — compile this file directly; #import takes only `cover`.\n");
    s.push_str("#cover\n");
    s
}

fn trim(v: f32) -> String {
    let r = (v * 100.0).round() / 100.0;
    let mut s = format!("{r}");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spine_width_from_pages() {
        // 320 pages at 0.06 mm/page ≈ 19.2 mm; a board adds to it; a thin book clamps to 3 mm.
        assert!((spine_width_mm(320, 0.06, 0.0) - 19.2).abs() < 0.01);
        assert!((spine_width_mm(320, 0.06, 4.0) - 23.2).abs() < 0.01);
        assert!((spine_width_mm(10, 0.06, 0.0) - 3.0).abs() < 0.01, "thin book clamps");
    }

    #[test]
    fn layout_places_three_panels_and_folds() {
        let layout = CoverLayout { trim_w: 148.0, trim_h: 210.0, spine_w: 20.0, flap_w: 0.0 };
        assert!((layout.total_w() - (2.0 * 148.0 + 20.0)).abs() < 0.01);
        let front = vec![TitleLine { role: "title".into(), text: Some("The Open Sea".into()), ..Default::default() }];
        let spine = vec![TitleLine { role: "title".into(), text: Some("THE OPEN SEA".into()), size: Some(12.0), ..Default::default() }];
        let back = vec![TitleLine { role: "note".into(), text: Some("A tale of the deep.".into()), ..Default::default() }];
        let out = cover_typst(&layout, &front, &spine, &back, None, Emit::default());
        assert!(out.contains("#let cover = {"), "reusable cover:\n{out}");
        assert!(out.contains("set page(width: 316mm, height: 210mm"), "wide sheet");
        assert!(out.contains("rotate(90deg"), "spine rotated");
        // Front panel sits to the RIGHT of the spine (dx = flap + trim + spine = 168mm).
        assert!(out.contains("dx: 168mm"), "front panel placed right of spine");
        assert!(out.contains("angle: 90deg"), "fold guides drawn");
    }

    #[test]
    fn flaps_widen_the_sheet() {
        let layout = CoverLayout { trim_w: 148.0, trim_h: 210.0, spine_w: 20.0, flap_w: 80.0 };
        assert!((layout.total_w() - (2.0 * 80.0 + 2.0 * 148.0 + 20.0)).abs() < 0.01);
        let out = cover_typst(&layout, &[], &[], &[], None, Emit::default());
        assert!(out.contains("flaps 80 mm"), "flap note:\n{out}");
    }
}
