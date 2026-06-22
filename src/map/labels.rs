//! MAP-3 — a tiny deterministic bitmap font for map labels. A hand-authored
//! 5×7 all-caps face (the classic cartographic small-caps look) rasterized at an
//! integer scale. No font asset, no rasterizer dep → the render stays byte-stable
//! across machines + toolchain versions (the corpus invariant). This is the
//! **default**, and it is Latin-only (accents fold: "Vethûn" → "VETHUN").
//!
//! For **non-Latin** scripts (the `language` field's `ru`/`zh`/…) build with
//! `--features shaped-labels` and pass `plakat map --map-font <PATH.ttf>`: a real
//! TrueType/OpenType font is rasterized via `ab_glyph` and overrides the bitmap
//! face for every label. (Cyrillic + CJK render directly; complex scripts that
//! need contextual shaping/RTL — e.g. Arabic — render glyphs but unshaped, pending
//! a full shaper.) No font is vendored — the user supplies one.

use image::{Rgb, RgbImage};

pub const GLYPH_W: u32 = 5;
pub const GLYPH_H: u32 = 7;
/// Blank columns between glyphs (in font pixels).
const ADVANCE_GAP: u32 = 1;
/// Blank rows between text lines (in font pixels).
const LINE_GAP: u32 = 2;

/// Pixel width of `text` rendered at `scale` (integer font-pixels per cell).
pub fn text_width(text: &str, scale: u32) -> u32 {
    #[cfg(feature = "shaped-labels")]
    if let Some(w) = shaped::with_font(|f| f.text_width(text, scale)) {
        return w;
    }
    let n = folded(text).chars().count() as u32;
    if n == 0 {
        return 0;
    }
    (n * GLYPH_W + (n - 1) * ADVANCE_GAP) * scale
}

/// Pixel height of a single line at `scale`.
pub fn text_height(scale: u32) -> u32 {
    #[cfg(feature = "shaped-labels")]
    if let Some(h) = shaped::with_font(|f| f.text_height(scale)) {
        return h;
    }
    GLYPH_H * scale
}

/// Full line advance (glyph height + inter-line gap) at `scale`.
pub fn line_advance(scale: u32) -> u32 {
    #[cfg(feature = "shaped-labels")]
    if let Some(a) = shaped::with_font(|f| f.line_advance(scale)) {
        return a;
    }
    (GLYPH_H + LINE_GAP) * scale
}

/// Draw `text` with its top-left at (`x`, `y`) in `color` at `scale`. Unknown
/// glyphs render as a blank cell. Out-of-bounds pixels are clipped. When a shaped
/// font is active (`--map-font`), it rasterizes instead of the bitmap face.
pub fn draw_text(img: &mut RgbImage, x: i32, y: i32, text: &str, scale: u32, color: [u8; 3]) {
    #[cfg(feature = "shaped-labels")]
    if shaped::with_font(|f| f.draw_text(img, x, y, text, scale, color)).is_some() {
        return;
    }
    let s = scale.max(1) as i32;
    let mut pen = x;
    for ch in folded(text).chars() {
        if let Some(rows) = pattern(ch) {
            for (ry, row) in rows.iter().enumerate() {
                for (cx, cell) in row.bytes().enumerate() {
                    if cell == b'#' {
                        fill_cell(img, pen + cx as i32 * s, y + ry as i32 * s, s, color);
                    }
                }
            }
        }
        pen += (GLYPH_W + ADVANCE_GAP) as i32 * s;
    }
}

/// Draw `text` with a 1-font-pixel halo (8-neighbour outline in `halo`, then the
/// body in `color`) so it stays legible over busy terrain.
pub fn draw_text_haloed(img: &mut RgbImage, x: i32, y: i32, text: &str, scale: u32, color: [u8; 3], halo: [u8; 3]) {
    let s = scale.max(1) as i32;
    for (dx, dy) in [(-s, 0), (s, 0), (0, -s), (0, s), (-s, -s), (s, -s), (-s, s), (s, s)] {
        draw_text(img, x + dx, y + dy, text, scale, halo);
    }
    draw_text(img, x, y, text, scale, color);
}

fn fill_cell(img: &mut RgbImage, x0: i32, y0: i32, s: i32, color: [u8; 3]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for dy in 0..s {
        for dx in 0..s {
            let (px, py) = (x0 + dx, y0 + dy);
            if px >= 0 && py >= 0 && px < w && py < h {
                img.put_pixel(px as u32, py as u32, Rgb(color));
            }
        }
    }
}

/// Shaped (TrueType/OpenType) labels via `ab_glyph` — the `shaped-labels` feature.
/// A process-wide active font (set by `--map-font`) overrides the bitmap face for
/// every label. Per-glyph, left-to-right, no contextual shaping (fine for Cyrillic
/// + CJK; Arabic renders unshaped). The output is deterministic for a given font.
#[cfg(feature = "shaped-labels")]
pub mod shaped {
    use super::RgbImage;
    use ab_glyph::{Font, FontVec, ScaleFont};
    use image::Rgb;
    use std::cell::RefCell;

    thread_local! {
        static FONT: RefCell<Option<ShapedFont>> = const { RefCell::new(None) };
    }

    /// Load a TTF/OTF from disk and make it the active label font (for this thread).
    pub fn load_font(path: &std::path::Path) -> anyhow::Result<()> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading --map-font {}: {e}", path.display()))?;
        let font = FontVec::try_from_vec(bytes)
            .map_err(|e| anyhow::anyhow!("parsing font {}: {e}", path.display()))?;
        FONT.with(|f| *f.borrow_mut() = Some(ShapedFont { font }));
        Ok(())
    }

    /// Run `f` with the active font if one is set, else `None` (bitmap fallback).
    pub(super) fn with_font<R>(f: impl FnOnce(&ShapedFont) -> R) -> Option<R> {
        FONT.with(|cell| cell.borrow().as_ref().map(f))
    }

    pub(super) struct ShapedFont {
        font: FontVec,
    }

    impl ShapedFont {
        /// Pixel height for a bitmap `scale` (≈ the 7px·scale the bitmap face uses).
        fn px(&self, scale: u32) -> f32 {
            (scale.max(1) * 9) as f32
        }

        pub(super) fn text_width(&self, text: &str, scale: u32) -> u32 {
            let sf = self.font.as_scaled(self.px(scale));
            let mut w = 0.0f32;
            let mut prev = None;
            for c in text.chars() {
                let id = sf.glyph_id(c);
                if let Some(p) = prev {
                    w += sf.kern(p, id);
                }
                w += sf.h_advance(id);
                prev = Some(id);
            }
            w.ceil().max(0.0) as u32
        }

        pub(super) fn text_height(&self, scale: u32) -> u32 {
            self.px(scale).ceil() as u32
        }

        pub(super) fn line_advance(&self, scale: u32) -> u32 {
            let sf = self.font.as_scaled(self.px(scale));
            (sf.height() + sf.line_gap()).ceil() as u32
        }

        /// Rasterize `text` with its top-left at (x, y), baseline at y + ascent.
        pub(super) fn draw_text(&self, img: &mut RgbImage, x: i32, y: i32, text: &str, scale: u32, color: [u8; 3]) {
            let sf = self.font.as_scaled(self.px(scale));
            let (iw, ih) = (img.width() as i32, img.height() as i32);
            let baseline = y as f32 + sf.ascent();
            let mut caret = x as f32;
            let mut prev = None;
            for c in text.chars() {
                let id = sf.glyph_id(c);
                if let Some(p) = prev {
                    caret += sf.kern(p, id);
                }
                let mut glyph = sf.scaled_glyph(c);
                glyph.position = ab_glyph::point(caret, baseline);
                if let Some(outline) = self.font.outline_glyph(glyph) {
                    let bb = outline.px_bounds();
                    outline.draw(|gx, gy, cov| {
                        if cov <= 0.02 {
                            return;
                        }
                        let (px, py) = (bb.min.x as i32 + gx as i32, bb.min.y as i32 + gy as i32);
                        if px < 0 || py < 0 || px >= iw || py >= ih {
                            return;
                        }
                        let bg = img.get_pixel(px as u32, py as u32).0;
                        let a = cov.clamp(0.0, 1.0);
                        let blend = |b: u8, f: u8| (b as f32 * (1.0 - a) + f as f32 * a).round() as u8;
                        img.put_pixel(px as u32, py as u32, Rgb([blend(bg[0], color[0]), blend(bg[1], color[1]), blend(bg[2], color[2])]));
                    });
                }
                caret += sf.h_advance(id);
                prev = Some(id);
            }
        }
    }
}

/// Upper-case + fold accents/punctuation to the covered glyph set.
fn folded(text: &str) -> String {
    text.to_uppercase().chars().map(fold_char).collect()
}

/// Map an upper-cased char to a glyph the font covers (Latin-1 accents → base).
fn fold_char(c: char) -> char {
    match c {
        'À'..='Å' => 'A',
        'Ç' => 'C',
        'È'..='Ë' => 'E',
        'Ì'..='Ï' => 'I',
        'Ñ' => 'N',
        'Ò'..='Ö' | 'Ø' => 'O',
        'Ù'..='Ü' => 'U',
        'Ý' => 'Y',
        '’' | '‘' | '`' => '\'',
        '–' | '—' => '-',
        _ => c,
    }
}

/// 5×7 glyph bitmap (7 rows of exactly 5 chars; `#` = ink). `None` = no glyph.
fn pattern(c: char) -> Option<[&'static str; 7]> {
    Some(match c {
        ' ' => ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
        'A' => [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
        'B' => ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
        'C' => [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
        'D' => ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
        'E' => ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
        'F' => ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
        'G' => [" ### ", "#   #", "#    ", "#  ##", "#   #", "#   #", " ### "],
        'H' => ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
        'I' => ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
        'J' => ["#####", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "],
        'K' => ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
        'L' => ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
        'M' => ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"],
        'N' => ["#   #", "##  #", "# # #", "# # #", "#  ##", "#   #", "#   #"],
        'O' => [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
        'P' => ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
        'Q' => [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
        'R' => ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
        'S' => [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
        'T' => ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
        'U' => ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
        'V' => ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
        'W' => ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
        'X' => ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
        'Y' => ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
        'Z' => ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
        '0' => [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
        '1' => ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
        '2' => [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"],
        '3' => ["#####", "    #", "   # ", "  ## ", "    #", "#   #", " ### "],
        '4' => ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
        '5' => ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
        '6' => [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
        '7' => ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
        '8' => [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
        '9' => [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
        '.' => ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "],
        ',' => ["     ", "     ", "     ", "     ", " ##  ", " ##  ", " #   "],
        '\'' => ["  #  ", "  #  ", "  #  ", "     ", "     ", "     ", "     "],
        '-' => ["     ", "     ", "     ", " ### ", "     ", "     ", "     "],
        ':' => ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "],
        '&' => [" ##  ", "#  # ", "#  # ", " ##  ", "#  # ", "#   #", " ## #"],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "shaped-labels")]
    #[test]
    fn no_active_font_falls_back_to_the_bitmap_face() {
        // With no `--map-font` loaded, the shaped path is inactive → bitmap metrics.
        assert!(shaped::with_font(|_| ()).is_none());
        assert_eq!(text_width("AB", 2), (GLYPH_W * 2 + ADVANCE_GAP) * 2);
    }

    #[test]
    fn every_glyph_row_is_five_columns() {
        // The whole printable upper-ASCII set the renderer can throw at it.
        let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,'-:&";
        for ch in chars.chars() {
            let p = pattern(ch).unwrap_or_else(|| panic!("missing glyph {ch:?}"));
            for (i, row) in p.iter().enumerate() {
                assert_eq!(row.chars().count(), GLYPH_W as usize, "glyph {ch:?} row {i} not 5 wide");
            }
            assert_eq!(p.len(), GLYPH_H as usize);
        }
    }

    #[test]
    fn accents_fold_into_the_glyph_set() {
        // "Vethûn" upper-cases + folds to all-covered glyphs.
        for ch in folded("The Isle of Vethûn").chars() {
            assert!(pattern(ch).is_some(), "unfolded glyph {ch:?}");
        }
        assert_eq!(folded("Vethûn"), "VETHUN");
    }

    #[test]
    fn width_matches_drawn_extent() {
        assert_eq!(text_width("", 2), 0);
        assert_eq!(text_width("A", 2), GLYPH_W * 2);
        assert_eq!(text_width("AB", 3), (GLYPH_W * 2 + ADVANCE_GAP) * 3);
    }

    #[test]
    fn draw_is_deterministic_and_in_bounds() {
        let render = || {
            let mut img = RgbImage::new(80, 16);
            draw_text_haloed(&mut img, 2, 2, "Saltmere", 2, [0, 0, 0], [255, 255, 255]);
            img
        };
        assert!(render().as_raw() == render().as_raw());
        // Drew *something* (not a blank image).
        assert!(render().pixels().any(|p| p.0 == [0, 0, 0]), "text drew ink");
    }
}
