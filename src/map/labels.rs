//! MAP-3 — a tiny deterministic bitmap font for map labels. A hand-authored
//! 5×7 all-caps face (the classic cartographic small-caps look) rasterized at an
//! integer scale. No font asset, no `ab_glyph` rasterizer → the render stays
//! byte-stable across machines + toolchain versions (the corpus invariant).
//!
//! Labels are upper-cased and accent-folded to the covered glyph set (so
//! "Vethûn" → "VETHUN"); non-Latin scripts (the `language` field's `ar`/`ru`/`zh`)
//! want a real shaped font + `ab_glyph` — a later refinement noted in the roadmap.

use image::{Rgb, RgbImage};

pub const GLYPH_W: u32 = 5;
pub const GLYPH_H: u32 = 7;
/// Blank columns between glyphs (in font pixels).
const ADVANCE_GAP: u32 = 1;
/// Blank rows between text lines (in font pixels).
const LINE_GAP: u32 = 2;

/// Pixel width of `text` rendered at `scale` (integer font-pixels per cell).
pub fn text_width(text: &str, scale: u32) -> u32 {
    let n = folded(text).chars().count() as u32;
    if n == 0 {
        return 0;
    }
    (n * GLYPH_W + (n - 1) * ADVANCE_GAP) * scale
}

/// Pixel height of a single line at `scale`.
pub fn text_height(scale: u32) -> u32 {
    GLYPH_H * scale
}

/// Full line advance (glyph height + inter-line gap) at `scale`.
pub fn line_advance(scale: u32) -> u32 {
    (GLYPH_H + LINE_GAP) * scale
}

/// Draw `text` with its top-left at (`x`, `y`) in `color` at `scale`. Unknown
/// glyphs render as a blank cell. Out-of-bounds pixels are clipped.
pub fn draw_text(img: &mut RgbImage, x: i32, y: i32, text: &str, scale: u32, color: [u8; 3]) {
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

/// Draw `text` with a 1-font-pixel halo (4-neighbour outline in `halo`, then the
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
