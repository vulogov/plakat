//! Ornament layout against the text block (RFC BOOKART-1 §6.2). Pure: given the resolved page and the
//! ornament type, produce the placement rectangle(s) (in canvas px) the finished ornament is sized into.
//! The text block is derived from margins + gutter; ornaments anchor to it, not the raw page.

use crate::bookart::geometry::page::PageResolved;
use crate::bookart::spec::BookArtSpec;

/// A placement rectangle in canvas pixels, with optional mirroring (corner pieces point inward).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub flip_h: bool,
    pub flip_v: bool,
}

impl Rect {
    fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Rect { x: x.max(0) as u32, y: y.max(0) as u32, w: w.max(1) as u32, h: h.max(1) as u32, flip_h: false, flip_v: false }
    }
    fn flipped(mut self, h: bool, v: bool) -> Self {
        self.flip_h = h;
        self.flip_v = v;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub rects: Vec<Rect>,
}

fn mm2px(mm: f32, dpi: u32) -> i32 {
    (mm / 25.4 * dpi as f32).round() as i32
}

/// The text block (content area) in canvas px, from the page margins + gutter (with book defaults).
pub fn text_block(page: &PageResolved, spec: &BookArtSpec) -> Rect {
    let dpi = page.dpi;
    let m = spec.page.as_ref().and_then(|p| p.margins.as_ref());
    let top = mm2px(m.and_then(|m| m.top).unwrap_or(18.0), dpi);
    let bottom = mm2px(m.and_then(|m| m.bottom).unwrap_or(20.0), dpi);
    let inner = mm2px(m.and_then(|m| m.inner).unwrap_or(18.0), dpi);
    let outer = mm2px(m.and_then(|m| m.outer).unwrap_or(15.0), dpi);
    let gutter = mm2px(spec.page.as_ref().and_then(|p| p.gutter_mm).unwrap_or(0.0), dpi);
    let x = inner + gutter;
    Rect::new(x, top, page.w_px as i32 - x - outer, page.h_px as i32 - top - bottom)
}

/// The placement rectangle(s) for an ornament type within a text block.
pub fn layout_for(kind: &str, tb: &Rect) -> Layout {
    let (tx, ty, tw, th) = (tb.x as i32, tb.y as i32, tb.w as i32, tb.h as i32);
    let rects = match kind {
        // a band across the top of the text block, aspect ~5:1.
        "headpiece" => vec![Rect::new(tx, ty, tw, tw / 5)],
        // a tapering ornament centred below the last line (its own bbox is ~1.6:1).
        "tailpiece" => {
            let w = (tw as f32 * 0.6) as i32;
            let h = (w as f32 * 0.6) as i32;
            vec![Rect::new(tx + (tw - w) / 2, ty + th - h, w, h)]
        }
        // a thin centred rule / mark.
        "divider" | "rule" => {
            let w = (tw as f32 * 0.5) as i32;
            let h = (tw as f32 * 0.06) as i32;
            vec![Rect::new(tx + (tw - w) / 2, ty + th / 2 - h / 2, w, h)]
        }
        "fleuron" | "dinkus" => {
            let s = (tw as f32 * 0.12) as i32;
            vec![Rect::new(tx + (tw - s) / 2, ty + th / 2 - s / 2, s, s)]
        }
        // the border occupies the whole text block (assembled from edge/corner units in B3).
        "border" | "endpaper" => vec![Rect::new(tx, ty, tw, th)],
        // four inward-pointing corner squares.
        "corner" => {
            let s = (tw.min(th) as f32 * 0.18) as i32;
            vec![
                Rect::new(tx, ty, s, s),
                Rect::new(tx + tw - s, ty, s, s).flipped(true, false),
                Rect::new(tx, ty + th - s, s, s).flipped(false, true),
                Rect::new(tx + tw - s, ty + th - s, s, s).flipped(true, true),
            ]
        }
        // a decorated initial: a square cell (3 text lines ≈ tw/8) at the top-left of the block.
        "initial" => {
            let s = (tw as f32 * 0.18) as i32;
            vec![Rect::new(tx, ty, s, s)]
        }
        // page-fill pictorial.
        "frontispiece" => vec![Rect::new(tx, ty, tw, th)],
        // a centred spot (~55% of the block).
        "vignette" | "marginalia" | "colophon" => {
            let w = (tw as f32 * 0.55) as i32;
            let h = w;
            vec![Rect::new(tx + (tw - w) / 2, ty + (th - h) / 2, w, h)]
        }
        _ => vec![*tb],
    };
    Layout { rects }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bookart::geometry::page::resolve_page;

    fn a5() -> PageResolved {
        resolve_page(None) // a5 / 300 / portrait → 1748 × 2480
    }

    #[test]
    fn text_block_inside_page_with_margins() {
        let p = a5();
        let tb = text_block(&p, &BookArtSpec::default());
        assert!(tb.x > 0 && tb.y > 0);
        assert!(tb.x + tb.w < p.w_px && tb.y + tb.h < p.h_px, "text block must fit inside the page");
    }

    #[test]
    fn headpiece_is_a_top_band_spanning_the_block() {
        let tb = text_block(&a5(), &BookArtSpec::default());
        let l = layout_for("headpiece", &tb);
        assert_eq!(l.rects.len(), 1);
        let r = l.rects[0];
        assert_eq!((r.x, r.y, r.w), (tb.x, tb.y, tb.w), "spans the block at its top");
        assert!(r.h < r.w, "a band is wider than tall");
    }

    #[test]
    fn corner_is_four_inward_flipped_squares() {
        let tb = text_block(&a5(), &BookArtSpec::default());
        let l = layout_for("corner", &tb);
        assert_eq!(l.rects.len(), 4);
        assert_eq!((l.rects[0].flip_h, l.rects[0].flip_v), (false, false));
        assert_eq!((l.rects[3].flip_h, l.rects[3].flip_v), (true, true));
    }
}
