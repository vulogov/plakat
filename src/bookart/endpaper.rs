//! ENDPAPER / decorative pattern: a seamless repeating diaper tiled from a single motif (a fleuron, a
//! dinkus, any transparent B/W ornament) across a page canvas — the patterned sheets pasted inside a
//! book's boards. Weight-free: no diffusion, pure compositing. Three lattices — a straight `grid`, a
//! `half-drop`, and a diagonal `diamond` diaper — with motifs clipped at the trim so the pattern reads as
//! continuing past the page.

use image::{Rgba, RgbaImage};

/// The repeat lattice.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// A straight grid — motifs aligned in rows and columns.
    Grid,
    /// Alternate columns dropped by half a cell (a half-drop repeat).
    HalfDrop,
    /// A diagonal diaper — alternate rows shifted half a cell, rows half-spaced.
    Diamond,
}

impl Layout {
    /// Permissive name resolution (unknown → grid).
    pub fn from_name(s: &str) -> Layout {
        match s.trim().to_lowercase().replace('_', "-").as_str() {
            "half-drop" | "halfdrop" | "drop" => Layout::HalfDrop,
            "diamond" | "diaper" | "diagonal" => Layout::Diamond,
            _ => Layout::Grid,
        }
    }
}

/// Endpaper render options (pixels).
pub struct EndpaperOpts {
    pub w: u32,
    pub h: u32,
    /// Target motif width in px (its height scales with the motif's aspect).
    pub tile: u32,
    /// Gap between motif cells in px.
    pub gap: u32,
    pub layout: Layout,
    /// Optional background fill (RGBA); `None` = transparent.
    pub bg: Option<[u8; 4]>,
}

/// The lattice node CENTRES (px) for a layout, starting one cell off-canvas so the trim clips motifs.
pub fn nodes(w: u32, h: u32, cell: u32, layout: Layout) -> Vec<(i64, i64)> {
    let cell = cell.max(1) as i64;
    let (w, h) = (w as i64, h as i64);
    let mut out = Vec::new();
    match layout {
        Layout::Grid | Layout::HalfDrop => {
            let mut col = 0;
            let mut x = 0i64;
            while x <= w + cell {
                let ydrop = if layout == Layout::HalfDrop && col % 2 == 1 { cell / 2 } else { 0 };
                let mut y = -cell + ydrop;
                while y <= h + cell {
                    out.push((x, y));
                    y += cell;
                }
                x += cell;
                col += 1;
            }
        }
        Layout::Diamond => {
            // Rows half-spaced; alternate rows shifted half a cell → a diagonal diaper.
            let rstep = (cell / 2).max(1);
            let mut row = 0;
            let mut y = -cell;
            while y <= h + cell {
                let xoff = if row % 2 == 1 { cell / 2 } else { 0 };
                let mut x = -cell + xoff;
                while x <= w + cell {
                    out.push((x, y));
                    x += cell;
                }
                y += rstep;
                row += 1;
            }
        }
    }
    out
}

/// Render the endpaper by tiling `motif` across a page canvas.
pub fn render_endpaper(motif: &RgbaImage, o: &EndpaperOpts) -> RgbaImage {
    let bg = o.bg.map(Rgba).unwrap_or(Rgba([0, 0, 0, 0]));
    let mut canvas = RgbaImage::from_pixel(o.w.max(1), o.h.max(1), bg);

    // Resize the motif to the target tile width, keeping its aspect.
    let (mw, mh) = motif.dimensions();
    let tw = o.tile.max(1);
    let th = ((tw as f32) * (mh as f32) / (mw.max(1) as f32)).round().max(1.0) as u32;
    let tile = image::imageops::resize(motif, tw, th, image::imageops::FilterType::Lanczos3);

    let cell = o.tile + o.gap;
    for (cx, cy) in nodes(o.w, o.h, cell, o.layout) {
        let x = cx - tw as i64 / 2;
        let y = cy - th as i64 / 2;
        image::imageops::overlay(&mut canvas, &tile, x, y);
    }
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_names() {
        assert!(matches!(Layout::from_name("half_drop"), Layout::HalfDrop));
        assert!(matches!(Layout::from_name("diaper"), Layout::Diamond));
        assert!(matches!(Layout::from_name("whatever"), Layout::Grid));
    }

    #[test]
    fn nodes_cover_and_clip_the_page() {
        // Grid nodes start off-canvas (negative) so motifs clip at the trim, and extend past the far edge.
        let ns = nodes(100, 100, 25, Layout::Grid);
        assert!(ns.iter().any(|&(x, y)| x < 0 || y < 0), "starts off-canvas");
        assert!(ns.iter().any(|&(x, y)| x > 100 || y > 100), "extends past the edge");
        // Half-drop drops odd columns by half a cell → some y not a multiple of the cell.
        let hd = nodes(100, 100, 20, Layout::HalfDrop);
        assert!(hd.iter().any(|&(_, y)| (y - (-20)).rem_euclid(20) != 0), "half-drop offsets a column");
    }

    #[test]
    fn renders_a_patterned_canvas() {
        // A 2×2 opaque-ink motif tiled over a small page leaves ink on the canvas.
        let mut motif = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 0]));
        for p in motif.pixels_mut() {
            *p = Rgba([10, 10, 10, 255]);
        }
        let o = EndpaperOpts { w: 60, h: 60, tile: 10, gap: 6, layout: Layout::Diamond, bg: Some([255, 255, 255, 255]) };
        let out = render_endpaper(&motif, &o);
        assert_eq!(out.dimensions(), (60, 60));
        let inked = out.pixels().filter(|p| p.0[0] < 200).count();
        assert!(inked > 0, "pattern left ink on the page");
    }
}
