//! The panel-layout engine (RFC COMIC-1 §1) — deterministic, weight-free. Resolves a [`ComicSpec`] to the
//! page pixel size + an ordered list of panel rectangles (rows of relative-width cells + gutter + border),
//! in reading order (`ltr`/`rtl`). No GPU.

use super::spec::ComicSpec;
use serde::Serialize;

/// A resolved panel rectangle on the page (px), with its reading index and the spec panel it holds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PanelRect {
    pub index: usize, // reading order
    pub panel: usize, // index into spec.panels
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// The resolved page plan.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub w: u32,
    pub h: u32,
    pub dpi: u32,
    pub gutter: u32,
    pub border: u32,
    pub bg: (u8, u8, u8),
    pub reading: String,
    pub panels: Vec<PanelRect>,
}

/// Named page size → (width_in, height_in). `custom` uses the spec's `w_in`/`h_in`.
fn size_inches(name: &str) -> Option<(f32, f32)> {
    Some(match name.to_ascii_lowercase().as_str() {
        "us-letter" | "letter" => (8.5, 11.0),
        "a4" => (8.27, 11.69),
        "a5" => (5.83, 8.27),
        "tabloid" | "ledger" => (11.0, 17.0),
        "square" => (10.0, 10.0),
        _ => return None,
    })
}

fn parse_bg(s: &str) -> (u8, u8, u8) {
    match s.to_ascii_lowercase().as_str() {
        "white" => (255, 255, 255),
        "black" => (0, 0, 0),
        "cream" => (247, 243, 233),
        other => {
            let c: Vec<u8> = other.split(',').filter_map(|p| p.trim().parse::<u8>().ok()).collect();
            if c.len() == 3 {
                (c[0], c[1], c[2])
            } else {
                (255, 255, 255)
            }
        }
    }
}

/// Resolve the spec → a [`Plan`]. Layout `rows` (relative-width cells) drive the grid; absent → the panels
/// are auto-gridded into a near-square.
pub fn resolve(spec: &ComicSpec) -> Plan {
    let page = spec.page.clone().unwrap_or_default();
    let dpi = page.dpi.unwrap_or(300).clamp(72, 1200);
    let (win, hin) = page
        .size
        .as_deref()
        .and_then(size_inches)
        .or_else(|| page.w_in.zip(page.h_in))
        .unwrap_or((8.5, 11.0));
    let (w, h) = ((win * dpi as f32) as u32, (hin * dpi as f32) as u32);
    let gutter = page.gutter.unwrap_or(dpi / 12).min(w / 4); // ~24px @ 300dpi
    let border = page.border.unwrap_or(dpi / 50); // ~6px @ 300dpi
    let bg = parse_bg(page.bg.as_deref().unwrap_or("white"));
    let reading = spec.reading.clone().unwrap_or_else(|| "ltr".into());
    let rtl = reading.eq_ignore_ascii_case("rtl");

    // the grid: rows of cell weights.
    let n = spec.panels.len().max(1);
    let rows: Vec<Vec<f32>> = spec
        .layout
        .as_ref()
        .and_then(|l| l.rows.clone())
        .unwrap_or_else(|| auto_grid(n));
    let row_heights = spec
        .layout
        .as_ref()
        .and_then(|l| l.row_heights.clone())
        .filter(|v| v.len() == rows.len())
        .unwrap_or_else(|| vec![1.0; rows.len()]);

    // inner content area (a page margin = gutter so panels don't kiss the edge).
    let (mx, my) = (gutter, gutter);
    let (iw, ih) = (w.saturating_sub(2 * mx), h.saturating_sub(2 * my));
    let rows_gut = gutter * (rows.len().saturating_sub(1)) as u32;
    let avail_h = ih.saturating_sub(rows_gut);
    let hsum: f32 = row_heights.iter().sum::<f32>().max(1e-3);

    let mut panels = Vec::new();
    let mut reading_idx = 0usize;
    let mut panel_idx = 0usize;
    let mut y = my;
    for (r, cells) in rows.iter().enumerate() {
        let row_h = (avail_h as f32 * row_heights[r] / hsum).round() as u32;
        let cells_gut = gutter * (cells.len().saturating_sub(1)) as u32;
        let avail_w = iw.saturating_sub(cells_gut);
        let wsum: f32 = cells.iter().sum::<f32>().max(1e-3);
        // cell x positions, L→R; reverse the assignment order for rtl.
        let mut x = mx;
        let mut row_rects = Vec::new();
        for &cw in cells {
            let cell_w = (avail_w as f32 * cw / wsum).round() as u32;
            row_rects.push((x, cell_w));
            x += cell_w + gutter;
        }
        let order: Vec<usize> = if rtl { (0..row_rects.len()).rev().collect() } else { (0..row_rects.len()).collect() };
        for &ci in &order {
            let (cx, cell_w) = row_rects[ci];
            panels.push(PanelRect { index: reading_idx, panel: panel_idx.min(spec.panels.len().saturating_sub(1)), x: cx, y, w: cell_w, h: row_h });
            reading_idx += 1;
            panel_idx += 1;
        }
        y += row_h + gutter;
    }
    Plan { w, h, dpi, gutter, border, bg, reading, panels }
}

/// Auto-grid `n` panels into rows of a near-square grid.
fn auto_grid(n: usize) -> Vec<Vec<f32>> {
    let cols = (n as f32).sqrt().ceil() as usize;
    let mut rows = Vec::new();
    let mut left = n;
    while left > 0 {
        let c = left.min(cols);
        rows.push(vec![1.0; c]);
        left -= c;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_letter_grid_in_reading_order_no_overlap() {
        let spec = ComicSpec::from_hjson(
            r#"{ page: { size: "us-letter", dpi: 100, gutter: 10, border: 2 },
                 layout: { rows: [[1,1],[1],[1,1,1]] },
                 panels: [ {}, {}, {}, {}, {}, {} ] }"#,
        )
        .unwrap();
        let p = resolve(&spec);
        assert_eq!((p.w, p.h), (850, 1100)); // 8.5×11 @ 100dpi
        assert_eq!(p.panels.len(), 6);
        // reading indices are 0..6 in order.
        assert!(p.panels.iter().enumerate().all(|(i, r)| r.index == i));
        // no two panels overlap.
        for i in 0..p.panels.len() {
            for j in i + 1..p.panels.len() {
                let (a, b) = (&p.panels[i], &p.panels[j]);
                let overlap = a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
                assert!(!overlap, "panels {i} and {j} overlap");
            }
        }
        // all panels inside the page.
        assert!(p.panels.iter().all(|r| r.x + r.w <= p.w && r.y + r.h <= p.h));
    }

    #[test]
    fn rtl_reverses_within_rows() {
        let spec = ComicSpec::from_hjson(r#"{ reading: "rtl", layout: { rows: [[1,1,1]] }, panels: [{},{},{}] }"#).unwrap();
        let p = resolve(&spec);
        // reading index 0 is the RIGHTMOST cell.
        let first = p.panels.iter().find(|r| r.index == 0).unwrap();
        let last = p.panels.iter().find(|r| r.index == 2).unwrap();
        assert!(first.x > last.x, "rtl: reading-0 is rightmost");
    }

    #[test]
    fn auto_grid_when_no_layout() {
        let spec = ComicSpec::from_hjson(r#"{ panels: [{},{},{},{},{}] }"#).unwrap();
        let p = resolve(&spec);
        assert_eq!(p.panels.len(), 5); // 5 panels auto-gridded (3 cols → rows of 3,2)
    }
}
