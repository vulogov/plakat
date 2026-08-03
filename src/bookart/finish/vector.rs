//! Born-vector SVG for the procedural tier (RFC BOOKART-1 §7.5 — opt-in, secondary). Procedural
//! ornament is authored as parametric **polylines** (`procedural::generate_paths`), so emitting SVG is
//! just serialising them — no tracer, mathematically exact, crisp at any DPI. (Raster→SVG tracing of the
//! diffusion/composite tiers is a documented fast-follow; the primary output is always the PNG, §7.5.)

use crate::bookart::geometry::layout::Rect;
use crate::bookart::procedural::Polyline;

/// Map ornament-local polylines (authored at `gen_w × gen_h`) onto a page-space rect: scale to the
/// rect, apply corner flips, translate to the rect origin. Mirrors `place_on_canvas` for the raster.
pub fn transform_to_rect(paths: &[Polyline], r: &Rect, gen_w: u32, gen_h: u32) -> Vec<Polyline> {
    let sx = r.w as f32 / gen_w.max(1) as f32;
    let sy = r.h as f32 / gen_h.max(1) as f32;
    paths
        .iter()
        .map(|p| {
            p.iter()
                .map(|&(x, y)| {
                    let mut xx = x * sx;
                    let mut yy = y * sy;
                    if r.flip_h {
                        xx = r.w as f32 - xx;
                    }
                    if r.flip_v {
                        yy = r.h as f32 - yy;
                    }
                    (xx + r.x as f32, yy + r.y as f32)
                })
                .collect()
        })
        .collect()
}

/// Serialise polylines to a print-sized SVG: physical `mm` width/height (so a layout tool places it at
/// the right size) over a pixel `viewBox`, stroked in the ink tint.
pub fn polylines_to_svg(paths: &[Polyline], w: u32, h: u32, dpi: u32, stroke_px: f32, tint: [u8; 3]) -> String {
    let mm = |px: u32| px as f32 / dpi as f32 * 25.4;
    let color = format!("#{:02x}{:02x}{:02x}", tint[0], tint[1], tint[2]);
    let mut s = String::new();
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.2}mm\" height=\"{:.2}mm\" viewBox=\"0 0 {w} {h}\">\n",
        mm(w),
        mm(h)
    ));
    s.push_str(&format!("<g fill=\"none\" stroke=\"{color}\" stroke-width=\"{stroke_px:.2}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n"));
    for path in paths {
        if path.len() < 2 {
            continue;
        }
        let mut d = format!("M{:.2} {:.2}", path[0].0, path[0].1);
        for &(x, y) in &path[1..] {
            d.push_str(&format!(" L{x:.2} {y:.2}"));
        }
        s.push_str(&format!("<path d=\"{d}\"/>\n"));
    }
    s.push_str("</g>\n</svg>\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svg_has_dims_and_paths() {
        let paths = vec![vec![(0.0, 0.0), (10.0, 10.0)], vec![(5.0, 0.0), (5.0, 10.0)]];
        let svg = polylines_to_svg(&paths, 300, 300, 300, 1.5, [0, 0, 0]);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("mm\""));
        assert!(svg.contains("viewBox=\"0 0 300 300\""));
        assert_eq!(svg.matches("<path").count(), 2);
    }

    #[test]
    fn transform_flips_and_translates() {
        let paths = vec![vec![(0.0, 0.0)]];
        let r = Rect { x: 100, y: 50, w: 10, h: 10, flip_h: true, flip_v: false };
        let out = transform_to_rect(&paths, &r, 10, 10);
        // (0,0) with flip_h in a 10-wide rect → x=10, then +100 → 110; y stays 0 → +50 → 50.
        assert_eq!(out[0][0], (110.0, 50.0));
    }
}
