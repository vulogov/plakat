//! Procedural ornament generators (RFC BOOKART-1 §5.3 procedural tier / ROADMAP B3). **Self-contained**
//! and **vector-native**: each generator emits parametric **polylines** (ready for born-vector SVG in
//! B6), which `rasterise` strokes into a clean line-art `GrayImage`. Pure, deterministic, no weights, no
//! RNG — and no dependency on the (feature-gated) fractal engine, so it builds under
//! `--no-default-features`. Geometric ornament this way is crisp at any size and exactly symmetric by
//! construction — the guarantee diffusion can't give.

use image::{GrayImage, Luma};
use std::f32::consts::TAU;

/// A parametric path (a polyline in pixel space). Born-vector: B6 serialises these to SVG directly.
pub type Polyline = Vec<(f32, f32)>;

const N: usize = 1440; // samples per closed curve

// --- parametric curve primitives ------------------------------------------------------------------

fn circle(cx: f32, cy: f32, r: f32) -> Polyline {
    (0..=N).map(|i| { let t = i as f32 / N as f32 * TAU; (cx + r * t.cos(), cy + r * t.sin()) }).collect()
}

/// Rhodonea (rose): `r = a·cos(k·θ)`. `k` even → 2k petals, odd → k petals.
fn rose(cx: f32, cy: f32, a: f32, k: f32) -> Polyline {
    (0..=N)
        .map(|i| {
            let t = i as f32 / N as f32 * TAU;
            let rr = a * (k * t).cos();
            (cx + rr * t.cos(), cy + rr * t.sin())
        })
        .collect()
}

/// Hypotrochoid (guilloché): a small circle rolling inside a big one, pen offset `d`.
fn hypotrochoid(cx: f32, cy: f32, scale: f32, big: f32, small: f32, d: f32) -> Polyline {
    let diff = big - small;
    let norm = (diff + d).max(1e-3);
    (0..=N * 3)
        .map(|i| {
            let t = i as f32 / N as f32 * TAU;
            let x = diff * t.cos() + d * (diff / small * t).cos();
            let y = diff * t.sin() - d * (diff / small * t).sin();
            (cx + x / norm * scale, cy + y / norm * scale)
        })
        .collect()
}

fn hline(x0: f32, x1: f32, y: f32) -> Polyline {
    vec![(x0, y), (x1, y)]
}

/// Petals per fold count → the rose `k` that yields exactly `p` petals.
fn rose_k(p: u32) -> f32 {
    if p % 2 == 0 { (p / 2) as f32 } else { p as f32 }
}

fn fold_count(symmetry: &str) -> u32 {
    match symmetry.split(':').next() {
        Some("radial") => symmetry.split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(8),
        Some("bilateral") => 6,
        _ => 8,
    }
    .clamp(3, 24)
}

// --- ornament generators (return their born-vector polylines) --------------------------------------

/// A radial rosette: bounding ring + a rose + an inner counter-rose + a centre dot. Colophon / fleuron /
/// endpaper motif, and the heart of dividers/corners.
fn rosette(w: u32, h: u32, folds: u32) -> Vec<Polyline> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let r = (w.min(h) as f32) * 0.44;
    let k = rose_k(folds);
    vec![
        circle(cx, cy, r),
        circle(cx, cy, r * 0.62),
        rose(cx, cy, r, k),
        rose(cx, cy, r * 0.6, k * 2.0),
        circle(cx, cy, r * 0.08),
    ]
}

/// A horizontal rule: a central rosette flanked by symmetric tapering double-lines with end dots.
fn divider(w: u32, h: u32, folds: u32) -> Vec<Polyline> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let hub = (h as f32 * 0.5).min(w as f32 * 0.12);
    let mut out = rosette_at(cx, cy, hub, folds);
    let gap = hub * 1.2;
    let end = w as f32 * 0.04;
    let off = h as f32 * 0.12;
    for &s in &[1.0f32, -1.0] {
        // mirrored left/right double line
        out.push(hline(cx + s * gap, cx + s * (w as f32 / 2.0 - end), cy - off));
        out.push(hline(cx + s * gap, cx + s * (w as f32 / 2.0 - end), cy + off));
        out.push(circle(cx + s * (w as f32 / 2.0 - end), cy, off * 0.9)); // end cap
    }
    out
}

/// A frame: two nested rectangles with a bead-and-reel run between them and corner rosettes.
fn border(w: u32, h: u32, folds: u32) -> Vec<Polyline> {
    let m = (w.min(h) as f32) * 0.04;
    let m2 = m * 2.2;
    let mut out = vec![rect(m, m, w as f32 - m, h as f32 - m), rect(m2, m2, w as f32 - m2, h as f32 - m2)];
    // beads along the mid-line of the two rectangles
    let bead_r = (m2 - m) * 0.28;
    let mid = (m + m2) / 2.0;
    let step = bead_r * 3.2;
    let mut x = m2;
    while x < w as f32 - m2 {
        out.push(circle(x, mid, bead_r));
        out.push(circle(x, h as f32 - mid, bead_r));
        x += step;
    }
    let mut y = m2;
    while y < h as f32 - m2 {
        out.push(circle(mid, y, bead_r));
        out.push(circle(w as f32 - mid, y, bead_r));
        y += step;
    }
    let cr = (w.min(h) as f32) * 0.09;
    for &(px, py) in &[(m2 + cr, m2 + cr), (w as f32 - m2 - cr, m2 + cr), (m2 + cr, h as f32 - m2 - cr), (w as f32 - m2 - cr, h as f32 - m2 - cr)] {
        out.extend(rosette_at(px, py, cr, folds));
    }
    out
}

/// An L-corner flourish: two edge segments + a quarter-guilloché scroll at the inner angle.
fn corner(w: u32, h: u32) -> Vec<Polyline> {
    let s = w.min(h) as f32;
    let m = s * 0.12;
    let mut out = vec![hline(m, s * 0.9, m), vec![(m, m), (m, s * 0.9)]];
    // a scroll: a guilloché rosette tucked into the inner angle
    out.push(hypotrochoid(s * 0.42, s * 0.42, s * 0.32, 5.0, 3.0, 5.0));
    out
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Polyline {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
}

fn rosette_at(cx: f32, cy: f32, r: f32, folds: u32) -> Vec<Polyline> {
    let k = rose_k(folds);
    vec![circle(cx, cy, r), rose(cx, cy, r, k), rose(cx, cy, r * 0.55, k * 2.0), circle(cx, cy, r * 0.12)]
}

/// A procedural **frame** for the composite tier (RFC §5.3): a nested-rectangle border with corner
/// rosettes, plus the **inner window** rect `(x, y, w, h)` where a diffusion picture is inlaid.
pub fn frame(symmetry: &str, w: u32, h: u32) -> (Vec<Polyline>, (u32, u32, u32, u32)) {
    let folds = fold_count(symmetry);
    let m = (w.min(h) as f32) * 0.045;
    let m2 = m * 2.0;
    let cr = (w.min(h) as f32) * 0.08;
    let mut out = vec![rect(m, m, w as f32 - m, h as f32 - m), rect(m2, m2, w as f32 - m2, h as f32 - m2)];
    for &(px, py) in &[(m2 + cr, m2 + cr), (w as f32 - m2 - cr, m2 + cr), (m2 + cr, h as f32 - m2 - cr), (w as f32 - m2 - cr, h as f32 - m2 - cr)] {
        out.extend(rosette_at(px, py, cr, folds));
    }
    let inset = m2 + cr * 0.4;
    let win = (inset as u32, inset as u32, (w as f32 - 2.0 * inset).max(1.0) as u32, (h as f32 - 2.0 * inset).max(1.0) as u32);
    (out, win)
}

/// The born-vector paths for an ornament type at a target pixel size.
pub fn generate_paths(kind: &str, symmetry: &str, w: u32, h: u32) -> Vec<Polyline> {
    let folds = fold_count(symmetry);
    match kind {
        "divider" | "rule" => divider(w, h, folds),
        "border" | "endpaper" => border(w, h, folds),
        "corner" => corner(w, h),
        "rosette" | "colophon" | "fleuron" | "dinkus" | "marginalia" => rosette(w, h, folds),
        _ => rosette(w, h, folds), // headpiece/tailpiece geometric fallback
    }
}

// --- rasterisation ---------------------------------------------------------------------------------

fn plot_disc(img: &mut GrayImage, cx: f32, cy: f32, r: f32) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let (x0, y0) = ((cx - r - 1.0).floor() as i32, (cy - r - 1.0).floor() as i32);
    let (x1, y1) = ((cx + r + 1.0).ceil() as i32, (cy + r + 1.0).ceil() as i32);
    for y in y0.max(0)..=y1.min(h - 1) {
        for x in x0.max(0)..=x1.min(w - 1) {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            let cov = (r + 0.5 - d).clamp(0.0, 1.0); // soft edge
            if cov > 0.0 {
                let px = img.get_pixel_mut(x as u32, y as u32);
                let v = (255.0 * (1.0 - cov)).round() as u8;
                px.0[0] = px.0[0].min(v);
            }
        }
    }
}

/// Stroke polylines into a white `GrayImage` with black ink of the given `width`.
pub fn rasterise(paths: &[Polyline], w: u32, h: u32, width: f32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([255]));
    let r = (width / 2.0).max(0.5);
    for path in paths {
        for seg in path.windows(2) {
            let (a, b) = (seg[0], seg[1]);
            let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            let steps = (len / 0.6).ceil().max(1.0) as u32;
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                plot_disc(&mut img, a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, r);
            }
        }
    }
    img
}

/// Generate a procedural ornament as a clean line-art `GrayImage` (ink on white).
pub fn generate(kind: &str, symmetry: &str, w: u32, h: u32) -> GrayImage {
    let paths = generate_paths(kind, symmetry, w, h);
    let width = (w.min(h) as f32 * 0.004).max(1.5);
    rasterise(&paths, w, h, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ink_frac(g: &GrayImage) -> f32 {
        g.pixels().filter(|p| p.0[0] < 128).count() as f32 / (g.width() * g.height()) as f32
    }

    #[test]
    fn generators_produce_ink_and_are_deterministic() {
        for kind in ["rosette", "divider", "border", "corner", "colophon", "fleuron"] {
            let a = generate(kind, "radial:8", 256, 256);
            let b = generate(kind, "radial:8", 256, 256);
            assert_eq!(a.as_raw(), b.as_raw(), "{kind} not deterministic");
            assert!(ink_frac(&a) > 0.005, "{kind} produced ~no ink ({})", ink_frac(&a));
            assert!(ink_frac(&a) < 0.6, "{kind} is a slab ({})", ink_frac(&a));
        }
    }

    #[test]
    fn rosette_is_radially_symmetric() {
        // an N-fold rosette rotated by 2π/N ≈ itself: compare ink counts in rotated quadrants.
        let g = generate("rosette", "radial:4", 256, 256);
        let (w, h) = (g.width(), g.height());
        let quad = |qx: u32, qy: u32| {
            let mut c = 0u32;
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    if g.get_pixel(qx * w / 2 + x, qy * h / 2 + y).0[0] < 128 {
                        c += 1;
                    }
                }
            }
            c
        };
        let (a, b, c, d) = (quad(0, 0), quad(1, 0), quad(1, 1), quad(0, 1));
        let max = a.max(b).max(c).max(d) as f32;
        let min = a.min(b).min(c).min(d) as f32;
        assert!(min / max.max(1.0) > 0.7, "quadrant ink imbalance: {a} {b} {c} {d}");
    }

    #[test]
    fn born_vector_paths_exist() {
        let paths = generate_paths("rosette", "radial:6", 200, 200);
        assert!(paths.len() >= 3, "rosette should emit several curves");
        assert!(paths.iter().all(|p| p.len() >= 2));
    }
}
