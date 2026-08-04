//! Procedural ornament generators (RFC BOOKART-1 §5.3 procedural tier / ROADMAP B3). **Self-contained**
//! and **vector-native**: each generator emits parametric **polylines** (ready for born-vector SVG in
//! B6), which `rasterise` strokes into a clean line-art `GrayImage`. Pure, deterministic, no weights, no
//! RNG — and no dependency on the (feature-gated) fractal engine, so it builds under
//! `--no-default-features`. Geometric ornament this way is crisp at any size and exactly symmetric by
//! construction — the guarantee diffusion can't give.

use image::{GrayImage, Luma};
use std::f32::consts::{PI, TAU};

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

/// A sine-wave polyline across `[x0,x1]` at baseline `cy` — a running ornament stroke.
fn wave(x0: f32, x1: f32, cy: f32, amp: f32, cycles: f32, phase: f32) -> Polyline {
    let n = 500;
    (0..=n).map(|i| { let t = i as f32 / n as f32; (x0 + (x1 - x0) * t, cy + amp * (t * cycles * TAU + phase).sin()) }).collect()
}

/// An Archimedean spiral scroll (a flourish); `dir` = ±1 for handedness.
fn scroll(cx: f32, cy: f32, r0: f32, r1: f32, turns: f32, start: f32, dir: f32) -> Polyline {
    let n = 240;
    (0..=n).map(|i| { let t = i as f32 / n as f32; let a = start + dir * t * turns * TAU; let r = r0 + (r1 - r0) * t; (cx + r * a.cos(), cy + r * a.sin()) }).collect()
}

// --- C1 net-new band motifs (Greek-key / L-system vine / knotwork interlace) ----------------------

/// Map a set of polylines (in their own bbox) into a target rect. `flip_y` grows the source's +y
/// downward (image space). Used to place a unit-space motif (e.g. an L-system sprig) into a band cell.
fn map_bbox(paths: &[Polyline], tx0: f32, ty0: f32, tx1: f32, ty1: f32, flip_y: bool) -> Vec<Polyline> {
    let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in paths {
        for &(x, y) in p {
            minx = minx.min(x); miny = miny.min(y); maxx = maxx.max(x); maxy = maxy.max(y);
        }
    }
    let (sx, sy) = ((maxx - minx).max(1e-3), (maxy - miny).max(1e-3));
    paths
        .iter()
        .map(|p| {
            p.iter()
                .map(|&(x, y)| {
                    let u = (x - minx) / sx;
                    let v = (y - miny) / sy;
                    let vy = if flip_y { 1.0 - v } else { v };
                    (tx0 + (tx1 - tx0) * u, ty0 + (ty1 - ty0) * vy)
                })
                .collect()
        })
        .collect()
}

/// Greek-key (meander) fret as one continuous polyline across `[x0,x1]` centred on `cy`. `variant`
/// sets the repeat count. Rectilinear interlocking hooks — the classic key border.
fn greek_key(x0: f32, x1: f32, cy: f32, height: f32, variant: u32) -> Polyline {
    let units = 5 + (variant % 4);
    let u = (x1 - x0) / units as f32;
    let (top, bot) = (cy - height * 0.5, cy + height * 0.5);
    let inset = u * 0.22;
    let mut p = vec![(x0, bot)];
    for i in 0..units {
        let ox = x0 + i as f32 * u;
        // one meander cell: up the left, across, down inside, back — an interlocking hook.
        for &pt in &[
            (ox, bot), (ox, top), (ox + u * 0.62, top), (ox + u * 0.62, bot - inset),
            (ox + inset, bot - inset), (ox + inset, top + inset), (ox + u * 0.62 - inset, top + inset),
            (ox + u * 0.62 - inset, bot - inset * 2.0), (ox + u, bot - inset * 2.0), (ox + u, bot),
        ] {
            p.push(pt);
        }
    }
    p
}

/// A tiny L-system sprig (RFC C1 "foliate scroll via a small L-system"): axiom `X`, rules
/// `X → F[+X][-X]FX` and `F → FF`, turtle-interpreted into branch polylines in local space (growing
/// up). `variant` perturbs the branch angle so sprigs in a set differ.
fn lsystem_sprig(depth: u32, variant: u32) -> Vec<Polyline> {
    let mut s = String::from("X");
    for _ in 0..depth {
        let mut n = String::with_capacity(s.len() * 3);
        for c in s.chars() {
            match c {
                'X' => n.push_str("F[+X][-X]FX"),
                'F' => n.push_str("FF"),
                o => n.push(o),
            }
        }
        s = n;
    }
    let ang = (22.0 + (variant % 3) as f32 * 6.0).to_radians();
    let (mut x, mut y, mut dir) = (0.0f32, 0.0f32, PI / 2.0); // pointing up
    let mut stack: Vec<(f32, f32, f32)> = Vec::new();
    let mut cur: Polyline = vec![(x, y)];
    let mut out: Vec<Polyline> = Vec::new();
    for c in s.chars() {
        match c {
            'F' => { x += dir.cos(); y += dir.sin(); cur.push((x, y)); }
            '+' => dir += ang,
            '-' => dir -= ang,
            '[' => { stack.push((x, y, dir)); if cur.len() > 1 { out.push(std::mem::take(&mut cur)); } cur = vec![(x, y)]; }
            ']' => {
                if cur.len() > 1 { out.push(std::mem::take(&mut cur)); }
                let (px, py, pd) = stack.pop().unwrap_or((x, y, dir));
                x = px; y = py; dir = pd; cur = vec![(x, y)];
            }
            _ => {}
        }
    }
    if cur.len() > 1 { out.push(cur); }
    out
}

/// A foliate-scroll band: mirrored L-system vine sprigs tiled across `[x0,x1]`, alternating which way
/// they lean — a running vine border.
fn foliate_band(x0: f32, x1: f32, cy: f32, height: f32, variant: u32) -> Vec<Polyline> {
    let sprig = lsystem_sprig(4, variant);
    let reps = 3 + (variant % 3);
    let cw = (x1 - x0) / reps as f32;
    let mut out = vec![hline(x0, x1, cy)]; // the stem line
    for i in 0..reps {
        let cx0 = x0 + i as f32 * cw;
        // alternate up / down so the vine reads as a running scroll.
        let up = i % 2 == 0;
        let (ty0, ty1) = if up { (cy, cy - height * 0.55) } else { (cy, cy + height * 0.55) };
        out.extend(map_bbox(&sprig, cx0 + cw * 0.1, ty0, cx0 + cw * 0.9, ty1, up));
    }
    out
}

/// A knotwork / interlace band (RFC C1, the net-new from G0.4): `strands` sine strands with periodic
/// gaps punched per-strand at crossings, faking the over/under weave of a plait.
fn interlace_band(x0: f32, x1: f32, cy: f32, amp: f32, cycles: f32, strands: u32) -> Vec<Polyline> {
    let mut out: Vec<Polyline> = Vec::new();
    let n = 900;
    for s in 0..strands {
        let phase = s as f32 / strands as f32 * TAU;
        let mut cur: Polyline = Vec::new();
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let ph = t * cycles * TAU + phase;
            let over = ((ph / PI).floor() as i32 + s as i32).rem_euclid(2) == 0;
            let near_cross = ph.sin().abs() < 0.30; // near a baseline crossing
            let x = x0 + (x1 - x0) * t;
            let y = cy + amp * ph.sin();
            if !over && near_cross {
                if cur.len() > 1 { out.push(std::mem::take(&mut cur)); }
                cur.clear();
                continue;
            }
            cur.push((x, y));
        }
        if cur.len() > 1 { out.push(cur); }
    }
    out
}

/// Fill a band cell `[x0,x1]` around `cy` with one of the C1 motifs, chosen by `variant` — so a set
/// (a manuscript's per-chapter bands) cycles guilloché-braid → Greek-key → foliate vine → interlace.
fn band_motif(x0: f32, x1: f32, cy: f32, height: f32, variant: u32) -> Vec<Polyline> {
    match variant % 4 {
        1 => vec![greek_key(x0, x1, cy, height * 0.8, variant)],
        2 => foliate_band(x0, x1, cy, height, variant),
        3 => interlace_band(x0, x1, cy, height * 0.5, 3.0 + (variant % 3) as f32, 3),
        _ => {
            // 0: the guilloché braid — two counter-phase waves.
            let cyc = 3.0 + (variant % 4) as f32;
            let amp = height * 0.30;
            vec![wave(x0, x1, cy, amp, cyc, 0.0), wave(x0, x1, cy, amp, cyc, PI)]
        }
    }
}

/// A radial rosette: ring + a rose + an inner counter-rose + a guilloché ring + centre. `variant`
/// perturbs the guilloché + counter-rose so a *set* of rosettes reads as kin, not clones.
fn rosette(w: u32, h: u32, folds: u32, variant: u32) -> Vec<Polyline> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let r = (w.min(h) as f32) * 0.44;
    let k = rose_k(folds);
    let g = 3.0 + (variant % 3) as f32; // guilloché lobes vary
    vec![
        circle(cx, cy, r),
        circle(cx, cy, r * 0.62),
        rose(cx, cy, r, k),
        rose(cx, cy, r * (0.54 + 0.08 * (variant % 2) as f32), k * 2.0),
        hypotrochoid(cx, cy, r * 0.30, folds as f32 + g, g, g * 0.7), // a small central guilloché, not a fill
        circle(cx, cy, r * 0.08),
    ]
}

/// A horizontal rule: a central medallion flanked by symmetric guilloché-wave lines with fleuron ends —
/// airy line ornament, not a black bar.
fn divider(w: u32, h: u32, folds: u32, variant: u32) -> Vec<Polyline> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let hub = (h as f32 * 0.5).min(w as f32 * 0.10);
    let mut out = rosette_at(cx, cy, hub, folds);
    let (gap, end, off) = (hub * 1.25, w as f32 * 0.03, h as f32 * 0.10);
    for &s in &[1.0f32, -1.0] {
        let (x0, x1) = (cx + s * gap, cx + s * (w as f32 / 2.0 - end));
        out.push(hline(x0, x1, cy));
        // C1: variant-selected running motif (mirrored across the medallion).
        out.extend(band_motif(x0.min(x1), x0.max(x1), cy, off * 2.0, variant));
        out.extend(rosette_at(cx + s * (w as f32 / 2.0 - end), cy, off * 1.1, (folds / 2).max(3))); // end fleuron
    }
    out
}

/// A frame: nested rectangles + a bead-and-reel run + corner rosettes. `variant` shifts bead density.
fn border(w: u32, h: u32, folds: u32, variant: u32) -> Vec<Polyline> {
    let m = (w.min(h) as f32) * 0.04;
    let m2 = m * 2.2;
    let mut out = vec![rect(m, m, w as f32 - m, h as f32 - m), rect(m2, m2, w as f32 - m2, h as f32 - m2)];
    let bead_r = (m2 - m) * 0.28;
    let mid = (m + m2) / 2.0;
    let step = bead_r * (3.0 + 0.4 * (variant % 3) as f32);
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

/// A bold L-corner: double edge rules + a corner rosette + a spiral scroll flourish.
fn corner(w: u32, h: u32, folds: u32, variant: u32) -> Vec<Polyline> {
    let s = w.min(h) as f32;
    let (m, d) = (s * 0.08, s * 0.035);
    let mut out = vec![
        hline(m, s * 0.94, m),
        hline(m, s * 0.94, m + d),
        vec![(m, m), (m, s * 0.94)],
        vec![(m + d, m), (m + d, s * 0.94)],
    ];
    out.extend(rosette_at(s * 0.30, s * 0.30, s * 0.18, folds));
    out.push(scroll(s * 0.56, s * 0.56, s * 0.04, s * 0.30, 1.2 + 0.2 * (variant % 3) as f32, 0.0, 1.0));
    out
}

/// A headpiece band (застАвка): top+bottom rules, a central medallion, interweaving guilloché waves
/// flanking it, and fleuron ends — an airy ornamental band, not a black block. `variant` = wave freq.
fn headpiece_band(w: u32, h: u32, folds: u32, variant: u32) -> Vec<Polyline> {
    let (wf, hf) = (w as f32, h as f32);
    let (cx, cy) = (wf / 2.0, hf / 2.0);
    let (m, top, bot) = (wf * 0.012, hf * 0.16, hf * 0.84);
    let med_r = hf * 0.40;
    let mut out = vec![hline(m, wf - m, top), hline(m, wf - m, bot)];
    out.extend(rosette_at(cx, cy, med_r, folds));
    // C1: the flanking fields carry a variant-selected motif (braid / Greek-key / vine / interlace).
    let bh = bot - top;
    let gap = med_r * 1.2;
    for &(x0, x1) in &[(m + wf * 0.02, cx - gap), (cx + gap, wf - m - wf * 0.02)] {
        out.extend(band_motif(x0, x1, cy, bh, variant));
    }
    for &ex in &[m + wf * 0.015, wf - m - wf * 0.015] {
        out.extend(rosette_at(ex, cy, (bot - top) * 0.16, (folds / 2).max(3)));
    }
    out
}

/// A tailpiece / cul-de-lampe: a central medallion above symmetric scrolls tapering to a point.
fn tailpiece_taper(w: u32, h: u32, folds: u32, variant: u32) -> Vec<Polyline> {
    let (wf, hf) = (w as f32, h as f32);
    let cx = wf / 2.0;
    let mut out = vec![hline(wf * 0.12, wf * 0.88, hf * 0.07)];
    out.extend(rosette_at(cx, hf * 0.28, wf.min(hf) * 0.20, folds));
    let rows = 3 + (variant % 3) as i32;
    // C1: per-fold variation — the flanking motif alternates scroll ↔ L-system leaf down the taper, and
    // the scroll turn count scales with the fold count so a set of tailpieces differs by tradition.
    let turns = 0.6 + (folds % 3) as f32 * 0.3;
    let leaf = lsystem_sprig(3, variant);
    for i in 0..rows {
        let t = (i as f32 + 1.0) / (rows as f32 + 1.0);
        let y = hf * 0.42 + hf * 0.5 * t;
        let spread = wf * 0.34 * (1.0 - t);
        let sz = hf * 0.06 * (1.0 - 0.5 * t);
        let use_leaf = (i + variant as i32) % 2 == 1;
        for &sgn in &[1.0f32, -1.0] {
            let (fx, fy) = (cx + sgn * spread, y);
            if use_leaf {
                out.extend(map_bbox(&leaf, fx - sz * sgn, y + sz, fx + sz * sgn, y - sz, true));
            } else {
                out.push(scroll(fx, fy, sz * 0.15, sz, turns, PI, sgn));
            }
        }
    }
    out.push(circle(cx, hf * 0.95, hf * 0.018));
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

/// The born-vector paths for an ornament type at a target pixel size. `variant` diversifies a set
/// (e.g. per-chapter ornaments) so they read as kin rather than clones.
pub fn generate_paths(kind: &str, symmetry: &str, w: u32, h: u32, variant: u32) -> Vec<Polyline> {
    let folds = fold_count(symmetry);
    match kind {
        "headpiece" => headpiece_band(w, h, folds, variant),
        "tailpiece" => tailpiece_taper(w, h, folds, variant),
        "divider" | "rule" => divider(w, h, folds, variant),
        "border" | "endpaper" => border(w, h, folds, variant),
        "corner" => corner(w, h, folds, variant),
        "rosette" | "colophon" | "fleuron" | "dinkus" | "marginalia" => rosette(w, h, folds, variant),
        _ => rosette(w, h, folds, variant),
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
pub fn generate(kind: &str, symmetry: &str, w: u32, h: u32, variant: u32) -> GrayImage {
    let paths = generate_paths(kind, symmetry, w, h, variant);
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
        for kind in ["rosette", "divider", "border", "corner", "colophon", "fleuron", "headpiece", "tailpiece"] {
            let a = generate(kind, "radial:8", 384, 384, 0);
            let b = generate(kind, "radial:8", 384, 384, 0);
            assert_eq!(a.as_raw(), b.as_raw(), "{kind} not deterministic");
            assert!(ink_frac(&a) > 0.004, "{kind} produced ~no ink ({})", ink_frac(&a));
            assert!(ink_frac(&a) < 0.6, "{kind} is a slab ({})", ink_frac(&a));
        }
    }

    #[test]
    fn variant_changes_the_ornament() {
        let a = generate("headpiece", "bilateral", 512, 128, 0);
        let b = generate("headpiece", "bilateral", 512, 128, 2);
        assert_ne!(a.as_raw(), b.as_raw(), "variant should diversify");
    }

    #[test]
    fn rosette_is_radially_symmetric() {
        // an N-fold rosette rotated by 2π/N ≈ itself: compare ink counts in rotated quadrants.
        let g = generate("rosette", "radial:4", 256, 256, 0);
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
    fn c1_band_motifs_are_distinct_and_clean() {
        // The four variant-selected headpiece motifs (braid / Greek-key / vine / interlace) must each
        // produce line-art ink (not a slab) and differ from one another.
        let bands: Vec<_> = (0..4).map(|v| generate("headpiece", "bilateral", 640, 160, v)).collect();
        for (v, g) in bands.iter().enumerate() {
            let f = ink_frac(g);
            assert!(f > 0.002 && f < 0.5, "headpiece motif {v} ink out of range ({f})");
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(bands[i].as_raw(), bands[j].as_raw(), "motifs {i} and {j} should differ");
            }
        }
    }

    #[test]
    fn c1_net_new_generators_emit_paths() {
        assert!(greek_key(0.0, 100.0, 20.0, 20.0, 0).len() > 10, "meander has vertices");
        assert!(!lsystem_sprig(4, 0).is_empty(), "L-system sprig has branches");
        assert!(foliate_band(0.0, 300.0, 20.0, 30.0, 0).len() > 3, "foliate band tiles sprigs");
        assert!(interlace_band(0.0, 300.0, 20.0, 10.0, 3.0, 3).len() >= 3, "interlace has broken strands");
    }

    #[test]
    fn born_vector_paths_exist() {
        let paths = generate_paths("rosette", "radial:6", 200, 200, 0);
        assert!(paths.len() >= 3, "rosette should emit several curves");
        assert!(paths.iter().all(|p| p.len() >= 2));
    }
}
