//! The bristle-brush stroke and its rasteriser (RFC PAINT-1 §8.5). A stroke is a loaded brush dragged along a
//! path. The brush is a bundle of **bristles**, each carrying its own pigment **load**; as the brush moves it
//! **deposits** load onto the canvas and **picks up** wet canvas pigment back into itself. Two consequences,
//! both emergent rather than coded as rules:
//!
//! * The dirty brush harmonises. Because pickup mixes canvas pigment into the load, consecutive strokes in a
//!   region drift toward a shared colour — colour unity with no explicit harmony rule.
//! * Stroke order is globally significant. Stroke *k* depends on the wet pigment strokes *1..k* left behind,
//!   so overlapping strokes within a pass cannot be reordered or parallelised (§8.8).
//!
//! No solver, no iteration: cost is O(path length × bristles), fully deterministic.

use crate::paint::canvas::Canvas;

/// Physics constants for a brush — later derived from the medium profile; passed explicitly for now.
#[derive(Clone, Copy, Debug)]
pub struct BrushConfig {
    /// Deposit rate `k_d`: fraction of load laid down per step at full contact on bare tooth.
    pub k_deposit: f32,
    /// Pickup rate `k_p`: how strongly the brush lifts wet canvas pigment back into its load.
    pub k_pickup: f32,
    /// Impasto viscosity: height added per unit of deposited concentration.
    pub viscosity: f32,
    /// Bristles across the brush width.
    pub bristles: usize,
    /// The full-load magnitude the pickup term saturates against.
    pub load_max: f32,
    /// Bristle STREAK (0..1): how uneven the bristle loads are — 0 = a smooth round mark, 1 = a raked, drybrush
    /// bristle mark. The brush's texture character (a soft round vs a stiff flat/fan).
    pub streak: f32,
    /// Cross-section ROUNDNESS (0..1): 1 = a round brush (soft feathered edges), 0 = a flat brush (harder,
    /// squarer edge across the width).
    pub round: f32,
}

impl Default for BrushConfig {
    fn default() -> Self {
        // Oil-direct-ish: deposit a solid fraction of the load per step so a charged brush lays OPAQUE paint
        // (hides the ground → deep darks, bright lights, saturated colour, no wash), drying toward its end (the
        // loaded-gradient). Pickup is MODEST — too much drags wet paint and smears every stroke into its
        // neighbour (the "smeared slop"); alla-prima keeps marks distinct, sitting on top, only lightly harmonised.
        Self { k_deposit: 0.34, k_pickup: 0.25, viscosity: 1.0, bristles: 7, load_max: 6.0, streak: 0.6, round: 0.7 }
    }
}

/// Saturation at which the tooth is full and deposit stops.
const SAT_FULL: f32 = 6.0;

/// How hard a full tooth throttles further deposit. At 1.0 (the old behaviour) a saturated pixel refuses ALL new
/// paint, so once the block-in fills the tooth no later pass can build a darker dark or a brighter light — the
/// value range freezes into a flat, washed mid-tone. Below 1.0 a full pixel still accepts a fraction of each
/// stroke, so opaque media keep building value by shifting the pigment RATIO (KM colour is by ratio, not amount).
const SAT_THROTTLE: f32 = 0.72;

/// Deterministic per-lane hash in `[0,1]` (a hashed LCG) — for reproducible bristle-load variation.
fn lane_hash(seed: u64, b: u64) -> f32 {
    let mut z = seed.wrapping_add(b.wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_add(0x1234_5678);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z as f32 / u64::MAX as f32
}

/// A loaded stroke: a path in pixel space, a width taper, and one pigment load (a concentration vector over
/// the canvas palette). `wetness` is how wet the paint goes on (it both deposits and wets the canvas).
#[derive(Clone, Debug)]
pub struct Stroke {
    pub path: Vec<[f32; 2]>,
    pub width0: f32,
    pub width1: f32,
    pub load: Vec<f32>,
    pub pressure: f32,
    pub wetness: f32,
}

/// Resample a polyline to ~1px arc-length spacing so the brush stamps continuously.
fn densify(path: &[[f32; 2]]) -> Vec<[f32; 2]> {
    if path.len() < 2 {
        return path.to_vec();
    }
    let mut out = vec![path[0]];
    for seg in path.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        let steps = len.ceil().max(1.0) as usize;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            out.push([a[0] + dx * t, a[1] + dy * t]);
        }
    }
    out
}

impl Stroke {
    /// The load magnitude (sum of concentrations).
    fn load_norm(load: &[f32]) -> f32 {
        load.iter().copied().map(|c| c.max(0.0)).sum()
    }

    /// Rasterise the stroke onto the canvas, evolving each bristle's load by deposit + pickup as it travels.
    pub fn rasterize(&self, canvas: &mut Canvas, brush: &BrushConfig) {
        let pts = densify(&self.path);
        if pts.is_empty() || self.load.is_empty() {
            return;
        }
        // Lanes track the stroke WIDTH — ~one lane per pixel — so the footprint is a SOLID swath, not a comb of
        // spaced tine-lines (a fixed handful of bristles at a wide block-in radius reads as fork/rake marks, not a
        // brush). Each lane still keeps its own evolving load along the whole stroke, so dirty-brush pickup and
        // stroke-order harmonisation survive. Capped so a very wide stroke can't explode the inner loop.
        let maxw = self.width0.max(self.width1).max(1.0);
        let nb = (maxw.ceil() as usize).max(brush.bristles).clamp(1, 256);
        // BRISTLE STREAKS (naturalness): a real brush is uneven — some bristles carry more paint than others, so
        // a stroke shows drybrush streaks along its length, not a uniform blob. Give each lane a slightly
        // different starting load (deterministic from the stroke's origin, so replay is exact). A few lanes run
        // nearly dry, laying broken texture; this is the single biggest cue that a mark was dragged, not stamped.
        let seed = (self.path[0][0].to_bits() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (self.path[0][1].to_bits() as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        let mut bload: Vec<Vec<f32>> = (0..nb)
            .map(|b| {
                // Streak amount from the brush: 0 = even (round), 1 = raked (stiff/fan). Centred on 1 so the
                // average load — and thus the mark's opacity — is preserved.
                let f = (1.0 + brush.streak.clamp(0.0, 1.0) * (lane_hash(seed, b as u64) - 0.5) * 1.6).max(0.08);
                self.load.iter().map(|c| c * f).collect()
            })
            .collect();
        let mut bwet: Vec<f32> = vec![self.wetness; nb];
        let n = canvas.n_pigments();
        let mut scratch: Vec<f32> = Vec::with_capacity(n); // reused deposit buffer — no per-pixel alloc

        for i in 0..pts.len() {
            let p = pts[i];
            // Local direction (forward difference), and the perpendicular the bristles spread along.
            let q = if i + 1 < pts.len() { pts[i + 1] } else { pts[i.saturating_sub(1)] };
            let (mut dx, mut dy) = (q[0] - p[0], q[1] - p[1]);
            let dl = (dx * dx + dy * dy).sqrt();
            if dl > 1e-4 {
                dx /= dl;
                dy /= dl;
            }
            let (perpx, perpy) = (-dy, dx);
            let t = if pts.len() > 1 { i as f32 / (pts.len() - 1) as f32 } else { 0.0 };
            let width = self.width0 + (self.width1 - self.width0) * t;

            // Ends taper: the first/last ~22% of the stroke deposits less, so a mark has soft ROUND tips, not a
            // rectangular butt (the "blocky patch" tell). Deeper falloff = more organic marks.
            let end = {
                let e = (t.min(1.0 - t) / 0.22).clamp(0.0, 1.0);
                0.15 + 0.85 * e * e
            };
            // At a tapered/narrow width many adjacent lanes round to the SAME pixel; skip the duplicates so a
            // wide stroke doesn't re-deposit the same pixel dozens of times (the block-in's dominant cost).
            let (mut last_x, mut last_y) = (i32::MIN, i32::MIN);
            for b in 0..nb {
                let fr = (b as f32 + 0.5) / nb as f32; // 0..1 across the width
                let off = (fr - 0.5) * width;
                let x = (p[0] + perpx * off).round();
                let y = (p[1] + perpy * off).round();
                if x < 0.0 || y < 0.0 || x >= canvas.w as f32 || y >= canvas.h as f32 {
                    continue;
                }
                let (px, py) = (x as u32, y as u32);
                if px as i32 == last_x && py as i32 == last_y {
                    continue;
                }
                last_x = px as i32;
                last_y = py as i32;
                // Cross-section falloff by brush ROUNDNESS: a round brush feathers gently to the edge (soft
                // mark); a flat brush holds a flatter top and drops sharper (a squarer edge). `round` in [0,1]
                // interpolates between them.
                let edge = {
                    let d = (fr - 0.5).abs() * 2.0; // 0 centre → 1 edge
                    let rnd = brush.round.clamp(0.0, 1.0);
                    let pw = 2.0 + (1.0 - rnd) * 6.0; // round → parabola, flat → flatter top
                    let floor = 0.12 + (1.0 - rnd) * 0.33; // flat brush deposits more evenly across its width
                    (1.0 - d.powf(pw)).max(floor)
                };
                self.apply(canvas, px, py, &mut bload[b], &mut bwet[b], brush, n, edge * end, &mut scratch);
            }
        }
    }

    /// Apply this stroke as a WIPE (§8.7): walk the path with the brush footprint and scrape `strength` of the
    /// pigment + height back at each covered pixel, exposing what's beneath. No load, no pickup.
    pub fn wipe(&self, canvas: &mut Canvas, brush: &BrushConfig, strength: f32) {
        let pts = densify(&self.path);
        // Solid swath, same as `rasterize` — a wipe must scrape a continuous band, not comb-teeth.
        let maxw = self.width0.max(self.width1).max(1.0);
        let nb = (maxw.ceil() as usize).max(brush.bristles).clamp(1, 256);
        for i in 0..pts.len() {
            let p = pts[i];
            let q = if i + 1 < pts.len() { pts[i + 1] } else { pts[i.saturating_sub(1)] };
            let (mut dx, mut dy) = (q[0] - p[0], q[1] - p[1]);
            let dl = (dx * dx + dy * dy).sqrt();
            if dl > 1e-4 {
                dx /= dl;
                dy /= dl;
            }
            let (perpx, perpy) = (-dy, dx);
            let t = if pts.len() > 1 { i as f32 / (pts.len() - 1) as f32 } else { 0.0 };
            let width = self.width0 + (self.width1 - self.width0) * t;
            for b in 0..nb {
                let off = ((b as f32 + 0.5) / nb as f32 - 0.5) * width;
                let x = (p[0] + perpx * off).round();
                let y = (p[1] + perpy * off).round();
                if x < 0.0 || y < 0.0 || x >= canvas.w as f32 || y >= canvas.h as f32 {
                    continue;
                }
                canvas.wipe(x as u32, y as u32, strength * self.pressure);
            }
        }
    }

    /// One bristle touching one pixel: deposit a fraction of load, pick up wet canvas pigment, update the load.
    /// `cover` (0..1) is the soft footprint weight — the cross-section falloff toward the width's edges and the
    /// end taper — so a stroke reads as a brush mark, not a hard rectangular slab.
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, canvas: &mut Canvas, px: u32, py: u32, load: &mut [f32], bwet: &mut f32, brush: &BrushConfig, n: usize, cover: f32, deposit: &mut Vec<f32>) {
        let p = py as usize * canvas.w as usize + px as usize;
        let tooth = canvas.tooth[p];
        let contact = (self.pressure * cover.clamp(0.0, 1.0) * tooth).clamp(0.0, 1.0);
        let sat = (canvas.saturation_at(px, py) / SAT_FULL).clamp(0.0, 1.0);

        // Deposit: a fraction of the current load, throttled by contact and remaining tooth. `deposit` is a
        // caller-owned scratch buffer, cleared here — no per-pixel heap allocation.
        let df = (brush.k_deposit * contact * (1.0 - SAT_THROTTLE * sat)).clamp(0.0, 1.0);
        deposit.clear();
        deposit.resize(n, 0.0);
        let mut dep_total = 0.0;
        for c in 0..n.min(load.len()) {
            let d = load[c].max(0.0) * df;
            deposit[c] = d;
            dep_total += d;
            load[c] -= d;
        }
        canvas.deposit(px, py, deposit, dep_total * brush.viscosity);

        // Pickup: lift wet canvas pigment into the load (the dirty brush). Scales with how empty the brush is.
        let cw = canvas.wetness[p];
        let empty = (1.0 - Self::load_norm(load) / brush.load_max.max(1e-3)).clamp(0.0, 1.0);
        let pf = brush.k_pickup * cw * empty;
        if pf > 0.0 {
            let csum = canvas.saturation_at(px, py).max(1e-6);
            let cconc = canvas.conc_at(px, py); // borrow directly — no clone
            for c in 0..n.min(load.len()) {
                load[c] += (cconc[c] / csum) * pf;
            }
        }

        // The stroke wets the canvas where it lands (wet-into-wet for later strokes); bristle dries slightly.
        canvas.wetness[p] = canvas.wetness[p].max((*bwet).min(1.0));
        *bwet *= 0.995;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::canvas::Canvas;
    use crate::paint::color::{delta_e76, srgb_to_lab};
    use crate::paint::palette;

    // Zorn indices: 0 ochre, 1 cad-red, 2 black, 3 white. A well-charged red load.
    fn red_load() -> Vec<f32> {
        vec![0.0, 4.0, 0.0, 0.0]
    }

    #[test]
    fn a_stroke_deposits_its_colour() {
        let mut c = Canvas::white(40, 40, palette::ZORN, 0.9);
        let s = Stroke { path: vec![[5.0, 20.0], [34.0, 20.0]], width0: 6.0, width1: 6.0, load: red_load(), pressure: 1.0, wetness: 1.0 };
        s.rasterize(&mut c, &BrushConfig::default());
        // The stroke deposits RED pigment: a loaded pixel is measurably shifted from the white ground toward red
        // (a* positive). Full opacity builds across overlapping passes (glaze model) — one thin pass need not be
        // fully opaque, but it must clearly carry its own colour.
        let on = srgb_to_lab(c.color_at(9, 20));
        let white = srgb_to_lab([252, 251, 248]);
        assert!(on.a > 3.0, "the stroke reads red-biased (a* {})", on.a);
        assert!(delta_e76(on, white) > 6.0, "the stroke changed the canvas from bare ground (ΔE {})", delta_e76(on, white));
        assert!(c.height[20 * 40 + 9] > 0.0, "impasto height laid");
    }

    #[test]
    fn the_stroke_dries_toward_its_end() {
        // A charged brush lays most paint at the start and less as it dries — the loaded-gradient.
        let mut c = Canvas::white(60, 20, palette::ZORN, 0.9);
        let s = Stroke { path: vec![[3.0, 10.0], [56.0, 10.0]], width0: 5.0, width1: 5.0, load: red_load(), pressure: 1.0, wetness: 1.0 };
        s.rasterize(&mut c, &BrushConfig::default());
        assert!(c.height[10 * 60 + 6] > c.height[10 * 60 + 52], "more paint near the loaded start than the dry end");
    }

    #[test]
    fn is_deterministic() {
        let brush = BrushConfig::default();
        let s = Stroke { path: vec![[2.0, 10.0], [30.0, 12.0]], width0: 5.0, width1: 3.0, load: red_load(), pressure: 0.9, wetness: 1.0 };
        let mut a = Canvas::white(32, 24, palette::ZORN, 0.8);
        let mut b = Canvas::white(32, 24, palette::ZORN, 0.8);
        s.rasterize(&mut a, &brush);
        s.rasterize(&mut b, &brush);
        assert_eq!(a.to_image().into_raw(), b.to_image().into_raw(), "same stroke → identical canvas");
    }

    #[test]
    fn the_dirty_brush_picks_up_and_harmonises() {
        // Comparative: the SAME white drag over a wet black band vs over a clean canvas. Where it crossed the
        // wet band, the brush picked up black, so its later deposit is darker than the clean-canvas control.
        let white = vec![0.0, 0.0, 0.0, 3.0];
        let drag = || Stroke { path: vec![[6.0, 15.0], [50.0, 15.0]], width0: 6.0, width1: 6.0, load: white.clone(), pressure: 0.9, wetness: 1.0 };

        // Control: clean white canvas. Sample just past the band, where carried pigment is still fresh (an
        // opaque brush sheds it as it lays more white further along).
        let mut clean = Canvas::white(60, 30, palette::ZORN, 0.9);
        drag().rasterize(&mut clean, &BrushConfig::default());
        let control = srgb_to_lab(clean.color_at(14, 15)).l;

        // With a wet black band the drag crosses near its start.
        let mut banded = Canvas::white(60, 30, palette::ZORN, 0.9);
        let band = Stroke { path: vec![[6.0, 3.0], [6.0, 27.0]], width0: 8.0, width1: 8.0, load: vec![0.0, 0.0, 1.0, 0.0], pressure: 1.0, wetness: 1.0 };
        band.rasterize(&mut banded, &BrushConfig::default());
        drag().rasterize(&mut banded, &BrushConfig::default());
        let picked = srgb_to_lab(banded.color_at(14, 15)).l;

        assert!(picked < control - 1.0, "carrying picked-up black darkens the later deposit ({picked} vs control {control})");
    }
}
