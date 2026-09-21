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
}

impl Default for BrushConfig {
    fn default() -> Self {
        // Oil-direct-ish: deposit a modest fraction of the load per step (so a charged brush covers a stroke
        // and dries toward its end — the natural loaded-gradient), real pickup, impasto.
        Self { k_deposit: 0.12, k_pickup: 0.6, viscosity: 1.0, bristles: 7, load_max: 6.0 }
    }
}

/// Saturation at which the tooth is full and deposit stops.
const SAT_FULL: f32 = 6.0;

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
        let nb = brush.bristles.max(1);
        // Each bristle starts with the stroke's load and its own wetness; they diverge as they pick up.
        let mut bload: Vec<Vec<f32>> = vec![self.load.clone(); nb];
        let mut bwet: Vec<f32> = vec![self.wetness; nb];
        let n = canvas.n_pigments();

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

            for b in 0..nb {
                let off = ((b as f32 + 0.5) / nb as f32 - 0.5) * width;
                let x = (p[0] + perpx * off).round();
                let y = (p[1] + perpy * off).round();
                if x < 0.0 || y < 0.0 || x >= canvas.w as f32 || y >= canvas.h as f32 {
                    continue;
                }
                let (px, py) = (x as u32, y as u32);
                self.apply(canvas, px, py, &mut bload[b], &mut bwet[b], brush, n);
            }
        }
    }

    /// One bristle touching one pixel: deposit a fraction of load, pick up wet canvas pigment, update the load.
    fn apply(&self, canvas: &mut Canvas, px: u32, py: u32, load: &mut [f32], bwet: &mut f32, brush: &BrushConfig, n: usize) {
        let p = py as usize * canvas.w as usize + px as usize;
        let tooth = canvas.tooth[p];
        let contact = (self.pressure * tooth).clamp(0.0, 1.0);
        let sat = (canvas.saturation_at(px, py) / SAT_FULL).clamp(0.0, 1.0);

        // Deposit: a fraction of the current load, throttled by contact and remaining tooth.
        let df = (brush.k_deposit * contact * (1.0 - sat)).clamp(0.0, 1.0);
        let mut deposit = vec![0f32; n];
        let mut dep_total = 0.0;
        for c in 0..n.min(load.len()) {
            let d = load[c].max(0.0) * df;
            deposit[c] = d;
            dep_total += d;
            load[c] -= d;
        }
        canvas.deposit(px, py, &deposit, dep_total * brush.viscosity);

        // Pickup: lift wet canvas pigment into the load (the dirty brush). Scales with how empty the brush is.
        let cw = canvas.wetness[p];
        let empty = (1.0 - Self::load_norm(load) / brush.load_max.max(1e-3)).clamp(0.0, 1.0);
        let pf = brush.k_pickup * cw * empty;
        if pf > 0.0 {
            let csum = canvas.saturation_at(px, py).max(1e-6);
            let cconc: Vec<f32> = canvas.conc_at(px, py).to_vec();
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
        // In the loaded first third of the stroke, a centreline pixel reads toward red, not white.
        let on = srgb_to_lab(c.color_at(8, 20));
        let d_red = delta_e76(on, srgb_to_lab(palette::CADMIUM_RED.masstone));
        let d_white = delta_e76(on, srgb_to_lab([252, 251, 248]));
        assert!(d_red < d_white, "stroke pixel is nearer red than white ({d_red} vs {d_white})");
        assert!(c.height[20 * 40 + 8] > 0.0, "impasto height laid");
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

        // Control: clean white canvas.
        let mut clean = Canvas::white(60, 30, palette::ZORN, 0.9);
        drag().rasterize(&mut clean, &BrushConfig::default());
        let control = srgb_to_lab(clean.color_at(30, 15)).l;

        // With a wet black band the drag crosses near its start.
        let mut banded = Canvas::white(60, 30, palette::ZORN, 0.9);
        let band = Stroke { path: vec![[6.0, 3.0], [6.0, 27.0]], width0: 8.0, width1: 8.0, load: vec![0.0, 0.0, 1.0, 0.0], pressure: 1.0, wetness: 1.0 };
        band.rasterize(&mut banded, &BrushConfig::default());
        drag().rasterize(&mut banded, &BrushConfig::default());
        let picked = srgb_to_lab(banded.color_at(30, 15)).l;

        assert!(picked < control - 2.0, "carrying picked-up black darkens the later deposit ({picked} vs control {control})");
    }
}
