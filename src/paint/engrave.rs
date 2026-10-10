//! COPPERPLATE ENGRAVING (`--medium durer`): the plate a burin cuts, planned from the picture's own values.
//! Like the pen drawing in [`super::ink`] it is geometry only — polylines with a width — so the painter
//! rasterises each cut as a stroke, records it, and the plate replays byte-exact from its score. No weights,
//! nothing random beyond the seed.
//!
//! What makes a plate read as an engraving rather than as a net of scratches is discipline, and each rule
//! here is one of the engraver's:
//!
//! * **One direction per surface.** The lines of a layer run along the form: the dominant orientation of the
//!   picture's edges, smoothed over a wide window, so a tyre is cut in rings and a tube along its length. Where
//!   no orientation dominates — the face of a box, an open ground — the plate falls back to the engraver's
//!   resting diagonal instead of wandering.
//! * **Even lines.** Each layer is a set of streamlines kept one spacing apart (Jobard–Lefer: every new line
//!   is seeded beside an existing one), so a tone is a field of parallel cuts, not a tangle.
//! * **Tone by the weight of the line, then by crossing.** A cut swells with the darkness under it and thins
//!   to a point where the tone lifts. A second layer crosses the first only where one layer at full weight
//!   cannot carry the tone, a third where two cannot — so the darks are deep and the lights are paper.
//! * **A firm contour of varying weight.** Edges are traced as continuous lines whose width follows the
//!   contrast they separate: heavy on a silhouette, a hair inside a form.

use super::ink::{edge_map, jitter, sobel, trace_chains, value_map};
use image::RgbImage;

/// One cut of the burin: a polyline whose width runs linearly from `w0` to `w1`. A swelling line is a chain of
/// cuts that share their end points. `layer` 0 is a contour, 1.. the hatch layers.
#[derive(Clone, Debug)]
pub struct Cut {
    pub layer: usize,
    pub path: Vec<[f32; 2]>,
    pub w0: f32,
    pub w1: f32,
}

/// A planned plate.
pub struct Plate {
    pub cuts: Vec<Cut>,
    /// The line spacing the plate ended on (it widens when the cuts would not fit the budget).
    pub spacing: f32,
}

/// The paper each hatch layer may ink at full weight. The first two stay open — a single layer never closes a
/// tone, it hands over to a crossing — and the last is allowed to cut heavy, which is what makes a black.
const LAYER_COVER: [f32; 4] = [0.33, 0.33, 0.45, 0.7];
/// How inked the subject's middle value is cut.
const MIDDLE_TONE: f32 = 0.26;
/// How much a surface is shaded against its surroundings.
const RELIEF: f32 = 0.4;
/// The deepest tone the plate cuts: a burin never fills a black, paper still shows between its lines.
const DEEPEST: f32 = 0.9;
/// How far below a hair's worth of tone the first layer still cuts, as flicks: a light tone is a broken line.
const FLICK: f32 = 0.3;
/// Each layer's angle off the form's direction: along it, across it, then the two obliques.
const LAYER_ANGLE: [f32; 4] = [0.0, 1.5708, 0.7854, 2.3562];

/// The share of the paper layer `k` inks where the plate's tone is `t`: the layers fill in order, each taking
/// what the ones before it left, so `k` crossings together ink exactly `t`.
fn cover(t: f32, k: usize) -> f32 {
    let mut paper = 1.0f32;
    for (j, &cap) in LAYER_COVER.iter().enumerate() {
        let c = (1.0 - (1.0 - t) / paper).clamp(0.0, cap);
        if j == k {
            return c;
        }
        paper *= 1.0 - c;
    }
    0.0
}

/// Box blur by running sums, `passes` times (three passes are a Gaussian to the eye).
fn blur(v: &[f32], w: usize, h: usize, r: usize, passes: usize) -> Vec<f32> {
    let mut cur = v.to_vec();
    if r == 0 {
        return cur;
    }
    let mut tmp = vec![0f32; w * h];
    let mut acc = vec![0f32; w.max(h) + 1];
    for _ in 0..passes {
        for y in 0..h {
            acc[0] = 0.0;
            for x in 0..w {
                acc[x + 1] = acc[x] + cur[y * w + x];
            }
            for x in 0..w {
                let (a, b) = (x.saturating_sub(r), (x + r + 1).min(w));
                tmp[y * w + x] = (acc[b] - acc[a]) / (b - a) as f32;
            }
        }
        for x in 0..w {
            acc[0] = 0.0;
            for y in 0..h {
                acc[y + 1] = acc[y] + tmp[y * w + x];
            }
            for y in 0..h {
                let (a, b) = (y.saturating_sub(r), (y + r + 1).min(h));
                cur[y * w + x] = (acc[b] - acc[a]) / (b - a) as f32;
            }
        }
    }
    cur
}

/// The direction the cuts run in, held at a fraction of the sheet's resolution as a doubled angle (a line has
/// no head or tail, so its direction is averaged as `2θ`).
struct Flow {
    w: usize,
    h: usize,
    scale: f32,
    c2: Vec<f32>,
    s2: Vec<f32>,
}

impl Flow {
    fn new(value: &[f32], w: usize, h: usize) -> Flow {
        let long = w.max(h);
        let f = long.div_ceil(512).max(1);
        let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
        let mut small = vec![0f32; sw * sh];
        let mut count = vec![0f32; sw * sh];
        for y in 0..h {
            for x in 0..w {
                let i = (y / f) * sw + x / f;
                small[i] += value[y * w + x];
                count[i] += 1.0;
            }
        }
        for (s, c) in small.iter_mut().zip(&count) {
            *s /= c.max(1.0);
        }
        let (gx, gy, _) = sobel(&small, sw, sh);
        let n = sw * sh;
        let (mut jxx, mut jyy, mut jxy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        for i in 0..n {
            jxx[i] = gx[i] * gx[i];
            jyy[i] = gy[i] * gy[i];
            jxy[i] = gx[i] * gy[i];
        }
        // Wide enough to take in a whole surface and the edges that bound it.
        let r = ((long as f32 / 55.0 / f as f32).round() as usize).max(2);
        let (jxx, jyy, jxy) = (blur(&jxx, sw, sh, r, 3), blur(&jyy, sw, sh, r, 3), blur(&jxy, sw, sh, r, 3));
        let energy = ((jxx.iter().sum::<f32>() + jyy.iter().sum::<f32>()) / n as f32).max(1e-12);
        let (mut c2, mut s2) = (vec![0f32; n], vec![0f32; n]);
        for i in 0..n {
            let (a, b) = (jxx[i] - jyy[i], 2.0 * jxy[i]);
            let disc = (a * a + b * b).sqrt();
            let coherence = disc / (jxx[i] + jyy[i] + 1e-12);
            // How far the form speaks for itself: below, the resting diagonal (up to the right) takes over.
            let t = ((coherence - 0.2) / 0.35).clamp(0.0, 1.0);
            // ... and only near the edges that give it: far inside a flat face the little orientation that reaches
            // it is the lighting's drift, and following that grains the face like wood.
            let e = ((jxx[i] + jyy[i]) / energy / 1.5).clamp(0.0, 1.0);
            let form = t * t * (3.0 - 2.0 * t) * e * e * (3.0 - 2.0 * e);
            // The edge's tangent is the gradient's orientation turned a quarter: the doubled angle negated.
            let (fc, fs) = if disc > 1e-12 { (-a / disc, -b / disc) } else { (0.0, -1.0) };
            c2[i] = form * fc;
            s2[i] = form * fs - (1.0 - form);
        }
        Flow { w: sw, h: sh, scale: 1.0 / f as f32, c2: blur(&c2, sw, sh, 2, 2), s2: blur(&s2, sw, sh, 2, 2) }
    }

    /// The unit direction at a sheet position, turned by the layer's angle (`sin`, `cos`).
    fn dir(&self, x: f32, y: f32, turn: (f32, f32)) -> [f32; 2] {
        let (fx, fy) = ((x * self.scale - 0.5).clamp(0.0, self.w as f32 - 1.0), (y * self.scale - 0.5).clamp(0.0, self.h as f32 - 1.0));
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let at = |v: &[f32]| (v[y0 * self.w + x0] * (1.0 - tx) + v[y0 * self.w + x1] * tx) * (1.0 - ty) + (v[y1 * self.w + x0] * (1.0 - tx) + v[y1 * self.w + x1] * tx) * ty;
        let theta = 0.5 * at(&self.s2).atan2(at(&self.c2));
        let (s, c) = theta.sin_cos();
        [c * turn.1 - s * turn.0, c * turn.0 + s * turn.1]
    }
}

/// The points already cut in a layer, binned so "is there a line within r of here" is a look at nine cells.
struct Near {
    cell: f32,
    cw: usize,
    ch: usize,
    cells: Vec<Vec<[f32; 2]>>,
}

impl Near {
    fn new(w: usize, h: usize, cell: f32) -> Near {
        let (cw, ch) = ((w as f32 / cell) as usize + 1, (h as f32 / cell) as usize + 1);
        Near { cell, cw, ch, cells: vec![Vec::new(); cw * ch] }
    }
    fn add(&mut self, p: [f32; 2]) {
        let (cx, cy) = (((p[0] / self.cell) as usize).min(self.cw - 1), ((p[1] / self.cell) as usize).min(self.ch - 1));
        self.cells[cy * self.cw + cx].push(p);
    }
    fn any_within(&self, p: [f32; 2], r: f32) -> bool {
        let (cx, cy) = ((p[0] / self.cell) as i64, (p[1] / self.cell) as i64);
        let reach = (r / self.cell).ceil() as i64;
        for y in (cy - reach).max(0)..=(cy + reach).min(self.ch as i64 - 1) {
            for x in (cx - reach).max(0)..=(cx + reach).min(self.cw as i64 - 1) {
                for q in &self.cells[y as usize * self.cw + x as usize] {
                    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
                    if dx * dx + dy * dy < r * r {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// What a layer's lines are cut through.
struct Field<'a> {
    w: usize,
    h: usize,
    /// The plate's tone, 0 = paper, 1 = black.
    tone: &'a [f32],
    /// Where a cut must stop: the contours.
    wall: &'a [bool],
    flow: &'a Flow,
    spacing: f32,
    /// The finest line the burin leaves.
    hair: f32,
}

impl Field<'_> {
    fn width(&self, p: [f32; 2], layer: usize) -> f32 {
        cover(self.tone[(p[1] as usize).min(self.h - 1) * self.w + (p[0] as usize).min(self.w - 1)], layer) * self.spacing
    }
    fn open(&self, p: [f32; 2], layer: usize) -> bool {
        if p[0] < 0.0 || p[1] < 0.0 || p[0] >= self.w as f32 || p[1] >= self.h as f32 {
            return false;
        }
        // The first layer runs on into the lights, where it is cut as flicks (see `plan`).
        let least = if layer == 0 { self.hair * FLICK } else { self.hair };
        !self.wall[p[1] as usize * self.w + p[0] as usize] && self.width(p, layer) >= least
    }

    /// One streamline through `start`, both ways, stopping at the paper, at a contour, beside another line, or
    /// when the direction field folds on itself.
    fn trace(&self, start: [f32; 2], layer: usize, near: &Near) -> Vec<[f32; 2]> {
        let turn = LAYER_ANGLE[layer].sin_cos();
        let max_steps = (self.spacing * 90.0) as usize;
        let d_test = self.spacing * 0.55;
        let mut line: Vec<[f32; 2]> = Vec::new();
        for sign in [-1.0f32, 1.0] {
            let d0 = self.flow.dir(start[0], start[1], turn);
            let mut prev = [d0[0] * sign, d0[1] * sign];
            let mut p = start;
            let mut swept = 0f32;
            let mut pts: Vec<[f32; 2]> = Vec::new();
            for _ in 0..max_steps {
                let mut d = self.flow.dir(p[0], p[1], turn);
                if d[0] * prev[0] + d[1] * prev[1] < 0.0 {
                    d = [-d[0], -d[1]];
                }
                // A kink is where two surfaces' directions meet; a burin lifts there.
                if d[0] * prev[0] + d[1] * prev[1] < 0.8 {
                    break;
                }
                swept += (prev[0] * d[1] - prev[1] * d[0]).clamp(-1.0, 1.0).asin();
                if swept.abs() > 5.5 {
                    break;
                }
                p = [p[0] + d[0], p[1] + d[1]];
                if !self.open(p, layer) || near.any_within(p, d_test) {
                    break;
                }
                pts.push(p);
                prev = d;
            }
            if sign < 0.0 {
                pts.reverse();
                line = pts;
                line.push(start);
            } else {
                line.extend(pts);
            }
        }
        line
    }

    /// Every line of one layer, evenly spaced: each accepted line seeds its neighbours one spacing to either
    /// side, and a coarse scan of the sheet starts the surfaces no neighbour reached.
    fn layer(&self, layer: usize, seed: u64) -> Vec<Vec<[f32; 2]>> {
        let mut near = Near::new(self.w, self.h, self.spacing);
        let mut lines: Vec<Vec<[f32; 2]>> = Vec::new();
        let min_len = (self.spacing * 2.0).max(4.0) as usize;
        let seed_gap = self.spacing * 0.92;
        let every = (self.spacing * 0.5).max(1.0) as usize;
        let mut next = 0usize;
        let grow = |from: [f32; 2], lines: &mut Vec<Vec<[f32; 2]>>, near: &mut Near, next: &mut usize| {
            if !self.open(from, layer) || near.any_within(from, seed_gap) {
                return;
            }
            let first = self.trace(from, layer, near);
            if first.len() < min_len {
                return;
            }
            for p in first.iter().step_by(2) {
                near.add(*p);
            }
            lines.push(first);
            while *next < lines.len() {
                let n = lines[*next].len();
                for i in (0..n).step_by(every) {
                    let (a, b) = (lines[*next][i.saturating_sub(1)], lines[*next][(i + 1).min(n - 1)]);
                    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                    let len = (dx * dx + dy * dy).sqrt().max(1e-4);
                    let here = lines[*next][i];
                    for side in [-1.0f32, 1.0] {
                        let c = [here[0] - dy / len * self.spacing * side, here[1] + dx / len * self.spacing * side];
                        if !self.open(c, layer) || near.any_within(c, seed_gap) {
                            continue;
                        }
                        let line = self.trace(c, layer, near);
                        if line.len() < min_len {
                            continue;
                        }
                        for p in line.iter().step_by(2) {
                            near.add(*p);
                        }
                        lines.push(line);
                    }
                }
                *next += 1;
            }
        };
        let step = self.spacing * 3.0;
        let (mut y, mut k) = (step * 0.5, 0u64);
        while y < self.h as f32 {
            let mut x = step * 0.5;
            while x < self.w as f32 {
                k += 1;
                let from = [x + jitter(seed ^ 0xE6A1 ^ layer as u64, k) * self.spacing, y + jitter(seed ^ 0xE6A2 ^ layer as u64, k) * self.spacing];
                grow(from, &mut lines, &mut near, &mut next);
                x += step;
            }
            y += step;
        }
        lines
    }
}

/// Cut a line of varying width into pieces whose width is linear, sharing their end points.
fn pieces(layer: usize, pts: &[[f32; 2]], widths: &[f32], out: &mut Vec<Cut>) {
    let n = pts.len();
    if n < 2 {
        return;
    }
    let mut a = 0usize;
    while a + 1 < n {
        let mut b = a + 1;
        'grow: while b + 1 < n && b + 1 - a <= 40 {
            let c = b + 1;
            for i in a + 1..c {
                let t = (i - a) as f32 / (c - a) as f32;
                if (widths[a] + (widths[c] - widths[a]) * t - widths[i]).abs() > 0.3 {
                    break 'grow;
                }
            }
            b = c;
        }
        out.push(Cut { layer, path: pts[a..=b].to_vec(), w0: widths[a], w1: widths[b] });
        a = b;
    }
}

/// Smooth a line's widths along it and bring both ends to a point, as a burin enters and leaves the copper.
fn swell(widths: &mut [f32], window: usize, point: f32, run: usize) {
    let n = widths.len();
    let src = widths.to_vec();
    for i in 0..n {
        let (a, b) = (i.saturating_sub(window), (i + window + 1).min(n));
        widths[i] = src[a..b].iter().sum::<f32>() / (b - a) as f32;
    }
    for i in 0..n {
        let end = i.min(n - 1 - i) as f32 / run.max(1) as f32;
        if end < 1.0 {
            widths[i] = widths[i].min(point + (widths[i] - point).max(0.0) * end);
        }
    }
}

/// Plan the plate for a picture. `contour` (0..1) is how much of the edge map is drawn; `budget` caps the
/// number of cuts — when the hatch would not fit, its spacing widens until it does.
pub fn plan(picture: &RgbImage, contour: f32, budget: usize, seed: u64) -> Plate {
    let (w, h) = (picture.width() as usize, picture.height() as usize);
    let long = w.max(h) as f32;
    if w < 8 || h < 8 {
        return Plate { cuts: Vec::new(), spacing: 0.0 };
    }
    // The value the plate is cut from: the picture, with the texture a burin does not follow smoothed away.
    let value = blur(&value_map(picture), w, h, ((long / 900.0).round() as usize).max(1), 3);

    // TONE. The paper is the picture's light, the deepest cut its dark; between them the plate keeps the
    // picture's order of values but spends its whole range, which is what gives an engraving its relief.
    let mut sorted: Vec<f32> = value.iter().step_by(7).copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f32| sorted[((sorted.len() as f32 - 1.0) * f) as usize];
    let white = q(0.90);
    let black = q(0.02).min(white - 0.2);
    // Tone is a broader thing than line: it is read off the picture a little out of focus, so a hatch does not
    // break on the grain of a canvas or the blocks of a JPEG. The contours keep the sharp reading.
    let soft = blur(&value, w, h, ((long / 700.0).round() as usize).max(1), 3);
    let dark: Vec<f32> = soft.iter().map(|v| ((white - v) / (white - black)).clamp(0.0, 1.0)).collect();
    // An engraver does not copy a photograph's key. He models — a surface is shaded against its surroundings,
    // so the turn of a form and the shadow in a joint read stronger than the flat between them — and he spends
    // the plate's whole range on the subject whatever range the picture gave it: mostly by the ORDER of its
    // values, so a subject that is dark all over is still cut from open line to closed shadow, with its middle
    // value about a third inked.
    let around = blur(&dark, w, h, ((long / 40.0) as usize).max(2), 3);
    let modelled: Vec<f32> = dark.iter().zip(&around).map(|(&d, &a)| (d + RELIEF * (d - a)).clamp(0.0, 1.0)).collect();
    let mut hist = [0f32; 256];
    for d in modelled.iter().filter(|d| **d > 0.08) {
        hist[((d * 255.0) as usize).min(255)] += 1.0;
    }
    let total: f32 = hist.iter().sum::<f32>().max(1.0);
    let mut rank = [0f32; 256];
    let mut run = 0f32;
    for (r, n) in rank.iter_mut().zip(&hist) {
        run += n;
        *r = run / total;
    }
    let key = (MIDDLE_TONE / DEEPEST).ln() / (0.7f32 * 0.5 + 0.3 * 0.5).ln();
    let tone: Vec<f32> = modelled.iter().map(|&d| DEEPEST * (0.7 * if d > 0.08 { rank[((d * 255.0) as usize).min(255)] } else { 0.0 } + 0.3 * d).powf(key)).collect();

    // CONTOURS: the edge map's chains, each a line whose weight follows the contrast it separates.
    let mut cuts: Vec<Cut> = Vec::new();
    let mut wall = vec![false; w * h];
    if contour > 0.0 {
        let edge = edge_map(&value, w, h, contour);
        let (_, _, mag) = sobel(&value, w, h);
        let chains = trace_chains(&edge, w, h, (long / 130.0).max(5.0) as usize);
        let at = |p: &[f32; 2]| mag[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)];
        let mut strengths: Vec<f32> = chains.iter().flat_map(|c| c.iter().map(at)).collect();
        strengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let strong = strengths.get(strengths.len() * 95 / 100).copied().unwrap_or(1.0).max(1e-6);
        // A line is worth drawing by its length AND its strength: a short firm edge is a rivet, a long faint
        // one the turn of a cheek, but a short faint one is craquelure, grain, noise — an engraver leaves it.
        let weight = |c: &Vec<[f32; 2]>| c.iter().map(|p| (at(p) / strong).min(1.0)).sum::<f32>() * 2.0;
        let mut chains: Vec<&Vec<[f32; 2]>> = chains.iter().filter(|c| weight(c) >= long / 140.0).collect();
        chains.sort_by_key(|c| std::cmp::Reverse(c.len()));
        let base = (long / 800.0).clamp(0.8, 3.0);
        for chain in chains {
            let mut widths: Vec<f32> = chain.iter().map(|p| base * (0.45 + 1.9 * (at(p) / strong).min(1.0).powf(0.8))).collect();
            swell(&mut widths, 4, base * 0.4, 5);
            pieces(0, chain, &widths, &mut cuts);
            // A hatch line stops at a drawn contour (thickened, so a diagonal step cannot slip through).
            for pair in chain.windows(2) {
                for p in [pair[0], [(pair[0][0] + pair[1][0]) * 0.5, (pair[0][1] + pair[1][1]) * 0.5], pair[1]] {
                    let (x, y) = ((p[0] as usize).min(w - 2), (p[1] as usize).min(h - 2));
                    for i in [y * w + x, y * w + x + 1, (y + 1) * w + x] {
                        wall[i] = true;
                    }
                }
            }
            if cuts.len() >= budget * 2 / 5 {
                break;
            }
        }
    }

    // HATCH: the layers, at a spacing that fits the budget.
    let flow = Flow::new(&value, w, h);
    let room = budget.saturating_sub(cuts.len());
    let mut spacing = (long / 340.0).max(3.0);
    let mut hatch: Vec<Cut> = Vec::new();
    for _ in 0..8 {
        hatch.clear();
        let field = Field { w, h, tone: &tone, wall: &wall, flow: &flow, spacing, hair: (spacing * 0.15).max(0.45) };
        for layer in 0..LAYER_COVER.len() {
            for (li, line) in field.layer(layer, seed).into_iter().enumerate() {
                let raw: Vec<f32> = line.iter().map(|p| field.width(*p, layer)).collect();
                // In the lights a line is not thinner than a hair — it is BROKEN: flicks whose length carries
                // the tone, each line breaking at its own place so the flicks do not fall into ranks.
                let period = spacing * 3.0;
                let phase = (jitter(seed ^ 0xF11C ^ layer as u64, li as u64) + 0.5) * period;
                let on = |i: usize| raw[i] >= field.hair || (i as f32 + phase) % period < period * (raw[i] / field.hair).max(0.25);
                let mut a = 0usize;
                while a < line.len() {
                    if !on(a) {
                        a += 1;
                        continue;
                    }
                    let mut b = a;
                    while b + 1 < line.len() && on(b + 1) {
                        b += 1;
                    }
                    if b - a >= 3 {
                        let mut widths: Vec<f32> = raw[a..=b].iter().map(|w| w.max(field.hair)).collect();
                        swell(&mut widths, 3, field.hair * 0.6, ((spacing * 1.5) as usize).min((b - a) / 2));
                        let thin: Vec<usize> = (0..=b - a).step_by(2).chain(std::iter::once(b - a).filter(|l| l % 2 == 1)).collect();
                        let pts: Vec<[f32; 2]> = thin.iter().map(|&i| line[a + i]).collect();
                        let ws: Vec<f32> = thin.iter().map(|&i| widths[i]).collect();
                        pieces(layer + 1, &pts, &ws, &mut hatch);
                    }
                    a = b + 1;
                }
            }
        }
        if hatch.len() <= room {
            break;
        }
        spacing *= 1.25;
    }
    hatch.truncate(room);
    cuts.extend(hatch);
    Plate { cuts, spacing }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_masses(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| if x > w / 4 && x < 3 * w / 4 && y > h / 4 && y < 3 * h / 4 { image::Rgb([40, 40, 45]) } else { image::Rgb([235, 232, 226]) })
    }

    #[test]
    fn the_layers_together_ink_the_tone_asked_for() {
        for t in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
            let paper: f32 = (0..LAYER_COVER.len()).map(|k| 1.0 - cover(t, k)).product();
            assert!((1.0 - paper - t).abs() < 1e-4, "tone {t}: the layers ink {}", 1.0 - paper);
        }
        assert_eq!(cover(0.2, 1), 0.0, "a light tone is one layer, not a crossing");
        assert!(cover(0.6, 1) > 0.0, "a dark one is crossed");
    }

    #[test]
    fn a_dark_mass_is_cross_hatched_and_the_ground_stays_paper() {
        let plate = plan(&two_masses(160, 160), 0.6, 50_000, 7);
        let hatch: Vec<&Cut> = plate.cuts.iter().filter(|c| c.layer > 0).collect();
        assert!(!hatch.is_empty(), "the dark mass is engraved");
        let outside = hatch.iter().flat_map(|c| c.path.iter()).filter(|p| !(p[0] > 36.0 && p[0] < 124.0 && p[1] > 36.0 && p[1] < 124.0)).count();
        assert_eq!(outside, 0, "no cut on the light ground");
        let layers: std::collections::BTreeSet<usize> = hatch.iter().map(|c| c.layer).collect();
        assert!(layers.len() >= 3, "a dark is several crossings deep ({layers:?})");
        assert!(plate.cuts.iter().any(|c| c.layer == 0), "its boundary is drawn");
    }

    #[test]
    fn lines_of_a_layer_keep_their_distance() {
        let plate = plan(&two_masses(200, 200), 0.0, 50_000, 7);
        let first: Vec<&Cut> = plate.cuts.iter().filter(|c| c.layer == 1).collect();
        let total: f32 = first.iter().map(|c| c.path.len() as f32 * 2.0).sum();
        // A 100-px square ruled every `spacing` holds about 100·100/spacing of line; a tangle holds far more.
        let even = 100.0 * 100.0 / plate.spacing;
        assert!(total < even * 1.5, "layer one is evenly ruled: {total} px of line against {even}");
        assert!(total > even * 0.5, "and it covers the mass: {total} px of line against {even}");
    }

    #[test]
    fn a_small_budget_opens_the_plate() {
        let img = two_masses(200, 200);
        let (full, tight) = (plan(&img, 0.0, 50_000, 7), plan(&img, 0.0, 60, 7));
        assert!(tight.cuts.len() <= 60, "the budget holds ({})", tight.cuts.len());
        assert!(tight.spacing > full.spacing, "by widening the spacing");
    }

    #[test]
    fn the_same_seed_cuts_the_same_plate() {
        let img = two_masses(120, 90);
        let (a, b) = (plan(&img, 0.5, 20_000, 3), plan(&img, 0.5, 20_000, 3));
        assert_eq!(a.cuts.len(), b.cuts.len());
        assert!(a.cuts.iter().zip(&b.cuts).all(|(x, y)| x.path == y.path && x.w0 == y.w0 && x.w1 == y.w1));
    }

    /// A plate for the eye: `PLAKAT_ENGRAVE_SRC=in.png PLAKAT_ENGRAVE_OUT=out.png cargo test --lib -- --ignored a_plate_for_the_eye`.
    #[test]
    #[ignore]
    fn a_plate_for_the_eye() {
        let (Ok(src), Ok(out)) = (std::env::var("PLAKAT_ENGRAVE_SRC"), std::env::var("PLAKAT_ENGRAVE_OUT")) else { return };
        let img = image::open(src).unwrap().to_rgb8();
        let plate = plan(&img, 0.55, 360_000, 42);
        let (w, h) = (img.width() as i64, img.height() as i64);
        let mut sheet = image::GrayImage::from_pixel(w as u32, h as u32, image::Luma([250]));
        // The stroke rasteriser's footprint for a flat-ended one-point brush: two lanes a pixel across the width.
        for cut in &plate.cuts {
            let mut pts = vec![cut.path[0]];
            for seg in cut.path.windows(2) {
                let (dx, dy) = (seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]);
                let steps = (dx * dx + dy * dy).sqrt().ceil().max(1.0) as usize;
                for s in 1..=steps {
                    pts.push([seg[0][0] + dx * s as f32 / steps as f32, seg[0][1] + dy * s as f32 / steps as f32]);
                }
            }
            let lanes = ((cut.w0.max(cut.w1).max(1.0) * 2.0).ceil() as usize).max(1);
            for i in 0..pts.len() {
                let q = if i + 1 < pts.len() { pts[i + 1] } else { pts[i - 1] };
                let (dx, dy) = (q[0] - pts[i][0], q[1] - pts[i][1]);
                let len = (dx * dx + dy * dy).sqrt().max(1e-4);
                let width = cut.w0 + (cut.w1 - cut.w0) * i as f32 / (pts.len() - 1) as f32;
                for lane in 0..lanes {
                    let off = ((lane as f32 + 0.5) / lanes as f32 - 0.5) * width;
                    let (px, py) = ((pts[i][0] - dy / len * off).round() as i64, (pts[i][1] + dx / len * off).round() as i64);
                    if px >= 0 && py >= 0 && px < w && py < h {
                        sheet.put_pixel(px as u32, py as u32, image::Luma([25]));
                    }
                }
            }
        }
        sheet.save(out).unwrap();
        eprintln!("{} cuts at spacing {:.1}", plate.cuts.len(), plate.spacing);
    }

    #[test]
    fn blank_paper_is_left_alone() {
        let plate = plan(&RgbImage::from_pixel(64, 64, image::Rgb([230, 228, 220])), 0.6, 5_000, 7);
        assert!(plate.cuts.is_empty(), "nothing to cut ({})", plate.cuts.len());
    }
}
