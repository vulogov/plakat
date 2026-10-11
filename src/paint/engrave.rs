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

/// The picture's relief, when it is known: a depth map at its own resolution, row-major, `z` in `[0,1]`, larger =
/// nearer (the Depth-Anything / ControlNet-Depth convention). It is an INPUT like the picture — the plate is
/// still nothing but geometry computed from the two — and it tells the burin what the picture's edges cannot:
/// which way a surface turns.
#[derive(Clone, Copy)]
pub struct Relief<'a> {
    pub w: usize,
    pub h: usize,
    pub z: &'a [f32],
}

/// The picture's surface normals, when they are known: unit vectors at their own resolution, row-major, in IMAGE
/// space — `x` right, `y` down, `z` toward the viewer. Like the relief they are an input, and a better one for
/// the line's direction: which way a surface faces is what a normal says outright and a depth map only implies.
#[derive(Clone, Copy)]
pub struct Normals<'a> {
    pub w: usize,
    pub h: usize,
    pub n: &'a [[f32; 3]],
}

/// What the picture's things are made of, where it was said: a kind per pixel at the map's own resolution, 0 for
/// nothing said. A burin cuts fur and polished metal differently from a plain surface.
#[derive(Clone, Copy)]
pub struct Materials<'a> {
    pub w: usize,
    pub h: usize,
    pub kind: &'a [u8],
}

/// FUR (and hair, feathers): the line runs the way the coat grows, and its crossings lie almost along it.
pub const FUR: u8 = 1;
/// GLASS and METAL: polished, so the lights are open paper and only the darks are cut.
pub const GLASS: u8 = 2;
pub const METAL: u8 = 3;

/// A planned plate.
pub struct Plate {
    pub cuts: Vec<Cut>,
    /// The line spacing the plate ended on (it widens when the cuts would not fit the budget).
    pub spacing: f32,
}

/// The paper each hatch layer may ink at full weight in the deepest shadow. Every layer stays open — a line is
/// never wider than two fifths of its spacing — and there are three of them, a third of a turn apart, so the
/// deepest shadow is a net of distinct lines with a triangle of paper in every mesh. A fourth layer would cut
/// through the meshes and close them: that is ink pooling, not engraving.
const LAYER_COVER: [f32; 3] = [0.42, 0.42, 0.40];
/// The weight at which a layer hands over to the next crossing. It is less than the full weight: a line goes on
/// swelling under its crossings all the way into the deepest shadow, so its weight is never flat.
const HAND_OVER: f32 = 0.27;
/// How inked the subject's middle value is cut: an engraving is mostly paper.
const MIDDLE_TONE: f32 = 0.2;
/// How much a surface is shaded against its surroundings.
const RELIEF: f32 = 0.4;
/// The deepest tone the plate cuts: a burin never fills a black, paper still shows between its lines.
const DEEPEST: f32 = 0.78;
/// How far below a hair's worth of tone the first layer still cuts: a light tone is a broken line — flicks,
/// and below `DOTS` of a hair, dots, so a form passes into the light through stipple, not over an edge.
const FLICK: f32 = 0.2;
const DOTS: f32 = 0.5;
/// A highlight is clean paper, not a faint tone: a spot among the lightest `LIT` of the subject that is also
/// lighter than its surroundings by `GLINT` is not cut at all. (A broad light — a lit cheek, a white dress — is
/// not a highlight: it is modelled, lightly.)
const LIT: f32 = 0.3;
const GLINT: f32 = 0.16;
/// AERIAL PERSPECTIVE, when the relief is known: how much lighter the farthest part of the subject is cut and
/// how much finer its contours are drawn (0 = the nearest part of the subject, 1 = the farthest). The lighter
/// tone is what thins the far plane's lines and drops its crossings — continuously, so no seam shows where a
/// layer ends. The fade is kept gentle: a dark far wall must still read darker than a near half-tone.
const FAR_LIGHTER: f32 = 0.3;
const FAR_FINER: f32 = 0.3;
/// How far below a hair's worth the first crossing still cuts, as dots: the edge of a shadow is stippled before
/// it is hatched. Only there — stipple everywhere is dust, not form.
const STIPPLE: f32 = 0.6;
/// How much a line's weight wavers along it with the pressure of the hand.
const HAND: f32 = 0.14;
/// How much of the plate's tone is the ORDER of the subject's values rather than the values themselves.
const BY_RANK: f32 = 0.45;
/// Each layer's angle off the form's direction: along it, then crossing it at a third of a turn either way, so
/// two layers leave lozenges and three leave triangles.
const LAYER_ANGLE: [f32; 3] = [0.0, 1.0472, 2.0944];
/// ... and in fur: hairs lie beside one another, so the crossings only lean a little off the coat's direction.
const FUR_ANGLE: [f32; 3] = [0.0, 0.3, -0.3];
/// How much of a polished surface's light tone is left to the paper: its middle values open, its darks stay.
const POLISH: f32 = 0.65;
/// The HALF-TONE band at a shadow's edge: a tone is not let fall below this share of the tone around it (read
/// `long / PENUMBRA_REACH` wide), so between a crossed shadow and the stippled light there is a breadth of
/// single hatch. Only where the surface carries some tone already — bare paper stays bare.
const PENUMBRA: f32 = 0.55;
const PENUMBRA_REACH: f32 = 70.0;
/// The TOE of the tone curve: the lightest tones fall off no faster than this share of the value — a line thins
/// and then breaks into ever sparser dots before the paper is reached, instead of stopping at an edge.
const TOE: f32 = 0.3;
/// How far a hatch line may turn in all over its length: past a right angle the burin lifts, so no line closes
/// into a ring or a whorl.
const SWEEP: f32 = 1.6;
/// How far each hair of fur may stray from the coat's direction (the full spread, in radians): hairs lie over
/// one another, not in ranks.
const FUR_SPREAD: f32 = 0.5;
/// Reading a normal map. The grain it is smoothed past, as a share of the sheet's long side (one part in ...);
/// how far the normal must swing more one way than the other, per side of the sheet, before a surface is taken
/// to TURN (from ... to fully); the swing past which it is a crease or a step, not a surface; and how far a
/// steady surface must be TILTED from facing the viewer (the length of its normal's `x`,`y`) to rest the line
/// level with it.
const NORMAL_GRAIN: usize = 128;
const TURNS: (f32, f32) = (3.0, 8.0);
const CREASE: f32 = 40.0;
const TILTED: (f32, f32) = (0.35, 0.65);
/// How strictly a surface the normals call flat and facing the viewer is hatched straight.
const STRAIGHT: f32 = 0.9;
/// A hair of FUR is a short stroke: so many spacings long, each way from where it starts.
const HAIR_LENGTH: (f32, f32) = (2.5, 6.0);

/// The share of the paper layer `k` inks where the plate's tone is `t`: the layers fill in order, each taking
/// what the ones before it left, so `k` crossings together ink exactly `t`.
fn cover(t: f32, k: usize) -> f32 {
    let mut paper = 1.0f32;
    let deep = ((t - HAND_OVER) / (DEEPEST - HAND_OVER)).clamp(0.0, 1.0);
    for (j, &full) in LAYER_COVER.iter().enumerate() {
        let cap = if j + 1 < LAYER_COVER.len() { HAND_OVER + (full - HAND_OVER) * deep } else { full };
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
    /// `ground` (at the sheet's resolution, 0..1) is where a tone lies alone on open paper: a cast shadow, the
    /// earth under the subject. There the line rests level instead of on the diagonal.
    fn new(value: &[f32], w: usize, h: usize, ground: &[f32], fur: Option<&[f32]>, relief: Option<Relief>, normals: Option<Normals>) -> Flow {
        let long = w.max(h);
        let f = long.div_ceil(512).max(1);
        let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
        let mut small = vec![0f32; sw * sh];
        let mut level = vec![0f32; sw * sh];
        let mut count = vec![0f32; sw * sh];
        for y in 0..h {
            for x in 0..w {
                let i = (y / f) * sw + x / f;
                small[i] += value[y * w + x];
                level[i] += ground[y * w + x];
                count[i] += 1.0;
            }
        }
        for ((s, l), c) in small.iter_mut().zip(level.iter_mut()).zip(&count) {
            *s /= c.max(1.0);
            *l /= c.max(1.0);
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
        let (mut c2, mut s2, mut led) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
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
            led[i] = form;
            // At rest the line lies on the diagonal — or level, on open ground.
            let g = level[i].clamp(0.0, 1.0);
            let (rc, rs) = (g, g - 1.0);
            let rest = (rc * rc + rs * rs).sqrt().max(1e-6);
            c2[i] = form * fc + (1.0 - form) * rc / rest;
            s2[i] = form * fs + (1.0 - form) * rs / rest;
        }
        // The normals say which way a surface turns outright; the relief only implies it, so it is asked second.
        if let Some(normals) = normals.filter(|m| m.w >= 2 && m.h >= 2 && m.n.len() == m.w * m.h) {
            turn_with_the_normals(&mut c2, &mut s2, &led, sw, sh, normals, STRAIGHT);
        } else if let Some(relief) = relief.filter(|r| r.w >= 2 && r.h >= 2 && r.z.len() == r.w * r.h) {
            turn_with_the_form(&mut c2, &mut s2, &led, sw, sh, relief);
        }
        // FUR grows its own way, whatever the form under it does: there the line takes the direction the coat's
        // own strands run in, read at the scale of a lock, wherever the strands agree on one.
        if let Some(fur) = fur {
            let lock = ((long as f32 / 110.0 / f as f32).round() as usize).max(2);
            let (mut a, mut b, mut e) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            for i in 0..n {
                a[i] = gx[i] * gx[i] - gy[i] * gy[i];
                b[i] = 2.0 * gx[i] * gy[i];
                e[i] = gx[i] * gx[i] + gy[i] * gy[i];
            }
            let (a, b, e) = (blur(&a, sw, sh, lock, 2), blur(&b, sw, sh, lock, 2), blur(&e, sw, sh, lock, 2));
            for y in 0..sh {
                for x in 0..sw {
                    let i = y * sw + x;
                    let len = (a[i] * a[i] + b[i] * b[i]).sqrt();
                    let agree = ((len / (e[i] + 1e-12) - 0.15) / 0.3).clamp(0.0, 1.0);
                    let t = fur[(y * f).min(h - 1) * w + (x * f).min(w - 1)] * agree;
                    if len > 1e-12 {
                        c2[i] += (-a[i] / len - c2[i]) * t;
                        s2[i] += (-b[i] / len - s2[i]) * t;
                    }
                }
            }
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

/// Where the relief says a surface TURNS — the barrel of a cylinder, the roll of a tyre, the bulge of a cushion —
/// the cuts run round it, along the direction of its greatest curvature, as an engraver's line wraps a limb.
/// That direction is the principal axis of the depth's second derivative; the surface normals are the depth's
/// first derivative, so this is the direction in which the normal swings fastest. Where the surface is flat, or
/// at the step between two objects (a depth map's edge is a cliff, not a curve), the picture's own direction
/// stands.
///
/// And where a flat surface RECEDES — the ground under the subject above all — and no edge of the picture leads
/// the line, the cuts rest level with it, along the lines of equal depth, instead of on the diagonal: a cast
/// shadow is laid in strokes that lie on the ground.
fn turn_with_the_form(c2: &mut [f32], s2: &mut [f32], led: &[f32], sw: usize, sh: usize, relief: Relief) {
    let n = sw * sh;
    // The depth at the flow's resolution, smoothed past the estimator's grain and an 8-bit map's steps.
    let mut z = vec![0f32; n];
    for y in 0..sh {
        for x in 0..sw {
            let (fx, fy) = ((x as f32 + 0.5) / sw as f32 * relief.w as f32 - 0.5, (y as f32 + 0.5) / sh as f32 * relief.h as f32 - 0.5);
            let (fx, fy) = (fx.clamp(0.0, relief.w as f32 - 1.0), fy.clamp(0.0, relief.h as f32 - 1.0));
            let (x0, y0) = (fx as usize, fy as usize);
            let (x1, y1) = ((x0 + 1).min(relief.w - 1), (y0 + 1).min(relief.h - 1));
            let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
            let at = |x: usize, y: usize| relief.z[y * relief.w + x];
            z[y * sw + x] = (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty;
        }
    }
    let r = (sw.max(sh) / 100).max(2);
    let z = blur(&z, sw, sh, r, 3);
    let at = |x: usize, y: usize| z[y.min(sh - 1) * sw + x.min(sw - 1)];
    let (mut vx, mut vy, mut bend, mut slope) = (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let (mut lx, mut ly, mut sx, mut sy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let d = r.max(1);
    for y in 0..sh {
        for x in 0..sw {
            let (xm, xp, ym, yp) = (x.saturating_sub(d), x + d, y.saturating_sub(d), y + d);
            let zxx = at(xp, y) - 2.0 * at(x, y) + at(xm, y);
            let zyy = at(x, yp) - 2.0 * at(x, y) + at(x, ym);
            let zxy = (at(xp, yp) - at(xp, ym) - at(xm, yp) + at(xm, ym)) * 0.25;
            // The doubled angle of the axis of greatest |curvature|: the larger eigenvalue's when the trace is
            // positive, the other's (a quarter turn off, so the doubled angle negated) when it is not.
            let sign = if zxx + zyy >= 0.0 { 1.0 } else { -1.0 };
            let i = y * sw + x;
            vx[i] = sign * (zxx - zyy);
            vy[i] = sign * 2.0 * zxy;
            bend[i] = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
            let (gx, gy) = (at(xp, y) - at(xm, y), at(x, yp) - at(x, ym));
            slope[i] = (gx * gx + gy * gy).sqrt();
            // The doubled angle of the line of equal depth: the slope's own, turned a quarter.
            lx[i] = -(gx * gx - gy * gy);
            ly[i] = -2.0 * gx * gy;
            sx[i] = gx;
            sy[i] = gy;
        }
    }
    // Curvature and slope are judged on an absolute scale — depth is 0..1 across the picture, lengths are in
    // sheet sides — not against the picture's own average: a picture whose depth is all but flat (a head against
    // a wall) must not have the estimator's ripples promoted to form.
    let side = sw.max(sh) as f32 / d as f32;
    let typical = 50.0 / (side * side);
    let steady = 2.6 / side;
    // A cliff between two objects bends harder than any surface, and the smoothing spreads it into a slope on
    // either side that is no surface at all: the cliff and its flanks are left out.
    let wide = r * 2;
    let edge: Vec<f32> = slope.iter().map(|s| ((s * side - 10.0) / 8.0).clamp(0.0, 1.0)).collect();
    let edge = blur(&edge, sw, sh, wide, 2);
    for i in 0..n {
        let keep = 1.0 - (edge[i] * 2.0).clamp(0.0, 1.0);
        vx[i] *= keep;
        vy[i] *= keep;
        bend[i] *= keep;
        lx[i] *= keep;
        ly[i] *= keep;
    }
    let (vx, vy, bend) = (blur(&vx, sw, sh, wide, 3), blur(&vy, sw, sh, wide, 3), blur(&bend, sw, sh, wide, 3));
    // Only a BROAD plane rests the line: the slope is averaged over a far wider window than the curvature, and
    // must fall ONE way across it — the two flanks of a bar slope opposite ways and are no plane. The small
    // flats of a machine keep the engraver's diagonal.
    let broad = wide * 3;
    let (lx, ly) = (blur(&lx, sw, sh, broad, 3), blur(&ly, sw, sh, broad, 3));
    let (sx, sy, steep) = (blur(&sx, sw, sh, broad, 3), blur(&sy, sw, sh, broad, 3), blur(&slope, sw, sh, broad, 3));
    for i in 0..n {
        let len = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
        // How surely the surface turns one way here: its curvature agrees with its neighbours' and is not noise.
        let agree = ((len / bend[i].max(1e-9) - 0.5) / 0.3).clamp(0.0, 1.0);
        let firm = ((bend[i] / typical - 0.7) / 1.0).clamp(0.0, 1.0);
        let t = agree * firm;
        let t = t * t * (3.0 - 2.0 * t);
        if len > 1e-9 {
            c2[i] += (vx[i] / len - c2[i]) * t;
            s2[i] += (vy[i] / len - s2[i]) * t;
        }
        // The slope is squared in `lx`,`ly`, so their length is the squared slope where it runs one way.
        let level = (lx[i] * lx[i] + ly[i] * ly[i]).sqrt();
        let sloped = ((level.sqrt() / steady - 0.6) / 0.8).clamp(0.0, 1.0);
        // ... and only where the surface really is flat: a turning one has its own direction, however weak.
        let flat = 1.0 - ((bend[i] / typical - 0.3) / 0.7).clamp(0.0, 1.0);
        let one_way = (((sx[i] * sx[i] + sy[i] * sy[i]).sqrt() / steep[i].max(1e-12) - 0.45) / 0.25).clamp(0.0, 1.0);
        let u = 0.85 * sloped * one_way * flat * (1.0 - t) * (1.0 - led[i]);
        if level > 1e-12 {
            c2[i] += (lx[i] / level - c2[i]) * u;
            s2[i] += (ly[i] / level - s2[i]) * u;
        }
    }
}

/// Which way a mark should run by the surface normals alone, for a medium that is not an engraving: a field of
/// doubled angles on a coarse grid. A vector's direction is the mark's (round a turning form, level on a tilted
/// plane); its length, 0..1, is how surely the normals say so — nothing where a surface faces the viewer.
pub struct FormFlow {
    w: usize,
    h: usize,
    scale: (f32, f32),
    c2: Vec<f32>,
    s2: Vec<f32>,
}

impl FormFlow {
    /// For a sheet of `w × h`. `None` when the map is too small to read.
    pub fn new(normals: Normals, w: usize, h: usize) -> Option<FormFlow> {
        if normals.w < 2 || normals.h < 2 || normals.n.len() != normals.w * normals.h || w < 8 || h < 8 {
            return None;
        }
        let f = w.max(h).div_ceil(512).max(1);
        let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
        let (mut c2, mut s2) = (vec![0f32; sw * sh], vec![0f32; sw * sh]);
        // A brush is led only where the normals speak; a flat wall is left to the painter.
        turn_with_the_normals(&mut c2, &mut s2, &vec![0f32; sw * sh], sw, sh, normals, 0.0);
        Some(FormFlow { w: sw, h: sh, scale: (sw as f32 / w as f32, sh as f32 / h as f32), c2: blur(&c2, sw, sh, 2, 2), s2: blur(&s2, sw, sh, 2, 2) })
    }

    /// The mark's unit direction at a sheet position, and how surely (0..1).
    pub fn at(&self, x: f32, y: f32) -> ([f32; 2], f32) {
        let (fx, fy) = ((x * self.scale.0 - 0.5).clamp(0.0, self.w as f32 - 1.0), (y * self.scale.1 - 0.5).clamp(0.0, self.h as f32 - 1.0));
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let at = |v: &[f32]| (v[y0 * self.w + x0] * (1.0 - tx) + v[y0 * self.w + x1] * tx) * (1.0 - ty) + (v[y1 * self.w + x0] * (1.0 - tx) + v[y1 * self.w + x1] * tx) * ty;
        let (c, s) = (at(&self.c2), at(&self.s2));
        let (sn, cs) = (0.5 * s.atan2(c)).sin_cos();
        ([cs, sn], (c * c + s * s).sqrt().clamp(0.0, 1.0))
    }
}

/// The same two rules as [`turn_with_the_form`], read off the surface normals instead of a depth map.
///
/// A normal's `x`,`y` part is the surface's tilt, so how the tilt changes across the sheet is the surface's
/// curvature: where it changes much more one way than the other — a barrel, a limb, a tyre — the cuts run that
/// way, round the form. Where the tilt is steady and large — the ground, a wall seen aslant — and no edge of the
/// picture leads the line, the cuts lie across the tilt, level with the surface.
///
/// Everything is measured in the normal's own units (how far it swings over one side of the sheet), so no
/// threshold depends on the picture.
fn turn_with_the_normals(c2: &mut [f32], s2: &mut [f32], led: &[f32], sw: usize, sh: usize, normals: Normals, straight: f32) {
    let n = sw * sh;
    let (mut nx, mut ny) = (vec![0f32; n], vec![0f32; n]);
    for y in 0..sh {
        for x in 0..sw {
            let (fx, fy) = ((x as f32 + 0.5) / sw as f32 * normals.w as f32 - 0.5, (y as f32 + 0.5) / sh as f32 * normals.h as f32 - 0.5);
            let (fx, fy) = (fx.clamp(0.0, normals.w as f32 - 1.0), fy.clamp(0.0, normals.h as f32 - 1.0));
            let (x0, y0) = (fx as usize, fy as usize);
            let (x1, y1) = ((x0 + 1).min(normals.w - 1), (y0 + 1).min(normals.h - 1));
            let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
            let at = |x: usize, y: usize, j: usize| normals.n[y * normals.w + x][j];
            let mix = |j: usize| (at(x0, y0, j) * (1.0 - tx) + at(x1, y0, j) * tx) * (1.0 - ty) + (at(x0, y1, j) * (1.0 - tx) + at(x1, y1, j) * tx) * ty;
            nx[y * sw + x] = mix(0);
            ny[y * sw + x] = mix(1);
        }
    }
    let r = (sw.max(sh) / NORMAL_GRAIN).max(2);
    let (nx, ny) = (blur(&nx, sw, sh, r, 3), blur(&ny, sw, sh, r, 3));
    let at = |v: &[f32], x: usize, y: usize| v[y.min(sh - 1) * sw + x.min(sw - 1)];
    let (mut vx, mut vy, mut bend, mut swing) = (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let (mut lx, mut ly) = (vec![0f32; n], vec![0f32; n]);
    let d = r.max(1);
    // A difference over `2d` cells, as the normal's swing over one side of the sheet.
    let side = sw.max(sh) as f32 / (2 * d) as f32;
    for y in 0..sh {
        for x in 0..sw {
            let (xm, xp, ym, yp) = (x.saturating_sub(d), x + d, y.saturating_sub(d), y + d);
            let kxx = (at(&nx, xp, y) - at(&nx, xm, y)) * side;
            let kyy = (at(&ny, x, yp) - at(&ny, x, ym)) * side;
            let kxy = ((at(&nx, x, yp) - at(&nx, x, ym)) + (at(&ny, xp, y) - at(&ny, xm, y))) * 0.5 * side;
            let sign = if kxx + kyy >= 0.0 { 1.0 } else { -1.0 };
            let i = y * sw + x;
            vx[i] = sign * (kxx - kyy);
            vy[i] = sign * 2.0 * kxy;
            bend[i] = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
            swing[i] = (kxx * kxx + kyy * kyy + 2.0 * kxy * kxy).sqrt();
            // The doubled angle of the level line: the tilt's own, turned a quarter.
            lx[i] = -(nx[i] * nx[i] - ny[i] * ny[i]);
            ly[i] = -2.0 * nx[i] * ny[i];
        }
    }
    // The crease between two faces and the step between two objects swing the normal harder than any surface
    // a hatch could wrap: they and their flanks are left to the picture's edges.
    let wide = r * 2;
    let edge: Vec<f32> = swing.iter().map(|s| ((s - CREASE) / (CREASE * 0.75)).clamp(0.0, 1.0)).collect();
    let edge = blur(&edge, sw, sh, wide, 2);
    for i in 0..n {
        let keep = 1.0 - (edge[i] * 2.0).clamp(0.0, 1.0);
        vx[i] *= keep;
        vy[i] *= keep;
        bend[i] *= keep;
        lx[i] *= keep;
        ly[i] *= keep;
    }
    let (vx, vy, bend) = (blur(&vx, sw, sh, wide, 3), blur(&vy, sw, sh, wide, 3), blur(&bend, sw, sh, wide, 3));
    let broad = wide * 3;
    let (lx, ly) = (blur(&lx, sw, sh, broad, 3), blur(&ly, sw, sh, broad, 3));
    for i in 0..n {
        let len = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
        let agree = ((len / bend[i].max(1e-9) - 0.5) / 0.3).clamp(0.0, 1.0);
        let firm = ((bend[i] - TURNS.0) / (TURNS.1 - TURNS.0)).clamp(0.0, 1.0);
        let t = agree * firm;
        let t = t * t * (3.0 - 2.0 * t);
        if len > 1e-9 {
            c2[i] += (vx[i] / len - c2[i]) * t;
            s2[i] += (vy[i] / len - s2[i]) * t;
        }
        // The tilt is squared in `lx`,`ly`: their length is the squared tilt where it holds one way.
        let level = (lx[i] * lx[i] + ly[i] * ly[i]).sqrt();
        let tilted = ((level.sqrt() - TILTED.0) / (TILTED.1 - TILTED.0)).clamp(0.0, 1.0);
        let u = 0.85 * tilted * (1.0 - t) * (1.0 - led[i]);
        if level > 1e-12 {
            c2[i] += (lx[i] / level - c2[i]) * u;
            s2[i] += (ly[i] / level - s2[i]) * u;
        }
        // A surface that neither turns nor tilts — a wall behind a head, the flat face of a machine — is hatched
        // STRAIGHT, on the resting diagonal: whatever orientation the picture's values have there is the drift
        // of its light or the grain of its canvas, and following it bends the lines of a surface that is flat.
        let still = 1.0 - (bend[i] / TURNS.0).clamp(0.0, 1.0);
        let plain = straight * still * (1.0 - tilted) * (1.0 - t);
        c2[i] -= c2[i] * plain;
        s2[i] += (-1.0 - s2[i]) * plain;
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
    /// Where the surface is fur (0..1), if any was named.
    fur: Option<&'a [f32]>,
    /// Where it is polished (0..1): there the hand is steady.
    polished: Option<&'a [f32]>,
}

impl Field<'_> {
    fn width(&self, p: [f32; 2], layer: usize) -> f32 {
        cover(self.tone[(p[1] as usize).min(self.h - 1) * self.w + (p[0] as usize).min(self.w - 1)], layer) * self.spacing
    }
    fn open(&self, p: [f32; 2], layer: usize) -> bool {
        if p[0] < 0.0 || p[1] < 0.0 || p[0] >= self.w as f32 || p[1] >= self.h as f32 {
            return false;
        }
        // The first layer and the first crossing run on below a hair's worth of tone, where they are cut as
        // flicks and as dots (see `plan`).
        let least = [FLICK, STIPPLE, 1.0][layer] * self.hair;
        !self.wall[p[1] as usize * self.w + p[0] as usize] && self.width(p, layer) >= least
    }

    /// One streamline through `start`, both ways, stopping at the paper, at a contour, beside another line, or
    /// when the direction field folds on itself.
    fn trace(&self, start: [f32; 2], layer: usize, near: &Near) -> Vec<[f32; 2]> {
        // The layer's angle off the form — in fur, off the coat — where the line starts; it keeps it to its end.
        let fur = self.fur.map_or(0.0, |f| f[(start[1] as usize).min(self.h - 1) * self.w + (start[0] as usize).min(self.w - 1)]);
        let own = |salt: u64| jitter(salt ^ layer as u64, (start[0] as u64) << 20 | start[1] as u64);
        let turn = (LAYER_ANGLE[layer] + (FUR_ANGLE[layer] - LAYER_ANGLE[layer] + FUR_SPREAD * own(0xF1B)) * fur).sin_cos();
        // A coat is not combed in long lines: it is short hairs, each its own length, lying over one another.
        let hair = HAIR_LENGTH.0 + (HAIR_LENGTH.1 - HAIR_LENGTH.0) * (own(0xF0E) + 0.5);
        let max_steps = (self.spacing * if fur > 0.5 { hair } else { 90.0 }) as usize;
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
                if swept.abs() > SWEEP {
                    break;
                }
                p = [p[0] + d[0], p[1] + d[1]];
                if !self.open(p, layer) || near.any_within(p, self.spacing * 0.55) {
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
        let every = (self.spacing * 0.5).max(1.0) as usize;
        let mut next = 0usize;
        let grow = |from: [f32; 2], lines: &mut Vec<Vec<[f32; 2]>>, near: &mut Near, next: &mut usize| {
            if !self.open(from, layer) || near.any_within(from, self.spacing * 0.92) {
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
                        if !self.open(c, layer) || near.any_within(c, self.spacing * 0.92) {
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

/// Join the edge map's fragments into the lines they were: a chain that ends where another begins, running the
/// same way, is one line the edge detector dropped a few pixels of. A burin does not lift there.
fn join(chains: Vec<Vec<[f32; 2]>>, reach: f32) -> Vec<Vec<[f32; 2]>> {
    // The direction a chain leaves by at its end (`tail`) or arrives by at its start.
    let heading = |c: &[[f32; 2]], tail: bool| {
        let n = c.len();
        let k = 6.min(n - 1);
        let (a, b) = if tail { (c[n - 1 - k], c[n - 1]) } else { (c[0], c[k]) };
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt().max(1e-4);
        [dx / len, dy / len]
    };
    let mut chains: Vec<Option<Vec<[f32; 2]>>> = chains.into_iter().filter(|c| c.len() >= 2).map(Some).collect();
    let mut out = Vec::new();
    for i in 0..chains.len() {
        let Some(mut line) = chains[i].take() else { continue };
        // Grow from the tail, then turn the line round and grow from what was its head.
        for _ in 0..2 {
            loop {
                let (end, go) = (line[line.len() - 1], heading(&line, true));
                let mut best: Option<(usize, bool, f32)> = None;
                for (j, other) in chains.iter().enumerate() {
                    let Some(o) = other else { continue };
                    for flip in [false, true] {
                        let (start, dir) = if flip { (o[o.len() - 1], heading(o, true)) } else { (o[0], heading(o, false)) };
                        let dir = if flip { [-dir[0], -dir[1]] } else { dir };
                        let (gx, gy) = (start[0] - end[0], start[1] - end[1]);
                        let gap = (gx * gx + gy * gy).sqrt();
                        if gap > reach || go[0] * dir[0] + go[1] * dir[1] < 0.75 || (gap > 1.5 && (gx * go[0] + gy * go[1]) / gap < 0.7) {
                            continue;
                        }
                        if best.is_none_or(|b| gap < b.2) {
                            best = Some((j, flip, gap));
                        }
                    }
                }
                let Some((j, flip, _)) = best else { break };
                let mut o = chains[j].take().unwrap_or_default();
                if flip {
                    o.reverse();
                }
                line.extend(o);
            }
            line.reverse();
        }
        out.push(line);
    }
    out
}

/// Pull a traced line taut: an edge map's chain steps from pixel to pixel, a burin's line does not.
fn taut(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = pts.len();
    let mut cur = pts.to_vec();
    for _ in 0..2 {
        let src = cur.clone();
        for i in 1..n.saturating_sub(1) {
            let r = 3.min(i).min(n - 1 - i);
            let (mut x, mut y) = (0f32, 0f32);
            for p in &src[i - r..=i + r] {
                x += p[0];
                y += p[1];
            }
            cur[i] = [x / (2 * r + 1) as f32, y / (2 * r + 1) as f32];
        }
    }
    cur
}

/// Plan the plate for a picture. `contour` (0..1) is how much of the edge map is drawn; `budget` caps the
/// number of cuts — when the hatch would not fit, its spacing widens until it does.
pub fn plan(picture: &RgbImage, relief: Option<Relief>, normals: Option<Normals>, materials: Option<Materials>, contour: f32, budget: usize, seed: u64) -> Plate {
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
    // DISTANCE, when the relief is known: 0 at the nearest part of the subject, 1 at its farthest.
    let far: Option<Vec<f32>> = relief.filter(|r| r.w >= 2 && r.h >= 2 && r.z.len() == r.w * r.h).map(|r| {
        let z: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (((i % w) * r.w / w).min(r.w - 1), ((i / w) * r.h / h).min(r.h - 1));
                r.z[y * r.w + x]
            })
            .collect();
        let mut of_subject: Vec<f32> = z.iter().zip(&dark).step_by(11).filter(|(_, d)| **d > 0.15).map(|(z, _)| *z).collect();
        if of_subject.len() < 16 {
            return vec![0.0; w * h];
        }
        of_subject.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (lo, hi) = (of_subject[of_subject.len() * 5 / 100], of_subject[of_subject.len() * 95 / 100]);
        let far: Vec<f32> = z.iter().map(|z| ((hi - z) / (hi - lo).max(0.02)).clamp(0.0, 1.0)).collect();
        blur(&far, w, h, ((long / 250.0) as usize).max(1), 2)
    });
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
    // MATERIALS, where they were named: how far each point is fur, and how far polished.
    let made_of = |kinds: &[u8]| -> Option<Vec<f32>> {
        let m = materials.filter(|m| m.w >= 1 && m.h >= 1 && m.kind.len() == m.w * m.h)?;
        let is: Vec<f32> = (0..w * h).map(|i| if kinds.contains(&m.kind[((i / w) * m.h / h).min(m.h - 1) * m.w + ((i % w) * m.w / w).min(m.w - 1)]) { 1.0 } else { 0.0 }).collect();
        is.iter().any(|v| *v > 0.0).then(|| blur(&is, w, h, ((long / 400.0) as usize).max(1), 2))
    };
    let (fur, polished) = (made_of(&[FUR]), made_of(&[GLASS, METAL]));
    let key = (MIDDLE_TONE / DEEPEST).ln() / 0.5f32.ln();
    let tone: Vec<f32> = (0..w * h)
        .map(|i| {
            let d = modelled[i];
            let r = if d > 0.08 { rank[((d * 255.0) as usize).min(255)] } else { 0.0 };
            // A highlight is left clean. The fringe of a tone that dies away into the paper — a cast shadow's
            // edge — is no highlight (nothing around it is darker by much): it thins out into flicks.
            // On a polished surface a light need outshine its surroundings by half as much to be a glint.
            let shine = polished.as_ref().map_or(0.0, |p| p[i]);
            if r < LIT && around[i] - dark[i] > GLINT * (1.0 - 0.5 * shine) {
                return 0.0;
            }
            let lighter = 1.0 - FAR_LIGHTER * far.as_ref().map_or(0.0, |f| f[i]);
            let x = BY_RANK * r + (1.0 - BY_RANK) * d;
            let t = DEEPEST * x.powf(key).max(TOE * x);
            // ... and its light and middle tones open to the paper, while its darks stay: polish is contrast.
            lighter * t * (1.0 - POLISH * shine * (1.0 - t / DEEPEST))
        })
        .collect();
    // The half-tone band: beside a shadow, a tone already begun is carried at single-hatch weight.
    let beside = blur(&tone, w, h, ((long / PENUMBRA_REACH) as usize).max(2), 3);
    let tone: Vec<f32> = tone.iter().zip(&beside).map(|(&t, &b)| t.max((PENUMBRA * b).min(HAND_OVER) * (t / (0.3 * HAND_OVER)).clamp(0.0, 1.0))).collect();

    // CONTOURS: the edge map's chains, each a line whose weight follows the contrast it separates.
    let mut cuts: Vec<Cut> = Vec::new();
    let mut wall = vec![false; w * h];
    if contour > 0.0 {
        let edge = edge_map(&value, w, h, contour);
        let (_, _, mag) = sobel(&value, w, h);
        let chains = join(trace_chains(&edge, w, h, (long / 130.0).max(5.0) as usize), (long / 200.0).max(3.0));
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
            // A burin's groove is a V: the line enters as a needle, swells where the edge is firm, and leaves
            // as a needle.
            let finer = |p: &[f32; 2]| 1.0 - FAR_FINER * far.as_ref().map_or(0.0, |f| f[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)]);
            let mut widths: Vec<f32> = chain.iter().map(|p| finer(p) * base * (0.45 + 1.6 * (at(p) / strong).min(1.0).powf(1.2))).collect();
            swell(&mut widths, 10, base * 0.2, ((long / 90.0) as usize).min(chain.len() / 3).max(3));
            pieces(0, &taut(chain), &widths, &mut cuts);
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
    // A tone that lies alone on open paper — little around it is dark — is a shadow on the ground.
    let ground: Vec<f32> = around.iter().map(|a| ((0.42 - a) / 0.2).clamp(0.0, 1.0)).collect();
    let flow = Flow::new(&value, w, h, &ground, fur.as_deref(), relief, normals);
    let room = budget.saturating_sub(cuts.len());
    let mut spacing = (long / 300.0).max(3.0);
    let mut hatch: Vec<Cut> = Vec::new();
    for _ in 0..8 {
        hatch.clear();
        let field = Field { w, h, tone: &tone, wall: &wall, flow: &flow, spacing, hair: (spacing * 0.15).max(0.45), fur: fur.as_deref(), polished: polished.as_deref() };
        for layer in 0..LAYER_COVER.len() {
            for (li, line) in field.layer(layer, seed).into_iter().enumerate() {
                let raw: Vec<f32> = line.iter().map(|p| field.width(*p, layer)).collect();
                // Below a hair's worth of tone a line is not made thinner — it is BROKEN. The first layer breaks into
                // flicks whose length carries the tone; the first crossing enters as dots. Each line breaks at its
                // own place, so the marks do not fall into ranks.
                let (period, pitch) = (spacing * 3.0, spacing * 1.15);
                let dot = (field.hair * 1.8).max(2.0);
                let phase = (jitter(seed ^ 0xF11C ^ layer as u64, li as u64) + 0.5) * period;
                let ruled = |i: usize| {
                    let p = line[i];
                    field.polished.map_or(0.0, |m| m[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)])
                };
                let on = |i: usize| {
                    let part = raw[i] / field.hair;
                    // On polished metal and glass the line is never broken: it runs on as a hair, straight.
                    if part >= 1.0 || ruled(i) > 0.5 {
                        return true;
                    }
                    let at = i as f32 + phase;
                    if layer == 0 && part >= DOTS {
                        return at % period < period * part;
                    }
                    let nth = (at / pitch) as u64;
                    let often = if layer == 0 { 0.25 + 0.75 * (part - FLICK) / (DOTS - FLICK) } else { (part - STIPPLE) / (1.0 - STIPPLE) };
                    at % pitch < dot && jitter(seed ^ 0xD07 ^ layer as u64, (li as u64) << 20 | nth) + 0.5 < often
                };
                // The hand: a line's weight breathes along it.
                let hand = |i: usize| {
                    let at = (i as f32 + phase) / (spacing * 7.0);
                    let (k, f) = (at as u64, at.fract());
                    let f = f * f * (3.0 - 2.0 * f);
                    let knot = |k: u64| jitter(seed ^ 0x4A2D ^ layer as u64, (li as u64) << 20 | k);
                    // On polished metal and glass the line is ruled: its weight does not waver.
                    1.0 + 2.0 * HAND * (1.0 - ruled(i)) * (knot(k) * (1.0 - f) + knot(k + 1) * f)
                };
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
                    if b - a >= 4 {
                        let mut widths: Vec<f32> = (a..=b).map(|i| (raw[i] * hand(i)).max(field.hair)).collect();
                        swell(&mut widths, 4, field.hair * 0.5, ((spacing * 6.0) as usize).min((b - a) / 2));
                        let thin: Vec<usize> = (0..=b - a).step_by(2).chain(std::iter::once(b - a).filter(|l| l % 2 == 1)).collect();
                        let pts: Vec<[f32; 2]> = thin.iter().map(|&i| line[a + i]).collect();
                        let ws: Vec<f32> = thin.iter().map(|&i| widths[i]).collect();
                        pieces(layer + 1, &pts, &ws, &mut hatch);
                    } else if b > a {
                        let width = field.hair * 1.5;
                        hatch.push(Cut { layer: layer + 1, path: vec![line[a], line[b]], w0: width, w1: width });
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
        for t in [0.1f32, 0.3, 0.5, 0.7] {
            let paper: f32 = (0..LAYER_COVER.len()).map(|k| 1.0 - cover(t, k)).product();
            assert!((1.0 - paper - t).abs() < 1e-4, "tone {t}: the layers ink {}", 1.0 - paper);
        }
        assert_eq!(cover(0.2, 1), 0.0, "a light tone is one layer, not a crossing");
        assert!(cover(0.6, 1) > 0.0, "a dark one is crossed");
    }

    #[test]
    fn a_dark_mass_is_cross_hatched_and_the_ground_stays_paper() {
        let plate = plan(&two_masses(160, 160), None, None, None, 0.6, 50_000, 7);
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
        let plate = plan(&two_masses(200, 200), None, None, None, 0.0, 50_000, 7);
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
        let (full, tight) = (plan(&img, None, None, None, 0.0, 50_000, 7), plan(&img, None, None, None, 0.0, 60, 7));
        assert!(tight.cuts.len() <= 60, "the budget holds ({})", tight.cuts.len());
        assert!(tight.spacing > full.spacing, "by widening the spacing");
    }

    #[test]
    fn the_same_seed_cuts_the_same_plate() {
        let img = two_masses(120, 90);
        let (a, b) = (plan(&img, None, None, None, 0.5, 20_000, 3), plan(&img, None, None, None, 0.5, 20_000, 3));
        assert_eq!(a.cuts.len(), b.cuts.len());
        assert!(a.cuts.iter().zip(&b.cuts).all(|(x, y)| x.path == y.path && x.w0 == y.w0 && x.w1 == y.w1));
    }

    /// A plate for the eye: `PLAKAT_ENGRAVE_SRC=in.png PLAKAT_ENGRAVE_OUT=out.png cargo test --lib -- --ignored a_plate_for_the_eye`.
    #[test]
    #[ignore]
    fn a_plate_for_the_eye() {
        let (Ok(src), Ok(out)) = (std::env::var("PLAKAT_ENGRAVE_SRC"), std::env::var("PLAKAT_ENGRAVE_OUT")) else { return };
        let img = image::open(src).unwrap().to_rgb8();
        let depth = std::env::var("PLAKAT_ENGRAVE_DEPTH").ok().map(|p| image::open(p).unwrap().to_luma8());
        let z: Option<Vec<f32>> = depth.as_ref().map(|d| d.pixels().map(|p| p.0[0] as f32 / 255.0).collect());
        let relief = depth.as_ref().zip(z.as_ref()).map(|(d, z)| Relief { w: d.width() as usize, h: d.height() as usize, z });
        let map = std::env::var("PLAKAT_ENGRAVE_NORMALS").ok().map(|p| image::open(p).unwrap().to_rgb8());
        let n: Option<Vec<[f32; 3]>> = map.as_ref().map(crate::pipelines::normals::from_png);
        let normals = map.as_ref().zip(n.as_ref()).map(|(m, n)| Normals { w: m.width() as usize, h: m.height() as usize, n });
        let made = std::env::var("PLAKAT_ENGRAVE_MATERIALS").ok().map(|p| image::open(p).unwrap().to_rgb8());
        let kind: Option<Vec<u8>> = made.as_ref().map(crate::pipelines::materials::from_png);
        let materials = made.as_ref().zip(kind.as_ref()).map(|(m, kind)| Materials { w: m.width() as usize, h: m.height() as usize, kind });
        let plate = plan(&img, relief, normals, materials, 0.55, 360_000, 42);
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
    fn a_relief_turns_the_cuts_round_the_form() {
        // A dark mass that the depth map says holds a rounded bar lying on its side: its surface turns from top
        // to bottom, so the first layer's cuts run up and down round it. Without the relief they rest on the diagonal.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let z: Vec<f32> = (0..w * h).map(|i| 0.3 + 0.5 * (1.0 - (((i / w) as f32 - 100.0) / 16.0).powi(2)).max(0.0).sqrt()).collect();
        let upright = |plate: &Plate| {
            let (mut dx, mut dy) = (0f32, 0f32);
            for c in plate.cuts.iter().filter(|c| c.layer == 1) {
                for s in c.path.windows(2).filter(|s| s[0][0] > 70.0 && s[0][0] < 130.0 && s[0][1] > 96.0 && s[0][1] < 104.0) {
                    dx += (s[1][0] - s[0][0]).abs();
                    dy += (s[1][1] - s[0][1]).abs();
                }
            }
            dy / (dx + dy).max(1e-6)
        };
        let (flat, round) = (plan(&img, None, None, None, 0.0, 50_000, 7), plan(&img, Some(Relief { w, h, z: &z }), None, None, 0.0, 50_000, 7));
        assert!(upright(&flat) < 0.56, "without the relief the cuts rest on the diagonal ({})", upright(&flat));
        assert!(upright(&round) > 0.68, "round the cylinder they turn upright ({})", upright(&round));
    }

    /// How upright layer `layer`'s cuts run inside a box of the sheet: 0 = level, 1 = up and down.
    fn upright_in(plate: &Plate, x: (f32, f32), y: (f32, f32)) -> f32 {
        let (mut dx, mut dy) = (0f32, 0f32);
        for c in plate.cuts.iter().filter(|c| c.layer == 1) {
            for s in c.path.windows(2).filter(|s| s[0][0] > x.0 && s[0][0] < x.1 && s[0][1] > y.0 && s[0][1] < y.1) {
                dx += (s[1][0] - s[0][0]).abs();
                dy += (s[1][1] - s[0][1]).abs();
            }
        }
        dy / (dx + dy).max(1e-6)
    }

    #[test]
    fn normals_turn_the_cuts_round_the_form() {
        // The same bar lying on its side, said by its normals: they swing from up to down across it.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let n: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let t = (((i / w) as f32 - 100.0) / 24.0).clamp(-0.95, 0.95);
                [0.0, t, (1.0 - t * t).sqrt()]
            })
            .collect();
        let (flat, round) = (plan(&img, None, None, None, 0.0, 50_000, 7), plan(&img, None, Some(Normals { w, h, n: &n }), None, 0.0, 50_000, 7));
        let band = |p: &Plate| upright_in(p, (70.0, 130.0), (90.0, 110.0));
        assert!(band(&flat) < 0.56, "without the normals the cuts rest on the diagonal ({})", band(&flat));
        assert!(band(&round) > 0.8, "round the bar they turn upright ({})", band(&round));
    }

    #[test]
    fn normals_lay_the_cuts_level_on_the_ground() {
        // A plane facing up and toward the viewer — the ground: the cuts lie level on it, not on the diagonal.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let n = vec![[0.0f32, -0.8, 0.6]; w * h];
        let plate = plan(&img, None, Some(Normals { w, h, n: &n }), None, 0.0, 50_000, 7);
        let level = upright_in(&plate, (70.0, 130.0), (70.0, 130.0));
        assert!(level < 0.2, "on the ground the cuts lie level ({level})");
        // A plane facing the viewer says nothing: the diagonal stands.
        let facing = vec![[0.0f32, 0.0, 1.0]; w * h];
        let plate = plan(&img, None, Some(Normals { w, h, n: &facing }), None, 0.0, 50_000, 7);
        let rest = upright_in(&plate, (70.0, 130.0), (70.0, 130.0));
        assert!(rest > 0.4 && rest < 0.6, "a wall facing the viewer keeps the diagonal ({rest})");
    }

    #[test]
    fn the_form_flow_is_sure_only_where_the_normals_speak() {
        let (w, h) = (200usize, 200usize);
        let ground = vec![[0.0f32, -0.8, 0.6]; w * h];
        let (dir, sure) = FormFlow::new(Normals { w, h, n: &ground }, w, h).unwrap().at(100.0, 100.0);
        assert!(sure > 0.7 && dir[1].abs() < 0.1, "level on the ground: {dir:?} at {sure}");
        let facing = vec![[0.0f32, 0.0, 1.0]; w * h];
        let (_, sure) = FormFlow::new(Normals { w, h, n: &facing }, w, h).unwrap().at(100.0, 100.0);
        assert!(sure < 0.01, "a surface facing the viewer leads nothing ({sure})");
    }

    #[test]
    fn fur_is_cut_along_the_coat_and_its_crossings_lean() {
        // A dark coat of level strands. Plain, the first crossing lies a third of a turn off the first layer; as
        // fur, it only leans off it.
        let (w, h) = (200usize, 200usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if x < 30 || x > 170 || y < 30 || y > 170 { 235 } else if y % 10 < 5 { 15 } else { 70 };
            image::Rgb([v, v, v])
        });
        let kind = vec![FUR; w * h];
        // The mean direction of a layer inside the coat, as an angle in [0, π).
        let lie = |plate: &Plate, layer: usize| {
            let (mut c, mut s) = (0f32, 0f32);
            for cut in plate.cuts.iter().filter(|c| c.layer == layer) {
                for seg in cut.path.windows(2).filter(|s| s[0][0] > 60.0 && s[0][0] < 140.0 && s[0][1] > 60.0 && s[0][1] < 140.0) {
                    let a = 2.0 * (seg[1][1] - seg[0][1]).atan2(seg[1][0] - seg[0][0]);
                    c += a.cos();
                    s += a.sin();
                }
            }
            0.5 * s.atan2(c)
        };
        let apart = |plate: &Plate| {
            let d = (lie(plate, 1) - lie(plate, 2)).abs() % std::f32::consts::PI;
            d.min(std::f32::consts::PI - d)
        };
        let (plain, furred) = (plan(&img, None, None, None, 0.0, 80_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 0.0, 80_000, 7));
        assert!(apart(&plain) > 0.85, "a plain surface is crossed at a third of a turn ({})", apart(&plain));
        assert!(apart(&furred) < 0.45, "fur's crossing only leans off the coat ({})", apart(&furred));
        assert!(lie(&furred, 1).abs() < 0.2, "and the coat is cut along its strands ({})", lie(&furred, 1));
    }

    #[test]
    fn a_polished_surface_opens_its_lights() {
        let (w, h) = (200usize, 200usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if y < 30 || y > 170 { 235 } else if x < 140 { 110 } else { 20 };
            image::Rgb([v, v, v])
        });
        let kind = vec![METAL; w * h];
        // The ink laid between two verticals: every point of every cut, by the cut's width.
        let ink = |plate: &Plate, x: (f32, f32)| plate.cuts.iter().map(|c| c.path.iter().filter(|p| p[0] > x.0 && p[0] < x.1).count() as f32 * (c.w0 + c.w1) * 0.5).sum::<f32>();
        let (plain, polished) = (plan(&img, None, None, None, 0.0, 80_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 0.0, 80_000, 7));
        let (mid, deep) = (ink(&polished, (10.0, 80.0)) / ink(&plain, (10.0, 80.0)), ink(&polished, (155.0, 190.0)) / ink(&plain, (155.0, 190.0)));
        assert!(ink(&plain, (10.0, 80.0)) > 100.0, "the plain middle tone is cut at all");
        assert!(mid < 0.7, "the middle tone opens ({mid})");
        assert!(deep > 0.85, "the dark stays ({deep})");
    }

    #[test]
    fn a_flat_wall_is_hatched_straight_whatever_is_painted_on_it() {
        // A dark wall with upright stripes painted on it. By the picture alone the cuts follow the stripes; told
        // by the normals that the wall is flat and faces the viewer, they keep the diagonal.
        let (w, h) = (200usize, 200usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if x < 30 || x > 170 || y < 30 || y > 170 { 235 } else if x % 16 < 8 { 15 } else { 80 };
            image::Rgb([v, v, v])
        });
        let facing = vec![[0.0f32, 0.0, 1.0]; w * h];
        let (led, flat) = (plan(&img, None, None, None, 0.0, 80_000, 7), plan(&img, None, Some(Normals { w, h, n: &facing }), None, 0.0, 80_000, 7));
        let (by_picture, by_normals) = (upright_in(&led, (60.0, 140.0), (60.0, 140.0)), upright_in(&flat, (60.0, 140.0), (60.0, 140.0)));
        assert!(by_picture > 0.62, "alone, the picture's stripes lead ({by_picture})");
        assert!(by_normals > 0.4 && by_normals < 0.6, "a flat wall keeps the diagonal ({by_normals})");
    }

    #[test]
    fn fur_is_cut_in_short_hairs() {
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let kind = vec![FUR; w * h];
        let longest = |plate: &Plate| {
            // Pieces of one line share end points; a line's length is its pieces' together.
            let mut runs: Vec<f32> = Vec::new();
            let mut last: Option<[f32; 2]> = None;
            for c in plate.cuts.iter().filter(|c| c.layer == 1) {
                let len: f32 = c.path.windows(2).map(|s| ((s[1][0] - s[0][0]).powi(2) + (s[1][1] - s[0][1]).powi(2)).sqrt()).sum();
                if last == Some(c.path[0]) && !runs.is_empty() {
                    *runs.last_mut().unwrap() += len;
                } else {
                    runs.push(len);
                }
                last = c.path.last().copied();
            }
            runs.into_iter().fold(0f32, f32::max)
        };
        let (plain, furred) = (plan(&img, None, None, None, 0.0, 80_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 0.0, 80_000, 7));
        assert!(longest(&plain) > 60.0, "a plain mass is cut in long lines ({})", longest(&plain));
        assert!(longest(&furred) < 2.0 * HAIR_LENGTH.1 * furred.spacing + 4.0, "fur in hairs ({} at spacing {})", longest(&furred), furred.spacing);
    }

    #[test]
    fn a_light_passes_into_the_paper_through_dots() {
        // A ramp from the dark to the paper: its lightest cut fifth is dots, not an edge.
        let (w, h) = (240usize, 120usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, _| {
            let v = (40.0 + 195.0 * x as f32 / w as f32) as u8;
            image::Rgb([v, v, v])
        });
        let plate = plan(&img, None, None, None, 0.0, 80_000, 7);
        let last = plate.cuts.iter().flat_map(|c| c.path.iter().map(|p| p[0])).fold(0f32, f32::max);
        let dots = plate.cuts.iter().filter(|c| c.path[0][0] > last - 25.0 && c.path.len() <= 3).count();
        let lines = plate.cuts.iter().filter(|c| c.path[0][0] > last - 25.0 && c.path.len() > 3).count();
        assert!(last > 170.0, "the ramp is cut well into its lights ({last})");
        assert!(dots > lines, "its last band is dots ({dots} dots, {lines} lines)");
    }

    #[test]
    fn no_hatch_line_closes_into_a_ring() {
        // A dark disc: the cuts run round it, but each stops within a right angle's turn.
        let (w, h) = (200usize, 200usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let r = ((x as f32 - 100.0).powi(2) + (y as f32 - 100.0).powi(2)).sqrt();
            let v = if r < 60.0 { (30.0 + r * 1.5) as u8 } else { 235 };
            image::Rgb([v, v, v])
        });
        let z: Vec<f32> = (0..w * h).map(|i| {
            let r = (((i % w) as f32 - 100.0).powi(2) + ((i / w) as f32 - 100.0).powi(2)).sqrt();
            0.3 + 0.6 * (1.0 - (r / 60.0).powi(2)).max(0.0).sqrt()
        }).collect();
        let plate = plan(&img, Some(Relief { w, h, z: &z }), None, None, 0.0, 80_000, 7);
        let sweep = |c: &Cut| {
            let mut s = 0f32;
            for t in c.path.windows(3) {
                let (ax, ay, bx, by) = (t[1][0] - t[0][0], t[1][1] - t[0][1], t[2][0] - t[1][0], t[2][1] - t[1][1]);
                s += (ax * by - ay * bx).atan2(ax * bx + ay * by);
            }
            s.abs()
        };
        let most = plate.cuts.iter().filter(|c| c.layer > 0).map(sweep).fold(0f32, f32::max);
        assert!(most < SWEEP + 0.6, "a hatch line turns at most about a right angle ({most})");
    }

    #[test]
    fn a_highlight_is_clean_paper() {
        // A light strip that outshines the dark around it is a highlight: the plate leaves it uncut.
        let img = RgbImage::from_fn(200, 200, |x, y| if x > 40 && x < 160 && y > 40 && y < 160 { let v = if x < 52 { 205 } else { 50 }; image::Rgb([v, v, v]) } else { image::Rgb([235, 232, 226]) });
        let plate = plan(&img, None, None, None, 0.0, 50_000, 7);
        let lit = plate.cuts.iter().filter(|c| c.layer > 0).flat_map(|c| c.path.iter()).filter(|p| p[0] > 44.0 && p[0] < 49.0).count();
        assert_eq!(lit, 0, "nothing is cut in the highlight");
        assert!(plate.cuts.iter().any(|c| c.layer > 0), "the rest is engraved");
    }

    #[test]
    fn blank_paper_is_left_alone() {
        let plate = plan(&RgbImage::from_pixel(64, 64, image::Rgb([230, 228, 220])), None, None, None, 0.6, 5_000, 7);
        assert!(plate.cuts.is_empty(), "nothing to cut ({})", plate.cuts.len());
    }
}
