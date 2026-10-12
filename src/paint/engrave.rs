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
/// SKIN (a face, a hand): the hatch is held at rest — the diagonal — and does not follow the form, which on a
/// face reads as a contour map round the nose and cheeks. The contours still bend; only the tone's lines are ruled.
pub const SKIN: u8 = 4;

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
/// ... except a face, which is modelled fuller than its ground, as a portrait engraver's is: fine lines over
/// the whole lit side, open only at the highlights. On skin the middle value is cut this far inked.
const SKIN_TONE: f32 = 0.33;
/// How much a surface is shaded against its surroundings.
const RELIEF: f32 = 0.4;
/// The deepest tone the plate cuts: a burin never fills a black, paper still shows between its lines.
const DEEPEST: f32 = 0.78;
/// Below the sparsest hairs (see `SPARSE`) the first layer still cuts, as dots that go on thinning, evenly, all
/// the way to the paper: there is no tone so light that it has no dot at all, so the light rolls off through an
/// ever sparser stipple instead of stopping at an edge — a lit cheek, a white sleeve, keep their form. `BARE`
/// is the share of a hair below which a tone is not even a dot: a little above nothing, so the faint unevenness
/// of a sheet of old paper, a whisker under the white point, is not dusted over.
const BARE: f32 = 0.03;
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
/// How far below a hair's worth a crossing still cuts, as dots: the edge of a shadow is stippled before it is
/// hatched, and the deepest crossing enters the same way — a layer that began at a hair's full width would
/// draw its own edge, a seam along the tone it starts at. Only there — stipple everywhere is dust, not form.
const STIPPLE: f32 = 0.6;
/// A LIGHT tone is not a thinner line — a burin cuts nothing finer than a hair — nor, first, a broken one: it
/// is the same hair, further apart. Below a hair's worth of tone the lines of a layer spread out, up to this
/// many spacings, dropping out one at a time as the tone lifts, the way an engraver lets a form go into the
/// light; only past that do the hairs break into flicks and dots. (A lit cheek cut as flicks and dots at the
/// full spacing read as a burn, not a light.)
const SPARSE: f32 = 4.0;
/// THE PAPER, when no white point is named: the picture's brightest broad flat value — the sheet of a drawing,
/// a sky, a wall — is a band of values a fiftieth of the range wide holding this share of the picture.
const PAPER_SHARE: f32 = 0.03;
/// How much a line's weight wavers along it with the pressure of the hand.
const HAND: f32 = 0.14;
/// How much of the plate's tone is the ORDER of the subject's values rather than the values themselves.
const BY_RANK: f32 = 0.45;
/// ... counted among the values that MODEL something: a value weighs as much as it differs from its
/// surroundings (fully at this much), so a mass of one value — a dark ground, a wall — counts for little and
/// is one tone, not a ramp. Ranked by count alone, a ground darkening by a twentieth toward the edge of the
/// sheet was cut from open hatch to the deepest black, and the crossing that entered along the way drew a
/// seam down the wall; and a face, a tenth of a dark picture, was ranked into the paper.
const RANK_RELIEF: f32 = 0.1;
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
const FUR_SPREAD: f32 = 0.7;
/// A contour that runs through fur is not one line but a run of hairs: pieces so many spacings long (least,
/// most), with a gap of about one spacing between them.
const FUR_PIECE: (f32, f32) = (3.0, 8.0);
/// An estimated normal map fades out across a silhouette: beside a head the background's normals tilt away
/// from it and die off over a short distance — a HALO that is no surface, and whose "curvature" would be cut as
/// a fan of lines radiating from the head. The halo is known by its tilt SHRINKING along its own direction (a
/// convex surface's tilt grows toward its silhouette) by at least this much, in the normal's units over one
/// side of the sheet. The inside of a fold, concave, is given up with it: its lines are led by its edges.
const HALO: (f32, f32) = (0.3, 1.2);
/// The level rule is for the GROUND: a surface tilted up or down (a floor, a table, a road) is hatched along
/// its lines of equal depth. A wall seen aslant — tilted sideways — is not: its lines would turn vertical
/// beside every head. How much of the tilt must be vertical (least, full).
const UPHILL: (f32, f32) = (0.3, 0.7);
/// Reading a normal map. The grain it is smoothed past, as a share of the sheet's long side (one part in ...);
/// how far the normal must swing more one way than the other, per side of the sheet, before a surface is taken
/// to TURN (from ... to fully); the swing past which it is a crease or a step, not a surface; and how far a
/// steady surface must be TILTED from facing the viewer (the length of its normal's `x`,`y`) to rest the line
/// level with it.
const NORMAL_GRAIN: usize = 128;
const TURNS: (f32, f32) = (3.0, 8.0);
const CREASE: f32 = 40.0;
const TILTED: (f32, f32) = (0.35, 0.65);
/// How strictly a surface the normals call flat and facing the viewer is hatched straight: wholly. What is left
/// of the picture's own drift at 0.9 is a line that leans a few degrees this way and that along a wall — a
/// wave, on a surface that has none.
const STRAIGHT: f32 = 1.0;
/// A PEN DRAWING (see `drawing`): the tone its hatch begins at — the shadows, not the half-tones — the tone its
/// deepest shadow is cut to (one layer, lying one way — `HAND_OVER`, where a crossing would begin: a pen's shadow
/// is a ruled tone over the wash, and a crossing lays a flat grid over every surface alike), and how much
/// finer a line it still draws than an engraver would (a chain's least weight, as a share of the sheet).
const PEN_FROM: f32 = 0.42;
const PEN_DEEPEST: f32 = HAND_OVER;
const PEN_FINE: f32 = 300.0;
const PEN_COARSE: f32 = 200.0;
/// A hair of FUR is a short stroke: so many spacings long, each way from where it starts.
const HAIR_LENGTH: (f32, f32) = (2.5, 6.0);
/// On skin, an edge weaker than this share of a firm one is not a contour at all (see `plan`).
const SKIN_FAINT: f32 = 0.5;

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
    /// `follow` (0..1) is how far the cuts follow the form at all: at 0 the plate is ruled — every hatch line on
    /// the resting diagonal, or level on the ground — and only the contours bend. `held` (at the sheet's
    /// resolution, 0..1) is where the hatch is held at rest whatever `follow` says: skin, and fur (whose own
    /// direction, the coat's, is read on top). `grained`: `fur` is no coat but whatever has a grain, read only
    /// where the grain is strong enough to be seen.
    fn new(value: &[f32], w: usize, h: usize, ground: &[f32], fur: Option<&[f32]>, held: Option<&[f32]>, relief: Option<Relief>, normals: Option<Normals>, follow: f32, grained: bool) -> Flow {
        let long = w.max(h);
        let f = long.div_ceil(512).max(1);
        let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
        let mut small = vec![0f32; sw * sh];
        let mut level = vec![0f32; sw * sh];
        let mut rest = vec![0f32; sw * sh];
        let mut count = vec![0f32; sw * sh];
        for y in 0..h {
            for x in 0..w {
                let i = (y / f) * sw + x / f;
                small[i] += value[y * w + x];
                level[i] += ground[y * w + x];
                rest[i] += held.map_or(0.0, |m| m[y * w + x]);
                count[i] += 1.0;
            }
        }
        for (((s, l), r), c) in small.iter_mut().zip(level.iter_mut()).zip(rest.iter_mut()).zip(&count) {
            *s /= c.max(1.0);
            *l /= c.max(1.0);
            *r = (*r / c.max(1.0)).clamp(0.0, 1.0);
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
            let form = follow * (1.0 - rest[i]) * t * t * (3.0 - 2.0 * t) * e * e * (3.0 - 2.0 * e);
            // The edge's tangent is the gradient's orientation turned a quarter: the doubled angle negated.
            let (fc, fs) = if disc > 1e-12 { (-a / disc, -b / disc) } else { (0.0, -1.0) };
            led[i] = form;
            // At rest the line lies on the diagonal — or level, on open ground. A cheek's shadow stands alone on
            // the paper as a cast shadow does, but it is no ground: on skin the rest is the diagonal.
            let g = level[i].clamp(0.0, 1.0) * (1.0 - rest[i]);
            let (rc, rs) = (g, g - 1.0);
            let rest = (rc * rc + rs * rs).sqrt().max(1e-6);
            c2[i] = form * fc + (1.0 - form) * rc / rest;
            s2[i] = form * fs + (1.0 - form) * rs / rest;
        }
        // The normals say which way a surface turns outright; the relief only implies it, so it is asked second.
        if let Some(normals) = normals.filter(|m| m.w >= 2 && m.h >= 2 && m.n.len() == m.w * m.h) {
            turn_with_the_normals(&mut c2, &mut s2, &led, &rest, sw, sh, normals, STRAIGHT, follow);
        } else if let Some(relief) = relief.filter(|r| r.w >= 2 && r.h >= 2 && r.z.len() == r.w * r.h) {
            turn_with_the_form(&mut c2, &mut s2, &led, &rest, sw, sh, relief, follow);
        }
        // FUR grows its own way, whatever the form under it does: there the line takes the direction the coat's
        // own strands run in, read at the scale of a lock, wherever the strands agree on one — and where they do
        // not, the way the coat as a whole runs there, read at the scale of a flank. Not the form's: a hair does
        // not follow the lines of equal tone round a light patch of the coat, which would ring it like the grain
        // of a sawn trunk; it lies the way its neighbours lie.
        if let Some(fur) = fur {
            let (mut a, mut b, mut e) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            for i in 0..n {
                a[i] = gx[i] * gx[i] - gy[i] * gy[i];
                b[i] = 2.0 * gx[i] * gy[i];
                e[i] = gx[i] * gx[i] + gy[i] * gy[i];
            }
            let at = |r: usize| (blur(&a, sw, sh, r, 2), blur(&b, sw, sh, r, 2), blur(&e, sw, sh, r, 2));
            let flank = at(((long as f32 / 30.0 / f as f32).round() as usize).max(3));
            let lock = at(((long as f32 / 110.0 / f as f32).round() as usize).max(2));
            for (a, b, e) in [&flank, &lock] {
                for y in 0..sh {
                    for x in 0..sw {
                        let i = y * sw + x;
                        let len = (a[i] * a[i] + b[i] * b[i]).sqrt();
                        let agree = ((len / (e[i] + 1e-12) - 0.15) / 0.3).clamp(0.0, 1.0);
                        // (A grain is something SEEN: in a flat dark the strands that agree are the noise's, and
                        // followed, they grain a window's shadow like a sawn board.)
                        let seen = if grained { ((e[i] / energy - 0.6) / 1.2).clamp(0.0, 1.0) } else { 1.0 };
                        let t = fur[(y * f).min(h - 1) * w + (x * f).min(w - 1)] * agree * seen;
                        if len > 1e-12 {
                            c2[i] += (-a[i] / len - c2[i]) * t;
                            s2[i] += (-b[i] / len - s2[i]) * t;
                        }
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
///
/// `held` (0..1 a cell) is where neither rule is applied — the hatch stays at rest there.
fn turn_with_the_form(c2: &mut [f32], s2: &mut [f32], led: &[f32], held: &[f32], sw: usize, sh: usize, relief: Relief, follow: f32) {
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
        let t = follow * (1.0 - held[i]) * t * t * (3.0 - 2.0 * t);
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
        let u = 0.85 * sloped * one_way * flat * (1.0 - t) * (1.0 - led[i]) * (1.0 - held[i]);
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
        let none = vec![0f32; sw * sh];
        turn_with_the_normals(&mut c2, &mut s2, &none, &none, sw, sh, normals, 0.0, 1.0);
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
/// threshold depends on the picture. `held` (0..1 a cell) is where neither rule is applied — skin — and the
/// hatch stays at rest.
fn turn_with_the_normals(c2: &mut [f32], s2: &mut [f32], led: &[f32], held: &[f32], sw: usize, sh: usize, normals: Normals, straight: f32, follow: f32) {
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
    let (mut lx, mut ly, mut up, mut fade) = (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let d = r.max(1);
    // A difference over `2d` cells, as the normal's swing over one side of the sheet.
    let side = sw.max(sh) as f32 / (2 * d) as f32;
    let tilt: Vec<f32> = nx.iter().zip(&ny).map(|(x, y)| (x * x + y * y).sqrt()).collect();
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
            // How much of the tilt is up or down the sheet (the ground's) rather than sideways (a wall's).
            up[i] = ny[i] * ny[i] / (tilt[i] * tilt[i]).max(1e-6);
            // How fast the tilt dies away along its own direction: a halo's does, a convex surface's grows.
            let (mx, my) = ((at(&tilt, xp, y) - at(&tilt, xm, y)) * side, (at(&tilt, x, yp) - at(&tilt, x, ym)) * side);
            fade[i] = -(mx * nx[i] + my * ny[i]) / tilt[i].max(1e-3);
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
    let (up, fade) = (blur(&up, sw, sh, wide, 2), blur(&fade, sw, sh, wide, 2));
    let broad = wide * 3;
    let (lx, ly) = (blur(&lx, sw, sh, broad, 3), blur(&ly, sw, sh, broad, 3));
    for i in 0..n {
        let len = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
        let agree = ((len / bend[i].max(1e-9) - 0.5) / 0.3).clamp(0.0, 1.0);
        let firm = ((bend[i] - TURNS.0) / (TURNS.1 - TURNS.0)).clamp(0.0, 1.0);
        // The halo of a silhouette is no surface: a tilt that dies away along its own direction is no
        // curvature, and nothing turns there — the surface is as flat as the picture says.
        let halo = ((fade[i] - HALO.0) / (HALO.1 - HALO.0)).clamp(0.0, 1.0);
        let bend = bend[i] * (1.0 - halo);
        let firm = firm * (1.0 - halo);
        let t = agree * firm;
        let t = follow * (1.0 - held[i]) * t * t * (3.0 - 2.0 * t);
        if len > 1e-9 {
            c2[i] += (vx[i] / len - c2[i]) * t;
            s2[i] += (vy[i] / len - s2[i]) * t;
        }
        // A surface that neither turns nor tilts — a wall behind a head, the flat face of a machine — is hatched
        // STRAIGHT, on the resting diagonal: whatever orientation the picture's values have there is the drift
        // of its light or the grain of its canvas, and following it bends the lines of a surface that is flat.
        let still = 1.0 - (bend / TURNS.0).clamp(0.0, 1.0);
        // The tilt is squared in `lx`,`ly`: their length is the squared tilt where it holds one way. Only a
        // FLAT surface tilted up or down — the ground — rests the line level: a halo bends, a wall tilts sideways.
        let level = (lx[i] * lx[i] + ly[i] * ly[i]).sqrt();
        let tilted = ((level.sqrt() - TILTED.0) / (TILTED.1 - TILTED.0)).clamp(0.0, 1.0);
        let uphill = ((up[i] - UPHILL.0) / (UPHILL.1 - UPHILL.0)).clamp(0.0, 1.0);
        let u = 0.85 * tilted * uphill * still * (1.0 - t) * (1.0 - led[i]) * (1.0 - held[i]);
        if level > 1e-12 {
            c2[i] += (lx[i] / level - c2[i]) * u;
            s2[i] += (ly[i] / level - s2[i]) * u;
        }
        let plain = straight * still * (1.0 - tilted * uphill) * (1.0 - t);
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
    /// How many spacings apart the layer's lines lie here: one, and below a hair's worth of tone more, up to
    /// `SPARSE` — a light is the same hair, further apart. Lighter still, where the hairs break into flicks
    /// and dots, the lines come back to one spacing: stipple is scattered, not laid in ranks.
    fn apart(&self, p: [f32; 2], layer: usize) -> f32 {
        let part = self.width(p, layer) / self.hair;
        if part >= 1.0 / SPARSE {
            (1.0 / part).clamp(1.0, SPARSE)
        } else {
            1.0
        }
    }
    fn open(&self, p: [f32; 2], layer: usize) -> bool {
        if p[0] < 0.0 || p[1] < 0.0 || p[0] >= self.w as f32 || p[1] >= self.h as f32 {
            return false;
        }
        // Every layer runs on below a hair's worth of tone: spread out, then as flicks and as dots (see `plan`).
        let least = [BARE, STIPPLE / SPARSE, STIPPLE / SPARSE][layer] * self.hair;
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
        // The turn the line has made so far, over BOTH its halves: it is one line, and its whole length turns
        // no further than `SWEEP`.
        let mut used = 0f32;
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
                if swept.abs() + used > SWEEP {
                    break;
                }
                p = [p[0] + d[0], p[1] + d[1]];
                // Into the light the lines lie further apart: of two running side by side, the later stops.
                if !self.open(p, layer) || near.any_within(p, self.spacing * 0.55 * self.apart(p, layer)) {
                    break;
                }
                pts.push(p);
                prev = d;
            }
            used += swept.abs();
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
    /// side (more in the light, where the lines lie further apart), and a coarse scan of the sheet starts the
    /// surfaces no neighbour reached.
    fn layer(&self, layer: usize, seed: u64) -> Vec<Vec<[f32; 2]>> {
        let mut near = Near::new(self.w, self.h, self.spacing);
        let mut lines: Vec<Vec<[f32; 2]>> = Vec::new();
        let min_len = (self.spacing * 2.0).max(4.0) as usize;
        let every = (self.spacing * 0.5).max(1.0) as usize;
        let mut next = 0usize;
        let grow = |from: [f32; 2], lines: &mut Vec<Vec<[f32; 2]>>, near: &mut Near, next: &mut usize| {
            if !self.open(from, layer) || near.any_within(from, self.spacing * 0.92 * self.apart(from, layer)) {
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
                    let gap = self.spacing * self.apart(here, layer);
                    for side in [-1.0f32, 1.0] {
                        let c = [here[0] - dy / len * gap * side, here[1] + dx / len * gap * side];
                        if !self.open(c, layer) || near.any_within(c, self.spacing * 0.92 * self.apart(c, layer)) {
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

/// Plan the plate for a picture. `contour` (0..1) is how much of the edge map is drawn; `white` (0.5..1) is the
/// WHITE POINT, as a share of the picture's values: what lies at or above it is clean paper (`None` finds the
/// paper in the picture — see `PAPER_SHARE`; a white dress wants more of the picture cut, 0.98); `follow` (0..1) how far the hatch follows the form (0 = a
/// ruled plate, every line on the diagonal or level on the ground); `budget` caps the number of cuts — when
/// the hatch would not fit, its spacing widens until it does.
pub fn plan(picture: &RgbImage, relief: Option<Relief>, normals: Option<Normals>, materials: Option<Materials>, contour: f32, white: Option<f32>, follow: f32, budget: usize, seed: u64) -> Plate {
    cut(picture, relief, normals, materials, contour, white, follow, budget, seed, None)
}

/// A PEN DRAWING of the picture, for a medium that draws a line over its own washes: the plate's contours —
/// with the finer ones an engraver leaves out, the joints of a wall, the folds of a sleeve: a pen draws
/// texture — and a hatch in the SHADOWS only, `shade` (0..1) of an engraving's weight there, lying one
/// way. The lights and the middle tones are left to the washes.
///
/// With `normals` the hatch turns with the form, as an engraving's does. `materials` name the SKIN: a face is
/// drawn, not shaded — its contours all kept, the faint ones too, and no hatch laid over it.
pub fn drawing(picture: &RgbImage, normals: Option<Normals>, materials: Option<Materials>, contour: f32, shade: f32, budget: usize, seed: u64) -> Plate {
    cut(picture, None, normals, materials, contour, None, 1.0, budget, seed, Some(shade.clamp(0.0, 1.0)))
}

fn cut(picture: &RgbImage, relief: Option<Relief>, normals: Option<Normals>, materials: Option<Materials>, contour: f32, white: Option<f32>, follow: f32, budget: usize, seed: u64, pen: Option<f32>) -> Plate {
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
    // THE PAPER. Named, the white point is a share of the picture's values. Unnamed, it is looked for: the
    // paper is the picture's brightest BROAD FLAT value — the sheet of a drawing, a sky, a wall — a band of
    // values holding `PAPER_SHARE` of the picture, in its lights (the upper quarter of its values, and no
    // darker than a middle grey — a dark ground is not paper), taken at the top of its peak, since a sheet's
    // own values straddle a band. Where there is no such value — a dark
    // painting with a lit face, its lights spread thin — the paper is the brightest thousandth. (A fixed share
    // of the values — the tenth-lightest — took the whole lit half of a face on a dark ground for paper.)
    // A sheet is not one value to a photograph: it yellows toward its edges, the lamp falls off across it —
    // and no engraver cuts that. The paper's RUN of values is the bands, each `PAPER_SHARE` broad, that hold
    // together round the peak; within it, a tone no darker than its surroundings is the sheet's own tint, and
    // is paper (below). `sheet` is the bottom of the run.
    let (white, sheet) = match white {
        Some(f) => (q(f.clamp(0.5, 1.0)), None),
        None => {
            let mut band = [0usize; 50];
            for v in &sorted {
                band[((v * 50.0) as usize).min(49)] += 1;
            }
            let broad = (sorted.len() as f32 * PAPER_SHARE) as usize;
            let lights = ((q(0.75) * 50.0) as usize).clamp(25, 49);
            match (lights..50).rev().find(|&b| band[b] >= broad) {
                Some(mut b) => {
                    while b > 0 && band[b - 1] > band[b] {
                        b -= 1;
                    }
                    let mut lo = b;
                    while lo > 0 && band[lo - 1] >= broad && band[lo - 1] <= band[lo] {
                        lo -= 1;
                    }
                    ((b + 1) as f32 / 50.0, Some(lo as f32 / 50.0))
                }
                None => (q(0.999), None),
            }
        }
    };
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
    // The sheet's tint: how far below the white point the paper's own run of values reaches. A tone whose
    // surroundings lie within it is on the sheet, and the sheet's level is taken out of it — wholly, and
    // fading to nothing by twice the run's depth, where the surroundings are a tone of the subject's.
    let tint = sheet.map_or(0.0, |s| ((white - s) / (white - black)).clamp(0.0, 1.0));
    let modelled: Vec<f32> = dark
        .iter()
        .zip(&around)
        .map(|(&d, &a)| {
            let sheet = a * (1.0 - ((a - tint) / tint.max(1e-3)).clamp(0.0, 1.0));
            (d - sheet + RELIEF * (d - a)).clamp(0.0, 1.0)
        })
        .collect();
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
    for ((&m, &d), &a) in modelled.iter().zip(&dark).zip(&around).filter(|((m, _), _)| **m > 0.08) {
        hist[((m * 255.0) as usize).min(255)] += ((d - a).abs() / RANK_RELIEF).clamp(0.05, 1.0);
    }
    let total: f32 = hist.iter().sum::<f32>().max(1.0);
    let mut rank = [0f32; 256];
    let mut run = 0f32;
    for (r, n) in rank.iter_mut().zip(&hist) {
        run += n;
        *r = run / total;
    }
    // MATERIALS, where they were named: how far each point is fur, how far polished, and how far skin.
    let made_of = |kinds: &[u8]| -> Option<Vec<f32>> {
        let m = materials.filter(|m| m.w >= 1 && m.h >= 1 && m.kind.len() == m.w * m.h)?;
        let is: Vec<f32> = (0..w * h).map(|i| if kinds.contains(&m.kind[((i / w) * m.h / h).min(m.h - 1) * m.w + ((i % w) * m.w / w).min(m.w - 1)]) { 1.0 } else { 0.0 }).collect();
        is.iter().any(|v| *v > 0.0).then(|| blur(&is, w, h, ((long / 400.0) as usize).max(1), 2))
    };
    let (fur, polished, skin) = (made_of(&[FUR]), made_of(&[GLASS, METAL]), made_of(&[SKIN]));
    let key = (MIDDLE_TONE / DEEPEST).ln() / 0.5f32.ln();
    let skin_key = (SKIN_TONE / DEEPEST).ln() / 0.5f32.ln();
    // A glint on skin outshines its NEAR surroundings too: a broad light's edge against the hair is no
    // highlight, and the forehead is modelled up to the hairline.
    let near = skin.as_ref().map(|_| blur(&dark, w, h, ((long / 150.0) as usize).max(1), 3));
    let tone: Vec<f32> = (0..w * h)
        .map(|i| {
            let d = modelled[i];
            let r = if d > 0.08 { rank[((d * 255.0) as usize).min(255)] } else { 0.0 };
            let face = skin.as_ref().map_or(0.0, |s| s[i]);
            // A highlight is left clean. The fringe of a tone that dies away into the paper — a cast shadow's
            // edge — is no highlight (nothing around it is darker by much): it thins out into stipple.
            // On a polished surface a light need outshine its surroundings by half as much to be a glint.
            let shine = polished.as_ref().map_or(0.0, |p| p[i]);
            let fringe = around[i] - dark[i];
            let lift = near.as_ref().map_or(fringe, |n| fringe - (fringe - fringe.min(n[i] - dark[i])) * face);
            if r < LIT && lift > GLINT * (1.0 - 0.5 * shine) {
                return 0.0;
            }
            let lighter = 1.0 - FAR_LIGHTER * far.as_ref().map_or(0.0, |f| f[i]);
            let x = BY_RANK * r + (1.0 - BY_RANK) * d;
            let t = DEEPEST * x.powf(key + (skin_key - key) * face).max(TOE * x);
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
        // one the fold of a sleeve, but a short faint one is craquelure, grain, noise — an engraver leaves it.
        // On SKIN a faint edge is no contour however long: the turn of a cheek, the shadow under a brow, is
        // modelled by the hatch, not outlined — outlined, a face is a contour map. Its firm edges (an eye, a
        // nostril, the line of the lips, the silhouette) are drawn as everywhere: a chain is cut where it runs
        // faint over skin, and what is left of it is judged on its own.
        let at_skin = |p: &[f32; 2]| skin.as_ref().map_or(0.0, |s| s[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)]);
        let faint = |p: &[f32; 2]| pen.is_none() && at_skin(p) > 0.5 && at(p) / strong < SKIN_FAINT;
        let weight = |c: &[[f32; 2]]| c.iter().map(|p| (at(p) / strong).min(1.0)).sum::<f32>() * 2.0;
        // (A pen's finest lines are a face's — the lid of an eye, a nostril. Off a face the finest are specks on
        // a tint: a stitch, a pore of a wall.)
        let skinned = |c: &[[f32; 2]]| c.iter().map(at_skin).sum::<f32>() > 0.5 * c.len() as f32;
        let mut chains: Vec<&[[f32; 2]]> = chains.iter().flat_map(|c| c.split(faint)).filter(|c| weight(c) >= long / if pen.is_none() { 140.0 } else if skinned(c) { PEN_FINE } else { PEN_COARSE }).collect();
        chains.sort_by_key(|c| std::cmp::Reverse(c.len()));
        let base = (long / 800.0).clamp(0.8, 3.0);
        let nominal = (long / 300.0).max(3.0);
        // A PEN hints at a texture, it does not trace it: the stitches of a knitted vest, the grain of a wall,
        // every lock of a head of hair are short edges crowding one another, and drawn each, they are a scribble
        // over the tint. The long lines are drawn first; a short one that runs where two lines already pass is
        // left out. (Not on skin: an eye is short lines close together, and all of them are the eye.)
        let crowd = (long / 60.0).max(8.0);
        let (gw, gh) = ((w as f32 / crowd) as usize + 1, (h as f32 / crowd) as usize + 1);
        let mut passed = vec![0u8; gw * gh];
        let cell = |p: &[f32; 2]| ((p[1] / crowd) as usize).min(gh - 1) * gw + ((p[0] / crowd) as usize).min(gw - 1);
        for (ci, chain) in chains.into_iter().enumerate() {
            if pen.is_some() {
                let short = chain.len() < (long / 12.0) as usize;
                let on_skin = chain.iter().map(at_skin).sum::<f32>() > 0.5 * chain.len() as f32;
                if short && !on_skin && chain.iter().filter(|p| passed[cell(p)] >= 1).count() * 10 > chain.len() * 5 {
                    continue;
                }
                // Nor does a pen OUTLINE a stitch: a short line that doubles back on itself round a sliver — the
                // rim of one rib of a knitted band, of one lock — is a worm lying on the tint. (A pane of a
                // window is as short and as closed, but it holds an area; a sliver holds none.)
                let run: f32 = chain.windows(2).map(|s| (s[1][0] - s[0][0]).hypot(s[1][1] - s[0][1])).sum();
                let (a, b) = (chain[0], chain[chain.len() - 1]);
                let n = chain.len() as f32;
                let mean = chain.iter().fold([0f32; 2], |m, p| [m[0] + p[0] / n, m[1] + p[1] / n]);
                let lie = chain.iter().fold([0f32; 3], |m, p| { let (dx, dy) = (p[0] - mean[0], p[1] - mean[1]); [m[0] + dx * dx / n, m[1] + dy * dy / n, m[2] + dx * dy / n] });
                let (mid, half) = ((lie[0] + lie[1]) * 0.5, (((lie[0] - lie[1]) * 0.5).powi(2) + lie[2] * lie[2]).sqrt());
                let sliver = (mid - half).max(0.0) < 0.16 * (mid + half);
                let closed = (b[0] - a[0]).hypot(b[1] - a[1]) < 0.3 * run;
                if !on_skin && chain.len() < (long / 6.0) as usize && closed && sliver {
                    continue;
                }
                // ... nor follow a row of them, to and fro: a line several times longer than the place it
                // stays in is a scribble.
                let (lo, hi) = chain.iter().fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), p| ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])]));
                if !on_skin && run > if closed { 3.2 } else { 2.2 } * (hi[0] - lo[0]).hypot(hi[1] - lo[1]) {
                    continue;
                }
                let mut last = usize::MAX;
                for p in chain {
                    let c = cell(p);
                    if c != last {
                        passed[c] = passed[c].saturating_add(1);
                        last = c;
                    }
                }
            }
            // A burin's groove is a V: the line enters as a needle, swells where the edge is firm, and leaves
            // as a needle.
            let finer = |p: &[f32; 2]| 1.0 - FAR_FINER * far.as_ref().map_or(0.0, |f| f[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)]);
            // (A pen's line is less the contrast it separates than a burin's: its weight is the hand's, leant on
            // in a shadow — see the drawing's own rule — and a faint edge under a chin is not a hair for being faint.)
            let (least, more) = if pen.is_some() { (0.8, 0.9) } else { (0.45, 1.6) };
            let mut widths: Vec<f32> = chain.iter().map(|p| finer(p) * base * (least + more * (at(p) / strong).min(1.0).powf(1.2))).collect();
            let at_fur = |p: &[f32; 2]| fur.as_ref().map_or(0.0, |f| f[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)]);
            if chain.iter().map(at_fur).sum::<f32>() > 0.5 * chain.len() as f32 {
                // Through fur a contour is no single line: the edge of a coat is hairs, lying end to end with
                // air between — pieces of their own length, each entering and leaving as a needle.
                let line = taut(chain);
                let (mut a, mut k) = (0usize, 0u64);
                while a + 2 < line.len() {
                    let own = |salt: u64| jitter(seed ^ salt, (ci as u64) << 16 | k) + 0.5;
                    let run = (nominal * (FUR_PIECE.0 + (FUR_PIECE.1 - FUR_PIECE.0) * own(0xFA1))) as usize;
                    let b = (a + run.max(3)).min(line.len() - 1);
                    let mut ws = widths[a..=b].to_vec();
                    swell(&mut ws, 4, base * 0.2, ((b - a) / 3).max(2));
                    pieces(0, &line[a..=b], &ws, &mut cuts);
                    a = b + (nominal * (0.6 + 0.8 * own(0xFA2))) as usize;
                    k += 1;
                }
            } else {
                swell(&mut widths, 10, base * 0.2, ((long / 90.0) as usize).min(chain.len() / 3).max(3));
                // (A pen's line is one sweep of the hand: pulled taut once more, so it does not tremble.)
                let line = if pen.is_some() { taut(&taut(&taut(chain))) } else { taut(chain) };
                pieces(0, &line, &widths, &mut cuts);
            }
            // A hatch line stops at a drawn contour (thickened, so a diagonal step cannot slip through).
            for pair in chain.windows(2) {
                for p in [pair[0], [(pair[0][0] + pair[1][0]) * 0.5, (pair[0][1] + pair[1][1]) * 0.5], pair[1]] {
                    let (x, y) = ((p[0] as usize).min(w - 2), (p[1] as usize).min(h - 2));
                    for i in [y * w + x, y * w + x + 1, (y + 1) * w + x] {
                        wall[i] = true;
                    }
                }
            }
            if cuts.len() >= budget / 5 * 2 {
                break;
            }
        }
    }

    // A PEN shades only its shadows: the tone below `PEN_FROM` is the washes', and what is above it is spread
    // over one layer and the beginning of a crossing.
    let tone: Vec<f32> = match pen {
        Some(shade) => tone.iter().enumerate().map(|(i, t)| ((t - PEN_FROM) / (DEEPEST - PEN_FROM)).clamp(0.0, 1.0) * PEN_DEEPEST * shade * (1.0 - skin.as_ref().map_or(0.0, |s| s[i]))).collect(),
        None => tone,
    };
    // HATCH: the layers, at a spacing that fits the budget.
    // A tone that lies alone on open paper — little around it is dark — is a shadow on the ground.
    let ground: Vec<f32> = around.iter().map(|a| ((0.42 - a) / 0.2).clamp(0.0, 1.0)).collect();
    // Skin and fur alike hold the hatch at rest: a face is not wrapped in lines, and a coat's hairs do not run
    // round the body under it as a hatch would — they lie the way the coat runs, which the flow reads on top.
    let held: Option<Vec<f32>> = match (&skin, &fur) {
        (Some(s), Some(f)) => Some(s.iter().zip(f).map(|(a, b)| a.max(*b)).collect()),
        (Some(s), None) => Some(s.clone()),
        (None, Some(f)) => Some(f.clone()),
        (None, None) => None,
    };
    // A PEN's hatch lies along the GRAIN of what it shades, everywhere but on skin: the locks of a head of hair,
    // the folds of a sleeve, the courses of a wall each have a direction of their own, and a shadow ruled across
    // them on one diagonal is a screen laid over the drawing, not the drawing's. Where a thing has no grain the
    // line rests as before.
    let grain: Option<Vec<f32>> = pen.map(|_| (0..w * h).map(|i| fur.as_ref().map_or(0.0, |f| f[i]).max(1.0 - skin.as_ref().map_or(0.0, |s| s[i]))).collect());
    let flow = Flow::new(&value, w, h, &ground, grain.as_deref().or(fur.as_deref()), held.as_deref(), relief, normals, follow.clamp(0.0, 1.0), pen.is_some());
    let room = budget.saturating_sub(cuts.len());
    let mut spacing = (long / if pen.is_some() { 200.0 } else { 300.0 }).max(3.0);
    let mut hatch: Vec<Cut> = Vec::new();
    for _ in 0..8 {
        hatch.clear();
        let field = Field { w, h, tone: &tone, wall: &wall, flow: &flow, spacing, hair: (spacing * 0.15).max(0.45), fur: fur.as_deref(), polished: polished.as_deref() };
        for layer in 0..LAYER_COVER.len() {
            for (li, line) in field.layer(layer, seed).into_iter().enumerate() {
                let raw: Vec<f32> = line.iter().map(|p| field.width(*p, layer)).collect();
                // Below a hair's worth of tone a line is not made thinner — it is laid further apart (see
                // `Field::apart`), and below the sparsest hairs it is BROKEN into dots, at one spacing again,
                // that thin out to the paper. Each line breaks at its own place, so the dots do not fall into ranks.
                let pitch = spacing * 1.15;
                let dot = (field.hair * 1.8).max(2.0);
                let phase = (jitter(seed ^ 0xF11C ^ layer as u64, li as u64) + 0.5) * pitch;
                let ruled = |i: usize| {
                    let p = line[i];
                    field.polished.map_or(0.0, |m| m[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)])
                };
                let on = |i: usize| {
                    // Below a hair's worth the lines have spread out (see `Field::apart`): a hair lying `SPARSE`
                    // spacings from the next carries a `SPARSE`th of its tone, so a line is whole down to there.
                    // On polished metal and glass the line is never broken: it runs on as a hair, straight.
                    let part = raw[i] / field.hair;
                    if part >= 1.0 / SPARSE || ruled(i) > 0.5 {
                        return true;
                    }
                    let at = i as f32 + phase;
                    let nth = (at / pitch) as u64;
                    // A dot at every pitch of a line at one spacing carries about what a hair `SPARSE` spacings
                    // from the next does, so the dots take over from the sparsest hairs at their full count and
                    // thin out from there — the first layer's evenly to the paper, so the light rolls off through
                    // stipple to the end; a crossing's over the share of a hair's worth under its sparsest hairs.
                    let often = if layer == 0 { (part - BARE) / (1.0 / SPARSE - BARE) } else { (part * SPARSE - STIPPLE) / (1.0 - STIPPLE) };
                    at % pitch < dot && jitter(seed ^ 0xD07 ^ layer as u64, (li as u64) << 20 | nth) + 0.5 < often.clamp(0.0, 1.0)
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
        let plate = plan(&two_masses(160, 160), None, None, None, 0.6, None, 1.0, 50_000, 7);
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
        let plate = plan(&two_masses(200, 200), None, None, None, 0.0, None, 1.0, 50_000, 7);
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
        let (full, tight) = (plan(&img, None, None, None, 0.0, None, 1.0, 50_000, 7), plan(&img, None, None, None, 0.0, None, 1.0, 60, 7));
        assert!(tight.cuts.len() <= 60, "the budget holds ({})", tight.cuts.len());
        assert!(tight.spacing > full.spacing, "by widening the spacing");
    }

    #[test]
    fn the_same_seed_cuts_the_same_plate() {
        let img = two_masses(120, 90);
        let (a, b) = (plan(&img, None, None, None, 0.5, None, 1.0, 20_000, 3), plan(&img, None, None, None, 0.5, None, 1.0, 20_000, 3));
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
        let plate = plan(&img, relief, normals, materials, 0.55, None, 1.0, 360_000, 42);
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
        let (flat, round) = (plan(&img, None, None, None, 0.0, None, 1.0, 50_000, 7), plan(&img, Some(Relief { w, h, z: &z }), None, None, 0.0, None, 1.0, 50_000, 7));
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
        let (flat, round) = (plan(&img, None, None, None, 0.0, None, 1.0, 50_000, 7), plan(&img, None, Some(Normals { w, h, n: &n }), None, 0.0, None, 1.0, 50_000, 7));
        let band = |p: &Plate| upright_in(p, (70.0, 130.0), (90.0, 110.0));
        assert!(band(&flat) < 0.56, "without the normals the cuts rest on the diagonal ({})", band(&flat));
        assert!(band(&round) > 0.8, "round the bar they turn upright ({})", band(&round));
    }

    #[test]
    fn skin_holds_the_hatch_at_rest_whatever_the_normals_say() {
        // The same bar, its normals swinging across it — but named skin: the cuts keep the diagonal there.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let n: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let t = (((i / w) as f32 - 100.0) / 24.0).clamp(-0.95, 0.95);
                [0.0, t, (1.0 - t * t).sqrt()]
            })
            .collect();
        let kind = vec![SKIN; w * h];
        let normals = Normals { w, h, n: &n };
        let (round, held) = (plan(&img, None, Some(normals), None, 0.0, None, 1.0, 50_000, 7), plan(&img, None, Some(normals), Some(Materials { w, h, kind: &kind }), 0.0, None, 1.0, 50_000, 7));
        let band = |p: &Plate| upright_in(p, (70.0, 130.0), (90.0, 110.0));
        assert!(band(&round) > 0.8, "round the bar the cuts turn upright ({})", band(&round));
        assert!(band(&held) > 0.4 && band(&held) < 0.6, "on skin they rest on the diagonal ({})", band(&held));
    }

    #[test]
    fn normals_lay_the_cuts_level_on_the_ground() {
        // A plane facing up and toward the viewer — the ground: the cuts lie level on it, not on the diagonal.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let n = vec![[0.0f32, -0.8, 0.6]; w * h];
        let plate = plan(&img, None, Some(Normals { w, h, n: &n }), None, 0.0, None, 1.0, 50_000, 7);
        let level = upright_in(&plate, (70.0, 130.0), (70.0, 130.0));
        assert!(level < 0.2, "on the ground the cuts lie level ({level})");
        // A plane facing the viewer says nothing: the diagonal stands.
        let facing = vec![[0.0f32, 0.0, 1.0]; w * h];
        let plate = plan(&img, None, Some(Normals { w, h, n: &facing }), None, 0.0, None, 1.0, 50_000, 7);
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
        let (plain, furred) = (plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 0.0, None, 1.0, 80_000, 7));
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
        let (plain, polished) = (plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 0.0, None, 1.0, 80_000, 7));
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
        let (led, flat) = (plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7), plan(&img, None, Some(Normals { w, h, n: &facing }), None, 0.0, None, 1.0, 80_000, 7));
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
        let (plain, furred) = (plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 0.0, None, 1.0, 80_000, 7));
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
        let plate = plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7);
        let last = plate.cuts.iter().flat_map(|c| c.path.iter().map(|p| p[0])).fold(0f32, f32::max);
        let dots = plate.cuts.iter().filter(|c| c.path[0][0] > last - 25.0 && c.path.len() <= 3).count();
        let lines = plate.cuts.iter().filter(|c| c.path[0][0] > last - 25.0 && c.path.len() > 3).count();
        // The white point is the ramp's tenth-lightest value, at x = 216: the dots run on to within a few
        // pixels of it, not to a bare band short of it.
        assert!(last > 202.0, "the ramp is cut to the paper's edge ({last})");
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
        let plate = plan(&img, Some(Relief { w, h, z: &z }), None, None, 0.0, None, 1.0, 80_000, 7);
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
    fn a_ruled_plate_keeps_the_diagonal_whatever_the_picture_does() {
        // The striped wall again: alone, the picture's stripes lead; at `follow` 0 the plate is ruled.
        let (w, h) = (200usize, 200usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if x < 30 || x > 170 || y < 30 || y > 170 { 235 } else if x % 16 < 8 { 15 } else { 80 };
            image::Rgb([v, v, v])
        });
        let (led, ruled) = (plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7), plan(&img, None, None, None, 0.0, None, 0.0, 80_000, 7));
        let (by_picture, by_rule) = (upright_in(&led, (60.0, 140.0), (60.0, 140.0)), upright_in(&ruled, (60.0, 140.0), (60.0, 140.0)));
        assert!(by_picture > 0.62, "alone, the picture's stripes lead ({by_picture})");
        assert!(by_rule > 0.4 && by_rule < 0.6, "ruled, the diagonal ({by_rule})");
    }

    #[test]
    fn the_white_point_is_a_lever() {
        // Paper, a broad light grey, a dark mass. With the white point at the picture's tenth-lightest value
        // the light grey is cut (lightly); brought down to it, the light grey is paper.
        let (w, h) = (200usize, 200usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if y < 80 { 235 } else if x < 40 { 40 } else { 200 };
            image::Rgb([v, v, v])
        });
        let in_light = |p: &Plate| p.cuts.iter().filter(|c| c.layer > 0).flat_map(|c| c.path.iter()).filter(|p| p[0] > 80.0 && p[1] > 100.0).count();
        let (cut, bare) = (plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7), plan(&img, None, None, None, 0.0, Some(0.5), 1.0, 80_000, 7));
        assert!(in_light(&cut) > 200, "the light grey is cut ({})", in_light(&cut));
        assert_eq!(in_light(&bare), 0, "with the white point brought down to it, the light grey is paper");
    }

    #[test]
    fn a_silhouettes_halo_turns_nothing() {
        // A dark disc on a mid grey wall, paper above. The normals say the wall is flat — except beside the disc,
        // where they tilt away from it and die off, as an estimated map's do. The wall beside the disc keeps
        // the diagonal.
        let (w, h) = (240usize, 240usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let (dx, dy) = (x as f32 - 120.0, y as f32 - 120.0);
            let v = if y < 40 { 235 } else if dx * dx + dy * dy < 40.0 * 40.0 { 40 } else { 120 };
            image::Rgb([v, v, v])
        });
        let n: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (dx, dy) = ((i % w) as f32 - 120.0, (i / w) as f32 - 120.0);
                let r = (dx * dx + dy * dy).sqrt();
                let tilt = if r < 40.0 { 0.0 } else { 0.6 * (1.0 - (r - 40.0) / 50.0).clamp(0.0, 1.0) };
                let (x, y) = (tilt * dx / r.max(1e-3), tilt * dy / r.max(1e-3));
                [x, y, (1.0 - x * x - y * y).max(0.0).sqrt()]
            })
            .collect();
        let plate = plan(&img, None, Some(Normals { w, h, n: &n }), None, 0.0, None, 1.0, 80_000, 7);
        // The band to the left of the disc: a fan would run level there (radial) or upright (level on the tilt).
        let beside = upright_in(&plate, (40.0, 75.0), (105.0, 135.0));
        assert!(beside > 0.38 && beside < 0.62, "beside the disc the wall keeps the diagonal ({beside})");
    }

    #[test]
    fn a_wall_seen_aslant_keeps_the_diagonal() {
        // The ground tilts up and rests the line level; a wall tilted sideways by as much does not — its lines
        // would turn vertical.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let wall = vec![[0.8f32, 0.0, 0.6]; w * h];
        let plate = plan(&img, None, Some(Normals { w, h, n: &wall }), None, 0.0, None, 1.0, 50_000, 7);
        let on_wall = upright_in(&plate, (70.0, 130.0), (70.0, 130.0));
        assert!(on_wall > 0.38 && on_wall < 0.62, "a wall seen aslant is hatched on the diagonal ({on_wall})");
    }

    #[test]
    fn a_contour_through_fur_is_hairs() {
        // The edge of a dark mass, drawn: one taut line — but through fur, pieces with air between them.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let kind = vec![FUR; w * h];
        let longest = |p: &Plate| p.cuts.iter().filter(|c| c.layer == 0).map(|c| c.path.windows(2).map(|s| ((s[1][0] - s[0][0]).powi(2) + (s[1][1] - s[0][1]).powi(2)).sqrt()).sum::<f32>()).fold(0f32, f32::max);
        let (line, hairs) = (plan(&img, None, None, None, 0.6, None, 1.0, 50_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 0.6, None, 1.0, 50_000, 7));
        assert!(longest(&line) > 30.0, "a drawn contour runs on ({})", longest(&line));
        assert!(longest(&hairs) <= FUR_PIECE.1 * 3.0 + 2.0, "through fur it is pieces ({})", longest(&hairs));
        assert!(hairs.cuts.iter().any(|c| c.layer == 0), "the pieces are still drawn");
    }

    #[test]
    fn a_highlight_is_clean_paper() {
        // A light strip that outshines the dark around it is a highlight: the plate leaves it uncut. (The strip
        // is read a few pixels out of focus, so its edges carry a trace of tone: the clean paper is judged
        // inside the fringe.)
        let img = RgbImage::from_fn(800, 800, |x, y| if x > 160 && x < 640 && y > 160 && y < 640 { let v = if x < 186 { 205 } else { 50 }; image::Rgb([v, v, v]) } else { image::Rgb([235, 232, 226]) });
        let plate = plan(&img, None, None, None, 0.0, None, 1.0, 200_000, 7);
        // The strip's side toward the paper is a tone dying away, and thins into dots; the side against the
        // dark is the glint, and is clean.
        let lit = plate.cuts.iter().filter(|c| c.layer > 0).flat_map(|c| c.path.iter()).filter(|p| p[0] > 173.0 && p[0] < 179.0 && p[1] > 200.0 && p[1] < 600.0).count();
        assert_eq!(lit, 0, "nothing is cut in the highlight");
        assert!(plate.cuts.iter().any(|c| c.layer > 0), "the rest is engraved");
    }

    #[test]
    fn the_deepest_crossing_enters_sparse_and_hair_fine() {
        // A long ramp out of the black: where the third layer begins, its lines are hairs, and lie further
        // apart than the spacing — a layer that began at its full width and density would draw its own edge,
        // a seam along the tone it starts at.
        let (w, h) = (2400usize, 64usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, _| {
            let v = (10.0 + 100.0 * x as f32 / w as f32) as u8;
            image::Rgb([v, v, v])
        });
        let plate = plan(&img, None, None, None, 0.0, None, 1.0, 400_000, 7);
        let deepest: Vec<&Cut> = plate.cuts.iter().filter(|c| c.layer == 3).collect();
        let end = |c: &Cut| c.path.iter().map(|p| p[0]).fold(0f32, f32::max);
        let last = deepest.iter().map(|c| end(c)).fold(0f32, f32::max);
        // The lines crossing a band of the ramp (the dots under them are not counted).
        let start = |c: &Cut| c.path.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        let crossing = |x0: f32, x1: f32| deepest.iter().filter(|c| c.path.len() > 3 && start(c) < x1 && end(c) > x0).count();
        let entering: Vec<&&Cut> = deepest.iter().filter(|c| end(c) > last - 20.0 && c.path.len() > 3).collect();
        let hair = (plate.spacing * 0.15).max(0.45);
        let widest = entering.iter().map(|c| c.w0.max(c.w1)).fold(0f32, f32::max);
        let (light, deep) = (crossing(last - 10.0, last), crossing(last - 100.0, last - 90.0));
        assert!(last > 100.0, "the third layer reaches out of the black ({last})");
        assert!(widest <= hair * 1.6, "where it begins its lines are hairs ({widest} against a hair of {hair})");
        assert!(light * 3 <= deep * 2, "and lie further apart than deeper in ({light} where it begins, {deep} deeper in)");
    }

    #[test]
    fn fur_holds_the_hatch_at_rest_whatever_the_normals_say() {
        // The bar with its normals swinging across it, named fur: a coat's hairs do not run round the body under
        // it as a hatch would. With no strands to read, they rest on the diagonal.
        let (w, h) = (200usize, 200usize);
        let img = two_masses(w as u32, h as u32);
        let n: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let t = (((i / w) as f32 - 100.0) / 24.0).clamp(-0.95, 0.95);
                [0.0, t, (1.0 - t * t).sqrt()]
            })
            .collect();
        let kind = vec![FUR; w * h];
        let normals = Normals { w, h, n: &n };
        let (round, coat) = (plan(&img, None, Some(normals), None, 0.0, None, 1.0, 50_000, 7), plan(&img, None, Some(normals), Some(Materials { w, h, kind: &kind }), 0.0, None, 1.0, 50_000, 7));
        let band = |p: &Plate| upright_in(p, (70.0, 130.0), (90.0, 110.0));
        assert!(band(&round) > 0.8, "round the bar the cuts turn upright ({})", band(&round));
        assert!(band(&coat) > 0.3 && band(&coat) < 0.7, "in fur they do not ({})", band(&coat));
    }

    #[test]
    fn on_skin_a_faint_edge_is_no_contour() {
        // A soft step down the middle of a light field, and a small dark square on it, three times as firm an
        // edge. Plain, the step is a long faint contour and is drawn; on skin it is left to the hatch, and the
        // square is drawn as everywhere.
        let (w, h) = (240usize, 240usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if (180..188).contains(&x) && (60..68).contains(&y) { 40 } else if x < 120 { 150 } else { 200 };
            image::Rgb([v, v, v])
        });
        let kind = vec![SKIN; w * h];
        let (plain, skin) = (plan(&img, None, None, None, 1.0, None, 1.0, 80_000, 7), plan(&img, None, None, Some(Materials { w, h, kind: &kind }), 1.0, None, 1.0, 80_000, 7));
        let step = |p: &Plate| p.cuts.iter().filter(|c| c.layer == 0).flat_map(|c| c.path.iter()).filter(|p| (p[0] - 120.0).abs() < 5.0 && p[1] > 20.0 && p[1] < 220.0).count();
        let square = |p: &Plate| p.cuts.iter().filter(|c| c.layer == 0).flat_map(|c| c.path.iter()).filter(|p| p[0] > 174.0 && p[0] < 194.0 && p[1] > 54.0 && p[1] < 74.0).count();
        assert!(step(&plain) > 100, "plain, the soft step is drawn ({})", step(&plain));
        assert_eq!(step(&skin), 0, "on skin it is not");
        assert!(square(&skin) > 12, "the firm edge is drawn on skin too ({})", square(&skin));
    }

    #[test]
    fn the_paper_is_the_brightest_broad_light_or_the_brightest_thousandth() {
        // A dark painting with a lit face: most of the sheet near black, a lit patch at a middle value, a speck of
        // highlight. No broad light — so the paper is the brightest thousandth and the lit patch is modelled, not
        // clipped: it is cut. A drawing on a tinted sheet: the sheet (most of the picture, at one value) is the
        // paper, clean, and its subject is cut.
        let (w, h) = (240usize, 240usize);
        let dark = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if x > 100 && x < 140 && y > 100 && y < 140 { 160 } else if x > 112 && x < 128 && y > 56 && y < 72 { 250 } else { 15 };
            image::Rgb([v, v, v])
        });
        let sheet = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if x > 100 && x < 140 && y > 100 && y < 140 { 60 } else { 222 + ((x * 7 + y * 3) % 5) as u8 };
            image::Rgb([v, v, v])
        });
        let within = |p: &Plate, x: (f32, f32), y: (f32, f32)| p.cuts.iter().flat_map(|c| c.path.iter()).filter(|p| p[0] > x.0 && p[0] < x.1 && p[1] > y.0 && p[1] < y.1).count();
        let (dark, sheet) = (plan(&dark, None, None, None, 0.0, None, 1.0, 80_000, 7), plan(&sheet, None, None, None, 0.0, None, 1.0, 80_000, 7));
        assert!(within(&dark, (104.0, 136.0), (104.0, 136.0)) > 20, "the lit face on a dark ground is cut, not clipped ({})", within(&dark, (104.0, 136.0), (104.0, 136.0)));
        assert_eq!(within(&sheet, (0.0, 90.0), (0.0, 90.0)), 0, "the tinted sheet is paper ({})", within(&sheet, (0.0, 90.0), (0.0, 90.0)));
        assert!(within(&sheet, (104.0, 136.0), (104.0, 136.0)) > 60, "its subject is cut ({})", within(&sheet, (104.0, 136.0), (104.0, 136.0)));
    }

    #[test]
    fn a_light_is_the_same_hair_further_apart() {
        // A light tone beside a middle one, each flat. In the light the first layer's lines are whole hairs,
        // not flicks, and lie further apart than in the middle tone; the lines that run from the one into
        // the other drop out one at a time.
        let (w, h) = (300usize, 200usize);
        let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let v = if y < 30 { 240 } else if x < 150 { 120 } else { 205 };
            image::Rgb([v, v, v])
        });
        let plate = plan(&img, None, None, None, 0.0, None, 1.0, 80_000, 7);
        let first: Vec<&Cut> = plate.cuts.iter().filter(|c| c.layer == 1).collect();
        // Crossings of a scanline: how many lines it meets in each half.
        let crossings = |x0: f32, x1: f32| -> usize {
            let y = 120.0f32;
            first.iter().filter(|c| c.path.windows(2).any(|s| (s[0][1] - y) * (s[1][1] - y) <= 0.0 && s[0][0] > x0 && s[0][0] < x1)).count()
        };
        let (mid, light) = (crossings(20.0, 130.0), crossings(170.0, 280.0));
        assert!(light * 2 <= mid, "in the light the lines lie further apart ({light} to {mid})");
        assert!(light >= 4, "but there are lines in the light ({light})");
        // ... and they are whole: the longest run of a line in the light is many spacings.
        let longest = first.iter().filter(|c| c.path.iter().all(|p| p[0] > 170.0 && p[1] > 40.0)).map(|c| c.path.len()).max().unwrap_or(0);
        assert!(longest as f32 > plate.spacing * 6.0, "a hair in the light is whole ({longest} points at spacing {})", plate.spacing);
    }

    #[test]
    fn blank_paper_is_left_alone() {
        let plate = plan(&RgbImage::from_pixel(64, 64, image::Rgb([230, 228, 220])), None, None, None, 0.6, None, 1.0, 5_000, 7);
        assert!(plate.cuts.is_empty(), "nothing to cut ({})", plate.cuts.len());
    }
}
