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
    /// A WASH brush (0 = off): the stroke lays at most this many times its own single-hit deposit of film at
    /// any pixel since the canvas's last `mark_film`, so overlapping strokes of one wash pass build ONE film —
    /// the water evens the pigment out — rather than doubling where they cross.
    pub film_cap: f32,
    /// A WASH brush: loaded with water, it lays its full film to the very ends of the mark (no end taper) and
    /// its charge does not run out along the stroke (no dry-brush tail), so strokes overlap into one flat
    /// wash whose only edges are where the wash stops. Off for every brush mark.
    pub flat_ends: bool,
    /// A loaded WASH brush holds its charge along the whole stroke (no dry-brush tail) — with or without
    /// the end taper.
    pub hold_charge: bool,
    /// CHARGE DEPLETION (1 = the brush's own): a multiplier on how fast a lane's load runs out along the mark —
    /// above 1 the mark goes down thick at the start and thins to a dry tail over the layer beneath.
    pub deplete: f32,
    /// BRISTLE RIDGES (0 = off): the paint a lane leaves stands as high as that lane was loaded relative to
    /// its neighbours, so a stroke's relief is striated across its width. The amount scales the striation.
    pub ridges: f32,
    /// DRY-BRUSH SKIPPING (0 = off): a bristle with little water touches only the raised fibres of the
    /// paper — the drier the bristle, the fewer fibres it reaches — so a dry mark is broken by the tooth
    /// along its drag instead of printing as a solid, gritty band. The amount scales how high the fibres
    /// stand. A loaded wet brush floods the hollows and is never broken. A bristle whose charge has run out
    /// (the dry tail of a depleting mark) skips as a dry one does.
    pub skip: f32,
    /// THE COMMA (0 = off): the hand's pressure along the mark — nothing at the touch, hardest a third of the
    /// way along, lifting to nothing at the end (sin(π·t^k) with the peak at t = 0.3) — so the mark swells
    /// where it is pressed and thins to a tail: a drop, a comma, as a round brush leaves them. The amount
    /// is how far the width and the deposit follow that pressure.
    pub comma: f32,
    /// CREST (0..1, 0 = off): a dry brush dragged over a DRIED, textured layer catches only the tops of the
    /// relief — the deposit at a pixel is scaled by how far its paint stands above its neighbours (a 5×5
    /// mean), so the colour sits on the ridges and the furrows keep what was there. NEGATIVE (−1..0), the
    /// wash into the furrows: the deposit is scaled by how far the paint lies BELOW its neighbours, so a
    /// thin liquid colour settles in the hollows and runs off the ridges. Per stroke, in the record.
    pub crest: f32,
    /// FILM OVER (0..1, 0 = off): this share of the mark's deposit is laid as a translucent FILM over the paint
    /// beneath instead of mixed into it (see `Canvas::deposit_over`) — a glaze read by the two-layer
    /// Kubelka-Munk solution, tinting the dried layer by its own transmittance. Per stroke, in the record.
    pub over: f32,
    /// BRISTLE SPLIT (0..1, 0 = off): a half-dry brush dragged fast parts along its vector — the clumped
    /// bristles separate into two to four bundles with bare canvas between them. The gaps open with the
    /// mark's dryness (laid dry, or its charge run out) and its speed (its length for its width), so a wet
    /// loaded dab stays whole and a fast dry sweep combs into parallel lines. Per stroke, in the record.
    pub split: f32,
    /// SPLIT TINT (0..1, 0 = off): the bundles of a split mark differ a little in tone — the paint not mixed
    /// to one colour on the palette, each clump of bristles carrying its own — each pigment's load in a bundle
    /// swung up to ±half by this amount, deterministic per bundle. Needs `split`. Per stroke, in the record.
    pub tint: f32,
    /// KNIFE (0..1, 0 = off): a palette knife, not a brush — a flat blade laid full width with square ends,
    /// no cross-section falloff, that SHEARS the paint already standing under it down toward a plateau (the
    /// 30th percentile of the height along its path) by this share before laying its own paint flat, so a
    /// knife mark is a smooth plane with sharp edges cut into the relief. Per stroke, in the record.
    pub knife: f32,
    /// TWIST (0..1, 0 = off): the brush turned in the fingers along the mark — its flat held full to the
    /// motion at one end and turned toward its edge by up to a quarter turn at the other (which end, by
    /// the mark's own hash), so the width runs from the brush's full breadth to a line: the calligraphic
    /// mark that opens from a point or draws out to one. Per stroke, in the record.
    pub twist: f32,
    /// CHALK (0..1, 0 = off): the mark is a stick of chalk on a toothed paper, not a brush — the deposit at a
    /// pixel is keyed to the paper's grain against the mark's PRESSURE (a light touch reaches only the fibres
    /// that stand up and the hollows stay the sheet's colour, a hard press fills them) and to how FULL the
    /// tooth already is (the filled hollows count as reached; past three fifths full the chalk no longer
    /// holds and slides over what is there, down to a seventh of its deposit). In the header, for the medium.
    pub chalk: f32,
    /// DUST (0..1, 0 = off): under the `chalk`, a hollow the stick's reach misses still takes this share of
    /// the reach's deposit — the charcoal's dust settling into the paper's tooth. In the header.
    pub dust: f32,
    /// REWET (0..1, 0 = off): the mark lifts this share of a DRIED layer's paint into its load, by its own
    /// wetness and how empty it is — gouache's gum softens again under water (the wet canvas's pickup is
    /// `k_pickup`'s). In the header.
    pub rewet: f32,
    /// TSUKE (0..1, 0 = off): tsuketate — the lanes across the width loaded from full (the side the path's
    /// left hand is on) to `1 − 0.85·tsuke`: one dark edge, one pale. In the header.
    pub tsuke: f32,
    /// KASURE (0..1, 0 = off): the flying white — the lanes dropping out along the mark by a slow hash of
    /// their own, by up to this share at its end. In the header.
    pub kasure: f32,
    /// NIJIMI (0..1, 0 = off): the ink feathering into the paper — the reach widened, the halo caught by the
    /// paper's fibres. In the header.
    pub nijimi: f32,
    /// TURN (0..1, 0 = off): the width narrowed by up to `0.6·turn` where the path bends (fully at a right angle
    /// over half a width each way) — the brush lifted on a turn. In the header.
    pub turn: f32,
}

/// The fill of the tooth (as a share of `SAT_FULL`) past which the chalk slides (see `BrushConfig::chalk`), and
/// how much of its deposit slides off a full tooth.
const CHALK_FULL: f32 = 0.6;
const CHALK_SLIP: f32 = 0.85;

/// The exponent that sets the comma's peak at t = 0.3 (ln 0.5 / ln 0.3).
const COMMA_K: f32 = 0.5757;

impl Default for BrushConfig {
    fn default() -> Self {
        // Oil-direct-ish: deposit a solid fraction of the load per step so a charged brush lays OPAQUE paint
        // (hides the ground → deep darks, bright lights, saturated colour, no wash), drying toward its end (the
        // loaded-gradient). Pickup is MODEST — too much drags wet paint and smears every stroke into its
        // neighbour (the "smeared slop"); alla-prima keeps marks distinct, sitting on top, only lightly harmonised.
        Self { k_deposit: 0.34, k_pickup: 0.25, viscosity: 1.0, bristles: 7, load_max: 6.0, streak: 0.6, round: 0.7, film_cap: 0.0, flat_ends: false, hold_charge: false, deplete: 1.0, ridges: 0.0, skip: 0.0, comma: 0.0, crest: 0.0, over: 0.0, split: 0.0, tint: 0.0, knife: 0.0, twist: 0.0, chalk: 0.0, dust: 0.0, rewet: 0.0, tsuke: 0.0, kasure: 0.0, nijimi: 0.0, turn: 0.0 }
    }
}

/// Saturation at which the tooth is full and deposit stops.
const SAT_FULL: f32 = 6.0;

/// The mark length (px) a brush's charge is sized for. A brush is loaded FOR THE MARK it is about to make: a
/// wide block-in stroke of 160px carries proportionally more paint than a 10px detail dab, and lays it along
/// its whole length. Without this every stroke spent its charge over the first ~12px (the load depleted
/// geometrically per pixel-hit regardless of length) and dragged nearly dry for the rest — the block-in came
/// out as a light raked wash over the ground (measured: block-in luma 0.67 against a 0.33 reference) while the
/// short detail marks landed at full strength, which is exactly the "dark rims, light cores" tell. Marks at or
/// under this length deplete as before; longer marks deplete proportionally slower, so the charge lasts the mark
/// (with a natural drybrush tail from the remaining geometric decay).
const CHARGE_PX: f32 = 12.0;

/// How hard a full tooth throttles further deposit. At 1.0 (the old behaviour) a saturated pixel refuses ALL new
/// paint, so once the block-in fills the tooth no later pass can build a darker dark or a brighter light — the
/// value range freezes into a flat, washed mid-tone. Below 1.0 a full pixel still accepts a fraction of each
/// stroke, so opaque media keep building value by shifting the pigment RATIO (KM colour is by ratio, not amount).
const SAT_THROTTLE: f32 = 0.4;

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
        self.rasterize_ramp(canvas, brush, None)
    }

    /// `rasterize`, with the brush's charge RAMPED along the mark (see `MediumProfile::ramp`): when `load_end`
    /// is given, every lane's load drifts from the stroke's `load` toward it in proportion to the distance
    /// travelled — the colour at the start where the mark begins, the colour at the end where it lifts, the
    /// pickup and depletion riding on top as ever. `None` is `rasterize` to the bit.
    pub fn rasterize_ramp(&self, canvas: &mut Canvas, brush: &BrushConfig, load_end: Option<&[f32]>) {
        let pts = densify(&self.path);
        if pts.is_empty() || self.load.is_empty() {
            return;
        }
        // Lanes track the stroke WIDTH — ~one lane per pixel — so the footprint is a SOLID swath, not a comb of
        // spaced tine-lines (a fixed handful of bristles at a wide block-in radius reads as fork/rake marks, not a
        // brush). Each lane still keeps its own evolving load along the whole stroke, so dirty-brush pickup and
        // stroke-order harmonisation survive. Capped so a very wide stroke can't explode the inner loop.
        // NIJIMI (see `BrushConfig::nijimi`): the reach widened by the bleed, more for a wetter mark.
        let spread = if brush.nijimi > 0.0 { 1.0 + 0.8 * brush.nijimi.clamp(0.0, 1.0) * (0.3 + 0.7 * self.wetness.clamp(0.0, 1.0)) } else { 1.0 };
        // THE FLOOD (nijimi's): a WASH — wetness over 0.5, fully at 0.9 — is set by its water, not its hairs: the
        // two tones of `tsuke` mixed, the flying white of `kasure` gone, the hairs not running dry (`deplete`),
        // the ends faded, the halo a feathered edge. 0 under a brush without nijimi (every other medium untouched).
        let flood = if brush.nijimi > 0.0 { ((self.wetness.clamp(0.0, 1.0) - 0.5) / 0.4).clamp(0.0, 1.0) } else { 0.0 };
        let maxw = self.width0.max(self.width1).max(1.0) * spread;
        // A WASH brush lays lanes at two per pixel: at an angle, one lane per pixel leaves every other pixel
        // untouched (the rounding), and a wash came out as a dotted screen.
        let dense = brush.flat_ends || brush.hold_charge;
        let nb = ((if dense { maxw * 2.0 } else { maxw }).ceil() as usize).max(brush.bristles).clamp(1, 256);
        // BRISTLE STREAKS (naturalness): a real brush is uneven — some bristles carry more paint than others, so
        // a stroke shows drybrush streaks along its length, not a uniform blob. Give each lane a slightly
        // different starting load (deterministic from the stroke's origin, so replay is exact). A few lanes run
        // nearly dry, laying broken texture; this is the single biggest cue that a mark was dragged, not stamped.
        let seed = (self.path[0][0].to_bits() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (self.path[0][1].to_bits() as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        let mut bload: Vec<Vec<f32>> = (0..nb)
            .map(|b| {
                // Streak amount from the brush: 0 = even (round), 1 = raked (stiff/fan). Centred on 1 so the
                // average load — and thus the mark's opacity — is preserved.
                // (NIJIMI: a wet mark's hairs carry the ink alike — the water evens them, the streaks gone by the
                // key's share of its wetness: a broad wet stroke of sumi is a wash, not a comb.)
                let streak = brush.streak.clamp(0.0, 1.0) * if brush.nijimi > 0.0 { 1.0 - brush.nijimi.clamp(0.0, 1.0) * self.wetness.clamp(0.0, 1.0) } else { 1.0 };
                let f = (1.0 + streak * (lane_hash(seed, b as u64) - 0.5) * 1.6).max(0.08);
                self.load.iter().map(|c| c * f).collect()
            })
            .collect();
        // TSUKE (see `BrushConfig::tsuke`): the brush loaded in two tones across its width. (A WASH — wetness
        // over 0.5, fully at 0.9 — evens the two tones: the water mixes them across the hairs, so a flooded
        // mark is one tone, not a wedge dark along one edge.)
        if brush.tsuke > 0.0 && nb > 1 {
            let k = 0.85 * brush.tsuke.clamp(0.0, 1.0) * (1.0 - flood);
            for (b, lane) in bload.iter_mut().enumerate() {
                let f = 1.0 - k * (b as f32 + 0.5) / nb as f32;
                for v in lane.iter_mut() {
                    *v *= f;
                }
            }
        }
        let mut bwet: Vec<f32> = vec![self.wetness; nb];
        let n = canvas.n_pigments();
        let mut scratch: Vec<f32> = Vec::with_capacity(n); // reused deposit buffer — no per-pixel alloc
        // The brush is charged for THIS mark's length (see `CHARGE_PX`): a lane's load depletes per hit by this
        // fraction of what it deposits, so a long stroke lays paint along its whole travel.
        // (Under the FLOOD the water carries the ink the whole sweep: a hair does not run dry mid-band — that
        // left pale dashes along every wash.)
        let deplete = if brush.flat_ends || brush.hold_charge { 0.0 } else { ((CHARGE_PX / pts.len().max(1) as f32).clamp(0.02, 1.0) * brush.deplete.max(0.0)).min(1.0) * (1.0 - flood) };
        // BRISTLE SPLIT (see `BrushConfig::split`): the lanes fall into two to four bundles across the width;
        // each lane knows how far it stands from its bundle's middle (0) toward its edge (1), and the gaps
        // between the bundles open along the mark as it dries — by the mark's own dryness and by each lane's
        // spent charge — the more for a fast mark (its length for its width: two widths slow, six fast).
        let split = brush.split.clamp(0.0, 1.0);
        let nbund = 2 + ((lane_hash(seed, 0xB1) * 3.0) as usize).min(2);
        let lane_edge: Vec<f32> = if split > 0.0 && nb > 2 {
            (0..nb).map(|b| { let u = ((b as f32 + 0.5) / nb as f32 * nbund as f32).fract(); (u - 0.5).abs() * 2.0 }).collect()
        } else {
            Vec::new()
        };
        // SPLIT TINT (see `BrushConfig::tint`): each bundle carries its own slight variant of the load — every
        // pigment swung up to ±half the amount, by a hash of the bundle — so the combed lines differ in tone.
        if brush.tint > 0.0 && split > 0.0 && nb > 2 {
            let amt = brush.tint.clamp(0.0, 1.0) * 0.5;
            for b in 0..nb {
                let bidx = ((b as f32 + 0.5) / nb as f32 * nbund as f32) as u64;
                for (c, v) in bload[b].iter_mut().enumerate() {
                    *v *= (1.0 + amt * (lane_hash(seed ^ 0x7147, bidx * 16 + c as u64) - 0.5) * 2.0).max(0.1);
                }
            }
        }
        // THE TWIST (see `BrushConfig::twist`): how far the brush turns along this mark, and from which end.
        let twist_turn = if brush.twist > 0.0 { brush.twist.clamp(0.0, 1.0) * if lane_hash(seed, 0x7E15) < 0.5 { -1.0 } else { 1.0 } } else { 0.0 };
        // THE KNIFE (see `BrushConfig::knife`): the plateau the blade shears the standing paint toward — the
        // 30th percentile of the height along the path, so the blade rides on the lower paint and cuts the
        // ridges that stand above it.
        let plateau = if brush.knife > 0.0 {
            let mut hs: Vec<f32> = pts
                .iter()
                .filter_map(|p| {
                    let (x, y) = (p[0].round(), p[1].round());
                    (x >= 0.0 && y >= 0.0 && x < canvas.w as f32 && y < canvas.h as f32).then(|| canvas.height[y as usize * canvas.w as usize + x as usize])
                })
                .collect();
            if hs.is_empty() {
                -1.0
            } else {
                hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                hs[hs.len() * 3 / 10]
            }
        } else {
            -1.0
        };
        let speed = if split > 0.0 { ((pts.len() as f32 / maxw - 2.0) / 4.0).clamp(0.0, 1.0) } else { 0.0 };
        let dry_base = ((0.6 - self.wetness) / 0.6).clamp(0.0, 1.0);
        let start_norm: Vec<f32> = if split > 0.0 { bload.iter().map(|l| Self::load_norm(l).max(1e-6)).collect() } else { Vec::new() };
        // THE RAMP: the per-lane shift laid on each step is the end-to-start difference times the step's share
        // of the path, so over the whole mark the lane has drifted from the one charge to the other.
        let ramp: Option<Vec<f32>> = load_end.filter(|_| pts.len() > 1).map(|le| (0..n.min(self.load.len()).min(le.len())).map(|c| (le[c] - self.load[c]) / (pts.len() - 1) as f32).collect());

        for i in 0..pts.len() {
            let p = pts[i];
            if let (Some(step), true) = (&ramp, i > 0) {
                for lane in bload.iter_mut() {
                    for (c, d) in step.iter().enumerate() {
                        lane[c] = (lane[c] + d).max(0.0);
                    }
                }
            }
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
            // THE COMMA (see `BrushConfig::comma`): the pressure along the mark, swelling the width and the
            // deposit where the hand presses and lifting them toward the ends.
            let press = if brush.comma > 0.0 && !brush.flat_ends {
                1.0 - brush.comma.clamp(0.0, 1.0) * (1.0 - (std::f32::consts::PI * t.powf(COMMA_K)).sin().max(0.0))
            } else {
                1.0
            };
            let width = (self.width0 + (self.width1 - self.width0) * t) * (0.5 + 0.5 * press);
            // THE FLOOD's ends (see `flood`): a wash fades in and out over half a width at either end — a
            // square-cut end left a notch of bare paper against the next band.
            let end_fade = if flood > 0.0 {
                let d = i.min(pts.len().saturating_sub(1 + i)) as f32 / (0.5 * maxw).max(1.0);
                1.0 - flood * (1.0 - d.clamp(0.0, 1.0))
            } else {
                1.0
            };
            // THE TWIST (see `BrushConfig::twist`): the brush turned toward its edge along the mark, its
            // breadth across the motion falling with the cosine of the turn — never under a quarter.
            let width = if twist_turn != 0.0 {
                let tt = if twist_turn > 0.0 { t } else { 1.0 - t };
                width * (twist_turn.abs() * std::f32::consts::FRAC_PI_2 * tt).cos().abs().max(0.25)
            } else {
                width
            };
            let width = width * spread;
            // TURN (see `BrushConfig::turn`): the brush lifted where the path bends.
            let width = if brush.turn > 0.0 && pts.len() > 4 {
                let m = ((width * 0.5).round() as usize).max(2);
                let (a, c) = (pts[i.saturating_sub(m)], pts[(i + m).min(pts.len() - 1)]);
                let (ux, uy, vx, vy) = (p[0] - a[0], p[1] - a[1], c[0] - p[0], c[1] - p[1]);
                let (lu, lv) = ((ux * ux + uy * uy).sqrt(), (vx * vx + vy * vy).sqrt());
                if lu > 1e-3 && lv > 1e-3 {
                    let ang = ((ux * vx + uy * vy) / (lu * lv)).clamp(-1.0, 1.0).acos();
                    width * (1.0 - 0.6 * brush.turn.clamp(0.0, 1.0) * (ang / std::f32::consts::FRAC_PI_2).clamp(0.0, 1.0))
                } else {
                    width
                }
            } else {
                width
            };

            // Ends taper: the first/last ~22% of the stroke deposits less, so a mark has soft ROUND tips, not a
            // rectangular butt (the "blocky patch" tell). Deeper falloff = more organic marks.
            let end = if brush.flat_ends || brush.knife > 0.0 {
                1.0
            } else {
                let e = (t.min(1.0 - t) / 0.22).clamp(0.0, 1.0);
                (0.15 + 0.85 * e * e) * press
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
                // A wash brush lays its film evenly across its whole width: the water, not the bristles,
                // sets the film (a cross-section falloff left a third-strength rim along every stroke).
                // (Under NIJIMI the body is the inner `1/spread` of the reach and the rest its halo.)
                let d_full = (fr - 0.5).abs() * 2.0;
                let (d_body, halo) = if spread > 1.0 {
                    let core = 1.0 / spread;
                    if d_full <= core { (d_full / core, None) } else { (1.0, Some((d_full - core) / (1.0 - core))) }
                } else {
                    (d_full, None)
                };
                let edge = if dense || brush.knife > 0.0 {
                    1.0
                } else {
                    let d = d_body; // 0 centre → 1 edge
                    let rnd = brush.round.clamp(0.0, 1.0);
                    let pw = 2.0 + (1.0 - rnd) * 6.0; // round → parabola, flat → flatter top
                    // (A flooded wash's edge goes soft to nothing through its HALO — see `flood` — never by a rim
                    // at zero: that left a seam of bare paper where two bands met.)
                    let floor = 0.12 + (1.0 - rnd) * 0.33; // flat brush deposits more evenly across its width
                    (1.0 - d.powf(pw)).max(floor)
                };
                // NIJIMI's halo: falling off outward, caught where the paper's fibres stand above a rising threshold.
                // A WASH (wetness over 0.5, fully at 0.9) floods the fibres: its threshold drops by up to nine
                // tenths, so the halo is a soft feathered edge, not a speckle on the paper's tops.
                let edge = match halo {
                    Some(hh) => {
                        let relief = crate::paint::canvas::paper_relief(px as usize, py as usize);
                        edge * (1.0 - hh).powi(2) * ((relief - (0.25 + 0.6 * hh) * (1.0 - 0.9 * flood)) / 0.15).clamp(0.0, 1.0)
                    }
                    None => edge,
                };
                // KASURE (see `BrushConfig::kasure`): the hairs part as the brush runs — a lane drops out where its
                // own slow hash falls under the dryness at this point of the mark.
                let edge = if brush.kasure > 0.0 {
                    let a = brush.kasure.clamp(0.0, 1.0) * t.powf(1.5) * (1.0 - flood);
                    let nz = 0.75 * lane_hash(seed ^ 0x6A5E, b as u64) + 0.25 * lane_hash(seed ^ 0x6A5F, ((b as u64) << 20) ^ (i as u64 / 8));
                    edge * ((nz - a) / 0.08).clamp(0.0, 1.0)
                } else {
                    edge
                };
                let edge = edge * end_fade;
                if edge <= 0.0 {
                    continue;
                }
                // BRISTLE RIDGES: the paint a lane leaves stands as high as that lane was loaded relative to its
                // neighbours — a loaded bristle drags a ridge, a dry one a furrow — so a stroke's relief is
                // striated across its width (the relight read a smooth hump before; every stroke was a
                // flat tube). The ridge follows the lane's STARTING load (its streak), so it runs the whole stroke.
                // And the PLOW: the brush pushes paint out to the stroke's lateral edges, so the section is
                // not a rounded bump but a trough with a steep ridge thrown up along each side — the outer
                // lanes stand higher than the middle, more so for a flat brush (a knife-like edge) than a
                // round one. The paint's own colour is laid as before; only its height is displaced.
                let ridge = if brush.ridges > 0.0 && brush.viscosity > 0.0 && nb > 2 {
                    let k = brush.ridges.clamp(0.0, 1.0);
                    let lateral = (fr - 0.5).abs() * 2.0; // 0 centre → 1 edge
                    let plow = 1.0 + k * (0.9 - 0.4 * brush.round.clamp(0.0, 1.0)) * (lateral.powi(3) * 1.6 - 0.35);
                    plow * (1.0 + k * ((lane_hash(seed, b as u64 ^ 0x5A5A) - 0.5) * 1.6 + 0.4 * ((b as f32 * 2.4).sin() * 0.5 + 0.5) - 0.4))
                } else {
                    1.0
                };
                // The split's gap: a lane near its bundle's edge drops out as the gap opens (over a fifth of
                // the bundle's half-width, so the bundles keep soft sides).
                let keep = if lane_edge.is_empty() {
                    1.0
                } else {
                    let spent = 1.0 - (Self::load_norm(&bload[b]) / start_norm[b]).clamp(0.0, 1.0);
                    let gap = split * (0.4 + 0.6 * speed) * dry_base.max(spent * spent);
                    (((1.0 - gap) - lane_edge[b]) / 0.2).clamp(0.0, 1.0)
                };
                if keep <= 0.0 {
                    continue;
                }
                self.apply(canvas, px, py, &mut bload[b], &mut bwet[b], brush, n, edge * end * keep, deplete, &mut scratch, if brush.knife > 0.0 { 1.0 } else { ridge.max(0.05) }, plateau);
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

    /// Apply this stroke as a SMUDGE (the pastelist's finger, a tortillon; see [`Canvas::smudge`]): walk the
    /// path with a soft footprint — each lane across the width carries its own smear — and at each pixel move
    /// `strength` of the paint there toward the smear (falling toward the width's edges as a finger's
    /// pressure does), the smear itself taking the pixel in at the finger's MEMORY: `1 / width` per pixel, so
    /// what it carries fades over a width of travel and a wide finger blends at a wide scale. The finger goes along the path and BACK (what it picked up on the way out it brings
    /// home), so the blend is even in both directions. No load, no pickup: the paint is dragged along the
    /// path and blended, the finger holding some of it.
    pub fn smudge(&self, canvas: &mut Canvas, strength: f32) {
        let pts = densify(&self.path);
        let maxw = self.width0.max(self.width1).max(1.0);
        let nb = (maxw.ceil() as usize).clamp(1, 256);
        let share = strength.clamp(0.0, 1.0);
        // The finger holds about a finger's width of chalk: what it carries fades over `width` pixels of
        // travel, whatever its strength (tied to the strength, a light wide finger remembered three widths
        // and carried the hair's dark a hundred pixels into a cheek as a block).
        let memory = (1.0 / maxw).clamp(0.0, 0.5);
        let mut carry: Vec<Option<crate::paint::canvas::Smear>> = vec![None; nb];
        let there_and_back: Vec<usize> = (0..pts.len()).chain((0..pts.len()).rev()).collect();
        // The finger lands and lifts off softly: its share fades over the first and last fifth of the path.
        let fade = ((pts.len() as f32) * 0.2).max(1.0);
        for &i in &there_and_back {
            let p = pts[i];
            let end = ((i as f32 + 0.5) / fade).min(((pts.len() - 1 - i) as f32 + 0.5) / fade).clamp(0.0, 1.0);
            let end = end * end * (3.0 - 2.0 * end);
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
                let u = (b as f32 + 0.5) / nb as f32 * 2.0 - 1.0;
                let soft = 1.0 - u * u;
                let off = u * 0.5 * width;
                let x = (p[0] + perpx * off).round();
                let y = (p[1] + perpy * off).round();
                if x < 0.0 || y < 0.0 || x >= canvas.w as f32 || y >= canvas.h as f32 {
                    continue;
                }
                canvas.smudge(x as u32, y as u32, &mut carry[b], share * self.pressure * soft * end, memory);
            }
        }
    }

    /// One bristle touching one pixel: deposit a fraction of load, pick up wet canvas pigment, update the load.
    /// `cover` (0..1) is the soft footprint weight — the cross-section falloff toward the width's edges and the
    /// end taper — so a stroke reads as a brush mark, not a hard rectangular slab.
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, canvas: &mut Canvas, px: u32, py: u32, load: &mut [f32], bwet: &mut f32, brush: &BrushConfig, n: usize, cover: f32, deplete: f32, deposit: &mut Vec<f32>, ridge: f32, plateau: f32) {
        let p = py as usize * canvas.w as usize + px as usize;
        if canvas.clipped(p) {
            return;
        }
        if brush.knife > 0.0 && plateau >= 0.0 {
            // The blade shears the standing paint down toward the plateau before laying its own.
            let h = canvas.height[p];
            if h > plateau {
                canvas.height[p] = plateau + (h - plateau) * (1.0 - brush.knife.clamp(0.0, 1.0));
            }
        }
        let tooth = canvas.tooth[p];
        let contact = (self.pressure * cover.clamp(0.0, 1.0) * tooth).clamp(0.0, 1.0);
        let sat = (canvas.saturation_at(px, py) / SAT_FULL).clamp(0.0, 1.0);

        // Deposit: a fraction of the current load, throttled by contact and remaining tooth. `deposit` is a
        // caller-owned scratch buffer, cleared here — no per-pixel heap allocation.
        let mut df = (brush.k_deposit * contact * (1.0 - SAT_THROTTLE * sat)).clamp(0.0, 1.0);
        // The dry brush: a mark laid below 0.6 wet no longer floods the hollows; the drier it is, the higher
        // the fibre it must find to leave paint. The paper's relief is the same deterministic grain the
        // granulation settles into, so the skips and the mottle share one tooth. Read from the mark's OWN
        // wetness, not the bristle's running one: that decays along every stroke, and read there the tail of
        // every long wash went dry and the whole sheet stippled and darkened.
        if brush.skip > 0.0 {
            // ...and on a mark laid DRY, a bristle whose charge has run out — the dry tail of a depleting
            // mark — reaches only the raised fibres, so the tail breaks on the tooth instead of thinning
            // evenly. A wet mark is never broken, charge or no charge: water floods the hollows. (Full from
            // 0.45 wet down, fading out to nothing at 0.6.)
            let base = ((0.6 - self.wetness) / 0.6).clamp(0.0, 1.0);
            let spent = 1.0 - (Self::load_norm(load) / Self::load_norm(&self.load).max(1e-6)).clamp(0.0, 1.0);
            let dryness = base.max(spent * spent * (base * 4.0).min(1.0));
            if dryness > 0.0 {
                let thr = brush.skip.clamp(0.0, 1.0) * dryness * 0.9;
                let relief = crate::paint::canvas::paper_relief(px as usize, py as usize);
                let t = ((relief - thr + 0.12) / 0.24).clamp(0.0, 1.0);
                df *= t * t * (3.0 - 2.0 * t);
            }
        }
        if brush.chalk > 0.0 {
            // THE CHALK ON THE TOOTH (see `BrushConfig::chalk`): the stick reaches the paper's raised grain
            // first — pressed lightly it touches only the fibres that stand up, pressed hard it fills the
            // hollows — and as the layers fill the tooth the hollows count as reached; past the point where
            // the tooth is full the chalk no longer holds and slides over what is there.
            // (The relief runs from about 0.2 to 0.8 across the sheet. The stick's REACH runs from nothing at a
            // pressure of 0.3 to the whole grain at 0.85 — the hand's ordinary pressure fills every hollow, so
            // no pixel is a hole that no pass can ever fill (a fixed threshold left the lowest few percent of
            // the grain bare under every mark, a crap of the sheet's colour on a black coat); the light touch
            // of the stick's side reaches the raised half.)
            let c = brush.chalk.clamp(0.0, 1.0);
            let fill = (canvas.saturation_at(px, py) / SAT_FULL).clamp(0.0, 1.0);
            let reach = ((self.pressure - 0.3) / 0.55).clamp(0.0, 1.0);
            let thr = (0.1 + 0.7 * (1.0 - reach)) * (1.0 - fill);
            let relief = crate::paint::canvas::paper_relief(px as usize, py as usize);
            let t = ((relief - thr + 0.06) / 0.12).clamp(0.0, 1.0);
            let catch = t * t * (3.0 - 2.0 * t);
            // The DUST (see `BrushConfig::dust`): the hollows take a share of the reach's deposit.
            let catch = if brush.dust > 0.0 { catch.max(brush.dust.clamp(0.0, 1.0) * reach) } else { catch };
            let slide = ((fill - CHALK_FULL) / (1.0 - CHALK_FULL)).clamp(0.0, 1.0);
            df *= (1.0 - c * (1.0 - catch)) * (1.0 - c * CHALK_SLIP * slide);
        }
        if brush.crest != 0.0 {
            // The dry brush over the ridges (see `BrushConfig::crest`): paint only where this pixel's relief
            // stands above the mean of its 5×5 neighbourhood, by a smoothstep over a third of that mean.
            // NEGATIVE, the wash into the furrows: paint only where the relief lies BELOW that mean — a thin
            // liquid colour that settles in the hollows and runs off the ridges.
            let (w, h) = (canvas.w as i64, canvas.h as i64);
            let (cx, cy) = (px as i64, py as i64);
            let mut sum = 0.0f32;
            let mut cnt = 0.0f32;
            for yy in (cy - 2).max(0)..=(cy + 2).min(h - 1) {
                for xx in (cx - 2).max(0)..=(cx + 2).min(w - 1) {
                    sum += canvas.height[(yy * w + xx) as usize];
                    cnt += 1.0;
                }
            }
            let mean = sum / cnt.max(1.0);
            let above = canvas.height[p] - mean;
            let t = ((if brush.crest > 0.0 { above } else { -above }) / (0.35 * mean + 1e-4)).clamp(0.0, 1.0);
            let catch = t * t * (3.0 - 2.0 * t);
            df *= 1.0 - brush.crest.abs().clamp(0.0, 1.0) * (1.0 - catch);
        }
        if brush.film_cap > 0.0 {
            let cap = Self::load_norm(&self.load) * brush.k_deposit * brush.film_cap;
            let laid = canvas.film_since_mark(p);
            let room = (cap - laid).max(0.0);
            let would = Self::load_norm(load) * df;
            if would > room {
                df *= room / would.max(1e-6);
            }
        }
        deposit.clear();
        deposit.resize(n, 0.0);
        let mut dep_total = 0.0;
        for c in 0..n.min(load.len()) {
            let d = load[c].max(0.0) * df;
            deposit[c] = d;
            dep_total += d;
            load[c] -= d * deplete;
        }
        if brush.over > 0.0 {
            // The FILM OVER (see `BrushConfig::over`): this share of the deposit laid as a translucent layer
            // over the paint beneath, the rest mixed in as any mark's.
            let o = brush.over.clamp(0.0, 1.0);
            let h = dep_total * brush.viscosity * ridge;
            if o < 1.0 {
                let under: Vec<f32> = deposit.iter().map(|d| d * (1.0 - o)).collect();
                canvas.deposit(px, py, &under, h * (1.0 - o));
            }
            let film: Vec<f32> = deposit.iter().map(|d| d * o).collect();
            canvas.deposit_over(px, py, &film, h * o);
        } else {
            canvas.deposit(px, py, deposit, dep_total * brush.viscosity * ridge);
        }

        // Pickup: lift wet canvas pigment into the load (the dirty brush). Scales with how empty the brush is.
        let cw = canvas.wetness[p];
        let empty = (1.0 - Self::load_norm(load) / brush.load_max.max(1e-3)).clamp(0.0, 1.0);
        // (REWET — see `BrushConfig::rewet`: the dried paint softened by the mark's own water.)
        let pf = brush.k_pickup * cw * empty + if brush.rewet > 0.0 { brush.rewet * (1.0 - cw).max(0.0) * (*bwet).clamp(0.0, 1.0) * empty } else { 0.0 };
        if pf > 0.0 {
            let cconc = canvas.conc_at(px, py); // borrow directly — no clone
            let csum = cconc.iter().map(|c| c.max(0.0)).sum::<f32>().max(1e-6);
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
    fn the_ramp_drifts_the_mark_from_its_own_charge_to_the_end_charge() {
        // Red at the start, white at the end: the mark reads red where it begins and far lighter where it
        // lifts; without an end charge `rasterize_ramp` is `rasterize` to the bit.
        let path = vec![[3.0, 10.0], [116.0, 10.0]];
        let mut ramped = Canvas::white(120, 20, palette::ZORN, 0.9);
        let s = Stroke { path: path.clone(), width0: 6.0, width1: 6.0, load: red_load(), pressure: 1.0, wetness: 1.0 };
        s.rasterize_ramp(&mut ramped, &BrushConfig::default(), Some(&[0.0, 0.0, 0.0, 4.0]));
        let start = srgb_to_lab(ramped.color_at(8, 10));
        let end = srgb_to_lab(ramped.color_at(111, 10));
        assert!(start.a > 3.0, "the start is the red it was charged with (a* {})", start.a);
        assert!(end.l > start.l + 4.0 && end.a < start.a - 3.0, "the end has drifted toward the white: L {} → {}, a* {} → {}", start.l, end.l, start.a, end.a);
        let mut plain = Canvas::white(120, 20, palette::ZORN, 0.9);
        s.rasterize(&mut plain, &BrushConfig::default());
        let mut none = Canvas::white(120, 20, palette::ZORN, 0.9);
        s.rasterize_ramp(&mut none, &BrushConfig::default(), None);
        assert_eq!(plain.to_image().into_raw(), none.to_image().into_raw(), "no end charge: the plain rasterize to the bit");
        assert_ne!(plain.to_image().into_raw(), ramped.to_image().into_raw());
    }

    #[test]
    fn a_dry_brush_skips_the_paper_and_a_wet_one_floods_it() {
        // The same long mark laid nearly dry with `skip` on is BROKEN along its drag (bare and painted pixels
        // alternate on the centre line); laid wet it is solid; and with `skip` off the dry mark is as solid as
        // the default path always made it.
        let mark = |wet: f32, skip: f32| -> Vec<f32> {
            let mut c = Canvas::white(120, 16, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[4.0, 8.0], [116.0, 8.0]], width0: 5.0, width1: 5.0, load: red_load(), pressure: 1.0, wetness: wet };
            let brush = BrushConfig { skip, ..BrushConfig::default() };
            s.rasterize(&mut c, &brush);
            (6..114).map(|x| c.saturation_at(x as u32, 8)).collect()
        };
        let gaps = |m: &[f32]| m.iter().filter(|&&f| f < 1e-4).count();
        let dry = mark(0.15, 1.0);
        let wet = mark(0.9, 1.0);
        let off = mark(0.15, 0.0);
        assert!(gaps(&dry) > 10, "the dry mark is broken by the tooth ({} bare pixels of {})", gaps(&dry), dry.len());
        assert_eq!(gaps(&wet), 0, "the wet mark floods the hollows");
        assert_eq!(gaps(&off), 0, "with the dial off the dry mark is solid, as before");
    }

    #[test]
    fn a_depleting_dry_mark_breaks_at_its_tail_and_a_wet_one_never() {
        // A mark laid dry with a fast-depleting charge: its tail (the last third) is more broken than its
        // head; the same mark laid wet is solid to the end however the charge runs out.
        let mark = |wet: f32, skip: f32, deplete: f32| -> Vec<f32> {
            let mut c = Canvas::white(120, 16, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[4.0, 8.0], [116.0, 8.0]], width0: 5.0, width1: 5.0, load: red_load(), pressure: 1.0, wetness: wet };
            let brush = BrushConfig { skip, deplete, ..BrushConfig::default() };
            s.rasterize(&mut c, &brush);
            (6..114).map(|x| c.saturation_at(x as u32, 8)).collect()
        };
        let gaps = |m: &[f32]| m.iter().filter(|&&f| f < 1e-4).count();
        let dry = mark(0.4, 0.4, 2.0);
        let (head, tail) = dry.split_at(dry.len() / 3);
        let tail = &tail[tail.len() / 2..];
        assert!(gaps(tail) > gaps(head), "the depleted tail is more broken than the head ({} vs {} bare pixels)", gaps(tail), gaps(head));
        // Wet: whatever bare pixels an emptied brush leaves, the skip adds none.
        assert_eq!(gaps(&mark(0.9, 0.4, 2.0)), gaps(&mark(0.9, 0.0, 2.0)), "a wet mark is never broken by the tooth");
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

    #[test]
    fn a_crest_mark_paints_the_ridge_tops_and_leaves_the_furrows() {
        // A ridge of paint across the canvas; a crest mark dragged over it leaves its colour on the ridge
        // and none in the flat beside it, where an ordinary mark paints both alike.
        let lay = |crest: f32| {
            let mut c = Canvas::white(60, 40, palette::ZORN, 0.9);
            let ridge = Stroke { path: vec![[5.0, 20.0], [55.0, 20.0]], width0: 3.0, width1: 3.0, load: vec![0.0, 0.0, 0.0, 4.0], pressure: 1.0, wetness: 0.3 };
            ridge.rasterize(&mut c, &BrushConfig { viscosity: 2.0, k_pickup: 0.0, ..BrushConfig::default() });
            c.dry(1.0);
            let mark = Stroke { path: vec![[30.0, 5.0], [30.0, 35.0]], width0: 4.0, width1: 4.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            mark.rasterize(&mut c, &BrushConfig { crest, k_pickup: 0.0, ..BrushConfig::default() });
            let red = |x: u32, y: u32| c.conc_at(x, y)[1];
            (red(30, 20), red(30, 10))
        };
        let (on_ridge, in_flat) = lay(1.0);
        assert!(on_ridge > 0.05, "the crest mark lays its red on the ridge ({on_ridge})");
        assert!(in_flat < on_ridge * 0.05, "and none in the flat beside it ({in_flat} vs {on_ridge})");
        let (plain_ridge, plain_flat) = lay(0.0);
        assert!(plain_flat > plain_ridge * 0.5, "an ordinary mark paints both alike ({plain_flat} vs {plain_ridge})");
    }

    #[test]
    fn a_sink_mark_settles_beside_the_ridge_and_runs_off_its_top() {
        // The same ridge; a wash with a NEGATIVE crest dragged over it leaves its colour in the hollow at
        // the ridge's foot and none on the ridge's top — and none in the flat far from any relief, where
        // there is no hollow to settle in.
        let mut c = Canvas::white(60, 40, palette::ZORN, 0.9);
        let ridge = Stroke { path: vec![[5.0, 20.0], [55.0, 20.0]], width0: 3.0, width1: 3.0, load: vec![0.0, 0.0, 0.0, 4.0], pressure: 1.0, wetness: 0.3 };
        ridge.rasterize(&mut c, &BrushConfig { viscosity: 2.0, k_pickup: 0.0, ..BrushConfig::default() });
        c.dry(1.0);
        let mark = Stroke { path: vec![[30.0, 5.0], [30.0, 35.0]], width0: 4.0, width1: 4.0, load: red_load(), pressure: 1.0, wetness: 0.7 };
        mark.rasterize(&mut c, &BrushConfig { crest: -1.0, k_pickup: 0.0, ..BrushConfig::default() });
        let red = |x: u32, y: u32| c.conc_at(x, y)[1];
        let (on_ridge, at_foot, in_flat) = (red(30, 20), red(30, 23), red(30, 8));
        assert!(at_foot > 0.05, "the wash settles at the ridge's foot ({at_foot})");
        assert!(on_ridge < at_foot * 0.1, "and runs off its top ({on_ridge} vs {at_foot})");
        assert!(in_flat < at_foot * 0.05, "and finds no hollow in the flat ({in_flat} vs {at_foot})");
    }

    #[test]
    fn a_split_brush_dragged_dry_and_fast_combs_into_bundles() {
        // A long dry mark (nine widths for its width) with the split on leaves bare lanes across its width
        // where the whole brush painted a solid swath; a wet loaded dab of the same brush stays whole.
        let count = |wet: f32, len: f32, split: f32| {
            let mut c = Canvas::white(140, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[5.0, 20.0], [5.0 + len, 20.0]], width0: 12.0, width1: 12.0, load: red_load(), pressure: 1.0, wetness: wet };
            s.rasterize(&mut c, &BrushConfig { split, k_pickup: 0.0, ..BrushConfig::default() });
            let x = (5.0 + len * 0.5) as u32;
            (10..30).filter(|&y| c.conc_at(x, y)[1] > 0.02).count()
        };
        let whole = count(0.3, 108.0, 0.0);
        let combed = count(0.3, 108.0, 1.0);
        assert!(whole >= 10, "the whole brush lays a solid swath ({whole} lanes)");
        assert!(combed <= whole - 3, "the split brush leaves bare lanes between its bundles ({combed} of {whole})");
        let dab = count(1.0, 20.0, 1.0);
        assert!(dab >= whole - 1, "a wet short dab stays whole ({dab} of {whole})");
    }

    #[test]
    fn a_thin_white_film_over_black_veils_it_where_the_same_white_mixed_in_is_swallowed() {
        // A heavy black passage, dried; the same thin white laid once as a FILM over it and once mixed in. A
        // film scatters the light back before it reaches the black — a grey veil; mixed into the black's
        // own layer it is outweighed by the black's absorption and the passage stays black.
        let lay = |over: f32| {
            let mut c = Canvas::white(40, 40, palette::ZORN, 0.9);
            let base = Stroke { path: vec![[5.0, 20.0], [35.0, 20.0]], width0: 10.0, width1: 10.0, load: vec![0.0, 0.0, 6.0, 0.0], pressure: 1.0, wetness: 0.3 };
            for _ in 0..12 {
                base.rasterize(&mut c, &BrushConfig { k_pickup: 0.0, ..BrushConfig::default() });
            }
            c.dry(0.0);
            let luma = |p: [u8; 3]| p[0] as f32 * 0.299 + p[1] as f32 * 0.587 + p[2] as f32 * 0.114;
            let before = luma(c.color_at(20, 20));
            let veil = Stroke { path: vec![[5.0, 20.0], [35.0, 20.0]], width0: 10.0, width1: 10.0, load: vec![0.0, 0.0, 0.0, 6.0], pressure: 1.0, wetness: 0.7 };
            veil.rasterize(&mut c, &BrushConfig { over, k_pickup: 0.0, ..BrushConfig::default() });
            (before, luma(c.color_at(20, 20)))
        };
        let ((before, mixed), (_, film)) = (lay(0.0), lay(1.0));
        assert!(before < 70.0, "the passage beneath is black ({before:.1})");
        assert!(film > mixed + 15.0, "the white film veils the black where the mixed white is swallowed ({film:.1} vs mixed {mixed:.1}, black {before:.1})");
    }

    #[test]
    fn a_knife_shears_the_ridge_it_crosses_where_a_brush_rides_over_it() {
        // A thick ridge, dried; a knife mark across it cuts the ridge down to the plateau of the lower paint
        // along its path, where a brush mark of the same paint leaves the ridge standing under its own.
        let lay = |knife: f32| {
            let mut c = Canvas::white(60, 40, palette::ZORN, 0.9);
            let ridge = Stroke { path: vec![[5.0, 20.0], [55.0, 20.0]], width0: 4.0, width1: 4.0, load: vec![0.0, 0.0, 0.0, 4.0], pressure: 1.0, wetness: 0.3 };
            ridge.rasterize(&mut c, &BrushConfig { viscosity: 3.0, k_pickup: 0.0, ..BrushConfig::default() });
            c.dry(0.0);
            let before = c.height[20 * 60 + 30];
            let mark = Stroke { path: vec![[30.0, 5.0], [30.0, 35.0]], width0: 6.0, width1: 6.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            mark.rasterize(&mut c, &BrushConfig { knife, k_pickup: 0.0, ..BrushConfig::default() });
            (before, c.height[20 * 60 + 30], c.height[10 * 60 + 30])
        };
        let (ridge_h, knifed, knife_flat) = lay(1.0);
        let (_, brushed, _) = lay(0.0);
        assert!(ridge_h > 0.5, "the ridge stands ({ridge_h})");
        assert!(brushed > ridge_h, "a brush rides over the ridge and adds to it ({brushed} vs {ridge_h})");
        assert!(knifed < ridge_h * 0.6, "the knife shears the ridge down ({knifed} vs {ridge_h})");
        assert!(knifed <= knife_flat + ridge_h * 0.3, "to no higher than its own paint beside it ({knifed} vs {knife_flat})");
    }

    #[test]
    fn a_split_marks_bundles_differ_in_tone_with_the_tint_and_not_without() {
        // A split mark of two pigments; with the tint on, the bundles' pigment ratios differ across the
        // width; without it every bundle carries the same mix.
        let ratios = |tint: f32| {
            let mut c = Canvas::white(140, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[5.0, 20.0], [113.0, 20.0]], width0: 12.0, width1: 12.0, load: vec![2.0, 2.0, 0.0, 0.0], pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { split: 1.0, tint, k_pickup: 0.0, ..BrushConfig::default() });
            (10..30)
                .filter_map(|y| {
                    let k = c.conc_at(59, y);
                    (k[0] + k[1] > 0.02).then(|| k[0] / (k[0] + k[1]))
                })
                .collect::<Vec<f32>>()
        };
        let spread = |r: &[f32]| r.iter().copied().fold(0.0_f32, f32::max) - r.iter().copied().fold(1.0_f32, f32::min);
        let (plain, tinted) = (ratios(0.0), ratios(1.0));
        assert!(plain.len() >= 4 && tinted.len() >= 4, "the mark lays lanes ({} / {})", plain.len(), tinted.len());
        assert!(spread(&plain) < 0.01, "without the tint every lane carries one mix (spread {})", spread(&plain));
        assert!(spread(&tinted) > 0.05, "with it the bundles differ (spread {})", spread(&tinted));
    }

    #[test]
    fn a_twisted_mark_runs_from_the_brushs_breadth_to_a_line_where_a_plain_one_keeps_its_width() {
        // A long mark of one width; with the twist on, one end of it is far narrower than the other, where
        // the plain mark is as wide at both.
        let widths = |twist: f32| {
            let mut c = Canvas::white(140, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[5.0, 20.0], [125.0, 20.0]], width0: 14.0, width1: 14.0, load: red_load(), pressure: 1.0, wetness: 0.8 };
            s.rasterize(&mut c, &BrushConfig { twist, k_pickup: 0.0, ..BrushConfig::default() });
            let across = |x: u32| (5..35).filter(|&y| c.conc_at(x, y)[1] > 0.01).count();
            (across(22), across(108))
        };
        let (a0, b0) = widths(0.0);
        let (a1, b1) = widths(1.0);
        assert!((a0 as i32 - b0 as i32).abs() <= 2, "a plain mark is as wide at both ends ({a0} / {b0})");
        assert!(a1.min(b1) * 2 < a1.max(b1), "a twisted mark narrows to a line at one end ({a1} / {b1})");
    }

    #[test]
    fn the_chalk_catches_the_raised_grain_and_slides_off_a_full_tooth() {
        // A light touch of chalk reaches the raised fibres only — less paint than the same mark pressed hard, and
        // less than the brush without the chalk law; and over a tooth already full the chalk slides.
        // (A lane that skips a pixel keeps its charge for the next, so the mark's TOTAL hardly tells; what
        // tells is its COVERAGE — the share of the band's pixels that took a fair deposit — and, over a
        // full tooth, the deposit at the mark's head where the charge is still whole.)
        let lay = |chalk: f32, pressure: f32, prefill: bool| -> (f32, f32) {
            let mut c = Canvas::white(60, 20, palette::ZORN, 0.9);
            if prefill {
                let fill = Stroke { path: vec![[0.0, 10.0], [60.0, 10.0]], width0: 20.0, width1: 20.0, load: vec![0.0, 0.0, 0.0, 12.0], pressure: 1.0, wetness: 0.3 };
                for _ in 0..8 {
                    fill.rasterize(&mut c, &BrushConfig { k_pickup: 0.0, hold_charge: true, ..BrushConfig::default() });
                }
            }
            let before: Vec<f32> = (0..20).flat_map(|y| (0..60).map(move |x| (x, y))).map(|(x, y)| c.saturation_at(x, y)).collect();
            let s = Stroke { path: vec![[4.0, 10.0], [56.0, 10.0]], width0: 6.0, width1: 6.0, load: red_load(), pressure, wetness: 0.2 };
            s.rasterize(&mut c, &BrushConfig { chalk, k_pickup: 0.0, hold_charge: true, ..BrushConfig::default() });
            let delta: Vec<f32> = (0..20).flat_map(|y| (0..60).map(move |x| (x, y))).map(|(x, y)| c.saturation_at(x, y) - before[y as usize * 60 + x as usize]).collect();
            let band: Vec<f32> = (9..=11).flat_map(|y| (8..52).map(move |x| (x, y))).map(|(x, y)| delta[y * 60 + x]).collect();
            let peak = band.iter().copied().fold(0.0f32, f32::max).max(1e-6);
            let coverage = band.iter().filter(|d| **d > 0.25 * peak).count() as f32 / band.len() as f32;
            let head: f32 = (6..16).map(|x| delta[10 * 60 + x]).sum();
            (coverage, head)
        };
        let (plain, _) = lay(0.0, 0.5, false);
        let (light, _) = lay(1.0, 0.5, false);
        let (hard, head_hard) = lay(1.0, 1.0, false);
        assert!(light < 0.75 * plain, "a light touch of chalk breaks on the grain: coverage {light} vs the brush's {plain}");
        assert!(hard > light + 0.15, "pressed hard the chalk fills the hollows: coverage {hard} vs {light}");
        let (_, head_full) = lay(1.0, 1.0, true);
        assert!(head_full < 0.4 * head_hard, "over a full tooth the chalk slides: {head_full} vs {head_hard}");
        // The chalk at 0 is the brush to the bit.
        let mut a = Canvas::white(30, 12, palette::ZORN, 0.9);
        let mut b = a.clone();
        let s = Stroke { path: vec![[2.0, 6.0], [28.0, 6.0]], width0: 5.0, width1: 4.0, load: red_load(), pressure: 0.7, wetness: 0.5 };
        s.rasterize(&mut a, &BrushConfig::default());
        s.rasterize(&mut b, &BrushConfig { chalk: 0.0, ..BrushConfig::default() });
        assert_eq!(a.to_image().into_raw(), b.to_image().into_raw());
    }

    #[test]
    fn the_finger_drags_the_paint_along_its_path_and_lays_none_of_its_own() {
        // A red patch on the left; a finger dragged from inside it to the right carries red into the bare
        // sheet beyond the patch, lays no pigment of its own (the film's total is conserved), and leaves the
        // sheet untouched off its path.
        let mut c = Canvas::white(80, 30, palette::ZORN, 0.9);
        let patch = Stroke { path: vec![[4.0, 15.0], [30.0, 15.0]], width0: 16.0, width1: 16.0, load: red_load(), pressure: 1.0, wetness: 0.5 };
        patch.rasterize(&mut c, &BrushConfig { k_pickup: 0.0, hold_charge: true, ..BrushConfig::default() });
        let film_before: f32 = (0..30).flat_map(|y| (0..80).map(move |x| (x, y))).map(|(x, y)| c.saturation_at(x, y)).sum();
        assert!(c.saturation_at(50, 15) < 1e-6, "bare beyond the patch");
        let before_off = c.color_at(10, 3);
        let finger = Stroke { path: vec![[20.0, 15.0], [60.0, 15.0]], width0: 8.0, width1: 8.0, load: Vec::new(), pressure: 1.0, wetness: 0.5 };
        finger.smudge(&mut c, 0.5);
        assert!(c.saturation_at(42, 15) > 0.01, "the red carried past the patch: {}", c.saturation_at(42, 15));
        assert!(c.saturation_at(42, 15) > c.saturation_at(50, 15) && c.saturation_at(50, 15) > 0.0, "fading with the travel");
        assert!(c.conc_at(42, 15)[1] > 0.0 && c.conc_at(42, 15)[0] == 0.0 && c.conc_at(42, 15)[2] == 0.0, "the carried paint is the patch's red alone");
        let film_after: f32 = (0..30).flat_map(|y| (0..80).map(move |x| (x, y))).map(|(x, y)| c.saturation_at(x, y)).sum();
        // (The finger's reservoir takes in at its memory and gives out at its share, so the sheet's total may
        // drift by a hair either way; it is moved paint, not laid paint.)
        assert!((film_after - film_before).abs() < 0.02 * film_before, "the finger moves the paint and lays none of its own: {film_before} → {film_after}");
        assert_eq!(c.color_at(10, 3), before_off, "untouched off the path");
        let lab = srgb_to_lab(c.color_at(42, 15));
        assert!(lab.a > 0.2 && lab.l < srgb_to_lab(before_off).l, "the carried colour reads red, the sheet toned by it: a* {} L {}", lab.a, lab.l);
    }
}
