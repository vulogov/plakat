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
    /// BEVEL (0..1, 0 = off; with `knife`): the THICKNESS PROFILE of a knife slice — the blade pressed down at
    /// the start lays the paint thin, and the paint it pushes ahead builds toward the lift-off into a sharp ridge
    /// (the height along the mark `0.5 + 0.8·t`, and a step of 1.2 over the last fifth); across the blade the paint
    /// squeezes out to its two edges (the outer lanes 1.8× the middle). Each a multiplier on the laid height, blended
    /// from flat by the key; the slice's colour is not touched. In the header.
    pub bevel: f32,
    /// MIX (0..1, 0 = off; a knife mark only): the blade carries TWO colours — the mark's charge and its second
    /// (`load_end`, the ramp's slot) laid in alternating BANDS of lanes across the width (two to five, by the
    /// mark's own hash), each band one colour or the other by the key's share, hard-edged, the length of the
    /// slice: the paint not fully blended on the knife, dragged as streaks. The ramp along the mark is off for it.
    pub mix: f32,
    /// FADE (0..1, 0 = off; a knife mark only): the paint on the blade RUNS OUT along the slice — the height it
    /// leaves fades toward the tail by the key (×(1 − key·t)), its deposit by half as much, the ground's texture
    /// showing through the tail of the mark. (Against the bevel's ridge at the lift-off: the two compose.)
    pub fade: f32,
    /// PILE (0..1, 0 = off; a knife mark only): the relief ACCUMULATES — the blade's shear of the standing paint
    /// is taken back by the key (at 1 nothing is sheared), and the height a slice adds saturates against what
    /// stands there (÷ (1 + key·h/2)): crossing slices pile into peaks and clots where the marks crowd, the
    /// paint never growing without bound.
    pub pile: f32,
    /// SKID (0..1, 0 = off; a knife mark only, with `fade`): the blade at LOW PRESSURE rides the standing relief
    /// — as the fade takes the pressure off toward the tail, the paint is laid only where the pixel stands above
    /// the mean of its 5×5 surround (by `skid × fade·t`), the hollows left alone: the skidding knife, the dry
    /// brush over the ridges.
    pub skid: f32,
    /// RIM (0..1, 0 = off; a knife mark only): the EDGE BEAD — a wide blade does not lay the paint flat, it
    /// squeezes it out to its two edges, so the slice's section is a U: the height at the outer lanes up to
    /// ×2.05 and the middle down to ×0.65 (a smoothstep over the outer 45% of the half-width), by the key and by
    /// how WIDE the mark is against a 60th of the sheet's short side (16 px at least) — the broad slabs get their
    /// beads of paint along the sides, a narrow slice keeps its flat section. A height multiplier only.
    pub rim: f32,
    /// SLIP (0..1, 0 = off; a knife mark only): the wide blade's UNEVEN PRESSURE — a low-frequency value noise
    /// (a period of 1.2 widths, the mark's own) gates the paint: where it falls under `0.4·key·wide` nothing is laid
    /// (over a ramp of 0.15), so the slab is torn at its edges and holed inside — the blade lifting from the
    /// canvas over a large passage. `wide` as the rim's. The pixels the slip skips are not sheared either.
    pub slip: f32,
    /// BED (0..1, 0 = off; a knife mark only): the THIN mark RIDES THE RELIEF beneath — where its paint lands,
    /// the height under it is first raised toward the local maximum of its 5×5 surround (by half the
    /// difference × the key × how THIN the mark is: `1 − wide`), a max with decay, so the thin slice bends up
    /// over the dried lumps of the layers below instead of lying across them as an appliqué: its own film is
    /// then laid on that bent bed.
    pub bed: f32,
    /// DRAG (0..1, 0 = off; a knife mark only, with `fade`): the CLOTH'S FRICTION on the fading blade — at each
    /// point of the slice the pressure is `1 − fade·t`, and the paint is gated by the cloth's TOPS (the linen's
    /// height — `Canvas::linen_height` at the canvas's `cloth_seed`, the finish's own weave: the interlock, the
    /// slubs, the fibre — six tenths) and by the threads running ACROSS the motion (the squared sine of the
    /// thread phase across the blade's direction, the warp's for a horizontal pull, the weft's for a vertical
    /// one — four tenths): the CRESTS (the sum over 0.4..0.7) take the paint whole, the FURROWS between
    /// them `(1 − key·(1 − pressure))²` of it, and nothing under 8% (a ramp to 20%) — under full pressure the
    /// slice is whole, at full key and no pressure only the thread crests take paint, the slice's plane torn into grains toward its tail. The
    /// pixels it skips are not sheared either.
    pub drag: f32,
    /// AGE (0..1, 0 = off; a knife mark only): the mark leaves the THINNESS of its blade at every pixel it lays
    /// (`Canvas::age`: `log10((short/20)/width)` clamped to 0..1 — a blade of a 20th of the short side 0, of a
    /// 200th 1), blended in by the contact. For the finish's `Finish::age`.
    pub age: f32,
    /// BRIDGE (0..1, 0 = off; a knife mark only, with `drag`): the paint's SURFACE TENSION over the cloth — the
    /// thick paint does not sink into every furrow, it spans them while there is enough of it on the blade. The
    /// paint left on the blade is the pressure `1 − fade·t`; a furrow of depth `1 − crest` (the drag's crest) is
    /// spanned whole where that volume exceeds `key × depth × (0.7..1.3, a slow noise at the mark's own period)`,
    /// over a step of 0.05: the tear is not a gradient along the pressure but a break — where the paint runs short
    /// the furrow opens at once, and the noise makes the break ragged. Below the break the drag's own gate.
    pub bridge: f32,
    /// LIFTOFF (0..1, 0 = off; a knife mark only): the RIDGE AT THE LIFT-OFF — over the last half-width of the
    /// mark the height rises by `2.5 × key × (the paint left on the blade: 1 − fade·t) × e²` (e the way along that
    /// tail): the paint still on the blade gathered into a bead across the mark where the knife leaves the canvas.
    /// A mark that ran its paint out has no bead; the painter sets it on the marks it cut hard (see
    /// `StrokeRecord::stop`), so a slice stopped at a colour's edge ends in a peak the relight catches.
    pub liftoff: f32,
    /// BROKEN (0..1, 0 = off; a knife mark only, with `drag`): the BROKEN COLOUR in the drag's voids — a pixel the
    /// drag leaves bare takes, instead of nothing, the colour of the highest paint of its 5×5 surround AS IT
    /// STOOD BEFORE THE MARK (the bed's snapshot: the dried relief below, never the mark's own fresh paint),
    /// laid as a thin film (the mark's own deposit rate × the key, no height): the skidding blade's smear of the
    /// lower layer's peaks into the hollows, the void an optical mix of the two layers and not the ground.
    pub broken: f32,
    /// LIP (0..1, 0 = off; a knife mark only, with `liftoff`): the lift-off's bead SKEWED to the very end — the
    /// paint drawn after the blade as it leaves, not heaped evenly over the tail. The tail lengthens to the mark's
    /// width (×(1 + key)), the rise over it goes from `e²` to `e^(2 + 8·key)` (the crest in the last eighth of the
    /// tail, 1.6× the even bead's height at full key), and behind the crest a shallow trough (`0.4 × key × e × (1 − g)`
    /// off the height) where the paint was pulled forward from: a sharp overhanging lip the relight's peak catches
    /// as a narrow bright line, not a fence.
    pub lip: f32,
    /// MARBLE (0..1, 0 = off; a knife mark only, with `mix`): the two colours' bands MARBLED — the band's edge not
    /// a straight line the length of the slice but wandering along it: every lane carries both colours, and the
    /// one it lays at each step is chosen by the band under a position across the blade shifted by a slow noise
    /// along the mark (±0.6 of a band at full key, a period of 1.5 widths) and breathed wider and narrower
    /// (±30%) by a second: the veins wind, pinch and swell within one plate.
    pub marble: f32,
    /// SLUMP (0..1, 0 = off; a knife mark only): the fresh paint SETTLES — the plastic impasto does not hold a
    /// vertical cliff. After the mark is laid, its own fresh height (over the bed's snapshot of the relief before
    /// it) is eased by a 3×3 binomial, and by a second pass where it stands thick, the share at each pixel the
    /// key × the local thickness against the mark's own mean: the slab's edges sag over a pixel or two, its body
    /// is kept, the thin tails hardly move. The colour is not touched, and the paint beneath is not.
    pub slump: f32,
    /// TILT (0..1, 0 = off; a knife mark only): the blade's ROLL across its motion — the hand seldom presses the
    /// knife flat. The roll of a mark is a slow spatial noise at its start (the canvas's own seed, a period of an
    /// eighth of the short side, ×1.5 and clamped, so most marks roll well one way or the other), by the key:
    /// across the blade the PRESSED edge shears the standing paint nearly to the ground (its plateau ×(1 − 0.9·k)
    /// at the edge) and lays thin (×(1 − 0.6·k)), the RAISED edge gathers the squeezed paint into one dominant
    /// bead (×(1 + 1.5·k) over its outer quarter): the section a wedge, not the rim's even U — the skewed planes
    /// of a wall or a stone.
    pub tilt: f32,
    /// STICTION (0..1, 0 = off; a knife mark only, with `liftoff`): the bead at the lift-off BROKEN ACROSS THE
    /// BLADE — the thick paint parts from the flat tool in threads, not as one line. Each lane's bead by its own
    /// hash: its height ×(0.15 + 1.7·h²) (most lanes low, a few high: discrete peaks) and its reach ×(1 − 0.5·h'),
    /// blended from the even bead by the key. The relight's peak then flickers along the line instead of ruling it.
    pub stiction: f32,
    /// SMEAR (0..1, 0 = off; a knife mark only): WET-IN-WET — the layer below the mark, where it is still wet,
    /// is DRAGGED along the blade's motion as the mark is laid over it. The canvas stamps each pixel with the tick
    /// of the smearing mark that laid it; a pixel's wetness decays by `e^(−Δ/500 marks)` and dries with the
    /// passes (`Canvas::dry`). At each pixel the mark covers, the paint of the pixel `0.3 × the width` behind it
    /// along the motion, as it stood BEFORE this mark (a snapshot), is blended in by `0.6 × key × that pixel's
    /// wetness` — the colour, the film and the height alike — before the mark's own paint is laid: the wet
    /// lower stroke shifted a fraction of a width along the new one, so crossing tufts of fur and foliage run into
    /// each other at their borders instead of sitting edge on edge.
    pub smear: f32,
    /// HEEL (0..1, 0 = off; a knife mark only): the blade's HEEL printed at the first contact — over the mark's
    /// first half-width the height rises by `2 × key × s²` (s falling from 1 at the start), a sharp bump at
    /// t = 0 where the knife comes down on the canvas; with the `tilt` the bump sits on the PRESSED edge (its
    /// share `1 − tilt·(1 − pressed)`), the raised edge's heel never touching.
    pub heel: f32,
    /// DIRTY (0..1, 0 = off; a knife mark only): the DIRTY KNIFE — the blade remembers what it carried. After each
    /// mark the canvas's blade holds half of what it held plus half of what this mark left on it: the lanes' unspent
    /// charge and the paint the plateau's shear scraped off the ridges beneath (its colour weighted by the height
    /// removed). The next mark's first half-width lays, by `key × (the residue against its own charge) × s²`, that
    /// residue's colour in place of its own, the lane's amount kept — but only where it starts within three of
    /// the last mark's widths of that mark's end (`Canvas::blade_at`): moved further, the hand wipes the knife
    /// (the marks come in no spatial order, and an unwiped blade carried the fur into the bare ground across the
    /// whole sheet). The colour of one passage carried into the starts of the next, fading by halves.
    pub dirty: f32,
    /// TEAR (0..1, 0 = off; a knife mark only): the slice's SIDES TORN — each side's edge cut in by a noise along
    /// the mark at a quarter-width's period (thresholded, so the cut is nothing for a stretch and then a tooth),
    /// up to 0.3 of the width at full key, more on a FAST slice (a length over twice the width ramps it from 0.4
    /// to 1) and toward the tail where the pressure falls (0.6 + 0.4 × the paint spent): the straight parallel
    /// flanks of the plates ragged, not ruled.
    pub tear: f32,
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
    /// KATSU (0..1, 0 = off): the dry brush by the paper's grain, its threshold rising with the mark's speed
    /// (see `MediumProfile::katsu`). In the header.
    pub katsu: f32,
    /// NIJIMI (0..1, 0 = off): the ink feathering into the paper — the reach widened, the halo caught by the
    /// paper's fibres. In the header.
    pub nijimi: f32,
    /// TURN (0..1, 0 = off): the width narrowed by up to `0.6·turn` where the path bends (fully at a right angle
    /// over half a width each way) — the brush lifted on a turn. In the header.
    pub turn: f32,
    /// JOIN (0..1, 0 = off): the bristles' spread turned WITH the path over a window of `join·width` along it
    /// (half a width each way at 1), so the stamps fan around a bend. Laid as a polyline, a mark's stamps are
    /// square to each straight segment: at every bend a wedge of bare paper is left outside the corner and a
    /// wedge is laid twice inside it, each as long as half the width — on a wide wash whose path has a control
    /// point every 68 px, a pale sliver and a dark sliver across the band at every bend (the "broken glass"
    /// panes and the pale dashes of a sumi-e's ground). A fine mark's wedges are under a pixel. And the swath laid
    /// ONCE per pixel, each lane's path filled between its stamps: the lanes' pixels are rounded, so a stamp a
    /// little turned from the last lands some of its lanes on the pixels the last one took and skips others — a
    /// moiré of dark and bare arcs sweeping across every wide wash. And the swath a 4-CONNECTED FILL — each
    /// lane's path from its last stamp, each stamp's run from the lane beside it — laid after the lane's gates:
    /// rounded sample points alone are a lattice turned against the pixel grid, which misses a pixel here and
    /// lands twice on another, and a pixel claimed by a lane that then laid nothing (the flying white, the
    /// halo's gate) stayed bare — a lattice of white pinholes over every wash. In the header.
    pub join: f32,
    /// WAVER (0..1, 0 = off): the hand's pressure wavering along the mark — a smooth noise of its own (a value
    /// noise over the mark's length, a knot every `max(12, 2.5·width)` px, cosine-eased, seeded by the mark's
    /// start) swinging the pressure by up to ±half the key (so the chalk's reach into the grain with it), the
    /// deposit by ±0.35 of it and the width by ±0.2 of it; a long ruled line in the half-tones goes lighter and
    /// darker, thinner and fuller as a pencil's does, not as a ruled line. In the header.
    pub waver: f32,
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
        Self { k_deposit: 0.34, k_pickup: 0.25, viscosity: 1.0, bristles: 7, load_max: 6.0, streak: 0.6, round: 0.7, film_cap: 0.0, flat_ends: false, hold_charge: false, deplete: 1.0, ridges: 0.0, skip: 0.0, comma: 0.0, crest: 0.0, over: 0.0, split: 0.0, tint: 0.0, knife: 0.0, bevel: 0.0, mix: 0.0, fade: 0.0, pile: 0.0, skid: 0.0, rim: 0.0, slip: 0.0, bed: 0.0, drag: 0.0, age: 0.0, bridge: 0.0, liftoff: 0.0, broken: 0.0, lip: 0.0, marble: 0.0, slump: 0.0, tilt: 0.0, stiction: 0.0, smear: 0.0, heel: 0.0, dirty: 0.0, tear: 0.0, twist: 0.0, chalk: 0.0, dust: 0.0, rewet: 0.0, tsuke: 0.0, kasure: 0.0, katsu: 0.0, nijimi: 0.0, turn: 0.0, join: 0.0, waver: 0.0 }
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
/// A VALUE NOISE in 0..1: the unit lattice's corners hashed from the seed, bilinear between them with a
/// smoothstep — slow and smooth at the scale of a unit (see `BrushConfig::slip`).
pub(crate) fn value_noise(seed: u64, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let at = |i: f32, j: f32| lane_hash(seed ^ ((i as i64 as u64).wrapping_mul(0x9E37_79B9)), (j as i64 as u64).wrapping_mul(0x85EB_CA6B) ^ 0x7F4A_7C15);
    let top = at(x0, y0) + (at(x0 + 1.0, y0) - at(x0, y0)) * sx;
    let bot = at(x0, y0 + 1.0) + (at(x0 + 1.0, y0 + 1.0) - at(x0, y0 + 1.0)) * sx;
    top + (bot - top) * sy
}

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
/// The pixels strictly between `a` and `b` along a 4-connected digital line, pushed onto `out` — none when the two
/// are further apart than `max` in either axis (the lane jumped: a new mark, not a step). See `BrushConfig::join`.
fn push_run(out: &mut Vec<(u32, u32)>, a: (i32, i32), b: (i32, i32), max: i32) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let steps = dx.abs().max(dy.abs());
    if steps < 1 || steps > max {
        return;
    }
    let mut prev = a;
    for s in 1..=steps {
        let q = (a.0 + dx * s / steps, a.1 + dy * s / steps);
        if q.0 != prev.0 && q.1 != prev.1 && q.0 >= 0 && prev.1 >= 0 {
            // A diagonal step: the corner first, so the run is 4-connected.
            out.push((q.0 as u32, prev.1 as u32));
        }
        if s < steps && q.0 >= 0 && q.1 >= 0 {
            out.push((q.0 as u32, q.1 as u32));
        }
        prev = q;
    }
}

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
        self.rasterize_inner(canvas, brush, load_end, None)
    }

    /// NIB WASH AGENT (see `MediumProfile::nib_wash_agent`): `rasterize_ramp` with the mark LOOKING AT THE SHEET —
    /// `profile` the picture's target luma (linear) at each point of `path`, and at every pixel a hair gives its
    /// ink by the residual between the sheet there and the target, `(C − T)/C · (1 + 4·k)`, nothing where the
    /// sheet has reached it.
    pub fn rasterize_agent(&self, canvas: &mut Canvas, brush: &BrushConfig, load_end: Option<&[f32]>, profile: &[f32], k: f32) {
        if profile.is_empty() {
            return self.rasterize_inner(canvas, brush, load_end, None);
        }
        self.rasterize_inner(canvas, brush, load_end, Some((profile, 1.0 + 4.0 * k.clamp(0.0, 1.0))))
    }

    fn rasterize_inner(&self, canvas: &mut Canvas, brush: &BrushConfig, load_end: Option<&[f32]>, agent: Option<(&[f32], f32)>) {
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
        // RIM / SLIP / BED (see `BrushConfig::rim`): how WIDE the mark is against a 60th of the sheet's short side.
        let wide = (maxw / (canvas.w.min(canvas.h) as f32 / 60.0).max(16.0)).clamp(0.0, 1.0);
        let slip_t = if brush.knife > 0.0 && brush.slip > 0.0 { brush.slip.clamp(0.0, 1.0) * wide } else { 0.0 };
        let slip_per = (maxw * 1.2).max(6.0);
        let bed_t = if brush.knife > 0.0 && brush.bed > 0.0 { brush.bed.clamp(0.0, 1.0) * (1.0 - wide) } else { 0.0 };
        // DRAG (see `BrushConfig::drag`): the cloth's friction on the fading blade — the linen's period as the finish's.
        let drag_k = if brush.knife > 0.0 && brush.drag > 0.0 && brush.fade > 0.0 { brush.drag.clamp(0.0, 1.0) } else { 0.0 };
        let cloth_per = (canvas.w.min(canvas.h) as f32 / 400.0).max(3.0);
        // AGE (see `BrushConfig::age`): the blade's thinness, a log scale over a decade under a 20th of the short side.
        let age_arg = if brush.knife > 0.0 && brush.age > 0.0 { ((canvas.w.min(canvas.h) as f32 / 20.0) / maxw).max(1e-3).log10().clamp(0.0, 1.0) } else { -1.0 };
        // BRIDGE / BROKEN (see `BrushConfig::bridge`, `broken`): with the drag only. LIFTOFF: the tail's length.
        let bridge_k = if drag_k > 0.0 && brush.bridge > 0.0 { brush.bridge.clamp(0.0, 1.0) } else { 0.0 };
        let broken_k = if drag_k > 0.0 && brush.broken > 0.0 { brush.broken.clamp(0.0, 1.0) } else { 0.0 };
        let lift_k = if brush.knife > 0.0 && brush.liftoff > 0.0 { brush.liftoff.clamp(0.0, 1.0) } else { 0.0 };
        // LIP (see `BrushConfig::lip`): the bead's tail lengthened toward the mark's width. SLUMP: the settling.
        let lip_k = if lift_k > 0.0 && brush.lip > 0.0 { brush.lip.clamp(0.0, 1.0) } else { 0.0 };
        let lift_len = (0.5 * maxw * (1.0 + lip_k)).max(2.0);
        let slump_k = if brush.knife > 0.0 && brush.slump > 0.0 { brush.slump.clamp(0.0, 1.0) } else { 0.0 };
        // TILT (see `BrushConfig::tilt`): the blade's roll across its motion — a slow spatial noise at the mark's start.
        let (tilt_k, tilt_sgn) = if brush.knife > 0.0 && brush.tilt > 0.0 && !pts.is_empty() {
            let per = (canvas.w.min(canvas.h) as f32 / 8.0).max(8.0);
            let roll = ((2.0 * value_noise(canvas.cloth_seed ^ 0x7117_0A11, pts[0][0] / per, pts[0][1] / per) - 1.0) * 1.5).clamp(-1.0, 1.0);
            (brush.tilt.clamp(0.0, 1.0) * roll.abs(), if roll < 0.0 { -1.0 } else { 1.0 })
        } else {
            (0.0, 1.0)
        };
        // STICTION (see `BrushConfig::stiction`): with the lift-off only. SMEAR (see `BrushConfig::smear`): the
        // wet layer below dragged a fraction of the width along the motion; the mark's pixels stamped at its end.
        let stict_k = if lift_k > 0.0 && brush.stiction > 0.0 { brush.stiction.clamp(0.0, 1.0) } else { 0.0 };
        let smear_k = if brush.knife > 0.0 && brush.smear > 0.0 { brush.smear.clamp(0.0, 1.0) } else { 0.0 };
        let smear_d = (0.3 * maxw).max(1.5);
        let mut laid_px: Vec<(u32, u32)> = Vec::new();
        if smear_k > 0.0 {
            canvas.begin_wet_mark();
        }
        // HEEL (see `BrushConfig::heel`): the bump at the first contact over the first half-width.
        let heel_k = if brush.knife > 0.0 && brush.heel > 0.0 { brush.heel.clamp(0.0, 1.0) } else { 0.0 };
        let heel_len = (0.5 * maxw).max(2.0);
        // DIRTY (see `BrushConfig::dirty`): the residue on the blade against this mark's charge, and what this
        // mark scrapes off the ridges beneath (gathered for the blade's memory at its end).
        let dirty_k = if brush.knife > 0.0 && brush.dirty > 0.0 { brush.dirty.clamp(0.0, 1.0) } else { 0.0 };
        let dirty_len = (0.5 * maxw).max(2.0);
        let charge = Self::load_norm(&self.load).max(1e-6);
        // (The knife is wiped when the hand moves on: the residue counts only within the last mark's reach.)
        let blade_res: Option<(Vec<f32>, f32)> = (dirty_k > 0.0 && !canvas.blade.is_empty() && !pts.is_empty()).then(|| {
            let nm = Self::load_norm(&canvas.blade);
            let (bx, by, reach) = canvas.blade_at;
            let near = if reach > 0.0 { (1.0 - ((pts[0][0] - bx).powi(2) + (pts[0][1] - by).powi(2)).sqrt() / reach).clamp(0.0, 1.0) } else { 0.0 };
            (canvas.blade.iter().map(|v| v.max(0.0) / nm.max(1e-9)).collect::<Vec<f32>>(), (nm / charge).clamp(0.0, 1.0) * near)
        }).filter(|(_, amt)| *amt > 1e-4);
        let mut scraped: Vec<f32> = if dirty_k > 0.0 { vec![0.0; canvas.n_pigments()] } else { Vec::new() };
        // TEAR (see `BrushConfig::tear`): the sides' cut by a fast slice.
        let tear_k = if brush.knife > 0.0 && brush.tear > 0.0 { brush.tear.clamp(0.0, 1.0) } else { 0.0 };
        let tear_amp = 0.3 * tear_k * (0.4 + 0.6 * ((pts.len() as f32 / maxw - 2.0) / 4.0).clamp(0.0, 1.0));
        let tear_per = (0.25 * maxw).max(2.0);
        // The BED reads the relief as it stands BEFORE this mark (a snapshot over the mark's reach): read live, the
        // mark's own fresh paint was the local maximum its next pixel rose toward, and the whole slice climbed.
        // (The BROKEN colour reads the same snapshot: the relief below the mark, not the mark's own; and the
        // SLUMP eases the height the mark adds over it.)
        let bed_base: Option<(Vec<f32>, i64, i64, usize, usize)> = (bed_t > 0.0 || broken_k > 0.0 || slump_k > 0.0 || smear_k > 0.0).then(|| {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for q in &pts {
                x0 = x0.min(q[0]);
                y0 = y0.min(q[1]);
                x1 = x1.max(q[0]);
                y1 = y1.max(q[1]);
            }
            let pad = maxw * 0.5 + 3.0;
            let (bx0, by0) = (((x0 - pad).floor().max(0.0)) as i64, ((y0 - pad).floor().max(0.0)) as i64);
            let (bx1, by1) = (((x1 + pad).ceil().min(canvas.w as f32 - 1.0)) as i64, ((y1 + pad).ceil().min(canvas.h as f32 - 1.0)) as i64);
            let (bw, bh) = ((bx1 - bx0 + 1).max(1) as usize, (by1 - by0 + 1).max(1) as usize);
            let mut v = vec![0f32; bw * bh];
            for yy in 0..bh {
                for xx in 0..bw {
                    let (sx, sy) = (bx0 as usize + xx, by0 as usize + yy);
                    if sx < canvas.w as usize && sy < canvas.h as usize {
                        v[yy * bw + xx] = canvas.height[sy * canvas.w as usize + sx];
                    }
                }
            }
            (v, bx0, by0, bw, bh)
        });
        let bed_arg: Option<(f32, &[f32], i64, i64, usize, usize)> = bed_base.as_ref().filter(|_| bed_t > 0.0).map(|(v, bx0, by0, bw, bh)| (bed_t, v.as_slice(), *bx0, *by0, *bw, *bh));
        let void_base: Option<(&[f32], i64, i64, usize, usize)> = bed_base.as_ref().filter(|_| broken_k > 0.0).map(|(v, bx0, by0, bw, bh)| (v.as_slice(), *bx0, *by0, *bw, *bh));
        // SMEAR (see `BrushConfig::smear`): the paint below the mark as it stands before it, over the same box.
        let smear_base: Option<(crate::paint::canvas::PaintPatch, i64, i64, usize, usize)> = bed_base.as_ref().filter(|_| smear_k > 0.0).map(|(_, bx0, by0, bw, bh)| (canvas.snapshot_paint(*bx0 as usize, *by0 as usize, *bw, *bh), *bx0, *by0, *bw, *bh));
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
        // MIX (see `BrushConfig::mix`): a knife slice's lanes in alternating bands of its two colours — each lane
        // keeps its own streak (its charge's scale against the mark's) and takes the band's colour.
        let mixed = brush.mix > 0.0 && brush.knife > 0.0 && load_end.is_some() && nb > 1;
        // MARBLE (see `BrushConfig::marble`): with it every lane carries BOTH colours (`bload2` the second), and
        // the band a lane lays at each step is chosen there, the bands' edges wandering along the mark.
        let marble_k = if mixed && brush.marble > 0.0 { brush.marble.clamp(0.0, 1.0) } else { 0.0 };
        let marble_per = (1.5 * maxw).max(8.0);
        let mut bload2: Vec<Vec<f32>> = Vec::new();
        let (mut mix_bands, mut mix_parity) = (2usize, 0usize);
        if let (true, Some(le)) = (mixed, load_end) {
            let k = brush.mix.clamp(0.0, 1.0);
            let bands = 2 + (lane_hash(seed ^ 0x3B1E, 7) * 4.0).min(3.0) as usize; // 2..5
            let parity = (lane_hash(seed ^ 0x3B1F, 11) < 0.5) as usize;
            let base = Self::load_norm(&self.load).max(1e-6);
            if marble_k > 0.0 {
                bload2 = bload
                    .iter()
                    .map(|lane| {
                        let r = Self::load_norm(lane) / base;
                        let mut l2 = lane.clone();
                        for c in 0..l2.len().min(le.len()).min(self.load.len()) {
                            l2[c] = (self.load[c] + (le[c] - self.load[c]) * k) * r;
                        }
                        l2
                    })
                    .collect();
                mix_bands = bands;
                mix_parity = parity;
            } else {
                for (b, lane) in bload.iter_mut().enumerate() {
                    let band = b * bands / nb;
                    if band % 2 != parity {
                        continue;
                    }
                    let r = Self::load_norm(lane) / base;
                    for c in 0..lane.len().min(le.len()).min(self.load.len()) {
                        lane[c] = (self.load[c] + (le[c] - self.load[c]) * k) * r;
                    }
                }
            }
        }
        let load_end = if mixed { None } else { load_end };
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

        // JOIN (see `BrushConfig::join`): the pixels this mark has laid, over its bounding box, and each lane's
        // last pixel — a pixel is laid once, and a lane's path between two stamps is filled.
        let mut seen: Option<(i64, i64, usize, usize, Vec<bool>)> = (brush.join > 0.0 && !pts.is_empty()).then(|| {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for q in &pts {
                x0 = x0.min(q[0]);
                y0 = y0.min(q[1]);
                x1 = x1.max(q[0]);
                y1 = y1.max(q[1]);
            }
            let r = maxw * 0.5 + 2.0;
            let (bx0, by0) = (((x0 - r).floor() as i64).max(0), ((y0 - r).floor() as i64).max(0));
            let (bx1, by1) = (((x1 + r).ceil() as i64).min(canvas.w as i64 - 1), ((y1 + r).ceil() as i64).min(canvas.h as i64 - 1));
            let (bw, bh) = ((bx1 - bx0 + 1).max(1) as usize, (by1 - by0 + 1).max(1) as usize);
            (bx0, by0, bw, bh, vec![false; bw * bh])
        });
        let mut lane_prev: Vec<(i32, i32)> = if seen.is_some() { vec![(i32::MIN, i32::MIN); nb] } else { Vec::new() };
        let mut targets: Vec<(u32, u32)> = Vec::with_capacity(8);
        let mut across_prev: Option<(i32, i32)>;
        // WAVER (see `BrushConfig::waver`): the knots of the pressure's noise along the mark.
        let waver = brush.waver.clamp(0.0, 1.0);
        let (wv_seed, wv_step) = (seed ^ 0x5EED_0A7E_u64, (2.5 * maxw).max(12.0));
        for i in 0..pts.len() {
            let p = pts[i];
            across_prev = None;
            // MARBLE (see `BrushConfig::marble`): the bands' drift and breathing at this step of the mark.
            let (mz, mz2) = if marble_k > 0.0 { (value_noise(seed ^ 0x3A4B_1E5, i as f32 / marble_per, 0.5) - 0.5, value_noise(seed ^ 0x3A4B_1E6, i as f32 / marble_per, 7.5) - 0.5) } else { (0.0, 0.0) };
            // TEAR (see `BrushConfig::tear`): how far each side is cut in at this step of the mark.
            // (The noise thresholded at 0.3: the cut is nothing for a stretch and then a tooth, not a wobble.)
            let (tear_l, tear_r) = if tear_k > 0.0 {
                let tooth = |nz: f32| ((nz - 0.3) / 0.7).clamp(0.0, 1.0);
                (tear_amp * tooth(value_noise(seed ^ 0x7EA4_001, i as f32 / tear_per, 0.3)), tear_amp * tooth(value_noise(seed ^ 0x7EA4_002, i as f32 / tear_per, 0.7)))
            } else {
                (0.0, 0.0)
            };
            if let (Some(step), true) = (&ramp, i > 0) {
                for lane in bload.iter_mut() {
                    for (c, d) in step.iter().enumerate() {
                        lane[c] = (lane[c] + d).max(0.0);
                    }
                }
            }
            // Local direction (forward difference), and the perpendicular the bristles spread along.
            // JOIN (see `BrushConfig::join`): the direction read over a window along the path, so the stamps fan
            // around a bend instead of switching square to each segment (a bare wedge outside, a double inside).
            let (mut dx, mut dy) = if brush.join > 0.0 && pts.len() > 2 {
                let m = ((brush.join.clamp(0.0, 1.0) * maxw * 0.5).round() as usize).max(1);
                let (a, c) = (pts[i.saturating_sub(m)], pts[(i + m).min(pts.len() - 1)]);
                (c[0] - a[0], c[1] - a[1])
            } else {
                let q = if i + 1 < pts.len() { pts[i + 1] } else { pts[i.saturating_sub(1)] };
                (q[0] - p[0], q[1] - p[1])
            };
            let dl = (dx * dx + dy * dy).sqrt();
            if dl > 1e-4 {
                dx /= dl;
                dy /= dl;
            }
            let (perpx, perpy) = (-dy, dx);
            let t = if pts.len() > 1 { i as f32 / (pts.len() - 1) as f32 } else { 0.0 };
            // NIB WASH AGENT: the target under this point of the mark, by its share of the path.
            let agent_t: Option<(f32, f32)> = agent.map(|(pf, k)| (pf[((t * (pf.len() - 1) as f32).round() as usize).min(pf.len() - 1)], k));
            // THE COMMA (see `BrushConfig::comma`): the pressure along the mark, swelling the width and the
            // deposit where the hand presses and lifting them toward the ends.
            let press = if brush.comma > 0.0 && !brush.flat_ends {
                1.0 - brush.comma.clamp(0.0, 1.0) * (1.0 - (std::f32::consts::PI * t.powf(COMMA_K)).sin().max(0.0))
            } else {
                1.0
            };
            let width = (self.width0 + (self.width1 - self.width0) * t) * (0.5 + 0.5 * press);
            // WAVER (see `BrushConfig::waver`): the noise in −1..1 at this point of the mark.
            let wv = if waver > 0.0 {
                let u = i as f32 / wv_step;
                let (k0, f) = (u.floor(), u - u.floor());
                let (a, b) = (lane_hash(wv_seed, k0 as u64), lane_hash(wv_seed, k0 as u64 + 1));
                let e = 0.5 - 0.5 * (std::f32::consts::PI * f).cos();
                waver * (2.0 * (a + (b - a) * e) - 1.0)
            } else {
                0.0
            };
            let width = if waver > 0.0 { width * (1.0 + 0.2 * wv) } else { width };
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
            let end = if waver > 0.0 { end * (1.0 + 0.35 * wv) } else { end };
            let wv_press = if waver > 0.0 { 1.0 + 0.5 * wv } else { 1.0 };
            // At a tapered/narrow width many adjacent lanes round to the SAME pixel; skip the duplicates so a
            // wide stroke doesn't re-deposit the same pixel dozens of times (the block-in's dominant cost).
            let (mut last_x, mut last_y) = (i32::MIN, i32::MIN);
            for b in 0..nb {
                let fr = (b as f32 + 0.5) / nb as f32; // 0..1 across the width
                let off = (fr - 0.5) * width;
                let x = (p[0] + perpx * off).round();
                let y = (p[1] + perpy * off).round();
                if x < 0.0 || y < 0.0 || x >= canvas.w as f32 || y >= canvas.h as f32 {
                    across_prev = None;
                    continue;
                }
                let (px, py) = (x as u32, y as u32);
                // JOIN (see `BrushConfig::join`): where this lane stood at the last stamp — its path from there
                // is filled below — kept whether or not it lays ink here.
                let lane_from = if seen.is_some() {
                    let q = lane_prev[b];
                    lane_prev[b] = (px as i32, py as i32);
                    Some(q)
                } else {
                    if px as i32 == last_x && py as i32 == last_y {
                        continue;
                    }
                    None
                };
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
                // KATSU (see `BrushConfig::katsu`): the brush's speed at this point of the mark.
                let katsu_t = if brush.katsu > 0.0 { brush.katsu.clamp(0.0, 1.0) * t.powf(1.5) } else { 0.0 };
                // BEVEL (see `BrushConfig::bevel`): a knife slice's height thin at the start, a ridge at the
                // lift-off, the paint squeezed out to the blade's two edges.
                let bevel_h = if brush.knife > 0.0 && brush.bevel > 0.0 {
                    let k = brush.bevel.clamp(0.0, 1.0);
                    let s = ((t - 0.8) / 0.2).clamp(0.0, 1.0);
                    let along = 0.5 + 0.8 * t + 1.2 * s * s * (3.0 - 2.0 * s);
                    let u = if nb > 1 { (2.0 * b as f32 / (nb - 1) as f32 - 1.0).abs() } else { 0.0 };
                    let across = 1.0 + 0.8 * ((u - 0.6) / 0.4).clamp(0.0, 1.0);
                    (1.0 - k) + k * along * across
                } else {
                    1.0
                };
                // RIM (see `BrushConfig::rim`): the wide slice's section a U — the beads of paint at its two edges.
                let rim_h = if brush.knife > 0.0 && brush.rim > 0.0 && nb > 2 && wide > 0.0 {
                    let u = (2.0 * b as f32 / (nb - 1) as f32 - 1.0).abs();
                    let s = ((u - 0.55) / 0.45).clamp(0.0, 1.0);
                    let s = s * s * (3.0 - 2.0 * s);
                    1.0 + brush.rim.clamp(0.0, 1.0) * wide * (1.4 * s - 0.35)
                } else {
                    1.0
                };
                // TILT (see `BrushConfig::tilt`): the pressed edge thin and sheared to the ground, the raised edge
                // one dominant bead — the section a wedge.
                let (tilt_h, plateau_b, pressed) = if tilt_k > 0.0 {
                    let side = (2.0 * fr - 1.0) * tilt_sgn;
                    let pressed = (-side).max(0.0);
                    let r = ((side - 0.5) / 0.5).clamp(0.0, 1.0);
                    let raised = r * r * (3.0 - 2.0 * r);
                    (1.0 - 0.6 * tilt_k * pressed + 1.5 * tilt_k * raised, if plateau >= 0.0 { plateau * (1.0 - 0.9 * tilt_k * pressed) } else { plateau }, pressed)
                } else {
                    (1.0, plateau, 0.0)
                };
                // HEEL (see `BrushConfig::heel`): the bump of the first contact, on the pressed edge under a tilt.
                let heel_h = if heel_k > 0.0 {
                    let s0 = (1.0 - i as f32 / heel_len).clamp(0.0, 1.0);
                    1.0 + 2.0 * heel_k * s0 * s0 * (1.0 - tilt_k * (1.0 - pressed))
                } else {
                    1.0
                };
                // FADE (see `BrushConfig::fade`): the paint runs out along the slice — the height toward the tail
                // by the key, the deposit by half as much.
                let fade_t = if brush.knife > 0.0 && brush.fade > 0.0 { 1.0 - brush.fade.clamp(0.0, 1.0) * t } else { 1.0 };
                // LIFTOFF (see `BrushConfig::liftoff`): the paint left on the blade gathered into a ridge across the
                // mark over its last half-width.
                let lift_h = if lift_k > 0.0 && pts.len() > 2 {
                    // STICTION (see `BrushConfig::stiction`): this lane's bead — its height and its reach by its hash.
                    let (st_amp, ll) = if stict_k > 0.0 {
                        let sn = lane_hash(seed ^ 0x57_1C7, b as u64);
                        ((1.0 - stict_k) + stict_k * (0.15 + 1.7 * sn * sn), lift_len * (1.0 - 0.5 * stict_k * lane_hash(seed ^ 0x57_1C8, b as u64)))
                    } else {
                        (1.0, lift_len)
                    };
                    let e = ((i as f32 - (pts.len() as f32 - 1.0 - ll)) / ll).clamp(0.0, 1.0);
                    if lip_k > 0.0 {
                        // LIP (see `BrushConfig::lip`): the bead skewed to the very end — a crest over the last
                        // eighth of the tail, and a shallow trough behind it where the paint was pulled from.
                        let g = e.powf(2.0 + 8.0 * lip_k);
                        (1.0 + 2.5 * lift_k * st_amp * fade_t * (1.0 + 0.6 * lip_k) * g - 0.4 * lip_k * lift_k * fade_t * e * (1.0 - g)).max(0.3)
                    } else {
                        1.0 + 2.5 * lift_k * st_amp * fade_t * e * e
                    }
                } else {
                    1.0
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
                // SLIP (see `BrushConfig::slip`): the wide blade's uneven pressure — the paint gated by a slow noise.
                let edge = if slip_t > 0.0 {
                    let nz = value_noise(seed ^ 0x5117_B1AD, x / slip_per, y / slip_per);
                    edge * ((nz - 0.4 * slip_t) / 0.15).clamp(0.0, 1.0)
                } else {
                    edge
                };
                // DRAG (see `BrushConfig::drag`): as the pressure falls along the slice, the paint settles only on
                // the crests of the cloth's threads that run across the blade's motion.
                let (edge, void_t) = if drag_k > 0.0 && fade_t < 1.0 {
                    let dt = drag_k * (1.0 - fade_t);
                    let (_, _, sx, sy) = crate::paint::canvas::linen_uv(x, y, cloth_per, canvas.cloth_seed);
                    let cloth = ((crate::paint::canvas::linen_height(x, y, cloth_per, canvas.cloth_seed) + 0.4) / 1.6).clamp(0.0, 1.0);
                    let (ax, ay) = (dx.abs(), dy.abs());
                    let across = (ax * sx * sx + ay * sy * sy) / (ax + ay).max(1e-6);
                    let crest = ((0.6 * cloth + 0.4 * across - 0.4) / 0.3).clamp(0.0, 1.0);
                    let keep = crest + (1.0 - crest) * (1.0 - dt) * (1.0 - dt);
                    // BRIDGE (see `BrushConfig::bridge`): the furrow spanned whole while the paint on the blade
                    // exceeds its depth's need — a step, ragged by a slow noise at the mark's own period.
                    let keep = if bridge_k > 0.0 {
                        let depth = 1.0 - crest;
                        let nz = value_noise(seed ^ 0xB21D_6E0, x / slip_per, y / slip_per);
                        let need = bridge_k * depth * (0.7 + 0.6 * nz);
                        let bridged = ((fade_t - need) / 0.05).clamp(0.0, 1.0);
                        bridged + (1.0 - bridged) * keep
                    } else {
                        keep
                    };
                    let gate = keep * ((keep - 0.08) / 0.12).clamp(0.0, 1.0);
                    // BROKEN (see `BrushConfig::broken`): the void takes the lower layer's colour instead of nothing.
                    if gate <= 0.0 && broken_k > 0.0 {
                        (edge, Some(broken_k))
                    } else {
                        (edge * gate, None)
                    }
                } else {
                    (edge, None)
                };
                let edge = edge * end_fade;
                // TEAR (see `BrushConfig::tear`): the lane outside either side's cut lays nothing.
                let edge = if tear_k > 0.0 {
                    let tp = 0.6 + 0.4 * (1.0 - fade_t);
                    edge * ((fr - tear_l * tp) / 0.05).clamp(0.0, 1.0) * ((1.0 - tear_r * tp - fr) / 0.05).clamp(0.0, 1.0)
                } else {
                    edge
                };
                if edge <= 0.0 {
                    across_prev = None;
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
                    across_prev = None;
                    continue;
                }
                targets.clear();
                if let Some((bx0, by0, bw, bh, seen)) = &mut seen {
                    // JOIN (see `BrushConfig::join`): the swath a 4-connected fill — this lane's path from its
                    // last stamp (up to eight pixels: a wide mark's outer lane on a turn), the stamp's run from
                    // the lane beside it, and this pixel — each pixel laid once per mark, and only by a lane
                    // that lays ink (a lane the flying white or the halo's gate silenced claims nothing).
                    let cur = (px as i32, py as i32);
                    if let Some(q) = lane_from {
                        if q.0 != i32::MIN {
                            push_run(&mut targets, q, cur, 8);
                        }
                    }
                    if let Some(a) = across_prev {
                        push_run(&mut targets, a, cur, 8);
                    }
                    targets.push((px, py));
                    across_prev = Some(cur);
                    targets.retain(|&(tx, ty)| {
                        let (ix, iy) = (tx as i64 - *bx0, ty as i64 - *by0);
                        if ix < 0 || iy < 0 || ix >= *bw as i64 || iy >= *bh as i64 {
                            return false;
                        }
                        let idx = iy as usize * *bw + ix as usize;
                        if seen[idx] {
                            false
                        } else {
                            seen[idx] = true;
                            true
                        }
                    });
                    if targets.is_empty() {
                        continue;
                    }
                } else {
                    targets.push((px, py));
                }
                // SMEAR (see `BrushConfig::smear`): the wet paint a fraction of the width behind this pixel, as it
                // stood before the mark, dragged into it by its wetness before the mark's own paint is laid.
                if let Some((patch, sbx0, sby0, sbw, sbh)) = smear_base.as_ref() {
                    let (sx, sy) = ((x - smear_d * dx).round(), (y - smear_d * dy).round());
                    if sx >= 0.0 && sy >= 0.0 && sx < canvas.w as f32 && sy < canvas.h as f32 {
                        let wt = 0.6 * smear_k * canvas.wet_at(sx as u32, sy as u32);
                        let (ix, iy) = (sx as i64 - *sbx0, sy as i64 - *sby0);
                        if wt > 0.02 && ix >= 0 && iy >= 0 && ix < *sbw as i64 && iy < *sbh as i64 {
                            canvas.smear_to(px, py, patch, iy as usize * *sbw + ix as usize, wt);
                        }
                    }
                }
                // MARBLE (see `BrushConfig::marble`): which of the two colours this lane lays at this step.
                let lane2 = marble_k > 0.0 && {
                    let u = (b as f32 / nb as f32 - 0.5) * (1.0 + 0.6 * marble_k * mz2) + 0.5 + marble_k * mz * 1.2 / mix_bands as f32;
                    ((u * mix_bands as f32).floor() as i64).rem_euclid(mix_bands as i64) as usize % 2 == mix_parity
                };
                // DIRTY (see `BrushConfig::dirty`): what the plateau's shear will scrape off here, for the blade's
                // memory; and over the first half-width the residue's colour laid in place of the lane's own.
                if dirty_k > 0.0 && plateau_b >= 0.0 {
                    let p = py as usize * canvas.w as usize + px as usize;
                    let h = canvas.height[p];
                    if h > plateau_b {
                        let d = (h - plateau_b) * brush.knife.clamp(0.0, 1.0) * (1.0 - brush.pile.clamp(0.0, 1.0));
                        for (c, v) in canvas.conc_at(px, py).iter().enumerate().take(scraped.len()) {
                            scraped[c] += d * v.max(0.0);
                        }
                    }
                }
                let w_dirty = match &blade_res {
                    Some((_, amt)) => dirty_k * amt * (1.0 - i as f32 / dirty_len).clamp(0.0, 1.0).powi(2),
                    None => 0.0,
                };
                for &(tx, ty) in &targets {
                    let lane = if lane2 { &mut bload2[b] } else { &mut bload[b] };
                    if w_dirty > 0.0 {
                        let dir = &blade_res.as_ref().unwrap().0;
                        let own = Self::load_norm(lane);
                        let mut tmp: Vec<f32> = lane.iter().enumerate().map(|(c, v)| v * (1.0 - w_dirty) + dir.get(c).copied().unwrap_or(0.0) * own * w_dirty).collect();
                        let before = Self::load_norm(&tmp).max(1e-9);
                        self.apply(canvas, tx, ty, &mut tmp, &mut bwet[b], brush, n, edge * end * keep * (0.5 + 0.5 * fade_t), deplete, &mut scratch, if brush.knife > 0.0 { bevel_h * rim_h * tilt_h * heel_h * fade_t * lift_h } else { ridge.max(0.05) }, plateau_b, wv_press, katsu_t, if brush.knife > 0.0 && brush.skid > 0.0 { brush.skid.clamp(0.0, 1.0) * (1.0 - fade_t) } else { 0.0 }, bed_arg, agent_t, age_arg, void_t.and_then(|bk| void_base.map(|(v, bx0, by0, bw, bh)| (bk, v, bx0, by0, bw, bh))));
                        let r = Self::load_norm(&tmp) / before;
                        for v in lane.iter_mut() {
                            *v *= r;
                        }
                    } else {
                        self.apply(canvas, tx, ty, lane, &mut bwet[b], brush, n, edge * end * keep * (0.5 + 0.5 * fade_t), deplete, &mut scratch, if brush.knife > 0.0 { bevel_h * rim_h * tilt_h * heel_h * fade_t * lift_h } else { ridge.max(0.05) }, plateau_b, wv_press, katsu_t, if brush.knife > 0.0 && brush.skid > 0.0 { brush.skid.clamp(0.0, 1.0) * (1.0 - fade_t) } else { 0.0 }, bed_arg, agent_t, age_arg, void_t.and_then(|bk| void_base.map(|(v, bx0, by0, bw, bh)| (bk, v, bx0, by0, bw, bh))));
                    }
                }
                if smear_k > 0.0 {
                    laid_px.push((px, py));
                }
            }
        }
        // SMEAR (see `BrushConfig::smear`): the mark's pixels stamped wet now, so its own paint was not its source.
        for &(px, py) in &laid_px {
            canvas.stamp_wet(px, py);
        }
        // DIRTY (see `BrushConfig::dirty`): the blade's memory — half of what it held, and half of what this mark
        // left on it: the lanes' unspent charge and the scraped paint, each as a charge's worth of its colour.
        if dirty_k > 0.0 {
            let n = canvas.n_pigments();
            let mut left = vec![0f32; n];
            for lane in &bload {
                for c in 0..n.min(lane.len()) {
                    left[c] += lane[c].max(0.0) / nb as f32;
                }
            }
            let sn = Self::load_norm(&scraped);
            let picked: Vec<f32> = (0..n).map(|c| if sn > 1e-9 { 0.5 * left[c] + 0.5 * charge * scraped[c] / sn } else { left[c] }).collect();
            if canvas.blade.len() != n {
                canvas.blade = vec![0.0; n];
            }
            for c in 0..n {
                canvas.blade[c] = 0.5 * canvas.blade[c] + 0.5 * picked[c];
            }
            if let Some(last) = pts.last() {
                canvas.blade_at = (last[0], last[1], 3.0 * maxw);
            }
        }
        // SLUMP (see `BrushConfig::slump`): the fresh paint settles — the height this mark added over the bed's
        // snapshot eased by a 3×3 binomial (and a second pass where it stands thick), the share at each pixel the
        // key × the local thickness against the mark's own mean: the cliffs at the slab's edges become slopes over
        // a pixel or two, the body keeps its height. The height alone; the colour and the paint beneath untouched.
        if slump_k > 0.0 {
            if let Some((base, bx0, by0, bw, bh)) = bed_base.as_ref() {
                let (bw, bh, w, h) = (*bw, *bh, canvas.w as usize, canvas.h as usize);
                let mut fresh = vec![0f32; bw * bh];
                let (mut sum, mut cnt) = (0f32, 0usize);
                for yy in 0..bh {
                    for xx in 0..bw {
                        let (sx, sy) = (*bx0 as usize + xx, *by0 as usize + yy);
                        if sx < w && sy < h {
                            let f = canvas.height[sy * w + sx] - base[yy * bw + xx];
                            if f > 1e-6 {
                                fresh[yy * bw + xx] = f;
                                sum += f;
                                cnt += 1;
                            }
                        }
                    }
                }
                if cnt > 0 {
                    let tref = (sum / cnt as f32).max(1e-6);
                    let blur = |src: &[f32]| -> Vec<f32> {
                        let mut out = vec![0f32; bw * bh];
                        for yy in 0..bh {
                            for xx in 0..bw {
                                let (mut acc, mut wsum) = (0f32, 0f32);
                                for dy in -1i64..=1 {
                                    for dx in -1i64..=1 {
                                        let (qx, qy) = (xx as i64 + dx, yy as i64 + dy);
                                        if qx < 0 || qy < 0 || qx >= bw as i64 || qy >= bh as i64 {
                                            continue;
                                        }
                                        let wgt = (if dx == 0 { 2.0 } else { 1.0 }) * (if dy == 0 { 2.0 } else { 1.0 });
                                        acc += wgt * src[qy as usize * bw + qx as usize];
                                        wsum += wgt;
                                    }
                                }
                                out[yy * bw + xx] = acc / wsum;
                            }
                        }
                        out
                    };
                    let b1 = blur(&fresh);
                    let b2 = blur(&b1);
                    for yy in 0..bh {
                        for xx in 0..bw {
                            let (sx, sy) = (*bx0 as usize + xx, *by0 as usize + yy);
                            if sx >= w || sy >= h {
                                continue;
                            }
                            let i = yy * bw + xx;
                            let sk = (b1[i] / tref).clamp(0.0, 1.0) * slump_k;
                            let eased = fresh[i] + sk * (b1[i] - fresh[i]) + sk * sk * (b2[i] - b1[i]);
                            let p = sy * w + sx;
                            canvas.height[p] = (canvas.height[p] + eased - fresh[i]).max(0.0);
                        }
                    }
                }
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
    fn apply(&self, canvas: &mut Canvas, px: u32, py: u32, load: &mut [f32], bwet: &mut f32, brush: &BrushConfig, n: usize, cover: f32, deplete: f32, deposit: &mut Vec<f32>, ridge: f32, plateau: f32, press_k: f32, katsu_t: f32, skid_t: f32, bed: Option<(f32, &[f32], i64, i64, usize, usize)>, agent_t: Option<(f32, f32)>, age_t: f32, void: Option<(f32, &[f32], i64, i64, usize, usize)>) {
        let p = py as usize * canvas.w as usize + px as usize;
        if canvas.clipped(p) {
            return;
        }
        if let Some((bk, base, bx0, by0, bw, bh)) = void {
            // BROKEN (see `BrushConfig::broken`): the drag's void — the blade did not touch here (no shear, no
            // pickup); the pixel takes a thin film of the colour of the highest paint of its 5×5 surround as it
            // stood before the mark, where that stands above the pixel's own.
            let (cx, cy) = (px as i64 - bx0, py as i64 - by0);
            let own = if cx >= 0 && cy >= 0 && (cx as usize) < bw && (cy as usize) < bh { base[cy as usize * bw + cx as usize] } else { 0.0 };
            let (mut best, mut top) = (None, own);
            for yy in (cy - 2).max(0)..=(cy + 2).min(bh as i64 - 1) {
                for xx in (cx - 2).max(0)..=(cx + 2).min(bw as i64 - 1) {
                    let v = base[yy as usize * bw + xx as usize];
                    if v > top {
                        top = v;
                        best = Some((xx as usize, yy as usize));
                    }
                }
            }
            if let Some((bx, by)) = best {
                if top > own + 1e-4 {
                    let (sx, sy) = ((bx0 as usize + bx) as u32, (by0 as usize + by) as u32);
                    let contact = (self.pressure * cover.clamp(0.0, 1.0) * canvas.tooth[p]).clamp(0.0, 1.0);
                    let amount = brush.k_deposit * contact * bk * Self::load_norm(load);
                    deposit.clear();
                    deposit.extend_from_slice(canvas.conc_at(sx, sy));
                    let sum: f32 = deposit.iter().copied().map(|c| c.max(0.0)).sum();
                    if amount > 1e-6 && sum > 1e-6 {
                        for c in deposit.iter_mut() {
                            *c = c.max(0.0) / sum * amount;
                        }
                        canvas.deposit(px, py, deposit, 0.0);
                    }
                }
            }
            return;
        }
        if brush.knife > 0.0 && plateau >= 0.0 {
            // The blade shears the standing paint down toward the plateau before laying its own.
            // (PILE: the shear taken back by the key — the relief kept to pile on.)
            let h = canvas.height[p];
            if h > plateau {
                canvas.height[p] = plateau + (h - plateau) * (1.0 - brush.knife.clamp(0.0, 1.0) * (1.0 - brush.pile.clamp(0.0, 1.0)));
            }
        }
        // PILE (see `BrushConfig::pile`): the height a slice adds saturates against what stands there.
        let ridge = if brush.knife > 0.0 && brush.pile > 0.0 { ridge / (1.0 + brush.pile.clamp(0.0, 1.0) * canvas.height[p].max(0.0) / 2.0) } else { ridge };
        let tooth = canvas.tooth[p];
        // (`press_k`: the WAVER's pressure at this point, 1 without it.)
        let pressure = if press_k != 1.0 { (self.pressure * press_k).clamp(0.0, 1.0) } else { self.pressure };
        let contact = (pressure * cover.clamp(0.0, 1.0) * tooth).clamp(0.0, 1.0);
        let sat = (canvas.saturation_at(px, py) / SAT_FULL).clamp(0.0, 1.0);

        // Deposit: a fraction of the current load, throttled by contact and remaining tooth. `deposit` is a
        // caller-owned scratch buffer, cleared here — no per-pixel heap allocation.
        let mut df = (brush.k_deposit * contact * (1.0 - SAT_THROTTLE * sat)).clamp(0.0, 1.0);
        // NIB WASH AGENT (see `Stroke::rasterize_agent`): the hair gives its ink by the residual between the sheet
        // under it, as it is now, and the target — nothing where the sheet has reached it.
        if let Some((tg, k)) = agent_t {
            let c = crate::paint::color::linear_luma(crate::paint::color::srgb_to_linear(canvas.color_at(px, py)));
            let f = ((c - tg) / c.max(1e-3) * k).clamp(0.0, 1.0);
            if f <= 1e-4 {
                return;
            }
            df *= f;
        }
        // The dry brush: a mark laid below 0.6 wet no longer floods the hollows; the drier it is, the higher
        // the fibre it must find to leave paint. The paper's relief is the same deterministic grain the
        // granulation settles into, so the skips and the mottle share one tooth. Read from the mark's OWN
        // wetness, not the bristle's running one: that decays along every stroke, and read there the tail of
        // every long wash went dry and the whole sheet stippled and darkened.
        // KATSU (see `BrushConfig::katsu`): the threshold raised by the mark's speed, whatever its water.
        if brush.skip > 0.0 || katsu_t > 0.0 {
            // ...and on a mark laid DRY, a bristle whose charge has run out — the dry tail of a depleting
            // mark — reaches only the raised fibres, so the tail breaks on the tooth instead of thinning
            // evenly. A wet mark is never broken, charge or no charge: water floods the hollows. (Full from
            // 0.45 wet down, fading out to nothing at 0.6.)
            let base = ((0.6 - self.wetness) / 0.6).clamp(0.0, 1.0);
            let spent = 1.0 - (Self::load_norm(load) / Self::load_norm(&self.load).max(1e-6)).clamp(0.0, 1.0);
            let dryness = base.max(spent * spent * (base * 4.0).min(1.0));
            let thr = (brush.skip.clamp(0.0, 1.0) * dryness * 0.9).max(katsu_t * 0.9);
            if thr > 0.0 {
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
            let reach = ((pressure - 0.3) / 0.55).clamp(0.0, 1.0);
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
        if skid_t > 0.0 {
            // SKID (see `BrushConfig::skid`): the blade at low pressure rides the standing relief — the paint
            // laid only where this pixel stands above the mean of its 5×5 surround, the hollows left alone.
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
            let t = ((canvas.height[p] - mean) / (0.35 * mean + 1e-4)).clamp(0.0, 1.0);
            let catch = t * t * (3.0 - 2.0 * t);
            df *= 1.0 - skid_t.clamp(0.0, 1.0) * (1.0 - catch);
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
        if let Some((bed_t, base, bx0, by0, bw, bh)) = bed {
            if df > 0.0 {
                // BED (see `BrushConfig::bed`): the thin mark rides the relief beneath — the height under its paint
                // raised toward the local maximum of the 5×5 surround AS IT STOOD BEFORE THE MARK (the snapshot) by
                // half the difference × the key: a max with decay, the slice bending up over the lumps of the
                // layers below, never over its own fresh paint.
                let (cx, cy) = (px as i64 - bx0, py as i64 - by0);
                let mut lm = f32::MIN;
                for yy in (cy - 2).max(0)..=(cy + 2).min(bh as i64 - 1) {
                    for xx in (cx - 2).max(0)..=(cx + 2).min(bw as i64 - 1) {
                        lm = lm.max(base[yy as usize * bw + xx as usize]);
                    }
                }
                let hp = canvas.height[p];
                if lm > hp {
                    canvas.height[p] = hp + (lm - hp) * 0.5 * bed_t.clamp(0.0, 1.0);
                }
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
        if age_t >= 0.0 {
            // AGE (see `BrushConfig::age`): the blade's thinness left at the pixel, blended in by the contact.
            if canvas.age.is_empty() {
                canvas.age = vec![0.0; canvas.w as usize * canvas.h as usize];
            }
            canvas.age[p] += (age_t - canvas.age[p]) * contact;
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
    fn an_agent_mark_gives_its_ink_by_the_residual_and_none_where_the_sheet_has_the_tone() {
        // A mark across a white sheet with a profile: the target 0.3 over its first half, the paper's 1.0 over
        // the second. Ink lands on the first half only; a second pass over it adds less than the first did.
        let pal = crate::paint::palette::SUMI;
        let ink = crate::paint::canvas::darkest_pigment(&pal);
        let mut load = vec![0f32; pal.pigments.len()];
        load[ink] = 3.0;
        let brush = BrushConfig::default();
        let st = Stroke { path: vec![[4.0, 10.0], [36.0, 10.0]], width0: 6.0, width1: 6.0, load, pressure: 1.0, wetness: 0.5 };
        let prof = vec![0.3f32, 1.0];
        let mut c = Canvas::white(40, 20, pal, 0.85);
        let luma = |c: &Canvas, x: u32, y: u32| crate::paint::color::linear_luma(crate::paint::color::srgb_to_linear(c.color_at(x, y)));
        // (the white sheet's own luma is a little under 1: the paper, not a lamp)
        let blank = luma(&c, 32, 10);
        st.rasterize_agent(&mut c, &brush, None, &prof, 1.0);
        let first = luma(&c, 10, 10);
        assert!(first < blank - 0.05, "ink laid where the target is dark ({first:.3} on {blank:.3})");
        assert!((luma(&c, 32, 10) - blank).abs() < 1e-6, "none where the sheet already has the target's tone ({:.3} vs {blank:.3})", luma(&c, 32, 10));
        st.rasterize_agent(&mut c, &brush, None, &prof, 1.0);
        let second = luma(&c, 10, 10);
        assert!(first - second < 0.5 * (1.0 - first), "the second mark adds less than the first ({first:.3} → {second:.3})");
        assert!(second >= 0.3 - 0.05, "and the sheet does not run past the target ({second:.3})");
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
    fn a_bevelled_knife_slice_is_thin_at_the_start_and_a_ridge_at_the_lift_off() {
        // One knife mark left to right on a white sheet: with the bevel the paint stands higher at the lift-off
        // than at the start and higher along the blade's edges than down its middle; without it the slice is flat.
        let lay = |bevel: f32| {
            let mut c = Canvas::white(80, 40, palette::ZORN, 0.9);
            let mark = Stroke { path: vec![[10.0, 20.0], [70.0, 20.0]], width0: 10.0, width1: 10.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            // (No depletion: the paint a knife pushes ahead does not run out along the slice as a brush's does.)
            mark.rasterize(&mut c, &BrushConfig { knife: 1.0, bevel, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let h = |x: usize, y: usize| c.height[y * 80 + x];
            (h(16, 20), h(62, 20), h(40, 20), h(40, 16).max(h(40, 24)))
        };
        let (start, end, mid, edge) = lay(1.0);
        assert!(end > start * 2.0, "the ridge at the lift-off ({end} vs {start} at the start)");
        assert!(edge > mid * 1.3, "the paint squeezed out to the blade's edges ({edge} vs {mid} down the middle)");
        let (start0, end0, _, _) = lay(0.0);
        assert!((end0 - start0).abs() < 0.15 * start0.max(end0), "flat without the bevel ({start0} vs {end0})");
    }

    #[test]
    fn piled_knife_slices_stand_higher_where_they_cross_and_the_skidding_tail_rides_the_peaks() {
        // Two knife slices crossing: without the pile the second shears the first to its plateau and the
        // crossing stands no higher than one slice; with it the crossing piles up. Then a tail at low pressure
        // (fade) over a standing ridge: with the skid it lays its paint on the ridge and not beside it.
        let knife = |pile: f32, fade: f32, skid: f32| BrushConfig { knife: 1.0, pile, fade, skid, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() };
        let cross = |pile: f32| {
            let mut c = Canvas::white(80, 80, palette::ZORN, 0.9);
            Stroke { path: vec![[8.0, 40.0], [72.0, 40.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife(pile, 0.0, 0.0));
            c.dry(0.0);
            Stroke { path: vec![[40.0, 8.0], [40.0, 72.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife(pile, 0.0, 0.0));
            (c.height[40 * 80 + 40], c.height[40 * 80 + 20])
        };
        let ((x0, s0), (x1, s1)) = (cross(0.0), cross(1.0));
        assert!(x0 < s0 * 1.2, "unpiled, the crossing stands no higher than a single slice ({x0} vs {s0})");
        assert!(x1 > s1 * 1.4, "piled, the crossing stands higher ({x1} vs {s1})");
        let tail = |skid: f32| {
            let mut c = Canvas::white(80, 40, palette::ZORN, 0.9);
            Stroke { path: vec![[58.0, 5.0], [58.0, 35.0]], width0: 6.0, width1: 6.0, load: vec![0.0, 0.0, 0.0, 4.0], pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &BrushConfig { viscosity: 3.0, k_pickup: 0.0, streak: 0.0, round: 0.0, deplete: 0.0, ..BrushConfig::default() });
            c.dry(0.0);
            Stroke { path: vec![[8.0, 20.0], [72.0, 20.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife(1.0, 1.0, skid));
            let img = c.to_image();
            let red = |x: u32| { let p = img.get_pixel(x, 20).0; p[0] as f32 - p[2] as f32 };
            (red(58), red(66))
        };
        let ((on0, off0), (on1, off1)) = (tail(0.0), tail(1.0));
        assert!(off1 < off0 * 0.7, "the skidding tail leaves the hollow beside the ridge alone ({off1} vs {off0})");
        assert!(on1 > off1 * 1.5 && on1 > on0 * 0.6, "and lays its paint on the ridge ({on1} vs {on0}, beside {off1})");
    }

    #[test]
    fn the_value_noise_is_bounded_and_slow() {
        // Samples in 0..1, and a step of a twentieth of a unit changes the value a little.
        let mut worst = 0f32;
        for i in 0..400 {
            let (x, y) = (i as f32 * 0.05, (i % 7) as f32 * 0.37);
            let v = value_noise(42, x, y);
            assert!((0.0..=1.0).contains(&v), "bounded ({v})");
            worst = worst.max((value_noise(42, x + 0.05, y) - v).abs());
        }
        assert!(worst < 0.12, "slow ({worst})");
    }

    #[test]
    fn a_wide_knife_slice_with_the_rim_stands_higher_at_its_edges_than_in_its_middle() {
        // A wide knife slice: flat in section without the rim; with it the outer lanes stand well above the
        // middle — the beads of paint squeezed out to the blade's two edges.
        let section = |rim: f32| {
            let mut c = Canvas::white(80, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[8.0, 20.0], [72.0, 20.0]], width0: 16.0, width1: 16.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, rim, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let row = |y: usize| (20..60).map(|x| c.height[y * 80 + x]).sum::<f32>() / 40.0;
            ((row(13) + row(28)) / 2.0, (row(20) + row(21)) / 2.0)
        };
        let ((e0, m0), (e1, m1)) = (section(0.0), section(1.0));
        assert!(e0 < m0 * 1.25 && e0 > m0 * 0.75, "without the rim the section is flat (edge {e0} vs middle {m0})");
        assert!(e1 > m1 * 1.8, "with the rim the edges stand well above the middle (edge {e1} vs middle {m1})");
    }

    #[test]
    fn the_slipping_wide_blade_leaves_holes_in_its_slab() {
        // A wide knife slice over a white canvas: without the slip every pixel of the swath takes paint; with it
        // a share of them stays bare — the blade's uneven pressure over a large passage.
        let bare = |slip: f32| {
            let mut c = Canvas::white(120, 60, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[10.0, 30.0], [110.0, 30.0]], width0: 24.0, width1: 24.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, slip, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let mut n = 0usize;
            for y in 20..40 {
                for x in 25..95 {
                    if c.height[y * 120 + x] < 1e-5 {
                        n += 1;
                    }
                }
            }
            n as f32 / (20.0 * 70.0)
        };
        let (b0, b1) = (bare(0.0), bare(1.0));
        assert!(b0 < 0.02, "without the slip the swath is solid ({b0} bare)");
        assert!(b1 > 0.08 && b1 < 0.7, "with the slip a share of the slab is holed ({b1} bare)");
    }

    #[test]
    fn a_thin_knife_slice_with_the_bed_rides_up_over_a_lump_beneath() {
        // A thick lump dried on the canvas, then a thin knife slice across it: without the bed the slice beside the
        // lump lies at its own height; with it the height beside the lump is raised toward the lump's — the thin
        // mark bending up over the relief beneath.
        let beside = |bed: f32| {
            let mut c = Canvas::white(80, 40, palette::ZORN, 0.9);
            Stroke { path: vec![[40.0, 12.0], [40.0, 28.0]], width0: 6.0, width1: 6.0, load: vec![0.0, 0.0, 0.0, 4.0], pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &BrushConfig { viscosity: 4.0, k_pickup: 0.0, streak: 0.0, round: 0.0, deplete: 0.0, ..BrushConfig::default() });
            c.dry(0.0);
            Stroke { path: vec![[8.0, 20.0], [72.0, 20.0]], width0: 4.0, width1: 4.0, load: red_load(), pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &BrushConfig { knife: 1.0, pile: 1.0, bed, viscosity: 1.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            ((19..22).map(|y| c.height[y * 80 + 45]).sum::<f32>() / 3.0, (19..22).map(|y| c.height[y * 80 + 20]).sum::<f32>() / 3.0)
        };
        let ((near0, far0), (near1, far1)) = (beside(0.0), beside(1.0));
        assert!((near0 - far0).abs() < far0 * 0.5, "without the bed the slice beside the lump lies at its own height ({near0} vs {far0})");
        assert!(near1 > near0 * 1.3 && (far1 - far0).abs() < far0 * 0.3, "with the bed the slice rises beside the lump and not away from it ({near1} vs {near0}; far {far1} vs {far0})");
    }

    #[test]
    fn a_fading_knife_slice_stands_lower_at_its_tail() {
        // The same knife slice with and without the fade: near the tail the faded one stands well under the
        // other; near the start they are alike.
        let height = |fade: f32| {
            let mut c = Canvas::white(80, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[8.0, 20.0], [72.0, 20.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, fade, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let at = |x: usize| (17..23).map(|y| c.height[y * 80 + x]).sum::<f32>() / 6.0;
            (at(14), at(62))
        };
        let ((s0, e0), (s1, e1)) = (height(0.0), height(1.0));
        assert!(e1 < e0 * 0.5, "the faded slice's tail stands under half the plain one's ({e1} vs {e0})");
        assert!(s1 > s0 * 0.8, "and its start is nearly as high ({s1} vs {s0})");
    }

    #[test]
    fn a_dragged_knife_slice_is_grainy_at_its_tail_and_whole_at_its_start() {
        // A fading knife slice with and without the drag: toward the tail the dragged one lays its paint only on
        // the cloth's thread crests (fewer pixels take paint); at the start both lay every pixel.
        let laid = |drag: f32| {
            let mut c = Canvas::white(160, 40, palette::ZORN, 0.9).with_cloth_seed(7);
            let s = Stroke { path: vec![[8.0, 20.0], [152.0, 20.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, fade: 1.0, drag, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let count = |x0: usize, x1: usize| (17..23).flat_map(|y| (x0..x1).map(move |x| y * 160 + x)).filter(|&i| c.height[i] > 1e-5).count();
            (count(12, 40), count(112, 148))
        };
        let ((s0, e0), (s1, e1)) = (laid(0.0), laid(1.0));
        assert!(s1 as f32 > s0 as f32 * 0.85, "the start, under full pressure, is nearly whole either way ({s1} vs {s0})");
        assert!((e1 as f32) < e0 as f32 * 0.75, "the dragged tail is grainy: fewer pixels take paint ({e1} vs {e0})");
    }

    #[test]
    fn the_age_keeps_the_thinness_of_the_last_knife_at_each_pixel() {
        // A wide slice across, then a thin one down over it, both with the age: under the wide slice alone the
        // age is low, where the thin one crossed it is high; without the key no age is kept at all.
        let mut c = Canvas::white(256, 256, palette::ZORN, 0.9);
        let knife = |age: f32| BrushConfig { knife: 1.0, age, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() };
        let wide = Stroke { path: vec![[20.0, 128.0], [236.0, 128.0]], width0: 12.0, width1: 12.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
        wide.rasterize(&mut c, &knife(0.0));
        assert!(c.age.is_empty(), "no key, no age");
        wide.rasterize(&mut c, &knife(1.0));
        Stroke { path: vec![[128.0, 20.0], [128.0, 236.0]], width0: 2.0, width1: 2.0, load: red_load(), pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife(1.0));
        let at = |x: usize, y: usize| c.age[y * 256 + x];
        assert!(at(60, 128) < 0.1, "under the wide knife alone the age is low ({})", at(60, 128));
        let cross = at(128, 128).max(at(129, 128));
        assert!(cross > 0.5, "where the thin knife crossed it is high ({cross})");
        let along = at(128, 60).max(at(129, 60));
        assert!(along > 0.5, "and along the thin slice ({along})");
    }

    #[test]
    fn the_bridge_spans_the_shallow_furrows_while_the_paint_lasts() {
        // A fading dragged slice with and without the bridge: in its middle, where the paint still spans the
        // shallow furrows, the bridged one stands higher on average (the furrows filled); at the very tail both
        // are torn alike.
        let mean_h = |bridge: f32| {
            let mut c = Canvas::white(160, 40, palette::ZORN, 0.9).with_cloth_seed(7);
            let s = Stroke { path: vec![[8.0, 20.0], [152.0, 20.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, fade: 1.0, drag: 1.0, bridge, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let m = |x0: usize, x1: usize| (17..23).flat_map(|y| (x0..x1).map(move |x| y * 160 + x)).map(|i| c.height[i]).sum::<f32>() / (6 * (x1 - x0)) as f32;
            (m(56, 96), m(140, 150))
        };
        let ((m0, e0), (m1, e1)) = (mean_h(0.0), mean_h(1.0));
        assert!(m1 > m0 * 1.05, "the bridged middle stands higher ({m1} vs {m0})");
        assert!((e1 - e0).abs() < e0 * 0.5 + 1e-3, "the tail is torn alike ({e1} vs {e0})");
    }

    #[test]
    fn the_liftoff_leaves_a_bead_where_the_blade_leaves_the_canvas() {
        // A fading knife slice with and without the lift-off: over its last few pixels the one with it stands
        // well above the plain one, and both are alike a width before the end.
        let at = |liftoff: f32| {
            let mut c = Canvas::white(160, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[8.0, 20.0], [152.0, 20.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, fade: 0.5, liftoff, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let h = |x: usize| (17..23).map(|y| c.height[y * 160 + x]).sum::<f32>() / 6.0;
            (h(140), h(151))
        };
        let ((b0, e0), (b1, e1)) = (at(0.0), at(1.0));
        assert!(e1 > e0 * 1.5, "the bead stands at the lift-off ({e1} vs {e0})");
        assert!((b1 - b0).abs() < b0 * 0.1 + 1e-3, "a width before the end the slices are alike ({b1} vs {b0})");
    }

    #[test]
    fn the_broken_colour_fills_the_drags_voids_with_the_lower_layers_peaks() {
        // A thick black slice dried below (its top lane on row 19); a white fading slice dragged over the canvas
        // just above it (rows 13..20): in the tail, the voids the drag leaves on row 17 — the black relief within
        // their 5×5 — take its colour with the key, and stay the ground's white without.
        let row = |broken: f32| {
            let mut c = Canvas::white(160, 48, palette::ZORN, 0.9).with_cloth_seed(7);
            let black = vec![0.0, 0.0, 4.0, 0.0];
            let white = vec![0.0, 0.0, 0.0, 4.0];
            Stroke { path: vec![[96.0, 24.0], [152.0, 24.0]], width0: 12.0, width1: 12.0, load: black, pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &BrushConfig { knife: 1.0, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            Stroke { path: vec![[8.0, 16.0], [152.0, 16.0]], width0: 8.0, width1: 8.0, load: white, pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &BrushConfig { knife: 1.0, fade: 1.0, drag: 1.0, broken, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            (112..148).map(|x| c.color_at(x, 17)[1] as f32).sum::<f32>() / 36.0
        };
        let (plain, broken) = (row(0.0), row(1.0));
        assert!(broken < plain - 8.0, "the voids beside the black relief take its colour ({broken} vs {plain})");
    }

    #[test]
    fn the_heel_prints_a_bump_at_the_first_contact() {
        // A flat knife slice with and without the heel: with it the height a pixel into the mark stands well
        // above the height at its middle; without, the two are alike.
        let ratio = |heel: f32| {
            let mut c = Canvas::white(120, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[10.0, 20.0], [110.0, 20.0]], width0: 12.0, width1: 12.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, heel, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let h = |x: usize| (16..25).map(|y| c.height[y * 120 + x]).sum::<f32>() / 9.0;
            h(11) / h(60).max(1e-6)
        };
        let (plain, heeled) = (ratio(0.0), ratio(1.0));
        assert!(plain < 1.05, "a flat slice starts as high as its middle ({plain})");
        assert!(heeled > 1.8, "the heel prints a bump at the first contact ({heeled})");
    }

    #[test]
    fn the_dirty_knife_carries_the_last_marks_colour_into_the_next_marks_start() {
        // A black slice, then a red one starting near its end (not touching it) with the same knife: with the key
        // the red one starts darker than its middle (the black still on the blade); without, its start is its
        // own colour; and with the key but starting far from the black's end, the knife was wiped.
        let start_vs_mid = |dirty: f32, far: bool| {
            let mut c = Canvas::white(160, 60, palette::ZORN, 0.9);
            let knife = BrushConfig { knife: 1.0, dirty, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() };
            Stroke { path: vec![[10.0, 12.0], [150.0, 12.0]], width0: 16.0, width1: 16.0, load: vec![0.0, 0.0, 4.0, 0.0], pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife);
            let (x0, x1) = if far { (10.0, 150.0) } else { (150.0, 10.0) };
            Stroke { path: vec![[x0, 34.0], [x1, 34.0]], width0: 10.0, width1: 10.0, load: red_load(), pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife);
            let luma = |x: u32| (30..39).map(|y| { let p = c.color_at(x, y); 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32 }).sum::<f32>() / 9.0;
            (luma(if far { 11 } else { 149 }), luma(80))
        };
        let ((s0, m0), (s1, m1), (s2, m2)) = (start_vs_mid(0.0, false), start_vs_mid(1.0, false), start_vs_mid(1.0, true));
        assert!((s0 - m0).abs() < 2.0, "a clean knife starts in its own colour ({s0} vs {m0})");
        assert!(s1 < m1 - 8.0, "the dirty knife starts darker with the black it carried ({s1} vs {m1})");
        assert!((s2 - m2).abs() < 2.0, "moved far from the last mark the knife is wiped ({s2} vs {m2})");
    }

    #[test]
    fn the_torn_slice_has_ragged_sides() {
        // A long slice with and without the tear: the row its edge lane lays is laid whole without it and
        // broken with it, while the middle row is laid whole either way.
        let rows = |tear: f32| {
            let mut c = Canvas::white(120, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[10.0, 20.0], [110.0, 20.0]], width0: 12.0, width1: 12.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, tear, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let laid = |y: usize| (20..100).filter(|&x| c.height[y * 120 + x] > 1e-5).count();
            (laid(15), laid(20))
        };
        let ((e0, m0), (e1, m1)) = (rows(0.0), rows(1.0));
        assert!(e0 == 80 && m0 == 80, "the plain slice lays its rows whole ({e0}, {m0})");
        assert!(m1 == 80, "the torn slice keeps its middle whole ({m1})");
        assert!(e1 < 68, "the torn slice's edge row is broken ({e1})");
    }

    #[test]
    fn the_tilted_blade_leaves_one_edge_thin_and_the_other_a_bead() {
        // A knife slice with the tilt over a few cloth seeds (the roll is a spatial noise): with the key some roll
        // leaves the slice's two edge rows unequal by half again; without it the edges stand alike for every seed.
        let edges = |tilt: f32, seed: u64| {
            let mut c = Canvas::white(120, 40, palette::ZORN, 0.9).with_cloth_seed(seed);
            let s = Stroke { path: vec![[10.0, 20.0], [110.0, 20.0]], width0: 12.0, width1: 12.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, tilt, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let h = |y: usize| (30..90).map(|x| c.height[y * 120 + x]).sum::<f32>() / 60.0;
            let (a, b) = (h(16), h(24));
            a.max(b) / a.min(b).max(1e-6)
        };
        assert!((0..12).all(|s| edges(0.0, s) < 1.05), "without the tilt the edges stand alike");
        assert!((0..12).any(|s| edges(1.0, s) > 1.5), "with the tilt some roll leaves one edge a bead and the other thin");
    }

    #[test]
    fn the_stiction_breaks_the_bead_into_peaks_across_the_blade() {
        // The lip's bead with and without the stiction: at the lift-off the heights across the lanes are alike
        // without it, and spread by a factor with it.
        let spread = |stiction: f32| {
            let mut c = Canvas::white(160, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[8.0, 20.0], [152.0, 20.0]], width0: 10.0, width1: 10.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, fade: 0.5, liftoff: 1.0, lip: 1.0, stiction, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let hs: Vec<f32> = (16..25).map(|y| c.height[y * 160 + 152]).collect();
            hs.iter().cloned().fold(0.0f32, f32::max) / hs.iter().cloned().fold(f32::MAX, f32::min).max(1e-6)
        };
        let (plain, broken) = (spread(0.0), spread(1.0));
        assert!(plain < 1.1, "the even bead stands alike across the blade ({plain})");
        assert!(broken > 1.8, "the stiction parts the bead into peaks ({broken})");
    }

    #[test]
    fn the_wet_paint_under_the_blade_is_dragged_along_its_motion() {
        // A black slice down the canvas; a white one across it, laid next with and without the smear, and with
        // the smear after the canvas dried: just past the crossing the white slice carries the black along only
        // while the black is wet.
        let past = |smear: f32, dry: bool| {
            let mut c = Canvas::white(160, 60, palette::ZORN, 0.9);
            let knife = |smear| BrushConfig { knife: 1.0, smear, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() };
            Stroke { path: vec![[60.0, 5.0], [60.0, 55.0]], width0: 16.0, width1: 16.0, load: vec![0.0, 0.0, 4.0, 0.0], pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife(smear));
            if dry {
                c.dry(0.0);
            }
            Stroke { path: vec![[10.0, 30.0], [150.0, 30.0]], width0: 10.0, width1: 10.0, load: vec![0.0, 0.0, 0.0, 4.0], pressure: 1.0, wetness: 0.3 }.rasterize(&mut c, &knife(smear));
            (69..72).map(|x| (27..33).map(|y| c.color_at(x, y)[1] as f32).sum::<f32>() / 6.0).sum::<f32>() / 3.0
        };
        let (plain, wet, dried) = (past(0.0, false), past(1.0, false), past(1.0, true));
        assert!(wet < plain - 10.0, "the wet black is dragged past the crossing ({wet} vs {plain})");
        assert!((dried - plain).abs() < 1.0, "the dried black stays where it was ({dried} vs {plain})");
    }

    #[test]
    fn the_lip_puts_the_liftoffs_crest_at_the_very_end() {
        // The lift-off bead with and without the lip: the lip's peak stands higher and falls off far more
        // sharply behind itself (the crest in the last pixels, a trough behind it), the even bead's rise gradual.
        let at = |lip: f32| {
            let mut c = Canvas::white(160, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[8.0, 20.0], [152.0, 20.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, fade: 0.5, liftoff: 1.0, lip, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let h = |x: usize| (17..23).map(|y| c.height[y * 160 + x]).sum::<f32>() / 6.0;
            let (px, peak) = (140..=153).map(|x| (x, h(x))).fold((0usize, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
            (peak, peak / h(px - 2).max(1e-6))
        };
        let ((p0, s0), (p1, s1)) = (at(0.0), at(1.0));
        assert!(p1 > p0 * 1.15, "the lip's crest stands higher ({p1} vs {p0})");
        assert!(s1 > s0 * 1.4, "the lip falls off sharper behind its crest ({s1} vs {s0})");
    }

    #[test]
    fn the_marbled_slice_winds_its_bands_along_the_mark() {
        // A mixed slice of red with black: without the marble each lane keeps one colour its whole length (the
        // column of tones across the width the same at every x); with it the bands wander, so the columns differ.
        let black = vec![0.0, 0.0, 4.0, 0.0];
        let run = |marble: f32| {
            let mut c = Canvas::white(160, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[8.0, 20.0], [152.0, 20.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize_ramp(&mut c, &BrushConfig { knife: 1.0, mix: 1.0, marble, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() }, Some(&black));
            c.dry(0.0);
            let img = c.to_image();
            (40..=120).step_by(8).map(|x| (16..24).map(|y| { let p = img.get_pixel(x, y).0; crate::paint::color::linear_luma(crate::paint::color::srgb_to_linear(p)) }).collect::<Vec<f32>>()).collect::<Vec<_>>()
        };
        let diff = |cols: &[Vec<f32>]| cols.windows(2).map(|w| w[0].iter().zip(&w[1]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
        let (plain, marbled) = (run(0.0), run(1.0));
        assert!(diff(&plain) < 0.01, "without the marble the bands run straight ({})", diff(&plain));
        assert!(diff(&marbled) > 0.05, "with the marble the bands wander along the mark ({})", diff(&marbled));
    }

    #[test]
    fn the_slump_eases_the_slabs_cliff_and_keeps_its_body() {
        // A thick flat slab with and without the slump: the biggest step of height across its edge is smaller
        // with it, and the slab's middle keeps its height.
        let at = |slump: f32| {
            let mut c = Canvas::white(120, 60, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[10.0, 30.0], [110.0, 30.0]], width0: 12.0, width1: 12.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize(&mut c, &BrushConfig { knife: 1.0, slump, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
            let h = |y: usize| (40..80).map(|x| c.height[y * 120 + x]).sum::<f32>() / 40.0;
            let jump = (20..42).map(|y| (h(y + 1) - h(y)).abs()).fold(0.0f32, f32::max);
            (h(30), jump)
        };
        let ((b0, j0), (b1, j1)) = (at(0.0), at(1.0));
        assert!((b1 - b0).abs() < b0 * 0.05 + 1e-4, "the body keeps its height ({b1} vs {b0})");
        assert!(j1 < j0 * 0.85, "the cliff is eased ({j1} vs {j0})");
    }

    #[test]
    fn a_mixed_knife_slice_lays_its_two_colours_in_bands_across_the_blade() {
        // A knife slice of red with black as its second colour: with the mix the width shows bands of both,
        // with the ramp (no mix) the middle of the mark is one tone across its width.
        let black = vec![0.0, 0.0, 4.0, 0.0];
        let run = |mix: f32| {
            let mut c = Canvas::white(80, 40, palette::ZORN, 0.9);
            let s = Stroke { path: vec![[8.0, 20.0], [72.0, 20.0]], width0: 16.0, width1: 16.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
            s.rasterize_ramp(&mut c, &BrushConfig { knife: 1.0, mix, viscosity: 2.0, k_pickup: 0.0, deplete: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() }, Some(&black));
            c.dry(0.0);
            let img = c.to_image();
            (13..28).map(|y| { let p = img.get_pixel(40, y).0; crate::paint::color::linear_luma(crate::paint::color::srgb_to_linear(p)) }).collect::<Vec<f32>>()
        };
        let (mixed, ramped) = (run(1.0), run(0.0));
        let spread = |v: &[f32]| v.iter().cloned().fold(0.0f32, f32::max) - v.iter().cloned().fold(1.0f32, f32::min);
        assert!(spread(&mixed) > 0.08, "the mixed slice shows both colours across its width ({:?})", mixed);
        assert!(spread(&ramped) < 0.03, "the ramped slice is one tone across its width at its middle ({:?})", ramped);
    }

    #[test]
    fn the_spread_reads_a_slabs_flank_over_a_band_not_a_hairline() {
        // A thick slab; relit sharp, its flank is a line or two at the edge; with the spread the band of pixels
        // inside the slab that differ from its interior is several pixels wide.
        let mut c = Canvas::white(160, 80, palette::ZORN, 0.9);
        let slab = Stroke { path: vec![[10.0, 40.0], [70.0, 40.0]], width0: 50.0, width1: 50.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
        slab.rasterize(&mut c, &BrushConfig { viscosity: 3.0, k_pickup: 0.0, streak: 0.0, round: 0.0, flat_ends: true, deplete: 0.0, ..BrushConfig::default() });
        c.dry(0.0);
        let luma = |img: &image::RgbImage, x: u32| -> f32 {
            (35..45).map(|y| { let p = img.get_pixel(x, y).0; crate::paint::color::linear_luma(crate::paint::color::srgb_to_linear(p)) }).sum::<f32>() / 10.0
        };
        let band = |spread: f32| {
            let img = c.to_image_finished(&crate::paint::canvas::Finish { impasto: 1.0, light: 360.0, spread, ..Default::default() });
            let inner = luma(&img, 30);
            (40..70).filter(|&x| (luma(&img, x) - inner).abs() > 0.01).count()
        };
        let (sharp, wide) = (band(0.0), band(1.0));
        assert!(wide >= sharp + 3, "the spread widens the flank's band ({sharp} px sharp, {wide} px spread)");
    }

    #[test]
    fn the_lamps_azimuth_turns_the_lit_flank_of_a_ridge_and_the_phong_adds_a_highlight() {
        // A thick vertical ridge; relit with the lamp from the right and from the left its two flanks swap
        // light and shadow; with the phong the lit flank is brighter still.
        let mut c = Canvas::white(60, 40, palette::ZORN, 0.9);
        let ridge = Stroke { path: vec![[30.0, 5.0], [30.0, 35.0]], width0: 8.0, width1: 8.0, load: red_load(), pressure: 1.0, wetness: 0.3 };
        ridge.rasterize(&mut c, &BrushConfig { viscosity: 3.0, k_pickup: 0.0, streak: 0.0, round: 0.0, ..BrushConfig::default() });
        c.dry(0.0);
        let luma = |img: &image::RgbImage, x: u32| -> f32 {
            (15..25).map(|y| { let p = img.get_pixel(x, y).0; crate::paint::color::linear_luma(crate::paint::color::srgb_to_linear(p)) }).sum::<f32>() / 10.0
        };
        // (A modest impasto: at 1.0 the diffuse shading alone saturates the relight's clamp on a steep flank.)
        let relit = |light: f32, phong: f32| c.to_image_finished(&crate::paint::canvas::Finish { impasto: 0.3, light, phong, ..Default::default() });
        let (r, l) = (relit(0.0 + 360.0, 0.0), relit(180.0, 0.0));
        let dr = luma(&r, 27) - luma(&r, 33);
        let dl = luma(&l, 27) - luma(&l, 33);
        assert!(dr * dl < 0.0 && dr.abs() > 0.005, "the lamp from the right and from the left light opposite flanks ({dr} vs {dl})");
        // The phong: a narrow highlight where a facet meets the half-vector — somewhere on the lit flank the
        // sheet is brighter with it than without.
        let ph = relit(360.0, 0.5);
        let xs: Vec<u32> = if dr > 0.0 { (20..30).collect() } else { (31..41).collect() };
        let best = xs.iter().map(|&x| luma(&ph, x) - luma(&r, x)).fold(f32::MIN, f32::max);
        assert!(best > 0.01, "the phong brightens a facet of the lit flank (best gain {best})");
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
