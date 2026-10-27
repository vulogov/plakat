//! Medium profiles and stage-schedule generation (RFC PAINT-1 §6).
//!
//! A stage schedule is not a property of the painter — it is a consequence of the medium's **irreversibility
//! structure**. Declare the invariants (opacity, where white comes from, reversibility, how many stages the
//! medium tolerates, whether it splits into light/shadow families) and the schedule *derives itself*. There
//! are no per-medium code paths; a new medium is a new row of invariants.
//!
//! The load-bearing rule the generator encodes: **the irreversible operation goes last** — opaque highlights
//! in oil, the darkest darks in watercolour.

/// The order value is built in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDirection {
    DarkToLight,
    LightToDark,
    MidOut,
}

/// How the paint covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opacity {
    Opaque,
    Transparent,
    Mixed,
}

/// Where the lightest value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhiteSource {
    /// An opaque white pigment laid on top (oil, gouache).
    Pigment,
    /// The bare paper/panel, reserved (watercolour, ink).
    Surface,
}

/// How long the paint stays workable — the axis stages are separated along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reversibility {
    None,
    Minimal,
    Hours,
    Days,
}

/// Whether the medium is painted as two families (light/shadow) or one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Families {
    Split,
    Unified,
}

/// How marks build value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkModel {
    /// Pigment concentration (a loaded brush).
    Continuous,
    /// Mark density (hatching / stippling).
    Density,
}

/// The MARK a medium makes — the physical character of its stroke, as distinct from its finish. Two media that
/// only differ in finish (chroma, grain) paint the same picture in two tints; a pencil and a tempera brush make
/// DIFFERENT marks: a dry point draws thin grey directional strokes and leaves the paper, tempera builds form in
/// short cross-hatched colour strokes. Applied on top of the pass-role brushes; `1.0`/`None` = the brush default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarkCharacter {
    /// A named brush ([`crate::paint::painter::BrushProfile::named`]) used for EVERY pass, or `None` for the
    /// per-role default (flat block-in → filbert → round detail).
    pub role: Option<&'static str>,
    /// Multiplier on stroke length.
    pub stroke_len: f32,
    /// Multiplier on stroke width.
    pub stroke_width: f32,
    /// Multiplier on the pigment charge (a pencil lays grey, not black).
    pub charge: f32,
    /// Cross-hatching: each restating pass rotates its stroke direction by this many radians more than the last
    /// (0 = every pass follows the form). Tempera / hatched media.
    pub hatch_angle: f32,
    /// The medium draws in its OWN black/grey regardless of the picture's colours (graphite).
    pub monochrome: bool,
    /// Density media only: cut the picture as a COPPERPLATE (`paint::engrave`) instead of drawing it with a pen.
    pub engrave: bool,
    /// Multiplier on the stroke budget: a medium of FEW marks (sumi-e) says so here.
    pub budget_scale: f32,
    /// Simplify the armature to this many value masses (sumi-e's paper / grey / black), or `None` for the plan's.
    pub levels: Option<u32>,
    /// Reserve threshold: cells lighter than this stay PAPER (sumi-e keeps most of the sheet white), or `None`
    /// for the medium/plan default.
    pub reserve: Option<f32>,
    /// Keep only the first N (widest) rungs of the brush ladder: a medium of FEW BOLD strokes (sumi-e) has no
    /// fine restating passes — every mark is a committed one. `None` = the full ladder.
    pub ladder_keep: Option<usize>,
    /// Multiplier on the finish contrast (sumi-e's black is black, its paper is paper).
    pub contrast: f32,
    /// Block-in coverage 0..1 (gap-free base): a medium of short opaque strokes (tempera) shows no ground.
    pub coverage: f32,
    /// After the tonal passes, DRAW the composition's contours (the ink planner's edge chains) in the medium's
    /// own dark on top — a pencil sketch is line AND tone.
    pub draw_contours: bool,
    /// LUMINOUS: a light-carrying wash medium. The plan keeps the paper's light instead of building a solid value
    /// backbone — no family separation, no committed shadows, a lifted shadow floor — so the washes stay
    /// transparent and the reserved paper glows (line-and-wash). The oil-minded defaults make a wash read heavy.
    pub luminous: bool,
    /// BOOK: the luminous washes as an early colour-book illustration — flat cumulative glazes from the keyed
    /// picture at pixel scale (the masks of a 3-5 colour print), high key, no modelling strokes. `luminous`
    /// alone is the watercolour: washes once each in the picture's own colour at wash scale, modelled within.
    pub book: bool,
    /// Density media only: the drawing is made with a loaded BRUSH (sumi-e) — bold contours, wide wet tone
    /// strokes only in the mid-to-dark values, paper for the light.
    pub brush_drawing: bool,
    /// SUMI-E: two registers — graded washes for the light family, bold dry-brush black shapes for the dark;
    /// no contours (see `painter::sumi_painting`).
    pub sumi: bool,
}

impl MarkCharacter {
    /// A loaded brush: the pass-role defaults, no hatching, the picture's colours.
    pub const BRUSH: MarkCharacter = MarkCharacter { role: None, stroke_len: 1.0, stroke_width: 1.0, charge: 1.0, hatch_angle: 0.0, monochrome: false, engrave: false, budget_scale: 1.0, levels: None, reserve: None, ladder_keep: None, contrast: 1.0, coverage: 0.0, draw_contours: false, brush_drawing: false, sumi: false, luminous: false, book: false };
}

/// Whether the canvas finishes uniformly or plane-by-plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishPolicy {
    Uniform,
    BackToFront,
}

/// The invariant set that generates a stage schedule and drives the stroke physics.
#[derive(Clone, Copy, Debug)]
pub struct MediumProfile {
    pub name: &'static str,
    pub value_direction: ValueDirection,
    pub opacity: Opacity,
    pub white_source: WhiteSource,
    pub reversibility: Reversibility,
    /// Max stages before the medium degrades (mud / over-working).
    pub stage_budget: usize,
    /// Pickup coefficient for the brush (0 = no pickup, 1 = strong).
    pub pickup: f32,
    /// Wet-into-wet BLEED (0..1): how much pigment fuses/blooms into wet neighbours — the wet media's signature.
    pub bleed: f32,
    /// BODY / opacity (0..1): how opaquely the paint covers — 1 = opaque (gouache/oil), low = transparent
    /// (watercolour/ink, the ground glows through).
    pub body: f32,
    /// IMPASTO (0..1): how much the paint stands off the surface and CATCHES LIGHT — the thick, textured,
    /// palette-knife quality of oil. 0 = flat (watercolour/ink). Relit at output from the stroke height.
    pub impasto: f32,
    // ── Paint MATERIAL physics (how the paint itself behaves, applied at output; §8.5) ──────────────────────
    /// CHROMA / saturation range (1 = neutral; >1 vivid oil; <1 muted gouache/watercolour).
    pub chroma: f32,
    /// DRY SHIFT — value change on drying (+ watercolour dries lighter; − gouache dries to a matte mid).
    pub dry_shift: f32,
    /// GRANULATION — pigment settling into the paper's tooth (watercolour / graphite grain).
    pub granulate: f32,
    /// SHEEN / gloss — specular highlight on the paint ridges (oil glossy; watercolour/gouache matte).
    pub sheen: f32,
    /// LIFT — how removable the paint is by a wipe (oil wet = high; watercolour staining = low; ink/pen = ~0).
    pub lift: f32,
    /// The palette that suits this medium, used when the spec names none.
    pub default_palette: &'static str,
    /// BROKEN COLOUR (0..1): per-stroke hue/chroma variation — adjacent marks are different pure-ish colours
    /// that optically mix (oil/gouache/pastel vibrancy) instead of one pre-mixed tone. 0 = a single solved tone.
    pub broken: f32,
    /// CONTOUR (0..1): a line-drawing pass that draws the strongest edges as clean strokes — for the LINE media
    /// (pen, pencil) that outline the subject, not just shade it. 0 = no drawn lines.
    pub contour: f32,
    pub families: Families,
    pub subtractive: bool,
    pub mark_model: MarkModel,
    /// The physical character of the medium's mark (see [`MarkCharacter`]).
    pub mark: MarkCharacter,
    pub finish_policy: FinishPolicy,
    // ── How the LAYERS go on (the multi-layer behaviour of the paint; the painter's pass loop reads these) ──
    /// INTER-PASS DRYING (0..1): how far the canvas dries between passes when `--dry` is not given — 0.5 for
    /// a paint that stays open (oil), 1 for one that is dry before the next layer (acrylic: each pass a
    /// clean overlay, nothing beneath is picked up or stirred).
    pub dries: f32,
    /// WASH-IN (0..1): the block-in laid THIN — a diluted first layer, flooded, carrying `1 − 0.7·wash_in` of
    /// the charge — the acrylic painter's wash underpainting, which the opaque layers cover where they
    /// restate and leave showing where they do not. 0 = the block-in at full charge.
    pub wash_in: f32,
    /// WET CAP (0..1, 1 = none): the wettest a mark after the block-in may go on. A fast-drying paint sits
    /// ON the layer beneath as a flat opaque patch — it neither dissolves into it nor wets it for the next
    /// mark — so its marks are capped dry; an oil mark goes on as wet as the pass asks.
    pub wet_cap: f32,
    /// SCUMBLE (0..1, 0 = off): the finest pass, off the faces, goes on as a DRY BRUSH — a thinner charge,
    /// nearly dry, raked, no pickup — the broken drag over a dried layer that acrylic is painted with on
    /// hair, foliage, cloth and ground. The faces keep their clean fine marks.
    pub scumble: f32,
    /// DRY-BRUSH SKIP (0..1, 0 = off) of the brush: a mark laid drier than 0.6 wet touches only the standing
    /// fibres of the canvas (see `BrushConfig::skip`) — in the score's header, so the replay skips alike.
    pub skip: f32,
    /// TURBULENCE (degrees, 0 = off): a low-frequency field of noise turns every pass's flow by up to this
    /// much either way — the marks of a hand are never quite parallel; on a face the turn is halved.
    pub turbulence: f32,
    /// ACCENTS (0..1, 0 = off): on the finest pass, where the picture's local contrast is sharp — a glint, a
    /// hard edge — the mark is a short, thick, opaque dab set off the flow at its own angle: the impasto
    /// accent that reads as a touch of paint, not as a trace of the edge.
    pub accent: f32,
    /// LOOSE (0..1, 0 = off): off the subject and the faces the broad passes' marks run longer and wider, and
    /// the restating passes thin with the ladder (the finest keeps `1 − loose` of its marks there) — the
    /// background is painted with the big brush and a free hand, the subject with the small one.
    pub loose: f32,
    /// RIDGES (0..1, 0 = off) of the brush: each bristle lane's paint stands as high as it was loaded, the
    /// mark's section plowed to its edges (see `BrushConfig::ridges`) — in the score's header.
    pub ridges: f32,
    /// IMPASTO MAP (0..1, 0 = off): the paint's thickness follows the picture — lights thick and shadows thin,
    /// the subject thick and the background thin, a face laid smooth (see `PaintParams::impasto_map`).
    pub impasto_map: f32,
    /// DEPLETE (1 = the brush's own): how fast a mark's charge runs out along it (see `BrushConfig::deplete`)
    /// — a heavy-body paint goes down thick at the start of the mark and thins to a dry tail.
    pub deplete: f32,
    /// GLAZE (0..1, 0 = off): on the finest pass, the marks seeded along the subject's silhouette (within two
    /// brush widths of it) are thin and wet, pick up the paint beneath, and may cross it — a translucent
    /// overlap that softens the seam between the figure and its ground instead of a cut edge.
    pub glaze: f32,
    /// SFUMATO (0..1, 0 = off): on a face, where the picture is smooth at the brush's scale, the fine passes
    /// thin with the ladder and the marks that remain go on broad, wet, half-charged, picking up the paint
    /// beneath — the skin fused, not hatched into a grain.
    pub sfumato: f32,
    /// GLAZE PASS (0..1, 0 = off — the charge): one more pass after the ladder, a mid brush, every mark a thin
    /// wet film laid over the ground and the clothes (no restate gate), unifying the detail under one tone —
    /// never over a face or hair, where its mid brush smudged a small head's features.
    pub glaze_pass: f32,
    /// VARY (0..1, 0 = off): a Gaussian spread of the mark's length (σ in log length), wider where the
    /// picture is flat — no two marks the same.
    pub vary: f32,
    /// COMMA (0..1, 0 = off): the hand's pressure along the mark — the width and the deposit swell a third of
    /// the way along and lift toward the ends, a drop or a comma (see `BrushConfig::comma`).
    pub comma: f32,
    /// SHADE (0..1, 0 = off): in the shadows, off the faces and hair, a restating mark lands only where the
    /// canvas is plainly wrong (the restate floor raised `1 + 4·shade·shadow`, against the committed shadows'
    /// drop of it), and the marks that land run longer and wider, laid calm — no dry skip, no hue drift — so
    /// the dark masses rest and the detail is kept for the light, where the eye goes.
    pub shade: f32,
    /// IMPRIMATURA (0..1, 0 = off): the block-in is laid darker and warmer than the picture — pulled toward a
    /// burnt umber — so what shows between the later marks (a dry mark's skips, the thinned shadows, the
    /// seams) is the dark warm underpainting of a classical picture, not the picture's own colour.
    pub imprimatura: f32,
    /// BODY VARY (σ, 0 = off): a Gaussian spread of each mark's body in its log — one mark thick and opaque,
    /// the next thinner, wetter and translucent — the paint's consistency never the same twice on the brush.
    pub body_vary: f32,
    /// LOST (0..1, 0 = off): on the finest ladder pass and the glaze pass, along the subject's silhouette where
    /// the picture is in shadow, a mark's colour is the picture blurred wide across the seam, its path grows
    /// along that blur and may cross the seam, and its charge is `0.4 + 0.6·lost` — the edge between a dark
    /// figure and its dark ground lost, as a painter rubs it out with a finger.
    pub lost: f32,
    /// CREST (0..1, the share): one more pass after the glaze, a fine brush (the ladder's third-finest), off
    /// the faces and hair, where the relief is broken — this share of its seeds are a dry brush dragged over
    /// the dried relief, a lighter, warmer tint that catches only the ridges' tops (see `BrushConfig::crest`),
    /// the texture lit without a new shape.
    pub crest: f32,
    /// SPATTER (0..1, 0 = off): the thickest marks, in the light and off the faces, fling a drop or three off
    /// their lift-off along the mark's direction — round thick dots of the mark's own paint, standing up.
    pub spatter: f32,
    /// AERIAL (0..1, 0 = off): the lost edges by DEPTH. A depth map is estimated for the picture, and where
    /// a passage lies far and in shadow the `lost` treatment is no longer kept to the subject's silhouette:
    /// the farther the passage, the more contrast its edges may hold and still be lost, and the wider the
    /// blur the mark takes its colour from — the background softens itself by its distance, an aerial
    /// perspective without a mask. Needs `lost` above zero.
    pub aerial: f32,
    /// SINK (0..1, the share): on the crest pass, in the shadow, where the relief is broken — this share of
    /// its seeds a thin wet wash, darker and cooler than the picture, that settles only in the furrows of
    /// the relief and runs off the ridges (a negative `BrushConfig::crest`): the shadow's texture deepened
    /// the way a wash is rubbed into a dried impasto.
    pub sink: f32,
    /// TAIL (0..1, 0 = off): a spatter drop's shape follows the SPEED of the mark that flung it — a long
    /// mark for its width is a fast one — the faster, the farther it flies and the longer it stretches along
    /// its flight, a round head and a thinning tail ahead of it; a slow dab's drops stay round.
    pub tail: f32,
    /// FILM (0..1, 0 = off): the glaze marks — the glaze pass, the sink wash, the lost edges, the overlaps
    /// along the silhouette — lay this share of their deposit as a translucent FILM over the dried paint
    /// instead of mixing into it (see `BrushConfig::over`): the two-layer Kubelka-Munk solution reads the
    /// film by its own transmittance over the layer beneath, so a thin glaze tints a heavy passage the way
    /// it does on a real canvas, where mixed in it was outweighed and read as a muddy average.
    pub film: f32,
    /// SPLIT (0..1, 0 = off): the dry marks — the scumble, the crest, the hair strands — comb into two to
    /// four bundles with bare canvas between them as they dry out and the faster they go (see
    /// `BrushConfig::split`): the clumped bristles of a half-dry brush parting along the mark.
    pub split: f32,
    /// SAG (0..1, 0 = off): at output, the stretched canvas gives under the weight of the thick paint — a
    /// broad shallow dip under every heavy passage and a slow unevenness of the cloth on its bars, a gentle
    /// swell of the light over the whole sheet (see `Finish::sag`).
    pub sag: f32,
    /// GLOSS (0..1, 0 = off): at output, a sharp specular on the PEAKS of the thick paint — acrylic dries with
    /// a gloss on the highest points of a heavy mark where the body of the sheet is satin (`sheen`), so the
    /// crests glint under the lamp and the thin passages stay matte (see `Finish::gloss`).
    pub gloss: f32,
    /// TINT (0..1, 0 = off): the bundles of a split mark differ a little in tone — the paint not mixed to one
    /// colour on the palette, each clump of bristles carrying its own (see `BrushConfig::tint`) — so the
    /// combed lines of a hair or a feather are not one colour repeated. Needs `split`.
    pub tint: f32,
    /// KNIFE (0..1, the share): off the subject, on the broad passes, this share of the marks are laid with a
    /// PALETTE KNIFE instead of a brush (see `BrushConfig::knife`): a flat blade full width with square ends
    /// that shears the paint standing under it down to a plateau and lays its own flat, thick and nearly dry —
    /// smooth planes with sharp edges cut into the relief, against the brushed faces.
    pub knife: f32,
    /// HIDING (0..1, 0 = off): each pigment covers by its OWN hiding power (`Pigment::hiding`) instead of the
    /// paint's amount alone — the cadmiums, the whites and the earths opaque, the lakes and the organic blues
    /// and greens translucent laid thick, a pigment the picture named read from its masstone — scaled toward
    /// the plain law by `1 − hiding`; in the score's header.
    pub hiding: f32,
    /// TWIST (0..1): the brush turned in the fingers along the mark (see `BrushConfig::twist`) — on every
    /// hair strand, and on this share of the finest pass's marks off the faces — the width running from the
    /// brush's breadth to a line, the calligraphic mark.
    pub twist: f32,
    /// SKIM (0..1, the share): this share of the knife marks SKIM — the blade at a low angle with little
    /// paint, pressed less and catching only the relief's tops, so the plane it lays is torn and holed over a
    /// low relief: old plaster, the glints on water. Needs `knife`.
    pub skim: f32,
    /// BLADE (0..1, the share; 7.14.0, `palette-knife`): the PALETTE KNIFE AS THE MEDIUM'S OWN TOOL — on every
    /// pass, the block-in and the finest alike, on the subject and the faces and the hair strands as off them,
    /// this share of ALL the marks are laid with the knife (see `BrushConfig::knife`): a flat blade full width
    /// with square ends, shearing the standing paint to a plateau and laying its own flat, thick and nearly
    /// dry, no pickup. Where `knife` is a share of the broad marks off the subject (a knife beside the brush),
    /// `blade` is the whole picture formed in knife slices — nothing of a brush's transparency, water or hair
    /// lines. In the stroke's own `knife`, so the replay lays it alike.
    pub blade: f32,
    /// BLADE PLANES (a count, 0 = off; `palette-knife`): the picture the knife paints from is CUT INTO FLAT
    /// PLANES OF VALUE — every pass's reference quantised to this many planes (see `blade_planes_ref`), each pixel's
    /// colour scaled in linear light to its plane's mean luma with its hue and chroma kept — so the form is built in
    /// steps of pure flat slices and no slice blends a gradation; the posterisation is the knife's own canon, the
    /// mean tone of the picture held.
    pub blade_planes: f32,
    /// BLADE LEVELS (0..1; with `blade_planes`): where the planes are cut — 0 even steps of the value's range on the
    /// eye's scale, 1 the picture's own quantiles (every plane the same area), between a blend.
    pub blade_levels: f32,
    /// BLADE SIZE (0..1, 0 = off; `palette-knife`): the slice's size by the picture's FLATNESS under it — on a flat
    /// passage (a sky, a wall) a wide long slab, at a sharp detail a small hard one: the mark's length and width
    /// both scaled by `1 + key·(2.4·flat − 0.4)`, flat = 1 − range/0.3 from the local range at the brush's scale
    /// (the block-in keeps its own size). The inverse of the gradient as the blade's reach.
    pub blade_size: f32,
    /// BLADE BEVEL (0..1, 0 = off; `palette-knife`): the thickness profile of a knife slice (see
    /// `BrushConfig::bevel`) — thin where the blade is pressed down at the start, the paint pushed ahead building
    /// to a sharp ridge at the lift-off, and squeezed out to the blade's two edges — so the relit relief shows every
    /// slice as a slab with a fascia, not a flat film; the IMPASTO Z-BUFFER's step. In the header (`bevel`).
    pub blade_bevel: f32,
    /// BLADE LIGHT (degrees, 0 = off): the azimuth of the raking lamp the relief is relit under (see
    /// `Finish::light`) — 0 from the right, 90 from below, 180 from the left, 270 from above; off, the old fixed lamp
    /// at about 56°. In the header (`light`).
    pub blade_light: f32,
    /// BLADE PHONG (0..1, 0 = off): a specular lobe on the relief's facets that face the lamp at output (see
    /// `Finish::phong`) — a narrow highlight on every fascia of the thick paint turned to the lamp, gated by its
    /// steepness, so the slabs' edges glint and their planes stay matte. In the header (`phong`).
    pub blade_phong: f32,
    /// BLADE FLOW (0..1, 0 = off): the slices follow the PICTURE's form, not the planes' — each pass's flow field
    /// (the structure tensor) is read from the reference BEFORE it is cut into planes (the fur's strands, the
    /// shoreline, not a plane's edge), and the slice is STRAIGHT: a rigid blade lays a quad along the flow at its
    /// start, every point of the grown path taken to the chord by the key's share.
    pub blade_flow: f32,
    /// BLADE REST (0..1, 0 = off): on a FLAT passage the restating passes leave the slab alone — a mark is skipped
    /// by the key × the flatness under it × the pass's depth in the ladder (nothing on the widest brush, the key's
    /// share on the narrowest): the sky stays the broad blade's slabs, not a comb of small corrections.
    pub blade_rest: f32,
    /// BLADE RELIEF (0..1, 0 = off): the relief's slope read over a window of the picture's scale at output (see
    /// `Finish::spread`) — a slab's flank a band of light and shadow the eye reads at the picture's size, not a
    /// hairline at its edge. In the header (`spread`).
    pub blade_relief: f32,
    /// BLADE MIX (0..1, 0 = off): a knife slice carries TWO colours across the blade — the picture under the slice's
    /// footprint split at its median value into a darker and a lighter half, the half further from the mark's own
    /// tone its second colour (taken toward it by the key), laid in unmixed bands of lanes (`BrushConfig::mix`,
    /// header `bmix`): the paint not fully blended on the knife. In the record as the mark's second charge.
    pub blade_mix: f32,
    /// HATCH (0..1, 0 = off): the restating passes lay HATCH lines — the tempera mark: straight (the hand's
    /// waver and the flow's curve taken out by this much), narrowed toward a line of one width end to end,
    /// longer for it, semi-opaque (`1 − 0.3·hatch` of the charge), dry, no pickup — so the form is built in a
    /// net of fine parallel strokes, each pass crossed at the medium's `hatch_angle`, the layers showing
    /// through one another as egg tempera is painted.
    pub hatch: f32,
    /// VERDACCIO (0..1, 0 = off): the flesh underpainted in the greenish grey of the Italian panel painters
    /// (terre verte, white, a little ochre and black): on the faces the block-in and the first restating pass
    /// are laid in that tone at the picture's value, and the later passes in the half-tones and shadows are
    /// pulled toward it and thinned, so the green shows through the hatched flesh where the form turns.
    pub verdaccio: f32,
    /// DISHES (the count, 0 = off): Cennini's dishes — each colour mixed in advance in this many value steps
    /// (a dark, a middle, a light, …) and every restating mark laid in the nearest step, never a tone between,
    /// so the modelling goes in steps the hatching softens; the block-in and the veils are not stepped.
    pub dishes: f32,
    /// LIGHTS (0..1, the share): on the finest pass, on the subject, where the picture is in its top values —
    /// this share of the marks are the LIGHTS hatched in: the picture's colour lifted toward white (the lead
    /// white of the tempera painter's highest dish), laid narrower and thinner, the last touches on a face
    /// and a fold of drapery.
    pub lights: f32,
    /// CROSS (degrees, 0 = the medium's `hatch_angle × layer`): the hatching turned OFF THE FORM by this
    /// step a hatched pass — the first hatched pass along the form (the local flow, the structure tensor's
    /// isophote), the next `+cross`, then `−2·cross`, then `+3·cross` off it, alternating sides — so the net
    /// of hatching wraps the volume at the panel painter's angles (15°, 30°, 45° off the form) instead of
    /// lying over it as one grid turned 45° a pass. Needs `hatch`.
    pub cross: f32,
    /// DITHER (0..1, 0 = off): the dishes chosen stochastically between the two nearest where a value lies
    /// between them — the chance of the upper dish the value's fraction of the way to it, blended toward the
    /// nearest by `1 − dither` — so the bands between the dishes break into an optical mixture of marks of
    /// both (a lamp's halo in rings at 0, a graded mixture at 1). Not the block-in (its flat local tones stay),
    /// and growing in over the finer half of the ladder — it is the hatching's mixture; on the broad
    /// restatements two dishes side by side were a chequer of planks. Needs `dishes`.
    pub dither: f32,
    /// SOLID (0..1, 0 = off): the hatch's charge keyed to the value — at 0 every hatch line `1 − 0.3·hatch`
    /// of the charge; at 1 a line in the picture's lightest value keeps that thinness (a glaze over the
    /// light) and a line in its darkest carries `1 + 0.3·hatch` (laid covering), linear between — so the
    /// deep shadows build a solid dark in layers and the lights stay veils. Needs `hatch`.
    pub solid: f32,
    /// FLESH (0..1, 0 = off): the verdaccio held to the WARM colours — the pull taken away by this share
    /// as the local colour goes cold or neutral (CIELAB a* and b* below ~2, fully warm above ~10/12), so
    /// a cold or grey passage inside the face's region (a hat, a wall, metal) keeps its own underpainting.
    /// Needs `verdaccio`.
    pub flesh: f32,
    /// SPAN (0..1, 0 = off): the hatch's LENGTH keyed to how much the form changes within the brush's reach
    /// (the reference's value range there) — on a flat passage (a sky, a wall) the line runs up to
    /// `1 + 0.5·span` of its length, the long calm stroke, and where the form is busy (a face, a hand, small
    /// folds) down to `1 − 0.5·span`, the short hatching. Needs `hatch`.
    pub span: f32,
    /// EDGE (0..1, 0 = off): the hatching held to a strong contour — near a plain edge (the value range within
    /// the brush's reach above ~0.15, fully at 0.5) the crossing turn is taken back toward the edge's own
    /// direction by this share and the mark stops sooner on a change of colour, so the hatching runs parallel
    /// to the silhouette and does not fringe across it. Needs `hatch`.
    pub edge: f32,
    /// GLINT (0..1, 0 = off): the pure LIGHTS set last — on the finer half of the ladder, where the picture is
    /// in its top twentieth of values (a star, the sun through the leaves, a lamp), the mark is lifted toward
    /// white by `0.4·glint`, laid narrower (`1 − 0.3·glint`), dry, with the full covering charge
    /// (`1 + 0.3·glint`) and standing thicker (its viscosity `1 + 1.5·glint`) — the dry, nearly raised touch
    /// the panel painter sets the final highlights with.
    pub glint: f32,
    /// SMOOTH (0..1, 0 = off): the CALM passages — where the form changes little within the brush's reach (a
    /// sky, a wall, still water; the reference softened by half the radius first, so a star in a still sky
    /// is not a change of form) — built up in soft veils instead of hatched: by the key's share of the
    /// passage's flatness the dishes relax toward the picture's own gradation (the block-in's too, so a sky is
    /// not banded), the hatching gives way (no straightening, no narrowing to a line), and the restating mark
    /// goes on wetter, shorter (a dab, not a plank across a gradient) and barely thinner so it
    /// blends into its neighbours — the smooth mottled ground of a panel's sky, over which the glints stay
    /// crisp dots and the contours (`edge`) stay sharp. The ground's only (scaled by `1 − subject`): a
    /// subject's dark mass is committed paint, built solid in its steps.
    pub smooth: f32,
    /// THE BOLE (0..1, 0 = off): the imprimatura laid by LOCAL tone, not one flat colour. A panel painter
    /// began on a coloured ground — a bole, a toned gesso — and the first lay-in was itself a tonal statement:
    /// the dark masses toned dark, the sky toned to its own colour. Here, before the block-in of a new
    /// painting, the sheet is laid in cells (a thirty-second of the short side) with the picture's own SHADOW
    /// tone at that scale — the mean colour of the cell's darker quarter of pixels; a toned ground is darker
    /// than the lights, and a mean tone round a star or a lamp glowed and lifted the broad marks over it —
    /// smoothed over the neighbouring cells and then spread as a CONTINUOUS field — interpolated between the
    /// cells' centres and laid in fine cells of a quarter of their side, each solved exactly — each cell's
    /// colour the ground's mean colour pulled toward that shadow tone by this key, laid thin (three tenths
    /// of the charge), edge to edge, so no cell edge is a line. Where a broad mark
    /// leaves a gap or goes on thin, what shows is the picture's local shadow tone, not a flat brown — and the
    /// block-in's marks, mixing by concentration with the thin bole beneath, are pulled toward the tone they
    /// lie on. Recorded as wash cells in the score (stage `ground`), replayed to the bit.
    pub bole: f32,
    /// THE RAMP (0..1, 0 = off): the colour of a mark FOLLOWS THE PICTURE along its path. A mark takes one
    /// colour at its seed; across a gradient — a sky, the ground falling into shadow — that flat colour laid
    /// as a plank is the dab structure that no veil hides. With the ramp, the brush's charge drifts along
    /// the mark from the seed's colour toward the picture's colour at the mark's end (the seed's own tone
    /// transforms kept, the picture's end-to-start difference added in linear light by the key's share), so
    /// the broad passes model the form before the fine ones hatch it. Full on the soft marks, fading as the
    /// hatching takes over (`1 − hatch`); off the glazes, the lost edges, the hair, the accents and the glints.
    /// The end mix rides in the score (`mix2`); without it a record is one colour as ever.
    pub ramp: f32,
    /// THE UMBRA (0..1, 0 = off): the shadow is one modelled mass, not steps. The dishes space the picture's
    /// values perceptually between its darkest and lightest; on a night scene the whole foreground falls in
    /// the darkest step, and the marks either side of its boundary were a chequer of black and dark grey
    /// planks that no veil hid. Within this share of the darkest dish's interval a tone is laid as it is —
    /// continuous, the shadow modelled by its own gradation (and by the ramp along the mark) — and the
    /// dishes step the lights above it, as a panel painter lays the shadow mass in one tone and builds the
    /// lights up in steps. Off the subject the shadow mass is laid CALM (see `smooth`) however busy the
    /// picture is there — built up in veils, its planks dissolved — as the sky is: a lamp-raked gravel
    /// foreground is one dark mass, not a rake of filbert marks.
    pub umbra: f32,
    /// CHALK (0..1, 0 = off): the mark is a stick of CHALK on a toothed paper (see `BrushConfig::chalk`) — the
    /// deposit keyed to the paper's grain against the mark's pressure (a light touch reaches only the raised
    /// fibres and the hollows keep the sheet's colour, a hard press fills them) and to how full the tooth is
    /// (past three fifths full the chalk slides over what is there), so the layers build as pastel does: the
    /// first ones broken on the grain, the later ones filling it, the last barely taking. The medium's, in
    /// the score's header.
    pub chalk: f32,
    /// THE SIDE OF THE STICK (0..1, 0 = off): on the broad passes, off the faces and the hair, this share of the
    /// marks (the key's share on the block-in, falling with the ladder) are laid with the stick's SIDE — a
    /// wide flat sweep (`1 + 0.6·side` of the width, the stick's facets as streaks, full width to the end,
    /// straighter by half the key) at a light touch (`1 − 0.5·side` of the pressure, so the chalk catches the
    /// raised grain only), `1 − 0.3·side` of the charge, dry, no pickup. The pressure in the record.
    pub side: f32,
    /// THE FINGER (0..1, 0 = off): after each pass of the broad half of the ladder the CALM passages are
    /// rubbed — a smudge along the flow with no pigment of its own that moves a share of the paint under it
    /// toward what it carries (see `Stroke::smudge`), so the underpainting's broad marks fuse into soft masses
    /// (a sky, a wall, a still water) while the fine passes over them stay discrete strokes and the busy
    /// passages, the faces and the hair keep their marks. The finger a fortieth of the sheet's short side;
    /// every cell of it rubbed by its calm (the key's share of its flatness, the subject at three tenths),
    /// `0.08·calm` strong, stopping at the picture's edges. Each rub a `smudge` record, replayed to the bit.
    pub blend: f32,
    /// THE SKIN (0..1, 0 = off): the finger on the FACES — a face's calm passages (a cheek, a forehead; not the
    /// eyes, the nose, the mouth, which the range reads as busy) are rubbed after the face's broad passes by
    /// this share of their flatness — after the finest pass too, the skin alone — the rub kept inside the
    /// face (it never crosses the hairline or the silhouette), so the skin's lights and half-tones fuse into
    /// modelling and only the busy features (the eyes, the nose, the mouth) keep their marks — the pastel
    /// portrait's blended skin. Needs `blend`. Each rub a `smudge` record.
    pub skin: f32,
    /// CARNATION (0..1, 0 = off): the flesh in WARM sticks — on a face (the hair apart) a mark's mixture is
    /// solved over the palette's warm and neutral pigments alone (a pigment whose masstone is cold in CIELAB,
    /// a* or b* below −6 — a teal, a blue, a green — is left in the box), so a half-tone is darkened with an
    /// umber or a black and never greyed with the sea's teal; and the LIGHTS on a face are lifted along their
    /// own hue (the lightness raised, the chroma kept by all but three tenths of the lift) instead of toward
    /// white — a Naples yellow over the cheek, not a chalky grey. The key's share: at 1 the cold pigments are
    /// shut out and the lift is all hue; between, the two mixtures are blended by it.
    pub carnation: f32,
    /// SMALL (px, 0 = off): the SMALL-DETAIL threshold — a face whose extent (the detector's median face
    /// width) is under this many pixels is handled at its own scale, by `k = 1 − extent/small`: its marks
    /// narrower (`1 − 0.5·k` of the width), its hatching thinned (`1 − k`) and never turned off the form (the
    /// cross turn taken away by `k` on the face, the strokes one way along the flow — a cross-hatched net on a
    /// small face was a scale of grit), its half-tones held to the local tone (the mark's CIELAB lightness
    /// clamped within `6 + 24·(1 − k)` of the picture's local mean there — a dark stroke across a lit cheek
    /// read as a cold scar), and more of it blended (the sfumato's flat gate widened by `1 + 2k`, its share
    /// raised halfway to one by `k`). A large face (a portrait) is untouched.
    pub small: f32,
    /// ANCHOR (0..1, 0 = off): on a SMALL face the structural darks are not softened — the eyes, the
    /// nostrils, the line of the mouth. A mark whose tone falls in this share of the darkest values around
    /// it (below the local mean by the matching normal quantile of the local spread, measured over the same
    /// box as `small`'s mean) is not held to the local tone and not thinned by the sfumato; and the
    /// sfumato's widened gate and raised share fall off with the local contrast, so a sharp step (a lip,
    /// the eye's slit, the jaw's edge) keeps its dry stroke and only the cheek and brow are blended.
    pub anchor: f32,
    /// TRIM (0..1, 0 = off): the face is the detector's box CUT BY THE FIGURE — the face mask multiplied by
    /// the subject matte (ramped from three to seven tenths), so the box's corners off the head (the sea in
    /// front of a profile) are ground, not flesh — and the ground's marks keep CLEAR of the face: within this
    /// share of the pass's brush radius around it a ground mark is not seeded and its path stops. Untrimmed,
    /// the box's corners took the face's tier and its warm sticks (the sea's teal mixed from the warm
    /// pigments alone was a grey-olive slab under the cheek), and the sea's broad block-in marks ran their
    /// whole length across the cheek; the chalk's tooth let both through every later layer as a pale
    /// grey-green crackle. (A full radius left a halo of ground around the profile; half closes it.)
    pub trim: f32,
    /// SMALL'S FLOOR (px, 0 = none): a face under this extent is a crowd's face — a few marks the size of
    /// the stick, which a softening only wipes out — and is left to the plain ladder: `small`, `anchor`,
    /// `trim` and the warm glint stay off for the picture (the festival's 73 px singers lost their mouths
    /// and eyes to the held tone and the blending, and the clearance tore the hat's edge).
    pub small_min: f32,
    /// PAPER (0..1, 0 = off): the LIGHT IS THE PAPER — charcoal only darkens. A mark is laid in the darkest
    /// pigment alone, its charge solved on a scrap with the mark's own brush so that over what is there it
    /// reads the picture's value (the residual: target over the canvas's value), and a mark is not laid
    /// where the sheet is already as dark as the picture asks. By the key's share against the plain
    /// mixture. (With the sumi box the lights were painted in titanium white — a grey oil, not a drawing.)
    pub paper: f32,
    /// ERASE (0..1, 0 = off): the KNEADED ERASER — where the sheet is darker than the picture asks in its
    /// lights (above a third of the picture's range, on the perceived scale), off the block-in, the mark is a lift instead: a wipe
    /// along its path whose strength takes the charcoal's film down to the target's value (by the key's
    /// share), recorded as a `W` record and replayed to the bit. Needs `paper`.
    pub erase: f32,
    /// TOUCH (0..1, 0 = off): the PRESSURE follows the value — a light tone is a light touch that catches
    /// only the paper's raised grain (the half-tones read as charcoal on a toothed sheet), a dark one a
    /// hard press that fills the hollows; `0.38 + 0.6·darkness^0.7` (the darkness perceived: one less the value's ratio to the paper's, to 1/2.2), blended with the hand's pressure by
    /// the key's share. Needs `paper` (and `chalk` for the grain).
    pub touch: f32,
    /// VELVET (0..1, 0 = off): the darkest share of the picture's range (on the perceived scale) is the darkest pigment itself,
    /// laid at a full press — a dark frame is one velvet black, not a grey porridge of restated darks.
    pub velvet: f32,
    /// PAPER TOP (0..1, 0 = off): the picture's light end (its 95th percentile) IS the paper, by this share —
    /// a mark's value read against it, so a near-white ground stays bare paper instead of a grain of grey
    /// that stops at a line where the photograph's ground crossed the tolerance. The lift is at most ×1.25 and
    /// weighted by the square of the value's share of the light end (a night's light end is itself dark). Needs `paper`.
    pub paper_top: f32,
    /// SWEEP (0..1, 0 = off): on a flat passage off the subject and the faces the flow is turned onto one
    /// direction (the horizontal), by this share of its flatness — the sky laid in calm side-of-the-stick
    /// sweeps instead of hatching along the noise of a flat field's flow (whirls over a grey sky).
    pub sweep: f32,
    /// STUMP (0..1, 0 = off): after the finest pass the DARK masses (below the middle of the perceived range)
    /// and the flat passages off the subject are rubbed with a stump — the finger's smudge records, off the faces and the hair, stopping at the
    /// picture's edges, by this share of the darkness — the paper's white specks in a dark passage filled
    /// with the charcoal around them.
    pub stump: f32,
    /// DUST (0..1, 0 = off): the charcoal's dust settles into the paper's hollows — under the `chalk`, a
    /// hollow the stick's reach misses still takes this share of the reach's deposit, so a light touch keeps
    /// the grain but a pressed half-tone fills (the hollows below the reach stayed the paper's white under
    /// every layer, a fill that never grew where nothing had been laid: white specks across a grey blanket).
    pub dust: f32,
    /// STUMP EDGE (0..1, 0 = off): the stump holds back where the picture's PERCEIVED local contrast stands
    /// at a finger's reach (the gamma range, fully at 0.12), by this share — a lapel, a sleeve against a
    /// dark ground kept, not rubbed into the mass. Needs `stump`.
    pub stump_edge: f32,
    /// STUMP FEATHER (0..1, 0 = off): the stump's stop at a face and the hair is soft — their mask blurred
    /// over this share of a finger, the rub fading toward them instead of ending at a line — and it may cross
    /// the subject's silhouette, stopped by the picture's own edges alone (a lost edge blended, a found
    /// one kept). Needs `stump`.
    pub stump_feather: f32,
    /// SWEEP WAVE (degrees, 0 = off): the swept flow turned by a slow wave of up to this many degrees — a
    /// low noise across a sixth of the sheet, the arc of an arm, not a ruler. Needs `sweep`.
    pub sweep_wave: f32,
    /// PAPER KNEE (0..1, 0 = off): the top of the tonal curve bent toward the paper — a value in the upper
    /// two fifths of the perceived range pulled toward the paper by this share (a smoothstep), in a light
    /// picture only (by the square of its light end against the paper), so a broad light mass (snow) is the
    /// paper with a touch of grain and the hatching kept for the half-tones. Needs `paper`.
    pub paper_knee: f32,
    /// REWET (0..1, 0 = off): a dried layer softens again under a wet mark — gouache's gum stays soluble — so
    /// the mark lifts this share of the paint beneath into its load, by its own wetness and how empty it is,
    /// and the layers fuse at their seams instead of sitting as cut-out shapes. In the score's header.
    pub rewet: f32,
    /// CHALK TINT (0..1, 0 = off): a light is mixed with white, and white takes the colour's chroma — the
    /// target's chroma (CIELAB a*, b*) cut by up to `0.6·chalk_tint`, rising by a smoothstep from L* 55 to 95,
    /// off the faces: the chalky, pastel-matte lights of a body colour.
    pub chalk_tint: f32,
    /// FLAT BRUSH (0..1, 0 = off): on the passes before the finest, off the faces and the hair, this share of
    /// the marks on a flat passage (by its flatness) are laid with a flat brush — a square chisel end (the
    /// mark full to its end, round 0) and straighter by half the key: a field, a wall, a sail in flat patches.
    pub flat_brush: f32,
    /// LINER (0..1, 0 = off): after the ladder a liner lays the LONG thin structures the brushwork lost — a
    /// rope, a rigging line, a mast's stay, a fence wire: a ridge at the thin scale (lighter or darker than both
    /// its sides) that agrees with its own direction over a long window, traced along itself while the ridge
    /// holds and the line turns gently, and laid as ONE thin mark only if it runs a thirtieth of the sheet —
    /// the short curls of a texture (a wheat field's ears) are not lines. Its own colour, the mean along it.
    pub liner: f32,
    /// TSUKE (0..1, 0 = off): tsuketate — the brush loaded in two tones, its tip in thick ink and its body in a
    /// thinner one, so a single mark runs dark along one edge and pale along the other (the lanes across its
    /// width loaded from full to `1 − 0.85·tsuke`), the dark edge turned to the darker side of the form: the
    /// form modelled in one movement. In the score's header.
    pub tsuke: f32,
    /// MA (0..1, 0 = off): the empty space — where the eye does not go (the focal field — the picture's contrast
    /// weighted to the centre — and the subject both low) the value is lifted toward the paper by this share of
    /// the square of the inattention, so the periphery is the bare sheet or a stroke or two. Needs `paper`.
    pub ma: f32,
    /// KASURE (0..1, 0 = off): the flying white — as the brush runs along the mark its hairs part, the lanes
    /// dropping out one by one by a slow hash of their own (by up to this share at the mark's end, rising with
    /// the square root of its cube), so the white of the paper streaks along the stroke's direction. In the header.
    pub kasure: f32,
    /// KATSU (0..1, 0 = off): 渇筆, the DRY BRUSH BY THE PAPER'S GRAIN AND THE MARK'S SPEED — along every mark
    /// the paint's threshold on the paper's relief rises with the brush's speed (its position along the mark,
    /// as `kasure` reads it: the three-halves power), to the key at the mark's end, whatever the mark's water,
    /// so a mark starts full and ends laying ink on the standing fibres alone — the bristles' tracks the
    /// paper's own, not a lane's hash (`kasure`) nor a contour's tear (`nib_wash_rag`). `skip` is the same gate
    /// read from the mark's dryness. On a pen's line it shows nothing (a hair's width has no grain to read);
    /// its place is the broad swept wash (`nib_wash_sweep`), whose marks it also lays drier (wetness
    /// `0.8 − 0.5·key`) with uneven hairs (streak `0.6·key`). In the header.
    pub katsu: f32,
    /// NIJIMI (0..1, 0 = off): the ink bleeding into an absorbent sheet — the mark's reach widened by up to
    /// `0.8·nijimi` (more for a wetter mark), the halo beyond its body falling off and caught by the paper's
    /// fibres (the relief standing above a threshold that rises outward), a soft feathered edge; and the water
    /// evens the hairs — the bristle streaks damped by the key's share of the mark's wetness, a broad wet stroke
    /// a wash, not a comb. In the header.
    pub nijimi: f32,
    /// TURN (0..1, 0 = off): the brush lifted on a turn — the mark's width narrowed by up to `0.6·turn` where its
    /// path bends (the angle between the incoming and outgoing direction over half a width each way, fully at a
    /// right angle): a wing's edge or a curl of fur drawn as a mark that thins round its bend, not a straight
    /// slab of constant width. In the score's header.
    pub turn: f32,
    /// YOHAKU (0..1, 0 = off): the blank white — where the picture is FLAT (its perceived local range at a
    /// sixty-fourth of the sheet under 0.03, by a smoothstep to 0.10) and off the subject, the value lifted to
    /// the paper by this share: a fog, a pale sky, a white ground left bare, not filled with a grey of strokes.
    /// Needs `paper`.
    pub yohaku: f32,
    /// BONE (0..1, 0 = off): the LINE register (骨法, the bone that holds the form) — the structure's contours
    /// drawn as calligraphic brush lines over the wash painting: the value's edges thinned to a pixel and kept by
    /// hysteresis (the pen-and-ink planner's edge map, opened by `bone`), linked into chains, each chain ONE
    /// brush line whose weight and ink follow the darker side of the edge along it (a near dark form a full
    /// black line, a far pale one a thin grey hair), tapering to its lift-off; `turn` and `kasure` act on it.
    /// Where the eye does not go (`ma`) a chain must run up to twice as far to be drawn. Stage `bone`.
    pub bone: f32,
    /// BONELESS (0 = off; a count): 没骨, the masses laid without the bone — the block-in AND the next `boneless`
    /// passes of the ladder go on as the block-in's flooded WASH (`wash_in`: wet, a thin charge, the hairs
    /// evened by the water), so a tonal mass is a soft wash and not a heap of dry wedges; the passes after
    /// them are the brush's. Needs `wash_in`.
    pub boneless: f32,
    /// HAKE (0..1, 0 = off): the broad flat wash brush's ONE sweep — on the washed passes (`boneless`) the flow
    /// turned to the horizontal by this share, off the subject, so a tonal mass is built from parallel bands
    /// that overlap along themselves (each stopping where the tone changes), not from bars crossing at every
    /// angle of the picture's flow (crossed, each translucent bar was a pane — a sheet of broken glass).
    /// Needs `boneless` and `wash_in`.
    pub hake: f32,
    /// JOIN (0..1, 0 = off): the brush's stamps turned with the path over a window of this share of the width
    /// (see `BrushConfig::join`) — a bend in a wide wash no longer leaves a sliver of bare paper outside the
    /// corner and a double sliver inside it. In the score's header.
    pub join: f32,
    /// BONE WIND (0..1, 0 = off): the bone's chains that WIND are texture, not structure — the isophote of a
    /// tuft of grass, a clump's lobed edge, a small closed loop. A chain's mean turning per `min_len` of travel
    /// (over samples half a `min_len` apart, so the pixel staircase averages out) must stay under
    /// `(1 − bone_wind)·2` radians — at 0.5 a curve must keep a radius of a `min_len` or more, at 0.75 of two,
    /// at 0.9 of five — or the chain is not drawn as a line (grass clumps were outlined in looping worms).
    pub bone_wind: f32,
    /// BONE DOT (0..1, 0 = off): the tenbyō — of the chains `bone_wind` turned away, this share are laid as DOTS
    /// instead: the tip of the brush touched along the chain every four base widths, each dot a round touch
    /// sized by the edge's darkness (`1.5..3.5 × base`) and varied by a third, so a mass of foliage or grass reads as a
    /// rhythm of touches, not a line and not nothing.
    pub bone_dot: f32,
    /// BONE SCATTER (0..1, 0 = off): the tenbyō's dots thrown off their chain — each dot moved along the chain's
    /// normal by a Gaussian of `bone_scatter × 4 base widths` (the dots' own spacing at 1), so they fall as the
    /// clusters of moss and grass a brush's tip leaves, not as a pricked line along a contour.
    pub bone_scatter: f32,
    /// BONE COHERENCE (0..1, 0 = off): a bone chain is drawn only where the value's gradients around it AGREE
    /// on a direction — the structure tensor summed over a window half a `min_len` wide, its coherence
    /// ((λ1−λ2)/(λ1+λ2))² averaged along the chain, must reach the key. The edge detector keeps any strong
    /// step; in grass, foliage or gravel the gradients point every way, and a strong step there is texture —
    /// laid as the tenbyō's dots (`bone_dot`) or nothing — while a coat's edge against the ground, where every
    /// gradient agrees, is a line (a hillside's grass ridges were drawn as thin lines).
    pub bone_coherence: f32,
    /// BONE END (0..1, 0 = off): the bone line's ends read as the calligrapher's — the TOME and the HARAI. An
    /// end that stops against another line (an edge pixel of another chain beside it: a junction, a corner
    /// met) is a tome: the brush halts and dwells, the ink spreads a little and pools — a round press laid
    /// there, wider and darker by the key. An end where the edge just fades into the ground is a harai: the
    /// brush sweeps off the paper, the line drawn toward that end and narrowing to a hair (its width there
    /// falls with the key), the hairs parting (`kasure` already rises along the mark). Unset, every line ran
    /// at one taper from whichever end the tracer began.
    pub bone_end: f32,
    /// SUMI FOCUS (0..1, 0 = off): the black register (`sumi`) laid only where the eye goes — the attention
    /// at a mass (the picture's saliency weighted to the centre, or the subject) must reach the key, or the
    /// mass is left to the washes beneath. Unset, the darkest masses in the far ground — a ruin's debris, a
    /// hillside's rocks — were black shapes as strong as the subject, competing with it.
    pub sumi_focus: f32,
    /// MA FOCUS (0..1, 0 = off): the eye's attention that `ma` reads narrowed to the focus — raised to the
    /// power 1 + 4·key, so a passage of middling saliency (a ruin's debris, a hillside's near ground) counts as
    /// far from the eye and is lifted toward the paper with the periphery, and the subject stands alone. Also
    /// read by the bone's chain length off the focus. Unset, the far ground's debris drew the eye as strongly as
    /// the figure.
    pub ma_focus: f32,
    /// MA FLOW (0..1, 0 = off): the attention `ma_focus` reads spread from the focus by an edge-stopping
    /// diffusion instead of an even blur — a recursive filter run along the rows and the columns (the domain
    /// transform), each step costing more the more the picture's value changes across it, so the halo runs
    /// along a hill's line or the sea's edge and stops at a contrasting boundary — and the subject's own
    /// silhouette (the matte and the segmenter's, not the detectors' boxes) kept at full attention over it, so
    /// a thin limb at the edge of the focus is not lifted with the ground. Unset, the halo was a round blur of
    /// the subject's mask, a box around a figure (a detected coat's box) a dark island in the landscape, and a
    /// cat's hind paws paled with the ground.
    pub ma_flow: f32,
    /// BONE RYAKU (0..1, 0 = off): ryakuga, the abbreviated drawing — a bone line off the focus generalised:
    /// its chain simplified (Ramer–Douglas–Peucker) to a tolerance of `0.5 + 8·key·base·(1 − a)²` px, with `a`
    /// the chain's attention (narrowed by `ma_focus`), and the polygon rounded into arcs (two Chaikin passes), so
    /// a wheel's many-cornered rim at the edge of the sheet is one or two sweeps while the subject keeps every
    /// turn of its contour. Unset, a far line was a pale but exact drawing of every corner.
    pub bone_ryaku: f32,
    /// BONE FADE (0..1, 0 = off): the line register absorbed into the washes at the edge of the empty space —
    /// a bone chain whose attention `a` (narrowed by `ma_focus`) is under `key·0.5` is not drawn at all, neither
    /// line nor dot: the far detail is left to the wash's soft masses. Unset, a pale structural line in the
    /// distance broke the aerial perspective.
    pub bone_fade: f32,
    /// BOKASHI (0..1, 0 = off): the brush charged with water toward the edge of the focus — a mark's wetness
    /// raised toward 0.95 by `key·(1 − a)²`, with `a` the attention at the mark (narrowed by `ma_focus`), on the
    /// ladder and the bone; a wet mark is the flood's (see `flood`), so the flying white is gone there, the hairs
    /// even and the edge feathered: the fading image dissolved into the paper, not merely lightened.
    pub bokashi: f32,
    /// VELVET DARK (0..1, 0 = off): the velvet black held to a dark that is dark on the sheet, not only in the
    /// picture's own range — `velvet` lays its black only where the perceived (gamma) value of the target is
    /// also under this key (0.3 ≈ sRGB 77). On a white bird the darkest share of the range was the grey shadow
    /// under a wing, and it went down as a black patch.
    pub velvet_dark: f32,
    /// SLANT (0..1, 0 = off): the pencil's hatching laid at the hand's own angle where the picture has no
    /// direction of its own — on every pass after the block-in, where the structure tensor's coherence
    /// ((λ1−λ2)/(λ1+λ2))² over 1.5 brush radii is low (grass, rubble, a flat tone), off the hair and a face,
    /// the flow turned (in the double angle, its magnitude kept) toward the right hand's diagonal, lower left
    /// to upper right, by `key·(1 − coherence)`; each later pass crossed off it by `cross` (alternating sides).
    /// A coherent passage (an edge, a strand of fur) keeps its flow. Unset, a straight hatch line followed a
    /// flow that circled in every incoherent patch and the shading read as loops.
    pub slant: f32,
    /// GRADE (0..1, 0 = off): the pencil's tonal key — the half-tones lifted toward the paper on the
    /// perceived scale, `g' = g^(1/(1+key))` of the value's share of the paper's (gamma), the velvet black kept;
    /// a graphite drawing is keyed lighter than the photograph's luminance (a red fur is a dark grey by its
    /// luma, a pencil sets it as a middle one) and its darkest accents carry the depth.
    pub grade: f32,
    /// WEIGHT (0..1, 0 = off): the pencil's LINE — the bone line (`bone`) laid with a graphite point's own
    /// weight instead of a brush's hairline: `1 + 1.5·key·d` of its width and `1 + key·d` of its charge, `d`
    /// the darkness of the edge's darker side (a cloud's light rim keeps a light line), so a contour reads
    /// over the hatched tone (at the brush's width the line went into the hatching as one more
    /// grey hair, and the wheel, the doll, a chair's legs were lost in the rubble).
    pub weight: f32,
    /// CLIP (0..1, 0 = off): the hatch stopped at the picture's CONTOURS — the marks (the block-in's too) cut where
    /// their path crosses a line of the picture's edge map (the bone's Canny map, opened by the key: the strongest
    /// 3% of the gradients at 0, ~15% at 1), a mark riding along a contour kept, the subject's own marks not
    /// cut (a fur's every hair is an edge, and cut at each the fox's head broke into blank islands); so the shading of one form
    /// does not run out over its neighbour (a doll's light body hatched over by the rubble round it, a coat's
    /// strokes running on into the grass).
    pub clip: f32,
    /// TIERS (0..1, 0 = off): the hatching in TONAL LAYERS — the k-th hatched pass (0 along the form, then
    /// crossed by `cross`) lays its lines only where the target's perceived darkness exceeds
    /// `key · min(0.15 + 0.2·k, 0.75)` (thinning out by a hash over ±0.1 round it, so a half-tone fades from one
    /// layer to the next instead of breaking into blank islands): the lights left to the paper, the half-tones one set of lines along
    /// the form, the shadows crossed layer over layer — a draughtsman's tonal scale instead of one density of
    /// tangled lines everywhere. The block-in's broad marks go only where the deepest layer does (no toned
    /// ground: laid everywhere they were grey zigzag bands across the lights).
    pub tiers: f32,
    /// WAVER (0..1, 0 = off): the hand's pressure wavering along each mark — a slow noise of the mark's own
    /// swinging the pressure (and with it the graphite's reach into the grain) by ±half the key, the deposit by
    /// ±0.35 and the width by ±0.2 of it (see `BrushConfig::waver`), so a long line in the half-tones goes
    /// lighter and darker along its run instead of lying as one ruled grey.
    pub waver: f32,
    /// NIB (0..1, 0 = off): a steel nib and black ink — under `paper`, a mark is laid at the ink's full black
    /// and full pressure (the residual's charge pulled to the fullest by the key) and the sheet it answers is read
    /// over the mark's cell (a window of 1.2 radii), not at one pixel: the tone is how many black lines a cell
    /// takes, not how grey each one is, and the hatching stops when the cell's mean reaches the value. And the
    /// width held under the pass's nib (a 600th of the long side on the broadest pass, a third of it on the finest).
    pub nib: f32,
    /// NIBS (a count, 0 = off): pens of several sizes in layers — under `nib`, the ladder's passes drawn with
    /// this many pens, the k-th taking the k-th share of it coarse to fine: the broadest pen (a 500th of the sheet's
    /// long side) lays the masses' hatching first, the finest (an 1800th) the detail last, the ones between in a
    /// geometric series of widths, each mark at its pen's width; the passes shared out evenly on the log of
    /// their radius. 0 keeps `nib`'s smooth narrowing pass by pass.
    pub nibs: f32,
    /// NIB TONE (0..1, 0 = off): under `nibs`, each pen laid only where the tone is dark enough for its width — the
    /// pen at place q of the set (0 the broadest, 1 the finest) needs a darkness over `key·(0.12 + 0.6·(1 − q))`,
    /// the darkness read against the eye's attention (`ma`; `(1 − v)·(0.4 + 0.6·attention)`): the broad nib in the
    /// near deep shadows, the fine one on the lights' edge and the far ground, the lights themselves blank.
    pub nib_tone: f32,
    /// NIB WHITE (0..1, 0 = off): under `nib`, a cell is never asked darker than this share of the paper's
    /// light — the deepest shadow keeps the paper's flecks between its lines instead of closing to a flat black.
    pub nib_white: f32,
    /// NIB ANCHOR (0..1, 0 = off): under `nib`, the cast shadows that set things on the ground — after the hatching
    /// and the bone, on a grid of a 150th of the sheet, where the picture is much darker than its surroundings (a
    /// 90th of the sheet round it) or just under the subject's silhouette (a 120th of the sheet deep, fading down),
    /// a cluster of `1 + 3·strength` short black strokes at the middle pen's width along the edge there, crossed;
    /// so a figure and a stone sit on the slope instead of floating on the paper (stage `nib` in the score).
    pub nib_anchor: f32,
    /// NIB HINT (0..1, 0 = off): under `nib`, the shorthand in the lights — on a grid of a 60th of the sheet, a light
    /// cell (value over 0.6) with a sharp contrast of its own (a grass blade, a pebble's rim) takes, by `key·contrast`,
    /// one short dash (0.4–0.8 of the cell) at the finest pen's width along the edge at its sharpest point; a sunlit slope keeps its ground
    /// without being hatched (stage `nib` in the score).
    pub nib_hint: f32,
    /// NIB FACE (0..1, 0 = off): under `nib_tone`, a face drawn with the finest pen alone (the key's share of the
    /// broader pens' marks there dropped), its value lifted toward the paper by `0.4·key` and its paper floor raised
    /// by `0.3·key` over `nib_white` — the cheek, the cheekbone and the nose left to the paper and a delicate line.
    pub nib_face: f32,
    /// NIB GAP (0..1, 0 = off): under `nib`, a band of white kept along the picture's long contours (the edge map's
    /// chains longer than a 30th of the sheet, widened by `key` · a 400th of the sheet each side) — a mark cut where it
    /// enters the band and not laid when it starts in it — so an umbrella, a sleeve and a body in one shadow are parted
    /// by a sliver of paper; not on the hair/fur region (as `clip`), where every lock is a contour.
    pub nib_gap: f32,
    /// NIB RELIEF (0..1, 0 = off): under `nib`, a place lighter than its surroundings (its value against the mean
    /// over a 100th of the sheet — the bulge of a sleeve, a fold's ridge) lifted toward the paper by `3·key` times its
    /// excess, so the hatching thins on the form's highlights and a coat is modelled, not filled; not on the hair/fur
    /// region, where every lock is lighter than its neighbour and the mass broke into white islands.
    pub nib_relief: f32,
    /// NIB FAR (0..1, 0 = off): under `nib_tone`, off the eye's attention (`ma`, under `0.6·key`) every pen is the
    /// finest — the marks the broad passes lay there kept, drawn at the finest pen's width, gated by their own pen's tone — so
    /// the sky's clouds, the far trees and the far slope are light broken lines, neither heavy nor bare.
    pub nib_far: f32,
    /// NIB OCCLUDE (0..1, 0 = off): under `nib_tone`, the broadest pen laid only in the cast shadows (where the
    /// `nib_anchor` field is over `0.3·key`) — under the figure and the umbrella — the other darks left to the middle
    /// pen, so the weight of the line sits where the forms meet the ground.
    pub nib_occlude: f32,
    /// NIB EYES (0..1, 0 = off): under `nib`, an animal's eyes — found in the picture by their catchlights (a light
    /// point ringed by the dark, inside the hair/fur region; the detector does not name an animal's eye) — taken out of
    /// the fur by the key's share, kept from the hatching, and drawn after the bone with the finest pen in pure black:
    /// the picture's contours inside them (edge strength `key`) and their dark in close diagonals, the catchlight left
    /// to the paper — so the look is not lost in the coat's dark markings.
    pub nib_eyes: f32,
    /// NIB FUR (0..1, 0 = off): under `nib`, on the hair/fur region the coat's half-tones lifted toward the paper by
    /// `0.5·key` (the full share from a value of 0.35 up, none under 0.12 — the dark markings kept), a place lighter
    /// than its surroundings at the body's scale (the value over a 60th of the sheet against the mean over a 15th — a
    /// flank, a back in the light) lifted further by `3·key` times its excess, and the fur's paper floor raised by
    /// `0.15·key`, so a coat thins smoothly on its bulges instead of filling to one black mass (`nib_relief`'s islands
    /// at a lock's scale).
    pub nib_fur: f32,
    /// NIB LIFT (0..1, 0 = off): under `nib`, on a figure's subject (a face found) off the fur, the half-tones lifted toward the paper by
    /// `0.8·key` (the full share from a value of 0.35 up, none under 0.12) — a figure's coat and face left to their
    /// contours and the hatching kept to the deep shadows of the folds.
    pub nib_lift: f32,
    /// NIB CONTACT (0..1, 0 = off): under `nib_anchor` and `nib_occlude`, the cast shadow under the subject made a
    /// contact patch — a band `key` · a 40th of the sheet deep below the silhouette (not a 120th), on a ground up to a
    /// value of 0.8 — so a seated figure rests on the ground instead of hanging over it.
    pub nib_contact: f32,
    /// NIB FIGURE (0..1, 0 = off): under `nib`, a figure (the silhouette, when a face is found) drawn as a pen draws
    /// a figure, not by the passes' marks along the flow (the key's share of them dropped there). Before the bone its
    /// tone is laid in pen hatching read from the picture as given (the armature has no features): the figure's value
    /// from its own 5th to 95th percentile (a coat that is a middle grey goes to the figure's dark), the face's from its
    /// own 30th to 95th lifted toward the paper by `0.8·nib_face` (its lit side blank). In cells a 40th of the sheet
    /// across (their edges wavering) the lines run along the form — the structure tensor's direction, the folds' way —
    /// a 280th of the sheet apart: along it under 0.85, across it at 60° under 0.55, and closed to a solid black under
    /// 0.25; every line broken at the picture's contours, so a fold's edge stays white, and cut into pen strokes at most
    /// a 45th of the sheet long, each a little off its line and angle. The face's features — the profile, the brow, the
    /// eye, the lips — drawn as the picture's contours on it; the figure's bone line made bold (`1 + 1.5·key` wide).
    pub nib_figure: f32,
    /// NIB OBJECT (0..1, 0 = off): under `nib_figure`, the rigid things the figure holds (an umbrella, a stick, a cane —
    /// found by the detector, their extent by SAM from the box's centre) joined to the figure: off the passes' marks,
    /// hatched with it, and their own contours (the picture's edges inside them) drawn as firm black lines
    /// `1 + 1.5·key` times the hatching's width, so a closed umbrella's ribs and handle are drawn, not textured.
    pub nib_object: f32,
    /// NIB TUFT (0..1, 0 = off): under `nib`, the near grass drawn as tufts — only on the grass the detector found
    /// (CLIP ViT-L over a grid of crops a fifth of the sheet across, a cell grass where the grass labels outweigh twenty kinds of ground, sky and wall by 0.6; none found, no tufts), read from the picture as given, off the
    /// figure and the sky (a cell half over 0.85), on a grid of a 70th of the sheet, a cell whose texture is strong (from the frame's 60th to its 90th percentile) and stands upright (its
    /// horizontal gradient 1.15 to 1.5 times its vertical), weighted by the eye's attention, takes by `key`·its strength
    /// a fan of 4 to 9 blades (more in the dark) rooted along its foot, leaning with the grass, bent outward, at the
    /// middle pen tapering to the finest, one and a half to three cells long by its strength; under the figure's
    /// silhouette (the contact band) twice as many, longer, and always. The passes' marks starting in such a cell
    /// dropped by `0.6·key`·its strength.
    pub nib_tuft: f32,
    /// NIB CLOSE (0..1, 0 = off): under `nib_figure`, the figure's outline kept whole — along the silhouette's
    /// boundary, a stretch not within 3 px of a drawn bone line and shorter than `key`·a 40th of the sheet, between two
    /// drawn stretches, laid as a thin black line (the finest pen, at least a pixel), so a shoulder or a back does not
    /// fall apart into separate strokes.
    pub nib_close: f32,
    /// NIB MASS (0..1, 0 = off): under `nib_tuft`, the dark grass merged into one mass — a tuft's cell whose mean value is
    /// under 0.6 takes `1 + round(4·key·dark)` fans (dark the smoothstep of `(0.6 − value)/0.25`, times the smoothstep
    /// of the share of its eight neighbours that are tufts too, from 0.25 to 0.6 — a clump, not a lone tuft), each
    /// rooted anywhere in the cell, so their blades cross and close into a solid dark; and a cell of a dark clump (under
    /// 0.6, half its neighbours tufts) drawn whatever its own strength, so the clump is whole.
    pub nib_mass: f32,
    /// NIB RIM (0..1, 0 = off): under `nib_figure`, the figure's silhouette (and what it holds) outlined in one line of
    /// changing weight that parts it from the ground — its boundary laid in pieces of a 60th of the sheet, each as wide
    /// as the finest pen times `1 + 4·key·shade`, shade the darkness just inside it, so the line swells on the shadow
    /// side and thins on the lit one.
    pub nib_rim: f32,
    /// NIB POOL (0..1, 0 = off): under `nib_figure`, the contact shadows closed to a solid black with the broad pen —
    /// off the figure, within `key`·a 40th of the sheet under its silhouette (and under what it holds) — only under the
    /// lowest quarter of each figure's height (each connected piece of the mask), where it meets the ground, not under an
    /// arm or a hem in the air — where the square
    /// root of the nearness times `0.5 + 0.5·`the smoothstep of `(0.75 − value)/0.35` passes 0.5, close lines of the
    /// broadest pen (as far apart as they are wide) — the band deep where the ground is dark, half as deep where it is
    /// light.
    pub nib_pool: f32,
    /// NIB AIR (0..1, 0 = off): under `nib_tone`, the far planes kept airy — where the eye's attention is under 0.5, the
    /// passes' marks dropped by `0.7·key` times the smoothstep of `(0.5 − attention)/0.4`, so the far hills are a few
    /// fine lines, not a hatched mass.
    pub nib_air: f32,
    /// NIB PEOPLE (0..1, 0 = off): under `nib_figure`, every person is the figure, not only the one whose face prompted
    /// the silhouette — OWL-ViT's boxes for a person, each box's extent by SAM (clipped to the box), and every detected
    /// face, joined to the figure, each face read against its own range; so a group is drawn whole, every face lit as
    /// its own, and a figure seen from behind (no face) is drawn as one.
    pub nib_people: f32,
    /// NIB VALUE (0..1, 0 = off): under `nib_figure`, the figure's tone read against the whole picture as well as against
    /// its own range — a value's rank (the share of the sheet darker than it) lifts the figure's stretched tone toward
    /// itself by `key` times the smoothstep of `(rank − 0.35)/0.3`, so clothing that is light in its scene (a shirt in
    /// a dark room, however grey in the photograph) keeps to one layer or paper, and only the scene's darks are stretched.
    pub nib_value: f32,
    /// NIB FINE FACE (0..1, 0 = off): under `nib_figure`, a small face (its patch under an eighth of the sheet across)
    /// drawn with the finest pen — its hatching at the finest width and twice as close, laid only under thresholds lowered
    /// by `0.04·key` (the skin left to the paper, its hair and deep shadow hatched), its features traced from the
    /// picture's contours ranked among the small faces' own gradients (not the whole sheet's) and opened by half the
    /// key, so a group's faces keep their eyes and noses on a soft photograph.
    pub nib_fine_face: f32,
    /// NIB SCATTER (0..1, 0 = off): under `nib_tuft`, the tufts rooted by Poisson-disk sampling instead of one per
    /// cell — the spacing modulated by a smooth noise (clumps where it is high, clearings where it is under `0.3·key`),
    /// so the grass grows in organic clusters, not on a grid.
    pub nib_scatter: f32,
    /// NIB FALLOFF (0..1, 0 = off): under `nib_pool`, the contact shadow fades — solid black only in the narrow band
    /// where the figure meets the ground (a 600th of the sheet), beyond it broken into long dashes of the broad pen,
    /// fewer with the distance (kept by `key·(1 − d)^1.5`), so the shadow dissolves into the ground.
    pub nib_falloff: f32,
    /// NIB DEPTH (0..1, 0 = off): under `nib_tone`, the eye's attention capped by the picture's depth (Depth-Anything:
    /// near the smoothstep of `(z − 0.1)/0.4`, by the key's share) — the far hills and the horizon taken by `nib_far`
    /// and `nib_air` to the finest pen and a few lines, so the planes part as on the near ground.
    pub nib_depth: f32,
    /// NIB LUSH (0..1, 0 = off): under `nib_tuft`, the grass the detector found grown thick — every grass cell a tuft's
    /// candidate by its darkness whatever its texture (its strength floored at `0.7·key` times the smoothstep of
    /// `(0.65 − value)/0.3`), the scattered roots closer (the Poisson
    /// radius cut by `0.2·key` in the light grass to `0.4·key` in the dark), so a meadow reads as a dense mass of blades
    /// and its shadowed clumps cross into one deep dark, not a few clumps.
    pub nib_lush: f32,
    /// NIB GLINT (0..1, 0 = off): the specular highlights kept to the paper — a small spot of the picture over
    /// `1 − 0.08·key` and brighter than its surroundings by a tenth (no larger than a 30th of the sheet across: a sky or
    /// a white wall is not a glint), opened by a 400th of the sheet; no mark of the passes, the figure's hatching or the
    /// light's dashes crosses it (a mark cut where it enters), so the metal and the glass keep their shine.
    pub nib_glint: f32,
    /// NIB WASH (0..1, 0 = off): INK AND WASH — the ladder laid twice. First as the ink's washes: the nib off and
    /// every hatching key with it (hatch, cross, tiers, slant, solid, span, edge, clip, lights) and the pen's lighter
    /// key (grade), the passes the medium's own brush, bled among themselves; then over the dried sheet as the pens'
    /// (stages `pen-…`): the nib
    /// on, the `pen` mark, the washes' keys off (boneless, hake, wash_in, bokashi), no bleed. The washes carry the
    /// tone up to the key's share of the way from the paper: the pens draw the rest — the lines, and the darks
    /// beyond the washes' reach. The bone and the pen's accents follow as in pen-ink.
    pub nib_wash: f32,
    /// NIB WASH DARK (0..1, 0 = off; with `nib_wash`): the pens' passes hatch only the darks — a cell asked lighter
    /// than `1 − 0.8·key` of the paper (on the eye's scale) is left to its wash, the pen there only the bone's contour and the accents.
    /// The wash carries the tone; the pen draws the line and deepens the shadows.
    pub nib_wash_dark: f32,
    /// NIB WASH DETAIL (0..1, 0 = off; with `nib_wash_dark`): where the picture has a structure of its own on a LINE — a
    /// step of value across four pixels over `0.1·(1 − 0.6·key)` where the gradients round it agree on a direction (a
    /// book's spine, a window's bar, a fold; not a gravel, a heap of rubble, a meadow — texture stays the wash's) — the
    /// wash stops short, lifted `0.5·key` of the way to the paper, and the finer pens (the finest and the middle) are
    /// let back into the half-tones and the lights: the detail drawn in lines over the wash.
    pub nib_wash_detail: f32,
    /// NIB WASH LINE (0/1, 0 = off; with `nib_wash`): the bone traced from the picture itself, not the armature —
    /// under --new the armature is smoothed, and a face, a beard, a book's spine are gone from it: the pen's line
    /// draws them over the washes.
    pub nib_wash_line: f32,
    /// NIB WASH SWELL (0..1, 0 = off; with `nib_wash`): the bone line's width by the strength of its edge — the finest
    /// pen's hair where the step is soft, swelling toward a 400th of the sheet where it is hard,
    /// `w = w_fine + (w_max − w_fine)·key·(1 − e^(−3·|∇v|))` over the chain's mean gradient: a line that breathes with
    /// the form, not one even wire.
    pub nib_wash_swell: f32,
    /// NIB WASH AIR (0..1, 0 = off; with `nib_wash` and the depth map, `nib_depth`): the aerial perspective by the wash —
    /// toward the far planes the washes lightened toward the paper by up to half the key, the bone's line thinned and
    /// past a tenth of the depth dropped: the far hills and the sky soft tonal pools, not fewer strokes.
    pub nib_wash_air: f32,
    /// NIB WASH SOFT (0..1, 0 = off; with `nib_wash`): the bands parted — the washes painted from the picture's LOW
    /// band, every wash pass's reference blurred by `key·long/150`: the masses, the shadows and the far planes soft
    /// tonal pools; the high band (the edges, the texture) left to the pens and the line, so a heap of rubble is a
    /// wash under a few lines, not a mottle of small washes.
    pub nib_wash_soft: f32,
    /// NIB WASH GAMMA (0..1, 0 = off; with `nib_wash`): the wash's density raised before the bleed spreads it — a wash
    /// asked `paper·(l/paper)^γ`, `γ = 1 + 0.5·key`: the half-tones and the darks deepened, the paper and the black
    /// kept, so the ink the bleed and the feathering carry over a wider area does not leave the sheet paler than
    /// the picture.
    pub nib_wash_gamma: f32,
    /// NIB WASH FLAT (0..1, 0 = off; with `nib_wash_dark`): the pens' hatching in the darks only where the dark has an
    /// edge or a texture of its own — the picture's gradient over a 9-pixel window above its own median (the key's
    /// share of it): a flat dark, a monolithic rock face or a deep shadow, is the wash's solid pool, not a hatch.
    pub nib_wash_flat: f32,
    /// NIB WASH SCURVE (0..1, 0 = off; with `nib_wash`): the wash's density by a contrast S-curve instead of a power —
    /// on the eye's scale `x = (l/paper)^(1/2.2)` taken to `x^a / (x^a + (1−x)^a)`, `a = 1 + key`: the half-tones
    /// below the middle deepened and the light ones kept light, the bleed's pale haloes made up without a power's
    /// pull of every dark toward one black. Applied after `nib_wash_gamma`.
    pub nib_wash_scurve: f32,
    /// NIB WASH VARY (0..1, 0 = off; with `nib_wash_gamma` or `nib_wash_scurve`): the washes' deepening held back where
    /// the picture is busy — scaled by `1 − key·(1 − e^{−σ²/σ²_med})`, `σ²` the picture's local variance of value over a
    /// 9-pixel window and `σ²_med` its median: a sky or a rock face takes the full density, a face or a fold of cloth
    /// keeps a transparent wash the pen then draws over.
    pub nib_wash_vary: f32,
    /// NIB WASH MASS (0..1, 0 = off; with `nib_wash`): the ink's mass kept across the bleed — before each wash after the
    /// block-in the sheet's darkness on the eye's scale, `Σ(1−l^(1/2.2))`, is measured against the picture's, and the
    /// wash's tone deepened by a gamma `paper·(l/paper)^r`, `r` the key's share of the loss (the ratio, held to 1..2):
    /// a tone mapping, not a scaling of the depth (scaled and clamped to black, every lit face in a crowd went to one
    /// black). `nib_wash_vary` holds the deepening back where the picture is busy.
    pub nib_wash_mass: f32,
    /// NIB WASH HOLD (0..1, 0 = off; with `nib_wash`): a wash held at the picture's hard edges toward a light — no hard
    /// clip at every contour (that was a heap of dabs), but a soft one by the ink's conductance `c = e^{−|∇I|/K}` (`K`
    /// four times the picture's median gradient): a wash mark is cut where its footprint (its middle and both sides,
    /// 0.8 of its half-width off) meets a pixel of `c < 0.5` lighter than the mark's own value by more than
    /// `0.35 − 0.2·key`. The ink runs through a dark coat but stops at a white collar or the profile of a face.
    /// On the figures only (the people's mask, the faces): on a landscape every blade and stone is such an edge.
    pub nib_wash_hold: f32,
    /// NIB WASH DEBT (0..1, 0 = off; with `nib_wash_mass`): the ink's loss measured only where the sheet is still lighter
    /// than the picture — a passage the bleed has already over-darkened can give no ink back, and counted in the
    /// sheet's darkness it hid the debt of the half-tones beside it.
    pub nib_wash_debt: f32,
    /// NIB WASH DRY (0..1, 0 = off; with `nib_wash`): the dark register laid by a DRY BRUSH over the washes — in
    /// sumi-e the dark accents go over the dried washes with less water, and they hardly spread. The washes' fine
    /// passes (the brush within 2.5 of the smallest but not the smallest itself, the last of the bleed's taper)
    /// read the picture itself blurred by their radius — not the smoothed armature or its low band — and lay
    /// only where its value is under 0.45 on the eye's scale; their ink is the picture's own value past the
    /// washes' curves (the S-curve, the mass, the air), the key the share of the way from the paper to it. Not
    /// detail passes: neither the focus nor the coherence gates them, and their mark is a DRAG along the mass's
    /// form (the flow smoothed over three radii, the length three times the profile's), not the detail's short
    /// touch — read sharp and short, the register was a heap of curls over every bush; and off the subject the
    /// drag is the flat brush's level sweep (`hake`), as the washes' — along the texture's own flow it was a rope
    /// winding round every bush. Toward the far planes (`nib_wash_air`) the dry brush lays less: its ink lightened
    /// by 0.35 (the washes' by 0.5, so the drag still shows over its wash), its register narrowed to the darker
    /// (0.34 at the farthest) — as dense at the horizon as at the feet, the drags were one flat wall.
    pub nib_wash_dry: f32,
    /// NIB WASH CAP (0..1, 0 = off; with `nib_wash`): the sheet's CAPACITY — a wash asks no tone darker than the
    /// key's share of the picture's own value at its point (in linear light), and where the sheet has reached it
    /// no mark is laid: the pool a contrast curve, a mass gamma or the low band's blur had pushed past the picture
    /// (the dark hair's ink over a lit face, a coat gone to one black) is ink the paper will not take.
    pub nib_wash_cap: f32,
    /// NIB WASH TONE (0..1, 0 = off; with `nib_wash`): the sheet corrected at OUTPUT, not in the washes — a wash
    /// may stay light as it is laid; at the end the sheet's mean value is set against the picture's and its
    /// mid-tones pulled down by `c·(1 − 2t·c(1−c))` (the paper and the blacks kept, the pull `t` solved by
    /// bisection so the two means meet), the key the share of that pull. Recorded in the score's header (`tone=`),
    /// so the replay prints the same sheet.
    pub nib_wash_tone: f32,
    /// NIB WASH DRAW (0..1, 0 = off; with `nib_wash`): the DRAWING FIRST — the pens' passes laid on the bare sheet,
    /// as pen-ink lays them, and the washes after, over the dried drawing. Laid after the washes (the key off), a
    /// pen's mark passed the sheet's gate only where the wash had stopped short of the tone — the pens drew the
    /// remainder, a line at the edges and nothing of the form (a station's pens laid a ninth of their strokes; the
    /// faces, the folds, the books of the pen-ink drawing were gone under the wash). Drawn first, the hatching
    /// carries its tone and the wash, reading the sheet over its mark's cell as a pen does, lays only what the
    /// hatching left. The washes' bleed tapers over the washes alone, wherever they stand in the schedule.
    pub nib_wash_draw: f32,
    /// NIB WASH PLANES (0 = off, else the count, 2..8; with `nib_wash`): the washes laid as FLAT PLANES — the
    /// picture (smoothed over a 256th of the sheet) quantised, on the eye's scale between the paper and black,
    /// to this many bands: the lightest the paper itself (no wash there), each other its band's lighter edge
    /// (four: the paper, and washes at three quarters, a half and a quarter of the value). The planes are laid as
    /// WASHES — area fills, not strokes: one per plane from the lightest to the darkest, each covering every
    /// plane as dark as itself (the washes stack, as three washes of one ink do), the INK ALONE (the palette's
    /// darkest pigment) thinned, its charge found on scrap paper so the stack reads the plane's tone (a grey's
    /// mixture carried white — a fog over the drawing, and at a light enough grey white alone, laid as a slab
    /// lighter than the paper); a plane's mask opened by a pixel or two and its specks dropped
    /// (a plane is a mass), its edge feathered a 500th of the sheet. The register's stroke passes then lay
    /// nothing (the dry brush's pass excepted, `nib_wash_dry`). A wash reading the smoothed picture with a
    /// soft brush tracked its every gradient — a smear of tone with no edge anywhere (the quantised reference
    /// alone did not help: the flooded sweeps and the bleed lost the edge again); a pool has sides, and the
    /// modelling within it is the pens'.
    pub nib_wash_planes: f32,
    /// NIB WASH EDGE (0..1, 0 = off; with `nib_wash_planes`): a plane's WET EDGE by the picture — the fill's bloom
    /// wide where the picture's gradient is gentle under the plane's boundary (the boundary there is the
    /// quantiser's, not the picture's: a sky's gradation cut into bands, each a hard-edged blot — the camouflage
    /// of set 10), and the base 500th of the sheet where the picture has an edge (a roof against the sky: the
    /// wash stops there as the painter's does). The edge's width varies along the boundary, ring point by ring
    /// point: the base times 1 + 12·key·(1 − g)², g the picture's gradient over the plane's smoothing radius
    /// against a step of a half of the value — a 13th of a 100th of the sheet at the key where the picture is
    /// flat. Recorded per point, replayed to the bit.
    pub nib_wash_edge: f32,
    /// NIB WASH DEPTH (0..1, 0 = off; with `nib_wash_planes`, and the depth map of `nib_depth`): AIR on the planes —
    /// before the quantising, the picture's value lifted toward the paper by a half of the key where the depth
    /// map is far (as `nib_wash_air` reads it: 1 at the farthest plane, 0 from a half of the depth in), so a far
    /// hill falls into a lighter band than a near one of the same value, and the farthest into the paper. The
    /// set-10 planes were quantised by one scale everywhere, the far hills as dark as the near.
    pub nib_wash_depth: f32,
    /// NIB WASH SCALE (0..1, 0 = off; with `nib_wash_planes`): the planes at DIFFERENT SMOOTHINGS — the lightest
    /// plane from the picture smoothed coarsest (a 256th of the sheet times 1 + 3·key: the large masses of light
    /// and shade, not a scatter of blots), the darkest from the finest (a 256th over 1 + 3·key: the deep shadows
    /// where they are, under the pens' line), the planes between by a geometric ladder; a darker plane kept
    /// within the lighter (the washes stack). Off, every plane from the one 256th.
    pub nib_wash_scale: f32,
    /// NIB WASH LEVELS (0..1, 0 = off; with `nib_wash_planes`): the planes' THRESHOLDS BY THE PICTURE — moved from
    /// the even steps of the value (a quarter, a half, three quarters) toward the picture's own quantiles (the
    /// values that cut its smoothed, air-lifted histogram into equal shares), by the key. A station's cliffs
    /// lay within one step and came out one grey plane; cut at the picture's quantiles, the planes divide
    /// what the picture has most of. Each plane's tone moves from its band's lighter edge toward the band's
    /// mean value by the key — a plane cut at a quantile is a quarter of the sheet, and the stack reads the
    /// picture's mass only at the mean (at the lighter edges a grassy hillside came out a fifth too light).
    pub nib_wash_levels: f32,
    /// NIB WASH FACES (0..1, 0 = off; with `nib_wash_planes` and the face mask): the FACES THE PEN'S — a face's
    /// value lifted toward the paper by a quarter of the key before the quantising, so it falls into a lighter
    /// plane than the figure's dark mass and the fill does not bury the drawing; and the pens' hatching on a
    /// face freed of the dark gate (`nib_wash_dark`) by the key, so the pen models the face as pen-ink does —
    /// the eyes, the profile, the folds — before any wash is laid.
    pub nib_wash_faces: f32,
    /// NIB WASH RAG (0..1, 0 = off; with `nib_wash_planes`): a plane's CONTOUR TORN — its rings resampled a base
    /// feather (a 500th of the sheet) apart and every point moved along the boundary's normal by a noise along
    /// the ring (nodes two feathers apart, interpolated), up to one and a half feathers times the key times the
    /// plane's darkness on the eye's scale (the dark plane's edge torn most, the lightest's a quarter as much):
    /// a dry brush's ragged edge, where the cut mass was a blot with an optical blur round it. The bloom
    /// (`nib_wash_edge`) rises from the torn edge. (Bristle tracks — streaks out along the normal, their reach
    /// cut by a noise — were a comb of rays round every light hole, at any period.) In the rings as recorded,
    /// so the replay lays the same.
    pub nib_wash_rag: f32,
    /// NIB WASH MASK (0..1, 0 = off; with `nib_wash_planes`): the PEN KEPT OUT OF THE DARKEST PLANE — a pen's
    /// value lifted toward the paper by the key times the square of how deep the point lies in the darkest
    /// plane (1 through its band, falling across the band above it; the planes' thresholds as the washes cut
    /// them), so past the dark gate (`nib_wash_dark`) no pen lands there and the mass is the wash's blot alone:
    /// with the washes the ink alone (transparent), the hatching under the darkest plane showed through it —
    /// a fox's body a black hatched slab, a cliff's crack a net. The lines (`nib_wash_detail`) and the faces
    /// (`nib_wash_faces`) are not masked.
    pub nib_wash_mask: f32,
    /// NIB WASH HALF (0..1, 0 = off; with `nib_wash_dark`): the HALF-TONES' PEN — the dark gate opened for the
    /// finest pen alone, on the passes laid along the form (not the crossing ones), on the near planes (the depth
    /// map's near half), where the picture has a texture (`nib_wash_flat`'s gate): from the gate's edge up by
    /// three quarters of the key of the way to the paper on the eye's scale (0.52 to 0.88 at `nib_wash_dark`
    /// 0.6 and key 1), the pen asked only a half of the key's share of the residual there, the rest the wash's.
    /// A single open hatch of the fine pen in the half-tones — a meadow, a gravel, a roof's tiles — where the
    /// gate had left nothing but the planes' five flat tones.
    pub nib_wash_half: f32,
    /// NIB WASH GRAIN (0..1, 0 = off; with `nib_wash_planes`, three planes or more): the SMALL DARKS KEPT —
    /// the picture's two darkest bands at the darkest plane's smoothing, their pieces over a `(1 + 15·key)`th
    /// of the plane's least area (a 4000th of the sheet; 65 pixels at key 1), added to every plane's mask past
    /// its opening and kept past its least area (so the stack carries them to the plane's full tone), their
    /// edge the base feather with no bloom (bloomed, a speck was a soft black blot): a stone,
    /// a window, a trunk, a stack of rubble — the darks under a 32-pixel mass the opening and the least area
    /// swept away. The light specks stay swept: a plane is a mass, and a dark speck is the drawing's.
    pub nib_wash_grain: f32,
    /// NIB WASH GRADE (0..1, 0 = off; with `nib_wash_planes`): a TONE WITHIN EACH PLANE — over a plane's own
    /// band (not the darker planes' it covers), two sub-steps: the band's darker two thirds filled again and
    /// its darkest third over that, each a third of the key's share of the step to the plane below darker
    /// (opened a pixel, its pieces over a quarter of the least area, the edge the base feather, no tearing):
    /// three tones to a plane, fifteen to five planes — the modelling inside a plane that five flat tones had
    /// none of: a cliff's strata, a heap of rubble's shadow side. On a SWEPT plane with the agent
    /// (`nib_wash_sweep`, `nib_wash_agent`) the key is instead the mark's TARGET — between the plane's flat tone
    /// (0) and the picture's own value at the darkest plane's smoothing (1): at 1 the sheet was the picture
    /// through frosted glass, at 0 a flat mass laid by the brush; a half keeps a plane's body with the
    /// picture's modelling inside it.
    pub nib_wash_grade: f32,
    /// NIB WASH PROFILE (0..1, 0 = off; with `nib_figure` and `nib_fine_face`): EVERY FACE DRAWN AS A SMALL ONE —
    /// whatever its size, the figure's hatching on it at the finest width and twice as close, its layers' thresholds
    /// lowered by `nib_fine_face`, its features traced among the faces' own gradients with the hair and the
    /// beard flattened out of the field (their edges took the face's whole budget of gradients, and a soft eye
    /// never ranked); the hatching's two layers dropped on a face, the wash modelling it and the pen keeping
    /// only the solid dark (the eye's socket, the nostril, the mouth's shadow) and the features — a face in a
    /// few lines, as the ink painters drew it; and on a face the ladder's dark gate opened (`nib_wash_faces`)
    /// for the finest pen alone. The coarse cross-hatch on a face at the figure's width (a 900th of the sheet,
    /// lines a 280th apart, three directions) read as scars; one way along the form, as a wire mesh.
    pub nib_wash_profile: f32,
    /// NIB WASH STREAK (0..1, 0 = off; with `nib_wash_rag`): the TEAR ALONG THE BRUSH'S SWEEP — the noise along a
    /// plane's contour advances slower where the contour runs with the sweep (the horizontal, as `hake`), by
    /// `1 + 6·key·|cos θ|`, so its fibres stretch into long tears there and stay a short comb where the contour
    /// crosses the sweep; and the noise sharpened toward its extremes by `1 + 2·key`, so a tear is a tooth, not
    /// a ripple. A dry brush's edge is drawn out along its travel; an isotropic tear was a jelly's rim.
    pub nib_wash_streak: f32,
    /// NIB WASH SWEEP (0..1, 0 = off; with `nib_wash_planes`): a plane laid as WIDE WET STROKES, not a fill — the
    /// plane's mask is only where the brush goes. Streamlines of the picture's flow (the structure tensor of the
    /// plane's smoothed image, along the isophotes; the horizontal where the picture has no direction, as `hake`)
    /// are traced from a grid of seeds a 0.6 width apart inside the mask eroded by a third of the width, up to an
    /// eighth of the sheet each way, and laid as flooded marks (wetness 0.9: the hairs evened by the water, the
    /// ends faded, the halo `nijimi`'s) a 96th of the sheet wide on the lightest plane and half that on the
    /// darkest; the ink along a mark FOLLOWS THE PICTURE under it — the mark cut into pieces two widths long,
    /// each ramped from the tone at its start to the tone at its end, the plane's tone raised to a power from
    /// a half at the band's light edge to one and a half at its dark edge, and each mark asked the overlap's
    /// root of it (the marks stack by five thirds). The overlaps accumulate, so the plane carries a gradient and
    /// a brush's grain where the fill carried one flat tone with a terrace at its edge; the wet edge, the tear
    /// and the sub-steps (`nib_wash_edge`, `nib_wash_rag`, `nib_wash_grade`) do not apply to a swept plane.
    /// Recorded as marks with a ramp (`mix2`), replayed to the bit. ALONE NOT A WASH: in pieces two widths long
    /// the marks were a chain of beads (the flood fades every mark's ends); whole, along the isophotes, they
    /// were black worms winding over the form, the mass gone to black (the overlaps past the root's count)
    /// and the picture's structure lost. With `nib_wash_agent` the mark gives its ink by the sheet under it
    /// and the overlaps stop darkening: the plane is then a wash — a graded mass with the brush's grain, no
    /// terrace at its edge, the tone held.
    pub nib_wash_sweep: f32,
    /// NIB WASH WATER (0..1, 0 = off; with `nib_wash_planes`): a LAYER OF WATER the ink seeps through — before a
    /// plane is laid the sheet is wetted (a fill of clear water, no ink) over the plane's mask grown by the key
    /// times a 64th of the sheet (a swept plane's by the key times the mark's width); and at the end of the planes' pass the ink that pass laid (and only it: the
    /// drawing under the washes stays where the pen put it) SEEPS — a cellular diffusion over the wet cells by
    /// their wetness, `24·key` steps — so a dark plane's ink runs out into the wet light one beside it in a
    /// soft halo (nijimi) that no softening of a polygon's edge gives; dry paper takes none. In the header
    /// (`water`), the replay seeps the same. The halo a fill's wet edge (`nib_wash_edge`) imitated with a
    /// dilation is here the water's own.
    pub nib_wash_water: f32,
    /// NIB WASH AGENT (0..1, 0 = off; with `nib_wash_sweep`): the MARK LOOKS AT THE SHEET UNDER IT — a swept
    /// plane's mark carries a profile of the picture's value along its path (the smoothed, lifted value the
    /// planes are cut from, in the score as `prof`), and at every pixel a hair gives its ink by the residual
    /// between the sheet there, as it is at that moment, and the target: `(C − T)/C · (1 + 4·key)`, nothing
    /// where the sheet has reached the target. Overlaps and seams stop darkening — the mark's charge is the
    /// plane's tone, and what has that tone under it takes no more — and the tone within a plane follows the
    /// picture point by point where the ramp (`mix2`) gave one line start to end. In the header (`agent`);
    /// the replay reads the same profile against the same sheet.
    pub nib_wash_agent: f32,
}

// ── The medium table (§6.2). P1 executes oil-direct + gouache; the rest are declared for the generator and
//    land as later phases wire their physics. ────────────────────────────────────────────────────────────
pub const OIL_DIRECT: MediumProfile = MediumProfile {
    name: "oil-direct",
    value_direction: ValueDirection::MidOut,
    opacity: Opacity::Opaque,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Hours,
    stage_budget: 1,
    // Modest pickup: alla-prima keeps marks distinct (sitting on top), not smeared into one another. High
    bleed: 0.08,
    body: 1.0,
    // 0.3, not 0.6: measured on an 11-image bench, the relief relight was the largest remaining source of
    // invented texture on smooth passages (edges 0.081 -> 0.059, dark flecks 0.026 -> 0.019, structure kept
    // 0.099 -> 0.085 at 0.3); it still reads as oil. 0 matches a flat reference painting best (`--impasto 0`).
    impasto: 0.3,
    chroma: 1.08,
    dry_shift: 0.0,
    granulate: 0.0,
    sheen: 0.15,
    lift: 0.8,
    default_palette: "zorn",
    broken: 0.35,
    contour: 0.0,
    // pickup drags wet paint and reads as a hazy smear.
    pickup: 0.3,
    families: Families::Split,
    subtractive: true,
    mark_model: MarkModel::Continuous,
    mark: MarkCharacter::BRUSH,
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const OIL_INDIRECT: MediumProfile = MediumProfile {
    name: "oil-indirect",
    value_direction: ValueDirection::DarkToLight,
    opacity: Opacity::Mixed,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Days,
    stage_budget: 8,
    pickup: 0.4,
    bleed: 0.1,
    body: 0.9,
    impasto: 0.4,
    chroma: 1.05,
    dry_shift: 0.0,
    granulate: 0.0,
    sheen: 0.12,
    lift: 0.7,
    default_palette: "zorn",
    broken: 0.25,
    contour: 0.0,
    families: Families::Split,
    subtractive: true,
    mark_model: MarkModel::Continuous,
    mark: MarkCharacter::BRUSH,
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const GOUACHE: MediumProfile = MediumProfile {
    name: "gouache",
    value_direction: ValueDirection::MidOut,
    opacity: Opacity::Opaque,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Minimal,
    stage_budget: 3,
    pickup: 0.8,
    bleed: 0.05,
    body: 1.0,
    impasto: 0.15,
    chroma: 0.85,
    dry_shift: -0.05,
    granulate: 0.0,
    sheen: 0.0,
    lift: 0.5,
    default_palette: "split-primary",
    broken: 0.3,
    contour: 0.0,
    families: Families::Split,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    mark: MarkCharacter::BRUSH,
    finish_policy: FinishPolicy::BackToFront,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.3,  // the gum stays soluble: a wet mark lifts the layer beneath
    chalk_tint: 0.5, // the lights mixed with white, chalky
    flat_brush: 0.5, // the flat passages in a flat brush's square patches
    liner: 0.5,  // the long thin lines in a liner's single stroke
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const WATERCOLOUR: MediumProfile = MediumProfile {
    name: "watercolour",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::Minimal,
    stage_budget: 3,
    pickup: 0.15,
    bleed: 0.55,
    body: 0.45,
    impasto: 0.0,
    // NOT a place to express a colour preference. This was raised to 1.25 by comparing against one
    // reference painting's vividness — a judgement about that picture, pinned into every picture. The
    // palette is derived from each image's own pixels and the medium must not overrule it.
    chroma: 0.92,
    dry_shift: 0.08,
    // Pigment SETTLES into the paper's tooth — the mottle is a watercolour tell, and 0.18 barely showed.
    granulate: 0.35,
    sheen: 0.0,
    lift: 0.2,
    default_palette: "limited-landscape",
    broken: 0.15,
    contour: 0.0,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    // WATERCOLOUR'S OWN MARK. It carried `BRUSH`, which is oil's mark, so the medium was oil physics with a
    // different palette: no washes, no pools, no splatter, and paper showing between strokes as white specks.
    // The wash machinery, the edge bloom and the splatter are all gated on `luminous`, so this is what turns
    // watercolour into watercolour.
    //
    // NOT line-and-wash, which is the other luminous medium and reads as an illustration: that one glazes
    // flat masses and does the drawing with an ink line (`draw_contours`). A painted watercolour has no line
    // at all — the drawing is in the washes — and it models WITHIN them, so it keeps a wider ladder of
    // brushes (`ladder_keep: None`) and a fuller charge. Twelve levels rather than eight, because a
    // watercolour's washes carry its modelling and eight steps read as flat paper cut-outs.
    mark: MarkCharacter {
        role: None,
        stroke_len: 1.0,
        stroke_width: 1.1,
        charge: 1.0,
        hatch_angle: 0.0,
        monochrome: false,
        engrave: false,
        // ECONOMY — how few marks, and how broad — is the heart of the technique, but it does NOT belong
        // here. Cutting the fine brushes in the medium left every lit face as bare reserved paper, because
        // nothing was left to paint over the reserve. It is a decision about a picture: how much paper this
        // one can afford to keep. So it lives in the plan, where it can be judged per image.
        budget_scale: 1.0,
        // FIVE glazes, not twelve. Under a technique the washes are glazed — each laid over everything darker
        // — and every level is another transparent layer on the stack. Twelve layers of mid-tones over each
        // other is mud, whatever the colours; a watercolour is built from a handful of clean glazes. The plan's
        // `levels` key tunes it per picture.
        levels: Some(5),
        reserve: None,
        ladder_keep: None,
        contrast: 1.0,
        coverage: 1.0,
        draw_contours: false,
        brush_drawing: false,
        sumi: false,
        luminous: true,
        book: false,
    },
    finish_policy: FinishPolicy::BackToFront,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

/// LINE-AND-WASH: a pen drawing over big loose watercolour washes, the paper reserved for the light — the
/// illustrator's line-and-wash. Watercolour physics (transparent, bleeding, granulating) laid as a few broad,
/// long washes from a four-level armature; the light stays paper; then the pen draws the contours in the
/// palette's darkest pigment over the dried washes (`draw_contours`). A LUMINOUS medium: see `MarkCharacter`.
pub const LINE_AND_WASH: MediumProfile = MediumProfile {
    name: "line-and-wash",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::Minimal,
    stage_budget: 3,
    pickup: 0.0,
    bleed: 0.4,
    body: 0.32,
    impasto: 0.0,
    chroma: 1.7,
    dry_shift: 0.08,
    granulate: 0.2,
    sheen: 0.0,
    lift: 0.2,
    default_palette: "limited-landscape",
    broken: 0.0,
    contour: 0.5,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    // WASHES, not a painting: only the four widest brushes of the ladder at a third of the budget (the full
    // ladder at full density reproduced the photograph and then keyed it — "pixelising and saturating"), each
    // pass dried before the next, transparent (body 0.6), never lifted (pickup 0); five value levels; the
    // paper reserved for the lights; then a fine COLOURED line in the local hue (see `ink_contours`), an accent
    // that carries the detail the washes leave out.
    mark: MarkCharacter { role: None, stroke_len: 1.0, stroke_width: 1.3, charge: 0.8, hatch_angle: 0.0, monochrome: false, engrave: false, budget_scale: 1.0, levels: Some(8), reserve: Some(0.72), ladder_keep: None, contrast: 1.0, coverage: 1.0, draw_contours: true, brush_drawing: false, sumi: false, luminous: true, book: false },
    finish_policy: FinishPolicy::BackToFront,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const EARLY_BOOK_ILLUSTRATION: MediumProfile = MediumProfile {
    name: "early-book-illustration",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::Minimal,
    stage_budget: 3,
    pickup: 0.0,
    bleed: 0.35,
    body: 0.32,
    impasto: 0.0,
    chroma: 1.15,
    dry_shift: 0.08,
    granulate: 0.05,
    sheen: 0.0,
    lift: 0.2,
    default_palette: "limited-landscape",
    broken: 0.0,
    contour: 1.0,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    // A DRAWING, TINTED: the picture's generalised masses in four inks found in the picture, three flat glazes
    // deep, the lights left as paper (see `book_tints`) — then the pen line that carries the picture, planned as
    // an engraving plans its contours (see `book_line`).
    // `budget_scale` 6: the line is the picture, and a swelling line is recorded as several cuts.
    mark: MarkCharacter { role: None, stroke_len: 1.0, stroke_width: 1.3, charge: 0.8, hatch_angle: 0.0, monochrome: false, engrave: false, budget_scale: 6.0, levels: Some(3), reserve: Some(0.72), ladder_keep: None, contrast: 1.0, coverage: 1.0, draw_contours: true, brush_drawing: false, sumi: false, luminous: true, book: true },
    finish_policy: FinishPolicy::BackToFront,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const INK_WASH: MediumProfile = MediumProfile {
    name: "ink-wash",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::None,
    stage_budget: 1,
    pickup: 0.9,
    bleed: 0.7,
    body: 0.4,
    impasto: 0.0,
    chroma: 0.9,
    dry_shift: 0.05,
    granulate: 0.13,
    sheen: 0.0,
    lift: 0.1,
    default_palette: "sumi",
    broken: 0.1,
    contour: 0.2,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    mark: MarkCharacter::BRUSH,
    finish_policy: FinishPolicy::BackToFront,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

/// JAPANESE INK (sumi-e): the ink-wash physics with a calligraphic brush and the ink's own black.
pub const JAPANESE_INK: MediumProfile = MediumProfile {
    name: "japanese-ink",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::None,
    stage_budget: 1,
    pickup: 0.9,
    bleed: 0.7,
    body: 0.4,
    impasto: 0.0,
    chroma: 0.9,
    dry_shift: 0.05,
    granulate: 0.08, // a little paper tooth
    sheen: 0.0,
    lift: 0.1,
    default_palette: "sumi",
    broken: 0.1,
    contour: 0.0, // no drawn lines
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous, // a PAINTING of two registers (see `MarkCharacter::sumi`)
    // SUMI-E: one soft wide brush loaded with rich black, long calligraphic strokes that follow the form, the
    // paper left as the light — in the ink's own grey, never the picture's colours.
    mark: MarkCharacter { role: None, stroke_len: 1.0, stroke_width: 1.0, charge: 1.0, hatch_angle: 0.0, monochrome: true, engrave: false, budget_scale: 1.0, levels: None, reserve: None, ladder_keep: None, contrast: 1.2, coverage: 0.0, draw_contours: false, brush_drawing: false, sumi: true, luminous: false, book: false },
    finish_policy: FinishPolicy::BackToFront,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.6,  // the brush in two tones: a dark edge, a pale one
    ma: 0.5,     // the empty space where the eye does not go
    kasure: 0.5, // the flying white along a running brush
    katsu: 0.0,
    nijimi: 0.4, // the ink feathering into the paper
    turn: 0.5,   // the brush lifted on a turn
    yohaku: 0.6, // the flat passages left blank
    bone: 0.5,     // the line register: the structure's contours as brush lines
    boneless: 2.0, // the block-in and two passes after it as washes
    hake: 0.7,     // the washes one horizontal sweep
    join: 1.0,     // the stamps fan around a bend: no bare sliver at a wash's corner
    bone_wind: 0.5, // a winding chain is texture, not a bone
    bone_dot: 0.5,  // half of them the tenbyō's dots
    bone_scatter: 0.5, // the dots thrown off their chain into clusters
    bone_coherence: 0.3, // a line only where the gradients around it agree
    bone_end: 0.5, // the tome where a line meets another, the harai where it fades
    sumi_focus: 0.3, // the black masses only where the eye goes
    ma_focus: 0.3, // the empty space reaching in toward the focus
    ma_flow: 0.5, // the focus's halo running along the picture's lines
    bone_ryaku: 0.5, // a far line simplified to a few arcs
    bone_fade: 0.4, // no line where the eye does not go
    bokashi: 0.5, // the fading edge washed out with water
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

/// PEN AND INK (7.12.0): steel nibs of several sizes and a black ink on white paper — drawn as the pencil is,
/// mark by mark along the form in passes coarse to fine (a broad nib for the masses' hatching, a fine one for the
/// detail), not by the ruled hatching of a density medium; the ink does not rub or lift (no stump, no eraser).
pub const PEN_INK: MediumProfile = MediumProfile {
    name: "pen-ink",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::None,
    stage_budget: 4,
    pickup: 0.0,
    bleed: 0.0,
    body: 1.0,
    impasto: 0.0,
    chroma: 0.8,
    dry_shift: 0.0,
    granulate: 0.0,
    sheen: 0.0,
    lift: 0.0,
    default_palette: "sumi",
    broken: 0.0,
    contour: 0.6,
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    mark: MarkCharacter { role: Some("pen"), stroke_len: 1.0, stroke_width: 0.4, charge: 1.0, hatch_angle: 0.0, monochrome: true, engrave: false, budget_scale: 1.0, levels: None, reserve: None, ladder_keep: None, contrast: 1.0, coverage: 0.0, draw_contours: true, brush_drawing: false, sumi: false, luminous: false, book: false },
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

/// ALBRECHT DÜRER (copperplate engraving): evenly spaced cuts that follow the form and swell with the tone,
/// crossed in the darks; a firm contour of varying weight; pure black on white (see `paint::engrave`).
pub const DURER: MediumProfile = MediumProfile {
    name: "durer",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::None,
    stage_budget: 1,
    pickup: 0.0,
    bleed: 0.0,
    body: 1.0,
    impasto: 0.0,
    chroma: 0.8,
    dry_shift: 0.0,
    granulate: 0.0,
    sheen: 0.0,
    lift: 0.0,
    default_palette: "sumi",
    broken: 0.0,
    contour: 0.55, // the plate draws its detail — windows, hands, eyes — before it shades
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Density,
    // The burin. `budget_scale` 3: a swelling line is recorded as several cuts, so a plate needs the count.
    mark: MarkCharacter { role: None, stroke_len: 1.0, stroke_width: 1.0, charge: 1.0, hatch_angle: 0.0, monochrome: true, engrave: true, budget_scale: 3.0, levels: None, reserve: None, ladder_keep: None, contrast: 1.0, coverage: 0.0, draw_contours: false, brush_drawing: false, sumi: false, luminous: false, book: false },
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const TEMPERA: MediumProfile = MediumProfile {
    name: "tempera",
    value_direction: ValueDirection::DarkToLight,
    opacity: Opacity::Mixed, // semi-opaque: a dark is built in layers, not a glaze over the gesso
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::None,
    // Tempera builds in a few deliberate hatched layers, not 40 — a budget of 40 fragmented into ~37 degenerate
    // colour passes that each restated almost nothing. Six matches the other layered media.
    stage_budget: 6,
    pickup: 0.05,
    bleed: 0.03,
    body: 0.9,
    impasto: 0.12,
    chroma: 0.95,
    dry_shift: 0.0,
    granulate: 0.07,
    sheen: 0.05,
    lift: 0.3,
    default_palette: "verdaccio",
    broken: 0.2,
    contour: 0.0,
    families: Families::Split,
    subtractive: false,
    mark_model: MarkModel::Continuous, // an opaque PAINTING built in short hatched colour strokes — not a pen drawing (the density model is pen-and-ink)
    // EGG TEMPERA (7.6.0): the ladder's own brushes — a flat block-in that covers the gesso, the restatements
    // and the detail each with their brush — the form built in SHORT strokes, each restating pass cross-hatched
    // 45° off the last: the classic tempera net of colour, matte and even. The tempera's own layers are below
    // (`hatch`, `verdaccio`, `dishes`, `lights`), its dials in the plan.
    mark: MarkCharacter { role: None, stroke_len: 0.6, stroke_width: 0.7, charge: 1.0, hatch_angle: 0.7854, monochrome: false, engrave: false, budget_scale: 1.0, levels: None, reserve: None, ladder_keep: None, contrast: 1.0, coverage: 1.0, draw_contours: false, brush_drawing: false, sumi: false, luminous: false, book: false },
    finish_policy: FinishPolicy::Uniform,
    dries: 1.0, // dry in seconds: nothing beneath is picked up or stirred
    wash_in: 0.3,
    wet_cap: 0.35,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.8,
    verdaccio: 0.6,
    dishes: 5.0,
    lights: 0.5,
    cross: 15.0,
    dither: 0.7,
    solid: 0.7,
    flesh: 0.8,
    span: 0.6,
    edge: 0.7,
    glint: 0.6,
    smooth: 0.9,
    bole: 1.0,
    ramp: 0.7,
    umbra: 1.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const PENCIL: MediumProfile = MediumProfile {
    name: "pencil",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::Minimal, // graphite lifts / smudges
    stage_budget: 4,
    pickup: 0.1,
    // Graphite SMUDGES into soft graded tones — a little wet-like fusion, far less than a wash.
    bleed: 0.15,
    body: 0.7, // greys, not opaque black
    impasto: 0.0,
    chroma: 0.7,
    dry_shift: 0.0,
    granulate: 0.18,
    sheen: 0.05,
    lift: 0.6,
    default_palette: "sumi",
    broken: 0.0,
    contour: 0.5,
    families: Families::Unified,
    subtractive: false,
    // Value is built by HATCHING / shading — mark density, like pen but softer and grey.
    mark_model: MarkModel::Continuous, // GRAPHITE: soft grey tonal strokes on the paper (like charcoal, lighter), plus the contour pass — not pen hatch
    // GRAPHITE: a dry point — thin, short, directional grey strokes that leave the paper, in its own grey.
    mark: MarkCharacter { role: Some("pencil"), stroke_len: 0.9, stroke_width: 0.5, charge: 0.45, hatch_angle: 0.0, monochrome: true, engrave: false, budget_scale: 1.0, levels: None, reserve: None, ladder_keep: None, contrast: 1.0, coverage: 0.0, draw_contours: true, brush_drawing: false, sumi: false, luminous: false, book: false },
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.3, // the velvet black only where the tone is dark on the sheet
    slant: 0.7, // the hatch at the hand's diagonal off the form
    grade: 0.4, // the pencil's lighter key
    weight: 0.6, // the contour at a pencil's weight
    clip: 0.6, // the hatch stopped at the contours
    tiers: 0.7, // lights blank, half-tones single, shadows crossed
    waver: 0.5, // the pressure wavering along the line
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

/// A SOFT BLACK PENCIL (6B / carbon): the graphite mark, but the point lays a rich dark instead of a grey —
/// deeper darks, the same paper lights.
pub const BLACK_PENCIL: MediumProfile = MediumProfile {
    name: "black-pencil",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface,
    reversibility: Reversibility::Minimal, // graphite lifts / smudges
    stage_budget: 4,
    pickup: 0.1,
    // Graphite SMUDGES into soft graded tones — a little wet-like fusion, far less than a wash.
    bleed: 0.15,
    body: 0.7, // greys, not opaque black
    impasto: 0.0,
    chroma: 0.7,
    dry_shift: 0.0,
    granulate: 0.18,
    sheen: 0.05,
    lift: 0.6,
    default_palette: "sumi",
    broken: 0.0,
    contour: 0.5,
    families: Families::Unified,
    subtractive: false,
    // Value is built by HATCHING / shading — mark density, like pen but softer and grey.
    mark_model: MarkModel::Continuous, // GRAPHITE: soft grey tonal strokes on the paper (like charcoal, lighter), plus the contour pass — not pen hatch
    // GRAPHITE: a dry point — thin, short, directional grey strokes that leave the paper, in its own grey.
    mark: MarkCharacter { role: Some("pencil"), stroke_len: 0.9, stroke_width: 0.6, charge: 0.9, hatch_angle: 0.0, monochrome: true, engrave: false, budget_scale: 1.0, levels: None, reserve: None, ladder_keep: None, contrast: 1.0, coverage: 0.0, draw_contours: true, brush_drawing: false, sumi: false, luminous: false, book: false },
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const PASTEL: MediumProfile = MediumProfile {
    name: "pastel",
    value_direction: ValueDirection::MidOut,
    opacity: Opacity::Opaque,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Minimal, // soft, blendable, liftable
    stage_budget: 4,
    pickup: 0.4, // strokes blend where they cross
    bleed: 0.1,  // soft dry blending
    body: 0.9,   // chalky, covers
    impasto: 0.1,
    chroma: 1.15, // HIGH chroma — vivid chalk
    dry_shift: 0.0,
    granulate: 0.15, // chalky tooth
    sheen: 0.0,      // matte
    lift: 0.5,
    default_palette: "split-primary",
    broken: 0.4, // pastel layers broken colour
    contour: 0.0,
    families: Families::Split,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    mark: MarkCharacter::BRUSH,
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 1.0,  // the stick on the tooth
    side: 0.5,   // the broad passes with the stick's side
    blend: 0.5,  // the calm passages rubbed with a finger
    skin: 0.5,   // the faces blended with a finger
    carnation: 1.0, // the flesh in warm sticks
    small: 250.0, // a face under 250 px drawn at its own scale
    anchor: 0.15, // the darkest sixth of a small face kept crisp
    trim: 0.5, // the face cut by the figure, the ground kept half a brush clear
    small_min: 100.0, // a crowd's face under 100 px left to the plain ladder
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const CHARCOAL: MediumProfile = MediumProfile {
    name: "charcoal",
    value_direction: ValueDirection::LightToDark,
    opacity: Opacity::Transparent,
    white_source: WhiteSource::Surface, // paper is the light; lift highlights out
    reversibility: Reversibility::Minimal,
    stage_budget: 3,
    pickup: 0.2,
    bleed: 0.3,  // SMUDGY — soft graded blacks
    body: 0.8,   // rich, dramatic black
    impasto: 0.0,
    chroma: 0.4, // near-monochrome, warm black
    dry_shift: 0.0,
    granulate: 0.2, // charcoal grain
    sheen: 0.0,
    lift: 0.6, // erase / lift highlights
    default_palette: "sumi",
    broken: 0.0,
    contour: 0.3, // draws as well as shades
    families: Families::Unified,
    subtractive: false,
    mark_model: MarkModel::Continuous, // tonal smudge, not hatch
    mark: MarkCharacter::BRUSH,
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 1.0,  // the light is the paper: charcoal only darkens
    erase: 0.6,  // the kneaded eraser lifts the lights
    touch: 0.6,  // a light tone a light touch on the grain
    velvet: 0.08, // the darkest twelfth one velvet black
    paper_top: 1.0, // the picture's light end is the paper
    sweep: 0.9,  // the flat passages in calm sweeps
    stump: 0.5,  // the dark masses rubbed with a stump
    dust: 0.85,  // the dust settles into the hollows
    stump_edge: 0.6, // the stump holds at a contrast
    stump_feather: 1.0, // the stump fades toward a face, crosses a lost edge
    sweep_wave: 4.0, // the sweeps on an arm's arc
    paper_knee: 0.5, // a broad light mass the paper
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

/// ACRYLIC: a fast-drying, opaque plastic paint, painted in LAYERS like the indirect oil — a thin wash
/// block-in, the masses, the restatements, the accents — but each layer DRY before the next goes on: nothing
/// is picked up, nothing blends, a mark sits on the one beneath as a flat patch with a crisp edge; the last
/// layer is dragged dry over the canvas's tooth off the faces; no wet bloom; the colour dries a little
/// darker and flatter than it went on, under a satin sheen and a thin relief.
pub const ACRYLIC: MediumProfile = MediumProfile {
    name: "acrylic",
    value_direction: ValueDirection::MidOut,
    opacity: Opacity::Opaque,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::None, // FAST-DRY — no reworking, layers stack cleanly
    stage_budget: 6,
    pickup: 0.05, // fast-drying: a mark sits on top, it does not lift the one beneath
    bleed: 0.0,   // no wet bloom: a dried acrylic edge is where the brush stopped
    body: 1.0,    // opaque plastic
    impasto: 0.3, // a heavy-body relief: the marks stand, with their micro-shadows, under the light
    chroma: 1.05,  // vivid, but it dries a touch flatter than it goes on
    dry_shift: -0.05, // dries darker
    granulate: 0.0,
    sheen: 0.2, // satin
    lift: 0.0,   // permanent once dry
    default_palette: "split-primary",
    broken: 0.15, // a little drift of hue from mark to mark: flat, even colour, never a fill
    contour: 0.0,
    families: Families::Split,
    subtractive: false,
    mark_model: MarkModel::Continuous,
    // A synthetic flat: smooth lanes, a square edge, a steady hand (see `BrushProfile::named`).
    mark: MarkCharacter { role: Some("acrylic"), ..MarkCharacter::BRUSH },
    finish_policy: FinishPolicy::Uniform,
    dries: 1.0,
    wash_in: 0.5,
    wet_cap: 0.45,
    scumble: 1.0,
    skip: 0.3,
    turbulence: 8.0,
    accent: 1.0,
    loose: 0.5,
    ridges: 0.3,
    impasto_map: 1.0,
    deplete: 1.8,
    glaze: 0.6,
    sfumato: 0.7,
    glaze_pass: 0.3,
    vary: 0.35,
    comma: 0.5,
    shade: 0.6,
    imprimatura: 0.6,
    body_vary: 0.3,
    lost: 0.6,
    crest: 0.4,
    spatter: 0.5,
    aerial: 0.6,
    sink: 0.5,
    tail: 0.6,
    film: 0.7,
    split: 0.5,
    sag: 0.4,
    gloss: 0.5,
    tint: 0.4,
    knife: 0.3,
    hiding: 0.5,
    twist: 0.5,
    skim: 0.4,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

/// Every declared medium.
/// PALETTE KNIFE (7.14.0): the picture formed in thick, flat slices of pure paint laid with the knife — no
/// transparency, no water, no hair lines of a brush, no blending: each slice covers what is under it whole, and the
/// relief of the slices, with their sharp edges, is relit at output. The profile is `oil-direct`'s physics with the
/// knife as the mark (`BrushProfile::named("knife")`), no pickup, a thicker relief and a stronger sheen; the knife's
/// own keys (`blade`, `knife`, `skim`, `hiding`, `impasto`, `sheen`, `gloss`) are set in the plan, as the other media
/// of 7.5.0 on are — see `Documentation/PAINT_EXAMPLES.md`.
pub const PALETTE_KNIFE: MediumProfile = MediumProfile {
    name: "palette-knife",
    value_direction: ValueDirection::MidOut,
    opacity: Opacity::Opaque,
    white_source: WhiteSource::Pigment,
    reversibility: Reversibility::Hours,
    stage_budget: 1,
    bleed: 0.08,
    body: 1.0,
    impasto: 0.5,
    chroma: 1.08,
    dry_shift: 0.0,
    granulate: 0.0,
    sheen: 0.3,
    lift: 0.8,
    default_palette: "zorn",
    broken: 0.2,
    contour: 0.0,
    pickup: 0.0,
    families: Families::Split,
    subtractive: true,
    mark_model: MarkModel::Continuous,
    mark: MarkCharacter { role: Some("knife"), ..MarkCharacter::BRUSH },
    finish_policy: FinishPolicy::Uniform,
    dries: 0.5,
    wash_in: 0.0,
    wet_cap: 1.0,
    scumble: 0.0,
    skip: 0.0,
    turbulence: 0.0,
    accent: 0.0,
    loose: 0.0,
    ridges: 0.0,
    impasto_map: 0.0,
    deplete: 1.0,
    glaze: 0.0,
    sfumato: 0.0,
    glaze_pass: 0.0,
    vary: 0.0,
    comma: 0.0,
    shade: 0.0,
    imprimatura: 0.0,
    body_vary: 0.0,
    lost: 0.0,
    crest: 0.0,
    spatter: 0.0,
    aerial: 0.0,
    sink: 0.0,
    tail: 0.0,
    film: 0.0,
    split: 0.0,
    sag: 0.0,
    gloss: 0.0,
    tint: 0.0,
    knife: 0.0,
    hiding: 0.0,
    twist: 0.0,
    skim: 0.0,
    blade: 0.0,
    blade_planes: 0.0,
    blade_levels: 0.0,
    blade_size: 0.0,
    blade_bevel: 0.0,
    blade_light: 0.0,
    blade_phong: 0.0,
    blade_flow: 0.0,
    blade_rest: 0.0,
    blade_relief: 0.0,
    blade_mix: 0.0,
    hatch: 0.0,
    verdaccio: 0.0,
    dishes: 0.0,
    lights: 0.0,
    cross: 0.0,
    dither: 0.0,
    solid: 0.0,
    flesh: 0.0,
    span: 0.0,
    edge: 0.0,
    glint: 0.0,
    smooth: 0.0,
    bole: 0.0,
    ramp: 0.0,
    umbra: 0.0,
    chalk: 0.0,
    side: 0.0,
    blend: 0.0,
    skin: 0.0,
    carnation: 0.0,
    small: 0.0,
    anchor: 0.0,
    trim: 0.0,
    small_min: 0.0,
    paper: 0.0,
    erase: 0.0,
    touch: 0.0,
    velvet: 0.0,
    paper_top: 0.0,
    sweep: 0.0,
    stump: 0.0,
    dust: 0.0,
    stump_edge: 0.0,
    stump_feather: 0.0,
    sweep_wave: 0.0,
    paper_knee: 0.0,
    rewet: 0.0,
    chalk_tint: 0.0,
    flat_brush: 0.0,
    liner: 0.0,
    tsuke: 0.0,
    ma: 0.0,
    kasure: 0.0,
    katsu: 0.0,
    nijimi: 0.0,
    turn: 0.0,
    yohaku: 0.0,
    bone: 0.0,
    boneless: 0.0,
    hake: 0.0,
    join: 0.0,
    bone_wind: 0.0,
    bone_dot: 0.0,
    bone_scatter: 0.0,
    bone_coherence: 0.0,
    bone_end: 0.0,
    sumi_focus: 0.0,
    ma_focus: 0.0,
    ma_flow: 0.0,
    bone_ryaku: 0.0,
    bone_fade: 0.0,
    bokashi: 0.0,
    velvet_dark: 0.0,
    slant: 0.0,
    grade: 0.0,
    weight: 0.0,
    clip: 0.0,
    tiers: 0.0,
    waver: 0.0,
    nib: 0.0,
    nibs: 0.0,
    nib_tone: 0.0,
    nib_white: 0.0,
    nib_anchor: 0.0,
    nib_hint: 0.0,
    nib_face: 0.0,
    nib_gap: 0.0,
    nib_relief: 0.0,
    nib_far: 0.0,
    nib_occlude: 0.0,
    nib_eyes: 0.0,
    nib_fur: 0.0,
    nib_lift: 0.0,
    nib_contact: 0.0,
    nib_figure: 0.0,
    nib_object: 0.0,
    nib_tuft: 0.0,
    nib_close: 0.0,
    nib_mass: 0.0,
    nib_rim: 0.0,
    nib_pool: 0.0,
    nib_air: 0.0,
    nib_people: 0.0,
    nib_value: 0.0,
    nib_fine_face: 0.0,
    nib_scatter: 0.0,
    nib_falloff: 0.0,
    nib_depth: 0.0,
    nib_lush: 0.0,
    nib_glint: 0.0,
    nib_wash: 0.0,
    nib_wash_dark: 0.0,
    nib_wash_detail: 0.0,
    nib_wash_line: 0.0,
    nib_wash_swell: 0.0,
    nib_wash_air: 0.0,
    nib_wash_soft: 0.0,
    nib_wash_gamma: 0.0,
    nib_wash_flat: 0.0,
    nib_wash_scurve: 0.0,
    nib_wash_vary: 0.0,
    nib_wash_mass: 0.0,
    nib_wash_hold: 0.0,
    nib_wash_debt: 0.0,
    nib_wash_dry: 0.0,
    nib_wash_cap: 0.0,
    nib_wash_tone: 0.0,
    nib_wash_draw: 0.0,
    nib_wash_planes: 0.0,
    nib_wash_edge: 0.0,
    nib_wash_depth: 0.0,
    nib_wash_scale: 0.0,
    nib_wash_levels: 0.0,
    nib_wash_faces: 0.0,
    nib_wash_rag: 0.0,
    nib_wash_mask: 0.0,
    nib_wash_half: 0.0,
    nib_wash_grain: 0.0,
    nib_wash_grade: 0.0,
    nib_wash_profile: 0.0,
    nib_wash_streak: 0.0,
    nib_wash_sweep: 0.0,
    nib_wash_water: 0.0,
    nib_wash_agent: 0.0,
};

pub const ALL: &[MediumProfile] = &[OIL_DIRECT, OIL_INDIRECT, GOUACHE, WATERCOLOUR, LINE_AND_WASH, EARLY_BOOK_ILLUSTRATION, INK_WASH, JAPANESE_INK, PEN_INK, DURER, TEMPERA, PENCIL, BLACK_PENCIL, PASTEL, CHARCOAL, ACRYLIC, PALETTE_KNIFE];

/// The media P1 can execute (opaque continuous).
pub const P1_EXECUTABLE: &[&str] = &["oil-direct", "gouache"];
/// The media executable through P2 — adds watercolour (reservation) and pen-ink (density).
pub const P2_EXECUTABLE: &[&str] = &["oil-direct", "gouache", "watercolour", "pen-ink"];
/// Every executable medium (P3 adds indirect oil, ink wash, tempera — they reuse the opaque-continuous,
/// transparent-reserve, and density paths respectively; tempera's density marks are monochrome for now).
pub const EXECUTABLE: &[&str] = &["oil-direct", "oil-indirect", "gouache", "watercolour", "line-and-wash", "early-book-illustration", "ink-wash", "japanese-ink", "pen-ink", "durer", "tempera", "pencil", "black-pencil", "pastel", "charcoal", "acrylic", "palette-knife"];

impl MediumProfile {
    /// Look up a medium by name (case-insensitive).
    pub fn by_name(name: &str) -> Option<MediumProfile> {
        ALL.iter().find(|m| m.name.eq_ignore_ascii_case(name.trim())).copied()
    }
    /// Whether P1 can render this medium.
    pub fn is_p1_executable(&self) -> bool {
        P1_EXECUTABLE.contains(&self.name)
    }
    /// Whether the engine can render this medium.
    pub fn is_executable(&self) -> bool {
        EXECUTABLE.contains(&self.name)
    }
}

/// A stage of a painting — a paint layer separated from its neighbours by a state change (§3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Plan the un-painted whites (surface-white media only).
    ReservePlan,
    /// Lay the toned ground / imprimatura.
    Ground,
    /// Establish the value structure in the medium's build direction.
    Value(ValueDirection),
    /// The thin, transparent shadow family.
    ShadowMass,
    /// The opaque light family.
    LightMass,
    /// A colour pass (dead-colour / glaze), numbered when a medium affords several.
    Colour(usize),
    /// The transition band at the terminator.
    Halftone,
    /// The final small dark/bright notes.
    Accents,
    /// The opaque highlights — the irreversible operation, always last for a pigment-white medium.
    Highlights,
}

impl Stage {
    /// A short stable slug for the score / reports.
    pub fn slug(&self) -> String {
        match self {
            Stage::ReservePlan => "reserve-plan".into(),
            Stage::Ground => "ground".into(),
            Stage::Value(_) => "value".into(),
            Stage::ShadowMass => "shadow-mass".into(),
            Stage::LightMass => "light-mass".into(),
            Stage::Colour(i) => format!("colour-{i}"),
            Stage::Halftone => "halftone".into(),
            Stage::Accents => "accents".into(),
            Stage::Highlights => "highlights".into(),
        }
    }
}

/// Generate the stage schedule for a medium (§6.4). The schedule is a pure function of the invariants.
pub fn generate_schedule(m: &MediumProfile) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();
    if m.white_source == WhiteSource::Surface {
        stages.push(Stage::ReservePlan);
    }
    if m.opacity != Opacity::Transparent {
        stages.push(Stage::Ground);
    }
    stages.push(Stage::Value(m.value_direction));
    if m.families == Families::Split {
        stages.push(Stage::ShadowMass);
        stages.push(Stage::LightMass);
    }
    // Colour stages fill whatever the stage budget affords beyond the structural stages already placed.
    let colour = (m.stage_budget as i64 - stages.len() as i64).max(0) as usize;
    for i in 0..colour {
        stages.push(Stage::Colour(i + 1));
    }
    if m.reversibility >= Reversibility::Hours {
        stages.push(Stage::Halftone);
    }
    stages.push(Stage::Accents);
    // The irreversible operation goes last.
    if m.white_source == WhiteSource::Pigment {
        stages.push(Stage::Highlights);
    }
    stages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_and_p1_executability() {
        assert_eq!(MediumProfile::by_name("Oil-Direct").unwrap().name, "oil-direct");
        assert!(OIL_DIRECT.is_p1_executable() && GOUACHE.is_p1_executable());
        assert!(!WATERCOLOUR.is_p1_executable(), "watercolour lands in P2");
        assert!(MediumProfile::by_name("nope").is_none());
    }

    #[test]
    fn surface_white_media_reserve_first() {
        let s = generate_schedule(&WATERCOLOUR);
        assert_eq!(s.first(), Some(&Stage::ReservePlan), "white from paper → plan the reserve before stroke one");
        assert!(!s.contains(&Stage::Ground), "transparent → no opaque ground");
        assert_ne!(s.last(), Some(&Stage::Highlights), "no opaque highlights when white is the surface");
    }

    #[test]
    fn pigment_white_media_end_on_highlights() {
        for m in [OIL_DIRECT, GOUACHE, OIL_INDIRECT] {
            let s = generate_schedule(&m);
            assert_eq!(s.last(), Some(&Stage::Highlights), "{}: irreversible highlights last", m.name);
            assert_eq!(s.first(), Some(&Stage::Ground), "{}: opaque → ground first", m.name);
        }
    }

    #[test]
    fn split_family_media_have_both_masses() {
        let s = generate_schedule(&OIL_DIRECT);
        assert!(s.contains(&Stage::ShadowMass) && s.contains(&Stage::LightMass), "split → two families: {s:?}");
        // Shadow before light (thin darks, then opaque lights).
        let si = s.iter().position(|x| *x == Stage::ShadowMass).unwrap();
        let li = s.iter().position(|x| *x == Stage::LightMass).unwrap();
        assert!(si < li, "shadow mass precedes light mass");
    }

    #[test]
    fn only_the_acrylic_lays_its_layers_dry() {
        // The layer fields are neutral on every other medium: those paint exactly as they did before them.
        for m in ALL.iter().filter(|m| m.name != "acrylic" && m.name != "tempera") {
            assert_eq!([m.dries, m.wash_in, m.wet_cap, m.scumble, m.skip, m.turbulence, m.accent, m.loose, m.ridges, m.impasto_map, m.deplete, m.glaze, m.sfumato, m.glaze_pass, m.vary, m.comma, m.shade, m.imprimatura, m.body_vary, m.lost, m.crest, m.spatter, m.aerial, m.sink, m.tail, m.film, m.split, m.sag, m.gloss, m.tint, m.knife, m.hiding, m.twist, m.skim, m.hatch, m.verdaccio, m.dishes, m.lights, m.cross, m.dither, m.solid, m.flesh, m.span, m.edge, m.glint, m.smooth, m.bole, m.ramp, m.umbra], [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "{} keeps the open-paint defaults", m.name);
        }
        assert_eq!(ACRYLIC.dries, 1.0, "acrylic is dry before the next layer");
        assert!(ACRYLIC.wash_in > 0.0 && ACRYLIC.wet_cap < 1.0 && ACRYLIC.scumble > 0.0 && ACRYLIC.bleed == 0.0);
        assert_eq!(ACRYLIC.mark.role, Some("acrylic"));
        // The tempera (7.6.0) lays its own layers: dry, a thin wash block-in, capped wet, hatched, the flesh
        // underpainted in verdaccio, the tones in dishes, the lights hatched in — and no acrylic layer.
        assert_eq!(TEMPERA.dries, 1.0);
        assert!(TEMPERA.wash_in > 0.0 && TEMPERA.wet_cap < 1.0 && TEMPERA.hatch > 0.0 && TEMPERA.verdaccio > 0.0 && TEMPERA.dishes >= 2.0 && TEMPERA.lights > 0.0);
        assert!(TEMPERA.cross > 0.0 && TEMPERA.cross <= 45.0 && TEMPERA.dither > 0.0 && TEMPERA.solid > 0.0 && TEMPERA.flesh > 0.0 && TEMPERA.span > 0.0 && TEMPERA.edge > 0.0 && TEMPERA.glint > 0.0 && TEMPERA.smooth > 0.0 && TEMPERA.bole > 0.0 && TEMPERA.bole <= 1.0 && TEMPERA.ramp > 0.0 && TEMPERA.ramp <= 1.0 && TEMPERA.umbra > 0.0 && TEMPERA.umbra <= 1.0, "the hatching off the form, the dishes dithered, the dark hatch solid, the verdaccio held to the flesh");
        assert!(TEMPERA.scumble == 0.0 && TEMPERA.knife == 0.0 && TEMPERA.impasto_map == 0.0 && TEMPERA.spatter == 0.0 && TEMPERA.sag == 0.0 && TEMPERA.gloss == 0.0);
        assert_eq!(TEMPERA.mark.role, None, "the tempera paints with the ladder's own brushes");
        for m in ALL.iter().filter(|m| m.name != "tempera") {
            assert_eq!([m.hatch, m.verdaccio, m.dishes, m.lights, m.cross, m.dither, m.solid, m.flesh, m.span, m.edge, m.glint, m.smooth, m.bole, m.ramp, m.umbra], [0.0; 15], "{} has no tempera layer", m.name);
        }
        // The pastel (7.7.0) lays its own: the chalk on the tooth, the side of the stick, the finger — and no
        // other medium has them.
        assert!(PASTEL.chalk > 0.0 && PASTEL.chalk <= 1.0 && PASTEL.side > 0.0 && PASTEL.side <= 1.0 && PASTEL.blend > 0.0 && PASTEL.blend <= 1.0 && PASTEL.skin > 0.0 && PASTEL.skin <= 1.0 && PASTEL.carnation > 0.0 && PASTEL.carnation <= 1.0 && PASTEL.small > 0.0 && PASTEL.anchor > 0.0 && PASTEL.trim > 0.0);
        for m in ALL.iter().filter(|m| m.name != "pastel") {
            assert_eq!([m.chalk, m.side, m.blend, m.skin, m.carnation, m.small, m.anchor, m.trim, m.small_min], [0.0; 9], "{} has no pastel layer", m.name);
            if m.name != "charcoal" {
                assert_eq!([m.paper, m.erase, m.touch, m.velvet, m.paper_top, m.sweep, m.stump, m.dust, m.stump_edge, m.stump_feather, m.sweep_wave, m.paper_knee], [0.0; 12], "{} has no charcoal layer", m.name);
            }
            if m.name != "gouache" {
                assert_eq!([m.rewet, m.chalk_tint, m.flat_brush, m.liner], [0.0; 4], "{} has no gouache layer", m.name);
            }
            if m.name != "pencil" {
                assert_eq!([m.velvet_dark, m.slant, m.grade, m.weight, m.clip, m.tiers, m.waver], [0.0; 7], "{} has no pencil layer", m.name);
            }
            if m.name != "japanese-ink" {
                assert_eq!([m.tsuke, m.ma, m.kasure, m.nijimi, m.turn, m.yohaku, m.bone, m.boneless, m.hake, m.join, m.bone_wind, m.bone_dot, m.bone_scatter, m.bone_coherence, m.bone_end, m.sumi_focus, m.ma_focus, m.ma_flow, m.bone_ryaku, m.bone_fade, m.bokashi], [0.0; 21], "{} has no sumi layer", m.name);
            }
        }
    }

    #[test]
    fn indirect_oil_affords_colour_stages() {
        // stage_budget 8 leaves room for colour passes beyond the structural stages.
        let s = generate_schedule(&OIL_INDIRECT);
        assert!(s.iter().any(|x| matches!(x, Stage::Colour(_))), "big stage budget → colour passes: {s:?}");
        assert!(s.contains(&Stage::Halftone), "reversible (days) → a halftone pass");
    }
}
