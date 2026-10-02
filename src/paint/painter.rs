//! The coarse-to-fine painter (RFC PAINT-1 P0) — and the filter-gate it exists to test.
//!
//! Given a reference IMAGE, paint it stroke by stroke under PAINT-1's hard constraints: a finite, inviolable
//! stroke **budget**; a limited **palette** under Kubelka-Munk mixing; a **minimum brush size**; and strokes
//! that follow an **orientation field** and are laid with the bristle brush (deposit + pickup). Crucially the
//! reference each brush layer paints from is BLURRED proportionally to the brush size, so a large brush has no
//! detail to trace — the constraints, not a matching objective, decide when a pass is done.
//!
//! This is the P0 make-or-break: if the output reads as a painterly filter with the budget and palette
//! constraints active, the core thesis is wrong. [`traceability`] measures how close the output is to the
//! full-resolution reference — a filter scores high, a painting lower (structure kept, surface invented).

use image::{imageops, RgbImage};

use crate::paint::canvas::Canvas;
use crate::paint::color::{self, Srgb};
use crate::paint::mixer;
use crate::paint::palette::Palette;
use crate::paint::score::{ScoreHeader, StrokeRecord, StrokeScore};
use crate::paint::stroke::{BrushConfig, Stroke};

/// One pass of the painter: a stage painted at a brush radius with its own stroke budget. When a plan (P1.3)
/// supplies passes, they drive the painter; otherwise the coarse→fine `brush_sizes` do.
#[derive(Clone, Debug)]
pub struct PassSpec {
    pub radius: f32,
    pub budget: usize,
    pub stage: String,
    /// This pass may lay marks only where the FACE is. A wash medium keeps its sheet to a few broad brushes
    /// — economy is its technique — but a face painted with nothing finer than a wash is not a face. So the
    /// brushes cut from the ladder are kept for the focal plane alone.
    pub face_only: bool,
}

/// A BRUSH from the vocabulary (RFC brush vocabulary): the FORM of a mark — size, length, bristle streak,
/// cross-section roundness, and hand waver. It composes with the MEDIUM's physics (opacity, pickup), so the same
/// vocabulary serves oil, gouache, watercolour and ink. A painting uses several: a broad flat for the sky, a
/// clean round for a face, a rigger for ropes.
#[derive(Clone, Copy, Debug)]
pub struct BrushProfile {
    pub radius_scale: f32,
    pub len: f32,
    pub streak: f32,
    pub round: f32,
    pub waver: f32,
}

impl BrushProfile {
    /// Look up a named brush. Unknown names fall back to the filbert (the general-purpose brush).
    pub fn named(name: &str) -> BrushProfile {
        match name.trim().to_ascii_lowercase().as_str() {
            // wide, long, raked — masses, skies, broad grounds
            "flat" => BrushProfile { radius_scale: 1.10, len: 1.30, streak: 0.70, round: 0.35, waver: 0.10 },
            // general purpose
            "filbert" => BrushProfile { radius_scale: 0.95, len: 1.00, streak: 0.50, round: 0.70, waver: 0.12 },
            // small, short, clean, soft — faces, features, the net's knots (DETAIL)
            "round" => BrushProfile { radius_scale: 0.72, len: 0.55, streak: 0.18, round: 0.94, waver: 0.14 },
            // streaky texture — foliage, hair, broken water / waves
            "fan" => BrushProfile { radius_scale: 1.00, len: 0.80, streak: 1.00, round: 0.60, waver: 0.20 },
            // thin, very long, smooth — rigging, ropes, masts, spars, fine lines
            "rigger" => BrushProfile { radius_scale: 0.45, len: 2.20, streak: 0.10, round: 0.92, waver: 0.08 },
            // broad, opaque, square-edged slabs — bold blocking
            "knife" => BrushProfile { radius_scale: 1.25, len: 1.10, streak: 0.00, round: 0.15, waver: 0.05 },
            // A graphite point: narrow, fairly short, a little streaky (the tooth), a hand's waver.
            "pencil" => BrushProfile { radius_scale: 0.35, len: 0.90, streak: 0.30, round: 0.80, waver: 0.08 },
            // A tempera brush: small, short, clean, even — the strokes read as a woven net.
            "tempera" => BrushProfile { radius_scale: 0.60, len: 0.60, streak: 0.15, round: 0.65, waver: 0.06 },
            // broad, long, soft, thin — watercolour / sky washes
            "wash" => BrushProfile { radius_scale: 1.30, len: 1.60, streak: 0.15, round: 0.85, waver: 0.10 },
            _ => BrushProfile { radius_scale: 0.95, len: 1.00, streak: 0.50, round: 0.70, waver: 0.12 },
        }
    }
}

/// The painting's fidelity register. `Legible` (default) resolves features: fine passes sharpen the reference,
/// lay crisp short strokes, and a final DEFINITION pass draws the strongest edges so objects read. `Impressionist`
/// deliberately keeps it loose: no sharpening, no edge definition, longer strokes — soft masses of colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PaintStyle {
    #[default]
    Legible,
    Impressionist,
    /// HIGH-FIDELITY: paint CLOSELY from a sharp armature — no per-pass blur, tight edge-following flow, clean
    /// low-waver strokes, and most passes treated as detail. A disciplined painterly RENDERING that tracks the
    /// reference (high traceability) rather than inventing a loose surface. For recognizable, detailed results.
    Fidelity,
}

/// How an EDGE is marked, the way a painter chooses (RFC §7 edge craft). Real painters do not only draw a line:
/// they mark an edge with a **line**, a **colour change** (a temperature/hue shift, no line), a **knife** (a
/// scraped/lifted crisp edge), or leave it **lost** (dissolved).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EdgeMode {
    /// A soft drawn contour line in the local dark (default).
    #[default]
    Line,
    /// A CHROMATIC edge — mark it by cooling/shifting the colour rather than a line (a temperature edge).
    Colour,
    /// A KNIFE edge — scrape/lift to a crisp lighter edge (a negative, palette-knife edge).
    Knife,
    /// A LOST edge — no marking (the edge dissolves).
    Lost,
}

/// What the painter reports while it works (the CLI's progress bar).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PaintProgress {
    /// Strokes laid so far (the bar's position).
    Placed(usize),
    /// A pass is MIXING its new colours before it lays a stroke — the bar's position stands still meanwhile;
    /// this is where the worker threads run.
    Mixing { pass: usize, passes: usize, radius: f32, colours: usize, threads: usize },
    /// A pass is laying strokes. The first passes are few, wide marks (each covers thousands of pixels); the
    /// last are many small ones — so the bar crawls at first and flies at the end.
    Painting { pass: usize, passes: usize, radius: f32 },
}

/// Parameters for a paint-from-image run.
#[derive(Clone, Debug)]
pub struct PaintParams {
    pub palette: Palette,
    /// Total stroke budget across all layers — inviolable.
    pub budget: usize,
    /// Explicit stage passes (from a compiled plan). `None` → derive passes from `brush_sizes`.
    pub passes: Option<Vec<PassSpec>>,
    /// Brush radii, coarse → fine. A radius below `min_brush` is skipped.
    pub brush_sizes: Vec<f32>,
    /// The smallest brush allowed, so the finest pass still cannot chase pixel detail.
    pub min_brush: f32,
    /// ARMATURE resolution (RFC §1.1/§5.1): the reference is coarsened to this longest-side before painting, so
    /// structure survives but DETAIL does not — the brush must invent the surface rather than trace it. `None`
    /// keeps the reference full-resolution (the P0 from-image behaviour).
    pub armature_side: Option<u32>,
    /// FOCAL armature resolution (RFC §5.2): when set with a `face_mask`, the face region is built from a FINER
    /// armature (this size) than the rest of the canvas (`armature_side`), so the beard/background become washes
    /// while the face stays crisp — variable-resolution structure in one pass. `None` = uniform armature.
    pub armature_face_side: Option<u32>,
    /// SUBJECT-BODY armature resolution (RFC §5.2, multi-region): a MID resolution for the subject body (between
    /// the background `armature_side` and the face `armature_face_side`), applied where `subject_mask` is set.
    /// Gives a three-tier armature — background coarsest, body mid, face fine.
    pub armature_body_side: Option<u32>,
    /// The subject (foreground) 0..1 mask for `armature_body_side` — from a matte model (U2Net). Row-major, canvas-sized.
    pub subject_mask: Option<Vec<f32>>,
    /// VALUE MASSES in the structure-preserving armature (RFC §5): how many value levels the armature quantises to
    /// (the block-in a painter sees). Fewer = bolder, flatter masses; more = subtler. ~5–7 reads as a painting.
    pub armature_levels: u32,
    /// SEMANTIC region tiers (RFC §5.2): extra `(mask, armature_resolution_px)` regions — e.g. a coarse hair/beard
    /// tier (kept as a wash) or a clothing tier — from OWL-ViT/SAM. Blended coarse→fine with the body/face tiers.
    pub region_tiers: Vec<(Vec<f32>, u32)>,
    /// SILHOUETTE (0..1, RFC §5): mark the detected SUBJECT boundary (the matte silhouette) so a light shirt /
    /// shoulders read against a light background by their edge instead of vanishing. Needs `subject_mask`.
    /// Recorded strokes (replay-exact). How the edge is marked is `silhouette_mode`.
    pub silhouette: f32,
    /// How the silhouette edge is MARKED: line / colour change / knife / lost (RFC §7 edge craft).
    pub silhouette_mode: EdgeMode,
    /// COMMIT SHADOWS (0..1, RFC §3.3 shadow family): paint the DARK value masses DECISIVELY — in the shadow
    /// regions the reserve is lifted (darks always paint), the restate floor drops (darks build to full depth),
    /// and the pigment charge is boosted (committed, not a thin wash). This is what turns a pale "washed
    /// photograph" into a painting with a solid value backbone. Derived from the reference's own value structure.
    pub commit_shadows: f32,
    /// Charge multiplier for a stroke's load (how much paint the brush holds vs its footprint).
    pub charge: f32,
    /// Medium name recorded in the score header (physics still comes from `brush` in this slice).
    pub medium: String,
    /// RESERVE (surface-white media): a luma threshold in `[0,1]` — cells brighter than this get NO stroke, so
    /// the paper/ground shows through (watercolour whites, §6.3). `None` → paint everywhere.
    pub reserve: Option<f32>,
    /// DENSITY mark model (§8.6): build value by black hatch marks whose count scales with darkness, instead
    /// of loaded continuous strokes (pen-ink). Uses the palette's darkest pigment.
    pub density: bool,
    /// A TONED ground (imprimatura) to prime the canvas with — for opaque media, so light passages show.
    /// `None` = a white ground / paper.
    pub ground: Option<Srgb>,
    /// NEGATIVE PAINTING (§8.7): a protect mask (`w*h`, true = leave unpainted) — strokes are not seeded inside
    /// it, so a shape is DEFINED by painting the space around it. Generalises the `reserve`. `None` = paint all.
    pub protect: Option<Vec<bool>>,
    /// FOCAL HARD-EDGE (§7.4): a region mask (`w*h`, true = subject) — a stroke's growth TERMINATES when it
    /// crosses the subject/background boundary, so the subject stays crisp against the ground instead of
    /// smearing across the seam. `None` = strokes cross freely (soft everywhere).
    pub region_mask: Option<Vec<bool>>,
    /// HAIR / BEARD / FUR (0..1, canvas-sized). Hair is high-frequency DIRECTIONAL texture, and the armature is
    /// structure with the texture taken out — so a from-scratch painting has, by construction, nothing to paint
    /// hair FROM, and a mane came out a lumpy mass. Where this mask is set the painter changes TOOL rather than
    /// reference: a finer floor, more bristle lanes with a rakier streak (the lanes ARE the strands), almost no
    /// pickup so strands stay distinct instead of smearing into mud, longer and narrower marks tapering to a
    /// point, and a minority of marks allowed to break the silhouette (see the seam below).
    pub hair_mask: Option<Vec<f32>>,
    /// What a stroke follows where the picture gives it nothing to follow. See [`FlowInfill`]; `Flat` is the
    /// long-standing behaviour and the default.
    pub infill: FlowInfill,
    /// THE RIGGER (0..1, 0 = off): put back the few shapes too THIN for the brush ladder to lay at all — a
    /// stem, a spoon handle, the line of a shelf. See [`rigger_pass`]; rationed hard, because a wiry picture is
    /// worse than a missing stem.
    pub rigger: f32,
    /// HOTSPOTS (0..1): how far a flat, blown specular highlight is re-modelled into a dome with a falloff.
    /// 0 leaves it as the plateau it is. The default softens the plateau and its rim while keeping the
    /// highlight — it is where the light is, and the picture wants it; what it lacks is its falloff. 1 models
    /// it fully. See [`hotspot_pass`].
    pub hotspot: f32,
    /// See [`WetTechnique`]. `None` keeps every medium exactly as it was.
    pub technique: WetTechnique,
    /// From this index on, the brush ladder is for the FACE only (see `PassSpec::face_only`). `None` = the
    /// whole ladder paints the whole sheet.
    pub face_ladder_from: Option<usize>,
    /// LEAKS (0..1, 0 = none): runs of pigment that drip down from the bottom of a wet wash and end in a
    /// drop — a gravity mark, and the one that says "this was liquid" more than any other. See `leak_pass`.
    pub leak: f32,
    pub seed: u64,
    pub brush: BrushConfig,
    /// COMPOSITION LAYER (per-element painting): only seed strokes where `paint_mask` is true — the element's
    /// footprint on the canvas. Painted ONTO whatever is already there (via [`paint_onto`]), so a nearer element
    /// occludes farther ones. `None` = paint the whole canvas (a single-element painting).
    pub paint_mask: Option<Vec<bool>>,
    /// The brush for this composition layer (RFC brush vocabulary) — overrides the pass-role default character
    /// (streak/roundness/length/waver/size) for every stroke of this element. `None` = the intelligent per-role
    /// default (flat masses → filbert → round detail). The PAINTING layers (coarse→fine passes) still apply.
    pub layer_brush: Option<BrushProfile>,
    /// DEPTH map (RFC §5.5 recession): per-pixel `[0,1]`, 1 = near, 0 = far (Depth-Anything / ControlNet-Depth
    /// convention). When present, the reference is conditioned with AERIAL PERSPECTIVE — distant passages are
    /// veiled toward the atmosphere and lose contrast, so the background RECEDES and the foreground ADVANCES
    /// (foreground/background separation and painted "layers"). `None` = a flat single plane.
    pub depth: Option<Vec<f32>>,
    /// Aerial-perspective strength (0..1): how strongly distance veils toward the atmosphere. Ignored w/o depth.
    pub haze: f32,
    /// STROKE LENGTH dial (author control): global multiplier on how far strokes run (1.0 = default). Longer =
    /// cleaner sweeping coverage; shorter = choppier dabs.
    pub stroke_len: f32,
    /// STROKE WIDTH dial (author control): global multiplier on brush width (1.0 = default). Wider strokes cover
    /// more per mark (fewer, cleaner); narrower = finer, more marks.
    pub stroke_width: f32,
    /// Wet-into-wet BLEED (0..1) applied after painting — the wet media's fusion/bloom. Default per medium.
    pub bleed: f32,
    /// PIGMENT DIFFUSION (-1..+1, default 0 = none): which way the pigment travels in the wet wash. +1 = into the
    /// darks (they charge up, lights stay clean), -1 = out into the lights (feathered halos). See `Canvas::bleed_with`.
    pub diffuse: f32,
    /// LUMINOUS medium (line-and-wash): the contour drawing is planned from the SOURCE (the armature has
    /// simplified the machine and the figures into washes) and laid as a light pen line over the washes.
    pub luminous: bool,
    /// BOOK (early-book-illustration): the washes as flat cumulative glazes from the keyed picture at pixel
    /// scale, no modelling strokes. See `MarkCharacter::book`.
    pub book: bool,
    /// PARALLELISM: worker threads for the up-front pigment MIXING of each pass (0 = every core, the default).
    /// One painter lays the strokes in the classic order whatever the count — the thread count never changes
    /// the picture. (Mixing was 99% of a stroke's cost; the brush itself is under 1%.)
    pub threads: usize,
    /// NEW PAINTING (`--new`, default false): paint FROM SCRATCH (RFC PAINT-1 §1.1-1.2). The reference is a
    /// genuinely LOW-RESOLUTION armature — structure with no detail to trace — nothing reads the source again
    /// (no texture restate; washes and lines read the armature), and every plane has a MINIMUM BRUSH that grows
    /// with depth (RFC §9): the background never receives a fine brush, the figure a medium one, the finest
    /// rungs fire on the focal plane alone. The surface is invented by the brushwork. Off = the default path,
    /// which tracks its source closely (a painterly rendering of the image).
    pub from_scratch: bool,
    /// FILL (0..1, default 0 = auto): a density floor. When the gates stop the painting below this share of
    /// the budget, the finest pass is repeated with its restate floor halved each round (up to six) until the
    /// share is spent or a round adds almost nothing. More worked and denser on demand; never more detail
    /// than the reference holds. Replay-exact (the extra marks are ordinary records of the finest stage).
    pub fill: f32,
    /// INTER-PASS DRYING (0..1): how much the canvas dries between passes. 0 = never dries (fully wet-into-wet —
    /// every later pass picks up the masses beneath and smears them into mud); 1 = bone dry between passes (each
    /// pass a crisp overlay). Default ~0.5. This is the single biggest lever against the muddy/washed look.
    pub dry: f32,
    /// BLOCK-IN COVERAGE (0..1): how gap-free the first pass lays its base. 0 = the raked, dry flat brush (the
    /// default — texture-forward brushwork); 1 = a smooth, opaque cover. Opt-in: its effect is marginal on most
    /// images, and a covering footprint reaches slightly further, so it must not be on by default — it bled a
    /// hair of pigment into RESERVED paper (the reserve invariant) when it was.
    pub coverage: f32,
    /// DETAIL COHERENCE bar (default 0.14): the minimum flow-coherence a detail stroke needs to land, so detail
    /// only resolves where there is real structure. Applied subject-aware — the background is held ~2.6x stricter
    /// so stray detail marks don't speckle a structureless sky/wall, while the subject keeps its modelling.
    /// Higher = cleaner (fewer stray marks, softer); lower = busier.
    pub detail_coherence: f32,
    /// FINEST-LAYER STROKE LENGTH multiplier (1.0 = the pass profile's own length). Stroke length TAPERS with the
    /// layers: the widest (block-in) keeps its long covering strokes, each thinner layer on top is shorter, down
    /// to this on the finest — a painter's depth order. A global shortening starves the block-in of coverage
    /// (white specks of ground between dabs); long strokes on the fine layers smear the features.
    pub detail_len: f32,
    /// DETAIL RESTATE floor (RGB distance, 0..1): a fine-layer mark lands only where the canvas still DISAGREES
    /// with the target by at least this much. Error-driven, not structure-driven: on a mass the block-in already
    /// got right the error is small → no dab (a crisp dab there is a SPECK); at a feature the wide brushes could
    /// not resolve the error is large → the dab lands. The old floor (0.03) let the fine layers restate nearly
    /// every cell, blanketing smooth masses with mark boundaries.
    pub detail_restate: f32,
    /// DETAIL SHARPEN (unsharp-mask radius as a fraction of the brush radius; 0 = none): the reference the fine
    /// layers paint from. Unsharp masking puts DARK overshoot along the hairline, eye sockets and beard edge; a few
    /// strokes sampling it are invisible, but the dense fine layers turn it into black scrawl on a face and dark
    /// blobs on a wall (measured: face dark-feature energy 0.107 vs the target's 0.087). Off by default — the
    /// armature already carries the structure; crisp short strokes give the detail its edge.
    pub detail_sharpen: f32,
    /// DETAIL TEXTURE (default 1.0): how much of the SOURCE's fine texture the fine layers may restate inside the
    /// flat masses the armature simplified — wheat, grass, bark, weave. The armature is structure, not detail
    /// (RFC §1.1), and a painter blocks in a wheat field as one mass; the texture strokes on top come from looking
    /// at the subject. The residual (source minus its blur at the pass's scale) is added to the armature only where
    /// the armature is locally FLAT, never at its edges (an edge residual is overshoot — the face-scrawl tell), and
    /// only OUTSIDE the matted subject (texture is for the background masses; on a face it is scrawl).
    /// 0 = fine layers read the plain armature.
    pub detail_texture: f32,
    /// GRADATION (0..1, default 0): keep slow RAMPS continuous in the armature. The value masses turn a cloud,
    /// a soft-lit wall or still water into a few flat tones with contour edges; where the picture is a ramp,
    /// not an edge (the flow rule's scale-free test), this brings the bilateral back toward a plain smooth of
    /// the source and holds the value snap off, so the mass keeps its turning form. Edges snap as before.
    pub gradation: f32,
    /// CROSS-HATCH (medium mark character): each restating pass rotates its stroke direction by this many radians
    /// more than the previous one; 0 = every pass follows the form.
    pub hatch_angle: f32,
    /// Density media: hatch as an engraving (see `ink::HatchStyle::ENGRAVING`).
    pub engrave: bool,
    /// After the tonal passes, draw the ink planner's contours on top in the darkest pigment (pencil sketch =
    /// line and tone). Replaces the legacy contour finish.
    pub draw_contours: bool,
    /// Density media: draw with a loaded brush (sumi-e) — see `ink::HatchStyle::SUMI`.
    pub brush_drawing: bool,
    /// SUMI-E (two registers on one sheet): the LIGHT value family as a few soft graded washes with dissolving
    /// edges, most of the sheet left paper; the DARK family as bold dry-brush black SHAPES with ragged edges;
    /// no drawn contours.
    pub sumi: bool,
    /// BODY / opacity (0.1..1) of the paint film — 1 = opaque, low = transparent (the ground glows through).
    pub opacity: f32,
    /// IMPASTO relight strength (0..1) applied at OUTPUT — the textured oil/knife look. Recorded for replay.
    pub impasto: f32,
    /// MATERIAL physics (§8.5) — how the paint itself behaves at output (chroma range, drying value-shift,
    /// granulation) and during painting (lift = wipe removability). All recorded for exact replay.
    pub chroma: f32,
    pub dry_shift: f32,
    pub granulate: f32,
    pub sheen: f32,
    pub lift: f32,
    /// BROKEN COLOUR (0..1): per-stroke hue/chroma variation so adjacent marks optically mix (vibrancy). The
    /// varied colour is baked into the recorded stroke, so replay is exact — no header field needed.
    pub broken: f32,
    /// CONTOUR (0..1): a final line-drawing pass that draws the strongest edges (pen/pencil). Recorded strokes.
    pub contour: f32,
    /// Fidelity register — `Legible` resolves features, `Impressionist` stays loose (§style knob).
    pub style: PaintStyle,
    /// EDGE-HARDNESS strength (0..1): how many boundaries are treated as HARD, where strokes terminate so the
    /// masses meet crisply (edge-control craft — not outlining). Higher = more hard edges. 0 = all edges soft.
    /// Ignored for `Impressionist` and for density media.
    pub define: f32,
    /// SALIENCY-GATED DENSITY (0..1, 0 = off — opt-in): reserve dense strokes for the focal, high-structure
    /// passages and lay flat, empty regions THIN. At `1` a restating/detail pass paints at full density only
    /// where saliency is high and drops (up to all of) its strokes where it is low; at `0.5` the background
    /// keeps ~half. The block-in is never gated (the canvas is always covered), so the background stays a calm
    /// smooth mass instead of being over-worked into a uniform hatch. Off = byte-identical to the ungated engine.
    pub saliency: f32,
    /// SELECTIVE DETAIL (0..1, 0 = off — opt-in): paint the masses in a LOOSE register but fire the crisp detail
    /// tier ONLY inside the focal region (the compact, central, high-contrast area — a portrait's eyes/glasses),
    /// so the rest stays a loose wash instead of chasing every high-frequency texture (a beard) into speckle.
    /// The value opens the focal region: small = only the very focus gets detail, `1` = the whole canvas (all
    /// fine passes detail everywhere). Pairs with a loose base (`--style impressionist`). Off = unchanged engine.
    pub focus_detail: f32,
    /// PRESERVE FACE (0..1, 0 = off — opt-in): strength of the DETECTED-FACE focal region. Like `focus_detail`
    /// but the focal region is the real face box(es) from the detector (via `face_mask`), not the centre prior —
    /// so the crisp detail tier lands on the actual face however it is placed. Higher = more of the (feathered)
    /// face gets preserved (crisp, reference-tracked); lower = only the face core. Needs `face_mask` set.
    pub preserve_face: f32,
    /// A feathered 0..1 face-region mask (row-major, canvas-sized) from the face detector, for `preserve_face`.
    /// Built by the CLI (which owns the model/device); `None` = no face preservation. Takes priority over the
    /// centre-prior focal field when present.
    pub face_mask: Option<Vec<f32>>,
    /// The detector's face BOXES at canvas scale (`[x1, y1, x2, y2]`, px, ungrown), when the CLI has them.
    /// A watercolour face's planes take their SHAPE from these (see `face_planes_scope`); the mask alone is a
    /// grown, feathered rectangle. Empty = shape the planes by the mask.
    pub face_boxes: Vec<[f32; 4]>,
    /// SPLATTER (0..1, 0 = off): flick fine pigment droplets across the painting — the watercolour/ink spatter
    /// mark. Higher = denser spray. Recorded as strokes (replay-exact). See `splatter_pass`.
    pub splatter: f32,
    /// EDGE POOLING (0..1, 0 = off): darken pigment where a wash meets a hard boundary — the pigment ring a
    /// watercolour wash dries into (the "cauliflower"/edge-bloom look). An output-stage effect (see `Finish`).
    pub edge_pool: f32,
    /// PAPER EDGE (0..1, 0 = off): fade the painting to bare paper at the borders with an irregular DECKLED edge
    /// — the torn-paper vignette a watercolour sits in. An output-stage effect (see `Finish`).
    pub paper_edge: f32,
    /// FINISH GRADE (painting-safe, recorded for replay). CONTRAST (0.5..2, 1 = neutral): S-curve around mid-grey.
    pub contrast: f32,
    /// WARMTH (−1..1, 0 = neutral): white-balance shift, + warm / − cool.
    pub warmth: f32,
    /// CLARITY (0..1, 0 = off): gentle LOCAL contrast (large-radius unsharp) — NOT edge sharpening.
    pub clarity: f32,
}

impl PaintParams {
    /// A sensible default over a palette at a stroke budget.
    pub fn new(palette: Palette, budget: usize) -> Self {
        Self { palette, budget, passes: None, brush_sizes: vec![28.0, 14.0, 7.0], min_brush: 4.0, armature_side: None, armature_face_side: None, armature_body_side: None, subject_mask: None, armature_levels: 8, region_tiers: Vec::new(), silhouette: 0.0, silhouette_mode: EdgeMode::Line, commit_shadows: 0.0, charge: 6.0, medium: "oil-direct".into(), reserve: None, density: false, ground: None, protect: None, region_mask: None, hair_mask: None, infill: FlowInfill::Flat, rigger: 0.0, hotspot: 0.0, technique: WetTechnique::None, face_ladder_from: None, leak: 0.0, seed: 42, brush: BrushConfig::default(), paint_mask: None, layer_brush: None, depth: None, haze: 0.0, stroke_len: 1.0, stroke_width: 1.0, bleed: 0.0, diffuse: 0.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, lift: 1.0, broken: 0.0, contour: 0.0, style: PaintStyle::Legible, define: 0.6, saliency: 0.0, focus_detail: 0.0, preserve_face: 0.0, face_mask: None, face_boxes: Vec::new(), splatter: 0.0, edge_pool: 0.0, paper_edge: 0.0, contrast: 1.0, warmth: 0.0, clarity: 0.0, dry: 0.5, coverage: 0.0, detail_coherence: 0.14, detail_len: 1.0, detail_restate: 0.08, detail_sharpen: 0.0, detail_texture: 1.0, gradation: 0.0, hatch_angle: 0.0, engrave: false, draw_contours: false, brush_drawing: false, sumi: false, luminous: false, book: false, threads: 0, from_scratch: false, fill: 0.0 }
    }
}

/// The result of a paint run.
pub struct PaintResult {
    pub canvas: Canvas,
    /// How many strokes were actually laid (≤ budget).
    pub strokes: usize,
    /// The replayable stroke score — the canonical artifact.
    pub score: StrokeScore,
    /// Stages the critic rejected (rolled back) — empty without a critic.
    pub rejected: Vec<String>,
    /// What each pass cost: its stage, the brush it used, how many marks it laid and how long it took.
    /// Gathered always, not only under the profiler, so a run can report where its time actually went.
    pub stats: Vec<PassStat>,
    /// Wall time inside the painter.
    pub seconds: f64,
}

/// One pass's cost (see [`PaintResult::stats`]).
#[derive(Clone, Debug)]
pub struct PassStat {
    pub stage: String,
    /// The brush radius in pixels, which is what makes one pass slower per mark than another.
    pub radius: f32,
    pub strokes: usize,
    pub seconds: f64,
}

fn luma_map(img: &RgbImage) -> Vec<f32> {
    img.pixels().map(|p| color::linear_luma(color::srgb_to_linear(p.0))).collect()
}

/// Sobel gradient of a luma map → `(gx, gy, magnitude)` per pixel.
fn sobel(luma: &[f32], w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let (wi, hi) = (w as i32, h as i32);
    let at = |x: i32, y: i32| -> f32 {
        let x = x.clamp(0, wi - 1) as usize;
        let y = y.clamp(0, hi - 1) as usize;
        luma[y * w as usize + x]
    };
    let mut gx = vec![0f32; luma.len()];
    let mut gy = vec![0f32; luma.len()];
    for y in 0..hi {
        for x in 0..wi {
            let i = (y as usize) * w as usize + x as usize;
            gx[i] = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1);
            gy[i] = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1);
        }
    }
    (gx, gy)
}

/// Separable box blur over an f32 field (radius `r` px, `2r+1` window), clamped at the borders. Cheap and
/// good enough for smoothing the structure tensor.
fn box_blur(src: &[f32], w: u32, h: u32, r: i32) -> Vec<f32> {
    if r <= 0 {
        return src.to_vec();
    }
    let (wi, hi) = (w as i32, h as i32);
    let norm = 1.0 / (2 * r + 1) as f32;
    // Horizontal.
    let mut tmp = vec![0f32; src.len()];
    for y in 0..hi {
        let row = y as usize * w as usize;
        for x in 0..wi {
            let mut s = 0.0;
            for d in -r..=r {
                s += src[row + (x + d).clamp(0, wi - 1) as usize];
            }
            tmp[row + x as usize] = s * norm;
        }
    }
    // Vertical.
    let mut out = vec![0f32; src.len()];
    for y in 0..hi {
        for x in 0..wi {
            let mut s = 0.0;
            for d in -r..=r {
                s += tmp[(y + d).clamp(0, hi - 1) as usize * w as usize + x as usize];
            }
            out[y as usize * w as usize + x as usize] = s * norm;
        }
    }
    out
}

/// The HAIR FLOW FIELD: which way hair actually GROWS, as a unit direction per pixel plus how confidently.
///
/// Every other direction in the painter comes from the armature's isophotes, and the armature is structure with
/// the texture taken out — so under `--new` a strand has no direction of its own and borrows the shape of the
/// value mass it sits in. That is why a mane gained fibres from the strand tool but still combed the wrong way,
/// and why curls resisted entirely: a ringlet's direction is not in the armature at all.
///
/// This reads the SOURCE, and the reduction is what keeps that honest (RFC §1.1: structure is low-resolution
/// and model-derived). The structure tensor is built at full resolution, then its components are averaged down
/// to `side` across the short edge and brought back — so what survives is an ANGLE PER ARMATURE CELL, the same
/// order of information the armature itself carries, not the picture's texture. No pigment, value or edge
/// crosses over; only which way the cell runs.
///
/// Orientation is mod π, so the components are averaged as the DOUBLE-ANGLE pair (Jxx−Jyy, 2Jxy) — averaging
/// angles directly makes strands at +80° and −80° cancel into nothing instead of agreeing.
///
/// Returns `(dx, dy, coherence)`: the direction structure RUNS (along a strand, perpendicular to the gradient)
/// and the tensor's anisotropy in 0..1, which is near zero on skin or cloth and high on hair, fur and grass.
pub struct HairFlow {
    /// Unit direction the structure RUNS (along a strand).
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
    /// Tensor anisotropy 0..1 — how much the neighbourhood agrees on one orientation.
    pub coherence: Vec<f32>,
    /// STRANDNESS: coherence gated by fine-scale contrast. Coherence ALONE does not tell hair from cloth — a
    /// sleeve's folds are strongly oriented too, and a flood that trusted coherence ran straight out of a
    /// beard and claimed both men's shirts and trousers. What separates them is SCALE: hair is oriented AND
    /// busy at strand scale, while a fold is oriented and SMOOTH between its edges. This is the measure to
    /// grow a hair region by.
    pub strandness: Vec<f32>,
}

pub fn hair_flow_field(src: &RgbImage, side: u32) -> HairFlow {
    let (w, h) = src.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;
    let luma = luma_map(src);
    let (gx, gy) = sobel(&luma, w, h);
    // The tensor, at the scale of a strand: small enough that a lock keeps its own direction, large enough
    // that single-pixel noise does not set it.
    let sigma = ((w.min(h) as f32) / 512.0).round().clamp(2.0, 6.0) as i32;
    let mut c = vec![0f32; n]; // Jxx − Jyy
    let mut sxy = vec![0f32; n]; // 2·Jxy
    let mut energy = vec![0f32; n]; // Jxx + Jyy
    for i in 0..n {
        c[i] = gx[i] * gx[i] - gy[i] * gy[i];
        sxy[i] = 2.0 * gx[i] * gy[i];
        energy[i] = gx[i] * gx[i] + gy[i] * gy[i];
    }
    let c = box_blur(&c, w, h, sigma);
    let sxy = box_blur(&sxy, w, h, sigma);
    let energy = box_blur(&energy, w, h, sigma);
    // THE REDUCTION: average the double-angle components down to the armature's grid and back. Whatever
    // finer-than-armature detail the tensor saw is gone after this; an angle per cell is all that survives.
    let short = w.min(h).max(1);
    let side = side.clamp(8, short);
    let scale = side as f32 / short as f32;
    let (sw, sh) = (((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1));
    let shrink = |v: &[f32]| -> Vec<f32> {
        let img = image::GrayImage::from_fn(w, h, |x, y| {
            // Carry the sign through an unsigned byte image: 0.5 is zero.
            image::Luma([((v[(y * w + x) as usize] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8])
        });
        let small = imageops::resize(&img, sw, sh, imageops::FilterType::Triangle);
        let back = imageops::resize(&small, w, h, imageops::FilterType::Triangle);
        back.pixels().map(|p| (p.0[0] as f32 / 255.0 - 0.5) * 2.0).collect()
    };
    // Normalise by the local energy first, so the reduction averages ORIENTATIONS rather than being dominated
    // by whichever cell happened to have the strongest contrast.
    let (mut cn, mut sn) = (vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        let e = energy[i] + 1e-6;
        cn[i] = (c[i] / e).clamp(-1.0, 1.0);
        sn[i] = (sxy[i] / e).clamp(-1.0, 1.0);
    }
    cn = shrink(&cn);
    sn = shrink(&sn);
    // Fine-scale contrast: how BUSY the picture is at strand scale. Hair is busy; a fold is smooth.
    let fine = local_range(&luma, wu, hu, sigma.max(2) as usize);
    let (mut dx, mut dy, mut coh, mut strand) = (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        // The magnitude of the double-angle vector IS the coherence: 1 when every gradient in the cell agrees
        // on an orientation (a lock of hair), 0 when they point every way (skin, flat cloth).
        let mag = (cn[i] * cn[i] + sn[i] * sn[i]).sqrt();
        coh[i] = mag.clamp(0.0, 1.0);
        // Dominant GRADIENT angle, then a quarter turn to lie ALONG the strand.
        let theta = 0.5 * sn[i].atan2(cn[i]);
        dx[i] = -theta.sin();
        dy[i] = theta.cos();
        // 0.06 of the value range across a strand-width window is about where a head of hair sits and a lit
        // sleeve does not; it is a contrast fact of the picture, not a number tuned to one of them.
        strand[i] = coh[i] * (fine[i] / 0.06).clamp(0.0, 1.0);
    }
    HairFlow { dx, dy, coherence: coh, strandness: strand }
}

/// How many `true` cells each (2r+1)² window holds, clipped at the edges — one 2-D prefix sum, O(1) a pixel.
fn box_count(src: &[bool], w: usize, h: usize, r: usize) -> (Vec<u32>, Vec<u32>) {
    let mut ps = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += src[y * w + x] as u32;
            ps[(y + 1) * (w + 1) + (x + 1)] = ps[y * (w + 1) + (x + 1)] + row;
        }
    }
    let mut cnt = vec![0u32; w * h];
    let mut area = vec![0u32; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let s = ps[y1 * (w + 1) + x1] + ps[y0 * (w + 1) + x0] - ps[y0 * (w + 1) + x1] - ps[y1 * (w + 1) + x0];
            cnt[y * w + x] = s;
            area[y * w + x] = ((y1 - y0) * (x1 - x0)) as u32;
        }
    }
    (cnt, area)
}

fn erode_bool(src: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let (cnt, area) = box_count(src, w, h, r);
    cnt.iter().zip(&area).map(|(c, a)| c == a).collect()
}

fn dilate_bool(src: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let (cnt, _) = box_count(src, w, h, r);
    cnt.iter().map(|c| *c > 0).collect()
}

/// Keep only the connected regions of at least `min_area` cells.
fn keep_large_regions(src: &[bool], w: usize, h: usize, min_area: usize) -> Vec<bool> {
    keep_regions_between(src, w, h, min_area, usize::MAX)
}

/// Keep only the connected regions of `min_area..=max_area` cells.
fn keep_regions_between(src: &[bool], w: usize, h: usize, min_area: usize, max_area: usize) -> Vec<bool> {
    let n = w * h;
    let mut out = vec![false; n];
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut region: Vec<usize> = Vec::new();
    for start in 0..n {
        if !src[start] || seen[start] {
            continue;
        }
        region.clear();
        stack.push(start);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            region.push(i);
            let (x, y) = (i % w, i / w);
            if x > 0 && src[i - 1] && !seen[i - 1] { seen[i - 1] = true; stack.push(i - 1); }
            if x + 1 < w && src[i + 1] && !seen[i + 1] { seen[i + 1] = true; stack.push(i + 1); }
            if y > 0 && src[i - w] && !seen[i - w] { seen[i - w] = true; stack.push(i - w); }
            if y + 1 < h && src[i + w] && !seen[i + w] { seen[i + w] = true; stack.push(i + w); }
        }
        if region.len() >= min_area && region.len() <= max_area {
            for &i in &region {
                out[i] = true;
            }
        }
    }
    out
}

/// POOLS. As a wash dries its pigment migrates to the edge and settles there, and the dark rim it leaves
/// — the cauliflower edge — is the mark that says "watercolour" more than any other. The output-time edge
/// pooling reads the paint-amount gradient, and by the time the modelling passes have been over the washes
/// that gradient is gone, which is why `edge_pool` could be set to anything and show nothing.
///
/// So the rim is PAINTED, as a finish stage after the modelling, along the rings every wash recorded: the
/// wash's own pigment, more concentrated, in a thin line on its boundary. A recorded stroke, so it replays
/// exactly; laid last, so no restate pass can read it as an error and paint it back out.
fn pool_pass(canvas: &mut Canvas, score: &mut StrokeScore, p: &PaintParams, placed: &mut usize, k: &mut u64) {
    let strength = p.edge_pool.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let (w, h) = (canvas.w, canvas.h);
    let short = w.min(h) as f32;
    let np = p.palette.pigments.len();
    // Every wash's rings and mixture, gathered first: the records are about to grow.
    let washes: Vec<(Vec<[f32; 2]>, Vec<f32>)> = score
        .strokes
        .iter()
        .filter(|r| r.wash)
        .map(|r| {
            let mut load = vec![0f32; np];
            for (name, v) in &r.mix {
                if let Some(idx) = p.palette.pigments.iter().position(|pg| pg.name == name) {
                    load[idx] = *v;
                }
            }
            (r.spline.clone(), load)
        })
        .collect();
    if washes.is_empty() {
        return;
    }
    // A rim, not an outline. At full strength this was a hard dark line round every mass and the picture read
    // as line-and-wash; the dried edge of a wash is a narrow band only a little deeper than the wash itself.
    let width = (short * 0.0016).max(1.2) * (0.6 + 0.5 * strength);
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.streak = 0.1;
    brush.round = 0.95;
    for (rings, load) in washes {
        let load: Vec<f32> = load.iter().map(|v| v * (0.5 + 0.7 * strength)).collect();
        for ring in rings.split(|pt| pt[0].is_nan()) {
            if ring.len() < 3 {
                continue;
            }
            *k += 1;
            let mut path = ring.to_vec();
            path.push(ring[0]); // close it
            let s = Stroke { path, width0: width, width1: width, load: load.clone(), pressure: 0.9, wetness: 0.35 };
            s.rasterize(canvas, &brush);
            let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
            *placed += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe: false,
                wash: false,
                stage: "pool".into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.0,
                mix,
                wet: s.wetness,
                press: 0.9,
                streak: brush.streak,
                round: brush.round,
                pickup: Some(0.0),
                bristles: None,
            });
        }
    }
}

/// LEAKS. Where a wash is wet enough, pigment runs out of its lower edge under gravity: a thin trail down the
/// paper that tapers and ends in a drop where it finally dried. It is the mark that says "this was liquid"
/// more plainly than any other, and a watercolourist either courts it or guards against it — which is why
/// it is a control and not a default.
///
/// Only the big wet masses leak, from points along their bottom edge; the trail carries the wash's own
/// pigment, a little concentrated (a run collects what it passes over). Wet-on-wet runs further; the dry
/// brush had no water to run and leaks nothing. Recorded strokes, laid after the rims and under the spatter.
fn leak_pass(canvas: &mut Canvas, score: &mut StrokeScore, p: &PaintParams, placed: &mut usize, k: &mut u64) {
    let strength = p.leak.clamp(0.0, 1.0);
    if strength <= 0.0 || p.technique == WetTechnique::DryOnDry {
        return;
    }
    let (w, h) = (canvas.w, canvas.h);
    let unit = (w.min(h) as f32 / 1024.0).max(0.5);
    let np = p.palette.pigments.len();
    let run_mul = if p.technique == WetTechnique::WetOnWet { 1.5 } else { 1.0 };
    // The washes, largest first: the big masses are the wet ones.
    let mut washes: Vec<(usize, Vec<[f32; 2]>, Vec<f32>)> = score
        .strokes
        .iter()
        .filter(|r| r.wash)
        .map(|r| {
            let mut load = vec![0f32; np];
            for (name, v) in &r.mix {
                if let Some(idx) = p.palette.pigments.iter().position(|pg| pg.name == name) {
                    load[idx] = *v;
                }
            }
            (r.spline.iter().filter(|pt| !pt[0].is_nan()).count(), r.spline.clone(), load)
        })
        .collect();
    washes.sort_by(|a, b| b.0.cmp(&a.0));
    let max_leaks = (8.0 + 48.0 * strength) as usize;
    let mut laid = 0usize;
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.streak = 0.05;
    brush.round = 1.0;
    for (_, rings, load) in washes.iter().take(max_leaks * 2) {
        if laid >= max_leaks {
            break;
        }
        // Only the OUTER ring (the first) can leak; holes do not have a bottom edge on the paper.
        let Some(ring) = rings.split(|pt| pt[0].is_nan()).next() else { continue };
        if ring.len() < 4 {
            continue;
        }
        let (ymin, ymax) = ring.iter().fold((f32::MAX, f32::MIN), |(a, b), pt| (a.min(pt[1]), b.max(pt[1])));
        if ymax - ymin < 12.0 * unit {
            continue;
        }
        *k += 1;
        // Does this wash leak at all? At full strength every big wet mass does; the wetter the technique,
        // the further down the list it reaches.
        if jitter(p.seed ^ 0x1EA7, *k) + 0.5 > (strength * run_mul).min(1.0) {
            continue;
        }
        // Where along its bottom edge: a point in the lowest sixth of the mass.
        let band = ymax - (ymax - ymin) * 0.16;
        let bottom: Vec<[f32; 2]> = ring.iter().copied().filter(|pt| pt[1] >= band).collect();
        if bottom.is_empty() {
            continue;
        }
        let pick = ((jitter(p.seed ^ 0x1EA8, k.wrapping_add(1)) + 0.5) * bottom.len() as f32) as usize;
        let [x0, y0] = bottom[pick.min(bottom.len() - 1)];
        let x0 = x0.clamp(1.0, w as f32 - 2.0);
        let y0 = y0.clamp(1.0, h as f32 - 2.0);
        // A run must READ against whatever it crosses, because a picture can be anything. Over a light or
        // mid ground it is concentrated pigment, plainly darker than the wash it left. Over a ground already
        // darker than its own pigment, a drip of water re-wets the dried wash and LIFTS it, leaving a pale
        // run — the same rule the spatter follows. Decided from the canvas below the start, not assumed.
        let below_l = color::linear_luma(color::srgb_to_linear(canvas.color_at(x0 as u32, (y0 + 10.0 * unit).min(h as f32 - 1.0) as u32)));
        let run_l = {
            let c = canvas.color_at(x0 as u32, (y0 - 1.0).max(0.0) as u32);
            color::linear_luma(color::srgb_to_linear(c)) * 0.55
        };
        // A dark run on a dark ground is invisible whatever its pigment — on a night scene, a run that reads
        // is a pale one. So the ground below decides: dark in absolute terms, or darker than the run would
        // be, and the drip lifts; otherwise it carries pigment.
        let lifts = below_l < 0.18 || below_l < run_l;
        // Never a run across a face.
        if p.face_mask.as_deref().and_then(|m| m.get(y0 as usize * w as usize + x0 as usize)).copied().unwrap_or(0.0) > 0.3 {
            continue;
        }
        let len = unit * (25.0 + 110.0 * strength * (jitter(p.seed ^ 0x1EA9, k.wrapping_add(2)) + 0.5)) * run_mul;
        let width = unit * (1.6 + 1.8 * (jitter(p.seed ^ 0x1EAA, k.wrapping_add(3)) + 0.5));
        let y1 = (y0 + len).min(h as f32 - 2.0);
        if y1 - y0 < 6.0 {
            continue;
        }
        let wobble = 1.5 * unit * jitter(p.seed ^ 0x1EAB, k.wrapping_add(4));
        let wet = if p.technique == WetTechnique::WetOnWet { 0.9 } else { 0.7 };
        // The run: a thin tapering trail straight down, and the drop where it dried.
        let path = vec![[x0, y0], [x0 + wobble, (y0 + y1) * 0.5], [x0, y1]];
        let drop = vec![[x0, y1], [x0 + 0.5, y1 + 1.5]];
        if lifts {
            // A pale run: water lifting the dried wash. Applied at wet*lift, the formula replay uses.
            let lw = 1.6_f32;
            for (pth, w0, w1) in [(path, width, width * 0.5), (drop, width * 1.7, width * 1.7)] {
                let s = Stroke { path: pth, width0: w0, width1: w1, load: vec![0.0; np], pressure: 1.0, wetness: lw };
                s.wipe(canvas, &brush, lw * p.lift);
                *placed += 1;
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: true, wash: false, stage: "leak".into(), spline: s.path, w0, w1, taper: 0.0, mix: Vec::new(), wet: lw, press: 1.0, streak: brush.streak, round: brush.round, pickup: None, bristles: None });
            }
        } else {
            // A dark run: the pigment a drip collects on its way down, well above the wash's own charge.
            let load: Vec<f32> = load.iter().map(|v| v * 2.0).collect();
            for (pth, w0, w1, pr) in [(path, width, width * 0.5, 0.9), (drop, width * 1.7, width * 1.7, 1.0)] {
                let s = Stroke { path: pth, width0: w0, width1: w1, load: load.clone(), pressure: pr, wetness: wet };
                s.rasterize(canvas, &brush);
                let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                *placed += 1;
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "leak".into(), spline: s.path, w0, w1, taper: 0.0, mix, wet, press: pr, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: None });
            }
        }
        laid += 1;
    }
}

/// A COHERENT flow field via the structure tensor (Kang/Hertzmann coherence-enhancing painterly rendering).
/// Raw per-pixel Sobel swirls on a smoothed armature — the direction jitters between neighbours, so strokes
/// wander and the painting reads as noise. Instead we build the tensor J = [[gx², gxgy],[gxgy, gy²]], blur it
/// so nearby gradients reinforce into one dominant orientation, then return a representative gradient vector
/// per pixel: direction = the tensor's dominant eigenvector (θ = ½·atan2(2Jxy, Jxx−Jyy)), magnitude = the
/// coherence (how anisotropic the neighbourhood is). `stroke_dir` takes the perpendicular of this, giving a
/// smooth, form-following stroke direction that only wavers where the image genuinely has no structure.
/// HOW WET THE PAPER IS when pigment lands — the decision that makes a watercolour look the way it does,
/// expressed as what the PIGMENT does, not as a set of finish amounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WetTechnique {
    /// The medium's own behaviour, unchanged.
    None,
    /// Pigment lands in standing water: a stroke goes down flooded and spreads, washes bloom into each
    /// other, nothing dries between layers, and no edge ever sets hard enough for a rim to form.
    WetOnWet,
    /// A loaded brush on dry paper: a wash lands where it is put and blooms only within itself, each layer
    /// SETS before the next, and as it dries its pigment migrates to the boundary and leaves the rim.
    WetOnDry,
    /// A barely-loaded brush dragged over dry paper: little pigment, laid raked, skipping on the tooth;
    /// nothing bleeds, nothing pools, and what pigment there is granulates into the hollows.
    DryOnDry,
}

/// What a stroke should follow where the picture gives it NOTHING to follow — a flat passage, which in a
/// dark interior is most of the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlowInfill {
    /// Lay it FLAT: long level marks, the way a painter blends a sky. Right for atmosphere and wrong for a
    /// dark mass, where every stroke then runs horizontally and the passage tiles into a rectangular quilt.
    Flat,
    /// FOLLOW the structure around it: the direction is carried inward from the nearest passage that has one,
    /// so a dark mass is stroked along the shelf edge or silhouette that bounds it instead of along the frame.
    Follow,
    /// A fixed STROKE angle in degrees, measured from horizontal. The painter's own decision about a passage.
    Angle(f32),
}

/// Carry an orientation inward from wherever the picture has one. The weighted double-angle field is blurred
/// at growing radii and each pixel takes the FIRST radius that reaches real structure, so a direction travels
/// as far as it must and no further. Orientation is mod π, hence the double angle: averaged as raw angles,
/// structure at +80° and −80° would cancel instead of agreeing.
fn infill_orientation(cw: &[f32], sw: &[f32], wt: &[f32], w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    // ON A COARSE GRID. Carrying a direction across a flat passage is by its nature LOW-FREQUENCY work — the
    // answer varies over hundreds of pixels — and doing it at full resolution means box blurs at radii up to
    // the sheet's own width. Measured: it turned a 43 s painting into 243 s, the flow field alone going from
    // 5 s to 206 s, which is the whole of why a new painting became slow.
    //
    // An eighth of the resolution keeps every radius and so every reach, at a sixty-fourth of the pixels.
    const D: u32 = 8;
    if w > D * 4 && h > D * 4 {
        let (sw2, sh2) = (w.div_ceil(D), h.div_ceil(D));
        let shrink = |v: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; (sw2 * sh2) as usize];
            let mut cnt = vec![0f32; (sw2 * sh2) as usize];
            for y in 0..h {
                for x in 0..w {
                    let j = ((y / D) * sw2 + (x / D)) as usize;
                    out[j] += v[(y * w + x) as usize];
                    cnt[j] += 1.0;
                }
            }
            for (o, c) in out.iter_mut().zip(&cnt) {
                *o /= c.max(1.0);
            }
            out
        };
        let (c2, s2, w2) = (shrink(cw), shrink(sw), shrink(wt));
        let (oc2, os2) = infill_orientation_at(&c2, &s2, &w2, sw2, sh2);
        let grow = |v: &[f32]| -> Vec<f32> {
            (0..(w * h) as usize)
                .map(|i| {
                    let (x, y) = ((i as u32 % w) / D, (i as u32 / w) / D);
                    v[(y.min(sh2 - 1) * sw2 + x.min(sw2 - 1)) as usize]
                })
                .collect()
        };
        return (grow(&oc2), grow(&os2));
    }
    infill_orientation_at(cw, sw, wt, w, h)
}

fn infill_orientation_at(cw: &[f32], sw: &[f32], wt: &[f32], w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let n = cw.len();
    let (mut oc, mut os) = (vec![0f32; n], vec![0f32; n]);
    let mut got = vec![false; n];
    let short = w.min(h).max(1) as i32;
    let mut r = 3i32;
    while r < short {
        let (bc, bs, bw) = (box_blur(cw, w, h, r), box_blur(sw, w, h, r), box_blur(wt, w, h, r));
        let mut any = false;
        for i in 0..n {
            if !got[i] && bw[i] > 1e-3 {
                oc[i] = bc[i] / bw[i];
                os[i] = bs[i] / bw[i];
                got[i] = true;
            }
            any |= !got[i];
        }
        if !any {
            break;
        }
        r *= 3;
    }
    (oc, os)
}

fn coherent_gradient(luma: &[f32], w: u32, h: u32, sigma: i32, infill: FlowInfill) -> (Vec<f32>, Vec<f32>) {
    let (gx, gy) = sobel(luma, w, h);
    let n = luma.len();
    let (mut jxx, mut jyy, mut jxy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        jxx[i] = gx[i] * gx[i];
        jyy[i] = gy[i] * gy[i];
        jxy[i] = gx[i] * gy[i];
    }
    let jxx = box_blur(&jxx, w, h, sigma);
    let jyy = box_blur(&jyy, w, h, sigma);
    let jxy = box_blur(&jxy, w, h, sigma);
    // FORM vs ATMOSPHERE: the tensor's coherence is normalised by gradient energy, so a smooth atmospheric
    // gradient (a sunset sky, haze, still water) is as "coherent" as a cheek — and strokes then circle the sun
    // along its isophotes, the whirl tell. A painter follows form only where there is form: an EDGE within the
    // window. Where the local value range is below a fraction of a level step there is nothing to follow, so
    // the flow fades to "lay it flat" (horizontal marks), and the detail gate (which reads this magnitude)
    // rejects marks there too. Range is a value fact of the image, not a tuned constant per scene.
    // Ramp vs edge, scale-free: over a window 3× wider a slow RAMP's range grows ~3× (it keeps climbing) while
    // an EDGE's range saturates (the step is the step). The ratio small/large therefore separates atmosphere
    // (≈1/3) from form (→1) whatever the image's contrast; a floor on the small-window range keeps noise out.
    let s_small = sigma.max(2) as usize;
    let range_s = local_range(luma, w as usize, h as usize, s_small);
    let range_l = local_range(luma, w as usize, h as usize, s_small * 3);
    // FOLLOW needs the confident directions gathered before any pixel is written, so build the weighted
    // double-angle field first and carry it inward.
    let follow = (infill == FlowInfill::Follow).then(|| {
        let (mut cw, mut sw, mut wt) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        for i in 0..n {
            let disc = ((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]).sqrt();
            let coh = (disc / (jxx[i] + jyy[i] + 1e-6)).clamp(0.0, 1.0);
            let ratio = range_s[i] / (range_l[i] + 1e-4);
            let conf = coh * ((ratio - 0.35) / 0.4).clamp(0.0, 1.0) * (range_s[i] / 0.02).clamp(0.0, 1.0);
            let th = 0.5 * (2.0 * jxy[i]).atan2(jxx[i] - jyy[i]);
            cw[i] = (2.0 * th).cos() * conf;
            sw[i] = (2.0 * th).sin() * conf;
            wt[i] = conf;
        }
        infill_orientation(&cw, &sw, &wt, w, h)
    });
    let (mut ox, mut oy) = (vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        // Dominant-eigenvector orientation of the smoothed 2×2 tensor.
        let theta = 0.5 * (2.0 * jxy[i]).atan2(jxx[i] - jyy[i]);
        // Coherence in [0,1]: anisotropy of the tensor.
        let disc = ((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]).sqrt();
        let coh = (disc / (jxx[i] + jyy[i] + 1e-6)).clamp(0.0, 1.0);
        let ratio = range_s[i] / (range_l[i] + 1e-4);
        let mut edge = ((ratio - 0.35) / 0.4).clamp(0.0, 1.0) * (range_s[i] / 0.02).clamp(0.0, 1.0);
        // ATMOSPHERE is a slow ramp that stays slow over a wide window too (a sky, haze, a glow): little value
        // change even across 3 windows. Form that merely has soft modelling still climbs a lot across the wide
        // window (a cheek turns, a sleeve folds), so it keeps following its isophotes. Only true atmosphere is
        // laid FLAT — long level marks — the way a painter blends a sky, instead of circling a sun's isophotes.
        let flat = ((0.18 - range_l[i]) / 0.08).clamp(0.0, 1.0) * (1.0 - edge);
        edge = edge.max(1.0 - flat);
        // Form: the local tensor at full magnitude. Atmosphere: a vertical "gradient" (= a horizontal stroke) at a
        // low magnitude, so the direction is defined but the detail gate (which reads this magnitude) does not fire.
        // Where the picture HAS form, the local tensor at full magnitude. Where it has none, a direction at a
        // low magnitude — defined, but too weak to fire the detail gate, so the infill decides where strokes
        // POINT and never which marks are allowed.
        let (fx, fy) = match (&follow, infill) {
            (Some((oc, os)), _) => {
                let th = 0.5 * os[i].atan2(oc[i]);
                (th.cos(), th.sin())
            }
            // A named STROKE angle; the field here is a gradient, so a quarter turn off it.
            (None, FlowInfill::Angle(deg)) => {
                let g = deg.to_radians() + std::f32::consts::FRAC_PI_2;
                (g.cos(), g.sin())
            }
            // FLAT: a gradient straight down the frame, which is a level stroke across it.
            _ => (0.0, 1.0),
        };
        ox[i] = theta.cos() * coh * edge + fx * 0.05 * (1.0 - edge);
        oy[i] = theta.sin() * coh * edge + fy * 0.05 * (1.0 - edge);
    }
    (ox, oy)
}

/// The isophote (stroke) direction at a pixel: perpendicular to the luminance gradient. In a flat region the
/// gradient vanishes, so we fall back to horizontal.
fn stroke_dir(gx: f32, gy: f32) -> [f32; 2] {
    let m = (gx * gx + gy * gy).sqrt();
    if m < 1e-4 {
        [1.0, 0.0]
    } else {
        // perpendicular to (gx,gy) is (-gy,gx)
        [-gy / m, gx / m]
    }
}

/// The mixture cache's key: the colour quantised to 6 bits a channel.
fn mixture_key(c: Srgb) -> u32 {
    ((c[0] as u32 >> 2) << 12) | ((c[1] as u32 >> 2) << 6) | (c[2] as u32 >> 2)
}

/// The pigment weights (unit charge) for a quantised key, solved from the bucket's own representative colour —
/// the same answer whoever asks, in whatever order (the tile schedule's shared table).
fn mixture_for_key(key: u32, palette: &Palette, n: usize) -> Vec<f32> {
    let c: Srgb = [(((key >> 12) & 63) * 4 + 2) as u8, (((key >> 6) & 63) * 4 + 2) as u8, ((key & 63) * 4 + 2) as u8];
    let m = mixer::solve_mixture(palette, c, 3);
    let mut v = vec![0f32; n];
    for (&idx, &w) in m.pigments.iter().zip(m.weights.iter()) {
        v[idx] = w;
    }
    v
}

/// The pigment mixture for a target colour at `charge`, through a cache keyed on the quantised colour and solved
/// from the bucket's representative (`mixture_for_key`) — so the answer never depends on which colour of the
/// bucket asked first, and a table filled up front in parallel is the same table this fills lazily.
fn mixture_cached(cache: &mut std::collections::HashMap<u32, Vec<f32>>, target: Srgb, palette: &Palette, charge: f32, n: usize) -> Vec<f32> {
    let key = mixture_key(target);
    let base = cache.entry(key).or_insert_with(|| mixture_for_key(key, palette, n));
    base.iter().map(|c| c * charge).collect()
}

/// Deterministic per-index jitter in `[-0.5,0.5]` (a hashed LCG — no RNG dependency, reproducible).
fn jitter(seed: u64, k: u64) -> f32 {
    let mut z = seed.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z as f32 / u64::MAX as f32) - 0.5
}

/// Displace a stroke path with a small smooth WOBBLE perpendicular to its travel — the uneven, not-straight
/// quality of a real hand-drawn mark (a characteristic, not an error). The offset is a low-frequency sine along
/// the path with a hashed phase/amplitude, so it wanders smoothly rather than jittering, and is deterministic
/// (replay-exact). `amp` is the peak displacement in pixels.
fn waver_path(path: &[[f32; 2]], amp: f32, seed: u64, k: u64) -> Vec<[f32; 2]> {
    let n = path.len();
    if amp <= 0.05 || n < 3 {
        return path.to_vec();
    }
    let phase = jitter(seed ^ 0xA13F, k) * std::f32::consts::TAU;
    let freq = 0.6 + 1.4 * (jitter(seed ^ 0xB7C1, k.wrapping_add(1)) + 0.5); // ~0.6..2 cycles over the stroke
    let a2 = amp * (0.6 + 0.8 * (jitter(seed ^ 0xC93D, k.wrapping_add(2)) + 0.5));
    let mut out = Vec::with_capacity(n);
    for (i, p) in path.iter().enumerate() {
        let t = i as f32 / (n - 1) as f32;
        // Local tangent → perpendicular.
        let q = if i + 1 < n { path[i + 1] } else { path[i - 1] };
        let (mut dx, mut dy) = (q[0] - p[0], q[1] - p[1]);
        let dl = (dx * dx + dy * dy).sqrt().max(1e-4);
        dx /= dl;
        dy /= dl;
        // Zero at the ends (endpoints stay put), max in the middle — a bowed, wandering line.
        let env = (std::f32::consts::PI * t).sin();
        let off = a2 * env * (std::f32::consts::TAU * freq * t + phase).sin();
        out.push([p[0] - dy * off, p[1] + dx * off]);
    }
    out
}

/// sRGB distance (linear-RGB Euclidean) — cheap, for the placement error map.
fn rgb_dist(a: Srgb, b: Srgb) -> f32 {
    let (la, lb) = (color::srgb_to_linear(a), color::srgb_to_linear(b));
    ((la[0] - lb[0]).powi(2) + (la[1] - lb[1]).powi(2) + (la[2] - lb[2]).powi(2)).sqrt()
}

/// BROKEN COLOUR: perturb a stroke's target colour — lift its chroma and jitter its hue a little, per stroke —
/// so adjacent marks are DIFFERENT pure-ish colours that OPTICALLY MIX (the vibrancy of oil / gouache / pastel)
/// instead of one pre-mixed muddy tone. The hue jitter is luma-neutral (weighted to sum zero over the sRGB luma
/// coefficients), so values stay put. Deterministic per stroke → the varied load is recorded and replays exact.
fn broken_color(target: Srgb, amt: f32, seed: u64, k: u64) -> Srgb {
    if amt <= 1e-3 {
        return target;
    }
    let a = amt.clamp(0.0, 1.0);
    let (r, g, b) = (target[0] as f32 / 255.0, target[1] as f32 / 255.0, target[2] as f32 / 255.0);
    let l = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 1.0 + a * 0.35; // lift chroma
    let (mut rr, mut gg, mut bb) = (l + (r - l) * cb, l + (g - l) * cb, l + (b - l) * cb);
    // Luma-neutral hue jitter in a BALANCED chroma plane (YCbCr's Cb/Cr axes), so no channel carries a 4x
    // compensation gain — solving the old "set blue to cancel the luma" rule dumped up to 3.9x the jitter into
    // blue, the blue flecks on every shadow mass. And PROPORTIONAL to the value that is there: broken colour
    // varies the colour a passage has, so a dark mass stays a solid dark (an absolute jitter was a 20% swing on
    // a luma-0.15 mass, the light flecks in the darks) while lights and mid-tones keep their optical vibrancy.
    let s = a * 0.16 * (0.2 + 0.8 * l);
    let dcb = s * jitter(seed ^ 0x00B4, k);
    let dcr = s * jitter(seed ^ 0x00B5, k.wrapping_add(1));
    rr += 1.402 * dcr;
    gg += -0.344 * dcb - 0.714 * dcr;
    bb += 1.772 * dcb;
    [(rr * 255.0).round().clamp(0.0, 255.0) as u8, (gg * 255.0).round().clamp(0.0, 255.0) as u8, (bb * 255.0).round().clamp(0.0, 255.0) as u8]
}

/// Grow a stroke in ONE direction (`sign` = +1 forward, −1 backward) from the seed along the orientation
/// field, ending when the reference colour drifts too far from the stroke's colour or the half-length cap hits.
#[allow(clippy::too_many_arguments)]
fn grow_half(x0: f32, y0: f32, sign: f32, radius: f32, gx: &[f32], gy: &[f32], reference: &RgbImage, color0: Srgb, protect: Option<&[bool]>, region: Option<(&[bool], bool)>, hard: Option<(&[f32], f32)>, len_mul: f32, stop_tol: f32) -> Vec<[f32; 2]> {
    let (w, h) = (reference.width(), reference.height());
    // Strokes read as brushwork when they follow form for a mark's length — but a stroke that runs radius×4 (a
    // third of a coarse pass's image width) drags one colour across whole objects and reads as SMEAR, the mush
    // tell. Cap the travel at a couple of brush-widths; the colour-drift and edge-stop rules below still cut it
    // shorter at a real boundary. `len_mul` shortens DETAIL strokes further so features get crisp marks.
    let max_len = (radius * 2.2 * len_mul).max(radius + 1.0);
    let step = (radius * 0.6).max(1.0);
    let mut pts = Vec::new();
    let mut last = [0f32, 0f32];
    let (mut x, mut y) = (x0, y0);
    let mut travelled = 0.0;
    while travelled < max_len {
        let (ix, iy) = (x.round().clamp(0.0, w as f32 - 1.0) as usize, y.round().clamp(0.0, h as f32 - 1.0) as usize);
        let i = iy * w as usize + ix;
        // Negative painting: the stroke terminates at the protected shape's edge (paint AROUND it).
        if let Some(m) = protect {
            if m.get(i).copied().unwrap_or(false) {
                break;
            }
        }
        // Focal hard-edge: the stroke terminates when it leaves its seed's region (subject vs ground), keeping
        // the silhouette crisp.
        if let Some((mask, seed_region)) = region {
            if travelled > 0.0 && mask.get(i).copied().unwrap_or(seed_region) != seed_region {
                break;
            }
        }
        // EDGE HARDNESS (§7 / edge-control craft): terminate at a hard (high value-contrast) boundary so the
        // two masses meet crisply instead of smearing across. Low-contrast boundaries aren't in the map, so the
        // stroke crosses them freely and stays soft — lost-and-found emerges along a contour.
        if let Some((hmap, hthr)) = hard {
            if travelled > 0.0 && hmap.get(i).copied().unwrap_or(0.0) >= hthr {
                break;
            }
        }
        let mut d = stroke_dir(gx[i], gy[i]);
        d = [d[0] * sign, d[1] * sign];
        // Keep the direction from flipping 180° between steps.
        if travelled > 0.0 && last[0] * d[0] + last[1] * d[1] < 0.0 {
            d = [-d[0], -d[1]];
        }
        x += d[0] * step;
        y += d[1] * step;
        if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
            break;
        }
        // Stop where the reference no longer matches this stroke's colour (Hertzmann's rule), after a MINIMUM
        // length so strokes read as deliberate brushwork, not one-step patchy dabs. A slightly looser colour
        // tolerance + a longer minimum keeps marks continuous instead of speckled.
        let here = reference.get_pixel(x as u32, y as u32).0;
        if rgb_dist(here, color0) > stop_tol && travelled > step * 2.0 {
            break;
        }
        pts.push([x, y]);
        last = d;
        travelled += step;
    }
    pts
}

/// How far the reference may drift from a stroke's colour before the stroke stops (Hertzmann's rule): the
/// classic tolerance. Under `gradation` a stroke on a slow ramp stops sooner, so a soft mass is laid as graded
/// marks instead of one flat patch.
const STOP_TOL: f32 = 0.16;

/// Grow a stroke through the seed in BOTH directions (Hertzmann), so a seed mid-feature paints the whole
/// isophote it sits on, not just the half below it.
#[allow(clippy::too_many_arguments)]
fn grow_path(x0: f32, y0: f32, radius: f32, gx: &[f32], gy: &[f32], reference: &RgbImage, color0: Srgb, protect: Option<&[bool]>, region: Option<(&[bool], bool)>, hard: Option<(&[f32], f32)>, len_mul: f32, stop_tol: f32) -> Vec<[f32; 2]> {
    let mut back = grow_half(x0, y0, -1.0, radius, gx, gy, reference, color0, protect, region, hard, len_mul, stop_tol);
    back.reverse();
    let fwd = grow_half(x0, y0, 1.0, radius, gx, gy, reference, color0, protect, region, hard, len_mul, stop_tol);
    back.push([x0, y0]);
    back.extend(fwd);
    back
}

/// Condition the reference with AERIAL PERSPECTIVE from a depth map (RFC §5.5). Distant passages are veiled
/// toward the scene's atmosphere colour and lose contrast, so — once painted — the background recedes and the
/// foreground advances. This is what gives the painting depth "layers" instead of one flat plane. It also makes
/// the downstream edge-hardness and detail gates behave by depth for free: a veiled, low-contrast background
/// yields few hard edges and little detail, exactly as a painter treats distance.
fn recede(input: &RgbImage, depth: &[f32], haze: f32) -> RgbImage {
    let (w, h) = input.dimensions();
    // The atmosphere colour: the mean of the brightest ~12% of pixels (usually sky / light haze). Distance
    // shifts toward it. Falls back to a light neutral if the image is uniform.
    let mut lumas: Vec<(f32, usize)> = input.pixels().enumerate().map(|(i, p)| (color::linear_luma(color::srgb_to_linear(p.0)), i)).collect();
    lumas.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = (lumas.len() / 8).max(1);
    let (mut ar, mut ag, mut ab) = (0f32, 0f32, 0f32);
    for &(_, i) in lumas.iter().take(take) {
        let p = input.get_pixel((i as u32) % w, (i as u32) / w).0;
        ar += p[0] as f32;
        ag += p[1] as f32;
        ab += p[2] as f32;
    }
    let atmos = [ar / take as f32, ag / take as f32, ab / take as f32];
    let haze = haze.clamp(0.0, 1.0);
    RgbImage::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let d = depth.get(i).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        // Distance = how far back; only the farther half veils appreciably (nonlinear), so the foreground keeps
        // its full colour and the recession reads as depth, not an overall fog.
        let far = (1.0 - d).clamp(0.0, 1.0);
        let veil = (haze * far * far).clamp(0.0, 0.92);
        let src = input.get_pixel(x, y).0;
        image::Rgb([
            (src[0] as f32 * (1.0 - veil) + atmos[0] * veil).round().clamp(0.0, 255.0) as u8,
            (src[1] as f32 * (1.0 - veil) + atmos[1] * veil).round().clamp(0.0, 255.0) as u8,
            (src[2] as f32 * (1.0 - veil) + atmos[2] * veil).round().clamp(0.0, 255.0) as u8,
        ])
    })
}

/// Coarsen an image to an ARMATURE: downsample to `side` (longest edge) then upsample back, smoothly — so the
/// structure survives but the fine detail is gone. The brush then invents the surface instead of tracing it.
/// Per-pixel blend of two same-size armatures by a 0..1 mask: `mask=1` takes `fine`, `mask=0` takes `coarse`.
/// Used for the focal armature — fine inside the (feathered) face, coarse outside.
fn blend_by_mask(coarse: &RgbImage, fine: &RgbImage, mask: &[f32], w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let m = mask[(y * w + x) as usize].clamp(0.0, 1.0);
        let c = coarse.get_pixel(x, y).0;
        let f = fine.get_pixel(x, y).0;
        image::Rgb([
            (c[0] as f32 * (1.0 - m) + f[0] as f32 * m).round() as u8,
            (c[1] as f32 * (1.0 - m) + f[1] as f32 * m).round() as u8,
            (c[2] as f32 * (1.0 - m) + f[2] as f32 * m).round() as u8,
        ])
    })
}

/// Build a STRUCTURE-PRESERVING armature (RFC §1.1/§5). The old `coarsen` was a BLUR, which destroys structure
/// (edges, the boundaries of value masses) along with texture — so the engine painted a structureless smear. This
/// instead flattens *texture* while KEEPING edges (an edge-preserving / bilateral smooth), then quantises the
/// result into a few VALUE MASSES with clean boundaries (posterise). The armature is therefore coarse in texture
/// but SHARP in structure — the beard is a dark mass with a defined edge, the face a light mass, the eyes dark
/// accents — so the strokes paint recognizable form instead of averaging blurry colour. `side` is the structure
/// resolution (LARGER = more structure retained → smaller smoothing radius); `levels` = number of value masses.
fn structure_armature(img: &RgbImage, side: u32, levels: u32, gradation: f32) -> RgbImage {
    let (w, h) = img.dimensions();
    // Structure resolution → spatial radius: a coarser armature removes more texture (bigger radius).
    let r = ((w.min(h) as f32 / side.max(1) as f32).round() as i32).clamp(1, 16);
    // SPEED: a large radius is run at REDUCED resolution (downsample → small-radius bilateral → upsample), a
    // standard bilateral speedup, so the cost is bounded regardless of coarseness. Fine tiers (small r) run at
    // full resolution so the focal region stays crisp.
    let mut sm;
    let bil;
    // The plain (un-bilateraled) picture at the armature's own resolution — what a ramp is eased toward under
    // `gradation`: no flattening, and no blur beyond the resolution the tier already has (easing toward a blur
    // smeared a background face).
    let mut plain: Option<RgbImage> = None;
    if r <= 3 {
        // FINE armature: keep the reference's own detail/texture — a bilateral here would over-smooth the face
        // into a photo-smooth "cut-and-paste" surface while the rest stays brushy. Natural, consistent brushwork.
        bil = img.clone();
    } else if r > 4 {
        // COARSE armature: bilateral at reduced resolution (bounded cost) to flatten texture, keep edges.
        let scale = (4.0 / r as f32).clamp(0.1, 1.0);
        let (sw2, sh2) = ((w as f32 * scale).round().max(1.0) as u32, (h as f32 * scale).round().max(1.0) as u32);
        sm = imageops::resize(img, sw2, sh2, imageops::FilterType::Triangle);
        if gradation > 0.0 {
            plain = Some(imageops::resize(&sm, w, h, imageops::FilterType::Triangle));
        }
        sm = bilateral(&sm, 4);
        bil = imageops::resize(&sm, w, h, imageops::FilterType::Triangle);
    } else {
        bil = bilateral(img, r);
    }
    // Quantise into VALUE MASSES with clean boundaries (posterise). Quantising each RGB channel INDEPENDENTLY
    // shifts hue at every step boundary — skin bands through magenta/green. Instead quantise the LUMA and rescale
    // the pixel to the snapped value, keeping its chroma: the masses read as value steps, not colour steps.
    let mut out = bil;
    // GRADATION: a slow RAMP is not an edge. The bilateral flattens a soft mass's modelling (a cloud's turning
    // form, a soft-lit wall, still water: differences below its colour sigma) and the value snap below then
    // cuts what is left into flat bands with contour edges — the posterised-cloud tell. The same scale-free
    // test as the flow rule tells a ramp from an edge: over a window 3× wider a ramp's value range keeps
    // growing (ratio ≈ 1/3) while an edge's saturates (→ 1). Where the picture is a ramp, `gradation` brings
    // the bilateral back toward a plain smooth of the source (texture gone, gradation kept) and holds the
    // snap off, so the mass keeps its turning form. 0 = the value masses exactly as before.
    let ramp: Option<Vec<f32>> = (gradation > 0.0 && r > 3 && levels >= 2).then(|| ramp_field(&out, r.max(2) as usize, levels).into_iter().map(|v| v * gradation).collect());
    if let (Some(ramp), Some(soft)) = (&ramp, &plain) {
        for (i, px) in out.pixels_mut().enumerate() {
            let k = ramp[i];
            if k <= 0.0 {
                continue;
            }
            let sp = soft.get_pixel((i % w as usize) as u32, (i / w as usize) as u32).0;
            for c in 0..3 {
                px.0[c] = (px.0[c] as f32 + k * (sp[c] as f32 - px.0[c] as f32)).clamp(0.0, 255.0) as u8;
            }
        }
    }
    if levels >= 2 {
        let step = 1.0 / (levels - 1) as f32;
        // Quantise only where there is an EDGE to make crisp. A value mass is flat, but a painter keeps a smooth
        // GRADIENT smooth — a sunset sky, a field falling off into haze — while posterising it snaps a wide, slow
        // gradient into flat bands with jagged contour edges the painting then faithfully copies (the tell on
        // any low-contrast scene). The snap strength is the local value RANGE over the smoothing window measured
        // against one level step: a real boundary (a step or more across the window) snaps fully; a slow
        // gradient (a small fraction of a step) is left continuous. Texture is already flattened by the bilateral.
        let yl = luma_map(&out);
        let range = local_range(&yl, w as usize, h as usize, r.max(2) as usize);
        for (i, p) in out.pixels_mut().enumerate() {
            let r = p.0[0] as f32 / 255.0;
            let g = p.0[1] as f32 / 255.0;
            let b = p.0[2] as f32 / 255.0;
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            if y > 1e-4 {
                // Snap to the nearest value level, but FLOOR the bottom bin at step/2: plain rounding sent every
                // value below step/2 to exactly 0 — a black hole that turned dark grass and shadow masses PURE
                // BLACK before a stroke was laid. A shadow mass is a solid dark, never black (RFC §3.3).
                let snapped = ((y / step).round() * step).max(0.5 * step);
                let mut edge = ((range[i] - 0.35 * step) / (0.5 * step)).clamp(0.0, 1.0);
                if let Some(rp) = &ramp {
                    edge *= 1.0 - rp[i];
                }
                let yq = y + (snapped - y) * edge;
                let s = (yq / y).clamp(0.0, 2.0);
                p.0[0] = (r * s * 255.0).clamp(0.0, 255.0) as u8;
                p.0[1] = (g * s * 255.0).clamp(0.0, 255.0) as u8;
                p.0[2] = (b * s * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Where the picture is a slow RAMP rather than an edge, per pixel in [0,1] (the flow rule's scale-free test at
/// window `r`: a ramp's value range keeps growing over a window 3× wider, an edge's saturates), and only where
/// there is a gradient worth keeping against the `levels` value step. The masses' stages (the armature's
/// bilateral and snap, the family invariant) hold off here under `gradation`, so a cloud, a soft-lit wall or
/// still water keep their turning form instead of a few flat tones.
pub fn ramp_field(img: &RgbImage, r: usize, levels: u32) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    // Judge the SMOOTHED picture: texture (a cloud's puffs, grass) and the armature's own tone steps read as
    // edges in a small window and hid the very ramps this is for (measured: the cloud cores came out 0).
    let smooth = imageops::blur(img, (r as f32).max(1.0));
    let yl = luma_map(&smooth);
    let range_s = local_range(&yl, w, h, r);
    let range_l = local_range(&yl, w, h, r * 3);
    let step = 1.0 / (levels.max(2) - 1) as f32;
    let raw: Vec<f32> = (0..yl.len())
        .map(|i| {
            let ratio = range_s[i] / (range_l[i] + 1e-4);
            let is_ramp = 1.0 - ((ratio - 0.35) / 0.4).clamp(0.0, 1.0);
            let has_slope = (range_l[i] / (0.5 * step)).clamp(0.0, 1.0);
            is_ramp * has_slope
        })
        .collect();
    // The square windows leave a blocky field; soften it so the gating has no seams of its own — but never let
    // the smoothing bleed a ramp onto its neighbouring EDGE (a timber beam next to a pale wall softened when it
    // did): the raw test gates the smoothed field.
    let smooth_field = box_blur(&raw, w as u32, h as u32, r as i32);
    smooth_field.iter().zip(&raw).map(|(sm, rw)| sm * rw.clamp(0.0, 1.0)).collect()
}

/// The ARMATURE of a from-scratch painting (RFC §1.1): the picture at `side` px on its short side — structure
/// with no detail to trace — its values snapped into masses at that resolution, brought back to canvas size.
/// (`structure_armature` smooths at full resolution with a radius capped at 16 px, so on a large sheet it can
/// never be coarse: the painter tracked the source through it.)
/// A NEW painting's armature tiers. The background is read at this many pixels across the sheet's short side
/// when the picture has faces, and [`NEW_BACKGROUND_SIDE_PLAIN`] when it has none (the picture itself is the
/// subject); a figure and a face are read at these many pixels ACROSS THEIR OWN EXTENT (RFC §5.2). Fine
/// enough that the THINGS in the picture survive — a spoon on a shelf, a pencil on a table — and coarse
/// enough that no texture does: the surface is the brush's. (At a quarter of this the armature threshold
/// experiment, RFC §12.1, lost every object smaller than a head.)
pub const NEW_BACKGROUND_SIDE: u32 = 384;
pub const NEW_BACKGROUND_SIDE_PLAIN: u32 = 512;
pub const NEW_FIGURE_ACROSS: f32 = 384.0;
pub const NEW_FACE_ACROSS: f32 = 256.0;

pub fn coarse_armature(img: &RgbImage, side: u32, levels: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let short = w.min(h).max(1);
    let side = side.clamp(8, short);
    let scale = side as f32 / short as f32;
    let (sw, sh) = (((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1));
    let small = imageops::resize(img, sw, sh, imageops::FilterType::Triangle);
    let up = imageops::resize(&small, w, h, imageops::FilterType::CatmullRom);
    // The DRAWING of a painting is its masses' contours. The upsampled field is smooth, so its value contours
    // are clean curves: snap the values into masses where a boundary runs (the edge-gated snap), and the
    // masses meet along firm edges — the block-in shapes — with nothing inside them to trace. (Left soft,
    // every edge was a blur and no stroke had a boundary to stop at: the picture lost its drawing.)
    structure_armature(&up, side.max(short / 16), levels, 0.0)
}

/// The per-plane MINIMUM BRUSH radius of a from-scratch painting (RFC §9), as (background, figure, focal):
/// each plane's brush stops at the size of ITS armature's pixel (`short / side`, a little under), so structure
/// and surface never meet at the same scale (§1.1) and the floor grows with depth as the tiers coarsen.
/// `sides` = the (background, figure, face) armature resolutions; a missing tier takes the one behind it.
pub fn plane_floors(w: u32, h: u32, min_brush: f32, sides: (u32, Option<u32>, Option<u32>)) -> (f32, f32, f32) {
    let short = w.min(h) as f32;
    let floor = |side: u32| (0.85 * short / side.max(1) as f32).max(min_brush * 0.5);
    let bg = sides.0;
    let body = sides.1.unwrap_or(bg).max(bg);
    let face = sides.2.unwrap_or(body).max(body);
    (floor(bg), floor(body), floor(face))
}

/// The characteristic EXTENT (px) of the things a mask marks: the median, over its MAIN connected regions
/// (those at least a quarter the area of the largest — stray fragments of a matte do not count), of each
/// region's smaller bounding-box side. A face mask's extent is a face's width; a matte's, a figure's.
/// Measured on a quarter-scale grid. `None` when the mask marks nothing.
pub fn region_extent(mask: &[f32], w: u32, h: u32) -> Option<f32> {
    let (w, h) = (w as usize, h as usize);
    if mask.len() != w * h {
        return None;
    }
    let q = 4usize;
    let (gw, gh) = (w.div_ceil(q), h.div_ceil(q));
    let on: Vec<bool> = (0..gw * gh).map(|i| mask[((i / gw) * q).min(h - 1) * w + ((i % gw) * q).min(w - 1)] > 0.5).collect();
    let mut seen = vec![false; gw * gh];
    let mut regions: Vec<(usize, f32)> = Vec::new();
    let min_cells = ((gw * gh) as f32 * 0.0005).max(4.0) as usize;
    for start in 0..gw * gh {
        if !on[start] || seen[start] {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1, mut n) = (usize::MAX, usize::MAX, 0usize, 0usize, 0usize);
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % gw, i / gw);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            n += 1;
            for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx < 0 || ny < 0 || nx >= gw as i64 || ny >= gh as i64 {
                    continue;
                }
                let j = ny as usize * gw + nx as usize;
                if on[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if n >= min_cells {
            regions.push((n, ((x1 - x0 + 1).min(y1 - y0 + 1) * q) as f32));
        }
    }
    let largest = regions.iter().map(|r| r.0).max()?;
    let mut extents: Vec<f32> = regions.iter().filter(|r| r.0 * 4 >= largest).map(|r| r.1).collect();
    extents.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(extents[extents.len() / 2])
}

/// The minimum brush at every pixel: the focal floor inside the face mask, the figure floor inside the subject
/// matte (or everywhere, when no subject was found — the picture itself is the subject), else the background's.
pub fn plane_floor_field(w: u32, h: u32, min_brush: f32, sides: (u32, Option<u32>, Option<u32>), subject: Option<&[f32]>, face: Option<&[f32]>, hair: Option<&[f32]>) -> Vec<f32> {
    let (bg, body, focal) = plane_floors(w, h, min_brush, sides);
    let at = |m: Option<&[f32]>, i: usize| m.and_then(|m| m.get(i).copied()).unwrap_or(0.0).clamp(0.0, 1.0);
    // The planes meet along a RAMP, not a step. Thresholding each mask at 0.5 put a hard line through
    // anything that crossed a plane boundary: the focal region is a detector's box, so a long beard was
    // painted with a 2 px brush above the chin and an 8 px brush below it, and the seam between them read
    // as a cut straight across the face. The masks are already feathered and the armature already blends
    // through them (`blend_by_mask`); only the minimum brush was stepping.
    //
    // The blend is GEOMETRIC because a brush ladder is: each pass halves its predecessor, so a linear ramp
    // from 2 px to 8 px would spend most of its width in the coarse half and still read as an edge. A
    // geometric ramp crosses the octaves evenly.
    let glerp = |a: f32, b: f32, t: f32| if t <= 0.0 { a } else if t >= 1.0 { b } else { a * (b / a).powf(t) };
    (0..(w as usize * h as usize))
        .map(|i| {
            // No subject found = the picture itself is the subject, so the body floor holds everywhere.
            let s = if subject.is_none() { 1.0 } else { at(subject, i) };
            // HAIR sits between the body and the focal plane: it is the finest structure on a figure after the
            // features, and the semantic pass used to call it a "coarse wash" — softer than the body — which is
            // backwards for anything with a mane or a beard. Applied BEFORE the face so a beard inside the face
            // box still gets the focal floor, never coarsened back up by its own mask.
            let hairy = glerp(glerp(bg, body, s), (body * focal).sqrt(), at(hair, i));
            glerp(hairy, focal, at(face, i))
        })
        .collect()
}

/// The COVERAGE budget of a from-scratch painting: for each rung of the brush ladder, the seed cells of the
/// planes that rung may touch, summed, with half again for the restatements. A count of marks needed to cover
/// and restate — not a density of marks per pixel.
pub fn from_scratch_budget(w: u32, h: u32, brush_sizes: &[f32], min_brush: f32, floor: &[f32]) -> usize {
    let n = floor.len().max(1) as f64;
    let mut total = 0.0f64;
    for &r in brush_sizes {
        let r = r.max(min_brush);
        let open = floor.iter().filter(|&&f| r >= f).count() as f64 / n;
        let grid = (r * 0.9).max(1.5) as f64;
        total += (w as f64 / grid).ceil() * (h as f64 / grid).ceil() * open;
    }
    ((total * 1.5) as usize).max(500)
}

/// Local value RANGE (max − min) of a luma field over a square window of half-width `r` — a cheap "is there an
/// edge nearby" measure. Separable (row max/min then column max/min), so it costs O(w·h·r).
fn local_range(luma: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    // A SLIDING min and max, not a re-scan per pixel. The old form was separable but still walked the whole
    // window at every pixel — O(radius) each — and the flow field asks for radii up to ~72 on a 2048² sheet,
    // which made this ONE function 93% of a watercolour's paint time (330 s of 352). A monotonic deque gives
    // the same answer in amortised O(1): each index is pushed and popped once per line. Bit-identical by
    // construction, because the front of the deque IS the window's extreme.
    let mut row_max = vec![0f32; w * h];
    let mut row_min = vec![0f32; w * h];
    let mut dmax: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut dmin: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for y in 0..h {
        let base = y * w;
        dmax.clear();
        dmin.clear();
        let mut emit = |i: usize, dmax: &mut std::collections::VecDeque<usize>, dmin: &mut std::collections::VecDeque<usize>| {
            let lo = i.saturating_sub(r);
            while dmax.front().is_some_and(|&f| f < lo) {
                dmax.pop_front();
            }
            while dmin.front().is_some_and(|&f| f < lo) {
                dmin.pop_front();
            }
            row_max[base + i] = luma[base + *dmax.front().unwrap()];
            row_min[base + i] = luma[base + *dmin.front().unwrap()];
        };
        for x in 0..w {
            let v = luma[base + x];
            while dmax.back().is_some_and(|&b| luma[base + b] <= v) {
                dmax.pop_back();
            }
            dmax.push_back(x);
            while dmin.back().is_some_and(|&b| luma[base + b] >= v) {
                dmin.pop_back();
            }
            dmin.push_back(x);
            if x >= r {
                emit(x - r, &mut dmax, &mut dmin);
            }
        }
        // The last `r` windows are clipped by the right edge, so they are finished after the scan.
        for i in w.saturating_sub(r)..w {
            emit(i, &mut dmax, &mut dmin);
        }
    }
    let mut out = vec![0f32; w * h];
    for x in 0..w {
        dmax.clear();
        dmin.clear();
        let mut emit = |i: usize, dmax: &mut std::collections::VecDeque<usize>, dmin: &mut std::collections::VecDeque<usize>| {
            let lo = i.saturating_sub(r);
            while dmax.front().is_some_and(|&f| f < lo) {
                dmax.pop_front();
            }
            while dmin.front().is_some_and(|&f| f < lo) {
                dmin.pop_front();
            }
            out[i * w + x] = row_max[*dmax.front().unwrap() * w + x] - row_min[*dmin.front().unwrap() * w + x];
        };
        for y in 0..h {
            let (vx, vn) = (row_max[y * w + x], row_min[y * w + x]);
            while dmax.back().is_some_and(|&b| row_max[b * w + x] <= vx) {
                dmax.pop_back();
            }
            dmax.push_back(y);
            while dmin.back().is_some_and(|&b| row_min[b * w + x] >= vn) {
                dmin.pop_back();
            }
            dmin.push_back(y);
            if y >= r {
                emit(y - r, &mut dmax, &mut dmin);
            }
        }
        for i in h.saturating_sub(r)..h {
            emit(i, &mut dmax, &mut dmin);
        }
    }
    out
}

/// The reference a FINE layer paints from: the armature (structure) plus the SOURCE's fine texture inside the
/// armature's flat masses. `residual = source − blur(source, σ ≈ 1.5·radius)` is the texture at this pass's
/// scale; it is added with weight `amount · flat`, where `flat` fades to zero wherever the armature's local value
/// range spans a level step (an edge) — texture in the masses, never overshoot at the boundaries.
fn texture_reference(armature: &RgbImage, source: &RgbImage, radius: f32, levels: u32, amount: f32, subject: Option<&[f32]>) -> RgbImage {
    let (w, h) = (armature.width() as usize, armature.height() as usize);
    let blurred = imageops::blur(source, (radius * 1.5).max(1.0));
    let luma = luma_map(armature);
    let range = local_range(&luma, w, h, (radius.round() as usize).max(2));
    let step = 1.0 / (levels - 1) as f32;
    let mut out = armature.clone();
    for (i, px) in out.pixels_mut().enumerate() {
        let flat = (1.0 - (range[i] - 0.35 * step) / (0.5 * step)).clamp(0.0, 1.0);
        // Texture is for the masses the plan simplified (a field, grass, foliage, the ground under a bench) —
        // never a FACE: there the source's residual is eye-socket and beard scrawl (the user's regression). The
        // detected face mask says where; without one, the subject matte stands in.
        let bg = 1.0 - subject.map(|m| m.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0);
        let k = amount * flat * bg;
        if k <= 0.0 {
            continue;
        }
        let (x, y) = ((i % w) as u32, (i / w) as u32);
        let s = source.get_pixel(x, y).0;
        let b = blurred.get_pixel(x, y).0;
        for c in 0..3 {
            let v = px.0[c] as f32 + k * (s[c] as f32 - b[c] as f32);
            px.0[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Edge-preserving bilateral smooth at spatial radius `r` (px): flatten texture while keeping edges.
fn bilateral(img: &RgbImage, r: i32) -> RgbImage {
    let (w, h) = img.dimensions();
    let r = r.max(1);
    let sigma_c = 30.0_f32;
    let inv2s2 = 1.0 / (2.0 * (r as f32 * 0.6).max(1.0).powi(2));
    let inv2c2 = 1.0 / (2.0 * sigma_c * sigma_c);
    let sw: Vec<f32> = (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (-((dx * dx + dy * dy) as f32) * inv2s2).exp())).collect();
    let mut out = RgbImage::new(w, h);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let c0 = img.get_pixel(x as u32, y as u32).0;
            let (mut acc, mut wsum) = ([0f32; 3], 0f32);
            let mut si = 0usize;
            for dy in -r..=r {
                let yy = (y + dy).clamp(0, h as i32 - 1) as u32;
                for dx in -r..=r {
                    let xx = (x + dx).clamp(0, w as i32 - 1) as u32;
                    let c = img.get_pixel(xx, yy).0;
                    let dc = (c[0] as f32 - c0[0] as f32).powi(2) + (c[1] as f32 - c0[1] as f32).powi(2) + (c[2] as f32 - c0[2] as f32).powi(2);
                    let wgt = sw[si] * (-dc * inv2c2).exp();
                    si += 1;
                    acc[0] += c[0] as f32 * wgt;
                    acc[1] += c[1] as f32 * wgt;
                    acc[2] += c[2] as f32 * wgt;
                    wsum += wgt;
                }
            }
            let p = out.get_pixel_mut(x as u32, y as u32);
            for k in 0..3 {
                p.0[k] = (acc[k] / wsum.max(1e-6)).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// A pass-level CRITIC (RFC PAINT-1 §10.1): scores a rendered canvas so the painter can accept or reject a
/// whole PASS. Injected as a closure so the loop logic is testable offline (the aesthetic/CLIP scorer is wired
/// by the CLI). Higher is better.
pub type PassCritic<'a> = dyn Fn(&RgbImage) -> f32 + 'a;

/// Paint a reference image under the PAINT-1 constraints. See [`paint_critiqued`] for the pass-level critic.
pub fn paint_from_image(input: &RgbImage, p: &PaintParams) -> PaintResult {
    paint_inner(input, p, None, 0.0, None, None)
}

/// As [`paint_from_image`] but reports progress: `progress(placed)` is called as strokes are laid, so the CLI
/// can render a live progress bar. `placed` counts up to (at most) the stroke budget.
pub fn paint_from_image_progress(input: &RgbImage, p: &PaintParams, progress: &dyn Fn(PaintProgress)) -> PaintResult {
    paint_inner(input, p, None, 0.0, None, Some(progress))
}

/// Paint an element ONTO an existing canvas (a composition layer): the strokes stack over whatever is already
/// there, so a nearer element occludes farther ones. Use `p.paint_mask` for the element's footprint and
/// `p.layer_brush` for its brush. The returned score holds only THIS layer's strokes (the caller concatenates).
pub fn paint_onto(base: Canvas, input: &RgbImage, p: &PaintParams) -> PaintResult {
    paint_inner(input, p, None, 0.0, Some(base), None)
}

/// Paint with a pass-level CRITIC (§10.1): after each stage pass the canvas is scored; a pass that does not
/// improve the score by at least `margin` is REJECTED (the canvas + score are rolled back) and its stage
/// recorded as tabu, so the loop never keeps a configuration that made the painting worse. Pass-level only —
/// per-stroke scoring is prohibitively expensive and rejected outright (§10.1, N4).
pub fn paint_critiqued(input: &RgbImage, p: &PaintParams, critic: &PassCritic, margin: f32) -> PaintResult {
    paint_inner(input, p, Some(critic), margin, None, None)
}

fn paint_inner(input: &RgbImage, p: &PaintParams, critic: Option<&PassCritic>, margin: f32, base: Option<Canvas>, progress: Option<&dyn Fn(PaintProgress)>) -> PaintResult {
    let (w, h) = (input.width(), input.height());
    // PROFILE (`PLAKAT_PAINT_PROFILE=1`): per-stage and per-pass wall-clock times, printed to stderr at the end.
    // This is how the stroke cost was found to be the mixture solver, not the brush — keep it.
    let started = std::time::Instant::now();
    let mut stats: Vec<PassStat> = Vec::new();
    let prof_on = std::env::var("PLAKAT_PAINT_PROFILE").is_ok();
    let mut prof_acc: Vec<(&'static str, f64)> = Vec::new();
    let mut prof_t = std::time::Instant::now();
    let lap = |name: &'static str, acc: &mut Vec<(&'static str, f64)>, t: &mut std::time::Instant| { if prof_on { acc.push((name, t.elapsed().as_secs_f64())); *t = std::time::Instant::now(); } };
    // The subject as given, before it is reduced to an armature: the fine layers look at it for TEXTURE.
    let source: &RgbImage = input;
    // HAIR FLOW (see `hair_flow_field`): which way hair GROWS, as one angle per armature cell. Built here,
    // from the picture as given, because two lines below `input` becomes the armature and the growth
    // direction is precisely what the armature has thrown away. Only computed where there is hair to paint.
    let hair_flow: Option<HairFlow> = p.hair_mask.as_ref().filter(|m| m.len() == (w * h) as usize).map(|_| {
        let side = p.armature_face_side.or(p.armature_body_side).unwrap_or(NEW_BACKGROUND_SIDE);
        hair_flow_field(source, side)
    });
    // The reference the strokes read is a low-resolution ARMATURE — structure without detail (§1.1). The output
    // canvas stays full size; only the thing being painted FROM is coarsened.
    let armature_owned;
    let input: &RgbImage = match p.armature_side.filter(|&s| s > 0) {
        Some(s) => {
            // MULTI-REGION armature (RFC §5.2): coarsest background, then each region (subject body, semantic
            // parts, face) painted from its OWN armature resolution, blended coarse→fine so a finer region wins
            // where they overlap (the face over a coarse beard tier). Falls back to a uniform coarse armature.
            let px = (w * h) as usize;
            let mut tiers: Vec<(&[f32], u32)> = Vec::new();
            if let (Some(m), Some(bs)) = (&p.subject_mask, p.armature_body_side) {
                if m.len() == px && bs > s {
                    tiers.push((m, bs));
                }
            }
            for (m, side) in &p.region_tiers {
                if m.len() == px && *side > s {
                    tiers.push((m, *side));
                }
            }
            if let (Some(m), Some(fs)) = (&p.face_mask, p.armature_face_side) {
                if m.len() == px && fs > s {
                    tiers.push((m, fs));
                }
            }
            tiers.sort_by_key(|(_, side)| *side); // coarse → fine, so the finest region is laid last and wins
            // STRUCTURE-PRESERVING armature (not a blur): value masses with sharp edges, per region resolution.
            let levels = p.armature_levels.max(2);
            // A from-scratch painting reads a genuinely LOW-RESOLUTION armature (see `coarse_armature`).
            let build = |side: u32, gradation: f32| if p.from_scratch { coarse_armature(input, side, levels) } else { structure_armature(input, side, levels, gradation) };
            let mut arm = build(s, p.gradation);
            for (mask, side) in tiers {
                let lvl = build(side, 0.0);
                arm = blend_by_mask(&arm, &lvl, mask, w, h);
            }
            armature_owned = arm;
                    &armature_owned
        }
        None => input,
    };
    // FROM SCRATCH: the armature is all the painter ever sees. Whatever reads `source` further down — the
    // fine layers' texture, a wash medium's masses, a drawn line — reads the armature instead.
    let source: &RgbImage = if p.from_scratch { input } else { source };
    // AERIAL PERSPECTIVE (§5.5): condition the reference by depth so the background recedes and the foreground
    // advances — this is what gives the painting foreground/background "layers" rather than one flat plane.
    let receded_owned;
    let input: &RgbImage = match &p.depth {
        Some(d) if d.len() == (w * h) as usize && p.haze > 0.0 => {
            receded_owned = recede(input, d, p.haze);
            &receded_owned
        }
        _ => input,
    };
    // PAINT DARKER, IT DRIES LIGHTER: a medium with a positive dry shift (watercolour, ink-wash) lightens every
    // painted value on drying (`Finish::dry_shift`, v' = v + s·(1 − v)). A watercolourist knows this and lays the
    // wash darker than the value wanted, so the DRIED sheet lands on it. Read the reference through the exact
    // inverse of the drying step, v = (v' − s)/(1 − s), and the finished painting matches the reference's value
    // instead of sitting a shift above it (measured: a watercolour 12% too light). Negative shifts (gouache's
    // matte compression) are a look, not an error, and are left alone.
    let dried_owned;
    let input: &RgbImage = if p.dry_shift > 1e-3 {
        let s = p.dry_shift.min(0.5);
        let mut img = input.clone();
        for px in img.pixels_mut() {
            for c in 0..3 {
                let v = px.0[c] as f32 / 255.0;
                px.0[c] = (((v - s) / (1.0 - s)).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        dried_owned = img;
        &dried_owned
    } else {
        input
    };
    // A composition layer paints ONTO the accumulated canvas (occlusion); a standalone painting starts fresh.
    // BODY/opacity is a medium property — transparent media (watercolour/ink) let the ground glow through.
    let mut canvas = base.unwrap_or_else(|| {
        match p.ground {
            Some(tone) => Canvas::toned(w, h, p.palette, tone, 0.85),
            None => Canvas::white(w, h, p.palette, 0.85),
        }
        .with_opacity(p.opacity)
    });
    // PRIME this layer's footprint back to the ground so the element paints fresh and OCCLUDES what's beneath
    // (the concentration-ratio colour model can't be covered by a thin layer otherwise).
    if let Some(mask) = &p.paint_mask {
        canvas.clear_mask(mask);
    }
    let n = p.palette.pigments.len();
    lap("setup(armature+canvas)", &mut prof_acc, &mut prof_t);
    // Mixture cache keyed on the quantised reference colour — thousands of strokes sample similar colours.
    // (A tile worker keeps its own; the solve is deterministic, so a cache never changes a mark.)
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();

    let mut score = StrokeScore {
        header: ScoreHeader { version: 1, palette: p.palette.name.to_string(), pigments: p.palette.pigments.iter().map(|pg| (pg.name.to_string(), pg.masstone)).collect(), medium: p.medium.clone(), seed: p.seed, width: w, height: h, tooth: 0.85, ground: p.ground, brush: p.brush, bleed: p.bleed, diffuse: p.diffuse, stages: None, dry: p.dry, opacity: p.opacity, impasto: p.impasto, chroma: p.chroma, dry_shift: p.dry_shift, granulate: p.granulate, sheen: p.sheen, edge_pool: p.edge_pool, paper_edge: p.paper_edge, contrast: p.contrast, warmth: p.warmth, clarity: p.clarity, lift: p.lift },
        strokes: Vec::new(),
    };

    // The passes to run: an explicit plan (P1.3), else coarse→fine from brush_sizes (P0).
    let sizes: Vec<f32> = p.brush_sizes.iter().copied().filter(|&r| r >= p.min_brush).collect();
    let passes: Vec<PassSpec> = p.passes.clone().unwrap_or_else(|| {
        sizes.iter().enumerate().map(|(i, &r)| PassSpec { face_only: p.face_ladder_from.is_some_and(|n| i >= n), radius: r, budget: p.budget, stage: if i == 0 { "block-in".into() } else { format!("restate-{i}") } }).collect()
    });
    // FROM SCRATCH: the minimum brush of every plane (RFC §9). A rung below the focal floor touches nothing, so
    // it is not a pass of this painting at all.
    let plane_floor: Option<Vec<f32>> = p.from_scratch.then(|| plane_floor_field(w, h, p.min_brush, (p.armature_side.unwrap_or(NEW_BACKGROUND_SIDE), p.armature_body_side, p.armature_face_side), p.subject_mask.as_deref(), p.face_mask.as_deref(), p.hair_mask.as_deref()));
    let passes: Vec<PassSpec> = match &plane_floor {
        Some(fl) => {
            let finest = fl.iter().copied().fold(f32::INFINITY, f32::min);
            passes.into_iter().filter(|q| q.radius.max(p.min_brush) >= finest).collect()
        }
        None => passes,
    };
    // The schedule goes into the score so a replay crosses every pass boundary the paint crossed — a pass that
    // lays no stroke included. Density media run no passes (the drawing is laid by `ink_drawing`).
    // A luminous medium: the washes (one stage per level), then the two finest brush passes at a fraction of
    // the budget (the modelling within the washes).
    let luminous_passes: Vec<PassSpec> = if p.luminous && !p.book {
        // The modelling passes keep the plan's budget for the fine brushes: the restate gate already confines
        // them to where the washes differ from the picture (a lit window, a machine, a face), so on a flat
        // wash they lay little and on a detailed passage they resolve it — a quarter share washed the detail
        // out of every picture with fine structure (a night scene's lamps and figures).
        // Only the FINE brushes (≤ ~14 px) model within the washes: a mid-scale pass over them re-painted the
        // masses' own edges and lost detail (measured on a night scene: 0.096 → 0.086 Laplacian σ).
        let fine: Vec<PassSpec> = passes.iter().filter(|q| q.radius.max(p.min_brush) <= 14.5).cloned().collect();
        // A planes-and-line face (see `face_is_planes`) takes no fine ladder: its planes are washes and its
        // likeness is a line, and brush modelling over them is the oil sketch the user rejected.
        fine.into_iter().filter(|q| !(q.face_only && face_is_planes(p))).map(|q| PassSpec { budget: ((q.budget as f32) * 0.7) as usize, ..q }).collect()
    } else {
        Vec::new()
    };
    // The face's own wash stages, after the sheet's (see `WashScope::Inside`), on the planes' shape.
    let face_scope: Option<Vec<bool>> = if face_is_planes(p) { face_planes_scope(source, p) } else { None };
    let face_wash_levels = if face_scope.is_some() { p.armature_levels.max(2) as usize } else { 0 };
    let n_stages_total = p.armature_levels.max(2) as usize + face_wash_levels + luminous_passes.len();
    score.header.stages = Some(if p.density {
        Vec::new()
    } else if p.luminous {
        (1..=p.armature_levels.max(2)).map(|l| format!("wash-{l}")).chain((1..=face_wash_levels).map(|l| format!("face-wash-{l}"))).chain(luminous_passes.iter().map(|q| q.stage.clone())).collect()
    } else {
        passes.iter().map(|q| q.stage.clone()).collect()
    });

    let mut placed = 0usize;
    let mut k = 0u64;
    let mut rejected: Vec<String> = Vec::new();
    // EDGE-HARDNESS field (edge-control craft): computed once from the reference so every pass terminates
    // strokes at the same hard boundaries — the masses meet crisply where value contrast is high, and cross
    // freely (soft/lost) elsewhere. Off for the loose Impressionist register and for density media.
    let hardness = (p.style != PaintStyle::Impressionist && p.define > 0.0 && !p.density).then(|| edge_hardness(input, p.define));
    let hard_ref = hardness.as_ref().map(|(m, t)| (m.as_slice(), *t));
    // SALIENCY field for the opt-in density gate — computed once from the reference (§ saliency). None = off.
    let saliency = (p.saliency > 0.0).then(|| saliency_field(input));
    // SHADOW field for COMMIT-SHADOWS: the dark value masses (1 = deep shadow → 0 = light), so the shadow family
    // paints decisively (see the stroke loop). The shadow/light boundary is DERIVED FROM THIS IMAGE'S OWN value
    // distribution (percentiles) — a fact — so it adapts to any image (dark, light, high-key, low-key) instead of
    // a hardcoded threshold. None = off.
    let shadow = (p.commit_shadows > 0.0).then(|| {
        let luma = luma_map(input);
        let mut sorted = luma.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len().max(1);
        let lo = sorted[n * 12 / 100]; // deep-shadow anchor
        let hi = sorted[n * 55 / 100]; // shadow→light boundary (this image's lower-mid value)
        let span = (hi - lo).max(1e-3);
        luma.iter().map(|&l| ((hi - l) / span).clamp(0.0, 1.0)).collect::<Vec<f32>>()
    });
    // RESERVE as a PROTECT (RFC watercolour "reserve-plan"): reserved paper stays paper. The seed gate in the
    // stroke loop refuses to START a stroke on a reserved cell, but a stroke seeded beside one still GROWS into
    // it, so any change to the deposit/colour physics bleeds pigment into the reserve. Terminate stroke growth at
    // reserved cells instead — a fact-driven protect, and replay-exact (only the recorded path is shorter).
    // The rule mirrors the seed gate exactly (bright, not a committed shadow, not inside the subject); merged
    // with the negative-painting protect when one is set.
    let protect_all: Option<Vec<bool>> = match p.reserve {
        Some(rt) => {
            let mut m: Vec<bool> = input
                .pixels()
                .enumerate()
                .map(|(i, px)| {
                    let l = color::linear_luma(color::srgb_to_linear(px.0));
                    let sh = shadow.as_ref().map(|s| s[i] * p.commit_shadows).unwrap_or(0.0);
                    let subj = p.subject_mask.as_ref().map(|s| s[i]).unwrap_or(0.0);
                    l > rt && sh < 0.35 && subj < 0.5
                })
                .collect();
            // SHAPE-AWARE, for a new painting. A per-pixel threshold reserves every bright pixel wherever it
            // falls — a fleck on a lamp, a glint in hair, the bead on a window frame — and the paper comes out
            // as scattered white holes, growing with every push of the threshold. A watercolourist reserves
            // SHAPES: the few large, simple, light areas the picture is built around, and never the face. So
            // the candidate is opened (ragged edges off), split into connected regions, and only regions at
            // least a thirty-second of the sheet across are kept.
            if p.from_scratch {
                let (wu, hu) = (w as usize, h as usize);
                let short = w.min(h) as usize;
                if let Some(fm) = p.face_mask.as_deref() {
                    for (a, &f) in m.iter_mut().zip(fm) {
                        if f > 0.3 {
                            *a = false;
                        }
                    }
                }
                let r = (short / 400).max(2);
                let opened = dilate_bool(&erode_bool(&m, wu, hu, r), wu, hu, r);
                // A sixty-fourth of the sheet across — a lit window pane is a shape worth reserving; the
                // opening above has already removed anything thinner than a few pixels.
                let min_side = (short / 64).max(6);
                m = keep_large_regions(&opened, wu, hu, min_side * min_side);
                // …and the face's own glints (see `face_paper_shapes`), which every pass then honours.
                if let Some(fs) = &face_scope {
                    for (a, b) in m.iter_mut().zip(face_paper_shapes(source, fs, rt)) {
                        *a |= b;
                    }
                }
            }
            if let Some(pm) = p.protect.as_deref() {
                for (a, &b) in m.iter_mut().zip(pm) {
                    *a |= b;
                }
            }
            Some(m)
        }
        None => p.protect.clone(),
    };
    // FOCAL field for the opt-in selective-detail gate. A detected FACE MASK (`--preserve-face`) takes priority
    // over the centre-prior focal field (`--focus-detail`), since the real face box targets the face however it
    // is placed. `focal_strength` opens the region for whichever source is active.
    // A focal source only ENGAGES when its strength is > 0. A face mask with `preserve_face == 0` (the default,
    // e.g. from `--plan auto`, which builds the mask for the silhouette but does not ask for selective face
    // detail) must NOT switch the restrictive focal gate on: with strength 0 the gate `f < 1 - strength` skips
    // every detail stroke outside the exact face core — the bug that erased detail across the whole canvas. When
    // no source asks for focus, `focal` stays None and the gate below is a no-op, so detail paints everywhere.
    let (focal, focal_strength) = if let (Some(fm), true) = (&p.face_mask, p.preserve_face > 0.0) {
        (Some(std::borrow::Cow::Borrowed(fm)), p.preserve_face)
    } else if p.focus_detail > 0.0 {
        (Some(std::borrow::Cow::Owned(focal_field(input))), p.focus_detail)
    } else {
        (None, 0.0)
    };
    let focus_on = focal.is_some();
    // GRADATION in the PAINTING itself: a soft mass is not only simplified by the armature — a wide stroke lays
    // ONE colour along its whole path (the reference may drift a full `STOP_TOL` before it stops), and the
    // restate floor then leaves a patch that is within tolerance of a slow gradient alone. So a cloud, a
    // soft-lit wall, still water end as a few flat patches meeting at contour edges even from a smooth
    // reference. Where the picture is a ramp (see `ramp_field`), `gradation` makes strokes stop sooner and the
    // restate floor drop, so the gradient is laid as graded marks and the fine passes restate it. 0 = as before.
    let ramp_soft: Option<Vec<f32>> = (p.gradation > 0.0 && !p.density).then(|| {
        let side = p.armature_side.unwrap_or(150).max(1);
        let r = ((w.min(h) as f32 / side as f32).round() as usize).clamp(2, 16);
        // Never on the SUBJECT: skin is a ramp by nature, and easing it smeared the faces (the user's regression).
        // The face mask and the matte say where the subject is; without either, everywhere counts.
        let at = |m: Option<&[f32]>, i: usize| m.and_then(|m| m.get(i).copied()).unwrap_or(0.0);
        ramp_field(input, r, p.armature_levels.max(2)).into_iter().enumerate().map(|(i, v)| v * p.gradation * (1.0 - at(p.face_mask.as_deref(), i).max(at(p.subject_mask.as_deref(), i)))).collect()
    });

    lap("fields(hardness/shadow/protect/head)", &mut prof_acc, &mut prof_t);
    // The HEAD (for the texture rule): the detected face box grown to take in hair, beard and neck — a face box
    // is tight, and the residual on a beard is scrawl just as it is on an eye socket. Grown by a fraction of the
    // short side (a head's margin is a fact of the sheet, not of the scene).
    let head_mask: Option<Vec<f32>> = p.face_mask.as_ref().map(|fm| {
        let g = image::GrayImage::from_fn(w, h, |x, y| image::Luma([(fm[(y * w + x) as usize] * 255.0).clamp(0.0, 255.0) as u8]));
        let sigma = (w.min(h) as f32 / 40.0).max(2.0);
        imageops::blur(&g, sigma).pixels().map(|px| (px.0[0] as f32 / 255.0 * 6.0).clamp(0.0, 1.0)).collect()
    });
    // DENSITY media (pen-and-ink) do not paint masses with a brush ladder: they DRAW the composition — the
    // structure's contour lines first, then TONE by hatching — and leave the paper everywhere else.
    let passes: Vec<PassSpec> = if p.density {
        ink_drawing(&mut canvas, &mut score, input, source, head_mask.as_deref(), p, protect_all.as_deref(), &mut placed, &mut k, progress);
        Vec::new()
    } else if p.luminous {
        // A LUMINOUS medium lays its masses as WASHES — area fills, level by level, light to dark, each level
        // dried before the next (see `wash_passes`) — then MODELS within them with the two finest brushes only
        // (thin transparent touches from the source's texture: a wash is flat, a watercolour is not), at a
        // fraction of the budget. The line and the splatter follow.
        match face_scope.as_deref() {
            Some(fs) => {
                // The sheet around the face, then the face as planes on its own value range.
                let sheet_levels = p.armature_levels.max(2) as usize;
                wash_passes_scoped(&mut canvas, &mut score, input, source, n_stages_total, 0, WashScope::Outside(fs), p, protect_all.as_deref(), &mut placed, &mut k, progress);
                wash_passes_scoped(&mut canvas, &mut score, input, source, n_stages_total, sheet_levels, WashScope::Inside(fs), p, protect_all.as_deref(), &mut placed, &mut k, progress);
            }
            None => wash_passes(&mut canvas, &mut score, input, source, n_stages_total, p, protect_all.as_deref(), &mut placed, &mut k, progress),
        }
        luminous_passes
    } else {
        passes
    };
    for (layer, pass) in passes.iter().enumerate() {
        let radius = pass.radius.max(p.min_brush);
        let pass_t0 = std::time::Instant::now();
        // The first pass is a block-in: it covers the whole canvas so no white ground survives. Later passes
        // only restate where the canvas is still wrong.
        let block_in = layer == 0;
        if placed >= p.budget {
            break;
        }
        let mut in_pass = 0usize;
        // Critic: snapshot the canvas + score BEFORE the pass, so a pass that hurts can be rolled back.
        let snapshot = critic.map(|c| (canvas.clone(), score.strokes.len(), placed, c(&canvas.to_image())));
        // DETAIL passes are the fine end of the schedule. Coarse passes lay soft masses from a BLURRED
        // reference; detail passes must RESOLVE features — the face, the net, branches, windows — so they read a
        // SHARPENED reference (unsharp-mask), place crisp short strokes on the high-frequency structure the soft
        // canvas is still missing, and use a drier brush that doesn't muddy them back into the masses.
        // DETAIL = the FINE brushes only (an ABSOLUTE size, not a fraction of the coarsest). A wide brush must
        // never be a "detail" brush laying short, contrasty dabs — that is the wide-brush "splatter". Wide passes
        // stay masses/filbert with long, form-following strokes; only the genuinely small brushes resolve detail.
        // Impressionist keeps every pass soft (no detail tier). HIGH-FIDELITY treats every non-block-in pass as
        // detail (tight tracking of the sharp reference); Legible reserves detail for the genuinely fine brushes.
        let fidelity = p.style == PaintStyle::Fidelity;
        // A fine brush (absolute size). SELECTIVE DETAIL (`focus_detail`) turns the fine passes into detail passes
        // even under a loose base register — but their strokes are gated to the FOCAL region below, so the masses
        // stay loose and only the eye's destination (eyes/glasses) gets crisp accents.
        let fine = radius <= p.min_brush * 2.5;
        let detail = !block_in && (fidelity || (p.style == PaintStyle::Legible && fine) || (focus_on && fine));
        // The reference this pass paints from. Fidelity paints from a SHARP reference at every scale (it tracks
        // real structure, doesn't invent) — a touch of unsharp even on the block-in. Legible blurs the masses and
        // sharpens only for detail.
        let reference = if fidelity {
            imageops::unsharpen(input, (radius * 0.22).max(0.5), 1)
        } else if detail {
            // Fine layers read the armature, plus the subject's own texture inside the flat masses (see
            // `detail_texture`), unless a detail sharpen is asked for instead (see `detail_sharpen`).
            if p.detail_sharpen > 0.0 {
                imageops::unsharpen(input, (radius * p.detail_sharpen).max(0.6), 1)
            } else if p.detail_texture > 0.0 && !std::ptr::eq(source, input) {
                // (The luminous media too: their modelling passes read the armature, which the washes already
                // match, so without the source's texture the restate gate found nothing to resolve — a night
                // scene's lamps and machine washed out. The residual greyed only the old cumulative mud washes.)
                texture_reference(input, source, radius, p.armature_levels.max(2), p.detail_texture, head_mask.as_deref().or(p.subject_mask.as_deref()))
            } else {
                input.clone()
            }
        } else {
            // Coarse passes lay masses from a softened reference — but a radius×0.5 blur erases the structure
            // (object edges, value boundaries) before a stroke is placed, so strokes see no boundary to stop at
            // and smear. A gentler blur keeps the masses' EDGES while still dropping texture.
            imageops::blur(input, (radius * 0.32).max(0.6))
        };
        lap("pass:reference", &mut prof_acc, &mut prof_t);
        let luma = luma_map(&reference);
        // Coherent flow (structure tensor). Fidelity + detail keep it TIGHT (small sigma) so strokes hug local
        // edges; coarse legible passes smooth it so masses follow gross form.
        let sigma = if detail || fidelity { 2 } else { (radius * 0.9).round().clamp(2.0, 24.0) as i32 };
        let (gx, gy) = coherent_gradient(&luma, w, h, sigma, p.infill);
        // CROSS-HATCH: rotate this pass's flow by the medium's hatch angle × layer, so successive restatements
        // cross the form at the classic angles instead of all lying along it (tempera's woven net).
        let (gx, gy) = if p.hatch_angle.abs() > 1e-3 && !block_in {
            let (sn, cs) = (p.hatch_angle * layer as f32).sin_cos();
            (gx.iter().zip(&gy).map(|(x, y)| x * cs - y * sn).collect::<Vec<f32>>(), gx.iter().zip(&gy).map(|(x, y)| x * sn + y * cs).collect::<Vec<f32>>())
        } else {
            (gx, gy)
        };
        // HAIR grows its own way. Everywhere else the direction is the armature's isophote — the shape of the
        // value mass — which is why the strand tool combed a mane along its lighting rather than along its
        // hair. Where the mask says hair AND the field is coherent, rotate the flow onto the growth direction.
        // The pass's own MAGNITUDE is kept: it feeds the detail gate, and swapping it here would silently
        // change which marks are allowed, not just which way they point.
        let (gx, gy) = match (&hair_flow, p.hair_mask.as_deref()) {
            (Some(hf), Some(hm)) => {
                let (hx, hy, hc) = (&hf.dx, &hf.dy, &hf.coherence);
                let (mut gx, mut gy) = (gx, gy);
                for i in 0..gx.len() {
                    let t = hm[i].clamp(0.0, 1.0) * hc[i];
                    if t <= 1e-3 {
                        continue;
                    }
                    let m = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
                    // A growth direction is an ORIENTATION, not an arrow: flip it into the same half-plane as
                    // the pass flow first, or blending a strand at +80° with one at −80° cancels to nothing.
                    let (ux, uy) = (-hy[i], hx[i]); // the "gradient" whose perpendicular is the strand
                    let (ux, uy) = if ux * gx[i] + uy * gy[i] < 0.0 { (-ux, -uy) } else { (ux, uy) };
                    gx[i] += (ux * m - gx[i]) * t;
                    gy[i] += (uy * m - gy[i]) * t;
                }
                (gx, gy)
            }
            _ => (gx, gy),
        };
        // BRUSH for this pass: the composition layer's `layer_brush` if set, else the intelligent per-role
        // default. HIGH-FIDELITY uses a clean, low-waver, short-tracking brush at every pass so strokes lie down
        // as disciplined marks that follow the reference — not the loose, wavering, streaky invention.
        let profile = if let Some(lb) = p.layer_brush {
            lb
        } else if fidelity {
            // Longer, cleaner strokes than a dab — short marks read as patchy speckle even in fidelity.
            BrushProfile { radius_scale: 0.9, len: if block_in { 1.2 } else { 0.85 }, streak: 0.10, round: 0.92, waver: 0.03 }
        } else {
            let role = if block_in { "flat" } else if detail { "round" } else { "filbert" };
            let mut pr = BrushProfile::named(role);
            if block_in {
                // COVERAGE: the raked flat block-in (streak 0.70) leaves the white ground showing between marks —
                // the speckly grain. Blend it toward a smooth, gap-free cover so the base actually HIDES the
                // ground; the restatement then modulates a covered field instead of filling holes in bare paper.
                let c = p.coverage.clamp(0.0, 1.0);
                pr.streak = pr.streak * (1.0 - c) + 0.10 * c;
                pr.round = pr.round * (1.0 - c) + 0.85 * c;
            }
            pr
        };
        let mut pass_brush = p.brush;
        pass_brush.streak = profile.streak;
        pass_brush.round = profile.round;
        // DEPTH ORDER of the layers: the first, widest pass COVERS with long strokes; each thinner layer laid on top
        // uses SHORTER ones, tapering (by brush radius) to the plan's `detail_len` on the finest — wide-and-long to
        // cover, then a few layers of thinner-and-shorter to resolve. Shortening every layer alike starved the
        // block-in (white specks of ground between dabs); long strokes on the fine layers smeared the features.
        let (r_coarse, r_fine) = (
            passes.first().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius),
            passes.last().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius),
        );
        let depth_t = if r_coarse > r_fine + 1e-6 { ((r_coarse - radius) / (r_coarse - r_fine)).clamp(0.0, 1.0) } else { 0.0 };
        let len_mul = profile.len * (1.0 + (p.detail_len - 1.0) * depth_t);
        let b_waver = profile.waver;
        lap("pass:flow", &mut prof_acc, &mut prof_t);
        let grid = (radius * 0.9).max(1.5);

        let cols = ((w as f32) / grid).ceil() as u32;
        let rows = ((h as f32) / grid).ceil() as u32;
        // Visit the pass's seed cells in a SCRAMBLED order (a deterministic hashed permutation), not row by row.
        // When the budget runs out mid-pass, row order left the bottom of every picture untouched by that pass —
        // measured: the 2px pass changed nothing below 55% of the height, so every face, figure or detail in the
        // lower half was painted without the fine layers. Scrambled, a cap thins the pass uniformly. Replay-exact.
        let n_cells = (rows as usize) * (cols as usize);
        let order_seed = p.seed ^ (layer as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // ONE STROKE ATTEMPT at seed cell `cell` with jitter key `k`, laid onto `cv` and returned as its record
        // (id unset), or None when the seed was skipped.
        // The restate floor's multiplier: 1 for the pass proper; the FILL rounds (below) halve it round by round.
        let floor_mul = std::cell::Cell::new(1.0f32);
        let attempt = |cell: u32, k: u64, cv: &mut Canvas, cache: &mut std::collections::HashMap<u32, Vec<f32>>| -> Option<StrokeRecord> {
            let (gyi, gxi) = (cell / cols, cell % cols);
            {
                let jx = jitter(p.seed, k) * grid;
                let jy = jitter(p.seed, k.wrapping_add(1)) * grid;
                let cx = (gxi as f32 + 0.5) * grid + jx;
                let cy = (gyi as f32 + 0.5) * grid + jy;
                if cx < 0.0 || cy < 0.0 || cx >= w as f32 || cy >= h as f32 {
                    return None;
                }
                let (ix, iy) = (cx as u32, cy as u32);
                // FROM SCRATCH: this plane's minimum brush — a finer rung does not touch it (RFC §9).
                if let Some(fl) = &plane_floor {
                    if radius < fl[iy as usize * w as usize + ix as usize] {
                        return None;
                    }
                }
                // COMPOSITION LAYER: only seed inside this element's footprint (so it paints its own region and
                // leaves the rest of the accumulated canvas untouched).
                if let Some(mask) = &p.paint_mask {
                    if !mask.get(iy as usize * w as usize + ix as usize).copied().unwrap_or(false) {
                        return None;
                    }
                }
                // NEGATIVE PAINTING: never seed a stroke inside the protected shape — paint around it.
                if let Some(mask) = &p.protect {
                    if mask.get(iy as usize * w as usize + ix as usize).copied().unwrap_or(false) {
                        return None;
                    }
                }
                let target = reference.get_pixel(ix, iy).0;
                let tluma = color::linear_luma(color::srgb_to_linear(target));
                // COMMIT SHADOWS (RFC §3.3): in the dark value masses, paint DECISIVELY — lift the reserve so darks
                // always land, drop the restate floor so they build to full depth, and boost the pigment charge so
                // they read as committed paint, not a thin wash. `sh` in [0,1] is the shadow strength here.
                let region_i = iy as usize * w as usize + ix as usize;
                // A face-only pass lays nothing off the face.
                if pass.face_only && p.face_mask.as_deref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) < 0.35 {
                    return None;
                }
                let soft = ramp_soft.as_ref().map(|m| m[region_i]).unwrap_or(0.0);
                let sh = shadow.as_ref().map(|s| s[region_i] * p.commit_shadows).unwrap_or(0.0);
                // RESERVE is a fact about the BACKGROUND, not the subject: bright cells keep the paper only OUTSIDE
                // the detected subject (matte). Inside the subject a light shirt / shoulders / skin is PAINTED as a
                // light mass — never reserved to blank paper (that is what made the shoulders vanish). And never
                // reserve inside a committed shadow. Determined over the detected extents, not raw luma.
                let subj = p.subject_mask.as_ref().map(|m| m[region_i]).unwrap_or(0.0);
                if let Some(rt) = p.reserve {
                    if tluma > rt && sh < 0.35 && subj < 0.5 {
                        return None;
                    }
                }
                // Later layers only restate where the canvas is still notably wrong; the block-in covers all.
                // Detail passes use a lower threshold so fine features (which the soft masses missed) still land.
                // Committed shadows drop the floor toward zero so the darks deepen pass over pass. The SUBJECT
                // also gets a tighter floor so it BUILDS DENSITY (layers) instead of being covered once and
                // skipped — the fix for a sparse, under-painted subject; the background stays sparse.
                let restate_floor = (if detail { p.detail_restate } else { 0.06 }) * (1.0 - 0.85 * sh) * (1.0 - 0.55 * subj) * (1.0 - 0.8 * soft) * floor_mul.get();
                if !block_in && !p.density && rgb_dist(cv.color_at(ix, iy), target) < restate_floor {
                    return None;
                }
                // Detail passes only add marks where there is COHERENT structure to resolve. On an incoherent,
                // structureless region (e.g. a tangled net the armature rendered as noise) the flow field has no
                // dominant direction — dropping detail marks there reads as random blocky specks, an "AI filter"
                // tell. Skip them; leave that area as the smooth mass the coarse passes laid.
                // (High-fidelity tracks EVERYTHING closely, so it doesn't gate on coherence — it restates flat
                // areas too. The gate is a Legible anti-speckle measure.)
                if detail && !fidelity {
                    let seed_i = iy as usize * w as usize + ix as usize;
                    let coh = (gx[seed_i] * gx[seed_i] + gy[seed_i] * gy[seed_i]).sqrt();
                    // SUBJECT-AWARE bar: the subject keeps its detail (a smooth-lit face reads flat but still wants
                    // modelling), while a structureless BACKGROUND passage (open sky, a flat wall) is held far
                    // stricter so stray detail marks don't speckle it — the sky-speckle tell.
                    let thresh = p.detail_coherence * (1.0 + 1.6 * (1.0 - subj));
                    if coh < thresh {
                        return None;
                    }
                }
                // SELECTIVE DETAIL (`focus_detail`): the crisp detail tier fires ONLY inside the focal region —
                // the compact, central, high-contrast area the eye goes to (eyes/glasses) — so the rest of the
                // painting keeps the loose wash the coarse passes laid. `focus_detail` opens the region (1 = the
                // whole canvas, small = only the very focus). The masses (non-detail passes) are never gated.
                if let (Some(foc), true) = (&focal, detail) {
                    let f = foc[iy as usize * w as usize + ix as usize];
                    if f < 1.0 - focal_strength {
                        return None;
                    }
                }
                // SALIENCY-GATED DENSITY (opt-in): thin the restating/detail passes in flat, low-structure
                // passages so the background stays a calm smooth mass while the focal subject keeps full
                // density. The block-in is never gated (the canvas must be covered). Deterministic skip, and
                // replay-exact since only the strokes that ARE placed get recorded into the score.
                if let Some(sal) = &saliency {
                    if !block_in {
                        let s = sal[iy as usize * w as usize + ix as usize];
                        let keep = (1.0 - p.saliency) + p.saliency * s;
                        if jitter(p.seed, k.wrapping_mul(0x1000_0001).wrapping_add(0x5EED)) + 0.5 > keep {
                            return None;
                        }
                    }
                }
                // BROKEN COLOUR: vary this stroke's colour so neighbours optically mix (vibrancy).
                let load_target = if p.broken > 0.0 { broken_color(target, p.broken, p.seed, k) } else { target };
                // Committed shadows carry MORE pigment so the dark masses read solid, not a thin transparent wash.
                let load = mixture_cached(cache, load_target, &p.palette, p.charge * (1.0 + 1.1 * sh), n);
                // Stroke-growth boundary: a COMPOSITION layer keeps its strokes inside the element's footprint
                // (they terminate at the mask edge, so the element doesn't bleed over its neighbours); otherwise
                // the focal hard-edge region_mask keeps a single subject crisp against the ground.
                // HAIR / FUR: how much this mark is a strand rather than a mass (see `PaintParams::hair_mask`).
                let hair = p.hair_mask.as_deref().and_then(|m| m.get(region_i).copied()).unwrap_or(0.0).clamp(0.0, 1.0);
                // SILHOUETTE STRANDS: hair reads as hair at its EDGE — a few strands escape the mass — but the
                // subject silhouette is a seam that strokes TERMINATE at, which is exactly wrong for a mane and
                // left every head and beard with a soft rounded outline. A MINORITY of hair marks (about a
                // third, chosen by the stroke's own hash so replay is exact) may cross it; they are narrow and
                // taper to a point, so what escapes is a filament, not a bulge. Letting them all cross turned
                // the outline into a fuzzy blob — the discipline is that most strands still stop at the seam.
                let strand_out = hair > 0.35 && jitter(p.seed ^ 0x9E37_79B9, k.wrapping_add(11)) + 0.5 < 0.35;
                let region = if let Some(m) = p.paint_mask.as_deref() {
                    Some((m, true))
                } else if strand_out {
                    None
                } else {
                    p.region_mask.as_deref().map(|m| (m, m.get(iy as usize * w as usize + ix as usize).copied().unwrap_or(false)))
                };
                // Per-stroke VARIATION (organic hand): a hand-painted mark varies from its neighbour — but only a
                // LITTLE. Loose (±40/50%) variation makes the marks read as PATCHY, inconsistent dabs; tight
                // variation keeps deliberate, controlled brushwork. `stroke_width`/`stroke_len` are the author's
                // global dials on mark size and length (a painter picks the brush and the gesture).
                let wvar = 1.0 + 0.15 * jitter(p.seed ^ 0x5B57, k);
                let lvar = 1.0 + 0.18 * jitter(p.seed ^ 0x91E3, k.wrapping_add(7));
                let pvar = 0.85 + 0.15 * (jitter(p.seed ^ 0x2C7D, k.wrapping_add(3)) + 0.5);
                // A strand is LONGER and NARROWER than a mass mark: a lock of hair is drawn in one gesture.
                let rw = (radius * profile.radius_scale * p.stroke_width * wvar * (1.0 - 0.35 * hair)).max(p.min_brush * 0.8);
                let path = grow_path(cx, cy, radius, &gx, &gy, &reference, target, protect_all.as_deref(), region, hard_ref, (len_mul * p.stroke_len * lvar * (1.0 + 1.1 * hair)).max(0.2), STOP_TOL * (1.0 - 0.6 * soft));
                // WAVER: a real hand doesn't draw a ruler-straight line — displace the path with a little smooth
                // wobble (a characteristic, not an error). Applied to the recorded path, so replay is exact.
                let path = waver_path(&path, b_waver * rw, p.seed, k);
                // Laid less wet (0.7, jittered) so strokes sit ON the canvas rather than dissolving into the wet
                // paint beneath — distinct marks, not a smear. Some wetness remains for light harmonisation.
                // DETAIL marks are laid DRIER still: a fine accent that resolves a feature (an eye, a window, a
                // figure) must DEFINE it, not dissolve into the mass beneath — so detail passes go on nearly dry.
                let wet = {
                    let base = (0.55 + 0.3 * (jitter(p.seed ^ 0x77A1, k.wrapping_add(5)) + 0.5)).clamp(0.4, 0.85);
                    if p.luminous && !detail {
                        // A watercolour's modelling touches are laid WET into the wash where the picture is a
                        // MASS, so they bloom — and DRY where it is an EDGE (the edge-hardness field), so a
                        // machine's parts, a window frame, a figure's outline sit crisp instead of dissolving
                        // into each other (a night scene's whole locomotive melted). The finest touches go on
                        // drier still (the accents that make the picture read).
                        let hard_here = hard_ref.map(|(m, t)| (m.get(region_i).copied().unwrap_or(0.0) / t.max(1e-4)).clamp(0.0, 1.0)).unwrap_or(0.0);
                        (base * (1.0 - 0.35 * hard_here) + 0.25 * (1.0 - hard_here)).min(0.95)
                    } else if detail {
                        base * 0.5
                    } else {
                        base
                    }
                };
                // Per-stroke brush character (from the pass profile, lightly jittered) — a hand-loaded brush is
                // never identical stroke to stroke.
                let s_streak = (pass_brush.streak + 0.15 * jitter(p.seed ^ 0x3F5B, k.wrapping_add(2))).clamp(0.0, 1.0);
                let s_round = (pass_brush.round + 0.12 * jitter(p.seed ^ 0x8A21, k.wrapping_add(4))).clamp(0.0, 1.0);
                let mut stroke_brush = pass_brush;
                // A DETAIL mark barely picks up the wet paint under it — otherwise a fine accent just smears the
                // mass back over itself and the feature never resolves (why thin brushes alone don't add detail).
                if detail {
                    stroke_brush.k_pickup *= 0.2;
                }
                stroke_brush.streak = s_streak;
                stroke_brush.round = s_round;
                // THE HAIR TOOL. A stroke is already a bundle of bristle LANES, each carrying its own load, and
                // `streak` is how unevenly they are loaded — so a raked, many-laned, narrow mark already draws a
                // LOCK of hair. What it needed was to stop behaving like a mass mark: more lanes than the width
                // alone would give, a rakier streak so the lanes read as separate strands, and almost no pickup,
                // because a strand that smears the wet paint under it becomes the mud a mane kept coming out as.
                let s_streak = if hair > 0.0 { (s_streak + 0.4 * hair).clamp(0.0, 1.0) } else { s_streak };
                if hair > 0.0 {
                    stroke_brush.streak = s_streak;
                    stroke_brush.bristles = ((stroke_brush.bristles as f32) * (1.0 + 1.4 * hair)).round().clamp(1.0, 256.0) as usize;
                    stroke_brush.k_pickup *= 1.0 - 0.85 * hair;
                }
                // THE TECHNIQUE, as what the pigment does on this stroke. Wet-on-wet lands flooded. Wet-on-dry
                // lands as a loaded wash, moderately wet, and relies on drying between layers. Dry-on-dry
                // carries little pigment and goes down raked, so the bristles skip and the paper's tooth
                // breaks the mark — the dry-brush drag.
                let (wet, load, s_streak) = match p.technique {
                    WetTechnique::WetOnWet => (wet.max(0.9), load, s_streak),
                    WetTechnique::WetOnDry => (wet.clamp(0.45, 0.7), load, s_streak),
                    WetTechnique::DryOnDry => {
                        let thin: Vec<f32> = load.iter().map(|v| v * 0.55).collect();
                        (wet.min(0.15), thin, s_streak.max(0.8))
                    }
                    WetTechnique::None => (wet, load, s_streak),
                };
                if p.technique != WetTechnique::None {
                    stroke_brush.streak = s_streak;
                    if p.technique == WetTechnique::DryOnDry {
                        stroke_brush.k_pickup = 0.0;
                    }
                }
                // A strand ends in a POINT (a hair has a tip); a mass mark lifts off at about half its width.
                let s = Stroke { path, width0: rw, width1: (rw * (0.55 - 0.42 * hair)).max(p.min_brush * 0.5 * (1.0 - 0.6 * hair)), load, pressure: pvar.clamp(0.4, 1.0), wetness: wet };
                s.rasterize(cv, &stroke_brush);
                // Record the stroke into the score (mix as pigment name → value, for the non-zero pigments).
                let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                Some(StrokeRecord {
                    id: 0,
                    wipe: false, wash: false,
                    stage: pass.stage.clone(),
                    spline: s.path,
                    w0: s.width0,
                    w1: s.width1,
                    taper: 0.4 + 0.45 * hair,
                    mix,
                    wet: s.wetness,
                    press: s.pressure,
                    streak: s_streak,
                    round: s_round,
                    // A detail accent's near-clean pickup is part of how it was laid — record it so replay is exact.
                    // A pickup the stroke's own brush differs in from the header's is part of how it was laid.
                    // The dry brush zeroes it; left unrecorded, replay rebuilt every dry-brush mark with the
                    // header's pickup and drifted — the unrecorded-bristles bug again.
                    pickup: if detail || hair > 0.0 || p.technique == WetTechnique::DryOnDry { Some(stroke_brush.k_pickup) } else { None },
                    bristles: (hair > 0.0).then_some(stroke_brush.bristles),
                })
            }
        };

        // CLASSIC ORDER: visit the pass's seed cells in a SCRAMBLED order (a deterministic hashed permutation),
        // not row by row. When the budget runs out mid-pass, row order left the bottom of every picture
        // untouched by that pass — measured: the 2px pass changed nothing below 55% of the height, so every
        // face, figure or detail in the lower half was painted without the fine layers. Scrambled, a cap thins
        // the pass uniformly. Replay-exact.
        let mut order: Vec<u32> = (0..n_cells as u32).collect();
        order.sort_by_key(|&c| jitter(order_seed, c as u64).to_bits());
        // MIXTURES UP FRONT, in parallel: a seed's target colour is known before any mark is laid (the reference
        // at the jittered seed, the broken-colour jitter by key), and a bucket's mixture is solved from its own
        // representative colour whoever asks, so the pass's NEW colours are solved here on every core and the
        // painter below only looks them up. Measured: the solver was 99% of a stroke's cost; this is where the
        // cores go, and it leaves the picture exactly as one painter in one order would lay it.
        {
            let mut keys: Vec<u32> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for (i, &cell) in order.iter().enumerate() {
                let (gyi, gxi) = (cell / cols, cell % cols);
                let kk = k + i as u64 + 1;
                let cx = (gxi as f32 + 0.5) * grid + jitter(p.seed, kk) * grid;
                let cy = (gyi as f32 + 0.5) * grid + jitter(p.seed, kk.wrapping_add(1)) * grid;
                if cx < 0.0 || cy < 0.0 || cx >= w as f32 || cy >= h as f32 {
                    continue;
                }
                if let Some(fl) = &plane_floor {
                    if radius < fl[cy as usize * w as usize + cx as usize] {
                        continue;
                    }
                }
                let target = reference.get_pixel(cx as u32, cy as u32).0;
                let t = if p.broken > 0.0 { broken_color(target, p.broken, p.seed, kk) } else { target };
                let key = mixture_key(t);
                if !cache.contains_key(&key) && seen.insert(key) {
                    keys.push(key);
                }
            }
            let threads = if p.threads == 0 { std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1) } else { p.threads };
            if let Some(pr) = progress {
                pr(PaintProgress::Mixing { pass: layer + 1, passes: passes.len(), radius, colours: keys.len(), threads });
            }
            if threads <= 1 || keys.len() < 64 {
                for key in &keys {
                    cache.insert(*key, mixture_for_key(*key, &p.palette, n));
                }
            } else {
                let chunk = keys.len().div_ceil(threads).max(1);
                let solved: Vec<Vec<(u32, Vec<f32>)>> = std::thread::scope(|sc| {
                    let handles: Vec<_> = keys.chunks(chunk).map(|ch| sc.spawn(move || ch.iter().map(|&key| (key, mixture_for_key(key, &p.palette, n))).collect::<Vec<_>>())).collect();
                    handles.into_iter().map(|hd| hd.join().expect("mixture solve thread")).collect()
                });
                for (key, v) in solved.into_iter().flatten() {
                    cache.insert(key, v);
                }
            }
            if prof_on { eprintln!("PROFILE   mixtures {} new keys solved up front ({threads} threads)", keys.len()); }
        }
        if let Some(pr) = progress {
            pr(PaintProgress::Painting { pass: layer + 1, passes: passes.len(), radius });
        }
        for cell in order {
            if placed >= p.budget || in_pass >= pass.budget {
                break;
            }
            k += 1;
            if let Some(mut rec) = attempt(cell, k, &mut canvas, &mut cache) {
                placed += 1;
                in_pass += 1;
                if let Some(pr) = progress {
                    if placed % 64 == 0 {
                        pr(PaintProgress::Placed(placed));
                    }
                }
                rec.id = placed as u32;
                score.strokes.push(rec);
            }
        }
        // FILL (see `PaintParams::fill`): the finest pass again, floors halving, until the share is spent.
        if p.fill > 0.0 && layer + 1 == passes.len() {
            let target = ((p.budget as f32) * p.fill.clamp(0.0, 1.0)) as usize;
            let mut round = 0usize;
            while placed < target && round < 6 {
                round += 1;
                floor_mul.set(0.5f32.powi(round as i32));
                let seed_r = order_seed ^ (round as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let mut order_r: Vec<u32> = (0..n_cells as u32).collect();
                order_r.sort_by_key(|&c| jitter(seed_r, c as u64).to_bits());
                let before = placed;
                for cell in order_r {
                    if placed >= target {
                        break;
                    }
                    k += 1;
                    if let Some(mut rec) = attempt(cell, k, &mut canvas, &mut cache) {
                        placed += 1;
                        in_pass += 1;
                        if let Some(pr) = progress {
                            if placed % 64 == 0 {
                                pr(PaintProgress::Placed(placed));
                            }
                        }
                        rec.id = placed as u32;
                        score.strokes.push(rec);
                    }
                }
                if placed - before < p.budget / 200 {
                    break; // the floor is no longer what stops it
                }
            }
            floor_mul.set(1.0);
            tracing::info!(target: "plakat", "fill {:.2}: {} rounds → {} of {} strokes", p.fill, round, placed, p.budget);
        }

        // Critic verdict: if the pass didn't improve the score by `margin`, ROLL BACK the canvas + score and
        // record the stage as tabu (§10.1). The block-in is never rejected — the canvas must be covered.
        if let (Some(c), Some((snap_canvas, snap_len, snap_placed, before))) = (critic, snapshot) {
            let after = c(&canvas.to_image());
            if !block_in && after < before + margin {
                canvas = snap_canvas;
                score.strokes.truncate(snap_len);
                placed = snap_placed;
                rejected.push(pass.stage.clone());
            }
        }
        // Dry the canvas before the next pass so the just-laid masses SET: the restatement then reads as fresh
        // overlays instead of picking the masses back up and stirring them into mud. Wet-into-wet is `--dry 0`.
        lap("pass:strokes", &mut prof_acc, &mut prof_t);
        // WET-INTO-WET per pass, tapering coarse → fine: the broad washes bloom into each other while still wet,
        // the later, finer work goes onto paper that has set and stays crisp. One bleed over the finished
        // painting fused EVERYTHING (measured: an ink-wash lost half its sharpness in that final step, and the
        // whole sheet read as one blur). The same schedule is reproduced by the score replay at each stage
        // boundary, so the drawing stays byte-exact.
        // (A luminous paint's brush passes come after its wash stages: their taper index continues from them.)
        let (b_idx, b_n) = if p.luminous { (p.armature_levels.max(2) as usize + face_wash_levels + layer, n_stages_total) } else { (layer, passes.len()) };
        if p.bleed > 0.0 {
            canvas.bleed_with(p.bleed * pass_bleed_taper(b_idx, b_n), p.diffuse);
        }
        if b_idx + 1 < b_n {
            canvas.dry(1.0 - p.dry);
        }
        lap("pass:bleed+dry", &mut prof_acc, &mut prof_t);
        stats.push(PassStat { stage: pass.stage.clone(), radius, strokes: in_pass, seconds: pass_t0.elapsed().as_secs_f64() });
        if prof_on { eprintln!("PROFILE pass {layer} r={radius:.1} strokes={in_pass} {:.2}s", pass_t0.elapsed().as_secs_f64()); }
    }
    if !rejected.is_empty() {
        tracing::info!(target: "plakat", "paint critic: rejected {} pass(es): {}", rejected.len(), rejected.join(", "));
    }

    // CONTOUR pass (line media): DRAW the strongest edges as clean lines — pen/pencil/charcoal outline the
    // subject, they don't only shade it. Laid before the bleed so a smudgy medium softens the lines a touch.
    // (Density media draw their contours in `ink_drawing`; this pass would stack its row-ordered dashes on top,
    // thickening whatever lies in the first rows until its cap hits.)
    // SUMI-E: over the wash painting, the darkest masses as bold dry-brush black shapes (the second register).
    if p.sumi && placed < p.budget {
        canvas.dry(0.0);
        sumi_ink(&mut canvas, &mut score, input, p, &mut placed, &mut k, progress);
    }
    // THE FACE'S LINE (see `face_line_pass`): over the dried planes, before any contour pass of the medium's.
    if let Some(fs) = face_scope.as_deref().filter(|_| !p.draw_contours && placed < p.budget) {
        face_line_pass(&mut canvas, &mut score, source, fs, p, protect_all.as_deref(), &mut placed, &mut k);
    }
    if p.draw_contours && placed < p.budget {
        ink_contours(&mut canvas, &mut score, input, source, p, protect_all.as_deref(), &mut placed, &mut k);
    } else if p.contour > 0.0 && !p.density && placed < p.budget {
        contour_pass(&mut canvas, &mut score, input, p, &mut placed, &mut k);
    }

    // THE RIGGER: the few shapes too thin for the ladder to have laid at all (see `rigger_pass`). Last, so it
    // can see what the brushwork actually carried and restore only what was lost.
    if p.rigger > 0.0 && placed < p.budget {
        rigger_pass(&mut canvas, &mut score, input, p, protect_all.as_deref(), &mut placed, &mut k);
    }

    // HOTSPOTS: flat blown highlights re-modelled into form (see `hotspot_pass`). After the rigger, so it
    // judges the canvas as finally painted.
    if p.hotspot > 0.0 && placed < p.budget {
        hotspot_pass(&mut canvas, &mut score, p, protect_all.as_deref(), &mut placed, &mut k);
    }

    // SILHOUETTE pass: draw a soft edge along the detected SUBJECT boundary so shoulders/collar read by their
    // contour against a light ground. Laid before the bleed so a wet medium softens the line.
    if p.silhouette > 0.0 && p.subject_mask.is_some() && placed < p.budget {
        silhouette_pass(&mut canvas, &mut score, input, p, &mut placed, &mut k);
    }

    // SPLATTER pass (watercolour / ink): flick droplets across the painting — the signature spatter. Laid before
    // the bleed so wet media soften a few of the spots into little blooms.
    // POOLS: the dried rim of every wash, painted last (see `pool_pass`). Under the spatter.
    // A rim needs an edge that SET: wet-on-wet never has one, and the dry brush had no water to pool.
    let rims_form = matches!(p.technique, WetTechnique::None | WetTechnique::WetOnDry);
    // A wash medium's rims and spatter are its own marks, not part of the stroke economy: a plan that asks
    // for six hundred washes and nothing else must still get them, where "placed < budget" starved both.
    let finish_ok = placed < p.budget || p.luminous;
    if p.luminous && p.edge_pool > 0.0 && rims_form && finish_ok {
        pool_pass(&mut canvas, &mut score, p, &mut placed, &mut k);
    }
    // LEAKS: runs out of the wet washes (see `leak_pass`), after the rims, under the spatter.
    if p.luminous && p.leak > 0.0 && finish_ok {
        leak_pass(&mut canvas, &mut score, p, &mut placed, &mut k);
    }
    if p.splatter > 0.0 && finish_ok {
        splatter_pass(&mut canvas, &mut score, input, p, &mut placed, &mut k);
    }

    // A last, light wet-into-wet touch over the finish passes (contour, silhouette, splatter) so a wet medium
    // softens those marks a little — the strong fusion happened pass by pass above.
    if p.bleed > 0.0 {
        canvas.bleed_with(p.bleed * FINAL_BLEED, p.diffuse);
    }

    lap("finish passes", &mut prof_acc, &mut prof_t);
    if prof_on {
        let mut agg: Vec<(&'static str, f64)> = Vec::new();
        for (n, t) in &prof_acc {
            match agg.iter_mut().find(|(m, _)| m == n) { Some(e) => e.1 += t, None => agg.push((n, *t)) }
        }
        let total: f64 = agg.iter().map(|(_, t)| t).sum();
        for (n, t) in &agg { eprintln!("PROFILE {:<40} {:7.2}s {:5.1}%", n, t, 100.0 * t / total.max(1e-9)); }
        eprintln!("PROFILE {:<40} {:7.2}s", "TOTAL paint_inner", total);
    }
    PaintResult { canvas, strokes: placed, score, rejected, stats, seconds: started.elapsed().as_secs_f64() }
}

/// The fraction of the medium's bleed applied after pass `layer` of `n`: full on the block-in, none on the
/// finest pass (linear in between). Shared with the score replay so both apply the identical schedule.
pub fn pass_bleed_taper(layer: usize, n: usize) -> f32 {
    if n <= 1 {
        1.0
    } else {
        1.0 - layer as f32 / (n - 1) as f32
    }
}

/// The fraction of the medium's bleed applied once over the finished painting (after the finish passes).
pub const FINAL_BLEED: f32 = 0.2;

/// The SILHOUETTE pass: draw a soft edge along the detected SUBJECT boundary (the matte's edge — a fact), so a
/// light subject (a white shirt, shoulders) reads against a light ground by its CONTOUR instead of vanishing.
/// The edge colour is the LOCAL reference colour darkened a little (a soft form edge, not an ink line); strokes
/// follow the boundary tangent, low charge, and are recorded like any other stroke (replay-exact).
fn silhouette_pass(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, placed: &mut usize, k: &mut u64) {
    let (w, h) = (input.width(), input.height());
    let mask = match &p.subject_mask {
        Some(m) if m.len() == (w * h) as usize => m,
        _ => return,
    };
    let strength = p.silhouette.clamp(0.0, 1.0);
    // The boundary = the gradient of the subject mask.
    let (gx, gy) = sobel(mask, w, h);
    let mag: Vec<f32> = gx.iter().zip(&gy).map(|(a, b)| (a * a + b * b).sqrt()).collect();
    let peak = mag.iter().copied().fold(0.0_f32, f32::max).max(1e-3);
    let radius = (p.min_brush * 1.1).max(2.0);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let brush = p.brush; // header brush (streak/round recorded) → replay-exact
    let grid = radius.max(2.0);
    let cols = ((w as f32) / grid).ceil() as u32;
    let rows = ((h as f32) / grid).ceil() as u32;
    let cap = ((*placed) as f32 + (p.budget - *placed) as f32 * (0.06 + 0.14 * strength)).min(p.budget as f32) as usize;
    for gyi in 0..rows {
        for gxi in 0..cols {
            if *placed >= cap {
                break;
            }
            *k += 1;
            let cx = (gxi as f32 + 0.5) * grid + jitter(p.seed ^ 0x51A1, *k) * grid;
            let cy = (gyi as f32 + 0.5) * grid + jitter(p.seed ^ 0x51A2, k.wrapping_add(1)) * grid;
            if cx < 1.0 || cy < 1.0 || cx >= w as f32 - 1.0 || cy >= h as f32 - 1.0 {
                continue;
            }
            let i = cy as usize * w as usize + cx as usize;
            let m = (mag[i] / peak).clamp(0.0, 1.0);
            if m < 0.35 {
                continue;
            }
            let base = input.get_pixel(cx as u32, cy as u32).0;
            let dl = color::linear_luma(color::srgb_to_linear(base));
            // Grow along the boundary TANGENT (grow_path follows the isophote, perpendicular to the gradient).
            let path = grow_path(cx, cy, radius, &gx, &gy, input, base, p.protect.as_deref(), None, None, 0.85, STOP_TOL);
            if path.len() < 2 {
                continue;
            }
            let path = waver_path(&path, 0.08 * radius, p.seed, *k);
            let np = p.palette.pigments.len();
            // EDGE MODE — how a painter marks this edge (RFC §7):
            let (load, wipe, wet, press): (Vec<f32>, bool, f32, f32) = match p.silhouette_mode {
                EdgeMode::Lost => continue,
                EdgeMode::Line => {
                    // A soft drawn line in the local dark.
                    let mut l = vec![0f32; np];
                    l[ink] = p.charge * strength * m * (0.35 + 0.45 * (1.0 - dl));
                    (l, false, 0.5, 0.8)
                }
                EdgeMode::Colour => {
                    // A CHROMATIC edge: mark by COOLING the local colour (no line) — mix the local tone with the
                    // palette's coolest pigment so the edge reads as a temperature shift, not a value line.
                    let cool = crate::paint::canvas::coolest_pigment(&p.palette);
                    let target = [(base[0] as f32 * 0.9) as u8, (base[1] as f32 * 0.95) as u8, base[2].saturating_add(12)];
                    let mx = mixer::solve_mixture(&p.palette, target, 3);
                    let mut l = vec![0f32; np];
                    for (&idx, &wt) in mx.pigments.iter().zip(mx.weights.iter()) {
                        l[idx] = wt * p.charge * strength * m * 0.7;
                    }
                    l[cool] += p.charge * strength * m * 0.5;
                    (l, false, 0.6, 0.7)
                }
                EdgeMode::Knife => {
                    // A KNIFE edge: scrape/lift to a crisp LIGHTER edge (a negative, palette-knife edge). Applied
                    // as a wipe at wet*lift, exactly as replay applies it.
                    (vec![0f32; np], true, 1.4, 1.0)
                }
            };
            let s = Stroke { path, width0: radius, width1: (radius * 0.8).max(1.0), load, pressure: press, wetness: wet };
            if wipe {
                s.wipe(canvas, &brush, wet * p.lift);
            } else {
                s.rasterize(canvas, &brush);
            }
            let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(idx, v)| (p.palette.pigments[idx].name.to_string(), *v)).collect();
            *placed += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe,
                wash: false,
                stage: "silhouette".into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.4,
                mix,
                wet: s.wetness,
                press: s.pressure,
                streak: brush.streak,
                round: brush.round, pickup: None, bristles: None,
            });
        }
    }
}

/// The CONTOUR / line pass: draw the reference's strongest edges as clean dark lines that follow the edge
/// tangent — the drawn linework of pen, pencil and charcoal (the boat, the figure, the tree outlines). Strokes
/// are thin, low-waver, no-pickup, in the local dark colour, recorded into the score like any other stroke.
/// THE RIGGER. A stem, a spoon handle, the line of a shelf: anything NARROWER than the minimum brush cannot be
/// laid by the brush ladder at all, so it does not soften — it disappears. A painter finishes with a rigger,
/// one long thin decisive stroke, and puts those few things back.
///
/// It draws RIDGES, not edges. A contour follows a boundary between two masses; a thin shape is lighter or
/// darker than BOTH its sides, so an edge detector fires twice beside it and never on it. The ridge map is the
/// difference between a blur at the thin scale and a blur well above it, which responds to exactly the band of
/// sizes the ladder cannot reach.
///
/// And it only restores what the painting LOST: the same ridge map is measured on the canvas as painted, and a
/// mark is spent only where the picture has a ridge and the canvas no longer does. That is what keeps the pass
/// honest — a shape the brushwork already carried is left alone — and it is self-limiting, because every
/// stroke it lays removes its own reason to lay another.
///
/// Rationed hard: too many thin lines is a wiry, over-drawn picture, which is worse than the missing stems and
/// far harder to walk back.
#[allow(clippy::too_many_arguments)]
/// HOTSPOTS. A specular highlight — the shine on a bald head, on a glazed pot, on wet stone — arrives at the
/// painter as a small bright region with a soft falloff, and the armature snaps it into ONE value mass. The
/// brush then fills that mass flat, and what should be a turning form reads as a hole cut in the picture: a
/// pale plateau with a hard rim.
///
/// The fix is not to remove it. A highlight is where the light is and the picture wants it; what it lacks is
/// its FALLOFF. So the plateau is re-modelled into a dome: brightest at its own centre, easing to the value
/// its rim already sits against, over the shape's own extent.
///
/// The gradient is INVENTED, not copied. Nothing here reads the source — a painter does not trace a highlight,
/// they know it has a soft edge and put one there. The dome comes from the spot's own geometry and the values
/// already on the canvas around it, so this stays brushwork inventing a surface (RFC §1.1).
///
/// Only the VALUE is re-modelled; each stroke keeps the hue already in that place, because a specular
/// highlight is a lightness event and shifting its colour would put a different material there.
#[allow(clippy::too_many_arguments)]
fn hotspot_pass(canvas: &mut Canvas, score: &mut StrokeScore, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64) {
    let strength = p.hotspot.clamp(0.0, 1.0);
    if strength <= 0.0 || *placed >= p.budget {
        return;
    }
    let img = canvas.to_image();
    let (w, h) = img.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;
    let l = luma_map(&img);
    // Find what is BRIGHT against its surroundings first, and only then ask whether it is a plateau. Testing
    // flatness per pixel cannot work: a window wide enough to see a highlight is as wide as the highlight, so
    // it always straddles the rim and every pixel reads as steep. Plateau-ness is a property of the REGION.
    let r_wide = ((w.min(h) as f32) / 40.0).round().clamp(4.0, 64.0) as i32;
    let wide = box_blur(&l, w, h, r_wide);
    let seed: Vec<bool> = (0..n).map(|i| l[i] - wide[i] > 0.10).collect();

    // Each plateau as its own region, and only ones small enough to BE a highlight: a lit wall is not one.
    let max_area = (n / 400).max(64);
    let mut seen = vec![false; n];
    let mut laid = 0usize;
    let mut cands: Vec<(f32, usize, f64, f64, f32, f32, f32)> = Vec::new();
    // ONE mixture cache for the pass. A fresh cache per mark means every mark solves its pigments from
    // scratch, which is the 99%-of-cost path the solver work removed — it made this pass take longer than
    // the whole painting.
    let mut mix_cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    for start in 0..n {
        if !seed[start] || seen[start] || *placed >= p.budget {
            continue;
        }
        let mut region = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        let mut bail = false;
        while let Some(i) = stack.pop() {
            region.push(i);
            if region.len() > max_area {
                bail = true;
                break;
            }
            let (x, y) = (i % wu, i / wu);
            for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx < 0 || ny < 0 || nx >= wu as i64 || ny >= hu as i64 {
                    continue;
                }
                let j = ny as usize * wu + nx as usize;
                if seed[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if bail || region.len() < 24 {
            continue;
        }
        if let Some(m) = protect {
            if region.iter().any(|&i| m.get(i).copied().unwrap_or(false)) {
                continue;
            }
        }
        // Its centre, its reach, and the value it has to ease INTO — the canvas just outside its own rim.
        let (mut cx, mut cy) = (0f64, 0f64);
        for &i in &region {
            cx += (i % wu) as f64;
            cy += (i / wu) as f64;
        }
        cx /= region.len() as f64;
        cy /= region.len() as f64;
        let radius = region
            .iter()
            .map(|&i| {
                let (dx, dy) = ((i % wu) as f64 - cx, (i / wu) as f64 - cy);
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0f64, f64::max)
            .max(1.0) as f32;
        let core = region.iter().map(|&i| l[i]).fold(0.0f32, f32::max);
        let rim = {
            let ring: Vec<f32> = region
                .iter()
                .filter_map(|&i| {
                    let (x, y) = ((i % wu) as f32, (i / wu) as f32);
                    let (dx, dy) = (x - cx as f32, y - cy as f32);
                    let d = (dx * dx + dy * dy).sqrt().max(1e-3);
                    let (sx, sy) = (x + dx / d * radius * 0.45, y + dy / d * radius * 0.45);
                    (sx >= 0.0 && sy >= 0.0 && sx < w as f32 && sy < h as f32).then(|| wide[sy as usize * wu + sx as usize])
                })
                .collect();
            if ring.is_empty() {
                continue;
            }
            ring.iter().sum::<f32>() / ring.len() as f32
        };
        if core - rim < 0.04 {
            continue;
        }
        // PLATEAU, OR ALREADY A DOME? Over a dome the top value belongs to the centre alone; over a blown
        // plateau most of the region sits up at it. That share is the whole test, and it is why a highlight
        // the brushwork modelled properly is left untouched.
        let high = rim + 0.8 * (core - rim);
        let plateau = region.iter().filter(|&&i| l[i] >= high).count() as f32 / region.len() as f32;
        if plateau < 0.30 {
            continue;
        }
        cands.push((core - rim, region.len(), cx, cy, radius, core, rim));
        let _ = plateau;
        continue;
    }

    // THE FEW THAT MATTER. A lit picture has bright passages everywhere, and re-modelling all of them spent
    // 680k marks — most of a whole painting's budget on a finish touch. A painter polishes the handful of
    // highlights that actually read, so the candidates are ranked by how badly each one steps (how far its
    // peak sits above its rim, weighted by how big it is) and only the top few are touched.
    cands.sort_by(|a, b| (b.0 * b.1 as f32).partial_cmp(&(a.0 * a.1 as f32)).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(12);
    // And a hard ceiling on the pass, as the rigger has. A finish touch must never be able to cost what the
    // painting costs: at one mark per pixel over a halo this ran to 680k marks, most of a whole budget.
    let ceiling = (*placed + (p.budget / 200).max(400)).min(p.budget);

    for (_, _, cx, cy, radius, core, rim) in cands {
        // THE DOME, laid OUTWARD. Easing from the peak to the rim across the patch itself is what a first
        // build did, and it is wrong in the one way that matters: it darkens the patch's own edge down to the
        // surroundings, so the highlight shrinks to a small hard core with a ring round it — a worse hole
        // than the plateau it replaced.
        //
        // The hard rim is a STEP, and a step is softened by spreading it, not by steepening what is inside
        // it. So the dome holds full strength across the patch and eases over a HALO beyond it, turning the
        // step into a ramp. The highlight keeps its size and its brightness; what changes is how it meets
        // the form around it.
        let reach = radius * 1.6;
        // Stepped at the brush's own spacing: a mark covers its own radius, so one per pixel lays the same
        // paint tens of times over and costs the score tens of thousands of records for it.
        let rr = (p.min_brush * 0.8).max(1.0);
        let band = {
            let r = reach.ceil() as i64;
            let (ix, iy) = (cx as i64, cy as i64);
            // At least a couple of pixels: a mark covers its own radius, and `min_brush` can be 1 on a
            // focal plane, which silently turns this back into one mark per pixel.
            let step = (rr.max(2.0)) as i64;
            let mut v = Vec::new();
            let mut y = (iy - r).max(0);
            while y <= (iy + r).min(hu as i64 - 1) {
                let mut x = (ix - r).max(0);
                while x <= (ix + r).min(wu as i64 - 1) {
                    let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                    if (dx * dx + dy * dy).sqrt() <= reach as f64 {
                        v.push(y as usize * wu + x as usize);
                    }
                    x += step;
                }
                y += step;
            }
            v
        };
        for &i in &band {
            if *placed >= ceiling {
                break;
            }
            let (x, y) = ((i % wu) as f32, (i / wu) as f32);
            let t = (((x - cx as f32).powi(2) + (y - cy as f32).powi(2)).sqrt() / reach).clamp(0.0, 1.0);
            // Flat out to the patch's own edge, then a cosine ease across the halo.
            let dome = if t <= 0.6 { 1.0 } else { 0.5 * (1.0 + (((t - 0.6) / 0.4) * std::f32::consts::PI).cos()) };
            let want = rim + (core - rim) * dome;
            let have = l[i];
            // `strength` is how far the flat plateau is taken toward that dome. At 0 it is left alone; the
            // default softens it; at 1 the highlight is fully modelled form.
            let target = have + (want - have) * strength;
            if (target - have).abs() < 0.004 {
                continue;
            }
            // Keep the hue, move the value: a specular highlight is a lightness event.
            let c = img.get_pixel(x as u32, y as u32).0;
            let scale = (target / have.max(1e-3)).clamp(0.0, 1.8);
            let tint: Srgb = [
                (c[0] as f32 * scale).clamp(0.0, 255.0) as u8,
                (c[1] as f32 * scale).clamp(0.0, 255.0) as u8,
                (c[2] as f32 * scale).clamp(0.0, 255.0) as u8,
            ];
            *k += 1;
            let load = mixture_cached(&mut mix_cache, tint, &p.palette, p.charge, p.palette.pigments.len());
            let path = vec![[x, y], [x + 0.6, y + 0.2]];
            let st = Stroke { path, width0: rr, width1: rr * 0.8, load, pressure: 0.8, wetness: 0.5 };
            st.rasterize(canvas, &p.brush);
            let mix: Vec<(String, f32)> = st.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(idx, v)| (p.palette.pigments[idx].name.to_string(), *v)).collect();
            *placed += 1;
            laid += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe: false,
                wash: false,
                stage: "hotspot".into(),
                spline: st.path,
                w0: st.width0,
                w1: st.width1,
                taper: 0.3,
                mix,
                wet: st.wetness,
                press: 0.8,
                streak: p.brush.streak,
                round: p.brush.round,
                pickup: None,
                bristles: None,
            });
        }
    }
    if laid > 0 {
        tracing::info!(target: "plakat", "hotspots: {laid} marks re-modelling flat highlights (strength {strength:.2})");
    }
}

fn rigger_pass(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64) {
    let (w, h) = (input.width(), input.height());
    let n = (w * h) as usize;
    let strength = p.rigger.clamp(0.0, 1.0);
    if strength <= 0.0 || *placed >= p.budget {
        return;
    }
    // The band of sizes the brush ladder cannot reach: narrower than its finest mark.
    let r_thin = (p.min_brush * 0.5).round().max(1.0) as i32;
    let r_wide = (p.min_brush * 2.5).round().max(3.0) as i32;
    let ridge_of = |img: &RgbImage| -> Vec<f32> {
        let l = luma_map(img);
        let a = box_blur(&l, w, h, r_thin);
        let b = box_blur(&l, w, h, r_wide);
        a.iter().zip(&b).map(|(x, y)| x - y).collect()
    };
    let want = ridge_of(input);
    let have = ridge_of(&canvas.to_image());
    // What the picture has and the canvas has lost. Sign matters: a light stem restored as a dark one is not
    // the same shape, so the two must agree in direction as well as presence.
    let missed: Vec<f32> = (0..n)
        .map(|i| {
            if want[i].signum() != have[i].signum() {
                want[i].abs()
            } else {
                (want[i].abs() - have[i].abs()).max(0.0)
            }
        })
        .collect();
    let mut sorted: Vec<f32> = missed.iter().copied().filter(|v| *v > 1e-4).collect();
    if sorted.len() < 32 {
        return;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Even at full strength this is the top few percent of what was lost — the few things worth a rigger.
    let keep = 0.01 + 0.05 * strength;
    let thr = sorted[((1.0 - keep) * (sorted.len() as f32 - 1.0)) as usize].max(1e-3);

    // Along the ridge, which is across its own gradient.
    let luma = luma_map(input);
    let (gx, gy) = coherent_gradient(&luma, w, h, r_thin.max(2), FlowInfill::Flat);
    // IS IT A LINE AT ALL? A ridge map answers "brighter than its surroundings", and a round highlight
    // answers yes — so the first build drew concentric rings around every onion, tracing each blob's isophote
    // into a target. A stem is a LINE: its structure tensor is strongly oriented. A blob's is isotropic. So
    // the rigger draws only where the neighbourhood agrees on a direction, which is what makes a thin shape
    // thin in the first place.
    let linear: Vec<f32> = {
        let (sx, sy) = sobel(&luma, w, h);
        let (mut jxx, mut jyy, mut jxy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        for i in 0..n {
            jxx[i] = sx[i] * sx[i];
            jyy[i] = sy[i] * sy[i];
            jxy[i] = sx[i] * sy[i];
        }
        let r = r_thin.max(2);
        let (jxx, jyy, jxy) = (box_blur(&jxx, w, h, r), box_blur(&jyy, w, h, r), box_blur(&jxy, w, h, r));
        (0..n)
            .map(|i| {
                let disc = ((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]).sqrt();
                (disc / (jxx[i] + jyy[i] + 1e-6)).clamp(0.0, 1.0)
            })
            .collect()
    };
    let radius = (p.min_brush * 0.55).max(1.0);
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.bristles = 1;
    brush.streak = 0.05;
    brush.round = 0.95;
    // A hard ration, and never more than a small share of the sheet however much budget is left over.
    let room = (p.budget - *placed) as f32 * (0.02 + 0.06 * strength);
    let cap = (*placed + (room.min(n as f32 * 0.0008 * (0.5 + strength)) as usize)).min(p.budget);
    let grid = (radius * 3.0).max(3.0);
    let cols = ((w as f32) / grid).ceil() as u32;
    let rows = ((h as f32) / grid).ceil() as u32;
    let trace = (w.max(h) as f32 * 0.03 / (2.2 * radius)).max(1.0);
    let mut mix_cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    for gyi in 0..rows {
        for gxi in 0..cols {
            if *placed >= cap {
                return;
            }
            *k += 1;
            let cx = (gxi as f32 + 0.5) * grid + jitter(p.seed ^ 0x81D6, *k) * grid;
            let cy = (gyi as f32 + 0.5) * grid + jitter(p.seed ^ 0x81D7, k.wrapping_add(1)) * grid;
            if cx < 1.0 || cy < 1.0 || cx >= w as f32 - 1.0 || cy >= h as f32 - 1.0 {
                continue;
            }
            let i = cy as usize * w as usize + cx as usize;
            if missed[i] < thr || linear[i] < 0.6 {
                continue;
            }
            // IS IT A LINE, OR THE EDGE OF SOMETHING BIG? Linearity alone cannot tell: a ring is linear at
            // every point along it, so the first build with that gate still drew each onion's highlight rim
            // as a target of concentric circles. A true thin shape is a local EXTREMUM ACROSS its width —
            // step off either side and the value falls the same way both times. An edge or a rim is a step:
            // one side is darker, the other lighter. So look at both sides and require them to agree.
            let gnorm = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt().max(1e-6);
            let (nx, ny) = (gx[i] / gnorm, gy[i] / gnorm);
            let off = (p.min_brush * 1.2).max(2.0);
            let at = |sx: f32, sy: f32| -> f32 {
                let px = (cx + sx).clamp(0.0, w as f32 - 1.0) as usize;
                let py = (cy + sy).clamp(0.0, h as f32 - 1.0) as usize;
                luma[py * w as usize + px]
            };
            let here = luma[i];
            let (a_side, b_side) = (at(nx * off, ny * off) - here, at(-nx * off, -ny * off) - here);
            if a_side.signum() != b_side.signum() || a_side.abs().min(b_side.abs()) < 0.02 {
                continue;
            }
            if protect.map(|m| m.get(i).copied().unwrap_or(false)).unwrap_or(false) {
                continue;
            }
            // NOT OVER HAIR, AND NOT OVER A FACE. A rigger is for the few things the brushwork could not lay
            // at all. A beard has already been painted strand by strand with its own tool, and a face is
            // modelled form; drawing lines over either adds a faint tracery across exactly the passages the
            // picture is about. Measured as a difference map, that tracery was most of what this pass was
            // putting down outside the shelf it was built for.
            if p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.25 {
                continue;
            }
            if p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.25 {
                continue;
            }
            // Its OWN colour, not ink: a pale stem over a dark shelf is light, and drawing it dark would put a
            // different object there.
            let c = input.get_pixel(cx as u32, cy as u32).0;
            let load = mixture_cached(&mut mix_cache, c, &p.palette, p.charge, p.palette.pigments.len());
            let path = grow_path(cx, cy, radius, &gx, &gy, input, c, protect, None, None, trace, STOP_TOL);
            if path.len() < 2 {
                continue;
            }
            let path = waver_path(&path, 0.05 * radius, p.seed, *k);
            let s = Stroke { path, width0: radius, width1: (radius * 0.45).max(0.6), load, pressure: 1.0, wetness: 0.35 };
            s.rasterize(canvas, &brush);
            let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(idx, v)| (p.palette.pigments[idx].name.to_string(), *v)).collect();
            *placed += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe: false,
                wash: false,
                stage: "rigger".into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.6,
                mix,
                wet: s.wetness,
                press: 1.0,
                streak: brush.streak,
                round: brush.round,
                pickup: Some(0.0),
                bristles: Some(1),
            });
        }
    }
}

fn contour_pass(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, placed: &mut usize, k: &mut u64) {
    let (w, h) = (input.width(), input.height());
    let sharp = imageops::unsharpen(input, 1.0, 1);
    let luma = luma_map(&sharp);
    let (gx, gy) = sobel(&luma, w, h);
    let mag: Vec<f32> = gx.iter().zip(&gy).map(|(a, b)| (a * a + b * b).sqrt()).collect();
    // Keep only the strongest edges; more `contour` → draw more of them.
    let mut sorted = mag.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let keep = (0.03 + 0.12 * p.contour.clamp(0.0, 1.0)).min(0.3);
    let thr = sorted[((1.0 - keep) * (sorted.len() as f32 - 1.0)) as usize].max(1e-3);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let radius = (p.min_brush * 0.7).max(1.5);
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.streak = 0.08;
    brush.round = 0.95;
    let cap = ((*placed) as f32 + (p.budget - *placed) as f32 * (0.08 + 0.14 * p.contour)).min(p.budget as f32) as usize;
    let grid = radius.max(1.5);
    let cols = ((w as f32) / grid).ceil() as u32;
    let rows = ((h as f32) / grid).ceil() as u32;
    for gyi in 0..rows {
        for gxi in 0..cols {
            if *placed >= cap {
                break;
            }
            *k += 1;
            let jx = jitter(p.seed ^ 0xC047, *k) * grid;
            let jy = jitter(p.seed ^ 0xC048, k.wrapping_add(1)) * grid;
            let cx = (gxi as f32 + 0.5) * grid + jx;
            let cy = (gyi as f32 + 0.5) * grid + jy;
            if cx < 1.0 || cy < 1.0 || cx >= w as f32 - 1.0 || cy >= h as f32 - 1.0 {
                continue;
            }
            let i = cy as usize * w as usize + cx as usize;
            if mag[i] < thr {
                continue;
            }
            if let Some(m) = &p.protect {
                if m.get(i).copied().unwrap_or(false) {
                    continue;
                }
            }
            // The line's colour: the darker side of the edge (its shadow line), toward the ink pigment.
            let gnorm = mag[i].max(1e-6);
            let (ux, uy) = (gx[i] / gnorm, gy[i] / gnorm);
            let sx = (cx + ux * radius).clamp(0.0, w as f32 - 1.0);
            let sy = (cy + uy * radius).clamp(0.0, h as f32 - 1.0);
            let dark = sharp.get_pixel(sx as u32, sy as u32).0;
            let dl = color::linear_luma(color::srgb_to_linear(dark));
            let mut load = vec![0f32; p.palette.pigments.len()];
            // Darker edges → a stronger (blacker) line; lighter edges → a fainter graphite line.
            load[ink] = p.charge * (0.5 + 0.5 * (1.0 - dl)).clamp(0.35, 1.0);
            // Grow along the edge tangent (the isophote): a clean, low-waver drawn line.
            // TRACE the edge: a drawn contour runs along its boundary for tens of pixels (the generic cap of 2.2
            // radii would make 3px dashes of a 1.5px pen). The colour-drift and protect stops still end it.
            let trace = (w.max(h) as f32 * 0.04 / (2.2 * radius)).max(1.0);
            let path = grow_path(cx, cy, radius, &gx, &gy, &sharp, dark, p.protect.as_deref(), None, None, trace, STOP_TOL);
            if path.len() < 2 {
                continue;
            }
            let path = waver_path(&path, 0.05 * radius, p.seed, *k);
            let s = Stroke { path, width0: radius, width1: (radius * 0.7).max(1.0), load, pressure: 1.0, wetness: 0.4 };
            s.rasterize(canvas, &brush);
            let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(idx, v)| (p.palette.pigments[idx].name.to_string(), *v)).collect();
            *placed += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe: false, wash: false,
                stage: "contour".into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.3,
                mix,
                wet: s.wetness,
                press: s.pressure,
                streak: brush.streak,
                round: brush.round, pickup: None, bristles: None,
            });
        }
    }
}

/// The SPLATTER pass: flick fine droplets of pigment across the painting — the signature spatter of a loaded
/// brush tapped over the paper (visible in nearly every loose watercolour). Most droplets are tiny DARK spots in
/// the palette's darkest pigment; a few are coarser blobs; and — on a surface-white medium — a fraction LIFT to
/// the paper for the bright speckle of spray/snow/sparkle. Positions come from a hash so the spatter is even but
/// unstructured, and every droplet is recorded into the score (the bright ones as wipe strokes at `wet*lift`,
/// exactly as replay applies them) so a re-render reproduces it. `splatter` scales the count.
fn splatter_pass(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, placed: &mut usize, k: &mut u64) {
    let (w, h) = (input.width(), input.height());
    let strength = p.splatter.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    // Enough to READ. One droplet per 1400 px laid 905 marks among 88,000 on a 2048² sheet — a tasteful
    // accent nobody could see. The references spatter in the hundreds of visible drops; this is still a
    // small share of any budget.
    let want = (w as f32 * h as f32 / 600.0 * strength) as usize;
    let n = if p.luminous { want.min(20000) } else { want.min(p.budget.saturating_sub(*placed)).min(20000) };
    // Sizes scale with the sheet, so a drop on a 2048 sheet is a drop and not a pixel.
    let unit = (w.min(h) as f32 / 1024.0).max(0.5);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let np = p.palette.pigments.len();
    let can_lift = p.reserve.is_some() && p.lift > 1e-3;
    // BACKGROUND BIAS — real spatter lands on the clean paper, not over the subject's face. Bias droplets toward
    // LOW-saliency (background/margin) regions and keep them OFF the detected face, so spatter reads as a tasteful
    // accent instead of muddying the subject. Deterministic (recorded strokes replay regardless).
    let sal = saliency_field(input);
    // Use the score's base brush UNCHANGED (streak/round are recorded per-stroke below): a tiny droplet is
    // barely affected by pickup, and keeping the header brush is what makes replay byte-identical (replay
    // rebuilds each stroke's brush from the header + recorded streak/round — an unrecorded override would drift).
    let mut brush = p.brush;
    brush.streak = 0.0;
    brush.round = 1.0;
    for _ in 0..n {
        *k = k.wrapping_add(1);
        let hx = jitter(p.seed ^ 0x5D19, *k) + 0.5;
        let hy = jitter(p.seed ^ 0x9C4B, k.wrapping_add(11)) + 0.5;
        let cx = (hx * w as f32).clamp(1.0, w as f32 - 2.0);
        let cy = (hy * h as f32).clamp(1.0, h as f32 - 2.0);
        let i = cy as usize * w as usize + cx as usize;
        if let Some(m) = &p.protect {
            if m.get(i).copied().unwrap_or(false) {
                continue;
            }
        }
        // Skip the detected FACE entirely — never spatter over the face.
        if let Some(fm) = &p.face_mask {
            if fm.get(i).copied().unwrap_or(0.0) > 0.35 {
                continue;
            }
        }
        // Background bias: the busier (higher-saliency) a spot, the more likely the droplet is dropped, so most
        // land on the calm background/margins. `hb` in [0,1] from an independent hash stream.
        let hb = jitter(p.seed ^ 0x1B7F, k.wrapping_add(17)) + 0.5;
        if hb < sal[i] * 0.9 {
            continue;
        }
        let rr = (jitter(p.seed ^ 0x2AE7, k.wrapping_add(3)) + 0.5).clamp(0.0, 1.0);
        // Mostly fine; a tail of coarser blobs, the biggest a few brush widths — the drop that flew furthest.
        let radius = unit * if rr > 0.9 { 2.0 + 5.0 * (rr - 0.9) / 0.1 } else { 0.6 + 1.4 * rr };
        // A drop on wet paper blooms wide and soft; on dry paper it is a hard dot.
        let (radius, drop_wet) = match p.technique {
            WetTechnique::WetOnWet => (radius * 1.5, 0.9),
            WetTechnique::DryOnDry => (radius * 0.8, 0.15),
            _ => (radius, 0.5),
        };
        let path = vec![[cx, cy], [cx + 0.6, cy + 0.4]];
        // A drop READS by contrast: on dark paint it is the paper showing through (a lift), on light paint it
        // is pigment. Decided from the canvas as it stands at this spot — a fact of the picture, not a setting.
        let here = canvas.color_at(cx as u32, cy as u32);
        let dark_here = 1.0 - color::linear_luma(color::srgb_to_linear(here));
        let lift = can_lift && (jitter(p.seed ^ 0x71C3, k.wrapping_add(5)) + 0.5) < (0.15 + 0.6 * dark_here) * strength;
        if lift {
            // Bright droplet: scrape to the paper. Apply at wet*lift so replay (same formula) matches exactly.
            let wet = 1.6_f32;
            let s = Stroke { path, width0: radius, width1: radius, load: vec![0.0; np], pressure: 1.0, wetness: wet };
            s.wipe(canvas, &brush, wet * p.lift);
            *placed += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe: true,
                wash: false,
                stage: "splatter".into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.0,
                mix: Vec::new(),
                wet,
                press: 1.0,
                streak: 0.0,
                round: 1.0, pickup: None, bristles: None,
            });
        } else {
            let mut load = vec![0f32; np];
            load[ink] = p.charge * (0.45 + 0.55 * rr);
            let s = Stroke { path, width0: radius, width1: radius, load, pressure: 1.0, wetness: drop_wet };
            s.rasterize(canvas, &brush);
            let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(idx, v)| (p.palette.pigments[idx].name.to_string(), *v)).collect();
            *placed += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe: false, wash: false,
                stage: "splatter".into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.0,
                mix,
                wet: s.wetness,
                press: 1.0,
                streak: 0.0,
                round: 1.0, pickup: None, bristles: None,
            });
        }
    }
}

/// An EDGE-HARDNESS field in `[0,1]` from the reference (RFC §7 / edge-control craft). Real painters don't
/// OUTLINE — they vary how hard two masses MEET: a hard edge is high VALUE contrast (crisp meeting), a soft/lost
/// edge is low contrast (shapes blend or merge). We measure the luma (value) gradient on a lightly denoised
/// reference and normalise by a high percentile, so `1` marks the strongest value boundaries. Strokes then
/// TERMINATE where hardness is high (masses meet crisply) and cross freely where it's low (soft/lost) — which
/// yields lost-and-found along a contour for free, and never a drawn line. `strength` opens the gate.
fn edge_hardness(input: &RgbImage, strength: f32) -> (Vec<f32>, f32) {
    let (w, h) = (input.width(), input.height());
    let sm = imageops::blur(input, 1.0);
    let luma = luma_map(&sm);
    let (gx, gy) = sobel(&luma, w, h);
    let mut mag: Vec<f32> = gx.iter().zip(&gy).map(|(a, b)| (a * a + b * b).sqrt()).collect();
    let mut sorted = mag.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = sorted[((sorted.len() as f32 - 1.0) * 0.95) as usize].max(1e-3);
    for m in &mut mag {
        *m = (*m / p95).clamp(0.0, 1.0);
    }
    // Threshold: the edges strong enough to stop a stroke (so two masses meet crisply). The old bar (~0.57 at
    // default) only caught the top few percent, so most real object boundaries — a roof against sky, a figure
    // against a wall — were crossed and smeared. A lower bar makes ordinary structural edges terminate strokes.
    let thr = (0.62 - 0.42 * strength.clamp(0.0, 1.0)).clamp(0.22, 0.85);
    (mag, thr)
}

/// SALIENCY field (opt-in density gate): a smooth per-pixel importance map — high where the reference carries
/// STRUCTURE (the focal subject, edges, texture) and low in flat, empty passages (a plain sky, a blank wall or
/// curtain). Built from the luma-gradient energy spread to a REGIONAL scale (so it marks "this area is busy",
/// not "this pixel is an edge") and normalised by a high percentile. Used to thin stroke density where it is
/// low, so the subject is worked up while the background stays a calm mass. Deterministic (replay-exact).
fn saliency_field(input: &RgbImage) -> Vec<f32> {
    let (w, h) = (input.width(), input.height());
    let luma = luma_map(&imageops::blur(input, 1.0));
    let (gx, gy) = sobel(&luma, w, h);
    let mut mag: Vec<f32> = gx.iter().zip(&gy).map(|(a, b)| (a * a + b * b).sqrt()).collect();
    // Normalise the edge energy first, then SPREAD it over a region so a busy area lifts its whole neighbourhood.
    let mut sorted = mag.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = sorted[((sorted.len() as f32 - 1.0) * 0.95) as usize].max(1e-3);
    for m in &mut mag {
        *m = (*m / p95).clamp(0.0, 1.0);
    }
    let buf = image::GrayImage::from_fn(w, h, |x, y| image::Luma([(mag[(y * w + x) as usize] * 255.0) as u8]));
    let region = (w.min(h) as f32 * 0.045).clamp(4.0, 48.0);
    let blurred = imageops::blur(&buf, region);
    let mut sal: Vec<f32> = blurred.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    // Renormalise the spread field so the busiest region reaches 1 (the flat passages sit near 0).
    let mut s2 = sal.clone();
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let hi = s2[((s2.len() as f32 - 1.0) * 0.90) as usize].max(1e-3);
    for s in &mut sal {
        *s = (*s / hi).clamp(0.0, 1.0);
    }
    sal
}

/// FOCAL field for SELECTIVE DETAIL: the saliency field weighted by a CENTRE PRIOR (a broad Gaussian centred on
/// the canvas). Raw saliency marks a large busy texture (a beard, foliage) just as high as a compact focal
/// feature, so it alone can't tell "the eye's destination" from "a busy mass". The centre prior — the standard
/// saliency centre-bias — lifts the compact, central, high-contrast region (a portrait's eyes/glasses) above
/// the peripheral mass, so the crisp detail tier fires where the eye goes and the rest stays a loose wash.
/// Generic (no face model, no scene assumption): just centre-weighted contrast. Normalised to [0,1].
fn focal_field(input: &RgbImage) -> Vec<f32> {
    let (w, h) = (input.width(), input.height());
    let sal = saliency_field(input);
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let s = (w.min(h) as f32 * 0.38).max(1.0);
    let denom = 2.0 * s * s;
    let mut f: Vec<f32> = (0..(w as usize * h as usize))
        .map(|i| {
            let x = (i as u32 % w) as f32;
            let y = (i as u32 / w) as f32;
            let d2 = (x - cx).powi(2) + (y - cy).powi(2);
            sal[i] * (-d2 / denom).exp()
        })
        .collect();
    let mx = f.iter().copied().fold(0.0_f32, f32::max).max(1e-3);
    for v in &mut f {
        *v /= mx;
    }
    f
}


/// The density mark model (§8.6): at a cell, lay black hatch marks whose count scales with the target
/// darkness — value is built by mark DENSITY, not pigment concentration. Darker cells add a crosshatch. No
/// pickup, single bristle. Records each mark into the score; respects the budget. Returns the count laid.
#[allow(clippy::too_many_arguments)]
/// PEN-AND-INK (density media): rasterise the drawing that [`crate::paint::ink::plan`] derives from the
/// armature — contour lines first, then the hatch — with a pen (no pickup, full ink, a hair of waver), recording
/// every line as a stroke so the drawing replays exactly like a painting.
#[allow(clippy::too_many_arguments)]
/// SUMI-E's second register: the DARKEST masses of the sheet (its own value quantiles — at most the darkest
/// eighth) as bold dry-brush BLACK SHAPES over the wash painting: wide strokes following each mass's form, a
/// split-hair brush (streaky, ragged), no pickup, each stroke confined to its mass so the edge is the mass's
/// edge. Deterministic, replay-exact.
#[allow(clippy::too_many_arguments)]
fn sumi_ink(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let (w, h) = (input.width(), input.height());
    let long = w.max(h) as f32;
    let firm = imageops::blur(input, (long / 400.0).max(1.0));
    let fluma = luma_map(&firm);
    let mut sorted = fluma.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ink_t = sorted[((sorted.len() as f32 - 1.0) * 0.12) as usize].min(0.10);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let n = p.palette.pigments.len();
    let (fgx, fgy) = coherent_gradient(&fluma, w, h, (long / 200.0).max(2.0) as i32, FlowInfill::Flat);
    let mut dry = p.brush;
    dry.k_pickup = 0.0;
    dry.bristles = 11;
    dry.streak = 0.75; // a dry brush: ragged, split hairs
    dry.round = 0.55;
    let ir = (long / 90.0).max(3.0);
    let igrid = ir * 0.7;
    let (icols, irows) = (((w as f32) / igrid).ceil() as u32, ((h as f32) / igrid).ceil() as u32);
    let mask: Vec<bool> = fluma.iter().map(|&l| l <= ink_t).collect();
    let amt = p.charge * 1.2;
    for gyi in 0..irows {
        for gxi in 0..icols {
            if *placed >= p.budget {
                break;
            }
            *k += 1;
            let cx = (gxi as f32 + 0.5) * igrid + jitter(p.seed ^ 0x51, *k) * igrid;
            let cy = (gyi as f32 + 0.5) * igrid + jitter(p.seed ^ 0x52, k.wrapping_add(1)) * igrid;
            if cx < 0.0 || cy < 0.0 || cx >= w as f32 || cy >= h as f32 {
                continue;
            }
            let i = cy as usize * w as usize + cx as usize;
            if !mask[i] {
                continue;
            }
            let mut load = vec![0f32; n];
            load[ink] = amt;
            let target = firm.get_pixel(cx as u32, cy as u32).0;
            let path = grow_path(cx, cy, ir, &fgx, &fgy, &firm, target, None, Some((&mask, true)), None, 2.2, STOP_TOL);
            if path.len() < 2 {
                continue;
            }
            let s = Stroke { path, width0: ir * 1.5, width1: ir * 0.6, load, pressure: 1.0, wetness: 0.15 };
            s.rasterize(canvas, &dry);
            *placed += 1;
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "ink".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), amt)], wet: s.wetness, press: s.pressure, streak: dry.streak, round: dry.round, pickup: Some(0.0), bristles: Some(dry.bristles) });
            if let Some(pr) = progress { if *placed % 64 == 0 { pr(PaintProgress::Placed(*placed)); } }
        }
    }
}

/// The ink planner's CONTOURS only (no hatch), drawn on top of a tonal painting in the darkest pigment — the
/// line of a pencil sketch over its shading. Uses the medium's `contour` strength for how many edges to keep.
#[allow(clippy::too_many_arguments)]
/// The WASHES of a luminous medium: the picture's value MASSES, read from the SOURCE smoothed to the scale of
/// a wash (~0.7% of the long side — a sky, a wall, a road, not every cobble: the masses quantised at pixel
/// scale were a paint-by-numbers photo), its luma quantised to `armature_levels`, connected components of one
/// level and one hue family (the small ones left to the line), laid as area fills one pass per level from the
/// lightest to the darkest — the watercolour order — each pass bled (the wet edges bloom) and dried before the
/// next. A mass carries its own mean colour from the source (the picture's colours, not a re-keyed version:
/// a watercolour is light because its washes are THIN over white paper), mixed from the palette; the reserved
/// paper (`protect`) is cut out of every mass as a hole, so the lights stay paper.
#[allow(clippy::too_many_arguments)]
/// Which part of the sheet a run of washes lays (see `wash_passes`). `Sheet` is every wash path there was;
/// the other two are the watercolour FACE under a new painting: the sheet is washed around the face, then the
/// face is washed on its own, as planes.
#[derive(Clone, Copy)]
enum WashScope<'a> {
    Sheet,
    /// The sheet with the face (see `face_planes_scope`) left out.
    Outside(&'a [bool]),
    /// The face alone: its levels span the FACE's own value range, its masses are read at face scale, and its
    /// small lights may stay paper.
    Inside(&'a [bool]),
}

/// THE SHAPE OF A FACE'S PLANES. The detector gives a BOX, and a box holds hair, a collar and a slice of
/// background; washed as planes on the face's value range those took the face's skin-coloured glazes, and
/// the box printed itself on the sheet as a lighter rectangle around every head. A face is the SKIN inside
/// the box: an ellipse inscribed in it, and within that the pixels whose colour is near the box centre's —
/// near by a spread measured on the centre itself (a tolerance relative to this face, never a named colour),
/// connected to the centre, with the eyes and the mouth filled back in as holes. Hair, beard and background
/// fall to the sheet, which washes them as the masses they are. Without boxes the mask is the shape.
fn face_planes_scope(source: &RgbImage, p: &PaintParams) -> Option<Vec<bool>> {
    let fm = p.face_mask.as_deref()?;
    let (w, h) = (source.width() as usize, source.height() as usize);
    if p.face_boxes.is_empty() {
        return Some(fm.iter().map(|&v| v > 0.5).collect());
    }
    let mut out = vec![false; w * h];
    for b in &p.face_boxes {
        let (x1, y1, x2, y2) = (b[0].max(0.0), b[1].max(0.0), b[2].min(w as f32), b[3].min(h as f32));
        let (bw, bh) = (x2 - x1, y2 - y1);
        if bw < 8.0 || bh < 8.0 {
            continue;
        }
        let short = bw.min(bh);
        let (cx, cy) = ((x1 + x2) * 0.5, (y1 + y2) * 0.5);
        // A hair wider than the box: the detector's box is tight to the face.
        let (ea, eb) = (bw * 0.5 * 1.05, bh * 0.5 * 1.05);
        let (gx0, gy0) = ((x1 - bw * 0.1).max(0.0) as usize, (y1 - bh * 0.1).max(0.0) as usize);
        let (gx1, gy1) = (((x2 + bw * 0.1) as usize).min(w), ((y2 + bh * 0.1) as usize).min(h));
        let (cw, ch) = (gx1 - gx0, gy1 - gy0);
        if cw < 4 || ch < 4 {
            continue;
        }
        let crop = imageops::blur(&imageops::crop_imm(source, gx0 as u32, gy0 as u32, cw as u32, ch as u32).to_image(), (short / 50.0).max(1.0));
        // Chromaticity (brightness-free) and luma: hair and skin of one hue part by value, a collar by chroma.
        let feat: Vec<[f32; 3]> = crop
            .pixels()
            .map(|px| {
                let (r, g, bl) = (px.0[0] as f32, px.0[1] as f32, px.0[2] as f32);
                let s = r + g + bl + 1.0;
                [r / s, g / s, color::linear_luma(color::srgb_to_linear(px.0))]
            })
            .collect();
        let in_ellipse = |i: usize| {
            let (x, y) = ((gx0 + i % cw) as f32 + 0.5, (gy0 + i / cw) as f32 + 0.5);
            let (dx, dy) = ((x - cx) / ea, (y - cy) / eb);
            dx * dx + dy * dy <= 1.0
        };
        // The sample is the band of the eyes and the cheeks — the box's upper middle. The box CENTRE is the
        // mouth, and on a bearded face that is beard: sampled there the spread was so wide the whole
        // ellipse passed as skin, hair and all.
        let in_core = |i: usize| {
            let (x, y) = ((gx0 + i % cw) as f32 + 0.5, (gy0 + i / cw) as f32 + 0.5);
            let (dx, dy) = ((x - cx) / (ea * 0.55), (y - (y1 + bh * 0.42)) / (eb * 0.22));
            dx * dx + dy * dy <= 1.0
        };
        // The sample's colour and spread: a median and a median absolute deviation per feature. The luma
        // spread is floored so an evenly lit patch still admits the face's own shadow side; the chroma
        // spread is capped so a mixed sample cannot admit the hair — skin in shadow keeps its chroma, hair
        // of any colour does not share it.
        let mut med = [0f32; 3];
        let mut sig = [0f32; 3];
        let core: Vec<usize> = (0..cw * ch).filter(|&i| in_core(i)).collect();
        if core.len() < 16 {
            continue;
        }
        for c in 0..3 {
            let mut v: Vec<f32> = core.iter().map(|&i| feat[i][c]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            med[c] = v[v.len() / 2];
            let mut d: Vec<f32> = v.iter().map(|x| (x - med[c]).abs()).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let s = 1.4826 * d[d.len() / 2];
            sig[c] = if c < 2 { s.clamp(0.012, 0.025) } else { s.clamp(0.10, 0.20) };
        }
        let near = |i: usize| (0..3).map(|c| ((feat[i][c] - med[c]) / sig[c]).powi(2)).sum::<f32>() < 9.0;
        let cand: Vec<bool> = (0..cw * ch).map(|i| in_ellipse(i) && near(i)).collect();
        // Connected to the centre.
        let mut region = vec![false; cw * ch];
        let mut stack: Vec<usize> = core.iter().copied().filter(|&i| cand[i]).collect();
        for &i in &stack {
            region[i] = true;
        }
        while let Some(i) = stack.pop() {
            let (x, y) = (i % cw, i / cw);
            let mut nb = [usize::MAX; 4];
            if x > 0 { nb[0] = i - 1; }
            if x + 1 < cw { nb[1] = i + 1; }
            if y > 0 { nb[2] = i - cw; }
            if y + 1 < ch { nb[3] = i + cw; }
            for &j in &nb {
                if j != usize::MAX && cand[j] && !region[j] {
                    region[j] = true;
                    stack.push(j);
                }
            }
        }
        // The eyes, the brows, the mouth, a nostril: holes in the skin, and part of the face. Filled back in
        // (a hole up to a quarter of the face across); the hair beyond the ellipse is no hole.
        let holes: Vec<bool> = region.iter().map(|&r| !r).collect();
        let small = keep_regions_between(&holes, cw, ch, 1, ((short / 4.0) as usize).pow(2));
        for (r, &s) in region.iter_mut().zip(&small) {
            *r |= s;
        }
        // Opened a little, so a thread of skin-coloured background does not hang off the face.
        let r = ((short / 60.0) as usize).max(1);
        let region = dilate_bool(&erode_bool(&region, cw, ch, r), cw, ch, r);
        for i in 0..cw * ch {
            if region[i] {
                out[(gy0 + i / cw) * w + gx0 + i % cw] = true;
            }
        }
        if std::env::var_os("PLAKAT_PAINT_MASKS").is_some() {
            eprintln!("face scope: box {:.0}×{:.0} · centre colour (chroma {:.3},{:.3} · luma {:.3}) · spread ({:.3},{:.3},{:.3}) · {}% of the box", bw, bh, med[0], med[1], med[2], sig[0], sig[1], sig[2], 100 * region.iter().filter(|&&r| r).count() / (cw * ch).max(1));
        }
    }
    // DIAGNOSTIC (`PLAKAT_PAINT_MASKS=<dir>`): the planes' shape, as the CLI writes its masks.
    if let Some(dir) = std::env::var_os("PLAKAT_PAINT_MASKS") {
        let g = image::GrayImage::from_fn(w as u32, h as u32, |x, y| image::Luma([if out[y as usize * w + x as usize] { 255 } else { 0 }]));
        let _ = g.save(std::path::Path::new(&dir).join("mask_face_planes.png"));
    }
    Some(out)
}

/// A watercolour face under a new painting is PLANES AND LINE: a handful of flat washes meeting at hard
/// edges, the smallest lights left as paper, and the likeness carried by a drawn line on the features. Not a
/// fine brush ladder — modelled strokes over a wash turn a watercolour face back into an oil sketch.
fn face_is_planes(p: &PaintParams) -> bool {
    p.from_scratch && p.luminous && !p.book && p.face_mask.is_some()
}

fn wash_passes(canvas: &mut Canvas, score: &mut StrokeScore, reference: &RgbImage, source: &RgbImage, n_stages_total: usize, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    wash_passes_scoped(canvas, score, reference, source, n_stages_total, 0, WashScope::Sheet, p, protect, placed, k, progress);
}

/// The bounding box (x0, y0, x1, y1 exclusive) of a boolean mask, or None when nothing is set.
fn bool_bbox(m: &[bool], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if m[y * w + x] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    (x1 > 0).then_some((x0, y0, x1, y1))
}

/// THE FACE'S PAPER. The shape-aware reserve never reserves inside a face (a lit face as a white hole), but a
/// watercolour face keeps its SMALLEST lights as paper — the bridge of the nose, a cheek, the lower lip. The
/// paper is the face's brightest one-and-a-half percent, where that is also above the reserve luma (a lit
/// forehead is above the reserve over its whole extent, and reserving it left a hole where the face was),
/// judged on the source barely softened (a glint read at plane scale is smoothed into the skin around it),
/// opened so nothing ragged survives, and kept only as compact shapes between a fortieth and an eighth of
/// the face across by area — a glint down the bridge of the nose is long and thin — never a hole.
fn face_paper_shapes(source: &RgbImage, scope: &[bool], reserve: f32) -> Vec<bool> {
    let (w, h) = (source.width() as usize, source.height() as usize);
    let Some((x0, y0, x1, y1)) = bool_bbox(scope, w, h) else { return vec![false; w * h] };
    let face_short = (x1 - x0).min(y1 - y0);
    let fine = luma_map(&imageops::blur(source, (face_short as f32 / 150.0).max(0.7)));
    let mut face_l: Vec<f32> = fine.iter().zip(scope.iter()).filter(|(_, b)| **b).map(|(&l, _)| l).collect();
    face_l.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let top = face_l.get(face_l.len().saturating_sub(1) * 985 / 1000).copied().unwrap_or(1.0);
    let t = reserve.max(top);
    let cand: Vec<bool> = (0..w * h).map(|i| scope[i] && fine[i] > t).collect();
    let r = (face_short / 60).max(1);
    let opened = dilate_bool(&erode_bool(&cand, w, h, r), w, h, r);
    let min_side = (face_short / 40).max(3);
    let max_side = (face_short / 8).max(min_side + 1);
    keep_regions_between(&opened, w, h, min_side * min_side, max_side * max_side)
}

fn wash_passes_scoped(canvas: &mut Canvas, score: &mut StrokeScore, reference: &RgbImage, source: &RgbImage, n_stages_total: usize, stage_offset: usize, scope: WashScope, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let (w, h) = (source.width() as usize, source.height() as usize);
    let levels = p.armature_levels.max(2) as usize;
    // The face, as a boolean, for the two face scopes. The face mask is feathered; the washes take it at a half.
    let face_in: Option<Vec<bool>> = match scope {
        WashScope::Sheet => None,
        WashScope::Outside(m) | WashScope::Inside(m) => Some(m.to_vec()),
    };
    let inside_face = matches!(scope, WashScope::Inside(_));
    // The face's extent sets the scale its masses are read at (a face is read at face scale, not sheet scale).
    let face_short: usize = match scope {
        WashScope::Inside(m) => bool_bbox(m, w, h).map(|(x0, y0, x1, y1)| (x1 - x0).min(y1 - y0)).unwrap_or(0),
        _ => 0,
    };
    // (A face too small to have planes still runs its stages — empty — so the replay crosses the same boundaries.)
    // The BOOK reads the keyed reference at pixel scale (its masks); the watercolour the source at wash scale.
    // The watercolour reads its masses from the source flattened at wash scale by an EDGE-PRESERVING smooth
    // (the armature's bilateral, no value snap): a plain blur spread every star and lamp into a pale halo that
    // became a light mass — a starry sky washed as blue-white blots. Points smaller than a wash are not a
    // wash; they are the paper's (and a later mark's) business.
    let smoothed;
    let input: &RgbImage = if p.book {
        reference
    } else {
        // A FACE's planes are read at a fiftieth of the face, not of the sheet: at sheet scale a face is one
        // or two masses — a flat blot with no features for the line to sit on.
        // A twentieth of the face: finer, and the planes broke into mottle — a hundred islands, not a handful.
        let r_wash = if inside_face { (face_short as f32 * 0.05).max(1.5) } else { (w.max(h) as f32 * 0.007).max(2.0) };
        smoothed = structure_armature(source, (w.min(h) as f32 / r_wash).round().max(1.0) as u32, 1, 0.0);
        &smoothed
    };
    let luma = luma_map(input);
    // The levels span the PICTURE's value range (its 2nd..98th luma percentiles), not 0..1: a night scene lives
    // in the bottom fifth of the scale, and fixed levels put its whole sky, walls and tracks into one or two
    // masses — the picture's structure vanished. Levels over its own range give a nocturne the same number of
    // washes a daylight picture gets (its brightest lights still reserve the paper by the reserve luma).
    // A FACE's levels span the face's own range: a lit face is a narrow band of the sheet's values, and the
    // sheet's levels gave it one plane. Over its own range the same few levels are its light, half-tone,
    // shadow and dark — the planes a watercolourist paints a face with.
    let (lo, hi) = {
        let mut sorted: Vec<f32> = match &face_in {
            Some(f) if inside_face => luma.iter().zip(f.iter()).filter(|(_, b)| **b).map(|(&l, _)| l).collect(),
            _ => luma.clone(),
        };
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len().max(1);
        let lo = sorted.get(n * 2 / 100).copied().unwrap_or(0.0);
        let hi = sorted.get((n * 98 / 100).min(n - 1)).copied().unwrap_or(1.0);
        if hi - lo < 0.05 { (0.0, 1.0) } else { (lo, hi) }
    };
    let norm = |v: f32| ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    let step = 1.0 / (levels - 1) as f32;
    // Per-pixel level; reserved paper is level `levels` (no wash). A mass is one level AND one hue family
    // (six hue bins, or neutral when the colour is grey): a level's mean colour across a sky and a roof went
    // grey, and a watercolour's washes are its colours.
    // The paper is reserved at WASH scale too — where the smoothed picture is above the reserve luma — so the
    // lights are areas of bare paper, never specks (a pixel-scale reserve left white flecks over everything).
    // The paper is reserved where the picture, smoothed to wash scale, is above the reserve luma — for the book
    // too (its masses are pixel-scale, but its paper must be AREAS: a pixel-scale reserve peppered the print
    // with white specks; an occasional fleck belongs to the style, a rash of them does not).
    let paper_luma: Vec<f32> = if p.book { luma_map(&imageops::blur(input, (w.max(h) as f32 * 0.004).max(1.5))) } else { luma.clone() };
    // The paper the WASHES leave is the same paper the strokes leave. The pass used to test each pixel against
    // the reserve luma on its own, so the shape-aware reserve — no flecks, never the face — governed the brush
    // marks while the washes still left every lit face and lamp as bare paper. Where the painter has built a
    // protect (a new painting always has), that is the paper; the luma test remains for the older path.
    let paper = |i: usize| match (protect, p.reserve, p.from_scratch) {
        (Some(m), _, true) => m.get(i).copied().unwrap_or(false),
        (_, Some(rt), _) => paper_luma[i] > rt,
        (Some(m), None, _) => m.get(i).copied().unwrap_or(false),
        (None, None, _) => false,
    };
    // (The face's own paper — its glints — is in `protect` already, see `face_paper_shapes`; `paper` reads it.)
    // A pixel outside the scope is in no pass at all (above every level, never "≤ lv" nor "== lv").
    const EXCLUDED: u16 = u16::MAX;
    let level: Vec<u16> = (0..w * h)
        .map(|i| {
            let out_of_scope = match &face_in {
                Some(f) => f[i] != inside_face,
                None => false,
            };
            if out_of_scope {
                EXCLUDED
            } else if paper(i) {
                levels as u16
            } else {
                ((norm(luma[i]) / step).round() as usize).min(levels - 1) as u16
            }
        })
        .collect();
    // A face is ONE hue family: skin sits on the red/orange bin boundary, and splitting it there cut every
    // plane in two along a line that is not in the picture.
    let hue_bin: Vec<u8> = input
        .pixels()
        .map(|px| {
            if inside_face {
                return 6;
            }
            let (r, g, b) = (px.0[0] as f32, px.0[1] as f32, px.0[2] as f32);
            let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
            // Chroma RELATIVE to brightness: a dark blue night sky is as much a hue as a bright one.
            if mx - mn < (24.0f32).min(mx * 0.25) {
                return 6;
            }
            let d = mx - mn;
            let hh = if mx == r { ((g - b) / d).rem_euclid(6.0) } else if mx == g { (b - r) / d + 2.0 } else { (r - g) / d + 4.0 };
            (hh.rem_euclid(6.0).floor() as u8).min(5)
        })
        .collect();
    // Each mass is washed ONCE, with its own colour (stacking every lighter level's glaze under a darker mass
    // mixed a sky's blue-grey under a wall's brown — mud); the light-to-dark order still lets a darker wash
    // bloom over a lighter neighbour's edge. A mass is one level and one hue family.
    // GLAZING. A technique lays washes the way a watercolourist does: each wash over everything DARKER than
    // it, so a pixel two steps below the paper carries two transparent layers and the darks are built by the
    // stack. Two things this buys that washing each mass once could not: there are NO GAPS — the lightest
    // glaze of a hue family covers that family's whole region, and the white slivers between neighbouring
    // masses, which were the medium's "white specks", cannot exist — and every wash boundary is a hard edge
    // on dry paper, the drawing of the picture. The mud the `==` form was chosen to avoid came from stacking
    // a sky's grey-blue under a wall's brown; the hue families now keep each family's glazes under its own.
    let glaze = p.book || p.technique != WetTechnique::None;
    let in_pass = |i: usize, lv: usize| if glaze { (level[i] as usize) <= lv } else { (level[i] as usize) == lv };
    let same = |i: usize, j: usize, lv: usize| in_pass(i, lv) && in_pass(j, lv) && hue_bin[i] == hue_bin[j];
    // Every mass is washed, however small: a mass below a size threshold was skipped, and its pixels — a
    // window pane, a fleck of light on a face, a hue island the hue split cut off — stayed bare paper: the
    // white-speck rash. The book (masks at pixel scale) fills down to a few pixels; the watercolour leaves
    // only the tiniest islands to its modelling passes.
    // A face's planes are small by the sheet's measure: its islands are kept down to a few pixels.
    // A face's planes are FEW: an island of a darker plane under a twentieth of the face across is not a
    // plane, and under glazing a dropped island simply stays the lighter plane it sits in — the simplification
    // a watercolourist makes with the brush. (Smaller, and the face broke into mottle.)
    let min_area = if p.book { 6 } else if inside_face { ((face_short / 20).max(2)).pow(2) } else { ((w * h) as f32 * 0.00005).max(24.0) as usize };
    let n = p.palette.pigments.len();
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let n_levels = levels;
    for lv in (0..n_levels).rev() {
        let mut labels = vec![u32::MAX; w * h];
        // Lightest level first: levels count up with luma, so the highest index is the lightest.
        let stage = if inside_face { format!("face-wash-{}", n_levels - lv) } else { format!("wash-{}", n_levels - lv) };
        if let Some(pr) = progress {
            pr(PaintProgress::Painting { pass: stage_offset + n_levels - lv, passes: stage_offset + n_levels, radius: 0.0 });
        }
        // Connected components (4-neighbour) of this level.
        // (pixels, colour sum over the whole component, colour sum over the pixels AT this level, their count)
        let mut comps: Vec<(Vec<usize>, [f64; 3], [f64; 3], usize)> = Vec::new();
        for start in 0..w * h {
            if !in_pass(start, lv) || labels[start] != u32::MAX {
                continue;
            }
            let id = comps.len() as u32;
            let mut px: Vec<usize> = Vec::new();
            let mut sum = [0f64; 3];
            let mut own = [0f64; 3];
            let mut own_n = 0usize;
            let mut stack = vec![start];
            labels[start] = id;
            while let Some(i) = stack.pop() {
                px.push(i);
                let c = input.get_pixel((i % w) as u32, (i / w) as u32).0;
                for ch in 0..3 {
                    sum[ch] += c[ch] as f64;
                }
                if level[i] as usize == lv {
                    for ch in 0..3 {
                        own[ch] += c[ch] as f64;
                    }
                    own_n += 1;
                }
                let (x, y) = (i % w, i / w);
                let mut nb = [usize::MAX; 4];
                if x > 0 { nb[0] = i - 1; }
                if x + 1 < w { nb[1] = i + 1; }
                if y > 0 { nb[2] = i - w; }
                if y + 1 < h { nb[3] = i + w; }
                for &j in &nb {
                    if j != usize::MAX && same(i, j, lv) && labels[j] == u32::MAX {
                        labels[j] = id;
                        stack.push(j);
                    }
                }
            }
            comps.push((px, sum, own, own_n));
        }
        // Largest first (a big wash under, the small ones over it — the order the record replays in).
        comps.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        for (px, sum, own, own_n) in &comps {
            if px.len() < min_area || *placed >= p.budget {
                continue;
            }
            // A GLAZE IS THE COLOUR OF ITS OWN LEVEL. Under glazing a component covers every darker pixel
            // beneath it as well, and averaging over all of them dragged every glaze toward the darks' mean —
            // grey over a sunlit wall, mud over a night. The pixels AT this level are what this layer is;
            // the darker ones get their own, darker glaze on top.
            let (csum, cn) = if glaze && *own_n > 0 { (own, *own_n) } else { (sum, px.len()) };
            let mean: Srgb = [(csum[0] / cn as f64) as u8, (csum[1] / cn as f64) as u8, (csum[2] / cn as f64) as u8];
            let mut mask = vec![false; w * h];
            for &i in px {
                mask[i] = true;
            }
            // Neighbouring washes TOUCH. Two masses traced exactly leave a hairline of paper between them at
            // every family boundary; a watercolourist lets the edge of one wash run into the next, so each
            // mass is grown a couple of pixels before its rings are traced.
            // …but never over the PAPER: the growth closes the hairline between two masses, it does not eat
            // a reserved shape (a face's nose highlight is five pixels wide; two from each side is all of it).
            let mask = if glaze {
                let mut d = dilate_bool(&mask, w, h, 2);
                for (i, m) in d.iter_mut().enumerate() {
                    if level[i] == levels as u16 {
                        *m = false;
                    }
                }
                d
            } else {
                mask
            };
            let rings = mask_rings(&mask, w, h, 1.2);
            if rings.is_empty() {
                continue;
            }
            // A GLAZE, not a coat: one thin charge per pass; the darks are built by the passes stacking.
            // A wash's charge grows with its DARKNESS: a thin transparent wash over white paper cannot read deep,
            // so a night sky washed at a light mass's charge came out grey-blue hatch; a painter floods the
            // darks (several passes of the same colour) and barely tints the lights.
            let darkness = 1.0 - (lv as f32 / (n_levels - 1) as f32);
            // A glaze is thin: the darks come from the stack, not from any one layer flooding.
            let charge = if p.book { 0.2 } else if glaze { 0.22 + 0.45 * darkness } else { 0.4 + 2.4 * darkness * darkness };
            let load = mixture_cached(&mut cache, mean, &p.palette, p.charge * charge, n);
            // A wash's wetness and its wet edge are the technique: flooded and wide-edged on wet paper, a set
            // wash with a rim's worth of edge on dry paper, and nearly dry with no wet edge at all for the
            // dry-brush — there is no standing water for an edge to be made of.
            let (wet, edge_mul) = match p.technique {
                WetTechnique::WetOnWet => (0.95, 2.2),
                WetTechnique::WetOnDry => (0.75, 1.0),
                WetTechnique::DryOnDry => (0.45, 0.0),
                WetTechnique::None => (0.75, 1.0),
            };
            // The wet edge: ~0.3% of the long side (6 px on a 2048 sheet), so a wash meets the next with a rim.
            let feather = ((w.max(h) as f32 * 0.002).max(1.0) * edge_mul).round();
            canvas.fill_rings(&rings, &load, wet, feather);
            *k += 1;
            *placed += 1;
            if let Some(pr) = progress {
                pr(PaintProgress::Placed(*placed));
            }
            let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.clone(), spline: rings, w0: feather, w1: 0.0, taper: 0.0, mix, wet, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None });
        }
        // The pass boundary, as the stroke passes cross it (and as the replay reproduces it).
        // The pass boundary as the replay reproduces it: the taper over EVERY stage of the painting (the washes
        // and the modelling passes after them), then drying when a stage follows.
        let idx = stage_offset + n_levels - lv - 1;
        if p.bleed > 0.0 {
            canvas.bleed_with(p.bleed * pass_bleed_taper(idx, n_stages_total), p.diffuse);
        }
        if idx + 1 < n_stages_total {
            canvas.dry(1.0 - p.dry);
        }
    }
}

/// THE LINE ON A WATERCOLOUR FACE (see `face_is_planes`). The planes carry the light; the likeness is a
/// drawn line on the eyes, brows, nose, lips and jaw — traced from the SOURCE's own edges over the face
/// (the armature has simplified them away), in a coloured ink mixed from the palette: the face's colour
/// under the line, darkened. The face is cut out with a margin and planned on its own, so the chains are
/// kept by the face's measure (a chain a sixth of a face long is a feature; by the sheet's measure it is
/// nothing) and the sheet's edges cost nothing.
fn face_line_pass(canvas: &mut Canvas, score: &mut StrokeScore, source: &RgbImage, face: &[bool], p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64) {
    let (w, h) = (source.width() as usize, source.height() as usize);
    let Some((x0, y0, x1, y1)) = bool_bbox(face, w, h) else { return };
    let short = (x1 - x0).min(y1 - y0);
    if short < 16 {
        return;
    }
    let margin = short / 8;
    let (cx0, cy0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
    let (cx1, cy1) = ((x1 + margin).min(w), (y1 + margin).min(h));
    let (cw, ch) = (cx1 - cx0, cy1 - cy0);
    let crop = imageops::crop_imm(source, cx0 as u32, cy0 as u32, cw as u32, ch as u32).to_image();
    // Lightly smoothed, as the luminous line is: pores and grain are not edges.
    let planned = imageops::blur(&crop, (short as f32 / 300.0).max(0.6));
    // The line may sit a little beyond the planes — the jaw's edge is where the skin ends.
    let grown = dilate_bool(face, w, h, (short / 40).max(1));
    let crop_mask: Vec<f32> = (0..cw * ch).map(|i| if grown[(cy0 + i / cw) * w + cx0 + i % cw] { 1.0 } else { 0.0 }).collect();
    let drawing = crate::paint::ink::plan_detailed(&planned, Some((&crop, &crop_mask)), p.contour.max(0.35), p.budget.saturating_sub(*placed), p.seed, crate::paint::ink::HatchStyle::NONE);
    let n = p.palette.pigments.len();
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let mut coloured_load = |c: Srgb| -> Vec<f32> {
        let lin = color::srgb_to_linear(c);
        let dark = color::linear_to_srgb([lin[0] * 0.3, lin[1] * 0.3, lin[2] * 0.3]);
        mixture_cached(&mut cache, dark, &p.palette, p.charge * 1.6, n)
    };
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.bristles = 1;
    brush.streak = 0.1;
    brush.round = 0.9;
    // The line's weight is the sheet's (a pen line on a 2048 sheet), not the crop's.
    let width = (w.max(h) as f32 / 800.0).clamp(1.0, 2.5);
    let blocked = |pt: &[f32; 2]| protect.map(|m| m.get(pt[1] as usize * w + pt[0] as usize).copied().unwrap_or(false)).unwrap_or(false);
    for poly in &drawing.contours {
        if *placed >= p.budget {
            break;
        }
        if poly.len() < 2 {
            continue;
        }
        let mid = poly[poly.len() / 2];
        let (mx, my) = ((mid[0].max(0.0) as usize).min(cw - 1), (mid[1].max(0.0) as usize).min(ch - 1));
        // Only the face's own chains: the margin is there so a jaw line is not cut, not to draw the collar.
        if crop_mask[my * cw + mx] <= 0.5 {
            continue;
        }
        let shifted: Vec<[f32; 2]> = poly.iter().map(|q| [q[0] + cx0 as f32, q[1] + cy0 as f32]).collect();
        if blocked(&shifted[shifted.len() / 2]) {
            continue;
        }
        *k += 1;
        let path = waver_path(&shifted, 0.15 * width, p.seed, *k);
        let c = crop.get_pixel(mx as u32, my as u32).0;
        let load_here = coloured_load(c);
        let wet = 0.2;
        let mix: Vec<(String, f32)> = load_here.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
        let s = Stroke { path, width0: width, width1: width, load: load_here, pressure: 0.9, wetness: wet };
        s.rasterize(canvas, &brush);
        *placed += 1;
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "face-line".into(), spline: s.path, w0: width, w1: width, taper: 0.15, mix, wet, press: 0.9, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles) });
    }
}

/// The boundary RINGS of a pixel mask (outer contours and holes), as polygons on the pixel lattice, simplified
/// to `tol` px, separated by `[NaN, NaN]` for `Canvas::fill_rings`. Each boundary edge between an inside and an
/// outside pixel is a unit segment; the segments chain into closed loops.
fn mask_rings(mask: &[bool], w: usize, h: usize, tol: f32) -> Vec<[f32; 2]> {
    let inside = |x: i64, y: i64| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && mask[y as usize * w + x as usize];
    // Directed edges (a → b), keyed by their start corner, so that the region is always on the LEFT — then
    // following edges from corner to corner traces every ring exactly once.
    let key = |x: i64, y: i64| (y * (w as i64 + 1) + x) as usize;
    let mut next: std::collections::HashMap<usize, Vec<(i64, i64)>> = std::collections::HashMap::new();
    let mut count = 0usize;
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            if !inside(x, y) {
                continue;
            }
            // Top edge (left→right) when the pixel above is outside; right edge (top→bottom) when the right
            // neighbour is outside; bottom (right→left) when below is outside; left (bottom→top) when left is.
            if !inside(x, y - 1) { next.entry(key(x, y)).or_default().push((x + 1, y)); count += 1; }
            if !inside(x + 1, y) { next.entry(key(x + 1, y)).or_default().push((x + 1, y + 1)); count += 1; }
            if !inside(x, y + 1) { next.entry(key(x + 1, y + 1)).or_default().push((x, y + 1)); count += 1; }
            if !inside(x - 1, y) { next.entry(key(x, y + 1)).or_default().push((x, y)); count += 1; }
        }
    }
    let mut out: Vec<[f32; 2]> = Vec::new();
    let mut used = 0usize;
    // Deterministic start order: scan corners in raster order.
    let mut starts: Vec<usize> = next.keys().copied().collect();
    starts.sort_unstable();
    for start_key in starts {
        while let Some(first) = next.get_mut(&start_key).and_then(|v| v.pop()) {
            used += 1;
            let sx = (start_key % (w + 1)) as i64;
            let sy = (start_key / (w + 1)) as i64;
            let mut ring: Vec<(i64, i64)> = vec![(sx, sy), first];
            let mut cur = first;
            while cur != (sx, sy) {
                let Some(v) = next.get_mut(&key(cur.0, cur.1)) else { break };
                // At a corner where four edges meet (two rings touch at a point) pick the turn that keeps the
                // region on the left: the candidate that turns right-most first, else any.
                let Some(nxt) = v.pop() else { break };
                used += 1;
                ring.push(nxt);
                cur = nxt;
            }
            if ring.len() < 4 {
                continue;
            }
            ring.pop(); // the closing repeat of the start
            let pts: Vec<[f32; 2]> = ring.iter().map(|&(x, y)| [x as f32, y as f32]).collect();
            let simp = simplify_ring(&pts, tol);
            if simp.len() >= 3 {
                if !out.is_empty() {
                    out.push([f32::NAN, f32::NAN]);
                }
                out.extend_from_slice(&simp);
            }
        }
    }
    debug_assert!(used == count, "every boundary edge belongs to a ring ({used} of {count})");
    out
}

/// Douglas–Peucker on a closed ring (split at its two farthest-apart points so the ends are stable).
fn simplify_ring(pts: &[[f32; 2]], tol: f32) -> Vec<[f32; 2]> {
    if pts.len() <= 4 || tol <= 0.0 {
        return pts.to_vec();
    }
    // Split at index 0 and the point farthest from it.
    let far = (1..pts.len()).max_by(|&a, &b| dist2(pts[0], pts[a]).partial_cmp(&dist2(pts[0], pts[b])).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(pts.len() / 2);
    let mut out = Vec::new();
    dp(&pts[0..=far], tol, &mut out);
    out.pop();
    let mut second: Vec<[f32; 2]> = pts[far..].to_vec();
    second.push(pts[0]);
    let mut o2 = Vec::new();
    dp(&second, tol, &mut o2);
    o2.pop();
    out.extend(o2);
    out
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn dp(pts: &[[f32; 2]], tol: f32, out: &mut Vec<[f32; 2]>) {
    if pts.len() < 3 {
        out.extend_from_slice(pts);
        return;
    }
    let (a, b) = (pts[0], pts[pts.len() - 1]);
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let mut worst = 0.0f32;
    let mut wi = 0usize;
    for (i, q) in pts.iter().enumerate().skip(1).take(pts.len() - 2) {
        let d = ((q[0] - a[0]) * dy - (q[1] - a[1]) * dx).abs() / len;
        if d > worst {
            worst = d;
            wi = i;
        }
    }
    if worst > tol {
        dp(&pts[..=wi], tol, out);
        out.pop();
        dp(&pts[wi..], tol, out);
    } else {
        out.push(a);
        out.push(b);
    }
}

fn ink_contours(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, source: &RgbImage, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64) {
    let w = input.width() as usize;
    // A luminous medium draws its pen line from the SOURCE, lightly smoothed (texture off, the machine's and
    // the figures' edges kept) — the washes' armature has simplified exactly what the line is for.
    let from_source;
    let plan_on: &RgbImage = if p.luminous && !std::ptr::eq(source, input) {
        from_source = imageops::blur(source, (input.width().max(input.height()) as f32 / 1000.0).max(0.8));
        &from_source
    } else {
        input
    };
    let drawing = crate::paint::ink::plan_with(plan_on, p.contour.max(0.3), p.budget.saturating_sub(*placed), p.seed, crate::paint::ink::HatchStyle::NONE);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let n = p.palette.pigments.len();
    let mut load = vec![0f32; n];
    load[ink] = p.charge * 1.6;
    // A luminous medium's line is a COLOURED ink: the reference's colour under the line, darkened, mixed from
    // the palette — a brown line on the iron, a blue-grey one in a shadow — never a black outline.
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let mut coloured_load = |c: Srgb| -> Vec<f32> {
        let lin = color::srgb_to_linear(c);
        let dark = color::linear_to_srgb([lin[0] * 0.3, lin[1] * 0.3, lin[2] * 0.3]);
        mixture_cached(&mut cache, dark, &p.palette, p.charge * 1.6, n)
    };
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.bristles = 1;
    brush.streak = 0.1;
    brush.round = 0.9;
    let width = (drawing.contour_width * 0.8).max(0.8);
    let blocked = |pt: &[f32; 2]| protect.map(|m| m.get(pt[1] as usize * w + pt[0] as usize).copied().unwrap_or(false)).unwrap_or(false);
    for poly in &drawing.contours {
        if *placed >= p.budget {
            break;
        }
        if poly.len() < 2 || blocked(&poly[poly.len() / 2]) {
            continue;
        }
        *k += 1;
        let path = waver_path(poly, 0.15 * width, p.seed, *k);
        let (load_here, wet) = if p.luminous {
            let m = poly[poly.len() / 2];
            let c = plan_on.get_pixel((m[0].max(0.0) as u32).min(plan_on.width() - 1), (m[1].max(0.0) as u32).min(plan_on.height() - 1)).0;
            (coloured_load(c), 0.2)
        } else {
            (load.clone(), 0.1)
        };
        let mix: Vec<(String, f32)> = load_here.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
        let s = Stroke { path, width0: width, width1: width, load: load_here, pressure: 0.9, wetness: wet };
        s.rasterize(canvas, &brush);
        *placed += 1;
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "contour".into(), spline: s.path, w0: width, w1: width, taper: 0.15, mix, wet, press: 0.9, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles) });
    }
}

fn ink_drawing(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, source: &RgbImage, head: Option<&[f32]>, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let w = input.width() as usize;
    let style = if p.brush_drawing { crate::paint::ink::HatchStyle::SUMI } else if p.engrave { crate::paint::ink::HatchStyle::ENGRAVING } else { crate::paint::ink::HatchStyle::PEN };
    // The head is DRAWN from the source's full detail (the armature has simplified the face away).
    let detail = head.map(|m| (source, m));
    let drawing = crate::paint::ink::plan_detailed(input, detail, p.contour, p.budget.saturating_sub(*placed), p.seed, style);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let mut load = vec![0f32; p.palette.pigments.len()];
    load[ink] = p.charge * 2.0; // a pen lays full ink
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    // A pen is one stiff point; a sumi brush is a soft loaded tuft (many bristles, a little streak, wet).
    if p.brush_drawing {
        brush.bristles = 7;
        brush.streak = 0.25;
        brush.round = 0.85;
    } else {
        brush.bristles = 1;
        brush.streak = 0.0;
        brush.round = 1.0;
    }
    let wet = if p.brush_drawing { 0.6 } else { 0.0 };
    let blocked = |pt: &[f32; 2]| protect.map(|m| m.get(pt[1] as usize * w + pt[0] as usize).copied().unwrap_or(false)).unwrap_or(false);
    let lines = drawing.contours.iter().map(|c| ("contour".to_string(), drawing.contour_width, c)).chain(drawing.hatch.iter().enumerate().map(|(i, (li, s))| (format!("hatch-{li}"), drawing.hatch_widths.as_ref().map(|v| v[i]).unwrap_or(drawing.hatch_width), s)));
    for (stage, width, poly) in lines {
        if *placed >= p.budget {
            break;
        }
        if poly.len() < 2 || blocked(&poly[poly.len() / 2]) {
            continue;
        }
        *k += 1;
        let path = waver_path(poly, 0.12 * width, p.seed, *k);
        // A brush stroke tapers to its lift-off; a pen line does not.
        let w1 = if p.brush_drawing { width * 0.35 } else { width };
        let s = Stroke { path, width0: width, width1: w1, load: load.clone(), pressure: 1.0, wetness: wet };
        s.rasterize(canvas, &brush);
        *placed += 1;
        if let Some(pr) = progress {
            if *placed % 64 == 0 {
                pr(PaintProgress::Placed(*placed));
            }
        }
        score.strokes.push(StrokeRecord {
            id: *placed as u32,
            wipe: false, wash: false,
            stage,
            spline: s.path,
            w0: width,
            w1: w1,
            taper: 0.15,
            mix: vec![(p.palette.pigments[ink].name.to_string(), load[ink])],
            wet,
            press: 1.0,
            streak: brush.streak,
            round: brush.round,
            pickup: None,
            bristles: Some(brush.bristles),
        });
    }
}

/// Traceability (RFC PAINT-1 §12.1): the mean linear-luma correlation between a painted image and its
/// reference at full resolution. High (→1) means the painting traced the reference (a filter); a real painting
/// keeps the structure but invents the surface, so it correlates moderately, not perfectly.
pub fn traceability(painted: &RgbImage, reference: &RgbImage) -> f32 {
    if painted.dimensions() != reference.dimensions() {
        return f32::NAN;
    }
    let a = luma_map(painted);
    let b = luma_map(reference);
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    fn gradient_img(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, _y| {
            let t = x as f32 / w as f32;
            image::Rgb([(40.0 + 180.0 * t) as u8, (60.0 + 120.0 * t) as u8, (30.0 + 60.0 * t) as u8])
        })
    }

    #[test]
    fn painter_respects_the_budget_and_is_deterministic() {
        let img = gradient_img(64, 48);
        let mut p = PaintParams::new(palette::EARTH, 120);
        p.brush_sizes = vec![16.0, 8.0];
        let a = paint_from_image(&img, &p);
        assert!(a.strokes <= 120, "budget honoured ({} ≤ 120)", a.strokes);
        assert!(a.strokes > 0, "some strokes laid");
        let b = paint_from_image(&img, &p);
        assert_eq!(a.canvas.to_image().into_raw(), b.canvas.to_image().into_raw(), "deterministic");
    }

    #[test]
    fn reserve_keeps_bright_cells_as_paper() {
        // A gradient dark→light; reserving above the mid value leaves the bright right side as bare paper.
        // The gradient's LINEAR luma runs ~0.15 → 0.47 (x=70 is 0.39, x=78 is 0.47), so the threshold must sit
        // inside that range — the old 0.55 was above the image's brightest value and reserved NOTHING; the
        // assertion then only measured the solver painting a light orange as near-white over the ground, a
        // 1-LSB accident that any colour-model change flipped. At 0.40 the right side is genuinely reserved:
        // no stroke seeds there, and growth from the mid-tones terminates at the reserve (the protect), so
        // column 78 is untouched ground — the paper itself, regardless of the colour model.
        let img = gradient_img(80, 30);
        let mut p = PaintParams::new(palette::SUMI, 400);
        p.brush_sizes = vec![10.0];
        p.reserve = Some(0.40);
        let out = paint_from_image(&img, &p).canvas.to_image();
        // The brightest column stays paper white (reserved); the dark left gets painted.
        let right = out.get_pixel(78, 15).0;
        assert!(right[0] > 235 && right[1] > 235, "bright side reserved (paper): {right:?}");
    }

    #[test]
    fn density_marks_build_value_by_count() {
        // Pen-ink density over a dark→light gradient: the dark side accumulates more black marks (lower luma)
        // than the light side.
        let img = gradient_img(80, 40);
        let mut p = PaintParams::new(palette::SUMI, 4000);
        p.brush_sizes = vec![8.0];
        p.density = true;
        let out = paint_from_image(&img, &p).canvas.to_image();
        // Density is statistical — average over a patch on each side rather than sampling one pixel.
        let patch_mean = |x0: u32, x1: u32| -> f32 {
            let mut s = 0.0;
            let mut n = 0.0;
            for y in 8..32 {
                for x in x0..x1 {
                    s += out.get_pixel(x, y).0[0] as f32;
                    n += 1.0;
                }
            }
            s / n
        };
        let dark = patch_mean(2, 18);
        let light = patch_mean(62, 78);
        // The fixture's light side is a 0.68 value (one hatch layer), its dark side 0.24 (three, cross-hatched).
        assert!(dark < light - 30.0, "hatch tone: the dark side is cross-hatched darker ({dark:.0} vs {light:.0})");
        assert!(light > 160.0, "a 0.68 value carries a single open hatch, not a cross-hatch ({light:.0})");
    }

    #[test]
    fn critic_rejects_passes_that_dont_improve() {
        let img = gradient_img(48, 48);
        let mut p = PaintParams::new(palette::EARTH, 300);
        p.brush_sizes = vec![16.0, 8.0, 4.0]; // → passes: block-in, restate, restate
        // A FLAT critic: no pass ever improves the score, so every non-block-in pass is rolled back.
        let flat = |_img: &RgbImage| 0.5f32;
        let r = paint_critiqued(&img, &p, &flat, 0.01);
        assert_eq!(r.rejected.len(), 2, "the two restate passes rejected; the block-in is never rejected");
        // An IMPROVING critic (rewards coverage): passes are kept.
        let rewarding = |img: &RgbImage| img.pixels().map(|px| 255 - px.0[0] as i32).sum::<i32>() as f32;
        let r2 = paint_critiqued(&img, &p, &rewarding, 0.0);
        assert!(r2.rejected.len() < 2, "passes that improve the score are kept ({} rejected)", r2.rejected.len());
    }

    #[test]
    fn focal_edge_paints_with_a_region_mask() {
        // Plumbing: painting with a region mask (subject vs ground) completes and lays strokes; the growth
        // termination is the same machinery as the tested `protect`.
        let img = gradient_img(60, 60);
        let mut p = PaintParams::new(palette::EARTH, 500);
        p.brush_sizes = vec![10.0];
        let mut mask = vec![false; 60 * 60];
        for y in 0..60 {
            for x in 0..30 {
                mask[y * 60 + x] = true; // left half = subject
            }
        }
        p.region_mask = Some(mask);
        assert!(paint_from_image(&img, &p).strokes > 0, "paints with a region mask");
    }

    #[test]
    fn negative_painting_leaves_the_protected_shape() {
        // Paint around a protected central square — the shape stays (near) the ground while the surround is
        // painted, defining the shape by negative space.
        let img = gradient_img(60, 60);
        let mut p = PaintParams::new(palette::EARTH, 600);
        p.brush_sizes = vec![8.0];
        let mut mask = vec![false; 60 * 60];
        for y in 22..38 {
            for x in 22..38 {
                mask[y * 60 + x] = true;
            }
        }
        p.protect = Some(mask);
        let c = paint_from_image(&img, &p).canvas;
        // The protected centre is (near) the bare white ground; a surround pixel is painted (has height).
        assert!(c.height[30 * 60 + 30] < 0.05, "protected shape left unpainted");
        assert!(c.height[30 * 60 + 5] > 0.0 || c.height[10 * 60 + 30] > 0.0, "the surround is painted");
    }

    #[test]
    fn composition_layers_occlude_and_mask() {
        // A composition: paint a LIGHT background full-frame, then paint a DARK element ONTO it, masked to the
        // right half. The right half darkens (occlusion); the left half stays light (mask respected).
        let light = image::RgbImage::from_pixel(48, 48, image::Rgb([210, 205, 195]));
        let dark = image::RgbImage::from_pixel(48, 48, image::Rgb([25, 22, 20]));
        let mut bg = PaintParams::new(palette::ZORN, 4000);
        bg.brush_sizes = vec![6.0, 3.0];
        let base = paint_from_image(&light, &bg).canvas;
        // Foreground element: dark, right half only.
        let mut fg = bg.clone();
        fg.seed = 7;
        let mut mask = vec![false; 48 * 48];
        for y in 0..48 {
            for x in 24..48 {
                mask[y * 48 + x] = true;
            }
        }
        fg.paint_mask = Some(mask);
        let out = paint_onto(base, &dark, &fg).canvas;
        // Patch means (avoids per-pixel coverage gaps): the masked right half is much darker; the left half
        // stayed light (the element painted only its footprint, occluding what was beneath).
        let patch = |x0: u32, x1: u32| -> f32 {
            let (mut s, mut n) = (0.0, 0.0);
            for y in 8..40 {
                for x in x0..x1 {
                    s += color::linear_luma(color::srgb_to_linear(out.color_at(x, y)));
                    n += 1.0;
                }
            }
            s / n
        };
        let left = patch(4, 20);
        let right = patch(28, 44);
        assert!(right < 0.5, "masked element occluded the right half dark (mean {right})");
        assert!(left > right + 0.3, "outside the mask stayed light ({left} vs {right})");
    }

    #[test]
    fn a_low_res_armature_paints_from_structure_not_detail() {
        // A detailed reference (fine checker over a gradient). Painting from a low-res ARMATURE keeps the broad
        // structure (still correlates) but the finest checker detail is gone, so it traces no MORE than painting
        // the full-resolution reference. (With the coherent structure-tensor flow field, full-res also declines
        // to chase per-pixel checker noise, so the two converge — the invariant is that coarsening adds no
        // spurious detail, i.e. the armature never traces *substantially* more than full-res.)
        // A reference with real STRUCTURE — two big value masses (dark left / light right) plus a fine checker
        // texture. The structure-preserving armature keeps the mass STRUCTURE (the boundary) while dropping the
        // checker TEXTURE, so it paints the broad structure and never traces the fine checker.
        let img = image::RgbImage::from_fn(96, 96, |x, y| {
            // Two big VALUE masses (dark left / light right) — the structure lives in LUMA, so all channels scale
            // with the mass (a warm tint), not just one. A single-channel mass isn't a value mass: its luma is
            // dominated by the other, constant channels. A fine checker texture rides on top.
            let mass = if x < 48 { 55u8 } else { 195u8 };
            let checker = if (x / 3 + y / 3) % 2 == 0 { 24 } else { 0 };
            let v = mass.saturating_add(checker);
            image::Rgb([v, (v as f32 * 0.85) as u8, (v as f32 * 0.7) as u8])
        });
        let mut full = PaintParams::new(palette::EARTH, 400);
        full.brush_sizes = vec![18.0, 9.0];
        let mut arm = full.clone();
        arm.armature_side = Some(24);
        let tr_full = traceability(&paint_from_image(&img, &full).canvas.to_image(), &img);
        let tr_arm = traceability(&paint_from_image(&img, &arm).canvas.to_image(), &img);
        // The structure-preserving armature keeps the value-mass structure (the boundary survives), so it paints
        // the broad structure — it never LOSES it into a smear, and never traces the fine checker.
        assert!(tr_arm > 0.1, "the armature paints the broad value-mass structure (corr {tr_arm})");
        assert!(tr_full > 0.01, "full-res still paints some structure ({tr_full})");
    }

    #[test]
    fn the_score_faithfully_records_the_paint() {
        // Replaying a paint's own score at native size reproduces the canvas byte-for-byte — the score IS the
        // painting (A4).
        let img = gradient_img(64, 48);
        let mut p = PaintParams::new(palette::EARTH, 150);
        p.brush_sizes = vec![16.0, 8.0];
        let result = paint_from_image(&img, &p);
        let painted = result.canvas.to_image().into_raw();
        let replayed = result.score.replay(64, 48).unwrap().to_image().into_raw();
        assert_eq!(painted, replayed, "score replay == the original paint at native size");
        assert_eq!(result.score.strokes.len(), result.strokes, "one record per stroke laid");
    }

    #[test]
    fn the_thread_count_never_changes_the_picture() {
        // Mixing runs up front on several threads; one painter lays the classic order. Any count, same bytes,
        // and the score replays byte-exact.
        let img = gradient_img(160, 120);
        let mut p = PaintParams::new(palette::EARTH, 3000);
        p.brush_sizes = vec![12.0, 6.0, 3.0];
        p.min_brush = 2.0;
        p.bleed = 0.3;
        p.dry = 0.5;
        p.threads = 1;
        let a = paint_from_image(&img, &p);
        p.threads = 3;
        let b = paint_from_image(&img, &p);
        assert!(a.strokes > 200, "a real number of strokes ({})", a.strokes);
        assert_eq!(a.canvas.to_image().into_raw(), b.canvas.to_image().into_raw(), "the thread count does not change the picture");
        assert_eq!(a.score.strokes.len(), b.score.strokes.len());
        let replayed = a.score.replay(160, 120).unwrap().to_image().into_raw();
        assert_eq!(a.canvas.to_image().into_raw(), replayed, "replays byte-exact from its score");
    }

    /// Diagnostic, run by hand: `PLAKAT_DIAG_IMG=in.png PLAKAT_DIAG_OUT=dir cargo test --lib dump_armature -- --ignored`
    /// writes the structure armature at gradation 0 and 1 (side 210, 8 levels) for a look at the value masses.
    #[test]
    #[ignore]
    fn dump_armature_for_a_look() {
        let (Ok(inp), Ok(out)) = (std::env::var("PLAKAT_DIAG_IMG"), std::env::var("PLAKAT_DIAG_OUT")) else { return };
        let img = image::open(inp).unwrap().to_rgb8();
        for g in [0.0f32, 1.0] {
            structure_armature(&img, 210, 8, g).save(format!("{out}/armature_g{g}.png")).unwrap();
        }
        for side in [64u32, 96, 200, 470] {
            coarse_armature(&img, side, 8).save(format!("{out}/coarse_{side}.png")).unwrap();
        }
    }

    #[test]
    fn mask_rings_trace_an_outer_ring_and_a_hole() {
        // A 12×12 square with a 4×4 hole: two rings, filling them back reproduces the mask exactly.
        let (w, h) = (16usize, 16usize);
        let mask: Vec<bool> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); (2..14).contains(&x) && (2..14).contains(&y) && !((6..10).contains(&x) && (6..10).contains(&y)) }).collect();
        let rings = mask_rings(&mask, w, h, 0.0);
        assert_eq!(rings.iter().filter(|p| p[0].is_nan()).count(), 1, "one separator: two rings");
        let mut c = Canvas::white(w as u32, h as u32, palette::ZORN, 0.7);
        c.fill_rings(&rings, &[0.0, 0.0, 3.0, 0.0], 0.5, 0.0);
        for i in 0..w * h {
            let dark = c.color_at((i % w) as u32, (i / w) as u32)[0] < 120;
            assert_eq!(dark, mask[i], "pixel {i} ({}, {})", i % w, i / w);
        }
    }

    #[test]
    fn a_luminous_paint_lays_washes_and_replays_byte_exact() {
        let img = gradient_img(96, 64);
        let mut p = PaintParams::new(palette::EARTH, 2000);
        p.brush_sizes = vec![12.0, 6.0];
        p.luminous = true;
        p.brush_sizes = vec![12.0, 6.0, 3.0];
        p.min_brush = 2.0;
        // (The contour pass is left out: a contour stroke's one-bristle pen is not in its record — a
        // pre-existing replay gap of the line media, not of the washes under test here.)
        p.draw_contours = false;
        p.armature_levels = 4;
        p.bleed = 0.3;
        p.dry = 1.0;
        let r = paint_from_image(&img, &p);
        let n_wash = r.score.strokes.iter().filter(|s| s.wash).count();
        assert!(n_wash > 0, "washes were laid");
        assert!(r.score.strokes.iter().any(|s| !s.wash), "…and modelling strokes within them");
        let painted = r.canvas.to_image().into_raw();
        assert_eq!(r.score.replay(96, 64).unwrap().to_image().into_raw(), painted, "washes replay byte-exact from the score");
        // The text form carries every wash (its rings, NaN separators included) and parses back.
        let back = StrokeScore::parse(&r.score.to_text()).unwrap();
        assert_eq!(back.strokes.iter().filter(|s| s.wash).count(), n_wash);
        assert_eq!(back.strokes.iter().find(|s| s.wash).map(|s| s.spline.len()), r.score.strokes.iter().find(|s| s.wash).map(|s| s.spline.len()));
    }

    #[test]
    fn fill_spends_more_of_the_budget_and_replays_byte_exact() {
        let img = gradient_img(96, 64);
        let mut p = PaintParams::new(palette::EARTH, 4000);
        p.brush_sizes = vec![12.0, 6.0, 3.0];
        p.min_brush = 2.0;
        p.bleed = 0.3;
        p.dry = 0.5;
        let base = paint_from_image(&img, &p);
        p.fill = 0.9;
        let filled = paint_from_image(&img, &p);
        assert!(filled.strokes > base.strokes, "fill lays more strokes: {} vs {}", filled.strokes, base.strokes);
        assert!(filled.strokes <= 4000);
        let painted = filled.canvas.to_image().into_raw();
        assert_eq!(filled.score.replay(96, 64).unwrap().to_image().into_raw(), painted, "a filled paint replays byte-exact");
        assert!(filled.score.header.stages.as_ref().map(|st| st.len()).unwrap_or(0) == 3, "fill adds no stage of its own");
    }

    #[test]
    fn a_coarse_armature_has_no_detail_to_trace() {
        // Pixel-scale detail (a checker texture over a ramp) is gone from the armature; the ramp's structure stays.
        let img = RgbImage::from_fn(256, 256, |x, y| {
            let base = 60.0 + 120.0 * (x as f32 / 256.0);
            let tex = if (x + y) % 2 == 0 { 30.0 } else { -30.0 };
            let v = (base + tex).clamp(0.0, 255.0) as u8;
            image::Rgb([v, v, v])
        });
        let arm = coarse_armature(&img, 32, 8);
        let lap = |im: &RgbImage| {
            let mut acc = 0.0f64;
            for y in 1..255u32 {
                for x in 1..255u32 {
                    let c = im.get_pixel(x, y).0[0] as f64;
                    let nb = im.get_pixel(x - 1, y).0[0] as f64 + im.get_pixel(x + 1, y).0[0] as f64 + im.get_pixel(x, y - 1).0[0] as f64 + im.get_pixel(x, y + 1).0[0] as f64;
                    acc += (4.0 * c - nb).abs();
                }
            }
            acc / (254.0 * 254.0)
        };
        assert!(lap(&arm) < lap(&img) * 0.05, "the texture is gone: {} vs {}", lap(&arm), lap(&img));
        let (l, r) = (arm.get_pixel(20, 128).0[0] as i32, arm.get_pixel(236, 128).0[0] as i32);
        assert!(r - l > 60, "the ramp survives as structure: {l} → {r}");
    }

    #[test]
    fn a_blown_highlight_is_re_modelled_into_a_dome_and_a_good_one_is_left_alone() {
        // A flat bright disc on a mid ground — the shape the armature leaves of a specular highlight, and
        // what reads as a hole cut in the picture. The pass must give it a falloff while KEEPING it bright:
        // the highlight is where the light is.
        let disc = |flat: bool| {
            image::RgbImage::from_fn(128, 128, |x, y| {
                let d = (((x as f32 - 64.0).powi(2) + (y as f32 - 64.0).powi(2)).sqrt() / 22.0).min(1.0);
                let v = if d >= 1.0 {
                    90.0
                } else if flat {
                    245.0
                } else {
                    // Already modelled: a cosine dome from the same peak to the same ground.
                    90.0 + 155.0 * 0.5 * (1.0 + (d * std::f32::consts::PI).cos())
                };
                image::Rgb([v as u8, v as u8, v as u8])
            })
        };
        let spread = |img: &image::RgbImage, p: &PaintParams| {
            let out = paint_from_image(img, p).canvas.to_image();
            let v: Vec<f32> = (0..128u32)
                .flat_map(|y| (0..128u32).map(move |x| (x, y)))
                .filter(|(x, y)| ((*x as f32 - 64.0).powi(2) + (*y as f32 - 64.0).powi(2)).sqrt() < 20.0)
                .map(|(x, y)| out.get_pixel(x, y).0[0] as f32)
                .collect();
            let m = v.iter().sum::<f32>() / v.len() as f32;
            ((v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt(), m)
        };
        let mut p = PaintParams::new(palette::EARTH, 30_000);
        p.brush_sizes = vec![16.0, 8.0, 4.0];
        p.min_brush = 3.0;
        let (flat_off, mean_off) = spread(&disc(true), &p);
        p.hotspot = 1.0;
        let (flat_on, mean_on) = spread(&disc(true), &p);
        // Measured at ~12.2 → ~14.6 on this disc. The brushwork already varies a plateau a little, so the
        // pass adds to a non-zero floor rather than creating the whole falloff.
        assert!(flat_on > flat_off + 1.5, "the plateau gains a falloff ({flat_off:.1} → {flat_on:.1})");
        assert!(mean_on > 110.0, "and stays a highlight, not a hole ({mean_on:.0})");

        // A highlight the brushwork already modelled is not a plateau, and must be left as it is.
        let (dome_off, _) = { p.hotspot = 0.0; spread(&disc(false), &p) };
        let (dome_on, _) = { p.hotspot = 1.0; spread(&disc(false), &p) };
        assert!((dome_on - dome_off).abs() < flat_on - flat_off, "a modelled highlight is barely touched ({dome_off:.1} → {dome_on:.1})");
    }

    #[test]
    fn the_sliding_local_range_matches_a_plain_rescan() {
        // The deque form must be the SAME answer, not merely a close one: the flow field, the detail gate and
        // the posterise edge test all read it, so any drift here moves every stroke in the picture.
        fn naive(luma: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
            let mut out = vec![0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (mut mx, mut mn) = (f32::MIN, f32::MAX);
                    for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                        for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                            let v = luma[yy * w + xx];
                            mx = mx.max(v);
                            mn = mn.min(v);
                        }
                    }
                    out[y * w + x] = mx - mn;
                }
            }
            out
        }
        let (w, h) = (37usize, 23usize);
        let luma: Vec<f32> = (0..w * h).map(|i| jitter(0xA5A5, i as u64) + 0.5).collect();
        // Radii either side of the dimensions, so the clipped ends and the whole-line case are both covered.
        for r in [0usize, 1, 3, 11, 22, 40] {
            assert_eq!(local_range(&luma, w, h, r), naive(&luma, w, h, r), "radius {r}");
        }
    }

    #[test]
    fn the_reserve_keeps_shapes_and_drops_flecks() {
        // A watercolourist reserves SHAPES — the few large simple light areas — never scattered bright pixels.
        // One big light block, one 2-px fleck, and a ragged 1-px spur on the block: the block survives with
        // its spur cleaned off, the fleck is gone.
        let (w, h) = (96usize, 96usize);
        let mut m = vec![false; w * h];
        for y in 20..60 {
            for x in 20..60 {
                m[y * w + x] = true;
            }
        }
        for x in 60..75 {
            m[40 * w + x] = true; // the spur, one pixel tall
        }
        m[80 * w + 80] = true; // the fleck
        m[80 * w + 81] = true;
        let opened = dilate_bool(&erode_bool(&m, w, h, 2), w, h, 2);
        let kept = keep_large_regions(&opened, w, h, 12 * 12);
        assert!(kept[40 * w + 40], "the block is kept");
        assert!(!kept[40 * w + 70], "the one-pixel spur is opened off");
        assert!(!kept[80 * w + 80], "the fleck is dropped");
        let area = kept.iter().filter(|&&b| b).count();
        assert!((1400..=1600).contains(&area), "the block keeps its size, not more and not less: {area}");
    }

    #[test]
    fn leaks_run_downward_from_a_wet_wash_and_replay_byte_exact() {
        // One dark mass on paper, leaking: the runs must start on the mass's lower edge and go DOWN, and the
        // dry brush must leak nothing because it had no water to run.
        let img = image::RgbImage::from_fn(96, 96, |x, y| {
            let inside = (20..76).contains(&x) && (16..52).contains(&y);
            let v = if inside { 60u8 } else { 235 };
            image::Rgb([v, v / 2 + 20, v])
        });
        let run = |t: WetTechnique| {
            let mut p = PaintParams::new(palette::EARTH, 400);
            p.brush_sizes = vec![12.0];
            p.min_brush = 4.0;
            p.luminous = true;
            p.armature_levels = 3;
            p.reserve = Some(0.8);
            p.bleed = 0.0;
            p.dry = 1.0;
            p.draw_contours = false;
            p.splatter = 0.0;
            p.edge_pool = 0.0;
            p.leak = 1.0;
            p.technique = t;
            let r = paint_from_image(&img, &p);
            let painted = r.canvas.to_image().into_raw();
            let text = r.score.to_text();
            let back = crate::paint::score::StrokeScore::parse(&text).unwrap().replay(96, 96).unwrap().to_image().into_raw();
            assert_eq!(back, painted, "{t:?} replays byte-exact through the text");
            r.score.strokes.into_iter().filter(|s| s.stage == "leak").collect::<Vec<_>>()
        };
        let leaks = run(WetTechnique::WetOnDry);
        assert!(!leaks.is_empty(), "a wet wash leaks");
        for l in leaks.iter().filter(|l| l.spline.len() == 3) {
            assert!(l.spline[2][1] > l.spline[0][1] + 5.0, "a run goes DOWN the paper: {:?}", l.spline);
            assert!(l.spline[0][1] > 40.0, "and starts near the mass's lower edge: {:?}", l.spline[0]);
        }
        assert!(run(WetTechnique::DryOnDry).is_empty(), "the dry brush leaks nothing");
    }

    #[test]
    fn a_watercolour_face_is_planes_and_line_and_replays_byte_exact() {
        // A sheet with a dark ground and a FACE: a lit oval whose values sit in a narrow band (a face is a
        // narrow band of the sheet's range), two dark eyes, a highlight on the nose above the reserve luma.
        // Under a new watercolour painting the sheet is washed AROUND the face, the face is washed as PLANES
        // on its own value range (several of them, not the one plane the sheet's levels would give), its
        // nose highlight stays paper, and a drawn LINE sits on the eyes. No fine ladder is laid on the face.
        let (w, h) = (128u32, 128u32);
        let face_px = |x: u32, y: u32| {
            let (dx, dy) = ((x as f32 - 64.0) / 32.0, (y as f32 - 64.0) / 42.0);
            dx * dx + dy * dy <= 1.0
        };
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            if face_px(x, y) {
                let eye = ((x as i32 - 54).abs() <= 2 || (x as i32 - 74).abs() <= 2) && (y as i32 - 56).abs() <= 2;
                let nose = (61..67).contains(&x) && (59..69).contains(&y);
                if nose {
                    return image::Rgb([255, 255, 255]);
                }
                let v = if eye { 40u8 } else { 150 + (y.saturating_sub(22) * 50 / 84).min(50) as u8 };
                image::Rgb([v, v.saturating_sub(20), v.saturating_sub(40)])
            } else {
                image::Rgb([50, 45, 60])
            }
        });
        let face: Vec<f32> = (0..w * h).map(|i| if face_px(i % w, i / w) { 1.0 } else { 0.0 }).collect();
        let mut p = PaintParams::new(palette::EARTH, 3000);
        p.brush_sizes = vec![16.0, 8.0, 4.0];
        p.min_brush = 3.0;
        p.from_scratch = true;
        p.armature_side = Some(64);
        p.luminous = true;
        p.technique = WetTechnique::WetOnDry;
        p.armature_levels = 4;
        p.reserve = Some(0.8);
        p.bleed = 0.0;
        p.dry = 1.0;
        p.splatter = 0.0;
        p.edge_pool = 0.0;
        p.face_mask = Some(face.clone());
        p.face_ladder_from = Some(1);
        let r = paint_from_image(&img, &p);
        let stages = r.score.header.stages.clone().unwrap_or_default();
        assert!(stages.iter().any(|s| s == "face-wash-1") && stages.iter().any(|s| s == "wash-1"), "the face has wash stages of its own after the sheet's: {stages:?}");
        let face_washes: Vec<_> = r.score.strokes.iter().filter(|s| s.stage.starts_with("face-wash")).collect();
        let face_levels: std::collections::BTreeSet<&str> = face_washes.iter().map(|s| s.stage.as_str()).collect();
        assert!(face_levels.len() >= 3, "a face is several planes on its own value range, not one: {face_levels:?}");
        // The sheet's washes stay off the face: no sheet wash ring has a vertex deep inside the oval.
        let deep = |q: &[f32; 2]| { let (dx, dy) = ((q[0] - 64.0) / 26.0, (q[1] - 64.0) / 36.0); dx * dx + dy * dy <= 1.0 };
        let sheet_inside = r.score.strokes.iter().filter(|s| s.stage.starts_with("wash-")).flat_map(|s| s.spline.iter()).filter(|q| !q[0].is_nan() && deep(q)).count();
        assert_eq!(sheet_inside, 0, "the sheet is washed around the face ({sheet_inside} vertices inside it)");
        let line: Vec<_> = r.score.strokes.iter().filter(|s| s.stage == "face-line").collect();
        assert!(!line.is_empty(), "the likeness is a drawn line");
        assert!(line.iter().all(|s| s.bristles == Some(1)), "drawn with a pen point");
        // On the face — or a hair beyond its edge, where the jaw line is.
        let near_face = |q: &[f32; 2]| { let (dx, dy) = ((q[0] - 64.0) / 35.0, (q[1] - 64.0) / 45.0); dx * dx + dy * dy <= 1.0 };
        assert!(line.iter().all(|s| near_face(&s.spline[s.spline.len() / 2])), "and only on the face");
        assert!(!r.score.strokes.iter().any(|s| s.stage.starts_with("restate") && !s.wash && face[((s.spline[0][1] as u32).min(h - 1) * w + (s.spline[0][0] as u32).min(w - 1)) as usize] > 0.5 && s.bristles.is_none() && p.face_ladder_from.is_some() && s.stage == "restate-2"), "no face-only fine ladder on a planes-and-line face");
        // The nose highlight stays paper: the brightest shape of the face is not washed.
        let out = r.canvas.to_image();
        let ground = Canvas::white(1, 1, palette::EARTH, 0.85).color_at(0, 0);
        assert_eq!(out.get_pixel(64, 64).0, ground, "the nose highlight is reserved paper");
        // And it all replays through the text.
        let painted = out.into_raw();
        let text = r.score.to_text();
        let back = crate::paint::score::StrokeScore::parse(&text).unwrap().replay(w, h).unwrap().to_image().into_raw();
        assert_eq!(back, painted, "the planes, the line and the stage crossings replay byte-exact");
    }

    #[test]
    fn the_face_planes_take_the_skin_not_the_box() {
        // A detector box around a face: a warm oval of skin with a dark cap of hair across the top of the box,
        // on a cool ground. The planes' shape must be the skin — forehead and cheeks in, the hair and the
        // ground out — and the eyes (two dark holes in the skin) must be filled back in as part of the face.
        let (w, h) = (160u32, 160u32);
        let skin = |x: u32, y: u32| {
            let (dx, dy) = ((x as f32 - 80.0) / 34.0, (y as f32 - 86.0) / 44.0);
            dx * dx + dy * dy <= 1.0
        };
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            let eye = ((x as i32 - 66).abs() <= 3 || (x as i32 - 94).abs() <= 3) && (y as i32 - 76).abs() <= 2;
            if skin(x, y) && y > 52 {
                if eye { image::Rgb([40, 30, 30]) } else { image::Rgb([210, 160, 130]) }
            } else if (40..120).contains(&x) && (36..56).contains(&y) {
                image::Rgb([40, 35, 30]) // the hair, across the top of the box
            } else {
                image::Rgb([90, 110, 150])
            }
        });
        let mut p = PaintParams::new(palette::EARTH, 100);
        p.face_mask = Some(vec![1.0; (w * h) as usize]);
        p.face_boxes = vec![[44.0, 40.0, 116.0, 132.0]];
        let scope = face_planes_scope(&img, &p).expect("a face gives a scope");
        let at = |x: u32, y: u32| scope[(y * w + x) as usize];
        assert!(at(80, 70), "the forehead is face");
        assert!(at(60, 100) && at(100, 100), "both cheeks are face");
        assert!(at(66, 76) && at(94, 76), "the eyes are holes in the skin, filled back in");
        assert!(!at(80, 45), "the hair across the top of the box is not");
        assert!(!at(50, 125) && !at(110, 125), "nor the ground in the box's lower corners");
        // Without boxes the mask is the shape.
        p.face_boxes.clear();
        p.face_mask = Some((0..w * h).map(|i| if skin(i % w, i / w) { 1.0 } else { 0.0 }).collect());
        let by_mask = face_planes_scope(&img, &p).unwrap();
        assert!(by_mask[(80 * w + 80) as usize] && !by_mask[(10 * w + 10) as usize]);
    }

    #[test]
    fn glazed_washes_leave_no_paper_between_neighbouring_masses() {
        // Three vertical bands of one hue at three values, on a sheet with no reserve. Washed once each, the
        // bands met along hairlines of bare paper — the medium's "white specks". Glazed, the lightest wash
        // covers the whole family and the darker ones stack inside it, so inside the painted region no pixel
        // is left at the ground.
        // The real gap mechanism: an ISLAND of a darker value too small to be a wash of its own. Washed once
        // per mass, it is skipped and left as bare ground — the white speck. Glazed, the band's lighter wash
        // covers everything darker, the island included. A sprinkling of 2×2 islands inside the lightest band.
        let img = image::RgbImage::from_fn(96, 48, |x, y| {
            let island = x >= 68 && (x % 12) < 4 && (y % 12) < 4;
            let v = if x < 32 { 70u8 } else if x < 64 { 130 } else if island { 70 } else { 190 };
            image::Rgb([v / 2, v / 2 + 10, v])
        });
        let run = |t: WetTechnique| {
            let mut p = PaintParams::new(palette::EARTH, 400);
            p.brush_sizes = vec![12.0];
            p.min_brush = 4.0;
            p.luminous = true;
            p.armature_levels = 3;
            p.reserve = None;
            p.bleed = 0.0;
            p.dry = 1.0;
            p.draw_contours = false;
            p.splatter = 0.0;
            p.edge_pool = 0.0;
            p.technique = t;
            let r = paint_from_image(&img, &p);
            let ground = Canvas::white(1, 1, palette::EARTH, 0.85).color_at(0, 0);
            let out = r.canvas.to_image();
            // Count bare-ground pixels well inside the sheet (the edges can legitimately feather).
            (6..90u32).flat_map(|x| (6..42u32).map(move |y| (x, y))).filter(|&(x, y)| out.get_pixel(x, y).0 == ground).count()
        };
        let once = run(WetTechnique::None);
        let glazed = run(WetTechnique::WetOnDry);
        assert_eq!(glazed, 0, "glazing leaves no bare ground inside the washes ({glazed} px)");
        assert!(once > glazed, "and the single-wash form did ({once} px) — otherwise this test proves nothing");
    }

    #[test]
    fn the_techniques_lay_pigment_differently_and_each_replays_byte_exact() {
        // Three techniques, one picture: the dry brush must carry LESS pigment and go down RAKED; the wet one
        // must land WETTER. And each is an ordinary recorded painting, so each must replay exactly.
        let img = gradient_img(96, 64);
        let run = |t: WetTechnique| {
            let mut p = PaintParams::new(palette::EARTH, 2500);
            p.brush_sizes = vec![14.0, 7.0, 4.0];
            p.min_brush = 3.0;
            p.luminous = true;
            p.armature_levels = 4;
            p.bleed = 0.3;
            p.dry = 1.0;
            p.draw_contours = false;
            p.technique = t;
            let r = paint_from_image(&img, &p);
            let strokes: Vec<_> = r.score.strokes.iter().filter(|s| !s.wash && s.stage != "splatter" && s.stage != "pool").collect();
            let mean = |f: &dyn Fn(&StrokeRecord) -> f32| strokes.iter().map(|s| f(s)).sum::<f32>() / strokes.len().max(1) as f32;
            let load = mean(&|s| s.mix.iter().map(|(_, v)| *v).sum::<f32>());
            let wet = mean(&|s| s.wet);
            let streak = mean(&|s| s.streak);
            let painted = r.canvas.to_image().into_raw();
            let text = r.score.to_text();
            let back = crate::paint::score::StrokeScore::parse(&text).unwrap().replay(96, 64).unwrap().to_image().into_raw();
            assert_eq!(back, painted, "{t:?} replays byte-exact through the text");
            (load, wet, streak)
        };
        let (l_wet, w_wet, _) = run(WetTechnique::WetOnWet);
        let (l_dry, w_dry, k_dry) = run(WetTechnique::DryOnDry);
        let (_, w_set, k_set) = run(WetTechnique::WetOnDry);
        assert!(l_dry < l_wet * 0.75, "the dry brush carries less pigment ({l_dry:.3} vs {l_wet:.3})");
        assert!(w_wet > w_set && w_set > w_dry, "wetness orders wet-on-wet > wet-on-dry > dry-on-dry ({w_wet:.2} / {w_set:.2} / {w_dry:.2})");
        // The dry brush is held at the raked floor (0.8); a wet mark keeps the medium's own streak, which on
        // this palette's brush already sits near 0.7, so the gap is real but not large. Measured 0.80 vs 0.69.
        assert!(k_dry >= 0.79 && k_dry > k_set, "the dry brush is raked ({k_dry:.2} vs {k_set:.2})");
    }

    #[test]
    fn the_infill_carries_direction_into_a_flat_passage() {
        // Left half carries diagonal structure; the right half is flat and has nothing of its own to follow.
        // FLAT lays the empty half level, which is right for a sky and is what tiles a dark mass into a
        // rectangular quilt. FOLLOW carries the neighbouring direction across instead.
        let (w, h) = (96u32, 64u32);
        let luma: Vec<f32> = (0..(w * h))
            .map(|i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                if x < 40 && ((x + y) / 4) % 2 == 0 { 0.15 } else { 0.85 }
            })
            .collect();
        let flat_at = |mode: FlowInfill| {
            let (gx, gy) = coherent_gradient(&luma, w, h, 3, mode);
            let i = (32 * w + 80) as usize; // deep in the empty half
            stroke_dir(gx[i], gy[i])
        };
        let level = flat_at(FlowInfill::Flat);
        assert!(level[1].abs() < 0.2, "flat lays the empty passage level: {level:?}");
        let followed = flat_at(FlowInfill::Follow);
        assert!(followed[1].abs() > 0.4, "follow carries the diagonal across instead: {followed:?}");
        // And a named angle is obeyed outright.
        let fixed = flat_at(FlowInfill::Angle(90.0));
        assert!(fixed[0].abs() < 0.2, "a named 90° is a vertical stroke: {fixed:?}");
    }


    #[test]
    fn the_hair_flow_field_finds_the_direction_structure_runs() {
        // Diagonal stripes at 45°: the field must report the direction ALONG them (not across), and say it is
        // confident. A flat field has no direction to find and must say so — otherwise strokes would follow
        // noise wherever the picture is smooth.
        let stripes = image::RgbImage::from_fn(128, 128, |x, y| {
            let v = if ((x + y) / 4) % 2 == 0 { 30u8 } else { 220 };
            image::Rgb([v, v, v])
        });
        let f = hair_flow_field(&stripes, 64);
        let mid = 64 * 128 + 64;
        // Stripes of constant (x+y) run along (1,−1)/√2; orientation is mod π, so either sign will do.
        let along = (f.dx[mid] * 0.70710678 + f.dy[mid] * -0.70710678).abs();
        assert!(along > 0.9, "the field runs along the stripes, not across them (|cos| = {along:.3})");
        assert!(f.coherence[mid] > 0.7, "and says it is confident ({:.2})", f.coherence[mid]);
        assert!(f.strandness[mid] > 0.5, "stripes at strand scale are strandy ({:.2})", f.strandness[mid]);

        let flat = image::RgbImage::from_pixel(128, 128, image::Rgb([128, 128, 128]));
        let g = hair_flow_field(&flat, 64);
        assert!(g.coherence[mid] < 0.3, "a flat field has no direction to find ({:.2})", g.coherence[mid]);
        assert!(g.strandness[mid] < 0.1, "and nothing strand-like in it ({:.2})", g.strandness[mid]);
    }

    #[test]
    fn the_hair_mask_changes_the_tool_and_still_replays_byte_exact() {
        // Hair is high-frequency DIRECTIONAL texture and the armature is structure with the texture taken out,
        // so a from-scratch painting has nothing to paint hair FROM and a mane came out a lumpy mass. Where the
        // hair mask is set the painter changes TOOL: more bristle lanes (the lanes ARE the strands), a rakier
        // streak, almost no pickup so strands stay distinct, and a mark that tapers to a point.
        let img = gradient_img(96, 64);
        let mut p = PaintParams::new(palette::EARTH, 900);
        p.brush_sizes = vec![16.0, 8.0, 4.0];
        p.min_brush = 3.0;
        p.from_scratch = true;
        p.armature_side = Some(48);
        // The left half is hair, the right half is not.
        p.hair_mask = Some((0..96 * 64).map(|i| if i % 96 < 48 { 1.0 } else { 0.0 }).collect());
        let r = paint_from_image(&img, &p);

        let strand: Vec<_> = r.score.strokes.iter().filter(|s| s.bristles.is_some()).collect();
        let mass: Vec<_> = r.score.strokes.iter().filter(|s| s.bristles.is_none()).collect();
        assert!(!strand.is_empty() && !mass.is_empty(), "both tools were used: {} strand, {} mass", strand.len(), mass.len());
        // A strand carries its own tool, so replay cannot fall back on the header's painting brush.
        assert!(strand.iter().all(|s| s.pickup.is_some()), "a strand records its near-zero pickup");
        assert!(strand.iter().all(|s| s.bristles.unwrap() > p.brush.bristles), "a strand has more lanes than the painting brush");
        let tap_s = strand.iter().map(|s| s.taper).sum::<f32>() / strand.len() as f32;
        let tap_m = mass.iter().map(|s| s.taper).sum::<f32>() / mass.len() as f32;
        assert!(tap_s > tap_m + 0.2, "a strand ends in a point, a mass mark lifts off ({tap_s:.2} vs {tap_m:.2})");

        // The whole point of recording the tool: the score still reproduces the painting exactly.
        let painted = r.canvas.to_image().into_raw();
        assert_eq!(r.score.replay(96, 64).unwrap().to_image().into_raw(), painted, "replays byte-exact in memory");
        let text = r.score.to_text();
        let parsed = crate::paint::score::StrokeScore::parse(&text).expect("the score parses back");
        assert_eq!(parsed.replay(96, 64).unwrap().to_image().into_raw(), painted, "and through the text");
    }

    #[test]
    fn the_planes_meet_along_a_ramp_not_a_step() {
        // A focal region is a face DETECTOR'S BOX, and a long beard runs straight out of the bottom of it.
        // While the minimum brush stepped at the mask's midpoint, that beard was painted with a fine brush
        // above the chin and a coarse one below, and the boundary read as a cut across the face. Through a
        // feathered mask the floor must CHANGE GRADUALLY.
        let (w, h) = (256u32, 256u32);
        // A 32 px feather centred on the middle row — the shape `build_face_mask`'s blur leaves.
        let face: Vec<f32> = (0..(w * h)).map(|i| (((128.0 - (i / w) as f32) / 32.0) + 0.5).clamp(0.0, 1.0)).collect();
        let subject = vec![1.0f32; (w * h) as usize];
        let field = plane_floor_field(w, h, 1.0, (32, Some(64), Some(128)), Some(&subject), Some(&face), None);

        let col: Vec<f32> = (0..h).map(|y| field[(y * w + 128) as usize]).collect();
        let (lo, hi) = col.iter().fold((f32::MAX, 0.0f32), |(l, g), &v| (l.min(v), g.max(v)));
        assert!(hi - lo > 1.0, "the two planes really do differ ({lo:.2}..{hi:.2} px)");
        let worst = col.windows(2).map(|p| (p[0] - p[1]).abs()).fold(0.0f32, f32::max);
        assert!(
            worst < (hi - lo) * 0.2,
            "no single row may carry the whole change: worst step {worst:.3} px of a {:.3} px range",
            hi - lo
        );
        // The ends still reach their own plane's floor — softening must not blunt the focal plane itself.
        assert!((col[0] - lo).abs() < 1e-3 && (col[(h - 1) as usize] - hi).abs() < 1e-3, "each end sits at its plane's floor");
    }

    #[test]
    fn a_from_scratch_painting_keeps_fine_brushes_on_the_focal_plane_and_replays_byte_exact() {
        let img = gradient_img(200, 160);
        let mut p = PaintParams::new(palette::EARTH, 6000);
        p.brush_sizes = vec![12.0, 6.0, 3.0, 1.5];
        p.min_brush = 1.0;
        p.armature_side = Some(12);
        p.armature_body_side = Some(24);
        p.armature_face_side = Some(64);
        p.from_scratch = true;
        // A figure in the left half, a face in its top-left corner; the right half is background.
        let subject: Vec<f32> = (0..200 * 160).map(|i| if i % 200 < 100 { 1.0 } else { 0.0 }).collect();
        let face: Vec<f32> = (0..200 * 160).map(|i| if i % 200 < 40 && i / 200 < 40 { 1.0 } else { 0.0 }).collect();
        p.subject_mask = Some(subject);
        p.face_mask = Some(face);
        let (bg, body, focal) = plane_floors(200, 160, p.min_brush, (12, Some(24), Some(64)));
        assert_eq!(region_extent(p.face_mask.as_deref().unwrap(), 200, 160), Some(40.0), "a 40 px face");
        assert!(bg > body && body > focal, "the minimum brush grows with depth: {bg} > {body} > {focal}");
        let r = paint_from_image(&img, &p);
        assert!(r.strokes > 50, "it paints ({} strokes)", r.strokes);
        // Every recorded mark respects its plane's floor (w0 is the mark's own width ≥ 0.8 × its pass radius…
        // so test by where fine-stage marks START).
        let stages: Vec<String> = r.score.header.stages.clone().unwrap_or_default();
        for rec in r.score.strokes.iter().filter(|s| !s.wash) {
            let Some(idx) = stages.iter().position(|st| st == &rec.stage) else { continue };
            let radius = [12.0f32, 6.0, 3.0, 1.5].into_iter().filter(|q| *q >= focal).nth(idx).unwrap_or(12.0);
            // the seed is mid-path; a mark of a rung finer than the background floor must sit in the figure
            let mid = rec.spline[rec.spline.len() / 2];
            if radius < bg {
                assert!(mid[0] < 100.0 + 2.0 * radius + 12.0, "a {radius}px mark at x={} is in the background", mid[0]);
            }
        }
        let painted = r.canvas.to_image().into_raw();
        assert_eq!(r.score.replay(200, 160).unwrap().to_image().into_raw(), painted, "a from-scratch painting replays byte-exact");
        // …and it is a painting of few marks: the same sheet on the default path lays many more.
        let mut d = p.clone();
        d.from_scratch = false;
        let dflt = paint_from_image(&img, &d);
        assert!(r.strokes < dflt.strokes, "from scratch {} < default {}", r.strokes, dflt.strokes);
    }

    #[test]
    fn a_wet_multi_pass_paint_replays_byte_exact() {
        // Wet media bleed after EVERY pass (tapering) and dry between them; the replay must reproduce that
        // schedule from the stage names alone, or the delivered image and its score would disagree.
        let img = gradient_img(64, 48);
        let mut p = PaintParams::new(palette::EARTH, 400);
        p.brush_sizes = vec![16.0, 8.0, 4.0];
        p.bleed = 0.5;
        p.dry = 0.5;
        let result = paint_from_image(&img, &p);
        let painted = result.canvas.to_image().into_raw();
        let replayed = result.score.replay(64, 48).unwrap().to_image().into_raw();
        assert_eq!(painted, replayed, "per-pass bleed + drying replay byte-for-byte");
        assert!(result.score.strokes.iter().any(|r| r.stage == "restate-2"), "every pass has its own stage name");
    }

    #[test]
    fn a_score_round_trips_through_text_exactly() {
        // The score is the CANONICAL artefact (RFC PAINT-1), and it lives on DISK as text: `paint replay
        // <score>` must reproduce the painting the run saved. Every other replay test compares the score held
        // in MEMORY, so none of them could see a serialisation that lost precision — and `{:.4}` did, by
        // truncating spline coordinates and mixture ratios (0.67% RMSE on an oil, 14.6% on pen-ink).
        //
        // A wet, multi-pass, many-pigment painting: splines, per-stroke widths, tapers and mixture ratios all
        // travel through the text. Write it, parse it back, replay BOTH, and demand the same pixels.
        let img = gradient_img(72, 56);
        let mut p = PaintParams::new(palette::EARTH, 500);
        p.brush_sizes = vec![16.0, 8.0, 4.0];
        p.min_brush = 3.0;
        p.bleed = 0.4;
        p.dry = 0.5;
        let result = paint_from_image(&img, &p);
        assert!(result.score.strokes.len() > 50, "a score with enough strokes to expose rounding");

        let text = result.score.to_text();
        let parsed = crate::paint::score::StrokeScore::parse(&text).expect("the written score parses back");
        assert_eq!(parsed.strokes.len(), result.score.strokes.len(), "no stroke lost in the text");

        let from_memory = result.score.replay(72, 56).unwrap().to_image().into_raw();
        let from_text = parsed.replay(72, 56).unwrap().to_image().into_raw();
        assert_eq!(from_text, from_memory, "a score written to text and read back replays byte-for-byte");

        // And the text itself is stable: serialising the parsed score reproduces the same bytes, so a score
        // that survives one round trip survives any number of them.
        assert_eq!(parsed.to_text(), text, "serialisation is idempotent");
    }

    #[test]
    fn painting_keeps_structure_without_perfectly_tracing() {
        // The painted output should correlate with the reference (structure survives) but NOT perfectly
        // (surface is invented, budget/palette constrain it) — the filter-gate signal.
        let img = gradient_img(80, 60);
        let mut p = PaintParams::new(palette::EARTH, 300);
        p.brush_sizes = vec![20.0, 10.0];
        let out = paint_from_image(&img, &p).canvas.to_image();
        let tr = traceability(&out, &img);
        // Structure survives (clear positive correlation) but the invented, WAVERED surface deliberately keeps
        // it well below a trace — the natural not-straight brushwork lowers correlation, as intended.
        // The bar is calibrated to SHORT, edge-stopping strokes with dry detail accents: on a smooth synthetic
        // gradient those correlate less than the old long smearing strokes did (long smears track a gradient
        // precisely BECAUSE they smear — the mush the engine no longer produces), while on real images they keep
        // structure far better. ~0.33 here; the filter-gate signal is "clearly positive, far below a trace".
        assert!(tr > 0.25, "structure survives (corr {tr})");
        assert!(tr < 0.999, "not a pixel-perfect trace (corr {tr})");
    }
}

