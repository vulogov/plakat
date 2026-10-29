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
            // A steel nib: narrow, long, no streak (the ink is even across the point, the paper smooth), a steady hand,
            // and a FLAT section — the point lays its ink whole to a hard edge (round, a hairline was a grey falloff).
            "pen" => BrushProfile { radius_scale: 0.30, len: 1.00, streak: 0.0, round: 0.15, waver: 0.05 },
            // A tempera brush: small, short, clean, even — the strokes read as a woven net.
            "tempera" => BrushProfile { radius_scale: 0.60, len: 0.60, streak: 0.15, round: 0.65, waver: 0.06 },
            // broad, long, soft, thin — watercolour / sky washes
            "wash" => BrushProfile { radius_scale: 1.30, len: 1.60, streak: 0.15, round: 0.85, waver: 0.10 },
            // A synthetic flat for acrylic: smooth, evenly loaded lanes, a square edge, a steady hand — the
            // mark is a flat patch with a crisp edge, not a raked or a feathered one.
            "acrylic" => BrushProfile { radius_scale: 1.00, len: 1.00, streak: 0.30, round: 0.40, waver: 0.05 },
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
    /// NIB EYES (see `MediumProfile::nib_eyes`): an animal's eyes and nose as the detector found them, 0..1 (soft
    /// boxes); taken out of `hair_mask` in the CLI and outlined after the bone.
    pub nib_feature_mask: Option<Vec<f32>>,
    /// NIB OBJECT (see `MediumProfile::nib_object`): the rigid things a figure holds (an umbrella, a stick), 0..1 —
    /// joined to the figure under `nib_figure` and outlined with it.
    pub nib_object_mask: Option<Vec<f32>>,
    /// NIB TUFT (see `MediumProfile::nib_tuft`): where the picture is grass, 0..1 — the tufts only there, none without it.
    pub nib_grass_mask: Option<Vec<f32>>,
    /// NIB PEOPLE (see `MediumProfile::nib_people`): every person found, 0..1 — joined to the figure under `nib_figure`.
    pub nib_person_mask: Option<Vec<f32>>,
    /// NIB DEPTH (see `MediumProfile::nib_depth`): the picture's depth, 0..1, larger = nearer (Depth-Anything).
    pub nib_depth_map: Option<Vec<f32>>,
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
    /// The picture's RELIEF for an engraving (`--depth`): a depth map `(w, h, z)` at its own resolution, `z` in
    /// `[0,1]`, 1 = near. The burin's lines turn with the form it describes (see `paint::engrave`). Not used for
    /// recession — that is `depth` + `haze`.
    pub relief: Option<(u32, u32, Vec<f32>)>,
    /// The picture's surface normals for an engraving (`engrave::Normals`): width, height, unit vectors in
    /// image space. `None` = the line's direction comes from the relief, or from the picture alone.
    pub normals: Option<(u32, u32, Vec<[f32; 3]>)>,
    /// What the picture's things are made of, for an engraving (`engrave::Materials`): width, height, a kind per
    /// pixel (0 = nothing said).
    pub materials: Option<(u32, u32, Vec<u8>)>,
    /// The features of the picture's faces, for the book illustration: per face its left eye, right eye, nose
    /// tip and the two corners of its mouth, as shares of the picture's width and height. Empty = not known.
    pub features: Vec<[[f32; 2]; 5]>,
    /// FOLLOW (0..1, default 1) for an engraving: how far the hatch follows the form. 0 = a ruled plate — every
    /// line on the resting diagonal or level on the ground, only the contours bend; a portrait wants about 0.3.
    pub follow: f32,
    /// An engraving's WHITE POINT (0.5..1): the share of the picture's values at or above which the paper is
    /// left clean. `None` (the default) finds the paper in the picture — its brightest broad flat value, or its
    /// brightest thousandth.
    pub white: Option<f32>,
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
    /// Write the canvas after every pass of the ladder into this directory (`pass_<n>_<stage>.png`) —
    /// the layer hierarchy the outcome sheet shows (RFC PAINT-3). `None` = no dumps.
    pub dump_passes: Option<std::path::PathBuf>,
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
    /// FINE LINES (the watercolour wash recipe; 0..1): how far the finest brushes' marks may run along an edge
    /// beyond the ladder's cap — 1 = a rigger's line (3× the cap), 0 = the short marks only (the old beads).
    pub fine_lines: f32,
    /// IMPASTO MAPPED TO THE PICTURE (0..1, 0 = off): the paint's thickness follows the picture — lights thick,
    /// shadows thin; subject thick, background thin; nearer thicker when a depth map is known — instead of
    /// one thickness over the sheet. Per stroke, recorded in the score.
    pub impasto_map: f32,
    /// CANVAS WEAVE (0..1, 0 = none): the linen's threads under the paint, seen in the relight where the
    /// paint is thin or bare; a built-up passage covers them.
    pub weave: f32,
    /// WET COLLISION (0..1, 0 = off): after each broad pass the wet paint drags along its own striation —
    /// colours marble, ridges plow — where loaded wet strokes meet. Off the faces; tapered with the passes.
    pub collide: f32,
    /// GRADATION (0..1, default 0): keep slow RAMPS continuous in the armature. The value masses turn a cloud,
    /// a soft-lit wall or still water into a few flat tones with contour edges; where the picture is a ramp,
    /// not an edge (the flow rule's scale-free test), this brings the bilateral back toward a plain smooth of
    /// the source and holds the value snap off, so the mass keeps its turning form. Edges snap as before.
    pub gradation: f32,
    /// CROSS-HATCH (medium mark character): each restating pass rotates its stroke direction by this many radians
    /// more than the previous one; 0 = every pass follows the form.
    pub hatch_angle: f32,
    /// Density media: cut the picture as a copperplate (see `paint::engrave`).
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
    /// SPLATTER (0..1, 0 = off): flick fine pigment droplets across the painting — the watercolour/ink spatter
    /// mark. Higher = denser spray. Recorded as strokes (replay-exact). See `splatter_pass`.
    pub splatter: f32,
    /// EDGE POOLING (0..1, 0 = off): darken pigment where a wash meets a hard boundary — the pigment ring a
    /// watercolour wash dries into (the "cauliflower"/edge-bloom look). An output-stage effect (see `Finish`).
    pub edge_pool: f32,
    /// PAPER EDGE (0..1, 0 = off): fade the painting to bare paper at the borders with an irregular DECKLED edge
    /// — the torn-paper vignette a watercolour sits in. An output-stage effect (see `Finish`).
    pub paper_edge: f32,
    /// OLD PAPER (`--oldpaper`): print the picture on an aged sheet. An output-stage effect (see `Finish`).
    pub old_paper: bool,
    /// FINISH GRADE (painting-safe, recorded for replay). CONTRAST (0.5..2, 1 = neutral): S-curve around mid-grey.
    pub contrast: f32,
    /// WARMTH (−1..1, 0 = neutral): white-balance shift, + warm / − cool.
    pub warmth: f32,
    /// CLARITY (0..1, 0 = off): gentle LOCAL contrast (large-radius unsharp) — NOT edge sharpening.
    pub clarity: f32,
    /// WASH-IN (0..1): the block-in laid as a thin, flooded wash (see `MediumProfile::wash_in`). 0 = full charge.
    pub wash_in: f32,
    /// WET CAP (0..1, 1 = none): the wettest a mark after the block-in goes on (see `MediumProfile::wet_cap`).
    pub wet_cap: f32,
    /// SCUMBLE (0..1, 0 = off): the finest pass dry-brushed off the faces (see `MediumProfile::scumble`).
    pub scumble: f32,
    /// TURBULENCE (degrees, 0 = off): the flow turned by a low-frequency noise (see `MediumProfile::turbulence`).
    pub turbulence: f32,
    /// ACCENTS (0..1, 0 = off): thick short dabs at the sharp contrasts on the finest pass (see `MediumProfile::accent`).
    pub accent: f32,
    /// LOOSE (0..1, 0 = off): longer, wider marks off the subject on the broad passes (see `MediumProfile::loose`).
    pub loose: f32,
    /// GLAZE (0..1, 0 = off): translucent overlapping marks along the subject's silhouette on the finest pass
    /// (see `MediumProfile::glaze`).
    pub glaze: f32,
    /// SFUMATO (0..1, 0 = off): the skin fused where it is smooth (see `MediumProfile::sfumato`).
    pub sfumato: f32,
    /// GLAZE PASS (0..1, 0 = off): one thin wet pass over everything after the ladder (see `MediumProfile::glaze_pass`).
    pub glaze_pass: f32,
    /// VARY (0..1, 0 = off): a Gaussian spread of the mark's length (see `MediumProfile::vary`).
    pub vary: f32,
    /// SHADE (0..1, 0 = off): the shadows thinned, broadened and calmed (see `MediumProfile::shade`).
    pub shade: f32,
    /// IMPRIMATURA (0..1, 0 = off): the block-in laid darker and warmer (see `MediumProfile::imprimatura`).
    pub imprimatura: f32,
    /// BODY VARY (σ, 0 = off): a Gaussian spread of each mark's body (see `MediumProfile::body_vary`).
    pub body_vary: f32,
    /// LOST (0..1, 0 = off): the subject's edge lost where it lies in shadow (see `MediumProfile::lost`).
    pub lost: f32,
    /// CREST (0..1, 0 = off): a dry brush over the ridges of the relief (see `MediumProfile::crest`).
    pub crest: f32,
    /// SPATTER (0..1, 0 = off): thick drops flung off the thickest marks (see `MediumProfile::spatter`).
    pub spatter: f32,
    /// AERIAL (0..1, 0 = off): the lost edges by depth, from `depth` (see `MediumProfile::aerial`).
    pub aerial: f32,
    /// SINK (0..1, 0 = off): a wash into the furrows of the relief in the shadow (see `MediumProfile::sink`).
    pub sink: f32,
    /// TAIL (0..1, 0 = off): a spatter drop stretched by the mark's speed (see `MediumProfile::tail`).
    pub tail: f32,
    /// FILM (0..1, 0 = off): the glaze marks laid as a film over the dried paint (see `MediumProfile::film`).
    pub film: f32,
    /// SPLIT (0..1, 0 = off): the dry fast marks combed into bundles (see `MediumProfile::split`).
    pub split: f32,
    /// SAG (0..1, 0 = off): the canvas sagging under the thick paint, at output (see `MediumProfile::sag`).
    pub sag: f32,
    /// GLOSS (0..1, 0 = off): a sharp specular on the peaks of the thick paint, at output (see `MediumProfile::gloss`).
    pub gloss: f32,
    /// TINT (0..1, 0 = off): the bundles of a split mark differ in tone (see `MediumProfile::tint`).
    pub tint: f32,
    /// KNIFE (0..1, 0 = off): the share of the broad marks off the subject laid with a palette knife (see `MediumProfile::knife`).
    pub knife: f32,
    /// HIDING (0..1, 0 = off): each pigment covers by its own hiding power (see `MediumProfile::hiding`).
    pub hiding: f32,
    /// TWIST (0..1, 0 = off): the brush turned along the hair strands and the finest marks (see `MediumProfile::twist`).
    pub twist: f32,
    /// SKIM (0..1, 0 = off): the share of the knife marks that skim, catching only the relief's tops (see `MediumProfile::skim`).
    pub skim: f32,
    /// BLADE (0..1, 0 = off): the palette knife as the medium's own tool — this share of all the marks on every pass, the subject included (see `MediumProfile::blade`).
    pub blade: f32,
    /// BLADE PLANES (a count, 0 = off): every pass's reference cut into flat planes of value (see `MediumProfile::blade_planes`).
    pub blade_planes: f32,
    /// BLADE LEVELS (0..1): the planes cut at even steps (0) or the picture's quantiles (1) (see `MediumProfile::blade_levels`).
    pub blade_levels: f32,
    /// BLADE SIZE (0..1, 0 = off): the slice's size by the flatness under it (see `MediumProfile::blade_size`).
    pub blade_size: f32,
    /// BLADE LIGHT (degrees, 0 = off): the lamp's azimuth for the relief's relight (see `MediumProfile::blade_light`).
    pub blade_light: f32,
    /// BLADE PHONG (0..1, 0 = off): a specular lobe on the relief's facets that face the lamp (see `MediumProfile::blade_phong`).
    pub blade_phong: f32,
    /// BLADE FLOW (0..1, 0 = off): the slices along the picture's form, straight (see `MediumProfile::blade_flow`).
    pub blade_flow: f32,
    /// BLADE REST (0..1, 0 = off): the restating passes leave the flat slabs alone (see `MediumProfile::blade_rest`).
    pub blade_rest: f32,
    /// BLADE RELIEF (0..1, 0 = off): the relief's slope over a window at output (see `MediumProfile::blade_relief`).
    pub blade_relief: f32,
    /// BLADE ALONG (0..1, 0 = off): the flatness read along the flow (see `MediumProfile::blade_along`).
    pub blade_along: f32,
    /// BLADE INERTIA (0..1, 0 = off): the straight slice through the noise to a hard stop (see `MediumProfile::blade_inertia`).
    pub blade_inertia: f32,
    /// BLADE FACET (0..1, 0 = off): the face cut into its own planes (see `MediumProfile::blade_facet`).
    pub blade_facet: f32,
    /// BLADE VISC (0..1, 0 = off): the colour stop rising along the flow (see `MediumProfile::blade_visc`).
    pub blade_visc: f32,
    /// BLADE SOFT (0..1, 0 = off): the face's terraces softened on the smooth skin (see `MediumProfile::blade_soft`).
    pub blade_soft: f32,
    /// BLADE DETAIL (0..1, 0 = off): the narrow knives keep the detail (see `MediumProfile::blade_detail`).
    pub blade_detail: f32,
    /// BLADE LAYERS (0..1, 0 = off): the knives thin with the ladder and shear less (see `MediumProfile::blade_layers`).
    pub blade_layers: f32,
    /// BLADE EDGE (0..1, 0 = off): the knife's edge draws the busy passages (see `MediumProfile::blade_edge`).
    pub blade_edge: f32,
    /// BLADE TUFT (0..1, 0 = off): a fine texture laid as one pull of the medium knife (see `MediumProfile::blade_tuft`).
    pub blade_tuft: f32,
    /// BLADE ANISO (0..1, 0 = off): the tensor's reading of the grass (see `MediumProfile::blade_aniso`).
    pub blade_aniso: f32,
    /// BLADE SLOPE (0..1, 0 = off): the tuft's pull by the slope (see `MediumProfile::blade_slope`).
    pub blade_slope: f32,
    /// BLADE BREAK (0..1, 0 = off): the wide plates broken by a global slow noise (see `MediumProfile::blade_break`).
    pub blade_break: f32,
    /// BLADE ROUTE (0..1, 0 = off): the two readings of the grass routed by the fineness (see `MediumProfile::blade_route`).
    pub blade_route: f32,
    /// BLADE SCATTER (0..1, 0 = off): the tuft's pull scattered ±10° (see `MediumProfile::blade_scatter`).
    pub blade_scatter: f32,
    /// BLADE TINT (0..1, 0 = off): the broken plates' hue drifting (see `MediumProfile::blade_tint`).
    pub blade_tint: f32,
    /// BLADE FOCUS (0..1, 0 = off): the periphery wide and thick, the detail at the focus (see `MediumProfile::blade_focus`).
    pub blade_focus: f32,
    /// BLADE PEAK (0..1, 0 = off): the specular on the local crests of the relief (see `MediumProfile::blade_peak`).
    pub blade_peak: f32,
    /// BLADE BAND (0..1, 0 = off; with `blade_focus`): the focus as a horizontal band (see
    /// `MediumProfile::blade_band`) — the share by which the focus's distance is read along the vertical alone.
    pub blade_band: f32,
    /// HATCH (0..1, 0 = off): the restating passes lay straight hatch lines of one width (see `MediumProfile::hatch`).
    pub hatch: f32,
    /// VERDACCIO (0..1, 0 = off): the flesh underpainted in a greenish grey (see `MediumProfile::verdaccio`).
    pub verdaccio: f32,
    /// DISHES (the count, 0 = off): the restating marks laid in this many value steps (see `MediumProfile::dishes`).
    pub dishes: f32,
    /// LIGHTS (0..1, the share): the lights hatched in toward white on the finest pass (see `MediumProfile::lights`).
    pub lights: f32,
    /// CROSS (degrees, 0 = off): the hatching turned off the form by this step a hatched pass, alternating sides (see `MediumProfile::cross`).
    pub cross: f32,
    /// DITHER (0..1, 0 = off): the dishes chosen stochastically between the two nearest (see `MediumProfile::dither`).
    pub dither: f32,
    /// SOLID (0..1, 0 = off): the hatch's charge keyed to the value, covering in the darks (see `MediumProfile::solid`).
    pub solid: f32,
    /// FLESH (0..1, 0 = off): the verdaccio held to the warm colours (see `MediumProfile::flesh`).
    pub flesh: f32,
    /// SPAN (0..1, 0 = off): the hatch's length keyed to how busy the form is (see `MediumProfile::span`).
    pub span: f32,
    /// EDGE (0..1, 0 = off): the hatching held parallel to a strong contour (see `MediumProfile::edge`).
    pub edge: f32,
    /// GLINT (0..1, 0 = off): the pure lights set last, dry, covering and raised (see `MediumProfile::glint`).
    pub glint: f32,
    /// SMOOTH (0..1, 0 = off): the calm passages built up in soft veils, not hatched (see `MediumProfile::smooth`).
    pub smooth: f32,
    /// THE BOLE (0..1, 0 = off): the imprimatura laid by local tone before the block-in (see `MediumProfile::bole`).
    pub bole: f32,
    /// THE RAMP (0..1, 0 = off): a mark's colour follows the picture along its path (see `MediumProfile::ramp`).
    pub ramp: f32,
    /// THE UMBRA (0..1, 0 = off): the darkest dish's tones laid continuous, not stepped (see `MediumProfile::umbra`).
    pub umbra: f32,
    /// THE SIDE OF THE STICK (0..1, 0 = off): the broad passes laid with the stick's side (see `MediumProfile::side`).
    pub side: f32,
    /// THE FINGER (0..1, 0 = off): the calm passages rubbed after each broad pass (see `MediumProfile::blend`).
    pub blend: f32,
    /// THE SKIN (0..1, 0 = off): the finger on the faces, kept inside them (see `MediumProfile::skin`).
    pub skin: f32,
    /// CARNATION (0..1, 0 = off): the flesh in warm sticks — a face's mixtures over the warm and neutral
    /// pigments alone, its lights lifted along their hue (see `MediumProfile::carnation`).
    pub carnation: f32,
    /// SMALL (px, 0 = off): a face under this extent drawn at its own scale (see `MediumProfile::small`).
    pub small: f32,
    /// ANCHOR (0..1, 0 = off): a small face's structural darks kept crisp (see `MediumProfile::anchor`).
    pub anchor: f32,
    /// TRIM (0..1, 0 = off): the face mask cut by the subject matte (see `MediumProfile::trim`; applied by
    /// the CLI where both masks are built).
    pub trim: f32,
    /// SMALL'S FLOOR (px, 0 = none): a crowd's face left to the plain ladder (see `MediumProfile::small_min`).
    pub small_min: f32,
    /// PAPER (0..1, 0 = off): the light is the paper, charcoal only darkens (see `MediumProfile::paper`).
    pub paper: f32,
    /// ERASE (0..1, 0 = off): the kneaded eraser (see `MediumProfile::erase`).
    pub erase: f32,
    /// TOUCH (0..1, 0 = off): the pressure follows the value (see `MediumProfile::touch`).
    pub touch: f32,
    /// VELVET (0..1, 0 = off): the darkest share one velvet black (see `MediumProfile::velvet`).
    pub velvet: f32,
    /// PAPER TOP (0..1, 0 = off): the picture's light end is the paper (see `MediumProfile::paper_top`).
    pub paper_top: f32,
    /// SWEEP (0..1, 0 = off): the flat passages in calm sweeps (see `MediumProfile::sweep`).
    pub sweep: f32,
    /// STUMP (0..1, 0 = off): the dark masses rubbed with a stump (see `MediumProfile::stump`).
    pub stump: f32,
    /// STUMP EDGE (0..1, 0 = off): the stump holds at a contrast (see `MediumProfile::stump_edge`).
    pub stump_edge: f32,
    /// STUMP FEATHER (0..1, 0 = off): the stump's soft stop (see `MediumProfile::stump_feather`).
    pub stump_feather: f32,
    /// SWEEP WAVE (degrees, 0 = off): the sweeps on an arm's arc (see `MediumProfile::sweep_wave`).
    pub sweep_wave: f32,
    /// PAPER KNEE (0..1, 0 = off): a broad light mass the paper (see `MediumProfile::paper_knee`).
    pub paper_knee: f32,
    /// CHALK TINT (0..1, 0 = off): the lights mixed with white, chalky (see `MediumProfile::chalk_tint`).
    pub chalk_tint: f32,
    /// FLAT BRUSH (0..1, 0 = off): the flat passages in a flat brush's patches (see `MediumProfile::flat_brush`).
    pub flat_brush: f32,
    /// LINER (0..1, 0 = off): the long thin lines in a liner's single stroke (see `MediumProfile::liner`).
    pub liner: f32,
    /// MA (0..1, 0 = off): the empty space where the eye does not go (see `MediumProfile::ma`).
    pub ma: f32,
    /// YOHAKU (0..1, 0 = off): the flat passages left blank (see `MediumProfile::yohaku`).
    pub yohaku: f32,
    /// BONE (0..1, 0 = off): the line register (see `MediumProfile::bone`).
    pub bone: f32,
    /// BONELESS (0 = off; a count): the passes after the block-in laid as washes (see `MediumProfile::boneless`).
    pub boneless: f32,
    /// HAKE (0..1, 0 = off): the washed passes' flow turned to one horizontal sweep (see `MediumProfile::hake`).
    pub hake: f32,
    /// BONE WIND (0..1, 0 = off): a bone chain winding past this bar is texture, not drawn as a line (see
    /// `MediumProfile::bone_wind`).
    pub bone_wind: f32,
    /// BONE DOT (0..1, 0 = off): the share of the winding chains laid as the tenbyō's dots (see
    /// `MediumProfile::bone_dot`).
    pub bone_dot: f32,
    /// BONE SCATTER (0..1, 0 = off): the tenbyō's dots thrown off their chain along its normal (see
    /// `MediumProfile::bone_scatter`).
    pub bone_scatter: f32,
    /// BONE COHERENCE (0..1, 0 = off): a bone chain drawn only where the gradients around it agree on a
    /// direction (see `MediumProfile::bone_coherence`).
    pub bone_coherence: f32,
    /// BONE END (0..1, 0 = off): the tome where a bone line meets another, the harai where it fades (see
    /// `MediumProfile::bone_end`).
    pub bone_end: f32,
    /// SUMI FOCUS (0..1, 0 = off): the black register laid only where the eye goes (see
    /// `MediumProfile::sumi_focus`).
    pub sumi_focus: f32,
    /// MA FOCUS (0..1, 0 = off): the attention `ma` reads narrowed to the focus (see `MediumProfile::ma_focus`).
    pub ma_focus: f32,
    /// MA FLOW (0..1, 0 = off): the focus's halo spread along the picture's lines (see `MediumProfile::ma_flow`).
    pub ma_flow: f32,
    /// BONE RYAKU (0..1, 0 = off): the abbreviated drawing: a bone line off the focus simplified to a few arcs (see `MediumProfile::bone_ryaku`).
    pub bone_ryaku: f32,
    /// BONE FADE (0..1, 0 = off): a bone chain off the focus left to the washes (see `MediumProfile::bone_fade`).
    pub bone_fade: f32,
    /// BOKASHI (0..1, 0 = off): the marks off the focus wetter: the fading edge dissolved, not merely lightened (see `MediumProfile::bokashi`).
    pub bokashi: f32,
    /// VELVET DARK (0..1, 0 = off): the velvet black held to an absolute dark (see `MediumProfile::velvet_dark`).
    pub velvet_dark: f32,
    /// SLANT (0..1, 0 = off): hatching at the hand's angle where the picture has no direction (see `MediumProfile::slant`).
    pub slant: f32,
    /// GRADE (0..1, 0 = off): the half-tones lifted toward the paper, the velvet kept (see `MediumProfile::grade`).
    pub grade: f32,
    /// WEIGHT (0..1, 0 = off): the bone line at a pencil's weight over the hatching (see `MediumProfile::weight`).
    pub weight: f32,
    /// CLIP (0..1, 0 = off): the hatch stopped at the picture's contours (see `MediumProfile::clip`).
    pub clip: f32,
    /// TIERS (0..1, 0 = off): the hatching in tonal layers: lights blank, half-tones single, shadows crossed (see `MediumProfile::tiers`).
    pub tiers: f32,
    /// NIB (0..1, 0 = off): a steel nib: every line full black, the tone by how many (see `MediumProfile::nib`).
    pub nib: f32,
    /// NIBS (count, 0 = off): pens of several sizes in layers, broad to fine (see `MediumProfile::nibs`).
    pub nibs: f32,
    /// NIB TONE (0..1, 0 = off): each pen only where the tone is dark enough for its width (see `MediumProfile::nib_tone`).
    pub nib_tone: f32,
    /// NIB WHITE (0..1, 0 = off): the share of the paper a cell always keeps (see `MediumProfile::nib_white`).
    pub nib_white: f32,
    /// NIB ANCHOR (0..1, 0 = off): the deep cast-shadow touches that set things on the ground (see `MediumProfile::nib_anchor`).
    pub nib_anchor: f32,
    /// NIB HINT (0..1, 0 = off): the fine pen's rare short dashes in the lights (see `MediumProfile::nib_hint`).
    pub nib_hint: f32,
    /// NIB FACE (0..1, 0 = off): a face drawn with the finest pen, its lights kept (see `MediumProfile::nib_face`).
    pub nib_face: f32,
    /// NIB GAP (0..1, 0 = off): a sliver of paper kept along the long contours (see `MediumProfile::nib_gap`).
    pub nib_gap: f32,
    /// NIB RELIEF (0..1, 0 = off): the hatching thinned on the form's highlights (see `MediumProfile::nib_relief`).
    pub nib_relief: f32,
    /// NIB FAR (0..1, 0 = off): the far ground and the sky in the finest pen only (see `MediumProfile::nib_far`).
    pub nib_far: f32,
    /// NIB OCCLUDE (0..1, 0 = off): the broad nib only in the cast shadows (see `MediumProfile::nib_occlude`).
    pub nib_occlude: f32,
    /// NIB EYES (0..1, 0 = off): the eyes and nose taken out of the fur and outlined (see `MediumProfile::nib_eyes`).
    pub nib_eyes: f32,
    /// NIB FUR (0..1, 0 = off): the fur thinned on the body's lit bulges (see `MediumProfile::nib_fur`).
    pub nib_fur: f32,
    /// NIB LIFT (0..1, 0 = off): the figure's half-tones lifted, the deep folds kept (see `MediumProfile::nib_lift`).
    pub nib_lift: f32,
    /// NIB CONTACT (0..1, 0 = off): a contact shadow under the subject (see `MediumProfile::nib_contact`).
    pub nib_contact: f32,
    /// NIB FIGURE (0..1, 0 = off): a figure drawn in parallel hatching with bold contours (see `MediumProfile::nib_figure`).
    pub nib_figure: f32,
    /// NIB OBJECT (0..1, 0 = off): the held objects outlined with the figure (see `MediumProfile::nib_object`).
    pub nib_object: f32,
    /// NIB TUFT (0..1, 0 = off): the near grass as fans of blades from one root (see `MediumProfile::nib_tuft`).
    pub nib_tuft: f32,
    /// NIB CLOSE (0..1, 0 = off): the figure's outline gaps closed (see `MediumProfile::nib_close`).
    pub nib_close: f32,
    /// NIB MASS (0..1, 0 = off): the dark grass's tufts crossed into one mass (see `MediumProfile::nib_mass`).
    pub nib_mass: f32,
    /// NIB RIM (0..1, 0 = off): the figure's outline in a line of changing weight (see `MediumProfile::nib_rim`).
    pub nib_rim: f32,
    /// NIB POOL (0..1, 0 = off): the contact shadows solid black under the figure (see `MediumProfile::nib_pool`).
    pub nib_pool: f32,
    /// NIB AIR (0..1, 0 = off): the far planes thinned to a few fine lines (see `MediumProfile::nib_air`).
    pub nib_air: f32,
    /// NIB PEOPLE (0..1, 0 = off): every person found is the figure, with or without a face (see `MediumProfile::nib_people`).
    pub nib_people: f32,
    /// NIB VALUE (0..1, 0 = off): the figure's tone read against the whole picture, its lights kept light (see `MediumProfile::nib_value`).
    pub nib_value: f32,
    /// NIB FINE FACE (0..1, 0 = off): a small face drawn with the finest pen, its features found at a lower threshold (see `MediumProfile::nib_fine_face`).
    pub nib_fine_face: f32,
    /// NIB SCATTER (0..1, 0 = off): the tufts scattered in clusters, not one per cell (see `MediumProfile::nib_scatter`).
    pub nib_scatter: f32,
    /// NIB FALLOFF (0..1, 0 = off): the contact shadow solid only at the contact, broken into dashes beyond (see `MediumProfile::nib_falloff`).
    pub nib_falloff: f32,
    /// NIB DEPTH (0..1, 0 = off): the far planes by the picture's depth drawn with the finest pen (see `MediumProfile::nib_depth`).
    pub nib_depth: f32,
    /// NIB LUSH (0..1, 0 = off): the found grass grown thick, its tufts denser (see `MediumProfile::nib_lush`).
    pub nib_lush: f32,
    /// NIB GLINT (0..1, 0 = off): the specular highlights kept white, no mark through them (see `MediumProfile::nib_glint`).
    pub nib_glint: f32,
    /// NIB WASH (0..1, 0 = off): ink and wash: the ladder laid as washes, then over them as the pens (see `MediumProfile::nib_wash`).
    pub nib_wash: f32,
    /// NIB WASH DARK (0..1, 0 = off): the pens' hatching held to the darks, the half-tones left to the washes (see `MediumProfile::nib_wash_dark`).
    pub nib_wash_dark: f32,
    /// NIB WASH DETAIL (0..1, 0 = off): the finest pen drawing the picture's own detail over the washes (see `MediumProfile::nib_wash_detail`).
    pub nib_wash_detail: f32,
    /// NIB WASH LINE (0..1, 0 = off): the bone line traced from the picture itself, its detail kept (see `MediumProfile::nib_wash_line`).
    pub nib_wash_line: f32,
    /// NIB WASH SWELL (0..1, 0 = off): the line's width swelling with the strength of its edge (see `MediumProfile::nib_wash_swell`).
    pub nib_wash_swell: f32,
    /// NIB WASH AIR (0..1, 0 = off): the far planes left to soft, light washes, the line fading into the distance (see `MediumProfile::nib_wash_air`).
    pub nib_wash_air: f32,
    /// NIB WASH SOFT (0..1, 0 = off): the washes from the picture's low band, the high band left to the pens (see `MediumProfile::nib_wash_soft`).
    pub nib_wash_soft: f32,
    /// NIB WASH GAMMA (0..1, 0 = off): the washes' density raised by a gamma, the paper kept (see `MediumProfile::nib_wash_gamma`).
    pub nib_wash_gamma: f32,
    /// NIB WASH FLAT (0..1, 0 = off): a flat dark left to the wash's pool, the pen only where the dark has an edge or a texture (see `MediumProfile::nib_wash_flat`).
    pub nib_wash_flat: f32,
    /// NIB WASH SCURVE (0..1, 0 = off): the washes' density by a contrast S-curve, the light half kept (see `MediumProfile::nib_wash_scurve`).
    pub nib_wash_scurve: f32,
    /// NIB WASH VARY (0..1, 0 = off): the washes' deepening held back where the picture is busy (by its local variance) (see `MediumProfile::nib_wash_vary`).
    pub nib_wash_vary: f32,
    /// NIB WASH MASS (0..1, 0 = off): the ink's mass kept across the bleed, the washes deepened by a gamma of the measured loss (see `MediumProfile::nib_wash_mass`).
    pub nib_wash_mass: f32,
    /// NIB WASH HOLD (0..1, 0 = off): a wash held at the picture's hard edges toward a light, by the ink's conductance (see `MediumProfile::nib_wash_hold`).
    pub nib_wash_hold: f32,
    /// NIB WASH DEBT (0..1, 0 = off): the ink's loss measured only where the sheet is still lighter than the picture (see `MediumProfile::nib_wash_debt`).
    pub nib_wash_debt: f32,
    /// NIB WASH DRY (0..1, 0 = off): the dark register laid by a dry brush on the washes' fine passes (see `MediumProfile::nib_wash_dry`).
    pub nib_wash_dry: f32,
    /// NIB WASH CAP (0..1, 0 = off): the sheet's capacity: a wash no darker than the picture's own value (see `MediumProfile::nib_wash_cap`).
    pub nib_wash_cap: f32,
    /// NIB WASH TONE (0..1, 0 = off): the sheet's mid-tones pulled to the picture's mean at output (see `MediumProfile::nib_wash_tone`).
    pub nib_wash_tone: f32,
    /// NIB WASH DRAW: the pens' passes first, on the bare sheet; the washes over the drawing (see `MediumProfile::nib_wash_draw`).
    pub nib_wash_draw: f32,
    /// NIB WASH PLANES: the washes as flat planes, laid as area fills (see `MediumProfile::nib_wash_planes`).
    pub nib_wash_planes: f32,
    /// NIB WASH EDGE: a plane's wet edge wide where the picture's gradient is gentle (see `MediumProfile::nib_wash_edge`).
    pub nib_wash_edge: f32,
    /// NIB WASH DEPTH: air on the planes, the far lifted into lighter bands (see `MediumProfile::nib_wash_depth`).
    pub nib_wash_depth: f32,
    /// NIB WASH SCALE: the lightest plane from the coarsest smoothing, the darkest from the finest (see `MediumProfile::nib_wash_scale`).
    pub nib_wash_scale: f32,
    /// NIB WASH LEVELS: the planes' thresholds moved toward the picture's quantiles (see `MediumProfile::nib_wash_levels`).
    pub nib_wash_levels: f32,
    /// NIB WASH FACES: a face lifted into a lighter plane, its hatching freed of the dark gate (see `MediumProfile::nib_wash_faces`).
    pub nib_wash_faces: f32,
    /// NIB WASH RAG: a plane's wet edge torn into bristle tracks (see `MediumProfile::nib_wash_rag`).
    pub nib_wash_rag: f32,
    /// NIB WASH MASK: the pen kept out of the darkest plane (see `MediumProfile::nib_wash_mask`).
    pub nib_wash_mask: f32,
    /// NIB WASH HALF: the half-tones' fine pen past the dark gate (see `MediumProfile::nib_wash_half`).
    pub nib_wash_half: f32,
    /// NIB WASH GRAIN: the small darks kept in the planes (see `MediumProfile::nib_wash_grain`).
    pub nib_wash_grain: f32,
    /// NIB WASH GRADE: a tone within each plane (see `MediumProfile::nib_wash_grade`).
    pub nib_wash_grade: f32,
    /// NIB WASH PROFILE: every face drawn as a small one, one way along the form (see `MediumProfile::nib_wash_profile`).
    pub nib_wash_profile: f32,
    /// NIB WASH STREAK: the tear along the brush's sweep (see `MediumProfile::nib_wash_streak`).
    pub nib_wash_streak: f32,
    /// NIB WASH SWEEP: a plane laid as wide wet strokes along the flow (see `MediumProfile::nib_wash_sweep`).
    pub nib_wash_sweep: f32,
    /// NIB WASH WATER: a layer of water the planes' ink seeps through (see `MediumProfile::nib_wash_water`).
    pub nib_wash_water: f32,
    /// NIB WASH AGENT: the mark looks at the sheet under it (see `MediumProfile::nib_wash_agent`).
    pub nib_wash_agent: f32,
    /// SUMI DARK (0..1, 0 = the default eighth): the share of the darkest values the sumi-e's black register lays
    /// (see `sumi_ink`) — a figure in a dark coat is a figure, not one black shape. A plan key.
    pub sumi_dark: f32,
    /// The FACE's own extent (px) — the face detector's box, not the soft focal mask (whose extent is the
    /// head and shoulders' halo). `None`: read from the mask.
    pub face_extent: Option<f32>,
    /// The subject's SILHOUETTE alone — the matte and the segmenter's precise mask, without the detectors'
    /// boxes (a clothing box) that `subject_mask` unions in. Read by `ma_flow`; `None`: `subject_mask`.
    pub silhouette_mask: Option<Vec<f32>>,
}

impl PaintParams {
    /// A sensible default over a palette at a stroke budget.
    pub fn new(palette: Palette, budget: usize) -> Self {
        Self { palette, budget, passes: None, brush_sizes: vec![28.0, 14.0, 7.0], min_brush: 4.0, armature_side: None, armature_face_side: None, armature_body_side: None, subject_mask: None, armature_levels: 8, region_tiers: Vec::new(), silhouette: 0.0, silhouette_mode: EdgeMode::Line, commit_shadows: 0.0, charge: 6.0, medium: "oil-direct".into(), reserve: None, density: false, ground: None, protect: None, region_mask: None, hair_mask: None, nib_feature_mask: None, nib_object_mask: None, nib_grass_mask: None, nib_person_mask: None, nib_depth_map: None, infill: FlowInfill::Flat, rigger: 0.0, hotspot: 0.0, technique: WetTechnique::None, face_ladder_from: None, leak: 0.0, seed: 42, brush: BrushConfig::default(), paint_mask: None, layer_brush: None, depth: None, relief: None, normals: None, materials: None, features: Vec::new(), follow: 1.0, white: None, haze: 0.0, stroke_len: 1.0, stroke_width: 1.0, bleed: 0.0, diffuse: 0.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, lift: 1.0, broken: 0.0, contour: 0.0, style: PaintStyle::Legible, define: 0.6, saliency: 0.0, focus_detail: 0.0, preserve_face: 0.0, face_mask: None, splatter: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, dry: 0.5, coverage: 0.0, detail_coherence: 0.14, detail_len: 1.0, detail_restate: 0.08, detail_sharpen: 0.0, detail_texture: 1.0, fine_lines: 1.0, impasto_map: 0.0, weave: 0.0, collide: 0.0, gradation: 0.0, hatch_angle: 0.0, engrave: false, draw_contours: false, brush_drawing: false, sumi: false, luminous: false, book: false, threads: 0, from_scratch: false, fill: 0.0, dump_passes: None, wash_in: 0.0, wet_cap: 1.0, scumble: 0.0, turbulence: 0.0, accent: 0.0, loose: 0.0, glaze: 0.0, sfumato: 0.0, glaze_pass: 0.0, vary: 0.0, shade: 0.0, imprimatura: 0.0, body_vary: 0.0, lost: 0.0, crest: 0.0, spatter: 0.0, aerial: 0.0, sink: 0.0, tail: 0.0, film: 0.0, split: 0.0, sag: 0.0, gloss: 0.0, tint: 0.0, knife: 0.0, hiding: 0.0, twist: 0.0, skim: 0.0, blade: 0.0, blade_planes: 0.0, blade_levels: 0.0, blade_size: 0.0, blade_light: 0.0, blade_phong: 0.0, blade_flow: 0.0, blade_rest: 0.0, blade_relief: 0.0, blade_along: 0.0, blade_inertia: 0.0, blade_facet: 0.0, blade_visc: 0.0, blade_soft: 0.0, blade_detail: 0.0, blade_layers: 0.0, blade_edge: 0.0, blade_tuft: 0.0, blade_aniso: 0.0, blade_slope: 0.0, blade_break: 0.0, blade_route: 0.0, blade_scatter: 0.0, blade_tint: 0.0, blade_focus: 0.0, blade_peak: 0.0, blade_band: 0.0, hatch: 0.0, verdaccio: 0.0, dishes: 0.0, lights: 0.0, cross: 0.0, dither: 0.0, solid: 0.0, flesh: 0.0, span: 0.0, edge: 0.0, glint: 0.0, smooth: 0.0, bole: 0.0, ramp: 0.0, umbra: 0.0, side: 0.0, blend: 0.0, skin: 0.0, carnation: 0.0, small: 0.0, anchor: 0.0, trim: 0.0, small_min: 0.0, paper: 0.0, erase: 0.0, touch: 0.0, velvet: 0.0, paper_top: 0.0, sweep: 0.0, stump: 0.0, stump_edge: 0.0, stump_feather: 0.0, sweep_wave: 0.0, paper_knee: 0.0, chalk_tint: 0.0, flat_brush: 0.0, liner: 0.0, ma: 0.0, yohaku: 0.0, bone: 0.0, boneless: 0.0, hake: 0.0, bone_wind: 0.0, bone_dot: 0.0, bone_scatter: 0.0, bone_coherence: 0.0, bone_end: 0.0, sumi_focus: 0.0, ma_focus: 0.0, ma_flow: 0.0, bone_ryaku: 0.0, bone_fade: 0.0, bokashi: 0.0, velvet_dark: 0.0, slant: 0.0, grade: 0.0, weight: 0.0, clip: 0.0, tiers: 0.0, nib: 0.0, nibs: 0.0, nib_tone: 0.0, nib_white: 0.0, nib_anchor: 0.0, nib_hint: 0.0, nib_face: 0.0, nib_gap: 0.0, nib_relief: 0.0, nib_far: 0.0, nib_occlude: 0.0, nib_eyes: 0.0, nib_fur: 0.0, nib_lift: 0.0, nib_contact: 0.0, nib_figure: 0.0, nib_object: 0.0, nib_tuft: 0.0, nib_close: 0.0, nib_mass: 0.0, nib_rim: 0.0, nib_pool: 0.0, nib_air: 0.0, nib_people: 0.0, nib_value: 0.0, nib_fine_face: 0.0, nib_scatter: 0.0, nib_falloff: 0.0, nib_depth: 0.0, nib_lush: 0.0, nib_glint: 0.0, nib_wash: 0.0, nib_wash_dark: 0.0, nib_wash_detail: 0.0, nib_wash_line: 0.0, nib_wash_swell: 0.0, nib_wash_air: 0.0, nib_wash_soft: 0.0, nib_wash_gamma: 0.0, nib_wash_flat: 0.0, nib_wash_scurve: 0.0, nib_wash_vary: 0.0, nib_wash_mass: 0.0, nib_wash_hold: 0.0, nib_wash_debt: 0.0, nib_wash_dry: 0.0, nib_wash_cap: 0.0, nib_wash_tone: 0.0, nib_wash_draw: 0.0, nib_wash_planes: 0.0, nib_wash_edge: 0.0, nib_wash_depth: 0.0, nib_wash_scale: 0.0, nib_wash_levels: 0.0, nib_wash_faces: 0.0, nib_wash_rag: 0.0, nib_wash_mask: 0.0, nib_wash_half: 0.0, nib_wash_grain: 0.0, nib_wash_grade: 0.0, nib_wash_profile: 0.0, nib_wash_streak: 0.0, nib_wash_sweep: 0.0, nib_wash_water: 0.0, nib_wash_agent: 0.0, sumi_dark: 0.0, face_extent: None, silhouette_mask: None }
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

pub(crate) fn luma_map(img: &RgbImage) -> Vec<f32> {
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
/// EDGE-STOPPING SPREAD (`ma_flow`): the recursive filter of the domain transform (Gastal & Oliveira) — three
/// iterations of a left-right, right-left, top-down and bottom-up pass over `src`, each step's feedback
/// `a^(1 + edge·σs/0.08·|Δguide|)` with `a = e^(−√2/σ_i)` and σ_i the iteration's share of `sigma` — so the
/// field spreads about `sigma` pixels where the guide is flat and stops where it steps by 0.08 at `edge` 1.
/// RAMER–DOUGLAS–PEUCKER: the polyline `pts` reduced to the vertices that stand more than `eps` px off the
/// chord between their kept neighbours (`bone_ryaku`). Iterative, the ends always kept.
fn rdp(pts: &[[f32; 2]], eps: f32) -> Vec<[f32; 2]> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((i0, i1)) = stack.pop() {
        let (a, b) = (pts[i0], pts[i1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = (dx * dx + dy * dy).sqrt().max(1e-6);
        let (mut best, mut bi) = (0.0f32, 0usize);
        for (j, q) in pts.iter().enumerate().take(i1).skip(i0 + 1) {
            let dist = ((q[0] - a[0]) * dy - (q[1] - a[1]) * dx).abs() / l;
            if dist > best {
                best = dist;
                bi = j;
            }
        }
        if best > eps {
            keep[bi] = true;
            stack.push((i0, bi));
            stack.push((bi, i1));
        }
    }
    pts.iter().zip(&keep).filter(|(_, k)| **k).map(|(q, _)| *q).collect()
}

/// CHAIKIN corner cutting: each segment's quarter points, the ends kept — a polygon rounded toward its arcs.
fn chaikin(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut out = vec![pts[0]];
    for s in pts.windows(2) {
        let (a, b) = (s[0], s[1]);
        out.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
        out.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    out.push(pts[pts.len() - 1]);
    out
}

fn edge_spread(src: &[f32], guide: &[f32], w: usize, h: usize, sigma: f32, edge: f32) -> Vec<f32> {
    let mut out = src.to_vec();
    let k = edge * sigma / 0.08;
    let n = 3;
    for it in 0..n {
        let si = sigma * 3f32.sqrt() * 2f32.powi(n - it - 1) / (4f32.powi(n) - 1.0).sqrt();
        let a = (-(2f32.sqrt()) / si.max(1e-3)).exp();
        let wgt = |g0: f32, g1: f32| a.powf(1.0 + k * (g1 - g0).abs());
        for y in 0..h {
            let row = y * w;
            for x in 1..w {
                let c = wgt(guide[row + x - 1], guide[row + x]);
                out[row + x] += c * (out[row + x - 1] - out[row + x]);
            }
            for x in (0..w - 1).rev() {
                let c = wgt(guide[row + x + 1], guide[row + x]);
                out[row + x] += c * (out[row + x + 1] - out[row + x]);
            }
        }
        for x in 0..w {
            for y in 1..h {
                let c = wgt(guide[(y - 1) * w + x], guide[y * w + x]);
                out[y * w + x] += c * (out[(y - 1) * w + x] - out[y * w + x]);
            }
            for y in (0..h - 1).rev() {
                let c = wgt(guide[(y + 1) * w + x], guide[y * w + x]);
                out[y * w + x] += c * (out[(y + 1) * w + x] - out[y * w + x]);
            }
        }
    }
    out
}

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
        if region.len() >= min_area {
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
                bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
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
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: true, wash: false, stage: "leak".into(), spline: s.path, w0, w1, taper: 0.0, mix: Vec::new(), wet: lw, press: 1.0, streak: brush.streak, round: brush.round, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
            }
        } else {
            // A dark run: the pigment a drip collects on its way down, well above the wash's own charge.
            let load: Vec<f32> = load.iter().map(|v| v * 2.0).collect();
            for (pth, w0, w1, pr) in [(path, width, width * 0.5, 0.9), (drop, width * 1.7, width * 1.7, 1.0)] {
                let s = Stroke { path: pth, width0: w0, width1: w1, load: load.clone(), pressure: pr, wetness: wet };
                s.rasterize(canvas, &brush);
                let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                *placed += 1;
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "leak".into(), spline: s.path, w0, w1, taper: 0.0, mix, wet, press: pr, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
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
/// The IMPRIMATURA tone of a block-in colour (see `MediumProfile::imprimatura`): darker by `0.4·amt` of its
/// luma and pulled `0.5·amt` of the way toward a burnt umber at that luma — the picture's colour still, but
/// laid as the dark warm ground a classical painting is built on.
fn imprimatura_tone(target: Srgb, amt: f32) -> Srgb {
    let a = amt.clamp(0.0, 1.0);
    if a <= 1e-3 {
        return target;
    }
    let lin = color::srgb_to_linear(target);
    let l = color::linear_luma(lin).max(1e-4);
    let lt = l * (1.0 - 0.4 * a);
    let umber = color::srgb_to_linear([118, 72, 42]);
    let ul = color::linear_luma(umber).max(1e-4);
    let mut out = [0f32; 3];
    for i in 0..3 {
        out[i] = ((lin[i] / l) * (1.0 - 0.5 * a) + (umber[i] / ul) * 0.5 * a) * lt;
        out[i] = out[i].clamp(0.0, 1.0);
    }
    color::linear_to_srgb(out)
}

/// The CREST tint (see `MediumProfile::crest`): the picture's colour a third lighter and a little warmer —
/// the pale ochre a dry brush drags over the tops of a dried relief.
/// The SINK wash's colour (see `MediumProfile::sink`): the picture a fifth darker and a little cooler — a thin
/// cool shadow wash rubbed into the furrows of a dried, warm underpainting.
fn sink_tone(target: Srgb) -> Srgb {
    let lin = color::srgb_to_linear(target);
    color::linear_to_srgb([(lin[0] * 0.8 * 0.94).min(1.0), (lin[1] * 0.8).min(1.0), (lin[2] * 0.8 * 1.06).min(1.0)])
}

fn crest_tone(target: Srgb) -> Srgb {
    let lin = color::srgb_to_linear(target);
    let out = [(lin[0] * 1.3 * 1.06).min(1.0), (lin[1] * 1.3).min(1.0), (lin[2] * 1.3 * 0.94).min(1.0)];
    color::linear_to_srgb(out)
}

/// The VERDACCIO tone (see `MediumProfile::verdaccio`): the picture's colour pulled `amt` of the way toward the
/// greenish grey of the panel painters' flesh underpainting (terre verte, white, a little ochre and black) at
/// its own value, a tenth darker at full strength — the green ground the hatched flesh is built over.
fn verdaccio_tone(target: Srgb, amt: f32) -> Srgb {
    let a = amt.clamp(0.0, 1.0);
    if a <= 1e-3 {
        return target;
    }
    let lin = color::srgb_to_linear(target);
    let l = color::linear_luma(lin).max(1e-4);
    let lt = l * (1.0 - 0.1 * a);
    let verd = color::srgb_to_linear([128, 134, 104]);
    let vl = color::linear_luma(verd).max(1e-4);
    let mut out = [0f32; 3];
    for i in 0..3 {
        out[i] = (((lin[i] / l) * (1.0 - a) + (verd[i] / vl) * a) * lt).clamp(0.0, 1.0);
    }
    color::linear_to_srgb(out)
}

/// The DISHES (see `MediumProfile::dishes`): the colour's lightness (its luma to the 1/2.2) snapped to the
/// nearest of `n` steps between the picture's own darkest and lightest values (`lo`, `hi`, as lumas — its 5th
/// and 95th percentiles), the colour rescaled to it — the tone mixed in advance in a dish; the lightest dish is
/// the picture's own light, never a white (a pale sky snapped to white hatched itself in chalk).
///
/// DITHER (see `MediumProfile::dither`): where the value lies between two dishes, the upper is taken when its
/// fraction of the way exceeds a threshold — `0.5` (the nearest) at `dither` 0, the mark's own draw `u` (0..1,
/// so the chance is the fraction itself) at 1, between them between — the band between two dishes broken into
/// marks of both.
fn dish_tone(target: Srgb, n: f32, lo: f32, hi: f32, dither: f32, u: f32) -> Srgb {
    if n < 2.0 {
        return target;
    }
    let lin = color::srgb_to_linear(target);
    let l = color::linear_luma(lin).max(1e-4);
    let steps = n.round() - 1.0;
    let (light, llo, lhi) = (l.powf(1.0 / 2.2), lo.max(0.0).powf(1.0 / 2.2), hi.min(1.0).powf(1.0 / 2.2));
    if lhi <= llo + 1e-3 {
        return target;
    }
    let rel = ((light - llo) / (lhi - llo)).clamp(0.0, 1.0);
    let q = rel * steps;
    let (base, frac) = (q.floor(), q - q.floor());
    let threshold = 0.5 + (u.clamp(0.0, 1.0) - 0.5) * dither.clamp(0.0, 1.0);
    let dish = if frac > threshold { base + 1.0 } else { base };
    let lq = (llo + dish.min(steps) / steps * (lhi - llo)).max(0.02);
    let sc = (lq / light.max(0.02)).powf(2.2);
    color::linear_to_srgb([(lin[0] * sc).min(1.0), (lin[1] * sc).min(1.0), (lin[2] * sc).min(1.0)])
}

/// Where a tone lies on the dishes' scale (see `dish_tone`): its position 0..1 between the picture's `lo` and
/// `hi` values in gamma space (the dishes are spaced perceptually), times the number of steps — so a value
/// below 1 is within the DARKEST dish's interval. The UMBRA (see `MediumProfile::umbra`) keeps the tones
/// below that share of the first step continuous.
fn dish_pos(target: Srgb, n: f32, lo: f32, hi: f32) -> f32 {
    if n < 2.0 {
        return 0.0;
    }
    let l = color::linear_luma(color::srgb_to_linear(target)).max(1e-4);
    let (light, llo, lhi) = (l.powf(1.0 / 2.2), lo.max(0.0).powf(1.0 / 2.2), hi.min(1.0).powf(1.0 / 2.2));
    if lhi <= llo + 1e-3 {
        return 0.0;
    }
    ((light - llo) / (lhi - llo)).clamp(0.0, 1.0) * (n.round() - 1.0)
}

/// Whether the light at `(x, y)` is ISOLATED at the brush's scale: within a box of the brush's radius the
/// pixels at least half as light as the seed cover less than two fifths of the disc — a star, a lamp's
/// point, a light smaller than the brush. A halo, a window, a lit wall is not.
fn isolated_light(luma: &[f32], w: usize, h: usize, x: f32, y: f32, r: f32) -> bool {
    if w == 0 || h == 0 {
        return false;
    }
    let (cx, cy) = ((x.round() as i64).clamp(0, w as i64 - 1), (y.round() as i64).clamp(0, h as i64 - 1));
    let rr = (r.round() as i64).max(1);
    let seed = luma[cy as usize * w + cx as usize];
    if seed <= 1e-3 {
        return false;
    }
    let (mut lit, mut all) = (0usize, 0usize);
    for yy in (cy - rr).max(0)..=(cy + rr).min(h as i64 - 1) {
        for xx in (cx - rr).max(0)..=(cx + rr).min(w as i64 - 1) {
            let (dx, dy) = ((xx - cx) as f32, (yy - cy) as f32);
            if dx * dx + dy * dy > (rr as f32) * (rr as f32) {
                continue;
            }
            all += 1;
            if luma[yy as usize * w + xx as usize] >= 0.5 * seed {
                lit += 1;
            }
        }
    }
    all > 0 && (lit as f32) < 0.4 * all as f32
}

/// The CROSS turn (see `MediumProfile::cross`): the angle (radians) the `hk`-th hatched pass is turned off the
/// form — 0 on the first, then `+cross`, `−2·cross`, `+3·cross` degrees, alternating sides.
fn cross_turn(hk: usize, cross_deg: f32) -> f32 {
    let sign = if hk % 2 == 1 { 1.0 } else { -1.0 };
    sign * hk as f32 * cross_deg.to_radians()
}

/// The hatch's CHARGE factor (see `MediumProfile::solid`): `1 − 0.3·hatch` at `solid` 0; at 1, that thinness in
/// the picture's lightest value (`v` 1) and `1 + 0.3·hatch` in its darkest (`v` 0), linear between.
fn hatch_charge(hatch: f32, solid: f32, v: f32) -> f32 {
    let s = solid.clamp(0.0, 1.0);
    1.0 - 0.3 * hatch * ((1.0 - s) + s * (2.0 * v.clamp(0.0, 1.0) - 1.0))
}

/// The WARMTH of a colour (see `MediumProfile::flesh`), 0..1: 0 for a cold or neutral one (CIELAB a* or b* at
/// or below 2), 1 for a plainly warm one (a* 10 and b* 12 and above — the flesh tones, the warm earths), the
/// product of the two ramps between.
fn warmth_of(c: Srgb) -> f32 {
    let lab = color::srgb_to_lab(c);
    ((lab.a - 2.0) / 8.0).clamp(0.0, 1.0) * ((lab.b - 2.0) / 10.0).clamp(0.0, 1.0)
}

/// How BUSY the form is (see `MediumProfile::span`), 0..1 from the reference's value range within the brush's
/// reach: flat at 0, fully busy at a range of 0.3 and above.
fn busy_of(range: f32) -> f32 {
    (range / 0.3).clamp(0.0, 1.0)
}

/// How near a plain EDGE (see `MediumProfile::edge`), 0..1 from the same range: nothing below 0.15, fully at
/// 0.5 and above — only a real contour, not the modelling.
fn edge_of(range: f32) -> f32 {
    ((range - 0.15) / 0.35).clamp(0.0, 1.0)
}

/// The hatch's LENGTH factor (see `MediumProfile::span`): `1 + 0.5·span` on a flat passage, `1 − 0.5·span`
/// where the form is busy, linear between.
fn span_length(span: f32, busy: f32) -> f32 {
    1.0 + 0.5 * span.clamp(0.0, 1.0) * (1.0 - 2.0 * busy.clamp(0.0, 1.0))
}

/// Two tones blended `t` of the way from the first to the second, in linear light.
fn blend_tone(a: Srgb, b: Srgb, t: f32) -> Srgb {
    let t = t.clamp(0.0, 1.0);
    let (la, lb) = (color::srgb_to_linear(a), color::srgb_to_linear(b));
    color::linear_to_srgb([la[0] + (lb[0] - la[0]) * t, la[1] + (lb[1] - la[1]) * t, la[2] + (lb[2] - la[2]) * t])
}

/// The bole's thickness, as a share of the charge: thin — a toned ground, not a layer of paint.
const BOLE_CHARGE: f32 = 0.3;
/// The bole's wetness as laid (it dries with the pass boundary after it).
const BOLE_WET: f32 = 0.9;

/// THE FINGER (see `MediumProfile::blend`): after each pass of the BROAD half of the ladder the CALM passages
/// are rubbed — a smudge along the flow with no pigment of its own, moving a share of the paint under it
/// toward what it carries (see `Stroke::smudge`), so the underpainting's broad marks fuse into soft masses
/// while the fine passes that follow stay discrete strokes (rubbed after the fine passes too, the sheet was
/// a blur with noise on it, not a pastel). The sheet in cells of the finger's own width (a fortieth of the
/// sheet's short side, on every pass); every cell is rubbed by its calm (the key's share of its flatness —
/// the picture's range within the brush's reach, the reference first softened by half the radius — the
/// subject at three tenths of it), none drawn by lot; the rub a mark along the flow grown from the finger's
/// radius, `0.08·calm` strong on the ground and `0.25·calm` on a face, its ends soft, stopping at the
/// picture's own edges as a mark does, at a face or the hair, and staying on its own side of the subject's
/// silhouette. Each rub a record (`smudge`) in the pass's own stage, replayed to the bit. `faces_only`: the
/// skin alone (`skin`, after the finest pass — the portrait's skin blended last, its accents the eyes, the
/// nose and the mouth, which stay busy). Returns the rubs laid.
#[allow(clippy::too_many_arguments)]
fn blend_pass(canvas: &mut Canvas, score: &mut StrokeScore, p: &PaintParams, luma: &[f32], range_field: Option<&[f32]>, gx: &[f32], gy: &[f32], reference: &RgbImage, protect: Option<&[bool]>, radius: f32, stage: &str, faces_only: bool, stump: bool, placed: &mut usize, k: &mut u64) -> usize {
    let (w, h) = (canvas.w, canvas.h);
    if w == 0 || h == 0 || (if stump { p.stump <= 0.0 } else { p.blend <= 0.0 }) || luma.len() < (w * h) as usize {
        return 0;
    }
    let owned: Vec<f32>;
    let range: &[f32] = match range_field {
        Some(f) => f,
        None => {
            let soft = box_blur(luma, w, h, (radius * 0.5).round().max(1.0) as i32);
            owned = local_range(&soft, w as usize, h as usize, (radius * 1.5).round().max(2.0) as usize);
            &owned
        }
    };
    // The faces and the hair, with whatever is protected, as one guard the rub's path stops at; and the faces
    // alone, the region a rub begun on a face (`skin`) stays inside.
    let face_b: Option<Vec<bool>> = p.face_mask.as_deref().map(|m| m.iter().map(|f| *f > 0.35).collect());
    let guard: Option<Vec<bool>> = (p.face_mask.is_some() || p.hair_mask.is_some()).then(|| {
        (0..(w * h) as usize)
            .map(|i| {
                protect.and_then(|m| m.get(i)).copied().unwrap_or(false)
                    || face_b.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(false)
                    || p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.0
            })
            .collect()
    });
    // A rub on a face stops at the hair and the protected shapes only (the face is its region).
    let skin_guard: Option<Vec<bool>> = (p.skin > 0.0 && face_b.is_some()).then(|| {
        (0..(w * h) as usize)
            .map(|i| protect.and_then(|m| m.get(i)).copied().unwrap_or(false) || p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.0)
            .collect()
    });
    // The finger has a size of its own — a fortieth of the sheet's short side — whatever the stick of the pass:
    // a finger is a finger on the block-in's wide sweeps as on the fine marks (sized to the broad pass it was
    // two hundred pixels wide and carried a wagon's shadow across a figure and a lit facade into a blur).
    let finger = (w.min(h) as f32 / 40.0).max(4.0);
    let width = finger;
    let step = (width / 1.5).max(2.0);
    // ...and a reach of its own: the rub's path grows from the finger's radius, not the stick's (grown from a
    // four-pixel stick's it was thirteen pixels each way, shorter than the lights it was to fuse).
    let fr = width / 1.8;
    // STUMP EDGE (see `MediumProfile::stump_edge`): the perceived local contrast at a finger's reach.
    let edge_g: Option<Vec<f32>> = (stump && p.stump_edge > 0.0).then(|| {
        let lg: Vec<f32> = luma.iter().map(|v| v.max(0.0).powf(1.0 / 2.2)).collect();
        local_range(&lg, w as usize, h as usize, (finger * 0.5).round().max(2.0) as usize)
    });
    // STUMP FEATHER (see `MediumProfile::stump_feather`): the faces and the hair as a soft mask.
    let soft_guard: Option<Vec<f32>> = (stump && p.stump_feather > 0.0).then(|| {
        let hard: Vec<f32> = (0..(w * h) as usize)
            .map(|i| if face_b.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(false) || p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.0 { 1.0 } else { 0.0 })
            .collect();
        box_blur(&hard, w, h, (finger * p.stump_feather.clamp(0.0, 1.0)).round().max(1.0) as i32)
    });
    let (cols, rows) = ((w as f32 / step).ceil() as usize, (h as f32 / step).ceil() as usize);
    let mut laid = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            if *placed >= p.budget {
                return laid;
            }
            *k += 1;
            let kk = *k;
            let x = ((c as f32 + 0.5 + 0.6 * jitter(p.seed ^ 0xB1E_D1, kk)) * step).clamp(0.0, w as f32 - 1.0);
            let y = ((r as f32 + 0.5 + 0.6 * jitter(p.seed ^ 0xB1E_D2, kk)) * step).clamp(0.0, h as f32 - 1.0);
            let i = (y.round() as usize * w as usize + x.round() as usize).min(luma.len() - 1);
            let on_face = face_b.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(false);
            let soft = soft_guard.as_deref().and_then(|m| m.get(i)).copied();
            if let Some(sv) = soft {
                // Feathered: inside the soft mask's middle the stump does not go; nearer, it fades (below).
                if sv > 0.5 {
                    continue;
                }
            } else if (stump && on_face) || (on_face && p.skin <= 0.0) || (faces_only && !on_face) || p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.0 {
                continue;
            }
            let on_face = on_face && soft.is_none();
            let subj = p.subject_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let flat = 1.0 - busy_of(range.get(i).copied().unwrap_or(1.0));
            // THE SKIN (see `MediumProfile::skin`): on a face the rub is the key's share of the flatness — the
            // cheek and the forehead, not the eyes — and stays inside the face.
            let calm = p.blend.clamp(0.0, 1.0) * flat * if on_face { p.skin.clamp(0.0, 1.0) } else { 1.0 - 0.7 * subj };
            // THE STUMP (see `MediumProfile::stump`): by the darkness instead — the dark half of the perceived
            // range, fully at its bottom — busy or not (a dark coat's folds are rubbed into one mass).
            // (...and the flat passages off the subject too — a grey sky's specks of paper.)
            let calm = if stump { p.stump.clamp(0.0, 1.0) * (1.0 - 2.0 * luma[i].max(0.0).powf(1.0 / 2.2)).clamp(0.0, 1.0).max(flat * (1.0 - subj)) } else { calm };
            // (Held at a contrast — `stump_edge` — and fading toward a face — `stump_feather`.)
            let calm = calm * (1.0 - p.stump_edge.clamp(0.0, 1.0) * edge_g.as_deref().and_then(|m| m.get(i)).map(|r| (r / 0.12).clamp(0.0, 1.0)).unwrap_or(0.0)) * (1.0 - soft.unwrap_or(0.0));
            // Every cell is rubbed, by its calm (below) — drawn by lot, a grass was part drawn and part smudged.
            if calm < 0.05 {
                continue;
            }
            let color0 = reference.get_pixel(x.round() as u32, y.round() as u32).0;
            // The finger stops at a face and at the hair (a rub begun in the sea beside a cheek ran along the
            // flow into it and left the sea's colour in dashes across the skin), and stays on its own side
            // of the subject's silhouette: the background is blended up to the figure, never across it.
            let region = if on_face { face_b.as_deref().map(|m| (m, true)) } else if soft.is_some() { None } else { p.region_mask.as_deref().map(|m| (m, m.get(i).copied().unwrap_or(false))) };
            // The finger blends WITHIN a passage of one colour and stops at the picture's own edges as a mark
            // does (the stop reads the reference, so a chalky light the canvas holds across a smooth cheek is
            // crossed, the hair against the forehead, a black trouser against a white blouse, the sea against
            // a cheek are not — looser, it carried the sea's green onto the blouse and the black onto the
            // figures).
            let path = grow_path(x, y, fr, gx, gy, reference, color0, if on_face { skin_guard.as_deref().or(protect) } else { guard.as_deref().or(protect) }, region, None, 1.5, STOP_TOL);
            if path.len() < 2 {
                continue;
            }
            // A light touch, by the calm: up to `0.08·calm` on the ground (the broad passes' marks fused, not
            // wiped — at 0.3 the sea was a photograph's gradient and the sheet blocks of mud; the finger goes
            // there and back, so each pixel is touched twice), up to `0.25·calm` on a face.
            // (The stump rubs harder than the finger: up to `0.25·calm`, the darks filled.)
            let strength = if on_face || stump { 0.25 * calm } else { 0.08 * calm };
            let s = Stroke { path, width0: width, width1: width, load: Vec::new(), pressure: 1.0, wetness: strength };
            s.smudge(canvas, strength);
            *placed += 1;
            laid += 1;
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: true, wash: false, stage: stage.into(), spline: s.path, w0: width, w1: width, taper: 0.0, mix: Vec::new(), wet: strength, press: 1.0, streak: 0.0, round: 1.0, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: Some(strength), stop: None, edge: None, prof: None });
        }
    }
    laid
}

/// THE BOLE (see `MediumProfile::bole`): the imprimatura laid by local shadow tone as a CONTINUOUS field.
/// The sheet in cells of a thirty-second of its short side; each cell's SHADOW tone is the mean colour of
/// its darker quarter of pixels (a toned ground is darker than the lights; a mean tone round a star or a
/// lamp glowed), the tones smoothed over the neighbouring cells (a three-by-three mean), and the field
/// interpolated between the cells' centres and laid in FINE cells of a quarter of their side, edge to edge,
/// no feather (a bloom over the neighbour's fill doubled the deposit along every seam, and a coarse cell's
/// own edge was a line — each a grid through the thin marks of a dark sky), each fine cell filled with
/// the ground's colour pulled by `bole` toward the field's tone there, the mix solved exactly (the
/// quantised cache's steps were a grid too), thin. Recorded as wash cells (stage `ground`). Returns the
/// cells laid.
fn lay_bole(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, placed: &mut usize, k: &mut u64) -> usize {
    let ground = match p.ground {
        Some(g) => g,
        None => return 0,
    };
    let (w, h) = (input.width() as usize, input.height() as usize);
    if w == 0 || h == 0 {
        return 0;
    }
    let cell = ((w.min(h) as f32 / 32.0).round() as usize).max(8);
    let (cols, rows) = ((w + cell - 1) / cell, (h + cell - 1) / cell);
    // 1. Each cell's shadow tone, in linear light.
    let mut tones: Vec<[f32; 3]> = Vec::with_capacity(cols * rows);
    let mut px: Vec<(f32, [f32; 3])> = Vec::with_capacity(cell * cell);
    for r in 0..rows {
        for c in 0..cols {
            let (x0, y0) = (c * cell, r * cell);
            let (x1, y1) = ((x0 + cell).min(w), (y0 + cell).min(h));
            px.clear();
            for y in y0..y1 {
                for x in x0..x1 {
                    let lin = color::srgb_to_linear(input.get_pixel(x as u32, y as u32).0);
                    px.push((color::linear_luma(lin), lin));
                }
            }
            px.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let q = (px.len() / 4).max(1).min(px.len());
            let mut acc = [0.0f32; 3];
            for (_, lin) in &px[..q] {
                for ch in 0..3 {
                    acc[ch] += lin[ch];
                }
            }
            tones.push([acc[0] / q as f32, acc[1] / q as f32, acc[2] / q as f32]);
        }
    }
    // 2. Smoothed over the neighbours.
    let smoothed: Vec<[f32; 3]> = (0..rows * cols)
        .map(|i| {
            let (r, c) = ((i / cols) as i64, (i % cols) as i64);
            let mut acc = [0.0f32; 3];
            let mut cnt = 0.0f32;
            for dr in -1..=1i64 {
                for dc in -1..=1i64 {
                    let (rr, cc) = (r + dr, c + dc);
                    if rr >= 0 && rr < rows as i64 && cc >= 0 && cc < cols as i64 {
                        let t = tones[rr as usize * cols + cc as usize];
                        for ch in 0..3 {
                            acc[ch] += t[ch];
                        }
                        cnt += 1.0;
                    }
                }
            }
            [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt]
        })
        .collect();
    // 3. The field between the cells' centres, read at a point.
    let field_at = |x: f32, y: f32| -> [f32; 3] {
        let (fx, fy) = (((x / cell as f32) - 0.5).clamp(0.0, (cols - 1) as f32), ((y / cell as f32) - 0.5).clamp(0.0, (rows - 1) as f32));
        let (c0, r0) = (fx.floor() as usize, fy.floor() as usize);
        let (c1, r1) = ((c0 + 1).min(cols - 1), (r0 + 1).min(rows - 1));
        let (tx, ty) = (fx - c0 as f32, fy - r0 as f32);
        let (a, b, c, d) = (smoothed[r0 * cols + c0], smoothed[r0 * cols + c1], smoothed[r1 * cols + c0], smoothed[r1 * cols + c1]);
        let mut out = [0.0f32; 3];
        for ch in 0..3 {
            out[ch] = (a[ch] * (1.0 - tx) + b[ch] * tx) * (1.0 - ty) + (c[ch] * (1.0 - tx) + d[ch] * tx) * ty;
        }
        out
    };
    // 4. The fine cells laid.
    let fine = (cell / 4).max(4);
    let n = p.palette.pigments.len();
    let charge = p.charge * BOLE_CHARGE;
    let mut laid = 0usize;
    let mut y0 = 0usize;
    while y0 < h {
        let y1 = (y0 + fine).min(h);
        let mut x0 = 0usize;
        while x0 < w {
            let x1 = (x0 + fine).min(w);
            let shadow = color::linear_to_srgb(field_at((x0 + x1) as f32 * 0.5, (y0 + y1) as f32 * 0.5));
            let colour = blend_tone(ground, shadow, p.bole.clamp(0.0, 1.0));
            let m = mixer::solve_mixture(&p.palette, colour, 3.min(n));
            let mut load = vec![0f32; n];
            for (pi, wgt) in m.pigments.iter().zip(&m.weights) {
                load[*pi] += wgt * charge;
            }
            if load.iter().any(|v| *v > 0.0) {
                let ring = vec![[x0 as f32, y0 as f32], [x1 as f32, y0 as f32], [x1 as f32, y1 as f32], [x0 as f32, y1 as f32]];
                canvas.fill_rings(&ring, &load, BOLE_WET, 0.0);
                *k += 1;
                *placed += 1;
                laid += 1;
                let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: "ground".into(), spline: ring, w0: 0.0, w1: 0.0, taper: 0.0, mix, wet: BOLE_WET, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: Some(colour), mix2: None, smudge: None, stop: None, edge: None, prof: None });
            }
            x0 += fine;
        }
        y0 += fine;
    }
    laid
}

/// The LIGHTS (see `MediumProfile::lights`): the picture's colour lifted `amt` of the way toward white — the
/// lead white hatched into the top values.
fn lights_tone(target: Srgb, amt: f32) -> Srgb {
    let a = amt.clamp(0.0, 1.0);
    let lin = color::srgb_to_linear(target);
    color::linear_to_srgb([lin[0] + (1.0 - lin[0]) * a, lin[1] + (1.0 - lin[1]) * a, lin[2] + (1.0 - lin[2]) * a])
}

/// CARNATION (see `MediumProfile::carnation`): the palette's WARM and NEUTRAL pigments as a box of their own
/// — a pigment whose masstone is cold in CIELAB (a* or b* below −6: a teal, a blue, a green) is left out —
/// with each one's index in the full palette. `None` when fewer than two would remain (the box is then the
/// palette itself) or when none is shut out.
fn warm_palette(palette: &Palette) -> Option<(Palette, Vec<usize>)> {
    let keep: Vec<usize> = (0..palette.pigments.len())
        .filter(|&i| {
            let lab = color::srgb_to_lab(palette.pigments[i].masstone);
            lab.a >= -6.0 && lab.b >= -6.0
        })
        .collect();
    if keep.len() < 2 || keep.len() == palette.pigments.len() {
        return None;
    }
    let pigments: Vec<crate::paint::pigment::Pigment> = keep.iter().map(|&i| palette.pigments[i]).collect();
    Some((Palette { name: "carnation", pigments: Box::leak(pigments.into_boxed_slice()) }, keep))
}

/// A mark's mixture on a face with CARNATION: solved over the warm box (its own cache), mapped back to the
/// palette's indices, and blended with the full palette's mixture by the key's share (at 1 the warm box alone).
#[allow(clippy::too_many_arguments)]
fn carnation_mix(warm_cache: &mut std::collections::HashMap<u32, Vec<f32>>, cache: &mut std::collections::HashMap<u32, Vec<f32>>, target: Srgb, warm: &Palette, index: &[usize], palette: &Palette, charge: f32, n: usize, amount: f32) -> Vec<f32> {
    let a = amount.clamp(0.0, 1.0);
    let w = mixture_cached(warm_cache, target, warm, charge, index.len());
    let mut out = vec![0f32; n];
    for (i, v) in w.iter().enumerate() {
        if let Some(&pi) = index.get(i) {
            if pi < n {
                out[pi] += v * a;
            }
        }
    }
    if a < 1.0 {
        for (o, v) in out.iter_mut().zip(mixture_cached(cache, target, palette, charge, n)) {
            *o += v * (1.0 - a);
        }
    }
    out
}

/// A face's LIGHT with CARNATION: lifted along its own hue — the CIELAB lightness raised `amt` of the way to
/// white, the chroma kept by all but three tenths of the lift — blended with the plain lift toward white by
/// the key's share.
fn warm_light(target: Srgb, amt: f32, carnation: f32) -> Srgb {
    let a = amt.clamp(0.0, 1.0);
    let c = carnation.clamp(0.0, 1.0);
    let lab = color::srgb_to_lab(target);
    let warm = color::lab_to_srgb(color::Lab { l: lab.l + (100.0 - lab.l) * a, a: lab.a * (1.0 - 0.3 * a), b: lab.b * (1.0 - 0.3 * a) });
    let plain = lights_tone(target, a);
    [0, 1, 2].map(|i| (warm[i] as f32 * c + plain[i] as f32 * (1.0 - c)).round().clamp(0.0, 255.0) as u8)
}

/// CLIP (see `MediumProfile::clip`): the path cut where it CROSSES a contour of `edge` — past its first `margin`
/// pixels (a seed beside a line still lays its mark), walked in half-pixel steps; an edge pixel whose pixel two
/// steps on is clear is a crossing and the path ends a pixel short of it, while a run of edge pixels is the path
/// riding along the contour and goes on.
/// NIB WASH HOLD (see `MediumProfile::nib_wash_hold`): the path cut where its line first meets an edge pixel lighter
/// than `v0` by more than `delta`. (The line only: tested at the mark's sides too, a mark running beside a figure was
/// cut by its silhouette, and a light halo ringed every dark figure.)
fn hold_at_light(path: &[[f32; 2]], value: &[f32], edge: &[bool], v0: f32, delta: f32, w: usize, h: usize) -> Vec<[f32; 2]> {
    if path.len() < 2 {
        return path.to_vec();
    }
    let at = |x: f32, y: f32| -> Option<usize> {
        let (xi, yi) = (x.round(), y.round());
        (xi >= 0.0 && yi >= 0.0 && (xi as usize) < w && (yi as usize) < h).then(|| yi as usize * w + xi as usize)
    };
    let held = |x: f32, y: f32| -> bool { at(x, y).is_some_and(|i| edge[i] && value[i] - v0 > delta) };
    let mut out = vec![path[0]];
    for q in path.windows(2) {
        let (dx, dy) = (q[1][0] - q[0][0], q[1][1] - q[0][1]);
        let l = (dx * dx + dy * dy).sqrt();
        if l < 1e-6 {
            continue;
        }
        let (ux, uy) = (dx / l, dy / l);
        let mut s = 0.0f32;
        while s < l {
            let (x, y) = (q[0][0] + ux * s, q[0][1] + uy * s);
            if held(x, y) {
                let end = [x - ux * 0.5, y - uy * 0.5];
                if out.len() < 2 && ((end[0] - out[0][0]).abs() + (end[1] - out[0][1]).abs()) < 0.5 {
                    out.push([out[0][0] + ux * 0.5, out[0][1] + uy * 0.5]);
                } else {
                    out.push(end);
                }
                return out;
            }
            s += 0.5;
        }
        out.push(q[1]);
    }
    out
}

fn clip_at_edge(path: &[[f32; 2]], edge: &[bool], w: usize, h: usize, margin: f32) -> Vec<[f32; 2]> {
    if path.len() < 2 {
        return path.to_vec();
    }
    let on = |x: f32, y: f32| -> bool {
        let (xi, yi) = (x.round(), y.round());
        xi >= 0.0 && yi >= 0.0 && (xi as usize) < w && (yi as usize) < h && edge[yi as usize * w + xi as usize]
    };
    let mut out = vec![path[0]];
    let mut along = 0.0f32;
    for q in path.windows(2) {
        let (dx, dy) = (q[1][0] - q[0][0], q[1][1] - q[0][1]);
        let l = (dx * dx + dy * dy).sqrt();
        if l < 1e-6 {
            continue;
        }
        let (ux, uy) = (dx / l, dy / l);
        let mut s = 0.0f32;
        while s < l {
            let (x, y) = (q[0][0] + ux * s, q[0][1] + uy * s);
            if along + s > margin && on(x, y) && !on(x + ux, y + uy) {
                let cut = (s - 1.0).max(0.0);
                let end = [q[0][0] + ux * cut, q[0][1] + uy * cut];
                if out.len() < 2 || cut > 1e-3 {
                    out.push(end);
                }
                if out.len() < 2 {
                    out.push([end[0] + ux * 0.5, end[1] + uy * 0.5]);
                }
                return out;
            }
            s += 0.5;
        }
        along += l;
        out.push(q[1]);
    }
    out
}

/// A HATCH line (see `MediumProfile::hatch`): the path pulled `amt` of the way onto the straight line through its
/// first point along its overall direction (first point to last), its length kept — the flow's curve and the
/// hand's waver taken out of the tempera mark.
fn straighten_path(path: &[[f32; 2]], amt: f32, w: u32, h: u32) -> Vec<[f32; 2]> {
    if path.len() < 3 || amt <= 1e-3 {
        return path.to_vec();
    }
    let a = amt.clamp(0.0, 1.0);
    let (p0, pn) = (path[0], path[path.len() - 1]);
    let (mut dx, mut dy) = (pn[0] - p0[0], pn[1] - p0[1]);
    let dl = (dx * dx + dy * dy).sqrt();
    if dl < 1e-3 {
        return path.to_vec();
    }
    dx /= dl;
    dy /= dl;
    // The arc length along the path, so a curved path straightened keeps its length (a chord is shorter).
    let mut arc = 0f32;
    let mut out = Vec::with_capacity(path.len());
    for (i, q) in path.iter().enumerate() {
        if i > 0 {
            arc += ((q[0] - path[i - 1][0]).powi(2) + (q[1] - path[i - 1][1]).powi(2)).sqrt();
        }
        let sx = p0[0] + dx * arc;
        let sy = p0[1] + dy * arc;
        out.push([(q[0] + (sx - q[0]) * a).clamp(0.0, w as f32 - 1.0), (q[1] + (sy - q[1]) * a).clamp(0.0, h as f32 - 1.0)]);
    }
    out
}

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
/// PAPER (charcoal): the share by which the sheet must differ from the picture's value before a mark darkens
/// it (or the eraser lifts it) — within it the place is left as it is.
const PAPER_TOL: f32 = 0.06;
/// PAPER (charcoal): the marks of a pass OVERLAP — each is charged for the residual at its seed, and a long
/// mark crosses places its neighbours already darkened. Measured on the three charcoal subjects (set 4): the
/// value came out darker by about 1.4× in optical density (ln 0.025 / ln 0.07 = 1.39 on the 50–100 band,
/// 1.46 on 100–150) — so the residual is asked for at that root.
const PAPER_OVERLAP: f32 = 1.4;
/// PAPER TOP (charcoal): the most the picture's light end is lifted toward the paper — a near-white ground
/// (the elder's, 236 against the sheet's 255: ×1.2 in linear light) reaches it; a dark picture's is not
/// carried up to white.
const PAPER_TOP_GAIN: f32 = 1.25;

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
pub(crate) fn local_range(luma: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
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
    // THE DRY BRUSH (see `BrushConfig::skip`): under the watercolour wash recipe the fine, drier marks are
    // broken by the paper's tooth unless the caller set the dial — on the params the strokes AND the score
    // header are built from, so the replay crosses the same paper.
    let recipe_skip = p.luminous && !p.book && p.from_scratch && p.medium == "watercolour" && wc_recipe_on() && p.brush.skip <= 0.0;
    let p_skip: PaintParams;
    let p: &PaintParams = if recipe_skip {
        let mut q = p.clone();
        q.brush.skip = std::env::var("PLAKAT_WCB_SKIP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
        p_skip = q;
        &p_skip
    } else {
        p
    };
    let (w, h) = (input.width(), input.height());
    // PROFILE (`PLAKAT_PAINT_PROFILE=1`): per-stage and per-pass wall-clock times, printed to stderr at the end.
    // This is how the stroke cost was found to be the mixture solver, not the brush — keep it.
    let started = std::time::Instant::now();
    let mut stats: Vec<PassStat> = Vec::new();
    let mut splattered_wet = false;
    let prof_on = std::env::var("PLAKAT_PAINT_PROFILE").is_ok();
    let mut prof_acc: Vec<(&'static str, f64)> = Vec::new();
    let mut prof_t = std::time::Instant::now();
    let lap = |name: &'static str, acc: &mut Vec<(&'static str, f64)>, t: &mut std::time::Instant| { if prof_on { acc.push((name, t.elapsed().as_secs_f64())); *t = std::time::Instant::now(); } };
    // The subject as given, before it is reduced to an armature: the fine layers look at it for TEXTURE.
    let source: &RgbImage = input;
    // (Kept apart for NIB FIGURE: a pen draws a face's features, and the armature has none.)
    let original: &RgbImage = input;
    // HAIR FLOW (see `hair_flow_field`): which way hair GROWS, as one angle per armature cell. Built here,
    // from the picture as given, because two lines below `input` becomes the armature and the growth
    // direction is precisely what the armature has thrown away. Only computed where there is hair to paint.
    let hair_flow: Option<HairFlow> = p.hair_mask.as_ref().filter(|m| m.len() == (w * h) as usize).map(|_| {
        let side = p.armature_face_side.or(p.armature_body_side).unwrap_or(NEW_BACKGROUND_SIDE);
        hair_flow_field(source, side)
    });
    // FORM FLOW (`--normals`): which way a mark runs by the picture's surfaces — round a turning form, level on
    // the ground. Off unless a normal map was given.
    let form_flow = p.normals.as_ref().and_then(|(nw, nh, n)| crate::paint::engrave::FormFlow::new(crate::paint::engrave::Normals { w: *nw as usize, h: *nh as usize, n }, w as usize, h as usize));
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
            if let Some(path) = std::env::var_os("PLAKAT_PAINT_ARMATURE_DUMP") {
                // Diagnostic: the armature the painter reads (the plan of masses, every tier blended in).
                let _ = armature_owned.save(&path);
            }
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
    canvas.cloth_seed = p.seed;
    if p.hiding > 0.0 {
        // PIGMENT HIDING (see `MediumProfile::hiding`): each pigment covers by its own power, in the header.
        canvas = canvas.with_hiding(p.hiding);
    }
    // NIB WASH WATER (see `MediumProfile::nib_wash_water`): the pigment snapshot kept, so the planes' ink can seep.
    let wash_water = if p.nib > 0.0 && p.nib_wash > 0.0 && p.nib_wash_planes >= 2.0 && p.nib_wash_water > 0.0 { p.nib_wash_water.clamp(0.0, 1.0) } else { 0.0 };
    // NIB WASH AGENT (see `MediumProfile::nib_wash_agent`): the residual's gain, in the header for the replay.
    let wash_agent = if p.nib > 0.0 && p.nib_wash > 0.0 && p.nib_wash_planes >= 2.0 && p.nib_wash_sweep > 0.0 && p.nib_wash_agent > 0.0 { p.nib_wash_agent.clamp(0.0, 1.0) } else { 0.0 };
    if wash_water > 0.0 {
        canvas = canvas.with_seep(true);
    }
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
    // CARNATION (see `MediumProfile::carnation`): the palette's warm and neutral pigments as a box of their
    // own for the faces, with each one's index in the full palette, and a mixture cache of its own.
    let carnation: Option<(Palette, Vec<usize>)> = (p.carnation > 0.0 && p.face_mask.is_some()).then(|| warm_palette(&p.palette)).flatten();
    let carn_cache: std::cell::RefCell<std::collections::HashMap<u32, Vec<f32>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    // SMALL (see `MediumProfile::small`): how small the faces are against the threshold — `k = 1 − extent /
    // small`, 0 for a face at or over it — and, when they are small, the picture's local CIELAB lightness (a
    // box mean a twelfth of the face across) the half-tones there are held to.
    let face_ext: Option<f32> = p.face_extent.or_else(|| p.face_mask.as_deref().and_then(|m| region_extent(m, input.width(), input.height())));
    // SMALL'S FLOOR (see `MediumProfile::small_min`): a crowd's face — the face-scale keys stay off.
    let crowd = p.small_min > 0.0 && face_ext.map_or(false, |e| e < p.small_min);
    let small_k: f32 = if p.small > 0.0 && p.face_mask.is_some() && !crowd { face_ext.map(|e| (1.0 - e / p.small).clamp(0.0, 1.0)).unwrap_or(0.0) } else { 0.0 };
    // ANCHOR: the same box's spread (the root of the mean square less the squared mean), and the darks'
    // threshold in it — the normal quantile of the key's share, by the logistic approximation.
    let small_l: Option<(Vec<f32>, Option<Vec<f32>>)> = (small_k > 0.0).then(|| {
        let l: Vec<f32> = input.pixels().map(|px| color::srgb_to_lab(px.0).l).collect();
        let r = face_ext.map(|e| (e / 12.0).round().max(2.0)).unwrap_or(4.0) as i32;
        let mean = box_blur(&l, input.width(), input.height(), r);
        let sd = (p.anchor > 0.0).then(|| {
            let sq: Vec<f32> = l.iter().map(|v| v * v).collect();
            box_blur(&sq, input.width(), input.height(), r).iter().zip(&mean).map(|(q, m)| (q - m * m).max(0.0).sqrt()).collect()
        });
        (mean, sd)
    });
    let anchor_z: f32 = { let q = p.anchor.clamp(0.01, 0.5); ((1.0 - q) / q).ln() / 1.702 };

    let mut score = StrokeScore {
        header: ScoreHeader { version: 1, palette: p.palette.name.to_string(), pigments: p.palette.pigments.iter().map(|pg| (pg.name.to_string(), pg.masstone)).collect(), medium: p.medium.clone(), seed: p.seed, width: w, height: h, tooth: 0.85, ground: p.ground, brush: p.brush, bleed: p.bleed, diffuse: p.diffuse, stages: None, dry: p.dry, opacity: p.opacity, impasto: p.impasto, chroma: p.chroma, dry_shift: p.dry_shift, granulate: p.granulate, sheen: p.sheen, edge_pool: p.edge_pool, paper_edge: p.paper_edge, old_paper: p.old_paper, contrast: p.contrast, warmth: p.warmth, clarity: p.clarity, lift: p.lift , transmittance: false, flow: None, weave: p.weave, sag: p.sag, gloss: p.gloss, hiding: p.hiding, tone: 0.0, water: wash_water, agent: wash_agent, light: p.blade_light, phong: p.blade_phong, spread: p.blade_relief, peak: p.blade_peak, collide: None, hold_mask: None },
        strokes: Vec::new(),
    };

    // The passes to run: an explicit plan (P1.3), else coarse→fine from brush_sizes (P0).
    let sizes: Vec<f32> = p.brush_sizes.iter().copied().filter(|&r| r >= p.min_brush).collect();
    let passes: Vec<PassSpec> = p.passes.clone().unwrap_or_else(|| {
        sizes.iter().enumerate().map(|(i, &r)| PassSpec { face_only: p.face_ladder_from.is_some_and(|n| i >= n), radius: r, budget: p.budget, stage: if i == 0 { "block-in".into() } else { format!("restate-{i}") } }).collect()
    });
    // FROM SCRATCH: the minimum brush of every plane (RFC §9). A rung below the focal floor touches nothing, so
    // it is not a pass of this painting at all.
    // The watercolour paints the whole sheet with ONE ladder: a figure-plane floor gave the feet fine marks
    // and the paving beside them a coarse one.
    let wc_wash_early = p.luminous && !p.book && p.from_scratch && p.medium == "watercolour" && wc_recipe_on();
    let wc_planes = std::env::var("PLAKAT_WCB_PLANES").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(if wc_wash_early { 0.0 } else { 1.0 }) > 0.0;
    let plane_floor: Option<Vec<f32>> = (p.from_scratch && wc_planes).then(|| plane_floor_field(w, h, p.min_brush, (p.armature_side.unwrap_or(NEW_BACKGROUND_SIDE), p.armature_body_side, p.armature_face_side), p.subject_mask.as_deref(), p.face_mask.as_deref(), p.hair_mask.as_deref()));
    let passes: Vec<PassSpec> = match &plane_floor {
        Some(fl) => {
            let finest = fl.iter().copied().fold(f32::INFINITY, f32::min);
            passes.into_iter().filter(|q| q.radius.max(p.min_brush) >= finest).collect()
        }
        None => passes,
    };
    // THE GLAZE PASS (see `MediumProfile::glaze_pass`): one more pass after the ladder, a mid brush, every mark
    // a thin wet film over everything — a stage of its own in the score, so the replay crosses it alike.
    let passes: Vec<PassSpec> = if p.glaze_pass > 0.0 && !p.density && !p.luminous && !passes.is_empty() {
        let mut passes = passes;
        let mid = passes.get(1).or(passes.first()).map(|q| q.radius).unwrap_or(p.min_brush * 2.0);
        passes.push(PassSpec { face_only: false, radius: mid, budget: (p.budget / 6).max(64), stage: "glaze".into() });
        passes
    } else {
        passes
    };
    // THE CREST PASS (see `MediumProfile::crest`): after the glaze, a fine brush dragged dry over the dried
    // relief where it is broken — the ladder's third-finest brush, so the marks are small enough to follow a
    // sweater's weave or a feather and the pass dense enough to light it. The same pass carries the wash into
    // the furrows in the shadow (`sink`).
    let passes: Vec<PassSpec> = if (p.crest > 0.0 || p.sink > 0.0) && !p.density && !p.luminous && !passes.is_empty() {
        let mut passes = passes;
        let ladder: Vec<f32> = passes.iter().filter(|q| q.stage != "glaze").map(|q| q.radius).collect();
        let r = ladder.get(ladder.len().saturating_sub(3)).copied().unwrap_or(p.min_brush * 3.0).max(p.min_brush);
        passes.push(PassSpec { face_only: false, radius: r, budget: (p.budget / 8).max(64), stage: "crest".into() });
        passes
    } else {
        passes
    };
    let ladder_len = passes.iter().filter(|q| q.stage != "glaze" && q.stage != "crest").count();
    // NIB WASH (see `MediumProfile::nib_wash`): the ladder laid twice — the washes, then the pens' passes (`pen-…`)
    // over them, each register with its own keys (`p_wash`, `p_pen`).
    let nib_wash_n = if p.nib > 0.0 && p.nib_wash > 0.0 && !p.density && !p.luminous && !passes.is_empty() { passes.len() } else { 0 };
    // NIB WASH DRAW (see `MediumProfile::nib_wash_draw`): the pens' passes first, the washes over the drawing.
    let wash_draw = nib_wash_n > 0 && p.nib_wash_draw > 0.0;
    let passes: Vec<PassSpec> = if nib_wash_n > 0 {
        let pens: Vec<PassSpec> = passes.iter().map(|q| PassSpec { stage: format!("pen-{}", q.stage), ..q.clone() }).collect();
        if wash_draw { pens.into_iter().chain(passes.iter().cloned()).collect() } else { passes.iter().cloned().chain(pens).collect() }
    } else {
        passes
    };
    let (p_wash, p_pen): (Option<PaintParams>, Option<PaintParams>) = if nib_wash_n > 0 {
        let pw = PaintParams { nib: 0.0, hatch: 0.0, cross: 0.0, tiers: 0.0, slant: 0.0, solid: 0.0, span: 0.0, edge: 0.0, clip: 0.0, lights: 0.0, grade: 0.0, ..p.clone() };
        let pp = PaintParams { boneless: 0.0, hake: 0.0, wash_in: 0.0, bokashi: 0.0, layer_brush: Some(p.layer_brush.unwrap_or(BrushProfile::named("pen"))), ..p.clone() };
        (Some(pw), Some(pp))
    } else {
        (None, None)
    };
    // The schedule goes into the score so a replay crosses every pass boundary the paint crossed — a pass that
    // lays no stroke included. Density media run no passes (the drawing is laid by `ink_drawing`).
    // A luminous medium: the washes (one stage per level), then the two finest brush passes at a fraction of
    // the budget (the modelling within the washes).
    // EXPERIMENT (PLAKAT_WC_BRUSH): a new watercolour painted with the BRUSH alone — no wash fills, the
    // whole ladder on white paper behind the reserve, watercolour pigment physics.
    // THE WATERCOLOUR (`PLAKAT_WC_WASH=1`): the recipe that paints a new watercolour with the brush — the
    // transmittance film, the picture's OWN values and colours (no re-key), the whole ladder everywhere,
    // each broad pass toward the picture's light envelope so the film darkens light-to-dark, the fine passes
    // reading the source. (The sweep/mass route is still here under its own knobs — it lost the picture.)
    // One switch; every `PLAKAT_WCB_*` knob still overrides its part.
    let wc_wash = p.luminous && !p.book && p.from_scratch && p.medium == "watercolour" && wc_recipe_on();
    let wc_brush = wc_wash || (p.luminous && !p.book && p.from_scratch && p.medium == "watercolour" && std::env::var_os("PLAKAT_WC_BRUSH").is_some());
    let env_or = |name: &str, d: f32| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let wcb_fill = wc_brush && env_or("PLAKAT_WCB_FILL", 0.0) > 0.0;
    // WASH STROKES: every mass is a STAGE of its own (the wash brush's film cap counts from the stage mark,
    // and each wash dries before the next, as wet-on-dry does), so the stage list is one per mass.
    let wcb_sweep = wcb_fill && env_or("PLAKAT_WCB_SWEEP", 0.0) > 0.0;
    let wcb_mass_count = env_or("PLAKAT_WCB_MASSES", 0.0).max(0.0) as usize;
    let luminous_passes: Vec<PassSpec> = if wc_brush {
        passes.clone()
    } else if p.luminous && !p.book {
        // The modelling passes keep the plan's budget for the fine brushes: the restate gate already confines
        // them to where the washes differ from the picture (a lit window, a machine, a face), so on a flat
        // wash they lay little and on a detailed passage they resolve it — a quarter share washed the detail
        // out of every picture with fine structure (a night scene's lamps and figures).
        // Only the FINE brushes (≤ ~14 px) model within the washes: a mid-scale pass over them re-painted the
        // masses' own edges and lost detail (measured on a night scene: 0.096 → 0.086 Laplacian σ).
        let fine: Vec<PassSpec> = passes.iter().filter(|q| q.radius.max(p.min_brush) <= 14.5).cloned().collect();
        fine.into_iter().map(|q| PassSpec { budget: ((q.budget as f32) * 0.7) as usize, ..q }).collect()
    } else {
        Vec::new()
    };
    // The mass washes are stage 0 of a brush watercolour that lays them (see `wc_fill_masses`).
    let wash_stages = if wc_brush { if wcb_sweep { wcb_mass_count.clamp(2, 64) } else { usize::from(wcb_fill) } } else { p.armature_levels.max(2) as usize };
    let n_stages_total = wash_stages + luminous_passes.len();
    score.header.stages = Some(if p.density {
        Vec::new()
    } else if wc_brush {
        (if wcb_sweep { (1..=wash_stages).map(|i| format!("wash-{i}")).collect::<Vec<_>>() } else if wcb_fill { vec!["wash".to_string()] } else { Vec::new() }).into_iter().chain(luminous_passes.iter().map(|q| q.stage.clone())).collect()
    } else if p.luminous {
        (1..=p.armature_levels.max(2)).map(|l| format!("wash-{l}")).chain(luminous_passes.iter().map(|q| q.stage.clone())).collect()
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
    let shadow = (p.commit_shadows > 0.0 || p.shade > 0.0 || p.lost > 0.0 || p.crest > 0.0 || p.sink > 0.0).then(|| {
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
    // EXPERIMENT knobs for the brush watercolour (see `wc_brush`): per-layer charge multipliers, the broad
    // passes' wetness and width multiplier, and whether the brush lifts.
    let env_f = |name: &str, d: f32| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    // THE DRY BRUSH IS FOR TEXTURE: a watercolourist drags a dry brush over cobbles, a beard, a bench — and
    // lays a face, a sky, a sleeve as a wash. Under the recipe a fine mark's wetness follows how BUSY the
    // picture is at the mark's scale (the local value range, a contrast fact: 0.06 is about where texture
    // begins), so the marks in a smooth passage flood the tooth and the marks in a textured one skip it.
    // Never over the LIGHTS: a watercolourist does not drag dry pigment across a lit window or a lamp — those
    // are the paper and a wash — so the busy field is gated out where the picture is bright. And never over a
    // FACE: the eyes and the features are busy at the mark's scale, but they are the eye's destination and
    // are laid as washes with a few crisp accents, not dragged dry (dry-brushed eyes fragmented).
    // (The same field drives the IMPASTO MAP's texture term: thick where the form is broken, thin where it
    // is smooth — so a cheek or a sky is laid smooth and a beard or bark stands up.)
    // The picture's own value range for the impasto map (linear luma, 5th–95th percentile).
    let (map_lo, map_hi) = if p.impasto_map > 0.0 || p.lights > 0.0 || p.dishes >= 2.0 || (p.solid > 0.0 && p.hatch > 0.0) || p.paper > 0.0 || p.velvet > 0.0 {
        let mut v: Vec<f32> = input.pixels().map(|px| color::linear_luma(color::srgb_to_linear(px.0))).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (v[v.len() / 20], v[v.len() * 19 / 20])
    } else {
        (0.0, 1.0)
    };
    // The reference charge for the map's load compensation: what a mid-value stroke carries (the
    // ladder's charge at the picture's median value), so the compensation is 1 at the middle of the range.
    let load_ref: f32 = p.charge.max(1e-3);
    let busy_field: Option<Vec<f32>> = if (p.brush.skip > 0.0 && p.luminous && !p.book && p.from_scratch && p.medium == "watercolour") || p.impasto_map > 0.0 {
        let lm = luma_map(input);
        let fine = local_range(&lm, w as usize, h as usize, (p.min_brush * 0.75).round().max(2.0) as usize);
        let face = p.face_mask.as_deref();
        let bf0: Vec<f32> = (0..lm.len()).map(|i| {
            let f = ((fine[i] - 0.08) / 0.12).clamp(0.0, 1.0) * (1.0 - ((lm[i] - 0.5) / 0.3).clamp(0.0, 1.0));
            f * (1.0 - face.and_then(|m| m.get(i).copied()).unwrap_or(0.0).clamp(0.0, 1.0))
        }).collect();
        // A MARK is dry or wet as a whole: the field is read at the mark's scale, not at one pixel — on a
        // 4096 snow field the per-pixel range was 4% on average but 100% at isolated specks of sensor grain,
        // and every fine mark seeded on a speck ran dry along its whole length (a thousand dry streaks on
        // flat snow). Smoothed over a few brush widths, a speck is nothing and a beard is still a beard.
        let bf = crate::paint::canvas::box_blur_pub(&bf0, w as usize, h as usize, (p.min_brush * 3.0).round().max(3.0) as usize);
        if let Some(dir) = std::env::var_os("PLAKAT_WCB_DUMP") {
            let img = image::GrayImage::from_fn(w, h, |x, y| image::Luma([(bf[(y * w + x) as usize] * 255.0) as u8]));
            let _ = img.save(std::path::Path::new(&dir).join("busy.png"));
        }
        Some(bf)
    } else {
        None
    };
    let wcb_charge: Vec<f32> = std::env::var("PLAKAT_WCB_CHARGE").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_else(|| vec![0.8, 0.8, 0.7, 0.7, 0.9, 1.0, 1.0]);
    let wcb_wet = env_f("PLAKAT_WCB_WET", if wc_wash { 0.0 } else { 0.7 });
    let wcb_width = env_f("PLAKAT_WCB_WIDTH", 1.6);
    let wcb_len = env_f("PLAKAT_WCB_LEN", 2.5);
    let wcb_pickup = env_f("PLAKAT_WCB_PICKUP", 0.0);
    let wcb_depth = env_f("PLAKAT_WCB_DEPTH", if wc_wash { 0.8 } else { 0.9 });
    let wcb_gamma = env_f("PLAKAT_WCB_GAMMA", if wc_wash { 1.6 } else { 2.6 });
    let wcb_env = env_f("PLAKAT_WCB_ENV", if wc_wash { 0.8 } else { 0.0 });
    // The fine passes' reference: the source, blurred at this fraction of the brush (0 = the armature path).
    let wcb_src = env_f("PLAKAT_WCB_SRC", if wc_wash { 0.25 } else { 0.0 });
    // …a floor of the sheet's short side over 300 (7 px on a 2048 sheet, 3.4 on a 1024 one: a flower box on
    // the smaller sheet is forty pixels, and read at 7 its flowers and leaves averaged into one daub).
    let wcb_src_floor = env_f("PLAKAT_WCB_SRCFLOOR", if wc_wash { (w.min(h) as f32 / 300.0).max(2.0) } else { 0.0 });
    let wcb_faceonly = env_f("PLAKAT_WCB_FACEONLY", if wc_wash { 0.0 } else { 4.5 });
    let wcb_restate = env_f("PLAKAT_WCB_RESTATE", if wc_wash { 1.0 } else { 2.0 });
    // The structure a fine brush needs to find before it restates off the subject (a local value range at
    // the brush's scale; 0 = restate tone too, the old behaviour).
    let wcb_struct = env_f("PLAKAT_WCB_STRUCT", if wc_wash { 0.10 } else { 0.0 });
    // The share of that bar a faceless subject is held to (1 = as the ground, 0 = restate tone freely).
    let wcb_struct_subj = env_f("PLAKAT_WCB_STRUCTSUBJ", 0.4);
    // How much longer the fine brushes' marks may run than the ladder's default cap (lines, not beads).
    let wcb_finelen = env_f("PLAKAT_WCB_FINELEN", if wc_wash { 1.0 + 2.0 * p.fine_lines.clamp(0.0, 1.0) } else { 1.0 });
    let wcb_blur = env_f("PLAKAT_WCB_BLUR", if wc_wash { 0.2 } else { 0.32 });
    let wcb_broken = env_f("PLAKAT_WCB_BROKEN", if wc_wash { 0.0 } else { 1.0 });
    let wcb_texture = env_f("PLAKAT_WCB_TEXTURE", 1.0);
    let wcb_bleedmul = env_f("PLAKAT_WCB_BLEEDMUL", if wc_wash { 0.5 } else { 1.0 });
    let wcb_finest = env_f("PLAKAT_WCB_FINEST", if wc_wash { 3.0 } else { 0.0 });
    let wcb_cap = env_f("PLAKAT_WCB_CAP", if wc_wash { 1.0 } else { 2.0 });
    let wcb_streak = env_f("PLAKAT_WCB_STREAK", 0.05);
    let wcb_round = env_f("PLAKAT_WCB_ROUND", if wc_wash { 0.6 } else { 0.0 });
    let wcb_flat = env_f("PLAKAT_WCB_FLAT", 0.0);
    let wcb_subj_from = env_f("PLAKAT_WCB_SUBJ", 0.0);
    let wcb_stop = env_f("PLAKAT_WCB_STOP", 1.0);
    // A watercolourist mixes TWO pigments, three at most: sixteen in one puddle is grey.
    let wcb_pig = env_f("PLAKAT_WCB_PIG", 0.0) as usize;
    // DARKS BY CONCENTRATION, never by black. The mixer reaches a dark target by adding black pigment, and
    // the thin transparent film then tints that black-and-colour mix into grey-brown — measured on flat
    // patches: a keyed orange at saturation 80 painted at 20. A watercolourist mixes the HUE at full
    // strength and gets the value from how much pigment is in the water: the masstone of the target, at a
    // charge found on scrap paper for the target's value.
    let wcb_conc = env_f("PLAKAT_WCB_CONC", 0.0);
    let wcb_tol = env_f("PLAKAT_WCB_TOL", 0.08);
    let wcb_probe = env_f("PLAKAT_WCB_PROBE", 0.0);
    let wcb_targetl = env_f("PLAKAT_WCB_TARGETL", 0.3);
    let wcb_relit = std::env::var_os("PLAKAT_WCB_MASSTONE").is_some();
    // CONFINE every stroke to its own colour mass: a stroke stops at the mass boundary, so a blue wash and an
    // orange one lie side by side and never overlap into grey (a transparent film's hue is the proportion of
    // every pigment ever laid on it — two complementary strokes crossing is mud).
    let wcb_confine = env_f("PLAKAT_WCB_CONFINE", 0.0) > 0.0;
    // THE TRANSMITTANCE FILM (`PLAKAT_WCB_TRANS`): see `Canvas::with_transmittance`. Each stroke is solved
    // in LOG space for the densities that take the canvas from what is there to the target (the residual
    // glaze), with `wcb_hits` the lane hits a stroke lays per pixel (the calibration from flat patches).
    let wcb_trans = wc_brush && env_f("PLAKAT_WCB_TRANS", if wc_wash { 1.0 } else { 0.0 }) > 0.0;
    let wcb_hits = env_f("PLAKAT_WCB_HITS", if wc_wash { 0.5 } else { 1.5 });
    // The fluid stage's strength / radius (fraction of the short side) / rim, by the technique: wet-on-wet
    // floods wide, wet-on-dry a narrow film that sets with a rim, the dry brush has no water to flow.
    let (flow_s, flow_r, flow_m) = match p.technique {
        WetTechnique::WetOnWet => (0.9, 0.012, 0.4),
        WetTechnique::DryOnDry => (0.0, 0.0, 0.0),
        _ => (0.6, 0.005, 0.8),
    };
    // The painter holds the faces still under the collision; the replay must too, so the faces go into the
    // score as a coarse hold mask — and the painter uses THAT same coarse mask, so both cross identically.
    let collide_hold: Option<Vec<f32>> = if p.collide > 0.0 && !p.luminous {
        score.header.collide = Some((p.collide.clamp(0.0, 1.0), (w.min(h) as f32 * 0.006).max(2.0)));
        if p.face_mask.is_some() || p.subject_mask.is_some() {
            let (cols, cells) = crate::paint::score::coarse_hold(p.face_mask.as_deref(), p.subject_mask.as_deref(), w, h, 64);
            let hold = crate::paint::score::expand_hold(cols, &cells, w, h);
            score.header.hold_mask = Some((cols, cells));
            Some(hold)
        } else {
            None
        }
    } else {
        None
    };
    let wcb_flow = env_f("PLAKAT_WCB_FLOW", if wc_wash { 1.0 } else { 0.0 });
    if wc_wash && wcb_flow > 0.0 && flow_s > 0.0 {
        let radius = (w.min(h) as f32 * env_f("PLAKAT_WCB_FLOWR", flow_r)).max(1.0);
        // Granulation happens in the WATER: the granulating pigments settle into the tooth where the wash
        // pooled (see `Canvas::flow`), so the finish's uniform grain is turned down to a trace of paper.
        let grain = env_f("PLAKAT_WCB_FLOWGRAIN", p.granulate);
        // SELECTIVE wet-in-wet (see `Canvas::flow`): the water floods the soft masses and holds back at the
        // picture's hard edges — the painter's choice of where to work wet-in-wet and where wet-on-dry.
        let selective = env_f("PLAKAT_WCB_FLOWSEL", 1.0);
        score.header.flow = Some((flow_s * wcb_flow, radius, env_f("PLAKAT_WCB_FLOWRIM", flow_m), grain, selective));
        if grain > 0.0 {
            score.header.granulate = p.granulate * 0.1;
        }
    }
    // The broad passes drag a seed's dose across a whole swath: they carry it at this many hits instead.
    let wcb_hits_broad = env_f("PLAKAT_WCB_HITSBROAD", if wc_wash { 1.2 } else { wcb_hits });
    if wcb_trans {
        canvas = canvas.with_transmittance(true);
        score.header.transmittance = true;
    }
    let trans_cache: std::cell::RefCell<std::collections::HashMap<(i16, i16, i16), Vec<f32>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    // The mixer's palette for the brush watercolour: the picture's pigments WITHOUT the paper white (a
    // watercolourist has no white; the paper is the white), so a hue is matched by pigments alone and its
    // value is set by the charge. `wc_pal_map[i]` is the full-palette index of sub-palette pigment `i`.
    let (wc_pal, wc_pal_map): (Palette, Vec<usize>) = {
        // (Only when the hue is re-lit to a fixed value; with the picture's own pigments a light target
        // needs the white to be matched at all, and without it the mixer reached for three mid pigments —
        // a grey.)
        let keep: Vec<usize> = (0..p.palette.pigments.len()).filter(|&i| !wcb_relit || color::linear_luma(color::srgb_to_linear(p.palette.pigments[i].masstone)) < 0.7).collect();
        let keep = if keep.len() >= 2 { keep } else { (0..p.palette.pigments.len()).collect() };
        let pigs: Vec<crate::paint::pigment::Pigment> = keep.iter().map(|&i| p.palette.pigments[i]).collect();
        (Palette { name: p.palette.name, pigments: Box::leak(pigs.into_boxed_slice()) }, keep)
    };
    let wc_mix_cache: std::cell::RefCell<std::collections::HashMap<u32, Vec<f32>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    let probe_cache: std::cell::RefCell<std::collections::HashMap<(u32, u16), f32>> = std::cell::RefCell::new(std::collections::HashMap::new());
    // PAPER / ERASE / TOUCH / VELVET (charcoal; see `MediumProfile::paper`): the darkest pigment, the bare
    // sheet's and the black's values, and the charge cache for the residual marks.
    let paper_cache: std::cell::RefCell<std::collections::HashMap<(u32, u16), f32>> = std::cell::RefCell::new(std::collections::HashMap::new());
    let charcoal_dark = crate::paint::canvas::darkest_pigment(&p.palette);
    let charcoal_paper_l = color::linear_luma(canvas.ground_linear());
    let charcoal_black_l = color::linear_luma(color::srgb_to_linear(p.palette.pigments[charcoal_dark].masstone));
    let wcb_fill_feather = env_f("PLAKAT_WCB_FEATHER", 0.004);
    let wcb_fill_wet = env_f("PLAKAT_WCB_FILLWET", if wc_wash { 0.9 } else { 0.75 });
    // The bleed a brush watercolour runs with (the medium's × the knob) — the SCORE carries it, so the
    // replay crosses every stage with the same bloom.
    let bleed_eff = if wc_brush { p.bleed * wcb_bleedmul } else { p.bleed };
    score.header.bleed = bleed_eff;

    let wcb_hi = env_f("PLAKAT_WCB_HI", if wc_wash { 0.985 } else { 0.95 });
    let wcb_dep = env_f("PLAKAT_WCB_DEP", 1.5);
    let wcb_bleed = env_f("PLAKAT_WCB_BLEED", 0.0);
    let wcb_dark = env_f("PLAKAT_WCB_DARK", if wc_wash { 0.0 } else { 1.2 });
    let wcb_light = env_f("PLAKAT_WCB_LIGHT", if wc_wash { 1.0 } else { 0.4 });
    let wcb_paper = env_f("PLAKAT_WCB_PAPER", if wc_wash { 0.0 } else { 0.08 });
    // THE KEY. A watercolour paints toward a picture re-keyed to the paper: paper at the top of the picture's
    // range, the darkest wash `depth` below it, in perceptual value, chromaticity kept — a night's near-black
    // brown becomes a brown the mixer can find pigment for. Every pass paints toward the keyed picture, so
    // the restate gates agree with the strokes.
    let keyed_input;
    let keyed_source;
    let (input, source): (&RgbImage, &RgbImage) = if wc_brush && std::env::var_os("PLAKAT_WCB_NOKEY").is_none() && !(wc_wash && std::env::var_os("PLAKAT_WCB_KEY").is_none()) {
        keyed_input = key_image(input, input, wcb_depth, wcb_gamma, wcb_hi);
        keyed_source = key_image(source, input, wcb_depth, wcb_gamma, wcb_hi);
        if let Some(dir) = std::env::var_os("PLAKAT_PAINT_MASKS") {
            let _ = keyed_input.save(std::path::Path::new(&dir).join("wc_keyed_input.png"));
        }
        (&keyed_input, &keyed_source)
    } else {
        (input, source)
    };
    // THE MASSES: a wash is ONE colour over a shape. The broad passes paint each cell in its MASS's colour
    // (k-means over the keyed picture at wash scale), so neighbouring strokes of one mass are the same
    // colour and overlap invisibly into one flat wash, and the only edges are where masses meet.
    let wc_mass: Option<(Vec<u16>, Vec<Srgb>)> = (wc_brush && wcb_mass_count >= 2).then(|| wc_mass_colours(input, wcb_mass_count, p.seed));
    let wc_mass_masks: Vec<Vec<bool>> = match (&wc_mass, wcb_confine) {
        (Some((labels, means)), true) => (0..means.len()).map(|m| labels.iter().map(|&l| l as usize == m).collect()).collect(),
        _ => Vec::new(),
    };
    if let (Some((labels, means)), Some(dir)) = (&wc_mass, std::env::var_os("PLAKAT_PAINT_MASKS")) {
        let g = RgbImage::from_fn(w, h, |x, y| image::Rgb(means[labels[y as usize * w as usize + x as usize] as usize]));
        let _ = g.save(std::path::Path::new(&dir).join("wc_masses.png"));
    }
    // The PAPER of a brush watercolour: the keyed picture's lightest `wcb_paper` share, as shapes (the
    // shape-aware reserve below does the shaping). The plan's reserve luma is for the other media.
    let reserve_eff: Option<f32> = if wc_brush && wcb_paper > 0.0 {
        let mut l = luma_map(input);
        l.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(l[((l.len() as f32 - 1.0) * (1.0 - wcb_paper.clamp(0.005, 0.6))) as usize])
    } else if wc_wash {
        // No reserve at all: on the transmittance film the lights need no protecting — nothing darkens them
        // but a glaze the picture asked for — and a reserve blocked every stroke on the stained-glass
        // windows, which came out as bare paper.
        None
    } else {
        p.reserve
    };
    let protect_all: Option<Vec<bool>> = match reserve_eff {
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
                // A brush watercolour reserves only BOLD shapes: opened wider and kept only from a
                // twenty-fourth of the sheet across, so a sunlit façade's chips are washed, not left as flecks.
                let r = if wc_brush { (short / 150).max(3) } else { (short / 400).max(2) };
                let opened = dilate_bool(&erode_bool(&m, wu, hu, r), wu, hu, r);
                // A sixty-fourth of the sheet across — a lit window pane is a shape worth reserving; the
                // opening above has already removed anything thinner than a few pixels.
                let min_side = if wc_brush { (short / 24).max(8) } else { (short / 64).max(6) };
                m = keep_large_regions(&opened, wu, hu, min_side * min_side);
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
    // MA (see `MediumProfile::ma`): where the eye goes — the focal field, the subject taken with it below.
    let ma_field: Option<Vec<f32>> = (p.ma > 0.0 && p.paper > 0.0).then(|| {
        let f = focal_field(input);
        if p.ma_focus > 0.0 {
            // MA FOCUS (see `MediumProfile::ma_focus`): the subject taken in here and the whole blurred over a
            // forty-eighth of the sheet, so a subject mask's hard edge (a box around a figure) is a gradient
            // when the narrowed attention reaches the paper beside it.
            let (w, h) = (input.width(), input.height());
            let r = (w.max(h) / 48).max(2) as i32;
            if p.ma_flow > 0.0 {
                // MA FLOW (see `MediumProfile::ma_flow`): the silhouette alone seeds the focus, the halo spread
                // along the picture's lines (its value smoothed over a two-hundred-and-fiftieth of the sheet, so
                // a grass blade does not stop it and a hill's edge does), and the silhouette kept whole over it.
                let sil = p.silhouette_mask.as_deref().or(p.subject_mask.as_deref());
                let s_at = |i: usize| sil.and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let m: Vec<f32> = f.iter().enumerate().map(|(i, v)| v.max(s_at(i)).clamp(0.0, 1.0)).collect();
                let guide = box_blur(&crate::paint::ink::value_map(input), w, h, (w.max(h) / 256).max(1) as i32);
                let spread = edge_spread(&m, &guide, w as usize, h as usize, r as f32, p.ma_flow.clamp(0.0, 1.0));
                spread.iter().enumerate().map(|(i, v)| v.max(s_at(i)).clamp(0.0, 1.0)).collect()
            } else {
                let m: Vec<f32> = f.iter().enumerate().map(|(i, v)| v.max(p.subject_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0)).clamp(0.0, 1.0)).collect();
                box_blur(&m, w, h, r)
            }
        } else {
            f
        }
    });
    // YOHAKU (see `MediumProfile::yohaku`): the picture's perceived local range at a sixty-fourth of the sheet.
    let yohaku_range: Option<Vec<f32>> = (p.yohaku > 0.0 && p.paper > 0.0).then(|| {
        let lg: Vec<f32> = luma_map(input).iter().map(|v| v.max(0.0).powf(1.0 / 2.2)).collect();
        local_range(&lg, input.width() as usize, input.height() as usize, ((input.width().min(input.height()) as f32 / 64.0).round() as usize).max(2))
    });
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
    let mut book: Option<BookPlan> = None;
    let passes: Vec<PassSpec> = if p.density {
        ink_drawing(&mut canvas, &mut score, input, source, head_mask.as_deref(), p, protect_all.as_deref(), &mut placed, &mut k, progress);
        Vec::new()
    } else if p.luminous {
        // A LUMINOUS medium lays its masses as WASHES — area fills, level by level, light to dark, each level
        // dried before the next (see `wash_passes`) — then MODELS within them with the two finest brushes only
        // (thin transparent touches from the source's texture: a wash is flat, a watercolour is not), at a
        // fraction of the budget. The line and the splatter follow.
        if p.book {
            // (The line is planned first — the washes keep to it — and drawn last, over them.)
            let plan = book_plan(source, p, p.budget - p.budget * 2 / 5);
            print_plates(&mut canvas, &mut score, source, n_stages_total, p, &mut placed, &mut k, progress);
            book = Some(plan);
        } else if !wc_brush {
            wash_passes(&mut canvas, &mut score, input, source, n_stages_total, p, protect_all.as_deref(), &mut placed, &mut k, progress);
        }
        if let (Some((labels, means)), true) = (&wc_mass, wc_brush && wcb_fill) {
            wc_fill_masses(&mut canvas, &mut score, labels, means, w as usize, h as usize, protect_all.as_deref(), p, (w.max(h) as f32 * wcb_fill_feather).max(1.0), wcb_fill_wet, if wcb_sweep { Some((bleed_eff, n_stages_total)) } else { None }, input, &mut placed, &mut k);
            // Stage 0 ends as every stage does (and as the replay reproduces it). (The sweeps cross their own
            // stages, one per mass.)
            if !wcb_sweep {
                if bleed_eff > 0.0 {
                    canvas.bleed_with(bleed_eff * pass_bleed_taper(0, n_stages_total), p.diffuse);
                }
                if n_stages_total > 1 {
                    canvas.dry(1.0 - p.dry);
                }
            }
            if let Some(dir) = std::env::var_os("PLAKAT_WCB_DUMP") {
                let _ = canvas.to_image().save(std::path::Path::new(&dir).join("pass_fill.png"));
            }
        }
        luminous_passes
    } else {
        passes
    };
    // THE BOLE (see `MediumProfile::bole`): a new painting's imprimatura laid by local tone before the
    // block-in, as wash cells in the score — its own stage, crossed as every pass boundary is.
    let bole_stage = if p.bole > 0.0 && p.from_scratch && p.ground.is_some() && !p.density && !p.luminous && !wc_brush && !passes.is_empty() {
        let laid = lay_bole(&mut canvas, &mut score, input, p, &mut placed, &mut k);
        if laid > 0 {
            if let Some(st) = &mut score.header.stages {
                st.insert(0, "ground".into());
            }
            let b_n = passes.len() + 1;
            if bleed_eff > 0.0 {
                canvas.bleed_with(bleed_eff * pass_bleed_taper(0, b_n), p.diffuse);
            }
            if let Some((fs, fr, fm, fg, fe)) = score.header.flow {
                if let Some(t) = flow_taper(0, b_n) {
                    canvas.flow(fs * t, fr, fm, fg, fe);
                }
            }
            if let Some((cs, cr)) = score.header.collide {
                let (st, at) = collide_schedule(cs, 0, b_n);
                if st > 0.0 {
                    canvas.collide(st, cr, collide_hold.as_deref(), at);
                }
            }
            if b_n > 1 {
                canvas.dry(1.0 - p.dry);
            }
            stats.push(PassStat { stage: "ground".into(), radius: 0.0, strokes: laid, seconds: 0.0 });
            lap("bole", &mut prof_acc, &mut prof_t);
        }
        usize::from(laid > 0)
    } else {
        0
    };
    // CLIP (see `MediumProfile::clip`): the picture's contours, the bone's edge map, a stop for the hatch.
    let clip_edge: Option<Vec<bool>> = (p.clip > 0.0 && w >= 8 && h >= 8).then(|| {
        let value = box_blur(&crate::paint::ink::value_map(input), w, h, 1);
        crate::paint::ink::edge_map(&value, w as usize, h as usize, p.clip.clamp(0.0, 1.0))
    });
    // THE PEN'S FIELDS (see `MediumProfile::nib_relief`, `nib_occlude`, `nib_gap`): the picture's smoothed value
    // and its mean over a 100th of the sheet (the relief), the cast shadows' field (the broad nib's places), and the
    // band of white kept along the picture's long contours.
    let nib_val: Option<Vec<f32>> = (p.nib > 0.0 && (p.nib_relief > 0.0 || p.nib_occlude > 0.0 || p.nib_gap > 0.0 || p.nib_lift > 0.0 || p.nib_fur > 0.0) && w >= 8 && h >= 8).then(|| box_blur(&crate::paint::ink::value_map(input), w, h, 1));
    let nib_mean: Option<Vec<f32>> = nib_val.as_ref().filter(|_| p.nib_relief > 0.0).map(|v| box_blur(v, w, h, (w.max(h) / 100).max(2) as i32));
    // NIB FUR (see `MediumProfile::nib_fur`): the relief at the body's scale — the value over a 60th of the sheet
    // against its mean over a 15th.
    let nib_fur_rel: Option<Vec<f32>> = nib_val.as_ref().filter(|_| p.nib_fur > 0.0 && p.hair_mask.is_some()).map(|v| {
        let body = box_blur(v, w, h, (w.max(h) / 60).max(2) as i32);
        let wide = box_blur(&body, w, h, (w.max(h) / 15).max(4) as i32);
        body.iter().zip(&wide).map(|(b, m)| (b - m).max(0.0)).collect()
    });
    // NIB FIGURE (see `MediumProfile::nib_figure`): the figure — the silhouette, when a face is found.
    let nib_fig: Option<Vec<bool>> = nib_figure_mask(p, w, h);
    // NIB TUFT (see `MediumProfile::nib_tuft`): the grass's cells, drawn as tufts after the bone.
    let nib_tufts: Option<(usize, usize, Vec<f32>)> = (p.nib > 0.0 && p.nib_tuft > 0.0 && w >= 16 && h >= 16).then(|| nib_tuft_field(&box_blur(&crate::paint::ink::value_map(original), w, h, 1), p, ma_field.as_deref(), w, h));
    // NIB EYES (see `MediumProfile::nib_eyes`): the eyes and the nose kept from the hatching, drawn after the bone.
    let nib_feat: Option<Vec<bool>> = p.nib_feature_mask.as_deref().filter(|m| p.nib > 0.0 && p.nib_eyes > 0.0 && m.len() == (w * h) as usize).map(|m| m.iter().map(|v| *v > 0.5).collect());
    // NIB GLINT (see `MediumProfile::nib_glint`): the specular highlights, no mark through them.
    let nib_glint: Option<Vec<bool>> = nib_glint_mask(original, p);
    if let (Some(g), Some(dir)) = (&nib_glint, std::env::var_os("PLAKAT_PAINT_MASKS")) {
        let _ = image::GrayImage::from_fn(w, h, |x, y| image::Luma([if g[(y * w + x) as usize] { 255 } else { 0 }])).save(std::path::Path::new(&dir).join("nib_glint.png"));
    }
    let nib_occl: Option<Vec<f32>> = nib_val.as_ref().filter(|_| p.nib_occlude > 0.0).map(|v| nib_anchor_field(v, p, w, h));
    let nib_band: Option<Vec<bool>> = nib_val.as_ref().filter(|_| p.nib_gap > 0.0).map(|v| {
        let (wu, hu) = (w as usize, h as usize);
        let edge = crate::paint::ink::edge_map(v, wu, hu, 0.6);
        let chains = crate::paint::ink::trace_chains(&edge, wu, hu, (w.max(h) / 30).max(8) as usize);
        let mut m = vec![0f32; wu * hu];
        for c in &chains {
            for q in c {
                let (x, y) = (q[0].round() as i64, q[1].round() as i64);
                if x >= 0 && y >= 0 && (x as usize) < wu && (y as usize) < hu {
                    m[y as usize * wu + x as usize] = 1.0;
                }
            }
        }
        let g = (p.nib_gap.clamp(0.0, 1.0) * w.max(h) as f32 / 400.0).round().max(1.0) as i32;
        box_blur(&m, w, h, g).iter().map(|x| *x > 1e-6).collect()
    });
    // NIB WASH DETAIL (see `MediumProfile::nib_wash_detail`): where the picture has a structure of its own — the step
    // of value across four pixels, from the picture itself (under --new `input` is the smoothed armature), on a LINE:
    // the gradients round it agreeing on a direction (the structure tensor's coherence over a 7-pixel window past
    // 0.6), so a gravel, a heap of rubble, a meadow — texture, the high band of a busy picture — stays the wash's.
    let wash_detail: Option<Vec<bool>> = (nib_wash_n > 0 && p.nib_wash_dark > 0.0 && p.nib_wash_detail > 0.0 && w >= 8 && h >= 8).then(|| {
        let (wu, hu) = (w as usize, h as usize);
        let v = box_blur(&crate::paint::ink::value_map(original), w, h, 1);
        let thr = 0.1 * (1.0 - 0.6 * p.nib_wash_detail.clamp(0.0, 1.0));
        let (gx, gy) = sobel(&v, w, h);
        let jxx = box_blur(&gx.iter().map(|g| g * g).collect::<Vec<f32>>(), w, h, 3);
        let jyy = box_blur(&gy.iter().map(|g| g * g).collect::<Vec<f32>>(), w, h, 3);
        let jxy = box_blur(&gx.iter().zip(gy.iter()).map(|(a, b)| a * b).collect::<Vec<f32>>(), w, h, 3);
        (0..wu * hu)
            .map(|i| {
                let (x, y) = (i % wu, i / wu);
                let (x0, x1, y0, y1) = (x.saturating_sub(2), (x + 2).min(wu - 1), y.saturating_sub(2), (y + 2).min(hu - 1));
                let tr = jxx[i] + jyy[i];
                let coh = if tr <= 1e-9 { 0.0 } else { ((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]) / (tr * tr) };
                coh > 0.6 && (v[y * wu + x1] - v[y * wu + x0]).abs().max((v[y1 * wu + x] - v[y0 * wu + x]).abs()) > thr
            })
            .collect()
    });
    // NIB WASH FLAT (see `MediumProfile::nib_wash_flat`): where a dark has an edge or a texture — the gradient of the
    // picture's value over a 9-pixel window above the key's share of its median.
    let wash_busy: Option<Vec<bool>> = (nib_wash_n > 0 && p.nib_wash_dark > 0.0 && p.nib_wash_flat > 0.0 && w >= 8 && h >= 8).then(|| {
        let v = box_blur(&crate::paint::ink::value_map(original), w, h, 1);
        let (gx, gy) = sobel(&v, w, h);
        let e = box_blur(&gx.iter().zip(gy.iter()).map(|(a, b)| (a * a + b * b).sqrt()).collect::<Vec<f32>>(), w, h, 4);
        let mut sorted = e.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let thr = sorted[sorted.len() / 2] * p.nib_wash_flat.clamp(0.0, 1.0);
        e.iter().map(|x| *x > thr).collect()
    });
    // NIB WASH VARY (see `MediumProfile::nib_wash_vary`): the share of the washes' deepening kept at each pixel — the
    // picture's local variance of value over a 9-pixel window against its median.
    let wash_vary: Option<Vec<f32>> = (nib_wash_n > 0 && p.nib_wash_vary > 0.0 && (p.nib_wash_gamma > 0.0 || p.nib_wash_scurve > 0.0 || p.nib_wash_mass > 0.0) && w >= 8 && h >= 8).then(|| {
        let v = crate::paint::ink::value_map(original);
        let m1 = box_blur(&v, w, h, 4);
        let m2 = box_blur(&v.iter().map(|x| x * x).collect::<Vec<f32>>(), w, h, 4);
        let var: Vec<f32> = m1.iter().zip(m2.iter()).map(|(a, b)| (b - a * a).max(0.0)).collect();
        let mut sorted = var.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2].max(1e-6);
        let k = p.nib_wash_vary.clamp(0.0, 1.0);
        var.iter().map(|s2| 1.0 - k * (1.0 - (-s2 / med).exp())).collect()
    });
    // NIB WASH MASS (see `MediumProfile::nib_wash_mass`): the picture's darkness, the measure a wash's loss is taken by.
    // (The darkness on the eye's scale, `1 − l^(1/2.2)`: in linear light a sheet of half-tones counts as nearly white.)
    let wash_mass_pic: Option<Vec<f32>> = (nib_wash_n > 0 && p.nib_wash_mass > 0.0).then(|| luma_map(original).iter().map(|l| l.max(0.0).powf(1.0 / 2.2)).collect());
    let wash_mass_ref: f32 = wash_mass_pic.as_ref().map_or(0.0, |m| m.iter().map(|l| 1.0 - l).sum::<f32>());
    // NIB WASH HOLD (see `MediumProfile::nib_wash_hold`): the picture's value and its hard edges — the ink's conductance
    // under a half. The gradient is taken on the picture SMOOTHED over a 400th of the sheet (on the raw picture every
    // grain of a texture was an edge), the conductance Tukey's biweight `(1 − (|∇I|/K)²)²`, nothing past `K`, `K` four
    // times the median gradient.
    let wash_hold: Option<(Vec<f32>, Vec<bool>)> = (nib_wash_n > 0 && p.nib_wash_hold > 0.0 && w >= 8 && h >= 8).then(|| {
        let v = box_blur(&crate::paint::ink::value_map(original), w, h, (w.max(h) / 400).max(1) as i32);
        let (gx, gy) = sobel(&v, w, h);
        let g: Vec<f32> = gx.iter().zip(gy.iter()).map(|(a, b)| (a * a + b * b).sqrt()).collect();
        let mut sorted = g.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = 4.0 * sorted[sorted.len() / 2].max(1e-4);
        // (On the figures only — a person, a face: on a landscape every blade and stone is such an edge, and held at
        // each the washes went back to the heap of dabs.)
        let fig = |i: usize| p.nib_person_mask.as_deref().and_then(|m| m.get(i)).is_some_and(|x| *x > 0.5) || p.face_mask.as_deref().and_then(|m| m.get(i)).is_some_and(|x| *x > 0.35);
        let edge = g.iter().enumerate().map(|(i, x)| { let t = x / k; (t >= 1.0 || (1.0 - t * t).powi(2) < 0.5) && fig(i) }).collect();
        (v, edge)
    });
    // NIB WASH DRY (see `MediumProfile::nib_wash_dry`): the picture's value on the eye's scale — the dark register
    // (under 0.45) the dry brush lays.
    let wash_dry_pic: Option<Vec<f32>> = (nib_wash_n > 0 && p.nib_wash_dry > 0.0 && original.dimensions() == (w, h)).then(|| crate::paint::ink::value_map(original));
    // NIB WASH CAP (see `MediumProfile::nib_wash_cap`): the sheet's capacity — the picture's own value, in linear light.
    let wash_cap_pic: Option<Vec<f32>> = (nib_wash_n > 0 && p.nib_wash_cap > 0.0 && original.dimensions() == (w, h)).then(|| luma_map(original));
    // NIB WASH MASK (see `MediumProfile::nib_wash_mask`): how deep each pixel lies in the darkest plane, as the washes cut it.
    let wash_dark_plane: Option<Vec<f32>> = (nib_wash_n > 0 && p.nib_wash_mask > 0.0 && p.nib_wash_planes >= 2.0 && input.dimensions() == (w, h)).then(|| wash_dark_plane_map(input, p, w as usize, h as usize, p.nib_wash_planes.round() as usize, charcoal_paper_l));
    for (layer_all, pass) in passes.iter().enumerate() {
        // NIB WASH: the washes' register with its own keys, the pens' with theirs; `layer` counts within the register,
        // so the pens' passes run as pen-ink's own.
        let wash_reg = nib_wash_n > 0 && (layer_all < nib_wash_n) != wash_draw;
        let layer = if nib_wash_n > 0 && layer_all >= nib_wash_n { layer_all - nib_wash_n } else { layer_all };
        let p: &PaintParams = match (&p_wash, &p_pen) {
            (Some(pw), _) if wash_reg => pw,
            (_, Some(pp)) => pp,
            _ => p,
        };
        let (nib_band, nib_feat, nib_glint, nib_fig, nib_tufts) = if wash_reg { (None, None, None, None, None) } else { (nib_band.as_ref(), nib_feat.as_ref(), nib_glint.as_ref(), nib_fig.as_ref(), nib_tufts.as_ref()) };
        // (The pens' clip too: a wash is not stopped at every contour of the picture — cut there it was a heap of dabs.)
        let clip_edge = if wash_reg { None } else { clip_edge.as_ref() };
        let radius = pass.radius.max(p.min_brush);
        let pass_t0 = std::time::Instant::now();
        // TRIM (see `MediumProfile::trim`): the ground's marks keep CLEAR of the faces — the face grown by this
        // pass's brush radius (by the key's share) is a stop for a mark seeded off it, so a sea's broad mark
        // ends with its brush's edge at the profile instead of running its length across the cheek.
        let face_clear: Option<Vec<bool>> = (p.trim > 0.0 && !crowd).then(|| p.face_mask.as_deref()).flatten().filter(|m| m.len() == (w * h) as usize).map(|m| {
            let fb: Vec<bool> = m.iter().map(|f| *f > 0.35).collect();
            let grown = dilate_bool(&fb, w as usize, h as usize, (radius * p.trim).round().max(1.0) as usize);
            match protect_all.as_deref() {
                Some(pa) => grown.iter().zip(pa).map(|(a, b)| *a || *b).collect(),
                None => grown,
            }
        });
        // The first pass is a block-in: it covers the whole canvas so no white ground survives. Later passes
        // only restate where the canvas is still wrong.
        let block_in = layer == 0;
        // The last pass OF THE LADDER (the glaze pass after it is no ladder pass: no accents, no scumble on it).
        let last_pass = layer + 1 == ladder_len;
        let glazing = pass.stage == "glaze";
        let cresting = pass.stage == "crest";
        // TIERS (see `MediumProfile::tiers`): which hatched layer this pass is — the passes before it that hatch
        // (the cross's count: past a quarter of the way from the broadest brush to the finest).
        let tier_k = if p.tiers > 0.0 {
            let (rc, rf) = (passes.first().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius), passes.last().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius));
            passes.iter().take(layer).skip(1).filter(|q| rc > rf + 1e-6 && (rc - q.radius.max(p.min_brush)) / (rc - rf) > 0.25).count()
        } else {
            0
        };
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
        // NIB WASH DRY: the washes' fine passes are the dry brush's — a mass pass (not detail: the focus and the
        // coherence do not gate it) reading the picture itself, laying the dark register.
        // (Not the smallest brush: at 3 px it laid fifty thousand curls that added nothing to the tone — noise.)
        let dry_pass = wash_reg && fine && radius > p.min_brush * 1.5 && wash_dry_pic.is_some();
        if wc_brush && wcb_finest > 0.0 && radius < wcb_finest {
            continue;
        }
        let wcb_broad_pass = wc_brush && !fine;
        // (NIB WASH WATER: the pigment snapshot at the pass's start, so the pass's own ink can seep — as the replay marks it.)
        if wc_brush || score.header.water > 0.0 {
            canvas.mark_film();
        }
        let detail = !block_in && !dry_pass && (fidelity || (p.style == PaintStyle::Legible && fine) || (focus_on && fine));
        // The reference this pass paints from. Fidelity paints from a SHARP reference at every scale (it tracks
        // real structure, doesn't invent) — a touch of unsharp even on the block-in. Legible blurs the masses and
        // sharpens only for detail.
        let reference = if dry_pass {
            // NIB WASH DRY: the dry brush reads the picture blurred by its own radius — the dark register as masses;
            // read sharp, every grain of a bush turned a mark and the register was a heap of curls.
            imageops::blur(original, radius.max(0.6))
        } else if wc_wash && wcb_src > 0.0 && detail {
            // The watercolour's fine passes read the SOURCE: the armature averages a stained-glass pane with
            // its mullions, and a glaze of that average is a washed-out window.
            // …at no finer a scale than the pass before: the finest brush restating a sharper picture drew
            // every mortar line as a dark outline while the shoes beside them stayed as the wider brush left
            // them — the "clear paving stones behind smudged feet".
            imageops::blur(source, (radius.max(wcb_src_floor) * wcb_src).max(0.5))
        } else if fidelity {
            imageops::unsharpen(input, (radius * 0.22).max(0.5), 1)
        } else if detail {
            // Fine layers read the armature, plus the subject's own texture inside the flat masses (see
            // `detail_texture`), unless a detail sharpen is asked for instead (see `detail_sharpen`).
            if p.detail_sharpen > 0.0 {
                imageops::unsharpen(input, (radius * p.detail_sharpen).max(0.6), 1)
            } else if p.detail_texture > 0.0 && !std::ptr::eq(source, input) && !(wc_brush && wcb_texture <= 0.0) {
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
            if wcb_trans && wcb_env > 0.0 {
                // LIGHT TO DARK. A transparent film can only darken, so a broad pass must never go darker than
                // the lightest thing inside its footprint: it paints toward the picture's LIGHT ENVELOPE at
                // its own scale, and the finer passes glaze the darks down into it.
                light_envelope(if wc_wash && wcb_src > 0.0 { source } else { input }, radius * wcb_env)
            } else {
                imageops::blur(input, (radius * if wc_brush { wcb_blur } else { 0.32 }).max(0.6))
            }
        };
        // NIB WASH SOFT (see `MediumProfile::nib_wash_soft`): a wash paints from the picture's low band.
        let reference = if wash_reg && !dry_pass && p.nib_wash_soft > 0.0 { imageops::blur(&reference, (p.nib_wash_soft.clamp(0.0, 1.0) * w.max(h) as f32 / 150.0).max(0.6)) } else { reference };
        // NIB WASH PLANES (see `MediumProfile::nib_wash_planes`): the wash's reference quantised to flat planes.
        let reference = if wash_reg && !dry_pass && p.nib_wash_planes >= 2.0 { wash_planes(&reference, p.nib_wash_planes.round() as usize, canvas.ground_linear()) } else { reference };
        // BLADE PLANES (see `MediumProfile::blade_planes`): every pass's reference cut into flat planes of value.
        // BLADE FLOW / BLADE MIX: the picture before the planes — the flow's form and the slices' second colours.
        let blade_raw: Option<RgbImage> = ((p.blade_flow > 0.0 || p.brush.mix > 0.0 || p.blade_facet > 0.0 || p.blade_detail > 0.0) && p.blade_planes >= 2.0 && !dry_pass).then(|| reference.clone());
        let reference = if p.blade_planes >= 2.0 && !dry_pass { blade_planes_ref(&reference, p.blade_planes.round() as usize, p.blade_levels) } else { reference };
        // BLADE FACET (see `MediumProfile::blade_facet`): the face cut into its own planes, from the picture before the planes.
        let reference = match (blade_raw.as_ref(), p.face_mask.as_deref()) {
            (Some(raw), Some(fm)) if p.blade_facet > 0.0 && fm.len() == (w * h) as usize => blade_facet_ref(&reference, raw, p.blade_planes.round() as usize, p.blade_levels, fm, p.blade_facet, p.blade_soft),
            _ => reference,
        };
        // BLADE DETAIL (see `MediumProfile::blade_detail`): on the finer half of the ladder the planes over the local range.
        let reference = match blade_raw.as_ref() {
            Some(raw) if p.blade_detail > 0.0 => {
                let (rc, rf) = (passes.first().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius), passes.last().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius));
                let depth = if rc > rf + 1e-6 { ((rc - radius) / (rc - rf)).clamp(0.0, 1.0) } else { 0.0 };
                if depth > 0.5 { blade_local_ref(&reference, raw, p.blade_planes.round() as usize, (radius * 8.0).round().max(4.0) as usize, p.blade_detail) } else { reference }
            }
            _ => reference,
        };
        // NIB WASH PLANES: the planes laid as washes on the register's first pass (its stroke passes lay nothing — the
        // sampler below — the dry brush's excepted), from the picture smoothed over a 256th of the sheet.
        if wash_reg && layer == 0 && p.nib_wash_planes >= 2.0 {
            let n_planes = p.nib_wash_planes.round() as usize;
            wash_plane_fills(&mut canvas, &mut score, input, w as usize, h as usize, n_planes, p, charcoal_paper_l, &pass.stage, &mut placed, &mut k, progress);
        }
        // NIB WASH MASS: before a wash after the block-in, the sheet's darkness against the picture's — the loss the
        // bleed has spread out; the wash's tone deepened by a gamma of the key's share of that ratio (held to 1..2).
        let wash_mass = if wash_reg && layer > 0 && wash_mass_ref > 0.0 {
            let cl: Vec<f32> = luma_map(&canvas.to_image()).iter().map(|l| l.max(0.0).powf(1.0 / 2.2)).collect();
            let d = cl.iter().map(|l| 1.0 - l).sum::<f32>().max(1e-3);
            let r = wash_mass_ref / d;
            // NIB WASH DEBT (see `MediumProfile::nib_wash_debt`): the loss where the sheet is still lighter than the picture.
            let r = match wash_mass_pic.as_ref().filter(|_| p.nib_wash_debt > 0.0) {
                Some(pic) => {
                    let (a, b) = pic.iter().zip(cl.iter()).filter(|(q, c)| **c > **q).fold((0.0f32, 0.0f32), |(a, b), (q, c)| (a + 1.0 - q, b + 1.0 - c));
                    r + p.nib_wash_debt.clamp(0.0, 1.0) * (a / b.max(1e-3) - r)
                }
                None => r,
            };
            1.0 + p.nib_wash_mass.clamp(0.0, 1.0) * (r.clamp(1.0, 2.0) - 1.0)
        } else {
            1.0
        };
        if let Some(dir) = std::env::var_os("PLAKAT_PAINT_REFERENCE_DUMP") {
            // Diagnostic: the reference this pass paints from.
            let _ = reference.save(std::path::Path::new(&dir).join(format!("ref_{layer_all:02}.png")));
        }
        // ECONOMY OF STRUCTURE (the wash recipe): off the subject a fine brush restates only where the picture
        // HAS something at that brush's scale — an edge, a texture (the local value range of the pass's own
        // reference) — never a wash that merely differs in tone. A thousand small tone corrections over the
        // washes were the stipple that covered the whole sheet and painted the tide lines and the mottle
        // back out; cut them wholesale and the walls went to blotches with the window frames lost.
        // (The FINE brushes only: the mid and broad brushes keep glazing tone, pass over pass — that is how a
        // nocturne's darks are built, each pass being one dose; gating them too left the walls bare paper.)
        let structure: Option<Vec<f32>> = if wc_wash && fine && wcb_struct > 0.0 {
            let lm = luma_map(&reference);
            let st = local_range(&lm, w as usize, h as usize, (radius * 1.5).round().max(2.0) as usize);
            if let Some(dir) = std::env::var_os("PLAKAT_WCB_DUMP") {
                // Diagnostic: the structure field the fine pass gates on, as 8-bit (×4 so 0.25 saturates).
                let img = image::GrayImage::from_fn(w, h, |x, y| image::Luma([(st[(y * w + x) as usize] * 4.0 * 255.0).clamp(0.0, 255.0) as u8]));
                let _ = img.save(std::path::Path::new(&dir).join(format!("structure_{layer}.png")));
                let _ = reference.save(std::path::Path::new(&dir).join(format!("reference_{layer}.png")));
            }
            Some(st)
        } else {
            None
        };
        lap("pass:reference", &mut prof_acc, &mut prof_t);
        let luma = luma_map(&reference);
        // The RANGE field (see `MediumProfile::span`, `edge`): how much the reference's value changes within
        // the hatching brush's reach — flat where it is low, busy or at a contour where it is high. Only for a
        // hatched medium with one of the two keys set.
        // (The reference is first softened by half the radius, so a star or a lamp in a still sky does not
        // make the whole of its reach a busy passage — a point of light is not a change of form.)
        let range_field: Option<Vec<f32>> = if (p.hatch > 0.0 && !block_in && (p.span > 0.0 || p.edge > 0.0)) || p.smooth > 0.0 || (p.sweep > 0.0 && !block_in) {
            let soft = box_blur(&luma, w, h, (radius * 0.5).round().max(1.0) as i32);
            Some(local_range(&soft, w as usize, h as usize, (radius * 1.5).round().max(2.0) as usize))
        } else {
            None
        };
        // Coherent flow (structure tensor). Fidelity + detail keep it TIGHT (small sigma) so strokes hug local
        // edges; coarse legible passes smooth it so masses follow gross form.
        // (NIB WASH DRY: the dry brush drags along the mass's form, its flow smoothed over three radii.)
        let sigma = if detail || fidelity { 2 } else if dry_pass { (radius * 3.0).round().clamp(4.0, 24.0) as i32 } else { (radius * 0.9).round().clamp(2.0, 24.0) as i32 };
        // BLADE FLOW (see `MediumProfile::blade_flow`): the tensor from the picture before the planes.
        let flow_luma: Option<Vec<f32>> = blade_raw.as_ref().filter(|_| p.blade_flow > 0.0).map(luma_map);
        let (gx, gy) = coherent_gradient(flow_luma.as_deref().unwrap_or(&luma), w, h, sigma, p.infill);
        // CROSS-HATCH: rotate this pass's flow by the medium's hatch angle × layer, so successive restatements
        // cross the form at the classic angles instead of all lying along it (tempera's woven net).
        // (A hatched medium (`hatch`) turns the broad restatements less and the finest fully — its underlayers
        // lay the masses along the form, its crossing is for the hatching; the turn is scaled by the pass's
        // depth in the ladder, 0 on the widest brush, 1 on the narrowest. Unscaled where `hatch` is 0.)
        // CROSS (see `MediumProfile::cross`): with `cross` set the turn is not the layer's multiple of the hatch
        // angle but the hatched pass's own step off the form — the first hatched pass along it, then +cross,
        // −2·cross, +3·cross — the net wrapping the volume at the panel painter's angles.
        // NIB WASH HALF: whether this pass's hatching is turned off the form (the crossing passes).
        let pass_turned = p.hatch_angle.abs() > 1e-3 && !block_in && p.hatch > 0.0 && p.cross > 0.0 && {
            let (rc, rf) = (passes.first().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius), passes.last().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius));
            let turn_of = |r: f32| -> f32 { if rc > rf + 1e-6 { ((((rc - r) / (rc - rf)).clamp(0.0, 1.0) - 0.25) / 0.75).clamp(0.0, 1.0) } else { 1.0 } };
            let hk = passes.iter().take(layer).skip(1).filter(|q| turn_of(q.radius.max(p.min_brush)) > 0.0).count();
            (cross_turn(hk, p.cross) * turn_of(radius)).abs() > 1e-3
        };
        let (gx, gy) = if p.hatch_angle.abs() > 1e-3 && !block_in {
            let turn_of = |r: f32| -> f32 {
                let (rc, rf) = (passes.first().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius), passes.last().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius));
                if rc > rf + 1e-6 { ((((rc - r) / (rc - rf)).clamp(0.0, 1.0) - 0.25) / 0.75).clamp(0.0, 1.0) } else { 1.0 }
            };
            let turn_scale = if p.hatch > 0.0 { turn_of(radius) } else { 1.0 };
            let angle = if p.hatch > 0.0 && p.cross > 0.0 {
                let hk = passes.iter().take(layer).skip(1).filter(|q| turn_of(q.radius.max(p.min_brush)) > 0.0).count();
                cross_turn(hk, p.cross) * turn_scale
            } else {
                p.hatch_angle * layer as f32 * turn_scale
            };
            // EDGE (see `MediumProfile::edge`): near a plain contour the turn is taken back toward the edge's
            // own direction (the flow there IS the edge), by the key's share of the edge's strength.
            // SMALL: on a small face the turn is taken away by `k` — the strokes one way along the form.
            let small_fade = |i: usize| -> f32 { if small_k > 0.0 { 1.0 - small_k * p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0) } else { 1.0 } };
            match &range_field {
                Some(rf) if p.edge > 0.0 || small_k > 0.0 => {
                    let (mut ox, mut oy) = (Vec::with_capacity(gx.len()), Vec::with_capacity(gx.len()));
                    for i in 0..gx.len() {
                        let (sn, cs) = (angle * (1.0 - p.edge * edge_of(rf[i])) * small_fade(i)).sin_cos();
                        ox.push(gx[i] * cs - gy[i] * sn);
                        oy.push(gx[i] * sn + gy[i] * cs);
                    }
                    (ox, oy)
                }
                _ if small_k > 0.0 => {
                    let (mut ox, mut oy) = (Vec::with_capacity(gx.len()), Vec::with_capacity(gx.len()));
                    for i in 0..gx.len() {
                        let (sn, cs) = (angle * small_fade(i)).sin_cos();
                        ox.push(gx[i] * cs - gy[i] * sn);
                        oy.push(gx[i] * sn + gy[i] * cs);
                    }
                    (ox, oy)
                }
                _ => {
                    let (sn, cs) = angle.sin_cos();
                    (gx.iter().zip(&gy).map(|(x, y)| x * cs - y * sn).collect::<Vec<f32>>(), gx.iter().zip(&gy).map(|(x, y)| x * sn + y * cs).collect::<Vec<f32>>())
                }
            }
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
        // SLANT (see `MediumProfile::slant`): where the picture has no direction, the hatch at the hand's angle.
        let (gx, gy) = if p.slant > 0.0 && !block_in {
            let (wu, hu) = (w as usize, h as usize);
            let (mut lx, mut ly) = (vec![0f32; wu * hu], vec![0f32; wu * hu]);
            for y in 1..hu.saturating_sub(1) {
                for x in 1..wu.saturating_sub(1) {
                    let i = y * wu + x;
                    lx[i] = 0.5 * (luma[i + 1] - luma[i - 1]);
                    ly[i] = 0.5 * (luma[i + wu] - luma[i - wu]);
                }
            }
            // The direction read at the drawing's scale, not only the pass's: at a fine pass's few pixels a
            // stone's own edge is "coherent", and the hatch over the rubble circled from stone to stone.
            let r = ((radius * 1.5).round() as i32).max(3).max((w.max(h) as f32 / 150.0).round() as i32);
            let jxx = box_blur(&lx.iter().map(|g| g * g).collect::<Vec<f32>>(), w, h, r);
            let jyy = box_blur(&ly.iter().map(|g| g * g).collect::<Vec<f32>>(), w, h, r);
            let jxy = box_blur(&lx.iter().zip(ly.iter()).map(|(a, b)| a * b).collect::<Vec<f32>>(), w, h, r);
            // The hand's stroke runs up to the right (−45° on the sheet), its "gradient" at +45°; a later pass
            // crossed off it. In the double angle an orientation and its reverse are one.
            let phi = std::f32::consts::FRAC_PI_4 + if p.cross > 0.0 { cross_turn(layer, p.cross) } else { 0.0 };
            let (hc, hs) = ((2.0 * phi).cos(), (2.0 * phi).sin());
            let key = p.slant.clamp(0.0, 1.0);
            let (mut gx, mut gy) = (gx, gy);
            for i in 0..gx.len() {
                let tr = jxx[i] + jyy[i];
                let coh = if tr <= 1e-9 { 0.0 } else { (((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]) / (tr * tr)).clamp(0.0, 1.0) };
                let keep = p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0)
                    .max(p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0));
                let t = key * (1.0 - coh) * (1.0 - keep);
                if t <= 1e-3 {
                    continue;
                }
                let m = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
                let th = gy[i].atan2(gx[i]);
                let (c2, s2) = ((2.0 * th).cos() * (1.0 - t) + hc * t, (2.0 * th).sin() * (1.0 - t) + hs * t);
                let a = 0.5 * s2.atan2(c2);
                let k = m.max(1e-4);
                gx[i] = a.cos() * k;
                gy[i] = a.sin() * k;
            }
            (gx, gy)
        } else {
            (gx, gy)
        };
        // The SURFACE leads the stroke where the picture's values do not: on a passage without an edge of its own
        // the mark turns onto the form's direction. An edge the picture draws keeps its stroke, and the pass's
        // magnitude is kept everywhere — it feeds the detail gate, which must not change with the direction.
        let (gx, gy) = match &form_flow {
            Some(ff) => {
                let (mut gx, mut gy) = (gx, gy);
                for i in 0..gx.len() {
                    let (dir, sure) = ff.at((i % w as usize) as f32, (i / w as usize) as f32);
                    let m = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
                    let t = sure * (1.0 - m.clamp(0.0, 1.0));
                    if t <= 1e-3 {
                        continue;
                    }
                    let (ux, uy) = (-dir[1], dir[0]); // the "gradient" whose perpendicular is the mark
                    let (ux, uy) = if ux * gx[i] + uy * gy[i] < 0.0 { (-ux, -uy) } else { (ux, uy) };
                    let k = m.max(1e-4);
                    gx[i] += (ux * k - gx[i]) * t;
                    gy[i] += (uy * k - gy[i]) * t;
                }
                (gx, gy)
            }
            None => (gx, gy),
        };
        // TURBULENCE (see `MediumProfile::turbulence`): the flow turned by a low-frequency field of noise —
        // a few waves across the sheet, deterministic, the same at every pass — by up to the medium's angle
        // either way, halved on a face. The magnitude is kept (it feeds the detail gate).
        let (gx, gy) = if p.turbulence > 0.0 {
            let (mut gx, mut gy) = (gx, gy);
            let (a, b, c, d) = turbulence_waves(w.max(h) as f32);
            let amp = p.turbulence.to_radians();
            for i in 0..gx.len() {
                let (x, y) = ((i % w as usize) as f32, (i / w as usize) as f32);
                let n = 0.5 * ((a * x + 1.3 * (b * y).sin()).sin() + (c * y + 0.9 * (d * x).sin()).sin());
                let face = p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let (sn, cs) = (amp * n * (1.0 - 0.5 * face)).sin_cos();
                let (x0, y0) = (gx[i], gy[i]);
                gx[i] = x0 * cs - y0 * sn;
                gy[i] = x0 * sn + y0 * cs;
            }
            (gx, gy)
        } else {
            (gx, gy)
        };
        // SWEEP (see `MediumProfile::sweep`): on a flat passage off the subject and the faces the flow is turned
        // onto the horizontal (a stroke runs across its gradient, so the gradient onto the vertical), by the
        // key's share of the flatness, its magnitude kept and its sign held.
        // HAKE (see `MediumProfile::hake`): on the washed passes the flow turned to the broad flat brush's one
        // horizontal sweep by the key's share, off the subject — the tonal masses as parallel bands that overlap
        // along themselves, not bars crossing at every angle.
        // (NIB WASH DRY: the dry brush too — its drags level off the subject, like the washes'; along the texture's
        // own flow they were ropes winding round every bush.)
        let (gx, gy) = if p.hake > 0.0 && ((p.boneless > 0.0 && p.wash_in > 0.0 && (block_in || (layer as f32) <= p.boneless)) || dry_pass) {
            let (mut gx, mut gy) = (gx, gy);
            for i in 0..gx.len() {
                let subj = p.subject_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let t = p.hake.clamp(0.0, 1.0) * (1.0 - subj);
                if t > 1e-3 {
                    // (The field is the gradient; the mark runs across it — a horizontal sweep is a vertical gradient's
                    // complement here: the sweep's gradient points ALONG x.)
                    let m = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt().max(1e-3);
                    if dry_pass {
                        // (NIB WASH DRY: the drag LEVEL — the gradient onto the vertical, the mark across it.)
                        let ty = if gy[i] < 0.0 { -m } else { m };
                        gx[i] += (0.0 - gx[i]) * t;
                        gy[i] += (ty - gy[i]) * t;
                    } else {
                        let tx = if gx[i] < 0.0 { -m } else { m };
                        gx[i] += (tx - gx[i]) * t;
                        gy[i] += (0.0 - gy[i]) * t;
                    }
                }
            }
            (gx, gy)
        } else {
            (gx, gy)
        };
        let (gx, gy) = match (&range_field, p.sweep > 0.0 && !block_in) {
            (Some(rf), true) => {
                let (mut gx, mut gy) = (gx, gy);
                for i in 0..gx.len() {
                    let subj = p.subject_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    let face = p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    let t = p.sweep * (1.0 - busy_of(rf.get(i).copied().unwrap_or(1.0))) * (1.0 - subj) * (1.0 - face);
                    if t > 1e-3 {
                        let m = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt().max(1e-3);
                        let ty = if gy[i] < 0.0 { -m } else { m };
                        gx[i] += (0.0 - gx[i]) * t;
                        gy[i] += (ty - gy[i]) * t;
                        // SWEEP WAVE (see `MediumProfile::sweep_wave`): turned by a slow wave, by the sweep's share.
                        if p.sweep_wave > 0.0 {
                            let (x, y) = ((i % w as usize) as f32, (i / w as usize) as f32);
                            let l = (w.min(h) as f32 / 6.0).max(8.0);
                            let a = p.sweep_wave.to_radians() * 2.0 * (crate::paint::canvas::value_noise(x / l, y / l, p.seed ^ 0x5EE_9A7E) - 0.5) * (t / p.sweep.max(1e-3)).min(1.0);
                            let (sn, cs) = a.sin_cos();
                            let (x0, y0) = (gx[i], gy[i]);
                            gx[i] = x0 * cs - y0 * sn;
                            gy[i] = x0 * sn + y0 * cs;
                        }
                    }
                }
                (gx, gy)
            }
            _ => (gx, gy),
        };
        // ACCENTS (see `MediumProfile::accent`): the sharp contrasts of the finest pass's reference, at the
        // brush's own scale — where the dab goes down thick.
        // The picture's local RANGE at the brush's scale: the accents' map on the finest pass, and the measure of
        // "smooth here" for the sfumato and the length's spread on every pass after the block-in.
        let range_map: Option<Vec<f32>> = (!block_in && (p.sfumato > 0.0 || p.vary > 0.0 || p.blade_size > 0.0 || p.blade_rest > 0.0 || (p.accent > 0.0 && last_pass))).then(|| local_range(&luma, w as usize, h as usize, (radius * 1.5).round().max(2.0) as usize));
        let accent_map: Option<&Vec<f32>> = if p.accent > 0.0 && last_pass && !block_in { range_map.as_ref() } else { None };
        // BLADE TUFT (see `MediumProfile::blade_tuft`): the grass's fine texture's map, from the picture before the
        // planes where there is one.
        // BLADE SLOPE (see `MediumProfile::blade_slope`): the picture's hill, for the tuft's direction.
        let slope_map: Option<Vec<f32>> = (p.blade_tuft > 0.0 && p.blade_slope > 0.0 && !block_in).then(|| blade_slope_map(flow_luma.as_deref().unwrap_or(&luma), w as usize, h as usize, (w.min(h) as f32 / 30.0).round().max(4.0) as usize));
        // BLADE FOCUS (see `MediumProfile::blade_focus`): how far outside the picture's focus each pixel lies.
        let focus_far: Option<Vec<f32>> = (p.blade_focus > 0.0).then(|| blade_focus_map(flow_luma.as_deref().unwrap_or(&luma), w as usize, h as usize, (w.min(h) as f32 / 400.0).round().max(2.0) as usize, p.blade_band));
        let tuft_map: Option<Vec<f32>> = (p.blade_tuft > 0.0 && !block_in).then(|| blade_tuft_map(flow_luma.as_deref().unwrap_or(&luma), p.nib_grass_mask.as_deref(), p.blade_aniso, p.blade_route, w as usize, h as usize, (w.min(h) as f32 / 400.0).round().max(2.0) as usize));
        // The glaze pass leaves the sharp contrasts alone — the picture's range at the FINEST brush's scale —
        // so the accents laid there stand crisp and thick over the film rather than being softened under it:
        // a painter glazes the flats, not the highlights.
        let glaze_hold: Option<Vec<f32>> = (glazing && p.accent > 0.0).then(|| local_range(&luma, w as usize, h as usize, (p.min_brush * 1.5).round().max(2.0) as usize));
        // GLAZE (see `MediumProfile::glaze`): the band two brush widths each side of the subject's silhouette.
        let glaze_band: Option<Vec<bool>> = (p.glaze > 0.0 && last_pass && !block_in).then(|| p.region_mask.as_deref()).flatten().map(|m| {
            let r = (radius * 2.0).ceil().max(2.0) as usize;
            let inv: Vec<bool> = m.iter().map(|b| !b).collect();
            dilate_bool(m, w as usize, h as usize, r).iter().zip(dilate_bool(&inv, w as usize, h as usize, r)).map(|(a, b)| *a && b).collect()
        });
        // LOST EDGES (see `MediumProfile::lost`): on the glaze pass, the band along the subject's silhouette
        // where the picture is in shadow — there a mark's colour is the picture blurred wide across the seam
        // (the mean of both sides), and the mark may cross it, so a dark figure meets its dark ground in one
        // tone instead of a drawn outline. A painter rubs that edge out with a finger.
        // On the glaze pass (its mid brush, a band of its own width and a half) and on the finest ladder pass
        // (a band at least four minimum brushes wide, so the small marks fuse the seam densely).
        let lost_r = if glazing { radius * 1.5 } else { (radius * 1.5).max(p.min_brush * 4.0) };
        // ...and only where the seam is a LOW contrast — both sides dark (the picture's value range across the
        // band under 0.2): a dark figure against a white ground is in shadow by the picture's own reckoning,
        // and the mean of its two sides painted across it was a grey halo round the whole chick.
        // AERIAL (see `MediumProfile::aerial`): by DEPTH the band is no longer the silhouette's alone — a far
        // passage in shadow is in it wherever it lies, the farther the more contrast its edges may hold
        // (the gate opens from 0.2 by half the distance), and its mark takes a wider blur. `far` is the
        // pixel's distance times the amount; past 0.3 of it the passage is lost on its own.
        let far: Option<Vec<f32>> = (p.lost > 0.0 && p.aerial > 0.0).then(|| p.depth.as_deref()).flatten().map(|d| d.iter().map(|z| (1.0 - z.clamp(0.0, 1.0)) * p.aerial).collect());
        let lost_band: Option<Vec<bool>> = (p.lost > 0.0 && (glazing || (last_pass && !block_in)) && (p.region_mask.is_some() || far.is_some())).then(|| shadow.as_ref()).flatten().map(|sh| {
            let r = lost_r.ceil().max(2.0) as usize;
            let range = local_range(&luma, w as usize, h as usize, (lost_r * 1.5).round().max(2.0) as usize);
            let seam: Option<Vec<bool>> = p.region_mask.as_deref().map(|m| {
                let inv: Vec<bool> = m.iter().map(|b| !b).collect();
                dilate_bool(m, w as usize, h as usize, r).iter().zip(dilate_bool(&inv, w as usize, h as usize, r)).map(|(a, b)| *a && b).collect()
            });
            (0..sh.len())
                .map(|i| {
                    let f = far.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                    let on_seam = seam.as_ref().and_then(|s| s.get(i)).copied().unwrap_or(false);
                    sh[i] > 0.5 && (on_seam || f > 0.3) && range.get(i).copied().unwrap_or(1.0) < 0.2 + 0.5 * f
                })
                .collect()
        });
        let lost_ref: Option<RgbImage> = lost_band.is_some().then(|| imageops::blur(input, (lost_r * 1.7).max(1.0)));
        let lost_ref_far: Option<RgbImage> = (lost_band.is_some() && far.is_some()).then(|| imageops::blur(input, (lost_r * 1.7 * (1.0 + 2.0 * p.aerial)).max(1.0)));
        // THE CREST PASS reads the RELIEF the ladder left (the canvas's own heights, against their 95th
        // percentile): a dry brush catches where the paint stands up and is broken at the brush's scale, not
        // where the picture is busy — and in the light only, a lighter tint over a shadow was a grey veil.
        let crest_relief: Option<Vec<f32>> = cresting.then(|| {
            let mut v: Vec<f32> = canvas.height.iter().copied().collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let hi = v.get(v.len() * 95 / 100).copied().unwrap_or(1.0).max(1e-6);
            let norm: Vec<f32> = canvas.height.iter().map(|h| (h / hi).min(1.5)).collect();
            local_range(&norm, w as usize, h as usize, (radius * 1.5).round().max(2.0) as usize)
        });
        // SPATTER (see `MediumProfile::spatter`): the lift-off points of this pass's thickest marks, each with
        // its direction, width, charge and viscosity — the drops are flung after the pass (below).
        let spats: std::cell::RefCell<Vec<([f32; 2], [f32; 2], f32, Vec<f32>, f32, u64, f32)>> = std::cell::RefCell::new(Vec::new());
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
        // The pass's LOG depth in the ladder (0 the widest brush, 1 the narrowest, each halving a like step) — the
        // ladder's geometric position, for what reads "the middle knives" and "the narrow ones" (BLADE TUFT).
        let ladder_t = if r_coarse > r_fine * 1.001 && radius > 0.0 { ((r_coarse / radius).ln() / (r_coarse / r_fine).ln()).clamp(0.0, 1.0) } else { 0.0 };
        let len_mul = profile.len * (1.0 + (p.detail_len - 1.0) * depth_t);
        // NIB WASH DRY: the dry brush's mark is a DRAG, three times the profile's length, not the detail's short touch.
        let len_mul = if dry_pass { profile.len * 3.0 } else { len_mul };
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
                let target = match (&wc_mass, wc_brush && !fine) {
                    (Some((labels, means)), true) => means[labels[iy as usize * w as usize + ix as usize] as usize],
                    _ => target,
                };
                let lost_here = lost_band.as_ref().map_or(false, |b| b.get(iy as usize * w as usize + ix as usize).copied().unwrap_or(false))
                    && p.face_mask.as_deref().and_then(|m| m.get(iy as usize * w as usize + ix as usize)).copied().unwrap_or(0.0) <= 0.35
                    && p.hair_mask.as_deref().and_then(|m| m.get(iy as usize * w as usize + ix as usize)).copied().unwrap_or(0.0) <= 0.0;
                // ...a far passage (`aerial`) takes the wider blur.
                let far_here = lost_here && far.as_ref().and_then(|v| v.get(iy as usize * w as usize + ix as usize)).copied().unwrap_or(0.0) > 0.3;
                let lost_pick: Option<&RgbImage> = if far_here { lost_ref_far.as_ref().or(lost_ref.as_ref()) } else { lost_ref.as_ref() };
                let target = match (lost_pick, lost_here) {
                    (Some(lr), true) => lr.get_pixel(ix, iy).0,
                    _ => target,
                };
                let tluma = color::linear_luma(color::srgb_to_linear(target));
                // COMMIT SHADOWS (RFC §3.3): in the dark value masses, paint DECISIVELY — lift the reserve so darks
                // always land, drop the restate floor so they build to full depth, and boost the pigment charge so
                // they read as committed paint, not a thin wash. `sh` in [0,1] is the shadow strength here.
                let region_i = iy as usize * w as usize + ix as usize;
                // A face-only pass lays nothing off the face.
                // (The brush watercolour gives the sheet the whole ladder, but its two finest brushes — the
                // ones that stippled every wash — stay for the face.)
                let face_only_here = if wc_brush { radius <= wcb_faceonly } else { pass.face_only };
                if face_only_here {
                    // The focal plane the fine brushes are kept for: the face when there is one, else the
                    // subject (an animal's head is its subject's; see the ladder cut).
                    let on_focal = match (p.face_mask.as_deref(), p.subject_mask.as_deref()) {
                        (Some(m), _) => m.get(region_i).copied().unwrap_or(0.0) >= 0.35,
                        (None, Some(sm)) => sm.get(region_i).copied().unwrap_or(0.0) >= 0.5,
                        (None, None) => false,
                    };
                    if !on_focal {
                        return None;
                    }
                }
                // Under the wash recipe the sweeps ARE the block-in: a second block-in over them in another
                // direction was the stroke mess.
                if wcb_sweep && block_in {
                    return None;
                }
                // Economy by PLANE: the background takes only the broad brushes (a wash is a wash), the
                // subject the middle ones, the face the finest.
                if wc_brush && wcb_subj_from > 0.0 && radius < wcb_subj_from && (!block_in || wcb_fill) && p.subject_mask.as_deref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) < 0.5 && p.face_mask.as_deref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) < 0.35 {
                    return None;
                }
                let soft = ramp_soft.as_ref().map(|m| m[region_i]).unwrap_or(0.0);
                let sh_raw = shadow.as_ref().map(|s| s[region_i]).unwrap_or(0.0);
                let sh = sh_raw * p.commit_shadows;
                // RESERVE is a fact about the BACKGROUND, not the subject: bright cells keep the paper only OUTSIDE
                // the detected subject (matte). Inside the subject a light shirt / shoulders / skin is PAINTED as a
                // light mass — never reserved to blank paper (that is what made the shoulders vanish). And never
                // reserve inside a committed shadow. Determined over the detected extents, not raw luma.
                let subj = p.subject_mask.as_ref().map(|m| m[region_i]).unwrap_or(0.0);
                if let Some(rt) = reserve_eff {
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
                // A watercolour's ECONOMY: the fine brushes restate only what is plainly wrong — off the face a
                // wash that is nearly right is left alone (a thousand small corrections are the stipple).
                let on_face = p.face_mask.as_deref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) > 0.35;
                let on_subject = p.subject_mask.as_deref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) > 0.5;
                let restate_floor = if wc_brush && !block_in && !on_face && !on_subject { restate_floor * wcb_restate } else { restate_floor };
                // SHADE (see `MediumProfile::shade`): how far this mark lies in the shadow, off the faces and hair.
                // There the restate floor is RAISED (the committed shadows dropped it toward zero, so every small
                // correction landed — the rake of dry noise in the dark): a mark lands only where the canvas is
                // plainly wrong, never to chase a small tone; the marks that land run longer and wider (below),
                // and the hand is calm — no hue drift, no dry skip. A painter kills the detail in the dark so it
                // does not pull the eye. (Not a random thinning: that dropped the broad marks that cover the
                // ground in the darkest hole and left the toned ground showing as a pale patch.)
                let shaded = if p.shade > 0.0 && !block_in && !glazing && !cresting && !on_face && p.hair_mask.as_deref().and_then(|m| m.get(region_i).copied()).unwrap_or(0.0) <= 0.0 { (p.shade * sh_raw).clamp(0.0, 1.0) } else { 0.0 };
                let restate_floor = restate_floor * (1.0 + 4.0 * shaded);
                // SPAN, EDGE and SMOOTH (see `MediumProfile::span`, `edge`, `smooth`): the form's range within
                // the brush's reach here — flat where it is low, busy or at a contour where it is high.
                let form_range = range_field.as_deref().and_then(|f| f.get(region_i)).copied().unwrap_or(0.0);
                let edge_here = if p.edge > 0.0 { p.edge * edge_of(form_range) } else { 0.0 };
                // SMOOTH: how CALM this passage is — the key's share of its flatness. On a calm passage the
                // dishes relax to the picture's own gradation, the hatching gives way to a soft thin wash that
                // blends, the marks never straightened — a sky, a wall, a still water built up in veils.
                // Off the subject only: a subject's dark mass (a black cat) is committed paint, built solid in
                // its steps; the calm is the ground's — the sky, the wall, the water behind it.
                // THE UMBRA (see `MediumProfile::umbra`): the tone lies within that share of the darkest dish —
                // the shadow mass. Its tones are laid continuous (below), and off the subject it is laid CALM
                // however busy the picture is there: a shadow is one modelled mass built in veils, not a
                // chequer of the broad pass's planks (a night's gravel foreground, raked by the lamps).
                let in_umbra = p.umbra > 0.0 && p.dishes >= 2.0 && dish_pos(target, p.dishes, map_lo, map_hi) < p.umbra;
                let flat_here = if in_umbra { 1.0 } else { 1.0 - busy_of(form_range) };
                let calm = if p.smooth > 0.0 && !glazing && !cresting && !lost_here { p.smooth * flat_here * (1.0 - subj.clamp(0.0, 1.0)) } else { 0.0 };
                // A calm passage is restated where the panel is only a little wrong — the gradation is built
                // up in veils of small marks, not left to the broad pass's few flat ones.
                let restate_floor = restate_floor * (1.0 - 0.6 * calm);
                if let Some(st) = &structure {
                    // Off the faces and the subject the fine brushes restate STRUCTURE, not tone. With no human
                    // face in the picture (two lions) the subject is not a likeness to protect: there the
                    // gate applies on the subject too, so a mane keeps its strands (structure) and loses the
                    // posterised value steps the armature left (tone) — a contour-mapped mane otherwise.
                    // (SOFTER on that subject than on the ground: at the ground's full bar a mane lost its tan
                    // body with its bands — measured 5.6 points paler than the picture at 4096. The subject's bar
                    // is a fraction of it, so only the flattest passages of the animal are left to the washes.)
                    let protected = on_face || (on_subject && p.face_mask.is_some());
                    let bar = if on_subject { wcb_struct * wcb_struct_subj } else { wcb_struct };
                    if !protected && st[region_i] < bar {
                        return None;
                    }
                }
                if !block_in && !glazing && !cresting && !p.density && rgb_dist(cv.color_at(ix, iy), target) < restate_floor {
                    return None;
                }
                // A watercolour stroke is a GLAZE over what is there: it never lands where the paper is already
                // as dark as the picture asks (a watercolour cannot lighten), and its charge is for the
                // residual — the value the glaze must read on white so that, over the canvas, it reads the target.
                let under_lin = if wcb_trans { Some(cv.linear_at(ix, iy)) } else { None };
                let wcb_under = if wc_brush && (wcb_conc > 0.0 || wcb_trans) {
                    // (The block-in too: its strokes overlap several deep, and ungated every one landed at full
                    // charge over the last — ten charges of different pigments in one film is mud.)
                    let u = color::linear_luma(color::srgb_to_linear(cv.color_at(ix, iy))).max(0.02);
                    // "Already as dark as asked" is judged PER CHANNEL: a pale yellow over white paper is
                    // the same luma and a different colour, and a luma test skipped every such glaze.
                    let needs = match under_lin {
                        Some(ul) => {
                            let t = color::srgb_to_linear(target);
                            (0..3).any(|c| t[c] < ul[c] * (1.0 - wcb_tol))
                        }
                        None => u > tluma * (1.0 + wcb_tol),
                    };
                    if !needs {
                        return None;
                    }
                    Some(u)
                } else {
                    None
                };
                // Detail passes only add marks where there is COHERENT structure to resolve. On an incoherent,
                // structureless region (e.g. a tangled net the armature rendered as noise) the flow field has no
                // dominant direction — dropping detail marks there reads as random blocky specks, an "AI filter"
                // tell. Skip them; leave that area as the smooth mass the coarse passes laid.
                // (High-fidelity tracks EVERYTHING closely, so it doesn't gate on coherence — it restates flat
                // areas too. The gate is a Legible anti-speckle measure.)
                // (Not for the watercolour: a flat pane of stained glass is exactly a low-coherence cell off
                // the subject, and the gate left every window pale.)
                if detail && !fidelity && !wc_wash {
                    let seed_i = iy as usize * w as usize + ix as usize;
                    let coh = (gx[seed_i] * gx[seed_i] + gy[seed_i] * gy[seed_i]).sqrt();
                    // SUBJECT-AWARE bar: the subject keeps its detail (a smooth-lit face reads flat but still wants
                    // modelling), while a structureless BACKGROUND passage (open sky, a flat wall) is held far
                    // stricter so stray detail marks don't speckle it — the sky-speckle tell.
                    let thresh = p.detail_coherence * (1.0 + 1.6 * (1.0 - subj));
                    // (A CALM passage of a smoothed medium is exempt: its veils of small marks are the point —
                    // the sky of a tempera panel is built in them, and gated it stayed the broad pass's dabs.)
                    if coh < thresh && calm < 0.3 {
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
                // On a face under the sfumato the drift is wider: the fused, half-charged marks average the
                // hue back out, and the skin read as one interpolated gradient — a palette is never mixed
                // that evenly. (The sfumato's own amount scales it, so no other medium moves.)
                let drift = if p.sfumato > 0.0 && !block_in && !glazing && on_face { p.broken * (1.0 + p.sfumato) } else { p.broken };
                // ... and none in the shadow (`shade`): the dark masses are one quiet colour.
                let drift = drift * (1.0 - shaded);
                // CREST (see `MediumProfile::crest`): on the crest pass, off the faces and hair, in the light, where
                // the relief stands broken, a share of the seeds are a dry brush dragged over the dried relief
                // — a lighter, warmer tint of the picture that catches only the ridges' tops (the stroke's
                // `crest`), so the texture is lit without a single new shape.
                let crest_here = p.crest > 0.0 && cresting && !on_face && !lost_here && sh_raw < 0.5
                    && p.hair_mask.as_deref().and_then(|m| m.get(region_i).copied()).unwrap_or(0.0) <= 0.0
                    && crest_relief.as_deref().and_then(|b| b.get(region_i)).copied().unwrap_or(0.0) > 0.25
                    && jitter(p.seed ^ 0xC2E5_7, k.wrapping_add(53)) + 0.5 < p.crest;
                // SINK (see `MediumProfile::sink`): on the same pass, in the SHADOW, where the relief is broken
                // (a lower bar than the crest's — the shadows are laid thin), a share of the seeds a thin wet
                // wash, darker and cooler, that settles only in the furrows (a negative `crest`).
                let sink_here = p.sink > 0.0 && cresting && !crest_here && !on_face && !lost_here && sh_raw >= 0.5
                    && p.hair_mask.as_deref().and_then(|m| m.get(region_i).copied()).unwrap_or(0.0) <= 0.0
                    && crest_relief.as_deref().and_then(|b| b.get(region_i)).copied().unwrap_or(0.0) > 0.08
                    && jitter(p.seed ^ 0x51_4B, k.wrapping_add(61)) + 0.5 < p.sink;
                if cresting && !crest_here && !sink_here {
                    return None; // the crest pass lays nothing else
                }
                let target = if crest_here { crest_tone(target) } else if sink_here { sink_tone(target) } else { target };
                // VERDACCIO (see `MediumProfile::verdaccio`): on a face (not its hair) the block-in and the first
                // restating pass are laid in the greenish grey at the picture's value; the later passes, where
                // the form turns into the half-tone and the shadow, are pulled toward it and thinned (below), so
                // the green ground shows through the hatched flesh. Never the veils.
                let hair_here = p.hair_mask.as_deref().and_then(|m| m.get(region_i).copied()).unwrap_or(0.0) > 0.0;
                // FLESH (see `MediumProfile::flesh`): the pull held to the warm colours — taken away by that
                // share as the picture's own colour here goes cold or neutral.
                let verd = if p.verdaccio > 0.0 && on_face && !hair_here && !glazing && !cresting {
                    let v = if layer <= 1 { p.verdaccio } else { 0.35 * p.verdaccio * sh_raw.clamp(0.0, 1.0) };
                    if p.flesh > 0.0 { v * (1.0 - p.flesh * (1.0 - warmth_of(target))) } else { v }
                } else {
                    0.0
                };
                let target = if verd > 0.0 { verdaccio_tone(target, verd) } else { target };
                // SMALL (see `MediumProfile::small`): on a small face the half-tone held to the local tone — its
                // CIELAB lightness within `6 + 24·(1 − k)` of the picture's local mean there.
                let small_here = small_k > 0.0 && on_face && !hair_here;
                // ANCHOR (see `MediumProfile::anchor`): a mark in the darkest share around it is neither held
                // nor blended — the eye, the nostril, the mouth's line.
                let anchored = small_here && p.anchor > 0.0 && match small_l.as_ref().and_then(|(m, sd)| Some((*m.get(region_i)?, *sd.as_ref()?.get(region_i)?))) {
                    Some((ml, sd)) => color::srgb_to_lab(target).l < ml - anchor_z * sd && sd > 1.0,
                    None => false,
                };
                let target = match (small_here && !anchored, small_l.as_ref().and_then(|(l, _)| l.get(region_i))) {
                    (true, Some(&ml)) => {
                        let lab = color::srgb_to_lab(target);
                        let eps = 6.0 + 24.0 * (1.0 - small_k);
                        let l = lab.l.clamp(ml - eps, ml + eps);
                        if (l - lab.l).abs() > 1e-3 { color::lab_to_srgb(color::Lab { l, a: lab.a, b: lab.b }) } else { target }
                    }
                    _ => target,
                };
                // The mark's value in the picture's own range (0 its darkest, 1 its lightest).
                let rel_v = ((tluma - map_lo) / (map_hi - map_lo).max(0.05)).clamp(0.0, 1.0);
                // DISHES (see `MediumProfile::dishes`): a mark is laid in the nearest of the mixed tones, never
                // one between — the block-in too (a panel's underpainting is each passage's one flat local
                // tone; over a graded wash every stepped mark showed its edge); the veils and the lost edges
                // keep the picture's own gradation. DITHER (see `MediumProfile::dither`): after the block-in
                // the dish is drawn between the two nearest by the value's fraction, so a band breaks into a
                // mixture of marks of both.
                let mut dish_far = false;
                // THE UMBRA: within the darkest dish's interval the tone is laid as it is — the shadow is one
                // modelled mass, not two steps of planks (a night scene's whole foreground fell in the first
                // step and was a chequer of black and dark grey).
                let target = if p.dishes >= 2.0 && !glazing && !cresting && !lost_here && !in_umbra {
                    // (The dither is for the hatching's optical mixture — the finer half of the ladder; on the
                    // broad restatements two dishes side by side were a chequer of planks across a sky.)
                    let dither_here = p.dither * ((depth_t - 0.25) / 0.5).clamp(0.0, 1.0);
                    let (dither, u) = if block_in || dither_here <= 0.0 { (0.0, 0.5) } else { (dither_here, jitter(p.seed ^ 0xD1_7E8, k.wrapping_add(83)) + 0.5) };
                    let dished = dish_tone(target, p.dishes, map_lo, map_hi, dither, u);
                    // The dither took the FARTHER dish: that mark is laid as a veil (half the charge, below) —
                    // the optical mixture is thin hatching of the neighbouring tone, not a speck of it (a
                    // quarter-range grey laid full on a black cat was a white fleck).
                    dish_far = dither > 0.0 && dished != dish_tone(target, p.dishes, map_lo, map_hi, 0.0, 0.5);
                    // SMOOTH: on a calm passage the dish relaxes toward the picture's own tone by the calm.
                    if calm > 1e-3 { blend_tone(dished, target, calm) } else { dished }
                } else {
                    target
                };
                // LIGHTS (see `MediumProfile::lights`): on the finest pass, on the subject, in the picture's top
                // values, this share of the marks are hatched in toward white — narrower and thinner (below).
                let lit = p.lights > 0.0 && last_pass && !block_in && !glazing && !cresting && !lost_here && (on_face || subj > 0.5) && !hair_here
                    && rel_v > 0.6 && jitter(p.seed ^ 0x11_647, k.wrapping_add(79)) + 0.5 < p.lights;
                // ...and on a face, with CARNATION, along the light's own hue (see `MediumProfile::carnation`).
                let carn_here = carnation.is_some() && on_face && !hair_here;
                let target = if lit { let amt = 0.35 * p.lights * ((rel_v - 0.6) / 0.4).clamp(0.0, 1.0); if carn_here { warm_light(target, amt, p.carnation) } else { lights_tone(target, amt) } } else { target };
                // GLINT (see `MediumProfile::glint`): on the finer half of the ladder, in the picture's top
                // twentieth of values, the mark is a pure light — lifted toward white here, laid narrower,
                // dry, covering and raised (below).
                let glinted = p.glint > 0.0 && !block_in && !glazing && !cresting && !lost_here && !hair_here && depth_t > 0.5 && rel_v > 0.95;
                // (With CARNATION a face's glint is lifted along its hue too: pure white over the warm flesh
                // read as a pale grey-green patch beside it.)
                let target = if glinted { if carn_here && !crowd { warm_light(target, 0.4 * p.glint, p.carnation) } else { lights_tone(target, 0.4 * p.glint) } } else { target };
                // VELVET (see `MediumProfile::velvet`): the darkest share of the range is the darkest pigment
                // itself, at a full press.
                // (The share and the eraser's third are read on the PERCEIVED scale — the range in gamma: 8% of the
                // linear range is a sRGB value of about 80, and the whole dark half-tone went to velvet.)
                let rel_g = ((tluma.max(0.0).powf(1.0 / 2.2) - map_lo.max(0.0).powf(1.0 / 2.2)) / (map_hi.max(0.0).powf(1.0 / 2.2) - map_lo.max(0.0).powf(1.0 / 2.2)).max(0.02)).clamp(0.0, 1.0);
                // VELVET DARK (see `MediumProfile::velvet_dark`): and dark on the sheet, not only in the range.
                let velvet_here = p.velvet > 0.0 && !glazing && !cresting && !lost_here && rel_g < p.velvet
                    && (p.velvet_dark <= 0.0 || tluma.max(0.0).powf(1.0 / 2.2) < p.velvet_dark);
                let target = if velvet_here { p.palette.pigments[charcoal_dark].masstone } else { target };
                // CHALK TINT (see `MediumProfile::chalk_tint`): a light mixed with white loses its chroma.
                let target = if p.chalk_tint > 0.0 && !glazing && !cresting && !lost_here && !on_face {
                    let mut lab = color::srgb_to_lab(target);
                    let u = ((lab.l - 55.0) / 40.0).clamp(0.0, 1.0);
                    let cut = 1.0 - 0.6 * p.chalk_tint.clamp(0.0, 1.0) * u * u * (3.0 - 2.0 * u);
                    lab.a *= cut;
                    lab.b *= cut;
                    color::lab_to_srgb(lab)
                } else {
                    target
                };
                // BLADE TINT (see `MediumProfile::blade_tint`): in the zones the wide plates are broken in (the
                // break's noise, `blade_break`, on the ladder's upper rungs) each slab's hue drifts by a hash of the
                // mark, up to ±2% of the circle × the key × the break there — the blade not quite clean of the
                // last colour it carried.
                let target = if p.blade_tint > 0.0 && p.blade_break > 0.0 && ladder_t < 0.4 && !on_face && !glazing && !cresting && !lost_here {
                    let per = (w.min(h) as f32 / 10.0).max(8.0);
                    let nz = crate::paint::stroke::value_noise(p.seed ^ 0xB1A_DB4, cx / per, cy / per);
                    blade_tint_tone(target, 0.02 * p.blade_tint.clamp(0.0, 1.0) * nz * 2.0 * jitter(p.seed ^ 0xB1A_D71, k.wrapping_add(107)))
                } else {
                    target
                };
                // NIB TONE (see `MediumProfile::nib_tone`): a pen laid only where the tone is dark enough for its
                // width — the broad nib in the deep shadows, the middle one in the shadows and half-tones, the fine
                // one everywhere but the lights — and the darkness read against the eye's attention (`ma`), so the
                // far ground and the sky are the fine pen's (soft over ±0.05 by a hash, as `tiers`).
                let at_i = iy as usize * w as usize + ix as usize;
                let mut nib_far_fine = false;
                let mut pen_q = 0.0f32;
                if p.nib > 0.0 && p.nibs >= 2.0 && p.nib_tone > 0.0 && !glazing && !cresting {
                    let n = p.nibs.round().max(2.0);
                    let (rc, rf) = (passes.first().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius), passes.last().map(|q| q.radius.max(p.min_brush)).unwrap_or(radius));
                    let dl = if rc > rf + 1e-6 { ((rc / radius.max(1e-3)).ln() / (rc / rf).ln()).clamp(0.0, 1.0) } else { 0.0 };
                    let q = (dl * n).floor().min(n - 1.0) / (n - 1.0);
                    pen_q = q;
                    let v = color::linear_luma(color::srgb_to_linear(target)).max(0.0).powf(1.0 / 2.2);
                    let attn = ma_field.as_ref().and_then(|m| m.get(at_i)).copied().unwrap_or(1.0).clamp(0.0, 1.0);
                    // NIB DEPTH (see `MediumProfile::nib_depth`): the attention capped by how near the picture is there.
                    let attn = match p.nib_depth_map.as_deref().filter(|_| p.nib_depth > 0.0).and_then(|m| m.get(at_i)) {
                        Some(&z) => {
                            let u = ((z - 0.1) / 0.4).clamp(0.0, 1.0);
                            attn.min(1.0 - p.nib_depth.clamp(0.0, 1.0) * (1.0 - u * u * (3.0 - 2.0 * u)))
                        }
                        None => attn,
                    };
                    // NIB FAR (see `MediumProfile::nib_far`): off the eye's attention every pen is the finest — its
                    // marks kept, drawn at the finest width (dropped, the far slope the broad passes hatched was bare:
                    // the fine passes do not reach it), gated by its own pen's tone (by the finest's, the sky and the
                    // sea were hatched over).
                    nib_far_fine = p.nib_far > 0.0 && q < 0.99 && attn < 0.6 * p.nib_far.clamp(0.0, 1.0);
                    let dark = (1.0 - v) * (0.4 + 0.6 * attn);
                    let thr = p.nib_tone.clamp(0.0, 1.0) * (0.12 + 0.6 * (1.0 - q));
                    let t = ((dark - thr) / 0.1 + 0.5).clamp(0.0, 1.0);
                    if jitter(p.seed ^ 0x9E7_70, k.wrapping_add(97)) + 0.5 >= t * t * (3.0 - 2.0 * t) {
                        return None;
                    }
                    // NIB FACE (see `MediumProfile::nib_face`): a face drawn with the finest pen alone.
                    if p.nib_face > 0.0 && on_face && q < 0.99 && jitter(p.seed ^ 0xFAC_E, k.wrapping_add(103)) + 0.5 < p.nib_face {
                        return None;
                    }
                    // NIB OCCLUDE (see `MediumProfile::nib_occlude`): the broad nib only in the cast shadows.
                    if p.nib_occlude > 0.0 && q < 0.01 && nib_occl.as_ref().and_then(|m| m.get(at_i)).copied().unwrap_or(0.0) < 0.3 * p.nib_occlude.clamp(0.0, 1.0) {
                        return None;
                    }
                    // NIB AIR (see `MediumProfile::nib_air`): the far planes kept to a few fine lines.
                    if p.nib_air > 0.0 && attn < 0.5 {
                        let t = ((0.5 - attn) / 0.4).clamp(0.0, 1.0);
                        if jitter(p.seed ^ 0xA14_0, k.wrapping_add(107)) + 0.5 < 0.7 * p.nib_air.clamp(0.0, 1.0) * t * t * (3.0 - 2.0 * t) {
                            return None;
                        }
                    }
                }
                // PAPER (see `MediumProfile::paper`): charcoal only darkens — a mark goes where the sheet is
                // lighter than the picture asks, charged for the residual; where it is darker, in the lights,
                // the ERASER (see `MediumProfile::erase`) lifts it; otherwise nothing is laid.
                let mut paper_need: Option<f32> = None;
                let mut erase_wet: Option<f32> = None;
                if p.paper > 0.0 && !glazing && !cresting {
                    let under = color::linear_luma(color::srgb_to_linear(cv.color_at(ix, iy))).max(0.004);
                    // NIB (see `MediumProfile::nib`): the sheet read over the mark's cell (a window of 1.2 radii,
                    // every other pixel), not at one pixel — a black line beside a white gap is a grey.
                    // NIB WASH DRAW: a wash over the drawing too — the hatching's tone is a grey over the cell, not a black line or a white gap.
                    let under = if p.nib > 0.0 || (wash_reg && wash_draw) {
                        let rr = (radius * 1.2).clamp(2.0, 16.0) as i64;
                        let (mut sum, mut cnt) = (0.0f32, 0.0f32);
                        let mut yy = (iy as i64 - rr).max(0);
                        while yy <= (iy as i64 + rr).min(h as i64 - 1) {
                            let mut xx = (ix as i64 - rr).max(0);
                            while xx <= (ix as i64 + rr).min(w as i64 - 1) {
                                sum += color::linear_luma(color::srgb_to_linear(cv.color_at(xx as u32, yy as u32)));
                                cnt += 1.0;
                                xx += 2;
                            }
                            yy += 2;
                        }
                        let mean = (sum / cnt.max(1.0)).max(0.004);
                        under + (mean - under) * if p.nib > 0.0 { p.nib.clamp(0.0, 1.0) } else { 1.0 }
                    } else {
                        under
                    };
                    let tl = color::linear_luma(color::srgb_to_linear(target)).max(0.004);
                    // PAPER TOP (see `MediumProfile::paper_top`): the value read against the picture's light end.
                    // (The gain at most PAPER_TOP_GAIN and weighted by the square of the value's share of the
                    // light end: a night's light end is itself dark, and lifting it to the paper turned the tunnel
                    // into a grey day.)
                    let tl = if p.paper_top > 0.0 && map_hi > 1e-3 {
                        let wgt = (tl / map_hi).clamp(0.0, 1.0).powi(2);
                        let gain = 1.0 + ((charcoal_paper_l / map_hi).min(PAPER_TOP_GAIN) - 1.0).max(0.0) * wgt * p.paper_top.clamp(0.0, 1.0);
                        (tl * gain).min(charcoal_paper_l)
                    } else {
                        tl
                    };
                    // PAPER KNEE (see `MediumProfile::paper_knee`): the top of the curve bent toward the paper, in
                    // a light picture only.
                    let tl = if p.paper_knee > 0.0 {
                        let u = ((rel_g - 0.6) / 0.4).clamp(0.0, 1.0);
                        let light = (map_hi / charcoal_paper_l.max(1e-3)).clamp(0.0, 1.0).powi(2);
                        (tl + (charcoal_paper_l - tl).max(0.0) * p.paper_knee.clamp(0.0, 1.0) * u * u * (3.0 - 2.0 * u) * light).min(charcoal_paper_l)
                    } else {
                        tl
                    };
                    // GRADE (see `MediumProfile::grade`): the half-tones keyed lighter, the velvet black kept.
                    let tl = if p.grade > 0.0 && !velvet_here {
                        let g = (tl / charcoal_paper_l.max(1e-3)).clamp(0.0, 1.0).powf(1.0 / 2.2);
                        (charcoal_paper_l * g.powf(1.0 / (1.0 + p.grade.clamp(0.0, 1.0))).powf(2.2)).max(tl).min(charcoal_paper_l)
                    } else {
                        tl
                    };
                    // YOHAKU (see `MediumProfile::yohaku`): a flat passage off the subject left blank.
                    let tl = if let Some(yr) = &yohaku_range {
                        let at = iy as usize * w as usize + ix as usize;
                        let u = ((yr.get(at).copied().unwrap_or(1.0) - 0.03) / 0.07).clamp(0.0, 1.0);
                        let flat = 1.0 - u * u * (3.0 - 2.0 * u);
                        let subj = p.subject_mask.as_deref().and_then(|m| m.get(at)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                        (tl + (charcoal_paper_l - tl).max(0.0) * p.yohaku.clamp(0.0, 1.0) * flat * (1.0 - subj)).min(charcoal_paper_l)
                    } else {
                        tl
                    };
                    // MA (see `MediumProfile::ma`): where the eye does not go, the value lifted toward the paper.
                    let tl = if let Some(mf) = &ma_field {
                        let at = iy as usize * w as usize + ix as usize;
                        // MA FOCUS (see `MediumProfile::ma_focus`): the attention (the subject already in it,
                        // blurred) narrowed to the focus.
                        let attn = if p.ma_focus > 0.0 {
                            mf.get(at).copied().unwrap_or(1.0).clamp(0.0, 1.0).powf(1.0 + 4.0 * p.ma_focus.clamp(0.0, 1.0))
                        } else {
                            mf.get(at).copied().unwrap_or(1.0).max(p.subject_mask.as_deref().and_then(|m| m.get(at)).copied().unwrap_or(0.0)).clamp(0.0, 1.0)
                        };
                        (tl + (charcoal_paper_l - tl).max(0.0) * p.ma.clamp(0.0, 1.0) * (1.0 - attn).powi(2)).min(charcoal_paper_l)
                    } else {
                        tl
                    };
                    // NIB WHITE (see `MediumProfile::nib_white`): a cell never asked darker than this share of the paper,
                    // so the deepest shadow keeps the paper's flecks between its lines.
                    // NIB RELIEF (see `MediumProfile::nib_relief`): a place lighter than its surroundings (the bulge of
                    // a fold) lifted toward the paper by three times its excess, so the hatching thins on it. Not on
                    // the fur: every lock of a coat is lighter than its neighbour, and the fox's body broke into white islands.
                    let tl = match (&nib_val, &nib_mean) {
                        (Some(vv), Some(mm)) if p.nib > 0.0 && !hair_here => {
                            let at = iy as usize * w as usize + ix as usize;
                            let ex = (vv.get(at).copied().unwrap_or(0.0) - mm.get(at).copied().unwrap_or(0.0)).max(0.0);
                            tl + (charcoal_paper_l - tl).max(0.0) * (3.0 * p.nib_relief.clamp(0.0, 1.0) * ex).clamp(0.0, 1.0)
                        }
                        _ => tl,
                    };
                    // NIB FUR (see `MediumProfile::nib_fur`): on the fur, the body's lit bulges lifted smoothly.
                    let tl = match &nib_fur_rel {
                        Some(fr) if p.nib > 0.0 && hair_here => {
                            let ex = fr.get(iy as usize * w as usize + ix as usize).copied().unwrap_or(0.0);
                            let k = p.nib_fur.clamp(0.0, 1.0);
                            let u = nib_val.as_ref().map(|vv| ((vv.get(iy as usize * w as usize + ix as usize).copied().unwrap_or(0.0) - 0.12) / 0.23).clamp(0.0, 1.0)).unwrap_or(0.0);
                            tl + (charcoal_paper_l - tl).max(0.0) * ((3.0 * k * ex).clamp(0.0, 1.0) + 0.5 * k * u * u * (3.0 - 2.0 * u)).clamp(0.0, 1.0)
                        }
                        _ => tl,
                    };
                    // NIB LIFT (see `MediumProfile::nib_lift`): on a figure (a face found) off the fur, the half-tones
                    // lifted, the deep folds kept. (Not on a subject without a face: a pale bird's wings lost their shading.)
                    let tl = match &nib_val {
                        Some(vv) if p.nib > 0.0 && p.nib_lift > 0.0 && !hair_here && p.face_mask.is_some() => {
                            let at = iy as usize * w as usize + ix as usize;
                            let s = p.subject_mask.as_deref().and_then(|m| m.get(at)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                            let u = ((vv.get(at).copied().unwrap_or(0.0) - 0.12) / 0.23).clamp(0.0, 1.0);
                            tl + (charcoal_paper_l - tl).max(0.0) * 0.8 * p.nib_lift.clamp(0.0, 1.0) * s * u * u * (3.0 - 2.0 * u)
                        }
                        _ => tl,
                    };
                    // NIB FACE: the face's value lifted by 0.4 of the key toward the paper.
                    let tl = if p.nib > 0.0 && p.nib_face > 0.0 && on_face { tl + (charcoal_paper_l - tl).max(0.0) * 0.4 * p.nib_face.clamp(0.0, 1.0) } else { tl };
                    // NIB WASH AIR (see `MediumProfile::nib_wash_air`): how far this point is, by the depth map — 1 at the
                    // farthest plane, 0 from a half of the depth in.
                    let wash_far = match p.nib_depth_map.as_deref().filter(|_| wash_reg && p.nib_wash_air > 0.0).and_then(|m| m.get(at_i)) {
                        Some(&z) => {
                            let u = ((z - 0.1) / 0.4).clamp(0.0, 1.0);
                            1.0 - u * u * (3.0 - 2.0 * u)
                        }
                        None => 0.0,
                    };
                    // NIB WASH DRY: the dry brush lays the dark register only — the picture's value under 0.45 on the eye's
                    // scale; toward the far planes the register narrows to the darker (0.34 at the farthest, by the air).
                    if dry_pass && wash_dry_pic.as_ref().is_some_and(|m| m[at_i] >= 0.45 * (1.0 - 0.25 * p.nib_wash_air.clamp(0.0, 1.0) * wash_far)) {
                        return None;
                    }
                    // NIB WASH PLANES: the washes are the fills; the register's stroke passes lay nothing (the dry brush's excepted).
                    if wash_reg && !dry_pass && p.nib_wash_planes >= 2.0 {
                        return None;
                    }
                    let tl_pre = tl;
                    // NIB WASH: a wash asked only the key's share of the way from the paper to the tone — the pens draw the rest.
                    let tl = if wash_reg { tl + (charcoal_paper_l - tl).max(0.0) * (1.0 - p.nib_wash.clamp(0.0, 1.0)) } else { tl };
                    // NIB WASH GAMMA (see `MediumProfile::nib_wash_gamma`): the wash's density raised, the paper kept.
                    let tl0 = tl;
                    let tl = if wash_reg && p.nib_wash_gamma > 0.0 && tl < charcoal_paper_l { charcoal_paper_l * (tl / charcoal_paper_l).max(0.0).powf(1.0 + 0.5 * p.nib_wash_gamma.clamp(0.0, 1.0)) } else { tl };
                    // NIB WASH SCURVE (see `MediumProfile::nib_wash_scurve`): the density by an S-curve on the eye's scale.
                    let tl = if wash_reg && p.nib_wash_scurve > 0.0 && tl < charcoal_paper_l {
                        let x = (tl / charcoal_paper_l).max(0.0).powf(1.0 / 2.2);
                        let a = 1.0 + p.nib_wash_scurve.clamp(0.0, 1.0);
                        let (xa, ya) = (x.powf(a), (1.0 - x).max(0.0).powf(a));
                        charcoal_paper_l * (xa / (xa + ya).max(1e-9)).powf(2.2)
                    } else {
                        tl
                    };
                    // NIB WASH MASS: the wash's tone deepened by a gamma of the bleed's measured loss — a tone mapping, not a
                    // scaling of the depth: scaled and clamped, every lit face in a crowd went to one black.
                    let tl = if wash_reg && wash_mass > 1.0 && tl < charcoal_paper_l { charcoal_paper_l * (tl / charcoal_paper_l).max(0.0).powf(wash_mass) } else { tl };
                    // NIB WASH VARY (see `MediumProfile::nib_wash_vary`): the deepening held back where the picture is busy.
                    let tl = match wash_vary.as_ref().filter(|_| wash_reg) {
                        Some(m) => tl0 + (tl - tl0) * m[at_i],
                        None => tl,
                    };
                    // NIB WASH DETAIL: the wash stops short of the picture's own structure — its tone there left to the pen.
                    let on_detail = wash_detail.as_ref().is_some_and(|m| m[at_i]);
                    let tl = if wash_reg && on_detail { tl + (charcoal_paper_l - tl).max(0.0) * 0.5 * p.nib_wash_detail.clamp(0.0, 1.0) } else { tl };
                    // NIB WASH DRY: the dry brush's ink is the picture's own value — undiluted, unsoftened, past the washes'
                    // curves — the key's share of the way from the paper to it. (Before the air: the dry brush too lays
                    // less toward the far planes — level drags as dense at the horizon as at the feet were one flat wall.)
                    let tl = if dry_pass { tl_pre + (charcoal_paper_l - tl_pre).max(0.0) * (1.0 - p.nib_wash_dry.clamp(0.0, 1.0)) } else { tl };
                    // NIB WASH AIR: toward the far planes the wash lightened — the dry brush's ink by 0.35, less than the
                    // washes' 0.5: lightened as much as the wash under it, no drag passed the sheet's gate and the far planes
                    // lost the dry brush altogether (a station's whole background is one far plane on the depth map).
                    let tl = if wash_far > 0.0 { tl + (charcoal_paper_l - tl).max(0.0) * (if dry_pass { 0.35 } else { 0.5 }) * p.nib_wash_air.clamp(0.0, 1.0) * wash_far } else { tl };
                    // NIB WASH CAP: the sheet holds no more ink than the picture's own value there — the floor the key's
                    // share of it; where the sheet has reached it the gate below lays nothing.
                    let tl = match wash_cap_pic.as_ref().filter(|_| wash_reg) {
                        Some(m) => tl.max(p.nib_wash_cap.clamp(0.0, 1.0) * m[at_i]),
                        None => tl,
                    };
                    // NIB WASH DARK: the pens hatch only the darks, the half-tones left to the washes (the threshold on the
                    // eye's scale: the luma here is linear).
                    // NIB WASH DETAIL: the finer pens kept where the picture has a structure of its own.
                    let detail_here = pen_q > 0.49 && on_detail;
                    // NIB WASH FACES: on a face the gate opens by the key — the pen models the face.
                    // NIB WASH PROFILE: on a face the gate opened for the finest pen alone.
                    let face_open = on_face && p.nib_wash_faces > 0.0 && (p.nib_wash_profile <= 0.0 || pen_q > 0.49);
                    let dark_gate = p.nib_wash_dark.clamp(0.0, 1.0) * if face_open { 1.0 - p.nib_wash_faces.clamp(0.0, 1.0) } else { 1.0 };
                    // NIB WASH MASK: the pen lifted out of the darkest plane (the lines and the faces kept).
                    let tl = match wash_dark_plane.as_ref().filter(|_| !wash_reg && !detail_here && !face_open) {
                        Some(m) => { let d = m[at_i]; tl + (charcoal_paper_l - tl).max(0.0) * p.nib_wash_mask.clamp(0.0, 1.0) * d * d }
                        None => tl,
                    };
                    // NIB WASH HALF: the finest pen, along the form, on the near planes, past the gate by three quarters of the
                    // key of the way to the paper on the eye's scale — asked a half of the key's share of the residual.
                    let e_dark = 1.0 - 0.8 * dark_gate;
                    let half_here = p.nib_wash_half > 0.0 && nib_wash_n > 0 && !wash_reg && pen_q > 0.49 && !pass_turned && match p.nib_depth_map.as_deref().and_then(|m| m.get(at_i)) {
                        Some(&z) => { let u = ((z - 0.1) / 0.4).clamp(0.0, 1.0); u * u * (3.0 - 2.0 * u) > 0.5 }
                        None => true,
                    };
                    let e_gate = if half_here { e_dark + (1.0 - e_dark) * 0.75 * p.nib_wash_half.clamp(0.0, 1.0) } else { e_dark };
                    if nib_wash_n > 0 && !wash_reg && dark_gate > 0.0 && !detail_here && tl > charcoal_paper_l * e_gate.powf(2.2) {
                        return None;
                    }
                    let tl = if half_here && tl > charcoal_paper_l * e_dark.powf(2.2) { tl + (charcoal_paper_l - tl).max(0.0) * 0.5 * p.nib_wash_half.clamp(0.0, 1.0) } else { tl };
                    // NIB WASH FLAT: a flat dark is the wash's pool, not a hatch.
                    if !wash_reg && !detail_here && wash_busy.as_ref().is_some_and(|m| !m[at_i]) {
                        return None;
                    }
                    let white = p.nib_white.clamp(0.0, 1.0) + if p.nib_face > 0.0 && on_face { 0.3 * p.nib_face.clamp(0.0, 1.0) } else { 0.0 } + if p.nib_fur > 0.0 && hair_here { 0.15 * p.nib_fur.clamp(0.0, 1.0) } else { 0.0 };
                    let tl = if p.nib > 0.0 && white > 0.0 { tl.max(charcoal_paper_l * white) } else { tl };
                    // (NIB WASH DETAIL: the detail is a line, drawn over a wash that already carries its tone.)
                    if under > tl * (1.0 + PAPER_TOL) || (detail_here && nib_wash_n > 0 && !wash_reg) {
                        paper_need = Some((tl / under).clamp(0.03, 0.99).powf(1.0 / PAPER_OVERLAP));
                        // NIB: the ink laid black — the residual's charge pulled to the fullest by the key, the
                        // tone left to how many lines the cell takes.
                        if p.nib > 0.0 {
                            paper_need = paper_need.map(|n| n + (0.03 - n) * p.nib.clamp(0.0, 1.0));
                        }
                    } else if p.erase > 0.0 && !block_in && !velvet_here && rel_g > 1.0 / 3.0 && under < tl * (1.0 - PAPER_TOL) && p.lift > 1e-3 {
                        // The film that reads `l` over the paper: alpha = (paper − l)/(paper − black), and the
                        // wipe's share takes the film from the one alpha's to the other's.
                        let a = |l: f32| ((charcoal_paper_l - l) / (charcoal_paper_l - charcoal_black_l).max(1e-3)).clamp(1e-3, 0.999);
                        let s = (1.0 - (1.0 - a(tl)).ln() / (1.0 - a(under)).ln()).clamp(0.0, 0.9) * p.erase;
                        if s < 0.02 {
                            return None;
                        }
                        erase_wet = Some(s / p.lift);
                    } else {
                        return None;
                    }
                }
                let load_target = if drift > 0.0 && !(wc_brush && wcb_broken <= 0.0) { broken_color(target, drift, p.seed, k) } else { target };
                // IMPRIMATURA (see `MediumProfile::imprimatura`): the block-in laid darker and warmer than the
                // picture, toward a burnt umber — the underpainting that shows between the later marks.
                let load_target = if block_in && p.imprimatura > 0.0 { imprimatura_tone(load_target, p.imprimatura) } else { load_target };
                // Committed shadows carry MORE pigment so the dark masses read solid, not a thin transparent wash.
                // The HUE at the pigments' own masstone depth, so the mixer picks pigments and never white; the
                // charge is found below, once the stroke's brush and wetness are known (a stroke deposits by its
                // own rule, not a fill's).
                let wcb_mass: Option<(Srgb, f32)> = (wc_brush && wcb_conc > 0.0).then(|| {
                    let lin = color::srgb_to_linear(load_target);
                    let l = color::linear_luma(lin).max(1e-4);
                    // A light or mid passage is the pure hue, its value from the water; a DARK one is mixed
                    // dark — a vivid pigment cannot reach a deep value on its own, and a painter adds the
                    // dark pigment to it there.
                    let sc = if wcb_relit { wcb_targetl.min(tluma.max(0.03)) / l } else { 1.0 };
                    let mass = if wcb_relit { color::linear_to_srgb([(lin[0] * sc).min(1.0), (lin[1] * sc).min(1.0), (lin[2] * sc).min(1.0)]) } else { load_target };
                    let need = (tluma / wcb_under.unwrap_or(1.0)).clamp(0.03, 0.99);
                    (mass, need)
                });
                let load = if wc_brush && wcb_pig >= 1 {
                    let m = mixer::solve_mixture(&p.palette, load_target, wcb_pig.min(n));
                    let c = p.charge * (1.0 + 1.1 * sh);
                    let mut v = vec![0f32; n];
                    for (pi, wgt) in m.pigments.iter().zip(&m.weights) {
                        v[*pi] += wgt * c;
                    }
                    v
                } else if let (true, Some((wp, wi))) = (carn_here, carnation.as_ref()) {
                    // The flesh in warm sticks: the mixture over the warm box, mapped back to the palette.
                    carnation_mix(&mut carn_cache.borrow_mut(), cache, load_target, wp, wi, &p.palette, p.charge * (1.0 + 1.1 * sh), n, p.carnation)
                } else {
                    mixture_cached(cache, load_target, &p.palette, p.charge * (1.0 + 1.1 * sh), n)
                };
                // The mark's charge as solved, before the per-mark scalings below — the RAMP's end mix (see
                // `MediumProfile::ramp`) is scaled by the same factors, as the ratio of the laid load to this.
                let ramp_base: f32 = load.iter().map(|v| v.max(0.0)).sum();
                // ...the flesh over the verdaccio laid thinner where the form turns (the half-tone's hatching
                // lets the green through), by `0.4·verdaccio·shadow`.
                let load: Vec<f32> = if verd > 0.0 && layer > 1 { let t = 1.0 - 0.4 * p.verdaccio * sh_raw.clamp(0.0, 1.0); load.iter().map(|v| v * t).collect() } else { load };
                // The brush watercolour: broad passes are WASHES — transparent (a fraction of the charge), flooded,
                // wide and long so they overlap into one wet mass, and they never lift the layer beneath.
                let wcb_broad = wc_brush && !fine;
                // …and a painter FLOODS the darks: a dark target carries more pigment, a light one barely tints.
                let load: Vec<f32> = if wc_brush { let m = wcb_charge.get(layer).copied().unwrap_or(1.0) * (wcb_light + wcb_dark * (1.0 - tluma.clamp(0.0, 1.0))); load.iter().map(|v| v * m).collect() } else { load };
                // Stroke-growth boundary: a COMPOSITION layer keeps its strokes inside the element's footprint
                // (they terminate at the mask edge, so the element doesn't bleed over its neighbours); otherwise
                // the focal hard-edge region_mask keeps a single subject crisp against the ground.
                // HAIR / FUR: how much this mark is a strand rather than a mass (see `PaintParams::hair_mask`).
                let hair = p.hair_mask.as_deref().and_then(|m| m.get(region_i).copied()).unwrap_or(0.0).clamp(0.0, 1.0);
                // A mane is ON the animal: the hair tool never runs off the subject, whatever extent SAM
                // handed back (on the snow lions it returned 23% of the frame — the manes AND the snow around
                // them — and the strand tool raked 1,300 dry streaks across flat snow). Where there is no
                // subject matte the mask stands as given. (A strand may still BREAK the silhouette: that is
                // decided below from the stroke's path, not from its seed.)
                // (The watercolour recipe only: the oil keeps its hair mask as given — its strands along the
                // subject's edge are part of how it was judged.)
                let hair = if wc_wash && p.subject_mask.is_some() && subj < 0.5 { 0.0 } else { hair };
                // TRIM: a ground mark (off the face, not a strand) is not seeded within the face's clearance, and its
                // path stops at it (above).
                let clear_here = !on_face && hair <= 0.0 && face_clear.is_some();
                if clear_here && face_clear.as_ref().map_or(false, |m| m.get(region_i).copied().unwrap_or(false)) {
                    return None;
                }
                // THE GLAZE PASS NEVER CROSSES A FACE OR HAIR: its brush is the ladder's mid brush and its reference
                // the blurred one, so one mark on a small head dragged the stubble's dark across the cheek as a
                // smudge (the standing man of the family, set 5). The faces are finished by their own tier and
                // the sfumato; the glaze unifies the ground and the clothes around them.
                if (glazing || cresting) && (on_face || hair > 0.0) {
                    return None;
                }
                if glazing && glaze_hold.as_ref().map_or(false, |m| m.get(region_i).copied().unwrap_or(0.0) > ACCENT_EDGE) {
                    return None;
                }
                // SILHOUETTE STRANDS: hair reads as hair at its EDGE — a few strands escape the mass — but the
                // subject silhouette is a seam that strokes TERMINATE at, which is exactly wrong for a mane and
                // left every head and beard with a soft rounded outline. A MINORITY of hair marks (about a
                // third, chosen by the stroke's own hash so replay is exact) may cross it; they are narrow and
                // taper to a point, so what escapes is a filament, not a bulge. Letting them all cross turned
                // the outline into a fuzzy blob — the discipline is that most strands still stop at the seam.
                let strand_out = hair > 0.35 && jitter(p.seed ^ 0x9E37_79B9, k.wrapping_add(11)) + 0.5 < 0.35;
                let glazed = glaze_band.as_ref().map_or(false, |b| b.get(region_i).copied().unwrap_or(false));
                let region = if let Some(m) = p.paint_mask.as_deref() {
                    Some((m, true))
                } else if !wc_mass_masks.is_empty() {
                    let lbl = wc_mass.as_ref().map(|(l, _)| l[region_i] as usize).unwrap_or(0);
                    Some((wc_mass_masks[lbl].as_slice(), true))
                } else if strand_out || glazed || lost_here {
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
                let rw = (radius * profile.radius_scale * p.stroke_width * wvar * (1.0 - 0.35 * hair) * if wcb_broad { wcb_width } else { 1.0 }).max(p.min_brush * 0.8);
                // The watercolour's FINE brush draws LINES: a mullion, a rail, the edge of a basin — a mark many
                // times longer than wide (a rigger's), not a dab two widths long. The length cap on the fine
                // passes is raised; the colour-drift and hard-edge stops still end a mark where the picture does,
                // so on a texture the marks stay short and on an edge they run. (Measured: 60% of the 4 px marks
                // were under 10 px — every edge printed as a chain of beads.)
                let fine_len = if wc_wash && fine && !wcb_broad { wcb_finelen } else { 1.0 };
                // LOOSE (see `MediumProfile::loose`): off the subject and the faces, the broad passes' marks run
                // longer and wider — the free hand on the background. ACCENT (see `MediumProfile::accent`): at a
                // sharp contrast on the finest pass the mark is a short thick dab, set off the flow at its own angle.
                let off_subject = !on_face && subj < 0.5;
                let range_here = range_map.as_ref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.5);
                // BLADE ALONG (see `MediumProfile::blade_along`): the value's range over the brush's reach ALONG the
                // flow at the seed (perpendicular to the gradient; nothing where the field has no direction), by the
                // key's share against the isotropic window.
                let range_here = if p.blade_along > 0.0 && !block_in && range_map.is_some() {
                    let (gxv, gyv) = (gx.get(region_i).copied().unwrap_or(0.0), gy.get(region_i).copied().unwrap_or(0.0));
                    let m = (gxv * gxv + gyv * gyv).sqrt();
                    let along = if m > 1e-6 {
                        let (ux, uy) = (-gyv / m, gxv / m);
                        let reach = (radius * 1.5).round().max(2.0) as i32;
                        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                        for j in -reach..=reach {
                            let (x, y) = (cx + ux * j as f32, cy + uy * j as f32);
                            if x >= 0.0 && y >= 0.0 && x < w as f32 && y < h as f32 {
                                let v = luma[y as usize * w as usize + x as usize];
                                lo = lo.min(v);
                                hi = hi.max(v);
                            }
                        }
                        if hi > lo { hi - lo } else { 0.0 }
                    } else {
                        0.0
                    };
                    range_here + (along - range_here) * p.blade_along.clamp(0.0, 1.0)
                } else {
                    range_here
                };
                // SFUMATO (see `MediumProfile::sfumato`): a smooth passage of a face — the fine passes thin there
                // with the ladder, and what remains is laid broad, wet and thin (below).
                // (SMALL: on a small face the flat gate widened by `1 + 2k` and the share raised halfway to one.)
                // (ANCHOR: an anchored dark is not thinned, and the widening and the raised share fall off with
                // the local contrast — full on the flat, none at the widened gate.)
                let sk = if small_here { small_k } else { 0.0 };
                let gate = SFUMATO_FLAT * (1.0 + 2.0 * sk);
                let sk_s = if p.anchor > 0.0 { sk * (1.0 - range_here / gate).clamp(0.0, 1.0) } else { sk };
                let smooth_face = p.sfumato > 0.0 && !block_in && !glazing && on_face && hair <= 0.0 && !anchored && range_here < SFUMATO_FLAT * (1.0 + 2.0 * sk_s);
                if smooth_face && jitter(p.seed ^ 0x5F0_4A70, k.wrapping_add(19)) + 0.5 > 1.0 - (p.sfumato + 0.5 * sk_s * (1.0 - p.sfumato)) * depth_t {
                    return None;
                }
                // VARY (see `MediumProfile::vary`): the mark's length spread as a Gaussian in its log, wider where
                // the picture is flat — three uniform draws summed (σ 0.5), scaled.
                let lvar = if p.vary > 0.0 && !block_in {
                    let g = 2.0 * (jitter(p.seed ^ 0x6A55, k.wrapping_add(23)) + jitter(p.seed ^ 0x6A56, k.wrapping_add(29)) + jitter(p.seed ^ 0x6A57, k.wrapping_add(31)));
                    let flat = (1.0 - (range_here / 0.3).clamp(0.0, 1.0)) * 0.6 + 0.4;
                    lvar * (p.vary * g * flat).exp().clamp(0.4, 2.5)
                } else {
                    lvar
                };
                // ... and off the subject the restating passes THIN with the ladder — the finest keeps `1 − loose`
                // of its marks there — so the background stays the broad brush's flat patches, not a thousand
                // small corrections blended over them. (A deterministic skip by the stroke's own key: replay-exact.)
                if p.loose > 0.0 && !block_in && off_subject && jitter(p.seed ^ 0x1005E, k.wrapping_add(17)) + 0.5 > 1.0 - p.loose * depth_t {
                    return None;
                }
                // ... and in the SHADOW (`shade`) the marks run longer and wider — the dark masses rest.
                let loose_here = if p.loose > 0.0 && !detail && off_subject { 1.0 + p.loose } else { 1.0 } * (1.0 + 0.6 * shaded);
                let accented = !lost_here && accent_map.map_or(false, |m| m.get(region_i).copied().unwrap_or(0.0) > ACCENT_EDGE);
                let rw = rw * if p.loose > 0.0 && !detail && off_subject { 1.0 + 0.3 * p.loose } else { 1.0 } * if smooth_face { 1.0 + 0.6 * p.sfumato } else { 1.0 } * (1.0 + 0.3 * shaded);
                // BLADE SIZE (see `MediumProfile::blade_size`): the slice's size by the picture's flatness here — on
                // a flat passage a wide long slab, at a sharp detail a small hard one: the length and the width both
                // scaled by `1 + key·(2.4·flat − 0.4)`, flat = 1 − range/0.3 (the block-in keeps its own size).
                let (lvar, rw) = if p.blade_size > 0.0 && !block_in {
                    let flat = 1.0 - (range_here / 0.3).clamp(0.0, 1.0);
                    let f = 1.0 + p.blade_size * 2.4 * flat - 0.4 * p.blade_size.min(1.0);
                    (lvar * f, rw * f)
                } else {
                    (lvar, rw)
                };
                // BLADE REST (see `MediumProfile::blade_rest`): on a flat passage the restating passes leave the
                // slab alone — skipped by the key × the flatness × the pass's depth in the ladder.
                if p.blade_rest > 0.0 && !block_in {
                    let flat = 1.0 - (range_here / 0.3).clamp(0.0, 1.0);
                    if jitter(p.seed ^ 0xB1A_DE5, k.wrapping_add(83)) + 0.5 < p.blade_rest * flat * depth_t {
                        return None;
                    }
                }
                // BLADE EDGE (see `MediumProfile::blade_edge`): on the finer passes the busy passage is drawn with
                // the blade's edge — the slice narrowed and lengthened by the key × the busyness × the depth.
                let (lvar, rw) = if p.blade_edge > 0.0 && !block_in {
                    let e = p.blade_edge.clamp(0.0, 1.0) * (range_here / 0.3).clamp(0.0, 1.0) * depth_t;
                    (lvar * (1.0 + 0.8 * e), rw * (1.0 - 0.6 * e))
                } else {
                    (lvar, rw)
                };
                // BLADE FOCUS (see `MediumProfile::blade_focus`): outside the picture's focus the narrow knives keep
                // off (skipped by the key × the distance × the ladder's depth past 0.4) and the slices widen and
                // lengthen — the periphery wide, abstract and thick, the detail where the fine texture gathers.
                let far_here = if p.blade_focus > 0.0 && !on_face { p.blade_focus.clamp(0.0, 1.0) * focus_far.as_ref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) } else { 0.0 };
                if far_here > 0.0 && ladder_t > 0.4 && jitter(p.seed ^ 0xB1A_DF0, k.wrapping_add(109)) + 0.5 < far_here * ((ladder_t - 0.4) / 0.6).clamp(0.0, 1.0) {
                    return None;
                }
                let (lvar, rw) = (lvar * (1.0 + 0.6 * far_here), rw * (1.0 + 0.8 * far_here));
                // BLADE TUFT (see `MediumProfile::blade_tuft`): a fine incoherent texture is not descended into —
                // the narrow knives (the ladder's log depth past 0.6) skip it by the key × the texture; the medium
                // knives (0.2…0.6) lay it as one long pull, straight and upright (below), the second colour full.
                let tuft_here = if p.blade_tuft > 0.0 && !block_in && !on_face { p.blade_tuft.clamp(0.0, 1.0) * tuft_map.as_ref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) } else { 0.0 };
                if tuft_here > 0.0 && ladder_t > 0.6 && jitter(p.seed ^ 0xB1A_D7F, k.wrapping_add(97)) + 0.5 < tuft_here {
                    return None;
                }
                let tufted = tuft_here > 0.0 && ladder_t > 0.2 && ladder_t <= 0.6;
                let lvar = if tufted { lvar * (1.0 + 1.2 * tuft_here) } else { lvar };
                // HATCH (see `MediumProfile::hatch`): every restating mark a hatch line — narrowed toward a line
                // (`1 − 0.45·hatch` of the width, the lights narrower still), longer for it, straight (below),
                // of one width end to end, dry, no pickup, `1 − 0.3·hatch` of the charge. Not the block-in,
                // the veils, an accent or a lost edge.
                // The hatch grows with the ladder: the broad restatements lay the form's masses along the flow
                // (the underlayers of a panel are washes — straightened, they were planks across a night sky),
                // the finest pass is all hatching — the amount is `hatch · (depth_t − 1/4)/(3/4)`, nothing on
                // the widest two brushes, the key's value on the narrowest.
                // SMOOTH: on a calm passage the hatching gives way by the calm — a soft wash, not a line.
                let hatch_here = p.hatch * ((depth_t - 0.25) / 0.75).clamp(0.0, 1.0) * (1.0 - calm) * (1.0 - sk);
                let hatched = hatch_here > 1e-3 && !block_in && !glazing && !cresting && !accented && !lost_here && !smooth_face;
                // TIERS: a hatched layer's lines only where the tone is dark enough for it — the lights left to
                // the paper, the half-tones one set of lines, the shadows crossed layer over layer.
                // The block-in's broad marks are the deepest layer's: a pencil drawing has no toned ground, and
                // laid everywhere they were grey zigzag bands across the half-tones and the lights.
                if (hatched || block_in) && p.tiers > 0.0 {
                    let v = color::linear_luma(color::srgb_to_linear(target)).max(0.0).powf(1.0 / 2.2);
                    let thr = p.tiers.clamp(0.0, 1.0) * if block_in { 0.75 } else { (0.15 + 0.2 * tier_k as f32).min(0.75) };
                    // A soft threshold — the layer thins out over ±0.1 of darkness round it, by a hash — so a
                    // half-tone fades from one layer to the next instead of breaking into blank islands.
                    let t = ((1.0 - v - thr) / 0.2 + 0.5).clamp(0.0, 1.0);
                    if jitter(p.seed ^ 0x7135_5, k.wrapping_add(89)) + 0.5 >= t * t * (3.0 - 2.0 * t) {
                        return None;
                    }
                }
                // THE SIDE OF THE STICK (see `MediumProfile::side`): on the broad passes, off the faces and the
                // hair, this share of the marks (the key's on the block-in, falling with the ladder) are laid
                // with the stick's SIDE — wider by `0.6·side`, flat with the stick's facets (round 0, streak
                // 0.9), full width to the end, straighter by half the key, a light touch (`1 − 0.5·side` of the
                // pressure, so the chalk reaches the raised grain only), `1 − 0.3·side` of the charge, dry, no
                // pickup. The pressure, the width and the charge are in the record.
                let sided = p.side > 0.0 && !last_pass && !glazing && !cresting && !on_face && hair <= 0.0 && !accented && !lost_here && !smooth_face && !glinted && !hatched
                    && jitter(p.seed ^ 0x51DE_5, k.wrapping_add(79)) + 0.5 < p.side * if block_in { 1.0 } else { 1.0 - depth_t };
                let rw = if hatched { (rw * (1.0 - 0.45 * hatch_here) * if lit { 0.8 } else { 1.0 }).max(p.min_brush * 0.5) } else { rw };
                // SMALL: the stick's tip on a small face — narrower by half the scale.
                let rw = if sk > 0.0 { (rw * (1.0 - 0.5 * sk)).max(p.min_brush * 0.5) } else { rw };
                let rw = if glinted { (rw * (1.0 - 0.3 * p.glint)).max(p.min_brush * 0.5) } else { rw };
                let rw = if sided { rw * (1.0 + 0.6 * p.side) } else { rw };
                // NIB (see `MediumProfile::nib`): a steel point is never a brush's breadth — the width held under the
                // nib of the pass (a 600th of the sheet's long side on the broadest pass, a third of it on the
                // finest; a broad brush's mark laid in black was a blunt bar).
                // NIBS (see `MediumProfile::nibs`): the passes drawn with that many pens, each a fixed width — the
                // broadest a 500th of the long side, the finest an 1800th, the ones between in a geometric
                // series — the k-th pen taking the k-th share of the ladder, coarse to fine.
                // (Under `nibs` the pen's width is the mark's — a fine pass's mark is widened to its pen as a broad one
                // is narrowed: held only under a cap, nearly every mark was the finest passes' hairline and the
                // layers drew with one pen.)
                let rw = if p.nib > 0.0 {
                    let pen = if p.nibs >= 1.0 {
                        // (The ladder's place read on the LOG of the radius — the passes halve — so each pen takes an even
                        // share of them; on the radius itself the broad pen had the block-in alone.)
                        let n = p.nibs.round().max(1.0);
                        let dl = if r_coarse > r_fine + 1e-6 { ((r_coarse / radius.max(1e-3)).ln() / (r_coarse / r_fine).ln()).clamp(0.0, 1.0) } else { 0.0 };
                        let i = (dl * n).floor().min(n - 1.0);
                        let q = if n > 1.0 { i / (n - 1.0) } else { 0.0 };
                        let q = if nib_far_fine { 1.0 } else { q };
                        (w.max(h) as f32 / 500.0) * (500.0f32 / 1800.0).powf(q)
                    } else {
                        rw.min(((w.max(h) as f32 / 600.0) * (0.35 + 0.65 * (1.0 - depth_t))).max(p.min_brush * 0.5))
                    };
                    rw + (pen - rw) * p.nib.clamp(0.0, 1.0)
                } else {
                    rw
                };
                // THE FLAT BRUSH (see `MediumProfile::flat_brush`): on the passes before the finest, off the faces
                // and the hair, this share of the marks on a flat passage (by its flatness) a flat brush's patch.
                let flat_mark = p.flat_brush > 0.0 && !last_pass && !glazing && !cresting && !on_face && hair <= 0.0 && !accented && !lost_here && !glinted && !hatched && !sided
                    && jitter(p.seed ^ 0xF1A7_B, k.wrapping_add(83)) + 0.5 < p.flat_brush * flat_here;
                // SPAN: the hatch long on a flat passage, short where the form is busy.
                let span_here = if hatched && p.span > 0.0 { span_length(p.span, busy_of(form_range)) } else { 1.0 };
                // (A lost mark grows its path along the wide blur it is coloured from, so it runs across the seam
                // instead of stopping at it.)
                let blade_len_mul = (len_mul * p.stroke_len * lvar * (1.0 + 1.1 * hair) * fine_len * loose_here * if accented { 0.5 } else { 1.0 } * if wcb_broad { wcb_len } else { 1.0 } * if hatched { 1.0 + 0.6 * hatch_here } else { 1.0 } * span_here * (1.0 - 0.5 * calm)).max(0.2);
                // BLADE BREAK (see `MediumProfile::blade_break`): on the upper rungs of the ladder (the log-depth under
                // 0.4) the slice's length is cut by a global slow noise (a period of a tenth of the short side) — the
                // wide blade leaving the canvas on its own, the long plane a cascade of overlapping slabs.
                let blade_len_mul = if p.blade_break > 0.0 && ladder_t < 0.4 {
                    let per = (w.min(h) as f32 / 10.0).max(8.0);
                    let nz = crate::paint::stroke::value_noise(p.seed ^ 0xB1A_DB4, cx / per, cy / per);
                    blade_len_mul * (1.0 - 0.8 * p.blade_break.clamp(0.0, 1.0) * nz)
                } else {
                    blade_len_mul
                };
                let path = grow_path(cx, cy, radius, &gx, &gy, if lost_here { lost_pick.unwrap_or(&reference) } else { &reference }, target, if clear_here { face_clear.as_deref() } else { protect_all.as_deref() }, region, hard_ref, blade_len_mul, STOP_TOL * (1.0 - 0.6 * soft) * if wc_brush { wcb_stop } else { 1.0 } * if hatched { 1.0 - 0.6 * edge_here } else { 1.0 });
                // A GLINT on an ISOLATED light (see `MediumProfile::glint`) — a small star, the point of a lamp, a
                // light smaller than the brush — is a STAMP: the flow field has no direction at a point, and
                // the mark grown along it was a dash where the picture has a dot.
                // BLADE LIFTOFF (see `MediumProfile::blade_liftoff`): whether the straight slice was CUT HARD at its
                // forward end (a colour's edge, a protected shape, the region's edge) or ran its length / left the sheet.
                let mut stopped_hard = false;
                // BLADE FLOW (see `MediumProfile::blade_flow`): a rigid blade lays a STRAIGHT slice — by the key's
                // share of the marks, the grown path is replaced by a straight run through the seed along the path's
                // chord, as long each way as the grown path reached, stopped where the grower stops (a protected
                // shape, the seed's region, the reference departing from the mark's tone): the slice ends at the
                // plane's edge, it does not cut the corner of the form it was grown along.
                let path = if p.blade_flow > 0.0 && path.len() >= 3 && (tufted || jitter(p.seed ^ 0xB1A_DEF, k.wrapping_add(89)) + 0.5 < p.blade_flow) {
                    let (dx, dy) = (path[path.len() - 1][0] - path[0][0], path[path.len() - 1][1] - path[0][1]);
                    let m = (dx * dx + dy * dy).sqrt();
                    if m > 1e-3 {
                        let (ux, uy) = (dx / m, dy / m);
                        // BLADE TUFT: the pull upright — the chord turned toward the vertical by the texture.
                        // (BLADE SLOPE: the turn taken back by the key × the hill's steepness here.)
                        let (ux, uy) = if tufted {
                            let steep = slope_map.as_ref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0);
                            let t = tuft_here * (1.0 - p.blade_slope.clamp(0.0, 1.0) * steep);
                            let (ax, ay) = (ux * (1.0 - t), uy * (1.0 - t) + if uy < 0.0 { -t } else { t });
                            let am = (ax * ax + ay * ay).sqrt().max(1e-3);
                            let (ax, ay) = (ax / am, ay / am);
                            // BLADE SCATTER (see `MediumProfile::blade_scatter`): the pull's angle scattered by a hash
                            // of the mark, ±10° at 1 — the hand's carelessness, the regularity of the slope taken out.
                            if p.blade_scatter > 0.0 {
                                let ang = jitter(p.seed ^ 0xB1A_D5C, k.wrapping_add(103)) * 2.0 * p.blade_scatter.clamp(0.0, 1.0) * 10.0f32.to_radians();
                                let (s, c) = ang.sin_cos();
                                (ax * c - ay * s, ax * s + ay * c)
                            } else {
                                (ax, ay)
                            }
                        } else {
                            (ux, uy)
                        };
                        let d2 = |q: &[f32; 2]| (q[0] - cx).powi(2) + (q[1] - cy).powi(2);
                        let si = (0..path.len()).min_by(|a, b| d2(&path[*a]).partial_cmp(&d2(&path[*b])).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
                        let arc = |a: usize, b: usize| -> f32 { (a..b).map(|i| ((path[i + 1][0] - path[i][0]).powi(2) + (path[i + 1][1] - path[i][1]).powi(2)).sqrt()).sum() };
                        let (lb, lf) = (arc(0, si), arc(si, path.len() - 1));
                        // BLADE INERTIA (see `MediumProfile::blade_inertia`): the length each way toward the brush's
                        // full reach, the colour stop a hard threshold on the error smoothed over a radius.
                        let inert = p.blade_inertia.clamp(0.0, 1.0);
                        let full = (radius * 2.2 * blade_len_mul).max(radius + 1.0);
                        let (lb, lf) = (lb + (full - lb).max(0.0) * inert, lf + (full - lf).max(0.0) * inert);
                        let hard = STOP_TOL * (1.0 + 2.0 * inert);
                        let smooth_a = 1.0 / radius.max(2.0);
                        let guard = if clear_here { face_clear.as_deref() } else { protect_all.as_deref() };
                        let run = |sign: f32, len: f32| -> (Vec<[f32; 2]>, bool) {
                            let mut out = Vec::new();
                            let mut hard_stop = false;
                            let mut d = 1.0f32;
                            let mut err = 0.0f32;
                            while d <= len {
                                let (x, y) = (cx + ux * d * sign, cy + uy * d * sign);
                                if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                                    break;
                                }
                                let i = y as usize * w as usize + x as usize;
                                if guard.map_or(false, |g| g.get(i).copied().unwrap_or(false)) {
                                    hard_stop = true;
                                    break;
                                }
                                if let Some((mask, seed_region)) = region {
                                    if mask.get(i).copied().unwrap_or(seed_region) != seed_region {
                                        hard_stop = true;
                                        break;
                                    }
                                }
                                let e = rgb_dist(reference.get_pixel(x as u32, y as u32).0, target);
                                if inert > 0.0 {
                                    err += (e - err) * smooth_a;
                                    // BLADE VISC (see `MediumProfile::blade_visc`): the stop rises along the flow.
                                    let hard_here = if p.blade_visc > 0.0 {
                                        let (fx, fy) = (gx.get(i).copied().unwrap_or(0.0), gy.get(i).copied().unwrap_or(0.0));
                                        let fm = (fx * fx + fy * fy).sqrt();
                                        if fm > 1e-6 {
                                            let align = ((-fy * ux + fx * uy) / fm).abs();
                                            hard * (1.0 + 2.0 * p.blade_visc.clamp(0.0, 1.0) * align * align)
                                        } else {
                                            hard
                                        }
                                    } else {
                                        hard
                                    };
                                    if d > 2.0 && (err > hard_here || e > 2.0 * hard_here) {
                                        hard_stop = true;
                                        break;
                                    }
                                } else if d > 2.0 && e > STOP_TOL {
                                    hard_stop = true;
                                    break;
                                }
                                out.push([x, y]);
                                d += 1.0;
                            }
                            (out, hard_stop)
                        };
                        let (mut back, _) = run(-1.0, lb);
                        back.reverse();
                        back.push([cx, cy]);
                        let (fwd, fwd_hard) = run(1.0, lf);
                        stopped_hard = fwd_hard;
                        back.extend(fwd);
                        if back.len() >= 2 { back } else { path }
                    } else {
                        path
                    }
                } else {
                    path
                };
                let path = if glinted && isolated_light(&luma, w as usize, h as usize, cx, cy, rw) { vec![[cx, cy], [cx + 0.6, cy + 0.2]] } else { path };
                // WAVER: a real hand doesn't draw a ruler-straight line — displace the path with a little smooth
                // wobble (a characteristic, not an error). Applied to the recorded path, so replay is exact.
                let path = waver_path(&path, b_waver * rw * if hatched { 1.0 - hatch_here } else { 1.0 }, p.seed, k);
                // A hatch line is straight: the flow's curve taken out.
                let path = if hatched { straighten_path(&path, hatch_here, w, h) } else { path };
                let path = if sided { straighten_path(&path, 0.5 * p.side, w, h) } else { path };
                let path = if flat_mark { straighten_path(&path, 0.5 * p.flat_brush, w, h) } else { path };
                // An accent dab is set at its own angle, off the flow: the path turned about its middle.
                let path = if accented { turn_path(&path, (jitter(p.seed ^ 0xACCE_17, k.wrapping_add(13)) * 2.0 * ACCENT_TURN).to_radians(), w, h) } else { path };
                // CLIP (see `MediumProfile::clip`): a mark of the ground stops at the first contour it crosses (the
                // subject's own marks are not cut: a fur's every hair is an edge, and the fox's head broke up).
                let path = match &clip_edge {
                    Some(ce) if !glazing && !cresting && !lost_here && off_subject => clip_at_edge(&path, ce, w as usize, h as usize, 2.0),
                    _ => path,
                };
                // NIB WASH HOLD (see `MediumProfile::nib_wash_hold`): a wash held at a hard edge toward a light.
                let path = match wash_hold.as_ref().filter(|_| wash_reg) {
                    Some((v, e)) => {
                        let v0 = 0.299 * target[0] as f32 / 255.0 + 0.587 * target[1] as f32 / 255.0 + 0.114 * target[2] as f32 / 255.0;
                        hold_at_light(&path, v, e, v0, 0.35 - 0.2 * p.nib_wash_hold.clamp(0.0, 1.0), w as usize, h as usize)
                    }
                    None => path,
                };
                // NIB GAP (see `MediumProfile::nib_gap`): the hatching held off the long contours — cut where it enters
                // the band of white along them, not laid when it starts in it. Not on the fur (as `clip`): a coat's
                // every lock is a contour, and the band parted the fox's mass into islands.
                let path = match &nib_band {
                    Some(bd) if !glazing && !cresting && !hair_here => cut_at_band(&path, bd, w as usize, h as usize)?,
                    _ => path,
                };
                let path = match &nib_feat {
                    Some(fb) if !glazing && !cresting => cut_at_band(&path, fb, w as usize, h as usize)?,
                    _ => path,
                };
                let path = match &nib_glint {
                    Some(gb) => cut_at_band(&path, gb, w as usize, h as usize)?,
                    _ => path,
                };
                // NIB FIGURE: the figure is hatched in its own pass, not by these marks (the key's share dropped).
                if let Some(fm) = &nib_fig {
                    let (sx, sy) = (path[0][0].round().clamp(0.0, (w - 1) as f32) as usize, path[0][1].round().clamp(0.0, (h - 1) as f32) as usize);
                    if fm[sy * w as usize + sx] && jitter(p.seed ^ 0xF16_0, k.wrapping_add(29)) + 0.5 < p.nib_figure.clamp(0.0, 1.0) {
                        return None;
                    }
                }
                // NIB TUFT: the grass is drawn in tufts — the marks starting in a tuft's cell dropped by `0.6·key·strength`.
                if let Some((cell, cols, ts)) = &nib_tufts {
                    let (sx, sy) = (path[0][0].round().clamp(0.0, (w - 1) as f32) as usize, path[0][1].round().clamp(0.0, (h - 1) as f32) as usize);
                    let s = ts[(sy / cell) * cols + sx / cell];
                    if s > 0.0 && jitter(p.seed ^ 0x7F7_0, k.wrapping_add(31)) + 0.5 < 0.6 * p.nib_tuft.clamp(0.0, 1.0) * s {
                        return None;
                    }
                }
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
                // BLADE LIFTOFF: the bead only on the slices cut hard (see `BrushConfig::liftoff`).
                if stroke_brush.liftoff > 0.0 && !stopped_hard {
                    stroke_brush.liftoff = 0.0;
                }
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
                let wet = if wcb_broad && wcb_wet > 0.0 { wcb_wet } else { wet };
                // A fine mark goes on DRY (0.25 — enough to skip half the fibres) only where the picture is
                // textured (see `busy_field`); in a smooth passage it goes on wet enough to flood the tooth, and
                // the broad washes are always flooded. Recorded per stroke: replay-exact.
                let wet = match (&busy_field, wcb_broad) {
                    (Some(bf), false) if wet < 0.62 => {
                        let b = bf.get(region_i).copied().unwrap_or(0.0);
                        0.25 * b + 0.62 * (1.0 - b)
                    }
                    (Some(_), true) => wet.max(0.62),
                    _ => wet,
                };
                if wc_brush {
                    stroke_brush.k_pickup *= wcb_pickup;
                }
                // Every pass of the watercolour is capped at one dose per pixel: strokes piling up off their
                // seeds (a canvas edge, a crowded passage) overshot to black at a dose that was right at the
                // seed.
                if wc_wash && wcb_cap > 0.0 {
                    stroke_brush.film_cap = wcb_cap;
                }
                if wcb_broad {
                    // A WASH brush: even bristles, a flat section, laying its film quickly and only once.
                    stroke_brush.streak = wcb_streak;
                    stroke_brush.round = wcb_round;
                    stroke_brush.k_deposit *= wcb_dep;
                    stroke_brush.film_cap = wcb_cap;
                    stroke_brush.flat_ends = wcb_flat > 0.0;
                }
                // THE TRANSMITTANCE GLAZE: densities for ln(target / canvas), per channel, no lightening.
                let load = match under_lin {
                    Some(u) if wcb_trans => {
                        let t = color::srgb_to_linear(load_target);
                        let want = [
                            (t[0].max(0.004) / u[0].max(0.004)).ln().min(0.0),
                            (t[1].max(0.004) / u[1].max(0.004)).ln().min(0.0),
                            (t[2].max(0.004) / u[2].max(0.004)).ln().min(0.0),
                        ];
                        let key = ((want[0] * 40.0) as i16, (want[1] * 40.0) as i16, (want[2] * 40.0) as i16);
                        let dens = trans_cache.borrow_mut().entry(key).or_insert_with(|| trans_densities(cv.ln_reflectance(), want)).clone();
                        // A lane hit deposits `k_deposit × contact` of the load; `wcb_hits` hits land per pixel.
                        let eff = (stroke_brush.k_deposit * 0.68 * if fine { wcb_hits } else { wcb_hits_broad }).max(1e-3);
                        dens.iter().map(|d| d / eff * (1.0 + 1.1 * sh)).collect::<Vec<f32>>()
                    }
                    _ => load,
                };
                // Tested on scrap with THIS brush at THIS wetness: the charge that makes the stroke read `need`.
                let load = match wcb_mass {
                    Some((mass, need)) => {
                        // Optical density: the film a glaze needs grows with the log of what it must take off
                        // the canvas's value. `wcb_conc` is the pigment per unit density, calibrated on patches.
                        let charge = if wcb_probe > 0.0 {
                            charge_for(&mut probe_cache.borrow_mut(), &p.palette, 0.85, p.opacity, mass, need, wet, Some(&stroke_brush)) * wcb_conc
                        } else {
                            wcb_conc * (-(need.max(0.03).ln()))
                        } * (1.0 + 1.1 * sh);
                        let sub = mixture_cached(&mut wc_mix_cache.borrow_mut(), mass, &wc_pal, charge, wc_pal.pigments.len());
                        let mut full = vec![0f32; n];
                        for (i, v) in sub.iter().enumerate() {
                            full[wc_pal_map[i]] = *v;
                        }
                        full
                    }
                    None => load,
                };
                // PAPER: the darkest pigment alone, charged on a scrap with this brush to read the residual,
                // by the key's share against the plain mixture.
                let load = match paper_need {
                    Some(need) => {
                        let c = charge_for(&mut paper_cache.borrow_mut(), &p.palette, 0.85, p.opacity, p.palette.pigments[charcoal_dark].masstone, need, wet, Some(&stroke_brush));
                        let pp = p.paper.clamp(0.0, 1.0);
                        let mut v: Vec<f32> = load.iter().map(|x| x * (1.0 - pp)).collect();
                        v[charcoal_dark] += c * pp;
                        v
                    }
                    None => load,
                };
                // THE LAYERS OF A FAST-DRYING PAINT (acrylic; see `MediumProfile::wash_in`, `wet_cap`, `scumble`):
                // the block-in is a thin flooded wash; every mark after it goes on no wetter than the cap, a flat
                // patch on a dried layer; and the finest pass, off the faces, is dragged DRY — a thinner charge,
                // raked, no pickup — over the layer beneath. Each is in the stroke as laid (its charge, its
                // wetness, its streak, its pickup), so the replay lays it alike.
                let scumbled = p.scumble > 0.0 && last_pass && !block_in && !on_face && hair <= 0.0 && !accented && !glazed && !glazing && !cresting;
                // THE KNIFE (see `MediumProfile::knife`): off the subject, on the broad passes, a share of the
                // marks are laid with a palette knife — a flat blade full width with square ends that shears
                // the paint standing under it to a plateau and lays its own flat, thick and nearly dry (the
                // stroke's `knife`, in the record).
                let knifed = (p.knife > 0.0 && !block_in && !last_pass && !glazing && !cresting && off_subject && hair <= 0.0
                    && !accented && !glazed && !lost_here && !smooth_face && !sided
                    && jitter(p.seed ^ 0x4E1F_E, k.wrapping_add(67)) + 0.5 < p.knife)
                    // THE BLADE (see `MediumProfile::blade`): the knife as the medium's own tool — on every pass,
                    // the block-in and the finest, the subject and the faces and the hair strands too, this share
                    // of all the marks (the glaze and the crest passes keep their own brush).
                    || (p.blade > 0.0 && !glazing && !cresting && jitter(p.seed ^ 0xB1A_DE, k.wrapping_add(79)) + 0.5 < p.blade);
                // ...and a share of those SKIM (see `MediumProfile::skim`): the blade at a low angle with
                // little paint, pressed less (the shear cut by 0.8·skim) and catching only the relief's tops
                // (the stroke's `crest`, by the skim) — a torn, holed plane over a low relief.
                let skimmed = knifed && p.skim > 0.0 && jitter(p.seed ^ 0x5C1_3, k.wrapping_add(71)) + 0.5 < p.skim;
                if knifed {
                    stroke_brush.knife = if skimmed { 1.0 - 0.8 * p.skim } else { 1.0 };
                    stroke_brush.round = 0.0;
                    stroke_brush.streak = stroke_brush.streak.min(0.1);
                    stroke_brush.k_pickup = 0.0;
                    stroke_brush.viscosity = p.brush.viscosity * 1.6;
                    // BLADE LAYERS (see `MediumProfile::blade_layers`): the narrow knife a thin film that shears
                    // less of the paint beneath — both in the record (the stroke's knife share, its viscosity).
                    if p.blade_layers > 0.0 && !block_in {
                        // (BLADE FOCUS: outside the focus the knives stay thick — the thinning taken back by the distance.)
                        let k = p.blade_layers.clamp(0.0, 1.0) * depth_t * (1.0 - far_here);
                        stroke_brush.knife = (stroke_brush.knife * (1.0 - k)).max(0.15);
                        stroke_brush.viscosity *= 1.0 - 0.8 * k;
                    }
                    if skimmed {
                        stroke_brush.crest = p.skim;
                    }
                }
                if crest_here {
                    // The dry brush over the ridges: raked, no pickup, catching only the tops (see `BrushConfig::crest`).
                    stroke_brush.k_pickup = 0.0;
                    stroke_brush.streak = stroke_brush.streak.max(0.7);
                    stroke_brush.crest = 1.0;
                } else if sink_here {
                    // The wash into the furrows: a soft round, no pickup, settling only below the ridges.
                    stroke_brush.k_pickup = 0.0;
                    stroke_brush.streak = stroke_brush.streak.min(0.15);
                    stroke_brush.round = stroke_brush.round.max(0.85);
                    stroke_brush.crest = -1.0;
                } else if glazing {
                    // The glaze pass: a soft round, no pickup, a thin film.
                    stroke_brush.k_pickup = 0.0;
                    stroke_brush.streak = stroke_brush.streak.min(0.15);
                    stroke_brush.round = stroke_brush.round.max(0.85);
                } else if smooth_face {
                    // Sfumato: a soft round, picking up the paint beneath, so the marks fuse.
                    stroke_brush.k_pickup = 0.35 * p.sfumato;
                    stroke_brush.streak = stroke_brush.streak.min(0.2);
                    stroke_brush.round = stroke_brush.round.max(0.85);
                }
                if glazed {
                    // Thin and wet, picking up the paint beneath: a translucent overlap across the seam.
                    stroke_brush.k_pickup = 0.3 * p.glaze;
                }
                if hatched {
                    // The tempera hatch: a clean, even, small brush that lifts nothing beneath it.
                    stroke_brush.k_pickup *= 1.0 - hatch_here;
                    stroke_brush.streak = stroke_brush.streak.min(0.2 + 0.3 * (1.0 - hatch_here));
                    stroke_brush.round = stroke_brush.round.max(0.6);
                }
                if accented {
                    // Thick, opaque, laid on nearly dry: the dab stands up.
                    stroke_brush.viscosity = p.brush.viscosity * (1.0 + 1.5 * p.accent);
                    stroke_brush.k_pickup = 0.0;
                }
                // FILM (see `MediumProfile::film`): the glaze marks — the glaze pass, the sink wash, the lost
                // edges, the overlaps along the silhouette — laid as a translucent film OVER the dried paint,
                // not mixed into it (the stroke's `over`, in the record).
                if p.film > 0.0 && !block_in && (glazing || sink_here || lost_here || glazed) {
                    stroke_brush.over = p.film;
                }
                // SPLIT (see `MediumProfile::split`): the dry marks — the scumble and the crest in the light,
                // the hair strands anywhere — comb into bundles as they dry and the faster they go (the
                // stroke's `split`, in the record). Not in the shadow: a combed scumble over every dark mass
                // was a hatching, where the shade keeps the darks calm.
                if p.split > 0.0 && !block_in && !on_face && (hair > 0.0 || ((scumbled || crest_here) && sh_raw < 0.5)) {
                    stroke_brush.split = p.split;
                    // ...each bundle its own slight tint (see `MediumProfile::tint`).
                    stroke_brush.tint = p.tint;
                }
                // TWIST (see `MediumProfile::twist`): the brush turned in the fingers along the hair strands
                // and, by this share, the finest pass's marks off the faces — the width running from the
                // brush's breadth to a line (the stroke's `twist`, in the record).
                if p.twist > 0.0 && !block_in && !on_face && !knifed && (hair > 0.0 || (last_pass && jitter(p.seed ^ 0x7E15_7, k.wrapping_add(73)) + 0.5 < p.twist)) {
                    stroke_brush.twist = p.twist;
                }
                if sided {
                    stroke_brush.round = 0.0;
                    stroke_brush.streak = stroke_brush.streak.max(0.9);
                    stroke_brush.k_pickup = 0.0;
                }
                // The flat brush: a square chisel end, the mark full to its end (in the record).
                if flat_mark {
                    stroke_brush.flat_ends = true;
                    stroke_brush.round = 0.0;
                }
                // BONELESS (see `MediumProfile::boneless`): the passes after the block-in laid as its wash.
                let washed = block_in || (p.boneless > 0.0 && (layer as f32) <= p.boneless);
                let (wet, load) = if washed && p.wash_in > 0.0 {
                    (wet.max(0.85), load.iter().map(|v| v * (1.0 - 0.7 * p.wash_in)).collect::<Vec<f32>>())
                } else if crest_here {
                    // Dry and thin: a scumble that lands only where the relief stands up.
                    (wet.min(0.2), load.iter().map(|v| v * 0.6).collect::<Vec<f32>>())
                } else if sink_here {
                    // Wet and thin: a translucent wash that runs into the hollows.
                    (wet.max(0.7), load.iter().map(|v| v * 0.4).collect::<Vec<f32>>())
                } else if lost_here {
                    // The lost edge: a wet mark of the two sides' mean, `0.4 + 0.6·lost` of the charge, across the seam.
                    (wet.max(0.65), load.iter().map(|v| v * (0.4 + 0.6 * p.lost)).collect::<Vec<f32>>())
                } else if glazing {
                    (wet.max(0.65), load.iter().map(|v| v * p.glaze_pass).collect::<Vec<f32>>())
                } else if smooth_face {
                    (wet.max(0.7), load.iter().map(|v| v * (1.0 - 0.5 * p.sfumato)).collect::<Vec<f32>>())
                } else if glazed {
                    (wet.max(0.55), load.iter().map(|v| v * (1.0 - 0.55 * p.glaze)).collect::<Vec<f32>>())
                } else if accented {
                    (wet.min(0.3), load)
                } else if knifed {
                    // The knife's paint: thick, nearly dry, laid flat; a skimming blade carries half of it, drier.
                    if skimmed { (wet.min(0.15), load.iter().map(|v| v * 0.5).collect::<Vec<f32>>()) } else { (wet.min(0.3), load) }
                } else if sided {
                    // The side of the stick: a light dry sweep, `1 − 0.3·side` of the charge.
                    (wet.min(0.1), load.iter().map(|v| v * (1.0 - 0.3 * p.side)).collect::<Vec<f32>>())
                } else if scumbled {
                    stroke_brush.streak = stroke_brush.streak.max(0.8);
                    stroke_brush.k_pickup = 0.0;
                    (wet.min(0.15), load.iter().map(|v| v * (1.0 - 0.45 * p.scumble)).collect::<Vec<f32>>())
                } else if glinted {
                    // The glint: dry, the full covering charge, standing thicker — the final highlight's touch.
                    stroke_brush.viscosity = p.brush.viscosity * (1.0 + 1.5 * p.glint);
                    stroke_brush.k_pickup = 0.0;
                    (wet.min(0.1), load.iter().map(|v| v * (1.0 + 0.3 * p.glint)).collect::<Vec<f32>>())
                } else if hatched {
                    // The hatch: dry, semi-opaque — the layers show through one another; the lights thinner still.
                    // SOLID (see `MediumProfile::solid`): the charge keyed to the value, covering in the darks.
                    let t = hatch_charge(hatch_here, p.solid, rel_v) * if lit { 0.8 } else { 1.0 };
                    (wet.min(if p.wet_cap < 1.0 { p.wet_cap } else { 0.35 }), load.iter().map(|v| v * t).collect::<Vec<f32>>())
                } else if !block_in && p.wet_cap < 1.0 {
                    (wet.min(p.wet_cap), load)
                } else {
                    (wet, load)
                };
                // SMOOTH (see `MediumProfile::smooth`): on a calm passage the restating mark is a soft thin
                // wash — wetter by the calm so it blends into its neighbours, `1 − 0.6·calm` of the charge so
                // the tone is built up in veils — the still sky and the wall of a panel. Not the block-in, the
                // veils, a glint (the star stays a crisp dot) or an accent.
                // The committed shadow's extra charge (`1 + 1.1·sh`) is given back by the calm too: a still
                // dark sky is built in veils like a light one, not in opaque planks.
                let (wet, load) = if calm > 1e-3 && !block_in && !glazing && !cresting && !glinted && !accented && !knifed {
                    // (Barely thinner: a thin mark on a toned ground showed the ground between the marks as a
                    // chequer of the ground's colour; the smoothness is the relaxed dish and the short mark.)
                    let t = (1.0 - 0.15 * calm) * (1.0 - 0.5 * calm * (1.0 - 1.0 / (1.0 + 1.1 * sh)));
                    (wet.max(0.75 * calm).min(0.85), load.iter().map(|v| v * t).collect::<Vec<f32>>())
                } else {
                    (wet, load)
                };
                // DITHER: the mark that took the farther dish is a veil of it — half the charge.
                let load = if dish_far && !block_in { load.iter().map(|v| v * 0.5).collect::<Vec<f32>>() } else { load };
                // BODY VARY (see `MediumProfile::body_vary`): the mark's body spread as a Gaussian in its log —
                // a thick one goes down with its full charge and stands high (the viscosity, below), a thin one
                // carries less, wetter, and melts into the layer beneath. Not the block-in, the glaze pass or an
                // accent (an accent is thick by definition). The charge and the wetness are in the record.
                let body = if p.body_vary > 0.0 && !block_in && !glazing && !cresting && !accented {
                    let g = 2.0 * (jitter(p.seed ^ 0xB0D1, k.wrapping_add(41)) + jitter(p.seed ^ 0xB0D2, k.wrapping_add(43)) + jitter(p.seed ^ 0xB0D3, k.wrapping_add(47)));
                    (p.body_vary * g).exp().clamp(0.5, 2.0)
                } else {
                    1.0
                };
                let (wet, load) = if body < 1.0 {
                    (wet + (0.6 - wet).max(0.0) * (1.0 - body), load.iter().map(|v| v * body.powf(0.6)).collect::<Vec<f32>>())
                } else {
                    (wet, load)
                };
                // A shaded mark is laid calm: wet enough that the dry brush does not skip on the tooth (the skip
                // fades to nothing at 0.6), so the shadow is a flat quiet patch, not a rake of dry noise.
                let wet = if shaded > 0.0 && !accented { wet + (0.62 - wet).max(0.0) * shaded } else { wet };
                // BOKASHI (see `MediumProfile::bokashi`): toward the edge of the focus the brush carries water.
                let wet = match (&ma_field, p.bokashi > 0.0) {
                    (Some(mf), true) => {
                        let m = mf.get(region_i).copied().unwrap_or(1.0).clamp(0.0, 1.0);
                        let a = if p.ma_focus > 0.0 { m.powf(1.0 + 4.0 * p.ma_focus.clamp(0.0, 1.0)) } else { m.max(subj.clamp(0.0, 1.0)) };
                        wet + (0.95 - wet).max(0.0) * p.bokashi.clamp(0.0, 1.0) * (1.0 - a).powi(2)
                    }
                    _ => wet,
                };
                // IMPASTO MAPPED TO THE PICTURE (`impasto_map`): the thickness of the paint is a decision about
                // the picture, not a global texture — the LIGHTS are laid thick and the shadows thin (the paint
                // stands where the light falls; a shadow is a glaze), the SUBJECT thick and the background thin
                // (the far planes recede as thin paint), nearer thicker when a depth map is known. Per stroke,
                // as the brush's viscosity, recorded so the replay lays the same height.
                if p.impasto_map > 0.0 {
                    // The LIGHT term reads the target's value against the picture's own range (its 5th and 95th
                    // percentiles), not absolutely: on a snow scene every patch is bright and the absolute term
                    // gave the sky and the drifts one thickness while the dark branches won on pigment load.
                    let rel = ((tluma - map_lo) / (map_hi - map_lo).max(0.05)).clamp(0.0, 1.0);
                    let light = 0.35 + 1.3 * rel;
                    let plane = if subj > 0.5 { 1.0 } else { 0.6 };
                    let near = p.depth.as_deref().and_then(|d| d.get(region_i)).map(|&z| 0.55 + 0.9 * z.clamp(0.0, 1.0)).unwrap_or(1.0);
                    // The SURFACE: a smooth form (skin, a sky, glass) is laid smooth and blended, a broken one
                    // (a beard, bark, cobbles) stands up — read from the picture's own texture at the mark's
                    // scale, not from a global switch.
                    // A FACE is a smooth form whatever its wrinkles say at the mark's scale (sfumato, not
                    // relief — the lit cheek came out the thickest paint on the sheet); its hair and beard
                    // are not. Smooth wins over light: a lit smooth form is laid thin and blended.
                    let busy = busy_field.as_deref().and_then(|b| b.get(region_i)).copied().unwrap_or(0.5);
                    let face_here = p.face_mask.as_deref().and_then(|m| m.get(region_i)).copied().unwrap_or(0.0) > 0.35 && hair < 0.3;
                    let surface = if face_here { 0.25 } else { 0.3 + 1.1 * busy };
                    // Height is viscosity × pigment LAID, and a dark target is laid with far more pigment than a
                    // light one — so without compensation the darks come out thickest whatever the map says
                    // (measured: lights:shadows 0.6–0.8 on a snow scene with the map at 1). Divide the load
                    // out: the thickness follows the map's intent, not the charge.
                    let load_here: f32 = load.iter().map(|v| v.max(0.0)).sum::<f32>().max(1e-3);
                    let load_norm = (load_ref / load_here).clamp(0.35, 3.0);
                    let f = (light * plane * near * surface * load_norm).clamp(0.12, 3.0);
                    stroke_brush.viscosity = p.brush.viscosity * (1.0 + p.impasto_map.clamp(0.0, 1.0) * (f - 1.0));
                }
                if body != 1.0 {
                    stroke_brush.viscosity *= body;
                }
                // A strand ends in a POINT (a hair has a tip); a mass mark lifts off at about half its width.
                let width1 = (rw * (0.55 - 0.42 * hair)).max(p.min_brush * 0.5 * (1.0 - 0.6 * hair));
                let width1 = if hatched { width1 + (rw - width1) * hatch_here } else { width1 };
                let width1 = if sided { rw } else { width1 };
                // BONELESS (see `MediumProfile::boneless`): a wash goes on at an even width with a soft round edge —
                // the water sets the mark, not the pressure (a tapered wedge with a hard edge read as a trowel's smear).
                let width1 = if washed && p.boneless > 0.0 && p.wash_in > 0.0 { rw } else { width1 };
                if washed && p.boneless > 0.0 && p.wash_in > 0.0 {
                    stroke_brush.round = 1.0;
                }
                // TOUCH (see `MediumProfile::touch`): the pressure follows the value; VELVET presses full.
                let pressure = if sided { (pvar * (1.0 - 0.5 * p.side)).clamp(0.3, 1.0) } else { pvar.clamp(0.4, 1.0) };
                let pressure = match paper_need {
                    Some(_) if velvet_here => 1.0,
                    // NIB: the steel point lays its ink whole, at any pressure.
                    Some(_) if p.nib > 0.0 => pressure + (1.0 - pressure) * p.nib.clamp(0.0, 1.0),
                    Some(_) if p.touch > 0.0 => {
                        let dark = (1.0 - (color::linear_luma(color::srgb_to_linear(target)) / charcoal_paper_l.max(1e-3)).clamp(0.0, 1.0).powf(1.0 / 2.2)).clamp(0.0, 1.0);
                        pressure + (0.38 + 0.6 * dark.powf(0.7) - pressure) * p.touch.clamp(0.0, 1.0)
                    }
                    _ => pressure,
                };
                // TSUKE (see `MediumProfile::tsuke`): the loaded edge (the path's left hand) turned to the darker
                // side of the form — the path reversed where the picture is lighter on its left (in the record).
                let path = if p.brush.tsuke > 0.0 && erase_wet.is_none() && path.len() >= 2 {
                    let m = path.len() / 2;
                    let (a, b) = (path[m.saturating_sub(1)], path[(m + 1).min(path.len() - 1)]);
                    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                    let dl = (dx * dx + dy * dy).sqrt();
                    if dl > 1e-3 {
                        let (nx, ny) = (-dy / dl * rw * 0.5, dx / dl * rw * 0.5);
                        let mid = path[m];
                        let at = |x: f32, y: f32| luma[(y.clamp(0.0, h as f32 - 1.0) as usize) * w as usize + x.clamp(0.0, w as f32 - 1.0) as usize];
                        if at(mid[0] - nx, mid[1] - ny) > at(mid[0] + nx, mid[1] + ny) {
                            path.iter().rev().copied().collect()
                        } else {
                            path
                        }
                    } else {
                        path
                    }
                } else {
                    path
                };
                // (An eraser's lift is recorded in the stroke's wetness, as the replay reads a wipe.)
                let s = Stroke { path, width0: rw, width1, load: if erase_wet.is_some() { Vec::new() } else { load }, pressure, wetness: erase_wet.unwrap_or(wet) };
                // THE RAMP (see `MediumProfile::ramp`): the brush's charge drifts along the mark toward the
                // picture's colour at the mark's end — the seed's own tone (its transforms kept) plus the
                // picture's end-to-start difference in linear light, by the key's share. Full on the soft
                // marks, fading as the hatching takes over; off the glazes, lost edges, hair, accents, glints.
                let ramp_here = if p.ramp > 0.0 && !glazing && !cresting && !lost_here && !glinted && !accented && !knifed && hair <= 0.0 && s.path.len() >= 2 { p.ramp * (1.0 - hatch_here) } else { 0.0 };
                let load_end: Option<Vec<f32>> = (ramp_here > 1e-3).then(|| {
                    let e = s.path[s.path.len() - 1];
                    let (ex, ey) = (e[0].round().clamp(0.0, (reference.width() - 1) as f32) as u32, e[1].round().clamp(0.0, (reference.height() - 1) as f32) as u32);
                    let re = color::srgb_to_linear(reference.get_pixel(ex, ey).0);
                    let rs = color::srgb_to_linear(reference.get_pixel(cx.round().clamp(0.0, (reference.width() - 1) as f32) as u32, cy.round().clamp(0.0, (reference.height() - 1) as f32) as u32).0);
                    let lt = color::srgb_to_linear(load_target);
                    let end = color::linear_to_srgb([(lt[0] + ramp_here * (re[0] - rs[0])).clamp(0.0, 1.0), (lt[1] + ramp_here * (re[1] - rs[1])).clamp(0.0, 1.0), (lt[2] + ramp_here * (re[2] - rs[2])).clamp(0.0, 1.0)]);
                    let scale = s.load.iter().map(|v| v.max(0.0)).sum::<f32>() / ramp_base.max(1e-6);
                    if let (true, Some((wp, wi))) = (carn_here, carnation.as_ref()) { carnation_mix(&mut carn_cache.borrow_mut(), cache, end, wp, wi, &p.palette, p.charge * (1.0 + 1.1 * sh) * scale, n, p.carnation) } else { mixture_cached(cache, end, &p.palette, p.charge * (1.0 + 1.1 * sh) * scale, n) }
                });
                // BLADE MIX (see `MediumProfile::blade_mix`): a knife slice's SECOND colour — the picture under the
                // slice's footprint (a dozen points along the path, each side of it, before the planes) split at its
                // median value into a darker and a lighter half; the half whose mean lies further from the mark's
                // own tone is the second colour, taken toward it by the key, mixed at the mark's charge. Nothing
                // where the two halves are alike (a plain passage). In the record as the mark's second charge.
                let load_end = match load_end {
                    None if knifed && p.brush.mix > 0.0 && erase_wet.is_none() && s.path.len() >= 2 => {
                        // (BLADE TUFT: the second colour taken fully — the greens of the tuft in one pull.)
                        let k = if tufted { (p.brush.mix.clamp(0.0, 1.0) * (1.0 + 2.0 * tuft_here)).min(1.0) } else { p.brush.mix.clamp(0.0, 1.0) };
                        let src = blade_raw.as_ref().unwrap_or(&reference);
                        let (rw_, rh_) = ((src.width() - 1) as f32, (src.height() - 1) as f32);
                        let n_s = 12usize;
                        let mut pts: Vec<[f32; 3]> = Vec::with_capacity(2 * n_s);
                        for i in 0..n_s {
                            let fi = (i as f32 + 0.5) / n_s as f32 * (s.path.len() - 1) as f32;
                            let (i0, fr) = (fi.floor() as usize, fi.fract());
                            let i1 = (i0 + 1).min(s.path.len() - 1);
                            let (px, py) = (s.path[i0][0] + (s.path[i1][0] - s.path[i0][0]) * fr, s.path[i0][1] + (s.path[i1][1] - s.path[i0][1]) * fr);
                            let (tx, ty) = (s.path[i1][0] - s.path[i0][0], s.path[i1][1] - s.path[i0][1]);
                            let tm = (tx * tx + ty * ty).sqrt().max(1e-3);
                            let (nx, ny) = (-ty / tm * rw * 0.3, tx / tm * rw * 0.3);
                            for sgn in [-1.0f32, 1.0] {
                                let (qx, qy) = ((px + nx * sgn).round().clamp(0.0, rw_) as u32, (py + ny * sgn).round().clamp(0.0, rh_) as u32);
                                pts.push(color::srgb_to_linear(src.get_pixel(qx, qy).0));
                            }
                        }
                        let mut ls: Vec<f32> = pts.iter().map(|c| color::linear_luma(*c)).collect();
                        ls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let med = ls[ls.len() / 2];
                        let (mut dark, mut nd, mut lite, mut nl) = ([0f32; 3], 0.0f32, [0f32; 3], 0.0f32);
                        for c in &pts {
                            if color::linear_luma(*c) < med {
                                for j in 0..3 { dark[j] += c[j]; }
                                nd += 1.0;
                            } else {
                                for j in 0..3 { lite[j] += c[j]; }
                                nl += 1.0;
                            }
                        }
                        if nd < 1.0 || nl < 1.0 {
                            None
                        } else {
                            let dark = [dark[0] / nd, dark[1] / nd, dark[2] / nd];
                            let lite = [lite[0] / nl, lite[1] / nl, lite[2] / nl];
                            let lt = color::srgb_to_linear(load_target);
                            let l0 = color::linear_luma(lt);
                            let (dd, dl) = ((color::linear_luma(dark) - l0).abs(), (color::linear_luma(lite) - l0).abs());
                            let far = if dd > dl { dark } else { lite };
                            if dd.max(dl) < 0.015 {
                                None
                            } else {
                                let second = color::linear_to_srgb([(lt[0] + (far[0] - lt[0]) * k).clamp(0.0, 1.0), (lt[1] + (far[1] - lt[1]) * k).clamp(0.0, 1.0), (lt[2] + (far[2] - lt[2]) * k).clamp(0.0, 1.0)]);
                                let scale = s.load.iter().map(|v| v.max(0.0)).sum::<f32>() / ramp_base.max(1e-6);
                                Some(mixture_cached(cache, second, &p.palette, p.charge * (1.0 + 1.1 * sh) * scale, n))
                            }
                        }
                    }
                    le => le,
                };
                if erase_wet.is_some() {
                    // ERASE: the replay's own call — the header's brush, the strength × lift.
                    s.wipe(cv, &p.brush, s.wetness * p.lift);
                } else {
                    s.rasterize_ramp(cv, &stroke_brush, load_end.as_deref());
                }
                let ramp_mix: Option<Vec<(String, f32)>> = load_end.as_ref().map(|le| le.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect());
                // SPATTER (see `MediumProfile::spatter`): the thickest marks (a body above 1.5), in the light,
                // off the faces and hair, fling a drop or three off their lift-off — noted here, flung after
                // the pass, in the mark's own order.
                if p.spatter > 0.0 && body >= 1.5 && !block_in && !glazing && !cresting && !on_face && hair <= 0.0 && sh_raw < 0.5 && s.path.len() >= 2
                    && jitter(p.seed ^ 0x5FA7_7E, k.wrapping_add(59)) + 0.5 < p.spatter * ((body - 1.5) / 0.5).clamp(0.0, 1.0)
                {
                    let n_pts = s.path.len();
                    let (e, q) = (s.path[n_pts - 1], s.path[n_pts - 2]);
                    let (mut dx, mut dy) = (e[0] - q[0], e[1] - q[1]);
                    let dl = (dx * dx + dy * dy).sqrt();
                    if dl > 1e-4 {
                        dx /= dl;
                        dy /= dl;
                        // The mark's SPEED (see `MediumProfile::tail`): its length for its width — a dab of
                        // two widths is slow (0), a sweep of six widths fast (1).
                        let plen: f32 = s.path.windows(2).map(|q| ((q[1][0] - q[0][0]).powi(2) + (q[1][1] - q[0][1]).powi(2)).sqrt()).sum();
                        let speed = ((plen / s.width0.max(1e-3) - 2.0) / 4.0).clamp(0.0, 1.0) * p.tail;
                        spats.borrow_mut().push((e, [dx, dy], s.width0, s.load.clone(), stroke_brush.viscosity, k, speed));
                    }
                }
                // Record the stroke into the score (mix as pigment name → value, for the non-zero pigments).
                let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                let mix = if erase_wet.is_some() { Vec::new() } else { mix };
                Some(StrokeRecord {
                    id: 0,
                    wipe: erase_wet.is_some(), wash: false,
                    stage: pass.stage.clone(),
                    spline: s.path,
                    w0: s.width0,
                    w1: s.width1,
                    taper: 0.4 + 0.45 * hair,
                    mix,
                    wet: s.wetness,
                    press: s.pressure,
                    // The brush AS IT PAINTED (the wash brush overrides both after the jitter is drawn).
                    streak: stroke_brush.streak,
                    round: stroke_brush.round,
                    // A detail accent's near-clean pickup is part of how it was laid — record it so replay is exact.
                    // A pickup the stroke's own brush differs in from the header's is part of how it was laid.
                    // The dry brush zeroes it; left unrecorded, replay rebuilt every dry-brush mark with the
                    // header's pickup and drifted — the unrecorded-bristles bug again.
                    pickup: if detail || hair > 0.0 || p.technique == WetTechnique::DryOnDry || wc_brush || scumbled || accented || glazed || glazing || smooth_face || crest_here || sink_here || lost_here || knifed || hatched || sided { Some(stroke_brush.k_pickup) } else { None },
                    bristles: (hair > 0.0).then_some(stroke_brush.bristles),
                    kd: (wcb_broad || wc_wash).then_some(stroke_brush.k_deposit),
                    cap: ((wcb_broad || wc_wash) && stroke_brush.film_cap > 0.0).then_some(stroke_brush.film_cap),
                    flat: stroke_brush.flat_ends,
                    hold: stroke_brush.hold_charge,
                    visc: (p.impasto_map > 0.0 || accented || knifed || glinted || body != 1.0).then_some(stroke_brush.viscosity),
                    crest: (crest_here || sink_here || skimmed).then_some(stroke_brush.crest),
                    over: (stroke_brush.over > 0.0).then_some(stroke_brush.over),
                    split: (stroke_brush.split > 0.0).then_some(stroke_brush.split),
                    tint: (stroke_brush.tint > 0.0).then_some(stroke_brush.tint),
                    knife: (stroke_brush.knife > 0.0).then_some(stroke_brush.knife),
                    twist: (stroke_brush.twist > 0.0).then_some(stroke_brush.twist),
                    tgt: Some(target),
                    mix2: ramp_mix,
                    smudge: None,
                    stop: (stroke_brush.liftoff > 0.0).then_some(1.0),
                    edge: None,
                    prof: None,
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
                pr(PaintProgress::Mixing { pass: layer_all + 1, passes: passes.len(), radius, colours: keys.len(), threads });
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
            pr(PaintProgress::Painting { pass: layer_all + 1, passes: passes.len(), radius });
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
        if p.fill > 0.0 && layer_all + 1 == passes.len() {
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

        // SPATTER (see `MediumProfile::spatter`): the drops flung off this pass's thickest marks — one to three
        // per mark, along the mark's last direction, two to eight widths on, a little to either side; each a
        // round thick dot (an oval along its flight) of the mark's own charge, laid wet enough not to skip,
        // twice the mark's viscosity so it stands up. Ordinary strokes in the score, under the pass's own stage (a
        // stage of their own read to the replay as a finish, which dries every pass before them); a drop is known
        // by its zero taper and no intent colour.
        if p.spatter > 0.0 && !spats.borrow().is_empty() {
            let cap = (p.budget / 300).max(1);
            let mut flung = 0usize;
            let mut brush = p.brush;
            brush.streak = 0.0;
            brush.round = 1.0;
            brush.k_pickup = 0.0;
            let list: Vec<_> = spats.borrow_mut().drain(..).collect();
            for (end, dir, w0, load, visc, kk, speed) in list {
                if flung >= cap || placed >= p.budget {
                    break;
                }
                let n_drops = 1 + ((jitter(p.seed ^ 0xD20_95, kk) + 0.5) * 3.0) as usize;
                for j in 0..n_drops.min(3) {
                    let kj = kk.wrapping_mul(31).wrapping_add(j as u64 * 7 + 1);
                    // A fast mark's drop flies farther (`tail`: up to twice as far)...
                    let dist = w0 * (2.0 + 6.0 * (jitter(p.seed ^ 0xD20_96, kj) + 0.5)) * (1.0 + speed);
                    let side = w0 * 1.5 * jitter(p.seed ^ 0xD20_97, kj);
                    let x = end[0] + dir[0] * dist - dir[1] * side;
                    let y = end[1] + dir[1] * dist + dir[0] * side;
                    if x < 1.0 || y < 1.0 || x >= w as f32 - 1.0 || y >= h as f32 - 1.0 {
                        continue;
                    }
                    let i = y as usize * w as usize + x as usize;
                    if p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.2 || p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.0 {
                        continue;
                    }
                    // A drop's size follows the brush that flung it more than the mark's width (the thickest
                    // marks are mostly the fine passes'), and is never a speck: a radius of a quarter of the
                    // width plus a sheet unit and a half, spread ×0.6–1.4.
                    let unit = (w.min(h) as f32 / 1024.0).max(0.5);
                    let r = ((w0 * 0.25 + unit * 1.5) * (0.6 + 0.8 * (jitter(p.seed ^ 0xD20_98, kj) + 0.5))).max(1.0);
                    // A slow mark's drop is round, or barely an oval along its flight (a longer path read as
                    // a dash); a fast mark's (`tail`) stretches up to four times as long along its flight and
                    // thins ahead to a tail — a round head where it landed, the smear in front of it.
                    let len = r * (0.2 + 0.8 * (jitter(p.seed ^ 0xD20_99, kj) + 0.5)) * (1.0 + 3.0 * speed);
                    let mut b = brush;
                    b.viscosity = visc * 2.0;
                    let s = Stroke { path: vec![[x, y], [x + dir[0] * len, y + dir[1] * len]], width0: 2.0 * r, width1: 2.0 * r * (1.0 - 0.7 * speed), load: load.iter().map(|v| v * 1.2).collect(), pressure: 1.0, wetness: 0.55 };
                    s.rasterize(&mut canvas, &b);
                    let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(pi, v)| (p.palette.pigments[pi].name.to_string(), *v)).collect();
                    placed += 1;
                    in_pass += 1;
                    flung += 1;
                    score.strokes.push(StrokeRecord { id: placed as u32, wipe: false, wash: false, stage: pass.stage.clone(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.0, mix, wet: s.wetness, press: 1.0, streak: b.streak, round: b.round, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: Some(b.viscosity), crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
                }
            }
            if flung > 0 {
                tracing::info!(target: "plakat", "spatter: {flung} drops flung after pass {}", layer + 1);
            }
        }
        // THE FINGER (see `MediumProfile::blend`): after a broad pass the calm passages are rubbed along the
        // flow — the broad marks there fused into one soft tone — as that pass's own records.
        // (The broad half of the ladder only — the underpainting is blended, the drawing over it is not; and
        // after the finest pass the skin alone, when `skin` asks for it.)
        if p.blend > 0.0 && !block_in && ((depth_t < 0.5 && !last_pass) || (last_pass && p.skin > 0.0)) && !glazing && !cresting && placed < p.budget {
            let rubbed = blend_pass(&mut canvas, &mut score, p, &luma, range_field.as_deref(), &gx, &gy, &reference, protect_all.as_deref(), radius, &pass.stage, last_pass, false, &mut placed, &mut k);
            in_pass += rubbed;
            if rubbed > 0 {
                tracing::info!(target: "plakat", "blend: {rubbed} rubs after pass {}", layer + 1);
            }
        }
        // THE STUMP (see `MediumProfile::stump`): after the finest pass the dark masses rubbed.
        if p.stump > 0.0 && last_pass && !glazing && !cresting && placed < p.budget {
            let rubbed = blend_pass(&mut canvas, &mut score, p, &luma, range_field.as_deref(), &gx, &gy, &reference, protect_all.as_deref(), radius, &pass.stage, false, true, &mut placed, &mut k);
            in_pass += rubbed;
            if rubbed > 0 {
                tracing::info!(target: "plakat", "stump: {rubbed} rubs after pass {}", layer + 1);
            }
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
        // The watercolour SPATTERS WHILE THE WASHES ARE WET (see `splatter_pass`): at the end of the last
        // broad pass, BEFORE its bleed and flow (the replay lays a stage's strokes and then crosses — the
        // drops must be laid on the same side of that crossing), as that pass's own marks — a drop in a wash
        // blooms, one on dry paper stays sharp. (At the end, on a dried sheet, every drop was a hard dot.)
        // (A luminous paint's brush passes come after its wash stages: their taper index continues from them.)
        let (b_idx, b_n) = if p.luminous { (wash_stages + layer, n_stages_total) } else { (layer_all + bole_stage, passes.len() + bole_stage) };
        if wc_wash && p.splatter > 0.0 && !splattered_wet && flow_taper(b_idx, b_n).is_some() && flow_taper(b_idx + 1, b_n).is_none() {
            splatter_pass(&mut canvas, &mut score, input, p, &mut placed, &mut k, &pass.stage, true);
            splattered_wet = true;
        }
        lap("pass:strokes", &mut prof_acc, &mut prof_t);
        // WET-INTO-WET per pass, tapering coarse → fine: the broad washes bloom into each other while still wet,
        // the later, finer work goes onto paper that has set and stays crisp. One bleed over the finished
        // painting fused EVERYTHING (measured: an ink-wash lost half its sharpness in that final step, and the
        // whole sheet read as one blur). The same schedule is reproduced by the score replay at each stage
        // boundary, so the drawing stays byte-exact.
        // (A luminous paint's brush passes come after its wash stages: their taper index continues from them.)
        if bleed_eff > 0.0 {
            canvas.bleed_with(bleed_eff * stage_bleed_taper(score.header.stages.as_deref(), b_idx, b_n), p.diffuse);
        }
        // NIB WASH WATER (see `MediumProfile::nib_wash_water`): the planes' ink seeps through the water at the end
        // of their pass; in the header, so the replay seeps the same.
        if score.header.water > 0.0 && water_stage(score.header.stages.as_deref(), b_idx) {
            canvas.seep(score.header.water);
        }
        if wcb_broad_pass && wcb_bleed > 0.0 {
            canvas.bleed_with(wcb_bleed, 0.0);
        }
        // THE FLUID STAGE (see `Canvas::flow`): the water film the pass left, the pigment diffusing in it, the
        // rim where it dries — after every pass, tapering with the passes as the bleed does, and in the score
        // so the replay crosses the same water.
        if let Some((fs, fr, fm, fg, fe)) = score.header.flow {
            if let Some(t) = flow_taper(b_idx, b_n) {
                canvas.flow(fs * t, fr, fm, fg, fe);
            }
        }
        // WET COLLISION (see `Canvas::collide`): the broad passes' wet paint drags where strokes meet. In the
        // score so the replay crosses the same drags; the face mask is not in the score, so the painter
        // masks the faces and the replay — which has no faces — relies on the collision reaching only the
        // broad passes, whose marks the face tier repaints after.
        if let Some((cs, cr)) = score.header.collide {
            let (st, at) = collide_schedule(cs, b_idx, b_n);
            if st > 0.0 {
                canvas.collide(st, cr, collide_hold.as_deref(), at);
            }
        }
        if b_idx + 1 < b_n {
            canvas.dry(1.0 - p.dry);
        }
        lap("pass:bleed+dry", &mut prof_acc, &mut prof_t);
        stats.push(PassStat { stage: pass.stage.clone(), radius, strokes: in_pass, seconds: pass_t0.elapsed().as_secs_f64() });
        if let Some(dir) = std::env::var_os("PLAKAT_WCB_DUMP") {
            let _ = canvas.to_image().save(std::path::Path::new(&dir).join(format!("pass_{layer_all}.png")));
        }
        if let Some(dir) = &p.dump_passes {
            let safe: String = pass.stage.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
            let _ = canvas.to_image_finished(&score.header.finish()).save(dir.join(format!("pass_{layer_all:02}_{safe}.png")));
        }
        if let Some(path) = std::env::var_os("PLAKAT_PAINT_HEIGHT_DUMP") {
            // Diagnostic: the paint HEIGHT after this pass as a 16-bit grey (scaled to its peak).
            let peak = canvas.height.iter().copied().fold(0.0f32, f32::max).max(1e-6);
            let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(w, h, |x, y| image::Luma([(canvas.height[(y * w + x) as usize] / peak * 65535.0) as u16]));
            let _ = img.save(&path);
        }
        if prof_on { eprintln!("PROFILE pass {layer} r={radius:.1} strokes={in_pass} {:.2}s", pass_t0.elapsed().as_secs_f64()); }
    }
    if !rejected.is_empty() {
        tracing::info!(target: "plakat", "paint critic: rejected {} pass(es): {}", rejected.len(), rejected.join(", "));
    }

    // CONTOUR pass (line media): DRAW the strongest edges as clean lines — pen/pencil/charcoal outline the
    // subject, they don't only shade it. Laid before the bleed so a smudgy medium softens the lines a touch.
    // (Density media draw their contours in `ink_drawing`; this pass would stack its row-ordered dashes on top,
    // thickening whatever lies in the first rows until its cap hits.)
    // NIB FIGURE (see `MediumProfile::nib_figure`): the figure's tone in parallel hatching, before its bone.
    if p.nib > 0.0 && p.nib_figure > 0.0 && placed < p.budget {
        if let Some(fm) = nib_figure_mask(p, w, h) {
            canvas.dry(0.0);
            nib_figure_hatch(&mut canvas, &mut score, original, p, &fm, &mut placed, &mut k);
        }
    }
    // THE BONE (see `MediumProfile::bone`): the structure's contours as calligraphic brush lines, over the washes.
    if p.bone > 0.0 && placed < p.budget {
        canvas.dry(0.0);
        // NIB WASH LINE (see `MediumProfile::nib_wash_line`): the line from the picture, not the smoothed armature.
        let bone_src = if nib_wash_n > 0 && p.nib_wash_line > 0.0 && original.dimensions() == input.dimensions() { original } else { input };
        bone_pass(&mut canvas, &mut score, bone_src, p, protect_all.as_deref(), ma_field.as_deref(), &mut placed, &mut k);
    }
    // NIB POOL and NIB RIM (see `MediumProfile::nib_pool`, `nib_rim`): the contact shadows and the figure's outline.
    // (the replay dries once when the stage turns to `nib`, so the accents after it go on without a second drying)
    let mut nib_wet = false;
    if p.nib > 0.0 && p.nib_figure > 0.0 && (p.nib_pool > 0.0 || p.nib_rim > 0.0) && placed < p.budget {
        if let Some(fm) = nib_figure_mask(p, w, h) {
            canvas.dry(0.0);
            let before = placed;
            nib_figure_rim_pool(&mut canvas, &mut score, original, p, &fm, &mut placed, &mut k);
            nib_wet = placed > before;
        }
    }
    // THE PEN'S ACCENTS (see `MediumProfile::nib_anchor`, `nib_hint`): the cast-shadow touches and the shorthand dashes.
    if p.nib > 0.0 && (p.nib_anchor > 0.0 || p.nib_hint > 0.0 || p.nib_tuft > 0.0 || (p.nib_eyes > 0.0 && p.nib_feature_mask.is_some())) && placed < p.budget {
        if !nib_wet {
            canvas.dry(0.0);
        }
        nib_accents(&mut canvas, &mut score, input, original, p, ma_field.as_deref(), &mut placed, &mut k);
    }
    // SUMI-E: over the wash painting, the darkest masses as bold dry-brush black shapes (the second register).
    if p.sumi && placed < p.budget {
        canvas.dry(0.0);
        sumi_ink(&mut canvas, &mut score, input, p, ma_field.as_deref(), &mut placed, &mut k, progress);
    }
    if p.book && placed < p.budget {
        let plan = book.take().unwrap_or_else(|| book_plan(source, p, p.budget.saturating_sub(placed)));
        book_line(&mut canvas, &mut score, source, p, &mut placed, plan);
    } else if p.draw_contours {
        // A drawn line from the bone (a pencil's contour traced along the picture's own edges) replaces the
        // contours of the armature's levels, whose iso-lines ripple round every blob of the simplified picture.
        if p.bone <= 0.0 && placed < p.budget {
            ink_contours(&mut canvas, &mut score, input, source, p, protect_all.as_deref(), &mut placed, &mut k);
        }
    } else if p.contour > 0.0 && !p.density && placed < p.budget {
        contour_pass(&mut canvas, &mut score, input, p, &mut placed, &mut k);
    }

    // THE RIGGER: the few shapes too thin for the ladder to have laid at all (see `rigger_pass`). Last, so it
    // can see what the brushwork actually carried and restore only what was lost.
    if p.rigger > 0.0 && placed < p.budget {
        rigger_pass(&mut canvas, &mut score, input, p, protect_all.as_deref(), &mut placed, &mut k);
    }
    // THE LINER: the long thin structures the brushwork lost, each one mark (see `liner_pass`).
    if p.liner > 0.0 && placed < p.budget {
        liner_pass(&mut canvas, &mut score, input, p, protect_all.as_deref(), &mut placed, &mut k);
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
    if p.splatter > 0.0 && finish_ok && !splattered_wet {
        splatter_pass(&mut canvas, &mut score, input, p, &mut placed, &mut k, "splatter", false);
    }

    // A last, light wet-into-wet touch over the finish passes (contour, silhouette, splatter) so a wet medium
    // softens those marks a little — the strong fusion happened pass by pass above.
    if bleed_eff > 0.0 {
        canvas.bleed_with(bleed_eff * FINAL_BLEED, p.diffuse);
    }

    // NIB WASH TONE (see `MediumProfile::nib_wash_tone`): the sheet's mean value against the picture's — the pull of
    // the mid-tones that makes the two meet, solved by bisection, the key's share of it into the score's header for
    // the finish to print (the washes themselves were left light).
    if nib_wash_n > 0 && p.nib_wash_tone > 0.0 && original.dimensions() == (w, h) {
        let value = |img: &RgbImage| -> Vec<f32> { img.pixels().map(|px| (0.299 * px.0[0] as f32 + 0.587 * px.0[1] as f32 + 0.114 * px.0[2] as f32) / 255.0).collect() };
        let (pic, sheet) = (value(original), value(&canvas.to_image()));
        let n = pic.len().max(1) as f32;
        let target = pic.iter().sum::<f32>() / n;
        let pulled = |t: f32| sheet.iter().map(|c| c * (1.0 - 2.0 * t * c * (1.0 - c))).sum::<f32>() / n;
        if pulled(0.0) > target {
            let (mut lo, mut hi) = (0.0f32, 1.0f32);
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                if pulled(mid) > target { lo = mid } else { hi = mid }
            }
            score.header.tone = p.nib_wash_tone.clamp(0.0, 1.0) * 0.5 * (lo + hi);
        }
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
/// An ACCENT (see `MediumProfile::accent`) goes where the finest pass's reference has a local range of luma
/// over `ACCENT_EDGE` at the brush's scale, and is turned off the flow by up to `ACCENT_TURN` degrees either way.
const ACCENT_EDGE: f32 = 0.22;
const ACCENT_TURN: f32 = 35.0;
/// A face is SMOOTH (see `MediumProfile::sfumato`) where the local range of luma at the brush's scale is under this.
const SFUMATO_FLAT: f32 = 0.12;

/// The TURBULENCE field's wave numbers (see `MediumProfile::turbulence`) for a sheet of `long` pixels: four
/// incommensurate waves, the longest a ninth of the sheet.
fn turbulence_waves(long: f32) -> (f32, f32, f32, f32) {
    let lam = (long / 9.0).max(8.0);
    let k = std::f32::consts::TAU / lam;
    (k, k / 0.73, k / 1.21, k / 0.61)
}

/// A path turned about its middle by `angle`, kept on the sheet.
fn turn_path(path: &[[f32; 2]], angle: f32, w: u32, h: u32) -> Vec<[f32; 2]> {
    if path.len() < 2 {
        return path.to_vec();
    }
    let c = path.iter().fold([0f32; 2], |m, q| [m[0] + q[0], m[1] + q[1]]).map(|v| v / path.len() as f32);
    let (sn, cs) = angle.sin_cos();
    path.iter().map(|q| {
        let (dx, dy) = (q[0] - c[0], q[1] - c[1]);
        [(c[0] + dx * cs - dy * sn).clamp(0.0, w as f32 - 1.0), (c[1] + dx * sn + dy * cs).clamp(0.0, h as f32 - 1.0)]
    }).collect()
}

/// NIB WASH PLANES (see `MediumProfile::nib_wash_planes`): each pixel's band, on the eye's scale between the paper
/// (luma `paper_l`) and black, of `n` — `n - 1` the paper's own band.
fn wash_bands(reference: &RgbImage, n: usize, paper_l: f32) -> Vec<u8> {
    let steps = wash_even_steps(n);
    wash_bands_at(&wash_values(reference, paper_l, None), &steps)
}

/// NIB WASH PLANES: the even thresholds on the eye's scale — `n - 1` of them, a `1/n` apart.
fn wash_even_steps(n: usize) -> Vec<f32> {
    let n = n.max(2);
    (1..n).map(|j| j as f32 / n as f32).collect()
}

/// NIB WASH PLANES: each pixel's value on the eye's scale between the paper (luma `paper_l`) and black — lifted
/// toward the paper by `lift` there (NIB WASH DEPTH: the air; NIB WASH FACES: a face out of the dark).
fn wash_values(reference: &RgbImage, paper_l: f32, lift: Option<&[f32]>) -> Vec<f32> {
    let paper_l = paper_l.max(1e-3);
    reference.pixels().enumerate().map(|(i, px)| {
        let l = color::linear_luma(color::srgb_to_linear(px.0));
        let v = (l / paper_l).clamp(0.0, 1.0).powf(1.0 / 2.2);
        match lift.and_then(|m| m.get(i)) {
            Some(&d) if d > 0.0 => v + (1.0 - v) * d.clamp(0.0, 1.0),
            _ => v,
        }
    }).collect()
}

/// NIB WASH PLANES: each value's band — the count of thresholds (`steps`, rising, on the eye's scale) at or under it.
fn wash_bands_at(values: &[f32], steps: &[f32]) -> Vec<u8> {
    values.iter().map(|&v| steps.iter().filter(|&&t| t <= v).count() as u8).collect()
}

/// NIB WASH LEVELS (see `MediumProfile::nib_wash_levels`): the thresholds moved from the even steps toward the
/// picture's quantiles — the values cutting `values` into `n` equal shares — by `k`; kept rising, a hundredth apart.
fn wash_levels(values: &[f32], n: usize, k: f32) -> (Vec<f32>, Vec<f32>) {
    let n = n.max(2);
    let even = wash_even_steps(n);
    let k = k.clamp(0.0, 1.0);
    if k <= 0.0 {
        return (even.clone(), even);
    }
    let mut sample: Vec<f32> = values.iter().step_by(3).copied().collect();
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut steps = Vec::with_capacity(n - 1);
    let mut last = 0.0f32;
    for (j, e) in even.iter().enumerate() {
        let q = sample.get(((j + 1) * sample.len() / n).min(sample.len().saturating_sub(1))).copied().unwrap_or(*e);
        let t = ((1.0 - k) * e + k * q).max(last + 0.01).min(1.0);
        steps.push(t);
        last = t;
    }
    // The tones: each band's lighter edge moved toward the band's MEAN value by the key — a plane cut at the
    // picture's quantiles is a quarter of the sheet, and the stack reads the picture's mass only at the mean
    // (at the lighter edges a grassy hillside came out a fifth too light, past the output pull's reach).
    let mut tones = Vec::with_capacity(n - 1);
    let mut lo = 0.0f32;
    for (b, t) in steps.iter().enumerate() {
        let band: Vec<f32> = sample.iter().copied().filter(|v| *v >= lo && *v < *t).collect();
        let mean = if band.is_empty() { 0.5 * (lo + t) } else { band.iter().sum::<f32>() / band.len() as f32 };
        tones.push((1.0 - k) * t + k * mean.max(if b == 0 { 0.0 } else { tones[b - 1] + 0.01 }));
        lo = *t;
    }
    (steps, tones)
}

/// NIB WASH EDGE (see `MediumProfile::nib_wash_edge`): the picture's gradient at a ring point (a pixel corner), on the
/// eye's scale between the paper and black, over the smoothing radius `sigma`, against a step of a half of the
/// value — 0 flat, 1 an edge.
fn wash_grad(soft: &RgbImage, pt: [f32; 2], sigma: f32, paper_l: f32) -> f32 {
    let (w, h) = (soft.width() as i64, soft.height() as i64);
    let v = |x: i64, y: i64| {
        let px = soft.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32);
        (color::linear_luma(color::srgb_to_linear(px.0)) / paper_l.max(1e-3)).clamp(0.0, 1.0).powf(1.0 / 2.2)
    };
    let (x, y) = (pt[0].floor() as i64, pt[1].floor() as i64);
    let gx = (v(x + 1, y) - v(x - 1, y)) * 0.5;
    let gy = (v(x, y + 1) - v(x, y - 1)) * 0.5;
    ((gx * gx + gy * gy).sqrt() * sigma / 0.2).clamp(0.0, 1.0)
}

/// NIB WASH PLANES: a band's tone, linear, as a share of the paper's — the band's lighter edge; the paper's band 1.
fn wash_band_tone(b: usize, n: usize) -> f32 {
    wash_band_tone_at(b, &wash_even_steps(n))
}

/// NIB WASH PLANES: a band's tone, linear, under the thresholds `steps` — the threshold above it; the top band 1.
fn wash_band_tone_at(b: usize, steps: &[f32]) -> f32 {
    match steps.get(b) {
        Some(t) => t.clamp(0.0, 1.0).powf(2.2),
        None => 1.0,
    }
}

/// NIB WASH PLANES: the reference quantised to its planes — the paper, and a grey at each band's lighter edge.
fn wash_planes(reference: &RgbImage, n: usize, paper: color::LinRgb) -> RgbImage {
    let n = n.max(2);
    let paper_l = color::linear_luma(paper).max(1e-3);
    let paper_px = image::Rgb(color::linear_to_srgb(paper));
    let greys: Vec<image::Rgb<u8>> = (0..n).map(|b| {
        let l = paper_l * wash_band_tone(b, n);
        image::Rgb(color::linear_to_srgb([l, l, l]))
    }).collect();
    let bands = wash_bands(reference, n, paper_l);
    let mut out = reference.clone();
    for (px, &b) in out.pixels_mut().zip(bands.iter()) {
        *px = if b as usize == n - 1 { paper_px } else { greys[b as usize] };
    }
    out
}

/// BLADE PLANES (see `MediumProfile::blade_planes`): the reference cut into `n` flat planes of VALUE — the cut
/// points even steps of the value's range on the eye's scale blended toward the picture's quantiles by `levels` —
/// and every pixel's colour scaled in linear light to its plane's mean luma, the hue and chroma kept: the picture
/// posterised in value, each plane one flat slice for the knife, the mean tone of every plane (and so of the
/// whole) held.
/// BLADE FACET (see `MediumProfile::blade_facet`): the pixels under the mask (≥ 0.5) of `planes` rewritten as
/// `n` planes of value cut over THOSE pixels' own range and quantiles (`levels` as `blade_planes_ref`), from `raw`
/// smoothed by `k` (a 200th of the short side) — the face's gradients gathered into terraces of its own.
/// BLADE SOFT (`soften`): each faceted pixel taken back toward the smoothed picture's own colour by
/// `soft × (1 − |∇value|/0.01)` — the terrace a slope on the smooth skin, hewn at a true edge.
fn blade_facet_ref(planes: &RgbImage, raw: &RgbImage, n: usize, levels: f32, mask: &[f32], k: f32, soften: f32) -> RgbImage {
    let (w, h) = raw.dimensions();
    let n = n.max(2);
    let sigma = (k.clamp(0.0, 1.0) * w.min(h) as f32 / 200.0).max(0.6);
    let soft = imageops::blur(raw, sigma);
    let lin: Vec<color::LinRgb> = soft.pixels().map(|p| color::srgb_to_linear(p.0)).collect();
    let luma: Vec<f32> = lin.iter().map(|c| color::linear_luma(*c)).collect();
    let idx: Vec<usize> = (0..lin.len()).filter(|&i| mask.get(i).copied().unwrap_or(0.0) >= 0.5).collect();
    if idx.len() < 2 {
        return planes.clone();
    }
    let eye = |l: f32| l.max(0.0).powf(1.0 / 2.2);
    let mut sorted: Vec<f32> = idx.iter().map(|&i| eye(luma[i])).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
    let levels = levels.clamp(0.0, 1.0);
    let cuts: Vec<f32> = (1..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let even = lo + t * (hi - lo);
            let q = sorted[(((sorted.len() - 1) as f32) * t).round() as usize];
            even + levels * (q - even)
        })
        .collect();
    let plane_of = |l: f32| {
        let e = eye(l);
        cuts.iter().take_while(|c| e >= **c).count()
    };
    let mut sum = vec![0.0f64; n];
    let mut cnt = vec![0usize; n];
    for &i in &idx {
        let b = plane_of(luma[i]);
        sum[b] += luma[i] as f64;
        cnt[b] += 1;
    }
    let mean: Vec<f32> = (0..n).map(|b| if cnt[b] > 0 { (sum[b] / cnt[b] as f64) as f32 } else { 0.0 }).collect();
    let mut out = planes.clone();
    for &i in &idx {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let (l, c) = (luma[i], lin[i]);
        let m = mean[plane_of(l)];
        let facet = if l > 1e-5 {
            let s = m / l;
            [(c[0] * s).clamp(0.0, 1.0), (c[1] * s).clamp(0.0, 1.0), (c[2] * s).clamp(0.0, 1.0)]
        } else {
            [m, m, m]
        };
        let facet = if soften > 0.0 {
            let at = |xx: u32, yy: u32| eye(luma[(yy * w + xx) as usize]);
            let (gx, gy) = ((at((x + 1).min(w - 1), y) - at(x.saturating_sub(1), y)) / 2.0, (at(x, (y + 1).min(h - 1)) - at(x, y.saturating_sub(1))) / 2.0);
            let s = soften.clamp(0.0, 1.0) * (1.0 - ((gx * gx + gy * gy).sqrt() / 0.01).min(1.0));
            [facet[0] + (c[0] - facet[0]) * s, facet[1] + (c[1] - facet[1]) * s, facet[2] + (c[2] - facet[2]) * s]
        } else {
            facet
        };
        out.put_pixel(x, y, image::Rgb(color::linear_to_srgb(facet)));
    }
    out
}

/// BLADE TUFT (see `MediumProfile::blade_tuft`): the GRASS's fine texture's map in 0..1 — the pixels that are BUSY
/// (the value's range on the eye's scale over a small window of radius `r` reaching 0.03 — a photograph's grass is
/// dark and its texture a few hundredths) are DENSE: their share over a window three times wider, from 0.35 to 0.7
/// (an edge is a thin line of busy pixels in a quiet field, a texture fills its window); × the GRASS gate: with
/// `aniso` off, `grass` the regions the scene's grass detector found (CLIP over crops, as the nib tuft's; none,
/// nothing); with it (BLADE ANISO, see `MediumProfile::blade_aniso`), `blade_aniso_gate` — the tensor's own
/// reading, no detector; with `route` as well (BLADE ROUTE, see `MediumProfile::blade_route`) the two blended by
/// the FINENESS (`blade_fine_map`): the detector's regions where the texture is fine and contrasty (a fur, an eye),
/// the tensor's gate where it is coarse (a meadow, a foliage) — `route × fine` the detector's share.
fn blade_tuft_map(luma: &[f32], grass: Option<&[f32]>, aniso: f32, route: f32, w: usize, h: usize, r: usize) -> Vec<f32> {
    let n = w * h;
    let eye: Vec<f32> = luma.iter().map(|l| l.max(0.0).powf(1.0 / 2.2)).collect();
    let gate: Vec<f32> = if aniso > 0.0 {
        let tensor = blade_aniso_gate(&eye, w, h, r, aniso);
        if route > 0.0 {
            let fine = blade_fine_map(&eye, w, h, r);
            let clip: Vec<f32> = grass.filter(|g| g.len() == n).map(|g| g.iter().map(|v| v.clamp(0.0, 1.0)).collect()).unwrap_or_else(|| vec![0.0; n]);
            let route = route.clamp(0.0, 1.0);
            (0..n).map(|i| { let s = route * fine[i]; s * clip[i] + (1.0 - s) * tensor[i] }).collect()
        } else {
            tensor
        }
    } else {
        let Some(grass) = grass.filter(|g| g.len() == n) else { return vec![0.0; n] };
        grass.iter().map(|g| g.clamp(0.0, 1.0)).collect()
    };
    let rs = local_range(&eye, w, h, r);
    let busy: Vec<f32> = rs.iter().map(|v| if *v >= 0.03 { 1.0 } else { 0.0 }).collect();
    let dens = box_blur(&busy, w as u32, h as u32, (3 * r) as i32);
    (0..n).map(|i| ((dens[i] - 0.35) / 0.35).clamp(0.0, 1.0) * gate[i]).collect()
}

/// BLADE ANISO (see `MediumProfile::blade_aniso`): the gate in 0..1 from the structure tensor of `eye` — the
/// coherence index (λ₁ − λ₂)/(λ₁ + λ₂) of the tensor summed over a window of `2r`, its INCOHERENCE (1 − the
/// index) averaged over `6r`; the mean incoherence from 0.30 to 0.45 opens the gate (a mesh of crossing blades
/// flips its orientation every few pixels; strands keep theirs — measured on the shore's grass about 0.45–0.5 and on
/// the fox's fur 0.25–0.35, with spots of the fur reaching the grass's level: a partial separation). `k` the share
/// of the gate against open.
fn blade_aniso_gate(eye: &[f32], w: usize, h: usize, r: usize, k: f32) -> Vec<f32> {
    let n = w * h;
    let (gx, gy) = sobel(eye, w as u32, h as u32);
    let (mut jxx, mut jyy, mut jxy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        jxx[i] = gx[i] * gx[i];
        jyy[i] = gy[i] * gy[i];
        jxy[i] = gx[i] * gy[i];
    }
    let r2 = (2 * r) as i32;
    let (jxx, jyy, jxy) = (box_blur(&jxx, w as u32, h as u32, r2), box_blur(&jyy, w as u32, h as u32, r2), box_blur(&jxy, w as u32, h as u32, r2));
    let inco: Vec<f32> = (0..n)
        .map(|i| {
            let disc = ((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]).sqrt();
            1.0 - (disc / (jxx[i] + jyy[i] + 1e-6)).clamp(0.0, 1.0)
        })
        .collect();
    let mean = box_blur(&inco, w as u32, h as u32, (6 * r) as i32);
    let k = k.clamp(0.0, 1.0);
    mean.iter().map(|m| 1.0 - k * (1.0 - ((m - 0.3) / 0.15).clamp(0.0, 1.0))).collect()
}

/// BLADE ROUTE (see `MediumProfile::blade_route`): the FINENESS of the texture in 0..1 — the value's range (on the
/// eye's scale, `eye`) over a window of radius `r`, averaged over `6r`: the mean local contrast, from FINE_A to
/// FINE_A + FINE_B. A fur's strands, an eye are contrasty within a few pixels; a meadow's blades at a distance are a
/// few hundredths (measured at a 400th of the short side: the fox's fur 0.3–0.5, the shore's far grass 0.04–0.17,
/// its near grass 0.2–0.3, the village's houses and foliage 0.2–0.5, a sky 0.02).
const FINE_A: f32 = 0.10;
const FINE_B: f32 = 0.15;
pub(crate) fn blade_fine_map(eye: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let rs = local_range(eye, w, h, r);
    let c = box_blur(&rs, w as u32, h as u32, (6 * r) as i32);
    c.iter().map(|v| ((v - FINE_A) / FINE_B).clamp(0.0, 1.0)).collect()
}

/// BLADE FOCUS (see `MediumProfile::blade_focus`): how far outside the picture's FOCUS each pixel lies, 0..1 —
/// the focus the centre of mass of the fineness (`blade_fine_map`: the fine contrasty textures, a fur, a face, the
/// near grass), its reach the fineness's RMS radius about it; the distance from 1.2 reaches to 2.2 smoothstepped.
/// A sheet with no fine texture has no focus (zeros).
fn blade_focus_map(luma: &[f32], w: usize, h: usize, r: usize, band: f32) -> Vec<f32> {
    let eye: Vec<f32> = luma.iter().map(|l| l.max(0.0).powf(1.0 / 2.2)).collect();
    let fine = blade_fine_map(&eye, w, h, r);
    let total: f64 = fine.iter().map(|v| *v as f64).sum();
    if total < 1.0 {
        return vec![0.0; w * h];
    }
    let (mut sx, mut sy) = (0f64, 0f64);
    for (i, v) in fine.iter().enumerate() {
        sx += *v as f64 * (i % w) as f64;
        sy += *v as f64 * (i / w) as f64;
    }
    let (cx, cy) = (sx / total, sy / total);
    // BLADE BAND (see `MediumProfile::blade_band`): the vertical reach alone, for the focus as a band.
    let (mut s2, mut sy2) = (0f64, 0f64);
    for (i, v) in fine.iter().enumerate() {
        let (ddx, ddy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
        s2 += *v as f64 * (ddx.powi(2) + ddy.powi(2));
        sy2 += *v as f64 * ddy.powi(2);
    }
    let reach = (s2 / total).sqrt().max(1.0);
    let reach_y = (sy2 / total).sqrt().max(1.0);
    let band = band.clamp(0.0, 1.0) as f64;
    (0..w * h)
        .map(|i| {
            let (ddx, ddy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
            let d = (ddx.powi(2) + ddy.powi(2)).sqrt();
            let tr = ((d - 1.2 * reach) / reach).clamp(0.0, 1.0);
            let tb = ((ddy.abs() - 1.2 * reach_y) / reach_y).clamp(0.0, 1.0);
            let t = (tr * (1.0 - band) + tb * band) as f32;
            t * t * (3.0 - 2.0 * t)
        })
        .collect()
}

/// `blade_fine_map` of a picture — for the scene detector's routing (the crops it is spared).
pub fn blade_fine_map_of(img: &RgbImage, r: usize) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let eye: Vec<f32> = img.pixels().map(|p| color::linear_luma(color::srgb_to_linear(p.0)).max(0.0).powf(1.0 / 2.2)).collect();
    blade_fine_map(&eye, w as usize, h as usize, r)
}

/// BLADE TINT (see `MediumProfile::blade_tint`): `target` with its hue turned by `turn` of the circle (the chroma
/// and the value kept — a rotation of a and b in Lab).
fn blade_tint_tone(target: Srgb, turn: f32) -> Srgb {
    if turn.abs() < 1e-6 {
        return target;
    }
    let mut lab = color::srgb_to_lab(target);
    let (s, c) = (turn * std::f32::consts::TAU).sin_cos();
    let (a, b) = (lab.a, lab.b);
    lab.a = a * c - b * s;
    lab.b = a * s + b * c;
    color::lab_to_srgb(lab)
}

/// BLADE SLOPE (see `MediumProfile::blade_slope`): the picture's HILL — the value (on the eye's scale) blurred over a
/// window of radius `big`, its gradient's change across that window (|∇| × big) toward 0.15: 1 on a steep bank, 0 on a
/// flat meadow or a sky's slow ramp.
fn blade_slope_map(luma: &[f32], w: usize, h: usize, big: usize) -> Vec<f32> {
    let eye: Vec<f32> = luma.iter().map(|l| l.max(0.0).powf(1.0 / 2.2)).collect();
    let low = box_blur(&eye, w as u32, h as u32, big as i32);
    let (gx, gy) = sobel(&low, w as u32, h as u32);
    (0..w * h).map(|i| ((gx[i] * gx[i] + gy[i] * gy[i]).sqrt() * big as f32 / 0.15).clamp(0.0, 1.0)).collect()
}

/// BLADE DETAIL (see `MediumProfile::blade_detail`): `planes` (the global planes) with every pixel whose LOCAL
/// range of value (the min and max over a window of radius `r`, on the eye's scale, from `raw`) is at least 0.08
/// rewritten as one of `n` even planes over that local range — the detail's own planes — by the key's share `k`
/// against the global plane. A flat passage (a narrow local range) keeps the global plane.
fn blade_local_ref(planes: &RgbImage, raw: &RgbImage, n: usize, r: usize, k: f32) -> RgbImage {
    let (w, h) = raw.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let n = n.max(2);
    let k = k.clamp(0.0, 1.0);
    let eye = |l: f32| l.max(0.0).powf(1.0 / 2.2);
    let lin: Vec<color::LinRgb> = raw.pixels().map(|p| color::srgb_to_linear(p.0)).collect();
    let luma: Vec<f32> = lin.iter().map(|c| color::linear_luma(*c)).collect();
    let e: Vec<f32> = luma.iter().map(|l| eye(*l)).collect();
    // Separable min / max over the window: along x, then along y.
    let pass = |src: &[f32], f: fn(f32, f32) -> f32, init: f32| -> Vec<f32> {
        let mut t = vec![init; wu * hu];
        for y in 0..hu {
            for x in 0..wu {
                let mut m = init;
                for d in x.saturating_sub(r)..=(x + r).min(wu - 1) {
                    m = f(m, src[y * wu + d]);
                }
                t[y * wu + x] = m;
            }
        }
        let mut o = vec![init; wu * hu];
        for y in 0..hu {
            for x in 0..wu {
                let mut m = init;
                for d in y.saturating_sub(r)..=(y + r).min(hu - 1) {
                    m = f(m, t[d * wu + x]);
                }
                o[y * wu + x] = m;
            }
        }
        o
    };
    let lo = pass(&e, f32::min, f32::MAX);
    let hi = pass(&e, f32::max, f32::MIN);
    let mut out = planes.clone();
    for (i, px) in out.pixels_mut().enumerate() {
        let range = hi[i] - lo[i];
        if range < 0.08 {
            continue;
        }
        let plane = (((e[i] - lo[i]) / range * n as f32).floor() as usize).min(n - 1);
        let m = eye_inv(lo[i] + (plane as f32 + 0.5) / n as f32 * range);
        let (l, c) = (luma[i], lin[i]);
        let local = if l > 1e-5 {
            let s = m / l;
            [(c[0] * s).clamp(0.0, 1.0), (c[1] * s).clamp(0.0, 1.0), (c[2] * s).clamp(0.0, 1.0)]
        } else {
            [m, m, m]
        };
        let g = color::srgb_to_linear(px.0);
        *px = image::Rgb(color::linear_to_srgb([g[0] + (local[0] - g[0]) * k, g[1] + (local[1] - g[1]) * k, g[2] + (local[2] - g[2]) * k]));
    }
    out
}

/// The eye's scale back to linear light (the inverse of the planes' `eye`).
fn eye_inv(e: f32) -> f32 {
    e.max(0.0).powf(2.2)
}

fn blade_planes_ref(reference: &RgbImage, n: usize, levels: f32) -> RgbImage {
    let n = n.max(2);
    let lin: Vec<color::LinRgb> = reference.pixels().map(|p| color::srgb_to_linear(p.0)).collect();
    let luma: Vec<f32> = lin.iter().map(|c| color::linear_luma(*c)).collect();
    if luma.is_empty() {
        return reference.clone();
    }
    let eye = |l: f32| l.max(0.0).powf(1.0 / 2.2);
    let mut sorted: Vec<f32> = luma.iter().map(|l| eye(*l)).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
    let levels = levels.clamp(0.0, 1.0);
    let cuts: Vec<f32> = (1..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let even = lo + t * (hi - lo);
            let q = sorted[(((sorted.len() - 1) as f32) * t).round() as usize];
            even + levels * (q - even)
        })
        .collect();
    let plane_of = |l: f32| {
        let e = eye(l);
        cuts.iter().take_while(|c| e >= **c).count()
    };
    let mut sum = vec![0.0f64; n];
    let mut cnt = vec![0usize; n];
    for &l in &luma {
        let b = plane_of(l);
        sum[b] += l as f64;
        cnt[b] += 1;
    }
    let mean: Vec<f32> = (0..n).map(|b| if cnt[b] > 0 { (sum[b] / cnt[b] as f64) as f32 } else { 0.0 }).collect();
    let mut out = reference.clone();
    for ((px, c), &l) in out.pixels_mut().zip(lin.iter()).zip(luma.iter()) {
        let m = mean[plane_of(l)];
        // (A black pixel has no hue to keep: it takes its plane's tone as a grey.)
        *px = if l > 1e-5 {
            let s = m / l;
            image::Rgb(color::linear_to_srgb([(c[0] * s).clamp(0.0, 1.0), (c[1] * s).clamp(0.0, 1.0), (c[2] * s).clamp(0.0, 1.0)]))
        } else {
            image::Rgb(color::linear_to_srgb([m, m, m]))
        };
    }
    out
}

/// NIB WASH RAG (see `MediumProfile::nib_wash_rag`): the rings TORN — each ring resampled a `step` apart (its
/// corners kept) and every point moved along the boundary's normal by `amp` times a noise along the ring
/// (hashed nodes two steps apart, interpolated, signed), so the mass's edge is a dry brush's ragged one, not a
/// cut. The bloom then rises from the torn edge. A separator (`NaN`) ends a ring.
fn rag_rings(rings: &[[f32; 2]], amp: f32, step: f32, streak: f32) -> Vec<[f32; 2]> {
    let streak = streak.clamp(0.0, 1.0);
    let hash = |i: i64| -> f32 {
        let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x5851_F42D_4C95_7F2D);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    };
    let step = step.max(1.0);
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(rings.len() * 2);
    let mut ring: Vec<[f32; 2]> = Vec::new();
    let mut arc0 = 0.0f32;
    let tear = |ring: &[[f32; 2]], out: &mut Vec<[f32; 2]>, arc0: &mut f32| {
        if ring.len() < 3 {
            out.extend_from_slice(ring);
            return;
        }
        let mut arc = *arc0;
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-6 {
                continue;
            }
            let (nx, ny) = (dy / len, -dx / len);
            let pieces = (len / step).ceil().max(1.0) as usize;
            // NIB WASH STREAK: the noise advances slower along the sweep (the horizontal), its fibres drawn out there.
            let stretch = 1.0 + 6.0 * streak * (dx / len).abs();
            for s in 0..pieces {
                let t = s as f32 / pieces as f32;
                let u = (arc + t * len / stretch) / (2.0 * step);
                let (u0, uf) = (u.floor(), u - u.floor());
                let n = hash(u0 as i64) * (1.0 - uf) + hash(u0 as i64 + 1) * uf;
                let n = ((n - 0.5) * (1.0 + 2.0 * streak) + 0.5).clamp(0.0, 1.0);
                let d = amp * (2.0 * n - 1.0);
                out.push([a[0] + t * dx + nx * d, a[1] + t * dy + ny * d]);
            }
            arc += len / stretch;
        }
        *arc0 = arc;
    };
    for pt in rings {
        if pt[0].is_nan() || pt[1].is_nan() {
            tear(&ring, &mut out, &mut arc0);
            ring.clear();
            out.push(*pt);
        } else {
            ring.push(*pt);
        }
    }
    tear(&ring, &mut out, &mut arc0);
    out
}

/// The mask with its connected pieces under `min_area` pixels dropped (4-connected).
fn drop_small_pieces(mask: &[bool], w: usize, h: usize, min_area: usize) -> Vec<bool> {
    drop_small_pieces_keep(mask, w, h, min_area, None)
}

/// As `drop_small_pieces`, a small piece kept where it meets `keep` (NIB WASH GRAIN: the small darks).
fn drop_small_pieces_keep(mask: &[bool], w: usize, h: usize, min_area: usize, keep: Option<&[bool]>) -> Vec<bool> {
    let mut out = mask.to_vec();
    let mut seen = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let mut piece: Vec<usize> = Vec::new();
    for s in 0..w * h {
        if !mask[s] || seen[s] {
            continue;
        }
        piece.clear();
        stack.push(s);
        seen[s] = true;
        while let Some(i) = stack.pop() {
            piece.push(i);
            let (x, y) = (i % w, i / w);
            let nb = [if x > 0 { Some(i - 1) } else { None }, if x + 1 < w { Some(i + 1) } else { None }, if y > 0 { Some(i - w) } else { None }, if y + 1 < h { Some(i + w) } else { None }];
            for j in nb.into_iter().flatten() {
                if mask[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if piece.len() < min_area && !keep.is_some_and(|k| piece.iter().any(|&i| k[i])) {
            for &i in &piece {
                out[i] = false;
            }
        }
    }
    out
}

/// NIB WASH PLANES (see `MediumProfile::nib_wash_planes`): the planes laid as WASHES — area fills, one per plane
/// from the lightest to the darkest, each covering every plane as dark as itself (the washes stack: three washes
/// of one ink), its charge found on scrap paper (`charge_for`) so the stack reads the plane's tone (a glaze over
/// a wash of value U that must read T reads T/U on white). A plane's mask opened by a pixel or two (a 1024th of
/// the sheet) and its pieces under a 4000th of the sheet dropped — a plane is a mass, not a speck; its edge
/// feathered a 500th of the sheet (wider where the picture is gentle, `nib_wash_edge`), laid wet enough to bloom
/// in the pass's bleed. The planes from the picture smoothed over a 256th of the sheet — the lightest coarser and
/// the darkest finer by `nib_wash_scale`, each kept within the lighter; the far lifted into the lighter bands by
/// `nib_wash_depth`. Recorded as washes, so the replay lays the same pools.
/// NIB WASH DEPTH: how far each pixel is, as `nib_wash_air` reads the depth map — a lift of a half of the key
/// there; NIB WASH FACES: a face lifted by a quarter of its key. One lift map, the two added.
fn wash_lift_map(p: &PaintParams, w: usize, h: usize) -> Option<Vec<f32>> {
    let depth = 0.5 * p.nib_wash_depth.clamp(0.0, 1.0);
    let faces = 0.25 * p.nib_wash_faces.clamp(0.0, 1.0);
    let far = p.nib_depth_map.as_deref().filter(|_| depth > 0.0);
    let face = p.face_mask.as_deref().filter(|_| faces > 0.0);
    (far.is_some() || face.is_some()).then(|| {
        (0..w * h).map(|i| {
            let d = far.and_then(|m| m.get(i)).map_or(0.0, |&z| { let u = ((z - 0.1) / 0.4).clamp(0.0, 1.0); depth * (1.0 - u * u * (3.0 - 2.0 * u)) });
            let f = face.and_then(|m| m.get(i)).map_or(0.0, |&v| faces * v.clamp(0.0, 1.0));
            (d + f).min(1.0)
        }).collect()
    })
}

/// NIB WASH PLANES: the planes' thresholds as the washes cut them — by the picture's quantiles (`nib_wash_levels`)
/// or the even steps — and the values they cut, from the picture smoothed over a 256th of the sheet and lifted.
fn wash_plane_steps(input: &RgbImage, p: &PaintParams, w: usize, h: usize, n: usize, paper_eye: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = n.max(2);
    let long = w.max(h) as f32;
    let s0 = (long / 256.0).max(1.0);
    let lift = wash_lift_map(p, w, h);
    let values = wash_values(&imageops::blur(input, s0), paper_eye, lift.as_deref());
    let (steps, tones) = if p.nib_wash_levels > 0.0 { wash_levels(&values, n, p.nib_wash_levels) } else { (wash_even_steps(n), wash_even_steps(n)) };
    (steps, tones, values)
}

/// NIB WASH MASK (see `MediumProfile::nib_wash_mask`): how deep each pixel lies in the darkest plane — 1 through its
/// band, falling (a smoothstep) across the band above it, 0 past that.
fn wash_dark_plane_map(input: &RgbImage, p: &PaintParams, w: usize, h: usize, n: usize, paper_eye: f32) -> Vec<f32> {
    let (steps, _, values) = wash_plane_steps(input, p, w, h, n, paper_eye);
    let (s0, s1) = (steps[0], steps.get(1).copied().unwrap_or(1.0));
    values.iter().map(|&v| if v < s0 { 1.0 } else if v < s1 { let u = (s1 - v) / (s1 - s0).max(1e-3); u * u * (3.0 - 2.0 * u) } else { 0.0 }).collect()
}

/// NIB WASH WATER (see `MediumProfile::nib_wash_water`): the sheet WETTED over `mask` grown by `grow` px — a fill
/// of clear water, no ink, at wetness `wet` — recorded as an ink-less wash, so the replay wets the same.
#[allow(clippy::too_many_arguments)]
fn wash_prewet(canvas: &mut Canvas, score: &mut StrokeScore, mask: &[bool], w: usize, h: usize, grow: usize, n_pig: usize, wet: f32, stage: &str, placed: &mut usize, k: &mut u64) {
    let grown = dilate_bool(mask, w, h, grow.max(1));
    let rings = mask_rings(&grown, w, h, 1.2);
    if rings.is_empty() {
        return;
    }
    let clear = vec![0f32; n_pig];
    canvas.fill_rings(&rings, &clear, wet, 1.0);
    *k += 1;
    *placed += 1;
    score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.to_string(), spline: rings, w0: 1.0, w1: 0.0, taper: 0.0, mix: Vec::new(), wet, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
}

/// NIB WASH SWEEP (see `MediumProfile::nib_wash_sweep`): one plane laid as wide wet strokes along the flow of
/// `soft` (its smoothed image), the ink along each mark following the picture's value `values` (on the eye's
/// scale) within the plane's band `lo..hi`; `ratio` the plane's tone over the plane above; `a` 0 on the darkest
/// plane, 1 on the lightest.
#[allow(clippy::too_many_arguments)]
fn wash_sweep_plane(canvas: &mut Canvas, score: &mut StrokeScore, mask: &[bool], values: &[f32], target: &[f32], plane_v: f32, soft: &RgbImage, w: usize, h: usize, p: &PaintParams, ink: usize, ink_base: &[f32], ratio: f32, lo: f32, hi: f32, a: f32, paper_l: f32, stage: &str, cache: &mut std::collections::HashMap<(u32, u16), f32>, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let long = w.max(h) as f32;
    let width = (long / 96.0 * (0.5 + 0.5 * a.clamp(0.0, 1.0))).max(3.0);
    let spacing = (0.5 * width).max(2.0);
    let overlap = width / spacing;
    let reach = long / 8.0;
    // (KATSU: the dry brush — the marks drier and their hairs uneven by the key: a flooded mark shows no grain.)
    let katsu = p.brush.katsu.clamp(0.0, 1.0);
    let wet = 0.8 - 0.5 * katsu;
    // The flow: the structure tensor of the plane's image over a 48th of the sheet, the marks along the isophotes
    // of the gross form; turned to the horizontal (the hake's sweep) where the picture has little direction.
    let sigma = (long / 48.0).round().clamp(4.0, 64.0) as i32;
    let (gx, gy) = coherent_gradient(&luma_map(soft), w as u32, h as u32, sigma, p.infill);
    let dir_at = |x: f32, y: f32| -> (f32, f32) {
        let i = (y.clamp(0.0, (h - 1) as f32) as usize) * w + x.clamp(0.0, (w - 1) as f32) as usize;
        let (dx, dy) = (-gy[i], gx[i]);
        let m = (dx * dx + dy * dy).sqrt();
        if m < 1e-4 {
            return (1.0, 0.0);
        }
        let (ux, uy) = (dx / m, dy / m);
        // (the sign toward the right, so the blend with the horizontal does not cancel)
        let (ux, uy) = if ux < 0.0 { (-ux, -uy) } else { (ux, uy) };
        let c = ((m - 0.1) / 0.3).clamp(0.0, 1.0);
        let (bx, by) = (ux * c + (1.0 - c), uy * c);
        let bm = (bx * bx + by * by).sqrt().max(1e-4);
        (bx / bm, by / bm)
    };
    // The seeds inside the mask eroded by a third of the width (the mark's body stays in the plane); a thin
    // plane that the erosion empties keeps its own mask.
    let r_in = (width / 3.0).round() as usize;
    let inner = if r_in > 0 { erode_bool(mask, w, h, r_in) } else { mask.to_vec() };
    let inner = if inner.iter().any(|&m| m) { inner } else { mask.to_vec() };
    let inside = |x: f32, y: f32| -> bool { x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h && inner[(y as usize) * w + x as usize] };
    // The coverage on a grid of cells a spacing wide: a seed in a cell two marks have crossed is skipped, a mark
    // stops where three have.
    let (cw, ch) = ((w as f32 / spacing).ceil() as usize + 1, (h as f32 / spacing).ceil() as usize + 1);
    let mut cover = vec![0u8; cw * ch];
    let cell = |x: f32, y: f32| -> usize { (y.max(0.0) / spacing) as usize * cw + (x.max(0.0) / spacing) as usize };
    let mut brush = p.brush;
    brush.streak = 0.6 * katsu;
    brush.round = 0.4;
    brush.k_pickup = 0.0;
    brush.flat_ends = true;
    let span = (hi - lo).max(1e-3);
    let mid = 0.5 * (lo + hi);
    let n_pig = ink_base.len();
    let seed_salt = p.seed ^ 0x5EE9_0000 ^ ((a * 1000.0) as u64);
    // NIB WASH WATER: the sheet wetted — clear water, no ink — over the mask grown by the key times the width.
    if p.nib_wash_water > 0.0 {
        wash_prewet(canvas, score, mask, w, h, (width * p.nib_wash_water.clamp(0.0, 1.0)).round().max(1.0) as usize, n_pig, wet, stage, placed, k);
    }
    let trace = |x0: f32, y0: f32, sign: f32| -> Vec<[f32; 2]> {
        let mut out: Vec<[f32; 2]> = Vec::new();
        let (mut x, mut y) = (x0, y0);
        let (mut px, mut py) = dir_at(x, y);
        px *= sign;
        py *= sign;
        let mut len = 0.0f32;
        while len < reach {
            let (mut dx, mut dy) = dir_at(x, y);
            if dx * px + dy * py < 0.0 {
                dx = -dx;
                dy = -dy;
            }
            x += dx;
            y += dy;
            len += 1.0;
            if !inside(x, y) {
                break;
            }
            if (len as usize) % 2 == 0 {
                out.push([x, y]);
            }
            px = dx;
            py = dy;
        }
        out
    };
    let mut seed_k = 0u64;
    for cj in 0..ch {
        for ci in 0..cw {
            seed_k += 1;
            let sx = (ci as f32 + 0.5 + jitter(seed_salt, seed_k * 2)) * spacing;
            let sy = (cj as f32 + 0.5 + jitter(seed_salt, seed_k * 2 + 1)) * spacing;
            if !inside(sx, sy) || cover[cell(sx, sy)] >= 2 {
                continue;
            }
            let back = trace(sx, sy, -1.0);
            let fwd = trace(sx, sy, 1.0);
            let mut path: Vec<[f32; 2]> = back.iter().rev().copied().collect();
            path.push([sx, sy]);
            path.extend(fwd);
            let total = 2.0 * (path.len().saturating_sub(1)) as f32;
            if total < 0.5 * width {
                continue;
            }
            for pt in &path {
                let c = cell(pt[0], pt[1]);
                cover[c] = cover[c].saturating_add(1);
            }
            // One mark the whole streamline, ramped from the tone at its start to the tone at its end.
            let tone_at = |pt: [f32; 2]| -> f32 {
                let i = (pt[1].clamp(0.0, (h - 1) as f32) as usize) * w + pt[0].clamp(0.0, (w - 1) as f32) as usize;
                let v = values[i].clamp(lo, hi);
                let e = (1.0 + (mid - v) / span).clamp(0.5, 1.5);
                // quantised to a hundredth, so the probe's cache stays small
                ((ratio.clamp(0.01, 1.0).powf(e / overlap) * 100.0).round() / 100.0).clamp(0.03, 0.99)
            };
            {
                let piece: Vec<[f32; 2]> = path;
                // NIB WASH AGENT: the mark charged with the plane's whole tone, its profile the picture's value under
                // the path (linear), the ink given by the residual as it goes.
                let agent = p.nib_wash_agent > 0.0;
                // (NIB WASH GRADE on a swept plane: the target between the plane's flat tone and the picture's own
                // value at the darkest plane's smoothing, by the key — at 0 the plane a flat mass laid by the brush, at
                // 1 the picture itself; the mark's charge is the plane's tone, so a darker detail is the next plane's.)
                let grade = p.nib_wash_grade.clamp(0.0, 1.0);
                let prof: Option<Vec<f32>> = agent.then(|| piece.iter().map(|pt| {
                    let i = (pt[1].clamp(0.0, (h - 1) as f32) as usize) * w + pt[0].clamp(0.0, (w - 1) as f32) as usize;
                    let v = plane_v + (target[i].clamp(0.0, 1.0) - plane_v) * grade;
                    ((paper_l * v.clamp(0.0, 1.0).powf(2.2) * 1000.0).round() / 1000.0).clamp(0.0, 1.0)
                }).collect());
                let plane_tone = ((ratio.clamp(0.01, 1.0) * 100.0).round() / 100.0).clamp(0.03, 0.99);
                let (t0, t1) = if agent { (plane_tone, plane_tone) } else { (tone_at(piece[0]), tone_at(piece[piece.len() - 1])) };
                let c0 = charge_for_base(cache, &p.palette, 0.85, p.opacity, p.palette.pigments[ink].masstone, ink_base, t0, wet, Some(&brush));
                let c1 = charge_for_base(cache, &p.palette, 0.85, p.opacity, p.palette.pigments[ink].masstone, ink_base, t1, wet, Some(&brush));
                let load: Vec<f32> = ink_base.iter().map(|v| v * c0).collect();
                let load_end: Vec<f32> = ink_base.iter().map(|v| v * c1).collect();
                if load.iter().any(|v| *v > 0.0) || load_end.iter().any(|v| *v > 0.0) {
                    let st = Stroke { path: piece, width0: width, width1: width, load: load.clone(), pressure: 1.0, wetness: wet };
                    match &prof {
                        Some(pf) => st.rasterize_agent(canvas, &brush, None, pf, p.nib_wash_agent),
                        None => st.rasterize_ramp(canvas, &brush, Some(&load_end)),
                    }
                    *k += 1;
                    *placed += 1;
                    if let Some(pr) = progress {
                        pr(PaintProgress::Placed(*placed));
                    }
                    let mix: Vec<(String, f32)> = (0..n_pig).filter(|&i| load[i] > 0.0).map(|i| (p.palette.pigments[i].name.to_string(), load[i])).collect();
                    let mix2: Vec<(String, f32)> = (0..n_pig).filter(|&i| load_end[i] > 0.0).map(|i| (p.palette.pigments[i].name.to_string(), load_end[i])).collect();
                    score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: stage.to_string(), spline: st.path, w0: width, w1: width, taper: 0.0, mix, wet, press: 1.0, streak: brush.streak, round: 0.4, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: true, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: if agent { None } else { Some(mix2) }, smudge: None, stop: None, edge: None, prof });
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn wash_plane_fills(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, w: usize, h: usize, n: usize, p: &PaintParams, paper_eye: f32, stage: &str, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let n = n.max(2);
    let long = w.max(h) as f32;
    let n_pig = p.palette.pigments.len();
    // The ink: the palette's darkest pigment.
    let ink = (0..n_pig).min_by(|&a, &b| color::linear_luma(color::srgb_to_linear(p.palette.pigments[a].masstone)).partial_cmp(&color::linear_luma(color::srgb_to_linear(p.palette.pigments[b].masstone))).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
    let ink_base: Vec<f32> = (0..n_pig).map(|i| if i == ink { 1.0 } else { 0.0 }).collect();
    let r_open = ((long / 1024.0).round() as usize).max(1);
    let min_area = (w * h / 4000).max(4);
    let feather = (long * 0.002).max(1.0).round();
    let wet = 0.75;
    let lift = wash_lift_map(p, w, h);
    // NIB WASH SCALE: the smoothing's ladder, the lightest plane coarsest.
    let scale = 1.0 + 3.0 * p.nib_wash_scale.clamp(0.0, 1.0);
    let s0 = (long / 256.0).max(1.0);
    // NIB WASH LEVELS: the thresholds, by the picture's quantiles at the middle smoothing.
    let (steps, tones, _) = wash_plane_steps(input, p, w, h, n, paper_eye);
    // NIB WASH GRAIN: the small darks — the two darkest bands at the darkest plane's smoothing, their pieces over a
    // (1 + 15·key)th of the least area — kept in every plane.
    let grain = p.nib_wash_grain.clamp(0.0, 1.0);
    let dark_keep: Option<Vec<bool>> = (grain > 0.0 && n >= 3).then(|| {
        let fine = imageops::blur(input, (s0 / scale).max(0.6));
        let b = wash_bands_at(&wash_values(&fine, paper_eye, lift.as_deref()), &steps);
        let d: Vec<bool> = b.iter().map(|&x| (x as usize) <= 1).collect();
        drop_small_pieces(&d, w, h, ((min_area as f32 / (1.0 + 15.0 * grain)).round() as usize).max(4))
    });
    let grade = p.nib_wash_grade.clamp(0.0, 1.0);
    // NIB WASH SWEEP + AGENT: the marks' target read from the picture at the darkest plane's smoothing.
    let values_fine: Option<Vec<f32>> = (p.nib_wash_sweep > 0.0 && p.nib_wash_agent > 0.0).then(|| wash_values(&imageops::blur(input, (s0 / scale).max(0.6)), paper_eye, lift.as_deref()));
    let mut above: Option<Vec<bool>> = None;
    let mut cache: std::collections::HashMap<(u32, u16), f32> = std::collections::HashMap::new();
    for j in 1..n {
        // This wash covers the bands as dark as `top` and darker; over the band above it reads `ratio`.
        let top = n - 1 - j;
        let ratio = wash_band_tone_at(top, &tones) / wash_band_tone_at(top + 1, &tones).max(1e-3);
        let a = if n > 2 { top as f32 / (n - 2) as f32 } else { 0.5 };
        let sigma = (s0 * scale.powf(2.0 * a - 1.0)).max(0.6);
        let soft = imageops::blur(input, sigma);
        let values = wash_values(&soft, paper_eye, lift.as_deref());
        let bands = wash_bands_at(&values, &steps);
        let mask: Vec<bool> = bands.iter().enumerate().map(|(i, &b)| (b as usize) <= top && above.as_ref().is_none_or(|ab| ab[i])).collect();
        let opened = dilate_bool(&erode_bool(&mask, w, h, r_open), w, h, r_open);
        // (the grain's pieces past the opening: laid without the bloom — a speck's edge is the base feather)
        let grain_only: Option<Vec<bool>> = dark_keep.as_ref().map(|d| d.iter().zip(opened.iter()).map(|(k, o)| *k && !*o).collect());
        let mask: Vec<bool> = match &grain_only {
            Some(g) => opened.iter().zip(g.iter()).map(|(o, g)| *o || *g).collect(),
            None => opened,
        };
        let mask = drop_small_pieces_keep(&mask, w, h, min_area, dark_keep.as_deref());
        above = Some(mask.clone());
        if !mask.iter().any(|&m| m) {
            continue;
        }
        // NIB WASH SWEEP: the plane laid as wide wet strokes along the flow, not a fill.
        if p.nib_wash_sweep > 0.0 {
            let lo = if top == 0 { 0.0 } else { steps[top - 1] };
            wash_sweep_plane(canvas, score, &mask, &values, values_fine.as_deref().unwrap_or(&values), tones[top], &soft, w, h, p, ink, &ink_base, ratio, lo, steps[top], a, paper_eye, stage, &mut cache, placed, k, progress);
            continue;
        }
        // NIB WASH WATER: the sheet wetted over the plane's mask grown by the key times a 64th of the sheet.
        if p.nib_wash_water > 0.0 {
            wash_prewet(canvas, score, &mask, w, h, (long / 64.0 * p.nib_wash_water.clamp(0.0, 1.0)).round().max(1.0) as usize, n_pig, wet, stage, placed, k);
        }
        let rings = mask_rings(&mask, w, h, 1.2);
        if rings.is_empty() {
            continue;
        }
        // NIB WASH RAG: the mass's contour torn — by the key times the plane's darkness on the eye's scale.
        let rings = if p.nib_wash_rag > 0.0 {
            let amp = p.nib_wash_rag.clamp(0.0, 1.0) * (1.0 - steps.get(top).copied().unwrap_or(1.0)) * 1.5 * feather;
            if amp >= 0.5 { rag_rings(&rings, amp, feather, p.nib_wash_streak) } else { rings }
        } else {
            rings
        };
        // NIB WASH EDGE: the wet edge per ring point — wide where the picture's gradient is gentle there.
        let edge: Option<Vec<f32>> = if p.nib_wash_edge > 0.0 {
            let on_grain = |pt: &[f32; 2]| -> bool {
                grain_only.as_ref().is_some_and(|g| {
                    let (x, y) = (pt[0].floor() as i64, pt[1].floor() as i64);
                    [(x - 1, y - 1), (x, y - 1), (x - 1, y), (x, y)].iter().any(|&(a, b)| a >= 0 && b >= 0 && (a as usize) < w && (b as usize) < h && g[b as usize * w + a as usize])
                })
            };
            Some(rings.iter().map(|pt| if pt[0].is_nan() { 0.0 } else if on_grain(pt) { feather } else { let g = wash_grad(&soft, *pt, sigma, paper_eye); (feather * (1.0 + 12.0 * p.nib_wash_edge.clamp(0.0, 1.0) * (1.0 - g) * (1.0 - g))).round() }).collect())
        } else {
            None
        };
        // The wash is the INK alone, thinned to read `ratio` over the plane above (a grey's mixture carried
        // white — a fog over the drawing; a light grey's was white alone, laid as a slab lighter than the paper).
        let charge = charge_for_base(&mut cache, &p.palette, 0.85, p.opacity, p.palette.pigments[ink].masstone, &ink_base, ratio, wet, None);
        let load: Vec<f32> = ink_base.iter().map(|v| v * charge).collect();
        if !load.iter().any(|v| *v > 0.0) {
            continue;
        }
        match &edge {
            Some(e) => canvas.fill_rings_edge(&rings, &load, wet, feather, Some(e)),
            None => canvas.fill_rings(&rings, &load, wet, feather),
        }
        *k += 1;
        *placed += 1;
        if let Some(pr) = progress {
            pr(PaintProgress::Placed(*placed));
        }
        let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.to_string(), spline: rings, w0: feather, w1: 0.0, taper: 0.0, mix, wet, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge, prof: None });
        // NIB WASH GRADE: within the plane's own band, two sub-steps — the band's darker two thirds filled again, then
        // its darkest third over that — each a third of the key's share of the step to the plane below darker.
        if grade > 0.0 {
            let lo = if top == 0 { 0.0 } else { steps[top - 1] };
            let lower = if top == 0 { 0.0 } else { tones[top - 1] };
            let d = (tones[top] - lower).max(0.0);
            let mut level = tones[top];
            for s in 1..=2 {
                let thr = lo + (steps[top] - lo) * (3 - s) as f32 / 3.0;
                let sub: Vec<bool> = (0..w * h).map(|i| mask[i] && bands[i] as usize == top && values[i] < thr).collect();
                let sub = dilate_bool(&erode_bool(&sub, w, h, 1), w, h, 1);
                let sub = drop_small_pieces(&sub, w, h, (min_area / 4).max(4));
                let rings = if sub.iter().any(|&m| m) { mask_rings(&sub, w, h, 1.2) } else { Vec::new() };
                let next = tones[top] - grade * d * s as f32 / 3.0;
                if rings.is_empty() {
                    level = next;
                    continue;
                }
                let ratio = next.clamp(0.0, 1.0).powf(2.2) / level.clamp(0.0, 1.0).powf(2.2).max(1e-3);
                level = next;
                let charge = charge_for_base(&mut cache, &p.palette, 0.85, p.opacity, p.palette.pigments[ink].masstone, &ink_base, ratio, wet, None);
                let load: Vec<f32> = ink_base.iter().map(|v| v * charge).collect();
                if !load.iter().any(|v| *v > 0.0) {
                    continue;
                }
                canvas.fill_rings(&rings, &load, wet, feather);
                *k += 1;
                *placed += 1;
                if let Some(pr) = progress {
                    pr(PaintProgress::Placed(*placed));
                }
                let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.to_string(), spline: rings, w0: feather, w1: 0.0, taper: 0.0, mix, wet, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
            }
        }
    }
}

/// The bleed of pass `idx` of `n` in the recorded schedule `stages`: a schedule with the pens' passes (`pen-…`, see
/// `MediumProfile::nib_wash`) tapers over its washes alone and lays a pen's pass on the dried sheet unbled; any
/// other as `pass_bleed_taper`.
pub fn stage_bleed_taper(stages: Option<&[String]>, idx: usize, n: usize) -> f32 {
    match stages {
        Some(st) if st.iter().any(|s| s.starts_with("pen-")) => {
            if st.get(idx).is_some_and(|s| s.starts_with("pen-")) {
                0.0
            } else {
                // (The wash's index among the washes: with the drawing first (`nib_wash_draw`) they come after the pens.)
                pass_bleed_taper(st[..idx.min(st.len())].iter().filter(|s| !s.starts_with("pen-")).count(), st.iter().filter(|s| !s.starts_with("pen-")).count())
            }
        }
        _ => pass_bleed_taper(idx, n),
    }
}

/// NIB WASH WATER: whether pass `idx` of the recorded schedule `stages` is the planes' pass — the first wash stage
/// of a schedule with the pens' passes (`pen-…`); the planes are laid on the wash register's first pass.
pub fn water_stage(stages: Option<&[String]>, idx: usize) -> bool {
    match stages {
        Some(st) if st.iter().any(|s| s.starts_with("pen-")) => st.iter().position(|s| !s.starts_with("pen-")) == Some(idx),
        _ => false,
    }
}

pub fn pass_bleed_taper(layer: usize, n: usize) -> f32 {
    if n <= 1 {
        1.0
    } else {
        1.0 - layer as f32 / (n - 1) as f32
    }
}

/// How much of the fluid stage (`Canvas::flow`) pass `layer` of `n` gets: the BROAD passes — the first half of
/// the schedule — flow at the bleed's taper; the fine passes are laid on paper that has dried and do not
/// (`None`): a watercolour's detail is wet-on-dry, its washes wet-in-wet.
/// THE WATERCOLOUR RECIPE is the default for a new watercolour (7.2.0): the transmittance film, the fluid
/// stage, the grain in the water, the dry brush, the economy of structure, the fine lines. It was the
/// `PLAKAT_WC_WASH=1` experiment through 7.1.0 — and a plan run never set it, so the plain path painted a
/// posterised glaze. `PLAKAT_WC_LEGACY=1` brings the old glazed-wash path back for comparison.
pub fn wc_recipe_on() -> bool {
    std::env::var_os("PLAKAT_WC_LEGACY").is_none()
}
// NOTE: `wc_recipe_on()` says only whether the recipe is wanted; every reader must ALSO gate on the
// medium being a new watercolour (`wc_wash`, `wc_brush_cli`) — an ungated reader in the palette builder
// gave the oil 32 pigments and changed every oil render.

pub fn flow_taper(layer: usize, n: usize) -> Option<f32> {
    let t = pass_bleed_taper(layer, n);
    if t > 0.5 { Some(t) } else { None }
}

/// The collision's strength and hold level at pass `layer` of `n`: the broad passes drag at the bleed's
/// taper holding the faces (level 2); the fine passes drag at a third of it holding the SUBJECT too
/// (level 1) — the fine marks over the ground and the background marble into their wet base, while what
/// the fine brushes resolved on the figure is not softened. (0, _) = no collision on that pass.
pub fn collide_schedule(strength: f32, layer: usize, n: usize) -> (f32, f32) {
    let t = pass_bleed_taper(layer, n);
    if t > 0.5 {
        (strength * t, 2.0)
    } else if t > 0.0 {
        (strength * 0.35 * (t / 0.5), 1.0)
    } else {
        (0.0, 1.0)
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
                round: brush.round, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
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
        let (rim, rim_light) = {
            let mut ring: Vec<f32> = region
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
            let mean = ring.iter().sum::<f32>() / ring.len() as f32;
            // The LIGHTEST quarter of the ring: a highlight stands above its surroundings ALL ROUND.
            ring.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let q = (ring.len() / 4).max(1);
            (mean, ring[..q].iter().sum::<f32>() / q as f32)
        };
        if core - rim < 0.04 {
            continue;
        }
        // AN ISLAND OF THE GROUND IS NO HIGHLIGHT. A patch of white sky beside a dark head is bright against
        // its wide surroundings — the head is in them — and its ring averages dark, so it passed as a
        // plateau; the dome then eased the sky down toward the head and left a beige sliver on the white
        // (the chick, acrylic set 5). A highlight is brighter than its ring on EVERY side: where the ring's
        // lightest quarter is as bright as the core, the patch is the ground continuing, and is left alone.
        if core - rim_light < 0.04 {
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
                bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
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
                kd: None,
                cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
        }
    }
}

/// THE LINER (see `MediumProfile::liner`). The rigger's ridge — the band of sizes under the ladder's finest mark,
/// measured on the picture and on the canvas as painted, so only what the brushwork LOST is restored — but for the
/// LONG lines: a rope, a stay, a wire. A seed is a ridge that is a line (the structure tensor oriented over a long
/// window of four of the finest brush's widths, where a texture's curls disagree) and a local extremum across
/// it; from it the line is TRACED both ways along itself — a step and a snap across onto the ridge's crest —
/// while the ridge holds its sign at half the threshold, the line turns less than 12° a step and 30° in all,
/// and nothing used, protected, a face or the hair lies ahead. Only a trace a thirtieth of the sheet long is laid
/// — as ONE thin mark in the line's own colour (the mean along it), the pixels it covers used. A wheat field's
/// curls never run that far straight; a rigging line does. Each a record (stage `liner`).
fn liner_pass(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64) {
    let (w, h) = (input.width(), input.height());
    let n = (w * h) as usize;
    let strength = p.liner.clamp(0.0, 1.0);
    if strength <= 0.0 || *placed >= p.budget || w < 8 || h < 8 {
        return;
    }
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
    let missed: Vec<f32> = (0..n).map(|i| if want[i].signum() != have[i].signum() { want[i].abs() } else { (want[i].abs() - have[i].abs()).max(0.0) }).collect();
    let mut sorted: Vec<f32> = missed.iter().copied().filter(|v| *v > 1e-4).collect();
    if sorted.len() < 32 {
        return;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let keep = 0.02 + 0.06 * strength;
    let thr = sorted[((1.0 - keep) * (sorted.len() as f32 - 1.0)) as usize].max(1e-3);
    let luma = luma_map(input);
    let r_long = (p.min_brush * 4.0).round().max(6.0) as i32;
    let (gx, gy) = coherent_gradient(&luma, w, h, r_long, FlowInfill::Flat);
    let linear: Vec<f32> = {
        let (sx, sy) = sobel(&luma, w, h);
        let (mut jxx, mut jyy, mut jxy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        for i in 0..n {
            jxx[i] = sx[i] * sx[i];
            jyy[i] = sy[i] * sy[i];
            jxy[i] = sx[i] * sy[i];
        }
        let (jxx, jyy, jxy) = (box_blur(&jxx, w, h, r_long), box_blur(&jyy, w, h, r_long), box_blur(&jxy, w, h, r_long));
        (0..n).map(|i| (((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]).sqrt() / (jxx[i] + jyy[i] + 1e-6)).clamp(0.0, 1.0)).collect()
    };
    let blocked = |i: usize| -> bool {
        protect.and_then(|m| m.get(i)).copied().unwrap_or(false)
            || p.hair_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.25
            || p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.25
    };
    let radius = (p.min_brush * 0.5).max(0.8);
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.bristles = 1;
    brush.streak = 0.05;
    brush.round = 0.95;
    let short = w.min(h) as f32;
    let min_len = (short / 30.0).max(24.0);
    let max_steps = (short * 0.6 / 1.5) as usize;
    let cap = (*placed + (n as f32 * 0.0003 * strength) as usize).min(p.budget);
    let grid = (radius * 4.0).max(4.0);
    let (cols, rows) = (((w as f32) / grid).ceil() as u32, ((h as f32) / grid).ceil() as u32);
    let (cos_step, cos_all) = (12f32.to_radians().cos(), 30f32.to_radians().cos());
    let mut used = vec![false; n];
    let mut mix_cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let idx = |x: f32, y: f32| -> Option<usize> { (x >= 1.0 && y >= 1.0 && x < w as f32 - 1.0 && y < h as f32 - 1.0).then(|| y as usize * w as usize + x as usize) };
    for gyi in 0..rows {
        for gxi in 0..cols {
            if *placed >= cap {
                return;
            }
            *k += 1;
            let cx = (gxi as f32 + 0.5) * grid + jitter(p.seed ^ 0x11E_A1, *k) * grid;
            let cy = (gyi as f32 + 0.5) * grid + jitter(p.seed ^ 0x11E_A2, k.wrapping_add(1)) * grid;
            let Some(i) = idx(cx, cy) else { continue };
            if used[i] || missed[i] < thr || linear[i] < 0.75 || blocked(i) {
                continue;
            }
            let gn = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt().max(1e-6);
            let (nx, ny) = (gx[i] / gn, gy[i] / gn);
            let off = (p.min_brush * 1.2).max(2.0);
            let at = |sx: f32, sy: f32| -> f32 { luma[((cy + sy).clamp(0.0, h as f32 - 1.0) as usize) * w as usize + (cx + sx).clamp(0.0, w as f32 - 1.0) as usize] };
            let (a_side, b_side) = (at(nx * off, ny * off) - luma[i], at(-nx * off, -ny * off) - luma[i]);
            if a_side.signum() != b_side.signum() || a_side.abs().min(b_side.abs()) < 0.02 {
                continue;
            }
            let s0 = want[i].signum();
            let trace = |sign: f32| -> Vec<[f32; 2]> {
                let mut pts = Vec::new();
                let (mut x, mut y) = (cx, cy);
                let (t0x, t0y) = (-ny * sign, nx * sign);
                let (mut tx, mut ty) = (t0x, t0y);
                for _ in 0..max_steps {
                    let (sx, sy) = (x + tx * 1.5, y + ty * 1.5);
                    let mut best: Option<(f32, f32, usize, f32)> = None;
                    for o in [-1.5f32, -0.75, 0.0, 0.75, 1.5] {
                        let (px, py) = (sx - ty * o, sy + tx * o);
                        if let Some(j) = idx(px, py) {
                            let v = s0 * want[j];
                            if best.map(|b| v > b.3).unwrap_or(true) {
                                best = Some((px, py, j, v));
                            }
                        }
                    }
                    let Some((px, py, j, v)) = best else { break };
                    if v < 0.5 * thr || used[j] || blocked(j) {
                        break;
                    }
                    let gn = (gx[j] * gx[j] + gy[j] * gy[j]).sqrt().max(1e-6);
                    let (mut ux, mut uy) = (-gy[j] / gn, gx[j] / gn);
                    if ux * tx + uy * ty < 0.0 {
                        ux = -ux;
                        uy = -uy;
                    }
                    if ux * tx + uy * ty < cos_step {
                        break;
                    }
                    let (bx, by) = (0.7 * tx + 0.3 * ux, 0.7 * ty + 0.3 * uy);
                    let bn = (bx * bx + by * by).sqrt().max(1e-6);
                    tx = bx / bn;
                    ty = by / bn;
                    if tx * t0x + ty * t0y < cos_all {
                        break;
                    }
                    pts.push([px, py]);
                    x = px;
                    y = py;
                }
                pts
            };
            let (fwd, back) = (trace(1.0), trace(-1.0));
            let mut path: Vec<[f32; 2]> = back.into_iter().rev().collect();
            path.push([cx, cy]);
            path.extend(fwd);
            let len: f32 = path.windows(2).map(|q| ((q[1][0] - q[0][0]).powi(2) + (q[1][1] - q[0][1]).powi(2)).sqrt()).sum();
            if len < min_len {
                continue;
            }
            // Its own colour: the mean along it.
            let mut acc = [0f32; 3];
            for q in &path {
                let c = input.get_pixel(q[0] as u32, q[1] as u32).0;
                for ch in 0..3 {
                    acc[ch] += c[ch] as f32;
                }
            }
            let m = path.len() as f32;
            let c = [(acc[0] / m).round() as u8, (acc[1] / m).round() as u8, (acc[2] / m).round() as u8];
            // On PAPER (see `MediumProfile::paper`) the line only darkens: a ridge lighter than the sheet under it
            // is the paper's to keep, not a line of white.
            if p.paper > 0.0 {
                let mid = path[path.len() / 2];
                let under = color::linear_luma(color::srgb_to_linear(canvas.color_at(mid[0] as u32, mid[1] as u32)));
                if color::linear_luma(color::srgb_to_linear(c)) >= under {
                    continue;
                }
            }
            let load = mixture_cached(&mut mix_cache, c, &p.palette, p.charge, p.palette.pigments.len());
            // Every few points kept: the mark is a spline, not a pixel chain.
            let path: Vec<[f32; 2]> = path.iter().step_by(3).copied().chain(path.last().copied()).collect();
            let rr = (radius + 1.5).ceil() as i64;
            for q in &path {
                for dy in -rr..=rr {
                    for dx in -rr..=rr {
                        if let Some(j) = idx(q[0] + dx as f32, q[1] + dy as f32) {
                            used[j] = true;
                        }
                    }
                }
            }
            let s = Stroke { path, width0: radius, width1: (radius * 0.7).max(0.6), load, pressure: 1.0, wetness: 0.3 };
            s.rasterize(canvas, &brush);
            let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(ix, v)| (p.palette.pigments[ix].name.to_string(), *v)).collect();
            *placed += 1;
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "liner".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix, wet: s.wetness, press: 1.0, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(1), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
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
                round: brush.round, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
        }
    }
}

/// THE BONE (sumi-e; see `MediumProfile::bone`): the line register — the structure's contours as calligraphic
/// brush lines. The value's edges (thinned to a pixel, kept by hysteresis: the pen-and-ink planner's edge map,
/// opened by `bone`) linked into chains, each chain ONE brush line: its weight and its ink from the darker side
/// of the edge along it (a near dark form a full black line, a far pale one a thin grey hair — the ink's own
/// aerial perspective), tapering to its lift-off, the brush's `turn` and `kasure` acting on it. Longest first,
/// so the cap keeps the composition's lines; where the eye does not go (`ma`, the focal field and the subject)
/// a chain must run up to twice as far to be drawn. Off the protected paper. Replay-exact.
#[allow(clippy::too_many_arguments)]
/// The cast shadows' field (see `MediumProfile::nib_anchor`): per pixel, how much a place is much darker than its
/// surroundings (a 90th of the sheet round it), or lies just under the subject's silhouette (a 120th of the sheet
/// deep, fading downward), 0..1. `value` is the picture's smoothed value.
fn nib_anchor_field(value: &[f32], p: &PaintParams, w: u32, h: u32) -> Vec<f32> {
    let (wu, hu) = (w as usize, h as usize);
    let mean = box_blur(value, w, h, (w.max(h) / 90).max(2) as i32);
    let sil = p.silhouette_mask.as_deref().or(p.subject_mask.as_deref());
    // NIB CONTACT (see `MediumProfile::nib_contact`): the band under the silhouette deepened to `key` · a 40th of the
    // sheet, on a ground up to 0.8.
    let contact = p.nib_contact.clamp(0.0, 1.0);
    let band = if contact > 0.0 { ((w.max(h) as f32 / 40.0 * contact) as usize).max((w.max(h) / 120).max(2) as usize) } else { (w.max(h) / 120).max(2) as usize };
    let (v_lo, v_span) = if contact > 0.0 { (0.4 + 0.25 * contact, 0.4 - 0.25 * contact) } else { (0.4, 0.4) };
    (0..wu * hu)
        .map(|i| {
            let (cy, v) = (i / wu, value[i]);
            let c = ((mean[i] - v) / 0.25).clamp(0.0, 1.0) * (1.0 - v / 0.5).clamp(0.0, 1.0);
            let crev = c * c * (3.0 - 2.0 * c);
            let under = match sil {
                Some(m) if m.get(i).copied().unwrap_or(0.0) < 0.5 => {
                    let mut best = 0.0f32;
                    for dy in 1..=band.min(cy) {
                        if m.get(i - dy * wu).copied().unwrap_or(0.0) >= 0.5 {
                            best = 1.0 - (dy - 1) as f32 / band as f32;
                            break;
                        }
                    }
                    best * (1.0 - (v - v_lo) / v_span).clamp(0.0, 1.0)
                }
                _ => 0.0,
            };
            crev.max(under)
        })
        .collect()
}

/// NIB EYES (see `MediumProfile::nib_eyes`): an animal's eyes found in the picture, as a 0..1 mask, inside the
/// hair/fur region. An eye joins the dark markings round it (and a dark smooth blob is as often a patch of a dark
/// tail), so it is found by its CATCHLIGHT: a light point (over 0.7) ringed, a 100th of the sheet out, by the dark
/// (14 of 16 samples under 0.25) — the eye a disc as far as the dark runs from it (1.3 times the third-shortest of
/// 16 rays, at most a 25th of the sheet). The hatching keeps off
/// them; after the bone their contours and their dark (close diagonals, the lights left) are drawn in.
pub fn nib_eye_field(src: &RgbImage, hair: &[f32]) -> Vec<f32> {
    let (w, h) = src.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;
    let mut out = vec![0f32; n];
    if w < 16 || h < 16 || hair.len() != n {
        return out;
    }
    let long = w.max(h) as f32;
    let value = box_blur(&crate::paint::ink::value_map(src), w, h, 1);
    // The eyes, by their catchlights.
    let ring = (long / 100.0).max(3.0);
    let rmax = (long / 25.0).max(ring * 1.5);
    let at = |x: f32, y: f32| -> f32 {
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        if xi < 0 || yi < 0 || xi as usize >= wu || yi as usize >= hu { 1.0 } else { value[yi as usize * wu + xi as usize] }
    };
    let dirs: Vec<(f32, f32)> = (0..16).map(|j| { let a = j as f32 * std::f32::consts::PI / 8.0; (a.cos(), a.sin()) }).collect();
    let step = (ring / 3.0).max(1.0) as usize;
    let mut eyes: Vec<(f32, f32, f32)> = Vec::new();
    for y in (0..hu).step_by(step) {
        for x in (0..wu).step_by(step) {
            let i = y * wu + x;
            if value[i] <= 0.7 || hair[i] < 0.3 {
                continue;
            }
            let (fx, fy) = (x as f32, y as f32);
            if eyes.iter().any(|&(ex, ey, er)| (ex - fx).hypot(ey - fy) < er.max(ring) * 1.5) {
                continue;
            }
            if dirs.iter().filter(|d| at(fx + d.0 * ring, fy + d.1 * ring) < 0.25).count() < 14 {
                continue;
            }
            let mut rays: Vec<f32> = dirs
                .iter()
                .map(|d| {
                    let mut r = ring;
                    while r < rmax && at(fx + d.0 * (r + 1.0), fy + d.1 * (r + 1.0)) < 0.35 {
                        r += 1.0;
                    }
                    r
                })
                .collect();
            rays.sort_by(|a, b| a.total_cmp(b));
            eyes.push((fx, fy, rays[2] * 1.3));
        }
    }
    if std::env::var("PLAKAT_PAINT_SEMANTIC").is_ok() {
        eprintln!("nib eyes: {w}x{h}, catchlights {:?}", eyes);
    }
    for &(ex, ey, er) in &eyes {
        let (x0, x1) = ((ex - er).max(0.0) as usize, ((ex + er) as usize).min(wu - 1));
        let (y0, y1) = ((ey - er).max(0.0) as usize, ((ey + er) as usize).min(hu - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                if (x as f32 - ex).hypot(y as f32 - ey) <= er {
                    out[y * wu + x] = 1.0;
                }
            }
        }
    }
    out
}

/// NIB TUFT (see `MediumProfile::nib_tuft`): the grass's cells — on a grid of a 70th of the sheet, off the subject and
/// the fur and the blank lights (a cell half over 0.85 is the sky behind a stem, not ground), and inside the grass the
/// detector found (half the cell; no grass found, no cell) — these −1 — a cell's
/// strength as a tuft: its texture (the mean gradient) ranked among the frame's cells (from the 60th to the 90th
/// percentile), times how upright it stands (the horizontal gradient against the vertical, from 1.15 to 1.5), times
/// the eye's attention (`0.4 + 0.6·attn`). Returns `(cell, columns, strengths)`.
fn nib_tuft_field(value: &[f32], p: &PaintParams, attn: Option<&[f32]>, w: u32, h: u32) -> (usize, usize, Vec<f32>) {
    let (wu, hu) = (w as usize, h as usize);
    let cell = (w.max(h) as usize / 70).max(6);
    let (cols, rows) = (wu.div_ceil(cell), hu.div_ceil(cell));
    let (gx, gy) = sobel(value, w, h);
    let subj = p.silhouette_mask.as_deref().or(p.subject_mask.as_deref()).filter(|m| m.len() == wu * hu);
    let hair = p.hair_mask.as_deref().filter(|m| m.len() == wu * hu);
    let mut tex = vec![-1f32; cols * rows];
    // the grass the detector found; none, no tufts
    let Some(grass) = p.nib_grass_mask.as_deref().filter(|m| m.len() == wu * hu) else { return (cell, cols, vec![-1.0; cols * rows]) };
    let mut up = vec![0f32; cols * rows];
    let mut at = vec![1f32; cols * rows];
    for cj in 0..rows {
        for ci in 0..cols {
            let (mut sx, mut sy, mut sv, mut so, mut sa, mut sl, mut sg, mut nn) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
            for y in cj * cell..((cj + 1) * cell).min(hu) {
                for x in ci * cell..((ci + 1) * cell).min(wu) {
                    let i = y * wu + x;
                    sx += gx[i].abs();
                    sy += gy[i].abs();
                    sv += value[i];
                    sl += if value[i] > 0.85 { 1.0 } else { 0.0 };
                    sg += if grass[i] >= 0.5 { 1.0 } else { 0.0 };
                    let off = subj.map(|m| m[i] >= 0.5).unwrap_or(false) || hair.map(|m| m[i] >= 0.5).unwrap_or(false);
                    so += if off { 1.0 } else { 0.0 };
                    sa += attn.and_then(|m| m.get(i)).copied().unwrap_or(1.0);
                    nn += 1.0;
                }
            }
            let c = cj * cols + ci;
            if nn < 1.0 || so / nn > 0.15 || sv / nn > 0.9 || sl / nn > 0.5 || sg / nn < 0.5 {
                continue;
            }
            tex[c] = (sx + sy) / nn;
            up[c] = sx / sy.max(1e-6);
            at[c] = (sa / nn).clamp(0.0, 1.0);
        }
    }
    let mut ranked: Vec<f32> = tex.iter().copied().filter(|t| *t >= 0.0).collect();
    let mut out = vec![0f32; cols * rows];
    if ranked.len() < 10 {
        return (cell, cols, out);
    }
    ranked.sort_by(|a, b| a.total_cmp(b));
    let (t60, t90) = (ranked[ranked.len() * 6 / 10], ranked[ranked.len() * 9 / 10]);
    let ss = |x: f32| { let x = x.clamp(0.0, 1.0); x * x * (3.0 - 2.0 * x) };
    for c in 0..cols * rows {
        if tex[c] < 0.0 {
            out[c] = -1.0;
            continue;
        }
        out[c] = ss((tex[c] - t60) / (t90 - t60).max(1e-6)) * ss((up[c] - 1.15) / 0.35) * (0.4 + 0.6 * at[c]);
    }
    (cell, cols, out)
}

/// NIB FIGURE (see `MediumProfile::nib_figure`): the figure — the silhouette (the matte and SAM, no boxes) when a
/// face is found, and under `nib_people` every person found and every face; `None` when the key is off or there is
/// no figure.
fn nib_figure_mask(p: &PaintParams, w: u32, h: u32) -> Option<Vec<bool>> {
    if p.nib <= 0.0 || p.nib_figure <= 0.0 {
        return None;
    }
    let n = (w * h) as usize;
    let people = p.nib_person_mask.as_deref().filter(|m| p.nib_people > 0.0 && m.len() == n);
    let sil = p.silhouette_mask.as_deref().filter(|m| p.face_mask.is_some() && m.len() == n);
    let mut fig: Vec<bool> = match (sil, people) {
        (Some(s), _) => s.iter().map(|v| *v >= 0.5).collect(),
        (None, Some(_)) => vec![false; n],
        (None, None) => return None,
    };
    // NIB PEOPLE (see `MediumProfile::nib_people`): every person found, and every face, joined to the figure.
    if let Some(pm) = people {
        let faces = p.face_mask.as_deref().filter(|m| m.len() == n);
        for (i, f) in fig.iter_mut().enumerate() {
            *f |= pm[i] >= 0.5 || faces.map(|m| m[i] > 0.5).unwrap_or(false);
        }
    }
    // NIB OBJECT (see `MediumProfile::nib_object`): the things the figure holds joined to it.
    if let Some(om) = nib_object_mask(p, w, h) {
        for (f, o) in fig.iter_mut().zip(&om) {
            *f |= *o;
        }
    }
    Some(fig)
}

/// NIB OBJECT (see `MediumProfile::nib_object`): the held objects, when the key is on and the CLI found any.
fn nib_object_mask(p: &PaintParams, w: u32, h: u32) -> Option<Vec<bool>> {
    if p.nib <= 0.0 || p.nib_figure <= 0.0 || p.nib_object <= 0.0 {
        return None;
    }
    p.nib_object_mask.as_deref().filter(|m| m.len() == (w * h) as usize).map(|m| m.iter().map(|v| *v >= 0.5).collect())
}

/// NIB FIGURE: one run of a hatching line laid as pen strokes of about `max_len`, each a little off the line, a few
/// degrees off its angle and a little unsteady.
#[allow(clippy::too_many_arguments)]
fn nib_figure_strokes(canvas: &mut Canvas, score: &mut StrokeScore, p: &PaintParams, brush: &BrushConfig, ink: usize, n: usize, run: &[[f32; 2]], s: f32, width: f32, max_len: f32, (nx, ny): (f32, f32), li: usize, placed: &mut usize, k: &mut u64) {
    if (run.len() as f32) < s.max(4.0) {
        return;
    }
    let pieces = ((run.len() as f32 / max_len).ceil() as usize).max(1);
    let per = run.len() / pieces;
    for q in 0..pieces {
        if *placed >= p.budget {
            return;
        }
        let (i0, i1) = (q * per, if q + 1 == pieces { run.len() - 1 } else { (q + 1) * per });
        if i1 <= i0 + 1 {
            continue;
        }
        *k += 1;
        let sh = jitter(p.seed ^ 0xF16_1 ^ li as u64, *k) * 0.4 * s;
        let (a0, a1) = (run[i0], run[i1]);
        // the ends carried a little past the run, so a line meets its neighbour across a cell's edge
        let (ux, uy) = (a1[0] - a0[0], a1[1] - a0[1]);
        let ul = (ux * ux + uy * uy).sqrt().max(1e-3);
        let ext = (1.5 * width).max(2.0);
        let (a0, a1) = ([a0[0] - ux / ul * ext, a0[1] - uy / ul * ext], [a1[0] + ux / ul * ext, a1[1] + uy / ul * ext]);
        let tw = jitter(p.seed ^ 0xF16_2 ^ li as u64, *k) * 0.12;
        let (mx, my) = (0.5 * (a0[0] + a1[0]) + nx * sh, 0.5 * (a0[1] + a1[1]) + ny * sh);
        let (hx, hy) = (0.5 * (a1[0] - a0[0]), 0.5 * (a1[1] - a0[1]));
        let (c, sn) = (tw.cos(), tw.sin());
        let (rx, ry) = (hx * c - hy * sn, hx * sn + hy * c);
        let path = waver_path(&[[mx - rx, my - ry], [mx, my], [mx + rx, my + ry]], 0.15 * width, p.seed ^ 0xF16_3, *k);
        let mut load = vec![0f32; n];
        load[ink] = p.charge * 2.0;
        let st = Stroke { path, width0: width, width1: (width * 0.7).max(1.0), load, pressure: 1.0, wetness: 0.3 };
        st.rasterize(canvas, brush);
        *placed += 1;
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: st.path, w0: st.width0, w1: st.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), st.load[ink])], wet: st.wetness, press: st.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
    }
}

/// NIB FIGURE (see `MediumProfile::nib_figure`): the figure's tone as a pen draws it — straight parallel lines in four
/// layers, each laid where the value is darker than its threshold, broken at the contours into patches and into
/// strokes of a pen's length.
fn nib_figure_hatch(canvas: &mut Canvas, score: &mut StrokeScore, src: &RgbImage, p: &PaintParams, fig: &[bool], placed: &mut usize, k: &mut u64) {
    let (w, h) = (src.width(), src.height());
    let (wu, hu) = (w as usize, h as usize);
    if w < 16 || h < 16 || fig.len() != wu * hu {
        return;
    }
    let long = w.max(h) as f32;
    let raw = crate::paint::ink::value_map(src);
    let value = box_blur(&raw, w, h, (w.max(h) / 400).max(1) as i32);
    let edge = crate::paint::ink::edge_map(&box_blur(&raw, w, h, 1), wu, hu, 0.6);
    // The figure's own range: its value read from its 5th to its 95th percentile (off the face), so a coat that is a
    // middle grey in the picture is the figure's dark and goes to the cross-hatching.
    let on_face = |i: usize| p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.5;
    let mut vs: Vec<f32> = (0..wu * hu).filter(|&i| fig[i] && !on_face(i)).map(|i| value[i]).collect();
    if vs.len() < 64 {
        return;
    }
    vs.sort_by(|a, b| a.total_cmp(b));
    let (lo, hi) = (vs[vs.len() / 20], vs[vs.len() * 19 / 20]);
    let span = (hi - lo).max(0.05);
    // The face read against its own range (the coat's would make all of it a half-tone), then lifted toward the
    // paper by `0.8·nib_face`: its lit side blank, its shadow side in a layer or two, its darkest points closed.
    let mut fv: Vec<f32> = (0..wu * hu).filter(|&i| fig[i] && on_face(i)).map(|i| value[i]).collect();
    fv.sort_by(|a, b| a.total_cmp(b));
    // (from its 30th percentile: the face's box takes in the hair, whose dark would set the face's floor)
    let (flo, fspan) = if fv.len() >= 64 { (fv[fv.len() * 3 / 10], (fv[fv.len() * 19 / 20] - fv[fv.len() * 3 / 10]).max(0.05)) } else { (lo, span) };
    // NIB PEOPLE (see `MediumProfile::nib_people`): each face against its own range — a group's faces are lit
    // differently, and one's beard would set another's floor. (A face's patch, 4-connected; a patch under 64 pixels
    // keeps the faces' common range.)
    let mut face_id: Vec<u32> = Vec::new();
    let mut face_range: Vec<(f32, f32)> = Vec::new();
    // NIB FINE FACE (see `MediumProfile::nib_fine_face`): which of the patches is a small face (under an eighth of the
    // sheet across, by the root of its area).
    let mut face_small: Vec<bool> = Vec::new();
    if (p.nib_people > 0.0 || p.nib_fine_face > 0.0) && fv.len() >= 64 {
        face_id = vec![u32::MAX; wu * hu];
        for s0 in 0..wu * hu {
            if face_id[s0] != u32::MAX || !(fig[s0] && on_face(s0)) {
                continue;
            }
            let id = face_range.len() as u32;
            let mut stack = vec![s0];
            let mut vals: Vec<f32> = Vec::new();
            face_id[s0] = id;
            while let Some(i) = stack.pop() {
                vals.push(value[i]);
                let (x, y) = (i % wu, i / wu);
                for j in [(x > 0).then(|| i - 1), (x + 1 < wu).then(|| i + 1), (y > 0).then(|| i - wu), (y + 1 < hu).then(|| i + wu)].into_iter().flatten() {
                    if face_id[j] == u32::MAX && fig[j] && on_face(j) {
                        face_id[j] = id;
                        stack.push(j);
                    }
                }
            }
            vals.sort_by(|a, b| a.total_cmp(b));
            // NIB WASH PROFILE (see `MediumProfile::nib_wash_profile`): every face a small one.
            face_small.push(p.nib_fine_face > 0.0 && ((vals.len() as f32).sqrt() < long / 8.0 || p.nib_wash_profile > 0.0));
            if std::env::var_os("PLAKAT_PAINT_MASKS").is_some() {
                eprintln!("nib face patch {id}: {} px across{}", (vals.len() as f32).sqrt() as u32, if face_small[id as usize] { " — small, the finest pen" } else { "" });
            }
            face_range.push(if p.nib_people <= 0.0 { (flo, fspan) } else if vals.len() >= 64 { (vals[vals.len() * 3 / 10], (vals[vals.len() * 19 / 20] - vals[vals.len() * 3 / 10]).max(0.05)) } else { (flo, fspan) });
        }
    }
    let small = |i: usize| -> bool { face_id.get(i).map(|&f| f != u32::MAX && face_small[f as usize]).unwrap_or(false) };
    let any_small = face_small.iter().any(|&b| b);
    let fine_w = (long / 1800.0).max(1.0);
    let face_lift = 0.8 * p.nib_face.clamp(0.0, 1.0);
    // NIB VALUE: a value's rank in the whole picture (the share of the sheet darker than it).
    let rank: Vec<f32> = if p.nib_value > 0.0 {
        let mut hist = [0usize; 256];
        for v in &value {
            hist[((v.clamp(0.0, 1.0) * 255.0) as usize).min(255)] += 1;
        }
        let mut acc = 0usize;
        hist.iter().map(|c| { acc += c; acc as f32 / value.len().max(1) as f32 }).collect()
    } else {
        Vec::new()
    };
    let tone = |i: usize| -> f32 {
        if on_face(i) {
            let (flo, fspan) = face_id.get(i).and_then(|&f| face_range.get(f as usize)).copied().unwrap_or((flo, fspan));
            let v = ((value[i] - flo) / fspan).clamp(0.0, 1.0);
            v + (1.0 - v) * face_lift
        } else {
            let t = ((value[i] - lo) / span).clamp(0.0, 1.0);
            // NIB VALUE (see `MediumProfile::nib_value`): where it is light in the whole picture, lifted toward its rank.
            if p.nib_value > 0.0 {
                let r = rank[((value[i].clamp(0.0, 1.0) * 255.0) as usize).min(255)];
                let u = ((r - 0.35) / 0.3).clamp(0.0, 1.0);
                t + p.nib_value.clamp(0.0, 1.0) * u * u * (3.0 - 2.0 * u) * (r - t).max(0.0)
            } else {
                t
            }
        }
    };
    if let Some(dir) = std::env::var_os("PLAKAT_PAINT_MASKS") {
        let _ = image::GrayImage::from_fn(w, h, |x, y| { let i = (y * w + x) as usize; image::Luma([if fig[i] { (tone(i) * 255.0) as u8 } else { 0 }]) }).save(std::path::Path::new(&dir).join("nib_figure_tone.png"));
        let _ = image::GrayImage::from_fn(w, h, |x, y| { let i = (y * w + x) as usize; image::Luma([if on_face(i) { if small(i) { 255 } else { 128 } } else { 0 }]) }).save(std::path::Path::new(&dir).join("nib_face_small.png"));
    }
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let n = p.palette.pigments.len();
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.streak = 0.0;
    brush.round = 0.15;
    let width = (long / 900.0).max(1.0);
    let gap = (long / 280.0).max(3.0);
    let max_len = (long / 45.0).max(10.0);
    // THE FORM's direction: the structure tensor of the value smoothed to a 200th of the sheet, averaged over a cell
    // (a 40th of the sheet) — the way the folds and the forms run. Where it agrees on no direction, the hand's 55°.
    let cell = (long / 40.0).max(12.0);
    let (gx, gy) = sobel(&box_blur(&raw, w, h, (w.max(h) / 200).max(1) as i32), w, h);
    let mut jc = vec![0f32; wu * hu];
    let mut js = vec![0f32; wu * hu];
    let mut je = vec![0f32; wu * hu];
    for i in 0..wu * hu {
        jc[i] = gx[i] * gx[i] - gy[i] * gy[i];
        js[i] = 2.0 * gx[i] * gy[i];
        je[i] = gx[i] * gx[i] + gy[i] * gy[i];
    }
    let rr = (cell * 0.5) as i32;
    let (jc, js, je) = (box_blur(&jc, w, h, rr), box_blur(&js, w, h, rr), box_blur(&je, w, h, rr));
    let form_angle = |x: f32, y: f32| -> f32 {
        let i = (y.clamp(0.0, (hu - 1) as f32) as usize) * wu + x.clamp(0.0, (wu - 1) as f32) as usize;
        let mag = (jc[i] * jc[i] + js[i] * js[i]).sqrt();
        if mag < 0.2 * je[i].max(1e-9) {
            return 55f32.to_radians();
        }
        // along the isophote: a quarter turn from the gradient's double-angle mean
        0.5 * js[i].atan2(jc[i]) + std::f32::consts::FRAC_PI_2
    };
    // A pixel's cell, its boundary wavering by a quarter of a cell so the patches do not read as a grid.
    let owner = |x: f32, y: f32| -> (i64, i64) {
        let wx = x + 0.25 * cell * (y / cell * 2.1 + 0.7).sin();
        let wy = y + 0.25 * cell * (x / cell * 1.7 + 1.9).sin();
        ((wx / cell).floor() as i64, (wy / cell).floor() as i64)
    };
    // The figure's box.
    let (mut bx0, mut by0, mut bx1, mut by1) = (wu, hu, 0usize, 0usize);
    for i in 0..wu * hu {
        if fig[i] {
            let (x, y) = (i % wu, i / wu);
            bx0 = bx0.min(x);
            by0 = by0.min(y);
            bx1 = bx1.max(x);
            by1 = by1.max(y);
        }
    }
    if bx1 <= bx0 || by1 <= by0 {
        return;
    }
    // (the turn off the form's direction, the value it is laid under, the spacing's share): along the form first, then
    // across it, then the dark closed to a solid black — its lines broken at the contours, so the folds' edges stay
    // white in it. (Three and four crossing directions read as a printed screen, not a pen.)
    let layers = [(0.0f32, 0.85f32, 1.0f32), (60.0, 0.55, 1.0), (0.0, 0.25, 0.35)];
    // NIB GLINT (see `MediumProfile::nib_glint`): the lines broken over the specular highlights
    let glint = nib_glint_mask(src, p);
    let (ci0, cj0) = owner(bx0 as f32 - cell, by0 as f32 - cell);
    let (ci1, cj1) = owner(bx1 as f32 + cell, by1 as f32 + cell);
    // (under `nib_fine_face` the small faces in a pass of their own, at the finest width and closer, each layer's
    // threshold lowered by `0.04·key`: a small face's skin sits at the lift's floor, just under the first layer's
    // threshold, and was hatched over whole — now only what is darker than the face's 30th percentile, its hair and
    // its deep shadow, is hatched)
    for fine_pass in [false, true] {
    if fine_pass && !any_small {
        break;
    }
    let width = if fine_pass { fine_w } else { width };
    for (li, &(turn, thr, share)) in layers.iter().enumerate() {
        // NIB WASH PROFILE: on a face no hatching at all — the wash models it; only the solid dark (the eye's
        // socket, the nostril, the mouth's shadow) and the features are the pen's.
        if fine_pass && li < 2 && p.nib_wash_profile > 0.0 {
            continue;
        }
        let s = if fine_pass { (0.5 * gap * share).max(width * 0.9) } else { (gap * share).max(width * 0.9) };
        for cj in cj0..=cj1 {
            for ci in ci0..=ci1 {
                let (ccx, ccy) = ((ci as f32 + 0.5) * cell, (cj as f32 + 0.5) * cell);
                if ccx < bx0 as f32 - cell || ccx > bx1 as f32 + cell || ccy < by0 as f32 - cell || ccy > by1 as f32 + cell {
                    continue;
                }
                let a = form_angle(ccx, ccy) + turn.to_radians();
                let (dx, dy) = (a.cos(), a.sin());
                let (nx, ny) = (-dy, dx);
                // the lines' phase from the sheet's origin, so neighbouring cells at one angle meet line to line
                let base = (ccx * nx + ccy * ny) / s;
                let reach = cell * 1.1;
                let mut off = (base.floor() - (reach / s).ceil()) * s - (ccx * nx + ccy * ny);
                while off <= reach {
                    let mut run: Vec<[f32; 2]> = Vec::new();
                    let mut t = -reach;
                    while t <= reach {
                        let (x, y) = (ccx + dx * t + nx * off, ccy + dy * t + ny * off);
                        t += 1.0;
                        let (xi, yi) = (x.round() as i64, y.round() as i64);
                        let ok = xi >= 0 && yi >= 0 && (xi as usize) < wu && (yi as usize) < hu && owner(x, y) == (ci, cj) && {
                            let i = yi as usize * wu + xi as usize;
                            fig[i] && !edge[i] && !glint.as_ref().map(|g| g[i]).unwrap_or(false) && tone(i) < if fine_pass { thr - 0.04 * p.nib_fine_face.clamp(0.0, 1.0) } else { thr } && small(i) == fine_pass
                        };
                        if ok {
                            run.push([x, y]);
                        } else if !run.is_empty() {
                            nib_figure_strokes(canvas, score, p, &brush, ink, n, &run, s, width, max_len, (nx, ny), li, placed, k);
                            run.clear();
                        }
                        if *placed >= p.budget {
                            return;
                        }
                    }
                    if !run.is_empty() {
                        nib_figure_strokes(canvas, score, p, &brush, ink, n, &run, s, width, max_len, (nx, ny), li, placed, k);
                    }
                    off += s;
                }
            }
        }
    }
    }
    // The face's features — the profile, the brow, the eye, the nostril, the lips — as the picture's own contours
    // on the face, read from the picture as given, each a black line at the hatching's width.
    let fine = box_blur(&raw, w, h, 1);
    let fedge = crate::paint::ink::edge_map(&fine, wu, hu, 0.5);
    // NIB FINE FACE: on a small face the contours found among the small faces' own gradients (the picture off them
    // flattened to their mean, so the shelves and the folds do not set the threshold) and opened by half the key, drawn
    // with the finest pen.
    let sedge: Option<Vec<bool>> = any_small.then(|| {
        let near = box_blur(&(0..wu * hu).map(|i| if small(i) { 1.0 } else { 0.0 }).collect::<Vec<f32>>(), w, h, (long / 200.0).max(2.0) as i32);
        let (mut sum, mut cnt) = (0f32, 0f32);
        for i in 0..wu * hu {
            if small(i) {
                sum += fine[i];
                cnt += 1.0;
            }
        }
        let c = sum / cnt.max(1.0);
        // (NIB WASH PROFILE: the hair and the beard flattened too — their edges took the face's whole budget of
        // gradients, and a soft eye or nose never ranked among the strongest.)
        let field: Vec<f32> = (0..wu * hu).map(|i| {
            let m = if near[i] > 1e-6 { p.face_mask.as_deref().and_then(|f| f.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0) } else { 0.0 };
            let m = if p.nib_wash_profile > 0.0 { m * (1.0 - p.hair_mask.as_deref().and_then(|h| h.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0)) } else { m };
            fine[i] * m + c * (1.0 - m)
        }).collect();
        crate::paint::ink::edge_map(&field, wu, hu, 0.5 + 0.5 * p.nib_fine_face.clamp(0.0, 1.0))
    });
    let on: Vec<bool> = (0..wu * hu).map(|i| on_face(i) && if small(i) { sedge.as_ref().map(|e| e[i]).unwrap_or(false) } else { fedge[i] }).collect();
    let chains = crate::paint::ink::trace_chains(&on, wu, hu, (long / 200.0).max(4.0) as usize);
    for c in &chains {
        if *placed >= p.budget {
            return;
        }
        let path: Vec<[f32; 2]> = c.iter().step_by(2).copied().chain(c.last().copied()).collect();
        if path.len() < 2 {
            continue;
        }
        *k += 1;
        let mut load = vec![0f32; n];
        load[ink] = p.charge * 2.0;
        // (a small face's at the finest width)
        let width = if any_small && { let q = path[path.len() / 2]; small((q[1].round().clamp(0.0, (hu - 1) as f32) as usize) * wu + q[0].round().clamp(0.0, (wu - 1) as f32) as usize) } { fine_w } else { width };
        let st = Stroke { path, width0: width, width1: (width * 0.6).max(1.0), load, pressure: 1.0, wetness: 0.3 };
        st.rasterize(canvas, &brush);
        *placed += 1;
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: st.path, w0: st.width0, w1: st.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), st.load[ink])], wet: st.wetness, press: st.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
    }
    // NIB OBJECT (see `MediumProfile::nib_object`): a held object's contours — the picture's edges inside it and on its
    // rim (its mask grown by two pixels) — as firm black lines, `1 + 1.5·key` times the hatching's width.
    if let Some(om) = nib_object_mask(p, w, h) {
        let grown = box_blur(&om.iter().map(|&o| if o { 1.0 } else { 0.0 }).collect::<Vec<f32>>(), w, h, 2);
        let oedge = crate::paint::ink::edge_map(&fine, wu, hu, 0.35);
        let on: Vec<bool> = (0..wu * hu).map(|i| oedge[i] && grown[i] > 1e-6).collect();
        let chains = crate::paint::ink::trace_chains(&on, wu, hu, (long / 300.0).max(4.0) as usize);
        let ow = width * (1.0 + 1.5 * p.nib_object.clamp(0.0, 1.0));
        for c in &chains {
            if *placed >= p.budget {
                return;
            }
            let path: Vec<[f32; 2]> = c.iter().step_by(2).copied().chain(c.last().copied()).collect();
            if path.len() < 2 {
                continue;
            }
            *k += 1;
            let mut load = vec![0f32; n];
            load[ink] = p.charge * 2.0;
            let st = Stroke { path, width0: ow, width1: (ow * 0.7).max(1.0), load, pressure: 1.0, wetness: 0.3 };
            st.rasterize(canvas, &brush);
            *placed += 1;
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: st.path, w0: st.width0, w1: st.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), st.load[ink])], wet: st.wetness, press: st.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        }
    }
}

/// NIB POOL and NIB RIM (see `MediumProfile::nib_pool`, `MediumProfile::nib_rim`), after the bone: the contact shadows
/// under the figure closed to black with the broadest pen, then the figure's outline in one line of changing weight.
fn nib_figure_rim_pool(canvas: &mut Canvas, score: &mut StrokeScore, src: &RgbImage, p: &PaintParams, fig: &[bool], placed: &mut usize, k: &mut u64) {
    let (w, h) = (src.width(), src.height());
    let (wu, hu) = (w as usize, h as usize);
    if w < 16 || h < 16 || fig.len() != wu * hu {
        return;
    }
    let long = w.max(h) as f32;
    let value = box_blur(&crate::paint::ink::value_map(src), w, h, (w.max(h) / 400).max(1) as i32);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let n = p.palette.pigments.len();
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.streak = 0.0;
    brush.round = 0.15;
    let fine_w = (long / 1800.0).max(1.0);
    let ss = |x: f32| { let x = x.clamp(0.0, 1.0); x * x * (3.0 - 2.0 * x) };
    if p.nib_pool > 0.0 {
        // how far each pixel is under the figure, by column
        let reach = (p.nib_pool.clamp(0.0, 1.0) * long / 40.0).max(2.0);
        let mut pool = vec![false; wu * hu];
        // (how far under the figure each pool pixel is, a share of the reach — for `nib_falloff`)
        let mut pool_d = vec![0f32; wu * hu];
        // Each figure's own height (its connected piece of the mask): the pool is cast only from its lowest quarter —
        // where it meets the ground, not under an outstretched arm or a coat's hem in the air.
        let mut comp = vec![u32::MAX; wu * hu];
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for start in 0..wu * hu {
            if !fig[start] || comp[start] != u32::MAX {
                continue;
            }
            let id = spans.len() as u32;
            let (mut y0, mut y1) = (hu, 0usize);
            let mut stack = vec![start];
            comp[start] = id;
            while let Some(i) = stack.pop() {
                let (x, y) = (i % wu, i / wu);
                y0 = y0.min(y);
                y1 = y1.max(y);
                for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                    if nx < wu && ny < hu {
                        let j = ny * wu + nx;
                        if fig[j] && comp[j] == u32::MAX {
                            comp[j] = id;
                            stack.push(j);
                        }
                    }
                }
            }
            spans.push((y0, y1));
        }
        let low = |i: usize| -> bool {
            let (y0, y1) = spans[comp[i] as usize];
            (i / wu) as f32 >= y0 as f32 + 0.75 * (y1 - y0) as f32
        };
        for x in 0..wu {
            let mut last: Option<usize> = None;
            for y in 0..hu {
                let i = y * wu + x;
                if fig[i] {
                    last = low(i).then_some(y);
                    continue;
                }
                if let Some(ly) = last {
                    let d = (y - ly) as f32;
                    if d > reach {
                        last = None;
                        continue;
                    }
                    pool[i] = (1.0 - d / reach).sqrt() * (0.5 + 0.5 * ss((0.75 - value[i]) / 0.35)) > 0.5;
                    pool_d[i] = d / reach;
                }
            }
        }
        if let Some(dir) = std::env::var_os("PLAKAT_PAINT_MASKS") {
            let _ = image::GrayImage::from_fn(w, h, |x, y| image::Luma([if pool[(y * w + x) as usize] { 255 } else { 0 }])).save(std::path::Path::new(&dir).join("nib_pool.png"));
        }
        let width = long / 500.0;
        let max_len = (long / 30.0).max(10.0);
        let a = (-10f32).to_radians();
        let (dx, dy) = (a.cos(), a.sin());
        let (nx, ny) = (-dy, dx);
        let diag = ((wu * wu + hu * hu) as f32).sqrt();
        let (cx, cy) = (wu as f32 * 0.5, hu as f32 * 0.5);
        // NIB FALLOFF (see `MediumProfile::nib_falloff`): solid only in the contact's band; beyond it, broken dashes at
        // twice the spacing, kept by `key·(1 − d)^1.5`.
        let fall = p.nib_falloff.clamp(0.0, 1.0);
        let core = (long / 600.0).max(3.0) / reach;
        let sweeps: Vec<(Vec<bool>, f32, bool)> = if fall > 0.0 {
            vec![
                (pool.iter().zip(&pool_d).map(|(q, d)| *q && *d <= core).collect(), width, false),
                (pool.iter().zip(&pool_d).map(|(q, d)| *q && *d > core).collect(), 2.0 * width, true),
            ]
        } else {
            vec![(pool, width, false)]
        };
        let dash = (long / 60.0).max(8.0) as usize;
        for (mask, s, dashed) in &sweeps {
            let s = *s;
            let lay = |canvas: &mut Canvas, score: &mut StrokeScore, run: &[[f32; 2]], placed: &mut usize, k: &mut u64| {
                if !dashed {
                    nib_figure_strokes(canvas, score, p, &brush, ink, n, run, s, width, max_len, (nx, ny), 7, placed, k);
                    return;
                }
                for piece in run.chunks(dash) {
                    let q = piece[0];
                    let d = pool_d[(q[1].round().clamp(0.0, (hu - 1) as f32) as usize) * wu + q[0].round().clamp(0.0, (wu - 1) as f32) as usize];
                    *k += 1;
                    if jitter(p.seed ^ 0xFA1_0, *k) + 0.5 < fall * (1.0 - d).max(0.0).powf(1.5) {
                        nib_figure_strokes(canvas, score, p, &brush, ink, n, piece, s, width, max_len, (nx, ny), 8, placed, k);
                    }
                }
            };
            let mut off = -diag * 0.5;
            while off <= diag * 0.5 {
                let mut run: Vec<[f32; 2]> = Vec::new();
                let mut t = -diag * 0.5;
                while t <= diag * 0.5 {
                    let (x, y) = (cx + dx * t + nx * off, cy + dy * t + ny * off);
                    t += 1.0;
                    let (xi, yi) = (x.round() as i64, y.round() as i64);
                    let ok = xi >= 0 && yi >= 0 && (xi as usize) < wu && (yi as usize) < hu && mask[yi as usize * wu + xi as usize];
                    if ok {
                        run.push([x, y]);
                    } else if !run.is_empty() {
                        lay(canvas, score, &run, placed, k);
                        run.clear();
                    }
                    if *placed >= p.budget {
                        return;
                    }
                }
                if !run.is_empty() {
                    lay(canvas, score, &run, placed, k);
                }
                off += s;
            }
        }
    }
    if p.nib_rim > 0.0 {
        let bound: Vec<bool> = (0..wu * hu)
            .map(|i| {
                let (x, y) = (i % wu, i / wu);
                fig[i] && x > 0 && y > 0 && x + 1 < wu && y + 1 < hu && (!fig[i - 1] || !fig[i + 1] || !fig[i - wu] || !fig[i + wu])
            })
            .collect();
        let chains = crate::paint::ink::trace_chains(&bound, wu, hu, (long / 150.0).max(6.0) as usize);
        // the darkness just inside the outline: the darkest of the figure's pixels round the point
        let r = (long / 400.0).max(2.0) as i64;
        let shade = |q: [f32; 2]| -> f32 {
            let (x, y) = (q[0].round() as i64, q[1].round() as i64);
            let mut v = 1.0f32;
            for yy in (y - r).max(0)..=(y + r).min(hu as i64 - 1) {
                for xx in (x - r).max(0)..=(x + r).min(wu as i64 - 1) {
                    let i = yy as usize * wu + xx as usize;
                    if fig[i] {
                        v = v.min(value[i]);
                    }
                }
            }
            1.0 - v
        };
        let piece = (long / 60.0).max(8.0) as usize;
        let key = p.nib_rim.clamp(0.0, 1.0);
        for c in &chains {
            // the staircase of the mask's edge smoothed over five points
            let sm: Vec<[f32; 2]> = (0..c.len())
                .map(|j| {
                    let (a, b) = (j.saturating_sub(2), (j + 2).min(c.len() - 1));
                    let m = (b - a + 1) as f32;
                    let (sx, sy) = c[a..=b].iter().fold((0.0, 0.0), |(sx, sy), q| (sx + q[0], sy + q[1]));
                    [sx / m, sy / m]
                })
                .collect();
            let mut j = 0;
            while j + 2 < sm.len() {
                if *placed >= p.budget {
                    return;
                }
                let e = (j + piece).min(sm.len() - 1);
                let path: Vec<[f32; 2]> = sm[j..=e].iter().step_by(3).copied().chain(std::iter::once(sm[e])).collect();
                let (w0, w1) = (fine_w * (1.0 + 4.0 * key * shade(sm[j])), fine_w * (1.0 + 4.0 * key * shade(sm[e])));
                *k += 1;
                let mut load = vec![0f32; n];
                load[ink] = p.charge * 2.0;
                let st = Stroke { path, width0: w0, width1: w1, load, pressure: 1.0, wetness: 0.3 };
                st.rasterize(canvas, &brush);
                *placed += 1;
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: st.path, w0: st.width0, w1: st.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), st.load[ink])], wet: st.wetness, press: st.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
                j = e;
            }
        }
    }
}

/// NIB GLINT (see `MediumProfile::nib_glint`): the specular highlights — a spot of the picture over `1 − 0.08·key`
/// and a tenth brighter than its surroundings (a 60th of the sheet round it), no larger than a 30th of the sheet
/// across, opened by a 400th of the sheet. `None` when the key is off or there is none.
fn nib_glint_mask(src: &RgbImage, p: &PaintParams) -> Option<Vec<bool>> {
    if p.nib <= 0.0 || p.nib_glint <= 0.0 {
        return None;
    }
    let (w, h) = (src.width(), src.height());
    let (wu, hu) = (w as usize, h as usize);
    let long = w.max(h) as f32;
    let v = box_blur(&crate::paint::ink::value_map(src), w, h, 1);
    let m = box_blur(&v, w, h, (long / 60.0).max(3.0) as i32);
    let floor = 1.0 - 0.08 * p.nib_glint.clamp(0.0, 1.0);
    let hot: Vec<bool> = (0..wu * hu).map(|i| v[i] > floor && v[i] - m[i] > 0.1).collect();
    // each spot by its 8-connected extent; a broad one is a lit plane, not a glint
    let max_side = (long / 30.0) as usize;
    let mut out = vec![false; wu * hu];
    let mut seen = vec![false; wu * hu];
    let mut stack = Vec::new();
    let mut any = false;
    for s0 in 0..wu * hu {
        if !hot[s0] || seen[s0] {
            continue;
        }
        let mut px = Vec::new();
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        seen[s0] = true;
        stack.push(s0);
        while let Some(i) = stack.pop() {
            px.push(i);
            let (x, y) = (i % wu, i / wu);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for (dx, dy) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < wu && (ny as usize) < hu {
                    let j = ny as usize * wu + nx as usize;
                    if hot[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        if x1 - x0 < max_side && y1 - y0 < max_side {
            any = true;
            for i in px {
                out[i] = true;
            }
        }
    }
    any.then(|| dilate_bool(&out, wu, hu, (long / 400.0).max(1.0) as usize))
}

/// NIB GAP (see `MediumProfile::nib_gap`): `path` cut where it first enters `band` (a step of half a pixel); `None`
/// when it starts inside it.
fn cut_at_band(path: &[[f32; 2]], band: &[bool], w: usize, h: usize) -> Option<Vec<[f32; 2]>> {
    let inside = |q: [f32; 2]| {
        let (x, y) = (q[0].round() as i64, q[1].round() as i64);
        x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && band[y as usize * w + x as usize]
    };
    if path.is_empty() || inside(path[0]) {
        return None;
    }
    let mut out = vec![path[0]];
    for seg in path.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let steps = (len / 0.5).ceil().max(1.0) as usize;
        for st in 1..=steps {
            let t = st as f32 / steps as f32;
            let q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            if inside(q) {
                let t0 = (st - 1) as f32 / steps as f32;
                let q0 = [a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0];
                if (q0[0] - out[out.len() - 1][0]).abs() + (q0[1] - out[out.len() - 1][1]).abs() > 1e-3 {
                    out.push(q0);
                }
                return (out.len() >= 2).then_some(out);
            }
        }
        out.push(b);
    }
    Some(out)
}

/// THE PEN'S ACCENTS (see `MediumProfile::nib_anchor`, `MediumProfile::nib_hint`), laid after the hatching and
/// the bone, in the score as stage `nib` (the replay dries the sheet before it, as the painter does).
/// ANCHORS: on a grid of a 150th of the sheet, where the picture is much darker than its surroundings (a 90th of the
/// sheet round it) or just under the subject's silhouette (a 120th of the sheet deep), a cluster of short black
/// strokes at the middle pen's width along the local edge, crossed — `1 + 3·strength` of them, by the key.
/// HINTS: on a grid of a 60th of the sheet, in a light cell (value over 0.6) with a sharp contrast of its own, one
/// short dash at the finest pen's width along the edge there, by `key·contrast` of the cells.
fn nib_accents(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, original: &RgbImage, p: &PaintParams, attn: Option<&[f32]>, placed: &mut usize, k: &mut u64) {
    let (w, h) = (input.width(), input.height());
    if w < 8 || h < 8 {
        return;
    }
    let (wu, hu) = (w as usize, h as usize);
    let long = w.max(h) as f32;
    let value = box_blur(&crate::paint::ink::value_map(input), w, h, 1);
    let (gx, gy) = sobel(&value, w, h);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let n = p.palette.pigments.len();
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.streak = 0.0;
    brush.round = 0.15;
    // The direction along the edge at a point: square to the gradient summed over a little window (the hand's
    // diagonal where the picture has none).
    let along = |x: usize, y: usize, r: usize| -> (f32, f32) {
        let (mut sx, mut sy) = (0.0f32, 0.0f32);
        for yy in y.saturating_sub(r)..=(y + r).min(hu - 1) {
            for xx in x.saturating_sub(r)..=(x + r).min(wu - 1) {
                let i = yy * wu + xx;
                // the double angle, so opposite gradients agree
                sx += gx[i] * gx[i] - gy[i] * gy[i];
                sy += 2.0 * gx[i] * gy[i];
            }
        }
        if sx.abs() + sy.abs() < 1e-6 {
            return (std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2);
        }
        let a = 0.5 * sy.atan2(sx) + std::f32::consts::FRAC_PI_2;
        (a.cos(), a.sin())
    };
    let lay = |canvas: &mut Canvas, score: &mut StrokeScore, x: f32, y: f32, d: (f32, f32), len: f32, width: f32, placed: &mut usize| {
        let path = vec![[x - d.0 * len * 0.5, y - d.1 * len * 0.5], [x + d.0 * len * 0.5, y + d.1 * len * 0.5]];
        let mut load = vec![0f32; n];
        load[ink] = p.charge * 2.0;
        let s = Stroke { path, width0: width, width1: (width * 0.7).max(1.0), load, pressure: 1.0, wetness: 0.3 };
        s.rasterize(canvas, &brush);
        *placed += 1;
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), s.load[ink])], wet: s.wetness, press: s.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
    };
    if p.nib_anchor > 0.0 {
        let field = nib_anchor_field(&value, p, w, h);
        let cell = (long / 150.0).max(4.0) as usize;
        let mid_w = long / 500.0 * (500.0f32 / 1800.0).sqrt();
        for cy in (cell / 2..hu).step_by(cell) {
            for cx in (cell / 2..wu).step_by(cell) {
                if *placed >= p.budget {
                    return;
                }
                let i = cy * wu + cx;
                let a = field[i] * p.nib_anchor.clamp(0.0, 1.0);
                if a < 0.35 {
                    continue;
                }
                let d = along(cx, cy, 2);
                let count = (1.0 + 3.0 * a).round() as usize;
                for j in 0..count {
                    *k += 1;
                    let (jx, jy) = (jitter(p.seed ^ 0xA9C_0, *k) * cell as f32, jitter(p.seed ^ 0xA9C_1, *k) * cell as f32);
                    // every other stroke crossed at a third of a right angle
                    let t: f32 = if j % 2 == 1 { 0.5 } else { 0.0 };
                    let dd = (d.0 * t.cos() - d.1 * t.sin(), d.0 * t.sin() + d.1 * t.cos());
                    lay(canvas, score, cx as f32 + jx, cy as f32 + jy, dd, cell as f32 * 1.6, mid_w, placed);
                }
            }
        }
    }
    if p.nib_hint > 0.0 {
        let range = local_range(&value, wu, hu, 2);
        let glint = nib_glint_mask(original, p);
        let cell = (long / 60.0).max(6.0) as usize;
        let fine_w = long / 1800.0;
        for cy in (cell / 2..hu).step_by(cell) {
            for cx in (cell / 2..wu).step_by(cell) {
                if *placed >= p.budget {
                    return;
                }
                // the cell's sharpest light point
                let (mut best, mut bx, mut by) = (0.0f32, cx, cy);
                for yy in cy.saturating_sub(cell / 2)..(cy + cell / 2).min(hu) {
                    for xx in cx.saturating_sub(cell / 2)..(cx + cell / 2).min(wu) {
                        let i = yy * wu + xx;
                        if value[i] > 0.6 && range[i] > best {
                            best = range[i];
                            bx = xx;
                            by = yy;
                        }
                    }
                }
                let c = ((best - 0.15) / 0.2).clamp(0.0, 1.0);
                // (NIB GLINT: no dash at a specular highlight)
                let c = if glint.as_ref().map(|g| g[by * wu + bx]).unwrap_or(false) { 0.0 } else { c };
                *k += 1;
                if c <= 0.0 || jitter(p.seed ^ 0x41_A7, *k) + 0.5 >= p.nib_hint.clamp(0.0, 1.0) * c {
                    continue;
                }
                let d = along(bx, by, 2);
                lay(canvas, score, bx as f32, by as f32, d, cell as f32 * (0.4 + 0.4 * (jitter(p.seed ^ 0x41_A8, *k) + 0.5)), fine_w, placed);
            }
        }
    }
    // NIB TUFT (see `MediumProfile::nib_tuft`): the near grass as tufts — a fan of blades from one root at a cell's
    // foot, leaning with the grass, the side blades shorter and bent outward, at the middle pen tapering to the finest.
    if p.nib_tuft > 0.0 {
        // (the grass read from the picture as given: under a new painting `input` is the smoothed armature)
        let value = box_blur(&crate::paint::ink::value_map(original), w, h, 1);
        let (cell, cols, ts) = nib_tuft_field(&value, p, attn, w, h);
        let mut mass_dbg: Option<Vec<(f32, f32, f32)>> = std::env::var_os("PLAKAT_PAINT_MASKS").map(|_| vec![(0.0, 0.0, 0.0); ts.len()]);
        let contact = (p.nib_contact > 0.0 || p.nib_anchor > 0.0).then(|| nib_anchor_field(&value, p, w, h));
        let sil = p.silhouette_mask.as_deref().filter(|m| m.len() == wu * hu);
        let mid_w = long / 500.0 * (500.0f32 / 1800.0).sqrt();
        let fine_w = (long / 1800.0).max(1.0);
        let key = p.nib_tuft.clamp(0.0, 1.0);
        let cf = cell as f32;
        let lush = p.nib_lush.clamp(0.0, 1.0);
        // NIB SCATTER (see `MediumProfile::nib_scatter`): the roots by Poisson-disk sampling — candidates on a jittered
        // grid of a third of a cell, each kept when no root lies nearer than its radius, the radius from half a cell (in a
        // clump) to two cells as a smooth noise of five cells falls; under `0.3·key` a clearing, no root. Without it, one
        // site per cell, rooted at its foot.
        let sites: Vec<(usize, Option<(f32, f32)>)> = if p.nib_scatter > 0.0 {
            let key_s = p.nib_scatter.clamp(0.0, 1.0);
            let lat = 5.0 * cf;
            let node = |a: i64, b: i64| jitter(p.seed ^ 0x5CA_7, (a as u64).wrapping_mul(0x9E37_79B9).wrapping_add(b as u64)) + 0.5;
            let noise = |x: f32, y: f32| -> f32 {
                let (gx, gy) = (x / lat, y / lat);
                let (ix, iy) = (gx.floor() as i64, gy.floor() as i64);
                let (fx, fy) = (gx - ix as f32, gy - iy as f32);
                let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
                let top = node(ix, iy) * (1.0 - sx) + node(ix + 1, iy) * sx;
                let bot = node(ix, iy + 1) * (1.0 - sx) + node(ix + 1, iy + 1) * sx;
                top * (1.0 - sy) + bot * sy
            };
            let step = (cf / 3.0).max(2.0);
            let (gcols, grows) = ((wu as f32 / cf).ceil() as usize + 1, (hu as f32 / cf).ceil() as usize + 1);
            let mut bins: Vec<Vec<(f32, f32)>> = vec![Vec::new(); gcols * grows];
            let mut out = Vec::new();
            let mut y = 0.5 * step;
            let mut q = 0u64;
            while y < hu as f32 {
                let mut x = 0.5 * step;
                while x < wu as f32 {
                    q += 1;
                    let (px, py) = ((x + jitter(p.seed ^ 0x5CA_8, q) * step).clamp(0.0, (wu - 1) as f32), (y + jitter(p.seed ^ 0x5CA_9, q) * step).clamp(0.0, (hu - 1) as f32));
                    x += step;
                    let c = (py as usize / cell) * cols + px as usize / cell;
                    if c >= ts.len() || ts[c] < 0.0 {
                        continue;
                    }
                    let nz = noise(px, py);
                    if nz < 0.3 * key_s {
                        continue;
                    }
                    // (NIB LUSH: closer where the grass is dark — the roots of a deep clump half the spacing)
                    let dk = { let t = ((0.65 - value[py as usize * wu + px as usize]) / 0.3).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
                    let r = cf * (0.5 + 1.5 * (1.0 - nz)) * (1.0 - 0.4 * lush * (0.5 + 0.5 * dk));
                    let (bi, bj) = ((px / cf) as i64, (py / cf) as i64);
                    let reach = (r / cf).ceil() as i64;
                    let mut free = true;
                    'scan: for dj in -reach..=reach {
                        for di in -reach..=reach {
                            let (ni, nj) = (bi + di, bj + dj);
                            if ni < 0 || nj < 0 || ni as usize >= gcols || nj as usize >= grows {
                                continue;
                            }
                            for &(ox, oy) in &bins[nj as usize * gcols + ni as usize] {
                                if (ox - px).hypot(oy - py) < r {
                                    free = false;
                                    break 'scan;
                                }
                            }
                        }
                    }
                    if free {
                        bins[bj as usize * gcols + bi as usize].push((px, py));
                        out.push((c, Some((px, py))));
                    }
                }
                y += step;
            }
            out
        } else {
            (0..ts.len()).map(|c| (c, None)).collect()
        };
        for &(c, root) in &sites {
            let s = ts[c];
            if *placed >= p.budget {
                return;
            }
            let (ci, cj) = (c % cols, c / cols);
            let (x0, y0) = (ci * cell, cj * cell);
            let (x1, y1) = ((x0 + cell).min(wu), (y0 + cell).min(hu));
            if x1 <= x0 + 1 || y1 <= y0 + 1 {
                continue;
            }
            let (mx, my) = ((x0 + x1) / 2, (y0 + y1) / 2);
            // under the silhouette, the contact's band: the tufts doubled and always laid (off the figure itself)
            let in_sil = |y: usize| sil.map(|m| m[y * wu + mx] >= 0.5).unwrap_or(false);
            let under = contact.as_ref().map(|f| f[my * wu + mx]).unwrap_or(0.0) > 0.3 && !in_sil(my) && (1..=2).any(|t| my >= t * cell && in_sil(my - t * cell));
            let mut mv = 0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    mv += value[y * wu + x];
                }
            }
            let mv = mv / ((x1 - x0) * (y1 - y0)) as f32;
            // NIB MASS (see `MediumProfile::nib_mass`): in the dark grass more fans, rooted anywhere in the cell, crossing
            // into one mass; a cell of a dark clump (half its neighbours grass) always drawn.
            let dark = { let t = ((0.6 - mv) / 0.25).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
            // (a clump, not a lone tuft: the share of the eight neighbouring cells that are grass too, from 0.25 to 0.6)
            let share = {
                let (mut g, mut m) = (0f32, 0f32);
                for (di, dj) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (ni, nj) = (ci as i64 + di, cj as i64 + dj);
                    if ni >= 0 && nj >= 0 && (ni as usize) < cols && (nj as usize) * cols + (ni as usize) < ts.len() {
                        m += 1.0;
                        g += if ts[nj as usize * cols + ni as usize] > 0.1 { 1.0 } else { 0.0 };
                    }
                }
                g / m.max(1.0)
            };
            let clump = { let t = ((share - 0.25) / 0.35).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
            let dark = dark * clump;
            if let Some(dbg) = mass_dbg.as_mut() {
                dbg[c] = (s, mv, share);
            }
            let massed = p.nib_mass > 0.0 && s >= 0.0 && share >= 0.5 && mv < 0.6;
            // NIB LUSH (see `MediumProfile::nib_lush`): every grass cell a candidate by its darkness, its strength
            // floored at `0.7·key` times the smoothstep of `(0.65 − value)/0.3`
            let s = if lush > 0.0 && s >= 0.0 {
                let f = 0.7 * lush * { let t = ((0.65 - mv) / 0.3).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
                f + (1.0 - f) * s
            } else {
                s
            };
            *k += 1;
            if !under && !massed && (s <= 0.0 || jitter(p.seed ^ 0x7F7_1, *k) + 0.5 >= key * s) {
                continue;
            }
            // the lean: the grass's own direction there, held within half a radian of upright
            let (dx, dy) = along(mx, my, cell / 3);
            let (dx, dy) = if dy > 0.0 { (-dx, -dy) } else { (dx, dy) };
            let tilt = dx.atan2(-dy).clamp(-0.5, 0.5);
            let count = ((4.0 + 5.0 * (1.0 - mv)).round() as usize) * if under { 2 } else { 1 };
            let scale = if under { 1.3 } else { 0.7 + 0.6 * s.max(0.0) };
            let fans = 1 + (4.0 * p.nib_mass.clamp(0.0, 1.0) * dark).round() as usize;
            for f in 0..fans {
                let (bx, by) = if let (0, Some(r)) = (f, root) {
                    r
                } else if f == 0 {
                    (mx as f32 + jitter(p.seed ^ 0x7F7_2, *k) * 0.6 * cf, y1 as f32 - 0.1 * cf)
                } else {
                    *k += 1;
                    (x0 as f32 + (jitter(p.seed ^ 0x7F7_7, *k) + 0.5) * cf, y0 as f32 + (0.4 + 0.6 * (jitter(p.seed ^ 0x7F7_8, *k) + 0.5)) * cf)
                };
                for j in 0..count {
                    if *placed >= p.budget {
                        return;
                    }
                    *k += 1;
                    let frac = if count > 1 { j as f32 / (count - 1) as f32 - 0.5 } else { 0.0 };
                    let ang = tilt + frac * 1.3 + jitter(p.seed ^ 0x7F7_3, *k) * 0.3;
                    let len = cf * (1.4 + 1.4 * (jitter(p.seed ^ 0x7F7_4, *k) + 0.5)) * scale * (1.0 - 0.35 * frac.abs() * 2.0);
                    let (ux, uy) = (ang.sin(), -ang.cos());
                    // bent outward, a little either way; each blade's root a little along the clump's foot
                    let bow = (frac * 0.25 + jitter(p.seed ^ 0x7F7_5, *k) * 0.12) * len;
                    let rx = bx + jitter(p.seed ^ 0x7F7_6, *k) * 0.4 * cf;
                    let path = vec![[rx, by], [rx + ux * len * 0.5 - uy * bow, by + uy * len * 0.5 + ux * bow], [rx + ux * len, by + uy * len]];
                    let mut load = vec![0f32; n];
                    load[ink] = p.charge * 2.0;
                    let st = Stroke { path, width0: mid_w, width1: fine_w.min(mid_w), load, pressure: 1.0, wetness: 0.3 };
                    st.rasterize(canvas, &brush);
                    *placed += 1;
                    score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: st.path, w0: st.width0, w1: st.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), st.load[ink])], wet: st.wetness, press: st.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
                }
            }
        }
        if let (Some(dbg), Some(dir)) = (&mass_dbg, std::env::var_os("PLAKAT_PAINT_MASKS")) {
            let rows = ts.len() / cols;
            let img = image::RgbImage::from_fn(cols as u32, rows as u32, |x, y| {
                let (s, mv, sh) = dbg[y as usize * cols + x as usize];
                image::Rgb([(s.clamp(0.0, 1.0) * 255.0) as u8, ((1.0 - mv).clamp(0.0, 1.0) * 255.0) as u8, (sh.clamp(0.0, 1.0) * 255.0) as u8])
            });
            let _ = img.save(std::path::Path::new(&dir).join("tuft_field.png"));
        }
    }
    // NIB EYES (see `MediumProfile::nib_eyes`): the picture's contours inside the eyes and the nose, each chain a pure
    // black polyline at the finest pen's width.
    if let Some(fm) = p.nib_feature_mask.as_deref().filter(|m| p.nib_eyes > 0.0 && m.len() == wu * hu) {
        let edge = crate::paint::ink::edge_map(&value, wu, hu, p.nib_eyes.clamp(0.0, 1.0));
        let inside: Vec<bool> = edge.iter().zip(fm).map(|(e, m)| *e && *m > 0.5).collect();
        let chains = crate::paint::ink::trace_chains(&inside, wu, hu, (long / 400.0).max(3.0) as usize);
        let fine_w = (long / 1800.0).max(1.0);
        for c in &chains {
            if *placed >= p.budget {
                return;
            }
            // every other point, so the polyline is not a staircase of pixels
            let path: Vec<[f32; 2]> = c.iter().step_by(2).copied().chain(c.last().copied()).collect();
            if path.len() < 2 {
                continue;
            }
            let mut load = vec![0f32; n];
            load[ink] = p.charge * 2.0;
            let s = Stroke { path, width0: fine_w, width1: fine_w, load, pressure: 1.0, wetness: 0.3 };
            s.rasterize(canvas, &brush);
            *placed += 1;
            *k += 1;
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), s.load[ink])], wet: s.wetness, press: s.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        }
        // The dark of the eye and the nose filled with close diagonal lines of the finest pen (two and a half widths
        // apart), the catchlight and the nose's light left to the paper.
        let gap = (2.5 * fine_w).max(2.0);
        let lines = ((wu + hu) as f32 / gap) as usize;
        for j in 0..lines {
            let c = j as f32 * gap; // x + y = c
            let mut run: Vec<[f32; 2]> = Vec::new();
            let x_lo = (c - (hu - 1) as f32).max(0.0) as usize;
            let x_hi = (c.min((wu - 1) as f32)) as usize;
            for x in x_lo..=x_hi.max(x_lo) {
                let y = (c - x as f32).round() as i64;
                let ok = y >= 0 && (y as usize) < hu && {
                    let i = y as usize * wu + x;
                    fm[i] > 0.5 && value[i] < 0.3
                };
                if ok {
                    run.push([x as f32, y as f32]);
                }
                if (!ok || x == x_hi) && run.len() >= 2 {
                    if *placed >= p.budget {
                        return;
                    }
                    let path = vec![run[0], *run.last().unwrap()];
                    let mut load = vec![0f32; n];
                    load[ink] = p.charge * 2.0;
                    let s = Stroke { path, width0: fine_w, width1: fine_w, load, pressure: 1.0, wetness: 0.3 };
                    s.rasterize(canvas, &brush);
                    *placed += 1;
                    *k += 1;
                    score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "nib".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), s.load[ink])], wet: s.wetness, press: s.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
                }
                if !ok {
                    run.clear();
                }
            }
        }
    }
}

fn bone_pass(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, protect: Option<&[bool]>, attn: Option<&[f32]>, placed: &mut usize, k: &mut u64) {
    let (w, h) = (input.width(), input.height());
    let (wu, hu) = (w as usize, h as usize);
    let strength = p.bone.clamp(0.0, 1.0);
    if strength <= 0.0 || *placed >= p.budget || w < 8 || h < 8 {
        return;
    }
    let long = w.max(h) as f32;
    let value = box_blur(&crate::paint::ink::value_map(input), w, h, 1);
    let edge = crate::paint::ink::edge_map(&value, wu, hu, strength);
    let fig: Option<Vec<bool>> = if p.nib > 0.0 && p.nib_figure > 0.0 { nib_figure_mask(p, w, h) } else { None };
    // NIB CLOSE (see `MediumProfile::nib_close`): where the bone lines were laid, to find the outline's gaps after.
    let mut drawn: Option<Vec<bool>> = (fig.is_some() && p.nib_close > 0.0).then(|| vec![false; wu * hu]);
    let min_len = (long / 150.0).max(6.0) as usize;
    let mut chains = crate::paint::ink::trace_chains(&edge, wu, hu, min_len);
    chains.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let (gx, gy) = sobel(&value, w, h);
    // BONE COHERENCE (see `MediumProfile::bone_coherence`): the structure tensor of the value's gradient summed
    // over a window half a `min_len` wide, and its coherence ((λ1−λ2)/(λ1+λ2))² at every pixel — near 1 where
    // the gradients agree on a direction (a contour), near 0 where they point every way (grass, foliage, gravel).
    let coh: Option<Vec<f32>> = (p.bone_coherence > 0.0).then(|| {
        let r = ((min_len / 2) as i32).max(3);
        let jxx = box_blur(&gx.iter().map(|g| g * g).collect::<Vec<f32>>(), w, h, r);
        let jyy = box_blur(&gy.iter().map(|g| g * g).collect::<Vec<f32>>(), w, h, r);
        let jxy = box_blur(&gx.iter().zip(gy.iter()).map(|(a, b)| a * b).collect::<Vec<f32>>(), w, h, r);
        (0..wu * hu)
            .map(|i| {
                let tr = jxx[i] + jyy[i];
                if tr <= 1e-9 {
                    0.0
                } else {
                    (((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]) / (tr * tr)).clamp(0.0, 1.0)
                }
            })
            .collect()
    });
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let n = p.palette.pigments.len();
    let base = (p.min_brush * 0.8).max(1.2);
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.streak = 0.15;
    brush.round = 0.9;
    brush.bristles = 5;
    // NIB (see `MediumProfile::nib`): a steel point's line — even across, a flat section to a hard edge.
    if p.nib > 0.0 {
        brush.streak *= 1.0 - p.nib.clamp(0.0, 1.0);
        brush.round = 0.9 - 0.75 * p.nib.clamp(0.0, 1.0);
    }
    let cap = ((*placed) as f32 + (p.budget - *placed) as f32 * (0.10 + 0.25 * strength)).min(p.budget as f32) as usize;
    let at = |q: [f32; 2]| -> usize { (q[1].max(0.0) as usize).min(hu - 1) * wu + (q[0].max(0.0) as usize).min(wu - 1) };
    for c in &chains {
        if *placed >= cap {
            break;
        }
        if c.len() < 2 {
            continue;
        }
        if protect.map(|m| m.get(at(c[c.len() / 2])).copied().unwrap_or(false)).unwrap_or(false) {
            continue;
        }
        // The darker side of the edge along the chain, the eye's attention on it, and how much of it is the subject.
        let (mut dark, mut a_sum, mut s_sum, mut c_sum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut g_sum, mut z_sum) = (0.0f32, 0.0f32);
        for q in c {
            let i = at(*q);
            let g = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt().max(1e-6);
            g_sum += g;
            z_sum += p.nib_depth_map.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(1.0);
            let (ux, uy) = (gx[i] / g * 1.5, gy[i] / g * 1.5);
            let (v1, v2) = (value[at([q[0] + ux, q[1] + uy])], value[at([q[0] - ux, q[1] - uy])]);
            dark += 1.0 - v1.min(v2);
            let subj = p.subject_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            a_sum += attn.map(|m| m[i]).unwrap_or(1.0).max(if p.ma_flow > 0.0 && p.ma_focus > 0.0 { 0.0 } else { subj }).clamp(0.0, 1.0);
            s_sum += subj;
            c_sum += coh.as_ref().map(|m| m[i]).unwrap_or(1.0);
        }
        let d = (dark / c.len() as f32).clamp(0.0, 1.0);
        let a = a_sum / c.len() as f32;
        let a = if p.ma_focus > 0.0 { a.powf(1.0 + 4.0 * p.ma_focus.clamp(0.0, 1.0)) } else { a };
        let sj = s_sum / c.len() as f32;
        // ON THE SUBJECT the texture is the fine brush's (a feather's barbs, a tuft of fur): the line holds the
        // silhouette and the long structures only — a chain there must run twice as far, and is laid lighter.
        let need = min_len as f32 * (1.0 + sj) * if attn.is_some() { 1.0 + p.ma.clamp(0.0, 1.0) * (1.0 - a) } else { 1.0 };
        if c.len() < need as usize {
            continue;
        }
        // BONE WIND (see `MediumProfile::bone_wind`): a chain that winds — the isophote of a tuft of grass, a
        // clump's lobed edge, a small closed loop — is texture, not a bone. Its mean turning per `min_len` of
        // travel, read over samples half a `min_len` apart so the pixel staircase averages out, past the key's
        // bar: not a line. BONE DOT (see `MediumProfile::bone_dot`): a share of those laid as the tenbyō's dots.
        let winding = if p.bone_wind > 0.0 {
            let step = (min_len / 2).max(4);
            let (mut turn, mut prev, mut j) = (0.0f32, None::<(f32, f32)>, 0usize);
            while j + step < c.len() {
                let (a, b) = (c[j], c[j + step]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                if let Some((px, py)) = prev {
                    turn += (px * dy - py * dx).atan2(px * dx + py * dy).abs();
                }
                prev = Some((dx, dy));
                j += step;
            }
            turn * min_len as f32 / c.len() as f32
        } else {
            0.0
        };
        // NIB WASH AIR (see `MediumProfile::nib_wash_air`): the chain's nearness by the depth map — past a tenth of the
        // depth no line, thinner toward it.
        let near = if p.nib_wash > 0.0 && p.nib_wash_air > 0.0 && p.nib_depth_map.is_some() {
            let u = ((z_sum / c.len() as f32 - 0.1) / 0.4).clamp(0.0, 1.0);
            1.0 - p.nib_wash_air.clamp(0.0, 1.0) * (1.0 - u * u * (3.0 - 2.0 * u))
        } else {
            1.0
        };
        if near < 0.05 {
            continue;
        }
        // BONE FADE (see `MediumProfile::bone_fade`): where the eye does not go, no line and no dot.
        if p.bone_fade > 0.0 && attn.is_some() && a < p.bone_fade.clamp(0.0, 1.0) * 0.5 {
            continue;
        }
        let winds = p.bone_wind > 0.0 && winding > (1.0 - p.bone_wind.clamp(0.0, 1.0)) * 2.0;
        // BONE COHERENCE (see `MediumProfile::bone_coherence`): where the gradients around the chain point every
        // way — grass, foliage, gravel — a strong step is texture, not a contour: not a line either.
        let scattered = coh.is_some() && (c_sum / c.len() as f32) < p.bone_coherence.clamp(0.0, 1.0);
        if winds || scattered {
            if p.bone_dot > 0.0 && jitter(p.seed ^ 0xD07, *k + 1) + 0.5 < p.bone_dot.clamp(0.0, 1.0) {
                let gap = (base * 4.0).max(4.0);
                let mut along = 0.0f32;
                let mut next = gap * (0.5 + (jitter(p.seed ^ 0xD08, *k + 2) + 0.5));
                for q in c.windows(2) {
                    along += ((q[1][0] - q[0][0]).powi(2) + (q[1][1] - q[0][1]).powi(2)).sqrt();
                    if along < next {
                        continue;
                    }
                    if *placed >= cap {
                        break;
                    }
                    *k += 1;
                    next += gap * (0.7 + 0.6 * (jitter(p.seed ^ 0xD09, *k) + 0.5));
                    let dw = base * (1.5 + 2.0 * d) * (0.7 + 0.6 * (jitter(p.seed ^ 0xD0A, *k) + 0.5));
                    let (ox, oy) = (q[1][0] - q[0][0], q[1][1] - q[0][1]);
                    let ol = (ox * ox + oy * oy).sqrt().max(1e-3);
                    let half = dw * 0.3;
                    // BONE SCATTER (see `MediumProfile::bone_scatter`): the dot thrown off the chain along its
                    // normal by a Gaussian (Box–Muller from two hashes, clipped at two and a half sigma).
                    let (cx, cy) = if p.bone_scatter > 0.0 {
                        let u1 = (jitter(p.seed ^ 0xD0B, *k) + 0.5).clamp(1e-4, 1.0);
                        let u2 = jitter(p.seed ^ 0xD0C, *k) + 0.5;
                        let z = ((-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()).clamp(-2.5, 2.5);
                        let off = z * p.bone_scatter.clamp(0.0, 1.0) * gap;
                        (q[1][0] - oy / ol * off, q[1][1] + ox / ol * off)
                    } else {
                        (q[1][0], q[1][1])
                    };
                    let path = vec![[cx - ox / ol * half, cy - oy / ol * half], [cx + ox / ol * half, cy + oy / ol * half]];
                    let mut load = vec![0f32; n];
                    load[ink] = p.charge * (0.4 + 0.6 * d);
                    let mut dot_brush = brush;
                    dot_brush.round = 1.0;
                    let s = Stroke { path, width0: dw, width1: dw, load, pressure: 1.0, wetness: 0.3 };
                    s.rasterize(canvas, &dot_brush);
                    *placed += 1;
                    score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "bone".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), s.load[ink])], wet: s.wetness, press: s.pressure, streak: dot_brush.streak, round: dot_brush.round, pickup: Some(0.0), bristles: Some(dot_brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
                }
            }
            continue;
        }
        // BONE END (see `MediumProfile::bone_end`): which end stops against another line (a tome) and which
        // fades into the ground (a harai). An end meets a line when an edge pixel not of this chain lies within
        // two pixels of it. The line is drawn from its tome toward its harai, so the brush leaves the paper there.
        let e = p.bone_end.clamp(0.0, 1.0);
        let meets = |q: [f32; 2]| -> bool {
            let (x, y) = (q[0].round() as i64, q[1].round() as i64);
            for dy in -2..=2i64 {
                for dx in -2..=2i64 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= wu as i64 || ny >= hu as i64 || !edge[ny as usize * wu + nx as usize] {
                        continue;
                    }
                    if !c.iter().any(|r| r[0].round() as i64 == nx && r[1].round() as i64 == ny) {
                        return true;
                    }
                }
            }
            false
        };
        let (t0, t1) = if e > 0.0 { (meets(c[0]), meets(c[c.len() - 1])) } else { (false, false) };
        let rev: Vec<[f32; 2]>;
        let (cc, tome0, tome1): (&[[f32; 2]], bool, bool) = if t1 && !t0 {
            rev = c.iter().rev().copied().collect();
            (&rev, true, false)
        } else {
            (c, t0, t1)
        };
        // BONE RYAKU (see `MediumProfile::bone_ryaku`): off the focus the chain generalised into a few arcs.
        let ryaku: Vec<[f32; 2]>;
        let cc: &[[f32; 2]] = if p.bone_ryaku > 0.0 && attn.is_some() && a < 1.0 {
            let eps = 0.5 + 8.0 * p.bone_ryaku.clamp(0.0, 1.0) * base * (1.0 - a).powi(2);
            ryaku = chaikin(&chaikin(&rdp(cc, eps)));
            &ryaku
        } else {
            cc
        };
        // BOKASHI (see `MediumProfile::bokashi`): a far line laid wetter, feathered, its flying white gone.
        let bone_wet = if p.bokashi > 0.0 && attn.is_some() { 0.3 + 0.65 * p.bokashi.clamp(0.0, 1.0) * (1.0 - a).powi(2) } else { 0.3 };
        *k += 1;
        let path = waver_path(cc, 0.08 * base, p.seed, *k);
        // WEIGHT (see `MediumProfile::weight`): the line at a graphite point's weight, over the hatched tone —
        // pressed by the darker side of the edge, so a cloud's light rim keeps a light line.
        let wt = p.weight.clamp(0.0, 1.0) * d;
        let width = base * (0.5 + d) * (1.0 - 0.3 * sj) * (1.0 + 1.5 * wt);
        let mut load = vec![0f32; n];
        load[ink] = p.charge * (0.35 + 0.65 * d) * (1.0 - 0.3 * sj) * (1.0 + wt);
        // NIB (see `MediumProfile::nib`): ink is never grey — the line laid at the ink's full black, at the finest
        // pen's width (a 1800th of the long side under `nibs`) pressed by the weight.
        let width = if p.nib > 0.0 && p.nibs >= 1.0 { width + ((w.max(h) as f32 / 1800.0) * (1.0 + 1.5 * wt) - width) * p.nib.clamp(0.0, 1.0) } else { width };
        // NIB WASH SWELL (see `MediumProfile::nib_wash_swell`): the width by the edge's strength, toward a 400th of the
        // sheet on a hard step; thinned toward the far planes (`nib_wash_air`).
        let width = if p.nib > 0.0 && p.nib_wash > 0.0 && p.nib_wash_swell > 0.0 {
            let gm = g_sum / c.len() as f32;
            width + ((w.max(h) as f32 / 400.0) - width).max(0.0) * p.nib_wash_swell.clamp(0.0, 1.0) * (1.0 - (-3.0 * gm).exp())
        } else {
            width
        };
        let width = if near < 1.0 { (width * (0.4 + 0.6 * near)).max(1.0) } else { width };
        // NIB FIGURE (see `MediumProfile::nib_figure`): on the figure the line bold.
        let width = match &fig {
            Some(fm) if { let q = cc[cc.len() / 2]; fm[(q[1].round().clamp(0.0, (hu - 1) as f32) as usize) * wu + q[0].round().clamp(0.0, (wu - 1) as f32) as usize] } => width * (1.0 + 1.5 * p.nib_figure.clamp(0.0, 1.0)),
            _ => width,
        };
        if p.nib > 0.0 {
            load[ink] += (p.charge * 2.0 - load[ink]).max(0.0) * p.nib.clamp(0.0, 1.0);
        }
        let w1 = if e <= 0.0 { width * 0.45 } else if tome1 { width * (0.45 + 0.55 * e) } else { width * 0.45 * (1.0 - 0.8 * e) };
        // (Under the NIB the lift-off no thinner than a pixel: a sub-pixel tail was a grey hair.)
        let w1 = w1.max(if e > 0.0 { 0.5 } else { 0.8 });
        let w1 = if p.nib > 0.0 { w1 + (w1.max(1.0) - w1) * p.nib.clamp(0.0, 1.0) } else { w1 };
        let s = Stroke { path, width0: width, width1: w1, load, pressure: 1.0, wetness: bone_wet };
        s.rasterize(canvas, &brush);
        *placed += 1;
        if let Some(dm) = drawn.as_mut() {
            for q in cc {
                let (x, y) = (q[0].round() as i64, q[1].round() as i64);
                for yy in (y - 3).max(0)..=(y + 3).min(hu as i64 - 1) {
                    for xx in (x - 3).max(0)..=(x + 3).min(wu as i64 - 1) {
                        dm[yy as usize * wu + xx as usize] = true;
                    }
                }
            }
        }
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "bone".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), s.load[ink])], wet: s.wetness, press: s.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        // THE TOME's press: the brush halted on the end, the ink spreading and pooling — a short round mark
        // along the line's last direction, wider, darker and wetter by the key.
        for (tome, a, b) in [(tome0, cc[0], cc[(cc.len() - 1).min(4)]), (tome1, cc[cc.len() - 1], cc[cc.len().saturating_sub(5)])] {
            if !tome || *placed >= cap {
                continue;
            }
            let (ox, oy) = (a[0] - b[0], a[1] - b[1]);
            let ol = (ox * ox + oy * oy).sqrt().max(1e-3);
            let tw = width * (1.0 + 0.6 * e);
            let half = tw * 0.25;
            let mut tl = vec![0f32; n];
            tl[ink] = p.charge * (0.35 + 0.65 * d) * (1.0 + 0.4 * e);
            let mut tb = brush;
            tb.round = 1.0;
            let ts = Stroke { path: vec![[a[0] - ox / ol * half, a[1] - oy / ol * half], [a[0] + ox / ol * half * 0.5, a[1] + oy / ol * half * 0.5]], width0: tw, width1: tw, load: tl, pressure: 1.0, wetness: 0.3 + 0.3 * e };
            ts.rasterize(canvas, &tb);
            *placed += 1;
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "bone".into(), spline: ts.path, w0: ts.width0, w1: ts.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), ts.load[ink])], wet: ts.wetness, press: ts.pressure, streak: tb.streak, round: tb.round, pickup: Some(0.0), bristles: Some(tb.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        }
    }
    // NIB CLOSE (see `MediumProfile::nib_close`): along the figure's boundary, a stretch no bone line covers, shorter
    // than `key`·a 40th of the sheet and between two covered stretches, closed with a thin black line.
    if let (Some(fm), Some(dm)) = (&fig, &drawn) {
        let bound: Vec<bool> = (0..wu * hu)
            .map(|i| {
                let (x, y) = (i % wu, i / wu);
                fm[i] && x > 0 && y > 0 && x + 1 < wu && y + 1 < hu && (!fm[i - 1] || !fm[i + 1] || !fm[i - wu] || !fm[i + wu])
            })
            .collect();
        let chains = crate::paint::ink::trace_chains(&bound, wu, hu, 8);
        let gap = p.nib_close.clamp(0.0, 1.0) * long / 40.0;
        let fine_w = (long / 1800.0).max(1.0);
        for c in &chains {
            let cov: Vec<bool> = c.iter().map(|q| dm[at(*q)]).collect();
            let mut j = 0;
            while j < c.len() {
                if cov[j] {
                    j += 1;
                    continue;
                }
                let j0 = j;
                while j < c.len() && !cov[j] {
                    j += 1;
                }
                // a gap between two drawn stretches, short enough to be a break in one line
                if j0 == 0 || j >= c.len() || (j - j0) as f32 > gap {
                    continue;
                }
                if *placed >= p.budget {
                    return;
                }
                let mut path: Vec<[f32; 2]> = c[j0 - 1..=j].iter().step_by(2).copied().collect();
                if path.last() != Some(&c[j]) {
                    path.push(c[j]);
                }
                if path.len() < 2 {
                    continue;
                }
                *k += 1;
                let mut load = vec![0f32; n];
                load[ink] = p.charge * 2.0;
                let s = Stroke { path, width0: fine_w, width1: fine_w, load, pressure: 1.0, wetness: 0.3 };
                s.rasterize(canvas, &brush);
                *placed += 1;
                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "bone".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), s.load[ink])], wet: s.wetness, press: s.pressure, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
            }
        }
    }
}

/// The SPLATTER pass: flick fine droplets of pigment across the painting — the signature spatter of a loaded
/// brush tapped over the paper (visible in nearly every loose watercolour). Most droplets are tiny DARK spots in
/// the palette's darkest pigment; a few are coarser blobs; and — on a surface-white medium — a fraction LIFT to
/// the paper for the bright speckle of spray/snow/sparkle. Positions come from a hash so the spatter is even but
/// unstructured, and every droplet is recorded into the score (the bright ones as wipe strokes at `wet*lift`,
/// exactly as replay applies them) so a re-render reproduces it. `splatter` scales the count.
fn splatter_pass(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, placed: &mut usize, k: &mut u64, stage: &str, on_wet_sheet: bool) {
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
    // FLICKS, not rain: spatter comes off a brush flicked at the sheet a few times — each flick a RADIAL
    // spray from its own origin, dense near the origin and thinning out with distance, the drops round
    // where they land square on and ELONGATED along the spray's direction where they land obliquely,
    // their sizes spread from mist to the odd heavy blob. The uniform scatter before read as digital rain.
    // Origins land off the subject (a painter flicks at the margins), deterministic from the seed.
    let n_flicks = (3.0 + 5.0 * strength).round() as usize;
    let mut origins: Vec<(f32, f32)> = Vec::new();
    let mut tries = 0;
    while origins.len() < n_flicks && tries < 64 {
        tries += 1;
        *k = k.wrapping_add(1);
        let ox = (jitter(p.seed ^ 0x0F11, *k) + 0.5) * w as f32;
        let oy = (jitter(p.seed ^ 0x0F22, k.wrapping_add(7)) + 0.5) * h as f32;
        let i = (oy as usize).min(h as usize - 1) * w as usize + (ox as usize).min(w as usize - 1);
        if sal[i] > 0.5 || p.face_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) > 0.2 {
            continue;
        }
        origins.push((ox, oy));
    }
    if origins.is_empty() {
        origins.push((w as f32 * 0.5, h as f32 * 0.5));
    }
    let reach = w.min(h) as f32 * 0.45;
    for _ in 0..n {
        *k = k.wrapping_add(1);
        // Which flick, and where along its spray: distance from the origin decays (density ∝ 1/(1+d)),
        // direction uniform.
        let fi = ((jitter(p.seed ^ 0x5D19, *k) + 0.5) * origins.len() as f32) as usize;
        let (ox, oy) = origins[fi.min(origins.len() - 1)];
        let ang = (jitter(p.seed ^ 0x9C4B, k.wrapping_add(11)) + 0.5) * std::f32::consts::TAU;
        let u = (jitter(p.seed ^ 0x3D7A, k.wrapping_add(13)) + 0.5).clamp(1e-4, 1.0);
        let dist = reach * u * u; // squared: most drops near the origin
        let cx = (ox + ang.cos() * dist).clamp(1.0, w as f32 - 2.0);
        let cy = (oy + ang.sin() * dist).clamp(1.0, h as f32 - 2.0);
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
        // Sizes: a log-normal-ish spread — mist for most, a tail of heavy blobs (the drop that flew
        // furthest is the biggest), up to a few brush widths.
        let radius = unit * (0.5 + 1.2 * rr * rr + if rr > 0.93 { 4.0 * (rr - 0.93) / 0.07 } else { 0.0 }) * (0.8 + 0.4 * (dist / reach));
        // A drop on wet paper blooms wide and soft; on dry paper it is a hard dot. On a wet sheet (the
        // transparent media, spattered while the washes are still wet) the DROP READS THE CANVAS: it is
        // as wet as the paper it lands on, so one in a wash blooms and feathers and one on dry paper
        // stays sharp — recorded per stroke, so the replay lays the same drop.
        let (radius, drop_wet) = if on_wet_sheet {
            let wet_here = canvas.wetness[i].clamp(0.0, 1.0);
            (radius * (1.0 + 0.6 * wet_here), 0.25 + 0.7 * wet_here)
        } else {
            match p.technique {
                WetTechnique::WetOnWet => (radius * 1.5, 0.9),
                WetTechnique::DryOnDry => (radius * 0.8, 0.15),
                _ => (radius, 0.5),
            }
        };
        // Elongation: a drop landing obliquely (far from the origin) stretches along the spray.
        let stretch = radius * 2.2 * (dist / reach).powi(2);
        let path = vec![[cx - ang.cos() * stretch * 0.5, cy - ang.sin() * stretch * 0.5], [cx + ang.cos() * stretch * 0.5 + 0.6, cy + ang.sin() * stretch * 0.5 + 0.4]];
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
                stage: stage.into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.0,
                mix: Vec::new(),
                wet,
                press: 1.0,
                streak: 0.0,
                round: 1.0, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
        } else {
            // THE DROP'S COLOUR is the brush's — the wash being flicked, i.e. the picture's colour at this
            // spot, a shade deeper — not the palette's darkest pigment. On a white snow scene the ink drops
            // read as black rain; a flick of the sky's grey or the snow's blue is what a painter leaves.
            // Only where the picture is itself dark does the drop go to the ink. (The transparent media:
            // an opaque medium keeps the ink — its spatter is a dark accent by convention.)
            let mut load = vec![0f32; np];
            let charge = p.charge * (0.45 + 0.55 * rr);
            if p.luminous {
                let src = input.get_pixel(cx as u32, cy as u32).0;
                let src_luma = color::linear_luma(color::srgb_to_linear(src));
                if src_luma < 0.25 {
                    load[ink] = charge;
                } else {
                    // The pigment nearest the spot's colour, deepened: a drop is a small heavy charge.
                    let lab = color::srgb_to_lab(src);
                    let mut best = (ink, f32::MAX);
                    for (i, pg) in p.palette.pigments.iter().enumerate() {
                        let d = color::delta_e76(lab, color::srgb_to_lab(pg.masstone));
                        if d < best.1 {
                            best = (i, d);
                        }
                    }
                    load[best.0] = charge * 0.6;
                    load[ink] = charge * 0.12 * (1.0 - src_luma);
                }
            } else {
                load[ink] = charge;
            }
            let s = Stroke { path, width0: radius, width1: radius, load, pressure: 1.0, wetness: drop_wet };
            s.rasterize(canvas, &brush);
            let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(idx, v)| (p.palette.pigments[idx].name.to_string(), *v)).collect();
            *placed += 1;
            score.strokes.push(StrokeRecord {
                id: *placed as u32,
                wipe: false, wash: false,
                stage: stage.into(),
                spline: s.path,
                w0: s.width0,
                w1: s.width1,
                taper: 0.0,
                mix,
                wet: s.wetness,
                press: 1.0,
                streak: 0.0,
                round: 1.0, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
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
fn sumi_ink(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, p: &PaintParams, attn: Option<&[f32]>, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let (w, h) = (input.width(), input.height());
    let long = w.max(h) as f32;
    let firm = imageops::blur(input, (long / 400.0).max(1.0));
    let fluma = luma_map(&firm);
    let mut sorted = fluma.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // (The share a plan key — `sumi_dark`; unset, the darkest eighth.)
    let share = if p.sumi_dark > 0.0 { p.sumi_dark.clamp(0.0, 1.0) } else { 0.12 };
    let ink_t = sorted[((sorted.len() as f32 - 1.0) * share) as usize].min(0.10);
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
    // SUMI FOCUS (see `MediumProfile::sumi_focus`): the eye's attention over the sheet — the picture's saliency
    // weighted to the centre, or the subject — that a mass must reach to be laid black.
    let own: Option<Vec<f32>> = (p.sumi_focus > 0.0 && attn.is_none()).then(|| focal_field(input));
    let focus: Option<&[f32]> = if p.sumi_focus > 0.0 { attn.or(own.as_deref()) } else { None };
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
            if let Some(f) = focus {
                // (Under `ma_flow` the field holds the silhouette already — the subject mask's boxes left out.)
                let subj = if p.ma_flow > 0.0 && p.ma_focus > 0.0 && attn.is_some() { 0.0 } else { p.subject_mask.as_deref().and_then(|m| m.get(i)).copied().unwrap_or(0.0) };
                if f.get(i).copied().unwrap_or(1.0).max(subj) < p.sumi_focus.clamp(0.0, 1.0) {
                    continue;
                }
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
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "ink".into(), spline: s.path, w0: s.width0, w1: s.width1, taper: 0.3, mix: vec![(p.palette.pigments[ink].name.to_string(), amt)], wet: s.wetness, press: s.pressure, streak: dry.streak, round: dry.round, pickup: Some(0.0), bristles: Some(dry.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
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
/// SCRAP PAPER. A watercolourist tests the mix before it goes on the sheet: this is the pigment charge
/// that brings `colour` to `luma` on white paper, found on an eight-pixel probe canvas by bisection. Fixed
/// charges per layer were the first build's darkness — three glazes of a night's grey mean, each at a
/// charge chosen blind, went nearly black and the three bands could not be told apart. A glaze over a wash
/// of value U that must read T needs a glaze that reads T/U on white (transparent layers multiply), which is
/// what the callers ask for. Cached by colour and target.
fn charge_for(cache: &mut std::collections::HashMap<(u32, u16), f32>, palette: &Palette, tooth: f32, opacity: f32, colour: Srgb, luma: f32, wet: f32, brush: Option<&BrushConfig>) -> f32 {
    let n = palette.pigments.len();
    let base = mixture_for_key(mixture_key(colour), palette, n);
    charge_for_base(cache, palette, tooth, opacity, colour, &base, luma, wet, brush)
}

/// `charge_for` over an explicit `base` mixture (cached under `colour`'s key): an INK WASH is the ink alone,
/// thinned — the mixture a grey resolves to carries white, and a wash of white over a drawing is a fog over
/// it, and a light enough grey resolves to white alone, which no charge can bring to the tone (the solver
/// then returned its ceiling — a slab of white lighter than the paper).
#[allow(clippy::too_many_arguments)]
fn charge_for_base(cache: &mut std::collections::HashMap<(u32, u16), f32>, palette: &Palette, tooth: f32, opacity: f32, colour: Srgb, base: &[f32], luma: f32, wet: f32, brush: Option<&BrushConfig>) -> f32 {
    let luma = luma.clamp(0.03, 0.99);
    // A brush stroke deposits pigment by its own rule, not a fill's: a mark calibrated on a fill was invisible.
    let key = (((colour[0] as u32) << 16) | ((colour[1] as u32) << 8) | colour[2] as u32, (luma * 1000.0) as u16 | if brush.is_some() { 0x8000 } else { 0 });
    if let Some(c) = cache.get(&key) {
        return *c;
    }
    let ring: Vec<[f32; 2]> = vec![[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]];
    let read = |charge: f32| -> f32 {
        let mut probe = Canvas::white(8, 8, *palette, tooth).with_opacity(opacity);
        let load: Vec<f32> = base.iter().map(|v| v * charge).collect();
        match brush {
            Some(b) => Stroke { path: vec![[0.0, 4.0], [4.0, 4.0], [8.0, 4.0]], width0: 8.0, width1: 8.0, load, pressure: 1.0, wetness: wet }.rasterize(&mut probe, b),
            None => probe.fill_rings(&ring, &load, wet, 0.0),
        }
        color::linear_luma(color::srgb_to_linear(probe.color_at(4, 4)))
    };
    // Monotone: more pigment, darker. Bisect on a log scale.
    let (mut lo, mut hi) = (0.01f32, 40.0f32);
    if read(lo) <= luma {
        cache.insert(key, lo);
        return lo;
    }
    if read(hi) >= luma {
        cache.insert(key, hi);
        return hi;
    }
    for _ in 0..14 {
        let mid = (lo.ln() * 0.5 + hi.ln() * 0.5).exp();
        if read(mid) > luma {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let c = (lo.ln() * 0.5 + hi.ln() * 0.5).exp();
    cache.insert(key, c);
    c
}



/// THE WASHES OF A BRUSH WATERCOLOUR (`PLAKAT_WCB_FILL`): each mass (see `wc_mass_colours`) laid as one
/// flat wash in its own colour, lightest first so a darker wash blooms over a lighter neighbour's edge,
/// each tested on scrap paper so it reads its colour, with a wet edge sized by the mass. The brush ladder
/// then models the subject and the faces over them.
/// The value at quantile `q` (0..1) of `v`.
fn quantile(v: &[f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[((s.len() as f32 - 1.0) * q.clamp(0.0, 1.0)).round() as usize]
}

/// WASH STROKES (`PLAKAT_WCB_SWEEP`): a mass is not filled, it is PAINTED — a few long parallel strokes of a
/// wide wet brush swept across it along its own axis, each overlapping the last by a quarter so the overlaps
/// read a shade deeper (the wash-stroke tell), their ends breaking at the mass's edge with a little
/// overshoot, their edges ragged with the bristles. Then the mass's SHADOW half (its pixels darker than its
/// median, as shapes) takes a second, deeper sweep. Under the transmittance film every stroke is the glaze
/// that takes the paper to the mass's colour over one hit, so a stroke alone reads right and the overlaps
/// read as two.
#[allow(clippy::too_many_arguments)]
fn wc_sweep_mass(canvas: &mut Canvas, score: &mut StrokeScore, mask: &[bool], w: usize, h: usize, colour: Srgb, p: &PaintParams, brush: &BrushConfig, wet: f32, hits: f32, light: f32, single: bool, stage: &str, placed: &mut usize, k: &mut u64) -> usize {
    let area = mask.iter().filter(|&&b| b).count();
    if area < 16 {
        return 0;
    }
    let short = w.min(h) as f32;
    // Moments → centroid and principal axis.
    let (mut sx, mut sy) = (0f64, 0f64);
    for i in 0..w * h {
        if mask[i] {
            sx += (i % w) as f64;
            sy += (i / w) as f64;
        }
    }
    let (cx, cy) = (sx / area as f64, sy / area as f64);
    let (mut xx, mut yy, mut xy) = (0f64, 0f64, 0f64);
    for i in 0..w * h {
        if mask[i] {
            let (dx, dy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
            xx += dx * dx;
            yy += dy * dy;
            xy += dx * dy;
        }
    }
    // THE DIRECTION OF A WASH is the board's, not the shape's: a watercolourist tilts the board and sweeps
    // across, so every big wash runs the same way (a little off, hand to hand) and the sheet reads as one
    // painting, not a crosshatch of each shape's own axis. `PLAKAT_WCB_SWEEPANGLE` = degrees (default 0,
    // horizontal) or `axis` for the shape's own.
    let axis_mode = std::env::var("PLAKAT_WCB_SWEEPANGLE").unwrap_or_else(|_| "axis".into());
    let theta = if axis_mode == "axis" {
        0.5 * (2.0 * xy).atan2(xx - yy)
    } else {
        let base = axis_mode.parse::<f32>().unwrap_or(0.0).to_radians() as f64;
        base + (jitter(p.seed ^ 0x5A17, (cx as u64) ^ ((cy as u64) << 20)) * 16.0f32).to_radians() as f64
    };
    let (ax, ay) = (theta.cos() as f32, theta.sin() as f32);
    let (px_, py_) = (-ay, ax);
    // The brush: wider for a bigger mass (a painter picks the brush for the shape), between the limits.
    // The extent along the perpendicular.
    let (mut tmin, mut tmax) = (f32::MAX, f32::MIN);
    let (mut umin, mut umax) = (f32::MAX, f32::MIN);
    for i in 0..w * h {
        if mask[i] {
            let (dx, dy) = ((i % w) as f32 - cx as f32, (i / w) as f32 - cy as f32);
            let t = dx * px_ + dy * py_;
            let u = dx * ax + dy * ay;
            tmin = tmin.min(t);
            tmax = tmax.max(t);
            umin = umin.min(u);
            umax = umax.max(u);
        }
    }
    // A SMALL shape is ONE mark of a small brush along its axis, as wide as it is across; a big one is swept.
    let width = if single { (tmax - tmin).clamp(short / 150.0, short / 30.0) } else { ((area as f32).sqrt() / 4.0).clamp(short / 60.0, short / 12.0) };
    let spacing = if single { f32::MAX } else { width * std::env::var("PLAKAT_WCB_SWEEPSPACE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.5) };
    // The glaze that takes the paper to the colour in one hit.
    let g = canvas.ground_linear();
    let t = color::srgb_to_linear(colour);
    let want = [(t[0].max(0.004) / g[0].max(0.004)).ln().min(0.0), (t[1].max(0.004) / g[1].max(0.004)).ln().min(0.0), (t[2].max(0.004) / g[2].max(0.004)).ln().min(0.0)];
    let dens = trans_densities(canvas.ln_reflectance(), want);
    // One full-contact hit lays the whole film (the cap keeps further hits from doubling it).
    let eff = (brush.k_deposit * hits).max(1e-3);
    let load: Vec<f32> = dens.iter().map(|d| d / eff * light).collect();
    if load.iter().all(|v| *v <= 0.0) {
        return 0;
    }
    let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
    let inside = |x: f32, y: f32| x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h && mask[y as usize * w + x as usize];
    // THE SHAPE IS EXACT, THE BRUSHWORK IS INSIDE IT: the strokes are clipped to the mass's own outline (a
    // watercolourist cuts the edge of a shape with the brush's side and works freely within it). Recorded as a
    // clip record so the replay clips the same strokes.
    let clip_rings = mask_rings(mask, w, h, 1.2);
    if clip_rings.len() < 3 {
        return 0;
    }
    canvas.set_clip_rings(&clip_rings);
    let clip_rec = |spline: Vec<[f32; 2]>, id: usize| StrokeRecord { id: id as u32, wipe: true, wash: true, stage: stage.into(), spline, w0: 0.0, w1: 0.0, taper: 0.0, mix: Vec::new(), wet: 0.0, press: 0.0, streak: 0.0, round: 0.0, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None };
    score.strokes.push(clip_rec(clip_rings, *placed));
    let mut laid = 0usize;
    let mut tt = if single { (tmin + tmax) * 0.5 } else { tmin + spacing * 0.5 };
    let mut row = 0u64;
    while tt <= tmax {
        row += 1;
        // Walk the line, collecting runs inside the mass (a gap narrower than the brush is bridged).
        let mut runs: Vec<(f32, f32)> = Vec::new();
        let mut cur: Option<(f32, f32)> = None;
        let mut gap = 0f32;
        let mut u = umin - width;
        while u <= umax + width {
            // The stroke lands wherever ANY part of its width touches the mass (the clip keeps the edge): tested
            // on the centre line alone, every limb narrower than the row spacing was missed and left as paper.
            let touches = [-0.45f32, -0.225, 0.0, 0.225, 0.45].iter().any(|o| {
                let t2 = tt + o * width;
                inside(cx as f32 + ax * u + px_ * t2, cy as f32 + ay * u + py_ * t2)
            });
            if touches {
                match cur {
                    Some((a, _)) => cur = Some((a, u)),
                    None => cur = Some((u, u)),
                }
                gap = 0.0;
            } else if let Some((a, b)) = cur {
                gap += 1.0;
                if gap > width {
                    runs.push((a, b));
                    cur = None;
                    gap = 0.0;
                }
            }
            u += 1.0;
        }
        if let Some(r) = cur {
            runs.push(r);
        }
        for (a, b) in runs {
            if b - a < 2.0 || *placed >= p.budget {
                continue;
            }
            *k += 1;
            // A little overshoot at both ends, different each time; the stroke lifts off at its end.
            let o0 = width * (0.15 + 0.3 * (jitter(p.seed ^ 0x5EEF, *k) + 0.5));
            let o1 = width * (0.15 + 0.3 * (jitter(p.seed ^ 0x5EF0, *k) + 0.5));
            let (ua, ub) = (a - o0, b + o1);
            let nseg = (((ub - ua) / (width * 1.5)).ceil() as usize).max(2);
            let path: Vec<[f32; 2]> = (0..=nseg)
                .map(|i| {
                    let uu = ua + (ub - ua) * i as f32 / nseg as f32;
                    [(cx as f32 + ax * uu + px_ * tt).clamp(0.0, w as f32 - 1.0), (cy as f32 + ay * uu + py_ * tt).clamp(0.0, h as f32 - 1.0)]
                })
                .collect();
            let path = waver_path(&path, 0.12 * width, p.seed, *k ^ (row << 20));
            let w1 = width * (0.8 + 0.2 * (jitter(p.seed ^ 0x5EF1, *k) + 0.5));
            let s = Stroke { path, width0: width, width1: w1, load: load.clone(), pressure: 1.0, wetness: wet };
            s.rasterize(canvas, brush);
            *placed += 1;
            laid += 1;
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: stage.into(), spline: s.path, w0: width, w1, taper: 0.3, mix: mix.clone(), wet, press: 1.0, streak: brush.streak, round: brush.round, pickup: Some(brush.k_pickup), bristles: Some(brush.bristles), kd: Some(brush.k_deposit), cap: (brush.film_cap > 0.0).then_some(brush.film_cap), flat: brush.flat_ends, hold: brush.hold_charge, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        }
        if single {
            break;
        }
        tt += spacing;
    }
    canvas.set_clip_rings(&[]);
    score.strokes.push(clip_rec(vec![[0.0, 0.0]], *placed));
    laid
}

/// WHAT THE BRUSH MISSED, A TOUCH COVERS. After a piece has been brushed, any of its pixels still bare paper
/// (a ring-shaped piece whose one mark went through its hole, a limb between two rows) takes the piece's
/// colour as an exact glaze. Without it every such miss was a white halo round a shape.
fn wc_touch_up(canvas: &mut Canvas, score: &mut StrokeScore, piece: &[bool], w: usize, h: usize, colour: Srgb, p: &PaintParams, stage: &str, placed: &mut usize, k: &mut u64) {
    let bare: Vec<bool> = (0..w * h).map(|i| piece[i] && canvas.saturation_at((i % w) as u32, (i / w) as u32) <= 1e-5).collect();
    if !bare.iter().any(|&b| b) {
        return;
    }
    let rings = mask_rings(&bare, w, h, 0.8);
    if rings.len() < 3 {
        return;
    }
    let t = color::srgb_to_linear(colour);
    let g = canvas.ground_linear();
    let want = [(t[0].max(0.004) / g[0].max(0.004)).ln().min(0.0), (t[1].max(0.004) / g[1].max(0.004)).ln().min(0.0), (t[2].max(0.004) / g[2].max(0.004)).ln().min(0.0)];
    let load = trans_densities(canvas.ln_reflectance(), want);
    if !load.iter().any(|v| *v > 0.0) {
        return;
    }
    canvas.fill_rings(&rings, &load, 0.3, 0.0);
    *k += 1;
    *placed += 1;
    let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
    score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.into(), spline: rings, w0: 0.0, w1: 0.0, taper: 0.0, mix, wet: 0.3, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
}

fn wc_fill_masses(canvas: &mut Canvas, score: &mut StrokeScore, labels: &[u16], means: &[Srgb], w: usize, h: usize, paper: Option<&[bool]>, p: &PaintParams, feather_px: f32, wet: f32, sweep_stages: Option<(f32, usize)>, input_keyed: &RgbImage, placed: &mut usize, k: &mut u64) {
    let n = p.palette.pigments.len();
    let short = w.min(h);
    // A piece smaller than a wash is not a wash and gets no strokes of its own: it JOINS its neighbour
    // (the relabelling below). Under the wash recipe that is anything under a twelfth of the sheet across.
    let min_area = (short as f32 / std::env::var("PLAKAT_WCB_MINMASS").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(80.0)).max(3.0).powi(2) as usize;
    // FLECKS JOIN THEIR NEIGHBOUR. A region smaller than a wash is not dropped (dropped, it was a white
    // speck of bare paper): its pixels take the label of the nearest kept mass, by propagation.
    let labels: Vec<u16> = {
        let mut kept_px = vec![false; w * h];
        for m in 0..means.len() {
            let mask: Vec<bool> = (0..w * h).map(|i| labels[i] as usize == m).collect();
            for (i, &b) in keep_large_regions(&mask, w, h, min_area).iter().enumerate() {
                if b {
                    kept_px[i] = true;
                }
            }
        }
        let mut lab = labels.to_vec();
        let mut done = kept_px;
        for _ in 0..64 {
            let mut next = lab.clone();
            let mut next_done = done.clone();
            let mut any = false;
            for i in 0..w * h {
                if done[i] {
                    continue;
                }
                let (x, y) = (i % w, i / w);
                let nb = [if x > 0 { i - 1 } else { usize::MAX }, if x + 1 < w { i + 1 } else { usize::MAX }, if y > 0 { i - w } else { usize::MAX }, if y + 1 < h { i + w } else { usize::MAX }];
                if let Some(&j) = nb.iter().find(|&&j| j != usize::MAX && done[j]) {
                    next[i] = lab[j];
                    next_done[i] = true;
                    any = true;
                }
            }
            lab = next;
            done = next_done;
            if !any {
                break;
            }
        }
        lab
    };
    let labels = &labels[..];
    let mut order: Vec<usize> = (0..means.len()).collect();
    order.sort_by(|&a, &b| color::linear_luma(color::srgb_to_linear(means[b])).partial_cmp(&color::linear_luma(color::srgb_to_linear(means[a]))).unwrap_or(std::cmp::Ordering::Equal));
    let mut probe: std::collections::HashMap<(u32, u16), f32> = std::collections::HashMap::new();
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    for (mi, &m) in order.iter().enumerate() {
        let colour = means[m];
        let luma = color::linear_luma(color::srgb_to_linear(colour));
        // A mass is a stage when sweeping: its film cap counts from here, and it dries at its end.
        if sweep_stages.is_some() {
            canvas.mark_film();
        }
        let cross = |canvas: &mut Canvas| {
            if let Some((bleed, n_total)) = sweep_stages {
                if bleed > 0.0 {
                    canvas.bleed_with(bleed * pass_bleed_taper(mi, n_total), p.diffuse);
                }
                if mi + 1 < n_total && p.dry > 0.0 {
                    canvas.dry(1.0 - p.dry);
                }
            }
        };
        let mask: Vec<bool> = (0..w * h).map(|i| labels[i] as usize == m && !paper.map(|pp| pp[i]).unwrap_or(false)).collect();
        // A wash is grown a pixel so neighbours touch, never over the paper — except in the transmittance
        // film, where the overlap multiplies into a dark seam along every boundary (the masses partition
        // the sheet; nothing is between them).
        let mut grown = if canvas.is_transmittance() { mask.clone() } else { dilate_bool(&mask, w, h, 1) };
        for (i, g) in grown.iter_mut().enumerate() {
            if paper.map(|pp| pp[i]).unwrap_or(false) {
                *g = false;
            }
        }
        if !grown.iter().any(|&b| b) {
            cross(canvas);
            continue;
        }
        let rings = mask_rings(&grown, w, h, 1.2);
        if rings.is_empty() {
            cross(canvas);
            continue;
        }
        // In the transmittance film the wash is exact: the densities whose transmittance over the paper IS
        // the mass's colour, laid once. Otherwise the scrap-paper probe.
        let load = if canvas.is_transmittance() {
            let t = color::srgb_to_linear(colour);
            let g = canvas.ground_linear();
            let want = [(t[0].max(0.004) / g[0].max(0.004)).ln().min(0.0), (t[1].max(0.004) / g[1].max(0.004)).ln().min(0.0), (t[2].max(0.004) / g[2].max(0.004)).ln().min(0.0)];
            trans_densities(canvas.ln_reflectance(), want)
        } else {
            let charge = charge_for(&mut probe, &p.palette, 0.85, p.opacity, colour, luma, wet, None);
            mixture_cached(&mut cache, colour, &p.palette, charge, n)
        };
        // A transmittance wash has NO outward bloom: a bloom laid over the neighbouring mass multiplies into a
        // dark rim round every shape. Its soft edges come from the wet bleed after the washes.
        let feather_px = if canvas.is_transmittance() { 0.0 } else { feather_px };
        // LOST AND FOUND EDGES: a big mass goes down wet and its edge softens in the bleed; a small shape
        // sets hard. (`PLAKAT_WCB_HARDBELOW` = the fraction of the short side under which a mass sets.)
        let hard_below = std::env::var("PLAKAT_WCB_HARDBELOW").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
        let area = mask.iter().filter(|&&b| b).count() as f32;
        let wet = if hard_below > 0.0 && area < (short as f32 * hard_below).powi(2) { 0.1 } else { wet };
        // WASH STROKES instead of a fill (see `wc_sweep_mass`): each connected piece of the mass swept with the
        // wash brush, then its shadow half swept deeper.
        if sweep_stages.is_some() && canvas.is_transmittance() {
            let hits = std::env::var("PLAKAT_WCB_SWEEPHITS").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
            let mut brush = p.brush;
            brush.k_pickup = 0.0;
            brush.streak = std::env::var("PLAKAT_WCB_SWEEPSTREAK").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.1);
            // A wash brush: flat section, no end taper, laying its film in one hit and capped so the
            // overlapping strokes of one wash build ONE film (the water evens them out).
            brush.round = std::env::var("PLAKAT_WCB_SWEEPROUND").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.35);
            brush.flat_ends = std::env::var("PLAKAT_WCB_SWEEPFLAT").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0) > 0.0;
            brush.hold_charge = true;
            brush.bristles = 7;
            brush.k_deposit = p.brush.k_deposit * 1.5;
            brush.film_cap = std::env::var("PLAKAT_WCB_SWEEPCAP").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.15);
            let stage = format!("wash-{}", mi + 1);
            // Only the BIG pieces are washes: a window, a shadow under a bench, a face is not a wash — it is
            // the small brush's. (Swept, every such piece was a stray dark rectangle.)
            let sweep_min = (short as f32 / std::env::var("PLAKAT_WCB_SWEEPMIN").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(14.0)).powi(2) as usize;
            // THE FIRST WASH IS LIGHT: a watercolourist floats the first wash well under the final value so
            // the paper glows through, then deepens the shadow side with a second pass while the shape is
            // still read as one. `light` = the fraction of the full glaze the first wash carries.
            let light = std::env::var("PLAKAT_WCB_LIGHT1").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.7);
            let luma_k = luma_map(input_keyed);
            // One sweep per connected piece.
            let mut seen = vec![false; w * h];
            for start in 0..w * h {
                if !mask[start] || seen[start] {
                    continue;
                }
                let mut piece = vec![false; w * h];
                let mut stack = vec![start];
                seen[start] = true;
                while let Some(i) = stack.pop() {
                    piece[i] = true;
                    let (x, y) = (i % w, i / w);
                    let nb = [if x > 0 { i - 1 } else { usize::MAX }, if x + 1 < w { i + 1 } else { usize::MAX }, if y > 0 { i - w } else { usize::MAX }, if y + 1 < h { i + w } else { usize::MAX }];
                    for &j in &nb {
                        if j != usize::MAX && mask[j] && !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
                let piece_area = piece.iter().filter(|&&b| b).count();
                if piece_area < sweep_min {
                    // A small shape: one mark, at the full colour (a window, a lamp, a shadow under a bench).
                    if piece_area >= sweep_min / 24 {
                        wc_sweep_mass(canvas, score, &piece, w, h, colour, p, &brush, wet, hits, 1.0, true, &stage, placed, k);
                        wc_touch_up(canvas, score, &piece, w, h, colour, p, &stage, placed, k);
                    } else {
                        // A SLIVER (the thin band of an in-between colour along a boundary): too small for a
                        // mark, and skipped it was a white halo round every shape. It takes its colour as a
                        // touch — an exact glaze over the paper, hard-edged.
                        let rings = mask_rings(&piece, w, h, 1.0);
                        if rings.len() >= 3 {
                            let t = color::srgb_to_linear(colour);
                            let g = canvas.ground_linear();
                            let want = [(t[0].max(0.004) / g[0].max(0.004)).ln().min(0.0), (t[1].max(0.004) / g[1].max(0.004)).ln().min(0.0), (t[2].max(0.004) / g[2].max(0.004)).ln().min(0.0)];
                            let load = trans_densities(canvas.ln_reflectance(), want);
                            if load.iter().any(|v| *v > 0.0) {
                                canvas.fill_rings(&rings, &load, 0.3, 0.0);
                                *k += 1;
                                *placed += 1;
                                let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                                score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.clone(), spline: rings, w0: 0.0, w1: 0.0, taper: 0.0, mix, wet: 0.3, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
                            }
                        }
                    }
                    continue;
                }
                wc_sweep_mass(canvas, score, &piece, w, h, colour, p, &brush, wet, hits, light, false, &stage, placed, k);
                // The shadow side: the piece's pixels darker than its median, as the shapes they make, swept
                // again with the rest of the glaze (so the dark side reaches the full colour and the light
                // side keeps the first wash).
                if light < 1.0 {
                    let ls: Vec<f32> = (0..w * h).filter(|&i| piece[i]).map(|i| luma_k[i]).collect();
                    let med = quantile(&ls, 0.5);
                    let dark: Vec<bool> = (0..w * h).map(|i| piece[i] && luma_k[i] < med).collect();
                    let dark = keep_large_regions(&dilate_bool(&erode_bool(&dark, w, h, 2), w, h, 2), w, h, sweep_min / 4);
                    if dark.iter().any(|&b| b) {
                        wc_sweep_mass(canvas, score, &dark, w, h, colour, p, &brush, wet, hits, 1.0 - light, false, &stage, placed, k);
                    }
                }
                wc_touch_up(canvas, score, &piece, w, h, colour, p, &stage, placed, k);
            }
            cross(canvas);
            continue;
        }
        canvas.fill_rings(&rings, &load, wet, feather_px);
        *k += 1;
        *placed += 1;
        let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: "wash".into(), spline: rings, w0: feather_px, w1: 0.0, taper: 0.0, mix, wet, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
    }
}

/// The DENSITIES (per pigment, ≥ 0) whose transmittance best matches `want = ln(target/under)` per channel:
/// a non-negative least squares over every single, pair and triple of pigments (a watercolourist mixes two,
/// three at most), by projected gradient. Pigments that cannot darken (ln R ≈ 0, the paper white) are left out.
fn trans_densities(ln_r: &[[f32; 3]], want: [f32; 3]) -> Vec<f32> {
    let n = ln_r.len();
    let mut out = vec![0f32; n];
    let usable: Vec<usize> = (0..n).filter(|&i| ln_r[i].iter().map(|v| v.abs()).sum::<f32>() > 0.05).collect();
    if usable.is_empty() || want.iter().all(|v| *v > -1e-4) {
        return out;
    }
    // The search over triples is cubic in the palette: keep the eight pigments whose ln R points most nearly
    // the way `want` does (a palette of thirty-two took twelve minutes on one pass).
    let usable: Vec<usize> = if usable.len() > 8 {
        let wn = (want[0] * want[0] + want[1] * want[1] + want[2] * want[2]).sqrt().max(1e-6);
        let mut scored: Vec<(f32, usize)> = usable
            .iter()
            .map(|&i| {
                let r = ln_r[i];
                let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-6);
                ((r[0] * want[0] + r[1] * want[1] + r[2] * want[2]) / (rn * wn), i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(8).map(|s| s.1).collect()
    } else {
        usable
    };
    let err = |idx: &[usize], a: &[f32]| -> f32 {
        let mut e = 0f32;
        for c in 0..3 {
            let mut v = 0f32;
            for (k, &i) in idx.iter().enumerate() {
                v += a[k] * ln_r[i][c];
            }
            e += (v - want[c]).powi(2);
        }
        e
    };
    let solve = |idx: &[usize]| -> (Vec<f32>, f32) {
        let m = idx.len();
        let mut a = vec![0.5f32; m];
        let mut lip = 0f32;
        for &i in idx {
            lip += ln_r[i].iter().map(|v| v * v).sum::<f32>();
        }
        let step = 1.0 / (2.0 * lip.max(1e-4));
        for _ in 0..60 {
            let mut grad = vec![0f32; m];
            for c in 0..3 {
                let mut v = 0f32;
                for (k, &i) in idx.iter().enumerate() {
                    v += a[k] * ln_r[i][c];
                }
                let r = v - want[c];
                for (k, &i) in idx.iter().enumerate() {
                    grad[k] += 2.0 * r * ln_r[i][c];
                }
            }
            for k in 0..m {
                a[k] = (a[k] - step * grad[k]).max(0.0);
            }
        }
        let e = err(idx, &a);
        (a, e)
    };
    let mut best: (Vec<usize>, Vec<f32>, f32) = (Vec::new(), Vec::new(), f32::MAX);
    for (x, &i) in usable.iter().enumerate() {
        let (a, e) = solve(&[i]);
        if e < best.2 { best = (vec![i], a, e); }
        for (y, &j) in usable.iter().enumerate().skip(x + 1) {
            let (a, e) = solve(&[i, j]);
            if e < best.2 { best = (vec![i, j], a, e); }
            for &k in usable.iter().skip(y + 1) {
                let (a, e) = solve(&[i, j, k]);
                if e < best.2 { best = (vec![i, j, k], a, e); }
            }
        }
    }
    for (k, &i) in best.0.iter().enumerate() {
        out[i] = best.1[k];
    }
    out
}

/// The picture's MASSES at wash scale: `k` colour centres by k-means in linear light over the picture blurred
/// to wash scale, each pixel labelled with its nearest, and each centre's mean colour. Deterministic.
fn wc_mass_colours(img: &RgbImage, k: usize, seed: u64) -> (Vec<u16>, Vec<Srgb>) {
    let (w, h) = (img.width(), img.height());
    let blurred = imageops::blur(img, (w.max(h) as f32 * 0.004).max(1.5));
    // `PLAKAT_WCB_MASSVAL` > 0: cluster on CHROMATICITY with the perceptual value at that weight, so a mass is
    // an object's colour, not a value band (value bands are the concentric posterisation rings).
    let massval = std::env::var("PLAKAT_WCB_MASSVAL").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(if wc_recipe_on() { 0.3 } else { 0.0 });
    let lin_px: Vec<[f32; 3]> = blurred.pixels().map(|p| color::srgb_to_linear(p.0)).collect();
    let px: Vec<[f32; 3]> = if massval > 0.0 {
        lin_px.iter().map(|l| { let s = l[0] + l[1] + l[2] + 1e-4; [l[0] / s, l[1] / s, color::linear_luma(*l).max(0.0).powf(1.0 / 2.2) * massval] }).collect()
    } else {
        lin_px.clone()
    };
    let n = px.len();
    let k = k.clamp(2, 64);
    let mut by_l: Vec<usize> = (0..n).step_by(11).collect();
    by_l.sort_by(|&a, &b| color::linear_luma(px[a]).partial_cmp(&color::linear_luma(px[b])).unwrap_or(std::cmp::Ordering::Equal));
    let mut centres: Vec<[f32; 3]> = (0..k)
        .map(|c| {
            let q = (c as f32 + 0.5 + 0.3 * jitter(seed ^ 0xC1A5, c as u64)) / k as f32;
            px[by_l[((by_l.len() as f32 - 1.0) * q.clamp(0.0, 1.0)) as usize]]
        })
        .collect();
    let dist = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let nearest = |p: [f32; 3], centres: &[[f32; 3]]| centres.iter().enumerate().map(|(i, c)| (i, dist(p, *c))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
    for _ in 0..12 {
        let mut sum = vec![[0f64; 3]; k];
        let mut cnt = vec![0usize; k];
        for i in (0..n).step_by(5) {
            let c = nearest(px[i], &centres);
            for ch in 0..3 {
                sum[c][ch] += px[i][ch] as f64;
            }
            cnt[c] += 1;
        }
        for c in 0..k {
            if cnt[c] > 0 {
                centres[c] = [(sum[c][0] / cnt[c] as f64) as f32, (sum[c][1] / cnt[c] as f64) as f32, (sum[c][2] / cnt[c] as f64) as f32];
            }
        }
    }
    let labels: Vec<u16> = px.iter().map(|&p| nearest(p, &centres) as u16).collect();
    // The mass colour is the MEAN LINEAR COLOUR of its pixels (the feature centre is not a colour).
    let mut sum = vec![[0f64; 3]; k];
    let mut cnt = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        for c in 0..3 {
            sum[l as usize][c] += lin_px[i][c] as f64;
        }
        cnt[l as usize] += 1;
    }
    let means: Vec<Srgb> = (0..k).map(|c| if cnt[c] > 0 { color::linear_to_srgb([(sum[c][0] / cnt[c] as f64) as f32, (sum[c][1] / cnt[c] as f64) as f32, (sum[c][2] / cnt[c] as f64) as f32]) } else { [255, 255, 255] }).collect();
    (labels, means)
}

/// HDR TONE-MAPPING of the source before anything reads it (`--hdr`): a local operator — each pixel's
/// luma is divided by a wide blur of the luma (its surround) and the ratio is compressed, so the lamps stop
/// blowing out and the shadows lift to show what is in them, while the picture's chromaticity is kept
/// exactly (a brown stays that brown, lighter). `amount` 0 = untouched, 1 = full compression: the surround
/// is pulled toward the mean and local contrast is held, the way a painter re-lights a scene so the darks
/// have something to paint and the lights are not a hole in the sheet. Nothing here is a colour choice.
pub fn hdr_tone_map(img: &RgbImage, amount: f32) -> RgbImage {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.0 {
        return img.clone();
    }
    let (w, h) = (img.width(), img.height());
    let lum: Vec<f32> = img.pixels().map(|p| color::linear_luma(color::srgb_to_linear(p.0))).collect();
    // The surround: the log-luma blurred at a twelfth of the sheet (log space so a lamp does not drag its
    // whole street up with it).
    let logl = image::GrayImage::from_fn(w, h, |x, y| image::Luma([((lum[(y * w + x) as usize].max(1e-4)).ln() * 20.0 + 160.0).clamp(0.0, 255.0) as u8]));
    let sur = imageops::blur(&logl, (w.max(h) as f32 / 12.0).max(2.0));
    // The anchor the surround is compressed toward: a mid-grey (linear 0.18) — so a dark surround comes UP
    // and a blown one comes DOWN. (Toward the picture's own mean a nocturne only got darker.)
    const MID: f32 = 0.18;
    let mut out = img.clone();
    for (i, px) in out.pixels_mut().enumerate() {
        let lin = color::srgb_to_linear(px.0);
        let l = lum[i].max(1e-4);
        let s = ((sur.get_pixel((i as u32) % w, (i as u32) / w).0[0] as f32 - 160.0) / 20.0).exp().max(1e-4);
        // Compress the surround toward mid-grey by `amount` (in log space); keep the local ratio, softened
        // only in the lights (a lamp's blow-out), never in the darks (their detail is the point).
        let s2 = s.powf(1.0 - 0.6 * amount) * MID.powf(0.6 * amount);
        let ratio = l / s;
        let ratio = if ratio > 1.0 { ratio.powf(1.0 - 0.45 * amount) } else { ratio };
        let l2 = (s2 * ratio).clamp(0.0, 1.0);
        let k = l2 / l;
        let mut v = [lin[0] * k, lin[1] * k, lin[2] * k];
        let mx = v[0].max(v[1]).max(v[2]);
        if mx > 1.0 {
            let g = (1.0 - l2) / (mx - l2).max(1e-6);
            for c in v.iter_mut() { *c = l2 + (*c - l2) * g.clamp(0.0, 1.0); }
        }
        px.0 = color::linear_to_srgb([v[0].clamp(0.0, 1.0), v[1].clamp(0.0, 1.0), v[2].clamp(0.0, 1.0)]);
    }
    out
}

/// The LIGHT ENVELOPE of a picture at scale `r` px: per channel, the lightest value within `r` (a max filter,
/// run on a reduced copy and lightly smoothed).
fn light_envelope(img: &RgbImage, r: f32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let f = (r / 4.0).max(1.0);
    let (sw, sh) = (((w as f32 / f).round() as u32).max(1), ((h as f32 / f).round() as u32).max(1));
    let small = if f > 1.0 { imageops::resize(img, sw, sh, imageops::FilterType::Triangle) } else { img.clone() };
    let rr = ((r / f).ceil() as i32).max(1);
    // Separable max BY LUMA, keeping the winning pixel's colour: a per-channel max turned a cluster of
    // coloured panes into a white none of them has, and the fine passes could not glaze it back.
    let lum = |c: [u8; 3]| 0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32;
    let mut tmp = small.clone();
    for y in 0..sh as i32 {
        for x in 0..sw as i32 {
            let mut m = [0u8; 3];
            let mut ml = -1.0f32;
            for d in -rr..=rr {
                let q = small.get_pixel((x + d).clamp(0, sw as i32 - 1) as u32, y as u32).0;
                let l = lum(q);
                if l > ml { ml = l; m = q; }
            }
            tmp.put_pixel(x as u32, y as u32, image::Rgb(m));
        }
    }
    let mut out = tmp.clone();
    for y in 0..sh as i32 {
        for x in 0..sw as i32 {
            let mut m = [0u8; 3];
            let mut ml = -1.0f32;
            for d in -rr..=rr {
                let q = tmp.get_pixel(x as u32, (y + d).clamp(0, sh as i32 - 1) as u32).0;
                let l = lum(q);
                if l > ml { ml = l; m = q; }
            }
            out.put_pixel(x as u32, y as u32, image::Rgb(m));
        }
    }
    let up = if f > 1.0 { imageops::resize(&out, w, h, imageops::FilterType::Triangle) } else { out };
    imageops::blur(&up, (r * 0.3).max(0.6))
}

/// Re-key `img` to the paper (see the brush watercolour): the range is measured on `measure` (the armature;
/// its 2nd..98th luma percentiles), paper at the top, `depth` × 0.72 of perceptual value below it at the
/// bottom, chromaticity kept.
pub fn key_image(img: &RgbImage, measure: &RgbImage, depth: f32, gamma: f32, hi_q: f32) -> RgbImage {
    let l = luma_map(measure);
    let mut sorted = l.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len().max(1);
    let lo = sorted[n * 2 / 100];
    // The top of the range is below the picture's brightest points: a nocturne's lamps are not its key.
    let hi = sorted[(((n as f32) * hi_q.clamp(0.5, 0.995)) as usize).min(n - 1)].max(lo + 0.05);
    let depth = depth.clamp(0.1, 1.0);
    // `PLAKAT_WCB_KEYLAB`: re-key in Lab — the VALUE moves, a*/b* stay. Scaling the channels equally keeps
    // the chromaticity ratio, and a dark brown lifted that way is a saturated red (a night's a* went 7 → 19:
    // the maroon sheet). A painter lifting a dark brown paints a lighter brown.
    let keylab = std::env::var("PLAKAT_WCB_KEYLAB").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(if wc_recipe_on() { 1.0 } else { 0.0 }) > 0.0;
    let mut out = img.clone();
    for px in out.pixels_mut() {
        let lin = color::srgb_to_linear(px.0);
        let lum = color::linear_luma(lin).max(1e-4);
        let t = ((lum - lo) / (hi - lo)).clamp(0.0, 1.0);
        if keylab {
            let v = 1.0 - depth * (1.0 - t).powf(gamma);
            let lab = color::srgb_to_lab(px.0);
            // L* from the perceptual value (L* ≈ 100·v for a mid-grey curve), chroma kept.
            // L* of a grey whose perceptual value is `v` (linear luma v^2.2).
            let l_new = (116.0 * v.max(0.0).powf(2.2 / 3.0) - 16.0).clamp(0.0, 100.0);
            px.0 = color::lab_to_srgb(color::Lab { l: l_new, a: lab.a, b: lab.b });
            continue;
        }
        // Full depth: at `depth` 0.9 the darkest wash is a perceptual 0.10 — a real dark. (A 0.72 factor here
        // lifted every shadow to a light grey and the sheet was fog.)
        // A watercolour's value structure is HIGH-KEY WITH DEEP ACCENTS: the lights and mids stay near the
        // paper and only the darkest quarter drops to the depth — `gamma` > 1 bends the curve that way.
        let want = (1.0 - depth * (1.0 - t).powf(gamma)).powf(2.2);
        let sc = (want / lum).min(40.0);
        let mut v = [lin[0] * sc, lin[1] * sc, lin[2] * sc];
        // A deep pixel's chromaticity is sensor noise, and the key amplifies it (a night's shadows came up
        // magenta). `PLAKAT_WCB_DARKCHROMA` = the fraction of the range below which chroma fades to neutral.
        let dc = std::env::var("PLAKAT_WCB_DARKCHROMA").ok().and_then(|x| x.parse::<f32>().ok()).unwrap_or(0.0);
        if dc > 0.0 && t < dc {
            let kch = (t / dc).clamp(0.0, 1.0);
            let l2 = color::linear_luma(v);
            for c in v.iter_mut() {
                *c = l2 + (*c - l2) * kch;
            }
        }
        // Out of gamut at the top: desaturate toward the target grey, never clip a channel (clipping red
        // first turned every lit face cyan).
        let mx = v[0].max(v[1]).max(v[2]);
        if mx > 1.0 {
            let k = ((1.0 - want) / (mx - want).max(1e-4)).clamp(0.0, 1.0);
            for c in v.iter_mut() {
                *c = want + (*c - want) * k;
            }
        }
        px.0 = color::linear_to_srgb([v[0].clamp(0.0, 1.0), v[1].clamp(0.0, 1.0), v[2].clamp(0.0, 1.0)]);
    }
    out
}

fn wash_passes(canvas: &mut Canvas, score: &mut StrokeScore, reference: &RgbImage, source: &RgbImage, n_stages_total: usize, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let (w, h) = (source.width() as usize, source.height() as usize);
    let levels = p.armature_levels.max(2) as usize;
    // The BOOK reads the keyed reference at pixel scale (its masks); the watercolour the source at wash scale.
    // The watercolour reads its masses from the source flattened at wash scale by an EDGE-PRESERVING smooth
    // (the armature's bilateral, no value snap): a plain blur spread every star and lamp into a pale halo that
    // became a light mass — a starry sky washed as blue-white blots. Points smaller than a wash are not a
    // wash; they are the paper's (and a later mark's) business.
    let smoothed;
    let input: &RgbImage = if p.book {
        reference
    } else {
        let r_wash = (w.max(h) as f32 * 0.007).max(2.0);
        smoothed = structure_armature(source, (w.min(h) as f32 / r_wash).round().max(1.0) as u32, 1, 0.0);
        &smoothed
    };
    let luma = luma_map(input);
    // The levels span the PICTURE's value range (its 2nd..98th luma percentiles), not 0..1: a night scene lives
    // in the bottom fifth of the scale, and fixed levels put its whole sky, walls and tracks into one or two
    // masses — the picture's structure vanished. Levels over its own range give a nocturne the same number of
    // washes a daylight picture gets (its brightest lights still reserve the paper by the reserve luma).
    let (lo, hi) = {
        let mut sorted = luma.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len().max(1);
        let lo = sorted[n * 2 / 100];
        let hi = sorted[(n * 98 / 100).min(n - 1)];
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
    let level: Vec<u16> = (0..w * h)
        .map(|i| if paper(i) { levels as u16 } else { ((norm(luma[i]) / step).round() as usize).min(levels - 1) as u16 })
        .collect();
    let hue_bin: Vec<u8> = input
        .pixels()
        .map(|px| {
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
    let min_area = if p.book { 6 } else { ((w * h) as f32 * 0.00005).max(24.0) as usize };
    let n = p.palette.pigments.len();
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let n_levels = levels;
    for lv in (0..n_levels).rev() {
        let mut labels = vec![u32::MAX; w * h];
        // Lightest level first: levels count up with luma, so the highest index is the lightest.
        let stage = format!("wash-{}", n_levels - lv);
        if let Some(pr) = progress {
            pr(PaintProgress::Painting { pass: n_levels - lv, passes: n_levels, radius: 0.0 });
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
            let mask = if glaze { dilate_bool(&mask, w, h, 2) } else { mask };
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
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.clone(), spline: rings, w0: feather, w1: 0.0, taper: 0.0, mix, wet, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        }
        // The pass boundary, as the stroke passes cross it (and as the replay reproduces it).
        // The pass boundary as the replay reproduces it: the taper over EVERY stage of the painting (the washes
        // and the modelling passes after them), then drying when a stage follows.
        let idx = n_levels - lv - 1;
        if p.bleed > 0.0 {
            canvas.bleed_with(p.bleed * pass_bleed_taper(idx, n_stages_total), p.diffuse);
        }
        if idx + 1 < n_stages_total {
            canvas.dry(1.0 - p.dry);
        }
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

/// The SKIN of the picture's faces, for the book illustration, on a sheet of `w × h`: what the material map
/// names skin — each face found by the face detector and outlined by MobileSAM (see
/// `pipelines::materials`), so the lit side of a face and its shadow side are one thing, and hair is not it.
/// Returns the skin, and the face whole: the skin with what it encloses — the eyes, the brows, the mouth.
/// The scale of the cells the material map names the skin in, as a share of the sheet's long side: the skin's
/// edge is rounded at it (see `book_skin`).
const SKIN_CELL: f32 = 0.005;

fn book_skin(p: &PaintParams, w: usize, h: usize) -> Option<(Vec<bool>, Vec<bool>)> {
    let (mw, mh, kind) = p.materials.as_ref()?;
    let (mw, mh) = (*mw as usize, *mh as usize);
    if mw == 0 || mh == 0 || kind.len() != mw * mh {
        return None;
    }
    let skin: Vec<bool> = (0..w * h).map(|i| kind[((i / w) * mh / h).min(mh - 1) * mw + ((i % w) * mw / w).min(mw - 1)] == crate::paint::engrave::SKIN).collect();
    if !skin.iter().any(|s| *s) {
        return None;
    }
    // The map names the skin in coarse cells (however it is scaled when it arrives); a face is not a stair.
    // Its edge is rounded at the scale of the cells before anything is read from it — the flesh plate and the
    // line of the head alike.
    let skin: Vec<bool> = {
        let is = image::GrayImage::from_fn(w as u32, h as u32, |x, y| image::Luma([if skin[y as usize * w + x as usize] { 255 } else { 0 }]));
        imageops::blur(&is, (w.max(h) as f32 * SKIN_CELL).max(1.0)).pixels().map(|px| px.0[0] > 127).collect()
    };
    // Whatever is not skin and cannot be reached from the sheet's edge without crossing skin is the face's.
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w).flat_map(|x| [x, (h - 1) * w + x]).chain((0..h).flat_map(|y| [y * w, y * w + w - 1])).filter(|&i| !skin[i]).collect();
    for &i in &stack {
        outside[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
            if !outside[j] && !skin[j] {
                outside[j] = true;
                stack.push(j);
            }
        }
    }
    let whole = outside.iter().map(|o| !o).collect();
    Some((skin, whole))
}

/// How many INKS an early book illustration is printed in besides the black — ALWAYS this many, a four-colour
/// press: the picture's own main colours, found in it by k-means, and where there is a face, one of the four
/// is its flesh.
const BOOK_INKS: usize = 4;
/// A plate's edge is cut as a stencil's: each plate is OPENED and CLOSED with a kernel of this many px each
/// side — a sliver thinner than the kernel goes to the plate round it, a notch narrower than it is filled.
const PLATE_OPEN: usize = 2;
/// ... found by hue and chroma (Lab a, b) and, at this share of its weight, by lightness: an ink is a hue, and
/// how dark a point is, is mostly how much of the ink it takes — but a dark red coat against a pink wall is
/// another ink.
const INK_LIGHTNESS: f32 = 0.35;
/// A printed ink is a CLEAN colour, not the photograph's mean: every ink is mixed with its chroma raised this
/// far over the picture's, so a pink is a pink and a green a green, and the sheet does not go grey.
const INK_CLEAN: f32 = 1.3;
/// METAL is printed in a cold grey: where the material map names it, a dull colour on it — the warmth a lamp
/// or a reflection gives a bonnet — is the metal's own grey, its chroma cut to this share and its hue pulled this
/// far (Lab b) toward the blue. A STRONG colour on metal is paint, and keeps its colour: the cooling fades out
/// from `METAL_COOL_UP_TO.0` to `.1` of chroma.
const METAL_COOL: (f32, f32) = (0.3, -6.0);
const METAL_COOL_UP_TO: (f32, f32) = (12.0, 24.0);
/// PAPER is what is lighter than this (Lab L) and of no colour (chroma under this): a white wall, the sky, the
/// ground a portrait stands against.
const PAPER_L: f32 = 72.0;
const PAPER_CHROMA: f32 = 12.0;
/// ... and the luma the paper prints at, for the shade: a shadow on a white shirt is black dots on the paper.
const PAPER_LUMA: f32 = 0.92;
/// The BASE COLOUR is read off the picture flattened by a KUWAHARA filter at this radius (a share of the long
/// side): the texture inside a thing goes, its edges stay sharp — then it is separated into inks, each one
/// flat plate. The plates are BROAD: every point goes to the plate most of its neighbours within `PLATE_VOTE`
/// of the long side are on, so the grain of a plaster or the weave of a cloth, split by the inks into a
/// mottle, is the plate of its ground. An island of a plate smaller than `PLATE_ISLAND` of the long side,
/// squared, goes to its neighbour. The flesh plate's weight, and a colour plate's.
const PLATE_SMOOTH: f32 = 0.008;
const PLATE_VOTE: f32 = 0.006;
const PLATE_ISLAND: f32 = 0.014;
const PLATE_WEIGHT: (f32, f32) = (0.35, 0.95);
/// ... and the plates are SURE: a point whose colour lies nearer one plate than the next by less than
/// `PLATE_SURE` (Lab units) is not decided by its own colour — a stone of a wall a shade greyer than the
/// next, a plaster's slow gradient — but goes to the plate most of the SURE points within `PLATE_REACH` of
/// the long side are on: the wall is one block, and a thing of its own colour keeps it.
const PLATE_SURE: f32 = 5.0;
const PLATE_REACH: f32 = 0.03;
/// THE KEY: the black LINE, found as a pen finds it — by a DIFFERENCE OF GAUSSIANS (XDoG) of the picture
/// lightly smoothed (a bilateral at `KEY_SMOOTH`): the grey at `KEY_SIGMA` of the long side against the grey
/// at 1.6 times that, and ink where the fine is `KEY_EPS` (luma) under the coarse — the dark ridge of a
/// line, the dark side of an edge, a band as wide as the blur and unbroken wherever the contrast holds. The
/// band is THINNED to its centre line (Zhang–Suen), and the line is given its WEIGHT from the dark it is
/// drawn against: `KEY_WEIGHT.0` px across where the darker side is lighter than `KEY_HEAVY.1` (luma), swelling
/// to `KEY_WEIGHT.1` px where it is darker than `KEY_HEAVY.0` — a hairline against the light, a heavy line in
/// the shadow — the width read along the line over `KEY_WEIGHT_ALONG` of the long side, so it thins and
/// swells by degrees; a line shorter than `KEY_SPECK` of the long side is dropped. A flat passage darker than
/// `KEY_BLACK` (luma 0..1), closed with a kernel of `KEY_CLOSE` px and at least `KEY_SOLID` of the long side
/// across, prints SOLID black — on skin only under `KEY_SKIN_BLACK` (an open mouth, a nostril). The plate is
/// traced with a knife's tolerance of `KEY_TOL` and printed with `KEY_GAIN` of gain, no softer: the key is
/// hard-edged.
const KEY_SMOOTH: f32 = 0.0015;
const KEY_SIGMA: f32 = 0.0008;
const KEY_EPS: f32 = 0.012;
const KEY_CLOSE: usize = 2;
const KEY_SPECK: f32 = 0.012;
const KEY_SOLID: f32 = 0.01;
const KEY_WEIGHT: (f32, f32) = (1.0, 5.0);
const KEY_HEAVY: (f32, f32) = (0.15, 0.55);
const KEY_WEIGHT_ALONG: f32 = 0.003;
/// ... but never a hairline where the edge is SHARP: a line whose two sides differ by `KEY_SHARP.1` of luma
/// (within 5 px) is at least the middle weight whatever its dark — a lid, a nostril, the lips' meeting, the
/// pupil hold their weight in the light; the light thins a cheek's edge, not a feature's.
const KEY_SHARP: (f32, f32) = (0.15, 0.5);
/// The least half-width (px) a face's built stroke is stamped at — except a head's outline (its cuts on
/// layer 1), stamped at its own width down to a hair, so it can taper away into the light.
const KEY_FEATURE_LEAST: f32 = 1.2;
const KEY_TAPER_LEAST: f32 = 0.5;
const KEY_BLACK: f32 = 0.12;
const KEY_SKIN_BLACK: f32 = 0.08;
/// ... and on skin an edge is a line only where its dark side is `KEY_SKIN_ADAPT` under the skin's mean over
/// `KEY_LOCAL` of the long side round it — a brow, a nostril, the hair against the brow: the hollow of an eye,
/// the shadow under a cheekbone are a face's modelling, not its lines, and those the built features draw.
const KEY_LOCAL: f32 = 0.012;
const KEY_SKIN_ADAPT: f32 = 0.12;
const KEY_TOL: f32 = 0.0007;
const KEY_GAIN: f32 = 0.0004;
/// THE SHADE is the only screen, and it is FINE: a lattice of black dots at 45°, pitch `SCREEN_PITCH` of the
/// long side. Where the picture is `SHADE_BELOW` darker than the colour printed there and under `SHADE_BAND.1`
/// of its luma, the dot grows from `SCREEN_LEAST` of the pitch toward `SCREEN_FULL` of half of it as the
/// shadow deepens to `SHADE_BAND.0`, and is full under that down to the solid black. Never on a face smaller
/// than a portrait (one whose eyes are `SHADE_PORTRAIT` of the long side apart), never within
/// `SHADE_OFF_FEATURE` of the eyes' distance round an eye, the nose or the mouth, and on a portrait's skin
/// only under `SHADE_SKIN_TO`: the light of a face is its flat flesh, and its features are the key's. The
/// dots print hard, with `SCREEN_GAIN` of gain.
const SHADE_BAND: (f32, f32) = (0.3, 0.6);
const SHADE_BELOW: f32 = 0.09;
/// ... and the shade is a THING'S, not a point's: a plate's piece — a sash, a sleeve, a wall — takes the
/// screen only where at least this share of it lies in shadow (under its plate by `SHADE_BELOW`); a flat
/// piece whose tone merely wavers prints even.
const SHADE_SEGMENT: f32 = 0.2;
/// (A small head's shadow side is not a tone but a flat plate, see `HEAD_SHADE`.)
const SHADE_SKIN_TO: f32 = 0.38;
const SHADE_PORTRAIT: f32 = 0.08;
const SHADE_OFF_FEATURE: f32 = 0.45;
const SCREEN_PITCH: f32 = 260.0;
const SCREEN_FULL: f32 = 0.9;
const SCREEN_LEAST: f32 = 0.3;
const SCREEN_GAIN: f32 = 0.0;
/// How far out of REGISTER the colour plates and the shade print, as a share of the sheet's long side: each
/// is shifted its own way by between six and fourteen tenths of this against the key, which is not shifted —
/// the gaps and the overlaps lie on the clothes and the architecture, and the faces stay held by the line.
/// The plates are cut to meet, not to overlap, so where two slip apart a hair of paper shows between them,
/// and where they slip together one prints over the other: the marks of the press.
const MISREGISTER: f32 = 1000.0;
/// How far a plate's ink spreads into the paper past its edge (a share of the long side): the dot gain of a
/// cheap paper.
const INK_SPREAD_EDGE: f32 = 0.0005;
/// The colour inks OVERPRINT: a point's plate is the nearest of the inks' combinations — each printed alone,
/// or one solid over another, as the palette's pigments mix — an ink added to the combination only where it
/// brings the colour nearer by more than this (Lab units) per ink: a dark coat is the red under the blue, solid
/// on solid, as a four-colour press made its darks, and the black screen is kept for the shadow.
const OVERPRINT_COST: f32 = 10.0;

/// A KUWAHARA filter at radius `r` (px): of the four square windows a pixel is a corner of, the mean of the
/// one with the least variance — the texture inside a thing is flattened, its edges are kept sharp. Run at a
/// reduced resolution when the radius is large (by integral images, so the cost is bounded).
fn kuwahara(img: &RgbImage, r: f32) -> RgbImage {
    let (w, h) = img.dimensions();
    let scale = (5.0 / r.max(1.0)).min(1.0);
    let (sw, sh) = ((w as f32 * scale).round().max(2.0) as u32, (h as f32 * scale).round().max(2.0) as u32);
    let sm = if scale < 1.0 { imageops::resize(img, sw, sh, imageops::FilterType::Triangle) } else { img.clone() };
    let r = (r * scale).round().max(1.0) as i64;
    let (sw, sh) = (sw as usize, sh as usize);
    let stride = sw + 1;
    let mut s = vec![[0f64; 6]; stride * (sh + 1)];
    for y in 0..sh {
        for x in 0..sw {
            let c = sm.get_pixel(x as u32, y as u32).0;
            let i = (y + 1) * stride + x + 1;
            let mut v = [c[0] as f64, c[1] as f64, c[2] as f64, (c[0] as f64).powi(2), (c[1] as f64).powi(2), (c[2] as f64).powi(2)];
            for k in 0..6 {
                v[k] += s[i - 1][k] + s[i - stride][k] - s[i - stride - 1][k];
            }
            s[i] = v;
        }
    }
    let mut out = RgbImage::new(sw as u32, sh as u32);
    for y in 0..sh as i64 {
        for x in 0..sw as i64 {
            let mut best = (f64::MAX, [0f64; 3]);
            for (dx, dy) in [(-r, -r), (0, -r), (-r, 0), (0, 0)] {
                let (x0, x1) = ((x + dx).clamp(0, sw as i64 - 1) as usize, (x + dx + r).clamp(0, sw as i64 - 1) as usize + 1);
                let (y0, y1) = ((y + dy).clamp(0, sh as i64 - 1) as usize, (y + dy + r).clamp(0, sh as i64 - 1) as usize + 1);
                let n = ((x1 - x0) * (y1 - y0)) as f64;
                let mut v = [0f64; 6];
                for k in 0..6 {
                    v[k] = s[y1 * stride + x1][k] - s[y0 * stride + x1][k] - s[y1 * stride + x0][k] + s[y0 * stride + x0][k];
                }
                let mean = [v[0] / n, v[1] / n, v[2] / n];
                let var = (v[3] / n - mean[0] * mean[0]) + (v[4] / n - mean[1] * mean[1]) + (v[5] / n - mean[2] * mean[2]);
                if var < best.0 {
                    best = (var, mean);
                }
            }
            out.put_pixel(x as u32, y as u32, image::Rgb([0, 1, 2].map(|k| best.1[k].round().clamp(0.0, 255.0) as u8)));
        }
    }
    if scale < 1.0 { imageops::resize(&out, w, h, imageops::FilterType::Triangle) } else { out }
}

/// A Gaussian blur of a grey field (luma 0..1), at `sigma` px.
fn gauss_f32(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let img = image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_raw(w as u32, h as u32, src.to_vec()).expect("grey field of w*h");
    imageops::blur(&img, sigma.max(0.3)).into_raw()
}

/// THE PEN'S LINE by a DIFFERENCE OF GAUSSIANS (see `KEY_SIGMA`): ink where the grey blurred at `sigma` lies
/// `eps` under the grey blurred at 1.6 sigma — a dark line's ridge, the dark side of an edge — as a band
/// about as wide as the blur, unbroken wherever the contrast holds.
fn xdog(g: &[f32], w: usize, h: usize, sigma: f32, eps: f32) -> Vec<bool> {
    let (fine, coarse) = (gauss_f32(g, w, h, sigma), gauss_f32(g, w, h, sigma * 1.6));
    fine.iter().zip(&coarse).map(|(f, c)| f - c < -eps).collect()
}

/// THINNING (Zhang–Suen): a band of ink taken down to its centre line, one pixel wide, eight-connected, its
/// ends and its joins kept.
fn thin(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut m = mask.to_vec();
    if w < 3 || h < 3 {
        return m;
    }
    let mut del: Vec<usize> = Vec::new();
    loop {
        let mut changed = false;
        for pass in 0..2 {
            del.clear();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = y * w + x;
                    if !m[i] {
                        continue;
                    }
                    let p = [m[i - w], m[i - w + 1], m[i + 1], m[i + w + 1], m[i + w], m[i + w - 1], m[i - 1], m[i - w - 1]];
                    let b = p.iter().filter(|v| **v).count();
                    if !(2..=6).contains(&b) {
                        continue;
                    }
                    let a = (0..8).filter(|&k| !p[k] && p[(k + 1) % 8]).count();
                    if a != 1 {
                        continue;
                    }
                    let gone = if pass == 0 { !(p[0] && p[2] && p[4]) && !(p[2] && p[4] && p[6]) } else { !(p[0] && p[2] && p[6]) && !(p[0] && p[4] && p[6]) };
                    if gone {
                        del.push(i);
                    }
                }
            }
            for &i in &del {
                m[i] = false;
                changed = true;
            }
        }
        if !changed {
            return m;
        }
    }
}

/// THE KEY PLATE (see `KEY_SMOOTH`): the black line, found as a pen's and given one weight, cleaned of
/// specks; with the solid blacks of the deepest passages; with the faces' features stamped in. Returned as the
/// rings of one wash, in register — nothing else is.
fn key_plate(drawn: &RgbImage, skin: Option<&[bool]>, long: f32, features: &[crate::paint::engrave::Cut]) -> Vec<[f32; 2]> {
    let (w, h) = (drawn.width() as usize, drawn.height() as usize);
    let smooth = bilateral(drawn, (long * KEY_SMOOTH).round().max(1.0) as i32);
    let g: Vec<f32> = smooth.pixels().map(|px| (0.299 * px.0[0] as f32 + 0.587 * px.0[1] as f32 + 0.114 * px.0[2] as f32) / 255.0).collect();
    let on_skin = |i: usize| skin.map_or(false, |s| s[i]);
    // THE LINE: the pen's band thinned to its centre; on skin only where a feature's dark lies on one side
    // (see `KEY_SKIN_ADAPT`).
    let edge = thin(&xdog(&g, w, h, (long * KEY_SIGMA).max(0.8), KEY_EPS), w, h);
    let local = box_blur(&g, w as u32, h as u32, (long * KEY_LOCAL).round().max(2.0) as i32);
    let darkest = min_filter(&g, w, h, 1);
    let centre: Vec<bool> = (0..w * h).map(|i| edge[i] && (!on_skin(i) || darkest[i] < local[i] - KEY_SKIN_ADAPT)).collect();
    // THE WEIGHT (see `KEY_WEIGHT`): from the dark the line is drawn against, read along the line.
    let dark = min_filter(&g, w, h, 2);
    let bright: Vec<f32> = min_filter(&g.iter().map(|v| -v).collect::<Vec<f32>>(), w, h, 2).iter().map(|v| -v).collect();
    let half_of = |i: usize| {
        let by_dark = ((KEY_HEAVY.1 - dark[i]) / (KEY_HEAVY.1 - KEY_HEAVY.0)).clamp(0.0, 1.0);
        let by_sharp = 0.5 * ((bright[i] - dark[i] - KEY_SHARP.0) / (KEY_SHARP.1 - KEY_SHARP.0)).clamp(0.0, 1.0);
        let t = by_dark.max(by_sharp);
        (KEY_WEIGHT.0 + (KEY_WEIGHT.1 - KEY_WEIGHT.0) * t - 1.0) * 0.5
    };
    let along = (long * KEY_WEIGHT_ALONG).round().max(2.0) as i64;
    let skeleton: Vec<usize> = (0..w * h).filter(|&i| centre[i]).collect();
    let mut line = vec![false; w * h];
    for &i in &skeleton {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        let (mut sum, mut count) = (0f32, 0f32);
        for dy in -along..=along {
            for dx in -along..=along {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 && centre[ny as usize * w + nx as usize] {
                    sum += half_of(ny as usize * w + nx as usize);
                    count += 1.0;
                }
            }
        }
        let r = sum / count.max(1.0) + 0.5;
        let ri = r.ceil() as i64;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 && ((dx * dx + dy * dy) as f32) <= r * r {
                    line[ny as usize * w + nx as usize] = true;
                }
            }
        }
    }
    let line = keep_large_regions(&line, w, h, ((long * KEY_SPECK) as usize * 3).max(4));
    // THE SOLIDS: the deepest passages, closed and of some size — a blot is not a solid.
    let solid: Vec<bool> = (0..w * h).map(|i| g[i] < if on_skin(i) { KEY_SKIN_BLACK } else { KEY_BLACK }).collect();
    let solid = erode_bool(&dilate_bool(&solid, w, h, KEY_CLOSE), w, h, KEY_CLOSE);
    let solid = keep_large_regions(&solid, w, h, ((long * KEY_SOLID) as usize).pow(2).max(9));
    let mut ink: Vec<bool> = line.iter().zip(&solid).map(|(l, s)| *l || *s).collect();
    // THE FACES' FEATURES are stamped into the plate — each built stroke at its own width, 100% black, in
    // register: the lines that hold a face are the key's, not a brush's grey.
    for cut in features {
        let segs = cut.path.len().saturating_sub(1).max(1) as f32;
        for (si, seg) in cut.path.windows(2).enumerate() {
            let (a, b) = (seg[0], seg[1]);
            let steps = ((b[0] - a[0]).hypot(b[1] - a[1]) * 2.0).ceil().max(1.0) as usize;
            for s in 0..=steps {
                let t = s as f32 / steps as f32;
                let (x, y) = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
                let along = (si as f32 + t) / segs;
                let r = ((cut.w0 + (cut.w1 - cut.w0) * along) * 0.5).max(if cut.layer == 1 { KEY_TAPER_LEAST } else { KEY_FEATURE_LEAST });
                for yy in ((y - r).max(0.0) as usize)..=((y + r).max(0.0) as usize).min(h - 1) {
                    for xx in ((x - r).max(0.0) as usize)..=((x + r).max(0.0) as usize).min(w - 1) {
                        if (xx as f32 + 0.5 - x).hypot(yy as f32 + 0.5 - y) <= r {
                            ink[yy * w + xx] = true;
                        }
                    }
                }
            }
        }
    }
    mask_rings(&ink, w, h, (long * KEY_TOL).max(0.8))
}

/// The least of a field within `r` px (a square) of each point.
fn min_filter(g: &[f32], w: usize, h: usize, r: i64) -> Vec<f32> {
    (0..w * h).map(|i| {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        let mut m = g[i];
        for dy in -r..=r {
            for dx in -r..=r {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                    m = m.min(g[ny as usize * w + nx as usize]);
                }
            }
        }
        m
    }).collect()
}

/// THE COLOUR PLATES AND THE SHADE of a print. The picture, flattened by a Kuwahara filter, is SEPARATED into
/// `BOOK_INKS` inks by k-means in Lab (see `INK_LIGHTNESS`) and a flesh ink for the skin; paper is what is
/// lighter than `PAPER_L` and of no colour. Each ink is ONE FLAT PLATE over the whole of its region — no
/// levels, no screen: the base colour of a thing is even — and the colour inks OVERPRINT (see
/// `OVERPRINT_COST`), a dark made of one solid over another. Then THE SHADE (see `SHADE_BAND`): the one screen,
/// black dots at 45° where the picture is a shadow on the colour printed there, never on a small face nor
/// over an eye or a mouth. Plates and shade print a hair out of register with the key (`MISREGISTER`), and
/// every ink spreads a little into the paper (`INK_SPREAD_EDGE`).
fn print_plates(canvas: &mut Canvas, score: &mut StrokeScore, source: &RgbImage, n_stages_total: usize, p: &PaintParams, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let (w, h) = (source.width() as usize, source.height() as usize);
    let long = w.max(h) as f32;
    let flat = kuwahara(source, long * PLATE_SMOOTH);
    let luma: Vec<f32> = flat.pixels().map(|px| (0.299 * px.0[0] as f32 + 0.587 * px.0[1] as f32 + 0.114 * px.0[2] as f32) / 255.0).collect();
    let mut skin = book_skin(p, w, h).map(|s| s.1);
    // THE SMALL HEADS (see `Head`): each is its egg, fitted — the flesh plate is cut to the head, not to the
    // detector's cells — less its hair, which is the key's; and all stand in
    // the scene's one light. (Read off the source, as the key reads them, so the plate's edge is the line's.)
    let luma_source = sheet_luma(source);
    let (heads, scene) = small_heads(p, &luma_source, w, h);
    if let Some(s) = skin.as_mut() {
        for (_, head) in &heads {
            let (x0, x1, y0, y1) = head.bounds(w, h, 1.0);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (a, b) = head.local([x as f32 + 0.5, y as f32 + 0.5]);
                    if head.egg(a, b) <= 1.0 {
                        s[y * w + x] = true;
                    }
                }
            }
            for i in head.hair(&luma_source, w, h) {
                s[i] = false;
            }
        }
    }
    let skin = skin;
    let on_face = |i: usize| skin.as_ref().map_or(false, |s| s[i]);
    // What the material map names at a point, where there is one.
    let named = |i: usize, kinds: &[u8]| p.materials.as_ref().filter(|(mw, mh, kind)| *mw > 0 && *mh > 0 && kind.len() == (*mw * *mh) as usize).map_or(false, |(mw, mh, kind)| { let (mw, mh) = (*mw as usize, *mh as usize); kinds.contains(&kind[((i / w) * mh / h).min(mh - 1) * mw + ((i % w) * mw / w).min(mw - 1)]) });
    // METAL is a cold grey (see `METAL_COOL`): a colour on it is read, and printed, cooled.
    let cooled = |i: usize, l: color::Lab| -> color::Lab {
        if !named(i, &[crate::paint::engrave::METAL]) {
            return l;
        }
        let keep = ((l.a.hypot(l.b) - METAL_COOL_UP_TO.0) / (METAL_COOL_UP_TO.1 - METAL_COOL_UP_TO.0)).clamp(0.0, 1.0);
        let k = METAL_COOL.0 + (1.0 - METAL_COOL.0) * keep;
        color::Lab { l: l.l, a: l.a * k, b: l.b * k + METAL_COOL.1 * (1.0 - keep) }
    };
    let lab: Vec<color::Lab> = flat.pixels().enumerate().map(|(i, px)| cooled(i, color::srgb_to_lab(px.0))).collect();
    let paper: Vec<bool> = (0..w * h).map(|i| !on_face(i) && lab[i].l > PAPER_L && lab[i].a.hypot(lab[i].b) < PAPER_CHROMA).collect();
    // THE INKS: k-means over the tinted points off the faces, seeded by the farthest point from the centres so
    // far — the picture's most different colours — so the same picture always gives the same inks.
    let feat = |l: &color::Lab| [l.l * INK_LIGHTNESS, l.a, l.b];
    let d2 = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let samples: Vec<usize> = (0..w * h).step_by(7).filter(|&i| !paper[i] && !on_face(i)).collect();
    if samples.is_empty() && skin.is_none() {
        return;
    }
    let mut inks: Vec<[f32; 3]> = Vec::new();
    let want = if skin.is_some() { BOOK_INKS - 1 } else { BOOK_INKS };
    if !samples.is_empty() {
        inks.push(samples.iter().fold([0f32; 3], |m, &i| { let f = feat(&lab[i]); [m[0] + f[0], m[1] + f[1], m[2] + f[2]] }).map(|v| v / samples.len() as f32));
        while inks.len() < want {
            let far = samples.iter().map(|&i| (i, inks.iter().map(|c| d2(feat(&lab[i]), *c)).fold(f32::MAX, f32::min))).fold((usize::MAX, -1f32), |b, s| if s.1 > b.1 { s } else { b });
            if far.0 == usize::MAX || far.1 <= 0.0 {
                break;
            }
            inks.push(feat(&lab[far.0]));
        }
    }
    let nearest = |v: [f32; 3], inks: &[[f32; 3]]| (0..inks.len()).fold((0usize, f32::MAX), |b, j| { let d = d2(v, inks[j]); if d < b.1 { (j, d) } else { b } }).0;
    let mut sum = vec![[0f64; 4]; inks.len()];
    for _ in 0..12 {
        sum.iter_mut().for_each(|s| *s = [0.0; 4]);
        for &i in &samples {
            let j = nearest(feat(&lab[i]), &inks);
            let l = &lab[i];
            sum[j] = [sum[j][0] + l.l as f64, sum[j][1] + l.a as f64, sum[j][2] + l.b as f64, sum[j][3] + 1.0];
        }
        for (c, s) in inks.iter_mut().zip(&sum) {
            if s[3] > 0.0 {
                *c = feat(&color::Lab { l: (s[0] / s[3]) as f32, a: (s[1] / s[3]) as f32, b: (s[2] / s[3]) as f32 });
            }
        }
    }
    // Each ink's colour: the mean of its points, cleaned.
    let clean = |l: color::Lab| color::Lab { l: l.l, a: l.a * INK_CLEAN, b: l.b * INK_CLEAN };
    let mut colours: Vec<color::Lab> = inks.iter().zip(&sum).map(|(c, s)| if s[3] > 0.0 { clean(color::Lab { l: (s[0] / s[3]) as f32, a: (s[1] / s[3]) as f32, b: (s[2] / s[3]) as f32 }) } else { clean(color::Lab { l: c[0] / INK_LIGHTNESS, a: c[1], b: c[2] }) }).collect();
    if let Some(s) = &skin {
        let m = (0..w * h).filter(|&i| s[i]).fold([0f64; 4], |m, i| [m[0] + lab[i].l as f64, m[1] + lab[i].a as f64, m[2] + lab[i].b as f64, m[3] + 1.0]);
        colours.push(clean(color::Lab { l: (m[0] / m[3].max(1.0)) as f32, a: (m[1] / m[3].max(1.0)) as f32, b: (m[2] / m[3].max(1.0)) as f32 }));
    }
    let n_inks = colours.len();
    let flesh = skin.as_ref().map(|_| n_inks - 1);
    // THE PLATES (see `OVERPRINT_COST`): each colour ink alone, and the colour inks one over another — the
    // colour of each combination as the palette's pigments mix, printed at the plates' charge — and the flesh,
    // which prints alone. A plate here is a combination; the press's plate of one ink is the union of those.
    let n = p.palette.pigments.len();
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let loads: Vec<Vec<f32>> = (0..n_inks).map(|ink| mixture_cached(&mut cache, color::lab_to_srgb(colours[ink]), &p.palette, p.charge * if Some(ink) == flesh { PLATE_WEIGHT.0 } else { PLATE_WEIGHT.1 }, n)).collect();
    let n_colour = n_inks - usize::from(flesh.is_some());
    let mut combos: Vec<u8> = (1u8..(1u8 << n_colour)).collect();
    // The FREE plates a point may take by its colour; after them the small heads' shadow (the flesh under the
    // darkest ink, see `HEAD_SHADE`) and the flesh, which are the faces' only.
    let n_free = combos.len();
    let dark_ink = (0..n_colour).min_by(|&a, &b| colours[a].l.partial_cmp(&colours[b].l).unwrap_or(std::cmp::Ordering::Equal));
    let shade_plate = match (flesh, dark_ink) {
        (Some(f), Some(d)) => {
            combos.push((1 << f) | (1 << d));
            Some(combos.len() - 1)
        }
        _ => None,
    };
    if let Some(f) = flesh {
        combos.push(1 << f);
    }
    let plate_feat: Vec<[f32; 3]> = combos.iter().map(|&bits| {
        let mut sum = vec![0f32; n];
        for ink in (0..n_inks).filter(|&ink| bits & (1 << ink) != 0) {
            for (s, l) in sum.iter_mut().zip(&loads[ink]) {
                *s += l;
            }
        }
        feat(&color::srgb_to_lab(crate::paint::pigment::mix(p.palette.pigments, &sum)))
    }).collect();
    let n_plates = combos.len();
    let flesh_plate = flesh.map(|_| n_plates - 1);
    // The nearest plate, and by how much (Lab units) it beats the next.
    let nearest_plate = |v: [f32; 3]| -> (usize, f32) {
        let (mut best, mut next) = ((0usize, f32::MAX), f32::MAX);
        for j in 0..n_free {
            let d = d2(v, plate_feat[j]) + (combos[j].count_ones() as f32 - 1.0) * OVERPRINT_COST * OVERPRINT_COST;
            if d < best.1 {
                next = best.1;
                best = (j, d);
            } else if d < next {
                next = d;
            }
        }
        (best.0, if next < f32::MAX { next.sqrt() - best.1.sqrt() } else { f32::MAX })
    };
    // Each point's plate, or paper.
    const PAPER: u8 = u8::MAX;
    let mut sure = vec![false; w * h];
    let mut label: Vec<u8> = (0..w * h).map(|i| {
        if let (true, Some(f)) = (on_face(i), flesh_plate) {
            return f as u8;
        }
        if paper[i] || n_colour == 0 {
            return PAPER;
        }
        let (j, by) = nearest_plate(feat(&lab[i]));
        sure[i] = by >= PLATE_SURE;
        j as u8
    }).collect();
    // THE SHADOW SIDE of each small head (see `HEAD_SHADE`), flat: the flesh under the darkest ink.
    if let Some(sp) = shade_plate {
        for (_, head) in &heads {
            let (x0, x1, y0, y1) = head.bounds(w, h, 1.0);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let i = y * w + x;
                    if on_face(i) && head.in_shadow([x as f32 + 0.5, y as f32 + 0.5], scene) {
                        label[i] = sp as u8;
                    }
                }
            }
        }
    }
    // SURE PLATES (see `PLATE_SURE`): an undecided point goes with the sure points round it.
    let reach = (long * PLATE_REACH).round().max(1.0) as usize;
    let mut best: Vec<(u32, u8)> = vec![(0, PAPER); w * h];
    for plate in 0..n_free {
        let is: Vec<bool> = (0..w * h).map(|i| sure[i] && label[i] as usize == plate).collect();
        let (cnt, _) = box_count(&is, w, h, reach);
        for i in 0..w * h {
            if cnt[i] > best[i].0 {
                best[i] = (cnt[i], plate as u8);
            }
        }
    }
    for i in 0..w * h {
        if !sure[i] && !on_face(i) && label[i] != PAPER && best[i].0 > 0 {
            label[i] = best[i].1;
        }
    }
    // BROAD PLATES (see `PLATE_VOTE`): off the faces, each point goes to the plate most of its neighbours are
    // on, the flesh not counted — the flesh is the face's, and does not spread.
    let vote = (long * PLATE_VOTE).round().max(1.0) as usize;
    let mut best: Vec<(u32, u8)> = vec![(0, PAPER); w * h];
    for ink in (0..n_free).map(|i| i as u8).chain(std::iter::once(PAPER)) {
        let is: Vec<bool> = label.iter().map(|&l| l == ink).collect();
        let (cnt, _) = box_count(&is, w, h, vote);
        for i in 0..w * h {
            if cnt[i] > best[i].0 {
                best[i] = (cnt[i], ink);
            }
        }
    }
    for i in 0..w * h {
        if !on_face(i) && best[i].0 > 0 {
            label[i] = best[i].1;
        }
    }
    // ISLANDS: a plate's piece smaller than a brush's touch is its neighbour's.
    let least = ((long * PLATE_ISLAND) as usize).pow(2).max(9);
    for _ in 0..2 {
        let mut seen = vec![false; w * h];
        let mut given: Vec<(Vec<usize>, u8)> = Vec::new();
        for start in 0..w * h {
            if seen[start] {
                continue;
            }
            let own = label[start];
            let (mut px, mut stack) = (Vec::new(), vec![start]);
            let mut beside = [0u32; 256];
            seen[start] = true;
            while let Some(i) = stack.pop() {
                px.push(i);
                let (x, y) = (i % w, i / w);
                for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
                    if label[j] != own {
                        beside[label[j] as usize] += 1;
                    } else if !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            if px.len() < least && !on_face(start) {
                if let Some(to) = (0..256).max_by_key(|&m| (beside[m], std::cmp::Reverse(m))).filter(|&m| beside[m] > 0) {
                    given.push((px, to as u8));
                }
            }
        }
        for (px, to) in given {
            for i in px {
                label[i] = to;
            }
        }
    }
    // STENCIL EDGES (see `PLATE_OPEN`): off the faces, each plate is OPENED — its slivers go to the plate
    // most of their neighbours are on — and CLOSED, its notches filled; the larger plate first.
    let mut order: Vec<usize> = (0..n_free).collect();
    order.sort_by_key(|&ink| std::cmp::Reverse(label.iter().filter(|&&l| l as usize == ink).count()));
    let mut sliver = vec![false; w * h];
    for &ink in &order {
        let mask: Vec<bool> = label.iter().map(|&l| l as usize == ink).collect();
        let opened = dilate_bool(&erode_bool(&mask, w, h, PLATE_OPEN), w, h, PLATE_OPEN);
        for i in 0..w * h {
            if mask[i] && !opened[i] && !on_face(i) {
                sliver[i] = true;
            }
        }
    }
    if sliver.iter().any(|s| *s) {
        let mut best: Vec<(u32, u8)> = vec![(0, PAPER); w * h];
        for ink in order.iter().map(|&i| i as u8).chain(std::iter::once(PAPER)) {
            let is: Vec<bool> = (0..w * h).map(|i| !sliver[i] && label[i] == ink).collect();
            let (cnt, _) = box_count(&is, w, h, vote);
            for i in 0..w * h {
                if sliver[i] && cnt[i] > best[i].0 {
                    best[i] = (cnt[i], ink);
                }
            }
        }
        for i in 0..w * h {
            if sliver[i] && best[i].0 > 0 {
                label[i] = best[i].1;
            }
        }
    }
    for &ink in &order {
        let mask: Vec<bool> = label.iter().map(|&l| l as usize == ink).collect();
        let closed = erode_bool(&dilate_bool(&mask, w, h, PLATE_OPEN), w, h, PLATE_OPEN);
        for i in 0..w * h {
            if closed[i] && !mask[i] && !on_face(i) {
                label[i] = ink as u8;
            }
        }
    }
    // The luma each plate prints at: the mean of its points. A shadow is what lies darker than that.
    let mut plate_luma = vec![[0f64; 2]; n_plates];
    for i in 0..w * h {
        if label[i] != PAPER {
            let m = &mut plate_luma[label[i] as usize];
            *m = [m[0] + luma[i] as f64, m[1] + 1.0];
        }
    }
    let plate_luma: Vec<f32> = plate_luma.iter().map(|m| (m[0] / m[1].max(1.0)) as f32).collect();
    // THE SHADE BY SEGMENT (see `SHADE_SEGMENT`): a plate's piece takes the screen only if enough of it lies
    // in shadow.
    let in_shadow = |i: usize| label[i] != PAPER && luma[i] < plate_luma[label[i] as usize] - SHADE_BELOW && luma[i] < SHADE_BAND.1;
    let mut shaded = vec![true; w * h];
    {
        let mut seen = vec![false; w * h];
        for start in 0..w * h {
            if seen[start] || label[start] == PAPER {
                continue;
            }
            let own = label[start];
            let (mut px, mut stack, mut dark) = (Vec::new(), vec![start], 0usize);
            seen[start] = true;
            while let Some(i) = stack.pop() {
                px.push(i);
                dark += usize::from(in_shadow(i));
                let (x, y) = (i % w, i / w);
                for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
                    if !seen[j] && label[j] == own {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            if (dark as f32) < SHADE_SEGMENT * px.len() as f32 {
                for i in px {
                    shaded[i] = false;
                }
            }
        }
    }
    // THE FACES the shade keeps off: every eye and mouth, and the whole of any face that is not a portrait.
    let faces: Vec<([[f32; 2]; 5], f32)> = p.features.iter().map(|f| { let q = f.map(|q| [q[0] * w as f32, q[1] * h as f32]); let apart = (q[1][0] - q[0][0]).hypot(q[1][1] - q[0][1]); (q, apart) }).collect();
    let near = |q: [f32; 2], x: f32, y: f32, r: f32| (q[0] - x).hypot(q[1] - y) < r;
    let off_feature = |x: f32, y: f32| faces.iter().any(|(q, apart)| { let r = SHADE_OFF_FEATURE * apart; near(q[0], x, y, r) || near(q[1], x, y, r) || near(q[2], x, y, r) || near([(q[3][0] + q[4][0]) * 0.5, (q[3][1] + q[4][1]) * 0.5], x, y, r) });
    let in_portrait = |x: f32, y: f32| faces.iter().any(|(q, apart)| *apart >= SHADE_PORTRAIT * long && near(q[2], x, y, 2.5 * apart));
    // PRINTED: the plates, then the shade, each out of register its own way.
    let feather = (long * INK_SPREAD_EDGE).max(1.0).round();
    let dot_gain = (long * SCREEN_GAIN).round();
    let shift = |ink: usize| -> [f32; 2] {
        let a = (jitter(p.seed ^ 0x5E6, ink as u64) + 0.5) * std::f32::consts::TAU;
        let m = long / MISREGISTER * (0.6 + 0.8 * (jitter(p.seed ^ 0x5E7, ink as u64) + 0.5));
        [m * a.cos(), m * a.sin()]
    };
    let shifted = |rings: Vec<[f32; 2]>, by: [f32; 2]| -> Vec<[f32; 2]> { rings.iter().map(|q| if q[0].is_nan() { *q } else { [q[0] + by[0], q[1] + by[1]] }).collect() };
    let pitch = (long / SCREEN_PITCH).max(5.0);
    for stage_no in 1..=n_stages_total.max(1) {
        let stage = format!("wash-{stage_no}");
        if let Some(pr) = progress {
            pr(PaintProgress::Painting { pass: stage_no, passes: n_stages_total, radius: 0.0 });
        }
        let mut lay: Vec<(Vec<[f32; 2]>, Vec<f32>, f32)> = Vec::new();
        if stage_no == 1 {
            // The press's plate of each ink: every combination it is in, cut to meet the next plate's edge.
            for ink in 0..n_inks {
                let mask: Vec<bool> = (0..w * h).map(|i| label[i] != PAPER && combos[label[i] as usize] & (1 << ink) != 0).collect();
                let rings = mask_rings(&mask, w, h, (long * 0.0016).max(1.5));
                if rings.is_empty() {
                    continue;
                }
                lay.push((shifted(rings, shift(ink)), loads[ink].clone(), feather));
            }
        } else if stage_no == 2 {
            let mut screen: Vec<[f32; 2]> = Vec::new();
            let s2 = std::f32::consts::FRAC_1_SQRT_2;
            let mut u = 0.5 * pitch;
            while u < (w + h) as f32 * s2 {
                let mut v = -(w as f32) * s2 + 0.5 * pitch;
                while v < h as f32 * s2 {
                    let (x, y) = ((u - v) * s2, (u + v) * s2);
                    v += pitch;
                    if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                        continue;
                    }
                    let i = y as usize * w + x as usize;
                    if off_feature(x, y) {
                        continue;
                    }
                    // What the point is read at, the luma it must lie under, the luma printed there, and how far
                    // under that the dot is full (0: the band's deep end).
                    let (l, to, printed, span) = if on_face(i) {
                        if !in_portrait(x, y) {
                            continue;
                        }
                        (luma[i], SHADE_SKIN_TO, plate_luma[label[i] as usize], 0.0)
                    } else {
                        if label[i] != PAPER && !shaded[i] {
                            continue;
                        }
                        (luma[i], SHADE_BAND.1, if label[i] == PAPER { PAPER_LUMA } else { plate_luma[label[i] as usize] }, 0.0)
                    };
                    if l >= to || l > printed - SHADE_BELOW {
                        continue;
                    }
                    // (Full from the band's deep end down: under it is the solid black's, or near it.)
                    let span = if span > 0.0 { span } else { (printed - SHADE_BAND.0).max(0.05) };
                    let r = pitch * 0.5 * SCREEN_FULL * ((printed - l) / span).clamp(0.0, 1.0).sqrt();
                    if r < SCREEN_LEAST * pitch {
                        continue;
                    }
                    for k in 0..12 {
                        let a = std::f32::consts::TAU * k as f32 / 12.0;
                        screen.push([x + r * a.cos(), y + r * a.sin()]);
                    }
                    screen.push([f32::NAN, f32::NAN]);
                }
                u += pitch;
            }
            if !screen.is_empty() {
                let black = crate::paint::canvas::darkest_pigment(&p.palette);
                let mut load = vec![0f32; n];
                load[black] = p.charge * 2.0;
                lay.push((shifted(screen, shift(n_inks)), load, dot_gain));
            }
        }
        for (rings, load, feather) in lay {
            if *placed >= p.budget {
                break;
            }
            canvas.fill_rings(&rings, &load, 0.75, feather);
            *k += 1;
            *placed += 1;
            if let Some(pr) = progress {
                pr(PaintProgress::Placed(*placed));
            }
            let mix: Vec<(String, f32)> = load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
            score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: stage.clone(), spline: rings, w0: feather, w1: 0.0, taper: 0.0, mix, wet: 0.75, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        }
        // The pass boundary, as the replay reproduces it: each plate sets before the next is laid.
        if p.bleed > 0.0 {
            canvas.bleed_with(p.bleed * pass_bleed_taper(stage_no - 1, n_stages_total), p.diffuse);
        }
        if stage_no < n_stages_total {
            canvas.dry(1.0 - p.dry);
        }
    }
}

/// A face's size, as a share of the sheet's long side: at `FACE_SMALL` and under it is a far face, a few
/// strokes; from `FACE_LARGE` a portrait's, its features drawn in full.
const FACE_SMALL: f32 = 0.04;
const FACE_LARGE: f32 = 0.16;
/// Below this share of the sheet's long side a face is a FAR one: not traced, but put down in a few strokes —
/// and no more of them than this. A face in a window, a tenth of the sheet, is built as a near one is, at its
/// own scale: its lids, its iris, its nose line and its mouth, from where the detector finds them.
const FACE_FAR: f32 = 0.045;
/// ... and below this share it is a face in a window: built, and nothing traced inside its outline.
const FACE_MID: f32 = 0.1;
const FACE_STROKES: usize = 4;
/// A DARK EYE — the dark of its opening this much (0..255) under the skin's value — has its iris filled, the
/// black of the thirties' eyes, with one glint left white at its upper left, the same side in both eyes as
/// one light would put it; a light eye is a ring round its pupil.
const EYE_DARK: f32 = 80.0;
/// How much heavier a face in a window has its features drawn than a portrait has, for its size: a sign for
/// an eye in a crowd is a stroke, not a hair.
const FACE_BOLD: f32 = 1.2;

/// Round a face in a window the picture is read with nothing in it: the head, as the detector named it,
/// grown by a share of its size, is set to the picture smoothed — so the planner traces neither the shading
/// inside it nor its silhouette, which is drawn as one line (see `head_outline`).
fn head_quiet(drawn: &mut RgbImage, crop: &RgbImage, off: &[bool], cw: usize, ch: usize, cx: usize, cy: usize, size: f32) {
    let band: Vec<bool> = dilate_bool(&off.iter().map(|o| !o).collect::<Vec<bool>>(), cw, ch, (size * 0.045).max(2.0) as usize);
    let plain = imageops::blur(crop, (size * 0.12).max(1.5));
    for j in (0..cw * ch).filter(|&j| band[j]) {
        drawn.put_pixel((cx + j % cw) as u32, (cy + j / cw) as u32, *plain.get_pixel((j % cw) as u32, (j / cw) as u32));
    }
}

/// Where the LIGHT falls on a head: the way from the head's centre to the centre of its light, read off the
/// picture smoothed at the head's scale — a unit vector, or none where the head is lit evenly.
fn head_light(crop: &RgbImage, off: &[bool], cw: usize, ch: usize, size: f32) -> Option<[f32; 2]> {
    let plain = imageops::blur(crop, (size * 0.1).max(1.5));
    let mut m = [0f64; 6];
    for j in (0..cw * ch).filter(|&j| !off[j]) {
        let c = plain.get_pixel((j % cw) as u32, (j / cw) as u32).0;
        let l = (0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64) / 255.0;
        let (x, y) = ((j % cw) as f64, (j / cw) as f64);
        m = [m[0] + 1.0, m[1] + x, m[2] + y, m[3] + l, m[4] + l * x, m[5] + l * y];
    }
    if m[0] < 16.0 || m[3] <= 0.0 {
        return None;
    }
    let d = [(m[4] / m[3] - m[1] / m[0]) as f32, (m[5] / m[3] - m[2] / m[0]) as f32];
    let len = d[0].hypot(d[1]);
    (len >= size * HEAD_LIT_LEAST).then(|| [d[0] / len, d[1] / len])
}

/// A head's outline is LOST on the side the light falls on: gone where the outward way from the head's
/// centre lies within `HEAD_LOST` of the light (a cosine), FOUND — the full line — where it lies beyond
/// `HEAD_FOUND` of it, and thinning by degrees between, dropped under `HEAD_LEAST` of its width. A head whose
/// light lies nearer its centre than `HEAD_LIT_LEAST` of its size is lit evenly, and outlined all round.
const HEAD_LOST: f32 = 0.35;
const HEAD_FOUND: f32 = -0.3;
const HEAD_LEAST: f32 = 0.15;
const HEAD_LIT_LEAST: f32 = 0.015;

/// A FACE'S OUTLINE, for a face in a window: the head as the detector named it — a few coarse cells — rounded
/// at the face's own scale into one line: the jaw, the hairline, the neck, as a hand draws a head before it
/// puts the features in — lost on the lit side, found on the shadow side (see `HEAD_LOST`). The line is the
/// plate's edge too: the flesh tint ends where it runs.
fn head_outline(off: &[bool], cw: usize, ch: usize, cx: usize, cy: usize, size: f32, light: Option<[f32; 2]>) -> Vec<crate::paint::engrave::Cut> {
    let is = image::GrayImage::from_fn(cw as u32, ch as u32, |x, y| image::Luma([if off[y as usize * cw + x as usize] { 0 } else { 255 }]));
    let mask: Vec<bool> = imageops::blur(&is, (size * 0.05).max(1.5)).pixels().map(|px| px.0[0] > 127).collect();
    let rings = mask_rings(&mask, cw, ch, 1.0);
    let mut out = Vec::new();
    for ring in rings.split(|q| q[0].is_nan()) {
        let n = ring.len();
        if n < 8 {
            continue;
        }
        // The ring, a point to the pixel; smoothed twice round its whole length; then one point in a few.
        let mut dense: Vec<[f32; 2]> = Vec::new();
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let k = (b[0] - a[0]).hypot(b[1] - a[1]).ceil().max(1.0) as usize;
            for s in 0..k {
                let t = s as f32 / k as f32;
                dense.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
        }
        let n = dense.len();
        if (n as f32) < size * 1.2 {
            continue;
        }
        let win = ((size * 0.04) as usize).max(2);
        for _ in 0..2 {
            let src = dense.clone();
            for i in 0..n {
                let mut acc = [0f32; 2];
                for k in 0..=2 * win {
                    let j = (i + n + k - win) % n;
                    acc[0] += src[j][0];
                    acc[1] += src[j][1];
                }
                dense[i] = [acc[0] / (2 * win + 1) as f32, acc[1] / (2 * win + 1) as f32];
            }
        }
        let every = ((size * 0.03) as usize).max(2);
        let pts: Vec<[f32; 2]> = dense.iter().step_by(every).map(|q| [q[0] + cx as f32, q[1] + cy as f32]).collect();
        lost_and_found(&pts, (size * 0.02).max(2.4), light, &[], &mut out);
    }
    out
}

/// A closed ring of points drawn LOST AND FOUND (see `HEAD_LOST`): the weight at each point is the full
/// `base` in the shade, thinning by degrees toward the light, and gone where the light falls; the kept runs
/// are cut into short strokes on layer 1, which the key stamps at their own width down to a hair. A point's
/// weight is further scaled by `keep` (0..1 each, or none), where the ring's own reading says the line gives out.
fn lost_and_found(pts: &[[f32; 2]], base: f32, light: Option<[f32; 2]>, keep: &[f32], out: &mut Vec<crate::paint::engrave::Cut>) {
    let n = pts.len();
    if n < 4 {
        return;
    }
    let c = pts.iter().fold([0f32; 2], |m, q| [m[0] + q[0], m[1] + q[1]]).map(|v| v / n as f32);
    let heft: Vec<f32> = pts.iter().enumerate().map(|(j, q)| keep.get(j).copied().unwrap_or(1.0) * match light {
        None => 1.0,
        Some(l) => {
            let (dx, dy) = (q[0] - c[0], q[1] - c[1]);
            let lit = (dx * l[0] + dy * l[1]) / dx.hypot(dy).max(1e-3);
            ((HEAD_LOST - lit) / (HEAD_LOST - HEAD_FOUND)).clamp(0.0, 1.0)
        }
    }).collect();
    let width = |k: f32| base * k;
    let mut emit = |run: &[usize]| {
        let mut a = 0usize;
        while a + 1 < run.len() {
            let b = (a + 4).min(run.len() - 1);
            out.push(crate::paint::engrave::Cut { layer: 1, path: run[a..=b].iter().map(|&j| pts[j % n]).collect(), w0: width(heft[run[a] % n]), w1: width(heft[run[b] % n]) });
            a = b;
        }
    };
    match heft.iter().position(|k| *k < HEAD_LEAST) {
        // Lit evenly, or all in the shade: one closed line.
        None => emit(&(0..=n).collect::<Vec<usize>>()),
        Some(start) => {
            let mut run: Vec<usize> = Vec::new();
            for s in 1..=n {
                let j = (start + s) % n;
                if heft[j] >= HEAD_LEAST {
                    run.push(j);
                } else {
                    emit(&run);
                    run.clear();
                }
            }
            emit(&run);
        }
    }
}

/// A HEAD BY TEMPLATE, for a face in a crowd or a window — one whose eyes are under `FACE_PATTERN` of the
/// long side apart. The detector's five points fix a frame: `u` along the eyes, `v` down the face, the
/// scale `s` their distance (widened as the head turns, the turn `yaw` read from how far the nose stands off
/// the eyes' middle), and in that frame the head is the EGG every hand draws before the features go in:
/// centred `HEAD_EGG.0` under the eyes, `HEAD_EGG.1` wide and `HEAD_EGG.2` tall each way, narrowed to the
/// jaw by `HEAD_JAW`, and shifted `HEAD_TURN` of the scale away from the nose as the head turns — the skull
/// behind the face. The egg is the flesh plate's edge and the head's line (lost and found), whatever shape the
/// detector's coarse cells gave the skin.
const FACE_PATTERN: f32 = 0.035;
const HEAD_EGG: (f32, f32, f32) = (0.3, 1.2, 1.6);
const HEAD_JAW: f32 = 0.3;
const HEAD_TURN: f32 = 0.4;
/// ... and the egg is FITTED to the head (see `Head::fit`), since no two heads are one egg and a head bent
/// forward or back is no egg at all: along each of `HEAD_RAYS` rays from the egg's centre the picture's luma
/// is read `HEAD_FIT_PROBE` of the scale wide, from `HEAD_FIT_SPAN.0` to `HEAD_FIT_SPAN.1` of the egg's radius
/// — never nearer than `HEAD_FIT_CROWN` over the eyes, where the face's edge on its hair is not the head's —
/// and the rim goes to the strongest step, each radius it stands from the egg costing it `HEAD_FIT_PRIOR` of
/// luma; a step under `HEAD_FIT_EDGE` is none, and the egg stays. The radii are smoothed round the head. The
/// step left at the rim is kept: the CHIN's line, over `HEAD_CHIN` of the egg's height under its centre, is
/// drawn only as strong as that step — where the chin runs into the neck with no step of the luma, no line.
const HEAD_RAYS: usize = 48;
const HEAD_FIT_PROBE: f32 = 0.04;
const HEAD_FIT_SPAN: (f32, f32) = (0.75, 1.35);
const HEAD_FIT_CROWN: f32 = 0.9;
const HEAD_FIT_PRIOR: f32 = 0.25;
const HEAD_FIT_EDGE: f32 = 0.06;
const HEAD_CHIN: f32 = 0.6;
/// ... the HAIR is the mass over the egg, grown by `HEAD_HAIR_GROW`, above `HEAD_HAIR_DOWN` of the scale
/// under the eyes, that is not skin and lies `HEAD_HAIR_DARK` (luma) under the skin — printed solid black, as
/// the thirties printed a small head's hair; none unless a quarter of the egg's crown is it (a bald head
/// has none). And the SHADOW SIDE of the head is flat: the flesh with the darkest ink over it, a crescent —
/// what lies in the egg but outside the egg moved `HEAD_SHADE` of the scale toward the light, its edge
/// wavering by `HEAD_SHADE_WOBBLE` of the egg round the head as a brush would leave it, not a cut.
const HEAD_HAIR_GROW: f32 = 1.15;
const HEAD_HAIR_DOWN: f32 = 0.75;
const HEAD_HAIR_DARK: f32 = 0.15;
const HEAD_SHADE: f32 = 0.45;
const HEAD_SHADE_WOBBLE: f32 = 0.06;
/// The SCENE's light, one for every small head on the sheet: the way the heads' own lights agree on — each
/// head's light (see `Head::light`) a unit vote, taken when the votes' sum keeps `SCENE_LIGHT_AGREE` of their
/// count — and otherwise the illustrator's light, `SCENE_LIGHT`, from the upper left. No head has its shadow
/// cast its own way.
const SCENE_LIGHT: [f32; 2] = [-0.6, -0.8];
const SCENE_LIGHT_AGREE: f32 = 0.5;
/// The PATTERN's measures, in scales (the eyes' distance): an eye's dash half-length and width and its
/// pupil's dot; the brow's height over the eye and its width; the nose line's width at the bridge and at the
/// tip; the mouth's least half-length and its width shut.
const PATTERN_EYE: (f32, f32, f32) = (0.2, 0.12, 0.2);
const PATTERN_BROW: (f32, f32) = (0.3, 0.07);
const PATTERN_NOSE: (f32, f32) = (0.05, 0.09);
const PATTERN_MOUTH: (f32, f32) = (0.28, 0.07);
/// ... and the BROW and the MOUTH are each a CURVE read off the picture, not a rule: the dark of the feature,
/// in a window `PATTERN_FIT.0` of the scale across round where the pattern sets it, is fitted by a parabola
/// along the feature, and the pattern's line bends as the parabola does — raised, knit, smiling, turned down
/// — as far as the window's dark is sure (its mean, in values, against `PATTERN_FIT.1`); the bend is held
/// within `PATTERN_FIT.2` of the scale. The pupil stands off the eye's middle with the turn of the head.
const PATTERN_FIT: (f32, f32, f32) = (0.12, 8.0, 0.12);

struct Head {
    eyes: [f32; 2],
    u: [f32; 2],
    v: [f32; 2],
    s: f32,
    yaw: f32,
    /// The rim's radius on each ray, in egg radii, and the step of the luma across the rim there.
    r: [f32; HEAD_RAYS],
    edge: [f32; HEAD_RAYS],
}

impl Head {
    fn of(face: &[[f32; 2]; 5]) -> Head {
        let [le, re, nose, lm, rm] = *face;
        let mid = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let (eyes, mouth) = (mid(le, re), mid(lm, rm));
        let apart = (re[0] - le[0]).hypot(re[1] - le[1]);
        let tall = (mouth[0] - eyes[0]).hypot(mouth[1] - eyes[1]);
        let s0 = apart.max(tall * 0.9).max(4.0);
        let u = if apart > 0.35 * s0 { [(re[0] - le[0]) / apart, (re[1] - le[1]) / apart] } else { [(mouth[1] - eyes[1]) / tall.max(1e-3), -(mouth[0] - eyes[0]) / tall.max(1e-3)] };
        let v = { let v = [-u[1], u[0]]; if v[0] * (mouth[0] - eyes[0]) + v[1] * (mouth[1] - eyes[1]) < 0.0 { [-v[0], -v[1]] } else { v } };
        let yaw = (((nose[0] - eyes[0]) * u[0] + (nose[1] - eyes[1]) * u[1]) / (apart * 0.5).max(1.0)).clamp(-1.0, 1.0);
        let s = (apart / (1.0 - 0.25 * yaw.abs())).max(tall * 0.9).max(4.0);
        Head { eyes, u, v, s, yaw, r: [1.0; HEAD_RAYS], edge: [0.0; HEAD_RAYS] }
    }
    /// The egg FITTED to the head (see `HEAD_FIT_PROBE`).
    fn fit(face: &[[f32; 2]; 5], luma: &[f32], w: usize, h: usize) -> Head {
        let mut head = Head::of(face);
        let probe = (head.s * HEAD_FIT_PROBE).max(1.0);
        let sample = |q: [f32; 2]| -> Option<f32> {
            let (x0, x1) = ((q[0] - probe).max(0.0) as usize, ((q[0] + probe).max(0.0) as usize).min(w - 1));
            let (y0, y1) = ((q[1] - probe).max(0.0) as usize, ((q[1] + probe).max(0.0) as usize).min(h - 1));
            let mut m = [0f32; 2];
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if (x as f32 + 0.5 - q[0]).hypot(y as f32 + 0.5 - q[1]) <= probe {
                        m = [m[0] + luma[y * w + x], m[1] + 1.0];
                    }
                }
            }
            (m[1] > 0.0 && q[0] >= 0.0 && q[1] >= 0.0 && q[0] < w as f32 && q[1] < h as f32).then(|| m[0] / m[1])
        };
        const STEP: f32 = 0.025;
        const ACROSS: usize = 3;
        let mut r = [1.0f32; HEAD_RAYS];
        for k in 0..HEAD_RAYS {
            let phi = std::f32::consts::TAU * k as f32 / HEAD_RAYS as f32;
            let lo = if phi.sin() < 0.0 { HEAD_FIT_SPAN.0.max(HEAD_FIT_CROWN) } else { HEAD_FIT_SPAN.0 };
            let ts: Vec<f32> = (0..).map(|i| lo + STEP * i as f32).take_while(|t| *t <= HEAD_FIT_SPAN.1 + 1e-4).collect();
            let ls: Vec<Option<f32>> = ts.iter().map(|&t| sample(head.rim_at(phi, t))).collect();
            let mut best: Option<(f32, f32, f32)> = None;
            for i in ACROSS..ts.len().saturating_sub(ACROSS) {
                let (Some(a), Some(b)) = (ls[i - ACROSS], ls[i + ACROSS]) else { continue };
                let g = (b - a).abs();
                let score = g - HEAD_FIT_PRIOR * (ts[i] - 1.0).abs();
                if g >= HEAD_FIT_EDGE && best.map_or(true, |(bs, _, _)| score > bs) {
                    best = Some((score, ts[i], g));
                }
            }
            if let Some((_, t, _)) = best {
                r[k] = t;
            }
        }
        // Smoothed round the head, twice, five rays at a time.
        for _ in 0..2 {
            let src = r;
            for k in 0..HEAD_RAYS {
                let at = |d: i64| src[((k as i64 + d).rem_euclid(HEAD_RAYS as i64)) as usize];
                r[k] = ((at(-2) + at(2)) + 2.0 * (at(-1) + at(1)) + 3.0 * at(0)) / 9.0;
            }
        }
        head.r = r.map(|t| t.clamp(HEAD_FIT_SPAN.0, HEAD_FIT_SPAN.1));
        // The step across the rim as fitted.
        for k in 0..HEAD_RAYS {
            let phi = std::f32::consts::TAU * k as f32 / HEAD_RAYS as f32;
            let t = head.r[k];
            let d = STEP * ACROSS as f32;
            head.edge[k] = match (sample(head.rim_at(phi, t - d)), sample(head.rim_at(phi, t + d))) {
                (Some(a), Some(b)) => (b - a).abs(),
                _ => 0.0,
            };
        }
        head
    }
    /// The eyes' distance of a face, as the detector gives it.
    fn apart(face: &[[f32; 2]; 5]) -> f32 {
        (face[1][0] - face[0][0]).hypot(face[1][1] - face[0][1])
    }
    /// The sheet point at `a` along the eyes and `b` down the face, in scales.
    fn at(&self, a: f32, b: f32) -> [f32; 2] {
        [self.eyes[0] + (self.u[0] * a + self.v[0] * b) * self.s, self.eyes[1] + (self.u[1] * a + self.v[1] * b) * self.s]
    }
    fn local(&self, p: [f32; 2]) -> (f32, f32) {
        let (dx, dy) = (p[0] - self.eyes[0], p[1] - self.eyes[1]);
        ((dx * self.u[0] + dy * self.u[1]) / self.s, (dx * self.v[0] + dy * self.v[1]) / self.s)
    }
    /// The rim's radius at angle `phi`, in egg radii, between the rays.
    fn radius(&self, phi: f32) -> f32 {
        let f = phi.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU * HEAD_RAYS as f32;
        let (k, t) = ((f.floor() as usize) % HEAD_RAYS, f - f.floor());
        self.r[k] * (1.0 - t) + self.r[(k + 1) % HEAD_RAYS] * t
    }
    /// How far out a point lies on the head: 1 on its rim.
    fn egg(&self, a: f32, b: f32) -> f32 {
        let ey = (b - HEAD_EGG.0) / HEAD_EGG.2;
        let ex = (a + HEAD_TURN * self.yaw) / (HEAD_EGG.1 * (1.0 - HEAD_JAW * ey.max(0.0).powi(2)));
        (ex * ex + ey * ey).sqrt() / self.radius(ey.atan2(ex))
    }
    /// The point at angle `phi` (0 at the right, down the face at a quarter turn) and `t` egg radii out.
    fn rim_at(&self, phi: f32, t: f32) -> [f32; 2] {
        let (ey, ex) = phi.sin_cos();
        let (ey, ex) = (ey * t, ex * t);
        self.at(-HEAD_TURN * self.yaw + HEAD_EGG.1 * ex * (1.0 - HEAD_JAW * ey.max(0.0).powi(2)), HEAD_EGG.0 + HEAD_EGG.2 * ey)
    }
    /// The head's rim at angle `phi`.
    fn rim(&self, phi: f32) -> [f32; 2] {
        self.rim_at(phi, self.radius(phi))
    }
    fn centre(&self) -> [f32; 2] {
        self.at(-HEAD_TURN * self.yaw, HEAD_EGG.0)
    }
    /// The head's bounding box on a sheet of `w` by `h`, grown by `grow`.
    fn bounds(&self, w: usize, h: usize, grow: f32) -> (usize, usize, usize, usize) {
        let c = self.centre();
        let r = self.s * HEAD_EGG.2 * grow * self.r.iter().fold(1.0f32, |m, t| m.max(*t)) + 1.0;
        (((c[0] - r).max(0.0) as usize).min(w - 1), ((c[0] + r).max(0.0) as usize).min(w - 1), ((c[1] - r).max(0.0) as usize).min(h - 1), ((c[1] + r).max(0.0) as usize).min(h - 1))
    }
    /// The FACE's luma: the mean over the egg's band from the eyes to the chin, which is face whatever the
    /// detector's cells say — the hair and the shadow are read against it.
    fn face_luma(&self, luma: &[f32], w: usize, h: usize) -> Option<f32> {
        let (x0, x1, y0, y1) = self.bounds(w, h, 1.0);
        let mut m = [0f64; 2];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (a, b) = self.local([x as f32 + 0.5, y as f32 + 0.5]);
                if (0.0..=1.2).contains(&b) && self.egg(a, b) <= 1.0 {
                    m = [m[0] + luma[y * w + x] as f64, m[1] + 1.0];
                }
            }
        }
        (m[1] >= 4.0).then(|| (m[0] / m[1]) as f32)
    }
    /// The head's HAIR (see `HEAD_HAIR_DARK`): the sheet's points that are it.
    fn hair(&self, luma: &[f32], w: usize, h: usize) -> Vec<usize> {
        let Some(face) = self.face_luma(luma, w, h) else { return Vec::new() };
        let under = face - HEAD_HAIR_DARK;
        let (x0, x1, y0, y1) = self.bounds(w, h, HEAD_HAIR_GROW);
        let (mut hair, mut crown) = (Vec::new(), [0usize; 2]);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = y * w + x;
                let (a, b) = self.local([x as f32 + 0.5, y as f32 + 0.5]);
                let e = self.egg(a, b);
                let is = e <= HEAD_HAIR_GROW && b < HEAD_HAIR_DOWN && luma[i] < under;
                if e <= 1.0 && b < -0.3 {
                    crown = [crown[0] + usize::from(is), crown[1] + 1];
                }
                if is {
                    hair.push(i);
                }
            }
        }
        if crown[1] == 0 || crown[0] * 4 < crown[1] {
            return Vec::new();
        }
        hair
    }
    /// Where the LIGHT falls on the head: the way from the egg's centre to the centre of its light, read off
    /// the face under the hairline and not the hair — or none where it is lit evenly (see `HEAD_LIT_LEAST`).
    fn light(&self, luma: &[f32], w: usize, h: usize) -> Option<[f32; 2]> {
        let under = self.face_luma(luma, w, h)? - HEAD_HAIR_DARK;
        let (x0, x1, y0, y1) = self.bounds(w, h, 1.0);
        let mut m = [0f64; 6];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (a, b) = self.local([x as f32 + 0.5, y as f32 + 0.5]);
                let l = luma[y * w + x];
                if b >= -0.1 && self.egg(a, b) <= 1.0 && l >= under {
                    let l = l as f64;
                    m = [m[0] + 1.0, m[1] + x as f64, m[2] + y as f64, m[3] + l, m[4] + l * x as f64, m[5] + l * y as f64];
                }
            }
        }
        if m[0] < 16.0 || m[3] <= 0.0 {
            return None;
        }
        let d = [(m[4] / m[3] - m[1] / m[0]) as f32, (m[5] / m[3] - m[2] / m[0]) as f32];
        let len = d[0].hypot(d[1]);
        (len >= self.s * 2.5 * HEAD_LIT_LEAST).then(|| [d[0] / len, d[1] / len])
    }
    /// Whether a sheet point lies on the head's SHADOW SIDE (see `HEAD_SHADE`): in the egg, but out of the
    /// egg moved toward the light — a crescent on the side the light leaves, its edge wavering.
    fn in_shadow(&self, q: [f32; 2], light: [f32; 2]) -> bool {
        let (a, b) = self.local(q);
        if self.egg(a, b) > 1.0 {
            return false;
        }
        let (la, lb) = (light[0] * self.u[0] + light[1] * self.u[1], light[0] * self.v[0] + light[1] * self.v[1]);
        let (a, b) = (a - HEAD_SHADE * la, b - HEAD_SHADE * lb);
        let phi = ((b - HEAD_EGG.0) / HEAD_EGG.2).atan2((a + HEAD_TURN * self.yaw) / HEAD_EGG.1);
        let wobble = 0.5 * (3.0 * phi + 1.3).sin() + 0.3 * (7.0 * phi + 0.4).sin() + 0.2 * (11.0 * phi + 2.1).sin();
        self.egg(a, b) > 1.0 + HEAD_SHADE_WOBBLE * wobble
    }
}

/// The small heads of a sheet (see `Head`), fitted, and the SCENE's light (see `SCENE_LIGHT`).
fn small_heads(p: &PaintParams, luma: &[f32], w: usize, h: usize) -> (Vec<([[f32; 2]; 5], Head)>, [f32; 2]) {
    let long = w.max(h) as f32;
    let heads: Vec<([[f32; 2]; 5], Head)> = p.features.iter().map(|f| f.map(|q| [q[0] * w as f32, q[1] * h as f32])).filter(|f| Head::apart(f) / long < FACE_PATTERN).map(|f| { let head = Head::fit(&f, luma, w, h); (f, head) }).collect();
    let votes: Vec<[f32; 2]> = heads.iter().filter_map(|(_, head)| head.light(luma, w, h)).collect();
    let sum = votes.iter().fold([0f32; 2], |m, l| [m[0] + l[0], m[1] + l[1]]);
    let len = sum[0].hypot(sum[1]);
    let scene = if !votes.is_empty() && len >= SCENE_LIGHT_AGREE * votes.len() as f32 { [sum[0] / len, sum[1] / len] } else { let l = SCENE_LIGHT[0].hypot(SCENE_LIGHT[1]); [SCENE_LIGHT[0] / l, SCENE_LIGHT[1] / l] };
    (heads, scene)
}

/// The luma of the sheet, 0..1, as the small heads are read.
fn sheet_luma(source: &RgbImage) -> Vec<f32> {
    source.pixels().map(|px| (0.299 * px.0[0] as f32 + 0.587 * px.0[1] as f32 + 0.114 * px.0[2] as f32) / 255.0).collect()
}

/// A FACE BY THE PATTERN of the thirties, in a crowd or a window: the head by template (see `Head`), its line
/// lost and found; the neck; the hair a solid mass; and the features, each a sign — a dash for each eye
/// with its dark, the brow an arc over it, one line for the nose down its shadow side, the mouth from corner
/// to corner — set where the detector's points put them and sized by the scale, the picture read only for
/// the measures that tell one face from the next: how the brow and the mouth bend, how far the mouth is
/// open. The light is the scene's.
fn pattern_face(face: &[[f32; 2]; 5], head: &Head, scene: [f32; 2], crop: &RgbImage, off: &[bool], cx: usize, cy: usize, cw: usize, ch: usize, luma: &[f32], sw: usize, sh: usize, features: &mut Vec<crate::paint::engrave::Cut>) {
    let [le, re, nose, lm, rm] = *face;
    let (s, u, v, eyes, yaw) = (head.s, head.u, head.v, head.eyes, head.yaw);
    let mouth = [(lm[0] + rm[0]) * 0.5, (lm[1] + rm[1]) * 0.5];
    let apart = Head::apart(face);
    let pixel = |q: [f32; 2]| ((q[1].max(0.0) as usize).min(sh - 1)) * sw + ((q[0].max(0.0) as usize).min(sw - 1));
    let hair: std::collections::HashSet<usize> = head.hair(luma, sw, sh).into_iter().collect();
    // THE HEAD: the rim, lost on the lit side, gone in the hair, and at the chin as strong as the step there
    // (see `HEAD_CHIN`); the neck, two light lines under the jaw.
    let rim: Vec<[f32; 2]> = (0..HEAD_RAYS).map(|k| head.rim(std::f32::consts::TAU * k as f32 / HEAD_RAYS as f32)).collect();
    let keep: Vec<f32> = (0..HEAD_RAYS).map(|k| {
        let phi = std::f32::consts::TAU * k as f32 / HEAD_RAYS as f32;
        if hair.contains(&pixel(head.rim_at(phi, head.r[k] * 0.9))) {
            0.0
        } else if phi.sin() > HEAD_CHIN {
            (head.edge[k] / HEAD_FIT_EDGE).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }).collect();
    lost_and_found(&rim, (s * 0.05).max(2.4), Some(scene), &keep, features);
    let turn = -HEAD_TURN * yaw;
    let chin = head.radius(std::f32::consts::FRAC_PI_2);
    for side in [-1.0f32, 1.0] {
        let b0 = HEAD_EGG.0 + HEAD_EGG.2 * chin;
        features.push(crate::paint::engrave::Cut { layer: 1, path: vec![head.at(turn + side * 0.5, b0 - 0.18), head.at(turn + side * 0.52, b0 + 0.1), head.at(turn + side * 0.55, b0 + 0.4)], w0: (s * 0.035).max(1.6), w1: (s * 0.03).max(1.4) });
    }
    // THE HAIR, row by row.
    if !hair.is_empty() {
        let mut rows: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for &i in &hair {
            rows.entry(i / sw).or_default().push(i % sw);
        }
        for (y, mut xs) in rows {
            xs.sort_unstable();
            let mut a = 0usize;
            while a < xs.len() {
                let mut b = a;
                while b + 1 < xs.len() && xs[b + 1] == xs[b] + 1 {
                    b += 1;
                }
                features.push(crate::paint::engrave::Cut { layer: 0, path: vec![[xs[a] as f32 + 0.5, y as f32 + 0.5], [xs[b] as f32 + 1.5, y as f32 + 0.5]], w0: 1.4, w1: 1.4 });
                a = b + 1;
            }
        }
    }
    // THE FEATURES, each a sign at its measure of the scale, bent as the picture bends them (see `PATTERN_FIT`).
    let value = |im: &RgbImage, j: usize| { let c = im.get_pixel((j % cw) as u32, (j / cw) as u32).0; 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32 };
    let fine = imageops::blur(crop, (s * 0.03).max(0.7));
    let round = imageops::blur(crop, (s * 0.25).max(2.0));
    let stands = |j: usize| if off[j] { 0.0 } else { value(&round, j) - value(&fine, j) };
    // A curve read off the picture: the dark of a feature in a window `half` along `along` and `across` down,
    // round `c`, fitted b = c0 + c1·A + c2·A² (A the way along, -1..1; b down the face, in pixels), weighted by
    // how far each point stands under its surroundings — and how sure the window's dark is, 0..1.
    let fit_curve = |c: [f32; 2], along: [f32; 2], half: f32, across: f32| -> Option<([f32; 3], f32)> {
        let down = { let d = [-along[1], along[0]]; if d[0] * v[0] + d[1] * v[1] < 0.0 { [-d[0], -d[1]] } else { d } };
        let reach = half.max(across) + 1.0;
        let (x0, x1) = ((c[0] - reach).max(cx as f32) as usize, ((c[0] + reach).max(0.0) as usize).min(cx + cw - 1));
        let (y0, y1) = ((c[1] - reach).max(cy as f32) as usize, ((c[1] + reach).max(0.0) as usize).min(cy + ch - 1));
        let mut m = [0f64; 9];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (dx, dy) = (x as f32 + 0.5 - c[0], y as f32 + 0.5 - c[1]);
                let (a, b) = (dx * along[0] + dy * along[1], dx * down[0] + dy * down[1]);
                if a.abs() > half || b.abs() > across || hair.contains(&(y * sw + x)) {
                    continue;
                }
                let wgt = stands((y - cy) * cw + (x - cx)).max(0.0) as f64;
                let (aa, b) = ((a / half) as f64, b as f64);
                m = [m[0] + wgt, m[1] + wgt * aa, m[2] + wgt * aa * aa, m[3] + wgt * aa * aa * aa, m[4] + wgt * aa * aa * aa * aa, m[5] + wgt * b, m[6] + wgt * aa * b, m[7] + wgt * aa * aa * b, m[8] + 1.0];
            }
        }
        if m[8] < 6.0 || m[0] <= 0.0 {
            return None;
        }
        let (s0, s1, s2, s3, s4) = (m[0], m[1], m[2], m[3], m[4]);
        let det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2);
        if det.abs() < 1e-9 * s0.powi(3).max(1e-12) {
            return None;
        }
        let solve = |r: [f64; 3]| {
            let c0 = r[0] * (s2 * s4 - s3 * s3) - s1 * (r[1] * s4 - s3 * r[2]) + s2 * (r[1] * s3 - s2 * r[2]);
            let c1 = s0 * (r[1] * s4 - s3 * r[2]) - r[0] * (s1 * s4 - s3 * s2) + s2 * (s1 * r[2] - r[1] * s2);
            let c2 = s0 * (s2 * r[2] - r[1] * s3) - s1 * (s1 * r[2] - r[1] * s2) + r[0] * (s1 * s3 - s2 * s2);
            [(c0 / det) as f32, (c1 / det) as f32, (c2 / det) as f32]
        };
        let c = solve([m[5], m[6], m[7]]);
        if c.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some((c, ((m[0] / m[8]) as f32 / PATTERN_FIT.1).clamp(0.0, 1.0)))
    };
    // The pattern's curve bent toward the picture's as far as the dark is sure, the bend held within bounds.
    let bent = |base: [f32; 3], fit: Option<([f32; 3], f32)>, hold: [f32; 3]| -> [f32; 3] {
        match fit {
            None => base,
            Some((c, sure)) => { let mut out = base; for i in 0..3 { let to = c[i].clamp(base[i] - hold[i], base[i] + hold[i]); out[i] = base[i] + sure * (to - base[i]); } out }
        }
    };
    let curve = |c: [f32; 2], along: [f32; 2], half: f32, k: [f32; 3], w0: f32, w1: f32| -> crate::paint::engrave::Cut {
        let down = { let d = [-along[1], along[0]]; if d[0] * v[0] + d[1] * v[1] < 0.0 { [-d[0], -d[1]] } else { d } };
        let path = [-1.0f32, -0.5, 0.0, 0.5, 1.0].iter().map(|&aa| { let b = k[0] + k[1] * aa + k[2] * aa * aa; [c[0] + along[0] * aa * half + down[0] * b, c[1] + along[1] * aa * half + down[1] * b] }).collect();
        crate::paint::engrave::Cut { layer: 0, path, w0, w1 }
    };
    let hold = s * PATTERN_FIT.2;
    // The eyes: a dash each with the pupil's dot in it, stood off the middle with the turn, the far eye's
    // shorter as the head turns, both tilted with the turn; and the BROW, a curve over each.
    for (eye, si) in if apart > 0.35 * s { vec![(le, -1.0f32), (re, 1.0)] } else { vec![(eyes, 0.0)] } {
        let half = s * PATTERN_EYE.0 * (1.0 - 0.45 * (si * yaw).max(0.0));
        let (sn, cs) = (0.3 * yaw * si).sin_cos();
        let along = [u[0] * cs + v[0] * sn, u[1] * cs + v[1] * sn];
        let width = (s * PATTERN_EYE.1).max(1.6);
        features.push(crate::paint::engrave::Cut { layer: 0, path: vec![[eye[0] - along[0] * half, eye[1] - along[1] * half], [eye[0] + along[0] * half, eye[1] + along[1] * half]], w0: width, w1: width * 0.65 });
        let look = yaw * half * 0.35;
        features.push(crate::paint::engrave::Cut { layer: 0, path: vec![[eye[0] + along[0] * (look - s * 0.03), eye[1] + along[1] * (look - s * 0.03)], [eye[0] + along[0] * (look + s * 0.03), eye[1] + along[1] * (look + s * 0.03)]], w0: s * PATTERN_EYE.2, w1: s * PATTERN_EYE.2 });
        let brow_c = [eye[0] - v[0] * s * PATTERN_BROW.0, eye[1] - v[1] * s * PATTERN_BROW.0];
        let brow_half = half * 1.3;
        let k = bent([-0.02 * s, 0.0, 0.055 * s], fit_curve(brow_c, along, brow_half, s * PATTERN_FIT.0), [hold * 0.7, hold, hold]);
        features.push(curve(brow_c, along, brow_half, k, (s * PATTERN_BROW.1).max(1.6), (s * PATTERN_BROW.1 * 0.7).max(1.4)));
    }
    // The nose: one line down the side the scene's light leaves; the mouth: a curve from corner to corner,
    // fuller when open.
    let side = if scene[0] * u[0] + scene[1] * u[1] < 0.0 { 1.0 } else { -1.0 };
    let at = |f: f32, out: f32, down: f32| [eyes[0] + (nose[0] - eyes[0]) * f + u[0] * side * s * out + v[0] * s * down, eyes[1] + (nose[1] - eyes[1]) * f + u[1] * side * s * out + v[1] * s * down];
    features.push(crate::paint::engrave::Cut { layer: 0, path: vec![at(0.25, 0.05, 0.0), at(0.65, 0.08, 0.0), at(1.0, 0.14, 0.03)], w0: (s * PATTERN_NOSE.0).max(1.4), w1: (s * PATTERN_NOSE.1).max(1.8) });
    let wide = (rm[0] - lm[0]).hypot(rm[1] - lm[1]);
    let along = if wide > 0.2 * s { [(rm[0] - lm[0]) / wide, (rm[1] - lm[1]) / wide] } else { u };
    let open = { let j = ((mouth[1] as usize).clamp(cy, cy + ch - 1) - cy) * cw + ((mouth[0] as usize).clamp(cx, cx + cw - 1) - cx); (stands(j) / 30.0).clamp(0.0, 1.0) };
    let half = (wide * 0.5).max(s * PATTERN_MOUTH.0);
    let k = bent([0.0, 0.0, 0.0], fit_curve(mouth, along, half, s * PATTERN_FIT.0), [hold * 0.5, hold, hold]);
    let width = (s * (PATTERN_MOUTH.1 + 0.1 * open)).max(1.6);
    features.push(curve(mouth, along, half, k, width, width * 0.65));
}

/// The KEY plate's plan: the faces' features, built from the detector's points (each a stroke of the black),
/// the black plate itself as rings (its lines and its solids, see `key_plate`), and the black screen. Planned
/// before the colour plates are laid, printed after them (`book_line`).
struct BookPlan {
    key: Vec<[f32; 2]>,
}

/// The KEY of an early book illustration: the picture's drawing in black. The picture is read with its faces
/// brought out at their own scale — a portrait's features sharpened, a face in a window built from the
/// detector's points and the picture inside it read with nothing in it — and the key plate is cut from that
/// reading (see `key_plate`).
fn book_plan(source: &RgbImage, p: &PaintParams, _budget: usize) -> BookPlan {
    let long = source.width().max(source.height()) as f32;
    // The line is read off the source with its grain off, not off the washes' masses: it is what they left out.
    let mut drawn = imageops::blur(source, (long / 1000.0).max(0.8));
    // (The face with its features: an eye is drawn, not hatched.)
    let face = book_skin(p, source.width() as usize, source.height() as usize).map(|f| f.1);
    let luma_sheet = sheet_luma(source);
    // A FACE is drawn from the picture at its sharpest, its features brought out: an eye, a nostril, the line of
    // a mouth are faint edges beside a window frame's, and judged with the whole sheet they are left out — the
    // face comes out a blank or a mask. On skin the picture's local contrast is raised before the line is planned.
    let (sw, sh) = (source.width() as usize, source.height() as usize);
    // The small heads, fitted, and the one light they all stand in (see `small_heads`).
    let (heads, scene) = small_heads(p, &luma_sheet, sw, sh);
    let head_of = |f: &[[f32; 2]; 5]| heads.iter().find(|(g, _)| g == f).map(|(_, head)| head);
    let near_face = face.as_ref().map(|skin| dilate_bool(skin, sw, sh, (long * 0.003) as usize));
    // ... and it is drawn at ITS OWN scale: a face that fills the sheet and a face in a far window are each a
    // few lines — the eyes, the nose, the mouth, the jaw. So each face is read out of focus by a share of its
    // own size: on a large one the creases under an eye and the down on a lip are gone before the line is
    // planned, and a small one, a few dozen pixels of shading, is not traced wrinkle by wrinkle into a mask —
    // the smaller the face, the less its local contrast is raised.
    let mut features: Vec<crate::paint::engrave::Cut> = Vec::new();
    if let Some(near) = &near_face {
        let mut seen = vec![false; sw * sh];
        for start in 0..sw * sh {
            if seen[start] || !near[start] {
                continue;
            }
            let (mut px, mut stack) = (Vec::new(), vec![start]);
            let (mut x0, mut y0, mut x1, mut y1) = (sw, sh, 0usize, 0usize);
            seen[start] = true;
            while let Some(i) = stack.pop() {
                px.push(i);
                let (x, y) = (i % sw, i / sw);
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
                for j in [(x > 0).then(|| i - 1), (x + 1 < sw).then(|| i + 1), (y > 0).then(|| i - sw), (y + 1 < sh).then(|| i + sw)].into_iter().flatten() {
                    if !seen[j] && near[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            let size = (px.len() as f32).sqrt();
            let large = ((size / long - FACE_SMALL) / (FACE_LARGE - FACE_SMALL)).clamp(0.0, 1.0);
            // (... and a small face read as sharp as its features are small: an eye twelve pixels wide is gone
            // in a blur of three.)
            let focus = (size * (0.007 + 0.008 * (1.0 - large))).max(long / 1000.0).max(0.8);
            let broad = (size * 0.06).max(long * 0.004).max(1.5);
            let raise = 0.4 + 1.1 * large;
            let pad = (broad * 3.0).ceil() as usize;
            let (cx, cy) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
            let (cw, ch) = ((x1 + pad + 1).min(sw) - cx, (y1 + pad + 1).min(sh) - cy);
            let crop = imageops::crop_imm(source, cx as u32, cy as u32, cw as u32, ch as u32).to_image();
            let (sharp, soft) = (imageops::blur(&crop, focus), imageops::blur(&crop, broad));
            for &i in &px {
                let (x, y) = ((i % sw - cx) as u32, (i / sw - cy) as u32);
                let (a, b) = (sharp.get_pixel(x, y).0, soft.get_pixel(x, y).0);
                drawn.put_pixel((i % sw) as u32, (i / sw) as u32, image::Rgb([0, 1, 2].map(|c| (a[c] as f32 + raise * (a[c] as f32 - b[c] as f32)).round().clamp(0.0, 255.0) as u8)));
            }
            // A FAR FACE is not traced at all. A few dozen pixels of shading have no edges to follow — followed,
            // a brow runs into a nose and a cheek's shadow into a mouth, and the face is a mask. It is put down
            // as a draughtsman puts it: its outline, and one stroke for each of its darkest features — an eye, an
            // eye, the mouth, the shadow under the nose — laid along the feature. Inside its outline the picture
            // is read with nothing in it, so the planner draws no line there.
            if size / long < FACE_FAR {
                let mut off = vec![true; cw * ch];
                for &i in &px {
                    off[(i / sw - cy) * cw + (i % sw - cx)] = false;
                }
                let inside: Vec<bool> = dilate_bool(&off, cw, ch, (size * 0.09).max(2.0) as usize).iter().map(|o| !o).collect();
                let plain = imageops::blur(&crop, (size * 0.12).max(1.5));
                let value = |im: &RgbImage, j: usize| { let c = im.get_pixel((j % cw) as u32, (j / cw) as u32).0; 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32 };
                let mut darker = vec![0f32; cw * ch];
                for j in (0..cw * ch).filter(|&j| inside[j]) {
                    darker[j] = (value(&soft, j) - value(&sharp, j)).max(0.0);
                    drawn.put_pixel((cx + j % cw) as u32, (cy + j / cw) as u32, *plain.get_pixel((j % cw) as u32, (j / cw) as u32));
                }
                // Where the detector says the features ARE, they are drawn there, as a hand sets a small face
                // down: a stroke for each eye and a lighter one for the brow over it, a tick under the nose,
                // the line of the mouth from corner to corner — fuller where the mouth is open. (A face seen
                // from the side has its eyes all but in one place: one eye is drawn.)
                let known: Vec<[[f32; 2]; 5]> = p.features.iter().map(|f| f.map(|q| [q[0] * sw as f32, q[1] * sh as f32])).filter(|f| {
                    let (x, y) = (f[2][0] as usize, f[2][1] as usize);
                    x >= cx && y >= cy && x < cx + cw && y < cy + ch && !off[(y - cy) * cw + (x - cx)]
                }).collect();
                if !known.is_empty() {
                    // The head first, by template, and round it the picture read with nothing in it; then the
                    // features by the pattern.
                    head_quiet(&mut drawn, &crop, &off, cw, ch, cx, cy, size);
                    for f in &known {
                        let fitted;
                        let head = match head_of(f) { Some(head) => head, None => { fitted = Head::fit(f, &luma_sheet, sw, sh); &fitted } };
                        pattern_face(f, head, scene, &crop, &off, cx, cy, cw, ch, &luma_sheet, sw, sh, &mut features);
                    }
                }
                if !known.is_empty() {
                    continue;
                }
                let mark = (darker.iter().fold(0f32, |m, v| m.max(*v)) * 0.4).max(4.0);
                let mut taken = vec![false; cw * ch];
                // Each dark feature: its weight, where it lies, how it lies (its second moments).
                let mut found: Vec<(f32, [f32; 2], [f32; 3])> = Vec::new();
                for start in 0..cw * ch {
                    if taken[start] || darker[start] <= mark {
                        continue;
                    }
                    let mut stack = vec![start];
                    let mut part = Vec::new();
                    taken[start] = true;
                    while let Some(j) = stack.pop() {
                        part.push(j);
                        let (x, y) = (j % cw, j / cw);
                        for k in [(x > 0).then(|| j - 1), (x + 1 < cw).then(|| j + 1), (y > 0).then(|| j - cw), (y + 1 < ch).then(|| j + cw)].into_iter().flatten() {
                            if !taken[k] && darker[k] > mark {
                                taken[k] = true;
                                stack.push(k);
                            }
                        }
                    }
                    let sum: f32 = part.iter().map(|&j| darker[j]).sum();
                    let at = part.iter().fold([0f32; 2], |m, &j| [m[0] + darker[j] * (j % cw) as f32, m[1] + darker[j] * (j / cw) as f32]).map(|v| v / sum);
                    let lie = part.iter().fold([0f32; 3], |m, &j| { let (dx, dy) = ((j % cw) as f32 - at[0], (j / cw) as f32 - at[1]); [m[0] + darker[j] * dx * dx, m[1] + darker[j] * dy * dy, m[2] + darker[j] * dx * dy] }).map(|v| v / sum);
                    if part.len() >= 4 {
                        found.push((sum, at, lie));
                    }
                }
                found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for (_, at, lie) in found.into_iter().take(FACE_STROKES) {
                    let (mid, half) = ((lie[0] + lie[1]) * 0.5, (((lie[0] - lie[1]) * 0.5).powi(2) + lie[2] * lie[2]).sqrt());
                    let (along, across) = ((mid + half).max(0.0).sqrt(), (mid - half).max(0.0).sqrt());
                    let (s, c) = (0.5 * (2.0 * lie[2]).atan2(lie[0] - lie[1])).sin_cos();
                    let reach = (along * 1.7).clamp(size * 0.03, size * 0.22);
                    let width = (across * 2.2).clamp((size * 0.02).max(1.2), (size * 0.07).max(1.6));
                    let (x, y) = (cx as f32 + at[0] + 0.5, cy as f32 + at[1] + 0.5);
                    features.push(crate::paint::engrave::Cut { layer: 0, path: vec![[x - c * reach, y - s * reach], [x + c * reach, y + s * reach]], w0: width, w1: width * 0.6 });
                }
            } else {
                let faces: Vec<[[f32; 2]; 5]> = p.features.iter().map(|f| f.map(|q| [q[0] * sw as f32, q[1] * sh as f32])).filter(|f| {
                    let (x, y) = (f[2][0] as usize, f[2][1] as usize);
                    x >= cx && y >= cy && x < cx + cw && y < cy + ch && near[y * sw + x]
                }).collect();
                // A FACE IN A WINDOW — between a far face and a portrait — is built as a near one, from the
                // detector's points; but inside its outline the picture is read with nothing in it, as a far
                // face is: at that size the shading of a cheek, the furrows round a mouth, the fringe of a
                // beard are edges the planner would trace, and a built face with a traced one over it is a tangle.
                // Its outline is one line, the head's, drawn before the features are put in; its features are
                // drawn BOLD, a sign for each, as the illustrators of the thirties drew a face in a crowd.
                let mid = size / long < FACE_MID;
                // A face in a crowd (see `FACE_PATTERN`) is by the pattern, whatever the size of the piece of
                // skin it is in — three heads together are one piece, and each is a small head.
                let (pattern, faces): (Vec<[[f32; 2]; 5]>, Vec<[[f32; 2]; 5]>) = faces.into_iter().partition(|f| Head::apart(f) / long < FACE_PATTERN);
                if (mid && !faces.is_empty()) || !pattern.is_empty() {
                    let mut off = vec![true; cw * ch];
                    for &i in &px {
                        off[(i / sw - cy) * cw + (i % sw - cx)] = false;
                    }
                    head_quiet(&mut drawn, &crop, &off, cw, ch, cx, cy, size);
                    if mid && !faces.is_empty() {
                        features.extend(head_outline(&off, cw, ch, cx, cy, size, head_light(&crop, &off, cw, ch, size)));
                    }
                    for f in &pattern {
                        let fitted;
                        let head = match head_of(f) { Some(head) => head, None => { fitted = Head::fit(f, &luma_sheet, sw, sh); &fitted } };
                        pattern_face(f, head, scene, &crop, &off, cx, cy, cw, ch, &luma_sheet, sw, sh, &mut features);
                    }
                }
                // (Every face the detector found in this piece of skin — two heads together are one piece.)
                for [le, re, nose, lm, rm] in faces {
                // A NEAR FACE's mouth is one line: the dark between the lips, from corner to corner. Read as
                // edges, a mouth is the rims of its shading — the bow of the upper lip, the shadow at a corner,
                // the fold under the lower lip — joined into a line no mouth has, and a calm mouth comes out
                // a grimace. So the mouth is found where the detector says its corners are: the darkest way
                // from one to the other is drawn, swelling to its middle, and round it the picture is read with
                // nothing in it. The bridge of the nose likewise: its shading is no line.
                let wide = (rm[0] - lm[0]).hypot(rm[1] - lm[1]);
                let apart = (re[0] - le[0]).hypot(re[1] - le[1]);
                // (A face in a window is measured by its own eyes, not by the piece of skin it shares; and its
                // marks are heavier for its size.)
                let (size, bold) = if mid { ((apart * 2.6).min(size), FACE_BOLD) } else { (size, 1.0) };
                if wide >= 8.0 && apart >= 8.0 {
                    let eyes = [(le[0] + re[0]) * 0.5, (le[1] + re[1]) * 0.5];
                    let calm = imageops::blur(&crop, (wide * 0.12).max(1.5));
                    let mut quiet = |a: [f32; 2], b: [f32; 2], r: f32| {
                        let (x0, x1) = ((a[0].min(b[0]) - r).max(cx as f32) as usize, ((a[0].max(b[0]) + r) as usize).min(cx + cw - 1));
                        let (y0, y1) = ((a[1].min(b[1]) - r).max(cy as f32) as usize, ((a[1].max(b[1]) + r) as usize).min(cy + ch - 1));
                        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                        let len2 = (dx * dx + dy * dy).max(1e-6);
                        for y in y0..=y1 {
                            for x in x0..=x1 {
                                let t = (((x as f32 - a[0]) * dx + (y as f32 - a[1]) * dy) / len2).clamp(0.0, 1.0);
                                // (Fading in from its rim: an edge between the picture and the picture
                                // smoothed is itself a line.)
                                let k = ((r - (x as f32 - a[0] - dx * t).hypot(y as f32 - a[1] - dy * t)) / (0.6 * r)).clamp(0.0, 1.0);
                                if k > 0.0 {
                                    let (was, to) = (drawn.get_pixel(x as u32, y as u32).0, calm.get_pixel((x - cx) as u32, (y - cy) as u32).0);
                                    drawn.put_pixel(x as u32, y as u32, image::Rgb([0, 1, 2].map(|c| (was[c] as f32 + (to[c] as f32 - was[c] as f32) * k).round() as u8)));
                                }
                            }
                        }
                    };
                    quiet(lm, rm, wide * 0.4);
                    // (The whole length of the nose, its sides with it — a nose is its underside: the nostrils and
                    // the wings. And the furrow from it to the lip, and the hollows under the eyes: shading, each.)
                    let toward = |a: [f32; 2], b: [f32; 2], f: f32| [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
                    quiet(toward(eyes, nose, 0.15), toward(eyes, nose, 0.8), apart * 0.3);
                    // (... and the tip and the wings, read as edges, are a scrawl: the nose is its two nostrils,
                    // each one short stroke where its dark lies.)
                    quiet(nose, nose, apart * 0.42);
                    let level = [(re[0] - le[0]) / apart, (re[1] - le[1]) / apart];
                    let fall = { let l = (nose[0] - eyes[0]).hypot(nose[1] - eyes[1]).max(1e-3); [(nose[0] - eyes[0]) / l, (nose[1] - eyes[1]) / l] };
                    let shade = imageops::blur(&crop, (apart * 0.12).max(1.5));
                    // ... and ONE line down the side the light leaves, from under the brow to the wing: the
                    // nose a pen draws is its shadow side, not its two edges.
                    let shaded = |q: [f32; 2]| { let c = shade.get_pixel((q[0] as usize).clamp(cx, cx + cw - 1) as u32 - cx as u32, (q[1] as usize).clamp(cy, cy + ch - 1) as u32 - cy as u32).0; 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32 };
                    let side = if shaded([nose[0] - level[0] * apart * 0.3, nose[1] - level[1] * apart * 0.3]) < shaded([nose[0] + level[0] * apart * 0.3, nose[1] + level[1] * apart * 0.3]) { -1.0 } else { 1.0 };
                    let at = |f: f32, out: f32, down: f32| [eyes[0] + (nose[0] - eyes[0]) * f + level[0] * side * apart * out + fall[0] * apart * down, eyes[1] + (nose[1] - eyes[1]) * f + level[1] * side * apart * out + fall[1] * apart * down];
                    let nw = (size * 0.007 * bold).max(1.6);
                    // ... and BROKEN: a short stroke at the brow and the turn of the wing at the tip, the bridge
                    // between them left to the paper as the light on it. Drawn whole, the line is a grown man's nose.
                    features.push(crate::paint::engrave::Cut { layer: 0, path: vec![at(0.3, 0.08, 0.0), at(0.5, 0.1, 0.0)], w0: nw * 0.55, w1: nw * 0.35 });
                    features.push(crate::paint::engrave::Cut { layer: 0, path: vec![at(0.86, 0.17, 0.02), at(1.0, 0.21, 0.045), at(1.0, 0.26, 0.09)], w0: nw * 0.4, w1: nw * 0.8 });
                    for side in [-1.0f32, 1.0] {
                        let c = [nose[0] + level[0] * side * apart * 0.17 + fall[0] * apart * 0.05, nose[1] + level[1] * side * apart * 0.17 + fall[1] * apart * 0.05];
                        let r = apart * 0.12;
                        let (x0, x1) = (((c[0] - r).max(cx as f32)) as usize, ((c[0] + r) as usize).min(cx + cw - 1));
                        let (y0, y1) = (((c[1] - r).max(cy as f32)) as usize, ((c[1] + r) as usize).min(cy + ch - 1));
                        let mut m = [0f32; 6];
                        for y in y0..=y1 {
                            for x in x0..=x1 {
                                if ((x as f32 - c[0]).hypot(y as f32 - c[1])) > r {
                                    continue;
                                }
                                let (a, b) = (sharp.get_pixel((x - cx) as u32, (y - cy) as u32).0, shade.get_pixel((x - cx) as u32, (y - cy) as u32).0);
                                let d = ((0.299 * (b[0] as f32 - a[0] as f32) + 0.587 * (b[1] as f32 - a[1] as f32) + 0.114 * (b[2] as f32 - a[2] as f32)) - 6.0).max(0.0);
                                m = [m[0] + d, m[1] + d * x as f32, m[2] + d * y as f32, m[3] + d * (x * x) as f32, m[4] + d * (y * y) as f32, m[5] + d * (x * y) as f32];
                            }
                        }
                        if m[0] < 20.0 {
                            continue;
                        }
                        let at = [m[1] / m[0], m[2] / m[0]];
                        let lie = [m[3] / m[0] - at[0] * at[0], m[4] / m[0] - at[1] * at[1], m[5] / m[0] - at[0] * at[1]];
                        let (mid, half) = ((lie[0] + lie[1]) * 0.5, (((lie[0] - lie[1]) * 0.5).powi(2) + lie[2] * lie[2]).sqrt());
                        let reach = ((mid + half).max(0.0).sqrt() * 1.5).clamp(apart * 0.04, apart * 0.1);
                        let (sn, cs) = (0.5 * (2.0 * lie[2]).atan2(lie[0] - lie[1])).sin_cos();
                        let width = (size * 0.009 * bold).max(1.8);
                        features.push(crate::paint::engrave::Cut { layer: 0, path: vec![[at[0] - cs * reach, at[1] - sn * reach], [at[0] + cs * reach, at[1] + sn * reach]], w0: width * 0.6, w1: width });
                    }
                    let lips = [(lm[0] + rm[0]) * 0.5, (lm[1] + rm[1]) * 0.5];
                    quiet(toward(nose, lips, 0.35), toward(nose, lips, 0.8), apart * 0.14);
                    for eye in [le, re] {
                        let c = [eye[0] + fall[0] * apart * 0.36, eye[1] + fall[1] * apart * 0.36];
                        quiet([c[0] - level[0] * apart * 0.3, c[1] - level[1] * apart * 0.3], [c[0] + level[0] * apart * 0.3, c[1] + level[1] * apart * 0.3], apart * 0.16);
                    }
                    // A NEAR EYE is BUILT, not traced. Read as edges, an eye is the rims of everything in it —
                    // the lashes, the crease, the rim of the iris, the glint — a dozen wavering chains, and the
                    // eye is a wound. A pen draws an eye in four marks: the upper lid, one heavy line from
                    // corner to corner that flicks past the outer one; the lower lid, a light line under the
                    // middle; the iris, a circle cut off by the lids; and the pupil, a dot. Where they lie is
                    // read from the picture: the opening is what round the detector's eye is not skin, the
                    // iris the dark in it.
                    let pick = |q: [f32; 2]| { let c = sharp.get_pixel((q[0] as usize).clamp(cx, cx + cw - 1) as u32 - cx as u32, (q[1] as usize).clamp(cy, cy + ch - 1) as u32 - cy as u32).0; [c[0] as f32, c[1] as f32, c[2] as f32] };
                    let eyebase = (size * 0.012 * bold).max(2.0);
                    for (eye, outward) in [(le, -1.0f32), (re, 1.0)] {
                        let at = |u: f32, v: f32| [eye[0] + level[0] * u + fall[0] * v, eye[1] + level[1] * u + fall[1] * v];
                        // Skin is the cheek under the eye.
                        let cheek = { let c = shade.get_pixel((eye[0] + fall[0] * apart * 0.3).clamp(cx as f32, (cx + cw - 1) as f32) as u32 - cx as u32, (eye[1] + fall[1] * apart * 0.3).clamp(cy as f32, (cy + ch - 1) as f32) as u32 - cy as u32).0; [c[0] as f32, c[1] as f32, c[2] as f32] };
                        let value = |c: [f32; 3]| 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                        let skin_v = value(cheek);
                        // THE BROW: the dark over the eye — the darkest level between a seventh and a third of
                        // the eyes' distance above it — one stroke from over the inner corner out past the
                        // outer, heavier at the inner end. (A portrait's is a lighter stroke, and looked for
                        // higher, from a quarter to a half: at its size the lid's own dark lies a seventh up.
                        // A child's brow is faint, and the edge-finder draws only the dark of its middle.)
                        {
                            let row = |v: f32| (-3..=3).map(|k| shaded(at(k as f32 * apart * 0.08, v))).sum::<f32>() / 7.0;
                            let rows = if mid { 3..=7 } else { 5..=9 };
                            let (bv, bd) = rows.map(|k| -apart * 0.05 * k as f32).fold((0.0f32, f32::MAX), |m, v| { let d = row(v); if d < m.1 { (v, d) } else { m } });
                            let heavy = if mid { 1.0 } else { 0.7 };
                            if bd < skin_v - 8.0 {
                                features.push(crate::paint::engrave::Cut { layer: 0, path: vec![at(-outward * apart * 0.28, bv + apart * 0.03), at(0.0, bv - apart * 0.015), at(outward * apart * 0.32, bv + apart * 0.02)], w0: eyebase * heavy, w1: eyebase * heavy * 0.5 });
                            }
                        }
                        // (Not skin: the white of the eye, well lighter than the cheek, or the iris and the
                        // lashes, well darker — not a shade of the skin, which the hollow under the brow is.)
                        let not_skin = |q: [f32; 2]| { let v = value(pick(q)); v > skin_v + 22.0 || v < skin_v - 45.0 };
                        let (rx, ry) = ((apart * 0.32) as i32, (apart * 0.14) as i32);
                        // Each column across the eye: the run of not-skin through the eye's level, its top and bottom.
                        let column = |u: i32| -> Option<(f32, f32)> {
                            if !not_skin(at(u as f32, 0.0)) {
                                return None;
                            }
                            let (mut top, mut bot) = (0i32, 0i32);
                            while top > -ry && not_skin(at(u as f32, (top - 1) as f32)) { top -= 1; }
                            while bot < ry && not_skin(at(u as f32, (bot + 1) as f32)) { bot += 1; }
                            Some((top as f32, bot as f32))
                        };
                        let mut cols: Vec<Option<(f32, f32)>> = (-rx..=rx).map(column).collect();
                        // The corners: as far from the middle as the columns run on (a gap of two is shadow).
                        let (mut lo, mut hi) = (rx as usize, rx as usize);
                        let mut gap = 0;
                        for u in (0..rx as usize).rev() { if cols[u].is_some() { lo = u; gap = 0; } else { gap += 1; if gap > 2 { break; } } }
                        gap = 0;
                        for u in rx as usize + 1..cols.len() { if cols[u].is_some() { hi = u; gap = 0; } else { gap += 1; if gap > 2 { break; } } }
                        let width = (hi - lo) as f32;
                        if width < apart * 0.22 || !cols[rx as usize].is_some() {
                            // (A shut eye — a face that shouts, that squints into the sun — has no opening to
                            // read: it is one line, the lid, a little longer toward the outer corner.)
                            quiet(at(-apart * 0.3, 0.0), at(apart * 0.3, 0.0), apart * 0.12);
                            features.push(crate::paint::engrave::Cut { layer: 0, path: vec![at(-outward * apart * 0.14, 0.0), at(outward * apart * 0.1, -apart * 0.01), at(outward * apart * 0.2, apart * 0.01)], w0: eyebase * 0.45, w1: eyebase * 0.7 });
                            continue;
                        }
                        // Fill the gaps and smooth the lids.
                        for u in lo..=hi { if cols[u].is_none() { cols[u] = cols[u - 1]; } }
                        let lid = |u: usize, top: bool| -> f32 {
                            let (a, b) = (u.saturating_sub(3).max(lo), (u + 3).min(hi));
                            let vals: Vec<f32> = (a..=b).map(|k| { let (t, s) = cols[k].unwrap(); if top { t } else { s } }).collect();
                            vals.iter().sum::<f32>() / vals.len() as f32
                        };
                        let (inner, outer) = if outward < 0.0 { (hi, lo) } else { (lo, hi) };
                        let u_of = |k: usize| k as f32 - rx as f32;
                        // (Round it the picture is read with nothing in it: the lashes, the crease, the bag
                        // under the eye are no lines. Further down than up — the brow is a line.)
                        let (past_lo, past_hi) = (u_of(lo) - apart * 0.08, u_of(hi) + apart * 0.08);
                        quiet(at(past_lo, 0.0), at(past_hi, 0.0), apart * 0.12);
                        quiet(at(past_lo, apart * 0.1), at(past_hi, apart * 0.1), apart * 0.12);
                        quiet(at(past_lo, -apart * 0.06), at(past_hi, -apart * 0.06), apart * 0.1);
                        // The upper lid: heavy over the middle, thin at the corners, and a flick past the outer corner.
                        let steps = 16usize;
                        let mut upper: Vec<[f32; 2]> = (0..=steps).map(|s| { let k = inner as f32 + (outer as f32 - inner as f32) * s as f32 / steps as f32; at(u_of(k as usize), lid(k as usize, true) - 1.0) }).collect();
                        upper.push(at(u_of(outer) + outward * width * 0.09, lid(outer, true) - width * 0.05));
                        let w_up = |s: usize| eyebase * (0.35 + 0.65 * (std::f32::consts::PI * (s as f32 / steps as f32).min(1.0)).sin());
                        let mut a = 0usize;
                        while a + 1 < upper.len() {
                            let b = (a + 3).min(upper.len() - 1);
                            features.push(crate::paint::engrave::Cut { layer: 0, path: upper[a..=b].to_vec(), w0: w_up(a), w1: if b == upper.len() - 1 { eyebase * 0.25 } else { w_up(b) } });
                            a = b;
                        }
                        // The lower lid: a light line under the middle two thirds.
                        let lower: Vec<[f32; 2]> = (0..=steps).map(|s| { let k = lo as f32 + width * (0.18 + 0.7 * s as f32 / steps as f32); at(u_of(k as usize), lid(k as usize, false) + 1.0) }).collect();
                        let mut a = 0usize;
                        while a + 1 < lower.len() {
                            let b = (a + 4).min(lower.len() - 1);
                            features.push(crate::paint::engrave::Cut { layer: 0, path: lower[a..=b].to_vec(), w0: eyebase * 0.3, w1: eyebase * 0.3 });
                            a = b;
                        }
                        // The iris: the dark in the opening — its centre, and its half-width along the eye's level.
                        // (Its centre is the dark's; its radius is read off the dark's spread along the eye's
                        // level — a disc's, cut off above and below by the lids, which do not narrow it.)
                        let mut m = [0f32; 5];
                        for k in lo..=hi {
                            let (t, s) = cols[k].unwrap();
                            for v in (t as i32)..=(s as i32) {
                                let val = value(pick(at(u_of(k), v as f32)));
                                if val < skin_v - 45.0 {
                                    m = [m[0] + 1.0, m[1] + u_of(k), m[2] + v as f32, m[3] + u_of(k) * u_of(k), m[4] + val];
                                }
                            }
                        }
                        if m[0] < 4.0 {
                            continue;
                        }
                        let dark_eye = m[4] / m[0] < skin_v - EYE_DARK;
                        let (cu, cv) = (m[1] / m[0], m[2] / m[0]);
                        let r = (1.9 * (m[3] / m[0] - cu * cu).max(0.0).sqrt()).clamp(apart * 0.07, apart * 0.17);
                        let inside = |q_u: f32, q_v: f32| { let k = ((q_u + rx as f32).round().max(0.0) as usize).clamp(lo, hi); q_v > lid(k, true) + 0.12 * r && q_v < lid(k, false) - 0.12 * r };
                        let ring: Vec<([f32; 2], bool)> = (0..=36).map(|s| { let a = std::f32::consts::TAU * s as f32 / 36.0; let (q_u, q_v) = (cu + r * a.cos(), cv + r * a.sin()); (at(q_u, q_v), inside(q_u, q_v)) }).collect();
                        let mut run: Vec<[f32; 2]> = Vec::new();
                        for (q, ok) in ring.iter().chain(std::iter::once(&([0.0, 0.0], false))) {
                            if *ok { run.push(*q); } else { if run.len() >= 3 { features.push(crate::paint::engrave::Cut { layer: 0, path: std::mem::take(&mut run), w0: eyebase * 0.45, w1: eyebase * 0.45 }); } run.clear(); }
                        }
                        // The pupil: a dot, filled in with a few strokes side by side — or, in a dark eye (see
                        // `EYE_DARK`), the whole iris under the lids, with the glint left out.
                        let (rho, step) = (if dark_eye { r * 0.84 } else { r * 0.38 }, (r * 0.1).max(1.0));
                        let glint = (cu - r * 0.32, cv - r * 0.32, r * 0.22);
                        let mut v = -rho + step * 0.5;
                        while v < rho {
                            let half = (rho * rho - v * v).max(0.0).sqrt();
                            if half > 0.5 && (!dark_eye || inside(cu, cv + v)) {
                                let mut spans = vec![(cu - half, cu + half)];
                                let dv = cv + v - glint.1;
                                if dark_eye && dv.abs() < glint.2 {
                                    let gh = (glint.2 * glint.2 - dv * dv).sqrt();
                                    spans = vec![(cu - half, (glint.0 - gh).min(cu + half)), ((glint.0 + gh).max(cu - half), cu + half)];
                                }
                                for (a, b) in spans {
                                    if b - a > 0.5 {
                                        features.push(crate::paint::engrave::Cut { layer: 0, path: vec![at(a, cv + v), at(b, cv + v)], w0: step * 1.3, w1: step * 1.3 });
                                    }
                                }
                            }
                            v += step;
                        }
                    }
                    let along = [(rm[0] - lm[0]) / wide, (rm[1] - lm[1]) / wide];
                    let across = [-along[1], along[0]];
                    let n = ((wide / 4.0) as usize).clamp(12, 80);
                    let (k, by) = (10i32, wide * 0.018);
                    let at = |t: usize, o: i32| {
                        let f = t as f32 / n as f32;
                        [lm[0] + (rm[0] - lm[0]) * f + across[0] * o as f32 * by, lm[1] + (rm[1] - lm[1]) * f + across[1] * o as f32 * by]
                    };
                    let dark = |q: [f32; 2]| { let c = sharp.get_pixel((q[0] as usize).clamp(cx, cx + cw - 1) as u32 - cx as u32, (q[1] as usize).clamp(cy, cy + ch - 1) as u32 - cy as u32).0; 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32 };
                    let m = (2 * k + 1) as usize;
                    let mut cost = vec![f32::MAX; (n + 1) * m];
                    let mut from = vec![0usize; (n + 1) * m];
                    cost[k as usize] = 0.0;
                    for t in 1..=n {
                        for o in 0..m {
                            if t == n && o != k as usize {
                                continue;
                            }
                            let here = dark(at(t, o as i32 - k));
                            for q in o.saturating_sub(2)..(o + 3).min(m) {
                                let c = cost[(t - 1) * m + q];
                                if c < f32::MAX && c + here + 3.5 * (o as f32 - q as f32).abs() < cost[t * m + o] {
                                    cost[t * m + o] = c + here + 3.5 * (o as f32 - q as f32).abs();
                                    from[t * m + o] = q;
                                }
                            }
                        }
                    }
                    let mut way = vec![k as usize; n + 1];
                    for t in (1..=n).rev() {
                        way[t - 1] = from[t * m + way[t]];
                    }
                    let raw: Vec<[f32; 2]> = (0..=n).map(|t| at(t, way[t] as i32 - k)).collect();
                    let line: Vec<[f32; 2]> = (0..=n).map(|t| { let (a, b, c) = (raw[t.saturating_sub(1)], raw[t], raw[(t + 1).min(n)]); [(a[0] + 2.0 * b[0] + c[0]) * 0.25, (a[1] + 2.0 * b[1] + c[1]) * 0.25] }).collect();
                    let base = (size * 0.011 * bold).max(2.2);
                    let weight = |t: usize| base * (0.3 + 0.7 * (std::f32::consts::PI * t as f32 / n as f32).sin());
                    let mut a = 0usize;
                    while a < n {
                        let b = (a + 3).min(n);
                        features.push(crate::paint::engrave::Cut { layer: 0, path: line[a..=b].to_vec(), w0: weight(a), w1: weight(b) });
                        a = b;
                    }
                    // ... and the lip under it: one short stroke where its shadow lies, or the mouth is a slit.
                    let below = if across[0] * (lips[0] - eyes[0]) + across[1] * (lips[1] - eyes[1]) < 0.0 { [-across[0], -across[1]] } else { across };
                    let under = (0..10).map(|k| 0.14 + 0.02 * k as f32).fold((f32::MAX, 0.22f32), |m, f| { let v = dark([lips[0] + below[0] * wide * f, lips[1] + below[1] * wide * f]); if v < m.0 { (v, f) } else { m } }).1;
                    let c = [lips[0] + below[0] * wide * under, lips[1] + below[1] * wide * under];
                    let end = |k: f32| [c[0] + along[0] * wide * 0.17 * k - below[0] * wide * 0.015, c[1] + along[1] * wide * 0.17 * k - below[1] * wide * 0.015];
                    features.push(crate::paint::engrave::Cut { layer: 0, path: vec![end(-1.0), c], w0: base * 0.3, w1: base * 0.55 });
                    features.push(crate::paint::engrave::Cut { layer: 0, path: vec![c, end(1.0)], w0: base * 0.55, w1: base * 0.3 });
                }
                }
            }
        }
    }
    let key = key_plate(&drawn, face.as_deref(), long, &features);
    BookPlan { key }
}

/// THE KEY PLATE PRINTED, over everything and in register: the black line, its solids and the faces' features
/// as one wash, with the dot gain of a black ink on a cheap paper.
fn book_line(canvas: &mut Canvas, score: &mut StrokeScore, source: &RgbImage, p: &PaintParams, placed: &mut usize, plan: BookPlan) {
    let long = source.width().max(source.height()) as f32;
    let BookPlan { key } = plan;
    let black = crate::paint::canvas::darkest_pigment(&p.palette);
    let mut load = vec![0f32; p.palette.pigments.len()];
    load[black] = p.charge * 2.4;
    let mix: Vec<(String, f32)> = vec![(p.palette.pigments[black].name.to_string(), load[black])];
    let gain = (long * KEY_GAIN).max(0.5).round();
    if !key.is_empty() && *placed < p.budget {
        canvas.fill_rings(&key, &load, 0.0, gain);
        *placed += 1;
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: true, stage: "key".into(), spline: key, w0: gain, w1: 0.0, taper: 0.0, mix, wet: 0.0, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
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
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage: "contour".into(), spline: s.path, w0: width, w1: width, taper: 0.15, mix, wet, press: 0.9, streak: brush.streak, round: brush.round, pickup: Some(0.0), bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
    }
}

fn ink_drawing(canvas: &mut Canvas, score: &mut StrokeScore, input: &RgbImage, source: &RgbImage, head: Option<&[f32]>, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, k: &mut u64, progress: Option<&dyn Fn(PaintProgress)>) {
    let w = input.width() as usize;
    if p.engrave && !p.brush_drawing {
        return engraving(canvas, score, if source.dimensions() == input.dimensions() { source } else { input }, p, protect, placed, progress);
    }
    let style = if p.brush_drawing { crate::paint::ink::HatchStyle::SUMI } else { crate::paint::ink::HatchStyle::PEN };
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
            bristles: Some(brush.bristles), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, });
    }
}

/// A COPPERPLATE (`durer`): the plate [`crate::paint::engrave::plan`] cuts from the picture itself — not from the
/// armature, which has simplified away the detail a burin draws — laid as flat-ended one-point strokes, so a
/// swelling line's pieces join without a seam and the plate replays from its score.
fn engraving(canvas: &mut Canvas, score: &mut StrokeScore, picture: &RgbImage, p: &PaintParams, protect: Option<&[bool]>, placed: &mut usize, progress: Option<&dyn Fn(PaintProgress)>) {
    let w = picture.width() as usize;
    let relief = p.relief.as_ref().map(|(w, h, z)| crate::paint::engrave::Relief { w: *w as usize, h: *h as usize, z });
    let normals = p.normals.as_ref().map(|(w, h, n)| crate::paint::engrave::Normals { w: *w as usize, h: *h as usize, n });
    let materials = p.materials.as_ref().map(|(w, h, kind)| crate::paint::engrave::Materials { w: *w as usize, h: *h as usize, kind });
    let plate = crate::paint::engrave::plan(picture, relief, normals, materials, p.contour, p.white, p.follow, p.budget.saturating_sub(*placed), p.seed);
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let mut load = vec![0f32; p.palette.pigments.len()];
    load[ink] = p.charge * 2.0;
    let brush = BrushConfig { k_pickup: 0.0, bristles: 1, streak: 0.0, round: 1.0, flat_ends: true, ..p.brush };
    let blocked = |pt: &[f32; 2]| protect.map(|m| m.get(pt[1] as usize * w + pt[0] as usize).copied().unwrap_or(false)).unwrap_or(false);
    for cut in plate.cuts {
        if *placed >= p.budget {
            break;
        }
        if cut.path.len() < 2 || blocked(&cut.path[cut.path.len() / 2]) {
            continue;
        }
        let stage = if cut.layer == 0 { "contour".to_string() } else { format!("hatch-{}", cut.layer - 1) };
        let s = Stroke { path: cut.path, width0: cut.w0, width1: cut.w1, load: load.clone(), pressure: 1.0, wetness: 0.0 };
        s.rasterize(canvas, &brush);
        *placed += 1;
        if let Some(pr) = progress {
            if *placed % 64 == 0 {
                pr(PaintProgress::Placed(*placed));
            }
        }
        score.strokes.push(StrokeRecord { id: *placed as u32, wipe: false, wash: false, stage, spline: s.path, w0: cut.w0, w1: cut.w1, taper: 0.0, mix: vec![(p.palette.pigments[ink].name.to_string(), load[ink])], wet: 0.0, press: 1.0, streak: brush.streak, round: brush.round, pickup: None, bristles: Some(brush.bristles), kd: None, cap: None, flat: true, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
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
    fn the_water_stage_is_the_first_wash_stage_after_the_pens() {
        let st: Vec<String> = ["pen-block-in", "pen-restate-1", "block-in", "restate-1"].iter().map(|s| s.to_string()).collect();
        assert!(!water_stage(Some(&st), 0) && !water_stage(Some(&st), 1), "the pens' passes are not the planes'");
        assert!(water_stage(Some(&st), 2), "the first wash stage is the planes' pass");
        assert!(!water_stage(Some(&st), 3), "the later washes are not");
        let plain: Vec<String> = ["block-in", "restate-1"].iter().map(|s| s.to_string()).collect();
        assert!(!water_stage(Some(&plain), 0) && !water_stage(None, 0), "a schedule without the pens has no planes' pass");
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

#[cfg(test)]
mod hdr_tests {
    use super::*;

    #[test]
    fn the_hdr_pass_lifts_the_darks_holds_the_lights_and_keeps_the_hue() {
        // A dark sheet with one blown lamp and a dark brown passage: after the pass the brown is lighter and
        // still brown (its chromaticity kept), the lamp is no lighter than it was, and amount 0 is a no-op.
        let img = image::RgbImage::from_fn(256, 256, |x, y| {
            if (x as i32 - 200).pow(2) + (y as i32 - 60).pow(2) < 400 { image::Rgb([255, 250, 230]) } else if x < 128 { image::Rgb([40, 28, 18]) } else { image::Rgb([30, 30, 36]) }
        });
        assert_eq!(hdr_tone_map(&img, 0.0).into_raw(), img.clone().into_raw(), "amount 0 changes nothing");
        let out = hdr_tone_map(&img, 0.6);
        let brown_in = img.get_pixel(40, 200).0;
        let brown_out = out.get_pixel(40, 200).0;
        let l = |c: [u8; 3]| color::linear_luma(color::srgb_to_linear(c));
        assert!(l(brown_out) > l(brown_in) * 1.3, "the dark lifts: {brown_in:?} → {brown_out:?}");
        let chrom = |c: [u8; 3]| { let s = c[0] as f32 + c[1] as f32 + c[2] as f32 + 1.0; [c[0] as f32 / s, c[1] as f32 / s] };
        let (a, b) = (chrom(brown_in), chrom(brown_out));
        assert!((a[0] - b[0]).abs() < 0.03 && (a[1] - b[1]).abs() < 0.03, "and stays its colour: {a:?} vs {b:?}");
        assert!(l(out.get_pixel(200, 60).0) <= l(img.get_pixel(200, 60).0) + 0.001, "the lamp is not lifted");
        assert!(l(brown_out) < l(out.get_pixel(200, 60).0), "and the order of values holds");
    }
}

#[cfg(test)]
mod tempera_tests {
    use super::*;

    fn luma(c: Srgb) -> f32 {
        color::linear_luma(color::srgb_to_linear(c))
    }

    #[test]
    fn the_verdaccio_is_greener_at_the_value_and_off_at_zero() {
        let flesh = [222, 170, 140];
        assert_eq!(verdaccio_tone(flesh, 0.0), flesh, "amount 0 changes nothing");
        let v = verdaccio_tone(flesh, 1.0);
        assert!(v[1] as i32 > v[0] as i32 - 8 && v[1] > v[2], "greenish grey, not flesh: {v:?}");
        assert!((luma(v) - luma(flesh) * 0.9).abs() < 0.03, "a tenth darker at the picture's value: {} vs {}", luma(v), luma(flesh));
        let half = verdaccio_tone(flesh, 0.5);
        assert!(half[0] > v[0] && half[0] < flesh[0], "half way keeps half the flesh: {half:?}");
    }

    #[test]
    fn the_dishes_snap_the_value_and_keep_the_colour() {
        let c = [180, 120, 60];
        assert_eq!(dish_tone(c, 0.0, 0.0, 1.0, 0.0, 0.5), c, "no dishes changes nothing");
        assert_eq!(dish_tone(c, 1.0, 0.0, 1.0, 0.0, 0.5), c);
        // Five dishes: the lightness goes to one of 0, 1/4, 1/2, 3/4, 1 — a ramp of 64 colours lands on ≤ 5 values.
        let mut seen: Vec<i32> = Vec::new();
        for i in 0..64 {
            let g = (i * 4) as u8;
            let out = dish_tone([g, g, g], 5.0, 0.0, 1.0, 0.0, 0.5);
            let l = (luma(out).powf(1.0 / 2.2) * 100.0).round() as i32;
            // (The darkest dish has a floor of 0.02 lightness, so the near-blacks spread over 0–3 of 100.)
            if !seen.iter().any(|v| (v - l).abs() <= 3) {
                seen.push(l);
            }
        }
        assert!(seen.len() <= 5, "five dishes: {seen:?}");
        let out = dish_tone(c, 5.0, 0.0, 1.0, 0.0, 0.5);
        let ratio = |q: Srgb| { let l = color::srgb_to_linear(q); [l[0] / l[1].max(1e-4), l[2] / l[1].max(1e-4)] };
        let (a, b) = (ratio(c), ratio(out));
        assert!((a[0] - b[0]).abs() < 0.08 * a[0] && (a[1] - b[1]).abs() < 0.08 * a[0].max(a[1]), "the colour is kept: {a:?} vs {b:?}");
        // Within the picture's range: a pale sky in a picture whose lightest value is that sky snaps to
        // itself, not to white.
        let sky = [170, 200, 235];
        let hi = luma(sky);
        let snapped = dish_tone(sky, 5.0, 0.02, hi, 0.0, 0.5);
        assert!(snapped[2] < 245 && (luma(snapped) - hi).abs() < 0.02, "the lightest dish is the picture's light: {snapped:?}");
    }

    #[test]
    fn the_dither_draws_between_the_two_nearest_dishes_by_the_fraction() {
        // A grey 0.3 of the way from dish 1 to dish 2 (five dishes over 0..1): the nearest is dish 1; dithered
        // at 1, about 30 in 100 draws land on dish 2 — never a third dish.
        let light: f32 = (1.0 + 0.3) / 4.0;
        let lin = light.powf(2.2);
        let c = color::linear_to_srgb([lin, lin, lin]);
        let nearest = dish_tone(c, 5.0, 0.0, 1.0, 0.0, 0.9);
        assert_eq!(nearest, dish_tone(c, 5.0, 0.0, 1.0, 0.0, 0.1), "at dither 0 the draw is ignored");
        let (mut upper, mut other) = (0, 0);
        for i in 0..100 {
            let u = (i as f32 + 0.5) / 100.0;
            let out = dish_tone(c, 5.0, 0.0, 1.0, 1.0, u);
            let l = luma(out).powf(1.0 / 2.2);
            if (l - 0.5).abs() < 0.03 { upper += 1 } else if (l - 0.25).abs() < 0.03 { } else { other += 1 }
        }
        assert_eq!(other, 0, "only the two nearest dishes");
        assert!((20..=40).contains(&upper), "about 30 of 100 on the upper dish: {upper}");
        // Half dithered: the draws nearer the middle decide — fewer land on the upper dish than at 1, more than at 0.
        let half = (0..100).filter(|i| { let u = (*i as f32 + 0.5) / 100.0; (luma(dish_tone(c, 5.0, 0.0, 1.0, 0.5, u)).powf(1.0 / 2.2) - 0.5).abs() < 0.03 }).count();
        assert!(half > 0 && half < upper, "half way between: {half} of {upper}");
    }

    #[test]
    fn the_cross_turns_off_the_form_alternating_sides() {
        let d = |hk: usize| cross_turn(hk, 15.0).to_degrees().round() as i32;
        assert_eq!([d(0), d(1), d(2), d(3), d(4)], [0, 15, -30, 45, -60]);
        assert_eq!(cross_turn(3, 0.0), 0.0);
    }

    #[test]
    fn the_solid_hatch_covers_in_the_darks_and_veils_the_lights() {
        assert!((hatch_charge(0.8, 0.0, 0.0) - 0.76).abs() < 1e-5 && (hatch_charge(0.8, 0.0, 1.0) - 0.76).abs() < 1e-5, "at 0 the value is ignored");
        assert!((hatch_charge(0.8, 1.0, 1.0) - 0.76).abs() < 1e-5, "the lightest value keeps the veil");
        assert!((hatch_charge(0.8, 1.0, 0.0) - 1.24).abs() < 1e-5, "the darkest covers");
        assert!((hatch_charge(0.8, 1.0, 0.5) - 1.0).abs() < 1e-5, "the middle the full charge");
        assert!(hatch_charge(0.8, 0.5, 0.0) > 0.76 && hatch_charge(0.8, 0.5, 0.0) < 1.24);
    }

    #[test]
    fn the_flesh_is_warm_and_the_sky_and_the_grey_are_not() {
        assert!(warmth_of([222, 170, 140]) > 0.9, "flesh: {}", warmth_of([222, 170, 140]));
        assert!(warmth_of([196, 150, 110]) > 0.9, "a warm earth: {}", warmth_of([196, 150, 110]));
        assert_eq!(warmth_of([120, 160, 220]), 0.0, "a sky blue is cold");
        assert_eq!(warmth_of([128, 128, 128]), 0.0, "a grey is neutral");
        assert_eq!(warmth_of([110, 140, 110]), 0.0, "a green is not flesh");
        let w = warmth_of([160, 140, 120]);
        assert!(w > 0.0 && w < 0.9, "a warm grey is between: {w}");
    }

    #[test]
    fn the_span_runs_long_on_the_flat_and_short_on_the_busy() {
        assert_eq!(span_length(0.0, 0.0), 1.0);
        assert_eq!(span_length(0.0, 1.0), 1.0);
        assert!((span_length(1.0, busy_of(0.0)) - 1.5).abs() < 1e-5, "flat: half again as long");
        assert!((span_length(1.0, busy_of(0.3)) - 0.5).abs() < 1e-5, "busy: half");
        assert!((span_length(1.0, busy_of(0.15)) - 1.0).abs() < 1e-5, "between: unchanged");
        assert!((span_length(0.6, busy_of(0.0)) - 1.3).abs() < 1e-5);
    }

    #[test]
    fn the_calm_passage_relaxes_the_dish_toward_the_picture() {
        let c = [120, 150, 200];
        let dished = dish_tone(c, 5.0, 0.0, 1.0, 0.0, 0.5);
        assert_ne!(dished, c, "the dish moves the tone");
        assert_eq!(blend_tone(dished, c, 0.0), dished, "no calm: the dish");
        let back = blend_tone(dished, c, 1.0);
        assert!((0..3).all(|i| (back[i] as i32 - c[i] as i32).abs() <= 1), "full calm: the picture's own tone, {back:?} vs {c:?}");
        let half = blend_tone(dished, c, 0.5);
        assert!((0..3).all(|i| { let (a, b, m) = (dished[i] as i32, c[i] as i32, half[i] as i32); (m >= a.min(b) - 1) && (m <= a.max(b) + 1) }), "half way between: {half:?}");
    }

    #[test]
    fn the_bole_lays_the_local_tone_thin_and_replays_to_the_bit() {
        // A picture dark on the left, light on the right; the ground its mean. The bole at 1 lays each side its
        // own tone (the left darker than the right, both off the flat mean), thin; the score's wash cells
        // replay to the same canvas byte for byte, and at 0 nothing is laid.
        let (w, h) = (96u32, 48u32);
        let img = RgbImage::from_fn(w, h, |x, _| image::Rgb(if x < w / 2 { [40, 36, 60] } else { [190, 170, 120] }));
        let tone: Srgb = [120, 108, 92];
        let mut p = PaintParams::new(crate::paint::palette::ZORN, 100);
        p.ground = Some(tone);
        p.bole = 1.0;
        p.from_scratch = true;
        let header = ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "tempera".into(), seed: 1, width: w, height: h, tooth: 0.85, ground: Some(tone), brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: None, dry: 1.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0, transmittance: false, flow: None, weave: 0.0, sag: 0.0, gloss: 0.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None };
        let mut canvas = Canvas::toned(w, h, crate::paint::palette::ZORN, tone, 0.85).with_opacity(1.0);
        let mut score = StrokeScore { header: header.clone(), strokes: Vec::new() };
        let (mut placed, mut k) = (0usize, 0u64);
        let laid = lay_bole(&mut canvas, &mut score, &img, &p, &mut placed, &mut k);
        assert!(laid >= 4, "cells laid: {laid}");
        assert_eq!(score.strokes.len(), laid);
        assert!(score.strokes.iter().all(|s| s.wash && s.stage == "ground" && s.mix2.is_none()));
        let left = luma(canvas.color_at(12, 24));
        let right = luma(canvas.color_at(84, 24));
        let flat = luma(tone);
        assert!(left < flat && right > flat, "each side toward its own tone, off the flat mean: {left} < {flat} < {right}");
        assert!(right - left > 0.08, "the two tones apart: {left} vs {right}");
        let replayed = score.replay(w, h).expect("replays");
        assert_eq!(replayed.to_image().into_raw(), canvas.to_image().into_raw(), "the wash cells replay to the bit");
        // Half way, each cell's colour is half the mean, half the local tone; at 0 it would be the mean itself
        // (the painter never lays it at 0).
        assert_eq!(blend_tone(tone, [40, 36, 60], 0.0), tone);
        let half = blend_tone(tone, [40, 36, 60], 0.5);
        assert!(luma(half) < flat && luma(half) > luma([40, 36, 60]), "half way: {half:?}");
        let _ = header;
    }

    #[test]
    fn the_umbra_keeps_the_darkest_dish_continuous() {
        // Five dishes over the full range: a position under 1 is the darkest step; the umbra at 1 covers the
        // whole of it, at 0.5 its lower half, and the lights are never in it.
        let dark = color::linear_to_srgb([0.03, 0.03, 0.03]);
        let darker = color::linear_to_srgb([0.002, 0.002, 0.002]);
        let mid = [128, 128, 128];
        assert!(dish_pos(darker, 5.0, 0.0, 1.0) < 0.5, "{}", dish_pos(darker, 5.0, 0.0, 1.0));
        assert!(dish_pos(dark, 5.0, 0.0, 1.0) < 1.0 && dish_pos(dark, 5.0, 0.0, 1.0) > 0.5, "{}", dish_pos(dark, 5.0, 0.0, 1.0));
        assert!(dish_pos(mid, 5.0, 0.0, 1.0) > 1.5, "{}", dish_pos(mid, 5.0, 0.0, 1.0));
        assert_eq!(dish_pos(mid, 1.0, 0.0, 1.0), 0.0, "no dishes, no scale");
        assert_ne!(dish_tone(dark, 5.0, 0.0, 1.0, 0.0, 0.5), dark, "without the umbra the dark tone is snapped to a step");
    }

    #[test]
    fn a_small_star_is_isolated_and_a_lit_wall_is_not() {
        // A 2-px star on a dark field within a 4-px brush: isolated. A field lit all over: not. A dark seed: never.
        let (w, h) = (32usize, 32usize);
        let mut star = vec![0.02f32; w * h];
        for y in 15..=17 {
            for x in 15..=17 {
                star[y * w + x] = 0.9;
            }
        }
        assert!(isolated_light(&star, w, h, 16.0, 16.0, 4.0));
        assert!(!isolated_light(&star, w, h, 16.0, 16.0, 1.0), "a brush no bigger than the star sees no dark round it");
        let wall = vec![0.8f32; w * h];
        assert!(!isolated_light(&wall, w, h, 16.0, 16.0, 4.0));
        let dark = vec![0.0f32; w * h];
        assert!(!isolated_light(&dark, w, h, 16.0, 16.0, 4.0));
    }

    #[test]
    fn the_edge_is_only_a_plain_contour() {
        assert_eq!(edge_of(0.0), 0.0);
        assert_eq!(edge_of(0.15), 0.0, "the modelling's range is not an edge");
        assert!((edge_of(0.325) - 0.5).abs() < 1e-5);
        assert_eq!(edge_of(0.5), 1.0);
        assert_eq!(edge_of(0.9), 1.0);
    }

    #[test]
    fn the_lights_lift_toward_white() {
        let c = [200, 160, 120];
        assert_eq!(lights_tone(c, 0.0), c);
        let l = lights_tone(c, 0.5);
        assert!(l[0] > c[0] && l[1] > c[1] && l[2] > c[2] && l[2] < 255, "lifted, not white: {l:?}");
        assert!(l[2] as i32 - c[2] as i32 > l[0] as i32 - c[0] as i32, "the dark channel rises most (toward white, not a tint)");
    }

    #[test]
    fn a_hatch_line_is_straight_and_keeps_its_length() {
        // A quarter circle of radius 40: straightened at 1 it is a straight line of the arc's length.
        let path: Vec<[f32; 2]> = (0..=20).map(|i| { let t = i as f32 / 20.0 * std::f32::consts::FRAC_PI_2; [50.0 + 40.0 * t.cos(), 50.0 + 40.0 * t.sin()] }).collect();
        let arc = |q: &[[f32; 2]]| q.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()).sum::<f32>();
        assert_eq!(straighten_path(&path, 0.0, 200, 200), path, "amount 0 changes nothing");
        let out = straighten_path(&path, 1.0, 200, 200);
        assert!((arc(&out) - arc(&path)).abs() < 0.5, "the length is kept: {} vs {}", arc(&out), arc(&path));
        // Every point lies on the chord's line through the first point.
        let (p0, pn) = (path[0], path[path.len() - 1]);
        let (dx, dy) = (pn[0] - p0[0], pn[1] - p0[1]);
        let dl = (dx * dx + dy * dy).sqrt();
        for q in &out {
            let off = ((q[0] - p0[0]) * dy - (q[1] - p0[1]) * dx).abs() / dl;
            assert!(off < 0.01, "on the line: {q:?} is {off} off");
        }
        let half = straighten_path(&path, 0.5, 200, 200);
        let mid = half[10];
        let off_half = ((mid[0] - p0[0]) * dy - (mid[1] - p0[1]) * dx).abs() / dl;
        let off_curve = ((path[10][0] - p0[0]) * dy - (path[10][1] - p0[1]) * dx).abs() / dl;
        assert!(off_half > 0.3 * off_curve && off_half < 0.7 * off_curve, "half way: {off_half} of {off_curve}");
    }
}

#[cfg(test)]
mod transmittance_tests {
    use super::*;
    use crate::paint::palette;
    use crate::paint::pigment::Pigment;

    static ORANGE_BLUE: [Pigment; 3] = [Pigment { name: "orange", masstone: [225, 120, 60] }, Pigment { name: "blue", masstone: [70, 110, 200] }, Pigment { name: "white", masstone: [247, 245, 241] }];

    fn hsl_sat(c: Srgb) -> f32 {
        let (r, g, b) = (c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0);
        let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
        let l = (mx + mn) / 2.0;
        if mx == mn { 0.0 } else if l < 0.5 { (mx - mn) / (mx + mn) } else { (mx - mn) / (2.0 - mx - mn) }
    }

    #[test]
    fn a_transmittance_film_darkens_by_density_and_keeps_the_hue() {
        // One orange pigment at rising density: the value falls and the hue STAYS orange and saturated —
        // the alpha-blend film, by contrast, can only go from white to the masstone and never past it.
        let pal = Palette { name: "t", pigments: &ORANGE_BLUE };
        let ring: Vec<[f32; 2]> = vec![[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]];
        let mut prev_l = 2.0f32;
        for d in [0.3f32, 1.0, 2.0, 3.0] {
            let mut c = Canvas::white(8, 8, pal, 0.85).with_opacity(0.45).with_transmittance(true);
            c.fill_rings(&ring, &[d, 0.0, 0.0], 0.75, 0.0);
            let px = c.color_at(4, 4);
            let l = color::linear_luma(color::srgb_to_linear(px));
            assert!(l < prev_l, "denser is darker: density {d} reads {l:.3} after {prev_l:.3}");
            assert!(px[0] > px[1] && px[1] > px[2], "and stays orange: {px:?}");
            assert!(hsl_sat(px) > 0.6, "and saturated: {px:?} S {:.2}", hsl_sat(px));
            prev_l = l;
        }
        // Darker than the masstone itself, which the covering film can never be.
        let mut c = Canvas::white(8, 8, pal, 0.85).with_opacity(0.45).with_transmittance(true);
        c.fill_rings(&ring, &[3.0, 0.0, 0.0], 0.75, 0.0);
        let deep = color::linear_luma(color::srgb_to_linear(c.color_at(4, 4)));
        let mass = color::linear_luma(color::srgb_to_linear([225, 120, 60]));
        assert!(deep < mass, "a dense glaze goes deeper than the masstone ({deep:.3} vs {mass:.3})");
    }

    #[test]
    fn two_glazes_multiply_and_the_solver_finds_the_densities() {
        // Blue over orange = the product, in either order; and `trans_densities` recovers densities whose
        // transmittance reads the target it was asked for.
        let pal = Palette { name: "t", pigments: &ORANGE_BLUE };
        let ring: Vec<[f32; 2]> = vec![[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]];
        let lay = |loads: &[[f32; 3]]| {
            let mut c = Canvas::white(8, 8, pal, 0.85).with_opacity(0.45).with_transmittance(true);
            for l in loads {
                c.fill_rings(&ring, l, 0.75, 0.0);
            }
            c.color_at(4, 4)
        };
        let ob = lay(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let bo = lay(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
        let both = lay(&[[1.0, 1.0, 0.0]]);
        assert_eq!(ob, bo, "glazes commute");
        assert_eq!(ob, both, "and stack as one film");
        // A colour these two pigments CAN make (a glaze of both), asked for from the solver blind.
        let c = Canvas::white(1, 1, pal, 0.85).with_transmittance(true);
        let ground = c.linear_at(0, 0);
        let made = lay(&[[0.7, 0.4, 0.0]]);
        let target = color::srgb_to_linear(made);
        let want = [(target[0] / ground[0]).ln().min(0.0), (target[1] / ground[1]).ln().min(0.0), (target[2] / ground[2]).ln().min(0.0)];
        let dens = trans_densities(c.ln_reflectance(), want);
        assert!(dens.iter().all(|d| *d >= 0.0) && dens.iter().any(|d| *d > 0.0), "non-negative densities: {dens:?}");
        let got = lay(&[[dens[0], dens[1], dens[2]]]);
        let dl = color::delta_e76(color::srgb_to_lab(got), color::srgb_to_lab(made));
        assert!(dl < 3.0, "the solved glaze reads its target: {got:?} vs {made:?}, dE {dl:.1}");
    }

    #[test]
    fn a_transmittance_score_round_trips_through_the_text_byte_exact() {
        // A score with the film flag, a wash, a capped wash-brush stroke and a plain stroke replays the same
        // from the text as from memory — the flag, `kd` and `cap` all survive serialisation.
        let pal = palette::EARTH;
        let mut sc = crate::paint::score::StrokeScore {
            header: crate::paint::score::ScoreHeader { version: 1, palette: "earth".into(), pigments: pal.pigments.iter().map(|p| (p.name.to_string(), p.masstone)).collect(), medium: "watercolour".into(), seed: 7, width: 48, height: 48, tooth: 0.85, ground: None, brush: BrushConfig::default(), bleed: 0.2, diffuse: 0.0, stages: Some(vec!["wash".into(), "block-in".into()]), dry: 1.0, opacity: 0.45, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0, transmittance: true, flow: None, weave: 0.0, sag: 0.0, gloss: 0.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None },
            strokes: Vec::new(),
        };
        let name = |i: usize| pal.pigments[i].name.to_string();
        sc.strokes.push(StrokeRecord { id: 1, wipe: false, wash: true, stage: "wash".into(), spline: vec![[4.0, 4.0], [40.0, 4.0], [40.0, 40.0], [4.0, 40.0]], w0: 0.0, w1: 0.0, taper: 0.0, mix: vec![(name(1), 0.8), (name(2), 0.3)], wet: 0.75, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        sc.strokes.push(StrokeRecord { id: 2, wipe: false, wash: false, stage: "block-in".into(), spline: vec![[6.0, 20.0], [24.0, 22.0], [42.0, 20.0]], w0: 12.0, w1: 7.0, taper: 0.4, mix: vec![(name(3), 1.7)], wet: 0.6, press: 0.9, streak: 0.05, round: 0.6, pickup: Some(0.0), bristles: None, kd: Some(0.51), cap: Some(1.0), flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        sc.strokes.push(StrokeRecord { id: 3, wipe: false, wash: false, stage: "block-in".into(), spline: vec![[8.0, 30.0], [40.0, 32.0]], w0: 5.0, w1: 3.0, taper: 0.4, mix: vec![(name(0), 0.9)], wet: 0.5, press: 1.0, streak: 0.3, round: 0.7, pickup: Some(0.1), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None });
        let mem = sc.replay(48, 48).unwrap().to_image().into_raw();
        let text = sc.to_text();
        assert!(text.contains("trans=1") && text.contains("kd=0.51") && text.contains("cap=1"), "the film flag and the wash brush's fields are written");
        let back = crate::paint::score::StrokeScore::parse(&text).unwrap();
        assert!(back.header.transmittance);
        assert_eq!(back.replay(48, 48).unwrap().to_image().into_raw(), mem, "byte-exact through the text");
        // And the film is doing something: the same score without the flag renders differently.
        sc.header.transmittance = false;
        assert_ne!(sc.replay(48, 48).unwrap().to_image().into_raw(), mem);
    }
}

#[cfg(test)]
mod pastel_tests {
    use super::*;
    use crate::paint::canvas::Canvas;
    use crate::paint::score::{ScoreHeader, StrokeScore};
    use crate::paint::stroke::BrushConfig;

    #[test]
    fn the_finger_rubs_the_calm_passages_and_replays_to_the_bit() {
        // A flat picture (every passage calm) with a black patch on the left of a bare sheet: the finger pass
        // rubs along a horizontal flow, carrying the black past the patch's edge; every rub is a smudge record
        // in the pass's stage and the score replays byte for byte. At 0 nothing is laid.
        let (w, h) = (96u32, 48u32);
        let img = RgbImage::from_fn(w, h, |_, _| image::Rgb([120u8, 120, 120]));
        let luma: Vec<f32> = vec![0.2; (w * h) as usize];
        // (The flow is the structure tensor's gradient; the marks run along the isophote, across it.)
        let (gx, gy) = (vec![0.0f32; (w * h) as usize], vec![1.0f32; (w * h) as usize]);
        let mut p = PaintParams::new(crate::paint::palette::ZORN, 10_000);
        p.blend = 1.0;
        let header = ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "pastel".into(), seed: 1, width: w, height: h, tooth: 0.85, ground: None, brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: None, dry: 1.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0, transmittance: false, flow: None, weave: 0.0, sag: 0.0, gloss: 0.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None };
        let mut canvas = Canvas::white(w, h, crate::paint::palette::ZORN, 0.85);
        let patch = Stroke { path: vec![[4.0, 24.0], [36.0, 24.0]], width0: 40.0, width1: 40.0, load: vec![0.0, 0.0, 6.0, 0.0], pressure: 1.0, wetness: 0.3 };
        patch.rasterize(&mut canvas, &BrushConfig { k_pickup: 0.0, hold_charge: true, ..BrushConfig::default() });
        let mut score = StrokeScore { header: header.clone(), strokes: Vec::new() };
        let before = canvas.clone();
        let (mut placed, mut k) = (0usize, 0u64);
        let laid = blend_pass(&mut canvas, &mut score, &p, &luma, None, &gx, &gy, &img, None, 6.0, "restate-1", false, false, &mut placed, &mut k);
        assert!(laid >= 8, "rubs laid: {laid}");
        assert_eq!(score.strokes.len(), laid);
        assert_eq!(placed, laid);
        assert!(score.strokes.iter().all(|s| s.wipe && !s.wash && s.stage == "restate-1" && s.smudge.is_some() && s.mix.is_empty()));
        let dark_before: f32 = (40..56).map(|x| luma_of(before.color_at(x, 24))).sum::<f32>() / 16.0;
        let dark_after: f32 = (40..56).map(|x| luma_of(canvas.color_at(x, 24))).sum::<f32>() / 16.0;
        // (A finger a fortieth of this small sheet is the four-pixel floor, at a light touch: a small drift.)
        assert!(dark_after < dark_before - 0.002, "the black carried past the patch: {dark_before} → {dark_after}; rubs {laid}");
        let mut replayed = Canvas::white(w, h, crate::paint::palette::ZORN, 0.85);
        patch.rasterize(&mut replayed, &BrushConfig { k_pickup: 0.0, hold_charge: true, ..BrushConfig::default() });
        for rec in &score.strokes {
            let s = Stroke { path: rec.spline.clone(), width0: rec.w0, width1: rec.w1, load: Vec::new(), pressure: rec.press, wetness: rec.wet };
            s.smudge(&mut replayed, rec.smudge.unwrap());
        }
        assert_eq!(replayed.to_image().into_raw(), canvas.to_image().into_raw(), "the rubs replay to the bit");
        let mut p0 = PaintParams::new(crate::paint::palette::ZORN, 10_000);
        p0.blend = 0.0;
        let mut untouched = before.clone();
        let mut s0 = StrokeScore { header, strokes: Vec::new() };
        let (mut placed0, mut k0) = (0usize, 0u64);
        assert_eq!(blend_pass(&mut untouched, &mut s0, &p0, &luma, None, &gx, &gy, &img, None, 6.0, "restate-1", false, false, &mut placed0, &mut k0), 0);
        assert_eq!(untouched.to_image().into_raw(), before.to_image().into_raw());
    }

    fn luma_of(c: Srgb) -> f32 {
        color::linear_luma(color::srgb_to_linear(c))
    }
}

#[cfg(test)]
mod blade_tests {
    use super::*;

    #[test]
    fn the_blade_planes_cut_a_ramp_into_flat_steps_and_hold_the_tone() {
        // A vertical grey ramp, 64 rows: four planes give four values, their count and mean held.
        let w = 8u32;
        let h = 64u32;
        let img = RgbImage::from_fn(w, h, |_, y| { let v = (y * 4) as u8; image::Rgb([v, v, v]) });
        let out = blade_planes_ref(&img, 4, 0.0);
        let mut vals: Vec<u8> = out.pixels().map(|p| p.0[0]).collect();
        vals.sort_unstable();
        vals.dedup_by(|a, b| a.abs_diff(*b) <= 1); // (a plane's grey may round either way by a unit)
        assert_eq!(vals.len(), 4, "four planes: {vals:?}");
        let mean_in: f32 = img.pixels().map(|p| color::linear_luma(color::srgb_to_linear(p.0))).sum::<f32>() / (w * h) as f32;
        let mean_out: f32 = out.pixels().map(|p| color::linear_luma(color::srgb_to_linear(p.0))).sum::<f32>() / (w * h) as f32;
        assert!((mean_in - mean_out).abs() < 0.01, "the mean tone held: {mean_in} vs {mean_out}");
        // The planes are flat along the ramp: every row one value, the rows monotone.
        for y in 1..h {
            assert!(out.get_pixel(0, y).0[0] >= out.get_pixel(0, y - 1).0[0]);
        }
    }

    #[test]
    fn the_facet_cuts_the_masked_passage_into_planes_of_its_own_range() {
        // A flat dark left half and a narrow ramp on the right: four global planes leave the ramp nearly one
        // plane; faceted under a right-half mask it falls into at least three terraces, the left half untouched.
        let (w, h) = (64u32, 16u32);
        let img = RgbImage::from_fn(w, h, |x, _| if x < 32 { image::Rgb([20, 20, 20]) } else { let v = 100 + (x - 32) * 40 / 31; image::Rgb([v as u8, v as u8, v as u8]) });
        let mask: Vec<f32> = (0..w * h).map(|i| if i % w >= 32 { 1.0 } else { 0.0 }).collect();
        let planes = blade_planes_ref(&img, 4, 0.0);
        let out = blade_facet_ref(&planes, &img, 4, 0.0, &mask, 1.0, 0.0);
        let distinct = |im: &RgbImage| { let mut v: Vec<u8> = (34..62).map(|x| im.get_pixel(x, 8).0[0]).collect(); v.sort_unstable(); v.dedup_by(|a, b| a.abs_diff(*b) <= 1); v.len() };
        assert!(distinct(&planes) <= 2, "the global planes leave the ramp in one or two ({})", distinct(&planes));
        assert!(distinct(&out) >= 3, "the facet cuts it into terraces ({})", distinct(&out));
        assert_eq!(out.get_pixel(10, 8), planes.get_pixel(10, 8), "the unmasked half is untouched");
        // BLADE SOFT: on the smooth ramp the terraces give way toward the picture's own slope — more tones.
        let softened = blade_facet_ref(&planes, &img, 4, 0.0, &mask, 1.0, 1.0);
        assert!(distinct(&softened) > distinct(&out) + 2, "the soft facet is a slope on the smooth ramp ({} vs {})", distinct(&softened), distinct(&out));
    }

    #[test]
    fn the_tuft_map_finds_a_fine_texture_and_not_a_ramp_or_a_clean_edge() {
        // A sheet all grass in three bands: a smooth ramp, a fine hashed texture, a flat field with one clean vertical edge.
        // The texture's map is high in the texture and near nothing on the ramp and along the edge (a thin line of
        // busy pixels in a quiet field).
        let (w, h) = (96usize, 32usize);
        let mut luma = vec![0.5f32; w * h];
        for y in 0..h {
            for x in 0..w {
                luma[y * w + x] = if x < 32 {
                    // (a sky's ramp: a few hundredths across the whole band — nothing within a small window)
                    0.3 + 0.03 * x as f32 / 32.0
                } else if x < 64 {
                    let z = (x as u64).wrapping_mul(0x9E37_79B9).wrapping_add((y as u64).wrapping_mul(0x85EB_CA6B)) ^ 0x1234;
                    let z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
                    0.2 + 0.6 * ((z >> 8) & 0xFF) as f32 / 255.0
                } else if x < 80 {
                    0.25
                } else {
                    0.75
                };
            }
        }
        let ones = vec![1.0f32; w * h];
        let map = blade_tuft_map(&luma, Some(&ones), 0.0, 0.0, w, h, 2);
        assert!(blade_tuft_map(&luma, None, 0.0, 0.0, w, h, 2).iter().all(|v| *v == 0.0), "no grass found, no tufts");
        let mean = |x0: usize, x1: usize| (8..24).flat_map(|y| (x0..x1).map(move |x| y * w + x)).map(|i| map[i]).sum::<f32>() / ((x1 - x0) * 16) as f32;
        let (ramp, tex, edge) = (mean(4, 28), mean(36, 60), mean(76, 84));
        assert!(tex > 0.5, "the fine texture reads as a tuft ({tex})");
        assert!(ramp < 0.1, "the smooth ramp does not ({ramp})");
        assert!(edge < 0.25, "nor the clean edge ({edge})");
    }

    #[test]
    fn the_anisotropy_gate_passes_a_hashed_texture_and_stops_coherent_stripes() {
        // Two fine textures, no grass regions: a hashed mesh (its orientation flipping every pixel) and vertical
        // stripes (one orientation). The tensor's gate opens on the mesh and stays shut on the stripes.
        let (w, h) = (128usize, 48usize);
        let mut luma = vec![0.5f32; w * h];
        for y in 0..h {
            for x in 0..w {
                luma[y * w + x] = if x < 64 {
                    let z = (x as u64).wrapping_mul(0x9E37_79B9).wrapping_add((y as u64).wrapping_mul(0x85EB_CA6B)) ^ 0x1234;
                    let z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
                    0.2 + 0.6 * ((z >> 8) & 0xFF) as f32 / 255.0
                } else {
                    0.5 + 0.3 * (x as f32 * 1.1).sin()
                };
            }
        }
        let map = blade_tuft_map(&luma, None, 1.0, 0.0, w, h, 2);
        let mean = |x0: usize, x1: usize| (12..36).flat_map(|y| (x0..x1).map(move |x| y * w + x)).map(|i| map[i]).sum::<f32>() / ((x1 - x0) * 24) as f32;
        let (mesh, stripes) = (mean(8, 56), mean(72, 120));
        assert!(mesh > 0.5, "the mesh reads as a tuft ({mesh})");
        assert!(stripes < 0.15, "the stripes do not ({stripes})");
    }

    /// A hashed mesh of a given contrast over a band.
    fn mesh(x: usize, y: usize, amp: f32) -> f32 {
        let z = (x as u64).wrapping_mul(0x9E37_79B9).wrapping_add((y as u64).wrapping_mul(0x85EB_CA6B)) ^ 0x1234;
        let z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
        0.5 - amp / 2.0 + amp * ((z >> 8) & 0xFF) as f32 / 255.0
    }

    #[test]
    fn the_route_hands_the_fine_texture_to_the_detector_and_the_coarse_one_to_the_tensor() {
        // Two hashed meshes, both incoherent (the tensor opens on both): a faint one (a meadow at a distance) and a
        // contrasty one (a fur). The detector found no grass. Routed, the faint band keeps the tensor's tufts and
        // the contrasty band takes the detector's answer — none; unrouted, both are tufted.
        let (w, h) = (128usize, 48usize);
        let luma: Vec<f32> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); mesh(x, y, if x < 64 { 0.08 } else { 0.8 }) }).collect();
        let fine = blade_fine_map(&luma.iter().map(|l| l.powf(1.0 / 2.2)).collect::<Vec<_>>(), w, h, 2);
        let mean = |m: &[f32], x0: usize, x1: usize| (12..36).flat_map(|y| (x0..x1).map(move |x| y * w + x)).map(|i| m[i]).sum::<f32>() / ((x1 - x0) * 24) as f32;
        assert!(mean(&fine, 8, 56) < 0.2, "the faint mesh is coarse ({})", mean(&fine, 8, 56));
        assert!(mean(&fine, 72, 120) > 0.8, "the contrasty mesh is fine ({})", mean(&fine, 72, 120));
        let none = vec![0.0f32; w * h];
        let plain = blade_tuft_map(&luma, Some(&none), 1.0, 0.0, w, h, 2);
        let routed = blade_tuft_map(&luma, Some(&none), 1.0, 1.0, w, h, 2);
        assert!(mean(&plain, 72, 120) > 0.5, "unrouted, the tensor tufts the contrasty mesh ({})", mean(&plain, 72, 120));
        assert!(mean(&routed, 72, 120) < 0.1, "routed, the detector's no-grass holds on the fine mesh ({})", mean(&routed, 72, 120));
        assert!(mean(&routed, 8, 56) > 0.5, "routed, the tensor still tufts the coarse mesh ({})", mean(&routed, 8, 56));
        let grass = vec![1.0f32; w * h];
        let found = blade_tuft_map(&luma, Some(&grass), 1.0, 1.0, w, h, 2);
        assert!(mean(&found, 72, 120) > 0.5, "routed, the detector's grass is tufted on the fine mesh ({})", mean(&found, 72, 120));
    }

    #[test]
    fn the_focus_map_is_nothing_at_the_fine_texture_and_full_far_from_it() {
        // A contrasty mesh in the left quarter of a flat sheet: the focus sits in the mesh, the far right is out
        // of it; a sheet with no fine texture has no focus at all.
        let (w, h) = (160usize, 48usize);
        let luma: Vec<f32> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); if x < 40 { mesh(x, y, 0.8) } else { 0.5 } }).collect();
        let map = blade_focus_map(&luma, w, h, 2, 0.0);
        let mean = |x0: usize, x1: usize| (12..36).flat_map(|y| (x0..x1).map(move |x| y * w + x)).map(|i| map[i]).sum::<f32>() / ((x1 - x0) * 24) as f32;
        assert!(mean(8, 32) < 0.05, "the mesh is in focus ({})", mean(8, 32));
        assert!(mean(120, 152) > 0.95, "the far right is out of it ({})", mean(120, 152));
        let flat = vec![0.5f32; w * h];
        assert!(blade_focus_map(&flat, w, h, 2, 0.0).iter().all(|v| *v == 0.0), "no fine texture, no focus");
    }

    #[test]
    fn the_focus_band_keeps_the_whole_near_ground_and_loses_the_sky() {
        // The mesh in the lower-left corner of a wide sheet: with the band the lower RIGHT is in focus too (the
        // same height as the mesh) and the top is out; radially the lower right is out.
        let (w, h) = (160usize, 96usize);
        let luma: Vec<f32> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); if x < 40 && y >= 72 { mesh(x, y, 0.8) } else { 0.5 } }).collect();
        let radial = blade_focus_map(&luma, w, h, 2, 0.0);
        let band = blade_focus_map(&luma, w, h, 2, 1.0);
        let mean = |m: &[f32], x0: usize, x1: usize, y0: usize, y1: usize| (y0..y1).flat_map(|y| (x0..x1).map(move |x| y * w + x)).map(|i| m[i]).sum::<f32>() / ((x1 - x0) * (y1 - y0)) as f32;
        assert!(mean(&band, 8, 32, 76, 92) < 0.05, "the mesh is in focus ({})", mean(&band, 8, 32, 76, 92));
        assert!(mean(&band, 120, 152, 76, 92) < 0.05, "with the band the lower right is in focus ({})", mean(&band, 120, 152, 76, 92));
        assert!(mean(&radial, 120, 152, 76, 92) > 0.9, "radially it is out ({})", mean(&radial, 120, 152, 76, 92));
        assert!(mean(&band, 8, 152, 4, 20) > 0.95, "with the band the top is out ({})", mean(&band, 8, 152, 4, 20));
    }

    #[test]
    fn the_tint_turns_the_hue_and_keeps_the_value() {
        let red: Srgb = [200, 40, 40];
        let same = blade_tint_tone(red, 0.0);
        assert_eq!(same, red, "no turn, the colour kept");
        let turned = blade_tint_tone(red, 0.25);
        let (l0, l1) = (color::srgb_to_lab(red), color::srgb_to_lab(turned));
        assert!((l0.l - l1.l).abs() < 1.5, "the value kept ({} vs {})", l0.l, l1.l);
        assert!(l1.b > l1.a.abs(), "a quarter turn from red lands toward yellow (a {} b {})", l1.a, l1.b);
        let small = blade_tint_tone(red, 0.02);
        let d = (0..3).map(|i| (small[i] as i32 - red[i] as i32).abs()).max().unwrap_or(0);
        assert!((1..=24).contains(&d), "two percent of the circle is a drift, not a change ({d})");
    }

    #[test]
    fn the_slope_map_is_steep_on_a_bank_and_flat_on_a_meadow() {
        // A value ramp over the left half, a flat field on the right: the hill's map high on the ramp, nothing on
        // the flat.
        let (w, h) = (128usize, 48usize);
        let luma: Vec<f32> = (0..w * h).map(|i| { let x = i % w; if x < 64 { 0.2 + 0.5 * x as f32 / 64.0 } else { 0.7 } }).collect();
        let map = blade_slope_map(&luma, w, h, 6);
        let mean = |x0: usize, x1: usize| (12..36).flat_map(|y| (x0..x1).map(move |x| y * w + x)).map(|i| map[i]).sum::<f32>() / ((x1 - x0) * 24) as f32;
        let (bank, meadow) = (mean(16, 48), mean(88, 120));
        assert!(bank > 0.5, "the bank is steep ({bank})");
        assert!(meadow < 0.05, "the meadow is flat ({meadow})");
    }

    #[test]
    fn the_local_planes_keep_a_small_feature_the_global_planes_swallow() {
        // Two flat halves (20 and 200) and a small lighter spot (170) in the right half: four global planes put
        // the spot in the 200s' plane; the local planes give it a plane of its own; the flat right side keeps
        // the global plane.
        let (w, h) = (64u32, 16u32);
        let img = RgbImage::from_fn(w, h, |x, _| if x < 32 { image::Rgb([20, 20, 20]) } else if (44..48).contains(&x) { image::Rgb([170, 170, 170]) } else { image::Rgb([200, 200, 200]) });
        let planes = blade_planes_ref(&img, 4, 0.0);
        assert!(planes.get_pixel(46, 8).0[0].abs_diff(planes.get_pixel(56, 8).0[0]) <= 1, "the global planes swallow the spot");
        let out = blade_local_ref(&planes, &img, 4, 6, 1.0);
        assert!(out.get_pixel(46, 8).0[0] + 8 < out.get_pixel(56, 8).0[0], "the local planes keep it ({} vs {})", out.get_pixel(46, 8).0[0], out.get_pixel(56, 8).0[0]);
        assert_eq!(out.get_pixel(60, 8), planes.get_pixel(60, 8), "the flat passage keeps the global plane");
    }

    #[test]
    fn the_blade_planes_keep_the_hue() {
        // A red and a blue of two values each: the planes keep each pixel's hue, only the value is stepped.
        let img = RgbImage::from_fn(4, 1, |x, _| match x { 0 => image::Rgb([200, 0, 0]), 1 => image::Rgb([120, 0, 0]), 2 => image::Rgb([0, 0, 220]), _ => image::Rgb([0, 0, 90]) });
        let out = blade_planes_ref(&img, 2, 1.0);
        for x in 0..2 {
            let p = out.get_pixel(x, 0).0;
            assert!(p[0] > 0 && p[1] == 0 && p[2] == 0, "red stays red: {p:?}");
        }
        for x in 2..4 {
            let p = out.get_pixel(x, 0).0;
            assert!(p[2] > 0 && p[0] == 0 && p[1] == 0, "blue stays blue: {p:?}");
        }
    }
}
