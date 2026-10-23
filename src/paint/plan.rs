//! The PAINTING PLAN (RFC PAINT-1 §5/§12.2). The model/analysis stage acts as an ART DIRECTOR: it inspects a
//! provided or generated image and produces a *plan* — how a painter would approach it — and the deterministic,
//! weight-free stroke engine then executes that plan. No pixel of the deliverable comes from a model (G1); the
//! plan only decides STRUCTURE (focal region, per-region armature resolution, value key, reserve), never surface.
//!
//! This is the first, deterministic analyzer (face + measured contrast + medium). A model-driven art director
//! (VLM / low-res diffusion block-in) can later produce the same `PaintPlan` behind this interface unchanged.

use serde::Deserialize;

/// An artefact switch in a plan: `true`/`false`, or the path to write it to.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum Artefact {
    On(bool),
    Path(String),
}

/// A painting plan: the structural decisions the paint stage executes. Serialisable to HJSON so it is
/// inspectable and hand-editable — the art director's written plan.
#[derive(Debug, Clone, Deserialize)]
pub struct PaintPlan {
    /// The medium to paint in.
    #[serde(default = "default_medium")]
    pub medium: String,
    /// Palette name, or `image` to derive from the reference.
    #[serde(default = "default_palette")]
    pub palette: String,
    /// Fidelity register.
    #[serde(default = "default_style")]
    pub style: String,
    /// COARSE body/background armature resolution (px) — structure, not detail (RFC §1.1).
    #[serde(default = "default_armature")]
    pub armature: u32,
    /// FINE focal (face) armature resolution (px), or `null` for a uniform armature (RFC §5.2).
    #[serde(default)]
    pub armature_face: Option<u32>,
    /// MID subject-body armature resolution (px) — the three-tier plan: background coarsest, body mid, face fine.
    /// Needs a subject matte (U2Net), run automatically when set (RFC §5.2 multi-region).
    #[serde(default)]
    pub armature_body: Option<u32>,
    /// AERIAL PERSPECTIVE strength (0..1): veil the BACKGROUND (matte-derived) so the subject advances (§5.5).
    #[serde(default)]
    pub recede: f32,
    /// SEMANTIC tiers (RFC §5.2): detect parts (hair/beard) with OWL-ViT and give them their own armature tier —
    /// e.g. a coarse wash for the beard. Runs the detector when true.
    #[serde(default)]
    pub semantic: bool,
    /// FAMILY SEPARATION (RFC §3.3): partition light/shadow masses so the painting reads solid, not washed.
    #[serde(default)]
    pub families: bool,
    /// COMMIT SHADOWS (0..1, RFC §3.3): paint the dark value masses decisively (a solid value backbone).
    #[serde(default)]
    pub commit_shadows: f32,
    /// SILHOUETTE (0..1, RFC §5/§7): mark the subject boundary so a light subject reads by its edge.
    #[serde(default)]
    pub silhouette: f32,
    /// How the silhouette edge is marked: `line` / `colour` / `knife` / `lost` (RFC §7 edge craft).
    #[serde(default)]
    pub silhouette_mode: Option<String>,
    /// SAM precise masks (RFC §5): use MobileSAM for a precise subject/face mask (sharp silhouette, face-shaped focal).
    #[serde(default)]
    pub sam: bool,
    /// Value-key strength (tonal-range expansion) — higher for a flat, low-contrast reference.
    #[serde(default)]
    pub value_key: f32,
    /// Shadow floor (0..0.45): the value re-key lifts the SHADOW family into a narrow band above this value instead
    /// of stretching darks to black (RFC §3.3 — a shadow mass is a solid dark, never crushed).
    #[serde(default = "default_shadow_floor")]
    pub shadow_floor: f32,
    /// Detail-tier stroke-length multiplier (1.0 = the profile's own length; the analyzer picks ~0.3: short dabs
    /// make features read where long drags smear them — the wide block-in keeps its long covering strokes).
    /// `paint from` uses it when `--detail-length` is unset.
    #[serde(default = "default_detail_len")]
    pub detail_len: f32,
    /// Reserve threshold for surface-white media (paper whites), or `null` for the medium default.
    #[serde(default)]
    pub reserve: Option<f32>,
    /// Stroke budget, or `null` for the size-derived default.
    #[serde(default)]
    pub budget: Option<usize>,
    /// NEW PAINTING: paint from scratch (RFC §1.1) — a low-resolution armature, a per-plane minimum brush, a
    /// coverage budget, a toned ground under an opaque medium. `false` (default) = the path that tracks its source.
    #[serde(default, rename = "new")]
    pub from_scratch: bool,
    /// HAIR MASK: a grey PNG, white where hair / beard / fur is, for the strand tool to work over. Given, it
    /// REPLACES what the part detector found — the detector boxes only the hair it can name, so a long beard
    /// came out strands at the top and a smooth mass at its fall, and no automatic region has managed the
    /// whole of one yet. `plakat segment` and `plakat remove --what` both write masks of this shape.
    #[serde(default)]
    pub hair_mask: Option<String>,
    /// INFILL: what a stroke follows where the picture has no structure — `follow`, `flat`, or an angle in
    /// degrees. Absent = `follow` for a new painting, `flat` otherwise.
    #[serde(default)]
    pub infill: Option<String>,
    /// RIGGER (0..1): put back the few shapes too thin for the brush ladder. Absent = 0.35 for a new painting.
    #[serde(default)]
    pub rigger: Option<f32>,
    /// HOTSPOT (0..1): how far a flat blown specular highlight is re-modelled into a dome. Absent = 0.5 for a
    /// new painting, 0 otherwise.
    #[serde(default)]
    pub hotspot: Option<f32>,
    /// TECHNIQUE: `wet-on-wet`, `wet-on-dry` or `dry-on-dry` — how wet the paper is when each layer goes down.
    #[serde(default)]
    pub technique: Option<String>,
    /// SPLATTER (0..1): flicked drops off the brush. A watercolour tell; the plan can ask for more or none.
    #[serde(default)]
    pub splatter: Option<f32>,
    /// EDGE POOL (0..1): the darker pigment rim where a wash dried — the cauliflower edge.
    #[serde(default)]
    pub edge_pool: Option<f32>,
    /// OLD PAPER: print the finished picture on an aged sheet (any medium; an engraving gets its plate mark).
    #[serde(default)]
    pub oldpaper: Option<bool>,
    /// GRANULATE (0..1): pigment settling into the paper's tooth.
    #[serde(default)]
    pub granulate: Option<f32>,
    /// SALIENCY (0..1): the restating passes dense only in the focal, high-structure passages, thinned where
    /// the picture is flat and out of focus (`--saliency`).
    #[serde(default)]
    pub saliency: Option<f32>,
    /// FOCUS DETAIL (0..1): the crisp detail tier fired only inside the focal region, the rest loose
    /// (`--focus-detail`).
    #[serde(default)]
    pub focus_detail: Option<f32>,
    /// BRUSH LADDER: keep only this many of the COARSEST brushes. A watercolour is broad decisive washes and
    /// a few darks, not a fine brush going back over them — but how far to cut is a judgement about the
    /// picture (cut too hard and a lit face is left as bare reserved paper), so it lives here.
    #[serde(default)]
    pub ladder_keep: Option<usize>,
    /// LEAKS (0..1): runs of pigment dripping out of the wet washes. Nothing leaks unless asked.
    #[serde(default)]
    pub leak: Option<f32>,
    /// HDR (true/false): tone-map the picture before painting — lamps stop blowing out, shadows lift to show
    /// their detail — so every stage paints the re-lit picture. `hdr_amount` 0..1 is how far.
    #[serde(default)]
    pub hdr: Option<bool>,
    /// BRISTLE RIDGES (0..1): a stroke's relief striated across its width by the lanes' loads (the impasto
    /// relight then shows bristle marks, not a smooth tube), with the relief scaled against a robust
    /// height so one heavy crossing cannot flatten the sheet. 0 = off (the finish as before).
    #[serde(default)]
    pub ridges: Option<f32>,
    /// DRY-BRUSH SKIPPING (0..1): a bristle with little water touches only the paper's standing fibres, so
    /// a dry mark is broken by the tooth along its drag. 0 = off.
    #[serde(default)]
    pub skip: Option<f32>,
    /// FINE LINES (watercolour, 0..1): how far the finest brushes' marks may run along an edge — 1 = a
    /// rigger's line, 0 = short marks only (no drawn lines).
    #[serde(default)]
    pub fine_lines: Option<f32>,
    /// IMPASTO MAPPED TO THE PICTURE (0..1): lights thick / shadows thin, subject thick / background thin.
    #[serde(default)]
    pub impasto_map: Option<f32>,
    /// CANVAS WEAVE (0..1): the linen's threads under thin paint and bare ground, in the relight.
    #[serde(default)]
    pub weave: Option<f32>,
    /// SHEEN (0..1): the specular gloss on the relief's light-facing slopes (wet oil; 0 = matte).
    #[serde(default)]
    pub sheen: Option<f32>,
    /// WET COLLISION (0..1): after each broad pass the wet paint drags along its striation where loaded
    /// strokes meet — colours marble, ridges plow. Oil/acrylic. 0 = off.
    #[serde(default)]
    pub collide: Option<f32>,
    /// ANALYSIS artefact (RFC PAINT-3): `true` writes a Markdown report beside the output
    /// (`<out>.md`); a string is the path to write it to.
    #[serde(default)]
    pub analysis: Option<Artefact>,
    /// INSIGHTS (RFC PAINT-3 P0.5): `true` runs the analysis through the configured LLM (`auto`); a
    /// string names the provider (`ollama`, `ollama:<model>`, `deepseek`, `gemini`, `local`).
    #[serde(default)]
    pub analysis_insights: Option<Artefact>,
    /// OUTCOME sheet (RFC PAINT-3): `true` writes the one-page sheet beside the output (`<out>_sheet.png`);
    /// a string is the path (`.png` or `.pdf`).
    #[serde(default)]
    pub outcome: Option<Artefact>,
    /// The MARK dials, as a plan decides them: impasto thickness, stroke width and length multipliers, and
    /// how far a fine pass must disagree with the picture before it restates (higher = looser, more paint).
    #[serde(default)]
    pub impasto: Option<f32>,
    #[serde(default)]
    pub stroke_width: Option<f32>,
    #[serde(default)]
    pub stroke_length: Option<f32>,
    #[serde(default)]
    pub detail_restate: Option<f32>,
    #[serde(default)]
    pub hdr_amount: Option<f32>,
    /// LEVELS: how many value masses the armature is cut into — for a wash medium, how many glazes the picture
    /// is built from. Few and clean; many stacked transparent layers are mud.
    #[serde(default)]
    pub levels: Option<u32>,
    /// THE LAYERS, as the plan decides them (each absent = the medium's own; see `MediumProfile`): `dry` — how
    /// far the canvas dries between passes; `wash_in` — the block-in thinned to a wash; `wet_cap` — the
    /// wettest a later mark goes on; `scumble` — the finest pass dry-brushed off the faces; `turbulence` — the
    /// flow turned by a low-frequency noise, in degrees; `accent` — thick short dabs at the sharp contrasts;
    /// `loose` — longer, wider, fewer marks off the subject; `glaze` — translucent overlaps along the
    /// subject's silhouette; `deplete` — how fast a mark's charge runs out along it; `broken` — the drift of
    /// hue from mark to mark.
    #[serde(default)]
    pub dry: Option<f32>,
    #[serde(default)]
    pub wash_in: Option<f32>,
    #[serde(default)]
    pub wet_cap: Option<f32>,
    #[serde(default)]
    pub scumble: Option<f32>,
    #[serde(default)]
    pub turbulence: Option<f32>,
    #[serde(default)]
    pub accent: Option<f32>,
    #[serde(default)]
    pub loose: Option<f32>,
    #[serde(default)]
    pub glaze: Option<f32>,
    #[serde(default)]
    pub deplete: Option<f32>,
    #[serde(default)]
    pub broken: Option<f32>,
    /// SFUMATO (0..1): on a face, where the picture is smooth (a cheek, a forehead), the fine passes thin and
    /// the marks that remain go on broad, wet, half-charged and picking up the paint beneath — fused, not
    /// hatched. GLAZE PASS (0..1, the charge): one more pass at the end with a mid brush, every mark thin and
    /// wet over everything but the faces and hair, unifying the detail under one tone. VARY (0..1): a Gaussian spread of the mark's
    /// length, wider where the picture is flat. COMMA (0..1): the hand's pressure along the mark — the width
    /// and the deposit swell a third of the way along and lift toward the ends, a drop or a comma.
    #[serde(default)]
    pub sfumato: Option<f32>,
    #[serde(default)]
    pub glaze_pass: Option<f32>,
    #[serde(default)]
    pub vary: Option<f32>,
    #[serde(default)]
    pub comma: Option<f32>,
    /// SHADE (0..1): in the shadows, off the faces, the fine passes thin and the marks run longer, wider and
    /// calm — the detail kept for the light. IMPRIMATURA (0..1): the block-in laid darker and warmer (toward
    /// burnt umber), the underpainting that shows between the later marks. BODY VARY (σ): a Gaussian spread
    /// of each mark's body — thick and opaque, or thinner, wetter and translucent.
    #[serde(default)]
    pub shade: Option<f32>,
    #[serde(default)]
    pub imprimatura: Option<f32>,
    #[serde(default)]
    pub body_vary: Option<f32>,
    /// LOST (0..1): the subject's edge lost where it lies in shadow — the finest pass and the glaze pass paint
    /// the mean of both sides across the seam. CREST (0..1): one more pass after the glaze, this share of its
    /// seeds a fine brush dragged dry over the relief, a lighter warmer tint on the ridges' tops. SPATTER
    /// (0..1): thick drops flung off the thickest marks.
    #[serde(default)]
    pub lost: Option<f32>,
    #[serde(default)]
    pub crest: Option<f32>,
    #[serde(default)]
    pub spatter: Option<f32>,
    /// AERIAL (0..1): the lost edges by depth — a depth map is estimated, and far shadowed passages lose
    /// their edges by their distance, off the subject's silhouette too. SINK (0..1): on the crest pass, in
    /// the shadow, this share of its seeds a thin cool wash that settles only in the furrows of the relief.
    /// TAIL (0..1): a spatter drop stretched along its flight by the speed of the mark that flung it.
    #[serde(default)]
    pub aerial: Option<f32>,
    #[serde(default)]
    pub sink: Option<f32>,
    #[serde(default)]
    pub tail: Option<f32>,
    /// FILM (0..1): the glaze marks laid as a translucent film over the dried paint, read by the two-layer
    /// Kubelka-Munk solution. SPLIT (0..1): the dry fast marks combed into bundles. SAG (0..1): the canvas
    /// sagging under the thick paint, in the relight.
    #[serde(default)]
    pub film: Option<f32>,
    #[serde(default)]
    pub split: Option<f32>,
    #[serde(default)]
    pub sag: Option<f32>,
    /// GLOSS (0..1): a sharp specular on the peaks of the thick paint. TINT (0..1): the bundles of a split
    /// mark differ in tone. KNIFE (0..1): the share of the broad marks off the subject laid with a palette knife.
    #[serde(default)]
    pub gloss: Option<f32>,
    #[serde(default)]
    pub tint: Option<f32>,
    #[serde(default)]
    pub knife: Option<f32>,
    /// HIDING (0..1): each pigment covers by its own hiding power. TWIST (0..1): the brush turned along the
    /// hair strands and this share of the finest marks. SKIM (0..1): the share of the knife marks that skim.
    #[serde(default)]
    pub hiding: Option<f32>,
    #[serde(default)]
    pub twist: Option<f32>,
    #[serde(default)]
    pub skim: Option<f32>,
    /// The tempera's layers: HATCH (0..1): the restating marks straight hatch lines of one width. VERDACCIO
    /// (0..1): the flesh underpainted in a greenish grey. DISHES (a count): the marks laid in this many value
    /// steps. LIGHTS (0..1): the share of the finest marks on the subject's top values hatched in toward white.
    /// CROSS (degrees): the hatching turned off the form by this step a hatched pass, alternating sides. DITHER
    /// (0..1): the dishes drawn stochastically between the two nearest. SOLID (0..1): the hatch's charge keyed
    /// to the value, covering in the darks. FLESH (0..1): the verdaccio held to the warm colours.
    #[serde(default)]
    pub hatch: Option<f32>,
    #[serde(default)]
    pub verdaccio: Option<f32>,
    #[serde(default)]
    pub dishes: Option<f32>,
    #[serde(default)]
    pub lights: Option<f32>,
    #[serde(default)]
    pub cross: Option<f32>,
    #[serde(default)]
    pub dither: Option<f32>,
    #[serde(default)]
    pub solid: Option<f32>,
    #[serde(default)]
    pub flesh: Option<f32>,
    /// SPAN (0..1): the hatch's length keyed to how busy the form is. EDGE (0..1): the hatching held parallel
    /// to a strong contour. GLINT (0..1): the pure lights set last, dry, covering, narrower, raised.
    #[serde(default)]
    pub span: Option<f32>,
    #[serde(default)]
    pub edge: Option<f32>,
    #[serde(default)]
    pub glint: Option<f32>,
    /// SMOOTH (0..1): the calm passages built up in soft veils, their dishes relaxed, not hatched.
    #[serde(default)]
    pub smooth: Option<f32>,
    /// THE BOLE (0..1): the imprimatura of a new painting laid by local tone, not one flat colour.
    #[serde(default)]
    pub bole: Option<f32>,
    /// THE RAMP (0..1): a mark's colour follows the picture along its path, from its seed to its end.
    #[serde(default)]
    pub ramp: Option<f32>,
    /// THE UMBRA (0..1): within this share of the darkest dish the tones are laid continuous, not stepped.
    #[serde(default)]
    pub umbra: Option<f32>,
    /// CHALK (0..1): the mark a stick of chalk on the paper's tooth (`MediumProfile::chalk`).
    pub chalk: Option<f32>,
    /// THE SIDE OF THE STICK (0..1): the broad passes' share laid with the stick's side (`MediumProfile::side`).
    pub side: Option<f32>,
    /// THE FINGER (0..1): the calm passages rubbed after each broad pass (`MediumProfile::blend`).
    pub blend: Option<f32>,
    /// THE SKIN (0..1): the finger on the faces, kept inside them (`MediumProfile::skin`).
    pub skin: Option<f32>,
    /// CARNATION (0..1): the flesh in warm sticks (`MediumProfile::carnation`).
    pub carnation: Option<f32>,
    /// SMALL (px): a face under this extent drawn at its own scale (`MediumProfile::small`).
    pub small: Option<f32>,
    /// ANCHOR (0..1): a small face's structural darks kept crisp (`MediumProfile::anchor`).
    pub anchor: Option<f32>,
    /// TRIM (0..1): the face cut by the figure's matte (`MediumProfile::trim`).
    pub trim: Option<f32>,
    /// SMALL'S FLOOR (px): a crowd's face under it left to the plain ladder (`MediumProfile::small_min`).
    pub small_min: Option<f32>,
    /// CONTOUR (0..1): the line-drawing pass over the strongest edges — the medium's own when absent
    /// (charcoal's 0.3 drew every hair, ripple and tile as a black worm of a line over the tonal drawing).
    pub contour: Option<f32>,
    /// PAPER (0..1): the light is the paper, charcoal only darkens (`MediumProfile::paper`).
    pub paper: Option<f32>,
    /// ERASE (0..1): the kneaded eraser lifts the lights (`MediumProfile::erase`).
    pub erase: Option<f32>,
    /// TOUCH (0..1): the pressure follows the value (`MediumProfile::touch`).
    pub touch: Option<f32>,
    /// VELVET (0..1): the darkest share of the range one velvet black (`MediumProfile::velvet`).
    pub velvet: Option<f32>,
    /// PAPER TOP (0..1): the picture's light end is the paper (`MediumProfile::paper_top`).
    pub paper_top: Option<f32>,
    /// SWEEP (0..1): the flat passages in calm sweeps (`MediumProfile::sweep`).
    pub sweep: Option<f32>,
    /// STUMP (0..1): the dark masses rubbed with a stump (`MediumProfile::stump`).
    pub stump: Option<f32>,
    /// DUST (0..1): the charcoal's dust in the paper's hollows (`MediumProfile::dust`).
    pub dust: Option<f32>,
    /// STUMP EDGE (0..1): the stump holds at a contrast (`MediumProfile::stump_edge`).
    pub stump_edge: Option<f32>,
    /// STUMP FEATHER (0..1): the stump's soft stop (`MediumProfile::stump_feather`).
    pub stump_feather: Option<f32>,
    /// SWEEP WAVE (degrees): the sweeps on an arm's arc (`MediumProfile::sweep_wave`).
    pub sweep_wave: Option<f32>,
    /// PAPER KNEE (0..1): a broad light mass the paper (`MediumProfile::paper_knee`).
    pub paper_knee: Option<f32>,
    /// REWET (0..1): a dried layer lifted by a wet mark (`MediumProfile::rewet`).
    pub rewet: Option<f32>,
    /// CHALK TINT (0..1): the lights mixed with white, chalky (`MediumProfile::chalk_tint`).
    pub chalk_tint: Option<f32>,
    /// FLAT BRUSH (0..1): the flat passages in a flat brush's patches (`MediumProfile::flat_brush`).
    pub flat_brush: Option<f32>,
    /// LINER (0..1): the long thin lines in a liner's single stroke (`MediumProfile::liner`).
    pub liner: Option<f32>,
    /// TSUKE (0..1): the brush loaded in two tones (`MediumProfile::tsuke`).
    pub tsuke: Option<f32>,
    /// MA (0..1): the empty space where the eye does not go (`MediumProfile::ma`).
    pub ma: Option<f32>,
    /// KASURE (0..1): the flying white along a running brush (`MediumProfile::kasure`).
    pub kasure: Option<f32>,
    /// NIJIMI (0..1): the ink feathering into the paper (`MediumProfile::nijimi`).
    pub nijimi: Option<f32>,
    /// TURN (0..1): the brush lifted on a turn (`MediumProfile::turn`).
    pub turn: Option<f32>,
    /// YOHAKU (0..1): the flat passages left blank (`MediumProfile::yohaku`).
    pub yohaku: Option<f32>,
    /// BONE (0..1): the line register — the structure's contours as brush lines (`MediumProfile::bone`).
    pub bone: Option<f32>,
    /// BONELESS (a count): the block-in and this many passes after it laid as washes (`MediumProfile::boneless`).
    pub boneless: Option<f32>,
    /// HAKE (0..1): the washed passes' flow turned to one horizontal sweep (`MediumProfile::hake`).
    pub hake: Option<f32>,
    /// JOIN (0..1): the brush's stamps turned with the path over a window of this share of the width
    /// (`MediumProfile::join`).
    pub join: Option<f32>,
    /// BONE WIND (0..1): a bone chain winding past this bar is not drawn as a line (`MediumProfile::bone_wind`).
    pub bone_wind: Option<f32>,
    /// BONE DOT (0..1): the share of the winding chains laid as the tenbyō's dots (`MediumProfile::bone_dot`).
    pub bone_dot: Option<f32>,
    /// BONE SCATTER (0..1): the tenbyō's dots thrown off their chain by a Gaussian along its normal
    /// (`MediumProfile::bone_scatter`).
    pub bone_scatter: Option<f32>,
    /// BONE COHERENCE (0..1): a bone chain drawn only where the gradients around it agree on a direction
    /// (`MediumProfile::bone_coherence`).
    pub bone_coherence: Option<f32>,
    /// BONE END (0..1): the bone line's ends as the tome where it meets another line and the harai where it
    /// fades (`MediumProfile::bone_end`).
    pub bone_end: Option<f32>,
    /// SUMI FOCUS (0..1): the black register laid only where the attention reaches the key
    /// (`MediumProfile::sumi_focus`).
    pub sumi_focus: Option<f32>,
    /// MA FOCUS (0..1): the attention `ma` reads narrowed to the focus (`MediumProfile::ma_focus`).
    pub ma_focus: Option<f32>,
    /// MA FLOW (0..1): the focus's halo spread along the picture's lines, stopping at its edges, the subject's
    /// silhouette kept whole (`MediumProfile::ma_flow`).
    pub ma_flow: Option<f32>,
    /// BONE RYAKU (0..1): the abbreviated drawing: a bone line off the focus simplified to a few arcs (`MediumProfile::bone_ryaku`).
    pub bone_ryaku: Option<f32>,
    /// BONE FADE (0..1): a bone chain off the focus left to the washes (`MediumProfile::bone_fade`).
    pub bone_fade: Option<f32>,
    /// BOKASHI (0..1): the marks off the focus wetter: the fading edge dissolved, not merely lightened (`MediumProfile::bokashi`).
    pub bokashi: Option<f32>,
    /// VELVET DARK (0..1): the velvet black held to an absolute dark (`MediumProfile::velvet_dark`).
    pub velvet_dark: Option<f32>,
    /// SLANT (0..1): hatching at the hand's angle where the picture has no direction (`MediumProfile::slant`).
    pub slant: Option<f32>,
    /// GRADE (0..1): the half-tones lifted toward the paper, the velvet kept (`MediumProfile::grade`).
    pub grade: Option<f32>,
    /// WEIGHT (0..1): the bone line at a pencil's weight over the hatching (`MediumProfile::weight`).
    pub weight: Option<f32>,
    /// CLIP (0..1): the hatch stopped at the picture's contours (`MediumProfile::clip`).
    pub clip: Option<f32>,
    /// TIERS (0..1): the hatching in tonal layers: lights blank, half-tones single, shadows crossed (`MediumProfile::tiers`).
    pub tiers: Option<f32>,
    /// WAVER (0..1): the pressure wavering along each mark (`MediumProfile::waver`).
    pub waver: Option<f32>,
    /// NIB (0..1): a steel nib: every line full black, the tone by how many (`MediumProfile::nib`).
    pub nib: Option<f32>,
    /// NIBS (count): pens of several sizes in layers, broad to fine (`MediumProfile::nibs`).
    pub nibs: Option<f32>,
    /// NIB TONE (0..1): each pen only where the tone is dark enough for its width (`MediumProfile::nib_tone`).
    pub nib_tone: Option<f32>,
    /// NIB WHITE (0..1): the share of the paper a cell always keeps (`MediumProfile::nib_white`).
    pub nib_white: Option<f32>,
    /// NIB ANCHOR (0..1): the deep cast-shadow touches that set things on the ground (`MediumProfile::nib_anchor`).
    pub nib_anchor: Option<f32>,
    /// NIB HINT (0..1): the fine pen's rare short dashes in the lights (`MediumProfile::nib_hint`).
    pub nib_hint: Option<f32>,
    /// NIB FACE (0..1): a face drawn with the finest pen, its lights kept (`MediumProfile::nib_face`).
    pub nib_face: Option<f32>,
    /// NIB GAP (0..1): a sliver of paper kept along the long contours (`MediumProfile::nib_gap`).
    pub nib_gap: Option<f32>,
    /// NIB RELIEF (0..1): the hatching thinned on the form's highlights (`MediumProfile::nib_relief`).
    pub nib_relief: Option<f32>,
    /// NIB FAR (0..1): the far ground and the sky in the finest pen only (`MediumProfile::nib_far`).
    pub nib_far: Option<f32>,
    /// NIB OCCLUDE (0..1): the broad nib only in the cast shadows (`MediumProfile::nib_occlude`).
    pub nib_occlude: Option<f32>,
    /// NIB EYES (0..1): the eyes and nose taken out of the fur and outlined (`MediumProfile::nib_eyes`).
    pub nib_eyes: Option<f32>,
    /// NIB FUR (0..1): the fur thinned on the body's lit bulges (`MediumProfile::nib_fur`).
    pub nib_fur: Option<f32>,
    /// NIB LIFT (0..1): the figure's half-tones lifted, the deep folds kept (`MediumProfile::nib_lift`).
    pub nib_lift: Option<f32>,
    /// NIB CONTACT (0..1): a contact shadow under the subject (`MediumProfile::nib_contact`).
    pub nib_contact: Option<f32>,
    /// NIB FIGURE (0..1): a figure drawn in parallel hatching with bold contours (`MediumProfile::nib_figure`).
    pub nib_figure: Option<f32>,
    /// NIB OBJECT (0..1): the held objects outlined with the figure (`MediumProfile::nib_object`).
    pub nib_object: Option<f32>,
    /// NIB TUFT (0..1): the near grass as fans of blades from one root (`MediumProfile::nib_tuft`).
    pub nib_tuft: Option<f32>,
    /// NIB CLOSE (0..1): the figure's outline gaps closed (`MediumProfile::nib_close`).
    pub nib_close: Option<f32>,
    /// NIB MASS (0..1): the dark grass's tufts crossed into one mass (`MediumProfile::nib_mass`).
    pub nib_mass: Option<f32>,
    /// NIB RIM (0..1): the figure's outline in a line of changing weight (`MediumProfile::nib_rim`).
    pub nib_rim: Option<f32>,
    /// NIB POOL (0..1): the contact shadows solid black under the figure (`MediumProfile::nib_pool`).
    pub nib_pool: Option<f32>,
    /// NIB AIR (0..1): the far planes thinned to a few fine lines (`MediumProfile::nib_air`).
    pub nib_air: Option<f32>,
    /// NIB PEOPLE (0..1): every person found is the figure, with or without a face (`MediumProfile::nib_people`).
    pub nib_people: Option<f32>,
    /// NIB VALUE (0..1): the figure's tone read against the whole picture, its lights kept light (`MediumProfile::nib_value`).
    pub nib_value: Option<f32>,
    /// NIB FINE FACE (0..1): a small face drawn with the finest pen, its features found at a lower threshold (`MediumProfile::nib_fine_face`).
    pub nib_fine_face: Option<f32>,
    /// NIB SCATTER (0..1): the tufts scattered in clusters, not one per cell (`MediumProfile::nib_scatter`).
    pub nib_scatter: Option<f32>,
    /// NIB FALLOFF (0..1): the contact shadow solid only at the contact, broken into dashes beyond (`MediumProfile::nib_falloff`).
    pub nib_falloff: Option<f32>,
    /// NIB DEPTH (0..1): the far planes by the picture's depth drawn with the finest pen (`MediumProfile::nib_depth`).
    pub nib_depth: Option<f32>,
    /// NIB LUSH (0..1): the found grass grown thick, its tufts denser (`MediumProfile::nib_lush`).
    pub nib_lush: Option<f32>,
    /// NIB GLINT (0..1): the specular highlights kept white, no mark through them (`MediumProfile::nib_glint`).
    pub nib_glint: Option<f32>,
    /// NIB WASH (0..1): ink and wash: the ladder laid as washes, then over them as the pens (`MediumProfile::nib_wash`).
    pub nib_wash: Option<f32>,
    /// NIB WASH DARK (0..1): the pens' hatching held to the darks, the half-tones left to the washes (`MediumProfile::nib_wash_dark`).
    pub nib_wash_dark: Option<f32>,
    /// NIB WASH DETAIL (0..1): the finest pen drawing the picture's own detail over the washes (`MediumProfile::nib_wash_detail`).
    pub nib_wash_detail: Option<f32>,
    /// NIB WASH LINE (0..1): the bone line traced from the picture itself, its detail kept (`MediumProfile::nib_wash_line`).
    pub nib_wash_line: Option<f32>,
    /// NIB WASH SWELL (0..1): the line's width swelling with the strength of its edge (`MediumProfile::nib_wash_swell`).
    pub nib_wash_swell: Option<f32>,
    /// NIB WASH AIR (0..1): the far planes left to soft, light washes, the line fading into the distance (`MediumProfile::nib_wash_air`).
    pub nib_wash_air: Option<f32>,
    /// NIB WASH SOFT (0..1): the washes from the picture's low band, the high band left to the pens (`MediumProfile::nib_wash_soft`).
    pub nib_wash_soft: Option<f32>,
    /// NIB WASH GAMMA (0..1): the washes' density raised by a gamma, the paper kept (`MediumProfile::nib_wash_gamma`).
    pub nib_wash_gamma: Option<f32>,
    /// NIB WASH FLAT (0..1): a flat dark left to the wash's pool, the pen only where the dark has an edge or a texture (`MediumProfile::nib_wash_flat`).
    pub nib_wash_flat: Option<f32>,
    /// SUMI DARK (0..1): the share of the darkest values the sumi-e's black register lays as dry-brush black
    /// shapes (the darkest eighth when unset) (`PaintParams::sumi_dark`).
    pub sumi_dark: Option<f32>,
    /// THE SMALLEST BRUSH (px): the ladder's finest rung and the floor under every mark — the tool's own
    /// limit (a stick of chalk cannot make a one-pixel mark). Overrides the CLI default (2) only.
    pub min_brush: Option<f32>,
    /// PICKUP (0..1): how strongly a mark lifts wet canvas pigment into its load (`MediumProfile::pickup`) —
    /// 0 for a stick of chalk, which carries nothing of the layer beneath.
    pub pickup: Option<f32>,
    /// BLEED (0..1): the wet-into-wet diffusion between the marks after each pass (`MediumProfile::bleed`).
    pub bleed: Option<f32>,
    /// Human-readable analysis notes (why these numbers) — informational, ignored by the paint stage.
    #[serde(default)]
    pub notes: Vec<String>,
}

fn default_medium() -> String {
    "watercolour".into()
}
fn default_palette() -> String {
    "image".into()
}
fn default_shadow_floor() -> f32 {
    0.16
}

fn default_detail_len() -> f32 {
    1.0
}

fn default_style() -> String {
    // LEGIBLE resolves features on the fine passes (it has a detail tier); IMPRESSIONIST has none, so an
    // analyzer that defaulted to impressionist produced masses-only mush. Legible is the right default; a user
    // who wants loose masses still asks for `--style impressionist` or writes it into the plan.
    "legible".into()
}
fn default_armature() -> u32 {
    72
}

impl Default for PaintPlan {
    fn default() -> Self {
        Self {
            medium: default_medium(),
            palette: default_palette(),
            style: default_style(),
            armature: default_armature(),
            armature_face: None,
            armature_body: None,
            recede: 0.0,
            semantic: false,
            families: false,
            commit_shadows: 0.0,
            silhouette: 0.0,
            silhouette_mode: None,
            sam: false,
            value_key: 0.0,
            shadow_floor: 0.16,
            detail_len: 1.0,
            reserve: None,
            budget: None,
            from_scratch: false,
            hair_mask: None,
            infill: None,
            rigger: None,
            hotspot: None,
            technique: None,
            splatter: None,
            edge_pool: None,
            oldpaper: None,
            granulate: None,
            saliency: None,
            focus_detail: None,
            ladder_keep: None,
            leak: None,
            hdr: None,
            ridges: None,
            skip: None,
            fine_lines: None,
            impasto_map: None,
            weave: None,
            sheen: None,
            collide: None,
            analysis: None,
            analysis_insights: None,
            outcome: None,
            impasto: None,
            stroke_width: None,
            stroke_length: None,
            detail_restate: None,
            hdr_amount: None,
            levels: None,
            dry: None,
            wash_in: None,
            wet_cap: None,
            scumble: None,
            turbulence: None,
            accent: None,
            loose: None,
            glaze: None,
            deplete: None,
            broken: None,
            sfumato: None,
            glaze_pass: None,
            vary: None,
            comma: None,
            shade: None,
            imprimatura: None,
            body_vary: None,
            lost: None,
            crest: None,
            spatter: None,
            aerial: None,
            sink: None,
            tail: None,
            film: None,
            split: None,
            sag: None,
            gloss: None,
            tint: None,
            knife: None,
            hiding: None,
            twist: None,
            skim: None,
            hatch: None,
            verdaccio: None,
            dishes: None,
            lights: None,
            cross: None,
            dither: None,
            solid: None,
            flesh: None,
            span: None,
            edge: None,
            glint: None,
            smooth: None,
            bole: None,
            ramp: None,
            umbra: None,
            chalk: None,
            side: None,
            blend: None,
            skin: None,
            carnation: None,
            small: None,
            anchor: None,
            trim: None,
            small_min: None,
            contour: None,
            paper: None,
            erase: None,
            touch: None,
            velvet: None,
            paper_top: None,
            sweep: None,
            stump: None,
            dust: None,
            stump_edge: None,
            stump_feather: None,
            sweep_wave: None,
            paper_knee: None,
            rewet: None,
            chalk_tint: None,
            flat_brush: None,
            liner: None,
            tsuke: None,
            ma: None,
            kasure: None,
            nijimi: None,
            turn: None,
            yohaku: None,
            bone: None,
            boneless: None,
            hake: None,
            join: None,
            bone_wind: None,
            bone_dot: None,
            bone_scatter: None,
            bone_coherence: None,
            bone_end: None,
            sumi_focus: None,
            ma_focus: None,
            ma_flow: None,
            bone_ryaku: None,
            bone_fade: None,
            bokashi: None,
            velvet_dark: None,
            slant: None,
            grade: None,
            weight: None,
            clip: None,
            tiers: None,
            waver: None,
            nib: None,
            nibs: None,
            nib_tone: None,
            nib_white: None,
            nib_anchor: None,
            nib_hint: None,
            nib_face: None,
            nib_gap: None,
            nib_relief: None,
            nib_far: None,
            nib_occlude: None,
            nib_eyes: None,
            nib_fur: None,
            nib_lift: None,
            nib_contact: None,
            nib_figure: None,
            nib_object: None,
            nib_tuft: None,
            nib_close: None,
            nib_mass: None,
            nib_rim: None,
            nib_pool: None,
            nib_air: None,
            nib_people: None,
            nib_value: None,
            nib_fine_face: None,
            nib_scatter: None,
            nib_falloff: None,
            nib_depth: None,
            nib_lush: None,
            nib_glint: None,
            nib_wash: None,
            nib_wash_dark: None,
            nib_wash_detail: None,
            nib_wash_line: None,
            nib_wash_swell: None,
            nib_wash_air: None,
            nib_wash_soft: None,
            nib_wash_gamma: None,
            nib_wash_flat: None,
            sumi_dark: None,
            min_brush: None,
            pickup: None,
            bleed: None,
            notes: Vec::new(),
        }
    }
}

impl PaintPlan {
    /// Parse a plan from HJSON.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(deser_hjson::from_str(text)?)
    }

    /// Serialise to HJSON (stable key order, with the analysis notes as comments).
    pub fn to_hjson(&self) -> String {
        let mut o = String::new();
        o.push_str("# plakat painting plan (RFC PAINT-1) — the art director's structural decisions.\n");
        for n in &self.notes {
            o.push_str(&format!("# {n}\n"));
        }
        o.push_str(&format!("medium: {}\n", self.medium));
        o.push_str(&format!("palette: {}\n", self.palette));
        o.push_str(&format!("style: {}\n", self.style));
        o.push_str(&format!("armature: {}\n", self.armature));
        if let Some(af) = self.armature_face {
            o.push_str(&format!("armature_face: {af}\n"));
        }
        if let Some(ab) = self.armature_body {
            o.push_str(&format!("armature_body: {ab}\n"));
        }
        if self.recede > 1e-3 {
            o.push_str(&format!("recede: {:.2}\n", self.recede));
        }
        if self.from_scratch {
            o.push_str("new: true\n");
        }
        if let Some(hm) = &self.hair_mask {
            o.push_str(&format!("hair_mask: {hm}\n"));
        }
        if let Some(inf) = &self.infill {
            o.push_str(&format!("infill: {inf}\n"));
        }
        if let Some(rg) = self.rigger {
            o.push_str(&format!("rigger: {rg:.2}\n"));
        }
        if let Some(hs) = self.hotspot {
            o.push_str(&format!("hotspot: {hs:.2}\n"));
        }
        if let Some(t) = &self.technique {
            o.push_str(&format!("technique: {t}\n"));
        }
        for (k, v) in [("splatter", self.splatter), ("edge_pool", self.edge_pool), ("granulate", self.granulate), ("saliency", self.saliency), ("focus_detail", self.focus_detail)] {
            if let Some(v) = v {
                o.push_str(&format!("{k}: {v:.2}\n"));
            }
        }
        if let Some(v) = self.oldpaper {
            o.push_str(&format!("oldpaper: {v}\n"));
        }
        if let Some(n) = self.ladder_keep {
            o.push_str(&format!("ladder_keep: {n}\n"));
        }
        if let Some(l) = self.leak {
            o.push_str(&format!("leak: {l:.2}\n"));
        }
        if let Some(hd) = self.hdr {
            o.push_str(&format!("hdr: {hd}\n"));
        }
        if let Some(v) = self.ridges {
            o.push_str(&format!("ridges: {v:.2}\n"));
        }
        if let Some(v) = self.skip {
            o.push_str(&format!("skip: {v:.2}\n"));
        }
        if let Some(v) = self.fine_lines {
            o.push_str(&format!("fine_lines: {v:.2}\n"));
        }
        if let Some(v) = self.impasto_map {
            o.push_str(&format!("impasto_map: {v:.2}\n"));
        }
        if let Some(v) = self.weave {
            o.push_str(&format!("weave: {v:.2}\n"));
        }
        if let Some(v) = self.sheen {
            o.push_str(&format!("sheen: {v:.2}\n"));
        }
        if let Some(v) = self.collide {
            o.push_str(&format!("collide: {v:.2}\n"));
        }
        for (k, v) in [("dry", self.dry), ("wash_in", self.wash_in), ("wet_cap", self.wet_cap), ("scumble", self.scumble), ("turbulence", self.turbulence), ("accent", self.accent), ("loose", self.loose), ("glaze", self.glaze), ("deplete", self.deplete), ("broken", self.broken), ("sfumato", self.sfumato), ("glaze_pass", self.glaze_pass), ("vary", self.vary), ("comma", self.comma), ("shade", self.shade), ("imprimatura", self.imprimatura), ("body_vary", self.body_vary), ("lost", self.lost), ("crest", self.crest), ("spatter", self.spatter), ("aerial", self.aerial), ("sink", self.sink), ("tail", self.tail), ("film", self.film), ("split", self.split), ("sag", self.sag), ("gloss", self.gloss), ("tint", self.tint), ("knife", self.knife), ("hiding", self.hiding), ("twist", self.twist), ("skim", self.skim), ("hatch", self.hatch), ("verdaccio", self.verdaccio), ("dishes", self.dishes), ("lights", self.lights), ("cross", self.cross), ("dither", self.dither), ("solid", self.solid), ("flesh", self.flesh), ("span", self.span), ("edge", self.edge), ("glint", self.glint), ("smooth", self.smooth), ("bole", self.bole), ("ramp", self.ramp), ("umbra", self.umbra), ("chalk", self.chalk), ("side", self.side), ("blend", self.blend), ("skin", self.skin), ("carnation", self.carnation), ("small", self.small), ("anchor", self.anchor), ("trim", self.trim), ("small_min", self.small_min), ("contour", self.contour), ("paper", self.paper), ("erase", self.erase), ("touch", self.touch), ("velvet", self.velvet), ("paper_top", self.paper_top), ("sweep", self.sweep), ("stump", self.stump), ("dust", self.dust), ("stump_edge", self.stump_edge), ("stump_feather", self.stump_feather), ("sweep_wave", self.sweep_wave), ("paper_knee", self.paper_knee), ("rewet", self.rewet), ("chalk_tint", self.chalk_tint), ("flat_brush", self.flat_brush), ("liner", self.liner), ("tsuke", self.tsuke), ("ma", self.ma), ("kasure", self.kasure), ("nijimi", self.nijimi), ("turn", self.turn), ("yohaku", self.yohaku), ("bone", self.bone), ("boneless", self.boneless), ("hake", self.hake), ("join", self.join), ("bone_wind", self.bone_wind), ("bone_dot", self.bone_dot), ("bone_scatter", self.bone_scatter), ("bone_coherence", self.bone_coherence), ("bone_end", self.bone_end), ("sumi_focus", self.sumi_focus), ("ma_focus", self.ma_focus), ("ma_flow", self.ma_flow), ("bone_ryaku", self.bone_ryaku), ("bone_fade", self.bone_fade), ("bokashi", self.bokashi), ("velvet_dark", self.velvet_dark), ("slant", self.slant), ("grade", self.grade), ("weight", self.weight), ("clip", self.clip), ("tiers", self.tiers), ("waver", self.waver), ("nib", self.nib), ("nibs", self.nibs), ("nib_tone", self.nib_tone), ("nib_white", self.nib_white), ("nib_anchor", self.nib_anchor), ("nib_hint", self.nib_hint), ("nib_face", self.nib_face), ("nib_gap", self.nib_gap), ("nib_relief", self.nib_relief), ("nib_far", self.nib_far), ("nib_occlude", self.nib_occlude), ("nib_eyes", self.nib_eyes), ("nib_fur", self.nib_fur), ("nib_lift", self.nib_lift), ("nib_contact", self.nib_contact), ("nib_figure", self.nib_figure), ("nib_object", self.nib_object), ("nib_tuft", self.nib_tuft), ("nib_close", self.nib_close), ("nib_mass", self.nib_mass), ("nib_rim", self.nib_rim), ("nib_pool", self.nib_pool), ("nib_air", self.nib_air), ("nib_people", self.nib_people), ("nib_value", self.nib_value), ("nib_fine_face", self.nib_fine_face), ("nib_scatter", self.nib_scatter), ("nib_falloff", self.nib_falloff), ("nib_depth", self.nib_depth), ("nib_lush", self.nib_lush), ("nib_glint", self.nib_glint), ("nib_wash", self.nib_wash), ("nib_wash_dark", self.nib_wash_dark), ("nib_wash_detail", self.nib_wash_detail), ("nib_wash_line", self.nib_wash_line), ("nib_wash_swell", self.nib_wash_swell), ("nib_wash_air", self.nib_wash_air), ("nib_wash_soft", self.nib_wash_soft), ("nib_wash_gamma", self.nib_wash_gamma), ("nib_wash_flat", self.nib_wash_flat), ("sumi_dark", self.sumi_dark), ("min_brush", self.min_brush), ("pickup", self.pickup), ("bleed", self.bleed)] {
            if let Some(v) = v {
                o.push_str(&format!("{k}: {v:.2}\n"));
            }
        }
        match &self.analysis {
            Some(Artefact::On(true)) => o.push_str("analysis: true\n"),
            Some(Artefact::Path(p)) => o.push_str(&format!("analysis: {p:?}\n")),
            _ => {}
        }
        match &self.analysis_insights {
            Some(Artefact::On(true)) => o.push_str("analysis_insights: true\n"),
            Some(Artefact::Path(p)) => o.push_str(&format!("analysis_insights: {p:?}\n")),
            _ => {}
        }
        match &self.outcome {
            Some(Artefact::On(true)) => o.push_str("outcome: true\n"),
            Some(Artefact::Path(p)) => o.push_str(&format!("outcome: {p:?}\n")),
            _ => {}
        }
        for (k, v) in [("impasto", self.impasto), ("stroke_width", self.stroke_width), ("stroke_length", self.stroke_length), ("detail_restate", self.detail_restate)] {
            if let Some(v) = v {
                o.push_str(&format!("{k}: {v:.2}\n"));
            }
        }
        if let Some(v) = self.hdr_amount {
            o.push_str(&format!("hdr_amount: {v:.2}\n"));
        }
        if let Some(n) = self.levels {
            o.push_str(&format!("levels: {n}\n"));
        }
        if self.semantic {
            o.push_str("semantic: true\n");
        }
        if self.families {
            o.push_str("families: true\n");
        }
        if self.commit_shadows > 1e-3 {
            o.push_str(&format!("commit_shadows: {:.2}\n", self.commit_shadows));
        }
        if self.silhouette > 1e-3 {
            o.push_str(&format!("silhouette: {:.2}\n", self.silhouette));
            o.push_str(&format!("silhouette_mode: {}\n", self.silhouette_mode.as_deref().unwrap_or("line")));
        }
        if self.sam {
            o.push_str("sam: true\n");
        }
        o.push_str(&format!("value_key: {:.2}\n", self.value_key));
        o.push_str(&format!("shadow_floor: {:.2}\n", self.shadow_floor));
        o.push_str(&format!("detail_len: {:.2}\n", self.detail_len));
        if let Some(r) = self.reserve {
            o.push_str(&format!("reserve: {r:.2}\n"));
        }
        if let Some(b) = self.budget {
            o.push_str(&format!("budget: {b}\n"));
        }
        o
    }
}

/// The measured signals the analyzer reads from the reference. Kept separate so the (GPU) face detection and the
/// (CPU) statistics are gathered by the caller and handed in — the analyzer itself is a pure function.
pub struct Analysis {
    /// Number of faces detected (the focal subjects).
    pub faces: usize,
    /// Global luma standard deviation in [0,1] — a proxy for tonal contrast (low = flat/foggy).
    pub luma_stddev: f32,
    /// Global mean luma in [0,1] — the picture's key (a nocturne sits well below 0.3).
    pub luma_mean: f32,
    /// The medium requested (drives the reserve default).
    pub medium: String,
    /// The palette requested (usually `image`).
    pub palette: String,
    /// Canvas short side (px) — scales the budget.
    pub short_side: u32,
    /// Long side (px) — for budget area.
    pub long_side: u32,
    /// Whether the medium reserves the paper white (watercolour / ink).
    pub surface_white: bool,
    /// Whether a NEW painting (from scratch, RFC §1.1) was asked for — the plan then chooses the coarse tiers.
    pub from_scratch: bool,
    /// STRUCTURE: the standard deviation of the 8-neighbour Laplacian of the luma (in [0,1] units) — how much
    /// pixel-scale detail the picture carries (a flat portrait ~0.2, a busy street or foliage ~0.5). Scales the
    /// stroke budget: a complex picture needs more marks per pixel to be restated than a plain one.
    pub structure: f32,
}

/// The stroke-budget multiplier a picture's `structure` earns, relative to the reference it was tuned on
/// (`STRUCTURE_NORM` → ×1). Never below ×1 (the accepted density is the floor — a plain picture is not thinned) and capped so a noisy one does not explode.
pub fn structure_factor(structure: f32) -> f32 {
    (structure / STRUCTURE_NORM).clamp(1.0, 1.6)
}
/// The structure of the pictures the area/10 density was tuned and accepted on.
pub const STRUCTURE_NORM: f32 = 0.3330;

/// `structure` for `Analysis`: σ of the 8-neighbour Laplacian of the [0,1] luma (border pixels excluded).
pub fn structure_of(img: &image::RgbImage) -> f32 {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let luma: Vec<f32> = img.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0).collect();
    let (mut sum, mut sq, mut n) = (0.0f64, 0.0f64, 0u64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = luma[y * w + x];
            let mut nb = 0.0f32;
            for dy in [-1i64, 0, 1] {
                for dx in [-1i64, 0, 1] {
                    if dx != 0 || dy != 0 {
                        nb += luma[((y as i64 + dy) as usize) * w + (x as i64 + dx) as usize];
                    }
                }
            }
            let l = (8.0 * c - nb) as f64;
            sum += l;
            sq += l * l;
            n += 1;
        }
    }
    let n = n.max(1) as f64;
    let mean = sum / n;
    ((sq / n - mean * mean).max(0.0)).sqrt() as f32
}

/// Turn measured signals into a plan — the deterministic art director. The rules encode the tuning that had to be
/// done by hand: paint from a coarse armature (never trace), give a detected FACE a finer armature so it stays
/// crisp while the rest becomes washes, expand a flat reference's values, and reserve the paper for wet media.
pub fn plan_from(a: &Analysis) -> PaintPlan {
    let mut notes = Vec::new();

    // The base armature (structure resolution). With the STRUCTURE-PRESERVING armature (edge-preserving + value
    // masses, not a blur) this can be fairly FINE without tracing — finer retains modelling, and density comes
    // from the budget, not from over-coarsening. (The background goes a bit coarser when a subject is present.)
    // A NEW painting (RFC §1.1, §5.2) reads a REDUCED armature: the things in the picture, none of its texture.
    let armature = if a.from_scratch { if a.faces > 0 { crate::paint::painter::NEW_BACKGROUND_SIDE } else { crate::paint::painter::NEW_BACKGROUND_SIDE_PLAIN } } else if a.short_side >= 900 { 150 } else { 120 };
    if a.from_scratch {
        notes.push("new painting: a reduced armature — the things in the picture, none of its texture; the brushwork invents the surface (RFC §1.1)".into());
    } else {
        notes.push("structure-preserving armature (value masses + edges), fine enough to keep modelling".into());
    }

    // A detected FACE is the focal region: give it a FINE armature so features stay crisp while the beard / hair /
    // background become washes (RFC §5.2). With a subject present, use a THREE-TIER plan — a coarser background, a
    // mid subject body, and the fine face — plus a touch of aerial recession so the subject advances.
    let (armature_face, armature_body, recede, armature) = if a.faces > 0 && a.from_scratch {
        notes.push(format!("{} face(s) → three-tier armature: background {armature}px · figure {}px and face {}px ACROSS THEIR OWN EXTENT (RFC §5.2), set once they are found; the finest brushes on the focal plane only", a.faces, crate::paint::painter::NEW_FIGURE_ACROSS as u32, crate::paint::painter::NEW_FACE_ACROSS as u32));
        notes.push("subject matte (U2Net) + SAM masks → the planes and their minimum brush".into());
        (Some(armature * 2), Some(armature), 0.0, armature)
    } else if a.faces > 0 {
        notes.push(format!("{} face(s) → three-tier armature: background {}px · body 210px · face 300px", a.faces, (armature as f32 * 0.85).round().max(90.0) as u32));
        notes.push("subject matte (U2Net) → body/background split; background recedes (aerial perspective)".into());
        // The background can go coarser than the default when the body/face carry the structure — a calmer ground.
        notes.push("semantic tiers (OWL-ViT): hair/beard → a slightly coarser tier".into());
        notes.push("SAM precise masks → sharp silhouette + face-shaped focal (finer face armature)".into());
        // FINE armatures build DENSITY: a fine (detailed) armature keeps the restate passes finding structure to
        // paint, so the budget is used and the painting reads rich — not a sparse wash. The structure-preserving
        // armature keeps this from tracing. Mild recession only (heavy recede erases soft periphery).
        // Background stays only a touch coarser than the body (0.85), not the old aggressive 0.62 that starved it.
        // No aerial veil by default: measured on a real scene it lifted mean luma +0.04 and moved the painting
        // AWAY from the target structure (SSIM); the reference's own depth cues carry the recession. `--recede`
        // stays available for a deliberate atmospheric treatment.
        ((Some(300)), Some(210), 0.0, (armature as f32 * 0.85).round().max(90.0) as u32)
    } else {
        notes.push("no face → uniform coarse armature".into());
        (None, None, 0.0, armature)
    };
    let semantic = a.faces > 0;
    // A LUMINOUS medium (line-and-wash) keeps the paper's light: the oil-minded backbone below — family
    // separation, committed shadows, a deep shadow floor — is what made every wash read heavy and muddy.
    let medium_mark = crate::paint::medium::MediumProfile::by_name(&a.medium).map(|m| m.mark);
    let luminous = medium_mark.map(|m| m.luminous).unwrap_or(false);
    let book = medium_mark.map(|m| m.book).unwrap_or(false);
    // FAMILY SEPARATION — group light/shadow masses so the painting reads SOLID, not a washed photographic average.
    let families = !luminous;
    if families {
        notes.push("family separation (light/shadow masses) → solid, not washed".into());
    } else {
        notes.push("luminous medium → no family separation, no committed shadows: the washes stay transparent".into());
    }
    // COMMIT SHADOWS — paint the dark masses decisively (a solid value backbone), the direct fix for a pale wash.
    // Measured (same ladder, budget, length): 0.40 → 0.85 → 1.00 cut speck flecks on smooth masses 0.040 → 0.031
    // → 0.029 (target painting 0.025) and gave the best structural agreement — committed darks are laid SOLID, so
    // nothing later flecks them. Full commitment is the right default; `--commit-shadows` lowers it deliberately.
    let commit_shadows = if luminous { 0.0 } else { 1.0 };
    if !luminous {
        notes.push("commit shadows 1.0 → decisive solid dark masses (value backbone; fewer flecks)".into());
    }
    // SILHOUETTE — OFF by default: an auto-drawn contour line reads as tacked-on on most subjects (it is only
    // wanted deliberately). Opt in with `--silhouette <n>` / `--silhouette-mode`; the plan leaves it disabled.
    let (silhouette, silhouette_mode) = (0.0, None);
    // SAM precise masks when a subject is present — a sharp silhouette and a face-shaped focal region.
    let sam = a.faces > 0;

    // VALUE KEY from measured contrast: a flat, foggy reference (low stddev) needs more tonal expansion for real
    // darks and lights; a punchy reference needs little. Map stddev∈[~0.10,0.28] → value_key∈[0.9,0.2].
    // Expand ONLY a genuinely flat/foggy reference (low luma σ). A normal-contrast reference gets no stretch: the
    // stretch lightened a faithful painting by +0.04 mean luma for nothing — a painter re-keys a fog, not a
    // clear day. σ ≥ 0.20 → 0; σ 0.08 → 0.7.
    // A LUMINOUS medium is re-keyed HIGH whatever the picture's contrast: the illustrator's wash keeps the darks
    // as transparent mid-darks and lets the paper carry the light (the CLI lifts the low end of the key too).
    // The BOOK is keyed high (a print's flat light); the watercolour only mildly — its colours are the
    // picture's own, lightened by thin washes over white paper.
    // A NOCTURNE stays a nocturne: the watercolour's mild key lifts a daylight picture toward the paper, but
    // on a dark picture (mean luma below ~0.3) it only washed the night out — no key there; the paper is for
    // the lamps and the stars alone. (The book keys high regardless: a print's flat light.)
    let nocturne = a.luma_mean < 0.3;
    let value_key = if book { 1.0 } else if luminous && nocturne { 0.0 } else if luminous { 0.45 } else { ((0.20 - a.luma_stddev) / (0.20 - 0.08) * 0.7).clamp(0.0, 0.7) };
    if book {
        notes.push("book illustration → high key: the whole range lifted, the paper carries the light".into());
    } else if luminous && nocturne {
        notes.push(format!("luminous medium, nocturne (mean luma {:.2}) → no key: the night keeps its depth, the paper is for the lights", a.luma_mean));
    } else if luminous {
        notes.push(format!("luminous medium → mild key {value_key:.2}: the picture's own colours, lightened by thin washes over paper"));
    } else {
        notes.push(format!("luma σ {:.3} → value-key {:.2} (expand a flat reference's tonal range)", a.luma_stddev, value_key));
    }
    // SHADOW FLOOR (RFC §3.3): the re-key keeps the shadow family a narrow, LIFTED band — a solid dark mass, not a
    // crush to black. 0.16 ≈ a deep but readable dark; raise for a high-key picture, lower for a nocturne.
    let shadow_floor = if luminous { 0.24 } else { 0.16 };
    notes.push(format!("shadow floor {shadow_floor:.2} → shadow family lifted off black ({})", if luminous { "a wash's darks stay transparent" } else { "solid dark masses, RFC §3.3" }));

    // Reserve the paper ONLY for the brightest highlights — a lower cutoff starves a light subject (a white
    // beard/shirt) into sparse, washed paper. 0.92 paints the light masses and keeps only the true whites as paper.
    let reserve = a.surface_white.then_some(0.92);
    if reserve.is_some() {
        notes.push("surface-white medium → reserve 0.92 (paper only for true highlights; paint the light masses)".into());
    }

    // Budget scales with area for DENSITY — the earlier low budgets starved the painting into a sparse, washed
    // look. Match a rich block-in (~1 stroke per ~9 px, like the 30k reference on 512²), capped so a big canvas
    // stays sane. This is the single biggest lever the analyzer was getting wrong.
    let area = a.short_side as u64 * a.long_side as u64;
    // THIN + SHORT + MANY (measured policy): thin brushes seed a denser grid and short marks cover less each, so
    // the budget must scale to match or coverage is lost (the ground shows and the picture lightens). ~1 stroke
    // per 10 px² lays ~105k at 1024² (all of them are placed); the cap keeps a huge canvas sane.
    // The budget is a DENSITY, not a count: `paint from` paints at the image's own resolution, and a cap made
    // every canvas above ~1.2 Mpx thinner (a 4 Mpx sheet got a third of the strokes per pixel — its fine
    // passes starved and the features smeared; a photo grew, the budget did not). Drawing media are the
    // exception: their lines are spaced by the sheet's size already and their budget only widens the hatch,
    // so they keep the cap (and the accepted renders).
    let drawing = crate::paint::medium::MediumProfile::by_name(&a.medium).map(|m| m.mark_model == crate::paint::medium::MarkModel::Density).unwrap_or(false);
    // ... and a density is not enough either: the SAME area of a plain portrait and of a busy street do not need
    // the same number of marks — the street's detail must be restated mark by mark, the portrait's masses are
    // covered by a few. So the painted budget is area/10 × a COMPLEXITY factor read from the picture's own
    // structure (Laplacian σ, see `structure_factor`), ×1 for the pictures the density was accepted on.
    if a.from_scratch {
        notes.push("new painting: the budget is a COVERAGE count, set once the planes are known".into());
    }
    let factor = structure_factor(a.structure);
    let budget = if drawing { ((area / 10) as usize).clamp(12000, 120000) } else { (((area / 10) as f64 * factor as f64) as usize).max(12000) };
    notes.push(if drawing {
        format!("budget {budget} strokes (thin marks need density — area/10, drawing cap 120k)")
    } else {
        format!("budget {budget} strokes (area/10 × complexity {factor:.2}: structure {:.3} vs norm {STRUCTURE_NORM})", a.structure)
    });
    // DEPTH ORDER: the widest layer covers with long strokes; each thinner layer on top is shorter, tapering to
    // ~0.3 of the profile length on the finest. Long drags on the fine layers smeared features into blobs; short
    // dabs let the face, hands, book and distant figures READ (validated against a target painting at 1:1).
    // 0.5, not 0.3: measured against the target painting, 0.3 popped detail but cost speck flecks on the smooth
    // masses (0.036 vs 0.033 at 0.5, target 0.025) and structural agreement (0.257 vs 0.254, best of the run).
    let detail_len = 0.5;
    notes.push("layers: wide+long block-in, thinner+shorter on top (finest at 0.5 length)".into());

    PaintPlan {
        medium: a.medium.clone(),
        palette: a.palette.clone(),
        style: "legible".into(),
        armature,
        armature_face,
        armature_body,
        recede,
        semantic,
        families,
        commit_shadows,
        silhouette,
        silhouette_mode,
        sam,
        value_key,
        shadow_floor,
        detail_len,
        reserve,
        budget: Some(budget),
        from_scratch: a.from_scratch,
        // The planner cannot know a hair region; it is named on the command line or in a hand-written plan.
        hair_mask: None,
            infill: None,
            rigger: None,
            hotspot: None,
            technique: None,
            splatter: None,
            edge_pool: None,
            oldpaper: None,
            granulate: None,
            saliency: None,
            focus_detail: None,
            ladder_keep: None,
            leak: None,
            hdr: None,
            ridges: None,
            skip: None,
            fine_lines: None,
            impasto_map: None,
            weave: None,
            sheen: None,
            collide: None,
            analysis: None,
            analysis_insights: None,
            outcome: None,
            impasto: None,
            stroke_width: None,
            stroke_length: None,
            detail_restate: None,
            hdr_amount: None,
            levels: None,
            dry: None,
            wash_in: None,
            wet_cap: None,
            scumble: None,
            turbulence: None,
            accent: None,
            loose: None,
            glaze: None,
            deplete: None,
            broken: None,
            sfumato: None,
            glaze_pass: None,
            vary: None,
            comma: None,
            shade: None,
            imprimatura: None,
            body_vary: None,
            lost: None,
            crest: None,
            spatter: None,
            aerial: None,
            sink: None,
            tail: None,
            film: None,
            split: None,
            sag: None,
            gloss: None,
            tint: None,
            knife: None,
            hiding: None,
            twist: None,
            skim: None,
            hatch: None,
            verdaccio: None,
            dishes: None,
            lights: None,
            cross: None,
            dither: None,
            solid: None,
            flesh: None,
            span: None,
            edge: None,
            glint: None,
            smooth: None,
            bole: None,
            ramp: None,
            umbra: None,
            chalk: None,
            side: None,
            blend: None,
            skin: None,
            carnation: None,
            small: None,
            anchor: None,
            trim: None,
            small_min: None,
            contour: None,
            paper: None,
            erase: None,
            touch: None,
            velvet: None,
            paper_top: None,
            sweep: None,
            stump: None,
            dust: None,
            stump_edge: None,
            stump_feather: None,
            sweep_wave: None,
            paper_knee: None,
            rewet: None,
            chalk_tint: None,
            flat_brush: None,
            liner: None,
            tsuke: None,
            ma: None,
            kasure: None,
            nijimi: None,
            turn: None,
            yohaku: None,
            bone: None,
            boneless: None,
            hake: None,
            join: None,
            bone_wind: None,
            bone_dot: None,
            bone_scatter: None,
            bone_coherence: None,
            bone_end: None,
            sumi_focus: None,
            ma_focus: None,
            ma_flow: None,
            bone_ryaku: None,
            bone_fade: None,
            bokashi: None,
            velvet_dark: None,
            slant: None,
            grade: None,
            weight: None,
            clip: None,
            tiers: None,
            waver: None,
            nib: None,
            nibs: None,
            nib_tone: None,
            nib_white: None,
            nib_anchor: None,
            nib_hint: None,
            nib_face: None,
            nib_gap: None,
            nib_relief: None,
            nib_far: None,
            nib_occlude: None,
            nib_eyes: None,
            nib_fur: None,
            nib_lift: None,
            nib_contact: None,
            nib_figure: None,
            nib_object: None,
            nib_tuft: None,
            nib_close: None,
            nib_mass: None,
            nib_rim: None,
            nib_pool: None,
            nib_air: None,
            nib_people: None,
            nib_value: None,
            nib_fine_face: None,
            nib_scatter: None,
            nib_falloff: None,
            nib_depth: None,
            nib_lush: None,
            nib_glint: None,
            nib_wash: None,
            nib_wash_dark: None,
            nib_wash_detail: None,
            nib_wash_line: None,
            nib_wash_swell: None,
            nib_wash_air: None,
            nib_wash_soft: None,
            nib_wash_gamma: None,
            nib_wash_flat: None,
            sumi_dark: None,
            min_brush: None,
            pickup: None,
            bleed: None,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_reference_gets_more_value_key_than_punchy() {
        let base = Analysis { faces: 1, luma_stddev: 0.12, luma_mean: 0.5, medium: "watercolour".into(), palette: "image".into(), short_side: 512, long_side: 682, surface_white: true, structure: STRUCTURE_NORM, from_scratch: false };
        let flat = plan_from(&base);
        let punchy = plan_from(&Analysis { luma_stddev: 0.26, luma_mean: 0.5, ..base_like(&base) });
        // The key reads the reference's CONTRAST — but only for a medium that builds its values with paint.
        // A LUMINOUS one takes a mild fixed key instead, because its lights are the paper and keying them
        // harder only lifts the picture off it. Watercolour became luminous when it gained its washes, so the
        // contrast relationship is checked on an opaque medium and the fixed key is checked on watercolour.
        let opaque = |sd: f32| plan_from(&Analysis { luma_stddev: sd, medium: "oil-direct".into(), ..base_like(&base) }).value_key;
        assert!(opaque(0.12) > opaque(0.26), "a flat reference is keyed harder ({} vs {})", opaque(0.12), opaque(0.26));
        assert_eq!(flat.value_key, punchy.value_key, "a luminous medium takes the same mild key either way");
        assert!(flat.value_key > 0.0 && flat.value_key < 0.6, "and it is mild ({})", flat.value_key);
        assert_eq!(flat.armature_face, Some(300), "a detected face gets a fine focal armature (SAM precise focal)");
        assert_eq!(flat.armature_body, Some(210), "a subject gets a mid body armature (three-tier)");
        // No aerial veil by default (measured: it lifted mean luma +0.04 and hurt structural agreement with the
        // target); the background still goes a touch coarser than the base armature with a subject present.
        assert!(flat.recede == 0.0 && flat.armature < 120, "no default veil; background a touch coarser with a subject ({} / {})", flat.recede, flat.armature);
        assert_eq!(flat.reserve, Some(0.92), "watercolour reserves the paper only for true highlights");
    }

    #[test]
    fn no_face_means_uniform_armature() {
        let a = Analysis { faces: 0, luma_stddev: 0.2, luma_mean: 0.5, medium: "oil-direct".into(), palette: "zorn".into(), short_side: 512, long_side: 512, surface_white: false, structure: STRUCTURE_NORM, from_scratch: false };
        let plan = plan_from(&a);
        assert_eq!(plan.armature_face, None);
        assert_eq!(plan.armature_body, None, "no subject → no body tier");
        assert_eq!(plan.reserve, None, "an opaque medium does not reserve paper");
    }

    #[test]
    fn hjson_round_trips() {
        let a = Analysis { faces: 1, luma_stddev: 0.15, luma_mean: 0.5, medium: "watercolour".into(), palette: "image".into(), short_side: 512, long_side: 682, surface_white: true, structure: STRUCTURE_NORM, from_scratch: false };
        let plan = plan_from(&a);
        let parsed = PaintPlan::parse(&plan.to_hjson()).expect("parses");
        assert_eq!(parsed.armature, plan.armature);
        assert_eq!(parsed.armature_face, plan.armature_face);
        assert_eq!(parsed.medium, "watercolour");
        assert_eq!(parsed.oldpaper, None, "old paper is not a plan's default");
        let aged = PaintPlan { oldpaper: Some(true), ..plan };
        assert_eq!(PaintPlan::parse(&aged.to_hjson()).expect("parses").oldpaper, Some(true));
    }

    #[test]
    fn budget_scales_with_size_and_complexity() {
        let plain = Analysis { faces: 0, luma_stddev: 0.2, luma_mean: 0.5, medium: "oil-direct".into(), palette: "image".into(), short_side: 1024, long_side: 1024, surface_white: false, structure: STRUCTURE_NORM, from_scratch: false };
        let b1 = plan_from(&plain).budget.unwrap();
        assert_eq!(b1, 1024 * 1024 / 10, "the accepted density at the norm structure: area/10");
        let busy = Analysis { structure: STRUCTURE_NORM * 1.4, ..base_like(&plain) };
        let b2 = plan_from(&busy).budget.unwrap();
        assert!((b2 as f32 / b1 as f32 - 1.4).abs() < 0.01, "a busier picture earns proportionally more marks: {b2} vs {b1}");
        let flat = Analysis { structure: 0.01, ..base_like(&plain) };
        assert_eq!(plan_from(&flat).budget.unwrap(), b1, "a blank sheet never drops below the accepted density");
        let noisy = Analysis { structure: 10.0, ..base_like(&plain) };
        assert_eq!(plan_from(&noisy).budget.unwrap(), (b1 as f32 * 1.6) as usize, "noise does not explode the budget: 1.6× cap");
        let big = Analysis { short_side: 2048, long_side: 2048, ..base_like(&plain) };
        assert_eq!(plan_from(&big).budget.unwrap(), 2048 * 2048 / 10, "budget is a density: 4× the area, 4× the strokes (no cap)");
        let drawing = Analysis { medium: "durer".into(), ..base_like(&big) };
        assert_eq!(plan_from(&drawing).budget.unwrap(), 120000, "drawing media keep the 120k cap");
        // structure_of on a flat sheet is 0, on a checkerboard is large
        let flat_img = image::RgbImage::from_pixel(16, 16, image::Rgb([128, 128, 128]));
        assert!(structure_of(&flat_img) < 1e-6);
        let chk = image::RgbImage::from_fn(16, 16, |x, y| if (x + y) % 2 == 0 { image::Rgb([255, 255, 255]) } else { image::Rgb([0, 0, 0]) });
        assert!(structure_of(&chk) > 1.0);
    }

    #[test]
    fn a_new_painting_plans_reduced_tiers() {
        let a = Analysis { faces: 1, luma_stddev: 0.2, luma_mean: 0.5, medium: "oil-direct".into(), palette: "image".into(), short_side: 1024, long_side: 1024, surface_white: false, structure: STRUCTURE_NORM, from_scratch: true };
        let p = plan_from(&a);
        assert!(p.from_scratch);
        assert_eq!((p.armature, p.armature_body, p.armature_face), (384, Some(384), Some(768)), "the reduced tiers (nominal: the figure and face tiers are set at their own extent once found)");
        assert!(p.to_hjson().contains("new: true"));
        assert!(PaintPlan::parse(&p.to_hjson()).unwrap().from_scratch, "the plan round-trips `new`");
        let d = plan_from(&Analysis { from_scratch: false, ..base_like(&a) });
        assert!(d.armature < 384 && !d.from_scratch, "the default plan is untouched");
    }

    fn base_like(a: &Analysis) -> Analysis {
        Analysis { faces: a.faces, luma_stddev: a.luma_stddev, luma_mean: a.luma_mean, medium: a.medium.clone(), palette: a.palette.clone(), short_side: a.short_side, long_side: a.long_side, surface_white: a.surface_white, structure: a.structure, from_scratch: a.from_scratch }
    }
}
