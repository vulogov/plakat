//! The stroke score (RFC PAINT-1 §11.2) — the canonical artifact. An ordered, line-oriented, replayable record
//! of every stroke: its geometry, its pigment mixture (by NAME, so it is palette-portable), the stage it
//! belongs to, and its physics. The image `plakat paint` writes is one *rendering* of the score; the score is
//! authoritative under re-render.
//!
//! Guarantees this slice establishes: **replay** at any resolution (geometry scales; native-resolution replay
//! is byte-identical to the original paint, A4), and a round-trip through the text format. Checkpoints,
//! export, and timelapse build on this record in later slices.

use anyhow::{bail, Context, Result};

use crate::paint::canvas::Canvas;
use crate::paint::palette::Palette;
use crate::paint::pigment::Pigment;
use crate::paint::stroke::{BrushConfig, Stroke};

/// Reconstruct a `Palette` from recorded pigment definitions (`name`, sRGB). The `Palette`/`Pigment` types hold
/// `&'static` data, so the reconstructed set is leaked — bounded by the number of palettes a single CLI run
/// replays, which is small.
fn palette_from_defs(name: &str, defs: &[(String, [u8; 3])]) -> Palette {
    let pigs: Vec<Pigment> = defs
        .iter()
        .map(|(n, rgb)| Pigment { name: Box::leak(n.clone().into_boxed_str()), masstone: *rgb })
        .collect();
    Palette { name: Box::leak(name.to_string().into_boxed_str()), pigments: Box::leak(pigs.into_boxed_slice()) }
}

/// The score header — everything replay needs to reconstruct the canvas identically.
#[derive(Clone, Debug)]
pub struct ScoreHeader {
    pub version: u32,
    pub palette: String,
    /// The palette's pigment definitions (`name`, sRGB masstone), so a DERIVED palette (`--palette image`) — or
    /// any palette — replays without the binary knowing it. Empty = resolve `palette` by name (old scores).
    pub pigments: Vec<(String, [u8; 3])>,
    pub medium: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub tooth: f32,
    /// The toned ground (imprimatura), if any — replay must prime the canvas the same way the paint did.
    pub ground: Option<crate::paint::color::Srgb>,
    pub brush: BrushConfig,
    /// Wet-into-wet BLEED strength applied after painting (0 = dry media) — replay reproduces it.
    pub bleed: f32,
    /// Signed PIGMENT DIFFUSION (-1..+1) of the bleed — replay reproduces it (absent in older scores = 0).
    pub diffuse: f32,
    /// The PASS SCHEDULE the paint ran — every pass's stage name in order, laid or empty. Replay crosses each
    /// pass boundary (its tapered bleed, then drying) from this list, so a pass that laid no stroke (a fine pass
    /// on a picture that was already right, a tiny sheet) and the finish stages (contour, splatter…) that are
    /// not passes still reproduce the delivered image exactly. `None` = a score written before the schedule was
    /// recorded: the distinct stage names present stand in for it.
    pub stages: Option<Vec<String>>,
    /// INTER-PASS DRYING (0..1): how much the canvas dried between passes during painting. Recorded so a replay
    /// (at any resolution) dries at the same stage boundaries and reproduces the delivered image. 0 = never dried.
    pub dry: f32,
    /// BODY / opacity of the paint film (1 = opaque; lower = transparent) — replay must match.
    pub opacity: f32,
    /// IMPASTO relight strength at output (0 = flat) — the textured oil/knife look, replayed from height.
    pub impasto: f32,
    /// Material finish (§8.5) — the paint's own behaviour, replayed at output.
    pub chroma: f32,
    pub dry_shift: f32,
    pub granulate: f32,
    pub sheen: f32,
    /// EDGE POOLING — darken pigment at wash boundaries (watercolour edge-bloom), applied at output.
    pub edge_pool: f32,
    /// PAPER EDGE — fade to a deckled paper border at output (torn-paper vignette).
    pub paper_edge: f32,
    /// OLD PAPER — the picture printed on an aged sheet, at output (`--oldpaper`).
    pub old_paper: bool,
    /// FINISH GRADE (painting-safe tonal grade, applied at output): contrast, warmth (WB), clarity (local contrast).
    pub contrast: f32,
    pub warmth: f32,
    pub clarity: f32,
    /// LIFT — wipe removability (staining), applied during painting.
    pub lift: f32,
    /// The TRANSMITTANCE film (see `Canvas::with_transmittance`): a transparent medium's glazes multiply.
    pub transmittance: bool,
    /// THE FLUID STAGE (see `Canvas::flow`) run after every broad pass: (strength, radius px, rim, grain,
    /// selective). None = off.
    pub flow: Option<(f32, f32, f32, f32, f32)>,
    /// BACKRUNS (see `Canvas::bloom`) run after every wash plane of an aqua schedule: (chance, reach px, lip 0..1).
    /// None = off.
    pub bloom: Option<(f32, f32, f32)>,
    /// The aqua water's granulation (see `Canvas::set_grain_water`): the midtone bell's key and the wet gate's key
    /// (0 = the flat, pixel-sharp grain).
    pub grain_mid: f32,
    pub grain_wet: f32,
    /// The tide line's wetness barrier (`Canvas::set_rim_barrier`; the aqua schedule's `aqua_barrier`).
    pub barrier: f32,
    /// The dry-brush edge (`Canvas::set_dry_edge`; the aqua schedule's `aqua_ragged`) and the surface tension
    /// (`Canvas::set_rim_tension`; `aqua_tension`).
    pub ragged: f32,
    pub tension: f32,
    /// THE FLUID SOLVER (see `Canvas::fluid`) run after every wash plane of an aqua schedule: (strength, reach px,
    /// edge evaporation 0..1, pigment weight 0..1, pool mingling 0..1). None = off.
    pub fluid: Option<(f32, f32, f32, f32, f32)>,
    /// THE BRUSH'S CHARGE on the aqua schedule's washes (see `Canvas::fill_rings_charged`; `aqua_charge`). 0 = even.
    pub charge: f32,
    /// THE WATER ENGINE (see `Canvas::aqua_settle`; `MediumProfile::aqua_engine`): the session's drying run at the end
    /// of every aqua stage: (the water's reach px, edge evaporation 0..1, pigment weight 0..1, pool mingling 0..1,
    /// the pigment's viscosity 0..1 — `MediumProfile::aqua_visc`, the dry-brush boost 0..1 — `aqua_dry_boost`, the
    /// edge's bias 0..1 — `aqua_bias`; read with four to seven values, the trailing zeros not written). None = off
    /// (the polygon washes and `fluid`).
    pub engine: Option<(f32, f32, f32, f32, f32, f32, f32)>,
    /// THE PENCIL LAID DIRECT (see `Canvas::pencil_line`; `MediumProfile::aqua_line_density`, 7.16.0 set 15): every
    /// `line` record's densities laid direct to this power instead of by the brush. 0 = the brush (old scores).
    pub pencil: f32,
    /// THE DIRECTIONAL GRAIN (see `Canvas::aqua_settle_lit`; `MediumProfile::aqua_grain_dir`, 7.16.0 set 15): (the
    /// share 0..1, the light's direction in degrees) for the engine's drying. None = none (old scores).
    pub grain_dir: Option<(f32, f32)>,
    /// THE PENCIL'S HAND (see `Canvas::pencil_line_hand`; `MediumProfile::aqua_line_press`, `aqua_line_tooth`,
    /// `aqua_line_lift`, 7.16.0 set 16): (press, tooth, lift) 0..1 for every `line` record laid direct. None = the
    /// plain pencil (old scores).
    pub hand: Option<(f32, f32, f32)>,
    /// THE TECHNIQUE'S GRIP on the engine's drying (`Canvas::aqua_settle_full`; see `MediumProfile::aqua_tech`,
    /// 7.16.0 set 18): `(spread, seal, hold, sink, drift)` — the water's permeability up by the spread (wet-on-wet),
    /// the solver's flow into a dry cell down by the seal (wet-on-dry: the hard edge); set 20 (`MediumProfile::
    /// aqua_hold`, `aqua_sink`, `aqua_drift`): the concentration barrier, the thick charge's settling, the pigment's
    /// drift down the sheet. In the text the first two always, the rest up to the last nonzero (a set-18/19 score has
    /// two). None = none of them (old scores).
    pub tech: Option<(f32, f32, f32, f32, f32)>,
    /// CANVAS WEAVE (see `Finish::weave`). 0 = none.
    pub weave: f32,
    /// CANVAS SAG (see `Finish::sag`). 0 = none.
    pub sag: f32,
    /// GLOSS on the peaks (see `Finish::gloss`). 0 = none.
    pub gloss: f32,
    /// GLOSS MAP (see `Finish::gloss_map`): the sheen's roughness by the thickness. 0 = off.
    pub gloss_map: f32,
    /// CAVITY (see `Finish::cavity`): the relief's self-shading. 0 = off.
    pub cavity: f32,
    /// CAVITY FACE (see `Finish::cavity_face`): the cavity's share on the face. 1 = off.
    pub cavity_face: f32,
    /// PIGMENT HIDING (see `Canvas::with_hiding`). 0 = off.
    pub hiding: f32,
    /// WASH TONE (see `Finish::tone`): the sheet's mid-tones pulled toward the picture's mean at output. 0 = off.
    pub tone: f32,
    /// NIB WASH WATER (see `Canvas::seep`): the planes' ink seeps through the water at the end of their pass. 0 = off.
    pub water: f32,
    /// LIGHT (see `Finish::light`): the azimuth of the lamp the relief is relit under, degrees. 0 = the old lamp.
    pub light: f32,
    /// PHONG (see `Finish::phong`): the specular lobe on the relief's facets that face the lamp. 0 = off.
    pub phong: f32,
    /// SPREAD (see `Finish::spread`): the relief's slope read over a window of the picture's scale. 0 = off.
    pub spread: f32,
    /// PEAK (see `Finish::peak`): the sharp specular on the local crests of the relief. 0 = off.
    pub peak: f32,
    /// NIB WASH AGENT (see `Stroke::rasterize_agent`): the residual's gain for the marks with a profile. 0 = off.
    pub agent: f32,
    /// WET COLLISION (see `Canvas::collide`) after the broad passes: (strength, radius px). None = off.
    pub collide: Option<(f32, f32)>,
    /// The HOLD mask the collision respects, coarse: a `cols`-wide grid of cells at the score's aspect —
    /// 2 = held always (the faces), 1 = held on the fine passes only (the subject), 0 = free — so a replay
    /// keeps the same paint still. None = none. (`M cols rle…` record; runs cycle 0,1,2,0,1,2…)
    pub hold_mask: Option<(u32, Vec<u8>)>,
}

/// One recorded stroke.
#[derive(Clone, Debug)]
pub struct StrokeRecord {
    pub id: u32,
    /// A `wipe` (subtractive) stroke rather than a deposit. (Deposit-only in this slice; the flag is recorded.)
    pub wipe: bool,
    /// A WASH: an area fill, not a brush stroke. `spline` holds the mass's boundary RINGS (outer and holes),
    /// separated by a `[NaN, NaN]` point, filled even-odd with `mix` at `wet`; `w0` is the wet-edge FEATHER in
    /// px (0 = hard), `w1`/`streak`/`round` unused. A pool of colour — the watercolour mark a stroke cannot make.
    pub wash: bool,
    /// A TOUCH of the water engine (see `Canvas::aqua_inject`): `spline` the brush's dabs, `w0` the brush's radius in
    /// px, `wet` its water (1 flooded, 0 dry), `mix` the pigment's densities at a dab's centre — laid on the water,
    /// the shape the water's when the stage dries (the header's `engine`), `taper` the brush's sweep along a stroke
    /// (`Canvas::aqua_inject_with`). Neither a stroke nor a wash.
    pub inject: bool,
    pub stage: String,
    pub spline: Vec<[f32; 2]>,
    pub w0: f32,
    pub w1: f32,
    pub taper: f32,
    /// Pigment mixture as `(pigment_name, load_value)` for the non-zero pigments.
    pub mix: Vec<(String, f32)>,
    pub wet: f32,
    pub press: f32,
    /// This stroke's brush character (RFC brush vocabulary): bristle streak and cross-section roundness. Lets
    /// each stroke carry its OWN brush (a flat for the sky, a round for a face) and replay exactly.
    pub streak: f32,
    pub round: f32,
    /// This stroke's own PICKUP (dirty-brush drag), when it differs from the score's base brush — a DETAIL accent
    /// is laid nearly clean so it defines a feature instead of smearing the mass beneath. `None` = the header's
    /// `k_pickup` (old scores, and every non-detail stroke), so replay reproduces the paint exactly.
    pub pickup: Option<f32>,
    /// This stroke's own BRISTLE COUNT, when it differs from the score's base brush. A drawn line is made with
    /// a different tool from the one that paints the masses: a pen is ONE stiff point, a sumi brush a loaded
    /// tuft. Replaying a one-bristle pen line with the header's seven-bristle painting brush redrew every
    /// contour as a rake — the whole reason `pen-ink` and `line-and-wash` did not replay. `None` = the
    /// header's `bristles` (old scores, and every ordinary painted stroke).
    pub bristles: Option<usize>,
    /// This stroke's deposit rate, when it differs from the header's (a wash brush lays its film faster).
    pub kd: Option<f32>,
    /// This stroke's film cap (see `BrushConfig::film_cap`); `None` = no cap.
    pub cap: Option<f32>,
    /// A flat-ended wash stroke (see `BrushConfig::flat_ends`).
    pub flat: bool,
    /// A stroke whose brush held its charge (see `BrushConfig::hold_charge`); on a TOUCH (`inject`) THE SPOT DRIED
    /// under it before it (see `Canvas::aqua_dry_spot`; `MediumProfile::aqua_spot_dry`).
    pub hold: bool,
    /// This stroke's impasto viscosity (height per unit of paint), when it differs from the header's — the
    /// thickness mapped to the picture (see `PaintParams::impasto_map`). `None` = the header's.
    pub visc: Option<f32>,
    /// CREST: this stroke was dragged dry over the ridges of the layer beneath, catching only their tops
    /// (see `BrushConfig::crest`). `None` = an ordinary mark.
    pub crest: Option<f32>,
    /// The FILM OVER share of a glaze mark (see `BrushConfig::over`). `None` = none.
    pub over: Option<f32>,
    /// The BRISTLE SPLIT of a dry mark (see `BrushConfig::split`). `None` = none.
    pub split: Option<f32>,
    /// The SPLIT TINT of a split mark (see `BrushConfig::tint`). `None` = none.
    pub tint: Option<f32>,
    /// The KNIFE share of a knife mark (see `BrushConfig::knife`). `None` = a brush.
    pub knife: Option<f32>,
    /// The TWIST of a mark (see `BrushConfig::twist`). `None` = none.
    pub twist: Option<f32>,
    /// The INTENT colour (RFC PAINT-2 P0): the sRGB the stroke was aiming at — its seed's target in the
    /// pass's reference — recorded so another medium can re-solve its own charge for the same drawing.
    /// Ignored by `replay` (the mix is what was laid). `None` = not recorded (older scores).
    pub tgt: Option<[u8; 3]>,
    /// The mix at the mark's END (see `MediumProfile::ramp`): the brush's charge drifts from `mix` to this
    /// along the path, so one mark carries the picture's gradient. `None` = one mix the whole mark.
    pub mix2: Option<Vec<(String, f32)>>,
    /// A SMUDGE (the pastelist's finger; see `Stroke::smudge`): a `wipe` record with this set drags the paint
    /// along its path at this strength instead of scraping it — no pigment of its own, nothing removed.
    pub smudge: Option<f32>,
    /// A knife slice CUT HARD at its forward end (see `BrushConfig::liftoff`, `MediumProfile::blade_liftoff`): the
    /// painter stopped it at a colour's edge or a shape, not at the end of its paint — the replay lays the bead at
    /// the lift-off with the header's key. `None` = no bead.
    pub stop: Option<f32>,
    /// A WASH's wet edge PER RING POINT (see `Canvas::fill_rings_edge`, `MediumProfile::nib_wash_edge`): one
    /// feather in px for each point of `spline` (a separator's ignored); the bloom between two points runs from
    /// the one's to the other's. `None` = `w0` all round.
    pub edge: Option<Vec<f32>>,
    /// NIB WASH AGENT (see `Stroke::rasterize_agent`): the picture's target luma (linear) at each point of the
    /// spline — the mark gives its ink by the residual against the sheet under it. `None` = the plain mark.
    pub prof: Option<Vec<f32>>,
    /// THE MASKING FLUID (see `Canvas::set_resist_rings`; the aqua schedule's `aqua_resist`): a `wash` record
    /// with this set lays no colour — its rings are the lights reserved, refusing this share of every deposit
    /// and of the water laid after it; `w0` is the fluid's feathered edge in px. `None` = a wash. On a record that
    /// is NEITHER a wash nor a touch (7.16.0 set 14): THE LINES' RESIST — `spline` the skeleton's lines as
    /// NaN-separated polylines, `w0` the line's radius, the fluid laid along them (`Canvas::add_resist_lines`).
    pub resist: Option<f32>,
}

/// A coarse hold mask (`cols` wide, cells at the score's aspect): 2 where the `always` mask covers more
/// than a third of the cell, else 1 where the `fine` mask does, else 0. 64 columns on a 2048 sheet =
/// 32-px cells.
pub fn coarse_hold(always: Option<&[f32]>, fine: Option<&[f32]>, w: u32, h: u32, cols: u32) -> (u32, Vec<u8>) {
    let rows = ((h as f32 / w as f32) * cols as f32).round().max(1.0) as u32;
    let cell_n = (cols * rows) as usize;
    let mut sa = vec![0f32; cell_n];
    let mut sf = vec![0f32; cell_n];
    let mut cnt = vec![0u32; cell_n];
    for y in 0..h {
        let cy = (y * rows / h).min(rows - 1);
        for x in 0..w {
            let cx = (x * cols / w).min(cols - 1);
            let i = (cy * cols + cx) as usize;
            let q = (y * w + x) as usize;
            if always.map_or(false, |m| m[q] > 0.35) { sa[i] += 1.0; }
            if fine.map_or(false, |m| m[q] > 0.5) { sf[i] += 1.0; }
            cnt[i] += 1;
        }
    }
    let cells = (0..cell_n).map(|i| {
        if cnt[i] == 0 { 0 } else if sa[i] / cnt[i] as f32 > 0.33 { 2 } else if sf[i] / cnt[i] as f32 > 0.33 { 1 } else { 0 }
    }).collect();
    (cols, cells)
}

/// The coarse hold mask back at canvas size (nearest cell), as a 0/1 mask.
pub fn expand_hold(cols: u32, cells: &[u8], w: u32, h: u32) -> Vec<f32> {
    let rows = (cells.len() as u32 / cols.max(1)).max(1);
    let mut out = vec![0f32; (w * h) as usize];
    for y in 0..h {
        let cy = (y * rows / h).min(rows - 1);
        for x in 0..w {
            let cx = (x * cols / w).min(cols - 1);
            out[(y * w + x) as usize] = cells[(cy * cols + cx) as usize] as f32;
        }
    }
    out
}

impl ScoreHeader {
    /// The medium's output material finish (§8.5), for `Canvas::to_image_finished`.
    /// The canvas rendered with this header's finish — and, for the cavity's face share, its coarse hold mask.
    pub fn render(&self, canvas: &crate::paint::canvas::Canvas) -> image::RgbImage {
        canvas.to_image_finished_with(&self.finish(), if self.cavity_face < 1.0 { self.hold_mask.as_ref() } else { None })
    }

    pub fn finish(&self) -> crate::paint::canvas::Finish {
        crate::paint::canvas::Finish {
            impasto: self.impasto,
            relief_robust: self.brush.ridges > 0.0,
            chroma: self.chroma,
            dry_shift: self.dry_shift,
            granulate: self.granulate,
            sheen: self.sheen,
            edge_pool: self.edge_pool,
            paper_edge: self.paper_edge,
            old_paper: self.old_paper,
            plate_mark: crate::paint::medium::MediumProfile::by_name(&self.medium).is_some_and(|m| m.mark.engrave),
            weave: self.weave,
            sag: self.sag,
            gloss: self.gloss,
            gloss_map: self.gloss_map,
            cavity: self.cavity,
            cavity_face: self.cavity_face,
            contrast: self.contrast,
            warmth: self.warmth,
            clarity: self.clarity,
            tone: self.tone,
            light: self.light,
            phong: self.phong,
            spread: self.spread,
            peak: self.peak,
            age: self.brush.age,
            seed: self.seed,
        }
    }
}

/// The whole score: a header plus the ordered strokes.
#[derive(Clone, Debug)]
pub struct StrokeScore {
    pub header: ScoreHeader,
    pub strokes: Vec<StrokeRecord>,
}

fn fmt_f(v: f32) -> String {
    // EXACTLY round-trippable, compact, locale-free. (A wash's ring separator is NaN.)
    //
    // The score is the CANONICAL artefact (RFC PAINT-1): replaying it must reproduce the painting the run
    // saved. `{:.4}` did not — it TRUNCATED spline coordinates and mixture ratios, so every replayed stroke
    // landed a little off its original. Measured at 1024²: 0.67% RMSE on an oil (83% of pixels differing),
    // and 14.6% on pen-ink, whose thin marks amplify a fraction of a pixel into a visibly different line.
    // The engine's replay test could not see it — it replays the score held in MEMORY, and nothing wrote the
    // text and read it back (see `a_score_round_trips_through_text_exactly`).
    //
    // Rust's `{}` prints the SHORTEST decimal that parses back to the very same f32, so the text round-trips
    // bit-for-bit and stays about as compact as the old truncation. It never uses exponent notation, and a
    // stroke's numbers (pixel coordinates, widths, 0..1 ratios) are bounded, so lines cannot blow up.
    if v.is_nan() {
        return "NaN".into();
    }
    format!("{v}")
}

impl StrokeScore {
    /// Serialise to the line-oriented text format.
    pub fn to_text(&self) -> String {
        let h = &self.header;
        let b = &h.brush;
        let mut o = String::new();
        o.push_str("# plakat stroke score v1\n");
        let ground = h.ground.map(|g| format!(" ground={},{},{}", g[0], g[1], g[2])).unwrap_or_default();
        let ground = match &h.stages {
            Some(st) => format!("{ground} stages={}", st.iter().map(|x| x.replace(' ', "_")).collect::<Vec<_>>().join(",")),
            None => ground,
        };
        let ground = if h.transmittance { format!("{ground} trans=1") } else { ground };
        let ground = match h.flow { Some((s, r, m, g, e)) => format!("{ground} flow={},{},{},{},{}", fmt_f(s), fmt_f(r), fmt_f(m), fmt_f(g), fmt_f(e)), None => ground };
        let ground = match h.bloom { Some((b, r, l)) => format!("{ground} bloom={},{},{}", fmt_f(b), fmt_f(r), fmt_f(l)), None => ground };
        let ground = if h.grain_mid > 0.0 || h.grain_wet > 0.0 { format!("{ground} grainwater={},{}", fmt_f(h.grain_mid), fmt_f(h.grain_wet)) } else { ground };
        let ground = if h.barrier > 0.0 { format!("{ground} barrier={}", fmt_f(h.barrier)) } else { ground };
        let ground = if h.ragged > 0.0 { format!("{ground} ragged={}", fmt_f(h.ragged)) } else { ground };
        let ground = if h.tension > 0.0 { format!("{ground} tension={}", fmt_f(h.tension)) } else { ground };
        let ground = match h.fluid { Some((s, r, e, w, pl)) => format!("{ground} fluid={},{},{},{},{}", fmt_f(s), fmt_f(r), fmt_f(e), fmt_f(w), fmt_f(pl)), None => ground };
        let ground = if h.charge > 0.0 { format!("{ground} charge={}", fmt_f(h.charge)) } else { ground };
        let ground = match h.engine {
            Some((r, e, w, pl, v, b, x)) => {
                let tail = [v, b, x];
                let last = tail.iter().rposition(|t| *t > 0.0).map(|i| i + 1).unwrap_or(0);
                let mut s = format!("{ground} engine={},{},{},{}", fmt_f(r), fmt_f(e), fmt_f(w), fmt_f(pl));
                for t in &tail[..last] {
                    s.push_str(&format!(",{}", fmt_f(*t)));
                }
                s
            }
            None => ground,
        };
        let ground = if h.pencil > 0.0 { format!("{ground} pencil={}", fmt_f(h.pencil)) } else { ground };
        let ground = match h.grain_dir {
            Some((k, a)) if k > 0.0 => format!("{ground} graindir={},{}", fmt_f(k), fmt_f(a)),
            _ => ground,
        };
        let ground = match h.hand {
            Some((a, b, c)) if a > 0.0 || b > 0.0 || c > 0.0 => format!("{ground} hand={},{},{}", fmt_f(a), fmt_f(b), fmt_f(c)),
            _ => ground,
        };
        let ground = match h.tech {
            Some((a, b, c, d, e)) if a > 0.0 || b > 0.0 || c > 0.0 || d > 0.0 || e > 0.0 => {
                let tail = [c, d, e];
                let last = tail.iter().rposition(|t| *t > 0.0).map(|i| i + 1).unwrap_or(0);
                let mut s = format!("{ground} tech={},{}", fmt_f(a), fmt_f(b));
                for t in &tail[..last] {
                    s.push_str(&format!(",{}", fmt_f(*t)));
                }
                s
            }
            _ => ground,
        };
        o.push_str(&format!(
            "H palette={} medium={} seed={} size={}x{} tooth={} kd={} kp={} visc={} bristles={} loadmax={} streak={} round={} bleed={} diffuse={} dry={} opacity={} impasto={} chroma={} dryshift={} granulate={} sheen={} edgepool={} paperedge={} contrast={} warmth={} clarity={} lift={}{}{}\n",
            h.palette, h.medium, h.seed, h.width, h.height, fmt_f(h.tooth), fmt_f(b.k_deposit), fmt_f(b.k_pickup), fmt_f(b.viscosity), b.bristles, fmt_f(b.load_max), fmt_f(b.streak), fmt_f(b.round), fmt_f(h.bleed), fmt_f(h.diffuse), fmt_f(h.dry), fmt_f(h.opacity), fmt_f(h.impasto), fmt_f(h.chroma), fmt_f(h.dry_shift), fmt_f(h.granulate), fmt_f(h.sheen), fmt_f(h.edge_pool), fmt_f(h.paper_edge), fmt_f(h.contrast), fmt_f(h.warmth), fmt_f(h.clarity), fmt_f(h.lift), if b.ridges > 0.0 { format!(" ridges={}", fmt_f(b.ridges)) } else { String::new() } + &if b.skip > 0.0 { format!(" skip={}", fmt_f(b.skip)) } else { String::new() } + &if b.settle > 0.0 { format!(" settle={}", fmt_f(b.settle)) } else { String::new() } + &if b.fray > 0.0 { format!(" fray={}", fmt_f(b.fray)) } else { String::new() } + &if b.chisel > 0.0 { format!(" chisel={}", fmt_f(b.chisel)) } else { String::new() } + &if b.haste > 0.0 { format!(" haste={}", fmt_f(b.haste)) } else { String::new() } + &if b.sideload > 0.0 { format!(" sideload={}", fmt_f(b.sideload)) } else { String::new() } + &if b.chisel_twist > 0.0 { format!(" chiseltwist={}", fmt_f(b.chisel_twist)) } else { String::new() } + &if b.advect > 0.0 { format!(" advect={}", fmt_f(b.advect)) } else { String::new() } + &if b.film_flat > 0.0 { format!(" filmflat={}", fmt_f(b.film_flat)) } else { String::new() } + &if (b.deplete - 1.0).abs() > 1e-6 { format!(" deplete={}", fmt_f(b.deplete)) } else { String::new() } + &if b.comma > 0.0 { format!(" comma={}", fmt_f(b.comma)) } else { String::new() } + &if b.chalk > 0.0 { format!(" chalk={}", fmt_f(b.chalk)) } else { String::new() } + &if b.dust > 0.0 { format!(" dust={}", fmt_f(b.dust)) } else { String::new() } + &if b.rewet > 0.0 { format!(" rewet={}", fmt_f(b.rewet)) } else { String::new() } + &if b.tsuke > 0.0 { format!(" tsuke={}", fmt_f(b.tsuke)) } else { String::new() } + &if b.kasure > 0.0 { format!(" kasure={}", fmt_f(b.kasure)) } else { String::new() } + &if b.katsu > 0.0 { format!(" katsu={}", fmt_f(b.katsu)) } else { String::new() } + &if b.bevel > 0.0 { format!(" bevel={}", fmt_f(b.bevel)) } else { String::new() } + &if b.mix > 0.0 { format!(" bmix={}", fmt_f(b.mix)) } else { String::new() } + &if b.fade > 0.0 { format!(" bfade={}", fmt_f(b.fade)) } else { String::new() } + &if b.pile > 0.0 { format!(" bpile={}", fmt_f(b.pile)) } else { String::new() } + &if b.skid > 0.0 { format!(" bskid={}", fmt_f(b.skid)) } else { String::new() } + &if b.rim > 0.0 { format!(" brim={}", fmt_f(b.rim)) } else { String::new() } + &if b.slip > 0.0 { format!(" bslip={}", fmt_f(b.slip)) } else { String::new() } + &if b.bed > 0.0 { format!(" bbed={}", fmt_f(b.bed)) } else { String::new() } + &if b.drag > 0.0 { format!(" bdrag={}", fmt_f(b.drag)) } else { String::new() } + &if b.age > 0.0 { format!(" bage={}", fmt_f(b.age)) } else { String::new() } + &if b.bridge > 0.0 { format!(" bbridge={}", fmt_f(b.bridge)) } else { String::new() } + &if b.liftoff > 0.0 { format!(" blift={}", fmt_f(b.liftoff)) } else { String::new() } + &if b.broken > 0.0 { format!(" bbroken={}", fmt_f(b.broken)) } else { String::new() } + &if b.lip > 0.0 { format!(" blip={}", fmt_f(b.lip)) } else { String::new() } + &if b.marble > 0.0 { format!(" bmarble={}", fmt_f(b.marble)) } else { String::new() } + &if b.slump > 0.0 { format!(" bslump={}", fmt_f(b.slump)) } else { String::new() } + &if b.tilt > 0.0 { format!(" btilt={}", fmt_f(b.tilt)) } else { String::new() } + &if b.stiction > 0.0 { format!(" bstict={}", fmt_f(b.stiction)) } else { String::new() } + &if b.smear > 0.0 { format!(" bsmear={}", fmt_f(b.smear)) } else { String::new() } + &if b.heel > 0.0 { format!(" bheel={}", fmt_f(b.heel)) } else { String::new() } + &if b.dirty > 0.0 { format!(" bdirty={}", fmt_f(b.dirty)) } else { String::new() } + &if b.tear > 0.0 { format!(" btear={}", fmt_f(b.tear)) } else { String::new() } + &if b.nijimi > 0.0 { format!(" nijimi={}", fmt_f(b.nijimi)) } else { String::new() } + &if b.turn > 0.0 { format!(" turn={}", fmt_f(b.turn)) } else { String::new() } + &if b.join > 0.0 { format!(" join={}", fmt_f(b.join)) } else { String::new() } + &if b.waver > 0.0 { format!(" waver={}", fmt_f(b.waver)) } else { String::new() } + &if h.weave > 0.0 { format!(" weave={}", fmt_f(h.weave)) } else { String::new() } + &if h.sag > 0.0 { format!(" sag={}", fmt_f(h.sag)) } else { String::new() } + &if h.gloss > 0.0 { format!(" gloss={}", fmt_f(h.gloss)) } else { String::new() } + &if h.gloss_map > 0.0 { format!(" glossmap={}", fmt_f(h.gloss_map)) } else { String::new() } + &if h.cavity > 0.0 { format!(" cavity={}", fmt_f(h.cavity)) } else { String::new() } + &if h.cavity_face < 1.0 { format!(" cavityface={}", fmt_f(h.cavity_face)) } else { String::new() } + &if h.hiding > 0.0 { format!(" hiding={}", fmt_f(h.hiding)) } else { String::new() } + &if h.tone > 0.0 { format!(" tone={}", fmt_f(h.tone)) } else { String::new() } + &if h.water > 0.0 { format!(" water={}", fmt_f(h.water)) } else { String::new() } + &if h.agent > 0.0 { format!(" agent={}", fmt_f(h.agent)) } else { String::new() } + &if h.light > 0.0 { format!(" light={}", fmt_f(h.light)) } else { String::new() } + &if h.phong > 0.0 { format!(" phong={}", fmt_f(h.phong)) } else { String::new() } + &if h.spread > 0.0 { format!(" spread={}", fmt_f(h.spread)) } else { String::new() } + &if h.peak > 0.0 { format!(" peak={}", fmt_f(h.peak)) } else { String::new() } + (if h.old_paper { " oldpaper=1" } else { "" }) + &h.collide.map(|(a, b)| format!(" collide={},{}", fmt_f(a), fmt_f(b))).unwrap_or_default(), ground,
        ));
        // Pigment definitions (self-contained palette) — so a derived/any palette replays without the binary.
        for (name, rgb) in &h.pigments {
            o.push_str(&format!("P {} {} {} {}\n", name, rgb[0], rgb[1], rgb[2]));
        }
        if let Some((cols, cells)) = &h.hold_mask {
            // Run-length: alternating counts of 0s and 1s, starting with 0s.
            // Run-length over the levels 0,1,2 cycling (a zero-length run where a level is skipped).
            let mut runs: Vec<usize> = Vec::new();
            let mut cur = 0u8;
            let mut n = 0usize;
            for &c in cells {
                if c == cur { n += 1; } else {
                    runs.push(n);
                    cur = (cur + 1) % 3;
                    while cur != c { runs.push(0); cur = (cur + 1) % 3; }
                    n = 1;
                }
            }
            runs.push(n);
            o.push_str(&format!("M {} {}\n", cols, runs.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(",")));
        }
        for s in &self.strokes {
            let spline = s.spline.iter().map(|p| format!("{},{}", fmt_f(p[0]), fmt_f(p[1]))).collect::<Vec<_>>().join(";");
            let mix = s.mix.iter().map(|(n, v)| format!("{n}:{}", fmt_f(*v))).collect::<Vec<_>>().join(",");
            o.push_str(&format!(
                "{} {} stage={} spline={} w0={} w1={} taper={} mix={} wet={} press={} streak={} round={}{}{}\n",
                if s.inject { "J" } else if s.wash && s.wipe { "K" } else if s.wash { "A" } else if s.wipe { "W" } else { "S" },
                s.id,
                s.stage,
                spline,
                fmt_f(s.w0),
                fmt_f(s.w1),
                fmt_f(s.taper),
                mix,
                fmt_f(s.wet),
                fmt_f(s.press),
                fmt_f(s.streak),
                fmt_f(s.round),
                s.pickup.map(|v| format!(" pickup={}", fmt_f(v))).unwrap_or_default(),
                s.bristles.map(|v| format!(" bristles={v}")).unwrap_or_default() + &s.kd.map(|v| format!(" kd={}", fmt_f(v))).unwrap_or_default() + &s.cap.map(|v| format!(" cap={}", fmt_f(v))).unwrap_or_default() + if s.flat { " flat=1" } else { "" } + if s.hold { " hold=1" } else { "" } + &s.visc.map(|v| format!(" visc={}", fmt_f(v))).unwrap_or_default() + &s.crest.map(|v| format!(" crest={}", fmt_f(v))).unwrap_or_default() + &s.over.map(|v| format!(" over={}", fmt_f(v))).unwrap_or_default() + &s.split.map(|v| format!(" split={}", fmt_f(v))).unwrap_or_default() + &s.tint.map(|v| format!(" tint={}", fmt_f(v))).unwrap_or_default() + &s.knife.map(|v| format!(" knife={}", fmt_f(v))).unwrap_or_default() + &s.twist.map(|v| format!(" twist={}", fmt_f(v))).unwrap_or_default() + &s.tgt.map(|t| format!(" tgt={},{},{}", t[0], t[1], t[2])).unwrap_or_default() + &s.mix2.as_ref().map(|m| format!(" mix2={}", m.iter().map(|(n, v)| format!("{n}:{}", fmt_f(*v))).collect::<Vec<_>>().join(","))).unwrap_or_default() + &s.smudge.map(|v| format!(" smudge={}", fmt_f(v))).unwrap_or_default() + &s.stop.map(|v| format!(" stop={}", fmt_f(v))).unwrap_or_default() + &s.edge.as_ref().map(|e| format!(" edge={}", e.iter().map(|v| fmt_f(*v)).collect::<Vec<_>>().join(";"))).unwrap_or_default() + &s.prof.as_ref().map(|e| format!(" prof={}", e.iter().map(|v| fmt_f(*v)).collect::<Vec<_>>().join(";"))).unwrap_or_default() + &s.resist.map(|v| format!(" resist={}", fmt_f(v))).unwrap_or_default(),
            ));
        }
        o
    }

    /// Parse the text format back into a score.
    pub fn parse(text: &str) -> Result<Self> {
        let mut header: Option<ScoreHeader> = None;
        let mut strokes = Vec::new();
        let mut pigments: Vec<(String, [u8; 3])> = Vec::new();
        let mut hold_mask: Option<(u32, Vec<u8>)> = None;
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let tag = it.next().unwrap_or("");
            let kv = |it: std::str::SplitWhitespace| -> std::collections::HashMap<String, String> {
                it.filter_map(|t| t.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect()
            };
            match tag {
                "H" => {
                    let m = kv(it);
                    let get = |k: &str| m.get(k).cloned().unwrap_or_default();
                    let (w, h) = get("size").split_once('x').map(|(a, b)| (a.parse().unwrap_or(0), b.parse().unwrap_or(0))).unwrap_or((0, 0));
                    header = Some(ScoreHeader {
                        version: 1,
                        palette: get("palette"),
                        pigments: Vec::new(),
                        medium: get("medium"),
                        seed: get("seed").parse().unwrap_or(0),
                        width: w,
                        height: h,
                        tooth: get("tooth").parse().unwrap_or(0.85),
                        ground: m.get("ground").and_then(|g| {
                            let v: Vec<u8> = g.split(',').filter_map(|x| x.parse::<u8>().ok()).collect();
                            (v.len() == 3).then_some([v[0], v[1], v[2]])
                        }),
                        bleed: get("bleed").parse().unwrap_or(0.0),
                        diffuse: get("diffuse").parse().unwrap_or(0.0),
                        stages: m.get("stages").map(|v| v.split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect()),
                        dry: get("dry").parse().unwrap_or(0.0),
                        opacity: get("opacity").parse().unwrap_or(1.0),
                        impasto: get("impasto").parse().unwrap_or(0.0),
                        chroma: get("chroma").parse().unwrap_or(1.0),
                        dry_shift: get("dryshift").parse().unwrap_or(0.0),
                        granulate: get("granulate").parse().unwrap_or(0.0),
                        sheen: get("sheen").parse().unwrap_or(0.0),
                        edge_pool: get("edgepool").parse().unwrap_or(0.0),
                        paper_edge: get("paperedge").parse().unwrap_or(0.0),
                        old_paper: get("oldpaper") == "1",
                        contrast: get("contrast").parse().unwrap_or(1.0),
                        warmth: get("warmth").parse().unwrap_or(0.0),
                        clarity: get("clarity").parse().unwrap_or(0.0),
                        lift: get("lift").parse().unwrap_or(1.0),
                        transmittance: m.get("trans").map(|v| v == "1").unwrap_or(false),
                        weave: get("weave").parse().unwrap_or(0.0),
                        sag: get("sag").parse().unwrap_or(0.0),
                        gloss: get("gloss").parse().unwrap_or(0.0),
                        gloss_map: get("glossmap").parse().unwrap_or(0.0),
                        cavity: get("cavity").parse().unwrap_or(0.0),
                        cavity_face: get("cavityface").parse().unwrap_or(1.0),
                        hiding: get("hiding").parse().unwrap_or(0.0),
                        tone: get("tone").parse().unwrap_or(0.0),
                        water: get("water").parse().unwrap_or(0.0),
                        light: get("light").parse().unwrap_or(0.0),
                        phong: get("phong").parse().unwrap_or(0.0),
                        spread: get("spread").parse().unwrap_or(0.0),
                        peak: get("peak").parse().unwrap_or(0.0),
                        agent: get("agent").parse().unwrap_or(0.0),
                        collide: m.get("collide").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 2).then(|| (p[0], p[1])) }),
                        hold_mask: None,
                        flow: m.get("flow").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (3..=5).contains(&p.len()).then(|| (p[0], p[1], p[2], p.get(3).copied().unwrap_or(0.0), p.get(4).copied().unwrap_or(0.0))) }),
                        bloom: m.get("bloom").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (2..=3).contains(&p.len()).then(|| (p[0], p[1], p.get(2).copied().unwrap_or(0.0))) }),
                        grain_mid: m.get("grainwater").and_then(|v| v.split(',').next()).and_then(|x| x.parse().ok()).unwrap_or(0.0),
                        grain_wet: m.get("grainwater").and_then(|v| v.split(',').nth(1)).and_then(|x| x.parse().ok()).unwrap_or(0.0),
                        barrier: get("barrier").parse().unwrap_or(0.0),
                        ragged: get("ragged").parse().unwrap_or(0.0),
                        tension: get("tension").parse().unwrap_or(0.0),
                        fluid: m.get("fluid").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 4 || p.len() == 5).then(|| (p[0], p[1], p[2], p[3], p.get(4).copied().unwrap_or(0.0))) }),
                        charge: get("charge").parse().unwrap_or(0.0),
                        engine: m.get("engine").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (4..=7).contains(&p.len()).then(|| (p[0], p[1], p[2], p[3], p.get(4).copied().unwrap_or(0.0), p.get(5).copied().unwrap_or(0.0), p.get(6).copied().unwrap_or(0.0))) }),
                        pencil: get("pencil").parse().unwrap_or(0.0),
                        grain_dir: m.get("graindir").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 2).then(|| (p[0], p[1])) }),
                        hand: m.get("hand").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 3).then(|| (p[0], p[1], p[2])) }),
                        tech: m.get("tech").and_then(|v| { let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (2..=5).contains(&p.len()).then(|| (p[0], p[1], p.get(2).copied().unwrap_or(0.0), p.get(3).copied().unwrap_or(0.0), p.get(4).copied().unwrap_or(0.0))) }),
                        brush: BrushConfig {
                            k_deposit: get("kd").parse().unwrap_or(0.12),
                            k_pickup: get("kp").parse().unwrap_or(0.6),
                            viscosity: get("visc").parse().unwrap_or(1.0),
                            bristles: get("bristles").parse().unwrap_or(7),
                            load_max: get("loadmax").parse().unwrap_or(6.0),
                            streak: get("streak").parse().unwrap_or(0.6),
                            round: get("round").parse().unwrap_or(0.7),
                            film_cap: 0.0,
                            flat_ends: false,
                            hold_charge: false,
                            ridges: get("ridges").parse().unwrap_or(0.0),
                            skip: get("skip").parse().unwrap_or(0.0),
                            settle: get("settle").parse().unwrap_or(0.0),
                            fray: get("fray").parse().unwrap_or(0.0),
                            chisel: get("chisel").parse().unwrap_or(0.0),
                            haste: get("haste").parse().unwrap_or(0.0),
                            sideload: get("sideload").parse().unwrap_or(0.0),
                            chisel_twist: get("chiseltwist").parse().unwrap_or(0.0),
                            advect: get("advect").parse().unwrap_or(0.0),
                            film_flat: get("filmflat").parse().unwrap_or(0.0),
                            deplete: get("deplete").parse().unwrap_or(1.0),
                            comma: get("comma").parse().unwrap_or(0.0),
                            crest: 0.0,
                            over: 0.0,
                            split: 0.0,
                            tint: 0.0,
                            knife: 0.0,
                            twist: 0.0,
                            chalk: get("chalk").parse().unwrap_or(0.0),
                            dust: get("dust").parse().unwrap_or(0.0),
                            rewet: get("rewet").parse().unwrap_or(0.0),
                            tsuke: get("tsuke").parse().unwrap_or(0.0),
                            kasure: get("kasure").parse().unwrap_or(0.0),
                            katsu: get("katsu").parse().unwrap_or(0.0),
                            bevel: get("bevel").parse().unwrap_or(0.0),
                            mix: get("bmix").parse().unwrap_or(0.0),
                            fade: get("bfade").parse().unwrap_or(0.0),
                            pile: get("bpile").parse().unwrap_or(0.0),
                            skid: get("bskid").parse().unwrap_or(0.0),
                            rim: get("brim").parse().unwrap_or(0.0),
                            slip: get("bslip").parse().unwrap_or(0.0),
                            bed: get("bbed").parse().unwrap_or(0.0),
                            drag: get("bdrag").parse().unwrap_or(0.0),
                            age: get("bage").parse().unwrap_or(0.0),
                            bridge: get("bbridge").parse().unwrap_or(0.0),
                            liftoff: get("blift").parse().unwrap_or(0.0),
                            broken: get("bbroken").parse().unwrap_or(0.0),
                            lip: get("blip").parse().unwrap_or(0.0),
                            marble: get("bmarble").parse().unwrap_or(0.0),
                            slump: get("bslump").parse().unwrap_or(0.0),
                            tilt: get("btilt").parse().unwrap_or(0.0),
                            stiction: get("bstict").parse().unwrap_or(0.0),
                            smear: get("bsmear").parse().unwrap_or(0.0),
                            heel: get("bheel").parse().unwrap_or(0.0),
                            dirty: get("bdirty").parse().unwrap_or(0.0),
                            tear: get("btear").parse().unwrap_or(0.0),
                            nijimi: get("nijimi").parse().unwrap_or(0.0),
                            turn: get("turn").parse().unwrap_or(0.0),
                            join: get("join").parse().unwrap_or(0.0),
                            waver: get("waver").parse().unwrap_or(0.0),
                        },
                    });
                }
                "S" | "W" | "A" | "K" | "J" => {
                    let id: u32 = it.clone().next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let m = kv(it);
                    let get = |k: &str| m.get(k).cloned().unwrap_or_default();
                    let spline = get("spline")
                        .split(';')
                        .filter(|s| !s.is_empty())
                        .filter_map(|p| p.split_once(',').and_then(|(a, b)| Some([a.parse().ok()?, b.parse().ok()?])))
                        .collect::<Vec<[f32; 2]>>();
                    let mix = get("mix")
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .filter_map(|e| e.split_once(':').map(|(n, v)| (n.to_string(), v.parse().unwrap_or(0.0))))
                        .collect::<Vec<(String, f32)>>();
                    strokes.push(StrokeRecord {
                        id,
                        wipe: tag == "W" || tag == "K",
                        wash: tag == "A" || tag == "K",
                        inject: tag == "J",
                        stage: get("stage"),
                        spline,
                        w0: get("w0").parse().unwrap_or(1.0),
                        w1: get("w1").parse().unwrap_or(1.0),
                        taper: get("taper").parse().unwrap_or(0.0),
                        mix,
                        wet: get("wet").parse().unwrap_or(1.0),
                        press: get("press").parse().unwrap_or(1.0),
                        streak: get("streak").parse().unwrap_or(0.6),
                        round: get("round").parse().unwrap_or(0.7),
                        pickup: get("pickup").parse().ok(),
                        bristles: get("bristles").parse().ok(),
                        kd: get("kd").parse().ok(),
                        visc: get("visc").parse().ok(),
                        crest: get("crest").parse().ok(),
                        over: get("over").parse().ok(),
                        split: get("split").parse().ok(),
                        tint: get("tint").parse().ok(),
                        knife: get("knife").parse().ok(),
                        twist: get("twist").parse().ok(),
                        tgt: {
                            let p: Vec<u8> = get("tgt").split(',').filter_map(|x| x.parse::<u8>().ok()).collect();
                            (p.len() == 3).then(|| [p[0], p[1], p[2]])
                        },
                        smudge: get("smudge").parse().ok(),
                        stop: get("stop").parse().ok(),
                        edge: {
                            let e: Vec<f32> = get("edge").split(';').filter(|s| !s.is_empty()).filter_map(|v| v.parse().ok()).collect();
                            (!e.is_empty()).then_some(e)
                        },
                        prof: {
                            let e: Vec<f32> = get("prof").split(';').filter_map(|v| v.parse().ok()).collect();
                            (!e.is_empty()).then_some(e)
                        },
                        resist: get("resist").parse().ok(),
                        mix2: {
                            let m: Vec<(String, f32)> = get("mix2").split(',').filter(|s| !s.is_empty()).filter_map(|e| e.split_once(':').map(|(n, v)| (n.to_string(), v.parse().unwrap_or(0.0)))).collect();
                            (!m.is_empty()).then_some(m)
                        },
                        cap: get("cap").parse().ok(),
                        flat: get("flat") == "1",
                        hold: get("hold") == "1",
                    });
                }
                "P" => {
                    // Pigment definition: `P <name> <r> <g> <b>`.
                    let name = it.next().unwrap_or("").to_string();
                    let mut rgb = [0u8; 3];
                    for c in rgb.iter_mut() {
                        *c = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    }
                    if !name.is_empty() {
                        pigments.push((name, rgb));
                    }
                }
                "M" => {
                    let cols: u32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let runs: Vec<usize> = it.next().unwrap_or("").split(',').filter_map(|v| v.parse::<usize>().ok()).collect();
                    let mut cells: Vec<u8> = Vec::new();
                    let mut cur = 0u8;
                    for r in runs {
                        cells.extend(std::iter::repeat(cur).take(r));
                        cur = (cur + 1) % 3;
                    }
                    if cols > 0 && !cells.is_empty() {
                        hold_mask = Some((cols, cells));
                    }
                }
                "C" => { /* checkpoint markers ignored on parse in this slice */ }
                _ => bail!("stroke score: unrecognised record on line {}: {:?}", lineno + 1, tag),
            }
        }
        let mut header = header.context("stroke score: missing H header line")?;
        header.pigments = pigments;
        header.hold_mask = hold_mask;
        Ok(StrokeScore { header, strokes })
    }

    /// The distinct stage names in first-seen order — the basis for stage separations / a stages contact sheet.
    pub fn stages(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for s in &self.strokes {
            if !seen.iter().any(|x| x == &s.stage) {
                seen.push(s.stage.clone());
            }
        }
        seen
    }

    /// Replay only the strokes matching `keep`, in order, onto a fresh canvas at `out_w × out_h`. The building
    /// block for separations (one stage), a cumulative stages sheet, `--until N`, and frame emission.
    pub fn replay_filtered(&self, out_w: u32, out_h: u32, keep: impl Fn(&StrokeRecord) -> bool) -> Result<Canvas> {
        let (palette, brush, sx, sy, ss) = self.replay_setup(out_w, out_h)?;
        let n = palette.pigments.len();
        let mut canvas = match self.header.ground {
            Some(g) => Canvas::toned(out_w, out_h, palette, g, self.header.tooth),
            None => Canvas::white(out_w, out_h, palette, self.header.tooth),
        }
        .with_opacity(self.header.opacity)
        .with_cloth_seed(self.header.seed)
        .with_transmittance(self.header.transmittance)
        .with_dry_edge(self.header.ragged)
        .with_seep(self.header.water > 0.0)
        .with_hiding(self.header.hiding);
        let index_of = |name: &str| palette.pigments.iter().position(|p| p.name.eq_ignore_ascii_case(name));
        let full = |k: &dyn Fn(&StrokeRecord) -> bool| self.strokes.iter().all(|r| k(r));
        // The pass schedule the paint ran (recorded, or the stage names present). Each pass boundary bled
        // (wet-into-wet, tapering coarse → fine, see `painter::pass_bleed_taper`) and then dried; a record from
        // a later pass crosses every boundary up to it (empty passes included), a record from a finish stage
        // (contour, splatter…) first crosses the rest of them.
        let stages = self.pass_stages(&keep);
        let n_stages = stages.len();
        let mut cur = 0usize;
        let mut dried: Option<String> = None;
        for rec in &self.strokes {
            if !keep(rec) {
                continue;
            }
            let at = stages.iter().position(|st| st == &rec.stage).unwrap_or(n_stages);
            if at > cur {
                self.cross_passes(&mut canvas, cur, at, n_stages);
                cur = at;
            }
            // SUMI-E's registers (see `painter::bone_pass`, `painter::sumi_ink`): the painter dries the sheet
            // before the bone and again before the ink, so a wet-media final bleed sees only those marks wet.
            if (rec.stage == "bone" || rec.stage == "ink" || rec.stage == "nib") && dried.as_deref() != Some(rec.stage.as_str()) {
                canvas.dry(0.0);
                dried = Some(rec.stage.clone());
            }
            let path: Vec<[f32; 2]> = rec.spline.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
            // A TOUCH of the water engine (a `J` record; see `Canvas::aqua_inject`): the brush's dabs on the water.
            if rec.inject {
                let mut load = vec![0f32; n];
                for (name, val) in &rec.mix {
                    if let Some(i) = index_of(name) {
                        load[i] += val * ss;
                    }
                }
                // THE SPOT DRIED (a touch with `hold`; see `Canvas::aqua_dry_spot`): the sheet under it dried first.
                if rec.hold {
                    canvas.aqua_dry_spot(&path, rec.w0 * ss, rec.w1 * ss);
                }
                let (eb, ex) = self.header.engine.map(|e| (e.5, e.6)).unwrap_or((0.0, 0.0));
                canvas.aqua_inject_with(&path, rec.w0 * ss, rec.w1 * ss, rec.wet, &load, rec.press, eb, ex, rec.taper);
                continue;
            }
            // A CLIP record (`K`): strokes after it lay paint only inside its rings; a one-point record clears.
            if rec.wash && rec.wipe {
                canvas.set_clip_rings(&path);
                continue;
            }
            // THE LINES' RESIST (a record with `resist`, no wash, no touch; see `Canvas::add_resist_lines`): the fluid
            // along the skeleton's lines.
            if let (false, Some(r)) = (rec.wash, rec.resist) {
                canvas.add_resist_lines(&path, rec.w0 * ss, r);
                continue;
            }
            // THE MASKING FLUID (an `A` record with `resist`; see `Canvas::set_resist_rings`): the lights reserved.
            if let (true, Some(r)) = (rec.wash, rec.resist) {
                canvas.set_resist_rings(&path, r, rec.w0 * ss);
                continue;
            }
            if rec.wash {
                let mut load = vec![0f32; n];
                for (name, val) in &rec.mix {
                    if let Some(i) = index_of(name) {
                        load[i] += val * ss;
                    }
                }
                // (THE BRUSH'S CHARGE on the aqua schedule's washes, from the header; 0 is the even fill.)
                let charge = if self.header.charge > 0.0 && rec.stage.starts_with("aqua-") { self.header.charge } else { 0.0 };
                match &rec.edge {
                    Some(e) => canvas.fill_rings_charged(&path, &load, rec.wet, rec.w0 * ss, Some(&e.iter().map(|v| v * ss).collect::<Vec<f32>>()), charge),
                    None => canvas.fill_rings_charged(&path, &load, rec.wet, rec.w0 * ss, None, charge),
                }
                continue;
            }
            if let (true, Some(sm)) = (rec.wipe, rec.smudge) {
                // A SMUDGE (the finger; see `Stroke::smudge`): no pigment, the paint dragged along the path.
                let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load: Vec::new(), pressure: rec.press, wetness: rec.wet };
                s.smudge(&mut canvas, sm);
                continue;
            }
            if rec.wipe {
                // A subtractive WIPE stroke: scrape pigment back (strength recorded in `wet`). LIFT scales how
                // much comes off — oil lifts freely, a staining watercolour barely at all.
                let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load: Vec::new(), pressure: rec.press, wetness: rec.wet };
                s.wipe(&mut canvas, &brush, rec.wet * self.header.lift);
                continue;
            }
            // THE PENCIL LAID DIRECT (a `line` record of a score with `pencil`; see `Canvas::pencil_line`): the
            // densities of the record's mix (unscaled: a density is a density) to the header's power.
            if self.header.pencil > 0.0 && rec.stage == "line" {
                let mut dens = vec![0f32; n];
                for (name, val) in &rec.mix {
                    if let Some(i) = index_of(name) {
                        dens[i] += val * self.header.pencil;
                    }
                }
                canvas.pencil_line_hand(&path, rec.w0 * ss, rec.w1 * ss, &dens, self.header.hand.unwrap_or((0.0, 0.0, 0.0)));
                continue;
            }
            let mut load = vec![0f32; n];
            for (name, val) in &rec.mix {
                if let Some(i) = index_of(name) {
                    load[i] += val * ss;
                }
            }
            let load_end = rec.mix2.as_ref().map(|m2| {
                let mut le = vec![0f32; n];
                for (name, val) in m2 {
                    if let Some(i) = index_of(name) {
                        le[i] += val * ss;
                    }
                }
                le
            });
            let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load, pressure: rec.press, wetness: rec.wet };
            // This stroke's own brush character (a flat, a round, …) over the score's base brush physics.
            let sb = BrushConfig { streak: rec.streak, round: rec.round, k_pickup: rec.pickup.unwrap_or(brush.k_pickup), bristles: rec.bristles.unwrap_or(brush.bristles), k_deposit: rec.kd.unwrap_or(brush.k_deposit), film_cap: rec.cap.unwrap_or(0.0), flat_ends: rec.flat, hold_charge: rec.hold, viscosity: rec.visc.unwrap_or(brush.viscosity), crest: rec.crest.unwrap_or(0.0), over: rec.over.unwrap_or(0.0), split: rec.split.unwrap_or(0.0), tint: rec.tint.unwrap_or(0.0), knife: rec.knife.unwrap_or(0.0), twist: rec.twist.unwrap_or(0.0), liftoff: if rec.stop.is_some() { brush.liftoff } else { 0.0 }, ..brush };
            match &rec.prof {
                Some(pf) => s.rasterize_agent(&mut canvas, &sb, load_end.as_deref(), pf, self.header.agent),
                None => s.rasterize_ramp(&mut canvas, &sb, load_end.as_deref()),
            }
        }
        // The remaining pass boundaries (the last pass's own bleed, any trailing empty pass), then the light
        // final touch — only on a FULL replay; a truncated/filtered replay is an intermediate state, before the
        // fusion.
        if full(&keep) {
            self.cross_passes(&mut canvas, cur, n_stages, n_stages);
            if self.header.bleed > 0.0 {
                canvas.bleed_with(self.header.bleed * crate::paint::painter::FINAL_BLEED, self.header.diffuse);
            }
        }
        Ok(canvas)
    }

    /// The pass schedule to reproduce: the recorded one (see `ScoreHeader::stages`), else the distinct stage
    /// names present among the kept strokes.
    fn pass_stages(&self, keep: &dyn Fn(&StrokeRecord) -> bool) -> Vec<String> {
        match &self.header.stages {
            Some(st) => st.clone(),
            None => self.stage_schedule(keep),
        }
    }

    /// Cross the pass boundaries `from..to` of an `n`-pass schedule exactly as the painter does after each pass:
    /// the pass's tapered wet-into-wet bleed, then — when another pass follows — the inter-pass drying.
    pub(crate) fn cross_passes(&self, canvas: &mut Canvas, from: usize, to: usize, n: usize) {
        for idx in from..to.min(n) {
            // THE WATER ENGINE (see `Canvas::aqua_settle`): the session's water dried and its pigment laid, before
            // anything else touches the sheet.
            if let Some((er, ee, ew, ep, ev, _, _)) = self.header.engine {
                if crate::paint::painter::aqua_stage(self.header.stages.as_deref(), idx) {
                    let (gd, ga) = self.header.grain_dir.unwrap_or((0.0, 0.0));
                    let (ts, tl, th, tk, td) = self.header.tech.unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0));
                    canvas.aqua_settle_full(er, ee, ew, ep, ev, gd, ga, ts, tl, th, tk, td);
                }
            }
            if self.header.bleed > 0.0 {
                canvas.bleed_with(self.header.bleed * crate::paint::painter::stage_bleed_taper(self.header.stages.as_deref(), idx, n), self.header.diffuse);
            }
            // NIB WASH WATER: the planes' ink seeps through the water at the end of their pass.
            if self.header.water > 0.0 && crate::paint::painter::water_stage(self.header.stages.as_deref(), idx) {
                canvas.seep(self.header.water);
            }
            // The aqua schedule's water: the paper's tooth at the sheet's scale for the granulation.
            if crate::paint::painter::aqua_stage(self.header.stages.as_deref(), idx) {
                canvas.set_grain_scale(canvas.w.min(canvas.h) as f32 / 512.0);
                canvas.set_grain_water(self.header.grain_mid, self.header.grain_wet);
                canvas.set_rim_barrier(self.header.barrier);
                canvas.set_rim_tension(self.header.tension);
            }
            // BACKRUNS (see `Canvas::bloom`): before the water's own diffusion, on the wash planes alone.
            if let Some((bs, br, bl)) = self.header.bloom {
                if crate::paint::painter::aqua_stage(self.header.stages.as_deref(), idx) {
                    canvas.bloom(bs, br, self.header.seed ^ (idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15), bl);
                }
            }
            // THE FLUID SOLVER (see `Canvas::fluid`): the plane's water simulated, on the wash planes alone, before
            // the flow's blur (which yields to it by the key).
            if let Some((fs, fr, fe, fw, fp)) = self.header.fluid {
                if crate::paint::painter::aqua_stage(self.header.stages.as_deref(), idx) {
                    canvas.fluid(fs, fr, fe, fw, fp);
                }
            }
            if let Some((s, r, m, g, e)) = self.header.flow {
                if let Some(t) = crate::paint::painter::stage_flow_taper(self.header.stages.as_deref(), idx, n) {
                    canvas.flow(s * t, r, m, g, e);
                }
            }
            if let Some((s, r)) = self.header.collide {
                let hold = self.header.hold_mask.as_ref().map(|(cols, cells)| expand_hold(*cols, cells, canvas.w, canvas.h));
                let (st, at) = crate::paint::painter::collide_schedule(s, idx, n);
                if st > 0.0 {
                    canvas.collide(st, r, hold.as_deref(), at);
                }
            }
            if self.header.dry > 0.0 && idx + 1 < n {
                canvas.dry(1.0 - self.header.dry);
            }
            // The next pass starts: a wash brush's film cap counts from here (a no-op for every other brush).
            canvas.mark_film();
        }
    }

    /// The ordered list of distinct stage names among the kept strokes — the pass schedule the paint ran.
    fn stage_schedule(&self, keep: &dyn Fn(&StrokeRecord) -> bool) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for rec in self.strokes.iter().filter(|r| keep(r)) {
            if out.last().map(|s| s != &rec.stage).unwrap_or(true) {
                out.push(rec.stage.clone());
            }
        }
        out
    }

    /// Replay the whole score in order, emitting a canvas frame every `every` strokes (plus the final frame).
    /// A single O(n) forward pass, so a long timelapse is linear rather than quadratic.
    pub fn replay_frames(&self, out_w: u32, out_h: u32, every: usize) -> Result<Vec<Canvas>> {
        let (palette, brush, sx, sy, ss) = self.replay_setup(out_w, out_h)?;
        let n = palette.pigments.len();
        let mut canvas = match self.header.ground {
            Some(g) => Canvas::toned(out_w, out_h, palette, g, self.header.tooth),
            None => Canvas::white(out_w, out_h, palette, self.header.tooth),
        }
        .with_opacity(self.header.opacity)
        .with_cloth_seed(self.header.seed)
        .with_transmittance(self.header.transmittance)
        .with_dry_edge(self.header.ragged)
        .with_seep(self.header.water > 0.0)
        .with_hiding(self.header.hiding);
        let index_of = |name: &str| palette.pigments.iter().position(|p| p.name.eq_ignore_ascii_case(name));
        let every = every.max(1);
        let mut frames = Vec::new();
        let mut laid = 0usize;
        let stages = self.pass_stages(&|_| true);
        let n_stages = stages.len();
        let mut cur = 0usize;
        let mut dried: Option<String> = None;
        for rec in &self.strokes {
            let at = stages.iter().position(|st| st == &rec.stage).unwrap_or(n_stages);
            if at > cur {
                self.cross_passes(&mut canvas, cur, at, n_stages);
                cur = at;
            }
            // SUMI-E's registers (see `painter::bone_pass`, `painter::sumi_ink`): the painter dries the sheet
            // before the bone and again before the ink, so a wet-media final bleed sees only those marks wet.
            if (rec.stage == "bone" || rec.stage == "ink" || rec.stage == "nib") && dried.as_deref() != Some(rec.stage.as_str()) {
                canvas.dry(0.0);
                dried = Some(rec.stage.clone());
            }
            if let (true, Some(sm)) = (rec.wipe, rec.smudge) {
                let path: Vec<[f32; 2]> = rec.spline.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
                let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load: Vec::new(), pressure: rec.press, wetness: rec.wet };
                s.smudge(&mut canvas, sm);
                laid += 1;
                if laid % every == 0 {
                    frames.push(canvas.snapshot());
                }
                continue;
            }
            if !rec.wipe {
                let mut load = vec![0f32; n];
                for (name, val) in &rec.mix {
                    if let Some(i) = index_of(name) {
                        load[i] += val * ss;
                    }
                }
                let path: Vec<[f32; 2]> = rec.spline.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
                if rec.inject {
                    if rec.hold {
                        canvas.aqua_dry_spot(&path, rec.w0 * ss, rec.w1 * ss);
                    }
                    let (eb, ex) = self.header.engine.map(|e| (e.5, e.6)).unwrap_or((0.0, 0.0));
                canvas.aqua_inject_with(&path, rec.w0 * ss, rec.w1 * ss, rec.wet, &load, rec.press, eb, ex, rec.taper);
                    laid += 1;
                    if laid % every == 0 {
                        frames.push(canvas.snapshot());
                    }
                    continue;
                }
                if rec.wash && rec.wipe {
                    canvas.set_clip_rings(&path);
                    continue;
                }
                if let (false, Some(r)) = (rec.wash, rec.resist) {
                    canvas.add_resist_lines(&path, rec.w0 * ss, r);
                    continue;
                }
                if let (true, Some(r)) = (rec.wash, rec.resist) {
                    canvas.set_resist_rings(&path, r, rec.w0 * ss);
                    continue;
                }
                if rec.wash {
                    let charge = if self.header.charge > 0.0 && rec.stage.starts_with("aqua-") { self.header.charge } else { 0.0 };
                    match &rec.edge {
                        Some(e) => canvas.fill_rings_charged(&path, &load, rec.wet, rec.w0 * ss, Some(&e.iter().map(|v| v * ss).collect::<Vec<f32>>()), charge),
                        None => canvas.fill_rings_charged(&path, &load, rec.wet, rec.w0 * ss, None, charge),
                    }
                    laid += 1;
                    if laid % every == 0 {
                        frames.push(canvas.snapshot());
                    }
                    continue;
                }
                if self.header.pencil > 0.0 && rec.stage == "line" {
                    let mut dens = vec![0f32; n];
                    for (name, val) in &rec.mix {
                        if let Some(i) = index_of(name) {
                            dens[i] += val * self.header.pencil;
                        }
                    }
                    canvas.pencil_line_hand(&path, rec.w0 * ss, rec.w1 * ss, &dens, self.header.hand.unwrap_or((0.0, 0.0, 0.0)));
                    laid += 1;
                    if laid % every == 0 {
                        frames.push(canvas.snapshot());
                    }
                    continue;
                }
                let load_end = rec.mix2.as_ref().map(|m2| {
                    let mut le = vec![0f32; n];
                    for (name, val) in m2 {
                        if let Some(i) = index_of(name) {
                            le[i] += val * ss;
                        }
                    }
                    le
                });
                let s = Stroke { path, width0: rec.w0 * ss, width1: rec.w1 * ss, load, pressure: rec.press, wetness: rec.wet };
                let sb = BrushConfig { streak: rec.streak, round: rec.round, k_pickup: rec.pickup.unwrap_or(brush.k_pickup), bristles: rec.bristles.unwrap_or(brush.bristles), k_deposit: rec.kd.unwrap_or(brush.k_deposit), film_cap: rec.cap.unwrap_or(0.0), flat_ends: rec.flat, hold_charge: rec.hold, viscosity: rec.visc.unwrap_or(brush.viscosity), crest: rec.crest.unwrap_or(0.0), over: rec.over.unwrap_or(0.0), split: rec.split.unwrap_or(0.0), tint: rec.tint.unwrap_or(0.0), knife: rec.knife.unwrap_or(0.0), twist: rec.twist.unwrap_or(0.0), liftoff: if rec.stop.is_some() { brush.liftoff } else { 0.0 }, ..brush };
                match &rec.prof {
                Some(pf) => s.rasterize_agent(&mut canvas, &sb, load_end.as_deref(), pf, self.header.agent),
                None => s.rasterize_ramp(&mut canvas, &sb, load_end.as_deref()),
            }
                laid += 1;
                if laid % every == 0 {
                    frames.push(canvas.snapshot());
                }
            }
        }
        self.cross_passes(&mut canvas, cur, n_stages, n_stages);
        if self.header.bleed > 0.0 {
            canvas.bleed_with(self.header.bleed * crate::paint::painter::FINAL_BLEED, self.header.diffuse); // the light final touch
        }
        frames.push(canvas); // always end on the finished painting
        Ok(frames)
    }

    /// Resolve the palette, scaled brush, and scale factors shared by every replay path.
    fn replay_setup(&self, out_w: u32, out_h: u32) -> Result<(Palette, BrushConfig, f32, f32, f32)> {
        // Prefer the score's own pigment definitions (self-contained — works for a derived `image` palette or any
        // palette the running binary doesn't know); fall back to a built-in by name for old, pigment-less scores.
        let palette = if !self.header.pigments.is_empty() {
            palette_from_defs(&self.header.palette, &self.header.pigments)
        } else {
            Palette::by_name(&self.header.palette).with_context(|| format!("stroke score: unknown palette {:?}", self.header.palette))?
        };
        let sx = out_w as f32 / self.header.width.max(1) as f32;
        let sy = out_h as f32 / self.header.height.max(1) as f32;
        let ss = (sx * sy).sqrt();
        let mut brush = self.header.brush;
        brush.k_deposit /= ss;
        brush.load_max *= ss;
        Ok((palette, brush, sx, sy, ss))
    }

    /// Replay the score onto a fresh canvas at `out_w × out_h`. Geometry (and the brush width + load charge)
    /// scale from the score's native size, so the same score renders at any resolution with no upscaler. At
    /// the native size this reproduces the original paint byte-for-byte.
    pub fn replay(&self, out_w: u32, out_h: u32) -> Result<Canvas> {
        self.replay_filtered(out_w, out_h, |_| true)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    fn sample() -> StrokeScore {
        StrokeScore {
            header: ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "oil-direct".into(), seed: 42, width: 64, height: 48, tooth: 0.85, ground: None, brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: None, dry: 0.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0, transmittance: false, flow: None, bloom: None, grain_mid: 0.0, grain_wet: 0.0, barrier: 0.0, ragged: 0.0, tension: 0.0, fluid: None, charge: 0.0, engine: None, pencil: 0.0, grain_dir: None, hand: None, tech: None, weave: 0.0, sag: 0.0, gloss: 0.0, gloss_map: 0.0, cavity: 0.0, cavity_face: 1.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None },
            strokes: vec![
                StrokeRecord { id: 1, wipe: false, wash: false, inject: false, stage: "shadow-mass".into(), spline: vec![[5.0, 20.0], [30.0, 22.0], [50.0, 20.0]], w0: 8.0, w1: 5.0, taper: 0.4, mix: vec![("cadmium-red".into(), 3.0), ("ivory-black".into(), 1.0)], wet: 1.0, press: 0.9, streak: 0.6, round: 0.7, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None },
                StrokeRecord { id: 2, wipe: false, wash: false, inject: false, stage: "light-mass".into(), spline: vec![[10.0, 10.0], [40.0, 12.0]], w0: 6.0, w1: 4.0, taper: 0.3, mix: vec![("yellow-ochre".into(), 2.0), ("titanium-white".into(), 3.0)], wet: 1.0, press: 1.0, streak: 0.6, round: 0.7, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None },
            ],
        }
    }

    #[test]
    fn old_paper_round_trips_and_gives_an_engraving_its_plate_mark() {
        let mut s = sample();
        assert!(!StrokeScore::parse(&s.to_text()).unwrap().header.old_paper, "off unless asked for");
        s.header.old_paper = true;
        let parsed = StrokeScore::parse(&s.to_text()).expect("parses");
        assert!(parsed.header.old_paper);
        assert!(parsed.header.finish().old_paper && !parsed.header.finish().plate_mark, "a painting gets the sheet, not a plate mark");
        s.header.medium = "durer".into();
        assert!(StrokeScore::parse(&s.to_text()).unwrap().header.finish().plate_mark, "an engraving gets both");
    }

    #[test]
    fn text_round_trips() {
        let s = sample();
        let parsed = StrokeScore::parse(&s.to_text()).expect("parses");
        assert_eq!(parsed.header.palette, "zorn");
        assert_eq!(parsed.header.width, 64);
        assert_eq!(parsed.strokes.len(), 2);
        assert_eq!(parsed.strokes[0].stage, "shadow-mass");
        assert_eq!(parsed.strokes[0].spline.len(), 3);
        assert_eq!(parsed.strokes[1].mix[0].0, "yellow-ochre");
    }

    #[test]
    fn a_strokes_own_viscosity_and_the_weave_round_trip() {
        let mut s = sample();
        s.strokes[0].visc = Some(1.75);
        s.strokes[0].tgt = Some([200, 30, 40]);
        s.strokes[1].mix2 = Some(vec![("ivory-black".into(), 2.5), ("yellow-ochre".into(), 0.5)]);
        s.header.weave = 0.4;
        s.header.collide = Some((0.5, 6.0));
        s.header.hold_mask = Some((8, vec![0, 0, 2, 2, 0, 1, 1, 0, 0, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        let parsed = StrokeScore::parse(&s.to_text()).expect("parses");
        assert_eq!(parsed.strokes[0].visc, Some(1.75));
        assert_eq!(parsed.strokes[0].tgt, Some([200, 30, 40]));
        assert_eq!(parsed.strokes[1].tgt, None);
        assert_eq!(parsed.strokes[1].mix2, s.strokes[1].mix2, "the end mix round-trips");
        assert_eq!(parsed.strokes[0].mix2, None);
        // The end mix changes what is laid: the ramped record's replay differs from the one-colour one.
        let mut one = s.clone();
        one.strokes[1].mix2 = None;
        assert_ne!(s.replay(64, 48).unwrap().to_image().into_raw(), one.replay(64, 48).unwrap().to_image().into_raw());
        assert_eq!(parsed.header.collide, Some((0.5, 6.0)));
        assert_eq!(parsed.header.hold_mask, s.header.hold_mask);
        // The intent colour does not change what is laid: replay with and without it is byte-identical.
        let mut bare = s.clone();
        bare.strokes[0].tgt = None;
        assert_eq!(s.replay(64, 48).unwrap().to_image().into_raw(), bare.replay(64, 48).unwrap().to_image().into_raw());
        assert_eq!(parsed.strokes[1].visc, None);
        assert_eq!(parsed.header.weave, 0.4);
        // And a mapped stroke lays a different height from an unmapped one at replay.
        let hi = s.replay(64, 48).unwrap();
        let mut flat = s.clone();
        flat.strokes[0].visc = None;
        let lo = flat.replay(64, 48).unwrap();
        let sum = |c: &crate::paint::canvas::Canvas| c.height.iter().sum::<f32>();
        assert!(sum(&hi) > sum(&lo) * 1.2, "a thicker stroke stands higher: {} vs {}", sum(&hi), sum(&lo));
    }

    #[test]
    fn the_pencil_the_dried_spot_and_the_grains_direction_round_trip() {
        // Set 15: the header's pencil and grain direction and a touch's dried spot in the text and back; the pencil's
        // line darker on the sheet than the brush's; the replay exact from the text.
        let pal = crate::paint::palette::LIMITED_LANDSCAPE;
        let mut s = StrokeScore { header: ScoreHeader { version: 1, palette: pal.name.into(), pigments: pal.pigments.iter().map(|p| (p.name.to_string(), p.masstone)).collect(), medium: "watercolour".into(), seed: 3, width: 64, height: 48, tooth: 0.85, ground: None, brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: Some(vec!["line".into(), "aqua-1".into()]), dry: 1.0, opacity: 0.45, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0, transmittance: true, flow: None, bloom: None, grain_mid: 0.0, grain_wet: 0.0, barrier: 0.0, ragged: 0.0, tension: 0.0, fluid: None, charge: 0.0, engine: Some((6.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0)), pencil: 0.0, grain_dir: None, hand: None, tech: None, weave: 0.0, sag: 0.0, gloss: 0.0, gloss_map: 0.0, cavity: 0.0, cavity_face: 1.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None }, strokes: Vec::new() };
        let line = StrokeRecord { id: 1, wipe: false, wash: false, inject: false, stage: "line".into(), spline: vec![[10.0, 4.0], [10.0, 44.0]], w0: 1.5, w1: 1.0, taper: 0.0, mix: vec![("ultramarine-blue".into(), 1.0)], wet: 0.0, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: Some(1), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        let touch = StrokeRecord { id: 2, wipe: false, wash: false, inject: true, stage: "aqua-1".into(), spline: vec![[40.0, 24.0]], w0: 8.0, w1: 0.0, taper: 0.0, mix: vec![("cadmium-red".into(), 0.5)], wet: 0.0, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: true, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        s.strokes = vec![line, touch];
        let brush = s.replay(64, 48).unwrap();
        s.header.pencil = 2.0;
        s.header.grain_dir = Some((0.5, 45.0));
        let text = s.to_text();
        assert!(text.contains(" pencil=2") && text.contains(" graindir=0.5,45") && text.contains(" hold=1"), "the tokens: {}", text.lines().take(1).chain(text.lines().skip_while(|l| !l.starts_with('J')).take(1)).collect::<Vec<_>>().join(" | "));
        let parsed = StrokeScore::parse(&text).unwrap();
        assert_eq!(parsed.header.pencil, 2.0);
        assert_eq!(parsed.header.grain_dir, Some((0.5, 45.0)));
        assert!(parsed.strokes[1].hold && parsed.strokes[1].inject, "the touch's dried spot");
        let direct = s.replay(64, 48).unwrap();
        let lum = |c: &Canvas, x: u32, y: u32| crate::paint::color::linear_luma(crate::paint::color::srgb_to_linear(c.color_at(x, y)));
        assert!(lum(&direct, 10, 24) < lum(&brush, 10, 24) - 0.2, "the pencil's line darker than the brush's: {} vs {}", lum(&direct, 10, 24), lum(&brush, 10, 24));
        assert_eq!(parsed.replay(64, 48).unwrap().to_image().into_raw(), direct.to_image().into_raw(), "replays byte-exact from the text");
    }

    #[test]
    fn the_pencils_hand_round_trips_tapers_lifts_and_skips_on_the_tooth() {
        // Set 16: the header's hand in the text and back; the pressure thins the line's ends against its middle;
        // the lift leaves fewer pixels of a long straight line; the tooth leaves fewer pixels of a light line; each
        // replays exact from the text; no hand = the set-15 pencil byte-identical.
        let pal = crate::paint::palette::LIMITED_LANDSCAPE;
        let mut s = StrokeScore { header: ScoreHeader { version: 1, palette: pal.name.into(), pigments: pal.pigments.iter().map(|p| (p.name.to_string(), p.masstone)).collect(), medium: "watercolour".into(), seed: 3, width: 64, height: 96, tooth: 0.85, ground: None, brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: Some(vec!["line".into()]), dry: 1.0, opacity: 0.45, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0, transmittance: true, flow: None, bloom: None, grain_mid: 0.0, grain_wet: 0.0, barrier: 0.0, ragged: 0.0, tension: 0.0, fluid: None, charge: 0.0, engine: None, pencil: 2.0, grain_dir: None, hand: None, tech: None, weave: 0.0, sag: 0.0, gloss: 0.0, gloss_map: 0.0, cavity: 0.0, cavity_face: 1.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None }, strokes: Vec::new() };
        let line = |dens: f32| StrokeRecord { id: 1, wipe: false, wash: false, inject: false, stage: "line".into(), spline: vec![[20.0, 4.0], [20.0, 92.0]], w0: 2.0, w1: 2.0, taper: 0.0, mix: vec![("ultramarine-blue".into(), dens)], wet: 0.0, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: Some(1), kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        s.strokes = vec![line(1.0)];
        let plain_c = s.replay(64, 96).unwrap();
        let plain = plain_c.to_image().into_raw();
        s.header.hand = Some((0.0, 0.0, 0.0));
        assert!(!s.to_text().contains(" hand="), "a hand of nothing is not written");
        assert_eq!(s.replay(64, 96).unwrap().to_image().into_raw(), plain, "and lays the plain pencil");
        let drawn = |c: &Canvas, y0: u32, y1: u32| -> usize { let mut n = 0; for y in y0..y1 { for x in 0..64 { if c.saturation_at(x, y) > 1e-6 { n += 1; } } } n };
        // The pressure: the ends thinner than the middle.
        s.header.hand = Some((1.0, 0.0, 0.0));
        let text = s.to_text();
        assert!(text.contains(" hand=1,0,0"), "the hand's token: {}", text.lines().next().unwrap_or(""));
        let parsed = StrokeScore::parse(&text).unwrap();
        assert_eq!(parsed.header.hand, Some((1.0, 0.0, 0.0)));
        let pressed = s.replay(64, 96).unwrap();
        assert!(drawn(&pressed, 4, 10) < drawn(&pressed, 44, 50), "the ends thinner: {} vs {}", drawn(&pressed, 4, 10), drawn(&pressed, 44, 50));
        assert_eq!(parsed.replay(64, 96).unwrap().to_image().into_raw(), pressed.to_image().into_raw(), "the pressure replays exact from the text");
        // The lift: a long straight line loses pixels.
        s.header.hand = Some((0.0, 0.0, 1.0));
        let lifted = s.replay(64, 96).unwrap();
        assert!(drawn(&lifted, 0, 96) < drawn(&plain_c, 0, 96), "the lead lifts: {} vs {}", drawn(&lifted, 0, 96), drawn(&plain_c, 0, 96));
        assert_eq!(StrokeScore::parse(&s.to_text()).unwrap().replay(64, 96).unwrap().to_image().into_raw(), lifted.to_image().into_raw(), "the lift replays exact from the text");
        // The tooth: a light line breaks on the peaks.
        s.strokes = vec![line(0.15)];
        s.header.hand = None;
        let light = s.replay(64, 96).unwrap();
        s.header.hand = Some((0.0, 1.0, 0.0));
        let toothed = s.replay(64, 96).unwrap();
        assert!(drawn(&toothed, 0, 96) < drawn(&light, 0, 96), "the light line on the peaks only: {} vs {}", drawn(&toothed, 0, 96), drawn(&light, 0, 96));
        assert_eq!(StrokeScore::parse(&s.to_text()).unwrap().replay(64, 96).unwrap().to_image().into_raw(), toothed.to_image().into_raw(), "the tooth replays exact from the text");
    }

    #[test]
    fn the_lines_resist_round_trips_and_walls_the_touch() {
        // A record with `resist` and neither wash nor touch (set 14): two NaN-separated lines of the skeleton's fluid,
        // back from the text with the NaN, laid before a touch of the engine so the sheet differs from the one
        // without it, and exact from the text.
        let mut s = sample();
        s.header.transmittance = true;
        s.header.stages = Some(vec!["aqua-1".into()]);
        s.header.engine = Some((6.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0));
        let lines = StrokeRecord { id: 1, wipe: false, wash: false, inject: false, stage: "aqua-1".into(), spline: vec![[30.0, 4.0], [30.0, 44.0], [f32::NAN, f32::NAN], [40.0, 4.0], [40.0, 44.0]], w0: 1.0, w1: 0.0, taper: 0.0, mix: Vec::new(), wet: 0.0, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: Some(1.0) };
        let touch = StrokeRecord { id: 2, wipe: false, wash: false, inject: true, stage: "aqua-1".into(), spline: vec![[20.0, 24.0]], w0: 8.0, w1: 0.0, taper: 0.0, mix: vec![("cadmium-red".into(), 0.5)], wet: 0.9, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        s.strokes = vec![lines, touch];
        let text = s.to_text();
        assert!(text.contains("NaN") && text.contains(" resist=1"), "the record's tokens: {text}");
        let parsed = StrokeScore::parse(&text).expect("parses");
        assert!(!parsed.strokes[0].wash && !parsed.strokes[0].inject && parsed.strokes[0].resist == Some(1.0) && parsed.strokes[0].spline[2][0].is_nan(), "the lines back from the text");
        let laid = s.replay(64, 48).unwrap().to_image().into_raw();
        let mut bare = s.clone();
        bare.strokes.remove(0);
        assert_ne!(laid, bare.replay(64, 48).unwrap().to_image().into_raw(), "the fluid along the lines walls the touch's water");
        assert_eq!(parsed.replay(64, 48).unwrap().to_image().into_raw(), laid, "and replays exact from the text");
    }

    #[test]
    fn the_techs_five_values_round_trip_and_a_set_18_score_reads_two() {
        // Set 20: the header's `tech=` with the barrier, the sinking and the drift as its third to fifth values —
        // written up to the last nonzero after the first two, read with two to five, a set-18 score's two values
        // back with zeros, nothing written for all zeros.
        let mut s = sample();
        s.header.tech = Some((0.0, 0.0, 1.0, 0.6, 0.4));
        let text = s.to_text();
        assert!(text.contains(" tech=0,0,1,0.6,0.4"), "the five values: {}", text.lines().nth(1).unwrap_or(""));
        assert_eq!(StrokeScore::parse(&text).unwrap().header.tech, Some((0.0, 0.0, 1.0, 0.6, 0.4)), "and back");
        s.header.tech = Some((2.0, 0.0, 0.0, 0.0, 0.0));
        let text = s.to_text();
        assert!(text.contains(" tech=2,0") && !text.contains(" tech=2,0,"), "the trailing zeros not written: {}", text.lines().nth(1).unwrap_or(""));
        assert_eq!(StrokeScore::parse(&text).unwrap().header.tech, Some((2.0, 0.0, 0.0, 0.0, 0.0)), "a set-18 score's two values");
        s.header.tech = Some((0.0, 0.8, 0.0, 0.5, 0.0));
        assert!(s.to_text().contains(" tech=0,0.8,0,0.5") && !s.to_text().contains(" tech=0,0.8,0,0.5,"), "up to the last nonzero");
        s.header.tech = Some((0.0, 0.0, 0.0, 0.0, 0.0));
        assert!(!s.to_text().contains(" tech="), "nothing for all zeros");
    }

    #[test]
    fn a_touch_of_the_water_engine_round_trips_and_replays_exact() {
        // A `J` record under an `engine` header: the tokens in the text, the record and the header back from it,
        // the touch laid when its stage dries (the sheet differs from the bare one), and the replay from the text
        // byte-identical to the replay from memory.
        let mut s = sample();
        s.header.transmittance = true;
        s.header.stages = Some(vec!["aqua-1".into()]);
        s.header.engine = Some((6.0, 1.0, 0.8, 1.0, 0.5, 0.0, 0.0));
        s.strokes = vec![StrokeRecord { id: 1, wipe: false, wash: false, inject: true, stage: "aqua-1".into(), spline: vec![[20.0, 20.0], [30.0, 24.0]], w0: 8.0, w1: 0.0, taper: 0.0, mix: vec![("cadmium-red".into(), 0.5)], wet: 0.9, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None }];
        let text = s.to_text();
        assert!(text.contains(" engine=6,1,0.8,1,0.5") && text.contains("\nJ 1 stage=aqua-1 "), "the tokens: {text}");
        let parsed = StrokeScore::parse(&text).expect("parses");
        assert!(parsed.strokes[0].inject && !parsed.strokes[0].wash, "the touch back from the text");
        assert_eq!(parsed.header.engine, Some((6.0, 1.0, 0.8, 1.0, 0.5, 0.0, 0.0)), "the engine back from the text");
        assert_eq!(StrokeScore::parse(&text.replace("engine=6,1,0.8,1,0.5", "engine=6,1,0.8,1")).unwrap().header.engine, Some((6.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0)), "a set-9 header without the viscosity");
        let mut boosted = s.clone();
        boosted.header.engine = Some((6.0, 1.0, 0.8, 1.0, 0.0, 1.0, 0.0));
        let bt = boosted.to_text();
        assert!(bt.contains(" engine=6,1,0.8,1,0,1"), "the boost as the sixth value: {bt}");
        assert_eq!(StrokeScore::parse(&bt).unwrap().header.engine, Some((6.0, 1.0, 0.8, 1.0, 0.0, 1.0, 0.0)), "and back");
        boosted.header.engine = Some((6.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.4));
        let xt = boosted.to_text();
        assert!(xt.contains(" engine=6,1,0.8,1,0,0,0.4"), "the bias as the seventh value: {xt}");
        assert_eq!(StrokeScore::parse(&xt).unwrap().header.engine, Some((6.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.4)), "and back");
        let laid = s.replay(64, 48).unwrap().to_image().into_raw();
        let mut bare = s.clone();
        bare.strokes.clear();
        assert_ne!(laid, bare.replay(64, 48).unwrap().to_image().into_raw(), "the touch lands when the stage dries");
        assert_eq!(parsed.replay(64, 48).unwrap().to_image().into_raw(), laid, "and replays exact from the text");
        // The brush's sweep in the record's `taper` (set 13): the touch lays less toward its ends, the text carries it.
        let mut swept = s.clone();
        swept.strokes[0].taper = 0.6;
        let st = swept.to_text();
        assert!(st.contains(" taper=0.6 "), "the sweep in the record: {st}");
        let swept_laid = swept.replay(64, 48).unwrap().to_image().into_raw();
        assert_ne!(swept_laid, laid, "the sweep runs the brush dry along the touch");
        assert_eq!(StrokeScore::parse(&st).unwrap().replay(64, 48).unwrap().to_image().into_raw(), swept_laid, "and replays exact from the text");
    }

    #[test]
    fn the_flow_header_round_trips_and_reads_the_older_three_value_form() {
        let mut s = sample();
        s.header.flow = Some((0.6, 10.24, 0.8, 0.12, 1.0));
        let parsed = StrokeScore::parse(&s.to_text()).expect("parses");
        assert_eq!(parsed.header.flow, Some((0.6, 10.24, 0.8, 0.12, 1.0)));
        // Scores written before the grain / selective values: those read as 0.
        let old = s.to_text().replace("flow=0.6,10.24,0.8,0.12,1", "flow=0.6,10.24,0.8");
        assert_eq!(StrokeScore::parse(&old).unwrap().header.flow, Some((0.6, 10.24, 0.8, 0.0, 0.0)));
        let old4 = s.to_text().replace("flow=0.6,10.24,0.8,0.12,1", "flow=0.6,10.24,0.8,0.12");
        assert_eq!(StrokeScore::parse(&old4).unwrap().header.flow, Some((0.6, 10.24, 0.8, 0.12, 0.0)));
    }

    #[test]
    fn replay_at_native_size_is_byte_identical_across_reparses() {
        let s = sample();
        let a = s.replay(64, 48).unwrap().to_image().into_raw();
        // Same score, and a round-tripped copy, must render identically.
        let b = StrokeScore::parse(&s.to_text()).unwrap().replay(64, 48).unwrap().to_image().into_raw();
        assert_eq!(a, b, "replay is deterministic and survives serialize/parse");
    }

    #[test]
    fn replay_scales_to_any_resolution_keeping_structure() {
        // A denser, real score (painted) — replay it at native size and at 4×, and the 4× render downscaled
        // keeps the same structure. Cross-resolution replay is structure-preserving (A2), not pixel-identical.
        use crate::paint::painter::{paint_from_image, PaintParams};
        let img = image::RgbImage::from_fn(96, 72, |x, y| image::Rgb([(30 + 2 * x) as u8, (200 - x - y) as u8, (60 + y) as u8]));
        let mut p = PaintParams::new(palette::EARTH, 400);
        p.brush_sizes = vec![22.0, 11.0];
        let s = paint_from_image(&img, &p).score;
        let native = s.replay(96, 72).unwrap().to_image();
        let big = s.replay(384, 288).unwrap().to_image();
        assert_eq!(big.dimensions(), (384, 288), "renders at the requested size, no upscaler");
        let big_small = image::imageops::resize(&big, 96, 72, image::imageops::FilterType::Triangle);
        let tr = crate::paint::painter::traceability(&big_small, &native);
        // Cross-resolution replay keeps the structure (well above chance); pixel-tight resolution-independence
        // (A2 to a high bar) is a later refinement of the deposit/charge scaling.
        assert!(tr > 0.65, "same score at 4× keeps structure (corr {tr})");
    }

    #[test]
    fn stages_and_filtered_replay() {
        let s = sample();
        assert_eq!(s.stages(), vec!["shadow-mass".to_string(), "light-mass".to_string()], "distinct stages in order");
        // Replaying only the light-mass stage differs from the full painting (the shadow stroke is absent).
        let full = s.replay(64, 48).unwrap().to_image().into_raw();
        let one = s.replay_filtered(64, 48, |r| r.stage == "light-mass").unwrap().to_image().into_raw();
        assert_ne!(full, one, "a single-stage separation is not the whole painting");
    }

    #[test]
    fn replay_frames_emits_progress_and_ends_finished() {
        let s = sample();
        let frames = s.replay_frames(64, 48, 1).unwrap();
        // 2 strokes, every=1 → a frame after each, plus the final = 3 (with a duplicate final, fine for a reel).
        assert!(frames.len() >= 2, "one frame per stroke + the finish");
        let last = frames.last().unwrap().to_image().into_raw();
        let full = s.replay(64, 48).unwrap().to_image().into_raw();
        assert_eq!(last, full, "the final frame is the finished painting");
    }

    #[test]
    fn replay_applies_a_wipe_record() {
        // A score that lays a dark stroke then WIPES part of it — the wiped band is lighter than without it.
        let base = ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "oil-direct".into(), seed: 1, width: 40, height: 20, tooth: 0.9, ground: None, brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: None, dry: 0.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 1.0, transmittance: false, flow: None, bloom: None, grain_mid: 0.0, grain_wet: 0.0, barrier: 0.0, ragged: 0.0, tension: 0.0, fluid: None, charge: 0.0, engine: None, pencil: 0.0, grain_dir: None, hand: None, tech: None, weave: 0.0, sag: 0.0, gloss: 0.0, gloss_map: 0.0, cavity: 0.0, cavity_face: 1.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None };
        let stroke = StrokeRecord { id: 1, wipe: false, wash: false, inject: false, stage: "mass".into(), spline: vec![[2.0, 10.0], [38.0, 10.0]], w0: 10.0, w1: 10.0, taper: 0.0, mix: vec![("ivory-black".into(), 5.0)], wet: 1.0, press: 1.0, streak: 0.6, round: 0.7, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        let no_wipe = StrokeScore { header: base.clone(), strokes: vec![stroke.clone()] };
        let wipe = StrokeRecord { id: 2, wipe: true, wash: false, inject: false, stage: "scrape".into(), spline: vec![[18.0, 4.0], [18.0, 16.0]], w0: 8.0, w1: 8.0, taper: 0.0, mix: vec![], wet: 0.9, press: 1.0, streak: 0.6, round: 0.7, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        let with_wipe = StrokeScore { header: base, strokes: vec![stroke, wipe] };
        let a = no_wipe.replay(40, 20).unwrap();
        let b = with_wipe.replay(40, 20).unwrap();
        use crate::paint::color::{delta_e76, srgb_to_lab};
        let lifted = srgb_to_lab(b.color_at(18, 10)).l - srgb_to_lab(a.color_at(18, 10)).l;
        // The wipe thins the paint film so the ground shows through and the band lightens (film-build opacity —
        // subtler than injecting white, but the direction is the point).
        assert!(lifted > 1.0, "the wipe lightened the scraped band (ΔL {lifted})");
        // Away from the wipe, the two are identical.
        assert!(delta_e76(srgb_to_lab(a.color_at(5, 10)), srgb_to_lab(b.color_at(5, 10))) < 1.0, "untouched elsewhere");
    }

    #[test]
    fn unknown_palette_replay_errors() {
        let mut s = sample();
        s.header.palette = "nope".into();
        assert!(s.replay(64, 48).is_err());
        let _ = palette::ZORN; // keep the import used
    }

    #[test]
    fn a_smudge_record_round_trips_and_replays_the_finger() {
        // A score that lays a black stroke on the left and then SMUDGES rightward from inside it: the record
        // survives the text round trip, and the replay carries the black beyond the stroke — darker there than
        // without the smudge — while the stroke's own start is unchanged.
        let base = ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "pastel".into(), seed: 1, width: 60, height: 20, tooth: 0.9, ground: None, brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: None, dry: 1.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 0.5, transmittance: false, flow: None, bloom: None, grain_mid: 0.0, grain_wet: 0.0, barrier: 0.0, ragged: 0.0, tension: 0.0, fluid: None, charge: 0.0, engine: None, pencil: 0.0, grain_dir: None, hand: None, tech: None, weave: 0.0, sag: 0.0, gloss: 0.0, gloss_map: 0.0, cavity: 0.0, cavity_face: 1.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None };
        let stroke = StrokeRecord { id: 1, wipe: false, wash: false, inject: false, stage: "block-in".into(), spline: vec![[2.0, 10.0], [24.0, 10.0]], w0: 10.0, w1: 10.0, taper: 0.0, mix: vec![("ivory-black".into(), 5.0)], wet: 0.5, press: 1.0, streak: 0.6, round: 0.7, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: true, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        let finger = StrokeRecord { id: 2, wipe: true, wash: false, inject: false, stage: "block-in".into(), spline: vec![[16.0, 10.0], [50.0, 10.0]], w0: 8.0, w1: 8.0, taper: 0.0, mix: vec![], wet: 0.5, press: 1.0, streak: 0.0, round: 1.0, pickup: None, bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: Some(0.5), stop: None, edge: None, prof: None, resist: None };
        let without = StrokeScore { header: base.clone(), strokes: vec![stroke.clone()] };
        let with = StrokeScore { header: base, strokes: vec![stroke, finger] };
        let text = with.to_text();
        assert!(text.contains("W 2 stage=block-in") && text.contains(" smudge=0.5"), "{text}");
        let back = StrokeScore::parse(&text).expect("parses");
        assert_eq!(back.strokes.len(), 2);
        assert_eq!(back.strokes[1].smudge, Some(0.5));
        assert!(back.strokes[1].wipe && !back.strokes[1].wash);
        let a = without.replay(60, 20).unwrap();
        let b = with.replay(60, 20).unwrap();
        let c = back.replay(60, 20).unwrap();
        assert_eq!(b.to_image().into_raw(), c.to_image().into_raw(), "the parsed score replays to the bit");
        use crate::paint::color::srgb_to_lab;
        let carried = srgb_to_lab(a.color_at(34, 10)).l - srgb_to_lab(b.color_at(34, 10)).l;
        assert!(carried > 1.0, "the black carried beyond the stroke (ΔL {carried})");
        assert_eq!(a.color_at(5, 10), b.color_at(5, 10), "the stroke's start untouched");
        // The frames replay carries it too.
        let frames = with.replay_frames(60, 20, 1).unwrap();
        assert_eq!(frames.last().unwrap().to_image().into_raw(), b.to_image().into_raw());
    }

    #[test]
    fn a_wash_with_an_edge_per_point_round_trips_and_blooms_by_it() {
        // A square wash, its wet edge 0 along its left side and 12 px along its right: the record survives the
        // text round trip; the replay blooms past the right side and not past the left; a score without the
        // edge (the one feather, 0) blooms past neither.
        let base = ScoreHeader { version: 1, palette: "zorn".into(), pigments: Vec::new(), medium: "watercolour".into(), seed: 1, width: 80, height: 40, tooth: 0.9, ground: None, brush: BrushConfig::default(), bleed: 0.0, diffuse: 0.0, stages: None, dry: 1.0, opacity: 1.0, impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, contrast: 1.0, warmth: 0.0, clarity: 0.0, lift: 0.5, transmittance: false, flow: None, bloom: None, grain_mid: 0.0, grain_wet: 0.0, barrier: 0.0, ragged: 0.0, tension: 0.0, fluid: None, charge: 0.0, engine: None, pencil: 0.0, grain_dir: None, hand: None, tech: None, weave: 0.0, sag: 0.0, gloss: 0.0, gloss_map: 0.0, cavity: 0.0, cavity_face: 1.0, hiding: 0.0, tone: 0.0, water: 0.0, agent: 0.0, light: 0.0, phong: 0.0, spread: 0.0, peak: 0.0, collide: None, hold_mask: None };
        let rings = vec![[30.0, 8.0], [50.0, 8.0], [50.0, 32.0], [30.0, 32.0]];
        let plain = StrokeRecord { id: 1, wipe: false, wash: true, inject: false, stage: "wash".into(), spline: rings.clone(), w0: 0.0, w1: 0.0, taper: 0.0, mix: vec![("ivory-black".into(), 5.0)], wet: 0.75, press: 1.0, streak: 0.0, round: 1.0, pickup: Some(0.0), bristles: None, kd: None, cap: None, flat: false, hold: false, visc: None, crest: None, over: None, split: None, tint: None, knife: None, twist: None, tgt: None, mix2: None, smudge: None, stop: None, edge: None, prof: None, resist: None };
        let edged = StrokeRecord { edge: Some(vec![0.0, 12.0, 12.0, 0.0]), ..plain.clone() };
        let without = StrokeScore { header: base.clone(), strokes: vec![plain] };
        let with = StrokeScore { header: base, strokes: vec![edged] };
        let text = with.to_text();
        assert!(text.contains("A 1 stage=wash") && text.contains(" edge=0;12;12;0"), "{text}");
        let back = StrokeScore::parse(&text).expect("parses");
        assert_eq!(back.strokes[0].edge.as_deref(), Some(&[0.0, 12.0, 12.0, 0.0][..]));
        let a = without.replay(80, 40).unwrap();
        let b = with.replay(80, 40).unwrap();
        let c = back.replay(80, 40).unwrap();
        assert_eq!(b.to_image().into_raw(), c.to_image().into_raw(), "the parsed score replays to the bit");
        use crate::paint::color::srgb_to_lab;
        let l = |cv: &Canvas, x: u32, y: u32| srgb_to_lab(cv.color_at(x, y)).l;
        assert!(l(&a, 54, 20) > 95.0 && l(&a, 25, 20) > 95.0, "one feather of 0: no bloom either side");
        assert!(l(&b, 54, 20) < 90.0, "the edge of 12 px blooms past the right side (L {})", l(&b, 54, 20));
        assert!(l(&b, 25, 20) > 95.0, "the edge of 0 blooms past the left side not at all (L {})", l(&b, 25, 20));
        assert_eq!(a.color_at(40, 20), b.color_at(40, 20), "the mass itself the same");
    }
}
