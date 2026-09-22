//! `PaintSpec` and the plan compiler (RFC PAINT-1 §11.1, §6.4, §9). A spec is authored in HJSON; the compiler
//! turns it into a `PaintPlan` — the medium's stage schedule, each stage assigned a brush radius (coarse →
//! fine) and a share of the stroke budget. P1 executes oil-direct and gouache over a `reference:` image; the
//! prose `subject:` / armature path lands in P2.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::paint::medium::{self, MediumProfile, Stage};
use crate::paint::painter::PassSpec;
use crate::paint::palette::{self, Palette};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SurfaceSpec {
    pub size: Option<String>,
    pub ground: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BudgetSpec {
    pub strokes: Option<usize>,
}

/// The `paint:` body of a spec.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PaintSpec {
    pub version: Option<u32>,
    /// P1: the image to paint. (Prose `subject` + armature is P2.)
    pub reference: Option<String>,
    pub subject: Option<String>,
    pub medium: Option<String>,
    pub palette: Option<String>,
    /// Fidelity register: `legible` (default) or `impressionist`.
    pub style: Option<String>,
    pub surface: Option<SurfaceSpec>,
    pub budget: Option<BudgetSpec>,
    pub seed: Option<u64>,
    // ── Armature (the model as art director) ────────────────────────────────────────────────────────────────
    /// Diffusion steps for the armature render — MORE steps = a clearer, more coherent armature (a clearer
    /// figure, net, sea), which the painter then renders. The single biggest lever on subject legibility.
    pub steps: Option<usize>,
    /// Negative prompt for the armature render (e.g. "blurry, deformed hands, extra limbs").
    pub negative: Option<String>,
    /// Armature model (default sdxl).
    pub model: Option<String>,
    // ── Painter controls (scene-authoritative; the code carries only defaults) ───────────────────────────────
    /// Edge-hardness strength (0..1) — how many boundaries read as HARD (crisp meetings). Legible only.
    pub define: Option<f32>,
    /// Aerial-perspective strength (0..1) — background recession via the depth map. 0 = flat.
    pub haze: Option<f32>,
    /// STROKE LENGTH multiplier (default 1.0) — longer = cleaner sweeping strokes, shorter = choppier.
    pub stroke_length: Option<f32>,
    /// STROKE WIDTH multiplier (default 1.0) — wider = fewer, broader marks; narrower = finer, more marks.
    pub stroke_width: Option<f32>,
    // ── Technique behaviour (each overrides the medium's default) ────────────────────────────────────────────
    /// Wet-into-wet BLEED (0..1): fusion/bloom of the wet media. Default per medium (watercolour/ink high).
    pub bleed: Option<f32>,
    /// BODY / opacity (0.1..1): 1 = opaque cover (gouache/oil), low = transparent (watercolour/ink glow).
    pub opacity: Option<f32>,
    /// PICKUP (0..1): the dirty-brush drag — high fuses neighbouring colour (oil/ink), low keeps marks clean.
    pub pickup: Option<f32>,
    /// IMPASTO (0..1): the textured, light-catching thick-paint relief at output — oil/knife high, flat media 0.
    pub impasto: Option<f32>,
    // ── Paint MATERIAL physics (each overrides the medium default) ───────────────────────────────────────────
    /// CHROMA / saturation range (1 neutral; >1 vivid oil; <1 muted gouache/watercolour).
    pub chroma: Option<f32>,
    /// DRY SHIFT — value change on drying (+ watercolour dries lighter; − gouache dries to a matte mid).
    pub dry_shift: Option<f32>,
    /// GRANULATION — pigment settling into the paper tooth (watercolour / graphite grain).
    pub granulate: Option<f32>,
    /// SHEEN / gloss — specular highlight on paint ridges (oil glossy; watercolour/gouache matte).
    pub sheen: Option<f32>,
    /// LIFT — wipe removability (oil high; watercolour staining low).
    pub lift: Option<f32>,
    /// BROKEN COLOUR (0..1): per-stroke hue/chroma variation for optical-mix vibrancy (oil/gouache/pastel).
    pub broken: Option<f32>,
    /// CONTOUR (0..1): line-drawing pass over the strongest edges (pen/pencil/charcoal).
    pub contour: Option<f32>,
    /// SALIENCY-GATED DENSITY (0..1, 0 = off — opt-in): reserve dense strokes for the focal, high-structure
    /// passages and lay flat/empty regions thin, so a large stroke budget does not over-work the background
    /// into a uniform hatch. The block-in always covers the canvas; only restating/detail passes are thinned.
    pub saliency: Option<f32>,
    /// RESERVE threshold (0..1, surface-white media): cells brighter than this keep the bare paper (no stroke).
    /// Raise toward 1 to close white holes in light passages; lower to keep more paper. Default 0.72 (watercolour/ink).
    pub reserve: Option<f32>,
    /// SELECTIVE DETAIL (0..1, 0 = off — opt-in): paint the masses loose but fire the crisp detail tier only in
    /// the focal region (eyes/glasses) — loose-wash plus sharp accents. Small = tighter focus; 1 = detail everywhere.
    pub focus_detail: Option<f32>,
    /// PRESERVE FACE (0..1, 0 = off — opt-in): detect the face(s) and fire the crisp detail tier only on the real
    /// face box (loose elsewhere). Model-targeted variant of `focus_detail`; needs a `reference:` to detect on.
    pub preserve_face: Option<f32>,
    /// SPLATTER (0..1, 0 = off — opt-in): flick fine pigment droplets across the painting — watercolour/ink spatter.
    pub splatter: Option<f32>,
    /// EDGE POOLING (0..1, 0 = off — opt-in): darken pigment at wash boundaries — the watercolour edge-bloom ring.
    pub edge_pool: Option<f32>,
    /// PAPER EDGE (0..1, 0 = off — opt-in): fade to a deckled bare-paper border — the torn-paper watercolour vignette.
    pub paper_edge: Option<f32>,
    /// FINISH GRADE (painting-safe, recorded for replay): CONTRAST (0.5..2, 1 = neutral).
    pub contrast: Option<f32>,
    /// WARMTH (−1..1, 0 = neutral): finish white-balance shift, + warm / − cool.
    pub warmth: Option<f32>,
    /// CLARITY (0..1, 0 = off): gentle local contrast (not edge sharpening).
    pub clarity: Option<f32>,
    /// Brushwork plan: the default brush and, later, per-element assignments (see `BrushworkSpec`).
    pub brushwork: Option<BrushworkSpec>,
    /// COMPOSITION LAYERS (per-element painting): render, matte and paint each element on its own layer, back to
    /// front. When present, this drives the painting instead of a single `subject` armature.
    pub composition: Option<CompositionSpec>,
}

/// A composition: an ordered list of ELEMENTS painted back-to-front (first = farthest).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompositionSpec {
    pub elements: Vec<ElementSpec>,
}

/// One composition ELEMENT (e.g. sky, sea, figure, net): its own render, mask, and brush.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ElementSpec {
    pub name: Option<String>,
    /// What to render for this element (prose → its own SDXL armature).
    pub subject: Option<String>,
    /// Negative prompt for this element's render.
    pub negative: Option<String>,
    /// The brush for this element (vocabulary name: flat / filbert / round / fan / rigger / knife / wash).
    pub brush: Option<String>,
    /// Coarse footprint / placement hint (also the fallback for `place`): `full` / `top` / `upper` / `bottom` /
    /// `lower` / `left` / `right` / `middle` / `subject`.
    pub mask: Option<String>,
    // ── Layered-plan placement + anchor strength (passed straight through to `plakat layers`) ────────────────
    /// Explicit placement words for the layered plan (e.g. "center-bottom", "center-top", "left mid front").
    /// Overrides the `mask`→place mapping.
    pub place: Option<String>,
    /// Element size word: `small` / `medium` / `large`.
    pub size: Option<String>,
    /// Anchor WEIGHT (0..1): how strongly this element's low-frequency guide holds (higher = the plan's layout
    /// dominates; lower = the finish diffusion is freer). The backdrop's weight comes from the first element.
    pub weight: Option<f32>,
    /// Anchor WINDOW (0..1): the step-fraction at which this element stops being anchored (higher = held longer).
    pub window: Option<f32>,
    /// Explicit depth (0 = nearest); else derived from the element's order.
    pub depth: Option<f32>,
    /// Per-element armature steps (else the spec-level `steps`).
    pub steps: Option<usize>,
    /// Per-element stroke budget (else derived from the element's area).
    pub strokes: Option<usize>,
}

/// The brushwork plan (RFC brush vocabulary). `default` names the brush for unassigned areas; `assign` maps a
/// prompt phrase / element to a brush + stroke character (per-element region assignment lands with detection).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct BrushworkSpec {
    pub default: Option<String>,
    #[serde(default)]
    pub assign: Vec<BrushAssign>,
}

/// One brushwork assignment: paint the region matching `where` with brush `brush` (and optional stroke length).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct BrushAssign {
    #[serde(rename = "where")]
    pub where_: Option<String>,
    pub brush: Option<String>,
    /// Stroke length register: `short` | `medium` | `long`.
    pub strokes: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpecWrap {
    paint: PaintSpec,
}

impl PaintSpec {
    /// Parse a spec from HJSON — the `paint: { … }` object, or a bare object.
    pub fn parse(text: &str) -> Result<Self> {
        if let Ok(w) = deser_hjson::from_str::<SpecWrap>(text) {
            return Ok(w.paint);
        }
        deser_hjson::from_str::<PaintSpec>(text).context("parsing PaintSpec HJSON")
    }

    /// The declared surface size, if any, as `(w,h)`.
    pub fn size(&self) -> Option<(u32, u32)> {
        self.surface.as_ref().and_then(|s| s.size.as_deref()).and_then(parse_size)
    }
}

fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once(['x', 'X', '*'])?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// A compiled plan: the resolved medium, its stage schedule, and the painter passes (radius + budget per stage).
#[derive(Debug, Clone)]
pub struct PaintPlan {
    pub palette: Palette,
    pub medium: MediumProfile,
    pub stages: Vec<Stage>,
    pub passes: Vec<PassSpec>,
    pub budget: usize,
    pub seed: u64,
    /// Output size — the spec's surface size, else the reference image's.
    pub size: (u32, u32),
}

/// A painting-scale default stroke budget derived from the canvas area and the medium (used when the spec
/// gives no explicit `budget.strokes`). ~14k marks per megapixel for a loaded-brush medium — an 832×1024
/// canvas lands near 12k — scaled up for density media, whose marks are many and small. Clamped to a sane band.
pub fn auto_budget(size: (u32, u32), medium: &MediumProfile) -> usize {
    let mp = (size.0 as f32 * size.1 as f32) / 1_000_000.0;
    let per_mp = match medium.mark_model {
        medium::MarkModel::Density => 60_000.0, // hatch/stipple: many small marks
        medium::MarkModel::Continuous => 30_000.0, // enough small marks for the detail passes to resolve features
    };
    ((per_mp * mp).round() as usize).clamp(1_500, 120_000)
}

/// A stage's share of the stroke budget — structural masses get the most, accents/highlights the fewest but
/// carry the focal notes (§9). Weights are relative; the compiler normalises them.
fn stage_weight(s: &Stage) -> f32 {
    match s {
        Stage::ReservePlan => 0.2,
        Stage::Ground => 1.0,
        Stage::Value(_) => 1.3,
        Stage::ShadowMass => 1.6,
        Stage::LightMass => 1.6,
        Stage::Colour(_) => 1.0,
        Stage::Halftone => 0.8,
        Stage::Accents => 0.5,
        Stage::Highlights => 0.5,
    }
}

/// Compile a spec into a plan against a reference of size `(ref_w, ref_h)`.
pub fn compile(spec: &PaintSpec, ref_w: u32, ref_h: u32) -> Result<PaintPlan> {
    let medium = MediumProfile::by_name(spec.medium.as_deref().unwrap_or("oil-direct"))
        .with_context(|| format!("unknown medium {:?}", spec.medium))?;
    if !medium.is_executable() {
        bail!("medium {:?} is not executable — the engine renders {}", medium.name, medium::EXECUTABLE.join(" / "));
    }
    // Palette: `image`/`auto` is resolved FROM the reference by the CLI (a placeholder here); otherwise the
    // named palette, defaulting to the one that SUITS this medium (sumi for ink/pencil, split-primary for gouache).
    let pal_name = spec.palette.as_deref().filter(|s| !matches!(s.trim().to_ascii_lowercase().as_str(), "image" | "auto")).unwrap_or(medium.default_palette);
    let palette = Palette::by_name(pal_name)
        .with_context(|| format!("unknown palette {:?} — try: {}", spec.palette, palette::ALL.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?;
    let seed = spec.seed.unwrap_or(42);
    let size = spec.size().unwrap_or((ref_w, ref_h));
    // Budget: the spec's explicit `budget.strokes` wins; otherwise DERIVE a painting-scale count from the
    // canvas area and the medium — a real painting is many thousands of marks across its layers, not a flat
    // default. Density media (pen-ink, tempera) pack far more, smaller marks than a loaded brush.
    let budget = spec.budget.as_ref().and_then(|b| b.strokes).unwrap_or_else(|| auto_budget(size, &medium)).max(1);

    let stages = medium::generate_schedule(&medium);
    let n = stages.len().max(1);
    let coarse = (size.0.max(size.1) as f32 / 16.0).max(6.0);
    let fine = 4.0_f32;
    let total_w: f32 = stages.iter().map(stage_weight).sum::<f32>().max(1e-3);

    // The first stage is the block-in; it must COVER the canvas or the painting reads as sparse scribble. Give
    // it at least a coverage floor — strokes ≈ canvas area / (a coarse stroke's footprint) — regardless of its
    // weighted share, and let the remaining stages split the rest of the budget.
    // ~grid cells at the coarse brush (grid ≈ 0.9·radius), ×1.4 so the block-in overlaps and truly covers.
    let coverage_floor = (1.4 * (size.0 as f32 * size.1 as f32) / (coarse * coarse * 0.81)).ceil() as usize;
    let ground_budget = coverage_floor.max(((budget as f32) * stage_weight(&stages[0]) / total_w).round() as usize);
    let rest_budget = budget.saturating_sub(ground_budget.min(budget)).max(stages.len());
    let rest_w: f32 = stages.iter().skip(1).map(stage_weight).sum::<f32>().max(1e-3);

    let passes: Vec<PassSpec> = stages
        .iter()
        .enumerate()
        .map(|(i, s)| {
            // Radius: geometric coarse → fine across the schedule.
            let frac = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
            let radius = coarse * (fine / coarse).powf(frac);
            let b = if i == 0 {
                ground_budget
            } else {
                ((rest_budget as f32) * stage_weight(s) / rest_w).round() as usize
            };
            PassSpec { radius: radius.max(fine), budget: b.max(1), stage: s.slug() }
        })
        .collect();

    Ok(PaintPlan { palette, medium, stages, passes, budget, seed, size })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wrapped_and_bare() {
        // HJSON unquoted values run to end-of-line, so specs are multi-line (one field per line).
        let wrapped = "{\n  paint: {\n    medium: gouache\n    palette: earth\n    budget: { strokes: 800 }\n    seed: 7\n  }\n}\n";
        let s = PaintSpec::parse(wrapped).unwrap();
        assert_eq!(s.medium.as_deref(), Some("gouache"));
        assert_eq!(s.budget.unwrap().strokes, Some(800));
        let bare = "{\n  medium: oil-direct\n  palette: zorn\n}\n";
        assert_eq!(PaintSpec::parse(bare).unwrap().palette.as_deref(), Some("zorn"));
    }

    #[test]
    fn compiles_a_plan_with_a_pass_per_stage_and_budget_conserved() {
        let spec = PaintSpec { medium: Some("oil-direct".into()), palette: Some("zorn".into()), budget: Some(BudgetSpec { strokes: Some(1000) }), ..Default::default() };
        let plan = compile(&spec, 512, 640).unwrap();
        assert_eq!(plan.passes.len(), plan.stages.len(), "one pass per stage");
        assert_eq!(plan.size, (512, 640), "falls back to the reference size");
        // Radii go coarse → fine; the first pass is the biggest brush.
        assert!(plan.passes[0].radius > plan.passes.last().unwrap().radius, "coarse → fine");
        // Budgets roughly sum to the total (rounding aside).
        let total: usize = plan.passes.iter().map(|p| p.budget).sum();
        assert!((total as i64 - 1000).abs() <= plan.passes.len() as i64, "budget conserved (got {total})");
    }

    #[test]
    fn accepts_every_declared_medium_and_rejects_unknown() {
        // All seven declared media are executable now (P3 added indirect-oil / ink-wash / tempera).
        for m in ["oil-direct", "oil-indirect", "gouache", "watercolour", "ink-wash", "pen-ink", "tempera"] {
            assert!(compile(&PaintSpec { medium: Some(m.into()), ..Default::default() }, 256, 256).is_ok(), "{m} compiles");
        }
        assert!(compile(&PaintSpec { medium: Some("crayon".into()), ..Default::default() }, 256, 256).is_err(), "unknown medium errors");
    }

    #[test]
    fn honours_the_surface_size_over_the_reference() {
        let spec = PaintSpec { surface: Some(SurfaceSpec { size: Some("1024x768".into()), ground: None }), ..Default::default() };
        assert_eq!(compile(&spec, 400, 400).unwrap().size, (1024, 768));
    }
}
