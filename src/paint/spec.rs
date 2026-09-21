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
    pub style: Option<String>,
    pub surface: Option<SurfaceSpec>,
    pub budget: Option<BudgetSpec>,
    pub seed: Option<u64>,
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
        bail!("medium {:?} is declared but not executable yet — the engine renders {}; the rest land in later phases", medium.name, medium::P2_EXECUTABLE.join(" / "));
    }
    let palette = Palette::by_name(spec.palette.as_deref().unwrap_or("zorn"))
        .with_context(|| format!("unknown palette {:?} — try: {}", spec.palette, palette::ALL.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?;
    let budget = spec.budget.as_ref().and_then(|b| b.strokes).unwrap_or(1500).max(1);
    let seed = spec.seed.unwrap_or(42);
    let size = spec.size().unwrap_or((ref_w, ref_h));

    let stages = medium::generate_schedule(&medium);
    let n = stages.len().max(1);
    let coarse = (size.0.max(size.1) as f32 / 16.0).max(6.0);
    let fine = 4.0_f32;
    let total_w: f32 = stages.iter().map(stage_weight).sum::<f32>().max(1e-3);

    let passes: Vec<PassSpec> = stages
        .iter()
        .enumerate()
        .map(|(i, s)| {
            // Radius: geometric coarse → fine across the schedule.
            let frac = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
            let radius = coarse * (fine / coarse).powf(frac);
            let b = ((budget as f32) * stage_weight(s) / total_w).round() as usize;
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
    fn accepts_p2_media_and_rejects_the_rest() {
        // Watercolour + pen-ink are executable through P2; tempera / ink-wash / indirect-oil are not yet.
        assert!(compile(&PaintSpec { medium: Some("watercolour".into()), ..Default::default() }, 256, 256).is_ok());
        assert!(compile(&PaintSpec { medium: Some("pen-ink".into()), ..Default::default() }, 256, 256).is_ok());
        assert!(compile(&PaintSpec { medium: Some("tempera".into()), ..Default::default() }, 256, 256).is_err(), "tempera is later");
    }

    #[test]
    fn honours_the_surface_size_over_the_reference() {
        let spec = PaintSpec { surface: Some(SurfaceSpec { size: Some("1024x768".into()), ground: None }), ..Default::default() };
        assert_eq!(compile(&spec, 400, 400).unwrap().size, (1024, 768));
    }
}
