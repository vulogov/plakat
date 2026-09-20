//! Scenario `type: layered` task (RFC LAYERED-1 P5). Runs the layered pipeline — plan-or-prose → drafts →
//! guide → one anchored finish trajectory — as one step of a scenario, writing `<out>/layered.png`. Reuses
//! [`crate::layered::render`]; a prose task runs the P4 planner first.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;

use crate::layered::{lint, plan, planner, render};

/// A `type: layered` task body: render from an existing `plan:` HJSON, or decompose `prompt:` prose into a
/// plan first (the P4 planner). Everything else has a sensible default.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LayeredTaskCfg {
    /// A plan HJSON path.
    pub plan: Option<String>,
    /// OR prose to decompose into a plan (planner).
    pub prompt: Option<String>,
    /// Finish model (SD family / Flux). Default sdxl.
    pub model: Option<String>,
    pub draft_model: Option<String>,
    pub draft_steps: Option<usize>,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    pub ramp: Option<f32>,
    pub size: Option<String>,
    /// The planner LLM alias (prose only).
    pub provider: Option<String>,
}

/// Resolve a `plan:` path relative to the scenario file's directory (so a compile-emitted sidecar next to the
/// scenario is found no matter the CWD). Absolute paths pass through.
fn resolve_plan(plan: &str, base_dir: &Path) -> std::path::PathBuf {
    base_dir.join(plan)
}

/// Validate up front (before any model load): a `plan:` path (which must exist, resolved against the scenario
/// dir) or a non-empty `prompt:`.
pub fn validate(cfg: &LayeredTaskCfg, base_dir: &Path) -> Result<()> {
    let has_plan = cfg.plan.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
    let has_prompt = cfg.prompt.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
    anyhow::ensure!(has_plan || has_prompt, "layered task needs a `plan:` path or a `prompt:` (prose)");
    if let Some(p) = cfg.plan.as_deref().filter(|s| !s.trim().is_empty()) {
        let resolved = resolve_plan(p, base_dir);
        anyhow::ensure!(resolved.exists(), "layered task: plan {} not found", resolved.display());
    }
    Ok(())
}

/// Run the layered pipeline → `<out_dir>/layered.png`. `seed` is the scenario-assigned task seed (a `seed:`
/// on the task cfg overrides it).
pub async fn run_layered_task(cfg: &LayeredTaskCfg, device: Device, out_dir: &Path, seed: u64, dry_run: bool, base_dir: &Path) -> Result<()> {
    validate(cfg, base_dir)?;
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out = out_dir.join("layered.png");
    if dry_run {
        return Ok(());
    }

    let model = cfg.model.clone().unwrap_or_else(|| "sdxl".into());
    let seed = cfg.seed.unwrap_or(seed);

    // Resolve the plan: parse a file (relative to the scenario dir), or run the planner over prose.
    let layer_plan = if let Some(path) = cfg.plan.as_deref().filter(|s| !s.trim().is_empty()) {
        let resolved = resolve_plan(path, base_dir);
        plan::parse(&std::fs::read_to_string(&resolved).with_context(|| format!("reading plan {}", resolved.display()))?)?
    } else {
        let (w, h) = plan::parse_size(cfg.size.as_deref(), (1216, 832));
        let draft = cfg.draft_model.clone().unwrap_or_else(|| "sdxl".into());
        let provider = cfg.provider.clone().unwrap_or_else(|| crate::llm::DEFAULT_ALIAS.to_string());
        let prose = cfg.prompt.clone().unwrap_or_default();
        let hjson = planner::plan_prose(&prose, w, h, &draft, &provider, &device, seed).await?;
        plan::parse(&hjson)?
    };

    let (w, h) = plan::parse_size(cfg.size.as_deref().or(layer_plan.size.as_deref()), (1216, 832));
    let geom = lint::geometry_for_model(&model);
    let draft_model = cfg.draft_model.clone().or_else(|| layer_plan.draft.model.clone()).unwrap_or_else(|| "sdxl".into());
    let opts = render::RenderOpts {
        model,
        out,
        draft_model,
        draft_steps: cfg.draft_steps.unwrap_or(8),
        steps: cfg.steps.unwrap_or(30),
        guidance: cfg.guidance.unwrap_or(7.0),
        seed,
        scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
        ramp: cfg.ramp.unwrap_or(0.1),
        guide: crate::layered::guide::GuideOpts::default(),
        verify: false,
        adapt: false,
        adapt_rounds: 1,
        adapt_boost: 0.12,
        repair: false,
        repair_rounds: 1,
        verify_threshold: 0.1,
        repair_strength: 0.6,
        keep: None,
    };
    render::render(&layer_plan, &geom, w, h, device, &opts).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_needs_a_plan_or_a_prompt() {
        let base = Path::new(".");
        assert!(validate(&LayeredTaskCfg::default(), base).is_err(), "empty cfg is an error");
        assert!(validate(&LayeredTaskCfg { prompt: Some("a busy market square".into()), ..Default::default() }, base).is_ok(), "prose is enough");
        assert!(validate(&LayeredTaskCfg { plan: Some("/no/such/plan.hjson".into()), ..Default::default() }, base).is_err(), "a missing plan file is an error");
    }
}
