//! `type: bookart` scenario task (6.1.0 A2) — compose a book ornament as one task in a batch
//! `scenario` file, driving the shared render core ([`crate::bookart::render_spec`]). The task body is a
//! `bookart:` block: either an inline `spec` or a `spec_file`, plus render knobs.
//!
//! ```hjson
//! tasks: [
//!   {
//!     name: "chapter-head"
//!     type: "bookart"
//!     bookart: {
//!       spec: { origin: "russian", technique: "line", ornament: { type: "headpiece" } }
//!       seed: 7
//!       svg: true
//!     }
//!   }
//! ]
//! ```

use crate::bookart::spec::BookArtSpec;
use crate::bookart::{finish, render, resolve};
use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;
use std::path::Path;

/// The `bookart:` task body.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct BookartTaskCfg {
    /// An inline `BookArtSpec` (the ornament/kit to render).
    pub spec: Option<BookArtSpec>,
    /// …or load the spec from a file (takes precedence over `spec` if both are set).
    pub spec_file: Option<String>,
    /// Base model for the diffusion/composite tiers (default `sd15`).
    pub model: Option<String>,
    /// Seed; falls back to the scenario's per-task seed.
    pub seed: Option<u64>,
    pub steps: Option<usize>,
    /// Also emit born-vector SVG (procedural tier).
    pub svg: Option<bool>,
    pub attempts: Option<u32>,
}

impl BookartTaskCfg {
    fn spec(&self) -> Result<BookArtSpec> {
        if let Some(p) = &self.spec_file {
            BookArtSpec::load(Path::new(p)).with_context(|| format!("loading bookart spec_file {p:?}"))
        } else {
            self.spec.clone().context("a `type: bookart` task needs a `spec:` (inline) or a `spec_file:`")
        }
    }
}

/// Validate a bookart task up front (spec sources + resolves) — before any model load.
pub fn validate(cfg: &BookartTaskCfg) -> Result<()> {
    let _ = resolve(&cfg.spec()?);
    Ok(())
}

/// Run a `type: bookart` task → `<out_dir>/ornament.png` (+ `ornament.svg` on request).
pub async fn run_bookart_task(cfg: &BookartTaskCfg, task_seed: u64, _device: Device, out_dir: &Path, dry_run: bool) -> Result<()> {
    let spec = cfg.spec()?;
    let opts = render::RenderOpts {
        model: cfg.model.clone().unwrap_or_else(|| "sd15".into()),
        seed: cfg.seed.unwrap_or(task_seed),
        steps: cfg.steps.unwrap_or(28),
        svg: cfg.svg.unwrap_or(false),
        attempts: cfg.attempts.unwrap_or(1),
    };
    if dry_run {
        let plan = resolve(&spec);
        crate::ui::progress::println(&format!("  [dry-run] bookart {} · {} → {}/ornament.png", plan.tier, plan.ornament_kind, out_dir.display()));
        return Ok(());
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let r = render::render_spec(&spec, &opts).await?;
    finish::canvas::save_png_dpi(&r.page, &out_dir.join("ornament.png"), r.plan.page.dpi)?;
    if let Some(svg) = &r.svg {
        std::fs::write(out_dir.join("ornament.svg"), svg)?;
    }
    Ok(())
}
