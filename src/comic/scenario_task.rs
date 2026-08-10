//! `type: comic` scenario task (6.8.0 P4) — render a multi-panel comic page as one task in a batch
//! `scenario` file, driving the shared core ([`crate::comic::render::render_spec`]). The task body is a
//! `comic:` block: either an inline `spec` or a `spec_file`, plus render knobs.
//!
//! ```hjson
//! tasks: [
//!   { name: "strip", type: "comic", comic: { spec_file: "strip.hjson", letter: true } }
//! ]
//! ```

use crate::comic::render::{self, RenderOpts};
use crate::comic::spec::ComicSpec;
use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;
use std::path::Path;

/// The `comic:` task body.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ComicTaskCfg {
    /// An inline `ComicSpec` (the page to render).
    pub spec: Option<ComicSpec>,
    /// …or load the spec from a file (takes precedence over `spec`).
    pub spec_file: Option<String>,
    /// Seed; falls back to the scenario's per-task seed.
    pub seed: Option<u64>,
    /// Draw the balloons/captions (default `true`); `false` renders scene art only.
    pub letter: Option<bool>,
    /// Keep the generated per-panel PNGs (relative to the task output dir).
    pub panels_out: Option<String>,
}

impl ComicTaskCfg {
    fn spec(&self) -> Result<ComicSpec> {
        if let Some(p) = &self.spec_file {
            ComicSpec::load(Path::new(p)).with_context(|| format!("loading comic spec_file {p:?}"))
        } else {
            self.spec.clone().context("a `type: comic` task needs a `spec:` (inline) or a `spec_file:`")
        }
    }
}

/// Validate a comic task up front (spec sources + lint errors) — before any model load.
pub fn validate(cfg: &ComicTaskCfg) -> Result<()> {
    let spec = cfg.spec()?;
    let errs = crate::comic::lint::lint(&spec).into_iter().filter(|f| f.level == crate::comic::lint::Level::Error).count();
    if errs > 0 {
        anyhow::bail!("comic task has {errs} lint error(s)");
    }
    Ok(())
}

/// Run a `type: comic` task → a comic page (`page.png` + sidecar) under `<out_dir>`.
pub async fn run_comic_task(cfg: &ComicTaskCfg, task_seed: u64, device: Device, out_dir: &Path, dry_run: bool) -> Result<()> {
    let mut spec = cfg.spec()?;
    if cfg.seed.is_some() {
        spec.seed = cfg.seed;
    } else if spec.seed.is_none() {
        spec.seed = Some(task_seed);
    }
    if dry_run {
        let plan = crate::comic::layout::resolve(&spec);
        crate::ui::progress::println(&format!("  [dry-run] comic {} panel(s) · {}×{} px → {}/page.png", plan.panels.len(), plan.w, plan.h, out_dir.display()));
        return Ok(());
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let dev_sel = match &device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    };
    let opts = RenderOpts {
        device: Some(dev_sel.to_string()),
        panels_out: cfg.panels_out.as_ref().map(|d| out_dir.join(d)),
        letter: cfg.letter.unwrap_or(true),
    };
    let _rep = render::render_spec(&spec, &out_dir.join("page.png"), &opts).await?;
    Ok(())
}
