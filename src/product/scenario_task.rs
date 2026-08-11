//! `type: product` scenario task (RFC PRODUCT-1) — render a studio product-shot as one task in a batch
//! `scenario` file, driving the shared core ([`crate::product::render::render_spec`]). The task body is a
//! `product:` block: either an inline `spec` or a `spec_file`, plus render knobs.
//!
//! ```hjson
//! tasks: [
//!   { name: "sneaker", type: "product", product: { spec_file: "sneaker.hjson", relight: true } }
//! ]
//! ```

use crate::product::render::{self, RenderOpts};
use crate::product::spec::ProductSpec;
use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;
use std::path::Path;

/// The `product:` task body.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ProductTaskCfg {
    /// An inline `ProductSpec`.
    pub spec: Option<ProductSpec>,
    /// …or load the spec from a file (takes precedence over `spec`).
    pub spec_file: Option<String>,
    /// Seed; falls back to the scenario's per-task seed.
    pub seed: Option<u64>,
    /// Override the subject cutout.
    pub subject: Option<String>,
    /// Relight to the `lighting` rig (else the spec's `lighting:` block opts in).
    pub relight: Option<bool>,
    /// Render a catalog contact sheet instead of a single shot.
    pub sheet: Option<bool>,
}

impl ProductTaskCfg {
    fn spec(&self) -> Result<ProductSpec> {
        if let Some(p) = &self.spec_file {
            ProductSpec::load(Path::new(p)).with_context(|| format!("loading product spec_file {p:?}"))
        } else {
            self.spec.clone().context("a `type: product` task needs a `spec:` (inline) or a `spec_file:`")
        }
    }
}

/// Validate a product task up front (spec sources + lint errors) — before any model load.
pub fn validate(cfg: &ProductTaskCfg) -> Result<()> {
    let spec = cfg.spec()?;
    let errs = crate::product::lint::lint(&spec).into_iter().filter(|f| f.level == crate::product::lint::Level::Error).count();
    if errs > 0 {
        anyhow::bail!("product task has {errs} lint error(s)");
    }
    Ok(())
}

/// Run a `type: product` task → a packshot (`shot.png`, or `sheet.png`) under `<out_dir>`.
pub async fn run_product_task(cfg: &ProductTaskCfg, task_seed: u64, device: Device, out_dir: &Path, dry_run: bool) -> Result<()> {
    let mut spec = cfg.spec()?;
    if cfg.seed.is_some() {
        spec.seed = cfg.seed;
    } else if spec.seed.is_none() {
        spec.seed = Some(task_seed);
    }
    if dry_run {
        let plan = crate::product::compose::resolve(&spec);
        crate::ui::progress::println(&format!("  [dry-run] product {}×{} px → {}/", plan.w, plan.h, out_dir.display()));
        return Ok(());
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let dev_sel = match &device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    };
    let opts = RenderOpts {
        subject: cfg.subject.as_ref().map(std::path::PathBuf::from),
        relight: cfg.relight.unwrap_or(false) || spec.lighting.is_some(),
        device: Some(dev_sel.to_string()),
    };
    if cfg.sheet.unwrap_or(false) {
        let _ = render::render_sheet(&spec, &opts, &out_dir.join("sheet.png")).await?;
    } else {
        let _ = render::render_spec(&spec, &out_dir.join("shot.png"), &opts).await?;
    }
    Ok(())
}
