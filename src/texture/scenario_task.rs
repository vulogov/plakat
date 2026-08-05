//! `type: texture` scenario task (6.3.0 B7) — synthesise a seamless PBR material as one task in a batch
//! `scenario` file, driving the shared render core ([`crate::texture::render::render_material`]). The
//! task body is a `texture:` block: either an inline `spec` or a `spec_file`, plus render knobs.
//!
//! ```hjson
//! tasks: [
//!   { name: "cobbles", type: "texture", texture: { spec: { material: "mossy cobblestone" }, upscale: "2k" } }
//! ]
//! ```

use crate::texture::compile;
use crate::texture::render::{self, RenderOpts};
use crate::texture::spec::TextureSpec;
use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;
use std::path::Path;

/// The `texture:` task body.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TextureTaskCfg {
    /// An inline `TextureSpec` (the material to render).
    pub spec: Option<TextureSpec>,
    /// …or load the spec from a file (takes precedence over `spec`).
    pub spec_file: Option<String>,
    /// Seed; falls back to the scenario's per-task seed.
    pub seed: Option<u64>,
    /// Tiled upscale override (`none`/`2k`/`4k`).
    pub upscale: Option<String>,
    pub attempts: Option<u32>,
}

impl TextureTaskCfg {
    fn spec(&self) -> Result<TextureSpec> {
        if let Some(p) = &self.spec_file {
            TextureSpec::load(Path::new(p)).with_context(|| format!("loading texture spec_file {p:?}"))
        } else {
            self.spec.clone().context("a `type: texture` task needs a `spec:` (inline) or a `spec_file:`")
        }
    }
}

/// Validate a texture task up front (spec sources + resolves) — before any model load.
pub fn validate(cfg: &TextureTaskCfg) -> Result<()> {
    let _ = compile::resolve(&cfg.spec()?);
    Ok(())
}

/// Run a `type: texture` task → a material directory under `<out_dir>`.
pub async fn run_texture_task(cfg: &TextureTaskCfg, task_seed: u64, _device: Device, out_dir: &Path, dry_run: bool) -> Result<()> {
    let mut spec = cfg.spec()?;
    if cfg.seed.is_some() {
        spec.seed = cfg.seed;
    } else if spec.seed.is_none() {
        spec.seed = Some(task_seed);
    }
    let opts = RenderOpts { attempts: cfg.attempts.unwrap_or(1), upscale: cfg.upscale.clone() };
    if dry_run {
        let plan = compile::resolve(&spec);
        let what = if plan.material.is_empty() { "image-to-material".to_string() } else { plan.material.clone() };
        crate::ui::progress::println(&format!("  [dry-run] texture {}² seamless · {what} → {}/", plan.size, out_dir.display()));
        return Ok(());
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let _sc = render::render_material(&spec, out_dir, &opts).await?;
    Ok(())
}
