//! The render orchestration (RFC PRODUCT-1). P1: the **weight-free** path — a supplied subject cutout →
//! resolve → ground → composite → `shot.png` + `shot.meta.json`. Subject-from-photo/prompt (matte /
//! generate) and relight (IC-Light) land in P2; this is the shared core they will extend.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use super::compose::{self, Plan};
use super::spec::ProductSpec;

/// Options for [`render_spec`].
#[derive(Debug, Clone, Default)]
pub struct RenderOpts {
    /// Override the spec's `subject.image` with this cutout path.
    pub subject: Option<PathBuf>,
    /// P2: relight the subject to the `lighting` rig (needs a model). Off (weight-free) in P1.
    pub relight: bool,
    /// Device selector (P2 — generation / relight / matte).
    pub device: Option<String>,
}

/// What a render produced.
#[derive(Debug, Clone)]
pub struct Report {
    pub shot: PathBuf,
    pub sidecar: PathBuf,
    pub w: u32,
    pub h: u32,
    pub weight_free: bool,
}

/// Resolve the subject to a cutout image. P1: only a supplied transparent PNG (`subject.image` or the
/// `--subject` override). A `photo`/`prompt` subject errors here (P2 wires matte/generate).
fn load_subject(spec: &ProductSpec, opts: &RenderOpts) -> Result<image::DynamicImage> {
    let path = opts
        .subject
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .or_else(|| spec.subject_image().map(|s| s.to_string()))
        .context("no subject cutout — set `subject.image` (a transparent PNG) or pass --subject (P1 is cutout-only; photo/prompt are P2)")?;
    let img = image::open(&path).with_context(|| format!("reading subject {path}"))?;
    if img.color().has_alpha() {
        // check it's actually a cutout (some transparency), else warn — a fully-opaque image grounds as a
        // rectangle.
        let rgba = img.to_rgba8();
        if rgba.pixels().all(|p| p.0[3] == 255) {
            tracing::warn!(target: "plakat", "product: subject `{path}` has no transparency — grounding a full rectangle. Use a cutout (or a `photo`/`prompt` subject in P2).");
        }
    } else {
        tracing::warn!(target: "plakat", "product: subject `{path}` has no alpha channel — grounding a full rectangle. Use a transparent cutout.");
    }
    Ok(img)
}

/// The full render: resolve → (P2 relight) → ground + composite → write shot + sidecar.
pub fn render_spec(spec: &ProductSpec, out: &Path, opts: &RenderOpts) -> Result<Report> {
    let plan: Plan = compose::resolve(spec);
    let subject = load_subject(spec, opts)?;
    // P2 will relight `subject` here when opts.relight; P1 keeps the cutout's own light (weight-free).
    let shot = compose::compose(&plan, &subject);
    shot.save(out).with_context(|| format!("writing {}", out.display()))?;
    let sidecar = out.with_extension("meta.json");
    std::fs::write(&sidecar, compose::meta_json(spec, &plan)).with_context(|| format!("writing {}", sidecar.display()))?;
    Ok(Report { shot: out.to_path_buf(), sidecar, w: plan.w, h: plan.h, weight_free: !opts.relight })
}
