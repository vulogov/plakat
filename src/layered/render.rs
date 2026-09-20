//! LAYERED-1 S3 — the **render** orchestration. Ties the three stages together:
//!
//!   1. **draft** (S1): each anchored subject + the backdrop are rendered alone ([`super::draft`]).
//!   2. **guide** (S2): the drafts are composed into the guide image + anchor maps ([`super::guide`]).
//!   3. **finish** (S3): the guide image is VAE-encoded into `G` by the FINISH model, and ONE ordinary
//!      txt2img trajectory of that model is steered toward `G`'s low frequencies inside each anchored pixel's
//!      window ([`super::hook::LayeredHook`], via the `refine_latent` seam wired across every family in P0).
//!
//! Bring-up is the SD family (SD 1.5 / SDXL) — the finish runs through [`crate::pipelines::t2i::run`] with a
//! `Request.layered` attached. Flux / SD3 / Sana / PixArt / Cascade finishes land next (the `LayeredGuide`
//! carried on the request is family-agnostic; only each family's `run` needs the encode+hook injection).

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::PathBuf;

use crate::compile::{classify_model, ModelFamily};
use crate::layered::draft::{self, DraftOpts};
use crate::layered::guide;
use crate::layered::plan::LayerPlan;
use crate::pipelines::noise_space::LatentGeometry;
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::t2i::{self, LayeredGuide};

/// A mild, generic finish negative (weight-free; no scene specifics).
const FINISH_NEGATIVE: &str = "lowres, blurry, deformed, bad anatomy, watermark, signature, text";

/// Options for [`render`].
pub struct RenderOpts {
    /// The finish model (SD family for bring-up).
    pub model: String,
    /// Output image path.
    pub out: PathBuf,
    /// The draft model (fast preset welcome).
    pub draft_model: String,
    pub draft_steps: usize,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    pub scheduler: SchedulerKind,
    /// The per-pixel window-close ramp `r` (default 0.1).
    pub ramp: f32,
    /// Guide-cohesion controls (grounding / colour-harmonisation / relight).
    pub guide: crate::layered::guide::GuideOpts,
    /// After the finish, verify each anchored/hinted subject actually landed (OWL-ViT, no diffusion).
    pub verify: bool,
    /// Auto-repair the layers verify flags as missing (implies verify) — a masked pass per failing layer.
    pub repair: bool,
    /// Max repair rounds. Default 1.
    pub repair_rounds: usize,
    /// OWL-ViT detection threshold for verify. Default 0.1.
    pub verify_threshold: f32,
    /// Inpaint strength for auto-repair. Default 0.6.
    pub repair_strength: f32,
    /// Also write the intermediate drafts + guide + anchor maps into this directory.
    pub keep: Option<PathBuf>,
}

/// The finish prompt: the plan's finish prompt (or, if absent, the layer + backdrop prompts) plus the global
/// medium / palette / light — the medium is anchored ONLY here (it never reaches the drafts). Nothing
/// scene-specific is invented; everything is derived from the plan.
fn finish_prompt(plan: &LayerPlan) -> String {
    let base = plan.prompt.clone().filter(|p| !p.trim().is_empty()).unwrap_or_else(|| {
        let mut parts: Vec<String> = plan.layers.iter().filter_map(|l| l.prompt.clone()).collect();
        if let Some(b) = plan.backdrop.as_ref().and_then(|b| b.prompt.clone()) {
            parts.push(b);
        }
        parts.join(", ")
    });
    let g = &plan.global;
    let mut out = base.trim().to_string();
    for extra in [g.medium.as_deref(), g.palette.as_deref(), g.light.as_deref()].into_iter().flatten() {
        let extra = extra.trim();
        if !extra.is_empty() && !out.to_lowercase().contains(&extra.to_lowercase()) {
            if !out.is_empty() {
                out.push_str(", ");
            }
            out.push_str(extra);
        }
    }
    out
}

/// The draft family's native area side (SD 1.5 = 512, else 1024).
fn native_side(model: &str) -> u32 {
    match classify_model(model) {
        ModelFamily::Sd15 => 512,
        _ => 1024,
    }
}

/// The single PNG a finish run wrote into `dir`.
fn first_png(dir: &std::path::Path) -> Result<PathBuf> {
    std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("png")).unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("the finish produced no PNG in {}", dir.display()))
}

/// Run the full layered pipeline (S1 → S2 → S3) and write the finished image to `o.out`.
pub async fn render(plan: &LayerPlan, geom: &LatentGeometry, out_w: u32, out_h: u32, device: Device, o: &RenderOpts) -> Result<()> {
    // Bring-up gate: the finish injection lives in the SD path (`t2i::run`) and the Flux path (`flux::run`).
    let v = t2i::Variant::detect(&o.model);
    anyhow::ensure!(
        !(v.is_sd3() || v.is_sana() || v.is_pixart() || v.is_cascade()),
        "layered render bring-up is the SD family (SD 1.5 / SDXL) + Flux; {} lands next — SD3 / Sana / PixArt / Cascade need their own encode+hook injection",
        o.model
    );

    // S1 — drafts (on the same device as the finish).
    let dopts = DraftOpts {
        out_w,
        out_h,
        model: o.draft_model.clone(),
        steps: o.draft_steps,
        scheduler: SchedulerKind::default(),
        native_side: native_side(&o.draft_model),
        only: None,
        device: crate::device::spec_of(&device).to_string(),
    };
    let drafts = draft::render_all(plan, &dopts).await.context("S1 draft stage")?;

    // S2 — guide (matte + compose + anchor maps).
    let matter = crate::pipelines::matting::Matter::load(&device).await.context("loading the U2Net matter")?;
    let g = guide::build(plan, &drafts, geom, out_w, out_h, &device, &o.guide, |img| matter.matte(img)).context("S2 guide stage")?;

    // The finish VAE reads the guide from a file (shared img2img preprocess path); keep it alive over run().
    let tmp = tempfile::Builder::new().prefix("plakat-layered-").tempdir().context("layered scratch dir")?;
    let guide_png = tmp.path().join("guide.png");
    g.canvas.save(&guide_png).with_context(|| format!("saving the guide image {}", guide_png.display()))?;

    if let Some(dir) = &o.keep {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        drafts.backdrop.save(dir.join("__backdrop.png")).ok();
        for (id, img) in &drafts.layers {
            img.save(dir.join(format!("{}.png", crate::cli::layers::sanitize(id)))).ok();
        }
        g.canvas.save(dir.join("__guide.png")).ok();
        if let Ok(w) = guide::map_to_gray(&g.weight) {
            image::imageops::resize(&w, out_w, out_h, image::imageops::FilterType::Nearest).save(dir.join("__weight.png")).ok();
        }
        if let Ok(e) = guide::map_to_gray(&g.window_end) {
            image::imageops::resize(&e, out_w, out_h, image::imageops::FilterType::Nearest).save(dir.join("__window.png")).ok();
        }
    }

    // S3 — the anchored finish trajectory.
    let vdevice = if o.verify || o.repair { Some(device.clone()) } else { None };
    let out_dir = tempfile::Builder::new().prefix("plakat-layered-out-").tempdir().context("layered output dir")?;
    let mut req = t2i::Request::simple(finish_prompt(plan), o.model.clone(), out_w, out_h, o.steps, Some(o.seed), device, out_dir.path().to_path_buf());
    req.negative = FINISH_NEGATIVE.to_string();
    req.guidance = o.guidance;
    req.scheduler = o.scheduler;
    req.count = 1;
    req.layered = Some(LayeredGuide {
        guide_path: guide_png.clone(),
        weight: g.weight,
        window_end: g.window_end,
        ramp: o.ramp,
        guide_seed: o.seed,
    });
    t2i::run(req).await.context("S3 finish stage")?;
    let mut produced = first_png(out_dir.path())?;

    // S4 — verify each anchored/hinted subject landed (OWL-ViT), and optionally auto-repair the misses with
    // a masked pass, re-verifying until clean or the round budget is spent.
    if let Some(vdev) = vdevice {
        let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(&vdev).await.context("loading OWL-ViT for verify")?;
        let rounds = if o.repair { o.repair_rounds.max(1) } else { 0 };
        for round in 0..=rounds {
            let report = {
                let img = produced.clone();
                let detect = |q: &str| -> Result<Vec<crate::layered::verify::Det>> {
                    let dets = owl.detect_all(&img, q, o.verify_threshold, 8)?;
                    Ok(dets.into_iter().map(|d| crate::layered::verify::Det { x0: d.x0, y0: d.y0, x1: d.x1, y1: d.y1, score: d.score }).collect())
                };
                crate::layered::verify::verify(plan, geom, out_w, out_h, &detect)?
            };
            if report.pass() {
                tracing::info!(target: "plakat", "layered verify: all {} checked subject(s) present", report.checked);
                break;
            }
            tracing::info!(target: "plakat", "layered verify: {} missing ({})", report.failures.len(), report.failures.join(", "));
            if round >= rounds {
                break; // out of repair rounds (or verify-only)
            }
            let repaired = out_dir.path().join(format!("repaired-{round}.png"));
            let ropts = crate::layered::repair::RepairOpts {
                model: o.model.clone(),
                out: repaired.clone(),
                strength: o.repair_strength,
                steps: o.steps,
                guidance: o.guidance,
                seed: o.seed.wrapping_add(round as u64 + 1),
                mask_feather: 8,
                device: crate::device::spec_of(&vdev).to_string(),
            };
            crate::layered::repair::repair(&produced, plan, &report.failures, &ropts).await.context("S4 repair")?;
            produced = repaired;
        }
    }

    // Move the finished image to the requested path.
    if let Some(parent) = o.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::copy(&produced, &o.out).with_context(|| format!("writing {}", o.out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered::plan::{Backdrop, Global, Layer};

    fn plan_with(prompt: Option<&str>, medium: Option<&str>) -> LayerPlan {
        LayerPlan {
            prompt: prompt.map(Into::into),
            global: Global { palette: Some("muted autumn".into()), light: Some("overcast".into()), medium: medium.map(Into::into) },
            backdrop: Some(Backdrop { prompt: Some("a cobblestone square".into()), weight: None, window: None }),
            layers: vec![
                Layer { id: "a".into(), prompt: Some("a boy in a raincoat".into()), ..Default::default() },
                Layer { id: "b".into(), prompt: Some("a red umbrella".into()), ..Default::default() },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn finish_prompt_uses_plan_prompt_and_anchors_medium() {
        let p = plan_with(Some("two friends in the rain"), Some("oil painting"));
        let fp = finish_prompt(&p);
        assert!(fp.starts_with("two friends in the rain"), "keeps the finish prompt: {fp}");
        assert!(fp.contains("oil painting"), "medium anchored only at the finish");
        assert!(fp.contains("muted autumn") && fp.contains("overcast"), "palette + light carried");
    }

    #[test]
    fn finish_prompt_falls_back_to_layers_and_backdrop() {
        let p = plan_with(None, Some("engraving"));
        let fp = finish_prompt(&p);
        assert!(fp.contains("a boy in a raincoat") && fp.contains("a red umbrella"), "layer prompts: {fp}");
        assert!(fp.contains("cobblestone square"), "backdrop prompt");
        assert!(fp.contains("engraving"));
    }

    #[test]
    fn finish_prompt_does_not_duplicate_present_terms() {
        let p = plan_with(Some("an oil painting of a market"), Some("oil painting"));
        let fp = finish_prompt(&p);
        assert_eq!(fp.matches("oil painting").count(), 1, "medium not doubled when already in the prompt: {fp}");
    }
}
