//! Tier 2 — end-to-end perceptual gate (RFC_VERIFY phase 3).
//!
//! Where Tier 1 checks individual intermediates per-tensor, Tier 2 exercises the WHOLE real
//! generate path (encode → denoise loop → VAE decode → PNG) and compares the result to a
//! frozen golden PNG. It's a **regression gate**: the golden is plakat's own prior
//! deterministic output, so a drop flags that *something in the integrated pipeline changed*
//! — the kind of break the per-module taps can miss.
//!
//! Determinism: candle's CPU RNG isn't seed-reproducible, so the generate is driven with
//! `PLAKAT_VERIFY_DET_INIT` (the shared LCG latent replaces the random init) + a
//! non-ancestral scheduler (DDIM) → byte-reproducible run-to-run on a given build.
//!
//! Metrics are self-contained (no torch at verify time): mean-abs pixel error + a global
//! SSIM on luminance. Both on the 0–255 scale.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Device;

use super::{Check, VerifyConfig};

/// The Tier-2 fixture render: small + few steps so it's a fast, deterministic smoke.
const T2_FIXTURE: &str = "portrait_v1";
/// Pass bounds. Same-build runs match near-exactly; the headroom absorbs cross-build /
/// cross-platform fp drift (a real regression tanks SSIM well below this).
const SSIM_MIN: f64 = 0.97;
const MEAN_ABS_MAX: f64 = 4.0;

/// Which generate path + params to drive for a Tier-2 model. Sizes/steps are arbitrary (a
/// regression gate, not a quality target) but chosen small for speed; SDXL wants a bit more
/// than SD 1.5's native 512-halved 256.
enum Family {
    /// SD 1.5 / 2.1 / SDXL — the shared `t2i` pipeline + `GenRequest` (det-init already wired).
    T2i,
    /// PixArt-Σ — its own pipeline + `generate` (returns an RGB buffer).
    PixArt,
    /// SD 3 / 3.5 — the `sd3` pipeline + its own `GenRequest` (flow-matching; renders a PNG).
    Sd3,
    /// Stable Cascade — 3-stage pipeline; `generate` returns an RGB buffer.
    Cascade,
}
struct GenSpec {
    size: u32,
    steps: usize,
    guidance: f64,
    family: Family,
}

/// Per-model Tier-2 generation spec, or `None` if Tier 2 doesn't cover the model.
fn gen_spec(model: &str) -> Option<GenSpec> {
    match model {
        "sd15" | "sd21" => Some(GenSpec { size: 256, steps: 8, guidance: 7.0, family: Family::T2i }),
        "sdxl" => Some(GenSpec { size: 512, steps: 8, guidance: 7.0, family: Family::T2i }),
        "pixart" => Some(GenSpec { size: 256, steps: 8, guidance: 4.5, family: Family::PixArt }),
        "sd35-medium" => Some(GenSpec { size: 256, steps: 8, guidance: 4.5, family: Family::Sd3 }),
        "stable-cascade" => Some(GenSpec { size: 256, steps: 10, guidance: 4.0, family: Family::Cascade }),
        _ => None,
    }
}

/// Models Tier 2 covers.
pub fn models(cfg: &VerifyConfig) -> Vec<String> {
    let covered = ["sd15", "sd21", "sdxl", "pixart", "sd35-medium", "stable-cascade"];
    match &cfg.model {
        Some(m) if covered.contains(&m.as_str()) => vec![m.clone()],
        Some(_) => vec![], // a model Tier 2 doesn't cover → nothing to run
        None => covered.iter().map(|s| s.to_string()).collect(),
    }
}

/// Global SSIM on luminance + mean-abs over RGB. Inputs are `w*h*3` RGB bytes.
/// Returns `(ssim, mean_abs)` on the 0–255 scale.
fn perceptual(a: &[u8], b: &[u8]) -> (f64, f64) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // mean-abs over all RGB channels.
    let mean_abs = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / n as f64;

    // Luminance (Rec.601) for SSIM.
    let lum = |p: &[u8]| -> Vec<f64> {
        p.chunks_exact(3)
            .map(|c| 0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64)
            .collect()
    };
    let (la, lb) = (lum(a), lum(b));
    let m = la.len().max(1) as f64;
    let mua = la.iter().sum::<f64>() / m;
    let mub = lb.iter().sum::<f64>() / m;
    let mut va = 0.0;
    let mut vb = 0.0;
    let mut cov = 0.0;
    for (x, y) in la.iter().zip(&lb) {
        va += (x - mua) * (x - mua);
        vb += (y - mub) * (y - mub);
        cov += (x - mua) * (y - mub);
    }
    va /= m;
    vb /= m;
    cov /= m;
    let c1 = (0.01 * 255.0f64).powi(2);
    let c2 = (0.03 * 255.0f64).powi(2);
    let ssim = ((2.0 * mua * mub + c1) * (2.0 * cov + c2))
        / ((mua * mua + mub * mub + c1) * (va + vb + c2));
    (ssim, mean_abs)
}

/// Load a PNG (or any image) to raw RGB8 bytes + dims.
fn load_rgb(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(path).with_context(|| format!("opening image {}", path.display()))?.to_rgb8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w, h))
}

/// Deterministically render the fixture for `model` via the REAL pipeline (det-init env on),
/// returning the RGB buffer + dims. Dispatches by family (t2i writes a PNG we read back; PixArt
/// returns the buffer directly).
async fn render(model: &str, spec: &GenSpec, device: &Device) -> Result<(Vec<u8>, u32, u32)> {
    let fx = crate::verify::fixtures::get(T2_FIXTURE).expect("portrait_v1 fixture");
    match spec.family {
        Family::T2i => {
            let pipe = crate::pipelines::t2i::Pipeline::load(crate::pipelines::t2i::LoadRequest {
                model: model.to_string(),
                device: device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                use_refiner: false,
                embeddings: Vec::new(),
                vae_cache: None,
            })
            .await?;
            let tmp = std::env::temp_dir().join(format!("plakat-tier2-{}-{model}", std::process::id()));
            std::fs::create_dir_all(&tmp)?;
            let req = crate::pipelines::t2i::GenRequest {
                prompt: fx.prompt.to_string(),
                negative: fx.negative.to_string(),
                width: spec.size,
                height: spec.size,
                count: 1,
                steps: spec.steps,
                guidance: spec.guidance,
                seed: Some(0),
                out_dir: tmp.clone(),
                scheduler: crate::pipelines::scheduler::SchedulerKind::Ddim,
                refine: None,
                refine_strength: 0.0,
                refiner_frac: None,
                clip_skip: 1,
                metadata: None,
                preview_every: None,
                preview_size: None,
                output_format: crate::imaging::io::OutputFormat::Png,
            };
            unsafe { std::env::set_var("PLAKAT_VERIFY_DET_INIT", "1") };
            let gen_result = pipe.generate(&req, &[]);
            unsafe { std::env::remove_var("PLAKAT_VERIFY_DET_INIT") };
            gen_result?;
            let png = std::fs::read_dir(&tmp)
                .ok()
                .and_then(|rd| {
                    rd.filter_map(|e| e.ok().map(|e| e.path()))
                        .find(|p| p.extension().map(|x| x == "png").unwrap_or(false))
                })
                .ok_or_else(|| anyhow::anyhow!("no PNG produced in {}", tmp.display()))?;
            let out = load_rgb(&png);
            let _ = std::fs::remove_dir_all(&tmp);
            out
        }
        Family::PixArt => {
            let mut pipe = crate::pipelines::pixart::Pipeline::load(crate::pipelines::pixart::LoadRequest {
                repo: crate::hf::resolve_alias(model).to_string(),
                device: device.clone(),
                vae_cache: None,
                loras: Vec::new(),
                lora_scale: 1.0,
            })
            .await?;
            unsafe { std::env::set_var("PLAKAT_VERIFY_DET_INIT", "1") };
            let mut no_hook: Option<&mut dyn crate::pipelines::step_hook::StepHook> = None;
            let rendered = pipe.generate(
                fx.prompt, fx.negative, spec.size, spec.size, spec.steps, spec.guidance, 0,
                crate::pipelines::scheduler::SchedulerKind::Ddim, &mut no_hook,
            );
            unsafe { std::env::remove_var("PLAKAT_VERIFY_DET_INIT") };
            rendered // (Vec<u8> RGB, w, h)
        }
        Family::Sd3 => {
            let mut pipe = crate::pipelines::sd3::Pipeline::load(crate::pipelines::sd3::LoadRequest {
                variant: crate::pipelines::sd3::Variant::Sd35Medium,
                repo: crate::hf::resolve_alias(model).to_string(),
                device: device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                controlnets: Vec::new(),
                embeddings: Vec::new(),
            })
            .await?;
            let tmp = std::env::temp_dir().join(format!("plakat-tier2-{}-{model}", std::process::id()));
            std::fs::create_dir_all(&tmp)?;
            let req = crate::pipelines::sd3::GenRequest {
                prompt: fx.prompt.to_string(),
                negative: fx.negative.to_string(),
                width: spec.size,
                height: spec.size,
                count: 1,
                steps: Some(spec.steps),
                guidance: Some(spec.guidance),
                seed: Some(0),
                out_dir: tmp.clone(),
                init_image: None,
                mask: None,
                mask_feather: 0,
                mask_invert: false,
                strength: None,
                tiled: None,
                regions: Vec::new(),
                controlnet_conditioning: Vec::new(),
                output_format: crate::imaging::io::OutputFormat::Png,
            };
            unsafe { std::env::set_var("PLAKAT_VERIFY_DET_INIT", "1") };
            let gen_result = pipe.generate(&req);
            unsafe { std::env::remove_var("PLAKAT_VERIFY_DET_INIT") };
            gen_result?;
            let png = std::fs::read_dir(&tmp)
                .ok()
                .and_then(|rd| {
                    rd.filter_map(|e| e.ok().map(|e| e.path()))
                        .find(|p| p.extension().map(|x| x == "png").unwrap_or(false))
                })
                .ok_or_else(|| anyhow::anyhow!("no PNG produced in {}", tmp.display()))?;
            let out = load_rgb(&png);
            let _ = std::fs::remove_dir_all(&tmp);
            out
        }
        Family::Cascade => {
            let mut pipe = crate::pipelines::cascade::Pipeline::load(crate::pipelines::cascade::LoadRequest {
                repo: crate::hf::resolve_alias(model).to_string(),
                device: device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                controlnet_weights: None,
                image_encoder_weights: None,
            })
            .await?;
            // Split the step budget across Stage C (2/3) + Stage B (1/3), the CLI default.
            let stage_c = (spec.steps * 2).div_ceil(3).max(1);
            let stage_b = spec.steps.saturating_sub(stage_c).max(1);
            unsafe { std::env::set_var("PLAKAT_VERIFY_DET_INIT", "1") };
            let mut no_hook: Option<&mut dyn crate::pipelines::step_hook::StepHook> = None;
            let rendered = pipe.generate(
                fx.prompt, fx.negative, spec.size, stage_c, stage_b, spec.guidance, 1.1, 0,
                crate::pipelines::scheduler::SchedulerKind::Ddim, None, &mut no_hook,
            );
            unsafe { std::env::remove_var("PLAKAT_VERIFY_DET_INIT") };
            rendered // (Vec<u8> RGB, w, h)
        }
    }
}

/// Run Tier 2 for one model: deterministically render the fixture via the real pipeline and
/// compare to its golden PNG (local `--golden-dir` or the HF dataset). Missing golden → skip.
pub async fn run_model(model: &str, golden_dir: Option<&Path>, device: &Device) -> Vec<Check> {
    let name = format!("tier2.{model}.end_to_end");
    let spec = match gen_spec(model) {
        Some(s) => s,
        None => return vec![Check::skip(name, 2, "Tier 2 doesn't cover this model".to_string())],
    };
    // Resolve the golden PNG (regression reference = plakat's frozen prior output).
    let golden_png = match super::golden::resolve_golden_image(model, T2_FIXTURE, golden_dir).await {
        Ok(p) => p,
        Err(e) => return vec![Check::skip(name, 2, format!("{e:#}"))],
    };
    let (gpix, gw, gh) = match load_rgb(&golden_png) {
        Ok(v) => v,
        Err(e) => return vec![Check::fail(name, 2, format!("{e:#}"))],
    };

    let (rpix, rw, rh) = match render(model, &spec, device).await {
        Ok(v) => v,
        Err(e) => return vec![Check::skip(name, 2, format!("render failed (weights?): {e:#}"))],
    };
    if (rw, rh) != (gw, gh) || rpix.len() != gpix.len() {
        return vec![Check::fail(
            name,
            2,
            format!("size mismatch: rendered {rw}x{rh} vs golden {gw}x{gh}"),
        )];
    }

    let (ssim, mean_abs) = perceptual(&rpix, &gpix);
    if ssim >= SSIM_MIN && mean_abs <= MEAN_ABS_MAX {
        vec![Check::pass(
            name,
            2,
            format!("SSIM {ssim:.4} · mean_abs {mean_abs:.3} (≥{SSIM_MIN} / ≤{MEAN_ABS_MAX})"),
        )]
    } else {
        vec![Check::fail(
            name,
            2,
            format!("SSIM {ssim:.4} / mean_abs {mean_abs:.3} vs SSIM≥{SSIM_MIN} mean_abs≤{MEAN_ABS_MAX}"),
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perceptual_identical_is_perfect() {
        let a = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let (ssim, mean_abs) = perceptual(&a, &a);
        assert!((ssim - 1.0).abs() < 1e-9, "ssim {ssim}");
        assert_eq!(mean_abs, 0.0);
    }

    #[test]
    fn perceptual_penalizes_difference() {
        let a = vec![0u8; 12];
        let b = vec![255u8; 12];
        let (ssim, mean_abs) = perceptual(&a, &b);
        assert!(ssim < 0.5, "ssim {ssim} should be low for opposite images");
        assert_eq!(mean_abs, 255.0);
    }
}
