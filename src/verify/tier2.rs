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
const T2_MODEL: &str = "sd15";
const T2_FIXTURE: &str = "portrait_v1";
const T2_STEPS: usize = 8;
const T2_SIZE: u32 = 256;
const T2_GUIDANCE: f64 = 7.0;
/// Pass bounds. Same-build runs match near-exactly; the headroom absorbs cross-build /
/// cross-platform fp drift (a real regression tanks SSIM well below this).
const SSIM_MIN: f64 = 0.97;
const MEAN_ABS_MAX: f64 = 4.0;

/// Models Tier 2 covers (just sd15 for now — the cheapest deterministic end-to-end).
pub fn models(cfg: &VerifyConfig) -> Vec<String> {
    match &cfg.model {
        Some(m) if m == T2_MODEL => vec![m.clone()],
        Some(_) => vec![], // a specific non-sd15 model → nothing to run at Tier 2 yet
        None => vec![T2_MODEL.to_string()],
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

/// Run Tier 2 for one model: deterministically render the fixture via the real pipeline and
/// compare to its golden PNG (local `--golden-dir` or the HF dataset). Missing golden → skip.
pub async fn run_model(model: &str, golden_dir: Option<&Path>, device: &Device) -> Vec<Check> {
    let name = format!("tier2.{model}.end_to_end");
    if model != T2_MODEL {
        return vec![Check::skip(name, 2, format!("Tier 2 covers {T2_MODEL} only for now"))];
    }
    // Resolve the golden PNG (regression reference = plakat's frozen prior output).
    let golden_png = match super::golden::resolve_golden_image(model, T2_FIXTURE, golden_dir).await {
        Ok(p) => p,
        Err(e) => return vec![Check::skip(name, 2, format!("{e:#}"))],
    };
    let (gpix, gw, gh) = match load_rgb(&golden_png) {
        Ok(v) => v,
        Err(e) => return vec![Check::fail(name, 2, format!("{e:#}"))],
    };

    // Load the pipeline + render deterministically to a temp dir.
    let pipe = match crate::pipelines::t2i::Pipeline::load(crate::pipelines::t2i::LoadRequest {
        model: model.to_string(),
        device: device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        use_refiner: false,
        embeddings: Vec::new(),
        vae_cache: None,
    })
    .await
    {
        Ok(p) => p,
        Err(e) => return vec![Check::skip(name, 2, format!("load failed (weights?): {e:#}"))],
    };

    let tmp = std::env::temp_dir().join(format!("plakat-tier2-{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&tmp) {
        return vec![Check::fail(name, 2, format!("temp dir: {e}"))];
    }
    let fx = crate::verify::fixtures::get(T2_FIXTURE).expect("portrait_v1 fixture");
    let req = crate::pipelines::t2i::GenRequest {
        prompt: fx.prompt.to_string(),
        negative: fx.negative.to_string(),
        width: T2_SIZE,
        height: T2_SIZE,
        count: 1,
        steps: T2_STEPS,
        guidance: T2_GUIDANCE,
        seed: Some(0), // stable filename; the LCG env override drives the actual latent
        out_dir: tmp.clone(),
        scheduler: crate::pipelines::scheduler::SchedulerKind::Ddim, // deterministic (non-ancestral)
        refine: None,
        refine_strength: 0.0,
        refiner_frac: None,
        clip_skip: 1,
        metadata: None,
        preview_every: None,
        preview_size: None,
        output_format: crate::imaging::io::OutputFormat::Png,
    };

    // Deterministic init for the duration of this render (scoped, then cleared).
    // SAFETY: verify runs single-threaded through the tiers; no concurrent generate.
    unsafe { std::env::set_var("PLAKAT_VERIFY_DET_INIT", "1") };
    let gen_result = pipe.generate(&req, &[]);
    unsafe { std::env::remove_var("PLAKAT_VERIFY_DET_INIT") };
    if let Err(e) = gen_result {
        let _ = std::fs::remove_dir_all(&tmp);
        return vec![Check::fail(name, 2, format!("generate failed: {e:#}"))];
    }

    // Read back the single rendered PNG.
    let rendered = std::fs::read_dir(&tmp)
        .ok()
        .and_then(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .find(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        });
    let result = match rendered {
        Some(p) => load_rgb(&p),
        None => Err(anyhow::anyhow!("no PNG produced in {}", tmp.display())),
    };
    let _ = std::fs::remove_dir_all(&tmp);
    let (rpix, rw, rh) = match result {
        Ok(v) => v,
        Err(e) => return vec![Check::fail(name, 2, format!("{e:#}"))],
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
