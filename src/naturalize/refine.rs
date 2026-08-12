//! Corrective refine (RFC QUALITY-1) — the **model-backed** half of the focus qualifiers. Grain and grade
//! can't fix a bad hand, incoherent joinery, or make two lookalikes distinct, so these focuses route to
//! img2img / inpaint instead of the analog pass:
//!   - `geometry` / `anatomy` → a whole-image **img2img** at a weight-scaled strength with a corrective
//!     prompt (let the model re-resolve incoherent structure / proportions),
//!   - `no-twins` → **detect** faces (SCRFD) and **inpaint** each duplicate face with a distinct seed so
//!     lookalikes diverge.
//! Runs BEFORE the analog pass; needs a model (best-effort — a missing detector just skips `no-twins`).

use anyhow::{Context, Result};
use image::{GrayImage, Luma};
use std::path::{Path, PathBuf};

/// The corrective focus weights (0 = off).
#[derive(Debug, Clone, Copy, Default)]
pub struct Corrective {
    pub geometry: f32,
    pub anatomy: f32,
    pub no_twins: f32,
}

impl Corrective {
    pub fn any(&self) -> bool {
        self.geometry > 0.0 || self.anatomy > 0.0 || self.no_twins > 0.0
    }
}

/// Run the corrective refine on `input`, writing the result to `out`. Steps use `model`/`device`.
pub async fn refine(input: &Path, out: &Path, c: &Corrective, model: &str, device: Option<&str>, steps: usize, tmp: &Path) -> Result<()> {
    let mut current = input.to_path_buf();

    // 1. geometry / anatomy → whole-image img2img with a corrective prompt.
    if c.geometry > 0.0 || c.anatomy > 0.0 {
        let mut parts: Vec<&str> = Vec::new();
        if c.geometry > 0.0 {
            parts.push("coherent geometry, correct structure, consistent perspective, clean continuous lines, well-formed objects");
        }
        if c.anatomy > 0.0 {
            parts.push("correct anatomy, natural body proportions, realistic hands and fingers");
        }
        let strength = (0.30 * c.geometry.max(c.anatomy)).clamp(0.12, 0.5);
        let mut g = crate::api::Img2img::new(model, &current)
            .prompt(parts.join(", "))
            .negative("deformed, distorted, incoherent, warped, extra limbs, extra fingers, bad hands, disconnected, malformed")
            .strength(strength)
            .steps(steps)
            .seed(0)
            .count(1);
        if let Some(d) = device {
            g = g.device(d);
        }
        let imgs = g.run().await.context("corrective img2img (geometry/anatomy)")?;
        let refined = tmp.join("refine_geom.png");
        imgs.first().context("img2img produced no image")?.save(&refined)?;
        current = refined;
    }

    // 2. no-twins → detect faces, inpaint each face beyond the first with a distinct seed so lookalikes
    //    diverge. Best-effort — no detector → skip.
    if c.no_twins > 0.0 {
        if let Some(varied) = vary_lookalike_faces(&current, c.no_twins, model, device, steps, tmp).await? {
            current = varied;
        }
    }

    if current != out {
        std::fs::copy(&current, out).with_context(|| format!("writing refined {}", out.display()))?;
    }
    Ok(())
}

/// Detect faces; for each face after the first, inpaint its box (feathered) with a distinct seed +
/// "distinct face" prompt so duplicate/lookalike faces diverge. Returns the varied image path, or `None`
/// when there are <2 faces / no detector.
async fn vary_lookalike_faces(input: &Path, weight: f32, model: &str, device: Option<&str>, steps: usize, tmp: &Path) -> Result<Option<PathBuf>> {
    let Some(scrfd) = crate::pipelines::scrfd::resolve_scrfd_weights().await.ok().flatten() else {
        tracing::warn!(target: "plakat", "naturalize: --no-twins needs a face detector (SCRFD) — skipping");
        return Ok(None);
    };
    let dev = match crate::api::device(device.unwrap_or("auto")) {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let Ok(det) = crate::pipelines::scrfd::SCRFDDetector::load(&scrfd, crate::pipelines::scrfd::SCRFDConfig::default(), &dev, candle_core::DType::F32) else {
        return Ok(None);
    };
    let Ok(mut faces) = det.detect(input) else { return Ok(None) };
    faces.retain(|f| f.score >= 0.4);
    if faces.len() < 2 {
        return Ok(None); // nothing to un-twin
    }
    // keep the largest face; vary every other one.
    faces.sort_by(|a, b| {
        let area = |f: &crate::pipelines::scrfd::Face| (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
        area(b).partial_cmp(&area(a)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let (iw, ih) = image::image_dimensions(input)?;
    let mut current = input.to_path_buf();
    let strength = (0.5 * weight).clamp(0.25, 0.7);
    for (i, f) in faces.iter().enumerate().skip(1) {
        // a feathered white box over this face → the inpaint region.
        let mut mask = GrayImage::from_pixel(iw, ih, Luma([0]));
        let (x0, y0) = (f.bbox[0].max(0.0) as u32, f.bbox[1].max(0.0) as u32);
        let (x1, y1) = ((f.bbox[2] as u32).min(iw), (f.bbox[3] as u32).min(ih));
        for y in y0..y1 {
            for x in x0..x1 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let mask_png = tmp.join(format!("twin_mask_{i}.png"));
        mask.save(&mask_png)?;
        let mut g = crate::api::Img2img::new(model, &current)
            .prompt("a distinct unique person, different facial features, clearly a different individual")
            .negative("identical, same face, twins, duplicate, clone")
            .strength(strength)
            .steps(steps)
            .seed((i as u64).wrapping_mul(0x9e3779b9))
            .count(1)
            .mask(&mask_png)
            .mask_feather(((f.bbox[2] - f.bbox[0]) * 0.15) as u32);
        if let Some(d) = device {
            g = g.device(d);
        }
        match g.run().await {
            Ok(imgs) => {
                if let Some(im) = imgs.first() {
                    let vp = tmp.join(format!("twin_{i}.png"));
                    im.save(&vp)?;
                    current = vp;
                }
            }
            Err(e) => tracing::warn!(target: "plakat", "naturalize: un-twin face {i} failed ({e})"),
        }
    }
    Ok(Some(current))
}
