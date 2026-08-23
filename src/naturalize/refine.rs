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
        // Lead with realism/coherence, keep it grounded so organic textures (foliage, sky) re-resolve as
        // themselves rather than faceting into shards.
        parts.push("photorealistic, natural coherent detail, believable depth and perspective, sharp well-formed structure");
        if c.geometry > 0.0 {
            parts.push("coherent geometry, correct structure, straight true edges, clean continuous lines, well-formed buildings and objects");
        }
        if c.anatomy > 0.0 {
            parts.push("correct anatomy, natural body proportions, realistic hands and fingers");
        }
        // A touch gentler than before (0.26) so structure is fixed without repainting the whole scene.
        let strength = (0.26 * c.geometry.max(c.anatomy)).clamp(0.12, 0.45);
        let mut g = crate::api::Img2img::new(model, &current)
            .prompt(parts.join(", "))
            .negative("deformed, distorted, incoherent, warped, faceted, fragmented, shattered, glassy shards, kaleidoscope, crystalline artifacts, extra limbs, extra fingers, bad hands, disconnected, malformed")
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

/// **Auto medium-detection** (RFC QUALITY-4 P2) — CLIP zero-shot: embed the image and a bank of medium
/// probes into the shared CLIP space and pick the closest, returning the matching **style anchor** string
/// for the model corrections (so `--repair`/`--geometry` hold the source medium without a manual
/// `--style`). Best-effort — `None` if CLIP can't load. Reuses the openai CLIP the aesthetic scorer caches.
pub async fn detect_medium(input: &Path, device: Option<&str>) -> Option<String> {
    let dev = crate::api::device(device.unwrap_or("auto")).ok()?;
    let clip = crate::pipelines::clip_embed::ClipEmbedder::load(&dev).await.ok()?;
    let img = clip.embed_image(input).ok()?;
    // (CLIP probe prompt, the style-anchor string handed to the corrective img2img).
    let bank: &[(&str, &str)] = &[
        ("a watercolor painting", "soft wet-on-wet watercolor illustration, natural pigment granulation, paper texture"),
        ("an oil painting", "oil painting, visible brush strokes, impasto texture, canvas"),
        ("an ink drawing", "ink drawing, pen and ink linework, cross-hatching"),
        ("a gouache painting", "gouache painting, matte opaque pigment, flat washes"),
        ("a graphite pencil sketch", "graphite pencil sketch, soft shading, paper tooth"),
        ("a soft pastel drawing", "soft pastel drawing, chalky pigment, blended tones"),
        ("an acrylic painting", "acrylic painting, bold brushwork"),
        ("a comic book illustration", "comic book illustration, clean ink lines, cel shading"),
        ("a digital painting", "digital painting, painterly rendering, coherent detail"),
        ("a 3d render", "detailed 3d render, coherent detail"),
        ("a photograph", "photograph, natural realistic detail, believable depth"),
    ];
    let mut best: (f32, &str) = (f32::MIN, "");
    for (probe, anchor) in bank {
        if let Ok(t) = clip.embed_text(probe) {
            let s = crate::pipelines::clip_embed::cosine(&img, &t);
            if s > best.0 {
                best = (s, anchor);
            }
        }
    }
    (!best.1.is_empty()).then(|| best.1.to_string())
}

/// How much of the frame a face-protected repair may touch (RFC QUALITY-4 P1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RepairScope {
    /// Repair only the **figures** (body boxes projected from the faces), faces AND background preserved.
    /// The default — fixes the "background changed / colours drifted" regression.
    #[default]
    Figures,
    /// Repair everything **except faces** (the 6.12 behaviour) — the background regenerates too.
    NonFace,
    /// Repair the **whole image** including faces (no protection) — effectively `--geometry`.
    Full,
}

impl RepairScope {
    pub fn parse(s: &str) -> Option<RepairScope> {
        match s.trim().to_ascii_lowercase().as_str() {
            "figures" | "figure" => Some(RepairScope::Figures),
            "non-face" | "nonface" | "non_face" => Some(RepairScope::NonFace),
            "full" | "all" => Some(RepairScope::Full),
            _ => None,
        }
    }
}

/// **Face-protected local repair** (RFC QUALITY-3/4) — the honest way to touch anatomy on cohesive art.
/// Whole-image img2img makes faces uncanny and strips the artistic character; this detects faces (SCRFD)
/// and **protects them** (masked out → the soft artistic faces survive exactly), then runs a gentle
/// **style-matched** img2img to attempt broken hands/feet/limbs. `scope` bounds where it may paint:
/// `Figures` (default) touches only the figures' bodies (projected from the faces) so the **background is
/// preserved** too; `NonFace` regenerates all non-face pixels; `Full` protects nothing. `strength` scales
/// the denoise; `style` names the medium to hold so it re-resolves IN-STYLE, never toward photoreal.
/// Returns `false` (skip) when there's no face detector or no faces.
pub async fn repair_protected(input: &Path, out: &Path, strength: f32, style: Option<&str>, scope: RepairScope, model: &str, device: Option<&str>, steps: usize, tmp: &Path) -> Result<bool> {
    let Some(scrfd) = crate::pipelines::scrfd::resolve_scrfd_weights().await.ok().flatten() else {
        tracing::warn!(target: "plakat", "naturalize --repair needs a face detector (SCRFD) — skipping");
        return Ok(false);
    };
    let dev = match crate::api::device(device.unwrap_or("auto")) {
        Ok(d) => d,
        Err(_) => return Ok(false),
    };
    let Ok(det) = crate::pipelines::scrfd::SCRFDDetector::load(&scrfd, crate::pipelines::scrfd::SCRFDConfig::default(), &dev, candle_core::DType::F32) else {
        return Ok(false);
    };
    let Ok(mut faces) = det.detect(input) else { return Ok(false) };
    faces.retain(|f| f.score >= 0.35);

    // QUALITY-5 P2: in Figures scope, also OWL-ViT-detect PEOPLE so figures whose face isn't found (back
    // turned / distant / occluded) are still covered. Best-effort — a missing detector just falls back to
    // the face-projected boxes.
    let persons: Vec<crate::pipelines::owlvit::Detection> = if scope == RepairScope::Figures {
        match crate::pipelines::owlvit::OwlViT::load_pretrained(&dev).await {
            Ok(owl) => owl.detect_all(input, "a person", 0.20, 12).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };
    if faces.is_empty() && persons.is_empty() {
        tracing::warn!(target: "plakat", "naturalize --repair: no faces or people detected — skipping (use --geometry/--anatomy for non-figure art)");
        return Ok(false);
    }
    let (iw, ih) = image::image_dimensions(input)?;
    // Inpaint mask — white = regenerate, black = PRESERVE.
    let paint = |m: &mut GrayImage, x0: f32, y0: f32, x1: f32, y1: f32, v: u8| {
        let (x0, y0) = (x0.max(0.0) as u32, y0.max(0.0) as u32);
        let (x1, y1) = ((x1 as u32).min(iw), (y1 as u32).min(ih));
        for y in y0..y1 {
            for x in x0..x1 {
                m.put_pixel(x, y, Luma([v]));
            }
        }
    };
    let mut max_face = 0.0f32;
    let mut mask = match scope {
        // Start all-BLACK (preserve everything), then paint each projected figure body WHITE.
        RepairScope::Figures => {
            let mut m = GrayImage::from_pixel(iw, ih, Luma([0]));
            for f in &faces {
                let (fw, fh) = (f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]);
                max_face = max_face.max(fw);
                let cx = (f.bbox[0] + f.bbox[2]) * 0.5;
                // a running child ≈ 5–6 head-heights tall, ≈ 2.2 head-widths wide (arms out).
                paint(&mut m, cx - 1.1 * fw, f.bbox[1] - 0.2 * fh, cx + 1.1 * fw, f.bbox[3] + 5.0 * fh, 255);
            }
            // union the OWL-ViT person boxes (covers figures with no detected face), slightly grown.
            for pd in &persons {
                let (pw, ph) = (pd.x1 - pd.x0, pd.y1 - pd.y0);
                paint(&mut m, pd.x0 - 0.08 * pw, pd.y0 - 0.06 * ph, pd.x1 + 0.08 * pw, pd.y1 + 0.06 * ph, 255);
            }
            m
        }
        // Everything regenerates (the 6.12 behaviour) — start all-WHITE.
        RepairScope::NonFace | RepairScope::Full => GrayImage::from_pixel(iw, ih, Luma([255])),
    };
    // Protect the faces (black) unless scope is Full. Grown to cover hairline/jaw/neck (a border cutting
    // the chin reads as uncanny).
    if scope != RepairScope::Full {
        for f in &faces {
            let (fw, fh) = (f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]);
            max_face = max_face.max(fw);
            paint(&mut mask, f.bbox[0] - fw * 0.35, f.bbox[1] - fh * 0.45, f.bbox[2] + fw * 0.35, f.bbox[3] + fh * 0.45, 0);
        }
    }
    let mask_png = tmp.join("repair_protect_mask.png");
    mask.save(&mask_png)?;

    let anatomy = "correct anatomy, natural body proportions, well-formed hands, correct feet and ankles, natural running pose, coherent structure";
    let prompt = match style {
        Some(s) if !s.trim().is_empty() => format!("{s}, {anatomy}"),
        _ => anatomy.to_string(),
    };
    let negative = "deformed, extra limbs, extra legs, extra arms, duplicated limbs, fused fingers, extra fingers, malformed hands, malformed feet, missing ankles, floating limbs, photorealistic, 3d render, plastic, uncanny, creepy face";
    let feather = ((max_face * 0.25) as u32).max(8);
    let mut g = crate::api::Img2img::new(model, input)
        .prompt(prompt)
        .negative(negative)
        .strength(strength.clamp(0.12, 0.6))
        .steps(steps)
        .seed(0)
        .count(1)
        .mask(&mask_png)
        .mask_feather(feather);
    if let Some(d) = device {
        g = g.device(d);
    }
    let imgs = g.run().await.context("face-protected repair img2img")?;
    imgs.first().context("repair produced no image")?.save(out)?;
    Ok(true)
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
