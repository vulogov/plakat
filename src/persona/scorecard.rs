//! The persona scorecard (RFC §12) — measure a render against the spec that produced it. Built before
//! more generation (the RFC's central sequencing argument): unevenness can't be fixed before it can be
//! measured.
//!
//! P1 keystone: the **landmark** probe — SCRFD detects the face → PIPNet-98 aligns it → scale-invariant
//! WFLW-98 ratio metrics (interpupillary/face-width, mouth-width/face-width, face aspect). These are the
//! measurements every geometric attribute is scored against. Colour/detect/identity probes reuse the
//! existing CLIP / OWL-ViT / ArcFace / LAION stacks and land next.
//!
//! Scoring a spec **scalar** (model-relative 0.5 prior) against an absolute metric needs the P4
//! calibration table (the per-family prior + response curve); until that exists, `verify` reports the
//! raw measured metrics + a directional read, which is already enough to see whether an attribute moved.

use anyhow::{Context, Result};
use candle_core::Device;

use crate::persona::aligner::{PipNet, NUM_LANDMARKS};
// WFLW-98 topology (frozen v1) lives in one place — the geometry engine's topology module. The
// scorecard references the same indices + named-anchor vocabulary rather than duplicating them.
use crate::persona::geometry::topology::{
    self as topo, CONTOUR, MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, PUPIL_LEFT, PUPIL_RIGHT,
};

/// Geometric measurements from one aligned face. All ratios are **scale-invariant** in the crop frame.
#[derive(Debug, Clone)]
pub struct FaceMetrics {
    /// Inter-pupillary distance / face width — the metric for `eyes.spacing`.
    pub interpupillary_over_facewidth: f32,
    /// Mouth width / face width — for `mouth.width`.
    pub mouth_over_facewidth: f32,
    /// Face height / face width — for `face.width` (inverse relationship).
    pub face_aspect: f32,
    /// The 98 landmarks in crop-normalised `[0,1]`.
    pub landmarks: Vec<(f32, f32)>,
    /// Face contour width / height in crop-normalised units (for anchor offset scaling).
    pub face_w: f32,
    pub face_h: f32,
    /// SCRFD detection score of the measured face.
    pub detection_score: f32,
    /// The aligned face crop (RGB) — the pixel source for `local_anomaly`.
    pub crop: image::RgbImage,
    /// Top-left of the crop in the ORIGINAL image's pixels — lets the detail compositor map a
    /// crop-normalised landmark back to full-image coordinates: `full = crop_origin + lm * crop_dim`.
    pub crop_origin: (u32, u32),
}

/// Result of a `local_anomaly` probe (RFC §12.1) — the probe that makes marks measurable.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Presence confidence in `[0,1]` (a darker/redder inner region vs the surrounding skin ring).
    pub presence: f32,
    /// Position error: distance from the requested anchor to the anomaly centroid, in crop-normalised
    /// units (fraction of crop width). Small = the mark landed where the spec asked.
    pub position_error: f32,
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Detect the largest face (SCRFD), align it (PIPNet-98), and compute the ratio metrics. Returns
/// `None` if no face is detected.
pub fn measure_landmarks(
    image_path: &std::path::Path,
    detector: &crate::pipelines::scrfd::SCRFDDetector,
    pipnet: &PipNet,
) -> Result<Option<FaceMetrics>> {
    let faces = detector.detect(image_path).context("SCRFD detect")?;
    let Some(face) = faces.into_iter().max_by(|a, b| {
        let area = |f: &crate::pipelines::scrfd::Face| (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
        area(a).partial_cmp(&area(b)).unwrap_or(std::cmp::Ordering::Equal)
    }) else {
        return Ok(None);
    };

    // Crop the face box with a 25% margin (PIPNet expects a loose crop), clamped to the image.
    let img = image::open(image_path)?.to_rgb8();
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let [x1, y1, x2, y2] = face.bbox;
    let (bw, bh) = (x2 - x1, y2 - y1);
    let m = 0.25;
    let cx1 = (x1 - bw * m).max(0.0);
    let cy1 = (y1 - bh * m).max(0.0);
    let cx2 = (x2 + bw * m).min(iw);
    let cy2 = (y2 + bh * m).min(ih);
    let crop = image::imageops::crop_imm(&img, cx1 as u32, cy1 as u32, (cx2 - cx1) as u32, (cy2 - cy1) as u32)
        .to_image();

    let x = pipnet.preprocess(&crop)?;
    let heads = pipnet.forward(&x)?;
    let lm = PipNet::decode(&heads)?; // 98 pts in [0,1] of the crop
    debug_assert_eq!(lm.len(), NUM_LANDMARKS);

    let face_min_x = CONTOUR.clone().map(|i| lm[i].0).fold(f32::INFINITY, f32::min);
    let face_max_x = CONTOUR.clone().map(|i| lm[i].0).fold(f32::NEG_INFINITY, f32::max);
    let face_min_y = CONTOUR.clone().map(|i| lm[i].1).fold(f32::INFINITY, f32::min);
    let face_max_y = CONTOUR.map(|i| lm[i].1).fold(f32::NEG_INFINITY, f32::max);
    let face_w = (face_max_x - face_min_x).max(1e-4);
    let face_h = (face_max_y - face_min_y).max(1e-4);

    Ok(Some(FaceMetrics {
        interpupillary_over_facewidth: dist(lm[PUPIL_RIGHT], lm[PUPIL_LEFT]) / face_w,
        mouth_over_facewidth: dist(lm[MOUTH_CORNER_RIGHT], lm[MOUTH_CORNER_LEFT]) / face_w,
        face_aspect: face_h / face_w,
        face_w,
        face_h,
        detection_score: face.score,
        landmarks: lm,
        crop,
        crop_origin: (cx1 as u32, cy1 as u32),
    }))
}

/// A named anchor region (§8.2/§10.1) → the WFLW-98 landmark indices whose centroid is its base point.
/// This is the start of the **frozen WFLW-98 anchor vocabulary** (topology v1). Face regions only —
/// ear/nose-piercing sites have no WFLW landmark and are handled by the body skeleton later.
/// Resolve an anchor (named region + face-normalised offset, §8.2) to a crop-normalised `[0,1]` point.
/// `offset` x is positive to the **subject's left** (= +x in image); scaled by face width/height.
pub fn resolve_anchor(
    anchor: &crate::persona::spec::Anchor,
    m: &FaceMetrics,
) -> Option<(f32, f32)> {
    // Prefer an explicit landmark region; fall back to `region` shorthand (same vocabulary).
    let name = anchor.landmark.as_deref().or(anchor.region.as_deref())?;
    let idxs = topo::named_region(name)?;
    let n = idxs.len() as f32;
    let (bx, by) = idxs.iter().fold((0.0, 0.0), |(ax, ay), &i| {
        (ax + m.landmarks[i].0 / n, ay + m.landmarks[i].1 / n)
    });
    let [dx, dy] = anchor.offset.unwrap_or([0.0, 0.0]);
    Some(((bx + dx * m.face_w).clamp(0.0, 1.0), (by + dy * m.face_h).clamp(0.0, 1.0)))
}

fn luma(p: &image::Rgb<u8>) -> f32 {
    0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32
}

// --- region_color probe (RFC §12.1): landmark-masked robust CIELAB + ΔE to target. ---

/// sRGB (0–255) → CIELAB (D65). Standard: gamma-expand → XYZ → Lab.
pub fn srgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
    let lin = |c: f32| {
        let c = c / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (r, g, b) = (lin(rgb[0] as f32), lin(rgb[1] as f32), lin(rgb[2] as f32));
    // linear sRGB → XYZ (D65), then normalise by the white point.
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIE76 colour difference (Euclidean in Lab) — sufficient for the coarse targets here.
pub fn delta_e(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Median Lab over a disc — robust to lashes / highlights / stray pixels.
fn disc_median_lab(crop: &image::RgbImage, cx: f32, cy: f32, r: f32) -> Option<[f32; 3]> {
    let (cw, ch) = (crop.width() as f32, crop.height() as f32);
    let mut labs: Vec<[f32; 3]> = Vec::new();
    let (x0, y0) = ((cx - r).max(0.0) as u32, (cy - r).max(0.0) as u32);
    let (x1, y1) = ((cx + r).min(cw - 1.0) as u32, (cy + r).min(ch - 1.0) as u32);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() <= r {
                labs.push(srgb_to_lab(crop.get_pixel(x, y).0));
            }
        }
    }
    if labs.len() < 3 {
        return None;
    }
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let mut ch: Vec<f32> = labs.iter().map(|l| l[c]).collect();
        ch.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out[c] = ch[ch.len() / 2];
    }
    Some(out)
}

/// Colour readings from a face: median Lab of the iris + a clean cheek-skin patch.
#[derive(Debug, Clone)]
pub struct ColorReadings {
    pub iris: Option<[f32; 3]>,
    pub skin: Option<[f32; 3]>,
}

/// Measure the iris (both pupils) and skin (both cheeks) median Lab.
pub fn measure_colors(m: &FaceMetrics) -> ColorReadings {
    let (cw, ch) = (m.crop.width() as f32, m.crop.height() as f32);
    let px = |i: usize| (m.landmarks[i].0 * cw, m.landmarks[i].1 * ch);
    // Iris: a small disc at each pupil, radius ~ a fraction of the inter-pupil distance.
    let (rp, lp) = (px(PUPIL_RIGHT), px(PUPIL_LEFT));
    let ipd = ((rp.0 - lp.0).powi(2) + (rp.1 - lp.1).powi(2)).sqrt();
    let ir = (ipd * 0.10).max(2.0);
    let iris = match (disc_median_lab(&m.crop, rp.0, rp.1, ir), disc_median_lab(&m.crop, lp.0, lp.1, ir)) {
        (Some(a), Some(b)) => Some([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    // Skin: cheek centroids (pupil + mouth-corner midpoint), a slightly larger patch.
    let cheek = |pupil: (f32, f32), corner: (f32, f32)| ((pupil.0 + corner.0) / 2.0, (pupil.1 + corner.1) / 2.0);
    let rc = cheek(rp, px(MOUTH_CORNER_RIGHT));
    let lc = cheek(lp, px(MOUTH_CORNER_LEFT));
    let sr = (m.face_w * cw * 0.06).max(3.0);
    let skin = match (disc_median_lab(&m.crop, rc.0, rc.1, sr), disc_median_lab(&m.crop, lc.0, lc.1, sr)) {
        (Some(a), Some(b)) => Some([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    ColorReadings { iris, skin }
}

// --- detect probe (RFC §12.1): OWL-ViT present/absent for salient objects (beard, glasses, braces). ---

/// Result of a `detect` probe: was the queried object found, and at what confidence.
#[derive(Debug, Clone)]
pub struct DetectResult {
    pub present: bool,
    pub score: f32,
}

/// Run OWL-ViT for `query` on `image` → present/absent + confidence.
pub fn detect_probe(
    owl: &crate::pipelines::owlvit::OwlViT,
    image: &std::path::Path,
    query: &str,
    threshold: f32,
) -> Result<DetectResult> {
    let det = owl.detect(image, query, threshold).context("OWL-ViT detect probe")?;
    Ok(DetectResult { present: det.is_some(), score: det.map(|d| d.score).unwrap_or(0.0) })
}

// --- scoring aggregate (RFC §12.2) ---

/// One scored attribute: measured vs target, with the weight it carries in the aggregate.
#[derive(Debug, Clone)]
pub struct AttrScore {
    pub path: String,
    pub pass: bool,
    /// `control × class` (× priority, default 1) — §9.2/§12.2.
    pub weight: f32,
    pub note: String,
}

/// The detail sub-score (§12.2): marks are realised differently, so they score separately on presence
/// (was it produced) + position (how far from its anchor).
#[derive(Debug, Clone)]
pub struct DetailSubscore {
    pub presence_mean: f32,
    pub position_mean: f32,
    pub n: usize,
}

/// Score a geometric scalar against its family calibration prior (§13.1): the realised metric is
/// normalised to `[0,1]` (median→0.5), and the attribute passes when the realised scalar lands within
/// `SCALAR_TOL` of the requested one, weighted by the per-family controllability grade (§13.3). `invert`
/// is set for `face.width` (a wider face has a *smaller* height/width aspect). Returns
/// `(realised_scalar, score)` — `None` when the family has no prior for this attribute.
pub fn scalar_score(
    path: &str,
    requested: f32,
    metric: f32,
    table: &crate::persona::calibration::CalibrationTable,
    invert: bool,
) -> Option<(f32, AttrScore)> {
    const SCALAR_TOL: f32 = 0.20;
    let prior = table.priors.get(path)?;
    let realised = {
        let r = prior.normalise(metric);
        if invert { 1.0 - r } else { r }
    };
    let pass = (realised - requested).abs() < SCALAR_TOL;
    let grade = table.grade(path);
    let weight = grade.map(|g| g.weight()).unwrap_or(0.7);
    let note = format!("req {requested:.2} → realised {realised:.2} [{}]", grade.map(|g| g.as_str()).unwrap_or("uncalibrated"));
    Some((realised, AttrScore { path: path.into(), pass, weight, note }))
}

/// The spec's `eyes.color` as a CIELAB target, if set (named → lab table, or an explicit lab triple).
pub fn eyes_color_target(spec: &crate::persona::spec::PersonaSpec) -> Option<[f32; 3]> {
    use crate::persona::spec::Color;
    match spec.eyes.as_ref().and_then(|e| e.color.as_ref())? {
        Color::Named(n) => color_name_to_lab(n),
        Color::Lab { lab } => Some(*lab),
    }
}

/// A compact per-render scorecard for ranking casting candidates (§11.1): the calibrated geometric
/// scalars + the `eyes.color` ΔE. No OWL-ViT (too slow per candidate); the detect probes stay in
/// `verify`. Returns a `Scorecard` whose `aggregate()` is the spec-conformance sort key.
pub fn score_render(
    spec: &crate::persona::spec::PersonaSpec,
    m: &FaceMetrics,
    table: Option<&crate::persona::calibration::CalibrationTable>,
) -> Scorecard {
    let mut sc = Scorecard::default();
    let scalars = [
        ("eyes.spacing", spec.eyes.as_ref().and_then(|e| e.spacing), m.interpupillary_over_facewidth, false),
        ("mouth.width", spec.mouth.as_ref().and_then(|mo| mo.width), m.mouth_over_facewidth, false),
        ("face.width", spec.face.as_ref().and_then(|f| f.width), m.face_aspect, true),
    ];
    for (path, req, metric, invert) in scalars {
        if let (Some(r), Some(t)) = (req, table) {
            if let Some((_, s)) = scalar_score(path, r, metric, t, invert) {
                sc.scored.push(s);
            }
        }
    }
    if let (Some(iris), Some(target)) = (measure_colors(m).iris, eyes_color_target(spec)) {
        let de = delta_e(iris, target);
        sc.scored.push(AttrScore { path: "eyes.color".into(), pass: de < 20.0, weight: 0.7, note: format!("ΔE {de:.1}") });
    }
    sc
}

/// The scorecard: the weighted pass-fraction over *scored* attributes, with the four exclusions and the
/// detail sub-score reported separately so a persona can't score 100% while expressing nothing.
#[derive(Debug, Clone, Default)]
pub struct Scorecard {
    pub scored: Vec<AttrScore>,
    pub detail: Option<DetailSubscore>,
    /// Measured but not yet scorable — scalar geometric attrs awaiting the P4 calibration prior.
    pub pending_calibration: Vec<String>,
    /// Set attributes with no probe wired yet.
    pub unmeasurable: Vec<String>,
    /// Set attributes not visible in this render (§8.6).
    pub non_manifesting: Vec<String>,
}

impl Scorecard {
    /// The aggregate = weighted pass fraction over the *scored* attributes. `None` if nothing scorable
    /// (which the caller must report honestly rather than as 0 or 1).
    pub fn aggregate(&self) -> Option<f32> {
        if self.scored.is_empty() {
            return None;
        }
        let wsum: f32 = self.scored.iter().map(|s| s.weight).sum::<f32>().max(1e-6);
        let pass: f32 = self.scored.iter().filter(|s| s.pass).map(|s| s.weight).sum();
        // A weighted pass fraction is a proportion in [0,1]; clamp guards against
        // negative-signed zero and any negative control weights leaking through.
        // `+ 0.0` normalises a negative-signed zero (0-of-N passing) to `+0.0`.
        Some((pass / wsum).clamp(0.0, 1.0) + 0.0)
    }
}

/// The OWL-ViT query phrase for a `facial_hair.style` value (§12 `detect` target).
pub fn facial_hair_query(style: &str) -> &'static str {
    match style {
        "none" => "a beard", // absence check — expect NOT present
        "stubble" => "stubble on a face",
        "moustache" => "a moustache",
        "goatee" => "a goatee",
        s if s.contains("beard") => "a beard",
        _ => "facial hair",
    }
}

/// Coarse target Lab for the common eye / hair colour names (§12 `region_color` target). Approximate
/// but enough to distinguish e.g. blue vs brown eyes; refined against real renders during calibration.
pub fn color_name_to_lab(name: &str) -> Option<[f32; 3]> {
    Some(match name {
        // eyes
        "brown" => [30.0, 8.0, 16.0],
        "hazel" => [42.0, 6.0, 22.0],
        "amber" => [50.0, 12.0, 34.0],
        "green" => [45.0, -16.0, 20.0],
        "blue" => [55.0, -4.0, -16.0],
        "grey" | "gray" => [55.0, 0.0, 0.0],
        // hair
        "black" => [15.0, 1.0, 2.0],
        "dark-brown" => [22.0, 6.0, 12.0],
        "auburn" => [30.0, 16.0, 20.0],
        "red" | "ginger" => [38.0, 26.0, 26.0],
        "blonde" | "blond" => [72.0, 5.0, 34.0],
        "white" => [92.0, 0.0, 0.0],
        _ => return None,
    })
}

/// The `local_anomaly` probe (RFC §12.1): go to where a mark *should* be and ask whether the skin there
/// deviates from its neighbourhood — a far easier question than searching for a 4-pixel mole, robust at
/// small scale, and it yields a position error. `radius_frac` ≈ the mark's size (fraction of crop width).
pub fn local_anomaly(crop: &image::RgbImage, pos: (f32, f32), radius_frac: f32) -> AnomalyResult {
    let (cw, ch) = (crop.width() as f32, crop.height() as f32);
    let (cx, cy) = (pos.0 * cw, pos.1 * ch);
    let r_in = (radius_frac * cw).max(2.0);
    let r_out = 2.5 * r_in;
    let (mut inner_sum, mut inner_n) = (0.0f32, 0.0f32);
    let mut ring: Vec<f32> = Vec::new();
    let x0 = (cx - r_out).max(0.0) as u32;
    let y0 = (cy - r_out).max(0.0) as u32;
    let x1 = (cx + r_out).min(cw - 1.0) as u32;
    let y1 = (cy + r_out).min(ch - 1.0) as u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            let l = luma(crop.get_pixel(x, y));
            if d <= r_in {
                inner_sum += l;
                inner_n += 1.0;
            } else if d <= r_out {
                ring.push(l);
            }
        }
    }
    if inner_n < 1.0 || ring.len() < 4 {
        return AnomalyResult { presence: 0.0, position_error: 1.0 };
    }
    let inner_mean = inner_sum / inner_n;
    let ring_mean = ring.iter().sum::<f32>() / ring.len() as f32;
    let ring_var = ring.iter().map(|v| (v - ring_mean).powi(2)).sum::<f32>() / ring.len() as f32;
    let ring_std = ring_var.sqrt().max(1.0);
    // A mark is darker than surrounding skin → inner_mean < ring_mean → positive z. z≥3 = confident.
    let z = (ring_mean - inner_mean) / ring_std;
    let presence = (z / 3.0).clamp(0.0, 1.0);
    // Centroid of the darkest pixels in the neighbourhood (weight = how much below the ring mean).
    let (mut wx, mut wy, mut wsum) = (0.0f32, 0.0f32, 0.0f32);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let w = (ring_mean - luma(crop.get_pixel(x, y))).max(0.0);
            wx += w * x as f32;
            wy += w * y as f32;
            wsum += w;
        }
    }
    let position_error = if wsum > 0.0 {
        (((wx / wsum - cx).powi(2) + (wy / wsum - cy).powi(2)).sqrt()) / cw
    } else {
        1.0
    };
    AnomalyResult { presence, position_error }
}

/// Load the aligner + detector (both weights auto-resolved) for a verify run.
pub async fn load_probes(device: &Device) -> Result<(crate::pipelines::scrfd::SCRFDDetector, PipNet)> {
    let scrfd_w = crate::pipelines::scrfd::resolve_scrfd_weights()
        .await?
        .context("the landmark probe needs SCRFD weights (none resolved)")?;
    let detector = crate::pipelines::scrfd::SCRFDDetector::load(
        &scrfd_w,
        crate::pipelines::scrfd::SCRFDConfig::default(),
        device,
        candle_core::DType::F32,
    )?;
    let pipnet = PipNet::load_pretrained(device).await?;
    Ok((detector, pipnet))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_and_topology_constants() {
        assert!((dist((0.0, 0.0), (3.0, 4.0)) - 5.0).abs() < 1e-6);
        // pupils are the last two WFLW-98 points; corners are within the outer-lip range.
        assert_eq!(PUPIL_LEFT, NUM_LANDMARKS - 1);
        assert_eq!(PUPIL_RIGHT, NUM_LANDMARKS - 2);
        assert!(MOUTH_CORNER_RIGHT < MOUTH_CORNER_LEFT);
    }

    /// Synthetic ground truth (§24): a dark spot composited at a known position must be found there by
    /// `local_anomaly`, and a blank region must not — the probe's self-test needing no annotation.
    #[test]
    fn local_anomaly_finds_a_dark_spot_and_ignores_blank_skin() {
        use image::{Rgb, RgbImage};
        let mut img = RgbImage::from_pixel(100, 100, Rgb([200, 200, 200]));
        let (sx, sy, r) = (60.0f32, 40.0f32, 3.5f32); // spot at (0.6, 0.4)
        for y in 0..100u32 {
            for x in 0..100u32 {
                if ((x as f32 - sx).powi(2) + (y as f32 - sy).powi(2)).sqrt() <= r {
                    img.put_pixel(x, y, Rgb([40, 30, 30]));
                }
            }
        }
        let hit = local_anomaly(&img, (0.6, 0.4), 0.04);
        assert!(hit.presence > 0.5, "presence {}", hit.presence);
        assert!(hit.position_error < 0.05, "position_error {}", hit.position_error);

        let blank = local_anomaly(&img, (0.2, 0.2), 0.04);
        assert!(blank.presence < 0.2, "blank presence {}", blank.presence);
    }

    #[test]
    fn srgb_lab_and_delta_e() {
        let white = srgb_to_lab([255, 255, 255]);
        assert!((white[0] - 100.0).abs() < 1.0 && white[1].abs() < 1.0 && white[2].abs() < 1.0, "{white:?}");
        let black = srgb_to_lab([0, 0, 0]);
        assert!(black[0].abs() < 1.0, "{black:?}");
        // a mid grey has ~0 chroma; red is far from grey in Lab.
        let grey = srgb_to_lab([128, 128, 128]);
        assert!(grey[1].abs() < 2.0 && grey[2].abs() < 2.0);
        assert!(delta_e(srgb_to_lab([200, 20, 20]), grey) > 40.0);
        assert!((delta_e(white, white)).abs() < 1e-4);
    }

    #[test]
    fn eye_colour_targets_are_distinguishable() {
        // blue vs brown eyes must be far apart in Lab (the probe's whole point).
        assert!(delta_e(color_name_to_lab("blue").unwrap(), color_name_to_lab("brown").unwrap()) > 25.0);
        assert!(color_name_to_lab("chartreuse").is_none());
    }

    #[test]
    fn facial_hair_query_mapping() {
        assert_eq!(facial_hair_query("none"), "a beard"); // absence check
        assert_eq!(facial_hair_query("full-beard"), "a beard");
        assert_eq!(facial_hair_query("moustache"), "a moustache");
        assert_eq!(facial_hair_query("sideburns"), "facial hair");
    }

    #[test]
    fn aggregate_is_the_weighted_pass_fraction() {
        let mut sc = Scorecard::default();
        assert_eq!(sc.aggregate(), None); // nothing scorable → honest None, not 0/1
        sc.scored.push(AttrScore { path: "eyes.color".into(), pass: true, weight: 0.7, note: String::new() });
        sc.scored.push(AttrScore { path: "facial_hair.style".into(), pass: false, weight: 0.7, note: String::new() });
        // one pass, one fail, equal weight → 0.5.
        assert!((sc.aggregate().unwrap() - 0.5).abs() < 1e-6);
        // heavier passing attribute pulls it up.
        sc.scored[0].weight = 3.0;
        assert!((sc.aggregate().unwrap() - 3.0 / 3.7).abs() < 1e-6);
    }
}
