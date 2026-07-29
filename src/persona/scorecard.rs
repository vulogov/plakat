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

// --- WFLW-98 landmark topology (frozen v1) — the indices the metrics reference. ---
/// Face contour (jaw line), 33 points.
const CONTOUR: std::ops::Range<usize> = 0..33;
/// Explicit pupil centres (WFLW-98's convenience points).
const PUPIL_RIGHT: usize = 96;
const PUPIL_LEFT: usize = 97;
/// Outer mouth corners (of the 76..=87 outer-lip loop).
const MOUTH_CORNER_RIGHT: usize = 76;
const MOUTH_CORNER_LEFT: usize = 82;

/// Geometric measurements from one aligned face. All are **scale-invariant ratios** in the crop frame,
/// so they need no mapping back to image pixels.
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
    /// SCRFD detection score of the measured face.
    pub detection_score: f32,
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
        landmarks: lm,
        detection_score: face.score,
    }))
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
}
