//! Face-scan (Track A): detect faces across the library with SCRFD and — when ArcFace weights are
//! available (`PLAKAT_ARCFACE_WEIGHTS`) — embed each face and greedily group them into `person-N`
//! clusters. Every image with a face is tagged (`has-face`, `faces-N`, and one `person-K` per person
//! present) so the existing tag filter / smart albums surface people. Detection-only (counts) when
//! ArcFace isn't provisioned — SCRFD auto-downloads, ArcFace needs a user-supplied safetensors file.

use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::pipelines::face_models::{prepare_face_tensor, FaceAlignment, IResnet50};
use crate::pipelines::scrfd::{resolve_scrfd_weights, SCRFDConfig, SCRFDDetector};

/// Cosine similarity above which two face embeddings are treated as the same person.
const IDENTITY_THRESHOLD: f32 = 0.5;

/// Result of a library face scan: the per-image tags to merge, plus counts for the status summary.
pub struct ScanResult {
    /// `(image path, tags to add)` — merged into each image's curation record.
    pub tags: Vec<(PathBuf, Vec<String>)>,
    pub images_with_faces: usize,
    pub total_faces: usize,
    /// Distinct people found (0 when identity grouping was unavailable).
    pub people: usize,
    /// Whether ArcFace identity clustering ran (vs. detection-only).
    pub grouped: bool,
}

impl ScanResult {
    pub fn summary(&self) -> String {
        if self.total_faces == 0 {
            return "face-scan: no faces detected".into();
        }
        if self.grouped {
            format!(
                "✓ face-scan: {} faces in {} images → {} people (person-N tags) · filter by tag",
                self.total_faces, self.images_with_faces, self.people
            )
        } else {
            format!(
                "✓ face-scan: {} faces in {} images (has-face / faces-N tags) · set PLAKAT_ARCFACE_WEIGHTS to group people",
                self.total_faces, self.images_with_faces
            )
        }
    }
}

/// Detect faces in one image and return them as `(cx, cy, rx, ry)` ellipses in **per-mille** of the
/// image dimensions (the compact form the `FacePolish` edit stores, so the op stays `Copy` and its
/// replay needs no model). Each face's bbox becomes an ellipse a little wider/taller than the tight
/// box (to cover cheeks/forehead). Returns at most `max` faces, score-descending. SCRFD auto-downloads.
pub async fn detect_ellipses(device: &Device, path: &Path, max: usize) -> Result<Vec<[i32; 4]>> {
    let scrfd_path = resolve_scrfd_weights()
        .await?
        .ok_or_else(|| anyhow::anyhow!("SCRFD weights unavailable — set PLAKAT_SCRFD_HF or PLAKAT_SCRFD_WEIGHTS"))?;
    let detector = SCRFDDetector::load(&scrfd_path, SCRFDConfig::default(), device, DType::F32)?;
    let (w, h) = image::image_dimensions(path)?;
    let (wf, hf) = (w as f32, h as f32);
    let mut out = Vec::new();
    for f in detector.detect(path)?.iter().take(max) {
        let [x1, y1, x2, y2] = f.bbox;
        let cx = ((x1 + x2) * 0.5 / wf).clamp(0.0, 1.0);
        let cy = ((y1 + y2) * 0.5 / hf).clamp(0.0, 1.0);
        let rx = ((x2 - x1) * 0.60 / wf).clamp(0.0, 0.9);
        let ry = ((y2 - y1) * 0.72 / hf).clamp(0.0, 0.9);
        out.push([
            (cx * 1000.0) as i32,
            (cy * 1000.0) as i32,
            (rx * 1000.0) as i32,
            (ry * 1000.0) as i32,
        ]);
    }
    Ok(out)
}

/// Dot product of two L2-normalised vectors == cosine similarity.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A detected face: which image it came from, and its identity embedding (when ArcFace ran).
struct DetFace {
    img: usize,
    emb: Option<Vec<f32>>,
}

/// Scan `items` for faces. Detects with SCRFD; if ArcFace weights are present, also embeds + clusters.
/// `progress(done, total)` is called per image.
pub async fn scan(
    device: &Device,
    items: Vec<PathBuf>,
    progress: impl Fn(usize, usize),
) -> Result<ScanResult> {
    let scrfd_path = resolve_scrfd_weights()
        .await?
        .ok_or_else(|| anyhow::anyhow!("SCRFD weights unavailable — set PLAKAT_SCRFD_HF or PLAKAT_SCRFD_WEIGHTS"))?;
    let detector = SCRFDDetector::load(&scrfd_path, SCRFDConfig::default(), device, DType::F32)?;

    // Optional ArcFace backbone for identity grouping (best-effort — a missing/incompatible file just
    // falls back to detection-only rather than failing the whole scan).
    let arcface: Option<IResnet50> = std::env::var("PLAKAT_ARCFACE_WEIGHTS")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .and_then(|p| {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[p], DType::F32, device).ok()? };
            IResnet50::new(vb).ok()
        });
    let grouped = arcface.is_some();

    let total = items.len();
    let mut per_image_count: Vec<usize> = vec![0; total];
    let mut faces: Vec<DetFace> = Vec::new();
    let mut images_with_faces = 0usize;
    let mut total_faces = 0usize;
    // 6.26.0 P4: per-image FACE-region sharpness (variance of the Laplacian inside the detected
    // face boxes), reusing the boxes the scan already found — no extra model pass. Later flagged
    // relative to the library median so blurry-face frames get a `soft-face` tag.
    let mut face_sharp: Vec<Option<f32>> = vec![None; total];

    for (i, path) in items.iter().enumerate() {
        progress(i + 1, total);
        let dets = match detector.detect(path) {
            Ok(d) => d,
            Err(_) => continue, // unreadable / undecodable image — skip, not fatal
        };
        if dets.is_empty() {
            continue;
        }
        images_with_faces += 1;
        per_image_count[i] = dets.len();
        total_faces += dets.len();
        // Normalize the pixel bboxes by the original dimensions, then measure sharpness inside
        // them on a 256px thumbnail (best-effort — a failed load just leaves this image unscored).
        if let Ok((ow, oh)) = image::image_dimensions(path) {
            let (ow, oh) = (ow.max(1) as f32, oh.max(1) as f32);
            let norm: Vec<[f32; 4]> = dets
                .iter()
                .map(|f| [f.bbox[0] / ow, f.bbox[1] / oh, f.bbox[2] / ow, f.bbox[3] / oh])
                .collect();
            if let Ok(img) = crate::photos::loader::thumbnail(path, 256) {
                face_sharp[i] = crate::photos::quality::region_sharpness(&img, &norm);
            }
        }
        for f in &dets {
            let emb = arcface.as_ref().and_then(|net| {
                let align = FaceAlignment::from_options(Some(f.bbox), Some(f.landmarks));
                let t = prepare_face_tensor(path, align, device, DType::F32).ok()?;
                net.forward(&t).ok()?.flatten_all().ok()?.to_vec1::<f32>().ok()
            });
            faces.push(DetFace { img: i, emb });
        }
    }

    // Greedy nearest-exemplar clustering by cosine over the face embeddings (each cluster keeps its
    // first face as the exemplar — simple + order-stable, good enough for library grouping).
    let mut person_of: Vec<Option<usize>> = vec![None; faces.len()];
    let mut exemplars: Vec<Vec<f32>> = Vec::new();
    if grouped {
        for (fi, face) in faces.iter().enumerate() {
            let Some(emb) = &face.emb else { continue };
            let mut best: Option<usize> = None;
            let mut best_sim = IDENTITY_THRESHOLD;
            for (pi, ex) in exemplars.iter().enumerate() {
                let s = cosine(emb, ex);
                if s > best_sim {
                    best_sim = s;
                    best = Some(pi);
                }
            }
            match best {
                Some(pi) => person_of[fi] = Some(pi),
                None => {
                    person_of[fi] = Some(exemplars.len());
                    exemplars.push(emb.clone());
                }
            }
        }
    }

    // 6.26.0 P4: flag blurry-face frames relative to the library. A face-region sharpness below
    // 40% of the median (over scored face-images) earns a `soft-face` tag — adaptive to the
    // library's overall scale, so it works whether the shots are crisp studio or soft phone snaps.
    let mut scored: Vec<f32> = face_sharp.iter().filter_map(|s| *s).collect();
    let soft_floor: Option<f32> = if scored.len() >= 4 {
        scored.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(scored[scored.len() / 2] * 0.4)
    } else {
        None
    };

    // Build per-image tags: has-face + a count band + each person present (+ soft-face).
    let mut tags: Vec<(PathBuf, Vec<String>)> = Vec::new();
    for (i, path) in items.iter().enumerate() {
        let count = per_image_count[i];
        if count == 0 {
            continue;
        }
        let mut t = vec!["has-face".to_string(), format!("faces-{}", count.min(9))];
        if let (Some(floor), Some(s)) = (soft_floor, face_sharp[i]) {
            if s < floor {
                t.push("soft-face".to_string());
            }
        }
        if grouped {
            let mut persons: Vec<usize> = faces
                .iter()
                .enumerate()
                .filter(|(_, f)| f.img == i)
                .filter_map(|(fi, _)| person_of[fi])
                .collect();
            persons.sort_unstable();
            persons.dedup();
            for p in persons {
                t.push(format!("person-{}", p + 1));
            }
        }
        tags.push((path.clone(), t));
    }

    Ok(ScanResult {
        tags,
        images_with_faces,
        total_faces,
        people: exemplars.len(),
        grouped,
    })
}

/// Rename a person tag inside one record's tag list (6.26.0 people management): replace `from`
/// with `to` (case-insensitive match on `from`), de-duplicating if `to` is already present.
/// Returns `true` if the list changed. Merging two clusters is just renaming one onto the other.
pub fn rename_tag(tags: &mut Vec<String>, from: &str, to: &str) -> bool {
    let had_from = tags.iter().any(|t| t.eq_ignore_ascii_case(from));
    if !had_from {
        return false;
    }
    let before = tags.len();
    // Drop the `from` tag (and any pre-existing `to`, to avoid a duplicate), then add `to` once.
    tags.retain(|t| !t.eq_ignore_ascii_case(from) && !t.eq_ignore_ascii_case(to));
    tags.push(to.to_string());
    // Changed unless the list was exactly `[to]`-equivalent already (same length, from==to case).
    !(before == tags.len() && from.eq_ignore_ascii_case(to))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_tag_renames_merges_and_dedups() {
        // Simple rename.
        let mut t = vec!["has-face".to_string(), "person-3".to_string()];
        assert!(rename_tag(&mut t, "person-3", "alice"));
        assert_eq!(t, vec!["has-face".to_string(), "alice".to_string()]);
        // Merge person-4 → alice when alice is already present → no duplicate.
        let mut t = vec!["person-4".to_string(), "alice".to_string(), "beach".to_string()];
        assert!(rename_tag(&mut t, "person-4", "alice"));
        assert_eq!(t.iter().filter(|x| *x == "alice").count(), 1);
        assert!(t.contains(&"beach".to_string()));
        // A record without the `from` tag is untouched.
        let mut t = vec!["person-2".to_string()];
        assert!(!rename_tag(&mut t, "person-9", "bob"));
        assert_eq!(t, vec!["person-2".to_string()]);
    }

    #[test]
    fn cosine_of_unit_vectors() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn summary_reports_detection_only_vs_grouped() {
        let base = ScanResult { tags: vec![], images_with_faces: 3, total_faces: 5, people: 0, grouped: false };
        assert!(base.summary().contains("PLAKAT_ARCFACE_WEIGHTS"));
        let g = ScanResult { tags: vec![], images_with_faces: 3, total_faces: 5, people: 2, grouped: true };
        assert!(g.summary().contains("2 people"));
    }
}
