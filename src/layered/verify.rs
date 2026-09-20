//! LAYERED-1 S4 — **verify**. For each anchored/hinted layer, an open-vocabulary detector (OWL-ViT) is asked
//! whether the layer's subject is present WHERE the plan put it. A layer passes when a detection above the
//! score threshold has its centre inside the planned box; the best-overlapping detection's score + IoU are
//! reported for diagnostics. Lifted layers are skipped (too small to detect reliably — they're structure-only
//! via the S5 lift), as are layers with no query.
//!
//! The detector is injected (`detect(query) -> Vec<Det>` in output pixels), so the verdict logic gates
//! offline; the CLI wires OWL-ViT. The failing layer ids drive the S4 [repair](super::repair).

use anyhow::Result;

use crate::layered::lint::{layer_class, Class};
use crate::layered::plan::{self, Layer, LayerPlan};
use crate::pipelines::noise_space::LatentGeometry;

/// A detection in OUTPUT pixels.
#[derive(Clone, Copy, Debug)]
pub struct Det {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

/// The verify verdict for one layer.
#[derive(Clone, Debug)]
pub struct LayerVerdict {
    pub id: String,
    pub query: String,
    pub class: Class,
    /// A detection above threshold has its centre inside the planned box.
    pub found: bool,
    /// The best overlapping detection's score (0 if none).
    pub score: f32,
    /// The best overlapping detection's IoU with the planned box.
    pub iou: f32,
    /// The best overlapping detection's box in output pixels (for a heatmap / diagnostics).
    pub detected: Option<[f32; 4]>,
}

/// The whole-plan verdict.
#[derive(Clone, Debug)]
pub struct Report {
    pub layers: Vec<LayerVerdict>,
    /// Layers that were actually checked (anchored/hinted with a query).
    pub checked: usize,
    /// Ids of checked layers that were NOT found — the repair targets.
    pub failures: Vec<String>,
}

impl Report {
    pub fn pass(&self) -> bool {
        self.failures.is_empty()
    }
}

/// The query for a layer: the explicit `query`, else its prompt.
pub fn layer_query(l: &Layer) -> String {
    l.query.clone().or_else(|| l.prompt.clone()).unwrap_or_default().trim().to_string()
}

/// Medium/style words that mean the render is NON-photographic. OWL-ViT is photo-trained, so it scores real
/// subjects far lower on a painting/illustration than on a photo — verifying a painterly scene at the photo
/// threshold produces phantom misses (subjects that ARE there read as absent), churning `--adapt`/`--repair`.
const STYLIZED_MEDIA: &[&str] = &[
    "painting", "painted", "illustration", "illustrated", "drawing", "drawn", "sketch", "engraving", "etching",
    "woodcut", "linocut", "watercolor", "watercolour", "gouache", "oil", "acrylic", "ink", "anime", "manga",
    "cartoon", "comic", "concept art", "digital art", "poster", "fresco", "pastel", "charcoal", "cel-shad",
    "stylized", "stylised", "storybook", "art nouveau", "ukiyo",
];

/// The verify detection threshold to actually use, given the plan's `medium`: for a NON-photographic medium the
/// base threshold is halved (floored at 0.02), since OWL-ViT scores stylized subjects low. Photographic (or
/// unknown) media keep the base unchanged.
pub fn style_threshold(base: f32, medium: Option<&str>) -> f32 {
    let stylized = medium.map(|m| m.to_ascii_lowercase()).is_some_and(|m| STYLIZED_MEDIA.iter().any(|w| m.contains(w)));
    if stylized {
        (base * 0.5).max(0.02)
    } else {
        base
    }
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ix0, iy0) = (a[0].max(b[0]), a[1].max(b[1]));
    let (ix1, iy1) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let ua = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0) + (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0) - inter;
    if ua <= 0.0 {
        0.0
    } else {
        inter / ua
    }
}

fn center_inside(det: [f32; 4], planned: [f32; 4]) -> bool {
    let (cx, cy) = ((det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0);
    cx >= planned[0] && cx <= planned[2] && cy >= planned[1] && cy <= planned[3]
}

/// Coverage = intersection / the SMALLER of the two boxes. High (→1) when one box largely contains the other,
/// regardless of size — so a big subject that spans past a small planned box (its centre landing outside)
/// still reads as present, and a subject filling most of a large box does too. Complements `center_inside`,
/// which alone false-fails a subject drawn larger than its box.
fn coverage(det: [f32; 4], planned: [f32; 4]) -> f32 {
    let (ix0, iy0) = (det[0].max(planned[0]), det[1].max(planned[1]));
    let (ix1, iy1) = (det[2].min(planned[2]), det[3].min(planned[3]));
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let da = (det[2] - det[0]).max(0.0) * (det[3] - det[1]).max(0.0);
    let pa = (planned[2] - planned[0]).max(0.0) * (planned[3] - planned[1]).max(0.0);
    let smaller = da.min(pa);
    if smaller <= 0.0 {
        0.0
    } else {
        inter / smaller
    }
}

/// A detection counts as landing on the planned box when its centre is inside OR it covers (is covered by) the
/// box past this fraction — the relaxed criterion that tolerates a subject rendered larger than its box.
const MIN_COVERAGE: f32 = 0.5;

fn matches_box(det: [f32; 4], planned: [f32; 4]) -> bool {
    center_inside(det, planned) || coverage(det, planned) >= MIN_COVERAGE
}

/// Verify a plan against a finished image at `w×h`. Anchored + hinted layers with a query are checked; each
/// passes when a detection (score ≥ the detector's own threshold) has its centre inside the planned box.
pub fn verify(plan: &LayerPlan, geom: &LatentGeometry, w: u32, h: u32, detect: &dyn Fn(&str) -> Result<Vec<Det>>) -> Result<Report> {
    let mut layers = Vec::new();
    let mut checked = 0usize;
    let mut failures = Vec::new();

    for l in &plan.layers {
        let class = layer_class(l, geom, w, h);
        let query = layer_query(l);
        // Lifted layers are structure-only (S5); empty queries can't be checked.
        if class == Class::Lifted || query.is_empty() {
            layers.push(LayerVerdict { id: l.id.clone(), query, class, found: false, score: 0.0, iou: 0.0, detected: None });
            continue;
        }
        checked += 1;

        let b = plan::layer_box(l);
        let planned = [b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32];
        let dets = detect(&query)?;

        // Prefer a detection that lands on the box (centre inside OR substantial coverage — so a subject drawn
        // larger than its box still counts); among those (or all, if none match) pick the best-IoU for the
        // reported score/box.
        let matched: Vec<&Det> = dets.iter().filter(|d| matches_box([d.x0, d.y0, d.x1, d.y1], planned)).collect();
        let found = !matched.is_empty();
        let pool: Vec<&Det> = if found { matched } else { dets.iter().collect() };
        let best = pool.into_iter().max_by(|a, b| {
            iou([a.x0, a.y0, a.x1, a.y1], planned).partial_cmp(&iou([b.x0, b.y0, b.x1, b.y1], planned)).unwrap_or(std::cmp::Ordering::Equal)
        });

        let (score, iou_v, detected) = match best {
            Some(d) => (d.score, iou([d.x0, d.y0, d.x1, d.y1], planned), Some([d.x0 / w as f32, d.y0 / h as f32, d.x1 / w as f32, d.y1 / h as f32])),
            None => (0.0, 0.0, None),
        };
        if !found {
            failures.push(l.id.clone());
        }
        layers.push(LayerVerdict { id: l.id.clone(), query, class, found, score, iou: iou_v, detected });
    }

    Ok(Report { layers, checked, failures })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> LatentGeometry {
        LatentGeometry { v: 8, u: 8, pool_levels: 2 }
    }

    fn layer(id: &str, bbox: [f32; 4], prompt: &str) -> Layer {
        Layer { id: id.into(), bbox: Some(bbox), prompt: Some(prompt.into()), ..Default::default() }
    }

    #[test]
    fn found_when_detection_centre_is_in_the_box() {
        // 128×128; box [0.25,0.25,0.75,0.75] → px [32,32,96,96], centre (64,64).
        let plan = LayerPlan { layers: vec![layer("bowl", [0.25, 0.25, 0.75, 0.75], "a bowl of pears")], ..Default::default() };
        // A detection centred at (64,64), above threshold.
        let detect = |_q: &str| Ok(vec![Det { x0: 40.0, y0: 40.0, x1: 88.0, y1: 88.0, score: 0.4 }]);
        let r = verify(&plan, &geom(), 128, 128, &detect).unwrap();
        assert!(r.pass(), "subject in the box → pass");
        assert_eq!(r.checked, 1);
        assert!(r.layers[0].found && r.layers[0].iou > 0.3);
    }

    #[test]
    fn fails_when_detection_is_elsewhere() {
        let plan = LayerPlan { layers: vec![layer("bowl", [0.25, 0.25, 0.75, 0.75], "a bowl of pears")], ..Default::default() };
        // Detection in the top-left corner, centre outside the box.
        let detect = |_q: &str| Ok(vec![Det { x0: 0.0, y0: 0.0, x1: 20.0, y1: 20.0, score: 0.4 }]);
        let r = verify(&plan, &geom(), 128, 128, &detect).unwrap();
        assert!(!r.pass(), "subject outside the box → fail");
        assert_eq!(r.failures, vec!["bowl".to_string()]);
        assert!(!r.layers[0].found);
    }

    #[test]
    fn found_when_subject_spans_past_a_small_box() {
        // A small planned box [0.05,0.05,0.20,0.20] → px [6,6,25,25], centre (15,15). A subject drawn LARGER
        // than its box — det [0,0,80,80], centre (40,40) OUTSIDE the box — used to false-fail on centre-in-box.
        // It fully covers the small box, so coverage → pass.
        let plan = LayerPlan { layers: vec![layer("lantern", [0.05, 0.05, 0.20, 0.20], "a paper lantern")], ..Default::default() };
        let detect = |_q: &str| Ok(vec![Det { x0: 0.0, y0: 0.0, x1: 80.0, y1: 80.0, score: 0.3 }]);
        let r = verify(&plan, &geom(), 128, 128, &detect).unwrap();
        assert!(r.pass(), "subject larger than its box still counts as present");
        assert!(r.layers[0].found);
    }

    #[test]
    fn fails_when_nothing_detected() {
        let plan = LayerPlan { layers: vec![layer("bowl", [0.25, 0.25, 0.75, 0.75], "a bowl of pears")], ..Default::default() };
        let detect = |_q: &str| Ok(vec![]);
        let r = verify(&plan, &geom(), 128, 128, &detect).unwrap();
        assert_eq!(r.failures, vec!["bowl".to_string()]);
        assert_eq!(r.layers[0].score, 0.0);
    }

    #[test]
    fn style_threshold_relaxes_for_painterly_media() {
        assert_eq!(style_threshold(0.1, Some("oil painting, warm")), 0.05, "painterly → halved");
        assert_eq!(style_threshold(0.1, Some("a cinematic illustration")), 0.05, "illustration → halved");
        assert_eq!(style_threshold(0.1, Some("photorealistic, 35mm photo")), 0.1, "photo → unchanged");
        assert_eq!(style_threshold(0.1, None), 0.1, "unknown medium → unchanged");
        assert_eq!(style_threshold(0.03, Some("engraving")), 0.02, "floored at 0.02");
    }

    #[test]
    fn lifted_and_queryless_layers_are_skipped() {
        let plan = LayerPlan {
            layers: vec![
                layer("tiny", [0.4, 0.4, 0.44, 0.44], "a tiny bird"),               // lifted (short side < u)
                Layer { id: "noq".into(), bbox: Some([0.1, 0.1, 0.9, 0.9]), ..Default::default() }, // no prompt/query
            ],
            ..Default::default()
        };
        let detect = |_q: &str| Ok(vec![]);
        let r = verify(&plan, &geom(), 128, 128, &detect).unwrap();
        assert_eq!(r.checked, 0, "neither layer is checked");
        assert!(r.pass(), "nothing to fail");
    }
}
