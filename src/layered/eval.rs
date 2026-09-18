//! LAYERED-1 P2 — eval/sweep: the **go/no-go on layered vs plain**.
//!
//! Given a plan and a pair of finished images (a PLAIN single-prompt render vs a LAYERED render), measure how
//! well each honours the plan, then report the delta and a verdict. Three complementary metrics, each
//! optional so eval runs with whatever's available:
//!
//!   * **layout** — model-free: mean low-frequency correlation to the GUIDE inside each anchored box (the
//!     direct measure of whether the anchor took). Needs the guide image (via [`super::diff`]).
//!   * **placement** — U2Net: the fraction of the output's salient mass that falls inside the anchored boxes
//!     (are the subjects where the plan put them?). Needs a matte of the output.
//!   * **semantic** — CLIP: mean per-anchored-layer adherence of the box CROP to the layer's query/prompt (is
//!     the RIGHT subject there?). Needs an injected text-image scorer.
//!
//! The scoring core is pure — the matte and the CLIP scorer are injected — so the whole thing gates offline;
//! the CLI loads the models and feeds it. A corpus [`decide`] turns a set of per-case deltas into GO / NO-GO /
//! INCONCLUSIVE. (This is measurement only; the renders it scores are produced separately.)

use anyhow::Result;
use image::imageops::{crop_imm, resize, FilterType};
use image::{GrayImage, RgbImage};

use crate::layered::diff;
use crate::layered::lint::{layer_class, Class};
use crate::layered::plan::{self, LayerPlan};
use crate::pipelines::noise_space::LatentGeometry;

/// A per-layer breakdown of the metrics that apply per layer.
#[derive(Clone, Debug, Default)]
pub struct LayerScore {
    pub layout: Option<f32>,
    pub semantic: Option<f32>,
}

/// How well one image honours a plan. Each metric is `None` when its inputs weren't supplied; `overall` is
/// the mean of the present components (all in `[0,1]`, higher = better).
#[derive(Clone, Debug)]
pub struct Score {
    pub layout: Option<f32>,
    pub placement: Option<f32>,
    pub semantic: Option<f32>,
    pub overall: f32,
    pub per_layer: Vec<(String, LayerScore)>,
}

/// A layer box in output pixels, clamped, `x1>x0 && y1>y0` (else `None`).
fn box_px(l: &plan::Layer, w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
    let b = plan::layer_box(l);
    let x0 = (b[0] * w as f32).round().clamp(0.0, w as f32) as u32;
    let y0 = (b[1] * h as f32).round().clamp(0.0, h as f32) as u32;
    let x1 = (b[2] * w as f32).round().clamp(0.0, w as f32) as u32;
    let y1 = (b[3] * h as f32).round().clamp(0.0, h as f32) as u32;
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// **layout** — mean low-frequency correlation to the guide over anchored boxes, clamped to `[0,1]`. Also
/// the per-anchored-layer correlations.
pub fn layout_adherence(plan: &LayerPlan, output: &RgbImage, guide: &RgbImage, geom: &LatentGeometry, w: u32, h: u32) -> Result<(f32, Vec<(String, f32)>)> {
    let rep = diff::compare(plan, output, guide, geom, w, h)?;
    let mut per = Vec::new();
    let (mut sum, mut n) = (0.0f32, 0usize);
    // `diff::compare` reports per-layer stats in `plan.layers` order.
    for (l, (id, st)) in plan.layers.iter().zip(rep.layers.iter()) {
        if layer_class(l, geom, w, h) == Class::Anchored {
            let c = st.corr.clamp(0.0, 1.0);
            per.push((id.clone(), c));
            sum += c;
            n += 1;
        }
    }
    Ok((if n > 0 { sum / n as f32 } else { 0.0 }, per))
}

/// **placement** — salient mass inside the anchored boxes / total salient mass, from a matte of the output.
pub fn placement(plan: &LayerPlan, matte: &GrayImage, geom: &LatentGeometry, w: u32, h: u32) -> f32 {
    let m = resize(matte, w, h, FilterType::Triangle);
    let boxes: Vec<(u32, u32, u32, u32)> = plan
        .layers
        .iter()
        .filter(|l| layer_class(l, geom, w, h) == Class::Anchored)
        .filter_map(|l| box_px(l, w, h))
        .collect();
    if boxes.is_empty() {
        return 0.0;
    }
    let (mut total, mut inside) = (0f64, 0f64);
    for (x, y, p) in m.enumerate_pixels() {
        let a = p.0[0] as f64;
        total += a;
        if boxes.iter().any(|&(x0, y0, x1, y1)| x >= x0 && x < x1 && y >= y0 && y < y1) {
            inside += a;
        }
    }
    if total > 1.0 {
        (inside / total) as f32
    } else {
        0.0
    }
}

/// Rescale a raw CLIP adherence (~`0.10..0.35` for a faithful crop) into `[0,1]`.
fn rescale_adherence(a: f32) -> f32 {
    ((a - 0.10) / 0.25).clamp(0.0, 1.0)
}

/// **semantic** — mean per-anchored-layer CLIP adherence of the box crop to the layer's query/prompt,
/// rescaled to `[0,1]`. `scorer(crop, text)` returns a raw adherence in ~`[-1,1]`.
pub fn semantic(
    plan: &LayerPlan,
    output: &RgbImage,
    geom: &LatentGeometry,
    w: u32,
    h: u32,
    scorer: &dyn Fn(&RgbImage, &str) -> Result<f32>,
) -> Result<(f32, Vec<(String, f32)>)> {
    let out = resize(output, w, h, FilterType::Triangle);
    let mut per = Vec::new();
    let (mut sum, mut n) = (0.0f32, 0usize);
    for l in &plan.layers {
        if layer_class(l, geom, w, h) != Class::Anchored {
            continue;
        }
        let text = l.query.clone().or_else(|| l.prompt.clone()).unwrap_or_default();
        if text.trim().is_empty() {
            continue;
        }
        let Some((x0, y0, x1, y1)) = box_px(l, w, h) else { continue };
        let crop = crop_imm(&out, x0, y0, x1 - x0, y1 - y0).to_image();
        let s = rescale_adherence(scorer(&crop, &text)?);
        per.push((l.id.clone(), s));
        sum += s;
        n += 1;
    }
    Ok((if n > 0 { sum / n as f32 } else { 0.0 }, per))
}

/// Combine the present metric components into a [`Score`] (overall = mean of the present ones).
pub fn combine(layout: Option<(f32, Vec<(String, f32)>)>, placement: Option<f32>, semantic: Option<(f32, Vec<(String, f32)>)>) -> Score {
    let (layout_v, layout_per) = match layout {
        Some((v, per)) => (Some(v), per),
        None => (None, Vec::new()),
    };
    let (semantic_v, semantic_per) = match semantic {
        Some((v, per)) => (Some(v), per),
        None => (None, Vec::new()),
    };
    let comps: Vec<f32> = [layout_v, placement, semantic_v].into_iter().flatten().collect();
    let overall = if comps.is_empty() { 0.0 } else { comps.iter().sum::<f32>() / comps.len() as f32 };

    // Merge the per-layer maps.
    let mut per_layer: Vec<(String, LayerScore)> = Vec::new();
    for (id, v) in layout_per {
        entry(&mut per_layer, &id).layout = Some(v);
    }
    for (id, v) in semantic_per {
        entry(&mut per_layer, &id).semantic = Some(v);
    }
    Score { layout: layout_v, placement, semantic: semantic_v, overall, per_layer }
}

/// Get (or insert) the per-layer score entry for `id`.
fn entry<'a>(per: &'a mut Vec<(String, LayerScore)>, id: &str) -> &'a mut LayerScore {
    if let Some(i) = per.iter().position(|(k, _)| k == id) {
        return &mut per[i].1;
    }
    per.push((id.to_string(), LayerScore::default()));
    let last = per.len() - 1;
    &mut per[last].1
}

/// The change from plain → layered (positive = layered better) for each present metric.
#[derive(Clone, Copy, Debug)]
pub struct Delta {
    pub layout: Option<f32>,
    pub placement: Option<f32>,
    pub semantic: Option<f32>,
    pub overall: f32,
}

/// Compute the plain → layered delta.
pub fn delta(plain: &Score, layered: &Score) -> Delta {
    let d = |a: Option<f32>, b: Option<f32>| match (a, b) {
        (Some(x), Some(y)) => Some(y - x),
        _ => None,
    };
    Delta {
        layout: d(plain.layout, layered.layout),
        placement: d(plain.placement, layered.placement),
        semantic: d(plain.semantic, layered.semantic),
        overall: layered.overall - plain.overall,
    }
}

/// The corpus decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Go,
    NoGo,
    Inconclusive,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Go => "GO",
            Verdict::NoGo => "NO-GO",
            Verdict::Inconclusive => "INCONCLUSIVE",
        }
    }
}

/// A corpus go/no-go over per-case overall deltas: **GO** when the mean advantage ≥ `margin` AND layered wins
/// at least 60% of cases; **NO-GO** when the mean ≤ `−margin`; otherwise **INCONCLUSIVE**. Returns the mean
/// delta and the win rate alongside.
pub fn decide(overall_deltas: &[f32], margin: f32) -> (Verdict, f32, f32) {
    if overall_deltas.is_empty() {
        return (Verdict::Inconclusive, 0.0, 0.0);
    }
    let n = overall_deltas.len() as f32;
    let mean = overall_deltas.iter().sum::<f32>() / n;
    let win_rate = overall_deltas.iter().filter(|&&d| d > 0.0).count() as f32 / n;
    let v = if mean >= margin && win_rate >= 0.6 {
        Verdict::Go
    } else if mean <= -margin {
        Verdict::NoGo
    } else {
        Verdict::Inconclusive
    };
    (v, mean, win_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered::plan::Layer;
    use image::{Luma, Rgb};

    fn geom() -> LatentGeometry {
        LatentGeometry { v: 8, u: 8, pool_levels: 2 }
    }

    fn anchored(id: &str, bbox: [f32; 4]) -> Layer {
        Layer { id: id.into(), bbox: Some(bbox), prompt: Some("a subject".into()), ..Default::default() }
    }

    #[test]
    fn layout_is_perfect_when_output_equals_guide() {
        let plan = LayerPlan { layers: vec![anchored("s", [0.25, 0.25, 0.75, 0.75])], ..Default::default() };
        let mut g = RgbImage::new(128, 128);
        for (x, y, p) in g.enumerate_pixels_mut() {
            *p = Rgb([(x * 2) as u8, (y * 2) as u8, 90]);
        }
        let (v, per) = layout_adherence(&plan, &g, &g, &geom(), 128, 128).unwrap();
        assert!(v > 0.98, "output==guide → layout ~1 (got {v})");
        assert_eq!(per.len(), 1);
    }

    #[test]
    fn placement_high_when_saliency_inside_box() {
        let plan = LayerPlan { layers: vec![anchored("s", [0.25, 0.25, 0.75, 0.75])], ..Default::default() };
        // Matte: salient only inside the box.
        let mut inside = GrayImage::new(128, 128);
        for y in 40..88 {
            for x in 40..88 {
                inside.put_pixel(x, y, Luma([255]));
            }
        }
        assert!(placement(&plan, &inside, &geom(), 128, 128) > 0.95, "all saliency in box → ~1");
        // Matte: salient only in a corner (outside the box).
        let mut outside = GrayImage::new(128, 128);
        for y in 0..16 {
            for x in 0..16 {
                outside.put_pixel(x, y, Luma([255]));
            }
        }
        assert!(placement(&plan, &outside, &geom(), 128, 128) < 0.05, "saliency outside box → ~0");
    }

    #[test]
    fn semantic_uses_the_injected_scorer_per_anchored_layer() {
        let plan = LayerPlan {
            layers: vec![anchored("s", [0.25, 0.25, 0.75, 0.75]), Layer { id: "tiny".into(), bbox: Some([0.4, 0.4, 0.44, 0.44]), prompt: Some("x".into()), ..Default::default() }],
            ..Default::default()
        };
        let out = RgbImage::from_pixel(128, 128, Rgb([10, 10, 10]));
        // Stub scorer: high adherence.
        let scorer = |_c: &RgbImage, _t: &str| Ok(0.30f32);
        let (v, per) = semantic(&plan, &out, &geom(), 128, 128, &scorer).unwrap();
        assert_eq!(per.len(), 1, "only the anchored layer is scored (tiny is lifted)");
        assert!(v > 0.7, "0.30 adherence rescales high (got {v})");
    }

    #[test]
    fn combine_averages_present_components() {
        let s = combine(Some((0.8, vec![("s".into(), 0.8)])), Some(0.6), None);
        assert_eq!(s.layout, Some(0.8));
        assert_eq!(s.placement, Some(0.6));
        assert_eq!(s.semantic, None);
        assert!((s.overall - 0.7).abs() < 1e-6, "mean of 0.8 and 0.6");
        assert_eq!(s.per_layer.len(), 1);
    }

    #[test]
    fn decide_go_nogo_inconclusive() {
        // Clear win: mean 0.076, win-rate 2/3.
        let (v, mean, wr) = decide(&[0.10, 0.15, -0.02], 0.03);
        assert_eq!(v, Verdict::Go);
        assert!(mean > 0.03 && wr > 0.6);
        // Clear loss.
        assert_eq!(decide(&[-0.1, -0.2, -0.05], 0.03).0, Verdict::NoGo);
        // Marginal / split.
        assert_eq!(decide(&[0.01, -0.01, 0.005], 0.03).0, Verdict::Inconclusive);
        // Empty.
        assert_eq!(decide(&[], 0.03).0, Verdict::Inconclusive);
    }
}
