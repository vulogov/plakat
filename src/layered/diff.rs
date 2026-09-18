//! LAYERED-1 `diff` — a model-free way to see how well one image honours another: the guide vs the finish,
//! or two finishes. It compares LOW frequencies (the band the anchor actually constrains) over the whole
//! canvas and inside each layer box, reporting a normalised mean-absolute error and a correlation, plus an
//! optional absolute-difference heatmap. This is the statistical/visual companion to the S4 semantic verify
//! (OWL-ViT "is the subject in its box?"), which lands in P3.

use anyhow::{Context, Result};
use image::{GrayImage, Luma, RgbImage};

use crate::layered::plan::{self, LayerPlan};
use crate::pipelines::noise_space::LatentGeometry;

/// Low-frequency agreement between two fields over a region.
#[derive(Clone, Copy, Debug)]
pub struct DiffStats {
    /// Mean absolute error of the low-passed luma, normalised to `[0,1]` (0 = identical).
    pub mae: f32,
    /// Pearson correlation of the low-passed luma over the region (`1` = same structure, `0` = unrelated).
    pub corr: f32,
    /// Number of pixels the stats were computed over.
    pub n: usize,
}

/// Rec.601 luma field of an image, `[0,255]`.
fn luma_field(img: &RgbImage) -> Vec<f32> {
    img.pixels().map(|p| 0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32).collect()
}

/// Separable box blur of a field (edge-clamped). `r = 0` is the identity.
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let win = (2 * r + 1) as f32;
    let ri = r as i32;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0f32;
            for k in -ri..=ri {
                let xx = (x as i32 + k).clamp(0, w as i32 - 1) as usize;
                s += src[y * w + xx];
            }
            tmp[y * w + x] = s / win;
        }
    }
    let mut out = vec![0f32; w * h];
    for x in 0..w {
        for y in 0..h {
            let mut s = 0f32;
            for k in -ri..=ri {
                let yy = (y as i32 + k).clamp(0, h as i32 - 1) as usize;
                s += tmp[yy * w + x];
            }
            out[y * w + x] = s / win;
        }
    }
    out
}

/// The low-pass radius for a geometry: one anchored unit, `2^L · v` px (matches the anchor's low band).
pub fn lowpass_radius(g: &LatentGeometry) -> usize {
    ((1usize << g.pool_levels) * g.v).max(1)
}

/// Compare two low-passed fields over the pixels selected by `mask` (`None` = all pixels).
fn stats_over(a: &[f32], b: &[f32], mask: Option<&dyn Fn(usize) -> bool>) -> DiffStats {
    let idx: Vec<usize> = (0..a.len()).filter(|&i| mask.map(|m| m(i)).unwrap_or(true)).collect();
    let n = idx.len();
    if n == 0 {
        return DiffStats { mae: 0.0, corr: 1.0, n: 0 };
    }
    let mae = idx.iter().map(|&i| (a[i] - b[i]).abs()).sum::<f32>() / n as f32 / 255.0;
    let (mut ma, mut mb) = (0f32, 0f32);
    for &i in &idx {
        ma += a[i];
        mb += b[i];
    }
    ma /= n as f32;
    mb /= n as f32;
    let (mut cov, mut va, mut vb) = (0f32, 0f32, 0f32);
    for &i in &idx {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    let corr = if denom > 1e-6 { cov / denom } else { 1.0 };
    DiffStats { mae, corr, n }
}

/// A whole-canvas + per-layer low-frequency comparison of two images (resized to `w×h`).
pub struct DiffReport {
    pub overall: DiffStats,
    pub layers: Vec<(String, DiffStats)>,
    /// Absolute-difference heatmap of the low-passed luma (`w×h`, 255 = max difference seen).
    pub heatmap: GrayImage,
}

/// Compare image `a` against image `b` at `w×h`, low-passing at the finish geometry's band, over the whole
/// canvas and inside every layer box.
pub fn compare(plan: &LayerPlan, a: &RgbImage, b: &RgbImage, geom: &LatentGeometry, w: u32, h: u32) -> Result<DiffReport> {
    use image::imageops::{resize, FilterType};
    let a = resize(a, w, h, FilterType::Triangle);
    let b = resize(b, w, h, FilterType::Triangle);
    let r = lowpass_radius(geom);
    let (wu, hu) = (w as usize, h as usize);
    let la = box_blur(&luma_field(&a), wu, hu, r);
    let lb = box_blur(&luma_field(&b), wu, hu, r);

    let overall = stats_over(&la, &lb, None);

    let mut layers = Vec::new();
    for l in &plan.layers {
        let bb = plan::layer_box(l);
        let (x0, y0) = ((bb[0] * w as f32) as usize, (bb[1] * h as f32) as usize);
        let (x1, y1) = ((bb[2] * w as f32).ceil() as usize, (bb[3] * h as f32).ceil() as usize);
        let inside = move |i: usize| {
            let (x, y) = (i % wu, i / wu);
            x >= x0 && x < x1.min(wu) && y >= y0 && y < y1.min(hu)
        };
        layers.push((l.id.clone(), stats_over(&la, &lb, Some(&inside))));
    }

    // Heatmap: |la − lb|, scaled so the largest difference maps to 255 (visual, not absolute).
    let mut diff = vec![0f32; wu * hu];
    let mut peak = 1e-6f32;
    for i in 0..diff.len() {
        diff[i] = (la[i] - lb[i]).abs();
        peak = peak.max(diff[i]);
    }
    let mut heatmap = GrayImage::new(w, h);
    for (i, &d) in diff.iter().enumerate() {
        heatmap.put_pixel((i % wu) as u32, (i / wu) as u32, Luma([((d / peak) * 255.0).round().clamp(0.0, 255.0) as u8]));
    }

    Ok(DiffReport { overall, layers, heatmap })
}

/// Load an RGB image from a path.
pub fn load_rgb(path: &std::path::Path) -> Result<RgbImage> {
    Ok(image::open(path).with_context(|| format!("loading {}", path.display()))?.to_rgb8())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn geom() -> LatentGeometry {
        LatentGeometry { v: 8, u: 8, pool_levels: 2 }
    }

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(rgb))
    }

    #[test]
    fn identical_images_are_perfect() {
        let plan = LayerPlan::default();
        let img = {
            let mut m = RgbImage::new(64, 64);
            for (x, y, p) in m.enumerate_pixels_mut() {
                *p = Rgb([(x * 4) as u8, (y * 4) as u8, 128]);
            }
            m
        };
        let r = compare(&plan, &img, &img, &geom(), 64, 64).unwrap();
        assert!(r.overall.mae < 1e-4, "identical → ~0 MAE (got {})", r.overall.mae);
        assert!(r.overall.corr > 0.99, "identical → corr ~1 (got {})", r.overall.corr);
    }

    #[test]
    fn different_images_have_higher_mae() {
        let plan = LayerPlan::default();
        let a = solid(64, 64, [20, 20, 20]);
        let b = solid(64, 64, [220, 220, 220]);
        let r = compare(&plan, &a, &b, &geom(), 64, 64).unwrap();
        assert!(r.overall.mae > 0.5, "black vs white → large MAE (got {})", r.overall.mae);
    }

    #[test]
    fn per_layer_box_isolates_the_region() {
        use crate::layered::plan::Layer;
        // `a` and `b` agree everywhere except a bright patch inside the layer box → the box's MAE is worse
        // than the corner's (checked via overall being dominated by agreement outside).
        let plan = LayerPlan {
            layers: vec![Layer { id: "s".into(), bbox: Some([0.25, 0.25, 0.75, 0.75]), ..Default::default() }],
            ..Default::default()
        };
        let a = solid(64, 64, [30, 30, 30]);
        let mut b = a.clone();
        for y in 16..48 {
            for x in 16..48 {
                b.put_pixel(x, y, Rgb([230, 230, 230]));
            }
        }
        let r = compare(&plan, &a, &b, &geom(), 64, 64).unwrap();
        let (_, box_stats) = &r.layers[0];
        assert!(box_stats.mae > r.overall.mae, "the changed box has worse MAE than the whole canvas ({} vs {})", box_stats.mae, r.overall.mae);
        assert!(box_stats.n > 0);
    }
}
