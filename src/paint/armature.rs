//! The armature and its merge (RFC PAINT-1 §5). The armature is the low-resolution structural stack the model
//! produces — value masses, a colour field, depth, surface normals, region labels. The **merge** resolves a
//! cast of figures into one composition, and every step is cheap at armature resolution and GPU-free:
//! plane assignment, family split (light/shadow), the family-separation invariant, and recession.
//!
//! This module is the structural half; **construction** (filling the channels from SDXL + depth/segmentation)
//! is the GPU consultant that lands in a later sub-slice. Everything here is a pure function of the channels,
//! so it is tested against hand-built synthetic armatures.

use crate::paint::color::{self, Srgb};

/// Construct an armature from PROSE via the model (RFC PAINT-1 §5): render the subject with SDXL, then estimate
/// its depth with Depth-Anything. The model is the consultant that produces STRUCTURE; the stroke engine
/// paints the surface from it. Returns the colour reference and the depth field — the channels the merge
/// consumes. GPU. (Per-figure decomposition + segmentation/normals/saliency are refinements; this single-image
/// construction is the prose→painting bring-up.)
pub async fn construct(subject: &str, w: u32, h: u32, model: &str, steps: usize, seed: u64, device: candle_core::Device) -> anyhow::Result<(image::RgbImage, Vec<f32>)> {
    use anyhow::Context;
    let spec = crate::device::spec_of(&device).to_string();
    let imgs = crate::api::Generate::new(model)
        .prompt(subject)
        .size(w, h)
        .steps(steps)
        .seed(seed)
        .device(&spec)
        .run()
        .await
        .with_context(|| format!("armature: rendering the subject with {model}"))?;
    let img = imgs.into_iter().next().context("armature: the render produced no image")?;
    let colour = image::RgbImage::from_raw(img.width(), img.height(), img.pixels().to_vec()).context("armature: render buffer size mismatch")?;
    // Depth-Anything reads from a path.
    let tmp = tempfile::Builder::new().prefix("plakat-armature-").suffix(".png").tempfile().context("armature scratch file")?;
    colour.save(tmp.path()).context("armature: saving the render for depth")?;
    let depth = crate::pipelines::depth::DepthPipeline::load(device)
        .await
        .context("armature: loading Depth-Anything")?
        .depth_map(tmp.path(), colour.width(), colour.height())
        .context("armature: depth estimation")?;
    Ok((colour, depth))
}

/// The low-resolution structural stack. All channels share the armature resolution `w × h` (small — structure
/// survives downsampling, detail does not).
#[derive(Clone, Debug)]
pub struct Armature {
    pub w: u32,
    pub h: u32,
    /// Notan value in `[0,1]`, row-major.
    pub value: Vec<f32>,
    /// Coarse colour intent (sRGB), row-major.
    pub colour: Vec<Srgb>,
    /// Depth in `[0,1]`, larger = closer (ControlNet-Depth convention).
    pub depth: Vec<f32>,
    /// Unit surface normals `(x,y,z)`, row-major; `z` toward the viewer.
    pub normal: Vec<[f32; 3]>,
    /// Region label per pixel (0 = background), row-major.
    pub region: Vec<u32>,
}

impl Armature {
    pub fn len(&self) -> usize {
        (self.w * self.h) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A per-pixel plane assignment: `plane[i]` in `0..n_planes`, where **0 is the nearest plane** (front) and
/// `n_planes-1` is the farthest (background) — the standard picture-plane numbering (§3.1).
#[derive(Clone, Debug)]
pub struct Planes {
    pub plane: Vec<u32>,
    pub n_planes: u32,
    /// The representative depth of each plane (index 0 = nearest = highest depth).
    pub band_depth: Vec<f32>,
}

impl Planes {
    /// Paint order is the reverse of plane order — far to near. This is the ONLY iterator, so the inversion is
    /// unrepresentable: a painter always lays the background first.
    pub fn far_to_near(&self) -> impl Iterator<Item = u32> {
        (0..self.n_planes).rev()
    }
}

/// Assign each pixel to one of `n_bands` depth planes by clustering the depth histogram into equal-population
/// bands (quantiles), ordered so plane 0 is the nearest. Robust to depth scale; deterministic.
pub fn assign_planes(depth: &[f32], n_bands: u32) -> Planes {
    let n = n_bands.max(1);
    if depth.is_empty() {
        return Planes { plane: Vec::new(), n_planes: n, band_depth: vec![0.0; n as usize] };
    }
    // Quantile thresholds over sorted depth → equal-population bands.
    let mut sorted: Vec<f32> = depth.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let thresh: Vec<f32> = (1..n).map(|k| sorted[((k as usize * sorted.len()) / n as usize).min(sorted.len() - 1)]).collect();

    // Band 0 (by depth ascending) = farthest; we want plane index 0 = NEAREST, so invert.
    let plane: Vec<u32> = depth
        .iter()
        .map(|&d| {
            let band = thresh.iter().filter(|&&t| d >= t).count() as u32; // 0=farthest … n-1=nearest
            (n - 1) - band // invert → 0 = nearest
        })
        .collect();

    // Representative depth per plane (mean), index 0 = nearest.
    let mut sum = vec![0f64; n as usize];
    let mut cnt = vec![0u64; n as usize];
    for (i, &p) in plane.iter().enumerate() {
        sum[p as usize] += depth[i] as f64;
        cnt[p as usize] += 1;
    }
    let band_depth = (0..n as usize).map(|i| if cnt[i] > 0 { (sum[i] / cnt[i] as f64) as f32 } else { 0.0 }).collect();
    Planes { plane, n_planes: n, band_depth }
}

/// Split each pixel into the LIGHT family or the SHADOW family (§3.3) by whether its surface faces the key
/// light: `is_light[i] = (normal · key_dir) > 0`. `key_dir` need not be unit; it is normalised here.
pub fn split_families(normal: &[[f32; 3]], key_dir: [f32; 3]) -> Vec<bool> {
    let m = (key_dir[0] * key_dir[0] + key_dir[1] * key_dir[1] + key_dir[2] * key_dir[2]).sqrt().max(1e-6);
    let k = [key_dir[0] / m, key_dir[1] / m, key_dir[2] / m];
    normal.iter().map(|nrm| nrm[0] * k[0] + nrm[1] * k[1] + nrm[2] * k[2] > 0.0).collect()
}

/// Does the family-separation invariant hold: **no light-family value darker than the lightest shadow-family
/// value** (§3.3)? A single comparison over the merged value field.
pub fn family_invariant_holds(value: &[f32], is_light: &[bool]) -> bool {
    let shadow_max = value.iter().zip(is_light).filter(|(_, l)| !**l).map(|(v, _)| *v).fold(f32::NEG_INFINITY, f32::max);
    if shadow_max == f32::NEG_INFINITY {
        return true; // no shadow family
    }
    value.iter().zip(is_light).filter(|(_, l)| **l).all(|(v, _)| *v >= shadow_max - 1e-6)
}

/// Enforce the family-separation invariant by RAISING any light-family value below the lightest shadow value up
/// to it (the shadow family is untouched). This is what stops a painting reading as spotty rather than solid.
pub fn enforce_family_invariant(value: &mut [f32], is_light: &[bool]) {
    let shadow_max = value.iter().zip(is_light.iter()).filter(|(_, l)| !**l).map(|(v, _)| *v).fold(f32::NEG_INFINITY, f32::max);
    if shadow_max == f32::NEG_INFINITY {
        return;
    }
    for (v, &l) in value.iter_mut().zip(is_light) {
        if l && *v < shadow_max {
            *v = shadow_max;
        }
    }
}

/// Apply atmospheric recession: compress each plane's value toward its mean and reduce its contrast, MORE for
/// farther planes (§5.5.6, §9). The nearest plane (index 0) is untouched; the farthest is flattened toward
/// `far_compression`. Deterministic, in place.
pub fn apply_recession(value: &mut [f32], planes: &Planes, far_compression: f32) {
    let n = planes.n_planes.max(1);
    // Per-plane mean.
    let mut sum = vec![0f64; n as usize];
    let mut cnt = vec![0u64; n as usize];
    for (i, &p) in planes.plane.iter().enumerate() {
        sum[p as usize] += value[i] as f64;
        cnt[p as usize] += 1;
    }
    let mean: Vec<f32> = (0..n as usize).map(|i| if cnt[i] > 0 { (sum[i] / cnt[i] as f64) as f32 } else { 0.5 }).collect();
    let far = far_compression.clamp(0.0, 1.0);
    for (i, v) in value.iter_mut().enumerate() {
        let p = planes.plane[i];
        // 0 at the nearest plane → `far` at the farthest.
        let t = if n > 1 { p as f32 / (n - 1) as f32 } else { 0.0 };
        let compress = far * t; // how much to pull toward the plane mean
        *v = *v + (mean[p as usize] - *v) * compress;
    }
}

/// Quantise the armature's colour field to a limited palette (§5.5.4) — the unification that lets a cast read
/// as one image. Returns, per pixel, the nearest palette mixture's sRGB; uses the ≤3-pigment solver, cached on
/// the quantised source colour (armature resolution is small, so this is cheap).
pub fn quantise_to_palette(colour: &[Srgb], palette: &crate::paint::palette::Palette) -> Vec<Srgb> {
    let mut cache: std::collections::HashMap<u32, Srgb> = std::collections::HashMap::new();
    colour
        .iter()
        .map(|&c| {
            let key = ((c[0] as u32 >> 3) << 10) | ((c[1] as u32 >> 3) << 5) | (c[2] as u32 >> 3);
            *cache.entry(key).or_insert_with(|| crate::paint::mixer::solve_mixture(palette, c, 3).srgb)
        })
        .collect()
}

/// The Rec.601 luma value field `[0,1]` of a colour field — a notan from colour.
pub fn value_from_colour(colour: &[Srgb]) -> Vec<f32> {
    colour.iter().map(|&c| color::linear_luma(color::srgb_to_linear(c))).collect()
}

/// Scale a colour to a target luma while keeping its hue (linear-RGB gain toward the target value).
fn revalue(c: Srgb, target: f32) -> Srgb {
    let lin = color::srgb_to_linear(c);
    let l = color::linear_luma(lin).max(1e-4);
    let g = (target / l).clamp(0.0, 4.0);
    color::linear_to_srgb([lin[0] * g, lin[1] * g, lin[2] * g])
}

/// VALUE RE-KEY (§5.5.2): expand a colour field's tonal range so the picture reads with real lights and darks
/// instead of collapsing toward a mid grey. The 5th/95th luma percentiles are stretched to `[out_low,
/// out_high]` and each pixel re-valued to its new luma (hue kept). Robust to outliers; deterministic.
pub fn value_key(colour: &[Srgb], out_low: f32, out_high: f32) -> Vec<Srgb> {
    if colour.is_empty() {
        return Vec::new();
    }
    let vals = value_from_colour(colour);
    let mut sorted = vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |q: f32| sorted[((q * (sorted.len() - 1) as f32).round() as usize).min(sorted.len() - 1)];
    let (lo, hi) = (p(0.05), p(0.95));
    let span = (hi - lo).max(1e-3);
    colour
        .iter()
        .zip(vals.iter())
        .map(|(&c, &v)| {
            let t = ((v - lo) / span).clamp(0.0, 1.0);
            revalue(c, out_low + t * (out_high - out_low))
        })
        .collect()
}

/// A crude, GPU-free monocular depth PROXY for bring-up: nearer = more central and lower in the frame (a
/// standing-subject / ground-plane prior). The real armature uses a depth estimator; this lets the merge run
/// on CPU. Returns `[0,1]`, larger = closer.
pub fn depth_proxy(w: u32, h: u32) -> Vec<f32> {
    let mut d = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let cx = (x as f32 / w as f32 - 0.5) * 2.0; // −1..1
            let cy = y as f32 / h as f32; // 0 top … 1 bottom
            let central = 1.0 - cx.abs(); // central → near
            let low = cy; // lower → near
            d.push((0.45 * central + 0.55 * low).clamp(0.0, 1.0));
        }
    }
    d
}

/// Compose the merge into a paintable REFERENCE (RFC PAINT-1 §5.5): assign planes from `depth`, apply
/// atmospheric recession to the value, quantise the colour to the palette, and recombine — so the reference
/// the painter works from already carries plane structure, recession, and palette unity. Returns an sRGB field.
pub fn merged_reference(colour: &[Srgb], depth: &[f32], palette: &crate::paint::palette::Palette, n_planes: u32, far_compression: f32) -> Vec<Srgb> {
    let planes = assign_planes(depth, n_planes);
    let mut value = value_from_colour(colour);
    apply_recession(&mut value, &planes, far_compression);
    let quant = quantise_to_palette(colour, palette);
    quant.iter().zip(value.iter()).map(|(&c, &v)| revalue(c, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    #[test]
    fn planes_go_front_to_back_and_paint_far_first() {
        // depth: left column far (0.1), right column near (0.9).
        let depth = vec![0.1, 0.1, 0.9, 0.9];
        let p = assign_planes(&depth, 2);
        assert_eq!(p.n_planes, 2);
        // The near pixels get plane 0 (nearest), the far ones plane 1.
        assert_eq!(p.plane[2], 0, "near → plane 0");
        assert_eq!(p.plane[0], 1, "far → plane 1");
        assert!(p.band_depth[0] > p.band_depth[1], "plane 0 is the nearer (higher-depth) band");
        // Paint order is far → near: 1 then 0.
        assert_eq!(p.far_to_near().collect::<Vec<_>>(), vec![1, 0]);
    }

    #[test]
    fn family_split_by_key_light() {
        // Two surfaces: one facing the light (+x), one facing away (−x). Key light from +x.
        let normal = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let is_light = split_families(&normal, [1.0, 0.0, 0.0]);
        assert_eq!(is_light, vec![true, false, false], "faces the light → light family");
    }

    #[test]
    fn family_invariant_check_and_enforce() {
        // A light-family pixel (value 0.2) darker than the lightest shadow (0.5) violates the invariant.
        let mut value = vec![0.2, 0.5, 0.9];
        let is_light = vec![true, false, true];
        assert!(!family_invariant_holds(&value, &is_light), "0.2 light < 0.5 shadow → violated");
        enforce_family_invariant(&mut value, &is_light);
        assert!(family_invariant_holds(&value, &is_light), "enforced");
        assert_eq!(value[0], 0.5, "the offending light value was raised to the shadow max");
        assert_eq!(value[1], 0.5, "shadow untouched");
    }

    #[test]
    fn recession_flattens_the_far_plane_more() {
        // Two planes; the far plane (index 1) has high internal contrast that recession should compress.
        let depth = vec![0.9, 0.9, 0.1, 0.1];
        let planes = assign_planes(&depth, 2);
        let mut value = vec![0.4, 0.6, 0.0, 1.0]; // near plane 0.4/0.6, far plane 0.0/1.0
        let before_far_range = 1.0 - 0.0;
        apply_recession(&mut value, &planes, 0.8);
        // Far-plane (index 1) values are pulled toward their mean (0.5) → its range shrinks.
        let far_vals: Vec<f32> = value.iter().enumerate().filter(|(i, _)| planes.plane[*i] == 1).map(|(_, &v)| v).collect();
        let far_range = far_vals.iter().cloned().fold(0.0_f32, f32::max) - far_vals.iter().cloned().fold(1.0_f32, f32::min);
        assert!(far_range < before_far_range * 0.5, "the far plane's contrast is compressed ({far_range})");
        // Near plane (index 0) is essentially untouched.
        let near: Vec<f32> = value.iter().enumerate().filter(|(i, _)| planes.plane[*i] == 0).map(|(_, &v)| v).collect();
        assert!((near[0] - 0.4).abs() < 0.02 && (near[1] - 0.6).abs() < 0.02, "near plane kept its range");
    }

    #[test]
    fn merged_reference_applies_recession_and_palette() {
        // Two vertical planes via the proxy? Use explicit depth: top row far, bottom row near.
        let w = 4;
        let h = 2;
        // Colours: high-contrast pair on each row.
        let colour = vec![[10, 10, 10], [240, 240, 240], [10, 10, 10], [240, 240, 240], [10, 10, 10], [240, 240, 240], [10, 10, 10], [240, 240, 240]];
        let depth = vec![0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]; // top far, bottom near
        let merged = merged_reference(&colour, &depth, &palette::EARTH, 2, 0.8);
        let val = value_from_colour(&merged);
        // Far (top) plane's value contrast is compressed vs the near (bottom) plane's.
        let far_range = (val[1] - val[0]).abs();
        let near_range = (val[5] - val[4]).abs();
        assert!(far_range < near_range, "recession compresses the far plane ({far_range} < {near_range})");
    }

    #[test]
    fn value_key_expands_a_flat_range() {
        // A low-contrast field (all mid-grey-ish) → value-key stretches its tonal range wide.
        let colour: Vec<Srgb> = (0..20).map(|i| { let v = (120 + i) as u8; [v, v, v] }).collect();
        let keyed = value_key(&colour, 0.05, 0.95);
        let vin = value_from_colour(&colour);
        let vout = value_from_colour(&keyed);
        let range = |v: &[f32]| v.iter().cloned().fold(0.0_f32, f32::max) - v.iter().cloned().fold(1.0_f32, f32::min);
        assert!(range(&vout) > range(&vin) * 3.0, "tonal range expanded ({} → {})", range(&vin), range(&vout));
    }

    #[test]
    fn depth_proxy_is_nearer_low_and_central() {
        let d = depth_proxy(3, 3);
        // Bottom-centre is nearer than top-corner.
        assert!(d[3 * 2 + 1] > d[0], "bottom-centre nearer than top-left");
    }

    #[test]
    fn palette_quantise_maps_into_gamut() {
        // An out-of-(zorn)-gamut cyan quantises to the nearest reachable warm colour.
        let colour = vec![[0, 200, 220], [190, 145, 60]];
        let q = quantise_to_palette(&colour, &palette::ZORN);
        assert_eq!(q.len(), 2);
        // The ochre stays close to ochre; the cyan is pulled toward the reachable gamut (not still cyan).
        assert!(q[1][0] > q[1][2], "ochre stays warm");
    }
}
