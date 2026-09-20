//! LAYERED-1 S2 — the **guide**. The per-layer drafts + backdrop (S1) are composed into a single guide
//! IMAGE — each ANCHORED subject matted off its draft, luma-normalised to the backdrop, laid back-to-front
//! by depth into its box — and a pair of per-pixel anchor maps is built at latent resolution: the weight `W`
//! (how hard to pull toward the guide) and the window end `E` (the step fraction at which anchoring stops).
//!
//! The FINISH stage (S3) VAE-encodes the guide image into `G` using the finish family's OWN VAE (so `G` is
//! guaranteed in-space) and runs one anchored trajectory with [`LayeredHook`](super::hook::LayeredHook)
//! `{G, W, E}`. Keeping the encode in the render stage keeps this module model-free and gate-safe: the
//! silhouette is injected, so the whole composition + maps path is exercised offline.
//!
//! Only ANCHORED layers paint into the guide and raise `W`/`E`. Hinted/lifted subjects are the finish
//! prompt's business (premise D1: layers are constraints, never pixels — only the low frequencies of `G` are
//! ever substituted, and only inside each anchored pixel's window).

use anyhow::Result;
use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use image::{GrayImage, Luma, Rgb, RgbImage};

use crate::layered::draft::DraftSet;
use crate::layered::lint::{layer_class, Class};
use crate::layered::plan::{self, Layer, LayerPlan};
use crate::pipelines::matting;
use crate::pipelines::noise_space::LatentGeometry;

/// Backdrop anchor defaults (a weak, whole-canvas pull that fades early).
const BACKDROP_WEIGHT: f32 = 0.6;
const BACKDROP_WINDOW: f32 = 0.25;
/// Anchored-subject anchor defaults (a firm pull held over the early-to-mid trajectory).
const LAYER_WEIGHT: f32 = 0.85;
const LAYER_WINDOW: f32 = 0.6;
/// Luma-normalisation gain is clamped so a mis-lit draft can't blow out or crush the composite.
const GAIN_LO: f32 = 0.6;
const GAIN_HI: f32 = 1.6;
/// How dark a contact shadow gets at full strength (multiplies the canvas underneath).
const SHADOW_DARK: f32 = 0.5;

/// Cohesion controls for [`compose`] / [`build`] — how each subject is blended into the scene so the guide
/// reads as one place, not a collage. All are pure image ops (no GPU); anchoring only sees low frequencies,
/// so coarse grounding + colour agreement is exactly the right level of effort here.
#[derive(Clone, Copy, Debug)]
pub struct GuideOpts {
    /// Lay a soft contact shadow under each subject (grounds floating cut-outs). Default on.
    pub ground: bool,
    /// Contact-shadow softness (penumbra scale). Default 1.0.
    pub ground_softness: f32,
    /// Colour-harmonise each subject toward the backdrop's mean colour, `0..1` (0 = off, replaces the old
    /// luma-only gain with a per-channel shift so palettes agree). Default 0.4.
    pub harmonize: f32,
    /// Directional relight amplitude `0..~0.4` — a coarse luminance gradient across each subject matching the
    /// key-light direction (0 = off; the finish does the real lighting). Default 0 (off).
    pub relight_amp: f32,
    /// Key-light direction in degrees for relight + shadow offset: `90` = overhead, `0` = from the right,
    /// `180` = from the left. Default 90 (overhead — soft symmetric shadow).
    pub light_angle: f32,
}

impl Default for GuideOpts {
    fn default() -> Self {
        Self { ground: true, ground_softness: 1.0, harmonize: 0.4, relight_amp: 0.0, light_angle: 90.0 }
    }
}

/// Mean per-channel colour of an image, `[r,g,b]` in `[0,255]`.
fn mean_rgb(img: &RgbImage) -> [f32; 3] {
    let n = (img.width() * img.height()).max(1) as f32;
    let mut s = [0f32; 3];
    for p in img.pixels() {
        for c in 0..3 {
            s[c] += p.0[c] as f32;
        }
    }
    [s[0] / n, s[1] / n, s[2] / n]
}

/// Shift `img`'s per-channel mean a fraction `strength` toward `target` (colour harmonisation). Subsumes luma
/// matching: brightness AND colour move toward the backdrop, while the subject keeps its own variation.
fn harmonize_toward(img: &mut RgbImage, target: [f32; 3], strength: f32) {
    let cur = mean_rgb(img);
    let shift = [(target[0] - cur[0]) * strength, (target[1] - cur[1]) * strength, (target[2] - cur[2]) * strength];
    for p in img.pixels_mut() {
        for c in 0..3 {
            p.0[c] = (p.0[c] as f32 + shift[c]).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Multiply a coarse luminance gradient across `img` matching a key-light `angle` (deg): the side toward the
/// light gets brighter by up to `amp`, the far side darker. Only the low frequencies survive into the guide,
/// so this conveys light DIRECTION without touching detail.
fn directional_shade(img: &mut RgbImage, angle_deg: f32, amp: f32) {
    if amp.abs() < 1e-4 {
        return;
    }
    let (w, h) = (img.width() as f32, img.height() as f32);
    let rad = angle_deg.to_radians();
    // Screen coords: +x right, +y DOWN. A light at `angle` (90=top) points toward (cos, -sin).
    let (lx, ly) = (rad.cos(), -rad.sin());
    for (x, y, p) in img.enumerate_pixels_mut() {
        // Offset from centre in [-1,1].
        let (ox, oy) = ((x as f32 / w) * 2.0 - 1.0, (y as f32 / h) * 2.0 - 1.0);
        let d = (ox * lx + oy * ly).clamp(-1.0, 1.0); // +1 toward light, -1 away
        let k = 1.0 + amp * d;
        for c in 0..3 {
            p.0[c] = (p.0[c] as f32 * k).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// The foot line: the lowest row of `alpha` (box-sized) that still has appreciable coverage — where a subject
/// meets the ground, so the contact shadow pools there.
fn foot_line(alpha: &GrayImage, w: u32, h: u32) -> usize {
    for y in (0..h).rev() {
        let covered = (0..w).any(|x| alpha.get_pixel(x, y).0[0] > 76);
        if covered {
            return y as usize;
        }
    }
    (h.saturating_sub(1)) as usize
}

/// The horizontal shadow key `[-1,1]` from a light `angle` (deg): light from the left throws the shadow right.
fn shadow_key(angle_deg: f32) -> f32 {
    (-angle_deg.to_radians().cos()).clamp(-1.0, 1.0)
}

/// A composed guide: the guide image + the per-pixel anchor maps at latent resolution.
pub struct Guide {
    /// The composed guide image at output resolution (backdrop + matted anchored subjects).
    pub canvas: RgbImage,
    /// Per-pixel anchor weight `W`, shape `(1,1,H,W)` at latent resolution, in `[0,1]`.
    pub weight: Tensor,
    /// Per-pixel window end `E`, shape `(1,1,H,W)` at latent resolution, in `[0,1]`.
    pub window_end: Tensor,
    /// Per-anchored-layer placed silhouette (canvas resolution, 255 = subject) — for `diff`/inspection.
    pub placed: Vec<(String, GrayImage)>,
}

/// The Rec.601 luma of an RGB triple.
fn luma(p: &Rgb<u8>) -> f32 {
    0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32
}

/// Mean luma of an image (0 for an empty image).
fn luma_mean(img: &RgbImage) -> f32 {
    let n = (img.width() * img.height()) as f32;
    if n == 0.0 {
        return 0.0;
    }
    img.pixels().map(luma).sum::<f32>() / n
}

/// Scale every channel of an image by `gain`, clamped to `[0,255]`.
fn apply_gain(img: &mut RgbImage, gain: f32) {
    for p in img.pixels_mut() {
        for c in 0..3 {
            p.0[c] = (p.0[c] as f32 * gain).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// A layer's resolved box in output pixels, clamped to the canvas: `(x0,y0,x1,y1)` with `x1>x0`, `y1>y0`.
fn box_px(layer: &Layer, out_w: u32, out_h: u32) -> Option<(u32, u32, u32, u32)> {
    let b = plan::layer_box(layer);
    let x0 = (b[0] * out_w as f32).round().clamp(0.0, out_w as f32) as u32;
    let y0 = (b[1] * out_h as f32).round().clamp(0.0, out_h as f32) as u32;
    let x1 = (b[2] * out_w as f32).round().clamp(0.0, out_w as f32) as u32;
    let y1 = (b[3] * out_h as f32).round().clamp(0.0, out_h as f32) as u32;
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// Compose the guide IMAGE from a draft set: matte each ANCHORED subject and lay it back-to-front by depth
/// into its box (luma-normalised to the backdrop). `matte` supplies the silhouette (injected so the
/// composition logic is testable without the U2Net model). Returns the canvas + the placed silhouettes
/// (full-canvas, 255 = subject) in paint order.
pub fn compose(
    plan: &LayerPlan,
    drafts: &DraftSet,
    geom: &LatentGeometry,
    out_w: u32,
    out_h: u32,
    opts: &GuideOpts,
    matte: impl Fn(&RgbImage) -> Result<GrayImage>,
) -> Result<(RgbImage, Vec<(String, GrayImage)>)> {
    let mut canvas = image::imageops::resize(&drafts.backdrop, out_w, out_h, FilterType::Triangle);
    let backdrop_mean = luma_mean(&canvas).max(1.0);
    let backdrop_rgb = mean_rgb(&canvas);
    let key = shadow_key(opts.light_angle);

    // Anchored layers that have a draft, laid back-to-front (largest depth first, so nearer wins overlaps).
    let mut items: Vec<(&Layer, &RgbImage)> = plan
        .layers
        .iter()
        .filter(|l| layer_class(l, geom, out_w, out_h) == Class::Anchored)
        .filter_map(|l| drafts.layers.iter().find(|(id, _)| id == &l.id).map(|(_, img)| (l, img)))
        .collect();
    items.sort_by(|(a, _), (b, _)| plan::layer_depth(b).partial_cmp(&plan::layer_depth(a)).unwrap_or(std::cmp::Ordering::Equal));

    let mut placed = Vec::new();
    for (layer, draft) in items {
        let Some((x0, y0, x1, y1)) = box_px(layer, out_w, out_h) else { continue };
        let (bw, bh) = (x1 - x0, y1 - y0);

        // Subject RGB, resized to its box, then matched to the scene: colour-harmonise toward the backdrop
        // (subsumes luma matching), else fall back to the clamped luma gain; optionally directional-relight.
        let mut sub = image::imageops::resize(draft, bw, bh, FilterType::Triangle);
        if opts.harmonize > 0.0 {
            harmonize_toward(&mut sub, backdrop_rgb, opts.harmonize.clamp(0.0, 1.0));
        } else {
            let gain = (backdrop_mean / luma_mean(&sub).max(1.0)).clamp(GAIN_LO, GAIN_HI);
            apply_gain(&mut sub, gain);
        }
        directional_shade(&mut sub, opts.light_angle, opts.relight_amp);

        // Silhouette from the *native* draft, resized to the box and edge-crisped.
        let alpha = matte(draft)?;
        let alpha = matting::refine_matte(&image::imageops::resize(&alpha, bw, bh, FilterType::Triangle));

        // Grounding: lay a soft contact shadow onto the canvas UNDER the subject (before compositing it),
        // pooled at the foot line — so the subject sits in the scene instead of floating on it.
        if opts.ground {
            let alpha_f: Vec<f32> = alpha.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
            let gy = foot_line(&alpha, bw, bh);
            let shadow = crate::product::ground::contact_shadow(&alpha_f, bw as usize, bh as usize, gy, crate::product::ground::ShadowKind::Soft, key, opts.ground_softness.max(0.2));
            for yy in 0..bh {
                for xx in 0..bw {
                    let s = shadow[(yy * bw + xx) as usize];
                    if s > 1e-3 {
                        let (cx, cy) = (x0 + xx, y0 + yy);
                        let p = canvas.get_pixel(cx, cy).0;
                        let k = 1.0 - s * SHADOW_DARK;
                        canvas.put_pixel(cx, cy, Rgb([(p[0] as f32 * k) as u8, (p[1] as f32 * k) as u8, (p[2] as f32 * k) as u8]));
                    }
                }
            }
        }

        let mut full = GrayImage::new(out_w, out_h);
        for yy in 0..bh {
            for xx in 0..bw {
                let a = alpha.get_pixel(xx, yy).0[0] as f32 / 255.0;
                let (cx, cy) = (x0 + xx, y0 + yy);
                let bg = *canvas.get_pixel(cx, cy);
                let fg = *sub.get_pixel(xx, yy);
                let mix = |i: usize| (a * fg.0[i] as f32 + (1.0 - a) * bg.0[i] as f32).round() as u8;
                canvas.put_pixel(cx, cy, Rgb([mix(0), mix(1), mix(2)]));
                full.put_pixel(cx, cy, Luma([alpha.get_pixel(xx, yy).0[0]]));
            }
        }
        placed.push((layer.id.clone(), full));
    }
    Ok((canvas, placed))
}

/// Latent resolution for an output size under a family geometry: `out / v`, at least 1.
fn latent_dims(out_w: u32, out_h: u32, v: u32) -> (u32, u32) {
    ((out_w / v).max(1), (out_h / v).max(1))
}

/// Separable box blur of an `[0,1]` field (edge-clamped). `r = 0` is the identity.
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

/// Area-average downsample of a source field to `(dw,dh)` (each destination cell = mean of the source pixels
/// that map into it). Robust to non-integer ratios.
fn downsample_avg(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
    let mut acc = vec![0f32; dw * dh];
    let mut cnt = vec![0u32; dw * dh];
    for y in 0..sh {
        let dy = (y * dh / sh).min(dh - 1);
        for x in 0..sw {
            let dx = (x * dw / sw).min(dw - 1);
            let di = dy * dw + dx;
            acc[di] += src[y * sw + x];
            cnt[di] += 1;
        }
    }
    for i in 0..acc.len() {
        if cnt[i] > 0 {
            acc[i] /= cnt[i] as f32;
        }
    }
    acc
}

/// Build the anchor maps `W` and `E` at latent resolution. Every pixel starts at the backdrop's (weak)
/// anchor; each anchored layer over-composites its own weight/window inside its FEATHERED silhouette (soft
/// by one latent cell), then the full-res maps are area-averaged down to `out / v`.
pub fn build_maps(
    plan: &LayerPlan,
    placed: &[(String, GrayImage)],
    geom: &LatentGeometry,
    out_w: u32,
    out_h: u32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let n = (out_w * out_h) as usize;
    let bw = plan.backdrop.as_ref().and_then(|b| b.weight).unwrap_or(BACKDROP_WEIGHT).clamp(0.0, 1.0);
    let be = plan.backdrop.as_ref().and_then(|b| b.window).unwrap_or(BACKDROP_WINDOW).clamp(0.0, 1.0);
    let mut wbuf = vec![bw; n];
    let mut ebuf = vec![be; n];

    let feather_r = geom.v.max(1); // one latent cell of soft edge
    for (id, sil) in placed {
        let layer = plan.layers.iter().find(|l| &l.id == id);
        let lw = layer.and_then(|l| l.weight).unwrap_or(LAYER_WEIGHT).clamp(0.0, 1.0);
        let le = layer.and_then(|l| l.window).unwrap_or(LAYER_WINDOW).clamp(0.0, 1.0);
        let raw: Vec<f32> = sil.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
        let a = box_blur(&raw, out_w as usize, out_h as usize, feather_r);
        for i in 0..n {
            wbuf[i] = a[i] * lw + (1.0 - a[i]) * wbuf[i];
            ebuf[i] = a[i] * le + (1.0 - a[i]) * ebuf[i];
        }
    }

    let (lw_, lh_) = latent_dims(out_w, out_h, geom.v as u32);
    let wlat = downsample_avg(&wbuf, out_w, out_h, lw_, lh_);
    let elat = downsample_avg(&ebuf, out_w, out_h, lw_, lh_);
    let shape = (1usize, 1usize, lh_ as usize, lw_ as usize);
    Ok((Tensor::from_vec(wlat, shape, device)?, Tensor::from_vec(elat, shape, device)?))
}

/// Compose the guide + build its anchor maps in one call.
pub fn build(
    plan: &LayerPlan,
    drafts: &DraftSet,
    geom: &LatentGeometry,
    out_w: u32,
    out_h: u32,
    device: &Device,
    opts: &GuideOpts,
    matte: impl Fn(&RgbImage) -> Result<GrayImage>,
) -> Result<Guide> {
    let (canvas, placed) = compose(plan, drafts, geom, out_w, out_h, opts, matte)?;
    let (weight, window_end) = build_maps(plan, &placed, geom, out_w, out_h, device)?;
    Ok(Guide { canvas, weight, window_end, placed })
}

/// Render a `(1,1,H,W)` `[0,1]` map tensor as a grayscale image (255 = full anchor), for `layers guide`
/// visualisation. Upscale to the canvas for viewing at the call site.
pub fn map_to_gray(map: &Tensor) -> Result<GrayImage> {
    let (_, _, h, w) = map.dims4()?;
    let v: Vec<f32> = map.flatten_all()?.to_vec1()?;
    let mut g = GrayImage::new(w as u32, h as u32);
    for (i, &a) in v.iter().enumerate() {
        g.put_pixel((i % w) as u32, (i / w) as u32, Luma([(a.clamp(0.0, 1.0) * 255.0).round() as u8]));
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered::plan::Backdrop;

    fn geom() -> LatentGeometry {
        LatentGeometry { v: 8, u: 8, pool_levels: 2 }
    }

    /// Cohesion fully OFF — exercises the base composite (the pre-cohesion behaviour these tests assert).
    fn bare() -> GuideOpts {
        GuideOpts { ground: false, ground_softness: 1.0, harmonize: 0.0, relight_amp: 0.0, light_angle: 90.0 }
    }

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(rgb))
    }

    /// A stub silhouette: the whole draft is foreground (255).
    fn full_matte(img: &RgbImage) -> Result<GrayImage> {
        Ok(GrayImage::from_pixel(img.width(), img.height(), Luma([255])))
    }

    fn anchored_layer(id: &str, bbox: [f32; 4], depth: f32) -> Layer {
        Layer { id: id.into(), bbox: Some(bbox), depth: Some(depth), ..Default::default() }
    }

    #[test]
    fn compose_lays_anchored_subject_over_backdrop() {
        // 128×128, box short side 64px → anchored (threshold 2·2^2·8 = 64).
        let plan = LayerPlan { layers: vec![anchored_layer("s", [0.25, 0.25, 0.75, 0.75], 0.3)], ..Default::default() };
        let drafts = DraftSet { backdrop: solid(128, 128, [0, 0, 255]), layers: vec![("s".into(), solid(64, 64, [255, 0, 0]))] };
        let (canvas, placed) = compose(&plan, &drafts, &geom(), 128, 128, &bare(), full_matte).unwrap();
        assert_eq!(placed.len(), 1, "one anchored subject placed");
        // Centre is inside the box → the red subject (full matte; luma-normalised, so red-dominant not
        // necessarily 255), corner stays backdrop blue.
        let centre = canvas.get_pixel(64, 64).0;
        assert!(centre[0] > 100 && centre[0] > centre[2], "red subject painted at centre: {centre:?}");
        let corner = canvas.get_pixel(2, 2).0;
        assert_eq!(corner, [0, 0, 255], "backdrop untouched outside the box");
    }

    #[test]
    fn compose_skips_non_anchored() {
        // A tiny box (short side < u = 8px) → Lifted, so it must NOT be composed.
        let plan = LayerPlan { layers: vec![anchored_layer("tiny", [0.4, 0.4, 0.44, 0.44], 0.3)], ..Default::default() };
        let drafts = DraftSet { backdrop: solid(128, 128, [10, 20, 30]), layers: vec![("tiny".into(), solid(8, 8, [255, 0, 0]))] };
        let (canvas, placed) = compose(&plan, &drafts, &geom(), 128, 128, &bare(), full_matte).unwrap();
        assert!(placed.is_empty(), "lifted layer is not painted into the guide");
        // Every pixel is still the backdrop.
        assert!(canvas.pixels().all(|p| p.0 == [10, 20, 30]));
    }

    #[test]
    fn luma_normalises_dark_subject_toward_backdrop() {
        // Bright backdrop, very dark subject → the composited subject is lifted (gain > 1).
        let plan = LayerPlan { layers: vec![anchored_layer("s", [0.25, 0.25, 0.75, 0.75], 0.3)], ..Default::default() };
        let drafts = DraftSet { backdrop: solid(128, 128, [200, 200, 200]), layers: vec![("s".into(), solid(64, 64, [40, 40, 40]))] };
        let (canvas, _) = compose(&plan, &drafts, &geom(), 128, 128, &bare(), full_matte).unwrap();
        let centre = canvas.get_pixel(64, 64).0[0];
        assert!(centre > 40, "dark subject brightened toward the backdrop (got {centre})");
    }

    #[test]
    fn nearer_layer_wins_overlap() {
        // Two overlapping anchored boxes; the nearer (smaller depth) is painted last → wins the centre.
        let far = Layer { id: "far".into(), bbox: Some([0.2, 0.2, 0.8, 0.8]), depth: Some(0.8), ..Default::default() };
        let near = Layer { id: "near".into(), bbox: Some([0.25, 0.25, 0.75, 0.75]), depth: Some(0.1), ..Default::default() };
        let plan = LayerPlan { layers: vec![far, near], ..Default::default() };
        let drafts = DraftSet {
            backdrop: solid(128, 128, [0, 0, 0]),
            layers: vec![("far".into(), solid(76, 76, [0, 255, 0])), ("near".into(), solid(64, 64, [255, 0, 0]))],
        };
        let (canvas, placed) = compose(&plan, &drafts, &geom(), 128, 128, &bare(), full_matte).unwrap();
        assert_eq!(placed.len(), 2);
        assert_eq!(placed[0].0, "far", "far painted first");
        let centre = canvas.get_pixel(64, 64).0;
        assert!(centre[0] > 100 && centre[0] > centre[1], "nearer (red) subject wins the overlap: {centre:?}");
    }

    #[test]
    fn maps_raise_weight_inside_and_downsample_to_latent_res() {
        let plan = LayerPlan {
            backdrop: Some(Backdrop { prompt: None, weight: Some(0.5), window: Some(0.2) }),
            layers: vec![anchored_layer("s", [0.25, 0.25, 0.75, 0.75], 0.3)],
            ..Default::default()
        };
        let sil = {
            let mut g = GrayImage::new(128, 128);
            for y in 32..96 {
                for x in 32..96 {
                    g.put_pixel(x, y, Luma([255]));
                }
            }
            g
        };
        let (w, e) = build_maps(&plan, &[("s".into(), sil)], &geom(), 128, 128, &Device::Cpu).unwrap();
        // Latent res = 128 / 8 = 16.
        assert_eq!(w.dims4().unwrap(), (1, 1, 16, 16));
        assert_eq!(e.dims4().unwrap(), (1, 1, 16, 16));
        // Centre latent cell (inside the silhouette) → the layer's weight/window.
        let wc: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        let ec: Vec<f32> = e.flatten_all().unwrap().to_vec1().unwrap();
        let centre = 8 * 16 + 8;
        assert!(wc[centre] > 0.8, "inside → layer weight ~0.85 (got {})", wc[centre]);
        assert!(ec[centre] > 0.55, "inside → layer window ~0.6 (got {})", ec[centre]);
        // A corner cell (outside) → the backdrop weight/window.
        assert!((wc[0] - 0.5).abs() < 0.05, "outside → backdrop weight 0.5 (got {})", wc[0]);
        assert!((ec[0] - 0.2).abs() < 0.05, "outside → backdrop window 0.2 (got {})", ec[0]);
    }

    #[test]
    fn harmonize_shifts_subject_toward_backdrop() {
        // A pure-red subject harmonised toward a blue backdrop: red falls, blue rises, mean moves toward blue.
        let mut sub = solid(16, 16, [220, 20, 20]);
        harmonize_toward(&mut sub, [20.0, 20.0, 220.0], 0.5);
        let m = mean_rgb(&sub);
        assert!(m[0] < 220.0 && m[2] > 20.0, "moved toward backdrop colour: {m:?}");
        assert!((m[0] - 120.0).abs() < 2.0 && (m[2] - 120.0).abs() < 2.0, "≈ halfway at strength 0.5: {m:?}");
    }

    #[test]
    fn directional_shade_brightens_toward_the_light() {
        // Light at 90° = overhead → the top of the subject is brighter than the bottom.
        let mut sub = solid(16, 16, [120, 120, 120]);
        directional_shade(&mut sub, 90.0, 0.3);
        let top = sub.get_pixel(8, 1).0[0];
        let bot = sub.get_pixel(8, 14).0[0];
        assert!(top > bot, "top (toward the overhead light) brighter than bottom ({top} vs {bot})");
    }

    #[test]
    fn grounding_darkens_beneath_the_subject() {
        // A subject occupying the UPPER part of its box (feet mid-box), so there is ground below it for the
        // contact shadow to pool onto. With grounding on, that ground is darker than with grounding off.
        let plan = LayerPlan { layers: vec![anchored_layer("s", [0.25, 0.2, 0.75, 0.8], 0.3)], ..Default::default() };
        let drafts = DraftSet { backdrop: solid(128, 128, [180, 180, 180]), layers: vec![("s".into(), solid(64, 76, [180, 180, 180]))] };
        // Matte: subject in the top ~55% of its box; the rest is ground.
        let upper_matte = |img: &RgbImage| -> Result<GrayImage> {
            let (w, h) = (img.width(), img.height());
            let mut a = GrayImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    a.put_pixel(x, y, Luma([if y < h * 55 / 100 { 255 } else { 0 }]));
                }
            }
            Ok(a)
        };
        let grounded = GuideOpts { ground: true, harmonize: 0.0, relight_amp: 0.0, light_angle: 90.0, ground_softness: 1.0 };
        let (c_on, _) = compose(&plan, &drafts, &geom(), 128, 128, &grounded, upper_matte).unwrap();
        let (c_off, _) = compose(&plan, &drafts, &geom(), 128, 128, &bare(), upper_matte).unwrap();
        // Box is y∈[0.2,0.8]·128 = [26,102]; feet ≈ row 41 (canvas y≈67). The soft shadow pools just below —
        // take the darkest ground pixel in the band right under the feet.
        let x = 64u32;
        let on = (68u32..76).map(|y| c_on.get_pixel(x, y).0[0] as i32).min().unwrap();
        let off = c_off.get_pixel(x, 70).0[0] as i32;
        assert!(on < off, "grounding darkens the ground below the subject ({on} < {off})");
    }
}
