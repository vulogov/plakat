//! Auto-annotators that turn an ordinary source image into the
//! conditioning tensor a [`ControlNet`] expects.
//!
//! v0.9 required the user to supply a pre-rendered conditioning
//! image (a depth map, edge map, etc.) via `--control-image PATH`.
//! v0.10 adds the convenience path: `--control-from PATH` runs the
//! appropriate annotator for the selected `--control` kind and
//! produces the conditioning tensor on the fly. The annotator
//! output shape matches [`prepare_conditioning`] — a
//! `(1, 3, H, W)` tensor at the pipeline's dtype, RGB-normalised
//! to `[0, 1]`.
//!
//! ## Kinds shipped in v0.10
//!
//! * **Depth** — reuses the [`DepthPipeline`] that already powers
//!   v0.7 smart-zones. Depth-Anything-V2-small produces a per-pixel
//!   relative-depth map (`[0, 1]`, larger = closer); we replicate
//!   the single channel across R/G/B since the ControlNet-Depth
//!   model is trained on RGB depth visualisations.
//!
//! Phase 3 will add **Canny** via the `imageproc` crate. The
//! `annotate` dispatch is a `match` over the kind enum, so adding
//! the canny arm is a one-line change once the canny annotator
//! function exists.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::pipelines::controlnet::ControlKind;
use crate::pipelines::depth::DepthPipeline;

/// Default Canny low-threshold (8-bit luminance). Same as diffusers'
/// CannyDetector default.
const CANNY_LOW: f32 = 100.0;
/// Default Canny high-threshold.
const CANNY_HIGH: f32 = 200.0;

/// Run the matching annotator for `kind` on the source image, then
/// pack the result into a `(1, 3, H, W)` conditioning tensor at
/// `dtype`. The result is drop-in compatible with what
/// [`crate::pipelines::controlnet::prepare_conditioning`] produces
/// for a user-supplied control image.
///
/// This is an `async fn` because depth annotation entails a one-off
/// model download (the Depth-Anything-V2-small safetensors,
/// ~99 MB) on cold cache; the function awaits that.
pub async fn annotate(
    kind: ControlKind,
    src_path: &Path,
    out_w: u32,
    out_h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    match kind {
        ControlKind::Depth => annotate_depth(src_path, out_w, out_h, device, dtype).await,
        ControlKind::Canny => annotate_canny(src_path, out_w, out_h, device, dtype),
        ControlKind::SoftEdge => annotate_softedge(src_path, out_w, out_h, device, dtype).await,
        ControlKind::Lineart => annotate_lineart(src_path, out_w, out_h, device, dtype).await,
        ControlKind::OpenPose => annotate_openpose(src_path, out_w, out_h, device, dtype).await,
        // Tile: the conditioning hint IS the (blurry) image — no annotator, just the identity
        // resize `prepare_conditioning` performs.
        ControlKind::Tile => {
            crate::pipelines::controlnet::prepare_conditioning(src_path, out_w, out_h, device, dtype)
        }
    }
}

/// Run Depth-Anything-V2-small on `src_path` and produce a
/// `(1, 3, H, W)` tensor with the depth map replicated across each
/// channel.
///
/// Convention matches ControlNet-Depth training data: brighter
/// pixels = closer to the camera, darker = farther. Depth-
/// Anything-V2 already emits inverse-relative depth in that
/// orientation after the head's ReLU + our min-max normalisation,
/// so no further inversion is needed.
async fn annotate_depth(
    src_path: &Path,
    out_w: u32,
    out_h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let pipeline = DepthPipeline::load(device.clone())
        .await
        .context("loading Depth-Anything-V2 for --control-from")?;
    let depth = pipeline
        .depth_map(src_path, out_w, out_h)
        .with_context(|| format!("running depth on {}", src_path.display()))?;
    if depth.len() != (out_w as usize) * (out_h as usize) {
        anyhow::bail!(
            "depth map size mismatch: expected {}x{} = {} pixels, got {}",
            out_w,
            out_h,
            (out_w as usize) * (out_h as usize),
            depth.len(),
        );
    }
    depth_to_rgb_tensor(&depth, out_w, out_h, device, dtype)
}

/// Run Canny edge detection on `src_path` and produce a
/// `(1, 3, H, W)` tensor with the binary edge image replicated
/// across each channel.
///
/// Output convention matches ControlNet-Canny training data: white
/// pixels = edges, black = background.
///
/// Synchronous (no model download — pure CPU image processing via
/// the `imageproc` crate). Resize happens *after* edge detection so
/// the edges are computed at the source resolution and then sampled
/// down, preserving thin-line sharpness better than the reverse.
fn annotate_canny(
    src_path: &Path,
    out_w: u32,
    out_h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let src = image::open(src_path)
        .with_context(|| format!("opening Canny source {}", src_path.display()))?;
    let gray = src.to_luma8();
    let edges = imageproc::edges::canny(&gray, CANNY_LOW, CANNY_HIGH);
    // Resize edge map to the generation resolution. Nearest-neighbour
    // would preserve hard edges best, but triangle is what every other
    // plakat path uses (consistency wins over a marginal sharpness
    // gain).
    let resized = image::imageops::resize(
        &edges,
        out_w,
        out_h,
        image::imageops::FilterType::Triangle,
    );
    let total = (out_w as usize) * (out_h as usize);
    if resized.as_raw().len() != total {
        anyhow::bail!(
            "canny edge map size mismatch: expected {} pixels, got {}",
            total,
            resized.as_raw().len()
        );
    }
    let depth: Vec<f32> = resized.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
    depth_to_rgb_tensor(&depth, out_w, out_h, device, dtype)
}

// =====================================================================
// SoftEdge / HED annotator (v0.11)
// =====================================================================

const HED_REPO: &str = "lllyasviel/Annotators";
const HED_FILE: &str = "ControlNetHED.pth";

/// Detection resolution used by lllyasviel's reference annotators —
/// HED runs at 512 px long-edge regardless of the final ControlNet
/// input size. We mirror that: edge map is computed at 512, then
/// triangle-resized to (out_w, out_h).
const HED_DETECT_RES: u32 = 512;

/// Run the HED softedge model on `src_path` and pack the result into
/// a `(1, 3, H, W)` ControlNet conditioning tensor.
///
/// First-run cost: downloads `lllyasviel/Annotators/ControlNetHED.pth`
/// (~30 MB) into the HF cache.
async fn annotate_softedge(
    src_path: &Path,
    out_w: u32,
    out_h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    use candle_nn::VarBuilder;

    let weights = crate::hf::download::get_file(HED_REPO, HED_FILE)
        .await
        .with_context(|| format!("downloading HED weights ({HED_REPO}/{HED_FILE})"))?;
    // HED runs at F32 — the model is small (~30 MB) and the precision
    // matters for the sigmoid-averaged edge probability map.
    let vb = VarBuilder::from_pth(&weights, DType::F32, device)?;
    let model = crate::pipelines::hed::HedModel::new(vb)
        .context("loading HED weights")?;

    // Read the source image, resize so the long edge is HED_DETECT_RES,
    // then snap dims to a multiple of 8 (HED's 4 down-pools want the
    // input to divide evenly enough to keep the upsample math clean).
    let src = image::open(src_path)
        .with_context(|| format!("opening softedge source {}", src_path.display()))?;
    let rgb = src.to_rgb8();
    let (src_w, src_h) = (rgb.width(), rgb.height());
    let scale = HED_DETECT_RES as f32 / src_w.max(src_h) as f32;
    let det_w = ((src_w as f32 * scale).round() as u32).max(64) & !7;
    let det_h = ((src_h as f32 * scale).round() as u32).max(64) & !7;
    let resized_in = image::imageops::resize(
        &rgb,
        det_w,
        det_h,
        image::imageops::FilterType::Triangle,
    );

    // (1, 3, H, W) f32 in [0, 255]. HED expects raw pixel values; the
    // learnt `norm` parameter subtracts the mean.
    let h = det_h as usize;
    let w = det_w as usize;
    let mut buf: Vec<f32> = Vec::with_capacity(3 * h * w);
    // CHW order: channel R first across all pixels, then G, then B.
    for c in 0..3 {
        for y in 0..det_h {
            for x in 0..det_w {
                let px = resized_in.get_pixel(x, y);
                buf.push(px[c] as f32);
            }
        }
    }
    let x = Tensor::from_vec(buf, (1, 3, h, w), device)?;

    let side_outputs = model
        .forward(&x)
        .context("HED forward")?;

    // Upsample each side output to the detect resolution, sum, divide
    // by count, sigmoid. `interpolate2d` is nearest neighbour in
    // candle 0.8 — slightly blockier than diffusers' bilinear resize
    // but adequate for ControlNet conditioning at the resolutions
    // SD/SDXL operates on.
    let mut sum = Tensor::zeros((1, 1, h, w), DType::F32, device)?;
    for e in &side_outputs {
        let up = e.interpolate2d(h, w)?;
        sum = (sum + up)?;
    }
    let avg = (sum / side_outputs.len() as f64)?;
    let sigmoid = candle_nn::ops::sigmoid(&avg)?;

    // Pull to host as a single-channel u8 grayscale image, then resize
    // to the requested (out_w, out_h) and replicate to 3 channels.
    let edge_vals: Vec<f32> = sigmoid
        .squeeze(0)?
        .squeeze(0)?
        .flatten_all()?
        .to_vec1()?;
    let mut gray = image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::new(det_w, det_h);
    for (i, p) in edge_vals.iter().enumerate() {
        let y = (i / w) as u32;
        let xp = (i % w) as u32;
        let v = (p.clamp(0.0, 1.0) * 255.0).round() as u8;
        gray.put_pixel(xp, y, image::Luma([v]));
    }
    let resized_out = image::imageops::resize(
        &gray,
        out_w,
        out_h,
        image::imageops::FilterType::Triangle,
    );
    let final_buf: Vec<f32> = resized_out
        .as_raw()
        .iter()
        .map(|&v| v as f32 / 255.0)
        .collect();
    depth_to_rgb_tensor(&final_buf, out_w, out_h, device, dtype)
}

// =====================================================================
// OpenPose annotator (v0.11)
// =====================================================================

const OPENPOSE_REPO: &str = "lllyasviel/Annotators";
const OPENPOSE_FILE: &str = "body_pose_model.pth";

/// OpenPose runs at 368 px short-edge (the boxsize the original CMU
/// network was trained at). Stride is 8 — the model outputs heatmaps
/// and PAFs at 1/8 the input resolution.
const OPENPOSE_BOXSIZE: u32 = 368;
const OPENPOSE_STRIDE: usize = 8;

/// Run OpenPose body-pose detection on `src_path` and pack a
/// coloured-skeleton-on-black RGB image into a `(1, 3, H, W)`
/// ControlNet conditioning tensor.
///
/// First-run cost: downloads
/// `lllyasviel/Annotators/body_pose_model.pth` (~205 MB) into the HF
/// cache.
///
/// Simplifications relative to lllyasviel's reference implementation
/// (documented in `pipelines::openpose_post`):
///   * Single-scale forward (no scale-search pyramid).
///   * Raw-heatmap NMS (no Gaussian smoothing).
///   * Greedy bipartite matching for limb assembly (no Hungarian).
///
/// Quality is adequate for ControlNet conditioning, but detection
/// reliability is below lllyasviel's reference, especially on small
/// or partially-occluded figures.
async fn annotate_openpose(
    src_path: &Path,
    out_w: u32,
    out_h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    use candle_nn::VarBuilder;

    let weights = crate::hf::download::get_file(OPENPOSE_REPO, OPENPOSE_FILE)
        .await
        .with_context(|| {
            format!("downloading OpenPose weights ({OPENPOSE_REPO}/{OPENPOSE_FILE})")
        })?;
    let vb = VarBuilder::from_pth(&weights, DType::F32, device)?;
    let model = crate::pipelines::openpose::BodyPoseModel::new(vb)
        .context("loading OpenPose weights")?;

    let src = image::open(src_path)
        .with_context(|| format!("opening openpose source {}", src_path.display()))?;
    let rgb = src.to_rgb8();
    let (src_w, src_h) = (rgb.width(), rgb.height());

    // Scale so the short edge is OPENPOSE_BOXSIZE (368), then snap
    // both dims to a multiple of the stride (8) so the heatmap math
    // is exact.
    let scale = OPENPOSE_BOXSIZE as f32 / src_w.min(src_h) as f32;
    let det_w = ((src_w as f32 * scale).round() as u32).max(64);
    let det_h = ((src_h as f32 * scale).round() as u32).max(64);
    let det_w = det_w.div_ceil(OPENPOSE_STRIDE as u32) * OPENPOSE_STRIDE as u32;
    let det_h = det_h.div_ceil(OPENPOSE_STRIDE as u32) * OPENPOSE_STRIDE as u32;
    let resized_in = image::imageops::resize(
        &rgb,
        det_w,
        det_h,
        image::imageops::FilterType::Triangle,
    );

    // (1, 3, H, W) f32 in [-0.5, 0.5] — matches lllyasviel's
    // `data = data.float() / 256 - 0.5` normalisation.
    let h = det_h as usize;
    let w = det_w as usize;
    let mut buf: Vec<f32> = Vec::with_capacity(3 * h * w);
    for c in 0..3 {
        for y in 0..det_h {
            for x in 0..det_w {
                let px = resized_in.get_pixel(x, y);
                buf.push((px[c] as f32) / 256.0 - 0.5);
            }
        }
    }
    let x = Tensor::from_vec(buf, (1, 3, h, w), device)?;

    let (paf, heatmap) = model.forward(&x).context("OpenPose forward")?;
    // PAFs and heatmaps come out at 1/stride spatial resolution.
    let map_h = h / OPENPOSE_STRIDE;
    let map_w = w / OPENPOSE_STRIDE;
    let paf_v: Vec<f32> = paf.squeeze(0)?.flatten_all()?.to_vec1()?;
    let hm_v: Vec<f32> = heatmap.squeeze(0)?.flatten_all()?.to_vec1()?;
    if paf_v.len() != 38 * map_h * map_w {
        anyhow::bail!(
            "OpenPose PAF tensor shape mismatch: expected 38×{map_h}×{map_w} = {}, got {}",
            38 * map_h * map_w,
            paf_v.len()
        );
    }
    if hm_v.len() != 19 * map_h * map_w {
        anyhow::bail!(
            "OpenPose heatmap tensor shape mismatch: expected 19×{map_h}×{map_w} = {}, got {}",
            19 * map_h * map_w,
            hm_v.len()
        );
    }

    // Render skeleton at the detect-image resolution, then resize
    // to the caller's requested (out_w, out_h).
    let skel = crate::pipelines::openpose_post::render_skeleton(
        &hm_v,
        &paf_v,
        map_h,
        map_w,
        det_w,
        det_h,
        OPENPOSE_STRIDE,
    )?;
    let resized_skel = image::imageops::resize(
        &skel,
        out_w,
        out_h,
        image::imageops::FilterType::Triangle,
    );

    // Pack into (1, 3, H, W) f32 in [0, 1]. The RGB skeleton image
    // already carries colour information per limb; ControlNet-OpenPose
    // is trained on coloured skeletons, so we keep the channels
    // separate (unlike depth/canny/etc. that replicate a single
    // grayscale value).
    let total = (out_w as usize) * (out_h as usize);
    let mut chw: Vec<f32> = Vec::with_capacity(3 * total);
    for c in 0..3 {
        for y in 0..out_h {
            for x in 0..out_w {
                let px = resized_skel.get_pixel(x, y);
                chw.push(px[c] as f32 / 255.0);
            }
        }
    }
    let t = Tensor::from_vec(chw, (1, 3, out_h as usize, out_w as usize), device)?
        .to_dtype(dtype)?;
    Ok(t)
}

// =====================================================================
// Lineart annotator (v0.11)
// =====================================================================

const LINEART_REPO: &str = "lllyasviel/Annotators";
const LINEART_FILE: &str = "sk_model.pth";

/// Lineart runs at 512 px long-edge to match lllyasviel's reference
/// annotator default.
const LINEART_DETECT_RES: u32 = 512;

/// Run the lineart generator on `src_path` and pack the result into
/// a `(1, 3, H, W)` ControlNet conditioning tensor.
///
/// First-run cost: downloads `lllyasviel/Annotators/sk_model.pth`
/// (~110 MB) into the HF cache.
///
/// ControlNet input convention for lineart is "bright lines on a dark
/// background" — same orientation the model emits via its sigmoid head
/// (model probability map → bright pixel = line). We do NOT invert
/// the output. (lllyasviel's reference inverts in some pipelines but
/// the `control_v11p_sd15_lineart` ControlNet expects the non-inverted
/// orientation.)
async fn annotate_lineart(
    src_path: &Path,
    out_w: u32,
    out_h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    use candle_nn::VarBuilder;

    let weights = crate::hf::download::get_file(LINEART_REPO, LINEART_FILE)
        .await
        .with_context(|| {
            format!("downloading Lineart weights ({LINEART_REPO}/{LINEART_FILE})")
        })?;
    let vb = VarBuilder::from_pth(&weights, DType::F32, device)?;
    let model = crate::pipelines::lineart::LineartModel::new(vb)
        .context("loading Lineart weights")?;

    let src = image::open(src_path)
        .with_context(|| format!("opening lineart source {}", src_path.display()))?;
    let rgb = src.to_rgb8();
    let (src_w, src_h) = (rgb.width(), rgb.height());
    let scale = LINEART_DETECT_RES as f32 / src_w.max(src_h) as f32;
    // Snap to a multiple of 8 — the down→up structure rounds spatial
    // dims; staying on a /8 grid keeps the output exact.
    let det_w = ((src_w as f32 * scale).round() as u32).max(64) & !7;
    let det_h = ((src_h as f32 * scale).round() as u32).max(64) & !7;
    let resized_in = image::imageops::resize(
        &rgb,
        det_w,
        det_h,
        image::imageops::FilterType::Triangle,
    );

    // (1, 3, H, W) f32 in [0, 1] — the lineart reference divides by 255.
    let h = det_h as usize;
    let w = det_w as usize;
    let mut buf: Vec<f32> = Vec::with_capacity(3 * h * w);
    for c in 0..3 {
        for y in 0..det_h {
            for x in 0..det_w {
                let px = resized_in.get_pixel(x, y);
                buf.push(px[c] as f32 / 255.0);
            }
        }
    }
    let x = Tensor::from_vec(buf, (1, 3, h, w), device)?;

    let line = model.forward(&x).context("Lineart forward")?;
    // line: (1, 1, H, W) in [0, 1]. Pull to host.
    let line_vals: Vec<f32> = line
        .squeeze(0)?
        .squeeze(0)?
        .flatten_all()?
        .to_vec1()?;
    let mut gray = image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::new(det_w, det_h);
    for (i, p) in line_vals.iter().enumerate() {
        let y = (i / w) as u32;
        let xp = (i % w) as u32;
        let v = (p.clamp(0.0, 1.0) * 255.0).round() as u8;
        gray.put_pixel(xp, y, image::Luma([v]));
    }
    let resized_out = image::imageops::resize(
        &gray,
        out_w,
        out_h,
        image::imageops::FilterType::Triangle,
    );
    let final_buf: Vec<f32> = resized_out
        .as_raw()
        .iter()
        .map(|&v| v as f32 / 255.0)
        .collect();
    depth_to_rgb_tensor(&final_buf, out_w, out_h, device, dtype)
}

/// Pack a row-major `[0, 1]` depth buffer into a `(1, 3, H, W)`
/// tensor by replicating the single channel across R/G/B. The
/// resulting tensor matches what
/// [`crate::pipelines::controlnet::prepare_conditioning`] emits for
/// a grayscale source image.
fn depth_to_rgb_tensor(
    depth: &[f32],
    w: u32,
    h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total = (w as usize) * (h as usize);
    let mut buf: Vec<f32> = Vec::with_capacity(3 * total);
    // R, G, B channels — same data, three copies.
    for _ in 0..3 {
        buf.extend_from_slice(depth);
    }
    let t = Tensor::from_vec(buf, (1, 3, h as usize, w as usize), device)?.to_dtype(dtype)?;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_to_rgb_tensor_replicates_channels() {
        let depth = vec![0.0, 0.5, 1.0, 0.25];
        let t = depth_to_rgb_tensor(&depth, 2, 2, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 2, 2]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // Three identical channels: indices [0..4], [4..8], [8..12]
        // should all be [0.0, 0.5, 1.0, 0.25].
        for ch in 0..3 {
            let base = ch * 4;
            assert!((v[base] - 0.0).abs() < 1e-6);
            assert!((v[base + 1] - 0.5).abs() < 1e-6);
            assert!((v[base + 2] - 1.0).abs() < 1e-6);
            assert!((v[base + 3] - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn depth_to_rgb_tensor_preserves_dtype() {
        let depth = vec![0.5_f32; 64];
        let t = depth_to_rgb_tensor(&depth, 8, 8, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.dims(), &[1, 3, 8, 8]);
    }

    /// Canny on a black image with a single white square should
    /// produce edge pixels along the square's border, zero
    /// elsewhere. Verifies imageproc integration + dispatch.
    #[test]
    fn annotate_canny_detects_square_edges() {
        use image::{Luma, Rgb, RgbImage};
        // 32×32 black image with a 12×12 white square in the centre.
        let mut img = RgbImage::from_pixel(32, 32, Rgb([0u8, 0, 0]));
        for y in 10..22 {
            for x in 10..22 {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        let tmp = std::env::temp_dir().join("plakat_canny_test.png");
        img.save(&tmp).unwrap();

        let t = annotate_canny(&tmp, 32, 32, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 32, 32]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        let total = 32 * 32;
        let r_channel = &v[..total];
        // There should be at least a few edge pixels (>= 4 sides ×
        // ~12 pixels / sampling = dozens) where R is bright.
        let bright_count = r_channel.iter().filter(|&&x| x > 0.5).count();
        assert!(
            bright_count > 20,
            "expected canny to find edges, got {bright_count} bright pixels"
        );
        // Far corners (0,0) and (31,31) should be black (no edge).
        assert!(v[0] < 0.05);
        assert!(v[total - 1] < 0.05);
        // Also: ignore unused import warning for Luma if it warns.
        let _ = Luma::<u8>([0]);
    }
}
