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
}
