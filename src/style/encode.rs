//! Encode a reference photo into a CLIP-H pooled image embedding.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::pipelines::ip_adapter::ImageEncoder;

/// Encode `photo` through `encoder` into the L2-normalized embedding the
/// catalog uses for cosine matching.
///
/// Returns shape `(1024,)` f32 — pass directly into
/// [`super::detect::detect_style`].
///
/// `encoder` is borrowed (not loaded inline) so a single runtime can
/// reuse it across multiple photos in scenarios without reloading
/// CLIP-H every task.
pub fn encode_reference_photo(
    encoder: &ImageEncoder,
    photo: &Path,
    device: &Device,
) -> Result<Tensor> {
    let pixels = crate::imaging::preprocess::clip_image_tensor(photo, 224, device, DType::F32)
        .with_context(|| format!("preprocessing {}", photo.display()))?;

    let pooled = encoder.encode(&pixels)?; // (1, 1024) f32
    let v = pooled.squeeze(0)?; // (1024,)

    // L2-normalize. Scalar norm has shape (), broadcast_div needs a same-rank
    // operand, so reshape to (1,) before dividing the (1024,) vector.
    let norm = v.sqr()?.sum_all()?.sqrt()?;
    let normed = v.broadcast_div(&norm.reshape((1,))?)?;
    Ok(normed)
}
