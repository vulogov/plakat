//! SDXL CLIP-G wrapper with `text_projection` + EOT-token pooling
//! (phase 8b).
//!
//! candle's `ClipTextTransformer` loads `text_model.embeddings`,
//! `text_model.encoder`, and `text_model.final_layer_norm` — but it
//! stops there. SDXL's CLIP-G text encoder (a.k.a. text_encoder_2)
//! additionally carries a top-level `text_projection` Linear that
//! diffusers applies to the EOT-token row of the final hidden state to
//! produce the pooled text embedding that feeds the UNet's
//! `add_embedding` projection.
//!
//! We can't subclass candle's type, so this module owns *both*: an
//! inner `ClipTextTransformer` (re-used unchanged for the per-token
//! hidden states we still need for SDXL's dual-encoder cross-attention)
//! plus the `text_projection` Linear loaded from the same VarBuilder
//! root. One forward call produces both outputs without re-running the
//! encoder twice.
//!
//! Used only for SDXL (base + refiner). SD 1.5 / SD 2.1 keep their
//! existing single-encoder CLIP path — they have no pooled output to
//! consume.
//!
//! v0.30 phase 0: the inner CLIP is now plakat's vendored CLIP
//! (`pipelines::vendored_clip`). The wrapper API is unchanged; the
//! swap is transparent to callers. AnimateDiff and SdCore both use
//! this wrapper for SDXL CLIP-G.

use candle_core::{D, Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;
use crate::pipelines::vendored_clip::{ClipTextTransformer, Config};

/// SDXL CLIP-G text encoder = candle's CLIP + a top-level
/// `text_projection` Linear (no bias) for the pooled output path.
#[derive(Debug)]
pub struct SdxlClipGTextTransformer {
    inner: ClipTextTransformer,
    /// `(embed_dim → projection_dim)` Linear. For stock SDXL CLIP-G
    /// both dims are 1280 and the weight has no bias.
    text_projection: nn::Linear,
}

impl SdxlClipGTextTransformer {
    /// Build the wrapper. `vs` is the **root** of the text encoder's
    /// safetensors — the dir/file under which `text_model.*` sits;
    /// `text_projection.weight` is a sibling of `text_model`.
    /// `embed_dim` should match `Config::embed_dim` — 1280 for stock
    /// SDXL CLIP-G. Kept as an explicit arg to preserve the original
    /// signature (callers used to need it because candle's
    /// `Config::embed_dim` was private; on the vendored Config it's
    /// public but we still take it explicitly for symmetry).
    pub fn new(vs: nn::VarBuilder, c: &Config, embed_dim: usize) -> Result<Self> {
        let inner = ClipTextTransformer::new(vs.clone(), c)?;
        // Diffusers' CLIP-G ships text_projection as a square Linear
        // without bias. We use `linear_no_bias` so VarBuilder doesn't
        // fail looking for a `bias` key that isn't there.
        let text_projection =
            nn::linear_no_bias(embed_dim, embed_dim, vs.pp("text_projection"))?;
        Ok(Self {
            inner,
            text_projection,
        })
    }

    /// Cheap accessor for callers that only need the per-token
    /// hidden states (e.g. the cross-attn path). Pre-existing
    /// behaviour — same as calling `inner.forward_until_encoder_layer`.
    pub fn forward_until_encoder_layer(
        &self,
        ids: &Tensor,
        mask_after: usize,
        until_layer: isize,
    ) -> Result<(Tensor, Tensor)> {
        self.inner.forward_until_encoder_layer(ids, mask_after, until_layer)
    }

    /// SDXL's combined output: the penultimate hidden state (used by
    /// the UNet's dual-encoder cross-attention path) **and** the
    /// pooled text embedding (projected EOT-token row of the final
    /// hidden state, used by the UNet's `add_embedding`).
    ///
    /// Returns `(penultimate_hidden, pooled)`:
    ///   * `penultimate_hidden` — `(B, seq_len, embed_dim)`, the
    ///     activation right after the `(num_layers - 1)`th encoder
    ///     layer (before `final_layer_norm`). Matches diffusers'
    ///     `hidden_states[-2]`.
    ///   * `pooled` — `(B, projection_dim)`, the projected EOT row
    ///     of `final_layer_norm(...)`. EOT location is read from the
    ///     input id with the highest value, mirroring diffusers'
    ///     `input_ids.argmax(-1)` convention (the CLIP tokenizer
    ///     places its EOS token at the top of the vocab).
    pub fn forward_for_sdxl(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (final_hidden, penultimate) =
            self.inner.forward_until_encoder_layer(ids, usize::MAX, -2)?;
        let (b, _seq_len) = ids.dims2()?;
        let argmax = ids.argmax(D::Minus1)?;
        // Pull each batch's EOT row out one at a time. B is 1 or 2
        // (CFG), so the tiny loop here is dwarfed by everything else.
        let argmax_v: Vec<u32> = argmax.to_dtype(candle_core::DType::U32)?.to_vec1()?;
        let mut rows = Vec::with_capacity(b);
        for (bi, &idx) in argmax_v.iter().enumerate() {
            rows.push(final_hidden.i((bi, idx as usize))?);
        }
        let pooled = Tensor::stack(&rows, 0)?;
        let pooled = self.text_projection.forward(&pooled)?;
        Ok((penultimate, pooled))
    }
}

// Local import alias — keeps the `i(...)` indexing call above terse.
use candle_core::IndexOp;
