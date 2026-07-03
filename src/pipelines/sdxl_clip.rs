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

use candle_core::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;
use crate::pipelines::vendored_clip::{ClipTextTransformer, Config};

/// CLIP end-of-text token id — the row diffusers pools for the SDXL/Cascade `add_embedding`.
const CLIP_EOS_ID: u32 = 49407;

/// Per-batch pooling row: the FIRST EOS(49407) position (diffusers' post-added-token
/// convention). A plain `argmax(-1)` assumes EOS is the highest id in the vocab — but SDXL
/// dual-encoder Textual Inversion appends trigger tokens at ids > EOS, so a prompt with a
/// TI trigger would make `argmax` select the trigger's position and pool the WRONG row.
/// Falls back to the highest-id token only when no EOS is present (unexpected).
fn eot_rows(ids: &Tensor) -> Result<Vec<usize>> {
    let rows: Vec<Vec<u32>> = ids.to_dtype(candle_core::DType::U32)?.to_vec2()?;
    Ok(rows
        .iter()
        .map(|row| {
            row.iter().position(|&t| t == CLIP_EOS_ID).unwrap_or_else(|| {
                row.iter().enumerate().max_by_key(|(_, v)| **v).map(|(i, _)| i).unwrap_or(0)
            })
        })
        .collect())
}

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
        // EOT row via the explicit EOS id (TI-safe — see `eot_rows`). B is 1 or 2 (CFG).
        let eot = eot_rows(ids)?;
        let mut rows = Vec::with_capacity(b);
        for (bi, &idx) in eot.iter().enumerate() {
            rows.push(final_hidden.i((bi, idx))?);
        }
        let pooled = Tensor::stack(&rows, 0)?;
        let pooled = self.text_projection.forward(&pooled)?;
        Ok((penultimate, pooled))
    }

    /// Token-embedding lookup only (no position embedding) — the SDXL
    /// Textual-Inversion training splice point for the CLIP-G half. Pass-through
    /// to the inner vendored CLIP's [`ClipTextTransformer::embed_tokens`].
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.inner.embed_tokens(token_ids)
    }

    /// From-embeds counterpart of [`Self::forward_for_sdxl`] for TI training:
    /// the trainable placeholder vector is spliced into `token_embeds` before
    /// the encoder runs. `ids` (the original token ids) locates the EOT row for
    /// pooling via `argmax`, mirroring `forward_for_sdxl`. Returns
    /// `(penultimate_hidden, pooled)` — bit-identical assembly to the id path.
    pub fn forward_for_sdxl_from_embeds(
        &self,
        token_embeds: &Tensor,
        ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (final_hidden, penultimate) = self
            .inner
            .forward_until_encoder_layer_from_embeds(token_embeds, usize::MAX, -2)?;
        let (b, _seq_len) = ids.dims2()?;
        let eot = eot_rows(ids)?;
        let mut rows = Vec::with_capacity(b);
        for (bi, &idx) in eot.iter().enumerate() {
            rows.push(final_hidden.i((bi, idx))?);
        }
        let pooled = Tensor::stack(&rows, 0)?;
        let pooled = self.text_projection.forward(&pooled)?;
        Ok((penultimate, pooled))
    }

    /// Stable Cascade's combined output. Differs from
    /// [`forward_for_sdxl`](Self::forward_for_sdxl) in ONE place: the
    /// per-token embeddings are the **LAST** hidden state
    /// (`hidden_states[-1]`, after the final encoder layer, before
    /// `final_layer_norm`), not the penultimate one. The Cascade prior
    /// pipeline reads `text_encoder_output.hidden_states[-1]`; SDXL
    /// reads `[-2]`. Using SDXL's penultimate layer here produced
    /// passable output for simple prompts but melted complex ones
    /// (v0.41 phase 2j — the steppe prompt) because the final layer
    /// carries the semantic refinement the prior was trained on.
    ///
    /// Returns `(last_hidden, pooled)`; `pooled` is identical to the
    /// SDXL path (projected EOT row of `final_layer_norm(...)`).
    pub fn forward_for_cascade(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // until_layer = -1 → the output after the final encoder layer.
        let (final_hidden, last_hidden) =
            self.inner.forward_until_encoder_layer(ids, usize::MAX, -1)?;
        let (b, _seq_len) = ids.dims2()?;
        let eot = eot_rows(ids)?;
        let mut rows = Vec::with_capacity(b);
        for (bi, &idx) in eot.iter().enumerate() {
            rows.push(final_hidden.i((bi, idx))?);
        }
        let pooled = Tensor::stack(&rows, 0)?;
        let pooled = self.text_projection.forward(&pooled)?;
        Ok((last_hidden, pooled))
    }
}

// Local import alias — keeps the `i(...)` indexing call above terse.
use candle_core::IndexOp;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn eot_rows_picks_eos_not_a_higher_ti_trigger() {
        let dev = Device::Cpu;
        // Row 0: EOS(49407) at index 2, then pad. Row 1: a TI trigger (49408, HIGHER than
        // EOS) sits around the EOS — plain argmax would wrongly select the trigger's row;
        // eot_rows must return the first real EOS position (index 3).
        let ids = Tensor::new(
            &[[49406u32, 10, 49407, 0, 0], [49406, 49408, 11, 49407, 49408]],
            &dev,
        )
        .unwrap();
        assert_eq!(eot_rows(&ids).unwrap(), vec![2, 3], "first EOS per row, not the max-id token");
    }
}
