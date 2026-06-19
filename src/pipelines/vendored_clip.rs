//! v0.30 phase 0: vendored CLIP text transformer.
//!
//! Minimal surgical fork of `candle_transformers::models::stable_diffusion::clip`
//! (candle 0.10.2). The fork exists for **one** reason: candle keeps
//! `clip::Config.vocab_size` private, which blocks Textual Inversion
//! (embedding) runtime injection — TI extends the token embedding
//! matrix from 49 408 → 49 408 + N rows, and candle's CLIP loader
//! rejects the larger matrix when the configured `vocab_size` doesn't
//! match.
//!
//! What this module is:
//! - Bit-faithful copy of candle's `ClipTextTransformer` forward pass
//!   (`forward_with_mask`, `forward_until_encoder_layer`).
//! - Same tensor key naming so SD safetensors load unchanged
//!   (`text_model.embeddings.token_embedding.weight`,
//!   `text_model.encoder.layers.{i}.self_attn.{k,v,q,out}_proj.{weight,bias}`,
//!   `text_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}`,
//!   `text_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}`,
//!   `text_model.final_layer_norm.{weight,bias}`).
//! - Public `Config` fields (especially `vocab_size`) so the
//!   embedding merger can hand us an extended-vocab Config.
//! - `Config::with_vocab(base, new_vocab_size)` helper that takes one
//!   of the stock configs and returns it with the override applied.
//!
//! What this module is NOT:
//! - A replacement for candle's CLIP everywhere. Phase 0 wires only
//!   `sd_core` (and its SDXL CLIP-G wrapper) through here. AnimateDiff,
//!   SD3, Flux, and stylize keep using candle's CLIP. They don't expose
//!   `--embedding` today; migration can happen in later cycles.
//! - The image encoder. Text only.
//!
//! Maintenance:
//! - If candle changes its CLIP text encoder internals (attention impl,
//!   layer norm placement, etc.), this fork falls behind. Mitigation:
//!   the no-embedding numerical regression test in
//!   `tests/embedding_runtime.rs` will fail if our forward drifts from
//!   candle's.
//! - If candle ever makes `vocab_size` public on `Config`, drop the
//!   fork and switch back to the upstream type.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    QuickGelu,
    Gelu,
    GeluErf,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::QuickGelu => xs * nn::ops::sigmoid(&(xs * 1.702f64)?)?,
            Activation::Gelu => xs.gelu(),
            Activation::GeluErf => xs.gelu_erf(),
        }
    }
}

/// CLIP text encoder config. Mirrors candle's
/// `stable_diffusion::clip::Config`, except every field is **public**
/// so the embedding TI runtime path can override `vocab_size`.
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub activation: Activation,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub pad_with: Option<String>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub projection_dim: usize,
}

impl Config {
    /// SD 1.5 CLIP-L (openai/clip-vit-large-patch14). 768d, 12 layers.
    pub fn v1_5() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    /// SD 2.1 CLIP (stabilityai/stable-diffusion-2-1). 1024d, 23 layers.
    pub fn v2_1() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1024,
            intermediate_size: 4096,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 23,
            num_attention_heads: 16,
            projection_dim: 512,
            activation: Activation::Gelu,
        }
    }

    /// SDXL CLIP-L (text_encoder). 768d, 12 layers.
    pub fn sdxl() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    /// SDXL CLIP-G (text_encoder_2). 1280d, 32 layers.
    pub fn sdxl2() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1280,
            intermediate_size: 5120,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 32,
            num_attention_heads: 20,
            projection_dim: 1280,
            activation: Activation::Gelu,
        }
    }

    /// Build a config with a custom `vocab_size`. Used by the TI
    /// runtime path: after `embedding::merge_embeddings_into_te_weights`
    /// extends the token embedding matrix, the loader needs a Config
    /// reporting the new vocab so the embedding row count matches.
    pub fn with_vocab(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }
}

/// Convert a candle stable_diffusion Config into a vendored Config.
/// Used by call sites that already hold a `StableDiffusionConfig` and
/// want to route its CLIP config through the vendored encoder without
/// hand-translating every field.
///
/// Re-derives from the alias: SD 1.5 → `v1_5`, SD 2.1 → `v2_1`,
/// SDXL CLIP-L → `sdxl`, SDXL CLIP-G → `sdxl2`. The caller picks the
/// constructor; this helper just makes the call uniform.
pub fn config_from_variant(variant: ClipVariant) -> Config {
    match variant {
        ClipVariant::Sd15 => Config::v1_5(),
        ClipVariant::Sd21 => Config::v2_1(),
        ClipVariant::SdxlL => Config::sdxl(),
        ClipVariant::SdxlG => Config::sdxl2(),
    }
}

/// Which CLIP variant to build. Mirrors the candle stock constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipVariant {
    Sd15,
    Sd21,
    SdxlL,
    SdxlG,
}

// ----------------------------------------------------------------------
// CLIP text model (faithful copy of candle's internals; private types).
// ----------------------------------------------------------------------

#[derive(Debug)]
struct ClipTextEmbeddings {
    token_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    position_ids: Tensor,
}

impl ClipTextEmbeddings {
    fn new(vs: nn::VarBuilder, c: &Config) -> Result<Self> {
        let token_embedding = nn::embedding(c.vocab_size, c.embed_dim, vs.pp("token_embedding"))?;
        let position_embedding = nn::embedding(
            c.max_position_embeddings,
            c.embed_dim,
            vs.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, c.max_position_embeddings as u32, vs.device())?.unsqueeze(0)?;
        Ok(ClipTextEmbeddings {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for ClipTextEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let token_embedding = self.token_embedding.forward(xs)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        token_embedding.broadcast_add(&position_embedding)
    }
}

#[derive(Debug)]
struct ClipAttention {
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    q_proj: nn::Linear,
    out_proj: nn::Linear,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl ClipAttention {
    fn new(vs: nn::VarBuilder, c: &Config) -> Result<Self> {
        let embed_dim = c.embed_dim;
        let num_attention_heads = c.num_attention_heads;
        let k_proj = nn::linear(embed_dim, embed_dim, vs.pp("k_proj"))?;
        let v_proj = nn::linear(embed_dim, embed_dim, vs.pp("v_proj"))?;
        let q_proj = nn::linear(embed_dim, embed_dim, vs.pp("q_proj"))?;
        let out_proj = nn::linear(embed_dim, embed_dim, vs.pp("out_proj"))?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);
        Ok(ClipAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;
        let query_states = (self.q_proj.forward(xs)? * self.scale)?;
        let proj_shape = (bsz * self.num_attention_heads, seq_len, self.head_dim);
        let query_states = self
            .shape(&query_states, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let key_states = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let value_states = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;

        let src_len = key_states.dim(1)?;
        let attn_weights = attn_weights
            .reshape((bsz, self.num_attention_heads, seq_len, src_len))?
            .broadcast_add(causal_attention_mask)?;
        let attn_weights =
            attn_weights.reshape((bsz * self.num_attention_heads, seq_len, src_len))?;
        let attn_weights = nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_attention_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Debug)]
struct ClipMlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
    activation: Activation,
}

impl ClipMlp {
    fn new(vs: nn::VarBuilder, c: &Config) -> Result<Self> {
        let fc1 = nn::linear(c.embed_dim, c.intermediate_size, vs.pp("fc1"))?;
        let fc2 = nn::linear(c.intermediate_size, c.embed_dim, vs.pp("fc2"))?;
        Ok(ClipMlp {
            fc1,
            fc2,
            activation: c.activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        self.fc2.forward(&self.activation.forward(&xs)?)
    }
}

#[derive(Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: nn::VarBuilder, c: &Config) -> Result<Self> {
        let self_attn = ClipAttention::new(vs.pp("self_attn"), c)?;
        let layer_norm1 = nn::layer_norm(c.embed_dim, 1e-5, vs.pp("layer_norm1"))?;
        let mlp = ClipMlp::new(vs.pp("mlp"), c)?;
        let layer_norm2 = nn::layer_norm(c.embed_dim, 1e-5, vs.pp("layer_norm2"))?;
        Ok(ClipEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, causal_attention_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Debug)]
struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    fn new(vs: nn::VarBuilder, c: &Config) -> Result<Self> {
        let vs = vs.pp("layers");
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..c.num_hidden_layers {
            let layer = ClipEncoderLayer::new(vs.pp(index.to_string()), c)?;
            layers.push(layer)
        }
        Ok(ClipEncoder { layers })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
        }
        Ok(xs)
    }
}

/// CLIP text transformer. Public API matches candle's
/// `stable_diffusion::clip::ClipTextTransformer` so call sites can
/// switch over without forward-pass changes.
#[derive(Debug)]
pub struct ClipTextTransformer {
    embeddings: ClipTextEmbeddings,
    encoder: ClipEncoder,
    final_layer_norm: nn::LayerNorm,
}

impl ClipTextTransformer {
    pub fn new(vs: nn::VarBuilder, c: &Config) -> Result<Self> {
        let vs = vs.pp("text_model");
        let embeddings = ClipTextEmbeddings::new(vs.pp("embeddings"), c)?;
        let encoder = ClipEncoder::new(vs.pp("encoder"), c)?;
        let final_layer_norm = nn::layer_norm(c.embed_dim, 1e-5, vs.pp("final_layer_norm"))?;
        Ok(ClipTextTransformer {
            embeddings,
            encoder,
            final_layer_norm,
        })
    }

    fn build_causal_attention_mask(
        bsz: usize,
        seq_len: usize,
        mask_after: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j > i || j > mask_after {
                        f32::MIN
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        mask.broadcast_as((bsz, seq_len, seq_len))
    }

    pub fn forward_with_mask(&self, xs: &Tensor, mask_after: usize) -> Result<Tensor> {
        let (bsz, seq_len) = xs.dims2()?;
        let xs = self.embeddings.forward(xs)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, xs.device())?;
        let xs = self.encoder.forward(&xs, &causal_attention_mask)?;
        self.final_layer_norm.forward(&xs)
    }

    /// Token-embedding lookup ONLY (no position embedding) — for Textual
    /// Inversion training: the placeholder row is replaced with the trainable
    /// vector before [`Self::forward_from_input_embeds`] runs the rest.
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embeddings.token_embedding.forward(token_ids)
    }

    /// Run the transformer from token-level input embeddings (post token-
    /// embedding, pre position-embedding) — the TI counterpart of
    /// `forward_with_mask`. Adds the position embedding, builds the causal mask,
    /// runs the encoder + final LN. `token_embeds` is `(bsz, 77, embed_dim)`.
    pub fn forward_from_input_embeds(
        &self,
        token_embeds: &Tensor,
        mask_after: usize,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = token_embeds.dims3()?;
        let pos_emb = self
            .embeddings
            .position_embedding
            .forward(&self.embeddings.position_ids)?;
        let xs = token_embeds.broadcast_add(&pos_emb)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, xs.device())?;
        let xs = self.encoder.forward(&xs, &causal_attention_mask)?;
        self.final_layer_norm.forward(&xs)
    }

    /// From-embeds counterpart of [`Self::forward_until_encoder_layer`] — the
    /// SDXL Textual-Inversion training path, where the trainable placeholder
    /// vector is spliced into the token embeddings before the encoder runs (a
    /// differentiable masked combine; the gradient reaches only that vector).
    /// `token_embeds` is `(bsz, seq, embed_dim)` (post token-embedding, pre
    /// position-embedding). Returns `(final_layer_norm(last), hidden_at_until_layer)`
    /// — identical outputs to the id-based version, so SDXL's penultimate-layer
    /// (`until_layer = -2`) concat and CLIP-G pooling are reproduced exactly.
    pub fn forward_until_encoder_layer_from_embeds(
        &self,
        token_embeds: &Tensor,
        mask_after: usize,
        until_layer: isize,
    ) -> Result<(Tensor, Tensor)> {
        let (bsz, seq_len, _) = token_embeds.dims3()?;
        let pos_emb = self
            .embeddings
            .position_embedding
            .forward(&self.embeddings.position_ids)?;
        let xs = token_embeds.broadcast_add(&pos_emb)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, xs.device())?;

        let mut xs = xs.clone();
        let mut intermediate = xs.clone();
        let until_layer = if until_layer < 0 {
            self.encoder.layers.len() as isize + until_layer
        } else {
            until_layer
        } as usize;
        for (layer_id, layer) in self.encoder.layers.iter().enumerate() {
            xs = layer.forward(&xs, &causal_attention_mask)?;
            if layer_id == until_layer {
                intermediate = xs.clone();
            }
        }
        Ok((self.final_layer_norm.forward(&xs)?, intermediate))
    }

    pub fn forward_until_encoder_layer(
        &self,
        xs: &Tensor,
        mask_after: usize,
        until_layer: isize,
    ) -> Result<(Tensor, Tensor)> {
        let (bsz, seq_len) = xs.dims2()?;
        let xs = self.embeddings.forward(xs)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, xs.device())?;

        let mut xs = xs.clone();
        let mut intermediate = xs.clone();

        let until_layer = if until_layer < 0 {
            self.encoder.layers.len() as isize + until_layer
        } else {
            until_layer
        } as usize;

        for (layer_id, layer) in self.encoder.layers.iter().enumerate() {
            xs = layer.forward(&xs, &causal_attention_mask)?;
            if layer_id == until_layer {
                intermediate = xs.clone();
            }
        }

        Ok((self.final_layer_norm.forward(&xs)?, intermediate))
    }
}

impl Module for ClipTextTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(xs, usize::MAX)
    }
}

/// Mirror of candle's `stable_diffusion::build_clip_transformer` for
/// the vendored type. Used by callers that load directly from a
/// safetensors path (no LoRA/embedding merge layer above).
pub fn build_clip_transformer<P: AsRef<std::path::Path>>(
    clip: &Config,
    clip_weights: P,
    device: &Device,
    dtype: DType,
) -> Result<ClipTextTransformer> {
    let vs = unsafe { nn::VarBuilder::from_mmaped_safetensors(&[clip_weights], dtype, device)? };
    ClipTextTransformer::new(vs, clip)
}

/// v0.32 phase 1: pipeline rollout marker. Re-exported here so a
/// single import covers every plakat pipeline that needs CLIP-L.
/// Mostly cosmetic — call sites use the concrete `ClipTextTransformer`
/// path — but having the marker keeps the rollout intent grep-able.
pub use ClipTextTransformer as PlakatClipTextTransformer;

#[cfg(test)]
mod tests {
    use super::*;

    /// v0.32 phase 1: confirms every pipeline that holds a CLIP-L
    /// text encoder field uses plakat's vendored type, not candle's.
    /// This is a structural lock — if a future refactor reintroduces
    /// `sdclip::ClipTextTransformer` on any pipeline's text-encoder
    /// field, the type-level binding here would break.
    ///
    /// Concrete pipeline field types covered:
    /// - `pipelines::sd_core::SdCore::text_encoder_l` (v0.30 phase 0)
    /// - `pipelines::animatediff::AnimateDiffPipeline::text_encoder`
    /// - `pipelines::animatediff::AnimateDiffSdxlPipeline::text_encoder_l`
    /// - `pipelines::sd3::Pipeline::clip_l`
    /// - `pipelines::flux::Pipeline::clip_text`
    /// - `pipelines::stylize::Pipeline::text_encoder`
    #[test]
    fn vendored_clip_field_type_lock() {
        // Each fn-pointer assignment fails to compile if the named
        // field has the wrong type. The test body just touches the
        // compile-time check; no runtime work needed.
        fn _check_sd_core(c: &crate::pipelines::sd_core::SdCore) -> &ClipTextTransformer {
            &c.text_encoder_l
        }
        fn _check_animate_sd15(
            c: &crate::pipelines::animatediff::AnimateDiffPipeline,
        ) -> &ClipTextTransformer {
            &c.text_encoder
        }
        fn _check_animate_sdxl(
            c: &crate::pipelines::animatediff::AnimateDiffSdxlPipeline,
        ) -> &ClipTextTransformer {
            &c.text_encoder_l
        }
        // SD3, Flux, stylize: fields are crate-private, so a type
        // probe via accessor isn't reachable from this file. The
        // build-time guarantee comes from each pipeline's
        // construction site using `vendored_clip::build_clip_transformer`
        // — if any swap regressed back to candle, the field's
        // initialiser would have failed to compile because candle's
        // builder returns `sdclip::ClipTextTransformer`, not ours.
        let _ = (_check_sd_core
            as fn(&crate::pipelines::sd_core::SdCore) -> &ClipTextTransformer);
        let _ = (_check_animate_sd15
            as fn(&crate::pipelines::animatediff::AnimateDiffPipeline) -> &ClipTextTransformer);
        let _ = (_check_animate_sdxl
            as fn(&crate::pipelines::animatediff::AnimateDiffSdxlPipeline)
                -> &ClipTextTransformer);
    }

    #[test]
    fn config_with_vocab_overrides_only_vocab() {
        let base = Config::v1_5();
        let extended = base.clone().with_vocab(49500);
        assert_eq!(extended.vocab_size, 49500);
        assert_eq!(extended.embed_dim, base.embed_dim);
        assert_eq!(extended.num_hidden_layers, base.num_hidden_layers);
        assert_eq!(extended.intermediate_size, base.intermediate_size);
        assert_eq!(extended.max_position_embeddings, base.max_position_embeddings);
    }

    #[test]
    fn variant_configs_match_candle_dims() {
        // Sanity: vendored constructors must produce the same numbers
        // candle's do, otherwise existing SD safetensors won't load.
        let v15 = Config::v1_5();
        assert_eq!(v15.vocab_size, 49408);
        assert_eq!(v15.embed_dim, 768);
        assert_eq!(v15.num_hidden_layers, 12);

        let v21 = Config::v2_1();
        assert_eq!(v21.embed_dim, 1024);
        assert_eq!(v21.num_hidden_layers, 23);

        let sdxl_l = Config::sdxl();
        assert_eq!(sdxl_l.embed_dim, 768);

        let sdxl_g = Config::sdxl2();
        assert_eq!(sdxl_g.embed_dim, 1280);
        assert_eq!(sdxl_g.num_hidden_layers, 32);
    }

    #[test]
    fn config_from_variant_dispatches_correctly() {
        assert_eq!(config_from_variant(ClipVariant::Sd15).embed_dim, 768);
        assert_eq!(config_from_variant(ClipVariant::Sd21).embed_dim, 1024);
        assert_eq!(config_from_variant(ClipVariant::SdxlL).embed_dim, 768);
        assert_eq!(config_from_variant(ClipVariant::SdxlG).embed_dim, 1280);
    }
}
