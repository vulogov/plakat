//! Vendored T5 **encoder** (faithful copy of candle-transformers 0.10.2 `models::t5`).
//!
//! This is a byte-equivalent vendoring of the encoder half of upstream T5,
//! preserving the exact math (relative-position bias, T5 layer-norm, gated/
//! ungated FF, F32 layer-norm handling) so that inference matches upstream to
//! the bit. The ONLY additions are two public methods on `T5EncoderModel`,
//! `embed_tokens` and `forward_from_input_embeds`, needed for Textual-Inversion
//! training (run the encoder stack on already-embedded inputs, bypassing the
//! shared id-lookup).
//!
//! Drop-in for `candle_transformers::models::t5::{T5EncoderModel, Config}` as
//! used in `sd3.rs` / `pixart.rs`:
//!   - `T5EncoderModel::load(vb, &cfg) -> Result<Self>`
//!   - `model.forward(&input_ids) -> Result<Tensor>`  (`&mut self`)
//!
//! Source: candle-transformers-0.10.2/src/models/t5.rs
//! (with `crate::models::with_tracing::Embedding` inlined as a thin wrapper so
//! this file has no dependency on candle-transformers internals).

#![allow(dead_code)]

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Inlined `with_tracing::Embedding` (upstream uses this wrapper; we mirror its
// public surface — `new`, `forward`, `embeddings`).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// Linear (no-bias, T5-style; verbatim from upstream)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    span: tracing::Span,
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let weight = vb.get_with_hints((d2, d1), "weight", init_ws)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { weight, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let weight = self.weight.to_dtype(xs.dtype())?;
        let w = match *xs.dims() {
            [b1, b2, _, _] => weight.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => weight.broadcast_left(bsize)?.t()?,
            _ => weight.t()?,
        };
        xs.matmul(&w)
    }
}

// ---------------------------------------------------------------------------
// Config + serde defaults (verbatim from upstream)
// ---------------------------------------------------------------------------

fn default_relative_attention_max_distance() -> usize {
    128
}

fn default_is_decoder() -> bool {
    false
}

fn default_use_cache() -> bool {
    true
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Deserialize, Default, Clone, PartialEq)]
pub struct ActivationWithOptionalGating {
    pub gated: bool,
    pub activation: candle_nn::Activation,
}

pub fn deserialize_feed_forward_proj_activation<'de, D>(
    deserializer: D,
) -> std::result::Result<ActivationWithOptionalGating, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    match String::deserialize(deserializer)?.as_str() {
        "gated-gelu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::NewGelu,
        }),
        "gated-silu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::Silu,
        }),
        buf => {
            // Upstream uses `serde_plain::from_str(buf)` to map a bare
            // activation name (e.g. "relu", "gelu_new") onto the
            // `candle_nn::Activation` enum. plakat doesn't depend on
            // serde_plain, so we route the same string through serde_json by
            // quoting it — byte-equivalent for the serde-renamed unit variants
            // (`#[serde(rename_all = "snake_case")]` / explicit renames) that
            // `Activation` uses.
            let quoted = format!("\"{}\"", buf.replace('\\', "\\\\").replace('"', "\\\""));
            let activation: candle_nn::Activation =
                serde_json::from_str(&quoted).map_err(serde::de::Error::custom)?;
            Ok(ActivationWithOptionalGating {
                gated: false,
                activation,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_factor: f64,
    #[serde(default, deserialize_with = "deserialize_feed_forward_proj_activation")]
    pub feed_forward_proj: ActivationWithOptionalGating,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_is_decoder")]
    pub is_decoder: bool,
    pub is_encoder_decoder: bool,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub decoder_start_token_id: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: false,
                activation: Activation::Relu,
            },
            tie_word_embeddings: true,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: Some(0),
        }
    }
}

impl Config {
    // https://huggingface.co/facebook/musicgen-small/blob/495da4ad086b3416a27c6187f9239f9fd96f3962/config.json#L184
    pub fn musicgen_small() -> Self {
        Self {
            d_ff: 3072,
            d_kv: 64,
            d_model: 768,
            dropout_rate: 0.1,
            eos_token_id: 1,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: false,
                activation: Activation::Relu,
            },
            tie_word_embeddings: true,
            initializer_factor: 1.0,
            is_decoder: false,
            is_encoder_decoder: true,
            layer_norm_epsilon: 1e-6,
            num_decoder_layers: Some(12),
            num_heads: 12,
            num_layers: 12,
            pad_token_id: 0,
            decoder_start_token_id: Some(0),
            relative_attention_max_distance: 128,
            relative_attention_num_buckets: 32,
            use_cache: true,
            vocab_size: 32128,
        }
    }
}

// ---------------------------------------------------------------------------
// T5LayerNorm (F32 variance; verbatim)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
    span: tracing::Span,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight.to_dtype(dtype)?)?;
        Ok(xs)
    }
}

// ---------------------------------------------------------------------------
// Feed-forward (ungated + gated; verbatim)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct T5DenseActDense {
    wi: Linear,
    wo: Linear,
    act: Activation,
    span: tracing::Span,
}

impl T5DenseActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi"))?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi,
            wo,
            act: Activation::Relu,
            span: tracing::span!(tracing::Level::TRACE, "dense-act-dense"),
        })
    }
}

impl Module for T5DenseActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.wi.forward(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseGatedActDense {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    act: Activation,
    span: tracing::Span,
}

impl T5DenseGatedActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi_0 = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi_0"))?;
        let wi_1 = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi_1"))?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi_0,
            wi_1,
            wo,
            act: cfg.feed_forward_proj.activation,
            span: tracing::span!(tracing::Level::TRACE, "dense-gated-act-dense"),
        })
    }
}

impl Module for T5DenseGatedActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_gelu = self.act.forward(&self.wi_0.forward(xs)?)?;
        let hidden_linear = self.wi_1.forward(xs)?;
        let xs = hidden_gelu.broadcast_mul(&hidden_linear)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5LayerFF {
    dense_act: Option<T5DenseActDense>,
    gated_dense_act: Option<T5DenseGatedActDense>,
    layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5LayerFF {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        let (dense_act, gated_dense_act) = if cfg.feed_forward_proj.gated {
            (
                None,
                Some(T5DenseGatedActDense::load(vb.pp("DenseReluDense"), cfg)?),
            )
        } else {
            (
                Some(T5DenseActDense::load(vb.pp("DenseReluDense"), cfg)?),
                None,
            )
        };
        Ok(Self {
            dense_act,
            gated_dense_act,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer-ff"),
        })
    }
}

impl Module for T5LayerFF {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.layer_norm.forward(xs)?;
        let ys = match &self.dense_act {
            Some(dense_act) => dense_act.forward(&ys)?,
            None => self.gated_dense_act.as_ref().unwrap().forward(&ys)?,
        };
        let xs = (xs + ys)?;
        Ok(xs)
    }
}

// ---------------------------------------------------------------------------
// Attention + relative-position bias (verbatim; the bias math is subtle)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    inner_dim: usize,
    use_cache: bool,
    kv_cache: Option<(Tensor, Tensor)>,
    span: tracing::Span,
    span_cache: tracing::Span,
    span_mm: tracing::Span,
    span_sm: tracing::Span,
}

impl T5Attention {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &Config,
    ) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = linear_no_bias(cfg.d_model, inner_dim, vb.pp("q"))?;
        let k = linear_no_bias(cfg.d_model, inner_dim, vb.pp("k"))?;
        let v = linear_no_bias(cfg.d_model, inner_dim, vb.pp("v"))?;
        let o = linear_no_bias(inner_dim, cfg.d_model, vb.pp("o"))?;
        let relative_attention_bias = if has_relative_attention_bias {
            let emb = Embedding::new(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                vb.pp("relative_attention_bias"),
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            inner_dim,
            use_cache: cfg.use_cache && decoder,
            kv_cache: None,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
            span_cache: tracing::span!(tracing::Level::TRACE, "attention-cache"),
            span_mm: tracing::span!(tracing::Level::TRACE, "attention-mm"),
            span_sm: tracing::span!(tracing::Level::TRACE, "attention-sm"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Performs Self-attention (if key_value_states is None) or attention
        // over source sentence (provided by key_value_states).
        let _enter = self.span.enter();
        let kv_input = match key_value_states {
            None => xs,
            Some(key_value_states) => key_value_states,
        };
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = kv_input.dim(1)?;
        let q = self.q.forward(xs)?;
        let k = self.k.forward(kv_input)?;
        let v = self.v.forward(kv_input)?;
        let q = q
            .reshape((b_sz, q_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;

        if self.use_cache && key_value_states.is_none() {
            let _enter = self.span_cache.enter();
            if let Some((kv_cache_k, kv_cache_v)) = &self.kv_cache {
                k = Tensor::cat(&[kv_cache_k, &k], 2)?;
                v = Tensor::cat(&[kv_cache_v, &v], 2)?;
            };
            self.kv_cache = Some((k.clone(), v.clone()));
        };
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // TODO: Use flash_attn.
        let scores = {
            let _enter = self.span_mm.enter();
            q.matmul(&k.t()?)?
        };
        let scores = match mask {
            None => scores,
            Some(mask) => masked_fill(
                &scores,
                &mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .repeat((b_sz, self.n_heads))?,
                f32::NEG_INFINITY,
            )?,
        };

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
                    // This only handles the bidirectional case.
                    let kv_len = k.dim(2)?;
                    let (q_start, q_end) = match self.use_cache {
                        true => ((kv_len - q_len) as u32, kv_len as u32),
                        false => (0_u32, kv_len as u32),
                    };
                    let num_buckets = self.relative_attention_num_buckets as u32 / 2;
                    let max_exact = num_buckets / 2;
                    let relative_position = (q_start..q_end)
                        .map(|i| {
                            (0..kv_len as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact {
                                            j - i + num_buckets
                                        } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32
                                                    / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(
                                                max_exact + num_buckets + b as u32,
                                                self.relative_attention_num_buckets as u32 - 1,
                                            )
                                        }
                                    } else if i - j < max_exact {
                                        i - j
                                    } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32
                                                / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        u32::min(max_exact + b as u32, num_buckets - 1)
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                    let relative_buckets = Tensor::new(relative_position, q.device())?;
                    let position_bias = relative_attention_bias
                        .forward(&relative_buckets)?
                        .permute((2, 0, 1))?
                        .unsqueeze(0)?
                        .to_dtype(scores.dtype())?;
                    (scores.broadcast_add(&position_bias)?, Some(position_bias))
                    // TODO: position_bias_masked?
                }
            },
        };

        let attn_weights = {
            let _enter = self.span_sm.enter();
            candle_nn::ops::softmax_last_dim(&scores)?
        };
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok((attn_output, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5LayerSelfAttention {
    fn load(h: bool, d: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, d, vb.pp("SelfAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            self_attention,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) =
            self.self_attention
                .forward(&normed_xs, position_bias, None, mask)?;
        let ys = (xs + ys)?;
        Ok((ys, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attention.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
struct T5LayerCrossAttention {
    cross_attention: T5Attention,
    layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5LayerCrossAttention {
    fn load(decoder: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let cross_attention = T5Attention::load(false, decoder, vb.pp("EncDecAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            cross_attention,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "cross-attn"),
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let normed_hidden_states = self.layer_norm.forward(hidden_states)?;
        let (ys, position_bias) = self.cross_attention.forward(
            &normed_hidden_states,
            position_bias,
            Some(key_value_states),
            None,
        )?;
        let ys = (hidden_states + ys)?;
        Ok((ys, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.cross_attention.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
    span: tracing::Span,
}

impl T5Block {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &Config,
    ) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn =
            T5LayerSelfAttention::load(has_relative_attention_bias, decoder, vb.pp("0"), cfg)?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(decoder, vb.pp("1"), cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(vb.pp(ff_i.to_string()), cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
            span: tracing::span!(tracing::Level::TRACE, "block"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        // TODO: Cache masks
        let mask = match self.cross_attn.is_some() {
            true => {
                let mask_len = xs.dim(1)?;
                // If the input seq length is 1, no need for a mask, this is also helpful to avoid shape
                // issues when using the KV cache in the decoder.
                if mask_len <= 1 {
                    None
                } else {
                    Some(get_mask(mask_len, xs.device())?)
                }
            }
            false => None,
        };
        let (mut xs, position_bias) = self.self_attn.forward(xs, position_bias, mask.as_ref())?;
        // TODO: clamp for f16?
        if let Some(cross_attn) = &mut self.cross_attn {
            (xs, _) = cross_attn.forward(&xs, None, encoder_hidden_states.unwrap())?;
            // TODO: clamp for f16?
        }
        let xs = self.ff.forward(&xs)?;
        // TODO: clamp for f16?
        Ok((xs, position_bias))
    }

    // Immutable encoder-only forward. The encoder never mutates kv_cache
    // (use_cache is false), has no cross-attn and no decoder mask, so this is
    // numerically identical to `forward` along the encoder path — it merely
    // avoids requiring `&mut self`, which lets `forward_from_input_embeds`
    // (and the shared `encode_from_embeds`) run on `&self`.
    fn forward_encoder(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let (xs, position_bias) = self.self_attn_forward_encoder(xs, position_bias)?;
        let xs = self.ff.forward(&xs)?;
        Ok((xs, position_bias))
    }

    fn self_attn_forward_encoder(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.self_attn.span.enter();
        let normed_xs = self.self_attn.layer_norm.forward(xs)?;
        let (ys, position_bias) =
            self.self_attn
                .self_attention
                .forward_encoder(&normed_xs, position_bias)?;
        let ys = (xs + ys)?;
        Ok((ys, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
        self.cross_attn.iter_mut().for_each(|c| c.clear_kv_cache());
    }
}

impl T5Attention {
    // Immutable self-attention for the encoder path (key_value_states = None,
    // mask = None, use_cache = false). Same math as `forward`, no kv_cache
    // mutation, so it can take `&self`.
    fn forward_encoder(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = xs.dim(1)?;
        let q = self.q.forward(xs)?;
        let k = self.k.forward(xs)?;
        let v = self.v.forward(xs)?;
        let q = q
            .reshape((b_sz, q_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let scores = {
            let _enter = self.span_mm.enter();
            q.matmul(&k.t()?)?
        };

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
                    // Bidirectional encoder case (use_cache = false).
                    let kv_len = k.dim(2)?;
                    let (q_start, q_end) = (0_u32, kv_len as u32);
                    let num_buckets = self.relative_attention_num_buckets as u32 / 2;
                    let max_exact = num_buckets / 2;
                    let relative_position = (q_start..q_end)
                        .map(|i| {
                            (0..kv_len as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact {
                                            j - i + num_buckets
                                        } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32
                                                    / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(
                                                max_exact + num_buckets + b as u32,
                                                self.relative_attention_num_buckets as u32 - 1,
                                            )
                                        }
                                    } else if i - j < max_exact {
                                        i - j
                                    } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32
                                                / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        u32::min(max_exact + b as u32, num_buckets - 1)
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                    let relative_buckets = Tensor::new(relative_position, q.device())?;
                    let position_bias = relative_attention_bias
                        .forward(&relative_buckets)?
                        .permute((2, 0, 1))?
                        .unsqueeze(0)?
                        .to_dtype(scores.dtype())?;
                    (scores.broadcast_add(&position_bias)?, Some(position_bias))
                }
            },
        };

        let attn_weights = {
            let _enter = self.span_sm.enter();
            candle_nn::ops::softmax_last_dim(&scores)?
        };
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok((attn_output, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5Stack {
    fn load(decoder: bool, vb: VarBuilder, shared: &Arc<Embedding>, cfg: &Config) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, decoder, vb.pp(format!("block.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("final_layer_norm"),
        )?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "stack"),
        })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_dt(input_ids, encoder_hidden_states, None)
    }

    fn forward_dt(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        dtype: Option<DType>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let input_embeds = match dtype {
            None => input_embeds,
            Some(dtype) => input_embeds.to_dtype(dtype)?,
        };
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter_mut() {
            (hidden_states, position_bias) = block.forward(
                &hidden_states,
                position_bias.as_ref(),
                encoder_hidden_states,
            )?
        }
        self.final_layer_norm.forward(&hidden_states)
    }

    // Encoder-only post-embedding path, shared by the normal `forward`
    // (which embeds ids first) and `forward_from_input_embeds` (which is
    // handed the embeddings directly). Bit-identical to `forward_dt` along
    // the encoder branch: no cross-attn, no decoder mask, no kv_cache.
    fn encode_from_embeds(&self, input_embeds: Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter() {
            (hidden_states, position_bias) =
                block.forward_encoder(&hidden_states, position_bias.as_ref())?
        }
        self.final_layer_norm.forward(&hidden_states)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.shared.as_ref().forward(input_ids)
    }

    fn clear_kv_cache(&mut self) {
        self.block.iter_mut().for_each(|b| b.clear_kv_cache())
    }
}

#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: T5Stack,
    device: Device,
    span: tracing::Span,
}

impl T5EncoderModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let shared_vb = if vb.contains_tensor("shared.weight") {
            vb.pp("shared")
        } else if vb.contains_tensor("decoder.embed_tokens") {
            vb.pp("decoder").pp("embed_tokens")
        } else {
            vb.pp("encoder").pp("embed_tokens")
        };
        let shared = Embedding::new(cfg.vocab_size, cfg.d_model, shared_vb)?;
        let shared = Arc::new(shared);
        let encoder = T5Stack::load(false, vb.pp("encoder"), &shared, cfg)?;
        Ok(Self {
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    /// Run the full encoder over token ids — drop-in for upstream
    /// `T5EncoderModel::forward`. Embeds the ids via the shared embedding,
    /// then runs the encoder block stack + final layer norm.
    ///
    /// Numerically equivalent to `forward_from_input_embeds(&embed_tokens(ids))`:
    /// both share `T5Stack::encode_from_embeds` for the post-embedding path.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_embeds = self.encoder.embed_tokens(input_ids)?;
        self.encoder.encode_from_embeds(input_embeds)
    }

    /// Shared token-embedding lookup for `input_ids` `(B, L)` → `(B, L, d_model)`.
    /// This is exactly the `self.shared.forward(input_ids)` step the encoder
    /// runs internally before the blocks. Exposed for Textual-Inversion
    /// training (so a trainable embedding can be substituted for some ids).
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.encoder.embed_tokens(input_ids)
    }

    /// Run the encoder block stack + final layer norm on already-embedded
    /// inputs `(B, L, d_model)`, SKIPPING the shared id-lookup. Identical to
    /// `forward` apart from the missing embedding step — they share
    /// `T5Stack::encode_from_embeds`, so
    /// `forward(ids) == forward_from_input_embeds(embed_tokens(ids))`.
    pub fn forward_from_input_embeds(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.encoder.encode_from_embeds(inputs_embeds.clone())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn clear_kv_cache(&mut self) {
        self.encoder.clear_kv_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    // Tiny encoder config both this vendored Config and candle's upstream
    // Config deserialize identically.
    const TINY_CFG: &str = r#"{
        "vocab_size": 96, "d_model": 32, "d_kv": 8, "d_ff": 64,
        "num_layers": 2, "num_heads": 4,
        "relative_attention_num_buckets": 16,
        "relative_attention_max_distance": 128,
        "dropout_rate": 0.0, "layer_norm_epsilon": 0.000001,
        "initializer_factor": 1.0,
        "feed_forward_proj": "relu", "is_encoder_decoder": true,
        "use_cache": true, "pad_token_id": 0, "eos_token_id": 1
    }"#;

    /// THE faithfulness guard: the vendored encoder must compute the SAME
    /// output as candle's upstream T5 on identical (random) weights. If the
    /// copy ever drifts (a wrong layernorm, missing relative-position bias,
    /// etc.) this fails — which is what protects the verified SD3.5 inference
    /// / LoRA-training paths that now route prompt encoding through this copy.
    #[test]
    fn vendored_matches_candle_t5_on_shared_random_weights() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let cfg_v: Config = serde_json::from_str(TINY_CFG).unwrap();
        let cfg_c: candle_transformers::models::t5::Config =
            serde_json::from_str(TINY_CFG).unwrap();

        // Build candle first → it creates every var in the shared VarMap.
        let mut candle =
            candle_transformers::models::t5::T5EncoderModel::load(vb.clone(), &cfg_c).unwrap();
        // Randomize every weight in place (zeros would mask many bugs:
        // relative bias, attention scores etc. all collapse at 0).
        {
            let data = varmap.data().lock().unwrap();
            for var in data.values() {
                let r = Tensor::randn(0f32, 0.5f32, var.shape().dims(), &dev).unwrap();
                var.set(&r).unwrap();
            }
        }
        // Build vendored from the SAME VarMap → reuses the identical weights.
        let mut vend = T5EncoderModel::load(vb, &cfg_v).unwrap();

        let ids = Tensor::new(&[[5u32, 9, 12, 1, 0, 0]], &dev).unwrap();
        let out_c = candle.forward(&ids).unwrap();
        let out_v = vend.forward(&ids).unwrap();
        let diff = (out_c - out_v)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "vendored T5 diverged from candle T5 by {diff}");
    }

    /// The new TI methods must agree with the normal path:
    /// `forward(ids) == forward_from_input_embeds(embed_tokens(ids))`.
    #[test]
    fn forward_equals_forward_from_input_embeds() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg: Config = serde_json::from_str(TINY_CFG).unwrap();
        let mut m = T5EncoderModel::load(vb, &cfg).unwrap();
        {
            let data = varmap.data().lock().unwrap();
            for var in data.values() {
                let r = Tensor::randn(0f32, 0.5f32, var.shape().dims(), &dev).unwrap();
                var.set(&r).unwrap();
            }
        }
        let ids = Tensor::new(&[[3u32, 7, 11, 1, 0, 0]], &dev).unwrap();
        let a = m.forward(&ids).unwrap();
        let emb = m.embed_tokens(&ids).unwrap();
        let b = m.forward_from_input_embeds(&emb).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "forward vs forward_from_input_embeds diverged by {diff}");
    }
}
