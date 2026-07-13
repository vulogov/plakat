//! Attention Based Building Blocks
//!
//! Vendored from candle_transformers 0.10.2 with one augmentation: the
//! `CrossAttention` q/k/v/out projections are `LoraLinear` (registered in
//! a shared `LoraRegistry`) so `plakat style train` can install trainable
//! adapters. With no adapter installed the forward is byte-identical to
//! candle's `nn::Linear`.
use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;
use std::sync::{Arc, RwLock};

use crate::pipelines::lora_linear::{LoraLinear, LoraRegistry, LoraRegistryEntry};

/// Build an `nn::Linear` and wrap it as a registered `LoraLinear` keyed by
/// the VarBuilder's full path (`<prefix>.weight`).
fn wrap_lin(
    vs: nn::VarBuilder,
    in_d: usize,
    out_d: usize,
    bias: bool,
    registry: &Arc<RwLock<LoraRegistry>>,
) -> Result<LoraLinear> {
    let base = if bias {
        nn::linear(in_d, out_d, vs.clone())?
    } else {
        nn::linear_no_bias(in_d, out_d, vs.clone())?
    };
    let ll = LoraLinear::from_linear(base)
        .map_err(|e| candle_core::Error::Msg(format!("sd_train wrap_lin at {}: {e}", vs.prefix())))?;
    let key = format!("{}.weight", vs.prefix());
    registry
        .write()
        .map_err(|_| candle_core::Error::Msg("sd_train LoRA registry poisoned".into()))?
        .insert(
            key,
            LoraRegistryEntry {
                handle: ll.slots_handle(),
                out_dim: out_d,
                in_dim: in_d,
                train: ll.train_handle(),
            },
        );
    Ok(ll)
}

#[derive(Debug)]
struct GeGlu {
    proj: nn::Linear,
    span: tracing::Span,
}

impl GeGlu {
    fn new(vs: nn::VarBuilder, dim_in: usize, dim_out: usize) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out * 2, vs.pp("proj"))?;
        let span = tracing::span!(tracing::Level::TRACE, "geglu");
        Ok(Self { proj, span })
    }
}

impl Module for GeGlu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states_and_gate = self.proj.forward(xs)?.chunk(2, D::Minus1)?;
        &hidden_states_and_gate[0] * hidden_states_and_gate[1].gelu()?
    }
}

/// A feed-forward layer.
#[derive(Debug)]
struct FeedForward {
    project_in: GeGlu,
    linear: nn::Linear,
    span: tracing::Span,
}

impl FeedForward {
    // The glu parameter in the python code is unused?
    // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L347
    /// Creates a new feed-forward layer based on some given input dimension, some
    /// output dimension, and a multiplier to be used for the intermediary layer.
    fn new(vs: nn::VarBuilder, dim: usize, dim_out: Option<usize>, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = vs.pp("net");
        let project_in = GeGlu::new(vs.pp("0"), dim, inner_dim)?;
        let linear = nn::linear(inner_dim, dim_out, vs.pp("2"))?;
        let span = tracing::span!(tracing::Level::TRACE, "ff");
        Ok(Self {
            project_in,
            linear,
            span,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.project_in.forward(xs)?;
        self.linear.forward(&xs)
    }
}

// plakat doesn't declare a `flash-attn` Cargo feature (it would pull in
// candle-flash-attn + CUDA build prerequisites). The vendored UNet is always
// built with `use_flash_attn = false`, so this is never reached — it exists only
// for upstream signature parity. Mirrors `mmdit_inner.rs` (no `cfg` gate → no
// `unexpected_cfgs` warning).
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("flash-attn not enabled in plakat; the softmax attention path is used")
}

/// InstantStyle decoupled IP cross-attention for ONE layer: separate K/V
/// projections for the image (style) tokens, whose attention output (same query
/// as the text branch) is added to the text attention scaled by `scale`. Only
/// the style block(s) carry this; everywhere else `CrossAttention.ip` is `None`
/// (behaviour unchanged). `tokens` is the shared projected style embedding, set
/// once before the denoise loop.
#[derive(Debug)]
pub struct IpInjection {
    to_k_ip: nn::Linear,
    to_v_ip: nn::Linear,
    scale: f64,
    tokens: Arc<RwLock<Option<Tensor>>>,
}

// PAG (Perturbed-Attention Guidance) for the own SD UNet. Two thread-locals (the denoise runs on one
// synchronous thread): the denoise loop marks the perturbed forward pass with `set_pag_pass`, and the
// UNet brackets ONLY the mid block with `set_pag_mid` — so self-attention there degenerates to
// identity (output = V) while the rest of the network is untouched. Diffusers' SDXL-PAG default is
// likewise the mid block; perturbing every block destabilises (the SD3 lesson).
thread_local! {
    static PAG_PASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PAG_MID: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Denoise loop: mark (true) the extra conditional forward as the PAG-perturbed pass.
pub fn set_pag_pass(on: bool) {
    PAG_PASS.with(|f| f.set(on));
}
/// UNet: is the current forward the PAG-perturbed pass? (→ perturb the mid block).
pub fn pag_pass_active() -> bool {
    PAG_PASS.with(|f| f.get())
}
/// UNet: enable/disable the mid-block self-attention perturbation (bracket the mid-block forward).
pub fn set_pag_mid(on: bool) {
    PAG_MID.with(|f| f.set(on));
}
fn pag_mid_active() -> bool {
    PAG_MID.with(|f| f.get())
}

#[derive(Debug)]
pub struct CrossAttention {
    to_q: LoraLinear,
    to_k: LoraLinear,
    to_v: LoraLinear,
    to_out: LoraLinear,
    heads: usize,
    scale: f64,
    slice_size: Option<usize>,
    span: tracing::Span,
    span_attn: tracing::Span,
    span_softmax: tracing::Span,
    use_flash_attn: bool,
    /// InstantStyle IP injection — `None` for self-attn and non-style layers.
    ip: Option<IpInjection>,
}

impl CrossAttention {
    // Defaults should be heads = 8, dim_head = 64, context_dim = None
    pub fn new(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        slice_size: Option<usize>,
        use_flash_attn: bool,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = wrap_lin(vs.pp("to_q"), query_dim, inner_dim, false, registry)?;
        let to_k = wrap_lin(vs.pp("to_k"), context_dim, inner_dim, false, registry)?;
        let to_v = wrap_lin(vs.pp("to_v"), context_dim, inner_dim, false, registry)?;
        let to_out = wrap_lin(vs.pp("to_out.0"), inner_dim, query_dim, true, registry)?;
        let span = tracing::span!(tracing::Level::TRACE, "xa");
        let span_attn = tracing::span!(tracing::Level::TRACE, "xa-attn");
        let span_softmax = tracing::span!(tracing::Level::TRACE, "xa-softmax");
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            scale,
            slice_size,
            span,
            span_attn,
            span_softmax,
            use_flash_attn,
            ip: None,
        })
    }

    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size, seq_len, self.heads, dim / self.heads))?
            .transpose(1, 2)?
            .reshape((batch_size * self.heads, seq_len, dim / self.heads))
    }

    fn reshape_batch_dim_to_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size / self.heads, self.heads, seq_len, dim))?
            .transpose(1, 2)?
            .reshape((batch_size / self.heads, seq_len, dim * self.heads))
    }

    fn sliced_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        slice_size: usize,
    ) -> Result<Tensor> {
        let batch_size_attention = query.dim(0)?;
        let mut hidden_states = Vec::with_capacity(batch_size_attention / slice_size);
        let in_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;

        for i in 0..batch_size_attention / slice_size {
            let start_idx = i * slice_size;
            let end_idx = (i + 1) * slice_size;

            let xs = query
                .i(start_idx..end_idx)?
                .matmul(&(key.i(start_idx..end_idx)?.t()? * self.scale)?)?;
            let xs = nn::ops::softmax(&xs, D::Minus1)?.matmul(&value.i(start_idx..end_idx)?)?;
            hidden_states.push(xs)
        }
        let hidden_states = Tensor::stack(&hidden_states, 0)?.to_dtype(in_dtype)?;
        self.reshape_batch_dim_to_heads(&hidden_states)
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        // v2.5: fused SDPA fast path (candle's Metal kernel — ~16× faster than eager, ~1e-6
        // correct). Inputs are `(b*heads, s, d)`; candle SDPA is 4D so we fold `b*heads` into the
        // head axis via `(1, b*heads, s, d)`. GPU-only (no CPU SDPA impl), head-dim-guarded (SDXL
        // d=64 ✓; SD1.5/2.1 use 40/80/160 → only 80 blocks qualify, others fall to eager). No mask
        // on this path. Escape hatch PLAKAT_NO_SDPA=1.
        let head_dim = query.dim(candle_core::D::Minus1)?;
        let sdpa_ok = !self.use_flash_attn
            && (query.device().is_metal() || query.device().is_cuda())
            && std::env::var("PLAKAT_NO_SDPA").is_err()
            && [32, 64, 72, 80, 96, 128, 256, 512].contains(&head_dim);
        let xs = if self.use_flash_attn {
            let init_dtype = query.dtype();
            let q = query
                .to_dtype(candle_core::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let k = key
                .to_dtype(candle_core::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let v = value
                .to_dtype(candle_core::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            flash_attn(&q, &k, &v, self.scale as f32, false)?
                .transpose(1, 2)?
                .squeeze(0)?
                .to_dtype(init_dtype)?
        } else if sdpa_ok {
            let q = query.unsqueeze(0)?.contiguous()?; // (1, b*heads, s, d)
            let k = key.unsqueeze(0)?.contiguous()?;
            let v = value.unsqueeze(0)?.contiguous()?;
            candle_nn::ops::sdpa(&q, &k, &v, None, false, self.scale as f32, 1.0)?.squeeze(0)?
        } else {
            let in_dtype = query.dtype();
            let query = query.to_dtype(DType::F32)?;
            let key = key.to_dtype(DType::F32)?;
            let value = value.to_dtype(DType::F32)?;
            let xs = query.matmul(&(key.t()? * self.scale)?)?;
            let xs = {
                let _enter = self.span_softmax.enter();
                nn::ops::softmax_last_dim(&xs)?
            };
            xs.matmul(&value)?.to_dtype(in_dtype)?
        };
        self.reshape_batch_dim_to_heads(&xs)
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        // PAG: on the perturbed pass, the mid block's SELF-attention (context = None) degenerates to
        // identity — its output is just V, the degenerate prediction guidance pushes away from. Cross-
        // attention (context = Some, the text conditioning) is never perturbed.
        let pag_identity = context.is_none() && pag_mid_active();
        let query = self.to_q.forward(xs)?;
        let context = context.unwrap_or(xs).contiguous()?;
        let key = self.to_k.forward(&context)?;
        let value = self.to_v.forward(&context)?;
        let query = self.reshape_heads_to_batch_dim(&query)?;
        let key = self.reshape_heads_to_batch_dim(&key)?;
        let value = self.reshape_heads_to_batch_dim(&value)?;
        let dim0 = query.dim(0)?;
        let slice_size = self.slice_size.and_then(|slice_size| {
            if dim0 < slice_size {
                None
            } else {
                Some(slice_size)
            }
        });
        let mut xs = if pag_identity {
            // Attention matrix = I → output = V (reshaped back from the per-head batch layout).
            self.reshape_batch_dim_to_heads(&value)?
        } else {
            match slice_size {
                None => self.attention(&query, &key, &value)?,
                Some(slice_size) => self.sliced_attention(&query, &key, &value, slice_size)?,
            }
        };
        // InstantStyle: add the decoupled IP (style) attention — same query,
        // separate K/V over the style tokens — scaled and summed before `to_out`.
        if let Some(ip) = &self.ip {
            if ip.scale != 0.0 {
                if let Ok(guard) = ip.tokens.read() {
                    if let Some(tokens) = guard.as_ref() {
                        let ip_k =
                            self.reshape_heads_to_batch_dim(&ip.to_k_ip.forward(tokens)?)?;
                        let ip_v =
                            self.reshape_heads_to_batch_dim(&ip.to_v_ip.forward(tokens)?)?;
                        let ip_xs = self.attention(&query, &ip_k, &ip_v)?;
                        xs = (xs + (ip_xs * ip.scale)?)?;
                    }
                }
            }
        }
        self.to_out.forward(&xs)
    }

    /// InstantStyle: attach a decoupled IP cross-attention to this layer. The
    /// caller does this only for the style block(s); everywhere else stays `None`.
    pub fn set_ip(&mut self, ip: IpInjection) {
        self.ip = Some(ip);
    }
}

impl IpInjection {
    /// Build from this layer's IP-Adapter K/V projections, the injection scale,
    /// and the shared style-token cell (set once before the denoise loop).
    pub fn new(
        to_k_ip: nn::Linear,
        to_v_ip: nn::Linear,
        scale: f64,
        tokens: Arc<RwLock<Option<Tensor>>>,
    ) -> Self {
        Self {
            to_k_ip,
            to_v_ip,
            scale,
            tokens,
        }
    }
}

/// A basic Transformer block.
#[derive(Debug)]
struct BasicTransformerBlock {
    attn1: CrossAttention,
    ff: FeedForward,
    attn2: CrossAttention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
    span: tracing::Span,
}

impl BasicTransformerBlock {
    fn new(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: Option<usize>,
        sliced_attention_size: Option<usize>,
        use_flash_attn: bool,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let attn1 = CrossAttention::new(
            vs.pp("attn1"),
            dim,
            None,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
            registry,
        )?;
        let ff = FeedForward::new(vs.pp("ff"), dim, None, 4)?;
        let attn2 = CrossAttention::new(
            vs.pp("attn2"),
            dim,
            context_dim,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
            registry,
        )?;
        let norm1 = nn::layer_norm(dim, 1e-5, vs.pp("norm1"))?;
        let norm2 = nn::layer_norm(dim, 1e-5, vs.pp("norm2"))?;
        let norm3 = nn::layer_norm(dim, 1e-5, vs.pp("norm3"))?;
        let span = tracing::span!(tracing::Level::TRACE, "basic-transformer");
        Ok(Self {
            attn1,
            ff,
            attn2,
            norm1,
            norm2,
            norm3,
            span,
        })
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = (self.attn1.forward(&self.norm1.forward(xs)?, None)? + xs)?;
        let xs = (self.attn2.forward(&self.norm2.forward(&xs)?, context)? + xs)?;
        self.ff.forward(&self.norm3.forward(&xs)?)? + xs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    pub depth: usize,
    pub num_groups: usize,
    pub context_dim: Option<usize>,
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            num_groups: 32,
            context_dim: None,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
enum Proj {
    Conv2d(nn::Conv2d),
    Linear(nn::Linear),
}

// Aka Transformer2DModel
#[derive(Debug)]
pub struct SpatialTransformer {
    norm: nn::GroupNorm,
    proj_in: Proj,
    transformer_blocks: Vec<BasicTransformerBlock>,
    proj_out: Proj,
    span: tracing::Span,
    pub config: SpatialTransformerConfig,
}

impl SpatialTransformer {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        use_flash_attn: bool,
        config: SpatialTransformerConfig,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let inner_dim = n_heads * d_head;
        let norm = nn::group_norm(config.num_groups, in_channels, 1e-6, vs.pp("norm"))?;
        let proj_in = if config.use_linear_projection {
            Proj::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_in"))?)
        } else {
            Proj::Conv2d(nn::conv2d(
                in_channels,
                inner_dim,
                1,
                Default::default(),
                vs.pp("proj_in"),
            )?)
        };
        let mut transformer_blocks = vec![];
        let vs_tb = vs.pp("transformer_blocks");
        for index in 0..config.depth {
            let tb = BasicTransformerBlock::new(
                vs_tb.pp(index.to_string()),
                inner_dim,
                n_heads,
                d_head,
                config.context_dim,
                config.sliced_attention_size,
                use_flash_attn,
                registry,
            )?;
            transformer_blocks.push(tb)
        }
        let proj_out = if config.use_linear_projection {
            Proj::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_out"))?)
        } else {
            Proj::Conv2d(nn::conv2d(
                inner_dim,
                in_channels,
                1,
                Default::default(),
                vs.pp("proj_out"),
            )?)
        };
        let span = tracing::span!(tracing::Level::TRACE, "spatial-transformer");
        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
            span,
            config,
        })
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (batch, _channel, height, weight) = xs.dims4()?;
        let residual = xs;
        let xs = self.norm.forward(xs)?;
        let (inner_dim, xs) = match &self.proj_in {
            Proj::Conv2d(p) => {
                let xs = p.forward(&xs)?;
                let inner_dim = xs.dim(1)?;
                let xs = xs
                    .transpose(1, 2)?
                    .t()?
                    .reshape((batch, height * weight, inner_dim))?;
                (inner_dim, xs)
            }
            Proj::Linear(p) => {
                let inner_dim = xs.dim(1)?;
                let xs = xs
                    .transpose(1, 2)?
                    .t()?
                    .reshape((batch, height * weight, inner_dim))?;
                (inner_dim, p.forward(&xs)?)
            }
        };
        let mut xs = xs;
        for block in self.transformer_blocks.iter() {
            xs = block.forward(&xs, context)?
        }
        let xs = match &self.proj_out {
            Proj::Conv2d(p) => p.forward(
                &xs.reshape((batch, height, weight, inner_dim))?
                    .t()?
                    .transpose(1, 2)?,
            )?,
            Proj::Linear(p) => p
                .forward(&xs)?
                .reshape((batch, height, weight, inner_dim))?
                .t()?
                .transpose(1, 2)?,
        };
        xs + residual
    }

    /// InstantStyle: mutable refs to every cross-attention (`attn2`) in this
    /// transformer, in order — for installing per-layer IP injections.
    pub fn attn2s_mut(&mut self) -> Vec<&mut CrossAttention> {
        self.transformer_blocks
            .iter_mut()
            .map(|tb| &mut tb.attn2)
            .collect()
    }
}

/// Configuration for an attention block.
#[derive(Debug, Clone, Copy)]
pub struct AttentionBlockConfig {
    pub num_head_channels: Option<usize>,
    pub num_groups: usize,
    pub rescale_output_factor: f64,
    pub eps: f64,
}

impl Default for AttentionBlockConfig {
    fn default() -> Self {
        Self {
            num_head_channels: None,
            num_groups: 32,
            rescale_output_factor: 1.,
            eps: 1e-5,
        }
    }
}

#[derive(Debug)]
pub struct AttentionBlock {
    group_norm: nn::GroupNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    proj_attn: nn::Linear,
    channels: usize,
    num_heads: usize,
    span: tracing::Span,
    config: AttentionBlockConfig,
}

// In the .safetensor weights of official Stable Diffusion 3 Medium Huggingface repo
// https://huggingface.co/stabilityai/stable-diffusion-3-medium
// Linear layer may use a different dimension for the weight in the linear, which is
// incompatible with the current implementation of the nn::linear constructor.
// This is a workaround to handle the different dimensions.
fn get_qkv_linear(channels: usize, vs: nn::VarBuilder) -> Result<nn::Linear> {
    match vs.get((channels, channels), "weight") {
        Ok(_) => nn::linear(channels, channels, vs),
        Err(_) => {
            let weight = vs
                .get((channels, channels, 1, 1), "weight")?
                .reshape((channels, channels))?;
            let bias = vs.get((channels,), "bias")?;
            Ok(nn::Linear::new(weight, Some(bias)))
        }
    }
}

impl AttentionBlock {
    pub fn new(vs: nn::VarBuilder, channels: usize, config: AttentionBlockConfig) -> Result<Self> {
        let num_head_channels = config.num_head_channels.unwrap_or(channels);
        let num_heads = channels / num_head_channels;
        let group_norm =
            nn::group_norm(config.num_groups, channels, config.eps, vs.pp("group_norm"))?;
        let (q_path, k_path, v_path, out_path) = if vs.contains_tensor("to_q.weight") {
            ("to_q", "to_k", "to_v", "to_out.0")
        } else {
            ("query", "key", "value", "proj_attn")
        };
        let query = get_qkv_linear(channels, vs.pp(q_path))?;
        let key = get_qkv_linear(channels, vs.pp(k_path))?;
        let value = get_qkv_linear(channels, vs.pp(v_path))?;
        let proj_attn = get_qkv_linear(channels, vs.pp(out_path))?;
        let span = tracing::span!(tracing::Level::TRACE, "attn-block");
        Ok(Self {
            group_norm,
            query,
            key,
            value,
            proj_attn,
            channels,
            num_heads,
            span,
            config,
        })
    }

    fn transpose_for_scores(&self, xs: Tensor) -> Result<Tensor> {
        let (batch, t, h_times_d) = xs.dims3()?;
        xs.reshape((batch, t, self.num_heads, h_times_d / self.num_heads))?
            .transpose(1, 2)
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let in_dtype = xs.dtype();
        let residual = xs;
        let (batch, channel, height, width) = xs.dims4()?;
        let xs = self
            .group_norm
            .forward(xs)?
            .reshape((batch, channel, height * width))?
            .transpose(1, 2)?;

        let query_proj = self.query.forward(&xs)?;
        let key_proj = self.key.forward(&xs)?;
        let value_proj = self.value.forward(&xs)?;

        let query_states = self
            .transpose_for_scores(query_proj)?
            .to_dtype(DType::F32)?;
        let key_states = self.transpose_for_scores(key_proj)?.to_dtype(DType::F32)?;
        let value_states = self
            .transpose_for_scores(value_proj)?
            .to_dtype(DType::F32)?;

        // scale is applied twice, hence the -0.25 here rather than -0.5.
        // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L87
        let scale = f64::powf(self.channels as f64 / self.num_heads as f64, -0.25);
        let attention_scores = (query_states * scale)?.matmul(&(key_states.t()? * scale)?)?;
        let attention_probs = nn::ops::softmax(&attention_scores, D::Minus1)?;

        // TODO: revert the call to force_contiguous once the three matmul kernels have been
        // adapted to handle layout with some dims set to 1.
        let xs = attention_probs.matmul(&value_states)?;
        let xs = xs.to_dtype(in_dtype)?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = xs.flatten_from(D::Minus2)?;
        let xs = self
            .proj_attn
            .forward(&xs)?
            .t()?
            .reshape((batch, channel, height, width))?;
        (xs + residual)? / self.config.rescale_output_factor
    }
}
