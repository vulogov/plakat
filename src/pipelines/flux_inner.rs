//! Vendored copy of candle-transformers 0.8.4's
//! `stable_diffusion::flux::model`, extended with a residual-aware
//! forward for Flux ControlNet (v0.12 phase 2a foundation).
//!
//! Upstream marks every block constructor and `forward` as private,
//! so any wrapper that needs to inject per-block residuals — i.e.
//! every Flux ControlNet implementation — has to vendor the whole
//! file. This module is mostly a 1:1 mechanical copy of upstream:
//!
//!   * Block / helper types: `Config`, `EmbedNd`, `MlpEmbedder`,
//!     `QkNorm`, `Modulation{1,2}`, `SelfAttention`, `Mlp`,
//!     `DoubleStreamBlock`, `SingleStreamBlock`, `LastLayer`,
//!     `Flux`. Same fields and same forward math as upstream.
//!   * Helpers: `layer_norm`, `scaled_dot_product_attention`, `rope`,
//!     `apply_rope`, `attention`, `timestep_embedding`.
//!
//! Visibility changes from upstream:
//!   * Every type and method we use is `pub`, so the FluxControlNet
//!     in phase 2b can construct its own DoubleStreamBlock /
//!     SingleStreamBlock instances against the same Config + VarBuilder
//!     paths candle's `Flux::new` expects.
//!
//! New surface added by this module:
//!   * `Flux::forward_with_residuals` — variant of the standard
//!     forward that accepts optional per-block residuals for the
//!     DoubleStream and (separately) the SingleStream loops. `None`
//!     reproduces upstream behaviour byte-for-byte. `Some` adds the
//!     supplied residual after the matching block (interleaving when
//!     the ControlNet has fewer residual heads than the main model
//!     has blocks — matches diffusers' FluxControlNet integration).
//!
//! When candle gains a residual-aware Flux upstream this module can
//! be deleted and `Flux` aliased back to the upstream type.

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, RmsNorm, VarBuilder};

// v0.15 phase 7b-3: every Linear in the vendored Flux backbone is now
// wrapped as a `LoraLinear` so the model can apply a runtime LoRA
// stack at forward time. The stack starts empty (behaves identically
// to the previous `nn::Linear` direct use) and is updated by
// `Flux::apply_loras` at scenario per-task dispatch time. The slot
// registry stores per-Linear handles keyed by full safetensors path
// so the dispatcher doesn't have to walk the model.
use crate::pipelines::lora_linear::{
    LoraLinear, LoraRegistry, LoraRegistryEntry, LoraSlot, LoraSpec,
};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------
// Config — same as upstream.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Config {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    pub qkv_bias: bool,
    pub guidance_embed: bool,
}

impl Config {
    pub fn dev() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
        }
    }

    pub fn schnell() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: false,
        }
    }

    /// v0.13 phase 2: Flux.1-Fill-dev. Structurally identical to
    /// `dev()` except `img_in` takes 384 input channels instead of 64.
    /// The 384 channels are laid out per Flux token as:
    ///
    /// ```text
    ///   [ noise_latent: 64 |  masked_image_latent: 64  |  mask: 256 ]
    /// ```
    ///
    /// * `noise_latent` is the standard Flux 2x2-patched noisy latent
    ///   (16 channels × 2×2 = 64) — same shape as Flux.1-dev's input.
    /// * `masked_image_latent` is the VAE-encoded init image with the
    ///   mask=1 (to-be-inpainted) region zeroed out, then 2x2-patched
    ///   the same way (64 channels per token).
    /// * `mask` is the **image-space** mask (1ch × H × W) reshaped
    ///   into 16×16 patches (each Flux token spans a 16-pixel patch
    ///   on the original image — 8× VAE downsample × 2× Flux patching)
    ///   → 256 channels per token.
    ///
    /// Everything else (DoubleStream/SingleStream blocks, AE, text
    /// encoders, guidance schedule) is identical to Flux.1-dev.
    pub fn fill_dev() -> Self {
        let mut cfg = Self::dev();
        cfg.in_channels = 384;
        cfg
    }

    /// v0.15 phase 4: Flux.1-Canny-dev / Flux.1-Depth-dev share an
    /// `img_in` layout that concatenates the 64-channel noise latent
    /// with a 64-channel VAE-encoded conditioning latent (canny edges
    /// or depth map). So `in_channels = 128`. Everything else
    /// (DoubleStream/SingleStream blocks, AE, T5/CLIP encoders,
    /// guidance schedule) is identical to Flux.1-dev — only the
    /// wider `img_in` Linear differs.
    pub fn canny_or_depth_dev() -> Self {
        let mut cfg = Self::dev();
        cfg.in_channels = 128;
        cfg
    }

    /// v0.18: Flux.1-Kontext-dev. Unlike Fill / Canny / Depth (which
    /// widen `img_in`), Kontext keeps `in_channels = 64` — the
    /// reference-image conditioning is sequence-concatenated onto
    /// the noise tokens at the DiT input level, not channel-concat
    /// at `img_in`. So the Config here is literally `Self::dev()`;
    /// the architectural difference lives in the pipeline-level
    /// `pack_kontext_reference` helper that prepends the VAE-encoded
    /// reference's tokens with `image_ids[..., 0] = 1` to mark them
    /// as the context half (Phase 2).
    pub fn kontext_dev() -> Self {
        Self::dev()
    }
}

// ---------------------------------------------------------------------
// Helpers — same as upstream, exposed `pub` so FluxControlNet can
// reuse them.
// ---------------------------------------------------------------------

pub fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
}

fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle_core::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?;
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())
}

pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    x.transpose(1, 2)?.flatten_from(2)
}

pub fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        candle_core::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

// ---------------------------------------------------------------------
// v0.15 phase 7b-3: wrap_linear / wrap_linear_b — load a candle
// Linear, hand it to `LoraLinear`, register the slots handle in the
// shared registry under `<prefix>.weight`. Every Linear in the
// vendored Flux backbone routes through these so the full path-keyed
// registry is ready for `Flux::apply_loras`.
// ---------------------------------------------------------------------

fn wrap_linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    registry: &Arc<RwLock<LoraRegistry>>,
) -> Result<LoraLinear> {
    let base = candle_nn::linear(in_dim, out_dim, vb.clone())?;
    let ll = LoraLinear::from_linear(base).map_err(|e| {
        candle_core::Error::Msg(format!("wrapping LoraLinear at {}: {e}", vb.prefix()))
    })?;
    let key = format!("{}.weight", vb.prefix());
    registry
        .write()
        .map_err(|_| {
            candle_core::Error::Msg("Flux LoRA registry poisoned during construction".into())
        })?
        .insert(
            key,
            LoraRegistryEntry {
                handle: ll.slots_handle(),
                out_dim,
                in_dim,
            },
        );
    Ok(ll)
}

fn wrap_linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    registry: &Arc<RwLock<LoraRegistry>>,
) -> Result<LoraLinear> {
    let base = candle_nn::linear_b(in_dim, out_dim, bias, vb.clone())?;
    let ll = LoraLinear::from_linear(base).map_err(|e| {
        candle_core::Error::Msg(format!("wrapping LoraLinear at {}: {e}", vb.prefix()))
    })?;
    let key = format!("{}.weight", vb.prefix());
    registry
        .write()
        .map_err(|_| {
            candle_core::Error::Msg("Flux LoRA registry poisoned during construction".into())
        })?
        .insert(
            key,
            LoraRegistryEntry {
                handle: ll.slots_handle(),
                out_dim,
                in_dim,
            },
        );
    Ok(ll)
}

// ---------------------------------------------------------------------
// EmbedNd, MlpEmbedder, QkNorm — same as upstream, made `pub`.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EmbedNd {
    #[allow(unused)]
    dim: usize,
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    pub fn new(dim: usize, theta: usize, axes_dim: Vec<usize>) -> Self {
        Self {
            dim,
            theta,
            axes_dim,
        }
    }
}

impl candle_core::Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            let r = rope(
                &ids.get_on_dim(D::Minus1, idx)?,
                self.axes_dim[idx],
                self.theta,
            )?;
            emb.push(r)
        }
        let emb = Tensor::cat(&emb, 2)?;
        emb.unsqueeze(1)
    }
}

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: LoraLinear,
    out_layer: LoraLinear,
}

impl MlpEmbedder {
    pub fn new(
        in_sz: usize,
        h_sz: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let in_layer = wrap_linear(in_sz, h_sz, vb.pp("in_layer"), registry)?;
        let out_layer = wrap_linear(h_sz, h_sz, vb.pp("out_layer"), registry)?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }
}

impl candle_core::Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

#[derive(Debug, Clone)]
pub struct QkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl QkNorm {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let query_norm = vb.get(dim, "query_norm.scale")?;
        let query_norm = RmsNorm::new(query_norm, 1e-6);
        let key_norm = vb.get(dim, "key_norm.scale")?;
        let key_norm = RmsNorm::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
}

// ---------------------------------------------------------------------
// Modulation — small enough to keep crate-private, but the Block types
// they're embedded in are `pub`, so these stay accessible via their
// callers.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    pub fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    pub fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Modulation1 {
    lin: LoraLinear,
}

impl Modulation1 {
    pub fn new(
        dim: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let lin = wrap_linear(dim, 3 * dim, vb.pp("lin"), registry)?;
        Ok(Self { lin })
    }

    pub fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Modulation2 {
    lin: LoraLinear,
}

impl Modulation2 {
    pub fn new(
        dim: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let lin = wrap_linear(dim, 6 * dim, vb.pp("lin"), registry)?;
        Ok(Self { lin })
    }

    pub fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

// ---------------------------------------------------------------------
// SelfAttention, Mlp — same math as upstream.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SelfAttention {
    qkv: LoraLinear,
    norm: QkNorm,
    pub proj: LoraLinear,
    num_heads: usize,
}

impl SelfAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = wrap_linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"), registry)?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = wrap_linear(dim, dim, vb.pp("proj"), registry)?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
        })
    }

    pub fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }
}

#[derive(Debug, Clone)]
pub struct Mlp {
    lin1: LoraLinear,
    lin2: LoraLinear,
}

impl Mlp {
    pub fn new(
        in_sz: usize,
        mlp_sz: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let lin1 = wrap_linear(in_sz, mlp_sz, vb.pp("0"), registry)?;
        let lin2 = wrap_linear(mlp_sz, in_sz, vb.pp("2"), registry)?;
        Ok(Self { lin1, lin2 })
    }
}

impl candle_core::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

// ---------------------------------------------------------------------
// DoubleStreamBlock, SingleStreamBlock — same math as upstream, with
// `pub` constructors and forwards so FluxControlNet (phase 2b) can
// drive them directly.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleStreamBlock {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let img_mod = Modulation2::new(h_sz, vb.pp("img_mod"), registry)?;
        let img_norm1 = layer_norm(h_sz, vb.pp("img_norm1"))?;
        let img_attn = SelfAttention::new(
            h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"), registry,
        )?;
        let img_norm2 = layer_norm(h_sz, vb.pp("img_norm2"))?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("img_mlp"), registry)?;
        let txt_mod = Modulation2::new(h_sz, vb.pp("txt_mod"), registry)?;
        let txt_norm1 = layer_norm(h_sz, vb.pp("txt_norm1"))?;
        let txt_attn = SelfAttention::new(
            h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"), registry,
        )?;
        let txt_norm2 = layer_norm(h_sz, vb.pp("txt_norm2"))?;
        let txt_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("txt_mlp"), registry)?;
        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
        })
    }

    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?;
        let img_modulated = img.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_modulated)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_modulated)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&img_attn.apply(&self.img_attn.proj)?))?;
        let img = (&img
            + img_mod2.gate(
                &img_mod2
                    .scale_shift(&img.apply(&self.img_norm2)?)?
                    .apply(&self.img_mlp)?,
            )?)?;

        let txt = (txt + txt_mod1.gate(&txt_attn.apply(&self.txt_attn.proj)?))?;
        let txt = (&txt
            + txt_mod2.gate(
                &txt_mod2
                    .scale_shift(&txt.apply(&self.txt_norm2)?)?
                    .apply(&self.txt_mlp)?,
            )?)?;

        Ok((img, txt))
    }
}

#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    linear1: LoraLinear,
    linear2: LoraLinear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_heads;
        let linear1 = wrap_linear(
            h_sz, h_sz * 3 + mlp_sz, vb.pp("linear1"), registry,
        )?;
        let linear2 = wrap_linear(h_sz + mlp_sz, h_sz, vb.pp("linear2"), registry)?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let pre_norm = layer_norm(h_sz, vb.pp("pre_norm"))?;
        let modulation = Modulation1::new(h_sz, vb.pp("modulation"), registry)?;
        Ok(Self {
            linear1,
            linear2,
            norm,
            pre_norm,
            modulation,
            h_sz,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    pub fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let mod_ = self.modulation.forward(vec_)?;
        let x_mod = mod_.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = x_mod.apply(&self.linear1)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output = Tensor::cat(&[attn, mlp.gelu()?], 2)?.apply(&self.linear2)?;
        xs + mod_.gate(&output)
    }
}

#[derive(Debug, Clone)]
pub struct LastLayer {
    norm_final: LayerNorm,
    linear: LoraLinear,
    ada_ln_modulation: LoraLinear,
}

impl LastLayer {
    pub fn new(
        h_sz: usize,
        p_sz: usize,
        out_c: usize,
        vb: VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.pp("norm_final"))?;
        let linear = wrap_linear(
            h_sz, p_sz * p_sz * out_c, vb.pp("linear"), registry,
        )?;
        let ada_ln_modulation = wrap_linear(
            h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"), registry,
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    pub fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.linear)
    }
}

// ---------------------------------------------------------------------
// Flux — outer transformer. Same fields and same standard forward as
// upstream. The new `forward_with_residuals` plumbs per-block
// residuals through both loops.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Flux {
    img_in: LoraLinear,
    txt_in: LoraLinear,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    pub double_blocks: Vec<DoubleStreamBlock>,
    pub single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
    /// v0.15 phase 7b-3: path → LoraLinear-slots handle map populated
    /// during construction. Consumed by `apply_loras` at scenario
    /// per-task dispatch time so we can update slots by safetensors
    /// path without re-walking blocks.
    lora_registry: LoraRegistry,
}

impl Flux {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // v0.15 phase 7b-3: shared LoRA registry — every constructed
        // LoraLinear writes its slot handle into this map. After all
        // sub-loaders go out of scope at the end of construction, we
        // unwrap the Arc and move the inner HashMap into the Flux
        // struct.
        let registry = Arc::new(RwLock::new(LoraRegistry::new()));
        let img_in = wrap_linear(
            cfg.in_channels, cfg.hidden_size, vb.pp("img_in"), &registry,
        )?;
        let txt_in = wrap_linear(
            cfg.context_in_dim, cfg.hidden_size, vb.pp("txt_in"), &registry,
        )?;
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.depth {
            let db = DoubleStreamBlock::new(cfg, vb_d.pp(idx), &registry)?;
            double_blocks.push(db)
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.depth_single_blocks {
            let sb = SingleStreamBlock::new(cfg, vb_s.pp(idx), &registry)?;
            single_blocks.push(sb)
        }
        let time_in = MlpEmbedder::new(256, cfg.hidden_size, vb.pp("time_in"), &registry)?;
        let vector_in = MlpEmbedder::new(
            cfg.vec_in_dim, cfg.hidden_size, vb.pp("vector_in"), &registry,
        )?;
        let guidance_in = if cfg.guidance_embed {
            let mlp = MlpEmbedder::new(
                256, cfg.hidden_size, vb.pp("guidance_in"), &registry,
            )?;
            Some(mlp)
        } else {
            None
        };
        let final_layer = LastLayer::new(
            cfg.hidden_size, 1, cfg.in_channels, vb.pp("final_layer"), &registry,
        )?;
        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.theta, cfg.axes_dim.to_vec());
        // Drop the loader chain so the registry Arc's refcount returns
        // to 1, then `try_unwrap`. The local `vb` clones in nested
        // constructors already went out of scope before this point;
        // the only remaining Arc is our local `registry`.
        let lora_registry = Arc::try_unwrap(registry)
            .map_err(|_| {
                candle_core::Error::Msg(
                    "Flux LoRA registry still has outstanding refs after construction"
                        .into(),
                )
            })?
            .into_inner()
            .map_err(|_| {
                candle_core::Error::Msg(
                    "Flux LoRA registry RwLock poisoned at construction".into(),
                )
            })?;
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_blocks,
            single_blocks,
            final_layer,
            lora_registry,
        })
    }

    /// v0.15 phase 7b-3: replace the runtime LoRA stack on every
    /// affected LoraLinear at once. `specs` is path-keyed (full
    /// safetensors key including `.weight`); each entry's
    /// `row_slice` pre-pads to the registered `out_dim`.
    ///
    /// Paths not present in the registry log at debug and skip.
    /// Returns the number of slots successfully applied.
    pub fn apply_loras(
        &self,
        specs: std::collections::HashMap<String, Vec<LoraSpec>>,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<usize> {
        let mut applied = 0usize;
        for (key, slot_specs) in specs {
            let Some(entry) = self.lora_registry.get(&key) else {
                tracing::debug!(
                    target: "plakat",
                    "Flux apply_loras: no Linear registered at {key} — skipping"
                );
                continue;
            };
            let mut new_slots = Vec::<LoraSlot>::with_capacity(slot_specs.len());
            for spec in slot_specs {
                let b_padded = crate::pipelines::lora_linear::pad_b_to_out_dim(
                    &spec.b,
                    spec.row_slice,
                    entry.out_dim,
                    dtype,
                    device,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "Flux apply_loras pad_b at {key}: {e}"
                    ))
                })?;
                let a = spec.a.to_dtype(dtype)?;
                new_slots.push(LoraSlot {
                    a,
                    b: b_padded,
                    scale: spec.scale,
                });
            }
            *entry.handle.write().map_err(|_| {
                candle_core::Error::Msg(format!(
                    "Flux LoRA slot handle for {key} poisoned"
                ))
            })? = new_slots;
            applied += 1;
        }
        Ok(applied)
    }

    /// v0.15 phase 7b-3: clear every active LoRA. Resets every
    /// LoraLinear to its as-loaded weight contribution only.
    pub fn clear_all_loras(&self) -> Result<()> {
        for entry in self.lora_registry.values() {
            entry
                .handle
                .write()
                .map_err(|_| {
                    candle_core::Error::Msg("Flux LoRA slot handle poisoned".into())
                })?
                .clear();
        }
        Ok(())
    }

    /// v0.15 phase 7b-3: snapshot of registered safetensors keys.
    /// Useful for verifying the loader walked the whole model.
    pub fn registered_keys(&self) -> Vec<String> {
        self.lora_registry.keys().cloned().collect()
    }

    /// v0.15 phase 7b-3: how many LoraLinears were registered.
    pub fn n_registered_linears(&self) -> usize {
        self.lora_registry.len()
    }

    /// Standard forward — no ControlNet residuals. Same math as
    /// candle's upstream `Flux::forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_residuals(
            img, img_ids, txt, txt_ids, timesteps, y, guidance, None, None,
        )
    }

    /// Forward with optional per-block ControlNet residuals. Both
    /// residual lists are interleaved with the corresponding block
    /// loop using `i // ceil(num_blocks / num_residuals)` — this
    /// matches diffusers' FluxControlNetModel integration when the
    /// ControlNet has fewer blocks than the main transformer.
    ///
    /// `double_residuals` — per-block residual tensors for the
    ///                       DoubleStream loop. Each tensor shape
    ///                       matches `img` after `img_in`:
    ///                       `(B, img_seq_len, hidden_size)`. The
    ///                       residual is added to `img` after each
    ///                       block runs.
    /// `single_residuals` — per-block residual tensors for the
    ///                       SingleStream loop. Each tensor shape
    ///                       matches the concatenated `[txt, img]`
    ///                       hidden state's image tail (the residual
    ///                       is added only to the img tail, not txt).
    ///
    /// Both `None` reproduces upstream `forward` byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_residuals(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        double_residuals: Option<&[Tensor]>,
        single_residuals: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            candle_core::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            candle_core::bail!("unexpected shape for img {:?}", img.shape())
        }
        let dtype = img.dtype();
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;
        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // Double-block residual interleave step. For the typical
        // 19-double-block Flux + 5-controlnet-block setup this lands
        // residual i at main-block i*4.
        let double_interval = match double_residuals {
            Some(r) if !r.is_empty() => {
                ((self.double_blocks.len() + r.len() - 1) / r.len()).max(1)
            }
            _ => 1,
        };
        for (i, block) in self.double_blocks.iter().enumerate() {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?;
            if let Some(residuals) = double_residuals {
                let idx = i / double_interval;
                if idx < residuals.len() {
                    img = (&img + &residuals[idx])?;
                }
            }
        }

        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.dim(1)?;
        let single_interval = match single_residuals {
            Some(r) if !r.is_empty() => {
                ((self.single_blocks.len() + r.len() - 1) / r.len()).max(1)
            }
            _ => 1,
        };
        for (i, block) in self.single_blocks.iter().enumerate() {
            img = block.forward(&img, &vec_, &pe)?;
            if let Some(residuals) = single_residuals {
                let idx = i / single_interval;
                if idx < residuals.len() {
                    // Single-block residuals only touch the image
                    // tail of the concat'd hidden state. txt tokens
                    // at indices 0..txt_len pass through unchanged.
                    let img_tail = img.narrow(1, txt_len, img.dim(1)? - txt_len)?;
                    let img_tail_updated = (img_tail + &residuals[idx])?;
                    img = Tensor::cat(
                        &[&img.narrow(1, 0, txt_len)?, &img_tail_updated],
                        1,
                    )?;
                }
            }
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}

/// `WithForward` trait impl — keeps the candle `sampling::denoise`
/// drop-in by matching the upstream signature.
impl candle_transformers::models::flux::WithForward for Flux {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        Self::forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance)
    }
}

#[cfg(test)]
mod bf16_lora_tests {
    use super::*;
    use candle_core::Module;
    use std::collections::HashMap;

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    /// Build a 2x2 LoraLinear with zero base weight via the wrap_linear
    /// helper. The VarMap-backed VarBuilder lets us construct a Linear
    /// with controlled init (zeros) without needing real safetensors.
    fn zero_wrapped(prefix: &str) -> (LoraLinear, Arc<RwLock<LoraRegistry>>) {
        let vmap = candle_nn::VarMap::new();
        vmap.get(
            (2, 2),
            &format!("{prefix}.weight"),
            candle_nn::Init::Const(0.0),
            DType::F32,
            &cpu(),
        )
        .unwrap();
        vmap.get(
            (2,),
            &format!("{prefix}.bias"),
            candle_nn::Init::Const(0.0),
            DType::F32,
            &cpu(),
        )
        .unwrap();
        let vb = VarBuilder::from_varmap(&vmap, DType::F32, &cpu());
        let registry = Arc::new(RwLock::new(LoraRegistry::new()));
        let ll = wrap_linear(2, 2, vb.pp(prefix), &registry).unwrap();
        (ll, registry)
    }

    #[test]
    fn wrap_linear_registers_at_full_path() {
        let (_ll, reg) = zero_wrapped("some.layer");
        let map = reg.read().unwrap();
        // The wrap helper registers at `<prefix>.weight`.
        assert!(map.contains_key("some.layer.weight"));
        let entry = &map["some.layer.weight"];
        assert_eq!(entry.out_dim, 2);
        assert_eq!(entry.in_dim, 2);
    }

    #[test]
    fn wrapped_linear_passes_through_when_no_lora() {
        // Base is zero; empty LoRA stack → forward(x) = 0.
        let (ll, _reg) = zero_wrapped("test");
        let x = Tensor::ones((1, 2), DType::F32, &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(yv, vec![0.0, 0.0]);
    }

    #[test]
    fn registry_drives_runtime_lora_via_handle() {
        // Apply identity LoRA via the registry handle (mimicking what
        // Flux::apply_loras does internally).
        let (ll, reg) = zero_wrapped("test");
        let id = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0],
            (2, 2),
            &cpu(),
        )
        .unwrap();
        let entry = reg.read().unwrap()["test.weight"].clone();
        *entry.handle.write().unwrap() = vec![LoraSlot {
            a: id.clone(),
            b: id.clone(),
            scale: 1.0,
        }];
        let x = Tensor::from_vec(vec![3.0f32, 7.0], (1, 2), &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // 0 (base) + 1 * I @ I @ [3, 7] = [3, 7].
        assert!((yv[0] - 3.0).abs() < 1e-5);
        assert!((yv[1] - 7.0).abs() < 1e-5);
    }
}
