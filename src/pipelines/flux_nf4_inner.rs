//! Vendored Flux backbone with NF4-quantized Linears (v0.14 phase 2c).
//!
//! Parallel to `flux_quantized_inner.rs` (the GGUF vendor) but every
//! Linear is replaced with [`NF4Linear`], which keeps the weight
//! NF4-packed at rest (4-bit, ~6 GB for the Flux transformer) and
//! dequantizes to BF16 on every forward call. Slower than GGUF Q4
//! (no kernel-fused dequant+matmul) but 4× the weight-memory savings
//! versus BF16 and works on any candle device.
//!
//! Forward shape and math match the BF16 vendor (`flux_inner.rs`)
//! byte-for-byte modulo the per-forward dequantization. ControlNet
//! residual hooks are NOT wired in this phase — `forward` is the
//! standard no-residuals path. Phase 2d's pipeline integration bails
//! loud on `--control-spec` + NF4.

use anyhow::Result as AnyResult;
use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{LayerNorm, Module, RmsNorm};

use crate::pipelines::flux_inner::{
    attention, timestep_embedding, Config, EmbedNd,
};
use crate::pipelines::nf4_codec::{NF4_BLOCK_SIZE, dequant_nf4};
use crate::pipelines::nf4_loader::Nf4Store;

/// NF4-quantized Linear. Stores the packed bytes + per-block scales
/// and recomputes the dense weight on every forward.
///
/// Per-call dequant cost: one extra `out * in` tensor allocation per
/// forward at the runtime dtype. For Flux's largest fused Linear
/// (single-block linear1 at out = 3*3072 + 12288 = 21504, in = 3072)
/// that's ~133 MB BF16 per forward. Cumulative GC pressure is the
/// main perf cost vs GGUF's fused dequant.
#[derive(Debug, Clone)]
pub struct NF4Linear {
    /// `(out * in / 2,)` packed `u8` codes.
    packed: Tensor,
    /// `(out * in / NF4_BLOCK_SIZE,)` `f32` per-block absmax.
    absmax: Tensor,
    /// Dense bias (no quantization).
    bias: Option<Tensor>,
    out_dim: usize,
    in_dim: usize,
    /// Runtime dtype the dequantized weight casts to before matmul
    /// (typically BF16 on GPU, F32 on CPU).
    out_dtype: DType,
}

impl NF4Linear {
    pub fn new(
        packed: Tensor,
        absmax: Tensor,
        bias: Option<Tensor>,
        out_dim: usize,
        in_dim: usize,
        out_dtype: DType,
    ) -> Self {
        Self {
            packed,
            absmax,
            bias,
            out_dim,
            in_dim,
            out_dtype,
        }
    }

    fn dequant_weight(&self) -> Result<Tensor> {
        let device = self.packed.device();
        let f32_w = dequant_nf4(
            &self.packed,
            &self.absmax,
            &[self.out_dim, self.in_dim],
            NF4_BLOCK_SIZE,
            device,
        )
        .map_err(|e| candle_core::Error::Msg(format!("NF4 dequant: {e}")))?;
        f32_w.to_dtype(self.out_dtype)
    }
}

impl Module for NF4Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.dequant_weight()?;
        let y = x.broadcast_matmul(&w.t()?)?;
        match &self.bias {
            None => Ok(y),
            Some(b) => y.broadcast_add(b),
        }
    }
}

/// Path-tracking wrapper around an [`Nf4Store`]. Each `pp(name)`
/// extends the namespace path the next `linear()` / `linear_b()`
/// call resolves against. Mirrors `LinearLoader` from the GGUF
/// vendor; the substantive difference is the loader doesn't carry
/// LoRA overrides (NF4 + LoRA composition is a follow-up).
#[derive(Clone)]
pub struct Nf4LinearLoader<'a> {
    pub store: &'a Nf4Store,
    pub path: String,
    pub dtype: DType,
}

impl<'a> Nf4LinearLoader<'a> {
    pub fn new(store: &'a Nf4Store, dtype: DType) -> Self {
        Self {
            store,
            path: String::new(),
            dtype,
        }
    }

    pub fn pp(&self, name: impl ToString) -> Self {
        let name = name.to_string();
        let path = if self.path.is_empty() {
            name
        } else {
            format!("{}.{name}", self.path)
        };
        Self {
            store: self.store,
            path,
            dtype: self.dtype,
        }
    }

    pub fn device(&self) -> &Device {
        self.store.device()
    }

    /// Load a Linear at `<self.path>`. Looks up `<path>.weight` (NF4
    /// packed), `<path>.weight.absmax`, and optionally `<path>.bias`.
    fn linear_b(&self, in_dim: usize, out_dim: usize, bias: bool) -> AnyResult<NF4Linear> {
        let weight_path = format!("{}.weight", self.path);
        let absmax_path = format!("{}.absmax", weight_path);
        let packed = self.store.get(&weight_path)?;
        let absmax = self.store.get(&absmax_path)?;
        let numel = out_dim * in_dim;
        // Sanity check sizes — fail at construction rather than at
        // first forward.
        if packed.elem_count() * 2 != numel {
            anyhow::bail!(
                "NF4 Linear {weight_path}: packed has {} bytes, expected {} (out={}, in={})",
                packed.elem_count(),
                numel / 2,
                out_dim,
                in_dim
            );
        }
        if absmax.elem_count() * NF4_BLOCK_SIZE != numel {
            anyhow::bail!(
                "NF4 Linear {weight_path}: absmax has {} entries, expected {} (block_size {})",
                absmax.elem_count(),
                numel / NF4_BLOCK_SIZE,
                NF4_BLOCK_SIZE
            );
        }
        let bias_t = if bias {
            let bias_path = format!("{}.bias", self.path);
            Some(self.store.get(&bias_path)?.to_dtype(self.dtype)?)
        } else {
            None
        };
        Ok(NF4Linear::new(packed, absmax, bias_t, out_dim, in_dim, self.dtype))
    }

    fn linear(&self, in_dim: usize, out_dim: usize) -> AnyResult<NF4Linear> {
        self.linear_b(in_dim, out_dim, true)
    }

    /// Read a dense (non-NF4) tensor — for LayerNorm / QkNorm scales
    /// and similar. Returns the tensor cast to the loader's runtime
    /// dtype.
    fn dense(&self, leaf: &str) -> AnyResult<Tensor> {
        let path = if self.path.is_empty() {
            leaf.to_string()
        } else {
            format!("{}.{leaf}", self.path)
        };
        Ok(self.store.get(&path)?.to_dtype(self.dtype)?)
    }
}

fn layer_norm_no_weights(dim: usize, dtype: DType, device: &Device) -> Result<LayerNorm> {
    // Flux's LayerNorms have no learnable scale/bias — the "no-bias"
    // variant uses an all-ones implicit scale.
    let ws = Tensor::ones(dim, dtype, device)?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: NF4Linear,
    out_layer: NF4Linear,
}

impl MlpEmbedder {
    pub fn new(in_sz: usize, h_sz: usize, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        let in_layer = loader.pp("in_layer").linear(in_sz, h_sz)?;
        let out_layer = loader.pp("out_layer").linear(h_sz, h_sz)?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }
}

impl Module for MlpEmbedder {
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
    pub fn new(dim: usize, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        // RmsNorm scales are dense (small tensors, not worth quantizing
        // in bnb's recipe). Loaded directly from the store at the
        // runtime dtype.
        let q_scale = loader.pp("query_norm").dense("scale")?;
        let k_scale = loader.pp("key_norm").dense("scale")?;
        let _ = dim;
        Ok(Self {
            query_norm: RmsNorm::new(q_scale, 1e-6),
            key_norm: RmsNorm::new(k_scale, 1e-6),
        })
    }
}

#[derive(Debug, Clone)]
struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }
    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1 {
    lin: NF4Linear,
}

impl Modulation1 {
    fn new(dim: usize, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        let lin = loader.pp("lin").linear(dim, 3 * dim)?;
        Ok(Self { lin })
    }
    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_.silu()?.apply(&self.lin)?.unsqueeze(1)?.chunk(3, D::Minus1)?;
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
struct Modulation2 {
    lin: NF4Linear,
}

impl Modulation2 {
    fn new(dim: usize, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        let lin = loader.pp("lin").linear(dim, 6 * dim)?;
        Ok(Self { lin })
    }
    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_.silu()?.apply(&self.lin)?.unsqueeze(1)?.chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        let m1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let m2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((m1, m2))
    }
}

#[derive(Debug, Clone)]
pub struct SelfAttention {
    qkv: NF4Linear,
    norm: QkNorm,
    proj: NF4Linear,
    num_heads: usize,
}

impl SelfAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        loader: &Nf4LinearLoader,
    ) -> AnyResult<Self> {
        let head_dim = dim / num_heads;
        let qkv = loader.pp("qkv").linear_b(dim, dim * 3, qkv_bias)?;
        let norm = QkNorm::new(head_dim, &loader.pp("norm"))?;
        let proj = loader.pp("proj").linear(dim, dim)?;
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
struct Mlp {
    lin1: NF4Linear,
    lin2: NF4Linear,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        let lin1 = loader.pp("0").linear(in_sz, mlp_sz)?;
        let lin2 = loader.pp("2").linear(mlp_sz, in_sz)?;
        Ok(Self { lin1, lin2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

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
    pub fn new(cfg: &Config, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let dev = loader.device();
        let dtype = loader.dtype;
        let img_mod = Modulation2::new(h_sz, &loader.pp("img_mod"))?;
        let img_norm1 = layer_norm_no_weights(h_sz, dtype, dev)?;
        let img_attn =
            SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, &loader.pp("img_attn"))?;
        let img_norm2 = layer_norm_no_weights(h_sz, dtype, dev)?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, &loader.pp("img_mlp"))?;
        let txt_mod = Modulation2::new(h_sz, &loader.pp("txt_mod"))?;
        let txt_norm1 = layer_norm_no_weights(h_sz, dtype, dev)?;
        let txt_attn =
            SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, &loader.pp("txt_attn"))?;
        let txt_norm2 = layer_norm_no_weights(h_sz, dtype, dev)?;
        let txt_mlp = Mlp::new(h_sz, mlp_sz, &loader.pp("txt_mlp"))?;
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
    linear1: NF4Linear,
    linear2: NF4Linear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    pub fn new(cfg: &Config, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_heads;
        let dev = loader.device();
        let dtype = loader.dtype;
        let linear1 = loader.pp("linear1").linear(h_sz, h_sz * 3 + mlp_sz)?;
        let linear2 = loader.pp("linear2").linear(h_sz + mlp_sz, h_sz)?;
        let norm = QkNorm::new(head_dim, &loader.pp("norm"))?;
        let pre_norm = layer_norm_no_weights(h_sz, dtype, dev)?;
        let modulation = Modulation1::new(h_sz, &loader.pp("modulation"))?;
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
    linear: NF4Linear,
    ada_ln_modulation: NF4Linear,
}

impl LastLayer {
    pub fn new(h_sz: usize, p_sz: usize, out_c: usize, loader: &Nf4LinearLoader) -> AnyResult<Self> {
        let dev = loader.device();
        let dtype = loader.dtype;
        let norm_final = layer_norm_no_weights(h_sz, dtype, dev)?;
        let linear_ = loader.pp("linear").linear(h_sz, p_sz * p_sz * out_c)?;
        let ada_ln_modulation = loader.pp("adaLN_modulation.1").linear(h_sz, 2 * h_sz)?;
        Ok(Self {
            norm_final,
            linear: linear_,
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

#[derive(Debug, Clone)]
pub struct Flux {
    img_in: NF4Linear,
    txt_in: NF4Linear,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    pub double_blocks: Vec<DoubleStreamBlock>,
    pub single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
}

impl Flux {
    pub fn new(cfg: &Config, store: &Nf4Store, dtype: DType) -> AnyResult<Self> {
        let root = Nf4LinearLoader::new(store, dtype);
        let img_in = root.pp("img_in").linear(cfg.in_channels, cfg.hidden_size)?;
        let txt_in = root.pp("txt_in").linear(cfg.context_in_dim, cfg.hidden_size)?;
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let d_root = root.pp("double_blocks");
        for idx in 0..cfg.depth {
            double_blocks.push(DoubleStreamBlock::new(cfg, &d_root.pp(idx))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let s_root = root.pp("single_blocks");
        for idx in 0..cfg.depth_single_blocks {
            single_blocks.push(SingleStreamBlock::new(cfg, &s_root.pp(idx))?);
        }
        let time_in = MlpEmbedder::new(256, cfg.hidden_size, &root.pp("time_in"))?;
        let vector_in = MlpEmbedder::new(cfg.vec_in_dim, cfg.hidden_size, &root.pp("vector_in"))?;
        let guidance_in = if cfg.guidance_embed {
            Some(MlpEmbedder::new(256, cfg.hidden_size, &root.pp("guidance_in"))?)
        } else {
            None
        };
        let final_layer =
            LastLayer::new(cfg.hidden_size, 1, cfg.in_channels, &root.pp("final_layer"))?;
        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.theta, cfg.axes_dim.to_vec());
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
        })
    }

    /// Standard forward — no ControlNet residuals (composition with
    /// CN is deferred for the NF4 backbone).
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
            (Some(g_in), Some(g)) => (vec_ + timestep_embedding(g, 256, dtype)?.apply(g_in))?,
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        for block in self.double_blocks.iter() {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?;
        }
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in self.single_blocks.iter() {
            img = block.forward(&img, &vec_, &pe)?;
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}

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
