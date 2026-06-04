//! Vendored copy of candle-transformers 0.8.4's
//! `flux::quantized_model`, extended with a residual-aware forward
//! that mirrors phase 2a's BF16 hook (`flux_inner.rs`).
//!
//! Same motivation as the BF16 vendor: upstream marks every block
//! constructor and forward private, so a wrapper that injects
//! ControlNet residuals between DoubleStream blocks has to vendor
//! the model in full. ~460 lines of mostly mechanical copy.
//!
//! Differences from `flux_inner.rs`:
//!   * Linear layers come from `candle_transformers::quantized_nn`
//!     instead of `candle_nn`. The math is the same; the storage is
//!     4-bit (or whichever quant the GGUF supplied).
//!   * QkNorm dequantizes the RmsNorm scales at load time —
//!     normalisation operates on already-dequantized activations, so
//!     the scale parameters don't need to stay quantized.
//!   * Shared helpers (`Config`, `EmbedNd`, `attention`,
//!     `timestep_embedding`) come from `flux_inner` so we don't
//!     duplicate them.
//!
//! Added on top:
//!   * `Flux::forward_with_residuals` — variant of the standard
//!     forward that accepts optional per-DoubleStream / per-SingleStream
//!     residual lists, identical interleave semantics to the BF16
//!     vendor. Both `None` reproduces upstream byte-for-byte.

use candle_core::quantized::QMatMul;
use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Module, RmsNorm};
use candle_transformers::quantized_var_builder::VarBuilder;
use std::collections::HashMap;
use std::sync::Arc;

use crate::pipelines::flux_inner::{attention, timestep_embedding, Config, EmbedNd};

/// Local quantized Linear that wraps a bare `candle_core::quantized::QMatMul`
/// — the upstream `quantized_nn::Linear` wraps a tracing-wrapper `QMatMul`
/// whose only public constructor takes a `QTensor`, which is incompatible
/// with LoRA-merged dense weights. The bare `QMatMul::Tensor(dense)` variant
/// runs as a regular matmul; `QMatMul::QTensor(arc)` runs as 4-bit dequant.
/// Forward and storage cost are identical to upstream when the weight is
/// quantized; LoRA-targeted layers carry a dense tensor (BF16) instead.
///
/// v0.15 phase 7b-4: gains a runtime LoRA stack (`slots`). Forward
/// applies `y += scale · (B @ A @ x)` per slot after the base matmul,
/// matching `LoraLinear`'s math. Composes with the load-time dense
/// override path (v0.13 phase 1e): when both are set, the override
/// is the "base" the runtime LoRA adds onto.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: QMatMul,
    bias: Option<Tensor>,
    out_dim: usize,
    in_dim: usize,
    /// Runtime LoRA stack — empty by default; updated in-place via
    /// the handle from `slots_handle()`. The handle is also what the
    /// loader registers in the shared `LoraRegistry` keyed by
    /// `<path>.weight` so the parent `Flux::apply_loras` can update
    /// slots by safetensors path without walking the model.
    slots: std::sync::Arc<
        std::sync::RwLock<Vec<crate::pipelines::lora_linear::LoraSlot>>,
    >,
}

impl Linear {
    pub fn new(weight: QMatMul, bias: Option<Tensor>, out_dim: usize, in_dim: usize) -> Self {
        Self {
            weight,
            bias,
            out_dim,
            in_dim,
            slots: std::sync::Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    /// v0.15 phase 7b-4: cheap Arc handle to this Linear's runtime
    /// LoRA stack. Used by `LinearLoader` to register the path in
    /// the shared registry.
    pub fn slots_handle(
        &self,
    ) -> std::sync::Arc<
        std::sync::RwLock<Vec<crate::pipelines::lora_linear::LoraSlot>>,
    > {
        self.slots.clone()
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base matmul: QMatMul handles QTensor (dequant per call) or
        // Tensor (dense, the LoRA-merged override path).
        let mut y = x.apply(&self.weight)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        // v0.15 phase 7b-4: runtime LoRA stack. Same math as
        // `LoraLinear::forward`: y += scale · (B @ A @ x) per slot.
        // Empty stack short-circuits via the iterator (zero overhead
        // when no LoRAs are active).
        let slots = self.slots.read().map_err(|_| {
            candle_core::Error::Msg("Flux GGUF Linear slots poisoned".into())
        })?;
        for slot in slots.iter() {
            let lo = x.broadcast_matmul(&slot.a.t()?)?;
            let delta = lo.broadcast_matmul(&slot.b.t()?)?;
            let delta = (delta * slot.scale as f64)?;
            y = y.broadcast_add(&delta)?;
        }
        Ok(y)
    }
}

/// Thin wrapper around `VarBuilder` that tracks the current namespace
/// path as a `String` and carries an optional LoRA-override map.
///
/// Why this exists: the upstream `quantized_var_builder::VarBuilder`
/// keeps its accumulated path private (`Vec<String>` field, no public
/// getter). The vendored quantized Flux has to know its full path at
/// each `linear()` call site so it can look up LoRA-merged dense
/// overrides — so we track the path alongside the `VarBuilder` and
/// keep them in sync via [`LinearLoader::pp`].
///
/// `overrides` maps the **full base tensor path including `.weight`**
/// (e.g. `"double_blocks.0.img_attn.qkv.weight"`) to a dense BF16
/// tensor pre-merged with all applicable LoRA deltas. When a Linear's
/// path is present in the map, the loader substitutes the dense
/// tensor via `QMatMul::Tensor(...)` instead of loading the GGUF's
/// 4-bit storage — preserving the standard `Linear` type while making
/// the LoRA-targeted layers run as if they were dense BF16.
///
/// An `Arc<HashMap>` lets every nested constructor cheaply share a
/// reference without lifetime annotations propagating across the
/// module.
#[derive(Clone)]
pub struct LinearLoader {
    pub vb: VarBuilder,
    pub path: String,
    pub overrides: Arc<HashMap<String, Tensor>>,
    /// v0.15 phase 7b-4: shared LoRA registry — every constructed
    /// `Linear` registers its slots handle under `<path>.weight`.
    /// `Arc<RwLock<...>>` so the chain of `pp()` clones all write
    /// into the same map. After construction, the parent `Flux`
    /// `try_unwrap`s the Arc and stores the inner HashMap.
    pub slot_registry: Arc<
        std::sync::RwLock<crate::pipelines::lora_linear::LoraRegistry>,
    >,
}

impl LinearLoader {
    pub fn new(vb: VarBuilder, overrides: Arc<HashMap<String, Tensor>>) -> Self {
        Self::with_registry(
            vb,
            overrides,
            Arc::new(std::sync::RwLock::new(
                crate::pipelines::lora_linear::LoraRegistry::new(),
            )),
        )
    }

    /// v0.15 phase 7b-4: build a loader that captures every
    /// constructed `Linear`'s slot handle into `slot_registry`.
    /// Cloned across nested `pp` calls so every sub-loader writes
    /// into the same registry.
    pub fn with_registry(
        vb: VarBuilder,
        overrides: Arc<HashMap<String, Tensor>>,
        slot_registry: Arc<
            std::sync::RwLock<crate::pipelines::lora_linear::LoraRegistry>,
        >,
    ) -> Self {
        Self {
            vb,
            path: String::new(),
            overrides,
            slot_registry,
        }
    }

    pub fn pp(&self, name: impl ToString) -> Self {
        let name = name.to_string();
        let path = if self.path.is_empty() {
            name.clone()
        } else {
            format!("{}.{name}", self.path)
        };
        Self {
            vb: self.vb.pp(&name),
            path,
            overrides: self.overrides.clone(),
            slot_registry: self.slot_registry.clone(),
        }
    }

    pub fn device(&self) -> &candle_core::Device {
        self.vb.device()
    }

    /// Build a quantized Linear, honoring LoRA overrides. If
    /// `<self.path>.weight` is in the overrides map, the merged dense
    /// tensor is wrapped as `QMatMul::Tensor` and used as the Linear's
    /// weight — at runtime this is a regular dense matmul, no dequant
    /// per step. Bias (when present) is dequantized from the GGUF as
    /// usual.
    fn linear_b(&self, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
        let bias_t = if bias {
            Some(self.vb.get(out_dim, "bias")?.dequantize(self.vb.device())?)
        } else {
            None
        };
        let weight_path = format!("{}.weight", self.path);
        let linear = if let Some(merged) = self.overrides.get(&weight_path) {
            // Sanity check shape — protects against a LoRA targeting a
            // mismatched layer (e.g. SD-family LoRA fed at a Flux model).
            let want = (out_dim, in_dim);
            let got = merged.dims2().map_err(|e| {
                candle_core::Error::Msg(format!(
                    "LoRA override {weight_path}: expected 2D dense tensor: {e}"
                ))
            })?;
            if got != want {
                candle_core::bail!(
                    "LoRA override {weight_path}: shape mismatch (got {got:?}, expected {want:?})"
                );
            }
            Linear::new(QMatMul::Tensor(merged.clone()), bias_t, out_dim, in_dim)
        } else {
            let weight = self.vb.get((out_dim, in_dim), "weight")?;
            let qmm = QMatMul::from_arc(weight)?;
            Linear::new(qmm, bias_t, out_dim, in_dim)
        };
        // v0.15 phase 7b-4: register the slots handle so the parent
        // Flux can drive `apply_loras` without walking the model.
        self.slot_registry
            .write()
            .map_err(|_| {
                candle_core::Error::Msg(
                    "Flux GGUF LoRA registry poisoned during construction".into(),
                )
            })?
            .insert(
                weight_path,
                crate::pipelines::lora_linear::LoraRegistryEntry {
                    handle: linear.slots_handle(),
                    out_dim,
                    in_dim,
                },
            );
        Ok(linear)
    }

    fn linear(&self, in_dim: usize, out_dim: usize) -> Result<Linear> {
        self.linear_b(in_dim, out_dim, true)
    }
}

fn layer_norm(dim: usize, device: &candle_core::Device) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, DType::F32, device)?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    pub fn new(in_sz: usize, h_sz: usize, loader: &LinearLoader) -> Result<Self> {
        let in_layer = loader.pp("in_layer").linear(in_sz, h_sz)?;
        let out_layer = loader.pp("out_layer").linear(h_sz, h_sz)?;
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
    pub fn new(dim: usize, loader: &LinearLoader) -> Result<Self> {
        // QkNorm scales are RMSNorm weights, not Linear weights — LoRAs
        // never target them, so we load straight from the GGUF (no
        // override lookup needed).
        let query_norm = loader
            .vb
            .get(dim, "query_norm.scale")?
            .dequantize(loader.device())?;
        let query_norm = RmsNorm::new(query_norm, 1e-6);
        let key_norm = loader
            .vb
            .get(dim, "key_norm.scale")?
            .dequantize(loader.device())?;
        let key_norm = RmsNorm::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
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
    lin: Linear,
}

impl Modulation1 {
    fn new(dim: usize, loader: &LinearLoader) -> Result<Self> {
        let lin = loader.pp("lin").linear(dim, 3 * dim)?;
        Ok(Self { lin })
    }
    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
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
struct Modulation2 {
    lin: Linear,
}

impl Modulation2 {
    fn new(dim: usize, loader: &LinearLoader) -> Result<Self> {
        let lin = loader.pp("lin").linear(dim, 6 * dim)?;
        Ok(Self { lin })
    }
    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
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

#[derive(Debug, Clone)]
pub struct SelfAttention {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_heads: usize,
}

impl SelfAttention {
    pub fn new(dim: usize, num_heads: usize, qkv_bias: bool, loader: &LinearLoader) -> Result<Self> {
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
    lin1: Linear,
    lin2: Linear,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, loader: &LinearLoader) -> Result<Self> {
        let lin1 = loader.pp("0").linear(in_sz, mlp_sz)?;
        let lin2 = loader.pp("2").linear(mlp_sz, in_sz)?;
        Ok(Self { lin1, lin2 })
    }
}

impl candle_core::Module for Mlp {
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
    pub fn new(cfg: &Config, loader: &LinearLoader) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let dev = loader.device();
        let img_mod = Modulation2::new(h_sz, &loader.pp("img_mod"))?;
        let img_norm1 = layer_norm(h_sz, dev)?;
        let img_attn = SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, &loader.pp("img_attn"))?;
        let img_norm2 = layer_norm(h_sz, dev)?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, &loader.pp("img_mlp"))?;
        let txt_mod = Modulation2::new(h_sz, &loader.pp("txt_mod"))?;
        let txt_norm1 = layer_norm(h_sz, dev)?;
        let txt_attn = SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, &loader.pp("txt_attn"))?;
        let txt_norm2 = layer_norm(h_sz, dev)?;
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
    linear1: Linear,
    linear2: Linear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    pub fn new(cfg: &Config, loader: &LinearLoader) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_heads;
        let dev = loader.device();
        let linear1 = loader.pp("linear1").linear(h_sz, h_sz * 3 + mlp_sz)?;
        let linear2 = loader.pp("linear2").linear(h_sz + mlp_sz, h_sz)?;
        let norm = QkNorm::new(head_dim, &loader.pp("norm"))?;
        let pre_norm = layer_norm(h_sz, dev)?;
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
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl LastLayer {
    pub fn new(h_sz: usize, p_sz: usize, out_c: usize, loader: &LinearLoader) -> Result<Self> {
        let norm_final = layer_norm(h_sz, loader.device())?;
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
    img_in: Linear,
    txt_in: Linear,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    pub double_blocks: Vec<DoubleStreamBlock>,
    pub single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
    /// v0.15 phase 7b-4: path → Linear slots handle map populated
    /// during construction. Consumed by `apply_loras` at scenario
    /// per-task dispatch time.
    lora_registry: crate::pipelines::lora_linear::LoraRegistry,
}

impl Flux {
    /// Load a quantized Flux from a GGUF VarBuilder with no LoRA
    /// overrides. Every Linear is 4-bit storage (or whatever the GGUF
    /// shipped), dequantized on the fly by `QMatMul::QTensor` forward.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_loras(cfg, vb, Arc::new(HashMap::new()))
    }

    /// Load a quantized Flux, substituting LoRA-merged dense tensors
    /// for any Linear whose path appears in `overrides` (keyed by full
    /// path including `.weight`). The substituted Linears run as
    /// regular dense matmul; everything else stays quantized.
    ///
    /// Call site responsibility: build `overrides` via
    /// `crate::pipelines::flux_lora::precompute_quantized_overrides`,
    /// which dequantizes only the LoRA-affected tensors and applies
    /// deltas with the right row-slice math.
    pub fn new_with_loras(
        cfg: &Config,
        vb: VarBuilder,
        overrides: Arc<HashMap<String, Tensor>>,
    ) -> Result<Self> {
        // v0.15 phase 7b-4: shared LoRA registry — every constructed
        // Linear writes its slot handle into this map. After all
        // sub-loaders go out of scope at the end of construction, we
        // unwrap the Arc and move the inner HashMap into Flux.
        let registry_arc = Arc::new(std::sync::RwLock::new(
            crate::pipelines::lora_linear::LoraRegistry::new(),
        ));
        let root = LinearLoader::with_registry(vb, overrides, registry_arc.clone());
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
        // Drop the loader chain so the registry Arc count returns to 1.
        drop(d_root);
        drop(s_root);
        drop(root);
        let lora_registry = Arc::try_unwrap(registry_arc)
            .map_err(|_| {
                candle_core::Error::Msg(
                    "Flux GGUF LoRA registry still has outstanding refs after construction"
                        .into(),
                )
            })?
            .into_inner()
            .map_err(|_| {
                candle_core::Error::Msg(
                    "Flux GGUF LoRA registry RwLock poisoned at construction".into(),
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

    /// v0.15 phase 7b-4: replace the runtime LoRA stack on every
    /// affected Linear at once. Same shape as the NF4 and BF16
    /// versions — path-keyed dispatch, pre-pads LoRA-B matrices to
    /// the registered `out_dim`. Returns the number of slots
    /// successfully applied.
    pub fn apply_loras(
        &self,
        specs: std::collections::HashMap<
            String,
            Vec<crate::pipelines::lora_linear::LoraSpec>,
        >,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<usize> {
        let mut applied = 0usize;
        for (key, slot_specs) in specs {
            let Some(entry) = self.lora_registry.get(&key) else {
                tracing::debug!(
                    target: "plakat",
                    "Flux GGUF apply_loras: no Linear registered at {key} — skipping"
                );
                continue;
            };
            let mut new_slots = Vec::<crate::pipelines::lora_linear::LoraSlot>::with_capacity(
                slot_specs.len(),
            );
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
                        "Flux GGUF apply_loras pad_b at {key}: {e}"
                    ))
                })?;
                let a = spec.a.to_dtype(dtype)?;
                new_slots.push(crate::pipelines::lora_linear::LoraSlot {
                    a,
                    b: b_padded,
                    scale: spec.scale,
                });
            }
            *entry.handle.write().map_err(|_| {
                candle_core::Error::Msg(format!(
                    "Flux GGUF LoRA slot handle for {key} poisoned"
                ))
            })? = new_slots;
            applied += 1;
        }
        Ok(applied)
    }

    /// v0.15 phase 7b-4: clear every active LoRA. Resets every Linear
    /// to its as-loaded contribution (GGUF dequant or BF16 dense
    /// override from the v0.13 phase 1e merge).
    pub fn clear_all_loras(&self) -> Result<()> {
        for entry in self.lora_registry.values() {
            entry
                .handle
                .write()
                .map_err(|_| {
                    candle_core::Error::Msg(
                        "Flux GGUF LoRA slot handle poisoned".into(),
                    )
                })?
                .clear();
        }
        Ok(())
    }

    /// v0.15 phase 7b-4: snapshot of registered safetensors keys.
    pub fn registered_keys(&self) -> Vec<String> {
        self.lora_registry.keys().cloned().collect()
    }

    /// v0.15 phase 7b-4: how many Linears were registered.
    pub fn n_registered_linears(&self) -> usize {
        self.lora_registry.len()
    }

    /// Standard forward — no ControlNet residuals.
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

    /// Forward with optional per-block ControlNet residuals — same
    /// signature + interleave semantics as the BF16 vendor's hook
    /// (`flux_inner::Flux::forward_with_residuals`). Both `None`
    /// reproduces upstream byte-for-byte.
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
        // v0.42: run the quantized transformer body in F32. GGUF tensors
        // stored full-precision dequantize to F32 *dense* weights
        // (candle `QMatMul::from_arc`), and candle's Metal quantized
        // matmul asserts F32 input — so on Metal (BF16 model) both the
        // dense-weight matmul and the quantized op mismatch the BF16
        // activations. F32 activations are the canonical candle
        // quantized-model dtype and make every path consistent. No-op on
        // CPU/CUDA (already F32). NOTE: this fixes the dtype CRASH; a
        // separate candle bug in the Metal mat×mat quantized kernel still
        // corrupts the output, which is why GGUF Flux is gated off on
        // Metal at the pipeline boundary (see `flux::run`). Keeping this
        // means the path is correct the moment candle fixes that kernel.
        let out_dtype = img.dtype();
        let dtype = DType::F32;
        let img = img.to_dtype(dtype)?;
        let txt = txt.to_dtype(dtype)?;
        let y = y.to_dtype(dtype)?;
        let timesteps = timesteps.to_dtype(dtype)?;
        let guidance = guidance.map(|g| g.to_dtype(dtype)).transpose()?;
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?.to_dtype(dtype)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;
        let vec_ = timestep_embedding(&timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance.as_ref()) {
            (Some(g_in), Some(g)) => {
                (vec_ + timestep_embedding(g, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // DoubleStream residual interleave — same `ceil(blocks/residuals)`
        // step the BF16 vendor uses so a ControlNet trained against
        // either backbone composes consistently.
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
                    img = (&img + &residuals[idx].to_dtype(dtype)?)?;
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
                    let img_tail = img.narrow(1, txt_len, img.dim(1)? - txt_len)?;
                    let img_tail_updated = (img_tail + &residuals[idx].to_dtype(dtype)?)?;
                    img = Tensor::cat(
                        &[&img.narrow(1, 0, txt_len)?, &img_tail_updated],
                        1,
                    )?;
                }
            }
        }
        let img = img.i((.., txt.dim(1)?..))?;
        // Cast back to the pipeline's working dtype (BF16 on Metal); the
        // F32 body above stays internal to the quantized transformer.
        self.final_layer.forward(&img, &vec_)?.to_dtype(out_dtype)
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

#[cfg(test)]
mod gguf_lora_tests {
    use super::*;
    use crate::pipelines::lora_linear::LoraSlot;

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    /// Build a 2x2 GGUF Linear with an identity dense override (so the
    /// base path is a regular matmul through `QMatMul::Tensor`). No
    /// real GGUF file needed.
    fn identity_quantized_linear_2x2() -> Linear {
        let dense = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0],
            (2, 2),
            &cpu(),
        )
        .unwrap();
        Linear::new(QMatMul::Tensor(dense), None, 2, 2)
    }

    #[test]
    fn gguf_linear_empty_stack_passes_through() {
        // Identity base, no LoRAs → forward(x) = x.
        let lin = identity_quantized_linear_2x2();
        let x = Tensor::from_vec(vec![2.0f32, 9.0], (1, 2), &cpu()).unwrap();
        let y = lin.forward(&x).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yv[0] - 2.0).abs() < 1e-5);
        assert!((yv[1] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn gguf_linear_with_identity_lora_doubles_output() {
        // Identity base + identity LoRA scale=1 → y = x + x = 2x.
        let lin = identity_quantized_linear_2x2();
        let id =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        *lin.slots_handle().write().unwrap() = vec![LoraSlot {
            a: id.clone(),
            b: id.clone(),
            scale: 1.0,
        }];
        let x = Tensor::from_vec(vec![2.0f32, 9.0], (1, 2), &cpu()).unwrap();
        let y = lin.forward(&x).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yv[0] - 4.0).abs() < 1e-5);
        assert!((yv[1] - 18.0).abs() < 1e-5);
    }

    #[test]
    fn gguf_linear_clear_returns_to_base() {
        let lin = identity_quantized_linear_2x2();
        let id =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        *lin.slots_handle().write().unwrap() = vec![LoraSlot {
            a: id.clone(),
            b: id.clone(),
            scale: 1.0,
        }];
        lin.slots_handle().write().unwrap().clear();
        let x = Tensor::from_vec(vec![2.0f32, 9.0], (1, 2), &cpu()).unwrap();
        let y = lin.forward(&x).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yv[0] - 2.0).abs() < 1e-5);
        assert!((yv[1] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn gguf_linear_two_loras_compose() {
        // Two identity LoRAs scale=0.5 → delta = x; y = x + x = 2x.
        let lin = identity_quantized_linear_2x2();
        let id =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        *lin.slots_handle().write().unwrap() = vec![
            LoraSlot {
                a: id.clone(),
                b: id.clone(),
                scale: 0.5,
            },
            LoraSlot {
                a: id.clone(),
                b: id.clone(),
                scale: 0.5,
            },
        ];
        let x = Tensor::from_vec(vec![3.0f32, 6.0], (1, 2), &cpu()).unwrap();
        let y = lin.forward(&x).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yv[0] - 6.0).abs() < 1e-5);
        assert!((yv[1] - 12.0).abs() < 1e-5);
    }
}

