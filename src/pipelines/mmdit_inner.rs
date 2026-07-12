//! Vendored MMDiT (SD3 / SD3.5) with optional per-block ControlNet
//! residual injection — v0.15 phase 6a.
//!
//! This module is a copy of `candle_transformers::models::mmdit` with
//! one substantive addition: `forward_with_residuals(...)` on both
//! `MMDiT` and `MMDiTCore`. When residuals are passed, they're added
//! to the joint-block output `x` stream after each block, using the
//! same `ceil(blocks/residuals)` interleave the BFL / GGUF / NF4 Flux
//! vendors expose. With residuals=None the forward is byte-identical
//! to candle's upstream — pre-phase-6 SD3 behaviour preserved.
//!
//! The vendoring follows the same pattern as `flux_inner.rs`
//! (v0.13 phase 1c): the upstream model doesn't expose internal
//! block iteration, so we copy + augment rather than fighting the
//! type system to inject behaviour from outside.
//!
//! ## What gets injected
//!
//! `joint_blocks[i].forward(context, x, c) -> (context, x)` produces
//! two streams; the residual is added to the `x` stream only
//! (the context branch carries text-conditioning state that the CN
//! model isn't trained to perturb). This matches SD3 ControlNet
//! conventions in diffusers + the InstantX reference implementation.
//!
//! The final `context_qkv_only_joint_block` (the last block, which
//! collapses the context branch and produces only `x`) does **not**
//! receive residuals — SD3 CN checkpoints don't ship per-block weights
//! that far out, and the diffusers reference stops the residual loop
//! at `depth - 1`. We mirror that.

use candle_core::{Device, Module, Result, Tensor, Var, D, bail, DType};
use candle_nn as nn;
// v0.15 phase 7b-5: every Linear in the vendored MMDiT becomes a
// `LoraLinear` so the model can apply a runtime LoRA stack at forward
// time. Stack starts empty (byte-identical to nn::Linear) and updates
// via `MMDiT::apply_loras` at scenario per-task dispatch time.
use crate::pipelines::lora_linear::{
    LoraLinear, LoraRegistry, LoraRegistryEntry, LoraSlot, LoraSpec,
};
use std::sync::{Arc, RwLock};


/// v0.15 phase 7b-5: wrap a candle Linear, register the slots handle
/// in `<vb.prefix()>.weight` of the shared LoRA registry, return the
/// LoraLinear ready to plug into a struct field. Same pattern as the
/// helper in `flux_inner`.
fn wrap_linear(
    in_dim: usize,
    out_dim: usize,
    vb: nn::VarBuilder,
    registry: &Arc<RwLock<LoraRegistry>>,
) -> Result<LoraLinear> {
    let base = nn::linear(in_dim, out_dim, vb.clone())?;
    let ll = LoraLinear::from_linear(base).map_err(|e| {
        candle_core::Error::Msg(format!("MMDiT wrap_linear at {}: {e}", vb.prefix()))
    })?;
    let key = format!("{}.weight", vb.prefix());
    registry
        .write()
        .map_err(|_| {
            candle_core::Error::Msg("MMDiT LoRA registry poisoned during construction".into())
        })?
        .insert(
            key,
            LoraRegistryEntry {
                handle: ll.slots_handle(),
                out_dim,
                in_dim,
                train: ll.train_handle(),
            },
        );
    Ok(ll)
}

// =====================================================================
// embedding.rs — copied verbatim from candle.
// =====================================================================

pub struct PatchEmbedder {
    proj: nn::Conv2d,
}

impl PatchEmbedder {
    pub fn new(
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let proj = nn::conv2d(
            in_channels,
            embed_dim,
            patch_size,
            nn::Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;

        Ok(Self { proj })
    }
}

impl Module for PatchEmbedder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?;
        let (b, c, h, w) = x.dims4()?;
        x.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

pub struct Unpatchifier {
    patch_size: usize,
    out_channels: usize,
}

impl Unpatchifier {
    pub fn new(patch_size: usize, out_channels: usize) -> Result<Self> {
        Ok(Self {
            patch_size,
            out_channels,
        })
    }

    pub fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let h = (h + 1) / self.patch_size;
        let w = (w + 1) / self.patch_size;
        let x = x.reshape((
            x.dim(0)?,
            h,
            w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        ))?;
        let x = x.permute((0, 5, 1, 3, 2, 4))?;
        x.reshape((
            x.dim(0)?,
            self.out_channels,
            self.patch_size * h,
            self.patch_size * w,
        ))
    }
}

pub struct PositionEmbedder {
    pos_embed: Tensor,
    patch_size: usize,
    pos_embed_max_size: usize,
}

impl PositionEmbedder {
    pub fn new(
        hidden_size: usize,
        patch_size: usize,
        pos_embed_max_size: usize,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let pos_embed = vb.get(
            (1, pos_embed_max_size * pos_embed_max_size, hidden_size),
            "pos_embed",
        )?;
        Ok(Self {
            pos_embed,
            patch_size,
            pos_embed_max_size,
        })
    }
    pub fn get_cropped_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let h = (h + 1) / self.patch_size;
        let w = (w + 1) / self.patch_size;
        if h > self.pos_embed_max_size || w > self.pos_embed_max_size {
            bail!("Input size is too large for the position embedding")
        }
        let top = (self.pos_embed_max_size - h) / 2;
        let left = (self.pos_embed_max_size - w) / 2;
        let pos_embed =
            self.pos_embed
                .reshape((1, self.pos_embed_max_size, self.pos_embed_max_size, ()))?;
        let pos_embed = pos_embed.narrow(1, top, h)?.narrow(2, left, w)?;
        pos_embed.reshape((1, h * w, ()))
    }
}

pub struct TimestepEmbedder {
    // v0.15 phase 7b-5: split the `nn::Sequential` into named
    // `LoraLinear` fields so each can register its slot handle in
    // the LoRA registry. Forward chains them manually with SiLU.
    mlp_0: LoraLinear,
    mlp_2: LoraLinear,
    frequency_embedding_size: usize,
    // The model dtype (BF16 on Metal/CUDA for SD3, F32 on CPU). The
    // sinusoidal embedding must match the MLP weight dtype — hardcoding
    // F16 mismatches BF16 weights (this only surfaced once SD3 actually
    // loaded and ran).
    dtype: DType,
}

impl TimestepEmbedder {
    pub fn new(
        hidden_size: usize,
        frequency_embedding_size: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let mlp_0 = wrap_linear(
            frequency_embedding_size,
            hidden_size,
            vb.pp("mlp.0"),
            registry,
        )?;
        let mlp_2 = wrap_linear(hidden_size, hidden_size, vb.pp("mlp.2"), registry)?;
        Ok(Self {
            mlp_0,
            mlp_2,
            frequency_embedding_size,
            dtype,
        })
    }

    fn timestep_embedding(t: &Tensor, dim: usize, max_period: f64, dtype: DType) -> Result<Tensor> {
        if dim % 2 != 0 {
            bail!("Embedding dimension must be even")
        }
        if t.dtype() != DType::F32 && t.dtype() != DType::F64 {
            bail!("Input tensor must be floating point")
        }
        let half = dim / 2;
        let freqs = Tensor::arange(0f32, half as f32, t.device())?
            .to_dtype(DType::F32)?
            .mul(&Tensor::full(
                (-f64::ln(max_period) / half as f64) as f32,
                half,
                t.device(),
            )?)?
            .exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .matmul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
        embedding.to_dtype(dtype)
    }
}

impl Module for TimestepEmbedder {
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_freq =
            Self::timestep_embedding(t, self.frequency_embedding_size, 10000.0, self.dtype)?;
        // Manual mlp.0 → SiLU → mlp.2 chain (replaces nn::Sequential).
        t_freq.apply(&self.mlp_0)?.silu()?.apply(&self.mlp_2)
    }
}

pub struct VectorEmbedder {
    // v0.15 phase 7b-5: same Sequential-to-explicit-fields refactor
    // as TimestepEmbedder so each Linear can register in the LoRA
    // registry.
    mlp_0: LoraLinear,
    mlp_2: LoraLinear,
}

impl VectorEmbedder {
    pub fn new(
        input_dim: usize,
        hidden_size: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let mlp_0 = wrap_linear(input_dim, hidden_size, vb.pp("mlp.0"), registry)?;
        let mlp_2 = wrap_linear(hidden_size, hidden_size, vb.pp("mlp.2"), registry)?;
        Ok(Self { mlp_0, mlp_2 })
    }
}

impl Module for VectorEmbedder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.mlp_0)?.silu()?.apply(&self.mlp_2)
    }
}

// =====================================================================
// projections.rs — copied verbatim from candle.
// =====================================================================

pub struct Qkv {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

pub struct Mlp {
    fc1: LoraLinear,
    act: nn::Activation,
    fc2: LoraLinear,
}

impl Mlp {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        vb: candle_nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let fc1 = wrap_linear(in_features, hidden_features, vb.pp("fc1"), registry)?;
        let act = nn::Activation::GeluPytorchTanh;
        let fc2 = wrap_linear(hidden_features, in_features, vb.pp("fc2"), registry)?;
        Ok(Self { fc1, act, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

pub struct QkvOnlyAttnProjections {
    qkv: LoraLinear,
    head_dim: usize,
    // SD3.5 applies QK-norm (RMSNorm) to the context Q/K even in the
    // final context-qkv-only block. Without it the context Q/K stay
    // un-normalized → blown-up attention scores → a catastrophic outlier
    // that the final LayerNorm then propagates to the whole output.
    // Auto-detected (absent on the original SD3, present on SD3.5).
    ln_k: Option<candle_nn::RmsNorm>,
    ln_q: Option<candle_nn::RmsNorm>,
}

impl QkvOnlyAttnProjections {
    pub fn new(
        dim: usize,
        num_heads: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = wrap_linear(dim, dim * 3, vb.pp("qkv"), registry)?;
        let (ln_k, ln_q) = if vb.contains_tensor("ln_k.weight") {
            (
                Some(candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_k"))?),
                Some(candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_q"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self { qkv, head_dim, ln_k, ln_q })
    }

    pub fn pre_attention(&self, x: &Tensor) -> Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        let Qkv { q, k, v } = split_qkv(&qkv, self.head_dim)?;
        let norm = |t: Tensor, ln: Option<&candle_nn::RmsNorm>| -> Result<Tensor> {
            match ln {
                None => Ok(t),
                Some(l) => {
                    let (b, s, h) = t.dims3()?;
                    Ok(l.forward(&t.reshape((b, s, (), self.head_dim))?)?.reshape((b, s, h))?)
                }
            }
        };
        let q = norm(q, self.ln_q.as_ref())?;
        let k = norm(k, self.ln_k.as_ref())?;
        Ok(Qkv { q, k, v })
    }
}

pub struct AttnProjections {
    head_dim: usize,
    qkv: LoraLinear,
    ln_k: Option<candle_nn::RmsNorm>,
    ln_q: Option<candle_nn::RmsNorm>,
    proj: LoraLinear,
}

impl AttnProjections {
    pub fn new(
        dim: usize,
        num_heads: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = wrap_linear(dim, dim * 3, vb.pp("qkv"), registry)?;
        let proj = wrap_linear(dim, dim, vb.pp("proj"), registry)?;
        let (ln_k, ln_q) = if vb.contains_tensor("ln_k.weight") {
            let ln_k = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_k"))?;
            let ln_q = candle_nn::rms_norm(head_dim, 1e-6, vb.pp("ln_q"))?;
            (Some(ln_k), Some(ln_q))
        } else {
            (None, None)
        };
        Ok(Self {
            head_dim,
            qkv,
            proj,
            ln_k,
            ln_q,
        })
    }

    pub fn pre_attention(&self, x: &Tensor) -> Result<Qkv> {
        let qkv = self.qkv.forward(x)?;
        let Qkv { q, k, v } = split_qkv(&qkv, self.head_dim)?;
        let q = match self.ln_q.as_ref() {
            None => q,
            Some(l) => {
                let (b, t, h) = q.dims3()?;
                l.forward(&q.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        let k = match self.ln_k.as_ref() {
            None => k,
            Some(l) => {
                let (b, t, h) = k.dims3()?;
                l.forward(&k.reshape((b, t, (), self.head_dim))?)?
                    .reshape((b, t, h))?
            }
        };
        Ok(Qkv { q, k, v })
    }

    pub fn post_attention(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

fn split_qkv(qkv: &Tensor, head_dim: usize) -> Result<Qkv> {
    let (batch_size, seq_len, _) = qkv.dims3()?;
    let qkv = qkv.reshape((batch_size, seq_len, 3, (), head_dim))?;
    let q = qkv.get_on_dim(2, 0)?;
    let q = q.reshape((batch_size, seq_len, ()))?;
    let k = qkv.get_on_dim(2, 1)?;
    let k = k.reshape((batch_size, seq_len, ()))?;
    let v = qkv.get_on_dim(2, 2)?;
    Ok(Qkv { q, k, v })
}

// =====================================================================
// blocks.rs — copied verbatim, with `pub` on items the CN model in
// v0.16's phase 6b will need to instantiate independently.
// =====================================================================

pub struct ModulateIntermediates {
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
}

pub struct DiTBlock {
    norm1: LayerNormNoAffine,
    attn: AttnProjections,
    norm2: LayerNormNoAffine,
    mlp: Mlp,
    // v0.15 phase 7b-5: split the `nn::Sequential` SiLU + Linear into
    // explicit fields so the `adaLN_modulation.1` Linear can register
    // in the LoRA registry. SD3 PEFT LoRAs target it as
    // `norm1.linear`.
    ada_ln_modulation_1: LoraLinear,
}

pub struct LayerNormNoAffine {
    eps: f64,
}

impl LayerNormNoAffine {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoAffine {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        nn::LayerNorm::new_no_bias(Tensor::ones_like(x)?, self.eps).forward(x)
    }
}

impl DiTBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = AttnProjections::new(hidden_size, num_heads, vb.pp("attn"), registry)?;
        let norm2 = LayerNormNoAffine::new(1e-6);
        let mlp_ratio = 4;
        let mlp = Mlp::new(
            hidden_size,
            hidden_size * mlp_ratio,
            vb.pp("mlp"),
            registry,
        )?;
        let n_mods = 6;
        let ada_ln_modulation_1 = wrap_linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
            registry,
        )?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            ada_ln_modulation_1,
        })
    }

    pub fn pre_attention(&self, x: &Tensor, c: &Tensor) -> Result<(Qkv, ModulateIntermediates)> {
        // Manual SiLU + Linear (replaces nn::Sequential).
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(6, D::Minus1)?;
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            chunks[0].clone(),
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
            chunks[4].clone(),
            chunks[5].clone(),
        );
        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        let qkv = self.attn.pre_attention(&modulated_x)?;
        Ok((
            qkv,
            ModulateIntermediates {
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            },
        ))
    }

    pub fn post_attention(
        &self,
        attn: &Tensor,
        x: &Tensor,
        mod_interm: &ModulateIntermediates,
    ) -> Result<Tensor> {
        let attn_out = self.attn.post_attention(attn)?;
        let x = x.add(&attn_out.broadcast_mul(&mod_interm.gate_msa.unsqueeze(1)?)?)?;
        let norm_x = self.norm2.forward(&x)?;
        let modulated_x = modulate(&norm_x, &mod_interm.shift_mlp, &mod_interm.scale_mlp)?;
        let mlp_out = self.mlp.forward(&modulated_x)?;
        let x = x.add(&mlp_out.broadcast_mul(&mod_interm.gate_mlp.unsqueeze(1)?)?)?;
        Ok(x)
    }
}

pub struct SelfAttnModulateIntermediates {
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
    gate_msa2: Tensor,
}

pub struct SelfAttnDiTBlock {
    norm1: LayerNormNoAffine,
    attn: AttnProjections,
    attn2: AttnProjections,
    norm2: LayerNormNoAffine,
    mlp: Mlp,
    ada_ln_modulation_1: LoraLinear,
}

impl SelfAttnDiTBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = AttnProjections::new(hidden_size, num_heads, vb.pp("attn"), registry)?;
        let attn2 = AttnProjections::new(hidden_size, num_heads, vb.pp("attn2"), registry)?;
        let norm2 = LayerNormNoAffine::new(1e-6);
        let mlp_ratio = 4;
        let mlp = Mlp::new(
            hidden_size,
            hidden_size * mlp_ratio,
            vb.pp("mlp"),
            registry,
        )?;
        let n_mods = 9;
        let ada_ln_modulation_1 = wrap_linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
            registry,
        )?;
        Ok(Self {
            norm1,
            attn,
            attn2,
            norm2,
            mlp,
            ada_ln_modulation_1,
        })
    }

    pub fn pre_attention(
        &self,
        x: &Tensor,
        c: &Tensor,
    ) -> Result<(Qkv, Qkv, SelfAttnModulateIntermediates)> {
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(9, D::Minus1)?;
        let (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = (
            chunks[0].clone(),
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
            chunks[4].clone(),
            chunks[5].clone(),
            chunks[6].clone(),
            chunks[7].clone(),
            chunks[8].clone(),
        );
        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        let qkv = self.attn.pre_attention(&modulated_x)?;
        let modulated_x2 = modulate(&norm_x, &shift_msa2, &scale_msa2)?;
        let qkv2 = self.attn2.pre_attention(&modulated_x2)?;
        Ok((
            qkv,
            qkv2,
            SelfAttnModulateIntermediates {
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                gate_msa2,
            },
        ))
    }

    pub fn post_attention(
        &self,
        attn: &Tensor,
        attn2: &Tensor,
        x: &Tensor,
        mod_interm: &SelfAttnModulateIntermediates,
    ) -> Result<Tensor> {
        let attn_out = self.attn.post_attention(attn)?;
        let x = x.add(&attn_out.broadcast_mul(&mod_interm.gate_msa.unsqueeze(1)?)?)?;
        let attn_out2 = self.attn2.post_attention(attn2)?;
        let x = x.add(&attn_out2.broadcast_mul(&mod_interm.gate_msa2.unsqueeze(1)?)?)?;
        let norm_x = self.norm2.forward(&x)?;
        let modulated_x = modulate(&norm_x, &mod_interm.shift_mlp, &mod_interm.scale_mlp)?;
        let mlp_out = self.mlp.forward(&modulated_x)?;
        let x = x.add(&mlp_out.broadcast_mul(&mod_interm.gate_mlp.unsqueeze(1)?)?)?;
        Ok(x)
    }
}

pub struct QkvOnlyDiTBlock {
    norm1: LayerNormNoAffine,
    attn: QkvOnlyAttnProjections,
    ada_ln_modulation_1: LoraLinear,
}

impl QkvOnlyDiTBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = QkvOnlyAttnProjections::new(hidden_size, num_heads, vb.pp("attn"), registry)?;
        let n_mods = 2;
        let ada_ln_modulation_1 = wrap_linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
            registry,
        )?;
        Ok(Self {
            norm1,
            attn,
            ada_ln_modulation_1,
        })
    }

    pub fn pre_attention(&self, x: &Tensor, c: &Tensor) -> Result<Qkv> {
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        // diffusers' context_pre_only block uses AdaLayerNormContinuous,
        // whose linear emits (scale, shift) — NOT AdaLayerNormZero's
        // (shift, scale). Reading them swapped corrupts the context K/V
        // for the final joint attention.
        let (scale_msa, shift_msa) = (chunks[0].clone(), chunks[1].clone());
        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        self.attn.pre_attention(&modulated_x)
    }
}

pub struct FinalLayer {
    norm_final: LayerNormNoAffine,
    linear: LoraLinear,
    ada_ln_modulation_1: LoraLinear,
}

impl FinalLayer {
    pub fn new(
        hidden_size: usize,
        patch_size: usize,
        out_channels: usize,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let norm_final = LayerNormNoAffine::new(1e-6);
        let linear = wrap_linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            vb.pp("linear"),
            registry,
        )?;
        let ada_ln_modulation_1 = wrap_linear(
            hidden_size,
            2 * hidden_size,
            vb.pp("adaLN_modulation.1"),
            registry,
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation_1,
        })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let modulation = c.silu()?.apply(&self.ada_ln_modulation_1)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        // diffusers' final norm is AdaLayerNormContinuous → (scale, shift),
        // not (shift, scale).
        let (scale, shift) = (chunks[0].clone(), chunks[1].clone());
        let norm_x = self.norm_final.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift, &scale)?;
        self.linear.forward(&modulated_x)
    }
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let shift = shift.unsqueeze(1)?;
    let scale = scale.unsqueeze(1)?;
    let scale_plus_one = scale.add(&Tensor::ones_like(&scale)?)?;
    shift.broadcast_add(&x.broadcast_mul(&scale_plus_one)?)
}

// v0.22 phase 3: Send + Sync bounds added so sd3::Pipeline can be
// held in the scripting OnceLock<RwLock<...>> cache. The concrete
// MMDiTJointBlock + MMDiTSelfAttnJointBlock impls only contain
// `candle_nn::Linear` / `LayerNorm` etc which are already
// Send + Sync; the trait just needed to advertise the bound.
pub trait JointBlock: Send + Sync {
    /// `pag = true` perturbs the **image (x) stream** self-attention to identity (attention
    /// output = V — the degenerate pass PAG guides away from). The context stream keeps its real
    /// joint attention. Mirrors the PixArt DiT PAG perturbation.
    fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor, pag: bool) -> Result<(Tensor, Tensor)>;
}

pub struct MMDiTJointBlock {
    x_block: DiTBlock,
    context_block: DiTBlock,
    num_heads: usize,
    use_flash_attn: bool,
}

impl MMDiTJointBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let x_block =
            DiTBlock::new(hidden_size, num_heads, vb.pp("x_block"), registry)?;
        let context_block =
            DiTBlock::new(hidden_size, num_heads, vb.pp("context_block"), registry)?;
        Ok(Self {
            x_block,
            context_block,
            num_heads,
            use_flash_attn,
        })
    }
}

impl JointBlock for MMDiTJointBlock {
    fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor, pag: bool) -> Result<(Tensor, Tensor)> {
        let (context_qkv, context_interm) = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_interm) = self.x_block.pre_attention(x, c)?;
        let (context_attn, x_attn) =
            joint_attn(&context_qkv, &x_qkv, self.num_heads, self.use_flash_attn)?;
        // PAG: replace the image self-attention output with its own V (attention matrix = I).
        // `v` is 4D (b,seq,heads,head_dim) out of split_qkv → flatten heads back to (b,seq,hidden),
        // the layout `attn()` returns and `post_attention`'s proj expects.
        let x_attn = if pag { x_qkv.v.flatten_from(2)?.contiguous()? } else { x_attn };
        let context_out =
            self.context_block
                .post_attention(&context_attn, context, &context_interm)?;
        let x_out = self.x_block.post_attention(&x_attn, x, &x_interm)?;
        Ok((context_out, x_out))
    }
}

pub struct MMDiTXJointBlock {
    x_block: SelfAttnDiTBlock,
    context_block: DiTBlock,
    num_heads: usize,
    use_flash_attn: bool,
}

impl MMDiTXJointBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let x_block =
            SelfAttnDiTBlock::new(hidden_size, num_heads, vb.pp("x_block"), registry)?;
        let context_block =
            DiTBlock::new(hidden_size, num_heads, vb.pp("context_block"), registry)?;
        Ok(Self {
            x_block,
            context_block,
            num_heads,
            use_flash_attn,
        })
    }
}

impl JointBlock for MMDiTXJointBlock {
    fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor, pag: bool) -> Result<(Tensor, Tensor)> {
        let (context_qkv, context_interm) = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_qkv2, x_interm) = self.x_block.pre_attention(x, c)?;
        let (context_attn, x_attn) =
            joint_attn(&context_qkv, &x_qkv, self.num_heads, self.use_flash_attn)?;
        let x_attn2 = attn(&x_qkv2, self.num_heads, self.use_flash_attn)?;
        // PAG: perturb BOTH the joint and the second (x-only) image self-attentions to identity.
        let (x_attn, x_attn2) = if pag {
            (x_qkv.v.flatten_from(2)?.contiguous()?, x_qkv2.v.flatten_from(2)?.contiguous()?)
        } else {
            (x_attn, x_attn2)
        };
        let context_out =
            self.context_block
                .post_attention(&context_attn, context, &context_interm)?;
        let x_out = self
            .x_block
            .post_attention(&x_attn, &x_attn2, x, &x_interm)?;
        Ok((context_out, x_out))
    }
}

pub struct ContextQkvOnlyJointBlock {
    x_block: DiTBlock,
    context_block: QkvOnlyDiTBlock,
    num_heads: usize,
    use_flash_attn: bool,
}

impl ContextQkvOnlyJointBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let x_block =
            DiTBlock::new(hidden_size, num_heads, vb.pp("x_block"), registry)?;
        let context_block = QkvOnlyDiTBlock::new(
            hidden_size, num_heads, vb.pp("context_block"), registry,
        )?;
        Ok(Self {
            x_block,
            context_block,
            num_heads,
            use_flash_attn,
        })
    }

    pub fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor, pag: bool) -> Result<Tensor> {
        let context_qkv = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_interm) = self.x_block.pre_attention(x, c)?;
        let (_, x_attn) = joint_attn(&context_qkv, &x_qkv, self.num_heads, self.use_flash_attn)?;
        // PAG: perturb the image self-attention to identity (output = V, heads flattened).
        let x_attn = if pag { x_qkv.v.flatten_from(2)?.contiguous()? } else { x_attn };
        self.x_block.post_attention(&x_attn, x, &x_interm)
    }
}

fn flash_compatible_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let q_dims_for_matmul = q.transpose(1, 2)?.dims().to_vec();
    let rank = q_dims_for_matmul.len();
    let q = q.transpose(1, 2)?.flatten_to(rank - 3)?;
    let k = k.transpose(1, 2)?.flatten_to(rank - 3)?;
    let v = v.transpose(1, 2)?.flatten_to(rank - 3)?;
    let attn_weights = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    attn_scores.reshape(q_dims_for_matmul)?.transpose(1, 2)
}

// v0.15 phase 6a: plakat doesn't declare a `flash-attn` Cargo feature
// (it would pull in `candle-flash-attn` + CUDA build prerequisites).
// We always take the candle-native softmax path; `use_flash_attn` is
// accepted on the JointBlock constructors for upstream-shape parity
// but is effectively dead. If a future cycle wants real flash-attn,
// add the feature + plug it in here.
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("flash-attn not enabled in plakat; use flash_compatible_attention")
}

fn joint_attn(
    context_qkv: &Qkv,
    x_qkv: &Qkv,
    num_heads: usize,
    use_flash_attn: bool,
) -> Result<(Tensor, Tensor)> {
    let qkv = Qkv {
        q: Tensor::cat(&[&context_qkv.q, &x_qkv.q], 1)?,
        k: Tensor::cat(&[&context_qkv.k, &x_qkv.k], 1)?,
        v: Tensor::cat(&[&context_qkv.v, &x_qkv.v], 1)?,
    };
    let seqlen = qkv.q.dim(1)?;
    let attn = attn(&qkv, num_heads, use_flash_attn)?;
    let context_qkv_seqlen = context_qkv.q.dim(1)?;
    let context_attn = attn.narrow(1, 0, context_qkv_seqlen)?;
    let x_attn = attn.narrow(1, context_qkv_seqlen, seqlen - context_qkv_seqlen)?;
    Ok((context_attn, x_attn))
}

fn attn(qkv: &Qkv, num_heads: usize, use_flash_attn: bool) -> Result<Tensor> {
    let batch_size = qkv.q.dim(0)?;
    let seqlen = qkv.q.dim(1)?;
    // v2.4: fused SDPA fast path for MMDiT joint attention (unmasked — SD3's joint attention
    // takes no mask). candle's Metal SDPA kernel is ~16× faster than eager and matches it to
    // ~1e-6 (probe). GPU-only — candle SDPA has NO CPU impl, so CPU + the verify harness keep
    // the exact eager path below. `use_flash_attn` (CUDA flash-attn) keeps priority. Guarded on
    // the kernel's supported head-dim set. Escape hatch: PLAKAT_NO_SDPA=1.
    let hidden = qkv.q.dim(2)?;
    let head_dim = hidden / num_heads;
    let sdpa_ok = !use_flash_attn
        && (qkv.q.device().is_metal() || qkv.q.device().is_cuda())
        && std::env::var("PLAKAT_NO_SDPA").is_err()
        && [32, 64, 72, 80, 96, 128, 256, 512].contains(&head_dim);
    if sdpa_ok {
        let to_bhsd = |t: &Tensor| -> Result<Tensor> {
            t.reshape((batch_size, seqlen, num_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (to_bhsd(&qkv.q)?, to_bhsd(&qkv.k)?, to_bhsd(&qkv.v)?);
        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?; // (b,h,s,d)
        return out.transpose(1, 2)?.reshape((batch_size, seqlen, hidden));
    }
    let qkv = Qkv {
        q: qkv.q.reshape((batch_size, seqlen, num_heads, ()))?,
        k: qkv.k.reshape((batch_size, seqlen, num_heads, ()))?,
        v: qkv.v.clone(),
    };
    let headdim = qkv.q.dim(D::Minus1)?;
    let softmax_scale = 1.0 / (headdim as f64).sqrt();
    let attn = if use_flash_attn {
        flash_attn(&qkv.q, &qkv.k, &qkv.v, softmax_scale as f32, false)?
    } else {
        flash_compatible_attention(&qkv.q, &qkv.k, &qkv.v, softmax_scale as f32)?
    };
    attn.reshape((batch_size, seqlen, ()))
}

// =====================================================================
// model.rs — Config copied verbatim; MMDiT + MMDiTCore augmented with
// `forward_with_residuals` and constructor parity with upstream.
// =====================================================================

#[derive(Debug, Clone)]
pub struct Config {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub depth: usize,
    pub head_size: usize,
    pub adm_in_channels: usize,
    pub pos_embed_max_size: usize,
    pub context_embed_size: usize,
    pub frequency_embedding_size: usize,
}

impl Config {
    pub fn sd3_medium() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            depth: 24,
            head_size: 64,
            adm_in_channels: 2048,
            pos_embed_max_size: 192,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }

    pub fn sd3_5_medium() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            depth: 24,
            head_size: 64,
            adm_in_channels: 2048,
            pos_embed_max_size: 384,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }

    pub fn sd3_5_large() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            depth: 38,
            head_size: 64,
            adm_in_channels: 2048,
            pos_embed_max_size: 192,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }
}

pub struct MMDiT {
    core: MMDiTCore,
    patch_embedder: PatchEmbedder,
    pos_embedder: PositionEmbedder,
    timestep_embedder: TimestepEmbedder,
    vector_embedder: VectorEmbedder,
    context_embedder: LoraLinear,
    unpatchifier: Unpatchifier,
    /// v0.15 phase 7b-5: path → LoraLinear slot handle map populated
    /// during construction. Consumed by `apply_loras` at scenario
    /// per-task dispatch time so we can update slots by safetensors
    /// path without re-walking joint blocks.
    lora_registry: LoraRegistry,
}

impl MMDiT {
    pub fn new(cfg: &Config, use_flash_attn: bool, vb: nn::VarBuilder) -> Result<Self> {
        let hidden_size = cfg.head_size * cfg.depth;
        // v0.15 phase 7b-5: shared LoRA registry — every constructed
        // LoraLinear writes its slot handle into this map. After all
        // sub-loaders go out of scope at the end of construction, we
        // unwrap the Arc and move the inner HashMap into MMDiT.
        let registry_arc = Arc::new(RwLock::new(LoraRegistry::new()));
        let core = MMDiTCore::new(
            cfg.depth,
            hidden_size,
            cfg.depth,
            cfg.patch_size,
            cfg.out_channels,
            use_flash_attn,
            vb.clone(),
            &registry_arc,
        )?;
        let patch_embedder = PatchEmbedder::new(
            cfg.patch_size,
            cfg.in_channels,
            hidden_size,
            vb.pp("x_embedder"),
        )?;
        let pos_embedder = PositionEmbedder::new(
            hidden_size,
            cfg.patch_size,
            cfg.pos_embed_max_size,
            vb.clone(),
        )?;
        let timestep_embedder = TimestepEmbedder::new(
            hidden_size,
            cfg.frequency_embedding_size,
            vb.pp("t_embedder"),
            &registry_arc,
        )?;
        let vector_embedder = VectorEmbedder::new(
            cfg.adm_in_channels,
            hidden_size,
            vb.pp("y_embedder"),
            &registry_arc,
        )?;
        let context_embedder = wrap_linear(
            cfg.context_embed_size,
            hidden_size,
            vb.pp("context_embedder"),
            &registry_arc,
        )?;
        let unpatchifier = Unpatchifier::new(cfg.patch_size, cfg.out_channels)?;
        // Move the registry out of the Arc — `core` and all the sub-
        // loaders are dropped at function exit; ref count goes to 1.
        let lora_registry = Arc::try_unwrap(registry_arc)
            .map_err(|_| {
                candle_core::Error::Msg(
                    "MMDiT LoRA registry still has outstanding refs after construction"
                        .into(),
                )
            })?
            .into_inner()
            .map_err(|_| {
                candle_core::Error::Msg(
                    "MMDiT LoRA registry RwLock poisoned at construction".into(),
                )
            })?;
        Ok(Self {
            core,
            patch_embedder,
            pos_embedder,
            timestep_embedder,
            vector_embedder,
            context_embedder,
            unpatchifier,
            lora_registry,
        })
    }

    /// v0.15 phase 7b-5: replace the runtime LoRA stack on every
    /// affected LoraLinear at once. Same shape as the NF4 / BF16 /
    /// GGUF versions. Returns the number of slots successfully
    /// applied.
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
                    "MMDiT apply_loras: no Linear registered at {key} — skipping"
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
                        "MMDiT apply_loras pad_b at {key}: {e}"
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
                    "MMDiT LoRA slot handle for {key} poisoned"
                ))
            })? = new_slots;
            applied += 1;
        }
        Ok(applied)
    }

    /// v0.15 phase 7b-5: clear every active LoRA. Resets MMDiT to its
    /// as-loaded weights.
    pub fn clear_all_loras(&self) -> Result<()> {
        for entry in self.lora_registry.values() {
            entry
                .handle
                .write()
                .map_err(|_| {
                    candle_core::Error::Msg("MMDiT LoRA slot handle poisoned".into())
                })?
                .clear();
        }
        Ok(())
    }

    /// v0.15 phase 7b-5: snapshot of registered safetensors keys.
    pub fn registered_keys(&self) -> Vec<String> {
        self.lora_registry.keys().cloned().collect()
    }

    /// v0.15 phase 7b-5: how many LoraLinears were registered.
    pub fn n_registered_linears(&self) -> usize {
        self.lora_registry.len()
    }

    /// `plakat style train` (Phase 1): install a fresh **trainable** LoRA
    /// adapter on every attention projection (registry keys containing
    /// `.attn` — the joint blocks' qkv / proj for attn + attn2). Returns
    /// `(registry_key, A, B)` for each, so the caller drives AdamW and
    /// writes the kohya save. Standard init: `A ~ N(0, 0.02)`, `B = 0`, so
    /// the adapter starts as a no-op on the frozen base and learns the
    /// style delta. Vars are F32 (training dtype).
    pub fn install_train_adapters(
        &self,
        rank: usize,
        scale: f64,
        device: &Device,
    ) -> Result<Vec<(String, Var, Var)>> {
        let mut keys: Vec<&String> =
            self.lora_registry.keys().filter(|k| k.contains(".attn.")).collect();
        keys.sort();
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let entry = &self.lora_registry[key];
            let a = Var::from_tensor(&Tensor::randn(
                0f32,
                0.02f32,
                (rank, entry.in_dim),
                device,
            )?)?;
            let b = Var::from_tensor(&Tensor::zeros((entry.out_dim, rank), DType::F32, device)?)?;
            *entry
                .train
                .write()
                .map_err(|_| candle_core::Error::Msg("MMDiT train slot poisoned".into()))? =
                Some((a.clone(), b.clone(), scale));
            out.push((key.clone(), a, b));
        }
        Ok(out)
    }

    /// Standard forward — no ControlNet residuals. Byte-identical to
    /// candle's upstream `MMDiT::forward`. Delegates to
    /// `forward_with_residuals(... None)` so the no-CN path stays a
    /// single source of truth.
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        self.forward_with_residuals(x, t, y, context, skip_layers, None)
    }

    /// v0.15 phase 6a: forward with optional per-joint-block ControlNet
    /// residuals on the `x` (image) stream. `residuals.len()` typically
    /// matches `cfg.depth - 1` (one per joint_block before the final
    /// `context_qkv_only_joint_block`). For shorter residual lists we
    /// fall back to the same `ceil(blocks/residuals)` interleave the
    /// BFL / GGUF / NF4 Flux vendors use so a single CN producing N
    /// residuals composes consistently across backbones.
    ///
    /// The context stream and the final block are unchanged. Diffusers'
    /// SD3 CN reference stops the residual loop at `depth - 1`; we
    /// mirror that.
    pub fn forward_with_residuals(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
        residuals: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let h = x.dim(D::Minus2)?;
        let w = x.dim(D::Minus1)?;
        let cropped_pos_embed = self.pos_embedder.get_cropped_pos_embed(h, w)?;
        let x = self
            .patch_embedder
            .forward(x)?
            .broadcast_add(&cropped_pos_embed)?;
        let c = self.timestep_embedder.forward(t)?;
        let y = self.vector_embedder.forward(y)?;
        let c = (c + y)?;
        let context = self.context_embedder.forward(context)?;
        let x = self
            .core
            .forward_with_residuals(&context, &x, &c, skip_layers, residuals, false)?;
        let x = self.unpatchifier.unpatchify(&x, h, w)?;
        x.narrow(2, 0, h)?.narrow(3, 0, w)
    }

    /// As [`forward`](Self::forward), with **PAG**: `pag = true` perturbs every joint block's
    /// image self-attention to identity, producing the degenerate prediction PAG guides away
    /// from. The caller runs this once per step on the conditional inputs and combines the two
    /// (`guided = cfg + pag·(cond − cond_perturbed)`). No ControlNet residuals on the PAG pass.
    pub fn forward_pag(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
        pag: bool,
    ) -> Result<Tensor> {
        let h = x.dim(D::Minus2)?;
        let w = x.dim(D::Minus1)?;
        let cropped_pos_embed = self.pos_embedder.get_cropped_pos_embed(h, w)?;
        let x = self
            .patch_embedder
            .forward(x)?
            .broadcast_add(&cropped_pos_embed)?;
        let c = self.timestep_embedder.forward(t)?;
        let y = self.vector_embedder.forward(y)?;
        let c = (c + y)?;
        let context = self.context_embedder.forward(context)?;
        let x = self
            .core
            .forward_with_residuals(&context, &x, &c, skip_layers, None, pag)?;
        let x = self.unpatchifier.unpatchify(&x, h, w)?;
        x.narrow(2, 0, h)?.narrow(3, 0, w)
    }

    /// Verify tap (`plakat verify` Tier 1, `mmdit.block0`): run the embed prologue
    /// (patch+pos, timestep+vector → c, context_embedder) and the FIRST joint block,
    /// returning the **x-stream** (image tokens) `(B, tokens, hidden)`. Additive — reuses
    /// the exact prologue of [`Self::forward_with_residuals`] and `core.joint_blocks[0]`.
    /// Corresponds to a diffusers hook on `transformer.transformer_blocks[0]` (its
    /// `hidden_states` output). The dumper feeds DETERMINISTIC `y`/`context` (shared LCG),
    /// isolating the MMDiT joint-block math from the CLIP/T5 encoders.
    pub fn capture_block0(&self, x: &Tensor, t: &Tensor, y: &Tensor, context: &Tensor) -> Result<Tensor> {
        let h = x.dim(D::Minus2)?;
        let w = x.dim(D::Minus1)?;
        let cropped_pos_embed = self.pos_embedder.get_cropped_pos_embed(h, w)?;
        let x = self.patch_embedder.forward(x)?.broadcast_add(&cropped_pos_embed)?;
        let c = self.timestep_embedder.forward(t)?;
        let y = self.vector_embedder.forward(y)?;
        let c = (c + y)?;
        let context = self.context_embedder.forward(context)?;
        let (_context, x) = self.core.joint_blocks[0].forward(&context, &x, &c, false)?;
        Ok(x)
    }
}

pub struct MMDiTCore {
    joint_blocks: Vec<Box<dyn JointBlock>>,
    context_qkv_only_joint_block: ContextQkvOnlyJointBlock,
    final_layer: FinalLayer,
}

impl MMDiTCore {
    pub fn new(
        depth: usize,
        hidden_size: usize,
        num_heads: usize,
        patch_size: usize,
        out_channels: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
        registry: &Arc<RwLock<LoraRegistry>>,
    ) -> Result<Self> {
        let mut joint_blocks = Vec::with_capacity(depth - 1);
        for i in 0..depth - 1 {
            let joint_block_vb_pp = format!("joint_blocks.{}", i);
            let joint_block: Box<dyn JointBlock> =
                if vb.contains_tensor(&format!("{}.x_block.attn2.qkv.weight", joint_block_vb_pp)) {
                    Box::new(MMDiTXJointBlock::new(
                        hidden_size,
                        num_heads,
                        use_flash_attn,
                        vb.pp(&joint_block_vb_pp),
                        registry,
                    )?)
                } else {
                    Box::new(MMDiTJointBlock::new(
                        hidden_size,
                        num_heads,
                        use_flash_attn,
                        vb.pp(&joint_block_vb_pp),
                        registry,
                    )?)
                };
            joint_blocks.push(joint_block);
        }
        Ok(Self {
            joint_blocks,
            context_qkv_only_joint_block: ContextQkvOnlyJointBlock::new(
                hidden_size,
                num_heads,
                use_flash_attn,
                vb.pp(format!("joint_blocks.{}", depth - 1)),
                registry,
            )?,
            final_layer: FinalLayer::new(
                hidden_size,
                patch_size,
                out_channels,
                vb.pp("final_layer"),
                registry,
            )?,
        })
    }

    /// Standard forward — delegates to `forward_with_residuals` with
    /// no residuals. Byte-identical to candle's upstream
    /// `MMDiTCore::forward` when called this way.
    pub fn forward(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        self.forward_with_residuals(context, x, c, skip_layers, None, false)
    }

    /// v0.15 phase 6a: forward with optional per-block residuals on
    /// the `x` stream. Residuals are added after each joint block's
    /// forward call (post-attention + MLP) — same injection point
    /// diffusers' SD3 CN uses. The context stream and the final
    /// `context_qkv_only_joint_block` are unchanged.
    pub fn forward_with_residuals(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        skip_layers: Option<&[usize]>,
        residuals: Option<&[Tensor]>,
        pag: bool,
    ) -> Result<Tensor> {
        let (mut context, mut x) = (context.clone(), x.clone());
        // Residual interleave. `ceil(blocks/residuals)` step — same
        // strategy the Flux vendors use so a CN producing N residuals
        // distributes evenly across `depth - 1` joint blocks. When
        // `residuals.len() == joint_blocks.len()` (the canonical case)
        // the interval is 1 and every block sees its own residual.
        let interval = match residuals {
            Some(r) if !r.is_empty() => {
                ((self.joint_blocks.len() + r.len() - 1) / r.len()).max(1)
            }
            _ => 1,
        };
        // PAG applied-layer set. Perturbing EVERY block's image self-attention to identity makes
        // the degenerate prediction so far from the real one that any guidance scale blows up the
        // latents (black frame + patch-grid). Diffusers' SD3 PAG restricts to a small subset; we
        // default to the single middle joint block and allow `PLAKAT_PAG_LAYERS=8,12,16` to tune.
        let pag_layers: Vec<usize> = if pag {
            resolve_pag_layers(self.joint_blocks.len())
        } else {
            Vec::new()
        };
        for (i, joint_block) in self.joint_blocks.iter().enumerate() {
            if let Some(skip_layers) = &skip_layers {
                if skip_layers.contains(&i) {
                    continue;
                }
            }
            (context, x) = joint_block.forward(&context, &x, c, pag && pag_layers.contains(&i))?;
            if let Some(r) = residuals {
                let idx = i / interval;
                if idx < r.len() {
                    x = (&x + &r[idx])?;
                }
            }
        }
        // Final (context-qkv-only) block index == joint_blocks.len(); perturb only if selected.
        let final_idx = self.joint_blocks.len();
        let x = self.context_qkv_only_joint_block.forward(
            &context,
            &x,
            c,
            pag && pag_layers.contains(&final_idx),
        )?;
        self.final_layer.forward(&x, c)
    }
}

/// PAG applied-layer indices for a model with `n_joint` joint blocks (the final context-qkv-only
/// block is index `n_joint`). `PLAKAT_PAG_LAYERS` is a comma-separated override (e.g. `10,12,14`);
/// otherwise defaults to the single middle joint block — the gentlest useful perturbation.
fn resolve_pag_layers(n_joint: usize) -> Vec<usize> {
    match std::env::var("PLAKAT_PAG_LAYERS") {
        Ok(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => vec![n_joint / 2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // v0.15 phase 6a — verify the residual interleave math without
    // standing up a full MMDiT (which requires real safetensors).

    #[test]
    fn interleave_one_to_one_when_residuals_match_blocks() {
        // 24 blocks, 24 residuals → interval = 1, every block gets a
        // residual at its own index. Matches the canonical SD3 CN
        // shape (InstantX ships per-block residuals).
        let n_blocks = 24usize;
        let n_res = 24usize;
        let interval = ((n_blocks + n_res - 1) / n_res).max(1);
        assert_eq!(interval, 1);
        for i in 0..n_blocks {
            assert_eq!(i / interval, i);
        }
    }

    #[test]
    fn interleave_step_when_residuals_sparse() {
        // 24 blocks, 12 residuals → interval = 2 (every other block
        // receives the residual at idx i/2). Mirrors the Flux pattern.
        let n_blocks = 24usize;
        let n_res = 12usize;
        let interval = ((n_blocks + n_res - 1) / n_res).max(1);
        assert_eq!(interval, 2);
        // First two blocks share residual 0; next two share residual 1; ...
        assert_eq!(0 / interval, 0);
        assert_eq!(1 / interval, 0);
        assert_eq!(2 / interval, 1);
        assert_eq!(23 / interval, 11);
    }

    #[test]
    fn interleave_floor_one_when_residuals_outnumber_blocks() {
        // 8 blocks, 16 residuals → interval = 1 (we cap at 1; extras
        // beyond index `n_blocks - 1` are unused).
        let n_blocks = 8usize;
        let n_res = 16usize;
        let interval = ((n_blocks + n_res - 1) / n_res).max(1);
        assert_eq!(interval, 1);
        for i in 0..n_blocks {
            assert!(i < n_res);
        }
    }

    // v0.15 phase 7b-5 — verify the wrap_linear helper registers in the
    // shared LoRA registry and the resulting LoraLinear applies the
    // runtime stack correctly. Standing up a full MMDiT requires real
    // safetensors (depth-24 / depth-38 blocks loaded from disk); the
    // helper test covers the substantive new infrastructure here.

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

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
        let vb = nn::VarBuilder::from_varmap(&vmap, DType::F32, &cpu());
        let registry = Arc::new(RwLock::new(LoraRegistry::new()));
        let ll = wrap_linear(2, 2, vb.pp(prefix), &registry).unwrap();
        (ll, registry)
    }

    #[test]
    fn mmdit_wrap_linear_registers_at_full_path() {
        let (_ll, reg) = zero_wrapped("joint_blocks.0.x_block.attn.qkv");
        let map = reg.read().unwrap();
        assert!(map.contains_key("joint_blocks.0.x_block.attn.qkv.weight"));
        let entry = &map["joint_blocks.0.x_block.attn.qkv.weight"];
        assert_eq!(entry.out_dim, 2);
        assert_eq!(entry.in_dim, 2);
    }

    #[test]
    fn mmdit_runtime_lora_via_registry_handle() {
        // Apply identity LoRA via the registry handle (mimicking what
        // MMDiT::apply_loras does internally) and verify forward
        // adds the delta to the (zero) base output.
        let (ll, reg) = zero_wrapped("test");
        let id =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        let entry = reg.read().unwrap()["test.weight"].clone();
        *entry.handle.write().unwrap() = vec![LoraSlot {
            a: id.clone(),
            b: id.clone(),
            scale: 1.0,
        }];
        let x = Tensor::from_vec(vec![3.0f32, 7.0], (1, 2), &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yv[0] - 3.0).abs() < 1e-5);
        assert!((yv[1] - 7.0).abs() < 1e-5);
    }
}
