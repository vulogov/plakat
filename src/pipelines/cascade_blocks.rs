//! v0.39 phase 0a: Stable Cascade architectural blocks
//! (upstream-aligned).
//!
//! Replaces v0.37/v0.38's SD-style approximations with the actual
//! Würstchen v3 / Stable Cascade block primitives. The new shapes
//! were derived from the safetensors headers of:
//! - `stabilityai/stable-cascade/vqgan/diffusion_pytorch_model.safetensors` (122 tensors, Stage A)
//! - `stabilityai/stable-cascade/decoder/diffusion_pytorch_model.safetensors` (1726 tensors, Stage B)
//! - `stabilityai/stable-cascade-prior/prior/diffusion_pytorch_model.safetensors` (1550 tensors, Stage C)
//!
//! ## Primitives
//!
//! - [`LayerNorm2d`] — channel-axis LayerNorm for (B, C, H, W),
//!   `elementwise_affine=False` (no learnable params).
//! - [`GlobalResponseNorm`] — ConvNeXt-v2 GRN with learnable
//!   `β`/`γ` of shape `(1, 1, 1, C)`. Operates on channel-last
//!   `(B, H, W, C)`.
//! - [`ResBlock`] — ConvNeXt-v2 block: depthwise Conv2d →
//!   LayerNorm2d → channelwise MLP (Linear → GELU → GRN → Linear)
//!   + residual skip.
//! - [`TimestepBlock`] — FiLM scale+shift with up to three
//!   mappers (`mapper`, `mapper_sca`, `mapper_crp`). Stage B uses
//!   two (mapper + mapper_sca); Stage C uses all three.
//! - [`AttnBlock`] — single fused self+cross attention. KV stream
//!   is `cat(flatten(image), kv_mapper(text))`; Q is image-only.
//!   `kv_mapper` is a `Sequential(SiLU, Linear)` so the upstream
//!   tensor key is `kv_mapper.1.{weight,bias}`.
//!
//! ## Tensor naming
//!
//! Matches the upstream safetensors keys exactly:
//! - `depthwise.{weight,bias}` (Conv2d groups=C)
//! - `channelwise.0.{weight,bias}` (Linear C → 4C)
//! - `channelwise.2.{beta,gamma}` (GRN params, shape `(1,1,1,4C)`)
//! - `channelwise.4.{weight,bias}` (Linear 4C → C)
//! - `mapper.{weight,bias}` (Linear `time_dim → 2*C`)
//! - `mapper_sca.{weight,bias}`, `mapper_crp.{weight,bias}` (same shape)
//! - `attention.to_q.{weight,bias}`, `to_k`, `to_v`, `to_out.0.{weight,bias}`
//! - `kv_mapper.1.{weight,bias}` (Linear `cond_dim → C`)

use anyhow::{Result, anyhow};
use candle_core::{D, Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

// ---------------------------------------------------------------------
// LayerNorm2d
// ---------------------------------------------------------------------

/// Channel-axis LayerNorm for spatial `(B, C, H, W)` inputs. Matches
/// PyTorch `nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)`.
///
/// No learnable affine — upstream Stable Cascade ResBlocks use
/// `elementwise_affine=False` everywhere (confirmed by the absence
/// of `norm.weight` / `norm.bias` tensors in the safetensors keys).
pub struct LayerNorm2d {
    channels: usize,
    eps: f64,
}

impl LayerNorm2d {
    pub fn new(channels: usize, eps: f64) -> Self {
        Self { channels, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, c, _h, _w) = x.dims4()?;
        anyhow::ensure!(
            c == self.channels,
            "LayerNorm2d: channel mismatch (got {c}, expected {})",
            self.channels
        );
        // mean / var across the channel axis at each (B, H, W) site.
        let mean = x.mean_keepdim(1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(1)?;
        let denom = var.affine(1.0, self.eps)?.sqrt()?;
        Ok(x_centered.broadcast_div(&denom)?)
    }
}

// ---------------------------------------------------------------------
// GlobalResponseNorm (ConvNeXt-v2 GRN)
// ---------------------------------------------------------------------

/// ConvNeXt-v2 GRN block. Stable Cascade's `channelwise.2` in each
/// ResBlock. Operates on channel-last `(B, H, W, C)` input.
///
/// ```text
///   G(x)[b, c] = sqrt(sum_{h,w} x[b, h, w, c]^2)
///   N(x)[b, c] = G / (mean(G, dim=C) + eps)
///   y          = γ * (x * N) + β + x
/// ```
///
/// `γ` and `β` are learnable parameters of shape `(1, 1, 1, C)`
/// matching the upstream tensor shapes inspected at v0.39 phase 0.
pub struct GlobalResponseNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl GlobalResponseNorm {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb
            .get((1, 1, 1, channels), "gamma")
            .map_err(|e| anyhow!("GRN gamma: {e}"))?;
        let beta = vb
            .get((1, 1, 1, channels), "beta")
            .map_err(|e| anyhow!("GRN beta: {e}"))?;
        Ok(Self {
            gamma,
            beta,
            eps: 1e-6,
        })
    }

    /// `x`: `(B, H, W, C)` channel-last.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // L2 norm across (H, W) → (B, 1, 1, C).
        let g = x.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        // Normalize each channel by mean across channels.
        let g_mean = g.mean_keepdim(D::Minus1)?;
        let n = g.broadcast_div(&g_mean.affine(1.0, self.eps)?)?;
        // γ * (x * N) + β + x
        let x_scaled = x.broadcast_mul(&n)?;
        let gated = self
            .gamma
            .broadcast_mul(&x_scaled)?
            .broadcast_add(&self.beta)?;
        Ok(gated.add(x)?)
    }
}

// ---------------------------------------------------------------------
// ResBlock (ConvNeXt-v2 style)
// ---------------------------------------------------------------------

/// Stable Cascade ResBlock. ConvNeXt-v2 design:
/// depthwise Conv2d → LayerNorm2d (no affine) → channelwise MLP
/// (Linear → GELU → GRN → Linear) + residual skip.
///
/// Tensor keys (relative to the ResBlock's VB prefix):
///   `depthwise.{weight,bias}`
///   `channelwise.0.{weight,bias}` (Linear C → 4C)
///   `channelwise.2.{beta,gamma}`  (GRN)
///   `channelwise.4.{weight,bias}` (Linear 4C → C)
///
/// Upstream's ResBlock optionally accepts a `c_skip` concat input
/// (for cross-stage skips in the decoder); that's handled at the
/// level wrapper in v0.39 phase 0b/0c.
pub struct ResBlock {
    depthwise: nn::Conv2d,
    norm: LayerNorm2d,
    channelwise_0: nn::Linear,
    grn: GlobalResponseNorm,
    channelwise_4: nn::Linear,
    channels: usize,
}

impl ResBlock {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            groups: channels,
            ..Default::default()
        };
        let depthwise = nn::conv2d(channels, channels, 3, conv_cfg, vb.pp("depthwise"))
            .map_err(|e| anyhow!("ResBlock depthwise: {e}"))?;
        let norm = LayerNorm2d::new(channels, 1e-6);
        let channelwise_0 = nn::linear(channels, channels * 4, vb.pp("channelwise").pp("0"))
            .map_err(|e| anyhow!("ResBlock channelwise.0: {e}"))?;
        let grn = GlobalResponseNorm::new(channels * 4, vb.pp("channelwise").pp("2"))?;
        let channelwise_4 = nn::linear(channels * 4, channels, vb.pp("channelwise").pp("4"))
            .map_err(|e| anyhow!("ResBlock channelwise.4: {e}"))?;
        Ok(Self {
            depthwise,
            norm,
            channelwise_0,
            grn,
            channelwise_4,
            channels,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_res = x.clone();
        let h = self.depthwise.forward(x)?;
        let h = self.norm.forward(&h)?;
        // Permute (B, C, H, W) → (B, H, W, C) for MLP + GRN.
        let h = h.permute((0, 2, 3, 1))?.contiguous()?;
        let h = self.channelwise_0.forward(&h)?;
        let h = h.gelu()?;
        let h = self.grn.forward(&h)?;
        let h = self.channelwise_4.forward(&h)?;
        // Back to channel-first.
        let h = h.permute((0, 3, 1, 2))?.contiguous()?;
        Ok(h.add(&x_res)?)
    }
}

// ---------------------------------------------------------------------
// TimestepBlock — FiLM with up to three mappers
// ---------------------------------------------------------------------

/// FiLM-style timestep injection. Configurable to accept up to
/// three independent conditioning streams via `mapper`,
/// `mapper_sca`, and `mapper_crp` Linears, each projecting from
/// `time_dim` (typically 64) to `2 * channels` so they can produce
/// per-channel (scale, shift). Contributions are summed before
/// applying:
///
/// ```text
///   (a, b) = sum(mapper(t), mapper_sca(sca)?, mapper_crp(crp)?).chunk(2, dim=1)
///   y      = x * (1 + a) + b
/// ```
///
/// Stage B uses `(mapper, mapper_sca)`; Stage C adds `mapper_crp`.
/// Choose by setting `has_sca` / `has_crp` at construction.
pub struct TimestepBlock {
    mapper: nn::Linear,
    mapper_sca: Option<nn::Linear>,
    mapper_crp: Option<nn::Linear>,
    channels: usize,
}

impl TimestepBlock {
    pub fn new(
        channels: usize,
        time_dim: usize,
        has_sca: bool,
        has_crp: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mapper = nn::linear(time_dim, channels * 2, vb.pp("mapper"))
            .map_err(|e| anyhow!("TimestepBlock mapper: {e}"))?;
        let mapper_sca = if has_sca {
            Some(
                nn::linear(time_dim, channels * 2, vb.pp("mapper_sca"))
                    .map_err(|e| anyhow!("TimestepBlock mapper_sca: {e}"))?,
            )
        } else {
            None
        };
        let mapper_crp = if has_crp {
            Some(
                nn::linear(time_dim, channels * 2, vb.pp("mapper_crp"))
                    .map_err(|e| anyhow!("TimestepBlock mapper_crp: {e}"))?,
            )
        } else {
            None
        };
        Ok(Self {
            mapper,
            mapper_sca,
            mapper_crp,
            channels,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// `x`: `(B, C, H, W)`. `t_emb`: `(B, time_dim)`.
    /// `sca_emb`/`crp_emb`: same shape as `t_emb`; required iff the
    /// corresponding mapper was created with `has_sca` / `has_crp`.
    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, c, _h, _w) = x.dims4()?;
        anyhow::ensure!(
            c == self.channels,
            "TimestepBlock: channel mismatch (got {c}, expected {})",
            self.channels
        );

        let proj_t = self.mapper.forward(t_emb)?;
        let mut combined = proj_t;

        if let Some(m_sca) = &self.mapper_sca {
            let s = sca_emb.ok_or_else(|| {
                anyhow!("TimestepBlock has mapper_sca but no sca_emb supplied")
            })?;
            combined = combined.add(&m_sca.forward(s)?)?;
        }
        if let Some(m_crp) = &self.mapper_crp {
            let c_t = crp_emb.ok_or_else(|| {
                anyhow!("TimestepBlock has mapper_crp but no crp_emb supplied")
            })?;
            combined = combined.add(&m_crp.forward(c_t)?)?;
        }

        // Split scale (first C) and shift (last C); reshape to
        // (B, C, 1, 1) for spatial broadcasting.
        let scale = combined.narrow(D::Minus1, 0, c)?.reshape((b, c, 1, 1))?;
        let shift = combined.narrow(D::Minus1, c, c)?.reshape((b, c, 1, 1))?;
        // FiLM: x' = x * (1 + scale) + shift.
        let one_plus_scale = scale.affine(1.0, 1.0)?;
        Ok(x.broadcast_mul(&one_plus_scale)?.broadcast_add(&shift)?)
    }
}

// ---------------------------------------------------------------------
// AttnBlock — fused self+cross attention with KV concat
// ---------------------------------------------------------------------

/// Stable Cascade AttnBlock.
///
/// ```text
///   kv     = kv_mapper(SiLU(text_cond))               // (B, T, C)
///   x_seq  = flatten(LayerNorm2d(x))                  // (B, HW, C)
///   K = V  = self_attn ? cat(x_seq, kv) : kv          // (B, HW+T or T, C)
///   Q      = x_seq                                    // (B, HW, C)
///   out    = MultiHeadAttention(Q, K, V)              // (B, HW, C)
///   y      = x + reshape(to_out(out))                 // (B, C, H, W)
/// ```
///
/// Upstream's `kv_mapper` is `Sequential(SiLU, Linear)`, so the
/// Linear lives at index `.1` (matched by the tensor key
/// `kv_mapper.1.weight`).
pub struct AttnBlock {
    norm: LayerNorm2d,
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    kv_mapper: nn::Linear,
    num_heads: usize,
    head_dim: usize,
    self_attn: bool,
}

impl AttnBlock {
    pub fn new(
        channels: usize,
        cond_dim: usize,
        num_heads: usize,
        self_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        anyhow::ensure!(
            channels % num_heads == 0,
            "AttnBlock channels {channels} not divisible by num_heads {num_heads}"
        );
        let head_dim = channels / num_heads;
        let norm = LayerNorm2d::new(channels, 1e-6);
        let attn_vb = vb.pp("attention");
        let to_q = nn::linear(channels, channels, attn_vb.pp("to_q"))
            .map_err(|e| anyhow!("AttnBlock to_q: {e}"))?;
        let to_k = nn::linear(channels, channels, attn_vb.pp("to_k"))
            .map_err(|e| anyhow!("AttnBlock to_k: {e}"))?;
        let to_v = nn::linear(channels, channels, attn_vb.pp("to_v"))
            .map_err(|e| anyhow!("AttnBlock to_v: {e}"))?;
        let to_out = nn::linear(channels, channels, attn_vb.pp("to_out").pp("0"))
            .map_err(|e| anyhow!("AttnBlock to_out.0: {e}"))?;
        // kv_mapper is Sequential(SiLU, Linear); Linear lives at .1.
        let kv_mapper = nn::linear(cond_dim, channels, vb.pp("kv_mapper").pp("1"))
            .map_err(|e| anyhow!("AttnBlock kv_mapper.1: {e}"))?;
        Ok(Self {
            norm,
            to_q,
            to_k,
            to_v,
            to_out,
            kv_mapper,
            num_heads,
            head_dim,
            self_attn,
        })
    }

    /// `x`: `(B, C, H, W)`. `kv`: `(B, T, cond_dim)`. Returns
    /// `(B, C, H, W)` (residual already applied).
    pub fn forward(&self, x: &Tensor, kv: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        // SiLU + Linear projection of the text/cond stream.
        let kv_mapped = self.kv_mapper.forward(&kv.silu()?)?;
        // LayerNorm + flatten the image stream to a sequence.
        let norm_x = self.norm.forward(x)?;
        let x_seq = norm_x
            .reshape((b, c, h * w))?
            .transpose(1, 2)?
            .contiguous()?; // (B, HW, C)
        let kv_combined = if self.self_attn {
            Tensor::cat(&[&x_seq, &kv_mapped], 1)?
        } else {
            kv_mapped
        };
        let q = self.to_q.forward(&x_seq)?;
        let k = self.to_k.forward(&kv_combined)?;
        let v = self.to_v.forward(&kv_combined)?;
        let (_, lq, _) = q.dims3()?;
        let (_, lkv, _) = k.dims3()?;
        let q = q
            .reshape((b, lq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, lkv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, lkv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q
            .contiguous()?
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            .affine(scale, 0.)?;
        let probs = nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((b, lq, self.num_heads * self.head_dim))?;
        let out = self.to_out.forward(&out)?;
        // Reshape (B, HW, C) back to (B, C, H, W) and add residual.
        let out_2d = out.transpose(1, 2)?.reshape((b, c, h, w))?;
        Ok(x.add(&out_2d)?)
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn vb_random() -> (VarMap, Device) {
        (VarMap::new(), Device::Cpu)
    }

    // ---- LayerNorm2d ----

    #[test]
    fn layer_norm_2d_preserves_shape() {
        let ln = LayerNorm2d::new(16, 1e-6);
        let x = Tensor::randn(0f32, 1f32, (2, 16, 4, 5), &Device::Cpu).unwrap();
        let out = ln.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 16, 4, 5]);
    }

    #[test]
    fn layer_norm_2d_zero_mean_unit_var_per_site() {
        // After channel-axis LayerNorm, each (b, h, w) site should
        // have mean ≈ 0 and var ≈ 1 across the channel axis.
        let ln = LayerNorm2d::new(32, 1e-6);
        let x = Tensor::randn(5f32, 3f32, (1, 32, 2, 2), &Device::Cpu).unwrap();
        let out = ln.forward(&x).unwrap();
        let m = out.mean_keepdim(1).unwrap();
        let v = out.var_keepdim(1).unwrap();
        let m_max = m.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let v_arr: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
        assert!(m_max < 1e-4, "mean per site not ~0: max abs {m_max}");
        for vv in v_arr {
            // Variance should be ~1 (within numerical tolerance).
            assert!(
                (vv - 1.0).abs() < 0.05,
                "variance per site not ~1: got {vv}"
            );
        }
    }

    #[test]
    fn layer_norm_2d_rejects_wrong_channel_count() {
        let ln = LayerNorm2d::new(16, 1e-6);
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &Device::Cpu).unwrap();
        let err = ln.forward(&x).unwrap_err();
        assert!(format!("{err}").contains("channel mismatch"));
    }

    // ---- GRN ----

    fn random_grn(channels: usize) -> (GlobalResponseNorm, VarMap) {
        let (varmap, device) = vb_random();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let grn = GlobalResponseNorm::new(channels, vb).unwrap();
        (grn, varmap)
    }

    #[test]
    fn grn_preserves_shape() {
        let (grn, _) = random_grn(8);
        let x = Tensor::randn(0f32, 1f32, (2, 4, 4, 8), &Device::Cpu).unwrap();
        let out = grn.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 4, 4, 8]);
    }

    #[test]
    fn grn_with_zero_gamma_returns_x_plus_beta() {
        // γ = 0 → y = β + x. Provides a numerical sanity check
        // independent of the spatial normalization machinery.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let grn = GlobalResponseNorm::new(4, vb).unwrap();
        // Patch γ to zero in-place via VarMap.
        for (name, var) in varmap.data().lock().unwrap().iter() {
            if name == "gamma" {
                let z = Tensor::zeros((1, 1, 1, 4), DType::F32, &device).unwrap();
                var.set(&z).unwrap();
            }
        }
        let x = Tensor::randn(0f32, 1f32, (1, 2, 2, 4), &device).unwrap();
        let out = grn.forward(&x).unwrap();
        // out should equal x + β. β is random; we just verify the
        // delta is finite and constant across spatial positions.
        let delta = (&out - &x).unwrap();
        let delta_flat: Vec<f32> = delta.flatten_all().unwrap().to_vec1().unwrap();
        let first = delta_flat[0];
        // β has shape (1,1,1,C) so broadcasted delta is identical
        // along (B, H, W) at each channel. Pick channel 0 (every
        // 4th element) and check all match `first` for that channel.
        for i in (0..delta_flat.len()).step_by(4) {
            assert!(
                (delta_flat[i] - first).abs() < 1e-5,
                "GRN with γ=0 must produce constant per-channel delta"
            );
        }
    }

    // ---- ResBlock ----

    fn random_resblock(channels: usize) -> (ResBlock, VarMap) {
        let (varmap, device) = vb_random();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let rb = ResBlock::new(channels, vb).unwrap();
        (rb, varmap)
    }

    #[test]
    fn resblock_preserves_shape() {
        let (rb, _) = random_resblock(8);
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &Device::Cpu).unwrap();
        let out = rb.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 8, 4, 4]);
    }

    #[test]
    fn resblock_skip_is_additive_with_zero_channelwise() {
        // Zero out `channelwise_4` weight so the MLP output is the
        // bias term only; then the residual structure is dominant
        // and the output should be close-ish to x. Tests the skip
        // path itself rather than exact values.
        let (rb, varmap) = random_resblock(8);
        for (name, var) in varmap.data().lock().unwrap().iter() {
            if name == "channelwise.4.weight" {
                let z = Tensor::zeros(var.shape(), DType::F32, &Device::Cpu).unwrap();
                var.set(&z).unwrap();
            }
            if name == "channelwise.4.bias" {
                let z = Tensor::zeros(var.shape(), DType::F32, &Device::Cpu).unwrap();
                var.set(&z).unwrap();
            }
        }
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &Device::Cpu).unwrap();
        let out = rb.forward(&x).unwrap();
        // With channelwise_4 = 0, ResBlock output = 0 + x = x exactly.
        let diff = (&out - &x).unwrap().abs().unwrap().max_all().unwrap();
        let diff = diff.to_scalar::<f32>().unwrap();
        assert!(
            diff < 1e-5,
            "ResBlock with channelwise_4 zeroed should equal input (got max diff {diff})"
        );
    }

    // ---- TimestepBlock ----

    fn random_timestep_block(channels: usize, time_dim: usize, has_sca: bool, has_crp: bool) -> (TimestepBlock, VarMap) {
        let (varmap, device) = vb_random();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let tb = TimestepBlock::new(channels, time_dim, has_sca, has_crp, vb).unwrap();
        (tb, varmap)
    }

    #[test]
    fn timestep_block_mapper_only_preserves_shape() {
        let (tb, _) = random_timestep_block(16, 64, false, false);
        let x = Tensor::randn(0f32, 1f32, (1, 16, 3, 3), &Device::Cpu).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let out = tb.forward(&x, &t, None, None).unwrap();
        assert_eq!(out.dims(), &[1, 16, 3, 3]);
    }

    #[test]
    fn timestep_block_with_sca_and_crp_sums_contributions() {
        // Stage C variant: all three mappers. Output must change
        // when each conditioning stream changes.
        let (tb, _) = random_timestep_block(8, 64, true, true);
        let x = Tensor::randn(0f32, 1f32, (1, 8, 2, 2), &Device::Cpu).unwrap();
        let t1 = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let s1 = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let c1 = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let s2 = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let c2 = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let out_ref = tb.forward(&x, &t1, Some(&s1), Some(&c1)).unwrap();
        let out_sca_changed = tb.forward(&x, &t1, Some(&s2), Some(&c1)).unwrap();
        let out_crp_changed = tb.forward(&x, &t1, Some(&s1), Some(&c2)).unwrap();
        for (label, other) in [("sca", out_sca_changed), ("crp", out_crp_changed)] {
            let diff = (&out_ref - &other)
                .unwrap()
                .abs()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff > 1e-4,
                "TimestepBlock should depend on {label} (mean abs diff {diff})"
            );
        }
    }

    #[test]
    fn timestep_block_errors_when_required_emb_missing() {
        // has_sca=true but no sca_emb supplied → clear error.
        let (tb, _) = random_timestep_block(8, 64, true, false);
        let x = Tensor::randn(0f32, 1f32, (1, 8, 2, 2), &Device::Cpu).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let err = tb.forward(&x, &t, None, None).unwrap_err();
        assert!(format!("{err}").contains("mapper_sca"));
    }

    // ---- AttnBlock ----

    fn random_attn_block(c: usize, cond: usize, nh: usize, self_attn: bool) -> (AttnBlock, VarMap) {
        let (varmap, device) = vb_random();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = AttnBlock::new(c, cond, nh, self_attn, vb).unwrap();
        (attn, varmap)
    }

    #[test]
    fn attn_block_self_attn_preserves_shape() {
        let (attn, _) = random_attn_block(16, 24, 4, true);
        let x = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &Device::Cpu).unwrap();
        let kv = Tensor::randn(0f32, 1f32, (1, 5, 24), &Device::Cpu).unwrap();
        let out = attn.forward(&x, &kv).unwrap();
        assert_eq!(out.dims(), &[1, 16, 4, 4]);
    }

    #[test]
    fn attn_block_cross_only_preserves_shape() {
        let (attn, _) = random_attn_block(16, 24, 4, false);
        let x = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &Device::Cpu).unwrap();
        let kv = Tensor::randn(0f32, 1f32, (1, 5, 24), &Device::Cpu).unwrap();
        let out = attn.forward(&x, &kv).unwrap();
        assert_eq!(out.dims(), &[1, 16, 4, 4]);
    }

    #[test]
    fn attn_block_output_depends_on_kv() {
        // Two different KV inputs must produce different outputs —
        // confirms the kv_mapper + attention actually consume kv.
        let (attn, _) = random_attn_block(16, 24, 4, true);
        let x = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &Device::Cpu).unwrap();
        let kv1 = Tensor::randn(0f32, 1f32, (1, 5, 24), &Device::Cpu).unwrap();
        let kv2 = Tensor::randn(0f32, 1f32, (1, 5, 24), &Device::Cpu).unwrap();
        let o1 = attn.forward(&x, &kv1).unwrap();
        let o2 = attn.forward(&x, &kv2).unwrap();
        let diff = (&o1 - &o2)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-5, "AttnBlock output must depend on kv ({diff})");
    }

    #[test]
    fn attn_block_rejects_indivisible_head_count() {
        let (varmap, device) = vb_random();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        match AttnBlock::new(17, 24, 4, true, vb) {
            Ok(_) => panic!("expected indivisible-head-count rejection"),
            Err(e) => assert!(format!("{e}").contains("not divisible")),
        }
    }
}
