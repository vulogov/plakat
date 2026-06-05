//! v0.26 phase 2: AnimateDiff temporal-attention building blocks.
//!
//! Implements the per-block temporal transformer that consumes a
//! batch of per-frame UNet block activations and applies attention
//! across the frame dimension, producing a temporally-coherent
//! output of the same shape.
//!
//! Per-module forward signature:
//!
//! ```text
//! hidden_states: (B*F, C, H, W)
//! num_frames:    F (the count, since C is the same as the parent UNet block's channel dim)
//! →
//! out:           (B*F, C, H, W)   // residual-added back into the host UNet's hidden state
//! ```
//!
//! The module reshapes `(B*F, C, H, W)` to `(B*H*W, F, C)`,
//! applies positional embedding on the frame dimension, runs N
//! temporal-transformer blocks (self-attn across frames + a
//! cross-attn slot that's identity in AnimateDiff V3 since motion
//! is text-agnostic + FFN), then reshapes back. Final output is
//! `proj_out(residual_added_to_input)`.
//!
//! Tensor naming matches the upstream safetensors layout for V3
//! SD 1.5 AND SDXL beta (verified via safetensors header dump
//! 2026-05-28):
//!
//! ```text
//! down_blocks.{i}.motion_modules.{j}.
//!     norm.{weight,bias}                                       (GroupNorm)
//!     proj_in.{weight,bias}                                    (Linear)
//!     transformer_blocks.0.
//!         attn1.{to_q,to_k,to_v}.weight                        (self-attn — no bias on q/k/v)
//!         attn1.to_out.0.{weight,bias}                         (self-attn out projection)
//!         attn2.{to_q,to_k,to_v}.weight                        (cross-attn slot)
//!         attn2.to_out.0.{weight,bias}
//!         norm1.{weight,bias}                                  (pre-attn1 LayerNorm)
//!         norm2.{weight,bias}                                  (pre-attn2 LayerNorm)
//!         norm3.{weight,bias}                                  (pre-FF LayerNorm)
//!         ff.net.0.proj.{weight,bias}                          (GEGLU first half)
//!         ff.net.2.{weight,bias}                               (GEGLU second half)
//!         pos_embed.pe                                         (positional table; shape (1, max_seq, C))
//!     proj_out.{weight,bias}                                   (Linear)
//! ```
//!
//! The `motion_layers_per_block` config value is the number of
//! `motion_modules.{j}` slots per UNet block (typically 2). Each
//! slot has exactly one inner `transformer_blocks.0` — there is no
//! N-fold inner loop.
//!
//! Phase 2 ships the modules + a `build_modules()` constructor on
//! [`super::motion_adapter::MotionAdapter`] that yields the
//! per-UNet-block motion modules ready for splicing. The actual
//! splice into the SD 1.5 UNet forward pass is phase 3 work
//! (combined with N-frame sampling).

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Activation, GroupNorm, LayerNorm, Linear, Module, VarBuilder, group_norm, layer_norm,
    linear, linear_no_bias,
};

use super::motion_adapter::{MotionAdapter, MotionAdapterConfig};

// ---------------------------------------------------------------------------
// Positional encoding
// ---------------------------------------------------------------------------

/// Learned 1-D positional embedding for the frame dimension.
/// Shape: `(max_seq_length, dim)`. Tensor key: `pe.weight`.
///
/// AnimateDiff V3 ships `motion_max_seq_length = 32`. Generation
/// beyond that runs out of position rows — phase 3 will either
/// loop or fail loud; for now [`forward`] asserts via `D::ge`.
#[derive(Debug)]
pub struct PositionalEncoding {
    table: Tensor,
    max_len: usize,
}

impl PositionalEncoding {
    fn new(vb: VarBuilder<'_>, max_len: usize, dim: usize) -> Result<Self> {
        // V3 uses learnable positional embedding (not sinusoidal).
        // The diffusers AnimateDiffMotionModule uses
        // `nn.Parameter(torch.zeros(1, max_seq_length, dim))` which
        // serializes to `pe` of shape `(1, max_seq_length, dim)`. We
        // strip the leading 1 for cleaner broadcast.
        let raw: Tensor = vb
            .get((1, max_len, dim), "pe")
            .context("loading positional encoding tensor (pe)")?;
        let table = raw.squeeze(0)?;
        Ok(Self { table, max_len })
    }

    /// Add positional embedding to `hidden_states` shaped
    /// `(N, F, D)`. F must be ≤ `max_len`.
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (_, f, _) = hidden_states.dims3().context("expected (N, F, D)")?;
        anyhow::ensure!(
            f <= self.max_len,
            "frame count {f} exceeds motion_max_seq_length {}",
            self.max_len,
        );
        let pe = self.table.narrow(0, 0, f)?;
        // Broadcast over the batch dim N.
        let out = hidden_states.broadcast_add(&pe)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Multi-head attention
// ---------------------------------------------------------------------------

/// Scaled dot-product attention with Q/K/V projections.
///
/// For temporal self-attention, K and V come from the same input
/// as Q (across all frames). For the cross-attention slot in
/// AnimateDiff V3, encoder_hidden_states is `None` (motion is
/// text-agnostic) and the block effectively reduces to a second
/// self-attention pass. Diffusers keeps both attention_blocks
/// loaded for forward-compatibility with motion variants that DO
/// use cross-attention.
///
/// Weights:
/// * `to_q.weight`, `to_k.weight`, `to_v.weight` — no bias.
/// * `to_out.0.weight`, `to_out.0.bias` — with bias.
#[derive(Debug)]
pub struct TemporalAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl TemporalAttention {
    fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize) -> Result<Self> {
        anyhow::ensure!(
            dim.is_multiple_of(num_heads),
            "attention dim {dim} not divisible by num_heads {num_heads}"
        );
        let head_dim = dim / num_heads;
        let to_q = linear_no_bias(dim, dim, vb.pp("to_q"))?;
        let to_k = linear_no_bias(dim, dim, vb.pp("to_k"))?;
        let to_v = linear_no_bias(dim, dim, vb.pp("to_v"))?;
        let to_out = linear(dim, dim, vb.pp("to_out.0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    /// `hidden_states` shape: `(N, F, D)`. Self-attention across
    /// the F dimension; output shape matches input.
    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (n, f, _d) = hidden_states.dims3().context("hidden_states (N, F, D)")?;
        let kv_source = encoder_hidden_states.unwrap_or(hidden_states);

        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(kv_source)?;
        let v = self.to_v.forward(kv_source)?;

        // Reshape (N, F, D) → (N, H, F, head_dim).
        let q = q
            .reshape((n, f, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (_, kv_f, _) = k.dims3()?;
        let k = k
            .reshape((n, kv_f, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((n, kv_f, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        // attn = softmax(Q K^T / sqrt(d)) V
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale;
        let scores = candle_nn::ops::softmax_last_dim(&scores?)?;
        let context = scores.matmul(&v)?;

        // (N, H, F, head_dim) → (N, F, D)
        let out = context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n, f, self.num_heads * self.head_dim))?;
        let out = self.to_out.forward(&out)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Feedforward (GEGLU + Linear)
// ---------------------------------------------------------------------------

/// GEGLU FFN: `Linear(D → 8D) → split → GELU(left) * right → Linear(4D → D)`.
///
/// Diffusers serializes this as:
/// * `net.0.proj.weight` shape `(8D, D)` plus bias  — the wide projection
///   that GEGLU then splits into two `4D`-wide halves.
/// * `net.2.weight` shape `(D, 4D)` plus bias — the narrow projection back.
///
/// The `net.1` slot is the activation (stateless — no weights).
#[derive(Debug)]
pub struct TemporalFeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl TemporalFeedForward {
    fn new(vb: VarBuilder<'_>, dim: usize, mult: usize) -> Result<Self> {
        let inner = dim * mult;
        let proj_in = linear(dim, inner * 2, vb.pp("net.0.proj"))?;
        let proj_out = linear(inner, dim, vb.pp("net.2"))?;
        Ok(Self { proj_in, proj_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let proj = self.proj_in.forward(x)?;
        let last_dim = proj.dims().last().copied().unwrap_or(0);
        let half = last_dim / 2;
        // GEGLU split — chunk in two halves on the last dim. diffusers
        // (and candle's GeGlu) gate the FIRST half by GELU of the SECOND:
        // `hidden * gelu(gate)`. The motion FFN was trained that way, so
        // the gate must be on `right`, not `left`.
        let hidden = proj.narrow(D::Minus1, 0, half)?;
        let gate = proj.narrow(D::Minus1, half, half)?;
        let gated = hidden.mul(&Activation::Gelu.forward(&gate)?)?;
        let out = self.proj_out.forward(&gated)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Transformer block (norm + attn1 + norm + attn2 + norm + FF)
// ---------------------------------------------------------------------------

/// One temporal transformer block. Three sub-residuals with
/// pre-norms — same shape as a standard diffusers
/// BasicTransformerBlock but every operation runs along the F
/// dimension instead of the spatial dimension.
///
/// Weights:
/// * `attention_blocks.0` — self-attention across frames
/// * `attention_blocks.1` — cross-attention slot (identity in V3)
/// * `ff` — GEGLU FFN
/// * `norms.{0,1,2}` — LayerNorms before each sub-residual
#[derive(Debug)]
pub struct TemporalTransformerBlock {
    // diffusers' BasicTransformerBlock applies the positional embedding to
    // the POST-NORM attention input (before attn1 AND before attn2), NOT
    // to the residual stream. The `pe` lives at
    // `transformer_blocks.0.pos_embed.pe`, so it belongs to the block.
    pos_embed: PositionalEncoding,
    norm1: LayerNorm,
    attn1: TemporalAttention,
    norm2: LayerNorm,
    attn2: TemporalAttention,
    norm3: LayerNorm,
    ff: TemporalFeedForward,
}

impl TemporalTransformerBlock {
    fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize, max_len: usize) -> Result<Self> {
        // v0.27 phase 2: tensor naming matches the actual upstream
        // safetensors for both V3 SD 1.5 + SDXL beta. (The v0.26
        // phase 2 docstring referencing `attention_blocks.{0,1}` +
        // `norms.{0,1,2}` was based on diffusers' Python class
        // attribute names, but the on-disk safetensors use
        // `attn1`/`attn2` + `norm1`/`norm2`/`norm3` — verified by
        // safetensors header dump 2026-05-28.)
        let pos_embed = PositionalEncoding::new(vb.pp("pos_embed"), max_len, dim)?;
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn1 = TemporalAttention::new(vb.pp("attn1"), dim, num_heads)?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let attn2 = TemporalAttention::new(vb.pp("attn2"), dim, num_heads)?;
        let norm3 = layer_norm(dim, 1e-5, vb.pp("norm3"))?;
        let ff = TemporalFeedForward::new(vb.pp("ff"), dim, 4)?;
        Ok(Self {
            pos_embed,
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // attn1: self-attention across frames. The positional embedding is
        // added to the NORMED input (not the residual) — diffusers applies
        // it inside the block before each attention, so it never persists
        // in the residual stream.
        let norm = self.pos_embed.forward(&self.norm1.forward(hidden_states)?)?;
        let attn = self.attn1.forward(&norm, None)?;
        let h = (attn + hidden_states)?;

        // attn2: cross-attention slot (identity in V3 — no
        // encoder_hidden_states wired). pos_embed is re-applied to the
        // normed input here too, exactly as diffusers does.
        let norm = self.pos_embed.forward(&self.norm2.forward(&h)?)?;
        let attn = self.attn2.forward(&norm, None)?;
        let h = (attn + h)?;

        // FFN.
        let norm = self.norm3.forward(&h)?;
        let ffn_out = self.ff.forward(&norm)?;
        let h = (ffn_out + h)?;

        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Per-block motion module (the splice unit)
// ---------------------------------------------------------------------------

/// One full temporal-transformer attached to a single UNet
/// down/up block. Each `motion_modules.N` slot in the safetensors
/// is one of these; `motion_layers_per_block` in the config is the
/// **number of `motion_modules.N`** per UNet block (typically 2),
/// not the number of `transformer_blocks` inside one motion module
/// (always 1 per upstream convention).
///
/// On-disk weight layout (per motion_modules slot):
/// * `norm` — GroupNorm before the in-projection
/// * `proj_in` — Linear (channels → channels)
/// * `transformer_blocks.0` — the single inner transformer block
///   (attn1 + attn2 + ff with norm{1,2,3})
/// * `proj_out` — Linear (channels → channels)
/// * `transformer_blocks.0.pos_embed.pe` — positional encoding
///   table (1, max_seq_length, channels)
#[derive(Debug)]
pub struct TemporalTransformer {
    norm: GroupNorm,
    proj_in: Linear,
    /// One inner transformer block (attn1 + attn2 + ff).
    /// The upstream V3 and SDXL motion modules each carry exactly
    /// one transformer_blocks slot per motion_modules slot.
    block: TemporalTransformerBlock,
    proj_out: Linear,
    /// Channels of the UNet block this motion module attaches to —
    /// used by the splice code to verify shapes match.
    pub channels: usize,
}

impl TemporalTransformer {
    /// Build from a VarBuilder rooted at the motion-modules slot
    /// (`down_blocks.{i}.motion_modules.{j}` — NOT one level deeper).
    /// The v0.26 path that used a `.temporal_transformer.*` prefix
    /// was based on a misread of the upstream JSON; the actual
    /// safetensors keys live directly under `motion_modules.{j}`.
    fn new(
        vb: VarBuilder<'_>,
        config: &MotionAdapterConfig,
        channels: usize,
    ) -> Result<Self> {
        let norm = group_norm(
            config.motion_norm_num_groups,
            channels,
            1e-5,
            vb.pp("norm"),
        )?;
        let proj_in = linear(channels, channels, vb.pp("proj_in"))?;
        let block = TemporalTransformerBlock::new(
            vb.pp("transformer_blocks.0"),
            channels,
            config.motion_num_attention_heads,
            config.motion_max_seq_length,
        )
        .context("loading transformer_blocks.0")?;
        let proj_out = linear(channels, channels, vb.pp("proj_out"))?;
        Ok(Self {
            norm,
            proj_in,
            block,
            proj_out,
            channels,
        })
    }

    /// Apply the motion module to a per-frame block activation.
    ///
    /// Input: `(B*F, C, H, W)` where C must equal `self.channels`.
    /// `num_frames` = F.
    /// Output: `(B*F, C, H, W)` — the residual-added motion-aware
    /// activation.
    pub fn forward(&self, hidden_states: &Tensor, num_frames: usize) -> Result<Tensor> {
        let (bf, c, h, w) = hidden_states
            .dims4()
            .context("expected hidden_states (B*F, C, H, W)")?;
        anyhow::ensure!(
            c == self.channels,
            "channel mismatch: motion module expects {} but got {}",
            self.channels,
            c,
        );
        anyhow::ensure!(
            bf.is_multiple_of(num_frames),
            "batch dim {bf} not divisible by num_frames {num_frames}",
        );
        let batch = bf / num_frames;

        // Save residual.
        let residual = hidden_states.clone();

        // GroupNorm must pool statistics ACROSS frames. diffusers applies
        // it to (B, C, F, H, W), so the per-channel-group mean/var span the
        // whole F×H×W extent. Applying it per-frame on (B*F, C, H, W) —
        // each frame normalized independently — gives different statistics
        // than the affine weights were trained on, structurally distorting
        // the temporal transformer's input (a directional error no scaling
        // fixes). Reshape so frames join the spatial extent, norm, restore.
        let x = hidden_states
            .reshape((batch, num_frames, c, h, w))?
            .permute((0, 2, 1, 3, 4))? // (B, C, F, H, W)
            .contiguous()?
            .reshape((batch, c, num_frames * h * w))?;
        let x = self.norm.forward(&x)?;
        let x = x
            .reshape((batch, c, num_frames, h, w))?
            .permute((0, 2, 1, 3, 4))? // (B, F, C, H, W)
            .contiguous()?;

        // (B, F, C, H, W) → (B, H, W, F, C) → (B*H*W, F, C).
        let x = x.permute((0, 3, 4, 1, 2))?.contiguous()?;
        let x = x.reshape((batch * h * w, num_frames, c))?;

        // Project in (Linear on the last dim).
        let x = self.proj_in.forward(&x)?;

        // Single inner transformer block (attn1 + attn2 + ff). The
        // positional embedding is applied INSIDE the block (post-norm,
        // before each attention) — not here on the residual stream.
        let x = self.block.forward(&x)?;

        // Project out.
        let x = self.proj_out.forward(&x)?;

        // Reshape back (B*H*W, F, C) → (B, H, W, F, C) → (B*F, C, H, W).
        let x = x.reshape((batch, h, w, num_frames, c))?;
        let x = x.permute((0, 3, 4, 1, 2))?.contiguous()?;
        let x = x.reshape((bf, c, h, w))?;

        // Residual add.
        let out = (x + residual)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MotionAdapterModules — per-UNet-block collection
// ---------------------------------------------------------------------------

/// Address of one motion module in the AnimateDiff splice
/// pattern: which UNet block, which layer within that block.
///
/// SD 1.5 has 4 down-blocks (indices 0..=3) and 4 up-blocks
/// (indices 0..=3). With `motion_layers_per_block = 2` (V3),
/// each block carries 2 layers. V3 SD 1.5 totals 16 modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleAddr {
    pub kind: BlockKind,
    pub block_idx: usize,
    pub layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockKind {
    DownBlock,
    UpBlock,
    /// V1/V2 only. V3 skips the mid block (`use_motion_mid_block = false`).
    MidBlock,
}

impl BlockKind {
    /// Diffusers state-dict prefix for tensor keys in this kind
    /// of block. Phase 3 uses this when assembling the splice
    /// site keys.
    #[allow(dead_code)]
    pub(crate) fn diffusers_prefix(self) -> &'static str {
        match self {
            Self::DownBlock => "down_blocks",
            Self::UpBlock => "up_blocks",
            Self::MidBlock => "mid_block",
        }
    }
}

/// All temporal-transformer modules built from a loaded
/// [`MotionAdapter`]. Phase 3 will index into this by
/// [`ModuleAddr`] at each SD 1.5 UNet block boundary.
pub struct MotionAdapterModules {
    /// The per-block motion modules. Order is deterministic
    /// (down × layers, up × layers, optional mid).
    pub modules: Vec<(ModuleAddr, TemporalTransformer)>,
    /// Echoed from the config — phase 3 reads this when picking
    /// the frame count + computing per-block channel dims.
    pub config: MotionAdapterConfig,
}

impl MotionAdapterModules {
    /// Find a motion module by address. `None` for the mid-block
    /// addresses on V3 (mid is skipped when `use_motion_mid_block`
    /// is false).
    pub fn get(&self, addr: ModuleAddr) -> Option<&TemporalTransformer> {
        self.modules
            .iter()
            .find(|(a, _)| *a == addr)
            .map(|(_, m)| m)
    }

    /// All addresses present in this adapter, in build order.
    pub fn addrs(&self) -> impl Iterator<Item = &ModuleAddr> {
        self.modules.iter().map(|(a, _)| a)
    }
}

impl MotionAdapter {
    /// Build every motion module from the loaded weights. Mirrors
    /// the V3 SD 1.5 module layout:
    /// * 4 down-blocks × 2 layers = 8 down modules
    /// * 4 up-blocks × 2 layers = 8 up modules
    /// * 0 mid (V3) — V1/V2 add 1 mid module
    ///
    /// Channels are taken from `config.block_out_channels` for
    /// down blocks and the reverse for up blocks (matching the
    /// SD 1.5 UNet's U-shape).
    pub fn build_modules(
        &self,
        device: &Device,
        dtype: DType,
    ) -> Result<MotionAdapterModules> {
        let vb = self.varbuilder(dtype, device)?;
        let cfg = &self.config;
        let nb = cfg.num_blocks();
        let mut modules: Vec<(ModuleAddr, TemporalTransformer)> =
            Vec::with_capacity(cfg.total_motion_modules());

        // The module count per block is NOT uniform: SD 1.5 down
        // blocks have 2 (one per resnet) but up blocks have 3
        // (`layers_per_block + 1` resnets), so the adapter ships 8 + 12
        // = 20 modules. Probe the checkpoint for each `motion_modules.{j}`
        // rather than assuming a fixed count — this also covers V1/V2 and
        // the SDXL-beta adapter without a per-variant table.
        let probe = |prefix: &str| vb.contains_tensor(&format!("{prefix}.proj_in.weight"));

        // Down blocks: channels[i] for block i, in order.
        for block_idx in 0..nb {
            let channels = cfg.block_out_channels[block_idx];
            let mut layer_idx = 0;
            loop {
                let prefix = format!("down_blocks.{block_idx}.motion_modules.{layer_idx}");
                if !probe(&prefix) {
                    break;
                }
                let m = TemporalTransformer::new(vb.pp(&prefix), cfg, channels)
                    .with_context(|| format!("building {prefix}"))?;
                modules.push((
                    ModuleAddr {
                        kind: BlockKind::DownBlock,
                        block_idx,
                        layer_idx,
                    },
                    m,
                ));
                layer_idx += 1;
            }
        }

        // Up blocks: channels are the reverse of down (SD 1.5
        // U-shape: 1280, 1280, 640, 320 for up_blocks 0..=3).
        for block_idx in 0..nb {
            let channels = cfg.block_out_channels[nb - 1 - block_idx];
            let mut layer_idx = 0;
            loop {
                let prefix = format!("up_blocks.{block_idx}.motion_modules.{layer_idx}");
                if !probe(&prefix) {
                    break;
                }
                let m = TemporalTransformer::new(vb.pp(&prefix), cfg, channels)
                    .with_context(|| format!("building {prefix}"))?;
                modules.push((
                    ModuleAddr {
                        kind: BlockKind::UpBlock,
                        block_idx,
                        layer_idx,
                    },
                    m,
                ));
                layer_idx += 1;
            }
        }

        // Mid block — only when `use_motion_mid_block` is true (V1/V2).
        if cfg.use_motion_mid_block {
            // Mid block uses the deepest channel dim (last of block_out_channels).
            let channels = cfg.block_out_channels[nb - 1];
            for layer_idx in 0..cfg.motion_mid_block_layers_per_block {
                let prefix =
                    format!("mid_block.motion_modules.{layer_idx}");
                let m = TemporalTransformer::new(vb.pp(&prefix), cfg, channels)
                    .with_context(|| format!("building {prefix}"))?;
                modules.push((
                    ModuleAddr {
                        kind: BlockKind::MidBlock,
                        block_idx: 0,
                        layer_idx,
                    },
                    m,
                ));
            }
        }

        Ok(MotionAdapterModules {
            modules,
            config: self.config.clone(),
        })
    }
}

/// Apply every motion module for `(kind, block_idx)` sequentially
/// to `xs`. V3 has `motion_layers_per_block = 2` modules per block;
/// they apply one after the other. Used by both
/// [`crate::pipelines::sd15_motion_unet::Sd15MotionUNet`] and the
/// SDXL motion-UNet forward path.
///
/// Address not present → silently skip. Happens when the adapter's
/// per-block count doesn't reach `layer_idx` for this block (won't
/// fire with V3 / SDXL-beta + their standard configs since both
/// pair `block_out_channels` with `motion_layers_per_block`).
pub fn apply_block_motion(
    xs: Tensor,
    kind: BlockKind,
    block_idx: usize,
    mm: &MotionAdapterModules,
    num_frames: usize,
) -> Result<Tensor> {
    let mut out = xs;
    for layer_idx in 0..mm.config.motion_layers_per_block {
        let addr = ModuleAddr {
            kind,
            block_idx,
            layer_idx,
        };
        if let Some(module) = mm.get(addr) {
            out = module
                .forward(&out, num_frames)
                .with_context(|| format!("applying motion module at {addr:?}"))?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::motion_adapter::MotionAdapterConfig;

    fn v3_config() -> MotionAdapterConfig {
        MotionAdapterConfig {
            class_name: "MotionAdapter".into(),
            diffusers_version: "test".into(),
            block_out_channels: vec![320, 640, 1280, 1280],
            motion_layers_per_block: 2,
            motion_max_seq_length: 32,
            motion_mid_block_layers_per_block: 1,
            motion_norm_num_groups: 32,
            motion_num_attention_heads: 8,
            use_motion_mid_block: false,
        }
    }

    /// REFERENCE-COMPARISON DUMP (diagnostic; `#[ignore]`d by default).
    /// Loads the real V3 adapter from the HF cache, runs
    /// `down_blocks.0.motion_modules.0` on a deterministic input, and
    /// writes input + output as raw little-endian f32 to /tmp for an
    /// element-wise diff against the diffusers ground truth.
    /// Run: `cargo test --release dump_motion_ref -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_motion_ref() {
        use std::io::Write;
        let home = std::env::var("HOME").unwrap();
        let base = format!(
            "{home}/.cache/huggingface/hub/models--guoyww--animatediff-motion-adapter-v1-5-3/snapshots"
        );
        let snap = std::fs::read_dir(&base)
            .expect("adapter cached")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("diffusion_pytorch_model.safetensors").exists())
            .expect("snapshot with weights");
        let weights = snap.join("diffusion_pytorch_model.safetensors");

        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg = v3_config();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device).unwrap() };

        let write = |path: String, v: &[f32]| {
            let mut fh = std::fs::File::create(path).unwrap();
            for x in v {
                fh.write_all(&x.to_le_bytes()).unwrap();
            }
        };

        // Every motion module: down 0..3 ×{0,1} (channels 320,640,1280,1280),
        // up 0..3 ×{0,1,2} (channels 1280,1280,640,320). Pipeline shape
        // CFG batch=2, 8 frames → bf=16.
        let down_ch = [320usize, 640, 1280, 1280];
        let num_frames = 8usize;
        let (bf, h, w) = (16usize, 8usize, 8usize);
        let mut addrs: Vec<(String, usize, usize)> = vec![];
        for (b, &ch) in down_ch.iter().enumerate() {
            for l in 0..2 {
                addrs.push((format!("down_blocks.{b}.motion_modules.{l}"), ch, l));
            }
        }
        for (b, &ch) in down_ch.iter().rev().enumerate() {
            for l in 0..3 {
                addrs.push((format!("up_blocks.{b}.motion_modules.{l}"), ch, l));
            }
        }
        for (prefix, c, _l) in &addrs {
            let tt = TemporalTransformer::new(vb.pp(prefix), &cfg, *c)
                .unwrap_or_else(|e| panic!("build {prefix}: {e}"));
            let n = bf * c * h * w;
            let data: Vec<f32> =
                (0..n).map(|i| ((i as f32 * 37.0) % 1000.0) / 1000.0 - 0.5).collect();
            let input = Tensor::from_vec(data.clone(), (bf, *c, h, w), &device).unwrap();
            let out = tt.forward(&input, num_frames).unwrap();
            let out_v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let safe = prefix.replace('.', "_");
            write(format!("/tmp/mm_in_{safe}.f32"), &data);
            write(format!("/tmp/mm_out_{safe}.f32"), &out_v);
        }
        eprintln!("DUMP {} modules written", addrs.len());
    }

    /// Build a synthetic weight map matching the actual upstream
    /// safetensors tensor names for ONE motion module slot
    /// (`down_blocks.X.motion_modules.Y`). Used by the
    /// shape/passthrough/channel-mismatch tests so the layout stays
    /// in one place.
    fn zero_weights_for_motion_module(
        cfg: &MotionAdapterConfig,
        channels: usize,
        device: &Device,
        dtype: DType,
    ) -> std::collections::HashMap<String, Tensor> {
        use std::collections::HashMap;
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        let z = Tensor::zeros((channels,), dtype, device).unwrap();
        let zw = Tensor::zeros((channels, channels), dtype, device).unwrap();
        let zw_ff_in =
            Tensor::zeros((channels * 4 * 2, channels), dtype, device).unwrap();
        let z_ff_in = Tensor::zeros((channels * 4 * 2,), dtype, device).unwrap();
        let zw_ff_out =
            Tensor::zeros((channels, channels * 4), dtype, device).unwrap();

        // Outer projections + GroupNorm.
        weights.insert("norm.weight".into(), z.clone());
        weights.insert("norm.bias".into(), z.clone());
        weights.insert("proj_in.weight".into(), zw.clone());
        weights.insert("proj_in.bias".into(), z.clone());
        weights.insert("proj_out.weight".into(), zw.clone());
        weights.insert("proj_out.bias".into(), z.clone());
        // pos_embed.pe — (1, max_seq_length, dim).
        weights.insert(
            "transformer_blocks.0.pos_embed.pe".into(),
            Tensor::zeros((1, cfg.motion_max_seq_length, channels), dtype, device).unwrap(),
        );
        // Inner single transformer_block: norm{1,2,3} + attn{1,2} + ff.
        let p = |s: &str| format!("transformer_blocks.0.{s}");
        for n in 1..=3 {
            weights.insert(p(&format!("norm{n}.weight")), z.clone());
            weights.insert(p(&format!("norm{n}.bias")), z.clone());
        }
        for attn in 1..=2 {
            let a = |s: &str| p(&format!("attn{attn}.{s}"));
            weights.insert(a("to_q.weight"), zw.clone());
            weights.insert(a("to_k.weight"), zw.clone());
            weights.insert(a("to_v.weight"), zw.clone());
            weights.insert(a("to_out.0.weight"), zw.clone());
            weights.insert(a("to_out.0.bias"), z.clone());
        }
        weights.insert(p("ff.net.0.proj.weight"), zw_ff_in);
        weights.insert(p("ff.net.0.proj.bias"), z_ff_in);
        weights.insert(p("ff.net.2.weight"), zw_ff_out);
        weights.insert(p("ff.net.2.bias"), z);
        weights
    }

    /// Pure-shape test: a synthetic motion module built from
    /// zero-tensors processes a `(B*F, C, H, W)` input and
    /// produces output of the SAME shape. Doesn't validate
    /// correctness of the math — just the reshape chain.
    #[test]
    fn temporal_transformer_preserves_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg = v3_config();
        let channels = 320usize;
        let weights = zero_weights_for_motion_module(&cfg, channels, &device, dtype);
        let vb = VarBuilder::from_tensors(weights, dtype, &device);
        let tt = TemporalTransformer::new(vb, &cfg, channels)
            .expect("build temporal transformer");

        let f = 8;
        let b = 1;
        let h = 8;
        let w = 8;
        let input = Tensor::randn(0.0f32, 1.0f32, (b * f, channels, h, w), &device).unwrap();
        let out = tt.forward(&input, f).expect("forward");
        assert_eq!(out.dims(), &[b * f, channels, h, w]);
    }

    /// Zero-weight motion module is a no-op: output == input.
    /// Validates the residual + reshape chain.
    #[test]
    fn zero_weight_temporal_transformer_is_residual_passthrough() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg = v3_config();
        let channels = 320usize;
        let weights = zero_weights_for_motion_module(&cfg, channels, &device, dtype);
        let vb = VarBuilder::from_tensors(weights, dtype, &device);
        let tt = TemporalTransformer::new(vb, &cfg, channels).unwrap();

        let f = 8;
        let h = 4;
        let w = 4;
        let input = Tensor::randn(0.0f32, 1.0f32, (f, channels, h, w), &device).unwrap();
        let out = tt.forward(&input, f).unwrap();
        // With all weights zero the per-block contribution is zero;
        // residual passes through → out ≈ input.
        let diff = (&out - &input).unwrap().abs().unwrap().mean_all().unwrap();
        let v: f32 = diff.to_vec0().unwrap();
        assert!(v < 1e-5, "non-zero diff with zero weights: {v}");
    }

    /// Position encoding adds learnable embedding row-wise on
    /// the F dim, broadcasting over N.
    #[test]
    fn positional_encoding_adds_per_frame() {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let max_len = 4;
        let dim = 2;
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        // pe: distinguishable per row.
        let pe = Tensor::new(
            &[[[1.0f32, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
            &device,
        )
        .unwrap();
        weights.insert("pe".into(), pe);
        let vb = VarBuilder::from_tensors(weights, dtype, &device);
        let pe = PositionalEncoding::new(vb, max_len, dim).unwrap();

        // (N=1, F=3, D=2) input of zeros — output should equal the first F rows of pe.
        let input = Tensor::zeros((1, 3, dim), dtype, &device).unwrap();
        let out = pe.forward(&input).unwrap();
        let got: Vec<Vec<Vec<f32>>> = out.to_vec3().unwrap();
        assert_eq!(got[0][0], vec![1.0, 1.0]);
        assert_eq!(got[0][1], vec![2.0, 2.0]);
        assert_eq!(got[0][2], vec![3.0, 3.0]);
    }

    /// Position encoding refuses frames beyond max_seq_length.
    #[test]
    fn positional_encoding_rejects_oversize_frames() {
        use std::collections::HashMap;
        let device = Device::Cpu;
        let dtype = DType::F32;
        let max_len = 4;
        let dim = 2;
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert(
            "pe".into(),
            Tensor::zeros((1, max_len, dim), dtype, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(weights, dtype, &device);
        let pe = PositionalEncoding::new(vb, max_len, dim).unwrap();
        let oversized = Tensor::zeros((1, max_len + 1, dim), dtype, &device).unwrap();
        let err = pe.forward(&oversized).unwrap_err();
        assert!(err.to_string().contains("exceeds"));
    }

    /// TemporalAttention preserves input shape.
    #[test]
    fn temporal_attention_preserves_shape() {
        use std::collections::HashMap;
        let device = Device::Cpu;
        let dtype = DType::F32;
        let dim = 16;
        let heads = 4;
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        let w = Tensor::zeros((dim, dim), dtype, &device).unwrap();
        let b = Tensor::zeros((dim,), dtype, &device).unwrap();
        weights.insert("to_q.weight".into(), w.clone());
        weights.insert("to_k.weight".into(), w.clone());
        weights.insert("to_v.weight".into(), w.clone());
        weights.insert("to_out.0.weight".into(), w);
        weights.insert("to_out.0.bias".into(), b);
        let vb = VarBuilder::from_tensors(weights, dtype, &device);
        let attn = TemporalAttention::new(vb, dim, heads).unwrap();
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 8, dim), &device).unwrap();
        let out = attn.forward(&input, None).unwrap();
        assert_eq!(out.dims(), &[2, 8, dim]);
    }

    /// Channel mismatch on forward fails loud.
    #[test]
    fn motion_module_rejects_channel_mismatch() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg = v3_config();
        let channels = 320usize;
        let weights = zero_weights_for_motion_module(&cfg, channels, &device, dtype);
        let vb = VarBuilder::from_tensors(weights, dtype, &device);
        let tt = TemporalTransformer::new(vb, &cfg, channels).unwrap();
        // Wrong number of channels.
        let wrong_input =
            Tensor::randn(0.0f32, 1.0f32, (8, channels + 64, 4, 4), &device).unwrap();
        let err = tt.forward(&wrong_input, 8).unwrap_err();
        assert!(err.to_string().contains("channel mismatch"));
    }

    /// V3 module address enumeration: 8 down + 8 up + 0 mid = 16.
    #[test]
    fn module_addr_enumeration_v3() {
        let cfg = v3_config();
        assert_eq!(cfg.total_motion_modules(), 16);
        // Just verify the address space matches without
        // building the actual modules (no weights).
        let nb = cfg.num_blocks();
        let mut addrs = Vec::new();
        for block_idx in 0..nb {
            for layer_idx in 0..cfg.motion_layers_per_block {
                addrs.push(ModuleAddr {
                    kind: BlockKind::DownBlock,
                    block_idx,
                    layer_idx,
                });
            }
        }
        for block_idx in 0..nb {
            for layer_idx in 0..cfg.motion_layers_per_block {
                addrs.push(ModuleAddr {
                    kind: BlockKind::UpBlock,
                    block_idx,
                    layer_idx,
                });
            }
        }
        if cfg.use_motion_mid_block {
            for layer_idx in 0..cfg.motion_mid_block_layers_per_block {
                addrs.push(ModuleAddr {
                    kind: BlockKind::MidBlock,
                    block_idx: 0,
                    layer_idx,
                });
            }
        }
        assert_eq!(addrs.len(), 16);
    }

    /// Network-required end-to-end test: download real V3
    /// weights, build all 16 modules. Costs ~1.4 GB on first
    /// run; subsequent runs use the HF cache.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    #[ignore]
    async fn build_modules_from_real_v3_weights() {
        let adapter = MotionAdapter::load_v3().await.expect("download V3");
        let modules = adapter
            .build_modules(&Device::Cpu, DType::F32)
            .expect("build all motion modules");
        assert_eq!(modules.modules.len(), 16);
        // Spot-check the per-block channels match SD 1.5 UNet.
        let expected_channels = [
            (BlockKind::DownBlock, 0, 320),
            (BlockKind::DownBlock, 1, 640),
            (BlockKind::DownBlock, 2, 1280),
            (BlockKind::DownBlock, 3, 1280),
            (BlockKind::UpBlock, 0, 1280),
            (BlockKind::UpBlock, 1, 1280),
            (BlockKind::UpBlock, 2, 640),
            (BlockKind::UpBlock, 3, 320),
        ];
        for (kind, block_idx, want_c) in expected_channels {
            for layer_idx in 0..2 {
                let addr = ModuleAddr {
                    kind,
                    block_idx,
                    layer_idx,
                };
                let m = modules.get(addr).expect("module present");
                assert_eq!(m.channels, want_c, "addr {addr:?} channels");
            }
        }
        // Quick forward smoke at the smallest block (320 channels,
        // 8×8 spatial, 8 frames, batch 1).
        let smallest_addr = ModuleAddr {
            kind: BlockKind::DownBlock,
            block_idx: 0,
            layer_idx: 0,
        };
        let m = modules.get(smallest_addr).unwrap();
        let input =
            Tensor::randn(0.0f32, 1.0f32, (8, 320, 8, 8), &Device::Cpu).unwrap();
        let out = m.forward(&input, 8).expect("forward smoke");
        assert_eq!(out.dims(), &[8, 320, 8, 8]);
    }
}
