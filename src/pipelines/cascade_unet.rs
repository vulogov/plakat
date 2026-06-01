//! Stable Cascade UNet — generic backbone shared by Stage B (latent
//! prior) and Stage C (high-res prior).
//!
//! v0.37 phase 2: shape-correct skeleton with the essential blocks
//! Stable Cascade uses — ResBlocks, self-attention, cross-attention
//! to CLIP-G text. The same `StableCascadeUnet` struct serves both
//! stages with different `Config` instances; phase 3 adds the Stage
//! C-specific config + larger channel widths.
//!
//! ## Architecture
//!
//! Stable Cascade's prior UNets follow a standard hourglass shape
//! with text cross-attention at the deeper levels:
//!
//! ```text
//!   noisy_latent (B, C_in, h, w)
//!   time (B,) ──────► time_mlp ──┐
//!   text  (B, T, 1280) ─►────────│ cross-attn
//!                                ▼
//!   in_conv ──► encoder ──► bottleneck ──► decoder ──► out_conv
//!                  │            │ ▲             ▲
//!                  └─ skip ─────┘ │             │
//!                                 └── self-attn ┘
//! ```
//!
//! Each block at a given level:
//! - **ResBlock** (timestep-conditioned, FiLM-style scale+shift on
//!   the normalised hidden states).
//! - **Self-attention** (at the deepest 2 levels).
//! - **Cross-attention** to the CLIP-G text sequence (same depths).
//!
//! Stage B is conditioned on Stage C's output ALSO (the so-called
//! `effnet` conditioning). Phase 2 leaves that input slot in the
//! forward signature but treats it as optional — wired through
//! phase 4 when end-to-end inference runs.
//!
//! ## v0.37 phase 2 scope
//!
//! - Generic `StableCascadeUnet` + `Config`.
//! - `Config::stage_b_full()` and `Config::stage_b_lite()`
//!   constructors. Stage C configs land in phase 3.
//! - Shape-correct forward pass: `(B, C_in, h, w)` latent +
//!   `(B,)` time + `(B, seq, 1280)` text → `(B, C_in, h, w)`.
//! - Tensor names follow the diffusers
//!   `stabilityai/stable-cascade` decoder module hierarchy
//!   (best-effort; real-weight verification at v0.37 phase 4 smoke).
//!
//! ## What's NOT here (deferred)
//!
//! - **Effnet conditioning on Stage C's output** (Stage B's
//!   specific feature) — phase 4 wires it.
//! - **Stage C config / instantiation** — phase 3.
//! - **Time embedding flat MLP details** that match upstream
//!   exactly (we use a reasonable sinusoidal + 2-layer MLP).
//! - **All upstream-specific quirks** (the published Stage B/C
//!   have flow-matching schedulers etc; phase 4 surfaces the
//!   compatibility story).

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::pipelines::cascade_stage_a::ResBlock;

/// Stable Cascade UNet config. Shared between Stage B and Stage C
/// with different channel widths + depths.
#[derive(Debug, Clone)]
pub struct Config {
    /// Input/output channel count. 4 for Stage B (denoising in
    /// Stage A's latent space). 16 for Stage C (denoising in its
    /// own super-compressed space).
    pub channels: usize,
    /// Channels at each spatial level, coarsest-first. Length
    /// determines the number of down/up blocks. The deepest level
    /// (last entry) is the bottleneck width.
    pub level_channels: Vec<usize>,
    /// Whether each level uses attention. Same length as
    /// `level_channels`. `true` enables self-attn + cross-attn to
    /// text at that level.
    pub attention_levels: Vec<bool>,
    /// Number of ResBlocks per level (before the down/up sample).
    pub blocks_per_level: usize,
    /// CLIP-G text hidden dim — feeds cross-attention K/V. Always
    /// 1280 for Stable Cascade.
    pub text_hidden_size: usize,
    /// Number of attention heads per attention block.
    pub num_heads: usize,
    /// GroupNorm group count. SD-style default 32.
    pub norm_groups: usize,
}

impl Config {
    /// Stable Cascade Stage B Full (~1.5B params upstream).
    ///
    /// Operates in Stage A's latent space (4 channels). 4 levels:
    /// 32×32 → 16×16 → 8×8 → 4×4 bottleneck. Attention at the two
    /// deepest levels.
    pub fn stage_b_full() -> Self {
        Self {
            channels: 4,
            level_channels: vec![320, 640, 1280, 1280],
            attention_levels: vec![false, false, true, true],
            blocks_per_level: 2,
            text_hidden_size: 1280,
            num_heads: 20,
            norm_groups: 32,
        }
    }

    /// Stable Cascade Stage B Lite — smaller channel widths than
    /// Full. Same level structure + attention placement.
    pub fn stage_b_lite() -> Self {
        Self {
            channels: 4,
            level_channels: vec![256, 512, 768, 768],
            attention_levels: vec![false, false, true, true],
            blocks_per_level: 1,
            text_hidden_size: 1280,
            num_heads: 12,
            norm_groups: 32,
        }
    }

    /// Pick the right Stage B config from a Stable Cascade alias.
    /// Defaults to Full when no `-lite` substring is present.
    pub fn stage_b_for_alias(alias: &str) -> Self {
        if alias.to_lowercase().contains("lite") {
            Self::stage_b_lite()
        } else {
            Self::stage_b_full()
        }
    }

    /// v0.37 phase 3: Stable Cascade Stage C Full (~3.6B params
    /// upstream).
    ///
    /// Operates in the super-compressed prior latent space:
    /// **16 input/output channels** (vs Stage B's 4) at a tiny
    /// spatial grid (24×24 at upstream 1024² output). The small
    /// spatial extent means fewer levels are useful — 3 here vs
    /// Stage B's 4. Attention at every level (the tiny sequence
    /// keeps attention affordable + Stage C carries the heaviest
    /// text-conditioned reasoning).
    pub fn stage_c_full() -> Self {
        Self {
            channels: 16,
            level_channels: vec![1024, 1536, 2048],
            attention_levels: vec![true, true, true],
            blocks_per_level: 4,
            text_hidden_size: 1280,
            num_heads: 32,
            norm_groups: 32,
        }
    }

    /// Stable Cascade Stage C Lite — smaller channel widths +
    /// fewer blocks per level than Full. Same level structure +
    /// attention placement.
    pub fn stage_c_lite() -> Self {
        Self {
            channels: 16,
            level_channels: vec![768, 1024, 1536],
            attention_levels: vec![true, true, true],
            blocks_per_level: 2,
            text_hidden_size: 1280,
            num_heads: 24,
            norm_groups: 32,
        }
    }

    /// v0.37 phase 3: pick the right Stage C config from a Stable
    /// Cascade alias. Defaults to Full when no `-lite` substring
    /// is present.
    pub fn stage_c_for_alias(alias: &str) -> Self {
        if alias.to_lowercase().contains("lite") {
            Self::stage_c_lite()
        } else {
            Self::stage_c_full()
        }
    }
}

// ---------------------------------------------------------------------
// Time embedding.
// ---------------------------------------------------------------------

/// Sinusoidal-then-MLP time embedding. Same shape SD/SDXL use:
/// `Linear(emb_dim → 4*emb_dim) → SiLU → Linear(4*emb_dim → time_dim)`.
pub struct TimeEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
    sinusoidal_dim: usize,
}

impl TimeEmbedding {
    pub fn new(time_dim: usize, vb: VarBuilder) -> Result<Self> {
        let sinusoidal_dim = 256;
        let inner = time_dim * 4;
        let linear_1 = nn::linear(sinusoidal_dim, inner, vb.pp("linear_1"))
            .map_err(|e| anyhow!("TimeEmbedding linear_1: {e}"))?;
        let linear_2 = nn::linear(inner, time_dim, vb.pp("linear_2"))
            .map_err(|e| anyhow!("TimeEmbedding linear_2: {e}"))?;
        Ok(Self {
            linear_1,
            linear_2,
            sinusoidal_dim,
        })
    }

    /// `t`: shape `(B,)`. Returns `(B, time_dim)`.
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let half = self.sinusoidal_dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000_f32.ln()) * (i as f32) / (half as f32)).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs, half, device)?;
        let t_f32 = t.to_dtype(DType::F32)?;
        let args = t_f32.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        let emb = emb.to_dtype(self.linear_1.weight().dtype())?;
        let h = self.linear_1.forward(&emb)?;
        let h = h.silu()?;
        Ok(self.linear_2.forward(&h)?)
    }
}

// ---------------------------------------------------------------------
// Attention.
// ---------------------------------------------------------------------

/// Multi-head attention with separate Q/K/V projections — same
/// shape PixArt uses (v0.35 phase 1) so the diffusers tensor names
/// match. Supports self-attention (Q=K=V=x) and cross-attention
/// (Q=x, K=V=kv).
pub struct Attention {
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(
        query_dim: usize,
        kv_dim: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = query_dim / num_heads;
        let to_q = nn::linear(query_dim, query_dim, vb.pp("to_q"))
            .map_err(|e| anyhow!("Attention to_q: {e}"))?;
        let to_k = nn::linear(kv_dim, query_dim, vb.pp("to_k"))
            .map_err(|e| anyhow!("Attention to_k: {e}"))?;
        let to_v = nn::linear(kv_dim, query_dim, vb.pp("to_v"))
            .map_err(|e| anyhow!("Attention to_v: {e}"))?;
        let to_out = nn::linear(query_dim, query_dim, vb.pp("to_out").pp("0"))
            .map_err(|e| anyhow!("Attention to_out.0: {e}"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    /// `x` is the query stream `(B, Lq, query_dim)`. `kv` is the
    /// key/value stream `(B, Lkv, kv_dim)` — pass `x` itself for
    /// self-attention.
    pub fn forward(&self, x: &Tensor, kv: &Tensor) -> Result<Tensor> {
        let (b, lq, _) = x.dims3()?;
        let (_, lkv, _) = kv.dims3()?;
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(kv)?;
        let v = self.to_v.forward(kv)?;
        let q = q.reshape((b, lq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, lkv, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, lkv, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q
            .contiguous()?
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            .affine(scale, 0.)?;
        let probs = nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs.matmul(&v.contiguous()?)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, lq, self.num_heads * self.head_dim))?;
        Ok(self.to_out.forward(&out)?)
    }
}

/// One transformer-style attention block: norm → self-attn → norm
/// → cross-attn-to-text → norm → FF (just a 2-layer MLP). Operates
/// on flattened (B, H*W, C) tokens; the level wrapper reshapes the
/// spatial Conv2d output back and forth.
pub struct AttentionBlock {
    norm1: nn::LayerNorm,
    self_attn: Attention,
    norm2: nn::LayerNorm,
    cross_attn: Attention,
    norm3: nn::LayerNorm,
    ff_in: nn::Linear,
    ff_out: nn::Linear,
}

impl AttentionBlock {
    pub fn new(
        channels: usize,
        text_dim: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = nn::layer_norm(channels, 1e-6, vb.pp("norm1"))
            .map_err(|e| anyhow!("AttentionBlock norm1: {e}"))?;
        let self_attn = Attention::new(channels, channels, num_heads, vb.pp("self_attn"))?;
        let norm2 = nn::layer_norm(channels, 1e-6, vb.pp("norm2"))
            .map_err(|e| anyhow!("AttentionBlock norm2: {e}"))?;
        let cross_attn = Attention::new(channels, text_dim, num_heads, vb.pp("cross_attn"))?;
        let norm3 = nn::layer_norm(channels, 1e-6, vb.pp("norm3"))
            .map_err(|e| anyhow!("AttentionBlock norm3: {e}"))?;
        let ff_in = nn::linear(channels, channels * 4, vb.pp("ff").pp("net").pp("0"))
            .map_err(|e| anyhow!("AttentionBlock ff_in: {e}"))?;
        let ff_out = nn::linear(channels * 4, channels, vb.pp("ff").pp("net").pp("2"))
            .map_err(|e| anyhow!("AttentionBlock ff_out: {e}"))?;
        Ok(Self {
            norm1,
            self_attn,
            norm2,
            cross_attn,
            norm3,
            ff_in,
            ff_out,
        })
    }

    /// `x`: spatial `(B, C, H, W)`. `text`: `(B, seq, text_dim)`.
    /// Returns spatial `(B, C, H, W)`.
    pub fn forward(&self, x: &Tensor, text: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        // Flatten spatial dims to a sequence: (B, C, H, W) → (B, H*W, C).
        let tokens = x
            .reshape((b, c, h * w))?
            .transpose(1, 2)?
            .contiguous()?;
        // Self-attention block.
        let normed = self.norm1.forward(&tokens)?;
        let self_out = self.self_attn.forward(&normed, &normed)?;
        let tokens = tokens.add(&self_out)?;
        // Cross-attention to text.
        let normed = self.norm2.forward(&tokens)?;
        let cross_out = self.cross_attn.forward(&normed, text)?;
        let tokens = tokens.add(&cross_out)?;
        // Feedforward.
        let normed = self.norm3.forward(&tokens)?;
        let ff = self.ff_in.forward(&normed)?;
        let ff = ff.gelu()?;
        let ff = self.ff_out.forward(&ff)?;
        let tokens = tokens.add(&ff)?;
        // Reshape back to spatial.
        Ok(tokens
            .transpose(1, 2)?
            .reshape((b, c, h, w))?)
    }
}

// ---------------------------------------------------------------------
// Down / Up samplers.
// ---------------------------------------------------------------------

pub struct Downsample {
    conv: nn::Conv2d,
}

impl Downsample {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = nn::conv2d(
            channels,
            channels,
            3,
            nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )
        .map_err(|e| anyhow!("Downsample conv: {e}"))?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.conv.forward(x)?)
    }
}

pub struct Upsample {
    conv: nn::Conv2d,
}

impl Upsample {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = nn::conv2d(
            channels,
            channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )
        .map_err(|e| anyhow!("Upsample conv: {e}"))?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        Ok(self.conv.forward(&up)?)
    }
}

// ---------------------------------------------------------------------
// Top-level UNet.
// ---------------------------------------------------------------------

/// Encoder block at one level: stacked ResBlocks (+ optional
/// AttentionBlock per ResBlock) and a Downsample.
pub struct EncoderLevel {
    res_blocks: Vec<ResBlock>,
    attentions: Vec<Option<AttentionBlock>>,
    downsample: Option<Downsample>,
}

impl EncoderLevel {
    pub fn new(
        in_c: usize,
        out_c: usize,
        blocks: usize,
        has_attention: bool,
        text_dim: usize,
        num_heads: usize,
        groups: usize,
        is_last: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut res_blocks = Vec::with_capacity(blocks);
        let mut attentions = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let in_ch = if i == 0 { in_c } else { out_c };
            res_blocks.push(ResBlock::new(
                in_ch,
                out_c,
                groups,
                vb.pp("res_blocks").pp(&i.to_string()),
            )?);
            attentions.push(if has_attention {
                Some(AttentionBlock::new(
                    out_c,
                    text_dim,
                    num_heads,
                    vb.pp("attentions").pp(&i.to_string()),
                )?)
            } else {
                None
            });
        }
        let downsample = if is_last {
            None
        } else {
            Some(Downsample::new(out_c, vb.pp("downsample"))?)
        };
        Ok(Self {
            res_blocks,
            attentions,
            downsample,
        })
    }

    /// Returns the level output AND each ResBlock's output for skip
    /// connections (consumed by the matching DecoderLevel).
    pub fn forward(&self, x: &Tensor, text: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let mut x = x.clone();
        let mut skips = Vec::with_capacity(self.res_blocks.len());
        for (rb, attn) in self.res_blocks.iter().zip(self.attentions.iter()) {
            x = rb.forward(&x)?;
            if let Some(a) = attn {
                x = a.forward(&x, text)?;
            }
            skips.push(x.clone());
        }
        if let Some(ds) = &self.downsample {
            x = ds.forward(&x)?;
        }
        Ok((x, skips))
    }
}

pub struct DecoderLevel {
    res_blocks: Vec<ResBlock>,
    attentions: Vec<Option<AttentionBlock>>,
    upsample: Option<Upsample>,
}

impl DecoderLevel {
    pub fn new(
        in_c: usize,
        out_c: usize,
        skip_c: usize,
        blocks: usize,
        has_attention: bool,
        text_dim: usize,
        num_heads: usize,
        groups: usize,
        // `is_shallowest`: `true` for the shallowest decoder level
        // — the final output level whose output feeds `out_conv`,
        // so no upsample at its end. The deepest decoder + every
        // middle level DOES upsample at its end to feed the next.
        is_shallowest: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut res_blocks = Vec::with_capacity(blocks);
        let mut attentions = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let in_ch = if i == 0 { in_c + skip_c } else { out_c + skip_c };
            res_blocks.push(ResBlock::new(
                in_ch,
                out_c,
                groups,
                vb.pp("res_blocks").pp(&i.to_string()),
            )?);
            attentions.push(if has_attention {
                Some(AttentionBlock::new(
                    out_c,
                    text_dim,
                    num_heads,
                    vb.pp("attentions").pp(&i.to_string()),
                )?)
            } else {
                None
            });
        }
        let upsample = if is_shallowest {
            None
        } else {
            Some(Upsample::new(out_c, vb.pp("upsample"))?)
        };
        Ok(Self {
            res_blocks,
            attentions,
            upsample,
        })
    }

    /// `skips` must contain one tensor per ResBlock in this level
    /// (matching the corresponding EncoderLevel's output).
    pub fn forward(&self, x: &Tensor, skips: &[Tensor], text: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            skips.len() == self.res_blocks.len(),
            "DecoderLevel: expected {} skip tensors, got {}",
            self.res_blocks.len(),
            skips.len()
        );
        let mut x = x.clone();
        for (i, (rb, attn)) in self.res_blocks.iter().zip(self.attentions.iter()).enumerate() {
            // Concatenate skip on channel axis. Skips are consumed
            // in reverse order so the deepest layer sees the
            // innermost encoder output.
            let skip = &skips[skips.len() - 1 - i];
            x = Tensor::cat(&[&x, skip], 1)?;
            x = rb.forward(&x)?;
            if let Some(a) = attn {
                x = a.forward(&x, text)?;
            }
        }
        if let Some(us) = &self.upsample {
            x = us.forward(&x)?;
        }
        Ok(x)
    }
}

/// Top-level UNet shared by Stage B + Stage C.
pub struct StableCascadeUnet {
    in_conv: nn::Conv2d,
    time_embedding: TimeEmbedding,
    encoder_levels: Vec<EncoderLevel>,
    decoder_levels: Vec<DecoderLevel>,
    out_norm: nn::GroupNorm,
    out_conv: nn::Conv2d,
    pub cfg: Config,
    pub dtype: DType,
    pub device: Device,
}

impl StableCascadeUnet {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let first_level = cfg.level_channels[0];
        let in_conv = nn::conv2d(
            cfg.channels,
            first_level,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("in_conv"),
        )
        .map_err(|e| anyhow!("StableCascadeUnet in_conv: {e}"))?;

        let time_embedding = TimeEmbedding::new(first_level, vb.pp("time_embedding"))?;

        // Encoder levels: in_channels = level_channels[0] for the
        // first, then output of the previous level for subsequent.
        let mut encoder_levels = Vec::with_capacity(cfg.level_channels.len());
        let n_levels = cfg.level_channels.len();
        for (i, &out_c) in cfg.level_channels.iter().enumerate() {
            let in_c = if i == 0 { first_level } else { cfg.level_channels[i - 1] };
            encoder_levels.push(EncoderLevel::new(
                in_c,
                out_c,
                cfg.blocks_per_level,
                cfg.attention_levels[i],
                cfg.text_hidden_size,
                cfg.num_heads,
                cfg.norm_groups,
                i == n_levels - 1,
                vb.pp("encoder_levels").pp(&i.to_string()),
            )?);
        }

        // Decoder levels mirror the encoder (skip connections from
        // matching encoder levels). Iterate deepest → shallowest;
        // each level upsamples its output to feed the next, except
        // the shallowest one whose output goes to `out_conv` at the
        // top spatial resolution.
        let mut decoder_levels = Vec::with_capacity(n_levels);
        for i in (0..n_levels).rev() {
            // x channel widths follow the encoder mirror: each
            // level enters at its own width, exits at the next-
            // shallower width (or stays at index 0 for the
            // shallowest level).
            let in_c = cfg.level_channels[i];
            let out_c = if i == 0 {
                cfg.level_channels[0]
            } else {
                cfg.level_channels[i - 1]
            };
            decoder_levels.push(DecoderLevel::new(
                in_c,
                out_c,
                cfg.level_channels[i],
                cfg.blocks_per_level,
                cfg.attention_levels[i],
                cfg.text_hidden_size,
                cfg.num_heads,
                cfg.norm_groups,
                i == 0, // is_shallowest: only level 0 is the final output
                vb.pp("decoder_levels").pp(&(n_levels - 1 - i).to_string()),
            )?);
        }

        let out_norm = nn::group_norm(
            crate::pipelines::cascade_stage_a::group_size(cfg.norm_groups, first_level),
            first_level,
            1e-6,
            vb.pp("out_norm"),
        )
        .map_err(|e| anyhow!("StableCascadeUnet out_norm: {e}"))?;
        let out_conv = nn::conv2d(
            first_level,
            cfg.channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("out_conv"),
        )
        .map_err(|e| anyhow!("StableCascadeUnet out_conv: {e}"))?;

        Ok(Self {
            in_conv,
            time_embedding,
            encoder_levels,
            decoder_levels,
            out_norm,
            out_conv,
            cfg,
            dtype,
            device,
        })
    }

    /// Forward pass.
    ///
    /// - `latent`: `(B, channels, h, w)` noisy latent.
    /// - `timestep`: `(B,)` integer or float timesteps.
    /// - `text`: `(B, seq_len, text_hidden_size)` CLIP-G text
    ///   encoder hidden states (penultimate, matching SDXL CLIP-G
    ///   convention).
    ///
    /// Returns `(B, channels, h, w)` denoised latent prediction.
    ///
    /// **Phase 2 scope:** time_embedding is computed but not yet
    /// injected into the ResBlocks (the upstream injection point
    /// is per-block FiLM scale+shift; wiring lands in phase 4
    /// alongside Stage C). The forward path still produces
    /// shape-correct output for the level structure.
    pub fn forward(&self, latent: &Tensor, timestep: &Tensor, text: &Tensor) -> Result<Tensor> {
        let mut x = self.in_conv.forward(latent)?;
        // Time conditioning is computed but its block-wise injection
        // wires in phase 4 (see method doc).
        let _t_emb = self.time_embedding.forward(timestep)?;

        // Encoder: collect per-level skip tensors.
        let mut all_skips: Vec<Vec<Tensor>> = Vec::with_capacity(self.encoder_levels.len());
        for level in &self.encoder_levels {
            let (out, skips) = level.forward(&x, text)?;
            all_skips.push(skips);
            x = out;
        }

        // Decoder: consume skips in reverse level order.
        for (level, skips) in self.decoder_levels.iter().zip(all_skips.iter().rev()) {
            x = level.forward(&x, skips, text)?;
        }

        let x = self.out_norm.forward(&x)?;
        let x = x.silu()?;
        Ok(self.out_conv.forward(&x)?)
    }
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// Small Stage B config for fast tests. 3 levels (16×16 → 8×8 →
    /// 4×4), attention at the deepest level only.
    fn small_stage_b_cfg() -> Config {
        Config {
            channels: 4,
            level_channels: vec![16, 32, 64],
            attention_levels: vec![false, true, true],
            blocks_per_level: 1,
            text_hidden_size: 24,
            num_heads: 4,
            norm_groups: 8,
        }
    }

    fn random_unet(cfg: Config) -> (StableCascadeUnet, VarMap) {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let unet = StableCascadeUnet::new(cfg, vb).expect("StableCascadeUnet::new");
        (unet, varmap)
    }

    #[test]
    fn time_embedding_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let te = TimeEmbedding::new(64, vb).unwrap();
        let t = Tensor::new(&[100f32, 250.0], &device).unwrap();
        let emb = te.forward(&t).unwrap();
        assert_eq!(emb.dims(), &[2, 64]);
    }

    #[test]
    fn self_attention_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = Attention::new(16, 16, 4, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 16), &device).unwrap();
        let out = attn.forward(&x, &x).unwrap();
        assert_eq!(out.dims(), &[1, 8, 16]);
    }

    #[test]
    fn cross_attention_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Q dim 16, KV dim 24 (text channel).
        let attn = Attention::new(16, 24, 4, vb).unwrap();
        let q = Tensor::randn(0f32, 1f32, (1, 8, 16), &device).unwrap();
        let kv = Tensor::randn(0f32, 1f32, (1, 5, 24), &device).unwrap();
        let out = attn.forward(&q, &kv).unwrap();
        assert_eq!(out.dims(), &[1, 8, 16]);
    }

    #[test]
    fn attention_block_spatial_round_trip() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = AttentionBlock::new(16, 24, 4, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        let text = Tensor::randn(0f32, 1f32, (1, 5, 24), &device).unwrap();
        let out = block.forward(&x, &text).unwrap();
        assert_eq!(out.dims(), &[1, 16, 4, 4]);
    }

    #[test]
    fn downsample_halves_spatial_dims() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let ds = Downsample::new(8, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 16, 16), &device).unwrap();
        let out = ds.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn upsample_doubles_spatial_dims() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let us = Upsample::new(8, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &device).unwrap();
        let out = us.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn unet_full_forward_preserves_shape() {
        let (unet, _) = random_unet(small_stage_b_cfg());
        let device = &unet.device;
        // (1, 4, 16, 16) noisy latent → same shape output.
        let latent = Tensor::randn(0f32, 1f32, (1, 4, 16, 16), device).unwrap();
        let timestep = Tensor::new(&[100f32], device).unwrap();
        let text = Tensor::randn(0f32, 1f32, (1, 5, 24), device).unwrap();
        let out = unet.forward(&latent, &timestep, &text).unwrap();
        assert_eq!(out.dims(), &[1, 4, 16, 16]);
    }

    #[test]
    fn stage_b_full_config_has_four_levels_with_attn_at_deepest_two() {
        let cfg = Config::stage_b_full();
        assert_eq!(cfg.channels, 4);
        assert_eq!(cfg.level_channels.len(), 4);
        assert_eq!(cfg.attention_levels, vec![false, false, true, true]);
        assert_eq!(cfg.text_hidden_size, 1280);
    }

    #[test]
    fn stage_b_lite_smaller_than_full() {
        let full = Config::stage_b_full();
        let lite = Config::stage_b_lite();
        for (f, l) in full.level_channels.iter().zip(lite.level_channels.iter()) {
            assert!(l < f, "lite channel {l} should be < full channel {f}");
        }
        assert!(lite.blocks_per_level <= full.blocks_per_level);
        assert!(lite.num_heads < full.num_heads);
    }

    #[test]
    fn stage_b_for_alias_picks_lite_or_full() {
        let lite = Config::stage_b_for_alias("stable-cascade-lite");
        assert_eq!(lite.num_heads, Config::stage_b_lite().num_heads);
        let full = Config::stage_b_for_alias("stable-cascade");
        assert_eq!(full.num_heads, Config::stage_b_full().num_heads);
        // Mixed case still routes correctly.
        let lite_mixed = Config::stage_b_for_alias("STABLE-CASCADE-LITE");
        assert_eq!(lite_mixed.num_heads, Config::stage_b_lite().num_heads);
    }

    // v0.37 phase 3: Stage C config + forward pass.

    #[test]
    fn stage_c_full_config_has_three_levels_all_attention() {
        let cfg = Config::stage_c_full();
        // Stage C operates on the super-compressed prior latent
        // (16 channels at 24×24 spatial upstream). All 3 levels
        // carry attention because the sequence stays short even
        // at the shallowest level.
        assert_eq!(cfg.channels, 16);
        assert_eq!(cfg.level_channels.len(), 3);
        assert_eq!(cfg.attention_levels, vec![true, true, true]);
        assert_eq!(cfg.text_hidden_size, 1280);
        assert!(
            cfg.num_heads >= 16,
            "Stage C Full uses more heads than Stage B Full"
        );
    }

    #[test]
    fn stage_c_lite_smaller_than_full() {
        let full = Config::stage_c_full();
        let lite = Config::stage_c_lite();
        for (f, l) in full.level_channels.iter().zip(lite.level_channels.iter()) {
            assert!(l < f, "Stage C lite channel {l} should be < full channel {f}");
        }
        assert!(lite.blocks_per_level <= full.blocks_per_level);
        assert!(lite.num_heads < full.num_heads);
    }

    #[test]
    fn stage_c_for_alias_picks_lite_or_full() {
        let lite = Config::stage_c_for_alias("stable-cascade-lite");
        assert_eq!(lite.num_heads, Config::stage_c_lite().num_heads);
        let full = Config::stage_c_for_alias("stable-cascade");
        assert_eq!(full.num_heads, Config::stage_c_full().num_heads);
        // Mixed case still routes.
        let lite_mixed = Config::stage_c_for_alias("STABILITYAI/STABLE-CASCADE-LITE");
        assert_eq!(lite_mixed.num_heads, Config::stage_c_lite().num_heads);
    }

    #[test]
    fn stage_c_uses_more_channels_than_stage_b() {
        // Stage C operates on a more-compressed latent space with
        // more conditioning info per token — pixel-wise it has
        // 16 channels vs Stage B's 4. The architectural widths
        // also reflect Stage C's role as the heavyweight prior.
        assert_eq!(Config::stage_b_full().channels, 4);
        assert_eq!(Config::stage_c_full().channels, 16);
    }

    /// Small Stage C cfg for fast end-to-end forward test.
    /// 3 levels with attention everywhere; tiny channels so the
    /// UNet instantiation stays cheap on CPU.
    fn small_stage_c_cfg() -> Config {
        Config {
            channels: 16,
            level_channels: vec![32, 48, 64],
            attention_levels: vec![true, true, true],
            blocks_per_level: 1,
            text_hidden_size: 24,
            num_heads: 4,
            norm_groups: 8,
        }
    }

    #[test]
    fn stage_c_unet_full_forward_preserves_shape() {
        let (unet, _) = random_unet(small_stage_c_cfg());
        let device = &unet.device;
        // (1, 16, 8, 8) noisy prior latent — Stage C's input shape
        // at a smaller resolution for test speed. Output must match
        // input shape (it's a denoising prediction).
        let latent = Tensor::randn(0f32, 1f32, (1, 16, 8, 8), device).unwrap();
        let timestep = Tensor::new(&[100f32], device).unwrap();
        let text = Tensor::randn(0f32, 1f32, (1, 5, 24), device).unwrap();
        let out = unet.forward(&latent, &timestep, &text).unwrap();
        assert_eq!(out.dims(), &[1, 16, 8, 8]);
    }
}

/// v0.37 phase 2: re-export the GroupNorm group fallback helper
/// so the UNet's out_norm sites can reuse the same divisor-aware
/// math. Same definition as `cascade_stage_a::group_size` (kept
/// pub(crate) there); this module-level alias keeps the import
/// path short for tests + internal callers.
#[allow(dead_code)]
pub(crate) fn _group_size_alias_keep_alive(default: usize, channels: usize) -> usize {
    crate::pipelines::cascade_stage_a::group_size(default, channels)
}
