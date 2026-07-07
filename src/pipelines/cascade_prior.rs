//! v0.39 phase 0b: Stable Cascade prior UNet (Stage B + Stage C),
//! upstream-aligned.
//!
//! Replaces v0.37 / v0.38's SD-style `cascade_unet.rs` with the
//! actual Würstchen v3 / Stable Cascade prior architecture. Tensor
//! naming matches the inspected safetensors keys from
//! `stabilityai/stable-cascade-prior/prior/diffusion_pytorch_model.safetensors`
//! exactly:
//!
//! ```text
//!   embedding.1.{weight,bias}              # input Conv2d (c_in → c_hidden, 1×1)
//!   clip_txt_mapper.{weight,bias}          # Linear (c_clip_text → c_hidden)
//!   clip_txt_pooled_mapper.{weight,bias}   # Linear (c_clip_text_pooled → c_cond * num_pooled_tokens)
//!   clip_img_mapper.{weight,bias}          # Linear (c_clip_img → c_cond * num_pooled_tokens)  [Stage C only]
//!   down_blocks.{level}.{pos}.{block_keys}
//!   down_downscalers.{level}.1.blocks.0.{weight,bias}   # 1×1 Conv at level→level+1 boundary
//!   up_blocks.{level}.{pos}.{block_keys}
//!   up_upscalers.{level}.1.blocks.1.{weight,bias}       # 1×1 Conv at level→level-1 boundary
//!   clf.1.{weight,bias}                    # output Conv2d (c_hidden → c_out, 1×1)
//! ```
//!
//! ## Block sequence
//!
//! Each (down_blocks | up_blocks) at level `L` is `blocks_per_level[L] * 3`
//! sub-blocks. The pattern is strictly repeating `[Res, Time, Attn]`:
//! position 0 is `ResBlock`, position 1 is `TimestepBlock`, position 2 is
//! `AttnBlock`, position 3 starts the next triple, and so on. Verified by
//! inspection of the upstream safetensors keys at every position in both
//! Stage C levels (24 + 72 sub-blocks).
//!
//! ## Up/down samplers
//!
//! `down_downscalers` and `up_upscalers` are sparse — only positioned at
//! level boundaries. Implemented as `Sequential(LayerNorm2d, UpDownBlock)`
//! where `UpDownBlock` carries a 1×1 Conv2d. Spatial scaling is via
//! nearest-2× upsample (for up) or 2× avg-pool (for down) at the
//! parameterless `interp` slot inside the `UpDownBlock`. The tensor
//! key path follows upstream: the Conv lives at
//! `{kind}.{level}.1.blocks.{0 if down, 1 if up}.{weight,bias}`.
//!
//! ## v0.39 phase 0b scope
//!
//! Stage C only. Phase 0c extends to Stage B (adds `effnet_mapper`
//! + `pixels_mapper` paths, drops `clip_img_mapper`).

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, Module, Tensor, Var};
use candle_nn::{self as nn, VarBuilder};
use std::sync::{Arc, RwLock};

use crate::pipelines::cascade_blocks::{
    AttnBlock, LayerNorm2d, ResBlock, TimestepBlock,
};
use crate::pipelines::lora_linear::LoraRegistry;

/// Architectural config for a Stable Cascade prior UNet (Stage B
/// or Stage C variant).
///
/// Upstream defaults (derived from safetensors-header inspection at
/// v0.39 phase 0):
///
/// **Stage C** (`stabilityai/stable-cascade-prior/prior/`):
/// `c_in=c_out=16`, `c_hidden_per_level=[2048, 2048]`,
/// `has_attention_per_level=[true, true]`, `has_sca=has_crp=true`,
/// `c_clip_text=Some(1280)`, `c_clip_img=Some(768)`,
/// `blocks_per_level=[8, 24]`, `sampler_style=OnePixel`.
///
/// **Stage B** (`stabilityai/stable-cascade/decoder/`):
/// `c_in=c_out=16`, `c_hidden_per_level=[320, 640, 1280, 1280]`,
/// `has_attention_per_level=[false, false, true, true]`,
/// `has_sca=true`, `has_crp=false`,
/// `c_clip_text=None` (pooled-only), `c_clip_img=None`,
/// `effnet_input_channels=Some(16)`, `pixels_input_channels=Some(3)`,
/// `blocks_per_level=[2, 6, 28, 6]`, `sampler_style=Strided`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Input channels at `embedding.1`. Always 16 (Stage A quantized
    /// latent channel count, shared by both stages).
    pub c_in: usize,
    /// Output channels at `clf.1`. Always 16.
    pub c_out: usize,
    /// Hidden channels per level, shallowest-first. Length = number
    /// of levels.
    pub c_hidden_per_level: Vec<usize>,
    /// Time conditioning dim. Always 64 — feeds every `mapper*` of
    /// every `TimestepBlock`.
    pub c_cond: usize,
    /// CLIP-G text sequence dim. `Some(1280)` for Stage C (projected
    /// via `clip_txt_mapper` to per-level c_hidden before entering
    /// `AttnBlock.kv_mapper`). `None` for Stage B (no
    /// `clip_txt_mapper` in the upstream checkpoint).
    pub c_clip_text: Option<usize>,
    /// CLIP-G pooled text dim (1280 in both stages).
    pub c_clip_text_pooled: usize,
    /// CLIP-H image dim. `Some(768)` for Stage C; `None` for
    /// Stage B.
    pub c_clip_img: Option<usize>,
    /// Number of "pooled conditioning tokens" — the pooled text /
    /// image mappers project to `num_pooled_tokens * c_pooled_token`.
    /// Upstream Stable Cascade uses **4** in both stages (verified at
    /// v0.40 phase 2: Stage C output 8192 = 4 × 2048; Stage B output
    /// 5120 = 4 × 1280).
    pub num_pooled_tokens: usize,
    /// Per-token dim of the pooled conditioning streams. Equals the
    /// `c_hidden` value the corresponding AttnBlock `kv_mapper`
    /// expects as input. Stage C: 2048 (uniform c_hidden). Stage B:
    /// 1280 (the bottleneck c_hidden_per_level value, matching the
    /// attention levels).
    pub c_pooled_token: usize,
    /// Attention heads per `AttnBlock`. Used only at levels where
    /// `has_attention_per_level[i] = true`. Head dim is always
    /// 64 → num_heads = c_hidden_per_level[i] / 64.
    pub head_dim: usize,
    /// Whether each level has `AttnBlock`s in its triple. False
    /// levels run shorter [Res, Time] pairs (2 sub-blocks/triple)
    /// instead of [Res, Time, Attn] (3 sub-blocks/triple).
    pub has_attention_per_level: Vec<bool>,
    /// Whether `TimestepBlock` has a `mapper_sca`. True for both
    /// stages currently.
    pub has_sca: bool,
    /// Whether `TimestepBlock` has a `mapper_crp`. True for Stage C
    /// only; Stage B uses 2 mappers (mapper + mapper_sca).
    pub has_crp: bool,
    /// Number of `[Res, Time, Attn]` (or `[Res, Time]`) triples per
    /// level. Length = number of levels.
    pub blocks_per_level: Vec<usize>,
    /// Effnet conditioning input channels (Stage C prior latent
    /// channels = 16). `Some` for Stage B; `None` for Stage C.
    pub effnet_input_channels: Option<usize>,
    /// Pixels conditioning input channels (RGB = 3). `Some` for
    /// Stage B; `None` for Stage C.
    pub pixels_input_channels: Option<usize>,
    /// Down/up sampler topology at level boundaries. Stage C uses
    /// 1×1 conv + parameterless interp; Stage B uses 2×2 stride 2
    /// (Conv2d for down, ConvTranspose2d for up).
    pub sampler_style: SamplerStyle,
    /// Upstream `switch_level` — one bool per level transition
    /// (length = num_levels - 1). When false, the OnePixel sampler's
    /// interpolation slot is Identity (spatial preserved); when true,
    /// it resamples 2×/0.5×. Stage C is `[false]` (its 2 levels are
    /// both 24×24). Ignored by the Strided style (Stage B), whose
    /// strided convs always resample.
    pub switch_level: Vec<bool>,
    /// Upstream `up_blocks_repeat_mappers` — per up-level (deepest
    /// first) count of how many times the level's block group runs.
    /// `up_repeat_mappers[level]` has `value - 1` 1×1 convs applied
    /// between repeats. Stage C is `[1, 1]` (runs once, no mappers);
    /// Stage B is `[3, 3, 2, 2]`.
    pub up_blocks_repeat_mappers: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerStyle {
    /// Stage C: `Sequential(LayerNorm2d, UpDownBlock(blocks=[1×1Conv,
    /// interp]))`. Channel-preserving (c_in == c_out); spatial 2×
    /// via avg_pool (down) or nearest upsample (up).
    OnePixel,
    /// Stage B: `Sequential(LayerNorm2d, Conv2d kernel=2 stride=2)`
    /// for down or `Sequential(LayerNorm2d, ConvTranspose2d
    /// kernel=2 stride=2)` for up. Combines channel change and
    /// spatial change in one strided conv.
    Strided,
}

impl Config {
    /// Upstream `stabilityai/stable-cascade-prior` Stage C config.
    pub fn stage_c_full() -> Self {
        Self {
            c_in: 16,
            c_out: 16,
            c_hidden_per_level: vec![2048, 2048],
            c_cond: 64,
            c_clip_text: Some(1280),
            c_clip_text_pooled: 1280,
            c_clip_img: Some(768),
            num_pooled_tokens: 4,
            c_pooled_token: 2048,
            head_dim: 64,
            has_attention_per_level: vec![true, true],
            has_sca: true,
            has_crp: true,
            blocks_per_level: vec![8, 24],
            effnet_input_channels: None,
            pixels_input_channels: None,
            sampler_style: SamplerStyle::OnePixel,
            switch_level: vec![false],
            up_blocks_repeat_mappers: vec![1, 1],
        }
    }

    /// Upstream `stabilityai/stable-cascade/decoder/` Stage B config.
    pub fn stage_b_full() -> Self {
        Self {
            c_in: 16,
            c_out: 16,
            c_hidden_per_level: vec![320, 640, 1280, 1280],
            c_cond: 64,
            c_clip_text: None,
            c_clip_text_pooled: 1280,
            c_clip_img: None,
            num_pooled_tokens: 4,
            c_pooled_token: 1280,
            head_dim: 64,
            has_attention_per_level: vec![false, false, true, true],
            has_sca: true,
            has_crp: false,
            blocks_per_level: vec![2, 6, 28, 6],
            effnet_input_channels: Some(16),
            pixels_input_channels: Some(3),
            sampler_style: SamplerStyle::Strided,
            // 4 levels → 3 transitions. Strided ignores these.
            switch_level: vec![true, true, true],
            up_blocks_repeat_mappers: vec![3, 3, 2, 2],
        }
    }

    /// Width at the shallowest level (used by `embedding` + `clf`).
    pub fn c_hidden_first(&self) -> usize {
        self.c_hidden_per_level[0]
    }
}

// ---------------------------------------------------------------------
// UpDownBlock — sparse up/downsample at level boundaries.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum SampleMode {
    Down,
    Up,
}

/// Level-boundary down/up sampler. Two upstream-aligned topologies:
///
/// - `OnePixel` (Stage C): `Sequential(LayerNorm2d, UpDownBlock(
///   blocks=[1×1 Conv2d, interp]))`. Channel-preserving. Spatial 2×
///   via `avg_pool2d` (down) or `upsample_nearest2d` (up).
///   Conv2d tensor key: `1.blocks.0.{weight,bias}` (down) or
///   `1.blocks.1.{weight,bias}` (up).
///
/// - `Strided` (Stage B): `Sequential(LayerNorm2d, Conv2d
///   kernel=2 stride=2)` (down) or `Sequential(LayerNorm2d,
///   ConvTranspose2d kernel=2 stride=2)` (up). Combines channel
///   change and spatial change. Conv tensor key: `1.{weight,bias}`.
pub enum UpDownBlock {
    OnePixel {
        norm: LayerNorm2d,
        conv: nn::Conv2d,
        mode: SampleMode,
        /// `switch_level` flag — when false (Stage C's only transition),
        /// the interpolation slot is `nn.Identity` and spatial
        /// resolution is preserved; the block is just norm + 1×1 conv.
        /// When true, bilinear 2×/0.5× resampling (upstream uses
        /// bilinear, align_corners=True).
        enabled: bool,
    },
    StridedDown {
        norm: LayerNorm2d,
        conv: nn::Conv2d,
    },
    StridedUp {
        norm: LayerNorm2d,
        conv_t: nn::ConvTranspose2d,
    },
}

impl UpDownBlock {
    /// Construct a OnePixel-style sampler (Stage C). Channels are
    /// preserved (in == out).
    pub fn new_one_pixel(
        channels: usize,
        mode: SampleMode,
        enabled: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = LayerNorm2d::new(channels, 1e-6);
        let block_idx = match mode {
            SampleMode::Down => "0",
            SampleMode::Up => "1",
        };
        let conv = nn::conv2d(
            channels,
            channels,
            1,
            Default::default(),
            vb.pp("1").pp("blocks").pp(block_idx),
        )
        .map_err(|e| anyhow!("UpDownBlock::OnePixel conv: {e}"))?;
        Ok(Self::OnePixel { norm, conv, mode, enabled })
    }

    /// Construct a Strided down-sampler (Stage B). 2×2 stride 2
    /// Conv2d changes channels + halves spatial in one step.
    pub fn new_strided_down(
        in_c: usize,
        out_c: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = LayerNorm2d::new(in_c, 1e-6);
        let conv = nn::conv2d(
            in_c,
            out_c,
            2,
            nn::Conv2dConfig {
                stride: 2,
                padding: 0,
                ..Default::default()
            },
            vb.pp("1"),
        )
        .map_err(|e| anyhow!("UpDownBlock::StridedDown conv: {e}"))?;
        Ok(Self::StridedDown { norm, conv })
    }

    /// Construct a Strided up-sampler (Stage B). 2×2 stride 2
    /// ConvTranspose2d changes channels + doubles spatial.
    pub fn new_strided_up(
        in_c: usize,
        out_c: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = LayerNorm2d::new(in_c, 1e-6);
        let conv_t = nn::conv_transpose2d(
            in_c,
            out_c,
            2,
            nn::ConvTranspose2dConfig {
                stride: 2,
                padding: 0,
                ..Default::default()
            },
            vb.pp("1"),
        )
        .map_err(|e| anyhow!("UpDownBlock::StridedUp conv_t: {e}"))?;
        Ok(Self::StridedUp { norm, conv_t })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            UpDownBlock::OnePixel { norm, conv, mode, enabled } => {
                let x = norm.forward(x)?;
                // Upstream UpDownBlock2d: down = [conv, interp],
                // up = [interp, conv]. When `enabled` is false the
                // interp slot is Identity (no spatial change).
                match mode {
                    SampleMode::Down => {
                        let x = conv.forward(&x)?;
                        if *enabled { Ok(x.avg_pool2d(2)?) } else { Ok(x) }
                    }
                    SampleMode::Up => {
                        if *enabled {
                            let (_b, _c, h, w) = x.dims4()?;
                            let up = x.upsample_nearest2d(h * 2, w * 2)?;
                            Ok(conv.forward(&up)?)
                        } else {
                            Ok(conv.forward(&x)?)
                        }
                    }
                }
            }
            UpDownBlock::StridedDown { norm, conv } => {
                let x = norm.forward(x)?;
                Ok(conv.forward(&x)?)
            }
            UpDownBlock::StridedUp { norm, conv_t } => {
                let x = norm.forward(x)?;
                Ok(conv_t.forward(&x)?)
            }
        }
    }
}

// ---------------------------------------------------------------------
// Block — one of [Res, Time, Attn] for the position-indexed sequence.
// ---------------------------------------------------------------------

pub enum Block {
    Res(ResBlock),
    Time(TimestepBlock),
    Attn(AttnBlock),
}

impl Block {
    /// True for `Block::Res`. Used by ControlNet injection, which
    /// counts ResBlocks across the down+up path and injects before
    /// each (the upstream cnet deliverer only fires before ResBlocks).
    pub fn is_res(&self) -> bool {
        matches!(self, Block::Res(_))
    }

    /// Forward without skip. Errors if the block is a ResBlock that
    /// was constructed with `c_skip > 0`.
    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
        clip: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Block::Res(b) => b.forward(x),
            Block::Time(b) => b.forward(x, t_emb, sca_emb, crp_emb),
            Block::Attn(b) => b.forward(x, clip),
        }
    }

    /// v0.40 phase 3 iter 1: forward with an optional skip. Required
    /// at the first ResBlock of each up-path level (after the
    /// upscaler); the skip carries the channel-concatenated feature
    /// from the matching down level. `Time` / `Attn` blocks ignore
    /// the skip arg.
    pub fn forward_maybe_skip(
        &self,
        x: &Tensor,
        x_skip: Option<&Tensor>,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
        clip: &Tensor,
    ) -> Result<Tensor> {
        match (self, x_skip) {
            (Block::Res(b), Some(skip)) if b.c_skip() > 0 => b.forward_with_skip(x, skip),
            (Block::Res(b), _) => b.forward(x),
            (Block::Time(b), _) => b.forward(x, t_emb, sca_emb, crp_emb),
            (Block::Attn(b), _) => b.forward(x, clip),
        }
    }

    /// Returns true if this is a ResBlock constructed with `c_skip > 0`.
    pub fn requires_skip(&self) -> bool {
        matches!(self, Block::Res(b) if b.c_skip() > 0)
    }
}

// ---------------------------------------------------------------------
// StableCascadePrior — Stage C UNet (v0.39 phase 0b).
// ---------------------------------------------------------------------

/// Stable Cascade prior UNet (Stage C variant in v0.39 phase 0b).
///
/// Stage B variant lands in phase 0c (same struct, different config
/// + the `effnet_mapper` / `pixels_mapper` paths).
pub struct StableCascadePrior {
    embedding_conv: nn::Conv2d,
    embedding_norm: LayerNorm2d,
    /// `Some` for Stage C; `None` for Stage B (no `clip_txt_mapper`
    /// in the upstream Stage B checkpoint).
    clip_txt_mapper: Option<nn::Linear>,
    clip_txt_pooled_mapper: nn::Linear,
    /// `Some` for Stage C; `None` for Stage B.
    clip_img_mapper: Option<nn::Linear>,
    /// `Some` for Stage B: pair of 1×1 Conv2d (Sequential indices
    /// .0 and .2 with a parameterless GELU at .1). Projects Stage
    /// C output (effnet_input_channels) → c_hidden_per_level[0].
    effnet_mapper: Option<(nn::Conv2d, nn::Conv2d)>,
    /// `Some` for Stage B: pair of 1×1 Conv2d. Projects RGB pixel
    /// input (pixels_input_channels=3) → c_hidden_per_level[0].
    pixels_mapper: Option<(nn::Conv2d, nn::Conv2d)>,
    down_blocks: Vec<Vec<Block>>,
    down_downscalers: Vec<UpDownBlock>,
    up_blocks: Vec<Vec<Block>>,
    up_upscalers: Vec<UpDownBlock>,
    /// Per up-level (deepest first) 1×1 convs applied between block-
    /// group repeats. `up_repeat_mappers[level]` has
    /// `up_blocks_repeat_mappers[level] - 1` entries.
    up_repeat_mappers: Vec<Vec<nn::Conv2d>>,
    clf_norm: LayerNorm2d,
    clf_conv: nn::Conv2d,
    pub cfg: Config,
    pub dtype: DType,
    pub device: Device,
    /// v1.10.0: path-keyed LoRA registry over every Stage-C attention
    /// projection (`*.attention.to_{q,k,v,out.0}.weight`). Populated
    /// during construction by `wrap_linear` in `cascade_blocks.rs`;
    /// consumed by `install_train_adapters` (`plakat style train`).
    /// Stage B builds + carries one too (it just never trains it).
    lora_registry: LoraRegistry,
}

impl StableCascadePrior {
    /// Construct Stage C prior. `vb` should point at the safetensors
    /// root — every tensor key matches the upstream convention.
    pub fn new_stage_c(cfg: Config, vb: VarBuilder) -> Result<Self> {
        anyhow::ensure!(
            cfg.sampler_style == SamplerStyle::OnePixel,
            "Stage C uses OnePixel sampler style"
        );
        Self::new(cfg, vb)
    }

    /// Construct Stage B prior. `vb` should point at the
    /// `decoder/diffusion_pytorch_model.safetensors` root.
    pub fn new_stage_b(cfg: Config, vb: VarBuilder) -> Result<Self> {
        anyhow::ensure!(
            cfg.sampler_style == SamplerStyle::Strided,
            "Stage B uses Strided sampler style"
        );
        anyhow::ensure!(
            cfg.effnet_input_channels.is_some(),
            "Stage B requires effnet_input_channels"
        );
        Self::new(cfg, vb)
    }

    /// Shared constructor — Config drives which mappers/samplers to
    /// build.
    fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        // v1.10.0: shared LoRA registry — every wrapped Stage-C
        // attention Linear registers its slots/train handles here
        // (mirrors `PixArtSigmaXL::new`).
        let registry = Arc::new(RwLock::new(LoraRegistry::new()));
        let c_first = cfg.c_hidden_first();
        let num_levels = cfg.blocks_per_level.len();
        anyhow::ensure!(
            cfg.c_hidden_per_level.len() == num_levels
                && cfg.has_attention_per_level.len() == num_levels,
            "Config arity mismatch: blocks_per_level={} c_hidden_per_level={} has_attn={}",
            cfg.blocks_per_level.len(),
            cfg.c_hidden_per_level.len(),
            cfg.has_attention_per_level.len()
        );

        // ---- Input embedding: Sequential(PixelUnshuffle(1), Conv2d, LayerNorm2d) ----
        let embedding_conv = nn::conv2d(
            cfg.c_in,
            c_first,
            1,
            Default::default(),
            vb.pp("embedding").pp("1"),
        )
        .map_err(|e| anyhow!("embedding.1: {e}"))?;
        let embedding_norm = LayerNorm2d::new(c_first, 1e-6);

        // ---- CLIP conditioning mappers ----
        let clip_txt_mapper = if let Some(c_clip_text) = cfg.c_clip_text {
            Some(
                nn::linear(c_clip_text, c_first, vb.pp("clip_txt_mapper"))
                    .map_err(|e| anyhow!("clip_txt_mapper: {e}"))?,
            )
        } else {
            None
        };
        // v0.40 phase 2: pooled mapper output = N tokens × c_hidden
        // (NOT c_cond × N). Upstream Stage C has 8192 = 4 × 2048;
        // Stage B has 5120 = 4 × 1280.
        let pooled_out_dim = cfg.num_pooled_tokens * cfg.c_pooled_token;
        let clip_txt_pooled_mapper = nn::linear(
            cfg.c_clip_text_pooled,
            pooled_out_dim,
            vb.pp("clip_txt_pooled_mapper"),
        )
        .map_err(|e| anyhow!("clip_txt_pooled_mapper: {e}"))?;
        let clip_img_mapper = if let Some(c_clip_img) = cfg.c_clip_img {
            Some(
                nn::linear(c_clip_img, pooled_out_dim, vb.pp("clip_img_mapper"))
                    .map_err(|e| anyhow!("clip_img_mapper: {e}"))?,
            )
        } else {
            None
        };

        // ---- Stage B effnet + pixels mappers (Sequential(Conv, GELU, Conv)) ----
        let effnet_mid = 4 * c_first; // upstream uses 4× hidden as mid width
        let effnet_mapper = if let Some(c_eff) = cfg.effnet_input_channels {
            let m0 = nn::conv2d(
                c_eff,
                effnet_mid,
                1,
                Default::default(),
                vb.pp("effnet_mapper").pp("0"),
            )
            .map_err(|e| anyhow!("effnet_mapper.0: {e}"))?;
            let m2 = nn::conv2d(
                effnet_mid,
                c_first,
                1,
                Default::default(),
                vb.pp("effnet_mapper").pp("2"),
            )
            .map_err(|e| anyhow!("effnet_mapper.2: {e}"))?;
            Some((m0, m2))
        } else {
            None
        };
        let pixels_mapper = if let Some(c_pix) = cfg.pixels_input_channels {
            let m0 = nn::conv2d(
                c_pix,
                effnet_mid,
                1,
                Default::default(),
                vb.pp("pixels_mapper").pp("0"),
            )
            .map_err(|e| anyhow!("pixels_mapper.0: {e}"))?;
            let m2 = nn::conv2d(
                effnet_mid,
                c_first,
                1,
                Default::default(),
                vb.pp("pixels_mapper").pp("2"),
            )
            .map_err(|e| anyhow!("pixels_mapper.2: {e}"))?;
            Some((m0, m2))
        } else {
            None
        };

        // ---- Down blocks: per-level width + attn flag (no skip-concat) ----
        let down_blocks = build_block_levels(
            &cfg.blocks_per_level,
            &cfg.c_hidden_per_level,
            cfg.c_cond,
            cfg.head_dim,
            &cfg.has_attention_per_level,
            cfg.has_sca,
            cfg.has_crp,
            None,
            vb.pp("down_blocks"),
            &registry,
        )?;

        // ---- Down downscalers (one per boundary; index .{level}.1) ----
        let mut down_downscalers = Vec::with_capacity(num_levels - 1);
        for i in 1..num_levels {
            let in_c = cfg.c_hidden_per_level[i - 1];
            let out_c = cfg.c_hidden_per_level[i];
            let sub_vb = vb.pp("down_downscalers").pp(&i.to_string());
            let blk = match cfg.sampler_style {
                SamplerStyle::OnePixel => {
                    anyhow::ensure!(
                        in_c == out_c,
                        "OnePixel downscaler requires in==out (got {in_c}/{out_c})"
                    );
                    UpDownBlock::new_one_pixel(
                        in_c,
                        SampleMode::Down,
                        cfg.switch_level[i - 1],
                        sub_vb,
                    )?
                }
                SamplerStyle::Strided => UpDownBlock::new_strided_down(in_c, out_c, sub_vb)?,
            };
            down_downscalers.push(blk);
        }

        // ---- Up blocks (mirror of down: deepest first, shallowest last) ----
        // v0.40 phase 3 iter 1: skip-concat at the FIRST ResBlock of each
        // up level i > 0 (after the upscaler brings us to the matching
        // down level's spatial). At up level 0 (deepest), we START from
        // the down level (n-1) output, so no skip-concat there.
        // At up level i, skip channels = down level (n-1-i)'s c_hidden
        // = up_c_hidden[i] (since up width at level i == down width at
        // level n-1-i, and the upscaler brings channels to match).
        let up_blocks_per_level: Vec<usize> =
            cfg.blocks_per_level.iter().rev().copied().collect();
        let up_c_hidden: Vec<usize> =
            cfg.c_hidden_per_level.iter().rev().copied().collect();
        let up_has_attn: Vec<bool> =
            cfg.has_attention_per_level.iter().rev().copied().collect();
        let up_skip_dims: Vec<usize> = (0..num_levels)
            .map(|i| if i == 0 { 0 } else { up_c_hidden[i] })
            .collect();
        let up_blocks = build_block_levels(
            &up_blocks_per_level,
            &up_c_hidden,
            cfg.c_cond,
            cfg.head_dim,
            &up_has_attn,
            cfg.has_sca,
            cfg.has_crp,
            Some(&up_skip_dims),
            vb.pp("up_blocks"),
            &registry,
        )?;

        // ---- Up upscalers (one per boundary, index .{level}.1) ----
        let mut up_upscalers = Vec::with_capacity(num_levels - 1);
        for i in 0..num_levels - 1 {
            let in_c = up_c_hidden[i];
            let out_c = up_c_hidden[i + 1];
            let sub_vb = vb.pp("up_upscalers").pp(&i.to_string());
            let blk = match cfg.sampler_style {
                SamplerStyle::OnePixel => {
                    anyhow::ensure!(
                        in_c == out_c,
                        "OnePixel upscaler requires in==out (got {in_c}/{out_c})"
                    );
                    // Up path mirrors the down transitions in reverse:
                    // up-index i corresponds to down transition
                    // (num_levels - 2 - i).
                    let enabled = cfg.switch_level[num_levels - 2 - i];
                    UpDownBlock::new_one_pixel(in_c, SampleMode::Up, enabled, sub_vb)?
                }
                SamplerStyle::Strided => UpDownBlock::new_strided_up(in_c, out_c, sub_vb)?,
            };
            up_upscalers.push(blk);
        }

        // ---- Up repeat mappers (1×1 convs between block-group
        // repeats). up_repeat_mappers.{level}.{j} for j in
        // 0..(up_blocks_repeat_mappers[level] - 1). ----
        let mut up_repeat_mappers: Vec<Vec<nn::Conv2d>> = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let n_map = cfg.up_blocks_repeat_mappers[level].saturating_sub(1);
            let c = up_c_hidden[level];
            let mut maps = Vec::with_capacity(n_map);
            for j in 0..n_map {
                let conv = nn::conv2d(
                    c,
                    c,
                    1,
                    Default::default(),
                    vb.pp("up_repeat_mappers").pp(&level.to_string()).pp(&j.to_string()),
                )
                .map_err(|e| anyhow!("up_repeat_mappers.{level}.{j}: {e}"))?;
                maps.push(conv);
            }
            up_repeat_mappers.push(maps);
        }

        // ---- Output classifier: Sequential(LayerNorm2d, Conv2d) ----
        let clf_norm = LayerNorm2d::new(c_first, 1e-6);
        let clf_conv = nn::conv2d(
            c_first,
            cfg.c_out,
            1,
            Default::default(),
            vb.pp("clf").pp("1"),
        )
        .map_err(|e| anyhow!("clf.1: {e}"))?;

        // Unwrap the shared registry into the owned struct field
        // (construction is done — no outstanding `Arc` clones remain
        // beyond the ones the wrapped Linears hold internally, which
        // are separate `Arc`s on the slot handles, not on the registry
        // map itself). Mirrors `PixArtSigmaXL::new`.
        let lora_registry = Arc::try_unwrap(registry)
            .map_err(|_| anyhow!("Cascade LoRA registry still has outstanding refs after construction"))?
            .into_inner()
            .map_err(|_| anyhow!("Cascade LoRA registry RwLock poisoned at construction"))?;

        Ok(Self {
            embedding_conv,
            embedding_norm,
            clip_txt_mapper,
            clip_txt_pooled_mapper,
            clip_img_mapper,
            effnet_mapper,
            pixels_mapper,
            down_blocks,
            down_downscalers,
            up_blocks,
            up_upscalers,
            up_repeat_mappers,
            clf_norm,
            clf_conv,
            cfg,
            dtype,
            device,
            lora_registry,
        })
    }

    /// `plakat style train` / DreamBooth: install a fresh **trainable**
    /// LoRA adapter on every Stage-C attention projection (registry keys
    /// containing `.attention.to_` — self/cross q/k/v/out; the
    /// conditioning `kv_mapper` is excluded). Returns `(registry_key, A,
    /// B)` for each, so the caller drives AdamW and writes the save.
    /// Standard init: `A ~ N(0, 0.02)`, `B = 0`, so the adapter starts as
    /// a no-op on the frozen base and learns the style delta. Vars are
    /// F32 (training dtype). Mirrors `PixArtSigmaXL::install_train_adapters`.
    pub fn install_train_adapters(
        &self,
        rank: usize,
        scale: f64,
        device: &Device,
    ) -> Result<Vec<(String, Var, Var)>> {
        let mut keys: Vec<&String> = self
            .lora_registry
            .keys()
            .filter(|k| {
                k.contains(".attention.to_q")
                    || k.contains(".attention.to_k")
                    || k.contains(".attention.to_v")
                    || k.contains(".attention.to_out")
            })
            .collect();
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
                .map_err(|_| anyhow!("Cascade train slot poisoned"))? =
                Some((a.clone(), b.clone(), scale));
            out.push((key.clone(), a, b));
        }
        Ok(out)
    }

    /// Build the AttnBlock KV conditioning sequence.
    ///
    /// Phase 0b/0c is conservative: returns the projected CLIP-G
    /// text sequence (Stage C) OR a zero-stream of the expected
    /// shape (Stage B, which has no `clip_txt_mapper`). The pooled
    /// text + pooled image streams are computed but not yet
    /// concatenated — the upstream feeding topology gets locked in
    /// at phase 0g during Pipeline integration.
    ///
    /// `clip_text`: `(B, T_text, c_clip_text)` — penultimate CLIP-G
    ///   hidden states. Required for Stage C; ignored for Stage B
    ///   (use the dummy-shape helper [`zero_kv_stream`] there).
    /// `clip_text_pooled`: `(B, c_clip_text_pooled)`.
    /// `clip_img`: `(B, c_clip_img)` or `None` (Stage C accepts None;
    ///   the pooled-image stream is then zeros).
    /// v0.40 phase 2: build the AttnBlock KV conditioning sequence.
    ///
    /// **Stage C** (has `clip_txt_mapper` + `clip_img_mapper`):
    /// returns the concatenation `(B, T_text + N_pooled + N_pooled,
    /// c_pooled_token)`:
    /// - text seq: `(B, T_text, c_pooled_token)` from
    ///   `clip_txt_mapper(clip_text)` — typically 77 tokens at 2048-dim.
    /// - pooled text: `(B, N_pooled, c_pooled_token)` from
    ///   `clip_txt_pooled_mapper(clip_pooled_text)` reshaped — typically
    ///   4 tokens at 2048-dim.
    /// - pooled image: `(B, N_pooled, c_pooled_token)` from
    ///   `clip_img_mapper(clip_pooled_img)` or zeros — 4 tokens at 2048.
    ///
    /// Total for Stage C upstream: `(B, 77 + 4 + 4 = 85, 2048)`.
    ///
    /// **Stage B** (no `clip_txt_mapper`, no `clip_img_mapper`):
    /// returns `(B, N_pooled, c_pooled_token)` — pooled-text only, at
    /// the bottleneck c_hidden=1280 that the attention-level
    /// `kv_mapper` consumes. Total upstream: `(B, 4, 1280)`.
    ///
    /// `clip_text`: `(B, T_text, c_clip_text)` — required for Stage C;
    ///   ignored for Stage B (any shape works since it's unused).
    /// `clip_text_pooled`: `(B, c_clip_text_pooled)`.
    /// `clip_img`: `(B, c_clip_img)` for Stage C (None → image stream
    ///   defaults to zeros); ignored for Stage B.
    pub fn build_clip_conditioning(
        &self,
        clip_text: &Tensor,
        clip_text_pooled: &Tensor,
        clip_img: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, _) = clip_text_pooled.dims2()?;
        // Pooled text is mandatory for both stages.
        let pooled_text = self
            .clip_txt_pooled_mapper
            .forward(clip_text_pooled)?
            .reshape((b, self.cfg.num_pooled_tokens, self.cfg.c_pooled_token))?;

        // Stage B path: pooled-only KV stream at c_hidden=c_pooled_token.
        let Some(text_mapper) = &self.clip_txt_mapper else {
            // v0.41 phase 2f: upstream applies clip_norm in BOTH the
            // pooled-only (Stage B) and concat (Stage C) return paths.
            return layer_norm_last_dim(&pooled_text, 1e-6);
        };

        // Stage C path: cat(text, pooled_text, pooled_img).
        let text = text_mapper.forward(clip_text)?;
        let pooled_img = if let (Some(img), Some(img_mapper)) =
            (clip_img, &self.clip_img_mapper)
        {
            img_mapper
                .forward(img)?
                .reshape((b, self.cfg.num_pooled_tokens, self.cfg.c_pooled_token))?
        } else {
            // No image conditioning provided; zero-pad the slot so the
            // sequence length matches upstream's expected 85 tokens.
            Tensor::zeros(
                (b, self.cfg.num_pooled_tokens, self.cfg.c_pooled_token),
                self.dtype,
                &self.device,
            )?
        };
        let clip = Tensor::cat(&[&text, &pooled_text, &pooled_img], 1)?;
        // v0.41 phase 2f: the missing final LayerNorm. Upstream
        // `get_clip_embeddings` ends `return self.clip_norm(clip)` —
        // `nn.LayerNorm(conditioning_dim, elementwise_affine=False,
        // eps=1e-6)` over the conditioning dim. Without it the KV
        // stream fed to every AttnBlock is unnormalised (off by ~80×
        // in magnitude per the phase-2f reference dump), so attention
        // produces garbage and the prior can't denoise.
        layer_norm_last_dim(&clip, 1e-6)
    }

    /// Build a zero KV stream of the right shape for an AttnBlock at
    /// the shallowest (or any specific) level. Used by Stage B
    /// callers that don't yet have a real-topology conditioning
    /// source (phase 0g will replace with the actual projection).
    ///
    /// `batch`: batch size.
    /// `level`: index into `c_hidden_per_level`; the returned tensor
    ///   matches `(batch, T, c_hidden_per_level[level])`.
    /// `seq_len`: token count (caller-chosen; typically
    ///   `num_pooled_tokens`).
    pub fn zero_kv_stream(
        &self,
        batch: usize,
        level: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let c = self.cfg.c_hidden_per_level[level];
        Tensor::zeros((batch, seq_len, c), self.dtype, &self.device)
            .map_err(|e| e.into())
    }

    /// Apply effnet conditioning (Stage B only). Returns the
    /// projected `(B, c_hidden_first, h, w)` tensor matching the
    /// embedding output spatial. Caller must spatially align effnet
    /// to the embedding output dims BEFORE calling this (upstream
    /// uses bilinear interpolation; phase 0g wires the alignment).
    pub fn apply_effnet_mapper(&self, effnet: &Tensor) -> Result<Tensor> {
        let mapper = self
            .effnet_mapper
            .as_ref()
            .ok_or_else(|| anyhow!("apply_effnet_mapper called on a Stage C prior"))?;
        let h = mapper.0.forward(effnet)?;
        let h = h.gelu_erf()?;
        let h = mapper.1.forward(&h)?;
        // v0.41 phase 2g: upstream effnet_mapper ends with
        // SDCascadeLayerNorm (no affine, eps 1e-6) over the channel
        // dim. Missing it left the effnet conditioning unnormalised
        // and corrupted the down path from down_lvl0 onward. The norm
        // is parameterless, so construct it inline.
        LayerNorm2d::new(self.cfg.c_hidden_first(), 1e-6).forward(&h)
    }

    /// Apply pixels conditioning (Stage B only). See
    /// [`apply_effnet_mapper`] for shape semantics.
    pub fn apply_pixels_mapper(&self, pixels: &Tensor) -> Result<Tensor> {
        let mapper = self
            .pixels_mapper
            .as_ref()
            .ok_or_else(|| anyhow!("apply_pixels_mapper called on a Stage C prior"))?;
        let h = mapper.0.forward(pixels)?;
        let h = h.gelu_erf()?;
        let h = mapper.1.forward(&h)?;
        LayerNorm2d::new(self.cfg.c_hidden_first(), 1e-6).forward(&h)
    }

    /// Forward pass.
    ///
    /// `x`: `(B, c_in, h, w)` noisy prior latent.
    /// `t_emb`: `(B, c_cond)` sinusoidal time encoding.
    /// `sca_emb`: `(B, c_cond)` — required when `cfg.has_sca`.
    /// `crp_emb`: `(B, c_cond)` — required when `cfg.has_crp`
    ///   (Stage C only); pass `None` for Stage B.
    /// `clip`: pre-built KV conditioning sequence `(B, T, c_hidden)`
    ///   matching the relevant level's c_hidden. For Stage B levels
    ///   with `has_attention[level] = true` the sequence width must
    ///   match `c_hidden_per_level[level]`; for levels without
    ///   attention the clip stream is ignored.
    /// `effnet`: `(B, c_eff_in, h', w')` projected via
    ///   `apply_effnet_mapper` and added to the embedding output.
    ///   Stage B only; pass `None` for Stage C. Caller must
    ///   pre-resize effnet to match the embedding spatial dims.
    /// `pixels`: `(B, 3, h'', w'')` for Stage B pixel conditioning.
    ///   `None` for Stage C.
    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
        clip: &Tensor,
        effnet: Option<&Tensor>,
        pixels: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_inner(
            x, t_emb, sca_emb, crp_emb, clip, effnet, pixels, None, None, 0.0, None,
        )
    }

    /// Verify tap (`plakat verify` Tier 1, `stage_c.block0`): the input embedding
    /// (conv + norm) followed by the FIRST `Res` + `Time` sub-blocks of down level 0 —
    /// the conditioned-conv core, tapped BEFORE the first `Attn`. Corresponds to a diffusers
    /// forward hook on `down_blocks[0][1]` (the first `SDCascadeTimestepBlock`).
    ///
    /// Stops before `Attn` deliberately: the attention self-attends over 24×24 = 576
    /// near-uniform tokens of the synthetic white-noise verify latent, which is numerically
    /// ill-conditioned (candle-vs-torch softmax/matmul accumulation diverges on OOD input —
    /// the same effect that made the deep `stage_c.out` a coarse 0.989 gate). Res (conv) +
    /// Time (conditioning modulation) are well-conditioned on any input → fine corr 1.0.
    /// Additive; Stage C has no effnet/pixels path.
    pub fn capture_block0(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
        clip: &Tensor,
    ) -> Result<Tensor> {
        let mut h = self.embedding_conv.forward(x)?;
        h = self.embedding_norm.forward(&h)?;
        // Strict [Res, Time, Attn] repeat — take the first Res + Time (stop before Attn).
        for block in self.down_blocks[0].iter().take(2) {
            h = block.forward(&h, t_emb, sca_emb, crp_emb, clip)?;
        }
        Ok(h)
    }

    /// v0.41 phase 2f: forward that also collects named intermediate
    /// activations (emb, per-down-level, per-up-level, clf) for
    /// reference comparison against the diffusers dump. Test-only.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_collect(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
        clip: &Tensor,
        effnet: Option<&Tensor>,
        pixels: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut dump: Vec<(String, Tensor)> = Vec::new();
        let out = self.forward_inner(
            x, t_emb, sca_emb, crp_emb, clip, effnet, pixels, None, None, 0.0, Some(&mut dump),
        )?;
        Ok((out, dump))
    }

    /// v0.41 phase 3: forward pass with ControlNet residual injection.
    ///
    /// `cn_residuals[j]` is the j-th CN projection head's output; it is
    /// injected BEFORE the ResBlock at global index `cn_blocks[j]` (in
    /// the down→up ResBlock sequence), bilinearly upsampled to the
    /// current feature spatial and added with the `cn_scale` strength.
    /// For the canny CN: 8 residuals, `cn_blocks = [0,4,8,12,51,55,59,
    /// 63]` — 4 in the down path, 4 in the up path. This matches the
    /// upstream cnet deliverer (Stability-AI/StableCascade stage_c.py),
    /// which fires only before ResBlocks.
    ///
    /// Stage C only — by design, not a limitation. Stable Cascade's
    /// decoupled architecture applies ControlNet (and LoRA) to Stage C
    /// **alone**; Stages B/A are fixed and "do not need to be updated"
    /// (Stability-AI/StableCascade). Stage B is a semantic-compressor /
    /// super-resolver that preserves Stage C's structure through the
    /// decode, so a Stage-B CN is redundant — and no upstream Stage-B CN
    /// weights exist to align against. The guard below asserts that
    /// invariant (effnet path ⇒ Stage B ⇒ no CN).
    pub fn forward_with_cn(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
        clip: &Tensor,
        cn_residuals: &[Tensor],
        cn_blocks: &[usize],
        cn_scale: f32,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            self.cfg.effnet_input_channels.is_none(),
            "forward_with_cn is Stage C only by design — Stable Cascade applies ControlNet to Stage C alone; Stage B is a fixed decoder (no Stage-B CN exists)"
        );
        anyhow::ensure!(
            !cn_residuals.is_empty() && cn_residuals.len() == cn_blocks.len(),
            "cn_residuals ({}) must be non-empty and match cn_blocks ({})",
            cn_residuals.len(), cn_blocks.len()
        );
        self.forward_inner(
            x,
            t_emb,
            sca_emb,
            crp_emb,
            clip,
            None,
            None,
            Some(cn_residuals),
            Some(cn_blocks),
            cn_scale,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        sca_emb: Option<&Tensor>,
        crp_emb: Option<&Tensor>,
        clip: &Tensor,
        effnet: Option<&Tensor>,
        pixels: Option<&Tensor>,
        cn_residuals: Option<&[Tensor]>,
        cn_blocks: Option<&[usize]>,
        cn_scale: f32,
        mut dump: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        // ---- Input embedding ----
        let mut h = self.embedding_conv.forward(x)?;
        h = self.embedding_norm.forward(&h)?;
        if let Some(d) = dump.as_deref_mut() {
            d.push(("emb".to_string(), h.clone()));
        }
        // v0.41 phase 2g: upstream interpolates effnet to the
        // embedding spatial with BILINEAR (align_corners=True) BEFORE
        // the effnet_mapper, then adds. v0.40's nearest approximation
        // was a measurable error (down_lvl0 diverged 12.4 on a ±34
        // range in the phase-2g reference dump).
        if let Some(eff) = effnet {
            let (_, _, hh, hw) = h.dims4()?;
            let (_, _, eh, ew) = eff.dims4()?;
            let eff_aligned = if (eh, ew) != (hh, hw) {
                eff.upsample_bilinear2d(hh, hw, true)?
            } else {
                eff.clone()
            };
            h = h.add(&self.apply_effnet_mapper(&eff_aligned)?)?;
        }
        // Pixels path: upstream ALWAYS applies pixels_mapper when the
        // module exists (Stage B), defaulting `pixels` to
        // `zeros(B, 3, 8, 8)` when not supplied — and pixels_mapper's
        // conv biases + final LayerNorm make that a non-zero learned
        // constant, NOT a no-op. v0.41 phase 2g: skipping it when
        // pixels=None left every Stage B forward missing this additive
        // term (rb_in diverged 5.87 from emb in the reference dump).
        // Interpolate AFTER the mapper (unlike effnet, before).
        if self.pixels_mapper.is_some() {
            let (b, _, hh, hw) = h.dims4()?;
            let px_owned;
            let px = match pixels {
                Some(p) => p,
                None => {
                    px_owned = Tensor::zeros((b, 3, 8, 8), h.dtype(), h.device())?;
                    &px_owned
                }
            };
            let mapped = self.apply_pixels_mapper(px)?;
            let (_, _, ph, pw) = mapped.dims4()?;
            let mapped_aligned = if (ph, pw) != (hh, hw) {
                mapped.upsample_bilinear2d(hh, hw, true)?
            } else {
                mapped
            };
            h = h.add(&mapped_aligned)?;
        }

        // ---- CN injection helper ----
        // v0.41 phase 3: upstream injects a ControlNet residual BEFORE
        // each ResBlock whose global index (across the down+up ResBlock
        // sequence, down=0.., up continues) is in `cn_blocks`. The
        // residual (CN projection head output, at the backbone's small
        // spatial) is bilinearly upsampled to the current feature
        // spatial and added. `cn_scale` is the user strength.
        // `rb_idx` is the running ResBlock counter shared down→up.
        let inject_cn = |h: &Tensor, rb_idx: usize| -> Result<Tensor> {
            if let (Some(res), Some(blocks)) = (cn_residuals, cn_blocks) {
                if let Some(j) = blocks.iter().position(|&b| b == rb_idx) {
                    let r = &res[j];
                    let (_, _, hh, hw) = h.dims4()?;
                    let (_, _, rh, rw) = r.dims4()?;
                    let r_aligned = if (rh, rw) != (hh, hw) {
                        r.upsample_bilinear2d(hh, hw, true)?
                    } else {
                        r.clone()
                    };
                    return h.add(&r_aligned.affine(cn_scale as f64, 0.0)?).map_err(|e| e.into());
                }
            }
            Ok(h.clone())
        };

        // ---- Down path with CN injection ----
        let num_levels = self.cfg.blocks_per_level.len();
        let mut level_outputs: Vec<Tensor> = Vec::with_capacity(num_levels);
        let mut rb_idx: usize = 0;
        for (i, blocks) in self.down_blocks.iter().enumerate() {
            if i > 0 {
                h = self.down_downscalers[i - 1].forward(&h)?;
            }
            for block in blocks.iter() {
                if block.is_res() {
                    h = inject_cn(&h, rb_idx)?;
                    rb_idx += 1;
                }
                h = block.forward(&h, t_emb, sca_emb, crp_emb, clip)?;
            }
            level_outputs.push(h.clone());
            if let Some(d) = dump.as_deref_mut() {
                d.push((format!("down_lvl{i}"), h.clone()));
            }
        }

        // ---- Up path: start from the deepest level output ----
        // v0.40 phase 3 iter 1: skip is consumed by the FIRST
        // ResBlock at each up level (channel-concat in the
        // channelwise MLP), NOT by an additive add. The first
        // ResBlock was constructed with c_skip == up_c_hidden[level]
        // so it expects the skip via forward_maybe_skip.
        let mut h = level_outputs.pop().expect("non-empty levels");
        for (i, blocks) in self.up_blocks.iter().enumerate() {
            let skip_for_level = if i > 0 {
                h = self.up_upscalers[i - 1].forward(&h)?;
                Some(
                    level_outputs
                        .pop()
                        .ok_or_else(|| anyhow!("missing skip at up level {i}"))?,
                )
            } else {
                None
            };
            // v0.41 phase 2g: run the level's block group
            // (up_blocks_repeat_mappers[i]) times, applying the 1×1
            // repeat mapper between iterations. Stage C has 0 mappers
            // (runs once); Stage B has [3,3,2,2]. The skip is consumed
            // by the FIRST ResBlock on EVERY repeat iteration.
            let mappers = &self.up_repeat_mappers[i];
            let n_repeats = mappers.len() + 1;
            for rep in 0..n_repeats {
                // Upstream `_up_decode` bilinearly resizes x to the
                // skip's spatial when they mismatch
                // (`F.interpolate(..., bilinear, align_corners=True)`).
                // Stage B's strided down floors odd dims (6→3→1) while
                // the strided up doubles exactly (1→2), so x (2×2) and
                // the skip (3×3) disagree at the deep levels.
                if let Some(skip) = skip_for_level.as_ref() {
                    let (_, _, sh, sw) = skip.dims4()?;
                    let (_, _, hh, hw) = h.dims4()?;
                    if (hh, hw) != (sh, sw) {
                        h = h.upsample_bilinear2d(sh, sw, true)?;
                    }
                }
                for (b_idx, block) in blocks.iter().enumerate() {
                    if block.is_res() {
                        h = inject_cn(&h, rb_idx)?;
                        rb_idx += 1;
                    }
                    let block_skip = if b_idx == 0 { skip_for_level.as_ref() } else { None };
                    h = block.forward_maybe_skip(
                        &h, block_skip, t_emb, sca_emb, crp_emb, clip,
                    )?;
                }
                if rep < mappers.len() {
                    h = mappers[rep].forward(&h)?;
                }
            }
            if let Some(d) = dump.as_deref_mut() {
                d.push((format!("up_lvl{i}"), h.clone()));
            }
        }

        // ---- Output classifier ----
        let h = self.clf_norm.forward(&h)?;
        let out = self.clf_conv.forward(&h)?;
        if let Some(d) = dump.as_deref_mut() {
            d.push(("clf".to_string(), out.clone()));
        }
        Ok(out)
    }

}

// ---------------------------------------------------------------------
// Helpers — block-sequence builder.
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_block_levels(
    blocks_per_level: &[usize],
    c_hidden_per_level: &[usize],
    c_cond: usize,
    head_dim: usize,
    has_attention_per_level: &[bool],
    has_sca: bool,
    has_crp: bool,
    // v0.40 phase 3 iter 1: per-level skip channel count for the
    // FIRST ResBlock at each level. `None` → no skip-concat (used by
    // the down path). `Some(&skip_dims)` with `skip_dims[level] > 0`
    // → the first ResBlock at that level has `c_skip == skip_dims[level]`.
    skip_dims_per_level: Option<&[usize]>,
    vb: VarBuilder,
    registry: &Arc<RwLock<LoraRegistry>>,
) -> Result<Vec<Vec<Block>>> {
    let mut out = Vec::with_capacity(blocks_per_level.len());
    for (level, n_triples) in blocks_per_level.iter().enumerate() {
        let c = c_hidden_per_level[level];
        let has_attn = has_attention_per_level[level];
        let level_vb = vb.pp(&level.to_string());
        let triple_size = if has_attn { 3 } else { 2 };
        let mut blocks = Vec::with_capacity(n_triples * triple_size);
        let num_heads = (c / head_dim).max(1);
        // Attention level uses the same c_hidden for the kv stream
        // (text projection target).
        let text_dim = c;
        let skip_c = skip_dims_per_level.and_then(|s| s.get(level).copied()).unwrap_or(0);
        for triple in 0..*n_triples {
            let pos_base = triple * triple_size;
            // Skip-concat only on the very FIRST ResBlock of this level.
            let c_skip = if triple == 0 { skip_c } else { 0 };
            blocks.push(Block::Res(ResBlock::new_with_skip(
                c,
                c_skip,
                level_vb.pp(&pos_base.to_string()),
            )?));
            blocks.push(Block::Time(TimestepBlock::new(
                c,
                c_cond,
                has_sca,
                has_crp,
                level_vb.pp(&(pos_base + 1).to_string()),
            )?));
            if has_attn {
                blocks.push(Block::Attn(AttnBlock::new(
                    c,
                    text_dim,
                    num_heads,
                    true,
                    level_vb.pp(&(pos_base + 2).to_string()),
                    registry,
                )?));
            }
        }
        out.push(blocks);
    }
    Ok(out)
}

/// Sinusoidal positional encoding for a `(B,)` timestep tensor.
/// Returns `(B, c_cond)` — matches the upstream `gen_r_embedding`
/// using `c_cond=64`. Pure math, no learnable params.
/// Functional LayerNorm over the last dim with no affine params,
/// matching upstream `nn.LayerNorm(dim, elementwise_affine=False)`.
/// Used by `build_clip_conditioning` for the final `clip_norm`.
fn layer_norm_last_dim(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = var.affine(1.0, eps)?.sqrt()?;
    xc.broadcast_div(&denom).map_err(|e| e.into())
}

pub fn sinusoidal_time_embedding(
    t: &Tensor,
    c_cond: usize,
    max_positions: f64,
) -> Result<Tensor> {
    // v0.41 phase 2d: match diffusers' `gen_r_embedding` / Wuerstchen
    // `get_timestep_ratio_embedding` exactly. Three corrections vs
    // v0.39's first-draft form:
    //
    // 1. **Scale the input by max_positions**: upstream does
    //    `r = timestep_ratio * max_positions` so r lives in [0, 10000]
    //    not [0, 1]. The model learned to read time from the
    //    high-frequency wraps of sin(10000*t) etc. — at [0, 1] the
    //    embedding is just smooth low-magnitude noise that carries
    //    no per-step signal.
    // 2. **Divisor is `half_dim - 1`**, not `half_dim`. This makes
    //    freq[0] = max_positions^0 = 1 (lowest frequency) and
    //    freq[half-1] = max_positions^(-1) (highest frequency)
    //    exactly cover the trained range.
    // 3. **Sin THEN cos** — `cat([sin, cos], -1)`. The downstream
    //    Linear mapper learned weights for this column order; even
    //    a perfectly numerical embedding with the columns flipped
    //    is permuted nonsense.
    //
    // Caught at v0.41 phase 2b when the Metal end-to-end run
    // produced visual noise even after the BF16 fix made the
    // numerics finite — the model couldn't denoise because the
    // time signal was meaningless.
    let device = t.device();
    let half = c_cond / 2;
    debug_assert!(half >= 2, "sinusoidal_time_embedding needs c_cond >= 4");
    let denom = (half - 1) as f64;
    let freq_log = max_positions.ln() / denom;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-freq_log * (i as f64)).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, half, device)?;
    let r = t.to_dtype(DType::F32)?.affine(max_positions, 0.0)?;
    let args = r.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.sin()?, args.cos()?], D::Minus1).map_err(|e| e.into())
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// Tiny Stage C config for fast tests. 2 levels, 1 triple each
    /// = [Res, Time, Attn] × 1 × 2 = 6 sub-blocks per path.
    fn small_stage_c_cfg() -> Config {
        Config {
            c_in: 4,
            c_out: 4,
            c_hidden_per_level: vec![16, 16],
            c_cond: 8,
            c_clip_text: Some(24),
            c_clip_text_pooled: 24,
            c_clip_img: Some(12),
            num_pooled_tokens: 4,
            c_pooled_token: 16,
            head_dim: 4,
            has_attention_per_level: vec![true, true],
            has_sca: true,
            has_crp: true,
            blocks_per_level: vec![1, 1],
            effnet_input_channels: None,
            pixels_input_channels: None,
            sampler_style: SamplerStyle::OnePixel,
            switch_level: vec![false],
            up_blocks_repeat_mappers: vec![1, 1],
        }
    }

    /// Tiny Stage B config. 2 levels, attention only at the deeper
    /// one. Widths 8/16 to exercise the strided downscaler with
    /// channel change.
    fn small_stage_b_cfg() -> Config {
        Config {
            c_in: 4,
            c_out: 4,
            c_hidden_per_level: vec![8, 16],
            c_cond: 8,
            c_clip_text: None,
            c_clip_text_pooled: 24,
            c_clip_img: None,
            num_pooled_tokens: 4,
            c_pooled_token: 16,
            head_dim: 4,
            has_attention_per_level: vec![false, true],
            has_sca: true,
            has_crp: false,
            blocks_per_level: vec![1, 1],
            effnet_input_channels: Some(4),
            pixels_input_channels: Some(3),
            sampler_style: SamplerStyle::Strided,
            switch_level: vec![true],
            up_blocks_repeat_mappers: vec![1, 1],
        }
    }

    fn random_prior_c(cfg: Config) -> (StableCascadePrior, VarMap) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let prior = StableCascadePrior::new_stage_c(cfg, vb)
            .expect("new_stage_c");
        (prior, varmap)
    }

    fn random_prior_b(cfg: Config) -> (StableCascadePrior, VarMap) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let prior = StableCascadePrior::new_stage_b(cfg, vb)
            .expect("new_stage_b");
        (prior, varmap)
    }

    #[test]
    fn stage_c_full_config_matches_upstream_inspection() {
        let cfg = Config::stage_c_full();
        assert_eq!(cfg.c_in, 16);
        assert_eq!(cfg.c_out, 16);
        assert_eq!(cfg.c_hidden_per_level, vec![2048, 2048]);
        assert_eq!(cfg.c_cond, 64);
        assert_eq!(cfg.c_clip_text, Some(1280));
        assert_eq!(cfg.c_clip_img, Some(768));
        assert_eq!(
            cfg.num_pooled_tokens, 4,
            "Stage C upstream: 8192 = 4 × 2048 (corrected at v0.40 phase 2)"
        );
        assert_eq!(cfg.c_pooled_token, 2048);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.has_attention_per_level, vec![true, true]);
        assert!(cfg.has_sca && cfg.has_crp);
        assert_eq!(cfg.blocks_per_level, vec![8, 24]);
        assert!(cfg.effnet_input_channels.is_none());
        assert_eq!(cfg.sampler_style, SamplerStyle::OnePixel);
    }

    #[test]
    fn stage_b_full_config_matches_upstream_inspection() {
        let cfg = Config::stage_b_full();
        assert_eq!(cfg.c_in, 16);
        assert_eq!(cfg.c_out, 16);
        assert_eq!(cfg.c_hidden_per_level, vec![320, 640, 1280, 1280]);
        assert_eq!(cfg.c_cond, 64);
        assert!(cfg.c_clip_text.is_none(), "Stage B has no clip_txt_mapper");
        assert!(cfg.c_clip_img.is_none(), "Stage B has no clip_img_mapper");
        assert_eq!(
            cfg.num_pooled_tokens, 4,
            "Stage B upstream: 5120 = 4 × 1280 (corrected at v0.40 phase 2)"
        );
        assert_eq!(cfg.c_pooled_token, 1280);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(
            cfg.has_attention_per_level,
            vec![false, false, true, true],
            "Stage B attention only at the deepest 2 levels"
        );
        assert!(cfg.has_sca);
        assert!(!cfg.has_crp, "Stage B uses 2 mappers (no mapper_crp)");
        assert_eq!(cfg.blocks_per_level, vec![2, 6, 28, 6]);
        assert_eq!(cfg.effnet_input_channels, Some(16));
        assert_eq!(cfg.pixels_input_channels, Some(3));
        assert_eq!(cfg.sampler_style, SamplerStyle::Strided);
    }

    #[test]
    fn sinusoidal_time_embedding_shape() {
        let device = Device::Cpu;
        let t = Tensor::new(&[100f32, 500.0, 999.0], &device).unwrap();
        let emb = sinusoidal_time_embedding(&t, 64, 10000.0).unwrap();
        assert_eq!(emb.dims(), &[3, 64]);
    }

    #[test]
    fn sinusoidal_time_embedding_changes_with_time() {
        let device = Device::Cpu;
        let t1 = Tensor::new(&[10f32], &device).unwrap();
        let t2 = Tensor::new(&[500f32], &device).unwrap();
        let e1 = sinusoidal_time_embedding(&t1, 64, 10000.0).unwrap();
        let e2 = sinusoidal_time_embedding(&t2, 64, 10000.0).unwrap();
        let diff = (&e1 - &e2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-3, "embedding must depend on time (got {diff})");
    }

    #[test]
    fn sinusoidal_time_embedding_matches_diffusers_gen_r_embedding() {
        // v0.41 phase 2d regression guard: hand-compute the upstream
        // formula in Rust and assert our embedding agrees element-
        // wise. The form is:
        //     r = t * max_positions
        //     freq[i] = exp(-log(max_positions)/(half-1) * i)
        //     args[i] = r * freq[i]
        //     emb = cat([sin(args), cos(args)], -1)
        // If a future refactor drops the max_positions scaling, the
        // wrong divisor, or the wrong sin/cos column order, this
        // test bites — those exact regressions are what produced
        // pure noise output in v0.41 phase 2b.
        let device = Device::Cpu;
        let half = 32usize;
        let c_cond = 2 * half;
        let max_positions = 10000.0f64;
        let t_val = 0.5f64;

        let denom = (half - 1) as f64;
        let freq_log = max_positions.ln() / denom;
        let mut expected = Vec::with_capacity(c_cond);
        let r = t_val * max_positions;
        let args: Vec<f64> = (0..half)
            .map(|i| r * (-freq_log * i as f64).exp())
            .collect();
        for &a in &args {
            expected.push(a.sin() as f32);
        }
        for &a in &args {
            expected.push(a.cos() as f32);
        }

        let t = Tensor::new(&[t_val as f32], &device).unwrap();
        let emb = sinusoidal_time_embedding(&t, c_cond, max_positions).unwrap();
        assert_eq!(emb.dims(), &[1, c_cond]);
        let got: Vec<f32> = emb.squeeze(0).unwrap().to_vec1().unwrap();

        // Tolerance is loose (1e-3) because our internal math is F32
        // while the comparison expectation is F64 — at args ≈ 5000
        // the trig functions amplify F32 rounding to mid-1e-4.
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "col {i}: got {g}, expected {e}, |diff|={}",
                (g - e).abs()
            );
        }
    }

    #[test]
    fn updownblock_one_pixel_down_halves_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let blk = UpDownBlock::new_one_pixel(8, SampleMode::Down, true, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 16, 16), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn updownblock_one_pixel_up_doubles_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let blk = UpDownBlock::new_one_pixel(8, SampleMode::Up, true, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn updownblock_strided_down_changes_channels_and_halves_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // 8 → 16 channels with 16×16 → 8×8 spatial.
        let blk = UpDownBlock::new_strided_down(8, 16, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 16, 16), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 16, 8, 8]);
    }

    #[test]
    fn updownblock_strided_up_changes_channels_and_doubles_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // 16 → 8 channels with 4×4 → 8×8 spatial.
        let blk = UpDownBlock::new_strided_up(16, 8, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn stage_c_forward_preserves_input_shape() {
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let crp = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let clip = Tensor::randn(0f32, 1f32, (1, 5, 16), device).unwrap();
        let y = prior
            .forward(&x, &t, Some(&sca), Some(&crp), &clip, None, None)
            .unwrap();
        assert_eq!(y.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn stage_c_output_changes_when_timestep_changes() {
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let clip = Tensor::randn(0f32, 1f32, (1, 5, 16), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let crp = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let t1 = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let t2 = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let y1 = prior
            .forward(&x, &t1, Some(&sca), Some(&crp), &clip, None, None)
            .unwrap();
        let y2 = prior
            .forward(&x, &t2, Some(&sca), Some(&crp), &clip, None, None)
            .unwrap();
        let diff = (&y1 - &y2)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-5, "output should depend on timestep (got {diff})");
    }

    #[test]
    fn stage_c_output_differs_between_zero_cond_and_t_emb_placeholder() {
        // v0.41 phase 1: the v0.40 generate path piped `t_emb` itself
        // as both `sca_emb` and `crp_emb` (placeholder). The correct
        // upstream behaviour for the "no aesthetic / no crop override"
        // default is sinusoidal embedding of a ZERO scalar — same
        // encoder, different input. This test exists to assert the
        // distinction is observable in network output: otherwise the
        // phase 1 change would have been a no-op rename.
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let clip = Tensor::randn(0f32, 1f32, (1, 5, 16), device).unwrap();

        // Realistic timestep ratio in [0, 1] — embedded with
        // c_cond=8 to match small_stage_c_cfg.
        let t_scalar = Tensor::from_vec(vec![0.7f32], 1, device).unwrap();
        let t_emb = sinusoidal_time_embedding(&t_scalar, 8, 10000.0).unwrap();

        // Upstream's `sca=None / crp=None` default: embed a zero
        // scalar through the SAME sinusoidal encoder. The result is
        // NOT the zero tensor — sin(0)=0 but cos(0)=1, so half the
        // dims sit at 1.
        let zero_scalar = Tensor::zeros(1, candle_core::DType::F32, device).unwrap();
        let zero_cond_emb = sinusoidal_time_embedding(&zero_scalar, 8, 10000.0).unwrap();

        let y_placeholder = prior
            .forward(&x, &t_emb, Some(&t_emb), Some(&t_emb), &clip, None, None)
            .unwrap();
        let y_real = prior
            .forward(
                &x,
                &t_emb,
                Some(&zero_cond_emb),
                Some(&zero_cond_emb),
                &clip,
                None,
                None,
            )
            .unwrap();

        let diff = (&y_placeholder - &y_real)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-5,
            "zero-cond sca/crp should change Stage C output vs t_emb placeholder (got {diff})"
        );
    }

    #[test]
    fn zero_cond_sinusoidal_embedding_is_not_zero_tensor() {
        // Sanity check on phase 1 reasoning: sinusoidal embedding of
        // a zero scalar produces `[cos(0), cos(0), …, sin(0), sin(0), …]`
        // → `[1, 1, …, 0, 0, …]`, NOT the zero tensor. If candle's
        // sinusoidal encoder ever changed convention so that zero
        // input gave zero output, this test would catch the
        // regression and force us to revisit phase 1's claim.
        let device = candle_core::Device::Cpu;
        let zero_scalar = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
        let emb = sinusoidal_time_embedding(&zero_scalar, 64, 10000.0).unwrap();
        let max_abs = emb
            .abs()
            .unwrap()
            .max_keepdim(D::Minus1)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (max_abs - 1.0).abs() < 1e-4,
            "expected cos(0)=1 to dominate; got max_abs={max_abs}"
        );
    }

    #[test]
    fn build_clip_conditioning_for_stage_c_returns_concat_85_at_c_pooled_token() {
        // v0.40 phase 2: Stage C returns concat(text, pooled_text,
        // pooled_img) at (B, T_text + 4 + 4, c_pooled_token).
        // small_stage_c_cfg uses 5 text tokens at c_pooled_token=16,
        // num_pooled_tokens=4 → expected shape (1, 5+4+4=13, 16).
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let text = Tensor::randn(0f32, 1f32, (1, 5, 24), device).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, 24), device).unwrap();
        let img = Tensor::randn(0f32, 1f32, (1, 12), device).unwrap();
        let clip = prior
            .build_clip_conditioning(&text, &pooled, Some(&img))
            .unwrap();
        assert_eq!(clip.dims(), &[1, 13, 16]);
    }

    #[test]
    fn build_clip_conditioning_for_stage_c_zero_pads_missing_image() {
        // When clip_img is None, the image pooled stream is zeros.
        // Output shape should still be (B, T_text + 4 + 4, c_pooled_token).
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let text = Tensor::randn(0f32, 1f32, (1, 5, 24), device).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, 24), device).unwrap();
        let clip = prior
            .build_clip_conditioning(&text, &pooled, None)
            .unwrap();
        assert_eq!(clip.dims(), &[1, 13, 16]);
    }

    #[test]
    fn build_clip_conditioning_for_stage_b_returns_pooled_only() {
        // v0.40 phase 2: Stage B has no clip_txt_mapper, no
        // clip_img_mapper. Returns just pooled-text: (B, 4, c_pooled_token).
        let (prior, _) = random_prior_b(small_stage_b_cfg());
        let device = &prior.device;
        // clip_text arg is ignored for Stage B; pass a dummy.
        let dummy_text = Tensor::randn(0f32, 1f32, (1, 1, 1), device).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, 24), device).unwrap();
        let clip = prior
            .build_clip_conditioning(&dummy_text, &pooled, None)
            .unwrap();
        // small_stage_b_cfg: num_pooled_tokens=4, c_pooled_token=16
        assert_eq!(clip.dims(), &[1, 4, 16]);
    }

    // ---- Stage B tests (phase 0c) ----

    #[test]
    fn stage_b_forward_preserves_input_shape_no_conditioning() {
        let (prior, _) = random_prior_b(small_stage_b_cfg());
        let device = &prior.device;
        // (1, c_in=4, 8, 8) input → strided down 2× → 8×8 → 4×4
        // → strided up 2× → back to 8×8.
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        // Stage B has no AttnBlock at level 0 (no kv consumed there),
        // and at the deepest level the kv dim must equal c_hidden_per_level[1]=16.
        let kv = prior.zero_kv_stream(1, 1, 5).unwrap();
        // Stage B: pass None for crp_emb (has_crp=false). Test without
        // effnet/pixels conditioning.
        let y = prior
            .forward(&x, &t, Some(&sca), None, &kv, None, None)
            .unwrap();
        assert_eq!(y.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn stage_b_forward_with_effnet_and_pixels_conditioning() {
        let (prior, _) = random_prior_b(small_stage_b_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let kv = prior.zero_kv_stream(1, 1, 5).unwrap();
        // Caller must pre-resize effnet + pixels to match embedding output (8×8).
        let effnet = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let pixels = Tensor::randn(0f32, 1f32, (1, 3, 8, 8), device).unwrap();
        let y = prior
            .forward(&x, &t, Some(&sca), None, &kv, Some(&effnet), Some(&pixels))
            .unwrap();
        assert_eq!(y.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn stage_b_effnet_changes_output() {
        let (prior, _) = random_prior_b(small_stage_b_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let kv = prior.zero_kv_stream(1, 1, 5).unwrap();
        let eff1 = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let eff2 = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let y1 = prior
            .forward(&x, &t, Some(&sca), None, &kv, Some(&eff1), None)
            .unwrap();
        let y2 = prior
            .forward(&x, &t, Some(&sca), None, &kv, Some(&eff2), None)
            .unwrap();
        let diff = (&y1 - &y2)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-5, "Stage B output should depend on effnet ({diff})");
    }

    #[test]
    fn stage_b_apply_effnet_mapper_shape() {
        let (prior, _) = random_prior_b(small_stage_b_cfg());
        let device = &prior.device;
        let effnet = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let projected = prior.apply_effnet_mapper(&effnet).unwrap();
        // Output channels = c_hidden_first = 8.
        assert_eq!(projected.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn stage_b_apply_pixels_mapper_shape() {
        let (prior, _) = random_prior_b(small_stage_b_cfg());
        let device = &prior.device;
        let pixels = Tensor::randn(0f32, 1f32, (1, 3, 8, 8), device).unwrap();
        let projected = prior.apply_pixels_mapper(&pixels).unwrap();
        assert_eq!(projected.dims(), &[1, 8, 8, 8]);
    }

    // ---- v0.39 phase 0h: parameter count + real-weight smoke ----

    #[test]
    fn stage_c_full_topology_produces_1550_param_tensors() {
        // Locks the upstream inspection count (1550 tensors in
        // stable-cascade-prior/prior/diffusion_pytorch_model.safetensors)
        // against our Stage C topology. Tensor count is determined by
        // structure (blocks_per_level, has_attention, has_sca, has_crp,
        // mappers) — not widths — so we use tiny widths for speed.
        //
        // If this test breaks: either our topology drifted from
        // upstream (bug) or we deliberately added/removed a module
        // (update the constant). Either way, the test fails loudly
        // so the divergence is conscious.
        let cfg = Config {
            c_in: 4,
            c_out: 4,
            c_hidden_per_level: vec![8, 8],
            c_cond: 8,
            c_clip_text: Some(8),
            c_clip_text_pooled: 8,
            c_clip_img: Some(8),
            num_pooled_tokens: 4,
            c_pooled_token: 8,
            head_dim: 2,
            has_attention_per_level: vec![true, true],
            has_sca: true,
            has_crp: true,
            blocks_per_level: vec![8, 24], // Upstream Stage C counts.
            effnet_input_channels: None,
            pixels_input_channels: None,
            sampler_style: SamplerStyle::OnePixel,
            switch_level: vec![false],
            up_blocks_repeat_mappers: vec![1, 1],
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _prior = StableCascadePrior::new_stage_c(cfg, vb).expect("new_stage_c");
        let n_params = varmap.data().lock().unwrap().len();
        assert_eq!(
            n_params, 1550,
            "Stage C full topology should produce 1550 param tensors \
             (matches upstream inspection at v0.39 phase 0)"
        );
    }

    /// Real-weight smoke test for Stage C. Skipped unless
    /// `STABLE_CASCADE_WEIGHTS_DIR` env var points at a directory
    /// containing `prior/diffusion_pytorch_model.safetensors`.
    ///
    /// When run: attempts VarBuilder.from_mmaped_safetensors against
    /// the real upstream Stage C checkpoint at FULL widths (c_hidden=
    /// 2048; ~3.6 GB RAM). Success means every tensor key in
    /// `cascade_prior::StableCascadePrior::new_stage_c` matches
    /// upstream — the v0.37/v0.38 caveat is closed.
    #[test]
    fn stage_c_loads_from_real_upstream_weights() {
        let dir = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let path = std::path::PathBuf::from(&dir)
            .join("prior/diffusion_pytorch_model.safetensors");
        if !path.exists() {
            eprintln!(
                "Skipping stage_c_loads_from_real_upstream_weights: \
                 {} doesn't exist (set STABLE_CASCADE_WEIGHTS_DIR to a \
                 directory containing prior/diffusion_pytorch_model.safetensors \
                 from stabilityai/stable-cascade-prior).",
                path.display()
            );
            return;
        }
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.as_path()],
                DType::F32,
                &device,
            )
            .expect("mmap stage_c weights")
        };
        match StableCascadePrior::new_stage_c(Config::stage_c_full(), vb) {
            Ok(_) => eprintln!("✓ Stage C real-weight load OK ({})", path.display()),
            Err(e) => panic!(
                "Stage C real-weight load FAILED — indicates tensor naming \
                 mismatch between v0.39 cascade_prior and upstream:\n  {e}"
            ),
        }
    }

    /// v0.41 phase 2f: reference-comparison harness. Loads the
    /// diffusers Stage C dump (`/tmp/cascade_ref.safetensors` produced
    /// by `tools/cascade_ref_dump.py`), feeds the IDENTICAL fixed
    /// inputs through our `forward_collect`, and prints the per-dump-
    /// point max-abs-diff so the first divergence localizes the bug.
    ///
    /// Skipped unless both `STABLE_CASCADE_WEIGHTS_DIR` is set AND
    /// `/tmp/cascade_ref.safetensors` exists.
    #[test]
    fn stage_c_matches_diffusers_reference() {
        let dir = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let ref_path = std::path::PathBuf::from("/tmp/cascade_ref.safetensors");
        if !ref_path.exists() {
            eprintln!("Skipping: /tmp/cascade_ref.safetensors not found (run tools/cascade_ref_dump.py)");
            return;
        }
        let weights = std::path::PathBuf::from(&dir)
            .join("prior/diffusion_pytorch_model.safetensors");
        if !weights.exists() {
            eprintln!("Skipping: {} not found", weights.display());
            return;
        }
        let device = Device::Cpu;
        let refs = candle_core::safetensors::load(&ref_path, &device)
            .expect("load reference dump");
        let get = |k: &str| refs.get(k).unwrap_or_else(|| panic!("ref missing {k}"));

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights.as_path()], DType::F32, &device)
                .expect("mmap stage_c weights")
        };
        let prior = StableCascadePrior::new_stage_c(Config::stage_c_full(), vb)
            .expect("new_stage_c");

        let latents = get("in_latents").to_dtype(DType::F32).unwrap();
        let clip_text = get("in_clip_text").to_dtype(DType::F32).unwrap();
        let clip_pooled = get("in_clip_text_pooled")
            .to_dtype(DType::F32).unwrap()
            .squeeze(1).unwrap(); // (B,1,1280) -> (B,1280)
        let clip_img = get("in_clip_img")
            .to_dtype(DType::F32).unwrap()
            .squeeze(1).unwrap(); // (B,1,768) -> (B,768)

        let max_abs_diff = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
        };

        // ---- 1. Conditioning comparison ----
        let our_clip = prior
            .build_clip_conditioning(&clip_text, &clip_pooled, Some(&clip_img))
            .expect("build_clip_conditioning");
        let ref_clip = get("clip_cond");
        eprintln!(
            "[ref] clip_cond shape ours={:?} ref={:?}  max_abs_diff={:.5}",
            our_clip.dims(), ref_clip.dims(), max_abs_diff(&our_clip, ref_clip)
        );

        // ---- 2. Time embedding (t=0.5, c_cond=64) + zero-cond sca/crp ----
        let t = Tensor::new(&[0.5f32], &device).unwrap();
        let t_emb = sinusoidal_time_embedding(&t, 64, 10000.0).unwrap();
        let zero = Tensor::zeros(1, DType::F32, &device).unwrap();
        let zero_emb = sinusoidal_time_embedding(&zero, 64, 10000.0).unwrap();

        // ---- 3. Forward with REFERENCE conditioning (isolates the body) ----
        let (out, dump) = prior
            .forward_collect(&latents, &t_emb, Some(&zero_emb), Some(&zero_emb), ref_clip, None, None)
            .expect("forward_collect");

        for (name, tens) in &dump {
            if let Some(r) = refs.get(name) {
                if tens.dims() != r.dims() {
                    eprintln!(
                        "[ref] {name:10} SHAPE MISMATCH ours={:?} ref={:?}",
                        tens.dims(), r.dims()
                    );
                    continue;
                }
                eprintln!(
                    "[ref] {name:10} shape={:?}  max_abs_diff={:.5}  (ref range [{:.2},{:.2}])",
                    tens.dims(),
                    max_abs_diff(tens, r),
                    r.min_all().unwrap().to_scalar::<f32>().unwrap(),
                    r.max_all().unwrap().to_scalar::<f32>().unwrap(),
                );
            }
        }
        let final_diff = max_abs_diff(&out, get("out_final"));
        eprintln!("[ref] out_final max_abs_diff={final_diff:.5}");
    }

    /// v0.41 phase 2g: Stage B (decoder) reference comparison. Loads
    /// `/tmp/cascade_ref_b.safetensors` (from tools/cascade_ref_dump_b.py)
    /// and diffs our Stage B forward against diffusers. The decoder
    /// `sample` is 4-channel and patchified internally; our bridge
    /// pre-unshuffles, so we feed `pixel_unshuffle(2, in_latents)`.
    #[test]
    fn stage_b_matches_diffusers_reference() {
        let dir = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let ref_path = std::path::PathBuf::from(
            std::env::var("CASCADE_REF_B").unwrap_or_else(|_| "/tmp/cascade_ref_b.safetensors".into())
        );
        if !ref_path.exists() {
            eprintln!("Skipping: /tmp/cascade_ref_b.safetensors not found (run tools/cascade_ref_dump_b.py)");
            return;
        }
        let weights = std::path::PathBuf::from(&dir)
            .join("decoder/diffusion_pytorch_model.safetensors");
        if !weights.exists() {
            eprintln!("Skipping: {} not found", weights.display());
            return;
        }
        let device = Device::Cpu;
        let refs = candle_core::safetensors::load(&ref_path, &device)
            .expect("load reference dump");
        let get = |k: &str| refs.get(k).unwrap_or_else(|| panic!("ref missing {k}"));

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights.as_path()], DType::F32, &device)
                .expect("mmap stage_b weights")
        };
        let prior = StableCascadePrior::new_stage_b(Config::stage_b_full(), vb)
            .expect("new_stage_b");

        // Decoder sample (4ch) -> our 16ch input via pixel_unshuffle(2).
        let latents4 = get("in_latents").to_dtype(DType::F32).unwrap();
        let latents = crate::pipelines::cascade_vae::pixel_unshuffle(&latents4, 2).unwrap();
        let effnet = refs.get("in_effnet").map(|t| t.to_dtype(DType::F32).unwrap());
        let clip_pooled = get("in_clip_text_pooled")
            .to_dtype(DType::F32).unwrap()
            .squeeze(1).unwrap();

        let max_abs_diff = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
        };

        // Conditioning (pooled-only path).
        let dummy_text = Tensor::zeros((1, 1, 1), DType::F32, &device).unwrap();
        let our_clip = prior
            .build_clip_conditioning(&dummy_text, &clip_pooled, None)
            .expect("build_clip_conditioning");
        eprintln!(
            "[refB] clip_cond ours={:?} ref={:?}  max_abs_diff={:.5}",
            our_clip.dims(), get("clip_cond").dims(),
            max_abs_diff(&our_clip, get("clip_cond"))
        );

        let t = Tensor::new(&[0.5f32], &device).unwrap();
        let t_emb = sinusoidal_time_embedding(&t, 64, 10000.0).unwrap();
        let zero = Tensor::zeros(1, DType::F32, &device).unwrap();
        let zero_emb = sinusoidal_time_embedding(&zero, 64, 10000.0).unwrap();

        // Feed reference conditioning to isolate the body.
        let (_out, dump) = prior
            .forward_collect(&latents, &t_emb, Some(&zero_emb), None, get("clip_cond"), effnet.as_ref(), None)
            .expect("forward_collect");

        for (name, tens) in &dump {
            if let Some(r) = refs.get(name) {
                if tens.dims() != r.dims() {
                    eprintln!("[refB] {name:10} SHAPE MISMATCH ours={:?} ref={:?}", tens.dims(), r.dims());
                    continue;
                }
                eprintln!(
                    "[refB] {name:10} shape={:?}  max_abs_diff={:.5}  (ref range [{:.2},{:.2}])",
                    tens.dims(), max_abs_diff(tens, r),
                    r.min_all().unwrap().to_scalar::<f32>().unwrap(),
                    r.max_all().unwrap().to_scalar::<f32>().unwrap(),
                );
            }
        }
    }

    /// v0.40 phase 3: real-weight smoke for Stage B. Stage B lives in
    /// the standard `stabilityai/stable-cascade` repo under
    /// `decoder/`. Largest stage at full widths (~3 GB), so this test
    /// is heaviest of the four.
    #[test]
    fn stage_b_loads_from_real_upstream_weights() {
        let dir = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let path = std::path::PathBuf::from(&dir)
            .join("decoder/diffusion_pytorch_model.safetensors");
        if !path.exists() {
            eprintln!(
                "Skipping stage_b_loads_from_real_upstream_weights: \
                 {} doesn't exist (set STABLE_CASCADE_WEIGHTS_DIR to a \
                 directory containing decoder/diffusion_pytorch_model.safetensors \
                 from stabilityai/stable-cascade).",
                path.display()
            );
            return;
        }
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.as_path()],
                DType::F32,
                &device,
            )
            .expect("mmap stage_b weights")
        };
        match StableCascadePrior::new_stage_b(Config::stage_b_full(), vb) {
            Ok(_) => eprintln!("✓ Stage B real-weight load OK ({})", path.display()),
            Err(e) => panic!(
                "Stage B real-weight load FAILED — indicates tensor naming \
                 mismatch between v0.40 cascade_prior and upstream:\n  {e}"
            ),
        }
    }

    // ---- v0.41 phase 3: ControlNet injection ----

    #[test]
    fn forward_with_cn_matches_forward_when_residuals_zeroed() {
        // Sanity: CN injection with all-zero residuals + cn_scale=1.0
        // should produce byte-identical output to the plain forward
        // path. Verifies the injection mechanism is additive and
        // doesn't otherwise modify the forward.
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let crp = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let clip = Tensor::randn(0f32, 1f32, (1, 5, 16), device).unwrap();
        // small_stage_c_cfg has 2 levels × 1 triple = 1 ResBlock per
        // down level (rb 0, 1) + 1 per up level (rb 2, 3). Inject at
        // one down ResBlock (0) and one up ResBlock (2). Spatial is
        // 8×8 throughout (switch_level=false), so the residuals match.
        let cn_blocks = [0usize, 2];
        let r0 = Tensor::zeros((1, 16, 8, 8), DType::F32, device).unwrap();
        let r1 = Tensor::zeros((1, 16, 8, 8), DType::F32, device).unwrap();
        let plain = prior
            .forward(&x, &t, Some(&sca), Some(&crp), &clip, None, None)
            .unwrap();
        let with_zero_cn = prior
            .forward_with_cn(&x, &t, Some(&sca), Some(&crp), &clip, &[r0, r1], &cn_blocks, 1.0)
            .unwrap();
        let diff = (&plain - &with_zero_cn)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "CN injection with zero residuals should match plain forward (got max diff {diff})"
        );
    }

    #[test]
    fn forward_with_cn_changes_output_when_residuals_nonzero() {
        // Load-bearing: residuals must actually influence the down
        // path. If they were silently dropped, output would match
        // plain forward.
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let crp = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let clip = Tensor::randn(0f32, 1f32, (1, 5, 16), device).unwrap();
        let r0 = Tensor::randn(0f32, 1f32, (1, 16, 8, 8), device).unwrap();
        let r1 = Tensor::randn(0f32, 1f32, (1, 16, 8, 8), device).unwrap();
        let cn_blocks = [0usize, 2];
        let plain = prior
            .forward(&x, &t, Some(&sca), Some(&crp), &clip, None, None)
            .unwrap();
        let with_cn = prior
            .forward_with_cn(&x, &t, Some(&sca), Some(&crp), &clip, &[r0, r1], &cn_blocks, 1.0)
            .unwrap();
        let diff = (&plain - &with_cn)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-5,
            "CN injection should change output with non-zero residuals (got mean abs diff {diff})"
        );
    }

    #[test]
    fn forward_with_cn_bilinear_resizes_mismatched_residuals() {
        // v0.41 phase 3: residuals at the CN backbone's small spatial
        // are bilinearly upsampled to the latent spatial (no more
        // exact-shape requirement). A 4×4 residual injects fine into
        // an 8×8 latent.
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let crp = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let clip = Tensor::randn(0f32, 1f32, (1, 5, 16), device).unwrap();
        let r0 = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), device).unwrap();
        let r1 = Tensor::randn(0f32, 1f32, (1, 16, 4, 4), device).unwrap();
        let out = prior
            .forward_with_cn(&x, &t, Some(&sca), Some(&crp), &clip, &[r0, r1], &[0, 2], 1.0)
            .unwrap();
        assert_eq!(out.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn forward_with_cn_rejects_empty_residuals() {
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let crp = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let clip = Tensor::randn(0f32, 1f32, (1, 5, 16), device).unwrap();
        let err = prior
            .forward_with_cn(&x, &t, Some(&sca), Some(&crp), &clip, &[], &[], 1.0)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("non-empty"), "got: {msg}");
    }

    #[test]
    fn forward_with_cn_rejects_stage_b_config() {
        let (prior, _) = random_prior_b(small_stage_b_cfg());
        let device = &prior.device;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), device).unwrap();
        let t = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let sca = Tensor::randn(0f32, 1f32, (1, 8), device).unwrap();
        let clip = prior.zero_kv_stream(1, 1, 5).unwrap();
        let r = Tensor::zeros((1, 16, 4, 4), DType::F32, device).unwrap();
        let err = prior
            .forward_with_cn(&x, &t, Some(&sca), None, &clip, &[r], &[0], 1.0)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Stage C only"),
            "expected Stage-C-only error; got: {msg}"
        );
    }

    #[test]
    fn stage_b_construction_rejects_one_pixel_style() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut cfg = small_stage_b_cfg();
        cfg.sampler_style = SamplerStyle::OnePixel;
        match StableCascadePrior::new_stage_b(cfg, vb) {
            Ok(_) => panic!("Stage B must reject OnePixel sampler"),
            Err(e) => assert!(format!("{e}").contains("Strided")),
        }
    }
}
