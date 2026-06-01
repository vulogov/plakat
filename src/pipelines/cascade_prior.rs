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
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::pipelines::cascade_blocks::{
    AttnBlock, LayerNorm2d, ResBlock, TimestepBlock,
};

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
    /// image mappers project to `c_cond * num_pooled_tokens`. Stage
    /// C upstream uses 128; Stage B upstream uses 80.
    pub num_pooled_tokens: usize,
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
            num_pooled_tokens: 128,
            head_dim: 64,
            has_attention_per_level: vec![true, true],
            has_sca: true,
            has_crp: true,
            blocks_per_level: vec![8, 24],
            effnet_input_channels: None,
            pixels_input_channels: None,
            sampler_style: SamplerStyle::OnePixel,
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
            num_pooled_tokens: 80,
            head_dim: 64,
            has_attention_per_level: vec![false, false, true, true],
            has_sca: true,
            has_crp: false,
            blocks_per_level: vec![2, 6, 28, 6],
            effnet_input_channels: Some(16),
            pixels_input_channels: Some(3),
            sampler_style: SamplerStyle::Strided,
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
        Ok(Self::OnePixel { norm, conv, mode })
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
            UpDownBlock::OnePixel { norm, conv, mode } => {
                let x = norm.forward(x)?;
                match mode {
                    SampleMode::Down => Ok(conv.forward(&x)?.avg_pool2d(2)?),
                    SampleMode::Up => {
                        let (_b, _c, h, w) = x.dims4()?;
                        let up = x.upsample_nearest2d(h * 2, w * 2)?;
                        Ok(conv.forward(&up)?)
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
    clf_norm: LayerNorm2d,
    clf_conv: nn::Conv2d,
    pub cfg: Config,
    pub dtype: DType,
    pub device: Device,
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
        let pooled_out_dim = cfg.c_cond * cfg.num_pooled_tokens;
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

        // ---- Down blocks: per-level width + attn flag ----
        let down_blocks = build_block_levels(
            &cfg.blocks_per_level,
            &cfg.c_hidden_per_level,
            cfg.c_cond,
            cfg.head_dim,
            &cfg.has_attention_per_level,
            cfg.has_sca,
            cfg.has_crp,
            vb.pp("down_blocks"),
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
                    UpDownBlock::new_one_pixel(in_c, SampleMode::Down, sub_vb)?
                }
                SamplerStyle::Strided => UpDownBlock::new_strided_down(in_c, out_c, sub_vb)?,
            };
            down_downscalers.push(blk);
        }

        // ---- Up blocks (mirror of down: deepest first, shallowest last) ----
        let up_blocks_per_level: Vec<usize> =
            cfg.blocks_per_level.iter().rev().copied().collect();
        let up_c_hidden: Vec<usize> =
            cfg.c_hidden_per_level.iter().rev().copied().collect();
        let up_has_attn: Vec<bool> =
            cfg.has_attention_per_level.iter().rev().copied().collect();
        let up_blocks = build_block_levels(
            &up_blocks_per_level,
            &up_c_hidden,
            cfg.c_cond,
            cfg.head_dim,
            &up_has_attn,
            cfg.has_sca,
            cfg.has_crp,
            vb.pp("up_blocks"),
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
                    UpDownBlock::new_one_pixel(in_c, SampleMode::Up, sub_vb)?
                }
                SamplerStyle::Strided => UpDownBlock::new_strided_up(in_c, out_c, sub_vb)?,
            };
            up_upscalers.push(blk);
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
            clf_norm,
            clf_conv,
            cfg,
            dtype,
            device,
        })
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
    pub fn build_clip_conditioning(
        &self,
        clip_text: &Tensor,
        clip_text_pooled: &Tensor,
        clip_img: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, _t, _) = clip_text.dims3()?;
        // Stage C projects text seq → c_hidden_first via clip_txt_mapper.
        let text = if let Some(mapper) = &self.clip_txt_mapper {
            mapper.forward(clip_text)?
        } else {
            return Err(anyhow!(
                "build_clip_conditioning called on a Prior without \
                 clip_txt_mapper (Stage B). Use zero_kv_stream + supply \
                 conditioning at the AttnBlock kv_mapper input shape."
            ));
        };
        // Pooled streams computed but held for phase 0g.
        let _pooled_text = self
            .clip_txt_pooled_mapper
            .forward(clip_text_pooled)?
            .reshape((b, self.cfg.num_pooled_tokens, self.cfg.c_cond))?;
        if let (Some(img), Some(mapper)) = (clip_img, &self.clip_img_mapper) {
            let _pooled_img = mapper
                .forward(img)?
                .reshape((b, self.cfg.num_pooled_tokens, self.cfg.c_cond))?;
        }
        Ok(text)
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
        let h = h.gelu()?;
        Ok(mapper.1.forward(&h)?)
    }

    /// Apply pixels conditioning (Stage B only). See
    /// [`apply_effnet_mapper`] for shape semantics.
    pub fn apply_pixels_mapper(&self, pixels: &Tensor) -> Result<Tensor> {
        let mapper = self
            .pixels_mapper
            .as_ref()
            .ok_or_else(|| anyhow!("apply_pixels_mapper called on a Stage C prior"))?;
        let h = mapper.0.forward(pixels)?;
        let h = h.gelu()?;
        Ok(mapper.1.forward(&h)?)
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
        // ---- Input embedding ----
        let mut h = self.embedding_conv.forward(x)?;
        h = self.embedding_norm.forward(&h)?;
        if let Some(eff) = effnet {
            h = h.add(&self.apply_effnet_mapper(eff)?)?;
        }
        if let Some(px) = pixels {
            h = h.add(&self.apply_pixels_mapper(px)?)?;
        }

        // ---- Down path ----
        let num_levels = self.cfg.blocks_per_level.len();
        let mut level_outputs: Vec<Tensor> = Vec::with_capacity(num_levels);
        for (i, blocks) in self.down_blocks.iter().enumerate() {
            if i > 0 {
                h = self.down_downscalers[i - 1].forward(&h)?;
            }
            for block in blocks {
                h = block.forward(&h, t_emb, sca_emb, crp_emb, clip)?;
            }
            level_outputs.push(h.clone());
        }

        // ---- Up path: start from the deepest level output ----
        let mut h = level_outputs.pop().expect("non-empty levels");
        for (i, blocks) in self.up_blocks.iter().enumerate() {
            if i > 0 {
                h = self.up_upscalers[i - 1].forward(&h)?;
                let skip = level_outputs
                    .pop()
                    .ok_or_else(|| anyhow!("missing skip at up level {i}"))?;
                h = h.add(&skip)?;
            }
            for block in blocks {
                h = block.forward(&h, t_emb, sca_emb, crp_emb, clip)?;
            }
        }

        // ---- Output classifier ----
        let h = self.clf_norm.forward(&h)?;
        Ok(self.clf_conv.forward(&h)?)
    }
}

// ---------------------------------------------------------------------
// Helpers — block-sequence builder.
// ---------------------------------------------------------------------

fn build_block_levels(
    blocks_per_level: &[usize],
    c_hidden_per_level: &[usize],
    c_cond: usize,
    head_dim: usize,
    has_attention_per_level: &[bool],
    has_sca: bool,
    has_crp: bool,
    vb: VarBuilder,
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
        for triple in 0..*n_triples {
            let pos_base = triple * triple_size;
            blocks.push(Block::Res(ResBlock::new(
                c,
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
pub fn sinusoidal_time_embedding(
    t: &Tensor,
    c_cond: usize,
    max_positions: f64,
) -> Result<Tensor> {
    let device = t.device();
    let half = c_cond / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(max_positions.ln()) * (i as f64) / (half as f64)).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, half, device)?;
    let t_f32 = t.to_dtype(DType::F32)?;
    let args = t_f32.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1).map_err(|e| e.into())
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
            head_dim: 4,
            has_attention_per_level: vec![true, true],
            has_sca: true,
            has_crp: true,
            blocks_per_level: vec![1, 1],
            effnet_input_channels: None,
            pixels_input_channels: None,
            sampler_style: SamplerStyle::OnePixel,
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
            head_dim: 4,
            has_attention_per_level: vec![false, true],
            has_sca: true,
            has_crp: false,
            blocks_per_level: vec![1, 1],
            effnet_input_channels: Some(4),
            pixels_input_channels: Some(3),
            sampler_style: SamplerStyle::Strided,
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
        assert_eq!(cfg.num_pooled_tokens, 128);
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
        assert_eq!(cfg.num_pooled_tokens, 80);
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
    fn updownblock_one_pixel_down_halves_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let blk = UpDownBlock::new_one_pixel(8, SampleMode::Down, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 16, 16), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8, 8, 8]);
    }

    #[test]
    fn updownblock_one_pixel_up_doubles_spatial() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let blk = UpDownBlock::new_one_pixel(8, SampleMode::Up, vb).unwrap();
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
    fn build_clip_conditioning_for_stage_c_returns_projected_text() {
        let (prior, _) = random_prior_c(small_stage_c_cfg());
        let device = &prior.device;
        let text = Tensor::randn(0f32, 1f32, (1, 5, 24), device).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, 24), device).unwrap();
        let img = Tensor::randn(0f32, 1f32, (1, 12), device).unwrap();
        let clip = prior
            .build_clip_conditioning(&text, &pooled, Some(&img))
            .unwrap();
        assert_eq!(clip.dims(), &[1, 5, 16]);
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
