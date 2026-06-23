//! v0.39 phase 0e: Stable Cascade ControlNet (upstream-aligned).
//!
//! Replaces (will replace, in phase 0g) v0.38 phase 5's compact
//! `cascade_controlnet.rs` with the actual upstream Cascade CN
//! architecture inspected at v0.39 phase 0:
//!
//! ```text
//!   conditioning (B, 1, 1024, 1024)
//!     ↓ MobileNetV3-Large backbone (8 stages, stem + 6 inverted-
//!     ↓   residual stages + 1×1 final → 1280 channels)
//!     ↓ feature map (B, 1280, h, w)
//!     ↓ × 8 projection heads (Conv 1280→1280, act, Conv 1280→2048)
//!     ↓
//!   residuals[0..7] (each: (B, 2048, h, w))
//! ```
//!
//! Each of the 8 residuals is injected at a specific position in
//! Stage C's `down_blocks` / `up_blocks` triples. The exact
//! injection mapping is phase 0g — for v0.39 we ship the
//! architecture + projection topology so weights load.
//!
//! ## Tensor naming (matches upstream
//! `stabilityai/stable-cascade/controlnet/canny.safetensors`)
//!
//! - `backbone.0.{0,1}` — stem (Conv + BN)
//! - `backbone.{1..6}.{block_idx}.block.{...}` — inverted residual blocks
//!   - block.0: expand 1×1 (point-wise) — OR the only conv for stage 1
//!     (3×3, no separate depthwise)
//!   - block.1: depthwise 3×3 (groups=c)
//!   - block.2.{fc1,fc2}: Squeeze-Excitation (stages 4-6 only)
//!   - block.3: project 1×1 (point-wise)
//! - `backbone.7.{0,1}` — final projection (Conv 1×1 + BN, 256 → 1280)
//! - `projections.{0..7}.{0,2}` — 8 projection heads, each
//!   Sequential(Conv 1×1, GELU, Conv 1×1) producing 1280 → 2048
//!
//! ## Phase 0e scope
//!
//! - Shape-correct backbone with tensor names matching upstream.
//! - 8 projection heads (Sequential(Conv, GELU, Conv)) producing
//!   residuals at 2048 channels.
//! - Forward returns `Vec<Tensor>` — 8 residuals.
//!
//! Activation choices (Hardswish vs SiLU vs ReLU) follow upstream
//! MobileNetV3 conventions approximately; phase 0g refines if
//! real-weight smoke shows numerical divergence. ReflectionPad
//! approximated with zero-pad (same caveat as the Stage A VAE
//! depthwise).

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{self as nn, VarBuilder};

// ---------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Config {
    /// Input conditioning channels — 1 for canny (grayscale edges).
    pub c_in: usize,
    pub stem: StemConfig,
    /// Stages 1..=6: each is a `Vec<BlockConfig>` of inverted
    /// residual blocks.
    pub stages: Vec<Vec<BlockConfig>>,
    pub final_proj: FinalConfig,
    /// Projection heads count — 8 in upstream Cascade CN.
    pub n_projections: usize,
    /// Projection input channels — must equal `final_proj.c_out` (1280).
    pub c_projection_in: usize,
    /// Projection output channels — 2048 (Stage C `c_hidden`).
    pub c_projection_out: usize,
    /// Upstream `controlnet_blocks` — the global ResBlock indices (in
    /// the Stage C down+up ResBlock sequence, down=0..31, up=32..63)
    /// where each projection head's residual is injected. Head `j`
    /// injects before ResBlock `controlnet_blocks[j]`. Canny:
    /// `[0, 4, 8, 12, 51, 55, 59, 63]`.
    pub controlnet_blocks: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct StemConfig {
    pub c_out: usize,
    pub kernel: usize,
    pub stride: usize,
}

#[derive(Debug, Clone)]
pub struct FinalConfig {
    pub c_in: usize,
    pub c_out: usize,
}

#[derive(Debug, Clone)]
pub enum BlockConfig {
    /// Stage 1 pattern: single Conv (3×3, no expand/project).
    /// Tensor keys: `block.0.0.{weight,bias}` + `block.0.1.*`.
    Basic {
        c_in: usize,
        c_out: usize,
        kernel: usize,
        stride: usize,
    },
    /// Stages 2-3 pattern: expand 3×3 + project 1×1. No SE.
    /// Tensor keys: `block.0.{...}` (expand) + `block.1.{...}` (project).
    SimpleMbConv {
        c_in: usize,
        c_mid: usize,
        c_out: usize,
        kernel: usize,
        stride: usize,
    },
    /// Stages 4-6 pattern: full MBConv with SE.
    /// Tensor keys: `block.{0,1,2,3}.{...}` (expand / depthwise / SE / project).
    FullMbConv {
        c_in: usize,
        c_mid: usize,
        c_out: usize,
        /// SE reduction channels (e.g. 16 for c_mid=256).
        se_channels: usize,
        kernel: usize,
        stride: usize,
    },
}

impl Config {
    /// Upstream `stabilityai/stable-cascade/controlnet/canny.safetensors`
    /// config derived from safetensors-header inspection at v0.39
    /// phase 0.
    pub fn canny_upstream() -> Self {
        Self {
            c_in: 1,
            stem: StemConfig {
                c_out: 24,
                kernel: 3,
                stride: 2,
            },
            stages: vec![
                // Stage 1: 2 × Basic blocks (Conv 24→24)
                vec![
                    BlockConfig::Basic { c_in: 24, c_out: 24, kernel: 3, stride: 1 },
                    BlockConfig::Basic { c_in: 24, c_out: 24, kernel: 3, stride: 1 },
                ],
                // Stage 2: 4 blocks (transition + 3 preserved), no SE
                vec![
                    BlockConfig::SimpleMbConv { c_in: 24, c_mid: 96, c_out: 48, kernel: 3, stride: 2 },
                    BlockConfig::SimpleMbConv { c_in: 48, c_mid: 192, c_out: 48, kernel: 3, stride: 1 },
                    BlockConfig::SimpleMbConv { c_in: 48, c_mid: 192, c_out: 48, kernel: 3, stride: 1 },
                    BlockConfig::SimpleMbConv { c_in: 48, c_mid: 192, c_out: 48, kernel: 3, stride: 1 },
                ],
                // Stage 3: 4 blocks (transition 48→64)
                vec![
                    BlockConfig::SimpleMbConv { c_in: 48, c_mid: 192, c_out: 64, kernel: 3, stride: 2 },
                    BlockConfig::SimpleMbConv { c_in: 64, c_mid: 256, c_out: 64, kernel: 3, stride: 1 },
                    BlockConfig::SimpleMbConv { c_in: 64, c_mid: 256, c_out: 64, kernel: 3, stride: 1 },
                    BlockConfig::SimpleMbConv { c_in: 64, c_mid: 256, c_out: 64, kernel: 3, stride: 1 },
                ],
                // Stage 4: 6 blocks (transition 64→128, SE)
                // v0.40 phase 3 iter 2: block 0 has c_mid=256/se=16 (4× expand
                // from c_in=64); blocks 1-5 have c_mid=512/se=32 (4× expand
                // from c_in=128). Verified by inspection of every backbone
                // .4.X.block.{0,2}.* tensor shape.
                vec![
                    BlockConfig::FullMbConv { c_in: 64,  c_mid: 256, c_out: 128, se_channels: 16, kernel: 3, stride: 2 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 512, c_out: 128, se_channels: 32, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 512, c_out: 128, se_channels: 32, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 512, c_out: 128, se_channels: 32, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 512, c_out: 128, se_channels: 32, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 512, c_out: 128, se_channels: 32, kernel: 3, stride: 1 },
                ],
                // Stage 5: 9 blocks (transition 128→160, SE wider).
                // v0.41 phase 3: block 0 is stride 1 — EfficientNetV2-S
                // stage 5 does NOT downsample (only stages 2,3,4,6 +
                // stem do). The v0.40 stride=2 here was wrong.
                vec![
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 768, c_out: 160, se_channels: 32, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                ],
                // Stage 6: 15 blocks (transition 160→256).
                // v0.41 phase 3: block 0 is stride 2 (the downsample
                // lives here, not in stage 5). The v0.40 config had
                // stage5=s2 + stage6=s1 — two errors that canceled to
                // the right 7×7 output shape but at wrong resolutions
                // internally.
                {
                    let mut v = vec![
                        BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 256, se_channels: 40, kernel: 3, stride: 2 },
                    ];
                    for _ in 0..14 {
                        v.push(BlockConfig::FullMbConv { c_in: 256, c_mid: 1536, c_out: 256, se_channels: 64, kernel: 3, stride: 1 });
                    }
                    v
                },
            ],
            final_proj: FinalConfig {
                c_in: 256,
                c_out: 1280,
            },
            n_projections: 8,
            c_projection_in: 1280,
            c_projection_out: 2048,
            controlnet_blocks: vec![0, 4, 8, 12, 51, 55, 59, 63],
        }
    }

    /// Stable Cascade `effnet_encoder.safetensors` config — the SAME
    /// reference-verified EfficientNetV2-S backbone as
    /// [`Config::canny_upstream`], but with `c_in: 3` (RGB input rather
    /// than the canny edge map) and `n_projections: 0` (the effnet
    /// encoder has no projection heads — it ends at the backbone's
    /// 1280-channel feature map, then a small mapper produces the
    /// 16-channel Stage-C latent). Built by copying `canny_upstream`'s
    /// body with those two changes, leaving `canny_upstream` untouched.
    pub fn effnet_v2_s() -> Self {
        let mut cfg = Self::canny_upstream();
        cfg.c_in = 3;
        cfg.n_projections = 0;
        cfg
    }
}

// ---------------------------------------------------------------------
// Building blocks — `ConvBn`, `SqueezeExcitation`.
// ---------------------------------------------------------------------

struct ConvBn {
    conv: nn::Conv2d,
    bn: nn::BatchNorm,
    /// v0.41 phase 3: torchvision `Conv2dNormActivation` applies SiLU
    /// EXCEPT in the project conv of each Fused/MBConv block (built
    /// with `activation_layer=None`). The stem, expand, depthwise and
    /// head convs all activate; the project convs do not.
    act: bool,
}

impl ConvBn {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        act: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let padding = kernel / 2;
        // v0.40 phase 3 iter 1: upstream Conv2d → BN pipelines have NO
        // Conv bias (the BN bias absorbs it). Verified by inspection:
        // backbone.{stage}.{block}.block.{0,1,3}.0.weight exists but
        // .0.bias does NOT.
        let conv = nn::conv2d_no_bias(
            in_c,
            out_c,
            kernel,
            nn::Conv2dConfig {
                stride,
                padding,
                groups,
                ..Default::default()
            },
            vb.pp("0"),
        )
        .map_err(|e| anyhow!("ConvBn conv: {e}"))?;
        // v0.41 phase 3: torchvision EfficientNet uses BatchNorm2d with
        // eps=1e-3 (not the candle/torch default 1e-5).
        let bn = nn::batch_norm(
            out_c,
            nn::BatchNormConfig { eps: 1e-3, ..Default::default() },
            vb.pp("1"),
        )
        .map_err(|e| anyhow!("ConvBn bn: {e}"))?;
        Ok(Self { conv, bn, act })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        if self.act { Ok(x.silu()?) } else { Ok(x) }
    }
}

struct SqueezeExcitation {
    fc1: nn::Conv2d,
    fc2: nn::Conv2d,
}

impl SqueezeExcitation {
    fn new(channels: usize, reduce_channels: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = nn::conv2d(
            channels,
            reduce_channels,
            1,
            Default::default(),
            vb.pp("fc1"),
        )
        .map_err(|e| anyhow!("SE fc1: {e}"))?;
        let fc2 = nn::conv2d(
            reduce_channels,
            channels,
            1,
            Default::default(),
            vb.pp("fc2"),
        )
        .map_err(|e| anyhow!("SE fc2: {e}"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;
        // Global average pool → (B, C, 1, 1).
        let pooled = x.mean_keepdim(2)?.mean_keepdim(3)?;
        let s = self.fc1.forward(&pooled)?;
        let s = s.silu()?;
        let s = self.fc2.forward(&s)?;
        // Sigmoid gate (hardsigmoid in MobileNetV3, approx with sigmoid).
        let gate = nn::ops::sigmoid(&s)?;
        // Broadcast-multiply: (B, C, 1, 1) × (B, C, H, W).
        let _ = (h, w); // already broadcastable
        Ok(x.broadcast_mul(&gate)?)
    }
}

// ---------------------------------------------------------------------
// InvertedResidual block — three variants matching upstream stages.
// ---------------------------------------------------------------------

enum InvertedResidual {
    Basic {
        block_0: ConvBn,
        residual: bool,
    },
    SimpleMb {
        expand: ConvBn,
        project: ConvBn,
        residual: bool,
    },
    FullMb {
        expand: ConvBn,
        depthwise: ConvBn,
        se: SqueezeExcitation,
        project: ConvBn,
        residual: bool,
    },
}

impl InvertedResidual {
    fn new(cfg: &BlockConfig, vb: VarBuilder) -> Result<Self> {
        let block_vb = vb.pp("block");
        match cfg {
            BlockConfig::Basic {
                c_in,
                c_out,
                kernel,
                stride,
            } => {
                let block_0 = ConvBn::new(
                    *c_in,
                    *c_out,
                    *kernel,
                    *stride,
                    1,
                    true, // FusedMBConv expand=1: single conv WITH SiLU
                    block_vb.pp("0"),
                )?;
                let residual = *stride == 1 && c_in == c_out;
                Ok(Self::Basic { block_0, residual })
            }
            BlockConfig::SimpleMbConv {
                c_in,
                c_mid,
                c_out,
                kernel,
                stride,
            } => {
                let expand = ConvBn::new(*c_in, *c_mid, *kernel, *stride, 1, true, block_vb.pp("0"))?;
                // Project conv: NO activation (torchvision FusedMBConv).
                let project = ConvBn::new(*c_mid, *c_out, 1, 1, 1, false, block_vb.pp("1"))?;
                let residual = *stride == 1 && c_in == c_out;
                Ok(Self::SimpleMb {
                    expand,
                    project,
                    residual,
                })
            }
            BlockConfig::FullMbConv {
                c_in,
                c_mid,
                c_out,
                se_channels,
                kernel,
                stride,
            } => {
                let expand = ConvBn::new(*c_in, *c_mid, 1, 1, 1, true, block_vb.pp("0"))?;
                let depthwise = ConvBn::new(
                    *c_mid,
                    *c_mid,
                    *kernel,
                    *stride,
                    *c_mid, // groups = c_mid → depthwise
                    true,
                    block_vb.pp("1"),
                )?;
                let se = SqueezeExcitation::new(*c_mid, *se_channels, block_vb.pp("2"))?;
                // Project conv: NO activation (torchvision MBConv).
                let project = ConvBn::new(*c_mid, *c_out, 1, 1, 1, false, block_vb.pp("3"))?;
                let residual = *stride == 1 && c_in == c_out;
                Ok(Self::FullMb {
                    expand,
                    depthwise,
                    se,
                    project,
                    residual,
                })
            }
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Basic { block_0, residual } => {
                let h = block_0.forward(x)?;
                if *residual { Ok(x.add(&h)?) } else { Ok(h) }
            }
            Self::SimpleMb {
                expand,
                project,
                residual,
            } => {
                let h = expand.forward(x)?;
                let h = project.forward(&h)?;
                if *residual { Ok(x.add(&h)?) } else { Ok(h) }
            }
            Self::FullMb {
                expand,
                depthwise,
                se,
                project,
                residual,
            } => {
                let h = expand.forward(x)?;
                let h = depthwise.forward(&h)?;
                let h = se.forward(&h)?;
                let h = project.forward(&h)?;
                if *residual { Ok(x.add(&h)?) } else { Ok(h) }
            }
        }
    }
}

// ---------------------------------------------------------------------
// ProjectionHead — Sequential(Conv 1×1, GELU, Conv 1×1).
// ---------------------------------------------------------------------

struct ProjectionHead {
    conv_0: nn::Conv2d,
    conv_2: nn::Conv2d,
}

impl ProjectionHead {
    fn new(c_in: usize, c_mid: usize, c_out: usize, vb: VarBuilder) -> Result<Self> {
        // v0.40 phase 3 iter 1: projections have weight only (no bias)
        // per inspection: projections.{0..7}.{0,2}.weight exists but
        // not .bias.
        let conv_0 = nn::conv2d_no_bias(c_in, c_mid, 1, Default::default(), vb.pp("0"))
            .map_err(|e| anyhow!("ProjectionHead.0: {e}"))?;
        // index 1 is the activation (GELU/SiLU) — no params.
        let conv_2 = nn::conv2d_no_bias(c_mid, c_out, 1, Default::default(), vb.pp("2"))
            .map_err(|e| anyhow!("ProjectionHead.2: {e}"))?;
        Ok(Self { conv_0, conv_2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv_0.forward(x)?;
        // v0.41 phase 3: upstream projection uses LeakyReLU(0.2), not
        // GELU (Stability-AI/StableCascade modules/controlnet.py).
        let h = leaky_relu(&h, 0.2)?;
        Ok(self.conv_2.forward(&h)?)
    }
}

/// LeakyReLU(x, slope) = max(x, 0) + slope * min(x, 0).
fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = (x - &pos)?.affine(slope, 0.0)?;
    pos.add(&neg).map_err(|e| e.into())
}

// ---------------------------------------------------------------------
// CascadeControlNet — full model.
// ---------------------------------------------------------------------

pub struct CascadeControlNet {
    stem: ConvBn,
    stages: Vec<Vec<InvertedResidual>>,
    final_proj: ConvBn,
    projections: Vec<ProjectionHead>,
    pub cfg: Config,
    pub dtype: DType,
    pub device: Device,
}

impl CascadeControlNet {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();

        let stem = ConvBn::new(
            cfg.c_in,
            cfg.stem.c_out,
            cfg.stem.kernel,
            cfg.stem.stride,
            1,
            true,
            vb.pp("backbone").pp("0"),
        )?;

        let mut stages = Vec::with_capacity(cfg.stages.len());
        for (stage_idx, blocks) in cfg.stages.iter().enumerate() {
            // Stages 1..=6 in upstream — our `stages` vec starts at
            // upstream index 1, so stage_idx 0 here = `backbone.1`.
            let stage_pp = stage_idx + 1;
            let stage_vb = vb.pp("backbone").pp(&stage_pp.to_string());
            let mut built = Vec::with_capacity(blocks.len());
            for (b, bc) in blocks.iter().enumerate() {
                built.push(InvertedResidual::new(bc, stage_vb.pp(&b.to_string()))?);
            }
            stages.push(built);
        }

        // Final projection at backbone.{stages.len() + 1} (upstream index 7).
        let final_idx = cfg.stages.len() + 1;
        let final_proj = ConvBn::new(
            cfg.final_proj.c_in,
            cfg.final_proj.c_out,
            1,
            1,
            1,
            true,
            vb.pp("backbone").pp(&final_idx.to_string()),
        )?;

        let mut projections = Vec::with_capacity(cfg.n_projections);
        for i in 0..cfg.n_projections {
            projections.push(ProjectionHead::new(
                cfg.c_projection_in,
                cfg.c_projection_in,
                cfg.c_projection_out,
                vb.pp("projections").pp(&i.to_string()),
            )?);
        }

        Ok(Self {
            stem,
            stages,
            final_proj,
            projections,
            cfg,
            dtype,
            device,
        })
    }

    /// Run the backbone over the conditioning input.
    pub fn backbone_features(&self, conditioning: &Tensor) -> Result<Tensor> {
        let mut h = self.stem.forward(conditioning)?;
        for stage in &self.stages {
            for blk in stage {
                h = blk.forward(&h)?;
            }
        }
        self.final_proj.forward(&h)
    }

    /// Full forward: backbone → 8 projection heads. Returns one
    /// residual per projection head; each has the same spatial size
    /// as the backbone output (the residuals share the feature map
    /// and get injected at distinct points in Stage C).
    pub fn forward(&self, conditioning: &Tensor) -> Result<Vec<Tensor>> {
        let features = self.backbone_features(conditioning)?;
        let mut residuals = Vec::with_capacity(self.projections.len());
        for head in &self.projections {
            residuals.push(head.forward(&features)?);
        }
        Ok(residuals)
    }

    /// Number of CN residuals available (= `n_projections`).
    pub fn n_residuals(&self) -> usize {
        self.projections.len()
    }
}

// =====================================================================
// EffNetEncoder — Stable Cascade `effnet_encoder.safetensors`.
// =====================================================================
//
// Architecture (upstream `EfficientNetEncoder`):
//
//   backbone = torchvision efficientnet_v2_s.features  → (B, 1280, 24, 24)
//   mapper   = Sequential(
//                Conv2d(1280, 16, kernel=1, bias=False),   # mapper.0
//                BatchNorm2d(16, affine=False),            # mapper.1
//              )
//
// Input: RGB, ImageNet-normalized, 768×768 → backbone /32 → 24×24 →
// mapper → (B, 16, 24, 24). That 16×24×24 tensor is the Stage-C latent
// (the x0 source for Stage-C LoRA training).
//
// The backbone is byte-for-byte the same EfficientNetV2-S we already
// build for the ControlNet (`CascadeControlNet::new`), reusing the
// reference-verified `ConvBn` / `InvertedResidual` blocks and the same
// `backbone.{0..7}` tensor-key layout. The only new pieces are the
// `mapper` conv (no bias) and the affine-free BatchNorm running stats.

pub struct EffNetEncoder {
    stem: ConvBn,
    stages: Vec<Vec<InvertedResidual>>,
    final_proj: ConvBn,
    /// mapper.0 — Conv2d(1280 → 16, kernel=1, bias=False).
    mapper_conv: nn::Conv2d,
    /// mapper.1 — BatchNorm2d(16, affine=False) running stats. Because
    /// affine=False there is NO weight/bias, only running_mean/var.
    bn_mean: Tensor,
    bn_var: Tensor,
    pub dtype: DType,
    pub device: Device,
}

impl EffNetEncoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let cfg = Config::effnet_v2_s();
        let dtype = vb.dtype();
        let device = vb.device().clone();

        // --- backbone (identical construction to CascadeControlNet::new) ---
        let stem = ConvBn::new(
            cfg.c_in,
            cfg.stem.c_out,
            cfg.stem.kernel,
            cfg.stem.stride,
            1,
            true,
            vb.pp("backbone").pp("0"),
        )?;

        let mut stages = Vec::with_capacity(cfg.stages.len());
        for (stage_idx, blocks) in cfg.stages.iter().enumerate() {
            let stage_pp = stage_idx + 1;
            let stage_vb = vb.pp("backbone").pp(&stage_pp.to_string());
            let mut built = Vec::with_capacity(blocks.len());
            for (b, bc) in blocks.iter().enumerate() {
                built.push(InvertedResidual::new(bc, stage_vb.pp(&b.to_string()))?);
            }
            stages.push(built);
        }

        let final_idx = cfg.stages.len() + 1;
        let final_proj = ConvBn::new(
            cfg.final_proj.c_in,
            cfg.final_proj.c_out,
            1,
            1,
            1,
            true,
            vb.pp("backbone").pp(&final_idx.to_string()),
        )?;

        // --- mapper ---
        let mapper_conv = nn::conv2d_no_bias(
            1280,
            16,
            1,
            Default::default(),
            vb.pp("mapper").pp("0"),
        )
        .map_err(|e| anyhow!("EffNetEncoder mapper.0: {e}"))?;
        // affine=False BatchNorm2d → only running_mean / running_var.
        let bn_vb = vb.pp("mapper").pp("1");
        let bn_mean = bn_vb
            .get(16, "running_mean")
            .map_err(|e| anyhow!("EffNetEncoder mapper.1.running_mean: {e}"))?;
        let bn_var = bn_vb
            .get(16, "running_var")
            .map_err(|e| anyhow!("EffNetEncoder mapper.1.running_var: {e}"))?;

        Ok(Self {
            stem,
            stages,
            final_proj,
            mapper_conv,
            bn_mean,
            bn_var,
            dtype,
            device,
        })
    }

    /// Run the EfficientNetV2-S backbone over a preprocessed image.
    /// `image` is already ImageNet-normalized (1, 3, 768, 768).
    fn backbone_features(&self, image: &Tensor) -> Result<Tensor> {
        let mut h = self.stem.forward(image)?;
        for stage in &self.stages {
            for blk in stage {
                h = blk.forward(&h)?;
            }
        }
        self.final_proj.forward(&h)
    }

    /// Encode a preprocessed RGB image (1, 3, 768, 768) into the
    /// Stage-C latent (B, 16, 24, 24). Applies the backbone, the
    /// mapper conv, then the affine-free BatchNorm (eps=1e-5).
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let features = self.backbone_features(image)?; // (B, 1280, 24, 24)
        let h = self.mapper_conv.forward(&features)?; // (B, 16, 24, 24)
        // Manual affine-free BatchNorm in inference mode:
        //   y = (h - running_mean) / sqrt(running_var + eps)
        let mean = self.bn_mean.reshape((1, 16, 1, 1))?;
        let var = self.bn_var.reshape((1, 16, 1, 1))?;
        let denom = (var + 1e-5)?.sqrt()?;
        let y = h
            .broadcast_sub(&mean)?
            .broadcast_div(&denom)?;
        Ok(y)
    }
}

/// Fetch the Stable Cascade effnet encoder weights, preferring the
/// full-precision file and falling back to the bf16 variant.
pub async fn download_effnet_encoder() -> Result<std::path::PathBuf> {
    crate::hf::download::get_first_of(&[
        ("stabilityai/stable-cascade", "effnet_encoder.safetensors"),
        ("stabilityai/stable-cascade", "effnet_encoder.bf16.safetensors"),
    ])
    .await
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn small_cfg() -> Config {
        // Tiny CN: 2 stages, single block each, 2 projections.
        // Stem 1→4 stride 2; stage 1 has 1 Basic; stage 2 has 1 SimpleMbConv 4→8.
        // Final 8→16. Projections 16→16→32.
        Config {
            c_in: 1,
            stem: StemConfig { c_out: 4, kernel: 3, stride: 2 },
            stages: vec![
                vec![BlockConfig::Basic { c_in: 4, c_out: 4, kernel: 3, stride: 1 }],
                vec![BlockConfig::SimpleMbConv { c_in: 4, c_mid: 16, c_out: 8, kernel: 3, stride: 2 }],
            ],
            final_proj: FinalConfig { c_in: 8, c_out: 16 },
            n_projections: 2,
            c_projection_in: 16,
            c_projection_out: 32,
            controlnet_blocks: vec![0, 1],
        }
    }

    fn random_cn(cfg: Config) -> (CascadeControlNet, VarMap) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cn = CascadeControlNet::new(cfg, vb).expect("CascadeControlNet::new");
        (cn, varmap)
    }

    #[test]
    fn canny_upstream_config_matches_inspection() {
        let cfg = Config::canny_upstream();
        assert_eq!(cfg.c_in, 1);
        assert_eq!(cfg.stem.c_out, 24);
        assert_eq!(cfg.stem.stride, 2);
        // 6 inverted-residual stages (upstream indices 1..=6).
        assert_eq!(cfg.stages.len(), 6);
        // Stage block counts match inspection: [2, 4, 4, 6, 9, 15].
        let counts: Vec<usize> = cfg.stages.iter().map(|s| s.len()).collect();
        assert_eq!(counts, vec![2, 4, 4, 6, 9, 15]);
        assert_eq!(cfg.final_proj.c_in, 256);
        assert_eq!(cfg.final_proj.c_out, 1280);
        assert_eq!(cfg.n_projections, 8);
        assert_eq!(cfg.c_projection_in, 1280);
        assert_eq!(cfg.c_projection_out, 2048);
    }

    #[test]
    fn small_cn_forward_returns_n_residuals() {
        let (cn, _) = random_cn(small_cfg());
        let device = &cn.device;
        // Input must be large enough for the 3 stride-2 stages
        // (stem + stage 2 first block) to leave a non-zero spatial
        // size. 16×16 → 8×8 → 4×4.
        let cond = Tensor::randn(0f32, 1f32, (1, 1, 16, 16), device).unwrap();
        let residuals = cn.forward(&cond).unwrap();
        assert_eq!(residuals.len(), 2);
        // Each residual: 32 channels output (c_projection_out).
        for r in &residuals {
            assert_eq!(r.dims()[0], 1);
            assert_eq!(r.dims()[1], 32);
        }
    }

    #[test]
    fn small_cn_residuals_differ_across_heads() {
        // Each projection head has independent params → outputs should
        // differ. If two heads happened to be byte-identical, we'd have
        // a registration bug.
        let (cn, _) = random_cn(small_cfg());
        let device = &cn.device;
        let cond = Tensor::randn(0f32, 1f32, (1, 1, 16, 16), device).unwrap();
        let residuals = cn.forward(&cond).unwrap();
        let r0 = &residuals[0];
        let r1 = &residuals[1];
        let diff = (r0 - r1)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-5,
            "projection heads should produce distinct residuals ({diff})"
        );
    }

    #[test]
    fn cn_residual_changes_with_input() {
        // Load-bearing: the backbone must actually consume the
        // conditioning input. If the forward dropped it, residuals
        // would be identical across inputs.
        let (cn, _) = random_cn(small_cfg());
        let device = &cn.device;
        let c1 = Tensor::randn(0f32, 1f32, (1, 1, 16, 16), device).unwrap();
        let c2 = Tensor::randn(0f32, 1f32, (1, 1, 16, 16), device).unwrap();
        let r1 = cn.forward(&c1).unwrap();
        let r2 = cn.forward(&c2).unwrap();
        let diff = (&r1[0] - &r2[0])
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-5, "residual must depend on input ({diff})");
    }

    #[test]
    fn squeeze_excitation_gates_input_via_channel_attention() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let se = SqueezeExcitation::new(8, 2, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &device).unwrap();
        let y = se.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8, 4, 4]);
    }

    #[test]
    fn inverted_residual_basic_with_residual_skip() {
        // Basic block with stride 1 + matching channels → residual.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = BlockConfig::Basic { c_in: 8, c_out: 8, kernel: 3, stride: 1 };
        let blk = InvertedResidual::new(&cfg, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8, 4, 4]);
    }

    #[test]
    fn inverted_residual_full_mb_with_se_and_no_residual() {
        // FullMbConv with stride 2 → no residual.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = BlockConfig::FullMbConv {
            c_in: 8, c_mid: 32, c_out: 16, se_channels: 4, kernel: 3, stride: 2,
        };
        let blk = InvertedResidual::new(&cfg, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 8, 8), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 16, 4, 4]);
    }

    /// Real-weight smoke for the Stable Cascade effnet encoder.
    /// Builds [`EffNetEncoder`] from the cached
    /// `effnet_encoder.safetensors` and pushes a random
    /// (1, 3, 768, 768) F32 input through `encode`, asserting the
    /// output is the (1, 16, 24, 24) Stage-C latent. The test SKIPS
    /// (early-returns) if the checkpoint isn't already in the HF cache
    /// — it never downloads. To run: ensure
    /// `stabilityai/stable-cascade/effnet_encoder.safetensors` is
    /// cached, then `cargo test --release --lib effnet`.
    #[test]
    fn effnet_encoder_loads_and_shapes() {
        // Locate effnet_encoder.safetensors in the HF cache (or via
        // STABLE_CASCADE_WEIGHTS_DIR), skipping if not present.
        let weights = match effnet_cached_weights() {
            Some(p) => p,
            None => {
                eprintln!(
                    "Skipping effnet_encoder_loads_and_shapes: \
                     effnet_encoder.safetensors not found in HF cache \
                     (~/.cache/huggingface) or STABLE_CASCADE_WEIGHTS_DIR."
                );
                return;
            }
        };

        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device)
                .expect("mmap effnet_encoder.safetensors")
        };
        let enc = EffNetEncoder::new(vb).expect("build EffNetEncoder from real weights");

        let image = Tensor::randn(0f32, 1f32, (1, 3, 768, 768), &device).unwrap();
        let latent = enc.encode(&image).expect("encode");
        assert_eq!(
            latent.dims(),
            &[1, 16, 24, 24],
            "Stage-C latent must be (1, 16, 24, 24)"
        );
        let mean = latent.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let std = {
            let m = latent.broadcast_sub(&latent.mean_all().unwrap()).unwrap();
            (m.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap()).sqrt()
        };
        eprintln!(
            "[effnet] latent dims={:?} mean={mean:.5} std={std:.5}",
            latent.dims()
        );
    }

    /// Find a cached `effnet_encoder.safetensors`. Checks
    /// `STABLE_CASCADE_WEIGHTS_DIR` first, then the standard HF hub
    /// snapshot layout under `~/.cache/huggingface`. Returns `None`
    /// (→ test skips) when absent. Never downloads.
    fn effnet_cached_weights() -> Option<std::path::PathBuf> {
        if let Ok(dir) = std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            for f in ["effnet_encoder.safetensors", "effnet_encoder.bf16.safetensors"] {
                let p = std::path::PathBuf::from(&dir).join(f);
                if p.exists() {
                    return Some(p);
                }
            }
        }
        let home = std::env::var("HOME").ok()?;
        let base = std::path::PathBuf::from(format!(
            "{home}/.cache/huggingface/hub/models--stabilityai--stable-cascade/snapshots"
        ));
        let snaps = std::fs::read_dir(&base).ok()?;
        for entry in snaps.filter_map(|e| e.ok()) {
            let snap = entry.path();
            for f in ["effnet_encoder.safetensors", "effnet_encoder.bf16.safetensors"] {
                let p = snap.join(f);
                if p.exists() {
                    return Some(p);
                }
            }
        }
        None
    }

    /// v0.40 phase 3: real-weight smoke for the Stable Cascade
    /// canny ControlNet. Skipped unless `STABLE_CASCADE_WEIGHTS_DIR`
    /// env var points at a directory containing
    /// `controlnet/canny.safetensors`. ~16 MB checkpoint.
    #[test]
    fn cn_canny_loads_from_real_upstream_weights() {
        let dir = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let path = std::path::PathBuf::from(&dir)
            .join("controlnet/canny.safetensors");
        if !path.exists() {
            eprintln!(
                "Skipping cn_canny_loads_from_real_upstream_weights: \
                 {} doesn't exist (set STABLE_CASCADE_WEIGHTS_DIR to a \
                 directory containing controlnet/canny.safetensors from \
                 stabilityai/stable-cascade).",
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
            .expect("mmap canny CN weights")
        };
        match CascadeControlNet::new(Config::canny_upstream(), vb) {
            Ok(_) => eprintln!(
                "✓ Cascade ControlNet (canny) real-weight load OK ({})",
                path.display()
            ),
            Err(e) => panic!(
                "Cascade ControlNet (canny) real-weight load FAILED — \
                 indicates tensor naming mismatch between v0.39 cascade_cn \
                 and upstream:\n  {e}"
            ),
        }
    }

    /// v0.41 phase 3: reference comparison vs torchvision EfficientNetV2-S
    /// + canny projections. Loads `/tmp/cascade_ref_cn.safetensors`
    /// (from tools/cascade_ref_dump_cn.py), feeds the same input through
    /// our CascadeControlNet.forward, diffs the 8 residuals.
    #[test]
    fn cn_forward_matches_reference() {
        let dir = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let ref_path = std::path::PathBuf::from("/tmp/cascade_ref_cn.safetensors");
        if !ref_path.exists() {
            eprintln!("Skipping: /tmp/cascade_ref_cn.safetensors not found (run tools/cascade_ref_dump_cn.py)");
            return;
        }
        let weights = std::path::PathBuf::from(&dir).join("controlnet/canny.safetensors");
        if !weights.exists() {
            return;
        }
        let device = Device::Cpu;
        let refs = candle_core::safetensors::load(&ref_path, &device).expect("load ref");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights.as_path()], DType::F32, &device)
                .expect("mmap")
        };
        let cn = CascadeControlNet::new(Config::canny_upstream(), vb).expect("new");
        let cond = refs.get("in_cond").unwrap().to_dtype(DType::F32).unwrap();
        let mad = |a: &Tensor, b: &Tensor| {
            (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
        };
        // Backbone feature first.
        let feat = cn.backbone_features(&cond).unwrap();
        eprintln!(
            "[refCN] backbone_feat ours={:?} ref={:?}  max_abs_diff={:.5}",
            feat.dims(), refs.get("backbone_feat").unwrap().dims(),
            mad(&feat, refs.get("backbone_feat").unwrap())
        );
        let bb_diff = mad(&feat, refs.get("backbone_feat").unwrap());
        assert!(
            bb_diff < 0.01,
            "CN backbone_feat must match torchvision EfficientNetV2-S (got {bb_diff})"
        );
        let residuals = cn.forward(&cond).unwrap();
        for (i, r) in residuals.iter().enumerate() {
            let key = format!("residual_{i}");
            if let Some(rr) = refs.get(&key) {
                let d = mad(r, rr);
                eprintln!(
                    "[refCN] {key} ours={:?} ref={:?}  max_abs_diff={d:.5}  (ref range [{:.2},{:.2}])",
                    r.dims(), rr.dims(),
                    rr.min_all().unwrap().to_scalar::<f32>().unwrap(),
                    rr.max_all().unwrap().to_scalar::<f32>().unwrap(),
                );
                assert!(d < 0.05, "CN {key} must match reference (got {d})");
            }
        }
    }
}
