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
                vec![
                    BlockConfig::FullMbConv { c_in: 64,  c_mid: 256, c_out: 128, se_channels: 16, kernel: 3, stride: 2 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 256, c_out: 128, se_channels: 16, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 256, c_out: 128, se_channels: 16, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 256, c_out: 128, se_channels: 16, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 256, c_out: 128, se_channels: 16, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 256, c_out: 128, se_channels: 16, kernel: 3, stride: 1 },
                ],
                // Stage 5: 9 blocks (transition 128→160, SE wider)
                vec![
                    BlockConfig::FullMbConv { c_in: 128, c_mid: 768, c_out: 160, se_channels: 32, kernel: 3, stride: 2 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                    BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 160, se_channels: 40, kernel: 3, stride: 1 },
                ],
                // Stage 6: 15 blocks (transition 160→256)
                {
                    let mut v = vec![
                        BlockConfig::FullMbConv { c_in: 160, c_mid: 960, c_out: 256, se_channels: 40, kernel: 3, stride: 1 },
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
        }
    }
}

// ---------------------------------------------------------------------
// Building blocks — `ConvBn`, `SqueezeExcitation`.
// ---------------------------------------------------------------------

struct ConvBn {
    conv: nn::Conv2d,
    bn: nn::BatchNorm,
}

impl ConvBn {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
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
        let bn = nn::batch_norm(out_c, nn::BatchNormConfig::default(), vb.pp("1"))
            .map_err(|e| anyhow!("ConvBn bn: {e}"))?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        Ok(x.silu()?)
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
                let expand = ConvBn::new(*c_in, *c_mid, *kernel, *stride, 1, block_vb.pp("0"))?;
                let project = ConvBn::new(*c_mid, *c_out, 1, 1, 1, block_vb.pp("1"))?;
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
                let expand = ConvBn::new(*c_in, *c_mid, 1, 1, 1, block_vb.pp("0"))?;
                let depthwise = ConvBn::new(
                    *c_mid,
                    *c_mid,
                    *kernel,
                    *stride,
                    *c_mid, // groups = c_mid → depthwise
                    block_vb.pp("1"),
                )?;
                let se = SqueezeExcitation::new(*c_mid, *se_channels, block_vb.pp("2"))?;
                let project = ConvBn::new(*c_mid, *c_out, 1, 1, 1, block_vb.pp("3"))?;
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
        let h = h.gelu()?;
        Ok(self.conv_2.forward(&h)?)
    }
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
}
