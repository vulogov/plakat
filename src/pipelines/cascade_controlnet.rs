//! Stable Cascade ControlNet — v0.38 phase 5.
//!
//! Stage C is where spatial conditioning lands in the 3-stage
//! architecture: Stage C generates the prior latent (24×24×16) from
//! text; Stage B refines it into Stage A's latent; Stage A decodes.
//! Adding a ControlNet to Stage C lets the user steer the
//! semantic prior with a conditioning image (canny edges, depth
//! map, etc.) the same way SD/SDXL/Flux/SD3 ControlNets steer
//! their backbones.
//!
//! ## Architecture
//!
//! Most community Stable Cascade ControlNets ship a compact
//! image-to-residual encoder: a few strided convs that compress
//! the 1024×1024×3 conditioning image down to Stage C's working
//! shape (24×24×16), then a small projection head that emits the
//! residual the Stage C noise latent gets added to BEFORE the
//! `in_conv`.
//!
//! ```text
//!   conditioning_image (B, 3, 1024, 1024)
//!     │
//!     ▼  strided 3×3 convs (5 stages, ×2 spatial)
//!     │    1024 → 512 → 256 → 128 → 64 → 32
//!     │
//!     ▼  bilinear-ish resize 32 → 24
//!     │
//!     ▼  3×3 conv + SiLU + 3×3 conv (zero-init head)
//!     │    `out_channels = 16` (matches Stage C `channels`)
//!     │
//!     ▼  residual (B, 16, 24, 24)
//!
//!   noisy_latent (B, 16, 24, 24)
//!     │ + residual * cn_scale
//!     ▼
//!   Stage C UNet forward
//! ```
//!
//! Time + text conditioning is NOT used by this minimal CN — the
//! residual is computed once before the denoise loop. Community
//! Cascade ControlNets follow this pattern; the more elaborate
//! "Full" pattern that produces per-level residuals adds another
//! ~600 lines and is deferred.
//!
//! ## Single-CN scope (v0.38)
//!
//! Multi-ControlNet for Stable Cascade is deferred to v0.39. The
//! CLI bails with an actionable error when more than one
//! `--control-spec` is supplied for a Cascade model.
//!
//! ## Tensor naming
//!
//! Same caveat as the rest of the v0.37/v0.38 Cascade stack —
//! plakat uses its own internal layout (`image_encoder.{i}.*`,
//! `final_conv.weight`, etc.); real-weight verification at user
//! smoke time is the gating step for output quality.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Architectural config for the Cascade ControlNet image encoder.
#[derive(Debug, Clone)]
pub struct Config {
    /// Output channel count — must match Stage C's input channels.
    /// Always 16 for Stable Cascade.
    pub out_channels: usize,
    /// Channels at each downsample stage, coarsest-LAST. Default:
    /// `[16, 32, 64, 128, 256, 512]` (5 strided convs).
    pub stage_channels: Vec<usize>,
    /// Target spatial side after the resize-then-refine head. Always
    /// 24 for stock Stable Cascade (Stage C operates on 24×24
    /// latents at 1024² output res).
    pub target_size: usize,
}

impl Config {
    /// Default config matching the upstream Stable Cascade Stage C
    /// input shape (24×24×16) at 1024² output.
    pub fn stable_cascade_default() -> Self {
        Self {
            out_channels: 16,
            stage_channels: vec![16, 32, 64, 128, 256, 512],
            target_size: 24,
        }
    }
}

/// Stable Cascade ControlNet.
pub struct CascadeControlNet {
    /// First conv: (3 → stage_channels[0]) at full image resolution
    /// (1024). 3×3 conv with padding=1.
    in_conv: nn::Conv2d,
    /// Strided downsample stack: each entry compresses spatial by
    /// 2×. Length = `stage_channels.len() - 1`.
    down_blocks: Vec<nn::Conv2d>,
    /// Refinement conv at the resized (target_size × target_size)
    /// grid. 3×3 conv with padding=1.
    refine_conv: nn::Conv2d,
    /// Zero-init projection head producing the final residual.
    /// 3×3 conv with padding=1; output channels = `out_channels`.
    final_conv: nn::Conv2d,
    pub cfg: Config,
    pub dtype: DType,
    pub device: Device,
}

impl CascadeControlNet {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let stride2_cfg = nn::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };

        let first = cfg.stage_channels[0];
        let in_conv = nn::conv2d(3, first, 3, conv_cfg, vb.pp("in_conv"))
            .map_err(|e| anyhow!("CascadeControlNet in_conv: {e}"))?;

        let mut down_blocks = Vec::with_capacity(cfg.stage_channels.len() - 1);
        for i in 0..cfg.stage_channels.len() - 1 {
            let in_c = cfg.stage_channels[i];
            let out_c = cfg.stage_channels[i + 1];
            down_blocks.push(
                nn::conv2d(
                    in_c,
                    out_c,
                    3,
                    stride2_cfg,
                    vb.pp("down_blocks").pp(&i.to_string()),
                )
                .map_err(|e| anyhow!("CascadeControlNet down_blocks.{i}: {e}"))?,
            );
        }

        let last = cfg.stage_channels[cfg.stage_channels.len() - 1];
        let refine_conv = nn::conv2d(last, last, 3, conv_cfg, vb.pp("refine_conv"))
            .map_err(|e| anyhow!("CascadeControlNet refine_conv: {e}"))?;
        let final_conv = nn::conv2d(
            last,
            cfg.out_channels,
            3,
            conv_cfg,
            vb.pp("final_conv"),
        )
        .map_err(|e| anyhow!("CascadeControlNet final_conv: {e}"))?;

        Ok(Self {
            in_conv,
            down_blocks,
            refine_conv,
            final_conv,
            cfg,
            dtype,
            device,
        })
    }

    /// Forward pass: `(B, 3, H, W)` conditioning image →
    /// `(B, out_channels, target_size, target_size)` residual.
    ///
    /// The caller scales by `cn_scale` and adds to the Stage C
    /// noisy latent at the input — see
    /// `cascade::Pipeline::generate` for the wiring.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let mut x = self.in_conv.forward(image)?;
        x = x.silu()?;
        for blk in &self.down_blocks {
            x = blk.forward(&x)?;
            x = x.silu()?;
        }
        // Resize the downsampled features to the target spatial
        // extent. Nearest-2d is what's in candle-core; close enough
        // for a 32→24 squeeze on coarse features.
        let (_b, _c, h, _w) = x.dims4()?;
        if h != self.cfg.target_size {
            x = x.upsample_nearest2d(self.cfg.target_size, self.cfg.target_size)?;
        }
        x = self.refine_conv.forward(&x)?;
        x = x.silu()?;
        Ok(self.final_conv.forward(&x)?)
    }
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn small_cfg() -> Config {
        // Tiny config for fast tests: 3 strided convs (8×
        // compression) targeting 4×4 output. 1024 / 8 = 128, then
        // resize 128 → 4.
        Config {
            out_channels: 16,
            stage_channels: vec![4, 8, 16, 32],
            target_size: 4,
        }
    }

    fn random_cn(cfg: Config) -> (CascadeControlNet, VarMap) {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let cn = CascadeControlNet::new(cfg, vb).expect("CascadeControlNet::new");
        (cn, varmap)
    }

    #[test]
    fn default_config_targets_stage_c_input_shape() {
        let cfg = Config::stable_cascade_default();
        assert_eq!(cfg.out_channels, 16);
        assert_eq!(cfg.target_size, 24);
        assert_eq!(cfg.stage_channels.len(), 6);
    }

    #[test]
    fn forward_produces_target_spatial_shape() {
        let (cn, _) = random_cn(small_cfg());
        // (1, 3, 32, 32) input — 3 strided convs → (1, 32, 4, 4),
        // already at target_size, refine + final → (1, 16, 4, 4).
        let img = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), &cn.device).unwrap();
        let r = cn.forward(&img).unwrap();
        assert_eq!(r.dims(), &[1, 16, 4, 4]);
    }

    #[test]
    fn forward_resizes_when_downsample_overshoots_target() {
        let (cn, _) = random_cn(small_cfg());
        // (1, 3, 64, 64) input — 3 strided convs → (1, 32, 8, 8),
        // resize 8 → 4, then refine + final → (1, 16, 4, 4).
        let img = Tensor::randn(0f32, 1f32, (1, 3, 64, 64), &cn.device).unwrap();
        let r = cn.forward(&img).unwrap();
        assert_eq!(r.dims(), &[1, 16, 4, 4]);
    }

    #[test]
    fn forward_changes_output_when_input_changes() {
        // Sanity: the CN must actually USE the input image. If a
        // forward dropped the conditioning on the floor, both calls
        // would produce identical residuals.
        let (cn, _) = random_cn(small_cfg());
        let img1 = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), &cn.device).unwrap();
        let img2 = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), &cn.device).unwrap();
        let r1 = cn.forward(&img1).unwrap();
        let r2 = cn.forward(&img2).unwrap();
        let diff = (&r1 - &r2)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-4,
            "CN output should depend on input image (mean abs diff {diff})"
        );
    }
}
