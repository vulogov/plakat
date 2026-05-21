//! Lineart annotator network used by ControlNet's `lineart` conditioner.
//!
//! Ported from lllyasviel's `sk_model.pth` (Generator from his
//! `awacke1/Image-to-Line-Drawings` lineart trainer). The architecture
//! is a small CycleGAN-style generator:
//!
//!   * `model0` — reflection-pad-3 + Conv2d(3 → 64, k=7) + InstanceNorm + ReLU
//!   * `model1` — two strided down-convs (64 → 128 → 256), each with
//!     InstanceNorm + ReLU.
//!   * `model2` — nine ResidualBlocks at 256 channels. Each block is
//!     two reflection-pad-1 + Conv2d 3×3 + InstanceNorm pairs with a
//!     ReLU between them, plus the skip add.
//!   * `model3` — two ConvTranspose2d up-blocks (256 → 128 → 64), each
//!     with output_padding=1, InstanceNorm + ReLU.
//!   * `model4` — reflection-pad-3 + Conv2d(64 → 1, k=7) + Sigmoid.
//!
//! ## candle 0.8 substitutions
//!
//! * **ReflectionPad2d → zero padding.** candle 0.8 doesn't expose
//!   `pad_with_reflection`; we use the `Conv2d`'s built-in zero
//!   padding (`padding = 1` for the 3×3 convs, `padding = 3` for the
//!   outer 7×7 convs). Border pixels of the output differ slightly
//!   from the reference; the impact on a downstream ControlNet input
//!   at SD/SDXL resolutions is negligible (border is a few pixels).
//! * **InstanceNorm2d (affine=False) → manual.** PyTorch's
//!   `nn.InstanceNorm2d` with default `affine=False` carries no
//!   weights in the state_dict, so candle's `group_norm` (which
//!   always loads weight + bias) can't be used here. We compute
//!   per-instance per-channel mean/variance over (H, W) and apply
//!   directly via `broadcast_sub` / `broadcast_div`.
//!
//! See [`annotate_lineart`](super::controlnet_annotator::annotate_lineart)
//! for the input/output convention and the post-processing
//! (sigmoid output → invert → resize → 3-channel replicate).

use anyhow::Result;
use candle_core::{D, DType, Module, Tensor};
use candle_nn::{conv2d, conv_transpose2d, Conv2d, Conv2dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, VarBuilder};

/// Per-instance per-channel normalisation over the spatial (H, W)
/// dimensions. Equivalent to PyTorch's `nn.InstanceNorm2d(affine=False)`.
fn instance_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim((D::Minus2, D::Minus1))?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim((D::Minus2, D::Minus1))?;
    let denom = (var + eps)?.sqrt()?;
    Ok(diff.broadcast_div(&denom)?)
}

/// One residual block of the generator: 2× (3×3 conv + InstanceNorm)
/// with a ReLU between, plus the residual skip add.
#[derive(Debug)]
struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualBlock {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        // In the reference Sequential the inner ordering is:
        //   ReflectionPad(1) → Conv2d ─┐    ← conv_block[1]
        //   InstanceNorm                │    ← (no weight)
        //   ReLU                        │
        //   ReflectionPad(1) → Conv2d ──┴──┐ ← conv_block[5]
        //   InstanceNorm
        //
        // So the state_dict keys we care about are
        // `conv_block.1.weight/bias` and `conv_block.5.weight/bias`.
        let conv1 = conv2d(channels, channels, 3, cfg, vb.pp("conv_block.1"))?;
        let conv2 = conv2d(channels, channels, 3, cfg, vb.pp("conv_block.5"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?;
        let h = instance_norm(&h, 1e-5)?;
        let h = h.relu()?;
        let h = self.conv2.forward(&h)?;
        let h = instance_norm(&h, 1e-5)?;
        Ok((x + h)?)
    }
}

/// Lineart generator. Mirrors the `Generator` class from
/// `awacke1/Image-to-Line-Drawings`. Built with `n_residual_blocks = 9`
/// and `sigmoid = true` (the defaults `sk_model.pth` was trained with).
#[derive(Debug)]
pub struct LineartModel {
    // model0
    conv_in: Conv2d,
    // model1 down-convs
    down1: Conv2d,
    down2: Conv2d,
    // model2 residual blocks
    blocks: Vec<ResidualBlock>,
    // model3 up-convs
    up1: ConvTranspose2d,
    up2: ConvTranspose2d,
    // model4 output
    conv_out: Conv2d,
}

impl LineartModel {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // model0[1]: Conv2d(3 → 64, k=7). padding=3 mimics
        // ReflectionPad2d(3) + Conv2d(k=7, padding=0) for the bulk
        // pixels (border values diverge).
        let cfg_p3 = Conv2dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv_in = conv2d(3, 64, 7, cfg_p3, vb.pp("model0.1"))?;

        // model1: two strided down-convs at indices [0] and [3] inside
        // the Sequential — each followed by InstanceNorm + ReLU.
        let cfg_s2_p1 = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let down1 = conv2d(64, 128, 3, cfg_s2_p1, vb.pp("model1.0"))?;
        let down2 = conv2d(128, 256, 3, cfg_s2_p1, vb.pp("model1.3"))?;

        // model2: 9 residual blocks numbered 0..9 in the Sequential.
        let mut blocks = Vec::with_capacity(9);
        for i in 0..9 {
            blocks.push(ResidualBlock::load(vb.pp(&format!("model2.{i}")), 256)?);
        }

        // model3: two ConvTranspose2d up-convs at indices [0] and [3].
        // PyTorch's ConvTranspose2d(k=3, stride=2, padding=1,
        // output_padding=1) matches the upsample factor of 2.
        let cfg_t = ConvTranspose2dConfig {
            padding: 1,
            output_padding: 1,
            stride: 2,
            ..Default::default()
        };
        let up1 = conv_transpose2d(256, 128, 3, cfg_t, vb.pp("model3.0"))?;
        let up2 = conv_transpose2d(128, 64, 3, cfg_t, vb.pp("model3.3"))?;

        // model4[1]: Conv2d(64 → 1, k=7). padding=3 mimics ReflectionPad2d(3).
        let conv_out = conv2d(64, 1, 7, cfg_p3, vb.pp("model4.1"))?;

        Ok(Self {
            conv_in,
            down1,
            down2,
            blocks,
            up1,
            up2,
            conv_out,
        })
    }

    /// Runs the network. Input: `(1, 3, H, W)` f32 in `[0, 1]` (the
    /// reference divides raw `[0, 255]` pixels by 255 before forward).
    /// Output: `(1, 1, H, W)` f32 in `[0, 1]` — line probability map.
    /// Bright pixels = lines.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // model0
        let h = self.conv_in.forward(x)?;
        let h = instance_norm(&h, 1e-5)?;
        let h = h.relu()?;
        // model1
        let h = self.down1.forward(&h)?;
        let h = instance_norm(&h, 1e-5)?;
        let h = h.relu()?;
        let h = self.down2.forward(&h)?;
        let h = instance_norm(&h, 1e-5)?;
        let h = h.relu()?;
        // model2
        let mut h = h;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        // model3
        let h = self.up1.forward(&h)?;
        let h = instance_norm(&h, 1e-5)?;
        let h = h.relu()?;
        let h = self.up2.forward(&h)?;
        let h = instance_norm(&h, 1e-5)?;
        let h = h.relu()?;
        // model4
        let h = self.conv_out.forward(&h)?;
        let h = candle_nn::ops::sigmoid(&h)?;
        Ok(h)
    }
}

/// Force the unused-import warning on `DType` and `VarBuilder` to be
/// satisfied when the module is the only consumer.
#[allow(dead_code)]
fn _force_use(_: DType) {}
