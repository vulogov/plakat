//! HED (Holistically-Nested Edge Detection) softedge model used by
//! ControlNet's `softedge` conditioner.
//!
//! Ported from lllyasviel's `ControlNetHED.pth` (VGG-16 backbone with
//! five side outputs + sigmoid mean for the final edge map):
//!
//!   * `norm` — `(1, 3, 1, 1)` learnable mean subtracted from input.
//!   * Five `DoubleConvBlock`s. Block 1 keeps full resolution; blocks
//!     2–5 each start with a 2×2 max-pool (so spatial dims at the
//!     deepest stage are 1/16 of input). Each block ends with a 1×1
//!     "projection" conv that emits a single-channel edge map at the
//!     block's current spatial resolution.
//!
//! See [`annotate_softedge`](super::controlnet_annotator::annotate_softedge)
//! for the input/output convention and the upsample+sigmoid+resize
//! post-processing.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

/// Two or three 3×3 conv layers (with ReLU between) followed by a 1×1
/// "projection" conv emitting a 1-channel side output.
#[derive(Debug)]
struct DoubleConvBlock {
    convs: Vec<Conv2d>,
    projection: Conv2d,
    down_sampling: bool,
}

impl DoubleConvBlock {
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        n_layers: usize,
        down_sampling: bool,
    ) -> Result<Self> {
        let cfg_p1 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let convs_vb = vb.pp("convs");
        let mut convs = Vec::with_capacity(n_layers);
        // First conv: in_ch → out_ch.
        convs.push(conv2d(in_ch, out_ch, 3, cfg_p1, convs_vb.pp("0"))?);
        // Subsequent convs in the block stay at out_ch.
        for i in 1..n_layers {
            convs.push(conv2d(
                out_ch,
                out_ch,
                3,
                cfg_p1,
                convs_vb.pp(i.to_string()),
            )?);
        }
        // 1×1 projection to single-channel side output. No padding.
        let cfg_p0 = Conv2dConfig::default();
        let projection = conv2d(out_ch, 1, 1, cfg_p0, vb.pp("projection"))?;
        Ok(Self {
            convs,
            projection,
            down_sampling,
        })
    }

    /// Forward through the block. Returns `(features, side_output)` —
    /// `features` feeds the next block, `side_output` is one of the five
    /// edge maps the annotator averages.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut h = if self.down_sampling {
            x.max_pool2d(2)?
        } else {
            x.clone()
        };
        for c in &self.convs {
            h = c.forward(&h)?;
            h = h.relu()?;
        }
        let projection = self.projection.forward(&h)?;
        Ok((h, projection))
    }
}

/// HED net. Loads from a PyTorch `state_dict` (`.pth`) — typically
/// `lllyasviel/Annotators/ControlNetHED.pth`.
#[derive(Debug)]
pub struct HedModel {
    norm: Tensor,
    block1: DoubleConvBlock,
    block2: DoubleConvBlock,
    block3: DoubleConvBlock,
    block4: DoubleConvBlock,
    block5: DoubleConvBlock,
}

impl HedModel {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let norm = vb.get((1, 3, 1, 1), "norm")?;
        let block1 = DoubleConvBlock::load(vb.pp("block1"), 3, 64, 2, false)?;
        let block2 = DoubleConvBlock::load(vb.pp("block2"), 64, 128, 2, true)?;
        let block3 = DoubleConvBlock::load(vb.pp("block3"), 128, 256, 3, true)?;
        let block4 = DoubleConvBlock::load(vb.pp("block4"), 256, 512, 3, true)?;
        let block5 = DoubleConvBlock::load(vb.pp("block5"), 512, 512, 3, true)?;
        Ok(Self {
            norm,
            block1,
            block2,
            block3,
            block4,
            block5,
        })
    }

    /// Runs the network on a `(1, 3, H, W)` input tensor in HED's
    /// preferred input domain (raw pixel values in `[0, 255]`; the
    /// learnt `norm` subtracts the mean).
    ///
    /// Returns the five side outputs in order, each `(1, 1, H', W')`
    /// with `H' = H >> i` and `W' = W >> i` for `i ∈ {0, 1, 2, 3, 4}`.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let h = x.broadcast_sub(&self.norm)?;
        let (h, p1) = self.block1.forward(&h)?;
        let (h, p2) = self.block2.forward(&h)?;
        let (h, p3) = self.block3.forward(&h)?;
        let (h, p4) = self.block4.forward(&h)?;
        let (_h, p5) = self.block5.forward(&h)?;
        Ok(vec![p1, p2, p3, p4, p5])
    }
}
