// Phase 4 scaffolding — nothing in the crate calls these yet. The
// `dead_code` allowance lifts when Phase 4b wires the encoder into
// `IdentityKind::FaceId`.
#![allow(dead_code)]

//! Face-identity model porting for Phase 4 (FaceID).
//!
//! Today's checkpoint: the **InsightFace IR-ResNet50** ArcFace backbone
//! and the **IP-Adapter-FaceID image projection** ported to candle. Both
//! compile and load from a `VarBuilder`. The end-to-end FaceID pipeline
//! still needs:
//!   1. **SCRFD face detector** — finds faces in an arbitrary photo.
//!   2. **5-landmark similarity-transform alignment** — produces the
//!      canonical 112×112 RGB crop ArcFace expects.
//!   3. **`IdentityEncoder` wiring** — `IdentityKind::FaceId` variant,
//!      `FromStr` arm, `load_encoder` arm, weight-download paths.
//!
//! Those three land in Phase 4b. Until then this module is **not invoked
//! from anywhere** — it sits compiled-but-unreachable until the detector
//! can feed it aligned crops.
//!
//! ## IR-ResNet50 architecture
//!
//! From Duta et al., "Improved Residual Networks for Image and Video
//! Recognition" (2020). Layer counts `[3, 4, 14, 3]` match InsightFace's
//! `iresnet50()` and the `w600k_r50` weights bundled with `antelopev2`
//! and `buffalo_l` — the most common ArcFace deployments.
//!
//! Differences from candle-transformers' stock ResNet:
//!   * **Pre-activation block**: `bn → conv → bn → prelu → conv → bn`,
//!     then add the (optionally downsampled) shortcut. No final activation
//!     after the residual sum.
//!   * **PReLU** activation (per-channel learnable negative slope), not
//!     ReLU. candle-nn has no PReLU module so this file implements it.
//!   * **Embedding head**: `bn → flatten → fc → bn1d`, then L2-normalise.
//!     No global pooling.
//!
//! Input contract:
//!   * shape `(B, 3, 112, 112)`, RGB
//!   * normalised to roughly `[-1, 1]` (InsightFace uses `(x - 127.5) / 127.5`)
//!   * **must be face-aligned** — ArcFace's training distribution is
//!     5-point landmark-aligned crops; unaligned input drops embedding
//!     quality ~30%. Phase 4b adds the aligner.
//!
//! Output: `(B, 512)` unit-norm face embedding.

use anyhow::Result;
use candle_core::{Module, ModuleT, Tensor};
use candle_nn::{BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, VarBuilder};

/// Inference flag for `ModuleT::forward_t`. Centralised so all the
/// per-BatchNorm call sites read the same.
const EVAL: bool = false;

// =====================================================================
// PReLU — candle-nn ships no PReLU module, so we build one.
// =====================================================================

/// PReLU(num_parameters): per-channel learnable slope on the negative half.
/// `forward(x) = max(0, x) + weight · min(0, x)`, broadcast across spatial
/// dims for 4D inputs.
struct PRelu {
    /// Shape `(num_parameters,)`. Reshaped to `(1, C, 1, 1)` at forward
    /// time for broadcasting over batch + spatial dims.
    weight: Tensor,
}

impl PRelu {
    fn new(vs: VarBuilder, num_parameters: usize) -> Result<Self> {
        let weight = vs.get(num_parameters, "weight")?;
        Ok(Self { weight })
    }

    /// 4D-only — IR-ResNet50 doesn't apply PReLU to the 2D embedding head.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, c, _h, _w) = x.dims4()?;
        let w = self.weight.reshape((1, c, 1, 1))?;
        let zero = x.zeros_like()?;
        let pos = x.maximum(&zero)?;
        let neg = x.minimum(&zero)?;
        let scaled_neg = neg.broadcast_mul(&w)?;
        Ok((pos + scaled_neg)?)
    }
}

// =====================================================================
// IBasicBlock — pre-activation residual block.
//
//   identity = x          (or downsample(x) when in/out shapes differ)
//   out = bn3( conv2( prelu( bn2( conv1( bn1(x) ) ) ) ) )
//   out = out + identity
// =====================================================================

struct IBasicBlock {
    bn1: BatchNorm,
    conv1: Conv2d,
    bn2: BatchNorm,
    prelu: PRelu,
    conv2: Conv2d,
    bn3: BatchNorm,
    /// `Some` when in/out channels differ OR stride > 1.
    /// Stored as `(conv 1×1, bn)` matching PyTorch's
    /// `nn.Sequential(conv, bn)` keying (`downsample.0`, `downsample.1`).
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl IBasicBlock {
    fn new(
        vs: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let bn_cfg = BatchNormConfig::default();
        let bn1 = candle_nn::batch_norm(in_channels, bn_cfg, vs.pp("bn1"))?;
        let conv1_cfg = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d_no_bias(
            in_channels,
            out_channels,
            3,
            conv1_cfg,
            vs.pp("conv1"),
        )?;
        let bn2 = candle_nn::batch_norm(out_channels, bn_cfg, vs.pp("bn2"))?;
        let prelu = PRelu::new(vs.pp("prelu"), out_channels)?;
        let conv2_cfg = Conv2dConfig {
            stride,
            padding: 1,
            ..Default::default()
        };
        let conv2 = candle_nn::conv2d_no_bias(
            out_channels,
            out_channels,
            3,
            conv2_cfg,
            vs.pp("conv2"),
        )?;
        let bn3 = candle_nn::batch_norm(out_channels, bn_cfg, vs.pp("bn3"))?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let cfg = Conv2dConfig {
                stride,
                padding: 0,
                ..Default::default()
            };
            let conv = candle_nn::conv2d_no_bias(
                in_channels,
                out_channels,
                1,
                cfg,
                vs.pp("downsample").pp("0"),
            )?;
            let bn = candle_nn::batch_norm(out_channels, bn_cfg, vs.pp("downsample").pp("1"))?;
            Some((conv, bn))
        } else {
            None
        };

        Ok(Self {
            bn1,
            conv1,
            bn2,
            prelu,
            conv2,
            bn3,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some((conv, bn)) => bn.forward_t(&conv.forward(x)?, EVAL)?,
            None => x.clone(),
        };
        let h = self.bn1.forward_t(x, EVAL)?;
        let h = self.conv1.forward(&h)?;
        let h = self.bn2.forward_t(&h, EVAL)?;
        let h = self.prelu.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = self.bn3.forward_t(&h, EVAL)?;
        Ok((h + identity)?)
    }
}

// =====================================================================
// IR-ResNet50 — the ArcFace backbone.
// =====================================================================

/// InsightFace IR-ResNet50, layer counts `[3, 4, 14, 3]`. Produces a
/// 512-d L2-normalised face embedding from a 112×112 RGB face crop.
pub struct IResnet50 {
    conv1: Conv2d,
    bn1: BatchNorm,
    prelu: PRelu,
    layer1: Vec<IBasicBlock>,
    layer2: Vec<IBasicBlock>,
    layer3: Vec<IBasicBlock>,
    layer4: Vec<IBasicBlock>,
    bn2: BatchNorm,
    fc: Linear,
    features: BatchNorm,
}

impl IResnet50 {
    /// Build from a `VarBuilder` rooted at an IR-ResNet50 PyTorch state
    /// dict (safetensors). Expected key layout:
    ///   conv1.weight, bn1.{weight,bias,running_mean,running_var}, prelu.weight,
    ///   layer{1..4}.<i>.{bn1,conv1,bn2,prelu,conv2,bn3}.<…>,
    ///   layer<X>.0.downsample.{0,1}.<…>   (when stride > 1 or channels change),
    ///   bn2.<…>, fc.{weight,bias}, features.<…>
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let bn_cfg = BatchNormConfig::default();
        let conv1_cfg = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d_no_bias(3, 64, 3, conv1_cfg, vs.pp("conv1"))?;
        let bn1 = candle_nn::batch_norm(64, bn_cfg, vs.pp("bn1"))?;
        let prelu = PRelu::new(vs.pp("prelu"), 64)?;

        // All four layers downsample (stride 2 on the first block);
        // channel widths double each stage: 64 → 128 → 256 → 512.
        // Block counts [3, 4, 14, 3] are InsightFace's iresnet50.
        let layer1 = make_layer(vs.pp("layer1"), 64, 64, 3, 2)?;
        let layer2 = make_layer(vs.pp("layer2"), 64, 128, 4, 2)?;
        let layer3 = make_layer(vs.pp("layer3"), 128, 256, 14, 2)?;
        let layer4 = make_layer(vs.pp("layer4"), 256, 512, 3, 2)?;

        let bn2 = candle_nn::batch_norm(512, bn_cfg, vs.pp("bn2"))?;
        // 7×7 = (112 / 2^4). Each of the four layers halves spatial dims.
        let fc = candle_nn::linear(512 * 7 * 7, 512, vs.pp("fc"))?;
        let features = candle_nn::batch_norm(512, bn_cfg, vs.pp("features"))?;

        Ok(Self {
            conv1,
            bn1,
            prelu,
            layer1,
            layer2,
            layer3,
            layer4,
            bn2,
            fc,
            features,
        })
    }

    /// Forward pass. `x: (B, 3, 112, 112)` → `(B, 512)` unit-norm.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward_t(&x, EVAL)?;
        let mut x = self.prelu.forward(&x)?;

        for block in &self.layer1 {
            x = block.forward(&x)?;
        }
        for block in &self.layer2 {
            x = block.forward(&x)?;
        }
        for block in &self.layer3 {
            x = block.forward(&x)?;
        }
        for block in &self.layer4 {
            x = block.forward(&x)?;
        }

        let x = self.bn2.forward_t(&x, EVAL)?;
        // Drop dropout — inference-only path.
        let (b, c, h, w) = x.dims4()?;
        let x = x.reshape((b, c * h * w))?;
        let x = self.fc.forward(&x)?;
        // candle-nn's BatchNorm treats dim 1 as channels for any rank,
        // so a 2D `(B, 512)` input is the BN1d path automatically.
        let x = self.features.forward_t(&x, EVAL)?;

        // L2-normalise along the embedding dim. ArcFace embeddings are
        // unit-norm by construction; downstream cosine-sim works only
        // when both sides are normalised.
        let norm_sq = x.sqr()?.sum_keepdim(1)?;
        let norm = norm_sq.sqrt()?;
        // Tiny epsilon prevents 0/0 if a (very degenerate) input gives
        // an all-zero embedding.
        let safe_norm = (norm + 1e-12_f64)?;
        Ok(x.broadcast_div(&safe_norm)?)
    }
}

fn make_layer(
    vs: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    blocks: usize,
    stride: usize,
) -> Result<Vec<IBasicBlock>> {
    let mut layers = Vec::with_capacity(blocks);
    layers.push(IBasicBlock::new(vs.pp("0"), in_ch, out_ch, stride)?);
    for i in 1..blocks {
        layers.push(IBasicBlock::new(vs.pp(i.to_string()), out_ch, out_ch, 1)?);
    }
    Ok(layers)
}

// =====================================================================
// FaceIdEncoder — Phase 4 scaffolding.
//
// Combines IR-ResNet50 (this file) with `ImageProj` (existing IP-Adapter
// projection from `ip_adapter.rs`) — exactly the shape FaceID needs:
//     ArcFace(112×112×3) → 512-d → ImageProj(512 → 4 × cross_attn_dim)
//
// **Not yet wired into `IdentityEncoder`** because we have no way to
// produce the aligned 112×112 input from an arbitrary photo. Phase 4b
// adds SCRFD detection + 5-landmark alignment and the trait impl.
// =====================================================================

/// Combined ArcFace + FaceID image-proj encoder. Once Phase 4b lands the
/// face detector, this gets an `encode(photo_path)` method and an
/// `IdentityEncoder` trait impl.
pub struct FaceIdEncoder {
    arcface: IResnet50,
    image_proj: crate::pipelines::ip_adapter::ImageProj,
    #[allow(dead_code)]
    device: candle_core::Device,
    #[allow(dead_code)]
    dtype: candle_core::DType,
}

impl FaceIdEncoder {
    /// Load ArcFace + FaceID image-proj weights.
    ///
    /// * `arcface_weights` — IR-ResNet50 safetensors. Most accessible
    ///   source: HF-hosted conversion of InsightFace's `w600k_r50.onnx`
    ///   (antelopev2 / buffalo_l bundle). Phase 4b wires the download.
    /// * `faceid_weights` — `h94/IP-Adapter/models/ip-adapter-faceid_sd15`
    ///   (the `image_proj.*` subtree). The same file also contains LoRA
    ///   weights for the UNet's cross-attention; those are NOT applied
    ///   here. Loading just the image_proj part is consistent with our
    ///   existing Plus-Face integration (which similarly skips decoupled
    ///   cross-attention).
    /// * `cross_attn_dim` — 768 for SD 1.5, 2048 for SDXL.
    pub fn load(
        arcface_weights: &std::path::Path,
        faceid_weights: &std::path::Path,
        cross_attn_dim: usize,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[arcface_weights], dtype, device)?
        };
        let arcface = IResnet50::new(vb)?;
        let image_proj = crate::pipelines::ip_adapter::ImageProj::load(
            faceid_weights,
            512,
            cross_attn_dim,
            4,
            device,
            dtype,
        )?;
        Ok(Self {
            arcface,
            image_proj,
            device: device.clone(),
            dtype,
        })
    }

    /// Given an aligned 112×112 RGB crop (pre-normalised to roughly
    /// `[-1, 1]`), produce `(1, 4, cross_attn_dim)` identity tokens
    /// ready for concatenation onto the text-token sequence.
    ///
    /// This is the only encode entry-point today. Phase 4b adds
    /// `encode(photo_path)` which detects + aligns first.
    #[allow(dead_code)]
    pub fn encode_aligned(&self, aligned: &Tensor) -> Result<Tensor> {
        let embedding = self.arcface.forward(aligned)?;
        self.image_proj.forward(&embedding)
    }
}
