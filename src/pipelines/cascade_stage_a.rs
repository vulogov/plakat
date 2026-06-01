//! Stable Cascade Stage A VAE — small custom VAE for the
//! image↔latent mapping at 32× compression per axis.
//!
//! v0.37 phase 1: full architectural skeleton + per-block shape
//! verification with random weights. Numerical verification
//! against the real upstream `stabilityai/stable-cascade` Stage A
//! checkpoint happens at v0.37 phase 4 smoke when 3-stage
//! end-to-end inference runs.
//!
//! ## Architecture
//!
//! Stable Cascade's Stage A is the small "Paella VAE" from the
//! Würstchen v3 paper. ~3.6M params total. Continuous latents
//! (not VQ-quantized — the v1/v2 Würstchen design used vector
//! quantization; v3 / Stable Cascade dropped it).
//!
//! ```text
//!   image (B, 3, 1024, 1024)
//!     │
//!     ▼  Encoder
//!     │     in_conv          (3 → 64)
//!     │     down_blocks.{0}  (64 → 128, ×2 downsample)
//!     │     down_blocks.{1}  (128 → 256, ×2 downsample)
//!     │     down_blocks.{2}  (256 → 384, ×2 downsample)
//!     │     down_blocks.{3}  (384 → 512, ×2 downsample)
//!     │     down_blocks.{4}  (512 → 512, ×2 downsample)
//!     │     out_conv         (512 → 4)
//!     │
//!     ▼  latent (B, 4, 32, 32)
//!     │
//!     ▼  Decoder
//!     │     in_conv          (4 → 512)
//!     │     up_blocks.{0}    (512 → 512, ×2 upsample)
//!     │     up_blocks.{1}    (512 → 384, ×2 upsample)
//!     │     up_blocks.{2}    (384 → 256, ×2 upsample)
//!     │     up_blocks.{3}    (256 → 128, ×2 upsample)
//!     │     up_blocks.{4}    (128 → 64,  ×2 upsample)
//!     │     out_conv         (64 → 3)
//!     │
//!     ▼  image (B, 3, 1024, 1024)
//! ```
//!
//! Each `down_block` is a `ResBlock` followed by a strided Conv2d
//! that halves the spatial dims. Each `up_block` is a `ResBlock`
//! followed by a nearest-upsample + Conv2d at the lower resolution.
//!
//! `ResBlock` is the standard SD-style residual block:
//! `GroupNorm → SiLU → Conv2d → GroupNorm → SiLU → Conv2d` + a
//! 1×1 skip conv when channel counts differ.
//!
//! ## Tensor naming
//!
//! Best-effort against the diffusers `stabilityai/stable-cascade`
//! `vqgan/diffusion_pytorch_model.safetensors` layout. The
//! diffusers safetensors uses module-name-prefixed keys like
//! `encoder.down_blocks.{i}.res_block.conv1.weight`. Mismatches
//! against the actual checkpoint will surface at v0.37 phase 4
//! smoke as precise VarBuilder "missing key" errors — fix
//! incrementally then.
//!
//! ## v0.37 phase 1 acceptance
//!
//! - `StageAVae::new(cfg, vb)` compiles + builds with random
//!   weights through a VarMap.
//! - `encoder.forward(image)` produces a `(B, 4, H/32, W/32)`
//!   latent.
//! - `decoder.forward(latent)` produces a `(B, 3, H, W)` image.
//! - Full round-trip preserves shapes through both directions.

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Stage A VAE architectural config.
///
/// The default `paella_v3()` constructor matches Stable Cascade's
/// published Stage A: 5 downsample/upsample stages → 32× compression.
#[derive(Debug, Clone)]
pub struct Config {
    /// Channels at each spatial scale, coarsest-first for the
    /// encoder. Length determines the number of down/up blocks.
    /// Default: `[64, 128, 256, 384, 512, 512]` (5 stages).
    pub channels: Vec<usize>,
    /// Latent channel count (Stage A's bottleneck width). Stable
    /// Cascade uses 4 — matches SD-family VAE channels.
    pub latent_channels: usize,
    /// Number of input/output image channels. Always 3 for RGB.
    pub image_channels: usize,
    /// GroupNorm group count. Standard SD VAE uses 32.
    pub norm_groups: usize,
}

impl Config {
    /// Stable Cascade Stage A (Paella v3 design).
    pub fn paella_v3() -> Self {
        Self {
            channels: vec![64, 128, 256, 384, 512, 512],
            latent_channels: 4,
            image_channels: 3,
            norm_groups: 32,
        }
    }
}

// ---------------------------------------------------------------------
// ResBlock — SD-style residual block reused by both encoder + decoder.
// ---------------------------------------------------------------------

/// Standard SD VAE residual block:
/// `norm → silu → conv1 → norm → silu → conv2 + skip`.
///
/// Tensor keys (relative to the ResBlock's VB prefix):
///   `norm1.{weight,bias}` + `conv1.{weight,bias}`
///   `norm2.{weight,bias}` + `conv2.{weight,bias}`
///   `skip.{weight,bias}`  (only present when in/out channels differ)
pub struct ResBlock {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2d,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2d,
    /// 1×1 conv that matches channel counts when `in != out`.
    /// `None` when the skip is the identity (same channel count).
    skip: Option<nn::Conv2d>,
}

impl ResBlock {
    pub fn new(in_c: usize, out_c: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = nn::group_norm(group_size(groups, in_c), in_c, 1e-6, vb.pp("norm1"))
            .map_err(|e| anyhow!("ResBlock norm1: {e}"))?;
        let conv1 = nn::conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))
            .map_err(|e| anyhow!("ResBlock conv1: {e}"))?;
        let norm2 = nn::group_norm(group_size(groups, out_c), out_c, 1e-6, vb.pp("norm2"))
            .map_err(|e| anyhow!("ResBlock norm2: {e}"))?;
        let conv2 = nn::conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))
            .map_err(|e| anyhow!("ResBlock conv2: {e}"))?;
        let skip = if in_c == out_c {
            None
        } else {
            Some(
                nn::conv2d(in_c, out_c, 1, Default::default(), vb.pp("skip"))
                    .map_err(|e| anyhow!("ResBlock skip: {e}"))?,
            )
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            skip,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;
        let skip = match &self.skip {
            None => x.clone(),
            Some(s) => s.forward(x)?,
        };
        Ok(h.add(&skip)?)
    }
}

/// Pick a GroupNorm group count that divides `channels`. Default 32
/// like SD VAE; fall back to a smaller divisor when needed for tiny
/// test configs (e.g. 64 ch → 32 groups; 8 ch → 4 groups).
///
/// `pub(crate)` so `pipelines::cascade_unet` reuses the same
/// divisor-aware math at its own GroupNorm sites.
pub(crate) fn group_size(default: usize, channels: usize) -> usize {
    if channels % default == 0 {
        default
    } else {
        // Greatest power of 2 that divides channels, capped at default.
        let mut g = default;
        while g > 1 && channels % g != 0 {
            g /= 2;
        }
        g.max(1)
    }
}

// ---------------------------------------------------------------------
// Encoder — image → latent.
// ---------------------------------------------------------------------

/// Encoder block at one resolution: `res_block` (channel transition)
/// then a strided Conv2d that halves spatial dims.
pub struct EncoderDownBlock {
    res_block: ResBlock,
    downsample: nn::Conv2d,
}

impl EncoderDownBlock {
    pub fn new(in_c: usize, out_c: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let res_block = ResBlock::new(in_c, out_c, groups, vb.pp("res_block"))?;
        let downsample = nn::conv2d(
            out_c,
            out_c,
            3,
            nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("downsample"),
        )
        .map_err(|e| anyhow!("EncoderDownBlock downsample: {e}"))?;
        Ok(Self {
            res_block,
            downsample,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.res_block.forward(x)?;
        Ok(self.downsample.forward(&h)?)
    }
}

pub struct Encoder {
    in_conv: nn::Conv2d,
    down_blocks: Vec<EncoderDownBlock>,
    out_norm: nn::GroupNorm,
    out_conv: nn::Conv2d,
}

impl Encoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let first = cfg.channels[0];
        let in_conv = nn::conv2d(
            cfg.image_channels,
            first,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("in_conv"),
        )
        .map_err(|e| anyhow!("Encoder in_conv: {e}"))?;
        // Build down blocks across channel transitions:
        //   ch[0]→ch[1], ch[1]→ch[2], … ch[N-2]→ch[N-1].
        let mut down_blocks = Vec::with_capacity(cfg.channels.len() - 1);
        for i in 0..cfg.channels.len() - 1 {
            let blk = EncoderDownBlock::new(
                cfg.channels[i],
                cfg.channels[i + 1],
                cfg.norm_groups,
                vb.pp("down_blocks").pp(&i.to_string()),
            )?;
            down_blocks.push(blk);
        }
        let last = cfg.channels[cfg.channels.len() - 1];
        let out_norm = nn::group_norm(
            group_size(cfg.norm_groups, last),
            last,
            1e-6,
            vb.pp("out_norm"),
        )
        .map_err(|e| anyhow!("Encoder out_norm: {e}"))?;
        let out_conv = nn::conv2d(
            last,
            cfg.latent_channels,
            1,
            Default::default(),
            vb.pp("out_conv"),
        )
        .map_err(|e| anyhow!("Encoder out_conv: {e}"))?;
        Ok(Self {
            in_conv,
            down_blocks,
            out_norm,
            out_conv,
        })
    }

    /// `image`: `(B, 3, H, W)`. Returns `(B, latent_channels, H/2^N, W/2^N)`
    /// where N = `len(channels) - 1` (number of down blocks).
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let mut x = self.in_conv.forward(image)?;
        for blk in &self.down_blocks {
            x = blk.forward(&x)?;
        }
        let x = self.out_norm.forward(&x)?;
        let x = x.silu()?;
        Ok(self.out_conv.forward(&x)?)
    }
}

// ---------------------------------------------------------------------
// Decoder — latent → image.
// ---------------------------------------------------------------------

/// Decoder block at one resolution: `res_block` (channel transition)
/// then nearest-upsample + Conv2d at the upsampled resolution.
pub struct DecoderUpBlock {
    res_block: ResBlock,
    upsample_conv: nn::Conv2d,
}

impl DecoderUpBlock {
    pub fn new(in_c: usize, out_c: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let res_block = ResBlock::new(in_c, out_c, groups, vb.pp("res_block"))?;
        let upsample_conv = nn::conv2d(
            out_c,
            out_c,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("upsample_conv"),
        )
        .map_err(|e| anyhow!("DecoderUpBlock upsample_conv: {e}"))?;
        Ok(Self {
            res_block,
            upsample_conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.res_block.forward(x)?;
        let (_, _, h_in, w_in) = h.dims4()?;
        // Nearest upsample by 2× then refine with Conv2d.
        let up = h.upsample_nearest2d(h_in * 2, w_in * 2)?;
        Ok(self.upsample_conv.forward(&up)?)
    }
}

pub struct Decoder {
    in_conv: nn::Conv2d,
    up_blocks: Vec<DecoderUpBlock>,
    out_norm: nn::GroupNorm,
    out_conv: nn::Conv2d,
}

impl Decoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Decoder mirrors the encoder. Channel order reversed:
        //   latent_channels → ch[N-1] → ch[N-2] → … → ch[0] → image_channels.
        let last = cfg.channels[cfg.channels.len() - 1];
        let in_conv = nn::conv2d(
            cfg.latent_channels,
            last,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("in_conv"),
        )
        .map_err(|e| anyhow!("Decoder in_conv: {e}"))?;
        // Build up blocks across reversed channel transitions:
        //   ch[N-1]→ch[N-2], ch[N-2]→ch[N-3], … ch[1]→ch[0].
        let mut up_blocks = Vec::with_capacity(cfg.channels.len() - 1);
        for i in (1..cfg.channels.len()).rev() {
            let idx = cfg.channels.len() - 1 - i; // 0-based block index
            let blk = DecoderUpBlock::new(
                cfg.channels[i],
                cfg.channels[i - 1],
                cfg.norm_groups,
                vb.pp("up_blocks").pp(&idx.to_string()),
            )?;
            up_blocks.push(blk);
        }
        let first = cfg.channels[0];
        let out_norm = nn::group_norm(
            group_size(cfg.norm_groups, first),
            first,
            1e-6,
            vb.pp("out_norm"),
        )
        .map_err(|e| anyhow!("Decoder out_norm: {e}"))?;
        let out_conv = nn::conv2d(
            first,
            cfg.image_channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("out_conv"),
        )
        .map_err(|e| anyhow!("Decoder out_conv: {e}"))?;
        Ok(Self {
            in_conv,
            up_blocks,
            out_norm,
            out_conv,
        })
    }

    /// `latent`: `(B, latent_channels, h, w)`. Returns `(B, 3,
    /// h*2^N, w*2^N)` where N = `len(channels) - 1`.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = self.in_conv.forward(latent)?;
        for blk in &self.up_blocks {
            x = blk.forward(&x)?;
        }
        let x = self.out_norm.forward(&x)?;
        let x = x.silu()?;
        Ok(self.out_conv.forward(&x)?)
    }
}

// ---------------------------------------------------------------------
// StageAVae — top-level encoder + decoder.
// ---------------------------------------------------------------------

/// Top-level Stage A VAE. Holds the encoder + decoder; provides
/// `encode` and `decode` methods for the 3-stage pipeline
/// orchestration in v0.37 phase 4.
pub struct StageAVae {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub cfg: Config,
    pub dtype: DType,
    pub device: Device,
}

impl StageAVae {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let encoder = Encoder::new(&cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(&cfg, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            cfg,
            dtype,
            device,
        })
    }

    /// `image`: `(B, 3, H, W)`. Encodes to `(B, 4, H/32, W/32)`
    /// at the default 5-block config.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        self.encoder.forward(image)
    }

    /// `latent`: `(B, 4, h, w)`. Decodes to `(B, 3, h*32, w*32)`
    /// at the default 5-block config.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.decoder.forward(latent)
    }

    /// Round-trip: encode then decode. Useful for sanity tests +
    /// the v0.37 phase 4 smoke (image → latent → image should
    /// approximately reconstruct the input on real weights).
    pub fn round_trip(&self, image: &Tensor) -> Result<Tensor> {
        let latent = self.encode(image)?;
        self.decode(&latent)
    }
}

// Helper to silence "unused" warning when D is imported but only
// used inside future modules.
#[allow(dead_code)]
fn _d_keep_alive() -> D {
    D::Minus1
}

// =====================================================================
// Tests — shape verification with random weights.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// Small test config — 2 down blocks (4× compression instead of
    /// 32×) at smaller channel counts. Lets shape tests run fast
    /// on CPU without instantiating ~3.6M params.
    fn small_cfg() -> Config {
        Config {
            channels: vec![8, 16, 32],
            latent_channels: 4,
            image_channels: 3,
            norm_groups: 8,
        }
    }

    fn random_vae(cfg: Config) -> (StageAVae, VarMap) {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let vae = StageAVae::new(cfg, vb).expect("StageAVae::new");
        (vae, varmap)
    }

    #[test]
    fn resblock_preserves_spatial_dims() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = ResBlock::new(8, 16, 8, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 16, 16), &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 16, 16]);
    }

    #[test]
    fn resblock_identity_skip_when_channels_match() {
        // Same in/out channels → no skip Conv2d allocated.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = ResBlock::new(16, 16, 8, vb).unwrap();
        assert!(block.skip.is_none());
        let x = Tensor::randn(0f32, 1f32, (1, 16, 8, 8), &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 8, 8]);
    }

    #[test]
    fn encoder_compresses_spatial_dims_by_two_per_block() {
        let (vae, _) = random_vae(small_cfg());
        // small_cfg has 2 down blocks → 4× compression.
        let image = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), &vae.device).unwrap();
        let latent = vae.encode(&image).unwrap();
        assert_eq!(latent.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn decoder_expands_spatial_dims_by_two_per_block() {
        let (vae, _) = random_vae(small_cfg());
        let latent = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &vae.device).unwrap();
        let image = vae.decode(&latent).unwrap();
        assert_eq!(image.dims(), &[1, 3, 32, 32]);
    }

    #[test]
    fn round_trip_returns_original_spatial_shape() {
        let (vae, _) = random_vae(small_cfg());
        let image = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), &vae.device).unwrap();
        let restored = vae.round_trip(&image).unwrap();
        assert_eq!(restored.dims(), image.dims());
    }

    #[test]
    fn paella_v3_config_has_five_down_blocks() {
        let cfg = Config::paella_v3();
        // 6 channel levels → 5 down blocks → 32× compression.
        assert_eq!(cfg.channels.len(), 6);
        assert_eq!(cfg.channels.len() - 1, 5);
        assert_eq!(cfg.latent_channels, 4);
        assert_eq!(cfg.image_channels, 3);
        assert_eq!(cfg.norm_groups, 32);
    }

    #[test]
    fn group_size_falls_back_to_smaller_divisor() {
        // 64 channels → 32 groups (default).
        assert_eq!(group_size(32, 64), 32);
        // 8 channels → 8 groups (default 32 doesn't divide; falls
        // back to 8 since 8 is the largest power-of-2 divisor ≤ 32).
        assert_eq!(group_size(32, 8), 8);
        // 24 channels → falls back to 8 (24 = 8 × 3; power-of-2 ≤ 32).
        assert_eq!(group_size(32, 24), 8);
        // 7 channels → falls back to 1.
        assert_eq!(group_size(32, 7), 1);
    }

    #[test]
    fn encoder_full_paella_compresses_by_32x_at_small_input() {
        // Use the full paella_v3 config but at smaller input so the
        // ~3.6M-param model instantiation stays cheap.
        let (vae, _) = random_vae(Config::paella_v3());
        // (1, 3, 64, 64) → 5 down blocks → (1, 4, 2, 2).
        let image = Tensor::randn(0f32, 1f32, (1, 3, 64, 64), &vae.device).unwrap();
        let latent = vae.encode(&image).unwrap();
        assert_eq!(latent.dims(), &[1, 4, 2, 2]);
        let restored = vae.decode(&latent).unwrap();
        assert_eq!(restored.dims(), &[1, 3, 64, 64]);
    }
}
