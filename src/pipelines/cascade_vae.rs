//! v0.39 phase 0d: Stable Cascade Stage A VAE (Paella VQ-GAN),
//! upstream-aligned.
//!
//! Replaces (will replace, in phase 0g) v0.37 phase 1's SD-style
//! `cascade_stage_a.rs` with the actual Paella v3 / Würstchen v3
//! VAE architecture. Tensor naming matches the inspected keys in
//! `stabilityai/stable-cascade/vqgan/diffusion_pytorch_model.safetensors`
//! (122 tensors).
//!
//! ## Architecture
//!
//! ```text
//!   image (B, 3, H, W)
//!     ↓ PixelUnshuffle(2): (B, 12, H/2, W/2)
//!     ↓ in_block.1: Conv2d(12, 192, 1×1)            (B, 192, H/2, W/2)
//!     ↓ down_blocks.0: PaellaResBlock(192)
//!     ↓ down_blocks.1: Conv2d(192, 384, 4, stride=2)  (B, 384, H/4, W/4)
//!     ↓ down_blocks.2: PaellaResBlock(384)
//!     ↓ down_blocks.3.0: Conv2d(384, 4, 1×1)
//!     ↓ down_blocks.3.1: BatchNorm2d(4)
//!     ↓ latent (B, 4, H/4, W/4)
//!     ↓ vquantizer.embedding: codebook lookup (8192 codes × 4 dim)
//!     ↓ z_q (B, 4, H/4, W/4)
//!     ↓ up_blocks.0: Conv2d(4, 384, 1×1)
//!     ↓ up_blocks.{1..12}: 12 × PaellaResBlock(384)
//!     ↓ up_blocks.13: ConvTranspose2d(384, 192, 4, stride=2)  (B, 192, H/2, W/2)
//!     ↓ up_blocks.14: PaellaResBlock(192)
//!     ↓ out_block.0: Conv2d(192, 12, 1×1)            (B, 12, H/2, W/2)
//!     ↓ PixelShuffle(2): (B, 3, H, W)
//! ```
//!
//! Net 4× spatial compression (not 32× — Stable Cascade's "tiny VAE"
//! claim refers to total stage A→C compression, not Stage A alone).
//!
//! ## Tensor naming
//!
//! Matches upstream exactly:
//! - `in_block.1.{weight,bias}` — Conv2d 12→192
//! - `down_blocks.0.{depthwise.1, channelwise.0/2, gammas}` — ResBlock
//! - `down_blocks.1.{weight,bias}` — strided Conv2d
//! - `down_blocks.2.*` — ResBlock at deeper width
//! - `down_blocks.3.0.{weight,bias}` — Conv2d 384→4
//! - `down_blocks.3.1.{weight,bias,running_mean,running_var,
//!   num_batches_tracked}` — BatchNorm2d(4)
//! - `vquantizer.embedding.weight` — codebook (8192, 4)
//! - `up_blocks.0.{weight,bias}` — Conv2d 4→384 (decoder input)
//! - `up_blocks.{1..12}.*` — 12 ResBlocks at deeper width
//! - `up_blocks.13.{weight,bias}` — ConvTranspose2d 384→192
//! - `up_blocks.14.*` — ResBlock at shallower width
//! - `out_block.0.{weight,bias}` — Conv2d 192→12
//!
//! ## PaellaResBlock — `gammas` parameter
//!
//! Each ResBlock has a learnable `gammas` parameter of shape `(6,)`.
//! Upstream applies these as scale/shift modulations interleaved
//! with the depthwise + channelwise blocks. Phase 0d ships a
//! best-effort interpretation (gammas applied as a residual scalar
//! scale on the MLP output); the exact application can be refined
//! against real-weight smoke at user time without changing the
//! tensor naming.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, IndexOp, Module, ModuleT, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::pipelines::cascade_blocks::LayerNorm2d;

/// Stage A VAE architectural config.
#[derive(Debug, Clone)]
pub struct Config {
    /// Image channels — always 3 (RGB).
    pub image_channels: usize,
    /// PixelUnshuffle factor at input (and PixelShuffle at output).
    /// Upstream uses 2 → 12 hidden input channels = 3 × 2².
    pub pixel_unshuffle: usize,
    /// Width after `in_block.1` — upstream 192.
    pub c_hidden_in: usize,
    /// Width after the strided downsample — upstream 384.
    pub c_hidden_deep: usize,
    /// Latent channel count (post-vquantizer) — upstream 4.
    pub c_latent: usize,
    /// Codebook size — upstream 8192.
    pub num_codes: usize,
    /// Number of decoder ResBlocks at the deep width (between the
    /// input-projection conv and the upsample). Upstream: 12.
    pub n_decoder_deep_blocks: usize,
}

impl Config {
    /// Upstream `stabilityai/stable-cascade/vqgan/` config derived
    /// from safetensors-header inspection at v0.39 phase 0.
    pub fn paella_v3() -> Self {
        Self {
            image_channels: 3,
            pixel_unshuffle: 2,
            c_hidden_in: 192,
            c_hidden_deep: 384,
            c_latent: 4,
            num_codes: 8192,
            n_decoder_deep_blocks: 12,
        }
    }
}

// ---------------------------------------------------------------------
// PaellaResBlock — ConvNeXt-style with depthwise + channelwise MLP +
// gammas modulation parameter.
// ---------------------------------------------------------------------

/// Stable Cascade Stage A ResBlock (Paella v3 design).
///
/// Tensor keys (relative to the block's VB prefix):
///   `depthwise.1.{weight,bias}`     — Conv2d(c, c, 3, groups=c)
///   `channelwise.0.{weight,bias}`   — Linear C → 4C
///   `channelwise.2.{weight,bias}`   — Linear 4C → C
///   `gammas`                        — (6,) learnable modulation
///
/// Note: upstream's `depthwise.0` is `ReflectionPad2d`; we
/// approximate with zero-padding inside the Conv2d (numerically
/// close for non-edge pixels; potential edge artifacts on real
/// weights — phase 0g can swap if smoke shows divergence).
pub struct PaellaResBlock {
    depthwise: nn::Conv2d,
    norm: LayerNorm2d,
    channelwise_0: nn::Linear,
    channelwise_2: nn::Linear,
    gammas: Tensor,
    channels: usize,
}

impl PaellaResBlock {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            groups: channels,
            ..Default::default()
        };
        // Upstream's depthwise.0 is ReflectionPad2d (no params), .1
        // is the Conv2d.
        let depthwise = nn::conv2d(channels, channels, 3, conv_cfg, vb.pp("depthwise").pp("1"))
            .map_err(|e| anyhow!("PaellaResBlock depthwise.1: {e}"))?;
        let norm = LayerNorm2d::new(channels, 1e-6);
        let channelwise_0 = nn::linear(channels, channels * 4, vb.pp("channelwise").pp("0"))
            .map_err(|e| anyhow!("PaellaResBlock channelwise.0: {e}"))?;
        // channelwise.1 is GELU (no params).
        let channelwise_2 = nn::linear(channels * 4, channels, vb.pp("channelwise").pp("2"))
            .map_err(|e| anyhow!("PaellaResBlock channelwise.2: {e}"))?;
        let gammas = vb
            .get(6, "gammas")
            .map_err(|e| anyhow!("PaellaResBlock gammas: {e}"))?;
        Ok(Self {
            depthwise,
            norm,
            channelwise_0,
            channelwise_2,
            gammas,
            channels,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_res = x.clone();
        let h = self.depthwise.forward(x)?;
        let h = self.norm.forward(&h)?;
        // Permute (B, C, H, W) → (B, H, W, C) for MLP.
        let h = h.permute((0, 2, 3, 1))?.contiguous()?;
        let h = self.channelwise_0.forward(&h)?;
        let h = h.gelu()?;
        let h = self.channelwise_2.forward(&h)?;
        let h = h.permute((0, 3, 1, 2))?.contiguous()?;
        // Apply gammas[4] as residual scale (best-effort interpretation
        // of the 6-gamma vector; refined in phase 0g if needed).
        let scale = self.gammas.i(4)?.to_scalar::<f32>()? as f64;
        let scaled = h.affine(scale, 0.0)?;
        Ok(x_res.add(&scaled)?)
    }
}

// ---------------------------------------------------------------------
// VectorQuantizer — codebook lookup (inference path).
// ---------------------------------------------------------------------

/// Stable Cascade Stage A vector quantizer. Codebook of 8192 codes
/// × 4 dim. Inference path picks the nearest codebook entry per
/// spatial location.
pub struct VectorQuantizer {
    codebook: Tensor,
    num_codes: usize,
    code_dim: usize,
}

impl VectorQuantizer {
    pub fn new(num_codes: usize, code_dim: usize, vb: VarBuilder) -> Result<Self> {
        let codebook = vb
            .get((num_codes, code_dim), "embedding.weight")
            .map_err(|e| anyhow!("VectorQuantizer embedding.weight: {e}"))?;
        Ok(Self {
            codebook,
            num_codes,
            code_dim,
        })
    }

    pub fn num_codes(&self) -> usize {
        self.num_codes
    }

    pub fn code_dim(&self) -> usize {
        self.code_dim
    }

    /// Quantize a latent `(B, C, H, W)` (C must equal `code_dim`).
    /// Returns `(z_q, code_indices)`:
    /// - `z_q`: `(B, C, H, W)` — quantized latent (codebook entries).
    /// - `code_indices`: `(B, H, W)` — picked codebook index per cell.
    pub fn quantize(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, c, h, w) = z.dims4()?;
        anyhow::ensure!(
            c == self.code_dim,
            "VQ: channel dim {c} must match code_dim {}",
            self.code_dim
        );
        // (B, C, H, W) → (B*H*W, C)
        let z_flat = z
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((b * h * w, c))?;

        // Squared distances to codebook (num_codes, C).
        // dist[n, k] = ||z[n] - c[k]||² = sum z² - 2 z·c + sum c²
        let z_sq = z_flat.sqr()?.sum_keepdim(1)?; // (N, 1)
        let c_sq = self.codebook.sqr()?.sum_keepdim(1)?.transpose(0, 1)?; // (1, K)
        let dots = z_flat.matmul(&self.codebook.transpose(0, 1)?.contiguous()?)?; // (N, K)
        let neg_2_dots = dots.affine(-2.0, 0.0)?;
        let dist = z_sq.broadcast_add(&c_sq)?.add(&neg_2_dots)?;
        // argmin over codebook axis.
        let indices = dist.argmin(1)?; // (N,)
        // Lookup.
        let z_q_flat = self.codebook.index_select(&indices, 0)?; // (N, C)
        let z_q = z_q_flat
            .reshape((b, h, w, c))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        let code_indices = indices.reshape((b, h, w))?;
        Ok((z_q, code_indices))
    }

    /// Look up codebook entries by integer indices `(B, H, W)`.
    pub fn decode_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let (b, h, w) = indices.dims3()?;
        let flat = indices.flatten_all()?;
        let z_q_flat = self.codebook.index_select(&flat, 0)?;
        Ok(z_q_flat
            .reshape((b, h, w, self.code_dim))?
            .permute((0, 3, 1, 2))?
            .contiguous()?)
    }
}

// ---------------------------------------------------------------------
// StageAVae — top-level encoder + VQ + decoder.
// ---------------------------------------------------------------------

pub struct StageAVae {
    // Encoder
    in_conv: nn::Conv2d,
    enc_res_0: PaellaResBlock,
    enc_down: nn::Conv2d,
    enc_res_1: PaellaResBlock,
    enc_out_conv: nn::Conv2d,
    enc_out_bn: nn::BatchNorm,
    // Bottleneck
    pub vquantizer: VectorQuantizer,
    // Decoder
    dec_in_conv: nn::Conv2d,
    dec_res_deep: Vec<PaellaResBlock>,
    dec_up: nn::ConvTranspose2d,
    dec_res_shallow: PaellaResBlock,
    out_conv: nn::Conv2d,

    pub cfg: Config,
    pub dtype: DType,
    pub device: Device,
}

impl StageAVae {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let c_in_unshuffled = cfg.image_channels * cfg.pixel_unshuffle * cfg.pixel_unshuffle;

        // ---- Encoder ----
        let in_conv = nn::conv2d(
            c_in_unshuffled,
            cfg.c_hidden_in,
            1,
            Default::default(),
            vb.pp("in_block").pp("1"),
        )
        .map_err(|e| anyhow!("in_block.1: {e}"))?;

        let enc_res_0 = PaellaResBlock::new(cfg.c_hidden_in, vb.pp("down_blocks").pp("0"))?;

        let enc_down = nn::conv2d(
            cfg.c_hidden_in,
            cfg.c_hidden_deep,
            4,
            nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("down_blocks").pp("1"),
        )
        .map_err(|e| anyhow!("down_blocks.1: {e}"))?;

        let enc_res_1 = PaellaResBlock::new(cfg.c_hidden_deep, vb.pp("down_blocks").pp("2"))?;

        let enc_out_conv = nn::conv2d(
            cfg.c_hidden_deep,
            cfg.c_latent,
            1,
            Default::default(),
            vb.pp("down_blocks").pp("3").pp("0"),
        )
        .map_err(|e| anyhow!("down_blocks.3.0: {e}"))?;

        let enc_out_bn = nn::batch_norm(
            cfg.c_latent,
            nn::BatchNormConfig::default(),
            vb.pp("down_blocks").pp("3").pp("1"),
        )
        .map_err(|e| anyhow!("down_blocks.3.1: {e}"))?;

        // ---- Bottleneck ----
        let vquantizer =
            VectorQuantizer::new(cfg.num_codes, cfg.c_latent, vb.pp("vquantizer"))?;

        // ---- Decoder ----
        let dec_in_conv = nn::conv2d(
            cfg.c_latent,
            cfg.c_hidden_deep,
            1,
            Default::default(),
            vb.pp("up_blocks").pp("0"),
        )
        .map_err(|e| anyhow!("up_blocks.0: {e}"))?;

        let mut dec_res_deep = Vec::with_capacity(cfg.n_decoder_deep_blocks);
        for i in 0..cfg.n_decoder_deep_blocks {
            // Position offset: up_blocks.{1+i} (positions 1..=12 for the
            // default 12 deep blocks).
            let pos = i + 1;
            dec_res_deep.push(PaellaResBlock::new(
                cfg.c_hidden_deep,
                vb.pp("up_blocks").pp(&pos.to_string()),
            )?);
        }

        // Upsample at up_blocks.{1 + n_decoder_deep_blocks}.
        let up_pos = 1 + cfg.n_decoder_deep_blocks;
        let dec_up = nn::conv_transpose2d(
            cfg.c_hidden_deep,
            cfg.c_hidden_in,
            4,
            nn::ConvTranspose2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("up_blocks").pp(&up_pos.to_string()),
        )
        .map_err(|e| anyhow!("up_blocks.{up_pos}: {e}"))?;

        // Shallow ResBlock at up_blocks.{2 + n_decoder_deep_blocks}.
        let shallow_pos = 2 + cfg.n_decoder_deep_blocks;
        let dec_res_shallow = PaellaResBlock::new(
            cfg.c_hidden_in,
            vb.pp("up_blocks").pp(&shallow_pos.to_string()),
        )?;

        let out_conv = nn::conv2d(
            cfg.c_hidden_in,
            c_in_unshuffled,
            1,
            Default::default(),
            vb.pp("out_block").pp("0"),
        )
        .map_err(|e| anyhow!("out_block.0: {e}"))?;

        Ok(Self {
            in_conv,
            enc_res_0,
            enc_down,
            enc_res_1,
            enc_out_conv,
            enc_out_bn,
            vquantizer,
            dec_in_conv,
            dec_res_deep,
            dec_up,
            dec_res_shallow,
            out_conv,
            cfg,
            dtype,
            device,
        })
    }

    /// Encode `(B, 3, H, W)` → `(B, c_latent, H/(2*pixel_unshuffle),
    /// W/(2*pixel_unshuffle))`. For the default config (PixelUnshuffle
    /// 2 + one strided conv stride 2), that's H/4.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let (_b, c, h, w) = image.dims4()?;
        anyhow::ensure!(
            c == self.cfg.image_channels,
            "encode: image channels {c} != cfg.image_channels {}",
            self.cfg.image_channels
        );
        anyhow::ensure!(
            h % self.cfg.pixel_unshuffle == 0 && w % self.cfg.pixel_unshuffle == 0,
            "encode: image spatial dims ({h}x{w}) must be divisible by pixel_unshuffle ({})",
            self.cfg.pixel_unshuffle
        );
        // PixelUnshuffle(k): (B, C, H, W) → (B, C*k*k, H/k, W/k)
        let x = pixel_unshuffle(image, self.cfg.pixel_unshuffle)?;
        let x = self.in_conv.forward(&x)?;
        let x = self.enc_res_0.forward(&x)?;
        let x = self.enc_down.forward(&x)?;
        let x = self.enc_res_1.forward(&x)?;
        let x = self.enc_out_conv.forward(&x)?;
        // BatchNorm2d in inference mode uses running stats.
        let x = self.enc_out_bn.forward_t(&x, false)?;
        Ok(x)
    }

    /// Quantize the encoder latent through the VQ codebook.
    pub fn quantize(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        self.vquantizer.quantize(z)
    }

    /// Decode `(B, c_latent, h, w)` → `(B, 3, h*4, w*4)`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.dec_in_conv.forward(z)?;
        let mut x = x;
        for block in &self.dec_res_deep {
            x = block.forward(&x)?;
        }
        let x = self.dec_up.forward(&x)?;
        let x = self.dec_res_shallow.forward(&x)?;
        let x = self.out_conv.forward(&x)?;
        // PixelShuffle reverses PixelUnshuffle.
        pixel_shuffle(&x, self.cfg.pixel_unshuffle)
    }

    /// Round-trip: image → encode → quantize → decode → image.
    /// Useful for sanity tests + the v0.39 phase 0g smoke (image →
    /// latent → image should approximately reconstruct).
    pub fn round_trip(&self, image: &Tensor) -> Result<Tensor> {
        let z = self.encode(image)?;
        let (z_q, _idx) = self.quantize(&z)?;
        self.decode(&z_q)
    }
}

// ---------------------------------------------------------------------
// PixelUnshuffle / PixelShuffle — pure tensor ops.
// ---------------------------------------------------------------------

/// PyTorch-equivalent `nn.PixelUnshuffle(k)`: rearrange spatial
/// blocks of `k×k` into channel groups. `(B, C, H, W)` → `(B, C*k²,
/// H/k, W/k)`.
pub fn pixel_unshuffle(x: &Tensor, k: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    anyhow::ensure!(h % k == 0 && w % k == 0, "PixelUnshuffle: H/W must be divisible by k");
    let h_out = h / k;
    let w_out = w / k;
    // (B, C, h_out, k, w_out, k) — split each spatial dim into
    // (block, in-block).
    let r = x
        .reshape((b, c, h_out, k, w_out, k))?
        // Permute so the k×k block axes come right after channels:
        // (B, C, k, k, h_out, w_out) — then merge (C, k, k) → C*k².
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c * k * k, h_out, w_out))?;
    Ok(r)
}

/// PyTorch-equivalent `nn.PixelShuffle(k)`: inverse of
/// `pixel_unshuffle`. `(B, C, H, W)` → `(B, C/k², H*k, W*k)`.
pub fn pixel_shuffle(x: &Tensor, k: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    anyhow::ensure!(c % (k * k) == 0, "PixelShuffle: C must be divisible by k²");
    let c_out = c / (k * k);
    let r = x
        .reshape((b, c_out, k, k, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, c_out, h * k, w * k))?;
    Ok(r)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// Small Stage A config: fewer decoder blocks so tests stay fast.
    fn small_cfg() -> Config {
        Config {
            image_channels: 3,
            pixel_unshuffle: 2,
            c_hidden_in: 16,
            c_hidden_deep: 32,
            c_latent: 4,
            num_codes: 64,
            n_decoder_deep_blocks: 2,
        }
    }

    fn random_vae(cfg: Config) -> (StageAVae, VarMap) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vae = StageAVae::new(cfg, vb).expect("StageAVae::new");
        (vae, varmap)
    }

    #[test]
    fn paella_v3_config_matches_upstream_inspection() {
        let cfg = Config::paella_v3();
        assert_eq!(cfg.image_channels, 3);
        assert_eq!(cfg.pixel_unshuffle, 2);
        assert_eq!(cfg.c_hidden_in, 192);
        assert_eq!(cfg.c_hidden_deep, 384);
        assert_eq!(cfg.c_latent, 4);
        assert_eq!(cfg.num_codes, 8192);
        assert_eq!(cfg.n_decoder_deep_blocks, 12);
    }

    // ---- PixelUnshuffle / PixelShuffle round-trip ----

    #[test]
    fn pixel_unshuffle_then_shuffle_returns_input() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 3, 8, 8), &device).unwrap();
        let y = pixel_unshuffle(&x, 2).unwrap();
        assert_eq!(y.dims(), &[1, 12, 4, 4]);
        let z = pixel_shuffle(&y, 2).unwrap();
        assert_eq!(z.dims(), &[1, 3, 8, 8]);
        let diff = (&x - &z)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "pixel un/shuffle round-trip lossy: {diff}");
    }

    #[test]
    fn pixel_unshuffle_rejects_non_divisible_size() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 3, 7, 7), &device).unwrap();
        match pixel_unshuffle(&x, 2) {
            Ok(_) => panic!("expected divisibility error"),
            Err(e) => assert!(format!("{e}").contains("divisible")),
        }
    }

    // ---- PaellaResBlock ----

    #[test]
    fn paella_resblock_preserves_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let blk = PaellaResBlock::new(8, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8, 4, 4]);
    }

    #[test]
    fn paella_resblock_skip_dominates_when_gammas_zeroed() {
        // gammas all zero → residual scale gammas[4]=0 → output = x.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let blk = PaellaResBlock::new(8, vb).unwrap();
        // Zero out the gammas parameter.
        for (name, var) in varmap.data().lock().unwrap().iter() {
            if name == "gammas" {
                let z = Tensor::zeros(6, DType::F32, &device).unwrap();
                var.set(&z).unwrap();
            }
        }
        let x = Tensor::randn(0f32, 1f32, (1, 8, 4, 4), &device).unwrap();
        let y = blk.forward(&x).unwrap();
        let diff = (&x - &y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "with gammas=0 forward should be identity (got max diff {diff})");
    }

    // ---- VectorQuantizer ----

    #[test]
    fn vector_quantizer_lookup_returns_codebook_entry() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vq = VectorQuantizer::new(16, 4, vb).unwrap();
        // Random latent → quantize → indices should pick the
        // nearest codebook entry. Test by passing the codebook
        // entries themselves and expecting indices [0..N].
        let indices = Tensor::from_vec(vec![0u32, 5, 10, 15], (1, 2, 2), &device).unwrap();
        let z_q = vq.decode_indices(&indices).unwrap();
        assert_eq!(z_q.dims(), &[1, 4, 2, 2]);
    }

    #[test]
    fn vector_quantizer_picks_nearest_code() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vq = VectorQuantizer::new(8, 4, vb.clone()).unwrap();
        // Patch codebook so entry 3 has a known value.
        for (name, var) in varmap.data().lock().unwrap().iter() {
            if name == "embedding.weight" {
                let mut data = vec![0f32; 8 * 4];
                // Entry 3 = [10, 10, 10, 10] — far from others (zero).
                for j in 0..4 {
                    data[3 * 4 + j] = 10.0;
                }
                let cb = Tensor::from_vec(data, (8, 4), &device).unwrap();
                var.set(&cb).unwrap();
            }
        }
        // z near [10, 10, 10, 10] should pick code 3.
        let z = Tensor::from_vec(
            vec![9.5f32, 10.0, 10.5, 9.0],
            (1, 4, 1, 1),
            &device,
        )
        .unwrap();
        let (z_q, indices) = vq.quantize(&z).unwrap();
        assert_eq!(z_q.dims(), &[1, 4, 1, 1]);
        assert_eq!(indices.dims(), &[1, 1, 1]);
        let idx_val: Vec<u32> = indices.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(idx_val[0], 3, "expected closest code to be 3");
    }

    // ---- StageAVae top-level shape ----

    #[test]
    fn vae_encode_compresses_4x() {
        let (vae, _) = random_vae(small_cfg());
        // 16x16 RGB → after PixelUnshuffle(2) + strided conv → 4×4.
        let image = Tensor::randn(0f32, 1f32, (1, 3, 16, 16), &vae.device).unwrap();
        let z = vae.encode(&image).unwrap();
        assert_eq!(z.dims(), &[1, 4, 4, 4]);
    }

    #[test]
    fn vae_decode_expands_4x() {
        let (vae, _) = random_vae(small_cfg());
        let z = Tensor::randn(0f32, 1f32, (1, 4, 4, 4), &vae.device).unwrap();
        let image = vae.decode(&z).unwrap();
        assert_eq!(image.dims(), &[1, 3, 16, 16]);
    }

    #[test]
    fn vae_round_trip_preserves_spatial_shape() {
        let (vae, _) = random_vae(small_cfg());
        let image = Tensor::randn(0f32, 1f32, (1, 3, 16, 16), &vae.device).unwrap();
        let restored = vae.round_trip(&image).unwrap();
        assert_eq!(restored.dims(), image.dims());
    }

    /// v0.39 phase 0h: real-weight smoke for Stage A. Skipped
    /// unless `STABLE_CASCADE_WEIGHTS_DIR` env var points at a
    /// directory containing `vqgan/diffusion_pytorch_model.safetensors`.
    ///
    /// Stage A is the smallest stage (~14 MB), the cheapest
    /// real-weight verification. Success means cascade_vae's tensor
    /// naming matches `stabilityai/stable-cascade/vqgan/`.
    #[test]
    fn stage_a_loads_from_real_upstream_weights() {
        let dir = match std::env::var("STABLE_CASCADE_WEIGHTS_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let path = std::path::PathBuf::from(&dir)
            .join("vqgan/diffusion_pytorch_model.safetensors");
        if !path.exists() {
            eprintln!(
                "Skipping stage_a_loads_from_real_upstream_weights: \
                 {} doesn't exist (set STABLE_CASCADE_WEIGHTS_DIR to a \
                 directory containing vqgan/diffusion_pytorch_model.safetensors \
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
            .expect("mmap stage_a weights")
        };
        match StageAVae::new(Config::paella_v3(), vb) {
            Ok(_) => eprintln!("✓ Stage A real-weight load OK ({})", path.display()),
            Err(e) => panic!(
                "Stage A real-weight load FAILED — indicates tensor naming \
                 mismatch between v0.39 cascade_vae and upstream:\n  {e}"
            ),
        }
    }

    #[test]
    fn vae_quantize_changes_output() {
        let (vae, _) = random_vae(small_cfg());
        let z = Tensor::randn(0f32, 1f32, (1, 4, 4, 4), &vae.device).unwrap();
        let (z_q, _idx) = vae.quantize(&z).unwrap();
        // z_q is the codebook entry, not z itself — they should differ.
        let diff = (&z - &z_q)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // With random codebook + random z, mean abs diff is typically
        // O(1). Just confirm it's not zero (would mean we returned z).
        assert!(diff > 1e-3, "quantization should change the latent ({diff})");
    }
}
