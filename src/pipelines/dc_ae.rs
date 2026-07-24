//! DC-AE — the deep-compression autoencoder used by Sana (ROADMAP_4.5.0 Phase 1).
//!
//! A faithful candle port of diffusers' `AutoencoderDC` (`mit-han-lab/dc-ae-f32c32-sana-1.0`):
//! 32× spatial compression, 32 latent channels, and — unlike every other VAE in plakat — a
//! **plain deterministic** autoencoder (encode → latent directly; no KL / no `.sample()`).
//!
//! Architecture (6 stages, `[128,256,512,512,1024,1024]`): the first 3 stages are `ResBlock`
//! stacks, the last 3 are `EfficientViTBlock` stacks (ReLU **linear** multiscale attention +
//! GLU-MBConv). Downsampling is stride-2 conv + a pixel-unshuffle-average shortcut; upsampling is
//! nearest-interpolate + conv + a channel-duplicate pixel-shuffle shortcut.
//!
//! **Numerics:** the linear attention is not self-normalizing (unbounded `Σφ(k)v`), so — exactly
//! like diffusers — the reduction runs in **F32**. The whole VAE is precision-sensitive and runs
//! once per image, so plakat keeps it in F32 (CPU-canonical for verify).

use anyhow::{Context, Result};
use candle_core::{DType, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder, conv2d, conv2d_no_bias, linear_no_bias};

/// A pluggable image autoencoder. Sana's DC-AE is the first (and, for now, only) implementor —
/// existing pipelines keep the concrete `AutoEncoderKL`, so this trait is additive.
pub trait ImageVae {
    /// Encode pixels `(B,3,H,W)` in `[-1,1]` → latent `(B,C,H/f,W/f)` (raw, UN-scaled).
    fn encode(&self, pixels: &Tensor) -> Result<Tensor>;
    /// Decode a raw (UN-scaled) latent → pixels `(B,3,H,W)` in `[-1,1]`.
    fn decode(&self, latent: &Tensor) -> Result<Tensor>;
    /// `z_model = z_raw * scaling_factor`; decode expects `z_raw = z_model / scaling_factor`.
    fn scaling_factor(&self) -> f64;
    fn latent_channels(&self) -> usize;
    fn spatial_compression(&self) -> usize;
}

// ── the Sana f32c32 architecture (fixed for this model) ──────────────────────────────────────
const BLOCK_OUT: [usize; 6] = [128, 256, 512, 512, 1024, 1024];
const ENC_LAYERS: [usize; 6] = [2, 2, 2, 3, 3, 3];
const DEC_LAYERS: [usize; 6] = [3, 3, 3, 3, 3, 3];
const HEAD_DIM: usize = 32;
const LATENT_CH: usize = 32;
const IN_CH: usize = 3;
const RMS_EPS: f64 = 1e-5;
const ATTN_EPS: f64 = 1e-15;
/// `EfficientViTBlock` at stages ≥ this index (the last 3); `ResBlock` below.
const VIT_FROM: usize = 3;

fn cfg(padding: usize, stride: usize, groups: usize) -> Conv2dConfig {
    Conv2dConfig { padding, stride, dilation: 1, groups, cudnn_fwd_algo: None }
}

/// RMSNorm over the **channel** dim of an NCHW tensor, with weight+bias (eps 1e-5). Matches
/// diffusers `RMSNorm(x.movedim(1,-1)).movedim(-1,1)`.
fn rms_norm_2d(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let c = x.dim(1)?;
    let var = x.sqr()?.mean_keepdim(1)?; // (B,1,H,W)
    let xn = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    let w = weight.reshape((1, c, 1, 1))?;
    let b = bias.reshape((1, c, 1, 1))?;
    Ok(xn.broadcast_mul(&w)?.broadcast_add(&b)?)
}

/// `F.pixel_unshuffle(x, r)`: (B,C,H,W) → (B, C·r², H/r, W/r), channel order (c, i, j).
fn pixel_unshuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let out = x
        .reshape((b, c, h / r, r, w / r, r))?
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c * r * r, h / r, w / r))?;
    Ok(out)
}

/// `F.pixel_shuffle(x, r)`: (B, C·r², H, W) → (B, C, H·r, W·r).
fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let cout = c / (r * r);
    let out = x
        .reshape((b, cout, r, r, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, cout, h * r, w * r))?;
    Ok(out)
}

// ── ResBlock ─────────────────────────────────────────────────────────────────────────────────
struct ResBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    norm_w: Tensor,
    norm_b: Tensor,
}
impl ResBlock {
    fn load(c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: conv2d(c, c, 3, cfg(1, 1, 1), vb.pp("conv1"))?,
            conv2: conv2d_no_bias(c, c, 3, cfg(1, 1, 1), vb.pp("conv2"))?,
            norm_w: vb.get(c, "norm.weight")?,
            norm_b: vb.get(c, "norm.bias")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = rms_norm_2d(&h, &self.norm_w, &self.norm_b, RMS_EPS)?;
        Ok((h + x)?)
    }
}

// ── SanaMultiscaleAttentionProjection (the 5×5 depthwise multiscale) ─────────────────────────
struct MultiscaleProj {
    proj_in: Conv2d,
    proj_out: Conv2d,
}
impl MultiscaleProj {
    fn load(inner: usize, num_heads: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let ch = 3 * inner;
        Ok(Self {
            proj_in: conv2d_no_bias(ch, ch, kernel, cfg(kernel / 2, 1, ch), vb.pp("proj_in"))?,
            proj_out: conv2d_no_bias(ch, ch, 1, cfg(0, 1, 3 * num_heads), vb.pp("proj_out"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.proj_out.forward(&self.proj_in.forward(x)?)?)
    }
}

// ── SanaMultiscaleLinearAttention ────────────────────────────────────────────────────────────
struct LinearAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    multiscale: Vec<MultiscaleProj>,
    to_out: Linear,
    norm_w: Tensor,
    norm_b: Tensor,
    num_heads: usize,
}
impl LinearAttention {
    fn load(in_ch: usize, vb: VarBuilder) -> Result<Self> {
        // num_heads = in_ch // head_dim * mult(=1); inner = num_heads * head_dim = in_ch.
        let num_heads = in_ch / HEAD_DIM;
        let inner = num_heads * HEAD_DIM;
        let n_scales = 1usize; // qkv_multiscales = (5,)
        let mut multiscale = Vec::with_capacity(n_scales);
        let ms_vb = vb.pp("to_qkv_multiscale");
        for i in 0..n_scales {
            multiscale.push(MultiscaleProj::load(inner, num_heads, 5, ms_vb.pp(i))?);
        }
        Ok(Self {
            to_q: linear_no_bias(in_ch, inner, vb.pp("to_q"))?,
            to_k: linear_no_bias(in_ch, inner, vb.pp("to_k"))?,
            to_v: linear_no_bias(in_ch, inner, vb.pp("to_v"))?,
            multiscale,
            to_out: linear_no_bias(inner * (1 + n_scales), in_ch, vb.pp("to_out"))?,
            norm_w: vb.get(in_ch, "norm_out.weight")?,
            norm_b: vb.get(in_ch, "norm_out.bias")?,
            num_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, h, w) = x.dims4()?;
        let n = h * w;
        let residual = x;
        // q/k/v on channels-last, then re-stack channel-first as [q,k,v].
        let xl = x.permute((0, 2, 3, 1))?.contiguous()?; // (B,H,W,C)
        let q = self.to_q.forward(&xl)?;
        let k = self.to_k.forward(&xl)?;
        let v = self.to_v.forward(&xl)?;
        let qkv = Tensor::cat(&[q, k, v], 3)?.permute((0, 3, 1, 2))?.contiguous()?; // (B,3*inner,H,W)

        // multiscale: [qkv, proj(qkv), …] concatenated on channel.
        let mut scales = vec![qkv.clone()];
        for ms in &self.multiscale {
            scales.push(ms.forward(&qkv)?);
        }
        let hs = Tensor::cat(&scales, 1)?; // (B, (1+s)*3*inner, H, W)

        // → (B, groups, 3*head_dim, N); split q/k/v on the 3*head_dim axis.
        let groups = self.num_heads * (1 + self.multiscale.len());
        let hs = hs.reshape((b, groups, 3 * HEAD_DIM, n))?;
        let query = hs.narrow(2, 0, HEAD_DIM)?.relu()?;
        let key = hs.narrow(2, HEAD_DIM, HEAD_DIM)?.relu()?;
        let value = hs.narrow(2, 2 * HEAD_DIM, HEAD_DIM)?;

        // Linear attention (F32): value gets a ones-row (denominator), then two matmuls.
        let ones = Tensor::ones((b, groups, 1, n), DType::F32, value.device())?;
        let value = Tensor::cat(&[value, ones], 2)?; // (B,g,head_dim+1,N)
        let scores = value.matmul(&key.transpose(2, 3)?.contiguous()?)?; // (B,g,head_dim+1,head_dim)
        let out = scores.matmul(&query.contiguous()?)?; // (B,g,head_dim+1,N)
        let num = out.narrow(2, 0, HEAD_DIM)?; // (B,g,head_dim,N)
        let den = (out.narrow(2, HEAD_DIM, 1)? + ATTN_EPS)?; // (B,g,1,N)
        let attn = num.broadcast_div(&den)?; // (B,g,head_dim,N)

        // → (B, inner*(1+s), H, W), then to_out (channels-last) + norm + residual.
        let attn = attn.reshape((b, groups * HEAD_DIM, h, w))?;
        let attn = attn.permute((0, 2, 3, 1))?.contiguous()?; // (B,H,W,inner*(1+s))
        let attn = self.to_out.forward(&attn)?.permute((0, 3, 1, 2))?.contiguous()?; // (B,C,H,W)
        let attn = rms_norm_2d(&attn, &self.norm_w, &self.norm_b, RMS_EPS)?;
        Ok((attn + residual)?)
    }
}

// ── GLU-MBConv (the Mix-FFN) ─────────────────────────────────────────────────────────────────
struct GluMbConv {
    conv_inverted: Conv2d,
    conv_depth: Conv2d,
    conv_point: Conv2d,
    norm_w: Tensor,
    norm_b: Tensor,
}
impl GluMbConv {
    fn load(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = 4 * in_ch; // expand_ratio = 4
        Ok(Self {
            conv_inverted: conv2d(in_ch, hidden * 2, 1, cfg(0, 1, 1), vb.pp("conv_inverted"))?,
            conv_depth: conv2d(hidden * 2, hidden * 2, 3, cfg(1, 1, hidden * 2), vb.pp("conv_depth"))?,
            conv_point: conv2d_no_bias(hidden, out_ch, 1, cfg(0, 1, 1), vb.pp("conv_point"))?,
            norm_w: vb.get(out_ch, "norm.weight")?,
            norm_b: vb.get(out_ch, "norm.bias")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.conv_inverted.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv_depth.forward(&h)?;
        let half = h.dim(1)? / 2;
        let a = h.narrow(1, 0, half)?;
        let gate = h.narrow(1, half, half)?;
        let h = (a * candle_nn::ops::silu(&gate)?)?;
        let h = self.conv_point.forward(&h)?;
        let h = rms_norm_2d(&h, &self.norm_w, &self.norm_b, RMS_EPS)?;
        Ok((h + residual)?)
    }
}

// ── EfficientViTBlock = LinearAttention + GLU-MBConv ─────────────────────────────────────────
struct EfficientVitBlock {
    attn: LinearAttention,
    conv_out: GluMbConv,
}
impl EfficientVitBlock {
    fn load(c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: LinearAttention::load(c, vb.pp("attn"))?,
            conv_out: GluMbConv::load(c, c, vb.pp("conv_out"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv_out.forward(&self.attn.forward(x)?)
    }
}

// ── either block, so a stage is a homogeneous Vec ────────────────────────────────────────────
enum Block {
    Res(ResBlock),
    Vit(EfficientVitBlock),
}
impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Block::Res(b) => b.forward(x),
            Block::Vit(b) => b.forward(x),
        }
    }
}
fn load_block(stage: usize, c: usize, vb: VarBuilder) -> Result<Block> {
    if stage >= VIT_FROM {
        Ok(Block::Vit(EfficientVitBlock::load(c, vb)?))
    } else {
        Ok(Block::Res(ResBlock::load(c, vb)?))
    }
}

// ── DCDownBlock2d (downsample_block_type = "Conv" → stride-2, no unshuffle on the conv path) ──
struct DownBlock {
    conv: Conv2d,
    group_size: usize,
}
impl DownBlock {
    fn load(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        // downsample=False ("Conv"): stride 2, conv out = out_ch. group_size = in*4/out.
        Ok(Self {
            conv: conv2d(in_ch, out_ch, 3, cfg(1, 2, 1), vb.pp("conv"))?,
            group_size: in_ch * 4 / out_ch,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let conv = self.conv.forward(x)?; // (B,out,H/2,W/2)
        // shortcut: pixel_unshuffle → group-average.
        let y = pixel_unshuffle(x, 2)?; // (B, in*4, H/2, W/2)
        let (b, cc, hh, ww) = y.dims4()?;
        let y = y.reshape((b, cc / self.group_size, self.group_size, hh, ww))?.mean(2)?; // (B,out,H/2,W/2)
        Ok((conv + y)?)
    }
}

// ── DCUpBlock2d (upsample_block_type = "interpolate") ────────────────────────────────────────
struct UpBlock {
    conv: Conv2d,
    repeats: usize,
    shortcut: bool,
}
impl UpBlock {
    fn load(in_ch: usize, out_ch: usize, shortcut: bool, vb: VarBuilder) -> Result<Self> {
        // interpolate=True: conv out = out_ch. repeats = out*4/in.
        Ok(Self {
            conv: conv2d(in_ch, out_ch, 3, cfg(1, 1, 1), vb.pp("conv"))?,
            repeats: out_ch * 4 / in_ch,
            shortcut,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        let conv = self.conv.forward(&up)?; // (B,out,2H,2W)
        if !self.shortcut {
            return Ok(conv);
        }
        // shortcut: repeat_interleave channels → pixel_shuffle.
        let y = x.repeat_interleave(self.repeats, 1)?; // (B, in*repeats, H, W)
        let y = pixel_shuffle(&y, 2)?; // (B, in*repeats/4, 2H, 2W) = (B, out, 2H, 2W)
        Ok((conv + y)?)
    }
}

/// `repeat_interleave` along the channel dim (candle lacks it): duplicate each channel `n×`,
/// preserving order (c0,c0,…,c1,c1,…), matching `torch.repeat_interleave(x, n, dim=1)`.
trait RepeatInterleave {
    fn repeat_interleave(&self, n: usize, dim: usize) -> candle_core::Result<Tensor>;
}
impl RepeatInterleave for Tensor {
    fn repeat_interleave(&self, n: usize, dim: usize) -> candle_core::Result<Tensor> {
        // insert a new axis after `dim`, broadcast to n, then merge back.
        let mut dims = self.dims().to_vec();
        let x = self.unsqueeze(dim + 1)?;
        let mut bshape = x.dims().to_vec();
        bshape[dim + 1] = n;
        let x = x.broadcast_as(bshape)?.contiguous()?;
        dims[dim] *= n;
        x.reshape(dims)
    }
}

// ── Encoder ──────────────────────────────────────────────────────────────────────────────────
struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<Vec<Block>>, // per stage: the block stack (+ trailing downsample folded in)
    downsamples: Vec<Option<DownBlock>>,
    conv_out: Conv2d,
    out_shortcut_group: usize,
}
impl Encoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv_in = conv2d(IN_CH, BLOCK_OUT[0], 3, cfg(1, 1, 1), vb.pp("conv_in"))?;
        let db = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(6);
        let mut downsamples = Vec::with_capacity(6);
        for (stage, (&c, &layers)) in BLOCK_OUT.iter().zip(ENC_LAYERS.iter()).enumerate() {
            let svb = db.pp(stage);
            let mut idx = 0;
            let mut blocks = Vec::with_capacity(layers);
            for _ in 0..layers {
                blocks.push(load_block(stage, c, svb.pp(idx))?);
                idx += 1;
            }
            // trailing downsample except the last stage.
            let down = if stage < 5 {
                let d = DownBlock::load(c, BLOCK_OUT[stage + 1], svb.pp(idx))?;
                Some(d)
            } else {
                None
            };
            down_blocks.push(blocks);
            downsamples.push(down);
        }
        Ok(Self {
            conv_in,
            down_blocks,
            downsamples,
            conv_out: conv2d(BLOCK_OUT[5], LATENT_CH, 3, cfg(1, 1, 1), vb.pp("conv_out"))?,
            out_shortcut_group: BLOCK_OUT[5] / LATENT_CH,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for (stage, blocks) in self.down_blocks.iter().enumerate() {
            for blk in blocks {
                h = blk.forward(&h)?;
            }
            if let Some(down) = &self.downsamples[stage] {
                h = down.forward(&h)?;
            }
        }
        // out_shortcut: channel-group average added to conv_out.
        let (b, c, hh, ww) = h.dims4()?;
        let sc = h.reshape((b, c / self.out_shortcut_group, self.out_shortcut_group, hh, ww))?.mean(2)?;
        Ok((self.conv_out.forward(&h)? + sc)?)
    }
}

// ── Decoder ──────────────────────────────────────────────────────────────────────────────────
struct Decoder {
    conv_in: Conv2d,
    in_shortcut_repeats: usize,
    up_blocks: Vec<Vec<Block>>, // per stage (high→low index order as stored): upsample folded in
    upsamples: Vec<Option<UpBlock>>,
    norm_w: Tensor,
    norm_b: Tensor,
    conv_out: Conv2d,
}
impl Decoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv_in = conv2d(LATENT_CH, BLOCK_OUT[5], 3, cfg(1, 1, 1), vb.pp("conv_in"))?;
        let ub = vb.pp("up_blocks");
        // up_blocks are stored index 0..5 (stage 0 = highest-res). Each stage i (for i<5) begins
        // with an upsample from block_out[i+1]→block_out[i], then `layers` blocks.
        let mut up_blocks = Vec::with_capacity(6);
        let mut upsamples = Vec::with_capacity(6);
        for (stage, (&c, &layers)) in BLOCK_OUT.iter().zip(DEC_LAYERS.iter()).enumerate() {
            let svb = ub.pp(stage);
            let mut idx = 0;
            let up = if stage < 5 {
                let u = UpBlock::load(BLOCK_OUT[stage + 1], c, true, svb.pp(idx))?;
                idx += 1;
                Some(u)
            } else {
                None
            };
            let mut blocks = Vec::with_capacity(layers);
            for _ in 0..layers {
                blocks.push(load_block(stage, c, svb.pp(idx))?);
                idx += 1;
            }
            up_blocks.push(blocks);
            upsamples.push(up);
        }
        Ok(Self {
            conv_in,
            in_shortcut_repeats: BLOCK_OUT[5] / LATENT_CH,
            up_blocks,
            upsamples,
            norm_w: vb.get(BLOCK_OUT[0], "norm_out.weight")?,
            norm_b: vb.get(BLOCK_OUT[0], "norm_out.bias")?,
            conv_out: conv2d(BLOCK_OUT[0], IN_CH, 3, cfg(1, 1, 1), vb.pp("conv_out"))?,
        })
    }
    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        // in_shortcut: repeat_interleave latent channels, add to conv_in.
        let sc = z.repeat_interleave(self.in_shortcut_repeats, 1)?;
        let mut h = (self.conv_in.forward(z)? + sc)?;
        // run stages high-index → low-index (diffusers iterates `reversed(up_blocks)`).
        for stage in (0..6).rev() {
            if let Some(up) = &self.upsamples[stage] {
                h = up.forward(&h)?;
            }
            for blk in &self.up_blocks[stage] {
                h = blk.forward(&h)?;
            }
        }
        let h = rms_norm_2d(&h, &self.norm_w, &self.norm_b, RMS_EPS)?;
        let h = h.relu()?; // conv_act = relu
        Ok(self.conv_out.forward(&h)?)
    }
}

/// The full DC-AE. Runs in F32 (precision-sensitive linear attention).
pub struct AutoencoderDc {
    encoder: Encoder,
    decoder: Decoder,
    scaling_factor: f64,
}

impl AutoencoderDc {
    /// Load from a diffusers `vae/` VarBuilder (F32). `scaling_factor` from the model config
    /// (0.41407 for f32c32-sana).
    pub fn load(vb: VarBuilder, scaling_factor: f64) -> Result<Self> {
        Ok(Self {
            encoder: Encoder::load(vb.pp("encoder")).context("DC-AE encoder")?,
            decoder: Decoder::load(vb.pp("decoder")).context("DC-AE decoder")?,
            scaling_factor,
        })
    }
}

impl ImageVae for AutoencoderDc {
    fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        let x = pixels.to_dtype(DType::F32)?;
        self.encoder.forward(&x)
    }
    fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let z = latent.to_dtype(DType::F32)?;
        self.decoder.forward(&z)
    }
    fn scaling_factor(&self) -> f64 {
        self.scaling_factor
    }
    fn latent_channels(&self) -> usize {
        LATENT_CH
    }
    fn spatial_compression(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Pearson correlation of two tensors' flattened F32 data.
    fn corr(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let mut num = 0.0;
        let (mut da, mut db) = (0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(&b) {
            num += (x - ma) * (y - mb);
            da += (x - ma).powi(2);
            db += (y - mb).powi(2);
        }
        num / (da.sqrt() * db.sqrt() + 1e-12)
    }
    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_vec0::<f32>().unwrap()
    }

    /// Verify the candle DC-AE against a diffusers reference dump. Opt-in (needs the weights +
    /// `tools/reference/out/sana-dcae/goldens.safetensors`): set `PLAKAT_DCAE_VERIFY=1`, and
    /// optionally `PLAKAT_DCAE_WEIGHTS=/path/to/diffusion_pytorch_model.safetensors`.
    #[test]
    fn dcae_matches_diffusers_reference() {
        if std::env::var("PLAKAT_DCAE_VERIFY").is_err() {
            return;
        }
        let dev = Device::Cpu;
        let weights = std::env::var("PLAKAT_DCAE_WEIGHTS").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            let snaps = format!(
                "{home}/.cache/huggingface/hub/models--mit-han-lab--dc-ae-f32c32-sana-1.0-diffusers/snapshots"
            );
            let snap = std::fs::read_dir(&snaps).unwrap().next().unwrap().unwrap().path();
            snap.join("diffusion_pytorch_model.safetensors").to_string_lossy().into_owned()
        });
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &dev).unwrap()
        };
        let vae = AutoencoderDc::load(vb, 0.41407).unwrap();

        let g = candle_core::safetensors::load("tools/reference/out/sana-dcae/goldens.safetensors", &dev).unwrap();
        let (image_in, latent_enc, recon_dec, latent_fixed, decode_fixed) = (
            &g["image_in"], &g["latent_enc"], &g["recon_dec"], &g["latent_fixed"], &g["decode_fixed"],
        );

        // 1) decode in isolation (the highest-value early check).
        let my_decode = vae.decode(latent_fixed).unwrap();
        let (c, m) = (corr(&my_decode, decode_fixed), max_abs(&my_decode, decode_fixed));
        eprintln!("decode_fixed: corr={c:.6} max_abs={m:.5}");
        assert!(c > 0.999, "decode corr {c} < 0.999");

        // 2) encode in isolation.
        let my_enc = vae.encode(image_in).unwrap();
        let (c2, m2) = (corr(&my_enc, latent_enc), max_abs(&my_enc, latent_enc));
        eprintln!("latent_enc:   corr={c2:.6} max_abs={m2:.5}");
        assert!(c2 > 0.999, "encode corr {c2} < 0.999");

        // 3) encode→decode round-trip.
        let my_recon = vae.decode(&my_enc).unwrap();
        let c3 = corr(&my_recon, recon_dec);
        eprintln!("recon_dec:    corr={c3:.6}");
        assert!(c3 > 0.999, "recon corr {c3} < 0.999");
    }
}
