//! InsightFace `inswapper_128` face-swap generator — Rust port.
//!
//! A SimSwap/FaceShifter-style generator: it takes an aligned target face
//! (`1×3×128×128`, RGB, /255) and a 512-d **source** identity latent
//! (`normalize(arcface_embedding) @ emap`, renormalised) and returns the target
//! face wearing the source identity (`1×3×128×128`, RGB, [0,1]).
//!
//! Architecture (verified against `inswapper_128.onnx` to <1e-3):
//! ```text
//!   encoder : reflect-pad3 → conv 3→128 7×7 → LeakyReLU(0.2)
//!             conv 128→256 3×3 → LReLU; conv 256→512 3×3 s2 → LReLU;
//!             conv 512→1024 3×3 s2 → LReLU            (→ 32×32×1024)
//!   bottleneck : 6 × residual AdaIN block
//!             x + [ pad1→conv→InstanceNorm→AdaIN(style0)→ReLU
//!                   →pad1→conv→InstanceNorm→AdaIN(style1) ]
//!             style = source · Gemm → [scale(1024), bias(1024)]
//!   decoder : bilinear×2 → conv 1024→512 → LReLU
//!             bilinear×2 → conv 512→256 → LReLU; conv 256→128 → LReLU
//!             reflect-pad3 → conv 128→3 7×7 → Tanh → (x+1)/2
//! ```
//! Setup is bring-your-own-weights: `plakat convert-onnx inswapper_128.onnx
//! inswapper_128.safetensors --arch inswapper-128`.

#![allow(dead_code)] // face-swap is opt-in; some helpers are public for reuse.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use std::path::Path;

const LRELU: f64 = 0.2;
const IN_EPS: f64 = 1e-8;

/// A conv2d with bias under `vb` (`weight` + `bias`).
fn conv(vb: VarBuilder, in_ch: usize, out_ch: usize, k: usize, stride: usize, padding: usize) -> Result<Conv2d> {
    let cfg = Conv2dConfig { padding, stride, dilation: 1, groups: 1, ..Default::default() };
    Ok(candle_nn::conv2d(in_ch, out_ch, k, cfg, vb)?)
}

/// A linear layer `y = x·Wᵀ + b` under `vb` (ONNX Gemm transB=1).
struct Linear {
    w: Tensor, // (out, in)
    b: Tensor, // (out,)
}
impl Linear {
    fn new(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self {
            w: vb.get((out_dim, in_dim), "weight")?,
            b: vb.get(out_dim, "bias")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x (1, in) · Wᵀ (in, out) + b
        let y = x.matmul(&self.w.t()?)?;
        Ok(y.broadcast_add(&self.b)?)
    }
}

/// LeakyReLU(0.2): relu(x) − 0.2·relu(−x).
fn leaky_relu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = x.neg()?.relu()?;
    Ok((pos - (neg * LRELU)?)?)
}

/// Reflect-pad the last two dims (H, W) by `p` each side (PyTorch `reflect`:
/// mirror without repeating the edge pixel).
fn reflect_pad2d(x: &Tensor, p: usize) -> Result<Tensor> {
    if p == 0 {
        return Ok(x.clone());
    }
    let dims = x.dims4()?;
    let (h, w) = (dims.2, dims.3);
    let x = reflect_pad_dim(x, 2, h, p)?;
    let x = reflect_pad_dim(&x, 3, w, p)?;
    Ok(x)
}

/// Reflect-pad one dim of size `n` by `p` each side via index_select.
fn reflect_pad_dim(x: &Tensor, dim: usize, n: usize, p: usize) -> Result<Tensor> {
    let mut idx: Vec<u32> = Vec::with_capacity(n + 2 * p);
    // left: p, p-1, …, 1
    for k in (1..=p).rev() {
        idx.push(k as u32);
    }
    // identity
    for k in 0..n {
        idx.push(k as u32);
    }
    // right: n-2, n-3, …, n-1-p
    for k in 1..=p {
        idx.push((n - 1 - k) as u32);
    }
    let index = Tensor::from_vec(idx, (n + 2 * p,), x.device())?;
    Ok(x.index_select(&index, dim)?)
}

/// Instance norm over spatial dims (per sample, per channel), no affine.
fn instance_norm(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(2)?.mean_keepdim(3)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?.mean_keepdim(3)?;
    let std = (var + IN_EPS)?.sqrt()?;
    Ok(centered.broadcast_div(&std)?)
}

/// AdaIN: `scale · instance_norm(x) + bias`, where `[scale,bias]` is the 2048-d
/// style split into two 1024-d halves and broadcast over space.
fn adain(x: &Tensor, style: &Tensor, ch: usize) -> Result<Tensor> {
    let norm = instance_norm(x)?;
    let scale = style.i((.., 0..ch))?.reshape((1, ch, 1, 1))?;
    let bias = style.i((.., ch..2 * ch))?.reshape((1, ch, 1, 1))?;
    Ok((norm.broadcast_mul(&scale)?.broadcast_add(&bias))?)
}

/// Bilinear 2× upsample with PyTorch `half_pixel` coords, separable (H then W).
fn bilinear_up2x(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let ah = interp_matrix(h, x.device(), x.dtype())?; // (2h, h)
    let aw = interp_matrix(w, x.device(), x.dtype())?; // (2w, w)
    // Resize H: move H last → (B*C*W, H) · (H, 2H) → (…, 2H).
    let xh = x
        .permute((0, 1, 3, 2))? // (B,C,W,H)
        .reshape((b * c * w, h))?
        .matmul(&ah.t()?)? // (B*C*W, 2H)
        .reshape((b, c, w, 2 * h))?
        .permute((0, 1, 3, 2))? // (B,C,2H,W)
        .contiguous()?;
    // Resize W: move W last → (B*C*2H, W) · (W, 2W).
    let xw = xh
        .reshape((b * c * 2 * h, w))?
        .matmul(&aw.t()?)?
        .reshape((b, c, 2 * h, 2 * w))?;
    Ok(xw)
}

/// `(2n, n)` bilinear interpolation matrix for half-pixel 2× upsampling.
fn interp_matrix(n: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let m = 2 * n;
    let mut a = vec![0f32; m * n];
    for o in 0..m {
        // half_pixel: src = (o + 0.5)/scale − 0.5, scale = 2.
        let src = ((o as f32 + 0.5) / 2.0) - 0.5;
        let src = src.clamp(0.0, (n - 1) as f32);
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = src - lo as f32;
        a[o * n + lo] += 1.0 - frac;
        a[o * n + hi] += frac;
    }
    Ok(Tensor::from_vec(a, (m, n), device)?.to_dtype(dtype)?)
}

/// One residual AdaIN block: `x + [conv→IN→AdaIN→ReLU→conv→IN→AdaIN]`.
struct AdaInBlock {
    conv0: Conv2d,
    conv1: Conv2d,
    style0: Linear,
    style1: Linear,
    ch: usize,
}
impl AdaInBlock {
    fn new(vb: VarBuilder, ch: usize) -> Result<Self> {
        Ok(Self {
            conv0: conv(vb.pp("conv0"), ch, ch, 3, 1, 0)?,
            conv1: conv(vb.pp("conv1"), ch, ch, 3, 1, 0)?,
            style0: Linear::new(vb.pp("style0"), 512, 2 * ch)?,
            style1: Linear::new(vb.pp("style1"), 512, 2 * ch)?,
            ch,
        })
    }
    fn forward(&self, x: &Tensor, source: &Tensor) -> Result<Tensor> {
        let h = reflect_pad2d(x, 1)?;
        let h = self.conv0.forward(&h)?;
        let h = adain(&h, &self.style0.forward(source)?, self.ch)?;
        let h = h.relu()?;
        let h = reflect_pad2d(&h, 1)?;
        let h = self.conv1.forward(&h)?;
        let h = adain(&h, &self.style1.forward(source)?, self.ch)?;
        Ok((x + h)?)
    }
}

/// The full `inswapper_128` generator.
pub struct Inswapper {
    enc: Vec<Conv2d>,
    blocks: Vec<AdaInBlock>,
    dec: Vec<Conv2d>,
    out_conv: Conv2d,
    device: Device,
    dtype: DType,
}

impl Inswapper {
    pub fn load(weights: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        Self::new(vb, device.clone(), dtype)
    }

    pub fn new(vb: VarBuilder, device: Device, dtype: DType) -> Result<Self> {
        let enc = vec![
            conv(vb.pp("enc0"), 3, 128, 7, 1, 0)?,   // reflect-pad3 done in forward
            conv(vb.pp("enc1"), 128, 256, 3, 1, 1)?, // zero-pad
            conv(vb.pp("enc2"), 256, 512, 3, 2, 1)?,
            conv(vb.pp("enc3"), 512, 1024, 3, 2, 1)?,
        ];
        let blocks = (0..6)
            .map(|b| AdaInBlock::new(vb.pp(format!("block{b}")), 1024))
            .collect::<Result<Vec<_>>>()?;
        let dec = vec![
            conv(vb.pp("dec0"), 1024, 512, 3, 1, 1)?,
            conv(vb.pp("dec1"), 512, 256, 3, 1, 1)?,
            conv(vb.pp("dec2"), 256, 128, 3, 1, 1)?,
        ];
        let out_conv = conv(vb.pp("out_conv"), 128, 3, 7, 1, 0)?; // reflect-pad3 in forward
        Ok(Self { enc, blocks, dec, out_conv, device, dtype })
    }

    /// `target` (1,3,128,128 RGB /255) + `source` (1,512 latent) → (1,3,128,128).
    pub fn forward(&self, target: &Tensor, source: &Tensor) -> Result<Tensor> {
        // Encoder.
        let mut x = reflect_pad2d(target, 3)?;
        x = leaky_relu(&self.enc[0].forward(&x)?)?;
        x = leaky_relu(&self.enc[1].forward(&x)?)?;
        x = leaky_relu(&self.enc[2].forward(&x)?)?;
        x = leaky_relu(&self.enc[3].forward(&x)?)?;
        // Bottleneck.
        for blk in &self.blocks {
            x = blk.forward(&x, source)?;
        }
        // Decoder.
        x = leaky_relu(&self.dec[0].forward(&bilinear_up2x(&x)?)?)?;
        x = leaky_relu(&self.dec[1].forward(&bilinear_up2x(&x)?)?)?;
        x = leaky_relu(&self.dec[2].forward(&x)?)?;
        x = reflect_pad2d(&x, 3)?;
        x = self.out_conv.forward(&x)?.tanh()?;
        // (tanh + 1) / 2 → [0,1].
        Ok(((x + 1.0)? * 0.5)?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_mirrors_without_edge_repeat() {
        let dev = Device::Cpu;
        // row [0,1,2,3], reflect-pad width by 2 → [2,1, 0,1,2,3, 2,1]
        // (reflect requires pad < dim size, as in PyTorch — exercise width only).
        let x = Tensor::from_vec(vec![0f32, 1., 2., 3.], (1, 1, 1, 4), &dev).unwrap();
        let p = reflect_pad_dim(&x, 3, 4, 2).unwrap();
        let row: Vec<f32> = p.i((0, 0, 0)).unwrap().to_vec1().unwrap();
        assert_eq!(row, vec![2., 1., 0., 1., 2., 3., 2., 1.]);
    }

    #[test]
    fn interp_matrix_halfpixel_2x_endpoints() {
        let dev = Device::Cpu;
        let a = interp_matrix(4, &dev, DType::F32).unwrap();
        assert_eq!(a.dims(), &[8, 4]);
        // row 0: src = 0.25-0.5 = -0.25 → clamp 0 → all weight on idx0
        let r0: Vec<f32> = a.i(0).unwrap().to_vec1().unwrap();
        assert!((r0[0] - 1.0).abs() < 1e-6);
    }
}
