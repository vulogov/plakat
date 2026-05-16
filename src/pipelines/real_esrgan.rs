//! Real-ESRGAN super-resolution.
//!
//! RRDBNet architecture from xinntao/Real-ESRGAN, ported to candle.
//! Three canonical variants:
//!   * x2plus       — 23 RRDB blocks, scale=2
//!   * x4plus       — 23 RRDB blocks, scale=4
//!   * x4plus_anime — 6 RRDB blocks, scale=4 (faster, optimized for line art)
//!
//! All variants share the same outer architecture; only `num_block` and
//! `scale` differ. Weights live on HuggingFace as flat safetensors with the
//! `body.<i>.rdb<j>.conv<k>.{weight,bias}` key convention (matching the
//! original PyTorch state_dict).

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct RealEsrganConfig {
    pub num_in_ch: usize,
    pub num_out_ch: usize,
    pub scale: usize, // 2 or 4
    pub num_feat: usize,
    pub num_block: usize,
    pub num_grow_ch: usize,
}

impl RealEsrganConfig {
    pub fn x2plus() -> Self {
        Self {
            num_in_ch: 3,
            num_out_ch: 3,
            scale: 2,
            num_feat: 64,
            num_block: 23,
            num_grow_ch: 32,
        }
    }
    pub fn x4plus() -> Self {
        Self {
            num_in_ch: 3,
            num_out_ch: 3,
            scale: 4,
            num_feat: 64,
            num_block: 23,
            num_grow_ch: 32,
        }
    }
    pub fn x4plus_anime_6b() -> Self {
        Self {
            num_in_ch: 3,
            num_out_ch: 3,
            scale: 4,
            num_feat: 64,
            num_block: 6,
            num_grow_ch: 32,
        }
    }
}

// LeakyReLU(0.2) used everywhere in Real-ESRGAN.
fn lrelu(x: &Tensor) -> Result<Tensor> {
    let neg = (x * 0.2_f64)?;
    Ok(x.maximum(&neg)?)
}

/// PyTorch's `pixel_unshuffle`: (B, C, H, W) → (B, C·s², H/s, W/s).
fn pixel_unshuffle(t: &Tensor, scale: usize) -> Result<Tensor> {
    let (b, c, h, w) = t.dims4()?;
    if h % scale != 0 || w % scale != 0 {
        anyhow::bail!(
            "pixel_unshuffle: dims {}x{} not divisible by scale {}",
            h,
            w,
            scale
        );
    }
    let (h_new, w_new) = (h / scale, w / scale);
    Ok(t.reshape((b, c, h_new, scale, w_new, scale))?
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c * scale * scale, h_new, w_new))?)
}

fn nearest_x2(t: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = t.dims4()?;
    Ok(t.upsample_nearest2d(h * 2, w * 2)?)
}

#[derive(Debug)]
struct ResidualDenseBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
}

impl ResidualDenseBlock {
    fn new(vs: VarBuilder, num_feat: usize, num_grow_ch: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv1: candle_nn::conv2d(num_feat, num_grow_ch, 3, cfg, vs.pp("conv1"))?,
            conv2: candle_nn::conv2d(num_feat + num_grow_ch, num_grow_ch, 3, cfg, vs.pp("conv2"))?,
            conv3: candle_nn::conv2d(
                num_feat + 2 * num_grow_ch,
                num_grow_ch,
                3,
                cfg,
                vs.pp("conv3"),
            )?,
            conv4: candle_nn::conv2d(
                num_feat + 3 * num_grow_ch,
                num_grow_ch,
                3,
                cfg,
                vs.pp("conv4"),
            )?,
            conv5: candle_nn::conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, cfg, vs.pp("conv5"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = lrelu(&self.conv1.forward(x)?)?;
        let x2 = lrelu(&self.conv2.forward(&Tensor::cat(&[x, &x1], 1)?)?)?;
        let x3 = lrelu(&self.conv3.forward(&Tensor::cat(&[x, &x1, &x2], 1)?)?)?;
        let x4 = lrelu(&self.conv4.forward(&Tensor::cat(&[x, &x1, &x2, &x3], 1)?)?)?;
        let x5 = self
            .conv5
            .forward(&Tensor::cat(&[x, &x1, &x2, &x3, &x4], 1)?)?;
        Ok(((x5 * 0.2_f64)? + x)?)
    }
}

#[derive(Debug)]
struct Rrdb {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl Rrdb {
    fn new(vs: VarBuilder, num_feat: usize, num_grow_ch: usize) -> Result<Self> {
        Ok(Self {
            rdb1: ResidualDenseBlock::new(vs.pp("rdb1"), num_feat, num_grow_ch)?,
            rdb2: ResidualDenseBlock::new(vs.pp("rdb2"), num_feat, num_grow_ch)?,
            rdb3: ResidualDenseBlock::new(vs.pp("rdb3"), num_feat, num_grow_ch)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.rdb1.forward(x)?;
        let h = self.rdb2.forward(&h)?;
        let h = self.rdb3.forward(&h)?;
        Ok(((h * 0.2_f64)? + x)?)
    }
}

/// The full RRDBNet model.
#[derive(Debug)]
pub struct Model {
    conv_first: Conv2d,
    body: Vec<Rrdb>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    scale: usize,
}

impl Model {
    pub fn load(weights: &Path, cfg: &RealEsrganConfig, device: &Device) -> Result<Self> {
        // Real-ESRGAN weights are usually F32; F16 would lose precision in
        // the residual scaling (0.2 multipliers) and produce muddy output.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)
                .with_context(|| format!("loading Real-ESRGAN weights {}", weights.display()))?
        };
        Self::new(vb, cfg)
    }

    pub fn new(vs: VarBuilder, cfg: &RealEsrganConfig) -> Result<Self> {
        let cfg3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        // For scale ≤ 2 the first conv sees pixel-unshuffled input — more
        // channels but a smaller spatial extent. (scale=1 → ×16 channels,
        // scale=2 → ×4 channels.)
        let first_in_ch = match cfg.scale {
            1 => cfg.num_in_ch * 16,
            2 => cfg.num_in_ch * 4,
            _ => cfg.num_in_ch,
        };
        let conv_first =
            candle_nn::conv2d(first_in_ch, cfg.num_feat, 3, cfg3, vs.pp("conv_first"))?;
        let body_vs = vs.pp("body");
        let mut body = Vec::with_capacity(cfg.num_block);
        for i in 0..cfg.num_block {
            body.push(Rrdb::new(
                body_vs.pp(i.to_string()),
                cfg.num_feat,
                cfg.num_grow_ch,
            )?);
        }
        let conv_body = candle_nn::conv2d(cfg.num_feat, cfg.num_feat, 3, cfg3, vs.pp("conv_body"))?;
        let conv_up1 = candle_nn::conv2d(cfg.num_feat, cfg.num_feat, 3, cfg3, vs.pp("conv_up1"))?;
        let conv_up2 = candle_nn::conv2d(cfg.num_feat, cfg.num_feat, 3, cfg3, vs.pp("conv_up2"))?;
        let conv_hr = candle_nn::conv2d(cfg.num_feat, cfg.num_feat, 3, cfg3, vs.pp("conv_hr"))?;
        let conv_last =
            candle_nn::conv2d(cfg.num_feat, cfg.num_out_ch, 3, cfg3, vs.pp("conv_last"))?;
        Ok(Self {
            conv_first,
            body,
            conv_body,
            conv_up1,
            conv_up2,
            conv_hr,
            conv_last,
            scale: cfg.scale,
        })
    }

    /// Forward pass. Input shape (1, 3, H, W) with values in [0, 1].
    /// Output shape (1, 3, H*scale, W*scale) — may have small over/undershoot
    /// that the caller should clamp before quantizing to u8.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feat = match self.scale {
            2 => pixel_unshuffle(x, 2)?,
            1 => pixel_unshuffle(x, 4)?,
            _ => x.clone(),
        };
        let feat = self.conv_first.forward(&feat)?;

        // body — 23 (or 6) RRDB blocks
        let mut h = feat.clone();
        for rrdb in &self.body {
            h = rrdb.forward(&h)?;
        }
        let body_out = self.conv_body.forward(&h)?;
        let feat = (&feat + body_out)?;

        // The upsample structure is fixed at ×4 in pixel space (two ×2
        // stages). The `scale` field controls whether we compressed the
        // input first via pixel_unshuffle — so effective net scale is
        // 4 / unshuffle_factor:
        //   scale 4 → unshuffle 1 → net ×4
        //   scale 2 → unshuffle 2 → net ×2
        //   scale 1 → unshuffle 4 → net ×1
        let feat = lrelu(&self.conv_up1.forward(&nearest_x2(&feat)?)?)?;
        let feat = lrelu(&self.conv_up2.forward(&nearest_x2(&feat)?)?)?;
        let feat = lrelu(&self.conv_hr.forward(&feat)?)?;
        Ok(self.conv_last.forward(&feat)?)
    }

    /// Run end-to-end on an image file. Loads, normalizes to [0, 1],
    /// runs the model, clamps, writes the result. Returns
    /// (in_w, in_h, out_w, out_h).
    pub fn upscale_file(
        &self,
        in_path: &Path,
        out_path: &Path,
        device: &Device,
    ) -> Result<(u32, u32, u32, u32)> {
        let img = image::open(in_path)
            .with_context(|| format!("opening {}", in_path.display()))?
            .to_rgb8();
        let (w, h) = (img.width(), img.height());

        // Build (1, 3, H, W) F32 tensor with values in [0, 1].
        let mut data: Vec<f32> = Vec::with_capacity(3 * (w * h) as usize);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    data.push(img.get_pixel(x, y).0[c] as f32 / 255.0);
                }
            }
        }
        let input = Tensor::from_vec(data, (1, 3, h as usize, w as usize), device)?;

        let out = self.forward(&input)?;
        let out = out.clamp(0.0_f32, 1.0_f32)?;
        let (_, _, oh, ow) = out.dims4()?;
        let out_u8 = (out * 255.0)?.to_dtype(DType::U8)?.i(0)?.permute((1, 2, 0))?;
        let buf = out_u8.flatten_all()?.to_vec1::<u8>()?;

        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        crate::imaging::io::save_rgb_u8(&buf, ow as u32, oh as u32, out_path)?;
        Ok((w, h, ow as u32, oh as u32))
    }
}

/// Variant → (HF repo, config). The repos at `hlky/...` ship safetensors;
/// the `config.json` in each repo matches the canonical xinntao values.
pub fn variant_repo_and_config(variant: Variant) -> (&'static str, RealEsrganConfig) {
    match variant {
        Variant::X2Plus => ("hlky/RealESRGAN_x2plus", RealEsrganConfig::x2plus()),
        Variant::X4Plus => ("hlky/RealESRGAN_x4plus", RealEsrganConfig::x4plus()),
        Variant::X4AnimeB6 => (
            "hlky/RealESRGAN_x4plus_anime_6B",
            RealEsrganConfig::x4plus_anime_6b(),
        ),
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Variant {
    X2Plus,
    X4Plus,
    X4AnimeB6,
}

impl Variant {
    #[allow(dead_code)]
    pub fn scale(self) -> usize {
        match self {
            Self::X2Plus => 2,
            Self::X4Plus | Self::X4AnimeB6 => 4,
        }
    }
}
