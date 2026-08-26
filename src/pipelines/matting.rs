//! Content-aware background matting (U2Net) — "smart" cut-outs that don't need
//! a chroma backdrop. A salient-object network predicts an alpha matte from
//! image content, so a photoreal/painted subject can be lifted off any
//! background. Used by `plakat transparent --matte` and the artefact library.
//!
//! Model: `jamino30/u2net-saliency` → `u2net-duts-msra.safetensors` — full
//! U2NET, **MIT-licensed, ungated**, F32 safetensors loadable by candle. Weight
//! keys carry a `module.` prefix and use the refactored `enc/mid/dec/convs/
//! lastconv` module layout (not the textbook `stage1.rebnconv1...`).

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Module, ModuleT, VarBuilder, ops,
};
use image::{RgbImage, RgbaImage};
use std::path::Path;

const SIZE: usize = 320;
/// Full U2NET converted from `Carve/u2net-universal` (Apache-2.0; verified to
/// fire, d0 max 1.0). candle can't read Carve's legacy-pickle `.pth`, so it is
/// re-serialised to safetensors (see `scripts/convert_u2net_to_safetensors.py`).
const WEIGHTS_FILE: &str = "u2net-universal.safetensors";
/// HF repo hosting the redistributed safetensors (auto-downloaded on first use).
const MATTE_REPO: &str = "vulogov98/u2net-universal";

// ---- REBNCONV: Conv2d(k3, dilation=d, pad=d, bias) → BatchNorm → ReLU ----
struct RebnConv {
    conv: Conv2d,
    bn: BatchNorm,
}

impl RebnConv {
    fn load(vb: VarBuilder, in_ch: usize, out_ch: usize, dilation: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: dilation,
            dilation,
            ..Default::default()
        };
        let conv = candle_nn::conv2d(in_ch, out_ch, 3, cfg, vb.pp("conv_s1"))?;
        let bn = candle_nn::batch_norm(out_ch, BatchNormConfig::default(), vb.pp("bn_s1"))?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?; // eval: use running stats
        Ok(x.relu()?)
    }
}

fn pool(x: &Tensor) -> Result<Tensor> {
    Ok(x.max_pool2d(2)?) // 2×2 stride 2 (dims stay even from 320 down to 10)
}

fn up(x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    Ok(x.upsample_bilinear2d(h, w, false)?) // align_corners=false (U2Net)
}

// ---- RSU block (height L; `dilated` = the RSU4F variant: no pool/upsample) ----
struct Rsu {
    conv: RebnConv,    // rebnconvin
    enc: Vec<RebnConv>, // enc[0..=L-2]
    mid: RebnConv,     // bottleneck (rebnconv{L})
    dec: Vec<RebnConv>, // dec[0..=L-2] (dec[0] deepest → dec[L-2] outputs out_ch)
    height: usize,
    dilated: bool,
}

impl Rsu {
    fn load(
        vb: VarBuilder,
        height: usize,
        in_ch: usize,
        mid_ch: usize,
        out_ch: usize,
        dilated: bool,
    ) -> Result<Self> {
        let conv = RebnConv::load(vb.pp("rebnconvin"), in_ch, out_ch, 1)?;
        let mut enc = Vec::new();
        for i in 0..(height - 1) {
            let ci = if i == 0 { out_ch } else { mid_ch };
            let d = if dilated { 1 << i } else { 1 };
            enc.push(RebnConv::load(
                vb.pp(format!("rebnconv{}", i + 1)),
                ci,
                mid_ch,
                d,
            )?);
        }
        let mid_d = if dilated { 1 << (height - 1) } else { 2 };
        let mid = RebnConv::load(vb.pp(format!("rebnconv{height}")), mid_ch, mid_ch, mid_d)?;
        let mut dec = Vec::new();
        for k in 0..(height - 1) {
            let co = if k == height - 2 { out_ch } else { mid_ch };
            let d = if dilated { 1 << (height - 2 - k) } else { 1 };
            dec.push(RebnConv::load(
                vb.pp(format!("rebnconv{}d", height - 1 - k)),
                mid_ch * 2,
                co,
                d,
            )?);
        }
        Ok(Self {
            conv,
            enc,
            mid,
            dec,
            height,
            dilated,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xin = self.conv.forward(x)?;
        let h = self.height;
        let mut enc_outs: Vec<Tensor> = Vec::with_capacity(h - 1);
        enc_outs.push(self.enc[0].forward(&xin)?);
        for i in 1..(h - 1) {
            let inp = if self.dilated {
                enc_outs[i - 1].clone()
            } else {
                pool(&enc_outs[i - 1])?
            };
            enc_outs.push(self.enc[i].forward(&inp)?);
        }
        let mut d = self.mid.forward(&enc_outs[h - 2])?;
        for k in 0..(h - 1) {
            let skip = &enc_outs[h - 2 - k];
            let cat = Tensor::cat(&[&d, skip], 1)?;
            d = self.dec[k].forward(&cat)?;
            if !self.dilated && k < h - 2 {
                let nxt = &enc_outs[h - 3 - k];
                d = up(&d, nxt.dim(2)?, nxt.dim(3)?)?;
            }
        }
        Ok((xin + d)?)
    }
}

// ---- full U2NET ----
struct U2Net {
    enc: Vec<Rsu>,    // En_1..En_6
    dec: Vec<Rsu>,    // De_5..De_1
    side: Vec<Conv2d>, // convs.0..5 (from De_1,De_2,De_3,De_4,De_5,En_6)
    outconv: Conv2d,  // lastconv: 6→1, 1×1
}

impl U2Net {
    fn load(vb: VarBuilder) -> Result<Self> {
        // (height, in_ch, mid_ch, out_ch, dilated)
        let enc_cfg = [
            (7usize, 3usize, 32usize, 64usize, false), // En_1 RSU7
            (6, 64, 32, 128, false),                   // En_2 RSU6
            (5, 128, 64, 256, false),                  // En_3 RSU5
            (4, 256, 128, 512, false),                 // En_4 RSU4
            (4, 512, 256, 512, true),                  // En_5 RSU4F
            (4, 512, 256, 512, true),                  // En_6 RSU4F
        ];
        let dec_cfg = [
            (4usize, 1024usize, 256usize, 512usize, true), // De_5 RSU4F
            (4, 1024, 128, 256, false),                    // De_4 RSU4
            (5, 512, 64, 128, false),                      // De_3 RSU5
            (6, 256, 32, 64, false),                       // De_2 RSU6
            (7, 128, 16, 64, false),                       // De_1 RSU7
        ];
        let mut enc = Vec::new();
        for (i, &(hh, ic, mc, oc, dl)) in enc_cfg.iter().enumerate() {
            enc.push(Rsu::load(vb.pp(format!("stage{}", i + 1)), hh, ic, mc, oc, dl)?);
        }
        let mut dec = Vec::new();
        for (i, &(hh, ic, mc, oc, dl)) in dec_cfg.iter().enumerate() {
            dec.push(Rsu::load(vb.pp(format!("stage{}d", 5 - i)), hh, ic, mc, oc, dl)?);
        }
        // side convs (3×3, pad 1): channels = the stage they read from.
        let side_ch = [64usize, 64, 128, 256, 512, 512];
        let scfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut side = Vec::new();
        for (i, &c) in side_ch.iter().enumerate() {
            side.push(candle_nn::conv2d(c, 1, 3, scfg, vb.pp(format!("side{}", i + 1)))?);
        }
        let outconv = candle_nn::conv2d(6, 1, 1, Conv2dConfig::default(), vb.pp("outconv"))?;
        Ok(Self {
            enc,
            dec,
            side,
            outconv,
        })
    }

    /// Returns the fused saliency map `d0` (sigmoid applied), shape `[1,1,SIZE,SIZE]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let e1 = self.enc[0].forward(x)?;
        let e2 = self.enc[1].forward(&pool(&e1)?)?;
        let e3 = self.enc[2].forward(&pool(&e2)?)?;
        let e4 = self.enc[3].forward(&pool(&e3)?)?;
        let e5 = self.enc[4].forward(&pool(&e4)?)?;
        let e6 = self.enc[5].forward(&pool(&e5)?)?;

        let hw = |t: &Tensor| -> Result<(usize, usize)> { Ok((t.dim(2)?, t.dim(3)?)) };
        let (h5, w5) = hw(&e5)?;
        let d5 = self.dec[0]
            .forward(&Tensor::cat(&[&up(&e6, h5, w5)?, &e5], 1)?)?;
        let (h4, w4) = hw(&e4)?;
        let d4 = self.dec[1]
            .forward(&Tensor::cat(&[&up(&d5, h4, w4)?, &e4], 1)?)?;
        let (h3, w3) = hw(&e3)?;
        let d3 = self.dec[2]
            .forward(&Tensor::cat(&[&up(&d4, h3, w3)?, &e3], 1)?)?;
        let (h2, w2) = hw(&e2)?;
        let d2 = self.dec[3]
            .forward(&Tensor::cat(&[&up(&d3, h2, w2)?, &e2], 1)?)?;
        let (h1, w1) = hw(&e1)?;
        let d1 = self.dec[4]
            .forward(&Tensor::cat(&[&up(&d2, h1, w1)?, &e1], 1)?)?;

        let (ih, iw) = hw(x)?;
        let s1 = self.side[0].forward(&d1)?;
        let s2 = up(&self.side[1].forward(&d2)?, ih, iw)?;
        let s3 = up(&self.side[2].forward(&d3)?, ih, iw)?;
        let s4 = up(&self.side[3].forward(&d4)?, ih, iw)?;
        let s5 = up(&self.side[4].forward(&d5)?, ih, iw)?;
        let s6 = up(&self.side[5].forward(&e6)?, ih, iw)?;
        let fused = self
            .outconv
            .forward(&Tensor::cat(&[&s1, &s2, &s3, &s4, &s5, &s6], 1)?)?;
        Ok(ops::sigmoid(&fused)?)
    }
}

// ---- preprocessing / postprocessing ----

fn preprocess(img: &RgbImage, device: &Device) -> Result<Tensor> {
    let r = image::imageops::resize(
        img,
        SIZE as u32,
        SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    // U2Net divides by the per-image max, then ImageNet-normalizes.
    let mut mx = 1f32;
    for p in r.pixels() {
        for c in 0..3 {
            mx = mx.max(p.0[c] as f32);
        }
    }
    if mx <= 0.0 {
        mx = 1.0;
    }
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut data = vec![0f32; 3 * SIZE * SIZE];
    for (i, p) in r.pixels().enumerate() {
        for c in 0..3 {
            data[c * SIZE * SIZE + i] = (p.0[c] as f32 / mx - mean[c]) / std[c];
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, SIZE, SIZE), device)?)
}

/// `d0` → a min/max-normalized grayscale alpha resized to the original WxH.
fn matte_alpha(d0: &Tensor, w: u32, h: u32) -> Result<image::GrayImage> {
    let raw: Vec<f32> = d0.flatten_all()?.to_vec1()?;
    let (mut mn, mut mxv) = (f32::MAX, f32::MIN);
    for &v in &raw {
        mn = mn.min(v);
        mxv = mxv.max(v);
    }
    let range = (mxv - mn).max(1e-6);
    if std::env::var("PLAKAT_MATTE_DEBUG").is_ok() {
        let mean: f32 = raw.iter().sum::<f32>() / raw.len() as f32;
        let over_half = raw.iter().filter(|&&v| v > 0.5).count();
        eprintln!(
            "[matte] d0 min={mn:.4} max={mxv:.4} mean={mean:.4} range={range:.4} frac>0.5={:.3}",
            over_half as f32 / raw.len() as f32
        );
    }
    let mut g = image::GrayImage::new(SIZE as u32, SIZE as u32);
    for (i, &v) in raw.iter().enumerate() {
        let a = (((v - mn) / range) * 255.0).clamp(0.0, 255.0) as u8;
        g.put_pixel((i % SIZE) as u32, (i / SIZE) as u32, image::Luma([a]));
    }
    Ok(image::imageops::resize(
        &g,
        w,
        h,
        image::imageops::FilterType::Triangle,
    ))
}

fn matte_bbox(img: &RgbaImage, thresh: u8) -> Option<(u32, u32, u32, u32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    let mut any = false;
    for (x, y, p) in img.enumerate_pixels() {
        if p.0[3] > thresh {
            any = true;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    any.then(|| (x0, y0, x1 - x0 + 1, y1 - y0 + 1))
}

/// Resolve the matte weights: `PLAKAT_MATTE_WEIGHTS` (a safetensors path) wins,
/// else a locally-converted file in the plakat cache (`~/.cache/plakat/u2net/`),
/// else download the redistributed safetensors from HF.
async fn matte_weights_path() -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var("PLAKAT_MATTE_WEIGHTS") {
        return Ok(p.into());
    }
    let base = std::env::var("HOME").unwrap_or_default();
    let local = std::path::PathBuf::from(base)
        .join(".cache/plakat/u2net")
        .join(WEIGHTS_FILE);
    if local.exists() {
        return Ok(local);
    }
    crate::hf::download::get_file(MATTE_REPO, WEIGHTS_FILE)
        .await
        .with_context(|| format!("downloading matte weights {MATTE_REPO}/{WEIGHTS_FILE}"))
}

/// Smart cut-out: predict the foreground matte, write it as the alpha channel,
/// optionally crop to the subject's bounding box. Output must keep alpha
/// (`.png` / `.webp`).
/// Predict the U2Net salient-object alpha matte for an image. Returns the RGB image (at native
/// resolution) and a single-channel alpha (`GrayImage`, 255 = foreground). Shared by [`cutout`] and
/// `plakat replace-bg`.
pub async fn matte(in_path: &Path, device: &Device) -> Result<(image::RgbImage, image::GrayImage)> {
    let weights = matte_weights_path().await?;
    let img = image::open(in_path)?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device)
            .context("loading U2Net safetensors")?
    };
    let net = U2Net::load(vb)?;
    let x = preprocess(&img, device)?;
    let d0 = net.forward(&x)?;
    let alpha = matte_alpha(&d0, w, h)?;
    Ok((img, alpha))
}

pub async fn cutout(in_path: &Path, out_path: &Path, crop: bool, device: &Device) -> Result<()> {
    if let Some(ext) = out_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
    {
        if matches!(ext.as_str(), "jpg" | "jpeg" | "bmp") {
            return Err(anyhow!(
                "output .{ext} doesn't support alpha — use a .png or .webp output path"
            ));
        }
    }

    let (img, alpha) = matte(in_path, device).await?;
    let (w, h) = (img.width(), img.height());
    if std::env::var("PLAKAT_MATTE_DEBUG").is_ok() {
        let dbg = out_path.with_extension("matte.png");
        let _ = alpha.save(&dbg);
        eprintln!("[matte] raw matte → {}", dbg.display());
    }

    let mut out = RgbaImage::new(w, h);
    for (x0, y0, p) in img.enumerate_pixels() {
        let a = alpha.get_pixel(x0, y0).0[0];
        out.put_pixel(x0, y0, image::Rgba([p.0[0], p.0[1], p.0[2], a]));
    }
    if crop {
        if let Some((cx, cy, cw, ch)) = matte_bbox(&out, 16) {
            out = image::imageops::crop_imm(&out, cx, cy, cw, ch).to_image();
        }
    }

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    out.save(out_path)?;
    Ok(())
}

/// Separable box blur (moving average, radius `r`) on an `w×h` f32 buffer. Edge-clamped.
fn box_blur_f32(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut s, mut c) = (0f32, 0f32);
            for dx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                s += src[y * w + dx];
                c += 1.0;
            }
            tmp[y * w + x] = s / c;
        }
    }
    let mut out = vec![0f32; w * h];
    for x in 0..w {
        for y in 0..h {
            let (mut s, mut c) = (0f32, 0f32);
            for dy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                s += tmp[dy * w + x];
                c += 1.0;
            }
            out[y * w + x] = s / c;
        }
    }
    out
}

/// **Foreground colour decontamination** (RFC SEAMS-1 Q1): a matte's semi-transparent EDGE pixels are a
/// blend of subject + the OLD background, so compositing them onto a NEW background shows the old bg as a
/// colour fringe/halo. Estimate the clean foreground colour with an **alpha-weighted (premultiplied)
/// blur** — each pixel becomes the alpha-weighted average of its neighbourhood, so solid-foreground
/// pixels dominate and the bg-tinted low-alpha pixels contribute little. Only edge pixels (`0<a<~1`) are
/// replaced; solid interior and full transparency are untouched. Pure. `radius` ≈ the matte edge width.
pub fn decontaminate(fg: &RgbImage, alpha: &image::GrayImage, radius: u32) -> RgbImage {
    let (w, h) = fg.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let r = radius.max(1) as usize;
    let a: Vec<f32> = alpha.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    let mut prem = [vec![0f32; wu * hu], vec![0f32; wu * hu], vec![0f32; wu * hu]];
    for (i, px) in fg.pixels().enumerate() {
        for c in 0..3 {
            prem[c][i] = px.0[c] as f32 * a[i]; // premultiplied
        }
    }
    let den = box_blur_f32(&a, wu, hu, r);
    let num: Vec<Vec<f32>> = (0..3).map(|c| box_blur_f32(&prem[c], wu, hu, r)).collect();
    let mut out = fg.clone();
    for (i, px) in out.pixels_mut().enumerate() {
        if a[i] > 0.0 && a[i] < 0.98 && den[i] > 1e-3 {
            for c in 0..3 {
                px.0[c] = (num[c][i] / den[i]).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// **Matte edge refinement** (RFC SEAMS-1 Q4): sharpen a soft/uncertain U2Net matte edge back toward a
/// confident 0/1 while keeping a thin anti-aliased ramp. In the uncertain **trimap band** (mid-alpha), a
/// smoothstep contrast around 0.5 pulls hair/fur edges crisp; solid interior (`≥hi`) and clear background
/// (`≤lo`) are locked. Pure. Improves every downstream cut-out (replace-bg / remove / product).
pub fn refine_matte(alpha: &image::GrayImage) -> image::GrayImage {
    let (lo, hi) = (0.15f32, 0.85f32);
    let mut out = alpha.clone();
    for p in out.pixels_mut() {
        let a = p.0[0] as f32 / 255.0;
        let refined = if a <= lo {
            0.0
        } else if a >= hi {
            1.0
        } else {
            let t = (a - lo) / (hi - lo); // remap the band to [0,1]
            t * t * (3.0 - 2.0 * t) // smoothstep → crisper edge, still anti-aliased
        };
        p.0[0] = (refined * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};

    #[test]
    fn decontaminate_removes_old_bg_spill_at_the_edge() {
        // A red subject (left) faded over a GREEN old background (right). The mid alpha-ramp column is
        // contaminated green; decontamination must pull it back toward the subject red.
        let (w, h) = (16u32, 8u32);
        let mut fg = RgbImage::new(w, h);
        let mut al = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = if x < 6 { 255 } else if x < 10 { 128 } else { 0 };
                // observed = a*red + (1-a)*green  (edge pixels look olive/green-tinted)
                let af = a as f32 / 255.0;
                let obs = |fgc: f32, bgc: f32| (fgc * af + bgc * (1.0 - af)).round() as u8;
                fg.put_pixel(x, y, Rgb([obs(200.0, 0.0), obs(0.0, 200.0), obs(0.0, 0.0)]));
                al.put_pixel(x, y, Luma([a]));
            }
        }
        let before_g = fg.get_pixel(7, 4).0[1] as i32; // green contamination at the edge
        let clean = decontaminate(&fg, &al, 3);
        let after_g = clean.get_pixel(7, 4).0[1] as i32;
        let after_r = clean.get_pixel(7, 4).0[0] as i32;
        assert!(after_g < before_g - 20, "green spill reduced at the edge ({before_g} → {after_g})");
        assert!(after_r > after_g, "edge colour pulled toward the red subject");
        // Solid interior is untouched.
        assert_eq!(clean.get_pixel(2, 4).0, fg.get_pixel(2, 4).0, "solid fg unchanged");
    }

    #[test]
    fn refine_matte_crisps_the_band_and_locks_extremes() {
        let mut al = GrayImage::new(5, 1);
        for (x, v) in [10u8, 60, 128, 200, 245].iter().enumerate() {
            al.put_pixel(x as u32, 0, Luma([*v]));
        }
        let r = refine_matte(&al);
        assert_eq!(r.get_pixel(0, 0).0[0], 0, "clear bg → 0");
        assert_eq!(r.get_pixel(4, 0).0[0], 255, "solid fg → 255");
        assert!((r.get_pixel(2, 0).0[0] as i32 - 128).abs() <= 3, "mid stays ~mid (smoothstep 0.5)");
        assert!(r.get_pixel(1, 0).0[0] < 60, "low band pushed toward 0");
        assert!(r.get_pixel(3, 0).0[0] > 200, "high band pushed toward 1");
    }
}
