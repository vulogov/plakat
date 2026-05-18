// Some helpers stay `pub` for `plakat doctor` and future toolage but
// aren't all called from the main pipeline — silence the warnings.
#![allow(dead_code)]

//! Face-identity models for the FaceID strategies.
//!
//! Three things live here:
//!   1. **InsightFace IR-ResNet50** — the ArcFace backbone (image → 512-d
//!      unit-norm embedding).
//!   2. **`FaceAlignment` + `prepare_face_tensor`** — the bridge from a
//!      photo on disk to the 112×112 RGB tensor ArcFace consumes. Three
//!      alignment modes in priority order: 5-point landmarks (proper
//!      similarity-transform alignment via Umeyama's method), bbox crop,
//!      centre-crop fallback.
//!   3. **`FaceIdEncoder`** — the `IdentityEncoder` impl that combines
//!      the ArcFace backbone, the IP-Adapter-FaceID image-proj MLP, and
//!      an optional SCRFD detector for auto landmark detection.
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
//!   * landmark-aligned for best results (5-point similarity transform).
//!     Centre-crop / bbox-crop also accepted at a quality cost.
//!
//! Output: `(B, 512)` unit-norm face embedding.

use anyhow::Result;
use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, VarBuilder};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::path::Path;

/// Inference flag for `ModuleT::forward_t`. Centralised so all the
/// per-BatchNorm call sites read the same.
const EVAL: bool = false;

/// ArcFace input edge in pixels. InsightFace's training pipeline aligns
/// every face crop to 112×112 via similarity transform from 5 landmarks.
pub const ARCFACE_INPUT: u32 = 112;

/// InsightFace's canonical 5-point reference for 112×112 aligned crops.
/// Order: left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner.
///
/// These are the exact pixel positions ArcFace was trained against —
/// every face crop in the training set was warped to put landmarks at
/// these coordinates. Aligning detected landmarks to this template via
/// a similarity transform (rotation + scale + translation) recovers the
/// last ~15–25% of ArcFace's discriminative power that centre-crop /
/// bbox-crop leave on the table.
///
/// (Source: InsightFace's `face_align.py` — `arcface_dst`.)
pub const ARCFACE_5PT_REF: [[f32; 2]; 5] = [
    [38.2946, 51.6963], // left eye
    [73.5318, 51.5014], // right eye
    [56.0252, 71.7366], // nose tip
    [41.5493, 92.3655], // left mouth corner
    [70.7299, 92.2041], // right mouth corner
];

/// Identifies the order in which a 5-landmark array is expected.
/// Documented as a top-level constant so users authoring scenarios or
/// passing `--face-landmarks` can find it: this is the same convention
/// InsightFace publishes its detection outputs in, so users grabbing
/// landmarks from any InsightFace-derived tool can use them as-is.
pub const LANDMARK_ORDER: &[&str] = &[
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth_corner",
    "right_mouth_corner",
];

/// Which alignment strategy `prepare_face_tensor` should use. Priority
/// from richest to crudest: landmarks > bbox > centre-crop.
#[derive(Clone, Copy, Debug)]
pub enum FaceAlignment {
    /// Resize + centre-crop fallback. Works for tight head-and-shoulders
    /// photos.
    CenterCrop,
    /// Crop to a user-supplied bbox in normalised photo coordinates
    /// `[x0, y0, x1, y1]` before the 112×112 resize.
    Bbox([f32; 4]),
    /// 5-point similarity-transform alignment to ArcFace's canonical
    /// 112×112 reference. The `[[x, y]; 5]` array is in normalised
    /// photo coordinates, ordered per `LANDMARK_ORDER`. Recovers the
    /// last ~15–25% of ArcFace's discriminative power.
    Landmarks([[f32; 2]; 5]),
}

impl FaceAlignment {
    pub fn from_options(
        bbox: Option<[f32; 4]>,
        landmarks: Option<[[f32; 2]; 5]>,
    ) -> Self {
        // Landmarks dominate when both are supplied — they're strictly
        // more informative.
        if let Some(lm) = landmarks {
            Self::Landmarks(lm)
        } else if let Some(b) = bbox {
            Self::Bbox(b)
        } else {
            Self::CenterCrop
        }
    }
}

/// Estimate a 2×3 similarity transform (rotation + uniform scale +
/// translation) that maps `src` onto `dst` in the least-squares sense.
/// Umeyama's method (1991) — produces the best similarity matrix for
/// any number of point pairs ≥ 2.
///
/// Returns `[a, b, tx; c, d, ty]` flat in row-major order (six floats:
/// `[a, b, tx, c, d, ty]`). Apply via `(a*x + b*y + tx, c*x + d*y + ty)`.
fn similarity_transform_2d(src: &[[f32; 2]], dst: &[[f32; 2]]) -> [f32; 6] {
    assert_eq!(src.len(), dst.len(), "src and dst must be same length");
    let n = src.len() as f32;
    debug_assert!(n >= 2.0, "similarity transform needs ≥2 points");

    // Means.
    let (mut sx, mut sy, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..src.len() {
        sx += src[i][0];
        sy += src[i][1];
        dx += dst[i][0];
        dy += dst[i][1];
    }
    let (sx_m, sy_m) = (sx / n, sy / n);
    let (dx_m, dy_m) = (dx / n, dy / n);

    // Centered points + cross-covariance H (2×2).
    let mut h = [[0.0f32; 2]; 2];
    let mut var_src = 0.0f32;
    for i in 0..src.len() {
        let sxi = src[i][0] - sx_m;
        let syi = src[i][1] - sy_m;
        let dxi = dst[i][0] - dx_m;
        let dyi = dst[i][1] - dy_m;
        // H = sum( src_centered_i^T @ dst_centered_i )
        h[0][0] += sxi * dxi;
        h[0][1] += sxi * dyi;
        h[1][0] += syi * dxi;
        h[1][1] += syi * dyi;
        var_src += sxi * sxi + syi * syi;
    }

    // 2×2 SVD via direct formulas (closed form). H = U Σ Vᵀ.
    // Reference: https://scicomp.stackexchange.com/a/14710
    let (a, b, c, d) = (h[0][0], h[0][1], h[1][0], h[1][1]);
    let e = (a + d) * 0.5;
    let f = (a - d) * 0.5;
    let g = (c + b) * 0.5;
    let q = (c - b) * 0.5;
    let r1 = (e * e + q * q).sqrt();
    let r2 = (f * f + g * g).sqrt();
    let sx_sv = r1 + r2;
    let sy_sv = (r1 - r2).max(0.0);
    let a1 = q.atan2(e);
    let a2 = g.atan2(f);
    let theta = (a1 - a2) * 0.5;
    let phi = (a1 + a2) * 0.5;
    // U = R(phi) reflected by sign(d_det), V = R(theta).
    let det_h = a * d - b * c;
    let sign = if det_h < 0.0 { -1.0 } else { 1.0 };
    let (cp, sp) = (phi.cos(), phi.sin());
    let (ct, st) = (theta.cos(), theta.sin());
    let u = [[cp, -sp * sign], [sp, cp * sign]]; // 2×2 U with reflection fix
    let v = [[ct, -st], [st, ct]]; // 2×2 V
    let s_diag = [sx_sv, sy_sv * sign];

    // Rotation R = U @ Vᵀ.
    let r = [
        [
            u[0][0] * v[0][0] + u[0][1] * v[0][1],
            u[0][0] * v[1][0] + u[0][1] * v[1][1],
        ],
        [
            u[1][0] * v[0][0] + u[1][1] * v[0][1],
            u[1][0] * v[1][0] + u[1][1] * v[1][1],
        ],
    ];

    // Scale c = sum(Σ) / var_src.
    let scale = if var_src > 0.0 {
        (s_diag[0] + s_diag[1]) / var_src
    } else {
        1.0
    };

    // Final 2×3: [scale*R | dst_mean - scale*R @ src_mean].
    let m00 = scale * r[0][0];
    let m01 = scale * r[0][1];
    let m10 = scale * r[1][0];
    let m11 = scale * r[1][1];
    let tx = dx_m - (m00 * sx_m + m01 * sy_m);
    let ty = dy_m - (m10 * sx_m + m11 * sy_m);
    [m00, m01, tx, m10, m11, ty]
}

/// Bilinear warp: produce an `out_w × out_h` RGB image by sampling
/// `src` at `inv_affine([dst_x, dst_y])`. `inv_affine` is the inverse
/// of the forward transform — we typically build forward dst-from-src
/// then invert before calling this.
fn bilinear_warp(
    src: &image::RgbImage,
    inv_affine: [f32; 6],
    out_w: u32,
    out_h: u32,
) -> image::RgbImage {
    let (sw, sh) = (src.width() as f32, src.height() as f32);
    let mut out = image::RgbImage::new(out_w, out_h);
    for dy in 0..out_h {
        for dx in 0..out_w {
            // Apply inverse affine to dst coords → src coords.
            let fx = inv_affine[0] * dx as f32
                + inv_affine[1] * dy as f32
                + inv_affine[2];
            let fy = inv_affine[3] * dx as f32
                + inv_affine[4] * dy as f32
                + inv_affine[5];
            // Bilinear sample, clamping outside coords to edge pixels
            // (avoids black borders when alignment over-extends).
            let x0 = fx.floor().clamp(0.0, sw - 1.0);
            let y0 = fy.floor().clamp(0.0, sh - 1.0);
            let x1 = (x0 + 1.0).min(sw - 1.0);
            let y1 = (y0 + 1.0).min(sh - 1.0);
            let ax = (fx - x0).clamp(0.0, 1.0);
            let ay = (fy - y0).clamp(0.0, 1.0);
            let p00 = src.get_pixel(x0 as u32, y0 as u32).0;
            let p10 = src.get_pixel(x1 as u32, y0 as u32).0;
            let p01 = src.get_pixel(x0 as u32, y1 as u32).0;
            let p11 = src.get_pixel(x1 as u32, y1 as u32).0;
            let mut pix = [0u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - ax) + p10[c] as f32 * ax;
                let bot = p01[c] as f32 * (1.0 - ax) + p11[c] as f32 * ax;
                let v = top * (1.0 - ay) + bot * ay;
                pix[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(dx, dy, image::Rgb(pix));
        }
    }
    out
}

/// Invert a 2×3 affine `[a, b, tx, c, d, ty]`.
fn invert_affine_2x3(a: [f32; 6]) -> [f32; 6] {
    let det = a[0] * a[4] - a[1] * a[3];
    debug_assert!(det.abs() > 1e-12, "near-singular similarity transform");
    let inv_det = 1.0 / det;
    let i00 = a[4] * inv_det;
    let i01 = -a[1] * inv_det;
    let i10 = -a[3] * inv_det;
    let i11 = a[0] * inv_det;
    let i_tx = -(i00 * a[2] + i01 * a[5]);
    let i_ty = -(i10 * a[2] + i11 * a[5]);
    [i00, i01, i_tx, i10, i11, i_ty]
}

/// Align a face image to ArcFace's canonical 112×112 template via 5-point
/// similarity transform. `landmarks` are in **pixel coordinates** within
/// the source image (left_eye, right_eye, nose, left_mouth, right_mouth).
/// Returns the aligned 112×112 RGB image.
fn align_to_arcface_template(
    src: &image::RgbImage,
    landmarks_px: [[f32; 2]; 5],
) -> image::RgbImage {
    // Forward transform: source landmarks → ArcFace reference.
    let src_pts: Vec<[f32; 2]> = landmarks_px.to_vec();
    let dst_pts: Vec<[f32; 2]> = ARCFACE_5PT_REF.to_vec();
    let forward = similarity_transform_2d(&src_pts, &dst_pts);
    // For backward sampling (dst → src), invert.
    let inverse = invert_affine_2x3(forward);
    bilinear_warp(src, inverse, ARCFACE_INPUT, ARCFACE_INPUT)
}

/// Load a photo and produce the 112×112 RGB tensor ArcFace's IR-ResNet50
/// expects, using the richest alignment available.
///
/// Alignment priority (richest first):
///   * `FaceAlignment::Landmarks` — 5-point similarity transform to
///     ArcFace's canonical template. The right way to align —
///     recovers the last ~15–25% of ArcFace's discriminative power vs
///     cruder alignment.
///   * `FaceAlignment::Bbox` — user-supplied bbox. Crops to the bbox,
///     then resizes to 112×112. No rotation/scale correction; better
///     than centre-crop on non-centred photos.
///   * `FaceAlignment::CenterCrop` — shorter-side resize + centre-crop.
///     Falls back when no better alignment supplied.
///
/// All paths use InsightFace's `(x − 127.5) / 127.5` normalisation.
/// Returns `(1, 3, 112, 112)`.
pub fn prepare_face_tensor(
    photo_path: &Path,
    alignment: FaceAlignment,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img_rgb = image::open(photo_path)?.to_rgb8();
    let img = DynamicImage::ImageRgb8(img_rgb.clone());
    let (w, h) = img.dimensions();

    let aligned_rgb = match alignment {
        FaceAlignment::Landmarks(lm_norm) => {
            // Normalised → pixel coords. The aligner does the warp
            // straight to 112×112 — no intermediate resize.
            let lm_px: [[f32; 2]; 5] = std::array::from_fn(|i| {
                [
                    lm_norm[i][0] * w as f32,
                    lm_norm[i][1] * h as f32,
                ]
            });
            align_to_arcface_template(&img_rgb, lm_px)
        }
        FaceAlignment::Bbox([x0, y0, x1, y1]) => {
            // Crop to bbox (clamped to image bounds), then resize.
            let px0 = (x0.clamp(0.0, 1.0) * w as f32).floor() as u32;
            let py0 = (y0.clamp(0.0, 1.0) * h as f32).floor() as u32;
            let px1 = (x1.clamp(0.0, 1.0) * w as f32).ceil() as u32;
            let py1 = (y1.clamp(0.0, 1.0) * h as f32).ceil() as u32;
            let bw = px1.saturating_sub(px0).max(1).min(w - px0);
            let bh = py1.saturating_sub(py0).max(1).min(h - py0);
            img.crop_imm(px0, py0, bw, bh)
                .resize_exact(ARCFACE_INPUT, ARCFACE_INPUT, FilterType::CatmullRom)
                .to_rgb8()
        }
        FaceAlignment::CenterCrop => {
            // Shorter-side resize to 2 × 112 = 224 (breathing room),
            // then centre-crop, then resize.
            let target_short: u32 = ARCFACE_INPUT * 2;
            let (rw, rh) = if w < h {
                let s = target_short;
                (s, ((h as f32) * (s as f32) / (w as f32)).round() as u32)
            } else {
                let s = target_short;
                (((w as f32) * (s as f32) / (h as f32)).round() as u32, s)
            };
            let resized = img.resize_exact(rw, rh, FilterType::CatmullRom);
            let cx = rw.saturating_sub(target_short) / 2;
            let cy = rh.saturating_sub(target_short) / 2;
            resized
                .crop_imm(cx, cy, target_short, target_short)
                .resize_exact(ARCFACE_INPUT, ARCFACE_INPUT, FilterType::CatmullRom)
                .to_rgb8()
        }
    };

    // InsightFace normalisation: x ∈ [0, 255] → (x − 127.5) / 127.5 ∈ [−1, 1].
    // Channel-first: RGB → (1, 3, 112, 112).
    let n = ARCFACE_INPUT as usize;
    let mut data: Vec<f32> = Vec::with_capacity(3 * n * n);
    for c in 0..3usize {
        for y in 0..n {
            for x in 0..n {
                let px = aligned_rgb.get_pixel(x as u32, y as u32).0[c];
                data.push((px as f32 - 127.5) / 127.5);
            }
        }
    }
    let t = Tensor::from_vec(data, (1, 3, n, n), device)?.to_dtype(dtype)?;
    Ok(t)
}

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
// IBasicBlock — BN-fused pre-activation residual block.
//
// Most ONNX exports of iresnet50 fuse `BatchNorm` into the preceding
// `Conv2d`, so the deployed weights have:
//   * a single per-block `bn1` (pre-activation — can't be fused away)
//   * biased `conv1` / `conv2` (each absorbed a post-conv BN)
//   * a biased `downsample` conv (no separate BN tensor)
//
//   identity = downsample(x) or x   (downsample is a single biased conv)
//   out = conv2( prelu( conv1( bn1(x) ) ) )
//   out = out + identity
//
// Verified against the `arcface_r50.safetensors` produced by the
// `onnx2torch` + `safetensors.torch` conversion path in PERSONA.md
// (see PERSONA.md, FaceID setup Route A). 263 tensors total, 16 per first-stage
// block + 14 per subsequent.
// =====================================================================

struct IBasicBlock {
    bn1: BatchNorm,
    conv1: Conv2d,
    prelu: PRelu,
    conv2: Conv2d,
    /// `Some` when in/out channels differ OR stride > 1. Flat single
    /// conv (with bias) — the post-downsample BN was folded into the
    /// conv's bias during ONNX export.
    downsample: Option<Conv2d>,
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
        // conv1 / conv2 have bias because they absorbed bn2 / bn3.
        let conv1 = candle_nn::conv2d(
            in_channels,
            out_channels,
            3,
            conv1_cfg,
            vs.pp("conv1"),
        )?;
        let prelu = PRelu::new(vs.pp("prelu"), out_channels)?;
        let conv2_cfg = Conv2dConfig {
            stride,
            padding: 1,
            ..Default::default()
        };
        let conv2 = candle_nn::conv2d(
            out_channels,
            out_channels,
            3,
            conv2_cfg,
            vs.pp("conv2"),
        )?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let cfg = Conv2dConfig {
                stride,
                padding: 0,
                ..Default::default()
            };
            // Single biased conv at `downsample.{weight,bias}` (no
            // Sequential[conv, bn] nesting). The post-downsample BN
            // was fused into this conv's bias.
            Some(candle_nn::conv2d(
                in_channels,
                out_channels,
                1,
                cfg,
                vs.pp("downsample"),
            )?)
        } else {
            None
        };

        Ok(Self {
            bn1,
            conv1,
            prelu,
            conv2,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        let h = self.bn1.forward_t(x, EVAL)?;
        let h = self.conv1.forward(&h)?;
        let h = self.prelu.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        Ok((h + identity)?)
    }
}

// =====================================================================
// IR-ResNet50 — the ArcFace backbone.
// =====================================================================

/// InsightFace IR-ResNet50 (BN-fused deployment variant), layer counts
/// `[3, 4, 14, 3]`. Produces a 512-d L2-normalised face embedding from
/// a 112×112 RGB face crop. See `IBasicBlock` for the fusion model.
pub struct IResnet50 {
    conv1: Conv2d,
    layer1: Vec<IBasicBlock>,
    layer2: Vec<IBasicBlock>,
    layer3: Vec<IBasicBlock>,
    layer4: Vec<IBasicBlock>,
    bn2: BatchNorm,
    fc: Linear,
    features: BatchNorm,
}

impl IResnet50 {
    /// Build from a `VarBuilder` rooted at a fused IR-ResNet50
    /// state dict (typical ONNX → safetensors export). Expected key
    /// layout (≈263 tensors total):
    ///   conv1.{weight,bias},
    ///   layer{1..4}.<i>.{bn1.<…>,conv1.{weight,bias},prelu.weight,conv2.{weight,bias}},
    ///   layer<X>.0.downsample.{weight,bias}   (first block of each stage),
    ///   bn2.<…>, fc.{weight,bias}, features.<…>
    ///
    /// (No top-level `bn1` / `prelu` — the stem's BN folded into
    /// `conv1.bias`. No per-block `bn2` / `bn3` — those folded into
    /// `conv1.bias` / `conv2.bias` respectively. No `downsample.<idx>`
    /// nesting — the post-downsample BN folded into the conv's bias.)
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let bn_cfg = BatchNormConfig::default();
        let conv1_cfg = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        // Biased stem conv. The bn1 + prelu that originally followed
        // were either fused into this bias or dropped during export —
        // either way, no top-level activations.
        let conv1 = candle_nn::conv2d(3, 64, 3, conv1_cfg, vs.pp("conv1"))?;

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
        // Stem: just the biased conv1 — no bn / prelu at this level
        // (they got folded out during export).
        let mut x = self.conv1.forward(x)?;

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
// FaceIdEncoder — combines IR-ResNet50 + IP-Adapter-FaceID image-proj.
//
// Combines IR-ResNet50 (this file) with `ImageProj` (existing IP-Adapter
// projection) — exactly the shape FaceID needs:
//     ArcFace(112×112×3) → 512-d → image-proj(512 → 4 × cross_attn_dim)
// =====================================================================

/// IP-Adapter-FaceID image-proj: a 2-layer MLP + LayerNorm. Maps a
/// 512-d ArcFace embedding to `num_tokens × cross_attn_dim` cross-
/// attention tokens.
///
/// Architecture (h94's `MLPProjModel`):
///
/// ```text
///   x (B, 512)
///     │
///     ▼ Linear(512, 1024)           ← proj.0.weight / proj.0.bias
///     ▼ GELU                        ← (no params; PyTorch index 1)
///     ▼ Linear(1024, T × D)         ← proj.2.weight / proj.2.bias
///     ▼ reshape (B, T, D)
///     ▼ LayerNorm(D)                ← norm.weight / norm.bias
///   tokens (B, T, D)
/// ```
///
/// where `T = num_tokens` (4 for FaceID) and `D = cross_attn_dim`
/// (768 SD 1.5, 2048 SDXL). Hidden width is fixed at `2 × embedding_dim`
/// per the reference.
///
/// Distinct from `ip_adapter::ImageProj` (which is just `Linear → LN`).
/// ArcFace's 512-d identity vector needs the extra MLP capacity to
/// project meaningfully into the cross-attention space — basic
/// IP-Adapter's single Linear isn't enough.
pub struct FaceIdImageProj {
    proj_0: Linear,
    proj_2: Linear,
    norm: candle_nn::LayerNorm,
    num_tokens: usize,
    cross_attn_dim: usize,
}

impl FaceIdImageProj {
    /// Load from a PyTorch `.bin` state dict rooted at a sub-key
    /// (`image_proj` in h94's FaceID `.bin`).
    pub fn load_from_pth_subtree(
        weights: &Path,
        state_key: &str,
        embedding_dim: usize,
        cross_attn_dim: usize,
        num_tokens: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_pth_with_state(weights, dtype, state_key, device)?;
        // Hidden width = 2 × embedding_dim per the reference impl.
        let hidden = embedding_dim * 2;
        let proj_0 = candle_nn::linear(
            embedding_dim,
            hidden,
            vb.pp("proj").pp("0"),
        )?;
        // `proj.1` is the GELU activation (no params); skip to `.2`.
        let proj_2 = candle_nn::linear(
            hidden,
            num_tokens * cross_attn_dim,
            vb.pp("proj").pp("2"),
        )?;
        let norm = candle_nn::layer_norm(
            cross_attn_dim,
            1e-5,
            vb.pp("norm"),
        )?;
        Ok(Self {
            proj_0,
            proj_2,
            norm,
            num_tokens,
            cross_attn_dim,
        })
    }

    /// `(B, embedding_dim)` → `(B, num_tokens, cross_attn_dim)`.
    pub fn forward(&self, embedding: &Tensor) -> Result<Tensor> {
        let b = embedding.dim(0)?;
        let h = self.proj_0.forward(embedding)?;
        let h = h.gelu()?;
        let h = self.proj_2.forward(&h)?;
        let h = h.reshape((b, self.num_tokens, self.cross_attn_dim))?;
        Ok(self.norm.forward(&h)?)
    }
}

/// Combined ArcFace + FaceID image-proj encoder, plus an optional
/// SCRFD detector that auto-fills 5-point landmarks when the caller
/// hasn't supplied any.
pub struct FaceIdEncoder {
    arcface: IResnet50,
    image_proj: FaceIdImageProj,
    /// Optional face detector. `Some` when `PLAKAT_SCRFD_WEIGHTS` is set
    /// and weights load successfully; auto-fills landmarks for ArcFace
    /// alignment. `None` falls back to centre-crop / user-supplied bbox /
    /// user-supplied landmarks.
    detector: Option<crate::pipelines::scrfd::SCRFDDetector>,
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
    ///   (antelopev2 / buffalo_l bundle). See PERSONA.md FaceID setup.
    /// * `faceid_weights` — `h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15`
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
        scrfd_weights: Option<&std::path::Path>,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[arcface_weights], dtype, device)?
        };
        let arcface = IResnet50::new(vb)?;
        // h94's FaceID image-proj lives in a PyTorch `.bin` under the
        // `image_proj.*` subtree — same convention as the per-variant
        // helpers below. No safetensors variant exists for FaceID's MLP.
        let image_proj = FaceIdImageProj::load_from_pth_subtree(
            faceid_weights,
            "image_proj",
            512,
            cross_attn_dim,
            4,
            device,
            dtype,
        )?;
        let detector = load_scrfd_detector_opt(scrfd_weights, device, dtype)?;
        Ok(Self {
            arcface,
            image_proj,
            detector,
            device: device.clone(),
            dtype,
        })
    }

    /// Specialised constructor for SD 1.5 FaceID — the path
    /// `IdentityKind::FaceId.load_encoder` takes.
    ///
    /// `arcface_weights` is a safetensors file (user-supplied via
    /// `PLAKAT_ARCFACE_WEIGHTS` or `PLAKAT_ARCFACE_HF`).
    /// `faceid_weights_pth` is the h94 `ip-adapter-faceid_sd15.bin`
    /// PyTorch file — we read just the `image_proj.*` subtree out of it
    /// (the file also contains UNet LoRA weights under `ip_adapter.*`
    /// handled separately by the LoRA merger).
    /// `scrfd_weights` is the pre-resolved SCRFD safetensors path
    /// (already downloaded if `PLAKAT_SCRFD_HF` was used); `None`
    /// means no auto-detection — falls back to manual alignment paths.
    pub fn load_faceid_sd15(
        arcface_weights: &Path,
        faceid_weights_pth: &Path,
        scrfd_weights: Option<&Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::load_faceid_with_dim(
            arcface_weights,
            faceid_weights_pth,
            768,
            scrfd_weights,
            device,
            dtype,
        )
    }

    /// Specialised constructor for SDXL FaceID — the path
    /// `IdentityKind::FaceIdSdxl.load_encoder` takes. Same ArcFace
    /// backbone as SD 1.5 (so the same env-var-supplied weights file
    /// works); differs only in the image-proj output dim (2048 vs 768)
    /// and the FaceID `.bin` file consumed.
    pub fn load_faceid_sdxl(
        arcface_weights: &Path,
        faceid_weights_pth: &Path,
        scrfd_weights: Option<&Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::load_faceid_with_dim(
            arcface_weights,
            faceid_weights_pth,
            2048,
            scrfd_weights,
            device,
            dtype,
        )
    }

    fn load_faceid_with_dim(
        arcface_weights: &Path,
        faceid_weights_pth: &Path,
        cross_attn_dim: usize,
        scrfd_weights: Option<&Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[arcface_weights], dtype, device)?
        };
        let arcface = IResnet50::new(vb)?;
        let image_proj = FaceIdImageProj::load_from_pth_subtree(
            faceid_weights_pth,
            "image_proj",
            512,
            cross_attn_dim,
            4,
            device,
            dtype,
        )?;
        let detector = load_scrfd_detector_opt(scrfd_weights, device, dtype)?;
        Ok(Self {
            arcface,
            image_proj,
            detector,
            device: device.clone(),
            dtype,
        })
    }

    /// Given an aligned 112×112 RGB crop (pre-normalised to roughly
    /// `[-1, 1]`), produce `(1, 4, cross_attn_dim)` identity tokens
    /// ready for concatenation onto the text-token sequence.
    pub fn encode_aligned(&self, aligned: &Tensor) -> Result<Tensor> {
        let embedding = self.arcface.forward(aligned)?;
        self.image_proj.forward(&embedding)
    }

    /// Photo → identity tokens. `alignment` picks how the 112×112 crop
    /// is produced (see `FaceAlignment`). Landmarks give the best
    /// quality, bbox is the middle option, centre-crop is the fallback.
    pub fn encode_photo(
        &self,
        photo_path: &Path,
        alignment: FaceAlignment,
    ) -> Result<Tensor> {
        let aligned =
            prepare_face_tensor(photo_path, alignment, &self.device, self.dtype)?;
        self.encode_aligned(&aligned)
    }

    pub fn num_tokens(&self) -> usize {
        // Matches what `ImageProj::load` was called with — kept in sync with
        // `FaceIdEncoder::load`'s constant `4`.
        4
    }
}

impl crate::pipelines::ip_adapter::IdentityEncoder for FaceIdEncoder {
    fn num_tokens(&self) -> usize {
        FaceIdEncoder::num_tokens(self)
    }

    fn encode(
        &self,
        photo_path: &Path,
        opts: crate::pipelines::ip_adapter::EncodeOptions,
    ) -> Result<Tensor> {
        if !photo_path.exists() {
            return Err(anyhow::anyhow!(
                "persona photo not found: {} (resolved from current working \
                 directory). Check the path and re-run.",
                photo_path.display()
            ));
        }
        // Alignment priority:
        //   1. Manual landmarks (caller-supplied — overrides everything)
        //   2. SCRFD-detected landmarks (auto-fill, if detector loaded)
        //   3. Manual bbox (caller-supplied)
        //   4. Centre-crop fallback
        let detected_landmarks = match (&opts.face_landmarks, &self.detector) {
            (Some(_), _) => None, // user supplied; don't override
            (None, Some(det)) => {
                let s = crate::ui::progress::spinner("SCRFD: detecting face");
                match det.detect_primary_normalised(photo_path) {
                    Ok(Some(lm)) => {
                        s.finish_with_message("✓ SCRFD landmarks ready");
                        Some(lm)
                    }
                    Ok(None) => {
                        s.finish_with_message(
                            "⚠ SCRFD found no face — falling back to bbox/centre-crop",
                        );
                        None
                    }
                    Err(e) => {
                        s.finish_with_message(format!(
                            "⚠ SCRFD failed ({e}) — falling back to bbox/centre-crop"
                        ));
                        None
                    }
                }
            }
            _ => None,
        };
        let landmarks = opts.face_landmarks.or(detected_landmarks);
        let alignment = FaceAlignment::from_options(opts.face_bbox, landmarks);
        self.encode_photo(photo_path, alignment)
    }
}

/// Construct an SCRFD detector from a pre-resolved safetensors path,
/// or return `None` if no path was provided. Sync — the async download
/// resolution happens at the `IdentityKind::load_encoder` layer; this
/// helper only does the model construction.
fn load_scrfd_detector_opt(
    path: Option<&Path>,
    device: &Device,
    dtype: DType,
) -> Result<Option<crate::pipelines::scrfd::SCRFDDetector>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let s = crate::ui::progress::spinner("Loading SCRFD face detector");
    let det = crate::pipelines::scrfd::SCRFDDetector::load(
        path,
        crate::pipelines::scrfd::SCRFDConfig::default(),
        device,
        dtype,
    )?;
    s.finish_with_message("✓ SCRFD ready");
    Ok(Some(det))
}
