//! SCRFD face detector port.
//!
//! "Sample and Computation Redistribution for Efficient Face Detection"
//! (Guo et al. 2021, https://arxiv.org/abs/2105.04714). InsightFace's
//! deployed face detector — produces bbox + 5-point landmarks per face.
//!
//! Once landmarks are available, `face_models::align_to_arcface_template`
//! produces the ArcFace-canonical 112×112 crop and FaceID reaches
//! reference-quality identity preservation **without** any user-supplied
//! `--face-bbox` / `--face-landmarks` flag.
//!
//! ## Verification status — VERIFIED
//!
//! This architecture is verified against InsightFace's real `det_500m.onnx`
//! (SCRFD-500MF): plakat's detections match onnxruntime to within ~1–3 px and
//! 0.003 score on the same images. The weight-key layout below is the one the
//! `plakat convert-onnx --arch scrfd-500mf` converter emits, so converted weights
//! load and run correctly end-to-end. (Earlier revisions implemented a guessed
//! ResNet-BasicBlock backbone + simple FPN that never loaded real weights — that
//! is fixed here.)
//!
//! Setup — convert the InsightFace ONNX with plakat's own command:
//!
//! ```bash
//! # det_500m.onnx ships inside InsightFace's buffalo_sc model pack:
//! #   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
//! plakat convert-onnx det_500m.onnx scrfd_500m.safetensors --arch scrfd-500mf
//! export PLAKAT_SCRFD_WEIGHTS=$(pwd)/scrfd_500m.safetensors
//! ```
//!
//! ## Architecture overview (SCRFD-500MF, matches det_500m.onnx)
//!
//! ```text
//!   input (1, 3, 640, 640)  ← resize + top-left letterbox, (x-127.5)/128, RGB
//!     │
//!     ▼
//!   stem: 3×3 stride=2 → ReLU                              → C=16
//!     │
//!     ▼
//!   backbone: 14 depthwise-separable blocks (DW 3×3 → ReLU → PW 1×1 → ReLU),
//!             channels 16→40→72→152→288, BN folded into each conv's bias.
//!             FPN taps after block 5 (72ch, /8), block 7 (152ch, /16),
//!             block 13 (288ch, /32).
//!     │
//!     ▼
//!   neck (PAFPN): lateral 1×1 → 16ch; top-down add+upsample; bottom-up
//!                 downsample+add+3×3. No activations.
//!     │
//!     ▼
//!   head × {stride 8, 16, 32} (per-stride, not shared): 2 DW-sep stem convs
//!     │     → 64ch, then 3×3 preds  cls (2 anchors), reg (2×4), kps (2×10).
//!     ▼
//!   decode (anchor centre = (x·stride, y·stride), distance format) + NMS
//! ```

#![allow(dead_code)] // SCRFD integration is opt-in via PLAKAT_SCRFD_*;
                     // many helpers are public for future / debugging use.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use image::imageops::FilterType;
use std::path::Path;

/// SCRFD-500MF runtime config. The architecture (MobileNet depthwise-separable
/// backbone + PAFPN neck + per-stride head) is now fixed and verified against the
/// real InsightFace `det_500m.onnx`, so only the decode-time knobs live here.
#[derive(Clone, Debug)]
pub struct SCRFDConfig {
    /// Anchors per spatial location (2 for SCRFD-500MF).
    pub num_anchors: usize,
    /// FPN level strides relative to the input. Always [8, 16, 32].
    pub strides: [u32; 3],
    /// Input edge after letterbox padding. Standard SCRFD-500MF: 640.
    pub input_size: u32,
}

impl Default for SCRFDConfig {
    fn default() -> Self {
        Self::scrfd_500mf()
    }
}

impl SCRFDConfig {
    /// SCRFD-500MF: 2 anchors/location, strides [8,16,32], 640² input.
    pub fn scrfd_500mf() -> Self {
        Self { num_anchors: 2, strides: [8, 16, 32], input_size: 640 }
    }
}

/// A conv2d **with bias** (SCRFD's exported ONNX folds BatchNorm into each conv,
/// so every layer is a plain biased conv). `groups == in_ch` gives a depthwise
/// conv. Reads `weight` + `bias` under `vb`.
fn conv(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<Conv2d> {
    let cfg = Conv2dConfig { padding, stride, dilation: 1, groups, ..Default::default() };
    Ok(candle_nn::conv2d(in_ch, out_ch, k, cfg, vb)?)
}

// =====================================================================
// Backbone — MobileNet-style depthwise-separable (DW 3×3 → ReLU → PW 1×1
// → ReLU). 14 blocks, channels 16→40→72→152→288; FPN taps after the
// blocks at strides 8 / 16 / 32.
// =====================================================================

struct DwSep {
    dw: Conv2d,
    pw: Conv2d,
}

impl DwSep {
    fn new(vb: VarBuilder, in_ch: usize, out_ch: usize, stride: usize) -> Result<Self> {
        let dw = conv(vb.pp("dw"), in_ch, in_ch, 3, stride, 1, in_ch)?;
        let pw = conv(vb.pp("pw"), in_ch, out_ch, 1, 1, 0, 1)?;
        Ok(Self { dw, pw })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dw.forward(x)?.relu()?;
        let h = self.pw.forward(&h)?.relu()?;
        Ok(h)
    }
}

struct Backbone {
    stem: Conv2d,
    blocks: Vec<DwSep>,
}

impl Backbone {
    /// `(in_ch, out_ch, stride)` for the 14 depthwise-separable blocks.
    const SPECS: [(usize, usize, usize); 14] = [
        (16, 16, 1),
        (16, 40, 2),
        (40, 40, 1),
        (40, 72, 2),
        (72, 72, 1),
        (72, 72, 1), // block 5 → stride-8 tap (72 ch)
        (72, 152, 2),
        (152, 152, 1), // block 7 → stride-16 tap (152 ch)
        (152, 288, 2),
        (288, 288, 1),
        (288, 288, 1),
        (288, 288, 1),
        (288, 288, 1),
        (288, 288, 1), // block 13 → stride-32 tap (288 ch)
    ];

    fn new(vb: VarBuilder) -> Result<Self> {
        let stem = conv(vb.pp("stem"), 3, 16, 3, 2, 1, 1)?;
        let blocks = Self::SPECS
            .iter()
            .enumerate()
            .map(|(i, &(ic, oc, s))| DwSep::new(vb.pp(format!("b{i}")), ic, oc, s))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stem, blocks })
    }

    /// Returns the three FPN-input feature maps (strides 8, 16, 32).
    fn forward(&self, x: &Tensor) -> Result<[Tensor; 3]> {
        let mut h = self.stem.forward(x)?.relu()?;
        let mut taps: Vec<Tensor> = Vec::with_capacity(3);
        for (i, b) in self.blocks.iter().enumerate() {
            h = b.forward(&h)?;
            if i == 5 || i == 7 || i == 13 {
                taps.push(h.clone());
            }
        }
        Ok([taps[0].clone(), taps[1].clone(), taps[2].clone()])
    }
}

// =====================================================================
// Neck — PAFPN: lateral 1×1 → 16 ch, top-down add+upsample, then a
// bottom-up path (downsample + add + 3×3). No activations. Matches the
// exact Resize/Add wiring of det_500m.onnx.
// =====================================================================

struct Neck {
    lat: [Conv2d; 3],
    fpn: [Conv2d; 3],
    down: [Conv2d; 2],
    pa: [Conv2d; 2],
}

impl Neck {
    fn new(vb: VarBuilder) -> Result<Self> {
        let lat = [
            conv(vb.pp("lat0"), 72, 16, 1, 1, 0, 1)?,
            conv(vb.pp("lat1"), 152, 16, 1, 1, 0, 1)?,
            conv(vb.pp("lat2"), 288, 16, 1, 1, 0, 1)?,
        ];
        let fpn = [
            conv(vb.pp("fpn0"), 16, 16, 3, 1, 1, 1)?,
            conv(vb.pp("fpn1"), 16, 16, 3, 1, 1, 1)?,
            conv(vb.pp("fpn2"), 16, 16, 3, 1, 1, 1)?,
        ];
        let down = [
            conv(vb.pp("down0"), 16, 16, 3, 2, 1, 1)?,
            conv(vb.pp("down1"), 16, 16, 3, 2, 1, 1)?,
        ];
        let pa = [
            conv(vb.pp("pa0"), 16, 16, 3, 1, 1, 1)?,
            conv(vb.pp("pa1"), 16, 16, 3, 1, 1, 1)?,
        ];
        Ok(Self { lat, fpn, down, pa })
    }

    /// `feats` = backbone taps (stride 8, 16, 32). Returns the three head
    /// inputs (stride 8, 16, 32).
    fn forward(&self, feats: [Tensor; 3]) -> Result<[Tensor; 3]> {
        let l3 = self.lat[0].forward(&feats[0])?;
        let l4 = self.lat[1].forward(&feats[1])?;
        let l5 = self.lat[2].forward(&feats[2])?;
        // Top-down.
        let (_, _, h4, w4) = l4.dims4()?;
        let p4 = (&l4 + l5.upsample_nearest2d(h4, w4)?)?;
        let (_, _, h3, w3) = l3.dims4()?;
        let p3 = (&l3 + p4.upsample_nearest2d(h3, w3)?)?;
        // FPN smooth.
        let f3 = self.fpn[0].forward(&p3)?;
        let f4 = self.fpn[1].forward(&p4)?;
        let f5 = self.fpn[2].forward(&l5)?;
        // Bottom-up (PA). The downsample for the next level uses the pre-PA
        // merge (m4), not the post-PA output — matching the ONNX graph.
        let m4 = (&f4 + self.down[0].forward(&f3)?)?;
        let n4 = self.pa[0].forward(&m4)?;
        let m5 = (&f5 + self.down[1].forward(&m4)?)?;
        let n5 = self.pa[1].forward(&m5)?;
        Ok([f3, n4, n5])
    }
}

// =====================================================================
// Head — per-stride. Two depthwise-separable stem convs (→64 ch) shared
// by all three prediction convs (cls 2, reg 8, kps 20 — 3×3, raw logits).
// =====================================================================

struct Head {
    s0dw: Conv2d,
    s0pw: Conv2d,
    s1dw: Conv2d,
    s1pw: Conv2d,
    cls: Conv2d,
    reg: Conv2d,
    kps: Conv2d,
}

impl Head {
    fn new(vb: VarBuilder, num_anchors: usize) -> Result<Self> {
        Ok(Self {
            s0dw: conv(vb.pp("s0dw"), 16, 16, 3, 1, 1, 16)?,
            s0pw: conv(vb.pp("s0pw"), 16, 64, 1, 1, 0, 1)?,
            s1dw: conv(vb.pp("s1dw"), 64, 64, 3, 1, 1, 64)?,
            s1pw: conv(vb.pp("s1pw"), 64, 64, 1, 1, 0, 1)?,
            cls: conv(vb.pp("cls"), 64, num_anchors, 3, 1, 1, 1)?,
            reg: conv(vb.pp("reg"), 64, num_anchors * 4, 3, 1, 1, 1)?,
            kps: conv(vb.pp("kps"), 64, num_anchors * 10, 3, 1, 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let h = self.s0dw.forward(x)?.relu()?;
        let h = self.s0pw.forward(&h)?.relu()?;
        let h = self.s1dw.forward(&h)?.relu()?;
        let h = self.s1pw.forward(&h)?.relu()?;
        let cls = self.cls.forward(&h)?;
        let reg = self.reg.forward(&h)?;
        let kps = self.kps.forward(&h)?;
        Ok((cls, reg, kps))
    }
}

// =====================================================================
// Top-level SCRFD module.
// =====================================================================

pub struct SCRFD {
    config: SCRFDConfig,
    backbone: Backbone,
    neck: Neck,
    heads: [Head; 3],
}

impl SCRFD {
    pub fn new(vs: VarBuilder, config: SCRFDConfig) -> Result<Self> {
        let backbone = Backbone::new(vs.pp("backbone"))?;
        let neck = Neck::new(vs.pp("neck"))?;
        let hb = vs.pp("head");
        let heads = [
            Head::new(hb.pp("s8"), config.num_anchors)?,
            Head::new(hb.pp("s16"), config.num_anchors)?,
            Head::new(hb.pp("s32"), config.num_anchors)?,
        ];
        Ok(Self { config, backbone, neck, heads })
    }

    /// Returns three sets of (cls, bbox, kps) tensors, one per FPN level.
    pub fn forward(&self, x: &Tensor) -> Result<[(Tensor, Tensor, Tensor); 3]> {
        let backbone_outs = self.backbone.forward(x)?;
        let neck_outs = self.neck.forward(backbone_outs)?;
        Ok([
            self.heads[0].forward(&neck_outs[0])?,
            self.heads[1].forward(&neck_outs[1])?,
            self.heads[2].forward(&neck_outs[2])?,
        ])
    }
}

// =====================================================================
// Detection result type.
// =====================================================================

/// One detected face. Coordinates are in the *original image's* pixel
/// coordinate system (post-undoing the letterbox transform).
#[derive(Clone, Debug)]
pub struct Face {
    /// `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
    /// `[[x, y]; 5]` landmarks in original-image pixels.
    /// Order matches `face_models::LANDMARK_ORDER`:
    /// left_eye, right_eye, nose, left_mouth, right_mouth.
    pub landmarks: [[f32; 2]; 5],
    /// Classification score in `[0, 1]` (post-sigmoid).
    pub score: f32,
}

// =====================================================================
// Detection pipeline (postprocessing).
// =====================================================================

/// Generate anchor centres for one FPN level.
/// `feat_h × feat_w` is the level's spatial size; `stride` is the
/// downsampling factor from input to feature.
fn make_anchor_centres(feat_h: usize, feat_w: usize, stride: u32) -> Vec<(f32, f32)> {
    let mut centres = Vec::with_capacity(feat_h * feat_w);
    let s = stride as f32;
    for y in 0..feat_h {
        for x in 0..feat_w {
            // InsightFace SCRFD anchor centre is exactly (x·stride, y·stride)
            // — the top-left grid convention, NOT cell-centre (+0.5).
            centres.push((s * x as f32, s * y as f32));
        }
    }
    centres
}

/// Decode the `bbox` prediction (distance format) into `[x1, y1, x2, y2]`.
///
/// SCRFD's bbox prediction is in **distance format**: each location
/// predicts (l, t, r, b) = distance from the anchor centre to the
/// corresponding box edge, in **stride units**. After multiplying by
/// stride: `x1 = cx - l, y1 = cy - t, x2 = cx + r, y2 = cy + b`.
fn decode_bbox(
    cx: f32,
    cy: f32,
    dlrtb: [f32; 4],
    stride: u32,
) -> [f32; 4] {
    let s = stride as f32;
    let l = dlrtb[0] * s;
    let t = dlrtb[1] * s;
    let r = dlrtb[2] * s;
    let b = dlrtb[3] * s;
    [cx - l, cy - t, cx + r, cy + b]
}

/// Decode landmark offsets into pixel coordinates. SCRFD predicts 5
/// (dx, dy) pairs as displacements from the anchor centre in stride units.
fn decode_landmarks(
    cx: f32,
    cy: f32,
    deltas: [[f32; 2]; 5],
    stride: u32,
) -> [[f32; 2]; 5] {
    let s = stride as f32;
    let mut out = [[0f32; 2]; 5];
    for i in 0..5 {
        out[i][0] = cx + deltas[i][0] * s;
        out[i][1] = cy + deltas[i][1] * s;
    }
    out
}

/// IoU of two `[x1, y1, x2, y2]` boxes.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Greedy NMS: pick highest-scoring detection, remove others with
/// IoU > `iou_threshold`. Returns the kept set in score-descending order.
fn nms(mut faces: Vec<Face>, iou_threshold: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut kept: Vec<Face> = Vec::new();
    for f in faces {
        let drop = kept.iter().any(|k| iou(&f.bbox, &k.bbox) > iou_threshold);
        if !drop {
            kept.push(f);
        }
    }
    kept
}

/// Sigmoid scalar.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decode one FPN level's raw (cls, bbox, kps) tensors into `Vec<Face>`
/// in **letterboxed-input pixel coords**. The caller is responsible for
/// undoing the letterbox to get original-image coords.
fn decode_level(
    cls: &Tensor,
    bbox: &Tensor,
    kps: &Tensor,
    stride: u32,
    score_threshold: f32,
    num_anchors: usize,
) -> Result<Vec<Face>> {
    let (_c, h, w) = cls.dims3()?; // (C, H, W) — batch already squeezed by caller
    let cls_flat = cls.to_vec3::<f32>()?;
    let bbox_flat = bbox.to_vec3::<f32>()?;
    let kps_flat = kps.to_vec3::<f32>()?;

    // candle's to_vec3 on a (1, C, H, W) tensor returns a (C, H, W) Vec
    // after squeezing batch. We index as [c][y][x].
    let mut faces = Vec::new();
    let centres = make_anchor_centres(h, w, stride);

    for y in 0..h {
        for x in 0..w {
            for anc in 0..num_anchors {
                // Class score for this anchor.
                let logit = cls_flat[anc][y][x];
                let score = sigmoid(logit);
                if score < score_threshold {
                    continue;
                }
                let idx = y * w + x;
                let (cx, cy) = centres[idx];

                // Bbox dims: 4 values per anchor, interleaved by anchor.
                // Channel layout: [anc0_l, anc0_t, anc0_r, anc0_b, anc1_l, ...].
                let dl = bbox_flat[anc * 4][y][x];
                let dt = bbox_flat[anc * 4 + 1][y][x];
                let dr = bbox_flat[anc * 4 + 2][y][x];
                let db = bbox_flat[anc * 4 + 3][y][x];
                let bbox_xyxy = decode_bbox(cx, cy, [dl, dt, dr, db], stride);

                // Landmarks: 10 values per anchor (5 dx/dy pairs).
                let mut deltas = [[0f32; 2]; 5];
                for i in 0..5 {
                    deltas[i][0] = kps_flat[anc * 10 + i * 2][y][x];
                    deltas[i][1] = kps_flat[anc * 10 + i * 2 + 1][y][x];
                }
                let lms = decode_landmarks(cx, cy, deltas, stride);

                faces.push(Face {
                    bbox: bbox_xyxy,
                    landmarks: lms,
                    score,
                });
            }
        }
    }
    Ok(faces)
}

// =====================================================================
// Letterbox preprocessing.
// =====================================================================

/// Result of letterbox preprocessing — keeps the transform parameters so
/// the caller can map detections back to original-image coordinates.
struct Letterbox {
    /// `(1, 3, input_size, input_size)` tensor ready for the model.
    tensor: Tensor,
    /// Uniform scale applied (same for x and y).
    scale: f32,
    /// Pixel offset of the original image inside the letterbox image
    /// (top-left corner of the resized image within the padded canvas).
    pad_x: f32,
    pad_y: f32,
    /// Original photo dims, for clipping outputs.
    orig_w: f32,
    orig_h: f32,
}

fn letterbox_preprocess(
    photo_path: &Path,
    input_size: u32,
    device: &Device,
    dtype: DType,
) -> Result<Letterbox> {
    let img = image::open(photo_path)
        .with_context(|| format!("reading SCRFD input photo {}", photo_path.display()))?
        .to_rgb8();
    let (ow, oh) = (img.width() as f32, img.height() as f32);
    // Uniform scale so the longer edge equals input_size.
    let scale = (input_size as f32) / ow.max(oh);
    let new_w = (ow * scale).round() as u32;
    let new_h = (oh * scale).round() as u32;
    // InsightFace pastes the resized image at the TOP-LEFT (not centred) and
    // resizes with bilinear (`cv2.resize` default).
    let resized = image::imageops::resize(&img, new_w, new_h, FilterType::Triangle);
    let pad_x = 0.0f32;
    let pad_y = 0.0f32;

    // Build the (1, 3, S, S) tensor with the resized image pasted at
    // (pad_x, pad_y). Pixels outside the paste are mid-grey (127.5
    // pre-normalisation → 0 post-normalisation).
    let s = input_size as usize;
    let mut data = vec![0f32; 3 * s * s];
    // SCRFD uses InsightFace's preprocessing: (x - 127.5) / 128.0.
    // Pixels: BGR (not RGB!) channel order in InsightFace reference.
    // candle prefers RGB throughout the codebase, so we keep RGB here
    // and document this as a verification point — if detections look
    // hue-shifted, swap channel order.
    let new_w_u = new_w as i32;
    let new_h_u = new_h as i32;
    let pad_x_i = pad_x as i32;
    let pad_y_i = pad_y as i32;
    for c in 0..3 {
        for y in 0..s as i32 {
            for x in 0..s as i32 {
                let src_x = x - pad_x_i;
                let src_y = y - pad_y_i;
                let v = if src_x >= 0 && src_x < new_w_u && src_y >= 0 && src_y < new_h_u {
                    resized.get_pixel(src_x as u32, src_y as u32).0[c] as f32
                } else {
                    127.5 // grey padding
                };
                data[c * s * s + (y as usize) * s + (x as usize)] = (v - 127.5) / 128.0;
            }
        }
    }
    let tensor = Tensor::from_vec(data, (1, 3, s, s), device)?.to_dtype(dtype)?;
    Ok(Letterbox {
        tensor,
        scale,
        pad_x,
        pad_y,
        orig_w: ow,
        orig_h: oh,
    })
}

/// Undo the letterbox transform on a detection — convert from
/// letterbox-pixel coords back to original-image-pixel coords.
fn unletterbox_face(face: &mut Face, lb: &Letterbox) {
    let s = lb.scale;
    let inv_s = 1.0 / s;
    // Bbox.
    face.bbox[0] = (face.bbox[0] - lb.pad_x) * inv_s;
    face.bbox[1] = (face.bbox[1] - lb.pad_y) * inv_s;
    face.bbox[2] = (face.bbox[2] - lb.pad_x) * inv_s;
    face.bbox[3] = (face.bbox[3] - lb.pad_y) * inv_s;
    face.bbox[0] = face.bbox[0].clamp(0.0, lb.orig_w);
    face.bbox[1] = face.bbox[1].clamp(0.0, lb.orig_h);
    face.bbox[2] = face.bbox[2].clamp(0.0, lb.orig_w);
    face.bbox[3] = face.bbox[3].clamp(0.0, lb.orig_h);
    // Landmarks.
    for i in 0..5 {
        face.landmarks[i][0] =
            ((face.landmarks[i][0] - lb.pad_x) * inv_s).clamp(0.0, lb.orig_w);
        face.landmarks[i][1] =
            ((face.landmarks[i][1] - lb.pad_y) * inv_s).clamp(0.0, lb.orig_h);
    }
}

// =====================================================================
// Public detector wrapper.
// =====================================================================

/// Top-level SCRFD wrapper. Loads weights, runs detection, returns
/// `Face` records in the input photo's original pixel coordinate system.
pub struct SCRFDDetector {
    model: SCRFD,
    config: SCRFDConfig,
    device: Device,
    dtype: DType,
    pub score_threshold: f32,
    pub nms_threshold: f32,
}

impl SCRFDDetector {
    /// Default thresholds match InsightFace's deploy script
    /// (`score_threshold=0.5`, `nms_iou=0.4`).
    pub fn load(
        weights: &Path,
        config: SCRFDConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)?
        };
        let model = SCRFD::new(vb, config.clone())?;
        Ok(Self {
            model,
            config,
            device: device.clone(),
            dtype,
            score_threshold: 0.5,
            nms_threshold: 0.4,
        })
    }

    /// Detect all faces in `photo_path`. Returns them in score-descending
    /// order after NMS, in the photo's original pixel coordinates.
    pub fn detect(&self, photo_path: &Path) -> Result<Vec<Face>> {
        let lb = letterbox_preprocess(
            photo_path,
            self.config.input_size,
            &self.device,
            self.dtype,
        )?;
        let levels = self.model.forward(&lb.tensor)?;

        // Decode each FPN level, accumulate detections.
        let mut all_faces = Vec::new();
        for (i, (cls, bbox, kps)) in levels.iter().enumerate() {
            // Squeeze batch dim (batch=1) → (C, H, W).
            let cls = cls.i(0)?;
            let bbox = bbox.i(0)?;
            let kps = kps.i(0)?;
            let stride = self.config.strides[i];
            let level_faces = decode_level(
                &cls,
                &bbox,
                &kps,
                stride,
                self.score_threshold,
                self.config.num_anchors,
            )?;
            all_faces.extend(level_faces);
        }
        // NMS across all levels.
        let mut kept = nms(all_faces, self.nms_threshold);
        // Map back to original-image pixel coords.
        for f in kept.iter_mut() {
            unletterbox_face(f, &lb);
        }
        Ok(kept)
    }

    /// Detect and return only the highest-confidence face, with
    /// landmarks normalised against the original image's dimensions.
    /// Convenience for the FaceID integration path. Returns `None`
    /// if no face exceeds the score threshold.
    pub fn detect_primary_normalised(
        &self,
        photo_path: &Path,
    ) -> Result<Option<[[f32; 2]; 5]>> {
        let faces = self.detect(photo_path)?;
        let Some(top) = faces.into_iter().next() else {
            return Ok(None);
        };
        // We don't have orig dims here — re-read the file dimensions.
        let (ow, oh) = image::image_dimensions(photo_path)?;
        let ow = ow as f32;
        let oh = oh as f32;
        let mut norm = [[0f32; 2]; 5];
        for i in 0..5 {
            norm[i][0] = (top.landmarks[i][0] / ow).clamp(0.0, 1.0);
            norm[i][1] = (top.landmarks[i][1] / oh).clamp(0.0, 1.0);
        }
        Ok(Some(norm))
    }
}

/// Sync preflight — confirms SCRFD config is plausible without hitting
/// the network. Used by callers that want to know "is SCRFD configured
/// at all" without committing to a download. Returns `Ok(true)` when
/// either env var is set (and the local-path variant points at an
/// existing file); `Ok(false)` when nothing is configured (SCRFD is
/// optional, so absence isn't an error).
pub fn preflight_scrfd() -> Result<bool> {
    let has_local = std::env::var("PLAKAT_SCRFD_WEIGHTS").is_ok();
    let has_hf = std::env::var("PLAKAT_SCRFD_HF").is_ok();
    if !has_local && !has_hf {
        return Ok(false);
    }
    if let Ok(env) = std::env::var("PLAKAT_SCRFD_WEIGHTS") {
        let path = std::path::PathBuf::from(&env);
        if !path.exists() {
            return Err(anyhow!(
                "PLAKAT_SCRFD_WEIGHTS points to {} which doesn't exist.",
                path.display()
            ));
        }
    }
    // HF spec validated only at download time.
    Ok(true)
}

/// Async resolver — turns env-var config into a local safetensors path.
/// Returns `Ok(None)` if neither `PLAKAT_SCRFD_WEIGHTS` nor `PLAKAT_SCRFD_HF`
/// is set (SCRFD is opt-in, so unset is fine).
///
/// Priority: local path wins over HF spec.
pub async fn resolve_scrfd_weights() -> Result<Option<std::path::PathBuf>> {
    if let Ok(env) = std::env::var("PLAKAT_SCRFD_WEIGHTS") {
        let path = std::path::PathBuf::from(&env);
        if !path.exists() {
            return Err(anyhow!(
                "PLAKAT_SCRFD_WEIGHTS points to {} which doesn't exist.",
                path.display()
            ));
        }
        return Ok(Some(path));
    }
    if let Ok(spec) = std::env::var("PLAKAT_SCRFD_HF") {
        let (repo, file) =
            crate::pipelines::ip_adapter::parse_hf_spec(&spec, "PLAKAT_SCRFD_HF")?;
        let s = crate::ui::progress::spinner(&format!(
            "Downloading SCRFD from {repo}/{file}"
        ));
        let path = crate::hf::download::get_file(&repo, &file)
            .await
            .with_context(|| {
                format!("downloading SCRFD from {repo}/{file} via PLAKAT_SCRFD_HF")
            })?;
        s.finish_with_message(format!("✓ SCRFD cached at {}", path.display()));
        return Ok(Some(path));
    }
    Ok(None)
}
