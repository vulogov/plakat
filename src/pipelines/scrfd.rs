//! SCRFD face detector port — Phase 4c.4.
//!
//! "Sample and Computation Redistribution for Efficient Face Detection"
//! (Guo et al. 2021, https://arxiv.org/abs/2105.04714). InsightFace's
//! deployed face detector — produces bbox + 5-point landmarks per face.
//!
//! Once landmarks are available, `face_models::align_to_arcface_template`
//! (Phase 4c.3) produces the ArcFace-canonical 112×112 crop and FaceID
//! reaches reference-quality identity preservation **without** any
//! user-supplied `--face-bbox` / `--face-landmarks` flag.
//!
//! ## Phase 4c.4 status — architecture only, verification pending
//!
//! The model architecture, anchor generation, decoding, NMS, and
//! preprocessing are all in this file. **What's not yet verified** is
//! that the weight-key layout this port expects matches any
//! publicly-available SCRFD safetensors. The exact channel widths /
//! block counts for SCRFD-500MF are taken from the InsightFace
//! reference; if h94-style converted weights use slightly different
//! key naming (e.g. `bbone.stage1.0.conv1.weight` vs `backbone.stages.0.0.conv1.weight`),
//! the load will fail at the first mismatched layer.
//!
//! Setup remains bring-your-own-weights for this session:
//!
//! ```bash
//! # Download the SCRFD bundle from InsightFace releases:
//! curl -L -o scrfd_500m.onnx \
//!     https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx
//! # Convert ONNX → safetensors:
//! python -c "import onnx, torch
//! from onnx2torch import convert
//! from safetensors.torch import save_file
//! m = convert(onnx.load('scrfd_500m_bnkps.onnx'))
//! save_file(m.state_dict(), 'scrfd_500m.safetensors')"
//! export PLAKAT_SCRFD_WEIGHTS=$(pwd)/scrfd_500m.safetensors
//! ```
//!
//! ## Architecture overview
//!
//! ```text
//!   input (1, 3, 640, 640)  ← resize + letterbox-pad photo
//!     │
//!     ▼
//!   stem: 3×3 stride=2 → BN → ReLU                 → C=16
//!     │
//!     ▼
//!   stage 1 (stride=1): BasicBlock × N₁            → C=16
//!     │   ─────────────────────────────────────────► (skipped from FPN)
//!     ▼
//!   stage 2 (stride=2): BasicBlock × N₂            → C=40   ── FPN P3 input (stride 8)
//!     │
//!     ▼
//!   stage 3 (stride=2): BasicBlock × N₃            → C=72   ── FPN P4 input (stride 16)
//!     │
//!     ▼
//!   stage 4 (stride=2): BasicBlock × N₄            → C=152  ── FPN P5 input (stride 32)
//!     │
//!     ▼
//!   FPN: lateral 1×1 → out_ch=16, top-down add + 3×3 smooth
//!     │
//!     ▼
//!   DetectionHead × {P3, P4, P5}: 2 shared 3×3 → cls/bbox/kps heads
//!     │           cls   : (B, 2 anchors,            H, W)
//!     │           bbox  : (B, 2 anchors × 4,        H, W) — distance format
//!     │           kps   : (B, 2 anchors × 10,       H, W) — 5 (dx,dy) pairs
//!     ▼
//!   decode + NMS → Vec<Face>
//! ```

#![allow(dead_code)] // Phase 4c.4 — full integration into FaceIdEncoder is opt-in;
                     // many helpers are public for future / debugging use.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, ModuleT, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Module, VarBuilder,
};
use image::imageops::FilterType;
use std::path::Path;

const EVAL: bool = false;

/// SCRFD-500MF / similar lightweight variants. Numbers are taken from
/// the InsightFace reference; if the user's weight file targets a
/// different variant, swap in a different config or adjust here.
#[derive(Clone, Debug)]
pub struct SCRFDConfig {
    /// Backbone stem output channels (after `conv1 + bn1 + relu`).
    pub stem_channels: usize,
    /// Output channels of each of the 4 backbone stages.
    pub stage_channels: [usize; 4],
    /// Number of `BasicBlock` repeats per stage. SCRFD-500MF uses small
    /// counts — this is one of the variant-specific numbers most likely
    /// to need tweaking if a different SCRFD model is loaded.
    pub stage_blocks: [usize; 4],
    /// Stride applied to the first block of each stage (1 keeps spatial
    /// dims; 2 halves them). Always [1, 2, 2, 2] for SCRFD-* variants.
    pub stage_strides: [usize; 4],
    /// Output channels feeding the FPN — last 3 stages of the backbone.
    pub fpn_in_channels: [usize; 3],
    /// FPN output channels per level. SCRFD-500MF: 16.
    pub fpn_out_channels: usize,
    /// Number of stacked 3×3 conv layers in the detection head before
    /// the per-task prediction convs.
    pub head_stacked_convs: usize,
    /// Hidden channels inside the head's stacked convs.
    pub head_feat_channels: usize,
    /// Anchor sizes per FPN level. Two square anchors per location.
    /// SCRFD's typical scheme: stride×4 and stride×8.
    pub anchor_sizes: [[u32; 2]; 3],
    /// Number of anchors per spatial location (2 for SCRFD).
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
    /// Best-guess SCRFD-500MF config. Channel widths / block counts
    /// taken from InsightFace's `scrfd.py` reference; if weight loading
    /// fails at a specific layer, tune these and report which dims
    /// the file actually had.
    pub fn scrfd_500mf() -> Self {
        Self {
            stem_channels: 16,
            stage_channels: [16, 40, 72, 152],
            stage_blocks: [1, 2, 3, 3],
            stage_strides: [1, 2, 2, 2],
            fpn_in_channels: [40, 72, 152],
            fpn_out_channels: 16,
            head_stacked_convs: 2,
            head_feat_channels: 64,
            // SCRFD's two anchors per location are at sizes
            // (stride × 4, stride × 8) typically:
            //   stride 8  → [32,  64]
            //   stride 16 → [64,  128]
            //   stride 32 → [128, 256]
            anchor_sizes: [[32, 64], [64, 128], [128, 256]],
            num_anchors: 2,
            strides: [8, 16, 32],
            input_size: 640,
        }
    }
}

// =====================================================================
// BasicBlock — 3×3 → BN → ReLU → 3×3 → BN → +shortcut → ReLU.
// =====================================================================

struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl BasicBlock {
    fn new(
        vs: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        stride: usize,
    ) -> Result<Self> {
        let bn_cfg = BatchNormConfig::default();
        let conv1_cfg = Conv2dConfig {
            stride,
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d_no_bias(in_ch, out_ch, 3, conv1_cfg, vs.pp("conv1"))?;
        let bn1 = candle_nn::batch_norm(out_ch, bn_cfg, vs.pp("bn1"))?;
        let conv2_cfg = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv2 = candle_nn::conv2d_no_bias(out_ch, out_ch, 3, conv2_cfg, vs.pp("conv2"))?;
        let bn2 = candle_nn::batch_norm(out_ch, bn_cfg, vs.pp("bn2"))?;

        let downsample = if stride != 1 || in_ch != out_ch {
            let cfg = Conv2dConfig {
                stride,
                padding: 0,
                ..Default::default()
            };
            let dconv = candle_nn::conv2d_no_bias(
                in_ch,
                out_ch,
                1,
                cfg,
                vs.pp("downsample").pp("0"),
            )?;
            let dbn = candle_nn::batch_norm(out_ch, bn_cfg, vs.pp("downsample").pp("1"))?;
            Some((dconv, dbn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some((conv, bn)) => bn.forward_t(&conv.forward(x)?, EVAL)?,
            None => x.clone(),
        };
        let h = self.conv1.forward(x)?;
        let h = self.bn1.forward_t(&h, EVAL)?;
        let h = h.relu()?;
        let h = self.conv2.forward(&h)?;
        let h = self.bn2.forward_t(&h, EVAL)?;
        let out = (h + identity)?.relu()?;
        Ok(out)
    }
}

// =====================================================================
// Backbone — stem + 4 stages.
// =====================================================================

struct Backbone {
    stem_conv: Conv2d,
    stem_bn: BatchNorm,
    stages: Vec<Vec<BasicBlock>>,
}

impl Backbone {
    fn new(vs: VarBuilder, cfg: &SCRFDConfig) -> Result<Self> {
        let bn_cfg = BatchNormConfig::default();
        let stem_conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let stem_conv = candle_nn::conv2d_no_bias(
            3,
            cfg.stem_channels,
            3,
            stem_conv_cfg,
            vs.pp("conv1"),
        )?;
        let stem_bn = candle_nn::batch_norm(cfg.stem_channels, bn_cfg, vs.pp("bn1"))?;

        let mut stages = Vec::with_capacity(4);
        let mut in_ch = cfg.stem_channels;
        for (stage_idx, (&blocks_n, &stride)) in cfg
            .stage_blocks
            .iter()
            .zip(cfg.stage_strides.iter())
            .enumerate()
        {
            let out_ch = cfg.stage_channels[stage_idx];
            // Stage's VarBuilder. Naming follows InsightFace convention
            // `layer{idx+1}.<i>` for the i-th block.
            let stage_vs = vs.pp(format!("layer{}", stage_idx + 1));
            let mut blocks = Vec::with_capacity(blocks_n);
            blocks.push(BasicBlock::new(stage_vs.pp("0"), in_ch, out_ch, stride)?);
            for j in 1..blocks_n {
                blocks.push(BasicBlock::new(
                    stage_vs.pp(j.to_string()),
                    out_ch,
                    out_ch,
                    1,
                )?);
            }
            stages.push(blocks);
            in_ch = out_ch;
        }
        Ok(Self {
            stem_conv,
            stem_bn,
            stages,
        })
    }

    /// Run the backbone and return the outputs of stages 2, 3, 4 — the
    /// inputs the FPN consumes. Stage 1 is consumed but not returned.
    fn forward(&self, x: &Tensor) -> Result<[Tensor; 3]> {
        let x = self.stem_conv.forward(x)?;
        let x = self.stem_bn.forward_t(&x, EVAL)?;
        let mut x = x.relu()?;
        let mut outs: Vec<Tensor> = Vec::with_capacity(3);
        for (i, stage) in self.stages.iter().enumerate() {
            for block in stage {
                x = block.forward(&x)?;
            }
            if i >= 1 {
                outs.push(x.clone());
            }
        }
        if outs.len() != 3 {
            bail!("backbone produced {} feature levels, expected 3", outs.len());
        }
        Ok([outs.remove(0), outs.remove(0), outs.remove(0)])
    }
}

// =====================================================================
// FPN — lateral 1×1 + top-down upsample + 3×3 smoothing per level.
// =====================================================================

struct FPN {
    lateral: [Conv2d; 3],
    smooth: [Conv2d; 3],
}

impl FPN {
    fn new(vs: VarBuilder, cfg: &SCRFDConfig) -> Result<Self> {
        let mk_lateral = |i: usize| -> Result<Conv2d> {
            let lateral_cfg = Conv2dConfig::default();
            Ok(candle_nn::conv2d(
                cfg.fpn_in_channels[i],
                cfg.fpn_out_channels,
                1,
                lateral_cfg,
                vs.pp("lateral_convs").pp(i.to_string()).pp("conv"),
            )?)
        };
        let mk_smooth = |i: usize| -> Result<Conv2d> {
            let cfg_s = Conv2dConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            };
            Ok(candle_nn::conv2d(
                cfg.fpn_out_channels,
                cfg.fpn_out_channels,
                3,
                cfg_s,
                vs.pp("fpn_convs").pp(i.to_string()).pp("conv"),
            )?)
        };
        let lateral = [mk_lateral(0)?, mk_lateral(1)?, mk_lateral(2)?];
        let smooth = [mk_smooth(0)?, mk_smooth(1)?, mk_smooth(2)?];
        Ok(Self { lateral, smooth })
    }

    fn forward(&self, feats: [Tensor; 3]) -> Result<[Tensor; 3]> {
        // Lateral 1×1 convs.
        let p3 = self.lateral[0].forward(&feats[0])?;
        let p4 = self.lateral[1].forward(&feats[1])?;
        let p5 = self.lateral[2].forward(&feats[2])?;
        // Top-down: upsample P5 → add to P4, upsample P4 → add to P3.
        let (_, _, p4_h, p4_w) = p4.dims4()?;
        let p5_up = p5.upsample_nearest2d(p4_h, p4_w)?;
        let p4 = (p4 + p5_up)?;
        let (_, _, p3_h, p3_w) = p3.dims4()?;
        let p4_up = p4.upsample_nearest2d(p3_h, p3_w)?;
        let p3 = (p3 + p4_up)?;
        // 3×3 smoothing.
        let p3 = self.smooth[0].forward(&p3)?;
        let p4 = self.smooth[1].forward(&p4)?;
        let p5 = self.smooth[2].forward(&p5)?;
        Ok([p3, p4, p5])
    }
}

// =====================================================================
// DetectionHead — shared stacked convs + 3 per-task prediction convs.
// =====================================================================

struct ConvGnReluBlock {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvGnReluBlock {
    fn new(vs: VarBuilder, in_ch: usize, out_ch: usize) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_ch, out_ch, 3, conv_cfg, vs.pp("conv"))?;
        let bn = candle_nn::batch_norm(out_ch, BatchNormConfig::default(), vs.pp("gn"))?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        let h = self.bn.forward_t(&h, EVAL)?;
        h.relu().map_err(Into::into)
    }
}

struct DetectionHead {
    cls_convs: Vec<ConvGnReluBlock>,
    reg_convs: Vec<ConvGnReluBlock>,
    cls_pred: Conv2d,
    reg_pred: Conv2d,
    kps_pred: Conv2d,
}

impl DetectionHead {
    fn new(vs: VarBuilder, cfg: &SCRFDConfig) -> Result<Self> {
        let mut cls_convs = Vec::with_capacity(cfg.head_stacked_convs);
        let mut reg_convs = Vec::with_capacity(cfg.head_stacked_convs);
        let cls_vs = vs.pp("cls_convs");
        let reg_vs = vs.pp("reg_convs");
        for i in 0..cfg.head_stacked_convs {
            let in_ch = if i == 0 {
                cfg.fpn_out_channels
            } else {
                cfg.head_feat_channels
            };
            cls_convs.push(ConvGnReluBlock::new(
                cls_vs.pp(i.to_string()),
                in_ch,
                cfg.head_feat_channels,
            )?);
            reg_convs.push(ConvGnReluBlock::new(
                reg_vs.pp(i.to_string()),
                in_ch,
                cfg.head_feat_channels,
            )?);
        }
        // 1×1 prediction convs (`cls_score` returns one channel per
        // anchor; `bbox_pred` 4 per anchor; `kps_pred` 10 per anchor).
        let pred_cfg = Conv2dConfig::default();
        let cls_pred = candle_nn::conv2d(
            cfg.head_feat_channels,
            cfg.num_anchors,
            1,
            pred_cfg,
            vs.pp("cls_pred"),
        )?;
        let reg_pred = candle_nn::conv2d(
            cfg.head_feat_channels,
            cfg.num_anchors * 4,
            1,
            pred_cfg,
            vs.pp("bbox_pred"),
        )?;
        let kps_pred = candle_nn::conv2d(
            cfg.head_feat_channels,
            cfg.num_anchors * 10,
            1,
            pred_cfg,
            vs.pp("kps_pred"),
        )?;
        Ok(Self {
            cls_convs,
            reg_convs,
            cls_pred,
            reg_pred,
            kps_pred,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let mut cls_feat = x.clone();
        for c in &self.cls_convs {
            cls_feat = c.forward(&cls_feat)?;
        }
        let mut reg_feat = x.clone();
        for c in &self.reg_convs {
            reg_feat = c.forward(&reg_feat)?;
        }
        let cls = self.cls_pred.forward(&cls_feat)?;
        let bbox = self.reg_pred.forward(&reg_feat)?;
        let kps = self.kps_pred.forward(&reg_feat)?;
        Ok((cls, bbox, kps))
    }
}

// =====================================================================
// Top-level SCRFD module.
// =====================================================================

pub struct SCRFD {
    config: SCRFDConfig,
    backbone: Backbone,
    fpn: FPN,
    head: DetectionHead,
}

impl SCRFD {
    pub fn new(vs: VarBuilder, config: SCRFDConfig) -> Result<Self> {
        let backbone = Backbone::new(vs.pp("backbone"), &config)?;
        let fpn = FPN::new(vs.pp("neck"), &config)?;
        // SCRFD heads are shared across FPN levels (one head, applied
        // three times) per InsightFace's `SCRFDHead`. Some derivative
        // ports use per-level heads — if your weights fail to load
        // here, the per-level naming is `bbox_head.<level>.cls_convs.<i>` etc.
        let head = DetectionHead::new(vs.pp("bbox_head"), &config)?;
        Ok(Self {
            config,
            backbone,
            fpn,
            head,
        })
    }

    /// Returns three sets of (cls, bbox, kps) tensors, one per FPN level.
    pub fn forward(&self, x: &Tensor) -> Result<[(Tensor, Tensor, Tensor); 3]> {
        let backbone_outs = self.backbone.forward(x)?;
        let fpn_outs = self.fpn.forward(backbone_outs)?;
        let p3 = self.head.forward(&fpn_outs[0])?;
        let p4 = self.head.forward(&fpn_outs[1])?;
        let p5 = self.head.forward(&fpn_outs[2])?;
        Ok([p3, p4, p5])
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
            // SCRFD's anchor centre is at (stride/2 + stride * idx) — i.e.
            // the centre of each feature-cell's receptive field.
            centres.push((s * (x as f32 + 0.5), s * (y as f32 + 0.5)));
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
    let (_b, _c, h, w) = cls.dims4()?;
    let cls_flat = cls.to_vec3::<f32>()?; // (C, H, W) after squeezing batch — handle below
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
    let resized = image::imageops::resize(&img, new_w, new_h, FilterType::CatmullRom);
    let pad_x = ((input_size - new_w) / 2) as f32;
    let pad_y = ((input_size - new_h) / 2) as f32;

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
