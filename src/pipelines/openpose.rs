//! OpenPose body-pose model used by ControlNet's `openpose` conditioner.
//!
//! Ported from lllyasviel's `body_pose_model.pth` (the CMU
//! body_25-derived 18-keypoint network packaged in
//! `lllyasviel/Annotators`):
//!
//!   * `model0`   — VGG-like backbone (10 conv + 3 max-pool layers)
//!     reducing the input from `(3, H, W)` to `(128, H/8, W/8)`.
//!   * Stage 1 splits into two branches:
//!       - `model1_1` — PAF branch (5 convs, output 38 channels =
//!         19 part-affinity-field pairs × 2 components).
//!       - `model1_2` — heatmap branch (5 convs, output 19 channels =
//!         18 keypoints + 1 background).
//!   * Stages 2–6 iteratively refine. The Mconv stages take the
//!     concat `[previous_PAF, previous_heatmap, backbone_features]`
//!     (185 channels) as input. Seven convs each, ending in the same
//!     output widths (38 and 19).
//!
//! The model returns only the **final stage**'s PAF + heatmap; earlier
//! stages exist for training-time supervision only.
//!
//! See [`annotate_openpose`](super::controlnet_annotator::annotate_openpose)
//! for input/output convention and the post-processing (peak NMS →
//! PAF line-integral scoring → bipartite matching → skeleton draw).

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

/// One conv layer of an OpenPose block. `name` matches the state_dict
/// key prefix (the network was originally an `nn.Sequential` of named
/// modules, so its state_dict carries those names verbatim).
#[derive(Debug, Clone, Copy)]
struct ConvDef {
    name: &'static str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    p: usize,
    /// Apply ReLU after this conv? OpenPose's final-stage convs
    /// (5_5_CPM_L*, Mconv7_*) emit logits — no ReLU.
    relu: bool,
}

const BLOCK0_DEFS: &[ConvDef] = &[
    ConvDef { name: "conv1_1", in_ch: 3, out_ch: 64, k: 3, p: 1, relu: true },
    ConvDef { name: "conv1_2", in_ch: 64, out_ch: 64, k: 3, p: 1, relu: true },
    // -- pool after conv1_2 --
    ConvDef { name: "conv2_1", in_ch: 64, out_ch: 128, k: 3, p: 1, relu: true },
    ConvDef { name: "conv2_2", in_ch: 128, out_ch: 128, k: 3, p: 1, relu: true },
    // -- pool after conv2_2 --
    ConvDef { name: "conv3_1", in_ch: 128, out_ch: 256, k: 3, p: 1, relu: true },
    ConvDef { name: "conv3_2", in_ch: 256, out_ch: 256, k: 3, p: 1, relu: true },
    ConvDef { name: "conv3_3", in_ch: 256, out_ch: 256, k: 3, p: 1, relu: true },
    ConvDef { name: "conv3_4", in_ch: 256, out_ch: 256, k: 3, p: 1, relu: true },
    // -- pool after conv3_4 --
    ConvDef { name: "conv4_1", in_ch: 256, out_ch: 512, k: 3, p: 1, relu: true },
    ConvDef { name: "conv4_2", in_ch: 512, out_ch: 512, k: 3, p: 1, relu: true },
    ConvDef { name: "conv4_3_CPM", in_ch: 512, out_ch: 256, k: 3, p: 1, relu: true },
    ConvDef { name: "conv4_4_CPM", in_ch: 256, out_ch: 128, k: 3, p: 1, relu: true },
];

/// Indices into BLOCK0_DEFS after which a max-pool fires.
const BLOCK0_POOLS_AFTER: &[usize] = &[1, 3, 7];

const BLOCK1_1_DEFS: &[ConvDef] = &[
    ConvDef { name: "conv5_1_CPM_L1", in_ch: 128, out_ch: 128, k: 3, p: 1, relu: true },
    ConvDef { name: "conv5_2_CPM_L1", in_ch: 128, out_ch: 128, k: 3, p: 1, relu: true },
    ConvDef { name: "conv5_3_CPM_L1", in_ch: 128, out_ch: 128, k: 3, p: 1, relu: true },
    ConvDef { name: "conv5_4_CPM_L1", in_ch: 128, out_ch: 512, k: 1, p: 0, relu: true },
    ConvDef { name: "conv5_5_CPM_L1", in_ch: 512, out_ch: 38, k: 1, p: 0, relu: false },
];

const BLOCK1_2_DEFS: &[ConvDef] = &[
    ConvDef { name: "conv5_1_CPM_L2", in_ch: 128, out_ch: 128, k: 3, p: 1, relu: true },
    ConvDef { name: "conv5_2_CPM_L2", in_ch: 128, out_ch: 128, k: 3, p: 1, relu: true },
    ConvDef { name: "conv5_3_CPM_L2", in_ch: 128, out_ch: 128, k: 3, p: 1, relu: true },
    ConvDef { name: "conv5_4_CPM_L2", in_ch: 128, out_ch: 512, k: 1, p: 0, relu: true },
    ConvDef { name: "conv5_5_CPM_L2", in_ch: 512, out_ch: 19, k: 1, p: 0, relu: false },
];

/// Stages 2–6 share an identical layout per branch (185-channel
/// concat → 7 convs ending in PAF=38 or heatmap=19). The names embed
/// the stage index — built dynamically at load time.
fn make_mconv_defs(stage: usize, l: usize, out_ch: usize) -> Vec<ConvDef> {
    let names: [String; 7] = (1..=7)
        .map(|i| format!("Mconv{i}_stage{stage}_L{l}"))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    // Leak each name to get a 'static lifetime — these are owned by
    // the BodyPoseModel until program exit, which is fine for a
    // singleton model (the alternative is changing ConvDef.name to
    // String everywhere, which churns the const tables above).
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let pad7 = 3usize; // 7×7 conv, padding=3
    vec![
        ConvDef { name: leak(names[0].clone()), in_ch: 185, out_ch: 128, k: 7, p: pad7, relu: true },
        ConvDef { name: leak(names[1].clone()), in_ch: 128, out_ch: 128, k: 7, p: pad7, relu: true },
        ConvDef { name: leak(names[2].clone()), in_ch: 128, out_ch: 128, k: 7, p: pad7, relu: true },
        ConvDef { name: leak(names[3].clone()), in_ch: 128, out_ch: 128, k: 7, p: pad7, relu: true },
        ConvDef { name: leak(names[4].clone()), in_ch: 128, out_ch: 128, k: 7, p: pad7, relu: true },
        ConvDef { name: leak(names[5].clone()), in_ch: 128, out_ch: 128, k: 1, p: 0, relu: true },
        ConvDef { name: leak(names[6].clone()), in_ch: 128, out_ch: out_ch, k: 1, p: 0, relu: false },
    ]
}

/// One named conv layer, owned by the model.
#[derive(Debug)]
struct NamedConv {
    conv: Conv2d,
    relu: bool,
}

fn load_block(
    vb: &VarBuilder,
    block_prefix: &str,
    defs: &[ConvDef],
) -> Result<Vec<NamedConv>> {
    let mut out = Vec::with_capacity(defs.len());
    for d in defs {
        let cfg = Conv2dConfig { padding: d.p, ..Default::default() };
        let conv = conv2d(
            d.in_ch,
            d.out_ch,
            d.k,
            cfg,
            vb.pp(&format!("{block_prefix}.{}", d.name)),
        )?;
        out.push(NamedConv { conv, relu: d.relu });
    }
    Ok(out)
}

fn forward_block(layers: &[NamedConv], x: &Tensor) -> Result<Tensor> {
    let mut h = x.clone();
    for l in layers {
        h = l.conv.forward(&h)?;
        if l.relu {
            h = h.relu()?;
        }
    }
    Ok(h)
}

/// Forward `model0`: 10 convs interleaved with three 2×2 max-pools.
fn forward_block0(layers: &[NamedConv], x: &Tensor) -> Result<Tensor> {
    let mut h = x.clone();
    let mut next_pool = 0;
    for (i, l) in layers.iter().enumerate() {
        h = l.conv.forward(&h)?;
        if l.relu {
            h = h.relu()?;
        }
        if next_pool < BLOCK0_POOLS_AFTER.len() && BLOCK0_POOLS_AFTER[next_pool] == i {
            h = h.max_pool2d(2)?;
            next_pool += 1;
        }
    }
    Ok(h)
}

/// Stages 2-6 of OpenPose body-pose. Each stage has a PAF (`L1`) and
/// heatmap (`L2`) branch, both taking the same 185-channel concat as
/// input.
#[derive(Debug)]
struct MStage {
    l1: Vec<NamedConv>,
    l2: Vec<NamedConv>,
}

/// Full body-pose network.
#[derive(Debug)]
pub struct BodyPoseModel {
    model0: Vec<NamedConv>,
    model1_1: Vec<NamedConv>,
    model1_2: Vec<NamedConv>,
    /// Stages 2..=6 (4 stages, indexed [0..=4]).
    refine: Vec<MStage>,
}

impl BodyPoseModel {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let model0 = load_block(&vb, "model0", BLOCK0_DEFS)?;
        let model1_1 = load_block(&vb, "model1_1", BLOCK1_1_DEFS)?;
        let model1_2 = load_block(&vb, "model1_2", BLOCK1_2_DEFS)?;
        let mut refine = Vec::with_capacity(5);
        for stage in 2..=6 {
            let l1_defs = make_mconv_defs(stage, 1, 38);
            let l2_defs = make_mconv_defs(stage, 2, 19);
            let l1 = load_block(&vb, &format!("model{stage}_1"), &l1_defs)?;
            let l2 = load_block(&vb, &format!("model{stage}_2"), &l2_defs)?;
            refine.push(MStage { l1, l2 });
        }
        Ok(Self {
            model0,
            model1_1,
            model1_2,
            refine,
        })
    }

    /// Runs the network. Input: `(1, 3, H, W)` f32 in `[-0.5, 0.5]`
    /// (the reference normalises raw `[0, 255]` pixels via
    /// `pixel/256 - 0.5`).
    ///
    /// Returns `(paf, heatmap)`:
    ///   * `paf`     — `(1, 38, H/8, W/8)`, part-affinity fields
    ///                 (19 limb pairs × 2 components per pair).
    ///   * `heatmap` — `(1, 19, H/8, W/8)`, keypoint heatmaps
    ///                 (18 body keypoints + 1 background).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let out1 = forward_block0(&self.model0, x)?;
        let mut paf = forward_block(&self.model1_1, &out1)?;
        let mut heatmap = forward_block(&self.model1_2, &out1)?;
        for stage in &self.refine {
            let stage_in = Tensor::cat(&[&paf, &heatmap, &out1], 1)?;
            paf = forward_block(&stage.l1, &stage_in)?;
            heatmap = forward_block(&stage.l2, &stage_in)?;
        }
        Ok((paf, heatmap))
    }
}
