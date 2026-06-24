//! `plakat convert-onnx` — convert an ONNX model into the `.safetensors` layout
//! a plakat pipeline expects.
//!
//! ONNX exports name their weights by graph node (`546`, `bbox_head.stride_cls.…`)
//! — not by the module tree a candle `VarBuilder` walks. This command walks the
//! ONNX graph in topological order and renames each weight to the plakat key its
//! consumer looks up, then writes a `.safetensors`.
//!
//! Currently the one supported architecture is **SCRFD-500MF** (InsightFace's
//! `det_500m.onnx`, the face detector behind `--identity faceid`, `--adetailer`,
//! and `multiperson` face-refine). Its weights are a mix of named (neck/head
//! prediction) and numbered (backbone, head stem) tensors, so the mapping is
//! positional over the conv nodes — verified against onnxruntime.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use clap::{Args, ValueEnum};

#[derive(Args, Debug)]
pub struct ConvertOnnxArgs {
    /// Input `.onnx` model (e.g. InsightFace `det_500m.onnx`).
    #[arg(value_name = "INPUT.onnx")]
    pub input: PathBuf,

    /// Output `.safetensors` path.
    #[arg(value_name = "OUTPUT.safetensors")]
    pub output: PathBuf,

    /// Source architecture (decides the weight-name mapping).
    #[arg(long, value_enum, default_value_t = Arch::Scrfd500mf)]
    pub arch: Arch,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Arch {
    /// InsightFace SCRFD-500MF face detector (`det_500m.onnx`).
    #[value(name = "scrfd-500mf")]
    Scrfd500mf,
    /// InsightFace `inswapper_128.onnx` face-swap generator.
    #[value(name = "inswapper-128")]
    Inswapper128,
}

impl Arch {
    /// plakat tensor-name prefix per node, keyed by ONNX op_type, each list in
    /// graph order. Every listed node contributes `<name>.weight` (+ `.bias`).
    fn name_map(self) -> Vec<(&'static str, Vec<String>)> {
        match self {
            Arch::Scrfd500mf => vec![("Conv", scrfd_500mf_conv_names())],
            Arch::Inswapper128 => vec![
                ("Conv", inswapper_conv_names()),
                ("Gemm", inswapper_gemm_names()),
            ],
        }
    }
}

pub async fn run(args: ConvertOnnxArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("input ONNX not found: {}", args.input.display());
    }
    let model = candle_onnx::read_file(&args.input)
        .with_context(|| format!("parsing ONNX {}", args.input.display()))?;
    let graph = model.graph.context("ONNX model has no graph")?;
    let inits: HashMap<&str, &candle_onnx::onnx::TensorProto> =
        graph.initializer.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut out: HashMap<String, Tensor> = HashMap::new();
    let mut total_nodes = 0usize;
    for (op_type, names) in args.arch.name_map() {
        // Nodes of this op type in topological (graph) order. SCRFD/inswapper
        // weights fold their norm into each conv's bias, so the per-node weight
        // (+ optional bias) is all that needs mapping.
        let nodes: Vec<&candle_onnx::onnx::NodeProto> =
            graph.node.iter().filter(|n| n.op_type == op_type).collect();
        if nodes.len() != names.len() {
            bail!(
                "{:?}: expected {} {op_type} nodes, ONNX has {} — wrong model or arch?",
                args.arch,
                names.len(),
                nodes.len()
            );
        }
        for (plakat_name, node) in names.iter().zip(&nodes) {
            let w_name = node.input.get(1).context("node has no weight input")?;
            let w = inits
                .get(w_name.as_str())
                .with_context(|| format!("weight initializer {w_name:?} not found in graph"))?;
            let weight = candle_onnx::eval::get_tensor(w, w_name)
                .with_context(|| format!("reading weight {w_name:?}"))?;
            out.insert(format!("{plakat_name}.weight"), weight);

            if let Some(b_name) = node.input.get(2) {
                if let Some(b) = inits.get(b_name.as_str()) {
                    let bias = candle_onnx::eval::get_tensor(b, b_name)
                        .with_context(|| format!("reading bias {b_name:?}"))?;
                    out.insert(format!("{plakat_name}.bias"), bias);
                }
            }
        }
        total_nodes += nodes.len();
    }

    candle_core::safetensors::save(&out, &args.output)
        .with_context(|| format!("writing {}", args.output.display()))?;
    println!(
        "✓ {:?}: {} layers → {} tensors → {}",
        args.arch,
        total_nodes,
        out.len(),
        args.output.display()
    );
    Ok(())
}

/// Conv-node plakat names for `inswapper_128`, in graph order (20 convs):
/// 4 encoder, 6 residual AdaIN blocks × 2 convs, 3 decoder + final.
fn inswapper_conv_names() -> Vec<String> {
    let mut n: Vec<String> = Vec::with_capacity(20);
    for i in 0..4 {
        n.push(format!("enc{i}"));
    }
    for b in 0..6 {
        n.push(format!("block{b}.conv0"));
        n.push(format!("block{b}.conv1"));
    }
    for i in 0..3 {
        n.push(format!("dec{i}"));
    }
    n.push("out_conv".into());
    debug_assert_eq!(n.len(), 20);
    n
}

/// Gemm-node plakat names for `inswapper_128`, in graph order (12 styles):
/// each AdaIN block's two style projections (source 512 → 2048 = scale+bias).
fn inswapper_gemm_names() -> Vec<String> {
    let mut n = Vec::with_capacity(12);
    for b in 0..6 {
        n.push(format!("block{b}.style0"));
        n.push(format!("block{b}.style1"));
    }
    n
}

/// plakat tensor-name prefix for each Conv node of SCRFD-500MF, in graph order.
/// Mirrors `pipelines::scrfd`'s module tree (backbone DW-sep blocks, PAFPN neck,
/// per-stride head). Verified positionally against `det_500m.onnx` (60 convs).
fn scrfd_500mf_conv_names() -> Vec<String> {
    let mut n: Vec<String> = Vec::with_capacity(60);
    n.push("backbone.stem".into()); // C00
    for i in 0..14 {
        // C01..C28: 14 depthwise-separable blocks (dw, pw)
        n.push(format!("backbone.b{i}.dw"));
        n.push(format!("backbone.b{i}.pw"));
    }
    for i in 0..3 {
        n.push(format!("neck.lat{i}")); // C29-31
    }
    for i in 0..3 {
        n.push(format!("neck.fpn{i}")); // C32-34
    }
    for i in 0..2 {
        n.push(format!("neck.down{i}")); // C35-36
    }
    for i in 0..2 {
        n.push(format!("neck.pa{i}")); // C37-38
    }
    for s in [8, 16, 32] {
        // C39-59: per-stride head (2 DW-sep stem convs + cls/reg/kps preds)
        for part in ["s0dw", "s0pw", "s1dw", "s1pw", "cls", "reg", "kps"] {
            n.push(format!("head.s{s}.{part}"));
        }
    }
    debug_assert_eq!(n.len(), 60);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrfd_names_cover_all_60_convs_uniquely() {
        let names = scrfd_500mf_conv_names();
        assert_eq!(names.len(), 60);
        let uniq: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(uniq.len(), 60, "duplicate plakat names");
        // spot-check the boundaries
        assert_eq!(names[0], "backbone.stem");
        assert_eq!(names[1], "backbone.b0.dw");
        assert_eq!(names[28], "backbone.b13.pw");
        assert_eq!(names[29], "neck.lat0");
        assert_eq!(names[59], "head.s32.kps");
    }
}
