"""PIPNet-98 (WFLW) aligner — weight conversion + reference dump (ROADMAP_5.0.0, the WFLW-98 aligner).

`pipnet_r18_wflw_98.onnx` (MIT, yakhyo/pipnet-onnx) is a ResNet-18 backbone + a pixel-in-pixel head:
input (1,3,256,256) → cls_map (1,98,8,8) + offset_x/offset_y (1,98,8,8) + nb_x/nb_y (1,980,8,8).
98 WFLW landmarks on an 8×8 grid.

This script:
  1. extracts every ONNX initializer → a `.safetensors` keyed by the ORIGINAL ONNX names (so the
     candle port loads by ONNX name — no fragile positional remap), for hosting + `OwlViT`-style load;
  2. runs onnxruntime on a fixed seeded input → dumps input + the 5 outputs as goldens for the corr test;
  3. prints the initializer-name structure so the candle module tree can mirror it.

Run:  python3 tools/reference/pipnet_dump.py --onnx <pipnet_r18_wflw_98.onnx> --out tools/reference/out
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="tools/reference/out")
    ap.add_argument("--name", default="pipnet-wflw98")
    args = ap.parse_args()

    import onnx
    from onnx import numpy_helper
    from safetensors.numpy import save_file

    m = onnx.load(args.onnx)
    inits = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in m.graph.initializer}

    # Canonical ResNet-18 conv order (from the graph): the 20 backbone convs map to clean names so the
    # candle port loads by a readable module tree instead of opaque `onnx::Conv_NNN` exporter names.
    r18 = [
        "conv1",
        "layer1.0.conv1", "layer1.0.conv2",
        "layer1.1.conv1", "layer1.1.conv2",
        "layer2.0.conv1", "layer2.0.conv2", "layer2.0.downsample",
        "layer2.1.conv1", "layer2.1.conv2",
        "layer3.0.conv1", "layer3.0.conv2", "layer3.0.downsample",
        "layer3.1.conv1", "layer3.1.conv2",
        "layer4.0.conv1", "layer4.0.conv2", "layer4.0.downsample",
        "layer4.1.conv1", "layer4.1.conv2",
    ]
    heads = {"cls_layer", "x_layer", "y_layer", "nb_x_layer", "nb_y_layer"}

    # 1) weights → safetensors, renamed. Each Conv node's inputs are [x, weight, bias]; BN is folded
    #    into (weight, bias). Backbone convs → r18 names in graph order; head convs keep their name.
    weights = {}
    conv_idx = 0
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        wname = n.input[1]
        bname = n.input[2] if len(n.input) > 2 else None
        head = next((h for h in heads if h in wname), None)
        clean = head if head else r18[conv_idx]
        if not head:
            conv_idx += 1
        weights[f"{clean}.weight"] = np.ascontiguousarray(inits[wname])
        if bname and bname in inits:
            weights[f"{clean}.bias"] = np.ascontiguousarray(inits[bname])
    assert conv_idx == len(r18), f"expected {len(r18)} backbone convs, saw {conv_idx}"
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    w_path = out_dir / "pipnet_r18_wflw_98.safetensors"
    save_file(weights, str(w_path))
    print(f"[pipnet] {len(weights)} weight tensors (renamed to a ResNet-18 tree) → {w_path}", file=sys.stderr)

    # 2) onnxruntime forward on a fixed input → goldens.
    import onnxruntime as ort
    rng = np.random.RandomState(0)
    x = rng.randn(1, 3, 256, 256).astype(np.float32)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {sess.get_inputs()[0].name: x})
    goldens = {"input": x}
    for name, arr in zip(out_names, outs):
        goldens[name] = np.ascontiguousarray(arr.astype(np.float32))
    save_file(goldens, str(out_dir / "goldens.safetensors"))
    print(f"[pipnet] goldens: input + {out_names} → {out_dir/'goldens.safetensors'}", file=sys.stderr)

    # 3) print the weight-name structure (grouped by top prefix) to plan the candle module tree.
    from collections import defaultdict
    groups = defaultdict(list)
    for k, v in weights.items():
        top = k.split(".")[0] if "." in k else k.split("/")[0] if "/" in k else k
        groups[top].append((k, list(v.shape)))
    print("=== initializer groups ===", file=sys.stderr)
    for g in sorted(groups):
        print(f"  [{g}] {len(groups[g])} tensors", file=sys.stderr)
        for k, shp in groups[g][:4]:
            print(f"      {k}  {shp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
