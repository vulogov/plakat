"""OWL-ViT reference dump (ROADMAP_4.10.0 Phase 2).

Loads `OwlViTForObjectDetection` (google/owlvit-base-patch32), runs a single forward on a fixed
image + a couple of text queries, and dumps the inputs (pixel_values, input_ids, attention_mask)
plus the outputs (pred_boxes cxcywh, logits) and a couple of intermediates (image_feats,
query_embeds) so the candle `owlvit.rs` port can be verified tensor-for-tensor.

Run:  python3 tools/reference/owlvit_dump.py --out tools/reference/out
      (uses a synthetic fixed image so no asset is needed; F32/CPU canonical)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

REPO = "google/owlvit-base-patch32"
QUERIES = ["a photo of a cat", "a red trash can"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/reference/out")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--name", default="owlvit")
    args = ap.parse_args()

    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    print(f"[owlvit] loading {args.repo} (F32/CPU)…", file=sys.stderr)
    model = OwlViTForObjectDetection.from_pretrained(args.repo, torch_dtype=torch.float32).to("cpu").eval()
    proc = OwlViTProcessor.from_pretrained(args.repo)

    # A fixed synthetic image (seeded), processed the standard way → 768×768 pixel_values.
    rng = np.random.RandomState(0)
    img = (rng.rand(480, 640, 3) * 255).astype(np.uint8)
    from PIL import Image
    pil = Image.fromarray(img)
    inputs = proc(text=[QUERIES], images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].float()
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print(f"[owlvit] pixel_values {list(pixel_values.shape)} input_ids {list(input_ids.shape)}", file=sys.stderr)

    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        # intermediates via the public helpers: returns (text_embeds, image_embeds, outputs)
        query_embeds, feat_map, _ = model.image_text_embedder(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask
        )
        b, hh, ww, d = feat_map.shape
        image_feats = feat_map.reshape(b, hh * ww, d)

    tensors = {
        "pixel_values": pixel_values.contiguous(),
        "input_ids": input_ids.to(torch.int64).contiguous(),
        "attention_mask": attention_mask.to(torch.int64).contiguous(),
        "image_feats": image_feats.contiguous(),
        "query_embeds": query_embeds.contiguous(),
        "pred_boxes": out.pred_boxes.contiguous(),  # (1, 576, 4) cxcywh
        "logits": out.logits.contiguous(),          # (1, 576, num_queries)
    }
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(torch.float32) if v.is_floating_point() else v.to(torch.int64) for k, v in tensors.items()}, str(out_dir / "goldens.safetensors"))
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "queries": QUERIES,
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "note": "F32/CPU. pixel_values are OwlViTProcessor output (768x768). logits pre-sigmoid; pred_boxes cxcywh in [0,1].",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[owlvit] wrote {len(tensors)} tensor(s) → {out_dir}", file=sys.stderr)
    for k, v in tensors.items():
        print(f"      {k:14} {list(v.shape)} {v.dtype}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
