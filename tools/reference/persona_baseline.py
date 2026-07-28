"""RFC PERSONA-1 §2.3 baseline measurement — the control numbers.

Quantifies how uneven inline-prompt personas are TODAY, per model family, so every later phase can
report the same statistics and the improvement is a tracked number rather than a claim.

For a fixed dense person-prompt rendered at N seeds, it reports:
  * identity variance   — pairwise ArcFace cosine over detected faces (mean / median / p5)
  * detection-failure   — fraction of renders with 0 or >1 detected faces
  * localized-detail hit — best-effort: does the prompted mole appear (OWL-ViT), and on which side

Measurement uses the SAME upstream models plakat verified against (InsightFace SCRFD+ArcFace via the
`insightface` package; OWL-ViT via `transformers`), so the numbers are comparable to plakat's own.

Rendering is DECOUPLED. Either:
  (a) point --images-dir at a folder of already-rendered PNGs (from `plakat generate ... -n 32`), or
  (b) pass --render --model <alias> --bin ./target/release/plakat to render them first.

Run:  python3 tools/reference/persona_baseline.py --images-dir out/baseline/sdxl --tag sdxl
      python3 tools/reference/persona_baseline.py --render --model sdxl --count 32 \
              --bin target/release/plakat --out out/baseline --tag sdxl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# The fixed, densely-descriptive person prompt (with one localized detail for the hit-rate probe).
BASELINE_PROMPT = (
    "a photorealistic headshot portrait of a 34 year old woman, oval face, wide-set hazel eyes, "
    "straight nose, full lips, auburn shoulder-length wavy hair, fair skin, a small dark mole below "
    "her left eye, neutral expression, soft studio lighting, plain grey background, sharp focus"
)


def render(bin_path, model, count, prompt, out_dir, base_seed=1000):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        seed = base_seed + i
        subprocess.run(
            [bin_path, "generate", prompt, "--model", model, "--seed", str(seed), "--out", str(out_dir)],
            check=False,
        )
    return sorted(out_dir.glob("*.png"))


def load_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def measure_identity(images, app):
    """Return (embeddings, detection_failures, per-image face counts)."""
    import cv2
    embeds, counts = [], []
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            counts.append(0)
            continue
        faces = app.get(img)
        counts.append(len(faces))
        if len(faces) == 1:
            e = faces[0].normed_embedding
            embeds.append(e / (np.linalg.norm(e) + 1e-9))
    fails = sum(1 for c in counts if c != 1)
    return np.array(embeds), fails, counts


def pairwise_cosine_stats(embeds):
    if len(embeds) < 2:
        return {"n": int(len(embeds)), "mean": None, "median": None, "p5": None}
    m = embeds @ embeds.T
    iu = np.triu_indices(len(embeds), k=1)
    vals = m[iu]
    return {
        "n": int(len(embeds)),
        "pairs": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p5": float(np.percentile(vals, 5)),
    }


def measure_detail_hit(images, query="a mole on a face", threshold=0.1):
    """Best-effort localized-detail hit rate + correct-side rate via OWL-ViT.
    NOTE: small marks are exactly what OWL-ViT struggles with (§2.1.4) — treat as a noisy proxy until
    the `local_anomaly` probe (Phase 1) exists."""
    try:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        from PIL import Image
    except Exception as e:
        return {"available": False, "reason": str(e)}
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval()
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    hits, correct_side, total = 0, 0, 0
    for p in images:
        total += 1
        im = Image.open(p).convert("RGB")
        inp = proc(text=[[query]], images=im, return_tensors="pt")
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, threshold=threshold, target_sizes=torch.tensor([im.size[::-1]])
        )[0]
        if len(res["scores"]) == 0:
            continue
        hits += 1
        # best box; prompt says "below her LEFT eye" → subject-left = image-right half.
        best = int(res["scores"].argmax())
        cx = float(res["boxes"][best][[0, 2]].mean()) / im.size[0]
        if cx > 0.5:  # image-right = subject-left
            correct_side += 1
    return {
        "available": True,
        "query": query,
        "hit_rate": hits / max(total, 1),
        "correct_side_rate_given_hit": (correct_side / hits) if hits else None,
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--bin", default="target/release/plakat")
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--prompt", default=BASELINE_PROMPT)
    ap.add_argument("--out", default="out/baseline")
    ap.add_argument("--tag", required=True, help="family label for the report")
    ap.add_argument("--no-detail", action="store_true", help="skip the OWL-ViT detail-hit probe")
    args = ap.parse_args()

    if args.render:
        if not args.model:
            print("--render requires --model", file=sys.stderr)
            return 2
        img_dir = Path(args.out) / args.tag
        images = render(args.bin, args.model, args.count, args.prompt, img_dir)
    else:
        if not args.images_dir:
            print("pass --images-dir or --render --model", file=sys.stderr)
            return 2
        images = sorted(Path(args.images_dir).glob("*.png"))
    if not images:
        print("no images found", file=sys.stderr)
        return 1

    print(f"[baseline:{args.tag}] {len(images)} images; loading InsightFace…", file=sys.stderr)
    app = load_face_app()
    embeds, fails, counts = measure_identity(images, app)
    ident = pairwise_cosine_stats(embeds)
    detail = {"skipped": True} if args.no_detail else measure_detail_hit(images)

    report = {
        "tag": args.tag,
        "prompt": args.prompt,
        "n_images": len(images),
        "detection_failure_rate": fails / len(images),
        "face_counts_histogram": {str(k): counts.count(k) for k in sorted(set(counts))},
        "identity_variance": ident,
        "localized_detail": detail,
    }
    out_json = Path(args.out) / f"baseline-{args.tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[baseline:{args.tag}] → {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
