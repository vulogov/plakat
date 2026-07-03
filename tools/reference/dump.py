#!/usr/bin/env python3
"""Author golden reference tensors for `plakat verify` Tier 1.

Run a diffusers model on a canonical fixture, capture named intermediate tensors via forward
hooks, and write `goldens.safetensors` + `manifest.json` in the format plakat's Rust side
(`src/verify/manifest.rs`) consumes.

    python tools/reference/dump.py --model sd15 --fixture portrait_v1 --out tools/reference/out

This is OFFLINE maintainer tooling — plakat never runs it (see README.md). Its output is
frozen on HF and consumed by the pure-Rust verifier.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import fixtures

# Per-family dumpers live in models/<name>.py. Add entries as families are authored.
KNOWN_MODELS = ["sd15", "sdxl", "sd35-medium", "pixart", "stable-cascade", "animatediff"]

# Fallback thresholds when a dumper doesn't specify per-tensor ones. Correlation must be
# high (structural correctness); max_abs is loose enough for BF16-vs-F32 rounding but tight
# enough to catch a real magnitude bug. Tune per tensor as data accrues.
DEFAULT_CORR_MIN = 0.999
DEFAULT_MAX_ABS = 0.05


def load_dumper(model: str):
    mod_name = model.replace("-", "_")
    try:
        return importlib.import_module(f"models.{mod_name}")
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"no dumper for model {model!r} (expected tools/reference/models/{mod_name}.py): {e}"
        )


def build_manifest(model, dumper, fx, tensors) -> dict:
    import diffusers  # local import so --help works without the deps installed

    thresholds = getattr(dumper, "DEFAULT_THRESHOLDS", {})
    tspec = {}
    for name, t in tensors.items():
        corr_min, max_abs = thresholds.get(name, (DEFAULT_CORR_MIN, DEFAULT_MAX_ABS))
        tspec[name] = {
            "shape": list(t.shape),
            "corr_min": corr_min,
            "max_abs": max_abs,
        }
    return {
        "model": model,
        "model_revision": getattr(dumper, "REVISION", ""),
        "fixture": fx.id,
        "plakat_arch": getattr(dumper, "PLAKAT_ARCH", ""),
        "provenance": f"diffusers=={diffusers.__version__}",
        "tensors": tspec,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=KNOWN_MODELS)
    ap.add_argument("--fixture", default="portrait_v1")
    ap.add_argument("--out", default="tools/reference/out", help="output root; artifacts land under <out>/<model>/<fixture>/")
    ap.add_argument("--device", default="cuda", help="cuda | mps | cpu")
    args = ap.parse_args()

    from safetensors.torch import save_file

    fx = fixtures.get(args.fixture)
    dumper = load_dumper(args.model)

    print(f"[reference] authoring goldens: model={args.model} fixture={fx.id} device={args.device}", file=sys.stderr)
    tensors = dumper.dump(fx, args.device)  # dict[str -> torch.Tensor], F32 on CPU
    if not tensors:
        raise SystemExit("dumper returned no tensors — nothing to author")

    # Normalize: F32, CPU, contiguous (safetensors + candle friendly).
    tensors = {k: v.detach().to("cpu").float().contiguous() for k, v in tensors.items()}

    out_dir = Path(args.out) / args.model / fx.id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "goldens.safetensors"))
    manifest = build_manifest(args.model, dumper, fx, tensors)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[reference] wrote {len(tensors)} tensor(s) → {out_dir}", file=sys.stderr)
    for name, t in tensors.items():
        print(f"           {name}  {tuple(t.shape)}", file=sys.stderr)
    print(
        "[reference] next: `plakat verify --tier 1 --model "
        f"{args.model} --golden-dir {args.out}` (once capture points are wired), then push to HF.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
