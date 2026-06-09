#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — Stable Cascade image variation
# ===================================================================
# `--image-variation` conditions generation on a reference image's CLIP
# ViT-L/14 embedding (unCLIP-style): the output keeps the reference's
# *semantics* — subject, palette, mood — but RE-COMPOSES it into a new image.
# Empty prompt → vary on the image alone; add a prompt to steer the variation.
# Cascade-only (loads `image_encoder/` from the Cascade repo on first use).
# Ungated; Stable Cascade is heavy (~16 GB on Metal).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
REF="$ROOT/corpus/images/sdxl/sdxl_landscape/plakat-42.png"   # a coastal cliff at sunset
mkdir -p "$ROOT/corpus/images/variation"

# 1. Pure variation — vary on the reference alone (empty prompt).
"$PLAKAT" generate "" \
  --model stable-cascade --image-variation "$REF" \
  --steps 30 --size 1024x1024 --seed 7 --device metal \
  --out "$ROOT/corpus/images/variation/pure"

# 2. Steered variation — same reference, nudged by a prompt.
"$PLAKAT" generate "the same rugged coastline under a dramatic stormy sky" \
  --model stable-cascade --image-variation "$REF" \
  --steps 30 --size 1024x1024 --seed 7 --device metal \
  --out "$ROOT/corpus/images/variation/steered"

echo "✓ wrote corpus/images/variation/{pure,steered}/ (Cascade image variation of the coastal-cliff ref)"
