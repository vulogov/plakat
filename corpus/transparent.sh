#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — transparent (background knock-out → RGBA)
# ===================================================================
# `plakat transparent` flood-fills the background to alpha 0 — starting from the
# image corners, growing while each step stays within `--tolerance` of its
# neighbour. That follows a gradient or soft shadow yet stops at a sharp subject
# edge, so it works on real (studio-lit) renders, not just a perfectly flat
# colour. We generate the subject on a chroma-key backdrop, then cut it out.
# Output must be `.png` (or `.webp`) to keep alpha. Ungated, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
OUT="$ROOT/corpus/images/transparent"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

# 1. Generate the subject on a flat chroma-key green backdrop. Green is far from
#    the apple's reds / creams / browns, so the fill stops cleanly at the
#    silhouette. Avoid "product photo / studio lighting" — those add the gradient
#    + shadow + reflection that defeat a cut-out.
"$PLAKAT" generate "a single ripe red apple, centered, floating on a plain flat solid chroma-key green background, evenly lit, no surface, no ground, no shadow, no reflection, sharp focus" \
  --model sdxl --negative "shadow, reflection, gradient, surface, floor, table, ground, vignette, studio lighting, depth of field" \
  --steps 30 --size 1024x1024 --seed 7 --device metal \
  --out "$WORK/apple-raw"

# 2. Flood-fill the green out → an RGBA cut-out (tolerance follows the green
#    gradient/shadow but stops at the apple edge).
"$PLAKAT" transparent \
  --in  "$WORK/apple-raw/plakat-7.png" \
  --out "$OUT/apple-cutout.png" \
  --tolerance 24

echo "✓ wrote corpus/images/transparent/apple-cutout.png (RGBA — background flood-filled out)"
