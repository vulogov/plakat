#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — segment (Segment-Anything / MobileSAM)
# ===================================================================
# The compose-&-edit ENABLER: click an object → a binary mask → drive an
# edit with it. Here we select the portrait subject with ONE click, invert
# (so the mask covers the BACKGROUND, white = repaint), then inpaint only
# the background — the subject is preserved untouched. "Keep me, change the
# scene", driven entirely by a SAM click.
#
# Two stages:
#   1. plakat segment  — MobileSAM, ~40 MB weights auto-download, Metal/CPU
#      fast (~0.4 s inference). Writes bg-mask.png. (cheap; safe to re-run)
#   2. plakat img2img --mask  — sd15 inpaint of the masked background. GPU.
#      Same single-pass cost as corpus/inpaint.sh (Metal-safe).
#
# Refine tip: if one click over/under-selects, add points —
#   --point 0.5,0.4 --point 0.5,0.7        (two foreground points)
#   --point 0.5,0.4 --point 0.1,0.1:bg     (carve a background corner away)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

SRC="$ROOT/corpus/images/portrait/sage_captain/plakat-portrait-43.png"
OUT="$ROOT/corpus/images/segment"
mkdir -p "$OUT"

# 1) Select the subject (three points down the figure for a robust mask —
#    a single click can land on an ambiguous sub-part like the face), invert
#    → the mask covers the background (white = repaint).
"$PLAKAT" segment --in "$SRC" --out "$OUT/bg-mask.png" \
  --point 0.5,0.45 --point 0.5,0.62 --point 0.5,0.78 --invert --device metal

# 2) Repaint ONLY the background (white), preserving the masked-black subject.
"$PLAKAT" img2img "$SRC" \
  --prompt "a misty ancient harbour at dawn, tall ships at anchor, soft golden light, distant hills" \
  --mask "$OUT/bg-mask.png" \
  --model sd15 --strength 0.9 --steps 28 --seed 42 --device metal \
  --out "$OUT"

echo "✓ wrote corpus/images/segment/ (subject kept, background swapped via one SAM click)"
