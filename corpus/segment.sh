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

# 1) Select the subject — a point on the HEAD plus three down the body, so the
#    whole figure (incl. head/hat) is in the mask; otherwise the crown lands in
#    the background and the inpaint repaints it into rigging. --invert → the mask
#    covers the background (white = repaint). --grow leaves a margin around the
#    subject so the repaint never touches its fringe; --feather softens the seam.
"$PLAKAT" segment --in "$SRC" --out "$OUT/bg-mask.png" \
  --point 0.5,0.22 --point 0.5,0.42 --point 0.5,0.62 --point 0.5,0.8 \
  --invert --grow 16 --feather 8 --device metal

# 2) Repaint ONLY the background (white), preserving the masked-black subject.
# 2) Repaint ONLY the background. The prompt steers the upper region to clear sky
#    and keeps the ships DISTANT/on the horizon, so sd15 doesn't fill the area
#    around the head with rigging.
"$PLAKAT" img2img "$SRC" \
  --prompt "a calm misty harbour at dawn, distant tall ships on the far horizon, glassy reflective water, soft golden light, clear pale sky" \
  --mask "$OUT/bg-mask.png" \
  --model sd15 --strength 0.85 --steps 28 --seed 42 --device metal \
  --out "$OUT"

echo "✓ wrote corpus/images/segment/ (subject kept, background swapped via one SAM click)"
