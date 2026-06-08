#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — outpaint (canvas extension)
# ===================================================================
# Grow a committed SD 1.5 landscape sideways and paint the new region
# in-context. Outpaint pads the canvas, auto-builds a mask of the new
# strip, and hands off to the inpaint pipeline (sd15-inpaint, ungated ~4 GB).
# The prompt must describe the WHOLE scene incl. the new region. Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" outpaint "$ROOT/corpus/images/sd15/sd15_landscape/plakat-42.png" \
  --prompt "a wide panoramic landscape, rolling hills and a distant river under a vast sky" \
  --left 192 --right 192 \
  --model sd15-inpaint --steps 28 --seed 42 --device metal \
  --out "$ROOT/corpus/images/outpaint"

echo "✓ wrote corpus/images/outpaint/ (canvas extended left + right, new region painted in-context)"
