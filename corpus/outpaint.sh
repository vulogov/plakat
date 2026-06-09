#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — outpaint (canvas extension)
# ===================================================================
# Grow a committed SDXL seascape sideways and paint the new region
# in-context. Outpaint pads the canvas, edge-replicates the border as a
# low-frequency hint, auto-builds a mask of the new strips (feathered at the
# inner seam so the extension blends into the original), and runs the inpaint
# pipeline.
#
# Model choice matters: the OUTPUT canvas (1408×1024) drives it, so we use
# sdxl-inpaint (the robust default) — SD 1.5-inpaint falls apart this far off
# its native 512². The prompt must describe the WHOLE scene incl. the new
# region. Needs ~24 GB+ on Metal; if it OOMs, shrink --left/--right.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" outpaint "$ROOT/corpus/images/sdxl/sdxl_landscape/plakat-42.png" \
  --prompt "a dramatic rocky coastline at sunset, crashing ocean waves against sea cliffs, seabirds, glowing orange sky, cinematic wide vista" \
  --left 192 --right 192 \
  --mask-feather 24 --model sdxl-inpaint --steps 40 --seed 42 --device metal \
  --out "$ROOT/corpus/images/outpaint"

echo "✓ wrote corpus/images/outpaint/ (canvas extended left + right, painted in-context)"
