#!/usr/bin/env bash
# 05_sdxl.sh — v0.10: SDXL ControlNet.
#
# Same depth-conditioning workflow as 01_basic.sh, but routed
# through SDXL. Plakat auto-detects the architecture from --model
# and downloads the SDXL ControlNet checkpoint
# (diffusers/controlnet-depth-sdxl-1.0-small, ~600 MB).
#
# SDXL outputs at 1024² yield significantly more detail than SD 1.5
# at 512², at the cost of ~3-4× the wall time and ~2× the memory
# headroom. On a 24 GB Apple Silicon Mac, expect ~25-40 s per
# image after first-run JIT.
#
# First-run downloads (one-time, cached):
#   * SDXL base (~7 GB)
#   * SDXL ControlNet-Depth-small (~600 MB)
#   * Depth-Anything-V2-small if not cached (~99 MB)
#
# Output: out/control-tutorial/05_sdxl/plakat-3010.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/control-tutorial/05_sdxl
mkdir -p "$OUT"

"$PLAKAT" generate "a fox sitting in a clearing, golden hour, photographic, shallow depth of field" \
    --model sdxl \
    --size 1024x1024 \
    --steps 25 --scheduler euler-a \
    --control depth \
    --control-image inputs/scene_depth.png \
    --seed 3010 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-3010.png"
echo "Same depth map as 01_basic.sh, but routed through SDXL at 1024²."
echo "Compare with 01_basic.sh's SD 1.5 output — the SDXL version"
echo "should have much sharper texture detail."
