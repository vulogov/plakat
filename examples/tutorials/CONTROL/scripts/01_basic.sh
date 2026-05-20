#!/usr/bin/env bash
# 01_basic.sh — minimal ControlNet-Depth: generate an image whose
# layout follows inputs/scene_depth.png.
#
# The depth map has a bright disc in the foreground (a "subject"),
# medium-grey bumps in the middle distance, and a dark sky above
# the horizon. The model should respect that 3-D arrangement.
#
# First run downloads:
#   * SD 1.5 (~4 GB)
#   * ControlNet-Depth SD 1.5 weights (~1.4 GB)
# Both cached afterwards.
#
# Output: out/control-tutorial/01_basic/plakat-3001.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/control-tutorial/01_basic
mkdir -p "$OUT"

"$PLAKAT" generate "a photograph of a cat sitting in a meadow, golden hour light" \
    --control depth \
    --control-image inputs/scene_depth.png \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 3001 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-3001.png"
echo "The cat should land in the bottom-centre of the frame —"
echo "where the bright foreground disc in the depth map is."
