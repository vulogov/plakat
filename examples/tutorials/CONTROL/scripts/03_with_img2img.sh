#!/usr/bin/env bash
# 03_with_img2img.sh — compose ControlNet with img2img.
#
# Use the depth map as control AND use it (as a grayscale image) as
# the img2img source. The model preserves the depth structure both
# from the control conditioning AND from the source image's
# composition. Useful when you want strong layout adherence.
#
# Output: out/control-tutorial/03_with_img2img/plakat-img2img-3003.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/control-tutorial/03_with_img2img
mkdir -p "$OUT"

"$PLAKAT" img2img inputs/scene_depth.png \
    --prompt "a watercolor painting of a fox on a hill" \
    --strength 0.85 \
    --control depth \
    --control-image inputs/scene_depth.png \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 3003 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-img2img-3003.png"
echo "img2img re-paints the depth map as a watercolor; ControlNet"
echo "locks the depth structure so the layout stays consistent."
