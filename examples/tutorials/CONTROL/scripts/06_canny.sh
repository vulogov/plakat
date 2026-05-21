#!/usr/bin/env bash
# 06_canny.sh — Canny edge conditioning.
#
# Canny is the second ControlNet conditioner shipped in v0.10.
# Unlike depth, canny doesn't need a separate annotator model —
# the edge detection runs on CPU via the `imageproc` crate
# (Sobel + non-maximum suppression + hysteresis thresholding).
#
# When --control-from is supplied, plakat runs canny on the
# source image and feeds the edge map to ControlNet-Canny.
# When --control-image is supplied, plakat treats the image as
# already-edged (white = edge, black = background).
#
# This script demonstrates the auto-annotate path: pass a real
# photo, plakat does the canny pass for you.
#
# First-run downloads (one-time, cached):
#   * SD 1.5 (~4 GB) — or SDXL if --model sdxl is added
#   * ControlNet-Canny SD 1.5 (~1.4 GB) from lllyasviel
#
# Output: out/control-tutorial/06_canny/plakat-3020.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/control-tutorial/06_canny
mkdir -p "$OUT"

# Reuse the IMG2IMG tutorial's landscape PNG. Any image with clear
# structural edges works well — architecture, sketches, line art,
# photos with strong contours.
SOURCE="../IMG2IMG/inputs/landscape.png"

"$PLAKAT" generate "an oil painting of a stylised landscape with rolling hills" \
    --control canny \
    --control-from "$SOURCE" \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 3020 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-3020.png"
echo "Plakat ran Canny edge detection on the landscape PNG and used"
echo "the result as the structural conditioner. The painted scene"
echo "should respect the source's edges (hill outlines, sun border)"
echo "while the prompt drives the oil-painting aesthetic."
