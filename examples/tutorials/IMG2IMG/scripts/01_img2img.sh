#!/usr/bin/env bash
# 01_img2img.sh — basic img2img: re-imagine the whole landscape as a
# watercolor painting. No mask, so every pixel is denoised at
# `--strength 0.55` (default for img2img is 0.6).
#
# Output: out/img2img-tutorial/01_img2img/plakat-img2img-2001.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/img2img-tutorial/01_img2img
mkdir -p "$OUT"

"$PLAKAT" img2img inputs/landscape.png \
    --prompt "soft watercolor landscape painting, wet-on-wet wash, paper texture, gentle rolling hills" \
    --strength 0.55 \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 2001 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-img2img-2001.png"
echo "Same composition as the source landscape but with watercolor"
echo "treatment. Try --strength 0.3 (subtle) or 0.75 (heavier)."
