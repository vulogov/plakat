#!/usr/bin/env bash
# 02_inpaint_sky.sh — replace only the sky with a stormy version,
# leaving the ground untouched. The `--mask` flag triggers inpaint
# mode: pixels where the mask is white get re-painted, the rest is
# preserved exactly.
#
# Output: out/img2img-tutorial/02_inpaint_sky/plakat-inpaint-2002.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/img2img-tutorial/02_inpaint_sky
mkdir -p "$OUT"

"$PLAKAT" img2img inputs/landscape.png \
    --mask inputs/sky_mask.png \
    --prompt "dramatic stormy sky, dark grey clouds, flashes of lightning, moody atmosphere" \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 2002 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-inpaint-2002.png"
echo "Compare side-by-side with inputs/landscape.png — the ground"
echo "should be pixel-identical, only the sky changed."
