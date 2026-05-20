#!/usr/bin/env bash
# 03_inpaint_ground.sh — same sky mask as 02, but inverted with
# `--mask-invert`, so this time the GROUND gets repainted (snow)
# while the sky stays intact.
#
# Demonstrates that one mask file can drive both directions of a
# region edit just by flipping the polarity.
#
# Output: out/img2img-tutorial/03_inpaint_ground/plakat-inpaint-2003.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/img2img-tutorial/03_inpaint_ground
mkdir -p "$OUT"

"$PLAKAT" img2img inputs/landscape.png \
    --mask inputs/sky_mask.png \
    --mask-invert \
    --prompt "snowy winter landscape, fresh powder, footprints, low warm light" \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 2003 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-inpaint-2003.png"
echo "The sky should match the original, the ground is now snowy."
