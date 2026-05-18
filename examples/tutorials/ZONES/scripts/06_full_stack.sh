#!/usr/bin/env bash
# 06_full_stack.sh — composite + blend + smart-zones, all together.
#
# This is the recommended setup for production use: smart zones
# place each artefact correctly relative to the actual painted
# scene, then the blend pass softens the edges so the cutouts
# integrate visually.
#
# Cost: one extra denoise pass for blend (~2-5s) + depth inference
# (~0.5-1.5s on GPU) per image. Both models load once and are
# reused across `--count` images.
#
# Output: out/zones-tutorial/06_full_stack/plakat-*.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/zones-tutorial/06_full_stack
mkdir -p "$OUT"

"$PLAKAT" generate "a moody nordic fjord at dusk, dramatic clouds" \
    --aspect 16:9 --base 768 \
    --artefact moon@sky/right \
    --artefact cloud@sky/center:0.6 \
    --artefact pine@middle_plan/left \
    --artefact pine@middle_plan/right \
    --artefact cottage@close_plan/center:0.5 \
    --artefact-library library \
    --smart-zones \
    --artefact-blend --artefact-blend-strength 0.35 \
    --seed 1006 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-1006.png"
echo "The complete pipeline: smart zone placement, alpha composite,"
echo "then a 35% blend to integrate."
