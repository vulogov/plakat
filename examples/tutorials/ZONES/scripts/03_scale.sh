#!/usr/bin/env bash
# 03_scale.sh — overriding zone AND scale per artefact.
#
# Grammar: `NAME@ZONE:SCALE` — SCALE is a positive float multiplier
# applied to the library's `natural_size_pct`. Default = 1.0.
#
# This demo places three oaks at increasing scales across the
# middle_plan band, illustrating depth-of-field via size: a near oak
# looks larger, a far oak smaller. Auto-stagger spaces them
# horizontally (none specifies an explicit offset).
#
# Output: out/zones-tutorial/03_scale/plakat-*.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/zones-tutorial/03_scale
mkdir -p "$OUT"

"$PLAKAT" generate "a serene oak grove in summer" \
    --artefact oak@far_plan/center:0.5 \
    --artefact oak@middle_plan/center:0.8 \
    --artefact oak@close_plan/center:1.2 \
    --artefact-library library \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 1003 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-1003.png"
echo "Three oaks at 0.5×, 0.8×, 1.2× scale — a cheap depth cue."
