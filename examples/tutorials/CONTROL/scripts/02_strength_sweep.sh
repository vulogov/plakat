#!/usr/bin/env bash
# 02_strength_sweep.sh — sweep --control-strength to show the dial in
# action. Same prompt + same seed + same depth map; only the strength
# differs.
#
# Output: out/control-tutorial/02_strength_sweep/strength-{0_5,1_0,1_4}.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/control-tutorial/02_strength_sweep
mkdir -p "$OUT"

for s in 0.5 1.0 1.4; do
    sub="$OUT/.tmp-$s"
    mkdir -p "$sub"
    "$PLAKAT" generate "a fox sitting in a clearing, soft natural light" \
        --control depth \
        --control-image inputs/scene_depth.png \
        --control-strength "$s" \
        --size 512x512 --steps 20 --scheduler euler-a \
        --seed 3002 \
        --out "$sub"
    cp "$sub/plakat-3002.png" "$OUT/strength-${s//./_}.png"
    rm -rf "$sub"
done

echo
echo "Wrote $OUT/strength-{0_5,1_0,1_4}.png"
echo "Compare: 0.5 = loose layout guidance, 1.0 = diffusers default,"
echo "1.4 = aggressive structure enforcement (may sacrifice prompt fit)."
