#!/usr/bin/env bash
# 04_strength_sweep.sh — generate three img2img variants at 0.3,
# 0.5, 0.7 strengths to show the dial in action. Lower = closer to
# the source, higher = more creative latitude.
#
# Output: out/img2img-tutorial/04_strength_sweep/strength-{0_3,0_5,0_7}.png
# (each file is the plakat-img2img-<seed>.png from its sub-run,
#  copied with a friendly name for side-by-side viewing).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/img2img-tutorial/04_strength_sweep
mkdir -p "$OUT"

for s in 0.3 0.5 0.7; do
    sub="$OUT/.tmp-$s"
    mkdir -p "$sub"
    "$PLAKAT" img2img inputs/landscape.png \
        --prompt "oil painting of a meadow at golden hour, visible brushwork" \
        --strength "$s" \
        --size 512x512 --steps 20 --scheduler euler-a \
        --seed 2004 \
        --out "$sub"
    cp "$sub/plakat-img2img-2004.png" "$OUT/strength-${s//./_}.png"
    rm -rf "$sub"
done

echo
echo "Wrote $OUT/strength-{0_3,0_5,0_7}.png"
echo "All same prompt + same seed; only --strength differs."
