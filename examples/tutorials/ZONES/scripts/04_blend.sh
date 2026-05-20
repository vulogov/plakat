#!/usr/bin/env bash
# 04_blend.sh — v2: smooth pasted edges with --artefact-blend.
#
# Adds a masked img2img pass after the alpha composite. The model
# re-paints inside a feathered mask covering each artefact's zone,
# softening hard silhouette edges and absorbing modest lighting
# mismatches.
#
# Cost: one extra short denoise pass (~2–5 s on GPU). Strength is
# the standard img2img dial — 0.3 is the recommended default.
#
# To compare with v1 (no blend), run 02_zones.sh first, then this
# script — same seed, side-by-side, same prompt. The blended version
# should have softer edges where the silhouettes meet the scene.
#
# Output: out/zones-tutorial/04_blend/plakat-*.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/zones-tutorial/04_blend
mkdir -p "$OUT"

"$PLAKAT" generate "a peaceful rural valley at golden hour" \
    --artefact sun@sky/right \
    --artefact cloud@sky/left \
    --artefact pine@far_plan/left \
    --artefact oak@middle_plan/right \
    --artefact cottage@close_plan/center \
    --artefact-library library \
    --artefact-blend \
    --artefact-blend-strength 0.30 \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 1002 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-1002.png"
echo "Compare with out/zones-tutorial/02_zones/plakat-1002.png — same"
echo "seed, same artefacts, plus a 30% blend pass."
