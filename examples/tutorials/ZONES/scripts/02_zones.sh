#!/usr/bin/env bash
# 02_zones.sh — multiple artefacts placed in different zones.
#
# Demonstrates `NAME@ZONE` syntax: override each artefact's natural
# zone, including the depth band (sky / far / middle / close) and
# the horizontal band (left / center / right).
#
# Layout:
#   sun       → sky / right        (top-right)
#   cloud     → sky / left         (top-left)
#   pine      → far_plan / left    (mid-left, behind oak)
#   oak       → middle_plan / right (foreground tree)
#   cottage   → close_plan / center (front and centre)
#
# Z-order = flag order: later flags render on top.
#
# Output: out/zones-tutorial/02_zones/plakat-*.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/zones-tutorial/02_zones
mkdir -p "$OUT"

"$PLAKAT" generate "a peaceful rural valley at golden hour" \
    --artefact sun@sky/right \
    --artefact cloud@sky/left \
    --artefact pine@far_plan/left \
    --artefact oak@middle_plan/right \
    --artefact cottage@close_plan/center \
    --artefact-library library \
    --seed 1002 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-1002.png"
echo "Five artefacts arranged across the zone grid. Note z-order:"
echo "  sun behind cloud (cloud came after), cottage on top of everything."
