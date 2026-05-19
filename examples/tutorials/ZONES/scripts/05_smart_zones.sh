#!/usr/bin/env bash
# 05_smart_zones.sh — v3: derive zones from depth + luminance.
#
# Replaces the rigid 4×3 grid with zones derived from the generated
# image itself. Depth (Depth-Anything-V2 small, ~99 MB on first run)
# drives the vertical bands; per-column luminance variance drives the
# horizontal split. Falls back to the grid with a warning if the
# depth model can't be loaded.
#
# This example uses a 16:9 panorama with a low horizon — the kind of
# scene where the rigid grid mismatches the painted layout. Smart
# zones tracks the actual horizon.
#
# To compare with the rigid grid: run this script, then re-run with
# the `--smart-zones` flag removed and the same seed. The sun + cloud
# placements should differ.
#
# Output: out/zones-tutorial/05_smart_zones/plakat-*.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/zones-tutorial/05_smart_zones
mkdir -p "$OUT"

"$PLAKAT" generate "a wide panoramic meadow with low horizon under a vast cloudy sky" \
    --aspect 16:9 --base 512 \
    --steps 20 --scheduler euler-a \
    --artefact sun@sky/right \
    --artefact cloud@sky/left \
    --artefact oak@middle_plan/center \
    --artefact-library library \
    --smart-zones \
    --seed 1005 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-1005.png"
echo "The sun and cloud land in the painted sky (top ~40-50%),"
echo "not the rigid top 25%."
