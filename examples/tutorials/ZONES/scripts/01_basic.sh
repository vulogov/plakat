#!/usr/bin/env bash
# 01_basic.sh — single artefact placed in its natural zone.
#
# Generates a meadow image, then alpha-composites the bundled `oak`
# silhouette into the middle_plan zone (the oak's library default).
# This is the simplest possible artefact usage — no overrides, no
# blend, no smart-zones. Just "drop a tree onto a meadow."
#
# Output: out/zones-tutorial/01_basic/plakat-*.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/zones-tutorial/01_basic
mkdir -p "$OUT"

"$PLAKAT" generate "a quiet green meadow under a wide blue sky" \
    --artefact oak \
    --artefact-library library \
    --seed 1001 \
    --out "$OUT"

echo
echo "Wrote $OUT/plakat-1001.png"
echo "Open it: the oak appears in the middle band (its library default)."
