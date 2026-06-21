#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — `plakat.map.*` scripting (MAP-4)
# ===================================================================
# A Bund script renders a map into an image handle (`plakat.map.render`) then
# saves it (`plakat.save`). Deterministic linework — no GPU — so the saved map
# must be byte-identical to the direct `plakat map --map-render`, proving the
# scripting surface shares the exact render path.
#
#   plakat run corpus/map_script.bund   (or: corpus/map_script.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
cd "$ROOT"

rm -rf "$ROOT/out"
"$PLAKAT" run corpus/map_script.bund >/dev/null
cmp "$ROOT/out/island-map.png" "$ROOT/corpus/images/map/island-render.png" \
  || { echo "✗ scripting map (plakat.map.render) drifted from the direct --map-render"; exit 1; }
rm -rf "$ROOT/out"

echo "✓ map scripting (MAP-4): plakat.map.render → handle → plakat.save byte-identical to --map-render"
