#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — map (MAP-1: MapSpec v2 + LLM geographic parser)
# ===================================================================
# `plakat map` turns a prose world description into a structured MapSpec v2 (the
# input the geometry engine + renderer consume in MAP-2+). MAP-1 ships the spec
# schema + the LLM parser; this proof exercises the DETERMINISTIC path — load a
# committed spec (no LLM), confirm it's valid + round-trips byte-stable, and that
# the scale flags override the grid. No GPU, no network, no API key.
#
# (The LLM parse path — prose → spec — reuses the --enhance provider stack;
# verify it live with e.g. `plakat map "a volcanic island kingdom" --map-dump-spec -`.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SPEC="$ROOT/corpus/map/island.spec.json"

# 1) The committed MapSpec loads with NO LLM and is schema-valid.
"$PLAKAT" map --map-spec "$SPEC" --map-dump-spec /tmp/plakat-map-1.json

# 2) Serialization round-trips byte-stable (idempotent serialize/deserialize).
"$PLAKAT" map --map-spec /tmp/plakat-map-1.json --map-dump-spec /tmp/plakat-map-2.json
diff -q /tmp/plakat-map-1.json /tmp/plakat-map-2.json >/dev/null \
  || { echo "✗ MapSpec round-trip is not byte-stable"; exit 1; }

# 3) --map-tiles overrides the grid (the single user-facing scale control).
"$PLAKAT" map --map-spec "$SPEC" --map-tiles 4x4 --map-dump-spec - 2>/dev/null \
  | grep -q '"cols": 4' || { echo "✗ --map-tiles override failed"; exit 1; }

rm -f /tmp/plakat-map-1.json /tmp/plakat-map-2.json

# 4) MAP-2 geometry is a deterministic function of (spec, seed) — re-dump each
#    layer's artifact and compare byte-for-byte against the committed proof.
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-heightmap /tmp/plakat-map-hm.png >/dev/null
cmp /tmp/plakat-map-hm.png "$ROOT/corpus/images/map/island-heightmap.png" \
  || { echo "✗ heightmap PNG drifted from the committed proof"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-rivers /tmp/plakat-map-riv.png >/dev/null
cmp /tmp/plakat-map-riv.png "$ROOT/corpus/images/map/island-rivers.png" \
  || { echo "✗ river overlay drifted from the committed proof"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-coast /tmp/plakat-map-coast.png >/dev/null
cmp /tmp/plakat-map-coast.png "$ROOT/corpus/images/map/island-coast.png" \
  || { echo "✗ coastline drifted from the committed proof"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-biome /tmp/plakat-map-biome.png >/dev/null
cmp /tmp/plakat-map-biome.png "$ROOT/corpus/images/map/island-biome.png" \
  || { echo "✗ biome map drifted from the committed proof"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-landmarks /tmp/plakat-map-lm.png >/dev/null
cmp /tmp/plakat-map-lm.png "$ROOT/corpus/images/map/island-landmarks.png" \
  || { echo "✗ landmark placement drifted from the committed proof"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-roads /tmp/plakat-map-roads.png >/dev/null
cmp /tmp/plakat-map-roads.png "$ROOT/corpus/images/map/island-roads.png" \
  || { echo "✗ road network drifted from the committed proof"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-features /tmp/plakat-map-feat.png >/dev/null
cmp /tmp/plakat-map-feat.png "$ROOT/corpus/images/map/island-features.png" \
  || { echo "✗ feature overlay drifted from the committed proof"; exit 1; }
rm -f /tmp/plakat-map-hm.png /tmp/plakat-map-riv.png /tmp/plakat-map-coast.png /tmp/plakat-map-biome.png /tmp/plakat-map-lm.png /tmp/plakat-map-roads.png /tmp/plakat-map-feat.png

echo "✓ map (MAP-1): island.spec.json loads (no LLM) + round-trips byte-stable; --map-tiles overrides grid"
echo "  + MAP-2 (L0+L1): tectonic heightmap byte-stable vs corpus/images/map/island-heightmap.png"
echo "  + MAP-2 (L2): river network byte-stable vs corpus/images/map/island-rivers.png"
echo "  + MAP-2 (L3): land/sea + coastline byte-stable vs corpus/images/map/island-coast.png"
echo "  + MAP-2 (L4): biome map byte-stable vs corpus/images/map/island-biome.png"
echo "  + MAP-2 (L5): landmarks resolved + placed byte-stable vs corpus/images/map/island-landmarks.png"
echo "  + MAP-2 (L6): road network byte-stable vs corpus/images/map/island-roads.png"
echo "  + MAP-2 (L7): assembled feature overlay byte-stable vs corpus/images/map/island-features.png — geometry engine COMPLETE"
