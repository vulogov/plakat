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
echo "✓ map (MAP-1): island.spec.json loads (no LLM) + round-trips byte-stable; --map-tiles overrides grid"
