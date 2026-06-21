#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — urban fabric (MAP-5: street graph + blocks)
# ===================================================================
# A city/town-scale spec (`scale_tier` 10–12 with an `urban` block) renders an
# urban street graph: a wall ring + gates, arterials centre→gate, a ring road,
# and a minor-street grid with block parcels — a pure fn of (spec, seed), so the
# committed proof is byte-stable. No GPU, no network.
#
#   plakat map --map-spec corpus/map/town.spec.json --seed 7 --map-dump-streets out.png
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SPEC="$ROOT/corpus/map/town.spec.json"
OUT="$ROOT/corpus/images/map"
mkdir -p "$OUT"

# The committed town spec loads (no LLM) + round-trips byte-stable.
"$PLAKAT" map --map-spec "$SPEC" --map-dump-spec /tmp/plakat-town-1.json >/dev/null
"$PLAKAT" map --map-spec /tmp/plakat-town-1.json --map-dump-spec /tmp/plakat-town-2.json >/dev/null
diff -q /tmp/plakat-town-1.json /tmp/plakat-town-2.json >/dev/null \
  || { echo "✗ urban MapSpec round-trip is not byte-stable"; exit 1; }
rm -f /tmp/plakat-town-1.json /tmp/plakat-town-2.json

# U0+U1 — the street graph + blocks are byte-stable vs the committed proof.
"$PLAKAT" map --map-spec "$SPEC" --seed 7 --map-dump-streets /tmp/plakat-town-streets.png >/dev/null
cmp /tmp/plakat-town-streets.png "$OUT/town-streets.png" \
  || { echo "✗ urban street graph drifted from the committed proof"; exit 1; }
rm -f /tmp/plakat-town-streets.png

echo "✓ map urban (MAP-5): town.spec.json loads + round-trips byte-stable"
echo "  + U0+U1: street graph (wall/gates/arterials/ring/grid) + blocks byte-stable vs corpus/images/map/town-streets.png"
