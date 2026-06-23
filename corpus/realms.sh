#!/usr/bin/env bash
# ===================================================================
# plakat map — compile the all-features showcase (realms.hjson)
# ===================================================================
#   ./corpus/realms.sh
#   plakat gallery corpus/images/realms --recursive --out /tmp/realms-gallery.md
#
# "Compiles" the comprehensive HJSON map spec (corpus/map/realms.hjson) into
# its full artifact set: every geometry layer, the styled labelled map, the
# GeoJSON/SVG vector export, and the seasonal + tabletop-grid variants. One
# continental spec exercising EVERY geographical feature plakat realizes —
# mountains, plateaus/mesas, dry canyons, lakes, a wetland/swamp region,
# rivers + coast, multiple biomes, the political layer (polity rings + a
# disputed border), landmarks, a road + bridge.
#
# Pure geometry — deterministic, NO GPU, NO network, NO API key. The byte-
# stability of the heightmap + parchment render is enforced by corpus/map.sh.
#
# Override with env vars: PLAKAT, SEED.
# ===================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SPEC="$ROOT/corpus/map/realms.hjson"
SEED="${SEED:-42}"
OUT="$ROOT/corpus/images/realms"
mkdir -p "$OUT" "$ROOT/corpus/map/export"

# 0) Validate the HJSON spec loads (no LLM) and round-trips.
"$PLAKAT" map --map-spec "$SPEC" --map-dump-spec /tmp/realms-rt.json >/dev/null
echo "✓ realms.hjson loads + round-trips"

# 1) Every geometry layer (L1–L7) — the engine's view of the world.
"$PLAKAT" map --map-spec "$SPEC" --seed "$SEED" \
  --map-dump-heightmap "$OUT/heightmap.png" \
  --map-dump-rivers    "$OUT/rivers.png" \
  --map-dump-coast     "$OUT/coast.png" \
  --map-dump-biome     "$OUT/biome.png" \
  --map-dump-landmarks "$OUT/landmarks.png" \
  --map-dump-roads     "$OUT/roads.png" \
  --map-dump-features  "$OUT/features.png" >/dev/null
echo "✓ geometry layers (heightmap → features) → corpus/images/realms/"

# 2) The styled, labelled map in each cartographic style.
for STYLE in parchment inked blueprint; do
  "$PLAKAT" map --map-spec "$SPEC" --seed "$SEED" --map-style "$STYLE" \
    --map-render "$OUT/render-$STYLE.png" >/dev/null
done
echo "✓ styled renders (parchment / inked / blueprint)"

# 3) Seasonal palettes + a tabletop coordinate grid (1.11.0 features).
"$PLAKAT" map --map-spec "$SPEC" --seed "$SEED" --map-season autumn \
  --map-render "$OUT/render-autumn.png" >/dev/null
"$PLAKAT" map --map-spec "$SPEC" --seed "$SEED" --map-season winter \
  --map-render "$OUT/render-winter.png" >/dev/null
"$PLAKAT" map --map-spec "$SPEC" --seed "$SEED" --map-grid 8 \
  --map-render "$OUT/render-grid.png" >/dev/null
echo "✓ seasonal (autumn/winter) + 8×8 grid variants"

# 4) Vector export (GeoJSON + SVG).
"$PLAKAT" map --map-spec "$SPEC" --seed "$SEED" \
  --map-export-geojson "$ROOT/corpus/map/export/realms.geojson" \
  --map-export-svg     "$ROOT/corpus/map/export/realms.svg" >/dev/null
echo "✓ vector export → corpus/map/export/realms.{geojson,svg}"

rm -f /tmp/realms-rt.json
echo "✓ realms.hjson compiled — all geographical features in corpus/images/realms/"
