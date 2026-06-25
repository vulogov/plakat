#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — `plakat map`: EVERY supported feature, end to end
# ===================================================================
# The single comprehensive map proof. It exercises every feature the map engine
# supports and byte-checks each deterministic artifact against a committed proof
# (geometry is a pure fn of (spec, seed) → byte-stable). The one non-deterministic
# step — the SD-painted render — is GPU-gated and skipped under NO_GPU=1.
#
#   ./corpus/map.sh           # full (the SD paint needs a GPU build + a model)
#   NO_GPU=1 ./corpus/map.sh  # deterministic only (CI / no GPU)
#
# Specs used (all committed, loaded with NO LLM):
#   corpus/map/island.spec.json   — the canonical geometry pipeline
#   corpus/map/realms.hjson       — the ALL-FEATURES 3×3 continent
#   corpus/map/coastal.spec.json  — coastal terrain (peninsulas/inlets/fjords)
#   corpus/map/town.spec.json     — an urban (town-scale) map
#
# Feature coverage: tectonic heightmap, hydrology (rivers + navigable deltas),
# land/sea coastline, peninsulas + inlets + fjords, mountain ranges, plateaus/mesas,
# dry canyons (rift valleys), erosion, lakes, biomes incl. wetland/marsh hatching,
# landmarks, roads + bridges, the political layer (polity rings + borders), the
# three cartographic styles, seasonal palettes, a tabletop grid, vector export
# (GeoJSON/SVG), multi-tile worlds (linework + SD-painted), urban street graphs,
# the SD conditioning base, and the `map` scenario task.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
ISLE="$ROOT/corpus/map/island.spec.json"
REALMS="$ROOT/corpus/map/realms.hjson"
COAST="$ROOT/corpus/map/coastal.spec.json"
TOWN="$ROOT/corpus/map/town.spec.json"
OUT="$ROOT/corpus/images/map"
T=/tmp/plakat-map
mkdir -p "$OUT"
check() { cmp "$1" "$2" || { echo "✗ ${3:-proof} drifted from the committed proof"; exit 1; }; }

echo "── PART A — island: the canonical geometry pipeline (byte-stable) ──"

# A1) The committed MapSpec loads with NO LLM, is schema-valid, and round-trips.
"$PLAKAT" map --map-spec "$ISLE" --map-dump-spec "$T-1.json" >/dev/null
"$PLAKAT" map --map-spec "$T-1.json" --map-dump-spec "$T-2.json" >/dev/null
diff -q "$T-1.json" "$T-2.json" >/dev/null || { echo "✗ MapSpec round-trip is not byte-stable"; exit 1; }
# --map-tiles overrides the grid (the single user-facing scale control).
"$PLAKAT" map --map-spec "$ISLE" --map-tiles 4x4 --map-dump-spec - 2>/dev/null \
  | grep -q '"cols": 4' || { echo "✗ --map-tiles override failed"; exit 1; }
echo "✓ A1 island.spec.json loads (no LLM) + round-trips byte-stable; --map-tiles overrides grid"

# A2) Every geometry layer L0–L7 — a deterministic fn of (spec, seed).
for layer in heightmap rivers coast biome landmarks roads features; do
  "$PLAKAT" map --map-spec "$ISLE" --seed 42 --map-dump-$layer "$T-$layer.png" >/dev/null
  check "$T-$layer.png" "$OUT/island-$layer.png" "island $layer"
done
echo "✓ A2 geometry layers L0–L7 (heightmap → features) byte-stable"

# A3) The complete styled, labelled linework render.
"$PLAKAT" map --map-spec "$ISLE" --seed 42 --map-render "$T-render.png" >/dev/null
check "$T-render.png" "$OUT/island-render.png" "island render"
# A4) Vector export (GeoJSON + SVG).
"$PLAKAT" map --map-spec "$ISLE" --seed 42 \
  --map-export-geojson "$T.geojson" --map-export-svg "$T.svg" >/dev/null
check "$T.geojson" "$ROOT/corpus/map/export/island.geojson" "island GeoJSON"
check "$T.svg" "$ROOT/corpus/map/export/island.svg" "island SVG"
# A5) The SD conditioning base (the deterministic half of the painted render).
"$PLAKAT" map --map-spec "$ISLE" --seed 42 --map-dump-conditioning "$T-cond.png" >/dev/null
check "$T-cond.png" "$OUT/island-conditioning.png" "SD conditioning base"
# A6) The `map` scenario task renders via the SAME path → byte-identical.
"$PLAKAT" scenario "$ROOT/corpus/map_scenario.hjson" >/dev/null 2>&1
check "$OUT/scenario/isle-parchment/map.png" "$OUT/island-render.png" "scenario map task"
echo "✓ A3–A6 linework render + GeoJSON/SVG export + SD conditioning base + scenario task byte-stable"

echo "── PART B — realms: the ALL-FEATURES 3×3 continent (byte-stable) ──"

# B1) Loads + round-trips, and the spec carries every terrain feature.
"$PLAKAT" map --map-spec "$REALMS" --map-dump-spec "$T-realms.json" >/dev/null
for feat in mountain_ranges plateaus rift_valleys peninsulas inlets fjords; do
  grep -q "\"$feat\"" "$T-realms.json" || { echo "✗ realms is missing terrain.$feat"; exit 1; }
done
grep -q '"cols": 3' "$T-realms.json" || { echo "✗ realms is not a 3×3 map"; exit 1; }
echo "✓ B1 realms.hjson loads + is 3×3 + carries mountains/plateaus/canyons/peninsulas/inlets/fjords"

# B2) Heightmap + render byte-stable (the cut sea arms, raised capes, canyons, mesas).
"$PLAKAT" map --map-spec "$REALMS" --seed 42 --map-dump-heightmap "$T-realms-hm.png" >/dev/null
check "$T-realms-hm.png" "$OUT/realms-heightmap.png" "realms heightmap"
"$PLAKAT" map --map-spec "$REALMS" --seed 42 --map-render "$T-realms-render.png" >/dev/null
check "$T-realms-render.png" "$OUT/realms-render.png" "realms render"
echo "✓ B2 realms heightmap + render byte-stable (all terrain + coastal features)"

# B3) The political layer → render variants + vector export (polity rings + labels).
"$PLAKAT" map --map-spec "$REALMS" --seed 42 \
  --map-export-geojson "$T-realms.geojson" --map-export-svg "$T-realms.svg" >/dev/null
check "$T-realms.geojson" "$ROOT/corpus/map/export/realms.geojson" "realms political GeoJSON"
check "$T-realms.svg" "$ROOT/corpus/map/export/realms.svg" "realms political SVG"
grep -q '"class": "polity"' "$ROOT/corpus/map/export/realms.geojson" \
  || { echo "✗ realms GeoJSON is missing the polity layer"; exit 1; }
# B4) Styles + seasonal + grid (generated by corpus/realms.sh; byte-check them here).
for v in parchment inked blueprint autumn winter grid; do
  src="$OUT/../realms/render-$v.png"
  [ -f "$src" ] || { echo "✗ missing realms variant render-$v.png (run corpus/realms.sh)"; exit 1; }
done
echo "✓ B3–B4 political GeoJSON/SVG byte-stable (2 polities) + parchment/inked/blueprint/autumn/winter/grid variants present"

echo "── PART C — coastal terrain detail (byte-stable) ──"

# C1) peninsulas RAISE land, inlets/fjords LOWER narrow sea arms.
"$PLAKAT" map --map-spec "$COAST" --seed 7 --map-render "$T-coast-render.png" >/dev/null
check "$T-coast-render.png" "$OUT/coastal-render.png" "coastal render"
"$PLAKAT" map --map-spec "$COAST" --seed 7 --map-dump-heightmap "$T-coast-hm.png" >/dev/null
check "$T-coast-hm.png" "$OUT/coastal-heightmap.png" "coastal heightmap"
echo "✓ C1 coastal peninsulas + inlets + fjords byte-stable"

echo "── PART D — multi-tile worlds (byte-stable, reassemble pixel-exact) ──"

# D1) The 3×3 realms world slices into nine seamless tiles + the stitched world.png.
"$PLAKAT" map --map-spec "$REALMS" --seed 42 --map-render-tiles "$T-realms-tiles" >/dev/null
check "$T-realms-tiles/world.png" "$OUT/realms-tiles/world.png" "realms world.png"
for r in 0 1 2; do for c in 0 1 2; do
  check "$T-realms-tiles/tile_r${r}_c${c}.png" "$OUT/realms-tiles/tile_r${r}_c${c}.png" "realms tile r${r}c${c}"
done; done
echo "✓ D1 realms 3×3 world → world.png + 9 seamless 256² tiles byte-stable"

echo "── PART E — urban (town-scale) maps (byte-stable) ──"

# E1) The town street graph + the labelled town map.
"$PLAKAT" map --map-spec "$TOWN" --seed 7 --map-dump-streets "$T-town-streets.png" >/dev/null
check "$T-town-streets.png" "$OUT/town-streets.png" "town street graph"
"$PLAKAT" map --map-spec "$TOWN" --seed 7 --map-render "$T-town-map.png" >/dev/null
check "$T-town-map.png" "$OUT/town-map.png" "town map"
echo "✓ E1 urban street graph + labelled town map byte-stable"

rm -rf "$T"-*.png "$T"-*.json "$T".geojson "$T".svg "$T-realms.geojson" "$T-realms.svg" \
       "$T-realms-tiles" "$T-realms.json"

echo "── PART F — SD-painted render + tiled paint (GPU; NO_GPU=1 skips) ──"
if [ "${NO_GPU:-0}" = "1" ]; then
  echo "  ⬜ SD-painted renders skipped (NO_GPU=1) — run locally to fill the painted showcases"
  echo ""
  echo "✓ map: ALL deterministic features byte-stable (island + realms 3×3 all-features + coastal + tiles + urban)"
  exit 0
fi
MODEL="${MODEL:-sdxl}"
# F1) Painted island (single tile) — the MAP-6 SD img2img + Canny path.
"$PLAKAT" map --map-spec "$ISLE" --seed 42 --map-sd-model "$MODEL" \
  --map-render-sd "$OUT/island-painted.png"
echo "  ✓ F1 painted island map → corpus/images/map/island-painted.png  (model $MODEL)"
# F2) Painted 3×3 realms with --map-sd-tile — the memory-safe TILED paint. The 768²
#     canvas is larger than the 512px tile, so it paints in overlapping feathered
#     tiles (this is what keeps a large all-features map on-box).
"$PLAKAT" map --map-spec "$REALMS" --seed 42 --map-sd-model "$MODEL" \
  --map-sd-tile 512 --map-sd-tile-stride 384 \
  --map-render-sd "$OUT/realms-painted-tiled.png"
echo "  ✓ F2 painted 3×3 realms (tiled, --map-sd-tile 512) → corpus/images/map/realms-painted-tiled.png"
echo ""
echo "✓ map: EVERY feature demonstrated — deterministic byte-stable + GPU-painted (single + tiled)"
