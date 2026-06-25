#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — coastal terrain + multi-tile worlds + political export
# ===================================================================
# Closes the 1.14.0-C gap: the 1.13.0 map features (coastal terrain, multi-tile
# worlds, political export) shipped CLI-only with thin coverage. This proof
# exercises the DETERMINISTIC path end-to-end — no GPU, no network, no API key:
#
#   1) COASTAL  — corpus/map/coastal.spec.json (peninsulas + inlets + fjords)
#      renders byte-stable linework + heightmap (the cut sea arms + raised spit).
#   2) TILED    — the world slices into seamless tiles that reassemble pixel-exact.
#   3) POLITICAL— realms.hjson exports the polity layer to GeoJSON + SVG.
#   4) SCENARIO — the SAME features via a `plakat scenario` batch, byte-identical
#      to the direct CLI (proving the automation path shares the render code).
#
#   ./corpus/coastal.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SPEC="$ROOT/corpus/map/coastal.spec.json"
OUT="$ROOT/corpus/images/map"

# 1) COASTAL — peninsulas RAISE land, inlets/fjords LOWER narrow sea arms. The
#    geometry is a pure fn of (spec, seed) → byte-identical to the committed proof.
"$PLAKAT" map --map-spec "$SPEC" --seed 7 --map-render /tmp/plakat-coastal-render.png >/dev/null
cmp /tmp/plakat-coastal-render.png "$OUT/coastal-render.png" \
  || { echo "✗ coastal linework render drifted from the committed proof"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 7 --map-dump-heightmap /tmp/plakat-coastal-hm.png >/dev/null
cmp /tmp/plakat-coastal-hm.png "$OUT/coastal-heightmap.png" \
  || { echo "✗ coastal heightmap (cut fjords + raised peninsula) drifted"; exit 1; }
"$PLAKAT" map --map-spec "$SPEC" --seed 7 --map-dump-coast /tmp/plakat-coastal-coast.png >/dev/null
cmp /tmp/plakat-coastal-coast.png "$OUT/coastal-coast.png" \
  || { echo "✗ coastal land/sea coastline drifted"; exit 1; }
echo "✓ coastal terrain: peninsulas + inlets + fjords byte-stable vs corpus/images/map/coastal-{render,heightmap,coast}.png"

# 2) TILED — slice the continuous world into a 2×2 grid of seamless tiles. The
#    tiles + world.png are byte-identical, and reassemble pixel-exact (continuous
#    canvas sliced, never per-tile re-rendered).
"$PLAKAT" map --map-spec "$SPEC" --seed 7 --map-render-tiles /tmp/plakat-coastal-tiles >/dev/null
for f in world.png tile_r0_c0.png tile_r0_c1.png tile_r1_c0.png tile_r1_c1.png; do
  cmp "/tmp/plakat-coastal-tiles/$f" "$OUT/coastal-tiles/$f" \
    || { echo "✗ tiled world $f drifted from the committed proof"; exit 1; }
done
echo "  + multi-tile world: world.png + 4 seamless tiles byte-stable vs corpus/images/map/coastal-tiles/"

# 3) POLITICAL — export realms.hjson (the all-features continent) to vector. The
#    polity layer (territory markers + labels) lands in both GeoJSON and SVG.
"$PLAKAT" map --map-spec "$ROOT/corpus/map/realms.hjson" --seed 42 \
  --map-export-geojson /tmp/plakat-realms.geojson --map-export-svg /tmp/plakat-realms.svg >/dev/null
cmp /tmp/plakat-realms.geojson "$ROOT/corpus/map/export/realms.geojson" \
  || { echo "✗ realms political GeoJSON export drifted from the committed proof"; exit 1; }
cmp /tmp/plakat-realms.svg "$ROOT/corpus/map/export/realms.svg" \
  || { echo "✗ realms political SVG export drifted from the committed proof"; exit 1; }
grep -q '"class": "polity"' "$ROOT/corpus/map/export/realms.geojson" \
  || { echo "✗ realms GeoJSON is missing the polity layer"; exit 1; }
echo "  + political export: realms polity layer byte-stable vs corpus/map/export/realms.{geojson,svg}"

# 4) SCENARIO — the SAME coastal render + tiled world via a `plakat scenario` batch
#    must be byte-identical to the direct CLI (proving automation shares the path).
"$PLAKAT" scenario "$ROOT/corpus/map_coastal_scenario.hjson" >/dev/null 2>&1
cmp "$OUT/scenario/coastal/map.png" "$OUT/coastal-render.png" \
  || { echo "✗ scenario coastal task drifted from the direct --map-render"; exit 1; }
for f in world.png tile_r0_c0.png tile_r0_c1.png tile_r1_c0.png tile_r1_c1.png; do
  cmp "$OUT/scenario/coastal-world/$f" "$OUT/coastal-tiles/$f" \
    || { echo "✗ scenario tiled task $f drifted from the direct --map-render-tiles"; exit 1; }
done
echo "  + scenario: coastal + multi-tile tasks byte-identical to the direct CLI (corpus/map_coastal_scenario.hjson)"

rm -rf /tmp/plakat-coastal-render.png /tmp/plakat-coastal-hm.png /tmp/plakat-coastal-coast.png \
       /tmp/plakat-coastal-tiles /tmp/plakat-realms.geojson /tmp/plakat-realms.svg
