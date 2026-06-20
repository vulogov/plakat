#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — map painted render (MAP-6: SD img2img + ControlNet)
# ===================================================================
# `plakat map --map-render-sd` paints the MAP-2 geometry: the styled base map
# (terrain/coast/rivers/roads, no labels) is run through SD img2img + a Canny
# ControlNet so it looks hand-drawn, then the 1.5.0 labels + furniture are
# re-composited on top so it stays legible.
#
# Two halves, mirroring the memory-wall discipline of the whole map track:
#
#   1) DETERMINISTIC (no GPU, byte-checked) — the SD *conditioning* (the styled
#      base, the img2img init + Canny source) is a pure fn of (spec, seed), so it
#      is byte-stable and committed like every other map layer.
#
#   2) PAINTED (GPU, non-deterministic, ⬜ until rendered) — the SD denoise. Any
#      model plakat supports works; SDXL-family models additionally get the
#      fantasy-map style LoRA by default. The output is NOT byte-checked (it's the
#      one non-deterministic map artifact, like every SD pipeline in the corpus).
#
#   Usage:  ./map_render.sh                 (sdxl + Muapi/fantasy-map LoRA)
#           MODEL=sd15 LORA=none ./map_render.sh   (any model, no LoRA)
#           MODEL=sdxl LORA="Muapi/fantasy-map" ./map_render.sh
#
# ⚠️ The painted half downloads the model (SDXL ~7 GB) + a ControlNet (~2.5 GB)
# on first use and runs a full SD denoise — minutes on Metal, and memory-bound
# (keep the island at its 512² canvas; bigger maps want the 1.6 tiled path).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SPEC="$ROOT/corpus/map/island.spec.json"
OUT="$ROOT/corpus/images/map"
MODEL="${MODEL:-sdxl}"
LORA="${LORA:-}"          # empty → model default (fantasy-map for SDXL, none else)
mkdir -p "$OUT"

# 1) DETERMINISTIC — the SD conditioning base is byte-stable vs the committed proof.
"$PLAKAT" map --map-spec "$SPEC" --seed 42 --map-dump-conditioning /tmp/plakat-map-cond.png >/dev/null
cmp /tmp/plakat-map-cond.png "$OUT/island-conditioning.png" \
  || { echo "✗ SD conditioning base drifted from the committed proof"; exit 1; }
rm -f /tmp/plakat-map-cond.png
echo "✓ map render (MAP-6): SD conditioning base byte-stable vs corpus/images/map/island-conditioning.png"

# 2) PAINTED — the SD render itself (GPU). Skipped in NO_GPU mode (CI / no model).
if [ "${NO_GPU:-0}" = "1" ]; then
  echo "  ⬜ painted SD render skipped (NO_GPU=1) — run locally to fill corpus/images/map/island-painted.png"
  exit 0
fi

LORA_ARG=()
[ -n "$LORA" ] && LORA_ARG=(--map-sd-lora "$LORA")
"$PLAKAT" map --map-spec "$SPEC" --seed 42 \
  --map-sd-model "$MODEL" "${LORA_ARG[@]}" \
  --map-render-sd "$OUT/island-painted.png"
echo "  ✓ painted map → corpus/images/map/island-painted.png  (model $MODEL, lora ${LORA:-auto})"
