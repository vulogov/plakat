#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — regional prompting (MultiDiffusion)
# ===================================================================
# Different prompts in different REGIONS of ONE image: the base prompt sets the
# global scene + lighting, and each `--region "x0,y0,x1,y1:prompt"` (coords are
# [0,1] canvas fractions) steers its box. Here the left half is alpine and the
# right half is tropical — two biomes blended into one coherent golden-hour
# landscape. SD-family, native resolution.
#
#   Usage:  ./regional.sh [sd15|sdxl]     (default: sd15 — lighter / faster)
#
# GPU. Regional does (1 + N_regions) UNet passes per step, so it's heavier than a
# plain generate; the OOM guard aborts cleanly if RAM is tight. ⬜ until rendered.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
BASE="${1:-sd15}"

case "$BASE" in
  sd15) MODEL=sd15; SIZE=512  ;;
  sdxl) MODEL=sdxl; SIZE=1024 ;;
  *) echo "base must be sd15 | sdxl"; exit 1 ;;
esac

"$PLAKAT" generate \
  "a vast natural landscape at golden hour, cinematic wide shot, ultra-detailed, sharp focus" \
  --model "$MODEL" --size "$SIZE" --seed 42 --steps 30 --device metal \
  --region "0.0,0.0,0.5,1.0:towering snow-capped mountains and glaciers, alpine" \
  --region "0.5,0.0,1.0,1.0:a lush green tropical rainforest with a waterfall" \
  --out "$ROOT/corpus/images/regional"

echo "✓ regional ($MODEL): left = alpine, right = tropical, one scene → corpus/images/regional/"
