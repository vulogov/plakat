#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — art-medium looks (--look)
# ===================================================================
# One subject rendered across the 8 bundled `--look` art-medium presets.
# Each `--look` auto-discovers + applies a medium LoRA (Civitai -> HF ->
# local cache) plus a prompt/sampler preset — a bundled, *generic*
# counterpart to the trained style LoRAs from style_train.sh (which learn a
# *specific* style from your images).
#
# NOTE: the first run of each look downloads a small LoRA from Civitai
# (network). Pass PLAKAT_OFFLINE=1 / `--offline` to skip discovery once the
# LoRAs are cached. Ungated SD 1.5, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SUBJECT="a stone cottage by a forest stream"

for LOOK in ink-wash watercolor oil-painting charcoal pencil chalk-pastel linocut gouache; do
  "$PLAKAT" generate "$SUBJECT" \
    --model sd15 --look "$LOOK" \
    --steps 28 --size 512x512 --seed 42 --device metal \
    --out "$ROOT/corpus/images/looks/$LOOK"
done

echo "✓ wrote corpus/images/looks/{ink-wash,watercolor,oil-painting,charcoal,pencil,chalk-pastel,linocut,gouache}/"
