#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — stylize (IP-Adapter style transfer)
# ===================================================================
# Transfer the STYLE of a reference image onto an input subject via
# IP-Adapter (SD 1.5) — distinct from `img2img` (prompt-steered) and from
# trained style LoRAs (`style_train.sh`): stylize reads the *look* straight
# off the `--ref` image, no prompt or training. It works best on a bold
# single subject (here a portrait); busy scenes restyle weakly. `--out` is
# a FILE; higher `--strength` = heavier restyle. Ungated SD 1.5, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
mkdir -p "$ROOT/corpus/images/stylize"

"$PLAKAT" stylize \
  --in  "$ROOT/corpus/images/sd15/sd15_portrait/plakat-43.png" \
  --ref "$ROOT/corpus/style/watercolour/figures.jpeg" \
  --strength 0.7 --model sd15 --seed 42 --device metal \
  --out "$ROOT/corpus/images/stylize/portrait-watercolour.png"

echo "✓ wrote corpus/images/stylize/portrait-watercolour.png"
