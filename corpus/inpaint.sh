#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — inpaint (img2img --mask)
# ===================================================================
# Repaint ONLY a masked region — here the sky of a committed SD 1.5
# landscape — while preserving the rest. The mask is a committed asset
# (assets/inpaint-sky-mask.png): white = repaint, black = preserve.
# Ungated SD 1.5, Metal-safe, fully self-contained (input + mask committed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" img2img "$ROOT/corpus/images/sd15/sd15_landscape/plakat-42.png" \
  --prompt "a dramatic stormy sunset sky, glowing orange and violet clouds, god rays" \
  --mask     "$ROOT/corpus/assets/inpaint-sky-mask.png" \
  --model sd15 --strength 0.85 --steps 28 --seed 42 --device metal \
  --out "$ROOT/corpus/images/inpaint"

echo "✓ wrote corpus/images/inpaint/ (sky region repainted, the rest preserved)"
