#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — stylize (IP-Adapter ref-guided variation)
# ===================================================================
# Stylize reads the *look* of a `--ref` image and applies it to an input
# subject via IP-Adapter — no prompt, no training. Runs on SD 1.5 or, by
# default here, SDXL (`--model sdxl`: sharper, native 1024²; small inputs are
# upscaled, since SDXL glitches below ~1024). SD 1.5 stays as a fallback.
#
# ⚠ WHAT IT IS (and isn't): the IP-Adapter transfers a reference's CONTENT /
# appearance / palette, NOT painterly *texture*. So stylize is a ref-guided
# *variation* tool — output tends to stay photoreal even on SDXL. This is an
# IP-Adapter limit, NOT a base limit (SDXL paints fine from prompts/LoRAs). For
# true painterly STYLE transfer use the LoRA paths (style_train.sh / civitai.sh)
# or `--look`.
#
# `--ref-blur` Gaussian-blurs the ref before encoding to suppress its content
# (blur also softens texture, so it suits palette-driven refs). `--ref-weight`
# scales the ref's influence. `--out` is a FILE; higher `--strength` = heavier
# restyle. Ungated, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
mkdir -p "$ROOT/corpus/images/stylize"

"$PLAKAT" stylize \
  --in  "$ROOT/corpus/images/sd15/sd15_portrait/plakat-43.png" \
  --ref "$ROOT/corpus/style/watercolour/figures.jpeg" \
  --ref-blur 4 --ref-weight 1.0 \
  --strength 0.6 --model sdxl --seed 42 --device metal \
  --out "$ROOT/corpus/images/stylize/portrait-stylized.png"

echo "✓ wrote corpus/images/stylize/portrait-stylized.png"
