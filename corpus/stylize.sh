#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — stylize (IP-Adapter style transfer)
# ===================================================================
# Transfer the STYLE of a reference image onto an input subject via
# IP-Adapter — a quick, cheap, dirty styling tool, distinct from `img2img`
# (prompt-steered) and from trained style LoRAs (`style_train.sh`): it reads
# the *look* straight off the `--ref` image, no prompt or training.
#
# ⚠ MODEL LIMITATION: stylize currently runs on SD 1.5, which renders
# photorealistically and does NOT transfer painterly/textured styles well —
# the output tends to stay photoreal regardless of the reference (the same
# base limit that made the anime genre fail on SD 1.5 until it moved to SDXL).
# So SD 1.5 stylize is best treated as a ref-guided *variation* tool, not a
# style machine. For real style transfer use the LoRA paths (style_train.sh /
# civitai.sh). An SDXL stylize path (stronger base) is the planned fix.
#
# `--ref-blur` is the "style not content" knob: it Gaussian-blurs the ref
# before encoding, so CLIP sees the broad style and NOT the ref's subject
# (which otherwise hijacks the output) — though blur also softens texture, so
# it suits palette-driven refs more than fine-texture ones. `--ref-weight`
# scales the ref's influence. `--out` is a FILE; higher `--strength` = heavier
# restyle. Ungated, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
mkdir -p "$ROOT/corpus/images/stylize"

"$PLAKAT" stylize \
  --in  "$ROOT/corpus/images/sd15/sd15_portrait/plakat-43.png" \
  --ref "$ROOT/corpus/style/watercolour/figures.jpeg" \
  --ref-blur 10 --ref-weight 1.0 \
  --strength 0.7 --model sd15 --seed 42 --device metal \
  --out "$ROOT/corpus/images/stylize/portrait-watercolour.png"

echo "✓ wrote corpus/images/stylize/portrait-watercolour.png"
