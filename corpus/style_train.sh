#!/usr/bin/env bash
# Train a watercolour style LoRA from corpus/style/watercolour/ on a chosen
# base. Usage: ./style_train.sh [sd15|sdxl|sd35]   (default: sd15).
#
# SLOW — full back-prop through the base UNet/MMDiT. Run ONCE; the output is
# reused by style_gen.sh (training and generation are separated). Checkpoints
# every 30 steps.
#
# Per-base recipe: SD 1.5 is a weaker base whose UNet already denoises the
# latents almost perfectly, so the LoRA gets a tiny gradient — it needs a
# much HOTTER lr + more steps + more rank to imprint the style (gradient
# clipping in the trainer keeps that stable through loss spikes). SDXL /
# SD 3.5 are stronger bases and learn the style with the lighter recipe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
BASE="${1:-sd15}"

case "$BASE" in
  sd15) SIZE=512; LR=3e-4;   STEPS=240; RANK=32; OUT=watercolour-sd15.safetensors ;;
  sdxl) SIZE=512; LR=1.5e-4; STEPS=90;  RANK=16; OUT=watercolour-sdxl.safetensors ;;
  sd35) SIZE=256; LR=1.5e-4; STEPS=90;  RANK=16; OUT=watercolour.safetensors ;;
  *) echo "base must be sd15 | sdxl | sd35"; exit 1 ;;
esac

"$PLAKAT" style train \
  --from-dir "$ROOT/corpus/style/watercolour" \
  --base    "$BASE" \
  --trigger "wcstyle watercolour painting illustration" \
  --out     "$ROOT/corpus/style/$OUT" \
  --steps "$STEPS" --rank "$RANK" --size "$SIZE" --lr "$LR"

echo "✓ trained $BASE → corpus/style/$OUT"
echo "  now run: corpus/style_gen.sh $BASE"
