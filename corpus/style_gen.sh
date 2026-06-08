#!/usr/bin/env bash
# Generate watercolour images with a trained style LoRA (FAST).
# Usage: ./style_gen.sh [sd15|sdxl|sd35]   (default: sd15).
# Requires the matching corpus/style/watercolour*.safetensors — run
# style_train.sh <base> first (training and generation are separated).
#
# The LoRA loads via `--lora <path>:<scale>`. All three look right at scale
# 1.0 — the sd15 LoRA is the swept step-120 checkpoint (a hotter scale just
# pushed it toward the over-cooked look).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
BASE="${1:-sd15}"
TRIGGER="wcstyle watercolour painting illustration"

case "$BASE" in
  sd15) MODEL=sd15;        LORA=watercolour-sd15.safetensors;  SCALE="";     SIZE=512x512; DIR=style-sd15 ;;
  sdxl) MODEL=sdxl;        LORA=watercolour-sdxl.safetensors;  SCALE="";     SIZE=768x768; DIR=style-sdxl ;;
  sd35) MODEL=sd35-medium; LORA=watercolour.safetensors;       SCALE="";     SIZE=768x768; DIR=style ;;
  *) echo "base must be sd15 | sdxl | sd35"; exit 1 ;;
esac

gen() {
  "$PLAKAT" generate "$1, $TRIGGER" \
    --model "$MODEL" --lora "$ROOT/corpus/style/$LORA$SCALE" \
    --steps 26 --size "$SIZE" --seed 42 --device metal \
    --out "$ROOT/corpus/images/$DIR/$2"
}

gen "a fishing harbour with wooden boats and distant hills"         harbour
gen "a snow-covered mountain village among pines at dusk"           winter-village
gen "a quiet riverside orchard with a wooden footbridge in autumn"  river-orchard

echo "✓ wrote corpus/images/$DIR/"
