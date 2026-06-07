#!/usr/bin/env bash
# Generate watercolour images with the trained style LoRA (FAST).
# Requires corpus/style/watercolour.safetensors — run style_train.sh first
# (training and generation are separated so rendering never retrains).
#
# The LoRA loads via plain `--lora`; include the trigger phrase in the
# prompt to invoke the style. Proves the trained style transfers onto
# fresh subjects the model never saw in the exemplars.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
LORA="$ROOT/corpus/style/watercolour.safetensors"
TRIGGER="wcstyle watercolour painting illustration"

gen() {
  "$PLAKAT" generate "$1, $TRIGGER" \
    --model sd35-medium --lora "$LORA" \
    --steps 26 --size 768x768 --seed 42 --device metal \
    --out "$ROOT/corpus/images/style/$2"
}

gen "a fishing harbour with wooden boats and distant hills"            harbour
gen "a snow-covered mountain village among pines at dusk"             winter-village
gen "a quiet riverside orchard with a wooden footbridge in autumn"    river-orchard

echo "✓ wrote corpus/images/style/{harbour,winter-village,river-orchard}/"
