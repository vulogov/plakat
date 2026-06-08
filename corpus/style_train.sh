#!/usr/bin/env bash
# Train a watercolour style LoRA from corpus/style/watercolour/ on a chosen
# base. Usage: ./style_train.sh [sd15|sdxl|sd35]   (default: sd15).
#
# SLOW — full back-prop through the base UNet/MMDiT. Run ONCE; the output is
# reused by style_gen.sh (training and generation are separated so you never
# retrain to render). Checkpoints every 30 steps, so a long run is usable
# early. SD1.5 is the fastest to validate; SDXL/SD3.5 are heavier.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
BASE="${1:-sd15}"

case "$BASE" in
  sd15) SIZE=512; OUT=watercolour-sd15.safetensors ;;   # kohya LoRA
  sdxl) SIZE=512; OUT=watercolour-sdxl.safetensors ;;   # kohya LoRA (gated:no)
  sd35) SIZE=256; OUT=watercolour.safetensors ;;        # diffusers-PEFT (gated)
  *) echo "base must be sd15 | sdxl | sd35"; exit 1 ;;
esac

"$PLAKAT" style train \
  --from-dir "$ROOT/corpus/style/watercolour" \
  --base    "$BASE" \
  --trigger "wcstyle watercolour painting illustration" \
  --out     "$ROOT/corpus/style/$OUT" \
  --steps 90 --rank 16 --size "$SIZE"

echo "✓ trained $BASE → corpus/style/$OUT"
echo "  now run: corpus/style_gen.sh $BASE"
