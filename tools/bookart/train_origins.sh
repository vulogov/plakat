#!/usr/bin/env bash
# Train the three bookart origin LoRAs (ROADMAP_BOOKART_1 B4 / G0.3) on the grayscale
# corpus, sequentially (plakat refuses concurrent heavy runs). sd15 base, 256², rank 16.
# Artefacts → bake/bookart-<origin>.safetensors, with periodic checkpoints to sweep.
#
#   tools/bookart/train_origins.sh            # all three
#   STEPS=90 tools/bookart/train_origins.sh   # quicker smoke run
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"
STEPS="${STEPS:-180}"
SIZE="${SIZE:-256}"
PLAKAT="${PLAKAT:-./target/release/plakat}"
mkdir -p bake

# origin → training corpus (russian=Bilibin, english=Beardsley, japanese=Hokusai)
PAIRS=("russian:bilibin" "english:beardsley" "japanese:hokusai")

for pair in "${PAIRS[@]}"; do
  origin="${pair%%:*}"; artist="${pair##*:}"
  echo "===== $(date '+%H:%M:%S') train origin=$origin artist=$artist steps=$STEPS ====="
  "$PLAKAT" style train \
    --from-dir "datasets/bookart_training/${artist}_gray" \
    --base sd15 \
    --trigger "bookart_${origin} style" \
    --size "$SIZE" --steps "$STEPS" --rank 16 --checkpoint-every 60 \
    --out "bake/bookart-${origin}.safetensors" \
    || echo "!! FAILED origin=$origin"
  echo
done

echo "===== $(date '+%H:%M:%S') done ====="
ls -la bake/*.safetensors 2>/dev/null
