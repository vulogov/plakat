#!/usr/bin/env bash
# Train the watercolour style LoRA from the 9 exemplars in
# corpus/style/watercolour/ against SD3.5 (Phase 1).
#
# SLOW — full backprop through the 2.5B MMDiT is ~1.7 min/step on Metal,
# so ~90 steps is a couple of hours. Run ONCE; the output safetensors is
# reused by style_gen.sh (generation is separated so you don't retrain to
# render). Periodic checkpoints mean corpus/style/watercolour.safetensors
# is usable from step 30 onward even if you stop early.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" style train \
  --from-dir "$ROOT/corpus/style/watercolour" \
  --base    sd35 \
  --trigger "wcstyle watercolour painting illustration" \
  --out     "$ROOT/corpus/style/watercolour.safetensors" \
  --steps 90 --rank 16 --size 256

echo "✓ trained → corpus/style/watercolour.safetensors"
echo "  now run: corpus/style_gen.sh"
