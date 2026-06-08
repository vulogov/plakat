#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — art-medium looks (--look)
# ===================================================================
# One subject rendered across the 8 bundled `--look` art-medium presets.
# Each look composes a medium-specific prompt + sampler/steps/guidance AND
# auto-discovers a per-look medium LoRA from Civitai for a stronger effect —
# the richer counterpart to the bundled prompt presets and to the trained
# style LoRAs from style_train.sh.
#
# Needs CIVITAI_API_KEY: the per-look LoRA downloads are auth-gated. The
# discovery is reliable now that the download timeout(0) bug is fixed.
# `--scheduler euler-a` because the look presets otherwise pick a DPM++/
# Karras scheduler candle's Metal backend can't run. Add `--offline` to skip
# the LoRA discovery (prompt presets only, no key needed). SD 1.5, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SUBJECT="a stone cottage by a forest stream"

for LOOK in ink-wash watercolor oil-painting charcoal pencil chalk-pastel linocut gouache; do
  "$PLAKAT" generate "$SUBJECT" \
    --model sd15 --look "$LOOK" \
    --scheduler euler-a \
    --steps 28 --size 512x512 --seed 42 --device metal \
    --out "$ROOT/corpus/images/looks/$LOOK"
done

echo "✓ wrote corpus/images/looks/{ink-wash,watercolor,oil-painting,charcoal,pencil,chalk-pastel,linocut,gouache}/"
