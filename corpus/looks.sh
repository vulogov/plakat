#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — art-medium looks (--look)
# ===================================================================
# One subject rendered across the 8 bundled `--look` art-medium presets.
# Each look composes a medium-specific prompt + sampler/steps/guidance — a
# bundled, *generic* counterpart to the trained style LoRAs from
# style_train.sh (which learn a *specific* style from your images).
#
# We run `--offline`: each look also has an OPTIONAL per-look LoRA that is
# auto-discovered from Civitai, but that path needs the network, is flaky
# (download timeouts), and isn't reliably available for SD 1.5 — so for a
# reproducible corpus run we use the prompt presets only. `--scheduler
# euler-a` because the look presets otherwise select a DPM++/Karras
# scheduler that uses F64 ops candle's Metal backend can't run. Drop
# `--offline` to let the LoRA discovery run. Ungated SD 1.5, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SUBJECT="a stone cottage by a forest stream"

# NOTE: 'linocut' is omitted — its catalog entry pins a Civitai LoRA that
# --offline doesn't suppress (it still tries to download + times out). The
# other 7 mediums resolve to their prompt presets offline. (Deferred bug.)
for LOOK in ink-wash watercolor oil-painting charcoal pencil chalk-pastel gouache; do
  "$PLAKAT" generate "$SUBJECT" \
    --model sd15 --look "$LOOK" --offline \
    --scheduler euler-a \
    --steps 28 --size 512x512 --seed 42 --device metal \
    --out "$ROOT/corpus/images/looks/$LOOK"
done

echo "✓ wrote corpus/images/looks/{ink-wash,watercolor,oil-painting,charcoal,pencil,chalk-pastel,gouache}/"
