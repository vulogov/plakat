#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — art-medium looks (--look)
# ===================================================================
# One subject rendered across the 8 `--look` art-medium presets on SDXL.
# Each look composes a medium-specific prompt + sampler/steps/guidance AND
# auto-discovers a per-look medium LoRA from Civitai — base-matched to SDXL,
# so the discovery pulls SDXL medium LoRAs. SDXL is a far stronger base than
# SD 1.5 (which muddied art styles), so the looks render crisply.
#
# `--smart-discovery` runs the candidate pool through a small local LLM that
# picks the best STYLE LoRA and rejects character LoRAs (generic medium terms
# otherwise match anime characters that hijack the subject). Falls back to the
# SDXL prompt preset when the judge finds nothing suitable.
#
# Needs CIVITAI_API_KEY: the per-look LoRA downloads are auth-gated, reliable
# now the download timeout(0) bug is fixed. `--scheduler euler-a` because the
# look presets otherwise pick a DPM++/Karras scheduler candle's Metal backend
# can't run. Add `--offline` to skip discovery (prompt presets only, no key).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
SUBJECT="a stone cottage by a forest stream"

for LOOK in ink-wash watercolor oil-painting charcoal pencil chalk-pastel linocut gouache; do
  # chalk-pastel has no genuine STYLE LoRA on Civitai — discovery only finds
  # characters (e.g. "Professor Chalk", which fooled the judge on the name), so
  # it renders prompt-only on SDXL, which is clean. The rest use the LLM-judged
  # style LoRA via --smart-discovery.
  DISCO="--smart-discovery"
  [ "$LOOK" = "chalk-pastel" ] && DISCO="--offline"
  "$PLAKAT" generate "$SUBJECT" \
    --model sdxl --look "$LOOK" $DISCO \
    --scheduler euler-a \
    --steps 30 --size 1024x1024 --seed 42 --device metal \
    --out "$ROOT/corpus/images/looks/$LOOK"
done

echo "✓ wrote corpus/images/looks/{ink-wash,watercolor,oil-painting,charcoal,pencil,chalk-pastel,linocut,gouache}/"
