#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — Textual Inversion (embedding)
# ===================================================================
# A Textual Inversion "embedding" is a single trigger word that stands in for a
# learned bundle of concepts/quality cues. `plakat generate --embedding` injects
# one at runtime (SD 1.5 / SD 2.1 — SDXL dual-encoder TIs bail in the parser).
#
# This demo uses **EasyNegative**, the well-known SD 1.5 *negative* embedding:
# putting its trigger in `--negative` steers the model away from bad anatomy /
# blur / artefacts. We render a baseline and a +EasyNegative pair on the same
# seed so the quality lift (hands, faces) is visible side by side.
#
# `--embedding` takes `PATH_OR_REPO[:trigger][:scale]`. A bare repo looks for
# `learned_embeds.safetensors`; use `repo#file.safetensors` to name a different
# file (here EasyNegative ships `EasyNegative.safetensors`). Cached on first run.
# Inspect any TI file with:
#   plakat embedding info <file.safetensors>     # trigger word + dims + variant
# Ungated, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
# embed/EasyNegative — the canonical ungated `embed/` org mirror (gsdf/EasyNegative
# was emptied to 0 files). Same EasyNegative.safetensors, resolves on Metal.
TI="embed/EasyNegative#EasyNegative.safetensors"
PROMPT="full-body photo of a woman standing in a sunlit garden, detailed hands, natural pose, sharp focus"
mkdir -p "$ROOT/corpus/images/embedding"

# Baseline — no embedding.
"$PLAKAT" generate "$PROMPT" \
  --model sd15 --negative "lowres, blurry" \
  --steps 28 --size 512x512 --seed 11 --device metal \
  --out "$ROOT/corpus/images/embedding/baseline"

# + EasyNegative — the embedding's trigger goes in the negative prompt.
"$PLAKAT" generate "$PROMPT" \
  --model sd15 --embedding "$TI" \
  --negative "EasyNegative, lowres, blurry" \
  --steps 28 --size 512x512 --seed 11 --device metal \
  --out "$ROOT/corpus/images/embedding/with-easynegative"

echo "✓ wrote corpus/images/embedding/{baseline,with-easynegative}/plakat-11.png"
