#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — subject-domain genres (--genre)
# ===================================================================
# The bundled `anime` genre — the independent axis from --look (genres set
# the subject DOMAIN, looks set the MEDIUM; they compose). Without --offline
# the genre auto-discovers + downloads its anime LoRA from Civitai (needs
# CIVITAI_API_KEY; reliable now the download timeout bug is fixed).
#
# To pin a SPECIFIC hand-picked anime LoRA instead of the auto-discovery,
# add it explicitly — e.g. add `--lora civitai:<model-id>:0.9` (use a LoRA
# whose base is one plakat handles: sdxl / pony / sd15). `--scheduler
# euler-a` for Metal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" generate "a lone knight resting under a blossoming cherry tree" \
  --model sd15 --genre anime \
  --scheduler euler-a \
  --steps 28 --size 512x512 --seed 42 --device metal \
  --out "$ROOT/corpus/images/genres/anime"

echo "✓ wrote corpus/images/genres/anime/  (compose with --look for medium × domain)"
