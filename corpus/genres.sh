#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — subject-domain genres (--genre)
# ===================================================================
# The bundled `anime` genre — the independent axis from --look (genres set
# the subject DOMAIN, looks set the MEDIUM; they compose). Only `anime`
# ships bundled; photoreal / fantasy / cyberpunk / etc. live in the
# user-extension genres directory. We run `--offline` so the genre uses its
# prompt preset deterministically (its optional LoRA is auto-discovered from
# Civitai — network, flaky); `--scheduler euler-a` for Metal. Drop
# `--offline` to let the LoRA discovery run. Ungated SD 1.5, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" generate "a lone knight resting under a blossoming cherry tree" \
  --model sd15 --genre anime --offline \
  --scheduler euler-a \
  --steps 28 --size 512x512 --seed 42 --device metal \
  --out "$ROOT/corpus/images/genres/anime"

echo "✓ wrote corpus/images/genres/anime/  (compose with --look for medium × domain)"
