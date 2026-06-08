#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — subject-domain genres (--genre)
# ===================================================================
# The anime subject-domain via a pinned Civitai LoRA: "Anime Screencap Style"
# (civitai.com/models/4982, SD 1.5, trigger "anime screencap"). plakat pulls
# it by id (`civitai:4982` = latest version), caches it, applies it — a real
# Civitai → LoRA → image demo on a domain LoRA (vs civitai.sh's style LoRA).
#
# REQUIRES CIVITAI_API_KEY (downloads are auth-gated; reliable now the
# download timeout bug is fixed). Pinned explicitly rather than via --genre
# auto-discovery so the result is deterministic. `--scheduler euler-a` for
# Metal. Compose with `--look <medium>` for medium × domain.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" generate "a lone knight resting under a blossoming cherry tree, anime screencap" \
  --model sd15 --lora "civitai:4982:0.9" \
  --scheduler euler-a \
  --steps 28 --size 512x512 --seed 42 --device metal \
  --out "$ROOT/corpus/images/genres/anime"

echo "✓ wrote corpus/images/genres/anime/  (Anime Screencap Style LoRA, Civitai model 4982)"
