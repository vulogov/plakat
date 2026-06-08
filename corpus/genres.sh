#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — subject-domain genres (--genre)
# ===================================================================
# The anime subject-domain via a pinned Civitai LoRA: "Anime Screencap Style"
# (civitai.com/models/4982, SD 1.5, trigger "anime screencap"). We pin the
# specific VERSION, not the model: `civitai:4982` resolves to the latest
# (v3-offset = a noise-offset variant that barely stylizes), so we pin the
# classic style version v2-35epochs (id 5744) via `civitai-version:5744`.
#
# REQUIRES CIVITAI_API_KEY for the first download (auth-gated; reliable now
# the download timeout bug is fixed); cached after. Pinned explicitly rather
# than via --genre auto-discovery so it's deterministic. `--scheduler
# euler-a` for Metal. Compose with `--look <medium>` for medium × domain.
#
# This is an old, gently-trained LoRA with a narrow usable band: at scale 1.0
# with the trigger trailing it stays photoreal; leading the prompt with the
# "anime screencap" trigger at scale 1.5 gives a clean cel-shaded screencap;
# >2.0 fries it. SD 1.5 + this LoRA mangles human figures, so we render an
# anime *background/scenery* (which screencaps do beautifully) and negative
# out people.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" generate "anime screencap, a tranquil Japanese shrine among blossoming cherry trees, koi pond, stone lanterns, vibrant flat colors, detailed anime background art" \
  --model sd15 --lora "civitai-version:5744:1.5" \
  --negative "people, person, figure, deformed, ugly, blurry" \
  --scheduler euler-a \
  --steps 30 --size 512x512 --seed 11 --device metal \
  --out "$ROOT/corpus/images/genres/anime"

echo "✓ wrote corpus/images/genres/anime/  (Anime Screencap Style LoRA, Civitai model 4982)"
