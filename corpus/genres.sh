#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — subject-domain genre via a Civitai LoRA (SDXL)
# ===================================================================
# The anime subject-domain via "Animestyle Slider XL v1.1" (civitai.com/
# models/129020, SDXL). plakat pulls it by id (`civitai:129020` = latest
# version), caches it, applies it — a real Civitai → LoRA → image demo.
# It's a SLIDER LoRA (no trigger word); the scale dials anime strength.
#
# Why SDXL, not SD 1.5: anime LoRAs need an anime-capable base. plakat's
# SD 1.5 is the *vanilla* checkpoint (not anime-tuned), so SD 1.5 anime LoRAs
# muddy instead of cel-shading. SDXL is a far stronger base and renders crisp
# anime (same reason civitai.sh's watercolour looked great on SDXL).
#
# REQUIRES CIVITAI_API_KEY for the first download (auth-gated; cached after).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

"$PLAKAT" generate "anime style, a lone samurai standing under a blossoming cherry tree, detailed anime illustration, vibrant flat colors, clean cel shading" \
  --model sdxl --lora "civitai:129020:1.5" \
  --negative "photorealistic, 3d render, deformed, ugly, blurry" \
  --steps 30 --size 1024x1024 --seed 7 --device metal \
  --out "$ROOT/corpus/images/genres/anime"

echo "✓ wrote corpus/images/genres/anime/  (Animestyle Slider XL, Civitai model 129020)"
