#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — Civitai integration (--lora civitai:<id>)
# ===================================================================
# Pulls "Eldritch Watercolor" (Civitai model 536722, SDXL 1.0, ~183 MB)
# directly by id: plakat resolves the model's latest version, downloads +
# caches the .safetensors, and applies it — end-to-end Civitai → LoRA →
# image. This LoRA has no trigger words; the style applies from the adapter.
#
# REQUIRES a Civitai API key — downloads are auth-gated (HTTP 401 without
# one). Create one at civitai.com → Account settings → API Keys, then:
#   export CIVITAI_API_KEY=...
#
# LoRA spec grammar (src/pipelines/lora.rs):
#   civitai:<model-id>[:scale]          → latest version of the model
#   civitai-version:<version-id>[:scale]→ a pinned version
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

if [[ -z "${CIVITAI_API_KEY:-}" ]]; then
  echo "⚠  CIVITAI_API_KEY is not set — the Civitai download will 401. Export it first." >&2
fi

"$PLAKAT" generate "an ancient cathedral shrouded in mist, towering spires, eldritch atmosphere, intricate detail" \
  --model sdxl --lora "civitai:536722:0.9" \
  --steps 30 --size 1024x1024 --seed 42 --device metal \
  --out "$ROOT/corpus/images/civitai"

echo "✓ wrote corpus/images/civitai/  (Eldritch Watercolor LoRA, Civitai model 536722)"
