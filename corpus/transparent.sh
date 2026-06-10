#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — transparent (smart content-aware cut-out → RGBA)
# ===================================================================
# `plakat transparent --matte` predicts the foreground subject with a U2Net
# salient-object model — NO chroma backdrop, works on photoreal / painted
# subjects on ANY background. We render a normal apple scene (apple on a real
# table), then lift the apple cleanly out into an RGBA PNG. (The corner
# flood-fill `--tolerance` path stays for flat studio/chroma backdrops.)
# Output must be `.png` (or `.webp`) to keep alpha. The matte runs on CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
OUT="$ROOT/corpus/images/transparent"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

# 1. A normal photoreal subject on a real surface — no chroma, no flat backdrop.
"$PLAKAT" generate "a single ripe red apple on a rustic wooden table, soft window light, photorealistic, sharp focus" \
  --model sdxl --steps 30 --size 1024x1024 --seed 7 --device metal \
  --out "$WORK/apple-raw"

# 2. Smart matte cut-out → RGBA (content-aware; lifts the apple off the scene).
"$PLAKAT" transparent \
  --in  "$WORK/apple-raw/plakat-7.png" \
  --out "$OUT/apple-cutout.png" \
  --matte --crop --device cpu

echo "✓ wrote corpus/images/transparent/apple-cutout.png (RGBA — smart matte cut-out)"
