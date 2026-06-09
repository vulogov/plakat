#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — transparent (background knock-out → RGBA)
# ===================================================================
# `plakat transparent` makes every pixel matching the **upper-left corner
# colour** transparent — a quick alpha cut-out for subjects shot/generated on a
# flat background. We generate a subject on a clean solid backdrop (so the
# corners are the background colour), then knock that colour out into an RGBA
# PNG. `--tolerance` widens the match to absorb anti-aliased / JPEG-noisy edges
# (0 = exact; 30–50 = soft edges). Output must be `.png` (or `.webp`) to keep
# the alpha channel. Ungated, Metal-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
mkdir -p "$ROOT/corpus/images/transparent"

# 1. Generate a subject on a clean solid background (corners = background).
"$PLAKAT" generate "a single ripe red apple, centered, on a plain solid white background, studio product photo, soft even lighting, sharp focus" \
  --model sdxl --negative "shadow, gradient, vignette, busy background, clutter, reflection" \
  --steps 30 --size 1024x1024 --seed 7 --device metal \
  --out "$ROOT/corpus/images/transparent/apple-on-white"

# 2. Knock out the white background → RGBA (tolerance absorbs the soft edge).
"$PLAKAT" transparent \
  --in  "$ROOT/corpus/images/transparent/apple-on-white/plakat-7.png" \
  --out "$ROOT/corpus/images/transparent/apple-cutout.png" \
  --tolerance 40

echo "✓ wrote corpus/images/transparent/apple-cutout.png (RGBA — background removed)"
