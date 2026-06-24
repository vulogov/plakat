#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — multi-person scene placement (`plakat multiperson`)
# ===================================================================
#   ./corpus/multiperson.sh
#   plakat gallery corpus/images --recursive --out corpus/GALLERY.md
#
# Places three SPECIFIC personas (the bundled assets/people/{1,2,3}.png
# portraits) into ONE generated scene, each at a relative location given in
# words — no pixel coordinates. Person 1 sits on the left, person 2 on the
# right, person 3 at the back of the table facing us. M1 renders via the
# portrait pipeline: generate the scene base, then inpaint each persona into
# their region (farther → closer for correct occlusion).
#
# GPU SD output is non-deterministic, so this is a committed SHOWCASE (a gallery
# row), not a byte-check. Build with a GPU backend for speed:
#   cargo build --release --features metal   (Apple Silicon)
#   cargo build --release --features cuda     (NVIDIA)
# SDXL + 3 inpaint passes fits 24 GB; keep --size modest.
#
# Override with env vars: PLAKAT, MODEL, IDENTITY, SIZE, STEPS, SEED.
# ===================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
PEOPLE="$ROOT/assets/people"
OUT="$ROOT/corpus/images/multiperson"
MODEL="${MODEL:-sdxl}"
IDENTITY="${IDENTITY:-plus-face-sdxl}"
SIZE="${SIZE:-1024x768}"
STEPS="${STEPS:-30}"
SEED="${SEED:-42}"
mkdir -p "$OUT"

# --composite is the EXACT-identity, model-agnostic path: generate the scene
# BACKGROUND with any text-to-image model, matte each persona's actual photo
# (U2Net — no face model), and place them at their `--at` positions. Identity is
# the real photo, so it's exact and works on ANY model. Best with a photo (or any
# portrait) on a plain/light background. The scene prompt describes the SETTING —
# the people come from the photos. Add `--harmonize 0.35` to blend them in.
"$PLAKAT" multiperson \
  "a cozy living room interior, warm afternoon light, fireplace and bookshelves // detailed digital painting" \
  --person "p1:$PEOPLE/1.png" --at "p1:left closer front" \
  --person "p2:$PEOPLE/2.png" --at "p2:center closer front" \
  --person "p3:$PEOPLE/3.png" --at "p3:right closer front" \
  --model "$MODEL" --composite \
  --size "$SIZE" --steps "$STEPS" --seed "$SEED" \
  --out "$OUT"

echo "✓ multiperson --composite: 3 real people placed into a scene → corpus/images/multiperson/"
echo "  exact identity (the actual photos), works on ANY model. Add --harmonize 0.35 to blend."
echo "  secondary: --swap (face-swap, frontal sizable faces only); --dry-run for placement plan"
