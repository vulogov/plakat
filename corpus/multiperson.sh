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

# Three people at a chess table: 1 left, 2 right, 3 at the back facing us.
# `--at` axes: position (left/center/right) · distance (closer/farther) · facing.
"$PLAKAT" multiperson \
  "three people sitting at a table playing chess, warm interior light // detailed digital painting" \
  --person "p1:$PEOPLE/1.png" --at "p1:left closer front" \
  --person "p2:$PEOPLE/2.png" --at "p2:right closer front" \
  --person "p3:$PEOPLE/3.png" --at "p3:center farther front" \
  --person-prompt "p1:studying the board, hand on chin" \
  --person-prompt "p2:reaching to move a piece" \
  --person-prompt "p3:watching the game, smiling" \
  --model "$MODEL" --identity "$IDENTITY" \
  --size "$SIZE" --steps "$STEPS" --guidance 7.0 --seed "$SEED" \
  --out "$OUT"

echo "✓ multiperson: 3 placed personas (1 left, 2 right, 3 back) → corpus/images/multiperson/"
echo "  inspect placement only:  add --dry-run to the command above"
