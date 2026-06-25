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

# --swap --pose is the identity path that puts the REAL people INTO the scene:
# generate a coherent scene with one OpenPose skeleton pinned per persona region
# (so figures land where each person goes), then face-swap each figure from the
# photo. Lessons that make it work (verified):
#   * use PHOTOS (photoreal, frontal, light background) — not paintings; close-up
#     crops are auto-padded so SCRFD detects them.
#   * keep the prompt MINIMAL ("people …") and DON'T describe each person — the
#     swap defines the faces. Describing "old man with a beard" bleeds that look
#     onto every figure.
#   * keep figures PROMINENT (fewer / closer = larger faces = stronger identity).
# TWO people, framed PROMINENTLY (upper body) so each face is large enough to read
# as a specific person. Identity strength scales with face size: a crowded scene
# (3+ people, tiny faces) swaps faintly; two prominent figures are recognizable.
"$PLAKAT" multiperson \
  "an old man on the left and a woman on the right, sitting close together at a small cafe table by a window, soft daylight, watercolor painting, upper body portrait, both facing the viewer" \
  --person "p1:$PEOPLE/1.png" --at "p1:left closer front" \
  --person "p2:$PEOPLE/2.png" --at "p2:right closer front" \
  --model "$MODEL" --swap --pose \
  --size "$SIZE" --steps "$STEPS" --guidance 7.0 --seed "$SEED" \
  --out "$OUT"

echo "✓ multiperson --swap --pose: 2 prominent people swapped into a watercolor scene → corpus/images/multiperson/"
echo "  identity scales with face size — keep figures few + prominent; use PHOTOS on a light bg; minimal prompt."
echo "  for >2 people the faces get small and read faintly. alt identity: --composite (exact-photo cut-outs, any model)."
