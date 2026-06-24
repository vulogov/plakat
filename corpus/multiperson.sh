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

# Three people in a FRONTAL group portrait — 1 left, 2 right, 3 centre.
# `--swap` is the identity path: generate ONE coherent scene, then face-swap each
# detected face with the persona matched to its placement region (SCRFD + ArcFace
# + inswapper, all auto-downloaded). `--at` regions disambiguate which face is who.
#
# IMPORTANT for face-swap: it needs roughly FRONTAL, reasonably-LARGE faces. A
# scene where people look down / away (e.g. studying a chess board) or have tiny
# faces will swap weakly — the alignment + inswapper assume near-frontal faces.
# This prompt keeps everyone facing the camera in a tight group shot.
"$PLAKAT" multiperson \
  "three people posing for a group portrait, facing the camera, head and shoulders, warm indoor lighting // detailed digital painting" \
  --person "p1:$PEOPLE/1.png" --at "p1:left closer front" \
  --person "p2:$PEOPLE/2.png" --at "p2:right closer front" \
  --person "p3:$PEOPLE/3.png" --at "p3:center closer front" \
  --model "$MODEL" --swap \
  --size "$SIZE" --steps "$STEPS" --guidance 7.0 --seed "$SEED" \
  --out "$OUT"

echo "✓ multiperson --swap: 3 personas face-swapped into a frontal group portrait → corpus/images/multiperson/"
echo "  face-swap needs frontal, sizable faces — looking-down / tiny-face scenes swap weakly"
echo "  inspect placement only:  add --dry-run to the command above"
