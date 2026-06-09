#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — artefact compositing (REAL cutouts, multi-artefact)
# ===================================================================
# The bundled artefact library (sun / cloud / oak / cottage …) is procedurally
# *drawn* — fine for layout, weak as imagery. This builds a **real SDXL**
# library and composites a MULTI-artefact scene, as realistic as the base gets.
#
# Per artefact: generate the subject on a flat chroma background → plakat
# TRANSPARENT knocks the chroma out → an RGBA cutout in
# corpus/assets/artefact_library/. Then ONE generate composites all three into a
# valley with `--artefact-blend` — a masked low-strength img2img pass that
# "inpaints" each cutout so it matches the scene's lighting and softens its seam.
#
# SDXL only: the blend runs through the SD-core img2img pipeline. SD 3.5's MMDiT
# has no blend path (it could alpha-composite but not integrate). The scenario
# counterpart — same library, declarative — is corpus/artefact.hjson.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
LIB="$ROOT/corpus/assets/artefact_library"
OUT="$ROOT/corpus/images/artefact"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$LIB/sky" "$LIB/trees" "$LIB/houses" "$OUT"

# make_artefact <prompt> <chroma colour> <tolerance> <out.png> <seed>
# Generates the subject on a flat chroma background, then knocks the chroma out.
# Chroma is picked per subject so it never collides with the subject's palette.
make_artefact() {
  local prompt="$1" chroma="$2" tol="$3" out="$4" seed="$5"
  "$PLAKAT" generate "$prompt, centered, on a plain solid $chroma background, studio lighting, sharp focus, no shadow, no ground" \
    --model sdxl --negative "shadow, gradient, landscape, multiple, busy background" \
    --steps 30 --size 1024x1024 --seed "$seed" --device metal \
    --out "$WORK/raw-$seed"
  "$PLAKAT" transparent --in "$WORK/raw-$seed/plakat-$seed.png" --out "$out" --tolerance "$tol"
}

# Three real cutouts — one per zone. Chroma avoids each subject's colours
# (green for the warm balloon / grey cottage; magenta for the green pine).
make_artefact "a red and yellow striped hot-air balloon, floating"  "chroma-key green" 45 "$LIB/sky/balloon.png"    7
make_artefact "a single tall green pine tree, full height"          "magenta"          50 "$LIB/trees/pine.png"     8
make_artefact "a small grey stone cottage with a dark slate roof"   "chroma-key green" 45 "$LIB/houses/cottage.png" 9

# Composite ALL THREE into one valley scene + blend them in (multi-artefact).
"$PLAKAT" generate "a wide alpine valley, grassy meadow, distant mountains, clear blue sky, golden hour, photorealistic landscape photograph, sharp focus" \
  --model sdxl --artefact-library "$LIB" \
  --artefact "balloon@sky" --artefact "pine@middle_plan" --artefact "cottage@close_plan" \
  --artefact-blend --artefact-blend-strength 0.3 \
  --steps 30 --size 1024x1024 --seed 11 --device metal \
  --out "$OUT/cli-multi"

echo "✓ built corpus/assets/artefact_library/ (balloon, pine, cottage)"
echo "✓ wrote corpus/images/artefact/cli-multi/ — 3 real artefacts composited + blended"
