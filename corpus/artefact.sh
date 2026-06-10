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

# --skip-artefacts-gen: reuse the already-generated cutouts in $LIB and only
# re-run the composite (fast iteration on scaling/blend without the GPU re-gen).
SKIP_GEN=0
for a in "$@"; do [ "$a" = "--skip-artefacts-gen" ] && SKIP_GEN=1; done

# make_artefact <prompt> <out.png> <seed>
# Generates the subject as an anime-style object, then the U2Net smart matte
# (`--matte`) lifts it out AND crops to the subject bbox. The artefact feature
# reads most naturally on stylized / anime scenes (the blend re-paint integrates
# there); on photoreal it's a grounded composite, not VFX relighting. No chroma.
make_artefact() {
  local prompt="$1" out="$2" seed="$3"
  "$PLAKAT" generate "$prompt, anime illustration, cel-shaded, clean lineart, vibrant colors, centered, isolated, single object, on a plain solid white background, nothing else in frame, sharp focus" \
    --model sdxl --negative "photorealistic, photograph, realistic, multiple, busy background, clutter, scenery, sky, clouds, landscape, ground" \
    --steps 30 --size 1024x1024 --seed "$seed" --device metal \
    --out "$WORK/raw-$seed"
  "$PLAKAT" transparent --in "$WORK/raw-$seed/plakat-$seed.png" --out "$out" --matte --crop --device cpu
}

# Three real cutouts — one per zone, each matted off its background.
if [ "$SKIP_GEN" = 0 ]; then
  make_artefact "a red and yellow striped hot-air balloon"          "$LIB/sky/balloon.png"    7
  make_artefact "a single tall green pine tree, full height"        "$LIB/trees/pine.png"     8
  make_artefact "a small grey stone cottage with a dark slate roof" "$LIB/houses/cottage.png" 9
else
  echo "↷ --skip-artefacts-gen: reusing existing cutouts in $LIB"
fi

# Composite ALL THREE into one valley scene + blend them in (multi-artefact).
"$PLAKAT" generate "anime landscape illustration, a wide alpine valley, grassy meadow, distant mountains, clear blue sky, golden hour, cel-shaded, vibrant colors, detailed scenery, Makoto Shinkai style" \
  --model sdxl --negative "photorealistic, photograph, realistic, 3d render, blurry" \
  --artefact-library "$LIB" \
  --artefact "balloon@sky" --artefact "pine@middle_plan" --artefact "cottage@close_plan" \
  --artefact-blend --artefact-blend-strength 0.55 \
  --steps 30 --size 1024x1024 --seed 11 --device metal \
  --out "$OUT/cli-multi"

echo "✓ built corpus/assets/artefact_library/ (balloon, pine, cottage)"
echo "✓ wrote corpus/images/artefact/cli-multi/ — 3 real artefacts composited + blended"
