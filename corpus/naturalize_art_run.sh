#!/usr/bin/env bash
#
# QUALITY driver — naturalizing AI-generated ART (RFC QUALITY-1/2, plakat 6.10–6.12).
#
# Input: assets/citystreet.png — a watercolour city-street scene that was clearly flagged as "AI slop":
# oversaturated, muddy contrast, uniform AI-painterly texture, incoherent tram-wire geometry. The point of
# `naturalize` here is NOT to pretend a human painted it — it's to make the computer output LESS SLOPPY: a
# cleaner, better-looking picture. The de-slop core is weight-free:
#   - `--polish`  gray-world white balance + robust auto-levels + vibrance + unsharp  → better colour+detail,
#   - content focuses (cityscape / vegetation / mechanics / people / sky) pre-tune the pass to this scene,
#   - `--micro`   adds fine texture only where the render is unnaturally smooth,
#   - `--designature` dissolves a foreign corner smudge.
# The GEOMETRY of the tram wires / architecture is structural — a model refine (`--geometry`), included
# under RENDER=1.
#
# Usage:   corpus/naturalize_art_run.sh                 # weight-free de-slop (no GPU)
#          RENDER=1 corpus/naturalize_art_run.sh        # + the model-backed geometry fix
#          PLAKAT=./target/debug/plakat corpus/naturalize_art_run.sh
#
# Build the RELEASE binary first for the model-backed part: cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
SRC="${SRC:-assets/citystreet.png}"
OUT="corpus/images/quality_art"
DEVICE="${DEVICE:-auto}"
RENDER="${RENDER:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== QUALITY art driver (naturalize AI slop) ===================="
echo "binary : $PLAKAT"
echo "src    : $SRC   (AI-slop watercolour cityscape)"
echo "out    : $OUT/    (RENDER=$RENDER)"
echo

# ---- 0. Baseline AI-tell of the raw AI-slop input (lower = cleaner / less-AI). ----
run "$PLAKAT" rank "$SRC" --ai-tells

# ---- 1. Quality improvement in isolation — --polish only (the pure de-slop: colour + contrast + detail). ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/polish_only.png" \
  --polish 0.85 --grain 0 --aberration 0 --vignette 0 --bloom 0 --desaturate 0 --defocus 0 --micro 0

# ---- 2. Presets on the art (weight-free): subtle | photo | painting. Each prints its AI-tell. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/preset_subtle.png"   --preset subtle
run "$PLAKAT" naturalize "$SRC" --out "$OUT/preset_photo.png"    --preset photo
run "$PLAKAT" naturalize "$SRC" --out "$OUT/preset_painting.png" --preset painting

# ---- 3. Scene-tuned de-slop — this street has ALL of: cityscape + tree + tram + crowd + sky. Focuses combine. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/focus_scene.png" \
  --cityscape 1.2 --vegetation 0.7 --mechanics 0.7 --people 0.5 --sky 0.5

# ---- 4. Strong, clearly-visible de-slop (reads the delta): tame oversaturation + crisp + gentle grade. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/deslop_strong.png" \
  --polish 1 --desaturate 0.3 --grain 0.15 --micro 0.4 --cityscape 1

# ---- 5. Ghost-signature removal (weight-free): dissolve a foreign corner smudge. ----
run "$PLAKAT" naturalize "$SRC" --out "$OUT/designature_br.png" --designature br --preset photo

# ---- 6. AI-tell ranking (weight-free): least-AI-looking first over everything produced above. ----
run "$PLAKAT" rank "$OUT" --ai-tells

# ---- 7. Model-backed geometry fix (RENDER=1) — the sloppy tram-wire / architecture structure. ----
if [ "$RENDER" = "1" ]; then
  run "$PLAKAT" naturalize "$SRC" --out "$OUT/corrective_geometry.png" \
    --geometry 1 --cityscape 1 --polish 0.7 --device "$DEVICE"
else
  echo "· RENDER=0 — skipping the model-backed geometry fix (--geometry, fixes the incoherent tram-wire structure)."
  echo "  run 'RENDER=1 corpus/naturalize_art_run.sh' with a release build to include it."
fi

echo "==================== done → $OUT/ ===================="
