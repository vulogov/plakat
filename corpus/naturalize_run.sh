#!/usr/bin/env bash
#
# QUALITY-1 feature-demonstration driver (RFC QUALITY-1, plakat 6.10).
#
# The `naturalize` analog post-pass reduces the "AI-generated" fingerprint of an image — film grain +
# chromatic aberration + vignette + a desaturating film grade, tuned per subject via focus qualifiers.
# All weight-free (no GPU). Outputs land under corpus/images/quality/.
#
# Usage:   corpus/naturalize_run.sh
#          PLAKAT=./target/debug/plakat corpus/naturalize_run.sh
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/quality"
SRC="$OUT/forest_before.png"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== QUALITY-1 corpus driver ===================="
echo "binary : $PLAKAT"
echo "src    : $SRC"
echo

# Weight-free: the three presets + a subject-focused pass.
run "$PLAKAT" naturalize "$SRC" --out "$OUT/forest_subtle.png"  --preset subtle
run "$PLAKAT" naturalize "$SRC" --out "$OUT/forest_photo.png"   --preset photo
run "$PLAKAT" naturalize "$SRC" --out "$OUT/forest_painting.png" --preset painting
# A forest is vegetation + sky — pre-tune to those tells (the committed forest_after.png).
run "$PLAKAT" naturalize "$SRC" --out "$OUT/forest_after.png"   --preset photo --vegetation 1 --sky 1

echo "==================== done → $OUT/ ===================="
