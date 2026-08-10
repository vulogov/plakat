#!/usr/bin/env bash
#
# COMIC-1 feature-demonstration driver (RFC COMIC-1, plakat 6.8).
#
# A ComicSpec → a lettered, multi-panel comic page: a panel grid, per-panel scene art, a recurring cast,
# and placed speech balloons + captions. The weight-free half (lint / show / layout / letter) needs no
# GPU; the final `comic render` generates the scene art and letters it face-aware. Everything lands under
# corpus/images/comic/.
#
# Usage:   corpus/comic_run.sh                         # everything, release binary
#          PLAKAT=./target/debug/plakat corpus/comic_run.sh
#          RENDER=0 corpus/comic_run.sh                # skip the model step (weight-free only)
#
# Build the RELEASE binary first (debug inference is ~50x slower): cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/comic"
SPEC="corpus/comic_noir.hjson"
RENDER="${RENDER:-1}"
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== COMIC-1 corpus driver ===================="
echo "binary : $PLAKAT"
echo "spec   : $SPEC"
echo "out    : $OUT/"
echo

# 1. Weights-free: validate + show the resolved page/panel plan.
run "$PLAKAT" comic lint "$SPEC"
run "$PLAKAT" comic show "$SPEC"

# 2. Weights-free: the layout skeleton (placeholder panels) + the sidecar.
run "$PLAKAT" comic layout "$SPEC" --out "$OUT/noir_layout.png"

# 3. Weights-free: letter the placeholder page (balloons over the grid — proves placement/lettering
#    without a model).
run "$PLAKAT" comic letter "$SPEC" --out "$OUT/noir_lettered_placeholder.png"

# 4. The full flagship (needs a model): generate each panel's scene art, composite, letter face-aware.
if [ "$RENDER" = "1" ]; then
  run "$PLAKAT" comic render "$SPEC" --out "$OUT/noir.png" --panels-out "$OUT/panels"
else
  echo "· RENDER=0 — skipping the model step (the weight-free page is at $OUT/noir_lettered_placeholder.png)"
fi

echo "==================== done → $OUT/ ===================="
