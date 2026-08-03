#!/usr/bin/env bash
#
# BOOKART-1 feature-demonstration driver (RFC BOOKART-1, plakat 6.0.0).
#
# Renders the corpus bookart specs across the hybrid router — procedural (zero weights), diffusion
# (origin LoRA auto-resolved from vulogov98/plakat-bookart), composite — plus the flagship kit and
# manuscript, into corpus/images/bookart/. Every ornament is a transparent, exact-page-sized PNG.
#
# Usage:   corpus/bookart_run.sh                    # everything, release binary
#          PLAKAT=./target/debug/plakat STEPS=16 corpus/bookart_run.sh
#
# Build the RELEASE binary first (debug inference is ~50x slower):  cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
STEPS="${STEPS:-24}"
OUT="corpus/images/bookart"
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== BOOKART-1 corpus driver ===================="
echo "binary : $PLAKAT"
echo "out    : $OUT/"
echo

# 1. Weights-free procedural ornament + born-vector SVG.
run "$PLAKAT" bookart lint corpus/bookart_ornament.hjson
run "$PLAKAT" bookart show corpus/bookart_ornament.hjson
run "$PLAKAT" bookart render corpus/bookart_ornament.hjson --out "$OUT/border.png"   # spec asks for png+svg

# 2. A diffusion plate (the russian origin LoRA downloads from HF on first use).
run "$PLAKAT" bookart illustrate "a firebird among oak branches" --origin russian --type vignette \
    --steps "$STEPS" --out "$OUT/plate.png"

# 3. A composite ornament (procedural frame + diffusion inlay).
run "$PLAKAT" bookart render corpus/bookart_kit.hjson --out "$OUT/composite.png" --steps "$STEPS" 2>/dev/null || true

# 4. The flagship: a coherent kit (contact sheet + coherence + manifest).
run "$PLAKAT" bookart kit corpus/bookart_kit.hjson --out "$OUT/kit" --steps "$STEPS"

# 5. The flagship: a whole book's per-chapter ornaments + LaTeX includes.
run "$PLAKAT" bookart manuscript corpus/bookart_book.md --kit corpus/bookart_kit.hjson \
    --out "$OUT/manuscript" --latex --steps "$STEPS"

echo "==================== done → $OUT/ ===================="
