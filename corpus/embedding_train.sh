#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — Textual Inversion training (embedding train)
# ===================================================================
# Learn a new "word" (one token embedding) from a few images, the whole model
# FROZEN. Self-contained: GENERATES a synthetic STYLE set (stained-glass renders
# of VARIED subjects, so the token learns the STYLE not a subject), trains the
# TI embedding, then renders NEW subjects in that learned style via the token.
#
#   Usage:  ./embedding_train.sh [sd15|sd21|sdxl]    STEPS=1000 default
#
# FAST — TI optimizes one vector for sd15/sd21 (single CLIP-L), a CLIP-L+CLIP-G
# pair for sdxl, the model frozen. A full run is minutes, not hours. Output loads
# with `--embedding PATH:trigger`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
BASE="${1:-sd15}"
STEPS="${STEPS:-1000}"
WORK="${WORK:-/tmp/plakat-ti}"
EMB="$WORK/glass-$BASE.safetensors"
OUT="$ROOT/corpus/images/embedding-train/$BASE"
TRIG="sgwin"
case "$BASE" in
  # Native resolutions differ; off-native rendering degrades (esp. SD 2.1), but
  # native is memory-heavy — 768²/1024² OOM a 24 GB Mac with apps open. Override
  # to trade quality for memory: `SIZE=640x640 ./embedding_train.sh sdxl`.
  sd15) SIZE="${SIZE:-512x512}"; LR="${LR:-5e-3}"; EMB_SCALE="${EMB_SCALE:-1.0}" ;;   # single CLIP-L (768d)
  sd21) SIZE="${SIZE:-768x768}"; LR="${LR:-5e-3}"; EMB_SCALE="${EMB_SCALE:-1.0}" ;;   # single CLIP-L (1024d)
  # SDXL's dual-encoder TI (incl. the 1280d CLIP-G vector + pooled path) over-cooks
  # to a featureless blob at the sd15 single-vector rate (5e-3) — diffusers' SDXL
  # textual inversion uses 5e-4 (10× lower). 1000 steps @ 5e-4 lands in the good window.
  # At full token strength (1.0) the learned "stained-glass-window" motif tiles the
  # subject into panels on SDXL's strong prior; 0.6 renders one clean subject.
  sdxl) SIZE="${SIZE:-768x768}"; LR="${LR:-5e-4}"; EMB_SCALE="${EMB_SCALE:-0.6}" ;;   # dual CLIP-L+CLIP-G; native 1024² (heavy), 768² safer
  # SD 3.5 TI learns a TRIPLE vector (CLIP-L 768 + CLIP-G 1280 + T5 4096) — the
  # T5 half makes both training (autograd through T5-XXL) and the render
  # MEMORY-BOUND: needs >24 GB unified or CUDA. Recipe only; not verified on a
  # 24 GB Mac. LR mirrors SDXL's dual-encoder rate (the single-vector 5e-3
  # over-cooks the multi-encoder case); tune EMB_SCALE down if the motif tiles.
  sd35) SIZE="${SIZE:-512x512}"; LR="${LR:-5e-4}"; EMB_SCALE="${EMB_SCALE:-0.6}" ;;   # triple CLIP-L+CLIP-G+T5
  *) echo "base must be sd15 | sd21 | sdxl | sd35"; exit 1 ;;
esac
# Concept (training-exemplar) generation can use a LIGHTER base than the TI base:
# the stained-glass set is plain training data (downscaled to 256² to train), so
# rendering it with SDXL just wastes memory — and SDXL generate OOMs a 24 GB Mac.
# Default SDXL's exemplars to sd15/512²; sd15/sd21 keep their own base. Override
# with CONCEPT_BASE / CONCEPT_SIZE. This isolates the *new* SDXL-TI training code
# from the (pre-existing, memory-heavy) SDXL inference path.
case "$BASE" in
  # sdxl + sd35 generate their exemplars with a LIGHT base (sd15/512²) — the
  # set is plain training data, and SDXL/SD3.5 inference OOMs a 24 GB Mac.
  sdxl|sd35) CONCEPT_BASE="${CONCEPT_BASE:-sd15}"; CONCEPT_SIZE="${CONCEPT_SIZE:-512x512}" ;;
  *)         CONCEPT_BASE="${CONCEPT_BASE:-$BASE}"; CONCEPT_SIZE="${CONCEPT_SIZE:-$SIZE}" ;;
esac
CONCEPT="$WORK/concept-$CONCEPT_BASE"
mkdir -p "$CONCEPT" "$OUT"

# 1) STYLE set — stained glass of VARIED subjects, so the token learns the STYLE.
#    Generated with CONCEPT_BASE (light), reused across runs of the same base.
SUBJ=("an owl" "a rose" "a sailing ship" "a leaping fox" "a mountain")
for i in "${!SUBJ[@]}"; do
  out="$CONCEPT/plakat-$((i+1)).png"
  [ -f "$out" ] && continue   # reuse already-generated exemplars
  "$PLAKAT" generate \
    "an ornate stained glass window of ${SUBJ[$i]}, vivid jewel colours, bold black leading, backlit" \
    --model "$CONCEPT_BASE" --seed "$((i+1))" --size "$CONCEPT_SIZE" --steps 28 --out "$CONCEPT"
done

# 2) Train the Textual Inversion embedding (one vector; a clip_l+clip_g pair for
#    sdxl — saved as a dual-encoder TI in a single file). Model frozen.
"$PLAKAT" embedding train --base "$BASE" --from-dir "$CONCEPT" \
  --token "$TRIG" --init-word "glass" --steps "$STEPS" --lr "$LR" --size 256 --out "$EMB"

# 3) Render NEW subjects "in the learned style" via the token (per-base subdir so
#    bases don't clobber each other; descriptive names).
"$PLAKAT" generate "a $TRIG cat"             --model "$BASE" --embedding "$EMB:$TRIG:$EMB_SCALE" --seed 42 --size "$SIZE" --steps 28 --out "$OUT"
"$PLAKAT" generate "a $TRIG steam locomotive" --model "$BASE" --embedding "$EMB:$TRIG:$EMB_SCALE" --seed 7  --size "$SIZE" --steps 28 --out "$OUT"
mv -f "$OUT/plakat-42.png" "$OUT/cat.png";        mv -f "$OUT/plakat-42.json" "$OUT/cat.json"        2>/dev/null || true
mv -f "$OUT/plakat-7.png"  "$OUT/locomotive.png"; mv -f "$OUT/plakat-7.json"  "$OUT/locomotive.json" 2>/dev/null || true

echo "✓ TI ($BASE): learned '$TRIG' (stained-glass style) → cat + locomotive in corpus/images/embedding-train/$BASE/"
