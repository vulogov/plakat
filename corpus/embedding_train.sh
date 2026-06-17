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
CONCEPT="$WORK/concept-$BASE"; EMB="$WORK/glass-$BASE.safetensors"
OUT="$ROOT/corpus/images/embedding-train/$BASE"
TRIG="sgwin"
case "$BASE" in
  # Native resolutions differ; off-native rendering degrades (esp. SD 2.1), but
  # native is memory-heavy — 768²/1024² OOM a 24 GB Mac with apps open. Override
  # to trade quality for memory: `SIZE=640x640 ./embedding_train.sh sdxl`.
  sd15) SIZE="${SIZE:-512x512}" ;;   # single CLIP-L (768d)
  sd21) SIZE="${SIZE:-768x768}" ;;   # single CLIP-L (1024d)
  sdxl) SIZE="${SIZE:-768x768}" ;;   # dual CLIP-L+CLIP-G; native 1024² (heavy), 768² safer default
  *) echo "base must be sd15 | sd21 | sdxl"; exit 1 ;;
esac
mkdir -p "$CONCEPT" "$OUT"

# 1) STYLE set — stained glass of VARIED subjects, so the token learns the STYLE.
SUBJ=("an owl" "a rose" "a sailing ship" "a leaping fox" "a mountain")
for i in "${!SUBJ[@]}"; do
  "$PLAKAT" generate \
    "an ornate stained glass window of ${SUBJ[$i]}, vivid jewel colours, bold black leading, backlit" \
    --model "$BASE" --seed "$((i+1))" --size "$SIZE" --steps 28 --out "$CONCEPT"
done

# 2) Train the Textual Inversion embedding (one vector; a clip_l+clip_g pair for
#    sdxl — saved as a dual-encoder TI in a single file). Model frozen.
"$PLAKAT" embedding train --base "$BASE" --from-dir "$CONCEPT" \
  --token "$TRIG" --init-word "glass" --steps "$STEPS" --size 256 --out "$EMB"

# 3) Render NEW subjects "in the learned style" via the token (per-base subdir so
#    bases don't clobber each other; descriptive names).
"$PLAKAT" generate "a $TRIG cat"             --model "$BASE" --embedding "$EMB:$TRIG" --seed 42 --size "$SIZE" --steps 28 --out "$OUT"
"$PLAKAT" generate "a $TRIG steam locomotive" --model "$BASE" --embedding "$EMB:$TRIG" --seed 7  --size "$SIZE" --steps 28 --out "$OUT"
mv -f "$OUT/plakat-42.png" "$OUT/cat.png";        mv -f "$OUT/plakat-42.json" "$OUT/cat.json"        2>/dev/null || true
mv -f "$OUT/plakat-7.png"  "$OUT/locomotive.png"; mv -f "$OUT/plakat-7.json"  "$OUT/locomotive.json" 2>/dev/null || true

echo "✓ TI ($BASE): learned '$TRIG' (stained-glass style) → cat + locomotive in corpus/images/embedding-train/$BASE/"
