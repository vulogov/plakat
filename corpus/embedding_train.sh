#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — Textual Inversion training (embedding train)
# ===================================================================
# Learn a new "word" (one token embedding) from a few images, the whole model
# FROZEN. Self-contained: GENERATES a synthetic STYLE set (stained-glass renders
# of VARIED subjects, so the token learns the STYLE not a subject), trains the
# TI embedding, then renders NEW subjects in that learned style via the token.
#
#   Usage:  ./embedding_train.sh [sd15|sd21]    STEPS=1000 default
#
# FAST — TI optimizes a single vector (~0.2 s/step at 256²), so a full run is a
# few minutes, not hours. Output loads with `--embedding PATH:trigger`. sd15/sd21
# (single CLIP-L); SDXL is a follow-up.
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
  sd15) SIZE=512x512 ;;
  sd21) SIZE=768x768 ;;   # SD 2.1 is a 768² model — rendering it at 512² degrades badly
  *) echo "base must be sd15 | sd21 (single CLIP-L)"; exit 1 ;;
esac
mkdir -p "$CONCEPT" "$OUT"

# 1) STYLE set — stained glass of VARIED subjects, so the token learns the STYLE.
SUBJ=("an owl" "a rose" "a sailing ship" "a leaping fox" "a mountain")
for i in "${!SUBJ[@]}"; do
  "$PLAKAT" generate \
    "an ornate stained glass window of ${SUBJ[$i]}, vivid jewel colours, bold black leading, backlit" \
    --model "$BASE" --seed "$((i+1))" --size "$SIZE" --steps 28 --out "$CONCEPT"
done

# 2) Train the Textual Inversion embedding (one vector, model frozen).
"$PLAKAT" embedding train --base "$BASE" --from-dir "$CONCEPT" \
  --token "$TRIG" --init-word "glass" --steps "$STEPS" --size 256 --out "$EMB"

# 3) Render NEW subjects "in the learned style" via the token (per-base subdir so
#    bases don't clobber each other; descriptive names).
"$PLAKAT" generate "a $TRIG cat"             --model "$BASE" --embedding "$EMB:$TRIG" --seed 42 --size "$SIZE" --steps 28 --out "$OUT"
"$PLAKAT" generate "a $TRIG steam locomotive" --model "$BASE" --embedding "$EMB:$TRIG" --seed 7  --size "$SIZE" --steps 28 --out "$OUT"
mv -f "$OUT/plakat-42.png" "$OUT/cat.png";        mv -f "$OUT/plakat-42.json" "$OUT/cat.json"        2>/dev/null || true
mv -f "$OUT/plakat-7.png"  "$OUT/locomotive.png"; mv -f "$OUT/plakat-7.json"  "$OUT/locomotive.json" 2>/dev/null || true

echo "✓ TI ($BASE): learned '$TRIG' (stained-glass style) → cat + locomotive in corpus/images/embedding-train/$BASE/"
