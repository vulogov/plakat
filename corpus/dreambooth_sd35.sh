#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — SD3.5 DreamBooth subject LoRA (prior preservation)
# ===================================================================
# Learns a SUBJECT (not a style) into an SD3.5 MMDiT LoRA and renders it in new
# scenes. Prior preservation: a generic CLASS set is trained alongside the
# subject (rectified-flow class loss, weight λ) so the rare token binds THIS
# subject without collapsing the broader class.
#
# Self-contained: GENERATES the subject + class sets, trains the sd35 subject
# LoRA, then renders the subject via its rare token. With a real subject you'd
# use real photos of the ONE thing; the synthetic set demonstrates the mechanism.
#
#   Usage:  ./dreambooth_sd35.sh            (STEPS=120 default ≈ 3.5 h)
#           STEPS=80  ./dreambooth_sd35.sh  (quicker, looser bind ≈ 2.3 h)
#           STEPS=250 ./dreambooth_sd35.sh  (binds harder ≈ 7 h)
#           SIZE=768x768 ./dreambooth_sd35.sh  (lighter renders if 1024² OOMs)
#
# ⚠️ SLOW. SD3.5 MMDiT training is ~100 s/step on Metal (DreamBooth runs a subject
# + a class forward each step — ~2× a plain step), so even 120 steps is hours. This
# is the model's inherent cost on this hardware, not a bug. Training runs at 256².
# The subject/class sets are generated with a LIGHT base (sd15) — plain training
# data — so SD3.5 is paid only for training + the final renders, which are cached
# across restarts. Native render res is 1024² (heavy); drop to 768² if the OOM
# guard trips. ⬜ until rendered.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
STEPS="${STEPS:-120}"
SIZE="${SIZE:-1024x1024}"            # sd35 native; SIZE=768x768 is lighter
WORK="${WORK:-/tmp/plakat-dreambooth-sd35}"
# Subject/class sets are base-independent training data → generate them light.
CONCEPT_BASE="${CONCEPT_BASE:-sd15}"; CONCEPT_SIZE="${CONCEPT_SIZE:-512x512}"
SUBJ="$WORK/subject"; CLASS="$WORK/class"; LORA="$WORK/dragon-sd35.safetensors"
OUT="$ROOT/corpus/images/dreambooth-sd35"
mkdir -p "$SUBJ" "$CLASS" "$OUT"

SUBJ_DESC="a round teal-blue dragon plush toy with tiny orange felt wings and a cream belly, studio photo, plain white background, centered"
CLASS_DESC="a plush animal toy, studio photo, plain white background"
TRIG="a sks dragon plush toy"         # the rare token bound during training
CLASS_PROMPT="a plush animal toy"     # keeps the broad class general (prior preservation)

# 1) SUBJECT set — 4 views of the distinctive teal dragon plush (the "subject").
for s in 1 2 3 4; do
  out="$SUBJ/plakat-$s.png"; [ -f "$out" ] && continue
  "$PLAKAT" generate "$SUBJ_DESC" --model "$CONCEPT_BASE" --seed "$s" --size "$CONCEPT_SIZE" --steps 28 --out "$SUBJ"
done
# 2) CLASS set — 6 generic plush toys (so the token doesn't overrun the class).
for s in 11 12 13 14 15 16; do
  out="$CLASS/plakat-$s.png"; [ -f "$out" ] && continue
  "$PLAKAT" generate "$CLASS_DESC" --model "$CONCEPT_BASE" --seed "$s" --size "$CONCEPT_SIZE" --steps 28 --out "$CLASS"
done
# 3) Train the SD3.5 SUBJECT LoRA — binds `sks` to THIS dragon; the class loss
#    regularizes "plush toy" so the token doesn't drag the whole class with it.
"$PLAKAT" style train --base sd35-medium --from-dir "$SUBJ" \
  --trigger "$TRIG" \
  --class-dir "$CLASS" --class-prompt "$CLASS_PROMPT" \
  --prior-weight 1.0 --steps "$STEPS" --rank 16 --lr 1e-4 --size 256 \
  --out "$LORA"
# 4) Render the subject in NEW scenes — the teal-with-orange-wings dragon plush
#    should reappear (proving the token binds the learned subject, not a random plush).
"$PLAKAT" generate "$TRIG sitting on a mossy log in a sunlit forest, shallow depth of field" \
  --model sd35-medium --lora "$LORA" --seed 42 --size "$SIZE" --steps 28 --out "$OUT"
mv -f "$OUT/plakat-42.png" "$OUT/forest.png";  mv -f "$OUT/plakat-42.json" "$OUT/forest.json"  2>/dev/null || true
"$PLAKAT" generate "$TRIG on a beach towel under a striped umbrella, bright daylight" \
  --model sd35-medium --lora "$LORA" --seed 7 --size "$SIZE" --steps 28 --out "$OUT"
mv -f "$OUT/plakat-7.png"  "$OUT/beach.png";   mv -f "$OUT/plakat-7.json"  "$OUT/beach.json"   2>/dev/null || true

echo "✓ SD3.5 DreamBooth: learned '$TRIG' (subject + prior preservation) → forest + beach in corpus/images/dreambooth-sd35/"
