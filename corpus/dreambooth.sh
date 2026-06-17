#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — DreamBooth subject LoRA (class prior-preservation)
# ===================================================================
# Learns a SUBJECT (not a style) and renders it in new scenes. Self-contained:
# it GENERATES a synthetic subject set + a class set with plakat, trains a
# subject LoRA (with prior preservation), then renders the subject via its rare
# token. With a real subject you'd use real photos of the ONE thing; this
# synthetic version demonstrates the mechanism end-to-end.
#
#   Usage:  ./dreambooth.sh            (sd15; STEPS=300 default)
#           STEPS=600 ./dreambooth.sh  (longer = binds harder)
#
# ⚠️ SLOW. DreamBooth doubles the per-step cost (subject + class), and sd15
# training on Metal is ~tens of seconds/step — a meaningful run is ~1–2 HOURS,
# not a quick demo. Trains at 256² to keep it tractable; the OOM guard protects.
# ⬜ until rendered.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
STEPS="${STEPS:-300}"
WORK="${WORK:-/tmp/plakat-dreambooth}"
SUBJ="$WORK/subject"; CLASS="$WORK/class"; LORA="$WORK/fox.safetensors"
OUT="$ROOT/corpus/images/dreambooth"
mkdir -p "$SUBJ" "$CLASS" "$OUT"

SUBJ_DESC="a round orange fox plush toy with a big bushy tail and a white belly, studio photo, plain white background, centered"
CLASS_DESC="a plush animal toy, studio photo, plain white background"
TRIG="a sks fox plush toy"            # the rare token learned during training
CLASS_PROMPT="a plush animal toy"     # keeps the broad class general (prior preservation)

# 1) SUBJECT set — 4 views of the distinctive plush (the "subject").
for s in 1 2 3 4; do
  "$PLAKAT" generate "$SUBJ_DESC" --model sd15 --seed "$s" --size 512x512 --steps 28 --out "$SUBJ"
done
# 2) CLASS set — 6 generic plush toys (so the token doesn't overrun the class).
for s in 11 12 13 14 15 16; do
  "$PLAKAT" generate "$CLASS_DESC" --model sd15 --seed "$s" --size 512x512 --steps 28 --out "$CLASS"
done
# 3) Train the SUBJECT LoRA — binds `sks` to THIS plush; class regularizes "plush toy".
"$PLAKAT" style train --base sd15 --from-dir "$SUBJ" \
  --trigger "$TRIG" \
  --class-dir "$CLASS" --class-prompt "$CLASS_PROMPT" \
  --prior-weight 1.0 --steps "$STEPS" --rank 16 --lr 1e-4 --size 256 \
  --out "$LORA"
# 4) Render the subject in NEW scenes — the bushy-tailed orange fox plush should
#    reappear (proving the token binds the learned subject, not a random plush).
"$PLAKAT" generate "$TRIG sitting on a snowy mountain peak at sunset, dramatic lighting" \
  --model sd15 --lora "$LORA" --seed 42 --size 512x512 --steps 28 --out "$OUT"
"$PLAKAT" generate "$TRIG riding a skateboard down a neon city street at night" \
  --model sd15 --lora "$LORA" --seed 7 --size 512x512 --steps 28 --out "$OUT"

echo "✓ DreamBooth: learned '$TRIG' (subject + prior preservation) → 2 renders in corpus/images/dreambooth/"
