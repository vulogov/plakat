#!/usr/bin/env bash
#
# v0.42 phase 3 outcome verification — Stage C image encoder
# (image variation + faithful img2img).
#
# Strategy: hold prompt + seed FIXED and toggle ONLY the image
# conditioning, so any difference in the output is attributable to the
# CLIP ViT-L/14 image embedding — nothing else.
#
# Three checks:
#   A. Baseline vs variation (same seed, neutral prompt) — the image
#      embedding must visibly pull the output toward the reference.
#   B. Cross-reference — two DIFFERENT references with the SAME prompt
#      and seed must yield two DIFFERENT outputs, each echoing its own
#      reference (proves the embedding tracks the input, not noise).
#   C. Faithful img2img — `--faithful` must hold the subject closer to
#      the init than plain img2img at the same strength.
#
# Run from the repo root. First invocation downloads the image encoder
# (~1.2 GB) from stabilityai/stable-cascade-prior.

set -euo pipefail

BIN=./target/release/plakat
OUT=/tmp/cascade-verify-phase3
COMMON=(--model stable-cascade --size 1024x1024 \
        --stage-c-steps 20 --stage-b-steps 10 --guidance 4.0)
REF_A=gallery/2.png   # elderly man, rustic portrait
REF_B=gallery/4.png   # steppe at orange sunrise
PROMPT="portrait photograph"
SEED=7

if [[ ! -x "$BIN" ]]; then
  echo "Build first: cargo build --bin plakat --release --features metal" >&2
  exit 1
fi
mkdir -p "$OUT"

echo "== A. baseline (no image conditioning), seed $SEED =="
"$BIN" generate "$PROMPT" "${COMMON[@]}" --seed "$SEED" \
    --out "$OUT/A_baseline"

echo "== A. variation on REF_A ($REF_A), SAME prompt + seed =="
"$BIN" generate "$PROMPT" "${COMMON[@]}" --seed "$SEED" \
    --image-variation "$REF_A" --out "$OUT/A_variation"

echo "== B. variation on REF_B ($REF_B), SAME prompt + seed =="
"$BIN" generate "$PROMPT" "${COMMON[@]}" --seed "$SEED" \
    --image-variation "$REF_B" --out "$OUT/B_variation"

# img2img takes INPUT positionally and uses --steps (it splits Stage C/B
# internally), NOT the generate-only --stage-c-steps/--stage-b-steps.
IMG2IMG=(--model stable-cascade --size 1024x1024 --steps 30 --guidance 4.0)

echo "== C. plain img2img (structure only), strength 0.7 =="
"$BIN" img2img "$REF_A" --prompt "a person" "${IMG2IMG[@]}" \
    --strength 0.7 --seed "$SEED" --out "$OUT/C_plain"

echo "== C. faithful img2img (+ Stage C semantic), strength 0.7 =="
"$BIN" img2img "$REF_A" --prompt "a person" "${IMG2IMG[@]}" \
    --strength 0.7 --seed "$SEED" --faithful --out "$OUT/C_faithful"

echo
echo "Outputs in $OUT:"
echo "  A_baseline   — generic portrait (no reference)"
echo "  A_variation  — MUST echo $REF_A (elderly/rustic vibe)  ← vs A_baseline"
echo "  B_variation  — MUST echo $REF_B (sunrise/landscape palette), NOT A"
echo "  C_plain      — img2img, structure only"
echo "  C_faithful   — img2img, subject held closer to the init    ← vs C_plain"
echo
echo "PASS if: A_variation differs from A_baseline and resembles $REF_A;"
echo "         B_variation resembles $REF_B and differs from A_variation;"
echo "         C_faithful stays nearer the init subject than C_plain."
