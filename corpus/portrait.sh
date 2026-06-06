#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — persona portraits (text-only)
# ===================================================================
#   ./corpus/portrait.sh
#   plakat gallery corpus/images --recursive --out corpus/GALLERY.md
#
# The TEXT-ONLY half of the Portrait/identity coverage row: `plakat
# portrait` with no --photo runs a portrait-tuned generate (3:4 aspect,
# face/anatomy negatives baked in) — invented "personas" defined purely
# by prompt. The identity-conditioned "lookalike" half (driven by a
# reference photo) lives in the companion corpus/portrait.hjson, which
# uses ./examples/persona/example.png.
#
# Ungated. Reuses the cached SD 1.5 base — no IP-Adapter / ArcFace
# download on this text-only path (those are only pulled by the
# photo-driven scenario).
#
# Override with env vars: PLAKAT, DEVICE, MODEL.
# ===================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAKAT="${PLAKAT:-}"
if [[ -z "$PLAKAT" ]]; then
    if [[ -x "$ROOT/target/release/plakat" ]]; then
        PLAKAT="$ROOT/target/release/plakat"
    else
        PLAKAT="plakat"
    fi
fi

DEVICE="${DEVICE:-metal}"
MODEL="${MODEL:-sd15}"
STEPS="${STEPS:-30}"
SEED="${SEED:-42}"
SIZE="${SIZE:-576x768}"   # SD 1.5-safe portrait; larger sizes OOM Metal's buffer
OUT_ROOT="$ROOT/corpus/images/portrait"

# name|prompt — each is an invented persona, identity from the prompt only.
PERSONAS=(
    "cartographer|a wise elderly cartographer with a kind weathered face and round spectacles, rolled maps and a brass compass on the desk behind, warm lamplit study, head and shoulders portrait, photorealistic, fine detail"
    "astronaut|a determined young woman astronaut, short cropped hair, soft technical lighting inside a spacecraft, photorealistic portrait"
    "blacksmith|a rugged middle-aged blacksmith with a soot-streaked face and focused gaze, glowing forge behind, dramatic warm light, photorealistic"
)

echo "plakat persona-portrait corpus — $PLAKAT"
echo "  model=$MODEL  device=$DEVICE  size=$SIZE  ${STEPS} steps  (text-only)"

for p in "${PERSONAS[@]}"; do
    name="${p%%|*}"
    prompt="${p#*|}"
    out="$OUT_ROOT/$name"
    mkdir -p "$out"
    echo
    echo "==> $name : $prompt"
    # --out is a directory (portrait writes plakat-portrait-<seed>.png inside).
    "$PLAKAT" portrait "$prompt" \
        --model "$MODEL" \
        --size "$SIZE" \
        --steps "$STEPS" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --out "$out"
done

echo
echo "Done. ${#PERSONAS[@]} persona portraits under $OUT_ROOT"
echo "Lookalike (reference-driven): ./corpus/portrait.hjson  (uses examples/persona/example.png)"
echo "Index with: plakat gallery corpus/images --recursive --out corpus/GALLERY.md"
