#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — AnimateDiff (SD 1.5 motion)
# ===================================================================
#   ./corpus/animate.sh
#   plakat gallery corpus/images --recursive --out corpus/GALLERY.md
#
# Renders a handful of short motion-coherent clips with the AnimateDiff
# V3 motion adapter. Each clip lands as zero-padded PNG frames plus an
# `animation.gif` under corpus/images/animate/<name>/.
#
# Base model: AnimateDiff rides on an SD 1.5 backbone, but the *vanilla*
# SD 1.5 base yields degraded/mosaic frames (reproduced 1:1 by diffusers
# — it is not a plakat bug). An aesthetic SD 1.5 fine-tune is the
# standard practice, so the corpus uses DreamShaper-8 via --model.
#
# Ungated. Downloads DreamShaper-8 (~2 GB) + the V3 motion adapter
# (~1.6 GB) on first run. 8-frame / 512² clips fit comfortably in 24 GB
# unified memory; 16-frame clips roughly double the activation peak (the
# whole frame window denoises as one batch), so the corpus stays at 8.
# ===================================================================
set -euo pipefail

# Resolve the plakat binary: an explicit $PLAKAT, else a release build,
# else whatever is on PATH.
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
OUT_ROOT="$ROOT/corpus/images/animate"

# Shared render settings — kept identical across clips so the corpus is
# reproducible and the only variable is the prompt.
MODEL="${MODEL:-Lykon/dreamshaper-8}"   # aesthetic SD 1.5 fine-tune
NEGATIVE="${NEGATIVE:-bad quality, worse quality, blurry, low resolution}"
FRAMES="${FRAMES:-8}"
STEPS="${STEPS:-25}"
SIZE="${SIZE:-512x512}"
SEED="${SEED:-42}"
GIF_DELAY_MS="${GIF_DELAY_MS:-100}"   # 10 fps

# name|prompt — one motion-bearing prompt per clip.
CLIPS=(
    "fox_snow|a red fox walking through a snowy forest, cinematic, highly detailed"
    "ocean_waves|ocean waves crashing on a rocky shore at sunset, cinematic, golden light"
    "flower_bloom|a time-lapse of a red rose blooming, macro photography, soft light"
)

echo "plakat AnimateDiff corpus — $PLAKAT"
echo "  base=$MODEL  device=$DEVICE  ${FRAMES}f  ${STEPS} steps  $SIZE"

for clip in "${CLIPS[@]}"; do
    name="${clip%%|*}"
    prompt="${clip#*|}"
    out="$OUT_ROOT/$name"
    echo
    echo "==> $name : $prompt"
    "$PLAKAT" animate \
        --animatediff \
        --model "$MODEL" \
        --from "$prompt" \
        --negative "$NEGATIVE" \
        --frames "$FRAMES" \
        --steps "$STEPS" \
        --size "$SIZE" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --gif \
        --gif-delay-ms "$GIF_DELAY_MS" \
        --out "$out"
done

echo
echo "Done. ${#CLIPS[@]} clips under $OUT_ROOT"
echo "Index with: plakat gallery corpus/images --recursive --out corpus/GALLERY.md"
