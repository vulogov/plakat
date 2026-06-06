#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — ML upscale (Real-ESRGAN ×2)
# ===================================================================
#   ./corpus/upscale.sh
#   plakat gallery corpus/images --recursive --out corpus/GALLERY.md
#
# Opens the "Transforms & post" coverage category: takes images already
# committed under corpus/images/ and runs `plakat upscale` with the ML
# Real-ESRGAN path (RRDBNet). No generation, no gated download — just the
# committed 512² SD 1.5 renders upscaled to 1024², proving the upscale
# pipeline end to end.
#
# Why ×2 (not ×4): the ×4 model's 2048² intermediate activations exceed
# Metal's single-buffer limit ("Failed to create metal resource: Buffer")
# on a 24 GB Mac. ×2 → 1024² fits Metal comfortably and is fast. For the
# bigger 2048² result, run on CPU:
#   DEVICE=cpu METHOD=real-esrgan-x4 ./corpus/upscale.sh
#
# Ungated. Downloads the Real-ESRGAN RRDBNet weights (~64 MB) on first
# run. Each upscale is a single forward pass.
#
# Override with env vars: PLAKAT, DEVICE, METHOD.
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
METHOD="${METHOD:-real-esrgan-x2}"   # ML RRDBNet ×2 (Metal-safe); ×4 needs CPU
OUT_ROOT="$ROOT/corpus/images/upscale"

# name|input — each input is an already-committed 512² corpus render.
# The ×2 model lifts them to 1024².
JOBS=(
    "landscape_x2|corpus/images/sd15/sd15_landscape/plakat-42.png"
    "still_life_x2|corpus/images/sd15/sd15_still_life/plakat-44.png"
    "portrait_x2|corpus/images/sd15/sd15_portrait/plakat-43.png"
)

echo "plakat upscale corpus — $PLAKAT"
echo "  method=$METHOD  device=$DEVICE"

for job in "${JOBS[@]}"; do
    name="${job%%|*}"
    input="${job#*|}"
    out="$OUT_ROOT/$name/$name.png"
    mkdir -p "$OUT_ROOT/$name"
    echo
    echo "==> $name : $input"
    "$PLAKAT" upscale \
        --in "$ROOT/$input" \
        --out "$out" \
        --method "$METHOD" \
        --device "$DEVICE"
done

echo
echo "Done. ${#JOBS[@]} upscales under $OUT_ROOT"
echo "Index with: plakat gallery corpus/images --recursive --out corpus/GALLERY.md"
