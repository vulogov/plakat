#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — img2img (style transfer)
# ===================================================================
#   ./corpus/img2img.sh
#   plakat gallery corpus/images --recursive --out corpus/GALLERY.md
#
# Extends the "Transforms & post" category: takes photographic renders
# already committed under corpus/images/sdxl/ and runs `plakat img2img`
# to restyle them into painterly mediums while keeping the composition.
# The medium is steered by the PROMPT (not `--look`), so there is no
# LoRA / Civitai download — it reuses the cached SDXL base only.
#
# Ungated, no new download (SDXL is already cached from sdxl.hjson).
# strength 0.55 keeps the source structure while letting the new medium
# take over; raise toward 0.7 for a looser restyle.
#
# Override with env vars: PLAKAT, DEVICE, MODEL, STRENGTH.
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
MODEL="${MODEL:-sdxl}"
STRENGTH="${STRENGTH:-0.55}"
STEPS="${STEPS:-28}"
SEED="${SEED:-42}"
NEGATIVE="${NEGATIVE:-blurry, low quality, watermark}"
OUT_ROOT="$ROOT/corpus/images/img2img"

# name|input|prompt — each input is a committed photographic SDXL render;
# the prompt names the same subject in a new medium.
JOBS=(
    "coast_oil|corpus/images/sdxl/sdxl_landscape/plakat-42.png|a dramatic coastal cliff at sunset, crashing waves and seabirds, expressive impressionist oil painting, thick visible brushstrokes, palette knife"
    "portrait_watercolor|corpus/images/sdxl/sdxl_portrait/plakat-43.png|a portrait of a young woman with freckles and curly red hair, delicate watercolor painting, soft translucent washes, loose wet-on-wet brushwork"
    "forest_inkwash|corpus/images/sdxl/sdxl_fantasy/plakat-44.png|a glowing mushroom forest at night with a small stone bridge, traditional Japanese sumi-e ink-wash painting, muted tones, soft bleeding ink"
)

echo "plakat img2img corpus — $PLAKAT"
echo "  model=$MODEL  device=$DEVICE  strength=$STRENGTH  ${STEPS} steps"

for job in "${JOBS[@]}"; do
    name="${job%%|*}"
    rest="${job#*|}"
    input="${rest%%|*}"
    prompt="${rest#*|}"
    out="$OUT_ROOT/$name"
    mkdir -p "$out"
    echo
    echo "==> $name : $input"
    # --out is a directory (img2img writes plakat-img2img-<seed>.png inside).
    "$PLAKAT" img2img "$ROOT/$input" \
        --prompt "$prompt" \
        --negative "$NEGATIVE" \
        --model "$MODEL" \
        --strength "$STRENGTH" \
        --steps "$STEPS" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --out "$out"
done

echo
echo "Done. ${#JOBS[@]} restyles under $OUT_ROOT"
echo "Index with: plakat gallery corpus/images --recursive --out corpus/GALLERY.md"
