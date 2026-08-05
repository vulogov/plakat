#!/usr/bin/env bash
#
# TEXTURE-1 feature-demonstration driver (RFC TEXTURE-1, plakat 6.3).
#
# A prompt → a seamless, tileable PBR material set (albedo/normal/roughness/metallic/height/AO + ORM +
# a lit preview + glTF), then the weight-free tail (derive from an albedo · verify · export for an
# engine). Everything lands under corpus/images/texture/.
#
# Usage:   corpus/texture_run.sh                       # everything, release binary
#          PLAKAT=./target/debug/plakat corpus/texture_run.sh
#
# Build the RELEASE binary first (debug inference is ~50x slower): cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/texture"
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== TEXTURE-1 corpus driver ===================="
echo "binary : $PLAKAT"
echo "out    : $OUT/"
echo

# 1. Weights-free: scaffold + validate + show the resolved plan.
run "$PLAKAT" texture lint corpus/texture_stone.hjson
run "$PLAKAT" texture show corpus/texture_stone.hjson

# 2. Generate a full seamless PBR material (the SDXL albedo downloads from HF on first use). Writes the
#    channel maps + ORM + a lit preview + a glTF material + material.json (recipe + scorecard).
#    Four materials spanning the axes the maps encode:
#      stone  — a matte DIELECTRIC (metallic.png is solid BLACK — stone is a non-metal)
#      steel  — a CONDUCTOR       (metallic.png is solid WHITE — the metallic channel in action)
#      leaves — a coloured organic dielectric, from-albedo roughness
#      river  — a WET dielectric  (glossy: the shine is low roughness, NOT metal)
run "$PLAKAT" texture render corpus/texture_stone.hjson  --out "$OUT/stone"
run "$PLAKAT" texture render corpus/texture_steel.hjson  --out "$OUT/steel"
run "$PLAKAT" texture render corpus/texture_leaves.hjson --out "$OUT/leaves"
run "$PLAKAT" texture render corpus/texture_river.hjson  --out "$OUT/river"

# 3. Weights-free tail: derive a full set from the generated albedo, score it, and re-pack for an engine.
run "$PLAKAT" texture derive "$OUT/stone/albedo.png" --out "$OUT/stone_derived"
run "$PLAKAT" texture verify "$OUT/stone"
run "$PLAKAT" texture export "$OUT/stone" --out "$OUT/stone_unreal" --naming unreal --gltf

echo "==================== done → $OUT/ ===================="
echo "The material tiles: open $OUT/stone/preview.png, or place the maps on a surface in any engine."
