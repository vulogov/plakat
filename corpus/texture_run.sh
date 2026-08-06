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
#    Materials spanning the axes the maps encode:
#      stone  — a matte DIELECTRIC (metallic.png is solid BLACK — stone is a non-metal)
#      steel  — a CONDUCTOR with a BRUSHED GRAIN (metallic solid WHITE + an anisotropy flow map; 6.4.0)
#      leaves — a coloured organic dielectric, auto roughness
#      river  — a WET dielectric  (glossy: the shine is low roughness, NOT metal)
#      rust   — a COMPOSITE (bare steel + orange rust): metallic:"auto" → a STRUCTURED metal mask (6.4.0)
run "$PLAKAT" texture render corpus/texture_stone.hjson  --out "$OUT/stone"
run "$PLAKAT" texture render corpus/texture_steel.hjson  --out "$OUT/steel"
run "$PLAKAT" texture render corpus/texture_leaves.hjson --out "$OUT/leaves"
run "$PLAKAT" texture render corpus/texture_river.hjson  --out "$OUT/river"
run "$PLAKAT" texture render corpus/texture_rust.hjson   --out "$OUT/rust"

# 3. Weights-free tail: derive a full set from the generated albedo, score it, and re-pack for an engine.
run "$PLAKAT" texture derive "$OUT/stone/albedo.png" --out "$OUT/stone_derived"
run "$PLAKAT" texture verify "$OUT/stone"
run "$PLAKAT" texture export "$OUT/stone" --out "$OUT/stone_unreal" --naming unreal --gltf

# 4. NEW in 6.4.0 — spatially-varying channels, blend, variations (weight-free where noted):
#    a) metallic:"auto" on the composite rust: re-derive from just the albedo → a STRUCTURED metal mask
#       (bare-metal patches white, rust black). Contrast with `--metallic 0` (a known dielectric).
run "$PLAKAT" texture derive "$OUT/rust/albedo.png" --out "$OUT/rust_auto"   --metallic auto --roughness auto
run "$PLAKAT" texture derive "$OUT/rust/albedo.png" --out "$OUT/rust_flat"   --metallic 0
#    b) anisotropy on brushed steel from just the albedo (preview highlight stretches along the grain).
run "$PLAKAT" texture derive "$OUT/steel/albedo.png" --out "$OUT/steel_aniso" --metallic 1 --anisotropy 0.85 --anisotropy-angle 0
#    c) blend two materials through a TILEABLE mask (stone → leaves) — the result still tiles.
run "$PLAKAT" texture blend "$OUT/stone" "$OUT/leaves" --out "$OUT/stone_leaves" --mask mix
#    d) variations: three seed variants side-by-side, keep the best-scoring one at the top.
run "$PLAKAT" texture render corpus/texture_stone.hjson --out "$OUT/stone_variants" --variations 3 --keep-best

echo "==================== done → $OUT/ ===================="
echo "Look: $OUT/rust/metallic.png (STRUCTURED, 6.4.0) vs $OUT/stone/metallic.png (flat black — correct)."
echo "      $OUT/steel_aniso/preview.png (grain-stretched highlight) · $OUT/stone_leaves/preview.png (blend)."
echo "Every channel tiles: open any preview.png, or place the maps on a surface in an engine."
