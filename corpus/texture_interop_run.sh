#!/usr/bin/env bash
#
# TEXTURE-1 6.6.0 ENGINE-INTEROP corpus driver.
#
# Renders two source materials (steel — with a brushed anisotropy grain — and stone), then re-packs each
# to EVERY engine target and shows how the naming + packing + material document differ. The re-pack half
# is fully weight-free (no GPU). Everything lands under corpus/images/texture/interop/.
#
# Usage:   corpus/texture_interop_run.sh                    # everything, release binary
#          PLAKAT=./target/debug/plakat corpus/texture_interop_run.sh
#
# Build the RELEASE binary first (debug inference is ~50x slower): cargo build --release --features metal
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/texture/interop"
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || echo "  ! step failed — continuing"; echo; }

echo "==================== TEXTURE-1 6.6.0 engine-interop driver ===================="
echo "binary : $PLAKAT"
echo "out    : $OUT/"
echo

# 1. Render the source materials (the SDXL albedo downloads from HF on first use). `steel` carries a
#    brushed ANISOTROPY grain → its glTF gets the KHR_materials_anisotropy extension.
run "$PLAKAT" texture render corpus/texture_steel.hjson --out "$OUT/steel"
run "$PLAKAT" texture render corpus/texture_stone.hjson --out "$OUT/stone"

# 2. WEIGHT-FREE: re-pack each material to every engine target. One `--engine` flag picks the naming
#    convention, the packed map (ORM vs Unity-HDRP mask map — they pack the SAME data DIFFERENTLY), and
#    the material document (glTF / MaterialX):
#      gltf       → material.gltf (+ KHR_materials_anisotropy for steel) · ORM
#      unreal     → T_BaseColor…T_ORM (ORM: R=AO/G=rough/B=metal)
#      unity-hdrp → mask_map.png (RGBA: R=metal/G=AO/B=detail/A=smoothness) — NOT ORM
#      godot      → ORM (Godot's occlusion/roughness/metallic layout)
#      materialx  → material.mtlx (1.38 standard_surface, for USD / Arnold / Substance)
for eng in gltf unreal unity-hdrp godot materialx; do
  run "$PLAKAT" texture export "$OUT/steel" --out "$OUT/steel_$eng" --engine "$eng"
  run "$PLAKAT" texture export "$OUT/stone" --out "$OUT/stone_$eng" --engine "$eng"
done

# 3. One-shot GENERATE + export straight into an engine convention (render --engine).
run "$PLAKAT" texture render corpus/texture_stone.hjson --out "$OUT/stone_unreal_direct" --engine unreal

echo "==================== done → $OUT/ ===================="
echo "glTF anisotropy : grep -o KHR_materials_anisotropy $OUT/steel_gltf/material.gltf"
echo "Unity HDRP mask : $OUT/steel_unity-hdrp/mask_map.png  (RGBA — differs from ORM)"
echo "Unreal ORM      : $OUT/steel_unreal/T_ORM.png"
echo "MaterialX       : $OUT/steel_materialx/material.mtlx  (1.38 standard_surface)"
