#!/usr/bin/env bash
# 07_timestep_windowing.sh — --control-start / --control-end:
# limit ControlNet to part of the denoise schedule.
#
# Diffusion runs from high noise (step 0) to clean image (step
# N-1). By default ControlNet is active at every step. With
# `--control-end 0.5`, ControlNet is active only for the first
# 50% of the schedule — it locks the composition early, then
# disengages so the prompt drives the late texture and atmosphere
# passes.
#
# Three variations on the same prompt + seed + depth map, varying
# only the control window:
#   * always:    --control-start 0.0 --control-end 1.0  (default)
#   * early:     --control-start 0.0 --control-end 0.5  (lock layout, free detail)
#   * late:      --control-start 0.5 --control-end 1.0  (let layout drift, refine)
#
# Compare the three side-by-side to see the effect of windowing.
#
# Output: out/control-tutorial/07_window/{always,early,late}.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/control-tutorial/07_window
mkdir -p "$OUT"

run() {
    local label="$1" start="$2" end="$3"
    local sub="$OUT/.tmp-$label"
    mkdir -p "$sub"
    "$PLAKAT" generate "a fox in a misty forest at sunrise, dramatic light, photographic" \
        --control depth \
        --control-image inputs/scene_depth.png \
        --control-start "$start" \
        --control-end "$end" \
        --size 512x512 --steps 24 --scheduler euler-a \
        --seed 3050 \
        --out "$sub"
    cp "$sub/plakat-3050.png" "$OUT/$label.png"
    rm -rf "$sub"
}

echo "=== always (control 0.0 → 1.0, the default) ==="
run always 0.0 1.0

echo "=== early-only (control 0.0 → 0.5; lock layout, free detail) ==="
run early 0.0 0.5

echo "=== late-only (control 0.5 → 1.0; let layout drift, refine late) ==="
run late 0.5 1.0

echo
echo "Wrote: $OUT/always.png, early.png, late.png"
echo
echo "Compare:"
echo "  always = strongest layout adherence end-to-end"
echo "  early  = locks layout in high-noise steps then lets the prompt"
echo "           drive texture/atmosphere. Often the best balance."
echo "  late   = layout drifts early then ControlNet snaps to it late."
echo "           Usually less useful than 'early' — the early steps"
echo "           commit to a layout that may fight the control signal."
