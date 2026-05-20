#!/usr/bin/env bash
# 04_auto_depth.sh — v0.10: auto-generate the depth map from a
# regular image with --control-from. No need to pre-render a
# depth map; plakat runs Depth-Anything-V2 on the source for you.
#
# Two scenarios shown:
#   * plakat generate: pass --control-from <photo> alongside a
#     prompt. The depth is estimated from the photo and used to
#     guide layout.
#   * plakat img2img: if --control depth is set but neither
#     --control-image nor --control-from is supplied, the
#     <INPUT> image is auto-annotated by default.
#
# First run downloads Depth-Anything-V2-small (~99 MB) on top of
# SD 1.5 + ControlNet-Depth. All cached after.
#
# Outputs:
#   out/control-tutorial/04_auto_depth/from_flag.png
#   out/control-tutorial/04_auto_depth/from_input_default.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_plakat.sh"
cd "$SCRIPT_DIR/.."

OUT=out/control-tutorial/04_auto_depth
mkdir -p "$OUT"

# We reuse the IMG2IMG tutorial's landscape PNG as a "regular image"
# (any photo works). It already ships in the repo.
SOURCE="../IMG2IMG/inputs/landscape.png"

echo "=== plakat generate with --control-from ==="
"$PLAKAT" generate "a quiet meadow with a single oak tree, golden hour" \
    --control depth \
    --control-from "$SOURCE" \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 3004 \
    --out "$OUT.tmp"
cp "$OUT.tmp/plakat-3004.png" "$OUT/from_flag.png"
rm -rf "$OUT.tmp"

echo
echo "=== plakat img2img with --control depth (no image/from set) ==="
echo "    The <INPUT> image is auto-annotated by default."
"$PLAKAT" img2img "$SOURCE" \
    --prompt "the same scene as a watercolor painting" \
    --strength 0.7 \
    --control depth \
    --size 512x512 --steps 20 --scheduler euler-a \
    --seed 3005 \
    --out "$OUT.tmp"
cp "$OUT.tmp/plakat-img2img-3005.png" "$OUT/from_input_default.png"
rm -rf "$OUT.tmp"

echo
echo "Wrote:"
echo "  $OUT/from_flag.png         (generate --control-from PATH)"
echo "  $OUT/from_input_default.png (img2img --control depth, no image/from)"
echo
echo "Both outputs use depth maps auto-estimated from the source"
echo "landscape — no manual depth-map prep required."
