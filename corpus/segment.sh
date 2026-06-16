#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — segment (Segment-Anything / MobileSAM)
# ===================================================================
# The compose-&-edit ENABLER: click an object → a binary mask → drive an edit
# with it. Here we select the astronaut with a few clicks, invert (so the mask
# covers the BACKGROUND, white = repaint), then inpaint only the background —
# the subject is preserved untouched. "Keep the subject, change the world",
# driven entirely by SAM clicks.
#
# Subject choice matters: a cleanly-separable figure on an uncluttered backdrop
# (the astronaut) makes an honest background-swap. A subject tangled with
# foreground props (e.g. the dockside captain, rope rigging around his head)
# can't be cleanly lifted — the props are preserved with him.
#
# Two stages:
#   1. plakat segment  — MobileSAM, ~40 MB weights auto-download, Metal/CPU
#      fast (~0.4 s inference). Writes bg-mask.png. (cheap; safe to re-run)
#   2. plakat img2img --mask  — SDXL-inpaint of the masked background. GPU,
#      single pass; the OOM guard aborts cleanly if RAM is too tight.
#
# Refine tip: if a click over/under-selects, add points —
#   --point 0.5,0.4 --point 0.5,0.7        (more foreground points)
#   --point 0.5,0.4 --point 0.1,0.1:bg     (carve a background corner away)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"

SRC="$ROOT/corpus/images/portrait/astronaut/plakat-portrait-42.png"
OUT="$ROOT/corpus/images/segment"
mkdir -p "$OUT"

# 1) Select the astronaut (helmet + torso + lower body), --invert → the mask
#    covers the background (white = repaint). --grow leaves a margin so the
#    repaint never touches the suit's edge; --feather softens the seam.
"$PLAKAT" segment --in "$SRC" --out "$OUT/bg-mask.png" \
  --point 0.5,0.2 --point 0.5,0.45 --point 0.5,0.7 \
  --invert --grow 16 --feather 4 --device metal

# 2) Repaint ONLY the background with SDXL-inpaint (clean boundaries, style-matches
#    the SDXL portrait) — swap the spacecraft interior for the lunar surface.
"$PLAKAT" img2img "$SRC" \
  --prompt "standing on the grey cratered surface of the Moon, the blue Earth rising in a black starry sky, harsh raking sunlight, sharp shadows" \
  --mask "$OUT/bg-mask.png" \
  --model sdxl-inpaint --strength 0.85 --steps 28 --seed 42 --device metal \
  --out "$OUT"

echo "✓ wrote corpus/images/segment/ (astronaut kept, background swapped to the Moon via SAM clicks)"
