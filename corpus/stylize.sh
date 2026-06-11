#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — stylize: ref-variation  vs  InstantStyle (true style)
# ===================================================================
# Stylize applies a `--ref` image's LOOK to a subject via IP-Adapter — no
# prompt, no training. As of v0.47 there are TWO paths, and this driver proves
# both, back-to-back, on the same subject + watercolour reference:
#
#   • DEFAULT (concat) — ref-guided VARIATION. Transfers the ref's palette /
#     appearance, NOT painterly *texture*, so output stays photoreal. This is an
#     IP-Adapter limit, not a base limit. `--ref-blur` suppresses the ref's
#     content; `--ref-weight` scales its influence.
#
#   • --instantstyle — TRUE painterly STYLE transfer (SD 1.5 + SDXL). Injects
#     the reference ONLY into the style block (SDXL up_blocks.0.attentions.1,
#     SD 1.5 up_blocks.1.attentions.1) via a decoupled IP cross-attention, so the
#     watercolour BRUSHWORK lands without cloning the ref's content. The real
#     style machine. `--style-scale` dials injection strength (default 1.0).
#     Loads a second (vendored) UNet for the backbone → extra memory.
#
# References are watercolour paintings in corpus/style/watercolour; the subject
# is a photoreal portrait → a watercolour portrait. SDXL is sharper (native
# 1024²); SD 1.5 is the lighter backbone. Ungated, Metal-safe.
#
# The A/B to look for: #2 (InstantStyle) should pick up the ref's WATERCOLOUR
# texture/brushwork while keeping the subject; #1 (concat) keeps the subject but
# stays photoreal — the texture doesn't transfer. That gap is the whole point.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAKAT="${PLAKAT:-$ROOT/target/release/plakat}"
OUT="$ROOT/corpus/images/stylize"
WC="$ROOT/corpus/style/watercolour"
SUBJECT="$ROOT/corpus/images/sd15/sd15_portrait/plakat-43.png"
mkdir -p "$OUT"

# 1. DEFAULT (concat) — ref-guided VARIATION. The baseline: subject preserved,
#    palette shifts toward the ref, but it stays photoreal (no brushwork).
"$PLAKAT" stylize \
  --in  "$SUBJECT" \
  --ref "$WC/figures.jpeg" \
  --ref-blur 4 --ref-weight 1.0 \
  --strength 0.6 --model sdxl --seed 42 --device metal \
  --out "$OUT/portrait-variation.png"

# 2. InstantStyle (SDXL) — TRUE style transfer. Same subject + same ref as #1,
#    so this is a direct A/B: the watercolour texture should now land.
"$PLAKAT" stylize \
  --in  "$SUBJECT" \
  --ref "$WC/figures.jpeg" \
  --instantstyle --style-scale 4 \
  --strength 0.8 --model sdxl --seed 42 --device metal \
  --out "$OUT/portrait-instantstyle.png"

# 3. InstantStyle (SDXL) — a DIFFERENT watercolour ref, to show the style range
#    (a snowy-village palette + its brushwork transfer onto the same subject).
"$PLAKAT" stylize \
  --in  "$SUBJECT" \
  --ref "$WC/snow-village.jpeg" \
  --instantstyle --style-scale 4 \
  --strength 0.8 --model sdxl --seed 42 --device metal \
  --out "$OUT/portrait-instantstyle-snow.png"

# NOTE: InstantStyle on SD 1.5 (`--model sd15 --instantstyle`) works but is
# EXPERIMENTAL — the correct style block is the full `up_blocks.1` (all 3
# attentions), but SD 1.5's style perception is weak (the InstantStyle repo flags
# this): it goes photoreal → faint → structure-melt with scale, with no clean
# strong-watercolour window like SDXL. So the corpus demos InstantStyle on SDXL
# only; SD 1.5 stays supported-but-experimental in code/docs.

echo "✓ wrote corpus/images/stylize/:"
echo "    portrait-variation.png          (concat — ref-variation baseline)"
echo "    portrait-instantstyle.png       (InstantStyle SDXL — A/B vs #1, warm watercolour)"
echo "    portrait-instantstyle-snow.png  (InstantStyle SDXL — cool watercolour, 2nd ref)"
