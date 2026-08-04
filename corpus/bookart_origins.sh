#!/usr/bin/env bash
# ===================================================================
# plakat proof corpus — the six bookart origin LoRAs (RFC BOOKART-1 B4 + B5)
# ===================================================================
#   corpus/bookart_origins.sh                 # render all six
#   OUT=/tmp/origins corpus/bookart_origins.sh
#
# Renders one signature frontispiece per illustration tradition, each driven by
# its hosted sd15 origin LoRA (auto-resolved from vulogov98/plakat-bookart —
# `<origin>-sd15.safetensors`, trigger `bookart_<origin> style`), then a contact
# sheet. The `generic` row uses NO LoRA (the line-art fallback) for comparison.
#
#   russian / english / japanese  — B4  (Bilibin / Beardsley / Hokusai corpora)
#   american / european / chinese — B5  (Pyle / Doré / Shan-Hai-Jing corpora)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="${OUT:-corpus/images/bookart/origins}"
SEED="${SEED:-3}"
mkdir -p "$OUT"

# origin : technique : prompt  (a motif idiomatic to each tradition)
ROWS=(
  "russian:woodcut:a firebird among oak branches"
  "english:line:a rose and peacock in foliate ornament"
  "japanese:line:a crane over pine boughs and waves"
  "american:woodcut:a sailing ship with a compass rose"
  "european:engraving:an acanthus and laurel cartouche"
  "chinese:line:a dragon among stylized clouds"
  "generic:line:a leafy scroll ornament"
)

for row in "${ROWS[@]}"; do
  IFS=: read -r origin tech prompt <<< "$row"
  echo "===== $(date '+%H:%M:%S') origin=$origin technique=$tech ====="
  "$PLAKAT" bookart illustrate "$prompt" \
    --origin "$origin" --technique "$tech" \
    --out "$OUT/${origin}.png" --seed "$SEED" \
    || echo "!! FAILED origin=$origin"
done

# A single contact sheet of all seven for side-by-side comparison.
"$PLAKAT" bookart proof "$OUT" --out "$OUT/contact_sheet.png" || true
echo "done → $OUT  (transparent PNGs; composite on a light page to view)"
