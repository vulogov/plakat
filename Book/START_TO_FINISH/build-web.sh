#!/usr/bin/env bash
# Build the print master and a web-optimized copy of "A Poster, Start to Finish".
#
#   START_TO_FINISH.pdf       — print master (full-resolution images)
#   START_TO_FINISH-web.pdf   — web/email build (150-DPI images, JPEG, deduped logo)
#
# Usage:  ./build-web.sh            (from the book directory)
# Needs:  typst, ghostscript (gs)

set -euo pipefail
cd "$(dirname "$0")"

SRC="START_TO_FINISH.typ"
PRINT="START_TO_FINISH.pdf"
WEB="START_TO_FINISH-web.pdf"
DPI="${DPI:-150}"          # override e.g.  DPI=120 ./build-web.sh  for a smaller file

command -v typst >/dev/null || { echo "error: typst not found on PATH" >&2; exit 1; }
command -v gs    >/dev/null || { echo "error: ghostscript (gs) not found on PATH" >&2; exit 1; }

echo "→ compiling $SRC …"
typst compile "$SRC" "$PRINT"

echo "→ web-optimizing → $WEB (images @ ${DPI} DPI) …"
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.6 -dNOPAUSE -dBATCH -dQUIET \
   -dPDFSETTINGS=/ebook \
   -dDownsampleColorImages=true -dColorImageResolution="$DPI" -dColorImageDownsampleType=/Bicubic \
   -dDownsampleGrayImages=true  -dGrayImageResolution="$DPI"  -dGrayImageDownsampleType=/Bicubic \
   -dDownsampleMonoImages=true  -dMonoImageResolution=300 \
   -dDetectDuplicateImages=true -dCompressFonts=true -dSubsetFonts=true \
   -o "$WEB" "$PRINT"

# Report the result.
pages=$(pdfinfo "$WEB" 2>/dev/null | awk '/Pages/{print $2}' || echo "?")
printf '\n✓ done — %s pages\n' "$pages"
ls -lh "$PRINT" "$WEB" | awk '{print "   "$5"\t"$9}'
