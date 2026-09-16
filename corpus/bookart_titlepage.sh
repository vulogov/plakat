#!/usr/bin/env bash
#
# plakat bookart title-page — demo driver.
#
# Generates a complete TWO-PAGE PDF — a BOOK title page (framed by a bookart ornament) + a
# CHAPTER page (opened by a bookart device) — in the old letterpress style, from HJSON. The
# ornaments are PROCEDURAL bookart (weight-free, no GPU): a `border` frame and a `fleuron`
# rosette. Each spec compiles to a reusable Typst `title-page`; a tiny book.typ #imports both
# and paginates them, so the whole thing renders in one Typst compile.
#
# Usage:   corpus/bookart_titlepage.sh
#          PLAKAT=./target/release/plakat corpus/bookart_titlepage.sh
#
# Needs:  the plakat binary (no GPU — everything here is weight-free) and `typst` on PATH.
set -u

PLAKAT="${PLAKAT:-./target/debug/plakat}"
OUT="corpus/images/bookart-titlepage"
BOOK="corpus/bookart_titlepage_book.hjson"
CHAP="corpus/bookart_titlepage_chapter.hjson"

command -v typst >/dev/null || {
  echo "error: typst not found on PATH (needed to compile the pages)"
  exit 1
}
[ -x "$PLAKAT" ] || {
  echo "error: plakat binary not found at $PLAKAT (set PLAKAT=…)"
  exit 1
}

mkdir -p "$OUT"
run() {
  echo "+ $*"
  "$@" || {
    echo "  ! step failed"
    exit 1
  }
  echo
}

echo "================ plakat bookart title-page — 2-page demo ================"
echo "binary : $PLAKAT"
echo "out    : $OUT/"
echo

# 1. Render the bookart ORNAMENTS (procedural = weight-free, deterministic): a frame + a rosette device.
rm -f "$OUT/frame_spec.hjson" "$OUT/device_spec.hjson"
run "$PLAKAT" bookart new "$OUT/frame_spec.hjson"  --type border  --origin generic --technique line --page a5
run "$PLAKAT" bookart render "$OUT/frame_spec.hjson"  --out "$OUT/frame.png"
run "$PLAKAT" bookart new "$OUT/device_spec.hjson" --type fleuron --origin generic --technique line --page a5
run "$PLAKAT" bookart render "$OUT/device_spec.hjson" --out "$OUT/device.png"

# 2. Each HJSON spec → a reusable Typst `title-page` (compile it directly with --verify to prove it renders).
#    The book page is FITTED inside the frame's clear window; the chapter page crops the rosette to a device.
run "$PLAKAT" bookart title-page "$BOOK" --out "$OUT/title.typ"   --verify
run "$PLAKAT" bookart title-page "$CHAP" --out "$OUT/chapter.typ" --verify

# 3. A tiny book that pulls in both pages and paginates them (title on p.1, chapter on p.2).
cat >"$OUT/book.typ" <<'TYP'
// The finished book: import each page's `title-page` and render them in order.
#import "title.typ": title-page
#import "chapter.typ": title-page as chapter-page

#title-page
#pagebreak()
#chapter-page
TYP

# 4. Compile the complete two-page PDF.
echo "+ typst compile $OUT/book.typ $OUT/book.pdf"
typst compile "$OUT/book.typ" "$OUT/book.pdf" || {
  echo "  ! typst compile failed"
  exit 1
}

pages=$(pdfinfo "$OUT/book.pdf" 2>/dev/null | awk '/Pages/{print $2}' || echo "?")
echo
echo "✓ done — $OUT/book.pdf ($pages page(s): a framed book title + a chapter opener, with bookart ornaments)"
ls -lh "$OUT"/book.pdf "$OUT"/frame.png "$OUT"/device.png 2>/dev/null | awk '{print "   "$5"\t"$9}'
