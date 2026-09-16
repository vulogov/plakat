#!/usr/bin/env bash
#
# plakat bookart title-page — demo driver.
#
# Generates a complete TWO-PAGE PDF — a BOOK title page + a CHAPTER title page — in the old
# letterpress style, entirely from HJSON, demonstrating `plakat bookart title-page`. Each spec
# compiles to a reusable Typst `title-page`; a tiny book.typ #imports both and renders them as
# two pages, so the whole thing paginates in one Typst compile.
#
# Usage:   corpus/bookart_titlepage.sh
#          PLAKAT=./target/debug/plakat corpus/bookart_titlepage.sh
#
# Needs:  the plakat binary (no GPU — title pages are weight-free) and `typst` on PATH.
set -u

PLAKAT="${PLAKAT:-./target/release/plakat}"
OUT="corpus/images/bookart-titlepage"
BOOK="corpus/bookart_titlepage_book.hjson"
CHAP="corpus/bookart_titlepage_chapter.hjson"

command -v typst >/dev/null || { echo "error: typst not found on PATH (needed to compile the pages)"; exit 1; }
[ -x "$PLAKAT" ] || { echo "error: plakat binary not found at $PLAKAT (set PLAKAT=…)"; exit 1; }

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || { echo "  ! step failed"; exit 1; }; echo; }

echo "================ plakat bookart title-page — 2-page demo ================"
echo "binary : $PLAKAT"
echo "out    : $OUT/"
echo

# 1. Each HJSON spec → a reusable Typst `title-page` (compile it directly with --verify to prove it renders).
run "$PLAKAT" bookart title-page "$BOOK" --out "$OUT/title.typ" --verify
run "$PLAKAT" bookart title-page "$CHAP" --out "$OUT/chapter.typ" --verify

# 2. A tiny book that pulls in both pages and paginates them (title on p.1, chapter on p.2).
cat > "$OUT/book.typ" <<'TYP'
// The finished book: import each page's `title-page` and render them in order.
#import "title.typ": title-page
#import "chapter.typ": title-page as chapter-page

#title-page
#pagebreak()
#chapter-page
TYP

# 3. Compile the complete two-page PDF.
echo "+ typst compile $OUT/book.typ $OUT/book.pdf"
typst compile "$OUT/book.typ" "$OUT/book.pdf" || { echo "  ! typst compile failed"; exit 1; }

pages=$(pdfinfo "$OUT/book.pdf" 2>/dev/null | awk '/Pages/{print $2}' || echo "?")
echo
echo "✓ done — $OUT/book.pdf ($pages page(s): book title + chapter I)"
ls -lh "$OUT"/book.pdf "$OUT"/title.typ "$OUT"/chapter.typ 2>/dev/null | awk '{print "   "$5"\t"$9}'
