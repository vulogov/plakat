#!/usr/bin/env bash
#
# plakat bookart title-page — full demo driver.
#
# Generates a complete SEVEN-PAGE PDF that exercises a broad subset of `plakat bookart`:
#
#   1. FRONTISPIECE       — a pictorial B/W plate from `bookart illustrate` (the DIFFUSION tier).
#   2. TITLE PAGE         — the dense antique-letterpress hierarchy with a PICTORIAL EMBLEM woven into the
#                           type (an `illustrate` vignette device) — the Buslaev-1858 treatment.
#   3. HALF / BOOK TITLE  — a second title treatment, framed by a `bookart kit`'s border ornament.
#   4. CHAPTER I          — opened by a `fleuron` rosette DEVICE.
#   5. · SECTION I        — a SUBCHAPTER title page under chapter I, opened by a minimal `dinkus` mark.
#   6. · SECTION II       — a second SUBCHAPTER title page, same subordinate style.
#   7. CHAPTER II         — opened by the kit's `divider` band — a DIFFERENT ornament TYPE (procedural
#                           rosettes are one geometric family; real variety comes from another type).
#
# Each page compiles to a reusable Typst `title-page`; a tiny book.typ #imports all seven and paginates
# them into one PDF. The kit / rosette / dinkus are weight-free (procedural); `illustrate` (the frontispiece
# plate and the title emblem) uses a model. Then a `bookart cover` lays out the book's DUST JACKET
# (back · spine · front on one sheet; the spine width computed from the page count), and finally
# `bookart book` assembles a whole typeset BOOK from a Markdown manuscript (the capstone) → thebook.pdf.
#
# Usage:   corpus/bookart_titlepage.sh
#          PLAKAT=./target/release/plakat STEPS=50 corpus/bookart_titlepage.sh   # release binary + finer plate
#
# Needs:  the plakat binary and `typst` on PATH. The `illustrate` step needs a model (auto-downloaded)
#         and is slow on a debug build — use a RELEASE binary (`cargo build --release --features metal`)
#         for the pictorial plate.
set -u

PLAKAT="${PLAKAT:-./target/debug/plakat}"
STEPS="${STEPS:-40}"
OUT="corpus/images/bookart-titlepage"
KIT="corpus/bookart_titlepage_kit.hjson"
FRONT="corpus/bookart_titlepage_frontispiece.hjson"
TITLE="corpus/bookart_titlepage_title.hjson"
BOOK="corpus/bookart_titlepage_book.hjson"
CHAP1="corpus/bookart_titlepage_chapter.hjson"
SUB1="corpus/bookart_titlepage_sub1.hjson"
SUB2="corpus/bookart_titlepage_sub2.hjson"
CHAP2="corpus/bookart_titlepage_chapter2.hjson"
COVER="corpus/bookart_titlepage_cover.hjson"
MANUSCRIPT="corpus/bookart_book_manuscript.md"
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"

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

echo "============ plakat bookart title-page — 7-page demo (kit · illustrate · emblem · subchapters) ============"
echo "binary : $PLAKAT"
echo "steps  : $STEPS   out: $OUT/"
echo

# 1. A coherent bookart KIT — a matched set (border / divider / fleuron), one hand.
rm -f "$OUT/rosette_spec.hjson" "$OUT/dinkus_spec.hjson"
run "$PLAKAT" bookart kit "$KIT" --out "$OUT/kit" --no-coherence

# 2. A fleuron ROSETTE device for chapter I (chapter II uses the kit's divider band — a different type),
#    and a minimal DINKUS mark for the two subchapter (section) pages — three distinct ornament families.
run "$PLAKAT" bookart new "$OUT/rosette_spec.hjson" --type fleuron --origin generic --technique line --page a5
run "$PLAKAT" bookart render "$OUT/rosette_spec.hjson" --out "$OUT/rosette-1.png" --seed 11
run "$PLAKAT" bookart new "$OUT/dinkus_spec.hjson" --type dinkus --origin generic --technique line --page a5
run "$PLAKAT" bookart render "$OUT/dinkus_spec.hjson" --out "$OUT/dinkus.png" --seed 5

# 3. Two pictorial plates from the diffusion tier (need a model; slow on debug):
#    a full-page FRONTISPIECE, and a small centred EMBLEM (a `vignette`) woven into the title-page type.
run "$PLAKAT" bookart illustrate "a tall sailing ship on stormy seas, antique wood engraving, dense cross-hatching, bold black lines, high contrast, black and white" \
  --origin generic --type frontispiece --page a5 --steps "$STEPS" --out "$OUT/frontispiece.png"
run "$PLAKAT" bookart illustrate "a naval trophy emblem, crossed anchors and signal flags, a ship's wheel, a sextant and coiled rope, wreathed in laurel, antique wood engraving, intricate cross-hatching, high contrast, black and white, symmetrical centered vignette on a white ground" \
  --origin generic --type vignette --page a5 --steps "$STEPS" --out "$OUT/emblem.png"

# 4. Each HJSON spec → a reusable Typst `title-page` (--verify compiles each on its own).
run "$PLAKAT" bookart title-page "$FRONT" --out "$OUT/00-frontispiece.typ" --verify
run "$PLAKAT" bookart title-page "$TITLE" --out "$OUT/01-title.typ" --margin 15 --fit --historical --verify   # dense hierarchy: wider box, auto-fit to one page, antique figures
run "$PLAKAT" bookart title-page "$BOOK"  --out "$OUT/02-book.typ"         --verify
run "$PLAKAT" bookart title-page "$CHAP1" --out "$OUT/03-chapter1.typ"     --verify
run "$PLAKAT" bookart title-page "$SUB1"  --out "$OUT/04-section1.typ"     --verify
run "$PLAKAT" bookart title-page "$SUB2"  --out "$OUT/05-section2.typ"     --verify
run "$PLAKAT" bookart title-page "$CHAP2" --out "$OUT/06-chapter2.typ"     --verify

# 5. Stitch the four pages into one book.
cat >"$OUT/book.typ" <<'TYP'
// The finished book: import each page's `title-page` and render them in order.
#import "00-frontispiece.typ": title-page as frontispiece
#import "01-title.typ": title-page as title-page-main
#import "02-book.typ": title-page as book-title
#import "03-chapter1.typ": title-page as chapter-one
#import "04-section1.typ": title-page as section-one
#import "05-section2.typ": title-page as section-two
#import "06-chapter2.typ": title-page as chapter-two

#frontispiece
#pagebreak()
#title-page-main
#pagebreak()
#book-title
#pagebreak()
#chapter-one
#pagebreak()
#section-one
#pagebreak()
#section-two
#pagebreak()
#chapter-two
TYP

# 6. Compile the complete PDF.
echo "+ typst compile $OUT/book.typ $OUT/book.pdf"
typst compile "$OUT/book.typ" "$OUT/book.pdf" || {
  echo "  ! typst compile failed"
  exit 1
}

pages=$(pdfinfo "$OUT/book.pdf" 2>/dev/null | awk '/Pages/{print $2}' || echo "?")

# 7. The book's COVER / dust jacket — back · spine · front on one wide sheet; the spine width is computed
#    from the page count. Reuses the maritime hand and the illustrate emblem as the front device.
run "$PLAKAT" bookart cover "$COVER" --out "$OUT/cover.typ" --verify

# 8. THE CAPSTONE — assemble the whole typeset book from a Markdown manuscript (bookart book): the emblem
#    title page as the first leaf, chapter openers (rosette headpiece · CHAPTER N · title), raised initials,
#    running heads + folios, a dinkus tailpiece per chapter, and a colophon.
run "$PLAKAT" bookart book "$MANUSCRIPT" --out "$OUT/thebook.typ" --page a5 \
  --title-page "$OUT/01-title.typ" --running-head "The Open Sea" \
  --headpiece "$OUT/rosette-1.png" --tailpiece "$OUT/dinkus.png" \
  --colophon "Set in Libertinus · Printed at the Admiralty Press · MDCCXLI" --verify

echo
echo "✓ done — $OUT/book.pdf ($pages pages: frontispiece · emblem title · framed title · chapter I · §I · §II · chapter II)"
echo "         $OUT/cover.pdf   (dust jacket: back · spine · front)"
echo "         $OUT/thebook.pdf (the assembled typeset book: title · chapters · folios · colophon)"
echo "   kit set + contact sheet → $OUT/kit/    rosette → rosette-1.png    subchapter mark → dinkus.png"
ls -lh "$OUT"/book.pdf "$OUT"/frontispiece.png "$OUT"/emblem.png "$OUT"/rosette-1.png "$OUT"/dinkus.png 2>/dev/null | awk '{print "   "$5"\t"$9}'
