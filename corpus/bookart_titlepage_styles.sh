#!/usr/bin/env bash
#
# plakat bookart title-page — STYLE SAMPLER.
#
# Renders ONE neutral title spec (corpus/bookart_titlepage_sampler.hjson) in every title-page style, so you
# can compare the hands side by side, and stitches them into a four-page `styles.pdf`:
#
#   letterpress — dense antique book type: bold upper display, small-caps series, fine rules (the default).
#   engraved    — copperplate / atlas: regular-weight wide-tracked caps, italic subtitle & author, airy.
#   modern      — minimal contemporary: regular weight, as-authored case, tiny wide-tracked labels, air.
#   playbill    — Victorian poster: everything heavy bold upper, big size jumps, tight leading, thick rules.
#
# Every page is `--fit` (shrunk to one sheet, measured by typst). All weight-free — no GPU.
#
# Usage:   corpus/bookart_titlepage_styles.sh
set -u

PLAKAT="${PLAKAT:-./target/debug/plakat}"
OUT="corpus/images/bookart-titlepage/styles"
SPEC="corpus/bookart_titlepage_sampler.hjson"
export PLAKAT_OOM_GUARD_GB="${PLAKAT_OOM_GUARD_GB:-0}"

command -v typst >/dev/null || { echo "error: typst not found on PATH"; exit 1; }
[ -x "$PLAKAT" ] || { echo "error: plakat binary not found at $PLAKAT (set PLAKAT=…)"; exit 1; }

mkdir -p "$OUT"
run() { echo "+ $*"; "$@" || { echo "  ! step failed"; exit 1; }; echo; }

echo "============ plakat bookart title-page — STYLE SAMPLER ============"
echo "binary : $PLAKAT   spec: $SPEC   out: $OUT/"
echo

STYLES=(letterpress engraved modern playbill)
i=0
for s in "${STYLES[@]}"; do
  run "$PLAKAT" bookart title-page "$SPEC" --style "$s" --out "$OUT/$(printf '%02d' "$i")-$s.typ" --fit --verify
  i=$((i + 1))
done

# Stitch the four styles into one sampler.
{
  echo "// A style sampler: the SAME spec, four hands."
  i=0
  for s in "${STYLES[@]}"; do
    echo "#import \"$(printf '%02d' "$i")-$s.typ\": title-page as style-$s"
    i=$((i + 1))
  done
  echo
  i=0
  for s in "${STYLES[@]}"; do
    [ "$i" -gt 0 ] && echo "#pagebreak()"
    echo "#style-$s"
    i=$((i + 1))
  done
} >"$OUT/styles.typ"

echo "+ typst compile $OUT/styles.typ $OUT/styles.pdf"
typst compile "$OUT/styles.typ" "$OUT/styles.pdf" || { echo "  ! typst compile failed"; exit 1; }

pages=$(pdfinfo "$OUT/styles.pdf" 2>/dev/null | awk '/Pages/{print $2}' || echo "?")
echo
echo "✓ done — $OUT/styles.pdf ($pages pages: letterpress · engraved · modern · playbill)"
