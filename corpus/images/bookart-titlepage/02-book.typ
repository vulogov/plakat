// ─────────────────────────────────────────────────────────────────────
// plakat bookart — an old-style (letterpress) TITLE PAGE, compilable to PDF.
//   • hierarchical centred type: series / title / subtitle / part / author / note / imprint
//   • `title-page` is reusable: compile this file directly, or #import it into your book.
// Preview:  typst compile <this-file>.typ
// In a book: #import "<this-file>.typ": title-page   then   #title-page
// ─────────────────────────────────────────────────────────────────────

#let page-width  = 148mm
#let page-height = 210mm
#let border-image = "00_border.png"

#let title-page = {
  set page(
    width: page-width, height: page-height,
    margin: (top: 61.58mm, bottom: 63.23mm, left: 41.84mm, right: 39.28mm),
    background: place(top + left, dx: 12mm, dy: 12mm, image(border-image, width: page-width - 12mm - 12mm, height: page-height - 12mm - 12mm, fit: "stretch")),
  )
  set text(size: 12pt)
  set par(leading: 0.7em, justify: false)
  set align(center)

  text(size: 18pt, weight: "bold", tracking: 0.02em)[#upper("VOYAGES")]
  v(0.5em)
  text(size: 18pt, weight: "bold", tracking: 0.02em)[#upper("& TRAVELS")]
  v(0.5em)
  v(0.25em)
  line(length: 26%, stroke: 0.5pt + black)
  v(0.4em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("An Account of divers Discoveries and Adventures upon the High Seas.")]
  v(0.6em)
  text(size: 15pt, weight: "regular", tracking: 0.05em)[#"By Captain James Hawkins"]
  v(0.6em)

  v(1fr)
  text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("LONDON · MDCCXLI")]
  v(0.25em)
}

// Preview — compile this file directly; #import takes only `title-page`.
#title-page
