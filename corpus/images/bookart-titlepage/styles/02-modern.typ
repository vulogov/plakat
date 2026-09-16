// ─────────────────────────────────────────────────────────────────────
// plakat bookart — an old-style (letterpress) TITLE PAGE, compilable to PDF.
//   • hierarchical centred type: series / title / subtitle / part / author / note / imprint
//   • `title-page` is reusable: compile this file directly, or #import it into your book.
// Preview:  typst compile <this-file>.typ
// In a book: #import "<this-file>.typ": title-page   then   #title-page
// ─────────────────────────────────────────────────────────────────────

#let page-width  = 148mm
#let page-height = 210mm

#let title-page = {
  set page(
    width: page-width, height: page-height,
    margin: (top: 22mm, bottom: 22mm, left: 22mm, right: 22mm),
  )
  set text(size: 12pt)
  set par(leading: 0.7em, justify: false)
  set align(center)

  text(size: 10pt, weight: "regular", tracking: 0.25em)[#upper("The Mariner's Library")]
  v(1em)
  v(0.25em)
  line(length: 26%, stroke: 0.4pt + black)
  v(0.4em)
  text(size: 28pt, weight: "regular")[#"The Open Sea"]
  v(0.8em)
  text(size: 14pt, weight: "regular")[#"A Voyage in Four Winds"]
  v(0.7em)
  text(size: 11pt, weight: "regular", tracking: 0.2em)[#upper("Volume the First")]
  v(0.7em)
  text(size: 13pt, weight: "regular")[#"By a Gentleman of Devon"]
  v(0.8em)
  v(0.5em)
  text(size: 11pt, weight: "regular", style: "italic")[#"They that go down to the sea in ships, that do business in great waters."]
  v(0.7em)

  v(1fr)
  text(size: 9pt, weight: "regular", tracking: 0.2em)[#upper("London · MDCCC")]
  v(0.4em)
}

// Preview — compile this file directly; #import takes only `title-page`.
#title-page
