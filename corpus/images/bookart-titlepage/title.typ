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

  text(size: 13pt, weight: "regular", tracking: 0.06em)[#smallcaps("THE")]
  v(0.7em)
  text(size: 30pt, weight: "bold", tracking: 0.02em)[#upper("VOYAGES")]
  v(0.15em)
  text(size: 30pt, weight: "bold", tracking: 0.02em)[#upper("&")]
  v(0.15em)
  text(size: 30pt, weight: "bold", tracking: 0.02em)[#upper("TRAVELS")]
  v(0.5em)
  text(size: 17pt, weight: "regular", tracking: 0.03em)[#upper("OF THE CELEBRATED MARINERS")]
  v(0.6em)
  v(0.25em)
  line(length: 26%, stroke: 0.5pt + black)
  v(0.4em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("Containing a full and particular Account of divers")]
  v(0.15em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("Discoveries, Shipwrecks, and Adventures upon the")]
  v(0.15em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("High Seas and in Foreign Lands.")]
  v(0.6em)
  v(0.5em)
  text(size: 15pt, weight: "regular", tracking: 0.05em)[#"By Captain James Hawkins"]
  v(0.6em)

  v(1fr)
  text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("LONDON")]
  v(0.15em)
  text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("Printed for J. Fletcher, at the Sign of the Anchor")]
  v(0.15em)
  text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("MDCCXLI")]
  v(0.25em)
}

// Preview — compile this file directly; #import takes only `title-page`.
#title-page
