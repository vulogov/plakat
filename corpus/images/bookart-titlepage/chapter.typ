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

  v(0.3em)
  image("device_crop.png", width: 24mm)
  v(0.3em)
  text(size: 15pt, weight: "bold", tracking: 0.04em)[#upper("CHAPTER I")]
  v(0.5em)
  v(0.25em)
  line(length: 26%, stroke: 0.5pt + black)
  v(0.4em)
  text(size: 30pt, weight: "bold", tracking: 0.02em)[#upper("THE DEPARTURE")]
  v(0.5em)
  text(size: 17pt, weight: "regular", tracking: 0.03em)[#upper("In which our narrative begins")]
  v(0.6em)
  v(1em)
  text(size: 11pt, weight: "regular", style: "italic")[#"The sea, once it casts its spell, holds one in its net of wonder for ever."]
  v(0.6em)
  v(0.5em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("Wherein the Author sets sail from Plymouth, describes his vessel")]
  v(0.15em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("and crew, and encounters the first of many perils.")]
  v(0.6em)
}

// Preview — compile this file directly; #import takes only `title-page`.
#title-page
