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
  image("frontispiece_crop.png", width: 60%)
  v(0.3em)
  v(0.5em)
  text(size: 11pt, weight: "regular", style: "italic")[#"A tall ship, and a star to steer her by."]
  v(0.6em)
}

// Preview — compile this file directly; #import takes only `title-page`.
#title-page
