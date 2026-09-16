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
    margin: (top: 15mm, bottom: 15mm, left: 15mm, right: 15mm),
  )
  set text(size: 12pt, number-type: "old-style", features: (hlig: 1))
  set par(leading: 0.7em, justify: false)
  set align(center)

  text(size: 13pt, weight: "regular", tracking: 0.06em)[#smallcaps("PUBLISHED FOR THE LORDS COMMISSIONERS")]
  v(0.15em)
  text(size: 13pt, weight: "regular", tracking: 0.06em)[#smallcaps("OF THE ADMIRALTY")]
  v(0.7em)
  v(0.25em)
  line(length: 26%, stroke: 0.5pt + black)
  v(0.4em)
  text(size: 12pt, weight: "regular", tracking: 0.03em)[#upper("AN ACCOUNT OF THE")]
  v(0.6em)
  text(size: 21pt, weight: "bold", tracking: 0.02em)[#upper("PRACTICE OF NAVIGATION")]
  v(0.5em)
  text(size: 12pt, weight: "regular", tracking: 0.03em)[#upper("UPON THE SOUTHERN OCEAN")]
  v(0.6em)
  text(size: 12pt, weight: "bold", tracking: 0.04em)[#upper("PART I · SEAMANSHIP")]
  v(0.5em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("Compiled, by command, from the journals of His Majesty's ships,")]
  v(0.15em)
  text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("and approved the 24th of December 1848,")]
  v(0.6em)
  text(size: 13pt, weight: "regular", tracking: 0.05em)[#"By Captain James Hawkins, R.N."]
  v(0.6em)
  v(0.3em)
  image("emblem_crop.png", width: 24%)
  v(0.3em)

  v(1fr)
  text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("LONDON")]
  v(0.15em)
  text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("Printed at the Admiralty Press")]
  v(0.15em)
  text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("MDCCXLI")]
  v(0.25em)
}

// Preview — compile this file directly; #import takes only `title-page`.
#title-page
