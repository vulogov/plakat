// ─────────────────────────────────────────────────────────────────────
// plakat bookart — a book COVER / dust jacket, laid flat (back · spine · front), compilable.
//   trim 148×210 mm · spine 19.2 mm · · total 315×210 mm
//   `cover` is reusable: compile this file directly, or #import it.
// ─────────────────────────────────────────────────────────────────────

#let cover = {
  set page(width: 315.2mm, height: 210mm, margin: 0mm)
  place(top + left, dx: 148mm, dy: 0mm, line(length: 210mm, angle: 90deg, stroke: (paint: luma(60%), thickness: 0.3pt, dash: "dashed")))
  place(top + left, dx: 167.2mm, dy: 0mm, line(length: 210mm, angle: 90deg, stroke: (paint: luma(60%), thickness: 0.3pt, dash: "dashed")))
  place(top + left, dx: 0mm, dy: 0mm, box(width: 148mm, height: 210mm)[
    #box(width: 148mm, height: 210mm, inset: 14mm)[#{
      set text(size: 12pt)
      set par(leading: 0.7em, justify: false)
      set align(center)
      v(1fr)
      text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("An account of divers discoveries and adventures upon the high seas,")]
      v(0.15em)
      text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("wherein the Author sets sail from Plymouth, describes his vessel and")]
      v(0.15em)
      text(size: 10pt, weight: "regular", tracking: 0.02em)[#smallcaps("crew, and encounters the first of many perils.")]
      v(0.6em)
      v(1.2em)
      text(size: 11pt, weight: "regular", style: "italic")[#"A tall ship, and a star to steer her by."]
      v(0.6em)
      v(1fr)
      text(size: 11pt, weight: "regular", tracking: 0.03em)[#smallcaps("London · The Admiralty Press · MDCCXLI")]
      v(0.25em)
      v(1fr)
    }]
  ])
  place(top + left, dx: 167.2mm, dy: 0mm, box(width: 148mm, height: 210mm)[
    #box(width: 148mm, height: 210mm, inset: 14mm)[#{
      set text(size: 12pt)
      set par(leading: 0.7em, justify: false)
      set align(center)
      v(1fr)
      text(size: 13pt, weight: "regular", tracking: 0.06em)[#smallcaps("The Mariner's Library")]
      v(0.7em)
      v(0.25em)
      line(length: 26%, stroke: 0.5pt + black)
      v(0.4em)
      text(size: 30pt, weight: "bold", tracking: 0.02em)[#upper("The Open Sea")]
      v(0.5em)
      text(size: 17pt, weight: "regular", tracking: 0.03em)[#upper("A Voyage in Four Winds")]
      v(0.6em)
      v(0.8em)
      v(0.3em)
      image("emblem_crop.png", width: 34mm)
      v(0.3em)
      v(0.8em)
      text(size: 15pt, weight: "regular", tracking: 0.05em)[#"By Captain James Hawkins, R.N."]
      v(0.6em)
      v(1fr)
    }]
  ])
  place(top + left, dx: 148mm, dy: 0mm, box(width: 19.2mm, height: 210mm)[
    #set align(center + horizon)
    #rotate(90deg, reflow: true, box(width: 186mm)[#{
      set text(size: 12pt)
      set par(leading: 0.7em, justify: false)
      set align(center)
      text(size: 13pt, weight: "bold", tracking: 0.02em)[#upper("The Open Sea")]
      v(0.5em)
      v(0.5em)
      text(size: 10pt, weight: "regular", tracking: 0.05em)[#"Hawkins"]
      v(0.6em)
    }])
  ])
}

// Preview — compile this file directly; #import takes only `cover`.
#cover
