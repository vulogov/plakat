// plakat — A Poster, Start to Finish : design tokens + page chrome.
//
// Adapted from Book/START_TO_FINISH/design.typ of the Inkhaven library
// (self-contained, built on the fonts Typst bundles). The app IS a terminal, so
// the book teaches mostly with monospace `screen()` mockups, plus a few fletcher
// diagrams for the pipeline pictures.

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let book_title    = "A Poster, Start to Finish"
#let book_subtitle = "One Image from Blank Project to Finished Print"
#let book_author   = "Vladimir Ulogov"
#let book_year     = "2026"
#let book_version  = "plakat 6.30.0"

// The project logo (transparent B/W), used as a brand mark on the cover, part
// dividers, and colophon. Path is relative to this design.typ (Typst resolves
// image() paths relative to the file the call lives in — here, design.typ).
#let plakat_logo   = "assets/plakat_logo_transparent.png"

// ── Palette — warm paper, cool ink, restrained accents ──────────────
#let ink_black   = rgb("#1a1a1a")
#let ink_gray    = rgb("#5d5d5d")
#let ink_faint   = rgb("#9a9a9a")
#let ink_rule    = rgb("#c6c0b5")
#let ink_accent  = rgb("#7a3b2f")            // burnt sienna — chapter numbers (paint = warm)
#let ink_smoke   = rgb("#7d736a")            // muted brown — cover eyebrow + screen bar
#let ink_paper   = rgb("#fdfaf3")            // warm cream — cover ground
#let ink_term    = rgb("#7a3b2f")            // sienna — term definitions
#let ink_code_bg = rgb("#f3eee4")
#let ink_call_bg = rgb("#f6f1e6")
#let ink_term_bg = rgb("#f6ece6")
#let ink_recap   = rgb("#3f6b4a")            // muted green — recap accent
#let ink_recap_bg = rgb("#e9f3ea")           // pastel mint — "what you learned"
#let ink_warn    = rgb("#8a6d1f")            // amber — the "burning tokens" warnings
#let ink_warn_bg = rgb("#f6f0df")

// Bundled families only (no host-font setup, no warnings).
#let body_family = ("Libertinus Serif", "New Computer Modern")
#let sans_family = ("Libertinus Serif", "New Computer Modern")
#let mono_family = ("DejaVu Sans Mono",)            // bundled with Typst

#let book_page = (
  paper: "iso-b5",
  margin: (inside: 26mm, outside: 20mm, top: 22mm, bottom: 24mm),
  numbering: "1",
)

// ── Part divider ────────────────────────────────────────────────────
#let part(number: "I", title: "") = {
  pagebreak(weak: true)
  hide(heading(level: 1, numbering: none, outlined: true, bookmarked: true, [Part #number — #title]))
  v(6cm)
  align(center)[
    #image(plakat_logo, width: 16mm)
    #v(7mm)
    #text(font: body_family, size: 11pt, tracking: 3pt, fill: ink_gray, upper("Part " + number))
    #v(6mm)
    #line(length: 36%, stroke: 0.5pt + ink_rule)
    #v(6mm)
    #text(font: body_family, size: 26pt, weight: "bold", fill: ink_black, title)
  ]
}

// ── Chapter opening ─────────────────────────────────────────────────
#let chapter(number: 0, title: "") = {
  pagebreak(weak: true)
  hide(heading(level: 1, numbering: none, outlined: true, bookmarked: true, [#str(number) — #title]))
  v(1.6cm)
  align(left)[
    #text(font: body_family, size: 9pt, tracking: 2pt, fill: ink_gray, upper("Chapter " + str(number)))
    #v(1mm)
    #text(font: body_family, size: 84pt, weight: "bold", fill: ink_accent, str(number))
    #v(-6mm)
    #text(font: body_family, size: 25pt, weight: "regular", fill: ink_black, title)
  ]
  v(1cm)
  line(length: 100%, stroke: 0.5pt + ink_rule)
  v(8mm)
}

// ── Appendix opening ────────────────────────────────────────────────
#let appendix(letter: "A", title: "") = {
  pagebreak(weak: true)
  hide(heading(level: 1, numbering: none, outlined: true, bookmarked: true, [Appendix #letter — #title]))
  v(1.6cm)
  align(left)[
    #text(font: body_family, size: 9pt, tracking: 2pt, fill: ink_gray, upper("Appendix " + letter))
    #v(1mm)
    #text(font: body_family, size: 84pt, weight: "bold", fill: ink_accent, letter)
    #v(-6mm)
    #text(font: body_family, size: 25pt, weight: "regular", fill: ink_black, title)
  ]
  v(1cm)
  line(length: 100%, stroke: 0.5pt + ink_rule)
  v(8mm)
}

// ── Section / subsection ────────────────────────────────────────────
#let section(title) = {
  hide(heading(level: 2, numbering: none, outlined: true, title))
  block(
    sticky: true, above: 8mm, below: 3.2mm,
    text(font: body_family, size: 15pt, weight: "bold", fill: ink_black, title),
  )
}
#let subsection(title) = {
  block(
    sticky: true, above: 5.5mm, below: 2.4mm,
    text(font: body_family, size: 11.5pt, weight: "bold", fill: ink_black, title),
  )
}

// ── Term box — DEFINE a term ─────────────────────────────────────────
#let term(name, body) = {
  v(2mm)
  block(
    fill: ink_term_bg, stroke: (left: 2pt + ink_term),
    inset: (left: 9pt, right: 9pt, top: 7pt, bottom: 7pt),
    width: 100%, radius: 1pt, breakable: false,
    {
      text(font: body_family, size: 8pt, weight: "bold", fill: ink_term, tracking: 1pt, "TERM")
      h(6pt)
      text(font: body_family, size: 11pt, weight: "bold", fill: ink_term, name)
      v(2mm)
      body
    },
  )
  v(2mm)
}

// ── Note / tip callout ──────────────────────────────────────────────
#let callout(label: "Note", body) = {
  v(2mm)
  block(
    fill: ink_call_bg, stroke: (left: 2pt + ink_accent),
    inset: (left: 9pt, right: 9pt, top: 7pt, bottom: 7pt),
    width: 100%, radius: 1pt, breakable: false,
    {
      text(font: body_family, size: 8pt, weight: "bold", fill: ink_accent, tracking: 1.5pt, upper(label))
      v(2mm)
      body
    },
  )
  v(2mm)
}

// ── Warning callout — the "you hit the wall" / cost boxes ────────────
#let warn(label: "Watch out", body) = {
  v(2mm)
  block(
    fill: ink_warn_bg, stroke: (left: 2pt + ink_warn),
    inset: (left: 9pt, right: 9pt, top: 7pt, bottom: 7pt),
    width: 100%, radius: 1pt, breakable: false,
    {
      text(font: body_family, size: 8pt, weight: "bold", fill: ink_warn, tracking: 1.5pt, upper(label))
      v(2mm)
      body
    },
  )
  v(2mm)
}

// ── Two-path callout — the same task the quick way and the controlled
//    way, side by side. Used to keep both audiences in view. ──────────
#let two_col(left_label, right_label, left, right) = {
  v(2mm)
  block(breakable: false, width: 100%, grid(
    columns: (1fr, 1fr), gutter: 5mm,
    block(
      fill: ink_call_bg, stroke: (left: 2pt + ink_accent),
      inset: 8pt, width: 100%, radius: 1pt, breakable: false,
      { text(font: body_family, size: 8pt, weight: "bold", fill: ink_accent, tracking: 1pt, upper(left_label)); v(2mm); left },
    ),
    block(
      fill: ink_recap_bg, stroke: (left: 2pt + ink_recap),
      inset: 8pt, width: 100%, radius: 1pt, breakable: false,
      { text(font: body_family, size: 8pt, weight: "bold", fill: ink_recap, tracking: 1pt, upper(right_label)); v(2mm); right },
    ),
  ))
  v(2mm)
}

// ── Chapter-end recap ───────────────────────────────────────────────
#let recap(items) = {
  v(7mm)
  block(
    fill: ink_recap_bg, stroke: (left: 2pt + ink_recap),
    inset: (left: 9pt, right: 9pt, top: 8pt, bottom: 8pt),
    width: 100%, radius: 1pt, breakable: false,
    {
      text(font: body_family, size: 9pt, weight: "bold", fill: ink_recap, tracking: 1.5pt, "WHAT YOU LEARNED")
      v(2mm)
      list(..items)
    },
  )
}

// ── Terminal screen — a faithful monospace rendering of a CLI / TUI
//    screen. `body` is a raw block; `caption` names it. ────────────────
#let screen(caption: "", body) = {
  v(2mm)
  block(breakable: false, width: 100%, {
    block(
      fill: ink_smoke,
      inset: (left: 8pt, right: 8pt, top: 3pt, bottom: 3pt),
      width: 100%,
      radius: (top-left: 2pt, top-right: 2pt),
      {
        text(font: mono_family, size: 8pt, fill: ink_paper, "● ● ●")
        h(6pt)
        text(font: body_family, size: 8.5pt, style: "italic", fill: ink_paper, caption)
      },
    )
    block(
      fill: ink_code_bg,
      stroke: 0.5pt + ink_rule,
      inset: 8pt,
      width: 100%,
      radius: (bottom-left: 2pt, bottom-right: 2pt),
      text(font: mono_family, size: 8.5pt, body),
    )
  })
  v(2mm)
}

// ── Afterword helpers ───────────────────────────────────────────────
#let dropcap(letter) = box(baseline: 0.62em,
  text(font: body_family, size: 2.7em, weight: "bold", fill: ink_accent, letter))

#let chord_row(name, desc) = (name, desc)
#let chord_table(rows) = block(width: 100%, {
  for (name, desc) in rows {
    block(breakable: false, width: 100%, {
      grid(columns: (34mm, 1fr), gutter: 4mm,
        text(font: mono_family, weight: "bold", size: 9pt, fill: ink_black, name),
        text(font: body_family, size: 10pt, fill: ink_black, desc))
      v(1.6mm)
    })
  }
})

#let figure_note(body) = align(center,
  text(font: body_family, style: "italic", size: 9pt, fill: ink_gray, body))

// ── Brand mark — the plakat logo, centred. image() lives here in design.typ so
//    the design-relative `plakat_logo` path resolves correctly from any chapter. ──
#let brand_mark(width: 22mm) = align(center, image(plakat_logo, width: width))

// ── Rendered-image figure — a framed plakat output with an italic caption. ──
#let figure_img(path, caption, width: 92%) = {
  v(3mm)
  block(breakable: false, width: 100%, align(center, {
    block(stroke: 0.5pt + ink_rule, radius: 2pt, clip: true, image(path, width: width))
    v(1.6mm)
    text(font: body_family, style: "italic", size: 9pt, fill: ink_gray, caption)
  }))
  v(3mm)
}

// ── Diagram helpers (fletcher) ──────────────────────────────────────
#let dnode(pos, body, fill: ink_call_bg) = node(
  pos, align(center, text(font: body_family, size: 8.5pt, body)),
  stroke: 0.6pt + ink_rule, fill: fill, corner-radius: 2pt, inset: 6pt,
)

// The authoring loop — the spine of the book: prose is the surface you edit.
#let pipeline_arc() = {
  v(3mm)
  align(center, diagram(
    spacing: 8mm,
    dnode((0, 0), [*Prose*\ a `prompts.txt`]),
    dnode((1, 0), [*Analyze*\ will it render?]),
    dnode((2, 0), [*Fix*\ trim & repair]),
    dnode((3, 0), [*Compile*\ a scenario]),
    dnode((4, 0), [*Run*\ the image], fill: ink_recap_bg),
    edge((0, 0), (1, 0), "->"), edge((1, 0), (2, 0), "->"),
    edge((2, 0), (3, 0), "->"), edge((3, 0), (4, 0), "->"),
    edge((2, 0), (1, 0), "->", bend: 40deg, stroke: (dash: "dashed"),
      label: text(size: 7pt, fill: ink_gray, [polish loop])),
  ))
  figure_note[The authoring loop. You stay in prose; analyze and fix tighten it *before* a render burns time.]
  v(3mm)
}

// What one scene passes through inside `compile`.
#let compile_stages() = {
  v(3mm)
  align(center, diagram(
    spacing: (11mm, 4mm),
    dnode((0, 0), [*translate*\ → English]),
    dnode((1, 0), [*compose*\ components]),
    dnode((2, 0), [*enhance*\ family-aware]),
    dnode((3, 0), [*negative*\ auto + seeds]),
    dnode((4, 0), [*fit budget*\ pack to model], fill: ink_recap_bg),
    edge((0, 0), (1, 0), "->"), edge((1, 0), (2, 0), "->"),
    edge((2, 0), (3, 0), "->"), edge((3, 0), (4, 0), "->"),
  ))
  figure_note[Each prose block becomes one scenario task by passing through these stages — `--explain` shows them.]
  v(3mm)
}

// The model-free budget pack — what survives, what is cut, and why.
#let budget_flow() = {
  v(3mm)
  align(center, diagram(
    spacing: (13mm, 4mm),
    dnode((0, 0.5), [*Over-budget\ prompt*]),
    dnode((1, 0), [*Score spans*\ weight · position\ · filler penalty]),
    dnode((1, 1), [*Keep the\ subject*\ always]),
    dnode((2, 0.5), [*Fit the model*\ drop filler first,\ record what & why], fill: ink_recap_bg),
    edge((0, 0.5), (1, 0), "->"), edge((0, 0.5), (1, 1), "->"),
    edge((1, 0), (2, 0.5), "->"), edge((1, 1), (2, 0.5), "->"),
  ))
  figure_note[Model-free budget packing: a trim you can audit, not a silent truncation. The LLM reword is the fallback.]
  v(3mm)
}

// Layers of control — how much you hand plakat vs. how much you pin.
#let control_ladder() = {
  v(3mm)
  align(center, diagram(
    spacing: (0mm, 5mm),
    dnode((0, 0), [*Pinned* — poses, contacts, regions, personas], fill: ink_recap_bg),
    dnode((0, 1), [*Guided* — control-generate wireframe + regional draft]),
    dnode((0, 2), [*Shaped* — foreground / objects / relate in prose]),
    dnode((0, 3), [*Free* — one prose sentence, let the model decide], fill: ink_code_bg),
    edge((0, 3), (0, 2), "->"), edge((0, 2), (0, 1), "->"), edge((0, 1), (0, 0), "->"),
    edge((1.25, 3.2), (1.25, -0.2), "->", stroke: 1pt + ink_accent,
      label: text(font: body_family, size: 8pt, fill: ink_accent, [more\ control]),
      label-side: right),
  ))
  figure_note[The control ladder. Start free; climb only as far as the image needs — every rung costs effort.]
  v(3mm)
}

// ── Master document wrapper ─────────────────────────────────────────
#let book(pages) = {
  set document(title: book_title, author: book_author)
  set text(font: body_family, size: 11pt, fill: ink_black, lang: "en")
  set par(leading: 0.72em, justify: true, first-line-indent: 1em)
  show raw.where(block: true): it => block(
    fill: ink_code_bg, stroke: 0.5pt + ink_rule, inset: 7pt, radius: 2pt, width: 100%,
    breakable: false,
    text(font: mono_family, size: 7.5pt, it),
  )
  show raw.where(block: false): it => box(
    fill: ink_code_bg, inset: (x: 2pt, y: 0pt), outset: (y: 2pt), radius: 1pt,
    text(font: mono_family, size: 9.5pt, it),
  )

  // ── Cover ──
  set page(paper: book_page.paper, margin: 0pt, numbering: none, header: none, fill: ink_paper)
  block(width: 100%, height: 100%)[
    #place(top + left, dx: 12mm, dy: 12mm,
      rect(width: 100% - 24mm, height: 100% - 24mm, stroke: 1pt + ink_accent))
    #place(top + left, dx: 14mm, dy: 14mm,
      rect(width: 100% - 28mm, height: 100% - 28mm, stroke: 0.4pt + ink_accent))
    #place(top + center, dy: 34mm, {
      let dot(dx, r) = place(top + center, dx: dx, dy: 0pt, circle(radius: r, fill: ink_accent))
      dot(-18mm, 1.6mm); dot(-9mm, 1.1mm); dot(0mm, 2.2mm); dot(9mm, 1.1mm); dot(18mm, 1.6mm)
    })
    #place(top + center, dy: 60mm, block(width: 74%)[
      #set par(justify: false)
      #align(center)[
        #text(font: body_family, size: 12pt, tracking: 4pt, fill: ink_smoke, upper("One Image, End to End"))
        #v(11mm)
        #text(font: body_family, size: 30pt, weight: "bold", fill: ink_black, book_title)
        #v(6mm)
        #line(length: 55%, stroke: 0.6pt + ink_accent)
        #v(6mm)
        #text(font: body_family, size: 12.5pt, style: "italic", fill: ink_smoke, book_subtitle)
      ]
    ])
    #place(bottom + center, dy: -46mm, image(plakat_logo, width: 22mm))
    #place(bottom + center, dy: -30mm, align(center)[
      #text(font: body_family, size: 10pt, fill: ink_smoke, book_author)
      #v(2mm)
      #text(font: body_family, size: 9pt, fill: ink_smoke, book_year + " · examples assume " + book_version + " or newer")
    ])
  ]
  pagebreak()

  // Contents
  set page(margin: book_page.margin, fill: white)
  text(font: body_family, size: 22pt, weight: "bold", fill: ink_black, "Contents")
  v(7mm)
  outline(title: none, indent: auto, depth: 2)
  pagebreak()

  // Body
  set page(
    numbering: "1", number-align: center,
    header: context {
      if counter(page).get().first() > 1 {
        align(center, text(font: body_family, size: 8pt, fill: ink_faint, tracking: 1.5pt, upper(book_title)))
      }
    },
  )
  counter(page).update(1)
  for p in pages [ #p ]
}
