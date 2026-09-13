#import "../design.typ": *

#pagebreak(weak: true)
#v(1.4cm)
#text(font: body_family, size: 9pt, tracking: 2pt, fill: ink_gray, upper("Afterword"))
#v(2mm)
#text(font: body_family, size: 25pt, weight: "bold", fill: ink_accent)[About This Book]
#v(6mm)
#line(length: 100%, stroke: 0.5pt + ink_rule)
#v(8mm)

#dropcap("T")his book followed one poster because that is how the craft is actually
learned — not by reading what every flag does, but by watching an image become
itself, and reaching for each tool at the moment it was needed. NIGHT MARKET was
never the point. The point was the *order*: install before you generate, write
before you render, analyze before you spend, subtract before you add, edit before
you re-roll, and finish before you ship. Learn the order and the flags follow.

If there is a single idea to carry out of these pages, it is the one the first
chapter opened with and the last one closed on: *you write in prose, and everything
else serves the prose or the pixels.* plakat is a large studio of specialised
tools — a compiler, a critic, a composer, a retoucher, an enlarger, a signer — and
the way to keep them from overwhelming you is to keep them in their places. The
prose is yours. The scenario is output. The corpus remembers why. Each render is a
bet you place only after the odds look good.

#section("Where to go next")

This book is the journey; the reference is the map. Every command here has a
`--help` with the full set of flags, and plakat's RFCs document the design behind
the bigger features — persona, control-generate, etch, naturalize, texture, comic —
when you want the reasoning, not just the lever. Two commands are worth keeping at
your fingertips as you strike out on your own poster:

#screen(caption: "Your two companions")[```
  $ plakat doctor --capability      # what can this machine make?
  $ plakat compile prompts.txt --analyze   # will this scene render?
```]

The first keeps your ambitions matched to your hardware. The second keeps your
evenings spent making instead of waiting. Between them, they are most of the
discipline this book tried to teach.

#section("Colophon")

Typeset with Typst in Libertinus Serif, with terminal mockups in DejaVu Sans Mono.
The screens are faithful to plakat as of #book_version; commands are real and
current, though a model's exact wording and timings will vary with your hardware and
your prose. The running example, the town of the night market and its vendor, is
invented — a scaffold for the features, not a place you can visit.

#v(1.2cm)
#brand_mark(width: 26mm)
#v(6mm)
#align(center)[
  #text(font: body_family, size: 11pt, style: "italic", fill: ink_gray)[
    Now go make the one you have been imagining.
  ]
  #v(3mm)
  #text(font: body_family, size: 10pt, fill: ink_smoke, book_author)
]
