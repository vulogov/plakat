#import "../design.typ": *

#pagebreak(weak: true, to: "odd")

#hide(heading(
  level: 1, numbering: none, outlined: true, bookmarked: true,
  "About the Author",
))

#v(2cm)
#align(left)[
  #text(font: body_family, size: 9pt, tracking: 2pt, fill: ink_gray, upper("Afterword"))
  #v(4mm)
  #text(font: body_family, size: 36pt, weight: "regular", fill: ink_black, "About the author")
]
#v(1cm)
#line(length: 100%, stroke: 0.5pt + ink_rule)
#v(12mm)

#grid(
  columns: (56mm, 1fr),
  gutter: 7mm,
  [
    #image("../assets/author-portrait.png", width: 100%)
    #v(2mm)
    #align(center, text(font: body_family, style: "italic", size: 9pt, fill: ink_gray, "Vladimir Ulogov."))
  ],
  [
    Vladimir Ulogov has spent decades building infrastructure for distributed
    systems — the kind of software that watches other software. Early in his career
    he worked on monitoring and telemetry platforms; later years took him into
    federated observability, telemetry buses, and the architecture of systems that
    have to make sense of millions of data points without losing the thread.

    Observability, in the end, is a discipline of *coherence* — of never reporting a
    state the system cannot account for, of insisting that every signal follow from
    something real. It is not an accident that a studio he built for image-makers
    carries the same instinct into every picture it helps make: never a render the
    prose cannot account for, and never a change whose reason is lost. The analyzer
    that reads a scene before it costs you, and the small provenance record that
    remembers why each edit was made, are observability turned toward art.
  ],
)

#v(4mm)

What makes him slightly unusual in his corner of the industry is a tendency to write
his own tools — not small utilities, but programming languages and formats. The Bund
language (its compiler, its VM, its document store, its parser) lives in a long series
of Rust crates on crates.io. `rust_dynamic`, `rust_multistackvm`, `bundcore` — each is
a building block that exists because the off-the-shelf options didn't fit the shape of
the work. plakat grew the same way: an engine for turning prose into pictures that,
along the way, learned to *reason about* the prose — and even the little knowledge
format it uses to remember its own decisions, smysl, is another of his crates, folded
in where a picture needed a memory.

#section("A work of love")

plakat is open source, under a permissive licence — you can read it, fork it, study
it, modify it, and pass it on. Strictly speaking the licence also lets you sell it;
the author would, gently but firmly, disagree with your doing that. plakat was not
designed as a #emph[for sale] project. It is a work of love made for the people who
can least afford to pay for software — the designer on a battered laptop, the student
making a poster for a club with no budget, the artist who simply wants a tool that
runs on their own machine and answers to no one. It carries no analytics, no
telemetry, no upsell; the binary will never phone home. Your images are rendered
entirely on your own machine, and nothing leaves it unless you deliberately choose to
route the optional prompt-reasoning steps through a hosted language model — a choice you
control, one plakat never makes for you, and one you can decline entirely by keeping
those steps on-device or on a local Ollama model.

#section("A note on cooperation")

Vladimir believes firmly in the human capacity for mutual help — that we make better
work, and live better lives, when we share what we know and what we build. Open source
is one of the most concrete expressions of cooperation our era has produced: code
read, improved, and passed forward without payment, without permission, by people who
will never meet. If this book helps you make one image you are proud of — a poster, a
portrait, a thing you imagined and can now hold — that is enough.

#section("Where to find more")

/ *GitHub*: `@vulogov` — the source for plakat, Bund, smysl, and the dozen-plus Rust crates that carry the infrastructure. Issues and pull requests welcome.
/ *LinkedIn*: `/in/vladimirulogov` — posts on observability, the occasional long-form essay.
/ *YouTube*: `@vulogov` — talks and walkthroughs from the conference trail.

#v(8mm)

#text(font: body_family, style: "italic", size: 11pt, fill: ink_gray,
  "If the tool ever gets in the way of the picture instead of out of it, open an issue on GitHub. The author reads them."
)

#v(2cm)
#align(center, text(font: body_family, size: 8pt, fill: ink_faint, tracking: 4pt,
  upper("end of the book")))
