#import "../design.typ": *

#v(1cm)
#text(font: body_family, size: 22pt, weight: "bold")[Following One Poster]
#v(6mm)
#line(length: 100%, stroke: 0.5pt + ink_rule)
#v(8mm)

#figure_img("assets/10-cover-hero.png", "NIGHT MARKET — the poster this book builds, one command at a time.", width: 62%)

plakat is a large program. It generates images from text, transfers styles,
places synthetic people, swaps faces, builds comics and textures and maps and
fractals, upscales, relights, removes objects, and more — each with its own
flags and its own reference page. Read the `--help` output end to end and you
will know *what every lever does* and still not know *which lever to pull first*.

This book does the opposite of a reference. It picks *one image* — a promotional
poster for a fictional weekend event — and follows it from an empty project on a
Tuesday evening to a finished, upscaled, print-ready file, reaching for each
feature exactly when a real designer would, and no sooner. By the last page you
will have watched the poster grow from a single lantern in the fog to a
two-figure scene with a hand-drawn look, and you will know where every tool sits
in the arc of making one picture.

If you learn best by watching something get built, start here. You can read this
book beside the reference (it names the relevant command whenever it uses one in
passing) or entirely on its own.

#section("The poster we are making")

Our example is deliberately modest — big enough to exercise a scene, a small
cast, and a distinct look; small enough to finish inside one volume.

#term("NIGHT MARKET")[
  A poster for a fictional weekend event. A lantern-lit cobbled lane at night: a
  food vendor works a wooden cart while a single customer leans across the counter
  to order. String lights overhead, cool fog beyond, a warm amber glow, and the
  words #smallcaps[Night Market] across the top. A small, human scene — one hero,
  one bystander, one prop, and a mood.
]

We chose a scene with *people, a prop, and a look* on purpose, because that is
what exercises the most of plakat: a composition to keep coherent, two figures
who must each stand where they are told, an object the model will try to melt
into a person, and a treatment that has to survive from draft to print. A single
still-life would teach you a third of the program. If you only ever make
single-subject images, the middle chapters will still show you the safety rails —
you just won't need to climb as high.

#section("The one idea to take away")

If you remember nothing else from this book, remember this: in plakat you write
in *prose*, and everything else serves the prose or the pixels.

#term("Prose")[
  A plain-language description of your scene in a `prompts.txt` file — sentences,
  not settings. plakat's `compile` command turns prose into a runnable *scenario*;
  the analyze and fix commands check and repair the prose before you spend a
  single render on it. You edit the prose. You almost never hand-edit the machine
  output.
]

Most of the frustration people feel with image models comes from editing the
wrong thing — poking at a raw prompt string, re-rolling seeds, waiting three
minutes to discover the scene was impossible from the start. plakat's answer is a
short *authoring loop* that keeps you in prose and tells you what is wrong before
it costs you anything.

#pipeline_arc()

#section("The shape of the journey")

#screen(caption: "From blank project to finished print")[```
  I    Setting Up ............. install · your first image · the prose project
  II   Authoring in Prose ..... write the scene · will it render? · fix it
  III  Compiling & Running .... compile a scenario · pick a model · tune
  IV   Composition & Control .. add figures · a cart & a relation · style
  V    Editing & Finishing .... edit · naturalize · upscale · export
  VI   Reference .............. directives · model aliases · the polish loop
```]

Each part is a stage every image passes through. You will not use every plakat
feature to make *your* poster — few images need all of them — but by the end of
this one you will have seen where each fits, and be able to reach for the right
one at the right time.

#callout(label: "How to read the screens")[
  plakat is a terminal program, so the book teaches with faithful monospace
  *screen* mockups like the one above, rather than screenshots. What you type is
  shown after a `$` prompt; what plakat prints back is shown plainly. Commands are
  real and current as of #book_version — type them and they run.
]

#section("What you need")

A computer that can run a diffusion model. plakat is pure Rust and runs on Apple
Silicon (Metal), NVIDIA (CUDA), or CPU — slowly, but it *will* finish on CPU. The
opening chapter checks your machine and downloads the one small model our first
image needs. You do not need an API key to begin: the default prompt enhancer is
a small model that runs on your own machine. If you have a key for a hosted model
later, the book shows where it slots in.

#recap((
  [This book follows *one poster*, "NIGHT MARKET", from `plakat init` to a
  finished print — learning by watching one image get made, feature by feature in
  the order you would actually reach for them.],
  [The central idea: you author in *prose*; `compile` turns prose into a runnable
  scenario, and the analyze/fix loop repairs the prose *before* a render costs you
  time.],
  [The journey runs in six parts — setup, authoring, compiling & running,
  composition & control, editing & finishing, and a short reference — each a stage
  every image passes through.],
  [You need only a machine that runs a diffusion model and no API key to start;
  the book climbs the "control ladder" only as far as the poster needs.],
))
