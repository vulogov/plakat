#import "../design.typ": *
#chapter(number: 15, title: "The Title, the Signature, and the Export")

#dropcap("A") finished picture is not yet a finished *poster*. A poster has a title
across the top; it carries proof of where it came from and a recipe for remaking it;
and it ships in the right file for wherever it is going. This last chapter of the
making adds the type, signs the work, and sends it out — and it is honest about the
one thing plakat should *not* be asked to do.

#section("The title: type belongs in a layout, not a prompt")

You may be tempted to ask the diffusion model for the words #smallcaps[Night Market]
across the sky. Don't. Diffusion models are weak at rendered text — they produce
convincing *shapes* of letters that spell nothing, or "NIGTH MARKPT," and no amount
of prompting reliably fixes it. The professional move is the one a real poster
designer makes: render the *image* with plakat, and set the *type* in a layout step,
where a font is a font.

plakat's `compose` command is that layout step. It stacks image layers — a
background, placed cut-outs, a title bar — with z-order, position, scale, and
opacity, on the CPU, no diffusion involved.

#term("compose")[
  A layered-composition command: `plakat compose layout.hjson` stacks image assets —
  a rendered background, cut-out subjects, a title graphic — with per-layer position,
  scale, z-order, and opacity. No model runs; it assembles finished pixels. It is
  where poster *type* and any hand-made graphic elements go, kept separate from the
  diffusion render.
]

#screen(caption: "Compose the poster: art + title")[```
  # layout.hjson
  {
    size: 4096x5600
    layers: [
      { image: out/market-print.png  z: 0  fit: cover }
      { image: assets/title-night-market.png  z: 1
        x: 0.5  y: 0.08  scale: 0.8  anchor: top-center }
    ]
  }

  $ plakat compose layout.hjson --out out/poster.png
    ✓ 2 layers → out/poster.png
```]

#figure_img("assets/09-poster-portrait.png", "Image 09 — the portrait poster crop, dark sky held clear at the top where the title type will be set in the layout step.", width: 58%)

The title graphic itself — the words #smallcaps[Night Market] in a chosen typeface —
you make in whatever tool sets type well (including plakat's own `bookart`, which
draws clean glyph-based lettering for exactly this). The point is separation: the
model paints the scene; type is placed as type. Your poster gets crisp, correct
letters, every time.

#callout(label: "Why this is the honest chapter")[
  A book that promised "just prompt the model for your title" would be selling you a
  frustration. plakat is a studio, not a single magic box: it renders the picture
  superbly and hands the lettering to a layout step that does *that* job well. Using
  each tool for what it is good at is the whole philosophy of the program.
]

#section("Signing the work: provenance etching")

A poster that leaves your machine should carry proof it came from you. plakat can
*etch* a provenance mark into any image it produces — an invisible, multi-layer
identifier that survives ordinary handling — with a single global flag.

#screen(caption: "Etch provenance on the way out")[```
  $ plakat compose layout.hjson --out out/poster.png --etch
    ↳ etched EtchId  (manifest · pixel · fingerprint layers)
```]

#term("Provenance etching (`--etch`)")[
  An opt-in mark written into images plakat creates: a 64-bit identifier carried
  across several layers (a metadata manifest, a pixel-domain watermark, a perceptual
  fingerprint). `plakat doctor --if-plakat IMAGE` reads the surviving layers back into
  a graded verdict on whether an image originated from plakat. Honest about its
  limits — it is not removal-proof, and absence of a mark is not proof of forgery.
]

Later — on your own image or someone else's — you can ask whether it came from
plakat, and get a graded answer rather than a guess.

#screen(caption: "Verify an image's origin")[```
  $ plakat doctor --if-plakat out/poster.png
    verdict: LIKELY plakat  (manifest ✓ · fingerprint ✓)
```]

#section("The recipe travels with the image")

Every plakat render already carries its own recipe — the metadata sidecar and the
in-PNG parameters block from Chapter 1. At the end of a project that pays off twice.
`metadata` reads a poster's settings back out of the file; `clone` turns them into a
runnable command, so months later you can reproduce or fork the exact render.

#screen(caption: "Recover how it was made")[```
  $ plakat metadata out/market-print.png
    model sdxl · seed 1000 · 30 steps · euler-a · 1024² · loras: ink-poster-xl
  $ plakat clone out/market-print.png
    plakat generate "night market lane, …" --model sdxl --seed 1000 \
      --steps 30 --scheduler euler-a --loras ink-poster-xl:0.65
```]

For a *series* of posters, `gallery` builds a browsable Markdown index of a whole
folder — each thumbnail with its prompt and settings — from the metadata alone. It is
the reproducible contact sheet for a body of work.

#screen(caption: "Index the series")[```
  $ plakat gallery out/ --out out/index.md
    ✓ 6 images · thumbnails + per-image recipe → out/index.md
```]

#section("Choosing the file")

Last, the container. PNG is the default and the right choice for a master and for
anything that will be re-edited or uploaded — it carries the parameters block that
other tools read. WebP ships perceptually-equivalent files about a third smaller for
the web, at the cost of that embedded block (the JSON sidecar still travels alongside,
so `metadata` and `clone` keep working). Master in PNG; export a WebP for the web when
size matters.

#section("Start to finish")

Trace the whole arc. We installed plakat and made a single lantern in the fog. We
built a prose project and wrote the scene in sentences. We *analyzed* it before
spending a render, *fixed* its safe problems, and let the smysl corpus remember why.
We compiled, chose SDXL, and tuned the frame. We added a second figure, separated the
cart, related the two people, dressed the image in a look, and pinned the vendor's
face for a series. We edited the last flaws, naturalized away the machine sheen,
enlarged to print, set the title in a proper layout, signed the work, and exported
it. One poster, start to finish — and every command you reached for, reached for in
the order the work asked.

That is the whole of it. What follows is reference: the directives, the models, and a
one-page map of the polish loop, for when you are making your *own* poster and want
the lever without the story.

#recap((
  [Set poster *type* in a layout, not a prompt: diffusion models garble text, so
  render the image with plakat and place the title as its own layer with `compose`.],
  [`--etch` signs an image with an invisible, multi-layer provenance mark; `doctor
  --if-plakat` reads it back into a graded origin verdict (honest about its limits).],
  [Every render carries its recipe: `metadata` reads it back, `clone` turns it into a
  runnable command, and `gallery` indexes a whole series from the metadata alone.],
  [Master in *PNG* (it carries the parameters block); export *WebP* for the web when
  file size matters — the JSON sidecar keeps `metadata`/`clone` working either way.],
  [The arc complete: install → prose → analyze → fix → compile → tune → compose →
  style → edit → naturalize → upscale → title → sign → export — one poster, each tool
  used for what it does best.],
))
