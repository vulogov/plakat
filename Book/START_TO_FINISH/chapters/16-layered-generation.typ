#import "../design.typ": *
#chapter(number: 16, title: "Layered Generation")

#dropcap("Y") ou have taken one poster from a blank project to a signed print, and along
the way you met the composition tools the compile path gives you: `foreground` to name
the figures, `objects:` to keep a cart from melting into its vendor, `relate:` to make
two people face each other. Those staged NIGHT MARKET beautifully — two figures, one
prop, one interaction. This chapter is about the scene those tools *can't* quite hold,
and a second engine built for it.

The ceiling is always the same one: *fusion*. A diffusion model paints the whole canvas
at once, so the more independent subjects you ask for, the more it blends them — three
distinct people become a two-and-a-half-person smear, each prompt bleeding into its
neighbour. `objects:` and `relate:` push that ceiling up a little. Layered generation
moves it somewhere else entirely: instead of asking the model to keep several subjects
apart *while* painting them, it draws each subject *alone*, then paints the final image
once with those solo drafts holding everyone in place.

#section("The idea: constraints, not pixels")

A layered render runs in three moves.

First it splits your scene into a *plan*: a backdrop (the environment) plus a small set
of independent *subject layers*, each with its own full-detail prompt and a box on the
canvas. Second, it *drafts* each layer alone — one subject per image, where binding is
trivial because there is nothing to fuse with — and composes those drafts into a single
low-resolution *guide*. Third, it runs *one ordinary denoising trajectory* of your
finish model, gently steered toward the guide's low frequencies inside each subject's box
during the early steps, then left free to paint its own detail.

#callout(label: "The one rule that makes it safe")[
  Layers are *constraints on layout*, never pixels copied to the output. Not one pixel of
  a draft survives into the final image — only the coarse, low-frequency answer to
  "roughly what goes where" is borrowed, and only early, and only inside each box. The
  finish is still a single, coherent render in your model's own hand. That is why a
  layered image looks painted, not pasted.
]

#section("The plan")

A plan is a small HJSON file. `plakat layers new` scaffolds one to edit; here is the one
this chapter renders, a busier NIGHT MARKET than the poster dared — a vendor, a customer,
and a glowing lantern, three subjects that plain prose would fuse.

#screen(caption: "corpus/layered/night-market.hjson")[```
  size: "1216x832"
  global: {
    palette: "warm amber and honey lantern glow, soft teal night shadows"
    light:   "warm lantern light filling the lane, soft glow on wet stones, gentle fog"
    medium:  "cinematic poster illustration, warm and atmospheric"   # FINISH only
  }
  prompt: "a lively night market lane glowing with warm lantern light, a vendor
           serving a customer, stalls and paper lanterns receding into soft fog"
  backdrop: { prompt: "a lively night market lane, wet cobblestones catching warm
                       lantern light, rows of glowing stalls and lanterns into fog",
              weight: 0.55, window: 0.30 }
  layers: [
    { id: "vendor",   prompt: "a food vendor in a dark apron leaning over a cart", box: [0.08, 0.34, 0.46, 0.95], depth: 0.30, weight: 0.78, window: 0.40 }
    { id: "customer", prompt: "a customer in a warm coat reaching for a paper cup", box: [0.52, 0.36, 0.88, 0.96], depth: 0.35, weight: 0.78, window: 0.40 }
    { id: "lantern",  prompt: "a large round red paper lantern, glowing",          box: [0.66, 0.10, 0.86, 0.34], depth: 0.60, weight: 0.70, window: 0.34 }
  ]
  draft: { model: "sdxl", seed: 11 }
```]

Each layer's `box` is `[x0, y0, x1, y1]` in fractions of the canvas (a `place:` phrase like
`"center-left mid"` works too, if you would rather describe than measure). `depth` orders
the layers front-to-back — `0` is nearest, `1` farthest — so a nearer subject wins where
boxes overlap. The `global` look is anchored into every draft *except* `medium`: technique
is the finish's job, so only palette and light travel down to the solo drafts.

The `weight` and `window` on each layer are the *anchor strength* — how hard, and for how
long, the finish is pulled toward that subject's place in the guide. We will come back to
them at the end of the chapter; they are the dial between "follow the plan exactly" and
"blend into one cohesive scene."

#term("box / place / depth")[
  A layer's place in the frame. `box: [x0,y0,x1,y1]` gives an explicit rectangle in
  `[0,1]`; `place: "center-right mid"` resolves the same from words. `depth` (0 near … 1
  far) sets the front-to-back order the drafts are composed in. Together they are the
  *layout* the guide will hold — the constraint, not the content.
]

#section("Size classes: what the anchor can hold")

Not every box is big enough to steer. `plakat layers lint` validates the plan and reports,
for your finish model, which class each layer falls in.

#screen(caption: "Lint the plan for SDXL")[```
  $ plakat layers lint corpus/layered/night-market.hjson --model sdxl
  ◆  lint … (sdxl · 1216×832 px · 3 layer(s))
      · layer "vendor":   anchored (462px shorter side)
      · layer "customer": anchored (438px shorter side)
      · layer "lantern":  anchored (200px shorter side)
  →  PASS — clean
```]

An #emph[anchored] layer is large enough to get the full treatment: a guide anchor, plus
the verify-and-repair pass later. A #emph[hinted] layer is smaller — named in the finish
prompt and checked, but not anchored. A #emph[lifted] layer is tiny — too small to steer,
so its structure comes from a dedicated crop-and-regenerate pass. All three subjects here
are comfortably anchored. `layers show` prints the same resolution and draws the boxes,
coloured by class, so you can *see* the layout before spending a single GPU-second:

#screen(caption: "Draw the plan (offline — no model)")[```
  $ plakat layers show corpus/layered/night-market.hjson \
      --model sdxl --boxes plan.png
```]

#figure_img("assets/11-layered-plan.png", "Image 11 — the plan, drawn: three anchored subject boxes on the canvas (vendor lower-left, customer lower-right, lantern upper-right). This is pure geometry — no diffusion — so it is the cheapest way to check a composition before you render it.")

#section("Rendering")

With the plan lint-clean, one command runs the whole pipeline — drafts, guide, and the
anchored finish.

#screen(caption: "The full layered render")[```
  $ plakat layers render corpus/layered/night-market.hjson \
      --model sdxl --draft-model sdxl --draft-steps 8 --steps 36 \
      --guidance 6.5 --ramp 0.2 --seed 11 --keep stages/ --out poster.png
```]

The finish model is SDXL — the book's model. `--draft-model` picks the model that draws
the solo drafts; the drafts only need to be *roughly* right — their high-frequency detail
is thrown away — so a few steps of SDXL are plenty (`--draft-steps 8`). `--ramp` softens
how the anchor lets go (more on that below). `--keep stages/` writes the intermediate
drafts, the composed guide, and the weight/window anchor maps next to the output, which is
worth doing the first few times so you can watch the machine think.

#figure_img("assets/12-layered-poster.png", "Image 12 — the finished layered render: a cohesive night-market lane, its stalls, wet cobblestones and strung lanterns tying the vendor, the customer and the crowd into one atmospheric scene — painted together in a single SDXL pass. Three planned subjects that plain prose would have fused into a smear, woven into a place instead of pasted onto a void.")

#callout(label: "Flux works too")[
  Layered generation is wired for the SD family (SD 1.5 / SDXL) and Flux. Point `--model`
  at a Flux alias and the guide is encoded in Flux's own latent space — the anchor is
  the same, only the finish family changes. (SD3, Sana, PixArt and Cascade finishes are
  on the roadmap.)
]

#section("Cohesion: the anchor is a dial")

Look again at that render. The vendor, the customer and the lantern are where the plan put
them — but they are not pasted onto a black void; they sit in a real lane, with stalls,
wet cobblestones and strung lanterns tying the scene together. That cohesion is not
automatic, and the naive version of this plan gives you the opposite: three well-drawn
subjects floating in the dark. Two things produce the difference.

The first is the *backdrop*. Each subject is drafted ALONE on a plain surround (so it mattes
cleanly), which means the only thing joining the subjects in the guide is the backdrop
draft. A thin, dark "night lane" backdrop makes the guide islands-in-black, and the finish
faithfully paints islands in black. Give the backdrop a real, lit environment — "wet
cobblestones catching warm lantern light, rows of glowing stalls" — and the finish has
something to weave the subjects into.

The second is *anchor strength* — each layer's `weight` (how hard) and `window` (how long),
plus the `--ramp` that softens the hand-off. Turn them up and the finish reproduces the
guide's layout exactly, subjects crisp but disconnected. Turn them down and the finish
treats the guide as a loose suggestion, blending the subjects into a scene of its own
painting. The plan above uses `weight: 0.78, window: 0.40` on the figures — down from the
firm default — and renders with `--ramp 0.2`: enough to keep the vendor at his cart on the
left and the lantern up-right, loose enough to let SDXL fill the lane between them.

The guide also helps the seam automatically, so subjects sit *in* the scene rather than on
it: each matted subject is colour-harmonised toward the backdrop's palette (`--harmonize`),
and a soft contact shadow is laid under it (`--no-ground` turns this off) — which is why the
figures in Image 12 cast shadows onto the wet cobblestones instead of hovering above them.

#callout(label: "The trade you are making")[
  Anchor strength is the dial between *layout fidelity* and *cohesion*. High weight / long
  window = subjects land exactly where planned, at the cost of looking composited; low
  weight / short window = one seamless scene, at the cost of the plan being a suggestion. If
  a render looks like a collage, lower the weights and enrich the backdrop; if a subject
  drifts out of its box, raise them. There is no universally right setting — only the one
  that suits the image you are making.
]

#section("Did it land? Verify, repair, lift")

The layered pipeline can *check its own work*. `plakat layers verify` asks an open-vocabulary
detector (OWL-ViT) whether each anchored subject actually rendered inside its box, and
exits non-zero if any missed — so it can gate a repair.

#screen(caption: "Verify each subject is where the plan put it")[```
  $ plakat layers verify corpus/layered/night-market.hjson \
      --image poster.png --model sdxl
      ✓ vendor    anchored score 0.14  iou 0.71  "a food vendor …"
      ✓ customer  anchored score 0.12  iou 0.68  "a customer …"
      ✗ lantern   anchored score 0.00  iou 0.00  "a large red paper lantern"
  →  FAIL — 1 missing: lantern
```]

When a subject misses, `plakat layers repair --auto` verifies first and then re-asserts
just the layers that failed, with a masked img2img pass over each one's box — the rest of
the image is preserved. And for a subject too small to anchor in the first place, `plakat
layers lift` crops its box, regenerates it at a workable resolution, and composites it
back with a feathered seam. Both are targeted touch-ups, not a full re-render.

#subsection("Diff: reading the anchor")

`plakat layers diff` compares two images at the plan's low-frequency band — the band the
anchor actually constrains — and reports, whole-canvas and per box, how closely they
agree. Point it at the guide and the finish to *measure* whether the anchor took:

#screen(caption: "How well did the finish track the guide?")[```
  $ plakat layers diff corpus/layered/night-market.hjson \
      --a stages/__guide.png --b poster.png --model sdxl -o heat.png
```]

It is model-free and instant, and the heatmap it writes shows exactly where the finish
drifted from the plan — a fast, honest read on a render you are not sure about.

#section("Skipping the hand-written plan: the planner")

You do not have to author the HJSON yourself. `plakat layers plan` hands a prose
description to an LLM and gets a plan back — global look, backdrop, and separated subject
layers with placements — ready to lint, edit, and render.

#screen(caption: "Prose → plan, via a local model")[```
  $ plakat layers plan \
      "a night market: a vendor at a cart, a customer ordering, a lantern above" \
      -o market.hjson --model sdxl --provider ollama:qwen2.5-coder:14b
```]

`--provider` chooses the LLM: omit it for the small in-process model, or point it at a
model you have pulled with Ollama (`ollama:qwen2.5-coder:14b`) for sharper decompositions —
a bigger model places the subjects on opposite sides of a "face-off" instead of stacking
them. Whatever you get is an ordinary plan file: read it, nudge a box, and render.

#section("Layering from prose: compile --layered")

Everything in this chapter so far has started from a *plan* — hand-written, or from
`layers plan`. But you have spent the whole book in the *compile* path: prose in a `.txt`,
`plakat compile` out to a scenario. That path can reach layered generation on its own.

Add `--layered` to any compile and it inspects each scene it resolves. When a scene has
*two or more independent foreground figures that do not interact* — exactly the fusion-prone
case — compile decomposes it into a layered plan automatically and points the scenario at it.
There is nothing new to write in the prose: you already name the heroes with `foreground:`
(Chapter 9), and `--layered` simply turns that figure list into subject layers.

#screen(caption: "Compile prose straight to a layered scenario")[```
  $ plakat compile market.txt --layered --out market.hjson
  ✓  compiled → market.hjson
  ✓  layered plan → a_bustling_night_market_lane_glowing.layered.hjson
```]

The scene that qualified becomes a `type: layered` task, pointing at a sidecar plan written
next to the scenario:

#screen(caption: "market.hjson — the layered task")[```
  tasks:
  [
    {
      name: a_bustling_night_market_lane_glowing
      type: layered
      layered: { plan: "a_bustling_night_market_lane_glowing.layered.hjson" }
    }
  ]
```]

That sidecar is an ordinary plan — the same HJSON you met at the top of this chapter, built
for you: the backdrop from the scene's prose, one placed subject layer per `foreground`
figure, the finish look from your `style:`. Read it, nudge a box, and render it with `plakat
scenario market.hjson` like any other file. A scene with a single hero — or two figures the
prose says are interacting, a `relate:` contact — is left on the one-pass compile path
untouched; `--layered` only reroutes the scenes that actually need it.

#callout(label: "--improve sharpens each layer on its own")[
  The polish loop understands layers too. Run `compile --layered --improve` and, for a
  layered scene, `--improve` optimises *each layer's prompt separately* — rendering and
  aesthetically scoring one subject at a time, keeping the best wording per layer — rather
  than scoring the whole crowd at once. Each layer keeps its own history in the smysl corpus,
  so `--improve-skip-good` gates them one by one, and the winning prompts are written straight
  back into the sidecar plan.
]

#section("Which road? Layered vs. the compile path")

Two engines now stage a scene, and they are not rivals — they are for different scenes.

#warn(label: "Pick the tool for the number of subjects")[
  For one to three figures with a clear interaction — the whole of NIGHT MARKET — the
  compile path (`foreground`, `objects:`, `relate:`, `control-generate`) is the right
  road: it is simpler, it is what the rest of this book uses, and it renders in one pass.
  Reach for *layered* when a scene has *several independent subjects that each need full
  detail* and keep fusing no matter how you phrase them — a market of distinct stalls, a
  group portrait where every face matters, a poster with a cast. When `compile --analyze`
  flags fusion you cannot phrase your way out of, layers are the next rung — and `compile
  --layered` is the bridge that climbs it without ever leaving the prose path.
]

That is the last engine in the book. You have every tool a poster needs — from a single
prose line to a self-checking, layer-by-layer composition — each for the job it does best.

#recap((
  [Diffusion *fuses* independent subjects; layered generation avoids it by drafting each
  subject *alone*, then painting the final image once with those drafts as a guide.],
  [Layers are *constraints on layout, never pixels* — only the coarse low frequencies of
  the drafts steer the finish, early and inside each box, so the result is painted not
  pasted.],
  [A *plan* is a small HJSON: a `global` look, a `backdrop`, and subject `layers` with a
  `box`/`place` and `depth`; `layers lint` reports each layer's size class
  (anchored / hinted / lifted) and `layers show` draws the boxes offline.],
  [`layers render` runs drafts → guide → one anchored finish (SD family / Flux);
  `--keep` saves the stages. `verify` (OWL-ViT) checks each subject landed, `repair
  --auto` fixes the misses, `lift` rebuilds tiny subjects, `diff` measures the anchor.],
  [*Cohesion is a dial*: a rich, lit `backdrop` gives the finish an environment to weave
  the subjects into, and each layer's `weight` / `window` (with `--ramp`) trades layout
  fidelity for a seamless scene — lower them and enrich the backdrop if a render reads as
  a collage.],
  [`layers plan "<prose>"` decomposes a description into a plan via an LLM
  (`--provider ollama:<model>` for a bigger local model). Reach for layered generation
  when *many* independent subjects keep fusing — otherwise the compile path is simpler.],
  [The *compile path reaches layered on its own*: `compile --layered` decomposes any
  fusion-prone scene (two or more independent `foreground` figures, no interaction) into a
  `type: layered` task plus an auto-built sidecar plan, and `compile --layered --improve`
  then sharpens *each layer's prompt separately*.],
))
