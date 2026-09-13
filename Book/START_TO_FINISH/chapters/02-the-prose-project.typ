#import "../design.typ": *
#chapter(number: 2, title: "The Prose Project")

#dropcap("Y")ou made an image in the last chapter by typing a sentence straight
into `plakat generate`. That is fine for a one-off. But a poster is not a one-off:
you will render it dozens of times as you tune the scene, add figures, and try
looks, and you want every one of those renders to share the same settings, the
same negative terms, the same reproducible seed. The moment you want *the same
image, changed a little*, you want a project.

This chapter sets up that project and introduces the two files at the heart of how
plakat wants you to work — one you write, one the machine writes — and the command
that turns the first into the second.

#section("Bootstrapping with `init`")

`plakat init` scaffolds a starter project in the current directory: a
ready-to-run scenario, a `wildcards/` folder for reusable phrase lists, and a
focused `.gitignore` so your renders and model cache don't end up in version
control.

#screen(caption: "Start a project")[```
  $ mkdir night-market && cd night-market
  $ plakat init
  ✓ wrote scenario.hjson
  ✓ wrote wildcards/  (adjective.txt, place.txt, …)
  ✓ wrote .gitignore
    next:  plakat scenario scenario.hjson
```]

Open `scenario.hjson` and you will see a heavily commented file: a `model`, a
`size`, a `seed`, an `out` directory, a sampler, and one or more *tasks*, each
with a prompt. This is a *scenario* — the machine-readable batch format plakat
actually renders from.

#term("Scenario")[
  An HJSON file describing one or more render *tasks* and the global settings they
  share (model, size, seed, sampler, output folder). `plakat scenario file.hjson`
  runs every task in it. It is precise, complete, and verbose — and it is *not*
  where you want to do your thinking.
]

You *could* write scenarios by hand, and for a one-task tweak you sometimes will.
But a scenario is a poor place to compose. It mixes what you mean ("a foggy lane
with a food cart") with how the machine should render it (steps, guidance,
negative terms, budget). Change the scene and you must hand-edit prompt strings,
negatives, and per-task settings in lockstep. That is editing the wrong thing.

#section("The file you actually write: `prompts.txt`")

plakat's answer is to let you author in prose and *compile* it. You write a plain
`prompts.txt` — sentences and a few simple directives — and `plakat compile` turns
it into a scenario, enhancing each prompt for the model family, adding sensible
negative terms, and fitting everything to the model's token budget.

#pipeline_arc()

Here is the beginning of our poster's prose file. The first block, before the
first blank line, is the *global block*: settings shared by every scene.
Everything after is a *scene* — for now, just the empty lane.

#screen(caption: "prompts.txt — the global block and one scene")[```
  # NIGHT MARKET — poster prose. Edit THIS file; compile the rest.
  model: sdxl
  size: 1024x1024
  seed: 1000
  out: ./out

  name: empty-lane
  A narrow cobbled market lane at night, wet stones catching the light.
  Two iron lanterns on stone posts throw a warm amber glow; soft fog
  swallows the far end of the lane. Quiet, empty, atmospheric.
```]

Notice what is *not* here: no comma-separated keyword soup, no "masterpiece, best
quality, 8k, highly detailed" boilerplate, no manually tuned negative list. You
write what the scene *is*, in sentences. The directives — `model`, `size`, `seed`,
`out`, and a scene's `name` — are the few settings prose can't express in words.

#term("Global block vs. scene block")[
  The block before the first blank line sets project-wide defaults (`model`,
  `size`, `seed`, `out`, `loras`, and reusable pieces). Each later block is one
  scene: free-text prose plus optional per-scene directives like `name:`, `seed:`,
  or `negative:`. A scene can override most globals for itself.
]

#section("What compile does — the short version")

Run `compile` and plakat reads the prose, resolves the global settings, and rewrites
each scene into a scenario task. We will spend all of Chapter 6 on this; here is
just the shape of it, so the next three chapters make sense.

#compile_stages()

Each scene passes through the same stages: any non-English is translated,
components are composed into one description, the prompt is *enhanced* in a way
appropriate to the model family (SDXL likes keyword clusters; Flux and SD3 like
prose), a negative prompt is assembled from your seeds plus a curated quality set,
and the whole thing is *fit to the model's token budget*. Out comes a scenario.

#screen(caption: "A first look at compile")[```
  $ plakat compile prompts.txt --out scenario.hjson
    compiling 1 scene(s) via local …
    ✓ 1/1 · empty-lane
  ◆ scene 'empty-lane' · SDXL · sdxl
      translate: (english) · compose: prose only
      enhance: local · negative: auto + 0 seeds
      fit: ~41 tokens (budget ~150, SDXL)
  ✓ compiled → scenario.hjson
  $ plakat scenario scenario.hjson
    ✓ ./out/empty-lane-1000.png
```]

Two commands: `compile` turned prose into a scenario, and `scenario` rendered it.
From here on you will almost always edit `prompts.txt` and re-compile — never
hand-edit `scenario.hjson`. The scenario is *output*, like a compiled binary. The
prose is your source.

#callout(label: "You can still go direct")[
  Nothing stops you from running `plakat generate` for a quick experiment, or
  hand-writing a scenario for a one-off batch. The prose project is the workflow
  that *scales* — with a poster you will render many times, it pays for itself by
  the second afternoon. Use the quick path to explore; move to prose to build.
]

#section("Why a whole workflow for one picture?")

It is fair to ask why an image needs source files, a compiler, and an output
format at all. The answer is the thing this book is really about: *the cost of a
wrong render.* A diffusion model is slow and a little unpredictable. Every time you
change a prompt and hit render, you are placing a bet — a minute or three of
compute — that the change will help. Most of the frustration in image work is
losing those bets: the scene was too crowded, a word meant nothing to the model, a
figure came out standing when you wanted it seated, and you find out *after* the
render.

The prose project exists so you can make those bets cheaply and read the odds
before you place them. Because your scene is prose, plakat can *analyze* it and
tell you what will probably fail. Because analysis is cheap, you can fix the prose
and re-check in seconds. Only when the prose looks sound do you compile and spend a
real render. That loop — write, analyze, fix, compile, run — is the next three
chapters, and it is the difference between an evening of waiting and an evening of
making.

#two_col("The quick path", "The prose project",
  [`plakat generate "..."` — one sentence, one image. Perfect for exploring an
  idea or checking a model. Nothing to maintain, nothing reproducible beyond the
  saved recipe.],
  [`prompts.txt` → `compile` → `scenario` → `run`. Edit prose, re-compile, render
  many consistent versions. Analyzable, fixable, reproducible. The way to *build* a
  poster rather than stumble onto one.])

#recap((
  [`plakat init` scaffolds a project: a `scenario.hjson`, a `wildcards/` folder,
  and a `.gitignore`.],
  [A *scenario* is the machine-readable batch format plakat renders from — precise
  but a poor place to compose. Treat it as compiled output, not source.],
  [You author in a plain `prompts.txt`: a *global block* of shared settings, then
  one *scene block* of prose per image. Write what the scene *is*, in sentences.],
  [`plakat compile prompts.txt` turns prose into a scenario — translating,
  composing, enhancing per model family, adding negatives, and fitting the token
  budget — and `plakat scenario` renders it.],
  [The whole workflow exists to make the *wrong* render cheap to avoid: prose can
  be analyzed and fixed before you spend compute, which is what the next three
  chapters do.],
))
