#import "../design.typ": *
#chapter(number: 8, title: "Tuning the Image")

#dropcap("T")he scene is correct; now we make it *good*. Tuning is the set of small
dials that turn a plausible render into the one you want to print — the seed that
frames it well, the sampler that sharpens it, the step count that earns its time,
the guidance that balances fidelity against life. None of these change *what* is in
the image; they change how well it is realised. This chapter walks the dials in the
order you actually reach for them, and closes on the one that works invisibly:
plakat fitting your prose to the model's budget.

#section("The seed, and controlled variation")

The seed is the initial noise the image grows from. Change it and you get a
different composition of the same scene. The first move in tuning is simply to see
a handful of seeds and pick the framing you like.

#screen(caption: "See several framings")[```
  $ plakat generate "<the market-lane prompt>" --model sdxl \
      --count 6 --grid
    ✓ ./out/plakat-<seed>.png  (×6)
    ✓ ./out/plakat-grid-<seed>.png
```]

`--grid` tacks a contact sheet next to the six frames so you can compare at a
glance. Found one you almost like? You do not have to abandon it to explore nearby —
that is what a *subseed* is for.

#term("Subseed (variation seed)")[
  A second seed whose noise is blended into the main seed's, set by
  `--subseed-strength` in `[0,1]`. Small strengths (`0.05–0.2`) give you "the same
  image, nudged" — a slightly different pose or drape without losing the framing you
  liked. It is controlled variation, not a fresh roll. SD 1.5 / SDXL.
]

#screen(caption: "Nudge a frame you like")[```
  $ plakat generate "<prompt>" --model sdxl \
      --seed 1000 --subseed 77 --subseed-strength 0.15
```]

#section("Steps, guidance, and the sampler")

Three dials govern the denoising itself, and they live in the scenario as much as on
the command line — set them in the global block and every task inherits them.

#chord_table((
  chord_row("steps", "How many denoising steps. More is smoother, to a point; 28–35 is plenty for SDXL. Past ~40 you mostly buy time."),
  chord_row("guidance", "How hard the model obeys the prompt (CFG). ~7 is a good middle; lower is more natural and loose, higher is more literal and can scorch."),
  chord_row("scheduler", "The sampler: default · ddim · euler-a · unipc. euler-a often sharpens SD/SDXL at the same step count; unipc (DPM++) converges fast."),
))

#screen(caption: "Tuning dials in the scenario")[```
  model: sdxl
  steps: 30
  guidance: 6.5
  scheduler: euler-a
```]

#subsection("Going fast: distilled presets")

When you are exploring and want frames *now*, the `--fast` presets bundle a
distillation LoRA with the right sampler and step count so a draft lands in a
handful of steps instead of thirty.

#screen(caption: "Four-to-eight-step drafts")[```
  $ plakat generate "<prompt>" --model sdxl --fast lightning-sdxl-8
  $ plakat generate "<prompt>" --model sdxl --fast hyper-sdxl-4
```]

Use a fast preset to scout seeds and compositions, then drop back to full steps for
the frame you commit to. (There are Flux presets too — `hyper-8`, `turbo-alpha` —
and `lcm-sd15` for SD 1.5.)

#section("Letting the scene evolve: prompt scheduling")

Sometimes you want the image to *start* as one thing and *finish* as another — lay
down a strong composition early, then let detail take over. Prompt scheduling
expresses that in the prompt itself.

#screen(caption: "Swap and alternate mid-render")[```
  # switch from A to B at 40% of the steps
  a [foggy empty lane : bustling lit stall : 0.4] at night

  # alternate between two every step
  a lane lit by [lanterns | string lights]
```]

`[a:b:when]` swaps `a` for `b` once the render passes `when` (a fraction of the
steps); `[a|b]` alternates each step. It is a precise way to buy structure early and
richness late without a second pass. (SD 1.5 / SDXL, txt2img.)

#section("Steering by region")

When one part of the frame needs its own emphasis — the amber cart bright, the fog
edges quiet — you can weight *regions* of the canvas independently.

#screen(caption: "Per-region weighting")[```
  $ plakat generate "<prompt>" --model sdxl \
      --region "0.55,0.35,1.0,0.9 w=1.25 feather=0.1" \
      --region "0.0,0.0,0.4,1.0 w=0.85"
```]

Each `--region` is a box in fractional coordinates with a weight and a feather.
This is the light touch; when you need to place whole *figures* by region — the
vendor here, the customer there — that graduates into the composition tools of
Part IV. For a mood adjustment on a single-subject frame, regions alone do the job.

#section("Picking the best of many: `rank` and `--keep-best`")

Tuning produces a lot of near-misses. Rather than eyeball forty frames, let plakat
score them by an aesthetic predictor and surface the best.

#screen(caption: "Score and keep the best")[```
  # generate 12, keep the 3 best by aesthetic score
  $ plakat generate "<prompt>" --model sdxl --count 12 --keep-best 3

  # or rank a folder you already have
  $ plakat rank out/ --top 5
```]

#callout(label: "Taste is yours; triage is the machine's")[
  The aesthetic score is a helper, not a judge — it is good at throwing out the
  obviously broken frames so you spend your attention on the contenders. Use
  `--keep-best` to cull, then choose with your own eye.
]

#section("The dial you don't turn: fitting the budget")

There is one more thing happening every time you compile, and it is the reason your
rich prose survives to the render intact. When a prompt runs past the model's token
budget (Chapter 7), plakat *packs* it to fit — and it does so in a way you can
audit, rather than silently dropping the end of your sentence.

#term("Model-free budget packing")[
  When a prompt exceeds the model's budget, plakat first tries a deterministic pack:
  it scores each phrase by attention weight, position, and whether it is generic
  filler, then drops the lowest-value spans — *never the subject* — until the prompt
  fits. It records exactly what it cut and why. Only if trimming filler isn't enough
  does it fall back to asking the language model to reword.
]

#screen(caption: "A budget trim you can read")[```
  $ plakat compile prompts.txt --model sd15
  ◆ scene 'market-lane' · SD15
      … prompt was ~86 tokens (over the SD15 ~77 budget) —
        packed model-free to ~75, dropped 2:
        dramatic lighting [low-value]; ultra detailed [low-value]
```]

Notice *what* it dropped: generic quality-boosters, not your vendor, cart, or fog.
And because the decision is recorded in the smysl corpus from Chapter 5, you can ask
about it later.

#screen(caption: "Why did that phrase disappear?")[```
  $ plakat compile prompts.txt --trace "dramatic lighting"
    • o/pack-3eh4u [observation]
        budget pack 'market-lane': dropped 2 spans to fit SD15 in 77 tokens
        used 75/77; dropped: dramatic lighting [low-value] …
```]

This is the same `--trace` that explained *why a phrase is in* your prompt in
Chapter 5, now explaining *why one is gone*. The corpus traces both the changes you
made and the budget the model imposed — the two forces that shape a final prompt —
so nothing about the render is a mystery. On SDXL, our poster fits with room to
spare; on a tighter model, this is what keeps the trim honest.

#recap((
  [Start tuning with the *seed*: render several with `--count`/`--grid`, then use a
  *subseed* (`--subseed-strength 0.05–0.2`) for "the same frame, nudged."],
  [Set *steps*, *guidance*, and *scheduler* in the scenario (28–35 steps, ~7
  guidance, `euler-a` a good default); scout fast with `--fast` distillation presets,
  then commit at full steps.],
  [Shape the render over time with prompt scheduling (`[a:b:when]`, `[a|b]`) and by
  space with per-region weighting (`--region`).],
  [Cull with the aesthetic scorer — `--keep-best N` on generation, or `plakat rank`
  on a folder — then choose with your own eye.],
  [Budget packing fits your prose to the model *auditably*: it drops generic filler
  first, never the subject, records what it cut, and `--trace` explains a dropped
  phrase just as it explains a kept one.],
))
