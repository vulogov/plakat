#import "../design.typ": *
#chapter(number: 8, title: "Tuning the Image")

#dropcap("T")he scene is correct; now we make it *good*. Tuning is the set of small
dials that turn a plausible render into the one you want to print — the seed that
frames it well, the sampler that sharpens it, the step count that earns its time,
the guidance that balances fidelity against life. None of these change *what* is in
the image; they change how well it is realised. And nearly all of them live where the
rest of your poster lives: in the *scenario*. You tune by editing the scenario (or the
prose it came from) and re-running it — staying inside the prose → compile → scenario
loop, not stepping outside it.

#section("Where the dials live: the scenario")

The four dials you reach for most — the seed, the step count, the guidance, and the
sampler — are all scenario directives. Set them in the global block and every task
inherits them; change one and re-run `plakat scenario` to see only that change.

#chord_table((
  chord_row("seed", "The initial noise the image grows from. A different seed is a different composition of the same scene. Fix it to change one thing at a time."),
  chord_row("steps", "Denoising steps. More is smoother, to a point; 28–35 is plenty for SDXL. Past ~40 you mostly buy time."),
  chord_row("guidance", "How hard the model obeys the prompt (CFG). ~7 is a good middle; lower is looser and more natural, higher is more literal and can scorch."),
  chord_row("scheduler", "The sampler: default · ddim · euler-a · unipc. euler-a often sharpens SD/SDXL at the same step count; unipc (DPM++) converges fast."),
))

#screen(caption: "Tuning dials in the scenario — edit, then re-run")[```
  model: sdxl
  seed: 1000
  steps: 30
  guidance: 6.5
  scheduler: euler-a

  $ plakat scenario scenario.hjson      # re-render with the new dials
    ✓ ./out/market-lane-1000.png
```]

Because the seed is fixed, changing `guidance` from 7.5 to 6.5 and re-running shows you
*exactly* what the guidance did — same composition, softer obedience — with nothing
else moving. That is the discipline the scenario gives you: reproducible, one-variable
tuning. Edit the source, re-run, compare. No hunting for a frame you can't get back.

#subsection("Seeing several framings at once")

To compare compositions rather than settings, ask the scenario for a *set*: bump
`count`, and each task renders several images at stepped, reproducible seeds. Pick the
framing you like and pin its seed back in the global block.

#screen(caption: "A reproducible set of seeds, from the scenario")[```
  count: 4          # renders seeds 1000, 1001, 1002, 1003 — all reproducible

  $ plakat scenario scenario.hjson
    ✓ ./out/market-lane-1000.png … market-lane-1003.png
```]

#section("The scratchpad: scouting with `generate`")

There is a faster, looser way to *explore* — and it is worth being clear about what it
is. `plakat generate` renders a single prompt straight from the command line. It is a
*scratchpad*: it sits outside the prose → compile → scenario loop, it keeps nothing you
have to maintain, and the images it makes are throwaways. Its job is to let you try a
seed, a variation, or a sampler in seconds, so you can carry the *setting you liked*
back into the prose or the scenario — where the poster you keep actually comes from.

#callout(label: "Generate explores; the scenario produces")[
  Use `generate` the way a painter uses a corner of the canvas to test a colour. Scout
  freely — but the finished poster is always a *scenario* render, driven by your
  compiled prose. Anything you discover on the scratchpad (a good seed, a step count, a
  sampler) you fold back into the scenario; you do not ship the scratch.
]

A few things the scratchpad is good at:

#subsection("Nudging a frame you like: subseed")

When a seed is *almost* right, a *subseed* blends a second noise into it for "the same
image, nudged" — a slightly different drape or pose without losing the framing.

#screen(caption: "Scout a variation, then keep the seed")[```
  $ plakat generate "<the market-lane prompt>" --model sdxl \
      --seed 1000 --subseed 77 --subseed-strength 0.15
```]

`--subseed-strength` in `[0,1]` sets the blend; `0.05–0.2` nudges, `1` is the subseed's
own noise (SD 1.5 / SDXL). Found the nudge you want? The seed and subseed are recorded
in the image's sidecar — carry them into the scenario and the batch reproduces it.

#subsection("Contact sheets and culling")

To eyeball many at once, `--count` with `--grid` writes a contact sheet; to let the
machine triage, `--keep-best` scores them by an aesthetic predictor and keeps the top
few. Both are scouting conveniences.

#screen(caption: "Scout wide, keep the best few")[```
  $ plakat generate "<prompt>" --model sdxl --count 12 --keep-best 3 --grid
  $ plakat rank out/ --top 5          # or score a folder you already have
```]

#subsection("Going fast while scouting")

When you want frames *now*, the `--fast` presets bundle a distillation LoRA with the
right sampler and step count so a draft lands in a handful of steps instead of thirty —
ideal for the scratchpad, then drop back to full steps in the scenario for the frame
you commit to.

#screen(caption: "Four-to-eight-step scouting drafts")[```
  $ plakat generate "<prompt>" --model sdxl --fast lightning-sdxl-8
  $ plakat generate "<prompt>" --model sdxl --fast hyper-sdxl-4
```]

(There are Flux presets too — `hyper-8`, `turbo-alpha` — and `lcm-sd15` for SD 1.5.)

#section("Letting plakat find the better prompt: `compile --improve`")

Culling a batch tells you which render scored best. The natural next question is whether
plakat can *use* that score to make the prompt better — and it can, without ever leaving
the loop. `compile --improve` runs the whole pipeline in one command: it *enhances* your
prose into a prompt (the translation pass), then *improves* it — rendering, scoring with
the same aesthetic predictor `--keep-best` uses, proposing one small rewrite, keeping it
only if the score truly rises — and finally *writes the winning prompt back into the
scenario*. The improvement lands in the HJSON you render, not just on the screen.

#screen(caption: "Enhance, improve, and emit — one command")[```
  $ plakat compile prompts.txt --improve --improve-model sdxl
    ── scene market-lane
       · aesthetic 6.18 → "wet stones" → "shimmering puddles" → 6.39  kept
       · aesthetic 6.27  (reverted)
    ✓ compiled (improved) → prompts.hjson
```]

#subsection("The corpus is what keeps it from marching")

Left to itself, an automatic rewrite loop *wanders*: it tries a change, undoes it, tries
it again — a #emph[death march] that spends renders and converges on nothing. plakat's
cure is the smysl corpus from Chapter 5, used now as an *active memory* rather than a
record you read later. Every move the loop tries — kept #emph[or reverted] — is written
to the corpus, and before proposing the next one the loop consults that record and
*refuses* to re-try a rejected change or its reverse. The corpus is created if it does
not exist and consulted on every run, so a second `--improve` never re-walks ground the
first one already ruled out. The loop always stops with a *reason* — a plateau, the pass
budget spent, or no fresh move left — and reports the best prompt it found.

#term("The corpus as tabu memory")[
  The same `<name>.smysl` sidecar that records *why* a fix was applied now also records
  every rewrite the improve loop tried and how it scored. That history is a #emph[tabu
  list]: the loop cannot repeat a move it already spent, which is precisely what turns a
  death march into a search that converges. The corpus is not a side-effect of the
  process — it #emph[is] the process's memory.
]

#subsection("Steering the loop")

#chord_table((
  chord_row("--improve-all · --improve-scene a,b", "Improve every scene, or a named subset (the flag repeats and takes commas). Default: the first scene only."),
  chord_row("--improve-seeds N", "The aesthetic score is noisy, so each candidate is rendered and scored on N seeds (default 2) and averaged — a kept/rejected verdict you can trust."),
  chord_row("--improve-skip-good", "Consult the corpus and SKIP any scene already at the best score a prior run reached, so recompiling doesn't re-polish what's done. Use --improve-target N for an absolute bar instead."),
  chord_row("--improve-passes N", "How many rewrites to try per scene before stopping (default 6)."),
  chord_row("--keep-compiled-images", "Archive every candidate it renders — normally scored then discarded — so you can see the whole trajectory as image files."),
))

#callout(label: "What --improve tunes — and what it doesn't")[
  `--improve` optimizes the *prompt*: wording, composition, palette, light. It renders a
  plain image of the emitted prompt — without your LoRAs or control-generate — so it
  tunes the *writing*, not the painted finish. Reach for it to find a stronger prompt;
  reach for the persona LoRAs and the control ladder (Chapters 11 and 9) for the look.
]

#section("Tuning that lives in the prose")

Not every dial is a setting; one is *prose*, and so it rides inside the loop with
everything else. *Prompt scheduling* lets the image start as one thing and finish as
another — lay down a composition early, let detail take over late — written right in the
scene's text, so it survives compile into the scenario prompt.

#screen(caption: "Scheduling, written in the prose")[```
  # in prompts.txt — switch from A to B at 40% of the steps
  a [foggy empty lane : bustling lit stall : 0.4] at night

  # alternate between two every step
  a lane lit by [lanterns | string lights]
```]

`[a:b:when]` swaps `a` for `b` once the render passes the fraction `when`; `[a|b]`
alternates each step (SD 1.5 / SDXL). Because it is prose, it is analyzable, fixable,
and reproducible like the rest of your scene — a tuning tool that never leaves the loop.

#section("The dial you don't turn: fitting the budget")

One more thing happens every time you *compile* — inside the loop, with no dial to set —
and it is the reason your rich prose survives to the render intact. When a prompt runs
past the model's token budget (Chapter 7), plakat *packs* it to fit, in a way you can
audit rather than silently dropping the end of your sentence.

#term("Model-free budget packing")[
  When a prompt exceeds the model's budget, plakat first tries a deterministic pack:
  it scores each phrase by attention weight, position, and whether it is generic
  filler, then drops the lowest-value spans — *never the subject* — until the prompt
  fits. It records exactly what it cut and why. Only if trimming filler isn't enough
  does it fall back to asking the language model to reword.
]

#screen(caption: "A budget trim you can read — at compile time")[```
  $ plakat compile prompts.txt --model sd15
  ◆ scene 'market-lane' · SD15
      … prompt was ~86 tokens (over the SD15 ~77 budget) —
        packed model-free to ~75, dropped 2:
        dramatic lighting [low-value]; ultra detailed [low-value]
```]

Notice *what* it dropped: generic quality-boosters, not your vendor, cart, or fog. And
because the decision is recorded in the smysl corpus from Chapter 5, you can ask about
it later — the same `--trace` that explained why a phrase *is* in your prompt now
explains why one is *gone*.

#screen(caption: "Why did that phrase disappear?")[```
  $ plakat compile prompts.txt --trace "dramatic lighting"
    • o/pack-3eh4u [observation]
        budget pack 'market-lane': dropped 2 spans to fit SD15 in 77 tokens
        used 75/77; dropped: dramatic lighting [low-value] …
```]

The corpus traces both the changes you made and the budget the model imposed — the two
forces that shape a final prompt. On SDXL our poster fits with room to spare; on a
tighter model, this is what keeps the trim honest. With the scene tuned in the scenario
and the budget fit at compile, the loop has done its work — and we can start adding
figures.

#recap((
  [The core dials — *seed*, *steps*, *guidance*, *scheduler* — are scenario directives:
  edit the scenario (or its prose) and re-run `plakat scenario` for reproducible,
  one-variable tuning. Use `count` for a reproducible set of framings.],
  [`plakat generate` is a *scratchpad outside the loop* — scout a seed, a subseed
  variation, a `--fast` draft, or cull with `--keep-best`/`rank`, then fold the setting
  you liked back into the scenario. You scout with generate; you ship with scenario.],
  [`compile --improve` closes an *automatic* loop inside compile: enhance → improve →
  emit. It renders, scores with the aesthetic predictor, keeps only rewrites that truly
  help, and *writes the winner into the scenario*. The smysl corpus is its memory — every
  tried move, kept or reverted, so it never re-marches. `--improve-skip-good` leaves
  already-good scenes alone.],
  [Prompt scheduling (`[a:b:when]`, `[a|b]`) is *prose*, so it stays inside the loop —
  analyzable and reproducible like the rest of the scene.],
  [Budget packing happens at *compile*, inside the loop: it drops generic filler first,
  never the subject, records what it cut, and `--trace` explains a dropped phrase just
  as it explains a kept one.],
))
