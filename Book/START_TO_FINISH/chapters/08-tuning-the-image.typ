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

#section("Tuning that lives in the prose")

Not every dial is a setting; one is *prose*, and so it rides inside the loop with
everything else. **Prompt scheduling** lets the image start as one thing and finish as
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
  [Prompt scheduling (`[a:b:when]`, `[a|b]`) is *prose*, so it stays inside the loop —
  analyzable and reproducible like the rest of the scene.],
  [Budget packing happens at *compile*, inside the loop: it drops generic filler first,
  never the subject, records what it cut, and `--trace` explains a dropped phrase just
  as it explains a kept one.],
))
