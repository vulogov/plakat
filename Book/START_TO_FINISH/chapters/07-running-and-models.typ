#import "../design.typ": *
#chapter(number: 7, title: "Running It, and Choosing a Model")

#dropcap("A") scenario is ready; now we render it, and we make the first big
craft decision of the project: which model. plakat speaks to a whole family of
diffusion models, and they are not interchangeable — they differ in size, in speed,
in how they like to be prompted, and in how much text they can hold. This chapter
runs the poster and then chooses, deliberately, the model it will wear to print.

#section("Running the scenario")

One command renders every task in a scenario.

#screen(caption: "Render the poster")[```
  $ plakat scenario scenario.hjson
    task 1/1 · market-lane
     INFO 30 steps ████████████████████ 30/30  22.4s
    ✓ ./out/market-lane-1000.png
```]

If you only want to see what *would* run — how many tasks, at what size, which
model — add `--dry-run`. And because each task carries a name and a seed, re-running
the same scenario reproduces the same images, while bumping the global `seed`
produces a fresh, reproducible set.

#section("The model families")

plakat groups models into families, and the family — more than any single model —
determines how you prompt and what you can expect.

#chord_table((
  chord_row("sd15", "Small, fast, token-free. 512² native. The learning and CPU-friendly default; a hard 77-token text limit."),
  chord_row("sd21", "Like SD 1.5 but 768² native, v-prediction."),
  chord_row("sdxl", "1024² native, dual text encoders, ~150-token effective range. The reliable general-purpose choice — our poster's model."),
  chord_row("sdxl-turbo", "Adversarial-distilled SDXL: 1–4 steps, no guidance. Very fast drafts."),
  chord_row("pony", "An SDXL fine-tune (a full SDXL, budget and all) tuned for stylised character art."),
  chord_row("sd35-medium / -large", "MMDiT with a T5-XXL encoder — a much larger ~256-token budget and strong prose comprehension. Gated (needs HF_TOKEN)."),
  chord_row("flux-schnell / -dev", "Transformer models, no CLIP token cap — write flowing prose. schnell is 4-step and Apache-licensed; dev is larger and gated."),
  chord_row("sana / pixart", "Efficient large-context prose transformers (Gemma-2 / T5-XXL). Big budgets, light on memory."),
))

#term("Model family")[
  A group of models that share a text encoder and prompting style: SD 1.5 (77-token
  CLIP, keyword-friendly), SDXL (dual CLIP, ~150 tokens), SD3 (T5-XXL, ~256
  tokens), Flux (transformer, no CLIP cap), Cascade, and the efficient prose
  transformers Sana and PixArt. The family sets how you prompt and how much text
  survives to the render.
]

#section("The token budget follows the model")

That "how much text survives" is not a footnote — it is the reason your careful
prose sometimes gets trimmed. Each family has an effective token budget, and plakat
fits your prompt to the *actual model you chose*. SD 1.5 holds about 77 tokens; SDXL
about 150; the T5-driven families 256 or more. Ask a rich scene of SD 1.5 and the
tail of your prompt is at risk; ask the same of SDXL and it fits with room to spare.

#screen(caption: "Same prompt, different budgets")[```
  $ plakat compile prompts.txt --model sd15 --dry-run
      market-lane · family SD15 · budget ~77 tokens
  $ plakat compile prompts.txt --model sdxl --dry-run
      market-lane · family SDXL · budget ~150 tokens
  $ plakat compile prompts.txt --model sana --dry-run
      market-lane · family SD3  · budget ~256 tokens
```]

#callout(label: "Proper budget for the proper model")[
  plakat keys the budget to the real model, aliases and all — `pony` gets SDXL's
  150-token budget (it *is* an SDXL fine-tune), and `pixart` and `sana` get the
  T5-scale 256, not the small default. You do not manage this; you only benefit from
  it. The next chapter shows what happens at the budget's edge — and how plakat trims
  to fit without silently dropping your subject.
]

#section("Sizes and the native-resolution rule")

Each family was trained at a native resolution, and straying far from it invites
artefacts — duplicated heads, stretched bodies, repeated horizons. The rule of
thumb is simple: render near the model's native size, then *upscale* (Chapter 14)
to reach print dimensions.

#chord_table((
  chord_row("sd15 / sd21", "512² / 768². Small and forgiving."),
  chord_row("sdxl / pony", "1024². Our poster renders here, then upscales."),
  chord_row("sd35 / flux / sana", "1024² native; larger with care and memory."),
))

#warn(label: "Don't fight the native size")[
  Rendering SD 1.5 at 1024² or SDXL at 512² is the fastest way to a two-headed
  figure. Set `--size` to the family's native square (or a matched aspect via
  `--aspect`/`--base`), and leave the big final dimensions to the upscaler. plakat
  warns when a control-generate draft size and model are mismatched, for exactly
  this reason.
]

#section("What will run here?")

Before committing to a heavier model, ask the machine. `doctor --capability` (from
Chapter 1) judges each model *runs / tight / won't-fit* on your hardware and names
the lever that helps when something is tight. For the gated models — SD 3.5, Flux
dev — you will also need a HuggingFace token in your environment; plakat tells you
when one is missing rather than failing deep in a download.

#screen(caption: "Match ambition to hardware")[```
  $ plakat doctor --capability
      sdxl          runs      · 1024²  ← our choice
      sd35-medium   runs      · needs HF_TOKEN
      sd35-large    tight     · use PLAKAT_SD3_LOWMEM=1, or sd35-medium
      flux-dev      won't fit  · → flux-dev-gguf Q4 + --quantize-t5
```]

#callout(label: "Running big models on modest machines")[
  A tight model is often still reachable. SD 3.5 has a low-memory mode
  (`PLAKAT_SD3_LOWMEM=1`) that fits it on a 24 GB Metal machine; Flux has quantised
  GGUF variants. `doctor --capability` names the specific lever each time — you
  rarely have to guess.
]

#section("The poster's model")

For NIGHT MARKET we choose *SDXL*. It reads our two-figure, one-prop night scene
comfortably, renders at a print-friendly 1024², prompts in the mix of prose and
keyword clusters our compiler already targets, and — importantly for the control
work coming in Part IV — has the strongest ecosystem of pose and style adapters. It
is the reliable middle of the road, and a poster wants reliability more than it
wants the last five per cent of a harder-to-drive model.

We set it once, in the global block, and every scene inherits it:

#screen(caption: "Committing to SDXL")[```
  model: sdxl
  size: 1024x1024
  scheduler: euler-a
```]

With the model chosen and the scene rendering, only one thing stands between us and
a good frame: the dozen small dials — seed, steps, guidance, sampler — that turn a
correct image into the *right* one. That is tuning, and it is the next chapter.

#recap((
  [`plakat scenario scenario.hjson` renders every task; names and seeds make runs
  reproducible, and `--dry-run` previews without rendering.],
  [Model *families* — SD 1.5, SDXL, SD3, Flux, Cascade, Sana/PixArt — set how you
  prompt and how much text survives; the family matters more than the individual
  model.],
  [The token budget follows the *actual* model (SD 1.5 ~77, SDXL ~150, T5 families
  ~256), and plakat keys it correctly even for aliases like `pony` (SDXL) and
  `sana`/`pixart` (T5-scale).],
  [Render near a model's *native resolution* (SDXL at 1024²) and upscale later;
  rendering far off-native invites duplicated or stretched anatomy.],
  [`doctor --capability` says what runs / is tight / won't-fit here and names the
  lever (low-mem mode, quantised GGUF) that helps; gated models need `HF_TOKEN`.],
  [We commit the poster to *SDXL*: reliable, 1024² native, well-supported by the
  pose and style adapters Part IV will use.],
))
