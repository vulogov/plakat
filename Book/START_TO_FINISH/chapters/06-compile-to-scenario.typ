#import "../design.typ": *
#chapter(number: 6, title: "Compiling to a Scenario")

#dropcap("O")ur prose is clean and the analyzer is happy. Now we turn it into
something plakat can render: a scenario. This chapter is the compiler in full — how
it enhances each scene for the model, how to see exactly what it will do before it
does it, how to split a growing prose file across includes, and how to keep the
whole thing honest with a validation pass. By the end, `prompts.txt` becomes a
`scenario.hjson` you can run with confidence.

#section("The basic compile")

At its simplest, `compile` takes a prose file and writes a scenario.

#screen(caption: "Compile the poster")[```
  $ plakat compile prompts.txt --out scenario.hjson
    compiling 1 scene(s) via local …
    ✓ 1/1 · market-lane
  ◆ scene 'market-lane' · SDXL · sdxl
      translate: (english) · compose: 2 components
      enhance: local · negative: auto + 5 seeds
      fit: ~96 tokens (budget ~150, SDXL)
  ✓ compiled → scenario.hjson
```]

The trace to the right of each `◆` is worth reading: it names the model family it
resolved (SDXL), how many components it folded in, which enhancer it used, how many
negative seeds you supplied on top of the automatic set, and — the line we will
care about in Chapter 8 — how the final prompt fit the model's token budget.

If you leave off `--out`, compile writes `<stem>.hjson` beside the prose; pass `-`
to print the scenario to your terminal instead.

#callout(label: "Compile can also improve the prompt")[
  Compile's job is prose → scenario, but it can go one step further. `compile --improve`
  #emph[enhances, then automatically improves]: it renders, scores the result, rewrites
  the prompt to raise its aesthetic quality, and writes the *winning* prompt into the
  scenario. The smysl corpus is its memory, so it never re-tries a change it already
  rejected. Chapter 8 covers the improve loop in full; here, just know that the same
  `compile` you run to produce a scenario can also polish it.
]

#section("The enhancer, and choosing a provider")

The *enhance* stage is where a scene's plain prose becomes a prompt tuned to the
model family. That stage runs through a language model — the "provider." plakat
defaults to a small model that runs on your own machine (no key, a one-time
download); you can run a larger local model under Ollama, or point it at a hosted
model if you have a key.

#screen(caption: "Picking the enhancement provider")[```
  # on-device, no key, nothing leaves the machine (default)
  $ plakat compile prompts.txt --compile-provider local

  # a larger LOCAL model via Ollama — still nothing leaves the machine
  $ plakat compile prompts.txt --compile-provider ollama:qwen2.5-coder:14b

  # a HOSTED model, if you have a key in the environment
  # (your prose is sent to that provider — see the note below)
  $ plakat compile prompts.txt --compile-provider deepseek
  $ plakat compile prompts.txt --compile-provider gemini

  # let plakat choose what's configured
  $ plakat compile prompts.txt --compile-provider auto
```]

#warn(label: "What leaves your machine — and what doesn't")[
  *Image generation is always local.* Every render runs on your own GPU or CPU and
  never touches a provider. The only thing that can leave your machine is the *text*
  of the optional prose-reasoning steps — enhancement here, and later `--analyze`,
  `--improve`, and the layered planner — and only if you point one at a *hosted*
  provider. Be clear-eyed about what that means:

  #list(
    [If you choose a hosted provider (`deepseek`, `gemini`, or `auto` when a key is
     configured), the request — your prose and the prompt being enhanced or judged —
     *is sent over the network to that provider.* With `local` or `ollama:<model>`,
     nothing leaves the machine.],
    [Hosted models are usually *metered*: those calls can *cost money*. The account,
     the API key, the token usage, and the bill are yours to set up, watch, and pay —
     plakat neither meters nor caps them for you.],
    [Any such request is governed by *that provider's* terms of service and privacy
     policy, and is subject to their handling and retention — not plakat's. plakat
     neither sees nor controls what a provider does with what you send.],
    [To keep everything on your machine while still using a capable LLM, run one under
     *Ollama* and select it with `--compile-provider ollama:<model>`.],
    [These reasoning arcs are *entirely optional*. plakat compiles and renders without
     any of them — `--no-enhance --no-negative` (below) skips the model altogether, and
     `--analyze` / `--improve` only run when you ask. Whether to use an external
     provider, and for what, is your call and your responsibility.],
  )
]

#callout(label: "Deterministic compiles")[
  If you want a compile with *no* language model at all — byte-for-byte
  reproducible, offline, instant — pass `--no-enhance --no-negative`. The prose is
  assembled verbatim, components folded in, weights preserved, and the budget still
  fitted (that part is model-free). It is the mode continuous-integration checks use,
  and a fine default when your prose is already exactly what you want.
]

#section("Seeing what it will do: `--explain`")

Before spending even the small cost of an enhancement pass, you can ask compile to
*show its work*: the model family it resolved for each scene, the exact system
prompt it will send the enhancer, and the deterministic negative it will build. It
calls no model and writes nothing.

#screen(caption: "compile --explain")[```
  $ plakat compile prompts.txt --explain
  ── scene "market-lane" · family SDXL
  [positive system]
  You write prompts for Stable Diffusion XL (dual CLIP, ~150-token range).
  Mix natural language with keyword clusters. Aim 60–150 tokens. …

  [negative (deterministic)]
  lowres, bad anatomy, extra limbs, watermark, text, blurry, …
  ↳ negative seeds: daylight, blue sky, modern signage, cars, crowd
```]

`--explain` is how you answer "why did my prompt come out like that?" without
guessing. If a scene is being enhanced in a way you don't want, this is where you
see the instruction responsible — and often the fix is a clearer sentence in the
prose, or a different `model` so a different family profile applies.

#subsection("A dry run and a validity check")

Two more no-render inspections earn their keep on a big project. `--dry-run`
summarises what a compile *would* do — how many scenes, how many model calls, a
rough token estimate — so you can gauge cost before you spend it. And `--check`
validates that a scenario (compiled or hand-written) actually loads and uses known
task types, a fast offline gate for continuous integration.

#screen(caption: "Estimate, then validate")[```
  $ plakat compile prompts.txt --dry-run
    compile dry-run · prompts.txt · provider local
      - market-lane · family SDXL · 2 LLM call(s) · ~180 tok
    total: 1 scene · 2 call(s) · ~180 tokens (rough)

  $ plakat compile scenario.hjson --check
    ✓ scenario is valid (loads · known task types)
```]

#section("Splitting the prose: `@include`")

As a poster grows into a series, one flat `prompts.txt` gets unwieldy. plakat lets
you split prose across files and pull them together with `@include`, optionally
passing values in.

#screen(caption: "Composing prose from parts")[```
  # prompts.txt
  @include shared/globals.txt
  @include shared/components.txt

  name: market-lane
  The lane opens before us. A vendor works the cart, warm amber glow.
```]

Includes are why `--fix` goes to the trouble of tracing a phrase back to the file
it truly lives in: when your cart's description sits in `shared/components.txt`, a
fix to the cart edits *that* file, not a flattened copy. Your prose stays modular
and your history stays honest.

#section("The scenario it produced")

Finally, look at what compile wrote. You will rarely edit this by hand, but you
should recognise it.

#screen(caption: "A peek inside scenario.hjson")[```
  {
    model: sdxl
    size: 1024x1024
    seed: 1000
    out: ./out
    scheduler: euler-a
    tasks: [
      {
        name: market-lane
        prompt: "night market lane, wet cobblestones, a food vendor at a
          wooden cart under warm string lights, (warm amber glow:1.3),
          soft blue fog, two iron lanterns, cinematic night lighting"
        negative: "daylight, blue sky, cars, crowd, lowres, extra limbs, …"
      }
    ]
  }
```]

Everything in there came from your prose: the globals from the global block, the
prompt from your sentences (enhanced, with your `(warm amber glow:1.3)` weight
intact), and the negative from your seeds plus the automatic quality set. This is
compiled output. When you want to change the image, you change the prose and
recompile — and the two commands that make that a habit are `--watch`, which
recompiles whenever you save the prose, and `--diff`, which shows what a recompile
*would* change against an existing scenario.

#screen(caption: "A tight edit loop")[```
  $ plakat compile prompts.txt --watch
    👀 watching prompts.txt — Ctrl-C to stop
    ✓ compiled → prompts.hjson         # every time you save
```]

#recap((
  [`plakat compile prompts.txt --out scenario.hjson` turns prose into a scenario;
  the per-scene trace names the family, components, enhancer, negatives, and budget
  fit.],
  [The *enhance* stage runs through a provider — `local` (on-device, default),
  `deepseek`/`gemini` (hosted), or `auto`; `--no-enhance --no-negative` gives a
  deterministic, offline, reproducible compile.],
  [Inspect before you spend: `--explain` shows the family and exact system prompt,
  `--dry-run` estimates cost, and `--check` validates that a scenario loads.],
  [`@include` splits prose across files (with value passing); this is why `--fix`
  and `--trace` can trace a phrase to the file it truly lives in.],
  [The `scenario.hjson` is *compiled output* — recognise it, but edit the prose and
  recompile; `--watch` rebuilds on save and `--diff` previews a recompile's
  changes.],
  [The same `compile` can also *improve*: `--improve` enhances, then renders, scores,
  and rewrites the prompt to raise its aesthetic quality, writing the winner into the
  scenario (Chapter 8).],
))
