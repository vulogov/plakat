#import "../design.typ": *
#appendix(letter: "B", title: "Models, Families, and Budgets")

A model is chosen by *family* first — the family sets how you prompt, the native
size, and the token budget — and by the specific alias second. This appendix is the
map: what to reach for, and what to expect from it. See the live list any time with
`plakat models aliases`, and what your machine can run with `plakat doctor
--capability`.

#section("The families at a glance")

#chord_table((
  chord_row("SD 1.5", "512² native · ~77-token CLIP · keyword-friendly. Small, fast, token-free; the CPU-friendly learning default. Hard text limit."),
  chord_row("SD 2.1", "768² native · OpenCLIP-H · v-prediction. Like 1.5, a step larger."),
  chord_row("SDXL", "1024² native · dual CLIP · ~150 tokens. The reliable general-purpose family; best pose/style adapter ecosystem. The book's choice."),
  chord_row("SD 3.x", "1024² · T5-XXL + dual CLIP · ~256 tokens · strong prose. MMDiT; gated (HF_TOKEN). Low-mem mode fits Medium on 24 GB."),
  chord_row("Flux", "Transformer · no CLIP token cap · write flowing prose. schnell 4-step (Apache); dev larger and gated. Quantised GGUF for modest RAM."),
  chord_row("Cascade", "Würstchen-based · ~120 tokens · efficient latent space."),
  chord_row("Sana / PixArt", "Efficient large-context prose transformers (Gemma-2 / T5-XXL) · ~256 tokens · light on memory."),
))

#section("Common aliases")

#chord_table((
  chord_row("sd15, sd-1.5", "SD 1.5 base (community mirror). Token-free."),
  chord_row("sd15-inpaint", "9-channel inpainting UNet for masked edits."),
  chord_row("sd21", "SD 2.1, 768² native."),
  chord_row("sdxl", "SDXL base 1.0 — the poster's model."),
  chord_row("sdxl-turbo", "Distilled SDXL: 1–4 steps, guidance 0."),
  chord_row("pony", "Pony Diffusion v6 XL — an SDXL fine-tune (SDXL budget). Pairs with --look pony."),
  chord_row("sd35-medium / sd35-large", "SD 3.5, gated. Large needs ≥32 GB or PLAKAT_SD3_LOWMEM=1."),
  chord_row("flux-schnell / flux-dev", "Flux 4-step (Apache) / larger gated dev; GGUF variants for low RAM."),
  chord_row("sana, sana-1.5, pixart", "Efficient prose transformers; T5-scale budget."),
))

#section("The budget, keyed to the model")

The token budget is not cosmetic — it is how much of your prose survives to the
render. plakat fits the prompt to the *actual* model you chose, and it gets the
aliases right: `pony` is budgeted as SDXL, `pixart` and `sana` as T5-scale, not as a
small default. When a prompt runs over, the model-free packer drops generic filler
first and never the subject, recording what it cut (Chapter 8).

#chord_table((
  chord_row("~77 tokens", "SD 1.5 / SD 2.1 — front-load the subject; the tail is at risk."),
  chord_row("~120 tokens", "Cascade."),
  chord_row("~150 tokens", "SDXL / SDXL-turbo / pony."),
  chord_row("~256 tokens", "SD 3.x / Sana / PixArt — room for rich prose."),
  chord_row("no CLIP cap", "Flux — write as much flowing prose as the scene wants."),
))

#section("Native sizes")

#chord_table((
  chord_row("512²", "SD 1.5, PixArt-512, Sana-512, sd15-inpaint."),
  chord_row("768²", "SD 2.1."),
  chord_row("1024²", "SDXL, pony, SD 3.5, Flux, Sana, PixArt — render here, upscale later."),
))

#section("Choosing, in one breath")

#chord_table((
  chord_row("Learning / CPU / speed", "sd15 — small, token-free, forgiving."),
  chord_row("Reliable general work", "sdxl — the book's default; best adapter support."),
  chord_row("Rich prose comprehension", "sd35-medium or flux — big budgets, strong text following."),
  chord_row("Fast drafts", "sdxl-turbo, or any base + --fast (lightning/hyper/lcm)."),
  chord_row("Stylised character art", "pony — an SDXL fine-tune; pair with a look."),
))

#callout(label: "Let the machine narrow it")[
  Don't guess what fits: `plakat doctor --capability` judges each model
  runs / tight / won't-fit on your hardware and names the lever (low-mem mode,
  quantised GGUF) that rescues a tight one. `plakat models recommend` and `plakat models aliases`
  keep the live catalogue in front of you.
]
