# plakat — release history

"What's new" sections for v0.13 through v0.19. The current
release's notes live in the [main README](../README.md). Older
cycles are archived here so the README stays focused on what's
new this turn.

For commit-level history see `git log`; for migration notes the
per-cycle commits carry the rationale + before/after.

## What's new in v0.19 — local enhancer polish, partial-rerun, WebP, Kontext compositions

Nine features in three groups. Pairs with v0.18's larger surface
(Flux Kontext, A1111 attention on Flux+SD3, BREAK, local prompt
enhancer) — v0.19 sands down rough edges and unblocks the two
Kontext compositions deferred from v0.18.

### Top picks (3 features)

- **Enhancer CLI flag surface + disk cache**. The v0.18 local
  enhancer's internals get CLI flags: `--enhance-temp F` (default
  greedy / `0.0`), `--enhance-max-tokens N` (default 96),
  `--enhance-system PATH` (custom system prompt), and
  `--enhance-cache` (opt-in SHA-256 disk cache at
  `~/.cache/plakat/enhance/`). Cache hits skip the LLM forward
  entirely — scenarios re-enhancing the same prompts across runs
  go from ~3-5s per prompt to instant.
- **`plakat animate --resume`**. Long animates that crash on frame
  23 of 24 no longer require re-rendering all 24. The flag scans
  `<out>/frame-NNNN.png`, skips frames already on disk, re-runs
  only what's missing. Mirrors the scenario `--resume` pattern
  added in v0.17.
- **scenario `--only TASK[,TASK,…]` + `--limit N`**. Partial-rerun
  affordances for long batches. `--only` runs just the named
  tasks (typo'd names bail up front with the supported list);
  `--limit` runs the first N. Both compose with `--resume` and
  `--dry-run`. `seed_offset` advances on skipped tasks so a
  partial run produces seeds identical to the full batch — no
  drift when iterating.

### Round-out (4 features)

- **`plakat doctor --json`**. Structured CI / scripting output
  alongside the v0.18 health-check sections. Covers build /
  runtime device match, libcuda driver shim probe, HF cache disk
  usage. `jq` consumers can assert `.device.aligned == true` or
  `.cache.severity == "ok"`.
- **`--negative-preset photo | painting | anime | cinematic`**.
  Four bundled negative-prompt presets. Combine with `--negative`
  for preset-plus-user-extras. Saves the daily-driver
  `"blurry, low quality, watermark, ..."` copy-paste.
- **`plakat clone PNG`**. Reverse of `plakat metadata` (v0.18):
  reads a generated PNG's recipe + emits the `plakat generate`
  shell command that would re-create it. JSON sidecar preferred
  for lossless reproduction; falls back to parsing the Auto1111
  `parameters` chunk for Civitai uploads / A1111 outputs.
  `--one-line` for pipes.
- **WebP output format**. `--format png | webp` on
  `plakat generate`. WebP ships ~30% smaller files at perceptually-
  equivalent quality. Trade-off: WebP can't carry the Auto1111
  tEXt chunk (no drag-and-drop into A1111 / Civitai / ComfyUI);
  the JSON sidecar still works, so `plakat metadata` / `plakat
  clone` round-trip on WebP outputs. SD-family pipeline only in
  this release; Flux / SD3 warn and fall back.

### FLUX.1 Kontext composition unlocks (2 features)

- **Kontext + ControlNet**. Lifts the v0.18 phase 2 bail.
  ControlNet residuals (computed per-block from the CN forward on
  noise tokens) get zero-padded along the seq dim for Kontext's
  reference half before being added to the per-block flux
  intermediate state. The reference tokens get no CN contribution
  — they're already conditioning via cross-attention. Unlocks
  "edit this image, preserve the depth/canny structure" workflows.

  ```bash
  plakat generate "make it golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --control-spec 'depth:from=input.png:strength=0.7'
  ```

- **Kontext + Redux**. Lifts the v0.18 phase 2 bail with a RoPE
  budget gate. Total effective attention seq (txt + img + ref + N
  Redux tokens) is computed at dispatch; soft warn at 3500
  positions, hard bail at 4096 with actionable cleanup hints.
  Unlocks "edit this image in the style of these references" —
  Kontext provides the layout, Redux provides the aesthetic.

  ```bash
  plakat generate "the same scene at golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --redux-image style_ref.png:weight=0.5
  ```

### Two new tutorials

- [`SCENARIOS_TUTORIAL.md`](Documentation/Tutorials/SCENARIOS_TUTORIAL.md)
  — batch generation via HJSON. Cross-product expansion, per-task
  overrides, partial-rerun filters, real-world series-production
  examples.
- [`OUTPAINT_TUTORIAL.md`](Documentation/Tutorials/OUTPAINT_TUTORIAL.md)
  — `plakat outpaint INPUT.png`. Per-side flag grammar,
  VAE-snapped dimensions, model choice, iterative-stage workflow.

509 lib tests green; +40 new tests across the cycle.

## What's new in v0.18 — Flux Kontext, SDXL animate, BREAK, local enhancer, polish

The largest single-version cycle yet. Three workstreams plus a
follow-on wave of QoL features and three new tutorials.

### Top picks + round-out (7 phases)

- **A1111 attention syntax on Flux + SD3**. The v0.17 per-token
  weight broadcast (CLIP) now applies to T5-XXL hidden states on
  Flux and to all three penultimate streams on SD3 / SD3.5
  (CLIP-L + CLIP-G + T5). Every Civitai Flux LoRA card already
  embeds `(token:1.4)`-style emphasis in its example prompts;
  these now work as written. Sentencepiece alignment caveat
  documented.
- **SDXL `plakat animate`**. The prompt-morph animator (v0.17)
  extended from SD 1.5 / SD 2.1 to SDXL. Dual CLIP-L + CLIP-G
  hidden lerp, pooled `add_text_embeds` lerp, `add_time_ids`
  micro-conditioning threaded through.
- **Animate frame metadata**. Each `frame-NNNN.png` carries the
  Auto1111 `parameters` PNG tEXt chunk + a JSON sidecar with the
  lerp `t` parameter + `Animate from` / `Animate to` extras.
- **LCM-LoRA SD 1.5 `--fast` preset**. Same recipe as v0.17's
  `lcm-sdxl` against the smaller backbone — `--fast lcm-sd15` for
  4-step inference on SD 1.5 hardware.
- **`--grid` on `img2img` / `portrait` / `outpaint`**. The v0.17
  grid bundling now works on every `--count`-bearing subcommand.
  Per-backbone filename prefix preserved.
- **`--negative` attention verification**. Tests confirming the
  per-token weight broadcast works on the uncond branch across
  SD 1.5 / 2.1, SDXL, SD3.
- **`plakat doctor` enhancements**. Build / runtime device match,
  `libcuda.so.1` driver shim probe (Linux + `--features cuda`),
  HF cache disk usage report. Catches the CI-style "binary built
  with CUDA, no driver on host" silent fallback.

### FLUX.1 Kontext (BFL image editing)

Four phases bringing BFL's image-editing Flux variant online:

- **`--model flux-kontext-dev`** on `plakat generate` and
  `plakat img2img`. Reference image is VAE-encoded and
  sequence-concatenated onto the noise tokens (with
  `img_ids[..., 0] = 1` as the RoPE marker) — distinct mechanism
  from Canny/Depth which widen `img_in` to 128 channels.
- **`--concept-image PATH`** reused as the reference flag (same
  grammar as Canny/Depth, semantically the "image to edit"). On
  `plakat img2img`, the input positional becomes the reference
  natively.
- **GGUF support** via `unsloth/FLUX.1-Kontext-dev-GGUF`
  (`--model flux-kontext-dev-gguf`). Composes with LoRA (Kontext
  shares Dev's transformer layer names) and `--quantize-t5`.
- **`--kontext-bucket`** opt-in flag — snaps `--size` to the
  closest of 17 BFL-recommended Kontext resolutions before VAE
  encoding (off by default, surprise-free for non-Kontext flows).

### Follow-on wave (6 features)

- **`plakat metadata FILE.png`**. New subcommand reads the v0.17
  `parameters` PNG tEXt chunk + JSON sidecar back into the
  terminal. `--json-only` pipes cleanly to `jq`.
- **`--aspect`** on `plakat img2img`. Resolution priority:
  `--size > --aspect + --base > input image dims`. Composes with
  `--kontext-bucket`.
- **`plakat scenario --dry-run` polish**. The summary line now
  reads `(dry-run) would have generated …` instead of `✓ done`,
  and per-task previews show the output directory path so you can
  see file layout before launching a long batch.
- **A1111 inline `<lora:NAME[:weight]>` syntax**. Civitai LoRA
  cards embed these directly; plakat extracts them at the CLI
  boundary, parses via the v0.17 `LoraSpec` grammar (paths /
  HF repos / `civitai:NNN` shorthand), prepends to the LoRA
  stack, removes from the prompt before encoding.
- **`BREAK` keyword in prompts**. A1111 convention for chunking
  past CLIP's 77-token cap. Each chunk gets its own 77-token
  CLIP context; hidden states sequence-concatenate before
  cross-attention. SD 1.5 / 2.1 / SDXL; Flux + SD3 strip + warn
  (their T5 already has a 256/512-token budget).
- **Local prompt enhancer**. `--enhance local` runs a small
  instruction-tuned LLM in-process via candle's quantized
  backends. Qwen2.5-1.5B-Instruct (Q4_K_M, ~1 GB) as default,
  SmolLM2-360M (~230 MB) as CPU-budget fallback. Greedy decoding
  for reproducibility; `--enhance auto` picks DeepSeek → Gemini →
  local based on what env vars are set. No API key required for
  the local arm.

### Three new tutorials

- [`ADVANCED_PROMPTING_TUTORIAL.md`](Documentation/Tutorials/ADVANCED_PROMPTING_TUTORIAL.md)
  — attention syntax, BREAK, inline `<lora:>` as a coherent set.
- [`PROMPT_ENHANCER_TUTORIAL.md`](Documentation/Tutorials/PROMPT_ENHANCER_TUTORIAL.md)
  — `--enhance deepseek / gemini / local / auto`.
- [`METADATA_TUTORIAL.md`](Documentation/Tutorials/METADATA_TUTORIAL.md)
  — recovering recipes from PNG metadata.

465 lib tests green; +84 new tests across the cycle.

## What's new in v0.17 — the prompt + reproducibility release

Ten phases focused on **prompt expressiveness**, **reproducibility**,
and **animation**. The cycle also upgrades the underlying candle ML
framework two minor versions and adds the long-asked `--lora civitai:`
shorthand:

- **A1111 prompt syntax**. `(red:1.4)` emphasis / `[blue]`
  de-emphasis / `((nested))` compounding / `\(escape\)` — the
  grammar every Civitai LoRA card uses in its example prompts.
  Applied to CLIP penultimate hidden states via per-token broadcast.
  SD 1.5 / SD 2.1 / SDXL.
- **PNG metadata + JSON sidecar**. Outputs ship with the
  Auto1111-compatible `parameters` PNG tEXt chunk + a sibling
  `<filename>.json` carrying the full recipe. A1111 / Civitai /
  ComfyUI / sd-prompt-reader all surface the prompt + seed + LoRAs
  + scheduler inline. `--no-metadata` opts out.
- **`--grid` output**. `--count N > 1` + `--grid` writes a single
  `plakat-grid-<seed>.png` combining all N outputs in a near-square
  layout. `--grid-cols` / `--grid-padding` for fine control.
- **`plakat animate`**. New subcommand for prompt-morph animation:
  lerp CLIP embeddings between two prompts at a fixed seed,
  producing a smooth N-frame sequence. `--gif` bundles into an
  animated GIF. SD 1.5 / SD 2.1.
- **Live preview during denoise**. `--preview-every N` writes a
  cheap latent-projection PNG every N steps so long runs aren't a
  black box. Microseconds per write — no meaningful runtime cost.
- **scenario `--resume` / `--force`**. Crashed scenario picks up
  where it left off by probing for already-existing output PNGs.
  No more restart-from-task-0.
- **`--lora civitai:NNNNNN`**. Skip the explicit
  `plakat civitai download` step — the LoRA spec parser now
  downloads + caches Civitai assets on first use via the
  shorthand. `civitai-version:NNNNNN` pins a specific version.
- **LCM-LoRA SDXL `--fast lcm-sdxl`**. Latent-Consistency
  distillation for SDXL bundled with the right scheduler and
  4-step / CFG-1.5 defaults. ~5× speedup over stock SDXL.
- **candle 0.8 → 0.10.2 upgrade**. Single 8-line trait-impl fix
  for `SimpleBackend::get_unchecked`. GGUF / NF4 / MMDiT /
  vendored Flux all intact, 304 tests still green at upgrade
  time.
- **SDXL refiner cleanup**. The "Known limitation" about missing
  `add_embedding` on the refiner was outdated since v0.11 phase
  8e. Stale docs replaced; regression tests pin the 5-time-id
  config so future refactors can't silently break the refiner's
  `text_time` micro-conditioning.

381 lib tests green; +77 new tests across the cycle.

## What's new in v0.16 — the productivity release

A dozen quality-of-life landings that connect community workflows
(Civitai browsing, ADetailer face fix, Hires fix, wildcards) to the
existing plakat backbone, plus deeper SD3 integration:

- **SD3 ControlNet (InstantX)**. `--control-spec` works on SD3 /
  SD3.5 via the InstantX adapter family. Multi-CN composition,
  step-gating, auto-annotation from a reference photo — same
  ergonomics SDXL + Flux ControlNet ship.
- **Tiled Flux Fill**. `--tiled` composes with Flux.1-Fill-dev for
  4K+ inpaint. Per-tile masked-latent + mask packing.
- **Tiled SD3 img2img + inpaint**. The rectified-flow init lerp +
  RePaint mask blend compose with the per-tile Hann blend.
- **Wildcards**. `{red|blue|green}` inline alternation +
  `__name__` file wildcards (Auto1111 / NovelAI grammar). Seeded
  from `--seed` for reproducibility.
- **CLIP-skip**. `--clip-skip N` for SD 1.5 / SD 2.1 — N=2 is the
  community default for anime checkpoints.
- **ADetailer-style face refinement**. `--adetailer` runs SCRFD
  on each output, crops + img2img-refines each face, feather-
  composites back. Reuses the t2i SdCore — no extra model load.
- **Hires fix**. `--hires-fix` escapes the trained-resolution
  ceiling: upscale (Lanczos / Real-ESRGAN) + img2img-refine.
  Composes with `--adetailer` for a 4K → fixed faces pipeline.
- **Civitai browser + downloader**. `plakat civitai search`,
  `info`, `download` — drop the resulting path into `--lora` /
  `--model`. Atomic streaming downloads with cache-hit
  short-circuit.
- **Auto-annotation for Flux concept variants**. `--concept-from
  PATH` auto-annotates a photo through Canny / Depth before feeding
  Flux.1-Canny-dev / Flux.1-Depth-dev.
- **SD3 pipeline caching + per-task LoRA**. Scenarios with
  `--model sd35-*` now share one SD3 pipeline across tasks; per-
  task `loras:` swap at runtime via the LoraLinear stack.
- **Textual Inversion** *(partial)*. Parser + `plakat embedding
  info` inspector. Runtime injection blocked by candle 0.8's
  private `clip::Config.vocab_size` — wiring lands when the
  candle API surface opens or alongside a vendored CLIP path.
- **SD UNet per-task LoRA preflight** *(partial)*. Detects the
  blocker upfront and emits actionable YAML-fold hints; bails
  loud with three concrete workarounds. Full UNet vendoring
  deferred — same candle private-internals blocker.
- **XLabs Flux IP-Adapter parser** *(partial)*. Inspector that
  reports per-block attention count + SigLIP/Flux dims. Per-block
  injection blocked by Flux's private `double_block_forward`;
  use `--redux-image` for working image conditioning today.

## What's new in v0.15 — runtime LoRA + SD3 maturation

- **Per-task LoRA in scenarios**. `tasks: [{ loras: [...] }]` applies
  and clears LoRAs between tasks at runtime — no model reload.
  Composes with the scenario-level LoRA set. Flux (BF16 / GGUF / NF4).
- **NF4 + ControlNet**. NF4 Flux composes with `--control-spec` via
  the residual-aware forward — same residual interleave the BF16 and
  GGUF backbones use, so a single CN checkpoint works on all three.
- **SD3 / SD3.5 img2img + inpaint**. RePaint-style inpaint with
  per-step mask blend, rectified-flow truncated schedule. Works
  across the lineup (Medium / Large / Turbo).
- **SD3 / SD3.5 LoRA**. Diffusers PEFT format, MMDiT-targeted keys.
- **Flux Canny-dev / Depth-dev variants**. BFL "concept" Flux
  checkpoints with conditioning baked into the 128-channel `img_in`.
  Pass `--concept-image PATH` with `--model flux-canny-dev`.
- **Tiled SD3**. MultiDiffusion-style tiled denoise for MMDiT —
  1024-px tiles work on every SD3 variant within the variant's
  `pos_embed_max_size` cap.
- **Scenario ↔ generate sync**. Per-task `fast`, `concept-image`,
  `enhance`, `tiled` overrides.
- **Two new tutorials**:
  [`FLUX_TUTORIAL.md`](Documentation/Tutorials/FLUX_TUTORIAL.md)
  walks through the Flux feature set end-to-end;
  [`SD3_TUTORIAL.md`](Documentation/Tutorials/SD3_TUTORIAL.md) does
  the same for the SD3 / SD3.5 family.

## What's new in v0.14 — the SD3.5 + NF4 + Redux release

- **Stable Diffusion 3 / 3.5 (MMDiT)**. New family — `sd35-medium`,
  `sd35-large`, `sd35-large-turbo`, `sd3-medium`. Triple text encoder
  (CLIP-L + CLIP-G + T5-XXL), 16-channel VAE, rectified-flow sampler
  with SD3 time-shift. CFG via `[neg, pos]` double-batch.
- **NF4 quantized Flux**. `--model flux-dev-nf4` loads lllyasviel's
  bitsandbytes NF4 pack — ~6 GB transformer at inference (4× weight
  savings vs BF16), pure-CPU dequant codec means it runs on any
  candle device. Phase 8b adds **NF4 + LoRA composition** via the
  same selective-dequant trick GGUF uses.
- **Flux Redux**. `--redux-image PATH` adds image conditioning via
  SigLIP-so400m + BFL's Redux adapter (729 tokens → seq-concat onto
  T5). Repeatable for multi-image stacks (`--redux-image
  style.png:weight=0.8 --redux-image subject.png:weight=0.5`). Cap
  of 4 with attention-cost guardrails. Composes with GGUF, NF4,
  LoRA, ControlNet, img2img, tiled.
- **Tiled SD 1.5 / 2.1**. `--tiled` now supported on the smaller
  SD backbones too (was SDXL-only in v0.12).
- **Flux Fill + ControlNet**. `plakat img2img --model flux-fill-dev
  --mask ... --control-spec depth:from=...` composes with the
  auto-annotator and multi-CN.
- **Hyper-FLUX / FLUX-Turbo presets**. `--fast hyper-8 | hyper-16 |
  turbo-alpha` bundles the matching distillation LoRA + recommended
  step count + guidance in one flag.
- **Shared SdCore**. Scenarios with mixed t2i + img2img tasks now
  load the SD backbone **once** (was: per-task). The t2i Pipeline's
  `Arc<SdCore>` is reused by img2img via the existing `from_core`
  path.

## What's new in v0.13 — the Flux modernization release

- **Quantized Flux (GGUF)**. Run FLUX.1-dev on 16 GB GPUs.
  `--model flux-dev-gguf` loads the 4-bit transformer (~7 GB vs ~24 GB BF16).
  `--quantize-t5` drops T5-XXL to ~3 GB. `--quant-level Q5_K_M` picks a
  different precision (Q2_K..F16 supported); same for `--t5-quant-level`.
- **Flux LoRA on quantized**. Diffusers PEFT and AI-Toolkit / kohya
  formats both compose with the GGUF backbone — affected Linears are
  dequantized once at load, rest of the model stays 4-bit.
- **Flux Inpainting**. `--model flux-fill-dev` + `--mask` runs BFL's
  dedicated 384-channel inpaint checkpoint via `plakat img2img`.
- **Flux Img2Img**. Rectified-flow init: `plakat img2img init.png
  --model flux-dev --strength 0.7 --prompt "..."`.
- **Tiled Flux denoise**. MultiDiffusion-style 2K–4K outputs on any
  Flux variant: `--tiled --tile-size 1024 --tile-stride 768`. Composes
  with ControlNet (per-tile residuals) and the tiled VAE decode.
- **Flux ControlNet polish**. Auto-annotators wire through to Flux
  (`--control-spec depth:from=photo.jpg` is now a one-liner). Step gating
  via `start=…:end=…`. Multi-Flux-CN with summed residuals.
- **Outpainting**. New `plakat outpaint` subcommand expands a canvas
  and hands off to the inpaint pipeline (SDXL-Inpaint, SD15-Inpaint, or
  Flux.1-Fill-dev).
- **Scenarios**. Every v0.13 feature above is now expressible in
  scenario HJSON: `quant-level:`, `t5-quant-level:`, `tiled:`,
  per-task `init-image:` / `mask:` / `strength:` / `outpaint:`, plus
  multi-CN via `controls: [...]`.

