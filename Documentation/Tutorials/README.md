# plakat tutorials

Beginner-friendly, step-by-step walkthroughs of plakat's main
features. No prior text-to-image experience assumed. Each tutorial
explains the *why* alongside the *how*.

For exhaustive flag-by-flag reference material, see the parent
[`Documentation/`](..) directory.

## Recommended reading order

If you're new to plakat, work through these in order:

0. [`GETTING_STARTED.md`](GETTING_STARTED.md) — **the fastest path
   from a fresh checkout to a rendered image** (v0.20). Uses
   `plakat init` to bootstrap a runnable starter project + walks
   through dry-run → first render → iteration. Zero HF token, zero
   API keys. If you want the "just show me it works" experience
   before reading flag references, start here.

1. [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md) — **the foundation.**
   Your first generation, the flags that matter, seeds for
   reproducibility, prompt wildcards, CLIP-skip, ADetailer face
   refinement, Civitai browser + downloader, Hires fix, Textual
   Inversion inspector, and moving from one-off CLI commands to
   batch scenarios. The foundation everything else builds on.

2. [`PORTRAIT_TUTORIAL.md`](PORTRAIT_TUTORIAL.md) — making portraits,
 including identity preservation (rendering a specific person from
 a reference photo), putting portraits into broader scenes via
 scenarios, and multi-persona compositions.

3. [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md) — applying art styles
 to your generations: pick by name, detect from a reference photo,
 combine styles with portraits, use styles in scenarios.

4. [`HOW_TO_CREATE_MY_OWN_STYLE.md`](HOW_TO_CREATE_MY_OWN_STYLE.md) —
 build your own style catalog from a folder of images. Covers the
 end-to-end pipeline (organize → init → build → use) and adding
 LoRAs to make detection turn into real style transfer.

5. [`ARTEFACTS_TUTORIAL.md`](ARTEFACTS_TUTORIAL.md) — composite named
 PNG cutouts (trees, sky elements, houses) into named zones of
 your generated images. Useful when you need specific objects in
 specific places, or for consistent visual elements across many
 scenes. The hands-on runnable companion lives at
 [`examples/tutorials/ZONES/`](../../examples/tutorials/ZONES/)
 (seven shell scripts + an HJSON scenario, end-to-end).

6. [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md) — transform an existing
 image with a prompt (img2img), or repaint just a masked region
 (inpaint). Same `plakat img2img` subcommand drives both modes —
 adding `--mask PATH` flips img2img into inpaint. The hands-on
 companion lives at
 [`examples/tutorials/IMG2IMG/`](../../examples/tutorials/IMG2IMG/)
 (four shell scripts + a procedurally-drawn sample input + mask).

7. [`CONTROLNET_TUTORIAL.md`](CONTROLNET_TUTORIAL.md) — add
 structural guidance to any generation. ships two
 conditioners (depth + canny) on both SD 1.5 and SDXL. Three
 ways to supply the conditioning image: `--control-image PATH`
 (pre-rendered), `--control-from PATH` (auto-annotate any
 image), or — on `plakat img2img` — let the source image
 annotate itself by default. Composes cleanly with generate /
 portrait / img2img / scenarios. The hands-on companion lives at
 [`examples/tutorials/CONTROL/`](../../examples/tutorials/CONTROL/)
 (six shell scripts + a procedurally-drawn sample depth map).

8. [`FLUX_TUTORIAL.md`](FLUX_TUTORIAL.md) — Black Forest Labs'
   Flux family. Covers quantization (GGUF + NF4), LoRA, img2img +
   Fill inpaint (including tiled Flux Fill for 4K+ inpaint),
   ControlNet, tiled hi-res, Redux image conditioning, the
   "concept" variants (Canny-dev / Depth-dev), and the `--fast`
   distillation presets. Memory tiers for picking the right
   backbone on your GPU.

9. [`SD3_TUTORIAL.md`](SD3_TUTORIAL.md) — Stable Diffusion 3 / 3.5
   family. MMDiT architecture, the four variants (SD3 / SD3.5
   Medium / SD3.5 Large / SD3.5 Large Turbo), the rectified-flow
   sampler, LoRA, img2img + RePaint-style inpaint, tiled hi-res
   (including tiled img2img / inpaint), and SD3 ControlNet via
   the InstantX adapter family.

10. [`CIVITAI_TUTORIAL.md`](CIVITAI_TUTORIAL.md) — browsing,
    downloading, and using Civitai community assets from the
    command line. `plakat civitai search` / `info` / `download`,
    pairing downloaded LoRAs with the right base model, the
    cache layout, `CIVITAI_API_KEY` setup for gated assets, and
    inspecting downloaded Textual Inversion files.

11. [`ANIMATE_TUTORIAL.md`](ANIMATE_TUTORIAL.md) — prompt-morph
    animations via `plakat animate`. Linearly interpolate the
    text-encoder embeddings between two prompts to produce a
    smooth N-frame sequence at a fixed seed. Optional GIF
    bundling. SD 1.5 / SD 2.1 / SDXL + **Flux Dev / Schnell**
    (v0.20 — T5 + CLIP-L lerp, flow-match per frame).

12. [`ADVANCED_PROMPTING_TUTORIAL.md`](ADVANCED_PROMPTING_TUTORIAL.md) —
    three power-user prompt features as a coherent set: A1111
    attention syntax `(red:1.5)` / `[blue]`, the `BREAK` keyword
    for chunking past CLIP's 77-token cap, and inline `<lora:>`
    tags that load LoRAs directly from the prompt. Per-backbone
    composition matrix; all three work together. v0.17 + v0.18 +
    v0.18.

13. [`PROMPT_ENHANCER_TUTORIAL.md`](PROMPT_ENHANCER_TUTORIAL.md) —
    `--enhance` to let an LLM rewrite your prompt with concrete
    visual detail before generation. Three providers: DeepSeek /
    Gemini (API-keyed) and `local` (v0.18, Qwen2.5-1.5B by
    default, ~1 GB GGUF, no API key). `--enhance auto` picks
    based on what's available.

14. [`METADATA_TUTORIAL.md`](METADATA_TUTORIAL.md) — `plakat
    metadata FILE.png` reads back the v0.17 Auto1111-compatible
    `parameters` PNG tEXt chunk + sibling JSON sidecar. Recover
    forgotten seeds, inspect Civitai downloads, audit scenario
    batches. `--json-only` pipes cleanly to jq. v0.19 adds the
    companion `plakat clone PNG` that translates a recipe into
    a re-runnable shell command.

15. [`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md) — batch
    generation via HJSON config files. Cross-product expansion
    (scene × weather × persona), per-task overrides, per-task
    LoRA stacks, partial-rerun filters (`--resume`, `--only`,
    `--limit`, `--dry-run`), real-world series-production
    examples. The power-user feature that turns plakat from a
    one-shot CLI into a job runner.

16. [`OUTPAINT_TUTORIAL.md`](OUTPAINT_TUTORIAL.md) — `plakat
    outpaint INPUT.png` grows the canvas of an existing image.
    Per-side flag grammar (`--left` / `--right` / `--top` /
    `--bottom` / `--expand`), VAE-snapped dimensions, model
    choice (`sdxl-inpaint` / `sd15-inpaint` /
    `flux-fill-dev`), iterative-stage workflow.

## Specialized portrait techniques

After the foundational portrait tutorial, these dive into specific
creative applications of plakat's weighted multi-reference portrait
feature:

- [`PORTRAIT_HOW_TO_AGE.md`](PORTRAIT_HOW_TO_AGE.md) — interpolate a
 person across ages using photos of the same person at different
 ages and weighted merging. Render plausible portraits at any
 intermediate age.

- [`PORTRAIT_CHILD_PHOTO.md`](PORTRAIT_CHILD_PHOTO.md) — blend two
 parent photos into a plausible child portrait. Combines identity-
 space merging with age-appropriate prompt cues to produce "average
 child" or "looks more like X" variants.

## What each tutorial assumes

| Tutorial | Prerequisites |
|---|---|
| `GETTING_STARTED.md` | plakat installed; can run `plakat --help`. |
| `GENERATE_TUTORIAL.md` | plakat installed; can run `plakat --help`. |
| `PORTRAIT_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. |
| `STYLES_TUTORIAL.md` | Above + finished `PORTRAIT_TUTORIAL.md` (helpful but not required). |
| `HOW_TO_CREATE_MY_OWN_STYLE.md` | Above + finished `STYLES_TUTORIAL.md`. Plus a corpus of images you want to teach plakat. |
| `PORTRAIT_HOW_TO_AGE.md` | Above + finished `PORTRAIT_TUTORIAL.md`. Plus 2-4 photos of the same person at different ages. |
| `PORTRAIT_CHILD_PHOTO.md` | Above + finished `PORTRAIT_TUTORIAL.md`. Plus one head-shot per parent. |
| `CIVITAI_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. Plus an internet connection. An optional `CIVITAI_API_KEY` if you want gated assets. |
| `ARTEFACTS_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. (No external assets required — uses the bundled placeholder set.) |
| `FLUX_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. GPU with ≥12 GB VRAM (NF4) / ≥16 GB (GGUF) / ≥24 GB (BF16). HuggingFace token for gated `flux-dev` / `flux-fill-dev` repos. |
| `SD3_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. GPU with ≥12 GB VRAM (Medium) / ≥24 GB (Large). HuggingFace token — all Stability SD3 repos are gated. |
| `ANIMATE_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. ~3 GB free for SD 1.5 weights, ~7 GB for SDXL. |
| `ADVANCED_PROMPTING_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. Comfortable with `--lora` and the relationship between seed + reproducibility. No new assets required. |
| `PROMPT_ENHANCER_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. ~1 GB free for the default Qwen2.5-1.5B GGUF (or ~230 MB for the SmolLM2 fallback). API key optional (only needed for `--enhance deepseek` / `--enhance gemini`). |
| `METADATA_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. One or more PNGs from a previous plakat / A1111 / Civitai run. |
| `SCENARIOS_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md` and (for the persona walkthrough) `PORTRAIT_TUTORIAL.md`. Optional API key for `enhancer: deepseek` / `gemini`. |
| `OUTPAINT_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md` and `IMG2IMG_TUTORIAL.md` (outpaint is a thin wrapper around the inpaint flow). An input image to extend. |

## When to use a reference manual instead

Tutorials are best when you're learning a feature or want a clear
end-to-end example. When you already know the feature and just need
to look up a specific flag or schema field, read the reference
manuals in [`Documentation/`](..):

- Looking up a specific `plakat generate` flag → [`GENERATE.md`](../GENERATE.md)
- Identity strategies and ArcFace setup → [`PERSONA.md`](../PERSONA.md)
- Style catalog JSON schema → [`STYLES.md`](../STYLES.md)
- Artefact compositing, smart zones, blend pass → [`ARTEFACTS.md`](../ARTEFACTS.md)
- Image-to-image and inpaint flags → [`IMG2IMG.md`](../IMG2IMG.md)
- ControlNet conditioning flags → [`CONTROLNET.md`](../CONTROLNET.md)
- Apple chip + memory tiers, expected speeds → [`APPLE_REQUIREMENTS.md`](../APPLE_REQUIREMENTS.md)
