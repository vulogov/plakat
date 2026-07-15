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

0a. [`UI_TUTORIAL.md`](UI_TUTORIAL.md) — **`plakat ui`, the interactive
   terminal UI.** Load a model once and *talk* to it: conversational
   generation + refinement, browse your history, drop in a specific
   person, paint an inpaint mask, search/apply LoRAs, compile prose
   into scenarios — eight screens, all keyboard-driven, on the same
   engine as the CLI. The friendliest way to explore everything below.

0b. [`PHOTOS_TUTORIAL.md`](PHOTOS_TUTORIAL.md) — **`plakat photos`, the
   photo & image collection manager (the 3.x flagship).** Browse a folder
   tree of images (RAW + every common format, EXIF), curate
   non-destructively (ratings, flags, colour labels, tags in a plain
   per-album `album.hjson`), filter and cull fast — and close the loop with
   `--import`, which lands anything you generate straight into an album with
   its full recipe. On by default.

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

3a. [`LOOKS_TUTORIAL.md`](LOOKS_TUTORIAL.md) — **v0.25.** Art-medium
 presets via `--look NAME`. Eight bundled mediums (ink-wash,
 watercolor, oil-painting, charcoal, pencil, chalk-pastel, linocut,
 gouache). Auto-LoRA discovery from Civitai → HF → local cache.
 Override-only sampler / steps / guidance. Composes with `--style`,
 `--fast`, `--lora`, `--genre`. User-extension directory for your
 own mediums.

3b. [`GENRES_TUTORIAL.md`](GENRES_TUTORIAL.md) — **v0.25.** Subject-
 domain presets via `--genre NAME`. Independent axis from `--look`.
 Bundled `anime` only; user-extension directory for photoreal /
 fantasy / cyberpunk / etc. Same discovery + override semantics.

3c. [`STYLIZE_TUTORIAL.md`](STYLIZE_TUTORIAL.md) — **v0.46–47.** Apply a
 reference image's look to a subject via IP-Adapter (`plakat stylize`).
 Default = ref-*variation* (content/palette); **`--instantstyle` (v0.47,
 SDXL)** = true painterly STYLE transfer (texture, via decoupled style-block
 injection). `--ref-blur` / `--ref-weight` / `--style-scale` knobs.

3d. [`EMBEDDING_TUTORIAL.md`](EMBEDDING_TUTORIAL.md) — **Textual Inversion.**
 Inject a TI embedding at generation time (`generate --embedding
 REPO#FILE:trigger`) — EasyNegative & friends, no training. SD 1.5 / SDXL.

4. [`HOW_TO_CREATE_MY_OWN_STYLE.md`](HOW_TO_CREATE_MY_OWN_STYLE.md) —
 build your own style catalog from a folder of images. Covers the
 end-to-end pipeline (organize → init → build → use) and adding
 LoRAs to make detection turn into real style transfer.

4a. [`TRAIN_STYLE_LORA_TUTORIAL.md`](TRAIN_STYLE_LORA_TUTORIAL.md) —
 **v0.45.** The *creation* companion to the catalog tutorial: `plakat
 style train` learns a style from a folder of images into a LoRA that
 actually **paints** in that style (not just detects it). Phase 1
 trains on SD 3.5. Covers corpus requirements, the train→generate
 split, checkpointing, and tuning strength at render time with the
 `--lora …:scale` suffix. Worked watercolour example in `corpus/`.

5. [`ARTEFACTS_TUTORIAL.md`](ARTEFACTS_TUTORIAL.md) — composite named
 PNG cutouts (trees, sky elements, houses) into named zones of
 your generated images. Useful when you need specific objects in
 specific places, or for consistent visual elements across many
 scenes. The hands-on runnable companion lives at
 [`examples/tutorials/ZONES/`](../../examples/tutorials/ZONES/)
 (seven shell scripts + an HJSON scenario, end-to-end).

5a. [`TRANSPARENT_TUTORIAL.md`](TRANSPARENT_TUTORIAL.md) — **smart cut-out.**
 `plakat transparent --matte` removes the background with content-aware U2Net
 matting (any background → clean RGBA); chroma-key fallback for studio shots.
 Builds the artefact cutout library.

5b. [`COMPOSE_TUTORIAL.md`](COMPOSE_TUTORIAL.md) — **v1.0. compose layered
 scenes.** `plakat compose <scene.hjson>` stacks a background + cut-outs by
 z-order, position (9-grid / `x,y`), scale, and opacity. No GPU — composes
 existing assets. Pairs with `transparent` (matte the cut-outs first).

5c. [`SEGMENT_TUTORIAL.md`](SEGMENT_TUTORIAL.md) — **v1.0. click to select.**
 `plakat segment --point X,Y` masks an object via MobileSAM (`--grow`/`--feather`
 for clean inpaint edges); feed the mask to `img2img --mask` for object removal /
 background swap. The compose-&-edit enabler.

5d. [`REGIONAL_TUTORIAL.md`](REGIONAL_TUTORIAL.md) — **v1.0. regional prompting.**
 `plakat generate "<base>" --region "x0,y0,x1,y1:prompt"` — different prompts in
 different regions of one image (SD 1.5 / SDXL / SD3.5; also a scenario `regions`
 key). Feathered blends, not seams.

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

9a. [`CASCADE_TUTORIAL.md`](CASCADE_TUTORIAL.md) — **Stable Cascade**
    (Würstchen v3). The three-stage pipeline and its two step budgets
    (`--stage-c-steps` / `--stage-b-steps`), prior vs decoder guidance
    (`--decoder-guidance`, v0.42), LoRA / **DoRA** (kohya + PEFT),
    **image variation** (`--image-variation`) and **faithful img2img**
    (`--faithful`) via the CLIP ViT-L/14 encoder (v0.42), the canny
    ControlNet, and driving Cascade from Bund scripts. The most
    memory-efficient high-quality model — 1024² on ~16 GB.

9b. [`VARIATION_TUTORIAL.md`](VARIATION_TUTORIAL.md) — **image variation.**
    `generate --image-variation REF` re-imagines a reference from its CLIP
    embedding (unCLIP-style, Cascade) — keeps semantics, re-composes. Pure
    (empty prompt) or steered.

10. [`CIVITAI_TUTORIAL.md`](CIVITAI_TUTORIAL.md) — browsing,
    downloading, and using Civitai community assets from the
    command line. `plakat civitai search` / `info` / `download`,
    pairing downloaded LoRAs with the right base model, the
    cache layout, `CIVITAI_API_KEY` setup for gated assets, and
    inspecting downloaded Textual Inversion files.

11. [`ANIMATE_TUTORIAL.md`](ANIMATE_TUTORIAL.md) — `plakat animate`.
    Two modes:
    - **Prompt-morph** (v0.20): lerp text-encoder embeddings between
      `--from` and `--to` for a smooth N-frame morph. SD 1.5 /
      SD 2.1 / SDXL + Flux Dev / Schnell + SD3 / SD3.5 (v0.26).
    - **AnimateDiff** (**v0.28 productivity polish**): motion-coherent
      N-frame generation from a single prompt via the V3 / SDXL beta
      / AnimateLCM motion adapters. v0.27: SD 1.5 + SDXL inference,
      single ControlNet, sliding-window long-form past V3's 32-frame
      cap. v0.28: **`--lcm`** for 4-step AnimateLCM generation
      (~5× speedup), **`--control-spec`** for multi-CN stacking,
      **`plakat motion-adapter info / list`** for adapter inspection,
      and **`plakat.animate`** for the Bund scripting bridge.
      `--motion-lora` stacking, `--window-size` / `--window-overlap`
      for long-form, and `--format {gif,mp4,webm,frames,all}`.
    See also [`Documentation/ANIMATEDIFF.md`](../ANIMATEDIFF.md)
    for the AnimateDiff architecture reference.

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
    examples. **v0.29 §9 adds animate scenarios** (`type:
    animatediff` tasks with per-task `frames` / `lcm` /
    `motion-lora` / `format` overrides) — the same batch driver
    now produces N-frame motion-coherent clips alongside single
    images. The power-user feature that turns plakat from a
    one-shot CLI into a job runner.

15a. [`COMPILE_TUTORIAL.md`](COMPILE_TUTORIAL.md) — **`plakat compile`.**
    Write a batch as prose paragraphs (a `prompts.txt`) and compile it into a
    scenario HJSON: one task per blank-line block, global→scene inheritance,
    model-family-aware prompt rewriting + auto-negatives. `--no-enhance` for a
    deterministic core, `--lint` / `--dry-run` / `--diff` / `--decompile`, an
    optional Tera template pre-pass, and a `type: map` block (worldbuilding +
    maps in one document). The prose front-end to `SCENARIOS_TUTORIAL.md`.

16. [`OUTPAINT_TUTORIAL.md`](OUTPAINT_TUTORIAL.md) — `plakat
    outpaint INPUT.png` grows the canvas of an existing image.
    Per-side flag grammar (`--left` / `--right` / `--top` /
    `--bottom` / `--expand`), VAE-snapped dimensions, model
    choice (`sdxl-inpaint` / `sd15-inpaint` /
    `flux-fill-dev`), iterative-stage workflow.

16a. [`UPSCALE_TUTORIAL.md`](UPSCALE_TUTORIAL.md) — `plakat upscale`
    enlarges an image: classical resampling (Lanczos, instant) or
    **Real-ESRGAN** ML super-resolution (`--method real-esrgan-x2/x4`).
    The Metal ×4 OOM → `--device cpu` knob.

17. [`SCRIPTING_TUTORIAL.md`](SCRIPTING_TUTORIAL.md) — **v0.21.**
    Drive plakat from a Bund script (`plakat run SCRIPT.bund`).
    Stack-based syntax (Forth-flavoured), the seven `plakat.*`
    host words (load + generate + img2img + portrait + upscale +
    save + config.set), handle reuse for `generate → upscale →
    save` chains, the interactive REPL (`plakat run --repl`),
    composition patterns + limitations.

18. [`UTILITIES_TUTORIAL.md`](UTILITIES_TUTORIAL.md) — the small commands
    around the generators: `doctor` (health-check), `models` (cache),
    `inspect` (.safetensors tensors), `gallery` (Markdown index), `clone`
    (PNG → re-runnable command), `init` (scaffold a project), and
    `motion-adapter` (AnimateDiff inspection).

19. [`MAP_TUTORIAL.md`](MAP_TUTORIAL.md) — **`plakat map`.** Turn prose into a
    fantasy map: a coastline, mountains, rivers, biomes, towns, roads, and
    labelled landmarks, all a pure function of (spec, seed). Linework styles
    (`--map-style`), SD-painted maps (`--map-render-sd`), tunable erosion
    (`--map-erosion`) + town street plans (`--map-urban-layout`), vector export
    (GeoJSON/SVG), non-Latin labels (`--map-font`), and the `scenario` / `compile`
    / scripting integration. No GPU for the linework path. **v1.11.0** adds
    HJSON specs, dry canyons + plateaus/mesas, the political layer (polity rings
    + borders), seasonal palettes (`--map-season`), and a tabletop grid
    (`--map-grid`).

20. [`RELIGHT_TUTORIAL.md`](RELIGHT_TUTORIAL.md) — **v1.11.0. `plakat relight`.**
    IC-Light re-illuminates a foreground subject under a lighting you describe in
    text, keeping the subject's identity while changing the light + scene. SD
    1.5-based (4→8-channel UNet, `lllyasviel/ic-light` offset). Wants LOW guidance
    (1.5–3). No reference photo of the light, no training.

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
| `UI_TUTORIAL.md` | plakat installed (default `ui` feature). A graphics-capable terminal (Kitty/Ghostty/WezTerm/iTerm2/Sixel) for inline images; runs without one (placeholders). |
| `PHOTOS_TUTORIAL.md` | plakat installed (default `photos` feature). A graphics-capable terminal for thumbnails/image view; runs without one (placeholders). A folder of images to browse. |
| `GENERATE_TUTORIAL.md` | plakat installed; can run `plakat --help`. |
| `PORTRAIT_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. |
| `STYLES_TUTORIAL.md` | Above + finished `PORTRAIT_TUTORIAL.md` (helpful but not required). |
| `LOOKS_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. First-time use of a look hits Civitai over the network; subsequent runs are network-free (or use `--offline`). |
| `GENRES_TUTORIAL.md` | Above + finished `LOOKS_TUTORIAL.md` (mirrors the look axis). |
| `HOW_TO_CREATE_MY_OWN_STYLE.md` | Above + finished `STYLES_TUTORIAL.md`. Plus a corpus of images you want to teach plakat. |
| `TRAIN_STYLE_LORA_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. A folder of 3+ style images, 24 GB Apple-Silicon/Metal, and a HuggingFace token (SD 3.5 is gated). Budget a couple of hours for training. |
| `PORTRAIT_HOW_TO_AGE.md` | Above + finished `PORTRAIT_TUTORIAL.md`. Plus 2-4 photos of the same person at different ages. |
| `PORTRAIT_CHILD_PHOTO.md` | Above + finished `PORTRAIT_TUTORIAL.md`. Plus one head-shot per parent. |
| `CIVITAI_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. Plus an internet connection. An optional `CIVITAI_API_KEY` if you want gated assets. |
| `ARTEFACTS_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. (No external assets required — uses the bundled placeholder set.) |
| `FLUX_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. GPU with ≥12 GB VRAM (NF4) / ≥16 GB (GGUF) / ≥24 GB (BF16). HuggingFace token for gated `flux-dev` / `flux-fill-dev` repos. |
| `SD3_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. GPU with ≥12 GB VRAM (Medium) / ≥24 GB (Large). HuggingFace token — all Stability SD3 repos are gated. |
| `CASCADE_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. ~16 GB unified memory / VRAM, ~20 GB free disk for weights. No HuggingFace token (ungated). |
| `ANIMATE_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. ~3 GB free for SD 1.5 weights, ~7 GB for SDXL. |
| `ADVANCED_PROMPTING_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. Comfortable with `--lora` and the relationship between seed + reproducibility. No new assets required. |
| `PROMPT_ENHANCER_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. ~1 GB free for the default Qwen2.5-1.5B GGUF (or ~230 MB for the SmolLM2 fallback). API key optional (only needed for `--enhance deepseek` / `--enhance gemini`). |
| `METADATA_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. One or more PNGs from a previous plakat / A1111 / Civitai run. |
| `SCENARIOS_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md` and (for the persona walkthrough) `PORTRAIT_TUTORIAL.md`. Optional API key for `enhancer: deepseek` / `gemini`. |
| `OUTPAINT_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md` and `IMG2IMG_TUTORIAL.md` (outpaint is a thin wrapper around the inpaint flow). An input image to extend. |
| `SCRIPTING_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. Stack-based syntax is unusual but the tutorial assumes no prior Forth experience. |

## When to use a reference manual instead

Tutorials are best when you're learning a feature or want a clear
end-to-end example. When you already know the feature and just need
to look up a specific flag or schema field, read the reference
manuals in [`Documentation/`](..):

- Looking up a specific `plakat generate` flag → [`GENERATE.md`](../GENERATE.md)
- Identity strategies and ArcFace setup → [`PERSONA.md`](../PERSONA.md)
- Style catalog JSON schema → [`STYLES.md`](../STYLES.md)
- Train your own style LoRA (flags + internals) → [`TRAIN_CUSTOM_LORA.md`](../TRAIN_CUSTOM_LORA.md)
- Look (art-medium) presets reference → [`LOOKS.md`](../LOOKS.md)
- Genre (subject-domain) presets reference → [`GENRES.md`](../GENRES.md)
- AnimateDiff architecture + roadmap → [`ANIMATEDIFF.md`](../ANIMATEDIFF.md)
- Artefact compositing, smart zones, blend pass → [`ARTEFACTS.md`](../ARTEFACTS.md)
- Image-to-image and inpaint flags → [`IMG2IMG.md`](../IMG2IMG.md)
- ControlNet conditioning flags → [`CONTROLNET.md`](../CONTROLNET.md)
- Apple chip + memory tiers, expected speeds → [`APPLE_REQUIREMENTS.md`](../APPLE_REQUIREMENTS.md)
