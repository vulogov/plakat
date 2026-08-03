# plakat — reference documentation

This directory holds the full reference manual for plakat. For
beginner-friendly walkthroughs, see [`Tutorials/`](Tutorials/).

## Reference manuals

| File | Covers |
|---|---|
| [`GENERATE.md`](GENERATE.md) | Text-to-image generation. Every flag on `plakat generate`, the model catalog (SD 1.5 / 2.1 / SDXL / SDXL-Turbo / Flux BF16+GGUF+NF4 / Flux Fill / Flux Canny-dev / Flux Depth-dev / **Flux Kontext-dev** (v0.18) / SD3 / SD3.5 Medium / Large / Large Turbo), scheduler reference, refiner pipeline, LoRA stacking (including the `civitai:NNNNNN` auto-resolve shorthand and v0.18 inline `<lora:>` syntax), prompt wildcards, A1111 attention syntax across all backbones (v0.18), `BREAK` keyword chunking (v0.18), CLIP-skip, ADetailer face refinement, Hires fix, Textual Inversion inspector, Flux Redux + concept-variant auto-annotation, **Kontext + ControlNet** + **Kontext + Redux** composition (v0.19), the `--fast` distillation presets (Flux Hyper / Turbo and LCM-SDXL + LCM-SD15), `--grid` output, `--preview-every` live previews, PNG metadata + JSON sidecar (with `--no-metadata` opt-out), `--format png\|webp` (v0.19), `--negative-preset` (v0.19), `--enhance local\|auto\|local:<alias>` + `--enhance-cache` / `-temp` / `-max-tokens` / `-system` (v0.18/v0.19), tiled hi-res, scenarios (HJSON format) with `--resume` / `--only` / `--limit` / `--dry-run`, upscaling, output naming, and the `plakat civitai` / `plakat embedding` / `plakat animate` / `plakat metadata` / `plakat clone` subcommands. |
| [`PERSONA.md`](PERSONA.md) | **v5.0 flagship (RFC PERSONA-1).** `plakat persona` — compose a *specific, reusable synthetic person* from a small HJSON `PersonaSpec` and render that same person recognisably across scenes and model families. The layer model (spec → resolver → geometry → details → calibration → cast → render → verify → repair), the command reference, the identity tiers, and the render-robustness guards. Companions: [`PERSONA_TUTORIAL.md`](PERSONA_TUTORIAL.md), [`PERSONA_CASTING.md`](PERSONA_CASTING.md), [`PERSONA_DETAILS_HOWTO.md`](PERSONA_DETAILS_HOWTO.md), [`PERSONA_LEXICON.md`](PERSONA_LEXICON.md), [`PERSONA_ANCHORS.md`](PERSONA_ANCHORS.md), [`PERSONA_CALIBRATION.md`](PERSONA_CALIBRATION.md), [`PERSONA_GATING.md`](PERSONA_GATING.md), [`RFC_PERSONA_1.md`](RFC_PERSONA_1.md). |
| [`BOOKART.md`](BOOKART.md) | **v6.0 flagship (RFC BOOKART-1).** `plakat bookart` — compose *reusable, print-ready, transparent black-and-white book ornaments* from a small HJSON spec, in a chosen illustration tradition × technique, at an exact page size. The ornament vocabulary, the `BookArtSpec` schema, the hybrid render router (procedural / diffusion / composite), the print/ink scorecard, and the full command reference (`render`/`illustrate`/`kit`/`manuscript`/`diff`/`edit`/`blend`). Companions: [`BOOKART_TRANSPARENCY.md`](BOOKART_TRANSPARENCY.md) (the B/W-native alpha model + print sizing + SVG), [`BOOKART_STYLES.md`](BOOKART_STYLES.md) (origins × techniques + the hosted LoRAs), [`Tutorials/BOOKART_TUTORIAL.md`](Tutorials/BOOKART_TUTORIAL.md), [`RFC_BOOKART_1.md`](RFC_BOOKART_1.md). |
| [`GENERATE.md` › `plakat portrait`](GENERATE.md#plakat-portrait) | Identity preservation in `plakat portrait` (the reference-photo path, distinct from `persona`): IP-Adapter-Plus-Face vs FaceID strategies, alignment (`--face-bbox`, `--face-landmarks`, SCRFD auto-detection), single- and multi-persona compositing in scenarios, ArcFace setup. |
| [`STYLES.md`](STYLES.md) | Art-style detection and transfer. The style catalog (`catalog.json` + `exemplars.safetensors`), `plakat style` subcommands (`detect`, `list`, `show`, `init`, `probe`), `--style-ref` / `--style` flags on generate + portrait + scenarios, building custom catalogs, the bundled 5-style catalog. |
| [`ARTEFACTS.md`](ARTEFACTS.md) | Artefact compositing — place named PNG cutouts (trees, sky elements, houses, etc.) into named zones of generated images. Library schema, zone references (4×3 default grid with override support), `plakat artefact` subcommand, `--artefact` flag on generate + portrait, per-task `artefacts:` in scenarios. |
| [`IMG2IMG.md`](IMG2IMG.md) | Image-to-image and inpaint via `plakat img2img`. Mode selection (whole-image vs masked region), the strength dial, mask conventions (white = inpaint), mask feathering, resolution handling. Covers SD-family, Flux (BF16 / GGUF / NF4) rectified-flow img2img, Flux.1-Fill-dev inpaint, SD3 / SD3.5 RePaint-style img2img + inpaint, `plakat outpaint`, and tiled inpaint on Flux Fill / SD3. |
| [`CONTROLNET.md`](CONTROLNET.md) | ControlNet conditioning across SD 1.5 / 2.1, SDXL, Flux (BF16 / GGUF / NF4 via Shakker-Labs Union Pro v2), and SD3 / SD3.5 (via the InstantX adapter family). `--control` (depth / canny / openpose / lineart / softedge), `--control-image` (pre-rendered) / `--control-from` (auto-annotate), `--control-spec` for the repeatable per-CN grammar with step-gating + multi-CN. Composition with LoRA, img2img, Flux.1-Fill-dev, tiled hi-res, NF4 / GGUF. Scenario integration. |
| [`APPLE_REQUIREMENTS.md`](APPLE_REQUIREMENTS.md) | Apple hardware requirements — minimum / recommended / ideal Apple Silicon tiers, Intel Mac CPU-only fallback, expected per-image speeds, model download sizes, macOS version compatibility, build prerequisites. |
| [`SCRIPTING.md`](SCRIPTING.md) | **v0.21.** `plakat run SCRIPT.bund` Bund scripting reference. The seven `plakat.*` host words (load + generate + img2img + portrait + upscale + save + config.set), full `GenerationConfig` knob list, REPL meta-commands, architecture notes, v0.21 limitations. Companion to [`Tutorials/SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md). |
| [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) | **v0.21.** Design RFC + the seven locked architectural decisions (embed crate, stdlib strategy, subcommand name, MVP word set, build gating, REPL, extension). Phase plan + integration constraints inherited from Bund. Read this if you're contributing a new `plakat.*` word. |

## Tutorials

If you're learning plakat from scratch or trying a feature for the
first time, start in [`Tutorials/`](Tutorials/) — these are
beginner-oriented walkthroughs that assume no prior text-to-image
experience. See [`Tutorials/README.md`](Tutorials/README.md) for a
suggested reading order.

## How the docs are organized

- **Reference manuals** (this directory) are exhaustive and
  organized by feature. Use them as lookup tables when you know what
  you want and need the precise flag or schema field.
- **Tutorials** are sequenced, narrative, and teach concepts as
  they're used. Read them top to bottom.
- The top-level [`README.md`](../README.md) is a one-screen overview
  pointing here. Recent-release history (the "What's new in vN.N"
  sections) lives there.
