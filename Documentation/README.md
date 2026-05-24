# plakat — reference documentation

This directory holds the full reference manual for plakat. For
beginner-friendly walkthroughs, see [`Tutorials/`](Tutorials/).

## Reference manuals

| File | Covers |
|---|---|
| [`GENERATE.md`](GENERATE.md) | Text-to-image generation. Every flag on `plakat generate`, scheduler reference, refiner pipeline, LoRA stacking, scenarios (HJSON format, scene/weather/task assembly, prompt enhancers), upscaling, output naming. **v0.13:** GGUF Flux (`--model flux-*-gguf`), `--quant-level` / `--t5-quant-level`, tiled hi-res for Flux. **v0.14:** SD3.5 family (`sd35-medium`/`large`/`large-turbo`/`sd3-medium`), NF4 Flux (`flux-dev-nf4`), Flux Redux (`--redux-image`), Hyper-FLUX / Turbo presets (`--fast`), tiled SD 1.5 / 2.1. **v0.15:** Flux Canny-dev / Depth-dev concept variants (`--concept-image`), Tiled SD3, per-task `fast` / `concept-image` / `enhance` / `tiled` / `loras` in scenarios, runtime per-task LoRA on Flux. **v0.16:** wildcards (`--wildcard-dir`), CLIP-skip (`--clip-skip`), ADetailer face refinement (`--adetailer*`), Hires fix (`--hires-fix*`), Textual Inversion inspector (`--embedding` — parser ships, runtime deferred), `plakat civitai` + `plakat embedding` subcommands, `--concept-from` Flux concept-variant auto-annotation, SD3 pipeline caching across scenario tasks. |
| [`PERSONA.md`](PERSONA.md) | Identity preservation in portraits. `plakat portrait` flags, IP-Adapter-Plus-Face vs FaceID strategies, alignment options (`--face-bbox`, `--face-landmarks`, SCRFD auto-detection), persona definitions in scenarios, single-persona and multi-persona compositing, ArcFace setup, troubleshooting. |
| [`STYLES.md`](STYLES.md) | Art-style detection and transfer. The style catalog (`catalog.json` + `exemplars.safetensors`), `plakat style` subcommands (`detect`, `list`, `show`, `init`, `probe`), `--style-ref` / `--style` flags on generate + portrait + scenarios, building custom catalogs, the bundled 5-style catalog. |
| [`ARTEFACTS.md`](ARTEFACTS.md) | Artefact compositing — place named PNG cutouts (trees, sky elements, houses, etc.) into named zones of generated images. Library schema, zone references (4×3 default grid with override support), `plakat artefact` subcommand, `--artefact` flag on generate + portrait, per-task `artefacts:` in scenarios. |
| [`IMG2IMG.md`](IMG2IMG.md) | Image-to-image and inpaint via `plakat img2img`. Mode selection (whole-image vs masked region), the strength dial, mask conventions (white = inpaint), mask feathering, resolution handling. **v0.13:** Flux img2img (rectified-flow init), Flux.1-Fill-dev inpaint, `plakat outpaint` wrapper. **v0.14:** Flux Fill + ControlNet composition. **v0.15:** SD3 / SD3.5 img2img + RePaint-style inpaint. **v0.16:** Tiled + Flux Fill, Tiled + SD3 img2img/inpaint. |
| [`CONTROLNET.md`](CONTROLNET.md) | ControlNet conditioning. `--control` (depth + canny), `--control-image` (pre-rendered) / `--control-from` (auto-annotate), `--control-strength` on generate / portrait / img2img. Both SD 1.5 and SDXL. Weight mirrors, the strength dial, composition with other features, scenario integration. **v0.13:** Flux ControlNet auto-annotators, step gating, multi-Flux-CN, tiled Flux + CN composition. **v0.14:** Fill + CN composition. **v0.15:** NF4 + ControlNet composes via the residual-aware NF4 forward. **v0.16:** SD3 / SD3.5 ControlNet via the InstantX adapter family (Canny / Depth / Blur / Pose by variant). |
| [`APPLE_REQUIREMENTS.md`](APPLE_REQUIREMENTS.md) | Apple hardware requirements — minimum / recommended / ideal Apple Silicon tiers, Intel Mac CPU-only fallback, expected per-image speeds, model download sizes, macOS version compatibility, build prerequisites. |

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
  pointing here.
