# plakat — reference documentation

This directory holds the full reference manual for plakat. For
beginner-friendly walkthroughs, see [`Tutorials/`](Tutorials/).

## Reference manuals

| File | Covers |
|---|---|
| [`GENERATE.md`](GENERATE.md) | Text-to-image generation. Every flag on `plakat generate`, scheduler reference, refiner pipeline, LoRA stacking, scenarios (HJSON format, scene/weather/task assembly, prompt enhancers), upscaling, output naming. |
| [`PERSONA.md`](PERSONA.md) | Identity preservation in portraits. `plakat portrait` flags, IP-Adapter-Plus-Face vs FaceID strategies, alignment options (`--face-bbox`, `--face-landmarks`, SCRFD auto-detection), persona definitions in scenarios, single-persona and multi-persona compositing, ArcFace setup, troubleshooting. |
| [`STYLES.md`](STYLES.md) | Art-style detection and transfer. The style catalog (`catalog.json` + `exemplars.safetensors`), `plakat style` subcommands (`detect`, `list`, `show`, `init`, `probe`), `--style-ref` / `--style` flags on generate + portrait + scenarios, building custom catalogs, the bundled 5-style catalog. |
| [`ARTEFACTS.md`](ARTEFACTS.md) | Artefact compositing — place named PNG cutouts (trees, sky elements, houses, etc.) into named zones of generated images. Library schema, zone references (4×3 default grid with override support), `plakat artefact` subcommand, `--artefact` flag on generate + portrait, per-task `artefacts:` in scenarios. |
| [`IMG2IMG.md`](IMG2IMG.md) | Image-to-image and inpaint via `plakat img2img`. Mode selection (whole-image vs masked region), the strength dial, mask conventions (white = inpaint), mask feathering, resolution handling. |
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
