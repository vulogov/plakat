# plakat — reference documentation

This directory holds the full reference manual for plakat. For
beginner-friendly walkthroughs, see [`Tutorials/`](Tutorials/).

## Reference manuals

| File | Covers |
|---|---|
| [`GENERATE.md`](GENERATE.md) | Text-to-image generation. Every flag on `plakat generate`, the model catalog (SD 1.5 / 2.1 / SDXL / SDXL-Turbo / Flux BF16+GGUF+NF4 / Flux Fill / Flux Canny-dev / Flux Depth-dev / SD3 / SD3.5 Medium / Large / Large Turbo), scheduler reference, refiner pipeline, LoRA stacking, prompt wildcards, CLIP-skip, ADetailer face refinement, Hires fix, Textual Inversion inspector, Flux Redux + concept-variant auto-annotation, the `--fast` distillation presets, tiled hi-res, scenarios (HJSON format), upscaling, output naming, and the `plakat civitai` / `plakat embedding` subcommands. |
| [`PERSONA.md`](PERSONA.md) | Identity preservation in portraits. `plakat portrait` flags, IP-Adapter-Plus-Face vs FaceID strategies, alignment options (`--face-bbox`, `--face-landmarks`, SCRFD auto-detection), persona definitions in scenarios, single-persona and multi-persona compositing, ArcFace setup, troubleshooting. |
| [`STYLES.md`](STYLES.md) | Art-style detection and transfer. The style catalog (`catalog.json` + `exemplars.safetensors`), `plakat style` subcommands (`detect`, `list`, `show`, `init`, `probe`), `--style-ref` / `--style` flags on generate + portrait + scenarios, building custom catalogs, the bundled 5-style catalog. |
| [`ARTEFACTS.md`](ARTEFACTS.md) | Artefact compositing — place named PNG cutouts (trees, sky elements, houses, etc.) into named zones of generated images. Library schema, zone references (4×3 default grid with override support), `plakat artefact` subcommand, `--artefact` flag on generate + portrait, per-task `artefacts:` in scenarios. |
| [`IMG2IMG.md`](IMG2IMG.md) | Image-to-image and inpaint via `plakat img2img`. Mode selection (whole-image vs masked region), the strength dial, mask conventions (white = inpaint), mask feathering, resolution handling. Covers SD-family, Flux (BF16 / GGUF / NF4) rectified-flow img2img, Flux.1-Fill-dev inpaint, SD3 / SD3.5 RePaint-style img2img + inpaint, `plakat outpaint`, and tiled inpaint on Flux Fill / SD3. |
| [`CONTROLNET.md`](CONTROLNET.md) | ControlNet conditioning across SD 1.5 / 2.1, SDXL, Flux (BF16 / GGUF / NF4 via Shakker-Labs Union Pro v2), and SD3 / SD3.5 (via the InstantX adapter family). `--control` (depth / canny / openpose / lineart / softedge), `--control-image` (pre-rendered) / `--control-from` (auto-annotate), `--control-spec` for the repeatable per-CN grammar with step-gating + multi-CN. Composition with LoRA, img2img, Flux.1-Fill-dev, tiled hi-res, NF4 / GGUF. Scenario integration. |
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
  pointing here. Recent-release history (the "What's new in vN.N"
  sections) lives there.
