# plakat proof corpus

A reproducible, self-documenting body of images demonstrating that
plakat's feature surface actually works end to end — and the tools to
regenerate and index it.

🖼️ **[Browse the rendered gallery → `GALLERY.md`](GALLERY.md)** — every
image with its embedded recipe (AnimateDiff clips as animated GIFs).

Two kinds of files live here:

- **Driver definitions** (committed): `*.hjson` scenarios + `*.sh` /
  `*.bund` scripts that render one+ representative image per capability.
- **Output images** (`images/`, committed as the proof): what the
  drivers produce. Each PNG is self-documenting — its full recipe is
  embedded in the `parameters` chunk + a JSON sidecar.

The index ([`README` is regenerated below the line](#corpus-index)) is
built by the `plakat gallery` subcommand straight from that embedded
metadata — no hand-maintenance.

## Workflow

```bash
# 1. Render a category (downloads its model on first run)
plakat scenario corpus/cascade.hjson

# 2. (repeat for other categories you can run — see COVERAGE.md)

# 3. Rebuild the index from every rendered image
plakat gallery corpus/images --recursive --out corpus/GALLERY.md
```

Validate a scenario without generating: `plakat scenario FILE --dry-run`.

## What's here

Two driver kinds: **scenarios** (`*.hjson` — batch text-to-image, run with
`plakat scenario FILE`) and **scripts** (`*.sh` — features the scenario
engine doesn't drive). Each renders into `images/<name>/`.

### Scenarios (`*.hjson`) — `plakat scenario corpus/<file>`

A scenario is an HJSON batch: a shared `prompt-header`/`-footer`, a list of
`tasks` (each a prompt + optional per-task `control:` / `style:` / size),
and optionally `scene`/`weather` axes that cross-product. Add `--dry-run`
to validate without generating.

| File | What it does |
|---|---|
| `sd15.hjson` | SD 1.5 text-to-image at 512² (ungated, ~4 GB), incl. a **canny ControlNet** task. |
| `sd21.hjson` | SD 2.1 at 768² native — OpenCLIP-H + **v-prediction** (proves the repointed ungated mirror). |
| `sdxl.hjson` | SDXL t2i variety + a **canny ControlNet** task (auto-annotated from the source). ~7 GB. |
| `sd35.hjson` | SD 3.5-medium — incl. a legible **"FRESH BREAD"** sign (its text-rendering strength). ⚠️ gated. |
| `pixart.hjson` | PixArt-Σ (DiT-XL/2 + T5) text-to-image variety. |
| `cascade.hjson` | Stable Cascade (Würstchen v3, 3-stage) at 1024² + a canny ControlNet task. ~16 GB. |
| `portrait.hjson` | Reference-photo **lookalike** portraits (IP-Adapter-Plus-Face) from `examples/persona/example.png`. |
| `weather-scene.hjson` | The engine's **`scene` × `weather`** axes: one area (a lighthouse coast, held in `prompt-header`) re-lit + re-weathered across the cross-product. |

### Scripts (`*.sh`) — features not scenario-drivable

Small shells whose outputs write into `images/` alongside the scenarios.

| File | What it does |
|---|---|
| `style_train.sh [sd15\|sdxl\|sd35]` | **Train** a watercolour style LoRA from `style/watercolour/` on the chosen base (`plakat style train`). Slow (full back-prop); run once → `style/watercolour-<base>.safetensors`. |
| `style_gen.sh [sd15\|sdxl\|sd35]` | **Generate** watercolour images with the trained LoRA (`--lora`). Fast; reuses the LoRA without retraining → `images/style-<base>/`. Run `style_train.sh` first. |
| `animate.sh` | **AnimateDiff** — text → a short motion clip (frames + GIF) on an aesthetic SD 1.5 base. |
| `img2img.sh` | **img2img style transfer** — repaint a photo into a medium (oil / watercolour / ink-wash) while keeping its composition, on SDXL. |
| `portrait.sh` | **Text-only persona portraits** (no reference photo) on SD 1.5. |
| `upscale.sh` | **ML super-resolution** — Real-ESRGAN ×2 of an existing image (Metal-safe; ×4 OOMs on Metal → `--device cpu`). |
| `script.sh` → `script.bund` | **Bund scripting** (`plakat run`) — a stack-based script proving the **load → generate → upscale → save** handle-reuse chain (render to an in-memory handle, upscale it with no disk round-trip, save). SD 1.5. |
| `looks.sh` | **Art-medium looks** (`--look`) — one subject across the 8 bundled mediums (ink-wash / watercolour / oil / charcoal / pencil / chalk-pastel / linocut / gouache); a generic counterpart to the trained style LoRAs. Civitai download per look on first run. |
| `genres.sh` | **Subject-domain genres** (`--genre`) — the bundled `anime` domain (independent axis from `--look`; they compose). Civitai download on first run. |
| `inpaint.sh` | **Inpaint** (`img2img --mask`) — repaint a masked region (the sky of a committed landscape) while preserving the rest. Self-contained (input + `assets/inpaint-sky-mask.png` committed). SD 1.5. |
| `outpaint.sh` | **Outpaint** (`plakat outpaint`) — extend an image's canvas sideways + paint the new region in-context (auto-mask, `sd15-inpaint`). |
| `stylize.sh` | **Stylize** (`plakat stylize`) — transfer a reference image's *look* onto a subject via IP-Adapter (no prompt, no training; distinct from img2img + style LoRAs). Bold subjects restyle best. SD 1.5. |

See [`COVERAGE.md`](COVERAGE.md) for the full capability matrix and which
drivers are still to be added.

## Notes

- **Gated models** (Flux-dev, SD 3.5) need a HuggingFace token (accept the
  licence first). SD 3.5-medium runs BF16-native on Metal; **Flux GGUF
  does not work on Apple Metal** (a candle kernel bug — use `--device
  cpu`, non-quantized Flux, or skip on Metal). The corpus marks these.
- Output images are committed so the proof is browsable without running
  anything; rerun the drivers to refresh them.
