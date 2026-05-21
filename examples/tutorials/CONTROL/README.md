# CONTROL — ControlNet conditioning

Runnable tutorial for `plakat --control`. Demonstrates both
ControlNet conditioners shipped in v0.10 — depth + canny — on
both SD 1.5 and SDXL.

ControlNet adds an extra layer of structural guidance to the
denoise: every diffusion step gets a parallel network's residuals
that tell the UNet "the layout should look like *this*". The
"this" can be:
- a **depth map** (3-D layout: foreground / mid-distance / sky)
- a **canny edge map** (2-D outlines and silhouettes)

Future releases will add scribble, pose, MLSD lines, normals,
and more.

## What's in here

```
CONTROL/
├── README.md
├── inputs/
│   └── scene_depth.png       ← procedurally drawn depth map
└── scripts/
    ├── 01_basic.sh                ← plain --control depth (SD 1.5)
    ├── 02_strength_sweep.sh       ← --control-strength dial sweep
    ├── 03_with_img2img.sh         ← compose control with img2img
    ├── 04_auto_depth.sh           ← v0.10: --control-from auto-annotates
    ├── 05_sdxl.sh                 ← v0.10: SDXL ControlNet at 1024²
    └── 06_canny.sh                ← v0.10: --control canny (edge conditioning)
```

Regenerate `inputs/scene_depth.png` at any time with:

```bash
cargo run --release --example draw_control_sample
```

## Prerequisites

- A `plakat` binary with GPU support. Apple Silicon:
  ```bash
  cargo build --release --features metal
  ```
  Linux + NVIDIA: `--features cuda`.
- First run downloads:
  - SD 1.5 (~4 GB)
  - ControlNet-Depth SD 1.5 (~1.4 GB)
  - Both cached in `~/.cache/huggingface/` after the first fetch.
- `--control` works on both SD 1.5 and SDXL (v0.10+). Plakat
  auto-detects the architecture from `--model` and downloads the
  matching ControlNet checkpoint. Flux ControlNet is not currently
  planned.

The scripts source `scripts/_plakat.sh`, which finds the binary in
`$PATH` or `target/{release,debug}/`. Run them from anywhere.

## Walkthrough

### 1. `01_basic.sh` — minimal depth-conditioned generation

```bash
./scripts/01_basic.sh
```

Generates a meadow scene with a cat. The depth map
(`inputs/scene_depth.png`) has a bright disc in the foreground —
the model should place the cat **at that disc's location** in the
output frame, not wherever it would have placed it naturally.

This is the simplest possible ControlNet use: prompt + conditioner
+ default strength (1.0).

### 2. `02_strength_sweep.sh` — the strength dial

```bash
./scripts/02_strength_sweep.sh
```

Three runs of the same prompt + same seed + same depth map,
varying only `--control-strength` (0.5, 1.0, 1.4). Open the three
outputs side-by-side:

| Strength | Effect |
|---|---|
| 0.0 | ControlNet disabled — pure t2i. |
| 0.5 | Loose layout suggestion. The fox roughly follows the depth but isn't pinned. |
| **1.0** (default) | The diffusers reference value. Layout follows the depth firmly. |
| 1.4 | Aggressive structural enforcement. The fox is pinned to the disc; prompt details may suffer. |
| 2.0+ | Often unusable — overrides the prompt almost entirely. |

The sweet spot is typically 0.7–1.1. If the layout drifts from
what the depth map specifies, push up. If the model fights you on
prompt details (textures, colours, atmosphere), pull down.

### 3. `03_with_img2img.sh` — compose with img2img

```bash
./scripts/03_with_img2img.sh
```

You can stack `--control` on top of any other plakat subcommand.
This script uses the depth map twice:
- As the **img2img source** (so the watercolor pass keeps the
  rough layout)
- As the **control image** (so the ControlNet pass pins the
  depth structure precisely)

Without the control flag, img2img alone at `--strength 0.85`
would let the model rewrite most of the composition. Adding
control locks it back to the depth map's structure.

### 4. `04_auto_depth.sh` — auto-annotate any image (v0.10)

```bash
./scripts/04_auto_depth.sh
```

Demonstrates the v0.10 ergonomic improvement: you don't need a
pre-rendered depth map. `--control-from PATH` tells plakat to
run Depth-Anything-V2 on `PATH` and use the result as the
ControlNet conditioning.

Two output files:
- `from_flag.png` — `plakat generate` with `--control-from <photo>`.
- `from_input_default.png` — `plakat img2img <photo>` with
  `--control depth` and **no `--control-image` or
  `--control-from`** — the input is auto-annotated by default.

First run downloads Depth-Anything-V2-small (~99 MB) once. The
weight is shared across smart-zones and ControlNet annotation; if
you've already used `--smart-zones`, no extra download.

### 5. `05_sdxl.sh` — SDXL ControlNet at 1024² (v0.10)

```bash
./scripts/05_sdxl.sh
```

Same depth-conditioning workflow as `01_basic.sh`, but routed
through SDXL. Plakat auto-detects the architecture from `--model
sdxl` and downloads the matching SDXL ControlNet
(`diffusers/controlnet-depth-sdxl-1.0-small`, ~600 MB).

Compare the output with `01_basic.sh`'s SD 1.5 version at 512² —
the SDXL output at 1024² should have noticeably sharper texture
detail. Tradeoff: ~3–4× the wall time per image, ~2× the
memory headroom required.

### 6. `06_canny.sh` — Canny edge conditioning (v0.10)

```bash
./scripts/06_canny.sh
```

The second conditioner shipped in v0.10. Where depth controls
*where* things sit in the frame (3-D layout), canny controls
*exact outlines* (2-D edges). The annotator is pure CPU image
processing — Sobel + non-maximum suppression + hysteresis
thresholding via the `imageproc` crate — no extra ML model.

This script passes `--control canny --control-from
../IMG2IMG/inputs/landscape.png`. Plakat runs Canny on the
landscape PNG and uses the binary edge map as the conditioner.
The painted output respects the source's edges (hill outlines,
sun border) while the prompt drives the oil-painting aesthetic.

## Choosing a conditioner

| Conditioner | Captures | Best for |
|---|---|---|
| `--control depth` | 3-D layout: foreground / mid / far distance | Compositional control. "I want the subject *here* in the frame." Tolerant of rough inputs. |
| `--control canny` | 2-D edges: silhouettes, contours, structural lines | Exact-shape control. "Follow *these outlines*." Most useful when shapes matter (architecture, line art, faithful re-renders). |

Switch between them by changing only the `--control` flag — the
rest of the CLI surface is identical. The matching ControlNet
checkpoint downloads automatically.

## Modifications to try

**Skip the manual depth map step.** Pass `--control-from PATH`
instead of `--control-image PATH` on any script. Plakat runs
Depth-Anything-V2 (for depth) or Canny edge detection (for
canny) on the source automatically.

**Bring your own depth map.** Replace `inputs/scene_depth.png`
with any grayscale image where bright = near, dark = far. Tools:
- [Depth-Anything-V2 online](https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2)
- MiDaS or DPT in a separate script
- Hand-paint in any image editor

**Swap depth for canny.** Change `--control depth` to
`--control canny` on any script. Use `--control-from PATH` to
auto-extract edges from any photo, or supply a pre-rendered
edge map via `--control-image PATH`.

**Switch to SDXL.** Add `--model sdxl --size 1024x1024` to any
script. The matching SDXL ControlNet downloads automatically.

**Combine with `--style-ref`.** A style reference photo + control
conditioning lets you control both composition (control) and
aesthetic (style) independently. Add `--style-ref my_painting.jpg`
to any script.

**Stack with `--smart-zones`** when artefacts are also in play.
The two features are orthogonal: ControlNet shapes the
generated image; smart zones place artefact PNGs on top of it.

## See also

- [`Documentation/CONTROLNET.md`](../../../Documentation/CONTROLNET.md)
  — full reference: every flag, the mask conventions, edge cases.
- [`Documentation/Tutorials/CONTROLNET_TUTORIAL.md`](../../../Documentation/Tutorials/CONTROLNET_TUTORIAL.md)
  — narrative walkthrough at a slower pace.
- [`Documentation/IMG2IMG.md`](../../../Documentation/IMG2IMG.md)
  — img2img reference (composes cleanly with `--control`).
- [`Documentation/APPLE_REQUIREMENTS.md`](../../../Documentation/APPLE_REQUIREMENTS.md)
  — expected speeds + memory tiers.

## License

Everything in this directory is CC0 / Unlicense — same as plakat
itself. The sample depth map is procedurally generated by plakat
and carries no third-party copyright.
