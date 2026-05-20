# CONTROL — ControlNet conditioning

Runnable tutorial for `plakat --control`. Demonstrates the v0.9
depth conditioner end-to-end.

ControlNet adds an extra layer of structural guidance to the
denoise: every diffusion step gets a parallel network's residuals
that tell the UNet "the layout should look like *this*". The
"this" can be a depth map (v0.9), or in future releases a canny
edge image, a pose skeleton, a scribble, etc.

## What's in here

```
CONTROL/
├── README.md
├── inputs/
│   └── scene_depth.png       ← procedurally drawn depth map
└── scripts/
    ├── 01_basic.sh                ← plain --control depth
    ├── 02_strength_sweep.sh       ← --control-strength dial sweep
    └── 03_with_img2img.sh         ← compose control with img2img
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
- `--control` is SD 1.5 only in v0.9. SDXL ControlNet weights are
  on the v0.10 roadmap; Flux ControlNet is not currently planned.

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

## Modifications to try

**Bring your own depth map.** Replace `inputs/scene_depth.png`
with any grayscale image where bright = near, dark = far. Tools:
- [Depth-Anything-V2 online](https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2)
- MiDaS or DPT in a separate script
- Hand-paint in any image editor
- Run plakat's own depth estimator on a reference photo (not
  yet a CLI subcommand — see `src/pipelines/depth.rs`)

**Use a non-depth image.** ControlNet-Depth was trained on
*proper* depth maps, but it sort of works with any grayscale
brightness-encodes-depth signal — e.g. a luminance-flattened
photograph. Results are less reliable but it's worth trying.

**Combine with `--style-ref`.** A style reference photo + depth
conditioning lets you control both composition (depth) and
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
