# `--control` — ControlNet conditioning

Reference for plakat's ControlNet integration (v0.9). For a runnable
walkthrough, see [`examples/tutorials/CONTROL/`](../examples/tutorials/CONTROL/).

ControlNet adds an extra layer of **structural guidance** to a
diffusion model's denoise. At every step, a parallel network
("ControlNet") produces residuals that are added to the UNet's
intermediate features. The result: the generated image follows the
*shape* specified by a conditioning image, while the prompt drives
the *content*.

In v0.9 plakat ships **depth** conditioning. Other conditioners
(canny edges, scribble, pose, normal, MLSD lines, segmentation) are
on the roadmap.

## What `--control depth` does

You provide a **depth map** (grayscale image, white = near, black =
far). ControlNet's parallel network processes it through a hint
encoder + a copy of the UNet's down-encoder, producing residuals
that get added to the UNet at every denoise step.

The output: an image whose 3-D layout follows the depth map. A
bright foreground disc in the conditioning image → a foreground
subject in the output frame. A receding gradient at the horizon →
a receding gradient in the result.

The prompt still controls **what** appears (cat? fox? robot?); the
depth map controls **where** it sits in the frame.

## Quick start

```bash
plakat generate "a cat sitting in a meadow, golden hour" \
    --control depth \
    --control-image scene_depth.png
```

```bash
# Compose with portrait (the conditioner shapes the body pose):
plakat portrait "a confident professional headshot" \
    --photo face.jpg \
    --control depth --control-image pose_depth.png
```

```bash
# Compose with img2img (use the source as both starting point and
# control signal):
plakat img2img sketch.png --prompt "polished oil painting" \
    --control depth --control-image sketch.png
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--control <KIND>` | (off) | Conditioner kind. v0.10: `depth` only. Triggers ControlNet activation. |
| `--control-image <PATH>` | — | Path to a **pre-rendered** conditioning image (a real depth map, edge map, etc.). Mutually exclusive with `--control-from`. |
| `--control-from <PATH>` | — | **v0.10**: path to an **ordinary image** to auto-annotate via the matching annotator (e.g. Depth-Anything-V2 for `--control depth`). Mutually exclusive with `--control-image`. |
| `--control-strength <F>` | `1.0` | Multiplier applied to ControlNet residuals before adding to the UNet. Range `[0.0, ~2.0]`. Sweet spot 0.6–1.1. |

All four flags work on `plakat generate`, `plakat portrait`, and
`plakat img2img`. They also have a scenario-level equivalent (see
[Scenarios](#scenarios) below).

**`plakat img2img` defaults**: when `--control` is set but neither
`--control-image` nor `--control-from` is supplied, the `<INPUT>`
image is auto-annotated. This makes the common case (`plakat
img2img photo.png --prompt "..." --control depth`) work out of
the box without an extra flag — the prompt-driven repaint preserves
the depth structure of the source.

## The conditioning image

For `--control depth`, the conditioning image is a depth map:

- **Brightness = depth.** White = near (foreground); black = far
  (sky, background); intermediate grey = mid-distance.
- **Same resolution as your generation** (or close — plakat resizes
  with a triangle filter; the actual model resolution is the latent
  grid you're generating at).
- **Grayscale, RGB, or RGBA accepted.** RGB inputs are read as RGB
  directly. Grayscale gets replicated across all three channels.
  RGBA: the RGB channels are used; alpha is ignored.

### Where to get a depth map

The simplest path, new in v0.10: **don't generate one manually**.
Use `--control-from PATH` and plakat runs Depth-Anything-V2 on
the source image for you, then feeds the result to ControlNet.

```bash
# Auto-annotate a reference photo:
plakat generate "a fox in tall grass" \
    --control depth --control-from photo_with_layout.jpg
```

When you need a pre-rendered map (more control, repeatable
output), the manual options:

1. **Run a depth estimator yourself.** [Depth-Anything-V2 online](https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2)
   or any MiDaS/DPT tool. Save the depth output as PNG; pass via
   `--control-image`.
2. **Use a rendering engine's depth pass.** Blender, Unreal,
   etc. expose a depth-buffer output. Save as PNG; pass via
   `--control-image`.
3. **Paint one by hand.** A grayscale painter's interpretation of
   "near = white, far = black" works surprisingly well.
4. **Generate one procedurally.** The runnable tutorial does this
   in 80 lines of Rust — see `examples/draw_control_sample.rs`.

ControlNet-Depth is trained on proper depth maps but tolerates
fairly rough approximations.

## The strength dial

`--control-strength` is the standard "controlnet conditioning
scale" parameter. It linearly scales every residual before the
UNet sums them into its features.

| Strength | Effect |
|---|---|
| 0.0 | Equivalent to ControlNet disabled — pure prompt-driven generation. |
| 0.3 | Faint suggestion; the layout follows the depth roughly. |
| 0.5–0.7 | Loose but noticeable. Subject placement tracks the depth, fine details don't. |
| **1.0** (default) | The diffusers reference value. The layout firmly follows the depth map. Best balance of structure + prompt fidelity. |
| 1.2–1.5 | Aggressive structural enforcement. The prompt may struggle to express texture/atmosphere if it conflicts with the depth. |
| 2.0+ | The depth map dominates. Usually unusable — feels like the model is fighting itself. |

If the model isn't respecting the depth map, push up. If the model
is ignoring your prompt's textures/lighting/style, pull down.

## Composition with other plakat features

`--control` is **additive**. It composes cleanly with every other
plakat feature:

| Feature | Interaction with `--control` |
|---|---|
| `--style` / `--style-ref` | Orthogonal. Style affects palette/aesthetic; control affects layout. |
| `--loras` | Additive — both run in the same denoise. |
| `--artefact` / `--artefact-blend` | Orthogonal. ControlNet shapes the generated image first; artefacts get composited / blended on top. |
| `--smart-zones` | Orthogonal. Both feature paths still apply. |
| `--refiner` (SDXL) | Currently SD 1.5 only, so refiner isn't relevant in v0.9. v0.10 will extend control to SDXL where this matters. |
| `--photo` (portrait) | Composes — portrait IP-Adapter and ControlNet operate at different attention layers. |
| `--mask` (img2img inpaint) | Composes — control applies inside the mask only (since the mask gates where denoise actually runs). |
| Multi-persona scenarios | Control applies to the base layout pass only; per-persona inpaint passes skip it to avoid double-conditioning. |
| Flux | **Not supported.** ControlNet is SD-family only in v0.9. Flux scenes can't use `--control`. |

## Scenarios

A scenario task accepts a `control` block. The conditioning
source is either `image:` (pre-rendered map) or `auto-from:`
(image to auto-annotate). Exactly one must be set.

```hjson
tasks:
[
    {
        name: depth_guided_meadow_prerendered
        scene: meadow
        weather: golden_hour
        prompt: "a fox in tall grass"
        control: {
            kind: depth
            image: ./hints/meadow_depth.png
            strength: 0.85       # optional, defaults to 1.0
        }
    }

    {
        # v0.10: auto-annotate any image — depth is estimated by
        # Depth-Anything-V2-small at task time.
        name: depth_guided_from_photo
        scene: meadow
        weather: golden_hour
        prompt: "a fox in tall grass"
        control: {
            kind: depth
            auto-from: ./references/composition.jpg
            strength: 0.9
        }
    }
]
```

The ControlNet network is **cached across tasks** in the same
scenario run — the first task that needs `kind: depth` triggers a
weight download; subsequent tasks reuse the loaded network. Same
lazy-load pattern as `--smart-zones` uses for Depth-Anything-V2.

Different kinds in the same scenario load independently (so if
v0.10 adds canny, you can have one task with `kind: depth` and
another with `kind: canny` in the same file, each loading its
own weights once).

## Cost and first-run behaviour

| Cost | Where it falls |
|---|---|
| Model download | ~1.4 GB on first use (SD 1.5 ControlNet-Depth diffusers safetensors). Cached afterwards. Plakat tries primary + 2 fallback mirrors via the same `get_first_of` pattern as the rest of plakat. |
| Inference cost per step | ~30–40 % over a plain UNet call. ControlNet's down-encoder is roughly half the UNet's compute footprint. |
| Total generation time | Expect ~1.3–1.5× the equivalent `--control`-free run on the same hardware. |
| Memory | +~1.5 GB resident GPU memory for the ControlNet network. |

On Apple Silicon M-class chips (Metal backend), SD 1.5 + ControlNet
at 512² runs ~7–10 s per image after first-run JIT. See
[`APPLE_REQUIREMENTS.md`](APPLE_REQUIREMENTS.md) for full hardware
tiers.

## Weight mirrors (HuggingFace)

In download-preference order:

1. `lllyasviel/sd-controlnet-depth` / `diffusion_pytorch_model.safetensors` (~1.4 GB)
2. `lllyasviel/sd-controlnet-depth` / `diffusion_pytorch_model.fp16.safetensors` (~700 MB, halved)
3. `lllyasviel/control_v11f1p_sd15_depth` / `diffusion_pytorch_model.safetensors` (~1.4 GB)

All three are diffusers-format. WebUI-format ControlNet checkpoints
(with `control_model.input_blocks.…` key naming) are not currently
supported — they'd need a key remapping layer we don't ship.

## Limits

- **SD 1.5 only.** SDXL ControlNet weights have a different
  architecture; v0.9 doesn't wire them. v0.10 will.
- **Flux: not supported.** ControlNet's residual-addition contract
  is SD-UNet-specific.
- **Depth only.** Canny, scribble, pose, MLSD, normal, openpose,
  segmentation, and InstantID-style face conditioners are all
  on the roadmap but not in v0.9.
- **No timestep windowing.** Diffusers' `control_guidance_start`
  and `control_guidance_end` (which let you apply control to a
  subset of the timestep schedule) are not exposed in v0.9. Either
  the conditioner is active for every step or for none.
- **No multi-controlnet.** Diffusers supports stacking multiple
  ControlNets (e.g. depth + pose) by summing their residuals.
  Plakat's v0.9 CLI grammar doesn't expose this.
- **Multi-persona scenarios** apply control only to the base layout
  pass, not the per-persona inpaint passes. That's the safer
  default; the inverse (control on each persona) is on the
  roadmap.

## See also

- [Runnable tutorial](../examples/tutorials/CONTROL/) — three
  scripts + a procedurally-drawn depth map.
- [`Documentation/Tutorials/CONTROLNET_TUTORIAL.md`](Tutorials/CONTROLNET_TUTORIAL.md)
  — narrative walkthrough.
- [`GENERATE.md`](GENERATE.md) — most flags are shared with `plakat
  generate`.
- [`IMG2IMG.md`](IMG2IMG.md) — img2img reference (composes cleanly
  with `--control`).
- [`PERSONA.md`](PERSONA.md) — for portrait conditioning.
- [`APPLE_REQUIREMENTS.md`](APPLE_REQUIREMENTS.md) — chip + memory
  tiers and expected per-image speeds.
