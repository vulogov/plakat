# `plakat --control` tutorial

ControlNet adds *structural guidance* to a diffusion model. The
prompt still controls what appears in the image; the ControlNet
conditioning controls **where** it appears and **how** it's
arranged.

This tutorial walks through plakat's v0.9 depth-conditioning
feature: what it is, when to use it, the strength dial, and how it
composes with the other plakat features. For the runnable
companion, see [`examples/tutorials/CONTROL/`](../../examples/tutorials/CONTROL/).

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md). You
  should be comfortable with `--prompt`, `--seed`, `--steps`, and
  the idea of starting from noise.
- Finished [`ARTEFACTS_TUTORIAL.md`](ARTEFACTS_TUTORIAL.md) is
  optional but useful — `--smart-zones` already uses
  Depth-Anything-V2, which is the same kind of "depth map" the
  ControlNet conditioner consumes.
- A `plakat` binary with GPU support
  ([`APPLE_REQUIREMENTS.md`](../APPLE_REQUIREMENTS.md) tiers).

## 1. The mental model

`plakat generate "a fox"` starts from noise and denoises toward an
image of a fox. The prompt is the *only* steering signal — the
model decides composition, pose, framing, lighting, all of it.

Adding `--control depth --control-image hint.png` adds a **second**
steering signal: a parallel network (ControlNet) processes
`hint.png` and injects residuals into the UNet at every denoise
step. These residuals push the generated image toward a layout
that matches the conditioning.

For depth conditioning: white pixels in the hint say "this is
foreground; put the subject here." Black pixels say "this is far
background; keep it empty or distant."

The two signals — text prompt + control conditioning — work
together. If they're consistent ("a fox" + a depth map with a
foreground subject), the model produces a fox in the foreground.
If they conflict ("an empty meadow" + a foreground subject in the
depth), the model tries to satisfy both, usually with the
prompt slightly losing.

## 2. Your first control-guided generation

```bash
plakat generate "a fox sitting in tall grass, golden hour light" \
    --control depth \
    --control-image scene_depth.png \
    --seed 42
```

What happens:

1. plakat loads SD 1.5 (downloads if not cached: ~4 GB).
2. plakat loads ControlNet-Depth-SD15 (downloads if not cached:
   ~1.4 GB).
3. `scene_depth.png` is loaded, resized to your working resolution,
   and packed into a `(1, 3, H, W)` conditioning tensor.
4. At each denoise step (28 by default):
   - ControlNet processes the conditioning + current latents
     and produces 12 down-block residuals + 1 mid-block residual.
   - The UNet adds those residuals into its own intermediate
     features.
   - The result: the latent moves toward an image consistent with
     both the prompt AND the depth map.
5. The final latent is VAE-decoded and saved as `out/plakat-42.png`.

Expect ~1.3–1.5× the wall-clock of a `--control`-free run — the
extra time is ControlNet's parallel down-encoder running once per
step.

## 3. Where to get a depth map

Four ways:

1. **From a reference photo.** Run [Depth-Anything-V2](https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2)
   online (free), or MiDaS/DPT in your own pipeline, on a photo
   that has the composition you want. Save the depth output as PNG.

2. **From a rendering engine.** Blender's Z-pass, Unreal's depth
   buffer, Three.js depth FBO — any 3D scene can emit a depth
   map. Useful when you want pixel-perfect layout control.

3. **By hand.** Open any image editor, paint a grayscale image
   where bright = near and dark = far. Looks crude but works
   surprisingly well — ControlNet-Depth is tolerant of rough input.

4. **Procedurally.** The runnable tutorial generates a sample
   depth map in 80 lines of Rust
   ([`examples/draw_control_sample.rs`](../../examples/draw_control_sample.rs)).
   Copy and modify.

White pixels = foreground / near. Black pixels = far / sky.
Intermediate grey = mid-distance. RGB inputs are accepted (used
directly); RGBA's alpha is ignored.

## 4. The strength dial

Same pattern as `--strength` on img2img: a knob between "the
conditioner does nothing" and "the conditioner runs the show".

```bash
plakat generate "a robot in a meadow" \
    --control depth --control-image scene_depth.png \
    --control-strength 0.7
```

Reach for these values:

| Strength | Use case |
|---|---|
| 0.0 | ControlNet disabled. Same as not passing `--control` at all. |
| 0.3 | Faint structural hint. The model takes it as a suggestion. |
| 0.6 | Moderate guidance. The layout broadly follows the depth; details deviate. |
| **1.0 (default)** | The diffusers reference value. The layout firmly follows the depth map. |
| 1.3 | Strong enforcement. Useful when 1.0 isn't getting tight enough layout match. |
| 1.8+ | Usually unusable. The depth map dominates; prompt details suffer. |

When to push up: the model is ignoring the depth map's layout.
When to pull down: the model is fighting your prompt's textures
or lighting.

## 5. Composing with portrait

`plakat portrait --control` works the same way:

```bash
plakat portrait "professional headshot, soft studio lighting" \
    --photo face.jpg \
    --face-strength 0.85 \
    --control depth \
    --control-image upper_body_pose.png
```

`--photo` controls **identity** (who's in the photo);
`--control` controls **layout** (where the body sits in frame,
which way the head is angled, etc.). They operate at different
attention layers and compose cleanly.

Particularly useful for: enforcing a specific pose (T-pose,
contrapposto, sitting, ...) while keeping the subject identity
locked.

## 6. Composing with img2img

`plakat img2img` already has a "starting point" (the input
image) and a strength dial (how much creative latitude). Adding
`--control` adds a **third** signal on top:

```bash
plakat img2img sketch.png \
    --prompt "polished oil painting, dramatic lighting" \
    --strength 0.85 \
    --control depth \
    --control-image sketch.png
```

This is a powerful pattern: at `--strength 0.85`, img2img alone
would let the model rewrite most of the composition. Adding
control with the **same image** as the conditioner locks the
structure back in — you get the texture / palette / aesthetic of
the prompt-driven repaint, but the layout of the source.

For depth conditioning specifically, this requires the source
image's brightness pattern to be roughly depth-like — a sketch
with bold dark/light values often works; a flat photo less so.

## 7. Composing with scenarios

In an HJSON scenario, add a `control` block to any task:

```hjson
tasks:
[
    {
        name: depth_guided_landscape
        scene: meadow
        weather: golden_hour
        prompt: "a fox in tall grass"
        control: {
            kind: depth
            image: ./hints/landscape_depth.png
            strength: 0.9
        }
    }
]
```

The ControlNet network is **cached across tasks** in the run,
so adding `control:` to many tasks doesn't multiply the download
or load cost. The conditioning image, however, is loaded
per-task (so different tasks can use different conditioning
images).

## 8. Common patterns

### "I want the subject in a specific spot"

Paint a depth map with a bright disc at the desired location.
ControlNet-Depth biases the subject toward foreground regions.

### "I want a specific composition replicated across many prompts"

Generate one image you like, run depth estimation on it, save
the depth map, then use it as `--control-image` for every
subsequent run. The layout will follow even though the prompt
varies.

### "I want a specific aspect ratio for the subject"

A tall depth map (1024×512, say) with the foreground at the
bottom-centre will produce a wide image with a foreground
subject roughly in that position. The depth map's aspect ratio
drives composition.

### "I want the camera angle from a reference photo"

Run depth estimation on the reference. The depth map encodes
camera angle (low angle: foreground large; high angle: foreground
smaller, ground takes more frame). ControlNet preserves the
angle.

## 9. Limits

- **SD 1.5 only in v0.9.** SDXL has its own ControlNet
  ecosystem; we'll wire those weights in v0.10.
- **Flux not supported.** ControlNet's residual-addition contract
  is SD-architecture-specific.
- **Depth conditioner only in v0.9.** Canny edges, scribble, MLSD
  lines, OpenPose, normals, segmentation, InstantID face — all on
  the roadmap, none in this release.
- **No timestep windowing.** Diffusers'
  `control_guidance_start`/`control_guidance_end` (apply control
  to a window of the schedule) is on the v0.10 list.
- **No multi-controlnet.** Stacking multiple conditioners
  (depth + pose, for example) isn't exposed via the v0.9 CLI.

## Where to next

- **Runnable companion**: three self-contained scripts +
  a procedurally-drawn depth sample
  ([`examples/tutorials/CONTROL/`](../../examples/tutorials/CONTROL/)).
- **Full reference**: every flag, every edge case
  ([`Documentation/CONTROLNET.md`](../CONTROLNET.md)).
- **Composing with img2img**:
  [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md).
- **Composing with artefacts**:
  [`ARTEFACTS_TUTORIAL.md`](ARTEFACTS_TUTORIAL.md).
- **Composing with style transfer**:
  [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md).
