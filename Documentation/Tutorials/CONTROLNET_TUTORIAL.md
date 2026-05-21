# `plakat --control` tutorial

ControlNet adds *structural guidance* to a diffusion model. The
prompt still controls what appears in the image; the ControlNet
conditioning controls **where** it appears and **how** it's
arranged.

This tutorial covers plakat's ControlNet integration in v0.10:

- Two conditioners: **depth** (Depth-Anything-V2) and **canny**
  (Sobel + non-maximum suppression edge detection).
- Two architectures: **SD 1.5** and **SDXL**. Auto-detected from
  `--model`; no extra flag.
- Three ways to supply the conditioning image:
  `--control-image PATH` (pre-rendered), `--control-from PATH`
  (auto-annotate any image), or — for `plakat img2img` — let
  the source image auto-annotate by default.

For the runnable companion, see
[`examples/tutorials/CONTROL/`](../../examples/tutorials/CONTROL/).

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md). You
  should be comfortable with `--prompt`, `--seed`, `--steps`, and
  the idea of starting from noise.
- A `plakat` binary with GPU support
  ([`APPLE_REQUIREMENTS.md`](../APPLE_REQUIREMENTS.md) tiers).

## 1. The mental model

`plakat generate "a fox"` starts from noise and denoises toward
an image of a fox. The prompt is the *only* steering signal — the
model decides composition, pose, framing, lighting, all of it.

Adding `--control <KIND> --control-image hint.png` adds a
**second** steering signal: a parallel network (ControlNet)
processes `hint.png` and injects residuals into the UNet at
every denoise step. These residuals push the generated image
toward a layout that matches the conditioning.

The two signals — text prompt + control conditioning — work
together. If they're consistent ("a fox" + a depth map with a
foreground subject), the model produces a fox in the foreground.
If they conflict, the model tries to satisfy both, usually with
the prompt slightly losing.

## 2. Choosing a conditioner: depth vs canny

Each conditioner answers a different question about your image:

| Conditioner | What it captures | When it's right |
|---|---|---|
| **depth** | 3-D layout — where in the frame is the foreground, mid-distance, far distance. | "I want the subject to sit *here* in the frame, with a horizon line *there*." Useful for compositional control without dictating exact shapes. Tolerant: rough depth maps work fine. |
| **canny** | 2-D structural lines — silhouettes, edges, contours. | "I want the output to follow *these exact shapes*." Useful when shapes matter (architecture, line art, faithful re-renders of a layout). Literal: every edge in your input becomes an edge in the output. |

**Rule of thumb:**
- "Where does stuff sit in 3-D?" → depth.
- "What are the precise outlines?" → canny.

Both run as `--control <KIND>` with the same strength/CLI surface.
You can experiment by switching the flag — the rest of the command
stays identical.

## 3. Your first control-guided generation (depth, auto-from a photo)

The simplest v0.10 invocation:

```bash
plakat generate "a fox sitting in tall grass, golden hour light" \
    --control depth \
    --control-from any_photo_with_layout.jpg \
    --seed 42
```

`--control-from PATH` says: "run the matching annotator on this
photo to produce the conditioning." For `depth`, plakat runs
Depth-Anything-V2-small on `any_photo_with_layout.jpg`, then
feeds the depth output to ControlNet-Depth.

What happens behind the scenes:

1. SD 1.5 loads (downloads ~4 GB if not cached).
2. ControlNet-Depth-SD15 loads (~1.4 GB on first use).
3. Depth-Anything-V2-small loads (~99 MB on first use). If
   you've used `--smart-zones` before, no extra download —
   it's the same weights file.
4. `any_photo_with_layout.jpg` is loaded, run through Depth-
   Anything-V2, and packed into a `(1, 3, H, W)` conditioning
   tensor.
5. At each denoise step, ControlNet's parallel down-encoder
   produces residuals that get added to the UNet's intermediate
   features.
6. The result lands at `out/plakat-42.png`.

Wall-time overhead: ~1.3–1.5× the equivalent `--control`-free
run, plus the one-off auto-annotation cost (~1 s on GPU).

## 4. Pre-rendered conditioning with `--control-image`

When you want a specific layout — e.g. a depth map you painted
by hand, or a known reference you want to reuse across many
prompts — pass the conditioning image directly:

```bash
plakat generate "a quiet meadow at dawn" \
    --control depth \
    --control-image my_painted_depth.png \
    --seed 42
```

This is the v0.9 path. `--control-image` and `--control-from`
are mutually exclusive — clap enforces this at parse time.

Where to source pre-rendered conditioning:

- **Depth maps**: Depth-Anything-V2 on a reference photo (online
  tools work; or use `--control-from` to skip this step), depth
  passes from 3D rendering engines (Blender Z-pass, Unreal
  depth buffer), or hand-painted grayscale.
- **Canny edge maps**: any image editor's edge-detect filter
  (GIMP/Photoshop both have one), or hand-drawn line art on
  white background then inverted.

## 5. `plakat img2img` default: auto-annotate the input

For img2img, the most common workflow is "lock the structure
of this image while repainting it." v0.10 makes that the
default — pass `--control` without an image or from-path:

```bash
plakat img2img source.png \
    --prompt "the same scene as an oil painting" \
    --strength 0.75 \
    --control depth
```

What plakat does: auto-annotates `source.png` (the img2img
input) and uses that depth map as the ControlNet conditioner.
The structural depth of the source is locked even as the
img2img strength rewrites everything else.

This is equivalent to writing `--control-from source.png`
explicitly — just shorter.

Override the default by passing either `--control-image PATH`
(supply your own conditioning) or `--control-from PATH`
(annotate a *different* image than the source).

## 6. SDXL ControlNet

ControlNet works on both SD 1.5 and SDXL. Plakat auto-detects
the architecture from `--model`:

```bash
# SDXL ControlNet — same flags, different model
plakat generate "a fox, photographic, shallow depth of field" \
    --model sdxl --size 1024x1024 \
    --control depth --control-from photo.jpg \
    --seed 42
```

Plakat downloads the matching SDXL ControlNet checkpoint
(`diffusers/controlnet-depth-sdxl-1.0-small` for depth,
`diffusers/controlnet-canny-sdxl-1.0-small` for canny — both
~600 MB).

**Speed/memory expectations on Apple Silicon (24 GB)**:
- SD 1.5 + ControlNet at 512²: ~7–10 s/image after JIT.
- SDXL + ControlNet at 1024²: ~25–40 s/image after JIT.

For full hardware tiers see
[`APPLE_REQUIREMENTS.md`](../APPLE_REQUIREMENTS.md).

## 7. The strength dial

`--control-strength` linearly scales every ControlNet residual
before it's added to the UNet. Range typically `[0.0, 2.0]`.

| Strength | Use case |
|---|---|
| 0.0 | ControlNet disabled. Same as not passing `--control`. |
| 0.3 | Faint structural hint. Model takes it as a suggestion. |
| 0.6 | Moderate guidance. Layout broadly follows the conditioning. |
| **1.0 (default)** | Diffusers reference value. Layout firmly follows the conditioning. |
| 1.3 | Strong enforcement. Useful when 1.0 isn't getting tight enough. |
| 1.8+ | Usually unusable — conditioning dominates, prompt details suffer. |

**Push up** when the model is ignoring the conditioning.
**Pull down** when the model is fighting your prompt's textures,
lighting, or atmosphere.

Canny tends to be more literal than depth at the same strength —
0.7–0.9 often gives a better balance for canny than depth's
default 1.0.

## 8. Composing with other plakat features

`--control` is additive. It composes cleanly with every other
feature:

| Feature | Interaction |
|---|---|
| `plakat portrait` `--photo` | Composes. Identity (who) + control (layout) operate at different attention layers. Useful for enforced poses on a specific person. |
| `plakat img2img` `--strength` | Composes. Strength controls "how much creative latitude"; control adds "respect this structure" on top. |
| `--mask` (inpaint) | Composes. Control applies inside the masked region (where denoise actually runs). |
| `--style` / `--style-ref` | Orthogonal. Style sets palette/aesthetic; control sets layout. |
| `--smart-zones` | Orthogonal. ControlNet shapes the generated image; smart-zones places artefact PNGs on top. |
| `--loras` | Additive — both run in the same denoise. |
| `--refiner` (SDXL) | Compatible. Control residuals apply to both base + refiner UNet passes. |
| Scenarios | Per-task `control: { ... }` block. The ControlNet network is cached across tasks. |
| Flux | **Not supported.** ControlNet is SD-family only. |

## 9. Composing with scenarios

```hjson
tasks:
[
    {
        name: depth_guided_meadow
        prompt: "a fox in tall grass"
        control: {
            kind: depth
            auto-from: ./references/composition.jpg
            strength: 0.9
        }
    }

    {
        name: canny_guided_architecture
        prompt: "a renaissance villa, oil painting"
        control: {
            kind: canny
            image: ./hints/villa_edges.png
            strength: 0.85
        }
    }
]
```

Set exactly one of `image:` (pre-rendered) or `auto-from:`
(auto-annotate) per task. ControlNet weights are downloaded
once per `(kind)` and reused across tasks in the same scenario
run. Different kinds in the same scenario load independently —
the task above with `kind: depth` and the next with `kind: canny`
each trigger their own weight download (once each).

## 10. Common patterns

### "I want this exact composition with a different aesthetic"

```bash
plakat img2img reference.jpg --prompt "...new aesthetic..." \
    --strength 0.8 --control depth
```

img2img rewrites the aesthetic; control locks the depth structure.

### "I want my subject to follow these architectural lines"

```bash
plakat generate "a robot inside a Gothic cathedral, dramatic light" \
    --control canny --control-from cathedral_photo.jpg
```

Canny extracts the cathedral's structural lines; the model fills
the interior with a robot scene that respects those lines.

### "Same depth map, many prompts"

```bash
DEPTH=depth_hint.png
for p in "fox" "wolf" "deer" "rabbit"; do
    plakat generate "a $p in tall grass" \
        --control depth --control-image "$DEPTH" \
        --seed 42 --out "out/$p"
done
```

Identical composition across variations. Use `--control-image`
(pre-rendered) here so the depth map is byte-identical every run.

### "Lock pose with depth, refine identity with portrait"

```bash
plakat portrait "a professional headshot, soft studio lighting" \
    --photo face.jpg --face-strength 0.85 \
    --control depth --control-from pose_reference.png
```

`--photo` controls who; `--control` controls how they're posed.
Particularly useful for enforcing a specific posture (T-pose,
contrapposto, sitting) while keeping subject identity locked.

## 11. Limits

- **SD 1.5 + SDXL only.** Flux uses a different architecture and
  isn't on the roadmap.
- **Depth + Canny only in v0.10.** Scribble, pose, MLSD, normal,
  openpose, segmentation, InstantID face — all on the roadmap.
- **No timestep windowing.** Diffusers'
  `control_guidance_start`/`control_guidance_end` (apply control
  to a subset of the schedule) lands in v0.10 phase 4.
- **No multi-controlnet.** Stacking multiple conditioners
  (depth + canny in one generation) isn't exposed via the CLI.
  v0.11 candidate.

## Where to next

- **Runnable companion**: six self-contained scripts demonstrating
  every feature ([`examples/tutorials/CONTROL/`](../../examples/tutorials/CONTROL/)).
- **Full reference**: every flag, every edge case
  ([`Documentation/CONTROLNET.md`](../CONTROLNET.md)).
- **Composing with img2img**:
  [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md).
- **Composing with artefacts**:
  [`ARTEFACTS_TUTORIAL.md`](ARTEFACTS_TUTORIAL.md).
- **Composing with style transfer**:
  [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md).
