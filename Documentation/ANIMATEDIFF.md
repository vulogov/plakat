# AnimateDiff (v0.30)

`plakat animate --animatediff` renders **motion-coherent N-frame
sequences** from a single prompt using AnimateDiff motion adapters
spliced into the SD UNet. Different from `plakat animate`'s default
prompt-morph mode, which interpolates between two prompts without
temporal-attention coherence.

**v0.27 shipped the full AnimateDiff feature set** (SD 1.5 + SDXL,
both with ControlNet + sliding-window long-form). **v0.28 made
it pleasant to use** (multi-CN stacking, AnimateLCM 4-step,
`plakat.animate` Bund word). **v0.29 brought it to scenarios**.
**v0.30 phase 2 closes the headline carry**: per-frame video
ControlNet via `--control-spec ...:video=PATH`.

| Capability | SD 1.5 | SDXL | Added in |
|---|---|---|---|
| Inference dispatch | ✓ | ✓ | v0.27 |
| Motion adapter | V3 + AnimateLCM | beta | v0.27 / v0.28 |
| Motion LoRAs | ✓ | ✓ | v0.26 / v0.27 |
| ControlNet (single) | ✓ | ✓ | v0.27 |
| **ControlNet stacking** (multi-CN) | ✓ | ✓ | **v0.28** |
| **Per-frame video ControlNet** | ✓ | ✓ | **v0.30** |
| Long-form sliding window | ✓ | ✓ | v0.27 |
| **4-step LCM generation** | ✓ (AnimateLCM) | — (no public SDXL repo) | **v0.28** |
| **Bund scripting (`plakat.animate`)** | ✓ | — (v0.29) | **v0.28** |
| Per-block motion modules | 16 (V3) / 17 (LCM) | 12 | — |
| Hard frame cap per window | 32 | 32 | — |
| Cross-fade long-form total | ~256 frames practical | ~256 frames practical | — |

## Quick start

### SD 1.5, 16-frame loop

```bash
plakat animate --animatediff --model sd15 \
    --from "a watercolor cottage at dawn" \
    --frames 16 --format mp4
```

### SDXL, 16-frame loop at training resolution

```bash
plakat animate --animatediff --model sdxl \
    --from "a knight in a forest, oil painting" \
    --frames 16 --size 1024x1024 --format mp4
```

### Motion LoRA stack (zoom-in)

```bash
plakat animate --animatediff --model sd15 \
    --from "a wizard's tower at sunset" \
    --motion-lora hf:guoyww/animatediff-motion-lora-zoom-in:0.8 \
    --format mp4
```

### ControlNet — same depth map applied to every frame

```bash
plakat animate --animatediff --model sd15 \
    --from "a fox in a snowy meadow" \
    --control depth --control-image ./depth.png \
    --frames 16 --format mp4
```

### Long-form (sliding window) — 64-frame clip

```bash
plakat animate --animatediff --model sd15 \
    --from "a misty forest at dawn" \
    --frames 64 --window-size 16 --window-overlap 4 \
    --format mp4
```

### v0.28: 4-step animate via AnimateLCM

```bash
# Switches motion adapter to wangfuyun/AnimateLCM, scheduler to LCM,
# defaults to --steps 4 --guidance 1.5 (overrideable). ~5× speedup.
plakat animate --animatediff --model sd15 --lcm \
    --from "a fox in a snowy meadow" \
    --format mp4
```

### v0.28: multi-CN stacking (depth + canny)

```bash
# Each --control-spec stacks one conditioner; residuals from every
# ControlNet sum per denoise step. SD 1.5 + SDXL both supported.
plakat animate --animatediff --model sdxl \
    --from "a knight in a forest" \
    --control-spec 'depth:image=./depth.png:strength=0.8' \
    --control-spec 'canny:from=./source.jpg:strength=0.4' \
    --frames 16 --size 1024x1024 --format mp4
```

### v0.30: per-frame video ControlNet (video-to-video)

```bash
# ffmpeg decodes the input video; frames are sub-sampled evenly to
# match --frames, each frame independently annotated, residuals
# injected per-frame during AnimateDiff sampling. SD 1.5 + SDXL.
plakat animate --animatediff --model sd15 \
    --from "a glowing neon dragon, cyberpunk alley, rain" \
    --control-spec 'openpose:video=./reference.mp4:strength=0.9' \
    --frames 16 --format mp4 --gif-delay-ms 80
```

Compose `video=` with the static `image=` / `from=` modes — each
`--control-spec` is independent. A typical recipe: one depth `video=`
controls macro motion, a canny `image=` from a frozen frame locks
edges:

```bash
plakat animate --animatediff --model sd15 \
    --from "watercolor rendering of a runner, soft light" \
    --control-spec 'depth:video=./jog.mp4:strength=0.7' \
    --control-spec 'canny:image=./reference.png:strength=0.3' \
    --frames 32 --window-size 16 --window-overlap 4 --format mp4
```

When `--frames` exceeds `--window-size`, sliding-window long-form
slices the video CN stack per window (no re-decoding). Input video
length is independent of `--frames`: short videos are tail-padded,
long videos are uniformly sub-sampled.

### v0.28: Bund scripting bridge

```bund
"sd15"   plakat.load
"true"   "animate_lcm"     plakat.config.set
16       "animate_frames"  plakat.config.set
"a watercolor cottage at dawn" "./out" plakat.animate
// → ./out/frame-0000.png … ./out/frame-0015.png + sidecars
```

Run with `plakat run my-anim.bund`. See [`SCRIPTING.md`](SCRIPTING.md)
for the full host-word reference.

### v0.28: inspect a motion adapter

```bash
plakat motion-adapter list                            # known + community
plakat motion-adapter info wangfuyun/AnimateLCM       # full dump
```

### v0.29: HJSON scenario batches

```hjson
{
  model: sd15
  type: animatediff       # scenario default — every task is animate
  frames: 16
  lcm: true               # 4-step AnimateLCM
  format: gif
  out: ./out/animations
  scene:   [ { name: dawn,  prompt: "at dawn" } ]
  weather: [ { name: mist,  prompt: "misty" } ]
  tasks: [
    { name: cottage, scene: dawn, weather: mist, prompt: "a watercolor cottage" }
    { name: knight,  scene: dawn, weather: mist, prompt: "a knight in a forest",
      frames: 24, format: mp4 }   # per-task overrides
  ]
}
```

```bash
plakat scenario animate.hjson --dry-run    # preview the plan
plakat scenario animate.hjson              # render every task
plakat scenario animate.hjson --resume     # skip already-rendered tasks
plakat scenario animate.hjson --only fox   # render one task
```

Per-task overrides for `frames`, `window-size`, `window-overlap`,
`lcm`, `motion-lora`, `motion-lora-scale`, `format`, `gif-delay-ms`
compose with scenario-level defaults. ControlNet per-task via the
existing `control:` / `controls:` fields.

### v0.29: SDXL `plakat.animate` in Bund

```bund
"sdxl" plakat.load
16 "animate_frames" plakat.config.set
1024 "width" plakat.config.set
1024 "height" plakat.config.set
"a knight in a forest, oil painting" "./out" plakat.animate
```

Same scripting surface as v0.28's SD 1.5 path, now with the SDXL
beta motion adapter. AnimateLCM remains SD 1.5 only (no public
SDXL repo).

### v0.29: format dispatch from Bund

```bund
"mp4" "animate_format" plakat.config.set
```

`animate_format` (`frames | gif | mp4 | webm | all`) closes the
final v0.28 Bund surface gap. MP4 / WebM need ffmpeg on `$PATH`.

## Architecture

### Motion adapter

| Variant | Repo | Block channels | Mid? | Modules |
|---|---|---|---|---|
| V3 SD 1.5 | `guoyww/animatediff-motion-adapter-v1-5-3` | `[320, 640, 1280, 1280]` | no | 16 |
| SDXL beta | `guoyww/animatediff-motion-adapter-sdxl-beta` | `[320, 640, 1280]` | no | 12 |
| AnimateLCM (v0.28) | `wangfuyun/AnimateLCM` | `[320, 640, 1280, 1280]` | yes | 17 |

All three share the same `MotionAdapterConfig` schema. Differences:

- `block_out_channels` matches each base UNet's block layout
  (SD 1.5 has 4 blocks; SDXL has 3).
- AnimateLCM flips `use_motion_mid_block` to true (V1/V2-style),
  adding 1 mid-block motion module — 17 total instead of V3's 16.

Adapter weights download to `$PLAKAT_CACHE_DIR/huggingface/hub/` on
first use (~1.4 GB for V3 / AnimateLCM, ~1.5 GB for SDXL beta).

Inspect any of the supported adapters via `plakat motion-adapter
info REPO` (config dump + per-block tensor breakdown + detected
base family).

### Per-block motion modules

Each `motion_modules.{j}` slot in the safetensors corresponds to
one `TemporalTransformer`:

- `GroupNorm` → `Linear` proj_in
- One inner `transformer_blocks.0`:
  - `norm1` + `attn1` — temporal self-attention across the frame axis
  - `norm2` + `attn2` — second attention slot (identity in V3/SDXL beta
    since motion is text-agnostic)
  - `norm3` + GEGLU FFN
- `Linear` proj_out
- Learnable positional embedding (`pos_embed.pe`, shape `(1, 32, channels)`)

The config field `motion_layers_per_block` (= 2) means **two
`motion_modules.{j}` slots per UNet block**, not two inner
transformer_blocks. Each motion module has exactly one
`transformer_blocks.0`.

Forward shape: `(B*F, C, H, W) → (B*H*W, F, C) → attention across
F → (B*F, C, H, W)` + residual.

### Block-boundary splice

Both `Sd15MotionUNet` and `SdxlUNet2DConditionModel::forward_with_motion`
splice motion modules at the **output boundary** of each down/up
block. Per-block motion modules apply sequentially (in V3 / SDXL
beta: two per block).

**Tradeoff vs the faithful diffusers `UNetMotionModel`** (which
interleaves motion modules per resnet+attn layer inside each block):
fewer LOC vendored, simpler parity test (`motion_modules: None` is
bit-identical to the stock UNet), but the skip-connection residuals
saved from inside each down block are not motion-aware. Real-world
quality validation against this approximation is the user-machine
acceptance step — escalation to per-layer vendoring lives in the
v0.27 RFC §3.2 escalation budget.

### Motion LoRAs

Motion LoRAs from the community (e.g.
`guoyww/animatediff-motion-lora-zoom-in`) target the motion adapter's
attention tensors. They use bare PEFT-style keys (no `lora_unet_` /
`text_encoder.` prefix):

```text
down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_q
  .lora.{down,up}.weight
```

Loaded via the `MergeTarget::MOTION_ADAPTER` variant of
`merge_loras_into_weights`. The merged safetensors lands in a
detached tempfile (`NamedTempFile::keep()`) that lives for the
`MotionAdapter`'s lifetime; OS reclaims at process exit.

Stacking: pass `--motion-lora SPEC` multiple times. Same LoraSpec
grammar as `--lora`:
- `hf:user/repo:0.7`
- `civitai:NNNNNN:0.5`
- `civitai-version:NNNNNN:0.8`
- `/local/path/file.safetensors:0.6`

### ControlNet conditioning (v0.27 phases 3 + 4)

Single conditioning image, same hint applied to every frame.

The pipeline pre-tiles the `(1, 3, H, W)` conditioning to the
per-step batch (`2F` with CFG, `F` without) before the denoise
loop. Each step runs ControlNet once at full batch with the
replicated latents + replicated text embeddings (SDXL also gets
pooled + add_time_ids), producing down + mid residuals that plug
straight into the motion UNet's existing residual hooks.

**Multi-CN stacking** (v0.28 phase 0): each `--control-spec`
stacks one conditioner; residuals from every conditioner sum
per denoise step inside the motion UNet. The spec grammar mirrors
`plakat generate`: `KIND[:option=value]*` with `KIND` ∈
`depth / canny / openpose / lineart / softedge` and options
`image=PATH`, `from=PATH`, `strength=F`, `start=F`, `end=F`.

CLI flags:
| Flag | Purpose |
|---|---|
| `--control KIND` | (legacy single-CN) `depth` / `canny` / `openpose` / `lineart` / `softedge` |
| `--control-image PATH` | Pre-rendered conditioning |
| `--control-from PATH` | Auto-annotate this image (mutex with `--control-image`) |
| `--control-strength F` | Residual scale (default 1.0) |
| `--control-spec SPEC` (v0.28, repeatable) | Multi-CN spec grammar; mutex with the legacy flags above |

```bash
# Single CN via the legacy flags
plakat animate --animatediff --model sd15 \
    --control depth --control-image ./d.png ...

# Multi CN via the spec form
plakat animate --animatediff --model sdxl \
    --control-spec 'depth:image=./d.png:strength=0.8' \
    --control-spec 'canny:from=./source.jpg:strength=0.4' ...
```

### Long-form sliding window (v0.27 phases 5 + 6)

V3's `motion_max_seq_length = 32` is a hard cap on a single window —
the positional embedding only has 32 rows. Long-form mode chains
overlapping windows with linear-ramp latent-space blend:

```
total_frames = 64, window_size = 16, window_overlap = 4
↓
window 0: frames 0..16   (frames 12..16 overlap with window 1)
window 1: frames 12..28  (frames 24..28 overlap with window 2)
window 2: frames 24..40  (frames 36..40 overlap with window 3)
window 3: frames 36..52  (frames 48..52 overlap with window 4)
window 4: frames 48..64
```

Per-window seed: `seed + win_i * window_size`. Distinct noise per
window plus the blended overlap region produces visual continuity
across boundaries.

Blend math (linear ramp; k = 0..overlap):
```
t = (k + 1) / (overlap + 1)
out[k] = (1 - t) * existing[k] + t * new[k]
```
Endpoint clipping (1/(N+1) and N/(N+1) rather than 0 and 1) keeps
both sides contributing at the seam.

CLI flags:
| Flag | Default | Purpose |
|---|---|---|
| `--frames N` | 16 | Total output frames; `> --window-size` engages sliding |
| `--window-size W` | 16 | Per-window frame count; must be ≤ 32 (V3 cap) |
| `--window-overlap O` | 4 | Cross-fade region in frames; must be < `--window-size` |

When `--frames ≤ --window-size`, sliding mode is a thin pass-through
to single-window inference (no overhead).

**Quality caveat (RFC §11)**: this is Approach B from the RFC
design space (per-window independent denoising + post-hoc latent
blend). FreeNoise / FreeInit style shared-noise schemes are
deferred to v0.28+ if seams are visibly bad on real prompts.

### AnimateLCM (v0.28 phase 1)

`--lcm` switches the motion adapter to
`wangfuyun/AnimateLCM`, the scheduler to LCM, and applies
defaults `--steps 4 --guidance 1.5` for a **~5× speedup** vs V3
+ DDIM at 20 steps. User-supplied `--steps` / `--guidance` take
precedence — `--lcm --steps 8` gets 8-step LCM at 2× the runtime
of the default for higher quality.

AnimateLCM is SD 1.5 only — the SDXL AnimateLCM repo isn't
publicly available. `--lcm` + `--model sdxl` bails loud with
the deferral pointer.

Composes with motion LoRAs, ControlNet (single + multi), and
sliding-window long-form. The motion-LoRA tensor key convention
is the same as V3 / SDXL beta, so V3-targeting motion LoRAs from
the community apply cleanly to AnimateLCM too.

| Mode | Steps | Wall-clock @ 16 frames × 512² × bf16 GPU |
|---|---|---|
| V3 + DDIM (default) | 20 | ~4 min |
| AnimateLCM + LCM | 4 | ~50 s |
| AnimateLCM + LCM (high quality) | 8 | ~95 s |

(Approximate — actual numbers depend on GPU. The speedup ratio
is what matters.)

### Bund scripting bridge (v0.28 phase 2)

`plakat.animate ( prompt out_dir -- )` exposes AnimateDiff to
the Bund scripting layer. Reads frames + window + LCM flag +
size + steps + guidance + scheduler + controls from `ctx.config`
and `ctx.controlnets`. Writes `frame-NNNN.png` plus JSON sidecars
to the given dir, matching the CLI animate output layout.

Four new config keys via `plakat.config.set`:
- `animate_frames` (default 16)
- `animate_window_size` (default 16, ≤ 32)
- `animate_window_overlap` (default 4)
- `animate_lcm` (default false)

```bund
"sd15" plakat.load
"true" "animate_lcm" plakat.config.set
"watercolor" plakat.look.apply             // v0.25 preset
"depth" "./d.png" plakat.controlnet.add
"a fox in a meadow" "./out" plakat.animate
```

Composes with every config / lora / look / genre / controlnet
mutation. SD 1.5 only in v0.28 — SDXL animate in scripting needs
a separate cache slot (v0.29). See [`SCRIPTING.md`](SCRIPTING.md).

### Inspection: `plakat motion-adapter` (v0.28 phase 3)

```bash
plakat motion-adapter list
plakat motion-adapter info <REPO>
```

`list` prints the plakat-supported repos (V3 SD 1.5, SDXL beta,
AnimateLCM) plus community refs (V1 / V2 SD 1.5, Hotshot-XL).
`info` downloads + dumps the adapter's config + per-block
tensor breakdown + detected base family. Routes through the
same loader paths as the real animate runs, so cache behavior
is identical.

## Output formats

Every animate mode (prompt-lerp + AnimateDiff, SD 1.5 + SDXL,
single-window + long-form) accepts `--format FMT`:

| `--format` | Effect | Requires |
|---|---|---|
| `frames` (default) | `<out>/frame-NNNN.png` per frame | nothing |
| `gif` | + animated GIF via the `image` crate | nothing |
| `mp4` | + MP4 via ffmpeg (libx264 + yuv420p + faststart) | ffmpeg on `$PATH` |
| `webm` | + WebM via ffmpeg (libvpx-vp9 + CRF 30) | ffmpeg on `$PATH` |
| `all` | every format above | ffmpeg on `$PATH` |

Install ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `apt install ffmpeg`
- Windows: `scoop install ffmpeg`

## Memory budget

Approximate peak VRAM for 16 frames × bf16 on GPU:

| Backbone | 512² | 768² | 1024² |
|---|---|---|---|
| SD 1.5 + V3 | ~9 GB | ~14 GB | ~22 GB |
| SDXL + beta | ~14 GB | ~20 GB | ~30 GB |
| SD 1.5 + V3 + CN | ~12 GB | ~18 GB | OOM on 24 GB |
| SDXL + beta + CN | ~18 GB | ~25 GB | OOM on 24 GB |

Tactics if you OOM:
- Drop frame count (16 → 8)
- Drop resolution (1024² → 768² or 512²)
- Drop the ControlNet
- Use `--window-size 8 --window-overlap 2` for long-form to halve
  the per-window batch

## Limitations

- **No SD 2.1 / Flux / SD3 motion adapters** upstream. SD 1.5 +
  SDXL only.
- **AnimateLCM is SD 1.5 only**. The SDXL AnimateLCM repo isn't
  publicly available; `--lcm --model sdxl` bails. v0.29 if
  upstream changes.
- **Per-frame video control deferred**. v0.28/v0.29 ship
  same-hint-every-frame conditioning; video-to-video (a depth /
  canny video as control) is v0.30+ territory.
- **Mixed-kind scenarios pay both pipeline costs**. A scenario
  with some `type: generate` and some `type: animatediff` tasks
  holds both the t2i and animate pipelines resident. All-animate
  and all-generate scenarios pay only one. v0.30+ optimization.
- **No img2img / inpaint hooks** on the animate path. Use
  `plakat generate` / `plakat img2img` for those if you need a
  single frame with the full adapter stack.
- **Block-boundary motion splice** (not faithful per-layer
  diffusers `UNetMotionModel`). Documented quality concern in
  RFC v0.27 §3.2; upgrade path budgeted.

## See also

- [`RFC_v0.29_BATCH_PRODUCTIVITY.md`](RFC_v0.29_BATCH_PRODUCTIVITY.md)
  — v0.29 design doc, two locked decisions, 6-phase plan
  (animate-in-scenarios + SDXL scripting + Bund format key).
- [`RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md`](RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md)
  — v0.28 design doc, two locked decisions, 6-phase plan.
- [`RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md`](RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md)
  — v0.27 design doc, four locked decisions, 8-phase plan.
- [`RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md`](RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md)
  — v0.26 infrastructure RFC, the eight original AnimateDiff
  decisions.
- [`Tutorials/ANIMATE_TUTORIAL.md`](Tutorials/ANIMATE_TUTORIAL.md)
  — narrative walkthrough (prompt-lerp + AnimateDiff).
- [`SCRIPTING.md`](SCRIPTING.md) — `plakat.animate` host word +
  the four `animate_*` config keys.
- [AnimateDiff paper (Guo et al., 2023)](https://arxiv.org/abs/2307.04725)
  — original architecture.
- [`guoyww/animatediff-motion-adapter-v1-5-3`](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3)
  — V3 SD 1.5 adapter.
- [`guoyww/animatediff-motion-adapter-sdxl-beta`](https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta)
  — SDXL beta adapter.
- [`wangfuyun/AnimateLCM`](https://huggingface.co/wangfuyun/AnimateLCM)
  — v0.28 4-step LCM motion adapter.
