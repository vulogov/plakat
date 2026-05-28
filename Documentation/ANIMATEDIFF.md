# AnimateDiff (v0.27)

`plakat animate --animatediff` renders **motion-coherent N-frame
sequences** from a single prompt using AnimateDiff motion adapters
spliced into the SD UNet. Different from `plakat animate`'s default
prompt-morph mode, which interpolates between two prompts without
temporal-attention coherence.

**v0.27 ships the full AnimateDiff feature set:** SD 1.5 + SDXL, both
with optional ControlNet conditioning and a sliding-window
long-form mode that lifts the V3 32-frame cap.

| Capability | SD 1.5 | SDXL |
|---|---|---|
| Inference dispatch | ✓ phase 0 (v0.27) | ✓ phase 2 |
| Motion adapter | V3 (`guoyww/animatediff-motion-adapter-v1-5-3`) | beta (`guoyww/animatediff-motion-adapter-sdxl-beta`) |
| Motion LoRAs | ✓ phase 4 (v0.26) | ✓ phase 1 |
| ControlNet | ✓ phase 3 | ✓ phase 4 |
| Long-form sliding window | ✓ phase 5 | ✓ phase 6 |
| Per-block motion modules | 16 (4 down × 2 + 4 up × 2) | 12 (3 down × 2 + 3 up × 2) |
| Hard frame cap per window | 32 | 32 |
| Cross-fade long-form total | ~256 frames practical | ~256 frames practical |

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

## Architecture

### Motion adapter

| Variant | Repo | Block channels | Modules |
|---|---|---|---|
| V3 SD 1.5 | `guoyww/animatediff-motion-adapter-v1-5-3` | `[320, 640, 1280, 1280]` | 16 |
| SDXL beta | `guoyww/animatediff-motion-adapter-sdxl-beta` | `[320, 640, 1280]` | 12 |

Both share the same `MotionAdapterConfig` schema. The only meaningful
difference is `block_out_channels`, which matches each base UNet's
block layout. Adapter weights download to
`$PLAKAT_CACHE_DIR/huggingface/hub/` on first use (~1.4 GB for V3,
~1.5 GB for SDXL beta).

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

Multi-conditioner is honoured by the existing
`pipelines::controlnet::sum_controlnet_residuals` helper but isn't
wired through `--animatediff` yet — v0.27 ships single-CN only.
Extras log a warning and are skipped.

CLI flags:
| Flag | Purpose |
|---|---|
| `--control KIND` | `depth` / `canny` / `openpose` / `lineart` / `softedge` |
| `--control-image PATH` | Pre-rendered conditioning |
| `--control-from PATH` | Auto-annotate this image (mutex with `--control-image`) |
| `--control-strength F` | Residual scale (default 1.0) |

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
- **Single ControlNet per run**. Multi-CN sum exists in
  `sum_controlnet_residuals` but isn't wired through animate yet.
- **Per-frame video control deferred**. v0.27 ships
  same-hint-every-frame conditioning; video-to-video (a depth
  video as control) is v0.28+ territory.
- **No img2img / inpaint hooks** on the animate path. Use
  `plakat generate` / `plakat img2img` for those if you need a
  single frame with the full adapter stack.
- **Block-boundary motion splice** (not faithful per-layer
  diffusers `UNetMotionModel`). Documented quality concern in
  RFC §3.2; upgrade path budgeted.

## See also

- [`RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md`](RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md)
  — v0.27 design doc, four locked decisions, 8-phase plan.
- [`RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md`](RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md)
  — v0.26 infrastructure RFC, the eight original AnimateDiff
  decisions.
- [`Tutorials/ANIMATE_TUTORIAL.md`](Tutorials/ANIMATE_TUTORIAL.md)
  — narrative walkthrough (prompt-lerp + AnimateDiff).
- [AnimateDiff paper (Guo et al., 2023)](https://arxiv.org/abs/2307.04725)
  — original architecture.
- [`guoyww/animatediff-motion-adapter-v1-5-3`](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3)
  — V3 SD 1.5 adapter.
- [`guoyww/animatediff-motion-adapter-sdxl-beta`](https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta)
  — SDXL beta adapter.
