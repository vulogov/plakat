# `plakat animate` — prompt-morph animations

`plakat animate` interpolates between two prompts to produce a
smooth N-frame sequence. The model gradually transitions from
"this prompt" to "that prompt" while the noise stays constant, so
the composition lerps rather than flickering. Output is a series
of `frame-NNNN.png` files in your `--out` directory, optionally
bundled into an `animation.gif` for sharing.

This tutorial covers the basic prompt-morph workflow, GIF
bundling, the seed-locking trick that keeps the animation smooth,
and the trade-offs between frame count + step count + size.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md). You
  should be comfortable with `--prompt`, `--seed`, `--steps`,
  and the relationship between seed and noise.
- A working `plakat generate` against SD 1.5 / SD 2.1 / SDXL or
  Flux Dev / Schnell. The animate path supports the full SD
  family plus Flux Dev / Schnell (v0.20) and SD3 / 3.5 (v0.26;
  see §9).
- ~3 GB free for the SD 1.5 weights on first run (one-time cost);
  ~7 GB for SDXL; ~24 GB for Flux Dev BF16.

## 1. Your first morph

The simplest invocation:

```bash
plakat animate \
    --from "a photo of a fox in a meadow" \
    --to "a photo of a cat in a meadow" \
    --frames 16 \
    --seed 42 \
    --model sd15 \
    --out ./fox_to_cat
```

What this does:

1. Loads SD 1.5 once.
2. Encodes both prompts through CLIP-L → two `(1, 77, 768)`
   hidden-state tensors.
3. For frame `i` in `0..16`, linearly interpolates the two
   tensors at `t = i / 15`. Frame 0 = fox; frame 15 = cat;
   frame 7 ≈ 50/50.
4. Runs the full denoise loop per frame with the lerped
   embedding + the same seed, so the noise stays constant and
   only the prompt-driven trajectory varies.
5. Saves `frame-0000.png` ... `frame-0015.png` into `./fox_to_cat/`.

Open the directory + flip through the frames — you should see a
smooth fox-to-cat morph. The midpoint (frame 7-8) often produces
ambiguous "creature with traits of both" outputs.

## 2. Bundling into a GIF

Add `--gif`:

```bash
plakat animate \
    --from "a peaceful forest stream" \
    --to "a dramatic mountain waterfall" \
    --frames 24 \
    --seed 100 \
    --out ./forest_to_mountain \
    --gif
```

plakat writes `forest_to_mountain/animation.gif` alongside the
individual frames. Default frame delay is 100 ms (10 fps). Tune
with `--gif-delay-ms`:

```bash
# Cinematic ~24 fps
plakat animate ... --gif --gif-delay-ms 41

# Slow contemplative pace
plakat animate ... --gif --gif-delay-ms 200
```

The GIF loops infinitely. The frames stay on disk after bundling
so you can also re-bundle elsewhere (ffmpeg → mp4, montage → grid,
etc.).

## 3. Frame count vs. smoothness

| Frames | Look | Notes |
|---|---|---|
| 4-8 | Choppy slideshow | Useful for seeing key intermediates fast. |
| 12-16 | Smooth-ish morph | Good default; the math is clear at each step. |
| 24-32 | Cinematic | Each consecutive pair is very similar; great for shareable GIFs. |
| 48+ | Smooth at any speed | Diminishing returns — the embedding lerp is linear, not exponential, so most of the visual change happens in the middle half regardless. |

Generation cost scales linearly: 16 frames at 20 steps each is
16 × the per-frame cost of `plakat generate --steps 20`. SD 1.5
at 512² + 20 steps is ~3-8s per frame on consumer hardware →
~50-130s for a 16-frame animation.

## 4. Frame size + steps trade-offs

Animations don't need per-frame perfection — temporal smoothness
across frames matters more than fine detail in one frame. Use
this to your advantage:

```bash
# Quick previews: small + few steps
plakat animate --from ... --to ... \
    --size 384x384 --steps 12 --frames 16

# Final renders: trained res + more steps
plakat animate --from ... --to ... \
    --size 512x512 --steps 28 --frames 24
```

A 384² draft at 12 steps × 16 frames runs in ~30s and tells you
whether the morph direction is interesting. Bump to 512² + 28
steps + 24 frames once you've locked the prompts.

## 5. Why the seed stays fixed

`plakat animate` shares one seed across every frame. The initial
random noise is the same; only the prompt-driven trajectory
changes between frames. Without this lock:

- Each frame's noise would be different.
- Adjacent frames would diverge in composition, not just content.
- The result would be a slideshow of unrelated images that
  happen to lerp through prompts — choppy and disorienting.

If you DO want a noise-driven sweep instead (different seeds, same
prompt), use `plakat generate --count N --seed S` — that gives
you `S`, `S+1`, ..., `S+N-1` as independent images.

## 6. Prompt patterns that morph well

Two prompts that share **structure + setting** but vary
**subject** or **attribute** produce the most coherent morphs:

✅ Good:
- "a photo of a {dog|cat|owl} in a meadow" (single attribute lerp)
- "a watercolor of a city street in {morning|evening|night}" (time-of-day)
- "a portrait of a person, {happy|sad|angry|surprised} expression" (emotion)

❌ Painful:
- "a fox" → "a spaceship" (no shared structure; the lerp has to
  invent geometry for the midpoints).
- "abstract art" → "a photograph" (style lerp introduces
  ambiguous artifacts).

The lerp is linear in CLIP token-embedding space, not in pixel
space — so semantically-close pairs morph cleanly, semantically-
distant pairs produce ugly midpoints.

## 7. Tips

**Pick your seed first.** Run `plakat generate --from`'s prompt
once with a candidate seed and check the composition. Once you've
got one you like, `plakat animate --seed THAT_SEED` keeps the
morph anchored to that composition.

**Negative prompt is shared.** `--negative "blurry, distorted"`
applies to every frame. Useful for keeping the morph cleaner
through the noisy midpoints.

**No LoRA / refiner on the prompt-lerp path.** The §1-§7 animate
path runs a narrow denoise loop without those adapters wired —
keeping the implementation simple. If you need them with prompt-
lerp, generate each frame manually via `plakat generate` and
bundle into a GIF externally. AnimateDiff mode (§10) supports
motion LoRAs natively and ControlNet via `--control` (§11).

**Frame metadata** (v0.18). Every `frame-NNNN.png` carries an
Auto1111-compatible `parameters` PNG tEXt chunk plus a sibling
`frame-NNNN.json` sidecar. The chunk's prompt field reads
`lerp(0.4375): "from prompt" | "to prompt"` so dragging a frame
into A1111 / Civitai / ComfyUI shows the morph state at that point;
the JSON sidecar carries the structured `Lerp t` / `Animate from`
/ `Animate to` extras so you can re-render any frame standalone.
Pass `--no-metadata` to skip both.

**Crash recovery with `--resume`** (v0.19). Long animates can
crash on frame 23 of 24 — Ctrl-C, OOM, transient I/O failure.
Pass `--resume` on the rerun and plakat scans `<out>/frame-NNNN.png`
over the requested range, skips the frames already on disk, and
re-runs only what's missing. The lerp parameter `t` per frame
index is recomputed identically, so the skipped frames stay
consistent with the freshly-rendered ones.

```bash
# Original run; crashes on frame 23
plakat animate --from A --to B --frames 24 --gif --out ./morph

# Recovery: re-run the same command + --resume
plakat animate --from A --to B --frames 24 --gif --out ./morph --resume
#   ✓ skips 22 already-rendered frames, re-runs frames 23 (the
#     crash point) and 24, bundles the GIF
```

Mirrors the scenario `--resume` semantics added in v0.17.

## 8. Flux animate (v0.20)

`--model flux-dev` and `--model flux-schnell` work the same way
the SD-family path does — pre-encodes both endpoints, lerps the
text embeddings per frame, renders. Flux uses CLIP-L pooled +
T5-XXL hidden states; the T5 encode is the expensive part, so
amortising it across frames is the whole point.

```bash
plakat animate \
    --from "an oil painting of a fox in a meadow" \
    --to   "an oil painting of a cat in a meadow" \
    --frames 24 --seed 42 --steps 20 --guidance 3.5 \
    --model flux-dev --size 1024x1024 --out ./flux_morph --gif
```

Two Flux-specific gotchas worth knowing:

- **`--guidance` defaults to 7.5 (right for SD; wrong for Flux).**
  Flux is guidance-distilled — the CFG signal is a scalar input
  to the model rather than a batched uncond+cond pass. Use
  `--guidance 3.5` on `flux-dev`, `--guidance 0` on
  `flux-schnell`. The default 7.5 won't blow up but produces
  over-baked output.
- **`--negative` is a no-op on Flux.** Since there's no CFG
  batching, the unconditional branch can't be steered. Move
  suppressors into the positive prompts; animate warns if you
  pass `--negative` on a Flux variant.

Flux Kontext / Fill / Canny / Depth aren't supported (they need
a reference image per call that doesn't fit the `--from` /
`--to` model). SD3 / SD3.5 are deferred to a follow-up — they
need three-encoder lerp (CLIP-L + CLIP-G + T5) plus the
rectified-flow MMDiT integrator wiring.

T5 cost: ~10s extra per frame on CPU vs ~0.1s on a 24 GB GPU.
For long Flux animations, use `--resume` aggressively — a
crashed Flux animate at frame 30 of 32 is much more painful to
restart than the equivalent SD 1.5 run.

## 9. SD3 / SD3.5 animate (v0.26)

The v0.20-era SD3 bail is gone. v0.26 wires `plakat animate`
through the SD3 / SD3.5 pipeline using the same A → B lerp
contract, adapted to SD3's three text encoders (CLIP-L +
CLIP-G + T5):

```bash
plakat animate --model sd35-medium \
    --from "a quiet temple at dawn" \
    --to   "a quiet temple at sunset" \
    --frames 16 --gif-delay-ms 125
```

Internals: pre-encode both endpoint prompts into
`(pooled_y, joint_context)` once, lerp per frame, run a single
MMDiT inference per frame with rectified-flow scheduling, VAE
decode. Pure text-to-image morph contract — no img2img / mask /
ControlNet.

Works on all four SD3 variants: `sd35-medium`, `sd35-large`,
`sd35-large-turbo`, `sd3-medium`. Memory budget matches the
non-animate generate (SD3 weights + per-frame latent buffer).

## 10. AnimateDiff (v0.27 — feature complete)

Different from the §1-§7 prompt-lerp morph mode. AnimateDiff uses
a downloaded motion adapter that adds **temporal attention** to the
SD UNet — every frame's denoise gets to see every other frame's
denoise state through the F dimension, producing motion that
holds together across frames rather than morphing between
independent renders.

v0.27 ships the full AnimateDiff picture for SD 1.5 + SDXL, both
with ControlNet and a sliding-window long-form mode:

```bash
# SD 1.5 baseline — 16-frame motion-coherent loop at 512²
plakat animate --animatediff --model sd15 \
    --from "a watercolor cottage at dawn, gentle wind" \
    --frames 16 --format mp4

# SDXL at training resolution — same flags, larger output
plakat animate --animatediff --model sdxl \
    --from "a knight in a forest, oil painting" \
    --frames 16 --size 1024x1024 --format mp4

# Motion LoRA: ride a zoom-in trajectory
plakat animate --animatediff --model sd15 \
    --from "a wizard's tower at sunset" \
    --motion-lora hf:guoyww/animatediff-motion-lora-zoom-in:0.8 \
    --frames 16 --format mp4

# Stack motion LoRAs (the per-spec :scale stacks with --motion-lora-scale)
plakat animate --animatediff --model sd15 --from "..." \
    --motion-lora hf:guoyww/animatediff-motion-lora-pan-left:0.7 \
    --motion-lora hf:guoyww/animatediff-motion-lora-zoom-in:0.5
```

Hard frame-per-window cap: 32 (`motion_max_seq_length`). Default
window is 16 frames (where V3 was trained). For longer outputs,
see §13.

Cold-cache download: ~1.4 GB for V3 SD 1.5, ~1.5 GB for SDXL beta.
Cached afterward under `$PLAKAT_CACHE_DIR/huggingface/hub/`.

## 11. AnimateDiff + ControlNet (v0.27)

The same conditioning signal applies to every frame — depth map,
canny edges, openpose skeleton, lineart, or HED softedge. Per-frame
video control (a depth video as guide) is v0.28+ territory; v0.27
ships single-image-applied-to-every-frame.

```bash
# Depth-guided motion (camera holds, subject moves through fixed scene)
plakat animate --animatediff --model sd15 \
    --from "a fox in a snowy meadow" \
    --control depth --control-image ./depth.png \
    --frames 16 --format mp4

# Auto-annotate the conditioning from a source photo
plakat animate --animatediff --model sd15 \
    --from "a watercolor of {SUBJECT}" \
    --control canny --control-from ./reference.jpg \
    --frames 16 --format mp4

# SDXL with strength dial
plakat animate --animatediff --model sdxl \
    --from "a knight standing in a forest, oil painting" \
    --control depth --control-image ./depth.png --control-strength 0.75 \
    --frames 16 --size 1024x1024 --format mp4
```

ControlNet runs at the full per-step batch (2F with CFG = 32 for 16
frames), which is the main memory-driver beyond the motion UNet
itself. If you OOM at 1024² + CN, drop to 768² first, then to
`--frames 8`, then drop the ControlNet.

Five kinds supported: `depth`, `canny`, `openpose`, `lineart`,
`softedge`. The CN model picker resolves automatically based on
`--model` (SD 1.5 → lllyasviel + control_v11 family; SDXL → official
SDXL CN variants).

## 12. Long-form AnimateDiff via sliding window (v0.27)

V3's 32-frame `motion_max_seq_length` is a hard cap on a single
window. For longer outputs, `plakat animate --animatediff` chains
overlapping windows and blends them in latent space:

```bash
# 64-frame clip (~4-second at 16 fps): four windows, 4-frame overlap
plakat animate --animatediff --model sd15 \
    --from "a misty forest at dawn" \
    --frames 64 --window-size 16 --window-overlap 4 \
    --format mp4
```

The math:

```
stride = window_size - window_overlap     # 12 for the defaults
windows: [0..16), [12..28), [24..40), [36..52), [48..64)
overlap region blended linearly per latent slot
```

Each window gets its own seed (`seed + win_i * window_size`) so
different windows produce distinct noise patterns; the blended
overlap region preserves visual continuity across boundaries.

Practical reach: ~64 frames produces clean output reliably;
~128–256 frames work but motion drift accumulates (the model
doesn't see past the current window). For longer-than-256, expect
to see the scene gradually re-converge to the prompt's "central"
interpretation.

Tuning:
- **Default overlap (4)** is the community sweet spot. Halving to 2
  speeds up generation (less redundant compute) at the cost of
  more visible seams.
- **Window size 16** is what V3 was trained on. Higher (24, 32)
  works but quality starts degrading; lower (8) gives smaller per-
  window memory but more windows total.
- Long-form composes with ControlNet — the same conditioning image
  applies to every frame across every window.

When `--frames ≤ --window-size`, long-form is a pass-through to the
single-window path (zero overhead from these flags).

See [`Documentation/ANIMATEDIFF.md`](../ANIMATEDIFF.md) for the
full reference + memory budget table + the architecture details
(motion module layout, block-boundary splice tradeoff,
ControlNet residual flow).

## 13. `--format` flag (v0.26)

Every animate mode (prompt-lerp on SD-family / Flux / SD3 +
AnimateDiff in every configuration) accepts `--format FMT`:

| `--format` | Effect | Requires |
|---|---|---|
| `frames` (default) | Per-frame PNGs `<out>/frame-NNNN.png` | nothing |
| `gif` | + animated GIF via the `image` crate | nothing |
| `mp4` | + MP4 via ffmpeg (libx264 + yuv420p + faststart) | ffmpeg on `$PATH` |
| `webm` | + WebM via ffmpeg (libvpx-vp9 + CRF 30) | ffmpeg on `$PATH` |
| `all` | every format above | ffmpeg on `$PATH` |

Install ffmpeg: macOS `brew install ffmpeg`, Ubuntu `apt install
ffmpeg`, Windows `scoop install ffmpeg`.

`--format gif` is equivalent to passing `--gif` (the legacy
v0.20 flag still works). When both are set, `--format` wins.

## 14. Limitations

**Prompt-lerp mode** (§1-§9):
- **SDXL** lerps the dual CLIP-L + CLIP-G hidden states plus the
  pooled `add_text_embeds` micro-conditioning each frame; expect
  ~2-3× the per-frame cost of SD 1.5 in exchange for SDXL's
  trained resolution + visual quality.
- **No CFG variations across frames.** Guidance is constant.
- **No keyframe-style trajectories.** It's a single A → B lerp,
  not a multi-keyframe spline. Chain multiple `plakat animate`
  runs + concat with ffmpeg for richer trajectories.

**AnimateDiff mode** (§10-§12):
- **SD 1.5 + SDXL only.** No SD 2.1 / Flux / SD3 motion adapters
  exist upstream.
- **Single ControlNet per run.** Multi-CN sum isn't wired through
  animate yet; use one conditioning at a time.
- **Same conditioning every frame.** Per-frame video control
  (e.g. a depth video) is v0.28+ territory.
- **No img2img / inpaint** on the animate path. Use
  `plakat generate` / `plakat img2img` for those if you need a
  single frame with the full adapter stack.
- **Block-boundary motion splice** rather than the faithful
  diffusers per-resnet+attn-layer splice. Quality concern
  documented in RFC §3.2; upgrade path budgeted in the v0.27
  cycle.

## Where to next

- **`GENERATE_TUTORIAL.md`** — the t2i foundation everything
  here builds on.
- **`GENERATE.md`** — full `plakat animate` flag reference (every
  knob covered in this tutorial is also documented there for
  copy-paste lookup).
- **`Documentation/ANIMATEDIFF.md`** (v0.27) — AnimateDiff
  architecture, motion adapter internals, ControlNet + long-form
  reference, memory budget table.
- **External tooling** — ffmpeg for MP4 / WebM bundling
  (now wired natively via `--format`), imagemagick `montage`
  for grids, gifski for higher-quality GIFs than the `image`
  crate produces.
