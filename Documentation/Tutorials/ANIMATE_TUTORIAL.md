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
- A working `plakat generate` against SD 1.5 / SD 2.1 / SDXL. The
  animate path now supports the full SD family; Flux + SD3 use T5
  / rectified-flow and need their own machinery, deferred to a
  follow-up.
- ~3 GB free for the SD 1.5 weights on first run (one-time cost);
  ~7 GB for SDXL.

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

**No LoRA / ControlNet / refiner in animate.** The animate path
runs a narrow denoise loop without those adapters wired —
keeping the implementation simple. If you need them, generate
each frame manually via `plakat generate` and bundle into a GIF
externally (the `image` crate's GIF encoder is what plakat uses
internally).

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

## 8. Limitations

- **SD-family only** (SD 1.5 / SD 2.1 / SDXL). SDXL animate (added
  in v0.18) lerps the dual CLIP-L + CLIP-G hidden states plus the
  pooled `add_text_embeds` micro-conditioning each frame; expect
  ~2-3× the per-frame cost of SD 1.5 in exchange for SDXL's
  trained resolution + visual quality.
- **Flux + SD3 deferred.** Their T5 + rectified-flow paths need
  separate machinery.
- **No CFG variations across frames.** Guidance is constant.
- **No keyframe-style trajectories.** It's a single A → B lerp,
  not a multi-keyframe spline. Chain multiple `plakat animate`
  runs + concat with ffmpeg for richer trajectories.

## Where to next

- **`GENERATE_TUTORIAL.md`** — the t2i foundation everything
  here builds on.
- **`GENERATE.md`** — full `plakat animate` flag reference (every
  knob covered in this tutorial is also documented there for
  copy-paste lookup).
- **External tooling** — ffmpeg for MP4 / WebM bundling,
  imagemagick `montage` for grids, gifski for higher-quality
  GIFs than the `image` crate produces.
