# Select objects (`plakat segment`)

Click an object and get a clean **mask** of it — the enabler for "compose & edit
scenes". Wraps **MobileSAM** (Segment Anything, ~40 MB weights, auto-downloaded);
runs in ~0.4 s on Metal/CPU. The mask feeds any `--mask` consumer (inpaint /
img2img), so *select → remove / replace / swap background* composes from pieces
plakat already owns.

## Basic — click to select

```bash
plakat segment --in photo.png --out mask.png --point 0.5,0.4
```

- `--point X,Y` — a click. Coords are normalized `0..1` by default, or pixels if a
  value exceeds 1 (`--point 0.5,0.4` or `--point 512,400`). Append `:bg` to
  *exclude* a region: `--point 0.1,0.1:bg`.
- Repeatable — add points to refine. A single click on an ambiguous spot (a face)
  can grab a sub-part; 2–3 points down a figure give a solid mask.
- The output PNG is **white = selected, black = not** — exactly what `--mask` wants.

## Mask-for-edit options

| Flag | What |
|---|---|
| `--invert` | select everything *except* the object (e.g. "change the background, keep the subject") |
| `--grow N` | dilate the selection by N px — leaves a margin so a downstream inpaint doesn't repaint the subject's fringe |
| `--feather N` | soften the mask edge (gaussian) for a smooth inpaint blend |

## Edit with the mask — swap a background

```bash
# 1. select the subject, invert + grow so the mask covers the BACKGROUND
plakat segment --in portrait.png --out bg-mask.png \
  --point 0.5,0.45 --point 0.5,0.7 --invert --grow 12 --feather 8

# 2. repaint only the background, preserving the subject
plakat img2img portrait.png --prompt "the surface of the Moon, Earth rising" \
  --mask bg-mask.png --model sd15 --strength 0.9
```

That's the committed corpus proof (`corpus/segment.sh`): astronaut → the Moon.

## Select by depth — no clicks (`--depth-band`)

Instead of (or together with) point clicks, select pixels by their **depth**.
`--depth-band LO,HI` runs Depth-Anything-V2 and keeps pixels whose normalized
depth falls in `[LO, HI]`, where **1.0 = nearest the camera, 0.0 = farthest**:

```bash
# Foreground only — the nearest subject, no clicks at all
plakat segment --in photo.png --out fg.png --depth-band 0.45,1.0

# The far background instead
plakat segment --in photo.png --out bg.png --depth-band 0.0,0.4
```

It composes with everything else (`--invert`, `--grow`, `--feather`), and you can
**combine it with `--point` to intersect** — "this specific object, but only the
part of it that's near the camera":

```bash
plakat segment --in photo.png --out near-subject.png \
  --point 0.5,0.5 --depth-band 0.5,1.0
```

The depth model is small (runs even on CPU), so this is cheap. Proof:
`corpus/segment.sh` stage 3 lifts the foreground astronaut click-free
(`corpus/images/segment/depth-foreground.png`).

## Tips

- **Subject choice matters.** A cleanly-separable subject swaps cleanly; a figure
  fused with its props (a dock captain tangled in rope rigging) keeps the props —
  even a perfect mask preserves what's *part of* the subject.
- For a content-aware whole-subject cut-out without clicking, `plakat transparent
  --matte` (U2Net) is often simpler; `segment` is for *picking a specific object*.
- See also [`COMPOSE_TUTORIAL.md`](COMPOSE_TUTORIAL.md) (stack the cut-outs into a
  scene) and [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md) (the `--mask` inpaint).
