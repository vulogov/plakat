# IMG2IMG — image-to-image and inpaint

End-to-end runnable tutorial for `plakat img2img`. Every script
under `scripts/` is independent and self-contained — run them in
order for a walkthrough or jump straight to whichever one matches
your use case.

`plakat img2img` is a single subcommand with two modes:

| Mode | Trigger | Default strength | What it does |
|---|---|---|---|
| **img2img** | no `--mask` | 0.6 | Re-paint every pixel at the requested strength. |
| **inpaint** | `--mask PATH` | 1.0 | Re-paint only the pixels where the mask is white; preserve the rest. |

## What's in here

```
IMG2IMG/
├── README.md
├── inputs/
│   ├── landscape.png      ← procedurally drawn sample image
│   └── sky_mask.png       ← matching sky mask (white = sky)
└── scripts/
    ├── 01_img2img.sh           ← whole-image transform, no mask
    ├── 02_inpaint_sky.sh       ← replace sky only
    ├── 03_inpaint_ground.sh    ← same mask, --mask-invert → replace ground
    └── 04_strength_sweep.sh    ← three variants at strengths 0.3/0.5/0.7
```

The `inputs/` PNGs ship with the tutorial. Regenerate them at any
time with:

```bash
cargo run --release --example draw_img2img_sample
```

## Prerequisites

- A `plakat` binary built with GPU support. Apple Silicon:
  ```bash
  cargo build --release --features metal
  ```
  Linux + NVIDIA:
  ```bash
  cargo build --release --features cuda
  ```
  CPU-only works but is **very** slow for img2img (each script will
  take several minutes per output). Apple Silicon Mac at SD 1.5 +
  20 steps + Metal: ~5–10 s per image after first-run JIT.
- First run also downloads SD 1.5 weights (~4 GB) from HuggingFace.

The scripts source `scripts/_plakat.sh`, which finds the binary in
`$PATH` or under `target/{release,debug}/`. Run them from anywhere.

## Walkthrough

### 1. `01_img2img.sh` — pure img2img, no mask

```bash
./scripts/01_img2img.sh
```

Transforms `inputs/landscape.png` into a watercolor painting:

```bash
plakat img2img inputs/landscape.png \
    --prompt "soft watercolor landscape painting, ..." \
    --strength 0.55
```

Every pixel goes through the partial-strength denoise. Composition
stays roughly the same as the source (hill silhouette, sky band,
sun disc) but with a different aesthetic. Lower strengths preserve
more of the source; higher strengths let the model rewrite more.

### 2. `02_inpaint_sky.sh` — masked region edit

```bash
./scripts/02_inpaint_sky.sh
```

Same input, but `--mask inputs/sky_mask.png` confines the changes
to the sky region. Open the result and compare against the source:
the ground should be **pixel-identical**, only the sky has been
replaced with the stormy version from the prompt.

The mask file uses the standard convention:
- **White** = inpaint here
- **Black** = preserve this pixel
- **Gray** = blend proportionally

### 3. `03_inpaint_ground.sh` — invert the mask polarity

```bash
./scripts/03_inpaint_ground.sh
```

Same `sky_mask.png` file, but with `--mask-invert`. Now the mask
treats black as inpaint, so the *ground* gets repainted (snow) and
the sky is preserved. One mask, two opposite edits — no need to
hand-create a second file.

### 4. `04_strength_sweep.sh` — the strength dial

```bash
./scripts/04_strength_sweep.sh
```

Three runs of img2img with the same prompt, same seed, varying
only `--strength` (0.3, 0.5, 0.7). Files land as
`strength-0_3.png`, `strength-0_5.png`, `strength-0_7.png` for
side-by-side comparison.

| Strength | Effect |
|---|---|
| 0.0–0.2 | Almost no change; the source dominates. |
| 0.3–0.5 | The prompt nudges colour palette + texture; composition preserved. |
| 0.5–0.7 | The model has real creative room; expect texture rewrite and minor composition drift. |
| 0.7–0.9 | The prompt drives most of the output; the source is a loose layout hint. |
| 1.0 | Equivalent to a fresh t2i generation; the source contributes nothing. |

## Modifications to try

**Different mask shapes.** Edit `inputs/sky_mask.png` in any image
editor (or generate your own grayscale mask of the same dimensions)
and re-run script 02. White is inpaint, black is preserve.

**Soften the boundary.** The default `--mask-feather 8` blurs the
mask edge by 8 px before applying. Drop to `--mask-feather 0` for a
hard line, or push to `--mask-feather 32` for very gradual blending.

**Bring your own image.** Replace the `inputs/landscape.png` reference
in any script with your own image path. The size will be detected
automatically and snapped to a multiple of 8 (the VAE constraint);
override with `--size WxH` if needed.

**Use SDXL.** Add `--model sdxl --size 1024x1024` to any script.
SDXL is slower but gives more detail.

## See also

- [`Documentation/IMG2IMG.md`](../../../Documentation/IMG2IMG.md) —
  full reference: every flag, the mask conventions, edge cases.
- [`Documentation/GENERATE.md`](../../../Documentation/GENERATE.md) —
  for the broader `plakat generate` surface, which shares most
  flags with `img2img`.
- [`Documentation/APPLE_REQUIREMENTS.md`](../../../Documentation/APPLE_REQUIREMENTS.md) —
  expected speeds + memory tiers on Apple Silicon.

## License

Everything in this directory is CC0 / Unlicense — same as plakat
itself. The sample input and mask are procedurally generated by
plakat and carry no third-party copyright.
