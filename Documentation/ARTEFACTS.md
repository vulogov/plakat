# Artefact compositing

Plakat can composite named **artefact** PNG cutouts (trees, houses,
sky elements, etc.) into named **zones** of a generated image. The
compositing happens after generation but before any optional stylize
or upscale pass, so the IP-Adapter stylize re-paints over the
artefacts and unifies the palette — pasted-on cutouts blend into the
final image more naturally than a strict alpha overlay would.

This is **alpha-compositing**, not generative inpainting. The artefact
PNG is alpha-blended onto the generated image at the specified zone +
scale + anchor. Two optional passes refine the result:

- `--artefact-blend` (v2) — masked img2img to smooth the pasted edges
  (see [Blend pass (v2)](#blend-pass-v2)).
- `--smart-zones` (v3) — derive zones from the generated image's own
  depth + luminance instead of the rigid 4×3 grid (see
  [Smart zones (v3)](#smart-zones-v3)).

## Quick start

```bash
plakat generate "a green meadow under a blue sky" \
    --artefact oak@middle_plan/left \
    --artefact sun@sky/right
```

Plakat:

1. Generates the meadow image as usual.
2. Looks up `oak` and `sun` in the bundled artefact library.
3. Resizes each PNG to fit its target zone (preserving aspect ratio).
4. Alpha-composites both onto the generated PNG.
5. Saves the final image, replacing the original.

Repeat `--artefact` for multiple cutouts. Their order = z-order (later
flags render on top of earlier ones).

## The artefact library

The library is a directory containing:

```
assets/artefact_library/
├── library.json          # metadata for every artefact
├── trees/
│   ├── oak.png
│   └── pine.png
├── sky/
│   ├── sun.png
│   ├── moon.png
│   └── cloud.png
└── houses/
    └── cottage.png
```

`library.json` schema:

```json
{
  "schema_version": 1,
  "artefacts": [
    {
      "name": "oak",
      "category": "tree",
      "path": "trees/oak.png",
      "natural_zone": "middle_plan",
      "natural_size_pct": 0.95,
      "anchor": "bottom_center",
      "license": "CC0",
      "license_url": null,
      "tags": ["nature", "vegetation"]
    }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique slug. What `--artefact <name>` references. |
| `category` | no | Grouping for `plakat artefact list --category`. Default: `uncategorized`. |
| `path` | yes | PNG file, relative to the library directory. Alpha optional — auto-chroma-key fallback runs if no alpha present. |
| `natural_zone` | yes | Default placement. See "Zone references" below. |
| `natural_size_pct` | no | Fraction of the **zone's height** the artefact occupies at default scale. Default: `0.7`. |
| `anchor` | no | The point on the artefact that aligns to its placement in the zone. Either a named position (`bottom_center` etc.) or a `{x, y}` fractional object. Default: `center`. |
| `license` | no | Free-form text for `plakat artefact show` display. |
| `license_url` | no | URL pointing at the license terms. |
| `tags` | no | Free-form list of strings. |

### Bundled library

Plakat ships with a minimal procedurally-drawn placeholder set at
`assets/artefact_library/` containing 6 CC0 silhouettes: `sun`,
`moon`, `cloud`, `oak`, `pine`, `cottage`. The drawings are
deliberately simple — they prove the pipeline works and provide
something for tutorials and tests. For production use, replace these
with your own curated PNGs and update `library.json` accordingly. Run
`examples/draw_default_artefacts.rs` to re-generate the bundled set
from scratch.

## Zone references

For an output image of width W × height H, the default 4×3 grid is:

| Depth band | Vertical extent |
|---|---|
| `sky` | top quarter (0 → H/4) |
| `far_plan` | second quarter (H/4 → H/2) |
| `middle_plan` | third quarter (H/2 → 3H/4) |
| `close_plan` | bottom quarter (3H/4 → H) |

| Horizontal band | Horizontal extent |
|---|---|
| `left` | 0 → W/3 |
| `center` | W/3 → 2W/3 |
| `right` | 2W/3 → W |

Zone references can name:

- A depth band alone: `sky`, `far_plan`, `middle_plan`, `close_plan` —
  full image width × that depth slice.
- A horizontal band alone: `left`, `center`, `right` — full image
  height × that horizontal slice.
- An intersection: `sky/right`, `middle_plan/left`,
  `close_plan/center`. Order doesn't matter (`right/sky` == `sky/right`).

### Overriding the grid

For a 16:9 panoramic composition you might want sky to occupy 40% of
the canvas instead of 25%. Override per scenario:

```hjson
zones:
{
    sky:         [0.0, 0.40]
    far_plan:    [0.40, 0.55]
    middle_plan: [0.55, 0.80]
    close_plan:  [0.80, 1.0]
}
```

Coordinates are normalized to `[0, 1]`. Missing bands fall back to the
default grid. Horizontal bands accept the same format under `left:`,
`center:`, `right:`.

## CLI: `plakat generate` and `plakat portrait`

```bash
--artefact NAME[@ZONE[:SCALE]]   # repeatable
--artefact-library <DIR>         # override bundled library
--artefact-blend                 # v2: masked img2img blend after composite
--artefact-blend-strength F      # default 0.3 (range 0..1)
--smart-zones                    # v3: derive zones from depth + luminance
```

Shorthand grammar:

| Form | Meaning |
|---|---|
| `--artefact oak` | Use library's natural zone, default scale. |
| `--artefact oak@middle_plan/left` | Override zone. |
| `--artefact sun@sky/right:0.6` | Override zone + scale (60% of default). |

`SCALE` is a positive float multiplier applied to the library's
`natural_size_pct`. Above ~1.5 the artefact pushes past its zone and
gets clamped to canvas bounds.

The full set of per-placement overrides (offset, anchor, flip, alpha)
is only available via the scenario HJSON form below. CLI shorthand
covers the common cases concisely; HJSON covers everything.

## Scenarios

A task can attach artefacts via the `artefacts:` field:

```hjson
{
    # Top-level: optionally override the library path + zone grid.
    artefact-library: ./my_artefacts/
    zones:
    {
        sky: [0.0, 0.35]
    }
    # v2: enable masked img2img blend across all tasks. Per-task
    # `artefact-blend: true/false` overrides this.
    artefact-blend: true
    artefact-blend-strength: 0.3
    # v3: derive zones from each image's own depth + luminance.
    # Per-task `smart-zones: true/false` overrides.
    smart-zones: true

    # ... model, scenes, weather, tasks ...

    tasks:
    [
        {
            name: village_scene
            scene: meadow
            weather: dawn
            prompt: "a quiet village in a valley"
            artefacts:
            [
                # Shorthand strings (CLI grammar).
                "oak@middle_plan/left"
                "oak@middle_plan/right"
                "sun@sky/right"

                # Full-object form with per-artefact overrides.
                {
                    name: cottage
                    zone: close_plan/center
                    scale: 0.6
                    offset: [0.0, 0.05]
                    anchor: bottom_center
                    flip: false
                    alpha: 0.95
                }
            ]
        }
    ]
}
```

### Full per-artefact object fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Library entry name. |
| `zone` | string | natural | Override the zone. |
| `scale` | float | 1.0 | Multiplier on `natural_size_pct`. |
| `offset` | `[dx, dy]` | auto-stagger | Fractional shift within zone (units = zone width/height). |
| `anchor` | string or `{x, y}` | library default | Override the natural anchor. |
| `flip` | bool | false | Horizontal flip. |
| `alpha` | float | 1.0 | Global alpha multiplier in `[0, 1]`. |

### Auto-stagger

If two or more artefacts target the same zone *and* none supplies an
explicit `offset`, plakat auto-spaces them horizontally so they don't
stack on top of each other. The spread tightens as you add more
artefacts to the same zone.

Supplying an explicit `offset` opts that artefact out of auto-stagger.

## Anchors

The anchor is the point on the artefact that aligns to the zone's
placement point. Named anchors cover the 9-point grid:

```
   top_left ─── top_center ─── top_right
       │            │             │
center_left ──── center ────── center_right
       │            │             │
bottom_left ─ bottom_center ─ bottom_right
```

Grounded objects (trees, buildings) typically use `bottom_center` so
their bases anchor to the bottom of the zone. Celestial objects (sun,
moon) typically use `center` so they float in the middle of the sky
band.

For fine control, use fractional anchors instead of named:

```hjson
artefacts:
[
    {
        name: sun
        anchor: { x: 0.5, y: 0.3 }   # artefact center-x, 30% from top
    }
]
```

## Auto-chroma-key fallback

Artefact PNGs ideally ship with a real alpha channel — every pixel
has its own opacity. When you don't have a tool that produces alpha
(or you've sourced JPEGs / non-alpha PNGs from somewhere), plakat
falls back to the same chroma-key logic used by `plakat transparent`:
the upper-left pixel's color becomes the "background color" and
matching pixels are dropped to alpha=0.

The fallback uses tolerance 10 (per-channel diff), which absorbs JPEG
noise and modest anti-aliased edges without eating into the artefact
itself. For best results:

- Make sure the upper-left pixel of your PNG is the background color
  you want to remove.
- Use a solid-colored background (one uniform RGB across the canvas
  minus the artefact silhouette).
- Avoid backgrounds whose colors appear in the artefact itself (e.g.
  white background + white shirt → the shirt becomes transparent).

When you control the input, exporting with a real alpha channel from
your editor sidesteps all this.

## Order of operations

The per-task pipeline runs:

1. **Generate** the base image (SD / SDXL / Flux).
2. **Composite artefacts** onto the generated image (this step).
3. **Blend** (optional, `--artefact-blend`) — masked low-strength
   img2img re-pass to smooth the pasted edges. See
   [Blend pass (v2)](#blend-pass-v2).
4. **Stylize** (per-task `style: <ref-photo>` field, IP-Adapter
   img2img).
5. **Upscale** (per-scenario `upscale: { upscale: true }`).

The composite is intentionally placed before stylize. When stylize is
enabled, the IP-Adapter pass re-paints over the composited artefacts
with the reference image's palette — unifying the cutouts visually
with the generated scene. Without stylize, artefacts retain their own
palette and look more "collaged."

## Blend pass (v2)

The v1 alpha composite leaves pasted artefacts looking pasted: hard
silhouette edges, no shadow interaction with the surroundings, and
any palette / lighting mismatch is left exactly as-is.

v2's `--artefact-blend` flag adds a short masked img2img pass on top
of the alpha composite. It re-noises a feathered mask covering every
artefact's *zone* (not just the artefact silhouette) and runs ~30 %
of a normal denoise — enough for the model to integrate the artefact
into the surrounding context without redrawing its shape.

```bash
plakat generate "a green meadow under a blue sky" \
    --artefact oak@middle_plan/left \
    --artefact sun@sky/right \
    --artefact-blend \
    --artefact-blend-strength 0.3
```

Default: off (preserves v1 behaviour byte-for-byte).

### Strength dial

`--artefact-blend-strength` is the standard img2img strength
parameter, applied only inside the masked region.

| Strength | Effect |
|---|---|
| 0.0 | No-op (mask is all zero in practice). |
| 0.15–0.25 | Soft touch — edge feathering only. Shape preserved. |
| **0.30 (default)** | Sweet spot for most scenes. Edges integrate cleanly without redrawing the artefact silhouette. |
| 0.40–0.55 | Heavier blend. The model may add detail (texture, shadow) but starts to drift on the artefact shape. |
| 0.60+ | The model treats the artefact as a hint and may redraw it as something else entirely. |

For palette-matched curated artefacts, leave at default or lower. For
mismatched silhouettes you want the model to "fix into shape," push
to 0.4–0.5.

### Cost

The blend pass is one extra denoise of `--steps` iterations (default
28) at the generation resolution. On an Apple M2 Max generating at
768 × 768, it adds **~2–4 s per image**; on RTX 4090 at 1024 × 1024
it adds **~3–5 s**. The mask is built once per generation and reused
across all images in a `--count` batch, so per-image cost is
dominated by the denoise itself.

The blend pass also re-loads the SD pipeline (text-encoder, UNet,
VAE) once. For scenarios with many `--count` images per task this is
amortized; for tiny scenarios the load can dominate. Future work may
re-use the in-memory pipeline from the generation step.

### When to skip

- **You already have a stylize pass.** Stylize re-paints the whole
  canvas through IP-Adapter; that unifies the palette far more
  aggressively than a 30 % blend can. Stacking blend + stylize is
  rarely worth the extra time.
- **You don't have a GPU and `--steps` is high.** A 28-step blend on
  CPU is minutes per image. Either drop `--steps`, drop the flag, or
  use `--artefact-blend-strength 0.15` which still pays ~30 % of
  full denoise cost (start_idx skips early timesteps).
- **You're using Flux.** Blend routes through plakat's SD pipeline.
  Flux generations cannot blend — the flag errors out at load time.

### How the mask is built

1. Start from a blank canvas at generation resolution.
2. Union of every resolved artefact's *zone rect* set to `1.0`.
3. Separable box blur, radius 16 px (~2 % of a 1024 px canvas).
4. Average-pool 8× into latent space.
5. Cast to the pipeline's dtype.

Using the zone (broader) instead of the artefact target rect
provides a natural feathering margin — the denoiser blends the
artefact's surroundings, not just its silhouette. Feathering softens
the mask edge so the transition from "regenerate" to "preserve" is
gradual.

## `plakat artefact` subcommand

Inspect the library without generating:

```bash
plakat artefact list                       # all entries
plakat artefact list --category tree       # filter by category prefix
plakat artefact show oak                   # full info for one entry
plakat artefact show oak --format json     # JSON for scripting
```

Both commands accept `--library <DIR>` to override the bundled set.

## Limits

- **Visible collage by default.** Pasted PNGs look pasted unless the
  artefact aesthetic matches the diffusion model's output. Mitigate
  with `--artefact-blend` (v2 masked img2img, see above), or chain a
  stylize pass that re-paints the whole canvas through IP-Adapter
  (unifying the palette aggressively).
- **Rigid grid is approximate** (by default). A "sky" zone is the top
  25 % of the canvas regardless of what the diffusion model actually
  painted there. Enable `--smart-zones` (v3) to derive zones from the
  image's depth + luminance instead, or override `zones:` per
  scenario when the layout is predictable.
- **No artefact generation.** Plakat composites, doesn't paint
  artefacts. Users prepare PNGs themselves (Photoshop, GIMP,
  rembg/AI cutout tools).
- **Z-order is list order.** If two artefacts overlap, the later one
  in the `--artefact` flags / `artefacts:` list covers the earlier
  one.
- **Lighting/color mismatch.** A daylight sun on a sunset scene looks
  wrong. The base composite has no automatic palette matching;
  `--artefact-blend` helps but doesn't fully fix it. The practical
  workaround is curating artefacts that match your scene aesthetic,
  enabling blend, or adding a stylize pass.
- **Blend doesn't support Flux.** v2's masked img2img routes through
  plakat's SD 1.5 / SDXL pipeline. Using `--artefact-blend` with
  `--model flux-*` errors at the blend pipeline's load step.

## Smart zones (v3)

The v1 grid (`sky` = top 25 %, `close_plan` = bottom 25 %, etc.)
assumes the diffusion model painted the scene exactly as the grid
expects. It doesn't. A meadow with a low horizon has most of its
canvas as ground; calling the top quarter "sky" misses the actual sky.

`--smart-zones` derives zones from the generated image itself. Two
cheap signals:

- **Depth** (Depth-Anything-V2 small, ~99 MB) — per-row mean depth is
  bucketed by quantile (q25 / q50 / q75). The lowest-depth rows
  become `sky`, highest become `close_plan`. The actual painted
  horizon lands at the `sky`↔`far_plan` boundary regardless of where
  the rigid grid would have put it.
- **Luminance** — per-column vertical variance ("how busy is this
  column?") gives a variance-weighted centroid. The `center` band
  shifts so it's actually over the busy part of the image; `left`
  and `right` fill the remainder.

```bash
plakat generate "a low-horizon meadow at sunset" \
    --artefact sun@sky/right \
    --artefact oak@middle_plan/left \
    --smart-zones
```

The `sun@sky/right` placement now lands wherever the model actually
painted sky (top 60 % if the horizon is low), not whichever pixels
happen to be in the top quarter.

### Cost

- One-time model download (~99 MB) on first use; cached afterwards.
- Per-image: ~0.5–1.5 s on Apple Silicon / RTX 4090, ~3–10 s on CPU.
- The depth model loads once per `plakat generate` invocation /
  scenario run and is reused for every image.

### Fallback

If the depth model can't be downloaded or fails to load (network out,
HuggingFace mirror unreachable), plakat logs a warning and falls back
to the rigid grid. The flag is non-fatal — a generation with
`--smart-zones` never errors out just because the depth signal is
unavailable.

If `--smart-zones` produces a degenerate signal on a particular image
(flat depth field, monochrome canvas), only the affected bands fall
back to the grid. Other bands still use the smart values. User-
supplied `zones:` overrides fill the gap for fallback bands.

### When `--smart-zones` helps

- **Wide aspect ratios.** A 16:9 panorama with a 30 %-tall sky strip
  shouldn't have artefacts pinned to "top 25 %". Smart zones tracks
  the actual horizon.
- **Compositional asymmetry.** If the generated scene has its
  centre of interest off-centre (subject on the right, sky on the
  left), the centroid-centred horizontal split places artefacts more
  naturally.
- **Variable scene geometry across `--count`.** Each image in a batch
  gets its own zone resolution. Different framings get different
  placements automatically.

### When to skip

- **Identical, predictable framings.** If you're batching 50 images
  of the same composition and they all hit the rigid grid acceptably,
  the depth pass is wasted overhead.
- **CPU-only runs at high `--count`.** ~3–10 s per image adds up.
  Disable smart zones for batches.
- **Tightly-cropped subjects.** A portrait headshot has no
  meaningful "sky" or "ground" — the depth quantiles end up mapping
  to face features, which isn't what you want.

### Interaction with other flags

- **`--artefact-blend`** — fully compatible. The blend mask is
  rebuilt per-image using smart zones, so the blended region tracks
  the smart-resolved rect rather than the rigid grid.
- **`zones:` overrides in scenarios** — kept as fallback for any band
  the smart signal can't resolve. Smart values win where present.
- **`--style-ref` / `--style`** — orthogonal. Style pass runs after
  compositing as before.

## See also

- [Tutorial walkthrough](Tutorials/ARTEFACTS_TUTORIAL.md) — start
  here if you've never used the feature.
- [Runnable hands-on tutorial](../examples/tutorials/ZONES/) — seven
  shell scripts + an HJSON scenario demonstrating v1 + v2 + v3
  end-to-end. The canonical "show me, don't tell me" reference.
- [PORTRAIT_TUTORIAL.md](Tutorials/PORTRAIT_TUTORIAL.md) — for
  identity-preserving portraits, which compose orthogonally with
  artefact compositing.
- [STYLES.md](STYLES.md) — for art-style transfer, which can run
  after compositing to unify palettes.
- [APPLE_REQUIREMENTS.md](APPLE_REQUIREMENTS.md) — for Apple chip +
  memory implications of stacking `--artefact-blend` and
  `--smart-zones` on top of generation.
