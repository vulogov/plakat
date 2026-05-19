# Composing scenes with artefact cutouts

This tutorial walks you through plakat's artefact-compositing feature:
placing pre-made PNG cutouts (trees, sky elements, houses, etc.) into
named zones of a generated image. By the end you'll be able to compose
scenes with specific elements where you want them, instead of hoping
the diffusion model paints them in the right places.

## What you'll learn

- What artefacts are and when this technique is useful
- How to inspect the bundled artefact library
- How to place a single artefact, and how zones work
- How to compose multi-artefact scenes
- How to override placement (scale, offset, anchor) for finer control
- How to use artefacts in scenarios
- The honest limits — what alpha compositing can and can't do

## Before you start

- Work through `GENERATE_TUTORIAL.md` first.
- The bundled artefact library at `assets/artefact_library/` ships
  with 6 placeholder silhouettes (sun, moon, cloud, oak, pine,
  cottage). Replace these with your own PNGs for production use.
- For best results, also work through `STYLES_TUTORIAL.md` —
  combining artefacts with a style pass unifies their palette with
  the generated scene.

---

## 1. What artefacts are (and aren't)

An **artefact** is a PNG cutout — a graphic with transparent
background — that you composite onto a generated image at a known
position. The artefact's pixels are alpha-blended onto the generated
scene; the transparent pixels let the scene show through.

This is fundamentally different from prompting the model to draw a
specific object:

- Prompting: "a meadow with three oak trees and a cottage" → the
  model paints whatever it thinks "three oak trees" look like, at
  positions you can't control precisely.
- Artefact compositing: generate "a meadow" then composite an oak
  PNG at `middle_plan/left` and a cottage PNG at `close_plan/center`
  → trees and cottage are exactly where you want them, but they're
  collaged on rather than painted in.

**When to use artefacts:**

- You need **specific objects in specific places** across many
  generations (e.g., the same logo silhouette in every image in a
  series).
- The diffusion model struggles to draw something legibly
  (text, branded items, specific architectural features).
- You're building **consistent storytelling** across batch scenarios
  — the same artefacts placed identically across many scenes.

**When NOT to use artefacts:**

- One-off "make me a beautiful image." Iterating prompts usually
  produces a more cohesive result than composited cutouts.
- You expect photorealistic integration. Without a stylize pass to
  re-paint everything, alpha-composited artefacts visibly look
  collaged.

---

## 2. Inspecting the bundled library

```bash
plakat artefact list
```

You should see six entries:

```
Name     Category   Natural zone    Size%   Anchor
───────  ─────────  ──────────────  ──────  ────────────────────
sun      celestial  sky             0.70    (0.50,0.50)
moon     celestial  sky             0.70    (0.50,0.50)
cloud    weather    sky             0.80    (0.50,0.50)
oak      tree       middle_plan     0.95    (0.50,1.00)
pine     tree       middle_plan     0.95    (0.50,1.00)
cottage  building   close_plan      0.80    (0.50,1.00)

6 artefact(s) total.
```

The columns:

- **Name**: how you reference it on the command line.
- **Category**: grouping for filtering.
- **Natural zone**: where it lands when you don't specify a zone.
- **Size%**: fraction of zone height the artefact occupies at default
  scale (0.95 = "takes up 95% of zone vertically").
- **Anchor**: the point on the artefact that aligns to its placement
  in the zone. `(0.50, 1.00)` = "bottom-center" (50% from left, 100%
  from top — the bottom-center pixel).

Look at one in detail:

```bash
plakat artefact show oak
```

```
Name:            oak
Category:        tree
Path:            assets/artefact_library/trees/oak.png
Natural zone:    middle_plan
Natural size: 0.95 (fraction of zone height)
Anchor:          (x=0.50, y=1.00) — fraction of artefact's own size
License:         CC0
Tags:            nature, vegetation
```

---

## 3. Your first composite

The simplest invocation — place an oak in its natural zone:

```bash
plakat generate "a green meadow under a clear sky" \
    --artefact oak \
    --seed 42
```

What happens:

1. Plakat generates "a green meadow under a clear sky" as usual.
2. After generation, the oak PNG is loaded.
3. Oak's natural zone is `middle_plan` (the third quarter of the
   canvas vertically) — that resolves to the middle band of your
   image.
4. Oak's natural size is 95% of zone height; anchor is bottom-center
   (the tree's base lines up with the bottom of the middle plan).
5. The oak is alpha-composited onto the meadow.
6. Output: a meadow with an oak silhouette growing from the middle
   band.

Note: passing `--seed 42` keeps the meadow composition fixed across
runs while you experiment with artefact placement.

---

## 4. Zones — where artefacts go

The canvas is partitioned into a 4×3 grid:

```
              left      center      right
           ┌─────────┬─────────┬─────────┐
   sky     │         │         │         │  (top quarter)
           ├─────────┼─────────┼─────────┤
   far_plan│         │         │         │  (second quarter)
           ├─────────┼─────────┼─────────┤
   middle  │         │         │         │  (third quarter)
   _plan   ├─────────┼─────────┼─────────┤
   close   │         │         │         │  (bottom quarter)
   _plan   └─────────┴─────────┴─────────┘
```

Reference a zone with `<depth>` (full-width band), `<horizontal>`
(full-height column), or `<depth>/<horizontal>` (the intersection):

```bash
# Sun in the top-right quarter of the sky:
plakat generate "..." --artefact sun@sky/right --seed 42

# Oak in the left third of the middle plan:
plakat generate "..." --artefact oak@middle_plan/left --seed 42

# Cottage at the bottom-center (close plan, center):
plakat generate "..." --artefact cottage@close_plan/center --seed 42
```

---

## 5. Composing a multi-artefact scene

Repeat `--artefact` to add multiple. The flag order is the z-order:
later flags render on top of earlier ones.

```bash
plakat generate "a quiet meadow at sunrise, mountains in the distance" \
    --artefact sun@sky/right \
    --artefact pine@middle_plan/left \
    --artefact oak@middle_plan/right \
    --artefact cottage@close_plan/center \
    --seed 42
```

Result: a meadow with:

- A sun in the upper-right corner
- A pine tree in the left-middle band
- An oak in the right-middle band
- A cottage centered at the bottom

The compositing log line tells you what happened:

```
  ◆ compositing 4 artefact(s) onto 1 image(s)
```

---

## 6. Adjusting scale

Append `:<float>` to override the default size:

```bash
plakat generate "..." \
    --artefact oak@middle_plan/left:0.6 \
    --artefact sun@sky/right:1.4
```

Scales:

- `:0.6` → 60% of the natural size (smaller, less prominent).
- `:1.0` → natural size (the default).
- `:1.4` → 140% (larger, more prominent).

Above ~1.5 the artefact pushes past its zone bounds and gets clamped
to the canvas edges. Below ~0.3 it becomes too small to notice
against the generated content.

---

## 7. Combining with a style pass

Alpha-composited artefacts can look "collaged" against a
photorealistic generated background. v2's `--artefact-blend` and
v3's `--smart-zones` (sections 11 and 12 below) each help in
different ways. The most aggressive fix is still a stylize pass
that re-paints the whole canvas through IP-Adapter, unifying
everything in one stylistic voice — and it composes cleanly with
the v2/v3 features.

In a scenario you'd write:

```hjson
tasks:
[
    {
        name: composited_meadow
        scene: meadow
        weather: dawn
        prompt: "a quiet meadow with traditional cottages"

        # The artefacts get composited first…
        artefacts:
        [
            "pine@middle_plan/left"
            "cottage@close_plan/center"
        ]

        # …then this stylize ref re-paints the whole image in its
        # palette, unifying the cutouts with the meadow.
        style: ./refs/watercolor_meadow.jpg
        style-strength: 0.6
    }
]
```

The order is fixed by the pipeline: artefact compositing happens
BEFORE the stylize pass, so the stylize re-paints over the cutouts.
This is what makes "collage-y" artefacts blend into the final image.

(Without a stylize ref, the artefacts retain their own palette — fine
for some looks, distracting for others.)

---

## 8. Scenarios — full HJSON surface

The scenario form unlocks per-artefact overrides that the CLI
shorthand doesn't expose:

```hjson
tasks:
[
    {
        name: detailed_scene
        scene: forest
        weather: dawn
        prompt: "a forest scene at dawn"
        artefacts:
        [
            # Shorthand entry (same grammar as CLI):
            "oak@middle_plan/right"

            # Full-object entry with every override available:
            {
                name: oak
                zone: middle_plan/left
                scale: 0.8                  # smaller than default
                offset: [0.1, 0.0]          # shift right by 10% of zone width
                anchor: bottom_center       # explicit
                flip: true                  # horizontal flip — variety!
                alpha: 0.85                 # slightly translucent
            }

            {
                name: sun
                zone: sky/right
                anchor: { x: 0.5, y: 0.3 }  # fractional anchor — sun's "center" is high
            }
        ]
    }
]
```

Field reference:

| Field | Type | Default | What it does |
|---|---|---|---|
| `name` | string | required | Library entry name. |
| `zone` | string | natural | Where the artefact lands. |
| `scale` | float | `1.0` | Multiplier on natural size. |
| `offset` | `[dx, dy]` | auto-stagger | Fractional shift inside zone. |
| `anchor` | string or `{x, y}` | library default | Override placement anchor. |
| `flip` | bool | `false` | Horizontal flip. |
| `alpha` | float in `[0, 1]` | `1.0` | Global opacity multiplier. |

---

## 9. Auto-stagger

When two or more artefacts share a zone *and* none has an explicit
offset, plakat auto-stagers them horizontally so they don't pile on
top of each other:

```bash
# Two oaks in the same zone — auto-spaced left and right.
plakat generate "a wide meadow" \
    --artefact oak@middle_plan \
    --artefact oak@middle_plan \
    --seed 42
```

The first one drifts to the left half of the zone, the second to the
right. Add a third and they redistribute evenly.

Setting an explicit `offset:` opts that artefact out of auto-stagger
— it lands exactly where you put it.

---

## 10. Bringing your own artefacts

The bundled set is placeholder-quality. For real work, source your
own PNGs:

1. Find or create cutout-style images. Trees, buildings, animals,
   logos, characters — anything with a transparent background.
2. Save as PNG with a real alpha channel (export with transparency
   from Photoshop / GIMP / Procreate / rembg).
3. Add them to the library directory:

```
my_artefacts/
├── library.json
├── trees/
│   └── willow.png
└── characters/
    └── walking_figure.png
```

4. Edit `library.json` to declare each new artefact:

```json
{
  "schema_version": 1,
  "artefacts": [
    {
      "name": "willow",
      "category": "tree",
      "path": "trees/willow.png",
      "natural_zone": "middle_plan",
      "natural_size_pct": 0.85,
      "anchor": "bottom_center",
      "license": "CC0",
      "tags": ["nature", "vegetation"]
    }
  ]
}
```

5. Run plakat against your library:

```bash
plakat generate "..." \
    --artefact willow@middle_plan/left \
    --artefact-library ./my_artefacts/
```

For scenarios, set it once at scenario root:

```hjson
{
    artefact-library: ./my_artefacts/
    # ... rest of scenario ...
}
```

### What if my PNG doesn't have a real alpha channel?

Plakat auto-falls-back to chroma-keying the upper-left pixel
(reusing the same logic as `plakat transparent`). It's not perfect —
anti-aliased edges can have artifacts — but it lets you use solid-
background images without preprocessing.

For best results:

- Use a solid-color background (uniform RGB across non-artefact pixels).
- Use a background color that doesn't appear in the artefact itself
  (e.g., bright magenta against a green tree → cleaner key than white
  against a white snowy tree).
- Export with real alpha when you can — it sidesteps all of this.

---

## 11. Smoothing the seams with `--artefact-blend`

The "collage" look from an alpha composite can be partly tamed without
going all the way to a stylize re-paint. The `--artefact-blend` flag
runs a short masked img2img pass over the artefact zones after the
composite — feathering the edges and integrating the artefact into
the surrounding context.

```bash
plakat generate "a green meadow under a blue sky at golden hour" \
    --artefact oak@middle_plan/left \
    --artefact sun@sky/right \
    --artefact-blend
```

The blend pass uses ~30 % img2img strength by default — strong enough
to soften the silhouette and absorb a modest lighting mismatch,
weak enough to leave the artefact recognisably itself.

Tune the strength when the default isn't right:

```bash
# Lighter — edge feathering only, shape strictly preserved.
plakat generate "a sunset meadow" \
    --artefact cottage@close_plan/center \
    --artefact-blend --artefact-blend-strength 0.20

# Heavier — let the model add texture / shadow detail. The artefact
# shape may drift a little.
plakat generate "a moody nordic forest" \
    --artefact pine@middle_plan/left \
    --artefact pine@middle_plan/right \
    --artefact-blend --artefact-blend-strength 0.45
```

**Cost.** One extra denoise pass per image. ~2–4 s on Apple Silicon /
~3–5 s on RTX 4090. The SD model loads once and gets re-used across
every `--count` image.

**When to stack with stylize.** Mostly: don't. Stylize re-paints the
whole canvas through IP-Adapter, which unifies the palette far more
aggressively than blend can. Use blend when you don't have (or don't
want) a style reference. Use stylize when you have one. Stacking both
is rarely worth the extra time.

**Flux.** Blend doesn't support Flux. If you're using `--model flux-*`
the flag errors at load time — drop it.

For the full strength dial, mask construction details, and when to
skip blend entirely, see
[`ARTEFACTS.md` § Blend pass (v2)](../ARTEFACTS.md#blend-pass-v2).

---

## 12. Following the painted horizon with `--smart-zones`

The rigid grid says "sky = top 25 %". But the diffusion model doesn't
read your grid. A 16:9 panoramic generation often paints sky into
the top 40–50 %; a low-horizon meadow generation puts ground into
the top half. The grid misplaces artefacts when the painted scene
disagrees with it.

`--smart-zones` reads the actual image: depth tells it where sky and
foreground actually sit, and luminance tells it where the "centre of
content" is horizontally. The zone references stay the same, but
their pixel extents track what the model painted.

```bash
plakat generate "a low-horizon meadow at sunset, golden light" \
    --aspect 16:9 \
    --artefact sun@sky/right \
    --artefact oak@middle_plan/left \
    --smart-zones
```

Without `--smart-zones`, `sun@sky/right` lands in the top 25 %
of the canvas — which, on a low-horizon sunset, might be mid-air
above the actual sky line. With it, sun lands in whatever rows the
depth model identified as actual sky.

**First-run cost.** The depth model is Depth-Anything-V2-small
(~99 MB). Downloaded once and cached. Inference is ~0.5–1.5 s per
image on a GPU. The model loads once per `plakat generate` /
scenario run and is shared across every image in the batch.

**Fallback.** If the model can't download (network out, mirror
unreachable), plakat falls back to the rigid grid with a warning.
The flag never blocks a generation.

**Stacking with `--artefact-blend`.** Fully compatible — recommended,
even. The blend mask is rebuilt per image using the smart-resolved
zone, so blending tracks the painted horizon too:

```bash
plakat generate "moody nordic fjord at dusk" \
    --aspect 16:9 \
    --artefact pine@middle_plan/left \
    --artefact pine@middle_plan/right \
    --artefact-blend --smart-zones
```

**When to skip smart zones.**

- *Tight portraits.* No meaningful sky / ground — depth quantiles
  end up mapping to face features. Use the rigid grid + manual
  `zones:` overrides.
- *Big CPU-only batches.* The depth pass adds seconds per image.
- *Identical framings.* If every image hits the rigid grid OK, the
  extra inference is wasted.

For the full strength dial, fallback details, and mask construction,
see [`ARTEFACTS.md` § Smart zones (v3)](../ARTEFACTS.md#smart-zones-v3).

---

## 13. Limits and honest tradeoffs

**It can still look collaged.** `--artefact-blend` softens edges but
doesn't fully match palettes. The most aggressive fix is still a
stylize pass that re-paints the whole canvas.

**Zone grid is rigid by default.** "Sky" is the top quarter of the
canvas unless you turn it off. Enable `--smart-zones` to derive zones
from the actual painted scene's depth, or override `zones:` per
scenario when the layout is predictable.

**No artefact generation.** Plakat composites cutouts; it doesn't
draw them. You provide the PNGs (or replace the bundled placeholders).

**Lighting and palette mismatch.** A daylight cutout dropped onto a
sunset scene looks wrong. There's no auto-palette matching.
`--artefact-blend` helps but doesn't fully fix it. The most reliable
workaround: curate stylistically uniform artefacts + chain a stylize
pass for palette unification.

**Z-order = list order.** The last artefact in `--artefact` /
`artefacts:` covers earlier ones at overlap. Reorder the list to
change z-order.

**Performance is negligible.** Compositing 6 artefacts on a 1024×1024
canvas adds well under 100 ms to a generation that already takes
seconds. Don't worry about cost.

---

## Where to next

- **Runnable companion** — seven self-contained shell scripts +
  an HJSON scenario that demonstrate every tier (v1 alpha
  composite, v2 blend, v3 smart zones) end-to-end:
  [`examples/tutorials/ZONES/`](../../examples/tutorials/ZONES/).
- **Full reference** — every field, every override, every limit:
  [`Documentation/ARTEFACTS.md`](../ARTEFACTS.md)
- **Combining with style transfer** to unify the palette:
  [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md)
- **Portraits + artefacts** — `plakat portrait` also accepts
  `--artefact`, so you can drop a sun or cottage into a portrait
  scene. Same grammar.
- **Scenarios** — for batches where many tasks need consistent
  artefact placement, see the HJSON form in section 8 above and
  [`Documentation/GENERATE.md`](../GENERATE.md) for the full scenario
  schema.
