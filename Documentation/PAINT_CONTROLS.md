# plakat paint — tuning controls per technique

`plakat paint` renders a painting in **stroke space**: the model makes a low‑res **armature**, and a
weight‑free **stroke engine** paints the full‑res picture from it in a chosen **medium**. Every knob below is
authored in the **HJSON spec** (the control surface) and can be overridden on the **CLI**. The **medium sets the
default behaviour** for each technique; the spec/CLI only override what you want to change.

Precedence: **CLI flag → HJSON field → medium default**.

```
plakat paint painting.paint.hjson --out out.png     # spec-driven (armature from `subject:`, or `reference:`)
plakat paint from photo.png --style fidelity ...     # paint an existing image (CPU, no GPU)
plakat paint new painting.paint.hjson                # scaffold a spec with all controls documented
```

## Painting your own image or photo (no generation)

plakat paint is **not a filter** — it re-paints a supplied image in stroke space (invents the surface, applies
the medium's real behaviour) and writes a **replayable stroke score**. Two ways to feed it an image:

```
# one command — repaint a photo in a medium, colours derived from the photo:
plakat paint from photo.png --medium oil-direct --palette image --style fidelity -o out.png

# or in a spec (full control), no generation:
#   reference: "photo.png"   medium: watercolour   style: fidelity   palette: image
plakat paint my.paint.hjson -o out.png
```

- **`--medium <name>`** on `paint from` applies the *full* technique (bleed/opacity/impasto/broken/contour/…),
  exactly like the spec path — the individual flags still override it.
- **`--palette image`** (or `palette: image` in a spec) **derives a palette from the reference's dominant
  colours** so the photo repaints cleanly in any medium, instead of being forced through a fixed palette that
  can't hold its colours. Recommended for photos.
- Everything else (style, define, stroke length/width, the material dials) applies as normal.

The **target quality** for each technique is a real painting in that medium — a palette‑knife impasto oil, a
luminous watercolour, a matte gouache, a graded ink wash, a graphite pencil landscape.

---

## Global controls (any technique)

| HJSON field        | CLI flag            | Range / values                         | What it does |
|--------------------|---------------------|----------------------------------------|--------------|
| `subject`          | —                   | prose                                  | What to render; the model builds the armature from it. |
| `reference`        | —                   | image path                             | Paint an existing image instead of rendering an armature. |
| `negative`         | —                   | prose                                  | What to avoid in the armature (`deformed hands, back turned`). |
| `model`            | —                   | `sdxl` `sd35` `sana` `sd15` `pixart` `pony` `flux` | Armature model — the **ceiling** for the painting. |
| `steps`            | —                   | 1–100                                  | Armature diffusion steps. More = clearer figure / net / faces. |
| `medium`           | —                   | see techniques below                   | The technique — sets all behaviour defaults. |
| `palette`          | `--palette`         | `zorn` `split-primary` `verdaccio` `earth` `limited-landscape` `sumi` · time/mood presets · `image` | Pigment set (subtractive Kubelka‑Munk mixing). `image`/`auto` derives from the reference. **All names & presets → [PAINT_PALETTE.md](PAINT_PALETTE.md).** |
| `style`            | `--style`           | `fidelity` \| `legible` \| `impressionist` | How closely to track the armature. |
| `define`           | `--define`          | 0..1                                   | Edge hardness — how many boundaries meet crisply. |
| `haze`             | `--haze`            | 0..1                                   | Aerial perspective (background recession). Use **0** with `fidelity`. |
| `budget.strokes`   | `--strokes`         | int                                    | Total marks. Auto‑derived from size × medium if unset. |
| `stroke_length`    | `--stroke-length`   | 0.2..4  (1 = default)                  | Mark **length** — longer = clean sweeping strokes; shorter = choppier. |
| `stroke_width`     | `--stroke-width`    | 0.3..3  (1 = default)                  | Mark **width** — wider = fewer/broader; narrower = finer/more. |
| `bleed`            | `--bleed`           | 0..1                                   | Wet‑into‑wet fusion / bloom (wet media). Applied pass by pass, tapering coarse → fine, plus a light final touch. |
| `diffuse`          | `--diffuse`         | -1..+1 (0 = none, default)             | **Pigment diffusion** — which way the wet pigment *travels*: `+1` into the darks (they charge up, lights stay clean, crisp light‑side edges), `-1` out into the lights (feathered halos, softened edges). Mass‑conserving, wetness‑gated, needs `bleed` > 0. Recorded (replay‑exact). |
| `gradation`        | `--gradation`       | 0..1 (0 = off, default)                | **Keep slow ramps continuous.** The armature's value masses turn a cloud, a soft‑lit wall or still water into a few flat tones with contour edges. Where the picture is a ramp, not an edge (the flow rule's scale‑free test), `gradation` restores the source's smooth gradation in the armature and holds the value snap off there, so the mass keeps its turning form. Edges snap as before. |
| `new`              | `--new`             | false (default) / true                 | **A new painting, from scratch** (RFC PAINT‑1 §1.1). By default `paint` tracks its source closely — a painterly rendering of the image. `--new` reads the picture once into a **reduced armature** — 384 px for the background (512 px when the picture has no faces), 384 px across each figure and 256 px across each face *at their own extent* (§5.2) — snapped into masses with firm contours, and never looks at the source again: the things in the picture survive, its texture does not. **`--armature N` is the detail dial** of a new painting: it names the background tier and the figure and face tiers follow in ratio (576 keeps more, 192 is looser). Each plane has a **minimum brush** the size of its armature's pixel (broad in the background, finest on faces only, §9), the figure's silhouette is a seam strokes end at, the budget is a **coverage count**, and an opaque medium starts on a **toned ground** (the picture's mean colour — a gap between broad marks is the picture's tone, not white priming; a transparent medium keeps its paper). The focal tier is sized by the smallest main face the detector finds. The brushwork invents the surface. Plan files take `new: true`. |
| `hair_mask`       | `--hair-mask`       | path to a grey PNG (white = hair)      | **The strand tool over a region you name.** Hair is high‑frequency *directional* texture and the armature is structure with the texture taken out, so a from‑scratch painting has nothing to paint hair *from*. Where this mask is white the painter changes **tool**: many raked bristle lanes (the lanes *are* the strands), almost no pickup so strands stay distinct instead of smearing into mud, long narrow marks tapering to a point, laid along the picture's own **growth direction** (a structure tensor reduced to armature resolution), and about a third of them may break the subject silhouette so hair reads as hair at its edge. Given, it **replaces** the part detector, which boxes only the hair it can name — a long beard came out strands at the top and a smooth mass at its fall. Build one with `plakat segment` or `plakat remove --what`. Resized to the painting and lightly feathered, since a hard edge in the mask is a hard edge in the tool. Works on both paths; `--new` additionally gives the region a finer minimum brush. |
| `infill`          | `--infill`          | `follow` · `flat` · DEGREES            | **What a stroke follows where the picture gives it nothing to follow** — a flat passage, which in a dark interior is most of the canvas. `flat` lays long level marks, the way a painter blends a sky: right for atmosphere, wrong for a dark mass, where every stroke then runs horizontally and the passage tiles into a visible **rectangular quilt**. `follow` carries the direction inward from the nearest structure that has one, so a dark mass is stroked along the shelf edge or silhouette that bounds it. A **number** is a fixed stroke angle in degrees from horizontal — the painter's own decision about a passage. Default: `follow` for `--new`, `flat` otherwise (the path whose renders are already accepted). |
| `rigger`          | `--rigger`          | 0..1 (0 = off)                         | **Put back the few shapes too THIN for the brush ladder to lay at all** — a stem, a spoon handle, the line of a shelf. Anything narrower than the finest brush does not soften, it *disappears*; a painter finishes with a rigger and puts those few things back. Draws **ridges**, not edges (a thin shape is lighter or darker than *both* its sides, so an edge detector fires beside it and never on it), in the shape's **own colour**, and only where the painting actually **lost** one — so a shape the brushwork already carried is left alone, and every stroke removes its own reason to lay another. Rationed hard: a wiry picture is worse than a missing stem. Default 0.35 for `--new`, 0 otherwise. |
| `hotspot`         | `--hotspot`         | 0..1 (0 = leave it)                    | **Polish a flat, blown specular highlight** — the shine on a bald head, a glazed pot, wet stone. The armature snaps such a highlight into *one* value mass and the brush fills it flat, so what should be a turning form reads as **a hole cut in the picture**: a pale plateau with a hard rim. This re-models it as a **dome**, brightest at its own centre and easing to the value its rim already sits against. The gradient is **invented** from the spot's geometry and the canvas around it — nothing is copied from the source, because a painter does not trace a highlight, they know it has a soft edge. Only the **value** moves; the hue stays, since a highlight is a lightness event. A highlight the brushwork already modelled is **not** a plateau and is left alone. `0` keeps the plateau; the default softens it while keeping the highlight; `1` models it fully. Default 0.5 for `--new`, 0 otherwise. |
| `technique`       | `--technique`       | `wet-on-wet` · `wet-on-dry` · `dry-on-dry` | **How wet the paper is when pigment lands — as what the pigment then does.** `wet-on-wet`: a stroke goes down flooded and spreads, washes bloom into each other with a wide wet edge, nothing dries between layers, spatter blooms soft and wide, and no edge ever sets hard enough for a rim to form. `wet-on-dry`: a loaded brush on dry paper — a wash lands where it is put and blooms only within itself, each layer **sets** before the next, and as it dries its pigment migrates to the boundary and leaves the **rim** (painted last, along every wash's recorded ring, so no restate pass can smooth it away). `dry-on-dry`: a barely-loaded brush dragged over dry paper — about half the pigment, laid **raked** so the bristles skip and the tooth breaks every mark; nothing bleeds, nothing pools, and what pigment there is granulates into the hollows; spatter is hard dry dots. Under any technique the washes are **glazed**: each laid over everything darker than it, so there are no slivers of paper between neighbouring masses (the old white specks) and every wash boundary is a hard edge — the drawing of the picture. The technique says *how* pigment behaves; the plan's `splatter` / `edge_pool` / `granulate` / `leak` say *how much*, and the two compose. |
| `ladder_keep`     | *(plan only)*       | integer                                | **Keep only this many of the coarsest brushes for the sheet.** Watercolour's economy: a wash is the statement, and a painter does not go back over it with a fine brush. The brushes cut from the ladder are **kept for the face alone** when the picture has one, so a face is never painted with nothing finer than a wash — faces stay recognisable. A judgement about a picture, so it lives in the plan. |
| `leak`            | `--leak`            | 0..1 (0 = none)                        | **Runs of pigment dripping out of the wet washes** under gravity, tapering to a drop where they dried — the mark that says *this was liquid* more than any other. Only the big wet masses leak, from their lower edge, in their own pigment; `wet-on-wet` runs further, `dry-on-dry` never leaks, and a run is never started across a face. A watercolourist courts it or guards against it, so nothing leaks unless asked. |
| `ridges` | `--ridges` | 0..1 (default 0) | **Bristle ridges in the relief.** A stroke is a bundle of lanes each carrying its own load; with `ridges` on, the paint a lane leaves stands as high as that lane was loaded relative to its neighbours, so the stroke's relief is striated across its width and the impasto relight shows bristle marks where it showed a smooth tube. The relief is then scaled against a robust height (the 98th percentile) rather than the single peak, so one heavy crossing of strokes cannot flatten every other ridge on the sheet. Pair with `impasto` 0.6–0.9 on an oil. Recorded in the score; replay-exact. |
| `skip` | `--skip` | 0..1 (default 0) | **Dry-brush skipping.** A bristle with little water no longer floods the paper's hollows — it touches only the standing fibres, and the drier it is the fewer it reaches — so a dry mark is **broken along its drag** by the tooth instead of printing as a solid, gritty band. Driven by the stroke's own wetness (a loaded wash is never broken; the fine detail marks, laid drier, and `--technique dry-on-dry` are), over the same deterministic grain the watercolour's granulation settles into. The watercolour wash recipe turns it on by itself. Recorded in the score; replay-exact. |
| `fine_lines` | `--fine-lines` | 0..1 (default 1) | **Watercolour fine lines.** How far the finest brushes' marks may run along an edge beyond the ladder's cap: 1 = a rigger's line (a mullion, a rail, a basin's edge drawn as one mark in the picture's own pigment), 0 = short marks only — no drawn lines, the edges left to the washes. Only the wash recipe reads it. |
| `impasto_map` | `--impasto-map` | 0..1 (default 0) | **Impasto mapped to the picture.** The paint's thickness follows the picture instead of being one thickness over the sheet: the **lights thick, the shadows thin** (the paint stands where the light falls; a shadow is a glaze), the **subject thick, the background thin** (the far planes recede as thin paint), nearer thicker when a depth map is known, and **thick where the form is broken, thin where it is smooth** (a beard, bark, cobbles stand up; a cheek, a sky, glass are laid smooth and blended — read from the picture's own texture at the mark's scale). Set per stroke as the brush's viscosity and recorded (`visc=`), so the replay lays the same height. Pair with `impasto`/`ridges` on an oil. |
| `weave` | `--weave` | 0..1 (default 0) | **Canvas weave.** A linen under the paint — the warp and weft domain‑warped (the cloth stretched unevenly on its bars), interlocked over‑and‑under, each thread's thickness wandering along its length (slubs), fibrous roughness over all — seen in the relight where the paint is **thin or bare**; a passage built up to the sheet's median height covers the threads entirely (judged on the local height, so a ridged stroke's furrows count as paint). Recorded in the score; replay‑exact. |
| `hdr` · `hdr_amount` | `--hdr` · `--hdr-amount` | true/false (default false) · 0..1 (default 0.6) | **Re-light the picture before painting.** A local tone-map on the source, before the palette, the armature and every pass read it: each pixel's value is compared with its surround (the picture blurred at a twelfth of the sheet), the surround is compressed toward a mid-grey so a dark passage comes **up** and a blown one comes **down**, the local contrast is kept (softened only in the lights, never in the darks), and the picture's colours are kept exactly — a brown stays that brown, lighter. The darks then have something to paint and the lamps are not a hole in the sheet; a nocturne stays a nocturne. Works for every medium (the detectors still read the original file). `hdr_amount` is how far. |
| `splatter` · `edge_pool` · `granulate` | *(also CLI flags)* | 0..1                | Now settable from the plan too, so a watercolour's **flicked drops**, its **dried wash rim** and its **pigment settling into the tooth** can be tuned per picture alongside `technique` and `reserve` rather than only on the command line. |
| `threads`          | `--threads`         | 0 = every core (default), 1, 2…        | **Parallel pigment mixing.** Each pass's new colours are solved up front on this many threads; one painter then lays the strokes in the classic order, so the thread count **never changes the picture** (mixing was 99% of a stroke's cost; measured 13× on 18 cores at 1024²). Not recorded. Set `PLAKAT_PAINT_PROFILE=1` to print per‑pass timings to stderr. |
| `fill`             | `--fill`            | 0..1 (0 = auto, default)               | **Density floor.** The gates stop a pass where the canvas already agrees with the reference; when the painting ends below this share of the budget, the finest pass repeats with its restate floor halved each round (up to six) until the share is spent. More worked and denser on demand — it adds marks, not detail the reference lacks. Replay‑exact. |
| `opacity`          | `--opacity`         | 0.1..1                                 | Body — 1 = opaque; low = transparent (ground glows through). |
| `pickup`           | `--pickup`          | 0..1                                   | Dirty‑brush drag — fuses neighbouring colour. |
| `impasto`          | `--impasto`         | 0..1                                   | Textured thick‑paint relief, relit from stroke height (oil/knife). |
| `chroma`           | `--chroma`          | 0.3..2  (1 = neutral)                  | Saturation range — >1 vivid (oil); <1 muted (gouache/watercolour). |
| `dry_shift`        | `--dry-shift`       | −0.4..0.4                              | Value change on drying — + lighter (watercolour); − matte mid (gouache). |
| `granulate`        | `--granulate`       | 0..1                                   | Pigment settling into the paper tooth — mottled watercolour/graphite grain. |
| `sheen`            | `--sheen`           | 0..1                                   | Gloss specular on paint ridges (oil glossy; watercolour/gouache matte). |
| `lift`             | —                   | 0..1                                   | Wipe removability — oil lifts freely; watercolour staining barely. |
| `broken`           | `--broken`          | 0..1                                   | Broken colour — per‑stroke hue/chroma variation so adjacent marks optically mix (vibrancy). |
| `contour`          | `--contour`         | 0..1                                   | Line pass — draws the strongest edges as clean lines (pen/pencil/charcoal). |
| `saliency`         | `--saliency`        | 0..1 (0 = off, opt‑in)                 | Saliency‑gated density — reserve strokes for the focal subject, lay flat/empty regions thin. Stops a big budget over‑working the background into a uniform hatch. Block‑in always covers. |
| `reserve`          | `--reserve`         | 0..1 (default 0.72, surface‑white)     | Paper‑white cutoff — cells brighter than this keep bare paper (no stroke). Raise toward 1 to **close white holes** in light passages; lower to keep more paper. |
| `focus_detail`     | `--focus-detail`    | 0..1 (0 = off, opt‑in)                 | Selective detail — paint the masses **loose** but fire the crisp detail tier only in a central focal region (saliency × centre prior). Loose‑wash + sharp accents. Pair with `--style impressionist`. Small = tighter focus. |
| `preserve_face`    | `--preserve-face`   | 0..1 (0 = off, opt‑in)                 | Like `focus_detail` but the focal region is the **detected face box** (SCRFD), so crisp detail lands on the real face however it sits. Higher = more of the (feathered) face kept crisp. Needs a reference/photo with a face. |
| `splatter`         | `--splatter`        | 0..1 (0 = off, opt‑in)                 | Flick fine pigment **droplets** across the painting — watercolour/ink spatter (spray, snow, sparkle, freckles). Recorded strokes (replay‑exact). |
| `edge_pool`        | `--edge-pool`       | 0..1 (0 = off, opt‑in)                 | Darken pigment at **wash boundaries** — the watercolour edge‑bloom / "cauliflower" ring. Output stage, recorded. |
| `paper_edge`       | `--paper-edge`      | 0..1 (0 = off, opt‑in)                 | Fade to a **deckled bare‑paper border** — the torn‑paper watercolour vignette. Output stage, recorded. |
| `contrast`         | `--contrast`        | 0.5..2 (1 = neutral)                   | Finish‑grade tonal **contrast** (S‑curve). Painting‑safe, recorded for replay. |
| `warmth`           | `--warmth`          | −1..1 (0 = neutral)                    | Finish **white‑balance** shift — + warm (amber), − cool (blue). Recorded for replay. |
| `clarity`          | `--clarity`         | 0..1 (0 = off)                         | Gentle **local contrast** (midtone punch) — NOT edge sharpening (that would re‑introduce photographic detail). Recorded for replay. |
| —                  | `--armature`        | px (~48–96)                            | **Paint from a COARSE armature**, not the photo — the plan‑vs‑pixels switch (stops tracing → no beard scribble). Unset = filter mode. See [PAINT_PLAN.md](PAINT_PLAN.md). |
| —                  | `--armature-face`   | px (~160–220)                          | With `--armature`, paint the detected FACE from a finer armature — crisp face + wash body in one pass. |
| —                  | `--plan`            | `auto` \| path                         | Analyze the image (art director) → structural plan, or load a `plan.hjson`. Fills unset structural flags. See [PAINT_PLAN.md](PAINT_PLAN.md). |
| —                  | `--value-key`       | 0..1 (`paint from`)                    | Expand the reference's tonal range before painting — real darks/lights vs a foggy midtone. |
| `surface.size`     | `--size`            | `WxH`                                  | Output size. |
| `seed`             | —                   | int                                    | Reproducible. |

**Application vs. material.** `bleed`/`opacity`/`pickup`/`impasto`/`stroke_*` control *how the paint is
applied*; `chroma`/`dry_shift`/`granulate`/`sheen`/`lift` model *how the paint itself behaves* (its saturation
range, drying value‑shift, granulation, gloss, and removability) — so a watercolour and an oil differ at the
material level, not only in softness. Every material value defaults per medium and is recorded in the score, so
`plakat paint replay` reproduces the technique exactly. The **palette** also defaults to the one that suits the
medium (sumi for ink/pencil, split‑primary for gouache, limited‑landscape for watercolour) when the spec names
none.

### `style` — the fidelity register
- **`fidelity`** — paint **closely** from a sharp armature: tight edge‑following, clean low‑waver strokes, most
  passes resolve detail. For **recognizable, detailed** results. Pair with `haze: 0`.
- **`legible`** — resolves features but keeps a painterly looseness (soft masses + a detail tier).
- **`impressionist`** — deliberately loose: soft masses of colour, no detail tier, no edge hardening.

### `model` — the armature is the ceiling
The stroke engine can only paint what the armature contains. `sd35` (best anatomy/hands/faces; low‑mem for
24 GB Metal) · `sdxl` (balanced default) · `sana` (light/fast, painterly — good drafts) · `sd15` (fastest) ·
`pixart` (artistic) · `pony` (stylized) · `flux` (coherent — **GGUF Flux broken on Metal**).
**Workflow:** draft cheap with `sana`/`sd15`, finalize with `sdxl`/`sd35`.

### Per‑element brushes (composition)
For a `composition.elements` scene, each element takes a `brush` from the vocabulary —
`flat` (masses/skies) · `filbert` (general) · `round` (faces/detail) · `fan` (foliage/waves) ·
`rigger` (ropes/masts/fine lines) · `knife` (broad opaque slabs) · `wash` (sky/watercolour washes).

---

## Techniques and their tuning

Each medium ships characteristic defaults; override any of them in the spec.

### Oil — `medium: oil-direct` (alla‑prima) / `oil-indirect` (layered)  🎯 *palette-knife impasto landscape*
Opaque, buttery, **distinct** marks that sit on top of one another; **thick paint catches light (impasto)**;
a dirty brush harmonises colour; broken colour laid in adjacent dabs.
- Defaults: `opacity 1.0` · `impasto 0.60` (oil‑direct) / `0.40` (indirect) · `bleed 0.08` · `pickup 0.30`/`0.40`.
- Tune: **`impasto` up (→0.8)** and **`stroke_width` up (→1.3)** for a chunky palette‑knife feel; `pickup` up for
  more wet‑blend; `stroke_length` up for confident strokes. Try `brush: knife` per element. Keep `bleed` low.
- Good with: `style: fidelity`, `palette: zorn` / `limited-landscape` / `split-primary`.

```hjson
medium: oil-direct
style: fidelity
impasto: 0.8
stroke_width: 1.3
pickup: 0.35
```

**Uniform "canvas‑weave" on full‑coverage subjects (portraits, dense scenes).** The `impasto 0.60` default is
tuned for a landscape with breathing room; on an image where *every* pixel is painted, the relief relights the
whole surface evenly and reads as a uniform embossed texture overlay (an AI‑detection tell) rather than genuine
brush ridges. Fix by **lowering the relief** — `impasto` (and its gloss companion `sheen`) are the knob, no
other change needed. Relief and *detail* are independent axes: keep your stroke budget and only drop the relief.
- `impasto 0.50–0.70` — heavy relief; reads as a texture overlay when the surface is fully covered.
- `impasto 0.15–0.30` · `sheen 0.05` — subtle brushwork relief, **no weave** (recommended default for portraits).
- `impasto 0` — dead flat.

```hjson
# Portrait / full-coverage oil — subtle relief, no canvas weave
medium: oil-direct
impasto: 0.22   // 0..1 — thick-paint relief; the source of the weave
sheen: 0.05     // 0..1 — glossy specular on the ridges
granulate: 0    // 0..1 — paper-tooth mottle (a watercolour/graphite thing; keep 0 for oil)
```
CLI equivalent: `plakat paint from photo.png --medium oil-direct --impasto 0.22 --sheen 0.05 --granulate 0 -o out.png`

### Watercolour — `medium: watercolour`  🎯 *luminous wet-in-wet washes*
Transparent, luminous, **the paper glows through**; pigment **bleeds and blooms** wet‑into‑wet; whites are the
reserved paper.
- Defaults: `opacity 0.45` · `bleed 0.55` · `pickup 0.15` · `impasto 0` · whites reserved automatically.
- Tune: `bleed` up (→0.7) for looser washes/bloom; `opacity` down (→0.35) for more paper glow; `stroke_length`
  up for flowing washes. Keep `pickup` low, `impasto` 0.
- Good with: `style: fidelity`/`legible`, `palette: limited-landscape`/`sumi`, `haze: 0`.
- **The fluid stage** (the wash recipe, `PLAKAT_WC_WASH=1`): after each **broad** pass the water the brush
  left is treated as a continuous film — the pass's own wet pigment (never the dried passes beneath) is split
  into **washes**, the regions where one pigment leads the mix, and inside each it diffuses and settles toward
  the wash's edge as it dries: the **tide line** where the water meets a different load (a lit window against
  the wall, a wash against bare paper — two washes of one weight laid together blend without a line), ragged
  with the paper and heavier along the wash's lower edge where the water runs down. A staining dye travels
  further than an earth, so mixed washes separate at their edges; and `granulate` happens **in the water** —
  the earths settle into the tooth where the wash pooled (the finish keeps only a trace of uniform grain).
  The fine passes are wet‑on‑dry and stay crisp. The water is **selective**: it floods the soft masses and holds
  back a brush‑radius short of the picture's hard edges (read from what the sheet shows at that moment, so a
  replay sees the same edges) — the painter's choice of where to work wet‑in‑wet and where wet‑on‑dry.
  `--technique wet-on-wet` floods wider and softer, `dry-on-dry` has no water to flow; the score records it
  (`flow=strength,radius,rim,grain,selective`) so a replay crosses the same water.

```hjson
medium: watercolour
style: fidelity
bleed: 0.6
opacity: 0.5
pickup: 0.15
```

### Gouache — `medium: gouache`  🎯 *matte opaque, flat washes + crisp marks*
Opaque **matte** body colour; flat even coverage; crisp opaque marks laid light‑over‑dark (back‑to‑front); no
gloss.
- Defaults: `opacity 1.0` · `impasto 0.15` (matte, slight body) · `bleed 0.05` · `pickup 0.80`.
- Tune: `pickup` down (→0.4) for flatter poster‑like blocks; `stroke_width` up for broad flat fills; keep
  `bleed` low and `impasto` low (matte, not oily).
- Good with: `style: legible`, opaque palettes (`split-primary`, `earth`).

```hjson
medium: gouache
opacity: 1.0
impasto: 0.15
pickup: 0.45
stroke_width: 1.2
```

### Ink wash — `medium: ink-wash`  🎯 *graded monochrome washes (sumi-e)*
Flowing transparent washes; strong bleed; whites reserved; graded from light to dark.
- Defaults: `opacity 0.40` · `bleed 0.70` (very fluid) · `pickup 0.90` · `impasto 0`.
- Tune: `bleed`/`pickup` are the character — lower for tighter washes. `palette: sumi`, `style: legible`.

```hjson
medium: ink-wash
palette: sumi
bleed: 0.7
opacity: 0.4
```

### Pen & ink — `medium: pen-ink`  🎯 *crisp line + hatching (line-and-wash)*
Crisp lines and **hatching / cross‑hatching**; value built by mark **density**, not paint body. No bleed/pickup.
For *line‑and‑wash*, render the ink first, then a separate watercolour pass (compose two layers).
- Defaults: `bleed 0.0` · `opacity 1.0` · `pickup 0.0` · density marks (darker = more hatch).
- Tune: `stroke_length` for longer hatches; `define` up (→0.8) for crisp contours. `palette: sumi`.
  `bleed`/`opacity` have little effect (marks are discrete).

```hjson
medium: pen-ink
palette: sumi
style: legible
define: 0.8
```

### Pencil / graphite — `medium: pencil`  🎯 *soft graphite landscape, hatched shading*
Monochrome **graphite grey** (not black); value by **hatching / shading density**; soft graded tones from light
**smudging**; fine lines; paper white for highlights. No colour, no impasto.
- Defaults: `body 0.7` (greys, semi‑transparent) · `bleed 0.15` (smudge → soft gradients) · `pickup 0.10` ·
  density marks · reserved paper whites.
- Tune: `bleed` up (→0.3) for smoother, smudgier shading; `bleed` down for crisper pencil lines;
  `stroke_length` for longer strokes; `define` up for a sharper drawn line. Use `palette: sumi`.

```hjson
medium: pencil
palette: sumi
style: fidelity
bleed: 0.2
define: 0.7
```

### Pastel — `medium: pastel`  🎯 *soft, chalky, high-chroma, blendable*
Chalky opaque strokes, **high chroma**, laid as **broken colour** and blended where they cross; matte, slight
tooth grain.
- Defaults: `chroma 1.15` · `broken 0.40` · `granulate 0.25` · `pickup 0.40` · `impasto 0.10` · matte (`sheen 0`).
- Tune: `broken` up for more shimmer; `pickup` up for softer blends; `chroma` for vividness.

```hjson
medium: pastel
style: fidelity
broken: 0.45
```

### Charcoal — `medium: charcoal`  🎯 *dramatic soft black, smudgy*
Rich near‑monochrome black, **smudgy** graded tones, drawn **and** shaded; lift highlights from the paper.
- Defaults: `chroma 0.40` · `bleed 0.30` (smudge) · `contour 0.30` (draws too) · `granulate 0.35` · `lift 0.60`.
- Tune: `bleed` up for softer smudging; `contour` up for stronger drawn lines; `palette: sumi`.

```hjson
medium: charcoal
palette: sumi
bleed: 0.35
contour: 0.4
```

### Acrylic — `medium: acrylic`  🎯 *fast-dry, plastic sheen, darkens on drying*
Opaque plastic colour, **fast‑dry** (clean stacked layers, little wet blend), a **plastic sheen**, and a slight
**darken on drying**; permanent (no lifting).
- Defaults: `body 1.0` · `impasto 0.35` · `sheen 0.20` · `dry_shift −0.03` · `pickup 0.15` · `lift 0.0` · `chroma 1.10`.
- Tune: `sheen` for gloss; `impasto` for thick relief; keep `pickup`/`bleed` low (fast‑dry).

```hjson
medium: acrylic
style: fidelity
impasto: 0.4
sheen: 0.25
```

### Tempera — `medium: tempera`  🎯 *fine cross-hatched build-up (egg tempera)*
Fine, semi‑opaque **cross‑hatching**; many small disciplined marks, little blending.
- Defaults: `opacity 0.85` · `bleed 0.03` · `pickup 0.05` · `impasto 0.12` · density marks · many stages tolerated.
- Tune: `stroke_width` down (→0.8) and moderate `stroke_length` for fine hatching; keep `bleed`/`pickup` low.

```hjson
medium: tempera
style: fidelity
stroke_width: 0.8
```

---

## Quick reference — default behaviour by medium

| medium        | body | bleed | pickup | impasto | chroma | dry_shift | granulate | sheen | lift | broken | contour | marks      | whites   | palette |
|---------------|:----:|:-----:|:------:|:-------:|:------:|:---------:|:---------:|:-----:|:----:|:------:|:-------:|------------|----------|---------|
| oil-direct    | 1.00 | 0.08  | 0.30   | 0.30    | 1.08   | 0.00      | 0.00      | 0.15  | 0.80 | 0.35   | 0.00    | continuous | pigment  | zorn |
| oil-indirect  | 0.90 | 0.10  | 0.40   | 0.40    | 1.05   | 0.00      | 0.00      | 0.12  | 0.70 | 0.25   | 0.00    | continuous | pigment  | zorn |
| gouache       | 1.00 | 0.05  | 0.80   | 0.15    | 0.85   | −0.05     | 0.00      | 0.00  | 0.50 | 0.30   | 0.00    | continuous | pigment  | split-primary |
| watercolour   | 0.45 | 0.55  | 0.15   | 0.00    | 0.92   | +0.08     | 0.35      | 0.00  | 0.20 | 0.15   | 0.00    | continuous | reserved | limited-landscape |
| ink-wash      | 0.40 | 0.70  | 0.90   | 0.00    | 0.90   | +0.05     | 0.22      | 0.00  | 0.10 | 0.10   | 0.20    | continuous | reserved | sumi |
| line-and-wash | 0.32 | 0.50  | 0.00   | 0.00    | 1.70   | +0.08     | 0.20      | 0.00  | 0.20 | 0.10   | 0.40    | WASHES · the picture's own colours, each mass once from the source at wash scale, bloomed edges, wet modelling touches within, a coloured ink line; LUMINOUS (no family split / committed shadows, mild key) | reserved | limited-landscape |
| early-book-illustration | 0.32 | 0.35 | 0.00 | 0.00 | 1.50 | +0.08 | 0.16 | 0.00 | 0.20 | 0.10 | 0.40 | WASHES · flat cumulative glazes through masks from the keyed picture, high key, a coloured line — the early colour‑book print | reserved | limited-landscape |
| japanese-ink  | 0.40 | 0.70  | 0.90   | 0.00    | 0.90   | +0.05     | 0.06      | 0.00  | 0.10 | 0.10   | 0.15    | continuous · one wide soft brush, long calligraphic strokes, its own black | reserved | sumi (always) |
| pen-ink       | 1.00 | 0.00  | 0.00   | 0.00    | 0.80   | 0.00      | 0.00      | 0.00  | 0.00 | 0.00   | 0.60    | density    | reserved | sumi |
| durer         | 1.00 | 0.00  | 0.00   | 0.00    | 0.80   | 0.00      | 0.00      | 0.00  | 0.00 | 0.00   | 0.70    | density · engraving: lines follow the form, fine, cross-hatched darks | reserved | sumi (always) |
| pencil        | 0.70 | 0.15  | 0.10   | 0.00    | 0.70   | 0.00      | 0.18      | 0.05  | 0.60 | 0.00   | 0.50    | continuous · graphite point: thin grey directional strokes, its own grey | reserved | sumi (always) |
| black-pencil  | 0.70 | 0.15  | 0.10   | 0.00    | 0.70   | 0.00      | 0.18      | 0.05  | 0.60 | 0.00   | 0.50    | continuous · the graphite mark with a soft dark point | reserved | sumi (always) |
| tempera       | 0.85 | 0.03  | 0.05   | 0.12    | 0.95   | 0.00      | 0.10      | 0.05  | 0.30 | 0.20   | 0.00    | continuous · short strokes, each pass cross-hatched 45° | pigment  | verdaccio |
| pastel        | 0.90 | 0.10  | 0.40   | 0.10    | 1.15   | 0.00      | 0.25      | 0.00  | 0.50 | 0.40   | 0.00    | continuous | pigment  | split-primary |
| charcoal      | 0.80 | 0.30  | 0.20   | 0.00    | 0.40   | 0.00      | 0.35      | 0.00  | 0.60 | 0.00   | 0.30    | continuous | reserved | sumi |
| acrylic       | 1.00 | 0.05  | 0.05   | 0.15    | 1.10   | −0.03     | 0.00      | 0.20  | 0.00 | 0.10   | 0.00    | continuous | pigment  | split-primary |

Every value is a **default** — set the same‑named field in the spec (or the CLI flag) to override it. The score
records the applied `bleed` / `opacity` / `impasto` so a re‑render (`plakat paint replay`) reproduces the
technique exactly.
