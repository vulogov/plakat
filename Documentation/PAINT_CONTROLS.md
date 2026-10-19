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
| `ridges` | `--ridges` | 0..1 (default 0) | **Bristle ridges in the relief.** A stroke is a bundle of lanes each carrying its own load; with `ridges` on, the paint a lane leaves stands as high as that lane was loaded relative to its neighbours, so the stroke's relief is striated across its width and the impasto relight shows bristle marks where it showed a smooth tube. The relief is then scaled against a robust height (the 98th percentile) rather than the single peak, so one heavy crossing of strokes cannot flatten every other ridge on the sheet. Pair with `impasto` 0.6–0.9 on an oil. With `ridges` on the section is also **plowed** — the brush pushes paint to the stroke's lateral edges, a trough with a steep ridge along each side (more so for a flat brush) — and the relief casts **micro‑shadows**: a thick stroke beside thin paint is a step, and the raking light throws the step's shadow onto the thin side, so thick paint sits *on* a smooth passage instead of fading into it. Recorded in the score; replay-exact. |
| `skip` | `--skip` | 0..1 (default 0) | **Dry-brush skipping.** A bristle with little water no longer floods the paper's hollows — it touches only the standing fibres, and the drier it is the fewer it reaches — so a dry mark is **broken along its drag** by the tooth instead of printing as a solid, gritty band. Driven by the stroke's own wetness (a loaded wash is never broken; the fine detail marks, laid drier, and `--technique dry-on-dry` are), over the same deterministic grain the watercolour's granulation settles into. The watercolour wash recipe turns it on by itself. Recorded in the score; replay-exact. |
| `fine_lines` | `--fine-lines` | 0..1 (default 1) | **Watercolour fine lines.** How far the finest brushes' marks may run along an edge beyond the ladder's cap: 1 = a rigger's line (a mullion, a rail, a basin's edge drawn as one mark in the picture's own pigment), 0 = short marks only — no drawn lines, the edges left to the washes. Only the wash recipe reads it. |
| `analysis` | `--analysis [PATH]` | true / path (default off) | **The analysis artefact** (RFC PAINT‑3). A Markdown report of the run beside the picture (`<out>.md`, or the path given): the plan verbatim with its comments, every dial the painter read with where its value came from (plan / cli / default) and what it does, what the run found (faces, matte, semantic regions, hair, HDR, the budget), the pass table with per‑pass stroke length, width and dry share, measurements from the finished canvas (paint height per plane and by value, fine‑mark lengths, pigment usage), and a glossary of the non‑default dials. Every number is a fact of the run; nothing is invented. |
| `analysis_insights` | `--analysis-insights [PROVIDER]` | true / provider (default off) | **Insights and recommendations** (RFC PAINT‑3 P0.5). Runs the analysis report through the configured LLM (`auto`, `ollama`, `ollama:<model>`, `deepseek`, `gemini`, `local`) with a versioned system prompt (`assets/prompts/paint_insights.md`) and appends `## Insights` — findings, each citing the measurement it rests on — and `## Recommendations` — an hjson block of `key: value # reason` plan lines you can paste, plus what the model could not judge. The model gets the report's **facts**, never the picture; it may only recommend dials from the glossary. **Privacy:** a hosted provider (DeepSeek, Gemini) receives the report text — it leaves your machine; Ollama/local keep it here. The terminal says which before sending. Implies `analysis`. |
| `outcome` | `--outcome [PATH]` | true / path (default off) | **The outcome sheet** (RFC PAINT‑3 P1). A one‑page sheet beside the picture (`<out>_sheet.png`, or `.png`/`.pdf` at the path): the master composition; four micro‑analysis lens crops chosen by **rule** — the faces (or the focal plane), the thickest paint off the faces, the most textured background passage, the brightest thin light — each captioned with its medium's technique word and two measured numbers; the layer hierarchy (the canvas after every pass, bottom to top); the palette ranked by use with the nearest classic pigment name and hex; the brushwork concept — the recorded stroke paths drawn as ink (broad faint, fine dark, drawn lines darkest) with one arrow per cell for the marks' dominant direction, a legend of mark types counted from the score, and the pass table. Typeset with Typst (`brew install typst`); the `.typ` source is kept beside the output and in `<out>_sheet/` with every image it uses, so the sheet can be edited and recompiled. Every panel is a fact of the run. |
| `impasto_map` | `--impasto-map` | 0..1 (default 0) | **Impasto mapped to the picture.** The paint's thickness follows the picture instead of being one thickness over the sheet: the **lights thick, the shadows thin** (the paint stands where the light falls; a shadow is a glaze), the **subject thick, the background thin** (the far planes recede as thin paint), nearer thicker when a depth map is known, and **thick where the form is broken, thin where it is smooth** (a beard, bark, cobbles stand up; a cheek, a sky, glass are laid smooth and blended — read from the picture's own texture at the mark's scale). Set per stroke as the brush's viscosity and recorded (`visc=`), so the replay lays the same height. Pair with `impasto`/`ridges` on an oil. **Mechanism:** at 0 the thickness follows the pigment LOAD, and dark masses carry the most pigment — so with the map off or weak the shadows come out thickest and the lights thinnest; the map at 1 is what reverses that. If a run measures shadows thicker than lights, the map is too *weak*, not too strong. |
| `weave` | `--weave` | 0..1 (default 0) | **Canvas weave.** A linen under the paint — the warp and weft domain‑warped (the cloth stretched unevenly on its bars), interlocked over‑and‑under, each thread's thickness wandering along its length (slubs), fibrous roughness over all — seen in the relight where the paint is **thin or bare**; a passage built up to the sheet's median height covers the threads entirely (judged on the local height, so a ridged stroke's furrows count as paint). Recorded in the score; replay‑exact. |
| `collide` | `--collide` | 0..1 (default 0) | **Wet collision.** After each broad pass the wet paint MOVES where loaded strokes meet: each pixel pulls colour and height from a point back along the paint's own striation (the height field's structure tensor — so a replay sees the same drags), the colours marbling in concentration (Kubelka‑Munk, not alpha) and the ridges plowing with them; nothing of it a blur. Only wet paint moves (a dried pass stays), only where the paint is coherent. The broad passes drag at the bleed's taper with the **faces held**; the fine passes drag at a third of it with the whole **subject held** too — so the fine marks over the ground and the background marble into their wet base while what the fine brushes resolved on the figure is never softened. The hold goes into the score as a coarse three‑level mask, so the replay crosses the same drags. Oil/acrylic. Recorded (`collide=`); replay‑exact. |
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
| `sheen`            | `--sheen`           | 0..1                                   | Gloss specular on paint ridges (oil glossy; watercolour/gouache matte) — an independent pass on the relief's light‑facing slopes, so the ridges read as wet oil rather than matte clay. Plan key too. |
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
| `oldpaper`         | `--oldpaper`        | `true` / `false` (default `false`)     | Print the finished picture on an **aged sheet** — yellowed laid paper, foxing, specks, worn edges; `durer` also gets its **plate mark**. Any medium. Output stage, recorded. |
| —                  | `--depth`           | `auto` / a grey image (white = near)   | `durer` only: the picture's **relief**, so the cuts turn round the form (round a tyre, round a barrel). `auto` estimates it with Depth-Anything and saves `<out>.depth.png`. CLI only. See [PAINT_DURER.md](PAINT_DURER.md). |
| —                  | `--normals`         | `auto` / a normal-map picture (green = up) | Off unless given. Which way the picture's **surfaces** face. `durer`: the cuts wrap the form and lie level on the ground. Painting media (oil, watercolour, …): strokes turn the same way on passages the picture gives no direction of their own; edges the picture draws keep their stroke. `auto` estimates them with Marigold-Normals as its own pass and saves `<out>.normals.png`. Ignored by the other drawing media. CLI only. See [PAINT_DURER.md](PAINT_DURER.md). |
| —                  | `--materials`       | `"fur: hare; glass: lamp; skin"` / a saved map | `durer` only: what things are **made of**, named in words; the masks are found by OWL-ViT + MobileSAM and saved as `<out>.materials.png`. Kinds: `fur`, `glass`, `metal`, `skin` (`skin` alone: the faces, by the face detector; the hatch is held at rest on them). CLI only. See [PAINT_DURER.md](PAINT_DURER.md). |
| —                  | `--follow`          | 0..1 (default 1)                       | `durer` only: how far the hatch **follows the form**. 0 = a ruled plate (every line on the diagonal or level on the ground); a portrait wants about 0.3. CLI only. See [PAINT_DURER.md](PAINT_DURER.md). |
| —                  | `--white`           | 0.5..1 (unset: found in the picture)   | `durer` only: the **white point** as a share of the picture's values — the highlight clipping threshold. At or above it the paper is clean. Unset, the paper is the picture's brightest broad flat value (a sheet, a sky, a wall) or its brightest thousandth; 0.98 cuts more of a white dress, 0.8 leaves a tinted sheet clean. CLI only. See [PAINT_DURER.md](PAINT_DURER.md). |
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
- **The watercolour** (7.2: the default for a `--new` watercolour; `PLAKAT_WC_LEGACY=1` gives the 7.0 glazed‑wash path back): the paint is a transparent film over the paper — darks by density, tints that keep their chroma — and the whole brush ladder paints on it, light to dark. **The fluid stage**: after each **broad** pass the water the brush
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

### Gouache — `medium: gouache`  🎯 *layered opaque body colour: rewetted seams, chalky lights, a flat brush and a liner (7.9.0)*
Opaque **matte** body colour; flat even coverage; crisp opaque marks laid light‑over‑dark (back‑to‑front); no
gloss. From 7.9.0 it is painted in **layers**, each dry before the next (the acrylic's machinery through the
plan), the colours **premixed in value steps** and drawn along the mark (the tempera's `dishes` and `ramp`), the
calm passages graded (`smooth`), and four mechanisms of its own:
- **Rewet** (`rewet`, 0..1, in the score's header): gouache's gum stays soluble — a wet mark lifts this share of
  the DRIED paint beneath into its load, by its own wetness and how empty it is, so the layers fuse at their
  seams instead of sitting as cut-out shapes.
- **Chalk tint** (`chalk_tint`, 0..1): a light is mixed with white, and white takes the colour's chroma — the
  target's chroma cut by up to `0.6·chalk_tint`, rising from L* 55 to 95, off the faces.
- **Flat brush** (`flat_brush`, 0..1): on the passes before the finest, off the faces and the hair, this share of
  the marks on a flat passage (by its flatness) are a flat brush's patch — a square chisel end, full to its end,
  straighter: a field, a wall, a sail.
- **Liner** (`liner`, 0..1): after the ladder the LONG thin structures the brushwork lost — a rigging line, a
  stay, a wire: the rigger's ridge, oriented over a long window, traced along itself both ways (snapped onto its
  crest, stopping at a 12° turn a step or 30° in all), and laid as ONE thin mark in its mean colour only if it
  runs a thirtieth of the sheet — a texture's curls are not lines. Each a record (stage `liner`).
- The value **settles** on drying with a NEGATIVE `dry_shift` (the darks a shade lighter, the lights a shade
  darker — a positive one is a plain lightening).
- `saliency` and `focus_detail` are plan keys from 7.9.0 (they were flags only): the fine brush in the focus.
- Defaults: `rewet 0.3` · `chalk_tint 0.5` · `flat_brush 0.5` · `liner 0.5` · `opacity 1.0` · `impasto 0.15` ·
  `bleed 0.05` · `pickup 0.80`; every other medium has the four at 0. The recommended plan is in `PAINT_EXAMPLES.md`.
- Tune: `dishes` down for a flatter poster, up (with `smooth`) for graded skies; `focus_detail` down for a
  broader periphery; `rewet` up for softer seams; `liner` 0 and `rigger` up for a finer, wirier drawing.

```hjson
medium: gouache
palette: image
new: true
dry: 1.0
dishes: 10
rewet: 0.3
liner: 0.5
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

### Pastel — `medium: pastel`  🎯 *soft pastel on a toothed paper: chalk, the stick's side, the finger (7.7.0)*
Chalky opaque marks, **high chroma**, laid as **broken colour**; matte. From 7.7.0 the medium is a stick on a
toothed sheet, built in layers:
- **Chalk** (`chalk`): the deposit at a pixel is keyed to the paper's grain against the mark's **pressure** — a
  light touch reaches only the raised fibres and the hollows keep the sheet's colour, a hard press fills them —
  and to how **full** the tooth already is (the filled hollows count as reached; past three fifths full the
  chalk no longer holds and slides over what is there). The layers build as pastel does: the first broken on
  the grain, the later filling it, the last barely taking. The medium's, in the score's header.
- **The side of the stick** (`side`): on the broad passes, off the faces and the hair, this share of the marks
  (the key's share on the block-in, falling with the ladder) are wide flat sweeps at a light touch —
  `1 + 0.6·side` of the width, the stick's facets as streaks, full width to the end, straighter, `1 − 0.5·side`
  of the pressure (the chalk catches the raised grain only), `1 − 0.3·side` of the charge, dry, no pickup.
- **The finger** (`blend`): after each pass of the broad half of the ladder the **calm** passages (a sky, a
  sea, a wall; the subject at three tenths, never a face or the hair) are rubbed along the flow — the
  underpainting blended, the fine strokes over it left discrete — a smudge with no pigment of its own that
  swaps a share of the paint under it with what it carries, there and back, so the marks there fuse into one
  soft tone and nothing is added or removed. Every cell of the finger's width is rubbed by its calm, a light
  touch (up to `0.08·calm` of the paint moved on the ground, `0.25·calm` on a face), the finger's ends soft,
  its path stopping at the picture's own edges as a mark does (it blends within a passage, never across a
  silhouette, a hairline or a change of colour); each rub a `smudge` record in the pass's stage, replayed to the bit.
- **The skin** (`skin`): the finger on the **faces** — a face's calm passages (a cheek, a forehead; not the eyes,
  which the range reads as busy) are rubbed after the face's passes — the finest too, the skin alone — by this
  share of their flatness, the rub kept inside the face, so the skin's chalky lights fuse into modelling and
  only the features keep their marks. The finger has a size of its own (a fortieth of the sheet's short side)
  whatever the stick of the pass. Needs `blend`.
- **Carnation** (`carnation`): the flesh in **warm sticks** — on a face (the hair apart) a mark's mixture is
  solved over the palette's warm and neutral pigments alone (a pigment cold in CIELAB, a* or b* below −6, is
  left in the box), so a half-tone is darkened with an umber or a black and never greyed with the sea's teal;
  and the face's lights are lifted along their own hue, the chroma kept, instead of toward white — a Naples
  yellow over the cheek, not a chalky grey. A face's glints too are lifted along their hue.
- **Small** (`small`, px): a face whose extent (the face detector's box, not the soft focal mask) is under
  this is drawn at its own scale, by `k = 1 − extent/small`: the marks narrower, the hatching thinned and one
  way along the form (no cross turn), the half-tones held near the local tone, more of it blended. A portrait
  is untouched. **Small's floor** (`small_min`, px): a crowd's face under it — a feature the width of the
  stick — is left to the plain ladder (`small`, `anchor`, `trim` and the warm glint off).
- **Anchor** (`anchor`, 0..1): on a small face the structural darks — the eyes, the nostrils, the mouth's
  line, the darkest share of the tones around a mark — are neither held to the local tone nor blended, and
  the blending falls off with the local contrast, so a lip keeps a dry stroke while a cheek is fused.
- **Trim** (`trim`, 0..1): the face is the detector's box **cut by the figure** (the subject matte), and the
  ground's marks keep this share of their brush's radius clear of it — the sea's broad marks no longer cross
  a profile, and the box's corners off the head are ground, not flesh (both came through the tooth as a pale
  grey-green crackle on the cheek).
- Defaults: `chalk 1.0` · `side 0.5` · `blend 0.5` · `skin 0.5` · `carnation 1.0` · `small 250` · `anchor 0.15` · `trim 0.5` · `small_min 100` · `chroma 1.15` · `broken 0.40` · `granulate 0.15` ·
  `pickup 0.40` · matte (`sheen 0`). The recommended plan is in `PAINT_EXAMPLES.md`.
- Layer keys: `chalk`, `side`, `blend`, `skin`, `carnation`, `small`, `small_min`, `anchor`, `trim`, with the shared `dry`, `wash_in`, `wet_cap`, `scumble`, `skip`, `deplete`,
  `granulate`, the tempera's hatch and dish keys for the final strokes and the limited box of sticks, and
  `pickup`, `bleed` and
  `min_brush` (plan keys from 7.7.0: a stick picks up nothing of the layer beneath and chalk does not run —
  both 0 in the pastel plan; and the stick's own limit, 4 px on a 2048 sheet — a stick of chalk cannot make a
  one-pixel mark, and the face tier's own floor fell below a pixel for a small face).
- Tune: `side` up for a broader, grainier underlayer; `blend` up for softer skies and walls; `chalk` down for a
  smoother, brush-like mark.

```hjson
medium: pastel
palette: image
new: true
dry: 1.0
wash_in: 0.0
chalk: 1.0
side: 0.6
blend: 0.6
carnation: 1.0
small: 400
small_min: 100
anchor: 0.15
trim: 0.5
```

### Charcoal — `medium: charcoal`  🎯 *a stick of charcoal on white paper: the paper is the light, the stump the dark (7.8.0)*
Rich near‑monochrome black, **smudgy** graded tones, drawn **and** shaded; lift highlights from the paper.
From 7.8.0 the medium is a drawing: the pastel's stick on the tooth (`chalk`, `side`, the finger `blend`)
in a single black, and the light left to the paper:
- **Paper** (`paper`, 0..1): charcoal only darkens. A mark goes where the sheet is still lighter than the
  picture asks, in the darkest pigment alone, charged for the residual over what is already there (the charge
  solved on a scrap of the same stroke, the overlap of a pass's marks counted); where the sheet is already as
  dark, nothing is laid.
- **Erase** (`erase`, 0..1): where the sheet went darker than the picture's lights ask, the mark is a lift — a
  kneaded eraser, a wipe record whose strength takes the film back toward the value (in the upper two thirds
  of the perceived range, off the block-in).
- **Touch** (`touch`, 0..1): the pressure follows the value — a light tone a light touch on the raised grain,
  a dark one a full press into the hollows (`0.38 + 0.6·dark^0.7`, by this share).
- **Velvet** (`velvet`, share of the perceived range): the darkest share one velvet black — the darkest pigment
  itself at a full press.
- **Paper top** (`paper_top`, 0..1): the picture's light end (its 95th percentile) is the paper, so a
  near-white ground stays bare paper; the lift at most ×1.25 and weighted by the square of the value's share
  of the light end (a night's light end is itself dark). **Paper knee** (`paper_knee`, 0..1): the top of the
  curve bent toward the paper — the upper two fifths of the perceived range pulled toward it by a smoothstep,
  in a light picture only — so a broad light mass (snow) is the paper with a touch of grain.
- **Dust** (`dust`, 0..1, in the score's header): under the `chalk`, a hollow the stick's reach misses still
  takes this share of the reach's deposit — the charcoal's dust settling into the tooth. A light touch keeps
  the grain, a pressed half-tone fills it (without it the hollows below the reach stayed white under every
  layer: white specks across a grey blanket and the snow).
- **Sweep** (`sweep`, 0..1): on a flat passage off the subject and the faces the flow is turned onto the
  horizontal, by this share of its flatness — the sky in calm side-of-the-stick sweeps. **Sweep wave**
  (`sweep_wave`, degrees): the swept flow turned by a slow noise across a sixth of the sheet — an arm's arc.
- **Stump** (`stump`, 0..1): after the finest pass the dark masses (by perceived darkness) and the flat
  passages off the subject are rubbed with a stump — the finger's `smudge` records, `0.25·calm` strong — the
  paper's specks in a dark passage filled. **Stump edge** (`stump_edge`, 0..1): it holds back where the
  perceived local contrast stands at a finger's reach (fully at a gamma range of 0.12) — a lapel against the
  dark kept. **Stump feather** (`stump_feather`, 0..1): its stop at a face and the hair is soft (their mask
  blurred over this share of a finger), and it may cross the subject's silhouette where the picture has no
  edge of its own — a lost edge blended, a found one kept.
- **Contour** (`contour`): a plan key from 7.8.0 (the profile's line pass was not one); 0 in the charcoal plan.
- Defaults: `paper 1.0` · `erase 0.6` · `touch 0.6` · `velvet 0.08` · `paper_top 1.0` · `paper_knee 0.5` ·
  `dust 0.85` · `sweep 0.9` · `sweep_wave 4` · `stump 0.5` · `stump_edge 0.6` · `stump_feather 1.0`; every
  other medium has them at 0. The recommended plan is in `PAINT_EXAMPLES.md`.
- Tune: `stump` up for softer, velvet darks; `stump_edge` up to keep more of the figures in a night scene;
  `paper_knee` up for whiter lights; `dust` down for a grainier sheet; `sweep` down for a sky that follows its
  clouds.

```hjson
medium: charcoal
palette: sumi
new: true
paper: 1.0
stump: 0.5
```

### Japanese ink — `medium: japanese-ink`  🎯 *sumi-e: the paper is the light, the washes boneless, the line the bone (7.10.0)*
One black ink in a few dilutions on an absorbent white sheet, in TWO REGISTERS over a wash painting: the ink's
own mechanisms, each a plan key and 0 in every other medium (all of them byte-identical), over the charcoal's
paper (the ink only darkens, the light is the paper, the pressure follows the value), the acrylic's wash-in
and the gouache's liner and focus, borrowed through the plan.
- **Tsuke** (`tsuke`, 0..1, in the score's header): tsuketate — the brush loaded in two tones across its
  width (the lanes `1 → 1−0.85·tsuke`), the dark edge turned to the picture's shadow side.
- **Ma** (`ma`, 0..1): the empty space — where the eye does not go (the focal field, the subject) the value is
  lifted toward the paper by `ma·(1−attention)²`: the periphery a stroke or two. Needs `paper`.
- **Kasure** (`kasure`, 0..1, header): the flying white — the hairs part as the brush runs, a lane dropping out
  where its own slow hash falls under `kasure·t^1.5` along the mark.
- **Nijimi** (`nijimi`, 0..1, header): the ink feathering into the sheet — the mark's reach widened by
  `0.8·nijimi·(0.3+0.7·wet)`, the outer part a halo falling off outward and caught where the paper's fibres
  stand; and the water evening the hairs, the streak damped by `nijimi·wet`. **The flood** (nijimi's): a mark
  wetter than 0.5 (fully at 0.9) is set by its water — tsuke's two tones mixed, kasure's white gone, the hairs
  not running dry along the sweep (`deplete` damped — a hair run dry mid-band was a pale dash on every wash),
  the ends faded in and out over half a width (a square-cut end was a notch of paper against the next band),
  the halo's gate opened so it is a feathered edge.
- **Turn** (`turn`, 0..1, header): the brush lifted on a turn — the width narrowed by up to `0.6·turn` where
  the path bends (fully at a right angle over half a width each way).
- **Yohaku** (`yohaku`, 0..1): the blank white — a FLAT passage (perceived local range at a sixty-fourth of
  the sheet under 0.03, by a smoothstep to 0.10) off the subject lifted to the paper by this share: a fog, a
  pale sky, a white ground left bare. Needs `paper`.
- **Sumi dark** (`sumi_dark`, 0..1): the share of the darkest values the black register lays as dry-brush
  black shapes (stage `ink`); unset, the darkest eighth (which drew a dark coat as one black shape).
- **Bone** (`bone`, 0..1): the LINE register (骨法) — the value's edges thinned to a pixel and kept by
  hysteresis (the pen-and-ink planner's edge map, opened by `bone`), linked into chains, each chain ONE brush
  line whose weight (`0.5..1.5 × 0.8·min_brush`) and ink (`0.35..1 × charge`) follow the darker side of the
  edge along it — a near dark form a full black line, a far pale one a thin grey hair; tapering to its
  lift-off, `turn` and `kasure` acting on it; longest first under a cap of `0.10+0.25·bone` of the budget; a
  chain where the eye does not go (`ma`) must run up to twice as far, and ON THE SUBJECT (whose texture is
  the fine brush's — a feather's barbs, a tuft of fur) twice as far again and laid lighter (×0.7): the line
  holds the silhouette and the long structures. Stage `bone`, after the ladder, before the black register.
  The painter dries the sheet before the bone and again before the black register, and the replay does the
  same at those stages, so a wet sheet (`bleed`) replays to the bit.
- **Boneless** (`boneless`, a count): 没骨 — the block-in and the next `boneless` passes of the ladder laid
  as the block-in's flooded wash (`wash_in`: wet, a thin charge), at an even width with a round soft edge: a
  tonal mass a wash, not a heap of dry tapered wedges. Needs `wash_in`.
- **Hake** (`hake`, 0..1): on the washed passes the flow turned to the broad flat brush's one horizontal
  sweep by this share, off the subject — the masses built from parallel bands that overlap along themselves
  and stop where the tone changes, not from bars crossing at every angle (crossed, each translucent bar was
  a pane: a sheet of broken glass). Needs `boneless`.
- Defaults: `tsuke 0.6` · `ma 0.5` · `kasure 0.5` · `nijimi 0.4` · `turn 0.5` · `yohaku 0.6` · `bone 0.5` ·
  `boneless 2` · `hake 0.7`; every other medium has them at 0. The recommended plan is in `PAINT_EXAMPLES.md`.
- No white: the plan sets `hotspot 0` (its dots of titanium were pale dashes on the ground — the light is the
  paper), and on paper the `liner` lays no line lighter than the sheet under it.
- Tune: `bone` up for a wirier drawing, down for a wash alone; `boneless` down (1) for a drier, more broken
  ground; `hake` 0 for washes that follow the form; `bleed` up for washes that flow into one another; `sumi_dark`
  up for more solid black; `ma`/`yohaku` up for more paper; `dishes` 4 holds the ink to four tones.

```hjson
medium: japanese-ink
palette: sumi
new: true
paper: 1.0
bone: 0.6
boneless: 3
```

### Acrylic — `medium: acrylic`  🎯 *fast-dry, layered like the indirect oil, darkens on drying*
Opaque plastic colour painted in LAYERS, each **dry before the next** goes on: the block-in is a thin, flooded
**wash underpainting** (`wash_in 0.5` — 65 % of the charge), every later mark goes on no wetter than `wet_cap
0.45` and sits on the dried layer as a **flat patch with a crisp edge** (no pickup, no bleed, no blending), the
**finest pass is dragged dry** off the faces (`scumble 1` — a thinner charge, raked, no pickup) over a brush
that **skips on the canvas's tooth** (`skip 0.3`); a synthetic flat brush (smooth lanes, a square edge); the
colour **dries darker** (`dry_shift −0.05`) under a satin sheen and a thin relief; permanent (no lifting). The
flow of every pass is turned by a low-frequency **turbulence** of ±8° (halved on a face), so the marks are never
quite parallel; on the finest pass an **accent** — a short, thick, opaque dab set off the flow at its own angle —
goes where the picture's local contrast is sharp (a glint, a hard edge); and off the subject and the faces the
painting is **loose**: the broad passes' marks run half again as long and wider, and the restating passes thin
with the ladder (the finest keeps half its marks there) — the background is the big brush's flat patches, the
subject the small brush's. Best as `--new --plan auto` (the painting from scratch with the face and figure
tiers: the face gets the fine armature and the finest brushes, the background the broad ones). The paint has
**body**: a heavy relief (`impasto 0.45`) with bristle **ridges** (`ridges 0.5`) and a thickness that follows the
picture (`impasto_map 0.6` — lights thick, shadows thin, the subject thick, a face laid smooth), relit with its
micro-shadows under a satin sheen; a mark's charge **runs out along it** (`deplete 1.8` — thick at the start, a
dry tail over the layer beneath); and along the subject's silhouette the finest pass lays **glazes** (`glaze
0.6` — thin, wet, picking up the paint beneath, free to cross the seam), so the figure meets its ground in a
translucent overlap, not a cut edge. On a face, where the picture is smooth at the brush's scale, the paint is
**sfumato** (`sfumato 0.7`): the fine passes thin there with the ladder and the marks that remain go on broad,
wet, half-charged and picking up the paint beneath, so the skin is fused, not hatched into a grain; one more
**glaze pass** after the ladder (`glaze_pass 0.3` — a mid brush, every mark a thin wet film over everything but the faces and hair, which their own tier finishes)
unifies the detail under one tone — leaving the sharp contrasts alone, so the accents stand crisp over the film;
and the mark's length is spread as a Gaussian (`vary 0.35`), wider where the picture is flat. The hand's
**pressure** runs along the mark (`comma 0.5`): the width and the deposit swell a third of the way along and
lift toward the ends, so a round brush leaves drops and commas, not tubes; a bristle whose charge has run out
skips on the tooth as a dry one does, so a depleting mark's tail breaks instead of thinning evenly; and on a
face under the sfumato the drift of hue is wider, so the fused skin is not one interpolated gradient. The
picture has a **hierarchy**: in the shadows, off the faces, a mark lands only where the canvas is plainly
wrong, never to chase a small tone, and the marks that land run longer, wider and calm — no dry skip, no hue
drift (`shade 0.6`) — so the dark masses rest and the detail is kept for the light; the block-in is laid as an **imprimatura** (`imprimatura 0.6` — darker and warmer than the
picture, toward a burnt umber), the underpainting that shows between the later marks, in a dry mark's skips
and the thinned shadows; and each mark's **body** is spread as a Gaussian (`body_vary 0.3`): one thick and
opaque, standing high, the next thinner, wetter and translucent. On the finest pass and the glaze pass the
subject's edge is **lost** where it lies in shadow (`lost 0.6`): the mark's colour is the picture blurred wide
across the seam and the mark runs across it, so a dark figure meets its dark ground in one tone, not a drawn
outline; one more pass after the glaze is a **dry brush over the ridges** (`crest 0.4`, the share of its
seeds): a fine brush, where the relief is broken, a lighter, warmer tint that catches only the tops of the
relief beneath (the stroke's `crest`, in the record), the texture lit without a new shape; on the same pass, in
the shadow, a **wash into the furrows** (`sink 0.5`, the share): a thin cool wash that settles only below the
ridges and runs off their tops (a negative `crest` in the record); the lost edges go **by depth** (`aerial
0.6`): a depth map is estimated (Depth-Anything) and a far passage in shadow loses its edges wherever it lies,
the farther the more contrast it may hold and the wider the blur its marks take — the background softens
itself by its distance, and the relief thins toward the far planes; and the thickest marks in the light fling **spatter** (`spatter 0.5`): a drop
or three off the lift-off along the mark's direction, round thick dots of its own paint, a fast mark's (a long
one for its width) flying farther and smearing ahead into a **tail** (`tail 0.6`). The glaze marks (the glaze
pass, the sink wash, the lost edges, the silhouette overlaps) lay a share of their paint as a **film over** the
dried layer (`film 0.7`): the engine mixes pigment by Kubelka-Munk already, but a glaze mixed into a heavy
passage is outweighed by it — laid as its own layer, read by the two-layer Kubelka-Munk solution (a film of
finite thickness over a substrate, the stroke's `over` in the record), a thin yellow over blue reads green
and a cool wash over a warm shadow stays a cool veil; the dry marks (the scumble, the crest, the hair strands)
**split** into two to four bristle bundles with bare canvas between as they dry out and the faster they go
(`split 0.5`, the stroke's `split` in the record); and at output the canvas **sags** under the thick paint
(`sag 0.4`): a broad shallow dip under every heavy passage and a slow unevenness of the cloth on its bars, a
gentle swell of the light over the whole sheet, recorded in the score's header; and a **gloss** on the peaks
(`gloss 0.5`, in the header too): a sharp glint only where the thick paint stands above its surround and high
against the sheet's own peak, the crests of the heavy marks glossy and the thin passages matte. The bundles of a
split mark differ a little in **tint** (`tint 0.4`, the stroke's `tint`): each clump of bristles carries its
own slight variant of the load. Off the subject, on the broad passes, a share of the marks are a **palette
knife** (`knife 0.3`, the stroke's `knife`): a flat blade full width with square ends that shears the paint
standing under it down to a plateau and lays its own flat, thick and nearly dry — smooth planes with sharp
edges cut into the relief, against the brushed faces; a share of the knife marks **skim** (`skim 0.4`): the
blade at a low angle with little paint, pressed less and catching only the relief's tops (the stroke's `crest`),
a torn, holed plane. Each pigment covers by its own **hiding** power (`hiding 0.5`, in the header): the
cadmiums, whites and earths opaque, the lakes and the organic blues and greens translucent laid thick, a
pigment the picture named read from its masstone (dark and saturated = a lake), so a glaze of a lake lets the
ground and the paint beneath through where a cadmium hides them. The brush is **twisted** in the fingers along
every hair strand and a share of the finest marks off the faces (`twist 0.5`, the stroke's `twist`): its flat
turned toward its edge by up to a quarter turn from one end, the width running from the brush's breadth to a
line, the calligraphic mark. **Every one of these is a plan key** — tune them in the plan, not in the source: see the recommended plan in
`PAINT_EXAMPLES.md`.
- Defaults: `dry 1.0` · `body 1.0` · `impasto 0.3` · `sheen 0.2` · `dry_shift −0.05` · `pickup 0.05` · `bleed 0` · `lift 0.0` · `chroma 1.05` · `broken 0.15`.
- Layer keys (plan keys; the medium's own values when absent): `dry 1` · `wash_in 0.5` · `wet_cap 0.45` · `scumble 1` · `skip 0.3` · `turbulence 8` · `accent 1` · `loose 0.5` · `vary 0.35` · `broken 0.15` · `sfumato 0.7` · `glaze 0.6` · `glaze_pass 0.3` · `comma 0.5` · `shade 0.6` · `imprimatura 0.6` · `body_vary 0.3` · `lost 0.6` · `aerial 0.6` · `crest 0.4` · `sink 0.5` · `spatter 0.5` · `tail 0.6` · `film 0.7` · `split 0.5` · `tint 0.4` · `knife 0.3` · `skim 0.4` · `twist 0.5` · `hiding 0.5` · `sag 0.4` · `gloss 0.5` · `deplete 1.8` · `ridges 0.3` · `impasto_map 1` · `impasto 0.3` · `sheen 0.2`.
- Tune: `--dry` below 1 lets the layers blend (an oil habit); `--skip 0` for a smooth canvas; `sheen` for gloss; `impasto` for relief.

```hjson
medium: acrylic
new: true
impasto: 0.2
sheen: 0.2
```

### Tempera — `medium: tempera`  🎯 *egg tempera on a gesso panel: dry layers, hatched (7.6.0)*
The panel painting as egg tempera is handled: the paint **dries in seconds**, so nothing beneath is ever picked
up or stirred (`dry 1.0`, `wet_cap 0.35`); the block-in is a wash that covers the white gesso (`wash_in`), laid
with the ladder's own flat brush; and every restating pass is **hatched** (`hatch`) — the mark a straight line
of one width end to end, longer for it, dry and semi-opaque, the hatching growing with the ladder from its
middle, so the broad restatements lay the form's masses along the flow and the finest pass is all hatching.
The hatching **crosses the form** (`cross`, degrees): the direction is the local flow (the structure tensor's
isophote — along the form, across the gradient), and each hatched pass is turned off it by its own step,
alternating sides — along the form, then +15°, −30°, +45° — so the net wraps the volume at the panel
painter's angles instead of lying over it as one grid turned 45° a pass (at `cross 0` the profile's
`hatch_angle × layer`). The hatch's charge is keyed to the value (`solid`): a veil in the picture's lightest
value, covering in its darkest, so the deep shadows are built solid in layers and the lights stay thin. The
hatch's length follows how busy the form is within the brush's reach (`span`): long calm lines on a sky or a
wall, short hatching on a face, a hand, small folds. Near a plain contour (`edge`) the crossing turn is taken
back toward the edge's own direction and the mark stops sooner on a change of colour, so the hatching runs
parallel to the silhouette and does not fringe across it. The pure lights are set last (`glint`): on the finer
half of the ladder, in the picture's top twentieth of values — a star, the sun through the leaves, a lamp —
the mark is lifted toward white, laid narrower, dry, with the full covering charge and standing thicker, the
raised final touch; a glint on an isolated light smaller than the brush (a small star) is a stamp, a dot,
not a mark grown along the flow. The **calm passages** are built up in soft veils, not hatched (`smooth`): where the form
changes little within the brush's reach — a sky, a wall, still water — the dishes relax toward the picture's
own gradation (the block-in's too, so a sky is not banded), the hatching gives way, and the restating mark goes
on wetter, shorter and thinner so it blends into its neighbours: the smooth mottled ground of a panel's sky,
over which the glints stay crisp dots and the contours stay sharp. The calm is the ground's only — a subject's
dark mass (a black cat) stays committed paint, built solid in its steps. A new painting begins on the **bole**
(`bole`): before the block-in the sheet is laid thin with the picture's own shadow tone as a continuous field (the
mean of each cell's darker quarter, smoothed and interpolated between the cells — a toned ground is darker
than the lights), the ground's mean colour pulled toward it by the key, so a gap between broad marks shows the local shadow tone and not a
flat brown, and the block-in's marks, mixing by concentration with the thin bole beneath them, are pulled
toward the tone they lie on (stage `ground` in the score). The **shadow is one modelled mass** (`umbra`):
within that share of the darkest dish's interval a tone is laid as it is, continuous, the dishes step the
lights above it, and off the subject the shadow is laid calm however busy the picture is there (built in
veils, as the sky is) — a night's foreground is no longer a chequer of black and dark-grey planks. A mark's colour
**follows the picture along its path** (`ramp`): the brush's charge drifts from the seed's colour toward the
picture's colour at the mark's end by the key's share, so the broad passes model a gradient — a sky, the
ground falling into shadow — instead of laying flat planks across it; full on the soft marks, fading as the
hatching takes over, off the glazes, lost edges, hair, accents and glints; the end mix rides in the score
(`mix2`). The flesh is underpainted in **verdaccio** (`verdaccio`): on a face the block-in and the first restatement are the
greenish grey of the panel painters (terre verte, white, a little ochre and black) at the picture's value, and
the later passes in the half-tones and shadows are pulled toward it and laid thinner, so the green shows
through the hatched flesh where the form turns; held to the **warm** colours (`flesh`) — where the picture's
own colour inside the face's region is cold or neutral (a hat, a wall, metal; CIELAB a*, b*) the pull is taken
away. The tones go in **dishes** (`dishes`, a count): each colour mixed in advance in that many value steps
between the picture's own darkest and lightest values, and every mark — the block-in's flat local tones too —
laid in the nearest one, the veils alone keeping the picture's gradation: the stepped modelling of a panel,
softened by the hatching over it; **dithered** after the block-in (`dither`) — where a value lies between two
dishes the mark draws between them by its fraction of the way, so the band between two steps (a lamp's halo
in rings) breaks into an optical mixture of marks of both. The **lights** are hatched in last (`lights`): on
the finest pass, on the subject's top values, a share of the marks are the picture's colour lifted toward
white, narrower and thinner. No impasto, no knife, no spatter, no sag: a thin matte skin on a smooth panel.
Every other medium is byte-identical. The recommended plan is in `PAINT_EXAMPLES.md`; tune there, not in the
source.
- Defaults: `body 0.9` · `bleed 0.03` · `pickup 0.05` · `impasto 0.12` · `dries 1.0` · `wash_in 0.3` · `wet_cap 0.35` · `hatch 0.8` · `verdaccio 0.6` · `dishes 5` · `lights 0.5` · `cross 15` · `dither 0.7` · `solid 0.7` · `flesh 0.8` · `span 0.6` · `edge 0.7` · `glint 0.6` · `smooth 0.9` · `bole 1.0` · `ramp 0.7` · `umbra 1.0` · the ladder's own brushes (flat → filbert → round).
- Layer keys: `dry`, `wash_in`, `wet_cap`, `hatch`, `verdaccio`, `dishes`, `lights`, `cross`, `dither`, `solid`, `flesh`, `span`, `edge`, `glint`, `smooth`, `bole`, `ramp`, `umbra`, plus the shared ones (`deplete`, `turbulence`, `loose`, `shade`, `vary`, `body_vary`, `broken`, `glaze`, `glaze_pass`, `film`, `lost`, `aerial`, `twist`, `hiding`).

```hjson
medium: tempera
palette: image
new: true
dry: 1.0
wash_in: 0.15
wet_cap: 0.3
hatch: 0.8
verdaccio: 0.6
dishes: 5
lights: 0.5
cross: 15
dither: 0.7
solid: 0.7
flesh: 0.8
span: 0.6
edge: 0.7
glint: 0.6
smooth: 0.9
bole: 1.0
ramp: 0.7
umbra: 1.0
glaze_pass: 0.0
hotspot: 0.2
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
| early-book-illustration | 0.32 | 0.35 | 0.00 | 0.00 | 1.50 | +0.08 | 0.16 | 0.00 | 0.20 | 0.10 | 0.40 | PRINT · a thirties four-colour book plate: a black key drawn from the picture over four flat inks found in it, the lights paper, one screen in the shadows, the faces drawn by the pattern — the early colour‑book print | reserved | limited-landscape |
| japanese-ink  | 0.40 | 0.70  | 0.90   | 0.00    | 0.90   | +0.05     | 0.06      | 0.00  | 0.10 | 0.10   | 0.15    | continuous · one wide soft brush, long calligraphic strokes, its own black | reserved | sumi (always) |
| pen-ink       | 1.00 | 0.00  | 0.00   | 0.00    | 0.80   | 0.00      | 0.00      | 0.00  | 0.00 | 0.00   | 0.60    | density    | reserved | sumi |
| durer         | 1.00 | 0.00  | 0.00   | 0.00    | 0.80   | 0.00      | 0.00      | 0.00  | 0.00 | 0.00   | 0.70    | density · copperplate: even cuts along the form, swelling with tone, crossed in the darks | reserved | sumi (always) |
| pencil        | 0.70 | 0.15  | 0.10   | 0.00    | 0.70   | 0.00      | 0.18      | 0.05  | 0.60 | 0.00   | 0.50    | continuous · graphite point: thin grey directional strokes, its own grey | reserved | sumi (always) |
| black-pencil  | 0.70 | 0.15  | 0.10   | 0.00    | 0.70   | 0.00      | 0.18      | 0.05  | 0.60 | 0.00   | 0.50    | continuous · the graphite mark with a soft dark point | reserved | sumi (always) |
| tempera       | 0.90 | 0.03  | 0.05   | 0.12    | 0.95   | 0.00      | 0.07      | 0.05  | 0.30 | 0.20   | 0.00    | continuous · dry layers on a gesso panel: a covering wash block-in, the restatements hatched in straight lines turned off the form by steps, long on the flat and short on the busy, parallel to the contours, solid in the darks, the warm flesh over verdaccio, the tones in dithered dishes, the lights hatched in and the pure lights set dry and raised, the calm passages soft veils, the sheet begun on a bole of the local shadow tone, the broad mark's charge ramped along its path, the darkest dish continuous | pigment  | verdaccio |
| pastel        | 0.90 | 0.10  | 0.40   | 0.10    | 1.15   | 0.00      | 0.25      | 0.00  | 0.50 | 0.40   | 0.00    | continuous · a stick of chalk on the tooth (a light touch on the raised grain, a hard press in the hollows, a full tooth sliding), the broad passes with the stick's side, the calm passages rubbed with a finger | pigment  | split-primary |
| charcoal      | 0.80 | 0.30  | 0.20   | 0.00    | 0.40   | 0.00      | 0.35      | 0.00  | 0.60 | 0.00   | 0.30    | continuous | reserved | sumi |
| acrylic       | 1.00 | 0.00  | 0.05   | 0.30    | 1.05   | −0.05     | 0.00      | 0.20  | 0.00 | 0.15   | 0.00    | continuous · a synthetic flat, each layer dry before the next, a wash block-in, a dry-brush finish | pigment  | split-primary |

Every value is a **default** — set the same‑named field in the spec (or the CLI flag) to override it. The score
records the applied `bleed` / `opacity` / `impasto` so a re‑render (`plakat paint replay`) reproduces the
technique exactly.
