# `plakat paint` — command examples

Copy‑paste recipes for every supported **medium** and every **method** of the stroke‑space painter. See
[`PAINT_CONTROLS.md`](PAINT_CONTROLS.md) for the meaning of each knob; this file is the quick command reference.

Conventions used below:
- `photo.png` — any input image (a photo or an already‑generated render). Swap in your own path.
- **`paint from` is entirely GPU‑free.** The two GPU refinements — `--crisp` (matte + crisp silhouette) and
  `--critic` (per‑pass aesthetic roll‑back) — live on the spec path (`paint <SPEC>`), shown in method 2.
- `--palette image` (alias `auto`) **derives** the palette from the image's own dominant colours — best for
  photos and portraits. Otherwise pass a named palette (`zorn`, `split-primary`, `verdaccio`, `earth`,
  `limited-landscape`, `sumi`) or let `--medium` pick its default.
- `--budget` is the total stroke count (the main quality dial) — but more is not always better: past a point a
  large budget over-works flat regions into a uniform hatch. Pair a big budget with `--saliency` (below).
- `--style`: `fidelity` (tight, crisp — oil/opaque subjects) · `legible` (overlapping washes — **wet media**) ·
  `impressionist` (loose masses).

---

## Media — repaint a photo in each technique

Each block repaints `photo.png` in one medium. `--medium` applies that technique's *full* behaviour
(application + material physics); the extra flags only fine‑tune it.

```bash
# ── OIL, alla-prima (oil-direct) ─────────────────────────────────────────────
# Opaque buttery marks, thick paint catches light. Default palette: zorn.
# NOTE: on a FULL-COVERAGE subject (portrait) drop --impasto/--sheen or the relief
# reads as a uniform canvas-weave. See PAINT_CONTROLS.md → "canvas-weave".
plakat paint from photo.png --medium oil-direct --palette image \
    --style fidelity --impasto 0.25 --sheen 0.05 -o out-oil.png

# Landscape oil with chunky palette-knife relief (breathing room → high impasto OK):
plakat paint from photo.png --medium oil-direct --palette zorn \
    --impasto 0.7 --stroke-width 1.3 --broken 0.4 -o out-oil-knife.png

# ── OIL, layered/indirect (oil-indirect) ─────────────────────────────────────
# Smoother glazed build-up, lower impasto than alla-prima. Default palette: zorn.
plakat paint from photo.png --medium oil-indirect --palette image \
    --style fidelity --impasto 0.25 -o out-oil-indirect.png

# ── WATERCOLOUR ──────────────────────────────────────────────────────────────
# Transparent, luminous, wet-into-wet bleed; paper glows through; whites reserved.
# Default palette: limited-landscape (use --palette image for a portrait).
# NOTE: use --style LEGIBLE (or IMPRESSIONIST), not fidelity — fidelity's short crisp marks leave gaps that
# stay near-white with transparent paint (a "lack of strokes" speckle). Legible lays overlapping washes.
# --reserve raises the paper-white cutoff (→ fewer white holes); --opacity firms the wash.
plakat paint from photo.png --medium watercolour --palette image \
    --style legible --bleed 0.6 --dry-shift 0.08 \
    --granulate 0.12 --opacity 0.6 --reserve 0.9 -o out-watercolour.png

# BEST for a PORTRAIT — loose wash everywhere, crisp preserved face (detected). The engine paints one register
# globally, so a detailed subject (beard) either goes crumbly (detail chases it) or mushy (all loose). Painting
# loose + firing crisp detail ONLY on the detected face box gives loose-wash + a recognizable face:
plakat paint from photo.png --medium watercolour --palette image \
    --style impressionist --bleed 0.6 --dry-shift 0.08 \
    --granulate 0.12 --opacity 0.6 --reserve 0.9 \
    --preserve-face 0.6 --budget 6000 -o out-watercolour-portrait.png
# (--focus-detail 0.3 is the no-detector fallback: a central focal region instead of the real face box.)

# ── GOUACHE ──────────────────────────────────────────────────────────────────
# Matte, opaque, flat washes + crisp marks. Default palette: split-primary.
plakat paint from photo.png --medium gouache --palette image \
    --style legible --impasto 0.15 --broken 0.3 -o out-gouache.png

# Layered gouache (7.9.0): each layer dry before the next, opaque over the one beneath, the colours premixed
# in value steps and drawn along the mark, the dried layer rewetted at the seams, the lights chalky with
# white, the flat passages in a flat brush's square patches, the long thin lines in a liner's single stroke,
# the fine brush in the focus only, the paper's grain under the paint. The plan:
#
#   medium: gouache            palette: image           style: legible
#   new: true                  armature: 768            armature_face: 768      armature_body: 768
#   semantic: true             sam: true                families: true          commit_shadows: 1.0
#   shadow_floor: 0.16         detail_len: 0.5          infill: follow          hotspot: 0.3
#   rigger: 0.0                liner: 0.5               focus_detail: 0.5
#   dry: 1.0                   pickup: 0.0              rewet: 0.3              bleed: 0.0
#   wash_in: 0.3               wet_cap: 0.45            scumble: 0.5            skip: 0.2         deplete: 1.2
#   dishes: 10                 dither: 0.0              ramp: 0.7               hiding: 0.0       broken: 0.1
#   turbulence: 4              loose: 0.3               vary: 0.3               comma: 0.3        body_vary: 0.1
#   smooth: 1.0                lost: 0.0                glaze: 0.0              glaze_pass: 0.0
#   impasto: 0.05              ridges: 0.0              impasto_map: 0.0        sheen: 0.0
#   granulate: 0.12            dry_shift: -0.08         chalk_tint: 0.5         flat_brush: 0.5
#   (rigger 0 — it traced a wheat field's ridges as thin light curls; dishes 6 banded a sky into a posterize,
#   dishes 0 was a soft photograph; dry_shift negative — the value settles toward the middle)
#
plakat paint from photo.png --medium gouache --plan gouache.hjson -o out-gouache.png

# ── INK WASH (sumi-e) ────────────────────────────────────────────────────────
# Graded monochrome washes, very fluid bleed. Default palette: sumi (black+white).
plakat paint from photo.png --medium ink-wash --palette sumi \
    --bleed 0.7 -o out-inkwash.png

# ── PEN & INK (line-and-wash) ────────────────────────────────────────────────
# Crisp line + hatching; the contour pass draws the strongest edges. Palette: sumi.
plakat paint from photo.png --medium pen-ink --palette sumi \
    --contour 0.6 -o out-penink.png

# ── PENCIL / GRAPHITE ────────────────────────────────────────────────────────
# Soft graphite, hatched shading, paper-white highlights, granular tooth. No colour.
plakat paint from photo.png --medium pencil --palette sumi \
    --contour 0.5 --granulate 0.2 -o out-pencil.png

# ── PASTEL ───────────────────────────────────────────────────────────────────
# Soft pastel on a toothed paper (7.7.0): the mark a stick of CHALK on the tooth — a light
# touch catches the raised grain only and the hollows keep the sheet's colour, a hard press fills them, and
# past three fifths full the tooth holds no more; the broad passes laid with the stick's SIDE (wide flat
# sweeps at a light touch); after each broad pass the calm passages rubbed with a FINGER, the marks there
# fused into one soft tone, nothing added or removed. Dry at once; matte; broken colour.
# Best painted from scratch with a PLAN — every layer dial a plan key, tuned there. The plan so far:
#
#   medium: pastel             palette: image           style: legible
#   new: true                  armature: 384            armature_face: 768      armature_body: 384
#   semantic: true             sam: true                families: true          commit_shadows: 1.0
#   shadow_floor: 0.16         detail_len: 0.5          infill: follow          rigger: 0.35      hotspot: 0.3
#   min_brush: 4.0             dry: 1.0                 pickup: 0.0             bleed: 0.0
#   wash_in: 0.0               wet_cap: 0.2             scumble: 0.8            skip: 0.0
#   deplete: 1.5               granulate: 0.0           chalk: 1.0              side: 0.4         blend: 0.6        skin: 0.5
#   sfumato: 0.7               carnation: 1.0           small: 400              small_min: 100    anchor: 0.15      trim: 0.5
#   hatch: 0.8                 cross: 25                span: 0.6               edge: 0.6         glint: 0.4
#   lights: 0.3                dishes: 0                dither: 0.0             umbra: 1.0
#   (the tempera's hatch and dish keys: the final strokes discrete and directional, the tones a limited box of sticks)
#   turbulence: 6              loose: 0.15              shade: 0.3              vary: 0.3         body_vary: 0.2
#   broken: 0.05               lost: 0.0                aerial: 0.0             hiding: 0.0       solid: 1.0
#   (no lost edges: a pastel's silhouette is drawn, not rubbed into its ground — the marks along a jaw in
#   shadow took the mean of the skin and the sea; the dark masses built solid so the paper's tooth is filled)
#
plakat paint from photo.png --medium pastel --plan pastel.hjson -o out-pastel.png

# ── CHARCOAL ─────────────────────────────────────────────────────────────────
# Dramatic soft black, smudgy, grainy tooth, contoured. Palette: sumi.
plakat paint from photo.png --medium charcoal --palette sumi \
    --contour 0.3 --granulate 0.35 -o out-charcoal.png

# A stick of charcoal on white paper (7.8.0): the paper is the light — every mark in the
# black alone, laid only where the sheet is lighter than the picture asks, lifted with an eraser where it went
# darker; the pressure follows the value, the darkest share one velvet black; the dust settles into the
# tooth; the flat passages in calm sweeps on an arm's arc; after the finest pass the darks rubbed with a
# stump that holds at a contrast and fades toward a face. The pastel's stick, side and finger, borrowed.
# The plan:
#
#   medium: charcoal           palette: sumi            style: legible
#   new: true                  armature: 384            armature_face: 768      armature_body: 384
#   semantic: true             sam: true                families: true          commit_shadows: 1.0
#   shadow_floor: 0.16         detail_len: 0.5          infill: follow          min_brush: 3.0
#   dry: 1.0                   pickup: 0.0              bleed: 0.0              wash_in: 0.0      wet_cap: 0.2
#   scumble: 0.0               deplete: 0.3             granulate: 0.0          contour: 0.0
#   chalk: 1.0                 side: 0.5                blend: 0.6              skin: 0.5         sfumato: 0.7
#   small: 400                 small_min: 100           anchor: 0.15            trim: 0.5
#   hatch: 0.6                 cross: 30                span: 0.6               edge: 0.6         umbra: 1.0
#   solid: 1.0                 loose: 0.15              lost: 0.0
#   paper: 1.0                 erase: 0.6               touch: 0.6              velvet: 0.08
#   paper_top: 1.0             paper_knee: 0.5          dust: 0.85
#   sweep: 0.9                 sweep_wave: 4.0          stump: 0.5              stump_edge: 0.6   stump_feather: 1.0
#   (dry, no pickup, no bleed: a replay to the bit; deplete 0.3 — at the pastel's 1.5 a long black mark faded
#   to grey on the white sheet; contour 0 — the line pass drew every hair and ripple as a black worm)
#
plakat paint from photo.png --medium charcoal --plan charcoal.hjson -o out-charcoal.png

# ── JAPANESE INK (sumi-e) ────────────────────────────────────────────────────
# One black ink on an absorbent white sheet (7.10.0): the paper is the light; the masses boneless washes —
# the block-in and three passes after it flooded, even, soft-edged, one horizontal sweep of the hake off the
# subject; the brush loaded in two tones (tsuketate), the hairs parting into flying white (kasure), the ink
# feathering into the sheet (nijimi), lifted on a turn; the empty space (ma) and the blank white (yohaku);
# the bone — the structure's contours as calligraphic lines, weighted and inked by their darker side; the
# darkest thirtieth in dry-brush black. The plan:
#
#   medium: japanese-ink       palette: sumi            style: legible
#   new: true                  armature: 384            armature_face: 768      armature_body: 384
#   semantic: true             sam: true                families: true          commit_shadows: 0.0
#   shadow_floor: 0.16         detail_len: 0.5          infill: follow          min_brush: 3.0
#   rigger: 0.0                liner: 0.5               focus_detail: 0.3       hotspot: 0.0
#   dry: 1.0                   pickup: 0.0              bleed: 0.3              wash_in: 0.6      wet_cap: 0.6
#   contour: 0.0               paper: 1.0               touch: 0.6              velvet: 0.03
#   paper_top: 1.0             paper_knee: 0.5
#   deplete: 1.0               skip: 0.0                comma: 0.5              turbulence: 0     loose: 0.0
#   vary: 0.4                  granulate: 0.05          dishes: 4               dither: 0.0
#   tsuke: 0.6                 ma: 0.8                  kasure: 0.5             nijimi: 0.8       turn: 0.6
#   yohaku: 0.7                sumi_dark: 0.03          bone: 0.6               boneless: 3       hake: 1.0
#   (commit_shadows 0 and sumi_dark 0.03 — at the profile's eighth a woman in a dark coat was one black shape;
#   contour 0 — the line pass drew worms; skip 0 and loose 0 — square textures and a trowel's smears; the bone
#   and the hake — without them the ruins were bars crossing at every angle, no line; hotspot 0 — its white
#   dots were pale dashes on the ground; bleed 0.3 — the wet bands flow into one another, replayed to the bit)
#
plakat paint from photo.png --medium japanese-ink --plan japanese-ink.hjson -o out-sumi.png

# ── ACRYLIC ──────────────────────────────────────────────────────────────────
# Opaque plastic colour in layers, each dry before the next: a thin wash block-in, flat crisp-edged marks on
# top, a dry-brush finish and a loose hand off the faces, thick accent dabs at the sharp contrasts; darkens on
# drying; the skin fused where it is smooth, a glaze pass over all but the faces, the shadows calm and broad over
# a dark warm imprimatura, the edges lost in shadow and by depth, a dry brush over the ridges and a cool wash
# into the furrows, the glazes a film over the dried paint, the dry marks split into bristle bundles of
# their own tints, a few thick drops with tails, a palette knife off the subject that sometimes skims, the
# brush twisted along the hair, each pigment hiding by its own power, the canvas sagging under the thick
# paint and glossy on its peaks. Best painted from scratch with a PLAN —
# the face and figure tiers (fine brushes on the face, broad on the background) and every layer dial as a
# plan key, tuned there rather than in the source. The recommended plan:
#
#   medium: acrylic            palette: image           style: legible
#   new: true                  armature: 384            armature_face: 768      armature_body: 384
#   semantic: true             sam: true                families: true          commit_shadows: 1.0
#   shadow_floor: 0.16         detail_len: 0.5          infill: follow          rigger: 0.35      hotspot: 0.5
#   dry: 1.0                   wash_in: 0.5             wet_cap: 0.45           scumble: 1.0      skip: 0.3
#   deplete: 1.8               turbulence: 8            accent: 1.0             loose: 0.5        vary: 0.35
#   broken: 0.15               sfumato: 0.7             glaze: 0.6              glaze_pass: 0.3   comma: 0.5
#   shade: 0.6                 imprimatura: 0.6         body_vary: 0.3          lost: 0.6         crest: 0.4
#   spatter: 0.5               tail: 0.6                aerial: 0.6             sink: 0.5
#   film: 0.7                  split: 0.5               tint: 0.4               knife: 0.3        skim: 0.4
#   twist: 0.5                 hiding: 0.5
#   impasto: 0.3               ridges: 0.3              impasto_map: 1.0        sheen: 0.2         sag: 0.4
#   gloss: 0.5
#
plakat paint from photo.png --medium acrylic --plan acrylic.hjson -o out-acrylic.png

# ── TEMPERA ──────────────────────────────────────────────────────────────────
# Egg tempera on a gesso panel (7.6.0): the paint dry in seconds, so nothing beneath is stirred; a wash
# block-in that covers the gesso, every restating pass hatched in straight lines of one width turned off the
# form by steps (along it, +15°, −30°, +45°), long on the flat and short on the busy, parallel to the plain
# contours, the hatching growing with the ladder and its charge solid in the darks; the warm flesh over a
# verdaccio underpainting that shows through the half-tones; the tones in Cennini's dishes, dithered between
# the steps; the lights hatched in toward white and the pure lights set last, dry and raised; the calm
# passages (a sky, a wall) built up in soft veils, not hatched; no impasto.
# Best painted from scratch with a PLAN — the face and figure tiers and every layer dial a plan key, tuned
# there rather than in the source. The recommended plan:
#
#   medium: tempera            palette: image           style: legible
#   new: true                  armature: 768            armature_face: 768      armature_body: 384
#   semantic: true             sam: true                families: true          commit_shadows: 1.0
#   shadow_floor: 0.16         detail_len: 0.5          infill: follow          rigger: 0.35      hotspot: 0.2
#   dry: 1.0                   wash_in: 0.15            wet_cap: 0.3            deplete: 1.2
#   hatch: 0.8                 verdaccio: 0.6           dishes: 5               lights: 0.5
#   cross: 15                  dither: 0.7              solid: 0.7              flesh: 0.8
#   span: 0.6                  edge: 0.7                glint: 0.6              smooth: 0.9
#   bole: 1.0                  ramp: 0.7                umbra: 1.0
#   turbulence: 4              loose: 0.1               shade: 0.1              vary: 0.2         body_vary: 0.1
#   broken: 0.05               glaze: 0.3               glaze_pass: 0.0         film: 0.6         lost: 0.3
#   aerial: 0.3                twist: 0.3               hiding: 0.0
#   impasto: 0.05              ridges: 0.0              impasto_map: 0.0        sheen: 0.05
#
plakat paint from photo.png --medium tempera --plan tempera.hjson -o out-tempera.png
```

---

## Methods — the `paint` sub‑commands

### 1. `paint from <IMAGE>` — repaint an image (the core, GPU‑free)
Turn a photo or a render into a painting in stroke space. See the media section above; the general shape is:

```bash
plakat paint from photo.png \
    --medium oil-direct \      # technique (optional — omit for a neutral repaint)
    --palette image \          # derive colours from the photo (or a named palette)
    --budget 4000 \            # more strokes = more detail — but past a point flat areas over-work into a hatch
    --style fidelity \         # legible | impressionist | fidelity (use LEGIBLE for wet media — see note below)
    --define 0.6 \             # edge hardness (legible/fidelity only)
    --saliency 0.85 \          # opt-in: keep the background thin, reserve density for the subject
    --seed 42 \                # deterministic; change for a variation
    --report \                 # print correlation-to-reference (high = filter, low = painting)
    -o out.png                 # also writes out.strokes (the replayable score)
```

**Register vs. medium.** `--style fidelity` suits oil and other opaque, legible subjects — tight, crisp,
detail-every-pass. For **transparent wet media (watercolour, ink-wash)** use `--style legible`: fidelity's short
non-overlapping marks leave gaps that stay near-white with transparent paint (a "lack of strokes" speckle),
whereas legible lays overlapping, mass-filling washes. `--reserve` (surface-white media) sets the paper-white
cutoff — raise it toward 1 to close stray white holes in light passages, lower it to keep more paper.

### 2. `paint <SPEC>` — paint from a PaintSpec (HJSON)
Drive everything from a file so a look is reusable and version‑controlled. This path also carries the GPU
refinements:

```bash
plakat paint portrait.paint.hjson -o out.png     # paint the spec
plakat paint portrait.paint.hjson --strokes 6000 --size 1024x1024 -o out.png   # override budget/size

# GPU refinements (spec path only):
plakat paint portrait.paint.hjson \
    --crisp \        # (GPU) matte the subject (U2Net), terminate strokes at its silhouette → crisp edge
    --critic \       # (GPU) score each stage pass, roll back any pass that worsens the painting
    --families \     # (GPU) split light/shadow families so the masses read solid
    -o out.png
```

A minimal spec that repaints a reference (`reference:`) — every field maps to a flag above:

```hjson
# portrait.paint.hjson
reference: photo.png
medium: oil-direct
palette: image          // derive from the reference
size: 768x768
budget: { strokes: 4000 }
seed: 42
style: fidelity
impasto: 0.22           // subtle relief — no canvas-weave on a portrait
sheen: 0.05
```

### 3. `paint new` — scaffold a spec
```bash
plakat paint new                       # writes painting.paint.hjson
plakat paint new mypiece.paint.hjson   # named
```

### 4. `paint show` / `paint lint` — inspect & validate a spec
```bash
plakat paint show portrait.paint.hjson   # compiled plan: stages, brush radii, per-stage budget
plakat paint lint portrait.paint.hjson   # check medium executable, palette known, reference present
```

### 5. `paint replay <SCORE>` — re‑render a stroke score at any size (GPU‑free)
The `.strokes` file next to every output is a **replayable score** — resolution‑independent.

```bash
plakat paint replay out.strokes -o big.png --size 2048x2048   # render at a new resolution
plakat paint replay out.strokes --until 800 -o partial.png    # stop after stroke 800 (an in-progress state)
```

### 6. `paint export <SCORE> <TARGET>` — derived products
```bash
plakat paint export out.strokes separations -o seps/    # one PNG per stage (block-in, mid, detail, …)
plakat paint export out.strokes stages -o stages/       # cumulative image after each stage
plakat paint export out.strokes height -o height.png    # 16-bit impasto height map
```

### 7. `paint timelapse <SCORE>` — stroke‑by‑stroke frames
```bash
plakat paint timelapse out.strokes -o frames/ --every 25   # a PNG every 25 strokes (GIF if few enough)
```

### 8. `paint palette [NAME]` — inspect the built‑in palettes
```bash
plakat paint palette                 # list all palettes
plakat paint palette zorn            # show the pigments in the Zorn palette
```

---

## Global reference

| Palette             | Pigments                                               | Suits |
| ------------------- | ------------------------------------------------------ | ----- |
| `zorn`              | ochre · cad‑red · black · white                        | oil portraits, warm skin |
| `split-primary`     | warm+cool of each primary                              | gouache, pastel, acrylic — full gamut |
| `verdaccio`         | ochre · black · terre‑verte · white                    | tempera underpainting |
| `earth`             | raw umber · burnt sienna · ochre · black · white       | landscapes, muted |
| `limited-landscape` | ultramarine · burnt sienna · ochre · cad‑yellow · white| watercolour / plein‑air |
| `sumi`              | black · white                                          | ink‑wash, pen‑ink, pencil, charcoal |
| `image` / `auto`    | *derived from the reference*                           | photos & portraits (any medium) |

**Styles:** `legible` (default — resolves features, draws hard edges) · `impressionist` (loose masses) ·
`fidelity` (tightest, sharp reference, every pass adds detail).

**Media (16):** `oil-direct` · `oil-indirect` · `gouache` · `watercolour` · `line-and-wash` · `early-book-illustration` · `ink-wash` · `japanese-ink` · `pen-ink` · `durer` · `tempera` ·
`pencil` · `black-pencil` · `pastel` · `charcoal` · `acrylic`.
