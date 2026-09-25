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
# Soft, chalky, high-chroma, blendable, matte. Default palette: split-primary.
plakat paint from photo.png --medium pastel --palette image \
    --chroma 1.15 --broken 0.4 --granulate 0.25 -o out-pastel.png

# ── CHARCOAL ─────────────────────────────────────────────────────────────────
# Dramatic soft black, smudgy, grainy tooth, contoured. Palette: sumi.
plakat paint from photo.png --medium charcoal --palette sumi \
    --contour 0.3 --granulate 0.35 -o out-charcoal.png

# ── ACRYLIC ──────────────────────────────────────────────────────────────────
# Opaque plastic colour, fast-dry (clean stacked layers), plastic sheen, darkens on drying.
plakat paint from photo.png --medium acrylic --palette image \
    --sheen 0.25 --impasto 0.4 --dry-shift -0.03 -o out-acrylic.png

# ── TEMPERA ──────────────────────────────────────────────────────────────────
# Fine cross-hatched build-up (egg tempera), low bleed, subtle body. Palette: verdaccio.
plakat paint from photo.png --medium tempera --palette verdaccio \
    --stroke-width 0.8 -o out-tempera.png
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
